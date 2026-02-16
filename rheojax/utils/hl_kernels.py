"""JAX-accelerated physics kernels for Hébraud–Lequeux (HL) model.

This module implements the core physics equations and time-evolution solvers
for the HL model using JAX. It uses a Finite Volume discretization of the
Fokker-Planck equation governing the probability density of local stresses.

Key functions:
- step_hl: Single time step evolution (advection + diffusion + yielding + renewal)
- run_flow_curve: Steady-state solver for flow curves
- run_startup: Transient solver for startup flow
- run_relaxation: Solver for stress relaxation
- run_creep: Solver for creep (stress control via servo)
- run_laos: Solver for oscillatory shear (SAOS/LAOS)

References:
    - P. Hébraud and F. Lequeux, Phys. Rev. Lett. 81, 2934 (1998)
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from rheojax.core.jax_config import safe_import_jax

# Safe import ensures float64
jax, jnp = safe_import_jax()
from jax import jit, lax  # noqa: E402

if TYPE_CHECKING:
    from jax import Array

# ============================================================================
# Types and Constants
# ============================================================================


class HLGrid(NamedTuple):
    """Grid specification for HL solver.

    Attributes:
        sigma: Stress grid points (centers)
        ds: Grid spacing (d_sigma)
        n_bins: Number of bins
    """

    sigma: Array
    ds: float
    n_bins: int


class HLState(NamedTuple):
    """State container for HL simulation.

    Attributes:
        P: Probability density array, shape (n_bins,)
        stress: Macroscopic stress (scalar)
        activity: Gamma (yield rate)
        diffusion: D (diffusion coefficient)
        time: Current simulation time
    """

    P: Array
    stress: float
    activity: float
    diffusion: float
    time: float


# ============================================================================
# Grid Construction
# ============================================================================


def make_grid(sigma_max: float = 5.0, n_bins: int = 501) -> HLGrid:
    """Create discretization grid for stress P(sigma).

    Args:
        sigma_max: Maximum stress (grid extends from -sigma_max to sigma_max)
        n_bins: Number of bins (should be odd to have a center bin at 0)

    Returns:
        HLGrid named tuple
    """
    sigma = jnp.linspace(-sigma_max, sigma_max, n_bins)
    ds = sigma[1] - sigma[0]
    return HLGrid(sigma, ds, n_bins)


# ============================================================================
# Core Physics Kernel
# ============================================================================


def _physics_step(
    state: HLState,
    gdot: float,
    grid: HLGrid,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
) -> HLState:
    """Single explicit Euler step for HL model physics."""
    P, _, _, _, t = state
    sigma, ds, n_bins = grid

    # 1. Calculate Activity (Gamma)
    # Mass in yield regions (|sigma| > sigma_c)
    mask_yield = jnp.abs(sigma) > sigma_c
    yield_pop = jnp.sum(P * mask_yield) * ds

    # Yield Rate: Gamma = (1/tau) * yield_pop
    Gamma = yield_pop / tau

    # 2. Calculate Diffusion (Closure)
    D = alpha * Gamma

    # 3. Evolution Operator (Operator Splitting)

    # A. Advection (Upwind - drift due to shear)
    # Flux J = v*P = gdot*P
    # First order upwind scheme

    # Efficient upwind implementation using roll
    # If gdot > 0, flow is rightward, use P[i] - P[i-1]
    # If gdot < 0, flow is leftward, use P[i+1] - P[i]

    # Rightward flow (gdot > 0): -v * (P[i] - P[i-1])/dx
    grad_upwind_pos = (P - jnp.roll(P, 1)) / ds
    # Leftward flow (gdot < 0): -v * (P[i+1] - P[i])/dx
    grad_upwind_neg = (jnp.roll(P, -1) - P) / ds

    dP_adv = -gdot * jnp.where(gdot > 0, grad_upwind_pos, grad_upwind_neg)

    # B. Diffusion (Central Difference)
    # D * d2P/dsigma2
    dP_diff = D * (jnp.roll(P, 1) - 2.0 * P + jnp.roll(P, -1)) / (ds**2)

    # C. Yielding (Sink)
    # -(1/tau) * P  where |sigma| > sigma_c
    dP_yield = -(1.0 / tau) * mask_yield * P

    # D. Reinjection (Source at sigma=0)
    # Total mass lost in step C
    mass_lost_rate = jnp.sum(jnp.abs(dP_yield)) * ds

    dP_source = jnp.zeros_like(P)
    center_idx = n_bins // 2
    # Add source to center bin (approx delta function)
    dP_source = dP_source.at[center_idx].set(mass_lost_rate / ds)

    # Update P with Forward Euler
    P_new = P + dt * (dP_adv + dP_diff + dP_yield + dP_source)

    # Boundary condition: Zero at edges
    P_new = P_new.at[0].set(0.0).at[-1].set(0.0)

    # Enforce positivity
    P_new = jnp.maximum(P_new, 0.0)

    # Renormalize
    norm = jnp.sum(P_new) * ds
    P_new = P_new / (norm + 1e-12)

    # Calculate Observables
    stress_new = jnp.sum(P_new * sigma) * ds

    return HLState(P_new, stress_new, Gamma, D, t + dt)


@jit
def step_hl(
    state: HLState,
    gdot: float,
    grid: HLGrid,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
) -> HLState:
    """Advance HL model by one time step with adaptive sub-stepping.

    Implements the Fokker-Planck equation with operator splitting.
    Automatically calculates stable sub-steps based on CFL conditions
    for advection and diffusion.

    Args:
        state: Current HLState
        gdot: Shear rate (1/s)
        grid: Discretization grid
        alpha: Coupling parameter (<0.5 for glass)
        tau: Yield timescale (s)
        sigma_c: Yield stress threshold (Pa)
        dt: Time step (s)

    Returns:
        Updated HLState
    """
    P, _, _, _, _ = state
    sigma, ds, _ = grid

    # Calculate current parameters for CFL check
    mask_yield = jnp.abs(sigma) > sigma_c
    yield_pop = jnp.sum(P * mask_yield) * ds
    Gamma = yield_pop / tau
    D = alpha * Gamma

    # CFL Stability Conditions
    # 1. Advection: dt < dx / v
    v_adv = jnp.abs(gdot) + 1e-12
    dt_adv = ds / v_adv

    # 2. Diffusion: dt < dx^2 / 2D
    D_safe = D + 1e-12
    dt_diff = (ds**2) / (2.0 * D_safe)

    # Stable time step (safety factor 0.5)
    dt_stable = 0.5 * jnp.minimum(dt_adv, dt_diff)

    # Determine number of sub-steps
    n_sub = jnp.ceil(dt / dt_stable).astype(int)
    n_sub = jnp.maximum(n_sub, 1)

    dt_sub = dt / n_sub

    # Execute sub-steps
    def body_fun(i, current_state):
        return _physics_step(current_state, gdot, grid, alpha, tau, sigma_c, dt_sub)

    final_state = lax.fori_loop(0, n_sub, body_fun, state)

    return final_state


# ============================================================================
# Protocol Kernels (JIT Compiled)
# ============================================================================


@partial(jit, static_argnames=["steps", "n_bins"])
def flow_curve_kernel(
    gdot: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
    sigma_max: float,
    n_bins: int,
    steps: int,
) -> float:
    """Core flow curve simulation kernel."""
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid

    # Initialize Gaussian centered at 0
    P0 = jnp.exp(-(sigma**2) / 0.1)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_state = HLState(P0, 0.0, 0.0, 0.0, 0.0)

    def body(state, _):
        return step_hl(state, gdot, grid, alpha, tau, sigma_c, dt), None

    final_state, _ = lax.scan(body, init_state, None, length=steps)
    return final_state.stress


@partial(jit, static_argnames=["n_steps", "n_bins"])
def startup_kernel(
    n_steps: int,
    gdot: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
    sigma_max: float,
    n_bins: int,
) -> tuple[Array, Array]:
    """Core startup simulation kernel."""
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid

    # Initialize relaxed state (approximate)
    P0 = jnp.exp(-(sigma**2) / 0.05)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_state = HLState(P0, 0.0, 0.0, 0.0, 0.0)

    def body(state, _):
        new_state = step_hl(state, gdot, grid, alpha, tau, sigma_c, dt)
        return new_state, new_state.stress

    _, stress_history = lax.scan(body, init_state, None, length=n_steps)
    time_history = jnp.arange(n_steps) * dt + dt

    return time_history, stress_history


@partial(jit, static_argnames=["n_steps", "n_bins"])
def relaxation_kernel(
    n_steps: int,
    gamma0: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
    sigma_max: float,
    n_bins: int,
) -> tuple[Array, Array]:
    """Core relaxation simulation kernel."""
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid

    # Initial state: Relaxed P0 shifted by gamma0
    P0 = jnp.exp(-((sigma - gamma0) ** 2) / 0.05)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_stress = jnp.sum(P0 * sigma) * ds
    init_state = HLState(P0, init_stress, 0.0, 0.0, 0.0)

    def body(state, _):
        # gdot = 0 for relaxation
        new_state = step_hl(state, 0.0, grid, alpha, tau, sigma_c, dt)
        return new_state, new_state.stress

    _, stress_history = lax.scan(body, init_state, None, length=n_steps)
    time_history = jnp.arange(n_steps) * dt + dt

    return time_history, stress_history


@partial(jit, static_argnames=["n_steps", "n_bins"])
def creep_kernel(
    n_steps: int,
    stress_target: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    kp: float,
    dt: float,
    sigma_max: float,
    n_bins: int,
) -> tuple[Array, Array]:
    """Core creep simulation kernel."""
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid

    # Initialize relaxed
    P0 = jnp.exp(-(sigma**2) / 0.05)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_state = HLState(P0, 0.0, 0.0, 0.0, 0.0)

    # Carry: (state, gdot_current, gamma_total)
    init_carry = (init_state, 0.0, 0.0)

    def body(carry, _):
        state, gdot, gamma = carry

        # 1. Step Physics
        new_state = step_hl(state, gdot, grid, alpha, tau, sigma_c, dt)

        # 2. Update total strain
        new_gamma = gamma + gdot * dt

        # 3. Servo Controller
        error = stress_target - new_state.stress
        gdot_new = gdot + kp * error * dt

        return (new_state, gdot_new, new_gamma), new_gamma

    _, gamma_history = lax.scan(body, init_carry, None, length=n_steps)
    time_history = jnp.arange(n_steps) * dt + dt

    return time_history, gamma_history


@partial(jit, static_argnames=["n_steps", "n_bins"])
def laos_kernel(
    n_steps: int,
    gamma0: float,
    omega: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float,
    sigma_max: float,
    n_bins: int,
) -> tuple[Array, Array]:
    """Core LAOS simulation kernel."""
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid

    # Initialize relaxed
    P0 = jnp.exp(-(sigma**2) / 0.05)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_state = HLState(P0, 0.0, 0.0, 0.0, 0.0)

    def body(state, step_idx):
        t_curr = step_idx * dt
        gdot = gamma0 * omega * jnp.cos(omega * t_curr)
        new_state = step_hl(state, gdot, grid, alpha, tau, sigma_c, dt)
        return new_state, new_state.stress

    _, stress_history = lax.scan(body, init_state, jnp.arange(n_steps))
    time_history = jnp.arange(n_steps) * dt + dt

    return time_history, stress_history


# ============================================================================
# Public Protocol Runners (Non-Jitted Wrappers)
# ============================================================================


def _compute_dt_and_steps_for_rate(
    gdot_val: float,
    tau: float,
    sigma_c: float,
    ds: float = 0.02,
    max_steps: int = 30_000,
    bucket_size: int = 5_000,
) -> tuple[float, int]:
    """Compute adaptive (dt, steps) for a given shear rate.

    Uses larger dt for low shear rates (CFL allows it) so that all rates
    converge within max_steps outer steps. The CFL sub-stepping inside
    step_hl handles numerical stability automatically.

    Returns:
        (dt, steps) tuple where dt is the outer time step and steps is
        the number of outer lax.scan iterations.
    """
    gdot_abs = abs(gdot_val) if not hasattr(gdot_val, "item") else abs(float(gdot_val))
    gdot_abs = max(gdot_abs, 1e-9)

    # Target simulation time for steady state
    strain_target = 10.0 * max(1.0, sigma_c)
    time_strain = strain_target / gdot_abs
    time_relax = 10.0 * tau
    t_total = max(time_strain, time_relax)

    # Adaptive dt: scale up to fit t_total in max_steps.
    # Constraint: keep CFL sub-steps per outer step <= 50
    # so that fori_loop overhead stays manageable.
    dt_desired = t_total / max_steps
    dt_max_cfl = 50 * 0.5 * ds / (gdot_abs + 1e-12)

    dt = max(dt_desired, 0.001)  # minimum dt = 1ms
    dt = min(dt, dt_max_cfl)  # cap to avoid excessive sub-stepping
    dt = min(dt, 0.5)  # absolute ceiling

    steps = int(t_total / dt) + 1
    steps = min(steps, max_steps)
    steps = max(steps, 5_000)

    # Bucket to nearest bucket_size to minimize JIT recompilations
    steps = ((steps + bucket_size - 1) // bucket_size) * bucket_size
    steps = min(steps, max_steps)

    return dt, steps


def run_flow_curve(
    gdots: Array,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float = 0.005,
    sigma_max: float = 5.0,
    n_bins: int = 501,
    steps: int = 20000,
    per_rate_schedule: list[tuple[float, int]] | None = None,
) -> Array:
    """Calculate flow curve (steady state stress vs shear rate).

    Args:
        per_rate_schedule: If provided, a list of (dt_i, steps_i) tuples,
            one per shear rate. Uses adaptive dt so low rates (which need
            more physical time) use larger dt while high rates use smaller dt.
            Must be computed outside any JIT context (concrete values).
    """
    if per_rate_schedule is not None:
        # Per-rate adaptive schedule: sequential with per-rate (dt, steps).
        # Avoids OOM from vmapping all rates at max step count.
        results = []
        for i in range(len(gdots)):
            dt_i, steps_i = per_rate_schedule[i]
            stress_i = flow_curve_kernel(
                gdots[i], alpha, tau, sigma_c, dt_i, sigma_max, n_bins,
                steps_i,
            )
            results.append(stress_i)
        return jnp.array(results)
    else:
        # Uniform step count: use vmap for efficiency
        vectorized_kernel = jax.vmap(
            lambda g: flow_curve_kernel(
                g, alpha, tau, sigma_c, dt, sigma_max, n_bins, steps
            )
        )
        return vectorized_kernel(gdots)


def run_startup(
    t: Array,
    gdot: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float = 0.005,
    sigma_max: float = 5.0,
    n_bins: int = 501,
) -> Array:
    """Simulate startup shear flow."""
    # Ensure t is accessible and calculate steps
    t_max = float(t[-1]) if hasattr(t[-1], "item") else float(t[-1])
    n_steps = int(t_max / dt) + 1

    time_hist, stress_hist = startup_kernel(
        n_steps, gdot, alpha, tau, sigma_c, dt, sigma_max, n_bins
    )

    time_full = jnp.concatenate([jnp.array([0.0]), time_hist])
    stress_full = jnp.concatenate([jnp.array([0.0]), stress_hist])

    return jnp.interp(t, time_full, stress_full)


def run_relaxation(
    t: Array,
    gamma0: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float = 0.005,
    sigma_max: float = 5.0,
    n_bins: int = 501,
) -> Array:
    """Simulate stress relaxation after step strain."""
    t_max = float(t[-1]) if hasattr(t[-1], "item") else float(t[-1])
    n_steps = int(t_max / dt) + 1

    time_hist, stress_hist = relaxation_kernel(
        n_steps, gamma0, alpha, tau, sigma_c, dt, sigma_max, n_bins
    )

    # Initial stress for t=0
    # Duplicating init logic for correct interpolation at t=0
    grid = make_grid(sigma_max, n_bins)
    sigma, ds, _ = grid
    P0 = jnp.exp(-((sigma - gamma0) ** 2) / 0.05)
    P0 = P0 / (jnp.sum(P0) * ds)
    init_stress = jnp.sum(P0 * sigma) * ds

    time_full = jnp.concatenate([jnp.array([0.0]), time_hist])
    stress_full = jnp.concatenate([jnp.array([init_stress]), stress_hist])

    sigma_t = jnp.interp(t, time_full, stress_full)
    return sigma_t / gamma0


def run_creep(
    t: Array,
    stress_target: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    kp: float = 1.0,
    dt: float = 0.005,
    sigma_max: float = 5.0,
    n_bins: int = 501,
) -> Array:
    """Simulate creep."""
    t_max = float(t[-1]) if hasattr(t[-1], "item") else float(t[-1])
    n_steps = int(t_max / dt) + 1

    time_hist, gamma_hist = creep_kernel(
        n_steps, stress_target, alpha, tau, sigma_c, kp, dt, sigma_max, n_bins
    )

    time_full = jnp.concatenate([jnp.array([0.0]), time_hist])
    gamma_full = jnp.concatenate([jnp.array([0.0]), gamma_hist])

    gamma_t = jnp.interp(t, time_full, gamma_full)
    return gamma_t / stress_target


def run_laos(
    time: Array,
    gamma0: float,
    omega: float,
    alpha: float,
    tau: float,
    sigma_c: float,
    dt: float = 0.005,
    sigma_max: float = 5.0,
    n_bins: int = 501,
) -> Array:
    """Simulate LAOS."""
    t_max = float(time[-1]) if hasattr(time[-1], "item") else float(time[-1])
    n_steps = int(t_max / dt) + 1

    time_hist, stress_hist = laos_kernel(
        n_steps, gamma0, omega, alpha, tau, sigma_c, dt, sigma_max, n_bins
    )

    time_full = jnp.concatenate([jnp.array([0.0]), time_hist])
    stress_full = jnp.concatenate([jnp.array([0.0]), stress_hist])

    return jnp.interp(time, time_full, stress_full)
