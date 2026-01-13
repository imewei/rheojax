"""
SGR Monte Carlo Simulator (Lagrangian Approach).

This module provides a vectorized JAX implementation of the Soft Glassy Rheology
(SGR) model using a Trap Monte Carlo method. Each mesoscopic element is tracked
individually with its trap depth E and local strain ell.

Best for:
- Complex flow histories (startup, step rate, etc.)
- Stochastic noise analysis
- LAOS time-domain stress prediction

References:
    - P. Sollich, Physical Review E, 1998, 58(1), 738-759
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

logger = get_logger(__name__)

if TYPE_CHECKING:
    from jax import Array


# ============================================================================
# State Container
# ============================================================================


class SGRMCState(NamedTuple):
    """State container for SGR Monte Carlo simulation.

    Attributes:
        E: Trap depths for N particles, shape (N,)
        ell: Local strains for N particles, shape (N,)
        time: Current simulation time (scalar)
    """

    E: Array
    ell: Array
    time: float


# ============================================================================
# Initialization
# ============================================================================


def initialize_equilibrium(
    key: Array,
    n_particles: int,
    x: float,
    xg: float = 1.0,
) -> SGRMCState:
    """Initialize ensemble at equilibrium (ell=0, E sampled from prior).

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility
    n_particles : int
        Number of mesoscopic elements
    x : float
        Effective noise temperature
    xg : float, optional
        Glass transition temperature (energy scale), default 1.0

    Returns
    -------
    SGRMCState
        Initial state with E sampled from exp(-E/xg) and ell=0
    """
    logger.info(
        "Initializing SGR MC equilibrium state",
        n_particles=n_particles,
        x=x,
        xg=xg,
    )
    # Sample trap depths from exponential distribution: E = -xg * ln(u)
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(n_particles,), minval=1e-12, maxval=1.0)
    E = -xg * jnp.log(u)

    # Initial strain is zero (equilibrium)
    ell = jnp.zeros(n_particles, dtype=jnp.float64)

    logger.debug(
        "Equilibrium state initialized",
        E_mean=float(jnp.mean(E)),
        E_std=float(jnp.std(E)),
    )
    return SGRMCState(E=E, ell=ell, time=0.0)


# ============================================================================
# Stepping Functions
# ============================================================================


@jax.jit
def step_mc(
    key: Array,
    state: SGRMCState,
    gamma_dot: float,
    dt: float,
    x: float,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[SGRMCState, float]:
    """Advance Monte Carlo simulation by one time step.

    Implements robust Poisson-based yielding with exponential survival probability.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key (will be split internally)
    state : SGRMCState
        Current state {E, ell, time}
    gamma_dot : float
        Applied shear rate (1/s)
    dt : float
        Time step (s)
    x : float
        Effective noise temperature
    k : float, optional
        Elastic spring constant, default 1.0
    Gamma0 : float, optional
        Attempt rate (1/s), default 1.0
    xg : float, optional
        Glass transition temperature, default 1.0

    Returns
    -------
    tuple[SGRMCState, float]
        (new_state, stress) where stress is k * mean(ell)

    Notes
    -----
    Algorithm:
    1. Affine deformation: ell += gamma_dot * dt
    2. Yield probability: P_yield = 1 - exp(-Gamma * dt)
    3. Monte Carlo: if u < P_yield, yield and renew
    4. Stress: sigma = k * mean(ell)
    """
    logger.debug(
        "MC step starting",
        time=state.time,
        gamma_dot=gamma_dot,
        dt=dt,
        x=x,
    )
    E, ell, time = state.E, state.ell, state.time
    n = E.shape[0]

    # --- 1. Affine Loading ---
    ell_new = ell + gamma_dot * dt
    logger.debug("Affine loading applied", ell_mean=float(jnp.mean(ell_new)))

    # --- 2. Yield Rates & Survival Probability ---
    # Energy barrier: E - (1/2) * k * ell^2
    barrier = E - 0.5 * k * ell_new**2
    barrier_safe = jnp.maximum(barrier, 0.0)  # Barrier cannot be negative

    # Yielding rate
    Gamma = Gamma0 * jnp.exp(-barrier_safe / x)

    # Survival probability (exponential form for numerical stability)
    P_surv = jnp.exp(-Gamma * dt)
    logger.debug(
        "Yield rates computed",
        Gamma_mean=float(jnp.mean(Gamma)),
        P_surv_mean=float(jnp.mean(P_surv)),
    )

    # --- 3. Monte Carlo Yielding ---
    key, subkey1, subkey2 = jax.random.split(key, 3)
    r_draw = jax.random.uniform(subkey1, shape=(n,))

    # Yield mask: True if particle yields
    yield_mask = r_draw > P_surv
    n_yielded = int(jnp.sum(yield_mask))
    logger.debug(
        "Yielding evaluated", n_yielded=n_yielded, yield_fraction=n_yielded / n
    )

    # --- 4. Update Yielded Particles ---
    # Reset strain to 0 for yielded particles
    ell_updated = jnp.where(yield_mask, 0.0, ell_new)

    # Sample new trap depths for yielded particles
    u_rand = jax.random.uniform(subkey2, shape=(n,), minval=1e-12, maxval=1.0)
    E_new = -xg * jnp.log(u_rand)
    E_updated = jnp.where(yield_mask, E_new, E)

    # --- 5. Compute Stress ---
    sigma = k * jnp.mean(ell_updated)

    # Update time
    new_time = time + dt

    new_state = SGRMCState(E=E_updated, ell=ell_updated, time=new_time)
    logger.debug("MC step complete", new_time=new_time, sigma=float(sigma))
    return new_state, sigma


# ============================================================================
# High-Level Simulation Functions
# ============================================================================


def simulate_constant_rate(
    key: Array,
    gamma_dot: float,
    t_total: float,
    dt: float,
    x: float,
    n_particles: int = 10000,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[Array, Array]:
    """Simulate steady shear at constant rate.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    gamma_dot : float
        Shear rate (1/s)
    t_total : float
        Total simulation time (s)
    dt : float
        Time step (s)
    x : float
        Effective noise temperature
    n_particles : int, optional
        Number of particles, default 10000
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[Array, Array]
        (time_array, stress_array)
    """
    n_steps = int(t_total / dt)
    logger.info(
        "Starting constant rate simulation",
        gamma_dot=gamma_dot,
        t_total=t_total,
        dt=dt,
        n_steps=n_steps,
        n_particles=n_particles,
        x=x,
    )

    # Initialize
    key, init_key = jax.random.split(key)
    state = initialize_equilibrium(init_key, n_particles, x, xg)

    # Storage
    times = jnp.zeros(n_steps)
    stresses = jnp.zeros(n_steps)

    # Time-stepping loop (use lax.scan for efficiency)
    def scan_fn(carry, _):
        key, state = carry
        key, step_key = jax.random.split(key)
        new_state, sigma = step_mc(step_key, state, gamma_dot, dt, x, k, Gamma0, xg)
        return (key, new_state), (new_state.time, sigma)

    logger.debug("Running lax.scan for time-stepping")
    _, (times, stresses) = jax.lax.scan(scan_fn, (key, state), None, length=n_steps)

    logger.info(
        "Constant rate simulation complete",
        final_time=float(times[-1]),
        final_stress=float(stresses[-1]),
        stress_mean=float(jnp.mean(stresses)),
    )
    return times, stresses


def simulate_step_strain(
    key: Array,
    gamma_0: float,
    t_total: float,
    dt: float,
    x: float,
    n_particles: int = 10000,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[Array, Array]:
    """Simulate stress relaxation after step strain.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    gamma_0 : float
        Step strain amplitude
    t_total : float
        Total simulation time (s)
    dt : float
        Time step (s)
    x : float
        Effective noise temperature
    n_particles : int, optional
        Number of particles
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[Array, Array]
        (time_array, G_t_array) where G_t = stress / gamma_0
    """
    n_steps = int(t_total / dt)
    logger.info(
        "Starting step strain simulation",
        gamma_0=gamma_0,
        t_total=t_total,
        dt=dt,
        n_steps=n_steps,
        n_particles=n_particles,
        x=x,
    )

    # Initialize with step strain applied
    key, init_key = jax.random.split(key)
    state = initialize_equilibrium(init_key, n_particles, x, xg)

    # Apply step strain
    state = SGRMCState(E=state.E, ell=state.ell + gamma_0, time=0.0)
    logger.debug("Step strain applied", gamma_0=gamma_0)

    # Time-stepping with zero shear rate (relaxation)
    def scan_fn(carry, _):
        key, state = carry
        key, step_key = jax.random.split(key)
        new_state, sigma = step_mc(step_key, state, 0.0, dt, x, k, Gamma0, xg)
        return (key, new_state), (new_state.time, sigma)

    logger.debug("Running relaxation lax.scan")
    _, (times, stresses) = jax.lax.scan(scan_fn, (key, state), None, length=n_steps)

    # Relaxation modulus
    G_t = stresses / gamma_0

    logger.info(
        "Step strain simulation complete",
        final_time=float(times[-1]),
        G_t_initial=float(G_t[0]),
        G_t_final=float(G_t[-1]),
    )
    return times, G_t


def simulate_oscillatory(
    key: Array,
    gamma_0: float,
    omega: float,
    n_cycles: int,
    points_per_cycle: int,
    x: float,
    n_particles: int = 10000,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Simulate oscillatory shear (LAOS or SAOS).

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key
    gamma_0 : float
        Strain amplitude
    omega : float
        Angular frequency (rad/s)
    n_cycles : int
        Number of oscillation cycles
    points_per_cycle : int
        Time points per cycle
    x : float
        Effective noise temperature
    n_particles : int, optional
        Number of particles
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[Array, Array, Array]
        (time_array, strain_array, stress_array)
    """
    period = 2.0 * jnp.pi / omega
    dt = period / points_per_cycle
    n_steps = n_cycles * points_per_cycle
    logger.info(
        "Starting oscillatory simulation",
        gamma_0=gamma_0,
        omega=omega,
        n_cycles=n_cycles,
        points_per_cycle=points_per_cycle,
        n_steps=n_steps,
        n_particles=n_particles,
        x=x,
    )

    # Initialize
    key, init_key = jax.random.split(key)
    state = initialize_equilibrium(init_key, n_particles, x, xg)

    # Time-stepping with oscillatory rate
    def scan_fn(carry, step_idx):
        key, state = carry
        t = state.time
        # gamma(t) = gamma_0 * sin(omega * t)
        # gamma_dot = gamma_0 * omega * cos(omega * t)
        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
        gamma_t = gamma_0 * jnp.sin(omega * t)

        key, step_key = jax.random.split(key)
        new_state, sigma = step_mc(step_key, state, gamma_dot, dt, x, k, Gamma0, xg)
        return (key, new_state), (t, gamma_t, sigma)

    logger.debug("Running oscillatory lax.scan")
    _, (times, strains, stresses) = jax.lax.scan(
        scan_fn, (key, state), jnp.arange(n_steps)
    )

    logger.info(
        "Oscillatory simulation complete",
        final_time=float(times[-1]),
        stress_amplitude=float(jnp.max(jnp.abs(stresses))),
        n_cycles_completed=n_cycles,
    )
    return times, strains, stresses


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SGRMCState",
    "initialize_equilibrium",
    "step_mc",
    "simulate_constant_rate",
    "simulate_step_strain",
    "simulate_oscillatory",
]
