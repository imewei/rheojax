"""
SGR Population Balance Solver (Eulerian Approach).

This module provides an Eulerian PDE solver for the Soft Glassy Rheology (SGR)
model using operator splitting. The probability density P(E, ell) is discretized
on a 2D grid and evolved via advection, yielding, and renewal operators.

Best for:
- Smooth SAOS spectra without stochastic noise
- LAOS harmonics analysis (I3/I1)
- Gradient-based fitting via autodiff

References:
    - P. Sollich, Physical Review E, 1998, 58(1), 738-759
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from jax import Array


# ============================================================================
# Grid and State Container
# ============================================================================


class SGRPBGrid(NamedTuple):
    """Grid specification for Population Balance solver.

    Attributes:
        E_edges: Energy bin edges, shape (N_E + 1,)
        ell_edges: Strain bin edges, shape (N_ell + 1,)
        E_centers: Energy bin centers, shape (N_E,)
        ell_centers: Strain bin centers, shape (N_ell,)
        dE: Energy spacing
        dell: Strain spacing
    """

    E_edges: Array
    ell_edges: Array
    E_centers: Array
    ell_centers: Array
    dE: float
    dell: float


class SGRPBState(NamedTuple):
    """State container for Population Balance simulation.

    Attributes:
        P: Probability density matrix, shape (N_E, N_ell)
        time: Current simulation time (scalar)
    """

    P: Array
    time: float


# ============================================================================
# Grid Construction
# ============================================================================


def create_grid(
    E_max: float = 10.0,
    ell_max: float = 2.0,
    N_E: int = 64,
    N_ell: int = 128,
) -> SGRPBGrid:
    """Create discretization grid for P(E, ell).

    Parameters
    ----------
    E_max : float
        Maximum trap energy
    ell_max : float
        Maximum absolute strain (grid is symmetric: [-ell_max, ell_max])
    N_E : int
        Number of energy bins
    N_ell : int
        Number of strain bins

    Returns
    -------
    SGRPBGrid
        Grid specification
    """
    E_edges = jnp.linspace(0.0, E_max, N_E + 1)
    ell_edges = jnp.linspace(-ell_max, ell_max, N_ell + 1)

    E_centers = 0.5 * (E_edges[:-1] + E_edges[1:])
    ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

    dE = E_edges[1] - E_edges[0]
    dell = ell_edges[1] - ell_edges[0]

    return SGRPBGrid(
        E_edges=E_edges,
        ell_edges=ell_edges,
        E_centers=E_centers,
        ell_centers=ell_centers,
        dE=dE,
        dell=dell,
    )


# ============================================================================
# Initialization
# ============================================================================


def initialize_equilibrium(grid: SGRPBGrid, xg: float = 1.0) -> SGRPBState:
    """Initialize P(E, ell) at equilibrium (ell=0, E from prior).

    The equilibrium distribution has all mass at ell=0 with E distributed
    according to rho(E) = exp(-E/xg).

    Parameters
    ----------
    grid : SGRPBGrid
        Discretization grid
    xg : float
        Glass transition temperature (energy scale)

    Returns
    -------
    SGRPBState
        Initial state with P concentrated at ell=0
    """
    N_E = len(grid.E_centers)
    N_ell = len(grid.ell_centers)

    # Trap distribution rho(E) = (1/xg) * exp(-E/xg)
    rho_E = (1.0 / xg) * jnp.exp(-grid.E_centers / xg)
    rho_E = rho_E / jnp.sum(rho_E * grid.dE)  # Normalize

    # Find index closest to ell=0
    ell_zero_idx = jnp.argmin(jnp.abs(grid.ell_centers))

    # Initialize P: all mass at ell=0
    P = jnp.zeros((N_E, N_ell), dtype=jnp.float64)
    P = P.at[:, ell_zero_idx].set(rho_E / grid.dell)

    return SGRPBState(P=P, time=0.0)


# ============================================================================
# Operators
# ============================================================================


@partial(jax.jit, donate_argnums=(0,))
def advection_operator(
    P: Array,
    gamma_dot: float,
    dt: float,
    dell: float,
) -> Array:
    """Advection step: ell -> ell + gamma_dot * dt.

    Uses adaptive sub-stepping to satisfy CFL condition for stability.

    Parameters
    ----------
    P : Array
        Probability density, shape (N_E, N_ell)
    gamma_dot : float
        Shear rate
    dt : float
        Time step
    dell : float
        Strain grid spacing

    Returns
    -------
    Array
        Updated P after advection
    """
    # Calculate Courant number for full step
    c_total = gamma_dot * dt / dell

    # Determine number of sub-steps needed (safety factor 0.9)
    # n_substeps >= |c_total| / 0.9
    n_substeps = jnp.ceil(jnp.abs(c_total) / 0.9).astype(int)
    n_substeps = jnp.maximum(n_substeps, 1)

    # Sub-step parameters
    # c_sub will be < 0.9 in magnitude
    c_sub = c_total / n_substeps

    def body_fn(_, P_curr):
        # Upwind scheme (handles both positive and negative gamma_dot)
        # Using c_sub directly (signed)

        # If gamma_dot > 0 (c_sub > 0): forward difference
        # P_new[j] = P[j] - c*(P[j] - P[j-1])
        fwd = P_curr - c_sub * (P_curr - jnp.roll(P_curr, 1, axis=1))

        # If gamma_dot < 0 (c_sub < 0): backward difference
        # P_new[j] = P[j] - c*(P[j+1] - P[j])
        bwd = P_curr - c_sub * (jnp.roll(P_curr, -1, axis=1) - P_curr)

        P_next = jnp.where(gamma_dot >= 0, fwd, bwd)

        # Zero boundary conditions
        P_next = P_next.at[:, 0].set(0.0)
        P_next = P_next.at[:, -1].set(0.0)

        return P_next

    # Run sub-steps
    P_final = jax.lax.fori_loop(0, n_substeps, body_fn, P)

    return P_final


@partial(jax.jit, donate_argnums=(0,))
def yield_operator(
    P: Array,
    grid: SGRPBGrid,
    dt: float,
    x: float,
    k: float = 1.0,
    Gamma0: float = 1.0,
) -> tuple[Array, float]:
    """Yielding step: remove mass from traps according to yield rate.

    Parameters
    ----------
    P : Array
        Probability density, shape (N_E, N_ell)
    grid : SGRPBGrid
        Discretization grid
    dt : float
        Time step
    x : float
        Effective noise temperature
    k : float
        Elastic spring constant
    Gamma0 : float
        Attempt rate

    Returns
    -------
    tuple[Array, float]
        (P_after_yield, total_yield_mass)
    """
    # Compute yield rate on grid
    E_2d, ell_2d = jnp.meshgrid(grid.E_centers, grid.ell_centers, indexing="ij")

    # Barrier height: E - (1/2)*k*ell^2
    barrier = E_2d - 0.5 * k * ell_2d**2
    barrier_safe = jnp.maximum(barrier, 0.0)

    # Yield rate
    Gamma = Gamma0 * jnp.exp(-barrier_safe / x)

    # Fraction yielding in dt
    yield_fraction = 1.0 - jnp.exp(-Gamma * dt)

    # Mass removed
    delta_P_yield = P * yield_fraction

    # Total yielded mass
    M_yield = jnp.sum(delta_P_yield) * grid.dE * grid.dell

    # Update P
    P_after_yield = P - delta_P_yield

    return P_after_yield, M_yield


@partial(jax.jit, donate_argnums=(0,))
def renewal_operator(
    P: Array,
    grid: SGRPBGrid,
    M_yield: float,
    xg: float = 1.0,
) -> Array:
    """Renewal step: inject yielded mass at ell=0 with trap prior rho(E).

    Parameters
    ----------
    P : Array
        Probability density after yielding, shape (N_E, N_ell)
    grid : SGRPBGrid
        Discretization grid
    M_yield : float
        Total mass that yielded
    xg : float
        Glass transition temperature

    Returns
    -------
    Array
        P after renewal
    """
    # Trap distribution rho(E)
    rho_E = (1.0 / xg) * jnp.exp(-grid.E_centers / xg)
    rho_E = rho_E / jnp.sum(rho_E * grid.dE)  # Normalize

    # Find ell=0 bin
    ell_zero_idx = jnp.argmin(jnp.abs(grid.ell_centers))

    # Add mass at ell=0 according to rho(E)
    P_new = P.at[:, ell_zero_idx].add(M_yield * rho_E / grid.dell)

    return P_new


# ============================================================================
# Full Step
# ============================================================================


@jax.jit
def step_pb(
    state: SGRPBState,
    grid: SGRPBGrid,
    gamma_dot: float,
    dt: float,
    x: float,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[SGRPBState, float]:
    """Advance Population Balance by one time step (operator splitting).

    Parameters
    ----------
    state : SGRPBState
        Current state
    grid : SGRPBGrid
        Discretization grid
    gamma_dot : float
        Shear rate
    dt : float
        Time step
    x : float
        Effective noise temperature
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[SGRPBState, float]
        (new_state, stress)
    """
    P, time = state.P, state.time

    # Step 1: Advection
    P1 = advection_operator(P, gamma_dot, dt, grid.dell)

    # Step 2: Yielding
    P2, M_yield = yield_operator(P1, grid, dt, x, k, Gamma0)

    # Step 3: Renewal
    P3 = renewal_operator(P2, grid, M_yield, xg)

    # Compute stress: sigma = k * integral(ell * P(E, ell) dE dell)
    ell_2d = jnp.meshgrid(grid.E_centers, grid.ell_centers, indexing="ij")[1]
    sigma = k * jnp.sum(ell_2d * P3) * grid.dE * grid.dell

    new_state = SGRPBState(P=P3, time=time + dt)
    return new_state, sigma


# ============================================================================
# High-Level Simulation Functions
# ============================================================================


def simulate_constant_rate(
    gamma_dot: float,
    t_total: float,
    dt: float,
    x: float,
    grid: SGRPBGrid | None = None,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[Array, Array]:
    """Simulate steady shear at constant rate using Population Balance.

    Parameters
    ----------
    gamma_dot : float
        Shear rate
    t_total : float
        Total simulation time
    dt : float
        Time step
    x : float
        Effective noise temperature
    grid : SGRPBGrid, optional
        Discretization grid (default: auto-generated)
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[Array, Array]
        (time_array, stress_array)
    """
    if grid is None:
        grid = create_grid()

    n_steps = int(t_total / dt)
    state = initialize_equilibrium(grid, xg)

    def scan_fn(state, _):
        new_state, sigma = step_pb(state, grid, gamma_dot, dt, x, k, Gamma0, xg)
        return new_state, (new_state.time, sigma)

    _, (times, stresses) = jax.lax.scan(scan_fn, state, None, length=n_steps)

    return times, stresses


def simulate_oscillatory(
    gamma_0: float,
    omega: float,
    n_cycles: int,
    points_per_cycle: int,
    x: float,
    grid: SGRPBGrid | None = None,
    k: float = 1.0,
    Gamma0: float = 1.0,
    xg: float = 1.0,
) -> tuple[Array, Array, Array]:
    """Simulate oscillatory shear using Population Balance.

    Parameters
    ----------
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
    grid : SGRPBGrid, optional
        Discretization grid
    k, Gamma0, xg : float
        Model parameters

    Returns
    -------
    tuple[Array, Array, Array]
        (time_array, strain_array, stress_array)
    """
    if grid is None:
        grid = create_grid()

    period = 2.0 * jnp.pi / omega
    dt = period / points_per_cycle
    n_steps = n_cycles * points_per_cycle

    state = initialize_equilibrium(grid, xg)

    def scan_fn(state, step_idx):
        t = state.time
        gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
        gamma_t = gamma_0 * jnp.sin(omega * t)

        new_state, sigma = step_pb(state, grid, gamma_dot, dt, x, k, Gamma0, xg)
        return new_state, (t, gamma_t, sigma)

    _, (times, strains, stresses) = jax.lax.scan(scan_fn, state, jnp.arange(n_steps))

    return times, strains, stresses


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "SGRPBGrid",
    "SGRPBState",
    "create_grid",
    "initialize_equilibrium",
    "advection_operator",
    "yield_operator",
    "renewal_operator",
    "step_pb",
    "simulate_constant_rate",
    "simulate_oscillatory",
]
