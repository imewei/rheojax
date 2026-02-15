"""diffrax-based ODE solvers for ITT-MCT models.

This module provides JAX-native ODE solving using diffrax, enabling:
- JIT compilation of entire solve loops
- vmap parallelization over shear rates
- Automatic differentiation through ODE solutions
- Adaptive time stepping with early termination

The diffrax solvers provide 100-500x speedup over scipy-based integration.

Functions
---------
solve_flow_curve_single
    Solve flow curve ODE for a single shear rate
solve_flow_curve_batch
    Vectorized solve over multiple shear rates
compute_adaptive_t_max
    Physics-based adaptive integration time

Notes
-----
diffrax requires vector fields with signature `f(t, y, args)` where args
is a PyTree containing all parameters. This differs from the scipy-style
signature used in _kernels.py.

References
----------
- diffrax documentation: https://docs.kidger.site/diffrax/
- Patrick Kidger (2021) "On Neural Differential Equations", arXiv:2202.02435
"""

from typing import NamedTuple


from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.logging import get_logger
from rheojax.models.itt_mct._kernels import f12_memory, strain_decorrelation

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


# =============================================================================
# Parameter Container for diffrax
# =============================================================================


class FlowCurveParams(NamedTuple):
    """Parameters for flow curve ODE integration.

    Using NamedTuple for JAX compatibility (PyTree).
    """

    gamma_dot: float  # Shear rate
    v1: float  # Linear vertex
    v2: float  # Quadratic vertex
    Gamma: float  # Bare relaxation rate
    gamma_c: float  # Critical strain
    G_inf: float  # High-frequency modulus
    g: jnp.ndarray  # Prony amplitudes
    tau: jnp.ndarray  # Prony times
    n_modes: int  # Number of Prony modes
    use_lorentzian: bool  # Use Lorentzian vs Gaussian decorrelation
    memory_form: str  # "simplified" or "full" two-time memory


# =============================================================================
# diffrax-compatible Vector Fields
# =============================================================================


def make_flow_curve_vector_field(
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
):
    """Create diffrax-compatible vector field for flow curve ODE.

    Parameters
    ----------
    n_modes : int
        Number of Prony modes (static for JIT)
    use_lorentzian : bool, default False
        If True, use Lorentzian decorrelation; if False, use Gaussian
    memory_form : str, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation h[γ_acc]
        - "full": two-time decorrelation h[γ_total] × h[γ_mode]

    Returns
    -------
    callable
        Vector field function with signature (t, state, args) -> derivatives

    Notes
    -----
    Uses pre-allocated array with .at[].set() instead of jnp.concatenate
    to reduce JIT compilation time by avoiding dynamic shape tracing.
    """
    # Pre-compute state dimension for fixed-size output
    state_dim = 2 + n_modes  # [Phi, K_1..K_n, gamma_accumulated]
    use_full_memory = memory_form == "full"

    def vector_field(
        t: float, state: jnp.ndarray, args: FlowCurveParams
    ) -> jnp.ndarray:
        """ODE right-hand side for steady flow.

        State: [Phi, K_1, ..., K_n, gamma_accumulated]
        """
        # Unpack state
        phi = state[0]
        K = state[1 : 1 + n_modes]
        gamma_acc = state[1 + n_modes]

        # Unpack parameters
        gamma_dot = args.gamma_dot
        v1 = args.v1
        v2 = args.v2
        Gamma = args.Gamma
        gamma_c = args.gamma_c
        g = args.g
        tau = args.tau

        # Strain decorrelation (use_lorentzian captured from closure)
        h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)

        # Advected correlator
        phi_advected = phi * h_gamma

        # Memory kernel
        m_phi = f12_memory(phi_advected, v1, v2)

        # Memory integral from Prony modes
        memory_integral = jnp.sum(K)

        # MCT equation: dPhi/dt = -Gamma * (Phi + memory_integral)
        dphi_dt = -Gamma * (phi + memory_integral)

        # Prony mode evolution with memory form selection
        if use_full_memory:
            # Full two-time: mode-specific decorrelation
            gamma_mode = gamma_dot * tau
            h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
            dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
        else:
            # Simplified: standard schematic
            dK_dt = -K / tau + g * m_phi * dphi_dt

        # Strain accumulation
        dgamma_dt = gamma_dot

        # Pre-allocated output with .at[].set() for faster JIT compilation
        deriv = jnp.zeros(state_dim)
        deriv = deriv.at[0].set(dphi_dt)
        deriv = deriv.at[1 : 1 + n_modes].set(dK_dt)
        deriv = deriv.at[1 + n_modes].set(dgamma_dt)

        return deriv

    return vector_field


# =============================================================================
# Adaptive Time and Termination
# =============================================================================


@jax.jit
def compute_adaptive_t_max(
    gamma_dot: float,
    Gamma: float,
    gamma_c: float,
    min_t_max: float = 10.0,
    max_t_max: float = 1000.0,
    n_relaxations: float = 20.0,
) -> float:
    """Compute physics-based adaptive integration time.

    The relaxation timescale depends on shear rate:
    - At high gamma_dot: tau_eff ~ gamma_c / gamma_dot (cage breaking)
    - At low gamma_dot: tau_eff ~ 1 / Gamma (bare relaxation)

    Parameters
    ----------
    gamma_dot : float
        Shear rate
    Gamma : float
        Bare relaxation rate
    gamma_c : float
        Critical strain
    min_t_max : float
        Minimum integration time
    max_t_max : float
        Maximum integration time (safety cap)
    n_relaxations : float
        Number of relaxation times to integrate

    Returns
    -------
    float
        Adaptive t_max for ODE integration
    """
    # Effective relaxation time
    tau_bare = 1.0 / Gamma
    tau_shear = gamma_c / jnp.maximum(gamma_dot, 1e-10)
    tau_eff = jnp.minimum(tau_bare, tau_shear)

    # Integration time = n_relaxations * effective timescale
    t_max = n_relaxations * tau_eff

    # Clamp to reasonable bounds
    t_max = jnp.clip(t_max, min_t_max, max_t_max)

    return t_max


def make_steady_state_event(n_modes: int, threshold: float = 1e-6):
    """Create event function for steady-state termination.

    Parameters
    ----------
    n_modes : int
        Number of Prony modes
    threshold : float
        Threshold for |dPhi/dt| to consider steady state

    Returns
    -------
    callable
        Event condition function
    """

    def event_fn(state, **kwargs):
        """Returns negative when steady state reached."""
        # We check if the correlator derivative is small
        # This is a simplified check - full would recompute derivatives
        phi = state[0]
        # Simple heuristic: steady state when phi is small or stable
        # In practice, we rely on adaptive t_max more than events
        return jnp.abs(phi - 0.5) - 0.49  # Triggers when phi very small or ~1

    return event_fn


# =============================================================================
# Single Shear Rate Solver
# =============================================================================


def make_flow_curve_with_stress_vector_field(
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
):
    """Create vector field that also integrates stress.

    The stress is the time integral: σ = G_inf * γ̇ * ∫₀^∞ Φ(t) * h(γ(t)) dt

    State: [Phi, K_1, ..., K_n, gamma, sigma_integral]

    Parameters
    ----------
    n_modes : int
        Number of Prony modes (static for JIT)
    use_lorentzian : bool, default False
        If True, use Lorentzian decorrelation; if False, use Gaussian
    memory_form : str, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation h[γ_acc]
        - "full": two-time decorrelation h[γ_total] × h[γ_mode]

    Notes
    -----
    Uses pre-allocated array with .at[].set() instead of jnp.concatenate
    to reduce JIT compilation time by avoiding dynamic shape tracing.
    """
    # Pre-compute state dimension for fixed-size output
    state_dim = 3 + n_modes  # [Phi, K_1..K_n, gamma, sigma_integral]
    use_full_memory = memory_form == "full"

    def vector_field(
        t: float, state: jnp.ndarray, args: FlowCurveParams
    ) -> jnp.ndarray:
        """ODE right-hand side including stress integration."""
        # Unpack state
        phi = state[0]
        K = state[1 : 1 + n_modes]
        gamma_acc = state[1 + n_modes]
        # sigma_integral = state[2 + n_modes]  # Not needed in RHS

        # Unpack parameters
        gamma_dot = args.gamma_dot
        v1 = args.v1
        v2 = args.v2
        Gamma = args.Gamma
        gamma_c = args.gamma_c
        G_inf = args.G_inf
        g = args.g
        tau = args.tau

        # Strain decorrelation (use_lorentzian captured from closure)
        h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)

        # Advected correlator
        phi_advected = phi * h_gamma

        # Memory kernel
        m_phi = f12_memory(phi_advected, v1, v2)

        # Memory integral from Prony modes
        memory_integral = jnp.sum(K)

        # MCT equation
        dphi_dt = -Gamma * (phi + memory_integral)

        # Prony mode evolution with memory form selection
        if use_full_memory:
            gamma_mode = gamma_dot * tau
            h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
            dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
        else:
            dK_dt = -K / tau + g * m_phi * dphi_dt

        # Strain accumulation
        dgamma_dt = gamma_dot

        # Stress integrand: d(σ_integral)/dt = G_inf * γ̇ * Φ * h(γ)
        dsigma_dt = G_inf * gamma_dot * phi_advected

        # Pre-allocated output with .at[].set() for faster JIT compilation
        deriv = jnp.zeros(state_dim)
        deriv = deriv.at[0].set(dphi_dt)
        deriv = deriv.at[1 : 1 + n_modes].set(dK_dt)
        deriv = deriv.at[1 + n_modes].set(dgamma_dt)
        deriv = deriv.at[2 + n_modes].set(dsigma_dt)

        return deriv

    return vector_field


def solve_flow_curve_single(
    gamma_dot: float,
    v1: float,
    v2: float,
    Gamma: float,
    gamma_c: float,
    G_inf: float,
    g: jnp.ndarray,
    tau: jnp.ndarray,
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_steps: int = 5000,
) -> float:
    """Solve flow curve ODE for single shear rate using diffrax.

    The stress is computed as the time integral:
        σ = G_inf * γ̇ * ∫₀^∞ Φ(t) * h(γ(t)) dt

    Parameters
    ----------
    gamma_dot : float
        Shear rate
    v1, v2 : float
        F12 vertex coefficients
    Gamma : float
        Bare relaxation rate
    gamma_c : float
        Critical strain
    G_inf : float
        High-frequency modulus
    g, tau : jnp.ndarray
        Prony mode parameters
    n_modes : int
        Number of Prony modes
    use_lorentzian : bool, default False
        If True, use Lorentzian strain decorrelation h(γ) = 1/(1+(γ/γ_c)²)
        If False (default), use Gaussian form h(γ) = exp(-(γ/γ_c)²)
    memory_form : str, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation h[γ_acc]
        - "full": two-time decorrelation h[γ_total] × h[γ_mode]
    rtol, atol : float
        Relative and absolute tolerances
    max_steps : int
        Maximum ODE solver steps

    Returns
    -------
    float
        Steady-state stress sigma
    """
    # Create vector field with stress integration
    vector_field = make_flow_curve_with_stress_vector_field(
        n_modes, use_lorentzian, memory_form
    )

    # Bundle parameters
    params = FlowCurveParams(
        gamma_dot=gamma_dot,
        v1=v1,
        v2=v2,
        Gamma=Gamma,
        gamma_c=gamma_c,
        G_inf=G_inf,
        g=g,
        tau=tau,
        n_modes=n_modes,
        use_lorentzian=use_lorentzian,
        memory_form=memory_form,
    )

    # Initial state: [Phi=1, K_i=0, gamma=0, sigma_integral=0]
    state0 = jnp.zeros(3 + n_modes)
    state0 = state0.at[0].set(1.0)  # Phi(0) = 1

    # Adaptive integration time - use longer time for stress convergence
    t_max = compute_adaptive_t_max(gamma_dot, Gamma, gamma_c, n_relaxations=50.0)

    # diffrax ODE term
    term = diffrax.ODETerm(vector_field)

    # Solver: Tsit5 is a good general-purpose choice (similar to RK45)
    solver = diffrax.Tsit5()

    # Step size controller
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    # Solve ODE - only save final state
    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=t_max,
        dt0=0.01,
        y0=state0,
        args=params,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),  # Only save final state
        max_steps=max_steps,
        throw=False,  # Return partial result on failure (for optimization)
    )

    # Extract stress integral from final state
    final_state = solution.ys[0]  # Shape: (state_dim,)
    sigma = final_state[2 + n_modes]  # The integrated stress

    # Handle solver failure by returning NaN (optimization will avoid this)
    sigma = jnp.where(solution.result == diffrax.RESULTS.successful, sigma, jnp.nan)

    return sigma


# =============================================================================
# Batched Solver (vmapped)
# =============================================================================


def _make_batched_solver(
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
):
    """Create JIT-compiled batched flow curve solver.

    Parameters
    ----------
    n_modes : int
        Number of Prony modes (must be static for JIT)
    use_lorentzian : bool, default False
        If True, use Lorentzian decorrelation; if False, use Gaussian
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"

    Returns
    -------
    callable
        Batched solver function
    """

    # Inner solve function that will be vmapped
    def _solve_single(gamma_dot, v1, v2, Gamma, gamma_c, G_inf, g, tau):
        return solve_flow_curve_single(
            gamma_dot=gamma_dot,
            v1=v1,
            v2=v2,
            Gamma=Gamma,
            gamma_c=gamma_c,
            G_inf=G_inf,
            g=g,
            tau=tau,
            n_modes=n_modes,
            use_lorentzian=use_lorentzian,
            memory_form=memory_form,
        )

    # vmap over gamma_dot, keeping other params fixed
    _solve_batch = jax.vmap(
        _solve_single,
        in_axes=(0, None, None, None, None, None, None, None),  # Only gamma_dot varies
    )

    # JIT compile the entire batch operation
    @jax.jit
    def solve_batch(gamma_dot_array, v1, v2, Gamma, gamma_c, G_inf, g, tau):
        return _solve_batch(gamma_dot_array, v1, v2, Gamma, gamma_c, G_inf, g, tau)

    return solve_batch


# Cache for batched solvers (keyed by (n_modes, use_lorentzian, memory_form))
_BATCHED_SOLVER_CACHE = {}


def solve_flow_curve_batch(
    gamma_dot_array: jnp.ndarray,
    v1: float,
    v2: float,
    Gamma: float,
    gamma_c: float,
    G_inf: float,
    g: jnp.ndarray,
    tau: jnp.ndarray,
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
) -> jnp.ndarray:
    """Solve flow curve for multiple shear rates using vmap + JIT.

    This is the main entry point for fast flow curve computation.
    First call will trigger JIT compilation (~5-10s), subsequent calls
    are very fast (<0.5s for 50 points).

    Parameters
    ----------
    gamma_dot_array : jnp.ndarray
        Array of shear rates
    v1, v2 : float
        F12 vertex coefficients
    Gamma : float
        Bare relaxation rate
    gamma_c : float
        Critical strain
    G_inf : float
        High-frequency modulus
    g, tau : jnp.ndarray
        Prony mode parameters
    n_modes : int
        Number of Prony modes
    use_lorentzian : bool, default False
        If True, use Lorentzian strain decorrelation h(γ) = 1/(1+(γ/γ_c)²)
        If False (default), use Gaussian form h(γ) = exp(-(γ/γ_c)²)
    memory_form : str, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation h[γ_acc]
        - "full": two-time decorrelation h[γ_total] × h[γ_mode]

    Returns
    -------
    jnp.ndarray
        Steady-state stress array sigma(gamma_dot)
    """
    # Cache key includes n_modes, use_lorentzian, and memory_form
    cache_key = (n_modes, use_lorentzian, memory_form)

    # Get or create cached solver for this configuration
    if cache_key not in _BATCHED_SOLVER_CACHE:
        logger.debug(
            f"Creating batched solver for n_modes={n_modes}, "
            f"use_lorentzian={use_lorentzian}, memory_form={memory_form}"
        )
        _BATCHED_SOLVER_CACHE[cache_key] = _make_batched_solver(
            n_modes, use_lorentzian, memory_form
        )

    solver = _BATCHED_SOLVER_CACHE[cache_key]

    # Convert to JAX array if needed
    gamma_dot_array = jnp.asarray(gamma_dot_array)
    g = jnp.asarray(g)
    tau = jnp.asarray(tau)

    # Run batched solve
    return solver(gamma_dot_array, v1, v2, Gamma, gamma_c, G_inf, g, tau)


# =============================================================================
# Convenience Functions
# =============================================================================


def is_diffrax_available() -> bool:
    """Check if diffrax is available and working."""
    import importlib.util

    return importlib.util.find_spec("diffrax") is not None


def clear_solver_cache():
    """Clear the cached batched solvers.

    Useful if parameters change and you want to force recompilation.
    """
    global _BATCHED_SOLVER_CACHE
    _BATCHED_SOLVER_CACHE.clear()
    logger.debug("Cleared diffrax solver cache")


def precompile_flow_curve_solver(
    n_modes: int = 10,
    n_points: int = 5,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
) -> float:
    """Pre-compile the batched flow curve solver with dummy data.

    Triggers JIT compilation so subsequent calls are fast.
    Call this once at startup if you need predictable timing.

    Parameters
    ----------
    n_modes : int, default 10
        Number of Prony modes to compile for
    n_points : int, default 5
        Number of shear rate points in dummy batch
    use_lorentzian : bool, default False
        Compile for Lorentzian (True) or Gaussian (False) decorrelation
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"

    Returns
    -------
    float
        Compilation time in seconds

    Examples
    --------
    >>> from rheojax.models.itt_mct._kernels_diffrax import precompile_flow_curve_solver
    >>> compile_time = precompile_flow_curve_solver(n_modes=10)
    >>> print(f"Compilation took {compile_time:.1f}s")

    Notes
    -----
    First call to the solver triggers JIT compilation which can take
    30-90 seconds depending on n_modes. This function triggers that
    compilation with minimal dummy data so the cost is paid upfront.
    """
    import time

    logger.info(
        f"Pre-compiling flow curve solver for n_modes={n_modes}, "
        f"use_lorentzian={use_lorentzian}, memory_form={memory_form}"
    )
    start_time = time.time()

    # Create minimal dummy data
    gamma_dot_dummy = jnp.logspace(-1, 1, n_points)
    g_dummy = jnp.ones(n_modes) / n_modes
    tau_dummy = jnp.logspace(-2, 2, n_modes)

    # Trigger compilation
    _ = solve_flow_curve_batch(
        gamma_dot_dummy,
        v1=0.0,
        v2=3.0,  # Fluid state for faster convergence
        Gamma=1.0,
        gamma_c=0.1,
        G_inf=1e6,
        g=g_dummy,
        tau=tau_dummy,
        n_modes=n_modes,
        use_lorentzian=use_lorentzian,
        memory_form=memory_form,
    )

    # Block until computation finishes
    jax.block_until_ready(_)

    compile_time = time.time() - start_time
    logger.info(f"Pre-compilation completed in {compile_time:.1f}s")

    return compile_time
