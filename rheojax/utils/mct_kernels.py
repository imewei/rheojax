"""MCT memory kernel computation utilities.

This module provides functions for computing Mode-Coupling Theory memory kernels
and related operations needed for ITT-MCT model implementations.

Functions
---------
f12_memory_kernel
    F₁₂ schematic model memory kernel m(Φ) = v₁Φ + v₂Φ²
advected_memory_decorrelation
    Strain decorrelation function h(γ) for advected correlators
prony_decompose_memory
    Fit memory kernel to Prony series for Volterra ODE integration
glass_transition_criterion
    Compute glass transition point from MCT vertex parameters
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

if TYPE_CHECKING:
    import jax

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


# =============================================================================
# F₁₂ Schematic Model Memory Kernel
# =============================================================================


@partial(jax.jit, static_argnames=())
def f12_memory_kernel(
    phi: jax.Array,
    v1: float,
    v2: float,
) -> jax.Array:
    """Compute F₁₂ schematic model memory kernel.

    The memory kernel for the F₁₂ model is a quadratic function of the
    density correlator Φ:

        m(Φ) = v₁Φ + v₂Φ²

    This represents the cage effect where particle motion is hindered
    by surrounding neighbors.

    Parameters
    ----------
    phi : jax.Array
        Density correlator Φ(t) or array of correlator values
    v1 : float
        Linear vertex coefficient (typically 0 for standard F₁₂)
    v2 : float
        Quadratic vertex coefficient (controls glass transition)

    Returns
    -------
    jax.Array
        Memory kernel m(Φ) with same shape as phi

    Notes
    -----
    The glass transition occurs at v₂ = 4 for v₁ = 0.
    For v₂ > 4, the system is a glass with arrested correlator.
    For v₂ < 4, the system is an ergodic fluid.

    References
    ----------
    Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids"
    """
    return v1 * phi + v2 * phi * phi


@partial(jax.jit, static_argnames=())
def f12_memory_kernel_derivative(
    phi: jax.Array,
    v1: float,
    v2: float,
) -> jax.Array:
    """Compute derivative of F₁₂ memory kernel with respect to Φ.

    dm/dΦ = v₁ + 2v₂Φ

    Parameters
    ----------
    phi : jax.Array
        Density correlator values
    v1 : float
        Linear vertex coefficient
    v2 : float
        Quadratic vertex coefficient

    Returns
    -------
    jax.Array
        Derivative dm/dΦ with same shape as phi
    """
    return v1 + 2.0 * v2 * phi


# =============================================================================
# Advected Memory / Strain Decorrelation
# =============================================================================


@partial(jax.jit, static_argnames=())
def advected_memory_decorrelation(
    gamma: jax.Array,
    gamma_c: float,
) -> jax.Array:
    """Compute strain decorrelation function h(γ).

    Under shear, the correlator decays due to advection of density fluctuations.
    The decorrelation function h(γ) quantifies this strain-induced destruction
    of structure:

        h(γ) = exp(-(γ/γ_c)²)

    where γ_c is the critical strain for cage breaking.

    Parameters
    ----------
    gamma : jax.Array
        Accumulated strain γ(t,t') = ∫_{t'}^t γ̇(s) ds
    gamma_c : float
        Critical strain parameter (typically 0.05-0.2)

    Returns
    -------
    jax.Array
        Decorrelation factor h(γ) ∈ [0, 1]

    Notes
    -----
    - h(0) = 1 (no decorrelation at zero strain)
    - h(γ) → 0 for |γ| >> γ_c (complete decorrelation)
    - The Gaussian form is physically motivated by isotropic strain
    """
    # KRN-009: Guard against gamma_c=0 division
    gamma_c_safe = jnp.maximum(gamma_c, 1e-10)
    gamma_normalized = gamma / gamma_c_safe
    return jnp.exp(-gamma_normalized * gamma_normalized)


@partial(jax.jit, static_argnames=())
def advected_correlator(
    phi_eq: jax.Array,
    gamma: jax.Array,
    gamma_c: float,
) -> jax.Array:
    """Compute advected correlator Φ(t,t') under shear.

    The advected correlator is the equilibrium correlator multiplied by the
    strain decorrelation function:

        Φ(t,t') = Φ_eq(t-t') × h(γ(t,t'))

    Parameters
    ----------
    phi_eq : jax.Array
        Equilibrium (quiescent) correlator Φ_eq(τ)
    gamma : jax.Array
        Accumulated strain between t' and t
    gamma_c : float
        Critical strain parameter

    Returns
    -------
    jax.Array
        Advected correlator Φ(t,t')
    """
    h_gamma = advected_memory_decorrelation(gamma, gamma_c)
    return phi_eq * h_gamma


@partial(jax.jit, static_argnames=("use_lorentzian",))
def two_time_strain_decorrelation(
    gamma_total: jax.Array,
    gamma_since_s: jax.Array,
    gamma_c: float,
    use_lorentzian: bool = False,
) -> jax.Array:
    """Compute two-time strain decorrelation for full ITT-MCT memory kernel.

    The full ITT-MCT memory kernel (Fuchs & Cates 2002) requires two separate
    strain decorrelation factors:

        m(t,s,t₀) = h[γ(t,t₀)] × h[γ(t,s)] × (v₁Φ + v₂Φ²)

    where:
    - h[γ(t,t₀)]: decorrelation from accumulated strain since flow start (t₀)
    - h[γ(t,s)]: decorrelation during the memory integral (from time s to t)

    This captures the physics that both the total strain history AND the strain
    accumulated during the memory integral contribute to cage breaking.

    Parameters
    ----------
    gamma_total : jax.Array
        Accumulated strain γ(t,t₀) since flow started (reference time t₀)
    gamma_since_s : jax.Array
        Strain accumulated since memory time s: γ(t,s) = ∫_s^t γ̇(τ) dτ.
        For Prony formulation, this is approximated as γ̇ × τᵢ per mode.
    gamma_c : float
        Critical strain for cage breaking (typically 0.05-0.2)
    use_lorentzian : bool, default False
        If True, use Lorentzian form h(γ) = 1/(1+(γ/γ_c)²)
        If False (default), use Gaussian form h(γ) = exp(-(γ/γ_c)²)

    Returns
    -------
    jax.Array
        Two-time decorrelation factor h[γ_total] × h[γ_since_s] in [0, 1]

    Notes
    -----
    In the Prony formulation, each mode Kᵢ has characteristic time τᵢ.
    The "memory age" for mode i is approximated as γ_since_s ≈ γ̇ × τᵢ.

    This captures the physical insight that:
    - Slow modes (large τ) see more accumulated strain → stronger decorrelation
    - Fast modes (small τ) retain more cage structure → weaker decorrelation

    References
    ----------
    Fuchs M. & Cates M.E. (2002) Phys. Rev. Lett. 89, 248304
    Fuchs M. & Cates M.E. (2003) Faraday Discuss. 123, 267
    """
    if use_lorentzian:
        h_total = 1.0 / (1.0 + (gamma_total / gamma_c) ** 2)
        h_since_s = 1.0 / (1.0 + (gamma_since_s / gamma_c) ** 2)
    else:
        h_total = jnp.exp(-((gamma_total / gamma_c) ** 2))
        h_since_s = jnp.exp(-((gamma_since_s / gamma_c) ** 2))

    return h_total * h_since_s


# =============================================================================
# Prony Series Decomposition
# =============================================================================


def prony_decompose_memory(
    t: np.ndarray,
    m_t: np.ndarray,
    n_modes: int = 10,
    method: str = "leastsq",
    n_starts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose memory kernel into Prony series.

    Fit the memory kernel m(t) to a sum of exponentials (Prony series):

        m(t) ≈ Σᵢ gᵢ exp(-t/τᵢ)

    This decomposition enables efficient Volterra ODE integration by
    converting the integral memory term to local auxiliary variables.

    Parameters
    ----------
    t : np.ndarray
        Time array (must be positive, monotonic)
    m_t : np.ndarray
        Memory kernel values m(t)
    n_modes : int, default 10
        Number of Prony modes to fit
    method : str, default "leastsq"
        Fitting method: "leastsq", "multistart", or "log_spacing"
    n_starts : int, default 5
        Number of starting points for multi-start optimization

    Returns
    -------
    g : np.ndarray
        Prony mode amplitudes [g₁, ..., g_n]
    tau : np.ndarray
        Prony mode relaxation times [τ₁, ..., τ_n]

    Notes
    -----
    The Volterra ODE approach introduces auxiliary variables Kᵢ(t):

        dKᵢ/dt = -Kᵢ/τᵢ + gᵢ × (source term)

    The memory integral ∫₀^t m(t-t') f(t') dt' then becomes Σᵢ Kᵢ(t).

    This converts the O(N²) full history integration to O(N) per step.

    The Prony fitting problem is ill-conditioned due to coupling between
    g and τ parameters. Multi-start optimization explores multiple local
    minima to find a robust solution.
    """
    t = np.asarray(t)
    m_t = np.asarray(m_t)

    # Filter valid data (positive kernel values)
    valid_mask = (t > 0) & (m_t > 1e-15)
    if valid_mask.sum() < n_modes:
        logger.warning(
            f"Only {valid_mask.sum()} valid points for {n_modes} modes. "
            "Reducing n_modes."
        )
        n_modes = max(2, valid_mask.sum() // 2)

    t_valid = t[valid_mask]
    m_valid = m_t[valid_mask]

    if method == "log_spacing":
        return _prony_log_spacing(t_valid, m_valid, n_modes)

    elif method == "multistart":
        return _prony_multistart(t_valid, m_valid, n_modes, n_starts)

    else:  # Default: leastsq with smart initialization and fallback
        try:
            return _prony_leastsq_robust(t_valid, m_valid, n_modes)
        except RuntimeError:
            logger.info("Robust leastsq failed, trying multi-start")
            try:
                return _prony_multistart(t_valid, m_valid, n_modes, n_starts)
            except RuntimeError:
                logger.warning("All optimization methods failed, using log-spacing")
                return _prony_log_spacing(t_valid, m_valid, n_modes)


def _prony_log_spacing(
    t_valid: np.ndarray,
    m_valid: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple Prony decomposition with log-spaced relaxation times.

    Uses fixed log-spaced τ values and linear least squares for g values.
    Fast and robust but may not capture optimal mode distribution.
    """
    tau = np.logspace(np.log10(t_valid.min()), np.log10(t_valid.max()), n_modes)

    # Solve for amplitudes via least squares: m(t) ≈ A @ g
    A = np.exp(-t_valid[:, None] / tau[None, :])
    g, _, _, _ = np.linalg.lstsq(A, m_valid, rcond=None)

    # Ensure non-negative amplitudes
    g = np.maximum(g, 0.0)

    # Sort by relaxation time (ascending)
    sort_idx = np.argsort(tau)
    return g[sort_idx], tau[sort_idx]


def _prony_smart_init(
    t_valid: np.ndarray,
    m_valid: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate smart initial guess for Prony parameters.

    Uses log-linear interpolation of kernel decay to estimate characteristic
    times, then estimates amplitudes from local slopes.
    """
    # Log-spaced relaxation times spanning data range
    tau_init = np.logspace(
        np.log10(t_valid.min() * 0.5),
        np.log10(t_valid.max() * 2),
        n_modes,
    )

    # Estimate amplitudes using log-linear interpolation of m(t)
    # For m(t) ~ g*exp(-t/τ), we have log(m) ~ log(g) - t/τ
    # Estimate g from local values of m at times near τ
    g_init = np.zeros(n_modes)
    for i, tau_i in enumerate(tau_init):
        # Find time point closest to tau_i
        idx = np.argmin(np.abs(t_valid - tau_i))
        # Estimate g from m(τ) ≈ g * exp(-1) = g * 0.368
        g_init[i] = m_valid[idx] / 0.368 / max(n_modes, 1)

    # Ensure positive and scale to match total kernel area
    g_init = np.maximum(g_init, m_valid[0] / max(n_modes, 1) / 10)

    return g_init, tau_init


def _prony_residuals(
    params: np.ndarray,
    t_data: np.ndarray,
    m_data: np.ndarray,
    n_modes: int,
) -> np.ndarray:
    """Compute residuals for Prony fit.

    Works in log-space for τ to handle multi-scale optimization.
    """
    g = params[:n_modes]
    log_tau = params[n_modes:]
    tau = np.exp(log_tau)  # Work in log-space for better conditioning

    # Compute Prony series
    m_pred = np.sum(g[None, :] * np.exp(-t_data[:, None] / tau[None, :]), axis=1)

    # Relative residuals for better scaling
    scale = np.maximum(m_data, 1e-15)
    return (m_pred - m_data) / scale


def _prony_leastsq_robust(
    t_valid: np.ndarray,
    m_valid: np.ndarray,
    n_modes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust Prony fit using scipy.optimize.least_squares with TRF method.

    Uses smart initialization, log-space τ parameterization, and increased
    iteration limits for better convergence on ill-conditioned problems.
    """
    # Smart initialization
    g_init, tau_init = _prony_smart_init(t_valid, m_valid, n_modes)

    # Work in log-space for τ (better conditioning)
    log_tau_init = np.log(tau_init)
    p0 = np.concatenate([g_init, log_tau_init])

    # Bounds: g >= 0, log_tau unconstrained but reasonable
    tau_min = t_valid.min() / 100
    tau_max = t_valid.max() * 100
    lower = np.concatenate(
        [
            np.zeros(n_modes),
            np.full(n_modes, np.log(tau_min)),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_modes, np.inf),
            np.full(n_modes, np.log(tau_max)),
        ]
    )

    # Run optimization with increased limits
    result = least_squares(
        _prony_residuals,
        p0,
        args=(t_valid, m_valid, n_modes),
        bounds=(lower, upper),
        method="trf",
        max_nfev=20000,  # Increased from scipy's default
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    if not result.success:
        raise RuntimeError(f"Prony leastsq failed: {result.message}")

    # Extract results
    g = result.x[:n_modes]
    log_tau = result.x[n_modes:]
    tau = np.exp(log_tau)

    # Sort by relaxation time (ascending)
    sort_idx = np.argsort(tau)
    return g[sort_idx], tau[sort_idx]


def _prony_multistart(
    t_valid: np.ndarray,
    m_valid: np.ndarray,
    n_modes: int,
    n_starts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-start Prony fit to escape local minima.

    Runs multiple optimizations with perturbed initial conditions and
    returns the best solution (lowest residual sum of squares).
    """
    tau_min = t_valid.min() / 100
    tau_max = t_valid.max() * 100

    # Bounds for optimization (log-space for τ)
    lower = np.concatenate(
        [
            np.zeros(n_modes),
            np.full(n_modes, np.log(tau_min)),
        ]
    )
    upper = np.concatenate(
        [
            np.full(n_modes, np.inf),
            np.full(n_modes, np.log(tau_max)),
        ]
    )

    best_cost = np.inf
    best_g = None
    best_tau = None
    rng = np.random.default_rng(42)

    for i_start in range(n_starts):
        # Generate starting point
        if i_start == 0:
            # First: smart initialization
            g_init, tau_init = _prony_smart_init(t_valid, m_valid, n_modes)
        else:
            # Others: perturbed log-spaced initialization
            log_tau_spread = np.log10(t_valid.max() / t_valid.min())
            tau_init = np.logspace(
                np.log10(t_valid.min()) - 0.5 + rng.uniform(-0.3, 0.3) * log_tau_spread,
                np.log10(t_valid.max()) + 0.5 + rng.uniform(-0.3, 0.3) * log_tau_spread,
                n_modes,
            )
            # Perturbed amplitude initialization
            g_init = m_valid[0] / n_modes * (0.5 + rng.uniform(0, 1, n_modes))

        log_tau_init = np.log(tau_init)
        p0 = np.concatenate([g_init, log_tau_init])

        try:
            result = least_squares(
                _prony_residuals,
                p0,
                args=(t_valid, m_valid, n_modes),
                bounds=(lower, upper),
                method="trf",
                max_nfev=10000,
                ftol=1e-8,
                xtol=1e-8,
            )

            if result.cost < best_cost:
                best_cost = result.cost
                g = result.x[:n_modes]
                log_tau = result.x[n_modes:]
                tau = np.exp(log_tau)
                sort_idx = np.argsort(tau)
                best_g = g[sort_idx]
                best_tau = tau[sort_idx]

        except Exception as e:
            logger.debug(f"Multi-start iteration {i_start} failed: {e}")
            continue

    if best_g is None:
        raise RuntimeError("All multi-start iterations failed")

    logger.debug(f"Multi-start Prony fit: best cost = {best_cost:.2e}")
    return best_g, best_tau


# =============================================================================
# Glass Transition Criterion
# =============================================================================


def glass_transition_criterion(
    v1: float,
    v2: float,
) -> dict:
    """Compute glass transition properties for F₁₂ model.

    For the F₁₂ model with m(Φ) = v₁Φ + v₂Φ², the glass transition occurs
    when the non-ergodicity parameter f (long-time limit of Φ) becomes non-zero.

    At the glass transition: f_c = (1 - √(1 - λ_c)) / λ_c where
    λ_c = 1 for the F₁₂ model, giving f_c ≈ 0.293 (for v₁=0, v₂=4).

    Parameters
    ----------
    v1 : float
        Linear vertex coefficient
    v2 : float
        Quadratic vertex coefficient

    Returns
    -------
    dict
        Glass transition properties:
        - "is_glass": bool, whether system is in glass state
        - "epsilon": float, separation parameter ε = (v₂ - v₂_c)/v₂_c
        - "v2_critical": float, critical v₂ for transition
        - "f_neq": float, non-ergodicity parameter f (0 for fluid)
        - "lambda_exponent": float, MCT exponent parameter λ

    Notes
    -----
    The separation parameter ε controls the dynamics:
    - ε < 0: ergodic fluid, Φ(t→∞) = 0
    - ε = 0: critical point, power-law decay Φ ~ t^{-a}
    - ε > 0: glass state, Φ(t→∞) = f > 0

    The MCT exponent λ determines the critical exponents a and b via:
    Γ(1-a)²/Γ(1-2a) = Γ(1+b)²/Γ(1+2b) = λ
    """
    # For F₁₂ model with v₁=0: v₂_c = 4
    # General case requires solving self-consistent equation
    if abs(v1) < 1e-10:
        v2_critical = 4.0
        lambda_exponent = 1.0  # F₁₂ limit
    else:
        # Solve v₁*f + v₂*f² = f for critical point
        # This requires numerical solution in general
        # For now, use approximate formula
        v2_critical = (4.0 - 2.0 * v1) / (1.0 - v1 / 4.0) if v1 < 4.0 else 4.0
        lambda_exponent = 1.0 - v1 / v2_critical  # Approximate

    epsilon = (v2 - v2_critical) / v2_critical
    is_glass = epsilon > 0

    # Non-ergodicity parameter
    if is_glass:
        # Solve f = m(f) = v₁f + v₂f² for f in glass state
        # f(1 - v₁ - v₂f) = 0 → f = (1 - v₁) / v₂ for glass
        f_neq = max(0.0, (1.0 - v1) / v2) if v2 > 0 else 0.0
    else:
        f_neq = 0.0

    return {
        "is_glass": is_glass,
        "epsilon": epsilon,
        "v2_critical": v2_critical,
        "f_neq": f_neq,
        "lambda_exponent": lambda_exponent,
    }


# =============================================================================
# Equilibrium Correlator Solver
# =============================================================================


# =============================================================================
# Microscopic Stress Computation
# =============================================================================


def setup_microscopic_stress_weights(
    phi_volume: float,
    k_min: float = 1.0,
    k_max: float = 30.0,
    n_k: int = 100,
    k_BT: float = 1.0,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute weights for microscopic stress integration.

    The microscopic MCT stress formula involves a k-space integral:

        σ = (k_BT / 60π²) × ∫₀^∞ dk k⁴ × [S'(k)/S(k)²]² × Φ(k,t)²

    This function pre-computes the quadrature weights:

        w(k) = (k_BT / 60π²) × k⁴ × [S'(k)/S(k)²]² × Δk

    so that the stress can be computed as σ ≈ Σᵢ w(kᵢ) × Φ(kᵢ)².

    Parameters
    ----------
    phi_volume : float
        Volume fraction φ ∈ (0, 0.64) for hard spheres.
        Used to compute Percus-Yevick structure factor S(k).
    k_min : float, default 1.0
        Minimum wave vector (should be above k=0 singularities)
    k_max : float, default 30.0
        Maximum wave vector (should capture S(k) peak around k*σ ≈ 7)
    n_k : int, default 100
        Number of k-points for quadrature
    k_BT : float, default 1.0
        Thermal energy k_B × T in Joules.
        Default 1.0 gives dimensionless stress; use 4.11e-21 J for T=298K.
    sigma : float, default 1.0
        Particle diameter (m). Default 1.0 for dimensionless units.

    Returns
    -------
    k_array : np.ndarray
        Wave vector array of shape (n_k,)
    weights : np.ndarray
        Pre-computed stress weights of shape (n_k,)

    Notes
    -----
    The structure factor S(k) is computed using the Percus-Yevick
    approximation from rheojax.utils.structure_factor.

    For schematic F₁₂ models with a single scalar correlator Φ(t),
    the k-dependence must be approximated. A common approach is to
    use the total weight: σ ≈ W_total × Φ(t)² where W_total = Σᵢ w(kᵢ).

    References
    ----------
    Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids", Eq. 4.35
    Fuchs M. & Cates M.E. (2009) J. Rheol. 53, 957
    """
    from rheojax.utils.structure_factor import percus_yevick_sk, sk_derivatives

    # Create k-grid
    k_array = np.linspace(k_min, k_max, n_k)
    dk = k_array[1] - k_array[0]

    # Compute structure factor
    S_k = percus_yevick_sk(k_array, phi_volume, sigma=sigma)

    # Compute derivative dS/dk
    dS_dk, _ = sk_derivatives(k_array, S_k, method="spline")

    # Prefactor: k_BT / (60 * π²)
    prefactor = k_BT / (60.0 * np.pi**2)

    # Compute weights: prefactor × k⁴ × (S'/S²)² × dk
    # Handle potential division issues near S(k) = 0
    S_k_safe = np.maximum(S_k, 1e-10)
    vertex_factor = (dS_dk / S_k_safe**2) ** 2

    weights = prefactor * k_array**4 * vertex_factor * dk

    logger.debug(
        f"Microscopic stress weights: φ={phi_volume:.3f}, "
        f"k=[{k_min:.1f},{k_max:.1f}], W_total={np.sum(weights):.3e}"
    )

    return k_array, weights


@partial(jax.jit, static_argnames=())
def compute_microscopic_stress(
    phi_squared_integrated: jax.Array,
    weights: jax.Array,
) -> float:
    """Compute microscopic stress from integrated correlator and weights.

    Parameters
    ----------
    phi_squared_integrated : jax.Array
        Time-integrated Φ² values at each k-point, shape (n_k,).
        For schematic models, this is a scalar broadcast to all k.
    weights : jax.Array
        Pre-computed stress weights from setup_microscopic_stress_weights()

    Returns
    -------
    float
        Stress σ (Pa if weights computed with proper k_BT units)

    Notes
    -----
    For schematic F₁₂ models with scalar Φ(t):
        phi_squared_integrated = ∫₀^∞ Φ(t)² dt  (scalar)
        σ ≈ sum(weights) × phi_squared_integrated

    For full k-resolved MCT:
        phi_squared_integrated[i] = ∫₀^∞ Φ(k_i,t)² dt
        σ = Σᵢ weights[i] × phi_squared_integrated[i]
    """
    return jnp.sum(weights * phi_squared_integrated)


def get_microscopic_stress_prefactor(
    phi_volume: float,
    k_min: float = 1.0,
    k_max: float = 30.0,
    n_k: int = 100,
    k_BT: float = 1.0,
    sigma: float = 1.0,
) -> float:
    """Get total microscopic stress prefactor for schematic models.

    For F₁₂ schematic models with a single correlator Φ(t), the full
    k-space integral reduces to a prefactor:

        σ = W_total × γ̇ × ∫₀^∞ Φ(t)² × h(γ(t)) dt

    where W_total = (k_BT/60π²) × ∫dk k⁴ [S'/S²]²

    This prefactor replaces G_∞ in the schematic stress formula.

    Parameters
    ----------
    phi_volume : float
        Volume fraction
    k_min, k_max : float
        k-space integration limits
    n_k : int
        Number of quadrature points
    k_BT : float
        Thermal energy
    sigma : float
        Particle diameter

    Returns
    -------
    float
        Total stress prefactor W_total (Pa·s when k_BT in J, σ in m)
    """
    _, weights = setup_microscopic_stress_weights(
        phi_volume, k_min, k_max, n_k, k_BT, sigma
    )
    return float(np.sum(weights))


@partial(jax.jit, static_argnames=("n_steps",))
def solve_equilibrium_correlator_f12(
    t_array: jax.Array,
    v1: float,
    v2: float,
    Gamma: float,
    n_steps: int = 10000,
) -> jax.Array:
    """Solve for equilibrium (quiescent) correlator Φ_eq(t) for F₁₂ model.

    Solves the MCT equation:

        ∂Φ/∂t + Γ[Φ + ∫₀^t m(Φ(t-s)) ∂Φ/∂s ds] = 0

    with initial condition Φ(0) = 1.

    Parameters
    ----------
    t_array : jax.Array
        Time points at which to evaluate correlator
    v1 : float
        Linear vertex coefficient
    v2 : float
        Quadratic vertex coefficient
    Gamma : float
        Bare relaxation rate (1/s)
    n_steps : int, default 10000
        Number of integration steps

    Returns
    -------
    jax.Array
        Equilibrium correlator Φ_eq(t) at requested times

    Notes
    -----
    Uses a simple explicit Euler scheme with decimation for the memory integral.
    For production use, consider adaptive time stepping or Volterra ODE approach.
    """
    t_max = jnp.max(t_array)
    dt = t_max / n_steps

    def scan_step(carry, _):
        """Single integration step."""
        phi, phi_hist, t_idx = carry

        # Memory integral via trapezoidal rule (simplified)
        m_phi = f12_memory_kernel(phi_hist, v1, v2)
        # Approximate derivative from finite difference
        dphi_hist = jnp.diff(phi_hist, prepend=phi_hist[0]) / dt

        # Memory integral: ∫₀^t m(Φ(t-s)) ∂Φ/∂s ds
        # Approximate with quadrature
        memory_integral = jnp.sum(m_phi * dphi_hist) * dt

        # MCT equation: dΦ/dt = -Γ(Φ + memory_integral)
        dphi_dt = -Gamma * (phi + memory_integral)

        # Update
        phi_new = phi + dt * dphi_dt
        phi_new = jnp.clip(phi_new, 0.0, 1.0)  # Physical bounds

        # Shift history
        phi_hist_new = jnp.roll(phi_hist, 1)
        phi_hist_new = phi_hist_new.at[0].set(phi_new)

        return (phi_new, phi_hist_new, t_idx + 1), phi_new

    # Initialize
    n_hist = min(1000, n_steps)  # Keep last 1000 steps
    phi_init = 1.0
    phi_hist_init = jnp.ones(n_hist)

    # Run integration
    (phi_final, _, _), phi_trajectory = jax.lax.scan(
        scan_step,
        (phi_init, phi_hist_init, 0),
        None,
        length=n_steps,
    )

    # Interpolate to requested times
    t_integration = jnp.linspace(0, t_max, n_steps)
    phi_at_times = jnp.interp(t_array, t_integration, phi_trajectory)

    return phi_at_times
