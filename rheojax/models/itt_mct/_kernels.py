"""JIT-compiled ODE kernels for ITT-MCT models.

This module provides high-performance JAX-jitted functions for ITT-MCT
protocol integrations using the Volterra ODE approach with Prony modes.

Functions
---------
f12_volterra_flow_curve_rhs
    ODE right-hand side for steady flow with F₁₂ model
f12_volterra_startup_rhs
    ODE right-hand side for startup flow
f12_volterra_oscillation_rhs
    ODE right-hand side for small amplitude oscillation
f12_volterra_creep_rhs
    ODE right-hand side for creep compliance
f12_volterra_relaxation_rhs
    ODE right-hand side for stress relaxation
f12_volterra_laos_rhs
    ODE right-hand side for large amplitude oscillatory shear

Notes
-----
The Volterra approach converts the integral MCT equation to ODEs using
auxiliary variables Kᵢ for each Prony mode:

    dKᵢ/dt = -Kᵢ/τᵢ + gᵢ × (source term)

The memory integral ∫₀^t m(t-s) f(s) ds becomes Σᵢ Kᵢ(t).

Schematic Approximations
------------------------
This module implements the F₁₂ *schematic* ITT-MCT model, which makes several
simplifications compared to the full microscopic theory:

1. **Scalar correlator**: Full MCT tracks Φ(k,t) at each wave vector k.
   The schematic model uses a single scalar Φ(t) representing an "average"
   correlator. This loses k-dependent information but enables fast computation.

2. **Polynomial memory kernel**: Full MCT computes m(k,t) from integrals over
   all wave vectors weighted by the structure factor S(k). The F₁₂ schematic
   uses m(Φ) = v₁Φ + v₂Φ², which captures the essential feedback mechanism
   (slow relaxation → strong cage → slower relaxation) with empirical vertices.

3. **Strain decorrelation**: Full ITT-MCT advects wave vectors under flow,
   computing how each k(t,t') evolves. The schematic model approximates this
   with a simple decorrelation function h(γ) that depends only on accumulated
   strain, not on the detailed advection history.

4. **Stress from single correlator**: Full theory integrates over k-space.
   The schematic uses σ(t) = ∫ γ̇(t') G_∞ Φ(t,t')² dt', which captures
   qualitative behavior but not quantitative stress magnitudes.

These approximations are standard in the MCT literature (see Fuchs & Cates 2002,
Brader et al. 2008) and provide good qualitative predictions with ~5 parameters.
"""

from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


# =============================================================================
# F₁₂ Memory Kernel Helpers
# =============================================================================


@partial(jax.jit, static_argnames=())
def f12_memory(
    phi: jnp.ndarray,
    v1: float,
    v2: float,
) -> jnp.ndarray:
    """F₁₂ memory kernel m(Φ) = v₁Φ + v₂Φ².

    SCHEMATIC APPROXIMATION: The full MCT memory kernel involves a k-space
    integral over vertex functions V(k,q,|k-q|) weighted by S(k). The F₁₂
    polynomial form is the simplest approximation that captures the glass
    transition bifurcation at v₂ = 4 (for v₁ = 0).
    """
    return v1 * phi + v2 * phi * phi


@partial(jax.jit, static_argnames=())
def strain_decorrelation_gaussian(
    gamma: jnp.ndarray,
    gamma_c: float,
) -> jnp.ndarray:
    """Gaussian strain decorrelation h(γ) = exp(-(γ/γ_c)²).

    This is the most common form in the ITT-MCT literature (Fuchs & Cates 2002).
    It provides faster decay at large strains than the Lorentzian form.

    SCHEMATIC APPROXIMATION: Full ITT-MCT tracks advected wave vectors k(t,t')
    and computes decorrelation from the angle-averaged correlation of advected
    density modes. The Gaussian form is an empirical fit to this behavior.
    """
    return jnp.exp(-((gamma / gamma_c) ** 2))


@partial(jax.jit, static_argnames=())
def strain_decorrelation_lorentzian(
    gamma: jnp.ndarray,
    gamma_c: float,
) -> jnp.ndarray:
    """Lorentzian strain decorrelation h(γ) = 1 / (1 + (γ/γ_c)²).

    Alternative form with slower algebraic decay at large strains.
    Used in some MCT studies (e.g., Brader et al. 2008).

    Compared to Gaussian:
    - Slower decay: h(γ) ~ 1/γ² vs exp(-γ²)
    - Broader transition region around γ_c
    - May better capture materials with extended yielding transitions
    """
    return 1.0 / (1.0 + (gamma / gamma_c) ** 2)


@partial(jax.jit, static_argnames=("use_lorentzian",))
def strain_decorrelation(
    gamma: jnp.ndarray,
    gamma_c: float,
    use_lorentzian: bool = False,
) -> jnp.ndarray:
    """Strain decorrelation function h(γ) for ITT-MCT.

    Parameters
    ----------
    gamma : jnp.ndarray
        Accumulated strain since reference time
    gamma_c : float
        Critical strain for cage breaking (typically 0.05-0.2)
    use_lorentzian : bool, default False
        If True, use Lorentzian form h(γ) = 1/(1+(γ/γ_c)²)
        If False (default), use Gaussian form h(γ) = exp(-(γ/γ_c)²)

    Returns
    -------
    jnp.ndarray
        Decorrelation factor h(γ) in [0, 1]

    Notes
    -----
    SCHEMATIC APPROXIMATION: This function replaces the full advected-wavevector
    calculation of the two-time correlator Φ(t,t'). In full ITT-MCT:

        Φ(t,t') = ⟨ρ_k(t,t')(t) ρ_{-k}(t')⟩ / NS(k)

    where k(t,t') is the back-advected wave vector. The schematic approximation
    factors this as Φ(t,t') ≈ Φ_eq(t-t') × h(γ(t,t')), which captures the
    essential physics that accumulated strain destroys the cage correlation.
    """
    if use_lorentzian:
        return strain_decorrelation_lorentzian(gamma, gamma_c)
    else:
        return strain_decorrelation_gaussian(gamma, gamma_c)


# =============================================================================
# Flow Curve (Steady Shear) Integration
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_volterra_flow_curve_rhs(
    state: jnp.ndarray,
    t: float,
    gamma_dot: float,
    v1: float,
    v2: float,
    Gamma: float,
    gamma_c: float,
    G_inf: float,
    g: jnp.ndarray,  # Prony amplitudes
    tau: jnp.ndarray,  # Prony times
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
) -> jnp.ndarray:
    """ODE right-hand side for steady flow correlator evolution.

    State vector: [Φ, K₁, K₂, ..., Kₙ, γ_accumulated]
    where Kᵢ are Prony mode auxiliary variables.

    Parameters
    ----------
    state : jnp.ndarray
        Current state [Φ, K₁...Kₙ, γ]
    t : float
        Current time
    gamma_dot : float
        Applied shear rate
    v1, v2 : float
        F₁₂ vertex coefficients
    Gamma : float
        Bare relaxation rate
    gamma_c : float
        Critical strain for decorrelation
    G_inf : float
        High-frequency modulus
    g, tau : jnp.ndarray
        Prony mode amplitudes and times
    n_modes : int
        Number of Prony modes
    use_lorentzian : bool, default False
        Use Lorentzian (True) or Gaussian (False) decorrelation
    memory_form : str, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation h[γ_acc] (standard schematic)
        - "full": two-time decorrelation h[γ_total] × h[γ_mode] per Prony mode

    Returns
    -------
    jnp.ndarray
        Time derivatives [dΦ/dt, dK₁/dt, ..., dKₙ/dt, dγ/dt]
    """
    # Unpack state
    phi = state[0]
    K = state[1 : 1 + n_modes]
    gamma_acc = state[1 + n_modes]

    # Strain decorrelation for advected correlator (always applied)
    h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)

    # Advected correlator
    phi_advected = phi * h_gamma

    # Memory kernel from correlator
    m_phi = f12_memory(phi_advected, v1, v2)

    # Memory integral from Prony modes: ∫ m(t-s) dΦ/ds ds ≈ Σ Kᵢ
    memory_integral = jnp.sum(K)

    # MCT equation: dΦ/dt = -Γ(Φ + memory_integral)
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony mode evolution with memory form selection
    if memory_form == "full":
        # Full two-time: mode-specific decorrelation h[γ_dot × τᵢ]
        # Each Prony mode sees strain accumulated over its characteristic time
        gamma_mode = gamma_dot * tau  # Effective strain age per mode
        h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
        # Mode evolution with mode-specific memory decorrelation
        dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
    else:
        # Simplified: single decorrelation (standard schematic behavior)
        dK_dt = -K / tau + g * m_phi * dphi_dt

    # Strain accumulation
    dgamma_dt = gamma_dot

    return jnp.concatenate([jnp.array([dphi_dt]), dK_dt, jnp.array([dgamma_dt])])


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_steady_state_stress(
    phi_steady: float,
    gamma_dot: float,
    G_inf: float,
    gamma_c: float,
    v1: float,
    v2: float,
    Gamma: float,
    g: jnp.ndarray,
    tau: jnp.ndarray,
    n_modes: int,
    use_lorentzian: bool = False,
    memory_form: str = "simplified",
) -> float:
    """Compute steady-state stress from steady-state correlator.

    The stress is given by:
        σ = γ̇ ∫₀^∞ G(τ) dτ = γ̇ × G_inf × ∫₀^∞ Φ_ss(τ) dτ

    For steady shear, the correlator depends on accumulated strain.

    Parameters
    ----------
    phi_steady : float
        Steady-state correlator value (may be non-zero for glass)
    gamma_dot : float
        Shear rate
    G_inf : float
        High-frequency modulus
    gamma_c : float
        Critical strain
    v1, v2 : float
        Vertex coefficients
    Gamma : float
        Bare relaxation rate
    g, tau : jnp.ndarray
        Prony parameters
    n_modes : int
        Number of modes
    use_lorentzian : bool, default False
        Use Lorentzian (True) or Gaussian (False) decorrelation
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"

    Returns
    -------
    float
        Steady-state stress σ
    """
    # For a simple estimate, use the integral of the memory kernel
    # Full calculation would require solving for Φ(t) and integrating

    # Characteristic time scale
    tau_eff = 1.0 / Gamma

    # Strain over one relaxation time
    gamma_eff = gamma_dot * tau_eff

    # Decorrelation factor
    h = strain_decorrelation(gamma_eff, gamma_c, use_lorentzian)

    # Memory contribution
    m = f12_memory(phi_steady * h, v1, v2)

    # For full memory form, include mode-averaged additional decorrelation
    if memory_form == "full":
        # Average mode decorrelation: mean over Prony modes
        gamma_modes = gamma_dot * tau
        h_modes = strain_decorrelation(gamma_modes, gamma_c, use_lorentzian)
        h_mode_avg = jnp.mean(h_modes)
        m = m * h_mode_avg

    # Approximate steady stress: σ ≈ G_inf × γ̇ × (Φ_ss + memory) / Γ
    stress = G_inf * gamma_dot * (phi_steady * h + m) / Gamma

    return stress


# =============================================================================
# Startup Flow Integration
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_volterra_startup_rhs(
    state: jnp.ndarray,
    t: float,
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
) -> jnp.ndarray:
    """ODE right-hand side for startup flow.

    State vector: [Φ, K₁, ..., Kₙ, γ, σ]

    Tracks stress evolution σ(t) = γ̇ ∫₀^t G(t') dt' during startup.

    Parameters
    ----------
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"
    """
    # Unpack state
    phi = state[0]
    K = state[1 : 1 + n_modes]
    gamma_acc = state[1 + n_modes]
    # state[2 + n_modes] is sigma (stress) - computed separately

    # Strain decorrelation
    h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)
    phi_advected = phi * h_gamma

    # Memory kernel
    m_phi = f12_memory(phi_advected, v1, v2)

    # Memory integral
    memory_integral = jnp.sum(K)

    # Correlator evolution
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony modes with memory form selection
    if memory_form == "full":
        gamma_mode = gamma_dot * tau
        h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
        dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
    else:
        dK_dt = -K / tau + g * m_phi * dphi_dt

    # Strain accumulation
    dgamma_dt = gamma_dot

    # Stress accumulation: dσ/dt = G(t) × γ̇ ≈ G_inf × Φ(t) × γ̇
    dsigma_dt = G_inf * phi_advected * gamma_dot

    return jnp.concatenate(
        [jnp.array([dphi_dt]), dK_dt, jnp.array([dgamma_dt, dsigma_dt])]
    )


# =============================================================================
# Small Amplitude Oscillation (SAOS)
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes",))
def f12_equilibrium_correlator_rhs(
    state: jnp.ndarray,
    t: float,
    v1: float,
    v2: float,
    Gamma: float,
    g: jnp.ndarray,
    tau: jnp.ndarray,
    n_modes: int,
) -> jnp.ndarray:
    """ODE for equilibrium (quiescent) correlator Φ_eq(t).

    State: [Φ, K₁, ..., Kₙ]

    No strain decorrelation - pure relaxation dynamics.
    """
    phi = state[0]
    K = state[1 : 1 + n_modes]

    # Memory kernel (no strain decorrelation)
    m_phi = f12_memory(phi, v1, v2)

    # Memory integral
    memory_integral = jnp.sum(K)

    # Correlator decay
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony modes
    dK_dt = -K / tau + g * m_phi * dphi_dt

    return jnp.concatenate([jnp.array([dphi_dt]), dK_dt])


@partial(jax.jit, static_argnames=())
def compute_complex_modulus_from_correlator(
    omega: jnp.ndarray,
    t: jnp.ndarray,
    phi_eq: jnp.ndarray,
    G_inf: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute G*(ω) from equilibrium correlator via Fourier transform.

    G*(ω) = iω ∫₀^∞ G(t) e^{-iωt} dt
          = iω × G_inf × ∫₀^∞ Φ_eq(t) e^{-iωt} dt

    Parameters
    ----------
    omega : jnp.ndarray
        Angular frequencies (rad/s)
    t : jnp.ndarray
        Time array for correlator
    phi_eq : jnp.ndarray
        Equilibrium correlator values
    G_inf : float
        High-frequency modulus

    Returns
    -------
    G_prime : jnp.ndarray
        Storage modulus G'(ω)
    G_double_prime : jnp.ndarray
        Loss modulus G''(ω)
    """

    # Fourier transform via trapezoidal integration
    # ∫ Φ(t) e^{-iωt} dt ≈ Σ Φ(tᵢ) e^{-iωtᵢ} Δt
    def fourier_transform_single_omega(omega_val):
        exp_factor = jnp.exp(-1j * omega_val * t)
        integral = jnp.trapezoid(phi_eq * exp_factor, t)
        return integral

    # Vectorize over omega
    integrals = jax.vmap(fourier_transform_single_omega)(omega)

    # G*(ω) = iω × G_inf × integral
    G_star = 1j * omega * G_inf * integrals

    G_prime = jnp.real(G_star)
    G_double_prime = jnp.imag(G_star)

    return G_prime, G_double_prime


# =============================================================================
# Creep Compliance Integration
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_volterra_creep_rhs(
    state: jnp.ndarray,
    t: float,
    sigma_applied: float,
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
    """ODE right-hand side for creep at constant stress.

    State: [Φ, K₁, ..., Kₙ, γ, γ̇]

    The shear rate γ̇(t) is implicitly determined by the constraint
    σ₀ = ∫ γ̇(t') G(t,t') dt'.

    For simplicity, we use an explicit approximation where γ̇ adjusts
    to maintain the applied stress.

    Parameters
    ----------
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"
    """
    # Unpack state
    phi = state[0]
    K = state[1 : 1 + n_modes]
    gamma_acc = state[1 + n_modes]
    gamma_dot_current = state[2 + n_modes]

    # Strain decorrelation
    h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)
    phi_advected = phi * h_gamma

    # Memory kernel
    m_phi = f12_memory(phi_advected, v1, v2)

    # Memory integral
    memory_integral = jnp.sum(K)

    # Correlator evolution
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony modes with memory form selection
    if memory_form == "full":
        gamma_mode = gamma_dot_current * tau
        h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
        dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
    else:
        dK_dt = -K / tau + g * m_phi * dphi_dt

    # Strain rate from stress constraint (simplified)
    # σ = G(t) × γ̇ → γ̇ ≈ σ / G(t) where G(t) ≈ G_inf × Φ
    G_current = G_inf * jnp.maximum(phi_advected, 1e-10)
    gamma_dot_target = sigma_applied / G_current

    # Smooth adjustment toward target rate
    tau_adjust = 1.0 / Gamma  # Adjustment timescale
    dgamma_dot_dt = (gamma_dot_target - gamma_dot_current) / tau_adjust

    # Strain accumulation
    dgamma_dt = gamma_dot_current

    return jnp.concatenate(
        [jnp.array([dphi_dt]), dK_dt, jnp.array([dgamma_dt, dgamma_dot_dt])]
    )


# =============================================================================
# Stress Relaxation Integration
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_volterra_relaxation_rhs(
    state: jnp.ndarray,
    t: float,
    gamma_pre: float,
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
    """ODE right-hand side for stress relaxation after cessation of flow.

    State: [Φ, K₁, ..., Kₙ, σ]

    After stopping, γ̇ = 0 but the correlator continues to evolve
    with memory of the pre-shear history.

    Parameters
    ----------
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"
    """
    # Unpack state
    phi = state[0]
    K = state[1 : 1 + n_modes]
    # state[1 + n_modes] is sigma (stress) - computed separately

    # For relaxation, strain is fixed at pre-shear value
    h_gamma = strain_decorrelation(gamma_pre, gamma_c, use_lorentzian)
    phi_advected = phi * h_gamma

    # Memory kernel
    m_phi = f12_memory(phi_advected, v1, v2)

    # Memory integral
    memory_integral = jnp.sum(K)

    # Correlator evolution (toward equilibrium, modulated by pre-strain)
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony modes with memory form selection
    # For relaxation (γ̇=0), full form reduces to simplified since γ_mode = 0
    # But we keep the structure for consistency
    if memory_form == "full":
        # During relaxation, γ̇=0, so h_mode → 1 (no additional decorrelation)
        # This is physically correct: no strain accumulation during relaxation
        dK_dt = -K / tau + g * m_phi * dphi_dt
    else:
        dK_dt = -K / tau + g * m_phi * dphi_dt

    # Stress relaxation: dσ/dt = -σ/τ_rel where τ_rel ~ correlator timescale
    # More accurate: σ(t) ∝ Φ(t) so dσ/dt ∝ dΦ/dt
    dsigma_dt = G_inf * gamma_pre * h_gamma * dphi_dt

    return jnp.concatenate([jnp.array([dphi_dt]), dK_dt, jnp.array([dsigma_dt])])


# =============================================================================
# LAOS (Large Amplitude Oscillatory Shear)
# =============================================================================


@partial(jax.jit, static_argnames=("n_modes", "use_lorentzian", "memory_form"))
def f12_volterra_laos_rhs(
    state: jnp.ndarray,
    t: float,
    gamma_0: float,
    omega: float,
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
    """ODE right-hand side for LAOS.

    State: [Φ, K₁, ..., Kₙ, γ, σ]

    Applied strain: γ(t) = γ₀ sin(ωt)
    Strain rate: γ̇(t) = γ₀ ω cos(ωt)

    Parameters
    ----------
    memory_form : str, default "simplified"
        Memory kernel form: "simplified" or "full"
    """
    # Unpack state
    phi = state[0]
    K = state[1 : 1 + n_modes]
    gamma_acc = state[1 + n_modes]  # Absolute accumulated strain
    # state[2 + n_modes] is sigma (stress) - computed separately

    # Current strain rate (strain itself not needed for ODE)
    gamma_dot_current = gamma_0 * omega * jnp.cos(omega * t)

    # Strain decorrelation based on accumulated |γ|
    h_gamma = strain_decorrelation(gamma_acc, gamma_c, use_lorentzian)
    phi_advected = phi * h_gamma

    # Memory kernel
    m_phi = f12_memory(phi_advected, v1, v2)

    # Memory integral
    memory_integral = jnp.sum(K)

    # Correlator evolution with oscillatory driving
    dphi_dt = -Gamma * (phi + memory_integral)

    # Prony modes with memory form selection
    if memory_form == "full":
        # Use absolute value since LAOS has oscillating γ̇
        gamma_mode = jnp.abs(gamma_dot_current) * tau
        h_mode = strain_decorrelation(gamma_mode, gamma_c, use_lorentzian)
        dK_dt = -K / tau + g * m_phi * h_mode * dphi_dt
    else:
        dK_dt = -K / tau + g * m_phi * dphi_dt

    # Accumulated strain magnitude (for decorrelation tracking)
    dgamma_acc_dt = jnp.abs(gamma_dot_current)

    # Stress evolution: dσ/dt = G(t) × γ̇(t)
    dsigma_dt = G_inf * phi_advected * gamma_dot_current

    return jnp.concatenate(
        [jnp.array([dphi_dt]), dK_dt, jnp.array([dgamma_acc_dt, dsigma_dt])]
    )


# =============================================================================
# Harmonic Analysis for LAOS
# =============================================================================


@partial(jax.jit, static_argnames=("n_harmonics",))
def extract_laos_harmonics(
    t: jnp.ndarray,
    sigma: jnp.ndarray,
    omega: float,
    n_harmonics: int = 5,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract Fourier harmonics from LAOS stress response.

    σ(t) = Σₙ [σ'ₙ sin(nωt) + σ''ₙ cos(nωt)]

    Parameters
    ----------
    t : jnp.ndarray
        Time array
    sigma : jnp.ndarray
        Stress response
    omega : float
        Fundamental frequency
    n_harmonics : int, default 5
        Number of harmonics to extract (including fundamental)

    Returns
    -------
    sigma_prime_n : jnp.ndarray
        In-phase (elastic) coefficients [σ'₁, σ'₃, σ'₅, ...]
    sigma_double_prime_n : jnp.ndarray
        Out-of-phase (viscous) coefficients [σ''₁, σ''₃, σ''₅, ...]
    """
    T_period = 2 * jnp.pi / omega

    def extract_single_harmonic(n):
        """Extract nth harmonic coefficient."""
        sin_nwt = jnp.sin(n * omega * t)
        cos_nwt = jnp.cos(n * omega * t)

        # Fourier coefficients via integration
        sigma_prime = 2 * jnp.trapezoid(sigma * sin_nwt, t) / T_period
        sigma_double_prime = 2 * jnp.trapezoid(sigma * cos_nwt, t) / T_period

        return sigma_prime, sigma_double_prime

    # Extract odd harmonics (1, 3, 5, ...) - even harmonics are zero for symmetric response
    sigma_primes = []
    sigma_double_primes = []
    for n in range(1, 2 * n_harmonics, 2):
        sp, sdp = extract_single_harmonic(n)
        sigma_primes.append(sp)
        sigma_double_primes.append(sdp)

    return jnp.array(sigma_primes), jnp.array(sigma_double_primes)
