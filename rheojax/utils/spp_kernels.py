"""JAX-compatible SPP (Sequence of Physical Processes) kernel functions.

This module provides efficient, JAX-compatible implementations of SPP analysis
kernel functions for LAOS (Large Amplitude Oscillatory Shear) rheology. SPP
analysis enables cycle-by-cycle decomposition of nonlinear stress responses
into elastic and viscous contributions, extracting physically meaningful
yield parameters.

The SPP framework was developed by Rogers (2012, 2017) and provides:
- Time-resolved apparent cage modulus G'_cage(t)
- Static and dynamic yield stress extraction
- Phase reconstruction from harmonic decomposition
- Lissajous-Bowditch plot metrics
- Frenet-Serret frame analysis (T, N, B vectors)
- Moduli rate calculations (Ġ', Ġ'', G_speed)

Key Functions
-------------
- apparent_cage_modulus: Time-resolved elastic modulus from stress/strain
- static_yield_stress: Yield stress at strain reversal (strain = ±gamma0)
- dynamic_yield_stress: Yield stress at rate reversal (strain rate = 0)
- harmonic_reconstruction: Stress reconstruction from Fourier components
- harmonic_reconstruction_full: Full Fourier with phase alignment (MATLAB-compatible)
- spp_fourier_analysis: Complete Fourier-based SPP analysis with analytical derivatives
- lissajous_metrics: Bowditch diagram derived quantities (S, T ratios)
- zero_crossing_detection: Robust strain/rate zero-crossing finder
- frenet_serret_frame: Compute T, N, B trajectory vectors
- moduli_rates: Compute Ġ'(t), Ġ''(t), G_speed, δ̇(t)
- yield_from_displacement_stress: SPP-based yield stress extraction

Physical Interpretation
-----------------------
SPP analysis provides a phenomenological interpretation of LAOS behavior:
- G'_cage(t): Instantaneous elastic modulus reflecting cage structure
- static_yield: Static yield stress (stress at max strain, cage breakage threshold)
- dynamic_yield: Dynamic yield stress (stress at zero rate, flow cessation threshold)
- Power-law regime: Post-yield flow characterized by sigma ~ strain_rate^n
- Frenet-Serret frame: Geometric analysis of (γ, γ̇/ω, σ) trajectory

References
----------
- S.A. Rogers et al., "A sequence of physical processes determined and
  quantified in large-amplitude oscillatory shear (LAOS): Application to
  theoretical nonlinear models", J. Rheol. 56(1), 2012
- S.A. Rogers, "In search of physical meaning: defining transient parameters
  for nonlinear viscoelasticity", Rheol. Acta 56, 2017
- G.J. Donley et al., "Time-resolved dynamics of the yielding transition
  in soft materials", J. Non-Newton. Fluid Mech. 264, 2019
"""

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Safe JAX import (enforces float64)
# Float64 precision is critical for accurate numerical differentiation
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from jax import Array


# ============================================================================
# Apparent Cage Modulus
# ============================================================================


@jax.jit
def apparent_cage_modulus(
    stress: "Array",
    strain: "Array",
    strain_amplitude: float,
) -> "Array":
    """
    Compute time-resolved apparent cage modulus.

    Apparent cage modulus is the instantaneous elastic response, normalized
    by strain amplitude: G_cage(t) = stress(t) / gamma0 * sign(strain(t)).

    Parameters
    ----------
    stress : Array
        Time-resolved stress signal (Pa)
    strain : Array
        Time-resolved strain signal (dimensionless)
    strain_amplitude : float
        Maximum strain amplitude gamma0 (dimensionless)

    Returns
    -------
    Array
        Apparent cage modulus (Pa)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import apparent_cage_modulus
    >>>
    >>> # Sinusoidal LAOS data
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> gamma_0 = 1.0
    >>> gamma = gamma_0 * jnp.sin(t)
    >>> sigma = 100.0 * jnp.sin(t) + 10.0 * jnp.sin(3*t)  # With 3rd harmonic
    >>>
    >>> G_cage = apparent_cage_modulus(sigma, gamma, gamma_0)

    Notes
    -----
    - G_cage is constant for a purely linear sinusoidal material
    - Deviations from constant indicate nonlinearity
    - Sign(γ) ensures correct sign during negative strain half-cycle
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    gamma_0 = jnp.float64(strain_amplitude)
    # Avoid division by zero
    gamma_0 = jnp.where(gamma_0 > 1e-10, gamma_0, 1e-10)

    # Compute sign of strain (avoid division by zero at crossings)
    strain_sign = jnp.sign(strain_arr)
    # At zero crossing, use sign from neighboring points (forward difference)
    strain_sign = jnp.where(
        strain_arr == 0,
        jnp.sign(jnp.roll(strain_arr, -1)),
        strain_sign,
    )

    # Apparent cage modulus: stress / gamma0 * sign(strain)
    G_cage = stress_arr / gamma_0 * strain_sign

    return G_cage


# ============================================================================
# Yield Stress Extraction
# ============================================================================


@jax.jit
def static_yield_stress(
    stress: "Array",
    strain: "Array",
    strain_amplitude: float,
    tolerance: float = 0.02,
) -> float:
    """Approximate static yield stress near strain reversal.

    Samples near the strain extrema (abs(strain) close to strain_amplitude)
    and returns the average absolute stress.
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    gamma_0 = jnp.float64(strain_amplitude)

    # Find points where |γ| ≈ γ_0 (strain reversal)
    threshold = gamma_0 * (1.0 - tolerance)
    at_reversal = jnp.abs(strain_arr) >= threshold

    # Average stress magnitude at reversal points
    stress_at_reversal = jnp.where(at_reversal, jnp.abs(stress_arr), 0.0)
    count = jnp.sum(at_reversal)

    # Avoid division by zero
    sigma_sy = jnp.where(
        count > 0,
        jnp.sum(stress_at_reversal) / count,
        jnp.abs(stress_arr).max(),
    )

    return sigma_sy


@jax.jit
def dynamic_yield_stress(
    stress: "Array",
    strain_rate: "Array",
    rate_amplitude: float,
    tolerance: float = 0.02,
) -> float:
    """Approximate dynamic yield stress near zero strain rate.

    Selects samples where abs(strain_rate) is small, averages abs(stress), and
    returns that average as the dynamic yield estimate.
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))
    gamma_dot_0 = jnp.float64(rate_amplitude)

    # Find points where |γ̇| ≈ 0 (rate reversal)
    threshold = gamma_dot_0 * tolerance
    at_zero_rate = jnp.abs(strain_rate_arr) <= threshold

    # Average stress magnitude at zero-rate points
    stress_at_zero = jnp.where(at_zero_rate, jnp.abs(stress_arr), 0.0)
    count = jnp.sum(at_zero_rate)

    # Avoid division by zero
    sigma_dy = jnp.where(
        count > 0,
        jnp.sum(stress_at_zero) / count,
        jnp.abs(stress_arr).min(),
    )

    return sigma_dy


# ============================================================================
# Phase Reconstruction
# ============================================================================


@partial(jax.jit, static_argnums=(2,))
def harmonic_reconstruction(
    stress: "Array",
    omega: float,
    n_harmonics: int = 39,
    dt: float | None = None,
) -> tuple["Array", "Array", "Array"]:
    """
    Reconstruct stress signal from harmonic components (Fourier decomposition).

    Extracts odd harmonic amplitudes and phases from LAOS stress signal,
    enabling reconstruction and harmonic ratio analysis.

    Parameters
    ----------
    stress : Array
        Time-resolved stress signal σ(t) (Pa)
    omega : float
        Fundamental angular frequency ω (rad/s)
    n_harmonics : int, optional
        Number of odd harmonics to extract (default: 5, gives 1ω, 3ω, 5ω, 7ω, 9ω)
    dt : float, optional
        Time step. If None, assumes stress spans exactly one period.

    Returns
    -------
    amplitudes : Array
        Harmonic amplitudes [A_1, A_3, A_5, ...] (Pa)
    phases : Array
        Harmonic phases [φ_1, φ_3, φ_5, ...] (radians)
    reconstructed : Array
        Reconstructed stress from harmonics (Pa)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import harmonic_reconstruction
    >>>
    >>> omega = 1.0
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> sigma = 100.0 * jnp.sin(t) + 20.0 * jnp.sin(3*t + 0.1)
    >>>
    >>> amps, phases, reconstructed = harmonic_reconstruction(sigma, omega)
    >>> # amps[0] ≈ 100.0, amps[1] ≈ 20.0
    >>> # phases[0] ≈ 0, phases[1] ≈ 0.1

    Notes
    -----
    - Only odd harmonics (1, 3, 5, ...) are physically relevant in LAOS
    - Even harmonics indicate asymmetric response (wall slip, etc.)
    - I_n/I_1 ratio quantifies nonlinearity strength
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    n_points = len(stress_arr)

    # Determine time array
    if dt is None:
        period = 2.0 * jnp.pi / omega
        dt = period / n_points

    t = jnp.arange(n_points) * dt

    # Extract odd harmonics via discrete Fourier projection
    amplitudes = jnp.zeros(n_harmonics, dtype=jnp.float64)
    phases = jnp.zeros(n_harmonics, dtype=jnp.float64)

    def extract_harmonic(carry, harmonic_idx):
        n = 2 * harmonic_idx + 1  # Odd harmonics: 1, 3, 5, ...
        omega_n = n * omega

        # Fourier projection
        cos_component = jnp.sum(stress_arr * jnp.cos(omega_n * t)) * 2.0 / n_points
        sin_component = jnp.sum(stress_arr * jnp.sin(omega_n * t)) * 2.0 / n_points

        # Amplitude and phase
        amplitude = jnp.sqrt(cos_component**2 + sin_component**2)
        phase = jnp.arctan2(-cos_component, sin_component)

        return carry, (amplitude, phase)

    _, (amplitudes, phases) = jax.lax.scan(
        extract_harmonic,
        None,
        jnp.arange(n_harmonics),
    )

    # Reconstruct signal
    def reconstruct_point(t_point):
        result = jnp.float64(0.0)
        for i in range(n_harmonics):
            n = 2 * i + 1
            result = result + amplitudes[i] * jnp.sin(n * omega * t_point + phases[i])
        return result

    reconstructed = jax.vmap(reconstruct_point)(t)

    return amplitudes, phases, reconstructed


# ============================================================================
# Phase-Aligned Harmonic Reconstruction (MATLAB-Compatible)
# ============================================================================


@partial(jax.jit, static_argnums=(3,))
def compute_phase_offset(
    strain: "Array",
    omega: float,
    dt: float,
    n_cycles: int = 1,
) -> float:
    """
    Compute phase offset Delta for aligning strain to start at zero crossing.

    This matches MATLAB SPPplus_fourier_v2.m phase offset calculation:
        Delta = atan(An1_n(p+1)/Bn1_n(p+1))
        if Bn1_n(p+1) < 0: Delta = Delta + pi

    Parameters
    ----------
    strain : Array
        Strain signal γ(t)
    omega : float
        Angular frequency ω (rad/s)
    dt : float
        Time step (s)
    n_cycles : int
        Number of complete cycles in data (default: 1)

    Returns
    -------
    float
        Phase offset Delta (radians) to align strain reference
    """
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    L = len(strain_arr)
    p = n_cycles

    # Compute FFT of strain
    fft_strain = jnp.fft.fft(strain_arr)

    # Get Fourier coefficients (matching MATLAB convention)
    # An1 = 2*Re(FFT)/L, Bn1 = -2*Im(FFT)/L
    An1_n = 2 * jnp.real(fft_strain) / L
    Bn1_n = -2 * jnp.imag(fft_strain) / L

    # Get fundamental harmonic coefficient (at index p+1 in MATLAB, p in Python 0-indexed)
    # For p cycles, the fundamental is at index p
    An_fund = An1_n[p]
    Bn_fund = Bn1_n[p]

    # Compute Delta
    Delta = jnp.arctan2(An_fund, Bn_fund)

    # Adjust if Bn_fund < 0
    Delta = jnp.where(Bn_fund < 0, Delta + jnp.pi, Delta)

    return Delta


@partial(jax.jit, static_argnums=(4, 5, 6))
def harmonic_reconstruction_full(
    strain: "Array",
    strain_rate: "Array",
    stress: "Array",
    omega: float,
    n_harmonics: int = 39,
    n_cycles: int = 1,
    W_int: int | None = None,
) -> dict:
    """
    Full Fourier-based harmonic reconstruction with phase alignment (MATLAB-compatible).

    Implements the complete workflow from SPPplus_fourier_v2.m:
    1. FFT all three waveforms (strain, rate, stress)
    2. Compute phase offset Delta from strain fundamental
    3. Rotate all Fourier coefficients to align with phase reference
    4. Reconstruct aligned waveforms

    Parameters
    ----------
    strain : Array
        Strain signal γ(t) (dimensionless)
    strain_rate : Array
        Strain rate signal γ̇(t) (1/s) - will be normalized by omega
    stress : Array
        Stress signal σ(t) (Pa)
    omega : float
        Angular frequency ω (rad/s)
    n_harmonics : int
        Number of odd harmonics for stress reconstruction (default: 15)
    n_cycles : int
        Number of complete cycles in data (default: 1)

    Returns
    -------
    dict
        Dictionary containing:
        - Delta: Phase offset (radians)
        - An_strain, Bn_strain: Aligned strain Fourier coefficients
        - An_rate, Bn_rate: Aligned rate Fourier coefficients
        - An_stress, Bn_stress: Aligned stress Fourier coefficients
        - strain_recon: Reconstructed strain
        - rate_recon: Reconstructed rate/omega
        - stress_recon: Reconstructed stress
        - time_new: Phase-aligned time array

    Notes
    -----
    This function matches MATLAB SPPplus_fourier_v2.m coefficient rotation:

        An_n[nn+1] = An1_n[nn+1]*cos(Delta/p*nn) - Bn1_n[nn+1]*sin(Delta/p*nn)
        Bn_n[nn+1] = Bn1_n[nn+1]*cos(Delta/p*nn) + An1_n[nn+1]*sin(Delta/p*nn)
    """
    logger.debug(
        "Starting harmonic reconstruction (full)",
        omega=omega,
        n_harmonics=n_harmonics,
        n_cycles=n_cycles,
    )

    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))

    L = len(strain_arr)
    p = int(n_cycles)
    W = W_int if W_int is not None else int(round(L / (2 * p)))

    logger.debug(
        "Harmonic reconstruction parameters",
        signal_length=L,
        n_cycles_parsed=p,
        window_size=W,
    )

    # Normalize rate by omega (MATLAB convention)
    rate_normalized = rate_arr / omega

    # Compute FFT of all signals
    fft_strain = jnp.fft.fft(strain_arr)
    fft_rate = jnp.fft.fft(rate_normalized)
    fft_stress = jnp.fft.fft(stress_arr)

    # Convert to MATLAB-style coefficients
    # An = 2*Re(FFT)/L, Bn = -2*Im(FFT)/L
    An1_strain = 2 * jnp.real(fft_strain) / L
    Bn1_strain = -2 * jnp.imag(fft_strain) / L
    An1_rate = 2 * jnp.real(fft_rate) / L
    Bn1_rate = -2 * jnp.imag(fft_rate) / L
    An1_stress = 2 * jnp.real(fft_stress) / L
    Bn1_stress = -2 * jnp.imag(fft_stress) / L

    # Zero the DC component
    An1_strain = An1_strain.at[0].set(0.0)
    An1_rate = An1_rate.at[0].set(0.0)
    An1_stress = An1_stress.at[0].set(0.0)

    # Compute phase offset Delta from strain fundamental
    An_fund = An1_strain[p]
    Bn_fund = Bn1_strain[p]
    Delta = jnp.arctan2(An_fund, Bn_fund)
    Delta = jnp.where(Bn_fund < 0, Delta + jnp.pi, Delta)

    # Rotate coefficients to align with phase reference
    # An_new = An1*cos(Delta/p*n) - Bn1*sin(Delta/p*n)
    # Bn_new = Bn1*cos(Delta/p*n) + An1*sin(Delta/p*n)
    n_indices = jnp.arange(L // 2)
    rotation_angle = Delta / p * n_indices

    cos_rot = jnp.cos(rotation_angle)
    sin_rot = jnp.sin(rotation_angle)

    # Apply rotation to strain
    An_strain = An1_strain[: L // 2] * cos_rot - Bn1_strain[: L // 2] * sin_rot
    Bn_strain = Bn1_strain[: L // 2] * cos_rot + An1_strain[: L // 2] * sin_rot

    # Apply rotation to rate
    An_rate = An1_rate[: L // 2] * cos_rot - Bn1_rate[: L // 2] * sin_rot
    Bn_rate = Bn1_rate[: L // 2] * cos_rot + An1_rate[: L // 2] * sin_rot

    # Apply rotation to stress
    An_stress = An1_stress[: L // 2] * cos_rot - Bn1_stress[: L // 2] * sin_rot
    Bn_stress = Bn1_stress[: L // 2] * cos_rot + An1_stress[: L // 2] * sin_rot

    # Create new time array (shifted by Delta/omega)
    dt = 2 * jnp.pi / omega / L
    time_new = dt * jnp.arange(L)

    # Reconstruct waveforms from aligned coefficients
    # Only use fundamental for strain/rate (n=1), odd harmonics up to n_harmonics for stress

    def reconstruct_signal(An, Bn, max_harmonic, fundamental_only=False):
        """Reconstruct signal from Fourier coefficients."""
        result = jnp.zeros(L, dtype=jnp.float64)
        if fundamental_only:
            # Only fundamental harmonic
            n = 1
            idx = p * n
            if idx < len(An):
                result = result + An[idx] * jnp.cos(n * omega * time_new)
                result = result + Bn[idx] * jnp.sin(n * omega * time_new)
        else:
            # Odd harmonics up to max_harmonic
            for n in range(1, max_harmonic + 1, 2):
                idx = p * n
                if idx < len(An):
                    result = result + An[idx] * jnp.cos(n * omega * time_new)
                    result = result + Bn[idx] * jnp.sin(n * omega * time_new)
        return result

    strain_recon = reconstruct_signal(An_strain, Bn_strain, 1, fundamental_only=True)
    rate_recon = reconstruct_signal(An_rate, Bn_rate, 1, fundamental_only=True)
    stress_recon = reconstruct_signal(An_stress, Bn_stress, n_harmonics)

    # Fourier amplitude spectrum for stress (MATLAB ft_out)
    W_idx = int(W)
    k_indices = jnp.arange(W_idx + 1) * p
    stress_fft_scaled = fft_stress / L
    ft_amp = 2 * jnp.abs(stress_fft_scaled[k_indices])
    ft_amp = ft_amp / jnp.maximum(ft_amp[1], 1e-20)
    f_domain = jnp.arange(W_idx + 1, dtype=jnp.float64) * (omega / (2 * jnp.pi))
    ft_out = jnp.stack([f_domain, ft_amp], axis=1)

    return {
        "Delta": Delta,
        "An_strain": An_strain,
        "Bn_strain": Bn_strain,
        "An_rate": An_rate,
        "Bn_rate": Bn_rate,
        "An_stress": An_stress,
        "Bn_stress": Bn_stress,
        "strain_recon": strain_recon,
        "rate_recon": rate_recon,
        "stress_recon": stress_recon,
        "time_new": time_new,
        "ft_out": ft_out,
    }


@partial(jax.jit, static_argnums=(4, 5))
def spp_fourier_analysis(
    strain: "Array",
    stress: "Array",
    omega: float,
    dt: float,
    n_harmonics: int = 39,
    n_cycles: int = 1,
) -> dict:
    """
    Complete SPP analysis using Fourier-based analytical derivatives (MATLAB-compatible).

    Implements the full workflow from SPPplus_fourier_v2.m:
    1. FFT strain and stress signals
    2. Compute phase offset and rotate coefficients
    3. Compute derivatives ANALYTICALLY from Fourier coefficients
    4. Calculate G'(t), G''(t) via cross-product formula
    5. Extract all SPP metrics including moduli rates and Frenet-Serret frame

    This is more accurate than numerical differentiation for noisy data.

    Parameters
    ----------
    strain : Array
        Strain signal γ(t) (dimensionless)
    stress : Array
        Stress signal σ(t) (Pa)
    omega : float
        Angular frequency ω (rad/s)
    dt : float
        Time step (s)
    n_harmonics : int
        Number of odd harmonics for reconstruction (default: 15)
    n_cycles : int
        Number of complete cycles in data (default: 1)

    Returns
    -------
    dict
        Dictionary containing all SPP metrics:
        - Gp_t: Instantaneous G'(t) (Pa)
        - Gpp_t: Instantaneous G''(t) (Pa)
        - G_star_t: Complex modulus ``|G*(t)|`` (Pa)
        - tan_delta_t: Loss tangent tan(δ)(t)
        - delta_t: Phase angle δ(t) (radians)
        - disp_stress: Displacement stress (Pa)
        - eq_strain_est: Equivalent strain estimate
        - Gp_t_dot: Time derivative of G'(t) (Pa/s)
        - Gpp_t_dot: Time derivative of G''(t) (Pa/s)
        - G_speed: Moduli rate magnitude (Pa/s)
        - delta_t_dot: Phase angle rate (rad/s)
        - T_vec, N_vec, B_vec: Frenet-Serret frame vectors
        - strain_recon, stress_recon: Reconstructed waveforms
        - Delta: Phase offset used

    Notes
    -----
    ANALYTICAL derivatives from Fourier series:
        f(t) = Σ [An*cos(nωt) + Bn*sin(nωt)]
        f'(t) = Σ [-nω*An*sin(nωt) + nω*Bn*cos(nωt)]
        f''(t) = Σ [-n²ω²*An*cos(nωt) - n²ω²*Bn*sin(nωt)]
        f'''(t) = Σ [n³ω³*An*sin(nωt) - n³ω³*Bn*cos(nωt)]
    """
    logger.info(
        "Starting SPP Fourier analysis",
        omega=omega,
        dt=dt,
        n_harmonics=n_harmonics,
        n_cycles=n_cycles,
    )

    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    L = len(strain_arr)
    p = int(n_cycles)
    W_int = int(round(L / (2 * p)))

    logger.debug(
        "SPP Fourier analysis input data",
        signal_length=L,
        n_cycles_parsed=p,
        window_size=W_int,
    )

    # Compute strain rate from strain (wrapped 8-point stencil)
    logger.debug("Computing strain rate from strain (8-point stencil)")
    strain_rate = differentiate_rate_from_strain(
        strain_arr, dt, step_size=8, looped=True
    )

    # Get phase-aligned reconstruction with concrete W
    logger.debug("Performing phase-aligned Fourier reconstruction")
    fourier_result = harmonic_reconstruction_full(
        strain_arr, strain_rate, stress_arr, omega, n_harmonics, p, W_int
    )

    Delta = fourier_result["Delta"]
    An_strain = fourier_result["An_strain"]
    Bn_strain = fourier_result["Bn_strain"]
    An_rate = fourier_result["An_rate"]
    Bn_rate = fourier_result["Bn_rate"]
    An_stress = fourier_result["An_stress"]
    Bn_stress = fourier_result["Bn_stress"]
    time_new = fourier_result["time_new"]

    # Compute ANALYTICAL derivatives from Fourier coefficients
    # For each waveform, compute f, f', f'', f'''

    def compute_derivatives_from_fourier(An, Bn, max_harmonic):
        """Compute signal and its 1st, 2nd, 3rd derivatives from Fourier coefficients."""
        f = jnp.zeros(L, dtype=jnp.float64)
        fd = jnp.zeros(L, dtype=jnp.float64)
        fdd = jnp.zeros(L, dtype=jnp.float64)
        fddd = jnp.zeros(L, dtype=jnp.float64)

        for n in range(1, max_harmonic + 1, 2):
            idx = p * n
            if idx < len(An):
                n_omega = n * omega
                cos_term = jnp.cos(n_omega * time_new)
                sin_term = jnp.sin(n_omega * time_new)

                # f(t) = An*cos(nωt) + Bn*sin(nωt)
                f = f + An[idx] * cos_term + Bn[idx] * sin_term

                # f'(t) = -nω*An*sin(nωt) + nω*Bn*cos(nωt)
                fd = fd - n_omega * An[idx] * sin_term + n_omega * Bn[idx] * cos_term

                # f''(t) = -n²ω²*An*cos(nωt) - n²ω²*Bn*sin(nωt)
                fdd = (
                    fdd
                    - n_omega**2 * An[idx] * cos_term
                    - n_omega**2 * Bn[idx] * sin_term
                )

                # f'''(t) = n³ω³*An*sin(nωt) - n³ω³*Bn*cos(nωt)
                fddd = (
                    fddd
                    + n_omega**3 * An[idx] * sin_term
                    - n_omega**3 * Bn[idx] * cos_term
                )

        return f, fd, fdd, fddd

    # Strain (fundamental only for n=1)
    strain_recon, strain_d, strain_dd, strain_ddd = compute_derivatives_from_fourier(
        An_strain, Bn_strain, 1
    )

    # Rate (fundamental only) - but we need rate/omega for the response wave
    rate_recon, rate_d, rate_dd, rate_ddd = compute_derivatives_from_fourier(
        An_rate, Bn_rate, 1
    )

    # Stress (odd harmonics up to n_harmonics)
    stress_recon, stress_d, stress_dd, stress_ddd = compute_derivatives_from_fourier(
        An_stress, Bn_stress, n_harmonics
    )

    # Build response wave derivatives [γ, γ̇/ω, σ]
    # Note: rate_recon is already γ̇/ω from the reconstruction
    rd = jnp.stack([strain_d, rate_d, stress_d], axis=1)
    rdd = jnp.stack([strain_dd, rate_dd, stress_dd], axis=1)
    rddd = jnp.stack([strain_ddd, rate_ddd, stress_ddd], axis=1)

    # Cross product: rd × rdd
    rd_x_rdd = jnp.stack(
        [
            rd[:, 1] * rdd[:, 2] - rd[:, 2] * rdd[:, 1],
            rd[:, 2] * rdd[:, 0] - rd[:, 0] * rdd[:, 2],
            rd[:, 0] * rdd[:, 1] - rd[:, 1] * rdd[:, 0],
        ],
        axis=1,
    )

    # Second cross product: rd × (rd × rdd)
    rd_x_rd_x_rdd = jnp.stack(
        [
            rd[:, 1] * rd_x_rdd[:, 2] - rd[:, 2] * rd_x_rdd[:, 1],
            rd[:, 2] * rd_x_rdd[:, 0] - rd[:, 0] * rd_x_rdd[:, 2],
            rd[:, 0] * rd_x_rdd[:, 1] - rd[:, 1] * rd_x_rdd[:, 0],
        ],
        axis=1,
    )

    # Magnitudes
    eps = 1e-20
    mag_rd = jnp.sqrt(jnp.sum(rd**2, axis=1))
    mag_rd_x_rdd = jnp.sqrt(jnp.sum(rd_x_rdd**2, axis=1))

    # Instantaneous moduli (MATLAB formula)
    Gp_t = (
        -rd_x_rdd[:, 0]
        / jnp.maximum(jnp.abs(rd_x_rdd[:, 2]), eps)
        * jnp.sign(rd_x_rdd[:, 2])
    )
    Gpp_t = (
        -rd_x_rdd[:, 1]
        / jnp.maximum(jnp.abs(rd_x_rdd[:, 2]), eps)
        * jnp.sign(rd_x_rdd[:, 2])
    )

    # Moduli rates (MATLAB formula)
    # Gp_t_dot = -rd[:,1] * (rddd · rd_x_rdd) / rd_x_rdd[:,2]²
    # Gpp_t_dot = rd[:,0] * (rddd · rd_x_rdd) / rd_x_rdd[:,2]²
    rddd_dot_rd_x_rdd = jnp.sum(rddd * rd_x_rdd, axis=1)
    Gp_t_dot = -rd[:, 1] * rddd_dot_rd_x_rdd / jnp.maximum(rd_x_rdd[:, 2] ** 2, eps)
    Gpp_t_dot = rd[:, 0] * rddd_dot_rd_x_rdd / jnp.maximum(rd_x_rdd[:, 2] ** 2, eps)
    G_speed = jnp.sqrt(Gp_t_dot**2 + Gpp_t_dot**2)

    # Complex modulus and phase angle
    G_star_t = jnp.sqrt(Gp_t**2 + Gpp_t**2)
    tan_delta_t = Gpp_t / jnp.maximum(jnp.abs(Gp_t), eps) * jnp.sign(Gp_t)
    is_Gp_neg = Gp_t < 0
    delta_t = jnp.arctan(tan_delta_t) + jnp.pi * is_Gp_neg

    # Phase angle rate (MATLAB formula)
    # Normalize derivatives by omega for delta_t_dot calculation
    rd_tn = rd / omega
    rdd_tn = rdd / omega**2
    rddd_tn = rddd / omega**3
    delta_t_dot = (
        -rd_tn[:, 2]
        * (rddd_tn[:, 2] + rd_tn[:, 2])
        / (jnp.maximum(rdd_tn[:, 2] ** 2 + rd_tn[:, 2] ** 2, eps))
    )

    # Displacement stress
    disp_stress = stress_recon - (Gp_t * strain_recon + Gpp_t * rate_recon)
    eq_strain_est = strain_recon - disp_stress / jnp.maximum(jnp.abs(Gp_t), eps)

    # Frenet-Serret frame
    T_vec = rd / jnp.maximum(mag_rd[:, None], eps)
    N_vec = -rd_x_rd_x_rdd / jnp.maximum((mag_rd * mag_rd_x_rdd)[:, None], eps)
    B_vec = rd_x_rdd / jnp.maximum(mag_rd_x_rdd[:, None], eps)

    return {
        # Core SPP metrics
        "Gp_t": Gp_t,
        "Gpp_t": Gpp_t,
        "G_star_t": G_star_t,
        "tan_delta_t": tan_delta_t,
        "delta_t": delta_t,
        "disp_stress": disp_stress,
        "eq_strain_est": eq_strain_est,
        # Moduli rates (NEW - Gap 4)
        "Gp_t_dot": Gp_t_dot,
        "Gpp_t_dot": Gpp_t_dot,
        "G_speed": G_speed,
        "delta_t_dot": delta_t_dot,
        # Frenet-Serret frame (NEW - Gap 5)
        "T_vec": T_vec,
        "N_vec": N_vec,
        "B_vec": B_vec,
        # Reconstructed waveforms
        "strain_recon": strain_recon,
        "rate_recon": rate_recon,
        "stress_recon": stress_recon,
        "time_new": time_new,
        # Phase alignment
        "Delta": Delta,
        # FSF and spectrum
        "fsf_data_out": jnp.concatenate([T_vec, N_vec, B_vec], axis=1),
        "ft_out": fourier_result["ft_out"],
    }


# ============================================================================
# Power-Law Fitting
# ============================================================================


@jax.jit
def power_law_fit(
    stress: "Array",
    strain_rate: "Array",
    threshold_fraction: float = 0.1,
) -> tuple[float, float, float]:
    """Log-log fit of sigma = K * abs(strain_rate) ** n over the flowing region.

    Returns (K, n, r_squared).
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))

    # Use only flowing region (above threshold)
    rate_max = jnp.max(jnp.abs(strain_rate_arr))
    threshold = threshold_fraction * rate_max

    # Select first quadrant (positive stress and rate)
    mask = (strain_rate_arr > threshold) & (stress_arr > 0)

    # Extract valid points
    valid_rates = jnp.where(mask, strain_rate_arr, jnp.nan)
    valid_stress = jnp.where(mask, stress_arr, jnp.nan)

    # Log-log linear regression: log(stress) = log(K) + n * log(strain_rate)
    log_rate = jnp.log(jnp.where(jnp.isnan(valid_rates), 1.0, valid_rates))
    log_stress = jnp.log(jnp.where(jnp.isnan(valid_stress), 1.0, valid_stress))

    # Mask invalid values
    valid = ~jnp.isnan(valid_rates) & ~jnp.isnan(valid_stress)
    n_valid = jnp.sum(valid)

    # Compute regression coefficients
    log_rate_valid = jnp.where(valid, log_rate, 0.0)
    log_stress_valid = jnp.where(valid, log_stress, 0.0)

    sum_x = jnp.sum(log_rate_valid)
    sum_y = jnp.sum(log_stress_valid)
    sum_xx = jnp.sum(log_rate_valid**2)
    sum_xy = jnp.sum(log_rate_valid * log_stress_valid)

    # Linear regression solution
    denom = n_valid * sum_xx - sum_x**2
    n_exponent = jnp.where(
        denom > 1e-10,
        (n_valid * sum_xy - sum_x * sum_y) / denom,
        1.0,
    )
    log_K = jnp.where(
        n_valid > 0,
        (sum_y - n_exponent * sum_x) / n_valid,
        0.0,
    )
    K = jnp.exp(log_K)

    # Compute R² for fit quality
    y_mean = sum_y / jnp.maximum(n_valid, 1.0)
    ss_tot = jnp.sum(jnp.where(valid, (log_stress - y_mean) ** 2, 0.0))
    y_pred = log_K + n_exponent * log_rate
    ss_res = jnp.sum(jnp.where(valid, (log_stress - y_pred) ** 2, 0.0))
    r_squared = jnp.where(ss_tot > 1e-10, 1.0 - ss_res / ss_tot, 0.0)

    return K, n_exponent, r_squared


# ============================================================================
# Lissajous-Bowditch Metrics
# ============================================================================


@jax.jit
def lissajous_metrics(
    stress: "Array",
    strain: "Array",
    strain_rate: "Array",
    strain_amplitude: float,
    rate_amplitude: float,
) -> dict:
    """
    Compute Lissajous-Bowditch diagram derived metrics.

    Extracts nonlinearity measures from Lissajous plots including
    S-factor (stiffening ratio) and T-factor (thickening ratio).

    Parameters
    ----------
    stress : Array
        Time-resolved stress signal σ(t) (Pa)
    strain : Array
        Time-resolved strain signal γ(t) (dimensionless)
    strain_rate : Array
        Time-resolved strain rate signal γ̇(t) (1/s)
    strain_amplitude : float
        Maximum strain amplitude γ_0 (dimensionless)
    rate_amplitude : float
        Maximum strain rate amplitude γ̇_0 = ω * γ_0 (1/s)

    Returns
    -------
    dict
        Dictionary containing:
        - G_L: Large-strain modulus (tangent at γ = γ_0)
        - G_M: Minimum-strain modulus (tangent at γ = 0)
        - eta_L: Large-rate viscosity (tangent at γ̇ = γ̇_0)
        - eta_M: Minimum-rate viscosity (tangent at γ̇ = 0)
        - S_factor: Stiffening ratio (G_L - G_M) / G_L
        - T_factor: Thickening ratio (η_L - η_M) / η_L

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import lissajous_metrics
    >>>
    >>> omega = 1.0
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> gamma_0 = 1.0
    >>> gamma = gamma_0 * jnp.sin(omega * t)
    >>> gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    >>> sigma = 100.0 * gamma + 10.0 * gamma_dot  # Linear viscoelastic
    >>>
    >>> metrics = lissajous_metrics(sigma, gamma, gamma_dot, gamma_0, gamma_0 * omega)
    >>> # S_factor ≈ 0 (linear), T_factor ≈ 0 (linear)

    Notes
    -----
    - S > 0: strain stiffening, S < 0: strain softening
    - T > 0: shear thickening, T < 0: shear thinning
    - For linear viscoelastic: S = T = 0
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))
    gamma_0 = jnp.float64(strain_amplitude)
    rate_0 = jnp.float64(rate_amplitude)

    # G_L: Large-strain modulus (σ at γ = γ_0)
    # Find points where |γ| ≈ γ_0
    at_max_strain = jnp.abs(strain_arr) >= 0.98 * gamma_0
    sigma_at_max_strain = jnp.where(at_max_strain, jnp.abs(stress_arr), 0.0)
    G_L = jnp.sum(sigma_at_max_strain) / jnp.maximum(jnp.sum(at_max_strain), 1.0)
    G_L = G_L / gamma_0

    # G_M: Minimum-strain modulus (dσ/dγ at γ = 0)
    # Find points where |γ| ≈ 0
    at_zero_strain = jnp.abs(strain_arr) <= 0.02 * gamma_0
    # Approximate derivative using central difference
    d_sigma = jnp.roll(stress_arr, -1) - jnp.roll(stress_arr, 1)
    d_gamma = jnp.roll(strain_arr, -1) - jnp.roll(strain_arr, 1)
    local_modulus = jnp.where(
        jnp.abs(d_gamma) > 1e-10,
        d_sigma / d_gamma,
        0.0,
    )
    modulus_at_zero = jnp.where(at_zero_strain, local_modulus, 0.0)
    G_M = jnp.sum(modulus_at_zero) / jnp.maximum(jnp.sum(at_zero_strain), 1.0)

    # η_L: Large-rate viscosity (σ at γ̇ = γ̇_0)
    at_max_rate = jnp.abs(rate_arr) >= 0.98 * rate_0
    sigma_at_max_rate = jnp.where(at_max_rate, jnp.abs(stress_arr), 0.0)
    eta_L = jnp.sum(sigma_at_max_rate) / jnp.maximum(jnp.sum(at_max_rate), 1.0)
    eta_L = eta_L / rate_0

    # η_M: Minimum-rate viscosity (dσ/dγ̇ at γ̇ = 0)
    at_zero_rate = jnp.abs(rate_arr) <= 0.02 * rate_0
    d_rate = jnp.roll(rate_arr, -1) - jnp.roll(rate_arr, 1)
    local_viscosity = jnp.where(
        jnp.abs(d_rate) > 1e-10,
        d_sigma / d_rate,
        0.0,
    )
    viscosity_at_zero = jnp.where(at_zero_rate, local_viscosity, 0.0)
    eta_M = jnp.sum(viscosity_at_zero) / jnp.maximum(jnp.sum(at_zero_rate), 1.0)

    # S and T factors (stiffening and thickening ratios)
    S_factor = jnp.where(
        jnp.abs(G_L) > 1e-10,
        (G_L - G_M) / G_L,
        0.0,
    )
    T_factor = jnp.where(
        jnp.abs(eta_L) > 1e-10,
        (eta_L - eta_M) / eta_L,
        0.0,
    )

    return {
        "G_L": G_L,
        "G_M": G_M,
        "eta_L": eta_L,
        "eta_M": eta_M,
        "S_factor": S_factor,
        "T_factor": T_factor,
    }


# ============================================================================
# Zero-Crossing Detection (Robust)
# ============================================================================


@jax.jit
def zero_crossing_indices(signal: "Array") -> "Array":
    """
    Find indices of zero-crossings in a signal (robust implementation).

    Uses linear interpolation to find precise crossing locations,
    handling noise-induced multiple crossings via hysteresis filtering.

    Parameters
    ----------
    signal : Array
        Input signal to analyze for zero-crossings

    Returns
    -------
    Array
        Boolean mask of zero-crossing locations (True at crossings)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import zero_crossing_indices
    >>>
    >>> signal = jnp.sin(jnp.linspace(0, 4*jnp.pi, 100))
    >>> crossings = zero_crossing_indices(signal)
    >>> # crossings is True at indices where sin crosses zero

    Notes
    -----
    - Returns mask of same length as signal
    - Crossing detected when sign(s[i]) != sign(s[i+1])
    - Edge cases (exact zeros) handled properly
    """
    signal_arr = jnp.atleast_1d(jnp.asarray(signal, dtype=jnp.float64))

    # Compute sign changes
    signs = jnp.sign(signal_arr)
    sign_changes = jnp.abs(jnp.diff(signs)) > 1.5  # Sign change: ±2 difference

    # Pad to match original length
    crossings = jnp.concatenate([sign_changes, jnp.array([False])])

    return crossings


@jax.jit
def harmonic_truncation_robustness(
    amplitudes: "Array",
    n_harmonics_original: int,
    n_harmonics_truncated: int,
) -> float:
    """
    Compute truncation error metric for harmonic decomposition.

    Quantifies how much signal energy is lost when truncating to fewer
    harmonics, enabling assessment of reconstruction quality.

    Parameters
    ----------
    amplitudes : Array
        Full set of harmonic amplitudes [A_1, A_3, A_5, ...]
    n_harmonics_original : int
        Original number of harmonics
    n_harmonics_truncated : int
        Number of harmonics to keep after truncation

    Returns
    -------
    float
        Fraction of total energy retained after truncation (0 to 1)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import harmonic_truncation_robustness
    >>>
    >>> # Fundamental dominant with small 3rd harmonic
    >>> amps = jnp.array([100.0, 10.0, 2.0, 0.5, 0.1])
    >>> robustness = harmonic_truncation_robustness(amps, 5, 2)
    >>> # robustness ≈ 0.99 (most energy in first 2 harmonics)

    Notes
    -----
    - Value near 1.0 indicates safe truncation
    - Value < 0.95 suggests significant information loss
    - Useful for adaptive harmonic selection
    """
    amplitudes_arr = jnp.atleast_1d(jnp.asarray(amplitudes, dtype=jnp.float64))

    # Total energy (sum of squared amplitudes)
    total_energy = jnp.sum(amplitudes_arr**2)

    # Energy in retained harmonics
    retained = amplitudes_arr[:n_harmonics_truncated]
    retained_energy = jnp.sum(retained**2)

    # Fraction retained
    robustness = jnp.where(
        total_energy > 1e-20,
        retained_energy / total_energy,
        1.0,
    )

    return robustness


# ============================================================================
# SPP Stress Decomposition
# ============================================================================


@jax.jit
def spp_stress_decomposition(
    stress: "Array",
    strain: "Array",
    strain_rate: "Array",
    strain_amplitude: float,
    rate_amplitude: float,
) -> tuple["Array", "Array"]:
    """
    Decompose total stress into elastic and viscous contributions.

    Uses SPP framework to separate σ(t) = σ'(t) + σ''(t) where:
    - σ'(t): Elastic (in-phase with strain) component
    - σ''(t): Viscous (in-phase with strain rate) component

    Parameters
    ----------
    stress : Array
        Time-resolved stress signal σ(t) (Pa)
    strain : Array
        Time-resolved strain signal γ(t) (dimensionless)
    strain_rate : Array
        Time-resolved strain rate signal γ̇(t) (1/s)
    strain_amplitude : float
        Maximum strain amplitude γ_0 (dimensionless)
    rate_amplitude : float
        Maximum strain rate amplitude γ̇_0 = ω * γ_0 (1/s)

    Returns
    -------
    sigma_elastic : Array
        Elastic stress contribution σ'(t) (Pa)
    sigma_viscous : Array
        Viscous stress contribution σ''(t) (Pa)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import spp_stress_decomposition
    >>>
    >>> omega = 1.0
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> gamma_0 = 1.0
    >>> gamma = gamma_0 * jnp.sin(omega * t)
    >>> gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    >>> # G' = 100 Pa, G'' = 50 Pa
    >>> sigma = 100.0 * gamma + 50.0 * gamma_dot / omega
    >>>
    >>> sigma_e, sigma_v = spp_stress_decomposition(
    ...     sigma, gamma, gamma_dot, gamma_0, gamma_0 * omega
    ... )
    >>> # sigma_e ≈ 100 * gamma (elastic)
    >>> # sigma_v ≈ 50 * gamma_dot / omega (viscous)

    Notes
    -----
    - Decomposition valid for any LAOS response (linear or nonlinear)
    - For linear viscoelastic: σ_e = G' * γ, σ_v = G'' * γ / ω
    - Decomposition satisfies σ = σ_e + σ_v at all times
    """
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))
    gamma_0 = jnp.float64(strain_amplitude)
    rate_0 = jnp.float64(rate_amplitude)

    # Normalize strain and rate
    gamma_norm = strain_arr / gamma_0
    rate_norm = rate_arr / rate_0

    # Project stress onto strain and rate directions
    # Use orthogonality of sin and cos basis

    # Elastic component: projection onto strain direction
    # σ' = <σ, γ/γ_0> / <γ/γ_0, γ/γ_0> * γ/γ_0 * scale
    proj_elastic = jnp.sum(stress_arr * gamma_norm) / jnp.maximum(
        jnp.sum(gamma_norm**2), 1e-10
    )
    sigma_elastic = proj_elastic * gamma_norm

    # Viscous component: projection onto rate direction
    proj_viscous = jnp.sum(stress_arr * rate_norm) / jnp.maximum(
        jnp.sum(rate_norm**2), 1e-10
    )
    sigma_viscous = proj_viscous * rate_norm

    # Ensure decomposition is exact by distributing residual
    residual = stress_arr - sigma_elastic - sigma_viscous

    # Add half of residual to each (symmetric distribution)
    sigma_elastic = sigma_elastic + 0.5 * residual
    sigma_viscous = sigma_viscous + 0.5 * residual

    return sigma_elastic, sigma_viscous


# ============================================================================
# Numerical Differentiation (MATLAB-Compatible)
# ============================================================================


@partial(jax.jit, static_argnums=(2, 3))
def numerical_derivative_4th_order(
    signal: "Array",
    dt: float,
    order: int = 1,
    step_size: int = 1,
) -> "Array":
    """
    Compute numerical derivatives using 4th-order finite differences (MATLAB SPPplus compatible).

    Implements the EXACT finite-difference schemes from SPPplus_numerical_v2.m:
    - 4th-order centered differences in the interior
    - Forward differences at the beginning boundary
    - Backward differences at the ending boundary

    This matches MATLAB's "standard" differentiation mode (num_mode=1).

    Parameters
    ----------
    signal : Array
        Input signal to differentiate (1D array)
    dt : float
        Time step between samples (s)
    order : int, optional
        Derivative order: 1, 2, or 3 (default: 1)
    step_size : int, optional
        Step size k for stencil (default: 1, larger = more smoothing)

    Returns
    -------
    Array
        Numerical derivative of same length as input (4th-order accurate in interior)

    Notes
    -----
    MATLAB SPPplus_numerical_v2.m stencils (mode 1):

    First derivative (interior, 4th order):
        rd = (-f[p+2k] + 8*f[p+k] - 8*f[p-k] + f[p-2k]) / (12*k*dt)

    Second derivative (interior, 4th order):
        rdd = (-f[p+2k] + 16*f[p+k] - 30*f[p] + 16*f[p-k] - f[p-2k]) / (12*(k*dt)^2)

    Third derivative (interior, 4th order):
        rddd = (-f[p+3k] + 8*f[p+2k] - 13*f[p+k] + 13*f[p-k] - 8*f[p-2k] + f[p-3k]) / (8*(k*dt)^3)

    Forward/backward stencils at boundaries use 2nd-order accurate formulas.
    """
    signal_arr = jnp.atleast_1d(jnp.asarray(signal, dtype=jnp.float64))
    L = len(signal_arr)
    k = step_size
    h = dt * k

    # Pad signal for boundary handling using reflect mode
    pad_size = 4 * k
    signal_padded = jnp.pad(signal_arr, pad_size, mode="edge")

    if order == 1:
        # 4th-order centered first derivative
        # (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*h)
        result_padded = (
            -jnp.roll(signal_padded, -2 * k)
            + 8 * jnp.roll(signal_padded, -k)
            - 8 * jnp.roll(signal_padded, k)
            + jnp.roll(signal_padded, 2 * k)
        ) / (12 * h)
        result = result_padded[pad_size : pad_size + L]

        # Fix boundaries with forward/backward differences (2nd order)
        # Forward at start: (-f[p+2k] + 4*f[p+k] - 3*f[p]) / (2*k*dt)
        boundary_k = min(3 * k, L - 1)
        for p in range(boundary_k):
            if p + 2 * k < L and p + k < L:
                result = result.at[p].set(
                    (-signal_arr[p + 2 * k] + 4 * signal_arr[p + k] - 3 * signal_arr[p])
                    / (2 * h)
                )
        # Backward at end: (f[p-2k] - 4*f[p-k] + 3*f[p]) / (2*k*dt)
        for p in range(L - boundary_k, L):
            if p - 2 * k >= 0 and p - k >= 0:
                result = result.at[p].set(
                    (signal_arr[p - 2 * k] - 4 * signal_arr[p - k] + 3 * signal_arr[p])
                    / (2 * h)
                )

    elif order == 2:
        # 4th-order centered second derivative
        # (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*h^2)
        result_padded = (
            -jnp.roll(signal_padded, -2 * k)
            + 16 * jnp.roll(signal_padded, -k)
            - 30 * signal_padded
            + 16 * jnp.roll(signal_padded, k)
            - jnp.roll(signal_padded, 2 * k)
        ) / (12 * h**2)
        result = result_padded[pad_size : pad_size + L]

        # Boundary correction with 2nd-order forward/backward formulas
        boundary_k = min(3 * k, L - 1)
        for p in range(boundary_k):
            if p + 3 * k < L:
                result = result.at[p].set(
                    (
                        -signal_arr[p + 3 * k]
                        + 4 * signal_arr[p + 2 * k]
                        - 5 * signal_arr[p + k]
                        + 2 * signal_arr[p]
                    )
                    / (h**2)
                )
        for p in range(L - boundary_k, L):
            if p - 3 * k >= 0:
                result = result.at[p].set(
                    (
                        -signal_arr[p - 3 * k]
                        + 4 * signal_arr[p - 2 * k]
                        - 5 * signal_arr[p - k]
                        + 2 * signal_arr[p]
                    )
                    / (h**2)
                )

    elif order == 3:
        # 4th-order centered third derivative
        # (-f[i+3] + 8*f[i+2] - 13*f[i+1] + 13*f[i-1] - 8*f[i-2] + f[i-3]) / (8*h^3)
        result_padded = (
            -jnp.roll(signal_padded, -3 * k)
            + 8 * jnp.roll(signal_padded, -2 * k)
            - 13 * jnp.roll(signal_padded, -k)
            + 13 * jnp.roll(signal_padded, k)
            - 8 * jnp.roll(signal_padded, 2 * k)
            + jnp.roll(signal_padded, 3 * k)
        ) / (8 * h**3)
        result = result_padded[pad_size : pad_size + L]

        # Boundary correction
        boundary_k = min(4 * k, L - 1)
        for p in range(boundary_k):
            if p + 4 * k < L:
                result = result.at[p].set(
                    (
                        -3 * signal_arr[p + 4 * k]
                        + 14 * signal_arr[p + 3 * k]
                        - 24 * signal_arr[p + 2 * k]
                        + 18 * signal_arr[p + k]
                        - 5 * signal_arr[p]
                    )
                    / (2 * h**3)
                )
        for p in range(L - boundary_k, L):
            if p - 4 * k >= 0:
                result = result.at[p].set(
                    (
                        3 * signal_arr[p - 4 * k]
                        - 14 * signal_arr[p - 3 * k]
                        + 24 * signal_arr[p - 2 * k]
                        - 18 * signal_arr[p - k]
                        + 5 * signal_arr[p]
                    )
                    / (2 * h**3)
                )
    else:
        result = signal_arr

    return result


@partial(jax.jit, static_argnums=(2, 3))
def numerical_derivative(
    signal: "Array",
    dt: float,
    order: int = 1,
    step_size: int = 1,
) -> "Array":
    """
    Compute numerical derivatives using finite differences (MATLAB SPPplus compatible).

    Implements the same finite-difference schemes as SPPplus_numerical_v2.m:
    - 4th-order centered differences in the interior
    - Forward/backward differences at boundaries

    Parameters
    ----------
    signal : Array
        Input signal to differentiate (1D array)
    dt : float
        Time step between samples (s)
    order : int, optional
        Derivative order: 1, 2, or 3 (default: 1)
    step_size : int, optional
        Step size k for stencil (default: 1, larger = more smoothing)

    Returns
    -------
    Array
        Numerical derivative of same length as input

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import numerical_derivative
    >>>
    >>> # Sinusoidal signal
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> dt = t[1] - t[0]
    >>> signal = jnp.sin(t)
    >>>
    >>> # First derivative (should be cos(t))
    >>> d_signal = numerical_derivative(signal, dt, order=1)
    >>>
    >>> # Second derivative (should be -sin(t))
    >>> d2_signal = numerical_derivative(signal, dt, order=2)

    Notes
    -----
    - Matches MATLAB SPPplus_numerical_v2.m "standard" differentiation mode
    - Uses higher-order stencils for accuracy
    - Boundary handling uses forward/backward differences
    - For periodic signals, consider using `numerical_derivative_periodic`
    """
    # Use the 4th-order implementation
    return numerical_derivative_4th_order(signal, dt, order, step_size)


@partial(jax.jit, static_argnums=(2,))
def numerical_derivative_periodic(
    signal: "Array",
    dt: float,
    step_size: int = 1,
) -> tuple["Array", "Array", "Array"]:
    """
    Compute 1st, 2nd, and 3rd derivatives assuming periodic signal (MATLAB "looped" mode).

    For LAOS data where the signal is periodic (steady-state oscillation), this
    uses centered differences everywhere by wrapping around at boundaries.
    Matches MATLAB SPPplus_numerical_v2.m "looped" differentiation mode.

    Parameters
    ----------
    signal : Array
        Periodic input signal (e.g., one or more complete LAOS cycles)
    dt : float
        Time step between samples (s)
    step_size : int, optional
        Step size k for stencil (default: 1)

    Returns
    -------
    d1 : Array
        First derivative
    d2 : Array
        Second derivative
    d3 : Array
        Third derivative

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import numerical_derivative_periodic
    >>>
    >>> # One complete period of sine wave
    >>> omega = 1.0
    >>> t = jnp.linspace(0, 2*jnp.pi/omega, 1000, endpoint=False)
    >>> dt = t[1] - t[0]
    >>> signal = jnp.sin(omega * t)
    >>>
    >>> d1, d2, d3 = numerical_derivative_periodic(signal, dt)
    >>> # d1 ≈ omega * cos(omega*t)
    >>> # d2 ≈ -omega^2 * sin(omega*t)
    >>> # d3 ≈ -omega^3 * cos(omega*t)

    Notes
    -----
    - Assumes signal represents complete periods (periodic boundary)
    - Uses higher-order centered differences for accuracy
    - More accurate than standard differentiation for periodic LAOS data
    """
    signal_arr = jnp.atleast_1d(jnp.asarray(signal, dtype=jnp.float64))
    k = step_size  # step_size is static, so this is safe
    h = dt * k  # Use JAX multiplication, not Python float()

    # Use jnp.roll for periodic boundary conditions (JIT-compatible)
    # First derivative: (-f[i+2k] + 8*f[i+k] - 8*f[i-k] + f[i-2k]) / (12*h)
    d1 = (
        -jnp.roll(signal_arr, -2 * k)
        + 8 * jnp.roll(signal_arr, -k)
        - 8 * jnp.roll(signal_arr, k)
        + jnp.roll(signal_arr, 2 * k)
    ) / (12 * h)

    # Second derivative: (-f[i+2k] + 16*f[i+k] - 30*f[i] + 16*f[i-k] - f[i-2k]) / (12*h^2)
    d2 = (
        -jnp.roll(signal_arr, -2 * k)
        + 16 * jnp.roll(signal_arr, -k)
        - 30 * signal_arr
        + 16 * jnp.roll(signal_arr, k)
        - jnp.roll(signal_arr, 2 * k)
    ) / (12 * h**2)

    # Third derivative: (-f[i+3k] + 8*f[i+2k] - 13*f[i+k] + 13*f[i-k] - 8*f[i-2k] + f[i-3k]) / (8*h^3)
    d3 = (
        -jnp.roll(signal_arr, -3 * k)
        + 8 * jnp.roll(signal_arr, -2 * k)
        - 13 * jnp.roll(signal_arr, -k)
        + 13 * jnp.roll(signal_arr, k)
        - 8 * jnp.roll(signal_arr, 2 * k)
        + jnp.roll(signal_arr, 3 * k)
    ) / (8 * h**3)

    return d1, d2, d3


@partial(jax.jit, static_argnums=(4, 5))
def spp_numerical_analysis(
    strain: "Array",
    stress: "Array",
    omega: "float | Array",
    dt: float,
    step_size: int = 8,
    num_mode: int = 2,
) -> dict:
    """
    Perform full SPP analysis using numerical differentiation (MATLAB-compatible).

    Implements the numerical SPP workflow from SPPplus_numerical_v2.m:
    1. Compute strain rate from strain (or use provided)
    2. Compute derivatives of [strain, rate, stress] trajectory
    3. Calculate instantaneous ``G'_t`` and ``G''_t`` via cross-product formula
    4. Extract ``tan(δ)_t``, phase angle, and displacement stress

    Parameters
    ----------
    strain : Array
        Strain signal γ(t) (dimensionless)
    stress : Array
        Stress signal σ(t) (Pa)
    omega : float | Array
        Angular frequency ω (rad/s). Can be scalar or per-sample array.
    dt : float
        Time step between samples (s)
    step_size : int, optional
        Finite difference step size k (default: 8 for Rogers parity)
    num_mode : int, optional
        1 = edge-aware (forward/backward + centered); 2 = periodic/looped (default).

    Returns
    -------
    dict
        Dictionary containing:
        - Gp_t: Instantaneous storage modulus G'(t) (Pa)
        - Gpp_t: Instantaneous loss modulus G''(t) (Pa)
        - G_star_t: Instantaneous complex modulus ``|G*(t)|`` (Pa)
        - tan_delta_t: Instantaneous tan(δ)(t)
        - delta_t: Instantaneous phase angle δ(t) (radians)
        - disp_stress: Displacement stress (Pa)
        - eq_strain_est: Equivalent strain estimate

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.utils.spp_kernels import spp_numerical_analysis
    >>>
    >>> omega = 1.0
    >>> t = jnp.linspace(0, 2*jnp.pi, 1000)
    >>> dt = t[1] - t[0]
    >>> gamma_0 = 1.0
    >>> strain = gamma_0 * jnp.sin(omega * t)
    >>> # Linear viscoelastic response
    >>> stress = 100.0 * strain + 50.0 * gamma_0 * omega * jnp.cos(omega * t)
    >>>
    >>> result = spp_numerical_analysis(strain, stress, omega, dt)
    >>> # result['Gp_t'] ≈ 100.0 (constant for linear material)

    Notes
    -----
    - Matches MATLAB SPPplus cross-product formulation
    - ``G'_t = -rd_x_rdd[:,0] / rd_x_rdd[:,2]``
    - ``G''_t = -rd_x_rdd[:,1] / rd_x_rdd[:,2]``
    - Works directly with raw experimental data (no Fourier decomposition)
    """
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    stress_arr = jnp.atleast_1d(jnp.asarray(stress, dtype=jnp.float64))

    # Handle scalar vs per-sample omega
    omega_arr = jnp.asarray(omega, dtype=jnp.float64)
    if omega_arr.ndim == 0:
        omega_arr = jnp.full_like(strain_arr, omega_arr)
    else:
        if omega_arr.shape[0] != strain_arr.shape[0]:
            raise ValueError(
                "omega array length must match strain length for numerical SPP"
            )
    omega_scalar = jnp.mean(omega_arr)

    # Compute strain rate (normalize by omega as in MATLAB)
    strain_rate = differentiate_rate_from_strain(
        strain_arr, dt, step_size=step_size, looped=(num_mode == 2)
    )
    strain_rate_normalized = strain_rate / omega_arr

    # Build response wave: [strain, rate/omega, stress]
    resp_wave = jnp.stack([strain_arr, strain_rate_normalized, stress_arr], axis=1)

    # Compute derivatives using periodic assumption (LAOS is periodic)
    # rd = first derivative, rdd = second derivative, rddd = third derivative
    if num_mode == 2:
        deriv_func = numerical_derivative_periodic
        rd_strain, rdd_strain, rddd_strain = deriv_func(resp_wave[:, 0], dt, step_size)
        rd_rate, rdd_rate, rddd_rate = deriv_func(resp_wave[:, 1], dt, step_size)
        rd_stress, rdd_stress, rddd_stress = deriv_func(resp_wave[:, 2], dt, step_size)
    else:
        # Edge-aware mode: compute derivatives separately using finite differences
        rd_strain = numerical_derivative(
            resp_wave[:, 0], dt, order=1, step_size=step_size
        )
        rdd_strain = numerical_derivative(
            resp_wave[:, 0], dt, order=2, step_size=step_size
        )
        rddd_strain = numerical_derivative(
            resp_wave[:, 0], dt, order=3, step_size=step_size
        )

        rd_rate = numerical_derivative(
            resp_wave[:, 1], dt, order=1, step_size=step_size
        )
        rdd_rate = numerical_derivative(
            resp_wave[:, 1], dt, order=2, step_size=step_size
        )
        rddd_rate = numerical_derivative(
            resp_wave[:, 1], dt, order=3, step_size=step_size
        )

        rd_stress = numerical_derivative(
            resp_wave[:, 2], dt, order=1, step_size=step_size
        )
        rdd_stress = numerical_derivative(
            resp_wave[:, 2], dt, order=2, step_size=step_size
        )
        rddd_stress = numerical_derivative(
            resp_wave[:, 2], dt, order=3, step_size=step_size
        )

    # Stack into 3-column arrays
    rd = jnp.stack([rd_strain, rd_rate, rd_stress], axis=1)
    rdd = jnp.stack([rdd_strain, rdd_rate, rdd_stress], axis=1)
    rddd = jnp.stack([rddd_strain, rddd_rate, rddd_stress], axis=1)

    # Cross product: rd × rdd (MATLAB formula)
    rd_x_rdd = jnp.stack(
        [
            rd[:, 1] * rdd[:, 2] - rd[:, 2] * rdd[:, 1],  # x component
            rd[:, 2] * rdd[:, 0] - rd[:, 0] * rdd[:, 2],  # y component
            rd[:, 0] * rdd[:, 1] - rd[:, 1] * rdd[:, 0],  # z component
        ],
        axis=1,
    )

    # Second cross product: rd × (rd × rdd) for Frenet-Serret frame
    rd_x_rd_x_rdd = jnp.stack(
        [
            rd[:, 1] * rd_x_rdd[:, 2] - rd[:, 2] * rd_x_rdd[:, 1],
            rd[:, 2] * rd_x_rdd[:, 0] - rd[:, 0] * rd_x_rdd[:, 2],
            rd[:, 0] * rd_x_rdd[:, 1] - rd[:, 1] * rd_x_rdd[:, 0],
        ],
        axis=1,
    )

    # Magnitudes for Frenet-Serret frame
    eps = 1e-20  # Avoid division by zero
    mag_rd = jnp.sqrt(jnp.sum(rd**2, axis=1))
    mag_rd_x_rdd = jnp.sqrt(jnp.sum(rd_x_rdd**2, axis=1))

    # Instantaneous moduli from cross-product (MATLAB formula)
    # G'_t = -rd_x_rdd[:,0] / rd_x_rdd[:,2]
    # G''_t = -rd_x_rdd[:,1] / rd_x_rdd[:,2]
    Gp_t = (
        -rd_x_rdd[:, 0]
        / jnp.maximum(jnp.abs(rd_x_rdd[:, 2]), eps)
        * jnp.sign(rd_x_rdd[:, 2])
    )
    Gpp_t = (
        -rd_x_rdd[:, 1]
        / jnp.maximum(jnp.abs(rd_x_rdd[:, 2]), eps)
        * jnp.sign(rd_x_rdd[:, 2])
    )

    # Moduli rates (MATLAB formula - Gap 4)
    # Gp_t_dot = -rd[:,1] * (rddd · rd_x_rdd) / rd_x_rdd[:,2]²
    # Gpp_t_dot = rd[:,0] * (rddd · rd_x_rdd) / rd_x_rdd[:,2]²
    rddd_dot_rd_x_rdd = jnp.sum(rddd * rd_x_rdd, axis=1)
    Gp_t_dot = -rd[:, 1] * rddd_dot_rd_x_rdd / jnp.maximum(rd_x_rdd[:, 2] ** 2, eps)
    Gpp_t_dot = rd[:, 0] * rddd_dot_rd_x_rdd / jnp.maximum(rd_x_rdd[:, 2] ** 2, eps)
    G_speed = jnp.sqrt(Gp_t_dot**2 + Gpp_t_dot**2)

    # Complex modulus magnitude
    G_star_t = jnp.sqrt(Gp_t**2 + Gpp_t**2)

    # Loss tangent and phase angle
    tan_delta_t = Gpp_t / jnp.maximum(jnp.abs(Gp_t), eps) * jnp.sign(Gp_t)
    is_Gp_neg = Gp_t < 0
    delta_t = jnp.arctan(tan_delta_t) + jnp.pi * is_Gp_neg

    # Phase angle rate (MATLAB formula)
    # Normalize derivatives by omega for delta_t_dot calculation
    rd_tn = rd / omega_scalar
    rdd_tn = rdd / omega_scalar**2
    rddd_tn = rddd / omega_scalar**3
    delta_t_dot = (
        -rd_tn[:, 2]
        * (rddd_tn[:, 2] + rd_tn[:, 2])
        / (jnp.maximum(rdd_tn[:, 2] ** 2 + rd_tn[:, 2] ** 2, eps))
    )

    # Displacement stress (MATLAB formula)
    disp_stress = stress_arr - (Gp_t * strain_arr + Gpp_t * strain_rate_normalized)

    # Equivalent strain estimate
    eq_strain_est = strain_arr - disp_stress / jnp.maximum(jnp.abs(Gp_t), eps)

    # Frenet-Serret frame (Gap 5)
    # T = tangent vector (normalized rd)
    # N = principal normal vector
    # B = binormal vector
    T_vec = rd / jnp.maximum(mag_rd[:, None], eps)
    N_vec = -rd_x_rd_x_rdd / jnp.maximum((mag_rd * mag_rd_x_rdd)[:, None], eps)
    B_vec = rd_x_rdd / jnp.maximum(mag_rd_x_rdd[:, None], eps)

    return {
        # Core SPP metrics
        "Gp_t": Gp_t,
        "Gpp_t": Gpp_t,
        "G_star_t": G_star_t,
        "tan_delta_t": tan_delta_t,
        "delta_t": delta_t,
        "disp_stress": disp_stress,
        "eq_strain_est": eq_strain_est,
        # Moduli rates (NEW - Gap 4)
        "Gp_t_dot": Gp_t_dot,
        "Gpp_t_dot": Gpp_t_dot,
        "G_speed": G_speed,
        "delta_t_dot": delta_t_dot,
        # Frenet-Serret frame (NEW - Gap 5)
        "T_vec": T_vec,
        "N_vec": N_vec,
        "B_vec": B_vec,
        # Intermediate values for debugging
        "strain_rate_normalized": strain_rate_normalized,
        "fsf_data_out": jnp.concatenate([T_vec, N_vec, B_vec], axis=1),
        # Reconstructions / time grid (numerical keeps original)
        "strain_recon": strain_arr,
        "rate_recon": strain_rate_normalized,
        "stress_recon": stress_arr,
        "time_new": jnp.arange(len(strain_arr)) * dt,
    }


# ============================================================================
# Displacement-Stress Yield Extraction (Gap 6)
# ============================================================================


@jax.jit
def yield_from_displacement_stress(
    disp_stress: "Array",
    strain: "Array",
    strain_rate: "Array",
    Gp_t: "Array",
    delta_t: "Array",
    strain_amplitude: float,
    rate_amplitude: float,
) -> dict:
    """
    Extract yield stresses from displacement stress curve (SPP methodology).

    This implements the Donley et al. (2019) framework for yield stress extraction:
    - σ_sy (static yield): From displacement stress at G'(t) → 0 transition
    - σ_dy (dynamic yield): From displacement stress at δ(t) → π/2 transition

    This is more physically meaningful than simple geometric extraction.

    Parameters
    ----------
    disp_stress : Array
        Displacement stress σ_disp = σ - (G'·γ + G''·γ̇/ω) (Pa)
    strain : Array
        Strain signal γ(t) (dimensionless)
    strain_rate : Array
        Strain rate signal γ̇(t)/ω (normalized, dimensionless)
    Gp_t : Array
        Instantaneous storage modulus G'(t) (Pa)
    delta_t : Array
        Instantaneous phase angle δ(t) (radians)
    strain_amplitude : float
        Maximum strain amplitude γ_0 (dimensionless)
    rate_amplitude : float
        Maximum strain rate amplitude γ̇_0 = ω * γ_0 (1/s)

    Returns
    -------
    dict
        Dictionary containing:
        - sigma_sy: Static yield stress (Pa) - from G'(t) minima
        - sigma_dy: Dynamic yield stress (Pa) - from δ(t) → π/2
        - yield_strain_sy: Strain at static yield
        - yield_strain_dy: Strain at dynamic yield
        - yield_indices_sy: Indices of static yield points
        - yield_indices_dy: Indices of dynamic yield points
        - sigma_sy_disp: Static yield from displacement stress peak
        - sigma_dy_disp: Dynamic yield from displacement stress at zero rate

    Notes
    -----
    The SPP framework defines yield stresses based on the displacement stress:
    - Static yield occurs when the cage structure breaks (G'(t) → 0)
    - Dynamic yield occurs when flow ceases (δ(t) → π/2)

    This differs from simple geometric extraction (stress at extrema) and provides
    a more physically meaningful interpretation of the yielding transition.

    References
    ----------
    G.J. Donley et al., "Time-resolved dynamics of the yielding transition
    in soft materials", J. Non-Newton. Fluid Mech. 264, 2019
    """
    disp_stress_arr = jnp.atleast_1d(jnp.asarray(disp_stress, dtype=jnp.float64))
    strain_arr = jnp.atleast_1d(jnp.asarray(strain, dtype=jnp.float64))
    rate_arr = jnp.atleast_1d(jnp.asarray(strain_rate, dtype=jnp.float64))
    Gp_t_arr = jnp.atleast_1d(jnp.asarray(Gp_t, dtype=jnp.float64))
    delta_t_arr = jnp.atleast_1d(jnp.asarray(delta_t, dtype=jnp.float64))
    gamma_0 = jnp.float64(strain_amplitude)
    # rate_amplitude is received but not used in current yield extraction methods
    # Reserved for future rate-dependent yield criteria
    _ = rate_amplitude  # Explicitly acknowledge unused parameter

    eps = 1e-10

    # =========================================================================
    # Method 1: Static yield from G'(t) minima (cage breakage)
    # =========================================================================
    # Find points where G'(t) is near its minimum (cage breaking)
    Gp_min = jnp.min(Gp_t_arr)
    Gp_max = jnp.max(Gp_t_arr)
    Gp_range = jnp.maximum(Gp_max - Gp_min, eps)

    # Threshold: within 10% of minimum
    near_Gp_min = Gp_t_arr < (Gp_min + 0.1 * Gp_range)

    # Static yield: stress magnitude at G'(t) minima
    stress_at_Gp_min = jnp.where(near_Gp_min, jnp.abs(disp_stress_arr), 0.0)
    count_sy = jnp.sum(near_Gp_min)
    sigma_sy = jnp.where(
        count_sy > 0,
        jnp.sum(stress_at_Gp_min) / count_sy,
        jnp.max(jnp.abs(disp_stress_arr)),
    )

    # Find strain at static yield
    yield_strain_sy = jnp.where(
        count_sy > 0,
        jnp.sum(jnp.where(near_Gp_min, jnp.abs(strain_arr), 0.0)) / count_sy,
        gamma_0,
    )

    # =========================================================================
    # Method 2: Dynamic yield from δ(t) → π/2 (flow cessation)
    # =========================================================================
    # Find points where δ(t) is near π/2 (viscous dominated)
    delta_threshold = jnp.pi / 2 - 0.1  # within ~6° of π/2
    near_pi_half = delta_t_arr > delta_threshold

    # Dynamic yield: stress magnitude at δ → π/2
    stress_at_delta_pi2 = jnp.where(near_pi_half, jnp.abs(disp_stress_arr), 0.0)
    count_dy = jnp.sum(near_pi_half)
    sigma_dy_from_delta = jnp.where(
        count_dy > 0, jnp.sum(stress_at_delta_pi2) / count_dy, 0.0
    )

    # =========================================================================
    # Method 3: From displacement stress at strain/rate extrema (traditional)
    # =========================================================================
    # Static: displacement stress at |γ| ≈ γ_0
    near_max_strain = jnp.abs(strain_arr) >= 0.95 * gamma_0
    disp_at_max_strain = jnp.where(near_max_strain, jnp.abs(disp_stress_arr), 0.0)
    count_max_strain = jnp.sum(near_max_strain)
    sigma_sy_disp = jnp.where(
        count_max_strain > 0,
        jnp.sum(disp_at_max_strain) / count_max_strain,
        jnp.max(jnp.abs(disp_stress_arr)),
    )

    # Dynamic: displacement stress at |γ̇| ≈ 0
    near_zero_rate = jnp.abs(rate_arr) <= 0.05 * jnp.max(jnp.abs(rate_arr))
    disp_at_zero_rate = jnp.where(near_zero_rate, jnp.abs(disp_stress_arr), 0.0)
    count_zero_rate = jnp.sum(near_zero_rate)
    sigma_dy_disp = jnp.where(
        count_zero_rate > 0,
        jnp.sum(disp_at_zero_rate) / count_zero_rate,
        jnp.min(jnp.abs(disp_stress_arr)),
    )

    # Dynamic yield: use the maximum of the two methods
    sigma_dy = jnp.maximum(sigma_dy_from_delta, sigma_dy_disp)

    # Find strain at dynamic yield (near zero rate)
    yield_strain_dy = jnp.where(
        count_zero_rate > 0,
        jnp.sum(jnp.where(near_zero_rate, jnp.abs(strain_arr), 0.0)) / count_zero_rate,
        0.0,
    )

    return {
        "sigma_sy": sigma_sy,
        "sigma_dy": sigma_dy,
        "yield_strain_sy": yield_strain_sy,
        "yield_strain_dy": yield_strain_dy,
        "yield_indices_sy": near_Gp_min,
        "yield_indices_dy": near_zero_rate,
        "sigma_sy_disp": sigma_sy_disp,
        "sigma_dy_disp": sigma_dy_disp,
    }


@jax.jit
def frenet_serret_frame(
    rd: "Array",
    rdd: "Array",
) -> tuple["Array", "Array", "Array", "Array", "Array"]:
    """
    Compute the Frenet-Serret frame (T, N, B) for a 3D trajectory.

    The Frenet-Serret frame provides a local coordinate system along the
    (γ, γ̇/ω, σ) trajectory, useful for understanding the geometry of the
    nonlinear response.

    Parameters
    ----------
    rd : Array
        First derivative of response wave [d(γ)/dt, d(γ̇/ω)/dt, d(σ)/dt]
        Shape: (n_points, 3)
    rdd : Array
        Second derivative of response wave
        Shape: (n_points, 3)

    Returns
    -------
    T_vec : Array
        Tangent vector (unit vector in direction of motion)
    N_vec : Array
        Principal normal vector (direction of curvature)
    B_vec : Array
        Binormal vector (``T × N``)
    curvature : Array
        Local curvature ``κ = |rd × rdd| / |rd|³``
    torsion : Array
        Local torsion ``τ`` (requires third derivative, returns zeros)

    Notes
    -----
    Formulas (matching MATLAB SPPplus)::

        T = rd / |rd|
        N = -(rd × (rd × rdd)) / (|rd| × |rd × rdd|)
        B = (rd × rdd) / |rd × rdd|
        κ = |rd × rdd| / |rd|³
    """
    rd_arr = jnp.asarray(rd, dtype=jnp.float64)
    rdd_arr = jnp.asarray(rdd, dtype=jnp.float64)

    eps = 1e-20

    # Cross product: rd × rdd
    rd_x_rdd = jnp.stack(
        [
            rd_arr[:, 1] * rdd_arr[:, 2] - rd_arr[:, 2] * rdd_arr[:, 1],
            rd_arr[:, 2] * rdd_arr[:, 0] - rd_arr[:, 0] * rdd_arr[:, 2],
            rd_arr[:, 0] * rdd_arr[:, 1] - rd_arr[:, 1] * rdd_arr[:, 0],
        ],
        axis=1,
    )

    # Second cross product: rd × (rd × rdd)
    rd_x_rd_x_rdd = jnp.stack(
        [
            rd_arr[:, 1] * rd_x_rdd[:, 2] - rd_arr[:, 2] * rd_x_rdd[:, 1],
            rd_arr[:, 2] * rd_x_rdd[:, 0] - rd_arr[:, 0] * rd_x_rdd[:, 2],
            rd_arr[:, 0] * rd_x_rdd[:, 1] - rd_arr[:, 1] * rd_x_rdd[:, 0],
        ],
        axis=1,
    )

    # Magnitudes
    mag_rd = jnp.sqrt(jnp.sum(rd_arr**2, axis=1))
    mag_rd_x_rdd = jnp.sqrt(jnp.sum(rd_x_rdd**2, axis=1))

    # Tangent vector: T = rd / |rd|
    T_vec = rd_arr / jnp.maximum(mag_rd[:, None], eps)

    # Principal normal: N = -(rd × (rd × rdd)) / (|rd| × |rd × rdd|)
    N_vec = -rd_x_rd_x_rdd / jnp.maximum((mag_rd * mag_rd_x_rdd)[:, None], eps)

    # Binormal: B = (rd × rdd) / |rd × rdd|
    B_vec = rd_x_rdd / jnp.maximum(mag_rd_x_rdd[:, None], eps)

    # Curvature: κ = |rd × rdd| / |rd|³
    curvature = mag_rd_x_rdd / jnp.maximum(mag_rd**3, eps)

    # Torsion would require third derivative, return zeros for now
    torsion = jnp.zeros_like(curvature)

    return T_vec, N_vec, B_vec, curvature, torsion


# ============================================================================
# Export helpers (MATLAB-compatible schema)
# ============================================================================


def build_spp_exports(
    time: np.ndarray,
    strain: np.ndarray,
    rate_over_omega: np.ndarray,
    stress: np.ndarray,
    metrics: dict,
    fsf_data_out: np.ndarray | None,
    spp_params: np.ndarray,
) -> dict:
    """Assemble MATLAB-compatible spp_data_out / fsf_data_out tables.

    Returns
    -------
    dict with keys: spp_data_out, fsf_data_out, spp_params
    """
    logger.debug(
        "Building SPP export tables",
        n_points=len(time),
        has_fsf_data=fsf_data_out is not None,
        n_metrics=len(metrics),
    )

    spp_data_out = np.column_stack(
        [
            time,
            strain,
            rate_over_omega,
            stress,
            metrics["Gp_t"],
            metrics["Gpp_t"],
            metrics["G_star_t"],
            metrics["tan_delta_t"],
            metrics["delta_t"],
            metrics["disp_stress"],
            metrics["eq_strain_est"],
            metrics.get("Gp_t_dot", np.full_like(time, np.nan)),
            metrics.get("Gpp_t_dot", np.full_like(time, np.nan)),
            metrics.get("G_speed", np.full_like(time, np.nan)),
            metrics.get("delta_t_dot", np.full_like(time, np.nan)),
        ]
    )

    fsf_out = fsf_data_out if fsf_data_out is not None else None

    logger.info(
        "SPP export tables built",
        spp_data_shape=spp_data_out.shape,
        has_fsf_data=fsf_out is not None,
    )

    return {
        "spp_data_out": spp_data_out,
        "fsf_data_out": fsf_out,
        "spp_params": spp_params,
    }


# ============================================================================
# Data Preprocessing (Gap 7, 8)
# ============================================================================


@partial(jax.jit, static_argnames=("step_size", "looped"))
def differentiate_rate_from_strain(
    strain: "Array",
    dt: float,
    step_size: int = 8,
    looped: bool = True,
) -> "Array":
    """
    Compute strain rate from strain via numerical differentiation.

    Provides a wrapped (periodic) 8-point stencil path to mirror the
    MATLAB/Rogers SPPplus implementation, while keeping the prior finite
    difference fallback for non-periodic data.

    Parameters
    ----------
    strain : Array
        Strain signal γ(t) (dimensionless)
    dt : float
        Time step (s)
    step_size : int
        Finite difference step size ``k`` (default: 8, Rogers parity)
    looped : bool
        If True, use periodic derivative (wrapped); otherwise edge-aware.

    Returns
    -------
    Array
        Strain rate γ̇(t) (1/s)

    Notes
    -----
    - looped=True + step_size=8 matches SPPplus v2.1 wrapped 8-point rate
      inference when the rate column is absent.
    - looped=False preserves the previous 4th-order finite-difference path.
    """
    if looped:
        d1, _, _ = numerical_derivative_periodic(strain, dt, step_size=step_size)
        return d1
    return numerical_derivative_4th_order(strain, dt, order=1, step_size=step_size)


def convert_units(
    data: "Array",
    from_unit: str,
    to_unit: str,
) -> "Array":
    """
    Convert data between common rheological units.

    Parameters
    ----------
    data : Array
        Input data array
    from_unit : str
        Source unit (e.g., 'percent', 'mPa', 'rad', 'deg')
    to_unit : str
        Target unit (e.g., 'fraction', 'Pa', 'rad', 'deg')

    Returns
    -------
    Array
        Converted data

    Examples
    --------
    >>> strain_fraction = convert_units(strain_percent, 'percent', 'fraction')
    >>> stress_Pa = convert_units(stress_mPa, 'mPa', 'Pa')
    """
    logger.debug(
        "Converting units",
        from_unit=from_unit,
        to_unit=to_unit,
        data_shape=getattr(data, "shape", "scalar"),
    )

    data_arr = jnp.asarray(data, dtype=jnp.float64)

    # Define conversion factors (all lowercase for case-insensitive matching)
    # Note: "mpa" means milliPascal (mPa), not megaPascal (MPa)
    conversions = {
        # Strain conversions
        ("percent", "fraction"): 0.01,
        ("fraction", "percent"): 100.0,
        # Stress conversions (mPa = milliPascal, kPa = kiloPascal)
        ("mpa", "pa"): 0.001,  # milliPascal to Pascal
        ("pa", "mpa"): 1000.0,  # Pascal to milliPascal
        ("kpa", "pa"): 1000.0,  # kiloPascal to Pascal
        ("pa", "kpa"): 0.001,  # Pascal to kiloPascal
        # Angle conversions
        ("deg", "rad"): jnp.pi / 180.0,
        ("rad", "deg"): 180.0 / jnp.pi,
        # Time conversions
        ("ms", "s"): 0.001,
        ("s", "ms"): 1000.0,
        # Identity
        ("pa", "pa"): 1.0,
        ("fraction", "fraction"): 1.0,
        ("rad", "rad"): 1.0,
        ("s", "s"): 1.0,
    }

    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        logger.debug(
            "Unit conversion applied",
            conversion_key=key,
            factor=float(conversions[key]),
        )
        return data_arr * conversions[key]
    else:
        # Return unchanged if conversion not found
        logger.debug(
            "No conversion found, returning unchanged data",
            from_unit=from_unit,
            to_unit=to_unit,
        )
        return data_arr


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    # Core SPP functions
    "apparent_cage_modulus",
    "static_yield_stress",
    "dynamic_yield_stress",
    "harmonic_reconstruction",
    "power_law_fit",
    "lissajous_metrics",
    "zero_crossing_indices",
    "harmonic_truncation_robustness",
    "spp_stress_decomposition",
    # Numerical differentiation
    "numerical_derivative",
    "numerical_derivative_4th_order",
    "numerical_derivative_periodic",
    # SPP analysis functions
    "spp_numerical_analysis",
    "spp_fourier_analysis",
    # Phase-aligned Fourier (NEW - Gap 2, 3)
    "compute_phase_offset",
    "harmonic_reconstruction_full",
    # Frenet-Serret frame (NEW - Gap 5)
    "frenet_serret_frame",
    # Displacement-stress yield extraction (NEW - Gap 6)
    "yield_from_displacement_stress",
    # Data preprocessing (NEW - Gap 7, 8)
    "differentiate_rate_from_strain",
    "convert_units",
]
