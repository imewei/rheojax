"""Protocol-aware data preprocessing for rheological experiments.

Provides per-protocol diagnostics and preprocessing that can be applied
before fitting. Each protocol has specific data quality checks and
optional cleaning/conditioning steps.

Public standalone functions
---------------------------
check_kramers_kronig : KK consistency test for oscillation data.
estimate_eta0        : Zero-shear viscosity from Newtonian plateau.
fit_gel_point        : Winter-Chambon gel-point exponent from G(t).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rheojax.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PreprocessingResult:
    """Result from protocol preprocessing.

    Attributes:
        X: Preprocessed input array.
        y: Preprocessed target array.
        diagnostics: Protocol-specific diagnostic information.
        warnings: List of warning messages.
        applied: List of preprocessing steps applied.
    """

    X: np.ndarray
    y: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Standalone public functions
# ---------------------------------------------------------------------------


def check_kramers_kronig(
    omega: np.ndarray,
    G_prime: np.ndarray,
    G_double_prime: np.ndarray,
    tolerance: float = 2.5,
) -> tuple[bool, float]:
    """Check approximate Kramers-Kronig consistency.

    Uses the d(ln G')/d(ln ω) slope test. The KK relation requires
    that |d(ln G')/d(ln ω)| ≤ 2 everywhere (from the integral
    constraint relating G' and G''). A tolerance > 2 accounts for
    noise and finite data.

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency (rad/s).
    G_prime : np.ndarray
        Storage modulus (Pa).
    G_double_prime : np.ndarray
        Loss modulus (Pa).
    tolerance : float, optional
        Maximum allowed log-log slope. Default is 2.5.

    Returns
    -------
    passes : bool
        True if the data passes the KK test.
    max_slope : float
        Maximum absolute log-log slope of G' with respect to ω.
    """
    ln_omega = np.log(omega)
    ln_G_prime = np.log(np.maximum(G_prime, 1e-30))
    # Use forward differences (consistent with the original inline implementation)
    slope = np.abs(np.diff(ln_G_prime) / np.diff(ln_omega))
    max_slope = float(np.max(slope))
    return max_slope <= tolerance, max_slope


def estimate_eta0(
    gamma_dot: np.ndarray,
    eta: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> float:
    """Estimate zero-shear viscosity from the Newtonian plateau.

    Uses the lowest 10% of shear rates to estimate η₀ from the
    plateau region of η(γ̇).

    Parameters
    ----------
    gamma_dot : np.ndarray
        Shear rate array (s⁻¹).
    eta : np.ndarray or None, optional
        Viscosity array (Pa·s). If None, computed from ``sigma / gamma_dot``.
    sigma : np.ndarray or None, optional
        Shear stress array (Pa). Required when ``eta`` is None.

    Returns
    -------
    float
        Estimated η₀ (Pa·s).

    Raises
    ------
    ValueError
        If both ``eta`` and ``sigma`` are None, or if ``gamma_dot`` is empty.
    """
    gamma_dot = np.asarray(gamma_dot, dtype=np.float64)
    if len(gamma_dot) == 0:
        raise ValueError("gamma_dot must not be empty.")

    if eta is not None:
        eta_arr = np.asarray(eta, dtype=np.float64)
    elif sigma is not None:
        sigma_arr = np.asarray(sigma, dtype=np.float64)
        eta_arr = sigma_arr / np.maximum(np.abs(gamma_dot), 1e-30)
    else:
        raise ValueError("Either eta or sigma must be provided.")

    # Sort ascending by shear rate
    order = np.argsort(gamma_dot)
    gamma_sorted = gamma_dot[order]
    eta_sorted = eta_arr[order]

    # Take lowest 10%, at least 3 points
    n_low = max(3, int(np.ceil(0.1 * len(gamma_sorted))))
    n_low = min(n_low, len(gamma_sorted))

    return float(np.median(eta_sorted[:n_low]))


def fit_gel_point(
    t: np.ndarray,
    G_t: np.ndarray,
) -> tuple[float, float]:
    """Fit gel strength S and relaxation exponent n from G(t) = S · t^(−n).

    Uses log-log linear regression on G(t) data (Winter-Chambon criterion).

    Parameters
    ----------
    t : np.ndarray
        Time array (s). Must contain at least two positive values.
    G_t : np.ndarray
        Relaxation modulus array (Pa).

    Returns
    -------
    S : float
        Gel strength prefactor such that G(t) = S · t^(−n).
    n : float
        Relaxation exponent (positive value; typically 0 < n < 1).

    Raises
    ------
    ValueError
        If fewer than two valid (t > 0, G_t > 0) data points are available.
    """
    t = np.asarray(t, dtype=np.float64)
    G_t = np.asarray(G_t, dtype=np.float64)

    mask = (t > 0) & (G_t > 0)
    if int(np.sum(mask)) < 2:
        raise ValueError(
            "At least two data points with t > 0 and G_t > 0 are required "
            "for gel-point fitting."
        )

    ln_t = np.log(t[mask])
    ln_G = np.log(G_t[mask])

    # Linear regression: ln G = intercept + slope * ln t
    # where slope = -n and intercept = ln S
    slope, intercept = np.polyfit(ln_t, ln_G, 1)

    S = float(np.exp(intercept))
    n = float(-slope)
    return S, n


# ---------------------------------------------------------------------------
# Protocol-specific preprocessors
# ---------------------------------------------------------------------------


def preprocess_for_protocol(
    X: np.ndarray,
    y: np.ndarray,
    test_mode: str,
    **kwargs,
) -> PreprocessingResult:
    """Apply protocol-aware preprocessing and diagnostics.

    This function dispatches to protocol-specific preprocessing based on
    the test mode. It does not modify data unless explicitly requested —
    diagnostics are always computed but data modification is opt-in.

    Parameters
    ----------
    X : np.ndarray
        Input array (time, frequency, shear rate, etc.).
    y : np.ndarray
        Target array (modulus, stress, viscosity, etc.).
    test_mode : str
        Protocol string (``"relaxation"``, ``"oscillation"``, etc.).
    **kwargs
        Protocol-specific options.

    Returns
    -------
    PreprocessingResult
        Result with diagnostics and optionally cleaned data.
    """
    X = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)

    dispatchers: dict[str, Callable[..., PreprocessingResult]] = {
        "relaxation": _preprocess_relaxation,
        "creep": _preprocess_creep,
        "oscillation": _preprocess_oscillation,
        "flow_curve": _preprocess_flow_curve,
        "startup": _preprocess_startup,
        "laos": _preprocess_laos,
    }

    func = dispatchers.get(test_mode)
    if func is None:
        logger.debug("No preprocessing for test_mode", test_mode=test_mode)
        return PreprocessingResult(X=X, y=y_arr)

    return func(X, y_arr, **kwargs)


def _preprocess_relaxation(
    t: np.ndarray, G_t: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Relaxation: detect inertia ringing and truncation artifacts.

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    G_t : np.ndarray
        Relaxation modulus array (Pa).
    apply_cutoff : bool, optional
        If True, trim data to remove the ringing region detected at short
        times. Default is False (diagnostics only).

    Returns
    -------
    PreprocessingResult
    """
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}
    applied: list[str] = []
    apply_cutoff: bool = bool(kwargs.get("apply_cutoff", False))

    G_t_real = np.real(G_t) if np.iscomplexobj(G_t) else G_t

    # Detect inertia ringing (oscillations at short times)
    ringing_end_idx: int | None = None
    if len(t) > 10:
        n_probe = min(20, len(t))
        dG = np.diff(G_t_real[:n_probe])
        sign_changes = np.sum(np.diff(np.sign(dG)) != 0)
        diagnostics["short_time_sign_changes"] = int(sign_changes)

        if sign_changes > 3:
            # Estimate where ringing ends: last sign change in first n_probe region
            sign_diff = np.abs(np.diff(np.sign(dG)))
            change_positions = np.where(sign_diff != 0)[0]
            if len(change_positions) > 0:
                # +2 because dG[i] = G[i+1]-G[i], and sign_diff[i] = dG[i+1]-dG[i]
                ringing_end_idx = int(change_positions[-1]) + 2
                diagnostics["ringing_end_idx"] = ringing_end_idx

            warnings_list.append(
                f"Possible inertia ringing detected: {sign_changes} sign changes "
                "in first 20 points. Consider truncating early data."
            )

            if apply_cutoff and ringing_end_idx is not None:
                cutoff = min(ringing_end_idx, len(t) - 1)
                t = t[cutoff:]
                G_t_real = G_t_real[cutoff:]
                G_t = G_t[cutoff:] if not np.iscomplexobj(G_t) else G_t[cutoff:]
                applied.append(f"inertia_ringing_cutoff(idx={cutoff})")
                diagnostics["cutoff_applied_idx"] = cutoff

    # Detect plateau vs decay
    if len(t) > 20:
        late_ratio = G_t_real[-1] / (G_t_real[0] + 1e-30)
        diagnostics["late_to_early_ratio"] = float(late_ratio)
        if late_ratio > 0.5:
            warnings_list.append(
                f"G(t) decay is weak (G_final/G_initial = {late_ratio:.3f}). "
                "Data may not span enough decades for reliable fitting."
            )

    return PreprocessingResult(
        X=t,
        y=G_t,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=applied,
    )


def _preprocess_creep(t: np.ndarray, J_t: np.ndarray, **kwargs) -> PreprocessingResult:
    """Creep: detect plateau vs viscous flow regime."""
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    J_real = np.real(J_t) if np.iscomplexobj(J_t) else J_t

    # Check for viscous flow (linear J(t) at long times)
    if len(t) > 20:
        late_frac = max(1, len(t) // 5)
        t_late = t[-late_frac:]
        J_late = J_real[-late_frac:]
        if len(t_late) > 2:
            slope = np.polyfit(t_late, J_late, 1)[0]
            diagnostics["late_slope"] = float(slope)
            diagnostics["has_viscous_flow"] = bool(
                slope > 0 and slope * t_late[-1] > 0.5 * J_late[-1]
            )

    # Check monotonicity
    dJ = np.diff(J_real)
    n_decreasing = int(np.sum(dJ < 0))
    diagnostics["n_decreasing_steps"] = n_decreasing
    if n_decreasing > len(dJ) * 0.1:
        warnings_list.append(
            f"Non-monotonic creep compliance: {n_decreasing} decreasing steps. "
            "Check for noise or recovery effects."
        )

    return PreprocessingResult(
        X=t,
        y=J_t,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=[],
    )


def _preprocess_oscillation(
    omega: np.ndarray, G_star: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Oscillation: Kramers-Kronig consistency check.

    Delegates to the public :func:`check_kramers_kronig` function so that
    callers can also run the KK test standalone.
    """
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    if np.iscomplexobj(G_star):
        G_prime = G_star.real
        G_double_prime = G_star.imag
    elif G_star.ndim == 2 and G_star.shape[1] == 2:
        G_prime = G_star[:, 0]
        G_double_prime = G_star[:, 1]
    else:
        return PreprocessingResult(
            X=omega,
            y=G_star,
            diagnostics=diagnostics,
            warnings=warnings_list,
            applied=[],
        )

    # KK check via the standalone public function
    if len(omega) > 5:
        passes, max_slope = check_kramers_kronig(omega, G_prime, G_double_prime)
        diagnostics["max_log_slope_G_prime"] = max_slope
        diagnostics["kk_passes"] = passes
        if not passes:
            warnings_list.append(
                f"G' slope exceeds KK limit: max |d(ln G')/d(ln ω)| = {max_slope:.2f}. "
                "Data may have Kramers-Kronig consistency issues."
            )

    # Check tan(delta) range
    tan_delta = G_double_prime / np.maximum(G_prime, 1e-30)
    diagnostics["tan_delta_range"] = (
        float(np.min(tan_delta)),
        float(np.max(tan_delta)),
    )

    return PreprocessingResult(
        X=omega,
        y=G_star,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=[],
    )


def _preprocess_flow_curve(
    gamma_dot: np.ndarray, sigma: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Flow curve: yield stress detection, shear-banding flag, and η₀ estimate."""
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    sigma_real = np.real(sigma) if np.iscomplexobj(sigma) else sigma

    # Yield stress detection via log-log slope
    if len(gamma_dot) > 5:
        log_gd = np.log10(gamma_dot)
        log_s = np.log10(np.maximum(sigma_real, 1e-30))
        slopes = np.diff(log_s) / np.diff(log_gd)
        min_slope = float(np.min(slopes))
        diagnostics["min_log_slope"] = min_slope

        # Yield stress: plateau at low rates (slope → 0)
        low_idx = max(1, len(gamma_dot) // 5)
        low_slope = float(np.mean(slopes[:low_idx]))
        diagnostics["low_rate_slope"] = low_slope
        if low_slope < 0.1:
            sigma_y_est = float(np.mean(sigma_real[:low_idx]))
            diagnostics["yield_stress_estimate"] = sigma_y_est
            diagnostics["has_yield_stress"] = True
        else:
            diagnostics["has_yield_stress"] = False

        # Shear banding: non-monotonic flow curve
        n_decreasing = int(np.sum(np.diff(sigma_real) < 0))
        diagnostics["n_stress_decreases"] = n_decreasing
        if n_decreasing > 0:
            diagnostics["shear_banding_flag"] = True
            warnings_list.append(
                f"Non-monotonic flow curve detected ({n_decreasing} decreasing steps). "
                "Possible shear banding."
            )
        else:
            diagnostics["shear_banding_flag"] = False

    # Zero-shear viscosity estimate from Newtonian plateau
    try:
        eta_0 = estimate_eta0(gamma_dot, sigma=sigma_real)
        diagnostics["eta_0"] = eta_0
    except (ValueError, ZeroDivisionError):
        pass

    return PreprocessingResult(
        X=gamma_dot,
        y=sigma,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=[],
    )


def _preprocess_startup(
    t: np.ndarray, sigma: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Startup: overshoot diagnostics."""
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    sigma_real = np.real(sigma) if np.iscomplexobj(sigma) else sigma

    # Find overshoot
    peak_idx = int(np.argmax(sigma_real))
    diagnostics["t_peak"] = float(t[peak_idx])
    diagnostics["sigma_peak"] = float(sigma_real[peak_idx])

    # Steady state: average of last 20% of data
    late_start = max(peak_idx + 1, int(0.8 * len(t)))
    if late_start < len(t):
        sigma_ss = float(np.mean(sigma_real[late_start:]))
        diagnostics["sigma_steady_state"] = sigma_ss
        if sigma_ss > 0:
            diagnostics["overshoot_ratio"] = float(sigma_real[peak_idx] / sigma_ss)

    return PreprocessingResult(
        X=t,
        y=sigma,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=[],
    )


def _preprocess_laos(
    t: np.ndarray, response: np.ndarray, **kwargs
) -> PreprocessingResult:
    """LAOS: Ewoldt classification and Q₀ extraction.

    Performs Fourier analysis to extract elastic (Chebyshev e₃/e₁) and
    viscous (v₃/v₁) harmonic ratios. These are used to classify the
    nonlinear regime according to Ewoldt et al. (2008).

    Parameters
    ----------
    t : np.ndarray
        Time array (s).
    response : np.ndarray
        Oscillatory response (stress, strain rate, etc.).
    gamma_0 : float, optional
        Strain amplitude. If provided, used for the corrected Q₀ formula
        Q₀ = (I₃/I₁) / γ₀². If omitted, Q₀ is reported as the raw ratio
        I₃/I₁ and a warning is issued.

    Returns
    -------
    PreprocessingResult
    """
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    response_real = np.real(response) if np.iscomplexobj(response) else response

    # Basic statistics
    diagnostics["n_points"] = len(t)
    diagnostics["response_range"] = (
        float(np.min(response_real)),
        float(np.max(response_real)),
    )

    # Estimate number of cycles from zero crossings
    signs = np.sign(response_real)
    zero_crossings = int(np.sum(np.abs(np.diff(signs)) > 0))
    diagnostics["zero_crossings"] = zero_crossings
    diagnostics["estimated_cycles"] = zero_crossings // 2

    # Fourier analysis — requires at least a few points
    if len(response_real) > 10:
        fft_vals = np.fft.rfft(response_real)
        magnitudes = np.abs(fft_vals)

        # Odd harmonics: indices 1, 3, 5, … correspond to ω, 3ω, 5ω
        # I₁ = magnitudes[1], I₃ = magnitudes[3]
        if len(magnitudes) > 3 and magnitudes[1] > 0:
            I1 = float(magnitudes[1])
            I3 = float(magnitudes[3])
            harmonic_ratio = I3 / I1  # I₃/I₁

            # Q₀ = (I₃/I₁) / γ₀²  (Ewoldt et al. 2008, eq. 9)
            gamma_0: float | None = kwargs.get("gamma_0")
            if gamma_0 is not None and float(gamma_0) > 0:
                Q0 = harmonic_ratio / float(gamma_0) ** 2
                diagnostics["Q0"] = Q0
                diagnostics["Q0_formula"] = "I3_I1 / gamma_0^2"
            else:
                # Fall back to raw ratio and warn
                Q0 = harmonic_ratio
                diagnostics["Q0"] = Q0
                diagnostics["Q0_formula"] = "I3_I1 (gamma_0 not provided)"
                warnings_list.append(
                    "gamma_0 (strain amplitude) not provided: Q₀ reported as "
                    "raw I₃/I₁. Supply gamma_0 kwarg for the correct Q₀."
                )

            if Q0 > 0.1:
                warnings_list.append(
                    f"Strong nonlinearity: Q₀ = {Q0:.4f}. "
                    "Consider using full LAOS analysis."
                )

            # Elastic Chebyshev ratio e₃/e₁ and viscous ratio v₃/v₁
            # In LAOS the real part of the FFT coefficients (cosine projection)
            # corresponds to the elastic (storage) component and the imaginary
            # part corresponds to the viscous (loss) component.
            # e_n = Re[σ̂_n] / (G' amplitude)  →  ratio e₃/e₁ = Re[σ̂₃]/Re[σ̂₁]
            # v_n = Im[σ̂_n]                   →  ratio v₃/v₁ = Im[σ̂₃]/Im[σ̂₁]
            re1 = float(fft_vals[1].real)
            re3 = float(fft_vals[3].real) if len(fft_vals) > 3 else 0.0
            im1 = float(fft_vals[1].imag)
            im3 = float(fft_vals[3].imag) if len(fft_vals) > 3 else 0.0

            e3_e1 = (re3 / re1) if abs(re1) > 1e-30 else 0.0
            v3_v1 = (im3 / im1) if abs(im1) > 1e-30 else 0.0

            # Ewoldt classification (Ewoldt et al. Macromolecules 2008)
            elastic_type = "strain_stiffening" if e3_e1 > 0 else "strain_softening"
            viscous_type = "shear_thickening" if v3_v1 > 0 else "shear_thinning"

            ewoldt: dict[str, Any] = {
                "e3_e1": e3_e1,
                "v3_v1": v3_v1,
                "elastic_behavior": elastic_type,
                "viscous_behavior": viscous_type,
                "I1": I1,
                "I3": I3,
                "harmonic_ratio_I3_I1": harmonic_ratio,
            }
            diagnostics["ewoldt_classification"] = ewoldt

    return PreprocessingResult(
        X=t,
        y=response,
        diagnostics=diagnostics,
        warnings=warnings_list,
        applied=[],
    )
