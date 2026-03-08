"""Protocol-aware data preprocessing for rheological experiments.

Provides per-protocol diagnostics and preprocessing that can be applied
before fitting. Each protocol has specific data quality checks and
optional cleaning/conditioning steps.
"""

from __future__ import annotations

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

    Args:
        X: Input array (time, frequency, shear rate, etc.).
        y: Target array (modulus, stress, viscosity, etc.).
        test_mode: Protocol string (``"relaxation"``, ``"oscillation"``, etc.).
        **kwargs: Protocol-specific options.

    Returns:
        PreprocessingResult with diagnostics and optionally cleaned data.
    """
    X = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y)

    dispatchers = {
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


# ---------------------------------------------------------------------------
# Protocol-specific preprocessors
# ---------------------------------------------------------------------------


def _preprocess_relaxation(
    t: np.ndarray, G_t: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Relaxation: detect inertia ringing and truncation artifacts."""
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}
    applied: list[str] = []

    G_t_real = np.real(G_t) if np.iscomplexobj(G_t) else G_t

    # Detect inertia ringing (oscillations at short times)
    if len(t) > 10:
        dG = np.diff(G_t_real[:20])
        sign_changes = np.sum(np.diff(np.sign(dG)) != 0)
        diagnostics["short_time_sign_changes"] = int(sign_changes)
        if sign_changes > 3:
            warnings_list.append(
                f"Possible inertia ringing detected: {sign_changes} sign changes "
                "in first 20 points. Consider truncating early data."
            )

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
        X=t, y=G_t, diagnostics=diagnostics,
        warnings=warnings_list, applied=applied,
    )


def _preprocess_creep(
    t: np.ndarray, J_t: np.ndarray, **kwargs
) -> PreprocessingResult:
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
            diagnostics["has_viscous_flow"] = bool(slope > 0 and
                slope * t_late[-1] > 0.5 * J_late[-1])

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
        X=t, y=J_t, diagnostics=diagnostics, warnings=warnings_list, applied=[],
    )


def _preprocess_oscillation(
    omega: np.ndarray, G_star: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Oscillation: Kramers-Kronig consistency check."""
    warnings_list: list[str] = []
    diagnostics: dict[str, Any] = {}

    if np.iscomplexobj(G_star):
        G_prime = G_star.real
        G_double_prime = G_star.imag
    elif G_star.ndim == 2 and G_star.shape[1] == 2:
        G_prime = G_star[:, 0]
        G_double_prime = G_star[:, 1]
    else:
        return PreprocessingResult(X=omega, y=G_star, diagnostics=diagnostics,
                                   warnings=warnings_list, applied=[])

    # Approximate KK check: d(ln G')/d(ln ω) should be ≤ 2
    if len(omega) > 5:
        log_omega = np.log(omega)
        log_Gp = np.log(np.maximum(G_prime, 1e-30))
        slopes = np.diff(log_Gp) / np.diff(log_omega)
        max_slope = float(np.max(np.abs(slopes)))
        diagnostics["max_log_slope_G_prime"] = max_slope
        if max_slope > 2.5:
            warnings_list.append(
                f"G' slope exceeds KK limit: max |d(ln G')/d(ln ω)| = {max_slope:.2f}. "
                "Data may have Kramers-Kronig consistency issues."
            )

    # Check tan(delta) range
    tan_delta = G_double_prime / np.maximum(G_prime, 1e-30)
    diagnostics["tan_delta_range"] = (float(np.min(tan_delta)), float(np.max(tan_delta)))

    return PreprocessingResult(
        X=omega, y=G_star, diagnostics=diagnostics,
        warnings=warnings_list, applied=[],
    )


def _preprocess_flow_curve(
    gamma_dot: np.ndarray, sigma: np.ndarray, **kwargs
) -> PreprocessingResult:
    """Flow curve: yield stress detection, shear-banding flag."""
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

    return PreprocessingResult(
        X=gamma_dot, y=sigma, diagnostics=diagnostics,
        warnings=warnings_list, applied=[],
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
            diagnostics["overshoot_ratio"] = float(
                sigma_real[peak_idx] / sigma_ss
            )

    return PreprocessingResult(
        X=t, y=sigma, diagnostics=diagnostics,
        warnings=warnings_list, applied=[],
    )


def _preprocess_laos(
    t: np.ndarray, response: np.ndarray, **kwargs
) -> PreprocessingResult:
    """LAOS: Ewoldt classification and Q₀ extraction."""
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

    # Q₀ (fundamental to third harmonic ratio) via FFT
    if len(response_real) > 10:
        fft_vals = np.fft.rfft(response_real)
        magnitudes = np.abs(fft_vals)
        if len(magnitudes) > 3 and magnitudes[1] > 0:
            Q0 = float(magnitudes[3] / magnitudes[1])
            diagnostics["Q0"] = Q0
            if Q0 > 0.1:
                warnings_list.append(
                    f"Strong nonlinearity: Q₀ = {Q0:.4f}. "
                    "Consider using full LAOS analysis."
                )

    return PreprocessingResult(
        X=t, y=response, diagnostics=diagnostics,
        warnings=warnings_list, applied=[],
    )
