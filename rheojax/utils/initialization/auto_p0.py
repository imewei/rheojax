"""Generic auto-initialization engine for rheological model parameters.

This module provides ``auto_p0``, a data-driven parameter initializer that
analyzes experimental data to produce physically meaningful starting points
for the non-linear optimizer.  It is intentionally protocol-agnostic: the
same function works for oscillation (G*/G'/G''), relaxation (G(t)), creep
(J(t)), and flow (sigma vs. gamma_dot) data.

Algorithm overview
------------------
1. Convert input arrays to plain NumPy (no JAX tracing overhead at init time).
2. For each parameter in ``model.parameters.keys()`` dispatch on name patterns
   to a dedicated estimator.
3. Clamp every estimated value to the declared parameter bounds.
4. Emit a ``RheoJaxInitWarning`` for any parameter whose estimation failed and
   fall back to the midpoint of its bounds.
5. Return a ``dict[str, float]`` suitable for warm-starting the optimizer.

Design choices
--------------
- Pure NumPy throughout: this code runs exactly once before fitting, so JAX
  compilation overhead is not desirable here.
- Pattern matching uses ``str`` methods only (no regex) for readability and
  speed.
- The module delegates to the existing ``extract_frequency_features()`` helper
  for fractional-order inference and to ``create_prony_parameter_set()``
  documentation to understand how Prony parameters are named.

References
----------
- Dealy & Larson, "Structure and Rheology of Molten Polymers", 2006.
- Ferry, "Viscoelastic Properties of Polymers", 3rd ed., 1980.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.io._exceptions import RheoJaxInitWarning
from rheojax.logging import get_logger
from rheojax.utils.initialization.base import extract_frequency_features

jax, jnp = safe_import_jax()

logger = get_logger(__name__)

if TYPE_CHECKING:
    from rheojax.core.base import BaseModel

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

# Small epsilon to guard against log(0) / division by zero
_EPS: float = 1e-12

# Default fallback values when estimation completely fails and no bounds exist
_DEFAULT_MODULUS: float = 1_000.0  # Pa
_DEFAULT_VISCOSITY: float = 1.0  # Pa·s
_DEFAULT_TAU: float = 1.0  # s
_DEFAULT_POWER_LAW_N: float = 0.5  # dimensionless
_DEFAULT_FRACTIONAL_ORDER: float = 0.5
_DEFAULT_YIELD_STRESS: float = 10.0  # Pa
_DEFAULT_V_ACT: float = 1e-5  # m³ (HVM activation volume)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_p0(
    X: np.ndarray,
    y: np.ndarray,
    model: BaseModel,
    test_mode: str | None = None,
) -> dict[str, float]:
    """Estimate initial parameter values from experimental data.

    Analyzes the structure of *X* and *y* together with the parameter names
    declared in ``model.parameters`` to produce physically reasonable starting
    points for the NLSQ optimizer.

    Parameters
    ----------
    X :
        Independent variable.  Interpretation depends on *test_mode*:

        - ``"oscillation"`` → angular frequency ω (rad/s), shape (N,).
        - ``"relaxation"``  → time t (s), shape (N,).
        - ``"creep"``       → time t (s), shape (N,).
        - ``"flow"``        → shear rate γ̇ (1/s), shape (N,).
    y :
        Dependent variable.  May be complex (G* = G' + iG''), a 2-D array
        [G', G''], or a 1-D real array depending on the protocol and model.
    model :
        An un-fitted (or already-fitted) :class:`~rheojax.core.base.BaseModel`
        instance.  Only ``model.parameters`` is read; the model is never
        mutated by this function.
    test_mode :
        Protocol identifier string.  If ``None``, the function attempts to
        infer it from the data characteristics (complex *y* → ``"oscillation"``,
        monotonically decaying real *y* → ``"relaxation"``, etc.).

    Returns
    -------
    dict[str, float]
        Mapping from parameter name to estimated initial value.  Every key is
        guaranteed to be present in ``model.parameters.keys()``.  Values are
        clamped to the parameter's declared bounds.

    Notes
    -----
    This function runs in pure NumPy before the optimizer starts.  It does not
    call any JAX primitives and is not JIT-compiled.

    Estimation failures for individual parameters produce a
    :class:`~rheojax.io._exceptions.RheoJaxInitWarning` and fall back to the
    midpoint of the parameter's declared bounds.
    """
    # ------------------------------------------------------------------
    # 0. Sanitize inputs — convert to plain NumPy arrays
    # ------------------------------------------------------------------
    X_np = np.asarray(X, dtype=np.float64 if not np.iscomplexobj(X) else complex)
    y_np = np.asarray(y)

    inferred_mode = _infer_test_mode(X_np, y_np, test_mode)
    logger.debug(
        "auto_p0 starting",
        model=model.__class__.__name__,
        test_mode=inferred_mode,
        n_points=len(X_np),
    )

    # ------------------------------------------------------------------
    # 1. Pre-compute a shared feature dictionary from raw data
    # ------------------------------------------------------------------
    features = _compute_data_features(X_np, y_np, inferred_mode)

    # ------------------------------------------------------------------
    # 2. Iterate over every declared parameter and estimate its value
    # ------------------------------------------------------------------
    param_names: list[str] = list(model.parameters.keys())
    result: dict[str, float] = {}
    failed: list[str] = []

    for name in param_names:
        try:
            raw = _estimate_single_parameter(
                name=name,
                features=features,
                test_mode=inferred_mode,
                model=model,
            )
            if raw is None:
                raise ValueError(f"Estimator returned None for '{name}'")
            clamped = _clamp_to_bounds(raw, name, model)
            result[name] = clamped
            logger.debug(
                "Parameter estimated",
                name=name,
                raw=raw,
                clamped=clamped,
            )
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            fallback = _bounds_midpoint(name, model)
            result[name] = fallback
            logger.debug(
                "Parameter estimation failed, using bounds midpoint",
                name=name,
                fallback=fallback,
                error=str(exc),
            )

    if failed:
        warnings.warn(
            f"auto_p0: could not estimate initial values for parameters "
            f"{failed!r} in {model.__class__.__name__}. "
            "Falling back to bounds midpoints for those parameters.",
            RheoJaxInitWarning,
            stacklevel=2,
        )

    logger.debug(
        "auto_p0 complete",
        model=model.__class__.__name__,
        n_params=len(result),
        n_failed=len(failed),
    )
    return result


# ---------------------------------------------------------------------------
# Helper functions — public, documented, and importable
# ---------------------------------------------------------------------------


def _estimate_crossover_frequency(
    omega: np.ndarray,
    G_star: np.ndarray,
) -> float:
    """Find the G'/G'' crossover frequency.

    The crossover occurs where G'(ω) = G''(ω), which marks the transition
    from solid-like to liquid-like behaviour and gives the reciprocal of the
    dominant relaxation time.

    Parameters
    ----------
    omega :
        Angular frequency array (rad/s), shape (N,).  Must be sorted in
        ascending order.
    G_star :
        Complex modulus G* = G' + iG'' (complex, shape (N,)) **or** a 2-D
        array [G', G''] with shape (N, 2).

    Returns
    -------
    float
        Crossover frequency ω_cross (rad/s).  Returns the geometric mean of
        ``omega`` if no crossover is found.
    """
    G_prime, G_double_prime = _split_G_star(G_star)

    if len(G_prime) < 2:
        return float(np.sqrt(omega.min() * omega.max() + _EPS))

    # Sign of (G' - G'') flips at crossover
    diff = G_prime - G_double_prime
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # No crossover found — use frequency of minimum tan(delta) instead
        tan_delta = G_double_prime / (G_prime + _EPS)
        return float(omega[np.argmin(tan_delta)])

    # Use the first crossover; interpolate log-linearly for accuracy
    i = sign_changes[0]
    if abs(diff[i + 1] - diff[i]) < _EPS:
        return float(omega[i])

    # Linear interpolation of sign crossing in log-omega space
    log_o0 = np.log(omega[i] + _EPS)
    log_o1 = np.log(omega[i + 1] + _EPS)
    frac = -diff[i] / (diff[i + 1] - diff[i])
    log_cross = log_o0 + frac * (log_o1 - log_o0)
    return float(np.exp(log_cross))


def _estimate_decay_time(t: np.ndarray, G_t: np.ndarray) -> float:
    """Find the 1/e decay time from relaxation modulus data.

    Locates the first time at which G(t) drops to G(t₀)/e, providing a
    natural estimate of the dominant relaxation time.

    Parameters
    ----------
    t :
        Time array (s), shape (N,).
    G_t :
        Relaxation modulus G(t) (Pa), shape (N,).  Must be positive and
        monotonically non-increasing.

    Returns
    -------
    float
        1/e decay time (s).  Returns ``t[-1]`` if the data does not decay to
        1/e of its initial value within the observation window, and ``t[0]``
        if *G_t* is constant.
    """
    G_t_real = np.real(G_t).astype(float)
    t_real = np.asarray(t, dtype=float)

    if len(t_real) < 2:
        return float(t_real[0]) if len(t_real) > 0 else _DEFAULT_TAU

    G0 = float(G_t_real[0])
    if G0 < _EPS:
        return _DEFAULT_TAU

    target = G0 / np.e
    # Find first crossing below 1/e
    below = np.where(G_t_real <= target)[0]
    if len(below) == 0:
        # G(t) stays above 1/e throughout — return last time as upper bound
        return float(t_real[-1])

    i = below[0]
    if i == 0:
        return float(t_real[0])

    # Linear interpolation in log-time space for better accuracy
    try:
        log_t0 = np.log(t_real[i - 1] + _EPS)
        log_t1 = np.log(t_real[i] + _EPS)
        frac = (G_t_real[i - 1] - target) / (G_t_real[i - 1] - G_t_real[i] + _EPS)
        tau_est = np.exp(log_t0 + frac * (log_t1 - log_t0))
        return float(np.clip(tau_est, t_real[0], t_real[-1]))
    except Exception:  # noqa: BLE001
        return float(t_real[i])


def _estimate_yield_stress(
    gamma_dot: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """Estimate yield stress by low-shear-rate log-log extrapolation.

    Fits a power law to the lowest 30 % of shear-rate data in log-log space
    and extrapolates to γ̇ → 0 to estimate σ_y.

    Parameters
    ----------
    gamma_dot :
        Shear rate array (1/s), shape (N,).  Must be positive.
    sigma :
        Shear stress array (Pa), shape (N,).

    Returns
    -------
    float
        Estimated yield stress σ_y (Pa).  Returns ``sigma[0]`` (lowest-rate
        stress) as a conservative fallback when extrapolation is not possible.
    """
    sigma_real = np.abs(np.real(sigma)).astype(float)
    gd_real = np.asarray(gamma_dot, dtype=float)

    if len(gd_real) < 3:
        return float(sigma_real[0]) if len(sigma_real) > 0 else _DEFAULT_YIELD_STRESS

    # Sort by ascending shear rate
    order = np.argsort(gd_real)
    gd_sorted = gd_real[order]
    sg_sorted = sigma_real[order]

    # Filter to lowest 30% of shear-rate points (at least 3)
    n_low = max(3, int(len(gd_sorted) * 0.30))
    gd_low = gd_sorted[:n_low]
    sg_low = sg_sorted[:n_low]

    # Work in log-log space; skip zeros
    mask = (gd_low > _EPS) & (sg_low > _EPS)
    if mask.sum() < 2:
        return float(sg_sorted[0])

    log_gd = np.log10(gd_low[mask])
    log_sg = np.log10(sg_low[mask])

    # Least-squares line: log_sg = m * log_gd + b
    try:
        coeffs = np.polyfit(log_gd, log_sg, 1)
        m, b = float(coeffs[0]), float(coeffs[1])
        # Extrapolate to γ̇ = 1e-6 (near-zero but avoids log(0))
        sigma_y = 10.0 ** (m * np.log10(1e-6) + b)
        if not np.isfinite(sigma_y) or sigma_y <= 0:
            return float(sg_sorted[0])
        return float(sigma_y)
    except Exception:  # noqa: BLE001
        return float(sg_sorted[0])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_test_mode(
    X: np.ndarray,
    y: np.ndarray,
    explicit: str | None,
) -> str:
    """Infer protocol from data characteristics when not specified explicitly."""
    if explicit is not None:
        # Normalize canonical aliases so downstream branches match
        _aliases = {"flow_curve": "flow", "flow curve": "flow"}
        return _aliases.get(explicit.lower(), explicit)

    # Complex y almost certainly means oscillation (G* = G' + iG'')
    if np.iscomplexobj(y):
        return "oscillation"

    # 2-D y with shape (N, 2) → [G', G''] — oscillation
    if y.ndim == 2 and y.shape[1] == 2:
        return "oscillation"

    # X spanning many orders of magnitude (> 2 decades) with positive small values
    # is characteristic of frequency or shear-rate sweeps
    x_pos = X[X > 0]
    y_real = np.real(y).astype(float)
    y_positive = np.abs(y_real)
    if len(x_pos) > 1:
        decades = np.log10(x_pos.max() / x_pos.min())
        # P2-Fit-4: Check for monotonically decreasing data first — this is
        # characteristic of flow curves (viscosity vs shear rate) even when
        # x_pos.min() < 1.0.  Without this, low-shear-rate flow curves are
        # misidentified as oscillation.
        if decades > 2.0 and len(y_positive) > 1 and np.all(np.diff(y_positive) <= 0):
            return "flow"
        if decades > 2.0 and x_pos.min() < 1.0:
            return "oscillation"
        if decades > 2.0:
            return "flow"
    if len(y_real) > 1 and np.all(np.diff(y_real) <= 0):
        return "relaxation"

    # Monotonically increasing y with time-like X → creep
    if len(y_real) > 1 and np.all(np.diff(y_real) >= 0):
        return "creep"

    return "relaxation"  # conservative default


def _split_G_star(
    G_star: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (G_prime, G_double_prime) from any supported G* format."""
    if np.iscomplexobj(G_star):
        return np.real(G_star).astype(float), np.imag(G_star).astype(float)
    if G_star.ndim == 2 and G_star.shape[1] == 2:
        return G_star[:, 0].astype(float), G_star[:, 1].astype(float)
    # 1-D real magnitude — cannot separate components
    mag = np.abs(G_star).astype(float)
    return mag / np.sqrt(2), mag / np.sqrt(2)


def _compute_data_features(
    X: np.ndarray,
    y: np.ndarray,
    test_mode: str,
) -> dict[str, Any]:
    """Extract a shared feature set from (X, y) data.

    This dictionary is passed to individual estimators so that each one avoids
    re-computing the same quantities.
    """
    features: dict[str, Any] = {
        "test_mode": test_mode,
        "X": X,
        "y": y,
        "n": len(X),
    }

    y_real = np.real(y).astype(float)

    # ---- Oscillation features -------------------------------------------
    if test_mode == "oscillation":
        omega = X.astype(float)
        G_prime, G_pp = _split_G_star(y)
        G_mag = np.sqrt(G_prime**2 + G_pp**2)

        # G' at minimum tan(delta) → plateau modulus
        tan_delta = G_pp / (G_prime + _EPS)
        idx_min_td = int(np.argmin(np.abs(tan_delta)))
        features["G_plateau"] = float(G_prime[idx_min_td])

        # Viscosity estimate: |G*|/ω at minimum ω
        idx_low = int(np.argmin(omega))
        features["eta_low_rate"] = float(G_mag[idx_low] / (omega[idx_low] + _EPS))

        # Crossover frequency → relaxation time
        try:
            omega_cross = _estimate_crossover_frequency(omega, y)
            features["tau_cross"] = float(1.0 / (omega_cross + _EPS))
        except Exception:  # noqa: BLE001
            features["tau_cross"] = _DEFAULT_TAU

        # Frequency-domain features for fractional order estimation
        try:
            freq_feats = extract_frequency_features(omega, y)
            features["freq_features"] = freq_feats
            features["alpha_estimate"] = float(
                freq_feats.get("alpha_estimate", _DEFAULT_FRACTIONAL_ORDER)
            )
        except Exception:  # noqa: BLE001
            features["freq_features"] = {}
            features["alpha_estimate"] = _DEFAULT_FRACTIONAL_ORDER

        features["G_prime"] = G_prime
        features["G_pp"] = G_pp
        features["G_mag"] = G_mag
        features["omega"] = omega

    # ---- Relaxation features ---------------------------------------------
    elif test_mode == "relaxation":
        t = X.astype(float)
        G_t = y_real

        features["G0_relaxation"] = float(G_t[0]) if len(G_t) > 0 else _DEFAULT_MODULUS
        try:
            features["tau_decay"] = _estimate_decay_time(t, G_t)
        except Exception:  # noqa: BLE001
            features["tau_decay"] = _DEFAULT_TAU

        features["t"] = t
        features["G_t"] = G_t

    # ---- Creep features --------------------------------------------------
    elif test_mode == "creep":
        t = X.astype(float)
        J_t = y_real
        features["J0"] = float(J_t[0]) if len(J_t) > 0 else 1.0 / _DEFAULT_MODULUS
        features["G0_creep"] = (
            float(1.0 / (J_t[0] + _EPS)) if len(J_t) > 0 else _DEFAULT_MODULUS
        )
        features["t"] = t
        features["J_t"] = J_t

    # ---- Flow curve features ---------------------------------------------
    elif test_mode == "flow":
        gamma_dot = X.astype(float)
        sigma = np.abs(y_real)
        order = np.argsort(gamma_dot)
        gd = gamma_dot[order]
        sg = sigma[order]

        # Low-rate viscosity (lowest 10% of shear rates)
        n10 = max(1, int(len(gd) * 0.10))
        features["eta_0_est"] = float(np.mean(sg[:n10] / (gd[:n10] + _EPS)))

        # Yield stress
        try:
            features["sigma_y_est"] = _estimate_yield_stress(gd, sg)
        except Exception:  # noqa: BLE001
            features["sigma_y_est"] = (
                float(sg[0]) if len(sg) > 0 else _DEFAULT_YIELD_STRESS
            )

        # Power-law exponent from log-log slope in medium shear-rate range
        if len(gd) >= 3:
            mask = gd > _EPS
            if mask.sum() >= 2:
                try:
                    coeffs = np.polyfit(
                        np.log10(gd[mask]), np.log10(sg[mask] + _EPS), 1
                    )
                    features["n_power_law"] = float(np.clip(coeffs[0], 0.0, 1.0))
                except Exception:  # noqa: BLE001
                    features["n_power_law"] = _DEFAULT_POWER_LAW_N
            else:
                features["n_power_law"] = _DEFAULT_POWER_LAW_N
        else:
            features["n_power_law"] = _DEFAULT_POWER_LAW_N

        features["gamma_dot"] = gd
        features["sigma"] = sg
        features["sigma_max"] = (
            float(sg.max()) if len(sg) > 0 else _DEFAULT_YIELD_STRESS
        )

    return features


def _estimate_single_parameter(
    name: str,
    features: dict[str, Any],
    test_mode: str,
    model: BaseModel,
) -> float:
    """Dispatch a single parameter name to the appropriate estimator.

    Returns a raw (unclamped) float estimate, or raises if estimation fails.
    """
    nm = name.lower()

    # ------------------------------------------------------------------
    # Viscosity parameters
    # ------------------------------------------------------------------
    if nm in ("eta_0", "eta"):
        return _est_eta_0(features, test_mode)

    if nm == "eta_inf":
        eta0 = _est_eta_0(features, test_mode)
        return eta0 * 0.01  # η_inf is typically 1-2 orders below η_0

    if nm == "eta_s":
        eta0 = _est_eta_0(features, test_mode)
        return eta0 * 0.01  # solvent viscosity ≈ 1% of zero-rate viscosity

    if nm == "eta_p":
        # Polymer contribution: η_p = η_0 - η_s
        eta0 = _est_eta_0(features, test_mode)
        return eta0 * 0.99

    # ------------------------------------------------------------------
    # Relaxation time / lambda parameters
    # ------------------------------------------------------------------
    if nm in (
        "tau",
        "lambda",
        "tau_r",
        "tau_d",
        "tau_e",
        "tau_rep",
        "tau_0",
        "tau_c",
        "tau_eq",
        "lam",
    ):
        return _est_tau(features, test_mode)

    if nm == "lambda_0":
        return _est_tau(features, test_mode)

    # ------------------------------------------------------------------
    # Modulus / stiffness parameters
    # ------------------------------------------------------------------
    if nm in ("g0", "g_0", "g_n0", "g_n", "g_e", "ge", "g_eq"):
        return _est_modulus(features, test_mode)

    if nm in ("g_g", "gg", "g_glassy"):
        # Glassy modulus: use high-frequency plateau for oscillation
        if test_mode == "oscillation" and "freq_features" in features:
            return float(
                features["freq_features"].get("high_plateau", _DEFAULT_MODULUS)
            )
        return _est_modulus(features, test_mode) * 10.0

    if nm in ("g_inf", "g_infinity"):
        # Equilibrium (rubbery) modulus: use low-frequency plateau
        if test_mode == "oscillation" and "freq_features" in features:
            return float(features["freq_features"].get("low_plateau", _DEFAULT_MODULUS))
        return _est_modulus(features, test_mode) * 0.1

    # ------------------------------------------------------------------
    # Yield stress
    # ------------------------------------------------------------------
    if nm in ("sigma_y", "sigma_yield", "tau_y", "tau_yield", "sigma_c"):
        if test_mode == "flow":
            return float(features.get("sigma_y_est", _DEFAULT_YIELD_STRESS))
        return _est_modulus(features, test_mode) * 0.01

    # ------------------------------------------------------------------
    # Power-law exponent (flow) — guard against multi-mode indices
    # ------------------------------------------------------------------
    if nm == "n" or (nm.startswith("n") and len(nm) == 1):
        if test_mode == "flow":
            return float(features.get("n_power_law", _DEFAULT_POWER_LAW_N))
        return _DEFAULT_POWER_LAW_N

    # ------------------------------------------------------------------
    # Fractional orders (alpha, beta)
    # ------------------------------------------------------------------
    if nm in ("alpha", "alpha_1"):
        return float(features.get("alpha_estimate", _DEFAULT_FRACTIONAL_ORDER))

    if nm in ("beta", "beta_1"):
        # Beta is often the complementary order: ~1 - alpha
        alpha = float(features.get("alpha_estimate", _DEFAULT_FRACTIONAL_ORDER))
        return float(np.clip(1.0 - alpha, 0.01, 0.99))

    # ------------------------------------------------------------------
    # Prony / GMM / multi-mode parameters
    # ------------------------------------------------------------------
    if _is_prony_modulus(nm):
        return _est_prony_modulus(name, features, test_mode, model)

    if _is_prony_tau(nm):
        return _est_prony_tau(name, features, test_mode, model)

    # ------------------------------------------------------------------
    # Multi-mode modal parameters (G_modes, kd_modes)
    # ------------------------------------------------------------------
    if "g_mode" in nm or "g_modes" in nm:
        return _est_modulus(features, test_mode)

    if "k_d" in nm or "kd" in nm or "kd_mode" in nm or "kd_modes" in nm:
        return 1.0  # rate coefficient — dimensionless default

    # ------------------------------------------------------------------
    # Spring extensibility (FENE / finitely extensible models)
    # ------------------------------------------------------------------
    if nm in ("l2", "l_max", "l_sq", "l_squared", "b_fene"):
        return 100.0  # dimensionless — typical FENE extensibility

    # ------------------------------------------------------------------
    # Gel strength (Winter-Chambon gel point parameter)
    # ------------------------------------------------------------------
    if nm in ("s", "s_gel", "gel_strength", "s_g"):
        return _est_modulus(features, test_mode) * 0.1

    # ------------------------------------------------------------------
    # Activation volume (HVM models)
    # ------------------------------------------------------------------
    if nm in ("v_act", "v_activation"):
        return _DEFAULT_V_ACT

    # ------------------------------------------------------------------
    # Rate / kinetic parameters
    # ------------------------------------------------------------------
    if nm in ("k_d", "kd", "k_0", "nu", "nu_0", "k_r", "k_f", "k_plus", "k_minus"):
        return 1.0  # s⁻¹ — generic rate

    if nm in ("k", "k1", "k2"):
        return 1.0

    # ------------------------------------------------------------------
    # Dimensionless parameters (Giesekus mobility, PTT epsilon, etc.)
    # ------------------------------------------------------------------
    if nm in (
        "alpha_giesekus",
        "alpha_mob",
        "xi",
        "eps",
        "epsilon_ptt",
        "f_c",
        "f_neq",
    ):
        return 0.1

    if nm in ("beta_ratio", "beta_mob"):
        return 0.5

    # ------------------------------------------------------------------
    # Concentration / composition parameters
    # ------------------------------------------------------------------
    if nm in ("phi", "phi_0", "c", "c_0"):
        return 0.1

    # ------------------------------------------------------------------
    # Temperature / activation energy parameters
    # ------------------------------------------------------------------
    if nm in ("e_a", "e_act", "delta_h"):
        return 50_000.0  # J/mol — typical polymer activation energy

    if nm in ("t_ref", "t_0"):
        return 298.15  # K — room temperature reference

    # ------------------------------------------------------------------
    # Catch-all: use the feature that best matches the physics
    # ------------------------------------------------------------------
    return _generic_fallback(nm, features, test_mode)


# ---------------------------------------------------------------------------
# Per-family estimators
# ---------------------------------------------------------------------------


def _est_eta_0(features: dict[str, Any], test_mode: str) -> float:
    """Estimate zero-shear viscosity."""
    if test_mode == "flow":
        return float(features.get("eta_0_est", _DEFAULT_VISCOSITY))
    if test_mode == "oscillation":
        return float(features.get("eta_low_rate", _DEFAULT_VISCOSITY))
    # Relaxation / creep: G0 * tau
    G0 = features.get("G0_relaxation", features.get("G0_creep", _DEFAULT_MODULUS))
    tau = features.get("tau_decay", _DEFAULT_TAU)
    return float(G0) * float(tau)


def _est_tau(features: dict[str, Any], test_mode: str) -> float:
    """Estimate dominant relaxation time."""
    if test_mode == "oscillation":
        return float(features.get("tau_cross", _DEFAULT_TAU))
    if test_mode == "relaxation":
        return float(features.get("tau_decay", _DEFAULT_TAU))
    if test_mode == "creep":
        return float(features.get("tau_decay", _DEFAULT_TAU))
    # Flow: τ = 1 / (critical shear rate) — use eta_0 / sigma_max heuristic
    eta0 = float(features.get("eta_0_est", _DEFAULT_VISCOSITY))
    sigma_max = float(features.get("sigma_max", _DEFAULT_YIELD_STRESS))
    return float(eta0 / (sigma_max + _EPS))


def _est_modulus(features: dict[str, Any], test_mode: str) -> float:
    """Estimate a characteristic modulus."""
    if test_mode == "oscillation":
        return float(features.get("G_plateau", _DEFAULT_MODULUS))
    if test_mode == "relaxation":
        return float(features.get("G0_relaxation", _DEFAULT_MODULUS))
    if test_mode == "creep":
        return float(features.get("G0_creep", _DEFAULT_MODULUS))
    if test_mode == "flow":
        # For flow models, use sigma_max as a modulus proxy
        return float(features.get("sigma_max", _DEFAULT_MODULUS))
    return _DEFAULT_MODULUS


# ---------------------------------------------------------------------------
# Prony / GMM mode helpers
# ---------------------------------------------------------------------------


def _is_prony_modulus(nm: str) -> bool:
    """Return True if the parameter name looks like a Prony modulus mode."""
    # Patterns: G_i, G_1, G_2, ..., E_i, E_1, ..., G_inf, E_inf
    # Exclude generic G0-type names already handled above
    for prefix in ("g_", "e_"):
        if nm.startswith(prefix):
            suffix = nm[len(prefix) :]
            if suffix.lstrip("-").isdigit() or suffix == "inf" or suffix == "infinity":
                return True
    return False


def _is_prony_tau(nm: str) -> bool:
    """Return True if the parameter name looks like a Prony relaxation time."""
    # Patterns: tau_1, tau_2, ..., tau_N
    if nm.startswith("tau_"):
        suffix = nm[4:]
        return suffix.lstrip("-").isdigit()
    return False


def _est_prony_modulus(
    name: str,
    features: dict[str, Any],
    test_mode: str,
    model: BaseModel,
) -> float:
    """Estimate a single Prony modulus mode by partitioning the total modulus."""
    # Count how many Prony modulus modes exist in the model
    param_names = list(model.parameters.keys())
    prony_G_names = [n for n in param_names if _is_prony_modulus(n.lower())]

    n_modes = len(prony_G_names)
    G_total = _est_modulus(features, test_mode)

    # Distribute modulus equally across modes (log-uniform partition)
    if n_modes <= 0:
        return G_total

    # Each mode gets 1/n_modes of total modulus — simple equipartition
    per_mode = G_total / n_modes
    return max(per_mode, _EPS)


def _est_prony_tau(
    name: str,
    features: dict[str, Any],
    test_mode: str,
    model: BaseModel,
) -> float:
    """Estimate a Prony relaxation time using log-spaced distribution."""
    # Count how many Prony tau modes exist
    param_names = list(model.parameters.keys())
    prony_tau_names = [n for n in param_names if _is_prony_tau(n.lower())]
    n_modes = len(prony_tau_names)

    # Determine the mode index from the parameter name (e.g., "tau_3" → index 2)
    nm = name.lower()
    suffix = nm.split("_")[-1]
    try:
        mode_idx = int(suffix) - 1  # 0-based
    except ValueError:
        mode_idx = 0

    if n_modes <= 1:
        return _est_tau(features, test_mode)

    # Span relaxation times log-uniformly over ±2 decades around τ_dominant
    tau_dominant = _est_tau(features, test_mode)
    log_tau_min = np.log10(tau_dominant) - 2.0
    log_tau_max = np.log10(tau_dominant) + 2.0
    log_taus = np.linspace(log_tau_min, log_tau_max, n_modes)
    mode_idx_clamped = int(np.clip(mode_idx, 0, n_modes - 1))
    return float(10.0 ** log_taus[mode_idx_clamped])


# ---------------------------------------------------------------------------
# Bound clamping and fallback
# ---------------------------------------------------------------------------


def _clamp_to_bounds(value: float, name: str, model: BaseModel) -> float:
    """Clamp *value* to the declared bounds of parameter *name*."""
    if not np.isfinite(value):
        value = _bounds_midpoint(name, model)

    try:
        bounds = model.parameters[name].bounds
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo >= hi:
            # P2-Fit-5: When bounds are degenerate (lo == hi), the parameter
            # is fixed — always return the bound value, not the estimate.
            return lo
        return float(np.clip(value, lo, hi))
    except Exception:  # noqa: BLE001
        return value


def _bounds_midpoint(name: str, model: BaseModel) -> float:
    """Return the geometric or arithmetic midpoint of a parameter's bounds."""
    try:
        bounds = model.parameters[name].bounds
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= 0 or hi <= 0:
            # Arithmetic midpoint for non-positive bounds
            return (lo + hi) / 2.0
        # Geometric midpoint — appropriate for log-scale parameters
        return float(np.sqrt(lo * hi))
    except Exception:  # noqa: BLE001
        return _DEFAULT_MODULUS


def _generic_fallback(
    nm: str,
    features: dict[str, Any],
    test_mode: str,
) -> float:
    """Heuristic fallback for unrecognised parameter names.

    Uses naming conventions to guess the physical dimension:
    - Contains "g" or "e" (modulus-like) → characteristic modulus
    - Contains "tau" or "lam" (time-like) → characteristic time
    - Contains "eta" or "visc" (viscosity-like) → viscosity estimate
    - Otherwise → 1.0
    """
    if any(k in nm for k in ("modulus", "_g", "g_", "stiffness")):
        return _est_modulus(features, test_mode)
    if any(k in nm for k in ("tau", "lam", "time", "relax")):
        return _est_tau(features, test_mode)
    if any(k in nm for k in ("eta", "visc")):
        return _est_eta_0(features, test_mode)
    return 1.0


__all__ = [
    "auto_p0",
]
