"""Post-fit physics validation for rheological models.

This module provides a strategy-based checker that inspects fitted model
parameters for physical plausibility.  It is deliberately JAX-free so that it
can run in any context (CLI, tests, GUI) without triggering device
initialisation.

Typical usage::

    from rheojax.utils.physics_checks import check_fit_physics

    violations = check_fit_physics(model)
    for v in violations:
        if v.severity == "error":
            warnings.warn(v.message, RheoJaxPhysicsWarning)

The function returns data only; it never calls ``warnings.warn`` directly.
The caller decides how to surface violations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rheojax.io._exceptions import (
    RheoJaxPhysicsWarning,  # noqa: F401 — re-exported for caller convenience
)
from rheojax.logging import get_logger

if TYPE_CHECKING:
    # Avoid hard import at module load — the caller already has these objects.
    from rheojax.core.fit_result import FitResult
    from rheojax.core.parameters import Parameter, ParameterSet

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicsViolation:
    """A single physics violation found in fitted parameters.

    Attributes:
        parameter: Name of the parameter that triggered the check.
        value: Fitted numeric value of that parameter.
        check: Short identifier for the rule that fired
               (e.g. ``"positive_moduli"``).
        message: Human-readable explanation of the violation.
        severity: ``"error"`` for thermodynamic impossibilities,
                  ``"warning"`` for implausible-but-not-impossible values.
    """

    parameter: str
    value: float
    check: str
    message: str
    severity: str  # "error" | "warning"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_BOUND_PROXIMITY_FRACTION = 0.001  # 0.1 % of (upper - lower)


def _coerce_float(value: Any) -> float | None:
    """Convert a parameter value to Python float, or return None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _build_param_map(model: Any, result: FitResult | None) -> dict[str, tuple[float, tuple[float, float] | None]]:
    """Return ``{name: (value, bounds)}`` from the model and optional result.

    The result's ``params`` dict takes precedence for values so that the
    freshest fitted values are always checked.  Bounds come from the model's
    ParameterSet because ``FitResult.params`` does not carry them.
    """
    param_map: dict[str, tuple[float, tuple[float, float] | None]] = {}

    # Collect from the model's ParameterSet first.
    try:
        ps: ParameterSet = model.parameters  # type: ignore[attr-defined]
        for name in ps.keys():
            p: Parameter = ps[name]
            value = _coerce_float(p.value)
            if value is None:
                continue
            bounds = p.bounds  # tuple[float, float] | None
            param_map[name] = (value, bounds)
    except AttributeError:
        logger.debug(
            "Model does not expose .parameters ParameterSet; skipping model-side extraction",
            model=type(model).__name__,
        )

    # Override values with result.params when a FitResult is supplied.
    if result is not None:
        try:
            for name, raw_value in result.params.items():
                value = _coerce_float(raw_value)
                if value is None:
                    continue
                bounds = param_map.get(name, (None, None))[1]
                param_map[name] = (value, bounds)
        except AttributeError:
            logger.debug(
                "FitResult does not expose .params dict; skipping result-side extraction",
                result_type=type(result).__name__,
            )

    return param_map


# ---------------------------------------------------------------------------
# Individual checker functions
# ---------------------------------------------------------------------------
# Each checker receives the full param_map and returns a list of violations.
# Checkers are pure functions with no side-effects.


def _check_positive_moduli(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Moduli (G, E, G_N, G_e, G0, G_i) must be strictly positive."""
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        # Match parameter names that represent moduli.
        # Heuristic: starts with G or E (case-insensitive), or is G_N / G0 / Ge.
        is_modulus = (
            lower.startswith("g")
            or lower.startswith("e_")
            or lower in {"e", "e0", "ge", "gn", "g_n", "g_e", "g0"}
            or (lower.startswith("e") and lower[1:2].isdigit())
        )
        # Exclude non-modulus parameters that happen to start with "g" or "e":
        # eta, epsilon, n, gamma-like names.
        excluded_prefixes = ("eta", "eps", "gamma", "g_dot", "gdot")
        if any(lower.startswith(ex) for ex in excluded_prefixes):
            is_modulus = False

        if not is_modulus:
            continue

        if value <= 0.0:
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="positive_moduli",
                    message=(
                        f"Modulus parameter '{name}' = {value:.4g} is non-positive. "
                        "Elastic moduli must be strictly positive."
                    ),
                    severity="error",
                )
            )
    return violations


def _check_positive_times(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Relaxation/retardation times (tau, lambda, tau_i) must be strictly positive."""
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        is_time = (
            lower.startswith("tau")
            or lower.startswith("lambda")
            or lower.startswith("lam_")
            or lower in {"lam", "t_relax", "t_ret", "t_r"}
        )
        if not is_time:
            continue
        if value <= 0.0:
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="positive_times",
                    message=(
                        f"Time-scale parameter '{name}' = {value:.4g} is non-positive. "
                        "Relaxation and retardation times must be strictly positive."
                    ),
                    severity="error",
                )
            )
    return violations


def _check_positive_viscosity(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Viscosities (eta, eta_0, eta_inf, eta_s) must be strictly positive."""
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        is_viscosity = lower.startswith("eta")
        if not is_viscosity:
            continue
        if value <= 0.0:
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="positive_viscosity",
                    message=(
                        f"Viscosity parameter '{name}' = {value:.4g} is non-positive. "
                        "Viscosities must be strictly positive."
                    ),
                    severity="error",
                )
            )
    return violations


def _check_fractional_orders(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Fractional orders (alpha, beta) must lie in the open interval (0, 1).

    Values outside (0, 1) violate the thermodynamic admissibility condition
    for fractional viscoelastic models (see Freed & Diethelm, 2007).
    """
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        is_fractional = lower.startswith("alpha") or lower.startswith("beta")
        if not is_fractional:
            continue
        if not (0.0 < value < 1.0):
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="fractional_orders",
                    message=(
                        f"Fractional order '{name}' = {value:.4g} is outside (0, 1). "
                        "Fractional viscoelastic exponents must satisfy 0 < α, β < 1 "
                        "for thermodynamic admissibility."
                    ),
                    severity="error",
                )
            )
    return violations


def _check_prony_positive(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """All Prony-series modulus weights (G_i, E_i) must be strictly positive.

    Negative Prony weights violate the fading-memory condition and produce
    non-monotone relaxation spectra.
    """
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        # Match indexed Prony weights: G_1, G_2, ..., G_i, E_1, E_2, ...
        # Pattern: letter G or E followed by underscore and digit(s),
        # or the same without underscore (G1, G2, E1, E2).
        is_prony_weight = False
        if lower.startswith("g_") and lower[2:].isdigit():
            is_prony_weight = True
        elif lower.startswith("e_") and lower[2:].isdigit():
            is_prony_weight = True
        elif lower.startswith("g") and lower[1:].isdigit():
            is_prony_weight = True
        elif lower.startswith("e") and lower[1:].isdigit():
            is_prony_weight = True
        elif lower.startswith("g_i") or lower.startswith("e_i"):
            # Symbolic names like g_i, e_i
            is_prony_weight = True

        if not is_prony_weight:
            continue

        if value <= 0.0:
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="prony_positive",
                    message=(
                        f"Prony weight '{name}' = {value:.4g} is non-positive. "
                        "All Prony-series modulus contributions must be positive "
                        "to ensure a non-negative relaxation spectrum."
                    ),
                    severity="error",
                )
            )
    return violations


def _check_power_law_range(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Power-law index n should lie in (0, 2).

    Values outside this range are physically implausible for shear-thinning or
    mildly shear-thickening fluids, though they are not thermodynamically
    forbidden.  Emitted as a warning.
    """
    violations: list[PhysicsViolation] = []
    for name, (value, _bounds) in param_map.items():
        lower = name.lower()
        # Match: n, n_pl, n_power, power_law_index — but not "nu", "n_modes", etc.
        is_power_law_n = lower in {"n", "n_pl", "n_power", "power_law_index", "flow_index"}
        if not is_power_law_n:
            continue
        if not (0.0 < value < 2.0):
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="power_law_range",
                    message=(
                        f"Power-law index '{name}' = {value:.4g} is outside (0, 2). "
                        "Flow indices outside this range are physically implausible "
                        "for most polymer melts and solutions."
                    ),
                    severity="warning",
                )
            )
    return violations


def _check_plausibility(
    param_map: dict[str, tuple[float, tuple[float, float] | None]],
) -> list[PhysicsViolation]:
    """Warn when a parameter has converged to within 0.1 % of one of its bounds.

    Optimisers that saturate a bound typically indicate a model mis-specification
    or an inadequate parameter range rather than a genuine physical optimum.
    Only parameters with finite, non-degenerate bounds are checked.
    """
    violations: list[PhysicsViolation] = []
    for name, (value, bounds) in param_map.items():
        if bounds is None:
            continue
        lo, hi = bounds
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        span = hi - lo
        if span <= 0.0:
            # Degenerate (fixed) bound — nothing meaningful to check.
            continue
        tol = _BOUND_PROXIMITY_FRACTION * span
        at_lower = value <= lo + tol
        at_upper = value >= hi - tol
        if at_lower or at_upper:
            which = "lower" if at_lower else "upper"
            bound_value = lo if at_lower else hi
            violations.append(
                PhysicsViolation(
                    parameter=name,
                    value=value,
                    check="at_bound",
                    message=(
                        f"Parameter '{name}' = {value:.4g} is within 0.1 % of its "
                        f"{which} bound ({bound_value:.4g}). "
                        "The optimiser may have saturated the bound; consider widening "
                        "the parameter range or checking model specification."
                    ),
                    severity="warning",
                )
            )
    return violations


# ---------------------------------------------------------------------------
# Strategy registry — checkers applied in order
# ---------------------------------------------------------------------------

_CHECKERS = [
    _check_positive_moduli,
    _check_positive_times,
    _check_positive_viscosity,
    _check_fractional_orders,
    _check_prony_positive,
    _check_power_law_range,
    _check_plausibility,
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def check_fit_physics(
    model: Any,
    result: FitResult | None = None,
) -> list[PhysicsViolation]:
    """Run all physics checkers against a fitted model.

    Parameters are read from ``model.parameters`` (a ``ParameterSet``).
    When ``result`` is supplied, its ``result.params`` dict overrides the
    model's stored values so that the freshest fitted values are always used.

    The function is side-effect free.  It logs a DEBUG summary and returns
    a list of :class:`PhysicsViolation` objects; it never calls
    ``warnings.warn`` — that decision belongs to the caller.

    Args:
        model: Any fitted RheoJAX model with a ``.parameters`` attribute.
        result: Optional :class:`~rheojax.core.fit_result.FitResult` from
                ``model.fit(..., return_result=True)``.  When provided, its
                ``params`` mapping overrides the model-level values.

    Returns:
        List of :class:`PhysicsViolation` objects, possibly empty.  Violations
        with ``severity == "error"`` indicate thermodynamically inadmissible
        parameters; ``severity == "warning"`` indicate implausible but not
        strictly forbidden values.

    Example::

        import warnings
        from rheojax.io._exceptions import RheoJaxPhysicsWarning
        from rheojax.utils.physics_checks import check_fit_physics

        model = Maxwell()
        model.fit(t, G_data)
        violations = check_fit_physics(model)
        for v in violations:
            warnings.warn(v.message, RheoJaxPhysicsWarning, stacklevel=2)
    """
    model_name = type(model).__name__
    logger.debug("Starting physics validation", model=model_name)

    param_map = _build_param_map(model, result)

    if not param_map:
        logger.debug(
            "No parameters found for physics validation; returning empty list",
            model=model_name,
        )
        return []

    all_violations: list[PhysicsViolation] = []
    for checker in _CHECKERS:
        try:
            found = checker(param_map)
            all_violations.extend(found)
        except Exception:
            # A buggy checker must never crash the caller.  Log at WARNING
            # so that developers notice, but let the remaining checkers run.
            logger.warning(
                "Physics checker raised an unexpected exception; skipping",
                checker=checker.__name__,
                model=model_name,
                exc_info=True,
            )

    n_errors = sum(1 for v in all_violations if v.severity == "error")
    n_warnings = sum(1 for v in all_violations if v.severity == "warning")

    logger.debug(
        "Physics validation complete",
        model=model_name,
        n_parameters=len(param_map),
        n_errors=n_errors,
        n_warnings=n_warnings,
    )

    if all_violations:
        logger.info(
            "Physics violations found after fit",
            model=model_name,
            n_errors=n_errors,
            n_warnings=n_warnings,
        )

    return all_violations
