"""Protocol-aware validation for rheological data loaded from files.

This module provides opt-in quality checks that can be run after loading data
with any reader. Checks are protocol-specific (relaxation, creep, oscillation,
rotation/flow_curve, startup) and emit :class:`RheoJaxValidationWarning` for
each issue found.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rheojax.io._exceptions import RheoJaxValidationWarning
from rheojax.io.readers._utils import validate_transform
from rheojax.logging import get_logger

if TYPE_CHECKING:
    from rheojax.core.data import RheoData

logger = get_logger(__name__)

__all__ = ["LoaderReport", "validate_protocol"]


# =============================================================================
# Report dataclass
# =============================================================================


@dataclass
class LoaderReport:
    """Summary of issues and metadata collected during loading/validation.

    Attributes:
        warnings: Non-fatal data quality messages.
        errors: Fatal issues that prevent reliable analysis.
        skipped_rows: Number of rows discarded during parsing.
        protocol_inferred: True when test mode was inferred (not explicit).
        units_converted: Mapping of field -> original unit for converted values.
        quality_flags: Named boolean flags for downstream consumers.
    """

    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped_rows: int = 0  # populated by caller if NaN rows were dropped during loading
    protocol_inferred: bool = False
    units_converted: dict[str, str] = field(default_factory=dict)
    quality_flags: dict[str, bool] = field(default_factory=dict)


# =============================================================================
# Helpers
# =============================================================================


def _infer_protocol(data: RheoData) -> tuple[str | None, bool]:
    """Return (protocol, was_inferred).

    Checks ``data.initial_test_mode`` first (set as ``_explicit_test_mode``
    internally), then falls back to ``metadata["detected_test_mode"]``.
    """
    # initial_test_mode is an InitVar; after __post_init__ it is stored in
    # _explicit_test_mode AND metadata["detected_test_mode"].
    explicit = getattr(data, "_explicit_test_mode", None)
    if explicit is not None:
        return explicit, False

    detected = data.metadata.get("detected_test_mode")
    if detected is not None:
        return detected, True

    # Last resort: "test_mode" key (some readers set this without the
    # "detected_" prefix)
    fallback = data.metadata.get("test_mode")
    if fallback is not None:
        return fallback, True

    return None, True


# =============================================================================
# Per-protocol checks
# =============================================================================


def _check_relaxation(data: RheoData, report: LoaderReport) -> None:
    """Quality checks for relaxation modulus G(t) data."""
    y = np.asarray(data.y)
    if np.iscomplexobj(y):
        y = np.abs(y)

    if y.ndim == 2:
        y = y[:, 0]

    # Check for monotonic decay: majority of diffs should be <= 0
    if len(y) >= 3:
        diffs = np.diff(y)
        n_increasing = int(np.sum(diffs > 0))
        fraction_increasing = n_increasing / len(diffs)
        if fraction_increasing > 0.3:
            msg = (
                f"Relaxation data is not monotonically decaying: "
                f"{n_increasing}/{len(diffs)} steps are increasing "
                f"({fraction_increasing:.1%}). Data may be noisy or "
                f"incorrectly labelled."
            )
            report.warnings.append(msg)
            report.quality_flags["monotonic_decay"] = False
            warnings.warn(msg, RheoJaxValidationWarning, stacklevel=3)
            logger.debug(
                "Relaxation monotonicity check failed",
                n_increasing=n_increasing,
                total_steps=len(diffs),
            )
        else:
            report.quality_flags["monotonic_decay"] = True

    # Check that t[0] is not too large relative to the total time range
    x = np.asarray(data.x)
    if len(x) >= 2:
        t_range = float(x[-1] - x[0])
        t_start = float(x[0])
        if t_range > 0 and t_start / t_range > 0.5:
            msg = (
                f"Relaxation data starts at t={t_start:.3g} which is "
                f"{t_start / t_range:.1%} of the total time range "
                f"({t_range:.3g}). Early transient may be missing."
            )
            report.warnings.append(msg)
            report.quality_flags["early_transient_present"] = False
            warnings.warn(msg, RheoJaxValidationWarning, stacklevel=3)
            logger.debug(
                "Relaxation start-time check failed",
                t_start=t_start,
                t_range=t_range,
            )
        else:
            report.quality_flags["early_transient_present"] = True


def _check_creep(data: RheoData, report: LoaderReport) -> None:
    """Quality checks for creep compliance J(t) data."""
    meta = data.metadata
    has_stress = any(k in meta for k in ("sigma_applied", "sigma_0", "stress_applied"))
    if not has_stress:
        msg = (
            "Creep data is missing applied stress metadata. "
            "Expected 'sigma_applied' or 'sigma_0' in metadata for "
            "accurate compliance scaling."
        )
        report.warnings.append(msg)
        report.quality_flags["sigma_metadata_present"] = False
        warnings.warn(msg, RheoJaxValidationWarning, stacklevel=4)
        logger.debug(
            "Creep sigma metadata check failed", metadata_keys=list(meta.keys())
        )
    else:
        report.quality_flags["sigma_metadata_present"] = True


def _check_oscillation(data: RheoData, report: LoaderReport) -> None:
    """Quality checks for oscillatory (SAOS/MAOS) data."""
    x = np.asarray(data.x)
    if len(x) >= 2:
        x_pos = x[x > 0]
        if len(x_pos) >= 2:
            decades = float(np.log10(x_pos.max() / x_pos.min()))
            if decades < 2.0:
                msg = (
                    f"Oscillation frequency range spans only {decades:.2f} decades "
                    f"(min={x_pos.min():.3g}, max={x_pos.max():.3g} rad/s). "
                    f"At least 2 decades are recommended for reliable fitting."
                )
                report.warnings.append(msg)
                report.quality_flags["frequency_range_sufficient"] = False
                warnings.warn(msg, RheoJaxValidationWarning, stacklevel=3)
                logger.debug(
                    "Oscillation frequency range check failed",
                    decades=decades,
                    omega_min=float(x_pos.min()),
                    omega_max=float(x_pos.max()),
                )
            else:
                report.quality_flags["frequency_range_sufficient"] = True
        else:
            msg = "Oscillation data has fewer than 2 positive frequency points."
            report.warnings.append(msg)
            report.quality_flags["frequency_range_sufficient"] = False
            warnings.warn(msg, RheoJaxValidationWarning, stacklevel=3)


def _check_rotation(data: RheoData, report: LoaderReport) -> None:
    """Quality checks for steady-state flow curve (rotation) data."""
    meta = data.metadata
    has_rate = any(k in meta for k in ("gamma_dot", "shear_rate", "applied_shear_rate"))
    if not has_rate:
        msg = (
            "Flow curve (rotation) data is missing shear rate metadata. "
            "Expected 'gamma_dot' or 'shear_rate' in metadata."
        )
        report.warnings.append(msg)
        report.quality_flags["shear_rate_metadata_present"] = False
        warnings.warn(msg, RheoJaxValidationWarning, stacklevel=4)
        logger.debug(
            "Rotation shear-rate metadata check failed",
            metadata_keys=list(meta.keys()),
        )
    else:
        report.quality_flags["shear_rate_metadata_present"] = True


def _check_startup(data: RheoData, report: LoaderReport) -> None:
    """Quality checks for startup-of-flow data."""
    meta = data.metadata
    has_rate = any(k in meta for k in ("gamma_dot", "shear_rate"))
    if not has_rate:
        msg = (
            "Startup data is missing applied shear rate metadata. "
            "Expected 'gamma_dot' or 'shear_rate' in metadata."
        )
        report.warnings.append(msg)
        report.quality_flags["shear_rate_metadata_present"] = False
        warnings.warn(msg, RheoJaxValidationWarning, stacklevel=4)
        logger.debug(
            "Startup shear-rate metadata check failed",
            metadata_keys=list(meta.keys()),
        )
    else:
        report.quality_flags["shear_rate_metadata_present"] = True


# =============================================================================
# Public API
# =============================================================================

_PROTOCOL_CHECKERS = {
    "relaxation": _check_relaxation,
    "creep": _check_creep,
    "oscillation": _check_oscillation,
    "rotation": _check_rotation,
    "flow_curve": _check_rotation,  # alias
    "startup": _check_startup,
}


def validate_protocol(
    data: RheoData,
    intended_transform: str | None = None,
) -> LoaderReport:
    """Run protocol-aware quality checks on loaded rheological data.

    Infers the test protocol from ``data.initial_test_mode`` or
    ``data.metadata["detected_test_mode"]`` and performs protocol-specific
    quality checks. A :class:`RheoJaxValidationWarning` is emitted for every
    issue found so that callers using ``warnings.filterwarnings`` can
    control visibility.

    Args:
        data: Loaded rheological data container.
        intended_transform: Optional transform name (e.g. ``"mastercurve"``,
            ``"owchirp"``) to validate transform compatibility in addition to
            protocol checks.

    Returns:
        :class:`LoaderReport` with all findings.
    """
    report = LoaderReport()

    # Guard: empty data
    x = np.asarray(data.x)
    if x.size == 0:
        msg = "Data is empty (zero points). No validation performed."
        report.errors.append(msg)
        logger.debug("validate_protocol: empty data, skipping all checks")
        return report

    # Infer protocol
    protocol, was_inferred = _infer_protocol(data)
    report.protocol_inferred = was_inferred

    logger.debug(
        "validate_protocol: protocol resolved",
        protocol=protocol,
        inferred=was_inferred,
    )

    if protocol is None:
        msg = (
            "Could not determine test protocol from data. "
            "Pass 'initial_test_mode' to the reader for reliable validation."
        )
        report.warnings.append(msg)
        warnings.warn(msg, RheoJaxValidationWarning, stacklevel=2)
    else:
        checker = _PROTOCOL_CHECKERS.get(protocol)
        if checker is not None:
            checker(data, report)
        else:
            msg = (
                f"Unknown protocol '{protocol}'. "
                f"Supported: {sorted(_PROTOCOL_CHECKERS.keys())}"
            )
            report.warnings.append(msg)
            warnings.warn(msg, RheoJaxValidationWarning, stacklevel=2)

    # Optional transform validation
    if intended_transform is not None:
        transform_warnings = validate_transform(
            intended_transform=intended_transform,
            domain=data.domain,
            metadata=data.metadata,
            test_mode=protocol,
        )
        for tw in transform_warnings:
            report.warnings.append(tw)
            warnings.warn(tw, RheoJaxValidationWarning, stacklevel=2)

    logger.debug(
        "validate_protocol: complete",
        protocol=protocol,
        n_warnings=len(report.warnings),
        n_errors=len(report.errors),
        quality_flags=report.quality_flags,
    )

    return report
