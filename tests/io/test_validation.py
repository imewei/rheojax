"""Tests for rheojax.io.readers._validation.

All tests are marked smoke so they run in the fast CI gate.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io._exceptions import RheoJaxValidationWarning
from rheojax.io.readers._validation import LoaderReport, validate_protocol

# =============================================================================
# Fixtures
# =============================================================================


def _relaxation_data(monotonic: bool = True, t_start_fraction: float = 0.0) -> RheoData:
    """Create synthetic relaxation data."""
    t = np.linspace(0.0, 10.0, 50)
    if not monotonic:
        rng = np.random.default_rng(0)
        G = 1000.0 * np.exp(-t) + rng.normal(0, 200, len(t))
        G = np.abs(G)
    else:
        G = 1000.0 * np.exp(-t)

    if t_start_fraction > 0:
        # Shift t so it starts later (large relative to range)
        t = t + t_start_fraction * (t[-1] - t[0])

    return RheoData(
        x=t,
        y=G,
        x_units="s",
        y_units="Pa",
        domain="time",
        initial_test_mode="relaxation",
    )


def _oscillation_data(n_decades: float = 3.0) -> RheoData:
    """Create synthetic oscillation data spanning n_decades."""
    omega_min = 1e-2
    omega_max = omega_min * 10**n_decades
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), 40)
    G_star = (100.0 * omega + 1j * 50.0 * omega).astype(np.complex128)
    return RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency",
        initial_test_mode="oscillation",
    )


def _creep_data(include_sigma: bool = True) -> RheoData:
    t = np.linspace(0.0, 100.0, 50)
    J = 1e-4 * (1 - np.exp(-t / 10.0))
    meta = {"sigma_applied": 100.0} if include_sigma else {}
    return RheoData(
        x=t,
        y=J,
        x_units="s",
        y_units="1/Pa",
        domain="time",
        initial_test_mode="creep",
        metadata=meta,
    )


def _rotation_data(include_rate: bool = True) -> RheoData:
    gdot = np.logspace(-2, 2, 20)
    eta = 10.0 * gdot**-0.5
    meta = {"gamma_dot": gdot} if include_rate else {}
    return RheoData(
        x=gdot,
        y=eta,
        x_units="1/s",
        y_units="Pa.s",
        domain="time",
        initial_test_mode="rotation",
        metadata=meta,
    )


# =============================================================================
# LoaderReport dataclass
# =============================================================================


@pytest.mark.smoke
def test_loader_report_defaults():
    report = LoaderReport()
    assert report.warnings == []
    assert report.errors == []
    assert report.skipped_rows == 0
    assert report.protocol_inferred is False
    assert report.units_converted == {}
    assert report.quality_flags == {}


@pytest.mark.smoke
def test_loader_report_custom_values():
    report = LoaderReport(
        warnings=["w1"],
        errors=["e1"],
        skipped_rows=3,
        protocol_inferred=True,
        units_converted={"omega": "Hz"},
        quality_flags={"monotonic_decay": True},
    )
    assert report.skipped_rows == 3
    assert report.protocol_inferred is True
    assert report.units_converted == {"omega": "Hz"}
    assert report.quality_flags["monotonic_decay"] is True


# =============================================================================
# validate_protocol: relaxation
# =============================================================================


@pytest.mark.smoke
def test_relaxation_monotonic_passes():
    data = _relaxation_data(monotonic=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) == 0
    assert report.quality_flags.get("monotonic_decay") is True


@pytest.mark.smoke
def test_relaxation_non_monotonic_warns():
    data = _relaxation_data(monotonic=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert any("monoton" in str(w.message).lower() for w in validation_warns)
    assert report.quality_flags.get("monotonic_decay") is False


@pytest.mark.smoke
def test_relaxation_late_start_warns():
    # t_start / t_range > 0.5 → should warn
    data = _relaxation_data(monotonic=True, t_start_fraction=0.6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert any("transient" in str(w.message).lower() or "start" in str(w.message).lower() for w in validation_warns)
    assert report.quality_flags.get("early_transient_present") is False


# =============================================================================
# validate_protocol: oscillation
# =============================================================================


@pytest.mark.smoke
def test_oscillation_sufficient_range_passes():
    data = _oscillation_data(n_decades=3.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) == 0
    assert report.quality_flags.get("frequency_range_sufficient") is True


@pytest.mark.smoke
def test_oscillation_insufficient_range_warns():
    data = _oscillation_data(n_decades=1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert any("decade" in str(w.message).lower() for w in validation_warns)
    assert report.quality_flags.get("frequency_range_sufficient") is False


# =============================================================================
# validate_protocol: creep
# =============================================================================


@pytest.mark.smoke
def test_creep_with_sigma_passes():
    data = _creep_data(include_sigma=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) == 0
    assert report.quality_flags.get("sigma_metadata_present") is True


@pytest.mark.smoke
def test_creep_missing_sigma_warns():
    data = _creep_data(include_sigma=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert any("sigma" in str(w.message).lower() or "stress" in str(w.message).lower() for w in validation_warns)
    assert report.quality_flags.get("sigma_metadata_present") is False


# =============================================================================
# validate_protocol: rotation
# =============================================================================


@pytest.mark.smoke
def test_rotation_with_rate_passes():
    data = _rotation_data(include_rate=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) == 0
    assert report.quality_flags.get("shear_rate_metadata_present") is True


@pytest.mark.smoke
def test_rotation_missing_rate_warns():
    data = _rotation_data(include_rate=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert report.quality_flags.get("shear_rate_metadata_present") is False


# =============================================================================
# validate_protocol: intended_transform
# =============================================================================


@pytest.mark.smoke
def test_validate_with_compatible_transform():
    # oscillation + mastercurve: need temperature in metadata
    omega = np.logspace(-2, 2, 40)
    G_star = (100.0 * omega + 1j * 50.0 * omega).astype(np.complex128)
    data = RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency",
        initial_test_mode="oscillation",
        metadata={"temperature": 25.0},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data, intended_transform="mastercurve")
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    # Should pass all checks (freq range OK, temperature present, test_mode consistent)
    assert len(validation_warns) == 0


@pytest.mark.smoke
def test_validate_with_incompatible_transform():
    # relaxation data + mastercurve transform: domain mismatch + test_mode conflict
    data = _relaxation_data(monotonic=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data, intended_transform="mastercurve")
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert len(report.warnings) >= 1


# =============================================================================
# validate_protocol: edge cases
# =============================================================================


@pytest.mark.smoke
def test_validate_unknown_test_mode_warns():
    t = np.linspace(0, 10, 20)
    y = np.ones(20)
    data = RheoData(
        x=t,
        y=y,
        x_units="s",
        y_units="Pa",
        domain="time",
        initial_test_mode="unknown_protocol",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert any("unknown" in str(w.message).lower() for w in validation_warns)


@pytest.mark.smoke
def test_validate_none_test_mode_warns():
    t = np.linspace(0, 10, 20)
    y = np.ones(20)
    # No initial_test_mode, no metadata key
    data = RheoData(
        x=t,
        y=y,
        x_units="s",
        y_units="Pa",
        domain="time",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = validate_protocol(data)
    validation_warns = [w for w in caught if issubclass(w.category, RheoJaxValidationWarning)]
    assert len(validation_warns) >= 1
    assert report.protocol_inferred is True


@pytest.mark.smoke
def test_validate_empty_data_returns_error():
    # RheoData requires non-empty x and y, so test with minimal 1-point data
    # and check that the report is returned without crashing
    t = np.array([1.0])
    y = np.array([100.0])
    data = RheoData(
        x=t,
        y=y,
        x_units="s",
        y_units="Pa",
        domain="time",
        initial_test_mode="relaxation",
    )
    report = validate_protocol(data)
    # Only 1 point: diff checks skipped, no crash
    assert isinstance(report, LoaderReport)
