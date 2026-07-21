"""Tests for lightweight shared option validation."""

from types import MappingProxyType

import numpy as np
import pytest

from rheojax.core._validation import reject_removed_options, validate_predict_input
from rheojax.core.post_fit_validator import PostFitValidator, RheoJaxUncertaintyWarning
from rheojax.io._exceptions import RheoJaxPhysicsWarning


@pytest.mark.parametrize(
    "removed_key",
    ("deformation_mode", "poisson_ratio"),
)
def test_reject_removed_options_names_single_offending_option(removed_key):
    with pytest.raises(TypeError) as exc_info:
        reject_removed_options({removed_key: object()})

    message = str(exc_info.value)
    assert f"Removed option(s) '{removed_key}'" in message
    assert "shear-only" in message
    assert "Remove them." in message


def test_reject_removed_options_orders_multiple_offending_options():
    options = {"poisson_ratio": 0.5, "deformation_mode": "tension"}

    with pytest.raises(TypeError) as exc_info:
        reject_removed_options(options)

    message = str(exc_info.value)
    assert "'deformation_mode'" in message
    assert "'poisson_ratio'" in message
    assert "shear-only" in message
    assert "Remove them." in message


def test_reject_removed_options_accepts_mapping_without_mutating_it():
    options = {"method": "scipy", "max_iter": 23}
    readonly_options = MappingProxyType(options)

    reject_removed_options(readonly_options)

    assert options == {"method": "scipy", "max_iter": 23}


def test_check_physics_surfaces_failure_via_warning_and_log(monkeypatch, caplog):
    """A bug in check_fit_physics must not silently defeat check_physics=True."""

    def _boom(model):
        raise AttributeError("model missing expected field")

    monkeypatch.setattr(
        "rheojax.utils.physics_checks.check_fit_physics", _boom
    )

    with caplog.at_level("WARNING", logger="rheojax.core.post_fit_validator"):
        with pytest.warns(RheoJaxPhysicsWarning, match="check_physics failed to run"):
            PostFitValidator.check_physics(object())

    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_compute_uncertainty_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown uncertainty method"):
        PostFitValidator.compute_uncertainty(
            model=object(),
            X=None,
            y=None,
            uncertainty="bootstrp",
            test_mode=None,
        )


def test_validate_predict_input_rejects_nan():
    with pytest.raises(ValueError, match="NaN"):
        validate_predict_input(np.array([1.0, np.nan, 3.0]))


def test_validate_predict_input_rejects_non_monotonic():
    with pytest.raises(ValueError, match="monotonic"):
        validate_predict_input(np.array([1.0, 3.0, 2.0]))


def test_validate_predict_input_rejects_empty_array():
    """Empty arrays must not silently skip validation (CRITICAL fix)."""
    with pytest.raises(ValueError, match="empty"):
        validate_predict_input(np.array([]))


def test_validate_predict_input_accepts_valid_array():
    validate_predict_input(np.array([1.0, 2.0, 3.0]))


def test_compute_uncertainty_surfaces_failure_via_warning_and_log(monkeypatch, caplog):
    """A bug in hessian_ci must not silently return None with no visible signal."""

    def _boom(model, X, y, test_mode=None):
        raise AttributeError("model missing expected field")

    monkeypatch.setattr("rheojax.utils.uncertainty.hessian_ci", _boom)

    with caplog.at_level("WARNING", logger="rheojax.core.post_fit_validator"):
        with pytest.warns(
            RheoJaxUncertaintyWarning, match="Uncertainty computation .* failed to run"
        ):
            result = PostFitValidator.compute_uncertainty(
                model=object(),
                X=None,
                y=None,
                uncertainty="hessian",
                test_mode=None,
            )

    assert result is None
    assert any(record.levelname == "WARNING" for record in caplog.records)
