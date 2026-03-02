"""Tests for TransformService parameter spec completeness."""

from rheojax.gui.services.transform_service import TransformService


def test_all_params_have_label_and_type():
    """Every param spec must have 'label' and 'type' keys."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            assert "type" in spec, f"{transform_key}.{param_name} missing 'type'"
            assert "label" in spec, f"{transform_key}.{param_name} missing 'label'"
            assert "default" in spec, f"{transform_key}.{param_name} missing 'default'"


def test_numeric_params_have_range():
    """Float and int params must have a 'range' tuple."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            if spec["type"] in ("float", "int"):
                assert "range" in spec, (
                    f"{transform_key}.{param_name} (type={spec['type']}) missing 'range'"
                )
                assert len(spec["range"]) == 2, (
                    f"{transform_key}.{param_name} 'range' must be (min, max) tuple"
                )


def test_choice_params_have_choices():
    """Choice params must have a 'choices' list."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            if spec["type"] == "choice":
                assert "choices" in spec, (
                    f"{transform_key}.{param_name} missing 'choices'"
                )
                assert len(spec["choices"]) >= 2, (
                    f"{transform_key}.{param_name} must have at least 2 choices"
                )


def test_mastercurve_has_shift_method():
    """Mastercurve must include shift_method param (was in UI but missing from service)."""
    service = TransformService()
    params = service.get_transform_params("mastercurve")
    assert "shift_method" in params
    assert params["shift_method"]["type"] == "choice"


def test_derivative_has_mode():
    """Derivative must include mode (padding) param."""
    service = TransformService()
    params = service.get_transform_params("derivative")
    assert "mode" in params
    assert params["mode"]["type"] == "choice"


def test_get_transform_metadata_returns_all_transforms():
    """get_transform_metadata() returns metadata for all registered transforms."""
    service = TransformService()
    metadata = service.get_transform_metadata()
    keys = service.get_available_transforms()
    assert len(metadata) == len(keys)
    for entry in metadata:
        assert "key" in entry
        assert "name" in entry
        assert "description" in entry
        assert "requires_multiple" in entry
        assert isinstance(entry["requires_multiple"], bool)


def test_multi_dataset_transforms_flagged():
    """Mastercurve and SRFS must be flagged as requires_multiple."""
    service = TransformService()
    metadata = service.get_transform_metadata()
    multi = {m["key"] for m in metadata if m["requires_multiple"]}
    assert "mastercurve" in multi
    assert "srfs" in multi


import numpy as np
from rheojax.core.data import RheoData


def test_preview_returns_plot_data():
    """preview_transform returns x/y arrays for Before and After."""
    service = TransformService()
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
    data = RheoData(x=x, y=y)

    result = service.preview_transform(
        "derivative", data, {
            "order": 1, "window_length": 11, "poly_order": 3,
            "method": "savgol", "validate_window": True,
            "smooth_before": False, "smooth_after": False, "mode": "mirror",
        }
    )
    assert "x_before" in result
    assert "y_before" in result
    assert "x_after" in result
    assert "y_after" in result
    assert len(result["x_after"]) > 0
    assert "error" not in result


def test_preview_returns_error_on_failure():
    """preview_transform returns error dict on failure, not exception."""
    service = TransformService()
    # Use a non-existent transform name to guarantee apply_transform raises
    data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
    result = service.preview_transform("nonexistent_transform", data, {})
    assert "error" in result
    assert isinstance(result["error"], str)
