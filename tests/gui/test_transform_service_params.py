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
