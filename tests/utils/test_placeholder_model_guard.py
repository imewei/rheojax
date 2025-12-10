"""Ensure placeholder model names do not hit the registry."""

from rheojax.gui.services.model_service import ModelService


def test_model_service_placeholders_are_ignored():
    svc = ModelService()

    info = svc.get_model_info("Select model...")
    assert info["parameters"] == {}
    assert "No model selected" in info.get("description", "")

    defaults = svc.get_parameter_defaults("Select model...")
    assert defaults == {}

    compat = svc.check_compatibility("Select model...", data=None, test_mode=None)  # type: ignore[arg-type]
    assert compat["compatible"] is False
    assert "No model selected" in compat["warnings"][0]

