import pytest

import rheojax.models  # noqa: ensure registration
from rheojax.core.registry import ModelRegistry


def test_all_flow_curve_models_have_flow_quantity():
    """Every flow_curve model must declare flow_quantity = 'viscosity' or 'stress'."""
    keys = ModelRegistry.find(protocol="flow_curve")
    assert len(keys) > 0, "no flow_curve models registered"
    for key in sorted(keys):
        model = ModelRegistry.create(key)
        assert hasattr(model, "flow_quantity"), f"{key} missing flow_quantity"
        assert model.flow_quantity in ("viscosity", "stress"), (
            f"{key}.flow_quantity = {model.flow_quantity!r}"
        )


@pytest.mark.parametrize(
    "key,quantity",
    [
        ("power_law", "viscosity"),
        ("cross", "viscosity"),
        ("carreau", "viscosity"),
        ("carreau_yasuda", "viscosity"),
        ("bingham", "stress"),
        ("herschel_bulkley", "stress"),
        ("spp_yield_stress", "stress"),
    ],
)
def test_spot_check_flow_quantity(key, quantity):
    assert ModelRegistry.create(key).flow_quantity == quantity
