import rheojax.models  # noqa: F401
from rheojax.gui.foundation.contract import input_contract


def test_oscillation_contract():
    c = input_contract("oscillation")
    roles = [col.role for col in c.columns]
    assert roles == ["omega", "G_prime", "G_double_prime"]
    assert c.domain == "frequency"
    assert c.unit_conversions.get("x") == "Hz->rad/s"


def test_flow_quantity_from_model():
    assert input_contract("flow_curve", "carreau").y_quantity == "viscosity"
    assert input_contract("flow_curve", "bingham").y_quantity == "stress"


def test_relaxation_contract():
    c = input_contract("relaxation")
    assert [col.role for col in c.columns] == ["time", "G_t"]
    assert c.domain == "time"


def test_flow_curve_default_has_y_column():
    c = input_contract("flow_curve")
    roles = [col.role for col in c.columns]
    assert "shear_rate" in roles
    assert "stress" in roles or "viscosity" in roles
