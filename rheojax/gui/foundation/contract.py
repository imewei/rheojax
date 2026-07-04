from __future__ import annotations

from dataclasses import dataclass, field

from rheojax.core.registry import ModelRegistry


@dataclass(frozen=True)
class ColumnSpec:
    role: str
    unit: str


@dataclass(frozen=True)
class InputContract:
    protocol: str
    columns: list[ColumnSpec]
    domain: str
    y_quantity: str | None = None
    unit_conversions: dict[str, str] = field(default_factory=dict)
    control_vars: list[str] = field(default_factory=list)


# Per-protocol column/domain templates (shear-only; from data_formats_by_protocol.md §2)
_TEMPLATES: dict[str, dict] = {
    "flow_curve": {
        "columns": [("shear_rate", "1/s")],
        "domain": "time",
        "control_vars": [],
    },
    "creep": {
        "columns": [("time", "s"), ("strain", "-")],
        "domain": "time",
        "control_vars": ["sigma0"],
    },
    "relaxation": {
        "columns": [("time", "s"), ("G_t", "Pa")],
        "domain": "time",
        "control_vars": ["gamma0"],
    },
    "startup": {
        "columns": [("time", "s"), ("stress", "Pa")],
        "domain": "time",
        "control_vars": ["gamma_dot0"],
    },
    "oscillation": {
        "columns": [("omega", "rad/s"), ("G_prime", "Pa"), ("G_double_prime", "Pa")],
        "domain": "frequency",
        "unit_conversions": {"x": "Hz->rad/s"},
        "control_vars": ["gamma0"],
    },
    "laos": {
        "columns": [("time", "s"), ("stress", "Pa")],
        "domain": "time",
        "control_vars": ["omega", "gamma0"],
    },
}


def input_contract(protocol: str, model_key: str | None = None) -> InputContract:
    if protocol not in _TEMPLATES:
        raise ValueError(f"unknown protocol: {protocol}")
    t = _TEMPLATES[protocol]
    y_quantity = None
    if protocol == "flow_curve":
        if model_key is not None:
            y_quantity = getattr(ModelRegistry.create(model_key), "flow_quantity", None)
        else:
            y_quantity = "stress"  # default per spec §8: every flow_curve contract needs a y-column
        cols = [
            ColumnSpec("shear_rate", "1/s"),
            ColumnSpec(
                "viscosity" if y_quantity == "viscosity" else "stress",
                "Pa·s" if y_quantity == "viscosity" else "Pa",
            ),
        ]
    else:
        cols = [ColumnSpec(role, unit) for role, unit in t["columns"]]
    return InputContract(
        protocol=protocol,
        columns=cols,
        domain=t["domain"],
        y_quantity=y_quantity,
        unit_conversions=dict(t.get("unit_conversions", {})),
        control_vars=list(t["control_vars"]),
    )
