from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotSpec:
    name: str
    accepts: str | None  # required protocol_type, or None = any
    is_list: bool


_MULTI = {"mastercurve", "srfs"}
_TYPED_PAIRS = {
    "cox_merz": [
        SlotSpec("oscillation", "oscillation", False),
        SlotSpec("flow_curve", "flow_curve", False),
    ],
}


def transform_slots(key: str) -> list[SlotSpec]:
    if key in _TYPED_PAIRS:
        return list(_TYPED_PAIRS[key])
    if key in _MULTI:
        return [SlotSpec("datasets", None, True)]
    return [SlotSpec("input", None, False)]
