from __future__ import annotations

from dataclasses import replace

from rheojax.gui.foundation.state import FitState

# Which fit fields each upstream change invalidates (downstream-only).
_FIT_CASCADE = {
    "protocol":     ("data_ref", "column_map", "control_vars", "model_config", "nlsq_result", "nuts_result"),
    "model_key":    ("data_ref", "column_map", "control_vars", "model_config", "nlsq_result", "nuts_result"),
    "model_config": ("nlsq_result", "nuts_result"),
    "data_ref":     ("column_map", "nlsq_result", "nuts_result"),
    "column_map":   ("nlsq_result", "nuts_result"),
    "nlsq_result":  ("nuts_result",),
}
_CLEAR = {"column_map": dict, "control_vars": dict, "model_config": dict}

def invalidate_downstream(fit: FitState, changed: str) -> FitState:
    fields = _FIT_CASCADE.get(changed, ())
    updates = {f: (_CLEAR[f]() if f in _CLEAR else None) for f in fields}
    return replace(fit, revision=fit.revision + 1, **updates)
