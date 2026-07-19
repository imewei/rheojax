from __future__ import annotations

from dataclasses import replace

from rheojax.gui.foundation.state import FitState

# Which fit fields each upstream change invalidates (downstream-only).
_FIT_CASCADE = {
    "protocol": (
        "data_ref",
        "column_map",
        "control_vars",
        "model_config",
        "nlsq_result",
        "nuts_result",
    ),
    "model_key": (
        "data_ref",
        "column_map",
        "control_vars",
        "model_config",
        "nlsq_result",
        "nuts_result",
    ),
    "model_config": ("nlsq_result", "nuts_result"),
    "data_ref": ("column_map", "nlsq_result", "nuts_result"),
    "column_map": ("nlsq_result", "nuts_result"),
    "nlsq_result": ("nuts_result",),
}
_CLEAR = {"column_map": dict, "control_vars": dict, "model_config": dict}

# Which transform fields each upstream change invalidates (downstream-only).
# Mirrors _FIT_CASCADE's vocabulary: picking a new transform re-derives
# slots/config/result from scratch; refilling a slot only stales the result.
_TRANSFORM_CASCADE = {
    "transform_key": ("slots", "config", "result"),
    "slots": ("result",),
}
_TRANSFORM_CLEAR = {"slots": dict, "config": dict}


def invalidate_downstream(fit: FitState, changed: str) -> FitState:
    fields = _FIT_CASCADE.get(changed, ())
    updates = {f: (_CLEAR[f]() if f in _CLEAR else None) for f in fields}
    return replace(fit, revision=fit.revision + 1, **updates)


def apply_cascade(live_state, cascade_table: dict, clear_table: dict, changed: str) -> None:
    """Clear *live_state*'s downstream fields for *changed* and bump revision.

    Step-widget bodies hold a direct reference to the live FitState/
    TransformState instance (not to app_state.fit/app_state.transform), so a
    plain dataclasses.replace() return value would be invisible to them --
    this mutates the live object in place instead. Always bumps revision,
    even when *changed* has no cascade entry, matching invalidate_downstream's
    existing (tested) behavior: callers only invoke this when something was
    actually edited.
    """
    fields = cascade_table.get(changed, ())
    updates = {f: (clear_table[f]() if f in clear_table else None) for f in fields}
    new_state = replace(live_state, revision=live_state.revision + 1, **updates)
    for attr, val in vars(new_state).items():
        setattr(live_state, attr, val)


def register_step(
    body,
    signal_name,
    on_relock,
    *,
    changed=None,
    live_state=None,
    cascade_table=None,
    clear_table=None,
    downstream=(),
) -> None:
    """Wire body's *signal_name* signal to relock-then-cascade-then-refresh, in
    that fixed order, in one place.

    Previously each controller wired cascade and downstream refresh as
    separate .connect() calls, relying on registration order (undocumented
    except in prose comments) to keep refresh handlers from seeing stale
    state. Owning the order here removes the chance of a future edit
    reordering .connect() calls and silently reintroducing that class of bug.
    """
    signal = getattr(body, signal_name, None)
    if signal is None:
        return

    def _handle() -> None:
        on_relock()
        if changed and live_state is not None:
            apply_cascade(live_state, cascade_table or {}, clear_table or {}, changed)
        for fn in downstream:
            fn()

    signal.connect(_handle)
