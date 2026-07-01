from __future__ import annotations

from rheojax.gui.foundation.invalidation import invalidate_downstream
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.controller import FitController, Step
from rheojax.gui.workspace.fit.step1_protocol_model import ProtocolModelStep
from rheojax.gui.workspace.fit.step2_data import DataStep
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep
from rheojax.gui.workspace.fit.step4_nuts import NutsStep
from rheojax.gui.workspace.fit.step5_visualize import VisualizeStep
from rheojax.gui.workspace.fit.step6_export import ExportStep

# Which FitState field each step edits (for invalidate_downstream). None means
# that step's edits don't cascade (visualize is read-only; export just saves).
_CHANGED = ["model_key", "column_map", "nlsq_result", "nuts_result", None, None]


def build_fit_controller(app_state: AppState):
    st = app_state.fit
    bodies = [
        ProtocolModelStep(st),
        DataStep(st, app_state.library),
        NlsqStep(st),
        NutsStep(st),
        VisualizeStep(st),
        ExportStep(st, app_state.library),
    ]
    steps = [
        Step(
            id=FitController.STEP_IDS[i],
            title=FitController.STEP_IDS[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=lambda: True,
        )
        for i in range(len(bodies))
    ]
    ctl = FitController(steps)

    for i, body in enumerate(bodies):
        if hasattr(body, "edited"):

            def _on_edit(idx=i):
                ctl.on_edit(idx)
                changed = _CHANGED[idx]
                if changed:
                    new_fit = invalidate_downstream(app_state.fit, changed)
                    # Mutate in place so step bodies (which hold a reference to
                    # app_state.fit via `st`) see the cleared fields immediately.
                    # FitState is non-frozen, so setattr works without replace().
                    for attr, val in vars(new_fit).items():
                        setattr(app_state.fit, attr, val)

            body.edited.connect(_on_edit)

    # Wire NUTS finish -> Visualize refresh so Diagnostics tab appears after sampling
    nuts_body, visualize_body = bodies[3], bodies[4]
    if hasattr(nuts_body, "finished") and hasattr(visualize_body, "refresh"):
        nuts_body.finished.connect(visualize_body.refresh)

    return ctl, bodies
