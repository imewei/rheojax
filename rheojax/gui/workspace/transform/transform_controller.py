from __future__ import annotations

from dataclasses import replace

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.controller import Step, TransformController
from rheojax.gui.workspace.transform.step1_pick import TransformPickStep
from rheojax.gui.workspace.transform.step2_slots import SlotsStep
from rheojax.gui.workspace.transform.step3_run import RunStep
from rheojax.gui.workspace.transform.step4_visualize import TransformVisualizeStep
from rheojax.gui.workspace.transform.step5_export import TransformExportStep


def build_transform_controller(app_state: AppState):
    st = app_state.transform
    bodies = [
        TransformPickStep(st),
        SlotsStep(st, app_state.library),
        RunStep(st),
        TransformVisualizeStep(st),
        TransformExportStep(st, app_state.library),
    ]
    steps = [
        Step(
            id=TransformController.STEP_IDS[i],
            title=TransformController.STEP_IDS[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=lambda: True,
        )
        for i in range(len(bodies))
    ]
    ctl = TransformController(steps)

    for i, body in enumerate(bodies):
        if hasattr(body, "edited"):

            def _on_edit(idx=i):
                ctl.on_edit(idx)
                if idx == 0:  # changing the transform clears slots/config/result
                    new_tx = replace(
                        app_state.transform,
                        slots={},
                        config={},
                        result=None,
                        revision=app_state.transform.revision + 1,
                    )
                    # Mutate in place so step bodies (which hold a reference
                    # to app_state.transform via `st`) see the cleared fields
                    # immediately. TransformState is non-frozen, so setattr
                    # works without reassigning app_state.transform (mirrors
                    # fit_controller.py's identical fix for the same hazard).
                    for attr, val in vars(new_tx).items():
                        setattr(app_state.transform, attr, val)

            body.edited.connect(_on_edit)

    # Wire TransformPick edits -> Slots refresh, so `_specs` (frozen at
    # transform_key=None construction time) gets rebuilt against the picked
    # transform. Connected after the _on_edit loop above, so the
    # slots/config/result invalidation runs first and refresh() rebuilds
    # specs against the already-cleared state (mirrors fit_controller.py's
    # ProtocolModel -> DataStep.refresh wiring).
    pick_body, slots_body = bodies[0], bodies[1]
    if hasattr(pick_body, "edited") and hasattr(slots_body, "refresh"):
        pick_body.edited.connect(slots_body.refresh)

    # Wire Run finish -> Visualize refresh so the tabs reflect the completed
    # transform_key/result instead of the construction-time (None) state
    # (mirrors fit_controller.py's NutsStep.finished -> VisualizeStep.refresh).
    run_body, visualize_body = bodies[2], bodies[3]
    if hasattr(run_body, "finished") and hasattr(visualize_body, "refresh"):
        run_body.finished.connect(visualize_body.refresh)

    return ctl, bodies
