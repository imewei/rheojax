from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QEventLoop, QThreadPool

from rheojax.core.registry import TransformRegistry
from rheojax.gui.foundation.state import AppState
from rheojax.gui.jobs.transform_worker import TransformWorker
from rheojax.gui.workspace.controller import Step, TransformController
from rheojax.gui.workspace.transform.slots_spec import transform_slots
from rheojax.gui.workspace.transform.step1_pick import TransformPickStep
from rheojax.gui.workspace.transform.step2_slots import SlotsStep
from rheojax.gui.workspace.transform.step3_run import RunStep
from rheojax.gui.workspace.transform.step4_visualize import TransformVisualizeStep
from rheojax.gui.workspace.transform.step5_export import TransformExportStep

_DOMAIN_CHANGING = {"spectral", "decomposition"}


def _resolve_slot_data(library, transform_key, slots):
    """Resolve filled slot ids to RheoData payloads, in transform_slots() order.

    Single slot -> one RheoData. is_list slot -> list[RheoData]. Multiple
    (typed-pair) slots -> list[RheoData] in spec order (matches cox_merz's
    positional [oscillation, flow_curve] expectation).
    """
    specs = transform_slots(transform_key)
    if len(specs) == 1 and not specs[0].is_list:
        return library.load_payload(slots[specs[0].name])
    if len(specs) == 1 and specs[0].is_list:
        return [library.load_payload(i) for i in slots[specs[0].name]]
    return [library.load_payload(slots[s.name]) for s in specs]


def _is_same_domain(transform_key: str) -> bool:
    info = TransformRegistry.get_info(transform_key)
    if info is None:
        return True  # unrecognized key -> conservative default, preserve type
    category = str(info.transform_type).split(".")[-1].lower()
    return category not in _DOMAIN_CHANGING


def _infer_protocol_type(library, transform_key, slots) -> str:
    """Same-domain transforms (superposition/processing/analysis) plausibly
    keep the input's protocol type; domain-changing transforms (spectral/
    decomposition), and any case where the input protocol can't be resolved,
    get "" (empty string, never None), matching the design spec's own
    documented fallback: such outputs are "stored but not offered to typed
    Fit slots" (design §7) -- `""` never equality-matches a real protocol
    string in `DatasetLibrary.datasets_of_type()`, so it's excluded from
    typed Fit slots while still being a valid, storable `DatasetRef`.
    Per-transform-exact output typing (e.g. does prony_conversion's spectrum
    count as "relaxation"?) is a refinement left for a follow-up once each of
    the 14 transforms' real output domain is confirmed -- this default only
    ever *under*-offers a derived dataset to Fit, never mis-tags one.
    """
    if not _is_same_domain(transform_key):
        return ""
    specs = transform_slots(transform_key)
    if not specs:
        return ""
    first = slots.get(specs[0].name)
    if isinstance(first, list):
        first = first[0] if first else None
    if not first:
        return ""
    return library.get(first).protocol_type


def _make_run_fn(library):
    def _run(transform_key, slots, config):

        data = _resolve_slot_data(library, transform_key, slots)
        worker = TransformWorker(transform_key, data, params=config)
        loop = QEventLoop()
        outcome: dict = {}

        def _on_completed(tr):
            outcome["result"] = tr
            loop.quit()

        def _on_failed(msg):
            outcome["error"] = msg
            loop.quit()

        def _on_cancelled():
            outcome["cancelled"] = True
            loop.quit()

        worker.signals.completed.connect(_on_completed)
        worker.signals.failed.connect(_on_failed)
        worker.signals.cancelled.connect(_on_cancelled)
        QThreadPool.globalInstance().start(worker)
        loop.exec()

        if "error" in outcome:
            raise RuntimeError(outcome["error"])
        if outcome.get("cancelled"):
            raise RuntimeError("Transform cancelled")

        tr = outcome["result"]
        return {
            # step4_visualize.py's "Input vs output" overlay tab reads
            # payload["input"] via _xy() (tolerant of the list shape from
            # is_list/typed-pair slots -- it just returns None for those,
            # same as a missing key). `data` is already resolved above;
            # without this the Input trace was silently never plotted.
            "input": data,
            "output": tr.data,
            "result": tr.extras,
            "protocol_type": _infer_protocol_type(library, transform_key, slots),
        }
    return _run


def build_transform_controller(app_state: AppState):
    st = app_state.transform
    bodies = [
        TransformPickStep(st),
        SlotsStep(st, app_state.library),
        RunStep(st, run_fn=_make_run_fn(app_state.library)),
        TransformVisualizeStep(st),
        TransformExportStep(st, app_state.library),
    ]
    validators = [
        lambda b=bodies[0]: b.is_ready(),
        lambda b=bodies[1]: b.is_ready(),
        lambda b=bodies[2]: b.is_ready(),
        lambda: True,   # Visualize is read-only
        lambda: True,   # Export just saves/writes
    ]
    steps = [
        Step(
            id=TransformController.STEP_IDS[i],
            title=TransformController.STEP_IDS[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=validators[i],
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
                elif idx == 1 and app_state.transform.result is not None:
                    # Refilling a slot clears only the stale result -- NOT
                    # slots/config, since slots is exactly the field the user
                    # is actively setting via SlotsStep.fill() right now.
                    # Without this, RunStep.is_ready() (result is not None)
                    # stays True after a slot change, and the window's
                    # _advance_and_unlock forward-unlock loop immediately
                    # re-adds Visualize/Export to `reached`, undoing the
                    # re-lock that ctl.on_edit(1) just applied and leaving a
                    # stale result reachable/exportable for a dataset that
                    # was never actually run.
                    app_state.transform.result = None
                    app_state.transform.revision += 1

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
