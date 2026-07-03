from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
import rheojax.transforms  # noqa: F401
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.transform.transform_controller import (
    build_transform_controller,
)


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _ref(i, protocol):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=protocol,
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_pick_fill_run_save_round_trips_through_library(qapp, monkeypatch):
    # End-to-end regression: a user picks a transform, fills a slot through
    # the real SlotsStep combo widget (not .fill() directly), clicks Run
    # (the real RunStep with the real run_fn built by
    # build_transform_controller, backed by TransformWorker ->
    # TransformService.apply_transform -- the actual compute boundary), and
    # saves the result to the library via the real TransformExportStep. This
    # proves the KeyError-eliminating chain the whole plan exists for: the
    # derived dataset registered by save_to_library() is actually reloadable
    # via library.load_payload(), not just individually testable per-step.
    def fake_apply_transform(self, name, data, params):
        return _RheoData([1.0, 2.0], [3.0, 4.0])

    monkeypatch.setattr(
        "rheojax.gui.services.transform_service.TransformService.apply_transform",
        fake_apply_transform,
    )

    library = DatasetLibrary()
    library.add(_ref("ds1", "oscillation"))
    library.store_payload("ds1", _RheoData([0.0, 1.0], [0.0, 1.0]))
    app_state = AppState(library=library)

    ctl, bodies = build_transform_controller(app_state)

    # TransformExportStep no longer commits to the library itself -- it only
    # requests a commit via dataset_commit_requested. Simulate the Task-10
    # WorkspaceWindow._commit_dataset handler that will perform it.
    def _handle(ref, payload, overwrite):
        app_state.library.add(ref, overwrite=overwrite)
        if payload is not None:
            app_state.library.store_payload(ref.id, payload)

    bodies[4].dataset_commit_requested.connect(_handle)

    # 1. Pick a single-slot transform ("smooth_derivative", not "derivative"
    #    -- the latter is a TransformService-only alias, not a real
    #    TransformRegistry entry).
    bodies[0].set_transform("smooth_derivative")

    # 2. Fill the slot through the real widget, driving the same
    #    currentTextChanged -> fill() path a user click would.
    combo = bodies[1]._combo_widgets["input"]
    combo.setCurrentText("ds1")
    assert app_state.transform.slots == {"input": "ds1"}

    # 3. Run the real RunStep with the real run_fn (TransformWorker ->
    #    TransformService.apply_transform, monkeypatched above).
    bodies[2].run()
    assert app_state.transform.result is not None
    assert app_state.transform.result["output"].x == [1.0, 2.0]
    # smooth_derivative is same-domain -> protocol_type carried through
    assert app_state.transform.result["protocol_type"] == "oscillation"

    # 4. Save to the library via the real TransformExportStep.
    new_id = bodies[4].save_to_library()
    assert new_id is not None

    # 5. The derived dataset must actually be reloadable -- the exact
    #    KeyError-elimination this whole plan exists to prove works
    #    end-to-end.
    ref = app_state.library.get(new_id)
    assert ref.protocol_type == "oscillation"
    assert ref.origin == "derived"
    payload = app_state.library.load_payload(new_id)
    assert payload.x == [1.0, 2.0]
    assert payload.y == [3.0, 4.0]
