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
    return DatasetRef(id=i, name=i, protocol_type=protocol, origin="imported",
                       units={}, row_count=1, hash="h", provenance={}, lineage=[])


def test_build_transform_controller_injects_real_run_fn(qapp, monkeypatch):
    def fake_apply_transform(self, name, data, params):
        return _RheoData([1.0], [2.0])

    monkeypatch.setattr(
        "rheojax.gui.services.transform_service.TransformService.apply_transform",
        fake_apply_transform,
    )

    app = AppState()
    app.library.add(_ref("d1", "flow_curve"))
    app.library.store_payload("d1", _RheoData([0.0], [0.0]))
    app.transform.transform_key = "smooth_derivative"
    app.transform.slots = {"input": "d1"}

    ctl, bodies = build_transform_controller(app)
    run_step = bodies[2]
    run_step.run()

    assert app.transform.result is not None
    assert app.transform.result["output"].x[0] == 1.0
    # "smooth_derivative" is same-domain (not spectral/decomposition) -> protocol_type preserved
    assert app.transform.result["protocol_type"] == "flow_curve"
    # Regression: step4_visualize.py's "Input vs output" overlay tab reads
    # payload["input"] -- _make_run_fn resolved the slot data into `data`
    # right before invoking the worker but never attached it to the result
    # dict, so the Input trace was silently never plotted in the real app.
    assert app.transform.result["input"] is not None
    assert app.transform.result["input"].x[0] == 0.0


def test_run_fn_raises_on_worker_failure(qapp, monkeypatch):
    def fake_apply_transform(self, name, data, params):
        raise ValueError("bad params")

    monkeypatch.setattr(
        "rheojax.gui.services.transform_service.TransformService.apply_transform",
        fake_apply_transform,
    )

    app = AppState()
    app.library.add(_ref("d1", "flow_curve"))
    app.library.store_payload("d1", _RheoData([0.0], [0.0]))
    app.transform.transform_key = "smooth_derivative"
    app.transform.slots = {"input": "d1"}

    ctl, bodies = build_transform_controller(app)
    run_step = bodies[2]
    # RunStep.run() only catches NotImplementedError; any other exception the
    # real run_fn raises propagates, and the status label stays untouched
    # ("" -- never reaches the "✓ done" line). This documents that contract.
    with pytest.raises(RuntimeError):
        run_step.run()
