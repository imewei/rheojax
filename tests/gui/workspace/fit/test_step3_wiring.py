from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit.fit_controller import _make_fit_fn, build_fit_controller


class _RheoData:
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)


def test_build_fit_controller_injects_real_fit_and_sample_fn(monkeypatch, qtbot):
    calls = {}

    def fake_run_fit_isolated(model_name, x_data, y_data, test_mode, initial_params,
                               options, progress_queue, cancel_event, y2_data=None,
                               metadata=None, dataset_id="", model_config=None):
        calls["fit"] = (model_name, model_config)
        return {"params": {"a": 1.0}, "r_squared": 0.9, "success": True}

    def fake_run_bayesian_isolated(model_name, x_data, y_data, test_mode, num_warmup,
                                    num_samples, num_chains, warm_start, priors, seed,
                                    progress_queue, cancel_event, y2_data=None,
                                    metadata=None, fitted_model_state=None,
                                    dataset_id="", target_accept=0.8):
        calls["sample"] = (model_name, target_accept)
        return {"posterior_samples": {"a": [1.0, 1.1]}, "r_hat": {"a": 1.0}}

    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.fit_controller.run_fit_isolated",
        fake_run_fit_isolated,
    )
    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.fit_controller.run_bayesian_isolated",
        fake_run_bayesian_isolated,
    )

    app = AppState()
    app.library.add(
        DatasetRef(id="d1", name="d1", protocol_type="flow_curve", origin="imported",
                   units={}, row_count=2, hash="h", provenance={}, lineage=[])
    )
    app.library.store_payload("d1", _RheoData([1.0, 2.0], [1.0, 2.0]))
    app.fit.model_key = "power_law"
    app.fit.model_config = {"n_modes": 2}
    app.fit.data_ref = "d1"
    app.fit.column_map = {"x": 0, "y": 1}

    ctl, bodies = build_fit_controller(app)
    nlsq_step = bodies[2]
    nlsq_step.run()
    assert calls["fit"] == ("power_law", {"n_modes": 2})
    assert app.fit.nlsq_result["r_squared"] == 0.9

    nuts_step = bodies[3]
    nuts_step._target.setValue(0.85)
    nuts_step.run()
    assert calls["sample"][1] == 0.85
    assert app.fit.nuts_result["r_hat"] == {"a": 1.0}


def test_make_fit_fn_normalizes_parameters_key_to_params(monkeypatch):
    # subprocess_fit.run_fit_isolated's real return dict uses key "parameters"
    # (see ModelService.fit()'s FitResult), but every downstream consumer of
    # FitState.nlsq_result (NutsStep.suggested_priors/run, step3_nlsq.py's own
    # FitResult-normalization branch, and the whole test suite) is written
    # against a "params" key. Without normalizing here, real NLSQ runs would
    # silently warm-start NUTS from an empty dict.
    def fake_run_fit_isolated(*args, **kwargs):
        return {"parameters": {"a": 1.0}, "r_squared": 0.9, "success": True}

    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.fit_controller.run_fit_isolated",
        fake_run_fit_isolated,
    )

    lib = DatasetLibrary()
    lib.add(
        DatasetRef(id="d1", name="d1", protocol_type="flow_curve", origin="imported",
                   units={}, row_count=2, hash="h", provenance={}, lineage=[])
    )
    lib.store_payload("d1", _RheoData([1.0, 2.0], [1.0, 2.0]))

    fit_fn = _make_fit_fn(lib)
    result = fit_fn("power_law", {}, "d1", {"x": 0, "y": 1})
    assert result["params"] == {"a": 1.0}
    assert result["parameters"] == {"a": 1.0}  # original key preserved
