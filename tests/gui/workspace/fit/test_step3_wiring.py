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
                                    dataset_id="", target_accept=0.8, model_config=None):
        calls["sample"] = (model_name, target_accept, model_config)
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
    # Regression: NUTS must forward the user's Step 1 model_config through to
    # run_bayesian_isolated (see _make_sample_fn in fit_controller.py), same
    # as NLSQ's fit_fn does above -- otherwise the model is silently
    # reconstructed with constructor defaults for NUTS only.
    assert calls["sample"][2] == {"n_modes": 2}
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


def test_nlsq_step_loads_parameter_table_and_passes_bounds(monkeypatch):
    from types import SimpleNamespace

    from rheojax.gui.foundation.state import FitState
    from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep

    # Mimics rheojax.core.parameters.Parameter's real shape (`.value`,
    # `.bounds` tuple, `.units`, `.description` -- no `.fixed`, no
    # `.min_bound`/`.max_bound`), which is what ModelRegistry.create(...)
    # .parameters actually returns in production.
    class _FakeInstance:
        parameters = {
            "a": SimpleNamespace(value=1.0, bounds=(0.0, 10.0), units="Pa", description=""),
            "b": SimpleNamespace(value=2.0, bounds=(0.0, 5.0), units="s", description=""),
        }

    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.step3_nlsq.ModelRegistry.create",
        lambda name, **kw: _FakeInstance(),
    )

    captured = {}

    def fake_fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                     multi_start=None):
        captured["initial_params"] = initial_params
        return {"params": {"a": 1.0, "b": 2.0}, "r_squared": 0.5, "success": True}

    st = FitState(model_key="m", model_config={})
    step = NlsqStep(st, fit_fn=fake_fit_fn)
    step.load_parameters_from_model()
    table_params = step.parameter_table().get_parameters()
    assert table_params["a"].value == 1.0
    assert table_params["a"].min_bound == 0.0
    assert table_params["a"].max_bound == 10.0
    assert table_params["b"].fixed is False  # defaults never carry fixed=True

    # Simulate a user checking "fixed" for "b" via the table (the real UI
    # path is the checkbox widget's toggled signal -> _on_fixed_toggled()).
    from dataclasses import replace

    table_params["b"] = replace(table_params["b"], fixed=True)
    step.parameter_table().set_parameters(table_params)

    step.run()
    ip = captured["initial_params"]
    assert ip["a"] == {"value": 1.0, "bounds": (0.0, 10.0), "fixed": False}
    assert ip["b"] == {"value": 2.0, "bounds": (0.0, 5.0), "fixed": True}


def test_multistart_forwards_config_to_fit_fn():
    from rheojax.gui.foundation.state import FitState
    from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep

    captured = {}

    def fake_fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                     multi_start=None):
        captured["multi_start"] = multi_start
        return {"params": {"a": 1.0}, "r_squared": 0.9, "success": True}

    st = FitState(model_key="m", model_config={})
    step = NlsqStep(st, fit_fn=fake_fit_fn)
    step.set_multistart(True, 5)
    step.run()
    assert captured["multi_start"] == {"enabled": True, "count": 5}


def test_multistart_restart_loop_keeps_best_and_preserves_fixed_params(monkeypatch):
    # Exercises _make_fit_fn's real restart loop (not just NlsqStep forwarding
    # the multi_start dict). Restart 2 (the 3rd call) returns the best
    # r_squared, so the winning result must come from that call. The "b"
    # parameter is fixed=True; its "value" must be identical across all 3
    # calls to run_fit_isolated -- if the jitter overwrote it (Finding 1),
    # calls 2 and 3 would see a perturbed "value" for "b".
    calls = []

    def fake_run_fit_isolated(model_name, x_data, y_data, test_mode, initial_params,
                               options, progress_queue, cancel_event, y2_data=None,
                               metadata=None, dataset_id="", model_config=None):
        calls.append(initial_params)
        r_squared = [0.5, 0.7, 0.95][len(calls) - 1]
        return {"params": {"a": 1.0, "b": 2.0}, "r_squared": r_squared, "success": True}

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
    initial_params = {
        "a": {"value": 1.0, "bounds": (0.0, 10.0), "fixed": False},
        "b": {"value": 2.0, "bounds": (0.0, 5.0), "fixed": True},
    }
    result = fit_fn(
        "power_law", {}, "d1", {"x": 0, "y": 1},
        initial_params=initial_params,
        multi_start={"enabled": True, "count": 3},
    )

    assert len(calls) == 3
    assert result["r_squared"] == 0.95

    # Fixed param "b" must be unchanged across every restart.
    for call_params in calls:
        assert call_params["b"]["value"] == 2.0
        assert call_params["b"]["fixed"] is True

    # Non-fixed param "a" is allowed to differ across restarts (jittered).
    a_values = [call_params["a"]["value"] for call_params in calls]
    assert a_values[0] == 1.0  # first run uses caller's value unperturbed
    assert any(v != 1.0 for v in a_values[1:])  # restarts jitter it


def test_multistart_disabled_by_default():
    from rheojax.gui.foundation.state import FitState
    from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep

    captured = {}

    def fake_fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                     multi_start=None):
        captured["multi_start"] = multi_start
        return {"params": {"a": 1.0}, "r_squared": 0.9, "success": True}

    st = FitState(model_key="m", model_config={})
    step = NlsqStep(st, fit_fn=fake_fit_fn)
    step.run()
    assert captured["multi_start"] == {"enabled": False, "count": 8}
