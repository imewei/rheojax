from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa: F401
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.fit import fit_controller
from rheojax.gui.workspace.fit.fit_controller import build_fit_controller


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _fake_nlsq_result(*args, **kwargs):
    # Realistically-shaped run_fit_isolated() output: "parameters" (not
    # "params"), "x_fit"/"y_fit" (not "x"/"y") -- matching the real function's
    # actual return shape (see subprocess_fit.py), deliberately WITHOUT the
    # "x"/"y"/"params" keys downstream consumers expect. y_fit is evaluated
    # at the same points as the input x (20 points), matching real usage.
    y_fit = list(np.linspace(0.25, 4.9, 20))
    return {
        "parameters": {"eta": 1.5, "G": 2.5},
        "r_squared": 0.9,
        "x_fit": list(np.linspace(0.1, 10.0, 20)),
        "y_fit": y_fit,
    }


def _fake_nuts_result(*args, **kwargs):
    # Realistically-shaped run_bayesian_isolated() output.
    return {
        "posterior_samples": {"eta": [1.4, 1.5, 1.6], "G": [2.4, 2.5, 2.6]},
        "sample_stats": {"energy": [1.0, 1.1, 0.9], "diverging": [False, False, False]},
        "r_hat": {"eta": 1.0, "G": 1.0},
        "ess": {"eta": 500, "G": 500},
        "bfmi": 0.5,
        "num_chains": 1,
    }


def test_full_fit_workflow_seeds_tables_and_exports(monkeypatch, qapp, tmp_path):
    # End-to-end regression: every task's unit tests hand-built the state
    # each step expected, so two Critical gaps went undetected in the
    # assembled app -- ParameterTable/PriorsEditor were never seeded from
    # real upstream output (Finding 1), and nlsq_result never carried "x"/"y"
    # (Finding 2, silently dropping fitted_curve.csv / overlay / residuals).
    # This test drives the real bodies produced by build_fit_controller
    # end-to-end and proves both fixes hold together.
    monkeypatch.setattr(fit_controller, "run_fit_isolated", _fake_nlsq_result)
    monkeypatch.setattr(fit_controller, "run_bayesian_isolated", _fake_nuts_result)

    library = DatasetLibrary()
    library.add(
        DatasetRef(
            id="ds1",
            name="ds1",
            protocol_type="oscillation",
            origin="imported",
            units={"x": "rad/s"},
            row_count=20,
            hash="h",
            provenance={},
            lineage=[],
        )
    )
    x = np.linspace(0.1, 10.0, 20)
    y = np.linspace(0.2, 5.0, 20)
    library.store_payload("ds1", _RheoData(x, y))
    app_state = AppState(library=library)

    ctl, bodies = build_fit_controller(app_state)

    # 1. Protocol/model selection -- fires the new protocol_body.edited ->
    #    nlsq_body.load_parameters_from_model() wiring (Finding 1, part 1).
    bodies[0].set_protocol("oscillation")
    bodies[0].set_model("maxwell")
    assert bodies[2].parameter_table().get_parameters() != {}

    # 2. Select the dataset through the real DataStep combo.
    bodies[1].select_dataset("ds1")
    assert app_state.fit.data_ref == "ds1"

    # 3. Run the real NlsqStep with the real fit_fn injected by
    #    build_fit_controller (backed by the monkeypatched run_fit_isolated).
    bodies[2].run()
    nlsq_result = app_state.fit.nlsq_result
    assert nlsq_result["params"] == {"eta": 1.5, "G": 2.5}  # parameters->params alias
    assert list(nlsq_result["x"]) == list(x)  # Finding 2: x attached
    assert list(nlsq_result["y"]) == list(y)  # Finding 2: y attached

    # 4. PriorsEditor must already be seeded -- fired by the new
    #    nlsq_body.finished -> nuts_body.load_suggested_priors() wiring
    #    (Finding 1, part 2), triggered as a side effect of run() above.
    assert bodies[3].priors_editor().get_all_priors() != {}

    # 5. Run the real NutsStep with the real sample_fn (backed by the
    #    monkeypatched run_bayesian_isolated).
    bodies[3].run()
    assert app_state.fit.nuts_result["posterior_samples"]

    # 6. Export: the CSV write was silently skipped before Finding 2's fix
    #    because nlsq_result["x"] was always None.
    written = bodies[5].export_bundle(tmp_path)
    assert written["fitted_curve"].exists()
