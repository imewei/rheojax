import types

import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState, ParameterState
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep


def test_nlsq_runs_and_stores_result(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["args"] = (model_key, model_config, data_ref)
        return {
            "params": {"G0": 1000.0},
            "r_squared": 0.99,
            "reduced_chi_squared": 0.8,
            "uncertainties": [10.0],
        }

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    assert step.is_ready() is False
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["r_squared"] == 0.99
    assert calls["args"][0] == "maxwell"
    assert step.is_ready() is True


def test_nlsq_reads_params_from_non_dict_result(qtbot):
    """Non-dict fit_fn results (e.g. FitResult) must be read via `.params`, not `.parameters`."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    fake_result = types.SimpleNamespace(
        params={"G0": 1000.0}, r_squared=0.9, success=True
    )

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        return fake_result

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["params"] == {"G0": 1000.0}
    assert st.nlsq_result["r_squared"] == 0.9


def test_nlsq_handles_none_r_squared(qtbot):
    """A present-but-None r_squared must not crash the result label formatting."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        return {"params": {"G0": 1000.0}, "r_squared": None}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()
    assert st.nlsq_result["r_squared"] is None


def test_nlsq_run_without_wired_fit_fn_does_not_crash(qtbot):
    """Clicking Run before the real solver is wired must not raise NotImplementedError
    from inside the Qt slot (which can abort the process under PySide6 6.x)."""
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    step = NlsqStep(st)  # no fit_fn injected -> falls back to _default_fit_fn stub
    qtbot.addWidget(step)
    step.run()  # must not raise
    assert st.nlsq_result is None
    assert step.is_ready() is False


def test_nlsq_solver_failure_is_reported_without_escaping_qt_slot(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    step = NlsqStep(st, fit_fn=fail)
    qtbot.addWidget(step)
    step.run()
    assert st.nlsq_result is None
    assert "optimizer failed" in step._result.text()


def test_refresh_display_shows_chi2_mpe_and_fit_time_when_present(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    st.nlsq_result = {
        "params": {"G0": 1000.0},
        "r_squared": 0.95,
        "chi_squared": 12.3,
        "mpe": 4.5,
        "fit_time": 1.23,
    }
    step = NlsqStep(st)
    qtbot.addWidget(step)

    step.refresh_display()

    text = step._result.text()
    assert "R²=0.950" in text
    assert "chi²=12.3" in text
    assert "MPE=4.50%" in text
    assert "time=1.23s" in text


def test_refresh_display_shows_parameter_uncertainties_from_pcov(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    st.nlsq_result = {
        "params": {"G0": 1000.0, "eta": 50.0},
        "r_squared": 0.95,
        "pcov": [[4.0, 0.0], [0.0, 9.0]],
    }
    step = NlsqStep(st)
    qtbot.addWidget(step)

    step.refresh_display()

    text = step._result.text()
    assert "G0=±2" in text
    assert "eta=±3" in text


def test_refresh_display_omits_optional_fields_when_absent(qtbot):
    # Regression: a result dict without chi_squared/mpe/fit_time/pcov (e.g.
    # a bare test fake, as used throughout this test file) must not crash
    # and must not show placeholder/garbage text for the missing fields.
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    st.nlsq_result = {"params": {"G0": 1000.0}, "r_squared": 0.9}
    step = NlsqStep(st)
    qtbot.addWidget(step)

    step.refresh_display()

    assert step._result.text() == "R²=0.900"


def test_options_button_opens_dialog_and_stores_result(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rheojax.gui.dialogs.fitting_options import FittingOptionsDialog

    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    step = NlsqStep(st)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        FittingOptionsDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    monkeypatch.setattr(
        FittingOptionsDialog,
        "get_options",
        lambda self: {"algorithm": "L-BFGS-B", "ftol": 1e-10},
    )

    step._options_btn.click()

    assert step.fit_options() == {"algorithm": "L-BFGS-B", "ftol": 1e-10}


def test_options_dialog_cancelled_keeps_previous_options(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rheojax.gui.dialogs.fitting_options import FittingOptionsDialog

    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    step = NlsqStep(st)
    qtbot.addWidget(step)

    monkeypatch.setattr(
        FittingOptionsDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )

    step._fit_options = {"ftol": 1e-9}
    step._options_btn.click()

    assert step.fit_options() == {"ftol": 1e-9}


def test_run_forwards_fit_options_to_fit_fn(qtbot):
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["options"] = kwargs.get("options")
        return {"params": {"G0": 1000.0}, "r_squared": 0.9}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    step._fit_options = {"ftol": 1e-10}

    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()

    assert calls["options"] == {"ftol": 1e-10}


def test_run_strips_dialog_multistart_keys_from_options(qtbot):
    """FittingOptionsDialog can set "multistart"/"num_starts" in _fit_options,
    which ModelService.fit() would translate into its own backend-level
    multi-start -- nesting with this step's own _ms_enabled/_ms_count outer
    restart loop. run() must strip those two keys (and only those two) from
    the dict passed as options=, without mutating self._fit_options itself.
    """
    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["options"] = kwargs.get("options")
        return {"params": {"G0": 1000.0}, "r_squared": 0.9}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    step._fit_options = {"multistart": True, "num_starts": 10, "ftol": 1e-10}

    with qtbot.waitSignal(step.finished, timeout=2000):
        step.run()

    assert calls["options"] == {"ftol": 1e-10}
    # self._fit_options (the dialog's stored state) must be untouched.
    assert step._fit_options == {
        "multistart": True,
        "num_starts": 10,
        "ftol": 1e-10,
    }


def _make_param_state(value=1.0, min_bound=0.0, max_bound=10.0):
    return ParameterState(
        name="G0",
        value=value,
        min_bound=min_bound,
        max_bound=max_bound,
        fixed=False,
        unit="Pa",
        description="",
    )


def test_run_button_blocked_by_invalid_row_shows_warning(qtbot, monkeypatch):
    """PR #104 gate: clicking the real Run NLSQ button (not calling run()
    directly) with an invalid parameter row must warn the user and must NOT
    invoke the underlying fit_fn."""
    from rheojax.gui.compat import QMessageBox

    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {"invoked": False}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["invoked"] = True
        return {"params": {"G0": 1000.0}, "r_squared": 0.9}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    step._table.set_parameters({"G0": _make_param_state()})
    # Set the value cell out of its [min_bound, max_bound] range --
    # ParameterTable._on_item_changed only reverts genuinely non-numeric
    # text; an out-of-range *numeric* value is left in place (red/tooltip
    # only), so has_invalid_rows() must catch it and the click handler must
    # refuse to launch.
    step._table.item(0, 1).setText("999")
    assert step._table.has_invalid_rows() is True

    warned = {}

    def fake_warning(*args, **kwargs):
        warned["called"] = True
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    step._run_btn.click()

    assert warned.get("called") is True
    assert calls["invoked"] is False
    assert st.nlsq_result is None


def test_run_button_proceeds_when_all_rows_valid(qtbot, monkeypatch):
    """Positive-path guard: an all-valid parameter table must NOT be blocked
    by the has_invalid_rows() gate -- clicking Run NLSQ must still launch the
    fit and must not pop the warning dialog."""
    from rheojax.gui.compat import QMessageBox

    st = FitState(
        protocol="oscillation", model_key="maxwell", data_ref="d", column_map={"x": 0}
    )
    calls = {"invoked": False}

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        calls["invoked"] = True
        return {"params": {"G0": 1000.0}, "r_squared": 0.9}

    step = NlsqStep(st, fit_fn=fake_fit)
    qtbot.addWidget(step)
    step._table.set_parameters({"G0": _make_param_state()})
    assert step._table.has_invalid_rows() is False

    warned = {"called": False}

    def fake_warning(*args, **kwargs):
        warned["called"] = True
        return QMessageBox.StandardButton.Ok

    monkeypatch.setattr(QMessageBox, "warning", fake_warning)

    with qtbot.waitSignal(step.finished, timeout=2000):
        step._run_btn.click()

    assert warned["called"] is False
    assert calls["invoked"] is True
    assert st.nlsq_result["r_squared"] == 0.9
