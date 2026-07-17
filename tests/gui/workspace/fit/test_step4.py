import pytest

pytest.importorskip("PySide6")
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step4_nuts import NutsStep


def test_nuts_warmstart_priors_and_skip(qtbot):
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}})
    step = NutsStep(st, sample_fn=lambda *a, **k: {"rhat": 1.0})
    qtbot.addWidget(step)
    pri = step.suggested_priors()  # MAP-centered
    assert set(pri) == {"G0", "eta", "sigma"} and pri["G0"]["type"] == "lognormal"
    step.skip()
    assert st.nuts_result is None and step.is_ready() is True  # skip => ready


def test_nuts_run_without_wired_sample_fn_does_not_crash(qtbot):
    """Clicking Sample before the real sampler is wired must not raise NotImplementedError
    from inside the Qt slot (which can abort the process under PySide6 6.x)."""
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}})
    step = NutsStep(
        st
    )  # no sample_fn injected -> falls back to _default_sample_fn stub
    qtbot.addWidget(step)
    step.run()  # must not raise
    assert st.nuts_result is None
    assert step.is_ready() is False


def test_nuts_sampler_failure_is_reported_without_escaping_qt_slot(qtbot):
    st = FitState(nlsq_result={"params": {"G0": 1000.0}})

    def fail(*args, **kwargs):
        raise RuntimeError("sampler failed")

    step = NutsStep(st, sample_fn=fail)
    qtbot.addWidget(step)
    step.run()
    assert st.nuts_result is None
    assert "sampler failed" in step._result.text()


def test_nuts_run_shows_cancelled_not_failed_on_cancellation(qtbot):
    """A user cancel must render as "Cancelled.", not be swallowed by the
    generic except Exception branch and shown as a NUTS failure."""
    from rheojax.gui.jobs.cancellation import CancellationError

    st = FitState(nlsq_result={"params": {"G0": 1000.0}})

    def cancelled(*args, **kwargs):
        raise CancellationError("Operation cancelled by user")

    step = NutsStep(st, sample_fn=cancelled)
    qtbot.addWidget(step)
    step.run()
    assert step._result.text() == "Cancelled."


def test_reset_skip_clears_stale_skip_decision(qtbot):
    # Regression: skip() sets _skipped=True permanently. If NLSQ is later
    # re-run (invalidating nuts_result via the cascade), is_ready() must not
    # keep reporting "ready" off a stale skip made against an old fit.
    st = FitState(nlsq_result={"params": {"G0": 1000.0, "eta": 50.0}})
    step = NutsStep(st)
    qtbot.addWidget(step)
    step.skip()
    assert step.is_ready() is True

    step.reset_skip()
    assert step.is_ready() is False  # nuts_result still None -> no longer ready
    assert st.nuts_result is None


def test_reset_skip_wired_to_nlsq_edited_in_fit_controller(qtbot):
    # Behavioral check of the cross-widget wiring: NlsqStep's `edited` (fired
    # whenever a new NLSQ result lands) must reset NutsStep's skip flag.
    import rheojax.models  # noqa: F401 - ensure model registry is populated
    from rheojax.gui.foundation.state import AppState
    from rheojax.gui.workspace.fit.fit_controller import build_fit_controller

    app = AppState()
    ctl, bodies = build_fit_controller(app)
    for b in bodies:
        qtbot.addWidget(b)
    nlsq_body, nuts_body = bodies[2], bodies[3]

    nuts_body.skip()
    assert nuts_body.is_ready() is True

    # Fake a completed NLSQ fit and fire the same signal build_fit_controller
    # wires reset_skip to, without needing the real (NotImplementedError) fit_fn.
    app.fit.nlsq_result = {"params": {"G0": 1.0}}
    nlsq_body.edited.emit()

    assert nuts_body.is_ready() is False
