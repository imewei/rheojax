"""Regression tests for the NLSQ/NUTS Cancel button and status-bar progress
wiring added to close a PR #100 review gap: neither had any coverage, and
both have real "silently does nothing" / "stuck forever" failure modes.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.state import ActiveJobsState, AppState
from rheojax.gui.workspace.fit.fit_controller import _make_fit_fn, _make_sample_fn
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep
from rheojax.gui.workspace.fit.step4_nuts import NutsStep


class _FakeWorker:
    def __init__(self):
        self.cancel_called = False

    def cancel(self):
        self.cancel_called = True


class _FakeStatusBar:
    def __init__(self):
        self.progress_calls: list[tuple[int, int, str]] = []
        self.hide_calls = 0

    def show_progress(self, value, maximum, text=""):
        self.progress_calls.append((value, maximum, text))

    def hide_progress(self):
        self.hide_calls += 1


def _run_qthreadpool_tasks(qtbot):
    from PySide6.QtCore import QThreadPool

    QThreadPool.globalInstance().waitForDone(2000)
    qtbot.wait(50)


def test_nlsq_cancel_button_hidden_without_active_jobs(qtbot):
    step = NlsqStep(state=AppState().fit, active_jobs=None)
    qtbot.addWidget(step)
    step._current_job_id = "d1:nlsq"  # simulate a run() in flight
    step._cancel_btn.setVisible(step._active_jobs is not None)
    assert step._cancel_btn.isVisible() is False


def test_nlsq_cancel_noop_when_no_active_jobs(qtbot):
    step = NlsqStep(state=AppState().fit, active_jobs=None)
    qtbot.addWidget(step)
    step._current_job_id = "d1:nlsq"
    step._on_cancel_clicked()  # must not raise


def test_nlsq_cancel_noop_when_job_not_registered(qtbot):
    active_jobs = ActiveJobsState()
    step = NlsqStep(state=AppState().fit, active_jobs=active_jobs)
    qtbot.addWidget(step)
    step._current_job_id = "d1:nlsq"  # no matching entry in active_jobs.by_id
    step._on_cancel_clicked()  # must not raise, must not crash on missing job


def test_nlsq_cancel_dispatches_to_registered_worker(qtbot):
    active_jobs = ActiveJobsState()
    worker = _FakeWorker()
    active_jobs.by_id["d1:nlsq"] = {"status": "running", "worker": worker}
    step = NlsqStep(state=AppState().fit, active_jobs=active_jobs)
    qtbot.addWidget(step)
    step._current_job_id = "d1:nlsq"

    step._on_cancel_clicked()
    _run_qthreadpool_tasks(qtbot)

    assert worker.cancel_called is True


def test_nuts_cancel_dispatches_to_registered_worker(qtbot):
    active_jobs = ActiveJobsState()
    worker = _FakeWorker()
    active_jobs.by_id["d1:nuts"] = {"status": "running", "worker": worker}
    step = NutsStep(state=AppState().fit, active_jobs=active_jobs)
    qtbot.addWidget(step)
    step._current_job_id = "d1:nuts"

    step._on_cancel_clicked()
    _run_qthreadpool_tasks(qtbot)

    assert worker.cancel_called is True


def test_nlsq_run_shows_and_hides_progress_via_status_bar(monkeypatch, qtbot):
    def fake_run_fit_isolated(
        model_name,
        x_data,
        y_data,
        test_mode,
        initial_params,
        options,
        progress_queue,
        cancel_event,
        y2_data=None,
        metadata=None,
        dataset_id="",
        model_config=None,
    ):
        progress_queue.put(
            {"type": "progress", "percent": 50, "total": 100, "message": "halfway"}
        )
        return {"params": {"a": 1.0}, "r_squared": 0.9, "success": True}

    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.fit_controller.run_fit_isolated",
        fake_run_fit_isolated,
    )

    app = AppState()
    app.library.add(
        DatasetRef(
            id="d1",
            name="d1",
            protocol_type="flow_curve",
            origin="imported",
            units={},
            row_count=2,
            hash="h",
            provenance={},
            lineage=[],
        )
    )

    class _RheoData:
        x = [1.0, 2.0]
        y = [1.0, 2.0]

    app.library.store_payload("d1", _RheoData())
    app.fit.protocol = "flow_curve"
    app.fit.model_key = "power_law"
    app.fit.model_config = {}
    app.fit.data_ref = "d1"

    status_bar = _FakeStatusBar()
    fit_fn = _make_fit_fn(app.library, app.fit, app.active_jobs, status_bar)
    step = NlsqStep(state=app.fit, fit_fn=fit_fn, active_jobs=app.active_jobs)
    qtbot.addWidget(step)

    step.run()

    # Initial "Fitting..." message, then the forwarded 50% progress message --
    # both must have reached the status bar, not just been computed and
    # silently discarded (the original bug this wiring fixes).
    assert status_bar.progress_calls[0] == (0, 100, "Fitting...")
    assert (50, 100, "halfway") in status_bar.progress_calls
    # hide_progress must fire exactly once when the run completes, or the
    # status bar is left showing a stale progress bar forever.
    assert status_bar.hide_calls == 1
    # Cancel button must not be left visible after a completed run.
    assert step._cancel_btn.isVisible() is False
    assert step._current_job_id is None


def test_nuts_run_shows_and_hides_progress_via_status_bar(monkeypatch, qtbot):
    def fake_run_bayesian_isolated(
        model_name,
        x_data,
        y_data,
        test_mode,
        num_warmup,
        num_samples,
        num_chains,
        warm_start,
        priors,
        seed,
        progress_queue,
        cancel_event,
        y2_data=None,
        metadata=None,
        fitted_model_state=None,
        dataset_id="",
        target_accept=0.8,
        model_config=None,
        max_tree_depth=None,
    ):
        return {"posterior_samples": {"a": [1.0, 1.1]}, "r_hat": {"a": 1.0}}

    monkeypatch.setattr(
        "rheojax.gui.workspace.fit.fit_controller.run_bayesian_isolated",
        fake_run_bayesian_isolated,
    )

    app = AppState()
    app.library.add(
        DatasetRef(
            id="d1",
            name="d1",
            protocol_type="oscillation",
            origin="imported",
            units={},
            row_count=2,
            hash="h",
            provenance={},
            lineage=[],
        )
    )

    class _RheoData:
        x = [1.0, 2.0]
        y = [1.0, 2.0]

    app.library.store_payload("d1", _RheoData())
    app.fit.data_ref = "d1"
    app.fit.model_key = "power_law"

    status_bar = _FakeStatusBar()
    sample_fn = _make_sample_fn(app.library, app.fit, app.active_jobs, status_bar)
    step = NutsStep(state=app.fit, sample_fn=sample_fn, active_jobs=app.active_jobs)
    qtbot.addWidget(step)

    step.run()

    assert status_bar.progress_calls[0] == (0, 100, "Sampling...")
    assert status_bar.hide_calls == 1
    assert step._cancel_btn.isVisible() is False
    assert step._current_job_id is None


def test_nlsq_run_shows_in_body_progress_bar_during_run_and_hides_after(qtbot):
    """The status-bar sliver above is easy to miss on a long run; the step
    also gets its own in-body indicator next to Cancel. Checked from inside
    fit_fn (not after run() returns) since the fake fit is synchronous --
    by the time run() returns, the finally: block has already hidden it."""
    step = None

    def fake_fit(model_key, model_config, data_ref, column_map, **kwargs):
        assert step._progress.isVisible() is True
        return {"params": {"a": 1.0}, "r_squared": 0.9}

    step = NlsqStep(state=AppState().fit, fit_fn=fake_fit)
    qtbot.addWidget(step)
    assert step._progress.isVisible() is False

    step.run()

    assert step._progress.isVisible() is False


def test_nlsq_run_hides_progress_bar_even_when_fit_raises(qtbot):
    """The finally: block's job -- prevents the bar from getting stuck
    visible forever on the error path."""

    def fail(*args, **kwargs):
        raise RuntimeError("optimizer failed")

    step = NlsqStep(state=AppState().fit, fit_fn=fail)
    qtbot.addWidget(step)

    step.run()

    assert step._progress.isVisible() is False


def test_nuts_run_shows_in_body_progress_bar_during_run_and_hides_after(qtbot):
    step = None

    def fake_sample(priors, warm_start, cfg):
        assert step._progress.isVisible() is True
        return {"posterior_samples": {"a": [1.0, 1.1]}, "r_hat": {"a": 1.0}}

    step = NutsStep(state=AppState().fit, sample_fn=fake_sample)
    qtbot.addWidget(step)
    assert step._progress.isVisible() is False

    step.run()

    assert step._progress.isVisible() is False


def test_nuts_run_hides_progress_bar_even_when_sample_raises(qtbot):
    def fail(*args, **kwargs):
        raise RuntimeError("sampler failed")

    step = NutsStep(state=AppState().fit, sample_fn=fail)
    qtbot.addWidget(step)

    step.run()

    assert step._progress.isVisible() is False
