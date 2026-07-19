from __future__ import annotations

import math
import multiprocessing
import queue

import numpy as np
from PySide6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, QTimer, Signal

from rheojax.gui.foundation.invalidation import _CLEAR, _FIT_CASCADE, register_step
from rheojax.gui.foundation.state import AppState
from rheojax.gui.jobs.cancellation import ProcessCancellationToken
from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated
from rheojax.gui.jobs.subprocess_fit import run_fit_isolated
from rheojax.gui.workspace.controller import FitController, Step
from rheojax.gui.workspace.fit.step1_protocol_model import ProtocolModelStep
from rheojax.gui.workspace.fit.step2_data import DataStep
from rheojax.gui.workspace.fit.step3_nlsq import NlsqStep
from rheojax.gui.workspace.fit.step4_nuts import NutsStep
from rheojax.gui.workspace.fit.step5_visualize import VisualizeStep
from rheojax.gui.workspace.fit.step6_export import ExportStep

# Which FitState field each step edits (for invalidate_downstream). None means
# that step's edits don't cascade (visualize is read-only; export just saves).
_CHANGED = ["model_key", "column_map", "nlsq_result", "nuts_result", None, None]


def _r2_sort_key(result: dict | None) -> float:
    """Multi-start restart-selection key: treat missing/NaN r_squared as the
    worst possible score (-1.0), never as itself.

    subprocess_fit.run_fit_isolated's r_squared can legitimately come back
    NaN (a degenerate fit). `nan > x` and `x > nan` are both always False in
    Python, so once a NaN became `best_r2` no later, better restart could
    ever replace it -- defeating multi-start under exactly the failure
    conditions it exists to escape.
    """
    if not result:
        return -1.0
    v = result.get("r_squared")
    if v is None:
        return -1.0
    try:
        v = float(v)
    except (TypeError, ValueError):
        return -1.0
    return -1.0 if math.isnan(v) else v


class _CallableWorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(object)


class _CallableWorker(QRunnable):
    """Runs an arbitrary no-arg callable on a QThreadPool thread."""

    def __init__(self, fn) -> None:
        super().__init__()
        self._fn = fn
        self.signals = _CallableWorkerSignals()

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # noqa: BLE001 - re-raised on the GUI thread by _run_on_thread
            self.signals.failed.emit(exc)
        else:
            self.signals.completed.emit(result)


def _run_on_thread(fn, *, progress_queue=None, on_progress=None):
    """Run *fn* on a QThreadPool thread while pumping a local QEventLoop, so
    the GUI stays responsive -- mirrors transform_controller.py's
    _make_run_fn. The calling function still returns synchronously (or
    re-raises *fn*'s exception here, on the caller's thread), but the actual
    NLSQ/NUTS computation no longer blocks the GUI thread directly, which
    used to freeze the entire app (repaints, Cancel, everything) for the
    full duration of every real fit/sample run.

    If *progress_queue* is given, it is drained on a QTimer (on the GUI
    thread, alongside the nested QEventLoop above) and each message is
    forwarded to *on_progress*. Without this, messages that
    subprocess_fit.py/subprocess_bayesian.py put on the queue during the
    run just accumulate unread until the caller closes the queue after
    the run finishes -- intermediate fit/sample progress never reaches
    the UI.
    """
    worker = _CallableWorker(fn)
    loop = QEventLoop()
    outcome: dict = {}

    def _on_completed(result):
        outcome["result"] = result
        loop.quit()

    def _on_failed(exc):
        outcome["error"] = exc
        loop.quit()

    worker.signals.completed.connect(_on_completed)
    worker.signals.failed.connect(_on_failed)

    drain_timer = None
    if progress_queue is not None:

        def _drain() -> None:
            while True:
                try:
                    msg = progress_queue.get_nowait()
                except queue.Empty:
                    return
                if on_progress is not None:
                    on_progress(msg)

        drain_timer = QTimer()
        drain_timer.timeout.connect(_drain)
        drain_timer.start(100)

    QThreadPool.globalInstance().start(worker)
    loop.exec()

    if drain_timer is not None:
        drain_timer.stop()
        _drain()  # final drain: catch messages put right before completion

    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _make_progress_reporter(active_jobs, job_id, status_bar=None):
    """Build an on_progress callback that stashes the latest progress
    message on the job's active_jobs.by_id entry, and -- if *status_bar* is
    given -- forwards it to StatusBar.show_progress() so a running NLSQ/NUTS
    job is actually visible (previously this only wrote to active_jobs,
    which nothing read: the run showed no feedback beyond a disabled
    button for its whole duration).
    GUI-thread only -- no lock needed, see ActiveJobsState.lock's docstring.
    """

    def _on_progress(msg) -> None:
        if active_jobs is not None:
            job = active_jobs.by_id.get(job_id)
            if job is not None:
                job["progress"] = msg
        if status_bar is not None and isinstance(msg, dict):
            status_bar.show_progress(
                msg.get("percent", 0), msg.get("total", 100), msg.get("message", "")
            )

    return _on_progress


def _make_fit_fn(library, fit_state, active_jobs=None, status_bar=None):
    """Build the real fit_fn NlsqStep.run() calls."""

    def _fit_fn(
        model_key,
        model_config,
        data_ref,
        column_map,  # ponytail: accepted but not read HERE -- unlike
        # FitState.control_vars (state.py), which no GUI step writes and no
        # fit call reads at all, column_map IS actively populated by
        # DataStep and gates its is_ready()/validation; it just never
        # reaches this fixed-convention fit call. _fit_fn_body below loads
        # the payload and always uses RheoData.x/.y directly regardless of
        # column_map's contents. Not currently exploitable; wire it in if/
        # when a protocol needs per-column selection instead of the fixed
        # convention.
        initial_params=None,
        multi_start=None,
        options=None,
    ):
        # A real, cancellable token (not a throwaway per-restart mp.Event) so
        # window.py's "Cancel them and continue?" dialog (_maybe_confirm_active_jobs)
        # can actually stop this run via CancelWorkerRunnable(worker).run() ->
        # worker.cancel() -> the same cancel_event run_fit_isolated already
        # polls between restarts. Previously no "worker" key was registered at
        # all, so Close/New/Open could only wait out the timeout, never cancel.
        # job_id carries a ":nlsq" suffix (not the bare data_ref) so a
        # concurrent NUTS run on the same dataset (_make_sample_fn below)
        # can't clobber this entry in active_jobs.by_id -- window.py's
        # dataset-delete guard does a prefix match on data_ref, see its
        # own NOTE at window.py's _on_delete_dataset_clicked.
        job_id = f"{data_ref}:nlsq"
        token = ProcessCancellationToken(job_id=job_id)
        # Register for the whole call (all multi-start restarts), not per
        # _run_on_thread call -- _run_on_thread pumps a nested QEventLoop, so
        # New/Open/Close (which only check active_jobs.by_id) would otherwise
        # see it empty and rebuild the workspace out from under this step
        # while the loop is still suspended waiting on the worker thread.
        if active_jobs is not None:
            active_jobs.by_id[job_id] = {"status": "running", "worker": token}
        if status_bar is not None:
            status_bar.show_progress(0, 100, "Fitting...")
        try:
            return _fit_fn_body(
                library,
                fit_state,
                model_key,
                model_config,
                data_ref,
                initial_params,
                multi_start,
                options,
                token.event,
                active_jobs,
                job_id,
                status_bar,
            )
        finally:
            if active_jobs is not None:
                active_jobs.by_id.pop(job_id, None)
            if status_bar is not None:
                status_bar.hide_progress()

    return _fit_fn


def _fit_fn_body(
    library,
    fit_state,
    model_key,
    model_config,
    data_ref,
    initial_params,
    multi_start,
    options,
    cancel_event,
    active_jobs=None,
    job_id=None,
    status_bar=None,
):
    rheo_data = library.load_payload(data_ref)
    # ponytail: Step 1's protocol combo uses the same vocabulary as
    # Protocol.value ("flow_curve"/"creep"/"relaxation"/"startup"/
    # "oscillation"/"laos"), which is exactly what ModelService.fit()'s
    # test_mode expects. Previously this was hardcoded to None, which
    # fell through to data.metadata.get("test_mode", "oscillation") on
    # an empty metadata dict (metadata was never forwarded either) --
    # every real fit silently ran as "oscillation" no matter what
    # protocol the user picked.
    test_mode = fit_state.protocol
    metadata = getattr(rheo_data, "metadata", None)
    ms = multi_start or {}
    count = ms.get("count", 1) if ms.get("enabled") else 1
    rng = np.random.default_rng(0)
    best = None
    for i in range(max(count, 1)):
        # Multi-start's only other cancellation check is inside
        # run_fit_isolated's progress_callback, which doesn't fire until a
        # restart's fit is already underway -- Cancel clicked in the gap
        # between one restart finishing and the next starting would
        # otherwise go unnoticed until that next restart's first iteration.
        if cancel_event is not None and cancel_event.is_set():
            from rheojax.gui.jobs.cancellation import CancellationError

            raise CancellationError("Operation cancelled by user")
        # First run uses the caller's initial_params as-is; restarts 2..N
        # jitter each value by +/-20% (seeded, so runs are reproducible).
        start_params = initial_params
        if i > 0 and initial_params:
            start_params = {
                name: (
                    cfg
                    if cfg.get("fixed") is True
                    else {
                        **cfg,
                        "value": cfg["value"] * (1 + rng.uniform(-0.2, 0.2)),
                    }
                )
                for name, cfg in initial_params.items()
            }
        progress_queue = multiprocessing.Queue()
        try:
            result = _run_on_thread(
                lambda start_params=start_params, progress_queue=progress_queue: run_fit_isolated(
                    model_key,
                    rheo_data.x,
                    rheo_data.y,
                    test_mode=test_mode,
                    initial_params=start_params or {},
                    options=options or {},
                    progress_queue=progress_queue,
                    cancel_event=cancel_event,
                    dataset_id=data_ref,
                    model_config=model_config,
                    metadata=metadata,
                ),
                progress_queue=progress_queue,
                on_progress=_make_progress_reporter(
                    active_jobs, job_id or data_ref, status_bar
                ),
            )
        finally:
            # Release the queue's feeder thread/fds/semaphore -- mirrors
            # ProcessWorkerAdapter._ensure_process_dead()'s queue cleanup.
            try:
                progress_queue.close()
                progress_queue.join_thread()
            except (OSError, ValueError, BrokenPipeError):
                pass
        r2 = _r2_sort_key(result)
        best_r2 = _r2_sort_key(best) if best else -1.0
        if best is None or r2 > best_r2:
            best = result
    # ponytail: run_fit_isolated's real result dict key is "parameters"
    # (see ModelService.fit()'s FitResult), but NlsqStep/NutsStep/tests
    # are all written against a "params" key (step3_nlsq.py's own
    # FitResult-normalization branch uses `res.params`). Alias it here
    # so NUTS warm-start/priors don't silently see an empty dict. Only
    # the winning restart needs this — comparisons above use r_squared,
    # which is present under either key.
    if isinstance(best, dict) and "params" not in best and "parameters" in best:
        best = {**best, "params": best["parameters"]}
    # run_fit_isolated never returns "x"/"y" (only "x_fit"/"y_fit") -- but
    # step5_visualize.py and step6_export.py both key off "x"/"y" to plot
    # the overlay/residuals and write fitted_curve.csv. Attach them from
    # the already-loaded rheo_data so downstream consumers see real data.
    if isinstance(best, dict):
        best = {**best, "x": rheo_data.x, "y": rheo_data.y}
    return best


def _make_sample_fn(library, fit_state, active_jobs=None, status_bar=None):
    """Build the real sample_fn NutsStep.run() calls."""

    def _sample_fn(priors, warm_start, config):
        rheo_data = library.load_payload(fit_state.data_ref)
        # Same real, cancellable token as _fit_fn (see its comment) so
        # "Cancel them and continue?" can actually stop a running NUTS sample.
        # job_id carries a ":nuts" suffix (not the bare data_ref) -- see
        # _make_fit_fn's matching comment: NLSQ and NUTS running against the
        # same dataset must not collide on the same active_jobs.by_id key.
        job_id = f"{fit_state.data_ref}:nuts"
        token = ProcessCancellationToken(job_id=job_id)
        # See _fit_fn's comment above: _run_on_thread pumps a nested
        # QEventLoop, so New/Open/Close must see this NUTS run as active for
        # its whole duration, not just while a signal happens to be in flight.
        if active_jobs is not None:
            active_jobs.by_id[job_id] = {
                "status": "running",
                "worker": token,
            }
        if status_bar is not None:
            status_bar.show_progress(0, 100, "Sampling...")
        progress_queue = multiprocessing.Queue()
        try:
            return _run_on_thread(
                lambda: run_bayesian_isolated(
                    fit_state.model_key,
                    rheo_data.x,
                    rheo_data.y,
                    test_mode=fit_state.protocol,
                    num_warmup=config.get("num_warmup", 500),
                    num_samples=config.get("num_samples", 1000),
                    num_chains=config.get("num_chains", 4),
                    warm_start=warm_start,
                    priors=priors,
                    seed=config.get("seed", 0),
                    progress_queue=progress_queue,
                    cancel_event=token.event,
                    dataset_id=fit_state.data_ref,
                    target_accept=config.get("target_accept", 0.8),
                    model_config=fit_state.model_config,
                    metadata=getattr(rheo_data, "metadata", None),
                    max_tree_depth=config.get("max_tree_depth"),
                ),
                progress_queue=progress_queue,
                on_progress=_make_progress_reporter(active_jobs, job_id, status_bar),
            )
        finally:
            # Release the queue's feeder thread/fds/semaphore -- mirrors
            # ProcessWorkerAdapter._ensure_process_dead()'s queue cleanup.
            try:
                progress_queue.close()
                progress_queue.join_thread()
            except (OSError, ValueError, BrokenPipeError):
                pass
            if active_jobs is not None:
                active_jobs.by_id.pop(job_id, None)
            if status_bar is not None:
                status_bar.hide_progress()

    return _sample_fn


def build_fit_controller(app_state: AppState, status_bar=None):
    st = app_state.fit
    bodies = [
        ProtocolModelStep(st),
        DataStep(st, app_state.library),
        NlsqStep(
            st,
            fit_fn=_make_fit_fn(
                app_state.library, st, app_state.active_jobs, status_bar
            ),
            active_jobs=app_state.active_jobs,
        ),
        NutsStep(
            st,
            sample_fn=_make_sample_fn(
                app_state.library, st, app_state.active_jobs, status_bar
            ),
            active_jobs=app_state.active_jobs,
        ),
        VisualizeStep(st),
        ExportStep(st, app_state.library, app_state.active_jobs),
    ]
    validators = [
        lambda b=bodies[0]: b.is_ready(),
        lambda b=bodies[1]: b.is_ready() and not b.validation_errors(),
        lambda b=bodies[2]: b.is_ready(),
        lambda b=bodies[3]: b.is_ready(),  # True if skipped, per NutsStep.is_ready()
        lambda: True,  # Visualize is read-only
        lambda: True,  # Export just saves/writes
    ]
    # STEP_IDS doubles as the display title for every step except the first:
    # "protocol_model" reads as a raw identifier rather than a label, unlike
    # its single-word siblings ("data", "nlsq", ...), so it gets an explicit
    # human-facing override here. `id` stays STEP_IDS[i] unchanged.
    step_titles = dict(enumerate(FitController.STEP_IDS))
    step_titles[0] = "protocol & model"
    steps = [
        Step(
            id=FitController.STEP_IDS[i],
            title=step_titles[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=validators[i],
        )
        for i in range(len(bodies))
    ]
    ctl = FitController(steps)

    protocol_body, data_body, nlsq_body, nuts_body, visualize_body = bodies[:5]

    # Downstream refreshes that must observe cascaded/relocked state, not the
    # stale pre-edit state -- register_step connects cascade before these, in
    # that fixed order, so getting it right no longer depends on where these
    # .connect() calls happen to sit in the file.
    protocol_downstream = [
        fn
        for fn in (
            getattr(data_body, "refresh", None),
            getattr(nlsq_body, "load_parameters_from_model", None),
            getattr(nlsq_body, "refresh_display", None),
        )
        if fn is not None
    ]
    data_downstream = [
        fn for fn in (getattr(nlsq_body, "refresh_display", None),) if fn is not None
    ]
    nlsq_downstream = [
        fn for fn in (getattr(nuts_body, "reset_skip", None),) if fn is not None
    ]

    register_step(
        protocol_body,
        "edited",
        lambda: ctl.on_edit(0),
        changed=_CHANGED[0],
        live_state=app_state.fit,
        cascade_table=_FIT_CASCADE,
        clear_table=_CLEAR,
        downstream=protocol_downstream,
    )
    # ProtocolModelStep also fires config_edited for constructor-config
    # widget changes (model/protocol unchanged) -- those must cascade with
    # the narrower "model_config" key, not _CHANGED[0] == "model_key" (which
    # would wipe the model_config edit right back to {} via invalidation.py's
    # _CLEAR, silently discarding it before any real NLSQ/NUTS run saw it).
    register_step(
        protocol_body,
        "config_edited",
        lambda: ctl.on_edit(0),
        changed="model_config",
        live_state=app_state.fit,
        cascade_table=_FIT_CASCADE,
        clear_table=_CLEAR,
        downstream=protocol_downstream,
    )
    register_step(
        data_body,
        "edited",
        lambda: ctl.on_edit(1),
        changed=_CHANGED[1],
        live_state=app_state.fit,
        cascade_table=_FIT_CASCADE,
        clear_table=_CLEAR,
        downstream=data_downstream,
    )
    register_step(
        nlsq_body,
        "edited",
        lambda: ctl.on_edit(2),
        changed=_CHANGED[2],
        live_state=app_state.fit,
        cascade_table=_FIT_CASCADE,
        clear_table=_CLEAR,
        downstream=nlsq_downstream,
    )
    register_step(
        nuts_body,
        "edited",
        lambda: ctl.on_edit(3),
        changed=_CHANGED[3],
        live_state=app_state.fit,
        cascade_table=_FIT_CASCADE,
        clear_table=_CLEAR,
    )

    # Wire NLSQ finish -> NutsStep.load_suggested_priors, so the PriorsEditor
    # is seeded from the fresh MAP estimate as soon as NLSQ produces a result.
    # `finished` (a fit completing) is a different signal than `edited` (a
    # user edit) -- not part of the cascade/relock ordering above.
    if hasattr(nlsq_body, "finished") and hasattr(nuts_body, "load_suggested_priors"):
        nlsq_body.finished.connect(nuts_body.load_suggested_priors)

    # Wire NUTS finish -> Visualize refresh so Diagnostics tab appears after sampling
    if hasattr(nuts_body, "finished") and hasattr(visualize_body, "refresh"):
        nuts_body.finished.connect(visualize_body.refresh)

    return ctl, bodies
