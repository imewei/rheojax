from __future__ import annotations

import math
import multiprocessing

import numpy as np
from PySide6.QtCore import QEventLoop, QObject, QRunnable, QThreadPool, Signal

from rheojax.gui.foundation.invalidation import invalidate_downstream
from rheojax.gui.foundation.state import AppState
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


def _run_on_thread(fn):
    """Run *fn* on a QThreadPool thread while pumping a local QEventLoop, so
    the GUI stays responsive -- mirrors transform_controller.py's
    _make_run_fn. The calling function still returns synchronously (or
    re-raises *fn*'s exception here, on the caller's thread), but the actual
    NLSQ/NUTS computation no longer blocks the GUI thread directly, which
    used to freeze the entire app (repaints, Cancel, everything) for the
    full duration of every real fit/sample run.
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
    QThreadPool.globalInstance().start(worker)
    loop.exec()

    if "error" in outcome:
        raise outcome["error"]
    return outcome["result"]


def _make_fit_fn(library, fit_state):
    """Build the real fit_fn NlsqStep.run() calls."""

    def _fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                multi_start=None):
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
            # First run uses the caller's initial_params as-is; restarts 2..N
            # jitter each value by +/-20% (seeded, so runs are reproducible).
            start_params = initial_params
            if i > 0 and initial_params:
                start_params = {
                    name: (
                        cfg
                        if cfg.get("fixed") is True
                        else {**cfg, "value": cfg["value"] * (1 + rng.uniform(-0.2, 0.2))}
                    )
                    for name, cfg in initial_params.items()
                }
            result = _run_on_thread(
                lambda start_params=start_params: run_fit_isolated(
                    model_key,
                    rheo_data.x,
                    rheo_data.y,
                    test_mode=test_mode,
                    initial_params=start_params or {},
                    options={},
                    progress_queue=multiprocessing.Queue(),
                    cancel_event=multiprocessing.Event(),
                    dataset_id=data_ref,
                    model_config=model_config,
                    metadata=metadata,
                )
            )
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

    return _fit_fn


def _make_sample_fn(library, fit_state):
    """Build the real sample_fn NutsStep.run() calls."""

    def _sample_fn(priors, warm_start, config):
        rheo_data = library.load_payload(fit_state.data_ref)
        return _run_on_thread(
            lambda: run_bayesian_isolated(
                fit_state.model_key,
                rheo_data.x,
                rheo_data.y,
                test_mode=fit_state.protocol,
                num_warmup=500,
                num_samples=1000,
                num_chains=4,
                warm_start=warm_start,
                priors=priors,
                seed=0,
                progress_queue=multiprocessing.Queue(),
                cancel_event=multiprocessing.Event(),
                dataset_id=fit_state.data_ref,
                target_accept=config.get("target_accept", 0.8),
                model_config=fit_state.model_config,
                metadata=getattr(rheo_data, "metadata", None),
            )
        )

    return _sample_fn


def build_fit_controller(app_state: AppState):
    st = app_state.fit
    bodies = [
        ProtocolModelStep(st),
        DataStep(st, app_state.library),
        NlsqStep(st, fit_fn=_make_fit_fn(app_state.library, st)),
        NutsStep(st, sample_fn=_make_sample_fn(app_state.library, st)),
        VisualizeStep(st),
        ExportStep(st, app_state.library),
    ]
    validators = [
        lambda b=bodies[0]: b.is_ready(),
        lambda b=bodies[1]: b.is_ready() and not b.validation_errors(),
        lambda b=bodies[2]: b.is_ready(),
        lambda b=bodies[3]: b.is_ready(),  # True if skipped, per NutsStep.is_ready()
        lambda: True,   # Visualize is read-only
        lambda: True,   # Export just saves/writes
    ]
    steps = [
        Step(
            id=FitController.STEP_IDS[i],
            title=FitController.STEP_IDS[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=validators[i],
        )
        for i in range(len(bodies))
    ]
    ctl = FitController(steps)

    def _cascade_and_relock(idx: int, changed: str | None) -> None:
        ctl.on_edit(idx)
        if changed:
            new_fit = invalidate_downstream(app_state.fit, changed)
            # Mutate in place so step bodies (which hold a reference to
            # app_state.fit via `st`) see the cleared fields immediately.
            # FitState is non-frozen, so setattr works without replace().
            for attr, val in vars(new_fit).items():
                setattr(app_state.fit, attr, val)

    for i, body in enumerate(bodies):
        if hasattr(body, "edited"):
            body.edited.connect(lambda idx=i: _cascade_and_relock(idx, _CHANGED[idx]))
        # ProtocolModelStep also fires config_edited for constructor-config
        # widget changes (model/protocol unchanged) -- those must cascade
        # with the narrower "model_config" key, not _CHANGED[idx] ==
        # "model_key" (which would wipe the model_config edit right back to
        # {} via invalidation.py's _CLEAR, silently discarding it before any
        # real NLSQ/NUTS run ever saw it).
        if hasattr(body, "config_edited"):
            body.config_edited.connect(lambda idx=i: _cascade_and_relock(idx, "model_config"))

    # Wire Protocol/Model edits -> Data step refresh, so the contract/combo
    # rebuild after Step 1 completes (connected after the _cascade_and_relock
    # loop above, so invalidation/state-update runs first, then refresh sees
    # fresh state).
    protocol_body, data_body = bodies[0], bodies[1]
    if hasattr(protocol_body, "edited") and hasattr(data_body, "refresh"):
        protocol_body.edited.connect(data_body.refresh)
    if hasattr(protocol_body, "config_edited") and hasattr(data_body, "refresh"):
        protocol_body.config_edited.connect(data_body.refresh)

    # Wire Protocol/Model edits -> NlsqStep.load_parameters_from_model, so the
    # ParameterTable stays seeded with whatever model is currently selected
    # (protocol-only edits are a safe no-op -- load_parameters_from_model()
    # guards on self._state.model_key being set). Constructor-config changes
    # (e.g. n_modes) can also change the model's parameter set, so
    # config_edited needs the same reseed.
    nlsq_body = bodies[2]
    if hasattr(protocol_body, "edited") and hasattr(nlsq_body, "load_parameters_from_model"):
        protocol_body.edited.connect(nlsq_body.load_parameters_from_model)
    if hasattr(protocol_body, "config_edited") and hasattr(nlsq_body, "load_parameters_from_model"):
        protocol_body.config_edited.connect(nlsq_body.load_parameters_from_model)

    # Wire every upstream edit that invalidates nlsq_result (protocol/model,
    # constructor-config, and data/column-map changes) -> NlsqStep.refresh_display,
    # so the "R²=..." label is cleared instead of showing a stale readout for
    # a fit that state.nlsq_result no longer reflects.
    if hasattr(protocol_body, "edited") and hasattr(nlsq_body, "refresh_display"):
        protocol_body.edited.connect(nlsq_body.refresh_display)
    if hasattr(protocol_body, "config_edited") and hasattr(nlsq_body, "refresh_display"):
        protocol_body.config_edited.connect(nlsq_body.refresh_display)
    if hasattr(data_body, "edited") and hasattr(nlsq_body, "refresh_display"):
        data_body.edited.connect(nlsq_body.refresh_display)

    # Wire NLSQ edits -> NutsStep.reset_skip, so a stale "skipped" decision
    # doesn't survive an NLSQ re-run that invalidates nuts_result.
    nuts_body = bodies[3]
    if hasattr(nlsq_body, "edited") and hasattr(nuts_body, "reset_skip"):
        nlsq_body.edited.connect(nuts_body.reset_skip)

    # Wire NLSQ finish -> NutsStep.load_suggested_priors, so the PriorsEditor
    # is seeded from the fresh MAP estimate as soon as NLSQ produces a result.
    if hasattr(nlsq_body, "finished") and hasattr(nuts_body, "load_suggested_priors"):
        nlsq_body.finished.connect(nuts_body.load_suggested_priors)

    # Wire NUTS finish -> Visualize refresh so Diagnostics tab appears after sampling
    visualize_body = bodies[4]
    if hasattr(nuts_body, "finished") and hasattr(visualize_body, "refresh"):
        nuts_body.finished.connect(visualize_body.refresh)

    return ctl, bodies
