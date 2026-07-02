from __future__ import annotations

import multiprocessing

import numpy as np

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


def _make_fit_fn(library):
    """Build the real (synchronous) fit_fn NlsqStep.run() calls.

    Runs subprocess_fit.run_fit_isolated to completion before returning --
    NlsqStep's control flow is fully synchronous today (no progress bar or
    cancel button exists yet), so a local, throwaway Queue/Event is enough.
    """

    def _fit_fn(model_key, model_config, data_ref, column_map, initial_params=None,
                multi_start=None):
        rheo_data = library.load_payload(data_ref)
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
                    name: {**cfg, "value": cfg["value"] * (1 + rng.uniform(-0.2, 0.2))}
                    for name, cfg in initial_params.items()
                }
            result = run_fit_isolated(
                model_key,
                rheo_data.x,
                rheo_data.y,
                test_mode=None,
                initial_params=start_params or {},
                options={},
                progress_queue=multiprocessing.Queue(),
                cancel_event=multiprocessing.Event(),
                dataset_id=data_ref,
                model_config=model_config,
            )
            if best is None or (result.get("r_squared") or -1) > (best.get("r_squared") or -1):
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
        return best

    return _fit_fn


def _make_sample_fn(library, fit_state):
    """Build the real (synchronous) sample_fn NutsStep.run() calls."""

    def _sample_fn(priors, warm_start, config):
        rheo_data = library.load_payload(fit_state.data_ref)
        return run_bayesian_isolated(
            fit_state.model_key,
            rheo_data.x,
            rheo_data.y,
            test_mode=None,
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
        )

    return _sample_fn


def build_fit_controller(app_state: AppState):
    st = app_state.fit
    bodies = [
        ProtocolModelStep(st),
        DataStep(st, app_state.library),
        NlsqStep(st, fit_fn=_make_fit_fn(app_state.library)),
        NutsStep(st, sample_fn=_make_sample_fn(app_state.library, st)),
        VisualizeStep(st),
        ExportStep(st, app_state.library),
    ]
    steps = [
        Step(
            id=FitController.STEP_IDS[i],
            title=FitController.STEP_IDS[i],
            is_ready=getattr(bodies[i], "is_ready", lambda: True),
            validate=lambda: True,
        )
        for i in range(len(bodies))
    ]
    ctl = FitController(steps)

    for i, body in enumerate(bodies):
        if hasattr(body, "edited"):

            def _on_edit(idx=i):
                ctl.on_edit(idx)
                changed = _CHANGED[idx]
                if changed:
                    new_fit = invalidate_downstream(app_state.fit, changed)
                    # Mutate in place so step bodies (which hold a reference to
                    # app_state.fit via `st`) see the cleared fields immediately.
                    # FitState is non-frozen, so setattr works without replace().
                    for attr, val in vars(new_fit).items():
                        setattr(app_state.fit, attr, val)

            body.edited.connect(_on_edit)

    # Wire Protocol/Model edits -> Data step refresh, so the contract/combo
    # rebuild after Step 1 completes (connected after the _on_edit loop above,
    # so invalidation/state-update runs first, then refresh sees fresh state).
    protocol_body, data_body = bodies[0], bodies[1]
    if hasattr(protocol_body, "edited") and hasattr(data_body, "refresh"):
        protocol_body.edited.connect(data_body.refresh)

    # Wire NLSQ edits -> NutsStep.reset_skip, so a stale "skipped" decision
    # doesn't survive an NLSQ re-run that invalidates nuts_result.
    nlsq_body, nuts_body = bodies[2], bodies[3]
    if hasattr(nlsq_body, "edited") and hasattr(nuts_body, "reset_skip"):
        nlsq_body.edited.connect(nuts_body.reset_skip)

    # Wire NUTS finish -> Visualize refresh so Diagnostics tab appears after sampling
    visualize_body = bodies[4]
    if hasattr(nuts_body, "finished") and hasattr(visualize_body, "refresh"):
        nuts_body.finished.connect(visualize_body.refresh)

    return ctl, bodies
