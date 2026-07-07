"""
Subprocess Bayesian
==================

Pure function for NUTS Bayesian sampling in a child process.
No Qt dependencies -- all communication via mp.Queue and mp.Event.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from datetime import datetime
from typing import Any

import numpy as np

from rheojax.logging import get_logger

logger = get_logger(__name__)


def _posterior_draw_indices(
    posterior_samples: dict[str, Any], max_draws: int
) -> np.ndarray:
    """Select evenly spaced draw indices, consistent across parameters."""
    lengths = []
    for samples in (posterior_samples or {}).values():
        arr = np.asarray(samples).reshape(-1)
        if arr.size:
            lengths.append(int(arr.size))
    if not lengths:
        return np.array([], dtype=int)
    total = int(min(lengths))
    if total <= 0:
        return np.array([], dtype=int)
    n = int(min(max_draws, total))
    if n <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, total - 1, num=n)
    return np.unique(idx.astype(int))


def _posterior_params_at_index(
    posterior_samples: dict[str, Any], index: int
) -> dict[str, float]:
    """Extract a single posterior draw as a parameter dict."""
    params: dict[str, float] = {}
    for name, samples in (posterior_samples or {}).items():
        arr = np.asarray(samples).reshape(-1)
        if arr.size == 0 or index >= arr.size:
            continue
        value = float(arr[index])
        if np.isfinite(value):
            params[name] = value
    return params


def _compute_y_band(
    model_name: str,
    posterior_samples_np: dict[str, np.ndarray],
    x_data: np.ndarray,
    test_mode: str,
    model_config: dict[str, Any] | None,
    fitted_model_state: dict[str, Any] | None,
    hdi_prob: float = 0.94,
    max_draws: int = 50,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Posterior-predictive credible band (lo, hi) at each x, or None on failure.

    Mirrors bayesian_page.py's _update_fit_plot_from_posterior band math
    (same draw-sampling/quantile logic), minus the Qt-specific plotting so
    the workspace wizard's step5_visualize.py can render the same band the
    legacy page already computes.
    """
    from rheojax.gui.services.model_service import ModelService, infer_model_kwargs

    draw_indices = _posterior_draw_indices(posterior_samples_np, max_draws=max_draws)
    if draw_indices.size == 0:
        return None

    model_kwargs = dict(model_config or {}) or infer_model_kwargs(
        model_name, list(posterior_samples_np.keys())
    )
    if fitted_model_state:
        model_kwargs = dict(model_kwargs)
        model_kwargs["fitted_model_state"] = fitted_model_state

    model_service = ModelService()
    x = np.asarray(x_data)
    y_draws: list[np.ndarray] = []
    for idx in draw_indices:
        params = _posterior_params_at_index(posterior_samples_np, int(idx))
        if not params:
            continue
        try:
            y_pred = np.asarray(
                model_service.predict(
                    model_name, params, x, test_mode=test_mode, model_kwargs=model_kwargs
                )
            )
        except Exception:
            continue
        if (
            y_pred.ndim == 2
            and y_pred.shape[1] == 2
            and y_pred.shape[0] == len(x)
        ):
            y_pred = y_pred[:, 0] + 1j * y_pred[:, 1]
        if y_pred.shape == x.shape:
            y_draws.append(y_pred)

    if not y_draws:
        return None

    y_stack = np.stack(y_draws, axis=0)
    alpha = (1.0 - float(hdi_prob)) / 2.0
    q_lo, q_hi = float(alpha), float(1.0 - alpha)

    is_oscillation = (test_mode or "") == "oscillation"
    if is_oscillation and np.iscomplexobj(y_stack):
        y_re, y_im = np.real(y_stack), np.imag(y_stack)
        lo = np.nanquantile(y_re, q_lo, axis=0) + 1j * np.nanquantile(
            y_im, q_lo, axis=0
        )
        hi = np.nanquantile(y_re, q_hi, axis=0) + 1j * np.nanquantile(
            y_im, q_hi, axis=0
        )
    else:
        y_scalar = np.abs(y_stack) if np.iscomplexobj(y_stack) else y_stack
        lo = np.nanquantile(y_scalar, q_lo, axis=0)
        hi = np.nanquantile(y_scalar, q_hi, axis=0)
    return lo, hi


def run_bayesian_isolated(
    model_name: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    test_mode: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    warm_start: dict[str, float] | None,
    priors: dict[str, Any],
    seed: int,
    progress_queue: mp.Queue,
    cancel_event: mp.Event,
    y2_data: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    fitted_model_state: dict[str, Any] | None = None,
    dataset_id: str = "",
    target_accept: float = 0.8,
    model_config: dict[str, Any] | None = None,
    max_tree_depth: int | None = None,
) -> dict[str, Any]:
    """Run NUTS Bayesian sampling in an isolated subprocess.

    Returns a serializable dict with all arrays as NumPy.
    The ``inference_data`` field is always ``None`` because ArviZ
    InferenceData objects are not picklable and too large for mp.Queue.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g., "maxwell").
    x_data, y_data : np.ndarray
        Independent and dependent variables.
    test_mode : str
        Rheological test mode (relaxation, oscillation, creep, ...).
    num_warmup, num_samples, num_chains : int
        NUTS sampling parameters.
    warm_start : dict or None
        Initial parameter values from NLSQ fit.
    priors : dict
        Custom prior specifications per parameter.
    seed : int
        PRNG seed for reproducibility.
    progress_queue : mp.Queue
        Queue for progress messages back to parent.
    cancel_event : mp.Event
        Event set by parent to request cancellation.
    y2_data : np.ndarray or None
        Imaginary part for complex modulus (oscillation).
    metadata : dict or None
        Additional metadata for RheoData.

    fitted_model_state : dict or None
        Fitted model instance variables to transfer.
    dataset_id : str
        Identifier for the dataset being analysed.
    model_config : dict or None
        Constructor kwargs for the model (n_modes/variant/kinetics/...), same
        shape as ModelService.fit()'s model_config -- overrides the
        warm-start-name inference heuristic in BayesianService.run_mcmc().

    Returns
    -------
    dict
        Serializable result dict with posterior_samples as NumPy arrays.
    """
    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    from rheojax.core.data import RheoData
    from rheojax.gui.services.bayesian_service import BayesianService

    # Ensure models are registered in this (sub)process.
    # In a subprocess the @ModelRegistry.register decorators haven't fired yet
    # because the model modules use lazy __getattr__ imports.
    # Registry.discover() cannot trigger __getattr__ (inspect.getmembers uses
    # dir() which doesn't list lazy names), so we must eagerly import all
    # submodules to fire the decorators.
    from rheojax.models import _ensure_all_registered

    _ensure_all_registered()

    # RheoData stores complex modulus as y = G' + i*G'' (no separate y2 field)
    if y2_data is not None:
        y_combined = y_data + 1j * y2_data
    else:
        y_combined = y_data

    rheo_data = RheoData(
        x=x_data,
        y=y_combined,
        initial_test_mode=test_mode,
        metadata=metadata or {},
    )

    start_time = time.perf_counter()

    def progress_callback(
        stage: str,
        chain: int,
        iteration: int,
        total: int,
    ):
        if cancel_event.is_set():
            from rheojax.gui.jobs.cancellation import CancellationError

            raise CancellationError("Operation cancelled by user")
        percent = min(int(iteration / total * 100), 100) if total > 0 else 0
        message = f"{stage}: chain {chain}, iteration {iteration}/{total}"
        progress_queue.put(
            {
                "type": "progress",
                "percent": percent,
                "total": 100,
                "message": message,
            }
        )

    # Build kwargs for BayesianService.run_mcmc()
    mcmc_kwargs: dict[str, Any] = {
        "seed": seed,
        "target_accept_prob": target_accept,
    }

    if priors:
        mcmc_kwargs["custom_priors"] = priors
    if fitted_model_state is not None:
        mcmc_kwargs["fitted_model_state"] = fitted_model_state
    if model_config:
        mcmc_kwargs["model_config"] = model_config
    if max_tree_depth is not None:
        mcmc_kwargs["max_tree_depth"] = max_tree_depth

    service = BayesianService()
    bayesian_result = service.run_mcmc(
        model_name=model_name,
        data=rheo_data,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        warm_start=warm_start,
        test_mode=test_mode,
        progress_callback=progress_callback,
        **mcmc_kwargs,
    )

    mcmc_time = time.perf_counter() - start_time

    # Convert ALL posterior samples from JAX arrays to NumPy
    posterior_samples_np: dict[str, np.ndarray] = {}
    if bayesian_result.posterior_samples:
        for name, samples in bayesian_result.posterior_samples.items():
            posterior_samples_np[name] = np.asarray(samples)

    # Compute credible intervals (uses NumPy internally)
    credible_intervals: dict[str, tuple[float, float, float]] = {}
    try:
        credible_intervals = service.get_credible_intervals(bayesian_result)
    except Exception as exc:
        logger.error(
            "Credible interval computation failed; returning empty intervals",
            model_name=model_name,
            error=str(exc),
            exc_info=True,
        )

    # Extract sample_stats arrays for energy/divergence plots.
    # Full InferenceData is not picklable, but raw NumPy arrays are.
    sample_stats_np: dict[str, np.ndarray] = {}
    idata = getattr(bayesian_result, "inference_data", None)
    if idata is not None:
        ss = getattr(idata, "sample_stats", None)
        if ss is not None:
            for var_name in ("energy", "diverging"):
                if var_name in ss:
                    sample_stats_np[var_name] = np.asarray(ss[var_name].values)

    from rheojax.gui.foundation.metrics import bfmi as _bfmi

    energy_arr = sample_stats_np.get("energy")
    if energy_arr is not None:
        # energy_arr shape: (chains, draws); compute per-chain BFMI and average
        if energy_arr.ndim == 2:
            bfmi_val = float(
                np.mean([_bfmi(energy_arr[c]) for c in range(energy_arr.shape[0])])
            )
        else:
            bfmi_val = _bfmi(energy_arr)
    else:
        bfmi_val = float("nan")

    y_band = None
    try:
        y_band = _compute_y_band(
            model_name,
            posterior_samples_np,
            x_data,
            test_mode,
            model_config,
            fitted_model_state,
        )
    except Exception as exc:
        logger.error(
            "Posterior-predictive band computation failed; overlay will skip it",
            model_name=model_name,
            error=str(exc),
            exc_info=True,
        )

    return {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "posterior_samples": posterior_samples_np,
        "summary": bayesian_result.summary,
        "r_hat": dict(bayesian_result.r_hat) if bayesian_result.r_hat else {},
        "ess": dict(bayesian_result.ess) if bayesian_result.ess else {},
        "divergences": int(bayesian_result.divergences),
        "credible_intervals": credible_intervals,
        "mcmc_time": mcmc_time,
        "timestamp": datetime.now().isoformat(),
        "num_warmup": num_warmup,
        "num_samples": num_samples,
        "num_chains": num_chains,
        "inference_data": None,  # Not picklable, too large for mp.Queue
        "sample_stats": sample_stats_np,  # Raw arrays for energy plot
        "bfmi": bfmi_val,
        "diagnostics_valid": bayesian_result.diagnostics_valid,
        "y_band": y_band,
    }
