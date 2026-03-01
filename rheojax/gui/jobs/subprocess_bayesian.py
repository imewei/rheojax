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
    deformation_mode: str | None = None,
    poisson_ratio: float | None = None,
    fitted_model_state: dict[str, Any] | None = None,
    dataset_id: str = "",
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
    deformation_mode : str or None
        Deformation mode (tension, shear, ...).
    poisson_ratio : float or None
        Poisson ratio for E* <-> G* conversion.
    fitted_model_state : dict or None
        Fitted model instance variables to transfer.
    dataset_id : str
        Identifier for the dataset being analysed.

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
    }
    if deformation_mode is not None:
        mcmc_kwargs["deformation_mode"] = deformation_mode
    if poisson_ratio is not None:
        mcmc_kwargs["poisson_ratio"] = poisson_ratio
    if priors:
        mcmc_kwargs["custom_priors"] = priors
    if fitted_model_state is not None:
        mcmc_kwargs["fitted_model_state"] = fitted_model_state

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
    except Exception:
        pass

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
        "diagnostics_valid": bayesian_result.diagnostics_valid,
    }
