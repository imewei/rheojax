"""
Subprocess Fit
=============

Pure function for NLSQ fitting in a child process.
No Qt dependencies — all communication via mp.Queue and mp.Event.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from datetime import datetime
from typing import Any

import numpy as np


def run_fit_isolated(
    model_name: str,
    x_data: np.ndarray,
    y_data: np.ndarray,
    test_mode: str,
    initial_params: dict[str, float],
    options: dict[str, Any],
    progress_queue: mp.Queue,
    cancel_event: mp.Event,
    y2_data: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    deformation_mode: str | None = None,
    poisson_ratio: float | None = None,
    dataset_id: str = "",
) -> dict[str, Any]:
    """Run NLSQ fitting in an isolated subprocess.

    Returns a serializable dict with all arrays as NumPy.
    """
    from rheojax.core.jax_config import safe_import_jax
    jax, jnp = safe_import_jax()

    from rheojax.core.data import RheoData
    from rheojax.gui.services.model_service import ModelService

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
        x=x_data, y=y_combined,
        initial_test_mode=test_mode, metadata=metadata or {},
    )

    start_time = time.perf_counter()
    fit_kwargs = options.copy()
    max_iter = int(fit_kwargs.get("max_iter", 100)) or 100

    if deformation_mode is not None:
        fit_kwargs["deformation_mode"] = deformation_mode
    if poisson_ratio is not None:
        fit_kwargs["poisson_ratio"] = poisson_ratio

    last_iteration = 0
    last_loss = float("inf")

    def progress_callback(iteration: int, loss: float, **kwargs):
        nonlocal last_iteration, last_loss
        if cancel_event.is_set():
            from rheojax.gui.jobs.cancellation import CancellationError
            raise CancellationError("Operation cancelled by user")
        last_iteration = iteration
        last_loss = loss
        percent = min(int(iteration / max_iter * 100), 100)
        message = f"Iteration {iteration}: loss = {loss:.6e}"
        progress_queue.put({
            "type": "progress", "percent": percent,
            "total": 100, "message": message,
        })

    service = ModelService()
    service_result = service.fit(
        model_name, rheo_data,
        params=initial_params,
        progress_callback=progress_callback,
        **fit_kwargs,
    )

    fit_time = time.perf_counter() - start_time

    if not getattr(service_result, "success", True):
        error_msg = getattr(service_result, "message", "Fit failed")
        return {"success": False, "error": error_msg, "model_name": model_name}

    def _to_numpy(arr):
        if arr is None:
            return None
        if hasattr(arr, "device"):  # JAX array
            return np.asarray(arr)
        return np.asarray(arr) if not isinstance(arr, np.ndarray) else arr

    svc_metadata = getattr(service_result, "metadata", {}) or {}

    return {
        "success": getattr(service_result, "success", False),
        "model_name": model_name,
        "parameters": dict(service_result.parameters),
        "r_squared": float(svc_metadata.get("r_squared", 0.0)),
        "mpe": float(svc_metadata.get("mpe", 0.0)),
        "chi_squared": float(getattr(service_result, "chi_squared", 0.0)),
        "fit_time": fit_time,
        "timestamp": datetime.now().isoformat(),
        "num_iterations": svc_metadata.get("n_iterations", last_iteration),
        "message": getattr(service_result, "message", ""),
        "dataset_id": dataset_id,
        "x_fit": _to_numpy(getattr(service_result, "x_fit", None)),
        "y_fit": _to_numpy(getattr(service_result, "y_fit", None)),
        "residuals": _to_numpy(getattr(service_result, "residuals", None)),
        "pcov": _to_numpy(getattr(service_result, "pcov", None)),
        "rmse": float(svc_metadata["rmse"]) if svc_metadata.get("rmse") is not None else None,
        "mae": float(svc_metadata["mae"]) if svc_metadata.get("mae") is not None else None,
        "aic": float(svc_metadata["aic"]) if svc_metadata.get("aic") is not None else None,
        "bic": float(svc_metadata["bic"]) if svc_metadata.get("bic") is not None else None,
        "metadata": {k: v for k, v in svc_metadata.items() if _is_serializable(v)},
    }


def _is_serializable(value: Any) -> bool:
    """Check if a value can be safely sent through mp.Queue."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_serializable(v) for v in value)
    if isinstance(value, dict):
        return all(_is_serializable(v) for v in value.values())
    if isinstance(value, np.ndarray):
        return True
    return False
