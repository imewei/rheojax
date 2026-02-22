"""Fit quality metrics for rheological model evaluation.

This module provides functions to compute standard statistical metrics
for evaluating model fit quality.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def compute_fit_quality(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> dict[str, float]:
    """Compute R² and RMSE fit quality metrics.

    Parameters
    ----------
    y_true : array-like
        Observed (ground truth) values.
    y_pred : array-like
        Predicted values from the model.

    Returns
    -------
    dict
        Dictionary containing:
        - 'R2': Coefficient of determination (R²)
        - 'RMSE': Root mean squared error
        - 'nrmse': Normalized RMSE (RMSE / range of y_true)

    Examples
    --------
    >>> y_true = [1.0, 2.0, 3.0, 4.0]
    >>> y_pred = [1.1, 1.9, 3.1, 3.9]
    >>> metrics = compute_fit_quality(y_true, y_pred)
    >>> metrics['R2'] > 0.99
    True
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # SUP-001: Reject complex inputs — use r2_complex() instead
    if np.iscomplexobj(y_true) or np.iscomplexobj(y_pred):
        raise TypeError(
            "compute_fit_quality does not support complex inputs. "
            "Use r2_complex() for complex-valued data."
        )

    # SUP-006: Return NaN metrics for empty input
    if y_true.size == 0 or y_pred.size == 0:
        return {"R2": float("nan"), "RMSE": float("nan"), "nrmse": float("nan")}

    # Flatten if multi-dimensional
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean(residuals**2))

    # Normalized RMSE
    y_range = np.ptp(y_true)  # peak-to-peak (max - min)
    nrmse = rmse / y_range if y_range > 0 else float("inf")

    return {"R2": float(r2), "RMSE": float(rmse), "nrmse": float(nrmse)}


def r2_complex(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute R² for complex-valued data using magnitudes.

    Parameters
    ----------
    y_true : array-like
        Observed complex values.
    y_pred : array-like
        Predicted complex values.

    Returns
    -------
    float
        Coefficient of determination computed on magnitudes.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Use magnitudes for complex data
    if np.iscomplexobj(y_true) or np.iscomplexobj(y_pred):
        y_true = np.abs(y_true)
        y_pred = np.abs(y_pred)

    return compute_fit_quality(y_true, y_pred)["R2"]
