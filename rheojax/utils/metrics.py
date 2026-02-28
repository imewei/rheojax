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

    if ss_tot > 0:
        r2 = 1.0 - ss_res / ss_tot
    elif ss_res == 0.0:
        r2 = 1.0  # Perfect fit on constant data
    else:
        r2 = 0.0  # Non-zero residuals on constant data
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
        Coefficient of determination computed on magnitudes |G*|.

    Note
    ----
    This metric evaluates magnitude fit only. Phase errors (e.g., correct
    |G*| but wrong tan(δ)) are not captured. For phase-sensitive evaluation,
    use :func:`r2_complex_components` which averages R² over the real and
    imaginary components independently.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Use magnitudes for complex data
    if np.iscomplexobj(y_true) or np.iscomplexobj(y_pred):
        y_true = np.abs(y_true)
        y_pred = np.abs(y_pred)

    return compute_fit_quality(y_true, y_pred)["R2"]


def r2_complex_components(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute R² for complex data using separate real and imaginary components.

    Returns the arithmetic mean of R²(real) and R²(imag), capturing both
    magnitude and phase accuracy. A model that fits |G*| perfectly but has
    the wrong phase angle will score lower here than with :func:`r2_complex`.

    Parameters
    ----------
    y_true : array-like
        Observed complex values (e.g., G* = G' + i·G'').
    y_pred : array-like
        Predicted complex values.

    Returns
    -------
    float
        Average R² across real (G') and imaginary (G'') components.

    Examples
    --------
    >>> import numpy as np
    >>> omega = np.logspace(-2, 2, 50)
    >>> G_star = omega * 1j  # Pure viscous
    >>> r2_complex_components(G_star, G_star)
    1.0
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # R3-U-004: detect effectively-real complex data to avoid inflated R² from zero imaginary
    if np.iscomplexobj(y_true) or np.iscomplexobj(y_pred):
        max_imag_true = (
            np.max(np.abs(np.imag(y_true))) if np.iscomplexobj(y_true) else 0.0
        )
        max_imag_pred = (
            np.max(np.abs(np.imag(y_pred))) if np.iscomplexobj(y_pred) else 0.0
        )
        if max_imag_true < 1e-15 and max_imag_pred < 1e-15:
            y_true = np.real(y_true)
            y_pred = np.real(y_pred)

    if not np.iscomplexobj(y_true) and not np.iscomplexobj(y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        # R11-METRICS-001: Handle constant data (ss_tot == 0) without returning -1e30
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else -1.0
        return 1.0 - ss_res / float(ss_tot)

    ss_res_real = np.sum((np.real(y_true) - np.real(y_pred)) ** 2)
    ss_tot_real = np.sum((np.real(y_true) - np.mean(np.real(y_true))) ** 2)

    ss_res_imag = np.sum((np.imag(y_true) - np.imag(y_pred)) ** 2)
    ss_tot_imag = np.sum((np.imag(y_true) - np.mean(np.imag(y_true))) ** 2)

    # R11-METRICS-001: Handle constant data per component
    if ss_tot_real == 0.0:
        r2_real = 1.0 if ss_res_real == 0.0 else -1.0
    else:
        r2_real = 1.0 - ss_res_real / float(ss_tot_real)
    if ss_tot_imag == 0.0:
        r2_imag = 1.0 if ss_res_imag == 0.0 else -1.0
    else:
        r2_imag = 1.0 - ss_res_imag / float(ss_tot_imag)

    return (r2_real + r2_imag) / 2.0
