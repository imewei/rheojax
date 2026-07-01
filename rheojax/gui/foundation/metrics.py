"""
Result Metrics
==============

Pure numerical functions for post-fit quality metrics.
No JAX, no Qt — numpy only so they run in the subprocess and in tests.
"""

from __future__ import annotations

import numpy as np


def reduced_chi_squared(
    rss: float, n: int, k: int, sigma2: float | None = None
) -> float:
    """Reduced chi-squared goodness-of-fit statistic.

    Parameters
    ----------
    rss:
        Residual sum of squares (sum of squared residuals).
    n:
        Number of data points.
    k:
        Number of free parameters.
    sigma2:
        Per-point noise variance. If None the result is NaN — returning 1.0
        by setting sigma2=rss/dof would be trivially circular and misleading.

    Returns
    -------
    float
        rss / (sigma2 * dof), where dof = max(n - k, 1).
        Returns float('nan') when sigma2 is None.
    """
    dof = max(n - k, 1)
    if sigma2 is None:
        return float("nan")
    return float(rss / (sigma2 * dof))


def param_uncertainties(covariance) -> list[float]:
    """Parameter 1-sigma uncertainties from a covariance matrix.

    Parameters
    ----------
    covariance:
        Square covariance matrix (array-like), e.g. ``pcov`` from scipy/NLSQ.

    Returns
    -------
    list[float]
        ``[sqrt(cov[0,0]), sqrt(cov[1,1]), ...]``.
        Negative diagonal entries are clamped to 0 before the sqrt.
    """
    cov = np.asarray(covariance)
    return [float(np.sqrt(max(v, 0.0))) for v in np.diag(cov)]


def bfmi(energy) -> float:
    """Energy Bayesian Fraction of Missing Information (E-BFMI).

    Computed as ``mean(diff(E)^2) / var(E)``.  Values below ~0.3 indicate
    poor HMC/NUTS mixing.  A constant energy sequence returns 0.0.

    Parameters
    ----------
    energy:
        1-D array of Hamiltonian energy samples from a NUTS chain.

    Returns
    -------
    float
        E-BFMI value.
    """
    e = np.asarray(energy, dtype=float)
    denom = np.var(e)
    if denom == 0.0:
        return 0.0
    return float(np.mean(np.diff(e) ** 2) / denom)
