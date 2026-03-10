"""Optimization utilities for parameter fitting using NLSQ.

This module provides GPU-accelerated optimization using the NLSQ package
(https://github.com/imewei/NLSQ). NLSQ provides 5-270x speedup over scipy
through JAX JIT compilation and automatic differentiation.

Critical: This module imports NLSQ, which must be imported before JAX to
enable float64 precision mode. The rheojax package handles this automatically
in __init__.py.

Example:
    >>> from rheojax.core.parameters import ParameterSet
    >>> from rheojax.utils.optimization import nlsq_optimize
    >>>
    >>> # Set up parameters
    >>> params = ParameterSet()
    >>> params.add("x", value=1.0, bounds=(0, 10))
    >>>
    >>> # Define objective function
    >>> def objective(values):
    ...     x = values[0]
    ...     return (x - 5.0) ** 2
    >>>
    >>> # Optimize
    >>> result = nlsq_optimize(objective, params, use_jax=True)
    >>> print(f"Optimal x: {result.x[0]:.4f}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import nlsq
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()


type ArrayLike = np.ndarray | list | float

# OPT-002/OPT-020: Module-level frozenset of RheoJAX-specific kwargs that must
# be filtered before forwarding to NLSQ/SciPy optimizers (prevents TypeError).
_RHEOJAX_RESERVED_KWARGS: frozenset[str] = frozenset(
    {
        "test_mode",
        "deformation_mode",
        "poisson_ratio",
        "seed",
        "method",
        "num_warmup",
        "num_samples",
        "num_chains",
        "gamma_dot",
        "sigma",
        "sigma_applied",
        "gamma_0",
        "omega_laos",
        "return_components",
        "return_full",
        "t_wait",
        "n_cycles",
        # FIKH/FMLIKH-specific protocol kwargs (F-003)
        "strain",
        "sigma_0",
        "T_init",
        "T",
        # Additional protocol kwargs (R3-U-005)
        "omega",
        "lam_init",
        "lam_0",
        "sigma_init",
        "points_per_cycle",
    }
)


def _validate_optimization_result(
    result: OptimizationResult,
    residuals: np.ndarray,
    y_data: np.ndarray | None = None,
    mse_threshold: float = 1e18,
) -> None:
    """Validate optimization result against pathological outcomes.

    Checks for non-finite or astronomically large residuals that indicate
    the optimizer "succeeded" numerically but produced meaningless parameters.

    Args:
        result: OptimizationResult to validate (uses result.fun for RSS).
        residuals: Residual vector at the optimal point.
        y_data: Original y data array. When provided, uses len(y_data) as the
            observation count so that complex data (where residuals has length
            2N) is not penalised with an inflated denominator.
        mse_threshold: Maximum allowed mean squared error (default: 1e18).
            The default accommodates GPa-scale moduli data where raw residuals
            can be O(1e9 Pa), yielding MSS/N ~ O(1e18 Pa²).
            For normalized residuals the threshold is effectively unbounded
            (RSS/N ≈ O(1) << 1e18).  Only truly diverged fits (non-finite or
            beyond 1e18) are rejected.

    Raises:
        RuntimeError: If MSE exceeds threshold or is non-finite.
    """
    if residuals.size == 0:
        raise RuntimeError(
            "Optimization produced empty residual vector. "
            "Check that the objective function returns a non-empty array."
        )
    residual_count = len(y_data) if y_data is not None else residuals.size
    # R10-OPT-001: auto-scale threshold by data magnitude so that the MSE
    # check works for both dimensionless log-residuals and raw Pa-scale data.
    # Fallback to 1e18 when y_data is unavailable or all-zero.
    if y_data is not None and len(y_data) > 0:
        y_scale = float(np.max(np.abs(np.asarray(y_data))))
        if y_scale > 0:
            mse_threshold = max(mse_threshold, 1e6 * y_scale**2)
    mse = result.fun / residual_count
    if not np.isfinite(mse) or mse > mse_threshold:
        logger.error(
            "Optimization failed: residual norm extremely large",
            mean_squared_error=float(mse) if np.isfinite(mse) else "inf",
            residual_count=residual_count,
            rss=float(result.fun),
        )
        raise RuntimeError(
            "Optimization failed: residual norm remains extremely large. "
            "Try providing better initial values, looser bounds, or scaling the data."
        )


def _extract_bounds(
    parameters: ParameterSet,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Extract initial values and bounds arrays from a ParameterSet.

    # TODO: Planned enhancement — log-transform bounded parameters before
    # passing to the optimizer and inverse-transform on the way out. This
    # would improve conditioning for parameters that span many orders of
    # magnitude (e.g. moduli 1e0–1e9, time constants 1e-6–1e3). Requires
    # careful integration with the NLSQ library's bound handling and the
    # complex-residual split path in create_least_squares_objective.

    Converts ParameterSet bounds (which may contain None) into the
    (lower_array, upper_array) format expected by SciPy and NLSQ.

    Args:
        parameters: ParameterSet with initial values and bounds.

    Returns:
        (x0, (lower_bounds, upper_bounds)) where x0 is the initial values
        array and bounds are float64 arrays with -inf/+inf for missing bounds.
    """
    x0 = np.asarray(parameters.get_values(), dtype=np.float64)
    bounds_list = parameters.get_bounds()

    lower_list: list[float] = []
    upper_list: list[float] = []
    for bound_pair in bounds_list:
        if bound_pair is None or (bound_pair[0] is None and bound_pair[1] is None):
            lower_list.append(-np.inf)
            upper_list.append(np.inf)
        else:
            lower_list.append(bound_pair[0] if bound_pair[0] is not None else -np.inf)
            upper_list.append(bound_pair[1] if bound_pair[1] is not None else np.inf)

    lower = np.asarray(lower_list, dtype=np.float64)
    upper = np.asarray(upper_list, dtype=np.float64)
    return x0, (lower, upper)


def _run_scipy_least_squares(
    objective: Callable[[np.ndarray], float | np.ndarray],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    ftol: float,
    xtol: float,
    gtol: float,
    max_iter: int,
) -> OptimizationResult:
    """Run SciPy's TRF least squares and return an OptimizationResult.

    Shared implementation for both the explicit method='scipy' path and the
    NLSQ failure fallback path. Computes covariance from the Jacobian when
    available.

    Args:
        objective: Residual function (values -> residual vector or scalar).
        x0: Initial parameter values.
        bounds: (lower_bounds, upper_bounds) arrays.
        ftol: Function tolerance.
        xtol: Parameter tolerance.
        gtol: Gradient tolerance.
        max_iter: Maximum iterations (max_nfev = max_iter * 10).

    Returns:
        OptimizationResult with optimal parameters and covariance.
    """
    from scipy.optimize import least_squares as scipy_least_squares

    def residual_fn(values: np.ndarray) -> np.ndarray:
        res = objective(values)
        # OPT-007: Use hasattr check instead of isinstance(res, jnp.ndarray)
        # which is unreliable on JAX >= 0.4.7
        if hasattr(res, "devices"):
            res = np.asarray(res)
        res = np.asarray(res)
        if np.iscomplexobj(res):
            res = np.concatenate([np.real(res), np.imag(res)])
        res = res.astype(np.float64)
        # Guard against NaN/Inf from ODE solvers — replace with large finite
        # penalty so scipy can still attempt to optimize (gradient-guided away
        # from the bad region). Without this, scipy raises ValueError at init.
        # UTILS-001: Apply nan_to_num first so NaN gets 1e10 (not 0.0 from sign(NaN))
        if not np.all(np.isfinite(res)):
            res = np.nan_to_num(res, nan=1e10, posinf=1e10, neginf=-1e10)
        return res

    scipy_result = scipy_least_squares(
        residual_fn,
        x0,
        bounds=bounds,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        max_nfev=max_iter * 10,
        method="trf",
    )

    cost_value = getattr(scipy_result, "cost", None)
    jac = None
    pcov = None
    if scipy_result.jac is not None:
        jac = np.asarray(scipy_result.jac, dtype=np.float64)
        residuals = residual_fn(scipy_result.x)
        pcov = compute_covariance_from_jacobian(jac, residuals)

    # P1-5: Include residuals, y_data, n_data, and _is_complex_split so that
    # downstream statistics (R², AIC, adj-R²) can be computed correctly.
    final_residuals = residual_fn(scipy_result.x)
    rss = float(np.sum(final_residuals**2))

    return OptimizationResult(
        x=np.asarray(scipy_result.x, dtype=np.float64),
        fun=(
            float(2.0 * scipy_result.cost)
            if hasattr(scipy_result, "cost")
            else rss
        ),
        jac=jac,
        pcov=pcov,
        success=bool(scipy_result.success),
        message=str(scipy_result.message),
        nit=int(getattr(scipy_result, "nit", scipy_result.nfev)),
        nfev=int(scipy_result.nfev),
        njev=int(getattr(scipy_result, "njev", 0)),
        optimality=(
            float(getattr(scipy_result, "optimality", np.nan))
            if getattr(scipy_result, "optimality", None) is not None
            else None
        ),
        active_mask=(
            np.asarray(scipy_result.active_mask)
            if getattr(scipy_result, "active_mask", None) is not None
            else None
        ),
        cost=float(cost_value) if cost_value is not None else None,
        residuals=final_residuals,
    )


def _run_differential_evolution(
    objective: Callable[[np.ndarray], float | np.ndarray],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    max_iter: int,
) -> OptimizationResult:
    """Run SciPy's differential_evolution as a last-resort global optimizer.

    Used as the final fallback in the ``workflow="auto_global"`` chain after
    both NLSQ and SciPy TRF least-squares have failed to converge. Differential
    evolution performs a population-based global search that is robust to
    non-convex and multimodal objective landscapes.

    The objective is wrapped to return a scalar RSS so that
    ``differential_evolution`` (which minimises a scalar) can consume the same
    residual functions used elsewhere.

    Args:
        objective: Residual function (params -> residual vector or scalar).
        x0: Initial parameter values — seeded into the initial population
            so that the global search starts from (or near) the best local
            solution found so far.
        bounds: (lower_bounds, upper_bounds) arrays.  Any infinite bound is
            clamped to ±1e10 so that ``differential_evolution`` can draw
            a finite initial population.
        max_iter: Maximum number of generations (``maxiter`` in SciPy).

    Returns:
        OptimizationResult compatible with the rest of the optimisation chain.
    """
    from scipy.optimize import differential_evolution

    lower, upper = bounds

    # differential_evolution requires finite bounds; clamp ±inf to ±1e10.
    finite_lower = np.where(np.isfinite(lower), lower, -1e10)
    finite_upper = np.where(np.isfinite(upper), upper, 1e10)
    de_bounds = list(zip(finite_lower.tolist(), finite_upper.tolist(), strict=True))

    # Seed x0 into the initial population so DE starts near the best local
    # solution.  We reshape x0 into a (1, n_params) array that
    # differential_evolution accepts via ``init``.
    x0_pop = x0.reshape(1, -1)

    def scalar_objective(values: np.ndarray) -> float:
        res = np.asarray(objective(values))
        if np.iscomplexobj(res):
            res = np.concatenate([np.real(res), np.imag(res)])
        res = res.astype(np.float64)
        if not np.all(np.isfinite(res)):
            res = np.nan_to_num(res, nan=1e10, posinf=1e10, neginf=-1e10)
        return float(np.sum(res**2))

    logger.info(
        "Running differential_evolution global optimizer",
        n_params=len(x0),
        maxiter=max_iter,
    )
    de_result = differential_evolution(
        scalar_objective,
        de_bounds,
        maxiter=max_iter,
        tol=1e-6,
        seed=0,
        polish=True,  # final local refinement with L-BFGS-B
        x0=x0_pop,  # seed best local solution into initial population
    )

    x_opt = np.asarray(de_result.x, dtype=np.float64)
    rss = float(de_result.fun)

    return OptimizationResult(
        x=x_opt,
        fun=rss,
        jac=None,
        pcov=None,
        success=bool(de_result.success),
        message=str(de_result.message),
        nit=int(getattr(de_result, "nit", 0)),
        nfev=int(getattr(de_result, "nfev", 0)),
        njev=0,
        optimality=None,
        active_mask=None,
        cost=rss,
    )


def compute_covariance_from_jacobian(
    jac: np.ndarray,
    residuals: np.ndarray | None = None,
    n_data: int | None = None,
) -> np.ndarray | None:
    """Compute parameter covariance matrix from Jacobian via SVD.

    Uses SVD-based Moore-Penrose pseudo-inverse for numerical stability:
        pcov = VT.T @ diag(1/s²) @ VT

    Scaled by residual variance when residuals provided:
        pcov *= RSS / (n_data - n_params)

    Args:
        jac: Jacobian matrix (m x n), where m = data points, n = parameters
        residuals: Optional residual vector for scaling
        n_data: Number of data points (default: inferred from jac.shape[0])

    Returns:
        Covariance matrix (n x n), or None if computation fails
    """
    if jac is None or jac.size == 0:
        logger.debug("Jacobian is None or empty, cannot compute covariance")
        return None

    try:
        jac = np.asarray(jac, dtype=np.float64)
        m, n = jac.shape  # m = data points, n = parameters
        logger.debug(
            "Computing covariance from Jacobian",
            jacobian_shape=(m, n),
            n_data_points=m,
            n_params=n,
        )

        # SVD of Jacobian: J = U @ S @ VT
        U, s, VT = np.linalg.svd(jac, full_matrices=False)
        logger.debug(
            "SVD computed",
            singular_values_range=(float(s.min()), float(s.max())),
            condition_number=float(s.max() / s.min()) if s.min() > 0 else float("inf"),
        )

        # Filter near-zero singular values
        threshold = np.finfo(np.float64).eps * max(m, n) * s[0]
        # Use safe division to avoid RuntimeWarning: divide by zero
        s_safe = np.where(s > threshold, s, np.inf)
        s_inv_sq = np.where(s > threshold, 1.0 / (s_safe**2), 0.0)
        n_filtered = np.sum(s <= threshold)
        if n_filtered > 0:
            logger.debug(
                "Filtered near-zero singular values",
                n_filtered=int(n_filtered),
                threshold=float(threshold),
            )

        # Compute covariance: (J.T @ J)^-1 = VT.T @ diag(1/s²) @ VT
        pcov = VT.T @ np.diag(s_inv_sq) @ VT

        # Scale by residual variance if available
        if residuals is not None:
            residuals = np.asarray(residuals, dtype=np.float64).ravel()
            rss = np.sum(residuals**2)
            n_data_actual = n_data if n_data is not None else m
            dof = n_data_actual - n  # degrees of freedom
            if dof > 0:
                pcov = pcov * (rss / dof)
                logger.debug(
                    "Scaled covariance by residual variance",
                    rss=float(rss),
                    degrees_of_freedom=dof,
                    scale_factor=float(rss / dof),
                )

        # Validate result
        if not np.all(np.isfinite(pcov)):
            logger.warning(
                "Covariance matrix contains inf/nan, returning None",
                has_inf=bool(np.any(np.isinf(pcov))),
                has_nan=bool(np.any(np.isnan(pcov))),
            )
            return None

        logger.debug(
            "Covariance computation completed",
            pcov_shape=pcov.shape,
            pcov_diagonal_range=(
                float(np.diag(pcov).min()),
                float(np.diag(pcov).max()),
            ),
        )
        return pcov

    except Exception as e:
        logger.error(
            "Failed to compute covariance from Jacobian",
            error=str(e),
            exc_info=True,
        )
        return None


@dataclass
class OptimizationResult:
    """Result from optimization with NLSQ 0.6.6 CurveFitResult-compatible properties.

    This dataclass stores the results of NLSQ optimization, including optimal
    parameter values, objective function value, convergence information, and
    statistical metrics compatible with NLSQ 0.6.6's CurveFitResult.

    Attributes:
        x: Optimal parameter values (float64 array)
        fun: Objective function value at optimum (RSS = sum of squared residuals)
        jac: Jacobian (gradient) at optimum
        pcov: Parameter covariance matrix (n_params x n_params)
        success: Whether optimization converged successfully
        message: Status message from optimizer
        nit: Number of iterations
        nfev: Number of function evaluations
        njev: Number of Jacobian evaluations
        optimality: Optimality metric (gradient norm)
        active_mask: Active bound constraints at solution
        cost: Final cost value
        grad: Final gradient
        nlsq_result: Full NLSQ result dictionary (for advanced diagnostics)
        residuals: Residual vector (y_data - y_pred) for statistical metrics
        y_data: Original dependent variable data (for R² computation)
        n_data: Number of data points (for AIC/BIC computation)
        diagnostics: Model health diagnostics (NLSQ 0.6.6, when compute_diagnostics=True)

    Statistical Properties (NLSQ 0.6.6 CurveFitResult compatible):
        r_squared: Coefficient of determination (R²)
        adj_r_squared: Adjusted R² accounting for number of parameters
        rmse: Root mean squared error
        mae: Mean absolute error
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion

    Methods:
        confidence_intervals(alpha): Compute parameter confidence intervals
        prediction_interval(x_new, alpha): Compute prediction intervals (NLSQ 0.6.6)
        get_parameter_uncertainties(): Get standard errors from covariance diagonal
    """

    x: np.ndarray
    fun: float
    jac: np.ndarray | None = None
    pcov: np.ndarray | None = None
    success: bool = False
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    optimality: float | None = None
    active_mask: np.ndarray | None = None
    cost: float | None = None
    grad: np.ndarray | None = None
    nlsq_result: dict[str, Any] | None = field(default=None, repr=False)
    # Fields for statistical metrics (NLSQ 0.6.6 compatibility)
    residuals: np.ndarray | None = field(default=None, repr=False)
    y_data: np.ndarray | None = field(default=None, repr=False)
    n_data: int | None = None
    # NLSQ 0.6.6 fields for native delegation
    diagnostics: dict[str, Any] | None = field(default=None, repr=False)
    _curve_fit_result: Any | None = field(default=None, repr=False)
    _model_fn: Callable | None = field(default=None, repr=False)
    _x_data: np.ndarray | None = field(default=None, repr=False)
    # OPT-AIC-BIC-001: True when residuals are concatenated [real, imag] from
    # complex data.  Used by _resolve_n_data() to halve the residual vector
    # length to recover the true observation count N.
    _is_complex_split: bool = field(default=False, repr=False)
    # P1-6: When residuals are normalized (divided by weights), store the
    # normalization weights so that R²/AIC/BIC can un-normalize for correct
    # statistics.  Shape matches residuals.  None when residuals are raw.
    _normalization_weights: np.ndarray | None = field(default=None, repr=False)

    def _resolve_n_data(self) -> int:
        """Resolve the true observation count N.

        Priority: n_data > len(y_data) > residual-length (halved when
        ``_is_complex_split`` is True, since the residual vector is
        ``[real, imag]`` with length 2N).
        """
        if self.n_data is not None:
            return self.n_data
        if self.y_data is not None:
            return len(self.y_data)
        if self.residuals is not None:
            raw_len = len(self.residuals)
            return raw_len // 2 if self._is_complex_split else raw_len
        return 0

    # =========================================================================
    # Statistical Properties (NLSQ 0.6.0 CurveFitResult compatible)
    # =========================================================================

    @property
    def r_squared(self) -> float | None:
        """Coefficient of determination (R²).

        Measures goodness of fit. Range: (-∞, 1], where 1 is perfect fit.

        R² = 1 - SS_res / SS_tot

        where SS_res = sum((y - y_pred)²) and SS_tot = sum((y - y_mean)²)

        Returns:
            R² value, or None if residuals/y_data not available
        """
        if self.residuals is None or self.y_data is None:
            return None

        # Handle complex data by using magnitude
        y_data = np.asarray(self.y_data)
        residuals = np.asarray(self.residuals)

        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)

        # P1-6: Un-normalize residuals when normalization weights are stored,
        # so that ss_res is in the same units as ss_tot (raw data units).
        if self._normalization_weights is not None:
            residuals = residuals * np.asarray(self._normalization_weights)

        if len(residuals) == 2 * len(y_data):
            half = len(y_data)
            ss_res = np.sum(residuals[:half] ** 2) + np.sum(residuals[half:] ** 2)
            if np.iscomplexobj(y_data):
                ss_tot = np.sum((y_data.real - np.mean(y_data.real)) ** 2) + np.sum(
                    (y_data.imag - np.mean(y_data.imag)) ** 2
                )
            else:
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        else:
            ss_res = np.sum(residuals**2)
            if np.iscomplexobj(y_data):
                y_data = np.abs(y_data)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)

        if ss_tot == 0:
            logger.warning(
                "Total sum of squares is zero (constant data). R² undefined."
            )
            return np.nan

        return float(1 - (ss_res / ss_tot))

    @property
    def adj_r_squared(self) -> float | None:
        """Adjusted R² accounting for number of parameters.

        Adj R² = 1 - (1 - R²) * (n - 1) / (n - p - 1)

        where n is number of data points and p is number of parameters.

        Returns:
            Adjusted R² value, or None if cannot be computed
        """
        r2 = self.r_squared
        if r2 is None:
            return None

        n = (
            self.n_data
            if self.n_data is not None
            else (len(self.y_data) if self.y_data is not None else None)
        )
        if n is None or n == 0:
            return None

        p = self.x.size
        if n - p - 1 <= 0:
            logger.warning("Not enough degrees of freedom for adjusted R².")
            return np.nan

        return float(1 - (1 - r2) * (n - 1) / (n - p - 1))

    @property
    def rmse(self) -> float | None:
        """Root mean squared error.

        RMSE = sqrt(mean(residuals²))

        Returns:
            RMSE value, or None if residuals not available
        """
        if self.residuals is None:
            return None

        residuals = np.asarray(self.residuals)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)

        return float(np.sqrt(np.mean(residuals**2)))

    @property
    def mae(self) -> float | None:
        """Mean absolute error.

        MAE = mean(abs(residuals))

        More robust to outliers than RMSE.

        Returns:
            MAE value, or None if residuals not available
        """
        if self.residuals is None:
            return None

        residuals = np.asarray(self.residuals)
        return float(np.mean(np.abs(residuals)))

    @property
    def aic(self) -> float | None:
        """Akaike Information Criterion.

        AIC = 2k + n*ln(RSS/n)

        where k is number of parameters, n is number of data points,
        and RSS is residual sum of squares.

        Lower is better. Used for model selection.

        Returns:
            AIC value, or None if cannot be computed
        """
        if self.residuals is None:
            return None

        residuals = np.asarray(self.residuals)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)

        # P1-6/P1-7: Un-normalize residuals for correct AIC computation.
        if self._normalization_weights is not None:
            residuals = residuals * np.asarray(self._normalization_weights)

        n = self._resolve_n_data()
        if n == 0:
            return None

        k = self.x.size
        rss = np.sum(residuals**2)

        if rss <= 0:
            logger.warning("RSS ≤ 0, AIC undefined.")
            return np.nan

        return float(2 * k + n * np.log(rss / n))

    @property
    def bic(self) -> float | None:
        """Bayesian Information Criterion.

        BIC = k*ln(n) + n*ln(RSS/n)

        where k is number of parameters, n is number of data points,
        and RSS is residual sum of squares.

        Lower is better. Penalizes model complexity more than AIC.

        Returns:
            BIC value, or None if cannot be computed
        """
        if self.residuals is None:
            return None

        residuals = np.asarray(self.residuals)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)

        # P1-6/P1-7: Un-normalize residuals for correct BIC computation.
        if self._normalization_weights is not None:
            residuals = residuals * np.asarray(self._normalization_weights)

        n = self._resolve_n_data()
        if n == 0:
            return None

        k = self.x.size
        rss = np.sum(residuals**2)

        if rss <= 0:
            logger.warning("RSS ≤ 0, BIC undefined.")
            return np.nan

        return float(k * np.log(n) + n * np.log(rss / n))

    # =========================================================================
    # Statistical Methods (NLSQ 0.6.0 CurveFitResult compatible)
    # =========================================================================

    def confidence_intervals(self, alpha: float = 0.95) -> np.ndarray | None:
        """Compute parameter confidence intervals.

        Parameters
        ----------
        alpha : float, optional
            Confidence level (default: 0.95 for 95% CI).

        Returns
        -------
        intervals : ndarray or None
            Array of shape (n_params, 2) with [lower, upper] bounds for each
            parameter, or None if covariance not available.

        Examples
        --------
        >>> result = nlsq_optimize(objective, params)
        >>> ci = result.confidence_intervals(alpha=0.95)
        >>> if ci is not None:
        ...     for i, (lower, upper) in enumerate(ci):
        ...         print(f"Parameter {i}: [{lower:.3f}, {upper:.3f}]")
        """
        if self.pcov is None:
            return None

        from scipy import stats

        n = self._resolve_n_data()
        p = self.x.size

        # Degrees of freedom
        dof = max(n - p, 1)

        # t-value for confidence level
        t_val = stats.t.ppf((1 + alpha) / 2, dof)

        # Standard errors from covariance diagonal
        # OPT-006: Guard against negative covariance diagonals from
        # near-singular Jacobians
        perr = np.sqrt(np.maximum(np.diag(self.pcov), 0.0))

        # Confidence intervals
        intervals = np.zeros((p, 2))
        intervals[:, 0] = self.x - t_val * perr  # Lower bound
        intervals[:, 1] = self.x + t_val * perr  # Upper bound

        return intervals

    def get_parameter_uncertainties(self) -> np.ndarray | None:
        """Get standard errors for parameters from covariance diagonal.

        Returns
        -------
        uncertainties : ndarray or None
            Standard errors for each parameter, or None if covariance not available.

        Examples
        --------
        >>> result = nlsq_optimize(objective, params)
        >>> std_errs = result.get_parameter_uncertainties()
        >>> if std_errs is not None:
        ...     for i, se in enumerate(std_errs):
        ...         print(f"Parameter {i}: {result.x[i]:.4f} ± {se:.4f}")
        """
        if self.pcov is None:
            return None

        return np.sqrt(np.maximum(np.diag(self.pcov), 0.0))

    def prediction_interval(
        self,
        x_new: np.ndarray | None = None,
        alpha: float = 0.95,
    ) -> np.ndarray | None:
        """Compute prediction intervals for new x values.

        Prediction intervals account for both parameter uncertainty and
        observation noise, providing bounds where future observations are
        expected to fall with the specified probability.

        Parameters
        ----------
        x_new : ndarray or None, optional
            New x values for prediction. If None, uses original x_data.
        alpha : float, optional
            Confidence level for intervals (default: 0.95 for 95% PI).

        Returns
        -------
        intervals : ndarray or None
            Array of shape (n_points, 2) with [lower, upper] bounds for each
            point, or None if prediction intervals cannot be computed.

        Notes
        -----
        When a native NLSQ CurveFitResult is available (from nlsq_curve_fit),
        this method delegates to NLSQ's prediction_interval for accuracy.
        Otherwise, it falls back to a manual computation using covariance
        propagation.

        Examples
        --------
        >>> result = nlsq_curve_fit(model, x_data, y_data, params)
        >>> pi = result.prediction_interval(x_new, alpha=0.95)
        >>> if pi is not None:
        ...     for i, (lower, upper) in enumerate(pi):
        ...         print(f"x={x_new[i]:.2f}: [{lower:.3f}, {upper:.3f}]")
        """
        # Delegate to native NLSQ CurveFitResult when available
        if self._curve_fit_result is not None:
            try:
                return self._curve_fit_result.prediction_interval(x_new, alpha)
            except Exception as e:
                logger.debug(
                    "Native prediction_interval failed, using fallback",
                    error=str(e),
                )

        # Fallback: manual computation requires model function and data
        if self._model_fn is None or self.pcov is None:
            logger.debug("Cannot compute prediction interval: missing model_fn or pcov")
            return None

        x_eval = x_new if x_new is not None else self._x_data
        if x_eval is None:
            logger.debug("Cannot compute prediction interval: no x data")
            return None

        from scipy import stats

        x_eval = np.asarray(x_eval, dtype=np.float64)
        n = self._resolve_n_data()
        p = self.x.size
        dof = max(n - p, 1)

        # t-value for prediction interval
        t_val = stats.t.ppf((1 + alpha) / 2, dof)

        # Compute predictions and standard errors via numerical differentiation
        try:
            y_pred = np.asarray(self._model_fn(x_eval, self.x))

            # Estimate prediction variance using residual variance + parameter uncertainty
            if self.residuals is not None:
                residuals = np.asarray(self.residuals)
                if np.iscomplexobj(residuals):
                    residuals = np.abs(residuals)
                mse = np.sum(residuals**2) / dof
            else:
                mse = (self.fun / n) if n > 0 else 0.0

            # P2-Fit-3: Use leverage-weighted prediction intervals when the
            # Jacobian is available: h_i = diag(J @ inv(J^T J) @ J^T), then
            # pred_std_i = sqrt(mse * (1 + h_i)).  Falls back to constant MSE
            # when Jacobian is unavailable (approximate but conservative).
            if self.jac is not None:
                try:
                    J = np.asarray(self.jac, dtype=np.float64)
                    # Compute J at x_new if different from training data
                    # For now, only use training-point leverages when x_new matches
                    JtJ = J.T @ J
                    JtJ_inv = np.linalg.inv(JtJ + 1e-12 * np.eye(JtJ.shape[0]))
                    hat_matrix_diag = np.sum((J @ JtJ_inv) * J, axis=1)
                    # hat_matrix_diag has shape (n_residuals,); map to x_eval length
                    if len(hat_matrix_diag) == len(x_eval):
                        pred_std = np.sqrt(mse * (1.0 + hat_matrix_diag))
                    else:
                        # Residual length != x_eval length (e.g. complex-split)
                        pred_std = np.sqrt(mse) * np.ones_like(y_pred)
                except Exception:
                    pred_std = np.sqrt(mse) * np.ones_like(y_pred)
            else:
                pred_std = np.sqrt(mse) * np.ones_like(y_pred)

            intervals = np.zeros((len(x_eval), 2))
            intervals[:, 0] = y_pred - t_val * pred_std
            intervals[:, 1] = y_pred + t_val * pred_std

            return intervals

        except Exception as e:
            logger.warning(
                "Failed to compute prediction interval",
                error=str(e),
            )
            return None

    @classmethod
    def from_curve_fit_result(
        cls,
        curve_fit_result: Any,
        y_data: np.ndarray | None = None,
        model_fn: Callable | None = None,
        x_data: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Create OptimizationResult from NLSQ 0.6.6 CurveFitResult.

        This factory method preserves the native CurveFitResult for property
        delegation, enabling access to NLSQ 0.6.6's statistical methods like
        prediction_interval() without reimplementation.

        Parameters
        ----------
        curve_fit_result : CurveFitResult
            Result from nlsq.curve_fit() call.
        y_data : ndarray, optional
            Original dependent variable data (for complex data handling).
        model_fn : callable, optional
            Model function f(x, params) for prediction intervals.
        x_data : ndarray, optional
            Original independent variable data for prediction intervals.

        Returns
        -------
        result : OptimizationResult
            Result with native delegation to CurveFitResult properties.

        Examples
        --------
        >>> curve_fit_result = nlsq.curve_fit(model_fn, x, y, p0=p0)
        >>> result = OptimizationResult.from_curve_fit_result(
        ...     curve_fit_result, y_data=y, model_fn=model_fn, x_data=x
        ... )
        >>> print(result.r_squared)  # Delegates to native
        >>> pi = result.prediction_interval(x_new)  # Delegates to native
        """
        # Extract standard fields
        popt = np.asarray(curve_fit_result.popt, dtype=np.float64)
        pcov = (
            np.asarray(curve_fit_result.pcov, dtype=np.float64)
            if curve_fit_result.pcov is not None
            else None
        )
        success = getattr(curve_fit_result, "success", True)
        message = getattr(curve_fit_result, "message", "Converged")
        nfev = getattr(curve_fit_result, "nfev", 0)
        njev = getattr(curve_fit_result, "njev", 0)
        cost = getattr(curve_fit_result, "cost", None)

        # Get residuals from native result
        residuals = None
        if hasattr(curve_fit_result, "residuals"):
            residuals = np.asarray(curve_fit_result.residuals)

        # Get diagnostics if available
        diagnostics = getattr(curve_fit_result, "diagnostics", None)

        # Compute RSS from residuals or cost
        if residuals is not None:
            rss = float(np.sum(residuals**2))
        elif cost is not None:
            # scipy/NLSQ convention: cost = 0.5 * sum(residuals²)
            rss = float(2.0 * cost)
        else:
            rss = 0.0

        # Handle y_data
        y_data_np = np.asarray(y_data) if y_data is not None else None
        n_data = len(y_data_np) if y_data_np is not None else None

        # OPT-AIC-BIC-001: detect complex-split residuals (length 2N from
        # concatenated [real, imag]).  Only flag when y_data is absent and
        # residuals are real-typed with even length — callers who pass y_data
        # get the authoritative n_data from len(y_data) instead.
        _complex_split = False
        if n_data is None and residuals is not None:
            _complex_split = (
                not np.iscomplexobj(residuals)
                and len(residuals) % 2 == 0
                and len(residuals) > 2
                and y_data_np is not None
                and np.iscomplexobj(y_data_np)
            )
            # If y_data is complex but residuals are real, residuals were split
            if y_data_np is not None and np.iscomplexobj(y_data_np):
                n_data = len(y_data_np)
            # Otherwise use residual length as-is (no unsafe halving)

        result = cls(
            x=popt,
            fun=rss,
            jac=None,
            pcov=pcov,
            success=bool(success),
            message=str(message),
            nit=int(nfev),
            nfev=int(nfev),
            njev=int(njev),
            optimality=None,
            active_mask=None,
            cost=float(cost) if cost is not None else rss,
            grad=None,
            nlsq_result=None,
            residuals=residuals,
            y_data=y_data_np,
            n_data=n_data,
            diagnostics=diagnostics,
            _curve_fit_result=curve_fit_result,
            _model_fn=model_fn,
            _x_data=np.asarray(x_data) if x_data is not None else None,
            _is_complex_split=_complex_split,
        )

        return result

    @classmethod
    def from_nlsq(
        cls,
        nlsq_result: dict[str, Any],
        residuals: np.ndarray | None = None,
        y_data: np.ndarray | None = None,
    ) -> OptimizationResult:
        """Create OptimizationResult from NLSQ result dictionary.

        Args:
            nlsq_result: Result dictionary from nlsq.LeastSquares.least_squares
            residuals: Optional residual vector for covariance scaling and metrics
            y_data: Optional original y data for R² computation

        Returns:
            OptimizationResult instance with fields extracted from NLSQ result
        """
        # Extract common fields
        x = np.asarray(nlsq_result.get("x", []), dtype=np.float64)
        fun = float(nlsq_result.get("cost", nlsq_result.get("fun", 0.0)))
        success = bool(nlsq_result.get("success", False))
        message = str(nlsq_result.get("message", ""))
        nfev = int(nlsq_result.get("nfev", 0))
        njev = int(nlsq_result.get("njev", 0))

        # Extract NLSQ-specific fields
        jac = nlsq_result.get("jac")
        if jac is not None:
            jac = np.asarray(jac, dtype=np.float64)

        grad = nlsq_result.get("grad")
        if grad is not None:
            grad = np.asarray(grad, dtype=np.float64)

        optimality = nlsq_result.get("optimality")
        if optimality is not None:
            optimality = float(optimality)

        active_mask = nlsq_result.get("active_mask")
        if active_mask is not None:
            active_mask = np.asarray(active_mask)

        # Note: NLSQ uses 'nfev' for iterations in some contexts
        nit = int(nlsq_result.get("nit", nlsq_result.get("nfev", 0)))

        # Compute covariance from Jacobian
        pcov = None
        if jac is not None:
            pcov = compute_covariance_from_jacobian(jac, residuals)

        # Store residuals as numpy array for statistical properties
        residuals_np = None
        if residuals is not None:
            residuals_np = np.asarray(residuals, dtype=np.float64)
            # Handle scalar residuals (0-d arrays)
            if residuals_np.ndim == 0:
                residuals_np = residuals_np.reshape(1)

        # Store y_data for R² computation
        y_data_np = None
        n_data = None
        _complex_split = False
        if y_data is not None:
            y_data_np = np.asarray(y_data)
            n_data = len(y_data_np)
            # Detect complex-split residuals: y_data is complex but residuals are real
            if (
                np.iscomplexobj(y_data_np)
                and residuals_np is not None
                and not np.iscomplexobj(residuals_np)
            ):
                _complex_split = True

        return cls(
            x=x,
            fun=fun,
            jac=jac,
            pcov=pcov,
            success=success,
            message=message,
            nit=nit,
            nfev=nfev,
            njev=njev,
            optimality=optimality,
            active_mask=active_mask,
            cost=fun,  # NLSQ uses 'cost' terminology
            grad=grad,
            nlsq_result=nlsq_result,
            residuals=residuals_np,
            y_data=y_data_np,
            n_data=n_data,
            _is_complex_split=_complex_split,
        )


def nlsq_optimize(
    objective: Callable[[np.ndarray], float | np.ndarray],
    parameters: ParameterSet,
    method: str = "auto",
    use_jax: bool = True,
    max_iter: int = 1000,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-6,
    # NLSQ 0.6.6 parameters
    workflow: str = "auto",
    auto_bounds: bool = False,
    stability: str | bool = False,
    fallback: bool = False,
    compute_diagnostics: bool = False,
    **kwargs,
) -> OptimizationResult:
    """Optimize objective function using NLSQ (GPU-accelerated).

    This function provides GPU-accelerated nonlinear least squares optimization
    using the NLSQ package. It achieves 5-270x speedup over scipy through JAX
    JIT compilation and automatic differentiation.

    The objective function should accept parameter values as a 1D array and
    return a scalar value (minimization) or vector of residuals (least squares).

    Args:
        objective: Objective function to minimize. Takes parameter values as
            array and returns scalar or residual vector. Should use jax.numpy
            for operations to enable GPU acceleration and automatic differentiation.
        parameters: ParameterSet with initial values and bounds
        method: Optimization method. Options:

            - "auto": Automatically select based on bounds (default)
            - "trf": Trust Region Reflective (supports bounds)
            - "lm": Levenberg-Marquardt (no bounds)
            - "scipy": Use SciPy's least_squares directly (bypasses NLSQ).
              Use this for models that use Diffrax ODE solvers which are
              incompatible with NLSQ's forward-mode autodiff.

            NLSQ internally selects the best algorithm regardless of this parameter.
        use_jax: Whether to use JAX for gradient computation (default: True).
            Should always be True for GPU acceleration and float64 precision.
        max_iter: Maximum number of iterations (default: 1000)
        ftol: Function tolerance for convergence (default: 1e-6).
            Relaxed from 1e-8 due to NLSQ's mixed precision management.
        xtol: Parameter tolerance for convergence (default: 1e-6).
            Relaxed from 1e-8 due to NLSQ's mixed precision management.
        gtol: Gradient tolerance for convergence (default: 1e-6).
            Relaxed from 1e-8 due to NLSQ's mixed precision management.
        workflow: NLSQ 0.6.6 workflow selection (default: "auto"):

            - "auto": Memory-aware local optimization (default)
            - "auto_global": Global optimization with bounds exploration
            - "hpc": HPC mode with checkpointing support

        auto_bounds: Enable automatic parameter bounds inference (default: False).
            When True, reasonable bounds are inferred based on data characteristics.
        stability: Numerical stability checks (default: False):

            - 'auto': Check and automatically fix stability issues
            - 'check': Check and warn but don't fix
            - False: Skip stability checks

        fallback: Enable NLSQ's native fallback strategies (default: False).
            When True, tries alternative approaches if optimization fails.
            Note: RheoJAX also has its own SciPy fallback independent of this.
        compute_diagnostics: Compute model health diagnostics (default: False).
            When True, result.diagnostics includes identifiability analysis,
            gradient health, and other diagnostic information.
        **kwargs: Additional arguments passed to nlsq.LeastSquares.least_squares

    Returns:
        OptimizationResult with optimal parameters, convergence info, and
        optional diagnostics (when compute_diagnostics=True).

    Raises:
        ValueError: If objective is not callable or parameters is not ParameterSet

    Example:
        >>> from rheojax.core.parameters import ParameterSet
        >>> params = ParameterSet()
        >>> params.add("a", value=1.0, bounds=(0, 10))
        >>> params.add("b", value=1.0, bounds=(0, 10))
        >>>
        >>> def objective(values):
        ...     a, b = values
        ...     return (a - 5.0) ** 2 + (b - 3.0) ** 2
        >>>
        >>> result = nlsq_optimize(objective, params)
        >>> print(result.x)  # Should be close to [5.0, 3.0]
        >>>
        >>> # With NLSQ 0.6.6 features
        >>> result = nlsq_optimize(
        ...     objective, params,
        ...     workflow="auto_global",  # Global optimization
        ...     stability="auto",        # Auto-fix stability issues
        ...     compute_diagnostics=True # Get diagnostics
        ... )
        >>> print(result.diagnostics)

    Notes:
        - This function automatically handles float64 precision through NLSQ
        - JAX JIT compilation provides 5-270x speedup over scipy
        - Automatic differentiation eliminates need for manual Jacobian
        - Bounds are automatically extracted from ParameterSet
        - Parameters are updated in-place with optimal values
    """
    # Validate inputs
    if not callable(objective):
        raise ValueError("objective must be callable")

    if not isinstance(parameters, ParameterSet):
        raise ValueError("parameters must be ParameterSet")

    # Get initial values and bounds from ParameterSet
    x0, nlsq_bounds = _extract_bounds(parameters)
    original_values = x0.copy()

    # If method='scipy', use SciPy directly (bypasses NLSQ autodiff issues with Diffrax)
    if method == "scipy":
        logger.info(
            "Using SciPy least_squares directly (method='scipy')",
            n_params=len(x0),
        )
        result = _run_scipy_least_squares(
            objective, x0, nlsq_bounds, ftol, xtol, gtol, max_iter
        )
        parameters.set_values(result.x)
        return result

    logger.info(
        "Starting NLSQ optimization",
        n_params=len(x0),
        method=method,
        max_iter=max_iter,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        workflow=workflow,
        auto_bounds=auto_bounds,
        stability=stability,
        fallback=fallback,
        compute_diagnostics=compute_diagnostics,
    )
    logger.debug(
        "Initial parameter values",
        x0=x0.tolist() if hasattr(x0, "tolist") else list(x0),
        lower_bounds=nlsq_bounds[0].tolist(),
        upper_bounds=nlsq_bounds[1].tolist(),
    )

    # NLSQ expects a residual function that returns a vector of residuals
    # The objective function from create_least_squares_objective() now returns
    # a proper residual vector, so we use it directly
    # NLSQ will minimize sum(residuals²) internally

    # Set up NLSQ optimization parameters
    nlsq_kwargs = {
        "fun": objective,  # Now a residual function returning vector
        "x0": x0,
        "bounds": nlsq_bounds,
        "method": "trf",  # Trust Region Reflective (supports bounds)
        "ftol": ftol,
        "xtol": xtol,
        "gtol": gtol,
        "max_nfev": max_iter * 10,  # NLSQ uses max_nfev for iteration limit
        "verbose": 0,
    }

    # Add NLSQ 0.6.6 parameters (feature detection for backward compatibility)
    # Note: LeastSquares.least_squares may not support all curve_fit params
    if workflow != "auto":
        nlsq_kwargs["workflow"] = workflow
    if auto_bounds:
        nlsq_kwargs["auto_bounds"] = auto_bounds
    if stability:
        nlsq_kwargs["stability"] = stability
    if fallback:
        nlsq_kwargs["fallback"] = fallback
    if compute_diagnostics:
        nlsq_kwargs["compute_diagnostics"] = compute_diagnostics

    # Merge with user-provided kwargs, filtering out rheojax-specific ones
    # that are not valid NLSQ optimizer parameters (prevents TypeError in NLSQ)
    clean_kwargs = {
        k: v for k, v in kwargs.items() if k not in _RHEOJAX_RESERVED_KWARGS
    }
    nlsq_kwargs.update(clean_kwargs)

    def _scipy_fallback(initial_guess: np.ndarray) -> OptimizationResult:
        """Fallback chain when NLSQ fails.

        For workflow="auto_global": SciPy TRF first, then differential_evolution
        as the final global-search fallback.  For all other workflows: SciPy TRF only.
        """
        logger.info("Using SciPy least_squares fallback")
        scipy_result = _run_scipy_least_squares(
            objective, initial_guess, nlsq_bounds, ftol, xtol, gtol, max_iter
        )
        if workflow == "auto_global" and not scipy_result.success:
            logger.info(
                "SciPy TRF fallback did not converge; trying differential_evolution"
            )
            de_result = _run_differential_evolution(
                objective, scipy_result.x, nlsq_bounds, max_iter
            )
            de_result.message = (
                f"[DE fallback] {de_result.message} "
                f"(SciPy TRF: {scipy_result.message})"
            )
            return de_result
        return scipy_result

    # Create NLSQ optimizer instance and run optimization
    try:
        logger.debug("Creating NLSQ optimizer instance")
        optimizer = nlsq.LeastSquares()
        nlsq_result = optimizer.least_squares(**nlsq_kwargs)
        logger.debug(
            "NLSQ optimization completed",
            success=nlsq_result.get("success", False),
            nfev=nlsq_result.get("nfev", 0),
            cost=float(nlsq_result.get("cost", 0.0)),
        )
    except (RuntimeError, TypeError, ValueError, FloatingPointError, OverflowError) as e:
        logger.warning(
            "NLSQ optimization raised exception, falling back to SciPy",
            error=str(e),
            exc_info=True,
        )
        result = _scipy_fallback(x0)
        result.message = f"[SciPy fallback] {result.message} (NLSQ failed: {e})"
        # Compute residuals for validation (OPT-001)
        _residuals_fb_raw = np.asarray(objective(result.x))
        if np.iscomplexobj(_residuals_fb_raw):
            residuals_fb = np.concatenate(
                [np.real(_residuals_fb_raw), np.imag(_residuals_fb_raw)]
            ).astype(np.float64)
        else:
            residuals_fb = _residuals_fb_raw.astype(np.float64)
        result.fun = float(np.sum(residuals_fb**2))
        # P1-6: Propagate normalization weights for correct R²/AIC/BIC
        _nw = getattr(objective, "_normalization_weights", None)
        if _nw is not None:
            result._normalization_weights = _nw
        _validate_optimization_result(result, residuals_fb)
        # Write back optimal params so model state reflects the fit
        parameters.set_values(result.x)
        return result

    # Compute residuals at optimal point for covariance scaling.
    # P2-Fit-2: Reuse residuals from NLSQ result when available (avoids an
    # extra objective evaluation that can be expensive for ODE models).
    x_opt = np.asarray(nlsq_result.get("x", x0), dtype=np.float64)
    # NLSQ v0.6.10: 'fun' is the residual vector (array); 'fun_vector' and
    # 'residuals' keys were removed.  Prefer the cached residual from the result
    # dict to avoid an expensive extra objective evaluation (especially costly
    # for ODE-based models).
    _cached_fun = nlsq_result.get("fun_vector", nlsq_result.get("residuals"))
    if _cached_fun is None:
        _fun_field = nlsq_result.get("fun")
        if _fun_field is not None and np.asarray(_fun_field).ndim >= 1:
            _cached_fun = _fun_field
    if _cached_fun is not None:
        residuals_raw = np.asarray(_cached_fun)
    else:
        residuals_raw = np.asarray(objective(x_opt))
    if np.iscomplexobj(residuals_raw):
        residuals_np = np.concatenate(
            [np.real(residuals_raw), np.imag(residuals_raw)]
        ).astype(np.float64)
    else:
        residuals_np = residuals_raw.astype(np.float64)

    # Convert NLSQ result to OptimizationResult (with residuals for covariance)
    result = OptimizationResult.from_nlsq(nlsq_result, residuals=residuals_np)

    # P1-6: Propagate normalization weights from the objective closure so that
    # R²/AIC/BIC can un-normalize residuals for correct statistics.
    _nw = getattr(objective, "_normalization_weights", None)
    if _nw is not None:
        result._normalization_weights = _nw

    # Store diagnostics if available (NLSQ 0.6.6+)
    if hasattr(nlsq_result, "diagnostics") or "diagnostics" in nlsq_result:
        result.diagnostics = nlsq_result.get(
            "diagnostics", getattr(nlsq_result, "diagnostics", None)
        )

    # P2-Fit-9: Trigger SciPy fallback on ANY NLSQ failure when fallback is
    # enabled, not just the "inner optimization loop exceeded" message.
    if not result.success and fallback:
        logger.warning(
            "NLSQ did not converge; retrying with SciPy least_squares for stability.",
            nlsq_message=result.message,
        )
        # OPT-003: warm-start from NLSQ's best result, not stale x0
        fallback_result = _scipy_fallback(x_opt)
        fallback_result.message = (
            f"[SciPy fallback] {fallback_result.message} (NLSQ inner loop limit)"
        )
        # OPT-004: validate the fallback result
        _residuals_fb_raw = np.asarray(objective(fallback_result.x))
        if np.iscomplexobj(_residuals_fb_raw):
            residuals_fb = np.concatenate(
                [np.real(_residuals_fb_raw), np.imag(_residuals_fb_raw)]
            ).astype(np.float64)
        else:
            residuals_fb = _residuals_fb_raw.astype(np.float64)
        fallback_result.fun = float(np.sum(residuals_fb**2))
        _validate_optimization_result(fallback_result, residuals_fb)
        parameters.set_values(fallback_result.x)
        return fallback_result

    # Ensure x is float64
    result.x = np.asarray(result.x, dtype=np.float64)

    # Compute RSS = sum(residuals²)
    # OPT-008: Use np.sum instead of jnp.sum on numpy array to avoid host-device round-trip
    result.fun = float(np.sum(residuals_np**2))

    # Guard against false "success" with astronomically large residuals
    try:
        _validate_optimization_result(result, residuals_np)
    except RuntimeError:
        parameters.set_values(original_values)
        raise

    # Update ParameterSet with optimal values
    parameters.set_values(result.x)

    logger.info(
        "Optimization completed successfully",
        success=result.success,
        rss=float(result.fun),
        nfev=result.nfev,
        nit=result.nit,
        r_squared=result.r_squared,
    )
    logger.debug(
        "Final parameter values",
        x_opt=result.x.tolist(),
        message=result.message,
    )

    return result


def nlsq_multistart_optimize(
    objective: Callable[[np.ndarray], float | np.ndarray],
    parameters: ParameterSet,
    n_starts: int = 5,
    perturb_factor: float = 0.3,
    method: str = "auto",
    use_jax: bool = True,
    max_iter: int = 1000,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-6,
    verbose: bool = False,
    parallel: bool = True,
    n_workers: int | None = None,
    y_data: np.ndarray | None = None,
    **kwargs,
) -> OptimizationResult:
    """Multi-start optimization to escape local minima.

    For complex objective functions (e.g., mastercurves with 10+ decades),
    single optimization runs may converge to poor local minima even from
    good initial guesses. This function performs multiple optimization runs
    from different starting points and returns the best result.

    Strategy:
        1. First attempt: Use current parameter values (from smart initialization)
        2. Additional attempts: Random perturbations around initial values (parallel)
        3. Return result with lowest final cost (best fit)

    Performance: With parallel=True (default), achieves 2-4x speedup for 5-10
    starts by running optimizations concurrently. JAX releases the GIL during
    computation, enabling effective thread-based parallelism.

    Args:
        objective: Objective function to minimize
        parameters: ParameterSet with initial values and bounds
        n_starts: Number of random starts (default: 5)
        perturb_factor: Perturbation factor for random starts (default: 0.3)
            Parameters are perturbed by ± perturb_factor * (value or range)
        method: Optimization method (default: "auto")
        use_jax: Whether to use JAX (default: True)
        max_iter: Max iterations per start (default: 1000)
        ftol: Function tolerance (default: 1e-6)
        xtol: Parameter tolerance (default: 1e-6)
        gtol: Gradient tolerance (default: 1e-6)
        verbose: Print progress messages (default: False)
        parallel: Run additional starts in parallel (default: True)
        n_workers: Number of parallel workers (default: min(n_starts-1, 4))
        **kwargs: Additional arguments for nlsq_optimize

    Returns:
        OptimizationResult with best parameters from all starts

    Example:
        >>> # For mastercurve data (12+ decades)
        >>> result = nlsq_multistart_optimize(
        ...     objective, parameters, n_starts=5, verbose=True
        ... )
        >>> print(f"Best cost: {result.fun:.3e}")
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Store original parameter values
    original_values = parameters.get_values()
    bounds_list = parameters.get_bounds()
    param_names = list(parameters.keys())

    logger.info(
        "Starting multi-start optimization",
        n_starts=n_starts,
        perturb_factor=perturb_factor,
        n_params=len(original_values),
        parallel=parallel,
    )

    # First attempt: Use smart initialization values (sequential)
    if verbose:
        logger.info("Multi-start optimization: Attempt 1 (smart initialization)")

    best_result = nlsq_optimize(
        objective,
        parameters,
        method=method,
        use_jax=use_jax,
        max_iter=max_iter,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
        **kwargs,
    )
    best_cost = best_result.fun

    logger.debug(
        "First attempt completed",
        cost=float(best_cost),
        success=best_result.success,
    )
    if verbose:
        logger.info(f"  Cost: {best_cost:.3e}, Success: {best_result.success}")

    # If only 1 start requested, return early
    if n_starts <= 1:
        return best_result

    # Generate all perturbed starting points
    def generate_perturbed_values(rng: np.random.Generator) -> list[float]:
        """Generate perturbed initial values using the provided RNG."""
        perturbed = []
        for orig_val, bounds in zip(original_values, bounds_list, strict=True):
            if bounds is None or (bounds[0] is None and bounds[1] is None):
                perturbation = rng.uniform(-perturb_factor, perturb_factor)
                if abs(orig_val) < 1e-30:
                    # Additive perturbation for zero-valued parameters
                    new_val = perturbation
                else:
                    new_val = orig_val * (1.0 + perturbation)
            else:
                lower = bounds[0] if bounds[0] is not None else orig_val - abs(orig_val)
                upper = bounds[1] if bounds[1] is not None else orig_val + abs(orig_val)
                range_size = upper - lower
                perturbation = rng.uniform(
                    -perturb_factor * range_size, perturb_factor * range_size
                )
                new_val = np.clip(orig_val + perturbation, lower, upper)
            perturbed.append(new_val)
        return perturbed

    def run_single_optimization(
        start_idx: int, initial_values: list[float]
    ) -> tuple[int, OptimizationResult | None]:
        """Run a single optimization from given starting point."""
        # Create a fresh ParameterSet copy for thread-safe operation.
        # Use initial_values[i] as the starting value for each parameter instead
        # of a hardcoded 0.0 — a zero initializer triggers a spurious clamping
        # RuntimeWarning for parameters whose lower bound is > 0 (e.g. (1e-3, 1e6)).
        # The perturbed values are always within bounds (generated by np.clip) so
        # no clamping occurs and the separate set_values() call is unnecessary.
        params_copy = ParameterSet()
        for name, bounds, init_val in zip(
            param_names, bounds_list, initial_values, strict=True
        ):
            # Handle None values in bounds
            bounds_tuple: tuple[float, float] | None = None
            if bounds is not None:
                if bounds[0] is not None and bounds[1] is not None:
                    bounds_tuple = (float(bounds[0]), float(bounds[1]))
            params_copy.add(name=name, value=float(init_val), bounds=bounds_tuple)

        try:
            result = nlsq_optimize(
                objective,
                params_copy,
                method=method,
                use_jax=use_jax,
                max_iter=max_iter,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                **kwargs,
            )
            return start_idx, result
        except Exception as e:
            logger.warning(
                "Multi-start attempt failed",
                attempt=start_idx + 1,
                error=str(e),
            )
            return start_idx, None

    # Prepare all starting points
    root_rng = np.random.default_rng(seed=42)
    all_starts = [
        generate_perturbed_values(np.random.default_rng(root_rng.integers(2**31)))
        for _ in range(1, n_starts)
    ]

    if parallel and n_starts > 2:
        # Parallel execution for additional starts
        if n_workers is None:
            n_workers = min(n_starts - 1, min(4, os.cpu_count() or 1))

        logger.debug(
            "Running parallel multi-start",
            n_workers=n_workers,
            n_additional_starts=n_starts - 1,
        )

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(run_single_optimization, i, starts): i
                for i, starts in enumerate(all_starts, start=1)
            }

            for future in as_completed(futures):
                start_idx, result = future.result()
                if result is not None:
                    logger.debug(
                        "Multi-start attempt completed",
                        attempt=start_idx + 1,
                        cost=float(result.fun),
                        success=result.success,
                    )
                    if verbose:
                        logger.info(
                            f"  Attempt {start_idx + 1}: Cost: {result.fun:.3e}, "
                            f"Success: {result.success}"
                        )

                    if result.fun < best_cost:
                        best_result = result
                        best_cost = result.fun
                        logger.debug(
                            "New best result found",
                            attempt=start_idx + 1,
                            best_cost=float(best_cost),
                        )
                        if verbose:
                            logger.info(f"  -> New best! Cost: {best_cost:.3e}")
                else:
                    if verbose:
                        logger.warning(f"  Attempt {start_idx + 1} failed")
    else:
        # Sequential execution (original behavior)
        for i, perturbed_values in enumerate(all_starts, start=1):
            logger.debug("Starting multi-start attempt", attempt=i + 1, total=n_starts)
            if verbose:
                logger.info(
                    f"Multi-start optimization: Attempt {i + 1} (random perturbation)"
                )

            start_idx, result = run_single_optimization(i, perturbed_values)

            if result is not None:
                logger.debug(
                    "Multi-start attempt completed",
                    attempt=i + 1,
                    cost=float(result.fun),
                    success=result.success,
                )
                if verbose:
                    logger.info(f"  Cost: {result.fun:.3e}, Success: {result.success}")

                if result.fun < best_cost:
                    best_result = result
                    best_cost = result.fun
                    logger.debug(
                        "New best result found",
                        attempt=i + 1,
                        best_cost=float(best_cost),
                    )
                    if verbose:
                        logger.info(f"  -> New best! Cost: {best_cost:.3e}")
            else:
                if verbose:
                    logger.warning(f"  Attempt {i + 1} failed")

    # Restore best parameters
    parameters.set_values(best_result.x)

    if y_data is not None and best_result.y_data is None:
        best_result.y_data = np.asarray(y_data)
    # R8-OPT-002: also set n_data for AIC/BIC/adj_R² computation
    if y_data is not None and getattr(best_result, "n_data", None) is None:
        best_result.n_data = len(np.asarray(y_data))

    logger.info(
        "Multi-start optimization completed",
        best_cost=float(best_cost),
        n_starts=n_starts,
        final_success=best_result.success,
        parallel=parallel,
    )
    if verbose:
        logger.info(
            f"\nMulti-start completed: Best cost = {best_cost:.3e} "
            f"({n_starts} starts, parallel={parallel})"
        )

    return best_result


def nlsq_optimize_global(
    objective: Callable[[np.ndarray], float | np.ndarray],
    parameters: ParameterSet,
    **kwargs,
) -> OptimizationResult:
    """Global optimization using NLSQ 0.6.6 workflow='auto_global'.

    Convenience function for global optimization that explores parameter space
    more thoroughly using the NLSQ 0.6.6 global optimization workflow.

    Args:
        objective: Objective function to minimize. Takes parameter values as
            array and returns scalar or residual vector.
        parameters: ParameterSet with initial values and bounds
        **kwargs: Additional arguments passed to nlsq_optimize

    Returns:
        OptimizationResult with optimal parameters from global search

    Example:
        >>> from rheojax.core.parameters import ParameterSet
        >>> params = ParameterSet()
        >>> params.add("a", value=1.0, bounds=(0, 10))
        >>> params.add("b", value=1.0, bounds=(0, 10))
        >>>
        >>> result = nlsq_optimize_global(objective, params)
        >>> print(f"Global optimum: {result.x}")

    Notes:
        - Uses workflow='auto_global' for bounds-aware global exploration
        - More thorough but slower than standard local optimization
        - Useful for multi-modal objective functions
    """
    return nlsq_optimize(
        objective,
        parameters,
        workflow="auto_global",
        **kwargs,
    )


def nlsq_curve_fit(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    y_data: np.ndarray,
    parameters: ParameterSet,
    auto_bounds: bool = False,
    stability: str | bool = False,
    fallback: bool = False,
    compute_diagnostics: bool = False,
    multistart: bool = False,
    n_starts: int = 10,
    workflow: str = "auto",
    **kwargs,
) -> OptimizationResult:
    """Curve fitting using NLSQ 0.6.6 curve_fit() API with advanced features.

    This function provides access to NLSQ 0.6.6's enhanced curve_fit() features
    including auto-bounds, stability checks, fallback strategies, model
    diagnostics, and workflow selection. It returns an OptimizationResult with
    CurveFitResult-compatible statistical properties (r_squared, rmse, aic, bic,
    prediction_interval, etc.).

    Args:
        model_fn: Model function f(x, params_array) -> y_pred.
            Takes x_data and parameter array, returns predictions.
        x_data: Independent variable data
        y_data: Dependent variable data (observations)
        parameters: ParameterSet with initial values and bounds
        auto_bounds: Enable automatic parameter bounds inference (default: False).
            When True, reasonable bounds are inferred based on data characteristics.
        stability: Numerical stability checks (default: False).
            - 'auto': Check and automatically fix stability issues
            - 'check': Check and warn but don't fix
            - False: Skip stability checks
        fallback: Enable automatic fallback strategies (default: False).
            When True, tries alternative approaches if optimization fails.
        compute_diagnostics: Compute model health diagnostics (default: False).
            When True, result includes identifiability analysis, gradient health, etc.
        multistart: Enable multi-start optimization (default: False).
            When True, explores multiple starting points to find global optimum.
        n_starts: Number of starting points for multi-start (default: 10).
        workflow: NLSQ 0.6.6 workflow selection (default: "auto"):

            - "auto": Memory-aware local optimization (default)
            - "auto_global": Global optimization with bounds exploration
            - "hpc": HPC mode with checkpointing support

        **kwargs: Additional arguments passed to nlsq.curve_fit()

    Returns:
        OptimizationResult with CurveFitResult-compatible statistical properties:
        - r_squared, adj_r_squared, rmse, mae, aic, bic
        - confidence_intervals(alpha) method
        - prediction_interval(x_new, alpha) method (NLSQ 0.6.6 native)
        - get_parameter_uncertainties() method

    Example:
        >>> from rheojax.core.parameters import ParameterSet
        >>> from rheojax.utils.optimization import nlsq_curve_fit
        >>>
        >>> def model(x, params):
        ...     a, b = params
        ...     return a * np.exp(-b * x)
        >>>
        >>> params = ParameterSet()
        >>> params.add("a", value=1.0, bounds=(0, 10))
        >>> params.add("b", value=0.5, bounds=(0, 5))
        >>>
        >>> result = nlsq_curve_fit(
        ...     model, x_data, y_data, params,
        ...     auto_bounds=True,
        ...     stability='auto',
        ...     fallback=True,
        ...     compute_diagnostics=True,
        ... )
        >>> print(f"R² = {result.r_squared:.4f}")
        >>> print(f"RMSE = {result.rmse:.4f}")
        >>> ci = result.confidence_intervals(alpha=0.95)
        >>>
        >>> # Prediction intervals (NLSQ 0.6.6)
        >>> pi = result.prediction_interval(x_new, alpha=0.95)
        >>> print(f"95% PI: [{pi[0, 0]:.3f}, {pi[0, 1]:.3f}]")

    Notes:
        - This function uses nlsq.curve_fit() directly (not LeastSquares.least_squares())
        - The model function signature is ``f(x, params_array)`` not ``f(x, *params)``
        - Results delegate to native CurveFitResult for prediction_interval() calls
        - Results include all CurveFitResult properties for model comparison
    """
    import nlsq as nlsq_module

    logger.info(
        "Starting curve fit",
        n_params=len(parameters),
        n_data=len(x_data),
        auto_bounds=auto_bounds,
        stability=stability,
        multistart=multistart,
        workflow=workflow,
    )

    # Extract p0 and bounds from ParameterSet
    p0, (lower, upper) = _extract_bounds(parameters)

    # Convert x_data and y_data to numpy arrays
    x_data_np = np.asarray(x_data, dtype=np.float64)
    y_data_np = np.asarray(y_data)  # Preserve complex type if present

    # Create wrapper function f(x, *params) -> y for nlsq.curve_fit
    # NLSQ curve_fit expects f(x, p0, p1, ...) not f(x, params_array)
    def f_wrapper(x, *params_tuple):
        params_array = jnp.asarray(params_tuple, dtype=jnp.float64)
        return model_fn(x, params_array)

    # Build kwargs for nlsq.curve_fit
    curve_fit_kwargs = {
        "p0": p0,
        "bounds": (lower, upper),
        "auto_bounds": auto_bounds,
        "stability": stability,
        "fallback": fallback,
        "compute_diagnostics": compute_diagnostics,
        "multistart": multistart,
        "n_starts": n_starts,
    }
    # Add workflow parameter (NLSQ 0.6.6)
    if workflow != "auto":
        curve_fit_kwargs["workflow"] = workflow
    # OPT-002: Filter RheoJAX-specific kwargs before forwarding to nlsq.curve_fit
    clean_kwargs = {
        k: v for k, v in kwargs.items() if k not in _RHEOJAX_RESERVED_KWARGS
    }
    curve_fit_kwargs.update(clean_kwargs)

    try:
        # Call nlsq.curve_fit() - returns CurveFitResult (tuple unpacking compatible)
        curve_fit_result = nlsq_module.curve_fit(
            f_wrapper, x_data_np, y_data_np, **curve_fit_kwargs
        )

        # CurveFitResult supports both tuple unpacking and attribute access
        # We need to check if it's a tuple (popt, pcov) or CurveFitResult object
        is_curve_fit_result = not isinstance(curve_fit_result, tuple)

        if isinstance(curve_fit_result, tuple):
            popt, pcov = curve_fit_result
        else:
            # It's a CurveFitResult object
            popt = np.asarray(curve_fit_result.popt)
            pcov = curve_fit_result.pcov

        # Compute residuals and y_pred at optimal point (for complex data handling)
        y_pred = model_fn(x_data_np, popt)
        y_pred_np = np.asarray(y_pred)

        # Compute residuals (handle complex data)
        if np.iscomplexobj(y_data_np):
            if np.iscomplexobj(y_pred_np):
                # Both complex: residuals for real and imaginary parts
                residuals_real = np.real(y_data_np) - np.real(y_pred_np)
                residuals_imag = np.imag(y_data_np) - np.imag(y_pred_np)
                residuals = np.concatenate([residuals_real, residuals_imag])
            else:
                # Complex data, real pred: use magnitude
                residuals = np.abs(y_data_np) - y_pred_np
        else:
            if np.iscomplexobj(y_pred_np):
                # Real data, complex pred: use magnitude of pred
                residuals = y_data_np - np.abs(y_pred_np)
            else:
                # Both real
                residuals = y_data_np - y_pred_np

        # Create OptimizationResult - use factory for native CurveFitResult delegation
        if is_curve_fit_result and hasattr(curve_fit_result, "prediction_interval"):
            # NLSQ 0.6.6+ CurveFitResult with native property delegation
            result = OptimizationResult.from_curve_fit_result(
                curve_fit_result,
                y_data=y_data_np,
                model_fn=model_fn,
                x_data=x_data_np,
            )
            # Override residuals for complex data handling
            result.residuals = residuals
            result.fun = float(np.sum(residuals**2))
            result.cost = result.fun
        else:
            # Legacy tuple result or no native delegation
            success = (
                True
                if isinstance(curve_fit_result, tuple)
                else getattr(curve_fit_result, "success", True)
            )
            message = (
                "Optimization converged successfully"
                if isinstance(curve_fit_result, tuple)
                else getattr(curve_fit_result, "message", "Converged")
            )
            nfev = (
                0
                if isinstance(curve_fit_result, tuple)
                else getattr(curve_fit_result, "nfev", 0)
            )
            njev = (
                0
                if isinstance(curve_fit_result, tuple)
                else getattr(curve_fit_result, "njev", 0)
            )
            diagnostics = (
                None
                if isinstance(curve_fit_result, tuple)
                else getattr(curve_fit_result, "diagnostics", None)
            )

            result = OptimizationResult(
                x=np.asarray(popt, dtype=np.float64),
                fun=float(np.sum(residuals**2)),
                jac=None,  # curve_fit doesn't return Jacobian directly
                pcov=np.asarray(pcov, dtype=np.float64) if pcov is not None else None,
                success=bool(success),
                message=str(message),
                nit=int(nfev),
                nfev=int(nfev),
                njev=int(njev),
                optimality=None,
                active_mask=None,
                cost=float(np.sum(residuals**2)),
                grad=None,
                nlsq_result=None,
                residuals=residuals,
                y_data=y_data_np,
                # UTIL-011: Use len(y_data_np) not len(residuals) — for complex data,
                # residuals is 2N (real + imag parts concatenated) while actual
                # observations are N. AIC/BIC must use the observation count N.
                n_data=len(y_data_np),
                diagnostics=diagnostics,
            )

        # Update ParameterSet with optimal values
        parameters.set_values(result.x)

        logger.info(
            "Curve fit completed successfully",
            r_squared=result.r_squared,
            rmse=result.rmse,
            success=result.success,
        )
        logger.debug(
            "Curve fit results",
            popt=result.x.tolist(),
            rss=float(result.fun),
            aic=result.aic,
            bic=result.bic,
        )

        return result

    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as e:
        logger.warning(
            "nlsq.curve_fit() failed, falling back to nlsq_optimize",
            error=str(e),
            exc_info=True,
        )

        # Fallback to nlsq_optimize with residual-based objective
        objective = create_least_squares_objective(model_fn, x_data_np, y_data_np)

        if multistart:
            result = nlsq_multistart_optimize(
                objective, parameters, n_starts=n_starts, **kwargs
            )
        else:
            result = nlsq_optimize(objective, parameters, **kwargs)

        # Annotate that this result came from a curve_fit→nlsq_optimize fallback
        result.message = (
            f"[curve_fit→nlsq_optimize fallback] {getattr(result, 'message', '')}"
        )
        # Preserve y_data for R² calculation (not set by nlsq_optimize fallback)
        result.y_data = y_data_np
        result._model_fn = model_fn
        result._x_data = x_data_np
        return result


def optimize_with_bounds(
    objective: Callable[[np.ndarray], float | np.ndarray],
    x0: np.ndarray,
    bounds: list[tuple[float | None, float | None]],
    use_jax: bool = True,
    **kwargs,
) -> OptimizationResult:
    """Optimize objective function with parameter bounds.

    Lower-level optimization function that works with arrays instead of
    ParameterSet. Useful for custom optimization workflows.

    Args:
        objective: Objective function to minimize
        x0: Initial parameter values
        bounds: List of (min, max) tuples for each parameter
        use_jax: Whether to use JAX for gradients (default: True)
        **kwargs: Additional arguments passed to nlsq_optimize

    Returns:
        OptimizationResult with optimal parameters

    Example:
        >>> def objective(x):
        ...     return x[0]**2 + x[1]**2
        >>> result = optimize_with_bounds(
        ...     objective,
        ...     x0=np.array([1.0, 1.0]),
        ...     bounds=[(0, 5), (0, 5)]
        ... )
    """
    # Create temporary ParameterSet for interface consistency

    params = ParameterSet()
    for i, (val, bound) in enumerate(zip(x0, bounds, strict=True)):
        # Handle None values in bounds
        bounds_tuple: tuple[float, float] | None = None
        if bound is not None:
            if bound[0] is not None and bound[1] is not None:
                bounds_tuple = (float(bound[0]), float(bound[1]))
        params.add(name=f"p{i}", value=val, bounds=bounds_tuple)

    # Use main optimization function
    return nlsq_optimize(objective, params, use_jax=use_jax, **kwargs)


def fit_with_nlsq(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    **kwargs,
) -> OptimizationResult:
    """Fit using nonlinear least squares with residual function.

    Convenience function for fitting models using a residual function
    that takes parameter array and returns residual vector.

    Args:
        residual_fn: Function that takes parameter array and returns residuals
        x0: Initial parameter values as 1D array
        bounds: Tuple of (lower, upper) bound arrays, or None for unbounded
        **kwargs: Additional arguments passed to optimize_with_bounds

    Returns:
        OptimizationResult with optimal parameters in .x attribute
    """
    # Convert bounds format: (lower_array, upper_array) -> list of tuples
    if bounds is not None:
        lower, upper = bounds
        bounds_list: list[tuple[float | None, float | None]] = [
            (float(lo), float(hi)) for lo, hi in zip(lower, upper, strict=True)
        ]
    else:
        bounds_list = [(None, None)] * len(x0)

    return optimize_with_bounds(residual_fn, x0, bounds_list, **kwargs)


def residual_sum_of_squares(
    y_true: ArrayLike, y_pred: ArrayLike, normalize: bool = True
) -> float:
    """Compute residual sum of squares (RSS).

    Handles both real and complex data correctly. For complex data (e.g.,
    oscillatory shear with G' + iG"), computes RSS for both real and imaginary
    parts separately and returns the sum.

    Args:
        y_true: True values (real or complex)
        y_pred: Predicted values (real or complex)
        normalize: Whether to normalize by y_true (relative error)

    Returns:
        RSS value (scalar, maintains float64 precision)

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> rss = residual_sum_of_squares(y_true, y_pred)
    """
    # Use JAX operations if inputs are JAX arrays for gradient support
    # OPT-013: Use hasattr check instead of isinstance(x, jnp.ndarray)
    if hasattr(y_pred, "devices") or hasattr(y_true, "devices"):
        # Convert to JAX arrays (preserving complex type)
        y_true_jax = jnp.asarray(y_true)
        y_pred_jax = jnp.asarray(y_pred)

        # Check if data is complex
        y_true_is_complex = jnp.iscomplexobj(y_true_jax)
        y_pred_is_complex = jnp.iscomplexobj(y_pred_jax)

        if y_pred_is_complex:
            if y_true_is_complex:
                # Both complex: fit real and imaginary parts separately
                residuals_real = jnp.real(y_pred_jax) - jnp.real(y_true_jax)
                residuals_imag = jnp.imag(y_pred_jax) - jnp.imag(y_true_jax)

                if normalize:
                    residuals_real = residuals_real / jnp.maximum(
                        jnp.abs(jnp.real(y_true_jax)), 1e-10
                    )
                    residuals_imag = residuals_imag / jnp.maximum(
                        jnp.abs(jnp.imag(y_true_jax)), 1e-10
                    )

                rss = jnp.sum(residuals_real**2) + jnp.sum(residuals_imag**2)
            else:
                # Complex predictions, real data: fit to magnitude
                y_pred_magnitude = jnp.abs(y_pred_jax)
                y_true_jax = jnp.asarray(y_true_jax, dtype=jnp.float64)
                residuals = y_pred_magnitude - y_true_jax

                if normalize:
                    residuals = residuals / jnp.maximum(jnp.abs(y_true_jax), 1e-10)

                rss = jnp.sum(residuals**2)
        else:
            # Real predictions
            if y_true_is_complex:
                # Real predictions, complex data: fit to magnitude of data
                y_true_magnitude = jnp.abs(y_true_jax)
                y_pred_jax = jnp.asarray(y_pred_jax, dtype=jnp.float64)
                residuals = y_pred_jax - y_true_magnitude

                if normalize:
                    residuals = residuals / jnp.maximum(y_true_magnitude, 1e-10)

                rss = jnp.sum(residuals**2)
            else:
                # Both real: standard case
                y_true_jax = jnp.asarray(y_true_jax, dtype=jnp.float64)
                y_pred_jax = jnp.asarray(y_pred_jax, dtype=jnp.float64)
                residuals = y_pred_jax - y_true_jax

                if normalize:
                    residuals = residuals / jnp.maximum(jnp.abs(y_true_jax), 1e-10)

                rss = jnp.sum(residuals**2)

        # Return scalar JAX array, don't convert to Python float (breaks gradients)
        return rss
    else:
        # NumPy path
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        # Check if data is complex
        y_true_is_complex = np.iscomplexobj(y_true_np)
        y_pred_is_complex = np.iscomplexobj(y_pred_np)

        if y_pred_is_complex:
            if y_true_is_complex:
                # Both complex: fit real and imaginary parts separately
                residuals_real = np.real(y_pred_np) - np.real(y_true_np)
                residuals_imag = np.imag(y_pred_np) - np.imag(y_true_np)

                if normalize:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        residuals_real = residuals_real / np.maximum(
                            np.abs(np.real(y_true_np)), 1e-10
                        )
                        residuals_imag = residuals_imag / np.maximum(
                            np.abs(np.imag(y_true_np)), 1e-10
                        )

                rss = float(np.sum(residuals_real**2) + np.sum(residuals_imag**2))
            else:
                # Complex predictions, real data: fit to magnitude
                y_pred_magnitude = np.abs(y_pred_np)
                residuals = y_pred_magnitude - y_true_np

                if normalize:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        residuals = residuals / np.maximum(np.abs(y_true_np), 1e-10)

                rss = float(np.sum(residuals**2))
        else:
            # Real predictions
            if y_true_is_complex:
                # Real predictions, complex data: fit to magnitude of data
                y_true_magnitude = np.abs(y_true_np)
                residuals = y_pred_np - y_true_magnitude

                if normalize:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        residuals = residuals / np.maximum(y_true_magnitude, 1e-10)

                rss = float(np.sum(residuals**2))
            else:
                # Both real: standard case
                y_true_np = np.asarray(y_true_np, dtype=np.float64)
                y_pred_np = np.asarray(y_pred_np, dtype=np.float64)
                residuals = y_pred_np - y_true_np

                if normalize:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        residuals = residuals / np.maximum(np.abs(y_true_np), 1e-10)

                rss = float(np.sum(residuals**2))

        return rss


class ResidualFunction:
    """Callable wrapper for residual functions that carries normalization metadata.

    This replaces the fragile pattern of attaching ``_normalization_weights`` as
    a function attribute (which breaks if the function is wrapped by decorators,
    ``functools.wraps``, ``jax.jit``, etc.).  The class is fully transparent to
    callers — it behaves like a plain function but safely exposes the weights.
    """

    __slots__ = ("_fn", "_normalization_weights")

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        normalization_weights: np.ndarray | None = None,
    ) -> None:
        self._fn = fn
        self._normalization_weights = normalization_weights

    def __call__(self, params: np.ndarray) -> np.ndarray:
        return self._fn(params)


def create_least_squares_objective(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    y_data: np.ndarray,
    normalize: bool = True,
    use_log_residuals: bool = False,
) -> ResidualFunction:
    """Create residual function for NLSQ least-squares fitting.

    IMPORTANT: This now returns a RESIDUAL FUNCTION (vector output), not a scalar
    objective. NLSQ minimizes sum(residuals²), so this provides per-point residuals
    to the optimizer, which enables proper gradient computation and weighting.

    For complex data (e.g., G* = G' + iG"), returns stacked real and imaginary
    residuals: [real₁, ..., real_n, imag₁, ..., imag_n] with shape (2N,).

    For real data, returns residuals with shape (N,).

    **Log-space residuals (NEW)**: For rheological data spanning many decades (e.g.,
    mastercurves with 8+ decades), use `use_log_residuals=True` to compute residuals
    in log10 space. This gives equal weight to all frequency ranges and prevents
    optimizer bias toward high-modulus regions.

    Args:
        model_fn: Model function that takes (x_data, parameters) and returns predictions
        x_data: Independent variable data
        y_data: Dependent variable data (observations, may be complex)
        normalize: Whether to use relative error (default: True)
        use_log_residuals: Whether to compute residuals in log10 space (default: False).
            Recommended for data spanning >8 decades. Formula:
            ``residual = log10(abs(y_pred)) - log10(abs(y_data))``

    Returns:
        Residual function that takes parameters and returns residual vector

    Example:
        >>> def linear_model(x, params):
        ...     a, b = params
        ...     return a * x + b
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
        >>> residual_fn = create_least_squares_objective(linear_model, x, y)
        >>> # Now use with nlsq_optimize - it receives proper residual vector
        >>>
        >>> # For mastercurve data (wide frequency range):
        >>> residual_fn_log = create_least_squares_objective(
        ...     model_fn, omega, G_star, use_log_residuals=True
        ... )
    """
    # Convert to JAX arrays and detect if complex
    x_data_jax = jnp.asarray(x_data, dtype=jnp.float64)

    # Preserve complex type for y_data
    y_data_is_complex = jnp.iscomplexobj(y_data) or np.iscomplexobj(y_data)
    if y_data_is_complex:
        y_data_jax = jnp.asarray(y_data, dtype=jnp.complex128)
    else:
        y_data_jax = jnp.asarray(y_data, dtype=jnp.float64)

    # P2-Fit-1: Compute a relative normalization floor from the overall data
    # magnitude.  At the elastic plateau G'' can be orders of magnitude below
    # G', so an absolute floor of 1e-10 Pa biases the optimizer.  Using a
    # relative floor (1e-10 * max|y|) keeps the weighting proportional.
    _norm_floor = jnp.float64(1e-10)
    if normalize and not use_log_residuals:
        _max_modulus = jnp.max(jnp.abs(y_data_jax))
        _norm_floor = jnp.maximum(jnp.float64(1e-10), jnp.float64(1e-10) * _max_modulus)

    def residuals(params: np.ndarray) -> np.ndarray:
        """Compute residual vector for all data points."""
        # Ensure params are JAX arrays
        params_jax = jnp.asarray(params, dtype=jnp.float64)

        # Get model predictions
        y_pred = model_fn(x_data_jax, params_jax)

        # Check prediction format: complex, 2D [G', G"], or real
        y_pred_is_complex = jnp.iscomplexobj(y_pred)
        y_pred_is_2d = y_pred.ndim == 2 and y_pred.shape[-1] == 2

        if y_pred_is_2d:
            # Case 1: 2D [G', G"] format (e.g., FractionalZenerSolidSolid)
            if y_data_is_complex:
                # Fit to real and imaginary parts separately
                if use_log_residuals:
                    # Log-space residuals: abs() intentionally strips sign because
                    # G' and G'' are physically positive; normalize is implicit
                    resid_real = jnp.log10(
                        jnp.maximum(jnp.abs(y_pred[:, 0]), 1e-20)
                    ) - jnp.log10(jnp.maximum(jnp.abs(jnp.real(y_data_jax)), 1e-20))
                    resid_imag = jnp.log10(
                        jnp.maximum(jnp.abs(y_pred[:, 1]), 1e-20)
                    ) - jnp.log10(jnp.maximum(jnp.abs(jnp.imag(y_data_jax)), 1e-20))
                else:
                    resid_real = y_pred[:, 0] - jnp.real(y_data_jax)
                    resid_imag = y_pred[:, 1] - jnp.imag(y_data_jax)

                    if normalize:
                        resid_real = resid_real / jnp.maximum(
                            jnp.abs(jnp.real(y_data_jax)), _norm_floor
                        )
                        resid_imag = resid_imag / jnp.maximum(
                            jnp.abs(jnp.imag(y_data_jax)), _norm_floor
                        )

                return jnp.concatenate([resid_real, resid_imag])
            else:
                # Check if data is also 2D [G', G"]
                y_data_is_2d = y_data_jax.ndim == 2 and y_data_jax.shape[-1] == 2
                if y_data_is_2d:
                    # Both (N, 2): fit both columns independently (stacked residuals)
                    if use_log_residuals:
                        # Log-space residuals: abs() intentionally strips sign because
                        # G' and G'' are physically positive; normalize is implicit
                        resid_col0 = jnp.log10(
                            jnp.maximum(jnp.abs(y_pred[:, 0]), 1e-20)
                        ) - jnp.log10(jnp.maximum(jnp.abs(y_data_jax[:, 0]), 1e-20))
                        resid_col1 = jnp.log10(
                            jnp.maximum(jnp.abs(y_pred[:, 1]), 1e-20)
                        ) - jnp.log10(jnp.maximum(jnp.abs(y_data_jax[:, 1]), 1e-20))
                    else:
                        resid_col0 = y_pred[:, 0] - y_data_jax[:, 0]
                        resid_col1 = y_pred[:, 1] - y_data_jax[:, 1]

                        if normalize:
                            resid_col0 = resid_col0 / jnp.maximum(
                                jnp.abs(y_data_jax[:, 0]), _norm_floor
                            )
                            resid_col1 = resid_col1 / jnp.maximum(
                                jnp.abs(y_data_jax[:, 1]), _norm_floor
                            )

                    return jnp.concatenate([resid_col0, resid_col1])
                else:
                    # (N, 2) pred, (N,) data: fit to magnitude |G*|
                    y_pred_magnitude = jnp.sqrt(
                        y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2
                    )

                    residuals = y_pred_magnitude - y_data_jax

                    if normalize:
                        residuals = residuals / jnp.maximum(
                            jnp.abs(y_data_jax), _norm_floor
                        )

                    return residuals

        elif y_pred_is_complex:
            # Case 2: Complex predictions (G' + iG")
            if y_data_is_complex:
                # Both complex: fit real and imaginary parts separately
                if use_log_residuals:
                    # Log-space residuals for rheological data (mastercurves)
                    # Use magnitudes to avoid log of negative numbers
                    resid_real = jnp.log10(
                        jnp.maximum(jnp.abs(jnp.real(y_pred)), 1e-20)
                    ) - jnp.log10(jnp.maximum(jnp.abs(jnp.real(y_data_jax)), 1e-20))
                    resid_imag = jnp.log10(
                        jnp.maximum(jnp.abs(jnp.imag(y_pred)), 1e-20)
                    ) - jnp.log10(jnp.maximum(jnp.abs(jnp.imag(y_data_jax)), 1e-20))
                    # Note: normalize has no effect in log space (already relative)
                else:
                    # Linear residuals (default)
                    resid_real = jnp.real(y_pred) - jnp.real(y_data_jax)
                    resid_imag = jnp.imag(y_pred) - jnp.imag(y_data_jax)

                    if normalize:
                        resid_real = resid_real / jnp.maximum(
                            jnp.abs(jnp.real(y_data_jax)), _norm_floor
                        )
                        resid_imag = resid_imag / jnp.maximum(
                            jnp.abs(jnp.imag(y_data_jax)), _norm_floor
                        )

                return jnp.concatenate([resid_real, resid_imag])
            else:
                # Complex predictions, real data: fit to magnitude |G*|
                # This is the common case for oscillation mode fitting
                y_pred_magnitude = jnp.abs(y_pred)
                residuals = y_pred_magnitude - y_data_jax

                if normalize:
                    residuals = residuals / jnp.maximum(jnp.abs(y_data_jax), _norm_floor)

                return residuals
        else:
            # Case 3: Real predictions
            if y_data_is_complex:
                # Real predictions, complex data: this is unusual but handle it
                # Fit to magnitude of data
                y_data_magnitude = jnp.abs(y_data_jax)

                if use_log_residuals:
                    # Log-space residuals
                    residuals = jnp.log10(
                        jnp.maximum(jnp.abs(y_pred), 1e-20)
                    ) - jnp.log10(jnp.maximum(y_data_magnitude, 1e-20))
                else:
                    residuals = y_pred - y_data_magnitude
                    if normalize:
                        residuals = residuals / jnp.maximum(y_data_magnitude, _norm_floor)

                return residuals
            else:
                # Both real: standard case
                if use_log_residuals:
                    # Log-space residuals for rheological data
                    # Handle both positive and negative values by using absolute value
                    residuals = jnp.log10(
                        jnp.maximum(jnp.abs(y_pred), 1e-20)
                    ) - jnp.log10(jnp.maximum(jnp.abs(y_data_jax), 1e-20))
                else:
                    residuals = y_pred - y_data_jax
                    if normalize:
                        residuals = residuals / jnp.maximum(jnp.abs(y_data_jax), _norm_floor)

                return residuals

    # P1-6: Compute normalization weights so that downstream
    # OptimizationResult can un-normalize for correct R²/AIC/BIC.
    weights: np.ndarray | None = None
    if normalize and not use_log_residuals:
        if y_data_is_complex:
            w_real = np.asarray(jnp.maximum(jnp.abs(jnp.real(y_data_jax)), _norm_floor))
            w_imag = np.asarray(jnp.maximum(jnp.abs(jnp.imag(y_data_jax)), _norm_floor))
            weights = np.concatenate([w_real, w_imag])
        elif y_data_jax.ndim == 2 and y_data_jax.shape[-1] == 2:
            w0 = np.asarray(jnp.maximum(jnp.abs(y_data_jax[:, 0]), _norm_floor))
            w1 = np.asarray(jnp.maximum(jnp.abs(y_data_jax[:, 1]), _norm_floor))
            weights = np.concatenate([w0, w1])
        else:
            weights = np.asarray(jnp.maximum(jnp.abs(y_data_jax), _norm_floor))

    return ResidualFunction(residuals, normalization_weights=weights)


# Convenience aliases for compatibility with different naming conventions
optimize = nlsq_optimize  # Generic name
fit_parameters = nlsq_optimize  # More descriptive for model fitting


__all__ = [
    "OptimizationResult",
    "ResidualFunction",
    "nlsq_optimize",
    "nlsq_multistart_optimize",
    "nlsq_curve_fit",
    "optimize_with_bounds",
    "residual_sum_of_squares",
    "create_least_squares_objective",
    "optimize",
    "fit_parameters",
]
