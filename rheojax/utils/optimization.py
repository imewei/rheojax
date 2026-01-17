"""Optimization utilities for parameter fitting using NLSQ.

This module provides GPU-accelerated optimization using the NLSQ package
(https://github.com/imewei/NLSQ). NLSQ provides 5-270x speedup over scipy
through JAX JIT compilation and automatic differentiation.

Critical: This module imports NLSQ, which must be imported before JAX to
enable float64 precision mode. The rheo package handles this automatically
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
    """Result from optimization with NLSQ 0.6.0 CurveFitResult-compatible properties.

    This dataclass stores the results of NLSQ optimization, including optimal
    parameter values, objective function value, convergence information, and
    statistical metrics compatible with NLSQ 0.6.0's CurveFitResult.

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

    Statistical Properties (NLSQ 0.6.0 CurveFitResult compatible):
        r_squared: Coefficient of determination (R²)
        adj_r_squared: Adjusted R² accounting for number of parameters
        rmse: Root mean squared error
        mae: Mean absolute error
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion

    Methods:
        confidence_intervals(alpha): Compute parameter confidence intervals
        get_parameter_uncertainties(): Get standard errors from covariance diagonal
    """

    x: np.ndarray
    fun: float
    jac: np.ndarray | None = None
    pcov: np.ndarray | None = None
    success: bool = True
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0
    optimality: float | None = None
    active_mask: np.ndarray | None = None
    cost: float | None = None
    grad: np.ndarray | None = None
    nlsq_result: dict[str, Any] | None = field(default=None, repr=False)
    # NEW: Fields for statistical metrics (NLSQ 0.6.0 compatibility)
    residuals: np.ndarray | None = field(default=None, repr=False)
    y_data: np.ndarray | None = field(default=None, repr=False)
    n_data: int | None = None

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

        if np.iscomplexobj(y_data):
            y_data = np.abs(y_data)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)

        ss_res = np.sum(residuals**2)
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

        n = self.n_data or (len(self.y_data) if self.y_data is not None else None)
        if n is None:
            return None

        p = len(self.x)
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

        n = self.n_data or (len(self.residuals) if self.residuals is not None else None)
        if n is None or n == 0:
            return None

        k = len(self.x)
        residuals = np.asarray(self.residuals)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)
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

        n = self.n_data or (len(self.residuals) if self.residuals is not None else None)
        if n is None or n == 0:
            return None

        k = len(self.x)
        residuals = np.asarray(self.residuals)
        if np.iscomplexobj(residuals):
            residuals = np.abs(residuals)
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

        n = self.n_data or (len(self.residuals) if self.residuals is not None else 0)
        p = len(self.x)

        # Degrees of freedom
        dof = max(n - p, 1)

        # t-value for confidence level
        t_val = stats.t.ppf((1 + alpha) / 2, dof)

        # Standard errors from covariance diagonal
        perr = np.sqrt(np.diag(self.pcov))

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

        return np.sqrt(np.diag(self.pcov))

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
        if y_data is not None:
            y_data_np = np.asarray(y_data)
            n_data = len(y_data_np)
        elif residuals_np is not None:
            n_data = residuals_np.size

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
        )


def nlsq_optimize(
    objective: Callable[[np.ndarray], float],
    parameters: ParameterSet,
    method: str = "auto",
    use_jax: bool = True,
    max_iter: int = 1000,
    ftol: float = 1e-6,
    xtol: float = 1e-6,
    gtol: float = 1e-6,
    **kwargs,
) -> OptimizationResult:
    """Optimize objective function using NLSQ (GPU-accelerated).

    This function provides GPU-accelerated nonlinear least squares optimization
    using the NLSQ package. It achieves 5-270x speedup over scipy through JAX
    JIT compilation and automatic differentiation.

    The objective function should accept parameter values as a 1D array and
    return a scalar value to minimize. NLSQ internally converts this to a
    residual minimization problem.

    Args:
        objective: Objective function to minimize. Takes parameter values as
            array and returns scalar. Should use jax.numpy for operations to
            enable GPU acceleration and automatic differentiation.
        parameters: ParameterSet with initial values and bounds
        method: Optimization method. Options:

            - "auto": Automatically select based on bounds (default)
            - "trf": Trust Region Reflective (supports bounds)
            - "lm": Levenberg-Marquardt (no bounds)

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
        **kwargs: Additional arguments passed to nlsq.LeastSquares.least_squares

    Returns:
        OptimizationResult with optimal parameters and convergence info

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
    x0 = parameters.get_values()
    original_values = np.asarray(x0, dtype=np.float64).copy()
    bounds_list = parameters.get_bounds()

    logger.info(
        "Starting NLSQ optimization",
        n_params=len(x0),
        method=method,
        max_iter=max_iter,
        ftol=ftol,
        xtol=xtol,
        gtol=gtol,
    )
    logger.debug(
        "Initial parameter values",
        x0=x0.tolist() if hasattr(x0, "tolist") else list(x0),
        bounds=bounds_list,
    )

    # Ensure float64 precision for initial values
    x0 = np.asarray(x0, dtype=np.float64)

    # Convert bounds to NLSQ format: (lower_array, upper_array)
    lower_bounds = []
    upper_bounds = []
    for bound_pair in bounds_list:
        if bound_pair is None or (bound_pair[0] is None and bound_pair[1] is None):
            # Unbounded
            lower_bounds.append(-np.inf)
            upper_bounds.append(np.inf)
        else:
            lower = bound_pair[0] if bound_pair[0] is not None else -np.inf
            upper = bound_pair[1] if bound_pair[1] is not None else np.inf
            lower_bounds.append(lower)
            upper_bounds.append(upper)

    lower_bounds = np.asarray(lower_bounds, dtype=np.float64)
    upper_bounds = np.asarray(upper_bounds, dtype=np.float64)
    nlsq_bounds = (lower_bounds, upper_bounds)

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

    # Merge with user-provided kwargs
    nlsq_kwargs.update(kwargs)

    def _scipy_fallback(initial_guess: np.ndarray) -> OptimizationResult:
        """Fallback to SciPy's least_squares when NLSQ fails."""
        logger.info("Using SciPy least_squares fallback")
        from scipy.optimize import least_squares as scipy_least_squares

        def residual_fn(values: np.ndarray) -> np.ndarray:
            res = objective(values)
            if isinstance(res, jnp.ndarray):
                res = np.asarray(res)
            return np.asarray(res, dtype=np.float64)

        scipy_result = scipy_least_squares(
            residual_fn,
            initial_guess,
            bounds=nlsq_bounds,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
            max_nfev=max_iter * 10,
            method="trf",
        )

        cost_value = getattr(scipy_result, "cost", None)

        # Extract Jacobian and compute covariance
        jac = None
        pcov = None
        if scipy_result.jac is not None:
            jac = np.asarray(scipy_result.jac, dtype=np.float64)
            residuals = residual_fn(scipy_result.x)
            pcov = compute_covariance_from_jacobian(jac, residuals)

        result = OptimizationResult(
            x=np.asarray(scipy_result.x, dtype=np.float64),
            fun=(
                float(2.0 * scipy_result.cost)
                if hasattr(scipy_result, "cost")
                else float(np.sum(residual_fn(scipy_result.x) ** 2))
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
        )

        return result

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
    except Exception as e:
        logger.warning(
            "NLSQ optimization raised exception, falling back to SciPy",
            error=str(e),
            exc_info=True,
        )
        return _scipy_fallback(x0)

    # Compute residuals at optimal point for covariance scaling
    x_opt = np.asarray(nlsq_result.get("x", x0), dtype=np.float64)
    residuals = objective(x_opt)
    residuals_np = np.asarray(residuals, dtype=np.float64)

    # Convert NLSQ result to OptimizationResult (with residuals for covariance)
    result = OptimizationResult.from_nlsq(nlsq_result, residuals=residuals_np)

    if (
        not result.success
        and "inner optimization loop exceeded" in result.message.lower()
    ):
        logger.warning(
            "NLSQ hit inner iteration limit; retrying with SciPy least_squares for stability."
        )
        return _scipy_fallback(x0)

    # Ensure x is float64
    result.x = np.asarray(result.x, dtype=np.float64)

    # Compute RSS = sum(residuals²)
    result.fun = float(jnp.sum(residuals**2))

    # Guard against false "success" with astronomically large residuals
    residuals_np = np.asarray(residuals)
    residual_count = residuals_np.size if residuals_np.size else 1
    mean_squared_error = result.fun / residual_count
    if not np.isfinite(mean_squared_error) or mean_squared_error > 1e6:
        logger.error(
            "Optimization failed: residual norm extremely large",
            mean_squared_error=(
                float(mean_squared_error) if np.isfinite(mean_squared_error) else "inf"
            ),
            residual_count=residual_count,
            rss=float(result.fun),
        )
        parameters.set_values(original_values)
        raise RuntimeError(
            "Optimization failed: residual norm remains extremely large. "
            "Try providing better initial values, looser bounds, or scaling the data."
        )

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
    objective: Callable[[np.ndarray], float],
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
    **kwargs,
) -> OptimizationResult:
    """Multi-start optimization to escape local minima.

    For complex objective functions (e.g., mastercurves with 10+ decades),
    single optimization runs may converge to poor local minima even from
    good initial guesses. This function performs multiple optimization runs
    from different starting points and returns the best result.

    Strategy:
        1. First attempt: Use current parameter values (from smart initialization)
        2. Additional attempts: Random perturbations around initial values
        3. Return result with lowest final cost (best fit)

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
    # Store original parameter values
    original_values = parameters.get_values()

    logger.info(
        "Starting multi-start optimization",
        n_starts=n_starts,
        perturb_factor=perturb_factor,
        n_params=len(original_values),
    )

    # First attempt: Use smart initialization values
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

    # Additional attempts: Random perturbations
    bounds_list = parameters.get_bounds()

    for i in range(1, n_starts):
        logger.debug("Starting multi-start attempt", attempt=i + 1, total=n_starts)
        if verbose:
            logger.info(
                f"Multi-start optimization: Attempt {i+1} (random perturbation)"
            )

        # Generate perturbed initial values
        perturbed_values = []
        for orig_val, bounds in zip(original_values, bounds_list, strict=True):
            if bounds is None or (bounds[0] is None and bounds[1] is None):
                # No bounds - perturb by fraction of value
                perturbation = np.random.uniform(-perturb_factor, perturb_factor)
                new_val = orig_val * (1.0 + perturbation)
            else:
                # With bounds - perturb within range
                lower = bounds[0] if bounds[0] is not None else orig_val - abs(orig_val)
                upper = bounds[1] if bounds[1] is not None else orig_val + abs(orig_val)
                range_size = upper - lower
                perturbation = np.random.uniform(
                    -perturb_factor * range_size, perturb_factor * range_size
                )
                new_val = np.clip(orig_val + perturbation, lower, upper)

            perturbed_values.append(new_val)

        # Set perturbed values and optimize
        parameters.set_values(perturbed_values)

        try:
            result = nlsq_optimize(
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

            logger.debug(
                "Multi-start attempt completed",
                attempt=i + 1,
                cost=float(result.fun),
                success=result.success,
            )
            if verbose:
                logger.info(f"  Cost: {result.fun:.3e}, Success: {result.success}")

            # Keep best result
            if result.success and result.fun < best_cost:
                best_result = result
                best_cost = result.fun
                logger.debug(
                    "New best result found",
                    attempt=i + 1,
                    best_cost=float(best_cost),
                )
                if verbose:
                    logger.info(f"  -> New best! Cost: {best_cost:.3e}")

        except Exception as e:
            logger.warning(
                "Multi-start attempt failed",
                attempt=i + 1,
                error=str(e),
            )
            if verbose:
                logger.warning(f"  Attempt {i+1} failed: {e}")
            continue

    # Restore best parameters
    parameters.set_values(best_result.x)

    logger.info(
        "Multi-start optimization completed",
        best_cost=float(best_cost),
        n_starts=n_starts,
        final_success=best_result.success,
    )
    if verbose:
        logger.info(
            f"\nMulti-start completed: Best cost = {best_cost:.3e} "
            f"({n_starts} starts)"
        )

    return best_result


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
    **kwargs,
) -> OptimizationResult:
    """Curve fitting using NLSQ 0.6.0 curve_fit() API with advanced features.

    This function provides access to NLSQ 0.6.0's enhanced curve_fit() features
    including auto-bounds, stability checks, fallback strategies, and model
    diagnostics. It returns an OptimizationResult with CurveFitResult-compatible
    statistical properties (r_squared, rmse, aic, bic, etc.).

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
        **kwargs: Additional arguments passed to nlsq.curve_fit()

    Returns:
        OptimizationResult with CurveFitResult-compatible statistical properties:
        - r_squared, adj_r_squared, rmse, mae, aic, bic
        - confidence_intervals(alpha) method
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

    Notes:
        - This function uses nlsq.curve_fit() directly (not LeastSquares.least_squares())
        - The model function signature is ``f(x, params_array)`` not ``f(x, *params)``
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
    )

    # Extract p0 and bounds from ParameterSet
    p0 = np.asarray(parameters.get_values(), dtype=np.float64)
    bounds_list = parameters.get_bounds()
    lower = np.array(
        [b[0] if b[0] is not None else -np.inf for b in bounds_list], dtype=np.float64
    )
    upper = np.array(
        [b[1] if b[1] is not None else np.inf for b in bounds_list], dtype=np.float64
    )

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
    curve_fit_kwargs.update(kwargs)

    try:
        # Call nlsq.curve_fit() - returns CurveFitResult (tuple unpacking compatible)
        curve_fit_result = nlsq_module.curve_fit(
            f_wrapper, x_data_np, y_data_np, **curve_fit_kwargs
        )

        # CurveFitResult supports both tuple unpacking and attribute access
        # We need to check if it's a tuple (popt, pcov) or CurveFitResult object
        if isinstance(curve_fit_result, tuple):
            popt, pcov = curve_fit_result
            success = True
            message = "Optimization converged successfully"
            nfev = 0
            njev = 0
            diagnostics = None
        else:
            # It's a CurveFitResult object
            popt = np.asarray(curve_fit_result.popt)
            pcov = curve_fit_result.pcov
            success = getattr(curve_fit_result, "success", True)
            message = getattr(curve_fit_result, "message", "Converged")
            nfev = getattr(curve_fit_result, "nfev", 0)
            njev = getattr(curve_fit_result, "njev", 0)
            diagnostics = getattr(curve_fit_result, "diagnostics", None)

        # Compute residuals and y_pred at optimal point
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

        # Create OptimizationResult with all metrics
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
            n_data=len(y_data_np),
        )

        # Store diagnostics if available
        if diagnostics is not None:
            result.nlsq_result = {"diagnostics": diagnostics}

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

    except Exception as e:
        logger.warning(
            "nlsq.curve_fit() failed, falling back to nlsq_optimize",
            error=str(e),
            exc_info=True,
        )

        # Fallback to nlsq_optimize with residual-based objective
        objective = create_least_squares_objective(model_fn, x_data_np, y_data_np)

        if multistart:
            return nlsq_multistart_optimize(
                objective, parameters, n_starts=n_starts, **kwargs
            )
        else:
            return nlsq_optimize(objective, parameters, **kwargs)


def optimize_with_bounds(
    objective: Callable[[np.ndarray], float],
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
    for i, (val, bound) in enumerate(zip(x0, bounds, strict=False)):
        params.add(name=f"p{i}", value=val, bounds=bound)

    # Use main optimization function
    return nlsq_optimize(objective, params, use_jax=use_jax, **kwargs)


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
    if isinstance(y_pred, jnp.ndarray) or isinstance(y_true, jnp.ndarray):
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


def create_least_squares_objective(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    y_data: np.ndarray,
    normalize: bool = True,
    use_log_residuals: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create residual function for NLSQ least-squares fitting.

    IMPORTANT: This now returns a RESIDUAL FUNCTION (vector output), not a scalar
    objective. NLSQ minimizes sum(residuals²), so this provides per-point residuals
    to the optimizer, which enables proper gradient computation and weighting.

    For complex data (e.g., G* = G' + iG"), returns stacked real and imaginary
    residuals: [real₁, ..., realₙ, imag₁, ..., imagₙ] with shape (2N,).

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
                resid_real = y_pred[:, 0] - jnp.real(y_data_jax)
                resid_imag = y_pred[:, 1] - jnp.imag(y_data_jax)

                if normalize:
                    resid_real = resid_real / jnp.maximum(
                        jnp.abs(jnp.real(y_data_jax)), 1e-10
                    )
                    resid_imag = resid_imag / jnp.maximum(
                        jnp.abs(jnp.imag(y_data_jax)), 1e-10
                    )

                return jnp.concatenate([resid_real, resid_imag])
            else:
                # Fit to magnitude: |G*| = sqrt(G'^2 + G"^2)
                y_pred_magnitude = jnp.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2)

                # Check if data is also 2D [G', G"]
                y_data_is_2d = y_data_jax.ndim == 2 and y_data_jax.shape[-1] == 2
                if y_data_is_2d:
                    # Data is also 2D: compute magnitude
                    y_data_magnitude = jnp.sqrt(
                        y_data_jax[:, 0] ** 2 + y_data_jax[:, 1] ** 2
                    )
                else:
                    # Data is already magnitude (1D)
                    y_data_magnitude = y_data_jax

                residuals = y_pred_magnitude - y_data_magnitude

                if normalize:
                    residuals = residuals / jnp.maximum(
                        jnp.abs(y_data_magnitude), 1e-10
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
                            jnp.abs(jnp.real(y_data_jax)), 1e-10
                        )
                        resid_imag = resid_imag / jnp.maximum(
                            jnp.abs(jnp.imag(y_data_jax)), 1e-10
                        )

                return jnp.concatenate([resid_real, resid_imag])
            else:
                # Complex predictions, real data: fit to magnitude |G*|
                # This is the common case for oscillation mode fitting
                y_pred_magnitude = jnp.abs(y_pred)
                residuals = y_pred_magnitude - y_data_jax

                if normalize:
                    residuals = residuals / jnp.maximum(jnp.abs(y_data_jax), 1e-10)

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
                        residuals = residuals / jnp.maximum(y_data_magnitude, 1e-10)

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
                        residuals = residuals / jnp.maximum(jnp.abs(y_data_jax), 1e-10)

                return residuals

    return residuals


# Convenience aliases for compatibility with different naming conventions
optimize = nlsq_optimize  # Generic name
fit_parameters = nlsq_optimize  # More descriptive for model fitting


__all__ = [
    "OptimizationResult",
    "nlsq_optimize",
    "nlsq_multistart_optimize",
    "nlsq_curve_fit",
    "optimize_with_bounds",
    "residual_sum_of_squares",
    "create_least_squares_objective",
    "optimize",
    "fit_parameters",
]
