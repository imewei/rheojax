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
from typing import Any, Union

import nlsq
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()


ArrayLike = Union[np.ndarray, jnp.ndarray, list, float]


@dataclass
class OptimizationResult:
    """Result from optimization.

    This dataclass stores the results of NLSQ optimization, including optimal
    parameter values, objective function value, convergence information, and
    NLSQ-specific diagnostic data.

    Attributes:
        x: Optimal parameter values (float64 array)
        fun: Objective function value at optimum
        jac: Jacobian (gradient) at optimum
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
    """

    x: np.ndarray
    fun: float
    jac: np.ndarray | None = None
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

    @classmethod
    def from_nlsq(cls, nlsq_result: dict[str, Any]) -> OptimizationResult:
        """Create OptimizationResult from NLSQ result dictionary.

        Args:
            nlsq_result: Result dictionary from nlsq.LeastSquares.least_squares

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

        return cls(
            x=x,
            fun=fun,
            jac=jac,
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
        )


def nlsq_optimize(
    objective: Callable[[np.ndarray], float],
    parameters: ParameterSet,
    method: str = "auto",
    use_jax: bool = True,
    max_iter: int = 1000,
    ftol: float = 1e-8,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
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
        ftol: Function tolerance for convergence (default: 1e-8)
        xtol: Parameter tolerance for convergence (default: 1e-8)
        gtol: Gradient tolerance for convergence (default: 1e-8)
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
    bounds_list = parameters.get_bounds()

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

    # NLSQ expects a residual function, but we have a scalar objective
    # Convert scalar objective to residual by taking square root
    # This maintains the same minimum while allowing NLSQ to work
    def residual_function(params):
        """Convert scalar objective to residual for NLSQ."""
        # Ensure params are JAX arrays for autodiff
        params_jax = jnp.asarray(params, dtype=jnp.float64)
        obj_value = objective(params_jax)

        # Convert scalar objective to residual (single-element array)
        # NLSQ minimizes sum(residuals**2), so we return sqrt(objective)
        # This gives the same minimum as minimizing objective directly
        residual = jnp.sqrt(jnp.maximum(obj_value, 0.0))  # Ensure non-negative
        return jnp.array([residual])  # Return as 1-element array

    # Set up NLSQ optimization parameters
    nlsq_kwargs = {
        "fun": residual_function,
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

    # Create NLSQ optimizer instance and run optimization
    try:
        optimizer = nlsq.LeastSquares()
        nlsq_result = optimizer.least_squares(**nlsq_kwargs)
    except Exception as e:
        # If NLSQ fails, return a failure result
        return OptimizationResult(
            x=x0,
            fun=float(objective(x0)),
            success=False,
            message=f"NLSQ optimization failed: {str(e)}",
            nit=0,
            nfev=0,
        )

    # Convert NLSQ result to OptimizationResult
    result = OptimizationResult.from_nlsq(nlsq_result)

    # Ensure x is float64
    result.x = np.asarray(result.x, dtype=np.float64)

    # Update ParameterSet with optimal values
    parameters.set_values(result.x)

    # Recompute objective at optimal point (NLSQ returns residual cost)
    result.fun = float(objective(result.x))

    return result


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
        is_complex = jnp.iscomplexobj(y_true_jax) or jnp.iscomplexobj(y_pred_jax)

        if is_complex:
            # Handle complex data: fit both real and imaginary parts
            # This is critical for oscillatory shear (G' + iG")
            residuals_real = jnp.real(y_pred_jax) - jnp.real(y_true_jax)
            residuals_imag = jnp.imag(y_pred_jax) - jnp.imag(y_true_jax)

            if normalize:
                # Normalize separately by magnitude of real and imaginary parts
                residuals_real = residuals_real / jnp.maximum(
                    jnp.abs(jnp.real(y_true_jax)), 1e-10
                )
                residuals_imag = residuals_imag / jnp.maximum(
                    jnp.abs(jnp.imag(y_true_jax)), 1e-10
                )

            # Sum of squares for both components
            rss = jnp.sum(residuals_real**2) + jnp.sum(residuals_imag**2)
        else:
            # Real data path (original behavior)
            y_true_jax = jnp.asarray(y_true_jax, dtype=jnp.float64)
            y_pred_jax = jnp.asarray(y_pred_jax, dtype=jnp.float64)
            residuals = y_pred_jax - y_true_jax

            if normalize:
                # Relative error (avoid division by zero)
                residuals = residuals / jnp.maximum(jnp.abs(y_true_jax), 1e-10)

            rss = jnp.sum(residuals**2)

        # Return scalar JAX array, don't convert to Python float (breaks gradients)
        return rss
    else:
        # NumPy path
        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)

        # Check if data is complex
        is_complex = np.iscomplexobj(y_true_np) or np.iscomplexobj(y_pred_np)

        if is_complex:
            # Handle complex data
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
            # Real data path (original behavior)
            y_true_np = np.asarray(y_true_np, dtype=np.float64)
            y_pred_np = np.asarray(y_pred_np, dtype=np.float64)
            residuals = y_pred_np - y_true_np

            if normalize:
                # Relative error (avoid division by zero)
                with np.errstate(divide="ignore", invalid="ignore"):
                    residuals = residuals / np.maximum(np.abs(y_true_np), 1e-10)

            rss = float(np.sum(residuals**2))

        return rss


def create_least_squares_objective(
    model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_data: np.ndarray,
    y_data: np.ndarray,
    normalize: bool = True,
) -> Callable[[np.ndarray], float]:
    """Create least squares objective function for model fitting.

    Args:
        model_fn: Model function that takes (x_data, parameters) and returns predictions
        x_data: Independent variable data
        y_data: Dependent variable data (observations)
        normalize: Whether to use relative error (default: True)

    Returns:
        Objective function that takes parameters and returns RSS

    Example:
        >>> def linear_model(x, params):
        ...     a, b = params
        ...     return a * x + b
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> y = np.array([2.1, 4.0, 5.9, 8.1, 10.0])
        >>> objective = create_least_squares_objective(linear_model, x, y)
        >>> # Now use with nlsq_optimize
    """

    def objective(params: np.ndarray) -> float:
        """Objective function to minimize."""
        y_pred = model_fn(x_data, params)
        return residual_sum_of_squares(y_data, y_pred, normalize=normalize)

    return objective


# Convenience aliases for compatibility with different naming conventions
optimize = nlsq_optimize  # Generic name
fit_parameters = nlsq_optimize  # More descriptive for model fitting


__all__ = [
    "OptimizationResult",
    "nlsq_optimize",
    "optimize_with_bounds",
    "residual_sum_of_squares",
    "create_least_squares_objective",
    "optimize",
    "fit_parameters",
]
