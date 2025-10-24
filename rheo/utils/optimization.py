"""Optimization utilities for parameter fitting.

This module provides optimization wrappers that integrate JAX automatic
differentiation with scipy.optimize for model parameter fitting.

The original plan was to use NLSQ (https://github.com/imewei/NLSQ), but as
NLSQ is not available, this implementation uses scipy.optimize as the fallback
with JAX gradients for enhanced performance.

Example:
    >>> from rheo.core.parameters import Parameter, ParameterSet
    >>> from rheo.utils.optimization import nlsq_optimize
    >>>
    >>> # Set up parameters
    >>> params = ParameterSet()
    >>> params.add(Parameter(name="x", value=1.0, bounds=(0, 10)))
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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize, OptimizeResult as ScipyOptimizeResult

from rheo.core.parameters import ParameterSet


@dataclass
class OptimizationResult:
    """Result from optimization.

    Attributes:
        x: Optimal parameter values
        fun: Objective function value at optimum
        jac: Jacobian (gradient) at optimum
        success: Whether optimization converged successfully
        message: Status message from optimizer
        nit: Number of iterations
        nfev: Number of function evaluations
        njev: Number of Jacobian evaluations
    """

    x: np.ndarray
    fun: float
    jac: Optional[np.ndarray] = None
    success: bool = True
    message: str = ""
    nit: int = 0
    nfev: int = 0
    njev: int = 0

    @classmethod
    def from_scipy(cls, scipy_result: ScipyOptimizeResult) -> "OptimizationResult":
        """Create OptimizationResult from scipy OptimizeResult.

        Args:
            scipy_result: Result from scipy.optimize

        Returns:
            OptimizationResult instance
        """
        return cls(
            x=np.array(scipy_result.x),
            fun=float(scipy_result.fun),
            jac=np.array(scipy_result.jac) if hasattr(scipy_result, "jac") else None,
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            nit=int(scipy_result.nit) if hasattr(scipy_result, "nit") else 0,
            nfev=int(scipy_result.nfev) if hasattr(scipy_result, "nfev") else 0,
            njev=int(scipy_result.njev) if hasattr(scipy_result, "njev") else 0,
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
    """Optimize objective function with respect to parameters.

    This function provides a unified interface for parameter optimization with
    automatic gradient computation via JAX. Originally designed to use NLSQ,
    it falls back to scipy.optimize with JAX gradients.

    Args:
        objective: Objective function to minimize. Takes parameter values as
            array and returns scalar.
        parameters: ParameterSet with initial values and bounds
        method: Optimization method. Options:
            - "auto": Automatically select based on bounds (default)
            - "L-BFGS-B": L-BFGS with bounds
            - "TNC": Truncated Newton with bounds
            - "SLSQP": Sequential Least Squares
            - "trust-constr": Trust region with constraints
        use_jax: Whether to use JAX for gradient computation (default: True)
        max_iter: Maximum number of iterations (default: 1000)
        ftol: Function tolerance for convergence (default: 1e-8)
        xtol: Parameter tolerance for convergence (default: 1e-8)
        gtol: Gradient tolerance for convergence (default: 1e-8)
        **kwargs: Additional arguments passed to scipy.optimize.minimize

    Returns:
        OptimizationResult with optimal parameters and convergence info

    Raises:
        ValueError: If objective is not callable or parameters is not ParameterSet

    Example:
        >>> from rheo.core.parameters import Parameter, ParameterSet
        >>> params = ParameterSet()
        >>> params.add(Parameter(name="a", value=1.0, bounds=(0, 10)))
        >>> params.add(Parameter(name="b", value=1.0, bounds=(0, 10)))
        >>>
        >>> def objective(values):
        ...     a, b = values
        ...     return (a - 5.0) ** 2 + (b - 3.0) ** 2
        >>>
        >>> result = nlsq_optimize(objective, params)
        >>> print(result.x)  # Should be close to [5.0, 3.0]
    """
    if not callable(objective):
        raise ValueError("objective must be callable")

    if not isinstance(parameters, ParameterSet):
        raise ValueError("parameters must be ParameterSet")

    # Get initial values and bounds
    x0 = parameters.get_values()
    bounds = parameters.get_bounds()

    # Auto-select method based on bounds
    if method == "auto":
        has_bounds = any(b is not None for b_pair in bounds for b in b_pair if b_pair)
        method = "L-BFGS-B" if has_bounds else "BFGS"

    # Convert bounds to scipy format
    # scipy expects list of (min, max) tuples or None
    scipy_bounds = []
    for b in bounds:
        if b is None or (b[0] is None and b[1] is None):
            scipy_bounds.append((None, None))
        else:
            scipy_bounds.append((b[0], b[1]))

    # Set up JAX gradient if requested
    if use_jax:
        # Create JAX-compatible objective wrapper for gradient computation
        # Note: We must NOT convert to float inside this function, as it will
        # be differentiated by JAX
        def jax_objective_for_grad(x):
            x_jax = jnp.asarray(x)
            result = objective(x_jax)
            # Ensure scalar output, but keep as JAX array for differentiation
            if isinstance(result, (jnp.ndarray, np.ndarray)):
                result = jnp.asarray(result).reshape(())
            return result

        # Compute gradient function
        jac = jax.grad(jax_objective_for_grad)

        # Wrapper to convert JAX arrays to numpy for scipy
        def jac_wrapper(x):
            grad = jac(jnp.asarray(x))
            return np.asarray(grad)

        # Objective wrapper for scipy (can convert to float here)
        def objective_wrapper(x):
            result = objective(x)
            if isinstance(result, (jnp.ndarray, np.ndarray)):
                return float(jnp.asarray(result).reshape(()))
            return float(result)

    else:
        # Use numerical gradients (scipy default)
        jac_wrapper = None
        objective_wrapper = objective

    # Set up options
    options = {
        "maxiter": max_iter,
        "ftol": ftol,
        "gtol": gtol,
    }

    # Add method-specific options
    if method in ["L-BFGS-B", "TNC"]:
        options["maxfun"] = max_iter * 10  # Function evaluation limit

    # Merge with user options
    options.update(kwargs.get("options", {}))

    # Run optimization
    scipy_result = minimize(
        fun=objective_wrapper,
        x0=x0,
        method=method,
        jac=jac_wrapper,
        bounds=scipy_bounds if scipy_bounds else None,
        options=options,
        **{k: v for k, v in kwargs.items() if k != "options"},
    )

    # Update parameters with optimal values
    parameters.set_values(scipy_result.x)

    # Convert result
    result = OptimizationResult.from_scipy(scipy_result)

    return result


def optimize_with_bounds(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: List[Tuple[Optional[float], Optional[float]]],
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
        use_jax: Whether to use JAX for gradients
        **kwargs: Additional arguments passed to scipy.optimize.minimize

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
    from rheo.core.parameters import Parameter

    params = ParameterSet()
    for i, (val, bound) in enumerate(zip(x0, bounds)):
        params.add(Parameter(name=f"p{i}", value=val, bounds=bound))

    # Use main optimization function
    return nlsq_optimize(objective, params, use_jax=use_jax, **kwargs)


def residual_sum_of_squares(
    y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True
) -> float:
    """Compute residual sum of squares (RSS).

    Args:
        y_true: True values
        y_pred: Predicted values
        normalize: Whether to normalize by y_true (relative error)

    Returns:
        RSS value (as scalar, not necessarily Python float for JAX compatibility)

    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 2.1, 2.9])
        >>> rss = residual_sum_of_squares(y_true, y_pred)
    """
    # Use JAX operations if inputs are JAX arrays for gradient support
    if isinstance(y_pred, jnp.ndarray):
        residuals = y_pred - y_true

        if normalize:
            # Relative error (avoid division by zero)
            residuals = residuals / jnp.maximum(jnp.abs(y_true), 1e-10)

        # Return scalar JAX array, don't convert to Python float (breaks gradients)
        return jnp.sum(residuals**2)
    else:
        # NumPy path
        residuals = y_pred - y_true

        if normalize:
            # Relative error (avoid division by zero)
            with np.errstate(divide="ignore", invalid="ignore"):
                residuals = residuals / np.maximum(np.abs(y_true), 1e-10)

        return float(np.sum(residuals**2))


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
