"""Utility functions for the rheo package.

This module provides:
- Mittag-Leffler function implementations
- Optimization utilities
- Helper functions for numerical computations
"""

from rheo.utils.optimization import (
    OptimizationResult,
    nlsq_optimize,
    optimize_with_bounds,
    residual_sum_of_squares,
    create_least_squares_objective,
    optimize,
    fit_parameters,
)

__all__ = [
    # Optimization utilities
    "OptimizationResult",
    "nlsq_optimize",
    "optimize_with_bounds",
    "residual_sum_of_squares",
    "create_least_squares_objective",
    "optimize",
    "fit_parameters",
    # These will be imported as they are implemented
    # "mittag_leffler",
    # "numerical_derivatives",
    # "interpolate_data",
]