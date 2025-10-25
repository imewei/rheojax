"""Utility functions for the rheo package.

This module provides:
- Mittag-Leffler function implementations
- Optimization utilities
- GPU detection and device management utilities
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
from rheo.utils.device import (
    check_gpu_availability,
    get_device_info,
    get_gpu_memory_info,
    print_device_summary,
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
    # Device utilities
    "check_gpu_availability",
    "get_device_info",
    "get_gpu_memory_info",
    "print_device_summary",
    # These will be imported as they are implemented
    # "mittag_leffler",
    # "numerical_derivatives",
    # "interpolate_data",
]