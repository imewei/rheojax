"""Utility functions for the rheojax package.

This module provides:
- Mittag-Leffler function implementations
- Optimization utilities
- GPU detection and device management utilities
- Helper functions for numerical computations
"""

from rheojax.utils.device import (
    check_gpu_availability,
    get_device_info,
    get_gpu_memory_info,
    print_device_summary,
)
from rheojax.utils.metrics import (
    compute_fit_quality,
    r2_complex,
)
from rheojax.utils.modulus_conversion import (
    POISSON_PRESETS,
    convert_modulus,
    convert_rheodata,
)
from rheojax.utils.optimization import (
    OptimizationResult,
    create_least_squares_objective,
    fit_parameters,
    nlsq_optimize,
    optimize,
    optimize_with_bounds,
    residual_sum_of_squares,
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
    # Metrics utilities
    "compute_fit_quality",
    "r2_complex",
    # Modulus conversion (DMTA/DMA support)
    "convert_modulus",
    "convert_rheodata",
    "POISSON_PRESETS",
]
