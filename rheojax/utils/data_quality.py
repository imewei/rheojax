"""Data quality and range detection utilities.

This module provides utilities for detecting data characteristics that affect
optimization quality, such as very wide frequency ranges (mastercurves).
"""

from __future__ import annotations

import warnings

import numpy as np

from rheojax.logging import get_logger

logger = get_logger(__name__)


def detect_data_range_decades(x: np.ndarray) -> float:
    """Detect the range of data in decades (log10 scale).

    Args:
        x: Data array (e.g., frequency, time)

    Returns:
        Range in decades (log10(max/min))

    Example:
        >>> freq = np.array([1e-8, 1e-6, 1e-4, 1e4])
        >>> decades = detect_data_range_decades(freq)
        >>> print(f"{decades:.1f} decades")  # 12.0 decades
    """
    logger.debug(
        "Detecting data range in decades",
        input_length=len(x) if hasattr(x, "__len__") else 1,
    )

    x_positive = x[x > 0]  # Filter out non-positive values
    if len(x_positive) == 0:
        logger.debug("No positive values in data, returning 0 decades")
        return 0.0

    x_min = np.min(x_positive)
    x_max = np.max(x_positive)

    if x_min <= 0 or x_max <= 0:
        logger.debug("Invalid min/max after filtering", x_min=x_min, x_max=x_max)
        return 0.0

    decades = float(np.log10(x_max / x_min))
    logger.debug(
        "Data range computed",
        x_min=x_min,
        x_max=x_max,
        decades=decades,
    )
    return decades


def check_wide_frequency_range(
    x: np.ndarray,
    threshold_decades: float = 8.0,
    warn: bool = True,
    recommend_log_residuals: bool = True,
) -> dict[str, bool | float | str]:
    """Check if data has a very wide frequency/time range (e.g., mastercurve).

    Wide-range data (>8 decades) can cause optimization problems:
    - Optimizer bias toward high-value regions
    - Poor parameter recovery
    - Convergence to local minima

    Recommended solutions:
    - Use log-space residuals (use_log_residuals=True)
    - Fit to subset of data for initialization
    - Use multi-start optimization

    Args:
        x: Independent variable data (frequency, time, etc.)
        threshold_decades: Threshold for "wide range" warning (default: 8.0)
        warn: Whether to emit a warning if range is wide (default: True)
        recommend_log_residuals: Whether to recommend log-residuals in warning

    Returns:
        Dictionary with keys:
            - 'is_wide_range': True if range > threshold
            - 'decades': Actual range in decades
            - 'recommendation': Recommended action (or empty string)

    Example:
        >>> omega = np.logspace(-8, 4, 100)  # 12 decades (mastercurve)
        >>> result = check_wide_frequency_range(omega)
        >>> if result['is_wide_range']:
        ...     print(f"Wide range detected: {result['decades']:.1f} decades")
        ...     print(result['recommendation'])
    """
    logger.debug(
        "Checking for wide frequency range",
        threshold_decades=threshold_decades,
        warn_enabled=warn,
    )

    decades = detect_data_range_decades(x)
    is_wide = decades > threshold_decades

    recommendation = ""
    if is_wide and recommend_log_residuals:
        recommendation = (
            f"Wide frequency range ({decades:.1f} decades > {threshold_decades:.0f}) detected. "
            f"Recommend using log-space residuals to prevent optimization bias:\n"
            f"  model.fit(X, y, use_log_residuals=True)\n"
            f"Or fit to a subset for initialization:\n"
            f"  X_subset = X[(X > 0.01) & (X < 100)]  # Middle 4 decades"
        )

        logger.info(
            "Wide frequency range detected",
            decades=decades,
            threshold_decades=threshold_decades,
        )

        if warn:
            warnings.warn(
                recommendation,
                UserWarning,
                stacklevel=3,
            )
    else:
        logger.debug(
            "Frequency range within normal bounds",
            decades=decades,
            threshold_decades=threshold_decades,
        )

    return {
        "is_wide_range": is_wide,
        "decades": decades,
        "recommendation": recommendation,
    }


def suggest_optimization_strategy(
    x: np.ndarray,
    y: np.ndarray,
    test_mode: str | None = None,
) -> dict[str, bool | str | float]:
    """Suggest optimization strategy based on data characteristics.

    Analyzes data range, complexity, and test mode to recommend:
    - Whether to use log-residuals
    - Whether to use multi-start optimization
    - Whether to use subset initialization

    Args:
        x: Independent variable (frequency, time, etc.)
        y: Dependent variable (modulus, stress, etc.)
        test_mode: Test mode ('oscillation', 'relaxation', 'creep')

    Returns:
        Dictionary with optimization recommendations:
            - 'use_log_residuals': Recommended for wide ranges
            - 'use_multi_start': Recommended for complex landscapes
            - 'use_subset_init': Recommended for very wide ranges
            - 'rationale': Explanation of recommendations

    Example:
        >>> omega = np.logspace(-8, 4, 100)
        >>> G_star = ...  # Complex modulus data
        >>> strategy = suggest_optimization_strategy(omega, G_star, 'oscillation')
        >>> print(strategy['rationale'])
    """
    logger.debug(
        "Analyzing data for optimization strategy",
        x_length=len(x) if hasattr(x, "__len__") else 1,
        y_length=len(y) if hasattr(y, "__len__") else 1,
        test_mode=test_mode,
        y_is_complex=np.iscomplexobj(y),
    )

    # Check data range
    range_check = check_wide_frequency_range(x, warn=False)
    decades: float = range_check["decades"]  # type: ignore[assignment]

    # Initialize recommendations
    use_log_residuals = False
    use_multi_start = False
    use_subset_init = False
    rationale_parts = []

    # Rule 1: Very wide range (>10 decades) - mastercurve
    if decades > 10:
        use_log_residuals = True
        use_subset_init = True
        use_multi_start = True
        rationale_parts.append(
            f"Very wide range ({decades:.1f} decades): Using log-residuals, "
            f"subset initialization, and multi-start optimization for robustness."
        )
        logger.debug(
            "Applied very wide range strategy",
            decades=decades,
            rule="rule_1_very_wide",
        )

    # Rule 2: Wide range (8-10 decades)
    elif decades > 8:
        use_log_residuals = True
        use_multi_start = True
        rationale_parts.append(
            f"Wide range ({decades:.1f} decades): Using log-residuals and "
            f"multi-start optimization."
        )
        logger.debug(
            "Applied wide range strategy",
            decades=decades,
            rule="rule_2_wide",
        )

    # Rule 3: Moderate range (5-8 decades)
    elif decades > 5:
        use_log_residuals = True
        rationale_parts.append(
            f"Moderate range ({decades:.1f} decades): Using log-residuals "
            f"to balance frequency regions."
        )
        logger.debug(
            "Applied moderate range strategy",
            decades=decades,
            rule="rule_3_moderate",
        )

    # Rule 4: Oscillation mode with complex data
    if test_mode == "oscillation" and np.iscomplexobj(y):
        if decades > 6 and not use_log_residuals:
            use_log_residuals = True
            rationale_parts.append(
                "Oscillation mode with complex modulus: Using log-residuals."
            )
            logger.debug(
                "Applied oscillation mode strategy",
                decades=decades,
                rule="rule_4_oscillation_complex",
            )

    # Default case
    if not rationale_parts:
        rationale_parts.append(
            f"Standard range ({decades:.1f} decades): Using default linear residuals."
        )
        logger.debug(
            "Applied standard strategy",
            decades=decades,
            rule="default",
        )

    strategy: dict[str, bool | str | float] = {
        "use_log_residuals": use_log_residuals,
        "use_multi_start": use_multi_start,
        "use_subset_init": use_subset_init,
        "decades": decades,
        "rationale": " ".join(rationale_parts),
    }

    logger.info(
        "Optimization strategy suggested",
        decades=decades,
        use_log_residuals=use_log_residuals,
        use_multi_start=use_multi_start,
        use_subset_init=use_subset_init,
    )

    return strategy


__all__ = [
    "detect_data_range_decades",
    "check_wide_frequency_range",
    "suggest_optimization_strategy",
]
