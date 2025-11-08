"""Constants for fractional model initialization.

This module centralizes all magic numbers and configuration constants used in
the initialization system, improving maintainability and testability.

Phase 2 of Template Method Refactoring: Constants Extraction
"""

from typing import NamedTuple


class InitializationConstants(NamedTuple):
    """Configuration constants for parameter initialization."""

    # Feature extraction parameters
    SAVGOL_WINDOW: int = 5
    """Savitzky-Golay filter window length."""

    SAVGOL_POLY: int = 2
    """Savitzky-Golay filter polynomial order."""

    PLATEAU_PERCENTILE: float = 0.1
    """Fraction of data points used for plateau averaging (10%)."""

    # Validation thresholds
    MIN_FREQUENCY_DECADES: float = 1.5
    """Minimum frequency range in decades for valid initialization."""

    MIN_PLATEAU_RATIO: float = 1.1
    """Minimum ratio between high and low plateaus (10% difference)."""

    # Parameter bounds
    MIN_ALPHA: float = 0.01
    """Minimum fractional order (prevents zero)."""

    MAX_ALPHA: float = 0.99
    """Maximum fractional order (prevents unity)."""

    EPSILON: float = 1e-12
    """Small value to prevent division by zero and log(0)."""

    # Default parameter values (fallback when initialization fails)
    DEFAULT_MODULUS: float = 1000.0
    """Default equilibrium/glassy modulus in Pa."""

    DEFAULT_COMPLIANCE: float = 1e-6
    """Default compliance in 1/Pa."""

    DEFAULT_ALPHA: float = 0.5
    """Default fractional order (mid-range)."""

    DEFAULT_TAU: float = 1.0
    """Default characteristic time in seconds."""


# Singleton instance
INIT_CONSTANTS = InitializationConstants()


# Feature extraction configuration
class FeatureExtractionConfig(NamedTuple):
    """Configuration for frequency-domain feature extraction."""

    savgol_window: int = INIT_CONSTANTS.SAVGOL_WINDOW
    savgol_poly: int = INIT_CONSTANTS.SAVGOL_POLY
    plateau_percentile: float = INIT_CONSTANTS.PLATEAU_PERCENTILE
    min_frequency_decades: float = INIT_CONSTANTS.MIN_FREQUENCY_DECADES
    min_plateau_ratio: float = INIT_CONSTANTS.MIN_PLATEAU_RATIO
    epsilon: float = INIT_CONSTANTS.EPSILON


# Parameter bounds configuration
class ParameterBounds(NamedTuple):
    """Standard parameter bounds for initialization."""

    min_alpha: float = INIT_CONSTANTS.MIN_ALPHA
    max_alpha: float = INIT_CONSTANTS.MAX_ALPHA


# Default values configuration
class DefaultParameters(NamedTuple):
    """Default parameter values when initialization fails."""

    modulus: float = INIT_CONSTANTS.DEFAULT_MODULUS
    compliance: float = INIT_CONSTANTS.DEFAULT_COMPLIANCE
    alpha: float = INIT_CONSTANTS.DEFAULT_ALPHA
    tau: float = INIT_CONSTANTS.DEFAULT_TAU


# Exported configurations
FEATURE_CONFIG = FeatureExtractionConfig()
PARAM_BOUNDS = ParameterBounds()
DEFAULT_PARAMS = DefaultParameters()


__all__ = [
    "InitializationConstants",
    "INIT_CONSTANTS",
    "FeatureExtractionConfig",
    "FEATURE_CONFIG",
    "ParameterBounds",
    "PARAM_BOUNDS",
    "DefaultParameters",
    "DEFAULT_PARAMS",
]
