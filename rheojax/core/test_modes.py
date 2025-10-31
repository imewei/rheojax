"""Test mode detection for rheological data.

This module provides automatic detection of rheological test modes based on
data characteristics, units, and metadata.
"""

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from rheo.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from rheo.core.data import RheoData


class TestModeEnum(str, Enum):
    """Enumeration of rheological test modes.

    Note: Named TestModeEnum (not TestMode) to avoid pytest collection warnings.
    Pytest treats classes starting with 'Test' and ending without 'Enum' as test classes.
    """

    RELAXATION = "relaxation"
    CREEP = "creep"
    OSCILLATION = "oscillation"
    ROTATION = "rotation"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value


# Backward compatibility aliases
RheoTestMode = TestModeEnum  # Transition name
TestMode = TestModeEnum  # Original name (deprecated)


def is_monotonic_increasing(
    data: np.ndarray | jnp.ndarray, strict: bool = False, tolerance: float = 1e-10
) -> bool:
    """Check if data is monotonically increasing.

    Args:
        data: Array to check
        strict: If True, require strictly increasing (no equal values)
        tolerance: Relative tolerance for considering values equal

    Returns:
        True if data is monotonically increasing
    """
    # Convert to numpy for easier checking
    if isinstance(data, jnp.ndarray):
        data = np.array(data)

    if len(data) < 2:
        return True

    diffs = np.diff(data)

    if strict:
        return np.all(diffs > tolerance)
    else:
        return np.all(diffs >= -tolerance)


def is_monotonic_decreasing(
    data: np.ndarray | jnp.ndarray, strict: bool = False, tolerance: float = 1e-10
) -> bool:
    """Check if data is monotonically decreasing.

    Args:
        data: Array to check
        strict: If True, require strictly decreasing (no equal values)
        tolerance: Relative tolerance for considering values equal

    Returns:
        True if data is monotonically decreasing
    """
    # Convert to numpy for easier checking
    if isinstance(data, jnp.ndarray):
        data = np.array(data)

    if len(data) < 2:
        return True

    diffs = np.diff(data)

    if strict:
        return np.all(diffs < -tolerance)
    else:
        return np.all(diffs <= tolerance)


def detect_test_mode(rheo_data: RheoData) -> TestModeEnum:
    """Detect rheological test mode from data characteristics.

    The detection algorithm uses the following heuristics:
    1. Check metadata['test_mode'] if explicitly provided
    2. Check domain and units:
       - frequency domain with rad/s or Hz → OSCILLATION
       - time domain with 1/s or s^-1 x-units → ROTATION
    3. Check monotonicity for time-domain data:
       - monotonic decreasing → RELAXATION
       - monotonic increasing → CREEP
    4. Fall back to UNKNOWN if ambiguous

    Args:
        rheo_data: RheoData object to analyze

    Returns:
        Detected TestMode

    Raises:
        ValueError: If rheo_data is invalid
    """
    if rheo_data is None or rheo_data.x is None or rheo_data.y is None:
        raise ValueError("Invalid RheoData: x and y must be provided")

    # 1. Check for explicit test_mode in metadata
    if "test_mode" in rheo_data.metadata:
        explicit_mode = rheo_data.metadata["test_mode"]
        try:
            return TestMode(
                explicit_mode.lower()
                if isinstance(explicit_mode, str)
                else explicit_mode
            )
        except (ValueError, AttributeError):
            warnings.warn(
                f"Invalid test_mode in metadata: {explicit_mode}. Attempting auto-detection.",
                UserWarning, stacklevel=2,
            )

    # 2. Check domain and units
    domain = rheo_data.domain
    x_units = rheo_data.x_units

    # Frequency domain → OSCILLATION (SAOS)
    if domain == "frequency":
        return TestModeEnum.OSCILLATION

    # Check x_units for frequency indicators
    if x_units is not None:
        x_units_lower = x_units.lower().strip()

        # Frequency units → OSCILLATION
        if any(unit in x_units_lower for unit in ["rad/s", "hz", "hertz"]):
            return TestModeEnum.OSCILLATION

        # Shear rate units → ROTATION (steady shear)
        if any(unit in x_units_lower for unit in ["1/s", "s^-1", "s-1", "/s"]):
            return TestModeEnum.ROTATION

    # 3. Time-domain analysis: check monotonicity
    if domain == "time" or domain is None:
        # Get y data (handle complex by taking real part if needed)
        y_data = rheo_data.y
        if isinstance(y_data, jnp.ndarray):
            y_data = np.array(y_data)

        if np.iscomplexobj(y_data):
            y_data = np.real(y_data)

        # Check for monotonic behavior
        # Use a tolerance to handle numerical noise
        tolerance = 1e-8

        # For relaxation: stress should decrease over time
        if is_monotonic_decreasing(y_data, strict=False, tolerance=tolerance):
            return TestModeEnum.RELAXATION

        # For creep: strain/compliance should increase over time
        if is_monotonic_increasing(y_data, strict=False, tolerance=tolerance):
            return TestModeEnum.CREEP

    # 4. Fall back to UNKNOWN if we can't determine
    warnings.warn(
        "Could not automatically detect test mode. "
        "Consider setting test_mode explicitly in metadata.",
        UserWarning, stacklevel=2,
    )
    return TestModeEnum.UNKNOWN


def validate_test_mode(test_mode: str | TestMode) -> TestMode:
    """Validate and convert test mode to TestMode enum.

    Args:
        test_mode: Test mode as string or TestMode enum

    Returns:
        TestMode enum

    Raises:
        ValueError: If test_mode is invalid
    """
    if isinstance(test_mode, TestMode):
        return test_mode

    try:
        return TestMode(test_mode.lower() if isinstance(test_mode, str) else test_mode)
    except (ValueError, AttributeError) as e:
        valid_modes = [mode.value for mode in TestMode]
        raise ValueError(
            f"Invalid test mode: {test_mode}. Valid modes are: {valid_modes}"
        ) from e


def get_compatible_test_modes(model_name: str) -> list[TestMode]:
    """Get compatible test modes for a given model.

    This is a placeholder for future model-test mode compatibility checking.

    Args:
        model_name: Name of the rheological model

    Returns:
        List of compatible TestMode values
    """
    # Placeholder implementation
    # In the future, this will query the model registry for compatibility info
    # For now, return common viscoelastic model modes
    return [TestMode.RELAXATION, TestMode.CREEP, TestMode.OSCILLATION]


def suggest_models_for_test_mode(test_mode: TestMode) -> list[str]:
    """Suggest appropriate models for a given test mode.

    This is a placeholder for future model recommendation system.

    Args:
        test_mode: Detected test mode

    Returns:
        List of recommended model names
    """
    # Placeholder implementation
    recommendations = {
        TestMode.RELAXATION: ["Maxwell", "Zener", "FractionalMaxwell"],
        TestMode.CREEP: ["Zener", "FractionalKelvinVoigt"],
        TestMode.OSCILLATION: ["Maxwell", "Zener", "SpringPot", "FractionalMaxwell"],
        TestMode.ROTATION: ["PowerLaw", "HerschelBulkley", "Carreau", "Cross"],
        TestMode.UNKNOWN: [],
    }
    return recommendations.get(test_mode, [])
