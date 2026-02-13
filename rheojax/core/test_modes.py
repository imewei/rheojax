"""Test mode detection for rheological data.

This module provides automatic detection of rheological test modes based on
data characteristics, units, and metadata.
"""

from __future__ import annotations

import warnings
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from rheojax.core.data import RheoData


class DeformationMode(StrEnum):
    """Deformation geometry mode for rheological measurements.

    This enum classifies the type of mechanical deformation applied during
    a rheological measurement. Shear-based instruments (rotational rheometers)
    measure G*(w), while tensile/bending/compression instruments (DMTA/DMA)
    measure E*(w). The relationship is:

        E*(w) = 2(1 + v) * G*(w)

    where v is Poisson's ratio of the material.
    """

    SHEAR = "shear"
    TENSION = "tension"
    BENDING = "bending"
    COMPRESSION = "compression"

    def __str__(self) -> str:
        return self.value

    def is_tensile(self) -> bool:
        """True if this deformation mode produces Young's modulus E*.

        Tension, bending, and compression geometries all measure E*,
        while shear measures G*.
        """
        return self in (
            DeformationMode.TENSION,
            DeformationMode.BENDING,
            DeformationMode.COMPRESSION,
        )


class TestModeEnum(StrEnum):
    """Enumeration of rheological test modes.

    Note: Named TestModeEnum (not TestMode) to avoid pytest collection warnings.
    Pytest treats classes starting with 'Test' and ending without 'Enum' as test classes.

    Note on EPM/Flow protocols:
        - FLOW_CURVE: Steady-state stress vs shear rate (same physics as ROTATION)
        - STARTUP: Transient stress evolution at constant shear rate
        - ROTATION: Generic rotational/steady shear mode (legacy)
    """

    RELAXATION = "relaxation"
    CREEP = "creep"
    OSCILLATION = "oscillation"
    LAOS = "laos"  # Large Amplitude Oscillatory Shear
    ROTATION = "rotation"
    FLOW_CURVE = "flow_curve"  # Steady-state flow curve protocol
    STARTUP = "startup"  # Startup shear protocol
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return string representation."""
        return self.value

    @classmethod
    def from_protocol(cls, protocol: Protocol) -> TestModeEnum:
        """Convert inventory Protocol to TestModeEnum."""
        if protocol == Protocol.FLOW_CURVE:
            return cls.FLOW_CURVE
        elif protocol == Protocol.CREEP:
            return cls.CREEP
        elif protocol == Protocol.RELAXATION:
            return cls.RELAXATION
        elif protocol == Protocol.STARTUP:
            return cls.STARTUP
        elif protocol == Protocol.OSCILLATION:
            return cls.OSCILLATION
        elif protocol == Protocol.SAOS:  # type: ignore[attr-defined]
            return cls.OSCILLATION
        elif protocol == Protocol.LAOS:
            return cls.OSCILLATION  # LAOS is a type of oscillation
        return cls.UNKNOWN  # type: ignore[unreachable]

    def to_protocol(self) -> Protocol | None:
        """Convert TestModeEnum to inventory Protocol (best effort)."""
        if self == self.ROTATION:
            return Protocol.FLOW_CURVE
        elif self == self.FLOW_CURVE:
            return Protocol.FLOW_CURVE
        elif self == self.STARTUP:
            return Protocol.STARTUP
        elif self == self.CREEP:
            return Protocol.CREEP
        elif self == self.RELAXATION:
            return Protocol.RELAXATION
        elif self == self.OSCILLATION:
            return Protocol.OSCILLATION
        elif self == self.LAOS:
            return Protocol.LAOS
        return None


# Backward compatibility aliases
RheoTestMode = TestModeEnum  # Transition name
TestMode = TestModeEnum  # Original name (deprecated)


def is_monotonic_increasing(
    data: np.ndarray | jnp.ndarray,  # type: ignore[name-defined]
    strict: bool = False,
    tolerance: float = 1e-10,
    allow_fraction: float = 0.1,
) -> bool:
    """Check if data is mostly monotonically increasing.

    Args:
        data: Array to check
        strict: If True, require strictly increasing (no equal values)
        tolerance: Relative tolerance based on data magnitude
        allow_fraction: Fraction of points allowed to violate monotonicity (0-1)

    Returns:
        True if data is mostly monotonically increasing
    """
    # Convert to numpy for easier checking
    if isinstance(data, jnp.ndarray):
        data = np.array(data)

    if len(data) < 2:
        return True

    # Check overall trend first
    overall_trend = data[-1] - data[0]
    if overall_trend < 0:
        return False

    # Calculate tolerance based on data magnitude
    data_mag = np.mean(np.abs(data))
    rel_tolerance = tolerance * data_mag

    diffs = np.diff(data)

    if strict:
        violations = np.sum(diffs <= rel_tolerance)
    else:
        violations = np.sum(diffs < -rel_tolerance)

    # Allow a small fraction of violations for noisy data
    max_violations = int(allow_fraction * len(diffs))
    return bool(violations <= max_violations)


def is_monotonic_decreasing(
    data: np.ndarray | jnp.ndarray,  # type: ignore[name-defined]
    strict: bool = False,
    tolerance: float = 1e-10,
    allow_fraction: float = 0.1,
) -> bool:
    """Check if data is mostly monotonically decreasing.

    Args:
        data: Array to check
        strict: If True, require strictly decreasing (no equal values)
        tolerance: Relative tolerance based on data magnitude
        allow_fraction: Fraction of points allowed to violate monotonicity (0-1)

    Returns:
        True if data is mostly monotonically decreasing
    """
    # Convert to numpy for easier checking
    if isinstance(data, jnp.ndarray):
        data = np.array(data)

    if len(data) < 2:
        return True

    # Check overall trend first
    overall_trend = data[-1] - data[0]
    if overall_trend > 0:
        return False

    # Calculate tolerance based on data magnitude
    data_mag = np.mean(np.abs(data))
    rel_tolerance = tolerance * data_mag

    diffs = np.diff(data)

    if strict:
        violations = np.sum(diffs >= -rel_tolerance)
    else:
        violations = np.sum(diffs > rel_tolerance)

    # Allow a small fraction of violations for noisy data
    max_violations = int(allow_fraction * len(diffs))
    return bool(violations <= max_violations)


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
                UserWarning,
                stacklevel=2,
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

        # Check if data is essentially flat (no significant change)
        # This handles elastic solids that have fully relaxed to equilibrium
        overall_change = abs(y_data[-1] - y_data[0])
        data_magnitude = np.mean(np.abs(y_data))
        relative_change = overall_change / data_magnitude if data_magnitude > 0 else 0

        # If change < 5% of magnitude, consider it flat
        # Flat time-domain data is more likely relaxation (reached equilibrium) than creep
        if relative_change < 0.05:
            # Default to relaxation for flat data in time domain
            return TestModeEnum.RELAXATION

        # Check for monotonic behavior
        # Use relative tolerance of 1% of data magnitude
        # Allow up to 30% of points to violate monotonicity (for noisy data that plateaus)
        tolerance = 0.01  # 1% of data magnitude
        allow_fraction = 0.3  # Allow 30% violations

        # For relaxation: stress should decrease over time
        if is_monotonic_decreasing(
            y_data, strict=False, tolerance=tolerance, allow_fraction=allow_fraction
        ):
            return TestModeEnum.RELAXATION

        # For creep: strain/compliance should increase over time
        if is_monotonic_increasing(
            y_data, strict=False, tolerance=tolerance, allow_fraction=allow_fraction
        ):
            return TestModeEnum.CREEP

    # 4. Fall back to UNKNOWN if we can't determine
    warnings.warn(
        "Could not automatically detect test mode. "
        "Consider setting test_mode explicitly in metadata.",
        UserWarning,
        stacklevel=2,
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

    Queries the ModelRegistry to determine which test modes are supported
    by the specified model, using the Protocol-Driven Inventory System.

    Args:
        model_name: Name of the rheological model

    Returns:
        List of compatible TestMode values
    """
    from rheojax.core.registry import ModelRegistry

    # Try to get model info from registry
    info = ModelRegistry.get_info(model_name)

    if info and info.protocols:
        # Convert Protocols to TestModes
        modes = []
        for p in info.protocols:
            tm = TestModeEnum.from_protocol(p)
            if tm != TestModeEnum.UNKNOWN:
                modes.append(tm)
        # Deduplicate while preserving order
        return list(dict.fromkeys(modes))

    # Fallback to legacy detection if no protocols registered
    try:
        if info is None:
            # Unknown model, return common modes
            return [TestMode.RELAXATION, TestMode.CREEP, TestMode.OSCILLATION]
        model_cls = info.plugin_class
    except (KeyError, ValueError):
        return [TestMode.RELAXATION, TestMode.CREEP, TestMode.OSCILLATION]

    # Check for supported_test_modes attribute (legacy)
    if hasattr(model_cls, "supported_test_modes"):
        modes = model_cls.supported_test_modes
        return [TestMode(m) if isinstance(m, str) else m for m in modes]

    # Check for _fit method and infer from docstring or signature (legacy)
    if hasattr(model_cls, "_fit"):
        modes = [TestMode.OSCILLATION, TestMode.RELAXATION]
        if hasattr(model_cls, "_fit_creep_mode") or hasattr(
            model_cls, "_predict_creep"
        ):
            modes.append(TestMode.CREEP)
        if hasattr(model_cls, "_fit_steady_shear_mode") or hasattr(
            model_cls, "_fit_rotation_mode"
        ):
            modes.append(TestMode.ROTATION)
        return modes  # type: ignore[return-value]

    return [TestMode.RELAXATION, TestMode.CREEP, TestMode.OSCILLATION]


def suggest_models_for_test_mode(test_mode: TestMode) -> list[str]:
    """Suggest appropriate models for a given test mode.

    Queries the ModelRegistry to find models compatible with the specified
    test mode using the Protocol-Driven Inventory System.

    Args:
        test_mode: Detected test mode

    Returns:
        List of recommended model names
    """
    from rheojax.core.registry import ModelRegistry

    # Convert TestMode to Protocol
    if isinstance(test_mode, str):
        try:
            test_mode = TestMode(test_mode.lower())
        except ValueError:
            return []

    protocol = test_mode.to_protocol()

    if protocol:
        # Use new inventory system
        compatible = ModelRegistry.find(protocol=protocol)
    else:
        # Fallback for UNKNOWN or unsupported modes
        return []

    # If we found compatible models, return them sorted by priority
    if compatible:
        # Prioritize common models
        priority = [
            "maxwell",
            "zener",
            "sgr_conventional",
            "herschel_bulkley",
            "power_law",
            "generalized_maxwell",
        ]
        sorted_models = sorted(
            compatible,
            key=lambda m: priority.index(m.lower()) if m.lower() in priority else 100,
        )
        return sorted_models

    return []
