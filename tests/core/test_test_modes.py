"""Tests for test mode detection functionality."""

import numpy as np
import pytest

from rheo.core.data import RheoData
from rheo.core.test_modes import TestMode, detect_test_mode



from rheo.core.jax_config import safe_import_jax

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
class TestModeDetection:
    """Test automatic test mode detection."""

    def test_relaxation_detection_monotonic_decrease(self):
        """Test relaxation detection with monotonic decreasing stress."""
        # Time-domain data with monotonic decreasing stress (relaxation)
        time = np.linspace(0, 100, 100)
        stress = 1000 * np.exp(-time / 10)  # Exponential decay

        data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

        mode = detect_test_mode(data)
        assert mode == TestMode.RELAXATION

    def test_creep_detection_monotonic_increase(self):
        """Test creep detection with monotonic increasing strain."""
        # Time-domain data with monotonic increasing strain (creep)
        time = np.linspace(0, 100, 100)
        strain = 0.01 * (1 - np.exp(-time / 20))  # Creep compliance curve

        data = RheoData(
            x=time, y=strain, x_units="s", y_units="unitless", domain="time"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.CREEP

    def test_oscillation_detection_frequency_domain(self):
        """Test oscillation detection with frequency-domain data."""
        # Frequency-domain data (oscillation/SAOS)
        freq = np.logspace(-2, 2, 50)  # 0.01 to 100 rad/s
        G_prime = 1000 * freq**0.5  # Storage modulus

        data = RheoData(
            x=freq, y=G_prime, x_units="rad/s", y_units="Pa", domain="frequency"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.OSCILLATION

    def test_oscillation_detection_hz_units(self):
        """Test oscillation detection with Hz units."""
        # Frequency-domain data with Hz units
        freq = np.logspace(-2, 2, 50)
        G_double_prime = 500 * freq**0.8  # Loss modulus

        data = RheoData(
            x=freq, y=G_double_prime, x_units="Hz", y_units="Pa", domain="frequency"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.OSCILLATION

    def test_rotation_detection_shear_rate(self):
        """Test rotation (steady shear) detection with shear rate."""
        # Steady shear data (rotation)
        shear_rate = np.logspace(-2, 3, 50)  # 0.01 to 1000 1/s
        viscosity = 100 * shear_rate ** (-0.3)  # Shear-thinning

        data = RheoData(
            x=shear_rate,
            y=viscosity,
            x_units="1/s",
            y_units="Pa.s",
            domain="time",  # Steady shear is typically time-domain measurement
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.ROTATION

    def test_rotation_detection_alternative_units(self):
        """Test rotation detection with alternative shear rate units."""
        shear_rate = np.logspace(-1, 2, 30)
        stress = 50 + 10 * shear_rate**0.5  # Herschel-Bulkley-like

        data = RheoData(
            x=shear_rate, y=stress, x_units="s^-1", y_units="Pa", domain="time"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.ROTATION

    def test_explicit_metadata_override(self):
        """Test explicit test mode specification in metadata."""
        # Data that might look like relaxation but explicitly marked as oscillation
        time = np.linspace(0, 100, 100)
        stress = 1000 * np.exp(-time / 10)

        data = RheoData(
            x=time,
            y=stress,
            x_units="s",
            y_units="Pa",
            domain="time",
            metadata={"test_mode": "oscillation"},
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.OSCILLATION

    def test_ambiguous_case_fallback(self):
        """Test ambiguous case returns unknown."""
        # Time-domain data with non-monotonic behavior (ambiguous)
        time = np.linspace(0, 100, 100)
        data_oscillating = 1000 * np.sin(time / 10)  # Oscillating in time

        data = RheoData(
            x=time, y=data_oscillating, x_units="s", y_units="Pa", domain="time"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.UNKNOWN

    def test_complex_oscillation_data(self):
        """Test detection with complex modulus data."""
        # Complex modulus (G* = G' + iG'')
        freq = np.logspace(-1, 2, 40)
        G_star = 1000 * (1 + 0.5j) * freq**0.5

        data = RheoData(
            x=freq, y=G_star, x_units="rad/s", y_units="Pa", domain="frequency"
        )

        mode = detect_test_mode(data)
        assert mode == TestMode.OSCILLATION

    def test_rheodata_test_mode_property(self):
        """Test that RheoData.test_mode property works correctly."""
        # Create relaxation data
        time = np.linspace(0, 100, 100)
        stress = 1000 * np.exp(-time / 10)

        data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

        # Access test_mode property
        mode = data.test_mode
        assert mode == TestMode.RELAXATION

        # Verify it's cached in metadata
        assert "detected_test_mode" in data.metadata
        assert data.metadata["detected_test_mode"] == TestMode.RELAXATION

    def test_jax_arrays_work(self):
        """Test that detection works with JAX arrays."""
        # Use JAX arrays
        time = jnp.linspace(0, 100, 100)
        stress = 1000 * jnp.exp(-time / 10)

        data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

        mode = detect_test_mode(data)
        assert mode == TestMode.RELAXATION


class TestMonotonicityChecks:
    """Test monotonicity checking utilities."""

    def test_monotonic_increasing(self):
        """Test detection of monotonic increasing."""
        from rheo.core.test_modes import is_monotonic_increasing

        data = np.array([1, 2, 3, 4, 5])
        assert is_monotonic_increasing(data)

        data = np.array([1, 1, 2, 3])  # Non-strict increasing
        assert is_monotonic_increasing(data, strict=False)
        assert not is_monotonic_increasing(data, strict=True)

    def test_monotonic_decreasing(self):
        """Test detection of monotonic decreasing."""
        from rheo.core.test_modes import is_monotonic_decreasing

        data = np.array([5, 4, 3, 2, 1])
        assert is_monotonic_decreasing(data)

        data = np.array([5, 5, 4, 3])  # Non-strict decreasing
        assert is_monotonic_decreasing(data, strict=False)
        assert not is_monotonic_decreasing(data, strict=True)

    def test_monotonic_with_noise(self):
        """Test monotonicity with small noise (tolerance)."""
        from rheo.core.test_modes import is_monotonic_decreasing

        # Mostly decreasing with small noise
        data = np.array([100, 95, 91, 90.5, 85, 80])
        assert is_monotonic_decreasing(data, strict=False)

        # Too much noise
        data = np.array([100, 50, 90, 40, 80])
        assert not is_monotonic_decreasing(data)


class TestModeEnum:
    """Test TestMode enumeration."""

    def test_test_mode_values(self):
        """Test TestMode enum values."""
        assert TestMode.RELAXATION == "relaxation"
        assert TestMode.CREEP == "creep"
        assert TestMode.OSCILLATION == "oscillation"
        assert TestMode.ROTATION == "rotation"
        assert TestMode.UNKNOWN == "unknown"

    def test_test_mode_from_string(self):
        """Test converting string to TestMode."""
        assert TestMode("relaxation") == TestMode.RELAXATION
        assert TestMode("oscillation") == TestMode.OSCILLATION


class TestValidationDataset:
    """Test detection accuracy on comprehensive validation dataset."""

    def test_detection_accuracy_target(self):
        """Test that detection achieves >95% accuracy on validation dataset."""
        from rheo.core.test_modes import TestMode, detect_test_mode

        # Create comprehensive validation dataset
        test_cases = []

        # Relaxation cases (20 variants)
        for i in range(20):
            time = np.linspace(0, 100, 100)
            # Vary decay rates and initial values
            tau = np.random.uniform(5, 50)
            initial = np.random.uniform(100, 10000)
            stress = initial * np.exp(-time / tau)

            test_cases.append(
                {
                    "data": RheoData(
                        x=time, y=stress, x_units="s", y_units="Pa", domain="time"
                    ),
                    "expected": TestMode.RELAXATION,
                    "description": f"Relaxation case {i+1}",
                }
            )

        # Creep cases (20 variants)
        for i in range(20):
            time = np.linspace(0, 100, 100)
            # Vary compliance parameters
            J0 = np.random.uniform(0.0001, 0.01)
            tau = np.random.uniform(10, 50)
            strain = J0 * (1 - np.exp(-time / tau))

            test_cases.append(
                {
                    "data": RheoData(
                        x=time, y=strain, x_units="s", y_units="unitless", domain="time"
                    ),
                    "expected": TestMode.CREEP,
                    "description": f"Creep case {i+1}",
                }
            )

        # Oscillation cases (30 variants)
        for i in range(30):
            # Vary frequency range and units
            if i < 15:
                freq = np.logspace(-2, 2, 50)
                x_units = "rad/s"
            else:
                freq = np.logspace(-1, 1, 50)
                x_units = "Hz"

            # Vary modulus scaling
            alpha = np.random.uniform(0.3, 0.8)
            G0 = np.random.uniform(100, 10000)
            modulus = G0 * freq**alpha

            test_cases.append(
                {
                    "data": RheoData(
                        x=freq,
                        y=modulus,
                        x_units=x_units,
                        y_units="Pa",
                        domain="frequency",
                    ),
                    "expected": TestMode.OSCILLATION,
                    "description": f"Oscillation case {i+1}",
                }
            )

        # Rotation cases (20 variants)
        for i in range(20):
            shear_rate = np.logspace(-2, 3, 50)
            # Vary flow behavior
            if i < 10:
                x_units = "1/s"
            else:
                x_units = "s^-1"

            # Power law viscosity
            n = np.random.uniform(0.2, 0.9)
            K = np.random.uniform(10, 1000)
            viscosity = K * shear_rate ** (n - 1)

            test_cases.append(
                {
                    "data": RheoData(
                        x=shear_rate,
                        y=viscosity,
                        x_units=x_units,
                        y_units="Pa.s",
                        domain="time",
                    ),
                    "expected": TestMode.ROTATION,
                    "description": f"Rotation case {i+1}",
                }
            )

        # Complex modulus cases (10 variants)
        for i in range(10):
            freq = np.logspace(-1, 2, 40)
            G_prime = 1000 * freq**0.5
            G_double_prime = 500 * freq**0.7
            G_star = G_prime + 1j * G_double_prime

            test_cases.append(
                {
                    "data": RheoData(
                        x=freq,
                        y=G_star,
                        x_units="rad/s",
                        y_units="Pa",
                        domain="frequency",
                    ),
                    "expected": TestMode.OSCILLATION,
                    "description": f"Complex modulus case {i+1}",
                }
            )

        # Run detection on all cases
        correct = 0
        total = len(test_cases)
        failed_cases = []

        for case in test_cases:
            detected = detect_test_mode(case["data"])
            if detected == case["expected"]:
                correct += 1
            else:
                failed_cases.append(
                    {
                        "description": case["description"],
                        "expected": case["expected"],
                        "detected": detected,
                    }
                )

        accuracy = (correct / total) * 100

        # Print results
        print(f"\n{'='*70}")
        print(f"Test Mode Detection Validation Results")
        print(f"{'='*70}")
        print(f"Total test cases: {total}")
        print(f"Correct detections: {correct}")
        print(f"Failed detections: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Target: >95%")
        print(f"{'='*70}")

        if failed_cases:
            print(f"\nFailed cases:")
            for case in failed_cases:
                print(f"  - {case['description']}")
                print(f"    Expected: {case['expected']}, Detected: {case['detected']}")

        # Assert >95% accuracy
        assert (
            accuracy > 95.0
        ), f"Detection accuracy {accuracy:.2f}% is below target of 95%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
