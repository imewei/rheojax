"""Tests for test mode detection functionality."""

import warnings

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode, detect_test_mode

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

    def test_rotation_detection_shear_rate_warns_ambiguous_units(self):
        """'1/s'/'s^-1' is dimensionally identical to Hz, so classifying it as
        ROTATION (not OSCILLATION) must not be silent.

        Regression test for R-TESTMODE-002: frequency-domain data labeled with
        SI shear-rate-style units was silently misclassified as ROTATION with
        zero warnings, even though '1/s' == Hz and could equally be a
        frequency sweep left at the default domain='time'.
        """
        shear_rate = np.logspace(-2, 3, 50)
        viscosity = 100 * shear_rate ** (-0.3)

        data = RheoData(
            x=shear_rate,
            y=viscosity,
            x_units="1/s",
            y_units="Pa.s",
            domain="time",
        )

        with pytest.warns(UserWarning, match="ambiguous between shear rate"):
            mode = detect_test_mode(data)

        # Classification itself must be unchanged (locked in by
        # test_rotation_detection_shear_rate above).
        assert mode == TestMode.ROTATION

    def test_oscillation_detection_frequency_domain_no_ambiguous_warning(self):
        """Explicit domain='frequency' with '1/s' units must resolve cleanly
        to OSCILLATION without the ambiguous-units warning firing."""
        freq = np.logspace(-2, 2, 50)
        G_prime = 1000 * freq**0.5

        data = RheoData(
            x=freq, y=G_prime, x_units="1/s", y_units="Pa", domain="frequency"
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mode = detect_test_mode(data)

        assert mode == TestMode.OSCILLATION

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

    def test_time_domain_oscillation_integer_cycles_not_flat(self):
        """Time-domain oscillatory data spanning whole cycles must not be
        misclassified as RELAXATION just because endpoints coincide.

        Regression test for R10-TESTMODE-001-B: the old flatness heuristic
        compared only y[0] and y[-1], so a LAOS-style waveform recorded over
        an integer number of periods (endpoints ~equal, large interior
        amplitude) was silently defaulted to 'relaxation'.
        """
        time = np.linspace(0, 4 * np.pi * 10, 200)  # exactly 2 periods of sin(t/10)
        stress = 1000 * np.sin(time / 10)

        data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

        mode = detect_test_mode(data)
        assert mode != TestMode.RELAXATION

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

    def test_nan_in_y_data_does_not_misclassify_as_flat_relaxation(self):
        """NaN-poisoned y-data must not be silently reported as flat RELAXATION.

        Regression test: np.max/np.min/np.mean all propagate NaN, making
        `data_magnitude > 0` False (NaN comparisons are always False), which
        used to make relative_change fall back to the literal 0 and trigger
        the flat-data heuristic. detect_test_mode() must instead return
        UNKNOWN with a warning that names NaN/Inf explicitly, not "flat".
        """
        time = np.linspace(0, 100, 100)
        stress = 1000 * np.exp(-time / 10)
        stress[10] = np.nan

        data = RheoData(
            x=time, y=stress, x_units="s", y_units="Pa", domain="time", validate=False
        )

        with pytest.warns(UserWarning, match="NaN or Inf"):
            mode = detect_test_mode(data)

        assert mode == TestMode.UNKNOWN

    def test_inf_in_y_data_does_not_misclassify_as_flat_relaxation(self):
        """Same as above, for +/-Inf rather than NaN."""
        time = np.linspace(0, 100, 100)
        stress = 1000 * np.exp(-time / 10)
        stress[-1] = np.inf

        data = RheoData(
            x=time, y=stress, x_units="s", y_units="Pa", domain="time", validate=False
        )

        with pytest.warns(UserWarning, match="NaN or Inf"):
            mode = detect_test_mode(data)

        assert mode == TestMode.UNKNOWN


class TestMonotonicityChecks:
    """Test monotonicity checking utilities."""

    def test_monotonic_increasing(self):
        """Test detection of monotonic increasing."""
        from rheojax.core.test_modes import is_monotonic_increasing

        data = np.array([1, 2, 3, 4, 5])
        assert is_monotonic_increasing(data)

        data = np.array([1, 1, 2, 3])  # Non-strict increasing
        assert is_monotonic_increasing(data, strict=False)
        assert not is_monotonic_increasing(data, strict=True)

    def test_monotonic_decreasing(self):
        """Test detection of monotonic decreasing."""
        from rheojax.core.test_modes import is_monotonic_decreasing

        data = np.array([5, 4, 3, 2, 1])
        assert is_monotonic_decreasing(data)

        data = np.array([5, 5, 4, 3])  # Non-strict decreasing
        assert is_monotonic_decreasing(data, strict=False)
        assert not is_monotonic_decreasing(data, strict=True)

    def test_monotonic_with_noise(self):
        """Test monotonicity with small noise (tolerance)."""
        from rheojax.core.test_modes import is_monotonic_decreasing

        # Mostly decreasing with small noise
        data = np.array([100, 95, 91, 90.5, 85, 80])
        assert is_monotonic_decreasing(data, strict=False)

        # Too much noise
        data = np.array([100, 50, 90, 40, 80])
        assert not is_monotonic_decreasing(data)

    def test_monotonic_increasing_with_noisy_endpoint(self):
        """A single noisy last sample must not defeat allow_fraction.

        Regression test: is_monotonic_increasing used to hard-fail on
        `data[-1] - data[0] < 0` before the tolerant diff-based violation
        count ever ran, so one bad endpoint could flip the whole result even
        though only 1/49 diffs (well under the 30% allow_fraction) actually
        violate monotonicity.
        """
        from rheojax.core.test_modes import is_monotonic_increasing

        data = np.concatenate([np.linspace(100, 200, 49), [50.0]])
        assert is_monotonic_increasing(data, allow_fraction=0.3)

    def test_monotonic_decreasing_with_noisy_endpoint(self):
        """Mirror of the increasing case for is_monotonic_decreasing."""
        from rheojax.core.test_modes import is_monotonic_decreasing

        data = np.concatenate([np.linspace(200, 100, 49), [250.0]])
        assert is_monotonic_decreasing(data, allow_fraction=0.3)


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
        from rheojax.core.test_modes import TestMode, detect_test_mode

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
                    "description": f"Relaxation case {i + 1}",
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
                    "description": f"Creep case {i + 1}",
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
                    "description": f"Oscillation case {i + 1}",
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
                    "description": f"Rotation case {i + 1}",
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
                    "description": f"Complex modulus case {i + 1}",
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
        print(f"\n{'=' * 70}")
        print(f"Test Mode Detection Validation Results")
        print(f"{'=' * 70}")
        print(f"Total test cases: {total}")
        print(f"Correct detections: {correct}")
        print(f"Failed detections: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Target: >95%")
        print(f"{'=' * 70}")

        if failed_cases:
            print(f"\nFailed cases:")
            for case in failed_cases:
                print(f"  - {case['description']}")
                print(f"    Expected: {case['expected']}, Detected: {case['detected']}")

        # Assert >95% accuracy
        assert accuracy > 95.0, (
            f"Detection accuracy {accuracy:.2f}% is below target of 95%"
        )


class TestModelSuggestionFunctions:
    """Test get_compatible_test_modes and suggest_models_for_test_mode.

    Regression coverage: these are documented public API
    (docs/source/api/core.rst) with zero prior test coverage anywhere in
    the repo.
    """

    def test_get_compatible_test_modes_registered_model_with_protocols(self):
        """A registered model with protocols should return protocol-derived modes."""
        from rheojax.core.test_modes import get_compatible_test_modes

        modes = get_compatible_test_modes("maxwell")
        assert TestMode.RELAXATION in modes
        assert TestMode.CREEP in modes
        assert TestMode.OSCILLATION in modes

    def test_get_compatible_test_modes_unregistered_model_default(self):
        """An unknown model name should fall back to the 3-mode default."""
        from rheojax.core.test_modes import get_compatible_test_modes

        modes = get_compatible_test_modes("nonexistent_model_xyz")
        assert modes == [TestMode.RELAXATION, TestMode.CREEP, TestMode.OSCILLATION]

    def test_suggest_models_for_test_mode_relaxation_priority_ordering(self):
        """suggest_models_for_test_mode should priority-sort, maxwell first."""
        from rheojax.core.test_modes import suggest_models_for_test_mode

        models = suggest_models_for_test_mode(TestMode.RELAXATION)
        assert len(models) > 0
        assert models[0].lower() == "maxwell"

    def test_suggest_models_for_test_mode_unknown_returns_empty_list(self):
        """TestMode.UNKNOWN has no Protocol mapping, so it must return []."""
        from rheojax.core.test_modes import suggest_models_for_test_mode

        assert suggest_models_for_test_mode(TestMode.UNKNOWN) == []

    def test_get_compatible_test_modes_falls_back_for_model_without_protocols(self):
        """Models registered without protocols (a still-supported, optional
        kwarg on ModelRegistry.register) must still get a sane default list
        via the legacy fallback branch, not an empty list or an exception.
        """
        from rheojax.core.registry import ModelRegistry
        from rheojax.core.test_modes import get_compatible_test_modes

        @ModelRegistry.register(
            "ponytail_throwaway_no_protocols_model", protocols=None, validate=False
        )
        class _ThrowawayModel:
            pass

        try:
            modes = get_compatible_test_modes("ponytail_throwaway_no_protocols_model")
            assert modes == [
                TestMode.RELAXATION,
                TestMode.CREEP,
                TestMode.OSCILLATION,
            ]
        finally:
            ModelRegistry.unregister("ponytail_throwaway_no_protocols_model")

    def test_get_compatible_test_modes_warns_on_invalid_supported_test_modes(self):
        """A malformed supported_test_modes value must not be silently
        swallowed -- it should log a warning before falling back to the
        default (instead of a bare `except: return [...]`).
        """
        from rheojax.core.registry import ModelRegistry
        from rheojax.core.test_modes import get_compatible_test_modes

        @ModelRegistry.register(
            "ponytail_throwaway_bad_modes_model", protocols=None, validate=False
        )
        class _ThrowawayBadModesModel:
            supported_test_modes = ["not_a_real_test_mode"]

        try:
            modes = get_compatible_test_modes("ponytail_throwaway_bad_modes_model")
            assert modes == [
                TestMode.RELAXATION,
                TestMode.CREEP,
                TestMode.OSCILLATION,
            ]
        finally:
            ModelRegistry.unregister("ponytail_throwaway_bad_modes_model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
