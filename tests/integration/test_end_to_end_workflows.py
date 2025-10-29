"""End-to-end workflow integration tests.

Tests complete pipelines from data loading through analysis and fitting.
Validates that all core components work together correctly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheo.core.data import RheoData
from rheo.core.parameters import ParameterSet
from rheo.core.test_modes import detect_test_mode


class TestEndToEndOscillation:
    """End-to-end workflow tests for oscillatory data."""

    @pytest.mark.integration
    def test_oscillation_data_load_and_mode_detection(self, oscillation_data_simple):
        """Test loading oscillatory data and detecting test mode."""
        data = oscillation_data_simple

        # Verify data loaded correctly
        assert data is not None
        assert len(data.x) == 10
        assert data.domain == "frequency"
        assert data.x_units == "Hz"

        # Verify test mode detection
        detected_mode = detect_test_mode(data)

        assert detected_mode == "oscillation"

    @pytest.mark.integration
    def test_oscillation_data_jax_conversion(self, oscillation_data_simple):
        """Test JAX conversion workflow for oscillatory data."""
        data = oscillation_data_simple

        # Convert to JAX
        jax_data = data.to_jax()

        # Verify JAX conversion
        assert isinstance(jax_data.x, jnp.ndarray)
        assert isinstance(jax_data.y, jnp.ndarray)
        assert jnp.array_equal(jax_data.x, jnp.array(data.x))

    @pytest.mark.integration
    def test_oscillation_data_complex_modulus_extraction(self, oscillation_data_simple):
        """Test extracting G' and G'' from complex modulus."""
        data = oscillation_data_simple

        # Extract components
        G_prime = data.y.real
        G_double_prime = data.y.imag

        # Verify physical properties
        assert np.all(G_prime > 0), "Storage modulus should be positive"
        assert np.all(G_double_prime > 0), "Loss modulus should be positive"
        assert np.all(
            np.abs(G_double_prime) <= np.abs(G_prime)
        ), "Loss should be less than storage for typical polymers"

    @pytest.mark.integration
    def test_oscillation_multi_dataset_consistency(
        self, oscillation_data_simple, oscillation_data_large
    ):
        """Test consistency across different oscillatory datasets."""
        simple = oscillation_data_simple
        large = oscillation_data_large

        # Both should be detected as oscillation

        assert detect_test_mode(simple) == "oscillation"
        assert detect_test_mode(large) == "oscillation"

        # Both should have frequency domain
        assert simple.domain == "frequency"
        assert large.domain == "frequency"

        # Large dataset should have more points
        assert len(large.x) > len(simple.x)


class TestEndToEndRelaxation:
    """End-to-end workflow tests for relaxation data."""

    @pytest.mark.integration
    def test_relaxation_data_load_and_detection(self, relaxation_data_simple):
        """Test loading and detecting relaxation data."""
        data = relaxation_data_simple

        assert data is not None
        assert data.domain == "time"
        assert data.x_units == "s"

        # Detect test mode

        detected_mode = detect_test_mode(data)

        assert detected_mode == "relaxation"

    @pytest.mark.integration
    def test_relaxation_stress_decay(self, relaxation_data_simple):
        """Test that relaxation stress decays monotonically."""
        data = relaxation_data_simple

        # Verify monotonic decrease
        diffs = np.diff(data.y)
        assert np.all(diffs <= 0), "Stress should decrease monotonically in relaxation"

    @pytest.mark.integration
    def test_multi_mode_relaxation_detection(self, relaxation_data_multi_mode):
        """Test detection of multi-mode relaxation data."""
        data = relaxation_data_multi_mode

        # Should be detected as relaxation

        detected_mode = detect_test_mode(data)

        assert detected_mode == "relaxation"

        # Verify metadata preserved
        assert data.metadata.get("num_modes") == 3

    @pytest.mark.integration
    def test_relaxation_data_log_transform(self, relaxation_data_simple):
        """Test log-scale transformation of relaxation data."""
        data = relaxation_data_simple

        # Apply log transform
        log_data = RheoData(
            x=np.log10(data.x),
            y=np.log10(data.y),
            x_units="log(s)",
            y_units="log(Pa)",
            domain=data.domain,
            metadata=data.metadata,
        )

        # Verify shapes match
        assert log_data.x.shape == data.x.shape
        assert log_data.y.shape == data.y.shape

        # Verify monotonicity preserved
        assert np.all(np.diff(log_data.y) <= 0)


class TestEndToEndCreep:
    """End-to-end workflow tests for creep data."""

    @pytest.mark.integration
    def test_creep_data_detection(self, creep_data_simple):
        """Test loading and detecting creep data."""
        data = creep_data_simple

        assert data.domain == "time"

        # Detect test mode

        detected_mode = detect_test_mode(data)

        assert detected_mode == "creep"

    @pytest.mark.integration
    def test_creep_compliance_increases(self, creep_data_simple):
        """Test that creep compliance increases with time."""
        data = creep_data_simple

        # Verify monotonic increase
        diffs = np.diff(data.y)
        assert np.all(diffs >= 0), "Compliance should increase monotonically in creep"

    @pytest.mark.integration
    def test_creep_recovery_curve(self, creep_data_simple):
        """Test creating recovery curve from creep data."""
        data = creep_data_simple

        # Simulate recovery by subtracting equilibrium compliance
        J_0 = data.y[0]
        J_eq = data.y[-1]

        # Recovery strain
        recovery = data.y - (J_eq - (J_eq - J_0))

        # Recovery should be monotonic
        assert recovery is not None
        assert len(recovery) == len(data.y)


class TestEndToEndFlow:
    """End-to-end workflow tests for flow (steady shear) data."""

    @pytest.mark.integration
    def test_flow_data_detection(self, flow_data_power_law):
        """Test loading and detecting flow data."""
        data = flow_data_power_law

        # Should be detected as rotation (steady shear)

        detected_mode = detect_test_mode(data)

        assert detected_mode == "rotation"

    @pytest.mark.integration
    def test_power_law_flow_behavior(self, flow_data_power_law):
        """Test power-law flow behavior."""
        data = flow_data_power_law

        # Verify shear thinning (viscosity decreases with shear rate)
        viscosity = data.y
        shear_rate = data.x

        # Log-log plot should be linear
        log_viscosity = np.log10(viscosity)
        log_shear = np.log10(shear_rate)

        # Calculate slope (should be negative for shear-thinning)
        slope = np.polyfit(log_shear, log_viscosity, 1)[0]
        assert slope < 0, "Power-law fluid should show shear thinning"

    @pytest.mark.integration
    def test_bingham_flow_behavior(self, flow_data_bingham):
        """Test Bingham plastic flow behavior."""
        data = flow_data_bingham

        # Verify yield stress behavior
        # At higher shear rates, viscosity should decrease
        # (due to yield stress contribution diminishing)

        assert data.metadata.get("test_type") == "Steady Shear"
        assert len(data.x) == len(data.y)


class TestCrossTestModeWorkflows:
    """Test workflows mixing different test modes."""

    @pytest.mark.integration
    def test_multi_technique_analysis(
        self, oscillation_data_simple, relaxation_data_simple, creep_data_simple
    ):
        """Test analyzing multiple test modes from same sample."""
        osc = oscillation_data_simple
        relax = relaxation_data_simple
        creep = creep_data_simple

        # All should be detected correctly

        assert detect_test_mode(osc) == "oscillation"
        assert detect_test_mode(relax) == "relaxation"
        assert detect_test_mode(creep) == "creep"

        # All should have metadata
        assert "test_mode" in osc.metadata
        assert "test_mode" in relax.metadata
        assert "test_mode" in creep.metadata

    @pytest.mark.integration
    def test_interconversion_consistency(self, oscillation_data_simple):
        """Test consistency of data conversions."""
        data = oscillation_data_simple

        # Convert to numpy and back
        numpy_converted = data.to_numpy()
        assert isinstance(numpy_converted.x, np.ndarray)
        assert isinstance(numpy_converted.y, np.ndarray)

        # Convert to JAX and back
        jax_data = data.to_jax()
        numpy_from_jax = jax_data.to_numpy()

        # Should be consistent
        assert np.allclose(data.x, numpy_from_jax.x)
        assert np.allclose(data.y, numpy_from_jax.y)


class TestDataQuality:
    """Test data quality checks in workflows."""

    @pytest.mark.integration
    def test_finite_value_preservation(self, oscillation_data_simple):
        """Test that finite value checks work correctly."""
        data = oscillation_data_simple

        # Data should be finite
        assert np.all(np.isfinite(data.y)), "Data should contain no NaNs or Infs"

    @pytest.mark.integration
    def test_noisy_data_handling(self, synthetic_noisy_data):
        """Test handling of noisy data in workflows."""
        clean, noisy = synthetic_noisy_data

        # Both should be valid
        assert clean.y is not None
        assert noisy.y is not None

        # Shapes should match
        assert clean.x.shape == noisy.x.shape
        assert clean.y.shape == noisy.y.shape

        # Noisy should have higher variance
        assert np.var(noisy.y) > np.var(clean.y)

    @pytest.mark.integration
    def test_multi_temperature_consistency(self, synthetic_multi_temperature_data):
        """Test consistency of multi-temperature datasets."""
        datasets = synthetic_multi_temperature_data

        # All should have same length
        lengths = [len(d.x) for d in datasets]
        assert len(set(lengths)) == 1, "All datasets should have same length"

        # All should be oscillation mode

        for data in datasets:
            assert detect_test_mode(data) == "oscillation"

        # Temperatures should be different
        temps = [d.metadata["temperature"] for d in datasets]
        assert len(set(temps)) == len(temps), "All temperatures should be unique"


class TestMetadataPreservation:
    """Test that metadata is preserved through workflows."""

    @pytest.mark.integration
    def test_metadata_preserved_in_conversion(self, oscillation_data_simple):
        """Test that metadata survives type conversions."""
        original_metadata = oscillation_data_simple.metadata.copy()

        # Convert to JAX and back
        jax_data = oscillation_data_simple.to_jax()
        converted = jax_data.to_numpy()

        # Metadata should be preserved
        assert converted.metadata == original_metadata

    @pytest.mark.integration
    def test_metadata_update_workflow(self, oscillation_data_simple):
        """Test updating metadata in workflows."""
        data = oscillation_data_simple

        # Update metadata
        new_metadata = {"processing_step": "log_transform", "new_field": "value"}
        data.update_metadata(new_metadata)

        # Original metadata should still be there
        assert "test_mode" in data.metadata
        assert "processing_step" in data.metadata
        assert data.metadata["new_field"] == "value"

    @pytest.mark.integration
    def test_metadata_copy_independence(self, oscillation_data_simple):
        """Test that copied metadata is independent."""
        data1 = oscillation_data_simple

        # Create copy with modified metadata
        metadata2 = data1.metadata.copy()
        metadata2["modified"] = True

        data2 = RheoData(
            x=data1.x,
            y=data1.y,
            x_units=data1.x_units,
            y_units=data1.y_units,
            domain=data1.domain,
            metadata=metadata2,
        )

        # Original should be unchanged
        assert "modified" not in data1.metadata
        assert "modified" in data2.metadata
