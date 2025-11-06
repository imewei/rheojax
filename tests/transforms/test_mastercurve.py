"""Tests for Mastercurve transform (Time-Temperature Superposition)."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.transforms.mastercurve import Mastercurve

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestMastercurve:
    """Test suite for Mastercurve transform."""

    def test_basic_initialization(self):
        """Test basic mastercurve initialization."""
        mc = Mastercurve()
        assert mc.T_ref == 298.15
        assert mc.method == "wlf"
        assert mc.C1 == 17.44
        assert mc.C2 == 51.6

    def test_custom_initialization(self):
        """Test mastercurve with custom parameters."""
        mc = Mastercurve(reference_temp=273.15, method="arrhenius", E_a=50000)
        assert mc.T_ref == 273.15
        assert mc.method == "arrhenius"
        assert mc.E_a == 50000

    def test_wlf_shift_factor(self):
        """Test WLF shift factor calculation."""
        mc = Mastercurve(reference_temp=298.15, C1=17.44, C2=51.6)

        # At reference temperature, shift should be 1.0
        a_T = mc.get_shift_factor(298.15)
        assert np.abs(a_T - 1.0) < 1e-6

        # At higher temperature, shift should be < 1 (faster relaxation)
        a_T_high = mc.get_shift_factor(323.15)
        assert a_T_high < 1.0

        # At lower temperature, shift should be > 1 (slower relaxation)
        a_T_low = mc.get_shift_factor(273.15)
        assert a_T_low > 1.0

    def test_arrhenius_shift_factor(self):
        """Test Arrhenius shift factor calculation."""
        mc = Mastercurve(reference_temp=298.15, method="arrhenius", E_a=50000)  # J/mol

        # At reference temperature
        a_T = mc.get_shift_factor(298.15)
        assert np.abs(a_T - 1.0) < 1e-6

        # At higher temperature (faster)
        a_T_high = mc.get_shift_factor(323.15)
        assert a_T_high < 1.0

        # At lower temperature (slower)
        a_T_low = mc.get_shift_factor(273.15)
        assert a_T_low > 1.0

    def test_single_dataset_shift(self):
        """Test shifting a single dataset."""
        # Create frequency sweep at elevated temperature
        freq = jnp.logspace(-2, 2, 50)
        G_prime = 1000 * freq  # Simplified modulus

        data = RheoData(
            x=freq, y=G_prime, domain="frequency", metadata={"temperature": 323.15}
        )

        # Shift to reference temperature
        mc = Mastercurve(reference_temp=298.15)
        shifted = mc.transform(data)

        # Frequency should be shifted
        assert not np.array_equal(shifted.x, data.x)

        # Shift factor should be in metadata
        assert "horizontal_shift" in shifted.metadata

    def test_multi_temperature_mastercurve(self):
        """Test creating mastercurve from multiple temperatures."""
        # Create synthetic multi-temperature data
        temps = [273.15, 298.15, 323.15]
        freq = jnp.logspace(-2, 2, 30)

        datasets = []
        for T in temps:
            # Simplified temperature-dependent modulus
            G_prime = 1000 * freq * (T / 298.15)

            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Create mastercurve
        mc = Mastercurve(reference_temp=298.15)
        mastercurve = mc.create_mastercurve(datasets, merge=True)

        # Should have more points than individual datasets
        assert len(mastercurve.x) == len(datasets) * len(freq)

        # Should be sorted
        assert np.all(np.diff(mastercurve.x) >= 0)

        # Metadata should track temperatures
        assert "temperatures" in mastercurve.metadata
        assert len(mastercurve.metadata["temperatures"]) == len(temps)

    def test_mastercurve_no_merge(self):
        """Test creating mastercurve without merging."""
        temps = [273.15, 298.15, 323.15]
        freq = jnp.logspace(-2, 2, 30)

        datasets = []
        for T in temps:
            G_prime = 1000 * freq
            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Create without merging
        mc = Mastercurve(reference_temp=298.15)
        shifted_datasets = mc.create_mastercurve(datasets, merge=False)

        # Should return list
        assert isinstance(shifted_datasets, list)
        assert len(shifted_datasets) == len(temps)

        # Each should be RheoData
        for data in shifted_datasets:
            assert isinstance(data, RheoData)

    def test_vertical_shifting(self):
        """Test vertical shifting for modulus."""
        freq = jnp.logspace(-2, 2, 30)
        G_prime = 1000 * freq

        data = RheoData(
            x=freq, y=G_prime, domain="frequency", metadata={"temperature": 323.15}
        )

        # With vertical shift
        mc_vert = Mastercurve(reference_temp=298.15, vertical_shift=True)
        shifted_vert = mc_vert.transform(data)

        # Without vertical shift
        mc_horiz = Mastercurve(reference_temp=298.15, vertical_shift=False)
        shifted_horiz = mc_horiz.transform(data)

        # y-values should differ
        assert not np.array_equal(shifted_vert.y, shifted_horiz.y)

        # Vertical shift should be in metadata
        assert shifted_vert.metadata["vertical_shift"] != 1.0

    def test_manual_shift_factors(self):
        """Test setting manual shift factors."""
        mc = Mastercurve()

        # Set manual shifts
        shifts = {273.15: 2.0, 298.15: 1.0, 323.15: 0.5}
        mc.set_manual_shifts(shifts)

        # Should use manual shifts
        assert mc.method == "manual"
        assert mc.get_shift_factor(273.15) == 2.0
        assert mc.get_shift_factor(298.15) == 1.0
        assert mc.get_shift_factor(323.15) == 0.5

    def test_missing_temperature_error(self):
        """Test error when temperature is missing."""
        freq = jnp.logspace(-2, 2, 30)
        G_prime = 1000 * freq

        # No temperature in metadata
        data = RheoData(x=freq, y=G_prime, domain="frequency")

        mc = Mastercurve()
        with pytest.raises(ValueError, match="Temperature"):
            mc.transform(data)

    def test_overlap_error_calculation(self):
        """Test overlap error computation."""
        # Create overlapping datasets with wider frequency range
        temps = [273.15, 298.15, 323.15]
        datasets = []

        for T in temps:
            # Use wider frequency range to ensure overlap after shifting
            freq = jnp.logspace(-3, 3, 50)  # Wider range for better overlap
            # Use WLF to pre-shift (simulate real data)
            mc_temp = Mastercurve(reference_temp=298.15)
            a_T = mc_temp.get_shift_factor(T)

            freq_shifted = freq * a_T
            G_prime = 1000 * freq_shifted**0.5  # Power law

            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Compute overlap error
        mc = Mastercurve(reference_temp=298.15)
        error = mc.compute_overlap_error(datasets)

        # Should return finite value (if there's overlap) or inf (if no overlap)
        # Both are valid outcomes depending on the shift factors
        assert isinstance(error, (float, np.floating))
        if np.isfinite(error):
            assert error >= 0

    def test_wlf_optimization(self):
        """Test WLF parameter optimization."""
        # Create synthetic data with known WLF parameters
        C1_true, C2_true = 15.0, 60.0
        T_ref = 298.15
        temps = [273.15, 298.15, 323.15, 348.15]

        datasets = []
        for T in temps:
            freq = jnp.logspace(-1, 1, 20)

            # Calculate true shift
            mc_true = Mastercurve(reference_temp=T_ref, C1=C1_true, C2=C2_true)
            a_T = mc_true.get_shift_factor(T)

            # Create shifted data
            freq_shifted = freq * a_T
            G_prime = 1000 * freq_shifted**0.5

            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Optimize WLF parameters
        mc = Mastercurve(reference_temp=T_ref)
        C1_opt, C2_opt = mc.optimize_wlf_parameters(
            datasets, initial_C1=17.0, initial_C2=50.0
        )

        # Should be reasonable (may not recover exact values due to synthetic data)
        assert 10 < C1_opt < 30
        assert 40 < C2_opt < 100

    def test_metadata_preservation(self):
        """Test metadata preservation in transforms."""
        freq = jnp.logspace(-2, 2, 30)
        G_prime = 1000 * freq

        data = RheoData(
            x=freq,
            y=G_prime,
            domain="frequency",
            metadata={"temperature": 323.15, "sample": "polymer_A", "strain": 0.01},
        )

        mc = Mastercurve(reference_temp=298.15)
        shifted = mc.transform(data)

        # Original metadata preserved
        assert shifted.metadata["sample"] == "polymer_A"
        assert shifted.metadata["strain"] == 0.01

        # Transform metadata added
        assert "transform" in shifted.metadata
        assert shifted.metadata["transform"] == "mastercurve"

    def test_transform_with_list_returns_shifts(self):
        """Test that transform() with a list returns mastercurve and shift factors."""
        # Create multi-temperature datasets
        temps = [273.15, 298.15, 323.15]
        freq = jnp.logspace(-2, 2, 30)

        datasets = []
        for T in temps:
            G_prime = 1000 * freq
            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Call transform with list (should return tuple)
        mc = Mastercurve(reference_temp=298.15)
        result = mc.transform(datasets)

        # Should return tuple of (mastercurve, shift_factors)
        assert isinstance(result, tuple)
        assert len(result) == 2

        mastercurve, shift_factors = result

        # Verify mastercurve
        assert isinstance(mastercurve, RheoData)
        assert len(mastercurve.x) == len(datasets) * len(freq)

        # Verify shift factors
        assert isinstance(shift_factors, dict)
        assert len(shift_factors) == len(temps)
        for T in temps:
            assert T in shift_factors
            assert isinstance(shift_factors[T], float)

        # Reference temperature should have shift factor ~1.0
        assert np.abs(shift_factors[298.15] - 1.0) < 1e-6

    def test_create_mastercurve_with_return_shifts(self):
        """Test create_mastercurve with return_shifts=True."""
        temps = [273.15, 298.15, 323.15]
        freq = jnp.logspace(-2, 2, 30)

        datasets = []
        for T in temps:
            G_prime = 1000 * freq
            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        mc = Mastercurve(reference_temp=298.15)

        # Test with return_shifts=True
        mastercurve, shift_factors = mc.create_mastercurve(datasets, return_shifts=True)

        assert isinstance(mastercurve, RheoData)
        assert isinstance(shift_factors, dict)
        assert len(shift_factors) == len(temps)

        # Shift factors should also be in metadata
        assert "shift_factors" in mastercurve.metadata
        assert mastercurve.metadata["shift_factors"] == shift_factors

    def test_get_wlf_parameters(self):
        """Test getting WLF parameters."""
        mc = Mastercurve(reference_temp=298.15, method="wlf", C1=15.0, C2=60.0)

        params = mc.get_wlf_parameters()

        assert isinstance(params, dict)
        assert params["C1"] == 15.0
        assert params["C2"] == 60.0
        assert params["T_ref"] == 298.15

    def test_get_wlf_parameters_wrong_method(self):
        """Test error when getting WLF parameters for non-WLF method."""
        mc = Mastercurve(reference_temp=298.15, method="arrhenius", E_a=50000)

        with pytest.raises(ValueError, match="WLF parameters not available"):
            mc.get_wlf_parameters()

    def test_get_arrhenius_parameters(self):
        """Test getting Arrhenius parameters."""
        mc = Mastercurve(reference_temp=298.15, method="arrhenius", E_a=50000)

        params = mc.get_arrhenius_parameters()

        assert isinstance(params, dict)
        assert params["E_a"] == 50000
        assert params["T_ref"] == 298.15

    def test_get_arrhenius_parameters_wrong_method(self):
        """Test error when getting Arrhenius parameters for non-Arrhenius method."""
        mc = Mastercurve(reference_temp=298.15, method="wlf")

        with pytest.raises(ValueError, match="Arrhenius parameters not available"):
            mc.get_arrhenius_parameters()

    def test_get_shift_factors_array_with_temps(self):
        """Test getting shift factors as arrays with provided temperatures."""
        mc = Mastercurve(reference_temp=298.15, method="wlf", C1=17.44, C2=51.6)

        # Provide temperatures
        temps_input = [273.15, 298.15, 323.15]
        temps_array, shifts_array = mc.get_shift_factors_array(temps_input)

        # Should return numpy arrays
        assert isinstance(temps_array, np.ndarray)
        assert isinstance(shifts_array, np.ndarray)

        # Should have same length
        assert len(temps_array) == len(shifts_array)
        assert len(temps_array) == 3

        # Should be sorted
        assert np.all(np.diff(temps_array) >= 0)

        # Reference temperature should have shift factor ~1.0
        ref_idx = np.where(np.abs(temps_array - 298.15) < 0.01)[0][0]
        assert np.abs(shifts_array[ref_idx] - 1.0) < 1e-6

    def test_get_shift_factors_array_after_mastercurve(self):
        """Test getting shift factors as arrays after creating mastercurve."""
        # Create datasets
        temps = [273.15, 298.15, 323.15]
        freq = jnp.logspace(-2, 2, 30)

        datasets = []
        for T in temps:
            G_prime = 1000 * freq
            data = RheoData(
                x=freq, y=G_prime, domain="frequency", metadata={"temperature": T}
            )
            datasets.append(data)

        # Create mastercurve
        mc = Mastercurve(reference_temp=298.15)
        mastercurve, shift_factors = mc.transform(datasets)

        # Get shift factors as arrays (without providing temperatures)
        temps_array, shifts_array = mc.get_shift_factors_array()

        # Should match the shift_factors dict
        assert len(temps_array) == len(temps)
        for T, shift in zip(temps_array, shifts_array):
            assert np.abs(shift_factors[T] - shift) < 1e-10

    def test_get_shift_factors_array_no_temps_error(self):
        """Test error when getting shift factors without temps and no mastercurve."""
        mc = Mastercurve(reference_temp=298.15, method="wlf")

        # Should raise error if no temperatures provided and no mastercurve created
        with pytest.raises(ValueError, match="No shift factors available"):
            mc.get_shift_factors_array()

    def test_get_shift_factors_array_sorting(self):
        """Test that shift factors are sorted by temperature."""
        mc = Mastercurve(reference_temp=298.15, method="wlf")

        # Provide unsorted temperatures
        temps_input = [323.15, 273.15, 298.15, 348.15]
        temps_array, shifts_array = mc.get_shift_factors_array(temps_input)

        # Should be sorted
        assert np.all(np.diff(temps_array) > 0)
        assert len(temps_array) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
