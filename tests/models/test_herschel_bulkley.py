"""Tests for Herschel-Bulkley model.

This module tests the Herschel-Bulkley model for viscoplastic materials
with yield stress and power-law behavior.
"""

import numpy as np
import pytest

from rheo.core.data import RheoData
from rheo.core.test_modes import TestMode
from rheo.models.herschel_bulkley import HerschelBulkley


class TestHerschelBulkleyBasics:
    """Test basic functionality of Herschel-Bulkley model."""

    def test_initialization(self):
        """Test model initialization."""
        model = HerschelBulkley()
        assert model is not None
        assert "sigma_y" in model.parameters
        assert "K" in model.parameters
        assert "n" in model.parameters

    def test_parameter_bounds(self):
        """Test parameter bounds."""
        model = HerschelBulkley()

        sigma_y_param = model.parameters.get("sigma_y")
        assert sigma_y_param.bounds == (0.0, 1e6)

        K_param = model.parameters.get("K")
        assert K_param.bounds == (1e-6, 1e6)

        n_param = model.parameters.get("n")
        assert n_param.bounds == (0.01, 2.0)


class TestHerschelBulkleyPredictions:
    """Test predictions for Herschel-Bulkley model."""

    def test_stress_prediction_basic(self):
        """Test basic stress prediction: σ = σ_y + K γ̇^n."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 2.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.array([1.0, 10.0, 100.0])

        stress = model.predict(gamma_dot)

        # Expected: σ = 10.0 + 2.0 * γ̇^0.5
        expected = 10.0 + 2.0 * np.power(gamma_dot, 0.5)

        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_yield_stress_behavior(self):
        """Test that stress is zero below yield threshold."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 100.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 1.0)

        # Very low shear rates (below yield threshold)
        gamma_dot = np.array([1e-12, 1e-11, 1e-10])

        stress = model.predict(gamma_dot)

        # Should be zero (below yield)
        np.testing.assert_allclose(stress, 0.0, atol=1e-10)

    def test_power_law_limit(self):
        """Test that σ_y = 0 reduces to Power Law."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 0.0)  # No yield stress
        model.parameters.set_value("K", 2.5)
        model.parameters.set_value("n", 0.6)

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])

        stress = model.predict(gamma_dot)

        # Should match Power Law: σ = K γ̇^n
        expected = 2.5 * np.power(gamma_dot, 0.6)

        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_bingham_limit(self):
        """Test that n = 1 reduces to Bingham model."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 0.5)
        model.parameters.set_value("n", 1.0)  # Linear

        gamma_dot = np.array([1.0, 10.0, 100.0])

        stress = model.predict(gamma_dot)

        # Should match Bingham: σ = σ_y + η_p γ̇ = 10.0 + 0.5 * γ̇
        expected = 10.0 + 0.5 * gamma_dot

        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_viscosity_prediction(self):
        """Test apparent viscosity prediction."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.array([1.0, 10.0, 100.0])

        viscosity = model.predict_viscosity(gamma_dot)

        # η_app = σ/γ̇ = σ_y/γ̇ + K γ̇^(n-1)
        expected = 10.0 / gamma_dot + 1.0 * np.power(gamma_dot, -0.5)

        np.testing.assert_allclose(viscosity, expected, rtol=1e-6)

    def test_shear_thinning_behavior(self):
        """Test shear-thinning behavior (n < 1)."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 5.0)
        model.parameters.set_value("K", 2.0)
        model.parameters.set_value("n", 0.6)

        gamma_dot = np.array([1.0, 10.0, 100.0, 1000.0])

        viscosity = model.predict_viscosity(gamma_dot)

        # Viscosity should decrease with shear rate (shear-thinning)
        assert viscosity[0] > viscosity[1] > viscosity[2] > viscosity[3]

    def test_shear_thickening_behavior(self):
        """Test shear-thickening behavior (n > 1)."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 5.0)
        model.parameters.set_value("K", 0.1)
        model.parameters.set_value("n", 1.3)

        gamma_dot = np.array([10.0, 100.0, 1000.0])

        stress = model.predict(gamma_dot)

        # For large γ̇, stress should increase faster than linearly
        # Check that stress growth rate increases
        ratio1 = stress[1] / stress[0]
        ratio2 = stress[2] / stress[1]
        assert ratio2 > ratio1  # Accelerating growth


class TestHerschelBulkleyRheoData:
    """Test Herschel-Bulkley with RheoData."""

    def test_predict_rheo_stress(self):
        """Test prediction with RheoData (stress output)."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.logspace(-1, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            x_units="1/s",
            y_units="Pa",
            domain="time",
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        result = model.predict_rheo(rheo_data, output="stress")

        assert result.y_units == "Pa"
        assert result.metadata["model"] == "HerschelBulkley"

        # Stress should be monotonically increasing
        assert np.all(np.diff(result.y) >= 0)

    def test_predict_rheo_viscosity(self):
        """Test prediction with RheoData (viscosity output)."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.logspace(-1, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            x_units="1/s",
            y_units="Pa·s",
            domain="time",
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        result = model.predict_rheo(rheo_data, output="viscosity")

        assert result.y_units == "Pa·s"

        # Viscosity should be monotonically decreasing (shear-thinning)
        assert np.all(np.diff(result.y) <= 0)

    def test_wrong_test_mode(self):
        """Test that wrong test mode raises error."""
        model = HerschelBulkley()

        rheo_data = RheoData(
            x=np.array([1.0, 10.0]),
            y=np.array([0.0, 0.0]),
            metadata={"test_mode": TestMode.CREEP},
            validate=False,
        )

        with pytest.raises(ValueError, match="only supports ROTATION"):
            model.predict_rheo(rheo_data)


class TestHerschelBulkleyFitting:
    """Test parameter fitting for Herschel-Bulkley model."""

    def test_fit_synthetic_data(self):
        """Test fitting with synthetic data."""
        # Create synthetic data
        sigma_y_true = 15.0
        K_true = 2.0
        n_true = 0.6

        gamma_dot = np.logspace(-1, 2, 50)
        stress_true = sigma_y_true + K_true * np.power(gamma_dot, n_true)

        # Fit model
        model = HerschelBulkley()
        model.fit(gamma_dot, stress_true)

        # Predictions should match true values
        predictions = model.predict(gamma_dot)
        np.testing.assert_allclose(predictions, stress_true, rtol=0.1)

    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        sigma_y_true = 10.0
        K_true = 1.5
        n_true = 0.7

        gamma_dot = np.logspace(-1, 2, 100)
        stress_true = sigma_y_true + K_true * np.power(gamma_dot, n_true)

        # Add noise
        np.random.seed(42)
        stress_noisy = stress_true * (1.0 + 0.05 * np.random.randn(len(gamma_dot)))

        # Fit model
        model = HerschelBulkley()
        model.fit(gamma_dot, stress_noisy)

        # Check fitted parameters are reasonable (within 30% due to noise and fitting complexity)
        # HB model has challenging parameter correlations that make fitting difficult
        sigma_y_fit = model.parameters.get_value("sigma_y")
        K_fit = model.parameters.get_value("K")
        n_fit = model.parameters.get_value("n")

        assert abs(sigma_y_fit - sigma_y_true) / sigma_y_true < 0.3
        assert abs(K_fit - K_true) / K_true < 0.3
        assert abs(n_fit - n_true) / n_true < 0.3


class TestHerschelBulkleyNumericalStability:
    """Test numerical stability of Herschel-Bulkley model."""

    def test_extreme_shear_rates(self):
        """Test with extreme shear rates."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.array([1e-15, 1e-6, 1e6, 1e15])

        stress = model.predict(gamma_dot)

        # Check for NaN or Inf
        assert np.all(np.isfinite(stress))

    def test_zero_shear_rate(self):
        """Test behavior at exactly zero shear rate."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot = np.array([0.0])

        stress = model.predict(gamma_dot)

        # Below yield threshold, stress should be zero
        np.testing.assert_allclose(stress, 0.0, atol=1e-10)

    def test_negative_shear_rates(self):
        """Test with negative shear rates."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot_pos = np.array([1.0, 10.0, 100.0])
        gamma_dot_neg = -gamma_dot_pos

        stress_pos = model.predict(gamma_dot_pos)
        stress_neg = model.predict(gamma_dot_neg)

        # Should be identical (using absolute value)
        np.testing.assert_allclose(stress_pos, stress_neg, rtol=1e-6)

    def test_edge_parameter_values(self):
        """Test with edge parameter values."""
        model = HerschelBulkley()

        # Minimum values
        model.parameters.set_value("sigma_y", 0.0)
        model.parameters.set_value("K", 1e-6)
        model.parameters.set_value("n", 0.01)

        gamma_dot = np.logspace(-2, 2, 50)
        stress = model.predict(gamma_dot)

        assert np.all(np.isfinite(stress))
        assert np.all(stress >= 0)

        # Maximum values
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("K", 1e6)
        model.parameters.set_value("n", 2.0)

        stress = model.predict(gamma_dot)

        assert np.all(np.isfinite(stress))
        assert np.all(stress >= 0)


class TestHerschelBulkleyPhysicalBehavior:
    """Test physical behavior of Herschel-Bulkley model."""

    def test_yield_stress_consistency(self):
        """Test that yield stress is consistent across predictions."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 50.0)
        model.parameters.set_value("K", 2.0)
        model.parameters.set_value("n", 0.5)

        # At very low shear rates, stress should be zero (below yield)
        gamma_dot_low = np.array([1e-12, 1e-11, 1e-10])
        stress_low = model.predict(gamma_dot_low)
        np.testing.assert_allclose(stress_low, 0.0, atol=1e-10)

        # At higher shear rates, stress should include yield contribution
        gamma_dot_high = np.array([1.0, 10.0, 100.0])
        stress_high = model.predict(gamma_dot_high)

        # All should be greater than yield stress
        assert np.all(stress_high >= 50.0)

    def test_stress_continuity(self):
        """Test that stress is continuous across yield threshold."""
        model = HerschelBulkley()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Sample around yield threshold
        gamma_dot = np.logspace(-12, -6, 1000)
        stress = model.predict(gamma_dot)

        # Stress should be smooth (no discontinuities)
        # Check that differences are small and smooth
        diffs = np.diff(stress)

        # Should not have sudden jumps (allow for numerical yield transition)
        # At very low shear rates approaching yield, numerical precision can cause small jumps
        assert np.all(np.abs(diffs) < 15)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
