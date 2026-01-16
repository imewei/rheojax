"""Tests for Bingham model.

This module tests the Bingham model for linear viscoplastic flow.
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.test_modes import TestMode
from rheojax.models import Bingham


class TestBinghamBasics:
    """Test basic functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = Bingham()
        assert "sigma_y" in model.parameters
        assert "eta_p" in model.parameters

    def test_parameter_bounds(self):
        """Test parameter bounds."""
        model = Bingham()
        sigma_y_param = model.parameters.get("sigma_y")
        assert sigma_y_param.bounds == (0.0, 1e6)

        eta_p_param = model.parameters.get("eta_p")
        assert eta_p_param.bounds == (1e-6, 1e12)


class TestBinghamPredictions:
    """Test predictions."""

    def test_stress_prediction_basic(self):
        """Test basic stress prediction: σ = σ_y + η_p γ̇."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.array([1.0, 10.0, 100.0])
        stress = model.predict(gamma_dot)

        # Expected: σ = 10.0 + 0.5 * γ̇
        expected = 10.0 + 0.5 * gamma_dot
        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_yield_stress_behavior(self):
        """Test that stress is zero below yield threshold."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 100.0)
        model.parameters.set_value("eta_p", 0.1)

        gamma_dot = np.array([1e-12, 1e-11, 1e-10])
        stress = model.predict(gamma_dot)

        np.testing.assert_allclose(stress, 0.0, atol=1e-10)

    def test_newtonian_limit(self):
        """Test that σ_y = 0 reduces to Newtonian."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 0.0)
        model.parameters.set_value("eta_p", 5.0)

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])
        stress = model.predict(gamma_dot)

        # Should be Newtonian: σ = η_p * γ̇ = 5.0 * γ̇
        expected = 5.0 * gamma_dot
        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_viscosity_prediction(self):
        """Test apparent viscosity prediction."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.array([1.0, 10.0, 100.0])
        viscosity = model.predict_viscosity(gamma_dot)

        # η_app = σ_y/γ̇ + η_p
        expected = 10.0 / gamma_dot + 0.5
        np.testing.assert_allclose(viscosity, expected, rtol=1e-6)

    def test_linear_stress_growth(self):
        """Test that stress grows linearly above yield."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 5.0)
        model.parameters.set_value("eta_p", 0.2)

        gamma_dot = np.array([10.0, 20.0, 30.0, 40.0])
        stress = model.predict(gamma_dot)

        # Differences should be constant (linear growth)
        diffs = np.diff(stress)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-6)

    def test_herschel_bulkley_special_case(self):
        """Test that this is Herschel-Bulkley with n=1."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 2.0)

        gamma_dot = np.array([1.0, 10.0, 100.0])
        stress = model.predict(gamma_dot)

        # Should match Herschel-Bulkley with n=1: σ = σ_y + K*γ̇^1
        expected = 10.0 + 2.0 * gamma_dot
        np.testing.assert_allclose(stress, expected, rtol=1e-6)


class TestBinghamRheoData:
    """Test with RheoData."""

    def test_predict_rheo_stress(self):
        """Test prediction with RheoData (stress output)."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.logspace(-1, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        result = model.predict_rheo(rheo_data, output="stress")
        assert result.metadata["model"] == "Bingham"
        assert result.y_units == "Pa"

        # Stress should be monotonically increasing
        assert np.all(np.diff(result.y) >= 0)

    def test_predict_rheo_viscosity(self):
        """Test prediction with RheoData (viscosity output)."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.logspace(-1, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        result = model.predict_rheo(rheo_data, output="viscosity")
        assert result.y_units == "Pa·s"

        # Viscosity should be monotonically decreasing
        assert np.all(np.diff(result.y) <= 0)

    def test_wrong_test_mode(self):
        """Test error for wrong test mode."""
        model = Bingham()
        rheo_data = RheoData(
            x=np.array([1.0]),
            y=np.array([0.0]),
            metadata={"test_mode": TestMode.OSCILLATION},
            validate=False,
        )

        with pytest.raises(ValueError, match="only supports ROTATION"):
            model.predict_rheo(rheo_data)


class TestBinghamFitting:
    """Test parameter fitting."""

    def test_fit_synthetic_data(self):
        """Test fitting with synthetic data."""
        sigma_y_true = 15.0
        eta_p_true = 0.5

        gamma_dot = np.logspace(-1, 2, 50)
        stress_true = sigma_y_true + eta_p_true * gamma_dot

        model = Bingham()
        model.fit(gamma_dot, stress_true)

        predictions = model.predict(gamma_dot)
        np.testing.assert_allclose(predictions, stress_true, rtol=0.01)

    def test_fit_with_noise(self):
        """Test fitting with noisy data."""
        sigma_y_true = 10.0
        eta_p_true = 0.3

        gamma_dot = np.logspace(-1, 2, 100)
        stress_true = sigma_y_true + eta_p_true * gamma_dot

        np.random.seed(42)
        stress_noisy = stress_true * (1.0 + 0.05 * np.random.randn(len(gamma_dot)))

        model = Bingham()
        model.fit(gamma_dot, stress_noisy)

        sigma_y_fit = model.parameters.get_value("sigma_y")
        eta_p_fit = model.parameters.get_value("eta_p")

        assert abs(sigma_y_fit - sigma_y_true) / sigma_y_true < 0.1
        assert abs(eta_p_fit - eta_p_true) / eta_p_true < 0.1


class TestBinghamStability:
    """Test numerical stability."""

    def test_extreme_shear_rates(self):
        """Test with extreme shear rates."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.array([1e-15, 1e15])
        stress = model.predict(gamma_dot)

        assert np.all(np.isfinite(stress))

    def test_zero_shear_rate(self):
        """Test at zero shear rate."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot = np.array([0.0])
        stress = model.predict(gamma_dot)

        np.testing.assert_allclose(stress, 0.0, atol=1e-10)

    def test_negative_shear_rates(self):
        """Test with negative shear rates."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 0.5)

        gamma_dot_pos = np.array([1.0, 10.0, 100.0])
        gamma_dot_neg = -gamma_dot_pos

        stress_pos = model.predict(gamma_dot_pos)
        stress_neg = model.predict(gamma_dot_neg)

        np.testing.assert_allclose(stress_pos, stress_neg, rtol=1e-6)


class TestBinghamPhysicalBehavior:
    """Test physical behavior."""

    def test_yield_stress_consistency(self):
        """Test yield stress consistency."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 50.0)
        model.parameters.set_value("eta_p", 1.0)

        # Below yield
        gamma_dot_low = np.array([1e-12, 1e-11])
        stress_low = model.predict(gamma_dot_low)
        np.testing.assert_allclose(stress_low, 0.0, atol=1e-10)

        # Above yield
        gamma_dot_high = np.array([1.0, 10.0])
        stress_high = model.predict(gamma_dot_high)
        assert np.all(stress_high >= 50.0)

    def test_plastic_viscosity_consistency(self):
        """Test that plastic viscosity is constant above yield."""
        model = Bingham()
        model.parameters.set_value("sigma_y", 10.0)
        model.parameters.set_value("eta_p", 2.0)

        gamma_dot = np.array([10.0, 20.0, 30.0])
        stress = model.predict(gamma_dot)

        # Slope should be constant = eta_p
        slopes = np.diff(stress) / np.diff(gamma_dot)
        np.testing.assert_allclose(slopes, 2.0, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
