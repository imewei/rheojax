"""Tests for Power Law model.

This module tests the Power Law (Ostwald-de Waele) model for non-Newtonian flow.
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.test_modes import TestMode
from rheojax.models import PowerLaw


class TestPowerLawBasics:
    """Test basic functionality of Power Law model."""

    @pytest.mark.smoke
    def test_initialization(self):
        """Test model initialization."""
        model = PowerLaw()
        assert model is not None
        assert "K" in model.parameters
        assert "n" in model.parameters

        # Check default values
        K = model.parameters.get_value("K")
        n = model.parameters.get_value("n")
        assert K == 1.0
        assert n == 0.5

    @pytest.mark.smoke
    def test_parameter_bounds(self):
        """Test parameter bounds."""
        model = PowerLaw()

        # K bounds
        K_param = model.parameters.get("K")
        assert K_param.bounds == (1e-6, 1e6)

        # n bounds
        n_param = model.parameters.get("n")
        assert n_param.bounds == (0.01, 2.0)

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test setting parameters."""
        model = PowerLaw()

        # Set valid parameters
        model.parameters.set_value("K", 10.0)
        model.parameters.set_value("n", 0.7)

        assert model.parameters.get_value("K") == 10.0
        assert model.parameters.get_value("n") == 0.7


class TestPowerLawPredictions:
    """Test predictions for Power Law model."""

    @pytest.mark.smoke
    def test_viscosity_prediction_shear_thinning(self):
        """Test viscosity prediction for shear-thinning (n < 1)."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Shear rates
        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])

        # Predict viscosity: η = K * γ̇^(n-1)
        viscosity = model.predict(gamma_dot)

        # Expected: η = 1.0 * γ̇^(-0.5)
        expected = 1.0 * np.power(gamma_dot, -0.5)

        np.testing.assert_allclose(viscosity, expected, rtol=1e-6)

        # Verify shear-thinning behavior (viscosity decreases with shear rate)
        assert viscosity[0] > viscosity[1] > viscosity[2] > viscosity[3]

    @pytest.mark.smoke
    def test_viscosity_prediction_shear_thickening(self):
        """Test viscosity prediction for shear-thickening (n > 1)."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 1.5)

        # Shear rates
        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])

        # Predict viscosity: η = K * γ̇^(n-1)
        viscosity = model.predict(gamma_dot)

        # Expected: η = 1.0 * γ̇^0.5
        expected = 1.0 * np.power(gamma_dot, 0.5)

        np.testing.assert_allclose(viscosity, expected, rtol=1e-6)

        # Verify shear-thickening behavior (viscosity increases with shear rate)
        assert viscosity[0] < viscosity[1] < viscosity[2] < viscosity[3]

    def test_stress_prediction(self):
        """Test stress prediction: σ = K * γ̇^n."""
        model = PowerLaw()
        model.parameters.set_value("K", 2.0)
        model.parameters.set_value("n", 0.6)

        gamma_dot = np.array([1.0, 10.0, 100.0])

        # Predict stress
        stress = model.predict_stress(gamma_dot)

        # Expected: σ = 2.0 * γ̇^0.6
        expected = 2.0 * np.power(gamma_dot, 0.6)

        np.testing.assert_allclose(stress, expected, rtol=1e-6)

    def test_newtonian_limit(self):
        """Test Newtonian limit (n = 1)."""
        model = PowerLaw()
        model.parameters.set_value("K", 5.0)
        model.parameters.set_value("n", 1.0)

        gamma_dot = np.array([0.1, 1.0, 10.0, 100.0])

        # Predict viscosity: η = K * γ̇^0 = K (constant)
        viscosity = model.predict(gamma_dot)

        # Should be constant viscosity = K = 5.0
        expected = np.full_like(gamma_dot, 5.0)

        np.testing.assert_allclose(viscosity, expected, rtol=1e-6)


class TestPowerLawRheoData:
    """Test Power Law with RheoData."""

    def test_predict_rheo_viscosity(self):
        """Test prediction with RheoData (viscosity output)."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Create RheoData with shear rate
        gamma_dot = np.logspace(-2, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),  # Placeholder
            x_units="1/s",
            y_units="Pa·s",
            domain="time",
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        # Predict
        result = model.predict_rheo(rheo_data, output="viscosity")

        # Check result
        assert result.x_units == "1/s"
        assert result.y_units == "Pa·s"
        assert result.metadata["model"] == "PowerLaw"

        # Verify values
        expected = 1.0 * np.power(gamma_dot, -0.5)
        np.testing.assert_allclose(result.y, expected, rtol=1e-6)

    def test_predict_rheo_stress(self):
        """Test prediction with RheoData (stress output)."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Create RheoData with shear rate
        gamma_dot = np.logspace(-2, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            x_units="1/s",
            y_units="Pa",
            domain="time",
            metadata={"test_mode": TestMode.ROTATION},
            validate=False,
        )

        # Predict
        result = model.predict_rheo(rheo_data, output="stress")

        # Check result
        assert result.y_units == "Pa"

        # Verify values
        expected = 1.0 * np.power(gamma_dot, 0.5)
        np.testing.assert_allclose(result.y, expected, rtol=1e-6)

    def test_wrong_test_mode(self):
        """Test that wrong test mode raises error."""
        model = PowerLaw()

        # Create RheoData with OSCILLATION mode
        rheo_data = RheoData(
            x=np.array([1.0, 10.0]),
            y=np.array([0.0, 0.0]),
            metadata={"test_mode": TestMode.OSCILLATION},
            validate=False,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="only supports ROTATION"):
            model.predict_rheo(rheo_data)


class TestPowerLawFitting:
    """Test parameter fitting for Power Law model."""

    def test_fit_synthetic_data(self):
        """Test fitting with synthetic data."""
        # Create synthetic data with known parameters
        K_true = 2.5
        n_true = 0.6

        gamma_dot = np.logspace(-2, 2, 50)
        viscosity_true = K_true * np.power(gamma_dot, n_true - 1.0)

        # Add small noise
        np.random.seed(42)
        viscosity_noisy = viscosity_true * (
            1.0 + 0.01 * np.random.randn(len(gamma_dot))
        )

        # Fit model
        model = PowerLaw()
        model.fit(gamma_dot, viscosity_noisy)

        # Check fitted parameters are close to true values
        K_fit = model.parameters.get_value("K")
        n_fit = model.parameters.get_value("n")

        # Allow 5% error due to noise
        assert abs(K_fit - K_true) / K_true < 0.05
        assert abs(n_fit - n_true) / n_true < 0.05

    def test_fit_predict(self):
        """Test fit_predict method."""
        K_true = 1.5
        n_true = 0.7

        gamma_dot = np.logspace(-1, 2, 30)
        viscosity_true = K_true * np.power(gamma_dot, n_true - 1.0)

        # Fit and predict
        model = PowerLaw()
        predictions = model.fit_predict(gamma_dot, viscosity_true)

        # Predictions should be close to true values
        np.testing.assert_allclose(predictions, viscosity_true, rtol=0.01)


class TestPowerLawNumericalStability:
    """Test numerical stability of Power Law model."""

    def test_extreme_shear_rates(self):
        """Test with extreme shear rates."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Very low and very high shear rates
        gamma_dot = np.array([1e-12, 1e-6, 1e6, 1e12])

        # Should not raise errors
        viscosity = model.predict(gamma_dot)

        # Check for NaN or Inf
        assert np.all(np.isfinite(viscosity))

    def test_zero_shear_rate(self):
        """Test behavior at zero shear rate."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Zero shear rate (should give infinite viscosity for n < 1)
        gamma_dot = np.array([0.0])

        viscosity = model.predict(gamma_dot)

        # For n < 1, η → ∞ as γ̇ → 0
        assert np.isinf(viscosity[0])

    def test_negative_shear_rates(self):
        """Test with negative shear rates (should use absolute value)."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        gamma_dot_pos = np.array([1.0, 10.0, 100.0])
        gamma_dot_neg = -gamma_dot_pos

        viscosity_pos = model.predict(gamma_dot_pos)
        viscosity_neg = model.predict(gamma_dot_neg)

        # Should be identical (using absolute value)
        np.testing.assert_allclose(viscosity_pos, viscosity_neg, rtol=1e-6)


class TestPowerLawPerformance:
    """Test JAX performance optimizations."""

    def test_jit_compilation(self):
        """Test that JIT compilation works."""
        model = PowerLaw()
        model.parameters.set_value("K", 1.0)
        model.parameters.set_value("n", 0.5)

        # Large array for performance testing
        gamma_dot = np.logspace(-3, 3, 10000)

        # First call (compilation + execution)
        viscosity1 = model.predict(gamma_dot)

        # Second call (should be faster, using compiled version)
        viscosity2 = model.predict(gamma_dot)

        # Results should be identical
        np.testing.assert_array_equal(viscosity1, viscosity2)

    def test_vectorization(self):
        """Test that vectorization works correctly."""
        model = PowerLaw()
        model.parameters.set_value("K", 2.0)
        model.parameters.set_value("n", 0.7)

        # Test with different array sizes
        for size in [10, 100, 1000]:
            gamma_dot = np.logspace(-2, 2, size)
            viscosity = model.predict(gamma_dot)

            assert viscosity.shape == gamma_dot.shape
            assert np.all(np.isfinite(viscosity))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
