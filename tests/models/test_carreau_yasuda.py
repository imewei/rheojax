"""Tests for Carreau-Yasuda model.

This module tests the Carreau-Yasuda model with transition parameter 'a'.
"""

import numpy as np
import pytest

from rheo.models.carreau_yasuda import CarreauYasuda
from rheo.core.data import RheoData
from rheo.core.test_modes import TestMode


class TestCarreauYasudaBasics:
    """Test basic functionality."""

    def test_initialization(self):
        """Test model initialization."""
        model = CarreauYasuda()
        assert 'a' in model.parameters
        assert model.parameters.get_value('a') == 2.0

    def test_reduces_to_carreau_when_a_equals_2(self):
        """Test that a=2 reduces to Carreau model."""
        model = CarreauYasuda()
        model.parameters.set_value('eta0', 100.0)
        model.parameters.set_value('eta_inf', 1.0)
        model.parameters.set_value('lambda_', 1.0)
        model.parameters.set_value('n', 0.5)
        model.parameters.set_value('a', 2.0)  # Carreau case

        gamma_dot = np.logspace(-2, 2, 50)
        viscosity = model.predict(gamma_dot)

        # Compare with Carreau formula: η = η∞ + (η0-η∞)[1+(λγ̇)²]^((n-1)/2)
        lambda_gamma = 1.0 * gamma_dot
        carreau_viscosity = 1.0 + (100.0 - 1.0) * np.power(1.0 + lambda_gamma**2, (0.5 - 1.0) / 2.0)

        np.testing.assert_allclose(viscosity, carreau_viscosity, rtol=1e-6)

    def test_transition_parameter_effect(self):
        """Test that parameter 'a' affects transition width."""
        model1 = CarreauYasuda()
        model1.parameters.set_value('eta0', 100.0)
        model1.parameters.set_value('eta_inf', 1.0)
        model1.parameters.set_value('lambda_', 1.0)
        model1.parameters.set_value('n', 0.5)
        model1.parameters.set_value('a', 0.5)  # Sharp transition

        model2 = CarreauYasuda()
        model2.parameters.set_value('eta0', 100.0)
        model2.parameters.set_value('eta_inf', 1.0)
        model2.parameters.set_value('lambda_', 1.0)
        model2.parameters.set_value('n', 0.5)
        model2.parameters.set_value('a', 2.0)  # Smooth transition

        gamma_dot = np.array([0.5, 1.0, 2.0])  # Transition region
        visc1 = model1.predict(gamma_dot)
        visc2 = model2.predict(gamma_dot)

        # Different 'a' values should give different results
        assert not np.allclose(visc1, visc2)


class TestCarreauYasudaPredictions:
    """Test predictions."""

    def test_low_shear_plateau(self):
        """Test Newtonian plateau at low shear rates."""
        model = CarreauYasuda()
        model.parameters.set_value('eta0', 1000.0)
        model.parameters.set_value('eta_inf', 1.0)
        model.parameters.set_value('lambda_', 1.0)
        model.parameters.set_value('n', 0.5)
        model.parameters.set_value('a', 1.0)

        gamma_dot = np.array([1e-6, 1e-5, 1e-4])
        viscosity = model.predict(gamma_dot)

        np.testing.assert_allclose(viscosity, 1000.0, rtol=0.01)

    def test_stress_prediction(self):
        """Test stress prediction."""
        model = CarreauYasuda()
        model.parameters.set_value('eta0', 100.0)
        model.parameters.set_value('eta_inf', 1.0)
        model.parameters.set_value('lambda_', 1.0)
        model.parameters.set_value('n', 0.5)
        model.parameters.set_value('a', 2.0)

        gamma_dot = np.array([1.0, 10.0])
        stress = model.predict_stress(gamma_dot)
        viscosity = model.predict(gamma_dot)

        expected_stress = viscosity * gamma_dot
        np.testing.assert_allclose(stress, expected_stress, rtol=1e-6)


class TestCarreauYasudaRheoData:
    """Test with RheoData."""

    def test_predict_rheo(self):
        """Test prediction with RheoData."""
        model = CarreauYasuda()
        model.parameters.set_value('eta0', 100.0)
        model.parameters.set_value('eta_inf', 1.0)
        model.parameters.set_value('lambda_', 1.0)
        model.parameters.set_value('n', 0.5)
        model.parameters.set_value('a', 1.5)

        gamma_dot = np.logspace(-2, 2, 50)
        rheo_data = RheoData(
            x=gamma_dot,
            y=np.zeros_like(gamma_dot),
            metadata={'test_mode': TestMode.ROTATION},
            validate=False
        )

        result = model.predict_rheo(rheo_data, output='viscosity')
        assert result.metadata['model'] == 'CarreauYasuda'
        assert np.all(np.diff(result.y) <= 0)  # Shear-thinning

    def test_wrong_test_mode(self):
        """Test error for wrong test mode."""
        model = CarreauYasuda()
        rheo_data = RheoData(
            x=np.array([1.0]),
            y=np.array([0.0]),
            metadata={'test_mode': TestMode.RELAXATION},
            validate=False
        )

        with pytest.raises(ValueError, match="only supports ROTATION"):
            model.predict_rheo(rheo_data)


class TestCarreauYasudaStability:
    """Test numerical stability."""

    def test_extreme_shear_rates(self):
        """Test with extreme shear rates."""
        model = CarreauYasuda()
        model.parameters.set_value('eta0', 100.0)
        model.parameters.set_value('eta_inf', 1.0)
        model.parameters.set_value('lambda_', 1.0)
        model.parameters.set_value('n', 0.5)
        model.parameters.set_value('a', 1.0)

        gamma_dot = np.array([1e-12, 1e12])
        viscosity = model.predict(gamma_dot)

        assert np.all(np.isfinite(viscosity))
        assert np.all(viscosity >= 1.0 * 0.99)
        assert np.all(viscosity <= 100.0 * 1.01)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
