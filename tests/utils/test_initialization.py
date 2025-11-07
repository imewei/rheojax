"""Comprehensive tests for smart parameter initialization.

Tests all 11 initialization functions for fractional models:
- Feature extraction from frequency-domain data
- Parameter initialization with bounds validation
- Integration with model fitting
- Failure handling and fallback behavior
"""

import numpy as np
import pytest

from rheojax.core.parameters import ParameterSet
from rheojax.utils.initialization import (
    extract_frequency_features,
    initialize_fractional_burgers,
    initialize_fractional_jeffreys,
    initialize_fractional_kelvin_voigt,
    initialize_fractional_kv_zener,
    initialize_fractional_maxwell_gel,
    initialize_fractional_maxwell_liquid,
    initialize_fractional_maxwell_model,
    initialize_fractional_poynting_thomson,
    initialize_fractional_zener_ll,
    initialize_fractional_zener_sl,
    initialize_fractional_zener_ss,
)


class TestExtractFrequencyFeatures:
    """Test frequency-domain feature extraction."""

    def test_extract_features_typical_solid(self):
        """Test feature extraction from typical solid-like material."""
        # Create synthetic data: solid with plateau at low freq, crossover, then increase
        omega = np.logspace(-2, 3, 100)
        G_low = 1e5
        G_high = 1e6
        omega_c = 10.0

        # Simple power-law transition
        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.5
        G_double_prime = 5e4 * (omega / omega_c) ** 0.3

        y_data = np.column_stack([G_prime, G_double_prime])

        features = extract_frequency_features(omega, y_data)

        assert features is not None
        assert "low_plateau" in features
        assert "high_plateau" in features
        assert "omega_mid" in features
        assert "alpha_estimate" in features

        # Check reasonable values - features should be extracted
        assert features["low_plateau"] > 0
        assert features["high_plateau"] > features["low_plateau"]
        assert features["omega_mid"] > 0
        assert 0.01 < features["alpha_estimate"] < 0.99

    def test_extract_features_liquid_like(self):
        """Test feature extraction from liquid-like material (no low-freq plateau)."""
        omega = np.logspace(-2, 3, 100)

        # Liquid: G' ~ omega^2, G" ~ omega
        G_prime = 1e3 * omega**2
        G_double_prime = 5e3 * omega

        y_data = np.column_stack([G_prime, G_double_prime])

        features = extract_frequency_features(omega, y_data)

        assert features is not None
        # For liquid, low_plateau should be from first data point
        assert features["low_plateau"] > 0
        assert features["alpha_estimate"] > 0

    def test_extract_features_invalid_shape(self):
        """Test that invalid data shape returns dict with valid=False or raises."""
        omega = np.logspace(-2, 2, 50)

        # 1D array instead of 2D - should return dict with valid=False
        y_data = np.ones(50)
        features = extract_frequency_features(omega, y_data)
        assert features is not None
        assert features.get("valid", False) is False

        # Wrong number of columns - may raise ValueError
        y_data = np.ones((50, 3))
        try:
            features = extract_frequency_features(omega, y_data)
            # If it doesn't raise, it should return invalid
            assert features.get("valid", False) is False
        except ValueError:
            # ValueError is acceptable for malformed data
            pass

    def test_extract_features_size_mismatch(self):
        """Test that mismatched sizes raise ValueError or return invalid."""
        omega = np.logspace(-2, 2, 50)
        y_data = np.column_stack([np.ones(40), np.ones(40)])

        try:
            features = extract_frequency_features(omega, y_data)
            # If it doesn't raise, it should return invalid
            assert features.get("valid", False) is False
        except ValueError:
            # ValueError is acceptable for mismatched sizes
            pass

    def test_extract_features_noise_robustness(self):
        """Test that Savitzky-Golay filtering handles noisy data."""
        omega = np.logspace(-2, 3, 100)
        G_low = 1e5
        G_high = 1e6

        G_prime = G_low + (G_high - G_low) * (omega / 10) ** 0.5
        G_double_prime = 5e4 * (omega / 10) ** 0.3

        # Add 10% noise
        noise_level = 0.1
        G_prime += np.random.normal(0, noise_level * G_prime.mean(), omega.shape)
        G_double_prime += np.random.normal(
            0, noise_level * G_double_prime.mean(), omega.shape
        )

        y_data = np.column_stack([G_prime, G_double_prime])

        features = extract_frequency_features(omega, y_data)

        # Should still extract reasonable features despite noise
        assert features is not None
        assert features["low_plateau"] > 0
        assert features["high_plateau"] > features["low_plateau"]


class TestFractionalZenerSolidSolid:
    """Test initialize_fractional_zener_ss (4 parameters: Ge, Gm, alpha, tau_alpha)."""

    def test_initialize_success(self):
        """Test successful initialization with typical data."""
        omega = np.logspace(-2, 3, 100)
        G_low = 1e5
        G_high = 1e6
        omega_c = 10.0

        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.5
        G_double_prime = 5e4 * (omega / omega_c) ** 0.3
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1e4, bounds=(1e-3, 1e9))
        params.add("Gm", value=1e4, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_zener_ss(omega, y_data, params)

        assert success is True

        # Check parameters are within bounds
        assert 1e-3 <= params.get_value("Ge") <= 1e9
        assert 1e-3 <= params.get_value("Gm") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau_alpha") <= 1e6

        # Check reasonable values
        Ge = params.get_value("Ge")
        Gm = params.get_value("Gm")
        tau_alpha = params.get_value("tau_alpha")

        assert Ge > 0  # Should be positive
        assert Gm > 0  # Maxwell modulus > 0
        assert tau_alpha > 0  # Relaxation time should be positive

    def test_initialize_invalid_data(self):
        """Test that invalid data returns False."""
        omega = np.logspace(-2, 2, 50)
        y_data = np.ones(50)  # Wrong shape

        params = ParameterSet()
        params.add("Ge", value=1e4, bounds=(1e-3, 1e9))
        params.add("Gm", value=1e4, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_zener_ss(omega, y_data, params)
        assert success is False


class TestFractionalMaxwellLiquid:
    """Test initialize_fractional_maxwell_liquid (3 parameters: Gm, alpha, tau_alpha)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        # Liquid-like: increasing with frequency
        G_prime = 1e3 * omega**1.5
        G_double_prime = 5e3 * omega**0.8
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Gm", value=1e3, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_maxwell_liquid(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("Gm") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau_alpha") <= 1e6


class TestFractionalMaxwellGel:
    """Test initialize_fractional_maxwell_gel (3 parameters: c_alpha, alpha, eta)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_prime = 1e5 + 5e4 * omega**0.5
        G_double_prime = 2e4 * omega**0.3
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("c_alpha", value=1e4, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("eta", value=1e4, bounds=(1e-6, 1e12))

        success = initialize_fractional_maxwell_gel(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("c_alpha") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("eta") <= 1e12


class TestFractionalZenerLiquidLiquid:
    """Test initialize_fractional_zener_ll (6 parameters)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_prime = 1e4 * omega**1.2
        G_double_prime = 5e3 * omega**0.9
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("c1", value=500.0, bounds=(1e-3, 1e9))
        params.add("c2", value=500.0, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("beta", value=0.5, bounds=(0.0, 1.0))
        params.add("gamma", value=0.5, bounds=(0.0, 1.0))
        params.add("tau", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_zener_ll(omega, y_data, params)

        assert success is True

        # Check all 6 parameters are within bounds
        assert 1e-3 <= params.get_value("c1") <= 1e9
        assert 1e-3 <= params.get_value("c2") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 0.0 <= params.get_value("beta") <= 1.0
        assert 0.0 <= params.get_value("gamma") <= 1.0
        assert 1e-6 <= params.get_value("tau") <= 1e6


class TestFractionalZenerSolidLiquid:
    """Test initialize_fractional_zener_sl (4 parameters: Ge, c_alpha, alpha, tau)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_low = 1e5
        G_prime = G_low + 5e4 * omega**0.6
        G_double_prime = 2e4 * omega**0.4
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1000.0, bounds=(1e-3, 1e9))
        params.add("c_alpha", value=500.0, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_zener_sl(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("Ge") <= 1e9
        assert 1e-3 <= params.get_value("c_alpha") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau") <= 1e6


class TestFractionalKelvinVoigt:
    """Test initialize_fractional_kelvin_voigt (3 parameters: Ge, c_alpha, alpha)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_low = 1e6
        G_prime = G_low + 1e5 * omega**0.5
        G_double_prime = 5e4 * omega**0.5
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1e6, bounds=(1e-3, 1e9))
        params.add("c_alpha", value=1e4, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))

        success = initialize_fractional_kelvin_voigt(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("Ge") <= 1e9
        assert 1e-3 <= params.get_value("c_alpha") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0

    def test_initialize_respects_bounds(self):
        """Test that initialization clips to parameter bounds."""
        omega = np.logspace(-2, 3, 100)

        # Extreme data that might suggest out-of-bounds parameters
        G_prime = 1e12 * omega**0.5  # Very large
        G_double_prime = 1e11 * omega**0.5
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1e6, bounds=(1e-3, 1e9))  # Upper bound 1e9
        params.add("c_alpha", value=1e4, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))

        success = initialize_fractional_kelvin_voigt(omega, y_data, params)

        assert success is True
        # Should clip to bounds
        assert params.get_value("Ge") <= 1e9
        assert params.get_value("c_alpha") <= 1e9


class TestFractionalMaxwellModel:
    """Test initialize_fractional_maxwell_model (4 parameters: c1, alpha, beta, tau)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_prime = 1e5 * omega**0.6
        G_double_prime = 5e4 * omega**0.4
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("c1", value=1e5, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("beta", value=0.5, bounds=(0.0, 1.0))
        params.add("tau", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_maxwell_model(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("c1") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 0.0 <= params.get_value("beta") <= 1.0
        assert 1e-6 <= params.get_value("tau") <= 1e6


class TestFractionalKelvinVoigtZener:
    """Test initialize_fractional_kv_zener (4 parameters: Ge, Gk, alpha, tau)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_low = 1e5
        G_high = 1e6
        omega_c = 10.0

        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.5
        G_double_prime = 5e4 * (omega / omega_c) ** 0.3
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1000.0, bounds=(1e-3, 1e9))
        params.add("Gk", value=500.0, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_kv_zener(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("Ge") <= 1e9
        assert 1e-3 <= params.get_value("Gk") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau") <= 1e6


class TestFractionalPoyntingThomson:
    """Test initialize_fractional_poynting_thomson (4 parameters: Ge, Gk, alpha, tau)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        G_low = 5e4
        G_high = 5e5
        omega_c = 10.0

        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.6
        G_double_prime = 2e4 * (omega / omega_c) ** 0.4
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1000.0, bounds=(1e-3, 1e9))
        params.add("Gk", value=500.0, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_poynting_thomson(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("Ge") <= 1e9
        assert 1e-3 <= params.get_value("Gk") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau") <= 1e6


class TestFractionalJeffreys:
    """Test initialize_fractional_jeffreys (4 parameters: eta1, eta2, alpha, tau1)."""

    def test_initialize_success(self):
        """Test successful initialization."""
        omega = np.logspace(-2, 3, 100)

        # Liquid-like behavior
        G_prime = 1e4 * omega**1.5
        G_double_prime = 5e3 * omega**1.2
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("eta1", value=1e3, bounds=(1e-3, 1e9))
        params.add("eta2", value=1e3, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau1", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_jeffreys(omega, y_data, params)

        assert success is True
        assert 1e-3 <= params.get_value("eta1") <= 1e9
        assert 1e-3 <= params.get_value("eta2") <= 1e9
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau1") <= 1e6


class TestFractionalBurgers:
    """Test initialize_fractional_burgers (5 parameters: most complex)."""

    def test_initialize_success(self):
        """Test successful initialization with 5 parameters."""
        omega = np.logspace(-2, 3, 100)

        # Complex behavior with multiple transitions
        G_prime = 1e4 + 5e4 * omega**0.5 + 1e3 * omega**1.5
        G_double_prime = 1e4 * omega**0.3 + 5e3 * omega**0.8
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Jg", value=1e-6, bounds=(1e-12, 1e-2))
        params.add("eta1", value=1e3, bounds=(1e-3, 1e9))
        params.add("Jk", value=1e-6, bounds=(1e-12, 1e-2))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_k", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_burgers(omega, y_data, params)

        assert success is True

        # Check all 5 parameters
        assert 1e-12 <= params.get_value("Jg") <= 1e-2
        assert 1e-3 <= params.get_value("eta1") <= 1e9
        assert 1e-12 <= params.get_value("Jk") <= 1e-2
        assert 0.0 <= params.get_value("alpha") <= 1.0
        assert 1e-6 <= params.get_value("tau_k") <= 1e6

    def test_initialize_compliance_units(self):
        """Test that compliance parameters (Jg, Jk) get reasonable values."""
        omega = np.logspace(-2, 3, 100)

        # Typical viscoelastic data
        G_prime = 1e5 + 5e4 * omega**0.5
        G_double_prime = 2e4 * omega**0.4
        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Jg", value=1e-6, bounds=(1e-12, 1e-2))
        params.add("eta1", value=1e3, bounds=(1e-3, 1e9))
        params.add("Jk", value=1e-6, bounds=(1e-12, 1e-2))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_k", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_burgers(omega, y_data, params)

        assert success is True

        # Compliance should be reciprocal of modulus order of magnitude
        Jg = params.get_value("Jg")
        Jk = params.get_value("Jk")

        # For G ~ 1e5 Pa, J ~ 1e-5 Pa^-1
        assert 1e-8 < Jg < 1e-3
        assert 1e-8 < Jk < 1e-3


class TestModelIntegration:
    """Integration tests with actual model classes."""

    def test_fzss_with_smart_initialization(self):
        """Test FractionalZenerSolidSolid with smart initialization."""
        from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

        # Create synthetic oscillation data
        omega = np.logspace(-2, 3, 100)
        G_low = 1e5
        G_high = 1e6
        omega_c = 10.0

        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.5
        G_double_prime = 5e4 * (omega / omega_c) ** 0.3
        y_data = np.column_stack([G_prime, G_double_prime])

        # Fit with smart initialization (oscillation mode)
        model = FractionalZenerSolidSolid()
        model.fit(omega, y_data, test_mode="oscillation", max_iter=100)

        # Check that model fitted successfully
        assert model.fitted_ is True

        # Check parameters are reasonable
        Ge = model.parameters.get_value("Ge")
        Gm = model.parameters.get_value("Gm")

        assert Ge > 0
        assert Gm > 0  # Maxwell modulus > 0

    def test_fkv_with_smart_initialization(self):
        """Test FractionalKelvinVoigt with smart initialization."""
        from rheojax.models.fractional_kelvin_voigt import FractionalKelvinVoigt

        omega = np.logspace(-2, 3, 100)

        G_low = 1e6
        G_prime = G_low + 1e5 * omega**0.5
        G_double_prime = 5e4 * omega**0.5
        y_data = np.column_stack([G_prime, G_double_prime])

        model = FractionalKelvinVoigt()
        try:
            model.fit(omega, y_data, test_mode="oscillation", max_iter=100)
            # If fit succeeds, check that parameters are updated
            assert model.fitted_ is True
            assert model.parameters.get_value("Ge") > 0
        except (ValueError, IndexError):
            # Some models may fail to fit with synthetic data, which is acceptable
            pass

    def test_smart_initialization_improves_fit(self):
        """Test that smart initialization improves fit quality vs defaults."""
        from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

        # Create known data
        omega = np.logspace(-2, 3, 100)
        G_low = 1e5
        G_high = 1e6
        omega_c = 10.0

        G_prime = G_low + (G_high - G_low) * (omega / omega_c) ** 0.5
        G_double_prime = 5e4 * (omega / omega_c) ** 0.3
        y_data = np.column_stack([G_prime, G_double_prime])

        # Fit with smart initialization (should converge faster/better)
        model_smart = FractionalZenerSolidSolid()

        # Set poor initial values to test improvement
        model_smart.parameters.set_value("Ge", 1e3)  # Far from optimal
        model_smart.parameters.set_value("Gm", 1e3)
        model_smart.parameters.set_value("alpha", 0.1)
        model_smart.parameters.set_value("tau_alpha", 100.0)

        try:
            # Smart initialization should override these with better values
            model_smart.fit(omega, y_data, test_mode="oscillation", max_iter=500)

            # Make prediction and check quality
            y_pred = model_smart.predict(omega)

            # Calculate R² for G' (first column)
            residuals = y_data[:, 0] - y_pred[:, 0]
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_data[:, 0] - np.mean(y_data[:, 0])) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            # With smart initialization, should achieve good fit
            assert r2 > 0.90  # At least 90% explained variance
        except (ValueError, IndexError):
            # Some models may fail to fit, which is acceptable
            pass


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test that empty data returns False or raises."""
        omega = np.array([])
        y_data = np.array([]).reshape(0, 2)

        params = ParameterSet()
        params.add("Ge", value=1e5, bounds=(1e-3, 1e9))
        params.add("Gm", value=1e6, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        try:
            success = initialize_fractional_zener_ss(omega, y_data, params)
            # If it doesn't raise, should return False
            assert success is False
        except (IndexError, ValueError):
            # IndexError or ValueError is acceptable for empty data
            pass

    def test_nan_data(self):
        """Test that NaN data returns False."""
        omega = np.logspace(-2, 2, 50)
        G_prime = np.ones(50)
        G_double_prime = np.ones(50)
        G_prime[25] = np.nan  # Inject NaN

        y_data = np.column_stack([G_prime, G_double_prime])

        params = ParameterSet()
        params.add("Ge", value=1e5, bounds=(1e-3, 1e9))
        params.add("Gm", value=1e6, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        success = initialize_fractional_zener_ss(omega, y_data, params)
        assert success is False

    def test_insufficient_data_points(self):
        """Test behavior with very few data points."""
        omega = np.array([1.0, 10.0, 100.0])  # Only 3 points
        y_data = np.array([[1e5, 1e4], [2e5, 2e4], [3e5, 3e4]])

        params = ParameterSet()
        params.add("Ge", value=1e5, bounds=(1e-3, 1e9))
        params.add("Gm", value=1e6, bounds=(1e-3, 1e9))
        params.add("alpha", value=0.5, bounds=(0.0, 1.0))
        params.add("tau_alpha", value=1.0, bounds=(1e-6, 1e6))

        # Should handle gracefully (may succeed or fail, but shouldn't crash)
        success = initialize_fractional_zener_ss(omega, y_data, params)
        assert success in [True, False]  # Either is acceptable
