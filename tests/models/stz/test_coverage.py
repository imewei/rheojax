from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from rheojax.models.stz.conventional import STZConventional


class TestSTZCoverage:
    def setup_method(self):
        self.model = STZConventional()
        self.t = np.linspace(0, 10, 100)
        self.y = np.sin(self.t)
        self.omega = np.logspace(-1, 1, 10)
        # Create complex modulus for SAOS
        self.G_star = np.ones_like(self.omega) + 1j * np.ones_like(self.omega)

    def test_fit_invalid_test_mode(self):
        """Test _fit raising ValueError for invalid test_mode."""
        # Case 1: test_mode not provided (and no internal test_mode set)
        with pytest.raises(ValueError, match="test_mode must be specified"):
            self.model.fit(self.t, self.y)

        # Case 2: Unsupported test_mode
        with pytest.raises(ValueError, match="Unsupported test_mode"):
            self.model.fit(self.t, self.y, test_mode="invalid_mode")

    def test_fit_transient_missing_kwargs(self):
        """Test _fit_transient raising ValueError when missing required kwargs."""
        # Case 1: startup missing gamma_dot
        with pytest.raises(ValueError, match="startup mode requires gamma_dot"):
            self.model.fit(self.t, self.y, test_mode="startup")

        # Case 2: creep missing sigma_applied
        with pytest.raises(ValueError, match="creep mode requires sigma_applied"):
            self.model.fit(self.t, self.y, test_mode="creep")

    def test_fit_oscillation_branching(self):
        """Test _fit_oscillation handling laos vs saos logic branches."""
        # Use mocks to check which method is called
        with (
            patch.object(self.model, "_fit_laos_mode") as mock_laos,
            patch.object(self.model, "_fit_saos_mode") as mock_saos,
        ):

            # Case 1: gamma_0 > 0.01 -> LAOS
            # We must pass omega as well for LAOS
            self.model.fit(
                self.omega, self.y, test_mode="oscillation", gamma_0=0.1, omega=1.0
            )
            mock_laos.assert_called_once()
            mock_saos.assert_not_called()

            # Reset
            mock_laos.reset_mock()
            mock_saos.reset_mock()

            # Case 2: gamma_0 <= 0.01 -> SAOS
            self.model.fit(
                self.omega, self.G_star, test_mode="oscillation", gamma_0=0.001
            )
            mock_saos.assert_called_once()
            mock_laos.assert_not_called()

            # Reset
            mock_laos.reset_mock()
            mock_saos.reset_mock()

            # Case 3: gamma_0 missing -> SAOS
            self.model.fit(self.omega, self.G_star, test_mode="oscillation")
            mock_saos.assert_called_once()
            mock_laos.assert_not_called()

    def test_fit_laos_mode_missing_args(self):
        """Test LAOS mode raising ValueError when missing gamma_0 or omega.

        This checks the validation in model_function (used for Bayesian) and predict.
        """
        # Default variant is "standard", which has 8 parameters:
        # G0, sigma_y, chi_inf, tau0, epsilon0, c0, ez, tau_beta
        params = [1e9, 1e6, 0.1, 1e-12, 0.1, 1.0, 1.0, 1.0]

        # Ensure internal state is None
        self.model._gamma_0 = None
        self.model._omega_laos = None

        # Check model_function (Bayesian path)
        with pytest.raises(ValueError, match="LAOS mode requires gamma_0 and omega"):
            self.model.model_function(self.t, params, test_mode="laos")

        # Also check setting one but not other
        self.model._gamma_0 = 0.1
        with pytest.raises(ValueError, match="LAOS mode requires gamma_0 and omega"):
            self.model.model_function(self.t, params, test_mode="laos")

    def test_predict_missing_test_mode(self):
        """Test predict raising ValueError when test_mode is missing."""
        # Specifically target _predict_transient which has the explicit check
        # as the main _predict might return zeros for None

        self.model._test_mode = None
        with pytest.raises(ValueError, match="Test mode not specified"):
            self.model._predict_transient(self.t)

        # Also check the LAOS specific check in _predict
        self.model.fitted_ = True
        self.model._test_mode = "laos"
        self.model._gamma_0 = None
        with pytest.raises(
            ValueError, match="LAOS prediction requires gamma_0 and omega"
        ):
            self.model.predict(self.t)

    def test_extract_harmonics_zero_fundamental(self):
        """Test extract_harmonics with zero fundamental (edge case)."""
        # Create a signal with 0 amplitude (all zeros)
        stress = np.zeros(512)
        harmonics = self.model.extract_harmonics(stress, n_points_per_cycle=256)

        assert harmonics["I_1"] == 0.0
        assert harmonics["I_3"] == 0.0
        assert harmonics["I_5"] == 0.0
        assert harmonics["I_3_I_1"] == 0.0
        assert harmonics["I_5_I_1"] == 0.0

        # Check DC signal (fundamental is 0)
        stress_dc = np.ones(512)
        harmonics_dc = self.model.extract_harmonics(stress_dc, n_points_per_cycle=256)
        # I_1 should be near zero (DC component is at index 0, fundamental at index 1)
        assert harmonics_dc["I_1"] == 0.0
        assert harmonics_dc["I_3_I_1"] == 0.0
