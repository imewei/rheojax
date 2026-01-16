"""Tests for STZ Oscillatory protocols (SAOS and LAOS).

Tests cover SAOS linear viscoelastic response and LAOS nonlinear dynamics.
Follows the 2-8 test rule per task group.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.models.stz import STZConventional


@pytest.mark.unit
class TestSTZOscillatory:
    """Test suite for STZ SAOS and LAOS protocols."""

    def test_saos_prediction_shape(self):
        """Test SAOS prediction returns correct shape [G', G'']."""
        model = STZConventional(variant="standard")

        # Use default parameters (physically consistent)
        omega = np.logspace(8, 14, 20)  # High frequency range matching tau0 ~ 1e-12

        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G0"),
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        G_star = np.array(G_star)

        # Shape should be (n_omega, 2)
        assert G_star.shape == (len(omega), 2)

        # G' and G'' should be finite and positive
        assert np.all(np.isfinite(G_star))
        assert np.all(G_star >= 0)

    def test_saos_maxwell_like_behavior(self):
        """Test SAOS shows Maxwell-like viscoelastic behavior."""
        model = STZConventional(variant="standard")

        G0 = model.parameters.get_value("G0")

        # Wide frequency range around relaxation time
        omega = np.logspace(6, 16, 50)

        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            G0,
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        G_star = np.array(G_star)
        G_prime = G_star[:, 0]
        G_double_prime = G_star[:, 1]

        # At high frequency, G' should approach G0
        assert G_prime[-1] > 0.5 * G0

        # G'' should have a peak (Maxwell behavior)
        # Check that G'' is non-monotonic (has a maximum)
        max_idx = np.argmax(G_double_prime)
        assert max_idx > 0 and max_idx < len(omega) - 1

    def test_saos_low_frequency_limit(self):
        """Test SAOS at low frequency produces low modulus."""
        model = STZConventional(variant="standard")

        # Very low frequency (relative to molecular time)
        omega_low = np.array([1e6])  # Still high but << 1/tau0

        G_star = model._predict_saos_jit(
            jnp.asarray(omega_low),
            model.parameters.get_value("G0"),
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        G_star = np.array(G_star)

        # G' should be less than G0 at low frequency
        assert G_star[0, 0] < model.parameters.get_value("G0")

    def test_saos_high_frequency_limit(self):
        """Test SAOS at high frequency approaches G0."""
        model = STZConventional(variant="standard")

        G0 = model.parameters.get_value("G0")

        # Very high frequency
        omega_high = np.array([1e16])

        G_star = model._predict_saos_jit(
            jnp.asarray(omega_high),
            G0,
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        G_star = np.array(G_star)

        # G' should approach G0 at high frequency
        assert G_star[0, 0] > 0.9 * G0

        # G'' should be small relative to G' at high frequency
        assert G_star[0, 1] < G_star[0, 0]

    @pytest.mark.slow
    def test_laos_simulation_runs(self):
        """Test LAOS simulation runs without errors."""
        model = STZConventional(variant="standard")

        # Set tau0=1e-9 for less stiff dynamics
        model.parameters.set_value("tau0", 1e-9)

        # Use parameters compatible with explicit solver
        gamma_0 = 0.1
        omega = 1e8  # 100 MHz - compatible with tau0=1e-9

        # Only run 1 cycle with few points to keep step count manageable
        period = 2 * np.pi / omega
        t = np.linspace(0, period, 64)

        p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}

        strain, stress = model._simulate_laos_internal(
            jnp.asarray(t), p_values, gamma_0, omega, model.variant
        )

        strain = np.array(strain)
        stress = np.array(stress)

        # Both arrays should be finite
        assert np.all(np.isfinite(strain))
        assert np.all(np.isfinite(stress))

        # Strain should oscillate with correct amplitude
        assert np.isclose(np.max(np.abs(strain)), gamma_0, rtol=0.05)

    def test_harmonic_extraction_synthetic(self):
        """Test FFT harmonic extraction on synthetic signal."""
        model = STZConventional(variant="standard")

        n_points = 256

        # Create synthetic signal with known harmonics
        t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        I_1_true = 100.0
        I_3_true = 10.0
        stress = I_1_true * np.sin(t) + I_3_true * np.sin(3 * t)

        harmonics = model.extract_harmonics(stress, n_points)

        # Check extraction accuracy
        assert np.isclose(harmonics["I_1"], I_1_true, rtol=0.01)
        assert np.isclose(harmonics["I_3"], I_3_true, rtol=0.01)
        assert np.isclose(harmonics["I_3_I_1"], 0.1, rtol=0.01)

    def test_chi_inf_effect_on_saos(self):
        """Test that chi_inf affects the relaxation behavior."""
        model = STZConventional(variant="standard")

        omega = np.logspace(8, 14, 30)

        # Low chi_inf (more glassy)
        model.parameters.set_value("chi_inf", 0.05)
        G_star_low = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G0"),
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        # High chi_inf (more liquid-like)
        model.parameters.set_value("chi_inf", 0.4)
        G_star_high = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G0"),
            model.parameters.get_value("sigma_y"),
            model.parameters.get_value("chi_inf"),
            model.parameters.get_value("tau0"),
            model.parameters.get_value("epsilon0"),
        )

        G_star_low = np.array(G_star_low)
        G_star_high = np.array(G_star_high)

        # Different chi_inf should give different G'' peaks
        max_idx_low = np.argmax(G_star_low[:, 1])
        max_idx_high = np.argmax(G_star_high[:, 1])

        # Peak positions should differ
        assert max_idx_low != max_idx_high
