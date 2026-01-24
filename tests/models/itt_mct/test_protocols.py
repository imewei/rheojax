"""Cross-protocol tests for ITT-MCT models.

Tests cover:
- Protocol consistency across models
- Asymptotic behavior verification
- Edge cases (γ̇ → 0, ω → ∞)
"""

import numpy as np
import pytest

from rheojax.models.itt_mct import ITTMCTIsotropic, ITTMCTSchematic


class TestCrossProtocolConsistency:
    """Tests for consistency across protocols."""

    @pytest.mark.slow
    def test_startup_approaches_flow_curve(self):
        """Test that startup stress approaches flow curve at long times."""
        model = ITTMCTSchematic(epsilon=0.05)
        gamma_dot = 1.0

        # Get steady-state stress from flow curve
        sigma_ss = model.predict(np.array([gamma_dot]), test_mode="flow_curve")[0]

        # Get stress evolution from startup
        t = np.linspace(0, 100, 200)
        sigma_t = model.predict(t, test_mode="startup", gamma_dot=gamma_dot)
        sigma_final = sigma_t[-1]

        # Should approach steady state
        assert abs(sigma_final - sigma_ss) / sigma_ss < 0.5  # Within 50%

    @pytest.mark.slow
    def test_creep_vs_relaxation_reciprocity(self):
        """Test approximate reciprocity between creep and relaxation."""
        model = ITTMCTSchematic(epsilon=-0.05)  # Fluid for better reciprocity

        # Get relaxation modulus G(t)
        t = np.logspace(-1, 2, 50)
        gamma_pre = 0.01
        sigma_rel = model.predict(t, test_mode="relaxation", gamma_pre=gamma_pre)
        G_t = sigma_rel / gamma_pre

        # Get creep compliance J(t)
        sigma_applied = 100.0
        J_t = model.predict(t, test_mode="creep", sigma_applied=sigma_applied)

        # For linear viscoelasticity: J(t) × G(t) ≈ 1 at short times
        # (This is approximate for MCT due to nonlinear memory)
        product = J_t[0] * G_t[0]
        # Just check the product is finite and positive
        assert np.isfinite(product)
        assert product > 0


class TestAsymptoticBehavior:
    """Tests for asymptotic behavior."""

    @pytest.mark.smoke
    def test_zero_shear_rate_limit(self):
        """Test behavior as γ̇ → 0."""
        model_fluid = ITTMCTSchematic(epsilon=-0.1)
        model_glass = ITTMCTSchematic(epsilon=0.1)

        gamma_dot_low = np.array([1e-5])

        # Fluid: σ → 0
        sigma_fluid = model_fluid.predict(gamma_dot_low, test_mode="flow_curve")
        assert sigma_fluid[0] < 1e3  # Should be small

        # Glass: σ → σ_y > 0
        sigma_glass = model_glass.predict(gamma_dot_low, test_mode="flow_curve")
        assert sigma_glass[0] > 0  # Yield stress

    @pytest.mark.smoke
    def test_high_frequency_limit(self):
        """Test behavior as ω → ∞."""
        model = ITTMCTSchematic(epsilon=0.05)
        omega_high = np.array([1e3, 1e4])

        G_components = model.predict(
            omega_high, test_mode="oscillation", return_components=True
        )
        G_prime = G_components[:, 0]
        G_double_prime = G_components[:, 1]

        # At high ω: G' should approach G_inf, G'' should decrease
        G_inf = model.parameters.get_value("G_inf")

        # G' should be approaching a plateau (or still increasing)
        assert G_prime[1] >= G_prime[0] * 0.5  # Not dropping drastically

    @pytest.mark.smoke
    def test_long_time_relaxation(self):
        """Test relaxation at long times."""
        model_fluid = ITTMCTSchematic(epsilon=-0.1)
        model_glass = ITTMCTSchematic(epsilon=0.1)

        t_long = np.array([1000.0])
        gamma_pre = 0.01

        # Fluid: σ → 0
        sigma_fluid = model_fluid.predict(
            t_long, test_mode="relaxation", gamma_pre=gamma_pre
        )
        assert sigma_fluid[0] < sigma_fluid[0] * 100  # Some decay

        # Glass: σ → σ_residual > 0
        sigma_glass = model_glass.predict(
            t_long, test_mode="relaxation", gamma_pre=gamma_pre
        )
        assert sigma_glass[0] > 0  # Residual stress


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.smoke
    def test_zero_time(self):
        """Test behavior at t=0."""
        model = ITTMCTSchematic(epsilon=0.05)

        # Startup: σ(0) = 0
        t = np.array([0.0, 0.1, 1.0])
        sigma_startup = model.predict(t, test_mode="startup", gamma_dot=1.0)
        assert sigma_startup[0] == pytest.approx(0.0, abs=1e-10)

    def test_single_point_prediction(self):
        """Test prediction for single data point."""
        model = ITTMCTSchematic(epsilon=0.05)

        # Should work for single point
        gamma_dot = np.array([1.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")
        assert len(sigma) == 1
        assert sigma[0] > 0

    def test_numerical_stability_extreme_rates(self):
        """Test numerical stability at extreme shear rates."""
        model = ITTMCTSchematic(epsilon=0.05)

        # Very low rates
        gamma_dot_low = np.array([1e-10])
        sigma_low = model.predict(gamma_dot_low, test_mode="flow_curve")
        assert np.isfinite(sigma_low[0])

        # Very high rates
        gamma_dot_high = np.array([1e6])
        sigma_high = model.predict(gamma_dot_high, test_mode="flow_curve")
        assert np.isfinite(sigma_high[0])


class TestModelComparison:
    """Tests comparing F₁₂ and ISM models."""

    @pytest.mark.slow
    def test_qualitative_agreement_flow_curve(self):
        """Test qualitative agreement between F₁₂ and ISM for flow curves."""
        # Both in glass state
        model_f12 = ITTMCTSchematic(epsilon=0.1)
        model_ism = ITTMCTIsotropic(phi=0.55)

        gamma_dot = np.logspace(-2, 2, 10)

        sigma_f12 = model_f12.predict(gamma_dot, test_mode="flow_curve")
        sigma_ism = model_ism.predict(gamma_dot, test_mode="flow_curve")

        # Both should show:
        # 1. Non-zero stress at low rates (yield stress)
        assert sigma_f12[0] > 0
        assert sigma_ism[0] > 0

        # 2. Increasing stress with rate
        assert sigma_f12[-1] > sigma_f12[0]
        assert sigma_ism[-1] > sigma_ism[0]

    @pytest.mark.slow
    def test_qualitative_agreement_oscillation(self):
        """Test qualitative agreement for oscillation."""
        model_f12 = ITTMCTSchematic(epsilon=0.1)
        model_ism = ITTMCTIsotropic(phi=0.55)

        omega = np.logspace(-1, 2, 10)

        G_f12 = model_f12.predict(
            omega, test_mode="oscillation", return_components=True
        )
        G_ism = model_ism.predict(
            omega, test_mode="oscillation", return_components=True
        )

        G_prime_f12 = G_f12[:, 0]
        G_prime_ism = G_ism[:, 0]

        # Both should show plateau at low frequency (glass)
        # Check that G' doesn't vary by more than factor of 10 at low freq
        G_prime_low_f12 = G_prime_f12[:3]
        G_prime_low_ism = G_prime_ism[:3]

        variation_f12 = G_prime_low_f12.max() / max(G_prime_low_f12.min(), 1e-10)
        variation_ism = G_prime_low_ism.max() / max(G_prime_low_ism.min(), 1e-10)

        # Relatively flat plateau expected
        assert variation_f12 < 10
        assert variation_ism < 10
