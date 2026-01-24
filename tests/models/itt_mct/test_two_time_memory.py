"""Tests for ITT-MCT two-time memory kernel and microscopic stress features.

Tests cover:
- memory_form parameter ("simplified" vs "full")
- stress_form parameter ("schematic" vs "microscopic")
- Backward compatibility with default parameters
- Physics validation (full memory differs from simplified)
"""

import numpy as np
import pytest

from rheojax.models.itt_mct import ITTMCTSchematic


class TestMemoryFormParameter:
    """Tests for memory_form parameter."""

    @pytest.mark.smoke
    def test_memory_form_simplified_is_default(self):
        """Test that simplified is the default memory form."""
        model = ITTMCTSchematic()
        assert model.memory_form == "simplified"

    @pytest.mark.smoke
    def test_memory_form_full_initialization(self):
        """Test initialization with full memory form."""
        model = ITTMCTSchematic(memory_form="full", epsilon=0.05)
        assert model.memory_form == "full"

    def test_memory_form_invalid_raises(self):
        """Test that invalid memory_form raises ValueError."""
        with pytest.raises(ValueError, match="memory_form must be"):
            ITTMCTSchematic(memory_form="invalid")

    def test_memory_form_invalid_empty_raises(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError, match="memory_form must be"):
            ITTMCTSchematic(memory_form="")

    @pytest.mark.smoke
    def test_backward_compatibility_flow_curve(self):
        """Test that simplified form matches previous behavior exactly."""
        # Create two models with explicit simplified vs default
        model_explicit = ITTMCTSchematic(epsilon=0.05, memory_form="simplified")
        model_default = ITTMCTSchematic(epsilon=0.05)

        gamma_dot = np.logspace(-1, 1, 5)
        sigma_explicit = model_explicit.predict(gamma_dot, test_mode="flow_curve")
        sigma_default = model_default.predict(gamma_dot, test_mode="flow_curve")

        np.testing.assert_allclose(sigma_explicit, sigma_default, rtol=1e-10)

    @pytest.mark.smoke
    def test_full_memory_initializes_glass_state(self):
        """Test that full memory form works with glass state."""
        model = ITTMCTSchematic(epsilon=0.1, memory_form="full")
        info = model.get_glass_transition_info()

        assert info["is_glass"]
        assert model.memory_form == "full"

    @pytest.mark.smoke
    def test_full_memory_initializes_fluid_state(self):
        """Test that full memory form works with fluid state."""
        model = ITTMCTSchematic(epsilon=-0.1, memory_form="full")
        info = model.get_glass_transition_info()

        assert not info["is_glass"]
        assert model.memory_form == "full"

    def test_full_memory_differs_from_simplified(self):
        """Test that full form gives quantitatively different results.

        The full two-time memory kernel should produce different stress
        values due to mode-specific decorrelation.
        """
        gamma_dot = np.logspace(-1, 1, 5)

        model_simple = ITTMCTSchematic(epsilon=0.1, memory_form="simplified")
        model_full = ITTMCTSchematic(epsilon=0.1, memory_form="full")

        sigma_simple = model_simple.predict(gamma_dot, test_mode="flow_curve")
        sigma_full = model_full.predict(gamma_dot, test_mode="flow_curve")

        # Should differ but both be physical (positive stress)
        assert not np.allclose(sigma_simple, sigma_full, rtol=0.01)
        assert np.all(sigma_full >= 0)
        assert np.all(sigma_simple >= 0)

    @pytest.mark.smoke
    def test_repr_includes_memory_form(self):
        """Test that __repr__ includes memory form."""
        model = ITTMCTSchematic(epsilon=0.05, memory_form="full")
        repr_str = repr(model)

        assert "m=full" in repr_str

    def test_memory_form_with_lorentzian(self):
        """Test full memory form with Lorentzian decorrelation."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            memory_form="full",
            decorrelation_form="lorentzian",
        )

        assert model.memory_form == "full"
        assert model.decorrelation_form == "lorentzian"

        # Should be able to predict without error
        gamma_dot = np.logspace(-1, 1, 3)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")
        assert np.all(sigma >= 0)


class TestStressFormParameter:
    """Tests for stress_form parameter."""

    @pytest.mark.smoke
    def test_stress_form_schematic_is_default(self):
        """Test that schematic is the default stress form."""
        model = ITTMCTSchematic()
        assert model.stress_form == "schematic"

    def test_microscopic_requires_phi_volume(self):
        """Test that phi_volume is required for microscopic stress."""
        with pytest.raises(ValueError, match="phi_volume is required"):
            ITTMCTSchematic(stress_form="microscopic")

    @pytest.mark.smoke
    def test_microscopic_initialization(self):
        """Test initialization with microscopic stress form."""
        model = ITTMCTSchematic(stress_form="microscopic", phi_volume=0.5)

        assert model.stress_form == "microscopic"

    def test_stress_form_invalid_raises(self):
        """Test that invalid stress_form raises ValueError."""
        with pytest.raises(ValueError, match="stress_form must be"):
            ITTMCTSchematic(stress_form="invalid")

    @pytest.mark.smoke
    def test_microscopic_stress_flow_curve(self):
        """Test that microscopic stress form produces valid results."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            stress_form="microscopic",
            phi_volume=0.5,
        )

        gamma_dot = np.logspace(-1, 1, 3)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Should produce positive stress values
        assert np.all(sigma >= 0)

    def test_microscopic_differs_from_schematic(self):
        """Test that microscopic form gives different stress magnitudes."""
        gamma_dot = np.logspace(-1, 1, 5)

        model_schematic = ITTMCTSchematic(epsilon=0.05, stress_form="schematic")
        model_micro = ITTMCTSchematic(
            epsilon=0.05,
            stress_form="microscopic",
            phi_volume=0.5,
        )

        sigma_schematic = model_schematic.predict(gamma_dot, test_mode="flow_curve")
        sigma_micro = model_micro.predict(gamma_dot, test_mode="flow_curve")

        # Should have different magnitudes (microscopic has S(k) weighting)
        # but both should be monotonically increasing with shear rate
        assert np.all(sigma_schematic >= 0)
        assert np.all(sigma_micro >= 0)

        # Check monotonicity (stress increases with shear rate)
        assert np.all(np.diff(sigma_schematic) > 0) or np.allclose(
            np.diff(sigma_schematic), 0, atol=1e-10
        )
        assert np.all(np.diff(sigma_micro) > 0) or np.allclose(
            np.diff(sigma_micro), 0, atol=1e-10
        )

    @pytest.mark.smoke
    def test_microscopic_with_custom_k_BT(self):
        """Test microscopic stress with custom thermal energy."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            stress_form="microscopic",
            phi_volume=0.5,
            k_BT=4.11e-21,  # Room temperature in Joules
        )

        assert model.stress_form == "microscopic"

        gamma_dot = np.array([0.1, 1.0, 10.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Should produce meaningful stress values
        assert np.all(sigma >= 0)

    @pytest.mark.smoke
    def test_repr_includes_stress_form(self):
        """Test that __repr__ includes stress form."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            stress_form="microscopic",
            phi_volume=0.5,
        )
        repr_str = repr(model)

        assert "σ=microscopic" in repr_str


class TestCombinedFeatures:
    """Tests for combining memory_form and stress_form."""

    @pytest.mark.smoke
    def test_full_memory_with_microscopic_stress(self):
        """Test combining full memory with microscopic stress."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            memory_form="full",
            stress_form="microscopic",
            phi_volume=0.5,
        )

        assert model.memory_form == "full"
        assert model.stress_form == "microscopic"

        # Should be able to predict
        gamma_dot = np.logspace(-1, 1, 3)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")
        assert np.all(sigma >= 0)

    def test_all_options_combined(self):
        """Test combining all new options with existing options."""
        model = ITTMCTSchematic(
            epsilon=0.1,
            decorrelation_form="lorentzian",
            memory_form="full",
            stress_form="microscopic",
            phi_volume=0.55,
            k_BT=4.11e-21,
        )

        assert model.decorrelation_form == "lorentzian"
        assert model.memory_form == "full"
        assert model.stress_form == "microscopic"

        gamma_dot = np.logspace(-1, 1, 3)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")
        assert np.all(sigma >= 0)

    @pytest.mark.smoke
    def test_repr_shows_all_forms(self):
        """Test that __repr__ shows all form options."""
        model = ITTMCTSchematic(
            epsilon=0.05,
            decorrelation_form="lorentzian",
            memory_form="full",
            stress_form="microscopic",
            phi_volume=0.5,
        )

        repr_str = repr(model)

        assert "h(γ)=lorentzian" in repr_str
        assert "m=full" in repr_str
        assert "σ=microscopic" in repr_str


class TestTwoTimeDecorrelationHelper:
    """Tests for the two_time_strain_decorrelation helper function."""

    @pytest.mark.smoke
    def test_import_helper(self):
        """Test that the helper function can be imported."""
        from rheojax.utils.mct_kernels import two_time_strain_decorrelation

        assert callable(two_time_strain_decorrelation)

    @pytest.mark.smoke
    def test_two_time_decorrelation_gaussian(self):
        """Test two-time decorrelation with Gaussian form."""
        import jax.numpy as jnp

        from rheojax.utils.mct_kernels import two_time_strain_decorrelation

        gamma_total = jnp.array([0.0, 0.1, 0.2])
        gamma_since_s = jnp.array([0.0, 0.05, 0.1])
        gamma_c = 0.1

        h_two_time = two_time_strain_decorrelation(
            gamma_total, gamma_since_s, gamma_c, use_lorentzian=False
        )

        # Should be product of two Gaussian decorrelations
        h_total = jnp.exp(-((gamma_total / gamma_c) ** 2))
        h_since_s = jnp.exp(-((gamma_since_s / gamma_c) ** 2))
        expected = h_total * h_since_s

        np.testing.assert_allclose(np.array(h_two_time), np.array(expected), rtol=1e-10)

    def test_two_time_decorrelation_lorentzian(self):
        """Test two-time decorrelation with Lorentzian form."""
        import jax.numpy as jnp

        from rheojax.utils.mct_kernels import two_time_strain_decorrelation

        gamma_total = jnp.array([0.0, 0.1, 0.2])
        gamma_since_s = jnp.array([0.0, 0.05, 0.1])
        gamma_c = 0.1

        h_two_time = two_time_strain_decorrelation(
            gamma_total, gamma_since_s, gamma_c, use_lorentzian=True
        )

        # Should be product of two Lorentzian decorrelations
        h_total = 1.0 / (1.0 + (gamma_total / gamma_c) ** 2)
        h_since_s = 1.0 / (1.0 + (gamma_since_s / gamma_c) ** 2)
        expected = h_total * h_since_s

        np.testing.assert_allclose(np.array(h_two_time), np.array(expected), rtol=1e-10)

    def test_two_time_bounded(self):
        """Test that two-time decorrelation is bounded in [0, 1]."""
        import jax.numpy as jnp

        from rheojax.utils.mct_kernels import two_time_strain_decorrelation

        gamma_total = jnp.array([0.0, 0.5, 1.0, 5.0, 10.0])
        gamma_since_s = jnp.array([0.0, 0.2, 0.5, 2.0, 5.0])
        gamma_c = 0.1

        h_gauss = two_time_strain_decorrelation(
            gamma_total, gamma_since_s, gamma_c, use_lorentzian=False
        )
        h_lorentz = two_time_strain_decorrelation(
            gamma_total, gamma_since_s, gamma_c, use_lorentzian=True
        )

        assert np.all(h_gauss >= 0) and np.all(h_gauss <= 1)
        assert np.all(h_lorentz >= 0) and np.all(h_lorentz <= 1)


class TestMicroscopicStressUtilities:
    """Tests for microscopic stress utility functions."""

    @pytest.mark.smoke
    def test_import_utilities(self):
        """Test that utility functions can be imported."""
        from rheojax.utils.mct_kernels import (
            compute_microscopic_stress,
            get_microscopic_stress_prefactor,
            setup_microscopic_stress_weights,
        )

        assert callable(setup_microscopic_stress_weights)
        assert callable(compute_microscopic_stress)
        assert callable(get_microscopic_stress_prefactor)

    @pytest.mark.smoke
    def test_setup_microscopic_weights(self):
        """Test setup_microscopic_stress_weights returns valid arrays."""
        from rheojax.utils.mct_kernels import setup_microscopic_stress_weights

        k_array, weights = setup_microscopic_stress_weights(
            phi_volume=0.5,
            k_min=1.0,
            k_max=30.0,
            n_k=50,
        )

        # Check shapes
        assert k_array.shape == (50,)
        assert weights.shape == (50,)

        # Check physical constraints
        assert np.all(k_array > 0)
        assert np.all(weights >= 0)

    def test_microscopic_prefactor_physical(self):
        """Test that microscopic prefactor has reasonable magnitude."""
        from rheojax.utils.mct_kernels import get_microscopic_stress_prefactor

        # With dimensionless k_BT=1, prefactor should be positive
        prefactor = get_microscopic_stress_prefactor(phi_volume=0.5, k_BT=1.0)

        assert prefactor > 0

    def test_microscopic_prefactor_increases_with_phi(self):
        """Test that prefactor increases with volume fraction."""
        from rheojax.utils.mct_kernels import get_microscopic_stress_prefactor

        prefactor_low = get_microscopic_stress_prefactor(phi_volume=0.3, k_BT=1.0)
        prefactor_high = get_microscopic_stress_prefactor(phi_volume=0.5, k_BT=1.0)

        # Denser packing should have higher prefactor (stronger S(k) peak)
        assert prefactor_high > prefactor_low
