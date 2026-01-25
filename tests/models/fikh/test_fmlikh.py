"""Unit tests for FMLIKH (multi-layer FIKH) model.

Tests cover:
- Multi-mode initialization
- Per-mode parameter handling
- Shared vs per-mode fractional order
- Multi-mode predictions
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fikh import FMLIKH

jax, jnp = safe_import_jax()


class TestFMLIKHInitialization:
    """Test FMLIKH model initialization."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test default FMLIKH initialization."""
        model = FMLIKH()
        assert model.n_modes == 3
        assert model.shared_alpha is True
        assert model.include_thermal is True

    @pytest.mark.smoke
    def test_custom_n_modes(self):
        """Test FMLIKH with custom number of modes."""
        for n in [1, 2, 4, 5]:
            model = FMLIKH(n_modes=n, include_thermal=False)
            assert model.n_modes == n

    def test_invalid_n_modes(self):
        """Test FMLIKH rejects invalid n_modes."""
        with pytest.raises(ValueError):
            FMLIKH(n_modes=0)

        with pytest.raises(ValueError):
            FMLIKH(n_modes=-1)

    def test_isothermal_initialization(self):
        """Test isothermal FMLIKH."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        assert "T_ref" not in model.parameters

    def test_repr(self):
        """Test string representation."""
        model = FMLIKH(n_modes=3, include_thermal=False)
        repr_str = repr(model)
        assert "FMLIKH" in repr_str
        assert "n_modes=3" in repr_str


class TestFMLIKHParameters:
    """Test FMLIKH parameter handling."""

    @pytest.mark.smoke
    def test_per_mode_parameters_exist(self):
        """Test per-mode parameters are created."""
        model = FMLIKH(n_modes=3, include_thermal=False)

        for i in range(3):
            assert f"G_{i}" in model.parameters
            assert f"eta_{i}" in model.parameters
            assert f"C_{i}" in model.parameters
            assert f"gamma_dyn_{i}" in model.parameters

    def test_single_mode_params_removed(self):
        """Test single-mode G, eta, C, gamma_dyn are removed."""
        model = FMLIKH(n_modes=2, include_thermal=False)

        assert "G" not in model.parameters
        assert "eta" not in model.parameters
        assert "C" not in model.parameters
        assert "gamma_dyn" not in model.parameters

    def test_shared_alpha_single_parameter(self):
        """Test shared_alpha=True gives single alpha parameter."""
        model = FMLIKH(n_modes=3, shared_alpha=True, include_thermal=False)

        assert "alpha_structure" in model.parameters
        assert "alpha_0" not in model.parameters
        assert "alpha_1" not in model.parameters

    def test_per_mode_alpha_parameters(self):
        """Test shared_alpha=False gives per-mode alpha."""
        model = FMLIKH(n_modes=3, shared_alpha=False, include_thermal=False)

        assert "alpha_structure" not in model.parameters
        for i in range(3):
            assert f"alpha_{i}" in model.parameters

    def test_shared_params_preserved(self):
        """Test shared parameters are preserved."""
        model = FMLIKH(n_modes=2, include_thermal=False)

        shared_params = ["sigma_y0", "delta_sigma_y", "tau_thix", "Gamma", "eta_inf", "mu_p", "m"]
        for param in shared_params:
            assert param in model.parameters, f"Missing shared param: {param}"

    def test_get_mode_params(self):
        """Test _get_mode_params extracts correct values."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        params = model._get_params_dict()

        mode_0_params = model._get_mode_params(params, 0)
        mode_1_params = model._get_mode_params(params, 1)

        # Check mode-specific params differ
        assert mode_0_params["G"] == params["G_0"]
        assert mode_1_params["G"] == params["G_1"]

        # Check shared params are same
        assert mode_0_params["sigma_y0"] == mode_1_params["sigma_y0"]


class TestFMLIKHPredictions:
    """Test FMLIKH model predictions."""

    @pytest.fixture
    def model(self):
        """Create 2-mode isothermal FMLIKH."""
        return FMLIKH(n_modes=2, include_thermal=False)

    @pytest.mark.smoke
    def test_startup_prediction(self, model):
        """Test startup prediction with multiple modes."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t
        stress = model._predict_from_params(t, strain, model._get_params_dict())

        assert stress.shape == t.shape
        assert jnp.isfinite(stress).all()

    @pytest.mark.smoke
    def test_flow_curve_prediction(self, model):
        """Test flow curve prediction with multiple modes."""
        gamma_dot = jnp.logspace(-2, 2, 20)
        stress = model._predict(gamma_dot, test_mode="flow_curve")

        assert stress.shape == gamma_dot.shape
        assert jnp.isfinite(stress).all()

    def test_more_modes_gives_different_response(self):
        """Test that more modes changes the response."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        model_2 = FMLIKH(n_modes=2, include_thermal=False, shared_alpha=True)
        model_4 = FMLIKH(n_modes=4, include_thermal=False, shared_alpha=True)

        stress_2 = model_2._predict_from_params(t, strain, model_2._get_params_dict())
        stress_4 = model_4._predict_from_params(t, strain, model_4._get_params_dict())

        # Responses should be different (default parameters differ)
        assert not jnp.allclose(stress_2, stress_4)

    def test_single_mode_matches_fikh(self):
        """Test single-mode FMLIKH is similar to FIKH."""
        from rheojax.models.fikh import FIKH

        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        # FIKH
        fikh = FIKH(include_thermal=False, alpha_structure=0.5)

        # FMLIKH with 1 mode and matching parameters
        fmlikh = FMLIKH(n_modes=1, include_thermal=False, alpha_structure=0.5)

        # Set FMLIKH params to match FIKH
        fmlikh_params = fmlikh._get_params_dict()
        fikh_params = fikh._get_params_dict()

        # Update FMLIKH mode-0 params to match FIKH
        fmlikh_params["G_0"] = fikh_params["G"]
        fmlikh_params["eta_0"] = fikh_params["eta"]
        fmlikh_params["C_0"] = fikh_params["C"]
        fmlikh_params["gamma_dyn_0"] = fikh_params["gamma_dyn"]

        stress_fikh = fikh._predict_from_params(t, strain, fikh_params)
        stress_fmlikh = fmlikh._predict_from_params(t, strain, fmlikh_params)

        # Should be identical (or very close)
        assert jnp.allclose(stress_fikh, stress_fmlikh, rtol=1e-4)


class TestFMLIKHModeInfo:
    """Test FMLIKH mode information utilities."""

    def test_get_mode_info(self):
        """Test get_mode_info returns correct structure."""
        model = FMLIKH(n_modes=3, include_thermal=False, shared_alpha=True)
        info = model.get_mode_info()

        assert info["n_modes"] == 3
        assert info["shared_alpha"] is True
        assert len(info["modes"]) == 3

        for i, mode in enumerate(info["modes"]):
            assert mode["mode"] == i
            assert "G" in mode
            assert "eta" in mode
            assert "tau" in mode
            assert "C" in mode

    def test_get_mode_info_per_mode_alpha(self):
        """Test get_mode_info with per-mode alpha."""
        model = FMLIKH(n_modes=2, include_thermal=False, shared_alpha=False)
        info = model.get_mode_info()

        assert info["shared_alpha"] is False
        assert "alpha_shared" not in info

        for mode in info["modes"]:
            assert "alpha" in mode


class TestFMLIKHModelFunction:
    """Test FMLIKH model_function for Bayesian inference."""

    def test_model_function_startup(self):
        """Test model_function for startup."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        model._test_mode = "startup"

        t = jnp.linspace(0, 5, 50)
        strain = 0.01 * t
        X = jnp.stack([t, strain], axis=0)

        params = model._get_params_dict()
        result = model.model_function(X, params)

        assert result.shape == t.shape

    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        model._test_mode = "flow_curve"

        gamma_dot = jnp.logspace(-2, 2, 20)
        params = model._get_params_dict()

        result = model.model_function(gamma_dot, params)
        assert result.shape == gamma_dot.shape
