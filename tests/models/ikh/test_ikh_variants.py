"""Tests for IKH model variants (MIKH, MLIKH).

These tests verify initialization, parameters, and basic prediction
for the MIKH and MLIKH model variants.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry

jax, jnp = safe_import_jax()

import rheojax.models  # noqa: F401
from rheojax.models.ikh.mikh import MIKH
from rheojax.models.ikh.ml_ikh import MLIKH

# ============================================================================
# MIKH Additional Tests
# ============================================================================


class TestMIKHSmoke:
    """Smoke tests for MIKH model."""

    @pytest.mark.smoke
    def test_import(self):
        assert MIKH is not None

    @pytest.mark.smoke
    def test_instantiation(self):
        model = MIKH()
        assert model is not None

    @pytest.mark.smoke
    def test_registry(self):
        assert "mikh" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_required_parameters_exist(self):
        model = MIKH()
        required = [
            "G",
            "eta",
            "C",
            "gamma_dyn",
            "m",
            "sigma_y0",
            "delta_sigma_y",
            "tau_thix",
            "Gamma",
            "eta_inf",
            "mu_p",
        ]
        for param in required:
            assert param in model.parameters, f"Missing parameter: {param}"

    @pytest.mark.smoke
    def test_parameter_bounds_physical(self):
        model = MIKH()
        # G must be positive
        assert model.parameters["G"].bounds[0] > 0
        # eta_inf can be zero
        assert model.parameters["eta_inf"].bounds[0] >= 0

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction (steady state)."""
        model = MIKH()
        gamma_dot = np.logspace(-2, 2, 20)
        result = model.predict(gamma_dot, test_mode="flow_curve")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))
        assert len(result_arr) == 20

    @pytest.mark.smoke
    def test_predict_startup(self):
        """Test startup shear prediction."""
        model = MIKH()
        dt = 0.01
        t = np.arange(0, 2.0, dt)
        gamma_dot_val = 1.0
        gamma = gamma_dot_val * t
        X = np.stack([t, gamma])
        result = model.predict(X, test_mode="startup")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))


# ============================================================================
# MLIKH Tests
# ============================================================================


class TestMLIKHSmoke:
    """Smoke tests for MLIKH model."""

    @pytest.mark.smoke
    def test_import(self):
        assert MLIKH is not None

    @pytest.mark.smoke
    def test_instantiation_default(self):
        model = MLIKH()
        assert model is not None
        assert model._n_modes == 2
        assert model._yield_mode == "per_mode"

    @pytest.mark.smoke
    def test_instantiation_custom_modes(self):
        model = MLIKH(n_modes=3)
        assert model._n_modes == 3

    @pytest.mark.smoke
    def test_instantiation_weighted_sum(self):
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        assert model._yield_mode == "weighted_sum"

    @pytest.mark.smoke
    def test_invalid_n_modes(self):
        with pytest.raises(ValueError, match="n_modes must be >= 1"):
            MLIKH(n_modes=0)

    @pytest.mark.smoke
    def test_invalid_yield_mode(self):
        with pytest.raises(ValueError, match="yield_mode must be"):
            MLIKH(yield_mode="invalid")

    @pytest.mark.smoke
    def test_registry(self):
        assert "ml_ikh" in ModelRegistry.list_models()

    @pytest.mark.smoke
    def test_per_mode_parameters_exist(self):
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        # Per-mode parameters should include mode-indexed params
        param_names = list(model.parameters.keys())
        assert len(param_names) > 0
        # eta_inf is global
        assert "eta_inf" in model.parameters

    @pytest.mark.smoke
    def test_weighted_sum_parameters_exist(self):
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        param_names = list(model.parameters.keys())
        assert len(param_names) > 0

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction."""
        model = MLIKH(n_modes=2)
        gamma_dot = np.logspace(-2, 2, 20)
        result = model.predict(gamma_dot, test_mode="flow_curve")
        result_arr = np.asarray(result)
        assert np.all(np.isfinite(result_arr))
        assert len(result_arr) == 20
