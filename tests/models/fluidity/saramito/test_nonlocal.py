"""Tests for FluiditySaramitoNonlocal model (TC-009).

Tests cover:
- Flow curve prediction (shape, finiteness, monotonicity)
- Creep simulation (above and below yield)
- model_function interface for flow_curve, startup, and creep protocols
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fluidity.saramito import FluiditySaramitoNonlocal

jax, jnp = safe_import_jax()


@pytest.fixture
def model():
    """Create nonlocal model with known parameters.

    Uses N_y=16 for speed and coarse spatial grid.
    """
    m = FluiditySaramitoNonlocal(coupling="minimal", N_y=16, H=1e-3, xi=5e-5)
    m.parameters.set_value("G", 1e4)
    m.parameters.set_value("tau_y0", 100.0)
    m.parameters.set_value("K_HB", 50.0)
    m.parameters.set_value("n_HB", 0.5)
    m.parameters.set_value("f_age", 1e-6)
    m.parameters.set_value("f_flow", 1e-2)
    m.parameters.set_value("t_a", 10.0)
    m.parameters.set_value("b", 1.0)
    m.parameters.set_value("n_rej", 1.0)
    return m


@pytest.mark.smoke
class TestNonlocalFlowCurve:
    """Tests for nonlocal flow curve prediction."""

    def test_flow_curve_shape(self, model):
        """Test flow curve output shape and finiteness."""
        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model._predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0)

    def test_flow_curve_monotonic(self, model):
        """Test flow curve is monotonically increasing."""
        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model._predict_flow_curve(gamma_dot)

        assert np.all(np.diff(sigma) > 0)


class TestNonlocalCreep:
    """Tests for nonlocal creep simulation."""

    @pytest.mark.smoke
    def test_creep_above_yield(self, model):
        """Test creep with stress above yield produces increasing strain."""
        t = np.linspace(0, 50, 50)
        sigma_applied = 150.0  # Above yield (tau_y0 = 100)

        gamma, f_field = model.simulate_creep(t, sigma_applied)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))
        assert gamma[-1] > gamma[0]

    def test_creep_below_yield(self, model):
        """Test creep with stress below yield has bounded strain."""
        t = np.linspace(0, 50, 50)
        sigma_applied = 50.0  # Below yield (tau_y0 = 100)

        gamma, f_field = model.simulate_creep(t, sigma_applied)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    def test_creep_returns_fluidity_field(self, model):
        """Test creep returns spatial fluidity field."""
        t = np.linspace(0, 50, 50)
        sigma_applied = 150.0

        gamma, f_field = model.simulate_creep(t, sigma_applied)

        assert f_field.shape == (model.N_y,)
        assert np.all(np.isfinite(f_field))
        assert np.all(f_field > 0)


class TestNonlocalModelFunction:
    """Tests for nonlocal model_function interface."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self, model):
        """Test model_function routes correctly for flow_curve."""
        gamma_dot = np.logspace(-1, 1, 10)

        # Fit first to set up internal state
        sigma = model._predict_flow_curve(gamma_dot)
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=20)

        params = [model.parameters.get_value(k) for k in model.parameters.keys()]
        result = model.model_function(gamma_dot, params, test_mode="flow_curve")

        assert result.shape == gamma_dot.shape
        assert np.all(np.isfinite(result))

    def test_model_function_startup(self, model):
        """Test model_function routes correctly for startup."""
        model._test_mode = "startup"
        model._gamma_dot_applied = 1.0

        t = np.linspace(0, 5, 10)
        params = [model.parameters.get_value(k) for k in model.parameters.keys()]

        result = model.model_function(t, params, test_mode="startup", gamma_dot=1.0)

        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_model_function_creep(self, model):
        """Test model_function routes correctly for creep."""
        model._test_mode = "creep"
        model._sigma_applied = 150.0

        t = np.linspace(0, 50, 10)
        params = [model.parameters.get_value(k) for k in model.parameters.keys()]

        result = model.model_function(
            t, params, test_mode="creep", sigma_applied=150.0
        )

        assert result.shape == t.shape
        assert np.all(np.isfinite(result))
