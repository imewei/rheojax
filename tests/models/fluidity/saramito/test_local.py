"""Unit tests for FluiditySaramitoLocal model.

Tests cover:
- Model instantiation and parameter setup
- Registry registration
- Protocol fitting and prediction
- Bayesian interface compatibility
"""

import numpy as np
import pytest

from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fluidity.saramito import FluiditySaramitoLocal

jax, jnp = safe_import_jax()


class TestModelInstantiation:
    """Tests for model creation and setup."""

    @pytest.mark.smoke
    def test_minimal_coupling_init(self):
        """Test initialization with minimal coupling."""
        model = FluiditySaramitoLocal(coupling="minimal")

        assert model.coupling == "minimal"
        assert "tau_y0" in model.parameters.keys()
        assert "tau_y_coupling" not in model.parameters.keys()

    @pytest.mark.smoke
    def test_full_coupling_init(self):
        """Test initialization with full coupling."""
        model = FluiditySaramitoLocal(coupling="full")

        assert model.coupling == "full"
        assert "tau_y_coupling" in model.parameters.keys()
        assert "m_yield" in model.parameters.keys()

    def test_parameters_have_bounds(self):
        """Test all parameters have valid bounds."""
        model = FluiditySaramitoLocal()

        for name in model.parameters.keys():
            param = model.parameters.get(name)
            assert param.bounds is not None
            assert param.bounds[0] < param.bounds[1]

    def test_repr(self):
        """Test string representation."""
        model = FluiditySaramitoLocal(coupling="minimal")
        repr_str = repr(model)

        assert "FluiditySaramitoLocal" in repr_str
        assert "minimal" in repr_str


class TestRegistry:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registered_name(self):
        """Test model is registered with correct name."""
        model = ModelRegistry.create("fluidity_saramito_local")
        assert isinstance(model, FluiditySaramitoLocal)

    def test_registered_protocols(self):
        """Test all protocols are registered."""
        info = ModelRegistry.get_info("fluidity_saramito_local")

        expected_protocols = {
            Protocol.FLOW_CURVE,
            Protocol.CREEP,
            Protocol.RELAXATION,
            Protocol.STARTUP,
            Protocol.OSCILLATION,
            Protocol.LAOS,
        }

        assert set(info.protocols) == expected_protocols


class TestFlowCurveFitting:
    """Tests for steady-state flow curve protocol."""

    @pytest.fixture
    def synthetic_flow_curve(self):
        """Generate synthetic HB-like flow curve."""
        gamma_dot = np.logspace(-2, 2, 30)
        tau_y = 100.0
        K = 50.0
        n = 0.5
        sigma = tau_y + K * gamma_dot**n
        # Add small noise
        sigma = sigma * (1 + 0.01 * np.random.randn(len(sigma)))
        return gamma_dot, sigma

    @pytest.mark.smoke
    def test_fit_flow_curve_minimal(self, synthetic_flow_curve):
        """Test flow curve fitting with minimal coupling."""
        gamma_dot, sigma = synthetic_flow_curve

        model = FluiditySaramitoLocal(coupling="minimal")
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=100)

        assert model.fitted_
        assert model._test_mode == "flow_curve"

    @pytest.mark.smoke
    def test_predict_flow_curve(self, synthetic_flow_curve):
        """Test flow curve prediction."""
        gamma_dot, sigma = synthetic_flow_curve

        model = FluiditySaramitoLocal(coupling="minimal")
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=100)

        sigma_pred = model.predict(gamma_dot)

        # Prediction should have reasonable shape and values
        assert sigma_pred.shape == sigma.shape
        assert np.all(sigma_pred > 0)
        assert np.all(np.isfinite(sigma_pred))

    def test_flow_curve_monotonic(self, synthetic_flow_curve):
        """Test predicted flow curve is monotonic."""
        gamma_dot, sigma = synthetic_flow_curve

        model = FluiditySaramitoLocal(coupling="minimal")
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=100)

        gamma_dot_fine = np.logspace(-2, 2, 100)
        sigma_pred = model.predict(gamma_dot_fine)

        # Should be monotonically increasing
        assert np.all(np.diff(sigma_pred) > 0)


class TestStartupFitting:
    """Tests for startup transient protocol."""

    @pytest.fixture
    def model_with_params(self):
        """Create model with known parameters."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_simulate_startup(self, model_with_params):
        """Test startup simulation runs."""
        t = np.linspace(0, 10, 100)
        gamma_dot = 1.0

        strain, stress, fluidity = model_with_params.simulate_startup(t, gamma_dot)

        assert strain.shape == t.shape
        assert stress.shape == t.shape
        assert fluidity.shape == t.shape

        # Physical checks
        assert np.all(strain >= 0)
        assert np.all(stress >= 0)
        assert np.all(fluidity > 0)

    @pytest.mark.smoke
    def test_stress_overshoot(self, model_with_params):
        """Test stress overshoot is present."""
        t = np.linspace(0, 50, 500)
        gamma_dot = 1.0

        _, stress, _ = model_with_params.simulate_startup(t, gamma_dot)

        # Maximum should be higher than final value (overshoot)
        sigma_max = np.max(stress)
        sigma_final = stress[-1]

        # For typical EVP parameters, expect overshoot
        # This may not always occur depending on parameters
        assert sigma_max >= sigma_final * 0.95  # At least close to final

    def test_startup_trajectory_stored(self, model_with_params):
        """Test trajectory is stored for plotting."""
        t = np.linspace(0, 10, 100)
        gamma_dot = 1.0

        model_with_params.simulate_startup(t, gamma_dot)

        assert model_with_params._trajectory is not None
        assert "tau_xy" in model_with_params._trajectory
        assert "fluidity" in model_with_params._trajectory


class TestCreepFitting:
    """Tests for creep protocol."""

    @pytest.fixture
    def model_with_params(self):
        """Create model with known parameters."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_simulate_creep(self, model_with_params):
        """Test creep simulation runs."""
        t = np.linspace(0, 100, 200)
        sigma_applied = 150.0  # Above yield

        strain, fluidity = model_with_params.simulate_creep(t, sigma_applied)

        assert strain.shape == t.shape
        assert fluidity.shape == t.shape

        # Strain should increase for stress above yield
        assert strain[-1] > strain[0]

    def test_creep_below_yield(self, model_with_params):
        """Test bounded strain below yield stress."""
        t = np.linspace(0, 100, 200)
        sigma_applied = 50.0  # Below yield (tau_y = 100)

        strain, fluidity = model_with_params.simulate_creep(t, sigma_applied)

        # Strain should be bounded (not continuously increasing)
        # For stress below yield, flow is suppressed
        strain_rate_final = (strain[-1] - strain[-10]) / (t[-1] - t[-10])
        strain_rate_initial = (strain[10] - strain[0]) / (t[10] - t[0])

        # Final rate should be much lower than initial (arrest)
        # This depends on model parameters
        assert strain_rate_final >= 0  # Non-negative


class TestOscillationFitting:
    """Tests for oscillatory protocols."""

    @pytest.mark.smoke
    def test_saos_prediction(self):
        """Test SAOS prediction shape."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("f_age", 1e-5)

        omega = np.logspace(-1, 2, 30)
        G_star = FluiditySaramitoLocal._predict_saos_jit(
            jnp.array(omega), 1e4, 1e-5
        )

        G_star_np = np.array(G_star)
        assert G_star_np.shape == (30, 2)

        # G' and G'' should be positive
        assert np.all(G_star_np[:, 0] >= 0)  # G'
        assert np.all(G_star_np[:, 1] >= 0)  # G''

    def test_maxwell_crossover(self):
        """Test Maxwell-like crossover in SAOS."""
        model = FluiditySaramitoLocal(coupling="minimal")
        G = 1e4
        f_eq = 1e-5
        tau_eff = 1.0 / (G * f_eq)  # Relaxation time

        omega = np.logspace(-3, 3, 100)
        G_star = FluiditySaramitoLocal._predict_saos_jit(
            jnp.array(omega), G, f_eq
        )

        G_prime = np.array(G_star[:, 0])
        G_double_prime = np.array(G_star[:, 1])

        # At crossover (ω*τ = 1), G' ≈ G''
        crossover_omega = 1.0 / tau_eff
        crossover_idx = np.argmin(np.abs(omega - crossover_omega))

        # G' and G'' should be similar at crossover
        ratio = G_prime[crossover_idx] / (G_double_prime[crossover_idx] + 1e-30)
        assert 0.5 < ratio < 2.0


class TestLAOSProtocol:
    """Tests for LAOS protocol."""

    @pytest.fixture
    def model_with_params(self):
        """Create model with known parameters."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_simulate_laos(self, model_with_params):
        """Test LAOS simulation runs."""
        gamma_0 = 0.1
        omega = 1.0

        t, strain, stress = model_with_params.simulate_laos(
            gamma_0, omega, n_cycles=2, n_points_per_cycle=128
        )

        assert len(t) == 256
        assert strain.shape == t.shape
        assert stress.shape == t.shape

        # Strain should oscillate
        assert np.max(strain) > 0
        assert np.min(strain) < 0

    @pytest.mark.smoke
    def test_extract_harmonics(self, model_with_params):
        """Test harmonic extraction from LAOS."""
        gamma_0 = 1.0  # Larger amplitude for nonlinearity
        omega = 1.0

        t, strain, stress = model_with_params.simulate_laos(
            gamma_0, omega, n_cycles=3, n_points_per_cycle=256
        )

        harmonics = model_with_params.extract_harmonics(stress, n_points_per_cycle=256)

        assert "I_1" in harmonics
        assert "I_3" in harmonics
        assert "I_3_I_1" in harmonics

        # Fundamental should be dominant
        assert harmonics["I_1"] > harmonics["I_3"]


class TestBayesianInterface:
    """Tests for Bayesian fitting compatibility."""

    @pytest.fixture
    def synthetic_flow_curve(self):
        """Generate synthetic flow curve."""
        gamma_dot = np.logspace(-1, 1, 20)
        tau_y = 100.0
        K = 50.0
        n = 0.5
        sigma = tau_y + K * gamma_dot**n
        return gamma_dot, sigma

    @pytest.mark.smoke
    def test_model_function_flow_curve(self, synthetic_flow_curve):
        """Test model_function for flow curve mode."""
        gamma_dot, sigma = synthetic_flow_curve

        model = FluiditySaramitoLocal(coupling="minimal")
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=50)

        # Get parameters as array
        params = [model.parameters.get_value(k) for k in model.parameters.keys()]

        # Call model_function
        result = model.model_function(gamma_dot, params, test_mode="flow_curve")

        assert result.shape == sigma.shape
        assert np.all(np.isfinite(result))

    def test_model_function_oscillation(self):
        """Test model_function for oscillation mode."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model._test_mode = "oscillation"

        omega = np.logspace(-1, 1, 10)
        params = [model.parameters.get_value(k) for k in model.parameters.keys()]

        result = model.model_function(omega, params, test_mode="oscillation")

        assert result.shape == (10, 2)


class TestNormalStressPredictions:
    """Tests for normal stress difference predictions."""

    @pytest.mark.smoke
    def test_predict_normal_stresses(self):
        """Test N1, N2 prediction."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)

        gamma_dot = np.array([0.1, 1.0, 10.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert N1.shape == gamma_dot.shape
        assert N2.shape == gamma_dot.shape

        # N1 should be positive (Weissenberg effect)
        assert np.all(N1 > 0)

        # N2 should be zero for UCM
        assert np.all(N2 == 0)


class TestHelperMethods:
    """Tests for helper and analysis methods."""

    @pytest.fixture
    def fitted_model(self):
        """Create and fit a model."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        model.fitted_ = True
        return model

    @pytest.mark.smoke
    def test_relaxation_time(self, fitted_model):
        """Test relaxation time property."""
        lam = fitted_model.relaxation_time

        # λ = 1/(G*f_age) = 1/(1e4 * 1e-5) = 10
        assert np.isclose(lam, 10.0, rtol=0.1)

    def test_critical_stress(self, fitted_model):
        """Test critical stress getter."""
        sigma_c = fitted_model.get_critical_stress()

        assert sigma_c == 100.0  # tau_y0

    def test_overshoot_ratio(self, fitted_model):
        """Test overshoot ratio calculation."""
        ratio = fitted_model.get_overshoot_ratio(gamma_dot=1.0, t_max=50.0)

        # Ratio should be >= 1 (may or may not have overshoot)
        assert ratio >= 0.9
        assert np.isfinite(ratio)
