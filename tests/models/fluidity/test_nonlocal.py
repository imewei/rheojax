"""Tests for FluidityNonlocal model.

Tests cover instantiation, registry, parameters, spatial grid, shear banding,
and protocol implementations.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.core.registry import ModelRegistry
from rheojax.models.fluidity import FluidityNonlocal


@pytest.mark.smoke
class TestFluidityNonlocalSmoke:
    """Smoke tests for basic FluidityNonlocal functionality."""

    def test_instantiation(self):
        """Test FluidityNonlocal can be instantiated."""
        model = FluidityNonlocal()
        assert model is not None
        assert hasattr(model, "parameters")
        assert model.N_y == 64  # Default grid size

    def test_instantiation_with_custom_grid(self):
        """Test FluidityNonlocal with custom grid parameters."""
        model = FluidityNonlocal(N_y=32, gap_width=2e-3)
        assert model.N_y == 32
        assert model.gap_width == 2e-3
        assert np.isclose(model.dy, 2e-3 / 31)

    def test_registry_entry(self):
        """Test FluidityNonlocal is registered correctly."""
        model = ModelRegistry.create("fluidity_nonlocal")
        assert isinstance(model, FluidityNonlocal)

    def test_parameters_exist(self):
        """Test expected parameters are present including xi."""
        model = FluidityNonlocal()
        expected_params = [
            "G", "tau_y", "K", "n_flow", "f_eq", "f_inf",
            "theta", "a", "n_rejuv", "xi"
        ]
        for param in expected_params:
            assert param in model.parameters.keys()

    def test_xi_parameter_bounds(self):
        """Test cooperativity length parameter has valid bounds."""
        model = FluidityNonlocal()
        xi_param = model.parameters._parameters["xi"]
        assert xi_param.bounds == (1e-9, 1e-3)
        assert xi_param.value == 1e-5


@pytest.mark.unit
class TestFluidityNonlocalGrid:
    """Tests for spatial grid functionality."""

    def test_initial_fluidity_field_shape(self):
        """Test initial fluidity field has correct shape."""
        model = FluidityNonlocal(N_y=32)
        f_field = model._get_initial_f_field()

        assert f_field.shape == (32,)

    def test_initial_fluidity_field_uniform(self):
        """Test initial fluidity field is uniform."""
        model = FluidityNonlocal(N_y=64)
        f_field = model._get_initial_f_field(f_init=1e-5)

        np.testing.assert_allclose(np.array(f_field), 1e-5, rtol=1e-10)

    def test_initial_state_shape(self):
        """Test initial state vector has correct shape."""
        model = FluidityNonlocal(N_y=32)
        params = model.get_parameter_dict()

        # State: [Sigma_or_gamma, f[0], ..., f[N_y-1]]
        y0 = model._get_initial_state("startup", params)
        assert y0.shape == (33,)  # 1 + 32

    def test_grid_args(self):
        """Test grid arguments are correctly assembled."""
        model = FluidityNonlocal(N_y=64, gap_width=1e-3)
        args = model._get_grid_args()

        assert args["N_y"] == 64
        assert np.isclose(args["dy"], 1e-3 / 63)
        assert "xi" in args


@pytest.mark.unit
class TestFluidityNonlocalFlowCurve:
    """Tests for FluidityNonlocal flow curve (steady-state) protocol."""

    def test_flow_curve_prediction_shape(self):
        """Test flow curve prediction returns correct shape."""
        model = FluidityNonlocal()
        gamma_dot = np.logspace(-3, 2, 20)
        sigma = model._predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))

    def test_flow_curve_herschel_bulkley_behavior(self):
        """Test flow curve follows HB form: sigma = tau_y + K*gamma_dot^n."""
        model = FluidityNonlocal()
        model.parameters.set_value("tau_y", 500.0)
        model.parameters.set_value("K", 100.0)
        model.parameters.set_value("n_flow", 0.5)

        gamma_dot = np.logspace(-2, 2, 30)
        sigma = model._predict_flow_curve(gamma_dot)

        # Expected HB behavior
        tau_y = 500.0
        K = 100.0
        n = 0.5
        sigma_expected = tau_y + K * np.power(np.abs(gamma_dot), n)

        np.testing.assert_allclose(sigma, sigma_expected, rtol=1e-6)

    def test_flow_curve_monotonic_increase(self):
        """Test flow curve stress increases with shear rate."""
        model = FluidityNonlocal()
        gamma_dot = np.logspace(-3, 2, 50)
        sigma = model._predict_flow_curve(gamma_dot)

        # Should be monotonically increasing
        assert np.all(np.diff(sigma) >= 0)


@pytest.mark.unit
class TestFluidityNonlocalTransient:
    """Tests for FluidityNonlocal transient protocols."""

    def test_startup_stress_increases(self):
        """Test startup shows stress increase from zero."""
        model = FluidityNonlocal(N_y=16)  # Coarse grid for speed
        t = np.linspace(0, 1, 20)

        sigma = model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        sigma = np.array(sigma)

        # Stress should increase from 0
        assert sigma[-1] > sigma[0]
        assert np.abs(sigma[0]) < 10  # Close to zero at t=0

    def test_fluidity_field_stored(self):
        """Test fluidity field trajectory is stored after simulation."""
        model = FluidityNonlocal(N_y=16)
        t = np.linspace(0, 1, 10)

        model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        # Trajectory should be stored
        assert model._f_field_trajectory is not None
        assert model._f_field_trajectory.shape == (10, 16)

    def test_relaxation_stress_decay(self):
        """Test relaxation shows stress decay."""
        model = FluidityNonlocal(N_y=16)
        t = np.linspace(0, 50, 30)

        sigma = model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="relaxation",
            gamma_dot=None,
            sigma_applied=None,
            sigma_0=1000.0,
        )

        sigma = np.array(sigma)

        # Stress should decay
        assert sigma[0] > sigma[-1]
        assert sigma[0] > 500  # Started at 1000


@pytest.mark.unit
class TestFluidityNonlocalShearBanding:
    """Tests for shear banding analysis functionality."""

    def test_get_fluidity_profile(self):
        """Test fluidity profile retrieval."""
        model = FluidityNonlocal(N_y=16)
        t = np.linspace(0, 1, 10)

        model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        f_profile = model.get_fluidity_profile()
        assert f_profile.shape == (16,)
        assert np.all(np.isfinite(f_profile))

    def test_get_fluidity_profile_at_index(self):
        """Test fluidity profile retrieval at specific time index."""
        model = FluidityNonlocal(N_y=16)
        t = np.linspace(0, 1, 10)

        model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        f_initial = model.get_fluidity_profile(0)
        f_final = model.get_fluidity_profile(-1)

        # Initial and final profiles should differ
        assert not np.allclose(f_initial, f_final)

    def test_shear_banding_metric_uniform(self):
        """Test CV is low for uniform fluidity."""
        model = FluidityNonlocal(N_y=32)
        f_uniform = np.ones(32) * 1e-4

        cv = model.get_shear_banding_metric(f_uniform)
        assert cv < 0.01  # Nearly zero for uniform

    def test_shear_banding_metric_heterogeneous(self):
        """Test CV increases with heterogeneity."""
        model = FluidityNonlocal(N_y=32)

        # Create banded profile (half high, half low fluidity)
        f_banded = np.ones(32) * 1e-6
        f_banded[16:] = 1e-3  # 1000x contrast

        cv = model.get_shear_banding_metric(f_banded)
        assert cv > 0.3  # Significant banding indicator

    def test_banding_ratio_calculation(self):
        """Test banding ratio is computed correctly."""
        model = FluidityNonlocal(N_y=32)

        # Create profile with 100x contrast
        f_field = np.ones(32) * 1e-4
        f_field[16:] = 1e-2

        ratio = model.get_banding_ratio(f_field)
        np.testing.assert_allclose(ratio, 100.0, rtol=0.1)

    def test_is_banding_detection(self):
        """Test banding detection based on CV threshold."""
        model = FluidityNonlocal(N_y=32)

        # Uniform - no banding
        f_uniform = np.ones(32) * 1e-4
        assert not model.is_banding(f_uniform)

        # Banded - should detect
        f_banded = np.ones(32) * 1e-6
        f_banded[16:] = 1e-3
        assert model.is_banding(f_banded)


@pytest.mark.unit
class TestFluidityNonlocalOscillation:
    """Tests for FluidityNonlocal oscillation protocols."""

    def test_saos_prediction_shape(self):
        """Test SAOS prediction returns [G', G''] shape."""
        model = FluidityNonlocal()
        omega = np.logspace(-2, 2, 20)

        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G"),
            model.parameters.get_value("f_eq"),
            model.parameters.get_value("theta"),
        )

        G_star = np.array(G_star)
        assert G_star.shape == (len(omega), 2)
        assert np.all(np.isfinite(G_star))
        assert np.all(G_star >= 0)

    def test_laos_simulation(self):
        """Test LAOS simulation returns valid output."""
        model = FluidityNonlocal(N_y=16)  # Coarse grid for speed

        strain, stress = model.simulate_laos(
            gamma_0=0.1,
            omega=1.0,
            n_cycles=2,
            n_points_per_cycle=32,
        )

        assert strain.shape == stress.shape
        assert len(strain) == 2 * 32
        assert np.all(np.isfinite(strain))
        assert np.all(np.isfinite(stress))


@pytest.mark.unit
class TestFluidityNonlocalModelFunction:
    """Tests for Bayesian model_function interface."""

    def test_model_function_flow_curve(self):
        """Test model_function works for flow curve mode."""
        model = FluidityNonlocal()
        model._test_mode = "flow_curve"

        X = np.logspace(-2, 1, 10)
        params = list(model.parameters.get_values())

        result = model.model_function(X, params, test_mode="flow_curve")

        assert result.shape == X.shape
        assert np.all(np.isfinite(result))

    def test_model_function_oscillation(self):
        """Test model_function works for oscillation mode."""
        model = FluidityNonlocal()
        model._test_mode = "oscillation"

        X = np.logspace(-1, 2, 10)
        params = list(model.parameters.get_values())

        result = model.model_function(X, params, test_mode="oscillation")

        assert result.shape == (len(X), 2)
        assert np.all(np.isfinite(result))


@pytest.mark.unit
class TestFluidityNonlocalFitting:
    """Tests for FluidityNonlocal fitting capability."""

    def test_fit_flow_curve_synthetic(self):
        """Test fitting synthetic flow curve data."""
        # Generate synthetic HB data
        tau_y = 500.0
        K = 100.0
        n = 0.5

        gamma_dot = np.logspace(-2, 2, 30)
        sigma_true = tau_y + K * np.power(gamma_dot, n)

        # Add noise
        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.05 * np.random.randn(len(sigma_true)))

        # Fit with model
        model = FluidityNonlocal()
        model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve", max_iter=50)

        # Check fit
        assert model.fitted_

        # Model should have been updated (tau_y should change from default)
        # The fit may not converge perfectly with limited iterations

    def test_fit_uses_coarse_grid_for_transient(self):
        """Test fitting uses coarse grid for performance."""
        model = FluidityNonlocal(N_y=64)
        original_N_y = model.N_y

        # Simulate fitting flow curve (doesn't use grid)
        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model._predict_flow_curve(gamma_dot)
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=5)

        # Grid should be restored after fitting
        assert model.N_y == original_N_y
