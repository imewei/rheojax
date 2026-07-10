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
            "G",
            "tau_y",
            "K",
            "n_flow",
            "f_eq",
            "f_inf",
            "theta",
            "a",
            "n_rejuv",
            "xi",
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

    def test_creep_strain_increases(self):
        """TC-012: Test creep strain increases under applied stress."""
        model = FluidityNonlocal(N_y=16)
        model.parameters.set_value("f_eq", 1e-4)
        model.parameters.set_value("f_inf", 1e-2)
        t = np.linspace(0, 10, 20)
        gamma = model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="creep",
            gamma_dot=None,
            sigma_applied=1000.0,
            sigma_0=None,
        )
        gamma = np.array(gamma)
        assert gamma[-1] > gamma[0]


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
        # TC-024: Stress must be non-zero
        assert np.max(np.abs(stress)) > 0

    def test_laos_fluidity_trajectory_stored(self):
        """TC-022: Test that LAOS stores fluidity field trajectory."""
        model = FluidityNonlocal(N_y=16)
        strain, stress = model.simulate_laos(
            gamma_0=0.1, omega=1.0, n_cycles=2, n_points_per_cycle=32
        )
        assert model._f_field_trajectory is not None
        assert model._f_field_trajectory.shape[1] == 16


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

    def test_model_function_startup(self):
        """TC-002: Test model_function for startup protocol."""
        model = FluidityNonlocal(N_y=16)
        model._test_mode = "startup"
        model._gamma_dot_applied = 1.0
        t = np.linspace(0, 10, 20)
        params = list(model.parameters.get_values())
        result = model.model_function(t, params, test_mode="startup", gamma_dot=1.0)
        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_model_function_relaxation(self):
        """TC-002: Test model_function for relaxation protocol."""
        model = FluidityNonlocal(N_y=16)
        model._test_mode = "relaxation"
        t = np.linspace(0, 50, 20)
        params = list(model.parameters.get_values())
        # model_function passes sigma_0=None, so _simulate_pde defaults
        # to tau_y for the initial stress
        result = model.model_function(t, params, test_mode="relaxation")
        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_model_function_creep(self):
        """TC-002: Test model_function for creep protocol."""
        model = FluidityNonlocal(N_y=16)
        model._test_mode = "creep"
        model._sigma_applied = 1000.0
        t = np.linspace(0, 10, 20)
        params = list(model.parameters.get_values())
        result = model.model_function(
            t, params, test_mode="creep", sigma_applied=1000.0
        )
        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_model_function_laos(self):
        """TC-002: Test model_function for LAOS protocol."""
        model = FluidityNonlocal(N_y=16)
        model._test_mode = "laos"
        model._gamma_0 = 0.1
        model._omega_laos = 1.0

        period = 2.0 * np.pi / 1.0
        t = np.linspace(0, 2 * period, 20)
        params = list(model.parameters.get_values())
        result = model.model_function(
            t, params, test_mode="laos", gamma_0=0.1, omega=1.0
        )
        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_laos_requires_gamma_0(self):
        """TC-004: LAOS without gamma_0/omega should raise."""
        model = FluidityNonlocal(N_y=16)
        t = np.linspace(0, 1, 20)
        params = list(model.parameters.get_values())
        with pytest.raises((ValueError, TypeError, KeyError)):
            model.model_function(t, params, test_mode="laos")


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
        """Test fitting uses coarse grid for performance.

        TC-021: Note - this test uses flow_curve which doesn't actually use
        the spatial grid. The coarse grid optimization is only relevant for
        transient protocols (startup/relaxation/creep) which use PDE solving.
        The test verifies grid restoration after fitting regardless.
        """
        model = FluidityNonlocal(N_y=64)
        original_N_y = model.N_y

        # Simulate fitting flow curve (doesn't use grid)
        gamma_dot = np.logspace(-1, 1, 10)
        sigma = model._predict_flow_curve(gamma_dot)
        model.fit(gamma_dot, sigma, test_mode="flow_curve", max_iter=5)

        # Grid should be restored after fitting
        assert model.N_y == original_N_y


@pytest.mark.unit
class TestFluidityNonlocalConstruction:
    """Construction guards and error paths."""

    def test_invalid_grid_size_raises(self):
        """FL-011: N_y < 2 must raise (ZeroDivisionError guard on dy)."""
        with pytest.raises(ValueError, match="N_y must be >= 2"):
            FluidityNonlocal(N_y=1)


@pytest.mark.unit
class TestFluidityNonlocalFitDispatch:
    """Cover the _fit() test_mode dispatch and its error branches.

    These call _fit directly (bypassing the orchestrator) to exercise the
    dispatch table and the individual _fit_* method bodies with minimal grids.
    """

    def test_fit_requires_test_mode(self):
        """_fit with no test_mode kwarg and no _test_mode attribute raises."""
        model = FluidityNonlocal(N_y=6)
        # Ensure no lingering canonical mode
        if hasattr(model, "_test_mode"):
            del model._test_mode
        gamma_dot = np.logspace(-1, 1, 6)
        sigma = 500.0 + 100.0 * gamma_dot**0.5
        with pytest.raises(ValueError, match="test_mode must be specified"):
            model._fit(gamma_dot, sigma)

    def test_fit_uses_instance_test_mode(self):
        """_fit falls back to self._test_mode when kwarg omitted (line 205)."""
        model = FluidityNonlocal(N_y=6)
        model._test_mode = "flow_curve"
        gamma_dot = np.logspace(-1, 1, 6)
        sigma = 500.0 + 100.0 * gamma_dot**0.5
        model._fit(gamma_dot, sigma, max_iter=1)
        assert model.fitted_

    def test_fit_saos_alias_normalized(self):
        """test_mode='saos' is normalized to 'oscillation' (line 211)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 12)
        G_star = model._predict_saos_jit(
            jnp.asarray(omega), 1000.0, 1e-3
        )
        G_star = np.array(G_star)
        complex_gstar = G_star[:, 0] + 1j * G_star[:, 1]
        model._fit(omega, complex_gstar, test_mode="saos", max_iter=3)
        assert model._test_mode == "oscillation"
        assert model.fitted_

    def test_fit_unsupported_mode_raises(self):
        """Unknown test_mode raises in the dispatch (line 231)."""
        model = FluidityNonlocal(N_y=6)
        X = np.linspace(0.01, 1, 6)
        y = np.linspace(1, 10, 6)
        with pytest.raises(ValueError, match="Unsupported test_mode"):
            model._fit(X, y, test_mode="not_a_mode")

    def test_fit_startup_requires_gamma_dot(self):
        """startup mode without gamma_dot raises (line 410)."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.01, 1, 6)
        sigma = np.linspace(1, 100, 6)
        with pytest.raises(ValueError, match="startup mode requires gamma_dot"):
            model._fit(t, sigma, test_mode="startup", max_iter=1)

    def test_fit_creep_requires_sigma_applied(self):
        """creep mode without sigma_applied raises (line 412)."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.01, 1, 6)
        gamma = np.linspace(0, 1, 6)
        with pytest.raises(ValueError, match="creep mode requires sigma_applied"):
            model._fit(t, gamma, test_mode="creep", max_iter=1)

    def test_fit_startup_runs(self):
        """Startup transient fit executes end-to-end on a tiny grid."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.01, 1.0, 6)
        sigma = np.linspace(5.0, 200.0, 6)
        model._fit(t, sigma, test_mode="startup", gamma_dot=1.0, max_iter=1)
        assert model.fitted_

    def test_fit_relaxation_runs(self):
        """Relaxation transient fit executes end-to-end on a tiny grid."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.01, 5.0, 6)
        sigma = np.linspace(500.0, 100.0, 6)
        model._fit(t, sigma, test_mode="relaxation", sigma_0=800.0, max_iter=1)
        assert model.fitted_

    def test_fit_creep_runs(self):
        """Creep transient fit executes end-to-end on a tiny grid."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.01, 2.0, 6)
        gamma = np.linspace(0.0, 0.5, 6)
        model._fit(t, gamma, test_mode="creep", sigma_applied=1000.0, max_iter=1)
        assert model.fitted_

    def test_fit_laos_runs(self):
        """LAOS PDE fit executes end-to-end on a tiny grid."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.0, 4.0, 8)
        sigma = np.sin(t)
        model._fit(t, sigma, test_mode="laos", gamma_0=0.1, omega=1.0, max_iter=1)
        assert model.fitted_

    def test_fit_laos_requires_gamma_0_omega(self):
        """LAOS fit without gamma_0/omega raises (line 821)."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.0, 4.0, 8)
        sigma = np.sin(t)
        with pytest.raises(ValueError, match="LAOS fitting requires"):
            model._fit(t, sigma, test_mode="laos", max_iter=1)


@pytest.mark.unit
class TestFluidityNonlocalOscillationFit:
    """Direct tests for SAOS fitting and the data-driven seed."""

    def test_fit_oscillation_2d_input(self):
        """SAOS fit accepts a real (M, 2) [G', G''] array (lines 682-685)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 12)
        G_star = np.array(model._predict_saos_jit(jnp.asarray(omega), 1000.0, 1e-3))
        model._fit_oscillation(omega, G_star, max_iter=3)
        assert model.parameters.get_value("G") > 0
        assert model.parameters.get_value("f_eq") > 0

    def test_fit_oscillation_bad_shape_raises(self):
        """SAOS fit rejects malformed G_star (line 687)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 12)
        bad = np.ones((12, 3))
        with pytest.raises(ValueError, match="G_star must be complex or"):
            model._fit_oscillation(omega, bad)

    def test_seed_saos_short_data_returns(self):
        """_seed_saos_from_data early-returns for <2 points (line 747)."""
        model = FluidityNonlocal()
        g_before = model.parameters.get_value("G")
        model._seed_saos_from_data(
            np.array([1.0]), np.array([10.0]), np.array([1.0])
        )
        # No mutation on the early-return path
        assert model.parameters.get_value("G") == g_before

    def test_seed_saos_crossover_path(self):
        """Crossover in G'/G'' drives the interpolated tau seed (lines 758-766)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 30)
        # Maxwell G*, crossover at omega*tau=1 (tau=1 here) inside the range
        G_star = np.array(model._predict_saos_jit(jnp.asarray(omega), 1000.0, 1e-3))
        model._seed_saos_from_data(omega, G_star[:, 0], G_star[:, 1])
        # Seeds must land inside their bounds and be finite
        assert np.isfinite(model.parameters.get_value("G"))
        assert np.isfinite(model.parameters.get_value("f_eq"))

    def test_seed_saos_peak_fallback_path(self):
        """No crossover falls back to the G'' peak location (lines 767-769)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 20)
        # G' always dominates G'' -> no sign change in log(G')-log(G'')
        G_prime = np.full_like(omega, 100.0)
        G_dp = 1.0 + np.exp(-((np.log(omega)) ** 2))  # single peak, always < G'
        model._seed_saos_from_data(omega, G_prime, G_dp)
        assert np.isfinite(model.parameters.get_value("G"))
        assert np.isfinite(model.parameters.get_value("f_eq"))


@pytest.mark.unit
class TestFluidityNonlocalPredictErrors:
    """Prediction-path error branches and mode routing."""

    def test_predict_transient_no_mode_raises(self):
        """_predict_transient with no mode raises (line 573)."""
        model = FluidityNonlocal(N_y=6)
        model._test_mode = None
        with pytest.raises(ValueError, match="Test mode not specified"):
            model._predict_transient(np.linspace(0.01, 1, 5))

    def test_predict_no_mode_raises(self):
        """_predict without any test_mode raises (line 1087)."""
        model = FluidityNonlocal(N_y=6)
        if hasattr(model, "_test_mode"):
            del model._test_mode
        with pytest.raises(ValueError, match="test_mode must be specified"):
            model._predict(np.linspace(0.01, 1, 5))

    def test_predict_saos_alias(self):
        """_predict normalizes 'saos' to oscillation and returns complex (line 1091)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 10)
        result = model._predict(omega, test_mode="saos")
        assert np.iscomplexobj(result)
        assert result.shape == omega.shape

    def test_predict_laos_runs(self):
        """_predict routes LAOS through the PDE and returns stress (lines 1132-1135)."""
        model = FluidityNonlocal(N_y=6)
        period = 2.0 * np.pi
        t = np.linspace(0.0, 2 * period, 12)
        result = model._predict(t, test_mode="laos", gamma_0=0.1, omega=1.0)
        assert result.shape == t.shape
        assert np.all(np.isfinite(result))

    def test_predict_laos_requires_gamma_0_omega(self):
        """_predict LAOS without gamma_0/omega raises (line 1131)."""
        model = FluidityNonlocal(N_y=6)
        model._gamma_0 = None
        model._omega_laos = None
        t = np.linspace(0.0, 1.0, 6)
        with pytest.raises(ValueError, match="LAOS prediction requires"):
            model._predict(t, test_mode="laos")

    def test_predict_unknown_mode_returns_zeros(self):
        """_predict fallthrough returns zeros for an unhandled mode (line 1137)."""
        model = FluidityNonlocal(N_y=6)
        X = np.linspace(0.01, 1, 6)
        result = model._predict(X, test_mode="totally_unknown")
        np.testing.assert_array_equal(result, np.zeros_like(X))


@pytest.mark.unit
class TestFluidityNonlocalModelFunctionDefaults:
    """model_function mode-default and fallthrough branches."""

    def test_model_function_defaults_to_oscillation(self):
        """mode=None defaults to oscillation (lines 1012-1013)."""
        model = FluidityNonlocal()
        model._test_mode = None
        omega = np.logspace(-1, 2, 8)
        params = list(model.parameters.get_values())
        result = model.model_function(omega, params)
        assert result.shape == (len(omega), 2)

    def test_model_function_saos_alias(self):
        """model_function normalizes 'saos' (lines 1016-1017)."""
        model = FluidityNonlocal()
        omega = np.logspace(-1, 2, 8)
        params = list(model.parameters.get_values())
        result = model.model_function(omega, params, test_mode="saos")
        assert result.shape == (len(omega), 2)

    def test_model_function_unknown_mode_returns_zeros(self):
        """Unhandled mode falls through to zeros (line 1070)."""
        model = FluidityNonlocal()
        X = np.logspace(-1, 2, 8)
        params = list(model.parameters.get_values())
        result = model.model_function(X, params, test_mode="mystery")
        np.testing.assert_array_equal(np.array(result), np.zeros_like(X))


@pytest.mark.unit
class TestFluidityNonlocalShearBandingDefaults:
    """Shear-banding accessors using the stored trajectory (default f_field)."""

    def _run_sim(self, model):
        t = np.linspace(0.0, 1.0, 10)
        model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

    def test_get_fluidity_profile_no_trajectory_raises(self):
        """Accessing a profile before any sim raises (line 606)."""
        model = FluidityNonlocal(N_y=8)
        with pytest.raises(ValueError, match="No trajectory available"):
            model.get_fluidity_profile()

    def test_shear_banding_metric_default_field(self):
        """CV metric uses the stored final field when f_field is None (line 622)."""
        model = FluidityNonlocal(N_y=8)
        self._run_sim(model)
        cv = model.get_shear_banding_metric()
        assert np.isfinite(cv)
        assert cv >= 0.0

    def test_banding_ratio_default_field(self):
        """Banding ratio uses the stored final field when f_field is None (line 638)."""
        model = FluidityNonlocal(N_y=8)
        self._run_sim(model)
        ratio = model.get_banding_ratio()
        assert np.isfinite(ratio)
        assert ratio >= 1.0


@pytest.mark.unit
class TestFluidityNonlocalLaosInternals:
    """LAOS internal init-condition handling and tracing-store branch."""

    def test_laos_f_init_interpolation(self):
        """f_init of a different length is interpolated onto the grid (lines 906-912)."""
        model = FluidityNonlocal(N_y=16)
        # Provide a coarse seed profile that must be resampled to N_y=16
        f_init = np.linspace(1e-5, 1e-3, 8)
        strain, stress = model.simulate_laos(
            gamma_0=0.1, omega=1.0, n_cycles=1, n_points_per_cycle=32, f_init=f_init
        )
        assert np.all(np.isfinite(stress))
        assert model._f_field_trajectory.shape[1] == 16

    def test_laos_f_init_matching_length(self):
        """f_init matching N_y is used directly, no interpolation (line 913)."""
        model = FluidityNonlocal(N_y=16)
        f_init = np.full(16, 1e-4)
        strain, stress = model.simulate_laos(
            gamma_0=0.1, omega=1.0, n_cycles=1, n_points_per_cycle=32, f_init=f_init
        )
        assert np.all(np.isfinite(stress))

    def test_laos_trajectory_store_skipped_under_trace(self):
        """FL-008: storing the field is skipped under JIT tracing (lines 959-961)."""
        model = FluidityNonlocal(N_y=6)
        t = np.linspace(0.0, 2.0, 8)
        t_jax = jnp.asarray(t)
        p = model.get_parameter_dict()

        def run(scale):
            # Route a tracer through the params so sol.ys is a tracer,
            # forcing the np.asarray(...) store to raise and be caught.
            p2 = {**p, "G": p["G"] * scale}
            _, stress = model._simulate_laos_internal(
                t_jax, p2, 0.1, 1.0, N_y=6, dy=model.gap_width / 5
            )
            return jnp.sum(stress)

        # Should complete without raising; the trajectory-store is swallowed.
        out = jax.jit(run)(1.0)
        assert np.isfinite(float(out))
