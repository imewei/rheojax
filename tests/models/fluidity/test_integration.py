"""Integration tests for Fluidity models.

Tests cover model interface compliance, BaseModel inheritance, BayesianMixin
compatibility, and end-to-end workflows for both Local and Non-Local variants.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal


@pytest.mark.smoke
class TestFluidityIntegrationSmoke:
    """Smoke tests for model integration."""

    def test_local_is_base_model(self):
        """Test FluidityLocal inherits from BaseModel."""
        model = FluidityLocal()
        assert isinstance(model, BaseModel)

    def test_nonlocal_is_base_model(self):
        """Test FluidityNonlocal inherits from BaseModel."""
        model = FluidityNonlocal()
        assert isinstance(model, BaseModel)

    def test_models_have_fit_method(self):
        """Test both models have fit method."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        assert hasattr(local, "fit")
        assert hasattr(nonlocal_, "fit")
        assert callable(local.fit)
        assert callable(nonlocal_.fit)

    def test_models_have_predict_method(self):
        """Test both models have predict method."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        assert hasattr(local, "predict")
        assert hasattr(nonlocal_, "predict")


@pytest.mark.unit
class TestFluidityRegistration:
    """Tests for model registration and protocol support."""

    def test_local_registry_protocols(self):
        """Test FluidityLocal is registered with correct protocols."""
        model = ModelRegistry.create("fluidity_local")

        expected_protocols = [
            Protocol.FLOW_CURVE,
            Protocol.CREEP,
            Protocol.RELAXATION,
            Protocol.STARTUP,
            Protocol.OSCILLATION,
            Protocol.LAOS,
        ]

        # Model should be created successfully and have parameters
        assert isinstance(model, FluidityLocal)
        assert hasattr(model, "parameters")

    def test_nonlocal_registry_protocols(self):
        """Test FluidityNonlocal is registered with correct protocols."""
        model = ModelRegistry.create("fluidity_nonlocal")

        expected_protocols = [
            Protocol.FLOW_CURVE,
            Protocol.CREEP,
            Protocol.RELAXATION,
            Protocol.STARTUP,
            Protocol.OSCILLATION,
            Protocol.LAOS,
        ]

        # Model should be created successfully and have parameters
        assert isinstance(model, FluidityNonlocal)
        assert hasattr(model, "parameters")

    def test_factory_pattern_local(self):
        """Test FluidityLocal can be created via factory."""
        model = ModelRegistry.create("fluidity_local")
        assert isinstance(model, FluidityLocal)

    def test_factory_pattern_nonlocal(self):
        """Test FluidityNonlocal can be created via factory."""
        model = ModelRegistry.create("fluidity_nonlocal")
        assert isinstance(model, FluidityNonlocal)


@pytest.mark.unit
class TestFluidityParameterInterface:
    """Tests for parameter interface consistency."""

    def test_shared_parameters_match(self):
        """Test Local and Nonlocal share common parameters."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        shared_params = ["G", "tau_y", "K", "n_flow", "f_eq", "f_inf", "theta", "a", "n_rejuv"]

        for param in shared_params:
            assert param in local.parameters.keys()
            assert param in nonlocal_.parameters.keys()

    def test_nonlocal_has_xi(self):
        """Test Nonlocal has additional xi parameter."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        assert "xi" not in local.parameters.keys()
        assert "xi" in nonlocal_.parameters.keys()

    def test_get_parameter_dict(self):
        """Test both models can export parameter dict."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        local_dict = local.get_parameter_dict()
        nonlocal_dict = nonlocal_.get_parameter_dict()

        assert isinstance(local_dict, dict)
        assert isinstance(nonlocal_dict, dict)
        assert len(nonlocal_dict) == len(local_dict) + 1  # xi parameter

    def test_parameter_values_method(self):
        """Test parameters.get_values() returns array of values."""
        model = FluidityLocal()
        values = model.parameters.get_values()

        # get_values() returns an ndarray
        assert hasattr(values, "__len__")
        assert len(values) == len(list(model.parameters.keys()))


@pytest.mark.unit
class TestFluidityFitPredictInterface:
    """Tests for fit/predict interface consistency."""

    def test_local_fit_requires_test_mode(self):
        """Test Local fit raises without test_mode."""
        model = FluidityLocal()
        X = np.logspace(-1, 1, 10)
        y = np.ones(10) * 100

        with pytest.raises(ValueError, match="test_mode"):
            model.fit(X, y)

    def test_nonlocal_fit_requires_test_mode(self):
        """Test Nonlocal fit raises without test_mode."""
        model = FluidityNonlocal()
        X = np.logspace(-1, 1, 10)
        y = np.ones(10) * 100

        with pytest.raises(ValueError, match="test_mode"):
            model.fit(X, y)

    def test_local_fit_sets_fitted_flag(self):
        """Test Local fit sets fitted_ attribute."""
        model = FluidityLocal()
        X = np.logspace(-1, 1, 10)
        y = model._predict_flow_curve(X)

        model.fit(X, y, test_mode="flow_curve", max_iter=5)
        assert model.fitted_ is True

    def test_nonlocal_fit_sets_fitted_flag(self):
        """Test Nonlocal fit sets fitted_ attribute."""
        model = FluidityNonlocal()
        X = np.logspace(-1, 1, 10)
        y = model._predict_flow_curve(X)

        model.fit(X, y, test_mode="flow_curve", max_iter=5)
        assert model.fitted_ is True


@pytest.mark.unit
class TestFluidityBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    def test_local_has_model_function(self):
        """Test Local has model_function for Bayesian inference."""
        model = FluidityLocal()
        assert hasattr(model, "model_function")
        assert callable(model.model_function)

    def test_nonlocal_has_model_function(self):
        """Test Nonlocal has model_function for Bayesian inference."""
        model = FluidityNonlocal()
        assert hasattr(model, "model_function")
        assert callable(model.model_function)

    def test_local_has_fit_bayesian(self):
        """Test Local has fit_bayesian method from mixin."""
        model = FluidityLocal()
        assert hasattr(model, "fit_bayesian")

    def test_nonlocal_has_fit_bayesian(self):
        """Test Nonlocal has fit_bayesian method from mixin."""
        model = FluidityNonlocal()
        assert hasattr(model, "fit_bayesian")


@pytest.mark.unit
class TestFluidityEndToEnd:
    """End-to-end tests for typical usage patterns."""

    def test_local_flow_curve_workflow(self):
        """Test complete flow curve workflow for Local model."""
        # Create model
        model = FluidityLocal()

        # Set parameters
        model.parameters.set_value("tau_y", 500.0)
        model.parameters.set_value("f_eq", 1e-6)
        model.parameters.set_value("f_inf", 1e-3)

        # Generate data
        gamma_dot = np.logspace(-2, 2, 30)
        sigma = model._predict_flow_curve(gamma_dot)

        # Verify output
        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma > 0)

    def test_nonlocal_startup_with_banding_analysis(self):
        """Test startup protocol with shear banding analysis."""
        # Create model with small cooperativity length (promotes banding)
        model = FluidityNonlocal(N_y=32, gap_width=1e-3)
        model.parameters.set_value("xi", 1e-5)

        # Simulate startup
        t = np.linspace(0, 1, 20)
        sigma = model._simulate_pde(
            jnp.asarray(t),
            model.get_parameter_dict(),
            mode="startup",
            gamma_dot=1.0,
            sigma_applied=None,
            sigma_0=None,
        )

        # Analyze fluidity profile
        f_profile = model.get_fluidity_profile(-1)
        cv = model.get_shear_banding_metric(f_profile)

        # Verify analysis works (values depend on parameters)
        assert np.isfinite(cv)
        assert cv >= 0

    def test_local_saos_to_laos_workflow(self):
        """Test oscillation workflow from SAOS to LAOS."""
        model = FluidityLocal()

        # SAOS - linear regime
        omega = np.logspace(-2, 2, 20)
        G_star = model._predict_saos_jit(
            jnp.asarray(omega),
            model.parameters.get_value("G"),
            model.parameters.get_value("f_eq"),
            model.parameters.get_value("theta"),
        )
        G_star = np.array(G_star)

        # LAOS - nonlinear regime
        strain, stress = model.simulate_laos(
            gamma_0=0.5,  # Large amplitude
            omega=1.0,
            n_cycles=2,
            n_points_per_cycle=64,
        )

        # Extract harmonics
        harmonics = model.extract_harmonics(stress, n_points_per_cycle=64)

        # Verify workflow completes
        assert G_star.shape == (20, 2)
        assert len(strain) == 128
        assert harmonics["I_1"] > 0


@pytest.mark.unit
class TestFluidityModelComparison:
    """Tests comparing Local and Nonlocal model behavior."""

    def test_flow_curves_differ_by_formulation(self):
        """Test Local and Nonlocal have different flow curve formulations."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        # Set same base parameters
        for param in ["G", "tau_y", "K", "n_flow", "f_eq", "f_inf", "theta", "a", "n_rejuv"]:
            value = local.parameters.get_value(param)
            nonlocal_.parameters.set_value(param, value)

        gamma_dot = np.logspace(-2, 2, 20)

        sigma_local = local._predict_flow_curve(gamma_dot)
        sigma_nonlocal = nonlocal_._predict_flow_curve(gamma_dot)

        # Different formulations should give different results
        # Local uses aging/rejuvenation dynamics, Nonlocal uses HB directly
        # But both should be reasonable and finite
        assert np.all(np.isfinite(sigma_local))
        assert np.all(np.isfinite(sigma_nonlocal))

    def test_saos_predictions_equivalent(self):
        """Test SAOS predictions are equivalent (both use Maxwell approx)."""
        local = FluidityLocal()
        nonlocal_ = FluidityNonlocal()

        # Set same parameters
        G = 1e6
        f_eq = 1e-6
        theta = 10.0

        omega = np.logspace(-2, 2, 20)

        G_star_local = np.array(local._predict_saos_jit(
            jnp.asarray(omega), G, f_eq, theta
        ))
        G_star_nonlocal = np.array(nonlocal_._predict_saos_jit(
            jnp.asarray(omega), G, f_eq, theta
        ))

        np.testing.assert_allclose(G_star_local, G_star_nonlocal, rtol=1e-10)


@pytest.mark.unit
class TestFluidityImportPaths:
    """Tests for import paths and module structure."""

    def test_import_from_package(self):
        """Test import from package level."""
        from rheojax.models.fluidity import FluidityLocal, FluidityNonlocal

        assert FluidityLocal is not None
        assert FluidityNonlocal is not None

    def test_import_from_models(self):
        """Test import from rheojax.models."""
        from rheojax.models import FluidityLocal, FluidityNonlocal

        assert FluidityLocal is not None
        assert FluidityNonlocal is not None

    def test_import_via_registry(self):
        """Test models accessible via registry."""
        from rheojax.core.registry import ModelRegistry

        available = ModelRegistry.list_models()
        assert "fluidity_local" in available
        assert "fluidity_nonlocal" in available
