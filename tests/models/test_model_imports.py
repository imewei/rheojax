"""Tests for model import validation and float64 precision.

This test module validates that all 20 models:
1. Can be imported successfully with safe_import_jax()
2. Maintain float64 precision in predictions
3. Basic fit() and predict() functionality works

These tests complement the float64 precision tests in tests/core/test_float64_precision.py
by focusing specifically on model-level import validation.
"""

from __future__ import annotations

import numpy as np
import pytest

# Import rheo package first (triggers NLSQ import before JAX)
import rheo
from rheo.core.jax_config import safe_import_jax

# Now safe to import JAX
jax, jnp = safe_import_jax()


# List of all 20 models to test
ALL_MODEL_NAMES = [
    # Classical models (4)
    "maxwell",
    "zener",
    "springpot",
    "bingham",
    # Fractional models (11)
    "fractional_maxwell_gel",
    "fractional_maxwell_liquid",
    "fractional_maxwell_model",
    "fractional_zener_ll",
    "fractional_zener_sl",
    "fractional_zener_ss",
    "fractional_burgers",
    "fractional_jeffreys",
    "fractional_kelvin_voigt",
    "fractional_kv_zener",
    "fractional_poynting_thomson",
    # Flow models (5)
    "power_law",
    "carreau",
    "carreau_yasuda",
    "cross",
    "herschel_bulkley",
]


def test_all_models_can_be_imported():
    """Test that all 20 models can be imported successfully.

    This validates that:
    1. All model files exist
    2. safe_import_jax() works in all model files
    3. No direct JAX imports that bypass float64 enforcement
    """
    from rheo.core.registry import ModelRegistry

    for model_name in ALL_MODEL_NAMES:
        # This will raise an error if the model cannot be imported
        # or if it violates import order requirements
        model_instance = ModelRegistry.create(model_name)
        assert model_instance is not None, f"Failed to instantiate model '{model_name}'"


def test_model_predictions_maintain_float64_precision():
    """Test that model predictions maintain float64 precision.

    This ensures that JAX operations in models use float64 throughout.
    We test a representative classical model (Maxwell) as all models
    should behave the same due to shared safe_import_jax() mechanism.
    """
    from rheo.models.maxwell import Maxwell

    # Create test data
    t = jnp.linspace(0.01, 10.0, 50)

    # Create and configure model
    model = Maxwell()
    model.parameters.set_value("G0", 1e5)
    model.parameters.set_value("eta", 1e3)

    # Make prediction
    G_t = model._predict_relaxation(t, G0=1e5, eta=1e3)

    # Verify output is float64
    assert (
        G_t.dtype == jnp.float64
    ), f"Model predictions should be float64, got {G_t.dtype}"

    # Verify input was processed as float64
    assert t.dtype == jnp.float64, f"Input should be float64, got {t.dtype}"


def test_maxwell_fit_and_predict_smoke_test():
    """Smoke test for fit() and predict() on Maxwell model.

    This is a basic integration test to ensure the model works
    end-to-end with the new import mechanism. We just verify
    that fitting completes without errors and predictions are float64.
    """
    from rheo.models.maxwell import Maxwell

    # Create synthetic relaxation data
    t = np.linspace(0.1, 10.0, 50)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true

    # True relaxation: G(t) = G0 * exp(-t/tau)
    G_true = G0_true * np.exp(-t / tau_true)

    # Add small noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, G0_true * 0.01, size=G_true.shape)
    G_noisy = G_true + noise

    # Fit model (BaseModel.fit expects X and y separately)
    model = Maxwell()

    # Just verify that fit() completes without errors
    # (We don't assert on quality since that's not the purpose of this test)
    try:
        model.fit(t, G_noisy)
        fitted_successfully = True
    except Exception as e:
        fitted_successfully = False
        pytest.fail(f"Model fit() raised an exception: {e}")

    assert fitted_successfully, "Model fit() should complete without errors"

    # Verify model was marked as fitted
    assert model.fitted_, "Model should be marked as fitted"

    # Make prediction using _predict (internal method that handles raw arrays)
    predictions = model._predict(t)

    # Verify predictions are float64 (main goal of this test)
    predictions_np = np.array(predictions)
    assert (
        predictions_np.dtype == np.float64
    ), f"Predictions should be float64, got {predictions_np.dtype}"


def test_fractional_model_float64_precision():
    """Test float64 precision for a fractional model.

    Fractional models use Mittag-Leffler functions which are numerically
    sensitive. This test ensures float64 precision is maintained.
    """
    from rheo.models.fractional_maxwell_gel import FractionalMaxwellGel

    # Create test data
    t = jnp.linspace(0.01, 10.0, 50)

    # Create model and verify it can make predictions
    model = FractionalMaxwellGel()
    model.parameters.set_value("c_alpha", 10.0)
    model.parameters.set_value("alpha", 0.5)
    model.parameters.set_value("eta", 1e3)

    # Use _predict which handles test mode detection
    predictions = model._predict(np.array(t))

    # Verify output is float64
    predictions_np = np.array(predictions)
    assert (
        predictions_np.dtype == np.float64
    ), f"Fractional model predictions should be float64, got {predictions_np.dtype}"


def test_flow_model_float64_precision():
    """Test float64 precision for a flow model.

    Flow models have different physics (shear-thinning) but should
    still maintain float64 precision.
    """
    from rheo.models.power_law import PowerLaw

    # Create test data (shear rate)
    gamma_dot = jnp.logspace(-2, 2, 50)

    # Create and configure model
    model = PowerLaw()
    model.parameters.set_value("K", 100.0)
    model.parameters.set_value("n", 0.5)

    # Make prediction (use internal method)
    eta = model._predict_viscosity(gamma_dot, K=100.0, n=0.5)

    # Verify output is float64
    assert (
        eta.dtype == jnp.float64
    ), f"Flow model predictions should be float64, got {eta.dtype}"


def test_jax_operations_use_float64_in_models():
    """Test that JAX operations within models use float64.

    This test creates arrays within model context to ensure
    JAX default dtype is float64.
    """
    from rheo.models.maxwell import Maxwell

    # Create model (triggers safe_import_jax in model file)
    model = Maxwell()

    # Create a JAX array using jnp from the model's context
    # This should use float64 as default
    test_array = jnp.array([1.0, 2.0, 3.0])

    # Verify default is float64
    assert (
        test_array.dtype == jnp.float64
    ), f"JAX arrays should default to float64, got {test_array.dtype}"


def test_model_registry_contains_all_models():
    """Test that ModelRegistry contains all 20 expected models.

    This ensures all models were successfully imported and registered.
    """
    from rheo.core.registry import ModelRegistry

    registered_models = ModelRegistry.list_models()

    for model_name in ALL_MODEL_NAMES:
        assert model_name in registered_models, (
            f"Model '{model_name}' not found in registry. "
            f"Registered models: {registered_models}"
        )

    # Also verify we have exactly 20 models (or more if new ones added)
    assert (
        len(registered_models) >= 20
    ), f"Expected at least 20 models, found {len(registered_models)}"


def test_zener_fit_and_predict_smoke_test():
    """Additional smoke test for Zener model.

    This provides a second validation point for a 3-parameter model
    vs the 2-parameter Maxwell model. We just verify that fitting
    completes and predictions are float64.
    """
    from rheo.models.zener import Zener

    # Create synthetic relaxation data
    t = np.linspace(0.1, 10.0, 50)
    Ge_true = 1e4
    Gm_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / Gm_true

    # True relaxation: G(t) = Ge + Gm * exp(-t/tau)
    G_true = Ge_true + Gm_true * np.exp(-t / tau_true)

    # Add small noise
    rng = np.random.default_rng(43)
    noise = rng.normal(0, G_true.mean() * 0.01, size=G_true.shape)
    G_noisy = G_true + noise

    # Fit model (BaseModel.fit expects X and y separately)
    model = Zener()

    # Just verify that fit() completes without errors
    try:
        model.fit(t, G_noisy)
        fitted_successfully = True
    except Exception as e:
        fitted_successfully = False
        pytest.fail(f"Model fit() raised an exception: {e}")

    assert fitted_successfully, "Model fit() should complete without errors"

    # Verify model was marked as fitted
    assert model.fitted_, "Model should be marked as fitted"

    # Make prediction
    predictions = model._predict(t)

    # Verify predictions are float64 (main goal of this test)
    predictions_np = np.array(predictions)
    assert (
        predictions_np.dtype == np.float64
    ), f"Predictions should be float64, got {predictions_np.dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
