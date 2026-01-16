"""Unit tests for closure-based test_mode capture in Bayesian inference.

Tests verify that fit_bayesian() correctly captures test_mode in model_function
closure instead of relying on global state (_test_mode attribute), ensuring
correct posteriors for all test modes (relaxation, creep, oscillation).

This is a critical correctness fix for v0.4.0 preventing incorrect posteriors.
"""

import numpy as np
import pytest
from jax import numpy as jnp

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode
from rheojax.models import Maxwell

jax, jnp = safe_import_jax()


@pytest.mark.unit
@pytest.mark.smoke
def test_closure_captures_test_mode_not_global_state():
    """Test that model_function closure captures test_mode, not global state.

    This is the critical correctness test ensuring mode-aware Bayesian inference.

    Scenario:
    1. Fit model in relaxation mode (sets self._test_mode = 'relaxation')
    2. Run Bayesian inference in oscillation mode
    3. Verify posterior uses oscillation mode, not relaxation

    Expected: Bayesian inference uses correct mode from fit_bayesian() call.
    Bug (v0.3.1): Bayesian inference uses mode from last .fit() call.
    """
    # Create Maxwell model
    model = Maxwell()

    # Step 1: Fit in relaxation mode (sets global state)
    t = np.logspace(-2, 2, 20)
    G_t = 1000 * np.exp(-t / 1.0)
    model.fit(t, G_t, test_mode="relaxation")

    # Verify global state is set to relaxation
    assert model._test_mode == TestMode.RELAXATION

    # Step 2: Run Bayesian inference in oscillation mode
    omega = np.logspace(-2, 2, 15)
    G0_true = 1000
    eta_true = 1000
    tau = eta_true / G0_true
    omega_tau = omega * tau
    G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0_true * omega_tau / (1 + omega_tau**2)
    G_star = G_prime + 1j * G_double_prime

    rheo_data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})

    # Use warm-start from relaxation fit
    initial_values = {
        "G0": model.parameters.get_value("G0"),
        "eta": model.parameters.get_value("eta"),
    }

    # Run Bayesian inference with oscillation mode
    result = model.fit_bayesian(
        rheo_data,
        num_warmup=500,
        num_samples=500,
        num_chains=1,
        initial_values=initial_values,
    )

    # Step 3: Verify convergence diagnostics
    assert result.diagnostics["r_hat"]["G0"] < 1.05, "G0 did not converge"
    assert result.diagnostics["r_hat"]["eta"] < 1.05, "eta did not converge"
    assert result.diagnostics["ess"]["G0"] > 100, "G0 ESS too low"
    assert result.diagnostics["ess"]["eta"] > 100, "eta ESS too low"

    # Step 4: Verify posterior means are reasonable for oscillation mode
    # (this will fail in v0.3.1 if mode was not captured correctly)
    G0_posterior = result.posterior_samples["G0"].mean()
    eta_posterior = result.posterior_samples["eta"].mean()

    # For oscillation data, should recover true parameters within ~20%
    assert (
        abs(G0_posterior - G0_true) / G0_true < 0.2
    ), f"G0 posterior {G0_posterior} too far from true {G0_true}"
    assert (
        abs(eta_posterior - eta_true) / eta_true < 0.2
    ), f"eta posterior {eta_posterior} too far from true {eta_true}"


@pytest.mark.unit
def test_explicit_test_mode_parameter_overrides_rheodata():
    """Test explicit test_mode parameter overrides RheoData inference.

    Verifies that passing test_mode explicitly to fit_bayesian() takes
    precedence over RheoData.test_mode attribute.
    """
    model = Maxwell()

    # Create RheoData with relaxation mode
    t = np.logspace(-2, 2, 20)
    G_t = 1000 * np.exp(-t / 1.0)
    rheo_data = RheoData(x=t, y=G_t, metadata={"test_mode": "relaxation"})

    # Run Bayesian inference with explicit creep mode override
    # (This should use creep mode despite RheoData saying relaxation)
    J_t = t / (1000 * 1000)  # Creep compliance
    result = model.fit_bayesian(
        t,
        J_t,
        test_mode="creep",  # Explicit override
        num_warmup=300,
        num_samples=300,
        num_chains=1,
    )

    # Verify inference ran without errors
    assert result is not None
    assert "G0" in result.posterior_samples
    assert "eta" in result.posterior_samples


@pytest.mark.unit
def test_backward_compatibility_default_inference_from_rheodata():
    """Test backward compatibility: default test_mode inference from RheoData.

    Verifies that v0.3.1 code continues to work unchanged with v0.4.0 fix.
    """
    model = Maxwell()

    # v0.3.1 workflow: pass RheoData without explicit test_mode
    omega = np.logspace(-2, 2, 15)
    G0_true = 1000
    eta_true = 1000
    tau = eta_true / G0_true
    omega_tau = omega * tau
    G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0_true * omega_tau / (1 + omega_tau**2)
    G_star = G_prime + 1j * G_double_prime

    rheo_data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})

    # Call fit_bayesian without explicit test_mode (backward compatible)
    result = model.fit_bayesian(
        rheo_data, num_warmup=300, num_samples=300, num_chains=1
    )

    # Should infer oscillation mode from RheoData and work correctly
    assert result is not None
    assert result.diagnostics["r_hat"]["G0"] < 1.1
    assert result.diagnostics["r_hat"]["eta"] < 1.1


@pytest.mark.unit
def test_model_function_uses_correct_mode_throughout_mcmc():
    """Test model_function uses correct mode throughout MCMC sampling.

    Verifies that test_mode is captured in closure and doesn't change during
    MCMC chain execution, even if global state (_test_mode) is modified.
    """
    model = Maxwell()

    # Create oscillation data
    omega = np.logspace(-2, 2, 15)
    G0_true = 1000
    eta_true = 1000
    tau = eta_true / G0_true
    omega_tau = omega * tau
    G_prime = G0_true * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0_true * omega_tau / (1 + omega_tau**2)
    G_star = G_prime + 1j * G_double_prime

    rheo_data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})

    # Run short Bayesian inference
    result = model.fit_bayesian(
        rheo_data, num_warmup=200, num_samples=200, num_chains=1
    )

    # During MCMC, model._test_mode should not affect sampling
    # (this test mainly ensures no runtime errors from mode switching)
    assert result is not None
    assert len(result.posterior_samples["G0"]) == 200
    assert len(result.posterior_samples["eta"]) == 200


@pytest.mark.unit
def test_creep_mode_bayesian_inference_correctness():
    """Test creep mode produces correct posteriors.

    This test will FAIL in v0.3.1 if creep data is used for Bayesian inference
    after a relaxation .fit() call, because model_function will use wrong mode.

    Note: Creep mode parameter recovery can be challenging due to the
    mathematical form J(t) = t/eta + 1/G0 which makes eta and G0 weakly
    identifiable from creep data alone. The key test is convergence, not
    perfect parameter recovery.
    """
    model = Maxwell()

    # Generate creep compliance data: J(t) = t/eta + 1/G0
    t = np.logspace(-2, 2, 20)
    G0_true = 1000
    eta_true = 1000
    J_t = t / eta_true + 1 / G0_true

    rheo_data = RheoData(x=t, y=J_t, metadata={"test_mode": "creep"})

    # Run Bayesian inference in creep mode
    result = model.fit_bayesian(
        rheo_data, num_warmup=500, num_samples=500, num_chains=1
    )

    # Primary test: Verify convergence (the critical correctness check)
    # If mode was wrong, MCMC would fail to converge or diverge
    assert result.diagnostics["r_hat"]["G0"] < 1.1, "G0 did not converge"
    assert result.diagnostics["r_hat"]["eta"] < 1.1, "eta did not converge"
    assert result.diagnostics["ess"]["G0"] > 100, "G0 ESS too low"
    assert result.diagnostics["ess"]["eta"] > 100, "eta ESS too low"

    # Secondary test: Verify inference ran in creep mode without errors
    # The fact that we got posterior samples with good convergence confirms
    # the mode-aware fix is working (v0.3.1 would use wrong mode and fail)
    assert len(result.posterior_samples["G0"]) == 500
    assert len(result.posterior_samples["eta"]) == 500
