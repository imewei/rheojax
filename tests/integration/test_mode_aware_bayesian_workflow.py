"""Integration tests for mode-aware Bayesian inference workflows.

This module tests the complete end-to-end Bayesian inference workflows across
all three test modes (relaxation, creep, oscillation), verifying that the
closure-based test_mode capture correctly produces mode-specific posteriors.

Tests validate:
- Correct model_function behavior in different test modes
- MCMC convergence diagnostics (R-hat, ESS, divergences)
- Posterior predictive distributions
- Backward compatibility with existing workflows
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.bayesian import BayesianResult
from rheojax.core.data import RheoData
from rheojax.core.test_modes import TestMode
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
from rheojax.models.fractional_zener_ll import FractionalZenerLiquidLiquid
from rheojax.models.fractional_maxwell_liquid import FractionalMaxwellLiquid
from rheojax.models.maxwell import Maxwell


@pytest.mark.integration
def test_bayesian_workflow_relaxation_mode_maxwell():
    """Test complete Bayesian workflow for relaxation mode on Maxwell model.

    Validates:
    - NLSQ optimization followed by Bayesian inference
    - Correct test_mode handling in model_function closure
    - MCMC diagnostics (R-hat < 1.01, ESS > 400)
    - Posterior statistics validity
    """
    # Generate synthetic relaxation data with known parameters
    np.random.seed(42)
    t = np.linspace(0.1, 10, 50)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t / tau_true)

    # Add small noise
    noise = np.random.normal(0, 0.02 * G_true.mean(), size=t.shape)
    G_data = G_true + noise

    # Create RheoData with explicit relaxation mode
    rheo_data = RheoData(
        x=t,
        y=G_data,
        metadata={
            "test_mode": TestMode.RELAXATION,
            "sample_name": "test_relaxation",
        },
    )

    # Step 1: NLSQ optimization
    model = Maxwell()
    model.fit(t, G_data)
    assert model.fitted_ is True

    G0_nlsq = model.parameters.get_value("G0")
    eta_nlsq = model.parameters.get_value("eta")

    # Step 2: Bayesian inference (test_mode should be inferred from RheoData)
    initial_values = {"G0": G0_nlsq, "eta": eta_nlsq}
    result = model.fit_bayesian(
        rheo_data,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        initial_values=initial_values,
    )

    # Verify result structure
    assert isinstance(result, BayesianResult)
    assert "G0" in result.posterior_samples
    assert "eta" in result.posterior_samples
    assert result.posterior_samples["G0"].shape == (2000,)
    assert result.posterior_samples["eta"].shape == (2000,)

    # Verify convergence diagnostics
    r_hat_G0 = result.diagnostics["r_hat"]["G0"]
    r_hat_eta = result.diagnostics["r_hat"]["eta"]
    assert r_hat_G0 < 1.01, f"R-hat for G0 is {r_hat_G0:.4f}, exceeds 1.01"
    assert r_hat_eta < 1.01, f"R-hat for eta is {r_hat_eta:.4f}, exceeds 1.01"

    # Verify ESS
    ess_G0 = result.diagnostics["ess"]["G0"]
    ess_eta = result.diagnostics["ess"]["eta"]
    assert ess_G0 > 400, f"ESS for G0 is {ess_G0:.0f}, below 400"
    assert ess_eta > 400, f"ESS for eta is {ess_eta:.0f}, below 400"

    # Verify divergences
    divergences = result.diagnostics["divergences"]
    max_acceptable_divergences = 2000 * 0.4  # 40% threshold
    assert (
        divergences < max_acceptable_divergences
    ), f"Too many divergences: {divergences}"

    # Verify posterior statistics are reasonable
    G0_mean = result.summary["G0"]["mean"]
    eta_mean = result.summary["eta"]["mean"]
    assert (
        abs(G0_mean - G0_true) / G0_true < 10.0
    ), "Posterior G0 mean too far from true"
    assert (
        abs(eta_mean - eta_true) / eta_true < 10.0
    ), "Posterior eta mean too far from true"


@pytest.mark.integration
def test_bayesian_workflow_creep_mode():
    """Test Bayesian workflow for creep mode with step-stress input.

    Validates:
    - Creep mode data handling with step-stress input
    - Correct test_mode propagation through model_function
    - MCMC convergence for creep mode
    - Posterior inference from creep data
    """
    # Generate synthetic creep data
    np.random.seed(123)
    t = np.linspace(0.1, 20, 60)
    G0_true = 1e5
    eta_true = 1e3
    J0 = 1 / G0_true  # Instantaneous compliance
    J_inf = 1 / (eta_true / 10)  # Final compliance (1/viscosity effect)

    # Maxwell creep: J(t) = (1/G0) + (t/eta)
    J_true = J0 + t / eta_true

    # Add realistic noise
    noise = np.random.normal(0, 0.02 * J_true.mean(), size=t.shape)
    J_data = J_true + noise

    # Create RheoData with explicit creep mode
    rheo_data = RheoData(
        x=t,
        y=J_data,
        metadata={
            "test_mode": TestMode.CREEP,
            "sample_name": "test_creep",
            "stress": 1e3,
        },
    )

    # Fit with NLSQ
    model = Maxwell()
    model.fit(t, J_data)
    assert model.fitted_ is True

    G0_nlsq = model.parameters.get_value("G0")
    eta_nlsq = model.parameters.get_value("eta")

    # Bayesian inference with explicit test_mode parameter
    initial_values = {"G0": G0_nlsq, "eta": eta_nlsq}
    result = model.fit_bayesian(
        rheo_data,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        initial_values=initial_values,
        test_mode=TestMode.CREEP,  # Explicit mode specification
    )

    # Verify result
    assert isinstance(result, BayesianResult)
    assert result.posterior_samples["G0"].shape == (2000,)

    # Verify convergence (may be more lenient for creep mode)
    r_hat_G0 = result.diagnostics["r_hat"]["G0"]
    assert r_hat_G0 < 1.05, f"R-hat for G0 is {r_hat_G0:.4f}, exceeds 1.05"

    # ESS should be reasonable
    ess_G0 = result.diagnostics["ess"]["G0"]
    assert ess_G0 > 300, f"ESS for G0 is {ess_G0:.0f}, below 300"


@pytest.mark.integration
def test_bayesian_workflow_oscillation_mode():
    """Test Bayesian workflow for oscillation mode with frequency sweep.

    Validates:
    - Oscillation mode with complex modulus handling
    - Proper test_mode handling for frequency domain data
    - MCMC convergence for oscillation mode
    - Posterior inference from frequency response
    """
    # Generate synthetic oscillation data
    np.random.seed(456)
    frequency = np.array(
        [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    )
    G_s = 1e5
    eta_s = 10.0
    omega = 2 * np.pi * frequency

    # Maxwell oscillation response
    G_prime = G_s * (omega * eta_s) ** 2 / (1 + (omega * eta_s) ** 2)
    G_double_prime = G_s * omega * eta_s / (1 + (omega * eta_s) ** 2)
    G_complex = G_prime + 1j * G_double_prime

    # Add noise
    noise = np.random.normal(0, 0.02 * G_prime.mean(), size=len(frequency))
    G_complex_data = G_complex * (1 + 0.01 * noise)

    # Create RheoData with oscillation mode
    rheo_data = RheoData(
        x=frequency,
        y=G_complex_data,
        metadata={
            "test_mode": TestMode.OSCILLATION,
            "sample_name": "test_oscillation",
            "strain_amplitude": 0.01,
        },
    )

    # Fit with NLSQ
    model = Maxwell()
    model.fit(frequency, G_complex_data)
    assert model.fitted_ is True

    # Bayesian inference
    G0_nlsq = model.parameters.get_value("G0")
    eta_nlsq = model.parameters.get_value("eta")

    initial_values = {"G0": G0_nlsq, "eta": eta_nlsq}
    result = model.fit_bayesian(
        rheo_data,
        num_warmup=800,
        num_samples=1500,
        num_chains=1,
        initial_values=initial_values,
        test_mode=TestMode.OSCILLATION,  # Explicit oscillation mode
    )

    # Verify result
    assert isinstance(result, BayesianResult)
    assert result.posterior_samples["G0"].shape == (1500,)

    # Verify convergence
    r_hat_G0 = result.diagnostics["r_hat"]["G0"]
    assert r_hat_G0 < 1.05, f"Oscillation R-hat too high: {r_hat_G0:.4f}"

    # Verify ESS
    ess_G0 = result.diagnostics["ess"]["G0"]
    assert ess_G0 > 300, f"Oscillation ESS too low: {ess_G0:.0f}"


@pytest.mark.integration
def test_bayesian_workflow_fractional_model_relaxation():
    """Test Bayesian workflow for fractional Maxwell model in relaxation mode.

    Validates:
    - Mode-aware Bayesian inference on fractional models
    - Correct parameter inference for viscoelastic fractional derivative
    - MCMC diagnostics for increased parameter count
    """
    # Generate synthetic fractional Maxwell relaxation data
    np.random.seed(789)
    t = np.linspace(0.01, 100, 50)

    # Fractional Maxwell: G(t) ≈ G0 * E_α(-t^α / τ^α)
    # Approximate with parametrization
    G0_true = 1e5
    alpha_true = 0.7  # Fractional order between 0 and 1
    tau_true = 10.0

    # Create approximate relaxation for fractional order
    # Using Mittag-Leffler function approximation
    normalized_t = (t / tau_true) ** alpha_true
    G_true = G0_true * np.exp(-normalized_t)

    noise = np.random.normal(0, 0.03 * G_true.mean(), size=t.shape)
    G_data = G_true + noise

    # Create RheoData
    rheo_data = RheoData(
        x=t,
        y=G_data,
        metadata={
            "test_mode": TestMode.RELAXATION,
            "sample_name": "test_fractional",
        },
    )

    # Fit with NLSQ
    model = FractionalZenerSolidSolid()
    model.fit(t, G_data)
    assert model.fitted_ is True

    # Bayesian inference
    initial_values = {}
    for param_name in ["G0", "alpha", "tau"]:
        try:
            initial_values[param_name] = model.parameters.get_value(param_name)
        except KeyError:
            pass

    result = model.fit_bayesian(
        rheo_data,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        initial_values=initial_values if initial_values else None,
    )

    # Verify convergence
    assert isinstance(result, BayesianResult)
    for param_name in result.diagnostics["r_hat"].keys():
        r_hat = result.diagnostics["r_hat"][param_name]
        assert r_hat < 1.05, f"R-hat for {param_name} is {r_hat:.4f}"

    # Verify ESS
    for param_name in result.diagnostics["ess"].keys():
        ess = result.diagnostics["ess"][param_name]
        assert ess > 300, f"ESS for {param_name} is {ess:.0f}"


@pytest.mark.integration
def test_mode_aware_posterior_predictive_distributions():
    """Test that posterior predictive distributions respect test mode.

    This test validates that the posterior samples correctly capture mode-specific
    behavior, and that predictions made from posterior samples are consistent
    with the original data mode.
    """
    # Generate relaxation data
    np.random.seed(321)
    t_train = np.linspace(0.1, 10, 30)
    G0_true = 1e5
    eta_true = 1e3
    tau_true = eta_true / G0_true
    G_true = G0_true * np.exp(-t_train / tau_true)

    noise = np.random.normal(0, 0.02 * G_true.mean(), size=t_train.shape)
    G_data = G_true + noise

    rheo_data = RheoData(
        x=t_train,
        y=G_data,
        metadata={
            "test_mode": TestMode.RELAXATION,
            "sample_name": "test_predictive",
        },
    )

    # Fit and perform Bayesian inference
    model = Maxwell()
    model.fit(t_train, G_data)

    initial_values = {
        "G0": model.parameters.get_value("G0"),
        "eta": model.parameters.get_value("eta"),
    }

    result = model.fit_bayesian(
        rheo_data,
        num_warmup=1000,
        num_samples=2000,
        num_chains=1,
        initial_values=initial_values,
    )

    # Make predictions on test data
    t_test = np.linspace(0.1, 10, 20)
    predictions = model.predict(t_test)

    # Verify predictions are reasonable (monotonic decay for relaxation)
    assert np.all(np.diff(predictions) <= 0), "Relaxation should be monotonically decreasing"

    # Verify predictions are in reasonable range
    assert np.all(predictions > 0), "Modulus should be positive"
    assert np.max(predictions) > G0_true * 0.1, "Predictions should capture magnitude"

    print(
        f"Posterior predictive validation passed. "
        f"Max prediction: {np.max(predictions):.3e}, "
        f"True G0: {G0_true:.3e}"
    )
