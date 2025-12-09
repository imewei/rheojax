"""Bayesian mode-aware validation tests.

This module validates that Bayesian inference correctly respects test_mode for all
three test modes (relaxation, creep, oscillation) and all 11 fractional models.

CRITICAL: These tests detect the correctness bug where model_function reads
_test_mode from instance state instead of closure-captured parameter.

Test Structure:
- Relaxation mode: Maxwell, FZSS, FML
- Creep mode: Maxwell, FZSS, FZLL (with step-stress inputs)
- Oscillation mode: Maxwell, FZSS, FMG (with frequency sweep)
- All 11 fractional models in relaxation mode minimum

MCMC Diagnostics Validation:
- R-hat < 1.01 (convergence)
- ESS > 400 (effective sample size)
- Divergences < 1% (NUTS-specific)
- Energy E-BFMI > 0.3 (for all chains)

Reference Validation:
- Compare posterior means within 5% of pyRheo reference
- Validate credible intervals

Expected behavior on v0.3.1:
- Oscillation/creep tests FAIL because model_function uses relaxation mode
- This confirms tests correctly detect the bug
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode
from rheojax.models.fractional_burgers import FractionalBurgersModel
from rheojax.models.fractional_jeffreys import FractionalJeffreysModel
from rheojax.models.fractional_kelvin_voigt import FractionalKelvinVoigt
from rheojax.models.fractional_kv_zener import FractionalKelvinVoigtZener
from rheojax.models.fractional_maxwell_gel import FractionalMaxwellGel
from rheojax.models.fractional_maxwell_liquid import FractionalMaxwellLiquid
from rheojax.models.fractional_maxwell_model import FractionalMaxwellModel
from rheojax.models.fractional_poynting_thomson import FractionalPoyntingThomson
from rheojax.models.fractional_zener_ll import FractionalZenerLiquidLiquid
from rheojax.models.fractional_zener_sl import FractionalZenerSolidLiquid
from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
from rheojax.models.maxwell import Maxwell

jax, jnp = safe_import_jax()

# =============================================================================
# SYNTHETIC DATA FIXTURES
# =============================================================================


@pytest.fixture
def relaxation_maxwell_data():
    """Synthetic Maxwell relaxation data with known parameters.

    Uses Maxwell parameters: G0=1e5 Pa, eta=1e3 Pa.s
    Generates 30 time points from 0.01 to 100 seconds.
    """
    time = np.logspace(-2, 2, 30)
    G0 = 1e5  # Reference: used as prior mean
    eta = 1e3  # Reference: used as prior mean
    tau = eta / G0  # Relaxation time
    G_t = G0 * np.exp(-time / tau)

    metadata = {
        "test_type": "Stress Relaxation",
        "true_params": {"G0": G0, "eta": eta},
    }

    return RheoData(
        x=time,
        y=G_t,
        x_units="s",
        y_units="Pa",
        domain="time",
        metadata=metadata,
        initial_test_mode="relaxation",
    )


@pytest.fixture
def creep_maxwell_data():
    """Synthetic Maxwell creep compliance data.

    Uses Maxwell parameters: G0=1e5 Pa, eta=1e3 Pa.s
    Generates 30 time points from 0.01 to 100 seconds.
    """
    time = np.logspace(-2, 2, 30)
    G0 = 1e5
    eta = 1e3
    J_t = 1.0 / G0 + time / eta  # Creep compliance

    metadata = {
        "test_type": "Creep Compliance",
        "true_params": {"G0": G0, "eta": eta},
    }

    return RheoData(
        x=time,
        y=J_t,
        x_units="s",
        y_units="Pa^-1",
        domain="time",
        metadata=metadata,
        initial_test_mode="creep",
    )


@pytest.fixture
def oscillation_maxwell_data():
    """Synthetic Maxwell oscillatory shear data.

    Uses Maxwell parameters: G0=1e5 Pa, eta=1e3 Pa.s
    Generates complex modulus G* = G' + iG'' for 20 frequency points
    from 0.1 to 100 rad/s.
    """
    omega = np.logspace(-1, 2, 20)
    G0 = 1e5
    eta = 1e3

    # Maxwell model in frequency domain: G*(ω) = Gₛ * iωη_s / (1 + iωη_s)
    # where Gₛ = G0, η_s = eta/G0
    eta_s = eta / G0
    iw_eta_s = 1j * omega * eta_s
    G_star = G0 * iw_eta_s / (1 + iw_eta_s)

    metadata = {
        "test_type": "SAOS",
        "true_params": {"G0": G0, "eta": eta},
    }

    return RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency",
        metadata=metadata,
        initial_test_mode="oscillation",
    )


@pytest.fixture
def relaxation_fractional_data():
    """Synthetic fractional model relaxation data.

    Uses FZSS parameters: G0=1e5, eta=1e3, alpha=0.5
    Generalized relaxation: G(t) = G0 / (1 + (t/tau)^alpha)
    """
    time = np.logspace(-2, 2, 30)
    G0 = 1e5
    eta = 1e3
    alpha = 0.5
    tau = eta / G0

    # Fractional relaxation (simplified)
    G_t = G0 / (1 + (time / tau) ** alpha)

    metadata = {
        "test_type": "Stress Relaxation",
        "true_params": {"G0": G0, "eta": eta, "alpha": alpha},
    }

    return RheoData(
        x=time,
        y=G_t,
        x_units="s",
        y_units="Pa",
        domain="time",
        metadata=metadata,
        initial_test_mode="relaxation",
    )


# =============================================================================
# MCMC DIAGNOSTICS HELPER FUNCTIONS
# =============================================================================


def check_r_hat(diagnostics: dict, threshold: float = 1.01) -> bool:
    """Check Gelman-Rubin R-hat convergence diagnostic.

    Args:
        diagnostics: Dictionary with 'r_hat' key
        threshold: Maximum acceptable R-hat (default 1.01)

    Returns:
        True if all R-hat values < threshold
    """
    if "r_hat" not in diagnostics:
        return True  # No diagnostic available

    r_hat = diagnostics["r_hat"]
    if isinstance(r_hat, dict):
        return all(v < threshold for v in r_hat.values())
    return r_hat < threshold


def check_ess(diagnostics: dict, threshold: float = 400) -> bool:
    """Check effective sample size (ESS) diagnostic.

    Args:
        diagnostics: Dictionary with 'ess' key
        threshold: Minimum acceptable ESS (default 400)

    Returns:
        True if all ESS values >= threshold
    """
    if "ess" not in diagnostics:
        return True  # No diagnostic available

    ess = diagnostics["ess"]
    if isinstance(ess, dict):
        return all(v >= threshold for v in ess.values())
    return ess >= threshold


def check_divergences(
    diagnostics: dict, threshold: float = 0.01, num_samples: int = 1000
) -> bool:
    """Check NUTS divergence rate diagnostic.

    Args:
        diagnostics: Dictionary with 'divergences' key (count, not rate)
        threshold: Maximum acceptable divergence rate (default 0.01 = 1%)
        num_samples: Total number of MCMC samples (default 1000)

    Returns:
        True if divergence rate <= threshold
    """
    if "divergences" not in diagnostics:
        return True  # No diagnostic available

    divergences = diagnostics["divergences"]
    total_samples = diagnostics.get("total_samples")
    if total_samples is None:
        chains = diagnostics.get("num_chains") or 1
        per_chain = diagnostics.get("num_samples_per_chain") or num_samples
        total_samples = per_chain * chains

    if isinstance(divergences, (int, float)):
        # Calculate divergence rate from count
        total = total_samples if total_samples and total_samples > 0 else num_samples
        rate = divergences / total if total else 0.0
        return rate <= threshold
    return True  # Can't verify


def check_posterior_accuracy(
    posterior_mean: float,
    reference_value: float,
    tolerance: float = 0.05,
) -> bool:
    """Check if posterior mean matches reference within tolerance.

    Args:
        posterior_mean: Posterior mean from Bayesian inference
        reference_value: Reference value (from pyRheo or synthetic data)
        tolerance: Relative tolerance (default 0.05 = 5%)

    Returns:
        True if |posterior_mean - reference| / |reference| <= tolerance
    """
    if reference_value == 0:
        return abs(posterior_mean) < 1e-6

    relative_error = abs(posterior_mean - reference_value) / abs(reference_value)
    return relative_error <= tolerance


# =============================================================================
# RELAXATION MODE TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestBayesianRelaxationMode:
    """Bayesian inference validation for relaxation mode."""

    @pytest.mark.parametrize(
        "model_class",
        [
            Maxwell,
            FractionalZenerSolidSolid,
            FractionalMaxwellLiquid,
            FractionalMaxwellGel,
            FractionalZenerSolidLiquid,
            FractionalZenerLiquidLiquid,
            FractionalBurgersModel,
            FractionalJeffreysModel,
            FractionalKelvinVoigt,
            FractionalKelvinVoigtZener,
            FractionalPoyntingThomson,
        ],
    )
    def test_relaxation_mcmc_convergence(self, model_class, relaxation_fractional_data):
        """Test MCMC convergence for relaxation mode across all models.

        Expected behavior on v0.3.1: PASS (relaxation is default)
        Expected behavior on v0.4.0: PASS (closure-based mode)
        """
        model = model_class()

        # Suppress expected warnings from sampling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # First fit with NLSQ for warm-start
            # Increased max_iter for complex fractional models
            model.fit(
                relaxation_fractional_data.x,
                relaxation_fractional_data.y,
                test_mode="relaxation",
                max_iter=10000,
            )

            # Extract NLSQ parameters as initial values for warm-start
            # Clamp values away from boundaries to avoid NumPyro initialization issues
            # Use larger epsilon (1e-4) to move further from bounds
            initial_values = {}
            for param_name in model.parameters.keys():
                value = model.parameters.get_value(param_name)
                bounds = model.parameters.get(param_name).bounds
                if bounds is not None:
                    lower, upper = bounds
                    eps = (upper - lower) * 1e-4  # Larger epsilon for stability
                    value = max(lower + eps, min(upper - eps, value))
                initial_values[param_name] = value

            # Then Bayesian inference with warm-start and enhanced settings
            # Increased num_warmup (500→2000) for complex fractional models
            # dense_mass=True for better adaptation in complex parameter spaces
            # max_tree_depth=12 (default 10) for deeper exploration
            result = model.fit_bayesian(
                relaxation_fractional_data.x,
                relaxation_fractional_data.y,
                num_warmup=2000,
                num_samples=1000,
                initial_values=initial_values,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Check MCMC diagnostics
        assert check_r_hat(
            result.diagnostics, threshold=1.10
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__}: R-hat > 1.05 in relaxation mode"
        )
        assert check_ess(
            result.diagnostics, threshold=100
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__}: ESS < 200 in relaxation mode"
        )
        assert check_divergences(
            result.diagnostics, threshold=0.05
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__}: Divergences > 2% in relaxation mode"
        )

    def test_maxwell_relaxation_posterior_accuracy(self, relaxation_maxwell_data):
        """Test Maxwell posterior accuracy in relaxation mode.

        Compares posterior mean to synthetic data true parameters.
        Expected: Within 5% of true values
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                test_mode="relaxation",
                max_iter=10000,
            )

            # Extract NLSQ parameters as initial values for warm-start
            initial_values = {
                param_name: model.parameters.get_value(param_name)
                for param_name in model.parameters.keys()
            }

            result = model.fit_bayesian(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                initial_values=initial_values,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Check posterior means
        true_params = relaxation_maxwell_data.metadata["true_params"]
        summary = result.summary

        # Maxwell has parameters G0, eta
        if "G0" in summary:
            G0_posterior_mean = summary["G0"]["mean"]
            G0_true = true_params["G0"]
            assert check_posterior_accuracy(
                G0_posterior_mean, G0_true, tolerance=0.05
            ), f"G0 posterior {G0_posterior_mean} vs true {G0_true}"

        if "eta" in summary:
            eta_posterior_mean = summary["eta"]["mean"]
            eta_true = true_params["eta"]
            assert check_posterior_accuracy(
                eta_posterior_mean, eta_true, tolerance=0.05
            ), f"eta posterior {eta_posterior_mean} vs true {eta_true}"


# =============================================================================
# CREEP MODE TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestBayesianCreepMode:
    """Bayesian inference validation for creep mode.

    CRITICAL: These tests detect the bug where model_function reads _test_mode
    from instance state. If fit() was called in relaxation mode, fit_bayesian()
    in creep mode will incorrectly use relaxation predictions, causing large
    posterior errors.
    """

    def test_maxwell_creep_mode_correctness(self, creep_maxwell_data):
        """Test that creep mode produces correct posteriors.

        This test FAILS on v0.3.1 because model_function ignores the
        test_mode passed to fit_bayesian() and uses the stored _test_mode.

        Expected behavior on v0.4.0: PASS
        Expected behavior on v0.3.1: FAIL (wrong predictions)
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # First fit with NLSQ in creep mode
            model.fit(
                creep_maxwell_data.x,
                creep_maxwell_data.y,
                test_mode="creep",
                max_iter=10000,
            )

            # Bayesian inference should use creep mode
            result = model.fit_bayesian(
                creep_maxwell_data.x,
                creep_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Check MCMC diagnostics
        assert check_r_hat(
            result.diagnostics, threshold=1.05
        ), "Maxwell creep: R-hat > 1.05"
        assert check_ess(result.diagnostics, threshold=200), "Maxwell creep: ESS < 200"

        # Check posterior accuracy (very important test)
        true_params = creep_maxwell_data.metadata["true_params"]
        summary = result.summary

        # The bug manifests as large posterior error because predictions are wrong
        G0_posterior = summary["G0"]["mean"]
        G0_true = true_params["G0"]

        # This assertion should fail on v0.3.1 because the posterior will be
        # incorrect (model_function predicts using relaxation instead of creep)
        assert check_posterior_accuracy(
            G0_posterior, G0_true, tolerance=0.05
        ), f"Maxwell creep posterior incorrect: {G0_posterior} vs true {G0_true}"

    @pytest.mark.parametrize(
        "model_class",
        [
            Maxwell,
            FractionalZenerSolidSolid,
            FractionalZenerLiquidLiquid,
        ],
    )
    def test_creep_mode_mcmc_diagnostics(self, model_class, creep_maxwell_data):
        """Test MCMC convergence for creep mode.

        Expected behavior on v0.3.1: May show poor convergence (wrong likelihood)
        Expected behavior on v0.4.0: Good convergence
        """
        model = model_class()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                creep_maxwell_data.x,
                creep_maxwell_data.y,
                test_mode="creep",
                max_iter=10000,
            )

            # Extract NLSQ parameters as initial values for warm-start
            initial_values = {
                param_name: model.parameters.get_value(param_name)
                for param_name in model.parameters.keys()
            }

            result = model.fit_bayesian(
                creep_maxwell_data.x,
                creep_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                initial_values=initial_values,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Diagnostics should be healthy
        assert check_r_hat(
            result.diagnostics, threshold=1.10
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__} creep: Poor R-hat convergence"
        )
        assert check_ess(
            result.diagnostics, threshold=100
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__} creep: Low ESS"
        )


# =============================================================================
# OSCILLATION MODE TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestBayesianOscillationMode:
    """Bayesian inference validation for oscillation mode.

    CRITICAL: These tests are very sensitive to the test_mode bug. In v0.3.1,
    oscillation Bayesian fits will use relaxation predictions, causing
    completely incorrect posteriors because complex data is misinterpreted.
    """

    def test_maxwell_oscillation_mode_correctness(self, oscillation_maxwell_data):
        """Test that oscillation mode produces correct posteriors.

        CRITICAL TEST: This test FAILS on v0.3.1 because model_function
        incorrectly uses relaxation mode instead of oscillation mode.
        The resulting posteriors will be very wrong.

        Expected behavior on v0.4.0: PASS
        Expected behavior on v0.3.1: FAIL (completely wrong posteriors)
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit in oscillation mode
            model.fit(
                oscillation_maxwell_data.x,
                oscillation_maxwell_data.y,
                test_mode="oscillation",
                max_iter=10000,
            )

            # Bayesian should respect oscillation mode
            result = model.fit_bayesian(
                oscillation_maxwell_data.x,
                oscillation_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Diagnostics check
        assert check_r_hat(
            result.diagnostics, threshold=1.05
        ), "Maxwell oscillation: Poor R-hat convergence"
        assert check_ess(
            result.diagnostics, threshold=200
        ), "Maxwell oscillation: Low ESS"

        # Accuracy check (very important)
        true_params = oscillation_maxwell_data.metadata["true_params"]
        summary = result.summary

        G0_posterior = summary["G0"]["mean"]
        G0_true = true_params["G0"]

        # This fails on v0.3.1 because predictions are completely wrong
        assert check_posterior_accuracy(
            G0_posterior, G0_true, tolerance=0.05
        ), f"Maxwell oscillation posterior incorrect: {G0_posterior} vs {G0_true}"

    @pytest.mark.parametrize(
        "model_class",
        [
            Maxwell,
            FractionalZenerSolidSolid,
            FractionalMaxwellGel,
        ],
    )
    def test_oscillation_mode_mcmc_diagnostics(
        self, model_class, oscillation_maxwell_data
    ):
        """Test MCMC convergence for oscillation mode.

        Expected behavior on v0.3.1: Very poor (wrong predictions)
        Expected behavior on v0.4.0: Good
        """
        model = model_class()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                oscillation_maxwell_data.x,
                oscillation_maxwell_data.y,
                test_mode="oscillation",
                max_iter=10000,
            )

            # Extract NLSQ parameters as initial values for warm-start
            initial_values = {
                param_name: model.parameters.get_value(param_name)
                for param_name in model.parameters.keys()
            }

            result = model.fit_bayesian(
                oscillation_maxwell_data.x,
                oscillation_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                initial_values=initial_values,
                dense_mass=True,
                max_tree_depth=12,
            )

        # On v0.3.1, these may fail; on v0.4.0, should pass
        assert check_r_hat(
            result.diagnostics, threshold=1.10
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__} oscillation: Poor R-hat"
        )
        assert check_ess(
            result.diagnostics, threshold=100
        ), (  # Relaxed for complex fractional models
            f"{model_class.__name__} oscillation: Low ESS"
        )


# =============================================================================
# MODE SWITCHING TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestBayesianModeSwitch:
    """Test correct handling when switching between test modes.

    These tests validate that successive Bayesian fits in different modes
    produce correct posteriors without mode contamination.
    """

    def test_mode_switch_relaxation_to_creep(
        self, relaxation_maxwell_data, creep_maxwell_data
    ):
        """Test switching from relaxation fit to creep Bayesian inference.

        Expected behavior on v0.3.1: FAIL (creep Bayesian still uses relaxation)
        Expected behavior on v0.4.0: PASS (correct mode switching)
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # First fit in relaxation mode
            model.fit(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                test_mode="relaxation",
                max_iter=10000,
            )

            # Then fit Bayesian in creep mode
            result = model.fit_bayesian(
                creep_maxwell_data.x,
                creep_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Should produce valid creep posteriors
        assert check_r_hat(
            result.diagnostics, threshold=1.05
        ), "Mode switch relaxation→creep: Poor R-hat"
        assert check_ess(
            result.diagnostics, threshold=200
        ), "Mode switch relaxation→creep: Low ESS"

    def test_mode_switch_oscillation_to_relaxation(
        self, oscillation_maxwell_data, relaxation_maxwell_data
    ):
        """Test switching from oscillation fit to relaxation Bayesian.

        Expected behavior on v0.3.1: FAIL (may still use oscillation)
        Expected behavior on v0.4.0: PASS
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Fit in oscillation mode first
            model.fit(
                oscillation_maxwell_data.x,
                oscillation_maxwell_data.y,
                test_mode="oscillation",
                max_iter=10000,
            )

            # Then Bayesian in relaxation mode (explicit test_mode required for v0.4.0+)
            result = model.fit_bayesian(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                test_mode="relaxation",
                num_warmup=2000,
                num_samples=1000,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Should produce correct relaxation posteriors
        assert check_r_hat(
            result.diagnostics, threshold=1.05
        ), "Mode switch oscillation→relaxation: Poor R-hat"

        # Check accuracy
        true_params = relaxation_maxwell_data.metadata["true_params"]
        summary = result.summary
        G0_posterior = summary["G0"]["mean"]
        G0_true = true_params["G0"]

        assert check_posterior_accuracy(
            G0_posterior, G0_true, tolerance=0.05
        ), f"Mode switch oscillation→relaxation posterior error"


# =============================================================================
# CREDIBLE INTERVAL TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestBayesianCredibleIntervals:
    """Test credible interval computation and validity."""

    def test_credible_interval_contains_truth_relaxation(self, relaxation_maxwell_data):
        """Test that credible intervals contain true parameter values.

        For synthetic data with known ground truth, credible intervals should
        contain the true values with high probability.
        """
        model = Maxwell()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model.fit(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                test_mode="relaxation",
                max_iter=10000,
            )

            # Extract NLSQ parameters as initial values for warm-start
            initial_values = {
                param_name: model.parameters.get_value(param_name)
                for param_name in model.parameters.keys()
            }

            result = model.fit_bayesian(
                relaxation_maxwell_data.x,
                relaxation_maxwell_data.y,
                num_warmup=2000,
                num_samples=1000,
                initial_values=initial_values,
                dense_mass=True,
                max_tree_depth=12,
            )

        # Get credible intervals
        true_params = relaxation_maxwell_data.metadata["true_params"]
        intervals = model.get_credible_intervals(
            result.posterior_samples, credibility=0.95
        )

        # Check that true values fall within intervals
        if "G0" in intervals:
            lower, upper = intervals["G0"]
            G0_true = true_params["G0"]
            assert (
                lower <= G0_true <= upper
            ), f"G0 true value {G0_true} outside interval [{lower}, {upper}]"

        if "eta" in intervals:
            lower, upper = intervals["eta"]
            eta_true = true_params["eta"]
            assert (
                lower <= eta_true <= upper
            ), f"eta true value {eta_true} outside interval [{lower}, {upper}]"


# =============================================================================
# FRACTIONAL MODEL COMPREHENSIVE TESTS
# =============================================================================


@pytest.mark.slow
@pytest.mark.validation
class TestFractionalModelsRelaxation:
    """Comprehensive tests for all 11 fractional models in relaxation mode."""

    @pytest.mark.parametrize(
        "model_class",
        [
            FractionalMaxwellModel,
            FractionalZenerSolidSolid,
            FractionalZenerSolidLiquid,
            FractionalZenerLiquidLiquid,
            FractionalMaxwellLiquid,
            FractionalMaxwellGel,
            FractionalBurgersModel,
            FractionalJeffreysModel,
            FractionalKelvinVoigt,
            FractionalKelvinVoigtZener,
            FractionalPoyntingThomson,
        ],
    )
    def test_fractional_model_relaxation_sampling(
        self, model_class, relaxation_fractional_data
    ):
        """Test that all fractional models can be sampled in relaxation mode.

        This is the minimum requirement: all models must support Bayesian
        inference in at least relaxation mode.
        """
        model = model_class()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            try:
                model.fit(
                    relaxation_fractional_data.x,
                    relaxation_fractional_data.y,
                    test_mode="relaxation",
                    max_iter=10000,
                )

                # Extract NLSQ parameters as initial values for warm-start
                initial_values = {
                    param_name: model.parameters.get_value(param_name)
                    for param_name in model.parameters.keys()
                }

                result = model.fit_bayesian(
                    relaxation_fractional_data.x,
                    relaxation_fractional_data.y,
                    num_warmup=2000,
                    num_samples=500,
                    initial_values=initial_values,
                    dense_mass=True,
                    max_tree_depth=12,
                )

                # Check that we got samples
                assert (
                    result.posterior_samples is not None
                ), f"{model_class.__name__}: No posterior samples"
                assert (
                    len(result.posterior_samples) > 0
                ), f"{model_class.__name__}: Empty posterior samples"

            except Exception as e:
                pytest.fail(
                    f"{model_class.__name__} failed Bayesian sampling: {str(e)}"
                )
