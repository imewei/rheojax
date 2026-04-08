"""Bayesian edge case tests.

Tests degenerate configurations for Bayesian inference (num_chains=1,
posterior shape, divergence reporting).
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

import rheojax.models  # noqa: F401
from rheojax.models.classical.maxwell import Maxwell


@pytest.mark.slow
class TestBayesianEdgeCases:
    """Bayesian inference edge cases using Maxwell model."""

    def test_single_chain(self):
        """fit_bayesian with num_chains=1 should complete (degenerate R-hat)."""
        model = Maxwell()
        t = np.logspace(-2, 2, 30)
        G = 1000.0 * np.exp(-t / 1.0)
        # Fit NLSQ first for warm start
        model.fit(t, G, test_mode="relaxation")
        result = model.fit_bayesian(
            t,
            G,
            test_mode="relaxation",
            num_chains=1,
            num_warmup=100,
            num_samples=200,
            seed=42,
        )
        assert result is not None
        assert result.posterior_samples is not None
        # With 1 chain, R-hat is degenerate but diagnostics should still exist
        assert result.diagnostics is not None

    def test_posterior_shape(self):
        """Posterior samples should have shape (num_chains * num_samples,) per param."""
        model = Maxwell()
        t = np.logspace(-2, 2, 30)
        G = 1000.0 * np.exp(-t / 1.0)
        model.fit(t, G, test_mode="relaxation")

        num_chains = 2
        num_samples = 100
        result = model.fit_bayesian(
            t,
            G,
            test_mode="relaxation",
            num_chains=num_chains,
            num_warmup=50,
            num_samples=num_samples,
            seed=42,
        )
        assert result.posterior_samples is not None
        for name in model.parameters.keys():
            if name in result.posterior_samples:
                samples = result.posterior_samples[name]
                total = num_chains * num_samples
                assert (
                    samples.size == total
                ), f"Expected {total} samples for {name}, got {samples.size}"

    def test_minimal_warmup(self):
        """Bayesian with very short warmup should complete."""
        model = Maxwell()
        t = np.logspace(-2, 2, 20)
        G = 1000.0 * np.exp(-t / 1.0)
        model.fit(t, G, test_mode="relaxation")
        result = model.fit_bayesian(
            t,
            G,
            test_mode="relaxation",
            num_chains=1,
            num_warmup=10,
            num_samples=50,
            seed=42,
        )
        assert result is not None
