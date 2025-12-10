"""Short Bayesian integration test using fixtures."""

import os

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService, BayesianResult


pytestmark = [pytest.mark.smoke]


def test_bayesian_integration_short_with_fixture():
    pytest.importorskip("jax")
    if os.environ.get("RHEOJAX_SKIP_MCMC"):
        pytest.skip("MCMC disabled via env")

    # Use bundled fixture
    fixture_path = "tests/fixtures/bayesian_multi_technique.csv"
    data_arr = np.loadtxt(fixture_path, delimiter=",", skiprows=1)
    x = data_arr[:, 0]
    y = data_arr[:, 1]
    data = RheoData(x=x, y=y, metadata={"test_mode": "relaxation"}, validate=False)

    svc = BayesianService()
    if not svc.get_default_priors("maxwell"):
        pytest.skip("maxwell model not registered for Bayesian tests")

    result = svc.run_mcmc(
        "maxwell",
        data,
        num_warmup=20,
        num_samples=40,
        num_chains=2,
        target_accept_prob=0.9,
        max_tree_depth=5,
    )

    assert isinstance(result, BayesianResult)
    assert result.posterior_samples
    diagnostics = result.diagnostics or {}
    assert diagnostics.get("divergences", 0) <= 5
    assert diagnostics.get("ess") is not None

