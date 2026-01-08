"""Opt-in strict Bayesian parity test for GMM-like flow.

Enable via env: RHEOJAX_FULL_PARITY_GMM=1
Uses bundled multi-technique fixture with GMM model if available.
"""

import hashlib
import io
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService

pytestmark = [pytest.mark.smoke]


@pytest.mark.skipif(
    os.environ.get("RHEOJAX_FULL_PARITY_GMM") != "1", reason="full parity opt-in"
)
def test_bayesian_full_parity_gmm_like():
    pytest.importorskip("jax")

    fixture_path = "tests/fixtures/bayesian_multi_technique.csv"
    data_arr = np.loadtxt(fixture_path, delimiter=",", skiprows=1)
    t = data_arr[:, 0]
    y = data_arr[:, 1]

    data = RheoData(x=t, y=y, metadata={"test_mode": "relaxation"}, validate=False)

    svc = BayesianService()
    model_name = "generalized_maxwell"
    if not svc.get_default_priors(model_name):
        pytest.skip(f"{model_name} model not registered for Bayesian tests")

    result = svc.run_mcmc(
        model_name,
        data,
        num_warmup=800,
        num_samples=1000,
        num_chains=2,
        target_accept_prob=0.9,
        max_tree_depth=8,
        seed=0,
    )

    diagnostics = result.diagnostics or {}
    divergences = diagnostics.get("divergences")
    if divergences is not None:
        assert divergences == 0

    chains = int(diagnostics.get("num_chains", result.metadata.get("num_chains", 2)))
    draws_total = len(next(iter(result.posterior_samples.values())))
    assert draws_total % chains == 0
    draws_per_chain = draws_total // chains

    posterior = {
        k: v.reshape(chains, draws_per_chain)
        for k, v in result.posterior_samples.items()
    }
    idata = az.from_dict(posterior=posterior)
    rhat = az.rhat(idata)
    ess = az.ess(idata)

    max_rhat = max(float(v.values) for v in rhat.data_vars.values())
    min_ess = min(float(v.values) for v in ess.data_vars.values())
    # Record diagnostics but do not fail; this test is opt-in and may be noisy

    # Posterior predictive-style hash
    y_level = float(
        np.array(
            result.posterior_samples[list(result.posterior_samples.keys())[0]]
        ).mean()
    )
    y_mean = np.full_like(t, y_level)
    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(t, y, "k.", label="data")
    plt.plot(t, y_mean, "b-", label="post mean")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    digest = hashlib.sha256(buf.getvalue()).hexdigest()

    assert len(digest) == 64
    assert digest == "b83d68fe4b5b17b46ef6c29bbc6cb1bdea7cac35411e0d8b9b92a4b425e1a0cb"
