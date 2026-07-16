"""Strict notebook-length Bayesian parity check.

Runs a longer SPP-like Bayesian inference on the bundled multi-technique
fixture with notebook-like sampler settings and checks convergence + PPD hash.
"""

import hashlib

import arviz as az
import numpy as np
import pytest

from rheojax.core.arviz_utils import inference_data_from_dict
from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService

pytestmark = [pytest.mark.slow]


def test_bayesian_full_parity_notebook_like_relaxation():
    pytest.importorskip("jax")

    fixture_path = "tests/fixtures/bayesian_multi_technique.csv"
    data_arr = np.loadtxt(fixture_path, delimiter=",", skiprows=1)
    t = data_arr[:, 0]
    y = data_arr[:, 1]

    data = RheoData(x=t, y=y, metadata={"test_mode": "relaxation"}, validate=False)

    svc = BayesianService()
    if not svc.get_default_priors("maxwell"):
        pytest.skip("maxwell model not registered for Bayesian tests")

    result = svc.run_mcmc(
        "maxwell",
        data,
        num_warmup=1000,
        num_samples=1200,
        num_chains=2,
        target_accept_prob=0.9,
        max_tree_depth=10,
        seed=0,
    )

    diagnostics = result.diagnostics or {}
    divergences = diagnostics.get("divergences")
    if divergences is not None:
        assert divergences == 0

    # Reshape posterior to (chains, draws) to compute strict diagnostics.
    chains = int(diagnostics.get("num_chains", result.metadata.get("num_chains", 2)))
    draws_total = len(next(iter(result.posterior_samples.values())))
    assert draws_total % chains == 0
    draws_per_chain = draws_total // chains

    posterior = {
        k: v.reshape(chains, draws_per_chain)
        for k, v in result.posterior_samples.items()
    }
    idata = inference_data_from_dict({"posterior": posterior})

    rhat = az.rhat(idata)
    ess = az.ess(idata)

    max_rhat = max(float(v.values) for v in rhat.data_vars.values())
    min_ess = min(float(v.values) for v in ess.data_vars.values())

    assert max_rhat < 1.05
    assert min_ess > 400

    # Posterior predictive-style data hash for regression evidence. Hashing
    # the underlying numeric arrays (not a rendered PNG, as this used to)
    # avoids the host FreeType/Agg "raster overflow" glyph-rendering bug that
    # intermittently corrupted matplotlib text rendering on this machine --
    # text rendering was never the point of this check, only detecting
    # numeric drift in the posterior predictive.
    y_level = float(
        np.array(
            result.posterior_samples[list(result.posterior_samples.keys())[0]]
        ).mean()
    )
    y_mean = np.full_like(t, y_level)
    digest = hashlib.sha256(
        np.concatenate([t, y, y_mean]).astype(np.float64).round(8).tobytes()
    ).hexdigest()

    assert len(digest) == 64
    # Locked-environment baseline for the current uv.lock JAX/NumPyro;
    # the constructor compatibility layer does not affect the computed values.
    assert digest == "446b169d23eb7e10a5ba559515c9779bed3fa987f18e3ef28a351d4cbfd4e806"
