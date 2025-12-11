"""Regression test: BayesianService accepts DatasetState inputs."""

import numpy as np

from rheojax.gui.services.bayesian_service import BayesianService
from rheojax.gui.state.store import DatasetState


def test_bayesian_with_dataset_state_small_samples():
    ds = DatasetState("ds1", "synthetic", None, "relaxation")
    ds.x_data = np.linspace(0, 1, 10)
    ds.y_data = np.exp(-np.linspace(0, 1, 10))
    ds.metadata = {"test_mode": "relaxation"}
    ds.domain = "time"

    svc = BayesianService()
    result = svc.run_mcmc(
        "generalized_maxwell",
        ds,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
    )

    assert result.posterior_samples
