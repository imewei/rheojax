"""Minimal integration smoke: load -> bayesian pipeline using BayesianService.

Uses very small chain/sample counts to keep runtime low.
"""

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService


def test_load_bayesian_smoke_maxwell():
    t = np.linspace(0, 2, 15)
    G0, tau = 1.0, 1.0
    y = G0 * np.exp(-t / tau)

    data = RheoData(x=t, y=y, domain="time", metadata={"test_mode": "relaxation"})

    svc = BayesianService()
    priors = svc.get_default_priors("maxwell")

    result = svc.run_mcmc(
        model_name="maxwell",
        data=data,
        num_warmup=10,
        num_samples=10,
        num_chains=1,
        test_mode="relaxation",
    )

    assert result.diagnostics
    assert "num_warmup" in result.metadata

