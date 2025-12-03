"""Integration smoke tests for SPP pipeline workflow."""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.pipeline import BayesianPipeline


def _spp_dataset():
    omega = 1.0
    gamma_0 = np.array([0.2, 0.5, 1.0])
    sigma = 6.0 * gamma_0**0.25
    data = RheoData(
        x=gamma_0,
        y=sigma,
        domain="time",
        metadata={"test_mode": "oscillation", "omega": omega, "gamma_0": gamma_0},
    )
    return data


@pytest.mark.smoke
def test_pipeline_nlsq_only():
    data = _spp_dataset()
    pipeline = BayesianPipeline(data=data)

    pipeline.fit_nlsq("spp_yield_stress", omega=1.0, yield_type="static")

    assert pipeline._last_model is not None
    assert pipeline._last_model.parameters.get_value("sigma_sy_scale") > 0


@pytest.mark.slow
def test_pipeline_bayesian_warm_start():
    data = _spp_dataset()
    pipeline = BayesianPipeline(data=data)

    pipeline.fit_nlsq("spp_yield_stress", omega=1.0, yield_type="static")
    pipeline.fit_bayesian(num_samples=50, num_warmup=20)

    posterior = pipeline.get_posterior_summary()
    assert not posterior.empty
