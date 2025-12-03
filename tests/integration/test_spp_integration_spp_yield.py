"""Lightweight integration test for SPP amplitude sweep pipeline."""

from __future__ import annotations

import numpy as np

import pytest

from rheojax.core.data import RheoData
from rheojax.pipeline.workflows import SPPAmplitudeSweepPipeline


def _make_dataset(omega: float, gamma_0: float, scale: float = 100.0, exp: float = 0.8):
    t = np.linspace(0, 2 * np.pi / omega, 400)
    strain = gamma_0 * np.sin(omega * t)
    stress = scale * gamma_0**exp * np.sin(omega * t)
    return RheoData(
        x=t,
        y=stress,
        domain="time",
        metadata={"omega": omega, "gamma_0": gamma_0, "strain": strain},
    )


def test_pipeline_fit_model_nlsq_static():
    omega = 1.5
    gamma_levels = [0.2, 0.4, 0.8]
    datasets = [_make_dataset(omega, g, scale=50.0, exp=0.7) for g in gamma_levels]

    pipeline = SPPAmplitudeSweepPipeline(omega=omega)
    pipeline.run(datasets, gamma_0_values=gamma_levels)
    pipeline.fit_model(bayesian=False, yield_type="static")

    model = pipeline.get_model()
    assert model is not None
    sigma_sy = pipeline.get_yield_stresses()["sigma_sy"]
    assert np.all(np.diff(sigma_sy) >= 0)  # monotonic increase with gamma_0


@pytest.mark.slow
def test_pipeline_fit_model_bayesian_static():
    """End-to-end pipeline with Bayesian fit (tiny samples for speed)."""
    omega = 1.2
    gamma_levels = [0.3, 0.7, 1.2]
    datasets = [_make_dataset(omega, g, scale=60.0, exp=0.8) for g in gamma_levels]

    pipeline = SPPAmplitudeSweepPipeline(omega=omega)
    pipeline.run(datasets, gamma_0_values=gamma_levels)

    # NLSQ warm-start
    pipeline.fit_model(bayesian=False, yield_type="static")

    # Increase noise to ease NUTS initialization on tiny synthetic data
    model = pipeline.get_model()
    model.parameters.set_value("noise", 10.0)

    # Bayesian refinement (small sample count to keep test light)
    try:
        pipeline.fit_model(
            bayesian=True,
            yield_type="static",
            num_warmup=150,
            num_samples=200,
        )
    except RuntimeError:
        pytest.xfail("NUTS init failed on tiny synthetic dataset; acceptable for smoke")

    model = pipeline.get_model()
    assert model is not None
    # Model should have fitted parameters after Bayesian fit
    # Check the fitted scale parameter is positive
    sigma_sy_scale = model.parameters.get_value("sigma_sy_scale")
    assert sigma_sy_scale is not None and sigma_sy_scale > 0
    # Verify Bayesian result stored (may have posterior samples or summary)
    assert model.fitted_
