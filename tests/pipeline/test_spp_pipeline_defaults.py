"""Pipeline smoke tests for SPP LAOS strain-sweep helper."""

from __future__ import annotations

import numpy as np

from rheojax.core.data import RheoData
from rheojax.pipeline.workflows import SPPAmplitudeSweepPipeline


def _make_laos_dataset(omega: float, gamma_0: float, n_points: int = 400) -> RheoData:
    t = np.linspace(0, 2 * np.pi / omega, n_points)
    strain = gamma_0 * np.sin(omega * t)
    stress = 100.0 * strain
    return RheoData(
        x=t,
        y=stress,
        domain="time",
        metadata={},
    )


def test_spp_pipeline_uses_rogers_defaults_and_sets_metadata():
    omega = 1.5
    gamma_levels = [0.2, 0.4]
    datasets = [_make_laos_dataset(omega, g) for g in gamma_levels]

    pipeline = SPPAmplitudeSweepPipeline(omega=omega)
    pipeline.run(datasets, gamma_0_values=gamma_levels)

    assert pipeline.n_harmonics == 39
    assert pipeline.step_size == 8
    assert pipeline.num_mode == 2

    # Metadata propagated for downstream Bayesian/NLSQ
    for data, g in zip(datasets, gamma_levels, strict=False):
        assert data.metadata["omega"] == omega
        assert data.metadata["gamma_0"] == g

    # Results stored per amplitude
    for g in gamma_levels:
        assert g in pipeline.results
        assert "sigma_sy" in pipeline.results[g]

