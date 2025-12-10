"""Minimal headless Bayesian smoke test to verify service wiring."""

import os
import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService, BayesianResult


pytestmark = [pytest.mark.smoke]


def test_bayesian_headless_smoke() -> None:
    pytest.importorskip("jax")
    if os.environ.get("RHEOJAX_SKIP_MCMC"):
        pytest.skip("MCMC disabled via env")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    x = np.linspace(0, 1, 6)
    y = 0.6 * np.exp(-1.2 * x) + np.random.default_rng(0).normal(0, 0.01, size=x.shape)
    data = RheoData(x=x, y=y, metadata={"test_mode": "relaxation"}, validate=False)

    service = BayesianService()
    if not service.get_default_priors("maxwell"):
        pytest.skip("maxwell model not registered for Bayesian tests")

    progress: list[tuple[int, int, int]] = []
    result = service.run_mcmc(
        "maxwell",
        data,
        num_warmup=5,
        num_samples=10,
        num_chains=1,
        max_tree_depth=3,
        progress_callback=lambda c, i, total: progress.append((c, i, total)),
    )

    assert isinstance(result, BayesianResult)
    assert result.posterior_samples
    for arr in result.posterior_samples.values():
        assert np.isfinite(arr).all()

    diagnostics = result.diagnostics or {}
    # Allow divergences in this micro-run but ensure diagnostics are present
    assert "divergences" in diagnostics
    assert diagnostics.get("ess") is not None
    # Progress callback may be skipped in short runs; ensure no crash.
