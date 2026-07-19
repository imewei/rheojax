"""Tests for BayesianService.compare_models under ArviZ 1.x (WAIC removed)."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.core.arviz_utils import inference_data_from_dict
from rheojax.gui.foundation.state import BayesianResult
from rheojax.gui.services.bayesian_service import BayesianService


def _make_result(model_name: str, seed: int) -> BayesianResult:
    rng = np.random.default_rng(seed)
    posterior = {
        "a": rng.normal(size=(2, 50)),
        "b": rng.normal(size=(2, 50)),
    }
    log_likelihood = {"obs": rng.normal(loc=-2.0, scale=0.3, size=(2, 50, 20))}
    idata = inference_data_from_dict(
        {"posterior": posterior, "log_likelihood": log_likelihood}
    )
    return BayesianResult(
        model_name=model_name,
        dataset_id="ds-1",
        posterior_samples={k: v.reshape(-1) for k, v in posterior.items()},
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.0,
        timestamp=datetime.now(),
        inference_data=idata,
    )


def test_compare_models_default_criterion_is_loo() -> None:
    service = BayesianService()
    results = [_make_result("model_a", seed=0), _make_result("model_b", seed=1)]

    comparison = service.compare_models(results)

    assert set(comparison) == {"model_a", "model_b"}
    assert all(isinstance(v, float) for v in comparison.values())


def test_compare_models_explicit_waic_raises_value_error() -> None:
    service = BayesianService()

    with pytest.raises(ValueError, match="WAIC removed"):
        service.compare_models([], criterion="waic")
