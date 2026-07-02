from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
import rheojax.models  # noqa: F401
from rheojax.core.data import RheoData
from rheojax.gui.services.bayesian_service import BayesianService


class _StopEarly(Exception):
    """Raised from a patched create_instance() to abort before real NUTS runs."""


def test_run_mcmc_passes_model_config_to_create_instance(monkeypatch):
    # Regression: NUTS sampling never forwarded the user's Step 1
    # model_config (n_modes/variant/kinetics/...) to the model constructor --
    # only a warm-start-parameter-name heuristic (infer_model_kwargs) was
    # used, so an explicit model_config was silently ignored for NUTS while
    # NLSQ's ModelService.fit() honored it.
    calls = {}

    def fake_create_instance(model_name, plugin_type=None, **kwargs):
        calls["kwargs"] = kwargs
        raise _StopEarly

    service = BayesianService()
    monkeypatch.setattr(service._registry, "create_instance", fake_create_instance)

    data = RheoData(
        x=np.linspace(0.1, 10.0, 20),
        y=np.linspace(1.0, 5.0, 20),
        initial_test_mode="oscillation",
    )
    with pytest.raises(RuntimeError):  # run_mcmc wraps all exceptions in RuntimeError
        service.run_mcmc(
            "generalized_maxwell",
            data,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            test_mode="oscillation",
            model_config={"n_modes": 3},
        )
    assert calls["kwargs"] == {"n_modes": 3}


def test_run_mcmc_model_config_overrides_warm_start_inference(monkeypatch):
    # Explicit model_config must win over the infer_model_kwargs() heuristic
    # (which would otherwise infer n_modes=2 from the G_1/G_2 warm-start keys).
    calls = {}

    def fake_create_instance(model_name, plugin_type=None, **kwargs):
        calls["kwargs"] = kwargs
        raise _StopEarly

    service = BayesianService()
    monkeypatch.setattr(service._registry, "create_instance", fake_create_instance)

    data = RheoData(
        x=np.linspace(0.1, 10.0, 20),
        y=np.linspace(1.0, 5.0, 20),
        initial_test_mode="oscillation",
    )
    with pytest.raises(RuntimeError):  # run_mcmc wraps all exceptions in RuntimeError
        service.run_mcmc(
            "generalized_maxwell",
            data,
            num_warmup=1,
            num_samples=1,
            num_chains=1,
            warm_start={"G_1": 1.0, "G_2": 2.0},
            test_mode="oscillation",
            model_config={"n_modes": 5},
        )
    assert calls["kwargs"]["n_modes"] == 5
