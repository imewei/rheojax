"""Test model_config passthrough: service -> worker -> registry.create."""

from __future__ import annotations

import nlsq  # noqa: F401 — must precede rheojax.core imports (float64 init)
import numpy as np

import rheojax.models  # noqa: F401 — ensure model registration
from rheojax.core.data import RheoData
from rheojax.gui.services.model_service import ModelService


def _relaxation_rheodata(x: np.ndarray, y: np.ndarray) -> RheoData:
    return RheoData(x=x, y=y, domain="time", metadata={"test_mode": "relaxation"})


def test_model_config_changes_param_set() -> None:
    """n_modes=2 via model_config must expose tau_1, tau_2 but not tau_3."""
    svc = ModelService()
    t = np.logspace(-2, 2, 20)
    y = 1000.0 * np.exp(-t)

    res = svc.fit(
        "generalized_maxwell",
        _relaxation_rheodata(t, y),
        {},
        model_config={"n_modes": 2},
    )
    names = set(res.parameters)
    assert "tau_1" in names, f"tau_1 missing from {names}"
    assert "tau_2" in names, f"tau_2 missing from {names}"
    assert "tau_3" not in names, f"tau_3 present but n_modes=2; got {names}"


def test_model_config_none_is_backward_compat() -> None:
    """Omitting model_config must give the same result as model_config=None."""
    svc = ModelService()
    t = np.logspace(-2, 2, 20)
    y = 1000.0 * np.exp(-t)
    data = _relaxation_rheodata(t, y)

    res_no_kwarg = svc.fit("generalized_maxwell", data, {})
    res_none = svc.fit("generalized_maxwell", data, {}, model_config=None)

    # Both default to n_modes=3; tau_3 must be present
    assert "tau_3" in set(res_no_kwarg.parameters)
    assert "tau_3" in set(res_none.parameters)
