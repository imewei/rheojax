"""Minimal integration smoke: load -> fit pipeline using ModelService."""

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.services.model_service import ModelService


def test_load_fit_smoke_maxwell():
    # Synthetic relaxation data for Maxwell: G(t) = G0 * exp(-t/tau)
    t = np.linspace(0, 5, 20)
    G0, tau = 1.0, 2.0
    y = G0 * np.exp(-t / tau)

    data = RheoData(x=t, y=y, domain="time", metadata={"test_mode": "relaxation"})

    svc = ModelService()
    result = svc.fit("maxwell", data, params={}, test_mode="relaxation")

    assert result.success is True
    assert np.isfinite(result.chi_squared)
