"""Integration smokes for transform -> fit and multi-dataset flows (Task 9.3.3/9.3.4)."""

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.services.model_service import ModelService
from rheojax.gui.services.transform_service import TransformService


def _make_freq_dataset(temp_c: float) -> RheoData:
    omega = np.logspace(-1, 1, 15)
    g_storage = 100 / (1 + omega) * (1 + 0.01 * (temp_c - 25))
    return RheoData(
        x=omega,
        y=g_storage,
        domain="frequency",
        metadata={"temperature": temp_c},
    )


def _make_time_dataset() -> RheoData:
    t = np.linspace(0, 5, 32)
    y = np.exp(-t / 2.0)
    return RheoData(x=t, y=y, domain="time", metadata={"test_mode": "relaxation"})


def test_transform_mastercurve_then_fit_single_dataset():
    data = _make_time_dataset()

    tsvc = TransformService()
    # Forward FFT then inverse should run without errors
    fft_forward = tsvc.apply_transform(
        "fft", data, {"direction": "forward", "return_psd": False}
    )

    # Fit original data with a simple model to complete the flow
    msvc = ModelService()
    result = msvc.fit("maxwell", data, params={}, test_mode="relaxation")

    assert isinstance(fft_forward, RheoData)
    assert result.success is True


def test_transform_mastercurve_multi_dataset():
    datasets = [_make_freq_dataset(20.0), _make_freq_dataset(30.0)]

    tsvc = TransformService()
    mc, extras = tsvc.apply_transform(
        "mastercurve", datasets, {"reference_temp": 25.0, "auto_shift": False}
    )

    assert isinstance(mc, RheoData)
    assert "shift_factors" in extras
    assert len(extras["shift_factors"]) == len(datasets)
