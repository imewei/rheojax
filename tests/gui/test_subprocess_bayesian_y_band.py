"""y_band posterior-predictive band computation (subprocess_bayesian.py).

Regression test for step5_visualize.py's previously-dead-code y_band read:
VisualizeStep reads nuts_result["y_band"], but nothing set that key until
run_bayesian_isolated started computing it (see _compute_y_band).
"""

import numpy as np

from rheojax.gui.jobs.subprocess_bayesian import (
    _compute_y_band,
    _posterior_draw_indices,
    _posterior_params_at_index,
)


def test_posterior_draw_indices_evenly_spaced():
    samples = {"a": np.arange(100.0)}
    idx = _posterior_draw_indices(samples, max_draws=10)
    assert idx.size == 10
    assert idx[0] == 0
    assert idx[-1] == 99


def test_posterior_draw_indices_empty_when_no_samples():
    assert _posterior_draw_indices({}, max_draws=10).size == 0


def test_posterior_params_at_index_skips_non_finite():
    samples = {"a": np.array([1.0, np.nan, 3.0]), "b": np.array([10.0, 20.0, 30.0])}
    params = _posterior_params_at_index(samples, 1)
    assert "a" not in params
    assert params["b"] == 20.0


def test_compute_y_band_returns_none_without_posterior_samples():
    x = np.linspace(0, 1, 5)
    assert _compute_y_band("maxwell", {}, x, "relaxation", None, None) is None


def test_compute_y_band_shape_and_ordering(monkeypatch):
    x = np.linspace(0.0, 1.0, 5)
    posterior_samples = {"tau": np.array([1.0, 2.0, 3.0, 4.0, 5.0])}

    def fake_predict(
        self, model_name, params, x_values, test_mode=None, model_kwargs=None
    ):
        # y scales with tau so quantiles are easy to reason about.
        return params["tau"] * np.ones_like(x_values)

    monkeypatch.setattr(
        "rheojax.gui.services.model_service.ModelService.predict",
        fake_predict,
    )

    band = _compute_y_band(
        "maxwell", posterior_samples, x, "relaxation", None, None, hdi_prob=0.6
    )
    assert band is not None
    lo, hi = band
    assert lo.shape == x.shape
    assert hi.shape == x.shape
    assert np.all(lo <= hi)


def test_predict_does_not_mutate_caller_model_kwargs():
    """Regression: ModelService.predict() used to .pop("fitted_model_state")
    in place on the caller's model_kwargs dict. _compute_y_band's posterior-draw
    loop (and bayesian_page.py's equivalent) build model_kwargs once and reuse
    it across many predict() calls, so only the first draw ever saw
    fitted_model_state -- every later draw for stateful models (HL/DMT/HVM/VLB)
    silently lost it.
    """
    from rheojax.gui.services.model_service import ModelService

    svc = ModelService()
    model_kwargs = {"fitted_model_state": {"_last_fit_kwargs": {}}}
    x = np.linspace(0.1, 1.0, 5)

    svc.predict("maxwell", {}, x, test_mode="relaxation", model_kwargs=model_kwargs)
    assert "fitted_model_state" in model_kwargs

    # Second call with the same dict must still see it too.
    svc.predict("maxwell", {}, x, test_mode="relaxation", model_kwargs=model_kwargs)
    assert "fitted_model_state" in model_kwargs
