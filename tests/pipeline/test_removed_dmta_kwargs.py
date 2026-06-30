"""Behavioral guards for removed DMTA kwargs at pipeline boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.pipeline import BatchPipeline, Pipeline
from rheojax.pipeline.bayesian import BayesianPipeline
from rheojax.pipeline.workflows import ModelComparisonPipeline

REMOVED_KWARGS = (
    ("deformation_mode", "tension"),
    ("poisson_ratio", 0.5),
)


class _SpyModel:
    """Permissive duck-typed model that records every received kwarg."""

    fit_calls: list[dict] = []
    bayesian_calls: list[dict] = []

    def __init__(self):
        self.parameters = {}
        self.fitted_ = False
        self._last_fit_kwargs = {}
        self._last_bayesian_kwargs = {}

    @classmethod
    def reset(cls):
        cls.fit_calls = []
        cls.bayesian_calls = []

    def fit(self, X, y, **kwargs):
        type(self).fit_calls.append(dict(kwargs))
        self.fitted_ = True
        return self

    def fit_bayesian(self, X, y, **kwargs):
        type(self).bayesian_calls.append(dict(kwargs))
        return SimpleNamespace(
            diagnostics={},
            num_samples=kwargs.get("num_samples"),
            num_chains=kwargs.get("num_chains"),
        )

    def predict(self, X):
        return np.ones_like(np.asarray(X), dtype=float)

    def score(self, X, y):
        return 1.0

    def get_nlsq_result(self):
        return None

    def get_params(self):
        return {"scale": 1.0}


@pytest.fixture
def shear_data():
    return RheoData(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0, 1.0, 1.0]),
        metadata={"test_mode": "relaxation"},
        validate=False,
    )


@pytest.mark.parametrize(("removed_key", "value"), REMOVED_KWARGS)
def test_pipeline_fit_rejects_removed_kwargs_before_model(
    shear_data, removed_key, value
):
    _SpyModel.reset()
    model = _SpyModel()

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        Pipeline(data=shear_data).fit(model, **{removed_key: value})

    assert _SpyModel.fit_calls == []


def test_pipeline_fit_preserves_supported_kwargs(shear_data):
    _SpyModel.reset()

    Pipeline(data=shear_data).fit(
        _SpyModel(), method="scipy", test_mode="creep", max_iter=17
    )

    assert _SpyModel.fit_calls == [
        {"method": "scipy", "test_mode": "creep", "max_iter": 17}
    ]


@pytest.mark.parametrize(("removed_key", "value"), REMOVED_KWARGS)
def test_pipeline_bayesian_rejects_removed_kwargs_before_model(
    shear_data, removed_key, value
):
    _SpyModel.reset()
    pipeline = Pipeline(data=shear_data)
    pipeline._last_model = _SpyModel()

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        pipeline.fit_bayesian(**{removed_key: value})

    assert _SpyModel.bayesian_calls == []


def test_pipeline_bayesian_preserves_sampling_and_test_mode(shear_data):
    _SpyModel.reset()
    pipeline = Pipeline(data=shear_data)
    pipeline._last_model = _SpyModel()

    pipeline.fit_bayesian(
        seed=7,
        num_warmup=11,
        num_samples=13,
        num_chains=2,
        target_accept_prob=0.91,
    )

    assert _SpyModel.bayesian_calls == [
        {
            "test_mode": "relaxation",
            "seed": 7,
            "num_warmup": 11,
            "num_samples": 13,
            "num_chains": 2,
            "target_accept_prob": 0.91,
        }
    ]


@pytest.mark.parametrize(
    ("entrypoint", "removed_key", "value"),
    (
        ("nlsq", "deformation_mode", "tension"),
        ("bayesian", "poisson_ratio", 0.5),
    ),
)
def test_specialized_bayesian_pipeline_rejects_before_model(
    shear_data, entrypoint, removed_key, value
):
    _SpyModel.reset()
    pipeline = BayesianPipeline(data=shear_data)
    model = _SpyModel()

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        if entrypoint == "nlsq":
            pipeline.fit_nlsq(model, **{removed_key: value})
        else:
            pipeline._last_model = model
            pipeline.fit_bayesian(**{removed_key: value})

    assert _SpyModel.fit_calls == []
    assert _SpyModel.bayesian_calls == []


@pytest.mark.parametrize(("removed_key", "value"), REMOVED_KWARGS)
def test_model_comparison_rejects_removed_kwargs_before_model(
    shear_data, removed_key, value
):
    _SpyModel.reset()

    with patch(
        "rheojax.pipeline.workflows.ModelRegistry.create",
        side_effect=lambda _name: _SpyModel(),
    ):
        with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
            ModelComparisonPipeline(["spy"]).run(shear_data, **{removed_key: value})

    assert _SpyModel.fit_calls == []


def test_model_comparison_preserves_supported_kwargs(shear_data):
    _SpyModel.reset()

    with patch(
        "rheojax.pipeline.workflows.ModelRegistry.create",
        side_effect=lambda _name: _SpyModel(),
    ):
        ModelComparisonPipeline(["spy"]).run(shear_data, method="scipy", max_iter=23)

    assert _SpyModel.fit_calls == [
        {"test_mode": "relaxation", "method": "scipy", "max_iter": 23}
    ]


@pytest.mark.parametrize(("removed_key", "value"), REMOVED_KWARGS)
def test_batch_fit_replay_rejects_removed_kwargs_before_model(
    shear_data, tmp_path, removed_key, value
):
    _SpyModel.reset()
    template_model = _SpyModel()
    template_model._last_fit_kwargs = {removed_key: value}
    template = Pipeline()
    template.steps = [("fit", template_model)]

    with pytest.raises(TypeError, match=rf"{removed_key}.*shear-only"):
        BatchPipeline(template)._process_file(
            tmp_path / "input.csv", preloaded_data=shear_data
        )

    assert _SpyModel.fit_calls == []


def test_batch_replay_preserves_fit_bayesian_and_export_fields(
    shear_data, tmp_path, monkeypatch
):
    _SpyModel.reset()
    template_model = _SpyModel()
    template_model._last_fit_kwargs = {
        "test_mode": "creep",
        "method": "scipy",
        "max_iter": 29,
    }
    template_model._last_bayesian_kwargs = {
        "num_warmup": 5,
        "num_samples": 7,
        "num_chains": 2,
        "seed": 3,
        "target_accept_prob": 0.92,
    }
    output_root = tmp_path / "exports"
    template = Pipeline()
    template.steps = [
        ("fit", template_model),
        ("fit_bayesian", template_model),
        ("export", {"output_path": str(output_root), "format": "json"}),
    ]
    export_calls = []

    def _record_export(self, output_path, format="auto", **kwargs):
        export_calls.append((Path(output_path), format, kwargs))
        return self

    monkeypatch.setattr(Pipeline, "export", _record_export)

    _, metrics = BatchPipeline(template)._process_file(
        tmp_path / "input.csv", preloaded_data=shear_data
    )

    assert _SpyModel.fit_calls == [
        {"test_mode": "creep", "method": "scipy", "max_iter": 29}
    ]
    assert _SpyModel.bayesian_calls == [
        {
            "test_mode": "creep",
            "num_warmup": 5,
            "num_samples": 7,
            "num_chains": 2,
            "seed": 3,
            "target_accept_prob": 0.92,
        }
    ]
    assert export_calls == [(output_root / "input", "json", {})]
    assert metrics["export_path"] == str(output_root / "input")
