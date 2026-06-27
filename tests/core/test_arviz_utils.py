"""Tests for ArviZ compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheojax.core import arviz_utils


def test_inference_data_from_dict_uses_grouped_arviz_1_api() -> None:
    posterior = {"tau": np.arange(6).reshape(2, 3)}
    sample_stats = {"diverging": np.array([[False, True, False], [False] * 3])}

    idata = arviz_utils.inference_data_from_dict(
        {"posterior": posterior, "sample_stats": sample_stats}
    )

    assert list(idata.posterior.data_vars) == ["tau"]
    assert idata.posterior["tau"].dims == ("chain", "draw")
    assert list(idata.sample_stats.data_vars) == ["diverging"]
    np.testing.assert_array_equal(idata.posterior["tau"].values, posterior["tau"])
    np.testing.assert_array_equal(
        idata.sample_stats["diverging"].values, sample_stats["diverging"]
    )


def test_inference_data_from_dict_uses_keyword_arviz_0_api(monkeypatch) -> None:
    calls = []

    def legacy_from_dict(**kwargs):
        calls.append(kwargs)
        return "legacy-inference-data"

    legacy_arviz = SimpleNamespace(__version__="0.23.4", from_dict=legacy_from_dict)
    monkeypatch.setitem(__import__("sys").modules, "arviz", legacy_arviz)

    posterior = {"tau": np.array([[1.0, 2.0]])}
    sample_stats = {"diverging": np.array([[False, True]])}

    result = arviz_utils.inference_data_from_dict(
        {"posterior": posterior, "sample_stats": sample_stats}
    )

    assert result == "legacy-inference-data"
    assert len(calls) == 1
    assert calls[0].keys() == {"posterior", "sample_stats"}
    np.testing.assert_array_equal(calls[0]["posterior"]["tau"], posterior["tau"])
    np.testing.assert_array_equal(
        calls[0]["sample_stats"]["diverging"], sample_stats["diverging"]
    )
