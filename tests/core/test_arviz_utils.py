"""Tests for ArviZ compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheojax.core import arviz_utils


def test_inference_data_from_dict_uses_grouped_arviz_1_api() -> None:
    posterior = {"tau": np.ones((2, 3))}

    idata = arviz_utils.inference_data_from_dict({"posterior": posterior})

    assert list(idata.posterior.data_vars) == ["tau"]
    assert idata.posterior["tau"].dims == ("chain", "draw")


def test_inference_data_from_dict_uses_keyword_arviz_0_api(monkeypatch) -> None:
    calls = []

    def legacy_from_dict(**kwargs):
        calls.append(kwargs)
        return "legacy-inference-data"

    legacy_arviz = SimpleNamespace(__version__="0.23.4", from_dict=legacy_from_dict)
    monkeypatch.setitem(__import__("sys").modules, "arviz", legacy_arviz)

    result = arviz_utils.inference_data_from_dict(
        {"posterior": {"tau": np.ones((1, 2))}, "sample_stats": None}
    )

    assert result == "legacy-inference-data"
    assert len(calls) == 1
    assert calls[0].keys() == {"posterior"}
    np.testing.assert_array_equal(calls[0]["posterior"]["tau"], np.ones((1, 2)))
