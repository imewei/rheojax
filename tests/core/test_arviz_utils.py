"""Tests for ArviZ compatibility helpers."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

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




def test_plot_pair_kwargs_map_arviz_1_divergences_and_marginal() -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_pair",
        kind="scatter",
        divergences=True,
        marginals=False,
        visuals={"scatter": {"alpha": 0.5}},
    )

    assert kwargs == {
        "marginal": False,
        "visuals": {"scatter": {"alpha": 0.5}, "divergence": True},
    }


def test_plot_pair_kwargs_preserve_legacy_default_without_marginals() -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_pair",
        divergences=False,
    )

    assert kwargs == {"marginal": False, "visuals": {"divergence": False}}


def test_plot_pair_kwargs_reject_unavailable_arviz_1_plot_kind() -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    with pytest.raises(ValueError, match="only supports.*scatter"):
        arviz_utils.arviz_plot_kwargs(current_arviz, "plot_pair", kind="kde")


def test_plot_posterior_kwargs_map_hdi_prob_and_set_defaults() -> None:
    current_arviz = SimpleNamespace(__version__="1.2.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_posterior",
        var_names=["a", "b"],
        hdi_prob=0.9,
    )

    assert kwargs == {
        "var_names": ["a", "b"],
        "ci_prob": 0.9,
        "ci_kind": "hdi",
        "point_estimate": "mean",
        "kind": "kde",
    }


def test_plot_posterior_kwargs_without_hdi_prob_still_sets_defaults() -> None:
    current_arviz = SimpleNamespace(__version__="1.2.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_posterior",
        var_names=["a"],
    )

    assert kwargs == {
        "var_names": ["a"],
        "ci_kind": "hdi",
        "point_estimate": "mean",
        "kind": "kde",
    }


@pytest.mark.parametrize(
    ("combined", "sample_dims"),
    [(False, ("draw",)), (True, ("chain", "draw"))],
)
def test_plot_trace_kwargs_map_arviz_1_chain_combination(
    combined: bool, sample_dims: tuple[str, ...]
) -> None:
    current_arviz = SimpleNamespace(__version__="1.2.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_trace",
        var_names=["a"],
        combined=combined,
    )

    assert kwargs == {"var_names": ["a"], "sample_dims": sample_dims}


def test_plot_trace_kwargs_preserve_default_without_combined() -> None:
    current_arviz = SimpleNamespace(__version__="1.2.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_trace",
        var_names=["a"],
    )

    assert kwargs == {"var_names": ["a"]}


@pytest.mark.parametrize(
    ("hdi_prob", "ci_probs"),
    [(0.95, (0.5, 0.95)), (0.4, (0.2, 0.4))],
)
def test_plot_forest_kwargs_map_arviz_1_hdi_probability(
    hdi_prob: float, ci_probs: tuple[float, float]
) -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_forest",
        combined=True,
        hdi_prob=hdi_prob,
    )

    assert kwargs == {
        "combined": True,
        "ci_kind": "hdi",
        "ci_probs": ci_probs,
    }


@pytest.mark.parametrize(
    ("combined", "sample_dims"),
    [(False, ("draw",)), (True, ("chain", "draw"))],
)
def test_plot_autocorr_kwargs_map_arviz_1_chain_combination(
    combined: bool, sample_dims: tuple[str, ...]
) -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_autocorr",
        combined=combined,
        max_lag=25,
    )

    assert kwargs == {"sample_dims": sample_dims, "max_lag": 25}


def test_plot_autocorr_kwargs_preserve_legacy_default_without_combined() -> None:
    current_arviz = SimpleNamespace(__version__="1.1.0")

    kwargs = arviz_utils.arviz_plot_kwargs(
        current_arviz,
        "plot_autocorr",
        max_lag=25,
    )

    assert kwargs == {"sample_dims": ("draw",), "max_lag": 25}


def test_arviz_figure_uses_legacy_axes_figure() -> None:
    figure = object()

    assert arviz_utils.arviz_figure(SimpleNamespace(figure=figure)) is figure


def test_arviz_figure_uses_arviz_1_plot_collection_viz() -> None:
    figure = object()
    plot_collection = SimpleNamespace(
        viz={"figure": SimpleNamespace(item=lambda: figure)}
    )

    assert arviz_utils.arviz_figure(plot_collection) is figure


def test_import_arviz_missing_required_attr_raises_import_error(monkeypatch) -> None:
    stub_arviz = SimpleNamespace(__version__="1.2.0")
    monkeypatch.setitem(__import__("sys").modules, "arviz", stub_arviz)

    with pytest.raises(ImportError) as excinfo:
        arviz_utils.import_arviz(required=("plot_pair",))

    message = str(excinfo.value)
    assert "plot_pair" in message
    assert "arviz[plots]" not in message
    assert "arviz[matplotlib]" in message


def test_arviz_figure_rejects_unknown_result_instead_of_guessing() -> None:
    active_figure = plt.figure()

    try:
        with pytest.raises(RuntimeError, match="unrecognized result type"):
            arviz_utils.arviz_figure(SimpleNamespace())
    finally:
        plt.close(active_figure)
