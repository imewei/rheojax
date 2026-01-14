from datetime import datetime

import numpy as np

from rheojax.gui.pages.bayesian_page import BayesianPage
from rheojax.gui.state.store import AppState, BayesianResult, DatasetState, StateStore
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas
from rheojax.gui.widgets.plot_canvas import PlotCanvas


def setup_function() -> None:
    StateStore.reset()


def test_bayesian_page_shows_fit_plot_only_and_no_arviz(qtbot) -> None:
    store = StateStore()
    x = np.logspace(-3, 1, 12)
    y = 1.0 + 0.8 * np.exp(-x / 0.03) + 0.3 * np.exp(-x / 0.4) + 0.1 * np.exp(-x / 3.0)
    ds = DatasetState(
        id="d1",
        name="synthetic",
        file_path=None,
        test_mode="relaxation",
        x_data=x,
        y_data=y,
        metadata={"test_mode": "relaxation"},
    )
    store._state = AppState(
        datasets={"d1": ds},
        active_dataset_id="d1",
        active_model_name="maxwell",
    )

    page = BayesianPage()
    qtbot.addWidget(page)

    # No ArviZ diagnostics widgets should exist in Bayesian tab.
    assert page.findChildren(ArvizCanvas) == []

    # Fit plot canvas should exist.
    assert isinstance(page._fit_plot_canvas, PlotCanvas)

    # Simulate a successful Bayesian completion with minimal posterior samples.
    posterior = {
        # Maxwell typically uses parameters like G0 and tau.
        "G0": np.array([1e3, 1.1e3, 0.9e3, 1.05e3]),
        "tau": np.array([0.5, 0.6, 0.4, 0.55]),
    }
    result = BayesianResult(
        model_name="maxwell",
        dataset_id="d1",
        posterior_samples=posterior,
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.1,
        timestamp=datetime.now(),
        num_warmup=5,
        num_samples=10,
    )
    store.dispatch(
        "STORE_BAYESIAN_RESULT",
        {"model_name": "maxwell", "dataset_id": "d1", "result": result},
    )

    page.show()
    qtbot.wait(10)

    # Updating the plot should not crash and should hide the placeholder on success.
    page._update_fit_plot_from_posterior(result)
    assert not page._fit_plot_placeholder.isVisible()


def test_bayesian_fit_plot_includes_credible_band(qtbot, monkeypatch) -> None:
    store = StateStore()
    x = np.logspace(-3, 1, 12)
    y = 1.0 + 0.8 * np.exp(-x / 0.03) + 0.3 * np.exp(-x / 0.4) + 0.1 * np.exp(-x / 3.0)
    ds = DatasetState(
        id="d1",
        name="synthetic",
        file_path=None,
        test_mode="relaxation",
        x_data=x,
        y_data=y,
        metadata={"test_mode": "relaxation"},
    )
    store._state = AppState(
        datasets={"d1": ds},
        active_dataset_id="d1",
        active_model_name="maxwell",
    )

    page = BayesianPage()
    qtbot.addWidget(page)

    # Keep plotting fast and deterministic: patch model predictions.
    def _fake_predict(
        model_name: str,
        parameters: dict[str, float],
        x_values: np.ndarray,
        test_mode=None,
        model_kwargs=None,
    ):
        g0 = float(parameters.get("G0", 1.0))
        tau = float(parameters.get("tau", 1.0))
        return g0 * np.exp(-np.asarray(x_values) / max(tau, 1e-9))

    monkeypatch.setattr(page._model_service, "predict", _fake_predict)

    posterior = {
        "G0": np.linspace(900.0, 1100.0, 20),
        "tau": np.linspace(0.4, 0.6, 20),
    }
    result = BayesianResult(
        model_name="maxwell",
        dataset_id="d1",
        posterior_samples=posterior,
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.1,
        timestamp=datetime.now(),
        num_warmup=5,
        num_samples=10,
    )

    page._update_fit_plot_from_posterior(result)

    ax = page._fit_plot_canvas.figure.gca()
    # fill_between creates a PolyCollection in axes.collections
    assert len(ax.collections) >= 1


def test_bayesian_oscillation_complex_includes_component_bands(
    qtbot, monkeypatch
) -> None:
    store = StateStore()
    x = np.logspace(0, 2, 12)
    y = (1000 / (1 + x)) + 1j * (300 / (1 + x))
    ds = DatasetState(
        id="d1",
        name="synthetic_osc",
        file_path=None,
        test_mode="oscillation",
        x_data=x,
        y_data=y,
        metadata={"test_mode": "oscillation"},
    )
    store._state = AppState(
        datasets={"d1": ds},
        active_dataset_id="d1",
        active_model_name="maxwell",
    )

    page = BayesianPage()
    qtbot.addWidget(page)

    def _fake_predict(
        model_name: str,
        parameters: dict[str, float],
        x_values: np.ndarray,
        test_mode=None,
        model_kwargs=None,
    ):
        g0 = float(parameters.get("G0", 1.0))
        tau = float(parameters.get("tau", 1.0))
        xv = np.asarray(x_values)
        g_prime = (g0 / (1.0 + xv / max(tau, 1e-9))).astype(float)
        g_double = (0.3 * g0 / (1.0 + xv / max(tau, 1e-9))).astype(float)
        return g_prime + 1j * g_double

    monkeypatch.setattr(page._model_service, "predict", _fake_predict)

    posterior = {
        "G0": np.linspace(900.0, 1100.0, 20),
        "tau": np.linspace(10.0, 20.0, 20),
    }
    result = BayesianResult(
        model_name="maxwell",
        dataset_id="d1",
        posterior_samples=posterior,
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.1,
        timestamp=datetime.now(),
        num_warmup=5,
        num_samples=10,
    )

    page._update_fit_plot_from_posterior(result)

    ax = page._fit_plot_canvas.figure.gca()
    # Expect two bands (G' and G")
    assert len(ax.collections) >= 2

    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert any("G'" in text for text in labels)
    assert any('G"' in text or 'G"' in text for text in labels)


def test_bayesian_oscillation_y_y2_combined_plots_correctly(qtbot, monkeypatch) -> None:
    store = StateStore()
    x = np.logspace(0, 2, 12)
    g_prime = 1000.0 / (1.0 + x)
    g_double = 300.0 / (1.0 + x)
    ds = DatasetState(
        id="d1",
        name="synthetic_osc_split",
        file_path=None,
        test_mode="oscillation",
        x_data=x,
        y_data=g_prime,
        y2_data=g_double,
        metadata={"test_mode": "oscillation"},
    )
    store._state = AppState(
        datasets={"d1": ds},
        active_dataset_id="d1",
        active_model_name="maxwell",
    )

    page = BayesianPage()
    qtbot.addWidget(page)

    def _fake_predict(
        model_name: str,
        parameters: dict[str, float],
        x_values: np.ndarray,
        test_mode=None,
        model_kwargs=None,
    ):
        xv = np.asarray(x_values)
        # Return a complex prediction with positive components.
        return (900.0 / (1.0 + xv)) + 1j * (250.0 / (1.0 + xv))

    monkeypatch.setattr(page._model_service, "predict", _fake_predict)

    posterior = {
        "G0": np.linspace(900.0, 1100.0, 20),
        "tau": np.linspace(10.0, 20.0, 20),
    }
    result = BayesianResult(
        model_name="maxwell",
        dataset_id="d1",
        posterior_samples=posterior,
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.1,
        timestamp=datetime.now(),
        num_warmup=5,
        num_samples=10,
    )

    page._update_fit_plot_from_posterior(result)

    ax = page._fit_plot_canvas.figure.gca()
    lines = {line.get_label(): line for line in ax.get_lines()}
    assert "G' (data)" in lines
    assert 'G" (data)' in lines

    np.testing.assert_allclose(lines["G' (data)"].get_ydata(), g_prime)
    np.testing.assert_allclose(lines['G" (data)'].get_ydata(), g_double)
