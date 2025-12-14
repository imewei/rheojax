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
    y = (
        1.0
        + 0.8 * np.exp(-x / 0.03)
        + 0.3 * np.exp(-x / 0.4)
        + 0.1 * np.exp(-x / 3.0)
    )
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

