from datetime import datetime

from PySide6.QtWidgets import QMessageBox

from rheojax.gui.pages.diagnostics_page import DiagnosticsPage
from rheojax.gui.state.store import AppState, BayesianResult, DatasetState, StateStore


def setup_function() -> None:
    StateStore.reset()


def test_diagnostics_resolves_bayesian_result_and_emits_completed(qtbot, monkeypatch) -> None:
    store = StateStore()
    ds = DatasetState(id="d1", name="ds", file_path=None, test_mode="oscillation")
    store._state = AppState(datasets={"d1": ds}, active_dataset_id="d1", active_model_name="generalized_maxwell")

    result = BayesianResult(
        model_name="generalized_maxwell",
        dataset_id="d1",
        posterior_samples={},
        summary=None,
        r_hat={},
        ess={},
        divergences=0,
        credible_intervals={},
        mcmc_time=0.1,
        timestamp=datetime.now(),
    )

    completed: list[tuple[str, str]] = []

    def _on_completed(model_name: str, dataset_id: str) -> None:
        completed.append((model_name, dataset_id))

    store.signals.bayesian_completed.connect(_on_completed)
    store.dispatch(
        "STORE_BAYESIAN_RESULT",
        {"model_name": "generalized_maxwell", "dataset_id": "d1", "result": result},
    )
    assert ("generalized_maxwell", "d1") in completed

    info_calls: list[tuple[tuple, dict]] = []

    def _fake_information(*args, **kwargs):
        info_calls.append((args, kwargs))
        return None

    monkeypatch.setattr(QMessageBox, "information", _fake_information)

    page = DiagnosticsPage()
    qtbot.addWidget(page)

    page.show_diagnostics("generalized_maxwell")

    assert info_calls == []
    assert page._current_model_id == "generalized_maxwell"

