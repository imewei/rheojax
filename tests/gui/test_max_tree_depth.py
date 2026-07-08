"""max_tree_depth passthrough — signature-level checks plus a wiring guard
for bayesian_page.py's Advanced Options dialog.

Regression test: config["target_accept_prob"]/config["max_tree_depth"] were
computed from the Advanced Options dialog but never reached
make_bayesian_worker() in _continue_bayesian_with_dataset -- silently
dropped before the sampler ever saw them.
"""

import inspect
from unittest.mock import MagicMock

import jax  # noqa: F401
import nlsq  # noqa: F401 — must precede any rheojax.core imports
import pytest
from PySide6.QtWidgets import QApplication

from rheojax.gui.jobs.bayesian_worker import BayesianWorker
from rheojax.gui.jobs.process_adapter import make_bayesian_worker
from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated
from rheojax.gui.pages.bayesian_page import BayesianPage


def test_max_tree_depth_param_present():
    assert "max_tree_depth" in inspect.signature(run_bayesian_isolated).parameters
    assert "max_tree_depth" in inspect.signature(make_bayesian_worker).parameters
    assert "max_tree_depth" in inspect.signature(BayesianWorker.__init__).parameters


def test_max_tree_depth_defaults_to_none():
    assert (
        inspect.signature(run_bayesian_isolated).parameters["max_tree_depth"].default
        is None
    )
    assert (
        inspect.signature(make_bayesian_worker).parameters["max_tree_depth"].default
        is None
    )
    assert (
        inspect.signature(BayesianWorker.__init__).parameters["max_tree_depth"].default
        is None
    )


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


class _FakeDataset:
    id = "ds1"

    def clone(self):
        return self


def test_advanced_options_reach_make_bayesian_worker(qapp, monkeypatch):
    page = BayesianPage()
    try:
        fake_worker = MagicMock()
        fake_worker.signals = MagicMock()
        captured = {}

        def _fake_make_bayesian_worker(**kwargs):
            captured.update(kwargs)
            return fake_worker

        monkeypatch.setattr(
            "rheojax.gui.jobs.process_adapter.make_bayesian_worker",
            _fake_make_bayesian_worker,
        )

        config = {
            "num_warmup": 100,
            "num_samples": 200,
            "num_chains": 1,
            "warm_start": False,
            "hdi_prob": 0.94,
            "target_accept_prob": 0.97,
            "max_tree_depth": 12,
        }
        page._continue_bayesian_with_dataset(_FakeDataset(), "maxwell", config)

        assert captured.get("target_accept") == 0.97
        assert captured.get("max_tree_depth") == 12
    finally:
        page.deleteLater()
        qapp.processEvents()
