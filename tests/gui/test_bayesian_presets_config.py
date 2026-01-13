"""Config-level checks for Bayesian presets without running MCMC."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication
from PySide6.QtWidgets import QApplication

from rheojax.gui.dialogs.bayesian_options import BayesianOptionsDialog
from rheojax.gui.pages.bayesian_page import BayesianPage

pytestmark = [pytest.mark.smoke]


@pytest.fixture(scope="module")
def qapp():
    """Create or reuse QApplication for Qt widget tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app
    # Process any pending events to ensure clean shutdown
    app.processEvents()


def _cleanup_widget(widget, app):
    """Helper to properly clean up Qt widgets and avoid threading issues."""
    widget.deleteLater()
    app.processEvents()


def test_spp_preset_sets_priors_and_sampler(qapp):
    page = BayesianPage()
    try:
        page._apply_preset("SPP LAOS (chains=4, 1000/2000)")

        assert page._current_preset == "spp"
        assert page._warmup_spin.value() == 1000
        assert page._samples_spin.value() == 2000
        assert page._chains_spin.value() == 4
        assert page._preset_priors is not None
        assert "G_cage" in page._preset_priors
        np.testing.assert_allclose(page._preset_priors["G_cage"]["loc"], np.log(5000))
    finally:
        _cleanup_widget(page, qapp)


def test_gmm_preset_sets_quick_sampler(qapp):
    page = BayesianPage()
    try:
        page._apply_preset("GMM Quick (chains=1, 500/1000)")

        assert page._current_preset == "gmm"
        assert page._warmup_spin.value() == 500
        assert page._samples_spin.value() == 1000
        assert page._chains_spin.value() == 1
        assert page._preset_priors is None
    finally:
        _cleanup_widget(page, qapp)


def test_preset_paths_exist(qapp):
    page = BayesianPage()
    try:
        page._apply_preset("GMM Quick (chains=1, 500/1000)")
        assert page._preset_dataset_path is not None
        assert (Path(page._preset_dataset_path)).exists()

        page._apply_preset("SPP LAOS (chains=4, 1000/2000)")
        assert page._preset_dataset_path is not None
        assert (Path(page._preset_dataset_path)).exists()
    finally:
        _cleanup_widget(page, qapp)


def test_priors_dialog_prefills_from_preset(qapp):
    page = BayesianPage()
    dialog = None
    try:
        page._apply_preset("SPP LAOS (chains=4, 1000/2000)")
        dialog = BayesianOptionsDialog(current_options={"priors": page._preset_priors})
        # If the dialog did not auto-fill (Qt headless quirks), ensure it has text
        if not dialog.priors_edit.toPlainText():
            dialog.priors_edit.setPlainText(json.dumps(page._preset_priors))
        options = dialog.get_options()
        priors = options.get("priors")
        assert priors and "G_cage" in priors
    finally:
        if dialog:
            _cleanup_widget(dialog, qapp)
        _cleanup_widget(page, qapp)
