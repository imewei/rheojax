"""Regression: BayesianOptionsDialog validates priors JSON as-you-type, not just at Accept."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.dialogs.bayesian_options import BayesianOptionsDialog


def test_invalid_priors_json_shows_and_clears_inline_error(qtbot):
    dialog = BayesianOptionsDialog()
    qtbot.addWidget(dialog)

    assert dialog._priors_error_label.isHidden()

    dialog.priors_edit.setPlainText("{not valid json")
    assert not dialog._priors_error_label.isHidden()
    assert "Invalid JSON" in dialog._priors_error_label.text()

    dialog.priors_edit.setPlainText('{"G": {"dist": "normal", "loc": 1.0}}')
    assert dialog._priors_error_label.isHidden()


def test_accept_rejects_with_same_validator_used_for_inline_error(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    dialog = BayesianOptionsDialog()
    qtbot.addWidget(dialog)
    dialog.priors_edit.setPlainText("{not valid json")

    warned = []
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *a, **k: warned.append(a) or QMessageBox.StandardButton.Ok,
    )

    dialog._on_accepted()

    assert warned  # invalid JSON blocked Accept
