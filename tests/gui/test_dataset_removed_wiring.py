"""Regression: dataset_removed was emitted by the store but never connected.

DatasetTree.remove_dataset() existed to consume it, but nothing wired the
signal to the slot, so deleting a dataset left a stale, still-clickable row
in the tree (clicking it fired dataset_selected(dead_id) for an id no longer
in state.datasets).
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_dataset_removed_signal_calls_remove_dataset(qtbot, qapp) -> None:
    from rheojax.gui.app.main_window import RheoJAXMainWindow
    from rheojax.gui.widgets.dataset_tree import DatasetTree

    with patch.object(DatasetTree, "remove_dataset") as mock_remove:
        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        window.store._signals.dataset_removed.emit("dataset-123")

    # patch.object replaces the class attribute with a plain Mock, which
    # (unlike a real function) isn't a descriptor -- so `self` is not bound
    # automatically when accessed off the instance, and the emitted signal's
    # single str argument is all `mock_remove` sees.
    mock_remove.assert_called_with("dataset-123")
