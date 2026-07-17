"""Regression check: ColumnMapperDialog auto-detect never maps X and Y to
the same column."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from rheojax.gui.dialogs.column_mapper import ColumnMapperDialog


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_auto_detect_does_not_map_x_and_y_to_same_column(qapp):
    """A single ambiguous column (matches both an X pattern like "shear"
    and a Y pattern like "stress") must be claimed by only one field --
    previously both find_match() calls ran independently and could select
    the identical column for X and Y. Populates the dialog's combos
    directly (bypassing file I/O/delimiter sniffing) to isolate
    _auto_detect() itself as the unit under test.
    """
    dialog = ColumnMapperDialog(file_path=None)
    dialog.columns = ["shear stress"]
    dialog.x_combo.set_items_safely(dialog.columns)
    dialog.y_combo.set_items_safely(dialog.columns)
    dialog.y2_combo.set_items_safely(dialog.columns)
    dialog.temp_combo.set_items_safely(dialog.columns)

    dialog._auto_detect()

    assert dialog.x_combo.currentText() == "shear stress"
    assert dialog.y_combo.currentText() != "shear stress"
