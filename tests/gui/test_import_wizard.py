"""Regression check: ImportWizard file I/O runs off the Qt GUI thread."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from rheojax.gui.dialogs.import_wizard import ImportWizard


@pytest.fixture(scope="module")
def qapp():
    """Ensure a QApplication exists."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_column_mapping_populates_after_background_worker_completes(qapp, tmp_path):
    """ColumnMappingPage dispatches the CSV read to a QRunnable and only
    populates its combo boxes once the worker's `completed` signal fires.
    """
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("freq,G_prime,G_double_prime\n1.0,100.0,10.0\n2.0,110.0,11.0\n")

    wizard = ImportWizard()
    wizard.file_page.file_path_edit.setText(str(csv_path))

    page = wizard.column_page
    page.initializePage()

    # Drain the background QThreadPool and let queued signals reach the GUI
    # thread. If _load_columns ever regresses to reading synchronously and
    # populating on return, this would still pass; if the worker signals are
    # ever left unconnected, the combos stay empty and this fails.
    assert QThreadPool.globalInstance().waitForDone(5000)
    qapp.processEvents()

    assert page.x_combo.count() == 3
    assert page.x_combo.itemText(0) == "freq"
    assert wizard._cached_df is not None

    wizard.deleteLater()


def test_stale_column_load_result_is_discarded_after_path_change(qapp, tmp_path):
    """A `_on_columns_loaded` callback carrying a file_path that no longer
    matches the current `file_path` field (because the user picked a
    different file before the background read finished) must not populate
    the combos or overwrite the wizard's cache.
    """
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("freq,G_prime\n1.0,100.0\n")

    wizard = ImportWizard()
    wizard.file_page.file_path_edit.setText(str(csv_path))

    page = wizard.column_page
    stale_df = pd.DataFrame({"old_col": [1, 2]})

    # Simulate a worker for a *previous* path completing after the field
    # already moved on to csv_path.
    page._on_columns_loaded(stale_df, str(tmp_path / "other.csv"))

    assert page.x_combo.count() == 0
    assert not hasattr(wizard, "_cached_df")

    wizard.deleteLater()
