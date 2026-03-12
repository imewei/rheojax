"""Tests for BatchPanel duplicate-basename full-path matching."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

pytestmark = pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.mark.smoke
class TestBatchPanelFileMatching:
    """Verify set_file_status matches by full path, not just basename."""

    def test_duplicate_basename_updates_correct_row(self, qapp):
        """Two files with same basename in different dirs must update independently."""
        from rheojax.gui.widgets.batch_panel import BatchPanel

        panel = BatchPanel()
        paths = ["/data/run1/results.csv", "/data/run2/results.csv"]
        panel._populate_table(paths)

        # Both rows should show PENDING initially
        assert panel.file_table.rowCount() == 2

        # Mark only the second file as DONE
        panel.set_file_status("/data/run2/results.csv", "DONE", 1.5)

        # Row 0 (run1) should still be PENDING, row 1 (run2) should be DONE
        status_0 = panel.file_table.item(0, 1).text()
        status_1 = panel.file_table.item(1, 1).text()
        assert status_0 == "PENDING"
        assert status_1 == "DONE"

    def test_full_path_stored_in_user_role(self, qapp):
        """Full path should be accessible via Qt.UserRole for exact matching."""
        from PySide6.QtCore import Qt

        from rheojax.gui.widgets.batch_panel import BatchPanel

        panel = BatchPanel()
        path = "/some/deep/dir/data.csv"
        panel._populate_table([path])

        item = panel.file_table.item(0, 0)
        assert item.text() == "data.csv"
        assert item.data(Qt.ItemDataRole.UserRole) == path
        assert item.toolTip() == path
