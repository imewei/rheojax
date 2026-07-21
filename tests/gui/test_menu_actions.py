"""RheoJAX GUI Menu Actions Tests.

Markers:
    gui: All GUI-related tests
    smoke: Critical smoke tests for CI/CD
"""

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtWidgets import QApplication  # noqa: F401

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


class TestDialogIntegration:
    """Test that dialogs are properly integrated with menu actions."""

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_preferences_dialog_accessible(self, qtbot, qapp) -> None:
        """Verify PreferencesDialog can be imported and instantiated."""
        from rheojax.gui.dialogs.preferences import PreferencesDialog

        dialog = PreferencesDialog()
        qtbot.addWidget(dialog)
        assert dialog is not None
        assert dialog.windowTitle() == "Preferences"

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_about_dialog_accessible(self, qtbot, qapp) -> None:
        """Verify AboutDialog can be imported and instantiated."""
        from rheojax.gui.dialogs.about import AboutDialog

        dialog = AboutDialog()
        qtbot.addWidget(dialog)
        assert dialog is not None
        assert "About" in dialog.windowTitle()
