"""RheoJAX GUI Menu Actions Tests.

Regression tests ensuring all menu bar actions are properly connected to handlers.
These tests prevent future regressions where menu items are added but not wired up.

Markers:
    gui: All GUI-related tests
    smoke: Critical smoke tests for CI/CD
"""

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtWidgets import QApplication
    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


def action_has_receivers(action) -> bool:
    """Check if a QAction has any receivers connected to its triggered signal.

    In PySide6, receivers() requires the Qt SIGNAL macro format with prefix "2".
    The triggered signal has signature triggered(bool) where bool is the checked state.
    """
    # Qt SIGNAL format: "2" prefix + signal signature
    # QAction.triggered has signature triggered(bool)
    return action.receivers("2triggered(bool)") > 0


class TestMenuBarActionConnections:
    """Test that all menu bar actions are connected to handlers.

    This test class ensures that every QAction in the menu bar has at least
    one signal receiver connected, preventing "dead" menu items.
    """

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_file_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all File menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        file_actions = [
            ("new_file_action", "New"),
            ("open_file_action", "Open"),
            ("save_file_action", "Save"),
            ("save_as_action", "Save As"),
            ("import_action", "Import"),
            ("export_action", "Export"),
            ("exit_action", "Exit"),
        ]

        for action_name, description in file_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"File > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_edit_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Edit menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        edit_actions = [
            ("undo_action", "Undo"),
            ("redo_action", "Redo"),
            ("cut_action", "Cut"),
            ("copy_action", "Copy"),
            ("paste_action", "Paste"),
            ("preferences_action", "Preferences"),
        ]

        for action_name, description in edit_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Edit > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_view_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all View menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        view_actions = [
            ("zoom_in_action", "Zoom In"),
            ("zoom_out_action", "Zoom Out"),
            ("reset_zoom_action", "Reset Zoom"),
            ("view_data_dock_action", "Data Panel"),
            ("view_param_dock_action", "Parameters Panel"),
            ("view_log_dock_action", "Log Panel"),
            ("theme_light_action", "Light Theme"),
            ("theme_dark_action", "Dark Theme"),
            ("theme_auto_action", "Auto Theme"),
        ]

        for action_name, description in view_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"View > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_data_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Data menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        data_actions = [
            ("new_dataset_action", "New Dataset"),
            ("delete_dataset_action", "Delete Dataset"),
            ("test_mode_oscillation", "Oscillation Mode"),
            ("test_mode_relaxation", "Relaxation Mode"),
            ("test_mode_creep", "Creep Mode"),
            ("test_mode_rotation", "Rotation Mode"),
            ("auto_detect_mode_action", "Auto-detect Mode"),
        ]

        for action_name, description in data_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Data > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_models_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Models menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        model_actions = [
            # Classical
            ("model_maxwell", "Maxwell"),
            ("model_zener", "Zener"),
            ("model_springpot", "SpringPot"),
            # Flow
            ("model_power_law", "Power Law"),
            ("model_carreau", "Carreau"),
            ("model_carreau_yasuda", "Carreau-Yasuda"),
            ("model_cross", "Cross"),
            ("model_herschel_bulkley", "Herschel-Bulkley"),
            ("model_bingham", "Bingham"),
            # Fractional Maxwell
            ("model_fmaxwell_gel", "Fractional Maxwell Gel"),
            ("model_fmaxwell_liquid", "Fractional Maxwell Liquid"),
            ("model_fmaxwell_model", "Fractional Maxwell Model"),
            ("model_fkelvin_voigt", "Fractional Kelvin-Voigt"),
            # Fractional Zener
            ("model_fzener_sl", "Fractional Zener SL"),
            ("model_fzener_ss", "Fractional Zener SS"),
            ("model_fzener_ll", "Fractional Zener LL"),
            ("model_fkv_zener", "Fractional KV-Zener"),
            # Advanced Fractional
            ("model_fburgers", "Fractional Burgers"),
            ("model_fpoynting", "Fractional Poynting-Thomson"),
            ("model_fjeffreys", "Fractional Jeffreys"),
            # Multi-mode
            ("model_gmaxwell", "Generalized Maxwell"),
            # SGR
            ("model_sgr_conventional", "SGR Conventional"),
            ("model_sgr_generic", "SGR GENERIC"),
        ]

        for action_name, description in model_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Models > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_transforms_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Transforms menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        transform_actions = [
            ("transform_fft", "FFT"),
            ("transform_mastercurve", "Mastercurve"),
            ("transform_srfs", "SRFS"),
            ("transform_mutation", "Mutation Number"),
            ("transform_owchirp", "OWChirp"),
            ("transform_derivatives", "Derivatives"),
        ]

        for action_name, description in transform_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Transforms > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_analysis_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Analysis menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        analysis_actions = [
            ("analysis_fit_nlsq", "Fit NLSQ"),
            ("analysis_fit_bayesian", "Fit Bayesian"),
            ("analysis_batch_fit", "Batch Fit"),
            ("analysis_compare", "Compare Models"),
            ("analysis_compatibility", "Check Compatibility"),
        ]

        for action_name, description in analysis_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Analysis > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_tools_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Tools menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        tools_actions = [
            ("tools_console", "Python Console"),
            ("tools_jax_profiler", "JAX Profiler"),
            ("tools_memory_monitor", "Memory Monitor"),
            ("tools_preferences", "Preferences"),
        ]

        for action_name, description in tools_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Tools > {description} ({action_name}) has no connected handler"
            )

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_help_menu_actions_connected(self, qtbot, qapp) -> None:
        """Verify all Help menu actions have handlers."""
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        help_actions = [
            ("help_docs", "Documentation"),
            ("help_tutorials", "Tutorials"),
            ("help_shortcuts", "Keyboard Shortcuts"),
            ("help_about", "About"),
        ]

        for action_name, description in help_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing action: {action_name}"
            assert action_has_receivers(action), (
                f"Help > {description} ({action_name}) has no connected handler"
            )


class TestToolbarActionConnections:
    """Test that toolbar/menu actions are connected to handlers."""

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_main_toolbar_actions_connected(self, qtbot, qapp) -> None:
        """Verify main menu bar actions have handlers.

        Note: The toolbar was refactored to use menu_bar for all actions.
        This test verifies the equivalent menu bar actions are connected.
        """
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        # Actions now live on menu_bar (toolbar was removed in refactor)
        menu_actions = [
            ("open_file_action", "Open"),
            ("save_file_action", "Save"),
            ("import_action", "Import"),
            ("zoom_in_action", "Zoom In"),
            ("zoom_out_action", "Zoom Out"),
            ("reset_zoom_action", "Reset Zoom"),
            ("preferences_action", "Settings/Preferences"),
        ]

        for action_name, description in menu_actions:
            action = getattr(window.menu_bar, action_name, None)
            assert action is not None, f"Missing menu bar action: {action_name}"
            assert action_has_receivers(action), (
                f"Menu Bar > {description} ({action_name}) has no connected handler"
            )


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

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_import_wizard_accessible(self, qtbot, qapp) -> None:
        """Verify ImportWizard can be imported and instantiated."""
        from rheojax.gui.dialogs.import_wizard import ImportWizard

        wizard = ImportWizard()
        qtbot.addWidget(wizard)
        assert wizard is not None
        assert "Import" in wizard.windowTitle()


class TestActionCountConsistency:
    """Test that no new unconnected actions are added."""

    @pytest.mark.smoke
    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
    def test_all_public_actions_connected(self, qtbot, qapp) -> None:
        """Verify all public QAction attributes on MenuBar have handlers.

        This test catches new actions that might be added without being wired up.
        """
        from PySide6.QtGui import QAction
        from rheojax.gui.app.main_window import RheoJAXMainWindow

        window = RheoJAXMainWindow()
        qtbot.addWidget(window)

        # Find all QAction attributes on menu_bar
        unconnected = []
        for attr_name in dir(window.menu_bar):
            if attr_name.startswith("_"):
                continue
            attr = getattr(window.menu_bar, attr_name, None)
            if isinstance(attr, QAction):
                if not action_has_receivers(attr):
                    unconnected.append(attr_name)

        assert len(unconnected) == 0, (
            f"Found {len(unconnected)} unconnected menu actions: {unconnected}"
        )
