import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.resources.styles.tokens import ThemeManager
from rheojax.gui.workspace.status_bar import StatusBar
from rheojax.gui.workspace.window import WorkspaceWindow


def _win(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_status_bar_is_installed(qtbot):
    win = _win(qtbot)
    assert isinstance(win.statusBar(), StatusBar)


def test_refresh_status_bar_calls_update_jax_status(qtbot, monkeypatch):
    win = _win(qtbot)
    calls = []
    monkeypatch.setattr(
        win.statusBar(),
        "update_jax_status",
        lambda **kw: calls.append(kw),
    )
    monkeypatch.setattr(
        "rheojax.gui.utils.jax_utils.get_jax_info",
        lambda: {
            "default_device": "cuda:0",
            "memory_used_mb": 100.0,
            "memory_total_mb": 8192.0,
            "float64_enabled": True,
        },
    )
    win._refresh_status_bar()
    assert calls == [
        {
            "device": "cuda:0",
            "memory_used": 100.0,
            "memory_total": 8192.0,
            "float64_enabled": True,
        }
    ]


def test_refresh_status_bar_survives_get_jax_info_failure(qtbot, monkeypatch):
    win = _win(qtbot)

    def _raise():
        raise RuntimeError("no JAX backend")

    monkeypatch.setattr("rheojax.gui.utils.jax_utils.get_jax_info", _raise)
    win._refresh_status_bar()  # must not raise


def test_on_new_shows_status_message(qtbot):
    win = _win(qtbot)
    win._on_new()
    assert win.statusBar().message_label.text() == "New project created"


def test_on_save_as_shows_status_message(qtbot, monkeypatch, tmp_path):
    win = _win(qtbot)
    path = str(tmp_path / "proj.rheojax")
    from PySide6.QtWidgets import QFileDialog

    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *a, **k: (path, "")
    )
    win._on_save_as()
    assert win.statusBar().message_label.text() == "Project saved"


def test_on_save_shows_status_message(qtbot, tmp_path):
    win = _win(qtbot)
    win._state.project.path = str(tmp_path / "proj.rheojax")
    win._on_save()
    assert win.statusBar().message_label.text() == "Project saved"


def test_on_open_shows_status_message(qtbot, monkeypatch, tmp_path):
    win = _win(qtbot)
    path = str(tmp_path / "proj.rheojax")
    from PySide6.QtWidgets import QFileDialog

    from rheojax.gui.foundation.project_codec import save_project_v2

    save_project_v2(win._state, path)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *a, **k: (path, "")
    )
    win._on_open()
    assert win.statusBar().message_label.text() == "Project opened"


def test_apply_theme_light_sets_state_and_theme_manager(qtbot):
    win = _win(qtbot)
    win._apply_theme("light")
    assert win._state.ui.theme == "light"
    assert ThemeManager.is_dark() is False
    assert win._theme_light_action.isChecked() is True
    assert win._theme_dark_action.isChecked() is False
    assert win._theme_system_action.isChecked() is False


def test_apply_theme_dark_sets_state_and_theme_manager(qtbot):
    win = _win(qtbot)
    win._apply_theme("dark")
    assert win._state.ui.theme == "dark"
    assert ThemeManager.is_dark() is True
    assert win._theme_dark_action.isChecked() is True


def test_apply_theme_system_stores_system_not_resolved_value(qtbot):
    win = _win(qtbot)
    win._apply_theme("system")
    # The stored preference must stay "system" even though ThemeManager
    # ends up holding whatever concrete light/dark value was resolved --
    # otherwise reloading a saved "system" preference would incorrectly
    # freeze at whatever the OS scheme happened to be at save time.
    assert win._state.ui.theme == "system"
    assert win._theme_system_action.isChecked() is True
    assert ThemeManager.is_dark() in (True, False)


def test_apply_theme_shows_status_message(qtbot):
    win = _win(qtbot)
    win._apply_theme("dark")
    assert win.statusBar().message_label.text() == "Theme: Dark"


def test_theme_menu_actions_call_apply_theme(qtbot):
    win = _win(qtbot)
    win._theme_dark_action.trigger()
    assert win._state.ui.theme == "dark"


def test_build_workspace_applies_saved_theme_on_construction(qtbot):
    from rheojax.gui.foundation.state import UiState

    win = WorkspaceWindow(AppState(ui=UiState(theme="dark")))
    qtbot.addWidget(win)
    assert ThemeManager.is_dark() is True
    assert win._theme_dark_action.isChecked() is True


def test_build_workspace_applies_saved_theme_on_rebuild(qtbot):
    from rheojax.gui.foundation.state import UiState

    win = _win(qtbot)
    win._apply_theme("light")
    win._rebuild(AppState(ui=UiState(theme="dark")))
    assert ThemeManager.is_dark() is True


def test_apply_preferences_applies_theme(qtbot):
    win = _win(qtbot)
    win.apply_preferences({"theme": "dark"})
    assert win._state.ui.theme == "dark"


def test_apply_preferences_without_theme_key_is_noop(qtbot):
    win = _win(qtbot)
    win._apply_theme("light")
    win.apply_preferences({"worker_isolation_mode": "subprocess"})
    assert win._state.ui.theme == "light"


def test_on_preferences_applies_dialog_result(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rheojax.gui.dialogs.preferences import PreferencesDialog

    win = _win(qtbot)
    monkeypatch.setattr(
        PreferencesDialog, "exec", lambda self: QDialog.DialogCode.Accepted
    )
    # PreferencesDialog.get_preferences() actually returns self.theme_combo.
    # currentText() -- "Light"/"Dark"/"System", the combo's exact item text
    # (preferences.py: self.theme_combo.addItems(["Light", "Dark", "System"])).
    # A lowercase "dark" mock here would not exercise the real integration
    # shape and would have hidden the case-mismatch bug this plan's
    # adversarial review round caught (_apply_theme now normalizes case
    # itself, but the test must still reflect what the dialog really sends).
    monkeypatch.setattr(
        PreferencesDialog, "get_preferences", lambda self: {"theme": "Dark"}
    )
    win._on_preferences()  # must not raise ValueError from load_stylesheet
    assert win._state.ui.theme == "dark"


def test_on_preferences_cancelled_does_not_apply(qtbot, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from rheojax.gui.dialogs.preferences import PreferencesDialog

    win = _win(qtbot)
    win._apply_theme("light")
    monkeypatch.setattr(
        PreferencesDialog, "exec", lambda self: QDialog.DialogCode.Rejected
    )
    win._on_preferences()
    assert win._state.ui.theme == "light"


def test_preferences_dialog_restores_current_theme_selection(qtbot):
    # Regression guard for the same case-mismatch bug class from the other
    # direction: self._state.ui.theme is always lowercase ("dark"), but the
    # dialog's combo box items are title-case ("Dark"). QComboBox.findText's
    # MatchFixedString flag is case-insensitive by default (verified
    # directly against this PySide6 version, not assumed from general Qt
    # knowledge), so passing the lowercase value through unmodified already
    # restores the correct selection -- this test pins that behavior so a
    # future PySide6/qtpy version change would be caught here.
    from rheojax.gui.dialogs.preferences import PreferencesDialog

    win = _win(qtbot)
    win._apply_theme("dark")
    dialog = PreferencesDialog(
        current_preferences={"theme": win._state.ui.theme}, parent=win
    )
    qtbot.addWidget(dialog)
    assert dialog.theme_combo.currentText() == "Dark"


def test_mode_switcher_button_stays_checked_on_reclick(qtbot):
    # Regression: the three mode-switcher pills had no exclusive QButtonGroup,
    # so Qt would uncheck the already-active pill on a re-click before
    # set_mode()'s no-op early return (mode == self._mode) ever reached
    # _sync_mode_buttons() -- the toolbar could show no mode selected while
    # the app stayed in that mode. QButtonGroup(exclusive=True) makes Qt
    # itself refuse to uncheck the sole checked button.
    from PySide6.QtCore import Qt

    win = _win(qtbot)
    assert win.mode() == "fit"
    assert win._fit_btn.isChecked() is True

    qtbot.mouseClick(win._fit_btn, Qt.MouseButton.LeftButton)

    assert win.mode() == "fit"
    assert win._fit_btn.isChecked() is True


def test_command_palette_action_list_has_expected_labels(qtbot):
    win = _win(qtbot)
    actions = win._command_palette_actions()
    assert set(actions.keys()) == {
        "New Project",
        "Open Project...",
        "Save Project",
        "Save Project As...",
        "Switch to Fit Mode",
        "Switch to Transform Mode",
        "Switch to Pipeline Mode",
        "Toggle Log Panel",
        "Preferences...",
        "Cycle Theme",
    }


def test_command_palette_switch_mode_action_calls_set_mode(qtbot):
    win = _win(qtbot)
    actions = win._command_palette_actions()
    actions["Switch to Transform Mode"]()
    assert win._mode == "transform"


def test_command_palette_cycle_theme_action_advances_theme(qtbot):
    win = _win(qtbot)
    win._apply_theme("light")
    actions = win._command_palette_actions()
    actions["Cycle Theme"]()
    assert win._state.ui.theme == "dark"
    actions = win._command_palette_actions()
    actions["Cycle Theme"]()
    assert win._state.ui.theme == "system"
    actions = win._command_palette_actions()
    actions["Cycle Theme"]()
    assert win._state.ui.theme == "light"


def test_command_palette_toggle_log_panel_action_triggers_action(qtbot):
    win = _win(qtbot)
    assert win.view_log_dock_action.isChecked() is False
    actions = win._command_palette_actions()
    actions["Toggle Log Panel"]()
    assert win.view_log_dock_action.isChecked() is True


def test_command_palette_new_project_action_calls_on_new(qtbot, monkeypatch):
    win = _win(qtbot)
    calls = []
    monkeypatch.setattr(win, "_on_new", lambda: calls.append(1))
    actions = win._command_palette_actions()
    actions["New Project"]()
    assert calls == [1]


def test_command_palette_open_project_action_calls_on_open(qtbot, monkeypatch):
    win = _win(qtbot)
    calls = []
    monkeypatch.setattr(win, "_on_open", lambda: calls.append(1))
    actions = win._command_palette_actions()
    actions["Open Project..."]()
    assert calls == [1]


def test_command_palette_save_as_action_calls_on_save_as(qtbot, monkeypatch):
    win = _win(qtbot)
    calls = []
    monkeypatch.setattr(win, "_on_save_as", lambda: calls.append(1))
    actions = win._command_palette_actions()
    actions["Save Project As..."]()
    assert calls == [1]


def test_open_command_palette_invokes_selected_action(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    win = _win(qtbot)
    monkeypatch.setattr(
        QInputDialog,
        "getItem",
        lambda *a, **k: ("Save Project", True),
    )
    calls = []
    monkeypatch.setattr(win, "_on_save", lambda: calls.append(1))
    win._open_command_palette()
    assert calls == [1]


def test_open_command_palette_cancelled_invokes_nothing(qtbot, monkeypatch):
    from PySide6.QtWidgets import QInputDialog

    win = _win(qtbot)
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("", False))
    calls = []
    monkeypatch.setattr(win, "_on_save", lambda: calls.append(1))
    win._open_command_palette()
    assert calls == []


def test_splitter_has_no_inspector_panel(qtbot):
    """Regression guard: InspectorPanel was permanently empty chrome (its
    set_tab_widget() had no caller) and was removed from the splitter. This
    catches an accidental re-add without the design work that would make it
    non-empty."""
    win = _win(qtbot)
    assert win._splitter.count() == 2  # rail + canvas only
    assert not hasattr(win, "_inspector")
