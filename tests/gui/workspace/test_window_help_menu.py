from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.state import AppState
from rheojax.gui.workspace.window import WorkspaceWindow


def _win(qtbot):
    win = WorkspaceWindow(AppState())
    qtbot.addWidget(win)
    return win


def test_help_menu_is_registered_on_menu_bar(qtbot):
    win = _win(qtbot)
    titles = [a.text() for a in win.menuBar().actions()]
    assert "&Help" in titles


def test_open_docs_opens_browser(qtbot, monkeypatch):
    win = _win(qtbot)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    win._on_open_docs()

    assert opened == ["https://rheojax.readthedocs.io"]


def test_open_tutorials_opens_browser(qtbot, monkeypatch):
    win = _win(qtbot)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    win._on_open_tutorials()

    assert opened == ["https://github.com/imewei/rheojax/tree/main/examples"]


def test_show_shortcuts_displays_message_box(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    win = _win(qtbot)
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *a, **k: shown.append(a[1:]) or None,
    )

    win._on_show_shortcuts()

    assert shown and shown[0][0] == "Keyboard Shortcuts"


def test_about_shows_about_dialog(qtbot, monkeypatch):
    from rheojax.gui.dialogs.about import AboutDialog

    win = _win(qtbot)
    shown = []
    monkeypatch.setattr(AboutDialog, "exec", lambda self: shown.append(True))

    win._on_about()

    assert shown == [True]


def test_documentation_action_trigger_reaches_open_docs_handler(qtbot, monkeypatch):
    # Every other test in this file calls win._on_open_docs()/etc directly --
    # a missing or mis-wired menu.addAction(text, handler) call (e.g. the
    # wrong handler connected to the wrong label) would leave those tests
    # passing while a real click on "Documentation" did nothing or opened
    # the wrong page. This goes through the real QAction instead, matching
    # test_library_rail.py's test_context_menu_signal_wiring_reaches_handler_through_real_emission.
    win = _win(qtbot)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    help_action = next(a for a in win.menuBar().actions() if a.text() == "&Help")
    doc_action = next(
        a for a in help_action.menu().actions() if a.text() == "&Documentation"
    )

    doc_action.trigger()

    assert opened == ["https://rheojax.readthedocs.io"]
