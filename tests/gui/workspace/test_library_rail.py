from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.workspace.library_rail import LibraryRail


def _ref(id_: str) -> DatasetRef:
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type="oscillation",
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_double_click_emits_dataset_preview_requested(qtbot):
    library = DatasetLibrary()
    library.add(_ref("d1"))
    rail = LibraryRail(library)
    qtbot.addWidget(rail)
    item = rail._list.item(0)

    with qtbot.waitSignal(rail.dataset_preview_requested, timeout=1000) as blocker:
        rail._list.itemDoubleClicked.emit(item)

    assert blocker.args == ["d1"]


def test_context_menu_on_item_emits_dataset_preview_requested(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMenu

    library = DatasetLibrary()
    library.add(_ref("d1"))
    rail = LibraryRail(library)
    qtbot.addWidget(rail)
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[0])
    pos = rail._list.visualItemRect(rail._list.item(0)).center()

    with qtbot.waitSignal(rail.dataset_preview_requested, timeout=1000) as blocker:
        rail._on_context_menu_requested(pos)

    assert blocker.args == ["d1"]


def test_context_menu_signal_wiring_reaches_handler_through_real_emission(
    qtbot, monkeypatch
):
    # Every other context-menu test calls rail._on_context_menu_requested(pos)
    # directly. A missing or broken setContextMenuPolicy()/
    # customContextMenuRequested.connect(...) in __init__ would leave those
    # tests passing while a real right-click did nothing -- this test goes
    # through the real Qt signal instead of calling the handler.
    from PySide6.QtWidgets import QMenu

    library = DatasetLibrary()
    library.add(_ref("d1"))
    rail = LibraryRail(library)
    qtbot.addWidget(rail)
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[0])
    pos = rail._list.visualItemRect(rail._list.item(0)).center()

    with qtbot.waitSignal(rail.dataset_preview_requested, timeout=1000) as blocker:
        rail._list.customContextMenuRequested.emit(pos)

    assert blocker.args == ["d1"]


def test_context_menu_on_empty_space_does_nothing(qtbot):
    library = DatasetLibrary()
    rail = LibraryRail(library)
    qtbot.addWidget(rail)

    # No items in the list -- itemAt() at any position returns None, so this
    # must not raise and must not show a menu.
    rail._on_context_menu_requested(rail._list.rect().center())


def test_context_menu_targets_clicked_item_not_current_selection(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMenu

    library = DatasetLibrary()
    library.add(_ref("d1"))
    library.add(_ref("d2"))
    rail = LibraryRail(library)
    qtbot.addWidget(rail)
    rail._list.setCurrentRow(0)  # "d1" is the current selection
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[0])
    pos = rail._list.visualItemRect(rail._list.item(1)).center()  # right-click on "d2"

    with qtbot.waitSignal(rail.dataset_preview_requested, timeout=1000) as blocker:
        rail._on_context_menu_requested(pos)

    assert blocker.args == ["d2"]


def test_context_menu_delete_action_emits_dataset_delete_requested(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMenu

    library = DatasetLibrary()
    library.add(_ref("d1"))
    rail = LibraryRail(library)
    qtbot.addWidget(rail)
    monkeypatch.setattr(QMenu, "exec", lambda self, *a, **k: self.actions()[1])
    pos = rail._list.visualItemRect(rail._list.item(0)).center()

    with qtbot.waitSignal(rail.dataset_delete_requested, timeout=1000) as blocker:
        rail._on_context_menu_requested(pos)

    assert blocker.args == ["d1"]
