"""Behavior tests for RheoComboBox."""

from __future__ import annotations

from rheojax.gui.widgets.dropdown import RheoComboBox


def test_set_items_safely_does_not_emit_signals_mid_repopulate(qtbot):
    """Regression guard: repopulating must not fire currentIndexChanged for
    intermediate/auto-selected states while items are being cleared+added.
    """
    combo = RheoComboBox()
    qtbot.addWidget(combo)
    combo.set_items_safely(["a", "b"])

    fired = []
    combo.currentIndexChanged.connect(lambda idx: fired.append(idx))

    combo.set_items_safely(["c", "d", "e"], selected_data="d")

    assert fired == []
    assert combo.currentText() == "d"


def test_add_group_header_is_disabled_and_unselectable(qtbot):
    combo = RheoComboBox()
    qtbot.addWidget(combo)
    combo.addItem("real item")
    combo.add_group_header("── Group ──")
    combo.addItem("another item")

    header_idx = 1
    model = combo.model()
    item = model.item(header_idx)

    assert item is not None
    assert item.isEnabled() is False
    assert item.isSelectable() is False


def test_placeholder_starts_unselected(qtbot):
    combo = RheoComboBox(placeholder="Select a column...")
    qtbot.addWidget(combo)
    combo.set_items_safely(["x", "y", "z"])

    assert combo.currentIndex() == -1
    assert combo.current_data() is None


def test_set_current_data_found_and_not_found(qtbot):
    combo = RheoComboBox()
    qtbot.addWidget(combo)
    combo.set_items_safely([("A", 1), ("B", 2), ("C", 3)])

    assert combo.set_current_data(2) is True
    assert combo.current_data() == 2

    assert combo.set_current_data(999) is False
    # A not-found lookup leaves the previous valid selection in place.
    assert combo.current_data() == 2

    assert combo.set_current_data(None) is True
    assert combo.currentIndex() == -1


def test_set_density_round_trip(qtbot):
    combo = RheoComboBox(density="standard")
    qtbot.addWidget(combo)
    assert combo.property("density") == "standard"

    combo.set_density("compact")
    assert combo.property("density") == "compact"

    combo.set_density("standard")
    assert combo.property("density") == "standard"
