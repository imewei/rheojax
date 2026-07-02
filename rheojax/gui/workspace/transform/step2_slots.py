from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.slots_spec import SlotSpec, transform_slots


class SlotsStep(QWidget):
    edited = Signal()

    def __init__(
        self,
        state: TransformState,
        library: DatasetLibrary,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._specs: list[SlotSpec] = []
        self._combo_widgets: dict[str, QComboBox] = {}
        self._list_widgets: dict[str, QListWidget] = {}
        self._list_add_combos: dict[str, QComboBox] = {}
        self._list_add_buttons: dict[str, QPushButton] = {}
        self._list_remove_buttons: dict[str, QPushButton] = {}
        self._layout = QVBoxLayout(self)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild slot specs + selector widgets from the current transform_key.

        Also re-validates any already-filled slot against the current
        candidates, dropping a selection that no longer resolves (e.g. the
        dataset was removed from the library) -- mirrors DataStep.refresh().
        """
        self._specs = transform_slots(self._state.transform_key)
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._combo_widgets.clear()
        self._list_widgets.clear()
        self._list_add_combos.clear()
        self._list_add_buttons.clear()
        self._list_remove_buttons.clear()

        for s in self._specs:
            self._layout.addWidget(
                QLabel(
                    f"Slot: {s.name} ({s.accepts or 'any'})"
                    + (" [list]" if s.is_list else "")
                )
            )
            if s.is_list:
                self._add_list_slot_widgets(s)
            else:
                self._add_single_slot_widget(s)

    def _add_single_slot_widget(self, spec: SlotSpec) -> None:
        candidates = self.candidates(spec.name)
        current = self._state.slots.get(spec.name)
        if isinstance(current, str) and current not in candidates:
            self._state.slots.pop(spec.name, None)
            current = None
        combo = QComboBox(self)
        combo.addItems([""] + candidates)
        if current:
            combo.setCurrentText(current)
        combo.currentTextChanged.connect(
            lambda text, name=spec.name: self.fill(name, text) if text else None
        )
        self._combo_widgets[spec.name] = combo
        self._layout.addWidget(combo)

    def _add_list_slot_widgets(self, spec: SlotSpec) -> None:
        current_list = [
            v for v in self._state.slots.get(spec.name, []) if v in self.candidates(spec.name)
        ]
        if current_list != self._state.slots.get(spec.name, []):
            self._state.slots[spec.name] = current_list

        list_widget = QListWidget(self)
        list_widget.addItems(current_list)
        add_combo = QComboBox(self)
        add_combo.addItems(
            [""] + [c for c in self.candidates(spec.name) if c not in current_list]
        )
        add_btn = QPushButton("Add", self)
        remove_btn = QPushButton("Remove", self)

        def _add() -> None:
            text = add_combo.currentText()
            if not text:
                return
            values = list(self._state.slots.get(spec.name, []))
            if text not in values:
                values.append(text)
                self.fill(spec.name, values)
                self.refresh()

        def _remove() -> None:
            item = list_widget.currentItem()
            if item is None:
                return
            values = [v for v in self._state.slots.get(spec.name, []) if v != item.text()]
            self.fill(spec.name, values)
            self.refresh()

        add_btn.clicked.connect(_add)
        remove_btn.clicked.connect(_remove)
        self._list_widgets[spec.name] = list_widget
        self._list_add_combos[spec.name] = add_combo
        self._list_add_buttons[spec.name] = add_btn
        self._list_remove_buttons[spec.name] = remove_btn
        for w in (list_widget, add_combo, add_btn, remove_btn):
            self._layout.addWidget(w)

    def slot_specs(self) -> list[SlotSpec]:
        return list(self._specs)

    def candidates(self, slot_name: str) -> list[str]:
        spec = next(s for s in self._specs if s.name == slot_name)
        refs = (
            self._library.datasets_of_type(spec.accepts)
            if spec.accepts
            else self._library.all()
        )
        return [r.id for r in refs]

    def fill(self, slot_name: str, value) -> None:
        self._state.slots[slot_name] = value
        self.edited.emit()

    def is_ready(self) -> bool:
        return all(s.name in self._state.slots for s in self._specs)
