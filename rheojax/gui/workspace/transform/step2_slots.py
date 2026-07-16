from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.resources.styles.tokens import field_label_style
from rheojax.gui.services.transform_service import TransformService
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.gui.widgets import RheoComboBox
from rheojax.gui.widgets.parameter_form import ParameterFormBuilder
from rheojax.gui.workspace.transform.slots_spec import SlotSpec, transform_slots


class SlotsStep(QWidget):
    edited = Signal()

    def __init__(
        self,
        state: TransformState,
        library: DatasetLibrary,
        parent: QWidget | None = None,
        transform_service: TransformService | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._transform_service = transform_service or TransformService()
        self._specs: list[SlotSpec] = []
        self._combo_widgets: dict[str, RheoComboBox] = {}
        self._list_widgets: dict[str, QListWidget] = {}
        self._list_add_combos: dict[str, RheoComboBox] = {}
        self._list_add_buttons: dict[str, QPushButton] = {}
        self._list_remove_buttons: dict[str, QPushButton] = {}
        self._param_form: ParameterFormBuilder | None = None
        self._layout = QVBoxLayout(self)
        set_panel_margins(self._layout)
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
        self._param_form = None

        for s in self._specs:
            caption = QLabel(
                f"Slot: {s.name} ({s.accepts or 'any'})"
                + (" [list]" if s.is_list else "")
            )
            caption.setStyleSheet(field_label_style())
            self._layout.addWidget(caption)
            if s.is_list:
                self._add_list_slot_widgets(s)
            else:
                self._add_single_slot_widget(s)

        self._add_param_form()

    def _add_param_form(self) -> None:
        """Auto-generated parameter form for the current transform.

        state.config previously stayed {} for every run (no UI ever wrote to
        it) -- transforms always ran with library defaults and the user had
        no way to see or change them. Seed state.config from the spec
        defaults so RunStep always passes an explicit config, not a silent {}.
        """
        params = (
            self._transform_service.get_transform_params(self._state.transform_key)
            if self._state.transform_key
            else {}
        )
        if not params:
            return
        caption = QLabel("Parameters:")
        caption.setStyleSheet(field_label_style())
        self._layout.addWidget(caption)
        # Preserve an already-edited value across a rebuild (e.g. triggered by
        # adding/removing a list slot, which calls refresh()) instead of
        # resetting the widget to the transform's spec default while
        # state.config silently keeps the edited value -- that mismatch left
        # the UI showing a stale default even though a different value was
        # actually in effect.
        specs_with_current_values = {
            name: (
                {**spec, "default": self._state.config[name]}
                if name in self._state.config
                else spec
            )
            for name, spec in params.items()
        }
        self._param_form = ParameterFormBuilder(specs_with_current_values, self)
        for name, spec in params.items():
            self._state.config.setdefault(name, spec["default"])
        self._param_form.values_changed.connect(self._on_params_changed)
        self._layout.addWidget(self._param_form)

    def _on_params_changed(self) -> None:
        if self._param_form is None:
            return
        self._state.config.update(self._param_form.get_values())
        self.edited.emit()

    def _add_single_slot_widget(self, spec: SlotSpec) -> None:
        candidates = self.candidates(spec.name)
        current = self._state.slots.get(spec.name)
        if isinstance(current, str) and current not in candidates:
            self._state.slots.pop(spec.name, None)
            current = None
        combo = RheoComboBox(self)
        combo.set_items_safely([""] + candidates, selected_data=current or None)
        combo.currentTextChanged.connect(
            lambda text, name=spec.name: self._on_single_slot_changed(name, text)
        )
        self._combo_widgets[spec.name] = combo
        self._layout.addWidget(combo)

    def _add_list_slot_widgets(self, spec: SlotSpec) -> None:
        current_list = [
            v
            for v in self._state.slots.get(spec.name, [])
            if v in self.candidates(spec.name)
        ]
        if current_list != self._state.slots.get(spec.name, []):
            self._state.slots[spec.name] = current_list

        list_widget = QListWidget(self)
        list_widget.addItems(current_list)
        add_combo = RheoComboBox(self)
        add_combo.set_items_safely(
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
            values = [
                v for v in self._state.slots.get(spec.name, []) if v != item.text()
            ]
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

    def _on_single_slot_changed(self, slot_name: str, text: str) -> None:
        if text:
            self.fill(slot_name, text)
            return
        # Selecting the blank "" entry must clear the slot in state too --
        # otherwise the combo shows no selection while state.slots still
        # holds the previous value, and is_ready() (which checks key
        # presence) keeps reporting this slot as filled.
        if slot_name in self._state.slots:
            del self._state.slots[slot_name]
            self.edited.emit()

    def slot_specs(self) -> list[SlotSpec]:
        return list(self._specs)

    def candidates(self, slot_name: str) -> list[str]:
        spec = next((s for s in self._specs if s.name == slot_name), None)
        if spec is None:
            return []
        refs = (
            self._library.datasets_of_type(spec.accepts)
            if spec.accepts
            else self._library.all()
        )
        return [r.id for r in refs]

    def fill(self, slot_name: str, value) -> None:
        if not any(s.name == slot_name for s in self._specs):
            raise ValueError(f"unknown slot: {slot_name!r}")
        self._state.slots[slot_name] = value
        self.edited.emit()

    def is_ready(self) -> bool:
        for s in self._specs:
            value = self._state.slots.get(s.name)
            if s.name not in self._state.slots:
                return False
            if s.is_list and not value:
                return False
        return True
