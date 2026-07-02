from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

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
        self._layout = QVBoxLayout(self)
        self.refresh()

    def refresh(self) -> None:
        """Rebuild slot specs + displayed labels from the current transform_key.

        Bodies are constructed eagerly (before the user picks a transform),
        so `_specs` must be recomputed here rather than frozen in __init__ --
        otherwise it stays stuck at the `transform_key=None` fallback forever.
        Call after Step 1 (TransformPickStep) emits `edited`.
        """
        self._specs = transform_slots(self._state.transform_key)
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        for s in self._specs:
            self._layout.addWidget(
                QLabel(
                    f"Slot: {s.name} ({s.accepts or 'any'})"
                    + (" [list]" if s.is_list else "")
                )
            )

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
