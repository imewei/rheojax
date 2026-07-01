from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget

from rheojax.gui.foundation.contract import input_contract
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState


class DataStep(QWidget):
    edited = Signal()

    def __init__(
        self, state: FitState, library: DatasetLibrary, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        # Guard: protocol may be None on a fresh AppState; defer contract build until set
        self._contract, cols_text = self._build_contract()
        self._expected = QLabel(cols_text, self)
        self._source = QComboBox(self)
        self._source.addItems([""] + self.available_datasets())
        self._guard = QLabel("", self)
        lay = QVBoxLayout(self)
        for w in (self._expected, QLabel("Source"), self._source, self._guard):
            lay.addWidget(w)
        self._source.currentTextChanged.connect(self._on_select)

    def _build_contract(self):
        """Derive (contract, label text) from the current state's protocol/model_key."""
        contract = (
            input_contract(self._state.protocol, self._state.model_key)
            if self._state.protocol
            else None
        )
        cols_text = (
            "Expecting: " + ", ".join(f"{c.role} [{c.unit}]" for c in contract.columns)
            if contract
            else "Choose a protocol first"
        )
        return contract, cols_text

    def refresh(self) -> None:
        """Rebuild contract/label/combo from current state (call after Step 1 edits)."""
        self._contract, cols_text = self._build_contract()
        self._expected.setText(cols_text)

        current = self._source.currentText()
        new_datasets = self.available_datasets()
        still_valid = bool(current) and current in new_datasets

        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItems([""] + new_datasets)
        self._source.setCurrentText(current if still_valid else "")
        self._source.blockSignals(False)

        if not still_valid and (self._state.data_ref is not None or self._state.column_map):
            self._state.data_ref = None
            self._state.column_map = {}
            self._guard.setText("")
            self.edited.emit()

    def expected_columns(self) -> list[str]:
        return [c.role for c in self._contract.columns] if self._contract else []

    def available_datasets(self) -> list[str]:
        return [r.id for r in self._library.datasets_of_type(self._state.protocol)]

    def select_dataset(self, ds_id: str) -> None:
        self._source.setCurrentText(ds_id)

    def _on_select(self, ds_id: str) -> None:
        self._state.data_ref = ds_id or None
        if ds_id and self._contract:
            # default column map = role -> positional index
            self._state.column_map = {
                c.role: i for i, c in enumerate(self._contract.columns)
            }
            self._guard.setText(
                "⚠ x in Hz → ×2π to rad/s before fit"
                if self.needs_hz_conversion()
                else ""
            )
        self.edited.emit()

    def needs_hz_conversion(self) -> bool:
        if (
            not self._contract
            or "x" not in self._contract.unit_conversions
            or not self._state.data_ref
        ):
            return False
        ref = self._library.get(self._state.data_ref)
        return ref.units.get("x", "").lower() in ("hz", "hertz")

    def is_ready(self) -> bool:
        return bool(self._state.data_ref and self._state.column_map)
