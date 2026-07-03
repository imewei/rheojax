from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.contract import input_contract
from rheojax.gui.foundation.library import DatasetLibrary
from rheojax.gui.foundation.state import FitState
from rheojax.gui.resources.styles.tokens import field_label_style
from rheojax.gui.utils.layout_helpers import set_panel_margins


def _validate_shape_and_values(rheo_data) -> list[str]:
    """Shape/NaN/monotonicity checks against a loaded RheoData. Empty = valid."""
    errors: list[str] = []
    x = np.asarray(rheo_data.x)
    y = np.asarray(rheo_data.y)
    if x.shape[0] != y.shape[0]:
        errors.append(f"x/y length mismatch: {x.shape[0]} vs {y.shape[0]}")
        return errors  # remaining checks assume matching length
    x_has_nan = bool(np.isnan(x).any())
    if x_has_nan:
        errors.append("x contains NaN values")
    if np.isnan(y).any():
        errors.append("y contains NaN values")
    # Monotonicity is undefined in the presence of NaNs (diff/comparisons
    # against NaN are always False, which would spuriously flag "not
    # monotonic" on top of the NaN error already reported above).
    if (
        not x_has_nan
        and x.shape[0] > 1
        and not (np.all(np.diff(x) > 0) or np.all(np.diff(x) < 0))
    ):
        errors.append("x is not monotonic")
    return errors


class DataStep(QWidget):
    edited = Signal()

    def __init__(
        self, state: FitState, library: DatasetLibrary, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._errors: list[str] = []
        self._unit_converted = False
        # Guard: protocol may be None on a fresh AppState; defer contract build until set
        self._contract, cols_text = self._build_contract()
        self._expected = QLabel(cols_text, self)
        self._source = QComboBox(self)
        self._source.addItems([""] + self.available_datasets())
        self._guard = QLabel("", self)
        self._convert_btn = QPushButton("⇄ Convert Hz → rad/s", self)
        self._error_label = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        source_caption = QLabel("Source")
        source_caption.setStyleSheet(field_label_style())
        for w in (
            self._expected,
            source_caption,
            self._source,
            self._guard,
            self._convert_btn,
            self._error_label,
        ):
            lay.addWidget(w)
        self._source.currentTextChanged.connect(self._on_select)
        self._convert_btn.clicked.connect(self.apply_unit_conversion)

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
        old_y_quantity = self._contract.y_quantity if self._contract else None
        self._contract, cols_text = self._build_contract()
        self._expected.setText(cols_text)

        current = self._source.currentText()
        new_datasets = self.available_datasets()
        # A model switch within the same protocol can change what the y
        # column *means* (e.g. flow_curve: "stress" vs "viscosity") without
        # changing which datasets are protocol-eligible. Force re-selection
        # rather than silently reinterpreting the previously-picked column
        # as a different physical quantity.
        y_quantity_changed = (
            self._contract is not None and self._contract.y_quantity != old_y_quantity
        )
        still_valid = (
            bool(current) and current in new_datasets and not y_quantity_changed
        )

        self._source.blockSignals(True)
        self._source.clear()
        self._source.addItems([""] + new_datasets)
        self._source.setCurrentText(current if still_valid else "")
        self._source.blockSignals(False)

        if still_valid:
            # The upstream Step-1 edit cascade clears state.data_ref/column_map
            # before refresh() runs (it can't tell the old selection is still
            # valid for the new protocol/model). Re-derive them here via the
            # same logic _on_select uses for a fresh pick, since signals were
            # blocked above and _on_select was never triggered for `current`.
            self._on_select(current)
        elif self._state.data_ref is not None or self._state.column_map:
            self._state.data_ref = None
            self._state.column_map = {}
            self._guard.setText("")
            self._set_errors([])
            self.edited.emit()

    def expected_columns(self) -> list[str]:
        return [c.role for c in self._contract.columns] if self._contract else []

    def available_datasets(self) -> list[str]:
        return [r.id for r in self._library.datasets_of_type(self._state.protocol)]

    def select_dataset(self, ds_id: str) -> None:
        self._source.setCurrentText(ds_id)

    def _on_select(self, ds_id: str) -> None:
        self._unit_converted = False
        self._state.data_ref = ds_id or None
        errors: list[str] = []
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
            try:
                payload = self._library.load_payload(ds_id)
            except KeyError:
                # No payload stored for this ref (e.g. a metadata-only/
                # catalog-browsed DatasetRef, or a derived dataset saved
                # without a payload -- see step6_export.py's save_to_library
                # when x/y_fit are missing). This is a real error, not a
                # "nothing to validate" no-op: without it, is_ready() would
                # report True for a dataset with no actual data, deferring
                # the failure to a KeyError crash later in NLSQ/NUTS.
                errors = ["dataset has no data payload loaded"]
                payload = None
            if payload is not None:
                errors = _validate_shape_and_values(payload)
        self._set_errors(errors)
        self.edited.emit()

    def _set_errors(self, errors: list[str]) -> None:
        self._errors = errors
        self._error_label.setText("⚠ " + "; ".join(errors) if errors else "")

    def validation_errors(self) -> list[str]:
        return list(self._errors)

    def needs_hz_conversion(self) -> bool:
        if self.unit_conversion_applied():
            return False
        if (
            not self._contract
            or "x" not in self._contract.unit_conversions
            or not self._state.data_ref
        ):
            return False
        ref = self._library.get(self._state.data_ref)
        return ref.units.get("x", "").lower() in ("hz", "hertz")

    def apply_unit_conversion(self) -> None:
        """Execute the Hz -> rad/s (x2pi) conversion flagged by needs_hz_conversion()."""
        if not self.needs_hz_conversion():
            # Subsumes the old data_ref/already-converted guard, and also
            # refuses to scale data whose units were never actually Hz --
            # important now that this is wired to a real button click
            # instead of only being called from tests.
            return
        rheo_data = self._library.load_payload(self._state.data_ref)
        rheo_data.x = np.asarray(rheo_data.x) * (2 * np.pi)
        # Persist the conversion in the library's authoritative units
        # metadata, not just this widget's transient _unit_converted flag --
        # _on_select() unconditionally resets that flag to False on every
        # call, including refresh()'s still_valid branch re-selecting the
        # SAME already-converted dataset. Without updating DatasetRef.units,
        # needs_hz_conversion() would report True again and a second click
        # would silently re-apply the x2pi conversion.
        # add(overwrite=True) clears any stale payload under this id, so it
        # must run BEFORE store_payload() re-establishes the converted data --
        # doing it after would wipe out the very payload just stored.
        ref = self._library.get(self._state.data_ref)
        self._library.add(replace(ref, units={**ref.units, "x": "rad/s"}), overwrite=True)
        self._library.store_payload(self._state.data_ref, rheo_data)
        self._unit_converted = True
        self._guard.setText("")
        self.edited.emit()

    def unit_conversion_applied(self) -> bool:
        return self._unit_converted

    def is_ready(self) -> bool:
        return bool(
            self._state.data_ref and self._state.column_map and not self._errors
        )
