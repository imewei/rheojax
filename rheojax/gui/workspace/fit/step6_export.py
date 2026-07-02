from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import FitState


class ExportStep(QWidget):
    exported = Signal()

    def __init__(
        self, state: FitState, library: DatasetLibrary, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._save_btn = QPushButton("＋ Save fit → library", self)
        self._export_btn = QPushButton("⤓ Export", self)
        lay = QVBoxLayout(self)
        for w in (QLabel("Export bundle"), self._save_btn, self._export_btn):
            lay.addWidget(w)
        self._save_btn.clicked.connect(self.save_to_library)

    def bundle_manifest(self) -> list[str]:
        items = ["parameters", "fitted_curve", "figures", "provenance"]
        if self._state.nuts_result is not None:
            items += ["posterior_samples", "diagnostics"]
        return items

    def provenance(self) -> dict:
        return {
            "model_key": self._state.model_key,
            "model_config": self._state.model_config,
            "protocol": self._state.protocol,
            "data_ref": self._state.data_ref,
            "revision": self._state.revision,
        }

    def save_to_library(self) -> str:
        new_id = f"{self._state.model_key}_fit_{self._state.revision}"
        self._library.add(
            DatasetRef(
                id=new_id,
                name=new_id,
                protocol_type=self._state.protocol,
                origin="derived",
                units={},
                row_count=0,
                hash="",
                provenance=self.provenance(),
                lineage=(
                    [self._state.data_ref]
                    if self._state.data_ref
                    else []
                ),
            )
        )
        result = self._state.nlsq_result or {}
        x, y_fit = result.get("x"), result.get("y_fit")
        if x is not None and y_fit is not None:
            from types import SimpleNamespace

            import numpy as np

            self._library.store_payload(
                new_id, SimpleNamespace(x=np.asarray(x), y=np.asarray(y_fit))
            )
        self.exported.emit()
        return new_id
