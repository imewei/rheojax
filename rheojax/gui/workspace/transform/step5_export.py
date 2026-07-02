from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.services.export_service import ExportService


class TransformExportStep(QWidget):
    exported = Signal()

    def __init__(
        self,
        state: TransformState,
        library: DatasetLibrary,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._export_service = ExportService()
        self._save_btn = QPushButton("＋ Save output → library", self)
        self._export_btn = QPushButton("⤓ Export", self)
        lay = QVBoxLayout(self)
        for w in (QLabel("Export"), self._save_btn, self._export_btn):
            lay.addWidget(w)
        self._save_btn.clicked.connect(self.save_to_library)
        self._export_btn.clicked.connect(lambda: self.export_bundle(Path.cwd()))

    def bundle_manifest(self) -> list[str]:
        return ["output", "result", "figures", "provenance"]

    def provenance(self) -> dict:
        return {
            "transform_key": self._state.transform_key,
            "slots": dict(self._state.slots),
            "config": dict(self._state.config),
        }

    def export_bundle(self, directory: Path | str) -> dict[str, Path]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        result = self._state.result or {}

        output = result.get("output")
        if output is not None and hasattr(output, "x") and hasattr(output, "y"):
            data_path = directory / f"{self._state.transform_key}_output.csv"
            self._export_service.export_data(output, data_path)
            written["output"] = data_path

        structured = result.get("result")
        if structured:
            result_path = directory / f"{self._state.transform_key}_result.json"
            result_path.write_text(json.dumps(structured, default=str, indent=2))
            written["result"] = result_path

        provenance_path = directory / "provenance.json"
        provenance_path.write_text(json.dumps(self.provenance(), default=str, indent=2))
        written["provenance"] = provenance_path

        return written

    def save_to_library(self) -> str | None:
        ptype = (self._state.result or {}).get("protocol_type")
        if not ptype:
            return None
        base_id = f"{self._state.transform_key}_out"
        new_id = base_id
        suffix = 2
        while True:
            try:
                self._library.get(new_id)
            except KeyError:
                break
            new_id = f"{base_id}_{suffix}"
            suffix += 1
        lineage: list[str] = []
        for v in self._state.slots.values():
            if isinstance(v, list):
                lineage.extend(str(x) for x in v)
            elif v is not None:
                lineage.append(str(v))
        self._library.add(
            DatasetRef(
                id=new_id,
                name=new_id,
                protocol_type=ptype,
                origin="derived",
                units={},
                row_count=0,
                hash="",
                provenance=self.provenance(),
                lineage=lineage,
            )
        )
        output = (self._state.result or {}).get("output")
        if output is not None:
            self._library.store_payload(new_id, output)
        self.exported.emit()
        return new_id
