from __future__ import annotations

import json
import uuid
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.compat import QFileDialog, QThreadPool
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import ActiveJobsState, TransformState
from rheojax.gui.jobs.export_worker import ExportWorker
from rheojax.gui.resources.styles.tokens import field_label_style
from rheojax.gui.services.export_service import ExportService
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.logging import get_logger

logger = get_logger(__name__)


class TransformExportStep(QWidget):
    dataset_commit_requested = Signal(
        object, object, bool
    )  # ref, payload | None, overwrite

    def __init__(
        self,
        state: TransformState,
        library: DatasetLibrary,
        active_jobs: ActiveJobsState | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._active_jobs = active_jobs
        self._export_service = ExportService()
        # Keyed by job_id (mirrors window.py's _active_import_workers); read
        # via .pop() below, so never a dead store.
        self._active_export_workers: dict[str, ExportWorker] = {}
        self._save_btn = QPushButton("＋ Save output → library", self)
        # "Bundle" (not bare "Export"): the plot canvas in the Transform tab
        # has its own "Export Chart..." button (single image), same naming
        # collision this disambiguates in the Fit tab's ExportStep.
        self._export_btn = QPushButton("⤓ Export Bundle...", self)
        self._export_btn.setToolTip(
            "Write the transform output, result, and provenance to a folder"
        )
        self._export_status = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        caption = QLabel("Export")
        caption.setStyleSheet(field_label_style())
        for w in (caption, self._save_btn, self._export_btn, self._export_status):
            lay.addWidget(w)
        lay.addStretch()  # see step1_protocol_model.py's addStretch() comment
        self._save_btn.clicked.connect(self.save_to_library)
        self._export_btn.clicked.connect(self._on_export_clicked)

    def _on_export_clicked(self) -> None:
        """Prompt for a destination directory rather than silently writing
        to Path.cwd() -- the user previously had no way to know or choose
        where the bundle landed."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Export bundle to...", str(Path.cwd())
        )
        if not chosen:
            return
        directory = Path(chosen)
        # export_bundle() does real disk I/O; run it off the GUI thread via
        # ExportWorker so it doesn't freeze the window for its duration.
        self._export_btn.setEnabled(False)
        self._export_status.setText("Exporting...")
        # Registered in active_jobs (mirrors window.py's _launch_import and
        # fit_controller.py's _make_fit_fn) so Close/New/Open wait for this
        # export to finish instead of tearing down this widget while the
        # worker thread is still mid-write and about to emit into it. No
        # "worker" key: like ImportWorker, ExportWorker has no cancel() path.
        job_id = f"export:{uuid.uuid4().hex}"
        if self._active_jobs is not None:
            self._active_jobs.by_id[job_id] = {}
        worker = ExportWorker(lambda _progress: self.export_bundle(directory))
        worker.signals.completed.connect(
            lambda _result, jid=job_id: self._on_export_finished(directory, jid)
        )
        worker.signals.failed.connect(
            lambda msg, jid=job_id: self._on_export_failed(msg, jid)
        )
        # QThreadPool takes ownership of `worker` on the C++ side, but that
        # doesn't keep its Python wrapper (or the plain QObject at
        # worker.signals, which has no Qt parent) alive once this method's
        # local `worker` goes out of scope -- GC could tear it down mid-run,
        # silently dropping the completed/failed signal. Hold a reference
        # until one of those signals fires.
        self._active_export_workers[job_id] = worker
        QThreadPool.globalInstance().start(worker)

    def _clear_export_job(self, job_id: str) -> None:
        self._active_export_workers.pop(job_id, None)
        if self._active_jobs is not None:
            self._active_jobs.by_id.pop(job_id, None)

    def _on_export_finished(self, directory: Path, job_id: str) -> None:
        self._clear_export_job(job_id)
        self._export_btn.setEnabled(True)
        self._export_status.setText(f"Exported to {directory}")

    def _on_export_failed(self, message: str, job_id: str) -> None:
        self._clear_export_job(job_id)
        self._export_btn.setEnabled(True)
        logger.error("Export failed", error=message)
        self._export_status.setText(f"Export failed: {message}")

    def bundle_manifest(self) -> list[str]:
        return ["output", "result", "provenance"]

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
        # `.get("protocol_type")` returns None when the result dict never
        # carried the key at all (e.g. a genuinely scalar/typeless output
        # with no dataset-shaped payload to register -- see
        # test_scalar_output_not_saved). transform_controller._run() always
        # includes the key, using "" (not None) for domain-changing
        # transforms whose output has no rheological protocol but is still a
        # real, storable RheoData -- per design §7 those must be "stored but
        # not offered to typed Fit slots", not silently dropped entirely.
        result = self._state.result or {}
        if "protocol_type" not in result:
            return None
        ptype = result.get("protocol_type") or ""
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
        ref = DatasetRef(
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
        output = (self._state.result or {}).get("output")
        self.dataset_commit_requested.emit(ref, output, False)
        return new_id
