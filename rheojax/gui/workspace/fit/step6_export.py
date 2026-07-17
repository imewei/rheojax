from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from rheojax.gui.compat import QFileDialog
from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import FitState
from rheojax.gui.resources.styles.tokens import field_label_style
from rheojax.gui.services.export_service import ExportService
from rheojax.gui.utils.layout_helpers import set_panel_margins
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ExportStep(QWidget):
    dataset_commit_requested = Signal(
        object, object, bool
    )  # ref, payload | None, overwrite

    def __init__(
        self, state: FitState, library: DatasetLibrary, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library
        self._export_service = ExportService()
        self._save_btn = QPushButton("＋ Save fit → library", self)
        self._export_btn = QPushButton("⤓ Export", self)
        self._export_status = QLabel("", self)
        lay = QVBoxLayout(self)
        set_panel_margins(lay)
        bundle_caption = QLabel("Export bundle")
        bundle_caption.setStyleSheet(field_label_style())
        for w in (bundle_caption, self._save_btn, self._export_btn, self._export_status):
            lay.addWidget(w)
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
        try:
            self.export_bundle(Path(chosen))
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            # ExportService wraps most of its own failures as RuntimeError
            # ("Export failed: ..."), but export_posterior_netcdf() (used
            # for the NUTS branch below) has no such wrapper -- a malformed/
            # empty posterior_samples dict can raise ValueError/TypeError
            # straight from arviz.from_dict()/to_netcdf() uncaught. An
            # OSError-only catch here let both of those escape this Qt slot.
            # Named to these four (not a bare Exception) so an actual
            # programming bug elsewhere in export_bundle() -- AttributeError,
            # KeyError from a typo -- still surfaces loudly instead of being
            # reported identically to a legitimate "your data is malformed"
            # failure; exc_info logs it either way for post-mortem digging.
            logger.error("Export failed", exc_info=True)
            self._export_status.setText(f"Export failed: {exc}")
            return
        self._export_status.setText(f"Exported to {Path(chosen)}")

    def bundle_manifest(self) -> list[str]:
        # ponytail: "figures" isn't listed -- export_bundle() has no figure
        # writer (VisualizeStep's canvases are pyqtgraph, not matplotlib
        # Figures export_service.export_figure() expects); add it back only
        # once that's actually wired.
        items = ["parameters"]
        # Must match export_bundle()'s own x/y_fit presence check below --
        # promising "fitted_curve" unconditionally (regardless of whether
        # nlsq_result actually carries x/y_fit) meant a caller trusting this
        # manifest as a preview would expect a file export_bundle() never
        # wrote, exactly the promise-vs-write mismatch already fixed below
        # for "diagnostics".
        result = self._state.nlsq_result or {}
        if result.get("x") is not None and result.get("y_fit") is not None:
            items.append("fitted_curve")
        items.append("provenance")
        if self._state.nuts_result is not None:
            items += ["posterior_samples", "diagnostics"]
        return items

    def provenance(self) -> dict:
        prov = {
            "model_key": self._state.model_key,
            "model_config": self._state.model_config,
            "protocol": self._state.protocol,
            "data_ref": self._state.data_ref,
            "revision": self._state.revision,
        }
        if self._state.nuts_result and "verdict" in self._state.nuts_result:
            prov["convergence_verdict"] = self._state.nuts_result["verdict"]
        return prov

    def export_bundle(self, directory: Path | str) -> dict[str, Path]:
        """Write parameters/fitted-curve/provenance (+ posterior, if run) to `directory`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}

        result = self._state.nlsq_result or {}
        params_path = directory / "parameters.csv"
        self._export_service.export_parameters(
            SimpleNamespace(parameters=result.get("params", {})), params_path
        )
        written["parameters"] = params_path

        x, y_fit = result.get("x"), result.get("y_fit")
        if x is not None and y_fit is not None:
            curve_path = directory / "fitted_curve.csv"
            np.savetxt(
                curve_path,
                np.column_stack([x, y_fit]),
                delimiter=",",
                header="x,y_fit",
                comments="",
            )
            written["fitted_curve"] = curve_path

        if self._state.nuts_result is not None:
            posterior_path = directory / "posterior_samples.nc"
            self._export_service.export_posterior_netcdf(
                self._state.nuts_result, posterior_path
            )
            written["posterior_samples"] = posterior_path

            # bundle_manifest() promises a "diagnostics" item whenever NUTS
            # ran; write the R-hat/ESS/BFMI/divergences/verdict data (all
            # already computed and stored on nuts_result) as its own file
            # instead of only folding the verdict into provenance.json.
            diagnostics_path = directory / "diagnostics.json"
            diagnostics = {
                k: self._state.nuts_result.get(k)
                for k in ("r_hat", "ess", "bfmi", "divergences", "verdict")
                if k in self._state.nuts_result
            }
            diagnostics_path.write_text(json.dumps(diagnostics, default=str, indent=2))
            written["diagnostics"] = diagnostics_path

        provenance_path = directory / "provenance.json"
        provenance_path.write_text(json.dumps(self.provenance(), default=str, indent=2))
        written["provenance"] = provenance_path

        return written

    def save_to_library(self) -> str:
        new_id = f"{self._state.model_key}_fit_{self._state.revision}"
        ref = DatasetRef(
            id=new_id,
            name=new_id,
            protocol_type=self._state.protocol,
            origin="derived",
            units={},
            row_count=0,
            hash="",
            provenance=self.provenance(),
            lineage=([self._state.data_ref] if self._state.data_ref else []),
        )
        result = self._state.nlsq_result or {}
        x, y_fit = result.get("x"), result.get("y_fit")
        output = None
        if x is not None and y_fit is not None:
            output = SimpleNamespace(x=np.asarray(x), y=np.asarray(y_fit))
        self.dataset_commit_requested.emit(ref, output, True)
        return new_id
