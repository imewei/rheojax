"""Pipeline mode's single step body: assemble a step sequence, select datasets, Run All.

No per-step "Run Step" button by design (design spec §3) -- running a single step interactively
is already what Fit/Transform modes are for. This widget only supports batch orchestration.

Each step type has real configuration controls (model/NUTS for "fit", path/format for
"export", transform key for "transform") -- an earlier version of this widget only collected
config for "transform" steps, so a user-added "fit" or "export" step had an empty {} config and
failed with KeyError at execute() time. is_ready() now also validates that every configured
step has the fields execute() actually requires, not just that the steps list is non-empty.
"""

from __future__ import annotations

import uuid

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.gui.foundation.state import PipelineStepConfig
from rheojax.gui.workspace.transform.slots_spec import _MULTI, _TYPED_PAIRS

_EXCLUDED_TRANSFORM_KEYS = set(_MULTI) | set(_TYPED_PAIRS)
_DATASET_ID_ROLE = Qt.ItemDataRole.UserRole


def _step_config_is_complete(step: PipelineStepConfig) -> bool:
    """A step's config must actually satisfy what execute() requires for that step_type
    (§3.2/§3.4) -- an empty {} is never valid for "fit" or "export"."""
    if step.step_type == "transform":
        return bool(step.config.get("name"))
    if step.step_type == "fit":
        return bool(step.config.get("model_name"))
    if step.step_type == "export":
        return bool(step.config.get("path"))
    return False


class PipelineConfigureRunStep(QWidget):
    edited = Signal()
    run_requested = Signal()

    def __init__(self, state, library, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._state = state
        self._library = library

        self._step_type_combo = QComboBox(self)
        self._step_type_combo.addItems(["transform", "fit", "export"])

        self._transform_key_combo = QComboBox(self)
        self._transform_key_combo.addItems(self.available_transform_keys())

        self._fit_model_combo = QComboBox(self)
        self._fit_model_combo.addItems(ModelRegistry.list_models())
        self._fit_run_nuts_checkbox = QCheckBox("Run NUTS after NLSQ", self)

        self._export_path_edit = QLineEdit(self)
        self._export_path_edit.setPlaceholderText(
            "/path/to/output_{id}.csv  ({id} = dataset id)"
        )
        self._export_format_combo = QComboBox(self)
        self._export_format_combo.addItems(["csv", "json", "xlsx", "hdf5"])

        self._add_step_btn = QPushButton("+ Add Step", self)
        self._add_step_btn.clicked.connect(self._on_add_step_clicked)
        self._step_list = QListWidget(self)
        self._remove_step_btn = QPushButton("Remove Selected Step", self)
        self._remove_step_btn.clicked.connect(self._on_remove_step_clicked)

        self._dataset_list = QListWidget(self)
        self._dataset_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self._refresh_dataset_list()
        self._dataset_list.itemSelectionChanged.connect(
            self._on_dataset_selection_changed
        )

        self._run_all_btn = QPushButton("▶ Run All", self)
        self._run_all_btn.clicked.connect(self.run_requested.emit)

        layout = QVBoxLayout(self)
        layout.addWidget(self._step_type_combo)
        layout.addWidget(self._transform_key_combo)
        layout.addWidget(self._fit_model_combo)
        layout.addWidget(self._fit_run_nuts_checkbox)
        layout.addWidget(self._export_path_edit)
        layout.addWidget(self._export_format_combo)
        layout.addWidget(self._add_step_btn)
        layout.addWidget(self._step_list)
        layout.addWidget(self._remove_step_btn)
        layout.addWidget(self._dataset_list)
        layout.addWidget(self._run_all_btn)

    def available_transform_keys(self) -> list[str]:
        return [
            k
            for k in TransformRegistry.list_transforms()
            if k not in _EXCLUDED_TRANSFORM_KEYS
        ]

    def add_step(self, step_type: str, config: dict) -> None:
        step = PipelineStepConfig(
            id=uuid.uuid4().hex, step_type=step_type, config=dict(config)
        )
        self._state.steps.append(step)
        self._step_list.addItem(QListWidgetItem(f"{step.step_type}: {step.config}"))
        self.edited.emit()

    def _on_add_step_clicked(self) -> None:
        step_type = self._step_type_combo.currentText()
        if step_type == "transform":
            config = {"name": self._transform_key_combo.currentText()}
        elif step_type == "fit":
            config = {
                "model_name": self._fit_model_combo.currentText(),
                "run_nuts": self._fit_run_nuts_checkbox.isChecked(),
            }
        else:  # "export"
            config = {
                "path": self._export_path_edit.text(),
                "format": self._export_format_combo.currentText(),
            }
        self.add_step(step_type, config)

    def _on_remove_step_clicked(self) -> None:
        row = self._step_list.currentRow()
        if row < 0:
            return
        self._step_list.takeItem(row)
        del self._state.steps[row]
        self.edited.emit()

    def _refresh_dataset_list(self) -> None:
        self._dataset_list.clear()
        for ref in self._library.all():
            item = QListWidgetItem(ref.name)
            item.setData(_DATASET_ID_ROLE, ref.id)
            self._dataset_list.addItem(item)

    def set_selected_dataset_ids(self, ids: list[str]) -> None:
        self._state.selected_dataset_ids = list(ids)
        self.edited.emit()

    def _on_dataset_selection_changed(self) -> None:
        selected = [
            item.data(_DATASET_ID_ROLE) for item in self._dataset_list.selectedItems()
        ]
        self.set_selected_dataset_ids(selected)

    def is_ready(self) -> bool:
        return (
            bool(self._state.steps)
            and all(_step_config_is_complete(s) for s in self._state.steps)
            and bool(self._state.selected_dataset_ids)
        )
