"""Regression: TransformPage's "Datasets (select 2+)" checklist had real
checkboxes, but nothing ever read their check state -- checking only 2 of 3
loaded datasets still ran the (mastercurve/srfs/cox_merz) transform over all
3. TransformPage.get_checked_dataset_ids() now exposes the checked subset,
and MainWindow._on_apply_transform() uses it to filter which datasets run.
"""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.gui

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QListWidgetItem

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_get_checked_dataset_ids_returns_only_checked_items(qtbot):
    from rheojax.gui.pages.transform_page import TransformPage
    from rheojax.gui.state.store import StateStore

    StateStore.reset()
    page = TransformPage()
    qtbot.addWidget(page)

    assert page.get_checked_dataset_ids() == []

    from PySide6.QtWidgets import QListWidget

    page._dataset_checklist = QListWidget()
    for ds_id, checked in [("d1", True), ("d2", False), ("d3", True)]:
        item = QListWidgetItem(ds_id)
        item.setData(Qt.UserRole, ds_id)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        page._dataset_checklist.addItem(item)

    assert page.get_checked_dataset_ids() == ["d1", "d3"]


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 required")
def test_apply_transform_uses_checked_subset_not_all_datasets(qtbot):
    import numpy as np

    from rheojax.gui.app.main_window import RheoJAXMainWindow
    from rheojax.gui.state.store import AppState, DatasetState, StateStore

    StateStore.reset()
    window = RheoJAXMainWindow()
    qtbot.addWidget(window)

    datasets = {
        ds_id: DatasetState(
            id=ds_id,
            name=ds_id,
            file_path=None,
            test_mode="oscillation",
            x_data=np.array([1.0, 2.0]),
            y_data=np.array([3.0, 4.0]),
        )
        for ds_id in ("d1", "d2", "d3")
    }
    window.store._state = AppState(datasets=datasets, active_dataset_id="d1")

    converted_ids: list[str] = []

    def fake_rheodata_from_dataset_state(ds):
        converted_ids.append(ds.id)
        return ds

    with (
        patch.object(
            window.transform_page,
            "get_checked_dataset_ids",
            return_value=["d1", "d3"],
        ),
        patch(
            "rheojax.gui.utils.rheodata.rheodata_from_dataset_state",
            side_effect=fake_rheodata_from_dataset_state,
        ),
        patch.object(window, "_run_transform_sync"),
        patch.object(window, "worker_pool", None),
    ):
        window._on_apply_transform("mastercurve", params={})

    assert sorted(converted_ids) == ["d1", "d3"]
