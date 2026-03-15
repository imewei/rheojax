"""Tests for the redesigned TransformPage."""

import os

import pytest

if not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

pytest.importorskip("PySide6")

from unittest.mock import MagicMock, patch

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def mock_store():
    """Mock StateStore for testing."""
    with patch("rheojax.gui.pages.transform_page.StateStore") as mock_cls:
        store = MagicMock()
        store.get_state.return_value = MagicMock(datasets={})
        store.get_active_dataset.return_value = None
        mock_cls.return_value = store
        yield store


@pytest.fixture
def page(qapp, mock_store):
    from rheojax.gui.pages.transform_page import TransformPage

    p = TransformPage()
    p._store = mock_store
    return p


def test_sidebar_populated_with_transforms(page):
    """Sidebar list has items for all registered transforms."""
    from rheojax.core.registry import TransformRegistry
    from rheojax.transforms import _ensure_all_registered

    _ensure_all_registered()
    expected = len(TransformRegistry.list_transforms())
    assert page._sidebar.count() == expected


def test_selecting_transform_shows_params(page):
    """Clicking a sidebar item populates the parameter form."""
    page._sidebar.setCurrentRow(0)
    assert page._param_form is not None
    values = page._param_form.get_values()
    assert len(values) > 0


def test_get_selected_params_returns_dict(page):
    """get_selected_params() returns param dict for selected transform."""
    page._sidebar.setCurrentRow(0)  # FFT
    params = page.get_selected_params()
    assert isinstance(params, dict)
    assert len(params) > 0


def test_apply_emits_signal(page, mock_store):
    """Apply Transform button emits transform_applied signal."""
    mock_ds = MagicMock()
    mock_ds.id = "test-ds-1"
    mock_store.get_active_dataset.return_value = mock_ds

    page._sidebar.setCurrentRow(0)  # FFT
    signals_received = []
    page.transform_applied.connect(
        lambda name, ds_id: signals_received.append((name, ds_id))
    )
    page._apply_transform()
    assert len(signals_received) == 1
    assert signals_received[0][1] == "test-ds-1"


def test_get_available_transforms_returns_metadata(page):
    """get_available_transforms() delegates to service."""
    from rheojax.core.registry import TransformRegistry
    from rheojax.transforms import _ensure_all_registered

    _ensure_all_registered()
    expected = len(TransformRegistry.list_transforms())
    transforms = page.get_available_transforms()
    assert len(transforms) == expected
    assert all("key" in t for t in transforms)


def test_multi_dataset_transform_shows_checklist(page, mock_store):
    """Selecting Mastercurve shows dataset checklist."""
    ds1 = MagicMock()
    ds1.id = "ds1"
    ds1.name = "foam_0C"
    ds2 = MagicMock()
    ds2.id = "ds2"
    ds2.name = "foam_25C"
    mock_store.get_state.return_value.datasets = {"ds1": ds1, "ds2": ds2}

    # Find and select Mastercurve
    for i in range(page._sidebar.count()):
        item = page._sidebar.item(i)
        if item.data(Qt.ItemDataRole.UserRole) == "mastercurve":
            page._sidebar.setCurrentRow(i)
            break

    assert page._dataset_checklist is not None
    assert page._dataset_checklist.count() == 2


def test_single_dataset_transform_no_checklist(page):
    """Selecting FFT does NOT show dataset checklist."""
    page._sidebar.setCurrentRow(0)  # FFT
    assert page._dataset_checklist is None


def test_empty_state_when_no_selection(page):
    """Empty state widget is visible before any transform is selected."""
    # Use isHidden() since the parent widget is not shown in tests;
    # isVisible() requires the full ancestor chain to be visible.
    assert not page._empty_state.isHidden()
    assert page._scroll.isHidden()
    assert page._apply_btn.isHidden()
