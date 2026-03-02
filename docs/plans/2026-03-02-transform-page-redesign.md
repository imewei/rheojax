# Transform Page Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the 988-line hardcoded transform page with a data-driven sidebar + ParameterFormBuilder + live PyQtGraph preview (~300 lines total).

**Architecture:** Sidebar list (QListWidget) drives a detail panel that auto-generates parameter forms from TransformService specs. A debounced preview engine computes transform results on a background thread and overlays Before/After on a single PyQtGraph canvas. Multi-dataset transforms show an inline dataset checklist.

**Tech Stack:** PySide6 (via `rheojax.gui.compat`), PyQtGraph, design tokens from `rheojax.gui.resources.styles.tokens`, StateStore singleton, TransformService.

---

## Task 1: Extend TransformService.get_transform_params() with full specs

**Files:**
- Modify: `rheojax/gui/services/transform_service.py:75-272`
- Test: `tests/gui/test_transform_service_params.py` (new)

The existing `get_transform_params()` returns param dicts with `type`, `default`, `description`, and optionally `choices`. We need to add `label` (human-readable) and `range` (for numeric types) so ParameterFormBuilder can fully generate the UI.

**Step 1: Write the failing test**

```python
# tests/gui/test_transform_service_params.py
"""Tests for TransformService parameter spec completeness."""

from rheojax.gui.services.transform_service import TransformService


def test_all_params_have_label_and_type():
    """Every param spec must have 'label' and 'type' keys."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            assert "type" in spec, f"{transform_key}.{param_name} missing 'type'"
            assert "label" in spec, f"{transform_key}.{param_name} missing 'label'"
            assert "default" in spec, f"{transform_key}.{param_name} missing 'default'"


def test_numeric_params_have_range():
    """Float and int params must have a 'range' tuple."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            if spec["type"] in ("float", "int"):
                assert "range" in spec, (
                    f"{transform_key}.{param_name} (type={spec['type']}) missing 'range'"
                )
                assert len(spec["range"]) == 2, (
                    f"{transform_key}.{param_name} 'range' must be (min, max) tuple"
                )


def test_choice_params_have_choices():
    """Choice params must have a 'choices' list."""
    service = TransformService()
    for transform_key in service.get_available_transforms():
        params = service.get_transform_params(transform_key)
        for param_name, spec in params.items():
            if spec["type"] == "choice":
                assert "choices" in spec, (
                    f"{transform_key}.{param_name} missing 'choices'"
                )
                assert len(spec["choices"]) >= 2, (
                    f"{transform_key}.{param_name} must have at least 2 choices"
                )
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_transform_service_params.py -v`
Expected: FAIL — `label` and `range` keys are missing from current specs.

**Step 3: Add `label` and `range` to all param specs**

In `transform_service.py:75-272`, update the `params_map` dict. Add `"label"` to every spec and `"range"` to every float/int spec. The values come from what the current TransformPage hardcodes in its if/elif chain:

```python
# Example — mastercurve section becomes:
"mastercurve": {
    "reference_temp": {
        "type": "float",
        "default": 25.0,
        "range": (-100.0, 300.0),
        "label": "Reference Temperature (\u00b0C)",
        "description": "Reference temperature for TTS",
    },
    "auto_shift": {
        "type": "bool",
        "default": True,
        "label": "Auto-detect shift factors",
        "description": "Automatically calculate shift factors",
    },
    "shift_method": {
        "type": "choice",
        "default": "wlf",
        "choices": ["wlf", "arrhenius", "manual"],
        "label": "Shift Method",
        "description": "Method for calculating shift factors",
    },
},
```

Apply the same pattern to all 7 transforms. Use the ranges from the current TransformPage's `QDoubleSpinBox.setRange()` calls:

| Transform | Param | Range |
|-----------|-------|-------|
| mastercurve | reference_temp | (-100, 300) |
| srfs | reference_gamma_dot | (0.001, 1000) |
| fft | — | no numeric params |
| owchirp | min_frequency | (0.0001, 1e6) |
| owchirp | max_frequency | (0.0001, 1e6) |
| owchirp | n_frequencies | (4, 5000) |
| owchirp | wavelet_width | (1.0, 20.0) |
| owchirp | max_harmonic | (1, 99) |
| derivative | order | (1, 4) |
| derivative | window_length | (3, 201) |
| derivative | poly_order | (1, 10) |
| mutation_number | — | no numeric params |
| spp | omega | (0.001, 1000) |
| spp | gamma_0 | (0.0001, 100) |
| spp | n_harmonics | (1, 99) |
| spp | yield_tolerance | (0.0001, 1.0) |
| spp | start_cycle | (0, 100) |

Also add `shift_method` to mastercurve (currently in TransformPage but NOT in TransformService — this is the drift issue).
Also add `mode` (padding) to derivative (same drift issue).

**Step 4: Run tests to verify they pass**

Run: `pytest tests/gui/test_transform_service_params.py -v`
Expected: 3/3 PASS

**Step 5: Commit**

```bash
git add rheojax/gui/services/transform_service.py tests/gui/test_transform_service_params.py
git commit -m "feat(gui): extend TransformService param specs with label, range for UI generation"
```

---

## Task 2: Add get_transform_metadata() to TransformService

**Files:**
- Modify: `rheojax/gui/services/transform_service.py`
- Test: `tests/gui/test_transform_service_params.py` (extend)

The TransformPage currently has a separate `get_available_transforms()` method returning display metadata (name, key, description, color, requires_multiple). This should live in TransformService so the page can be fully data-driven.

**Step 1: Write the failing test**

```python
# Append to tests/gui/test_transform_service_params.py

def test_get_transform_metadata_returns_all_transforms():
    """get_transform_metadata() returns metadata for all registered transforms."""
    service = TransformService()
    metadata = service.get_transform_metadata()
    keys = service.get_available_transforms()
    assert len(metadata) == len(keys)
    for entry in metadata:
        assert "key" in entry
        assert "name" in entry
        assert "description" in entry
        assert "requires_multiple" in entry
        assert isinstance(entry["requires_multiple"], bool)


def test_multi_dataset_transforms_flagged():
    """Mastercurve and SRFS must be flagged as requires_multiple."""
    service = TransformService()
    metadata = service.get_transform_metadata()
    multi = {m["key"] for m in metadata if m["requires_multiple"]}
    assert "mastercurve" in multi
    assert "srfs" in multi
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gui/test_transform_service_params.py::test_get_transform_metadata_returns_all_transforms -v`
Expected: FAIL — `AttributeError: 'TransformService' object has no attribute 'get_transform_metadata'`

**Step 3: Implement get_transform_metadata()**

Add to `TransformService` after `get_transform_params()`:

```python
def get_transform_metadata(self) -> list[dict[str, Any]]:
    """Return display metadata for all transforms.

    Returns
    -------
    list[dict]
        Each dict has: key, name, description, requires_multiple
    """
    meta = [
        {
            "key": "fft",
            "name": "FFT",
            "description": "Fast Fourier Transform for frequency analysis",
            "requires_multiple": False,
        },
        {
            "key": "mastercurve",
            "name": "Mastercurve",
            "description": "Time-temperature superposition",
            "requires_multiple": True,
        },
        {
            "key": "srfs",
            "name": "SRFS",
            "description": "Strain-rate frequency superposition",
            "requires_multiple": True,
        },
        {
            "key": "mutation_number",
            "name": "Mutation Number",
            "description": "Calculate mutation number",
            "requires_multiple": False,
        },
        {
            "key": "owchirp",
            "name": "OW Chirp",
            "description": "Optimally-windowed chirp analysis",
            "requires_multiple": False,
        },
        {
            "key": "spp",
            "name": "SPP Analysis",
            "description": "LAOS yield stress and cage modulus extraction",
            "requires_multiple": False,
        },
        {
            "key": "derivative",
            "name": "Derivatives",
            "description": "Calculate numerical derivatives",
            "requires_multiple": False,
        },
    ]
    return meta
```

**Step 4: Run tests**

Run: `pytest tests/gui/test_transform_service_params.py -v`
Expected: 5/5 PASS

**Step 5: Commit**

```bash
git add rheojax/gui/services/transform_service.py tests/gui/test_transform_service_params.py
git commit -m "feat(gui): add TransformService.get_transform_metadata() for data-driven UI"
```

---

## Task 3: Create ParameterFormBuilder widget

**Files:**
- Create: `rheojax/gui/widgets/parameter_form.py`
- Modify: `rheojax/gui/widgets/__init__.py` (add export)
- Test: `tests/gui/test_parameter_form.py` (new)

This is the core new reusable widget that replaces the 534-line if/elif chain.

**Step 1: Write the failing test**

```python
# tests/gui/test_parameter_form.py
"""Tests for ParameterFormBuilder widget."""

import pytest

pytest.importorskip("PySide6")

from rheojax.gui.widgets.parameter_form import ParameterFormBuilder


@pytest.fixture
def sample_specs():
    return {
        "temperature": {
            "type": "float",
            "default": 25.0,
            "range": (-100.0, 300.0),
            "label": "Temperature (\u00b0C)",
            "description": "Reference temperature",
        },
        "auto_detect": {
            "type": "bool",
            "default": True,
            "label": "Auto-detect",
            "description": "Enable auto-detection",
        },
        "method": {
            "type": "choice",
            "default": "wlf",
            "choices": ["wlf", "arrhenius", "manual"],
            "label": "Method",
            "description": "Calculation method",
        },
        "order": {
            "type": "int",
            "default": 1,
            "range": (1, 4),
            "label": "Order",
            "description": "Derivative order",
        },
    }


def test_form_creates_from_specs(qapp, sample_specs):
    """ParameterFormBuilder creates widgets from param specs."""
    form = ParameterFormBuilder(sample_specs)
    values = form.get_values()
    assert values["temperature"] == 25.0
    assert values["auto_detect"] is True
    assert values["method"] == "wlf"
    assert values["order"] == 1


def test_form_emits_values_changed(qapp, sample_specs, qtbot):
    """values_changed signal fires on parameter modification."""
    form = ParameterFormBuilder(sample_specs)
    with qtbot.waitSignal(form.values_changed, timeout=1000):
        # Programmatically change the float spinbox
        form._widgets["temperature"].setValue(50.0)
    assert form.get_values()["temperature"] == 50.0


def test_form_handles_empty_specs(qapp):
    """Empty spec dict produces empty form."""
    form = ParameterFormBuilder({})
    assert form.get_values() == {}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/gui/test_parameter_form.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rheojax.gui.widgets.parameter_form'`

**Step 3: Implement ParameterFormBuilder**

```python
# rheojax/gui/widgets/parameter_form.py
"""Auto-generated parameter forms from spec dictionaries.

Given a dict of parameter specs (type, default, range, label, choices),
builds a QFormLayout with the appropriate widget per type and emits
values_changed on any modification.
"""

from __future__ import annotations

import math
from typing import Any

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles.tokens import Spacing, Typography
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ParameterFormBuilder(QWidget):
    """Dynamically build a parameter form from spec dicts.

    Parameters
    ----------
    specs : dict[str, dict]
        Mapping of param_name -> spec. Each spec has:
        - type: "float" | "int" | "bool" | "choice"
        - default: default value
        - label: human-readable label
        - range: (min, max) for float/int (optional)
        - choices: list[str] for choice type
        - description: tooltip text (optional)
    parent : QWidget, optional
        Parent widget.

    Signals
    -------
    values_changed()
        Emitted when any parameter value changes.
    """

    values_changed = Signal()

    def __init__(
        self, specs: dict[str, dict[str, Any]], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._specs = specs
        self._widgets: dict[str, QWidget] = {}
        self._build_form()

    def _build_form(self) -> None:
        layout = QFormLayout(self)
        layout.setSpacing(Spacing.SM)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        for name, spec in self._specs.items():
            ptype = spec["type"]
            label = spec.get("label", name)
            tooltip = spec.get("description", "")

            if ptype == "float":
                widget = QDoubleSpinBox()
                lo, hi = spec.get("range", (0.0, 1e6))
                widget.setRange(lo, hi)
                # Adaptive decimals: 4 for small ranges, 2 for large
                decimals = 4 if (hi - lo) < 10 else 2
                widget.setDecimals(decimals)
                # Step size: ~1% of range or based on magnitude
                step = 10 ** (math.floor(math.log10(max(abs(hi - lo), 1e-10))) - 2)
                widget.setSingleStep(step)
                widget.setValue(spec["default"])
                widget.valueChanged.connect(self._on_change)
            elif ptype == "int":
                widget = QSpinBox()
                lo, hi = spec.get("range", (0, 1000))
                widget.setRange(int(lo), int(hi))
                widget.setValue(int(spec["default"]))
                widget.valueChanged.connect(self._on_change)
            elif ptype == "bool":
                widget = QCheckBox()
                widget.setChecked(bool(spec["default"]))
                widget.stateChanged.connect(self._on_change)
            elif ptype == "choice":
                widget = QComboBox()
                widget.addItems(spec.get("choices", []))
                widget.setCurrentText(str(spec["default"]))
                widget.currentTextChanged.connect(self._on_change)
            else:
                logger.warning("Unknown param type", param=name, type=ptype)
                continue

            widget.setToolTip(tooltip)
            widget.setStyleSheet(f"font-size: {Typography.SIZE_MD_SM}pt;")
            self._widgets[name] = widget
            layout.addRow(label + ":", widget)

    def _on_change(self, *_args: object) -> None:
        self.values_changed.emit()

    def get_values(self) -> dict[str, Any]:
        """Read current values from all widgets.

        Returns
        -------
        dict[str, Any]
            Mapping of param_name -> current value.
        """
        values: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
        return values


__all__ = ["ParameterFormBuilder"]
```

**Step 4: Add to widgets `__init__.py`**

In `rheojax/gui/widgets/__init__.py`, add:

```python
from rheojax.gui.widgets.parameter_form import ParameterFormBuilder
```

And add `"ParameterFormBuilder"` to the `__all__` list.

**Step 5: Run tests**

Run: `pytest tests/gui/test_parameter_form.py -v`
Expected: 3/3 PASS

**Step 6: Commit**

```bash
git add rheojax/gui/widgets/parameter_form.py rheojax/gui/widgets/__init__.py tests/gui/test_parameter_form.py
git commit -m "feat(gui): add ParameterFormBuilder widget for data-driven parameter forms"
```

---

## Task 4: Wire TransformService.preview_transform() to compute actual results

**Files:**
- Modify: `rheojax/gui/services/transform_service.py:615-674`
- Test: `tests/gui/test_transform_service_params.py` (extend)

The current `preview_transform()` returns metadata only (shapes, ranges). We need it to compute the actual transform and return plot data.

**Step 1: Write the failing test**

```python
# Append to tests/gui/test_transform_service_params.py

import numpy as np

def test_preview_returns_plot_data():
    """preview_transform returns x/y arrays for Before and After."""
    from rheojax.core.data import RheoData

    service = TransformService()
    # Synthetic time-domain data
    x = np.linspace(0, 10, 200)
    y = np.sin(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
    data = RheoData(x=x, y=y)

    result = service.preview_transform(
        "derivative", data, {"order": 1, "window_length": 11, "poly_order": 3,
                             "method": "savgol", "validate_window": True,
                             "smooth_before": False, "smooth_after": False}
    )
    assert "x_before" in result
    assert "y_before" in result
    assert "x_after" in result
    assert "y_after" in result
    assert len(result["x_after"]) > 0


def test_preview_returns_error_on_failure():
    """preview_transform returns error dict on failure, not exception."""
    from rheojax.core.data import RheoData

    service = TransformService()
    # Data too short for SPP
    data = RheoData(x=np.array([1.0, 2.0]), y=np.array([1.0, 2.0]))
    result = service.preview_transform("spp", data, {"omega": 1.0, "gamma_0": 1.0})
    assert "error" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_transform_service_params.py::test_preview_returns_plot_data -v`
Expected: FAIL — `x_before` not in result dict.

**Step 3: Rewrite preview_transform()**

Replace the body of `preview_transform()` in `transform_service.py`:

```python
def preview_transform(
    self,
    name: str,
    data: RheoData | list[RheoData],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Compute transform and return plot data for preview.

    Returns
    -------
    dict
        On success: x_before, y_before, x_after, y_after (numpy arrays)
        On failure: error (str)
    """
    logger.debug("Computing preview", transform=name, params=params)
    try:
        # Capture before data
        if isinstance(data, list):
            # Multi-dataset: use first dataset as "before" representative
            x_before = np.asarray(data[0].x)
            y_before = np.asarray(data[0].y)
        else:
            x_before = np.asarray(data.x)
            y_before = np.asarray(data.y)

        # Compute actual transform
        result = self.apply_transform(name, data, params)

        # Unpack result (some transforms return (RheoData, metadata_dict))
        if isinstance(result, tuple):
            result_data = result[0]
        else:
            result_data = result

        x_after = np.asarray(result_data.x)
        y_after = np.asarray(result_data.y)

        return {
            "x_before": x_before,
            "y_before": y_before,
            "x_after": x_after,
            "y_after": y_after,
        }
    except Exception as e:
        logger.warning("Preview failed", transform=name, error=str(e))
        return {"error": str(e)}
```

**Step 4: Run tests**

Run: `pytest tests/gui/test_transform_service_params.py -v`
Expected: 7/7 PASS

**Step 5: Commit**

```bash
git add rheojax/gui/services/transform_service.py tests/gui/test_transform_service_params.py
git commit -m "feat(gui): wire preview_transform() to compute actual transform results"
```

---

## Task 5: Rewrite TransformPage — sidebar + detail panel

**Files:**
- Rewrite: `rheojax/gui/pages/transform_page.py` (988 lines → ~300 lines)
- Test: `tests/gui/test_transform_page_redesign.py` (new)

This is the main task. The entire file is rewritten.

**Step 1: Write the failing tests**

```python
# tests/gui/test_transform_page_redesign.py
"""Tests for the redesigned TransformPage."""

import pytest

pytest.importorskip("PySide6")

import numpy as np
from unittest.mock import MagicMock, patch

from rheojax.gui.pages.transform_page import TransformPage


@pytest.fixture
def page(qapp):
    """Create TransformPage with mocked StateStore."""
    with patch("rheojax.gui.pages.transform_page.StateStore") as mock_store_cls:
        mock_store = MagicMock()
        mock_store.get_state.return_value = MagicMock(datasets={})
        mock_store.get_active_dataset.return_value = None
        mock_store_cls.return_value = mock_store
        p = TransformPage()
        p._store = mock_store
        yield p


def test_sidebar_populated_with_transforms(page):
    """Sidebar list has items for all 7 transforms."""
    assert page._sidebar.count() == 7


def test_selecting_transform_shows_params(page, qtbot):
    """Clicking a sidebar item populates the parameter form."""
    page._sidebar.setCurrentRow(0)  # FFT
    # The form should now have widgets
    assert page._param_form is not None
    values = page._param_form.get_values()
    assert len(values) > 0


def test_empty_state_when_no_selection(page):
    """Detail panel shows empty state before any transform is selected."""
    assert page._sidebar.currentRow() == -1 or page._param_form is None


def test_get_selected_params_returns_dict(page):
    """get_selected_params() returns param dict for selected transform."""
    page._sidebar.setCurrentRow(0)  # FFT
    params = page.get_selected_params()
    assert isinstance(params, dict)
    assert "direction" in params  # FFT has direction param


def test_apply_emits_signal(page, qtbot):
    """Apply Transform button emits transform_applied signal."""
    # Mock active dataset
    mock_ds = MagicMock()
    mock_ds.id = "test-ds-1"
    page._store.get_active_dataset.return_value = mock_ds

    page._sidebar.setCurrentRow(0)  # FFT
    with qtbot.waitSignal(page.transform_applied, timeout=1000):
        page._apply_transform()


def test_multi_dataset_transform_shows_checklist(page):
    """Selecting Mastercurve shows dataset checklist."""
    # Add some datasets to mock state
    ds1 = MagicMock()
    ds1.id = "ds1"
    ds1.name = "foam_0C"
    ds2 = MagicMock()
    ds2.id = "ds2"
    ds2.name = "foam_25C"
    page._store.get_state.return_value.datasets = {"ds1": ds1, "ds2": ds2}

    # Find and select Mastercurve (index 1 based on metadata order)
    for i in range(page._sidebar.count()):
        if page._sidebar.item(i).data(256) == "mastercurve":  # Qt.UserRole = 256
            page._sidebar.setCurrentRow(i)
            break

    assert page._dataset_checklist is not None
    assert page._dataset_checklist.count() == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/gui/test_transform_page_redesign.py -v`
Expected: FAIL — old TransformPage doesn't have `_sidebar`, `_param_form`, etc.

**Step 3: Rewrite transform_page.py**

Replace the entire file with the new implementation. Key structure:

```python
# rheojax/gui/pages/transform_page.py
"""
Transform Page
=============

Data-driven transform interface with sidebar navigation,
auto-generated parameter forms, and live PyQtGraph preview.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rheojax.gui.compat import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRunnable,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    Qt,
    QThreadPool,
    QTimer,
    QVBoxLayout,
    QWidget,
    Signal,
    Slot,
)
from rheojax.gui.resources.styles.tokens import (
    ColorPalette,
    Spacing,
    Typography,
    button_style,
    section_header_style,
)
from rheojax.gui.services.transform_service import TransformService
from rheojax.gui.state.store import StateStore
from rheojax.gui.widgets.empty_state import EmptyStateWidget
from rheojax.gui.widgets.parameter_form import ParameterFormBuilder
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Lazy import to handle missing pyqtgraph gracefully
try:
    from rheojax.gui.widgets.pyqtgraph_canvas import PyQtGraphCanvas, PYQTGRAPH_AVAILABLE
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class _PreviewWorker(QRunnable):
    """Compute transform preview off the main thread."""

    def __init__(
        self,
        service: TransformService,
        name: str,
        data: Any,
        params: dict[str, Any],
        callback: Any,
        generation: int,
    ) -> None:
        super().__init__()
        self._service = service
        self._name = name
        self._data = data
        self._params = params
        self._callback = callback
        self._generation = generation
        self.setAutoDelete(True)

    def run(self) -> None:
        result = self._service.preview_transform(self._name, self._data, self._params)
        result["_generation"] = self._generation
        self._callback(result)


class TransformPage(QWidget):
    """Data-driven transform page with sidebar + live preview."""

    transform_selected = Signal(str)
    transform_applied = Signal(str, str)  # transform_name, dataset_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        logger.debug("Initializing", class_name="TransformPage")
        self._store = StateStore()
        self._service = TransformService()
        self._selected_key: str | None = None
        self._param_form: ParameterFormBuilder | None = None
        self._dataset_checklist: QListWidget | None = None
        self._preview_generation = 0
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(500)
        self._debounce_timer.timeout.connect(self._compute_preview)
        self._setup_ui()
        logger.debug("Initialization complete", class_name="TransformPage")

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Left: Sidebar ---
        sidebar_widget = QWidget()
        sidebar_widget.setFixedWidth(200)
        sidebar_widget.setStyleSheet(
            f"background-color: {ColorPalette.BG_SURFACE};"
            f" border-right: 1px solid {ColorPalette.BORDER_DEFAULT};"
        )
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(
            Spacing.SM, Spacing.MD, Spacing.SM, Spacing.SM
        )

        header = QLabel("Transforms")
        header.setStyleSheet(section_header_style())
        sidebar_layout.addWidget(header)

        self._sidebar = QListWidget()
        self._sidebar.setStyleSheet(f"""
            QListWidget {{
                border: none;
                background: transparent;
                font-size: {Typography.SIZE_MD_SM}pt;
            }}
            QListWidget::item {{
                padding: {Spacing.SM}px {Spacing.MD}px;
                border-radius: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {ColorPalette.PRIMARY};
                color: {ColorPalette.TEXT_INVERSE};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {ColorPalette.BG_HOVER};
            }}
        """)
        self._sidebar.currentRowChanged.connect(self._on_sidebar_changed)

        # Populate from service
        for meta in self._service.get_transform_metadata():
            item = QListWidgetItem(meta["name"])
            item.setData(Qt.UserRole, meta["key"])
            item.setToolTip(meta["description"])
            self._sidebar.addItem(item)

        sidebar_layout.addWidget(self._sidebar)

        # Contextual hint area
        self._hint_label = QLabel("")
        self._hint_label.setWordWrap(True)
        self._hint_label.setStyleSheet(
            f"color: {ColorPalette.TEXT_MUTED};"
            f" font-size: {Typography.SIZE_SM}pt;"
            f" padding: {Spacing.SM}px;"
        )
        sidebar_layout.addWidget(self._hint_label)

        layout.addWidget(sidebar_widget)

        # --- Right: Detail Panel ---
        self._detail_area = QWidget()
        detail_layout = QVBoxLayout(self._detail_area)
        detail_layout.setContentsMargins(
            Spacing.LG, Spacing.LG, Spacing.LG, Spacing.LG
        )
        detail_layout.setSpacing(Spacing.MD)

        # Header (transform name + description)
        self._detail_header = QLabel("")
        self._detail_header.setStyleSheet(
            f"font-size: {Typography.SIZE_XL}pt;"
            f" font-weight: {Typography.WEIGHT_BOLD};"
        )
        detail_layout.addWidget(self._detail_header)

        self._detail_desc = QLabel("")
        self._detail_desc.setStyleSheet(
            f"color: {ColorPalette.TEXT_SECONDARY};"
            f" font-size: {Typography.SIZE_MD_SM}pt;"
        )
        detail_layout.addWidget(self._detail_desc)

        # Scrollable content area for params + datasets + preview
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.NoFrame)
        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setSpacing(Spacing.MD)
        scroll.setWidget(self._content_widget)
        detail_layout.addWidget(scroll, 1)

        # Apply button
        self._btn_apply = QPushButton("Apply Transform")
        self._btn_apply.setStyleSheet(button_style("success", "lg"))
        self._btn_apply.setEnabled(False)
        self._btn_apply.clicked.connect(self._apply_transform)
        detail_layout.addWidget(self._btn_apply, alignment=Qt.AlignRight)

        # Empty state (shown initially)
        self._empty_state = EmptyStateWidget(
            "Select a transform",
            "Choose a transform from the list to configure and preview it.",
        )
        detail_layout.addWidget(self._empty_state)

        layout.addWidget(self._detail_area, 1)

    @Slot(int)
    def _on_sidebar_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._sidebar.item(row)
        key = item.data(Qt.UserRole)
        self._select_transform(key)

    def _select_transform(self, key: str) -> None:
        self._selected_key = key
        meta_list = self._service.get_transform_metadata()
        meta = next((m for m in meta_list if m["key"] == key), None)
        if not meta:
            return

        logger.debug("Transform selected", transform=key)
        self.transform_selected.emit(meta["name"])

        # Update header
        self._detail_header.setText(meta["name"])
        self._detail_header.show()
        self._detail_desc.setText(meta["description"])
        self._detail_desc.show()
        self._empty_state.hide()
        self._btn_apply.setEnabled(True)

        # Update hint
        if meta["requires_multiple"]:
            self._hint_label.setText("Requires 2+ datasets")
        else:
            self._hint_label.setText("")

        # Clear content area
        while self._content_layout.count():
            child = self._content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Build parameter form from service specs
        specs = self._service.get_transform_params(key)
        self._param_form = ParameterFormBuilder(specs)
        self._param_form.values_changed.connect(self._on_params_changed)
        self._content_layout.addWidget(self._param_form)

        # Dataset checklist for multi-dataset transforms
        self._dataset_checklist = None
        if meta["requires_multiple"]:
            ds_label = QLabel("Datasets (select 2+):")
            ds_label.setStyleSheet(
                f"font-weight: {Typography.WEIGHT_SEMIBOLD};"
                f" font-size: {Typography.SIZE_MD_SM}pt;"
                f" margin-top: {Spacing.SM}px;"
            )
            self._content_layout.addWidget(ds_label)

            self._dataset_checklist = QListWidget()
            self._dataset_checklist.setMaximumHeight(150)
            datasets = self._store.get_state().datasets
            for ds_id, ds in datasets.items():
                item = QListWidgetItem(ds.name)
                item.setData(Qt.UserRole, ds_id)
                item.setCheckState(Qt.Unchecked)
                self._dataset_checklist.addItem(item)
            self._content_layout.addWidget(self._dataset_checklist)

        # Preview canvas
        preview_label = QLabel("Preview")
        preview_label.setStyleSheet(
            f"font-weight: {Typography.WEIGHT_SEMIBOLD};"
            f" font-size: {Typography.SIZE_MD_SM}pt;"
            f" margin-top: {Spacing.SM}px;"
        )
        self._content_layout.addWidget(preview_label)

        if PYQTGRAPH_AVAILABLE:
            self._preview_canvas = PyQtGraphCanvas()
        else:
            # Fallback to matplotlib
            from rheojax.gui.widgets.plot_canvas import PlotCanvas
            self._preview_canvas = PlotCanvas()
        self._preview_canvas.setMinimumHeight(250)
        self._content_layout.addWidget(self._preview_canvas)

        self._content_layout.addStretch()

        # Trigger initial preview
        self._on_params_changed()

    def _on_params_changed(self) -> None:
        """Debounce param changes before computing preview."""
        self._debounce_timer.start()

    def _compute_preview(self) -> None:
        """Compute preview on background thread."""
        if not self._selected_key or not self._param_form:
            return

        dataset = self._store.get_active_dataset()
        if dataset is None:
            return

        # For multi-dataset transforms, gather checked datasets
        if self._dataset_checklist is not None:
            # Multi-dataset: build RheoData list
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state
            data_list = []
            for i in range(self._dataset_checklist.count()):
                item = self._dataset_checklist.item(i)
                if item.checkState() == Qt.Checked:
                    ds_id = item.data(Qt.UserRole)
                    ds = self._store.get_state().datasets.get(ds_id)
                    if ds:
                        try:
                            data_list.append(rheodata_from_dataset_state(ds))
                        except Exception:
                            pass
            if len(data_list) < 2:
                return
            data = data_list
        else:
            # Single dataset
            from rheojax.gui.utils.rheodata import rheodata_from_dataset_state
            try:
                data = rheodata_from_dataset_state(dataset)
            except Exception:
                return

        params = self._param_form.get_values()
        self._preview_generation += 1
        gen = self._preview_generation

        worker = _PreviewWorker(
            self._service,
            self._selected_key,
            data,
            params,
            lambda result: self._on_preview_ready(result),
            gen,
        )
        QThreadPool.globalInstance().start(worker)

    @Slot(dict)
    def _on_preview_ready(self, result: dict[str, Any]) -> None:
        """Update preview plot with computed results (called from worker thread)."""
        # Discard stale results
        if result.get("_generation", 0) != self._preview_generation:
            return

        if "error" in result:
            logger.debug("Preview error", error=result["error"])
            return

        canvas = getattr(self, "_preview_canvas", None)
        if canvas is None:
            return

        canvas.clear()

        x_before = result["x_before"]
        y_before = result["y_before"]
        x_after = result["x_after"]
        y_after = result["y_after"]

        if PYQTGRAPH_AVAILABLE and isinstance(canvas, PyQtGraphCanvas):
            canvas.plot_data(
                x_before, y_before, name="Before",
                color=ColorPalette.CHART_1, line_width=2,
            )
            canvas.plot_data(
                x_after, y_after, name="After",
                color=ColorPalette.SUCCESS, line_width=2,
            )
        else:
            # Matplotlib fallback
            canvas.plot(
                x_before, y_before, label="Before",
                xlabel="x", ylabel="y", title="Preview",
            )

    def _apply_transform(self) -> None:
        if not self._selected_key:
            return

        # Find display name from key
        meta_list = self._service.get_transform_metadata()
        meta = next((m for m in meta_list if m["key"] == self._selected_key), None)
        display_name = meta["name"] if meta else self._selected_key

        dataset = self._store.get_active_dataset()
        if dataset:
            self.transform_applied.emit(display_name, dataset.id)
            logger.info("Transform applied", transform=display_name, dataset_id=dataset.id)
        else:
            logger.debug("No active dataset for transform")

    def get_selected_params(self) -> dict[str, Any]:
        """Return current parameter values for the selected transform."""
        if self._param_form is None:
            return {}
        values = self._param_form.get_values()
        # Enforce Savitzky-Golay odd window if present
        if "validate_window" in values and values.get("validate_window", False):
            wl = int(values.get("window_length", 11))
            if wl % 2 == 0:
                values["window_length"] = wl + 1
        return values

    def get_available_transforms(self) -> list[dict[str, Any]]:
        """Return list of available transforms with metadata."""
        return self._service.get_transform_metadata()
```

**Step 4: Run tests**

Run: `pytest tests/gui/test_transform_page_redesign.py -v`
Expected: 6/6 PASS

**Step 5: Run existing transform tests to verify no regressions**

Run: `pytest tests/gui/test_transform_presets.py -v`
Expected: PASS (these test TransformService directly, not TransformPage UI)

**Step 6: Commit**

```bash
git add rheojax/gui/pages/transform_page.py tests/gui/test_transform_page_redesign.py
git commit -m "feat(gui): rewrite TransformPage with sidebar, ParameterFormBuilder, live preview

Replace 988-line hardcoded transform page with data-driven implementation:
- Sidebar QListWidget populated from TransformService.get_transform_metadata()
- ParameterFormBuilder auto-generates controls from param specs
- Debounced live preview via QThreadPool + PyQtGraph overlay
- Inline dataset checklist for multi-dataset transforms (Mastercurve, SRFS)
- ~300 lines (down from 988)"
```

---

## Task 6: Update MainWindow name mapping

**Files:**
- Modify: `rheojax/gui/app/main_window.py:2033-2050`

The old TransformPage emitted display names ("FFT", "Mastercurve") while MainWindow had a `name_map` to convert to service keys. The new TransformPage still emits display names for backward compatibility. Verify the mapping still works.

**Step 1: Check the existing mapping matches**

The new `_apply_transform()` emits `meta["name"]` which is the same display name as before (e.g., "FFT", "Mastercurve"). The MainWindow `name_map` at line 2037-2044 converts these to service keys. This should still work unchanged.

**Step 2: Verify by running the existing GUI smoke test**

Run: `pytest tests/gui/test_gui_smoke.py -v -k transform`
Expected: PASS

**Step 3: Commit (if any changes needed)**

If the smoke test passes with no changes, no commit needed. If the name mapping needs adjustment, fix and commit:

```bash
git add rheojax/gui/app/main_window.py
git commit -m "fix(gui): update MainWindow transform name mapping for redesigned page"
```

---

## Task 7: Run full test suite and fix regressions

**Files:**
- Any files with test failures

**Step 1: Run GUI tests**

Run: `pytest tests/gui/ -v --timeout=60`
Expected: All pass. If any fail, fix the specific regression.

**Step 2: Run smoke tests**

Run: `pytest -m smoke -v --timeout=120`
Expected: All pass.

**Step 3: Final commit if needed**

```bash
git commit -m "fix(gui): resolve test regressions from transform page redesign"
```

---

## Summary

| Task | What | LOC Change | Key Files |
|------|------|-----------|-----------|
| 1 | Extend param specs (label, range) | +~100 | transform_service.py |
| 2 | Add get_transform_metadata() | +~40 | transform_service.py |
| 3 | Create ParameterFormBuilder | +~100 new | parameter_form.py |
| 4 | Wire preview_transform() | +~20 net | transform_service.py |
| 5 | Rewrite TransformPage | -688 net | transform_page.py |
| 6 | Verify MainWindow wiring | ~0 | main_window.py |
| 7 | Regression fixes | TBD | Various |

**Net change:** ~-500 lines (988 → ~300 page + ~100 widget)
