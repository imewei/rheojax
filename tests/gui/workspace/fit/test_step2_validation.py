from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.state import FitState
from rheojax.gui.workspace.fit.step2_data import DataStep, _validate_shape_and_values


class _RheoData:
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)


def _ref(i, protocol):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=protocol,
        origin="imported",
        units={},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_validate_shape_and_values_catches_mismatch_nan_nonmonotonic():
    assert _validate_shape_and_values(_RheoData([1, 2], [1, 2, 3])) == [
        "x/y length mismatch: 2 vs 3"
    ]
    assert _validate_shape_and_values(_RheoData([1, float("nan"), 3], [1, 2, 3])) == [
        "x contains NaN values"
    ]
    assert _validate_shape_and_values(_RheoData([1, 3, 2], [1, 2, 3])) == [
        "x is not monotonic"
    ]
    assert _validate_shape_and_values(_RheoData([1, 2, 3], [1, 2, 3])) == []
    assert _validate_shape_and_values(_RheoData([], [])) == ["dataset has no rows"]


def test_data_step_blocks_advance_on_invalid_data(qtbot):
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("bad", "oscillation"))
    lib.store_payload("bad", _RheoData([1, 3, 2], [1, 2, 3]))  # non-monotonic
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("bad")
    assert step.validation_errors() == ["x is not monotonic"]
    assert step.is_ready() is False


def test_data_step_ready_on_valid_data(qtbot):
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("good", "oscillation"))
    lib.store_payload("good", _RheoData([1, 2, 3], [1, 2, 3]))
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("good")
    assert step.validation_errors() == []
    assert step.is_ready() is True


def test_apply_unit_conversion_multiplies_x_by_2pi(qtbot):
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("hz_data", "oscillation"))
    rd = _RheoData([1.0, 2.0, 3.0], [1, 2, 3])
    lib.store_payload("hz_data", rd)
    lib._by_id["hz_data"] = DatasetRef(
        id="hz_data",
        name="hz_data",
        protocol_type="oscillation",
        origin="imported",
        units={"x": "Hz"},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("hz_data")
    assert step.needs_hz_conversion() is True
    assert step.unit_conversion_applied() is False

    step.apply_unit_conversion()

    converted = lib.load_payload("hz_data")
    assert np.allclose(converted.x, np.array([1.0, 2.0, 3.0]) * 2 * np.pi)
    assert step.unit_conversion_applied() is True
    assert step.needs_hz_conversion() is False  # already converted, guard clears


def test_apply_unit_conversion_is_idempotent(qtbot):
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("hz_data", "oscillation"))
    lib.store_payload("hz_data", _RheoData([1.0, 2.0, 3.0], [1, 2, 3]))
    lib._by_id["hz_data"] = DatasetRef(
        id="hz_data",
        name="hz_data",
        protocol_type="oscillation",
        origin="imported",
        units={"x": "Hz"},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("hz_data")

    step.apply_unit_conversion()
    once = np.asarray(lib.load_payload("hz_data").x).copy()

    step.apply_unit_conversion()  # calling again must not re-scale
    twice = np.asarray(lib.load_payload("hz_data").x)

    assert np.allclose(once, twice)


def test_convert_button_wired_to_apply_unit_conversion(qtbot):
    # Regression: apply_unit_conversion() was fully implemented but never
    # wired to any UI control -- grep for its only caller found itself, so
    # Hz-unit datasets never actually got converted in the running app. The
    # step must now expose a real Qt button that calls it.
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("hz_data", "oscillation"))
    lib.store_payload("hz_data", _RheoData([1.0, 2.0, 3.0], [1, 2, 3]))
    lib._by_id["hz_data"] = DatasetRef(
        id="hz_data",
        name="hz_data",
        protocol_type="oscillation",
        origin="imported",
        units={"x": "Hz"},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("hz_data")

    step._convert_btn.click()

    converted = lib.load_payload("hz_data")
    assert np.allclose(converted.x, np.array([1.0, 2.0, 3.0]) * 2 * np.pi)
    assert step.unit_conversion_applied() is True


def test_apply_unit_conversion_refuses_non_hz_data(qtbot):
    # apply_unit_conversion() is now reachable from a real button click, so
    # it must not blindly scale data whose units were never Hz (it used to
    # only guard on data_ref/already-converted, not on needs_hz_conversion()).
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("rad_data", "oscillation"))  # units={} -> not Hz
    rd = _RheoData([1.0, 2.0, 3.0], [1, 2, 3])
    lib.store_payload("rad_data", rd)
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("rad_data")

    step.apply_unit_conversion()

    assert np.allclose(lib.load_payload("rad_data").x, [1.0, 2.0, 3.0])
    assert step.unit_conversion_applied() is False


def test_apply_unit_conversion_persists_across_reselect(qtbot):
    # Regression: _unit_converted is a transient per-widget flag reset to
    # False at the top of every _on_select() call (including refresh()'s
    # still_valid branch, which re-runs _on_select(current) for the SAME
    # already-selected dataset). Before this fix, DatasetRef.units["x"] was
    # never updated by apply_unit_conversion(), so re-selecting the same
    # already-converted dataset made needs_hz_conversion() report True again
    # -- and, now that Convert is wired to a real button, a second click
    # would silently re-scale already-converted data by another x2pi.
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("hz_data", "oscillation"))
    lib.store_payload("hz_data", _RheoData([1.0, 2.0, 3.0], [1, 2, 3]))
    lib._by_id["hz_data"] = DatasetRef(
        id="hz_data",
        name="hz_data",
        protocol_type="oscillation",
        origin="imported",
        units={"x": "Hz"},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("hz_data")
    step.apply_unit_conversion()
    converted_once = np.asarray(lib.load_payload("hz_data").x).copy()

    # Simulate refresh()'s still_valid branch re-running _on_select() for
    # the same dataset (e.g. after an unrelated Step-1 model-only edit).
    step._on_select("hz_data")
    assert step.needs_hz_conversion() is False  # library units now say rad/s

    step._convert_btn.click()  # must be a no-op, not a second x2pi scaling
    assert np.allclose(np.asarray(lib.load_payload("hz_data").x), converted_once)


def test_data_step_not_ready_when_payload_missing_from_library(qtbot):
    # Regression: a DatasetRef with no stored payload (metadata-only entry,
    # or a derived dataset saved without a payload) used to be silently
    # swallowed by a bare `except KeyError: payload = None`, skipping
    # _validate_shape_and_values() entirely so is_ready() could report True
    # for a dataset with no actual data -- deferring the failure to a later
    # crash in NLSQ/NUTS instead of blocking advancement at Step 2.
    st = FitState(protocol="oscillation")
    lib = DatasetLibrary()
    lib.add(_ref("no_payload", "oscillation"))  # never store_payload()'d
    step = DataStep(st, lib)
    qtbot.addWidget(step)
    step.select_dataset("no_payload")
    assert step.validation_errors() != []
    assert step.is_ready() is False
