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
        id=i, name=i, protocol_type=protocol, origin="imported",
        units={}, row_count=3, hash="h", provenance={}, lineage=[],
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
