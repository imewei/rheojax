import pytest

pytest.importorskip("PySide6")

import rheojax.transforms  # noqa: F401
from rheojax.gui.foundation.state import TransformState
from rheojax.gui.workspace.transform.step1_pick import TransformPickStep


def test_groups_and_select(qtbot):
    st = TransformState(); step = TransformPickStep(st); qtbot.addWidget(step)
    groups = step.groups()
    assert "cox_merz" in groups.get("analysis", [])
    assert "fft_analysis" in groups.get("spectral", [])
    with qtbot.waitSignal(step.edited, timeout=1000):
        step.set_transform("cox_merz")
    assert st.transform_key == "cox_merz" and step.is_ready() is True
