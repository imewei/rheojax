import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QObject, Signal

from rheojax.gui.foundation.invalidation import (
    _CLEAR,
    _FIT_CASCADE,
    _TRANSFORM_CASCADE,
    _TRANSFORM_CLEAR,
    apply_cascade,
    invalidate_downstream,
    register_step,
)
from rheojax.gui.foundation.state import FitState, NlsqConfig, TransformState


def _fit_state(**overrides):
    defaults = dict(
        protocol="oscillation",
        model_key="maxwell",
        model_config={"n_modes": 3},
        data_ref="d1",
        column_map={"x": 0},
        control_vars={"gamma_dot": 1.0},
        nlsq_config=NlsqConfig(),
        nlsq_result=object(),
        nuts_result={"rhat": 1.0},
        step=4,
        revision=0,
    )
    defaults.update(overrides)
    return FitState(**defaults)


def test_apply_cascade_mutates_in_place_instead_of_returning_new_object():
    # Step-widget bodies hold a direct reference to the live FitState, so
    # apply_cascade must mutate it, not return a replacement the bodies
    # would never see.
    fit = _fit_state()
    identity = id(fit)
    apply_cascade(fit, _FIT_CASCADE, _CLEAR, "model_key")
    assert id(fit) == identity
    assert fit.nlsq_result is None
    assert fit.data_ref is None
    assert fit.revision == 1


def test_apply_cascade_matches_invalidate_downstream_field_values():
    # apply_cascade generalizes invalidate_downstream; both must agree on
    # which fields a given change clears.
    fit_a = _fit_state()
    fit_b = _fit_state()
    expected = invalidate_downstream(fit_a, changed="protocol")
    apply_cascade(fit_b, _FIT_CASCADE, _CLEAR, "protocol")
    assert fit_b.control_vars == expected.control_vars
    assert fit_b.model_config == expected.model_config
    assert fit_b.nlsq_result == expected.nlsq_result
    assert fit_b.revision == expected.revision


def test_apply_cascade_transform_key_clears_slots_config_result():
    tx = TransformState(
        transform_key="cox_merz",
        slots={"oscillation": "d1"},
        config={"n_points": 50},
        result={"output": "d2"},
        step=2,
        revision=0,
    )
    apply_cascade(tx, _TRANSFORM_CASCADE, _TRANSFORM_CLEAR, "transform_key")
    assert tx.slots == {}
    assert tx.config == {}
    assert tx.result is None
    assert tx.revision == 1


def test_apply_cascade_slots_clears_only_result():
    tx = TransformState(
        transform_key="cox_merz",
        slots={"oscillation": "d1"},
        config={"n_points": 50},
        result={"output": "d2"},
        step=2,
        revision=0,
    )
    apply_cascade(tx, _TRANSFORM_CASCADE, _TRANSFORM_CLEAR, "slots")
    assert tx.slots == {"oscillation": "d1"}  # untouched -- caller is setting this
    assert tx.config == {"n_points": 50}  # untouched
    assert tx.result is None  # cleared
    assert tx.revision == 1


def test_apply_cascade_bumps_revision_even_when_result_already_cleared():
    # Standardizes on invalidate_downstream's existing behavior: a cascade
    # call always bumps revision, even on a no-op field clear.
    tx = TransformState(
        transform_key="cox_merz", slots={"oscillation": "d1"}, config={}, result=None
    )
    apply_cascade(tx, _TRANSFORM_CASCADE, _TRANSFORM_CLEAR, "slots")
    assert tx.result is None
    assert tx.revision == 1


class _FakeStep(QObject):
    edited = Signal()


def test_register_step_runs_cascade_before_downstream():
    body = _FakeStep()
    tx = TransformState(
        transform_key="cox_merz", slots={}, config={}, result={"output": "d2"}
    )
    relocked = []
    seen_result_at_downstream_time = []

    register_step(
        body,
        "edited",
        lambda: relocked.append(True),
        changed="slots",
        live_state=tx,
        cascade_table=_TRANSFORM_CASCADE,
        clear_table=_TRANSFORM_CLEAR,
        downstream=[lambda: seen_result_at_downstream_time.append(tx.result)],
    )

    body.edited.emit()

    assert relocked == [True]
    # The downstream callback must observe the cascaded state (result
    # already cleared), not the pre-edit state -- this is the guarantee
    # register_step exists to enforce regardless of connection order.
    assert seen_result_at_downstream_time == [None]


def test_register_step_noop_when_signal_missing():
    class _NoSignals:
        pass

    # Must not raise even though _NoSignals has no "edited" attribute.
    register_step(_NoSignals(), "edited", lambda: None)
