"""Regression test: invalid parameter values/bounds must not emit signals.

Guards against invalid scientific inputs (NaN, inf, out-of-range, inverted
bounds) reaching FitPage's state-store handlers, which apply them unchecked.
"""

from rheojax.gui.foundation.state import ParameterState
from rheojax.gui.widgets.parameter_table import ParameterTable


def _make_table(qapp) -> ParameterTable:
    table = ParameterTable()
    table.set_parameters(
        {"G0": ParameterState(name="G0", value=1.0, min_bound=0.0, max_bound=10.0)}
    )
    return table


def test_invalid_value_is_not_emitted(qapp):
    table = _make_table(qapp)
    emitted = []
    table.parameter_changed.connect(lambda name, value: emitted.append((name, value)))

    for bad_text in ("nan", "inf", "1e400", "-5"):  # -5 is out of [0, 10] bounds
        table.item(0, 1).setText(bad_text)

    assert emitted == []


def test_invalid_bounds_are_not_emitted(qapp):
    table = _make_table(qapp)
    emitted = []
    table.bounds_changed.connect(
        lambda name, lo, hi: emitted.append((name, lo, hi))
    )

    table.item(0, 3).setText("inf")  # non-finite max
    table.item(0, 2).setText("20")  # min > max (inverted)

    assert emitted == []


def test_get_parameters_skips_invalid_rows(qapp):
    # Regression test: get_parameters() is the read path a Run click uses
    # (fit/step3_nlsq.py), separate from the itemChanged signal path the
    # tests above cover. A cell can hold invalid text without ever emitting
    # (e.g. "-5" is out of [0, 10] bounds -- itemChanged blocks the signal
    # but leaves the cell text as "-5"), so get_parameters() must re-check
    # rather than trust that "no signal fired" means "no invalid state".
    table = _make_table(qapp)
    table.item(0, 1).setText("-5")  # out of bounds, itemChanged already saw this

    assert table.get_parameters() == {}


def test_narrowing_bounds_below_current_value_invalidates_it(qapp):
    # Bounds edit itself is valid (0 <= 0.5, both finite) so bounds_changed
    # fires -- but it retroactively excludes the current value (fixture's
    # G0 = 1.0), which must be caught by the value-cell revalidation inside
    # the same handler and reflected in get_parameters(), not just left as
    # red styling with no observable effect on the read path.
    table = _make_table(qapp)
    bounds = []
    table.bounds_changed.connect(lambda name, lo, hi: bounds.append((lo, hi)))

    table.item(0, 3).setText("0.5")  # narrow max below the current value (1.0)

    # Qt may re-fire itemChanged when styling is applied inside the handler
    # (see test_valid_value_and_bounds_are_emitted's identical caveat).
    assert bounds and all(b == (0.0, 0.5) for b in bounds)
    assert table.get_parameters() == {}


def test_valid_value_and_bounds_are_emitted(qapp):
    table = _make_table(qapp)
    values = []
    bounds = []
    table.parameter_changed.connect(lambda name, value: values.append(value))
    table.bounds_changed.connect(lambda name, lo, hi: bounds.append((lo, hi)))

    table.item(0, 1).setText("5.0")
    table.item(0, 3).setText("20")

    # Qt may re-fire itemChanged when styling (font/foreground) is applied
    # inside the handler, so just assert every emission carries the correct
    # (valid) value/bounds rather than an exact call count.
    assert values and all(v == 5.0 for v in values)
    assert bounds and all(b == (0.0, 20.0) for b in bounds)


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
