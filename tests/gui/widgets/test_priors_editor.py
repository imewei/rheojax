"""Regression test: PriorsEditor.Apply must not store an invalid prior."""

from __future__ import annotations

from rheojax.gui.widgets.priors_editor import PriorsEditor


def test_apply_refuses_invalid_prior(qtbot):
    # _compute_pdf() (used by the live preview) already rejects an inverted
    # uniform range, but _apply_prior() used to store it anyway -- an
    # invalid prior would then reach NUTS unfiltered and fail there with an
    # opaque NumPyro/JAX error instead of being caught at the Apply button.
    editor = PriorsEditor()
    qtbot.addWidget(editor)
    editor.set_parameters(["a"])
    default_prior = editor.get_prior("a")
    emitted = []
    editor.prior_changed.connect(lambda *args: emitted.append(args))

    editor._current_param = "a"
    # blockSignals on both the combo and spinboxes: driving these through
    # their real signals would fire _on_dist_changed()/_update_preview(),
    # which renders a matplotlib Figure -- a documented, environment-specific
    # FreeType crash in this sandbox unrelated to what this test checks (see
    # tests/conftest.py's FT_Render_Glyph note). _apply_prior() itself never
    # renders, so blocking these signals sidesteps the crash without
    # weakening the assertion below.
    editor._dist_combo.blockSignals(True)
    editor._dist_combo.set_current_data("uniform")
    editor._dist_combo.blockSignals(False)
    editor._param_spinboxes["low"].blockSignals(True)
    editor._param_spinboxes["high"].blockSignals(True)
    editor._param_spinboxes["low"].setValue(10.0)
    editor._param_spinboxes["high"].setValue(5.0)  # inverted: low > high
    editor._param_spinboxes["low"].blockSignals(False)
    editor._param_spinboxes["high"].blockSignals(False)

    editor._apply_prior()

    assert editor.get_prior("a") == default_prior
    assert emitted == []
