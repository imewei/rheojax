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
    # Rejection must be visible at Apply-time, not just logged -- the preview
    # pane's own error text can be stale/unnoticed by the time Apply is clicked.
    # isHidden() reflects this widget's own show()/hide() state regardless of
    # whether the (unshown, qtbot-parented-only) editor's ancestor chain is
    # visible -- isVisible() would need editor.show() to mean anything here.
    assert not editor._apply_error_label.isHidden()
    assert "Not applied" in editor._apply_error_label.text()


def test_to_numpyro_priors_converts_editor_shape(qtbot):
    editor = PriorsEditor()
    qtbot.addWidget(editor)
    editor.set_prior("G0", "lognormal", loc=0.0, scale=1.0)

    result = editor.to_numpyro_priors()

    assert result == {"G0": {"type": "lognormal", "loc": 0.0, "scale": 1.0}}


def test_to_numpyro_priors_applies_exponential_scale_to_rate(qtbot):
    # foundation.priors.adapt_prior's NumPyro-specific correction must be
    # reachable through the editor's own interface, not just the free
    # function -- this is the whole point of wrapping it here.
    editor = PriorsEditor()
    qtbot.addWidget(editor)
    editor.set_prior("tau", "exponential", scale=2.0)

    result = editor.to_numpyro_priors()

    assert result == {"tau": {"type": "exponential", "rate": 0.5}}


def test_load_numpyro_priors_seeds_editor_shape(qtbot):
    editor = PriorsEditor()
    qtbot.addWidget(editor)

    editor.load_numpyro_priors(
        {
            "G0": {"type": "lognormal", "loc": 0.0, "scale": 1.0},
            "sigma": {"type": "halfnormal", "scale": 1.0},
        }
    )

    assert editor.get_prior("G0") == {
        "distribution": "lognormal",
        "params": {"loc": 0.0, "scale": 1.0},
    }
    assert editor.get_prior("sigma") == {
        "distribution": "halfnormal",
        "params": {"scale": 1.0},
    }


def test_load_numpyro_priors_round_trips_through_to_numpyro_priors(qtbot):
    editor = PriorsEditor()
    qtbot.addWidget(editor)
    suggested = {"G0": {"type": "lognormal", "loc": 2.3, "scale": 1.0}}

    editor.load_numpyro_priors(suggested)

    assert editor.to_numpyro_priors() == suggested
