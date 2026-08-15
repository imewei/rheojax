"""Numerical regression coverage for ResidualsPanel._plot_autocorr().

PR #127 replaced the O(n^2) np.correlate(mode="full") autocorrelation
computation with a direct O(max_lag * n) dot-product loop bounded to the
plotted lags, to fix a GUI-thread freeze on large residual arrays. No test
exercised this method at all before or after that change (flagged by
pr-test-analyzer during PR #127's review) -- these tests close that gap by
asserting the plotted bar heights against an independent reference
(the original np.correlate(mode="full") formula), not just "renders without
error".
"""

import numpy as np
import pytest

pytest.importorskip("PySide6")

from rheojax.gui.widgets.residuals_panel import ResidualsPanel


def _reference_autocorr(residuals: np.ndarray, max_lag: int) -> np.ndarray:
    """Independent oracle: the full np.correlate(mode="full") formula that
    _plot_autocorr() used before PR #127's dot-product rewrite.
    """
    n = len(residuals)
    centered = residuals - np.mean(residuals)
    full = np.correlate(centered, centered, mode="full")
    autocorr = full[n - 1 : n - 1 + max_lag]
    if autocorr[0] < 1e-15:
        return np.zeros_like(autocorr)
    return autocorr / autocorr[0]


def test_plot_autocorr_matches_full_correlate_reference(qtbot):
    """The dot-product rewrite must reproduce np.correlate(mode="full")
    exactly, to double precision, for real-valued residuals.
    """
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    rng = np.random.default_rng(42)
    n = 60
    residuals = rng.normal(0, 1, size=n)
    max_lag = min(40, n // 4)

    panel.set_plot_type("autocorr")
    panel.set_residuals(residuals)

    assert panel._current_plot_type == "autocorr"
    ax = panel._figure.axes[0]
    bar_heights = np.array([p.get_height() for p in ax.patches])
    assert len(bar_heights) == max_lag

    expected = _reference_autocorr(residuals, max_lag)
    np.testing.assert_allclose(bar_heights, expected, rtol=1e-10, atol=1e-12)

    # Lag 0 is always fully self-correlated (normalized to 1.0).
    assert bar_heights[0] == pytest.approx(1.0)


def test_plot_autocorr_complex_residuals_splits_re_im(qtbot):
    """Complex (oscillation-mode G'+iG'') residuals must plot two
    independently-normalized autocorrelation series (Re, Im), each matching
    the reference formula on its own real-valued part.
    """
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    rng = np.random.default_rng(7)
    n = 50
    residuals = rng.normal(0, 1, size=n) + 1j * rng.normal(0, 2, size=n)
    max_lag = min(40, n // 4)

    panel.set_plot_type("autocorr")
    panel.set_residuals(residuals)

    ax = panel._figure.axes[0]
    bar_heights = np.array([p.get_height() for p in ax.patches])
    # Two interleaved bar series (Re, Im), each of length max_lag.
    assert len(bar_heights) == 2 * max_lag

    expected_re = _reference_autocorr(residuals.real, max_lag)
    expected_im = _reference_autocorr(residuals.imag, max_lag)
    np.testing.assert_allclose(
        bar_heights[:max_lag], expected_re, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        bar_heights[max_lag:], expected_im, rtol=1e-10, atol=1e-12
    )


def test_plot_autocorr_constant_residuals_normalizes_to_zero(qtbot):
    """VIS-018 guard: constant (perfect-fit) residuals have zero variance,
    so autocorr[0] is ~0 and every lag must report exactly 0.0 rather than
    dividing by a near-zero normalizer.
    """
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    n = 40
    residuals = np.full(n, 3.5)
    max_lag = min(40, n // 4)

    panel.set_plot_type("autocorr")
    panel.set_residuals(residuals)

    ax = panel._figure.axes[0]
    bar_heights = np.array([p.get_height() for p in ax.patches])
    assert len(bar_heights) == max_lag
    np.testing.assert_array_equal(bar_heights, np.zeros(max_lag))


def test_plot_autocorr_below_minimum_points_shows_message_not_bars(qtbot):
    """Fewer than 10 residuals hits the early-return guard: an explanatory
    text annotation is drawn and no bars are plotted.
    """
    panel = ResidualsPanel()
    qtbot.addWidget(panel)

    panel.set_plot_type("autocorr")
    panel.set_residuals(np.array([1.0, 2.0, 3.0]))

    ax = panel._figure.axes[0]
    assert len(ax.patches) == 0
    assert len(ax.texts) >= 1
    assert "not enough data" in ax.texts[0].get_text().lower()
