"""RheoJAX GUI Visual Regression Tests.

Image-based regression tests for GUI plot widgets.
Uses pytest-image-diff for golden image comparisons.

Markers:
    gui: All GUI-related tests
    visual: Visual regression tests

Run with:
    pytest tests/gui/test_visual_regression.py -v
    pytest tests/gui/ -v -m visual

Golden images stored in: tests/gui/golden_images/
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.gui.conftest import run_gui_code_subprocess

# Mark all tests as GUI and visual tests
pytestmark = [pytest.mark.gui, pytest.mark.visual]

# Check if PySide6 is available
try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False

# Golden images directory
GOLDEN_DIR = Path(__file__).parent / "golden_images"


@pytest.fixture
def golden_dir() -> Path:
    """Get or create golden images directory."""
    GOLDEN_DIR.mkdir(exist_ok=True)
    return GOLDEN_DIR


@pytest.fixture
def sample_sine_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample sine wave data for plots."""
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    return x, y


@pytest.fixture
def sample_maxwell_data() -> dict[str, np.ndarray]:
    """Generate sample Maxwell model data."""
    omega = np.logspace(-2, 2, 50)
    G0, tau = 1000.0, 1.0
    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)
    return {
        "omega": omega,
        "G_prime": G_prime,
        "G_double_prime": G_double_prime,
    }


@pytest.fixture
def sample_residuals() -> dict[str, np.ndarray]:
    """Generate sample residual data for diagnostics."""
    rng = np.random.default_rng(42)
    n = 100
    fitted = np.linspace(0, 100, n)
    residuals = rng.standard_normal(n) * 5
    return {"fitted": fitted, "residuals": residuals}


# =============================================================================
# Matplotlib Figure Comparison Utilities
# =============================================================================


def figure_to_array(figure: Any) -> np.ndarray:
    """Convert matplotlib figure to numpy array for comparison.

    Parameters
    ----------
    figure : matplotlib.Figure
        Matplotlib figure object

    Returns
    -------
    np.ndarray
        RGB image array
    """
    figure.canvas.draw()
    buf = figure.canvas.buffer_rgba()
    return np.asarray(buf)


# Rendering runs in a fresh, isolated subprocess rather than the pytest
# worker process. Root cause (confirmed empirically): this host's
# matplotlib/FreeType text-metrics state reliably becomes corrupted after
# enough Figure/FigureCanvasQTAgg churn within a single process -- the exact
# same widget-creation code produces a correct ~7x5 inch tight bbox in a
# fresh interpreter, but a corrupted ~1447x1946 inch bbox (-> MemoryError:
# std::bad_alloc trying to allocate the resulting ~28-billion-pixel buffer)
# after running inside a pytest session with other GUI tests, and a raw
# `FT_Render_Glyph ... raster overflow` even with bbox_inches=None (so the
# corruption is in glyph rasterization itself, not just tight-bbox
# measurement). No savefig argument or memory cap avoids it -- only a fresh
# process does. `widget_code` must define a variable named `fig` bound to
# the Figure to save.
def _render_widget_figure(widget_code: str, save_path: Path, timeout: float = 15.0) -> None:
    preamble = (
        "from PySide6.QtWidgets import QApplication\n"
        "app = QApplication.instance() or QApplication([])\n"
    )
    full_code = (
        preamble
        + widget_code
        + f'\nfig.savefig({str(save_path)!r}, dpi=100, bbox_inches="tight")\n'
        + 'print("RENDER_OK")\n'
    )
    result = run_gui_code_subprocess(full_code, timeout=timeout)
    assert not result.crashed and "RENDER_OK" in result.stdout, (
        f"Widget render failed in subprocess (rc={result.return_code}, "
        f"signal={result.signal_name}):\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def save_golden_image(widget_code: str, path: Path) -> None:
    """Render widget_code's figure in an isolated subprocess and save it as
    the golden reference image."""
    _render_widget_figure(widget_code, path)


def compare_figures(widget_code: str, expected_path: Path, threshold: float = 0.01) -> bool:
    """Render widget_code's figure in an isolated subprocess and compare it
    against the golden image."""
    import tempfile

    from matplotlib.testing.compare import compare_images

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        actual_path = Path(f.name)

    try:
        _render_widget_figure(widget_code, actual_path)
        result = compare_images(
            str(expected_path), str(actual_path), tol=threshold * 100
        )
        return result is None  # None means images match
    finally:
        actual_path.unlink(missing_ok=True)


# =============================================================================
# PlotCanvas Visual Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestPlotCanvasVisual:
    """Visual regression tests for PlotCanvas widget."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_plot_canvas_line_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_sine_data: tuple[np.ndarray, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test basic line plot rendering."""
        x, y = sample_sine_data
        widget_code = f"""
import numpy as np
from rheojax.gui.widgets.plot_canvas import PlotCanvas
canvas = PlotCanvas()
x = np.array({x.tolist()!r})
y = np.array({y.tolist()!r})
canvas.plot_data(x, y, label="sin(x)")
fig = canvas.figure
"""
        golden_path = golden_dir / "plot_canvas_line.png"

        # Generate golden if doesn't exist
        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        # Compare against golden
        assert compare_figures(widget_code, golden_path), "Line plot visual mismatch"

    def test_plot_canvas_cleanup_closes_figure_despite_dead_canvas(
        self,
        qapp: QApplication,
        qtbot: Any,
        monkeypatch: Any,
    ) -> None:
        """cleanup() must still close the Figure when the C++ canvas widget
        is already gone (RuntimeError from accessing canvas.callbacks) —
        otherwise the Figure leaks via matplotlib's global registry."""
        import matplotlib.pyplot as plt

        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)
        fig = canvas.figure

        closed_figs = []
        monkeypatch.setattr(plt, "close", lambda f: closed_figs.append(f))

        # Simulate the real failure: accessing .callbacks on a deleted C++
        # canvas widget raises RuntimeError (Qt/PySide6 wrapped-object error).
        def _dead_callbacks(self: Any) -> None:
            raise RuntimeError("wrapped C/C++ object has been deleted")

        monkeypatch.setattr(
            type(canvas.canvas), "callbacks", property(_dead_callbacks)
        )

        canvas.cleanup()

        assert closed_figs == [fig], "plt.close(figure) was skipped after RuntimeError"

    def test_plot_canvas_scatter_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        golden_dir: Path,
    ) -> None:
        """Test scatter plot rendering."""
        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)

        rng = np.random.default_rng(42)
        x = rng.uniform(0, 10, 50)
        y = rng.uniform(0, 10, 50)

        widget_code = f"""
import numpy as np
from rheojax.gui.widgets.plot_canvas import PlotCanvas
canvas = PlotCanvas()
x = np.array({x.tolist()!r})
y = np.array({y.tolist()!r})
canvas.axes.scatter(x, y, alpha=0.6)
canvas.axes.set_xlabel("X")
canvas.axes.set_ylabel("Y")
canvas.canvas.draw()
fig = canvas.figure
"""
        golden_path = golden_dir / "plot_canvas_scatter.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path), "Scatter plot visual mismatch"

    def test_plot_canvas_loglog_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_maxwell_data: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test log-log rheology plot rendering."""
        omega = sample_maxwell_data["omega"]
        G_prime = sample_maxwell_data["G_prime"]
        G_double_prime = sample_maxwell_data["G_double_prime"]

        widget_code = f"""
import numpy as np
from rheojax.gui.widgets.plot_canvas import PlotCanvas
canvas = PlotCanvas()
omega = np.array({omega.tolist()!r})
G_prime = np.array({G_prime.tolist()!r})
G_double_prime = np.array({G_double_prime.tolist()!r})
canvas.axes.loglog(omega, G_prime, "o-", label="G'")
canvas.axes.loglog(omega, G_double_prime, "s-", label="G''")
canvas.axes.set_xlabel("\\u03c9 (rad/s)")
canvas.axes.set_ylabel("G', G'' (Pa)")
canvas.axes.legend()
canvas.canvas.draw()
fig = canvas.figure
"""
        golden_path = golden_dir / "plot_canvas_loglog.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path), "Log-log plot visual mismatch"


# =============================================================================
# ResidualsPanel Visual Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestResidualsPanelVisual:
    """Visual regression tests for ResidualsPanel widget."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    @staticmethod
    def _residuals_widget_code(
        y_true: np.ndarray, y_pred: np.ndarray, plot_type: str
    ) -> str:
        return f"""
import numpy as np
from rheojax.gui.widgets.residuals_panel import ResidualsPanel
panel = ResidualsPanel()
y_true = np.array({y_true.tolist()!r})
y_pred = np.array({y_pred.tolist()!r})
panel.plot_residuals(y_true, y_pred)
panel.set_plot_type({plot_type!r})
fig = panel.get_figure()
"""

    def test_residuals_vs_fitted(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test residuals vs fitted plot."""
        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        widget_code = self._residuals_widget_code(fitted + residuals, fitted, "residuals")
        golden_path = golden_dir / "residuals_vs_fitted.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path), "Residuals plot visual mismatch"

    def test_qq_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test Q-Q plot."""
        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        widget_code = self._residuals_widget_code(fitted + residuals, fitted, "qq")
        golden_path = golden_dir / "residuals_qq.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path), "Q-Q plot visual mismatch"

    def test_histogram(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test residuals histogram."""
        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        widget_code = self._residuals_widget_code(fitted + residuals, fitted, "histogram")
        golden_path = golden_dir / "residuals_histogram.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path), "Histogram visual mismatch"


# =============================================================================
# MultiView Visual Tests
# =============================================================================


# =============================================================================
# ArvizCanvas Visual Tests (Placeholder - requires InferenceData)
# =============================================================================

# Runs inside the isolated render subprocess (see _render_widget_figure) --
# must be self-contained, no references to names from the outer test module.
_ARVIZ_SYNTHETIC_IDATA_CODE = """
import numpy as np
from rheojax.core.arviz_utils import inference_data_from_dict

rng = np.random.default_rng(0)
posterior = {
    "a": rng.normal(loc=2.0, scale=0.5, size=(2, 50)),
    "b": rng.normal(loc=-1.0, scale=0.3, size=(2, 50)),
}
sample_stats = {
    "energy": rng.normal(loc=10.0, scale=2.0, size=(2, 50)),
    "diverging": np.zeros((2, 50), dtype=bool),
}
idata = inference_data_from_dict({"posterior": posterior, "sample_stats": sample_stats})
"""


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestArvizCanvasVisual:
    """Visual regression tests for ArvizCanvas widget.

    Note: Full ArviZ plots require InferenceData from actual MCMC runs.
    These tests verify basic canvas functionality.
    """

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_arviz_canvas_empty_state(
        self,
        qapp: QApplication,
        qtbot: Any,
        golden_dir: Path,
    ) -> None:
        """Test ArviZ canvas empty state."""
        widget_code = """
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas
canvas = ArvizCanvas()
fig = canvas.get_figure()
"""
        golden_path = golden_dir / "arviz_canvas_empty.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path, threshold=0.05), (
            "ArviZ empty state visual mismatch"
        )

    @pytest.mark.parametrize(
        "plot_type",
        ["trace", "pair", "forest", "posterior", "energy", "rank", "ess", "autocorr"],
    )
    def test_arviz_canvas_plot_type_renders(
        self,
        qapp: QApplication,
        qtbot: Any,
        golden_dir: Path,
        plot_type: str,
    ) -> None:
        """Golden-image coverage for each ArviZ 1.x plot type. Pixel-exact
        parity with the old 0.x rendering engine is not a goal -- only that
        each plot type renders without error and stays visually stable."""
        widget_code = (
            _ARVIZ_SYNTHETIC_IDATA_CODE
            + f"""
from rheojax.gui.widgets.arviz_canvas import ArvizCanvas
canvas = ArvizCanvas()
canvas.set_inference_data(idata)
canvas.set_plot_type({plot_type!r})
fig = canvas.get_figure()
"""
        )
        golden_path = golden_dir / f"arviz_canvas_{plot_type}.png"

        if not golden_path.exists():
            save_golden_image(widget_code, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(widget_code, golden_path, threshold=0.05), (
            f"ArviZ {plot_type} plot visual mismatch"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
