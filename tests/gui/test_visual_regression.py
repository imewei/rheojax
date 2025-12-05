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


def save_golden_image(figure: Any, path: Path) -> None:
    """Save figure as golden reference image.

    Parameters
    ----------
    figure : matplotlib.Figure
        Matplotlib figure object
    path : Path
        Output file path
    """
    figure.savefig(path, dpi=100, bbox_inches="tight")


def compare_figures(
    actual: Any, expected_path: Path, threshold: float = 0.01
) -> bool:
    """Compare figure against golden image.

    Parameters
    ----------
    actual : matplotlib.Figure
        Actual figure to compare
    expected_path : Path
        Path to golden reference image
    threshold : float
        Maximum allowed difference ratio (0-1)

    Returns
    -------
    bool
        True if images match within threshold
    """
    import matplotlib.pyplot as plt
    from matplotlib.testing.compare import compare_images

    # Save actual to temp file
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        actual_path = Path(f.name)

    actual.savefig(actual_path, dpi=100, bbox_inches="tight")

    # Compare
    try:
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
        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)

        x, y = sample_sine_data
        canvas.plot_data(x, y, label="sin(x)")

        # Get figure for comparison (PlotCanvas uses .figure not get_figure())
        fig = canvas.figure
        golden_path = golden_dir / "plot_canvas_line.png"

        # Generate golden if doesn't exist
        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        # Compare against golden
        assert compare_figures(fig, golden_path), "Line plot visual mismatch"

        canvas.close()

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

        # PlotCanvas uses .axes for the subplot
        canvas.axes.scatter(x, y, alpha=0.6)
        canvas.axes.set_xlabel("X")
        canvas.axes.set_ylabel("Y")
        canvas.canvas.draw()

        fig = canvas.figure
        golden_path = golden_dir / "plot_canvas_scatter.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "Scatter plot visual mismatch"

        canvas.close()

    def test_plot_canvas_loglog_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_maxwell_data: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test log-log rheology plot rendering."""
        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)

        omega = sample_maxwell_data["omega"]
        G_prime = sample_maxwell_data["G_prime"]
        G_double_prime = sample_maxwell_data["G_double_prime"]

        # Use PlotCanvas's axes and set scale
        canvas.axes.loglog(omega, G_prime, "o-", label="G'")
        canvas.axes.loglog(omega, G_double_prime, "s-", label="G''")
        canvas.axes.set_xlabel("ω (rad/s)")
        canvas.axes.set_ylabel("G', G'' (Pa)")
        canvas.axes.legend()
        canvas.canvas.draw()

        fig = canvas.figure
        golden_path = golden_dir / "plot_canvas_loglog.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "Log-log plot visual mismatch"

        canvas.close()


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

    def test_residuals_vs_fitted(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test residuals vs fitted plot."""
        from rheojax.gui.widgets.residuals_panel import ResidualsPanel

        panel = ResidualsPanel()
        qtbot.addWidget(panel)

        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        y_true = fitted + residuals
        y_pred = fitted

        panel.plot_residuals(y_true, y_pred)
        panel.set_plot_type("residuals")

        fig = panel.get_figure()
        golden_path = golden_dir / "residuals_vs_fitted.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "Residuals plot visual mismatch"

        panel.close()

    def test_qq_plot(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test Q-Q plot."""
        from rheojax.gui.widgets.residuals_panel import ResidualsPanel

        panel = ResidualsPanel()
        qtbot.addWidget(panel)

        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        y_true = fitted + residuals
        y_pred = fitted

        panel.plot_residuals(y_true, y_pred)
        panel.set_plot_type("qq")

        fig = panel.get_figure()
        golden_path = golden_dir / "residuals_qq.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "Q-Q plot visual mismatch"

        panel.close()

    def test_histogram(
        self,
        qapp: QApplication,
        qtbot: Any,
        sample_residuals: dict[str, np.ndarray],
        golden_dir: Path,
    ) -> None:
        """Test residuals histogram."""
        from rheojax.gui.widgets.residuals_panel import ResidualsPanel

        panel = ResidualsPanel()
        qtbot.addWidget(panel)

        fitted = sample_residuals["fitted"]
        residuals = sample_residuals["residuals"]
        y_true = fitted + residuals
        y_pred = fitted

        panel.plot_residuals(y_true, y_pred)
        panel.set_plot_type("histogram")

        fig = panel.get_figure()
        golden_path = golden_dir / "residuals_histogram.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "Histogram visual mismatch"

        panel.close()


# =============================================================================
# MultiView Visual Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestMultiViewVisual:
    """Visual regression tests for MultiView widget."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_multiview_2x2_layout(
        self,
        qapp: QApplication,
        qtbot: Any,
        golden_dir: Path,
    ) -> None:
        """Test 2x2 multi-view layout."""
        from rheojax.gui.widgets.multi_view import MultiView

        view = MultiView(layout="2x2")
        qtbot.addWidget(view)

        # Add different plots to each panel
        rng = np.random.default_rng(42)

        for i in range(4):
            panel = view.get_panel(i)
            if panel:
                fig = panel.get_figure()
                ax = fig.add_subplot(111)
                x = np.linspace(0, 10, 50)
                y = np.sin(x + i * np.pi / 4) + rng.standard_normal(50) * 0.1
                ax.plot(x, y)
                ax.set_title(f"Panel {i + 1}")
                panel.refresh()

        # Export combined view - use first panel's figure for comparison
        panel0 = view.get_panel(0)
        assert panel0 is not None
        fig = panel0.get_figure()
        golden_path = golden_dir / "multiview_panel0.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(fig, golden_path), "MultiView panel visual mismatch"

        view.close()


# =============================================================================
# ArvizCanvas Visual Tests (Placeholder - requires InferenceData)
# =============================================================================


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
        from rheojax.gui.widgets.arviz_canvas import ArvizCanvas

        canvas = ArvizCanvas()
        qtbot.addWidget(canvas)

        # Canvas should show "No data loaded" message
        fig = canvas.get_figure()
        golden_path = golden_dir / "arviz_canvas_empty.png"

        if not golden_path.exists():
            save_golden_image(fig, golden_path)
            pytest.skip("Generated golden image - rerun test to validate")

        assert compare_figures(
            fig, golden_path, threshold=0.05
        ), "ArviZ empty state visual mismatch"

        canvas.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
