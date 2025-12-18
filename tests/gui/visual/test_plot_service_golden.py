"""Generate and compare PlotService goldens using sample data.

On first run, goldens are created and the test is skipped. Subsequent runs
compare generated images against the stored goldens to catch regressions.

Uses perceptual image comparison with tolerance to handle minor rendering
differences across systems (fonts, anti-aliasing, matplotlib versions).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest
from PIL import Image

from rheojax.gui.services.plot_service import PlotService
from rheojax.gui.state.store import FitResult
from rheojax.core.data import RheoData


GOLDEN_DIR = Path(__file__).parent / "golden_images"
GOLDEN_FIT = GOLDEN_DIR / "fit_plot.png"

# Tolerance for image comparison (0-100, percentage of pixels that can differ)
PIXEL_DIFF_TOLERANCE = 5.0  # Allow up to 5% pixel differences
# Maximum allowed per-pixel difference (0-255)
MAX_PIXEL_DIFF = 30


def _images_similar(img1_path: Path, img2_path: Path) -> tuple[bool, str]:
    """Compare two images with perceptual tolerance.

    Returns
    -------
    tuple[bool, str]
        (is_similar, message) where is_similar indicates if images match
        within tolerance, and message describes any differences.
    """
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # Check dimensions
    if img1.size != img2.size:
        return False, f"Size mismatch: {img1.size} vs {img2.size}"

    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)

    # Calculate per-pixel difference
    diff = np.abs(arr1 - arr2)
    max_diff = diff.max()

    # Count pixels with significant differences
    significant_diff_mask = diff.max(axis=2) > MAX_PIXEL_DIFF
    diff_pixel_count = significant_diff_mask.sum()
    total_pixels = arr1.shape[0] * arr1.shape[1]
    diff_percentage = (diff_pixel_count / total_pixels) * 100

    if diff_percentage > PIXEL_DIFF_TOLERANCE:
        return False, (
            f"Image differs: {diff_percentage:.2f}% pixels differ "
            f"(tolerance: {PIXEL_DIFF_TOLERANCE}%), max diff: {max_diff}"
        )

    return True, f"Images match (diff: {diff_percentage:.2f}%, max_diff: {max_diff})"


def _generate_fit_plot(path: Path) -> None:
    """Generate a fit plot for testing."""
    plot_service = PlotService()
    x = np.logspace(-1, 2, 20)
    y = 1e4 * (x / (1 + x**2))
    y_fit = y * 0.98
    data = RheoData(x=x, y=y, metadata={"test_mode": "oscillation"})
    fit_result = FitResult(
        model_name="test_model",
        dataset_id="dummy",
        parameters={},
        r_squared=0.99,
        mpe=1.0,
        chi_squared=0.1,
        fit_time=0.5,
        timestamp=None,
        num_iterations=10,
        convergence_message="",
        x_fit=x,
        y_fit=y_fit,
        residuals=y - y_fit,
    )
    fig = plot_service.create_fit_plot(data, fit_result, style="default", test_mode="oscillation")
    fig.savefig(path, dpi=150)
    matplotlib.pyplot.close(fig)


@pytest.mark.gui
def test_plot_service_fit_golden(tmp_path):
    """Test PlotService fit plot against golden image."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Generate current image
    current = tmp_path / "fit_plot_current.png"
    _generate_fit_plot(current)

    if not GOLDEN_FIT.exists():
        # First run: create golden and skip
        _generate_fit_plot(GOLDEN_FIT)
        pytest.skip(f"Golden created at {GOLDEN_FIT}, re-run to compare.")

    # Compare with perceptual tolerance
    is_similar, message = _images_similar(GOLDEN_FIT, current)
    assert is_similar, f"PlotService fit plot differs from golden: {message}"
