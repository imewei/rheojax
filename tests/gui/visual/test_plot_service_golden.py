"""Generate and compare PlotService goldens using sample data.

On first run, goldens are created and the test is skipped. Subsequent runs
compare generated images against the stored goldens to catch regressions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from rheojax.gui.services.plot_service import PlotService
from rheojax.gui.state.store import FitResult
from rheojax.core.data import RheoData


GOLDEN_DIR = Path(__file__).parent / "golden_images"
GOLDEN_FIT = GOLDEN_DIR / "fit_plot.png"


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _generate_fit_plot(path: Path) -> None:
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
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)

    # Generate current image
    current = tmp_path / "fit_plot_current.png"
    _generate_fit_plot(current)

    if not GOLDEN_FIT.exists():
        # First run: create golden and skip
        _generate_fit_plot(GOLDEN_FIT)
        pytest.skip(f"Golden created at {GOLDEN_FIT}, re-run to compare.")

    # Compare hashes
    golden_hash = _hash_file(GOLDEN_FIT)
    current_hash = _hash_file(current)
    assert golden_hash == current_hash, "PlotService fit plot differs from golden"
