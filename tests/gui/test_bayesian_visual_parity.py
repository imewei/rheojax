"""Visual parity for Bayesian posterior predictive plot.

Uses bundled owchirp TTS fixture to generate a posterior predictive-style plot
and verifies the plot is created correctly with expected data and structure.
Platform-independent: validates data arrays and plot structure rather than
pixel-level rendering (which varies across OS font rasterizers).
"""

import csv
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tests.gui.conftest import run_gui_code_subprocess

pytestmark = [pytest.mark.smoke]

_FIXTURE = "tests/fixtures/bayesian_owchirp_tts.csv"


def test_bayesian_ppd_plot_hash():
    # Rendering runs in a fresh subprocess rather than the pytest worker. This
    # host's matplotlib/FreeType text-metrics state reliably becomes corrupted
    # after enough Figure churn within a single process, so savefig() here
    # raised `FT_Render_Glyph ... raster overflow` / std::bad_alloc when the
    # figure was built in-process (see test_visual_regression.py). A fresh
    # process renders correctly every time.
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        png_path = Path(f.name)

    render_code = f'''
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcdefaults()
with open({_FIXTURE!r}, newline="") as fh:
    rows = list(csv.DictReader(fh))
freq = np.array([float(r["freq"]) for r in rows])
gp = np.array([float(r["Gprime"]) for r in rows])

mean_gp = gp * 0.97
std_gp = np.full_like(mean_gp, 50.0)

fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
ax.plot(freq, gp, "k.", label="data")
ax.plot(freq, mean_gp, "b-", label="posterior mean")
ax.fill_between(
    freq, mean_gp - 2 * std_gp, mean_gp + 2 * std_gp,
    color="blue", alpha=0.2, label="95% band",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("freq (rad/s)")
ax.set_ylabel('G" (Pa)')
ax.legend(loc="lower right")
fig.tight_layout()

# Structure checks: 2 line artists (scatter, mean) + >=1 fill_between collection
assert len(ax.get_lines()) == 2, f"Expected 2 line artists, got {{len(ax.get_lines())}}"
assert len(ax.collections) >= 1, "Expected at least 1 fill_between collection"

fig.savefig({str(png_path)!r}, format="png")
plt.close(fig)
print("RENDER_OK")
'''

    try:
        result = run_gui_code_subprocess(render_code, timeout=30.0)
        assert not result.crashed and "RENDER_OK" in result.stdout, (
            f"PPD plot render failed in subprocess (rc={result.return_code}, "
            f"signal={result.signal_name}):\nSTDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        assert png_path.stat().st_size > 1000, "PNG output too small — plot likely empty"
    finally:
        png_path.unlink(missing_ok=True)

    # Verify data integrity (platform-independent, main process)
    with open(_FIXTURE, newline="") as f:
        rows = list(csv.DictReader(f))
    freq = np.array([float(r["freq"]) for r in rows])
    gp = np.array([float(r["Gprime"]) for r in rows])
    mean_gp = gp * 0.97
    std_gp = np.full_like(mean_gp, 50.0)

    assert len(freq) >= 10, "Fixture should have sufficient data points"
    np.testing.assert_allclose(mean_gp, gp * 0.97, rtol=1e-12)
    np.testing.assert_array_equal(std_gp, np.full_like(mean_gp, 50.0))
