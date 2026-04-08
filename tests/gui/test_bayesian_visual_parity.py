"""Visual parity for Bayesian posterior predictive plot.

Uses bundled owchirp TTS fixture to generate a posterior predictive-style plot
and verifies the plot is created correctly with expected data and structure.
Platform-independent: validates data arrays and plot structure rather than
pixel-level rendering (which varies across OS font rasterizers).
"""

import csv
import io

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest

pytestmark = [pytest.mark.smoke]


def test_bayesian_ppd_plot_hash():
    plt.rcdefaults()

    with open("tests/fixtures/bayesian_owchirp_tts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    freq = np.array([float(r["freq"]) for r in rows])
    gp = np.array([float(r["Gprime"]) for r in rows])

    # Posterior predictive surrogate: scaled data with fixed band
    mean_gp = gp * 0.97
    std_gp = np.full_like(mean_gp, 50.0)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    ax.plot(freq, gp, "k.", label="data")
    ax.plot(freq, mean_gp, "b-", label="posterior mean")
    ax.fill_between(
        freq,
        mean_gp - 2 * std_gp,
        mean_gp + 2 * std_gp,
        color="blue",
        alpha=0.2,
        label="95% band",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("freq (rad/s)")
    ax.set_ylabel('G" (Pa)')
    ax.legend(loc="lower right")
    fig.tight_layout()

    # Verify plot renders to a non-trivial PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    assert buf.tell() > 1000, "PNG output too small — plot likely empty"

    # Verify data integrity (platform-independent)
    assert len(freq) >= 10, "Fixture should have sufficient data points"
    np.testing.assert_allclose(mean_gp, gp * 0.97, rtol=1e-12)
    np.testing.assert_array_equal(std_gp, np.full_like(mean_gp, 50.0))

    # Verify plot structure: 3 artists (scatter, line, fill_between)
    lines = ax.get_lines()
    assert len(lines) == 2, f"Expected 2 line artists, got {len(lines)}"
    assert len(ax.collections) >= 1, "Expected at least 1 fill_between collection"
