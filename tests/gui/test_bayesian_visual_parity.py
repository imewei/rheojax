"""Visual-ish parity via deterministic plot hash from fixture.

Uses bundled owchirp TTS fixture to generate a posterior predictive-style plot
and asserts the PNG SHA-256 matches a stored golden hash. Avoids binary assets.
"""

import csv
import hashlib
import io

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytestmark = [pytest.mark.smoke]


def test_bayesian_ppd_plot_hash():
    rng = np.random.default_rng(0)

    with open("tests/fixtures/bayesian_owchirp_tts.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    freq = np.array([float(r["freq"]) for r in rows])
    gp = np.array([float(r["Gprime"]) for r in rows])
    gpp = np.array([float(r["Gdoubleprime"]) for r in rows])

    # Posterior predictive surrogate: scaled data with fixed band
    mean_gp = gp * 0.97
    std_gp = np.full_like(mean_gp, 50.0)

    plt.figure(figsize=(4, 3), dpi=100)
    plt.plot(freq, gp, "k.", label="data")
    plt.plot(freq, mean_gp, "b-", label="posterior mean")
    plt.fill_between(
        freq,
        mean_gp - 2 * std_gp,
        mean_gp + 2 * std_gp,
        color="blue",
        alpha=0.2,
        label="95% band",
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("freq (rad/s)")
    plt.ylabel('G" (Pa)')
    plt.legend(loc="lower right")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()

    digest = hashlib.sha256(buf.getvalue()).hexdigest()

    assert digest == "ed07b1d189a95871a795c14a7df02ecbbdc5c296e3ca37970e24a3e59b62af33"
