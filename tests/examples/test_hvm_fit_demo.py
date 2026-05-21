from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


def test_hvm_demo_fits_overlap_generated_rheology_data():
    from utils.hvm_demo_fit import run_hvm_demo_fits

    results = run_hvm_demo_fits(noise_level=0.01)

    expected_protocols = {"flow_curve", "relaxation", "creep", "startup"}
    assert set(results) == expected_protocols

    for protocol, result in results.items():
        assert result.r_squared > 0.95, protocol
        assert np.all(np.isfinite(result.x_data)), protocol
        assert np.all(np.isfinite(result.y_data)), protocol
        assert np.all(np.isfinite(result.y_fit)), protocol
        assert result.x_fit.shape == result.y_fit.shape, protocol


def test_hvm_fit_demo_notebook_exists():
    notebook = EXAMPLES_DIR / "hvm" / "14_hvm_fit_demo.ipynb"
    assert notebook.exists()
