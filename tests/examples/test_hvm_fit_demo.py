from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_hvm_demo_fits_overlap_generated_rheology_data():
    # HVMLocal's ODE-based protocols (relaxation/creep/startup) fall back to
    # scipy's TRF with a numerical Jacobian (see HVMLocal._fit -- NLSQ's
    # forward-mode AD is incompatible with diffrax's custom_vjp), where each
    # residual evaluation re-solves the ODE at ~2s and can't be cached. Even
    # after loosening fit tolerances and halving the startup multi-start
    # count (see examples/utils/hvm_demo_fit.py), this measured 854s; 1200s
    # gives ~1.4x margin without masking a genuine hang.
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
    notebook = EXAMPLES_DIR / "hvm" / "08_hvm_fit_demo.ipynb"
    assert notebook.exists()
