"""FSF-output sanity check for SPP.

RheoJAX's SPP+ implementation was validated for numerical parity against
the reference MATLAB implementation (scripts/run_sppplus_v2p1.m) previously;
that cross-implementation check is a one-time validation, not something this
suite re-runs on every test session (no MATLAB/R installation or golden
reference data ships with this repo). This module keeps the structural
smoke check that doesn't depend on that external tooling: RheoJAX's own FSF
output has the expected column shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.gen_inputs import main as gen_inputs_main
from scripts.run_rheojax import run_dataset as run_rheojax_dataset

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
INPUT_DIR = SCRIPTS_DIR / "golden_data" / "input"
RJ_OUT_DIR = SCRIPTS_DIR / "golden_data" / "outputs" / "rheojax"

DATASETS = ["sin_fundamental", "sin_noisy"]


def _ensure_inputs():
    if not INPUT_DIR.exists():
        gen_inputs_main([])  # Pass empty list to avoid picking up pytest args


def _ensure_rheojax_outputs(datasets):
    for ds in datasets:
        out_file = RJ_OUT_DIR / f"{ds}_spp_data_out.csv"
        if not out_file.exists():
            run_rheojax_dataset(ds, INPUT_DIR, RJ_OUT_DIR)


def _load_data(path: Path):
    """Load a RheoJAX CSV output file, skipping a header row if present."""
    with open(path) as f:
        first_line = f.readline()
    skip = (
        1
        if first_line
        and not first_line[0]
        .lstrip("-")
        .replace(".", "")
        .replace("e", "")
        .replace("+", "")
        .isdigit()
        else 0
    )
    return np.loadtxt(path, delimiter=",", skiprows=skip)


@pytest.mark.parametrize("dataset", DATASETS)
def test_fsf_columns_present(dataset):
    _ensure_inputs()
    _ensure_rheojax_outputs([dataset])
    path = RJ_OUT_DIR / f"{dataset}_fsf_data_out.csv"
    if not path.exists():
        pytest.skip("FSF output not present for RheoJAX run")
    arr = _load_data(path)
    assert arr.shape[1] == 9
