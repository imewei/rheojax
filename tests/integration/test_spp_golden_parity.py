"""Golden-data parity checks for SPP.

If MATLAB/R goldens are present under golden_data/outputs/{matlab,r}, compare
RheoJAX outputs to those goldens. If goldens are missing, tests are skipped
with a clear reason.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scripts.run_rheojax import run_dataset as run_rheojax_dataset
from scripts.gen_inputs import main as gen_inputs_main


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
INPUT_DIR = SCRIPTS_DIR / "golden_data" / "input"
RJ_OUT_DIR = SCRIPTS_DIR / "golden_data" / "outputs" / "rheojax"
MAT_OUT_DIR = SCRIPTS_DIR / "golden_data" / "outputs" / "matlab"
R_OUT_DIR = SCRIPTS_DIR / "golden_data" / "outputs" / "r"

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
    """Load data from CSV or MATLAB txt output files."""
    with open(path) as f:
        lines = f.readlines()

    # MATLAB .txt files have header info; find the data start
    if path.suffix == ".txt":
        # MATLAB format: skip lines until we find the data (after header row with units)
        # Header pattern: "Time\tStrain\t..." followed by "[s]\t[-]\t..."
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("[s]") or line.startswith("Time"):
                continue
            # Check if line starts with a number (data row)
            first_token = line.split()[0] if line.split() else ""
            try:
                float(first_token)
                data_start = i
                break
            except ValueError:
                continue
        return np.loadtxt(path, skiprows=data_start)
    else:
        # CSV format (R or RheoJAX)
        first_line = lines[0] if lines else ""
        skip = 1 if first_line and not first_line[0].lstrip('-').replace('.', '').replace('e', '').replace('+', '').isdigit() else 0
        return np.loadtxt(path, delimiter=",", skiprows=skip)


def _maybe_skip(m_path: Path, r_path: Path):
    # Prefer MATLAB goldens (reference implementation); R goldens may have issues
    if m_path.exists():
        return m_path
    if not r_path.exists():
        pytest.skip("No MATLAB golden output available; run scripts/run_sppplus_v2p1.m in MATLAB")
    # R goldens exist but MATLAB preferred - skip with note
    pytest.skip("MATLAB golden not available (R golden exists but may have issues); run scripts/run_sppplus_v2p1.m")


@pytest.mark.parametrize("dataset", DATASETS)
def test_spp_parity_against_goldens(dataset):
    _ensure_inputs()
    _ensure_rheojax_outputs([dataset])

    # MATLAB outputs are .txt, R outputs are .csv
    golden_mat = MAT_OUT_DIR / f"{dataset}_spp_data_out_fourier.txt"
    golden_r = R_OUT_DIR / f"{dataset}_spp_data_out_fourier.csv"
    golden_path = _maybe_skip(golden_mat, golden_r)

    gold = _load_data(golden_path)
    rj = _load_data(RJ_OUT_DIR / f"{dataset}_spp_data_out.csv")

    # Align length (gold may include different samples); take min length
    n = min(len(gold), len(rj))
    gold = gold[:n, :]
    rj = rj[:n, :]

    # Compare core columns (time, strain, rate, stress, Gp, Gpp, |G*|, tanδ, δ)
    cols = list(range(9))
    np.testing.assert_allclose(rj[:, cols], gold[:, cols], rtol=1e-2, atol=1e-4)


@pytest.mark.parametrize("dataset", DATASETS)
def test_fsf_columns_present(dataset):
    _ensure_inputs()
    _ensure_rheojax_outputs([dataset])
    path = RJ_OUT_DIR / f"{dataset}_fsf_data_out.csv"
    if not path.exists():
        pytest.skip("FSF output not present for RheoJAX run")
    arr = _load_data(path)
    assert arr.shape[1] == 9

