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
INPUT_DIR = ROOT / "golden_data" / "input"
RJ_OUT_DIR = ROOT / "golden_data" / "outputs" / "rheojax"
MAT_OUT_DIR = ROOT / "golden_data" / "outputs" / "matlab"
R_OUT_DIR = ROOT / "golden_data" / "outputs" / "r"

DATASETS = ["sin_fundamental", "sin_noisy"]


def _ensure_inputs():
    if not INPUT_DIR.exists():
        gen_inputs_main()


def _ensure_rheojax_outputs(datasets):
    for ds in datasets:
        out_file = RJ_OUT_DIR / f"{ds}_spp_data_out.csv"
        if not out_file.exists():
            run_rheojax_dataset(ds, INPUT_DIR, RJ_OUT_DIR)


def _load_csv(path: Path):
    return np.loadtxt(path, delimiter=",")


def _maybe_skip(m_path: Path, r_path: Path):
    if not m_path.exists() and not r_path.exists():
        pytest.skip("No MATLAB or R golden output available; run scripts/run_sppplus_v2p1.m and scripts/run_oreo.R")
    return m_path if m_path.exists() else r_path


@pytest.mark.parametrize("dataset", DATASETS)
def test_spp_parity_against_goldens(dataset):
    _ensure_inputs()
    _ensure_rheojax_outputs([dataset])

    golden_mat = MAT_OUT_DIR / f"{dataset}_spp_data_out_fourier.csv"
    golden_r = R_OUT_DIR / f"{dataset}_spp_data_out_fourier.csv"
    golden_path = _maybe_skip(golden_mat, golden_r)

    gold = _load_csv(golden_path)
    rj = _load_csv(RJ_OUT_DIR / f"{dataset}_spp_data_out.csv")

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
    arr = _load_csv(path)
    assert arr.shape[1] == 9

