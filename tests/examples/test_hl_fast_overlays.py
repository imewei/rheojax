import numpy as np

from examples.utils.hl_tutorial_utils import (
    compute_hl_fast_creep_overlay,
    compute_hl_fast_flow_curve_overlay,
    compute_hl_fast_relaxation_overlay,
)


def test_fast_flow_curve_overlay_tracks_emulsion_data():
    raw = np.loadtxt("examples/data/flow/emulsions/0.80.csv", delimiter=",", skiprows=1)
    gamma_dot = raw[:, 0]
    stress = raw[:, 1]

    x_fit, stress_fit, fit_info = compute_hl_fast_flow_curve_overlay(gamma_dot, stress)

    assert x_fit.shape == stress_fit.shape
    assert x_fit.size == 200
    assert np.all(np.isfinite(stress_fit))
    assert np.all(stress_fit > 0)
    assert fit_info["log_r2"] > 0.95


def test_fast_relaxation_overlay_tracks_laponite_data():
    raw = np.loadtxt(
        "examples/data/relaxation/clays/rel_lapo_3600.csv",
        delimiter="\t",
        skiprows=1,
    )
    time = raw[:, 0]
    modulus = raw[:, 1]

    x_fit, modulus_fit, fit_info = compute_hl_fast_relaxation_overlay(time, modulus)

    assert x_fit.shape == modulus_fit.shape
    assert x_fit.size == 200
    assert np.all(np.isfinite(modulus_fit))
    assert np.all(modulus_fit > 0)
    assert fit_info["log_r2"] > 0.90


def test_fast_creep_overlay_tracks_polymer_data():
    raw = np.loadtxt(
        "examples/data/creep/polymers/creep_ps145_data.csv",
        delimiter="\t",
        skiprows=1,
    )
    time = raw[:, 0]
    compliance = raw[:, 1]

    x_fit, compliance_fit, fit_info = compute_hl_fast_creep_overlay(time, compliance)

    assert x_fit.shape == compliance_fit.shape
    assert x_fit.size == 200
    assert np.all(np.isfinite(compliance_fit))
    assert np.all(compliance_fit > 0)
    assert fit_info["log_r2"] > 0.95
