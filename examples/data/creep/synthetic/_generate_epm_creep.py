"""Generate reference creep datasets used by examples/epm/04_epm_creep.ipynb.

Two CSVs are produced:

* ``epm_creep_synthetic.csv`` — pure ground-truth creep from ``LatticeEPM`` with
  known parameters. Three stress levels exercise the bounded / near-yield /
  unbounded regimes. The notebook uses this dataset to demonstrate that the
  NLSQ fit recovers the generating parameters (self-consistency).

* ``emulsion_creep_hb_calibrated.csv`` — creep curve generated from a
  ``LatticeEPM`` whose Herschel-Bulkley-like parameters were *first fitted to
  a real emulsion flow curve* (``examples/data/flow/emulsions/0.74.csv``,
  volume fraction φ=0.74). The flow-curve parameters are real (σ_y, τ_pl,
  n_fluid come from a measurement); the creep trajectory at a stress 20 %
  above σ_y is then simulated with the overstress kernel. The notebook uses
  this as a "real-material calibrated" yield-stress creep demonstration.

Regenerate with:
    cd /Users/b80985/Projects/rheojax
    uv run python examples/data/creep/synthetic/_generate_epm_creep.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.lattice import LatticeEPM

jax, jnp = safe_import_jax()

HERE = Path(__file__).parent
REPO = HERE.parents[3]
EMULSION_CSV = REPO / "examples" / "data" / "flow" / "emulsions" / "0.74.csv"


def _simulate_creep(
    model: LatticeEPM,
    time: np.ndarray,
    stress: float,
    seed: int,
) -> np.ndarray:
    """Return strain(t) for the given LatticeEPM at constant applied stress."""
    data = RheoData(
        x=jnp.asarray(time),
        y=jnp.zeros_like(jnp.asarray(time)),
        initial_test_mode="creep",
        metadata={"stress": float(stress)},
    )
    return np.asarray(model.predict(data, smooth=True, seed=seed).y)


def generate_synthetic_epm_creep(out_path: Path) -> dict:
    """Ground-truth EPM creep at three stress levels (bounded / near / above)."""
    # Ground-truth parameters. The yield stress is σ_c,mean = 0.5 Pa.
    mu = 1.0
    tau_pl = 1.0
    sigma_c_mean = 0.5
    sigma_c_std = 0.08  # tight distribution → sharper bounded/unbounded transition
    n_fluid = 2.0  # HB exponent n_HB = 1/n_fluid = 0.5

    model = LatticeEPM(
        L=32,
        dt=0.01,
        mu=mu,
        tau_pl=tau_pl,
        sigma_c_mean=sigma_c_mean,
        sigma_c_std=sigma_c_std,
        n_fluid=n_fluid,
        fluidity_form="overstress",
    )

    t = np.linspace(0.5, 10.0, 30)
    stress_levels = {
        "bounded": 0.30,  # σ < σ_y - σ_std (0.30 < 0.42) → bounded
        "near_yield": 0.55,  # σ ≈ σ_y → transition
        "unbounded": 0.80,  # σ > σ_y + σ_std (0.80 > 0.58) → unbounded
    }

    columns: dict[str, np.ndarray] = {"time_s": t}
    for label, sigma0 in stress_levels.items():
        gamma = _simulate_creep(model, t, sigma0, seed=42)
        columns[f"strain_{label}"] = gamma

    header = ",".join(columns.keys())
    data = np.column_stack(list(columns.values()))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        out_path,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6e",
    )
    return {
        "ground_truth_params": {
            "mu": mu,
            "tau_pl": tau_pl,
            "sigma_c_mean": sigma_c_mean,
            "sigma_c_std": sigma_c_std,
            "n_fluid": n_fluid,
        },
        "stress_levels": stress_levels,
        "path": out_path,
    }


def generate_hb_calibrated_emulsion_creep(out_path: Path) -> dict:
    """Fit LatticeEPM to a real emulsion flow curve, then simulate creep.

    The flow curve is ``0.74.csv`` (volume fraction φ=0.74). Fitting the EPM
    overstress kernel extracts a yield stress and HB-like exponent that
    reproduce the measured flow curve; we then simulate a creep trajectory at
    1.2·σ_y (i.e. above yield → unbounded creep).
    """
    raw = np.loadtxt(EMULSION_CSV, delimiter=",", skiprows=1)
    shear_rate = raw[:, 0]
    stress = raw[:, 1]

    fit_model = LatticeEPM(
        L=16,  # smaller lattice — flow-curve fit already slow enough
        dt=0.01,
        mu=1.0,
        tau_pl=1.0,
        sigma_c_mean=float(np.min(stress)) * 0.9,
        sigma_c_std=0.1 * float(np.min(stress)),
        n_fluid=2.0,
        fluidity_form="overstress",
    )
    # Generous bounds so the overstress kernel can fit the real curve.
    fit_model.parameters["mu"].bounds = (0.1, 100.0)
    fit_model.parameters["tau_pl"].bounds = (0.01, 100.0)
    fit_model.parameters["sigma_c_mean"].bounds = (
        0.5 * float(np.min(stress)),
        1.5 * float(np.min(stress)),
    )
    fit_model.parameters["sigma_c_std"].bounds = (1e-3, 5.0)
    fit_model.parameters["n_fluid"].bounds = (0.5, 5.0)

    fit_model.fit(shear_rate, stress, test_mode="flow_curve", method="scipy")

    fitted = {
        name: float(fit_model.parameters.get_value(name))
        for name in ("mu", "tau_pl", "sigma_c_mean", "sigma_c_std", "n_fluid")
    }
    sigma_y = fitted["sigma_c_mean"]
    sigma0 = 1.2 * sigma_y  # above yield → unbounded creep

    # Now simulate creep with the fitted parameters.
    creep_model = LatticeEPM(
        L=32,
        dt=0.01,
        mu=fitted["mu"],
        tau_pl=fitted["tau_pl"],
        sigma_c_mean=fitted["sigma_c_mean"],
        sigma_c_std=fitted["sigma_c_std"],
        n_fluid=fitted["n_fluid"],
        fluidity_form="overstress",
    )
    t = np.linspace(0.5, 10.0, 30)
    gamma = _simulate_creep(creep_model, t, sigma0, seed=42)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = "time_s,strain"
    np.savetxt(
        out_path,
        np.column_stack([t, gamma]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6e",
    )

    return {
        "source_flow_curve": str(EMULSION_CSV.relative_to(REPO)),
        "fitted_params": fitted,
        "applied_stress": sigma0,
        "path": out_path,
    }


def main() -> None:
    synth_meta = generate_synthetic_epm_creep(HERE / "epm_creep_synthetic.csv")
    print("[A] Synthetic EPM creep:")
    print(f"    -> {synth_meta['path']}")
    print(f"    ground truth: {synth_meta['ground_truth_params']}")
    print(f"    stress levels: {synth_meta['stress_levels']}")

    hb_meta = generate_hb_calibrated_emulsion_creep(
        HERE / "emulsion_creep_hb_calibrated.csv"
    )
    print("\n[B] HB-calibrated emulsion creep:")
    print(f"    -> {hb_meta['path']}")
    print(f"    source flow curve: {hb_meta['source_flow_curve']}")
    print(f"    fitted params   : {hb_meta['fitted_params']}")
    print(f"    applied σ₀      : {hb_meta['applied_stress']:.4g} Pa")


if __name__ == "__main__":
    main()
