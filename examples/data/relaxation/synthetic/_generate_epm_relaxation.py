"""Generate reference relaxation datasets for examples/epm/05_epm_relaxation.ipynb.

Two CSVs are produced:

* ``epm_relaxation_synthetic.csv`` — pure ground-truth stress relaxation from
  ``LatticeEPM`` with known overstress-form parameters, sampled on a
  uniformly-spaced time grid (the kernel's assumption — see notes in
  ``_jit_relaxation_kernel``). The notebook uses this dataset to demonstrate
  that the NLSQ fit recovers the generating parameters (self-consistency).

* ``emulsion_relaxation_hb_calibrated.csv`` — relaxation trajectory generated
  from the *same* HB parameters that were fitted to a real emulsion flow
  curve (``examples/data/flow/emulsions/0.74.csv``, volume fraction φ=0.74)
  in ``examples/data/creep/synthetic/_generate_epm_creep.py``. The HB
  parameters are real; the step-strain protocol is simulated. The notebook
  uses this as a "real-material calibrated" yield-stress relaxation
  demonstration.

Regenerate with:
    cd /Users/b80985/Projects/rheojax
    uv run python examples/data/relaxation/synthetic/_generate_epm_relaxation.py

Note: both datasets are on *uniformly-spaced* time grids. The EPM relaxation
kernel assumes uniform spacing and will silently mis-map log-spaced inputs
onto the wrong physical time range. This is a known limitation and will be
addressed in a separate fix.
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


def _simulate_relaxation(
    model: LatticeEPM,
    time: np.ndarray,
    strain_step: float,
    seed: int,
) -> np.ndarray:
    """Return G(t) for the given LatticeEPM at a step strain."""
    data = RheoData(
        x=jnp.asarray(time),
        y=jnp.zeros_like(jnp.asarray(time)),
        initial_test_mode="relaxation",
        metadata={"gamma": float(strain_step)},
    )
    return np.asarray(model.predict(data, smooth=True, seed=seed).y)


def generate_synthetic_epm_relaxation(out_path: Path) -> dict:
    """Ground-truth EPM relaxation at an above-yield step strain."""
    mu = 1.0
    tau_pl = 1.0
    sigma_c_mean = 0.3
    sigma_c_std = 0.05
    n_fluid = 2.0
    strain_step = 1.0  # mu·γ = 1.0 >> σ_c_mean = 0.3 → above yield

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

    t = np.linspace(0.01, 5.0, 50)  # uniform grid (kernel assumption)
    g = _simulate_relaxation(model, t, strain_step, seed=42)

    header = "time_s,modulus_Pa"
    np.savetxt(
        out_path,
        np.column_stack([t, g]),
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
        "strain_step": strain_step,
        "path": out_path,
    }


def generate_hb_calibrated_emulsion_relaxation(out_path: Path) -> dict:
    """Relaxation from the same HB fit to the real phi=0.74 emulsion flow curve.

    Mirrors the creep companion dataset. Re-fits the flow curve to extract HB
    parameters and then simulates the step-strain response.
    """
    raw = np.loadtxt(EMULSION_CSV, delimiter=",", skiprows=1)
    shear_rate = raw[:, 0]
    stress = raw[:, 1]

    fit_model = LatticeEPM(
        L=16,
        dt=0.01,
        mu=1.0,
        tau_pl=1.0,
        sigma_c_mean=float(np.min(stress)) * 0.9,
        sigma_c_std=0.1 * float(np.min(stress)),
        n_fluid=2.0,
        fluidity_form="overstress",
    )
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
    mu_fit = fitted["mu"]
    # Choose a step strain so that mu·γ0 = 1.5 · σ_y → clearly above yield.
    strain_step = 1.5 * sigma_y / mu_fit

    relax_model = LatticeEPM(
        L=32,
        dt=0.01,
        mu=fitted["mu"],
        tau_pl=fitted["tau_pl"],
        sigma_c_mean=fitted["sigma_c_mean"],
        sigma_c_std=fitted["sigma_c_std"],
        n_fluid=fitted["n_fluid"],
        fluidity_form="overstress",
    )
    t = np.linspace(0.01, 5.0, 50)  # uniform grid
    g = _simulate_relaxation(relax_model, t, strain_step, seed=42)

    header = "time_s,modulus_Pa"
    np.savetxt(
        out_path,
        np.column_stack([t, g]),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.6e",
    )
    return {
        "source_flow_curve": str(EMULSION_CSV.relative_to(REPO)),
        "fitted_params": fitted,
        "strain_step": strain_step,
        "path": out_path,
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

    synth = generate_synthetic_epm_relaxation(HERE / "epm_relaxation_synthetic.csv")
    print("[A] Synthetic EPM relaxation:")
    print(f"    -> {synth['path']}")
    print(f"    truth  : {synth['ground_truth_params']}")
    print(f"    strain : {synth['strain_step']}")

    hb = generate_hb_calibrated_emulsion_relaxation(
        HERE / "emulsion_relaxation_hb_calibrated.csv"
    )
    print("\n[B] HB-calibrated emulsion relaxation:")
    print(f"    -> {hb['path']}")
    print(f"    source : {hb['source_flow_curve']}")
    print(f"    fitted : {hb['fitted_params']}")
    print(f"    strain : {hb['strain_step']:.4g}")


if __name__ == "__main__":
    main()
