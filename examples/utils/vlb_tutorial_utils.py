"""Shared utilities for VLB (Vernerey-Long-Brighenti) NLSQ-to-NUTS tutorial notebooks.

Provides data loaders, diagnostics, posterior predictive plotting, and result
saving for 6 protocols: flow_curve, creep, stress_relaxation, startup, SAOS, LAOS.

Data Sources:
- ec_shear_viscosity_07-00.csv: Ethyl cellulose 7% solution flow curve (pyRheo demos)
- creep_ps190_data.csv: Polystyrene creep at 190 C (pyRheo demos)
- stressrelaxation_liquidfoam_data.csv: Liquid foam relaxation (pyRheo demos)
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 startup + LAOS
- epstein.csv: Epstein et al. metal-organic coordination network SAOS
"""

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# FAST_MODE Configuration
# =============================================================================
# True  → CI / quick demo (~1-2 min per notebook, 1 chain, few samples)
# False → publication-quality (~10-30 min, 4 chains, more samples)

FAST_MODE = True


def get_fast_mode() -> bool:
    """Return current FAST_MODE setting."""
    return FAST_MODE


def get_bayesian_config() -> dict[str, int]:
    """Return MCMC configuration based on FAST_MODE."""
    if FAST_MODE:
        return {"num_warmup": 50, "num_samples": 100, "num_chains": 1}
    return {"num_warmup": 500, "num_samples": 1000, "num_chains": 4}


# =============================================================================
# Data Loaders
# =============================================================================


def load_ec_flow_curve(
    concentration: str = "07-00",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ethyl cellulose solution flow curve.

    European CSV format (semicolons, comma decimals).

    Args:
        concentration: EC concentration string (e.g., "07-00" for 7 wt%).

    Returns:
        (shear_rate, stress, viscosity) in (1/s, Pa, Pa*s).
    """
    data_dir = Path(__file__).parent / ".." / "data" / "flow" / "solutions"
    fpath = data_dir / f"ec_shear_viscosity_{concentration}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"EC data not found at: {fpath.resolve()}")

    with open(fpath, encoding="latin-1") as f:
        lines = f.readlines()

    # Skip header (2 rows: column names + units)
    rows = []
    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(";")
        row = [float(p.replace(",", ".")) for p in parts[:3]]
        rows.append(row)

    arr = np.array(rows)
    gamma_dot = arr[:, 0]
    stress = arr[:, 1]
    viscosity = arr[:, 2]

    idx = np.argsort(gamma_dot)
    return gamma_dot[idx], stress[idx], viscosity[idx]


def load_epstein_saos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAOS data from Epstein metal-organic coordination network.

    Returns:
        (omega, G_prime, G_double_prime) in (rad/s, Pa, Pa).
    """
    data_dir = Path(__file__).parent / ".." / "data" / "oscillation" / "metal_networks"
    fpath = data_dir / "epstein.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Epstein data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    omega = df.iloc[:, 0].values.astype(float)
    G_prime = df.iloc[:, 1].values.astype(float)
    G_double_prime = df.iloc[:, 2].values.astype(float)

    idx = np.argsort(omega)
    return omega[idx], G_prime[idx], G_double_prime[idx]


def load_foam_relaxation(
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load liquid foam stress relaxation data.

    Args:
        max_points: If set, subsample to this many points.

    Returns:
        (time, G_t) in (s, Pa).
    """
    data_dir = Path(__file__).parent / ".." / "data" / "relaxation" / "foams"
    fpath = data_dir / "stressrelaxation_liquidfoam_data.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Foam data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    # Remove any non-positive modulus values
    mask = G_t > 0
    time, G_t = time[mask], G_t[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, G_t = time[indices], G_t[indices]

    return time, G_t


def load_ps_creep(
    temperature: int = 190,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load polystyrene creep compliance data.

    Args:
        temperature: Temperature in Celsius (130, 145, 160, 175, 190).
        max_points: If set, subsample to this many points.

    Returns:
        (time, J) in (s, 1/Pa).
    """
    data_dir = Path(__file__).parent / ".." / "data" / "creep" / "polymers"
    fpath = data_dir / f"creep_ps{temperature}_data.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"PS creep data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    J = df.iloc[:, 1].values.astype(float)

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, J = time[indices], J[indices]

    return time, J


def load_pnas_startup(
    gamma_dot: float = 1.0,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load startup shear data from PNAS Digital Rheometer Twin dataset.

    Source: PNAS 2022 Digital Rheometer Twin.

    Args:
        gamma_dot: Shear rate. Available: 0.056, 0.32, 1, 56.2, 100.
        max_points: If set, subsample to this many points.

    Returns:
        (time, stress) in (s, Pa).
    """
    rate_sheets = {
        0.056: "StartUp_0.056",
        0.32: "StartUp_0.32",
        1.0: "StartUp_1",
        56.2: "StartUp_56.2",
        100.0: "StartUp_100",
    }

    available = list(rate_sheets.keys())
    closest = min(available, key=lambda x: abs(x - gamma_dot))
    sheet = rate_sheets[closest]

    data_dir = Path(__file__).parent / ".." / "data" / "ikh"
    fpath = data_dir / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    if not fpath.exists():
        raise FileNotFoundError(f"PNAS data not found at: {fpath.resolve()}")

    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    data = df.iloc[3:, [0, 1]].dropna().astype(float)
    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, stress = time[indices], stress[indices]

    return time, stress


def load_pnas_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LAOS data from PNAS Digital Rheometer Twin dataset.

    Source: PNAS 2022 Digital Rheometer Twin.

    Args:
        omega: Angular frequency (1, 3, or 5 rad/s).
        strain_amplitude_index: Index for strain amplitude (0-11, increasing).
        max_points: If set, subsample to this many points.

    Returns:
        (time, strain, stress) in (s, -, Pa).
    """
    omega_sheets = {1.0: "LAOS_w1", 3.0: "LAOS_w3", 5.0: "LAOS_w5"}
    if omega not in omega_sheets:
        raise ValueError(f"omega must be 1, 3, or 5, got {omega}")
    if strain_amplitude_index not in range(12):
        raise ValueError(f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}")

    sheet = omega_sheets[omega]
    data_dir = Path(__file__).parent / ".." / "data" / "ikh"
    fpath = data_dir / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    if not fpath.exists():
        raise FileNotFoundError(f"PNAS data not found at: {fpath.resolve()}")

    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    col_t = strain_amplitude_index * 4
    col_strain = strain_amplitude_index * 4 + 1
    col_stress = strain_amplitude_index * 4 + 2

    data = df.iloc[3:, [col_t, col_strain, col_stress]].dropna().astype(float)
    time = data.iloc[:, 0].values
    strain = data.iloc[:, 1].values
    stress = data.iloc[:, 2].values

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, strain, stress = time[indices], strain[indices], stress[indices]

    return time, strain, stress


# =============================================================================
# Diagnostics
# =============================================================================


def print_convergence(result: Any, param_names: list[str]) -> bool:
    """Print convergence diagnostics from a BayesianResult.

    Returns:
        True if all R-hat < 1.05 and ESS > 100.
    """
    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 50)
    print(f"{'Parameter':>10s}  {'R-hat':>8s}  {'ESS':>8s}")
    print("-" * 50)

    all_ok = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))
        flag = ""
        if r_hat > 1.05 or ess < 100:
            flag = " *"
            all_ok = False
        print(f"{p:>10s}  {r_hat:8.4f}  {ess:8.0f}{flag}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")
    print(f"Convergence: {'PASSED' if all_ok else 'CHECK REQUIRED'}")
    return all_ok


def print_parameter_table(
    param_names: list[str],
    nlsq_vals: dict[str, float],
    posterior: dict[str, Any],
    true_vals: dict[str, float] | None = None,
) -> None:
    """Print NLSQ vs Bayesian parameter comparison."""
    header = f"{'Param':>8s}  "
    if true_vals:
        header += f"{'True':>10s}  "
    header += f"{'NLSQ':>10s}  {'Bayes (median)':>14s}  {'95% CI':>20s}"
    print("Parameter Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for p in param_names:
        samples = np.array(posterior[p])
        med = np.median(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        row = f"{p:>8s}  "
        if true_vals and p in true_vals:
            row += f"{true_vals[p]:10.4g}  "
        row += f"{nlsq_vals[p]:10.4g}  {med:14.4g}  [{lo:.4g}, {hi:.4g}]"
        print(row)


# =============================================================================
# Plotting
# =============================================================================


def plot_trace_and_forest(
    result: Any,
    param_names: list[str],
    figsize_trace: tuple[float, float] = (12, 5),
    figsize_forest: tuple[float, float] = (10, 3),
) -> tuple[plt.Figure, plt.Figure]:
    """Plot ArviZ trace and forest plots.

    Returns:
        (trace_fig, forest_fig).
    """
    import arviz as az

    idata = result.to_inference_data()

    axes = az.plot_trace(idata, var_names=param_names, figsize=figsize_trace)
    fig_trace = axes.ravel()[0].figure
    fig_trace.suptitle("Trace Plots", fontsize=14, y=1.02)
    plt.tight_layout()

    axes = az.plot_forest(
        idata, var_names=param_names, combined=True, hdi_prob=0.95,
        figsize=figsize_forest,
    )
    fig_forest = axes.ravel()[0].figure
    plt.tight_layout()

    return fig_trace, fig_forest


# =============================================================================
# Save / Load
# =============================================================================


def get_output_dir(protocol: str) -> str:
    """Return path to output directory for a given protocol.

    Creates directory if it doesn't exist.
    """
    out = Path(__file__).parent / ".." / "outputs" / "vlb" / protocol
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def save_results(
    output_dir: str,
    model: Any,
    result: Any | None = None,
    param_names: list[str] | None = None,
    extra_meta: dict | None = None,
) -> None:
    """Save NLSQ params, posterior samples, and summary CSV.

    Args:
        output_dir: Directory to write into.
        model: Fitted VLBLocal model.
        result: BayesianResult (optional).
        param_names: Parameter names.
        extra_meta: Extra metadata for the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if param_names is None:
        param_names = ["G0", "k_d"]

    # NLSQ parameters
    nlsq = {p: float(getattr(model, p, model.parameters.get_value(p))) for p in param_names}
    if extra_meta:
        nlsq.update(extra_meta)
    with open(os.path.join(output_dir, "fitted_params_nlsq.json"), "w") as f:
        json.dump(nlsq, f, indent=2)

    if result is None:
        return

    # Posterior samples
    posterior = result.posterior_samples
    np.savez(
        os.path.join(output_dir, "posterior_samples.npz"),
        **{k: np.array(v) for k, v in posterior.items()},
    )

    # Summary CSV
    rows = []
    for p in param_names:
        samples = np.array(posterior[p])
        rows.append({
            "parameter": p,
            "nlsq": nlsq.get(p),
            "posterior_mean": float(np.mean(samples)),
            "posterior_median": float(np.median(samples)),
            "ci_2.5": float(np.percentile(samples, 2.5)),
            "ci_97.5": float(np.percentile(samples, 97.5)),
        })
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"Saved to {output_dir}/")


def save_figure(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save a figure to the figures/ subdirectory."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
