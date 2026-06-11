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
        return {"num_warmup": 100, "num_samples": 100, "num_chains": 1}
    return {"num_warmup": 500, "num_samples": 1000, "num_chains": 4}


# =============================================================================
# Synthetic Data Generators (fallback when experimental files are missing)
# =============================================================================


def _generate_synthetic_creep(
    max_points: int | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell creep compliance J(t) = (1 + k_d·t) / G0 with 3% relative noise.

    True params: G0=500 Pa, k_d=0.2 1/s → τ=5 s, η₀=2500 Pa·s.
    Time window spans the elastic intercept (t << τ) AND viscous slope (t >> τ)
    so that both G0 and k_d are individually identifiable.
    """
    G0_true, k_d_true = 500.0, 0.2
    tau = 1.0 / k_d_true
    t = np.logspace(np.log10(0.05 * tau), np.log10(30.0 * tau), 200)
    J_true = (1.0 + k_d_true * t) / G0_true
    rng = np.random.default_rng(seed)
    J = J_true * (1.0 + 0.03 * rng.standard_normal(len(t)))
    J = np.maximum(J, 1e-12)
    if max_points is not None and len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
        t, J = t[idx], J[idx]
    return t, J


def _generate_synthetic_startup(
    gamma_dot: float = 1.0, max_points: int | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Maxwell startup shear σ(t) = η₀·γ̇·(1 − exp(−k_d·t)) with 3% noise.

    True params: G0=100 Pa, k_d=0.5 1/s → τ=2 s, η₀=200 Pa·s.
    """
    G0_true, k_d_true = 100.0, 0.5
    tau = 1.0 / k_d_true
    t = np.logspace(np.log10(0.05 * tau), np.log10(20.0 * tau), 200)
    eta0 = G0_true / k_d_true
    sigma_true = eta0 * gamma_dot * (1.0 - np.exp(-k_d_true * t))
    rng = np.random.default_rng(seed)
    sigma = sigma_true * (1.0 + 0.03 * rng.standard_normal(len(t)))
    sigma = np.maximum(sigma, 0.0)
    if max_points is not None and len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
        t, sigma = t[idx], sigma[idx]
    return t, sigma


def _generate_synthetic_laos(
    omega: float = 1.0,
    gamma_0: float = 0.5,
    max_points: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """VLBLocal LAOS stress with 3% additive noise.

    True params: G0=100 Pa, k_d=2.0 1/s.
    Uses VLBLocal.predict() directly so data and fitting model are self-consistent.
    """
    from rheojax.models import VLBLocal

    G0_true, k_d_true = 100.0, 2.0
    T_cycle = 2.0 * np.pi / omega
    t = np.linspace(0.0, 5.0 * T_cycle, 500)
    strain = gamma_0 * np.sin(omega * t)

    model_gen = VLBLocal()
    model_gen.parameters.set_value("G0", G0_true)
    model_gen.parameters.set_value("k_d", k_d_true)
    stress_true = np.asarray(
        model_gen.predict(t, test_mode="laos", gamma_0=gamma_0, omega=omega)
    )

    rng = np.random.default_rng(seed)
    noise_std = 0.03 * float(np.std(stress_true))
    stress = stress_true + noise_std * rng.standard_normal(len(t))
    if max_points is not None and len(t) > max_points:
        idx = np.linspace(0, len(t) - 1, max_points, dtype=int)
        t, strain, stress = t[idx], strain[idx], stress[idx]
    return t, strain, stress


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


def load_synthetic_creep(
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Maxwell creep compliance data.

    Used in place of real creep data to ensure both G0 and k_d are individually
    identifiable: the time window spans both the elastic intercept (t << τ) and
    the viscous flow regime (t >> τ) so the NUTS posterior converges cleanly.

    Args:
        max_points: If set, subsample to this many points.

    Returns:
        (time, J) in (s, 1/Pa). True: G0=500 Pa, k_d=0.2 1/s.
    """
    print("[SYNTHETIC] True params: G0=500.0 Pa, k_d=0.2 1/s  (τ=5.0 s, η₀=2500 Pa·s)")
    return _generate_synthetic_creep(max_points=max_points)


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
        import warnings as _w
        _w.warn(
            f"PNAS data not found at: {fpath.resolve()}\n"
            "Falling back to synthetic Maxwell startup data (G0=100 Pa, k_d=0.5 1/s).",
            UserWarning,
            stacklevel=2,
        )
        t, sigma = _generate_synthetic_startup(
            gamma_dot=gamma_dot, max_points=max_points
        )
        print(
            f"[SYNTHETIC] True params: G0=100.0 Pa, k_d=0.5 1/s"
            f"  (γ̇={gamma_dot} 1/s, σ_ss={100.0 / 0.5 * gamma_dot:.1f} Pa)"
        )
        return t, sigma

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
        raise ValueError(
            f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}"
        )

    sheet = omega_sheets[omega]
    data_dir = Path(__file__).parent / ".." / "data" / "ikh"
    fpath = data_dir / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    if not fpath.exists():
        import warnings as _w
        _w.warn(
            f"PNAS data not found at: {fpath.resolve()}\n"
            "Falling back to synthetic Maxwell LAOS data (G0=100 Pa, k_d=2.0 1/s).",
            UserWarning,
            stacklevel=2,
        )
        # Map strain_amplitude_index 0..11 → γ₀ 0.05..0.60 (logarithmic spacing)
        gamma_0 = 0.05 * 10 ** (strain_amplitude_index / 11.0 * np.log10(12.0))
        t, s_arr, stress = _generate_synthetic_laos(
            omega=omega, gamma_0=gamma_0, max_points=max_points
        )
        print(
            f"[SYNTHETIC] True params: G0=100.0 Pa, k_d=2.0 1/s"
            f"  (ω={omega} rad/s, γ₀={gamma_0:.3f})"
        )
        return t, s_arr, stress

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
# Fit Quality Guard
# =============================================================================


def check_nlsq_fit_quality(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    test_mode: str,
    r2_threshold: float = 0.90,
    noise_fraction: float | None = None,
    noise_sigma: float | None = None,
    n_params: int = 2,
    **predict_kwargs: Any,
) -> float:
    """Compute R² (and optionally reduced χ²) before Bayesian inference.

    A poor NLSQ warm-start (R² < threshold) causes NUTS to explore unphysical
    parameter space: model-data mismatch flattens the likelihood, so the sampler
    drifts far from the warm-start values.

    Args:
        model: Fitted VLBLocal (or any VLBBase subclass).
        X: Independent variable (gamma_dot, omega, time, …).
        y: Observed data — real or complex G*.
        test_mode: Protocol string ('flow_curve', 'relaxation', …).
        r2_threshold: Warn threshold (default 0.90).
        noise_fraction: Relative noise level ε (e.g. 0.03 for 3%).  When set,
            computes reduced chi-squared χ²_red = Σ[(y-ŷ)²/(ε|y|)²] / (N-k).
            Use for protocols with proportional noise (creep, startup, SAOS).
        noise_sigma: Absolute noise std σ (e.g. 0.03*std(y)).  When set,
            computes χ²_red = Σ[(y-ŷ)²/σ²] / (N-k).
            Use for LAOS (additive noise uniform across all time points).
            Takes precedence over noise_fraction when both are supplied.
        n_params: Number of free parameters k for χ²_red denominator (default 2).
        **predict_kwargs: Forwarded to model.predict()
            (e.g. gamma_dot=, sigma_applied=, omega=).

    Returns:
        R² value (float). Raises UserWarning if R² < r2_threshold.
    """
    import warnings as _warnings

    y_pred = np.asarray(model.predict(X, test_mode=test_mode, **predict_kwargs))
    y_arr = np.asarray(y)

    if np.iscomplexobj(y_arr):
        y_ref = np.abs(y_arr)
        y_pred_ref = np.abs(y_pred)
    else:
        y_ref = np.asarray(y_arr, dtype=float)
        y_pred_ref = np.asarray(y_pred, dtype=float)

    ss_res = np.sum((y_ref - y_pred_ref) ** 2)
    ss_tot = np.sum((y_ref - np.mean(y_ref)) ** 2)
    r2 = float(1.0 - ss_res / max(float(ss_tot), 1e-12))

    if r2 < r2_threshold:
        _warnings.warn(
            f"\n⚠  Poor NLSQ fit quality (R²={r2:.3f} < {r2_threshold}, "
            f"test_mode='{test_mode}').\n"
            f"   NUTS may explore unphysical parameter space from this warm-start.\n"
            f"   Remedies: (1) use synthetic data matching the model physics; "
            f"(2) switch to VLBMultiNetwork for multi-mode materials; "
            f"(3) check data units or protocol mismatch.",
            UserWarning,
            stacklevel=2,
        )
    else:
        print(f"✓  NLSQ fit quality R²={r2:.4f} ≥ {r2_threshold} — safe to proceed with Bayesian")

    if noise_sigma is not None or noise_fraction is not None:
        n = len(y_ref)
        dof = max(n - n_params, 1)
        if noise_sigma is not None:
            sigma_i = float(noise_sigma)
            noise_label = f"σ={noise_sigma:.3g} (absolute)"
        else:
            sigma_i = noise_fraction * np.maximum(np.abs(y_ref), 1e-30)  # type: ignore[operator]
            noise_label = f"ε={noise_fraction:.1%} (relative)"  # type: ignore[str-format]
        chi2 = float(np.sum(((y_ref - y_pred_ref) / sigma_i) ** 2))
        chi2_red = chi2 / dof
        if 0.5 <= chi2_red <= 2.0:
            print(f"✓  χ²_red = {chi2_red:.3f} ≈ 1.0 — fit consistent with {noise_label} noise")
        elif chi2_red > 2.0:
            print(
                f"⚠  χ²_red = {chi2_red:.3f} >> 1 — systematic deviations "
                f"(model–data mismatch or noise underestimated)"
            )
        else:
            print(
                f"⚠  χ²_red = {chi2_red:.3f} << 1 — possible overfitting "
                f"or noise overestimated"
            )

    return r2


# =============================================================================
# Diagnostics
# =============================================================================


def print_convergence(result: Any, param_names: list[str]) -> bool:
    """Print convergence diagnostics from a BayesianResult.

    Thresholds are scaled to FAST_MODE (1 chain, 100 samples) vs full mode
    (4 chains, 1000 samples).  In FAST_MODE R-hat > 1.10 or ESS < 40 flags
    a parameter; in full mode the stricter R-hat > 1.05 or ESS < 400 applies.

    Returns:
        True if all parameters pass the mode-appropriate thresholds.
    """
    diag = result.diagnostics

    rhat_max = 1.10 if FAST_MODE else 1.05
    ess_min = 40 if FAST_MODE else 400

    print("Convergence Diagnostics")
    print("=" * 50)
    print(f"{'Parameter':>10s}  {'R-hat':>8s}  {'ESS':>8s}")
    print("-" * 50)

    all_ok = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))
        flag = ""
        if r_hat > rhat_max or ess < ess_min:
            flag = " *"
            all_ok = False
        print(f"{p:>10s}  {r_hat:8.4f}  {ess:8.0f}{flag}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")
    print(f"Convergence: {'PASSED' if all_ok else 'CHECK REQUIRED'}")
    if FAST_MODE:
        print(
            "\nNote: FAST_MODE uses 1 chain (100 samples). R-hat is a split-chain"
            f"\n      estimate — less reliable than multi-chain. Thresholds relaxed to"
            f"\n      R-hat < {rhat_max}, ESS > {ess_min}. Use FAST_MODE=False for"
            "\n      publication-quality diagnostics (4 chains, 1000 samples)."
        )
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
        idata,
        var_names=param_names,
        combined=True,
        hdi_prob=0.95,
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
    nlsq = {
        p: float(getattr(model, p, model.parameters.get_value(p))) for p in param_names
    }
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
        rows.append(
            {
                "parameter": p,
                "nlsq": nlsq.get(p),
                "posterior_mean": float(np.mean(samples)),
                "posterior_median": float(np.median(samples)),
                "ci_2.5": float(np.percentile(samples, 2.5)),
                "ci_97.5": float(np.percentile(samples, 97.5)),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    print(f"Saved to {output_dir}/")


def save_figure(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save a figure to the figures/ subdirectory."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig.savefig(os.path.join(fig_dir, name), dpi=150, bbox_inches="tight")
