"""Shared utilities for HL tutorial notebooks.

Provides consistent data loading, result saving, and diagnostic printing
across all HL (Hébraud-Lequeux) protocol tutorials.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def load_emulsion_flow_curve(phi: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    """Load emulsion flow curve data for a given volume fraction.

    Args:
        phi: Volume fraction (0.69, 0.70, 0.72, 0.74, 0.76, or 0.80).

    Returns:
        Tuple of (shear_rate, stress) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "flow" / "emulsions" / f"{phi:.2f}.csv"

    if not data_path.exists():
        # Try alternative path for Colab
        data_path = Path(f"{phi:.2f}.csv")
        if not data_path.exists():
            raise FileNotFoundError(
                f"Emulsion data not found for phi={phi}. "
                f"Expected at: examples/data/flow/emulsions/{phi:.2f}.csv"
            )

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    gamma_dot = raw[:, 0]
    stress = raw[:, 1]

    return gamma_dot, stress


def load_clay_relaxation(aging_time: int = 600) -> tuple[np.ndarray, np.ndarray]:
    """Load Laponite clay relaxation data for a given aging time.

    Args:
        aging_time: Waiting time in seconds (600, 1200, 1800, 2400, or 3600).

    Returns:
        Tuple of (time, relaxation_modulus) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "relaxation" / "clays" / f"rel_lapo_{aging_time}.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Clay relaxation data not found for aging_time={aging_time}. "
            f"Expected at: examples/data/relaxation/clays/rel_lapo_{aging_time}.csv"
        )

    # Tab-separated with header "Time\tRelaxation Modulus"
    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    time = raw[:, 0]
    G_t = raw[:, 1]

    return time, G_t


def load_polymer_creep(temp: int = 145) -> tuple[np.ndarray, np.ndarray]:
    """Load polystyrene creep compliance data for a given temperature.

    Args:
        temp: Temperature in °C (130, 145, 160, 175, or 190).

    Returns:
        Tuple of (time, compliance) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "creep" / "polymers" / f"creep_ps{temp}_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Polymer creep data not found for T={temp}°C. "
            f"Expected at: examples/data/creep/polymers/creep_ps{temp}_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    compliance = raw[:, 1]

    return time, compliance


def generate_hl_synthetic(
    params: dict[str, float],
    protocol: str,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic HL data from calibrated parameters.

    Args:
        params: Dictionary of HL parameters (alpha, tau, sigma_c).
        protocol: Protocol type ('startup', 'saos', 'laos').
        n_points: Number of data points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.
        **kwargs: Protocol-specific parameters:
            - startup: gamma_dot, t_end
            - saos: omega_min, omega_max
            - laos: gamma0, omega, t_end

    Returns:
        Tuple of (x, y) arrays.
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.hl import HebraudLequeux

    jax, jnp = safe_import_jax()
    rng = np.random.default_rng(seed)

    # Create model with calibrated parameters
    model = HebraudLequeux()
    model.parameters.set_value("alpha", params.get("alpha", 0.3))
    model.parameters.set_value("tau", params.get("tau", 1.0))
    model.parameters.set_value("sigma_c", params.get("sigma_c", 1.0))

    if protocol == "startup":
        gamma_dot = kwargs.get("gamma_dot", 1.0)
        t_end = kwargs.get("t_end", 10.0)
        time = np.linspace(0.01, t_end, n_points)

        # Fit with dummy data to set test_mode
        model.fit(time, np.ones_like(time), test_mode="startup", gdot=gamma_dot, max_iter=1)
        # Reset parameters
        model.parameters.set_value("alpha", params.get("alpha", 0.3))
        model.parameters.set_value("tau", params.get("tau", 1.0))
        model.parameters.set_value("sigma_c", params.get("sigma_c", 1.0))

        stress_clean = model.predict(time)
        noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
        stress = stress_clean + noise

        return time, stress

    elif protocol == "saos":
        omega_min = kwargs.get("omega_min", 0.01)
        omega_max = kwargs.get("omega_max", 100.0)
        omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)

        # For SAOS, we predict G* (complex modulus)
        # HL doesn't have a direct SAOS prediction in the current API,
        # so we generate simplified synthetic data based on HL physics
        alpha = params.get("alpha", 0.3)
        tau = params.get("tau", 1.0)
        sigma_c = params.get("sigma_c", 1.0)

        # Simplified HL SAOS response (elastic plateau modulated by alpha)
        G0 = sigma_c  # Approximate elastic modulus
        omega_tau = omega * tau

        # G' ~ G0 * (omega*tau)^2 / (1 + (omega*tau)^2) but modified by alpha
        G_prime = G0 * (omega_tau ** (2 * alpha)) / (1 + omega_tau ** (2 * alpha))
        G_double_prime = G0 * omega_tau ** alpha / (1 + omega_tau ** (2 * alpha))

        # Add noise
        noise_p = rng.normal(0, noise_level * np.mean(G_prime), size=G_prime.shape)
        noise_pp = rng.normal(0, noise_level * np.mean(G_double_prime), size=G_double_prime.shape)

        G_star = np.column_stack([G_prime + noise_p, G_double_prime + noise_pp])
        return omega, G_star

    elif protocol == "laos":
        gamma0 = kwargs.get("gamma0", 0.1)
        omega = kwargs.get("omega", 1.0)
        t_end = kwargs.get("t_end", 20.0)
        time = np.linspace(0.01, t_end, n_points)

        # Fit with dummy data to set test_mode
        model.fit(
            time, np.ones_like(time), test_mode="laos", gamma0=gamma0, omega=omega, max_iter=1
        )
        # Reset parameters
        model.parameters.set_value("alpha", params.get("alpha", 0.3))
        model.parameters.set_value("tau", params.get("tau", 1.0))
        model.parameters.set_value("sigma_c", params.get("sigma_c", 1.0))

        stress_clean = model.predict(time)
        noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
        stress = stress_clean + noise

        return time, stress

    else:
        raise ValueError(f"Unsupported protocol: {protocol}")


def save_hl_results(
    model: Any,
    result: Any,
    output_dir: str,
    protocol: str,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted HL model.
        result: Bayesian fit result with posterior_samples attribute.
        output_dir: Directory to save results.
        protocol: Protocol name for labeling files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save NLSQ point estimates
    nlsq_params = {}
    for name in ["alpha", "tau", "sigma_c"]:
        nlsq_params[name] = float(model.parameters.get_value(name))

    with open(os.path.join(output_dir, f"nlsq_params_{protocol}.json"), "w") as f:
        json.dump(nlsq_params, f, indent=2)

    # Save posterior samples
    posterior = result.posterior_samples
    posterior_dict = {k: np.array(v).tolist() for k, v in posterior.items()}
    with open(os.path.join(output_dir, f"posterior_{protocol}.json"), "w") as f:
        json.dump(posterior_dict, f)

    print(f"Results saved to {output_dir}/")
    print(f"  nlsq_params_{protocol}.json: {len(nlsq_params)} parameters")
    print(f"  posterior_{protocol}.json: {len(list(posterior.values())[0])} draws")


def load_hl_parameters(
    protocol: str,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Load previously calibrated HL parameters.

    Args:
        protocol: Protocol name (e.g., 'flow_curve').
        output_dir: Directory with saved results. Defaults to ../outputs/hl/.

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    if output_dir is None:
        module_dir = Path(__file__).parent
        output_dir = module_dir / ".." / "outputs" / "hl" / protocol

    param_file = Path(output_dir) / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for protocol '{protocol}'. "
            f"Run notebook 01_hl_flow_curve.ipynb first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


def print_convergence_summary(
    result: Any,
    param_names: list[str] | None = None,
) -> bool:
    """Print formatted convergence diagnostics table.

    Args:
        result: Bayesian fit result with diagnostics attribute.
        param_names: List of parameter names to report. Defaults to HL params.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
    if param_names is None:
        param_names = ["alpha", "tau", "sigma_c"]

    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 50)
    print(f"{'Parameter':>12s}  {'R-hat':>8s}  {'ESS':>8s}  {'Status':>8s}")
    print("-" * 50)

    all_pass = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))

        # Check thresholds
        r_hat_ok = r_hat < 1.05
        ess_ok = ess > 100

        if r_hat_ok and ess_ok:
            status = "PASS"
        else:
            status = "CHECK"
            all_pass = False

        print(f"{p:>12s}  {r_hat:8.4f}  {ess:8.0f}  {status:>8s}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")

    if n_div > 0:
        all_pass = False

    if all_pass:
        print("\nAll convergence criteria PASSED")
    else:
        print("\nCHECK REQUIRED: Increase num_warmup/num_samples or check warm-start")

    return all_pass


def print_parameter_comparison(
    model: Any,
    posterior: dict[str, np.ndarray],
    param_names: list[str] | None = None,
) -> None:
    """Print NLSQ vs Bayesian parameter comparison table.

    Args:
        model: Fitted model with parameters attribute.
        posterior: Dictionary of posterior samples.
        param_names: List of parameter names to compare. Defaults to HL params.
    """
    if param_names is None:
        param_names = ["alpha", "tau", "sigma_c"]

    print("\nParameter Comparison: NLSQ vs Bayesian")
    print("=" * 65)
    print(f"{'Param':>12s}  {'NLSQ':>12s}  {'Median':>12s}  {'95% CI':>24s}")
    print("-" * 65)

    for name in param_names:
        nlsq_val = model.parameters.get_value(name)
        samples = posterior[name]
        median = float(np.median(samples))
        lo = float(np.percentile(samples, 2.5))
        hi = float(np.percentile(samples, 97.5))
        print(f"{name:>12s}  {nlsq_val:12.4g}  {median:12.4g}  [{lo:.4g}, {hi:.4g}]")


def compute_glass_probability(alpha_samples: np.ndarray) -> float:
    """Compute probability of glass phase from posterior alpha samples.

    The HL model predicts a glass phase (yield stress) for alpha < 0.5.

    Args:
        alpha_samples: Array of posterior alpha samples.

    Returns:
        P(glass) = P(alpha < 0.5)
    """
    return float(np.mean(alpha_samples < 0.5))


def plot_glass_probability_histogram(
    alpha_samples: np.ndarray,
    ax: Any = None,
) -> Any:
    """Plot histogram of alpha samples with glass/fluid classification.

    Args:
        alpha_samples: Array of posterior alpha samples.
        ax: Matplotlib axes. If None, creates new figure.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(alpha_samples, bins=50, density=True, alpha=0.7, color="C0", edgecolor="black")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Glass transition (α=0.5)")

    p_glass = compute_glass_probability(alpha_samples)
    ax.axvspan(0, 0.5, alpha=0.2, color="red", label=f"Glass (P={p_glass:.2%})")
    ax.axvspan(0.5, 1.0, alpha=0.2, color="blue", label=f"Fluid (P={1-p_glass:.2%})")

    ax.set_xlabel("α (coupling parameter)")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Phase Classification: P(glass) = {p_glass:.1%}")
    ax.legend()
    ax.set_xlim(0, 1)

    return ax
