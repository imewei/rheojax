"""Shared utilities for EPM tutorial notebooks.

Provides consistent data loading, result saving, and diagnostic printing
across all EPM protocol tutorials.
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
    # Determine data path relative to this file
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


def load_polystyrene_oscillation(temp: int = 145) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load polystyrene oscillation data for a given temperature.

    Args:
        temp: Temperature in °C (130, 145, 160, 175, or 190).

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "oscillation" / "polystyrene" /
        f"oscillation_ps{temp}_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Polystyrene oscillation data not found for T={temp}°C. "
            f"Expected at: examples/data/oscillation/polystyrene/oscillation_ps{temp}_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    omega = raw[:, 0]
    G_prime = raw[:, 1]
    G_double_prime = raw[:, 2]

    return omega, G_prime, G_double_prime


def load_mucus_creep() -> tuple[np.ndarray, np.ndarray]:
    """Load mucus creep data.

    Returns:
        Tuple of (time, strain) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "creep" / "biological" / "creep_mucus_data.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            "Mucus creep data not found. "
            "Expected at: examples/data/creep/biological/creep_mucus_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    strain = raw[:, 1]

    return time, strain


def load_polymer_relaxation(name: str = "ps145") -> tuple[np.ndarray, np.ndarray]:
    """Load polymer stress relaxation data.

    Args:
        name: Dataset name (ps130, ps145, ps160, ps175, ps190, pp, hdpe).

    Returns:
        Tuple of (time, G_t) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "relaxation" / "polymers" /
        f"stressrelaxation_{name}_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Relaxation data not found for {name}. "
            f"Expected at: examples/data/relaxation/polymers/stressrelaxation_{name}_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter=",", skiprows=1)
    time = raw[:, 0]
    G_t = raw[:, 1]

    return time, G_t


def save_epm_results(
    model: Any,
    result: Any,
    output_dir: str,
    protocol: str,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted EPM model (LatticeEPM or TensorialEPM).
        result: Bayesian fit result with posterior_samples attribute.
        output_dir: Directory to save results.
        protocol: Protocol name for labeling files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save NLSQ point estimates
    nlsq_params = {}
    for name in model.parameters.keys():
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


def load_epm_parameters(
    protocol: str,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Load previously calibrated EPM parameters.

    Args:
        protocol: Protocol name (e.g., 'flow_curve').
        output_dir: Directory with saved results. Defaults to ../outputs/epm/.

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    if output_dir is None:
        module_dir = Path(__file__).parent
        output_dir = module_dir / ".." / "outputs" / "epm" / protocol

    param_file = Path(output_dir) / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for protocol '{protocol}'. "
            f"Run notebook 01_epm_flow_curve.ipynb first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


def generate_synthetic_startup(
    params: dict[str, float],
    gamma_dot: float = 1.0,
    t_end: float = 10.0,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic startup shear data from calibrated EPM parameters.

    Uses the LatticeEPM model to generate stress(t) data for startup flow,
    then adds Gaussian noise.

    Args:
        params: Dictionary of EPM parameters (mu, tau_pl, sigma_c_mean, sigma_c_std).
        gamma_dot: Applied shear rate [1/s].
        t_end: End time [s].
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays.
    """
    # Import here to avoid circular imports and allow use in Colab
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.epm.lattice import LatticeEPM
    from rheojax.core.data import RheoData

    jax, jnp = safe_import_jax()

    # Create model with calibrated parameters
    model = LatticeEPM(
        L=32,
        dt=0.01,
        mu=params.get("mu", 1.0),
        tau_pl=params.get("tau_pl", 1.0),
        sigma_c_mean=params.get("sigma_c_mean", 1.0),
        sigma_c_std=params.get("sigma_c_std", 0.1),
    )

    # Generate time array
    time = np.linspace(0, t_end, n_points)

    # Create RheoData for startup
    rheo_data = RheoData(
        x=time,
        y=np.zeros_like(time),  # Dummy y
        initial_test_mode="startup",
        metadata={"gamma_dot": gamma_dot},
    )

    # Simulate (smooth=True for deterministic output)
    result = model.predict(rheo_data, smooth=True, seed=seed)
    stress_clean = np.array(result.y)

    # Add noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_level * np.mean(stress_clean), size=stress_clean.shape)
    stress = stress_clean + noise

    return time, stress


def print_convergence_summary(
    result: Any,
    param_names: list[str],
) -> bool:
    """Print formatted convergence diagnostics table.

    Args:
        result: Bayesian fit result with diagnostics attribute.
        param_names: List of parameter names to report.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
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
            status = "✓"
        else:
            status = "✗"
            all_pass = False

        print(f"{p:>12s}  {r_hat:8.4f}  {ess:8.0f}  {status:>8s}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")

    if n_div > 0:
        all_pass = False

    if all_pass:
        print("\n✓ All convergence criteria PASSED")
    else:
        print("\n✗ CHECK REQUIRED: Increase num_warmup/num_samples or check warm-start")

    return all_pass


def print_parameter_comparison(
    model: Any,
    posterior: dict[str, np.ndarray],
    param_names: list[str],
) -> None:
    """Print NLSQ vs Bayesian parameter comparison table.

    Args:
        model: Fitted model with parameters attribute.
        posterior: Dictionary of posterior samples.
        param_names: List of parameter names to compare.
    """
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
