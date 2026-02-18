"""Shared utilities for STZ tutorial notebooks.

Provides consistent data loading, synthetic data generation, result saving,
and diagnostic printing across all STZ protocol tutorials.
"""

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def compute_fit_quality(
    y_data: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute R-squared and RMSE fit quality metrics.

    Args:
        y_data: Observed data array.
        y_pred: Model prediction array.

    Returns:
        Dictionary with 'r_squared' and 'rmse' keys.
    """
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y_data - y_pred) ** 2)))
    return {"r_squared": r_squared, "rmse": rmse}


def generate_synthetic_flow_curve(
    params: dict[str, float] | None = None,
    gamma_dot_range: tuple[float, float] = (1e-3, 1e3),
    n_points: int = 30,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Generate synthetic steady-state flow curve from the STZ arctanh formula.

    Uses the analytical formula sigma = sigma_y * arctanh(gamma_dot * tau0 / term)
    and adds multiplicative Gaussian noise in log-space.

    Args:
        params: Dictionary of STZ parameters. If None, uses physically motivated
            defaults suitable for a soft-matter demonstration.
        gamma_dot_range: Shear rate range (min, max) in 1/s.
        n_points: Number of logarithmically-spaced data points.
        noise_level: Relative noise level in log-space (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (gamma_dot, stress, true_params) where true_params is the
        dictionary of parameters used to generate the data.
    """
    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    if params is None:
        params = {
            "sigma_y": 50.0,
            "chi_inf": 0.26,
            "tau0": 1e-4,
            "ez": 0.8,
        }

    gamma_dot = np.logspace(
        np.log10(gamma_dot_range[0]),
        np.log10(gamma_dot_range[1]),
        n_points,
    )

    # Compute clean stress via the arctanh formula
    sigma_y = params["sigma_y"]
    chi_inf = params["chi_inf"]
    tau0 = params["tau0"]
    ez = params["ez"]

    term = np.exp(-(1.0 + ez) / chi_inf)
    arg = gamma_dot * tau0 / (term + 1e-30)
    arg_clamped = np.clip(arg, -0.999999, 0.999999)
    stress_clean = sigma_y * np.arctanh(arg_clamped)

    # Add multiplicative noise in log-space
    rng = np.random.default_rng(seed)
    log_noise = rng.normal(0, noise_level, size=stress_clean.shape)
    stress = stress_clean * np.exp(log_noise)

    return gamma_dot, stress, dict(params)


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


def load_laponite_relaxation(
    t_age: int = 3600,
) -> tuple[np.ndarray, np.ndarray]:
    """Load laponite clay relaxation data for a given aging time.

    Args:
        t_age: Aging time in seconds (600, 1200, 1800, 2400, or 3600).

    Returns:
        Tuple of (time, G_t) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "relaxation" / "clays" / f"rel_lapo_{t_age}.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Laponite relaxation data not found for t_age={t_age}s. "
            f"Expected at: examples/data/relaxation/clays/rel_lapo_{t_age}.csv"
        )

    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    time = raw[:, 0]
    G_t = raw[:, 1]

    return time, G_t


def load_mucus_creep() -> tuple[np.ndarray, np.ndarray]:
    """Load mucus creep compliance data.

    Returns:
        Tuple of (time, J_t) arrays where J_t is creep compliance.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "creep" / "biological" / "creep_mucus_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            "Mucus creep data not found. "
            "Expected at: examples/data/creep/biological/creep_mucus_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    time = raw[:, 0]
    J_t = raw[:, 1]

    return time, J_t


def load_polystyrene_oscillation(
    temp: int = 145,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load polystyrene oscillation data for a given temperature.

    Args:
        temp: Temperature in C (130, 145, 160, 175, or 190).

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays.

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir
        / ".."
        / "data"
        / "oscillation"
        / "polystyrene"
        / f"oscillation_ps{temp}_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Polystyrene oscillation data not found for T={temp}C. "
            f"Expected at: examples/data/oscillation/polystyrene/oscillation_ps{temp}_data.csv"
        )

    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    omega = raw[:, 0]
    G_prime = raw[:, 1]
    G_double_prime = raw[:, 2]

    return omega, G_prime, G_double_prime


def load_stz_calibrated_params(
    protocol: str,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Load previously calibrated STZ parameters.

    Args:
        protocol: Protocol name (e.g., 'flow_curve').
        output_dir: Directory with saved results. Defaults to ../outputs/stz/<protocol>.

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    if output_dir is None:
        module_dir = Path(__file__).parent
        output_dir_path = module_dir / ".." / "outputs" / "stz" / protocol
    else:
        output_dir_path = Path(output_dir)

    param_file = output_dir_path / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for protocol '{protocol}'. "
            f"Run notebook 01_stz_flow_curve.ipynb first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


def save_stz_results(
    model: Any,
    result: Any,
    output_dir: str,
    protocol: str,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted STZ model (STZConventional).
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


def generate_synthetic_startup(
    params: dict[str, float],
    gamma_dot: float = 10.0,
    t_end: float = 10.0,
    n_points: int = 200,
    noise_level: float = 0.03,
    seed: int = 42,
    variant: str = "standard",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic startup shear data from calibrated STZ parameters.

    Uses the STZConventional model to generate stress(t) data for startup flow,
    then adds Gaussian noise.

    Args:
        params: Dictionary of STZ parameters.
        gamma_dot: Applied shear rate [1/s].
        t_end: End time [s].
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.
        variant: STZ variant ('minimal', 'standard', 'full').

    Returns:
        Tuple of (time, stress) arrays.
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.stz import STZConventional

    jax, jnp = safe_import_jax()

    model = STZConventional(variant=variant)

    # Widen bounds for soft-matter scales (defaults target metallic glasses)
    for p_name in ["G0", "sigma_y", "tau0"]:
        if p_name in model.parameters.keys():
            model.parameters.set_bounds(p_name, (1e-20, 1e20))

    # Set parameters from calibrated values
    for name, value in params.items():
        if name in model.parameters.keys():
            model.parameters[name].value = value

    # Generate time array (avoid t=0 for ODE solver)
    time = np.linspace(1e-4, t_end, n_points)

    # Store required attributes for prediction
    model._gamma_dot_applied = gamma_dot
    model._sigma_applied = None
    model._test_mode = "startup"
    model.fitted_ = True

    # Predict clean stress
    stress_clean = model.predict(time)

    # Add noise
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape
    )
    stress = stress_clean + noise

    return time, stress


def generate_synthetic_laos(
    params: dict[str, float],
    gamma_0: float = 0.1,
    omega: float = 1.0,
    n_cycles: int = 2,
    n_points_per_cycle: int = 256,
    noise_level: float = 0.03,
    seed: int = 42,
    variant: str = "standard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic LAOS data from calibrated STZ parameters.

    Args:
        params: Dictionary of STZ parameters.
        gamma_0: Strain amplitude.
        omega: Angular frequency (rad/s).
        n_cycles: Number of oscillation cycles.
        n_points_per_cycle: Points per cycle.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.
        variant: STZ variant ('minimal', 'standard', 'full').

    Returns:
        Tuple of (time, strain, stress) arrays.
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.stz import STZConventional

    jax, jnp = safe_import_jax()

    model = STZConventional(variant=variant)

    # Widen bounds for soft-matter scales (defaults target metallic glasses)
    for p_name in ["G0", "sigma_y", "tau0"]:
        if p_name in model.parameters.keys():
            model.parameters.set_bounds(p_name, (1e-20, 1e20))

    # Set parameters from calibrated values
    for name, value in params.items():
        if name in model.parameters.keys():
            model.parameters[name].value = value

    # Simulate LAOS
    strain_clean, stress_clean = model.simulate_laos(
        gamma_0=gamma_0,
        omega=omega,
        n_cycles=n_cycles,
        n_points_per_cycle=n_points_per_cycle,
    )

    # Build time array
    period = 2.0 * np.pi / omega
    t_max = n_cycles * period
    n_points = n_cycles * n_points_per_cycle
    time = np.linspace(0, t_max, n_points, endpoint=False)

    # Add noise to stress only
    rng = np.random.default_rng(seed)
    noise = rng.normal(
        0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape
    )
    stress = stress_clean + noise

    return time, strain_clean, stress


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
