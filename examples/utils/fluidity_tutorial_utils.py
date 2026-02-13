"""Shared utilities for Fluidity tutorial notebooks.

Provides consistent data loading, result saving, and diagnostic printing
across all Fluidity model protocol tutorials (FluidityLocal, FluidityNonlocal,
FluiditySaramitoLocal, FluiditySaramitoNonlocal).
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

# =============================================================================
# Data Loaders
# =============================================================================


def load_emulsion_flow_curve(phi: float = 0.80) -> tuple[np.ndarray, np.ndarray]:
    """Load emulsion flow curve data for a given volume fraction.

    Args:
        phi: Volume fraction (0.69, 0.70, 0.72, 0.74, 0.76, or 0.80).

    Returns:
        Tuple of (shear_rate, stress) arrays in (1/s, Pa).

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


def load_polymer_saos(temp: int = 145) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load polystyrene SAOS data for a given temperature.

    Args:
        temp: Temperature in °C (130, 145, 160, 175, or 190).

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays in (rad/s, Pa, Pa).

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
            f"Polymer SAOS data not found for T={temp}°C. "
            f"Expected at: examples/data/oscillation/polystyrene/oscillation_ps{temp}_data.csv"
        )

    # Tab-separated with header "Angular Frequency\tStorage Modulus\tLoss Modulus"
    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    omega = raw[:, 0]
    G_prime = raw[:, 1]
    G_double_prime = raw[:, 2]

    return omega, G_prime, G_double_prime


def load_mucus_creep() -> tuple[np.ndarray, np.ndarray]:
    """Load mucus creep compliance data.

    Returns:
        Tuple of (time, compliance) arrays in (s, 1/Pa).

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

    # Tab-separated with header "Time\tCreep Compliance"
    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    time = raw[:, 0]
    compliance = raw[:, 1]

    return time, compliance


def load_clay_relaxation(aging_time: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Load laponite clay stress relaxation data for a given aging time.

    Args:
        aging_time: Aging time in minutes (10, 30, 60, 120, or 240).

    Returns:
        Tuple of (time, relaxation_modulus) arrays in (s, Pa).

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir
        / ".."
        / "data"
        / "relaxation"
        / "clays"
        / f"rel_lapo_{aging_time}min_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Clay relaxation data not found for t_wait={aging_time} min. "
            f"Expected at: examples/data/relaxation/clays/rel_lapo_{aging_time}min_data.csv"
        )

    # Tab-separated with header "Time\tRelaxation Modulus"
    raw = np.loadtxt(data_path, delimiter="\t", skiprows=1)
    time = raw[:, 0]
    G_t = raw[:, 1]

    return time, G_t


# =============================================================================
# Synthetic Data Generators
# =============================================================================


def generate_synthetic_startup(
    model: Any,
    gamma_dot: float = 1.0,
    t_end: float = 10.0,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic startup shear data from a fitted Fluidity model.

    Args:
        model: Fitted Fluidity model instance.
        gamma_dot: Applied shear rate in 1/s.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rng = np.random.default_rng(seed)

    # Generate time points
    time = np.linspace(0.01, t_end, n_points)

    # Use model's transient simulation
    model._gamma_dot_applied = gamma_dot
    model._test_mode = "startup"
    stress_clean = model.predict(time)

    # Handle array conversion
    stress_clean = np.asarray(stress_clean).flatten()

    # Add noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
    stress = stress_clean + noise

    return time, stress


def generate_synthetic_creep(
    model: Any,
    sigma_applied: float = 100.0,
    t_end: float = 100.0,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic creep data from a fitted Fluidity model.

    Args:
        model: Fitted Fluidity model instance.
        sigma_applied: Applied stress in Pa.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, strain) arrays in (s, dimensionless).
    """
    rng = np.random.default_rng(seed)

    # Generate time points
    time = np.linspace(0.01, t_end, n_points)

    # Use model's transient simulation
    model._sigma_applied = sigma_applied
    model._test_mode = "creep"
    strain_clean = model.predict(time)

    # Handle array conversion
    strain_clean = np.asarray(strain_clean).flatten()

    # Add noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(strain_clean)), size=strain_clean.shape)
    strain = strain_clean + noise

    return time, strain


def generate_synthetic_relaxation(
    model: Any,
    sigma_0: float = 1000.0,
    t_end: float = 100.0,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stress relaxation data from a fitted Fluidity model.

    Args:
        model: Fitted Fluidity model instance.
        sigma_0: Initial stress in Pa.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rng = np.random.default_rng(seed)

    # Generate time points (logarithmic for relaxation)
    time = np.logspace(-2, np.log10(t_end), n_points)

    # Use model's transient simulation
    model._test_mode = "relaxation"
    stress_clean = model.predict(time)

    # Handle array conversion
    stress_clean = np.asarray(stress_clean).flatten()

    # Scale to sigma_0
    if stress_clean.max() > 0:
        stress_clean = stress_clean * (sigma_0 / stress_clean[0])

    # Add noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
    stress = stress_clean + noise

    return time, stress


def generate_synthetic_laos(
    model: Any,
    gamma_0: float = 0.5,
    omega: float = 1.0,
    n_cycles: int = 10,
    n_points_per_cycle: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic LAOS data from a fitted Fluidity model.

    Args:
        model: Fitted Fluidity model instance.
        gamma_0: Strain amplitude (dimensionless).
        omega: Angular frequency in rad/s.
        n_cycles: Number of oscillation cycles.
        n_points_per_cycle: Points per cycle.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'time', 'strain', 'stress', 'strain_rate'.
    """
    rng = np.random.default_rng(seed)

    # Generate time points
    period = 2.0 * np.pi / omega
    t_end = n_cycles * period
    n_total = n_cycles * n_points_per_cycle
    time = np.linspace(0.01, t_end, n_total)

    # Compute strain and strain rate
    strain = gamma_0 * np.sin(omega * time)
    strain_rate = gamma_0 * omega * np.cos(omega * time)

    # Use model's LAOS simulation
    strain_sim, stress_clean = model.simulate_laos(
        gamma_0=gamma_0,
        omega=omega,
        n_cycles=n_cycles,
        n_points_per_cycle=n_points_per_cycle,
    )

    # Handle array conversion
    stress_clean = np.asarray(stress_clean).flatten()

    # Ensure lengths match
    if len(stress_clean) != len(time):
        stress_clean = np.interp(time, np.linspace(0, t_end, len(stress_clean)), stress_clean)

    # Add noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
    stress = stress_clean + noise

    return {
        "time": time,
        "strain": strain,
        "strain_rate": strain_rate,
        "stress": stress,
    }


# =============================================================================
# Result Persistence
# =============================================================================


def get_output_dir(
    model_variant: Literal["local", "nonlocal", "saramito_local", "saramito_nonlocal"],
    protocol: str,
) -> Path:
    """Get the output directory path for a given model variant and protocol.

    Args:
        model_variant: One of 'local', 'nonlocal', 'saramito_local', 'saramito_nonlocal'.
        protocol: Protocol name (e.g., 'flow_curve', 'startup').

    Returns:
        Path to output directory.
    """
    module_dir = Path(__file__).parent
    return module_dir / ".." / "outputs" / "fluidity" / model_variant / protocol


def save_fluidity_results(
    model: Any,
    result: Any,
    model_variant: Literal["local", "nonlocal", "saramito_local", "saramito_nonlocal"],
    protocol: str,
    param_names: list[str] | None = None,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted Fluidity model.
        result: Bayesian fit result with posterior_samples attribute.
        model_variant: One of 'local', 'nonlocal', 'saramito_local', 'saramito_nonlocal'.
        protocol: Protocol name for labeling files.
        param_names: List of parameter names to save. If None, saves all.
    """
    output_dir = get_output_dir(model_variant, protocol)
    os.makedirs(output_dir, exist_ok=True)

    # Get parameter names from model if not provided
    if param_names is None:
        param_names = list(model.parameters.keys())

    # Save NLSQ point estimates
    nlsq_params = {}
    for name in param_names:
        try:
            nlsq_params[name] = float(model.parameters.get_value(name))
        except (KeyError, AttributeError):
            pass

    with open(output_dir / f"nlsq_params_{protocol}.json", "w") as f:
        json.dump(nlsq_params, f, indent=2)

    # Save posterior samples
    posterior = result.posterior_samples
    posterior_dict = {k: np.array(v).tolist() for k, v in posterior.items()}
    with open(output_dir / f"posterior_{protocol}.json", "w") as f:
        json.dump(posterior_dict, f)

    print(f"Results saved to {output_dir}/")
    print(f"  nlsq_params_{protocol}.json: {len(nlsq_params)} parameters")
    print(f"  posterior_{protocol}.json: {len(list(posterior.values())[0])} draws")


def load_fluidity_parameters(
    model_variant: Literal["local", "nonlocal", "saramito_local", "saramito_nonlocal"],
    protocol: str,
) -> dict[str, float]:
    """Load previously calibrated Fluidity parameters.

    Args:
        model_variant: One of 'local', 'nonlocal', 'saramito_local', 'saramito_nonlocal'.
        protocol: Protocol name (e.g., 'flow_curve').

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    output_dir = get_output_dir(model_variant, protocol)
    param_file = output_dir / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for {model_variant}/{protocol}. "
            "Run the flow_curve notebook first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


def set_model_parameters(model: Any, params: dict[str, float]) -> None:
    """Set model parameters from a dictionary.

    Args:
        model: Fluidity model instance.
        params: Dictionary of parameter name to value.
    """
    for name, value in params.items():
        try:
            model.parameters.set_value(name, value)
        except (KeyError, AttributeError):
            pass


# =============================================================================
# Diagnostic Printing
# =============================================================================


def print_convergence_summary(
    result: Any,
    param_names: list[str] | None = None,
) -> bool:
    """Print formatted convergence diagnostics table.

    Args:
        result: Bayesian fit result with diagnostics attribute.
        param_names: List of parameter names to report. Defaults to fluidity params.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
    if param_names is None:
        param_names = ["G", "tau_y", "K", "n_flow", "f_eq", "f_inf", "theta", "a", "n_rejuv"]

    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 55)
    print(f"{'Parameter':>12s}  {'R-hat':>8s}  {'ESS':>8s}  {'Status':>8s}")
    print("-" * 55)

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
        param_names: List of parameter names to compare.
    """
    if param_names is None:
        param_names = list(model.parameters.keys())

    print("\nParameter Comparison: NLSQ vs Bayesian")
    print("=" * 70)
    print(f"{'Param':>12s}  {'NLSQ':>12s}  {'Median':>12s}  {'95% CI':>28s}")
    print("-" * 70)

    for name in param_names:
        try:
            nlsq_val = model.parameters.get_value(name)
            if name not in posterior:
                continue
            samples = posterior[name]
            median = float(np.median(samples))
            lo = float(np.percentile(samples, 2.5))
            hi = float(np.percentile(samples, 97.5))
            print(f"{name:>12s}  {nlsq_val:12.4g}  {median:12.4g}  [{lo:.4g}, {hi:.4g}]")
        except (KeyError, AttributeError):
            pass


def compute_fit_quality(y_data: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute fit quality metrics (R², RMSE).

    Args:
        y_data: Observed data array.
        y_pred: Predicted data array.

    Returns:
        Dictionary with 'R2' and 'RMSE' keys.
    """
    y_data = np.asarray(y_data).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # R-squared
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))

    return {"R2": r2, "RMSE": rmse}


# =============================================================================
# Fluidity-Specific Physics Functions
# =============================================================================


def get_fluidity_param_names(
    model_variant: Literal["local", "nonlocal", "saramito_local", "saramito_nonlocal"],
) -> list[str]:
    """Get the list of parameter names for a given fluidity model variant.

    Args:
        model_variant: One of 'local', 'nonlocal', 'saramito_local', 'saramito_nonlocal'.

    Returns:
        List of parameter names.
    """
    base_params = ["G", "tau_y", "K", "n_flow", "f_eq", "f_inf", "theta", "a", "n_rejuv"]

    if model_variant == "nonlocal":
        return base_params + ["xi"]
    elif model_variant in ["saramito_local", "saramito_nonlocal"]:
        # Saramito models have additional parameters
        saramito_params = ["G", "tau_y", "eta_s", "f_age", "f_flow", "t_a", "b", "n"]
        if model_variant == "saramito_nonlocal":
            return saramito_params + ["xi"]
        return saramito_params
    else:
        return base_params


def compute_herschel_bulkley(
    gamma_dot: np.ndarray,
    tau_y: float,
    K: float,
    n: float,
) -> np.ndarray:
    """Compute Herschel-Bulkley steady-state stress.

    σ = τ_y + K|γ̇|^n

    Args:
        gamma_dot: Shear rate array (1/s).
        tau_y: Yield stress (Pa).
        K: Flow consistency (Pa·s^n).
        n: Flow exponent (dimensionless).

    Returns:
        Stress array (Pa).
    """
    return tau_y + K * np.abs(gamma_dot) ** n


def compute_steady_fluidity(
    gamma_dot: float | np.ndarray,
    f_eq: float,
    f_inf: float,
    theta: float,
    a: float,
    n_rejuv: float,
) -> float | np.ndarray:
    """Compute steady-state fluidity at a given shear rate.

    f_ss = (f_eq/θ + a|γ̇|^n_rejuv * f_inf) / (1/θ + a|γ̇|^n_rejuv)

    Args:
        gamma_dot: Shear rate (1/s).
        f_eq: Equilibrium fluidity (1/(Pa·s)).
        f_inf: High-shear fluidity (1/(Pa·s)).
        theta: Relaxation time (s).
        a: Rejuvenation amplitude.
        n_rejuv: Rejuvenation exponent.

    Returns:
        Steady-state fluidity (1/(Pa·s)).
    """
    gamma_dot = np.abs(gamma_dot)
    rate_term = a * gamma_dot**n_rejuv
    return (f_eq / theta + rate_term * f_inf) / (1.0 / theta + rate_term)


def detect_shear_banding(
    gamma_dot_field: np.ndarray,
    threshold: float = 0.3,
) -> tuple[bool, dict[str, float]]:
    """Detect shear banding from a fluidity/shear-rate field.

    Banding is indicated when the coefficient of variation (CV) exceeds
    a threshold or the max/min ratio is large.

    Args:
        gamma_dot_field: Shear rate field across the gap.
        threshold: CV threshold for banding detection.

    Returns:
        Tuple of (is_banding, metrics_dict).
    """
    gamma_dot_field = np.asarray(gamma_dot_field)

    # Coefficient of variation
    cv = np.std(gamma_dot_field) / np.mean(gamma_dot_field) if np.mean(gamma_dot_field) > 0 else 0.0

    # Max/min ratio
    min_val = np.min(gamma_dot_field)
    max_val = np.max(gamma_dot_field)
    ratio = max_val / min_val if min_val > 1e-10 else np.inf

    is_banding = cv > threshold or ratio > 10

    return is_banding, {
        "CV": float(cv),
        "max_min_ratio": float(ratio),
        "min": float(min_val),
        "max": float(max_val),
    }
