"""Shared utilities for Giesekus tutorial notebooks.

Provides consistent data loading, result saving, and diagnostic printing
across all Giesekus viscoelastic model protocol tutorials.
"""

import json
import os
from pathlib import Path
from typing import Any

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


def load_polymer_relaxation(temp: int = 145) -> tuple[np.ndarray, np.ndarray]:
    """Load polystyrene stress relaxation data for a given temperature.

    Args:
        temp: Temperature in °C (130, 145, 160, 175, or 190).

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
        / "polymers"
        / f"stressrelaxation_ps{temp}_data.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Polymer relaxation data not found for T={temp}°C. "
            f"Expected at: examples/data/relaxation/polymers/stressrelaxation_ps{temp}_data.csv"
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
    params: dict[str, float],
    gamma_dot: float = 1.0,
    t_end: float = 10.0,
    n_points: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Giesekus startup flow data from calibrated parameters.

    Args:
        params: Dictionary of Giesekus parameters (eta_p, lambda_1, alpha, eta_s).
        gamma_dot: Applied shear rate in 1/s.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.giesekus import GiesekusSingleMode

    jax, jnp = safe_import_jax()
    rng = np.random.default_rng(seed)

    # Create model with calibrated parameters
    model = GiesekusSingleMode()
    model.parameters.set_value("eta_p", params.get("eta_p", 100.0))
    model.parameters.set_value("lambda_1", params.get("lambda_1", 1.0))
    model.parameters.set_value("alpha", params.get("alpha", 0.3))
    model.parameters.set_value("eta_s", params.get("eta_s", 0.0))

    # Generate time points
    time = np.linspace(0.01, t_end, n_points)

    # Simulate startup stress
    stress_clean = np.array(model.simulate_startup(time, gamma_dot=gamma_dot))

    # Add noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
    stress = stress_clean + noise

    return time, stress


def generate_synthetic_normal_stresses(
    params: dict[str, float],
    gamma_dot_range: tuple[float, float] = (0.01, 100.0),
    n_points: int = 50,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic normal stress difference data from calibrated parameters.

    Uses the Giesekus analytical predictions for N₁ and N₂.
    Key relation: N₂/N₁ = -α/2 (exact theoretical prediction).

    Args:
        params: Dictionary of Giesekus parameters (eta_p, lambda_1, alpha, eta_s).
        gamma_dot_range: (min, max) shear rate range in 1/s.
        n_points: Number of shear rate points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (gamma_dot, N1, N2) arrays in (1/s, Pa, Pa).
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.giesekus import GiesekusSingleMode

    jax, jnp = safe_import_jax()
    rng = np.random.default_rng(seed)

    # Create model with calibrated parameters
    model = GiesekusSingleMode()
    model.parameters.set_value("eta_p", params.get("eta_p", 100.0))
    model.parameters.set_value("lambda_1", params.get("lambda_1", 1.0))
    model.parameters.set_value("alpha", params.get("alpha", 0.3))
    model.parameters.set_value("eta_s", params.get("eta_s", 0.0))

    # Generate shear rate points
    gamma_dot = np.logspace(
        np.log10(gamma_dot_range[0]), np.log10(gamma_dot_range[1]), n_points
    )

    # Predict normal stresses
    N1_clean, N2_clean = model.predict_normal_stresses(gamma_dot)
    N1_clean = np.array(N1_clean)
    N2_clean = np.array(N2_clean)

    # Add noise
    noise_N1 = rng.normal(0, noise_level * np.mean(np.abs(N1_clean)), size=N1_clean.shape)
    noise_N2 = rng.normal(0, noise_level * np.mean(np.abs(N2_clean)), size=N2_clean.shape)
    N1 = N1_clean + noise_N1
    N2 = N2_clean + noise_N2

    return gamma_dot, N1, N2


def generate_synthetic_laos(
    params: dict[str, float],
    gamma_0: float = 0.5,
    omega: float = 1.0,
    n_cycles: int = 10,
    n_points_per_cycle: int = 100,
    noise_level: float = 0.03,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic Giesekus LAOS data from calibrated parameters.

    Args:
        params: Dictionary of Giesekus parameters (eta_p, lambda_1, alpha, eta_s).
        gamma_0: Strain amplitude (dimensionless).
        omega: Angular frequency in rad/s.
        n_cycles: Number of oscillation cycles.
        n_points_per_cycle: Points per cycle.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'time', 'strain', 'stress', 'strain_rate'.
    """
    from rheojax.core.jax_config import safe_import_jax
    from rheojax.models.giesekus import GiesekusSingleMode

    jax, jnp = safe_import_jax()
    rng = np.random.default_rng(seed)

    # Create model with calibrated parameters
    model = GiesekusSingleMode()
    model.parameters.set_value("eta_p", params.get("eta_p", 100.0))
    model.parameters.set_value("lambda_1", params.get("lambda_1", 1.0))
    model.parameters.set_value("alpha", params.get("alpha", 0.3))
    model.parameters.set_value("eta_s", params.get("eta_s", 0.0))

    # Generate time points
    period = 2 * np.pi / omega
    t_end = n_cycles * period
    n_total = n_cycles * n_points_per_cycle
    time = np.linspace(0.01, t_end, n_total)

    # Compute strain and strain rate
    strain = gamma_0 * np.sin(omega * time)
    strain_rate = gamma_0 * omega * np.cos(omega * time)

    # Simulate LAOS stress
    result = model.simulate_laos(time, gamma_0=gamma_0, omega=omega, n_cycles=n_cycles)
    stress_clean = np.array(result["stress"] if isinstance(result, dict) else result)

    # Handle case where simulate_laos returns just stress array
    if len(stress_clean) != len(time):
        # Fallback: interpolate or truncate
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


def save_giesekus_results(
    model: Any,
    result: Any,
    output_dir: str,
    protocol: str,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted Giesekus model.
        result: Bayesian fit result with posterior_samples attribute.
        output_dir: Directory to save results.
        protocol: Protocol name for labeling files.
    """
    os.makedirs(output_dir, exist_ok=True)

    param_names = ["eta_p", "lambda_1", "alpha", "eta_s"]

    # Save NLSQ point estimates
    nlsq_params = {}
    for name in param_names:
        try:
            nlsq_params[name] = float(model.parameters.get_value(name))
        except (KeyError, AttributeError):
            pass

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


def load_giesekus_parameters(
    protocol: str,
    output_dir: str | None = None,
) -> dict[str, float]:
    """Load previously calibrated Giesekus parameters.

    Args:
        protocol: Protocol name (e.g., 'flow_curve').
        output_dir: Directory with saved results. Defaults to ../outputs/giesekus/.

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    if output_dir is None:
        module_dir = Path(__file__).parent
        output_dir = module_dir / ".." / "outputs" / "giesekus" / protocol

    param_file = Path(output_dir) / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for protocol '{protocol}'. "
            "Run notebook 01_giesekus_flow_curve.ipynb first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


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
        param_names: List of parameter names to report. Defaults to Giesekus params.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
    if param_names is None:
        param_names = ["eta_p", "lambda_1", "alpha", "eta_s"]

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
        param_names: List of parameter names to compare. Defaults to Giesekus params.
    """
    if param_names is None:
        param_names = ["eta_p", "lambda_1", "alpha", "eta_s"]

    print("\nParameter Comparison: NLSQ vs Bayesian")
    print("=" * 70)
    print(f"{'Param':>12s}  {'NLSQ':>12s}  {'Median':>12s}  {'95% CI':>28s}")
    print("-" * 70)

    for name in param_names:
        try:
            nlsq_val = model.parameters.get_value(name)
            samples = posterior[name]
            median = float(np.median(samples))
            lo = float(np.percentile(samples, 2.5))
            hi = float(np.percentile(samples, 97.5))
            print(f"{name:>12s}  {nlsq_val:12.4g}  {median:12.4g}  [{lo:.4g}, {hi:.4g}]")
        except (KeyError, AttributeError):
            pass


# =============================================================================
# Giesekus-Specific Physics Functions
# =============================================================================


def compute_normal_stress_ratio(alpha: float | np.ndarray) -> float | np.ndarray:
    """Compute the theoretical N₂/N₁ ratio for Giesekus model.

    The Giesekus model predicts: N₂/N₁ = -α/2 (exact theoretical result).

    This ratio is independent of shear rate and provides a direct
    experimental route to determine the mobility parameter α.

    Args:
        alpha: Giesekus mobility parameter(s) in [0, 0.5].

    Returns:
        N₂/N₁ ratio (always negative since N₂ < 0 and N₁ > 0).
    """
    return -alpha / 2


def validate_ucm_limit(model: Any, tolerance: float = 0.01) -> bool:
    """Check if model is in the UCM (Upper-Convected Maxwell) limit.

    The Giesekus model reduces to UCM when α → 0:
    - N₂ → 0 (no second normal stress difference)
    - Shear viscosity becomes constant (no shear-thinning)

    Args:
        model: Giesekus model instance.
        tolerance: Maximum α value to consider as UCM limit.

    Returns:
        True if α < tolerance (UCM-like behavior).
    """
    alpha = model.parameters.get_value("alpha")
    return abs(alpha) < tolerance


def get_critical_weissenberg(alpha: float) -> float:
    """Compute the critical Weissenberg number for shear-thinning onset.

    For Giesekus, significant shear-thinning begins around Wi ≈ 1/√α.
    At this Wi, the viscosity has dropped by ~50% from its zero-shear value.

    Args:
        alpha: Giesekus mobility parameter in (0, 0.5].

    Returns:
        Critical Weissenberg number Wi_c = 1/√α.

    Raises:
        ValueError: If α ≤ 0 (no finite critical Wi for UCM).
    """
    if alpha <= 0:
        raise ValueError(
            "Critical Weissenberg undefined for α=0 (UCM limit). "
            "UCM shows no shear-thinning at any shear rate."
        )
    return 1.0 / np.sqrt(alpha)


def compute_weissenberg_number(
    gamma_dot: float | np.ndarray, lambda_1: float
) -> float | np.ndarray:
    """Compute Weissenberg number Wi = λ·γ̇.

    The Weissenberg number characterizes the relative importance of
    elastic to viscous forces. For Wi >> 1, elastic effects dominate
    and nonlinear behavior (shear-thinning, normal stresses) is pronounced.

    Args:
        gamma_dot: Shear rate(s) in 1/s.
        lambda_1: Relaxation time in s.

    Returns:
        Weissenberg number (dimensionless).
    """
    return lambda_1 * gamma_dot


def compute_deborah_number(omega: float | np.ndarray, lambda_1: float) -> float | np.ndarray:
    """Compute Deborah number De = λ·ω.

    The Deborah number characterizes the relative importance of
    relaxation time to observation time (1/ω for oscillatory flow).

    Args:
        omega: Angular frequency in rad/s.
        lambda_1: Relaxation time in s.

    Returns:
        Deborah number (dimensionless).
    """
    return lambda_1 * omega


def estimate_alpha_from_normal_stresses(
    N1: np.ndarray, N2: np.ndarray
) -> tuple[float, float]:
    """Estimate α from measured normal stress differences.

    Uses the theoretical relation N₂/N₁ = -α/2 to extract α.

    Args:
        N1: First normal stress difference array (Pa).
        N2: Second normal stress difference array (Pa).

    Returns:
        Tuple of (alpha_mean, alpha_std) estimated from data.
    """
    # Avoid division by zero
    valid = np.abs(N1) > 1e-10
    if not np.any(valid):
        raise ValueError("All N1 values are near zero, cannot estimate alpha.")

    ratio = N2[valid] / N1[valid]
    alpha_samples = -2 * ratio

    return float(np.mean(alpha_samples)), float(np.std(alpha_samples))


def plot_shear_thinning_regime(
    model: Any,
    ax: Any = None,
    gamma_dot_range: tuple[float, float] = (1e-3, 1e3),
    n_points: int = 100,
) -> Any:
    """Plot viscosity vs shear rate with shear-thinning regime markers.

    Args:
        model: Fitted Giesekus model.
        ax: Matplotlib axes. If None, creates new figure.
        gamma_dot_range: (min, max) shear rate range in 1/s.
        n_points: Number of points for smooth curve.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    gamma_dot = np.logspace(
        np.log10(gamma_dot_range[0]), np.log10(gamma_dot_range[1]), n_points
    )

    # Get parameters
    eta_p = model.parameters.get_value("eta_p")
    eta_s = model.parameters.get_value("eta_s")
    lambda_1 = model.parameters.get_value("lambda_1")
    alpha = model.parameters.get_value("alpha")

    # Predict flow curve and compute viscosity
    stress = np.array(model.predict(gamma_dot, test_mode="flow_curve"))
    eta = stress / gamma_dot

    # Zero-shear viscosity
    eta_0 = eta_p + eta_s

    # Critical shear rate
    if alpha > 0:
        gamma_dot_c = 1.0 / (lambda_1 * np.sqrt(alpha))
    else:
        gamma_dot_c = None

    # Plot
    ax.loglog(gamma_dot, eta, "-", lw=2, color="C0", label="η(γ̇)")
    ax.axhline(eta_0, color="gray", linestyle="--", alpha=0.7, label=f"η₀ = {eta_0:.1f} Pa·s")

    if gamma_dot_c is not None and gamma_dot_range[0] < gamma_dot_c < gamma_dot_range[1]:
        ax.axvline(
            gamma_dot_c,
            color="C3",
            linestyle="--",
            alpha=0.7,
            label=f"γ̇_c = {gamma_dot_c:.2g} 1/s (Wi=1/√α)",
        )

    ax.set_xlabel("Shear rate γ̇ [1/s]")
    ax.set_ylabel("Viscosity η [Pa·s]")
    ax.set_title(f"Giesekus Shear-Thinning (α={alpha:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    return ax
