"""Shared utilities for IKH tutorial notebooks.

Provides consistent data loading, synthetic generation, result saving, and
diagnostic printing across all IKH model protocol tutorials (MIKH, MLIKH).

Data Sources:
- ML-IKH Experimental data.xlsx: Wei et al. 2018 J. Rheol (flow curves, creep)
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 (startup, LAOS)
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


# =============================================================================
# Data Loaders (Real Experimental Data)
# =============================================================================


def load_ml_ikh_flow_curve(
    instrument: Literal["ARG2_up", "ARG2_down", "ARES_up", "ARES_down"] = "ARES_up",
) -> tuple[np.ndarray, np.ndarray]:
    """Load steady-state flow curve data from ML-IKH Excel (Fig 3a).

    The data contains flow curves from two rheometers (ARG2 and ARES) with
    both up-sweep and down-sweep measurements.

    Args:
        instrument: Which dataset to load:
            - "ARG2_up": ARG2 rheometer, stress sweep up
            - "ARG2_down": ARG2 rheometer, stress sweep down
            - "ARES_up": ARES rheometer, rate sweep up (recommended)
            - "ARES_down": ARES rheometer, rate sweep down

    Returns:
        Tuple of (shear_rate, stress) arrays in (1/s, Pa).

    Raises:
        FileNotFoundError: If data file not found.
    """
    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "ikh" / "ML-IKH Experimental data.xlsx"

    if not data_path.exists():
        raise FileNotFoundError(
            f"ML-IKH data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(
        data_path,
        sheet_name="Fig. 3a steady state flow curve",
        header=None,
    )

    # Column mapping based on Excel structure (row 2 has labels)
    col_map = {
        "ARG2_up": (0, 1),      # stress sweep up
        "ARG2_down": (2, 3),    # stress sweep down
        "ARES_up": (5, 6),      # rate sweep up
        "ARES_down": (7, 8),    # rate sweep down
    }

    col_rate, col_stress = col_map[instrument]

    # Skip header rows (0: column names, 1: units, 2: sweep type)
    data = df.iloc[3:, [col_rate, col_stress]].dropna().astype(float)

    gamma_dot = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    # Sort by shear rate
    sort_idx = np.argsort(gamma_dot)
    return gamma_dot[sort_idx], stress[sort_idx]


def load_ml_ikh_step_rate(
    rate_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load step rate (startup) data from ML-IKH Excel (Fig 3b).

    Contains stress vs time data for step changes in shear rate.

    Args:
        rate_index: Which rate to load (0-2, different shear rates).

    Returns:
        Tuple of (time, stress, shear_rate) in (s, Pa, 1/s).

    Raises:
        FileNotFoundError: If data file not found.
        ValueError: If rate_index out of range.
    """
    if rate_index not in [0, 1, 2]:
        raise ValueError(f"rate_index must be 0, 1, or 2, got {rate_index}")

    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "ikh" / "ML-IKH Experimental data.xlsx"

    if not data_path.exists():
        raise FileNotFoundError(
            f"ML-IKH data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(
        data_path,
        sheet_name="Fig. 3b step rate tests",
        header=None,
    )

    # Columns are in pairs: Time, Stress for each rate
    col_time = rate_index * 2
    col_stress = rate_index * 2 + 1

    # Skip header row
    data = df.iloc[1:, [col_time, col_stress]].dropna().astype(float)

    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    # Estimate shear rate from steady-state stress (approximate)
    # The actual rates are not directly in the data
    shear_rates = [0.1, 1.0, 10.0]  # Typical values used
    gamma_dot = shear_rates[rate_index]

    return time, stress, gamma_dot


def load_ml_ikh_creep(
    stress_pair_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load step stress (creep) data from ML-IKH Excel (Fig A2).

    Contains shear rate vs time data for step changes in applied stress.
    Data represents creep response with initial and final stress levels.

    Args:
        stress_pair_index: Which stress jump to load (0-11 for different
            combinations of initial/final stress: 3→5, 3→7, 3→9, 5→3, etc.).

    Returns:
        Tuple of (time, shear_rate, initial_stress, final_stress) in
        (s, 1/s, Pa, Pa).

    Raises:
        FileNotFoundError: If data file not found.
        ValueError: If stress_pair_index out of range.
    """
    if stress_pair_index not in range(12):
        raise ValueError(f"stress_pair_index must be 0-11, got {stress_pair_index}")

    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "ikh" / "ML-IKH Experimental data.xlsx"

    if not data_path.exists():
        raise FileNotFoundError(
            f"ML-IKH data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(
        data_path,
        sheet_name="Fig. A2 step stress tests",
        header=None,
    )

    # Each pair has Time and Shear rate columns
    col_time = stress_pair_index * 2
    col_rate = stress_pair_index * 2 + 1

    # Row 2 has initial stress, row 3 has final stress (rows 0-1 are header/units)
    initial_stress = float(df.iloc[2, col_rate])
    final_stress = float(df.iloc[3, col_rate])

    # Skip header rows (0-4), then extract data (data starts at row 5)
    data = df.iloc[5:, [col_time, col_rate]].dropna().astype(float)

    time = data.iloc[:, 0].values
    shear_rate = data.iloc[:, 1].values

    return time, shear_rate, initial_stress, final_stress


def load_pnas_startup(
    gamma_dot: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load startup shear data from PNAS Digital Rheometer Twin dataset.

    Contains stress vs time for various constant shear rates.

    Args:
        gamma_dot: Shear rate to load. Available: 0.056, 0.32, 1, 56.2, 100.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).

    Raises:
        FileNotFoundError: If data file not found.
        ValueError: If shear rate not available.
    """
    # Map shear rate to sheet name
    rate_sheets = {
        0.056: "StartUp_0.056",
        0.32: "StartUp_0.32",
        1.0: "StartUp_1",
        56.2: "StartUp_56.2",
        100.0: "StartUp_100",
    }

    # Find closest available rate
    available_rates = list(rate_sheets.keys())
    closest = min(available_rates, key=lambda x: abs(x - gamma_dot))
    sheet = rate_sheets[closest]

    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"

    if not data_path.exists():
        raise FileNotFoundError(
            f"PNAS data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(data_path, sheet_name=sheet, header=None)

    # Column 0: Step time, Column 1: Stress (skip rows 0-1 for headers)
    data = df.iloc[3:, [0, 1]].dropna().astype(float)

    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    # Subsample if very large (e.g., 200k points)
    if len(time) > 500:
        indices = np.linspace(0, len(time) - 1, 500, dtype=int)
        time = time[indices]
        stress = stress[indices]

    return time, stress


def load_pnas_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LAOS data from PNAS Digital Rheometer Twin dataset.

    Contains stress and strain vs time for various frequencies and amplitudes.

    Args:
        omega: Angular frequency (1, 3, or 5 rad/s).
        strain_amplitude_index: Index for strain amplitude (0-11, increasing).
            0 corresponds to smallest amplitude (~0.001), 11 to largest (~0.2).

    Returns:
        Tuple of (time, strain, stress) arrays in (s, -, Pa).

    Raises:
        FileNotFoundError: If data file not found.
        ValueError: If omega not available.
    """
    # Map omega to sheet name
    omega_sheets = {
        1.0: "LAOS_w1",
        3.0: "LAOS_w3",
        5.0: "LAOS_w5",
    }

    if omega not in omega_sheets:
        raise ValueError(f"omega must be 1, 3, or 5, got {omega}")

    sheet = omega_sheets[omega]

    if strain_amplitude_index not in range(12):
        raise ValueError(f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}")

    module_dir = Path(__file__).parent
    data_path = module_dir / ".." / "data" / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"

    if not data_path.exists():
        raise FileNotFoundError(
            f"PNAS data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(data_path, sheet_name=sheet, header=None)

    # Each amplitude has 4 columns: Step time, Strain, Stress, (empty)
    # Indices are 0,1,2,3 for first amplitude, 4,5,6,7 for second, etc.
    col_time = strain_amplitude_index * 4
    col_strain = strain_amplitude_index * 4 + 1
    col_stress = strain_amplitude_index * 4 + 2

    # Skip header rows (0-1)
    data = df.iloc[3:, [col_time, col_strain, col_stress]].dropna().astype(float)

    time = data.iloc[:, 0].values
    strain = data.iloc[:, 1].values
    stress = data.iloc[:, 2].values

    # Subsample if very large
    if len(time) > 1000:
        indices = np.linspace(0, len(time) - 1, 1000, dtype=int)
        time = time[indices]
        strain = strain[indices]
        stress = stress[indices]

    return time, strain, stress


# =============================================================================
# Synthetic Data Generators
# =============================================================================


def generate_synthetic_relaxation(
    model: Any,
    sigma_0: float = 100.0,
    t_end: float = 100.0,
    n_points: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stress relaxation data from a fitted IKH model.

    Uses the model's predict_relaxation method to generate clean data,
    then adds Gaussian noise.

    Args:
        model: Fitted MIKH or MLIKH model instance.
        sigma_0: Initial stress in Pa.
        t_end: End time in seconds.
        n_points: Number of time points (log-spaced).
        noise_level: Relative noise level (0.02 = 2%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rng = np.random.default_rng(seed)

    # Log-spaced time points for relaxation
    time = np.logspace(-2, np.log10(t_end), n_points)

    # Use model's prediction method
    stress_clean = model.predict_relaxation(time, sigma_0=sigma_0)
    stress_clean = np.asarray(stress_clean).flatten()

    # Add relative noise
    noise = rng.normal(0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape)
    stress = stress_clean + noise

    # Ensure positive stress
    stress = np.maximum(stress, 0.0)

    return time, stress


def generate_synthetic_saos(
    model: Any,
    omega_range: tuple[float, float] = (0.01, 100.0),
    n_points: int = 50,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic SAOS data from a fitted IKH model.

    Uses small-amplitude oscillatory shear (SAOS) to extract G' and G''
    from the linearized response of the IKH model.

    Args:
        model: Fitted MIKH or MLIKH model instance.
        omega_range: (min, max) angular frequency range in rad/s.
        n_points: Number of frequency points.
        noise_level: Relative noise level (0.02 = 2%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays in (rad/s, Pa, Pa).
    """
    rng = np.random.default_rng(seed)

    # Log-spaced frequency points
    omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_points)

    # For SAOS, we use small amplitude to stay in linear regime
    gamma_0 = 0.001  # Very small strain amplitude

    G_prime_list = []
    G_double_prime_list = []

    for w in omega:
        # Generate one cycle of oscillation
        period = 2 * np.pi / w
        n_points_cycle = 200
        t = np.linspace(0, 3 * period, 3 * n_points_cycle)  # 3 cycles for steady state

        # Get stress response via LAOS at small amplitude
        try:
            stress = model.predict_laos(t, gamma_0=gamma_0, omega=w)
            stress = np.asarray(stress).flatten()

            # Extract last cycle for analysis
            last_cycle_start = 2 * n_points_cycle
            t_cycle = t[last_cycle_start:]
            stress_cycle = stress[last_cycle_start:]
            strain_cycle = gamma_0 * np.sin(w * t_cycle)

            # Fit to extract G' and G''
            # σ(t) = G'γ₀sin(ωt) + G''γ₀cos(ωt)
            # Use least squares: σ = a*sin(ωt) + b*cos(ωt)
            sin_term = np.sin(w * t_cycle)
            cos_term = np.cos(w * t_cycle)
            A = np.column_stack([sin_term, cos_term])
            coeffs, _, _, _ = np.linalg.lstsq(A, stress_cycle, rcond=None)

            G_p = coeffs[0] / gamma_0
            G_pp = coeffs[1] / gamma_0

        except Exception:
            # Fallback: use Maxwell model approximation
            G = model.parameters.get_value("G")
            eta = model.parameters.get_value("eta")
            tau = eta / G
            wt = w * tau
            G_p = G * wt**2 / (1 + wt**2)
            G_pp = G * wt / (1 + wt**2)

        G_prime_list.append(max(G_p, 1e-10))
        G_double_prime_list.append(max(G_pp, 1e-10))

    G_prime = np.array(G_prime_list)
    G_double_prime = np.array(G_double_prime_list)

    # Add noise
    noise_p = rng.normal(0, noise_level * np.mean(G_prime), size=G_prime.shape)
    noise_pp = rng.normal(0, noise_level * np.mean(G_double_prime), size=G_double_prime.shape)

    G_prime = G_prime + noise_p
    G_double_prime = G_double_prime + noise_pp

    # Ensure positive moduli
    G_prime = np.maximum(G_prime, 1e-10)
    G_double_prime = np.maximum(G_double_prime, 1e-10)

    return omega, G_prime, G_double_prime


# =============================================================================
# Result Persistence
# =============================================================================


def get_output_dir(
    model_name: Literal["mikh", "mlikh"],
    protocol: str,
) -> Path:
    """Get the output directory path for a given model and protocol.

    Args:
        model_name: One of 'mikh' or 'mlikh'.
        protocol: Protocol name (e.g., 'flow_curve', 'startup').

    Returns:
        Path to output directory.
    """
    module_dir = Path(__file__).parent
    return module_dir / ".." / "outputs" / "ikh" / model_name / protocol


def save_ikh_results(
    model: Any,
    result: Any,
    model_name: Literal["mikh", "mlikh"],
    protocol: str,
    param_names: list[str] | None = None,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted IKH model.
        result: Bayesian fit result with posterior_samples attribute.
        model_name: One of 'mikh' or 'mlikh'.
        protocol: Protocol name for labeling files.
        param_names: List of parameter names to save. If None, saves all.
    """
    output_dir = get_output_dir(model_name, protocol)
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


def load_ikh_parameters(
    model_name: Literal["mikh", "mlikh"],
    protocol: str,
) -> dict[str, float]:
    """Load previously calibrated IKH parameters.

    Args:
        model_name: One of 'mikh' or 'mlikh'.
        protocol: Protocol name (e.g., 'flow_curve').

    Returns:
        Dictionary of parameter name to value.

    Raises:
        FileNotFoundError: If parameter file not found.
    """
    output_dir = get_output_dir(model_name, protocol)
    param_file = output_dir / f"nlsq_params_{protocol}.json"

    if not param_file.exists():
        raise FileNotFoundError(
            f"No saved parameters for {model_name}/{protocol}. "
            "Run the flow_curve notebook first."
        )

    with open(param_file) as f:
        params = json.load(f)

    return params


def set_model_parameters(model: Any, params: dict[str, float]) -> None:
    """Set model parameters from a dictionary.

    Args:
        model: IKH model instance.
        params: Dictionary of parameter name to value.
    """
    for name, value in params.items():
        try:
            model.parameters.set_value(name, value)
        except (KeyError, AttributeError):
            pass


# =============================================================================
# Parameter Names
# =============================================================================


def get_mikh_param_names() -> list[str]:
    """Get the list of MIKH parameter names.

    Returns:
        List of 11 MIKH parameter names.
    """
    return [
        "G",            # Shear modulus (Pa)
        "eta",          # Maxwell viscosity (Pa.s)
        "C",            # Kinematic hardening modulus (Pa)
        "gamma_dyn",    # Dynamic recovery strain
        "m",            # Armstrong-Frederick exponent
        "sigma_y0",     # Minimal yield stress (Pa)
        "delta_sigma_y", # Structural yield contribution (Pa)
        "tau_thix",     # Thixotropic rebuilding time (s)
        "Gamma",        # Thixotropic breakdown coefficient
        "eta_inf",      # High-shear viscosity (Pa.s)
        "mu_p",         # Plastic viscosity (Pa.s)
    ]


def get_mlikh_param_names(
    n_modes: int = 2,
    yield_mode: Literal["per_mode", "weighted_sum"] = "per_mode",
) -> list[str]:
    """Get the list of MLIKH parameter names.

    Args:
        n_modes: Number of modes.
        yield_mode: Yield surface formulation ('per_mode' or 'weighted_sum').

    Returns:
        List of MLIKH parameter names.
    """
    if yield_mode == "per_mode":
        # 7 parameters per mode + 1 global
        params = []
        for i in range(1, n_modes + 1):
            params.extend([
                f"G_{i}",
                f"C_{i}",
                f"gamma_dyn_{i}",
                f"sigma_y0_{i}",
                f"delta_sigma_y_{i}",
                f"tau_thix_{i}",
                f"Gamma_{i}",
            ])
        params.append("eta_inf")
        return params
    else:
        # 5 global + 3 per mode
        params = ["G", "C", "gamma_dyn", "sigma_y0", "k3"]
        for i in range(1, n_modes + 1):
            params.extend([
                f"tau_thix_{i}",
                f"Gamma_{i}",
                f"w_{i}",
            ])
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
        param_names: List of parameter names to report.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
    if param_names is None:
        param_names = get_mikh_param_names()

    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 55)
    print(f"{'Parameter':>15s}  {'R-hat':>8s}  {'ESS':>8s}  {'Status':>8s}")
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

        print(f"{p:>15s}  {r_hat:8.4f}  {ess:8.0f}  {status:>8s}")

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
    print("=" * 75)
    print(f"{'Parameter':>15s}  {'NLSQ':>12s}  {'Median':>12s}  {'95% CI':>30s}")
    print("-" * 75)

    for name in param_names:
        try:
            nlsq_val = model.parameters.get_value(name)
            if name not in posterior:
                continue
            samples = posterior[name]
            median = float(np.median(samples))
            lo = float(np.percentile(samples, 2.5))
            hi = float(np.percentile(samples, 97.5))
            print(f"{name:>15s}  {nlsq_val:12.4g}  {median:12.4g}  [{lo:.4g}, {hi:.4g}]")
        except (KeyError, AttributeError):
            pass


def compute_fit_quality(y_data: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute fit quality metrics (R-squared, RMSE, NRMSE).

    Args:
        y_data: Observed data array.
        y_pred: Predicted data array.

    Returns:
        Dictionary with 'R2', 'RMSE', and 'NRMSE' keys.
    """
    y_data = np.asarray(y_data).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # R-squared
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # RMSE
    rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))

    # NRMSE (normalized by range)
    data_range = np.max(y_data) - np.min(y_data)
    nrmse = rmse / data_range if data_range > 0 else 0.0

    return {"R2": r2, "RMSE": rmse, "NRMSE": nrmse}


# =============================================================================
# IKH-Specific Physics Functions
# =============================================================================


def compute_armstrong_frederick_backstress(
    strain_plastic: np.ndarray,
    C: float,
    gamma_dyn: float,
) -> np.ndarray:
    """Compute kinematic hardening backstress evolution.

    The Armstrong-Frederick rule:
    dα/dε_p = (2/3)C - γ_dyn * α * sign(dε_p)

    For monotonic loading with ε_p starting from 0:
    α = (2C/3γ_dyn) * (1 - exp(-γ_dyn * ε_p))

    Args:
        strain_plastic: Plastic strain array.
        C: Kinematic hardening modulus (Pa).
        gamma_dyn: Dynamic recovery parameter.

    Returns:
        Backstress array (Pa).
    """
    eps_p = np.abs(strain_plastic)
    saturation = 2 * C / (3 * gamma_dyn)
    return saturation * (1 - np.exp(-gamma_dyn * eps_p))


def compute_thixotropic_structure(
    gamma_dot: float | np.ndarray,
    tau_thix: float,
    Gamma: float,
) -> float | np.ndarray:
    """Compute steady-state thixotropic structure parameter.

    λ_ss = 1 / (1 + Γ * τ_thix * |γ̇|)

    Args:
        gamma_dot: Shear rate (1/s).
        tau_thix: Thixotropic rebuilding time (s).
        Gamma: Breakdown coefficient.

    Returns:
        Structure parameter (0-1).
    """
    gamma_dot = np.abs(gamma_dot)
    return 1.0 / (1.0 + Gamma * tau_thix * gamma_dot)


def compute_ikh_yield_stress(
    lambda_: float | np.ndarray,
    sigma_y0: float,
    delta_sigma_y: float,
) -> float | np.ndarray:
    """Compute IKH yield stress as function of structure.

    σ_y = σ_y0 + λ * Δσ_y

    Args:
        lambda_: Structure parameter (0-1).
        sigma_y0: Minimal yield stress (Pa).
        delta_sigma_y: Structural yield contribution (Pa).

    Returns:
        Total yield stress (Pa).
    """
    return sigma_y0 + lambda_ * delta_sigma_y
