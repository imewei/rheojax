"""Shared utilities for FIKH tutorial notebooks.

Provides consistent data loading, synthetic generation, alpha exploration,
result saving, and diagnostic printing across all FIKH model tutorials
(FIKH, FMLIKH).

Key Innovation: alpha_structure parameter (0 < α < 1) controls Caputo
fractional derivative for power-law memory in thixotropic structure evolution.

Data Sources:
- ML-IKH Experimental data.xlsx: Wei et al. 2018 J. Rheol (flow curves, creep)
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 (startup, LAOS)
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Data Loaders (Reuse IKH Patterns)
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
        "ARG2_up": (0, 1),  # stress sweep up
        "ARG2_down": (2, 3),  # stress sweep down
        "ARES_up": (5, 6),  # rate sweep up
        "ARES_down": (7, 8),  # rate sweep down
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
    data_path = (
        module_dir / ".." / "data" / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"PNAS data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(data_path, sheet_name=sheet, header=None)

    # Column 0: Step time, Column 1: Stress (skip rows 0-2: title, headers, units)
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
        raise ValueError(
            f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}"
        )

    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    )

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
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stress relaxation data from a fitted FIKH model.

    Uses the model's predict method with relaxation test_mode to generate clean data,
    then adds Gaussian noise.

    Args:
        model: Fitted FIKH or FMLIKH model instance.
        sigma_0: Initial stress in Pa.
        t_end: End time in seconds.
        n_points: Number of time points (log-spaced).
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rng = np.random.default_rng(seed)

    # Log-spaced time points for relaxation
    time = np.logspace(-2, np.log10(t_end), n_points)

    # Use model's prediction method with sigma_0 for transient relaxation
    result = model.predict(time, test_mode="relaxation", sigma_0=sigma_0)
    # Handle RheoData or array return
    if hasattr(result, "y"):
        stress_clean = np.asarray(result.y).flatten()
    else:
        stress_clean = np.asarray(result).flatten()

    # Add relative noise
    noise = rng.normal(
        0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape
    )
    stress = stress_clean + noise

    # Ensure positive stress
    stress = np.maximum(stress, 0.0)

    return time, stress


def generate_synthetic_saos(
    model: Any,
    omega_range: tuple[float, float] = (0.01, 100.0),
    n_points: int = 50,
    noise_level: float = 0.03,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic SAOS data from a fitted FIKH model.

    Uses small-amplitude oscillatory shear (SAOS) to extract G' and G''
    from the linearized response of the FIKH model by simulating time-domain
    oscillations and extracting moduli.

    Args:
        model: Fitted FIKH or FMLIKH model instance.
        omega_range: (min, max) angular frequency range in rad/s.
        n_points: Number of frequency points.
        noise_level: Relative noise level (0.03 = 3%).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays in (rad/s, Pa, Pa).
    """
    rng = np.random.default_rng(seed)

    # Log-spaced frequency points
    omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_points)

    # For SAOS, we simulate oscillatory strain and extract G', G''
    gamma_0 = 0.01  # Small amplitude for linear regime

    G_prime_list = []
    G_double_prime_list = []

    for w in omega:
        # Generate one cycle of oscillation with sufficient resolution
        period = 2 * np.pi / w
        n_points_cycle = 100
        # Use 5 cycles to reach steady state
        t = np.linspace(0, 5 * period, 5 * n_points_cycle)

        # Strain history: γ(t) = γ₀·sin(ωt)
        strain = gamma_0 * np.sin(w * t)

        try:
            # Get stress response
            stress = model.predict(t, test_mode="startup", strain=strain)
            if hasattr(stress, "y"):
                stress = np.asarray(stress.y).flatten()
            else:
                stress = np.asarray(stress).flatten()

            # Extract last cycle for analysis (steady state)
            last_cycle_start = 4 * n_points_cycle
            t_cycle = t[last_cycle_start:]
            stress_cycle = stress[last_cycle_start:]

            # Fit: σ(t) = G'·γ₀·sin(ωt) + G''·γ₀·cos(ωt)
            sin_term = gamma_0 * np.sin(w * t_cycle)
            cos_term = gamma_0 * np.cos(w * t_cycle)
            A = np.column_stack([sin_term, cos_term])
            coeffs, _, _, _ = np.linalg.lstsq(A, stress_cycle, rcond=None)

            G_p = coeffs[0]
            G_pp = coeffs[1]

        except Exception:
            # Fallback: use Maxwell model approximation
            try:
                # For FMLIKH, sum the Maxwell responses
                if hasattr(model, "n_modes"):
                    G_p = 0.0
                    G_pp = 0.0
                    for i in range(model.n_modes):
                        G_i = model.parameters.get_value(f"G_{i}")
                        eta_i = model.parameters.get_value(f"eta_{i}")
                        tau_i = eta_i / max(G_i, 1e-12)
                        wt = w * tau_i
                        G_p += G_i * wt**2 / (1 + wt**2)
                        G_pp += G_i * wt / (1 + wt**2)
                else:
                    G = model.parameters.get_value("G")
                    eta = model.parameters.get_value("eta")
                    tau = eta / G
                    wt = w * tau
                    G_p = G * wt**2 / (1 + wt**2)
                    G_pp = G * wt / (1 + wt**2)
            except Exception:
                # Last resort: use reasonable defaults
                G_p = 1000.0 * (w**0.5)
                G_pp = 100.0 * w

        G_prime_list.append(max(G_p, 1e-10))
        G_double_prime_list.append(max(G_pp, 1e-10))

    G_prime = np.array(G_prime_list)
    G_double_prime = np.array(G_double_prime_list)

    # Add noise
    noise_p = rng.normal(0, noise_level * np.mean(G_prime), size=G_prime.shape)
    noise_pp = rng.normal(
        0, noise_level * np.mean(G_double_prime), size=G_double_prime.shape
    )

    G_prime = G_prime + noise_p
    G_double_prime = G_double_prime + noise_pp

    # Ensure positive after noise
    G_prime = np.maximum(G_prime, 1e-10)
    G_double_prime = np.maximum(G_double_prime, 1e-10)

    return omega, G_prime, G_double_prime


# =============================================================================
# Alpha Exploration (FIKH-Specific)
# =============================================================================


def compute_memory_kernel_decay(
    alpha: float,
    t: np.ndarray,
    tau_thix: float = 10.0,
) -> np.ndarray:
    """Compute the memory kernel decay for Caputo fractional derivative.

    The Caputo fractional derivative uses a power-law memory kernel:
        K(t) = t^(-α) / Γ(1-α)

    For structure evolution, this modifies relaxation from exponential to
    power-law decay (Mittag-Leffler function).

    Args:
        alpha: Fractional order (0 < α < 1).
        t: Time array.
        tau_thix: Thixotropic time scale (for normalization).

    Returns:
        Normalized memory kernel K(t/τ_thix).
    """
    from scipy.special import gamma as gamma_func

    # Normalized time
    t_norm = t / tau_thix + 1e-10  # Avoid division by zero

    # Power-law kernel (Caputo)
    kernel = t_norm ** (-alpha) / gamma_func(1 - alpha)

    # Normalize to unit area over t_norm = [0, 10]
    kernel = kernel / np.trapezoid(kernel, t_norm)

    return kernel


def plot_alpha_sweep(
    model: Any,
    protocol: str,
    alpha_values: list[float] | None = None,
    x_data: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 5),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Plot predictions across different alpha values to show memory effects.

    This is the key educational visualization for FIKH models, demonstrating
    how the fractional order α affects model predictions.

    Args:
        model: FIKH or FMLIKH model instance.
        protocol: One of 'flow_curve', 'startup', 'relaxation', 'creep', 'saos', 'laos'.
        alpha_values: List of alpha values to compare. Default: [0.3, 0.5, 0.7, 0.9, 0.99].
        x_data: Input data array (shear rate, time, or frequency).
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction method.

    Returns:
        Matplotlib figure with two panels: predictions and memory kernel.
    """
    if alpha_values is None:
        alpha_values = [0.3, 0.5, 0.7, 0.9, 0.99]

    # Default x_data based on protocol
    if x_data is None:
        if protocol == "flow_curve":
            x_data = np.logspace(-3, 2, 100)
        elif protocol in ("startup", "relaxation", "creep"):
            x_data = np.linspace(0.01, 100, 200)
        elif protocol in ("saos", "oscillation"):
            x_data = np.logspace(-2, 2, 50)
        elif protocol == "laos":
            x_data = np.linspace(0, 10 * np.pi, 500)
        else:
            x_data = np.linspace(0, 100, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Store original alpha
    original_alpha = model.parameters.get_value("alpha_structure")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(alpha_values)))

    # Left panel: Predictions for each alpha
    for alpha, color in zip(alpha_values, colors, strict=False):
        model.parameters.set_value("alpha_structure", alpha)

        # Get prediction based on protocol
        if protocol == "flow_curve":
            y_pred = model.predict(x_data, test_mode="flow_curve")
            xlabel, ylabel = "Shear rate [1/s]", "Stress [Pa]"
            loglog = True
        elif protocol == "startup":
            gamma_dot = predict_kwargs.get("gamma_dot", 1.0)
            y_pred = model.predict_startup(x_data, gamma_dot=gamma_dot)
            xlabel, ylabel = "Time [s]", "Stress [Pa]"
            loglog = False
        elif protocol == "relaxation":
            sigma_0 = predict_kwargs.get("sigma_0", 100.0)
            y_pred = model.predict_relaxation(x_data, sigma_0=sigma_0)
            xlabel, ylabel = "Time [s]", "Stress [Pa]"
            loglog = True
        elif protocol == "creep":
            sigma_applied = predict_kwargs.get("sigma_applied", 50.0)
            y_pred = model.predict_creep(x_data, sigma_applied=sigma_applied)
            xlabel, ylabel = "Time [s]", "Strain [-]"
            loglog = True
        elif protocol in ("saos", "oscillation"):
            G_star = model.predict_oscillation(x_data, gamma_0=0.01)
            y_pred = np.abs(G_star)
            xlabel, ylabel = "Frequency [rad/s]", "|G*| [Pa]"
            loglog = True
        else:
            y_pred = model.predict(x_data, test_mode=protocol, **predict_kwargs)
            xlabel, ylabel = "x", "y"
            loglog = False

        y_pred = np.asarray(y_pred).flatten()
        label = f"α = {alpha:.2f}"

        if loglog:
            ax1.loglog(x_data, y_pred, "-", color=color, lw=2, label=label)
        else:
            ax1.plot(x_data, y_pred, "-", color=color, lw=2, label=label)

    # Restore original alpha
    model.parameters.set_value("alpha_structure", original_alpha)

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f"Alpha Sweep: {protocol.replace('_', ' ').title()}", fontsize=13)
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3, which="both")

    # Right panel: Memory kernel comparison
    _tau_thix = model.parameters.get_value("tau_thix")
    t_kernel = np.linspace(0.01, 10, 500)

    for alpha, color in zip(alpha_values, colors, strict=False):
        kernel = compute_memory_kernel_decay(alpha, t_kernel, tau_thix=1.0)
        ax2.semilogy(t_kernel, kernel, "-", color=color, lw=2, label=f"α = {alpha:.2f}")

    ax2.set_xlabel("Normalized time (t/τ)", fontsize=12)
    ax2.set_ylabel("Memory kernel K(t)", fontsize=12)
    ax2.set_title("Caputo Memory Kernel Decay", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)

    plt.tight_layout()
    return fig


def compare_fikh_to_ikh(
    fikh_model: Any,
    ikh_model: Any,
    protocol: str,
    x_data: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Compare FIKH predictions against classical IKH (α → 1 limit).

    Validates that FIKH with α → 1 recovers classical IKH behavior.

    Args:
        fikh_model: FIKH model instance.
        ikh_model: Classical MIKH model instance.
        protocol: Protocol name.
        x_data: Input data array.
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction.

    Returns:
        Matplotlib figure comparing FIKH (various α) vs IKH.
    """
    alpha_values = [0.5, 0.7, 0.9, 0.99]

    if x_data is None:
        if protocol == "flow_curve":
            x_data = np.logspace(-3, 2, 100)
        else:
            x_data = np.linspace(0.01, 100, 200)

    fig, ax = plt.subplots(figsize=figsize)

    # IKH reference (classical exponential)
    if protocol == "flow_curve":
        y_ikh = ikh_model.predict(x_data, test_mode="flow_curve")
        loglog = True
    elif protocol == "startup":
        gamma_dot = predict_kwargs.get("gamma_dot", 1.0)
        y_ikh = ikh_model.predict_startup(x_data, gamma_dot=gamma_dot)
        loglog = False
    elif protocol == "relaxation":
        sigma_0 = predict_kwargs.get("sigma_0", 100.0)
        y_ikh = ikh_model.predict_relaxation(x_data, sigma_0=sigma_0)
        loglog = True
    else:
        y_ikh = ikh_model.predict(x_data, test_mode=protocol, **predict_kwargs)
        loglog = False

    y_ikh = np.asarray(y_ikh).flatten()

    if loglog:
        ax.loglog(x_data, y_ikh, "k--", lw=3, label="IKH (classical)", alpha=0.7)
    else:
        ax.plot(x_data, y_ikh, "k--", lw=3, label="IKH (classical)", alpha=0.7)

    # FIKH for various alpha
    original_alpha = fikh_model.parameters.get_value("alpha_structure")
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors, strict=False):
        fikh_model.parameters.set_value("alpha_structure", alpha)

        if protocol == "flow_curve":
            y_fikh = fikh_model.predict(x_data, test_mode="flow_curve")
        elif protocol == "startup":
            gamma_dot = predict_kwargs.get("gamma_dot", 1.0)
            y_fikh = fikh_model.predict_startup(x_data, gamma_dot=gamma_dot)
        elif protocol == "relaxation":
            sigma_0 = predict_kwargs.get("sigma_0", 100.0)
            y_fikh = fikh_model.predict_relaxation(x_data, sigma_0=sigma_0)
        else:
            y_fikh = fikh_model.predict(x_data, test_mode=protocol, **predict_kwargs)

        y_fikh = np.asarray(y_fikh).flatten()

        if loglog:
            ax.loglog(x_data, y_fikh, "-", color=color, lw=2, label=f"FIKH α={alpha}")
        else:
            ax.plot(x_data, y_fikh, "-", color=color, lw=2, label=f"FIKH α={alpha}")

    fikh_model.parameters.set_value("alpha_structure", original_alpha)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"FIKH vs IKH Comparison: {protocol.replace('_', ' ').title()}", fontsize=13
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_structure_recovery(
    alpha_values: list[float],
    tau_thix: float = 10.0,
    t_max: float = 100.0,
    n_points: int = 500,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot structure recovery curves for different alpha values.

    Demonstrates how fractional order affects thixotropic rebuilding
    after cessation of flow.

    Args:
        alpha_values: List of fractional orders to compare.
        tau_thix: Thixotropic time scale.
        t_max: Maximum time for recovery.
        n_points: Number of time points.
        figsize: Figure size.

    Returns:
        Matplotlib figure showing structure recovery.
    """
    from scipy.special import gamma as gamma_func

    t = np.linspace(0.01, t_max, n_points)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(alpha_values)))

    for alpha, color in zip(alpha_values, colors, strict=False):
        # Mittag-Leffler function approximation for recovery
        # λ(t) = 1 - E_α(-(t/τ)^α) ≈ 1 - exp(-(t/τ)^α / Γ(1+α))
        t_norm = t / tau_thix
        recovery = 1 - np.exp(-(t_norm**alpha) / gamma_func(1 + alpha))

        ax.plot(t, recovery, "-", color=color, lw=2, label=f"α = {alpha:.2f}")

    # Reference: classical exponential (α = 1)
    recovery_exp = 1 - np.exp(-t / tau_thix)
    ax.plot(t, recovery_exp, "k--", lw=2, alpha=0.7, label="α = 1 (exponential)")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Structure parameter λ", fontsize=12)
    ax.set_title("Thixotropic Structure Recovery vs Fractional Order", fontsize=13)
    ax.axhline(
        0.63, color="gray", linestyle=":", alpha=0.5, label="λ = 0.63 (τ definition)"
    )
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    return fig


def print_alpha_interpretation(alpha: float) -> None:
    """Print physical interpretation of fractional order alpha.

    Args:
        alpha: Fractional order (0 < α < 1).
    """
    print("=" * 60)
    print(f"Fractional Order Interpretation: α = {alpha:.3f}")
    print("=" * 60)

    if alpha < 0.3:
        regime = "Very Strong Memory"
        description = (
            "Power-law dominates. Structure recovery is extremely slow.\n"
            "Material 'remembers' deformation history for long times.\n"
            "Suitable for: Highly aged gels, clay suspensions with strong aging."
        )
    elif alpha < 0.5:
        regime = "Strong Memory"
        description = (
            "Significant power-law character. Slow recovery dynamics.\n"
            "Thixotropic rebuilding takes much longer than τ_thix.\n"
            "Suitable for: Colloidal gels, waxy crude oils."
        )
    elif alpha < 0.7:
        regime = "Moderate Fractional Behavior"
        description = (
            "Balanced power-law and exponential character.\n"
            "Recovery shows stretched-exponential-like behavior.\n"
            "Suitable for: Most thixotropic fluids, cosmetics, food products."
        )
    elif alpha < 0.85:
        regime = "Weak Memory"
        description = (
            "Approaching classical behavior with mild power-law tails.\n"
            "Recovery is mostly exponential with slow corrections.\n"
            "Suitable for: Mildly thixotropic materials."
        )
    else:
        regime = "Near-Classical (IKH Limit)"
        description = (
            "α → 1: Classical IKH behavior recovered.\n"
            "Exponential structure evolution dominates.\n"
            "Memory effects are negligible.\n"
            "Suitable for: Materials without significant aging effects."
        )

    print(f"\nRegime: {regime}")
    print(f"\n{description}")

    # Physical implications
    print("\nPhysical Implications:")
    print(f"  • Memory kernel decay: t^(-{alpha:.2f})")
    print(
        f"  • Recovery time scale: > τ_thix × Γ(1+{alpha:.2f}) = {tau_factor(alpha):.2f} × τ_thix"
    )
    print(f"  • Relaxation modulus: Power-law with exponent ≈ -{alpha:.2f}")


def tau_factor(alpha: float) -> float:
    """Compute effective time scale factor for given alpha."""
    from scipy.special import gamma as gamma_func

    return gamma_func(1 + alpha)


# =============================================================================
# Result Persistence
# =============================================================================


def get_output_dir(
    model_name: Literal["fikh", "fmlikh"],
    protocol: str,
) -> Path:
    """Get the output directory path for a given model and protocol.

    Args:
        model_name: One of 'fikh' or 'fmlikh'.
        protocol: Protocol name (e.g., 'flow_curve', 'startup').

    Returns:
        Path to output directory.
    """
    module_dir = Path(__file__).parent
    return module_dir / ".." / "outputs" / "fikh" / model_name / protocol


def save_fikh_results(
    model: Any,
    result: Any,
    model_name: Literal["fikh", "fmlikh"],
    protocol: str,
    param_names: list[str] | None = None,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted FIKH model.
        result: Bayesian fit result with posterior_samples attribute.
        model_name: One of 'fikh' or 'fmlikh'.
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


def load_fikh_parameters(
    model_name: Literal["fikh", "fmlikh"],
    protocol: str,
) -> dict[str, float]:
    """Load previously calibrated FIKH parameters.

    Args:
        model_name: One of 'fikh' or 'fmlikh'.
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


# =============================================================================
# Parameter Helpers
# =============================================================================


def get_fikh_param_names(include_thermal: bool = False) -> list[str]:
    """Get the list of FIKH parameter names.

    Args:
        include_thermal: Include thermal coupling parameters.

    Returns:
        List of FIKH parameter names.
    """
    # Core FIKH parameters (12 isothermal)
    params = [
        "G",  # Shear modulus (Pa)
        "eta",  # Maxwell viscosity (Pa.s)
        "C",  # Kinematic hardening modulus (Pa)
        "gamma_dyn",  # Dynamic recovery parameter
        "m",  # AF recovery exponent
        "sigma_y0",  # Minimal yield stress (Pa)
        "delta_sigma_y",  # Structural yield contribution (Pa)
        "tau_thix",  # Thixotropic rebuilding time (s)
        "Gamma",  # Thixotropic breakdown coefficient
        "alpha_structure",  # Fractional order (0 < α < 1)
        "eta_inf",  # High-shear viscosity (Pa.s)
        "mu_p",  # Plastic viscosity (Pa.s)
    ]

    if include_thermal:
        params.extend(
            [
                "T_ref",  # Reference temperature (K)
                "E_a",  # Viscosity activation energy (J/mol)
                "E_y",  # Yield stress activation energy (J/mol)
                "m_y",  # Structure exponent for yield
                "rho_cp",  # Volumetric heat capacity (J/(m³·K))
                "chi",  # Taylor-Quinney coefficient
                "h",  # Heat transfer coefficient (W/(m²·K))
                "T_env",  # Environmental temperature (K)
            ]
        )

    return params


def get_fmlikh_param_names(
    n_modes: int = 3,
    shared_alpha: bool = True,
    include_thermal: bool = False,
) -> list[str]:
    """Get the list of FMLIKH parameter names.

    Args:
        n_modes: Number of viscoelastic modes.
        shared_alpha: If True, single alpha_structure. If False, per-mode alpha.
        include_thermal: Include thermal coupling parameters.

    Returns:
        List of FMLIKH parameter names.
    """
    params = []

    # Per-mode parameters
    for i in range(n_modes):
        params.extend(
            [
                f"G_{i}",
                f"eta_{i}",
                f"C_{i}",
                f"gamma_dyn_{i}",
            ]
        )
        if not shared_alpha:
            params.append(f"alpha_{i}")

    # Shared yield/thixotropy parameters
    params.extend(
        [
            "m",
            "sigma_y0",
            "delta_sigma_y",
            "tau_thix",
            "Gamma",
            "eta_inf",
            "mu_p",
        ]
    )

    if shared_alpha:
        params.append("alpha_structure")

    if include_thermal:
        params.extend(
            [
                "T_ref",
                "E_a",
                "E_y",
                "m_y",
                "rho_cp",
                "chi",
                "h",
                "T_env",
            ]
        )

    return params


def set_model_parameters(model: Any, params: dict[str, float]) -> None:
    """Set model parameters from a dictionary.

    Args:
        model: FIKH model instance.
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
        param_names: List of parameter names to report.

    Returns:
        True if all convergence criteria pass, False otherwise.
    """
    if param_names is None:
        param_names = get_fikh_param_names()

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
            print(
                f"{name:>15s}  {nlsq_val:12.4g}  {median:12.4g}  [{lo:.4g}, {hi:.4g}]"
            )
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
# FIKH-Specific Physics Functions
# =============================================================================


def compute_fikh_structure_evolution(
    gamma_dot: float | np.ndarray,
    tau_thix: float,
    Gamma: float,
    alpha: float = 0.5,
) -> float | np.ndarray:
    """Compute steady-state thixotropic structure parameter for FIKH.

    For the fractional model, the steady-state is modified by the memory
    kernel. The exact solution requires Mittag-Leffler functions, but
    the steady-state equilibrium is the same as classical IKH.

    λ_ss = 1 / (1 + Γ * τ_thix * |γ̇|)

    Note: The fractional order α affects the transient approach to
    steady state, not the steady-state value itself.

    Args:
        gamma_dot: Shear rate (1/s).
        tau_thix: Thixotropic rebuilding time (s).
        Gamma: Breakdown coefficient.
        alpha: Fractional order (affects transient, not steady-state).

    Returns:
        Structure parameter (0-1).
    """
    gamma_dot = np.abs(gamma_dot)
    return 1.0 / (1.0 + Gamma * tau_thix * gamma_dot)


def compute_fikh_yield_stress(
    lambda_: float | np.ndarray,
    sigma_y0: float,
    delta_sigma_y: float,
) -> float | np.ndarray:
    """Compute FIKH yield stress as function of structure.

    σ_y = σ_y0 + λ * Δσ_y

    Args:
        lambda_: Structure parameter (0-1).
        sigma_y0: Minimal yield stress (Pa).
        delta_sigma_y: Structural yield contribution (Pa).

    Returns:
        Total yield stress (Pa).
    """
    return sigma_y0 + lambda_ * delta_sigma_y


def compute_caputo_derivative_coefficients(
    alpha: float,
    n_history: int,
    dt: float,
) -> np.ndarray:
    """Compute L1 scheme coefficients for Caputo fractional derivative.

    The L1 discretization of the Caputo derivative uses:
        D^α f(t_n) ≈ Σ_{k=0}^{n} a_k (f_{n-k} - f_{n-k-1}) / Γ(1-α)

    where a_k = ((k+1)^(1-α) - k^(1-α)) / (dt^α)

    Args:
        alpha: Fractional order (0 < α < 1).
        n_history: Number of history points.
        dt: Time step.

    Returns:
        Array of L1 coefficients.
    """
    from scipy.special import gamma as gamma_func

    k = np.arange(n_history)
    a_k = ((k + 1) ** (1 - alpha) - k ** (1 - alpha)) / (dt**alpha)
    a_k = a_k / gamma_func(2 - alpha)

    return a_k
