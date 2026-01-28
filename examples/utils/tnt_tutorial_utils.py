"""Shared utilities for TNT (Transient Network Theory) tutorial notebooks.

Provides consistent data loading, physics visualizations, result saving,
diagnostic printing, and parameter helpers across all 30 TNT tutorials
(TNTSingleMode, TNTCates, TNTLoopBridge, TNTMultiSpecies, TNTStickyRouse).

Data Sources:
- ML-IKH Experimental data.xlsx: Wei et al. 2018 J. Rheol (flow curves, creep)
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 (startup, LAOS)
- epstein.csv: Metal-organic coordination networks (SAOS)
- rel_lapo_1800.csv: Laponite clay gel relaxation
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Section 1: Data Loaders
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

    col_map = {
        "ARG2_up": (0, 1),
        "ARG2_down": (2, 3),
        "ARES_up": (5, 6),
        "ARES_down": (7, 8),
    }

    col_rate, col_stress = col_map[instrument]
    data = df.iloc[3:, [col_rate, col_stress]].dropna().astype(float)

    gamma_dot = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    sort_idx = np.argsort(gamma_dot)
    return gamma_dot[sort_idx], stress[sort_idx]


def load_pnas_startup(
    gamma_dot: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load startup shear data from PNAS Digital Rheometer Twin dataset.

    Args:
        gamma_dot: Shear rate to load. Available: 0.056, 0.32, 1, 56.2, 100.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rate_sheets = {
        0.056: "StartUp_0.056",
        0.32: "StartUp_0.32",
        1.0: "StartUp_1",
        56.2: "StartUp_56.2",
        100.0: "StartUp_100",
    }

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
    data = df.iloc[3:, [0, 1]].dropna().astype(float)

    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    if len(time) > 500:
        indices = np.linspace(0, len(time) - 1, 500, dtype=int)
        time = time[indices]
        stress = stress[indices]

    return time, stress


def load_ml_ikh_creep(
    stress_pair_index: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Load step stress (creep) data from ML-IKH Excel (Fig A2).

    Args:
        stress_pair_index: Which stress jump to load (0-11).

    Returns:
        Tuple of (time, shear_rate, initial_stress, final_stress) in
        (s, 1/s, Pa, Pa).
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

    col_time = stress_pair_index * 2
    col_rate = stress_pair_index * 2 + 1

    initial_stress = float(df.iloc[2, col_rate])
    final_stress = float(df.iloc[3, col_rate])

    data = df.iloc[5:, [col_time, col_rate]].dropna().astype(float)

    time = data.iloc[:, 0].values
    shear_rate = data.iloc[:, 1].values

    return time, shear_rate, initial_stress, final_stress


def load_pnas_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LAOS data from PNAS Digital Rheometer Twin dataset.

    Args:
        omega: Angular frequency (1, 3, or 5 rad/s).
        strain_amplitude_index: Index for strain amplitude (0-11, increasing).

    Returns:
        Tuple of (time, strain, stress) arrays in (s, -, Pa).
    """
    omega_sheets = {
        1.0: "LAOS_w1",
        3.0: "LAOS_w3",
        5.0: "LAOS_w5",
    }

    if omega not in omega_sheets:
        raise ValueError(f"omega must be 1, 3, or 5, got {omega}")

    if strain_amplitude_index not in range(12):
        raise ValueError(
            f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}"
        )

    sheet = omega_sheets[omega]
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"PNAS data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_excel(data_path, sheet_name=sheet, header=None)

    col_time = strain_amplitude_index * 4
    col_strain = strain_amplitude_index * 4 + 1
    col_stress = strain_amplitude_index * 4 + 2

    data = df.iloc[3:, [col_time, col_strain, col_stress]].dropna().astype(float)

    time = data.iloc[:, 0].values
    strain = data.iloc[:, 1].values
    stress = data.iloc[:, 2].values

    if len(time) > 1000:
        indices = np.linspace(0, len(time) - 1, 1000, dtype=int)
        time = time[indices]
        strain = strain[indices]
        stress = stress[indices]

    return time, strain, stress


def load_epstein_saos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAOS data from Epstein metal-organic coordination network.

    Returns:
        Tuple of (omega, G_prime, G_double_prime) in (rad/s, Pa, Pa).
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir / ".." / "data" / "oscillation" / "metal_networks" / "epstein.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Epstein data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path, sep="\t")

    omega = df.iloc[:, 0].values.astype(float)
    G_prime = df.iloc[:, 1].values.astype(float)
    G_double_prime = df.iloc[:, 2].values.astype(float)

    sort_idx = np.argsort(omega)
    return omega[sort_idx], G_prime[sort_idx], G_double_prime[sort_idx]


def load_laponite_relaxation(
    aging_time: int = 1800,
) -> tuple[np.ndarray, np.ndarray]:
    """Load stress relaxation data from laponite clay gel.

    Args:
        aging_time: Aging time in seconds (available: 1800).

    Returns:
        Tuple of (time, G_t) in (s, Pa).
    """
    module_dir = Path(__file__).parent
    data_path = (
        module_dir
        / ".."
        / "data"
        / "relaxation"
        / "clays"
        / f"rel_lapo_{aging_time}.csv"
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Laponite data not found. Expected at: {data_path.resolve()}"
        )

    df = pd.read_csv(data_path, sep="\t")

    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    sort_idx = np.argsort(time)
    return time[sort_idx], G_t[sort_idx]


# =============================================================================
# Section 2: TNT Physics Visualizations
# =============================================================================


def plot_breakage_rate_comparison(
    nu_values: list[float] | None = None,
    tau_b: float = 1.0,
    stretch_range: tuple[float, float] = (1.0, 5.0),
    n_points: int = 200,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot Bell breakage rate beta(stretch) for different force sensitivities.

    Shows how the force sensitivity nu controls the enhancement of bond
    breakage under chain stretch. Higher nu = stronger shear thinning.

    Args:
        nu_values: Force sensitivities to compare.
        tau_b: Base bond lifetime (s).
        stretch_range: (min, max) stretch ratio.
        n_points: Number of points.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if nu_values is None:
        nu_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    stretch = np.linspace(stretch_range[0], stretch_range[1], n_points)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(nu_values)))

    for nu, color in zip(nu_values, colors):
        # Bell model: beta = (1/tau_b) * exp(nu * (stretch - 1))
        beta = (1.0 / tau_b) * np.exp(nu * (stretch - 1.0))
        tau_eff = tau_b * np.exp(-nu * (stretch - 1.0))

        ax1.semilogy(stretch, beta, "-", color=color, lw=2, label=f"nu = {nu}")
        ax2.semilogy(stretch, tau_eff, "-", color=color, lw=2, label=f"nu = {nu}")

    # Constant breakage reference
    ax1.axhline(1.0 / tau_b, color="gray", ls="--", alpha=0.5, label="Constant")
    ax2.axhline(tau_b, color="gray", ls="--", alpha=0.5, label="Constant")

    ax1.set_xlabel("Chain stretch ratio", fontsize=12)
    ax1.set_ylabel("Breakage rate beta [1/s]", fontsize=12)
    ax1.set_title("Bell Breakage Rate vs Stretch", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Chain stretch ratio", fontsize=12)
    ax2.set_ylabel("Effective lifetime [s]", fontsize=12)
    ax2.set_title("Effective Bond Lifetime vs Stretch", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_conformation_tensor_evolution(
    model: Any,
    gamma_dot: float = 1.0,
    t_end: float = 20.0,
    n_points: int = 500,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot conformation tensor components during startup shear.

    Shows S_xx, S_yy, S_xy evolution and chain stretch lambda(t).

    Args:
        model: Fitted TNTSingleMode or subclass.
        gamma_dot: Applied shear rate (1/s).
        t_end: End time (s).
        n_points: Number of time points.
        figsize: Figure size.

    Returns:
        Matplotlib figure with tensor component and stretch panels.
    """
    time = np.linspace(0, t_end, n_points)

    # Get startup predictions
    try:
        result = model.simulate_startup(time, gamma_dot=gamma_dot)
        if isinstance(result, tuple) and len(result) >= 2:
            stress = np.asarray(result[1]) if len(result) > 1 else np.asarray(result[0])
        else:
            stress = np.asarray(result)
    except (AttributeError, TypeError):
        stress = np.asarray(model.predict(time, test_mode="startup", gamma_dot=gamma_dot))

    # Analytical conformation tensor for constant breakage (Tanaka-Edwards)
    tau_b = float(model.parameters.get_value("tau_b"))
    G = float(model.parameters.get_value("G"))
    Wi = gamma_dot * tau_b

    # S_xx(t) = 1 + 2*Wi^2*(1 - exp(-t/tau_b))
    # S_xy(t) = Wi*(1 - exp(-t/tau_b))
    # S_yy(t) = 1 (constant for simple shear)
    exp_decay = np.exp(-time / tau_b)
    S_xx = 1.0 + 2.0 * Wi**2 * (1.0 - exp_decay)
    S_xy = Wi * (1.0 - exp_decay)
    S_yy = np.ones_like(time)
    stretch = np.sqrt((S_xx + S_yy + 1.0) / 3.0)  # lambda = sqrt(tr(S)/3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.plot(time, S_xx, "-", lw=2, color="C0", label=r"$S_{xx}$")
    ax1.plot(time, S_yy, "-", lw=2, color="C1", label=r"$S_{yy}$")
    ax1.plot(time, S_xy, "-", lw=2, color="C2", label=r"$S_{xy}$")
    ax1.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Equilibrium")
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Conformation tensor component", fontsize=12)
    ax1.set_title(
        f"Conformation Tensor (Wi = {Wi:.2f})", fontsize=13
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, stretch, "-", lw=2, color="C3", label=r"$\lambda = \sqrt{\mathrm{tr}(S)/3}$")
    ax2.axhline(1.0, color="gray", ls="--", alpha=0.5, label="Equilibrium")
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Chain stretch ratio", fontsize=12)
    ax2.set_title("Chain Stretch Evolution", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_weissenberg_number_effects(
    model: Any,
    gamma_dot_range: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot flow curve colored by Weissenberg number regime.

    Wi < 1: Linear viscoelastic regime (Newtonian-like)
    Wi ~ 1: Onset of nonlinearity
    Wi > 1: Nonlinear regime (shear thinning for non-constant breakage)

    Args:
        model: Fitted TNT model with tau_b parameter.
        gamma_dot_range: Shear rate array. Default: logspace(-3, 3).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if gamma_dot_range is None:
        gamma_dot_range = np.logspace(-3, 3, 200)

    tau_b = float(model.parameters.get_value("tau_b"))
    Wi = gamma_dot_range * tau_b

    stress = np.asarray(model.predict_flow_curve(gamma_dot_range))

    fig, ax = plt.subplots(figsize=figsize)

    # Color by Wi regime
    linear = Wi < 0.5
    transition = (Wi >= 0.5) & (Wi <= 2.0)
    nonlinear = Wi > 2.0

    if np.any(linear):
        ax.loglog(gamma_dot_range[linear], stress[linear], "o",
                  color="C0", ms=4, label="Wi < 0.5 (linear)")
    if np.any(transition):
        ax.loglog(gamma_dot_range[transition], stress[transition], "s",
                  color="C1", ms=4, label="0.5 < Wi < 2 (transition)")
    if np.any(nonlinear):
        ax.loglog(gamma_dot_range[nonlinear], stress[nonlinear], "^",
                  color="C2", ms=4, label="Wi > 2 (nonlinear)")

    # Mark Wi = 1 crossover
    ax.axvline(1.0 / tau_b, color="red", ls="--", alpha=0.5,
               label=f"Wi = 1 (1/tau_b = {1.0/tau_b:.2g} 1/s)")

    ax.set_xlabel("Shear rate [1/s]", fontsize=12)
    ax.set_ylabel("Stress [Pa]", fontsize=12)
    ax.set_title("Flow Curve with Weissenberg Number Regimes", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_variant_comparison(
    x_data: np.ndarray,
    protocol: str,
    models: dict[str, Any] | None = None,
    y_data: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Plot predictions from multiple TNT breakage variants on same axes.

    Args:
        x_data: Independent variable array.
        protocol: Protocol name (flow_curve, startup, etc.).
        models: Dict of {label: model} to compare. If None, creates
            constant and bell variants.
        y_data: Optional experimental data to overlay.
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if y_data is not None:
        if protocol in ("flow_curve", "relaxation", "creep", "saos"):
            ax.loglog(x_data, y_data, "ko", ms=7, label="Data", zorder=5)
        else:
            ax.plot(x_data, y_data, "ko", ms=5, label="Data", zorder=5)

    if models is None:
        return fig

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(models)))

    for (label, model), color in zip(models.items(), colors):
        try:
            if protocol == "flow_curve":
                y_pred = np.asarray(model.predict_flow_curve(x_data))
            elif protocol == "saos":
                G_p, G_pp = model.predict_saos(x_data)
                y_pred = np.sqrt(np.asarray(G_p)**2 + np.asarray(G_pp)**2)
            else:
                y_pred = np.asarray(
                    model.predict(x_data, test_mode=protocol, **predict_kwargs)
                )
        except Exception:
            continue

        y_pred = np.asarray(y_pred).flatten()

        if protocol in ("flow_curve", "relaxation", "creep", "saos"):
            ax.loglog(x_data, y_pred, "-", color=color, lw=2.5, label=label)
        else:
            ax.plot(x_data, y_pred, "-", color=color, lw=2.5, label=label)

    ax.set_xlabel(_get_xlabel(protocol), fontsize=12)
    ax.set_ylabel(_get_ylabel(protocol), fontsize=12)
    ax.set_title(f"TNT Variant Comparison: {protocol.replace('_', ' ').title()}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_bell_nu_sweep(
    model: Any,
    nu_values: list[float] | None = None,
    x_data: np.ndarray | None = None,
    protocol: str = "flow_curve",
    figsize: tuple[float, float] = (10, 6),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Plot predictions for different Bell force sensitivity values.

    Analogous to FIKH alpha sweep but for Bell parameter nu.

    Args:
        model: TNTSingleMode with breakage="bell".
        nu_values: Force sensitivity values to compare.
        x_data: Independent variable array.
        protocol: Protocol name.
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction.

    Returns:
        Matplotlib figure.
    """
    if nu_values is None:
        nu_values = [0.5, 1.0, 2.0, 5.0, 10.0]

    if x_data is None:
        if protocol == "flow_curve":
            x_data = np.logspace(-3, 3, 200)
        elif protocol in ("startup", "relaxation", "creep"):
            x_data = np.linspace(0.01, 50, 300)
        elif protocol in ("saos", "oscillation"):
            x_data = np.logspace(-2, 2, 100)
        else:
            x_data = np.linspace(0, 30, 500)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(nu_values)))

    original_nu = float(model.parameters.get_value("nu"))

    for nu, color in zip(nu_values, colors):
        model.parameters.set_value("nu", nu)
        try:
            if protocol == "flow_curve":
                y_pred = np.asarray(model.predict_flow_curve(x_data))
            elif protocol == "saos":
                # SAOS is independent of nu (linear regime)
                G_p, G_pp = model.predict_saos(x_data)
                y_pred = np.sqrt(np.asarray(G_p)**2 + np.asarray(G_pp)**2)
            else:
                y_pred = np.asarray(
                    model.predict(x_data, test_mode=protocol, **predict_kwargs)
                )
        except Exception:
            continue

        y_pred = np.asarray(y_pred).flatten()

        if protocol in ("flow_curve", "relaxation", "saos"):
            ax.loglog(x_data, y_pred, "-", color=color, lw=2,
                      label=f"nu = {nu}")
        else:
            ax.plot(x_data, y_pred, "-", color=color, lw=2,
                    label=f"nu = {nu}")

    model.parameters.set_value("nu", original_nu)

    ax.set_xlabel(_get_xlabel(protocol), fontsize=12)
    ax.set_ylabel(_get_ylabel(protocol), fontsize=12)
    ax.set_title(f"Bell Force Sensitivity Sweep: {protocol.replace('_', ' ').title()}", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_mode_decomposition(
    model: Any,
    x_data: np.ndarray,
    protocol: str,
    figsize: tuple[float, float] = (10, 6),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Plot per-species or per-mode stress contributions for multi-mode models.

    Works with TNTMultiSpecies and TNTStickyRouse by temporarily isolating
    each mode/species.

    Args:
        model: Fitted multi-mode TNT model.
        x_data: Independent variable array.
        protocol: Protocol name.
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction.

    Returns:
        Matplotlib figure with total and per-mode contributions.
    """
    fig, ax = plt.subplots(figsize=figsize)
    logscale = protocol in ("flow_curve", "relaxation", "saos")

    # Total prediction
    try:
        if protocol == "flow_curve":
            y_total = np.asarray(model.predict_flow_curve(x_data))
        elif protocol == "saos":
            G_p, G_pp = model.predict_saos(x_data)
            y_total = np.sqrt(np.asarray(G_p)**2 + np.asarray(G_pp)**2)
        else:
            y_total = np.asarray(
                model.predict(x_data, test_mode=protocol, **predict_kwargs)
            )
    except Exception:
        y_total = np.zeros_like(x_data)

    y_total = y_total.flatten()

    if logscale:
        ax.loglog(x_data, y_total, "k-", lw=3, label="Total", zorder=5)
    else:
        ax.plot(x_data, y_total, "k-", lw=3, label="Total", zorder=5)

    # Determine number of modes
    n_modes = _get_n_modes(model)
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_modes, 2)))

    # Per-mode analytical contribution (Maxwell superposition for SAOS)
    if protocol == "saos" and n_modes > 0:
        eta_s = float(model.parameters.get_value("eta_s"))
        for k in range(n_modes):
            G_k, tau_k = _get_mode_params(model, k)
            G_p_k, G_pp_k = compute_maxwell_moduli(x_data, G_k, tau_k, 0.0)
            G_star_k = np.sqrt(G_p_k**2 + G_pp_k**2)

            if logscale:
                ax.loglog(x_data, G_star_k, "--", color=colors[k], lw=2,
                          label=f"Mode {k}: G={G_k:.0f}, tau={tau_k:.2g}")
            else:
                ax.plot(x_data, G_star_k, "--", color=colors[k], lw=2,
                        label=f"Mode {k}: G={G_k:.0f}, tau={tau_k:.2g}")

        # Solvent contribution
        if eta_s > 0:
            sigma_s = eta_s * x_data
            if logscale:
                ax.loglog(x_data, sigma_s, ":", color="gray", lw=1.5,
                          label=f"Solvent (eta_s={eta_s:.2g})")

    ax.set_xlabel(_get_xlabel(protocol), fontsize=12)
    ax.set_ylabel(_get_ylabel(protocol), fontsize=12)
    ax.set_title(f"Mode Decomposition: {protocol.replace('_', ' ').title()}", fontsize=13)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def print_nu_interpretation(nu: float) -> None:
    """Print physical interpretation of Bell force sensitivity parameter.

    Args:
        nu: Force sensitivity (dimensionless).
    """
    print("=" * 60)
    print(f"Bell Force Sensitivity: nu = {nu:.2f}")
    print("=" * 60)

    if nu < 0.5:
        regime = "Weak Force Sensitivity"
        description = (
            "Bond breakage is nearly force-independent.\n"
            "Shear thinning is minimal (UCM-like behavior).\n"
            "Suitable for: Weakly associating polymers, dilute solutions."
        )
    elif nu < 2.0:
        regime = "Moderate Force Sensitivity"
        description = (
            "Bond lifetime decreases moderately with stretch.\n"
            "Moderate shear thinning at Wi > 1.\n"
            "Suitable for: Typical associative polymer networks."
        )
    elif nu < 5.0:
        regime = "Strong Force Sensitivity"
        description = (
            "Significant force-enhanced breakage.\n"
            "Strong shear thinning; stress maximum possible.\n"
            "Suitable for: Telechelic polymers, biological gels."
        )
    else:
        regime = "Very Strong Force Sensitivity"
        description = (
            "Extreme force-enhanced breakage (catch-slip bonds).\n"
            "Very strong shear thinning; stress overshoot likely.\n"
            "Suitable for: Receptor-ligand bonds, mechanosensitive gels."
        )

    print(f"\nRegime: {regime}")
    print(f"\n{description}")
    print(f"\nPhysical implications:")
    print(f"  - Effective lifetime: tau_eff = tau_b * exp(-{nu:.1f} * (lambda - 1))")
    print(f"  - At lambda = 2: tau_eff/tau_b = {np.exp(-nu):.4f}")
    print(f"  - At lambda = 3: tau_eff/tau_b = {np.exp(-2*nu):.4g}")


# =============================================================================
# Section 3: Class-Specific Visualizations
# =============================================================================


def plot_cates_cole_cole(
    model: Any,
    omega_range: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot Cole-Cole diagram for Cates (living polymer) model.

    The semicircle diagnostic: a perfect semicircle indicates single-mode
    Maxwell behavior (fast-breaking limit, tau_break << tau_rep).

    Args:
        model: Fitted TNTCates model.
        omega_range: Frequency array. Default: logspace(-3, 3).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if omega_range is None:
        omega_range = np.logspace(-3, 3, 500)

    G_p, G_pp = model.predict_saos(omega_range)
    G_p = np.asarray(G_p)
    G_pp = np.asarray(G_pp)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(G_p, G_pp, "o-", ms=3, lw=1.5, color="C0", label="TNTCates")

    # Maxwell semicircle reference
    G_0 = float(model.parameters.get_value("G_0"))
    theta = np.linspace(0, np.pi, 200)
    G_p_ref = (G_0 / 2) * (1 + np.cos(theta))
    G_pp_ref = (G_0 / 2) * np.sin(theta)
    ax.plot(G_p_ref, G_pp_ref, "k--", lw=1.5, alpha=0.5,
            label="Single-mode Maxwell")

    ax.set_xlabel("G' [Pa]", fontsize=12)
    ax.set_ylabel("G'' [Pa]", fontsize=12)
    ax.set_title("Cole-Cole Diagram (Cates Model)", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate tau_d
    tau_rep = float(model.parameters.get_value("tau_rep"))
    tau_break = float(model.parameters.get_value("tau_break"))
    tau_d = compute_cates_tau_d(tau_rep, tau_break)
    zeta = tau_break / tau_rep
    ax.set_title(
        f"Cole-Cole Diagram (tau_d = {tau_d:.3g} s, zeta = {zeta:.3g})",
        fontsize=13,
    )

    plt.tight_layout()
    return fig


def plot_loop_bridge_fraction(
    model: Any,
    gamma_dot_range: np.ndarray | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot bridge fraction f_B vs shear rate for LoopBridge model.

    Shows how shear drives the loop-bridge equilibrium toward loops
    (network softening under flow).

    Args:
        model: Fitted TNTLoopBridge model.
        gamma_dot_range: Shear rate array. Default: logspace(-3, 3).
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    if gamma_dot_range is None:
        gamma_dot_range = np.logspace(-3, 3, 200)

    tau_b = float(model.parameters.get_value("tau_b"))
    tau_a = float(model.parameters.get_value("tau_a"))
    nu = float(model.parameters.get_value("nu"))
    f_B_eq = float(model.parameters.get_value("f_B_eq"))

    # Steady-state bridge fraction under shear
    # f_B_ss = tau_a / (tau_a + tau_b_eff)
    # where tau_b_eff = tau_b * exp(-nu * (stretch - 1))
    # and stretch ~ sqrt(1 + 2*Wi^2) for constant breakage
    Wi = gamma_dot_range * tau_b
    stretch = np.sqrt(1.0 + 2.0 * Wi**2)
    tau_b_eff = tau_b * np.exp(-nu * (stretch - 1.0))
    f_B_ss = tau_a / (tau_a + tau_b_eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    ax1.semilogx(gamma_dot_range, f_B_ss, "-", lw=2.5, color="C0")
    ax1.axhline(f_B_eq, color="gray", ls="--", alpha=0.5,
                label=f"f_B_eq = {f_B_eq:.2f}")
    ax1.set_xlabel("Shear rate [1/s]", fontsize=12)
    ax1.set_ylabel("Bridge fraction f_B", fontsize=12)
    ax1.set_title("Bridge Fraction vs Shear Rate", fontsize=13)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.loglog(gamma_dot_range, tau_b_eff, "-", lw=2.5, color="C1",
               label="tau_b_eff (Bell)")
    ax2.axhline(tau_b, color="gray", ls="--", alpha=0.5,
                label=f"tau_b = {tau_b:.2g} s")
    ax2.set_xlabel("Shear rate [1/s]", fontsize=12)
    ax2.set_ylabel("Effective lifetime [s]", fontsize=12)
    ax2.set_title("Force-Dependent Bond Lifetime", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def plot_multi_species_spectrum(
    model: Any,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot discrete relaxation spectrum for MultiSpecies model.

    Shows G_i vs tau_b_i as a bar chart.

    Args:
        model: Fitted TNTMultiSpecies model.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_species = _get_n_modes(model)

    G_vals = []
    tau_vals = []
    for i in range(n_species):
        G_vals.append(float(model.parameters.get_value(f"G_{i}")))
        tau_vals.append(float(model.parameters.get_value(f"tau_b_{i}")))

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.Set2(np.linspace(0, 0.8, n_species))
    bars = ax.bar(
        range(n_species),
        G_vals,
        color=colors,
        edgecolor="black",
        alpha=0.8,
    )

    # Add tau labels on bars
    for i, (bar, tau) in enumerate(zip(bars, tau_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(G_vals) * 0.02,
            f"tau={tau:.2g} s",
            ha="center",
            fontsize=10,
        )

    ax.set_xlabel("Species index", fontsize=12)
    ax.set_ylabel("Modulus G_i [Pa]", fontsize=12)
    ax.set_title(f"Discrete Relaxation Spectrum ({n_species} species)", fontsize=13)
    ax.set_xticks(range(n_species))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_sticky_rouse_effective_times(
    model: Any,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot effective relaxation times for StickyRouse model.

    Shows tau_eff_k = max(tau_R_k, tau_s) for each Rouse mode.

    Args:
        model: Fitted TNTStickyRouse model.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    n_modes = _get_n_modes(model)
    tau_s = float(model.parameters.get_value("tau_s"))

    G_vals = []
    tau_R_vals = []
    tau_eff_vals = []
    for k in range(n_modes):
        G_k = float(model.parameters.get_value(f"G_{k}"))
        tau_R_k = float(model.parameters.get_value(f"tau_R_{k}"))
        tau_eff_k = max(tau_R_k, tau_s)

        G_vals.append(G_k)
        tau_R_vals.append(tau_R_k)
        tau_eff_vals.append(tau_eff_k)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    modes = np.arange(n_modes)

    ax1.semilogy(modes, tau_R_vals, "o-", ms=8, lw=2, color="C0",
                 label=r"$\tau_{R,k}$ (Rouse)")
    ax1.semilogy(modes, tau_eff_vals, "s--", ms=8, lw=2, color="C2",
                 label=r"$\tau_{\mathrm{eff},k} = \max(\tau_{R,k}, \tau_s)$")
    ax1.axhline(tau_s, color="red", ls=":", alpha=0.7,
                label=f"tau_s = {tau_s:.2g} s (sticker)")
    ax1.set_xlabel("Mode index k", fontsize=12)
    ax1.set_ylabel("Relaxation time [s]", fontsize=12)
    ax1.set_title("Rouse vs Effective Times", fontsize=13)
    ax1.set_xticks(modes)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(modes, G_vals, color=plt.cm.Set2(np.linspace(0, 0.8, n_modes)),
            edgecolor="black", alpha=0.8)
    ax2.set_xlabel("Mode index k", fontsize=12)
    ax2.set_ylabel("Modulus G_k [Pa]", fontsize=12)
    ax2.set_title("Mode Moduli", fontsize=13)
    ax2.set_xticks(modes)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_tnt_family_comparison(
    x_data: np.ndarray,
    protocol: str,
    models: dict[str, Any],
    y_data: np.ndarray | None = None,
    figsize: tuple[float, float] = (12, 6),
    **predict_kwargs: Any,
) -> plt.Figure:
    """Plot all 5 TNT model classes on same axes for comparison.

    Args:
        x_data: Independent variable array.
        protocol: Protocol name.
        models: Dict of {class_name: fitted_model}.
        y_data: Optional experimental data.
        figsize: Figure size.
        **predict_kwargs: Additional kwargs for prediction.

    Returns:
        Matplotlib figure.
    """
    return plot_variant_comparison(
        x_data, protocol, models=models, y_data=y_data,
        figsize=figsize, **predict_kwargs,
    )


# =============================================================================
# Section 4: Result Persistence
# =============================================================================


def get_output_dir(
    model_name: str,
    protocol: str,
) -> Path:
    """Get the output directory path for a given model and protocol.

    Args:
        model_name: One of 'single_mode', 'cates', 'loop_bridge',
            'multi_species', 'sticky_rouse'.
        protocol: Protocol name (flow_curve, startup, etc.).

    Returns:
        Path to output directory.
    """
    module_dir = Path(__file__).parent
    return module_dir / ".." / "outputs" / "tnt" / model_name / protocol


def save_tnt_results(
    model: Any,
    result: Any,
    model_name: str,
    protocol: str,
    param_names: list[str] | None = None,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted TNT model.
        result: Bayesian fit result with posterior_samples attribute.
        model_name: Model identifier for directory naming.
        protocol: Protocol name for labeling files.
        param_names: List of parameter names to save. If None, saves all.
    """
    output_dir = get_output_dir(model_name, protocol)
    os.makedirs(output_dir, exist_ok=True)

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


def load_tnt_parameters(
    model_name: str,
    protocol: str,
) -> dict[str, float]:
    """Load previously calibrated TNT parameters.

    Args:
        model_name: Model identifier.
        protocol: Protocol name.

    Returns:
        Dictionary of parameter name to value.
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
        model: TNT model instance.
        params: Dictionary of parameter name to value.
    """
    for name, value in params.items():
        try:
            model.parameters.set_value(name, value)
        except (KeyError, AttributeError):
            pass


# =============================================================================
# Section 5: Parameter Helpers
# =============================================================================


def get_tnt_single_mode_param_names(
    breakage: str = "constant",
    stress_type: str = "linear",
) -> list[str]:
    """Get parameter names for TNTSingleMode.

    Args:
        breakage: Breakage type (constant, bell, power_law, stretch_creation).
        stress_type: Stress type (linear, fene).

    Returns:
        List of parameter names.
    """
    params = ["G", "tau_b", "eta_s"]

    if breakage == "bell":
        params.append("nu")
    elif breakage == "power_law":
        params.append("m_break")
    elif breakage == "stretch_creation":
        params.append("kappa")

    if stress_type == "fene":
        params.append("L_max")

    return params


def get_tnt_cates_param_names() -> list[str]:
    """Get parameter names for TNTCates.

    Returns:
        List of parameter names: [G_0, tau_rep, tau_break, eta_s].
    """
    return ["G_0", "tau_rep", "tau_break", "eta_s"]


def get_tnt_loop_bridge_param_names() -> list[str]:
    """Get parameter names for TNTLoopBridge.

    Returns:
        List of parameter names: [G, tau_b, tau_a, nu, f_B_eq, eta_s].
    """
    return ["G", "tau_b", "tau_a", "nu", "f_B_eq", "eta_s"]


def get_tnt_multi_species_param_names(n_species: int = 2) -> list[str]:
    """Get parameter names for TNTMultiSpecies.

    Args:
        n_species: Number of bond species.

    Returns:
        List of parameter names: [G_0, tau_b_0, G_1, tau_b_1, ..., eta_s].
    """
    params = []
    for i in range(n_species):
        params.extend([f"G_{i}", f"tau_b_{i}"])
    params.append("eta_s")
    return params


def get_tnt_sticky_rouse_param_names(n_modes: int = 3) -> list[str]:
    """Get parameter names for TNTStickyRouse.

    Args:
        n_modes: Number of Rouse modes.

    Returns:
        List of parameter names: [G_0, tau_R_0, ..., tau_s, eta_s].
    """
    params = []
    for k in range(n_modes):
        params.extend([f"G_{k}", f"tau_R_{k}"])
    params.extend(["tau_s", "eta_s"])
    return params


def get_tnt_param_names(model_class: str, **kwargs: Any) -> list[str]:
    """Unified dispatch for TNT parameter names.

    Args:
        model_class: One of 'single_mode', 'cates', 'loop_bridge',
            'multi_species', 'sticky_rouse'.
        **kwargs: Additional kwargs (breakage, n_species, n_modes, etc.).

    Returns:
        List of parameter names.
    """
    dispatch = {
        "single_mode": lambda: get_tnt_single_mode_param_names(
            breakage=kwargs.get("breakage", "constant"),
            stress_type=kwargs.get("stress_type", "linear"),
        ),
        "cates": get_tnt_cates_param_names,
        "loop_bridge": get_tnt_loop_bridge_param_names,
        "multi_species": lambda: get_tnt_multi_species_param_names(
            n_species=kwargs.get("n_species", 2),
        ),
        "sticky_rouse": lambda: get_tnt_sticky_rouse_param_names(
            n_modes=kwargs.get("n_modes", 3),
        ),
    }

    if model_class not in dispatch:
        raise ValueError(
            f"Unknown model_class '{model_class}'. "
            f"Available: {list(dispatch.keys())}"
        )

    return dispatch[model_class]()


# =============================================================================
# Section 6: Diagnostic Printing
# =============================================================================


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
    print("=" * 55)
    print(f"{'Parameter':>15s}  {'R-hat':>8s}  {'ESS':>8s}  {'Status':>8s}")
    print("-" * 55)

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
    param_names: list[str],
) -> None:
    """Print NLSQ vs Bayesian parameter comparison table.

    Args:
        model: Fitted model with parameters attribute.
        posterior: Dictionary of posterior samples.
        param_names: List of parameter names to compare.
    """
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


def compute_fit_quality(
    y_data: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute fit quality metrics (R-squared, RMSE, NRMSE).

    Args:
        y_data: Observed data array.
        y_pred: Predicted data array.

    Returns:
        Dictionary with 'R2', 'RMSE', and 'NRMSE' keys.
    """
    y_data = np.asarray(y_data).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    rmse = np.sqrt(np.mean((y_data - y_pred) ** 2))

    data_range = np.max(y_data) - np.min(y_data)
    nrmse = rmse / data_range if data_range > 0 else 0.0

    return {"R2": r2, "RMSE": rmse, "NRMSE": nrmse}


# =============================================================================
# Section 7: TNT Physics Functions
# =============================================================================


def compute_steady_state_stretch(
    gamma_dot: np.ndarray | float,
    tau_b: float,
) -> np.ndarray:
    """Compute analytical steady-state stretch for constant breakage.

    For constant breakage: lambda_ss = sqrt((1 + 2*Wi^2 + 1) / 3)
    where Wi = gamma_dot * tau_b.

    Args:
        gamma_dot: Shear rate (1/s).
        tau_b: Bond lifetime (s).

    Returns:
        Steady-state chain stretch ratio.
    """
    gamma_dot = np.asarray(gamma_dot)
    Wi = gamma_dot * tau_b
    # tr(S_ss) = S_xx + S_yy + S_zz = (1 + 2*Wi^2) + 1 + 1 = 3 + 2*Wi^2
    return np.sqrt(1.0 + (2.0 / 3.0) * Wi**2)


def compute_bell_effective_lifetime(
    stretch: np.ndarray | float,
    tau_b: float,
    nu: float,
) -> np.ndarray:
    """Compute effective bond lifetime under Bell force-dependent breakage.

    tau_eff = tau_b * exp(-nu * (lambda - 1))

    Args:
        stretch: Chain stretch ratio (>=1).
        tau_b: Base bond lifetime (s).
        nu: Force sensitivity (dimensionless).

    Returns:
        Effective lifetime array (s).
    """
    stretch = np.asarray(stretch)
    return tau_b * np.exp(-nu * (stretch - 1.0))


def compute_maxwell_moduli(
    omega: np.ndarray,
    G: float,
    tau: float,
    eta_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-mode Maxwell storage and loss moduli.

    G'(omega) = G * (omega*tau)^2 / (1 + (omega*tau)^2)
    G''(omega) = G * (omega*tau) / (1 + (omega*tau)^2) + eta_s*omega

    Args:
        omega: Angular frequency array (rad/s).
        G: Modulus (Pa).
        tau: Relaxation time (s).
        eta_s: Solvent viscosity (Pa.s).

    Returns:
        Tuple of (G_prime, G_double_prime) arrays (Pa).
    """
    omega = np.asarray(omega)
    wt = omega * tau
    wt2 = wt**2

    G_prime = G * wt2 / (1.0 + wt2)
    G_double_prime = G * wt / (1.0 + wt2) + eta_s * omega

    return G_prime, G_double_prime


def compute_multi_mode_moduli(
    omega: np.ndarray,
    G_arr: list[float] | np.ndarray,
    tau_arr: list[float] | np.ndarray,
    eta_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute multi-mode Maxwell storage and loss moduli.

    G'(omega) = sum_k G_k * (omega*tau_k)^2 / (1 + (omega*tau_k)^2)
    G''(omega) = sum_k G_k * (omega*tau_k) / (1 + (omega*tau_k)^2) + eta_s*omega

    Args:
        omega: Angular frequency array (rad/s).
        G_arr: Mode moduli array (Pa).
        tau_arr: Mode relaxation times array (s).
        eta_s: Solvent viscosity (Pa.s).

    Returns:
        Tuple of (G_prime, G_double_prime) arrays (Pa).
    """
    omega = np.asarray(omega)
    G_arr = np.asarray(G_arr)
    tau_arr = np.asarray(tau_arr)

    G_prime = np.zeros_like(omega)
    G_double_prime = eta_s * omega

    for G_k, tau_k in zip(G_arr, tau_arr):
        wt = omega * tau_k
        wt2 = wt**2
        G_prime += G_k * wt2 / (1.0 + wt2)
        G_double_prime += G_k * wt / (1.0 + wt2)

    return G_prime, G_double_prime


def compute_tnt_normal_stress(
    gamma_dot: np.ndarray | float,
    G: float,
    tau_b: float,
) -> np.ndarray:
    """Compute first normal stress difference for constant breakage TNT.

    N1 = 2*G*(tau_b * gamma_dot)^2

    Args:
        gamma_dot: Shear rate (1/s).
        G: Network modulus (Pa).
        tau_b: Bond lifetime (s).

    Returns:
        First normal stress difference N1 (Pa).
    """
    gamma_dot = np.asarray(gamma_dot)
    Wi = gamma_dot * tau_b
    return 2.0 * G * Wi**2


def compute_cates_tau_d(tau_rep: float, tau_break: float) -> float:
    """Compute Cates effective relaxation time.

    In the fast-breaking limit (tau_break << tau_rep):
    tau_d = sqrt(tau_rep * tau_break)

    Args:
        tau_rep: Reptation time (s).
        tau_break: Breaking time (s).

    Returns:
        Effective relaxation time tau_d (s).
    """
    return np.sqrt(tau_rep * tau_break)


# =============================================================================
# Internal Helpers
# =============================================================================


def _get_xlabel(protocol: str) -> str:
    """Get x-axis label for protocol."""
    labels = {
        "flow_curve": "Shear rate [1/s]",
        "startup": "Time [s]",
        "relaxation": "Time [s]",
        "creep": "Time [s]",
        "saos": "Angular frequency [rad/s]",
        "oscillation": "Angular frequency [rad/s]",
        "laos": "Time [s]",
    }
    return labels.get(protocol, "x")


def _get_ylabel(protocol: str) -> str:
    """Get y-axis label for protocol."""
    labels = {
        "flow_curve": "Stress [Pa]",
        "startup": "Stress [Pa]",
        "relaxation": "Relaxation modulus [Pa]",
        "creep": "Strain [-]",
        "saos": "|G*| [Pa]",
        "oscillation": "|G*| [Pa]",
        "laos": "Stress [Pa]",
    }
    return labels.get(protocol, "y")


def _get_n_modes(model: Any) -> int:
    """Get number of modes/species from model."""
    if hasattr(model, "_n_species"):
        return model._n_species
    if hasattr(model, "_n_modes"):
        return model._n_modes
    return 1


def _get_mode_params(model: Any, k: int) -> tuple[float, float]:
    """Get (G_k, tau_k) for mode k from multi-mode model."""
    # TNTMultiSpecies: G_k, tau_b_k
    try:
        G_k = float(model.parameters.get_value(f"G_{k}"))
    except (KeyError, AttributeError):
        G_k = 0.0

    # Try tau_b_k (MultiSpecies) then tau_R_k (StickyRouse)
    try:
        tau_k = float(model.parameters.get_value(f"tau_b_{k}"))
    except (KeyError, AttributeError):
        try:
            tau_k = float(model.parameters.get_value(f"tau_R_{k}"))
        except (KeyError, AttributeError):
            tau_k = 1.0

    return G_k, tau_k
