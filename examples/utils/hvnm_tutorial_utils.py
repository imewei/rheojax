"""Shared utilities for HVNM (Hybrid Vitrimer Nanocomposite Model) tutorial notebooks.

Provides data loaders, protocol schema, diagnostics, posterior predictive plotting,
and result saving for 6 protocols: flow_curve, creep, stress_relaxation, startup, SAOS, LAOS.

Data Sources (all local):
- ec_shear_viscosity_07-00.csv: Ethyl cellulose 7% solution flow curve
- creep_ps190_data.csv: Polystyrene creep at 190 C
- stressrelaxation_liquidfoam_data.csv: Liquid foam relaxation
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 startup + LAOS
- epstein.csv: Epstein et al. metal-organic coordination network SAOS
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# FAST_MODE Configuration
# =============================================================================
# True  -> CI / quick demo (~1-2 min per notebook, 1 chain, few samples)
# False -> publication-quality (~10-30 min, 4 chains, more samples)

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
# Protocol Schema
# =============================================================================


@dataclass
class ProtocolData:
    """Standardized data container for a single rheological protocol.

    Attributes:
        protocol: One of 'flow_curve', 'creep', 'relaxation', 'startup',
                  'oscillation', 'laos'.
        x: Independent variable array (gamma_dot, time, omega, or time).
        y: Measured response array (stress, compliance, modulus, etc.).
        y_label: Label for measured response (e.g., 'Stress [Pa]').
        x_label: Label for independent variable.
        metadata: Experimental metadata (temperature, geometry, etc.).
        mask: Boolean mask for valid data points (True = include).
        y2: Optional second response (e.g., G'' for SAOS, strain for LAOS).
        y2_label: Label for second response.
        protocol_kwargs: Extra kwargs needed by model (gamma_dot, sigma_applied, etc.).
    """

    protocol: str
    x: np.ndarray
    y: np.ndarray
    y_label: str = "Response"
    x_label: str = "x"
    metadata: dict[str, Any] = field(default_factory=dict)
    mask: np.ndarray | None = None
    y2: np.ndarray | None = None
    y2_label: str | None = None
    protocol_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mask is None:
            self.mask = np.ones(len(self.x), dtype=bool)

    @property
    def x_masked(self) -> np.ndarray:
        return self.x[self.mask]

    @property
    def y_masked(self) -> np.ndarray:
        return self.y[self.mask]

    @property
    def n_points(self) -> int:
        return int(np.sum(self.mask))

    def summary(self) -> str:
        lines = [
            f"Protocol: {self.protocol}",
            f"  Points: {self.n_points} / {len(self.x)}",
            f"  x range: [{self.x_masked.min():.4g}, {self.x_masked.max():.4g}]",
            f"  y range: [{self.y_masked.min():.4g}, {self.y_masked.max():.4g}]",
        ]
        for k, v in self.metadata.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# =============================================================================
# Data Loaders
# =============================================================================

_DATA_DIR = Path(__file__).parent / ".." / "data"
_FIG_DIR = Path(__file__).parent / ".." / "figures" / "hvnm"


def load_ec_flow_curve(
    concentration: str = "07-00",
) -> ProtocolData:
    """Load ethyl cellulose solution flow curve.

    European CSV format (semicolons, comma decimals).

    Args:
        concentration: EC concentration string (e.g., "07-00" for 7 wt%).

    Returns:
        ProtocolData with shear_rate (1/s) and stress (Pa).
    """
    fpath = _DATA_DIR / "flow" / "solutions" / f"ec_shear_viscosity_{concentration}.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"EC data not found at: {fpath.resolve()}")

    with open(fpath, encoding="latin-1") as f:
        lines = f.readlines()

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
    gamma_dot, stress, viscosity = gamma_dot[idx], stress[idx], viscosity[idx]

    return ProtocolData(
        protocol="flow_curve",
        x=gamma_dot,
        y=stress,
        x_label=r"$\dot{\gamma}$ [1/s]",
        y_label=r"$\sigma$ [Pa]",
        y2=viscosity,
        y2_label=r"$\eta$ [Pa$\cdot$s]",
        metadata={
            "material": f"Ethyl Cellulose {concentration.replace('-', '.')} wt%",
            "source": "pyRheo demos",
            "units_x": "1/s",
            "units_y": "Pa",
        },
    )


def load_epstein_saos() -> ProtocolData:
    """Load SAOS data from Epstein metal-organic coordination network.

    Returns:
        ProtocolData with omega (rad/s), G_star (Pa), and G'/G'' in y2-style.
    """
    fpath = _DATA_DIR / "oscillation" / "metal_networks" / "epstein.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Epstein data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    omega = df.iloc[:, 0].values.astype(float)
    G_prime = df.iloc[:, 1].values.astype(float)
    G_double_prime = df.iloc[:, 2].values.astype(float)

    idx = np.argsort(omega)
    omega = omega[idx]
    G_prime = G_prime[idx]
    G_double_prime = G_double_prime[idx]
    G_star = np.sqrt(G_prime**2 + G_double_prime**2)

    return ProtocolData(
        protocol="oscillation",
        x=omega,
        y=G_star,
        x_label=r"$\omega$ [rad/s]",
        y_label=r"$|G^*|$ [Pa]",
        y2=np.column_stack([G_prime, G_double_prime]),
        y2_label="G', G'' [Pa]",
        metadata={
            "material": "Metal-organic coordination network (Epstein et al.)",
            "source": "Epstein et al. (JACS 2019)",
            "units_x": "rad/s",
            "units_y": "Pa",
            "note": "Exchangeable metal-ligand bonds, structurally analogous to vitrimer BER",
        },
    )


def load_foam_relaxation(
    max_points: int | None = None,
) -> ProtocolData:
    """Load liquid foam stress relaxation data.

    Args:
        max_points: If set, subsample to this many points.

    Returns:
        ProtocolData with time (s) and G(t) (Pa).
    """
    fpath = _DATA_DIR / "relaxation" / "foams" / "stressrelaxation_liquidfoam_data.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"Foam data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    mask = (G_t > 0) & (time > 0)
    time, G_t = time[mask], G_t[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, G_t = time[indices], G_t[indices]

    return ProtocolData(
        protocol="relaxation",
        x=time,
        y=G_t,
        x_label="Time [s]",
        y_label="G(t) [Pa]",
        metadata={
            "material": "Liquid foam",
            "source": "pyRheo demos",
            "units_x": "s",
            "units_y": "Pa",
        },
    )


def load_ps_creep(
    temperature: int = 190,
    max_points: int | None = None,
) -> ProtocolData:
    """Load polystyrene creep compliance data.

    Args:
        temperature: Temperature in Celsius (130, 145, 160, 175, 190).
        max_points: If set, subsample to this many points.

    Returns:
        ProtocolData with time (s) and J(t) (1/Pa).
    """
    fpath = _DATA_DIR / "creep" / "polymers" / f"creep_ps{temperature}_data.csv"
    if not fpath.exists():
        raise FileNotFoundError(f"PS creep data not found at: {fpath.resolve()}")

    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    J = df.iloc[:, 1].values.astype(float)

    mask = (time > 0) & (J > 0)
    time, J = time[mask], J[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, J = time[indices], J[indices]

    return ProtocolData(
        protocol="creep",
        x=time,
        y=J,
        x_label="Time [s]",
        y_label="J(t) [1/Pa]",
        metadata={
            "material": f"Polystyrene at {temperature} C",
            "source": "pyRheo demos",
            "units_x": "s",
            "units_y": "1/Pa",
            "temperature_C": temperature,
        },
        protocol_kwargs={"sigma_applied": 1.0},
    )


def load_pnas_startup(
    gamma_dot: float = 1.0,
    max_points: int | None = None,
) -> ProtocolData:
    """Load startup shear data from PNAS Digital Rheometer Twin dataset.

    Args:
        gamma_dot: Shear rate. Available: 0.056, 0.32, 1, 56.2, 100.
        max_points: If set, subsample to this many points.

    Returns:
        ProtocolData with time (s) and stress (Pa).
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

    fpath = _DATA_DIR / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
    if not fpath.exists():
        raise FileNotFoundError(f"PNAS data not found at: {fpath.resolve()}")

    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    data = df.iloc[3:, [0, 1]].dropna().astype(float)
    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, stress = time[indices], stress[indices]

    return ProtocolData(
        protocol="startup",
        x=time,
        y=stress,
        x_label="Time [s]",
        y_label=r"$\sigma$ [Pa]",
        metadata={
            "material": "PNAS Digital Rheometer Twin sample",
            "source": "PNAS 2022 Digital Rheometer Twin",
            "units_x": "s",
            "units_y": "Pa",
            "gamma_dot": closest,
        },
        protocol_kwargs={"gamma_dot": closest},
    )


def load_pnas_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
    max_points: int | None = None,
) -> ProtocolData:
    """Load LAOS data from PNAS Digital Rheometer Twin dataset.

    Args:
        omega: Angular frequency (1, 3, or 5 rad/s).
        strain_amplitude_index: Index for strain amplitude (0-11, increasing).
        max_points: If set, subsample to this many points.

    Returns:
        ProtocolData with time (s), stress (Pa), and strain in y2.
    """
    omega_sheets = {1.0: "LAOS_w1", 3.0: "LAOS_w3", 5.0: "LAOS_w5"}
    if omega not in omega_sheets:
        raise ValueError(f"omega must be 1, 3, or 5, got {omega}")
    if strain_amplitude_index not in range(12):
        raise ValueError(f"strain_amplitude_index must be 0-11, got {strain_amplitude_index}")

    sheet = omega_sheets[omega]
    fpath = _DATA_DIR / "ikh" / "PNAS_DigitalRheometerTwin_Dataset.xlsx"
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

    # Estimate gamma_0 from strain amplitude
    gamma_0 = float(np.max(np.abs(strain)))

    if max_points is not None and len(time) > max_points:
        indices = np.linspace(0, len(time) - 1, max_points, dtype=int)
        time, strain, stress = time[indices], strain[indices], stress[indices]

    return ProtocolData(
        protocol="laos",
        x=time,
        y=stress,
        x_label="Time [s]",
        y_label=r"$\sigma$ [Pa]",
        y2=strain,
        y2_label=r"$\gamma$",
        metadata={
            "material": "PNAS Digital Rheometer Twin sample",
            "source": "PNAS 2022 Digital Rheometer Twin",
            "units_x": "s",
            "units_y": "Pa",
            "omega": omega,
            "gamma_0": gamma_0,
            "strain_amplitude_index": strain_amplitude_index,
        },
        protocol_kwargs={"gamma_0": gamma_0, "omega": omega},
    )


def load_multi_technique() -> dict[str, ProtocolData]:
    """Load multi-technique data from TRIOS format file.

    Returns amplitude sweep, frequency sweep, and flow ramp as dict.

    Returns:
        Dict mapping protocol name to ProtocolData.
    """
    fpath = _DATA_DIR / "multi_technique" / "multi_technique.txt"
    if not fpath.exists():
        raise FileNotFoundError(f"Multi-technique data not found at: {fpath.resolve()}")

    with open(fpath) as f:
        content = f.read()

    # Parse step sections
    sections = content.split("[step]")
    result = {}

    for section in sections[1:]:
        lines = section.strip().split("\n")
        step_name = lines[0].strip()
        header_line = lines[1].strip()
        units_line = lines[2].strip()
        headers = header_line.split("\t")
        data_lines = lines[3:]

        rows = []
        for line in data_lines:
            line = line.strip()
            if not line or line.startswith("["):
                break
            vals = line.split("\t")
            try:
                row = [float(v) for v in vals]
                rows.append(row)
            except ValueError:
                continue

        if not rows:
            continue

        arr = np.array(rows)

        if "Amplitude" in step_name:
            # Amplitude sweep: find oscillation strain and moduli
            strain_idx = headers.index("Oscillation strain") if "Oscillation strain" in headers else 16
            Gp_idx = headers.index("Storage modulus") if "Storage modulus" in headers else 19
            Gdp_idx = headers.index("Loss modulus") if "Loss modulus" in headers else 18
            strain_pct = arr[:, strain_idx]
            G_prime = arr[:, Gp_idx] * 1e6  # MPa -> Pa
            G_double_prime = arr[:, Gdp_idx] * 1e6

            result["amplitude_sweep"] = ProtocolData(
                protocol="amplitude_sweep",
                x=strain_pct,
                y=G_prime,
                x_label="Strain [%]",
                y_label="G' [Pa]",
                y2=G_double_prime,
                y2_label="G'' [Pa]",
                metadata={"step": step_name, "source": "TRIOS multi-technique"},
            )

        elif "Frequency" in step_name:
            omega_idx = 0
            Gp_idx = headers.index("Storage modulus") if "Storage modulus" in headers else 19
            Gdp_idx = headers.index("Loss modulus") if "Loss modulus" in headers else 18
            omega = arr[:, omega_idx]
            G_prime = arr[:, Gp_idx] * 1e6  # MPa -> Pa
            G_double_prime = arr[:, Gdp_idx] * 1e6
            G_star = np.sqrt(G_prime**2 + G_double_prime**2)

            idx = np.argsort(omega)
            result["frequency_sweep"] = ProtocolData(
                protocol="oscillation",
                x=omega[idx],
                y=G_star[idx],
                x_label=r"$\omega$ [rad/s]",
                y_label=r"$|G^*|$ [Pa]",
                y2=np.column_stack([G_prime[idx], G_double_prime[idx]]),
                y2_label="G', G'' [Pa]",
                metadata={"step": step_name, "source": "TRIOS multi-technique"},
            )

        elif "Flow" in step_name:
            # Flow ramp: shear rate, viscosity, stress
            sr_idx = headers.index("Shear rate") if "Shear rate" in headers else 21
            stress_idx = headers.index("Stress") if "Stress" in headers else 12
            visc_idx = headers.index("Viscosity") if "Viscosity" in headers else 24
            gamma_dot = arr[:, sr_idx]
            stress = arr[:, stress_idx] * 1e6  # MPa -> Pa
            viscosity = arr[:, visc_idx]

            mask = gamma_dot > 0
            gamma_dot = gamma_dot[mask]
            stress = np.abs(stress[mask])
            viscosity = viscosity[mask]
            idx = np.argsort(gamma_dot)

            result["flow_ramp"] = ProtocolData(
                protocol="flow_curve",
                x=gamma_dot[idx],
                y=stress[idx],
                x_label=r"$\dot{\gamma}$ [1/s]",
                y_label=r"$\sigma$ [Pa]",
                y2=viscosity[idx],
                y2_label=r"$\eta$ [Pa$\cdot$s]",
                metadata={"step": step_name, "source": "TRIOS multi-technique"},
            )

    return result


# =============================================================================
# HVNM Parameter Helpers
# =============================================================================


# Default parameter values for a "baseline" HVNM that can fit simple polymer data.
# In tutorials, only a subset of these are free; the rest are fixed.
HVNM_DEFAULT_PARAMS = {
    "G_P": 5000.0,
    "G_E": 3000.0,
    "nu_0": 1e10,
    "E_a": 80e3,
    "V_act": 1e-5,
    "T": 300.0,
    "G_D": 1000.0,
    "k_d_D": 1.0,
    "beta_I": 3.0,
    "nu_0_int": 1e10,
    "E_a_int": 90e3,
    "V_act_int": 5e-6,
    "phi": 0.05,
    "R_NP": 20e-9,
    "delta_m": 10e-9,
}


# Which parameters are identifiable from each protocol
FITTABLE_PARAMS = {
    "flow_curve": ["G_D", "k_d_D"],
    "relaxation": ["G_P", "G_E", "G_D", "nu_0", "k_d_D"],
    "creep": ["G_P", "G_E", "G_D", "nu_0", "k_d_D"],
    "oscillation": ["G_P", "G_E", "G_D", "nu_0", "k_d_D", "phi", "beta_I"],
    "startup": ["G_P", "G_E", "G_D", "nu_0", "k_d_D", "V_act"],
    "laos": ["G_P", "G_E", "G_D", "nu_0", "k_d_D", "V_act"],
}


def configure_hvnm_for_fit(
    model: Any,
    protocol: str,
    overrides: dict[str, float] | None = None,
) -> list[str]:
    """Configure HVNM parameter values and return list of fittable param names.

    Sets all parameters to defaults, then applies overrides.
    Marks non-fittable params as fixed (bounds collapsed).

    Args:
        model: HVNMLocal instance.
        protocol: Protocol name for determining fittable params.
        overrides: Optional parameter value overrides.

    Returns:
        List of fittable parameter names.
    """
    params = {**HVNM_DEFAULT_PARAMS}
    if overrides:
        params.update(overrides)

    for name, value in params.items():
        if name in model.parameters.keys():
            model.parameters.set_value(name, value)

    fittable = FITTABLE_PARAMS.get(protocol, list(HVNM_DEFAULT_PARAMS.keys()))
    return fittable


def get_nlsq_values(model: Any, param_names: list[str]) -> dict[str, float]:
    """Extract current parameter values from a fitted model."""
    return {p: float(model.parameters.get_value(p)) for p in param_names}


# =============================================================================
# Convergence Diagnostics
# =============================================================================


def print_convergence(result: Any, param_names: list[str]) -> bool:
    """Print convergence diagnostics from a BayesianResult.

    Returns:
        True if all R-hat < 1.05 and ESS > 100.
    """
    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 50)
    print(f"{'Parameter':>12s}  {'R-hat':>8s}  {'ESS':>8s}")
    print("-" * 50)

    all_ok = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))
        flag = ""
        if r_hat > 1.05 or ess < 100:
            flag = " *"
            all_ok = False
        print(f"{p:>12s}  {r_hat:8.4f}  {ess:8.0f}{flag}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")
    print(f"Convergence: {'PASSED' if all_ok else 'CHECK REQUIRED'}")
    return all_ok


def print_parameter_table(
    param_names: list[str],
    nlsq_vals: dict[str, float],
    posterior: dict[str, Any],
) -> None:
    """Print NLSQ vs Bayesian parameter comparison."""
    header = f"{'Param':>12s}  {'NLSQ':>12s}  {'Bayes(med)':>12s}  {'95% CI':>24s}"
    print("Parameter Comparison")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for p in param_names:
        samples = np.array(posterior[p])
        med = np.median(samples)
        lo, hi = np.percentile(samples, [2.5, 97.5])
        print(f"{p:>12s}  {nlsq_vals[p]:12.4g}  {med:12.4g}  [{lo:.4g}, {hi:.4g}]")


# =============================================================================
# Plotting Helpers
# =============================================================================


def setup_style():
    """Set up consistent matplotlib style for all notebooks."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
        "figure.figsize": (10, 5),
    })


def plot_raw_data(data: ProtocolData, ax: plt.Axes | None = None) -> plt.Figure:
    """Plot raw data from a ProtocolData object."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    protocol = data.protocol

    if protocol in ("flow_curve", "oscillation", "relaxation"):
        ax.loglog(data.x_masked, data.y_masked, "o", ms=4, alpha=0.8)
    elif protocol == "creep":
        ax.semilogx(data.x_masked, data.y_masked, "o", ms=4, alpha=0.8)
    else:
        ax.plot(data.x_masked, data.y_masked, "-", lw=1, alpha=0.8)

    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.y_label)
    title = data.metadata.get("material", data.protocol)
    ax.set_title(f"{data.protocol}: {title}")
    ax.grid(True, alpha=0.3, which="both")

    return fig


def plot_fit_comparison(
    data: ProtocolData,
    model: Any,
    param_names: list[str] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot data vs model fit."""
    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = data.x_masked
    y = data.y_masked

    # Model prediction
    y_pred = model.predict(x, test_mode=data.protocol, **data.protocol_kwargs)

    # Left: data vs fit
    logscale = data.protocol in ("flow_curve", "oscillation", "relaxation")
    if logscale:
        ax1.loglog(x, y, "o", ms=5, alpha=0.6, label="Data", color="steelblue")
        ax1.loglog(x, y_pred, "-", lw=2, label="HVNM fit", color="orangered")
    elif data.protocol == "creep":
        ax1.semilogx(x, y, "o", ms=5, alpha=0.6, label="Data", color="steelblue")
        ax1.semilogx(x, y_pred, "-", lw=2, label="HVNM fit", color="orangered")
    else:
        ax1.plot(x, y, "o", ms=3, alpha=0.5, label="Data", color="steelblue")
        ax1.plot(x, y_pred, "-", lw=2, label="HVNM fit", color="orangered")

    ax1.set_xlabel(data.x_label)
    ax1.set_ylabel(data.y_label)
    ax1.set_title(title or f"HVNM Fit: {data.protocol}")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # Right: residuals
    residuals = (y - y_pred) / np.maximum(np.abs(y), 1e-30) * 100
    if logscale:
        ax2.semilogx(x, residuals, "o", ms=4, alpha=0.6, color="seagreen")
    else:
        ax2.plot(x, residuals, "o", ms=4, alpha=0.6, color="seagreen")
    ax2.axhline(0, color="k", ls="--", lw=0.8)
    ax2.set_xlabel(data.x_label)
    ax2.set_ylabel("Relative Residual [%]")
    ax2.set_title("Residual Analysis")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_ppc(
    data: ProtocolData,
    model: Any,
    posterior: dict[str, Any],
    param_names: list[str],
    n_draws: int = 50,
    title: str | None = None,
) -> plt.Figure:
    """Posterior predictive check: overlay posterior draws on data.

    Args:
        data: ProtocolData.
        model: HVNMLocal instance (parameter values will be temporarily modified).
        posterior: Dict of posterior samples.
        param_names: Names of parameters that were sampled.
        n_draws: Number of posterior draws to plot.
        title: Plot title.
    """
    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = data.x_masked

    n_total = len(list(posterior.values())[0])
    draw_indices = np.random.default_rng(42).choice(n_total, size=min(n_draws, n_total), replace=False)

    # Save original values
    orig_vals = {p: model.parameters.get_value(p) for p in param_names}

    ppc_curves = []
    for idx in draw_indices:
        for p in param_names:
            model.parameters.set_value(p, float(np.array(posterior[p])[idx]))
        try:
            y_draw = model.predict(x, test_mode=data.protocol, **data.protocol_kwargs)
            ppc_curves.append(np.asarray(y_draw))
        except Exception:
            continue

    # Restore
    for p, v in orig_vals.items():
        model.parameters.set_value(p, v)

    if ppc_curves:
        ppc_arr = np.array(ppc_curves)
        lo = np.percentile(ppc_arr, 2.5, axis=0)
        hi = np.percentile(ppc_arr, 97.5, axis=0)
        med = np.median(ppc_arr, axis=0)

        logscale = data.protocol in ("flow_curve", "oscillation", "relaxation")
        plot_fn = ax.loglog if logscale else (ax.semilogx if data.protocol == "creep" else ax.plot)

        plot_fn(x, data.y_masked, "o", ms=5, alpha=0.7, color="steelblue", label="Data", zorder=3)
        plot_fn(x, med, "-", lw=2, color="orangered", label="Median", zorder=2)
        ax.fill_between(x, lo, hi, alpha=0.25, color="orangered", label="95% CI")

    ax.set_xlabel(data.x_label)
    ax.set_ylabel(data.y_label)
    ax.set_title(title or f"Posterior Predictive Check: {data.protocol}")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


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


def plot_saos_components(
    data: ProtocolData,
    model: Any,
    title: str = "SAOS Components",
) -> plt.Figure:
    """Plot G' and G'' data vs model for SAOS protocol.

    Assumes data.y2 is a (N, 2) array with [G', G''].
    """
    if data.y2 is None or data.y2.ndim != 2:
        raise ValueError("data.y2 must be (N, 2) array of [G', G'']")

    omega = data.x_masked
    G_prime_data = data.y2[data.mask, 0]
    G_double_prime_data = data.y2[data.mask, 1]

    G_prime_pred, G_double_prime_pred = model.predict_saos(omega)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(omega, G_prime_data, "s", ms=5, alpha=0.7, color="steelblue", label="G' data")
    ax.loglog(omega, G_double_prime_data, "o", ms=5, alpha=0.7, color="coral", label="G'' data")
    ax.loglog(omega, G_prime_pred, "-", lw=2, color="navy", label="G' fit")
    ax.loglog(omega, G_double_prime_pred, "--", lw=2, color="darkred", label="G'' fit")
    ax.set_xlabel(r"$\omega$ [rad/s]")
    ax.set_ylabel("G', G'' [Pa]")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    return fig


# =============================================================================
# Save / Load
# =============================================================================


def get_output_dir(protocol: str) -> str:
    """Return path to output directory for a given protocol.

    Creates directory if it doesn't exist.
    """
    out = Path(__file__).parent / ".." / "outputs" / "hvnm" / protocol
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def get_figures_dir() -> str:
    """Return path to figures directory, creating if needed."""
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    return str(_FIG_DIR)


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
        model: Fitted HVNMLocal model.
        result: BayesianResult (optional).
        param_names: Parameter names.
        extra_meta: Extra metadata for the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    if param_names is None:
        param_names = list(FITTABLE_PARAMS.get("flow_curve", ["G_D", "k_d_D"]))

    nlsq = get_nlsq_values(model, param_names)
    if extra_meta:
        nlsq.update(extra_meta)
    with open(os.path.join(output_dir, "fitted_params_nlsq.json"), "w") as f:
        json.dump(nlsq, f, indent=2)

    if result is None:
        return

    posterior = result.posterior_samples
    np.savez(
        os.path.join(output_dir, "posterior_samples.npz"),
        **{k: np.array(v) for k, v in posterior.items()},
    )

    rows = []
    for p in param_names:
        if p not in posterior:
            continue
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


def save_figure(fig: plt.Figure, name: str, output_dir: str | None = None) -> None:
    """Save a figure to the figures/hvnm/ directory (or custom dir)."""
    if output_dir is None:
        output_dir = get_figures_dir()
    os.makedirs(output_dir, exist_ok=True)
    fpath = os.path.join(output_dir, name)
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {fpath}")
