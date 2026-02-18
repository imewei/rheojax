"""Data loaders for HVM (Hybrid Vitrimer Model) tutorial notebooks.

Provides protocol-specific data loading, unit normalization, and quality
checks for real experimental rheology data used in NLSQ → NUTS inference.

Dataset Registry:
    SAOS:
        - Epstein metal-organic coordination network (crossover visible)
        - Polystyrene oscillation at 130-190°C (temperature series)
    Relaxation:
        - Fish muscle stress relaxation (567 high-density points)
        - Polystyrene relaxation at 130-190°C
    Creep:
        - Polystyrene creep compliance at 130-190°C
    Flow Curve:
        - Oil-water emulsions at φ = 0.69-0.80
        - Ethyl cellulose solutions 2-10 wt%
    Startup:
        - PNAS Digital Rheometer Twin (0.056-100 1/s)
    LAOS:
        - PNAS Digital Rheometer Twin (ω = 1, 3, 5 rad/s)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# Paths
# =============================================================================

_DATA_ROOT = Path(__file__).resolve().parent / ".." / "data"


def _resolve(subpath: str) -> Path:
    """Resolve a data path relative to examples/data/."""
    p = _DATA_ROOT / subpath
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    return p


# =============================================================================
# SAOS Loaders
# =============================================================================


def load_epstein_saos() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SAOS data from Epstein metal-organic coordination network.

    This dataset shows a clear single crossover, making it ideal for fitting
    the HVM model's E-network relaxation time.

    Returns:
        (omega, G_prime, G_double_prime) in (rad/s, Pa, Pa).
    """
    fpath = _resolve("oscillation/metal_networks/epstein.csv")
    df = pd.read_csv(fpath, sep="\t")
    omega = df.iloc[:, 0].values.astype(float)
    G_prime = df.iloc[:, 1].values.astype(float)
    G_double_prime = df.iloc[:, 2].values.astype(float)
    idx = np.argsort(omega)
    return omega[idx], G_prime[idx], G_double_prime[idx]


def load_ps_saos(
    temperature: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load polystyrene oscillation data at a given temperature.

    Args:
        temperature: Temperature in °C. Available: 130, 145, 160, 175, 190.

    Returns:
        (omega, G_prime, G_double_prime) in (rad/s, Pa, Pa).
    """
    valid = [130, 145, 160, 175, 190]
    if temperature not in valid:
        raise ValueError(f"temperature must be in {valid}, got {temperature}")
    fpath = _resolve(f"oscillation/polystyrene/oscillation_ps{temperature}_data.csv")
    df = pd.read_csv(fpath, sep="\t")
    omega = df.iloc[:, 0].values.astype(float)
    G_prime = df.iloc[:, 1].values.astype(float)
    G_double_prime = df.iloc[:, 2].values.astype(float)
    idx = np.argsort(omega)
    return omega[idx], G_prime[idx], G_double_prime[idx]


def load_ps_saos_temperature_series(
    temperatures: list[int] | None = None,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load PS SAOS data at multiple temperatures for Arrhenius analysis.

    Args:
        temperatures: List of temperatures (°C). Default: all five.

    Returns:
        Dict mapping temperature → (omega, G_prime, G_double_prime).
    """
    if temperatures is None:
        temperatures = [130, 145, 160, 175, 190]
    return {T: load_ps_saos(T) for T in temperatures}


# =============================================================================
# Relaxation Loaders
# =============================================================================


def load_fish_muscle_relaxation(
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load fish muscle stress relaxation (567 high-density points).

    This dataset has exceptional temporal resolution, revealing fine features
    in the relaxation modulus that distinguish HVM subnetwork contributions.

    Args:
        max_points: If set, subsample to this many log-spaced points.

    Returns:
        (time, G_t) in (s, Pa).
    """
    fpath = _resolve("relaxation/biological/stressrelaxation_fishmuscle_data.csv")
    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    mask = (time > 0) & (G_t > 0)
    time, G_t = time[mask], G_t[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, G_t = time[indices], G_t[indices]

    return time, G_t


def load_ps_relaxation(
    temperature: int = 160,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load polystyrene stress relaxation data.

    Args:
        temperature: Temperature in °C (130, 145, 160, 175, 190).
        max_points: If set, subsample.

    Returns:
        (time, G_t) in (s, Pa).
    """
    valid = [130, 145, 160, 175, 190]
    if temperature not in valid:
        raise ValueError(f"temperature must be in {valid}, got {temperature}")
    fpath = _resolve(f"relaxation/polymers/stressrelaxation_ps{temperature}_data.csv")
    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    mask = (time > 0) & (G_t > 0)
    time, G_t = time[mask], G_t[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, G_t = time[indices], G_t[indices]

    return time, G_t


def load_foam_relaxation(
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load liquid foam stress relaxation data.

    Returns:
        (time, G_t) in (s, Pa).
    """
    fpath = _resolve("relaxation/foams/stressrelaxation_liquidfoam_data.csv")
    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    G_t = df.iloc[:, 1].values.astype(float)

    mask = G_t > 0
    time, G_t = time[mask], G_t[mask]

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, G_t = time[indices], G_t[indices]

    return time, G_t


# =============================================================================
# Creep Loaders
# =============================================================================


def load_ps_creep(
    temperature: int = 160,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load polystyrene creep compliance data.

    Args:
        temperature: Temperature in °C (130, 145, 160, 175, 190).
        max_points: If set, subsample.

    Returns:
        (time, J) in (s, 1/Pa).
    """
    valid = [130, 145, 160, 175, 190]
    if temperature not in valid:
        raise ValueError(f"temperature must be in {valid}, got {temperature}")
    fpath = _resolve(f"creep/polymers/creep_ps{temperature}_data.csv")
    df = pd.read_csv(fpath, sep="\t")
    time = df.iloc[:, 0].values.astype(float)
    J = df.iloc[:, 1].values.astype(float)

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, J = time[indices], J[indices]

    return time, J


# =============================================================================
# Flow Curve Loaders
# =============================================================================


def load_emulsion_flow_curve(
    phi: float = 0.74,
) -> tuple[np.ndarray, np.ndarray]:
    """Load oil-water emulsion flow curve at given volume fraction.

    Emulsion flow curves show yield-stress-like behavior at high φ,
    analogous to vitrimer plateau stress from the P-network.

    Args:
        phi: Volume fraction. Available: 0.69, 0.70, 0.72, 0.74, 0.76, 0.80.

    Returns:
        (shear_rate, stress) in (1/s, Pa).
    """
    valid = [0.69, 0.70, 0.72, 0.74, 0.76, 0.80]
    closest = min(valid, key=lambda x: abs(x - phi))
    fname = f"{closest:.2f}".rstrip("0").rstrip(".")
    # Handle 0.70 → "0.7" and 0.80 → "0.8" filename format
    fpath = _resolve(f"flow/emulsions/{fname}.csv")

    if not fpath.exists():
        # Try exact two-decimal format
        fpath = _resolve(f"flow/emulsions/{closest:.2f}.csv")

    df = pd.read_csv(fpath)
    gamma_dot = df.iloc[:, 0].values.astype(float)
    stress = df.iloc[:, 1].values.astype(float)

    idx = np.argsort(gamma_dot)
    return gamma_dot[idx], stress[idx]


def load_ec_flow_curve(
    concentration: str = "07-00",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ethyl cellulose solution flow curve.

    European CSV format (semicolons, comma decimals).

    Args:
        concentration: EC concentration string (e.g., "07-00" for 7 wt%).

    Returns:
        (shear_rate, stress, viscosity) in (1/s, Pa, Pa·s).
    """
    fpath = _resolve(f"flow/solutions/ec_shear_viscosity_{concentration}.csv")

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
    return gamma_dot[idx], stress[idx], viscosity[idx]


# =============================================================================
# Startup Loaders
# =============================================================================


def load_pnas_startup(
    gamma_dot: float = 1.0,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load startup shear data from PNAS Digital Rheometer Twin dataset.

    Source: PNAS 2022 Digital Rheometer Twin.

    Args:
        gamma_dot: Shear rate. Available: 0.056, 0.32, 1.0, 56.2, 100.0.
        max_points: If set, subsample.

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

    fpath = _resolve("ikh/PNAS_DigitalRheometerTwin_Dataset.xlsx")
    df = pd.read_excel(fpath, sheet_name=sheet, header=None)
    data = df.iloc[3:, [0, 1]].dropna().astype(float)
    time = data.iloc[:, 0].values
    stress = data.iloc[:, 1].values

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, stress = time[indices], stress[indices]

    return time, stress


# =============================================================================
# LAOS Loaders
# =============================================================================


def load_pnas_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
    max_points: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LAOS data from PNAS Digital Rheometer Twin dataset.

    Args:
        omega: Angular frequency (1.0, 3.0, or 5.0 rad/s).
        strain_amplitude_index: Strain amplitude index (0-11, increasing).
        max_points: If set, subsample.

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
    fpath = _resolve("ikh/PNAS_DigitalRheometerTwin_Dataset.xlsx")
    df = pd.read_excel(fpath, sheet_name=sheet, header=None)

    col_t = strain_amplitude_index * 4
    col_strain = strain_amplitude_index * 4 + 1
    col_stress = strain_amplitude_index * 4 + 2

    data = df.iloc[3:, [col_t, col_strain, col_stress]].dropna().astype(float)
    time = data.iloc[:, 0].values
    strain = data.iloc[:, 1].values
    stress = data.iloc[:, 2].values

    if max_points is not None and len(time) > max_points:
        indices = np.unique(
            np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
        )
        time, strain, stress = time[indices], strain[indices], stress[indices]

    return time, strain, stress


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_REGISTRY: dict[str, dict] = {
    "epstein_saos": {
        "loader": "load_epstein_saos",
        "protocol": "oscillation",
        "material": "Metal-organic coordination network",
        "source": "Epstein et al.",
        "description": "Single crossover visible, ~20 points",
    },
    "ps_saos": {
        "loader": "load_ps_saos",
        "protocol": "oscillation",
        "material": "Polystyrene",
        "source": "pyRheo demos",
        "description": "Temperature series 130-190°C, ~23 points each",
    },
    "fish_muscle_relaxation": {
        "loader": "load_fish_muscle_relaxation",
        "protocol": "relaxation",
        "material": "Fish muscle tissue",
        "source": "Biological mechanics",
        "description": "567 high-density points, excellent temporal resolution",
    },
    "ps_relaxation": {
        "loader": "load_ps_relaxation",
        "protocol": "relaxation",
        "material": "Polystyrene",
        "source": "pyRheo demos",
        "description": "Temperature series 130-190°C",
    },
    "ps_creep": {
        "loader": "load_ps_creep",
        "protocol": "creep",
        "material": "Polystyrene",
        "source": "pyRheo demos",
        "description": "Compliance J(t), temperature series 130-190°C",
    },
    "emulsion_flow": {
        "loader": "load_emulsion_flow_curve",
        "protocol": "flow_curve",
        "material": "Oil-water emulsion",
        "source": "Volume fraction series",
        "description": "Yield-stress-like behavior at high φ",
    },
    "ec_flow": {
        "loader": "load_ec_flow_curve",
        "protocol": "flow_curve",
        "material": "Ethyl cellulose solution",
        "source": "Concentration series",
        "description": "Shear-thinning polymer solution",
    },
    "pnas_startup": {
        "loader": "load_pnas_startup",
        "protocol": "startup",
        "material": "Thixotropic yield stress fluid",
        "source": "PNAS 2022 Digital Rheometer Twin",
        "description": "Stress overshoot at 5 shear rates (0.056-100 1/s)",
    },
    "pnas_laos": {
        "loader": "load_pnas_laos",
        "protocol": "laos",
        "material": "Thixotropic yield stress fluid",
        "source": "PNAS 2022 Digital Rheometer Twin",
        "description": "3 frequencies × 12 strain amplitudes",
    },
}


def list_datasets(protocol: str | None = None) -> list[str]:
    """List available datasets, optionally filtered by protocol.

    Args:
        protocol: Filter by protocol name (oscillation, relaxation, etc.).

    Returns:
        List of dataset keys.
    """
    if protocol is None:
        return list(DATASET_REGISTRY.keys())
    return [k for k, v in DATASET_REGISTRY.items() if v["protocol"] == protocol]


def print_registry() -> None:
    """Print a formatted summary of all available datasets."""
    print("HVM Tutorial Dataset Registry")
    print("=" * 70)
    for key, info in DATASET_REGISTRY.items():
        print(f"\n  {key}")
        print(f"    Protocol:    {info['protocol']}")
        print(f"    Material:    {info['material']}")
        print(f"    Source:      {info['source']}")
        print(f"    Description: {info['description']}")


# =============================================================================
# Data Quality Checks
# =============================================================================


def check_data_quality(
    x: np.ndarray,
    y: np.ndarray,
    name: str = "data",
    require_positive: bool = True,
    require_monotonic_x: bool = False,
) -> dict[str, bool | int | float]:
    """Run basic quality checks on loaded data.

    Args:
        x: Independent variable.
        y: Dependent variable.
        name: Dataset name for messages.
        require_positive: Check that y > 0.
        require_monotonic_x: Check that x is monotonically increasing.

    Returns:
        Dict with check results.
    """
    report = {
        "name": name,
        "n_points": len(x),
        "x_range": (float(x.min()), float(x.max())),
        "y_range": (float(y.min()), float(y.max())),
        "has_nan": bool(np.any(np.isnan(x)) or np.any(np.isnan(y))),
        "has_inf": bool(np.any(np.isinf(x)) or np.any(np.isinf(y))),
        "all_positive_y": bool(np.all(y > 0)),
        "monotonic_x": bool(np.all(np.diff(x) > 0)),
    }

    issues = []
    if report["has_nan"]:
        issues.append("Contains NaN values")
    if report["has_inf"]:
        issues.append("Contains Inf values")
    if require_positive and not report["all_positive_y"]:
        issues.append("y contains non-positive values")
    if require_monotonic_x and not report["monotonic_x"]:
        issues.append("x is not monotonically increasing")

    report["issues"] = issues
    report["passed"] = len(issues) == 0

    print(f"Data QC: {name}")
    print(f"  Points: {report['n_points']}")
    print(f"  x range: [{report['x_range'][0]:.4g}, {report['x_range'][1]:.4g}]")
    print(f"  y range: [{report['y_range'][0]:.4g}, {report['y_range'][1]:.4g}]")
    if issues:
        for issue in issues:
            print(f"  WARNING: {issue}")
    else:
        print("  Status: PASSED")

    return report
