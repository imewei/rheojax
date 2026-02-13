"""Shared utilities for rheological protocol validation notebooks.

Provides consistent data discovery, schema validation, derived quantity computation,
plotting functions, and validation report generation across all protocol validation notebooks.

Protocols supported:
- Flow Curve: σ vs γ̇
- Creep: J(t) = γ(t)/σ₀
- Stress Relaxation: G(t) = σ(t)/γ₀
- Startup Shear: σ(t) at constant γ̇
- SAOS: G'(ω), G''(ω)
- LAOS: time-domain strain/stress, Fourier harmonics, Lissajous curves

References:
- Ferry, J.D. "Viscoelastic Properties of Polymers" (1980)
- Macosko, C.W. "Rheology: Principles, Measurements, and Applications" (1994)
- Ewoldt et al. "New measures for characterizing nonlinear viscoelasticity" (2008) J. Rheol.
- Tschoegl, N.W. "Phenomenological Theory of Linear Viscoelastic Behavior" (1989)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# Type Definitions
# =============================================================================

Protocol = Literal[
    "flow_curve",
    "creep",
    "stress_relaxation",
    "startup_shear",
    "saos",
    "laos",
]


@dataclass
class ValidationResult:
    """Container for validation check results."""

    check_name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidation:
    """Container for all validation results for a dataset."""

    file_path: str
    protocol: Protocol
    results: list[ValidationResult] = field(default_factory=list)
    derived_quantities: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Return True if all checks passed."""
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        """Count of passed checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Count of failed checks."""
        return sum(1 for r in self.results if not r.passed)


# =============================================================================
# File Discovery
# =============================================================================


def discover_files_by_protocol(
    data_dir: Path | str,
    protocol: Protocol,
) -> list[Path]:
    """Discover data files for a given protocol.

    Args:
        data_dir: Root data directory (e.g., examples/data).
        protocol: One of the supported protocols.

    Returns:
        List of file paths matching the protocol.
    """
    data_dir = Path(data_dir)
    files: list[Path] = []

    # Protocol to directory mapping
    protocol_dirs = {
        "flow_curve": ["flow"],
        "creep": ["creep"],
        "stress_relaxation": ["relaxation"],
        "startup_shear": ["ikh"],  # IKH Excel files have startup data
        "saos": ["oscillation"],
        "laos": ["laos", "ikh"],  # Both directories have LAOS data
    }

    # File patterns by protocol
    extensions = {
        "flow_curve": ["*.csv"],
        "creep": ["*.csv", "*.txt"],
        "stress_relaxation": ["*.csv"],
        "startup_shear": ["*.xlsx"],
        "saos": ["*.csv", "*.txt"],
        "laos": ["*.txt", "*.xlsx"],
    }

    dirs = protocol_dirs.get(protocol, [])
    patterns = extensions.get(protocol, ["*.csv"])

    for dir_name in dirs:
        search_dir = data_dir / dir_name
        if search_dir.exists():
            for pattern in patterns:
                files.extend(search_dir.rglob(pattern))

    # Sort by name for reproducibility
    return sorted(files, key=lambda p: p.name)


def get_data_dir() -> Path:
    """Get the default data directory relative to this module."""
    return Path(__file__).parent.parent.parent / "data"


def get_output_dir(protocol: Protocol) -> Path:
    """Get output directory for a protocol."""
    return Path(__file__).parent.parent / "outputs" / protocol


# =============================================================================
# Schema Validation
# =============================================================================


def validate_schema(
    df: pd.DataFrame,
    required_cols: list[str],
    dtypes: dict[str, type] | None = None,
) -> ValidationResult:
    """Validate DataFrame has required columns and types.

    Args:
        df: DataFrame to validate.
        required_cols: List of required column names.
        dtypes: Optional dict mapping column names to expected types.

    Returns:
        ValidationResult with pass/fail status.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return ValidationResult(
            check_name="schema_validation",
            passed=False,
            message=f"Missing required columns: {missing}",
            details={"missing_columns": missing, "available_columns": list(df.columns)},
        )

    # Check dtypes if provided
    if dtypes:
        type_errors = []
        for col, expected_type in dtypes.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if not np.issubdtype(actual_type, expected_type):
                    type_errors.append(f"{col}: expected {expected_type}, got {actual_type}")
        if type_errors:
            return ValidationResult(
                check_name="schema_validation",
                passed=False,
                message=f"Type mismatches: {type_errors}",
                details={"type_errors": type_errors},
            )

    return ValidationResult(
        check_name="schema_validation",
        passed=True,
        message=f"All {len(required_cols)} required columns present",
        details={"columns": required_cols},
    )


# =============================================================================
# Unit Normalization
# =============================================================================


def normalize_units(
    df: pd.DataFrame,
    unit_map: dict[str, tuple[str, float]],
) -> pd.DataFrame:
    """Normalize columns to SI units.

    Args:
        df: DataFrame with data.
        unit_map: Dict mapping column names to (new_name, conversion_factor).
            E.g., {"shear_rate (1/s)": ("shear_rate", 1.0), "stress (mPa)": ("stress", 0.001)}

    Returns:
        DataFrame with normalized units and renamed columns.
    """
    df = df.copy()
    for old_name, (new_name, factor) in unit_map.items():
        if old_name in df.columns:
            df[new_name] = df[old_name] * factor
            if old_name != new_name:
                df = df.drop(columns=[old_name])
    return df


# =============================================================================
# Value Checks
# =============================================================================


def check_finite(arr: np.ndarray, name: str = "array") -> ValidationResult:
    """Check that array contains no NaN or Inf values.

    Args:
        arr: NumPy array to check.
        name: Name for error messages.

    Returns:
        ValidationResult with pass/fail status.
    """
    arr = np.asarray(arr)
    n_nan = np.sum(np.isnan(arr))
    n_inf = np.sum(np.isinf(arr))

    if n_nan > 0 or n_inf > 0:
        return ValidationResult(
            check_name=f"finite_values_{name}",
            passed=False,
            message=f"{name}: {n_nan} NaN, {n_inf} Inf values",
            details={"n_nan": int(n_nan), "n_inf": int(n_inf), "total": len(arr)},
        )

    return ValidationResult(
        check_name=f"finite_values_{name}",
        passed=True,
        message=f"{name}: all {len(arr)} values finite",
        details={"total": len(arr)},
    )


def check_positive(
    arr: np.ndarray,
    name: str = "array",
    strict: bool = True,
) -> ValidationResult:
    """Check that array contains only positive values.

    Args:
        arr: NumPy array to check.
        name: Name for error messages.
        strict: If True, require > 0; if False, require >= 0.

    Returns:
        ValidationResult with pass/fail status.
    """
    arr = np.asarray(arr)
    if strict:
        n_bad = np.sum(arr <= 0)
        condition = "positive (> 0)"
    else:
        n_bad = np.sum(arr < 0)
        condition = "non-negative (>= 0)"

    if n_bad > 0:
        return ValidationResult(
            check_name=f"positive_{name}",
            passed=False,
            message=f"{name}: {n_bad}/{len(arr)} values not {condition}",
            details={
                "n_violations": int(n_bad),
                "total": len(arr),
                "min_value": float(np.nanmin(arr)),
            },
        )

    return ValidationResult(
        check_name=f"positive_{name}",
        passed=True,
        message=f"{name}: all {len(arr)} values {condition}",
        details={"min_value": float(np.nanmin(arr)), "total": len(arr)},
    )


def check_monotonic(
    arr: np.ndarray,
    name: str = "array",
    increasing: bool = True,
    strict: bool = True,
) -> ValidationResult:
    """Check that array is monotonically increasing or decreasing.

    Args:
        arr: NumPy array to check.
        name: Name for error messages.
        increasing: If True, check increasing; if False, check decreasing.
        strict: If True, require strictly monotonic (no equal values).

    Returns:
        ValidationResult with pass/fail status.
    """
    arr = np.asarray(arr)
    diff = np.diff(arr)

    if increasing:
        if strict:
            n_bad = np.sum(diff <= 0)
            condition = "strictly increasing"
        else:
            n_bad = np.sum(diff < 0)
            condition = "non-decreasing"
    else:
        if strict:
            n_bad = np.sum(diff >= 0)
            condition = "strictly decreasing"
        else:
            n_bad = np.sum(diff > 0)
            condition = "non-increasing"

    if n_bad > 0:
        return ValidationResult(
            check_name=f"monotonic_{name}",
            passed=False,
            message=f"{name}: {n_bad}/{len(diff)} intervals not {condition}",
            details={"n_violations": int(n_bad), "total": len(diff)},
        )

    return ValidationResult(
        check_name=f"monotonic_{name}",
        passed=True,
        message=f"{name}: all {len(diff)} intervals {condition}",
        details={"total": len(diff)},
    )


def check_range(
    arr: np.ndarray,
    name: str = "array",
    min_val: float | None = None,
    max_val: float | None = None,
) -> ValidationResult:
    """Check that array values are within a specified range.

    Args:
        arr: NumPy array to check.
        name: Name for error messages.
        min_val: Minimum allowed value (None to skip).
        max_val: Maximum allowed value (None to skip).

    Returns:
        ValidationResult with pass/fail status.
    """
    arr = np.asarray(arr)
    actual_min = float(np.nanmin(arr))
    actual_max = float(np.nanmax(arr))

    violations = []
    if min_val is not None and actual_min < min_val:
        violations.append(f"min {actual_min:.2e} < {min_val:.2e}")
    if max_val is not None and actual_max > max_val:
        violations.append(f"max {actual_max:.2e} > {max_val:.2e}")

    if violations:
        return ValidationResult(
            check_name=f"range_{name}",
            passed=False,
            message=f"{name}: {', '.join(violations)}",
            details={"actual_min": actual_min, "actual_max": actual_max},
        )

    return ValidationResult(
        check_name=f"range_{name}",
        passed=True,
        message=f"{name}: range [{actual_min:.2e}, {actual_max:.2e}] within bounds",
        details={"actual_min": actual_min, "actual_max": actual_max},
    )


# =============================================================================
# Derived Quantity Computation
# =============================================================================


def compute_flow_derived(
    gamma_dot: np.ndarray,
    stress: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute derived quantities for flow curve data.

    Args:
        gamma_dot: Shear rate array (1/s).
        stress: Shear stress array (Pa).

    Returns:
        Dict with 'eta' (apparent viscosity in Pa.s).
    """
    gamma_dot = np.asarray(gamma_dot)
    stress = np.asarray(stress)

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(gamma_dot > 0, stress / gamma_dot, np.nan)

    return {"eta": eta}


def compute_creep_derived(
    time: np.ndarray,
    gamma: np.ndarray,
    sigma_0: float,
) -> dict[str, np.ndarray]:
    """Compute derived quantities for creep data.

    Args:
        time: Time array (s).
        gamma: Strain array (dimensionless or %).
        sigma_0: Applied stress (Pa).

    Returns:
        Dict with 'J' (creep compliance in 1/Pa).
    """
    time = np.asarray(time)
    gamma = np.asarray(gamma)

    # Compliance J(t) = γ(t) / σ₀
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(sigma_0 > 0, gamma / sigma_0, np.nan)

    return {"J": J, "log_J": np.log10(np.maximum(J, 1e-20))}


def compute_relaxation_derived(
    time: np.ndarray,
    stress: np.ndarray,
    gamma_0: float,
) -> dict[str, np.ndarray]:
    """Compute derived quantities for stress relaxation data.

    Args:
        time: Time array (s).
        stress: Stress array (Pa).
        gamma_0: Applied strain (dimensionless).

    Returns:
        Dict with 'G' (relaxation modulus in Pa).
    """
    time = np.asarray(time)
    stress = np.asarray(stress)

    # Modulus G(t) = σ(t) / γ₀
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(gamma_0 > 0, stress / gamma_0, np.nan)

    return {"G": G, "log_G": np.log10(np.maximum(G, 1e-20))}


def compute_saos_derived(
    omega: np.ndarray,
    G_prime: np.ndarray,
    G_double_prime: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute derived quantities for SAOS data.

    Args:
        omega: Angular frequency array (rad/s).
        G_prime: Storage modulus array (Pa).
        G_double_prime: Loss modulus array (Pa).

    Returns:
        Dict with 'G_star' (complex modulus) and 'tan_delta' (loss tangent).
    """
    omega = np.asarray(omega)
    G_prime = np.asarray(G_prime)
    G_double_prime = np.asarray(G_double_prime)

    # Complex modulus |G*| = sqrt(G'² + G''²)
    G_star = np.sqrt(G_prime**2 + G_double_prime**2)

    # Loss tangent tan(δ) = G'' / G'
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_delta = np.where(G_prime > 0, G_double_prime / G_prime, np.nan)

    return {"G_star": G_star, "tan_delta": tan_delta}


def compute_laos_harmonics(
    time: np.ndarray,
    stress: np.ndarray,
    omega_0: float,
    n_harmonics: int = 5,
) -> dict[str, Any]:
    """Extract Fourier harmonics from LAOS stress signal.

    Args:
        time: Time array (s).
        stress: Stress signal array (Pa).
        omega_0: Fundamental angular frequency (rad/s).
        n_harmonics: Number of harmonics to extract.

    Returns:
        Dict with:
        - 'harmonics': array of harmonic amplitudes I_n
        - 'I1': fundamental amplitude
        - 'I3_I1': third harmonic ratio (nonlinearity metric)
        - 'omega_recovered': recovered fundamental frequency
    """
    time = np.asarray(time)
    stress = np.asarray(stress)

    # FFT
    n = len(stress)
    dt = np.mean(np.diff(time))
    freqs = np.fft.fftfreq(n, dt)
    fft_vals = np.fft.fft(stress)
    amplitudes = 2 * np.abs(fft_vals) / n

    # Find fundamental frequency
    positive_mask = freqs > 0
    positive_freqs = freqs[positive_mask]
    positive_amps = amplitudes[positive_mask]

    # Expected fundamental frequency in Hz
    f0_expected = omega_0 / (2 * np.pi)

    # Find index closest to expected fundamental
    idx_fund = np.argmin(np.abs(positive_freqs - f0_expected))
    f0_recovered = positive_freqs[idx_fund]
    omega_recovered = f0_recovered * 2 * np.pi

    # Extract harmonic amplitudes
    harmonics = []
    for n_harm in range(1, n_harmonics + 1):
        f_harm = n_harm * f0_recovered
        idx_harm = np.argmin(np.abs(positive_freqs - f_harm))
        harmonics.append(positive_amps[idx_harm])

    harmonics = np.array(harmonics)
    I1 = harmonics[0] if len(harmonics) > 0 else 0.0
    I3 = harmonics[2] if len(harmonics) > 2 else 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        I3_I1 = I3 / I1 if I1 > 0 else np.nan

    return {
        "harmonics": harmonics,
        "I1": float(I1),
        "I3": float(I3) if len(harmonics) > 2 else 0.0,
        "I3_I1": float(I3_I1),
        "omega_recovered": float(omega_recovered),
        "omega_expected": float(omega_0),
        "frequency_error": abs(omega_recovered - omega_0) / omega_0 if omega_0 > 0 else np.nan,
    }


def detect_startup_overshoot(
    time: np.ndarray,
    stress: np.ndarray,
) -> dict[str, Any]:
    """Detect stress overshoot in startup shear data.

    Args:
        time: Time array (s).
        stress: Stress array (Pa).

    Returns:
        Dict with:
        - 'has_overshoot': bool
        - 'sigma_max': peak stress
        - 'sigma_ss': steady-state stress (last 10%)
        - 'overshoot_ratio': σ_max / σ_ss
        - 't_peak': time at peak stress
    """
    time = np.asarray(time)
    stress = np.asarray(stress)

    # Find peak
    idx_max = np.argmax(stress)
    sigma_max = float(stress[idx_max])
    t_peak = float(time[idx_max])

    # Estimate steady-state from last 10% of data
    n_ss = max(1, len(stress) // 10)
    sigma_ss = float(np.mean(stress[-n_ss:]))

    # Overshoot ratio
    with np.errstate(divide="ignore", invalid="ignore"):
        overshoot_ratio = sigma_max / sigma_ss if sigma_ss > 0 else np.nan

    # Has overshoot if peak is not at the end and ratio > 1.05
    has_overshoot = (idx_max < len(stress) - n_ss) and (overshoot_ratio > 1.05)

    return {
        "has_overshoot": has_overshoot,
        "sigma_max": sigma_max,
        "sigma_ss": sigma_ss,
        "overshoot_ratio": float(overshoot_ratio),
        "t_peak": t_peak,
        "idx_peak": int(idx_max),
    }


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_flow_curve(
    gamma_dot: np.ndarray,
    stress: np.ndarray,
    eta: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Flow Curve",
) -> plt.Figure:
    """Create standard flow curve plots.

    Args:
        gamma_dot: Shear rate array (1/s).
        stress: Shear stress array (Pa).
        eta: Apparent viscosity array (Pa.s).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stress vs shear rate (log-log)
    ax = axes[0]
    ax.loglog(gamma_dot, stress, "o-", markersize=4)
    ax.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    ax.set_ylabel(r"Shear stress $\sigma$ (Pa)")
    ax.set_title(f"{title}: Stress vs Shear Rate")
    ax.grid(True, alpha=0.3)

    # Viscosity vs shear rate (log-log)
    ax = axes[1]
    mask = np.isfinite(eta) & (eta > 0)
    ax.loglog(gamma_dot[mask], eta[mask], "s-", markersize=4, color="C1")
    ax.set_xlabel(r"Shear rate $\dot{\gamma}$ (1/s)")
    ax.set_ylabel(r"Viscosity $\eta$ (Pa.s)")
    ax.set_title(f"{title}: Viscosity vs Shear Rate")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_creep(
    time: np.ndarray,
    J: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Creep",
) -> plt.Figure:
    """Create standard creep plots.

    Args:
        time: Time array (s).
        J: Creep compliance array (1/Pa).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # J(t) vs t (log-log)
    ax = axes[0]
    mask = (time > 0) & (J > 0) & np.isfinite(J)
    ax.loglog(time[mask], J[mask], "o-", markersize=4)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Compliance $J(t)$ (1/Pa)")
    ax.set_title(f"{title}: Log-Log")
    ax.grid(True, alpha=0.3)

    # J(t) vs t (linear)
    ax = axes[1]
    ax.plot(time, J, "s-", markersize=4, color="C1")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Compliance $J(t)$ (1/Pa)")
    ax.set_title(f"{title}: Linear")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_relaxation(
    time: np.ndarray,
    G: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "Stress Relaxation",
) -> plt.Figure:
    """Create standard stress relaxation plots.

    Args:
        time: Time array (s).
        G: Relaxation modulus array (Pa).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # G(t) vs t (log-log)
    ax = axes[0]
    mask = (time > 0) & (G > 0) & np.isfinite(G)
    ax.loglog(time[mask], G[mask], "o-", markersize=4)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Modulus $G(t)$ (Pa)")
    ax.set_title(f"{title}: Log-Log")
    ax.grid(True, alpha=0.3)

    # G(t) vs t (semi-log)
    ax = axes[1]
    ax.semilogy(time[mask], G[mask], "s-", markersize=4, color="C1")
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel("Modulus $G(t)$ (Pa)")
    ax.set_title(f"{title}: Semi-Log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_startup(
    time: np.ndarray,
    stress: np.ndarray,
    overshoot_info: dict[str, Any],
    gamma_dot: float | None = None,
    save_path: Path | str | None = None,
    title: str = "Startup Shear",
) -> plt.Figure:
    """Create standard startup shear plots.

    Args:
        time: Time array (s).
        stress: Stress array (Pa).
        overshoot_info: Dict from detect_startup_overshoot().
        gamma_dot: Shear rate (optional, for strain axis).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Stress vs time
    ax = axes[0]
    ax.plot(time, stress, "b-", linewidth=1.5)
    if overshoot_info["has_overshoot"]:
        ax.axhline(overshoot_info["sigma_ss"], color="r", linestyle="--", label="Steady state")
        ax.plot(
            overshoot_info["t_peak"],
            overshoot_info["sigma_max"],
            "ro",
            markersize=8,
            label=f"Peak (ratio={overshoot_info['overshoot_ratio']:.2f})",
        )
        ax.legend()
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"Stress $\sigma$ (Pa)")
    ax.set_title(f"{title}: Stress vs Time")
    ax.grid(True, alpha=0.3)

    # Stress vs strain (if gamma_dot provided)
    ax = axes[1]
    if gamma_dot is not None and gamma_dot > 0:
        strain = gamma_dot * time
        ax.plot(strain, stress, "g-", linewidth=1.5)
        ax.set_xlabel(r"Strain $\gamma$")
    else:
        ax.plot(time, stress, "g-", linewidth=1.5)
        ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"Stress $\sigma$ (Pa)")
    ax.set_title(f"{title}: Stress vs Strain")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_saos(
    omega: np.ndarray,
    G_prime: np.ndarray,
    G_double_prime: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "SAOS",
) -> plt.Figure:
    """Create standard SAOS plots.

    Args:
        omega: Angular frequency array (rad/s).
        G_prime: Storage modulus array (Pa).
        G_double_prime: Loss modulus array (Pa).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # G', G'' vs omega (log-log)
    ax = axes[0]
    mask = (omega > 0) & (G_prime > 0) & (G_double_prime > 0)
    ax.loglog(omega[mask], G_prime[mask], "o-", markersize=4, label="$G'$")
    ax.loglog(omega[mask], G_double_prime[mask], "s-", markersize=4, label="$G''$")
    ax.set_xlabel(r"Angular frequency $\omega$ (rad/s)")
    ax.set_ylabel("Modulus (Pa)")
    ax.set_title(f"{title}: Dynamic Moduli")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # tan(delta) vs omega
    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        tan_delta = G_double_prime / G_prime
    mask = (omega > 0) & np.isfinite(tan_delta) & (tan_delta > 0)
    ax.loglog(omega[mask], tan_delta[mask], "^-", markersize=4, color="C2")
    ax.set_xlabel(r"Angular frequency $\omega$ (rad/s)")
    ax.set_ylabel(r"$\tan(\delta) = G''/G'$")
    ax.set_title(f"{title}: Loss Tangent")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_laos_lissajous(
    strain: np.ndarray,
    stress: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "LAOS Lissajous",
) -> plt.Figure:
    """Create Lissajous curve (stress vs strain) plot.

    Args:
        strain: Strain array (dimensionless).
        stress: Stress array (Pa).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(strain, stress, "b-", linewidth=1)
    ax.set_xlabel(r"Strain $\gamma$")
    ax.set_ylabel(r"Stress $\sigma$ (Pa)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("auto")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_laos_harmonics(
    harmonics: dict[str, Any],
    save_path: Path | str | None = None,
    title: str = "LAOS Harmonics",
) -> plt.Figure:
    """Plot LAOS harmonic spectrum.

    Args:
        harmonics: Dict from compute_laos_harmonics().
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    harm_vals = harmonics["harmonics"]
    n_harm = len(harm_vals)
    harm_numbers = np.arange(1, n_harm + 1)

    ax.bar(harm_numbers, harm_vals, color="C0", edgecolor="k", alpha=0.7)
    ax.set_xlabel("Harmonic number $n$")
    ax.set_ylabel("Amplitude $I_n$ (Pa)")
    ax.set_title(f"{title}\n$I_3/I_1 = {harmonics['I3_I1']:.4f}$")
    ax.set_xticks(harm_numbers)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_laos_timeseries(
    time: np.ndarray,
    strain: np.ndarray,
    stress: np.ndarray,
    save_path: Path | str | None = None,
    title: str = "LAOS Time Series",
) -> plt.Figure:
    """Plot LAOS time series (strain and stress vs time).

    Args:
        time: Time array (s).
        strain: Strain array.
        stress: Stress array (Pa).
        save_path: Path to save figure (optional).
        title: Figure title.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(time, strain, "b-", linewidth=1)
    ax.set_ylabel(r"Strain $\gamma$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time, stress, "r-", linewidth=1)
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"Stress $\sigma$ (Pa)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# =============================================================================
# Validation Report
# =============================================================================


def write_validation_report(
    report: dict[str, Any],
    save_path: Path | str,
) -> None:
    """Write validation report to JSON file.

    Args:
        report: Dict with validation results.
        save_path: Path to write JSON file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, ValidationResult):
            return {
                "check_name": obj.check_name,
                "passed": obj.passed,
                "message": obj.message,
                "details": convert(obj.details),
            }
        elif isinstance(obj, DatasetValidation):
            return {
                "file_path": obj.file_path,
                "protocol": obj.protocol,
                "passed": obj.passed,
                "n_passed": obj.n_passed,
                "n_failed": obj.n_failed,
                "results": convert(obj.results),
            }
        return obj

    report = convert(report)
    report["generated_at"] = datetime.now().isoformat()

    with open(save_path, "w") as f:
        json.dump(report, f, indent=2)


def print_validation_summary(validations: list[DatasetValidation]) -> None:
    """Print a summary table of validation results.

    Args:
        validations: List of DatasetValidation objects.
    """
    print("\nValidation Summary")
    print("=" * 80)
    print(f"{'File':<40} {'Status':<10} {'Passed':<8} {'Failed':<8}")
    print("-" * 80)

    total_passed = 0
    total_failed = 0

    for v in validations:
        status = "PASS" if v.passed else "FAIL"
        file_name = Path(v.file_path).name[:38]
        print(f"{file_name:<40} {status:<10} {v.n_passed:<8} {v.n_failed:<8}")
        total_passed += v.n_passed
        total_failed += v.n_failed

    print("-" * 80)
    all_pass = all(v.passed for v in validations)
    overall = "ALL PASS" if all_pass else "SOME FAIL"
    print(f"{'TOTAL':<40} {overall:<10} {total_passed:<8} {total_failed:<8}")
    print("=" * 80)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_output_directories(protocol: Protocol) -> dict[str, Path]:
    """Create output directory structure for a protocol.

    Args:
        protocol: One of the supported protocols.

    Returns:
        Dict with paths for 'cleaned_data', 'derived_quantities', 'plots'.
    """
    base = get_output_dir(protocol)
    paths = {
        "cleaned_data": base / "cleaned_data",
        "derived_quantities": base / "derived_quantities",
        "plots": base / "plots",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_csv_flexible(
    file_path: Path | str,
    sep: str | None = None,
) -> pd.DataFrame:
    """Load CSV with flexible separator detection.

    Args:
        file_path: Path to CSV file.
        sep: Separator (None to auto-detect).

    Returns:
        DataFrame with loaded data.
    """
    file_path = Path(file_path)

    # Try to detect separator
    if sep is None:
        with open(file_path) as f:
            first_line = f.readline()
            if "\t" in first_line:
                sep = "\t"
            elif ";" in first_line:
                sep = ";"
            else:
                sep = ","

    return pd.read_csv(file_path, sep=sep)


def check_uniform_sampling(
    time: np.ndarray,
    tolerance: float = 0.1,
) -> ValidationResult:
    """Check if time array is uniformly sampled.

    Args:
        time: Time array.
        tolerance: Relative tolerance for dt variation.

    Returns:
        ValidationResult with pass/fail status.
    """
    time = np.asarray(time)
    dt = np.diff(time)
    dt_mean = np.mean(dt)
    dt_std = np.std(dt)

    rel_std = dt_std / dt_mean if dt_mean > 0 else np.inf

    if rel_std > tolerance:
        return ValidationResult(
            check_name="uniform_sampling",
            passed=False,
            message=f"Non-uniform sampling: dt_std/dt_mean = {rel_std:.3f} > {tolerance}",
            details={"dt_mean": float(dt_mean), "dt_std": float(dt_std), "rel_std": float(rel_std)},
        )

    return ValidationResult(
        check_name="uniform_sampling",
        passed=True,
        message=f"Uniform sampling: dt = {dt_mean:.4g} ± {dt_std:.2g} s",
        details={"dt_mean": float(dt_mean), "dt_std": float(dt_std), "rel_std": float(rel_std)},
    )
