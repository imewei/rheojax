"""Shared utilities for ITT-MCT tutorial notebooks.

Provides consistent data loading, synthetic generation, result saving, and
diagnostic printing across all ITT-MCT model protocol tutorials (Schematic, ISM).

Data Sources (reusing IKH carbopol datasets):
- ML-IKH Experimental data.xlsx: Wei et al. 2018 J. Rheol (flow curves)
- PNAS_DigitalRheometerTwin_Dataset.xlsx: PNAS 2022 (startup, LAOS)

Note: While ITT-MCT is designed for dense colloidal suspensions/glasses and
carbopol is a yield-stress polymer gel, we use this data to demonstrate the
modeling workflow for educational purposes.
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Reuse IKH data loaders
try:
    from ikh_tutorial_utils import (
        load_ml_ikh_flow_curve,
        load_pnas_laos,
        load_pnas_startup,
    )
except ModuleNotFoundError:
    from utils.ikh_tutorial_utils import (
        load_ml_ikh_flow_curve,
        load_pnas_laos,
        load_pnas_startup,
    )


# =============================================================================
# Data Loaders (Wrappers for IKH loaders)
# =============================================================================


def load_carbopol_flow_curve(
    instrument: Literal["ARG2_up", "ARG2_down", "ARES_up", "ARES_down"] = "ARES_up",
) -> tuple[np.ndarray, np.ndarray]:
    """Load steady-state flow curve data.

    Wraps IKH data loader for consistent naming across tutorials.

    Args:
        instrument: Which dataset to load (default ARES_up is recommended).

    Returns:
        Tuple of (shear_rate, stress) arrays in (1/s, Pa).
    """
    return load_ml_ikh_flow_curve(instrument=instrument)


def load_carbopol_startup(
    gamma_dot: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load startup shear data from PNAS dataset.

    Args:
        gamma_dot: Shear rate to load. Available: 0.056, 0.32, 1, 56.2, 100.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    return load_pnas_startup(gamma_dot=gamma_dot)


def load_carbopol_laos(
    omega: float = 1.0,
    strain_amplitude_index: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load LAOS data from PNAS dataset.

    Args:
        omega: Angular frequency (1, 3, or 5 rad/s).
        strain_amplitude_index: Index for strain amplitude (0-11).

    Returns:
        Tuple of (time, strain, stress) arrays in (s, -, Pa).
    """
    return load_pnas_laos(omega=omega, strain_amplitude_index=strain_amplitude_index)


# =============================================================================
# Synthetic Data Generators
# =============================================================================


def generate_synthetic_relaxation_schematic(
    model: Any,
    sigma_0: float = 100.0,
    t_end: float = 100.0,
    n_points: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stress relaxation data from a fitted Schematic model.

    Uses the model's _predict_relaxation method to generate clean data,
    then adds Gaussian noise.

    Args:
        model: Fitted ITTMCTSchematic model instance.
        sigma_0: Initial stress in Pa (used to compute initial strain).
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

    # Compute pre-shear strain from initial stress
    G_inf = model.parameters.get_value("G_inf")
    gamma_pre = sigma_0 / G_inf

    # Use model's prediction method
    stress_clean = model._predict_relaxation(time, gamma_pre=gamma_pre)
    stress_clean = np.asarray(stress_clean).flatten()

    # Add relative noise
    noise = rng.normal(
        0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape
    )
    stress = stress_clean + noise

    # Ensure positive stress
    stress = np.maximum(stress, 0.0)

    return time, stress


def generate_synthetic_relaxation_isotropic(
    model: Any,
    sigma_0: float = 100.0,
    t_end: float = 100.0,
    n_points: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stress relaxation data from a fitted ISM model.

    Args:
        model: Fitted ITTMCTIsotropic model instance.
        sigma_0: Initial stress in Pa.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level.
        seed: Random seed.

    Returns:
        Tuple of (time, stress) arrays in (s, Pa).
    """
    rng = np.random.default_rng(seed)

    time = np.logspace(-2, np.log10(t_end), n_points)

    # Estimate pre-shear strain
    kBT = model.parameters.get_value("kBT")
    sigma_d = model.parameters.get_value("sigma_d")
    G_scale = kBT / sigma_d**3
    gamma_pre = sigma_0 / G_scale

    stress_clean = model._predict_relaxation(time, gamma_pre=gamma_pre)
    stress_clean = np.asarray(stress_clean).flatten()

    noise = rng.normal(
        0, noise_level * np.mean(np.abs(stress_clean)), size=stress_clean.shape
    )
    stress = stress_clean + noise
    stress = np.maximum(stress, 0.0)

    return time, stress


def generate_synthetic_saos_schematic(
    model: Any,
    omega_range: tuple[float, float] | None = None,
    omega_min: float | None = None,  # Alias for omega_range[0]
    omega_max: float | None = None,  # Alias for omega_range[1]
    n_points: int = 50,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic SAOS data from a fitted Schematic model.

    Args:
        model: Fitted ITTMCTSchematic model instance.
        omega_range: (min, max) angular frequency range in rad/s.
        omega_min: Alias for omega_range[0].
        omega_max: Alias for omega_range[1].
        n_points: Number of frequency points.
        noise_level: Relative noise level.
        seed: Random seed.

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays in (rad/s, Pa, Pa).
    """
    # Handle parameter aliasing
    if omega_range is None:
        o_min = omega_min if omega_min is not None else 0.01
        o_max = omega_max if omega_max is not None else 100.0
        omega_range = (o_min, o_max)

    rng = np.random.default_rng(seed)

    omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_points)

    # Use model's oscillation prediction
    G_components = model._predict_oscillation(omega, return_components=True)
    G_prime = G_components[:, 0]
    G_double_prime = G_components[:, 1]

    # Add noise (use absolute mean for scale to handle negative values in fluid state)
    noise_scale_p = noise_level * np.abs(np.mean(G_prime))
    noise_scale_pp = noise_level * np.abs(np.mean(G_double_prime))
    # Ensure positive scale
    noise_scale_p = max(noise_scale_p, 1e-10)
    noise_scale_pp = max(noise_scale_pp, 1e-10)
    noise_p = rng.normal(0, noise_scale_p, size=G_prime.shape)
    noise_pp = rng.normal(0, noise_scale_pp, size=G_double_prime.shape)

    G_prime = G_prime + noise_p
    G_double_prime = G_double_prime + noise_pp

    # Ensure positive moduli
    G_prime = np.maximum(G_prime, 1e-10)
    G_double_prime = np.maximum(G_double_prime, 1e-10)

    return omega, G_prime, G_double_prime


def generate_synthetic_saos_isotropic(
    model: Any,
    omega_range: tuple[float, float] | None = None,
    omega_min: float | None = None,  # Alias for omega_range[0]
    omega_max: float | None = None,  # Alias for omega_range[1]
    n_points: int = 50,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic SAOS data from a fitted ISM model.

    Args:
        model: Fitted ITTMCTIsotropic model instance.
        omega_range: (min, max) angular frequency range in rad/s.
        omega_min: Alias for omega_range[0].
        omega_max: Alias for omega_range[1].
        n_points: Number of frequency points.
        noise_level: Relative noise level.
        seed: Random seed.

    Returns:
        Tuple of (omega, G_prime, G_double_prime) arrays in (rad/s, Pa, Pa).
    """
    # Handle parameter aliasing
    if omega_range is None:
        o_min = omega_min if omega_min is not None else 0.01
        o_max = omega_max if omega_max is not None else 100.0
        omega_range = (o_min, o_max)

    rng = np.random.default_rng(seed)

    omega = np.logspace(np.log10(omega_range[0]), np.log10(omega_range[1]), n_points)

    G_components = model._predict_oscillation(omega, return_components=True)
    G_prime = G_components[:, 0]
    G_double_prime = G_components[:, 1]

    noise_p = rng.normal(0, noise_level * np.mean(G_prime), size=G_prime.shape)
    noise_pp = rng.normal(
        0, noise_level * np.mean(G_double_prime), size=G_double_prime.shape
    )

    G_prime = np.maximum(G_prime + noise_p, 1e-10)
    G_double_prime = np.maximum(G_double_prime + noise_pp, 1e-10)

    return omega, G_prime, G_double_prime


def generate_synthetic_creep_schematic(
    model: Any,
    sigma_applied: float | None = None,
    sigma_0: float | None = None,  # Alias for sigma_applied
    t_end: float = 100.0,
    n_points: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic creep compliance data from a fitted Schematic model.

    Args:
        model: Fitted ITTMCTSchematic model instance.
        sigma_applied: Applied stress in Pa (alternative: sigma_0).
        sigma_0: Alias for sigma_applied.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level.
        seed: Random seed.

    Returns:
        Tuple of (time, compliance) arrays in (s, 1/Pa).
    """
    # Handle parameter aliasing
    if sigma_0 is not None and sigma_applied is None:
        sigma_applied = sigma_0
    elif sigma_applied is None:
        sigma_applied = 50.0

    rng = np.random.default_rng(seed)

    time = np.logspace(-2, np.log10(t_end), n_points)

    J_clean = model._predict_creep(time, sigma_applied=sigma_applied)
    J_clean = np.asarray(J_clean).flatten()

    log_noise = rng.normal(0.0, noise_level, size=J_clean.shape)
    J = np.maximum(J_clean, 1e-30) * np.exp(log_noise)

    return time, J


def generate_synthetic_creep_isotropic(
    model: Any,
    sigma_applied: float | None = None,
    sigma_0: float | None = None,  # Alias for sigma_applied
    t_end: float = 100.0,
    n_points: int = 200,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic creep compliance data from a fitted ISM model.

    Returns the creep compliance J(t) = γ(t)/σ₀, which is the quantity the
    ITT-MCT creep fit (``_fit_creep``) and ``model.predict(test_mode="creep")``
    operate on — matching ``generate_synthetic_creep_schematic`` and the
    codebase-wide convention that the creep observable is compliance, not
    strain. In the linear regime J(t) is stress-independent, so ``sigma_0``
    sets only the experimental strain scale, not the compliance.

    Args:
        model: Fitted ITTMCTIsotropic model instance.
        sigma_applied: Applied stress in Pa (alternative: sigma_0).
        sigma_0: Alias for sigma_applied.
        t_end: End time in seconds.
        n_points: Number of time points.
        noise_level: Relative noise level for compliance.
        seed: Random seed.

    Returns:
        Tuple of (time, compliance) arrays in (s, 1/Pa).
    """
    # Handle parameter aliasing
    if sigma_0 is not None and sigma_applied is None:
        sigma_applied = sigma_0
    elif sigma_applied is None:
        sigma_applied = 50.0

    rng = np.random.default_rng(seed)

    time = np.logspace(-2, np.log10(t_end), n_points)

    # Creep compliance J(t) from the model (stress-independent in linear regime).
    J_clean = model._predict_creep(time, sigma_applied=sigma_applied)
    J_clean = np.asarray(J_clean).flatten()

    # Multiplicative noise on the compliance.
    log_noise = rng.normal(0.0, noise_level, size=J_clean.shape)
    J = np.maximum(J_clean, 1e-30) * np.exp(log_noise)

    return time, J


def generate_synthetic_laos_isotropic(
    model: Any,
    gamma_0: float = 0.15,
    omega: float = 1.0,
    n_cycles: int = 4,
    n_points: int = 300,
    noise_level: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic LAOS (time, strain, stress) from an ISM model.

    The imposed strain is gamma(t) = gamma_0 * sin(omega * t); the stress is the
    model's LAOS prediction. Unlike the positive-definite moduli/compliance
    generators (which use multiplicative log-normal noise), LAOS stress is a
    signed oscillatory signal crossing zero, so additive Gaussian noise scaled
    to the stress amplitude is the appropriate noise model.

    Args:
        model: ITTMCTIsotropic instance (parameters define the ground truth).
        gamma_0: Strain amplitude (gamma_0 > gamma_c gives nonlinear response).
        omega: Angular frequency (rad/s).
        n_cycles: Number of oscillation cycles spanned by the time grid.
        n_points: Number of time samples.
        noise_level: Additive noise as a fraction of the stress amplitude.
        seed: Random seed.

    Returns:
        Tuple of (time, strain, stress) arrays in (s, -, Pa).
    """
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, n_cycles * 2.0 * np.pi / omega, n_points)
    strain = gamma_0 * np.sin(omega * time)

    stress_clean = np.asarray(
        model.predict(time, test_mode="laos", gamma_0=gamma_0, omega=omega)
    ).flatten()
    amp = float(np.max(np.abs(stress_clean)))
    stress = stress_clean + rng.normal(
        0.0, noise_level * max(amp, 1e-30), size=stress_clean.shape
    )
    return time, strain, stress


# =============================================================================
# Physics Functions
# =============================================================================


def compute_f12_memory_kernel(
    phi: float | np.ndarray,
    v1: float,
    v2: float,
) -> float | np.ndarray:
    """Compute F12 schematic memory kernel m(Phi) = v1*Phi + v2*Phi^2.

    Args:
        phi: Correlator value(s) in [0, 1].
        v1: Linear vertex coefficient.
        v2: Quadratic vertex coefficient.

    Returns:
        Memory kernel value(s).
    """
    return v1 * phi + v2 * phi**2


def compute_gaussian_decorrelation(
    gamma: float | np.ndarray,
    gamma_c: float,
) -> float | np.ndarray:
    """Compute Gaussian strain decorrelation function h(gamma).

    h(gamma) = exp(-(gamma/gamma_c)^2)

    This function describes how accumulated strain breaks the cage and
    decorrelates the density fluctuations.

    Args:
        gamma: Accumulated strain.
        gamma_c: Critical strain for cage breaking.

    Returns:
        Decorrelation factor in [0, 1].
    """
    return np.exp(-((gamma / gamma_c) ** 2))


def compute_lorentzian_decorrelation(
    gamma: float | np.ndarray,
    gamma_c: float,
) -> float | np.ndarray:
    """Compute Lorentzian strain decorrelation function h(gamma).

    h(gamma) = 1 / (1 + (gamma/gamma_c)^2)

    Alternative to Gaussian with slower algebraic decay.

    Args:
        gamma: Accumulated strain.
        gamma_c: Critical strain for cage breaking.

    Returns:
        Decorrelation factor in [0, 1].
    """
    return 1.0 / (1.0 + (gamma / gamma_c) ** 2)


def compute_non_ergodicity_parameter(
    v1: float,
    v2: float,
) -> float:
    """Compute non-ergodicity parameter f from F12 vertices.

    Solves the MCT self-consistency equation:
        f / (1 - f) = v1*f + v2*f^2
    The physical solution is the *largest* root f > f_c (≈ 0.5 for v1=0).

    For v1=0, the equation reduces to a quadratic with exact solution:
        f = (1 + sqrt(1 - 4/v2)) / 2

    For v1 != 0, uses Brent's method on (f_c, 1) where f_c is the critical
    non-ergodicity parameter at the glass transition.

    Reference: Götze & Sjögren 1992, Eq. (2.6).

    Args:
        v1: Linear vertex coefficient.
        v2: Quadratic vertex coefficient.

    Returns:
        Non-ergodicity parameter f in [0, 1). Returns 0 for fluid state.
    """
    # Critical v2 for glass transition
    if abs(v1) < 1e-10:
        v2_c = 4.0
    else:
        v2_c = (4.0 - 2.0 * v1) / (1.0 - v1 / 4.0) if v1 < 4.0 else 4.0

    epsilon = (v2 - v2_c) / v2_c
    if epsilon <= 0:
        return 0.0

    # For v1=0: exact quadratic solution
    # f/(1-f) = v2*f^2  =>  1 = v2*f*(1-f)  =>  f = (1 + sqrt(1 - 4/v2))/2
    if abs(v1) < 1e-10:
        disc = 1.0 - 4.0 / v2
        if disc < 0:
            return 0.0
        return (1.0 + np.sqrt(disc)) / 2.0

    # General case (v1 != 0): numerical root-finding
    # The physical root is f > f_c where f_c ≈ 0.5 at the transition.
    # Use f_c as the lower bracket (residual is ~0 there, slightly positive
    # just above transition) and 1-eps as upper bracket (residual is negative
    # because f/(1-f) -> inf while v1*f+v2*f^2 stays bounded).
    from scipy.optimize import brentq

    # Estimate f_c at the critical point (v2=v2_c)
    # At transition: f_c/(1-f_c) = v1*f_c + v2_c*f_c^2, solved by bisection
    f_c = 0.5  # Good starting estimate

    def _f_neq_eq(f: float) -> float:
        if f >= 1.0:
            return -1.0
        return f / (1.0 - f) - (v1 * f + v2 * f * f)

    try:
        f_neq = brentq(_f_neq_eq, f_c, 1.0 - 1e-8)
    except ValueError:
        # Fallback: try wider bracket
        try:
            f_neq = brentq(_f_neq_eq, 0.01, 1.0 - 1e-8)
        except ValueError:
            f_neq = 0.0

    return f_neq


def compute_separation_parameter(
    v1: float,
    v2: float,
) -> float:
    """Compute separation parameter epsilon = (v2 - v2_c) / v2_c.

    Args:
        v1: Linear vertex coefficient.
        v2: Quadratic vertex coefficient.

    Returns:
        Separation parameter (negative = fluid, positive = glass).
    """
    if abs(v1) < 1e-10:
        v2_c = 4.0
    else:
        v2_c = (4.0 - 2.0 * v1) / (1.0 - v1 / 4.0) if v1 < 4.0 else 4.0

    return (v2 - v2_c) / v2_c


def interpret_glass_state(info: dict[str, Any]) -> str:
    """Format glass transition info as human-readable string.

    Args:
        info: Dictionary from model.get_glass_transition_info()

    Returns:
        Formatted interpretation string.
    """
    lines = []

    if "epsilon" in info:
        # Schematic model
        epsilon = info["epsilon"]
        is_glass = info.get("is_glass", epsilon > 0)
        state = "GLASS" if is_glass else "FLUID"
        lines.append(f"State: {state}")
        lines.append(f"Separation parameter: epsilon = {epsilon:.4f}")
        if is_glass:
            f_neq = info.get(
                "f_neq", compute_non_ergodicity_parameter(0, 4 * (1 + epsilon))
            )
            lines.append(f"Non-ergodicity parameter: f = {f_neq:.4f}")
            lines.append("  -> Correlator plateaus at f > 0 (arrested dynamics)")
            lines.append("  -> Material shows yield stress")
        else:
            lines.append("  -> Correlator decays to 0 (ergodic dynamics)")
            lines.append("  -> Material flows like a viscous liquid")
    elif "phi" in info:
        # Isotropic model
        phi = info["phi"]
        phi_mct = info.get("phi_mct", 0.516)
        is_glass = info.get("is_glass", phi > phi_mct)
        state = "GLASS" if is_glass else "FLUID"
        lines.append(f"State: {state}")
        lines.append(f"Volume fraction: phi = {phi:.4f}")
        lines.append(f"MCT transition: phi_MCT = {phi_mct:.4f}")
        if is_glass:
            lines.append("  -> phi > phi_MCT: cage effect dominates")
            lines.append("  -> Material shows yield stress")
        else:
            lines.append("  -> phi < phi_MCT: particles can diffuse")
            lines.append("  -> Material flows")

    return "\n".join(lines)


# =============================================================================
# Parameter Names
# =============================================================================


def get_schematic_param_names() -> list[str]:
    """Get the list of ITT-MCT Schematic (F12) parameter names.

    Returns:
        List of 5 Schematic parameter names.
    """
    return [
        "v1",  # Linear vertex coefficient (typically 0)
        "v2",  # Quadratic vertex coefficient (glass at v2 > 4)
        "Gamma",  # Bare relaxation rate (1/s)
        "gamma_c",  # Critical strain for cage breaking
        "G_inf",  # High-frequency modulus (Pa)
    ]


def get_isotropic_param_names() -> list[str]:
    """Get the list of ITT-MCT Isotropic (ISM) parameter names.

    Returns:
        List of 5 ISM parameter names.
    """
    return [
        "phi",  # Volume fraction (glass at phi > 0.516)
        "sigma_d",  # Particle diameter (m)
        "D0",  # Bare diffusion coefficient (m^2/s)
        "kBT",  # Thermal energy (J)
        "gamma_c",  # Critical strain for cage breaking
    ]


# =============================================================================
# Result Persistence
# =============================================================================


def get_output_dir(
    model_name: Literal["schematic", "isotropic"],
    protocol: str,
) -> Path:
    """Get the output directory path for a given model and protocol.

    Args:
        model_name: One of 'schematic' or 'isotropic'.
        protocol: Protocol name (e.g., 'flow_curve', 'startup').

    Returns:
        Path to output directory.
    """
    module_dir = Path(__file__).parent
    return module_dir / ".." / "outputs" / "itt_mct" / model_name / protocol


def save_itt_mct_results(
    model: Any,
    result: Any,
    model_name: Literal["schematic", "isotropic"],
    protocol: str,
    param_names: list[str] | None = None,
) -> None:
    """Save NLSQ and Bayesian results for reuse in other notebooks.

    Args:
        model: Fitted ITT-MCT model.
        result: Bayesian fit result with posterior_samples attribute, or None if not available.
        model_name: One of 'schematic' or 'isotropic'.
        protocol: Protocol name for labeling files.
        param_names: List of parameter names to save. If None, uses defaults.
    """
    output_dir = get_output_dir(model_name, protocol)
    os.makedirs(output_dir, exist_ok=True)

    # Get parameter names from model type if not provided
    if param_names is None:
        if model_name == "schematic":
            param_names = get_schematic_param_names()
        else:
            param_names = get_isotropic_param_names()

    # Save NLSQ point estimates
    nlsq_params = {}
    for name in param_names:
        try:
            nlsq_params[name] = float(model.parameters.get_value(name))
        except (KeyError, AttributeError):
            pass

    with open(output_dir / f"nlsq_params_{protocol}.json", "w") as f:
        json.dump(nlsq_params, f, indent=2)

    # Save posterior samples (only if result is available)
    if result is not None:
        posterior = result.posterior_samples
        posterior_dict = {k: np.array(v).tolist() for k, v in posterior.items()}
        with open(output_dir / f"posterior_{protocol}.json", "w") as f:
            json.dump(posterior_dict, f)
        print(f"  posterior_{protocol}.json: {len(list(posterior.values())[0])} draws")
    else:
        print("  Skipping posterior (Bayesian inference not available for ITT-MCT)")

    # Save glass transition info
    try:
        glass_info = model.get_glass_transition_info()
        # Convert any numpy types to Python types
        glass_info_clean = {
            k: float(v) if hasattr(v, "item") else v for k, v in glass_info.items()
        }
        with open(output_dir / f"glass_info_{protocol}.json", "w") as f:
            json.dump(glass_info_clean, f, indent=2)
    except Exception:
        pass

    print(f"Results saved to {output_dir}/")
    print(f"  nlsq_params_{protocol}.json: {len(nlsq_params)} parameters")


def load_itt_mct_parameters(
    model_name: Literal["schematic", "isotropic"],
    protocol: str,
    require_glass: bool = False,
) -> dict[str, float]:
    """Load previously calibrated ITT-MCT parameters.

    Args:
        model_name: One of 'schematic' or 'isotropic'.
        protocol: Protocol name (e.g., 'flow_curve').
        require_glass: If True, validates that v2 > 4 for glass state.
                       Falls back to defaults if fluid state detected.

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

    # Validate glass state if required. Schematic uses v2 > 4; the isotropic
    # (ISM) model is a glass for phi above the MCT transition (~0.516 for
    # Percus-Yevick hard spheres), so it must be validated on phi, not v2.
    if require_glass:
        schematic_fluid = model_name == "schematic" and params.get("v2", 99) <= 4.0
        isotropic_fluid = model_name == "isotropic" and params.get("phi", 0) <= 0.516
        if schematic_fluid or isotropic_fluid:
            bad = (
                f"v2={params.get('v2')}"
                if schematic_fluid
                else f"phi={params.get('phi')}"
            )
            print(f"Warning: Loaded parameters are in the fluid state ({bad}).")
            print("Falling back to default glass-state parameters.")
            if model_name == "schematic":
                params = {"v2": 4.2, "Gamma": 1.0, "gamma_c": 0.1, "G_inf": 1000.0}
            else:
                # Representative dense colloidal-glass values (1 um PMMA-like
                # hard spheres at 298 K): these yield a physical Pa-scale
                # plateau modulus, not the 1 m / 1 J placeholders.
                params = {
                    "phi": 0.58,
                    "sigma_d": 1e-6,
                    "D0": 1e-12,
                    "kBT": 4.11e-21,
                    "gamma_c": 0.1,
                }

    return params


def load_itt_mct_glass_info(
    model_name: Literal["schematic", "isotropic"],
    protocol: str,
) -> dict[str, Any]:
    """Load previously saved glass transition info.

    Args:
        model_name: One of 'schematic' or 'isotropic'.
        protocol: Protocol name.

    Returns:
        Dictionary of glass transition properties.
    """
    output_dir = get_output_dir(model_name, protocol)
    info_file = output_dir / f"glass_info_{protocol}.json"

    if not info_file.exists():
        raise FileNotFoundError(f"No saved glass info for {model_name}/{protocol}.")

    with open(info_file) as f:
        return json.load(f)


def fit_isotropic_phi(
    model: Any,
    X: np.ndarray,
    y_data: np.ndarray,
    test_mode: str,
    phi_bounds: tuple[float, float] = (0.50, 0.63),
    **predict_kwargs: Any,
) -> float:
    """Refine the volume fraction phi against any-protocol data.

    For the ITT-MCT Isotropic model the microscopic parameters (D0, sigma_d,
    kBT) are fixed by the flow-curve calibration; the volume fraction phi is the
    one shape parameter the *other* protocols constrain. Because phi enters the
    structure factor S(k) and the MCT vertex, it must be refit through
    ``update_structure_factor`` (recomputing S(k)/vertex), not via ``set_value``
    alone — the protocol ``model.fit`` path keeps S(k) cached and also tries to
    move the degenerate, badly-scaled microscopic parameters, so a direct
    ``model.fit`` is ill-conditioned here.

    A 1-D bounded search over phi is used. Positive-definite observables
    (compliance, complex modulus) are compared in log space; signed observables
    (e.g. LAOS stress) use a scale-normalised linear residual.

    Args:
        model: ITTMCTIsotropic instance (mutated in place to the best phi).
        X: Independent variable (time or angular frequency).
        y_data: Observable — real array, or complex G* for oscillation.
        test_mode: One of "creep", "oscillation", "laos", "relaxation", ...
        phi_bounds: Search bounds for phi.
        **predict_kwargs: Extra kwargs forwarded to ``model.predict`` (e.g.
            ``sigma_0``, ``gamma_0``, ``omega``, ``gamma_pre``).

    Returns:
        The fitted volume fraction phi.
    """
    from scipy.optimize import minimize_scalar

    y = np.asarray(y_data)
    is_complex = np.iscomplexobj(y)
    if is_complex:
        y_vec = np.concatenate([np.real(y).ravel(), np.imag(y).ravel()])
    else:
        y_vec = y.ravel()

    # Log residuals for positive observables; linear (scaled) for signed data.
    use_log = is_complex or bool(np.all(y_vec > 0))
    target = np.log(np.maximum(y_vec, 1e-30)) if use_log else y_vec
    scale = max(float(np.std(y_vec)), 1e-30)

    def _loss(phi: float) -> float:
        model.update_structure_factor(phi=float(phi))
        p = np.asarray(model.predict(X, test_mode=test_mode, **predict_kwargs))
        if is_complex:
            p_vec = np.concatenate([np.real(p).ravel(), np.imag(p).ravel()])
        else:
            p_vec = p.ravel()
        if use_log:
            return float(np.mean((np.log(np.maximum(p_vec, 1e-30)) - target) ** 2))
        return float(np.mean(((p_vec - target) / scale) ** 2))

    res = minimize_scalar(_loss, bounds=phi_bounds, method="bounded")
    model.update_structure_factor(phi=float(res.x))
    return float(res.x)


def fit_isotropic_gamma_c(
    model: Any,
    X: np.ndarray,
    y_data: np.ndarray,
    test_mode: str,
    gamma_c_bounds: tuple[float, float] | None = None,
    **predict_kwargs: Any,
) -> float:
    """Refine the cage-breaking strain gamma_c against nonlinear-protocol data.

    Unlike phi (which enters the structure factor S(k) and the MCT vertex, and
    so must be refit through ``update_structure_factor``), gamma_c only scales
    the strain decorrelation h(gamma/gamma_c). It is therefore set directly with
    ``set_value`` and found by a fast 1-D bounded search over the cached kernels.

    gamma_c is the parameter that *nonlinear* protocols (LAOS, large-amplitude
    startup) constrain, because it sets the strain scale at which cages break —
    in the linear regime (creep, SAOS) gamma_c is inert.

    Args:
        model: ITTMCTIsotropic instance (mutated in place to the best gamma_c).
        X: Independent variable (time).
        y_data: Observable (e.g. LAOS stress, signed).
        test_mode: Protocol name, e.g. "laos" or "startup".
        gamma_c_bounds: Search bounds for gamma_c. If None, the model's own
            parameter constraints are used; otherwise the request is clamped to
            those constraints so the search never proposes a value ``set_value``
            would reject.
        **predict_kwargs: Extra kwargs forwarded to ``model.predict`` (e.g.
            ``gamma_0``, ``omega``).

    Returns:
        The fitted gamma_c.
    """
    from scipy.optimize import minimize_scalar

    y = np.asarray(y_data).ravel()
    scale = max(float(np.std(y)), 1e-30)

    # Respect the model's hard constraints on gamma_c (set_value enforces them).
    lo_c, hi_c = model.parameters["gamma_c"].bounds
    if gamma_c_bounds is None:
        lo, hi = float(lo_c), float(hi_c)
    else:
        lo = max(float(gamma_c_bounds[0]), float(lo_c))
        hi = min(float(gamma_c_bounds[1]), float(hi_c))

    def _loss(gamma_c: float) -> float:
        model.parameters.set_value("gamma_c", float(gamma_c))
        p = np.asarray(model.predict(X, test_mode=test_mode, **predict_kwargs)).ravel()
        return float(np.mean(((p - y) / scale) ** 2))

    res = minimize_scalar(_loss, bounds=(lo, hi), method="bounded")
    model.parameters.set_value("gamma_c", float(res.x))
    return float(res.x)


def fit_isotropic_phi_creep(
    model: Any,
    t: np.ndarray,
    J_data: np.ndarray,
    sigma_0: float,
    phi_bounds: tuple[float, float] = (0.50, 0.63),
) -> float:
    """Refine phi from creep-compliance data (thin wrapper of fit_isotropic_phi)."""
    return fit_isotropic_phi(
        model, t, J_data, "creep", phi_bounds=phi_bounds, sigma_0=sigma_0
    )


def set_model_parameters(model: Any, params: dict[str, float]) -> None:
    """Set model parameters from a dictionary.

    Args:
        model: ITT-MCT model instance.
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
        param_names = get_schematic_param_names()

    diag = result.diagnostics

    print("Convergence Diagnostics")
    print("=" * 55)
    print(f"{'Parameter':>15s}  {'R-hat':>8s}  {'ESS':>8s}  {'Status':>8s}")
    print("-" * 55)

    all_pass = True
    for p in param_names:
        r_hat = diag.get("r_hat", {}).get(p, float("nan"))
        ess = diag.get("ess", {}).get(p, float("nan"))

        # Check thresholds (relaxed for fast demo mode)
        r_hat_ok = r_hat < 1.1  # 1.05 for production
        ess_ok = ess > 50  # 100 for production

        if r_hat_ok and ess_ok:
            status = "PASS"
        else:
            status = "CHECK"
            all_pass = False

        print(f"{p:>15s}  {r_hat:8.4f}  {ess:8.0f}  {status:>8s}")

    n_div = diag.get("divergences", diag.get("num_divergences", 0))
    print(f"\nDivergences: {n_div}")

    if n_div > 10:  # Relaxed threshold for fast demo
        all_pass = False

    if all_pass:
        print("\nAll convergence criteria PASSED (fast demo thresholds)")
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


def print_glass_state_summary(model: Any) -> None:
    """Print summary of model's glass transition state.

    Args:
        model: ITT-MCT model instance.
    """
    try:
        info = model.get_glass_transition_info()
        print("\nGlass Transition State")
        print("=" * 50)
        print(interpret_glass_state(info))
    except Exception as e:
        print(f"Could not get glass state info: {e}")


def compute_fit_quality(y_data: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute fit quality metrics (R-squared, RMSE, NRMSE).

    Args:
        y_data: Observed data array.
        y_pred: Predicted data array.

    Returns:
        Dictionary with 'R2', 'RMSE', and 'NRMSE' keys.
    """
    y_data = np.asarray(y_data)
    y_pred = np.asarray(y_pred)
    # Complex G* → stack real (G') and imag (G'') as real residuals
    if np.iscomplexobj(y_data) or np.iscomplexobj(y_pred):
        y_data = np.concatenate([np.real(y_data).ravel(), np.imag(y_data).ravel()])
        y_pred = np.concatenate([np.real(y_pred).ravel(), np.imag(y_pred).ravel()])
    else:
        y_data = y_data.flatten()
        y_pred = y_pred.flatten()

    # R-squared
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    if r2 < 0:
        import warnings

        warnings.warn(
            f"R² = {r2:.4f} < 0: model fits worse than the mean. "
            "Check initial parameters, data quality, or model suitability.",
            stacklevel=2,
        )

    # RMSE
    rmse = float(np.sqrt(np.mean((y_data - y_pred) ** 2)))

    # NRMSE (normalized by range)
    data_range = np.max(y_data) - np.min(y_data)
    nrmse = float(rmse / data_range) if data_range > 0 else 0.0

    return {"R2": r2, "RMSE": rmse, "NRMSE": nrmse}


# =============================================================================
# MCT-Specific Physics Functions
# =============================================================================


def compute_mct_yield_stress(
    G_inf: float,
    gamma_c: float,
    f_neq: float,
) -> float:
    """Estimate yield stress from MCT parameters.

    For the Schematic model, the yield stress scales as:
        sigma_y ~ G_inf * gamma_c * f

    Args:
        G_inf: High-frequency modulus (Pa).
        gamma_c: Critical strain for cage breaking.
        f_neq: Non-ergodicity parameter.

    Returns:
        Estimated yield stress (Pa).
    """
    return G_inf * gamma_c * f_neq


def compute_mct_viscosity(
    G_inf: float,
    Gamma: float,
    f_neq: float,
) -> float:
    """Estimate zero-shear viscosity from MCT parameters.

    For a fluid state (f=0), viscosity is finite:
        eta_0 ~ G_inf / Gamma

    For a glass state (f>0), viscosity diverges.

    Args:
        G_inf: High-frequency modulus (Pa).
        Gamma: Bare relaxation rate (1/s).
        f_neq: Non-ergodicity parameter.

    Returns:
        Estimated viscosity (Pa.s), or inf for glass.
    """
    if f_neq > 0.01:
        return float("inf")
    return G_inf / Gamma


def compute_alpha_relaxation_time(
    Gamma: float,
    epsilon: float,
) -> float:
    """Estimate alpha relaxation time from MCT.

    Near the glass transition, the alpha relaxation time diverges as:
        tau_alpha ~ |epsilon|^(-gamma_MCT)

    with gamma_MCT ~ 2.5 for the F12 model.

    Args:
        Gamma: Bare relaxation rate (1/s).
        epsilon: Separation parameter.

    Returns:
        Alpha relaxation time (s).
    """
    if epsilon >= 0:
        return float("inf")

    gamma_mct = 2.5  # MCT exponent
    tau_0 = 1.0 / Gamma
    return tau_0 * abs(epsilon) ** (-gamma_mct)
