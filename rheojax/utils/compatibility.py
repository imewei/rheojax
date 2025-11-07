"""Model-data compatibility checking for rheological models.

This module provides functions to assess whether a given model is appropriate
for a dataset based on the underlying physics and data characteristics.

The compatibility checker helps users understand when model failures are expected
due to physics mismatch rather than optimization issues.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.stats import linregress

if TYPE_CHECKING:
    from rheojax.core.base import BaseModel

logger = logging.getLogger(__name__)


class DecayType(Enum):
    """Types of relaxation decay behavior."""

    EXPONENTIAL = "exponential"  # Simple Maxwell-like exp(-t/tau)
    POWER_LAW = "power_law"  # Power-law t^(-alpha)
    STRETCHED = "stretched"  # Stretched exponential exp(-(t/tau)^beta)
    MITTAG_LEFFLER = "mittag_leffler"  # Mittag-Leffler E_alpha(-(t/tau)^alpha)
    MULTI_MODE = "multi_mode"  # Multiple relaxation modes
    UNKNOWN = "unknown"  # Cannot determine


class MaterialType(Enum):
    """Types of material behavior."""

    SOLID = "solid"  # Solid-like (finite equilibrium modulus)
    LIQUID = "liquid"  # Liquid-like (zero equilibrium modulus, flows)
    GEL = "gel"  # Gel-like (power-law relaxation)
    VISCOELASTIC_SOLID = "viscoelastic_solid"  # Viscoelastic solid
    VISCOELASTIC_LIQUID = "viscoelastic_liquid"  # Viscoelastic liquid
    UNKNOWN = "unknown"


def detect_decay_type(t: np.ndarray, G_t: np.ndarray) -> DecayType:
    """Detect the type of relaxation decay from time-domain data.

    Analyzes the decay pattern to determine if it follows exponential,
    power-law, stretched exponential, or Mittag-Leffler behavior.

    Parameters
    ----------
    t : np.ndarray
        Time array (s)
    G_t : np.ndarray
        Relaxation modulus array (Pa)

    Returns
    -------
    DecayType
        Detected decay type
    """
    if len(t) < 10 or len(G_t) < 10:
        return DecayType.UNKNOWN

    # Remove any invalid values
    valid = np.isfinite(t) & np.isfinite(G_t) & (t > 0) & (G_t > 0)
    if np.sum(valid) < 10:
        return DecayType.UNKNOWN

    t = t[valid]
    G_t = G_t[valid]

    # Ensure data is sorted by time
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    G_t = G_t[sort_idx]

    # 1. Check for exponential decay: log(G) vs t should be linear
    try:
        log_G = np.log(G_t)
        slope_exp, intercept_exp, r_exp, _, _ = linregress(t, log_G)
        r_exp_sq = r_exp**2
    except (ValueError, RuntimeWarning):
        r_exp_sq = 0.0

    # 2. Check for power-law decay: log(G) vs log(t) should be linear
    try:
        log_t = np.log(t)
        log_G = np.log(G_t)
        slope_pow, intercept_pow, r_pow, _, _ = linregress(log_t, log_G)
        r_pow_sq = r_pow**2
    except (ValueError, RuntimeWarning):
        r_pow_sq = 0.0

    # 3. Check for stretched exponential: log(-log(G/G0)) vs log(t)
    try:
        G0 = G_t[0]
        G_norm = G_t / G0
        G_norm = np.clip(G_norm, 1e-10, 1.0)  # Avoid log(0)
        log_log = np.log(-np.log(G_norm))
        log_t = np.log(t)
        valid_stretch = np.isfinite(log_log)
        if np.sum(valid_stretch) >= 5:
            slope_stretch, _, r_stretch, _, _ = linregress(
                log_t[valid_stretch], log_log[valid_stretch]
            )
            r_stretch_sq = r_stretch**2
        else:
            r_stretch_sq = 0.0
    except (ValueError, RuntimeWarning):
        r_stretch_sq = 0.0

    # Decision logic
    threshold_high = 0.90  # High confidence
    threshold_med = 0.75  # Medium confidence

    # Exponential decay (Maxwell-like)
    if r_exp_sq > threshold_high:
        # Check if slope is negative (decay)
        if slope_exp < 0:
            return DecayType.EXPONENTIAL

    # Power-law decay (gel-like)
    if r_pow_sq > threshold_high:
        # Check if slope is negative (decay)
        if slope_pow < 0:
            return DecayType.POWER_LAW

    # Stretched exponential
    if r_stretch_sq > threshold_high:
        return DecayType.STRETCHED

    # Multi-mode if nothing fits well but data shows decay
    if G_t[-1] < G_t[0] * 0.9:  # At least 10% decay
        # Check if it's a combination
        if r_exp_sq > threshold_med or r_pow_sq > threshold_med:
            return DecayType.MULTI_MODE
        else:
            # Complex decay pattern, likely Mittag-Leffler
            return DecayType.MITTAG_LEFFLER

    return DecayType.UNKNOWN


def detect_material_type(
    t: np.ndarray | None = None,
    G_t: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    G_star: np.ndarray | None = None,
) -> MaterialType:
    """Detect the material type from relaxation or oscillation data.

    Parameters
    ----------
    t : np.ndarray, optional
        Time array for relaxation data (s)
    G_t : np.ndarray, optional
        Relaxation modulus (Pa)
    omega : np.ndarray, optional
        Frequency array for oscillation data (rad/s)
    G_star : np.ndarray, optional
        Complex modulus array with shape (N, 2) where [:, 0] is G' and [:, 1] is G"

    Returns
    -------
    MaterialType
        Detected material type
    """
    # Use relaxation data if available
    if t is not None and G_t is not None and len(t) >= 10:
        return _detect_from_relaxation(t, G_t)

    # Use oscillation data
    if omega is not None and G_star is not None and len(omega) >= 10:
        return _detect_from_oscillation(omega, G_star)

    return MaterialType.UNKNOWN


def _detect_from_relaxation(t: np.ndarray, G_t: np.ndarray) -> MaterialType:
    """Detect material type from relaxation modulus."""
    # Check for equilibrium modulus (finite G at long times)
    if len(G_t) < 10:
        return MaterialType.UNKNOWN

    # Check decay type first - power-law is special
    decay_type = detect_decay_type(t, G_t)

    # Power-law relaxation is characteristic of gels, regardless of decay magnitude
    if decay_type == DecayType.POWER_LAW:
        # Power-law materials are gels or viscoelastic solids
        # Distinguish based on equilibrium modulus
        G_final = np.mean(G_t[-5:])  # Average last 5 points
        G_initial = G_t[0]
        decay_ratio = G_final / G_initial

        if decay_ratio > 0.3:  # Significant final modulus
            return MaterialType.VISCOELASTIC_SOLID
        else:
            return MaterialType.GEL

    # For non-power-law materials, use decay ratio
    G_final = np.mean(G_t[-5:])  # Average last 5 points
    G_initial = G_t[0]
    decay_ratio = G_final / G_initial

    if decay_ratio > 0.5:
        # Finite equilibrium modulus → solid-like
        return MaterialType.VISCOELASTIC_SOLID

    elif decay_ratio < 0.1:
        # Strong decay → liquid-like
        return MaterialType.VISCOELASTIC_LIQUID

    else:
        # Intermediate behavior
        return MaterialType.UNKNOWN


def _detect_from_oscillation(omega: np.ndarray, G_star: np.ndarray) -> MaterialType:
    """Detect material type from complex modulus."""
    if G_star.shape[1] != 2:
        return MaterialType.UNKNOWN

    G_prime = G_star[:, 0]  # Storage modulus
    G_double_prime = G_star[:, 1]  # Loss modulus

    # Low-frequency behavior
    low_freq_idx = np.argmin(omega)

    # Check if G' > G" at low frequency (solid-like)
    # or G" > G' (liquid-like)
    if G_prime[low_freq_idx] > G_double_prime[low_freq_idx]:
        return MaterialType.SOLID
    elif G_double_prime[low_freq_idx] > G_prime[low_freq_idx]:
        return MaterialType.LIQUID
    else:
        return MaterialType.UNKNOWN


def check_model_compatibility(
    model: BaseModel,
    t: np.ndarray | None = None,
    G_t: np.ndarray | None = None,
    omega: np.ndarray | None = None,
    G_star: np.ndarray | None = None,
    test_mode: str | None = None,
) -> dict[str, Any]:
    """Check if a model is compatible with the given data.

    This function analyzes the data characteristics and compares them with
    the model's underlying physics to assess compatibility.

    Parameters
    ----------
    model : BaseModel
        The rheological model to check
    t : np.ndarray, optional
        Time array for relaxation data (s)
    G_t : np.ndarray, optional
        Relaxation modulus (Pa)
    omega : np.ndarray, optional
        Frequency array for oscillation data (rad/s)
    G_star : np.ndarray, optional
        Complex modulus array with shape (N, 2)
    test_mode : str, optional
        Test mode ('relaxation', 'creep', 'oscillation')

    Returns
    -------
    dict
        Dictionary with compatibility information:
        - 'compatible': bool, whether model is likely compatible
        - 'confidence': float, confidence level (0-1)
        - 'decay_type': DecayType, detected decay pattern
        - 'material_type': MaterialType, detected material behavior
        - 'warnings': list[str], compatibility warnings
        - 'recommendations': list[str], suggested alternative models
    """
    warnings = []
    recommendations = []

    # Detect data characteristics
    decay_type = DecayType.UNKNOWN
    material_type = MaterialType.UNKNOWN

    if test_mode == "relaxation" and t is not None and G_t is not None:
        decay_type = detect_decay_type(t, G_t)
        material_type = detect_material_type(t=t, G_t=G_t)
    elif test_mode == "oscillation" and omega is not None and G_star is not None:
        material_type = detect_material_type(omega=omega, G_star=G_star)

    # Get model name
    model_name = model.__class__.__name__

    # Define model requirements
    compatible = True
    confidence = 0.5  # Default medium confidence

    # Fractional Zener Solid-Solid (FZSS)
    if "FractionalZenerSolidSolid" in model_name or model_name == "FZSS":
        # FZSS expects Mittag-Leffler decay with finite equilibrium modulus
        if decay_type == DecayType.EXPONENTIAL:
            compatible = False
            confidence = 0.9
            warnings.append(
                "FZSS model expects Mittag-Leffler (power-law) relaxation, "
                "but data shows exponential decay."
            )
            recommendations.append("Maxwell")
            recommendations.append("Zener")
        elif decay_type == DecayType.POWER_LAW:
            compatible = True
            confidence = 0.8
        elif material_type == MaterialType.VISCOELASTIC_LIQUID:
            compatible = False
            confidence = 0.7
            warnings.append(
                "FZSS model is designed for solid-like materials, "
                "but data shows liquid-like behavior."
            )
            recommendations.append("FractionalMaxwellLiquid")

    # Fractional Maxwell Liquid (FML)
    elif "FractionalMaxwellLiquid" in model_name or model_name == "FML":
        # FML expects liquid-like behavior (no equilibrium modulus)
        if material_type == MaterialType.SOLID:
            compatible = False
            confidence = 0.8
            warnings.append(
                "FractionalMaxwellLiquid expects liquid-like behavior, "
                "but data shows solid-like characteristics."
            )
            recommendations.append("FractionalZenerSolidSolid")
            recommendations.append("FractionalKelvinVoigt")

    # Fractional Maxwell Gel (FMG)
    elif "FractionalMaxwellGel" in model_name or model_name == "FMG":
        # FMG expects power-law relaxation (gel-like)
        if decay_type == DecayType.EXPONENTIAL:
            compatible = False
            confidence = 0.85
            warnings.append(
                "FractionalMaxwellGel expects power-law relaxation, "
                "but data shows exponential decay."
            )
            recommendations.append("Maxwell")

    # Classical Maxwell model
    elif model_name == "Maxwell":
        # Maxwell expects exponential decay
        if decay_type == DecayType.POWER_LAW:
            compatible = False
            confidence = 0.85
            warnings.append(
                "Maxwell model expects exponential decay, "
                "but data shows power-law behavior."
            )
            recommendations.append("FractionalMaxwellGel")
            recommendations.append("FractionalZenerSolidSolid")

    # Classical Zener model
    elif model_name == "Zener":
        # Zener expects exponential decay with equilibrium modulus
        if decay_type == DecayType.POWER_LAW:
            compatible = False
            confidence = 0.8
            warnings.append(
                "Zener model expects exponential decay, "
                "but data shows power-law behavior."
            )
            recommendations.append("FractionalZenerSolidSolid")

    # Fractional Kelvin-Voigt
    elif "FractionalKelvinVoigt" in model_name and "Zener" not in model_name:
        # FKV expects solid-like behavior
        if material_type == MaterialType.VISCOELASTIC_LIQUID:
            compatible = False
            confidence = 0.75
            warnings.append(
                "FractionalKelvinVoigt is designed for solid-like materials, "
                "but data shows liquid-like behavior."
            )
            recommendations.append("FractionalMaxwellLiquid")

    return {
        "compatible": compatible,
        "confidence": confidence,
        "decay_type": decay_type,
        "material_type": material_type,
        "warnings": warnings,
        "recommendations": recommendations,
    }


def format_compatibility_message(compatibility: dict) -> str:
    """Format compatibility check results as a user-friendly message.

    Parameters
    ----------
    compatibility : dict
        Compatibility check results from check_model_compatibility()

    Returns
    -------
    str
        Formatted message
    """
    lines = []

    if compatibility["compatible"]:
        lines.append("✓ Model appears compatible with data")
        lines.append(f"  Confidence: {compatibility['confidence']*100:.0f}%")
    else:
        lines.append("⚠ Model may not be appropriate for this data")
        lines.append(f"  Confidence: {compatibility['confidence']*100:.0f}%")

    # Add detected characteristics
    decay = compatibility["decay_type"]
    material = compatibility["material_type"]

    if decay != DecayType.UNKNOWN:
        lines.append(f"  Detected decay: {decay.value}")
    if material != MaterialType.UNKNOWN:
        lines.append(f"  Material type: {material.value}")

    # Add warnings
    if compatibility["warnings"]:
        lines.append("\nWarnings:")
        for warning in compatibility["warnings"]:
            lines.append(f"  • {warning}")

    # Add recommendations
    if compatibility["recommendations"]:
        lines.append("\nRecommended alternative models:")
        for rec in compatibility["recommendations"]:
            lines.append(f"  • {rec}")

    return "\n".join(lines)


__all__ = [
    "DecayType",
    "MaterialType",
    "detect_decay_type",
    "detect_material_type",
    "check_model_compatibility",
    "format_compatibility_message",
]
