"""Modulus conversion utilities for DMTA/DMA data analysis.

This module provides functions to convert between shear modulus G* (measured
by rotational rheometers) and Young's modulus E* (measured by DMTA/DMA
instruments in tension, bending, or compression).

The fundamental relationship from isotropic linear elasticity:

    E*(w) = 2(1 + v) * G*(w)

where v is the Poisson's ratio of the material.

Example:
    >>> from rheojax.utils.modulus_conversion import convert_modulus
    >>> from rheojax.core.test_modes import DeformationMode
    >>>
    >>> # Convert E* (DMTA) to G* (shear) for rubber (v=0.5 -> factor=3)
    >>> G_star = convert_modulus(E_star, DeformationMode.TENSION, DeformationMode.SHEAR, poisson_ratio=0.5)
    >>>
    >>> # Use preset materials
    >>> from rheojax.utils.modulus_conversion import POISSON_PRESETS
    >>> nu = POISSON_PRESETS["glassy_polymer"]  # 0.35
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from rheojax.core.data import RheoData

logger = get_logger(__name__)

# Common Poisson's ratio values by material class
POISSON_PRESETS: dict[str, float] = {
    "rubber": 0.5,
    "elastomer": 0.5,
    "glassy_polymer": 0.35,
    "semicrystalline": 0.40,
    "metal": 0.30,
    "thermoset": 0.38,
    "hydrogel": 0.50,
    "foam": 0.30,
}


def _validate_poisson_ratio(nu: float) -> None:
    """Validate Poisson's ratio is within physical bounds.

    For isotropic materials, thermodynamic stability requires -1 < v <= 0.5.
    Most polymers have v in [0.3, 0.5].

    Args:
        nu: Poisson's ratio to validate

    Raises:
        ValueError: If Poisson's ratio is outside physical bounds
    """
    if not isinstance(nu, (int, float)):
        raise TypeError(f"Poisson's ratio must be a number, got {type(nu).__name__}")
    if nu <= -1.0 or nu > 0.5:
        raise ValueError(
            f"Poisson's ratio must be in (-1, 0.5], got {nu}. "
            f"Common values: rubber=0.5, glassy_polymer=0.35, metal=0.30"
        )


def _conversion_factor(poisson_ratio: float) -> float:
    """Compute E/G conversion factor: 2(1 + v).

    Args:
        poisson_ratio: Poisson's ratio of the material

    Returns:
        Conversion factor (E = factor * G)
    """
    return 2.0 * (1.0 + poisson_ratio)


def convert_modulus(
    data: np.ndarray | Any,
    from_mode: DeformationMode | str,
    to_mode: DeformationMode | str,
    poisson_ratio: float = 0.5,
) -> np.ndarray | Any:
    """Convert modulus data between shear (G*) and tensile (E*) representations.

    Applies the isotropic elasticity relationship E* = 2(1+v) * G*.
    Works with both real and complex arrays, and both NumPy and JAX arrays.

    Args:
        data: Modulus data array (real or complex). Can be NumPy or JAX array.
        from_mode: Source deformation mode (e.g., "tension", "shear")
        to_mode: Target deformation mode (e.g., "shear", "tension")
        poisson_ratio: Poisson's ratio of the material (default: 0.5 for rubber)

    Returns:
        Converted modulus data in the same array type as input

    Raises:
        ValueError: If Poisson's ratio is out of bounds or modes are invalid

    Example:
        >>> E_star = np.array([1e9 + 1e8j, 2e9 + 2e8j])  # E* in Pa
        >>> G_star = convert_modulus(E_star, "tension", "shear", poisson_ratio=0.5)
        >>> # G_star ≈ E_star / 3 for rubber
    """
    # Normalize mode strings to enums
    if isinstance(from_mode, str):
        from_mode = DeformationMode(from_mode)
    if isinstance(to_mode, str):
        to_mode = DeformationMode(to_mode)

    # No conversion needed if modes are equivalent
    if from_mode == to_mode:
        return data
    if from_mode.is_tensile() == to_mode.is_tensile():
        # Both tensile or both shear — no conversion needed
        return data

    _validate_poisson_ratio(poisson_ratio)
    factor = _conversion_factor(poisson_ratio)

    # SUP-010: Warn for non-finite values in input data
    data_arr = np.asarray(data) if not hasattr(data, "devices") else data
    if hasattr(data_arr, "size") and not np.all(np.isfinite(data_arr)):
        import warnings
        warnings.warn(
            "Non-finite values (NaN/Inf) detected in modulus data. "
            "These will propagate through the conversion.",
            UserWarning,
            stacklevel=2,
        )

    if from_mode.is_tensile() and not to_mode.is_tensile():
        # E* -> G*: divide by factor
        logger.debug(
            "Converting E* to G*",
            from_mode=str(from_mode),
            to_mode=str(to_mode),
            poisson_ratio=poisson_ratio,
            factor=factor,
        )
        return data / factor
    else:
        # G* -> E*: multiply by factor
        logger.debug(
            "Converting G* to E*",
            from_mode=str(from_mode),
            to_mode=str(to_mode),
            poisson_ratio=poisson_ratio,
            factor=factor,
        )
        return data * factor


def convert_rheodata(
    data: RheoData,
    to_mode: DeformationMode | str,
    poisson_ratio: float = 0.5,
) -> RheoData:
    """Convert RheoData between shear and tensile modulus representations.

    Creates a new RheoData with converted y-values and updated metadata.
    The original RheoData is not modified.

    Args:
        data: Source RheoData object
        to_mode: Target deformation mode
        poisson_ratio: Poisson's ratio of the material

    Returns:
        New RheoData with converted modulus values and updated metadata

    Example:
        >>> from rheojax.core.data import RheoData
        >>> # DMTA data in tension
        >>> dmta_data = RheoData(x=omega, y=E_star,
        ...     metadata={"deformation_mode": "tension"})
        >>> # Convert to shear for model fitting
        >>> shear_data = convert_rheodata(dmta_data, "shear", poisson_ratio=0.5)
    """
    from rheojax.core.data import RheoData

    if isinstance(to_mode, str):
        to_mode = DeformationMode(to_mode)

    # Determine source mode from metadata
    from_mode_str = data.metadata.get("deformation_mode", "shear")
    from_mode = DeformationMode(from_mode_str)

    # Convert y-data
    y_converted = convert_modulus(data.y, from_mode, to_mode, poisson_ratio)

    # Build updated metadata
    new_metadata = dict(data.metadata)
    new_metadata["deformation_mode"] = to_mode.value
    new_metadata["poisson_ratio"] = poisson_ratio
    if from_mode != to_mode:
        new_metadata["converted_from"] = from_mode.value

    # Update y-units label if present (only replace modulus symbols, not SI prefixes)
    new_y_units = data.y_units
    if new_y_units:
        import re

        if to_mode.is_tensile():
            # Replace G-modulus symbols: G*, G', G" but NOT G in GPa/GHz
            new_y_units = re.sub(r"\bG([*'\"\s])", r"E\1", new_y_units)
            # Handle standalone G at end of string
            new_y_units = re.sub(r"\bG$", "E", new_y_units)
        elif not to_mode.is_tensile():
            new_y_units = re.sub(r"\bE([*'\"\s])", r"G\1", new_y_units)
            new_y_units = re.sub(r"\bE$", "G", new_y_units)

    return RheoData(
        x=data.x,
        y=y_converted,
        x_units=data.x_units,
        y_units=new_y_units,
        domain=data.domain,
        metadata=new_metadata,
        validate=False,
    )


__all__ = [
    "POISSON_PRESETS",
    "convert_modulus",
    "convert_rheodata",
]
