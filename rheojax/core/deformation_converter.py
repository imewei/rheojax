"""Deformation mode conversion (E* <-> G*) at the BaseModel boundary.

Centralizes the tensile-to-shear (and back) modulus conversion that
appears in both fit() and fit_bayesian(), eliminating duplication.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    import numpy as np

    type ArrayLike = np.ndarray

logger = get_logger(__name__)


class DeformationModeConverter:
    """Handles E*/G* conversion at the fit/predict boundary.

    Usage:
        converter = DeformationModeConverter()
        y, deformation_mode = converter.convert_to_shear(
            y, deformation_mode, poisson_ratio, model_name
        )
        # ... fit in shear space ...
        result = converter.convert_from_shear(
            result, deformation_mode, poisson_ratio, model_name
        )
    """

    @staticmethod
    def resolve_deformation_mode(
        deformation_mode: str | DeformationMode | None,
    ) -> DeformationMode | None:
        """Normalize a string or enum deformation_mode to DeformationMode or None."""
        if deformation_mode is None:
            return None
        if isinstance(deformation_mode, str):
            return DeformationMode(deformation_mode)
        return deformation_mode

    @staticmethod
    def convert_to_shear(
        y: ArrayLike | None,
        deformation_mode: DeformationMode | None,
        poisson_ratio: float,
        model_name: str = "",
    ) -> ArrayLike | None:
        """Convert tensile modulus E* to shear modulus G* if needed.

        Args:
            y: Target data (may be E* or G*).
            deformation_mode: Resolved DeformationMode enum or None.
            poisson_ratio: Poisson's ratio for conversion.
            model_name: For logging only.

        Returns:
            y in shear space (unchanged if already shear or deformation_mode is None).
        """
        if deformation_mode is None or y is None:
            return y
        if not deformation_mode.is_tensile():
            return y

        from rheojax.utils.modulus_conversion import convert_modulus

        y_shear = convert_modulus(
            y, deformation_mode, DeformationMode.SHEAR, poisson_ratio
        )
        logger.info(
            "Converted tensile modulus to shear for fitting",
            model=model_name,
            from_mode=str(deformation_mode),
            poisson_ratio=poisson_ratio,
        )
        return y_shear

    @staticmethod
    def convert_from_shear(
        result: ArrayLike,
        deformation_mode: DeformationMode | None,
        poisson_ratio: float,
        model_name: str = "",
    ) -> ArrayLike:
        """Convert shear modulus G* back to tensile E* if needed.

        Args:
            result: Prediction in shear space.
            deformation_mode: Resolved DeformationMode enum or None.
            poisson_ratio: Poisson's ratio for conversion.
            model_name: For logging only.

        Returns:
            result in the original deformation space.
        """
        if deformation_mode is None:
            return result
        if not deformation_mode.is_tensile():
            return result

        from rheojax.utils.modulus_conversion import convert_modulus

        return convert_modulus(
            result, DeformationMode.SHEAR, deformation_mode, poisson_ratio
        )
