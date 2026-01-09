"""Mixin for common fractional model behavior.

This module provides FractionalModelMixin, which encapsulates shared functionality
across all 11 fractional viscoelastic models, reducing code duplication and
improving maintainability.

Phase 3 of Template Method Refactoring: FractionalModelMixin
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.test_modes import TestMode
from rheojax.logging import get_logger

logger = get_logger(__name__)

FRACTIONAL_ORDER_EPS = 1e-3
# Shared bounds for fractional orders (public API remains [0, 1])
FRACTIONAL_ORDER_BOUNDS = (0.0, 1.0)

if TYPE_CHECKING:
    from rheojax.core.parameters import ParameterSet


class FractionalModelMixin:
    """Mixin providing common functionality for fractional viscoelastic models.

    This mixin encapsulates shared behavior across all 11 fractional models:
    - Smart initialization for oscillation mode
    - Common parameter validation
    - Consistent error handling

    Usage:
        class FractionalZenerSolidSolid(BaseModel, FractionalModelMixin):
            ...
    """

    # Subclasses must define this mapping to their concrete initializer
    _INITIALIZER_MAP = {
        "FractionalZenerSolidSolid": "FractionalZenerSSInitializer",
        "FractionalMaxwellLiquid": "FractionalMaxwellLiquidInitializer",
        "FractionalMaxwellGel": "FractionalMaxwellGelInitializer",
        "FractionalZenerLiquidLiquid": "FractionalZenerLLInitializer",
        "FractionalZenerSolidLiquid": "FractionalZenerSLInitializer",
        "FractionalKelvinVoigt": "FractionalKelvinVoigtInitializer",
        "FractionalKVZener": "FractionalKVZenerInitializer",
        "FractionalMaxwellModel": "FractionalMaxwellModelInitializer",
        "FractionalPoyntingThomson": "FractionalPoyntingThomsonInitializer",
        "FractionalJeffreysModel": "FractionalJeffreysInitializer",
        "FractionalBurgersModel": "FractionalBurgersInitializer",
    }

    def _apply_smart_initialization(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | TestMode,
        parameters: ParameterSet,
    ) -> bool:
        """Apply smart initialization for oscillation mode.

        This method provides consistent smart initialization across all fractional
        models by delegating to model-specific initializers.

        Parameters
        ----------
        X : np.ndarray
            Input data (time or frequency)
        y : np.ndarray
            Output data (modulus or compliance)
        test_mode : str | TestMode
            Test mode (relaxation, creep, oscillation, rotation)
        parameters : ParameterSet
            ParameterSet to initialize

        Returns
        -------
        bool
            True if initialization succeeded, False otherwise

        Notes
        -----
        Only applies initialization for oscillation mode. Other modes use
        default parameter values.
        """
        # Convert test_mode to TestMode enum if needed
        if isinstance(test_mode, str):
            test_mode = TestMode(test_mode)

        # Only initialize for oscillation mode
        if test_mode != TestMode.OSCILLATION:
            return False

        # Get class name to lookup initializer
        class_name = self.__class__.__name__

        # Get initializer class name
        initializer_name = self._INITIALIZER_MAP.get(class_name)
        if initializer_name is None:
            logger.warning(
                "No initializer mapping for model class, skipping smart initialization",
                class_name=class_name,
            )
            return False

        try:
            # Dynamically import initializer module
            import importlib

            # Convert class name to module name (e.g., FractionalZenerSSInitializer -> fractional_zener_ss)
            module_name = self._class_to_module_name(class_name)
            module_path = f"rheojax.utils.initialization.{module_name}"

            # Import initializer class
            initializer_module = importlib.import_module(module_path)
            initializer_class = getattr(initializer_module, initializer_name)

            # Create initializer and run
            initializer = initializer_class()
            logger.debug(
                "Attempting smart initialization",
                class_name=class_name,
                initializer=initializer_name,
                data_shape=X.shape,
            )
            success = initializer.initialize(X, y, parameters)

            if success:
                logger.debug(
                    "Smart initialization applied from frequency-domain features",
                    class_name=class_name,
                )
                return True
            else:
                logger.debug(
                    "Smart initialization failed validation, using defaults",
                    class_name=class_name,
                )
                return False

        except Exception as e:
            # Silent fallback to defaults - don't break if initialization fails
            logger.debug(
                "Smart initialization error, falling back to defaults",
                class_name=class_name,
                error=str(e),
                exc_info=True,
            )
            return False

    @staticmethod
    def _class_to_module_name(class_name: str) -> str:
        """Convert class name to module name.

        Examples:
            FractionalZenerSolidSolid -> fractional_zener_ss
            FractionalMaxwellLiquid -> fractional_maxwell_liquid
        """
        # Handle special cases first
        abbrev_map = {
            "FractionalZenerSolidSolid": "fractional_zener_ss",
            "FractionalZenerLiquidLiquid": "fractional_zener_ll",
            "FractionalZenerSolidLiquid": "fractional_zener_sl",
            "FractionalKVZener": "fractional_kv_zener",
            "FractionalJeffreysModel": "fractional_jeffreys",
            "FractionalBurgersModel": "fractional_burgers",
            "FractionalPoyntingThomson": "fractional_poynting_thomson",
        }

        if class_name in abbrev_map:
            return abbrev_map[class_name]

        # Default: convert CamelCase to snake_case
        import re

        # Insert underscore before capitals
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        # Handle sequences of capitals
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    def _validate_fractional_parameters(self, parameters: ParameterSet) -> None:
        """Validate fractional model parameters.

        Checks:
        - Fractional order α ∈ (0, 1)
        - Positive moduli/compliances
        - Positive characteristic times

        Parameters
        ----------
        parameters : ParameterSet
            ParameterSet to validate

        Raises
        ------
        ValueError
            If parameters are invalid
        """
        logger.debug(
            "Validating fractional parameters",
            class_name=self.__class__.__name__,
        )

        # Validate alpha parameter if it exists
        alpha_param = parameters.get("alpha")
        if alpha_param is not None and alpha_param.value is not None:
            alpha = alpha_param.value
            if alpha_param.was_clamped:
                logger.error(
                    "Fractional order alpha clamped outside valid range",
                    alpha=alpha,
                    valid_range="(0, 1)",
                )
                raise ValueError(
                    f"Fractional order alpha must be in (0, 1), got {alpha}"
                )
            if not (0 < alpha < 1):
                logger.error(
                    "Fractional order alpha outside valid range",
                    alpha=alpha,
                    valid_range="(0, 1)",
                )
                raise ValueError(
                    f"Fractional order alpha must be in (0, 1), got {alpha}"
                )

        # Validate beta parameter if it exists (for Maxwell Model)
        beta_param = parameters.get("beta")
        if beta_param is not None and beta_param.value is not None:
            beta = beta_param.value
            if beta_param.was_clamped:
                logger.error(
                    "Fractional order beta clamped outside valid range",
                    beta=beta,
                    valid_range="(0, 1)",
                )
                raise ValueError(f"Fractional order beta must be in (0, 1), got {beta}")
            if not (0 < beta < 1):
                logger.error(
                    "Fractional order beta outside valid range",
                    beta=beta,
                    valid_range="(0, 1)",
                )
                raise ValueError(f"Fractional order beta must be in (0, 1), got {beta}")

        # Validate moduli are positive
        for param_name in ["Ge", "Gm", "G0", "Gg"]:
            value = parameters.get_value(param_name)
            if value is not None and value <= 0:
                logger.error(
                    "Modulus parameter must be positive",
                    parameter=param_name,
                    value=value,
                )
                raise ValueError(f"{param_name} must be positive, got {value}")

        # Validate compliances are positive
        for param_name in ["Jg", "Jm", "Je"]:
            value = parameters.get_value(param_name)
            if value is not None and value <= 0:
                logger.error(
                    "Compliance parameter must be positive",
                    parameter=param_name,
                    value=value,
                )
                raise ValueError(f"{param_name} must be positive, got {value}")

        # Validate time scales are positive
        for param_name in ["tau_alpha", "tau_beta", "tau1", "tau2"]:
            value = parameters.get_value(param_name)
            if value is not None and value <= 0:
                logger.error(
                    "Time scale parameter must be positive",
                    parameter=param_name,
                    value=value,
                )
                raise ValueError(f"{param_name} must be positive, got {value}")

        logger.debug("Fractional parameter validation passed")


__all__ = ["FractionalModelMixin", "FRACTIONAL_ORDER_BOUNDS", "FRACTIONAL_ORDER_EPS"]
