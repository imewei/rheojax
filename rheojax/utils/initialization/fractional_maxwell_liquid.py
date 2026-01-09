"""Initializer for FractionalMaxwellLiquid model from oscillation data.

Model equation:
    G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)

Extraction strategy:
    - Gm: Maxwell modulus from high-frequency plateau
    - tau_alpha: relaxation time from transition frequency
    - alpha: fractional order from slope or default to 0.5
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalMaxwellLiquidInitializer(BaseInitializer):
    """Smart initialization for FractionalMaxwellLiquid from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FML parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: Gm, alpha, tau_alpha
        """
        logger.debug(
            "Estimating FractionalMaxwellLiquid parameters",
            model="FractionalMaxwellLiquid",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Gm: Maxwell modulus from high-frequency plateau
        Gm_init = max(features["high_plateau"], epsilon)
        logger.debug(
            "Estimated Gm from high-frequency plateau",
            Gm=Gm_init,
            high_plateau=features["high_plateau"],
        )

        # tau_alpha: relaxation time from transition frequency
        tau_alpha_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated tau_alpha from transition frequency",
            tau_alpha=tau_alpha_init,
            omega_mid=features["omega_mid"],
        )

        # alpha: fractional order from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
            logger.debug(
                "Using alpha from slope estimate",
                alpha=alpha_init,
                source="slope_estimate",
            )
        else:
            alpha_init = 0.5
            logger.debug(
                "Using default alpha (slope estimate out of range)",
                alpha=alpha_init,
                alpha_estimate=features["alpha_estimate"],
                source="default",
            )

        estimated = {
            "Gm": Gm_init,
            "alpha": alpha_init,
            "tau_alpha": tau_alpha_init,
        }

        logger.info(
            "FractionalMaxwellLiquid initialization complete",
            model="FractionalMaxwellLiquid",
            Gm=Gm_init,
            alpha=alpha_init,
            tau_alpha=tau_alpha_init,
        )

        return estimated

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FML parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        logger.debug(
            "Setting FractionalMaxwellLiquid parameters in ParameterSet",
            Gm=clipped_params["Gm"],
            alpha=clipped_params["alpha"],
            tau_alpha=clipped_params["tau_alpha"],
        )

        self._safe_set_parameter(param_set, "Gm", clipped_params["Gm"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau_alpha", clipped_params["tau_alpha"])
