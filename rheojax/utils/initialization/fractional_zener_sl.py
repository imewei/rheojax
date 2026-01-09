"""Initializer for FractionalZenerSolidLiquid model from oscillation data.

Model equation:
    G*(ω) = G_e + c_α * (iω)^α / (1 + (iωτ)^(1-α))

Extraction strategy:
    - Ge: equilibrium modulus from low-frequency plateau
    - c_alpha: from plateau difference (high - low)
    - alpha: fractional order from slope or default to 0.5
    - tau: relaxation time from transition frequency
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalZenerSLInitializer(BaseInitializer):
    """Smart initialization for FractionalZenerSolidLiquid from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalZenerSL parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: Ge, c_alpha, alpha, tau
        """
        logger.debug(
            "Estimating FractionalZenerSolidLiquid parameters",
            model="FractionalZenerSolidLiquid",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Ge: equilibrium modulus from low-frequency plateau
        Ge_init = max(features["low_plateau"], epsilon)
        logger.debug(
            "Estimated Ge from low-frequency plateau",
            Ge=Ge_init,
            low_plateau=features["low_plateau"],
        )

        # c_alpha: from plateau difference (high - low)
        c_alpha_init = max(features["high_plateau"] - features["low_plateau"], epsilon)
        logger.debug(
            "Estimated c_alpha from plateau difference",
            c_alpha=c_alpha_init,
            difference=features["high_plateau"] - features["low_plateau"],
        )

        # tau: relaxation time from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated tau from transition frequency",
            tau=tau_init,
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
            "Ge": Ge_init,
            "c_alpha": c_alpha_init,
            "alpha": alpha_init,
            "tau": tau_init,
        }

        logger.info(
            "FractionalZenerSolidLiquid initialization complete",
            model="FractionalZenerSolidLiquid",
            Ge=Ge_init,
            c_alpha=c_alpha_init,
            alpha=alpha_init,
            tau=tau_init,
        )

        return estimated

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalZenerSL parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        logger.debug(
            "Setting FractionalZenerSolidLiquid parameters in ParameterSet",
            Ge=clipped_params["Ge"],
            c_alpha=clipped_params["c_alpha"],
            alpha=clipped_params["alpha"],
            tau=clipped_params["tau"],
        )

        self._safe_set_parameter(param_set, "Ge", clipped_params["Ge"])
        self._safe_set_parameter(param_set, "c_alpha", clipped_params["c_alpha"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
