"""Initializer for FractionalPoyntingThomson model from oscillation data.

Model equation (in compliance):
    J*(ω) = 1/G_e + (1/G_k) / (1 + (iωτ)^α)
    G*(ω) = 1 / J*(ω)

Extraction strategy:
    - Ge: from high-frequency limit
    - Gk: from modulus difference
    - tau: from transition frequency
    - alpha: from slope or default to 0.5
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalPoyntingThomsonInitializer(BaseInitializer):
    """Smart initialization for FractionalPoyntingThomson from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalPoyntingThomson parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: Ge, Gk, alpha, tau
        """
        logger.debug(
            "Estimating FractionalPoyntingThomson parameters",
            model="FractionalPoyntingThomson",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Ge: instantaneous modulus from high-frequency plateau
        Ge_init = max(features["high_plateau"], epsilon)
        logger.debug(
            "Estimated instantaneous modulus Ge",
            model="FractionalPoyntingThomson",
            Ge=Ge_init,
        )

        # Gk: retarded modulus from difference
        Gk_init = max(features["high_plateau"] - features["low_plateau"], epsilon)
        logger.debug(
            "Estimated retarded modulus Gk",
            model="FractionalPoyntingThomson",
            Gk=Gk_init,
        )

        # tau: from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated characteristic time tau",
            model="FractionalPoyntingThomson",
            tau=tau_init,
        )

        # alpha: from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
            logger.debug(
                "Using estimated alpha from slope",
                model="FractionalPoyntingThomson",
                alpha=alpha_init,
            )
        else:
            alpha_init = 0.5
            logger.debug(
                "Alpha estimate out of range, using default",
                model="FractionalPoyntingThomson",
                alpha=alpha_init,
                original_estimate=features["alpha_estimate"],
            )

        return {
            "Ge": Ge_init,
            "Gk": Gk_init,
            "alpha": alpha_init,
            "tau": tau_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalPoyntingThomson parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "Ge", clipped_params["Ge"])
        self._safe_set_parameter(param_set, "Gk", clipped_params["Gk"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
