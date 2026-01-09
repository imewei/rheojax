"""Initializer for FractionalKelvinVoigtZener model from oscillation data.

Model equation (in compliance):
    J*(ω) = 1/G_e + (1/G_k) / (1 + (iωτ)^α)
    G*(ω) = 1 / J*(ω)

Extraction strategy:
    - Ge: from high-frequency limit (1/J_min)
    - Gk: from difference in compliances
    - tau: from transition frequency
    - alpha: from slope or default to 0.5
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalKVZenerInitializer(BaseInitializer):
    """Smart initialization for FractionalKelvinVoigtZener from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalKVZener parameters from frequency features.

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
            "Estimating FractionalKVZener parameters",
            model="FractionalKVZener",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Ge: from high-frequency plateau (series spring)
        Ge_init = max(features["high_plateau"], epsilon)
        logger.debug(
            "Estimated Ge from high-frequency plateau",
            Ge=Ge_init,
            high_plateau=features["high_plateau"],
        )

        # Gk: from modulus difference
        Gk_init = max(features["high_plateau"] - features["low_plateau"], epsilon)
        logger.debug(
            "Estimated Gk from modulus difference",
            Gk=Gk_init,
            difference=features["high_plateau"] - features["low_plateau"],
        )

        # tau: from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated tau from transition frequency",
            tau=tau_init,
            omega_mid=features["omega_mid"],
        )

        # alpha: from slope or default to 0.5
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
            "Gk": Gk_init,
            "alpha": alpha_init,
            "tau": tau_init,
        }

        logger.info(
            "FractionalKVZener initialization complete",
            model="FractionalKVZener",
            Ge=Ge_init,
            Gk=Gk_init,
            alpha=alpha_init,
            tau=tau_init,
        )

        return estimated

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalKVZener parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        logger.debug(
            "Setting FractionalKVZener parameters in ParameterSet",
            Ge=clipped_params["Ge"],
            Gk=clipped_params["Gk"],
            alpha=clipped_params["alpha"],
            tau=clipped_params["tau"],
        )

        self._safe_set_parameter(param_set, "Ge", clipped_params["Ge"])
        self._safe_set_parameter(param_set, "Gk", clipped_params["Gk"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
