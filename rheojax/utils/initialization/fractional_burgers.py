"""Initializer for FractionalBurgers model from oscillation data.

Model equation (in compliance):
    J*(ω) = J_g + (iω)^(-α) / (η_1 Γ(1-α)) + J_k / (1 + (iωτ_k)^α)
    G*(ω) = 1 / J*(ω)

Extraction strategy (simplified for 5-parameter model):
    - Jg: from 1/high_plateau (glassy compliance)
    - Jk: from compliance difference
    - eta1: from low-frequency flow behavior
    - tau_k: from transition frequency
    - alpha: from slope or default to 0.5

Note: This is a simplified initialization for a complex model.
The optimizer will refine these starting values.
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalBurgersInitializer(BaseInitializer):
    """Smart initialization for FractionalBurgers from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalBurgers parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: Jg, eta1, Jk, alpha, tau_k
        """
        logger.debug(
            "Estimating FractionalBurgers parameters",
            model="FractionalBurgers",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Jg: glassy compliance from 1/high_plateau
        Jg_init = 1.0 / max(features["high_plateau"], epsilon)
        logger.debug(
            "Estimated glassy compliance Jg",
            model="FractionalBurgers",
            Jg=Jg_init,
        )

        # Jk: Kelvin compliance from compliance difference
        Jk_init = (1.0 / max(features["low_plateau"], epsilon)) - Jg_init
        Jk_init = max(Jk_init, epsilon)
        logger.debug(
            "Estimated Kelvin compliance Jk",
            model="FractionalBurgers",
            Jk=Jk_init,
        )

        # eta1: viscosity from low-frequency behavior
        # Approximate omega_low as omega_mid / 10
        omega_low = features["omega_mid"] / 10.0
        eta1_init = max(features["low_plateau"] / (omega_low + epsilon), epsilon)
        logger.debug(
            "Estimated viscosity eta1",
            model="FractionalBurgers",
            eta1=eta1_init,
            omega_low=omega_low,
        )

        # tau_k: from transition frequency
        tau_k_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated characteristic time tau_k",
            model="FractionalBurgers",
            tau_k=tau_k_init,
        )

        # alpha: from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
            logger.debug(
                "Using estimated alpha from slope",
                model="FractionalBurgers",
                alpha=alpha_init,
            )
        else:
            alpha_init = 0.5
            logger.debug(
                "Alpha estimate out of range, using default",
                model="FractionalBurgers",
                alpha=alpha_init,
                original_estimate=features["alpha_estimate"],
            )

        return {
            "Jg": Jg_init,
            "eta1": eta1_init,
            "Jk": Jk_init,
            "alpha": alpha_init,
            "tau_k": tau_k_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalBurgers parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "Jg", clipped_params["Jg"])
        self._safe_set_parameter(param_set, "eta1", clipped_params["eta1"])
        self._safe_set_parameter(param_set, "Jk", clipped_params["Jk"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau_k", clipped_params["tau_k"])
