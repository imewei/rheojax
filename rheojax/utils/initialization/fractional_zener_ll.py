"""Initializer for FractionalZenerLiquidLiquid model from oscillation data.

Model equation:
    G*(ω) = c_1 * (iω)^α / (1 + (iωτ)^β) + c_2 * (iω)^γ

Extraction strategy (simplified for 6-parameter model):
    - c1, c2: Split high-frequency plateau between both terms
    - alpha, beta, gamma: Use slope estimate or defaults
    - tau: From transition frequency

Note: This is a simplified initialization for a highly complex model.
The optimizer will refine these starting values.
"""

from __future__ import annotations

from rheojax.logging import get_logger
from rheojax.utils.initialization.base import BaseInitializer

logger = get_logger(__name__)


class FractionalZenerLLInitializer(BaseInitializer):
    """Smart initialization for FractionalZenerLiquidLiquid from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalZenerLL parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: c1, c2, alpha, beta, gamma, tau
        """
        logger.debug(
            "Estimating FractionalZenerLiquidLiquid parameters",
            model="FractionalZenerLiquidLiquid",
            low_plateau=features["low_plateau"],
            high_plateau=features["high_plateau"],
            omega_mid=features["omega_mid"],
            alpha_estimate=features["alpha_estimate"],
        )

        epsilon = 1e-12

        # Split the high-frequency modulus between c1 and c2
        total_modulus = max(features["high_plateau"], epsilon)
        c1_init = total_modulus * 0.6  # Allocate more to first term
        c2_init = total_modulus * 0.4
        logger.debug(
            "Split modulus between c1 and c2",
            model="FractionalZenerLiquidLiquid",
            total_modulus=total_modulus,
            c1=c1_init,
            c2=c2_init,
        )

        # Use slope estimate for alpha, defaults for beta and gamma
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
            logger.debug(
                "Using estimated alpha from slope",
                model="FractionalZenerLiquidLiquid",
                alpha=alpha_init,
            )
        else:
            alpha_init = 0.5
            logger.debug(
                "Alpha estimate out of range, using default",
                model="FractionalZenerLiquidLiquid",
                alpha=alpha_init,
                original_estimate=features["alpha_estimate"],
            )

        beta_init = 0.5  # Default for second fractional order
        gamma_init = 0.5  # Default for third fractional order
        logger.debug(
            "Using default fractional orders for beta and gamma",
            model="FractionalZenerLiquidLiquid",
            beta=beta_init,
            gamma=gamma_init,
        )

        # tau from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)
        logger.debug(
            "Estimated characteristic time tau",
            model="FractionalZenerLiquidLiquid",
            tau=tau_init,
        )

        return {
            "c1": c1_init,
            "c2": c2_init,
            "alpha": alpha_init,
            "beta": beta_init,
            "gamma": gamma_init,
            "tau": tau_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalZenerLL parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "c1", clipped_params["c1"])
        self._safe_set_parameter(param_set, "c2", clipped_params["c2"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "beta", clipped_params["beta"])
        self._safe_set_parameter(param_set, "gamma", clipped_params["gamma"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
