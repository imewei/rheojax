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

from rheojax.utils.initialization.base import BaseInitializer


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
        epsilon = 1e-12

        # Ge: equilibrium modulus from low-frequency plateau
        Ge_init = max(features["low_plateau"], epsilon)

        # c_alpha: from plateau difference (high - low)
        c_alpha_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

        # tau: relaxation time from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)

        # alpha: fractional order from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

        return {
            "Ge": Ge_init,
            "c_alpha": c_alpha_init,
            "alpha": alpha_init,
            "tau": tau_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalZenerSL parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "Ge", clipped_params["Ge"])
        self._safe_set_parameter(param_set, "c_alpha", clipped_params["c_alpha"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
