"""Initializer for FractionalMaxwellLiquid model from oscillation data.

Model equation:
    G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)

Extraction strategy:
    - Gm: Maxwell modulus from high-frequency plateau
    - tau_alpha: relaxation time from transition frequency
    - alpha: fractional order from slope or default to 0.5
"""

from __future__ import annotations

from rheojax.utils.initialization.base import BaseInitializer


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
        epsilon = 1e-12

        # Gm: Maxwell modulus from high-frequency plateau
        Gm_init = max(features["high_plateau"], epsilon)

        # tau_alpha: relaxation time from transition frequency
        tau_alpha_init = 1.0 / (features["omega_mid"] + epsilon)

        # alpha: fractional order from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

        return {
            "Gm": Gm_init,
            "alpha": alpha_init,
            "tau_alpha": tau_alpha_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FML parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "Gm", clipped_params["Gm"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau_alpha", clipped_params["tau_alpha"])
