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

from rheojax.utils.initialization.base import BaseInitializer


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
        epsilon = 1e-12

        # Ge: instantaneous modulus from high-frequency plateau
        Ge_init = max(features["high_plateau"], epsilon)

        # Gk: retarded modulus from difference
        Gk_init = max(features["high_plateau"] - features["low_plateau"], epsilon)

        # tau: from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)

        # alpha: from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

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
