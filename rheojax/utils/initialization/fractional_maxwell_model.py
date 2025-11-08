"""Initializer for FractionalMaxwellModel from oscillation data.

Model equation:
    G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

Extraction strategy:
    - c1: from high-frequency plateau
    - alpha, beta: from slope or defaults to 0.5
    - tau: from transition frequency
"""

from __future__ import annotations

from rheojax.utils.initialization.base import BaseInitializer


class FractionalMaxwellModelInitializer(BaseInitializer):
    """Smart initialization for FractionalMaxwellModel from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalMaxwellModel parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: c1, alpha, beta, tau
        """
        epsilon = 1e-12

        # c1: from high-frequency plateau
        c1_init = max(features["high_plateau"], epsilon)

        # alpha: from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

        # beta: default to 0.5 (optimizer will refine)
        beta_init = 0.5

        # tau: from transition frequency
        tau_init = 1.0 / (features["omega_mid"] + epsilon)

        return {
            "c1": c1_init,
            "alpha": alpha_init,
            "beta": beta_init,
            "tau": tau_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalMaxwellModel parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "c1", clipped_params["c1"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "beta", clipped_params["beta"])
        self._safe_set_parameter(param_set, "tau", clipped_params["tau"])
