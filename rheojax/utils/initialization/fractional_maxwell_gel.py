"""Initializer for FractionalMaxwellGel model from oscillation data.

Model equation:
    G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))
    where τ = η / c_α^(1/(1-α))

Extraction strategy:
    - c_alpha: estimated from high-frequency plateau
    - alpha: fractional order from slope or default to 0.5
    - eta: estimated from transition frequency and approximate tau
"""

from __future__ import annotations

from rheojax.utils.initialization.base import BaseInitializer


class FractionalMaxwellGelInitializer(BaseInitializer):
    """Smart initialization for FractionalMaxwellGel from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FMG parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: c_alpha, alpha, eta
        """
        epsilon = 1e-12

        # alpha: fractional order from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

        # c_alpha: approximate from high-frequency behavior
        # At high frequency, G* ~ c_α τ^(α-1)
        # We'll use high plateau as first estimate
        c_alpha_init = max(features["high_plateau"], epsilon)

        # eta: estimate from transition frequency
        # tau ~ 1/omega_mid, and tau = eta / c_alpha^(1/(1-alpha))
        # So eta ~ tau * c_alpha^(1/(1-alpha))
        tau_est = 1.0 / (features["omega_mid"] + epsilon)
        eta_init = tau_est * (c_alpha_init ** (1.0 / (1.0 - alpha_init + epsilon)))

        return {
            "c_alpha": c_alpha_init,
            "alpha": alpha_init,
            "eta": eta_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FMG parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "c_alpha", clipped_params["c_alpha"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "eta", clipped_params["eta"])
