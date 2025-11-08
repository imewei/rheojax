"""Initializer for FractionalJeffreys model from oscillation data.

Model equation:
    G*(ω) = η_1(iω) · [1 + (iωτ_2)^α] / [1 + (iωτ_1)^α]
    where τ_2 = (η_2/η_1) · τ_1

Extraction strategy (simplified):
    - eta1, eta2: from high-frequency slope and plateau
    - tau1: from transition frequency
    - alpha: from slope or default to 0.5
"""

from __future__ import annotations

from rheojax.utils.initialization.base import BaseInitializer


class FractionalJeffreysInitializer(BaseInitializer):
    """Smart initialization for FractionalJeffreys from oscillation data."""

    def _estimate_parameters(self, features: dict) -> dict:
        """Estimate FractionalJeffreys parameters from frequency features.

        Parameters
        ----------
        features : dict
            Frequency features with low_plateau, high_plateau, omega_mid, alpha_estimate

        Returns
        -------
        dict
            Estimated parameters: eta1, eta2, alpha, tau1
        """
        epsilon = 1e-12

        # eta1: from high-frequency behavior (G* ~ eta1 * omega at high freq)
        # We need omega values, but we'll use a reasonable estimate
        omega_high = features["omega_mid"] * 10.0  # Approximate high frequency
        eta1_init = max(features["high_plateau"] / (omega_high + epsilon), epsilon)

        # eta2: assume ratio around 0.5
        eta2_init = eta1_init * 0.5

        # tau1: from transition frequency
        tau1_init = 1.0 / (features["omega_mid"] + epsilon)

        # alpha: from slope or default to 0.5
        if 0.01 <= features["alpha_estimate"] <= 0.99:
            alpha_init = features["alpha_estimate"]
        else:
            alpha_init = 0.5

        return {
            "eta1": eta1_init,
            "eta2": eta2_init,
            "alpha": alpha_init,
            "tau1": tau1_init,
        }

    def _set_parameters(self, param_set, clipped_params: dict) -> None:
        """Set FractionalJeffreys parameters in ParameterSet.

        Parameters
        ----------
        param_set : ParameterSet
            ParameterSet to update
        clipped_params : dict
            Clipped parameter values
        """
        self._safe_set_parameter(param_set, "eta1", clipped_params["eta1"])
        self._safe_set_parameter(param_set, "eta2", clipped_params["eta2"])
        self._safe_set_parameter(param_set, "alpha", clipped_params["alpha"])
        self._safe_set_parameter(param_set, "tau1", clipped_params["tau1"])
