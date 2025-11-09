"""Herschel-Bulkley model for non-Newtonian flow with yield stress.

This module implements the Herschel-Bulkley model, which combines yield stress
behavior with power-law flow. This is the most general viscoplastic model,
reducing to simpler models as special cases (ROTATION test mode).

Theory:
    σ(γ̇) = σ_y + K ``|γ̇|`` ^n  for σ > σ_y
    γ̇ = 0                     for σ ≤ σ_y

    - σ_y: Yield stress (material flows only when σ > σ_y)
    - K: Consistency index (viscosity-like parameter)
    - n: Flow behavior index (power-law exponent)

References:
    - Herschel, W.H., Bulkley, R. (1926). Proc. ASTM 26, 621-633.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from functools import partial

import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode


@ModelRegistry.register("herschel_bulkley")
class HerschelBulkley(BaseModel):
    """Herschel-Bulkley model for viscoplastic flow (ROTATION only).

    The Herschel-Bulkley model describes materials that require a minimum
    stress (yield stress σ_y) to flow, and then exhibit power-law behavior.
    This is widely used for pastes, slurries, and suspensions.

    Parameters:
        sigma_y: Yield stress (Pa), minimum stress required for flow
        K: Consistency index (Pa·s^n), controls viscosity magnitude
        n: Flow behavior index (dimensionless), power-law exponent

    Constitutive Equation:
        σ(γ̇) = σ_y + K ``|γ̇|`` ^n  for ``|σ|`` > σ_y
        γ̇ = 0                     for ``|σ|`` ≤ σ_y

    Special Cases:
        σ_y = 0: Reduces to Power Law model
        n = 1: Reduces to Bingham model (linear viscoplastic)
        σ_y = 0, n = 1: Newtonian fluid

    Test Mode:
        ROTATION (steady shear) only
    """

    def __init__(self):
        """Initialize Herschel-Bulkley model."""
        super().__init__()
        self.parameters = ParameterSet()
        self.parameters.add(
            name="sigma_y",
            value=10.0,
            bounds=(0.0, 1e6),
            units="Pa",
            description="Yield stress",
        )
        self.parameters.add(
            name="K",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="Pa·s^n",
            description="Consistency index",
        )
        self.parameters.add(
            name="n",
            value=0.5,
            bounds=(0.01, 2.0),
            units="dimensionless",
            description="Flow behavior index",
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> HerschelBulkley:
        """Fit Herschel-Bulkley parameters to data.

        Args:
            X: Shear rate data (γ̇)
            y: Stress data (σ)
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        # Sort by shear rate
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Estimate yield stress from low shear rate extrapolation
        # σ_y ≈ stress at γ̇ → 0
        sigma_y_est = np.min(y_sorted[: len(y_sorted) // 10 + 1])
        if sigma_y_est < 0:
            sigma_y_est = 0.0

        # Subtract yield stress for power-law fitting
        y_corrected = y_sorted - sigma_y_est
        y_corrected = np.maximum(y_corrected, 1e-10)  # Avoid log(0)

        # Fit power law to corrected data: log(σ - σ_y) = log(K) + n*log(γ̇)
        # Use middle to high shear rate region
        start_idx = len(X_sorted) // 4
        log_gamma = np.log(np.abs(X_sorted[start_idx:]))
        log_stress = np.log(y_corrected[start_idx:])

        coeffs = np.polyfit(log_gamma, log_stress, 1)
        n_est = coeffs[0]
        K_est = np.exp(coeffs[1])

        # Clip to bounds
        sigma_y_est = np.clip(sigma_y_est, 0.0, 1e6)
        K_est = np.clip(K_est, 1e-6, 1e6)
        n_est = np.clip(n_est, 0.01, 2.0)

        self.parameters.set_value("sigma_y", float(sigma_y_est))
        self.parameters.set_value("K", float(K_est))
        self.parameters.set_value("n", float(n_est))

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress for given shear rates.

        Args:
            X: Shear rate data (γ̇)

        Returns:
            Predicted stress σ(γ̇)
        """
        sigma_y = self.parameters.get_value("sigma_y")
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX for computation
        gamma_dot = jnp.array(X)

        # Compute stress
        stress = self._predict_stress(gamma_dot, sigma_y, K, n)

        # Convert back to numpy
        return np.array(stress)

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (shear rate γ̇)
            params: Array of parameter values [sigma_y, K, n]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        sigma_y = params[0]
        K = params[1]
        n = params[2]

        # Flow model only supports ROTATION test mode
        # Compute prediction using the internal JAX method
        return self._predict_stress(X, sigma_y, K, n)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_stress(
        self,
        gamma_dot: jnp.ndarray,
        sigma_y: float,
        K: float,
        n: float,
        threshold: float = 1e-9,
    ) -> jnp.ndarray:
        """Compute shear stress using Herschel-Bulkley model.

        Args:
            gamma_dot: Shear rate (s^-1)
            sigma_y: Yield stress (Pa)
            K: Consistency index (Pa·s^n)
            n: Flow behavior index
            threshold: Threshold shear rate for yield (default: 1e-9)

        Returns:
            Shear stress (Pa)
        """
        # σ(γ̇) = σ_y + K |γ̇|^n for |γ̇| > threshold
        # σ(γ̇) = 0 for |γ̇| ≤ threshold (below yield)
        abs_gamma_dot = jnp.abs(gamma_dot)

        # Compute stress above yield
        stress_above_yield = sigma_y + K * jnp.power(abs_gamma_dot, n)

        # Apply yield condition using jnp.where
        return jnp.where(abs_gamma_dot > threshold, stress_above_yield, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_viscosity(
        self,
        gamma_dot: jnp.ndarray,
        sigma_y: float,
        K: float,
        n: float,
        threshold: float = 1e-9,
    ) -> jnp.ndarray:
        """Compute apparent viscosity using Herschel-Bulkley model.

        Args:
            gamma_dot: Shear rate (s^-1)
            sigma_y: Yield stress (Pa)
            K: Consistency index (Pa·s^n)
            n: Flow behavior index
            threshold: Threshold shear rate for yield (default: 1e-9)

        Returns:
            Apparent viscosity (Pa·s)
        """
        # η_app(γ̇) = σ(γ̇) / γ̇ = σ_y/|γ̇| + K |γ̇|^(n-1)
        abs_gamma_dot = jnp.abs(gamma_dot)

        # Compute viscosity above yield
        viscosity_above_yield = sigma_y / (abs_gamma_dot + threshold) + K * jnp.power(
            abs_gamma_dot, n - 1.0
        )

        # Apply yield condition
        return jnp.where(abs_gamma_dot > threshold, viscosity_above_yield, jnp.inf)

    def predict_viscosity(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict apparent viscosity for given shear rates.

        Args:
            gamma_dot: Shear rate data (γ̇)

        Returns:
            Predicted apparent viscosity η_app(γ̇)
        """
        sigma_y = self.parameters.get_value("sigma_y")
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX for computation
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute viscosity
        viscosity = self._predict_viscosity(gamma_dot_jax, sigma_y, K, n)

        # Convert back to numpy
        return np.array(viscosity)

    def predict_rheo(
        self,
        rheo_data: RheoData,
        test_mode: TestMode | None = None,
        output: str = "stress",
    ) -> RheoData:
        """Predict rheological response for RheoData.

        Args:
            rheo_data: Input rheological data
            test_mode: Test mode (must be ROTATION)
            output: Output type ('stress' or 'viscosity')

        Returns:
            RheoData with predictions

        Raises:
            ValueError: If test mode is not ROTATION
        """
        # Detect test mode if not provided
        if test_mode is None:
            test_mode = detect_test_mode(rheo_data)

        # Validate test mode
        if test_mode != TestMode.ROTATION:
            raise ValueError(
                f"Herschel-Bulkley model only supports ROTATION test mode, got {test_mode}"
            )

        # Get shear rate data
        gamma_dot = rheo_data.x

        # Get parameters
        sigma_y = self.parameters.get_value("sigma_y")
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute prediction based on output type
        if output == "stress":
            y_pred = self._predict_stress(gamma_dot_jax, sigma_y, K, n)
            y_units = "Pa"
        elif output == "viscosity":
            y_pred = self._predict_viscosity(gamma_dot_jax, sigma_y, K, n)
            y_units = "Pa·s"
        else:
            raise ValueError(
                f"Invalid output type: {output}. Must be 'stress' or 'viscosity'"
            )

        # Convert back to numpy
        y_pred = np.array(y_pred)

        # Create output RheoData
        return RheoData(
            x=np.array(gamma_dot),
            y=y_pred,
            x_units=rheo_data.x_units or "1/s",
            y_units=y_units,
            domain="time",
            metadata={
                "model": "HerschelBulkley",
                "test_mode": TestMode.ROTATION,
                "output": output,
                "sigma_y": sigma_y,
                "K": K,
                "n": n,
            },
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        sigma_y = self.parameters.get_value("sigma_y")
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")
        return f"HerschelBulkley(sigma_y={sigma_y:.3e}, K={K:.3e}, n={n:.3f})"


__all__ = ["HerschelBulkley"]
