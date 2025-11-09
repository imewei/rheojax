"""Carreau-Yasuda model for non-Newtonian flow.

This module implements the Carreau-Yasuda model, an extension of the Carreau
model with an additional parameter 'a' for better control over the transition
region between Newtonian and power-law behavior (ROTATION test mode).

Theory:
    η(γ̇) = η_∞ + (η_0 - η_∞) [1 + (λγ̇)^a]^((n-1)/a)

    - η_0: Zero-shear viscosity (Newtonian plateau at low γ̇)
    - η_∞: Infinite-shear viscosity (Newtonian plateau at high γ̇)
    - λ: Time constant (characteristic relaxation time)
    - n: Power-law index (controls asymptotic behavior)
    - a: Transition parameter (controls transition width/smoothness)

References:
    - Yasuda, K., et al. (1981). Rheol. Acta 20, 163-178.
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


@ModelRegistry.register("carreau_yasuda")
class CarreauYasuda(BaseModel):
    """Carreau-Yasuda model for non-Newtonian flow (ROTATION only).

    The Carreau-Yasuda model extends the Carreau model with an additional
    parameter 'a' that provides better control over the transition region
    between Newtonian and power-law behavior. When a=2, it reduces to the
    standard Carreau model.

    Parameters:
        eta0: Zero-shear viscosity (Pa·s), Newtonian plateau at low γ̇
        eta_inf: Infinite-shear viscosity (Pa·s), Newtonian plateau at high γ̇
        lambda_: Time constant (s), characteristic relaxation time
        n: Power-law index (dimensionless), controls asymptotic behavior
        a: Transition parameter (dimensionless), controls transition width

    Constitutive Equation:
        η(γ̇) = η_∞ + (η_0 - η_∞) [1 + (``λ`` γ̇)^a]^((n-1)/a)

    Special Cases:
        a = 2: Reduces to Carreau model
        ``λ`` → 0: Newtonian fluid with η = η_0
        n = 1: Newtonian fluid for all shear rates

    Test Mode:
        ROTATION (steady shear) only
    """

    def __init__(self):
        """Initialize Carreau-Yasuda model."""
        super().__init__()
        self.parameters = ParameterSet()
        self.parameters.add(
            name="eta0",
            value=1000.0,
            bounds=(1e-3, 1e12),
            units="Pa·s",
            description="Zero-shear viscosity",
        )
        self.parameters.add(
            name="eta_inf",
            value=0.001,
            bounds=(1e-6, 1e6),
            units="Pa·s",
            description="Infinite-shear viscosity",
        )
        self.parameters.add(
            name="lambda_",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s",
            description="Time constant",
        )
        self.parameters.add(
            name="n",
            value=0.5,
            bounds=(0.01, 1.0),
            units="dimensionless",
            description="Power-law index",
        )
        self.parameters.add(
            name="a",
            value=2.0,
            bounds=(0.1, 2.0),
            units="dimensionless",
            description="Transition parameter",
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> CarreauYasuda:
        """Fit Carreau-Yasuda parameters to data.

        Args:
            X: Shear rate data (γ̇)
            y: Viscosity data
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        # Simple heuristic fitting (similar to Carreau)
        # Sort by shear rate
        sort_idx = np.argsort(X)
        X_sorted = X[sort_idx]
        y_sorted = y[sort_idx]

        # Estimate plateaus
        eta0_est = np.max(y_sorted[: len(y_sorted) // 10 + 1])
        eta_inf_est = np.min(y_sorted[-len(y_sorted) // 10 :])

        # Find characteristic shear rate (midpoint)
        eta_mid = (eta0_est + eta_inf_est) / 2.0
        idx_mid = np.argmin(np.abs(y_sorted - eta_mid))
        lambda_est = 1.0 / X_sorted[idx_mid] if X_sorted[idx_mid] > 0 else 1.0

        # Estimate n from power-law region slope
        mid_start = len(X_sorted) // 3
        mid_end = 2 * len(X_sorted) // 3
        if mid_end > mid_start + 1:
            log_gamma = np.log(X_sorted[mid_start:mid_end])
            log_eta = np.log(y_sorted[mid_start:mid_end])
            coeffs = np.polyfit(log_gamma, log_eta, 1)
            n_est = coeffs[0] + 1.0
        else:
            n_est = 0.5

        # Default a to 2.0 (Carreau model)
        a_est = 2.0

        # Clip to bounds
        eta0_est = np.clip(eta0_est, 1e-3, 1e12)
        eta_inf_est = np.clip(eta_inf_est, 1e-6, 1e6)
        lambda_est = np.clip(lambda_est, 1e-6, 1e6)
        n_est = np.clip(n_est, 0.01, 1.0)
        a_est = np.clip(a_est, 0.1, 2.0)

        # Ensure eta0 > eta_inf
        if eta0_est <= eta_inf_est:
            eta0_est = eta_inf_est * 10.0

        self.parameters.set_value("eta0", float(eta0_est))
        self.parameters.set_value("eta_inf", float(eta_inf_est))
        self.parameters.set_value("lambda_", float(lambda_est))
        self.parameters.set_value("n", float(n_est))
        self.parameters.set_value("a", float(a_est))

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict viscosity for given shear rates.

        Args:
            X: Shear rate data (γ̇)

        Returns:
            Predicted viscosity η(γ̇)
        """
        eta0 = self.parameters.get_value("eta0")
        eta_inf = self.parameters.get_value("eta_inf")
        lambda_ = self.parameters.get_value("lambda_")
        n = self.parameters.get_value("n")
        a = self.parameters.get_value("a")

        # Convert to JAX for computation
        gamma_dot = jnp.array(X)

        # Compute viscosity
        viscosity = self._predict_viscosity(gamma_dot, eta0, eta_inf, lambda_, n, a)

        # Convert back to numpy
        return np.array(viscosity)

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (shear rate γ̇)
            params: Array of parameter values [eta0, eta_inf, ``lambda_``, n, a]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        eta0 = params[0]
        eta_inf = params[1]
        lambda_ = params[2]
        n = params[3]
        a = params[4]

        # Flow model only supports ROTATION test mode
        # Compute prediction using the internal JAX method
        return self._predict_viscosity(X, eta0, eta_inf, lambda_, n, a)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_viscosity(
        self,
        gamma_dot: jnp.ndarray,
        eta0: float,
        eta_inf: float,
        lambda_: float,
        n: float,
        a: float,
    ) -> jnp.ndarray:
        """Compute viscosity using Carreau-Yasuda model.

        Args:
            gamma_dot: Shear rate (s^-1)
            eta0: Zero-shear viscosity (Pa·s)
            eta_inf: Infinite-shear viscosity (Pa·s)
            lambda_: Time constant (s)
            n: Power-law index
            a: Transition parameter

        Returns:
            Viscosity (Pa·s)
        """
        # η(γ̇) = η_∞ + (η_0 - η_∞) [1 + (λγ̇)^a]^((n-1)/a)
        lambda_gamma = lambda_ * jnp.abs(gamma_dot)
        factor = jnp.power(1.0 + jnp.power(lambda_gamma, a), (n - 1.0) / a)
        return eta_inf + (eta0 - eta_inf) * factor

    @partial(jax.jit, static_argnums=(0,))
    def _predict_stress(
        self,
        gamma_dot: jnp.ndarray,
        eta0: float,
        eta_inf: float,
        lambda_: float,
        n: float,
        a: float,
    ) -> jnp.ndarray:
        """Compute shear stress using Carreau-Yasuda model.

        Args:
            gamma_dot: Shear rate (s^-1)
            eta0: Zero-shear viscosity (Pa·s)
            eta_inf: Infinite-shear viscosity (Pa·s)
            lambda_: Time constant (s)
            n: Power-law index
            a: Transition parameter

        Returns:
            Shear stress (Pa)
        """
        # σ(γ̇) = η(γ̇) * γ̇
        viscosity = self._predict_viscosity(gamma_dot, eta0, eta_inf, lambda_, n, a)
        return viscosity * jnp.abs(gamma_dot)

    def predict_stress(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict shear stress for given shear rates.

        Args:
            gamma_dot: Shear rate data (γ̇)

        Returns:
            Predicted shear stress σ(γ̇)
        """
        eta0 = self.parameters.get_value("eta0")
        eta_inf = self.parameters.get_value("eta_inf")
        lambda_ = self.parameters.get_value("lambda_")
        n = self.parameters.get_value("n")
        a = self.parameters.get_value("a")

        # Convert to JAX for computation
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute stress
        stress = self._predict_stress(gamma_dot_jax, eta0, eta_inf, lambda_, n, a)

        # Convert back to numpy
        return np.array(stress)

    def predict_rheo(
        self,
        rheo_data: RheoData,
        test_mode: TestMode | None = None,
        output: str = "viscosity",
    ) -> RheoData:
        """Predict rheological response for RheoData.

        Args:
            rheo_data: Input rheological data
            test_mode: Test mode (must be ROTATION)
            output: Output type ('viscosity' or 'stress')

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
                f"Carreau-Yasuda model only supports ROTATION test mode, got {test_mode}"
            )

        # Get shear rate data
        gamma_dot = rheo_data.x

        # Get parameters
        eta0 = self.parameters.get_value("eta0")
        eta_inf = self.parameters.get_value("eta_inf")
        lambda_ = self.parameters.get_value("lambda_")
        n = self.parameters.get_value("n")
        a = self.parameters.get_value("a")

        # Convert to JAX
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute prediction based on output type
        if output == "viscosity":
            y_pred = self._predict_viscosity(
                gamma_dot_jax, eta0, eta_inf, lambda_, n, a
            )
            y_units = "Pa·s"
        elif output == "stress":
            y_pred = self._predict_stress(gamma_dot_jax, eta0, eta_inf, lambda_, n, a)
            y_units = "Pa"
        else:
            raise ValueError(
                f"Invalid output type: {output}. Must be 'viscosity' or 'stress'"
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
                "model": "CarreauYasuda",
                "test_mode": TestMode.ROTATION,
                "output": output,
                "eta0": eta0,
                "eta_inf": eta_inf,
                "lambda_": lambda_,
                "n": n,
                "a": a,
            },
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        eta0 = self.parameters.get_value("eta0")
        eta_inf = self.parameters.get_value("eta_inf")
        lambda_ = self.parameters.get_value("lambda_")
        n = self.parameters.get_value("n")
        a = self.parameters.get_value("a")
        return (
            f"CarreauYasuda(eta0={eta0:.3e}, eta_inf={eta_inf:.3e}, "
            f"lambda={lambda_:.3e}, n={n:.3f}, a={a:.3f})"
        )


__all__ = ["CarreauYasuda"]
