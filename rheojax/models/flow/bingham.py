"""Bingham model for linear viscoplastic flow.

This module implements the Bingham model, which describes materials that
require a minimum stress (yield stress) to flow, and then exhibit Newtonian
(linear) behavior. This is a special case of the Herschel-Bulkley model
with n=1 (ROTATION test mode).

Theory:
    σ(γ̇) = σ_y + η_p γ̇  for σ > σ_y
    γ̇ = 0                for σ ≤ σ_y

    - σ_y: Yield stress (material flows only when σ > σ_y)
    - η_p: Plastic viscosity (constant, Newtonian behavior above yield)

References:
    - Bingham, E.C. (1922). Fluidity and Plasticity. McGraw-Hill.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from functools import partial

import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, TestMode, detect_test_mode
from rheojax.logging import get_logger, log_fit

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register("bingham", protocols=[Protocol.FLOW_CURVE], deformation_modes=[DeformationMode.SHEAR])
class Bingham(BaseModel):
    """Bingham model for linear viscoplastic flow (ROTATION only).

    The Bingham model describes materials that require a minimum stress
    (yield stress σ_y) to flow, and then exhibit constant (Newtonian)
    viscosity. This is the simplest viscoplastic model, used for materials
    like toothpaste, mayonnaise, and drilling mud.

    Parameters:
        sigma_y: Yield stress (Pa), minimum stress required for flow
        eta_p: Plastic viscosity (Pa·s), constant viscosity above yield

    Constitutive Equation:
        σ(γ̇) = σ_y + η_p ``|γ̇|``  for ``|σ|`` > σ_y
        γ̇ = 0                      for ``|σ|`` ≤ σ_y

    Special Cases:
        σ_y = 0: Reduces to Newtonian fluid with η = η_p
        This is a special case of Herschel-Bulkley with n = 1

    Test Mode:
        ROTATION (steady shear) only
    """

    def __init__(self):
        """Initialize Bingham model."""
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
            name="eta_p",
            value=0.1,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Plastic viscosity",
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Bingham:
        """Fit Bingham parameters to data.

        Args:
            X: Shear rate data (gamma_dot)
            y: Stress data (sigma)
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        with log_fit(
            logger,
            self.__class__.__name__,
            data_shape=X.shape,
            test_mode="ROTATION",
        ) as ctx:
            logger.debug(
                "Processing input data",
                gamma_dot_range=(float(X.min()), float(X.max())),
                stress_range=(float(y.min()), float(y.max())),
                n_points=len(X),
            )

            # Sort by shear rate
            sort_idx = np.argsort(X)
            X_sorted = X[sort_idx]
            y_sorted = y[sort_idx]

            # For Bingham model: sigma = sigma_y + eta_p * gamma_dot
            # This is linear, so we can use linear regression
            # However, need to account for yield stress

            # Estimate yield stress from intercept at gamma_dot = 0
            # Use linear fit in high shear rate region
            mid_idx = len(X_sorted) // 2
            logger.debug(
                "Performing linear regression",
                mid_idx=mid_idx,
                fit_region_start=float(X_sorted[mid_idx]),
                fit_region_end=float(X_sorted[-1]),
            )

            try:
                coeffs = np.polyfit(X_sorted[mid_idx:], y_sorted[mid_idx:], 1)
            except Exception as e:
                logger.error(
                    "Linear regression failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

            eta_p_est = coeffs[0]  # Slope = eta_p
            sigma_y_est = coeffs[1]  # Intercept = sigma_y

            logger.debug(
                "Initial parameter estimates from linear fit",
                eta_p_raw=eta_p_est,
                sigma_y_raw=sigma_y_est,
            )

            # Ensure positive values
            if sigma_y_est < 0:
                logger.debug(
                    "Negative yield stress estimated, clipping to 0",
                    sigma_y_raw=sigma_y_est,
                )
                sigma_y_est = 0.0
            if eta_p_est < 0:
                logger.debug(
                    "Negative plastic viscosity estimated, setting to default",
                    eta_p_raw=eta_p_est,
                )
                eta_p_est = 0.1

            # Clip to bounds
            sigma_y_est = np.clip(sigma_y_est, 0.0, 1e6)
            eta_p_est = np.clip(eta_p_est, 1e-6, 1e12)

            self.parameters.set_value("sigma_y", float(sigma_y_est))
            self.parameters.set_value("eta_p", float(eta_p_est))

            # Log fitted parameters
            ctx["sigma_y"] = float(sigma_y_est)
            ctx["eta_p"] = float(eta_p_est)

            logger.debug(
                "Fitting completed successfully",
                sigma_y=float(sigma_y_est),
                eta_p=float(eta_p_est),
            )

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict stress for given shear rates.

        Args:
            X: Shear rate data (γ̇)

        Returns:
            Predicted stress σ(γ̇)
        """
        sigma_y = self.parameters.get_value("sigma_y")
        eta_p = self.parameters.get_value("eta_p")

        # Convert to JAX for computation
        gamma_dot = jnp.array(X)

        # Compute stress
        stress = self._predict_stress(gamma_dot, sigma_y, eta_p)

        # Convert back to numpy
        return np.array(stress)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (shear rate γ̇)
            params: Array of parameter values [sigma_y, eta_p]

        Returns:
            Model predictions as JAX array (shear stress σ)
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        sigma_y = params[0]
        eta_p = params[1]

        # Bingham model only supports ROTATION test mode
        # Compute stress using the internal JAX method
        return self._predict_stress(X, sigma_y, eta_p)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_stress(
        self,
        gamma_dot: jnp.ndarray,
        sigma_y: float,
        eta_p: float,
        threshold: float = 1e-9,
    ) -> jnp.ndarray:
        """Compute shear stress using Bingham model.

        Args:
            gamma_dot: Shear rate (s^-1)
            sigma_y: Yield stress (Pa)
            eta_p: Plastic viscosity (Pa·s)
            threshold: Threshold shear rate for yield (default: 1e-9)

        Returns:
            Shear stress (Pa)
        """
        # σ(γ̇) = σ_y + η_p |γ̇| for |γ̇| > threshold
        # σ(γ̇) = 0 for |γ̇| ≤ threshold (below yield)
        abs_gamma_dot = jnp.abs(gamma_dot)

        # Compute stress above yield
        stress_above_yield = sigma_y + eta_p * abs_gamma_dot

        # Apply yield condition using jnp.where
        return jnp.where(abs_gamma_dot > threshold, stress_above_yield, 0.0)

    @partial(jax.jit, static_argnums=(0,))
    def _predict_viscosity(
        self,
        gamma_dot: jnp.ndarray,
        sigma_y: float,
        eta_p: float,
        threshold: float = 1e-9,
    ) -> jnp.ndarray:
        """Compute apparent viscosity using Bingham model.

        Args:
            gamma_dot: Shear rate (s^-1)
            sigma_y: Yield stress (Pa)
            eta_p: Plastic viscosity (Pa·s)
            threshold: Threshold shear rate for yield (default: 1e-9)

        Returns:
            Apparent viscosity (Pa·s)
        """
        # η_app(γ̇) = σ(γ̇) / γ̇ = σ_y/|γ̇| + η_p
        abs_gamma_dot = jnp.abs(gamma_dot)

        # Compute viscosity above yield
        viscosity_above_yield = sigma_y / (abs_gamma_dot + threshold) + eta_p

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
        eta_p = self.parameters.get_value("eta_p")

        # Convert to JAX for computation
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute viscosity
        viscosity = self._predict_viscosity(gamma_dot_jax, sigma_y, eta_p)

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
                f"Bingham model only supports ROTATION test mode, got {test_mode}"
            )

        # Get shear rate data
        gamma_dot = rheo_data.x

        # Get parameters
        sigma_y = self.parameters.get_value("sigma_y")
        eta_p = self.parameters.get_value("eta_p")

        # Convert to JAX
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute prediction based on output type
        if output == "stress":
            y_pred = self._predict_stress(gamma_dot_jax, sigma_y, eta_p)
            y_units = "Pa"
        elif output == "viscosity":
            y_pred = self._predict_viscosity(gamma_dot_jax, sigma_y, eta_p)
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
                "model": "Bingham",
                "test_mode": TestMode.ROTATION,
                "output": output,
                "sigma_y": sigma_y,
                "eta_p": eta_p,
            },
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        sigma_y = self.parameters.get_value("sigma_y")
        eta_p = self.parameters.get_value("eta_p")
        return f"Bingham(sigma_y={sigma_y:.3e}, eta_p={eta_p:.3e})"


__all__ = ["Bingham"]
