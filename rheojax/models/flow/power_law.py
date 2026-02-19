"""Power Law model for non-Newtonian flow.

This module implements the Power Law (Ostwald-de Waele) model for shear-thinning
and shear-thickening fluids in steady shear (ROTATION test mode).

Theory:
    Viscosity: η(γ̇) = K γ̇^(n-1)
    Shear stress: σ(γ̇) = K γ̇^n
    - n < 1: shear-thinning behavior
    - n > 1: shear-thickening behavior
    - n = 1: Newtonian fluid (reduces to η = K)

References:
    - Ostwald, W. (1925). Kolloid-Z. 36, 99-117.
    - de Waele, A. (1923). J. Oil Colour Chem. Assoc. 6, 33-88.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, TestMode, detect_test_mode
from rheojax.logging import get_logger, log_fit

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "power_law",
    protocols=[Protocol.FLOW_CURVE],
    deformation_modes=[DeformationMode.SHEAR],
)
class PowerLaw(BaseModel):
    """Power Law model for non-Newtonian flow (ROTATION only).

    The Power Law (Ostwald-de Waele) model is a simple empirical relationship
    that describes shear-thinning (n < 1) and shear-thickening (n > 1) behavior
    in steady shear flow.

    Parameters:
        K: Consistency index (Pa·s^n), controls viscosity magnitude
        n: Flow behavior index (dimensionless), controls shear-thinning/thickening

    Constitutive Equation:
        σ(γ̇) = K ``|γ̇|`` ^n
        η(γ̇) = K ``|γ̇|`` ^(n-1)

    Special Cases:
        n = 1: Newtonian fluid with viscosity η = K
        n < 1: Shear-thinning (viscosity decreases with shear rate)
        n > 1: Shear-thickening (viscosity increases with shear rate)

    Test Mode:
        ROTATION (steady shear) only
    """

    def __init__(self):
        """Initialize Power Law model."""
        super().__init__()
        self.parameters = ParameterSet()
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

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> PowerLaw:
        """Fit Power Law parameters to data.

        Args:
            X: Shear rate data (γ̇)
            y: Viscosity or stress data
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger, model="PowerLaw", data_shape=data_shape, test_mode="rotation"
        ) as ctx:
            logger.debug(
                "Starting Power Law model fit",
                n_points=data_shape[0] if data_shape else None,
                initial_K=self.parameters.get_value("K"),
                initial_n=self.parameters.get_value("n"),
            )

            try:
                # Use log-log linear regression for initial guess
                # log(η) = log(K) + (n-1)*log(γ̇) for viscosity
                # log(σ) = log(K) + n*log(γ̇) for stress

                # Assume viscosity data by default
                log_gamma_dot = np.log(np.maximum(np.abs(X), 1e-30))
                log_y = np.log(np.maximum(np.abs(y), 1e-30))

                logger.debug("Performing log-log linear regression")

                # Linear fit: log(y) = a + b*log(γ̇)
                coeffs = np.polyfit(log_gamma_dot, log_y, 1)

                # For viscosity: b = n-1, a = log(K)
                # Assume viscosity data (can be refined with metadata)
                n_fit = coeffs[0] + 1.0
                K_fit = np.exp(coeffs[1])

                logger.debug(
                    "Log-log regression results",
                    slope=coeffs[0],
                    intercept=coeffs[1],
                    n_raw=n_fit,
                    K_raw=K_fit,
                )

                # Clip to bounds
                n_fit = np.clip(n_fit, 0.01, 2.0)
                K_fit = np.clip(K_fit, 1e-6, 1e6)

                self.parameters.set_value("K", float(K_fit))
                self.parameters.set_value("n", float(n_fit))

                logger.debug(
                    "Power Law fit completed successfully",
                    fitted_K=float(K_fit),
                    fitted_n=float(n_fit),
                    behavior=(
                        "shear-thinning"
                        if n_fit < 1
                        else ("shear-thickening" if n_fit > 1 else "Newtonian")
                    ),
                )

                ctx["K"] = float(K_fit)
                ctx["n"] = float(n_fit)

            except Exception as e:
                logger.error(
                    "Power Law fit failed",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:  # type: ignore[override]
        """Predict viscosity for given shear rates.

        Args:
            X: Shear rate data (γ̇)

        Returns:
            Predicted viscosity η(γ̇) = K |γ̇|^(n-1)
        """
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX for computation
        gamma_dot = jnp.array(X)

        # Compute viscosity
        viscosity = self._predict_viscosity(gamma_dot, K, n)

        # Convert back to numpy
        return np.array(viscosity)

    def model_function(self, X, params, test_mode=None, **kwargs):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (shear rate γ̇)
            params: Array of parameter values [K, n]

        Returns:
            Model predictions as JAX array (viscosity η)
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        K = params[0]
        n = params[1]

        # Power Law model only supports ROTATION test mode
        # Compute viscosity using the internal JAX method
        return self._predict_viscosity(X, K, n)

    @staticmethod
    @jax.jit
    def _predict_viscosity(gamma_dot: jnp.ndarray, K: float, n: float) -> jnp.ndarray:
        """Compute viscosity: η(γ̇) = K |γ̇|^(n-1).

        Args:
            gamma_dot: Shear rate (s^-1)
            K: Consistency index (Pa·s^n)
            n: Flow behavior index

        Returns:
            Viscosity (Pa·s)
        """
        return K * jnp.power(jnp.maximum(jnp.abs(gamma_dot), 1e-30), n - 1.0)

    @staticmethod
    @jax.jit
    def _predict_stress(gamma_dot: jnp.ndarray, K: float, n: float) -> jnp.ndarray:
        """Compute shear stress: σ(γ̇) = K |γ̇|^n.

        Args:
            gamma_dot: Shear rate (s^-1)
            K: Consistency index (Pa·s^n)
            n: Flow behavior index

        Returns:
            Shear stress (Pa)
        """
        return K * jnp.power(jnp.maximum(jnp.abs(gamma_dot), 1e-30), n)

    def predict_stress(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict shear stress for given shear rates.

        Args:
            gamma_dot: Shear rate data (γ̇)

        Returns:
            Predicted shear stress σ(γ̇) = K ``|γ̇|^n``
        """
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX for computation
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute stress
        stress = self._predict_stress(gamma_dot_jax, K, n)

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
                f"Power Law model only supports ROTATION test mode, got {test_mode}"
            )

        # Get shear rate data
        gamma_dot = rheo_data.x

        # Get parameters
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")

        # Convert to JAX
        gamma_dot_jax = jnp.array(gamma_dot)

        # Compute prediction based on output type
        if output == "viscosity":
            y_pred = self._predict_viscosity(gamma_dot_jax, K, n)
            y_units = "Pa·s"
        elif output == "stress":
            y_pred = self._predict_stress(gamma_dot_jax, K, n)
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
                "model": "PowerLaw",
                "test_mode": TestMode.ROTATION,
                "output": output,
                "K": K,
                "n": n,
            },
            validate=False,
        )

    def __repr__(self) -> str:
        """String representation."""
        K = self.parameters.get_value("K")
        n = self.parameters.get_value("n")
        return f"PowerLaw(K={K:.3e}, n={n:.3f})"


__all__ = ["PowerLaw"]
