"""Fractional Kelvin-Voigt Model (FKV).

This model consists of a spring and a SpringPot element in parallel. It describes
materials with solid-like behavior with power-law creep, typical of filled polymers
and soft solids.

Mathematical Description:
    Relaxation Modulus: G(t) = G_e + c_α t^(-α) / Γ(1-α)
    Complex Modulus: G*(ω) = G_e + c_α (iω)^α
    Creep Compliance: J(t) = (1/G_e) (1 - E_α(-(t/τ_ε)^α))

where τ_ε = (c_α/G_e)^(1/α) is a characteristic retardation time.

Parameters:
    Ge (float): Equilibrium modulus (Pa), bounds [1e-3, 1e9]
    c_alpha (float): SpringPot constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): Fractional order, bounds [0.0, 1.0]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Bagley, R. L., & Torvik, P. J. (1983). A theoretical basis for the application
      of fractional calculus to viscoelasticity. Journal of Rheology, 27(3), 201-210.
    - Makris, N., & Constantinou, M. C. (1991). Fractional-derivative Maxwell model
      for viscous dampers. Journal of Structural Engineering, 117(9), 2708-2724.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


import numpy as np
from jax.scipy.special import gamma as jax_gamma

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger, log_fit
from rheojax.utils.mittag_leffler import mittag_leffler_e

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_kelvin_voigt",
    protocols=[
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class FractionalKelvinVoigt(BaseModel):
    """Fractional Kelvin-Voigt model: Spring and SpringPot in parallel.

    This model describes solid-like materials with power-law creep behavior,
    typical of filled polymers and soft solids.

    Attributes:
        parameters: ParameterSet with Ge, c_alpha, alpha

    Examples:
        >>> from rheojax.models import FractionalKelvinVoigt
        >>> from rheojax.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalKelvinVoigt()
        >>> model.parameters.set_value('Ge', 1e6)
        >>> model.parameters.set_value('c_alpha', 1e4)
        >>> model.parameters.set_value('alpha', 0.5)
        >>>
        >>> # Predict relaxation modulus
        >>> t = np.logspace(-3, 3, 50)
        >>> data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        >>> data.metadata['test_mode'] = 'relaxation'
        >>> G_t = model.predict(data)
        >>>
        >>> # Predict complex modulus
        >>> omega = np.logspace(-2, 2, 50)
        >>> data = RheoData(x=omega, y=np.zeros_like(omega), domain='frequency')
        >>> G_star = model.predict(data)
    """

    def __init__(self):
        """Initialize Fractional Kelvin-Voigt model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name="Ge",
            value=1e6,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Equilibrium modulus",
        )

        self.parameters.add(
            name="c_alpha",
            value=1e4,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="SpringPot constant",
        )

        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="dimensionless",
            description="Fractional order",
        )

        self.fitted_ = False

    def _compute_tau_epsilon(self, Ge: float, c_alpha: float, alpha: float) -> float:
        """Compute characteristic retardation time.

        Args:
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Characteristic time τ_ε = (c_α/G_e)^(1/α)
        """
        epsilon = 1e-12
        alpha_safe = max(alpha, epsilon)
        return (c_alpha / Ge) ** (1.0 / alpha_safe)

    @staticmethod
    @jax.jit
    def _predict_relaxation_jax(
        t: jnp.ndarray, Ge: float, c_alpha: float, alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = G_e + c_α t^(-α) / Γ(1-α)

        Args:
            t: Time array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        t_safe = jnp.maximum(t, epsilon)

        # Elastic part
        G_elastic = Ge

        # Viscous part: c_α t^(-α) / Γ(1-α)
        gamma_term = jax_gamma(1.0 - alpha_safe)
        G_viscous = c_alpha * (t_safe ** (-alpha_safe)) / gamma_term

        # Total relaxation modulus
        G_t = G_elastic + G_viscous

        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep_jax(
        t: jnp.ndarray, Ge: float, c_alpha: float, alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/G_e) (1 - E_α(-(t/τ_ε)^α))

        where τ_ε = (c_α/G_e)^(1/α)

        Args:
            t: Time array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Creep compliance array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha to safe range
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        t_safe = jnp.maximum(t, epsilon)

        # Characteristic retardation time
        tau_epsilon = (c_alpha / Ge) ** (1.0 / alpha_safe)

        # Argument for Mittag-Leffler function
        z = -((t_safe / tau_epsilon) ** alpha_safe)

        # Compute E_α(z) with concrete alpha
        ml_value = mittag_leffler_e(z, alpha=alpha_safe)

        # Creep compliance
        J_t = (1.0 / Ge) * (1.0 - ml_value)

        return J_t

    @staticmethod
    @jax.jit
    def _predict_oscillation_jax(
        omega: jnp.ndarray, Ge: float, c_alpha: float, alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = G_e + c_α (iω)^α

        Args:
            omega: Angular frequency array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Complex modulus array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha to safe range
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        omega_safe = jnp.maximum(omega, epsilon)

        # Elastic part
        G_elastic = Ge

        # SpringPot part: c_α (iω)^α = c_α |ω|^α exp(i α π/2)
        G_springpot = (
            c_alpha * (omega_safe**alpha_safe) * jnp.exp(1j * alpha_safe * jnp.pi / 2.0)
        )

        # Complex modulus
        G_star = G_elastic + G_springpot

        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalKelvinVoigt:
        """Fit model parameters to data.

        Args:
            X: Independent variable (time or frequency)
            y: Dependent variable (modulus or compliance)
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Handle RheoData input
        if isinstance(X, RheoData):
            rheo_data = X
            x_data = jnp.array(rheo_data.x)
            y_data = jnp.array(rheo_data.y)
            test_mode = rheo_data.test_mode
        else:
            x_data = jnp.array(X)
            y_data = jnp.array(y)
            test_mode = kwargs.get("test_mode", "relaxation")

        # Get test mode string for logging
        test_mode_str = (
            test_mode.value if hasattr(test_mode, "value") else str(test_mode)
        )
        data_shape = (int(x_data.shape[0]),) if hasattr(x_data, "shape") else None

        with log_fit(
            logger,
            model="FractionalKelvinVoigt",
            data_shape=data_shape,
            test_mode=test_mode_str,
        ) as ctx:
            logger.debug(
                "Starting Fractional Kelvin-Voigt model fit",
                test_mode=test_mode_str,
                n_points=data_shape[0] if data_shape else None,
                initial_Ge=self.parameters.get_value("Ge"),
                initial_c_alpha=self.parameters.get_value("c_alpha"),
                initial_alpha=self.parameters.get_value("alpha"),
            )

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == "oscillation":
                try:
                    from rheojax.utils.initialization import (
                        initialize_fractional_kelvin_voigt,
                    )

                    success = initialize_fractional_kelvin_voigt(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            Ge=self.parameters.get_value("Ge"),
                            c_alpha=self.parameters.get_value("c_alpha"),
                            alpha=self.parameters.get_value("alpha"),
                        )
                except Exception as e:
                    # Silent fallback to defaults - don't break if initialization fails
                    logger.debug(
                        "Smart initialization failed, using defaults",
                        error_type=type(e).__name__,
                        error_message=str(e),
                    )

            # Create objective function with stateless predictions
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                Ge, c_alpha, alpha = params[0], params[1], params[2]

                # Direct prediction based on test mode (stateless, calls _jax methods)
                if test_mode == "relaxation":
                    return self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
                elif test_mode == "creep":
                    return self._predict_creep_jax(x, Ge, c_alpha, alpha)
                elif test_mode == "oscillation":
                    return self._predict_oscillation_jax(x, Ge, c_alpha, alpha)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            logger.debug("Creating least squares objective", normalize=True)
            objective = create_least_squares_objective(
                model_fn, x_data, y_data, normalize=True
            )

            # Optimize using NLSQ
            logger.debug(
                "Starting NLSQ optimization",
                use_jax=kwargs.get("use_jax", True),
                method=kwargs.get("method", "auto"),
                max_iter=kwargs.get("max_iter", 1000),
            )

            try:
                result = nlsq_optimize(
                    objective,
                    self.parameters,
                    use_jax=kwargs.get("use_jax", True),
                    method=kwargs.get("method", "auto"),
                    max_iter=kwargs.get("max_iter", 1000),
                )
            except Exception as e:
                logger.error(
                    "NLSQ optimization raised exception",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

            # Validate optimization succeeded
            if not result.success:
                logger.error(
                    "Optimization failed",
                    message=result.message,
                    test_mode=test_mode_str,
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True

            # Log fitted parameters
            fitted_Ge = self.parameters.get_value("Ge")
            fitted_c_alpha = self.parameters.get_value("c_alpha")
            fitted_alpha = self.parameters.get_value("alpha")

            # Compute characteristic time
            tau_epsilon = self._compute_tau_epsilon(
                fitted_Ge, fitted_c_alpha, fitted_alpha
            )

            logger.debug(
                "Fractional Kelvin-Voigt fit completed successfully",
                fitted_Ge=fitted_Ge,
                fitted_c_alpha=fitted_c_alpha,
                fitted_alpha=fitted_alpha,
                characteristic_time=tau_epsilon,
            )

            ctx["Ge"] = fitted_Ge
            ctx["c_alpha"] = fitted_c_alpha
            ctx["alpha"] = fitted_alpha

        return self

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation.

        Args:
            X: RheoData object or array of x-values

        Returns:
            Predicted values
        """
        # Handle RheoData input
        if isinstance(X, RheoData):
            return self.predict_rheodata(X)

        # Handle raw array input (assume relaxation mode)
        x = jnp.asarray(X)
        Ge = self.parameters.get_value("Ge")
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")

        result = self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
        return np.array(result)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Ge, c_alpha, alpha]

        Returns:
            Model predictions as JAX array
        """
        # Use explicit test_mode parameter (closure-captured in fit_bayesian)
        # Fall back to self._test_mode only for backward compatibility
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", "relaxation")

        # Normalize test_mode to string
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        # Extract parameter names from function signature
        params_dict = {name: params[i] for i, name in enumerate(self.parameters.keys())}

        # Dispatch to appropriate prediction method
        if test_mode == "relaxation":
            return self._predict_relaxation_jax(X, **params_dict)
        elif test_mode == "creep":
            return self._predict_creep_jax(X, **params_dict)
        elif test_mode == "oscillation":
            # Return complex array for oscillation mode
            complex_result = self._predict_oscillation_jax(X, **params_dict)
            return complex_result[..., 0] + 1j * complex_result[..., 1]
        else:
            # Default to relaxation for unknown modes
            return self._predict_relaxation_jax(X, **params_dict)

    def predict_rheodata(
        self, rheo_data: RheoData, test_mode: str | None = None
    ) -> RheoData:
        """Predict response for RheoData.

        Args:
            rheo_data: Input RheoData with x values
            test_mode: Test mode ('relaxation', 'creep', 'oscillation')
                      If None, auto-detect from rheo_data

        Returns:
            RheoData with predicted y values
        """
        # Auto-detect test mode if not provided
        if test_mode is None:
            # Check for explicit test_mode in metadata first
            if "test_mode" in rheo_data.metadata:
                test_mode = rheo_data.metadata["test_mode"]
            else:
                test_mode = rheo_data.test_mode

        # Get parameters
        Ge = self.parameters.get_value("Ge")
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            y_pred = self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
        elif test_mode == "creep":
            y_pred = self._predict_creep_jax(x, Ge, c_alpha, alpha)
        elif test_mode == "oscillation":
            y_pred_stacked = self._predict_oscillation_jax(x, Ge, c_alpha, alpha)
            y_pred = y_pred_stacked[..., 0] + 1j * y_pred_stacked[..., 1]
        else:
            raise ValueError(
                f"Unknown test mode: {test_mode}. "
                f"Must be 'relaxation', 'creep', or 'oscillation'"
            )

        # Create output RheoData
        result = RheoData(
            x=np.array(x),
            y=np.array(y_pred),
            x_units=rheo_data.x_units,
            y_units=rheo_data.y_units,
            domain=rheo_data.domain,
            metadata=rheo_data.metadata.copy(),
        )

        return result

    def predict(self, X, test_mode: str | None = None):
        """Predict response.

        Args:
            X: RheoData object or array of x-values
            test_mode: Test mode for prediction ('relaxation', 'creep', 'oscillation')
                       Required when X is a raw array. If None, defaults to 'relaxation'.

        Returns:
            Predicted values (RheoData if input is RheoData, else array)
        """
        if isinstance(X, RheoData):
            return self.predict_rheodata(X, test_mode=test_mode)
        else:
            # Get parameters
            Ge = self.parameters.get_value("Ge")
            c_alpha = self.parameters.get_value("c_alpha")
            alpha = self.parameters.get_value("alpha")
            x = jnp.asarray(X)

            # Route to appropriate prediction method based on test_mode
            mode = test_mode or "relaxation"
            if mode == "relaxation":
                result = self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
            elif mode == "creep":
                result = self._predict_creep_jax(x, Ge, c_alpha, alpha)
            elif mode == "oscillation":
                result = self._predict_oscillation_jax(x, Ge, c_alpha, alpha)
            else:
                raise ValueError(
                    f"Unknown test mode: {mode}. "
                    f"Must be 'relaxation', 'creep', or 'oscillation'"
                )
            return np.array(result)
