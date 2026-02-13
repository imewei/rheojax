"""Fractional Maxwell Model (FMM).

This is the most general fractional Maxwell model with two SpringPots in series,
each with independent fractional orders. It provides maximum flexibility in
describing viscoelastic materials with fractional dynamics.

Mathematical Description:
    Relaxation Modulus: G(t) = c_1 t^(-α) E_{1-α}(-(t/τ)^β)
    Complex Modulus: G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

where α and β are independent fractional orders.

Parameters:
    c1 (float): Material constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): First fractional order, bounds [0.0, 1.0]
    beta (float): Second fractional order, bounds [0.0, 1.0]
    tau (float): Relaxation time (s), bounds [1e-6, 1e6]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Schiessel, H., & Blumen, A. (1993). Hierarchical analogues to fractional relaxation
      equations. Journal of Physics A: Mathematical and General, 26(19), 5057.
    - Heymans, N., & Bauwens, J. C. (1994). Fractal rheological models and fractional
      differential equations for viscoelastic behavior. Rheologica Acta, 33(3), 210-219.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_maxwell_model",
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
class FractionalMaxwellModel(BaseModel):
    """Fractional Maxwell Model: Two SpringPots in series with independent orders.

    This is the most general fractional Maxwell model, allowing for complex
    viscoelastic behavior with two independent fractional orders.

    Attributes:
        parameters: ParameterSet with c1, alpha, beta, tau

    Examples:
        >>> from rheojax.models import FractionalMaxwellModel
        >>> from rheojax.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalMaxwellModel()
        >>> model.parameters.set_value('c1', 1e5)
        >>> model.parameters.set_value('alpha', 0.5)
        >>> model.parameters.set_value('beta', 0.7)
        >>> model.parameters.set_value('tau', 1.0)
        >>>
        >>> # Predict relaxation modulus
        >>> t = np.logspace(-3, 3, 50)
        >>> data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        >>> data.metadata['test_mode'] = 'relaxation'
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize Fractional Maxwell Model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name="c1",
            value=1e5,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="Material constant",
        )

        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="dimensionless",
            description="First fractional order",
        )

        self.parameters.add(
            name="beta",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="dimensionless",
            description="Second fractional order",
        )

        self.parameters.add(
            name="tau",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s",
            description="Relaxation time",
        )

        self.fitted_ = False

    @staticmethod
    @jax.jit
    def _predict_relaxation_jax(
        t: jnp.ndarray, c1: float, alpha: float, beta: float, tau: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = c_1 t^(-α) E_{1-α}(-(t/τ)^β)

        Args:
            t: Time array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha and beta to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        beta_safe = jnp.clip(beta, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)
        tau_safe = jnp.maximum(tau, epsilon)

        # Compute argument for Mittag-Leffler function
        z = -((t_safe / tau_safe) ** beta_safe)

        # Compute E_{1-α}(z) (requires concrete alpha)
        ml_alpha = 1.0 - alpha_safe
        ml_value = mittag_leffler_e(z, alpha=ml_alpha)

        # Compute G(t)
        G_t = c1 * (t_safe ** (-alpha_safe)) * ml_value

        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep_jax(
        t: jnp.ndarray, c1: float, alpha: float, beta: float, tau: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        For the general FMM, the creep compliance is obtained through
        the relationship J(s) = 1/[s²G(s)] in Laplace domain.

        Args:
            t: Time array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Creep compliance array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha and beta to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        beta_safe = jnp.clip(beta, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)
        tau_safe = jnp.maximum(tau, epsilon)
        c1_safe = jnp.maximum(c1, epsilon)

        # For general FMM, creep is more complex
        # Approximate using J(t) ≈ (1/c1) t^α E_{α,1+α}((t/τ)^β)
        # Limit the argument to prevent overflow
        z_raw = (t_safe / tau_safe) ** beta_safe
        z = jnp.minimum(z_raw, 50.0)  # Limit argument to prevent ML overflow

        # Compute E_{α,1+α}(z) (requires concrete alpha/beta)
        ml_alpha = alpha_safe
        ml_beta = 1.0 + alpha_safe
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Limit ML value to prevent overflow
        ml_value_safe = jnp.minimum(ml_value, 1e15)

        # Creep compliance
        J_t = (1.0 / c1_safe) * (t_safe**alpha_safe) * ml_value_safe

        # Clip to reasonable range for compliance values
        J_t_clipped = jnp.clip(J_t, epsilon, 1e10)

        # Ensure monotonicity: creep compliance must increase with time
        # Use cumulative maximum to enforce J(t_i) >= J(t_{i-1})
        J_t_monotonic = jnp.maximum.accumulate(J_t_clipped)

        return J_t_monotonic

    def _predict_oscillation_jax(
        self, omega: jnp.ndarray, c1: float, alpha: float, beta: float, tau: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

        Args:
            omega: Angular frequency array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Complex modulus array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha and beta to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        beta_safe = jnp.clip(beta, epsilon, 1.0 - epsilon)

        # Compute safe values
        omega_safe = jnp.maximum(omega, epsilon)
        tau_safe = jnp.maximum(tau, epsilon)

        # (iω)^α = |ω|^α * exp(i α π/2)
        i_omega_alpha = (omega_safe**alpha_safe) * jnp.exp(
            1j * alpha_safe * jnp.pi / 2.0
        )

        # (iωτ)^β = |ωτ|^β * exp(i β π/2)
        omega_tau = omega_safe * tau_safe
        i_omega_tau_beta = (omega_tau**beta_safe) * jnp.exp(
            1j * beta_safe * jnp.pi / 2.0
        )

        # Complex modulus
        G_star = c1 * i_omega_alpha / (1.0 + i_omega_tau_beta)

        return G_star

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellModel:
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

        # Determine data shape for logging
        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger,
            model="FractionalMaxwellModel",
            data_shape=data_shape,
            test_mode=test_mode if isinstance(test_mode, str) else str(test_mode),
        ) as ctx:
            logger.debug(
                "Starting FMM fit",
                n_points=len(X) if hasattr(X, "__len__") else 1,
                test_mode=str(test_mode),
                initial_params=self.parameters.to_dict(),
            )

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == "oscillation":
                try:
                    from rheojax.utils.initialization import (
                        initialize_fractional_maxwell_model,
                    )

                    success = initialize_fractional_maxwell_model(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            initialized_params=self.parameters.to_dict(),
                        )
                except Exception as e:
                    logger.debug(
                        "Smart initialization failed, using defaults",
                        error=str(e),
                    )

            # Smart initialization for creep/relaxation mode
            elif test_mode in ("creep", "relaxation"):
                try:
                    x_np = np.asarray(X) if not isinstance(X, np.ndarray) else X
                    y_np = np.asarray(y) if not isinstance(y, np.ndarray) else y
                    y_real = np.abs(y_np) if np.iscomplexobj(y_np) else y_np

                    # Filter valid data points
                    valid = (x_np > 0) & (y_real > 0) & np.isfinite(y_real)
                    if np.sum(valid) >= 2:
                        x_valid = x_np[valid]
                        y_valid = y_real[valid]

                        # tau: geometric mean of time range (characteristic time)
                        tau_init = np.sqrt(x_valid.min() * x_valid.max())

                        if test_mode == "creep":
                            # For creep: J(t) ~ 1/c1, so c1 ~ 1/J_mid
                            y_mid = np.sqrt(y_valid.min() * y_valid.max())
                            c1_init = 1.0 / y_mid
                        else:
                            # For relaxation: G(t) ~ c1
                            c1_init = np.sqrt(y_valid.min() * y_valid.max())

                        # Clip to parameter bounds
                        c1_bounds = self.parameters.get("c1").bounds
                        tau_bounds = self.parameters.get("tau").bounds
                        c1_init = np.clip(c1_init, c1_bounds[0], c1_bounds[1])
                        tau_init = np.clip(tau_init, tau_bounds[0], tau_bounds[1])

                        self.parameters.set_value("c1", float(c1_init))
                        self.parameters.set_value("tau", float(tau_init))

                        logger.debug(
                            f"Smart initialization applied for {test_mode} mode",
                            c1=c1_init,
                            tau=tau_init,
                        )
                except Exception as e:
                    logger.debug(
                        f"Smart initialization failed for {test_mode} mode, using defaults",
                        error=str(e),
                    )

            # Create objective function with stateless predictions
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                c1, alpha, beta, tau = params[0], params[1], params[2], params[3]

                # Direct prediction based on test mode (stateless, calls _jax methods)
                if test_mode == "relaxation":
                    return self._predict_relaxation_jax(x, c1, alpha, beta, tau)
                elif test_mode == "creep":
                    return self._predict_creep_jax(x, c1, alpha, beta, tau)
                elif test_mode == "oscillation":
                    return self._predict_oscillation_jax(x, c1, alpha, beta, tau)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            logger.debug("Creating least squares objective", normalize=True)
            objective = create_least_squares_objective(
                model_fn, x_data, y_data, normalize=True
            )

            # Optimize using NLSQ (JAX enabled by default)
            logger.debug(
                "Starting NLSQ optimization",
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
                    error=str(e),
                    exc_info=True,
                )
                raise

            # Validate optimization succeeded
            if not result.success:
                logger.error(
                    "Optimization failed",
                    message=result.message,
                    final_params=self.parameters.to_dict(),
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True
            ctx["final_params"] = self.parameters.to_dict()
            ctx["success"] = True
            logger.debug(
                "FMM fit completed successfully",
                final_params=self.parameters.to_dict(),
            )

        return self

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Internal predict implementation.

        Args:
            X: RheoData object or array of x-values
            **kwargs: Additional arguments (test_mode handled via self._test_mode)

        Returns:
            Predicted values
        """
        # Handle RheoData input
        if isinstance(X, RheoData):
            return self.predict_rheodata(X)

        # Handle raw array input
        from rheojax.core.test_modes import TestMode

        x = jnp.asarray(X)
        c1 = self.parameters.get_value("c1")
        alpha = self.parameters.get_value("alpha")
        beta = self.parameters.get_value("beta")
        tau = self.parameters.get_value("tau")

        test_mode = getattr(self, "_test_mode", None) or kwargs.get("test_mode")
        if test_mode in ("oscillation", TestMode.OSCILLATION):
            result = self._predict_oscillation_jax(x, c1, alpha, beta, tau)
        elif test_mode in ("creep", TestMode.CREEP):
            result = self._predict_creep_jax(x, c1, alpha, beta, tau)
        else:
            result = self._predict_relaxation_jax(x, c1, alpha, beta, tau)
        return np.array(result)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        CRITICAL: test_mode is now passed as parameter (NOT read from self._test_mode)
        to ensure correct posteriors in Bayesian inference (v0.4.0 fix).

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c1, alpha, beta, tau]
            test_mode: Explicit test mode for predictions. If None, defaults
                to 'relaxation' for backward compatibility.

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        c1 = params[0]
        alpha = params[1]
        beta = params[2]
        tau = params[3]

        # Use explicit test_mode parameter (closure-captured in fit_bayesian)
        # Fall back to self._test_mode only for backward compatibility
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", "relaxation")

        # Normalize test_mode to string
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        # Dispatch to appropriate prediction method
        if test_mode == "relaxation":
            return self._predict_relaxation_jax(X, c1, alpha, beta, tau)
        elif test_mode == "creep":
            return self._predict_creep_jax(X, c1, alpha, beta, tau)
        elif test_mode == "oscillation":
            return self._predict_oscillation_jax(X, c1, alpha, beta, tau)
        else:
            # Default to relaxation for unknown modes
            return self._predict_relaxation_jax(X, c1, alpha, beta, tau)

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
        c1 = self.parameters.get_value("c1")
        alpha = self.parameters.get_value("alpha")
        beta = self.parameters.get_value("beta")
        tau = self.parameters.get_value("tau")

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            y_pred = self._predict_relaxation_jax(x, c1, alpha, beta, tau)
        elif test_mode == "creep":
            y_pred = self._predict_creep_jax(x, c1, alpha, beta, tau)
        elif test_mode == "oscillation":
            y_pred = self._predict_oscillation_jax(x, c1, alpha, beta, tau)
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

    def predict(self, X, test_mode: str | None = None, **kwargs):
        """Predict response.

        Args:
            X: RheoData object or array of x-values
            test_mode: Test mode for prediction ('relaxation', 'creep', 'oscillation')
                       Required when X is a raw array. If None, defaults to 'relaxation'.
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Predicted values (RheoData if input is RheoData, else array)
        """
        if isinstance(X, RheoData):
            return self.predict_rheodata(X, test_mode=test_mode)
        else:
            # Get parameters
            c1 = self.parameters.get_value("c1")
            alpha = self.parameters.get_value("alpha")
            beta = self.parameters.get_value("beta")
            tau = self.parameters.get_value("tau")
            x = jnp.asarray(X)

            # Normalize test_mode to string
            mode = test_mode or "relaxation"
            if hasattr(mode, "value"):
                mode = mode.value

            # Route to appropriate prediction method based on test_mode
            if mode == "relaxation":
                result = self._predict_relaxation_jax(x, c1, alpha, beta, tau)
            elif mode == "creep":
                result = self._predict_creep_jax(x, c1, alpha, beta, tau)
            elif mode == "oscillation":
                result = self._predict_oscillation_jax(x, c1, alpha, beta, tau)
            else:
                raise ValueError(
                    f"Unknown test mode: {mode}. "
                    f"Must be 'relaxation', 'creep', or 'oscillation'"
                )
            return np.array(result)
