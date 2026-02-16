"""Fractional Maxwell Gel (FMG) model.

This model consists of a SpringPot element (power-law viscoelastic element) in series
with a dashpot (Newtonian viscous element). It captures the transition from power-law
viscoelastic behavior to terminal flow.

Mathematical Description:
    Relaxation Modulus: G(t) = c_α t^(-α) E_{1-α,1-α}(-t^(1-α)/τ)
    Complex Modulus: G*(ω) = c_α (iω)^α · (iωτ) / (1 + iωτ)
    Creep Compliance: J(t) = (1/c_α) t^α E_{1+α,1+α}(-(t/τ)^(1-α))

where τ = η / c_α^(1/(1-α)) is a characteristic relaxation time.

Parameters:
    c_alpha (float): Material constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): Power-law exponent, bounds [0.0, 1.0]
    eta (float): Viscosity (Pa·s), bounds [1e-6, 1e12]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Blair, G. S., Veinoglou, B. C., & Caffyn, J. E. (1947). Limitations of the Newtonian
      time scale in relation to non-equilibrium rheological states and a theory of
      quasi-properties. Proc. R. Soc. Lond. A, 189(1016), 69-87.
    - Friedrich, C., & Braun, H. (1992). Generalized Cole-Cole behavior and its rheological
      relevance. Rheologica Acta, 31(4), 309-322.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger, log_fit
from rheojax.utils.mittag_leffler import mittag_leffler_e2

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_maxwell_gel",
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
class FractionalMaxwellGel(BaseModel):
    """Fractional Maxwell Gel model: SpringPot in series with dashpot.

    This model describes the rheology of materials transitioning from power-law
    viscoelastic behavior to terminal flow, such as polymer solutions and gels.

    Attributes:
        parameters: ParameterSet with c_alpha, alpha, eta

    Examples:
        >>> from rheojax.models import FractionalMaxwellGel
        >>> from rheojax.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalMaxwellGel()
        >>> model.parameters.set_value('c_alpha', 1e5)
        >>> model.parameters.set_value('alpha', 0.5)
        >>> model.parameters.set_value('eta', 1e3)
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
        """Initialize Fractional Maxwell Gel model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name="c_alpha",
            value=10.0,  # Chosen to keep tau numerically stable across alpha ∈ [0,1]
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="SpringPot material constant",
        )

        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="dimensionless",
            description="Power-law exponent",
        )

        self.parameters.add(
            name="eta",
            value=1e4,  # Chosen to keep tau~O(1) for alpha=0.5 with c_alpha=100
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Dashpot viscosity",
        )

        self.fitted_ = False

    def bayesian_nuts_kwargs(self) -> dict:
        """Prefer conservative NUTS settings for the stiff Mittag-Leffler kernel."""
        return {"target_accept_prob": 0.99}

    def _compute_tau(self, c_alpha: float, alpha: float) -> float:
        """Compute characteristic relaxation time.

        Args:
            c_alpha: SpringPot constant
            alpha: Power-law exponent

        Returns:
            Characteristic time τ = (η / c_α)^(1/(1-α))
        """
        eta = self.parameters.get_value("eta")
        # Add small epsilon to prevent division by zero
        epsilon = 1e-12

        try:
            # Check for alpha close to 1
            if alpha > 1.0 - 1e-6:
                return float("inf")

            # Use algebraic simplification to avoid overflow
            # tau^(1-alpha) = eta / c_alpha
            # tau = (eta / c_alpha)^(1/(1-alpha))

            assert eta is not None and c_alpha is not None
            exponent = 1.0 / (1.0 - alpha + epsilon)
            base = eta / c_alpha

            # Check for potential overflow before computing
            if base > 1.0 and exponent > 700:  # approx limit for exp(709)
                return float("inf")

            return base**exponent
        except (OverflowError, ZeroDivisionError):
            return float("inf")

    @staticmethod
    @jax.jit
    def _predict_relaxation_jax(
        t: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = c_α t^(-α) E_{1-α,1-α}(-t^(1-α)/τ)
        """
        # Add small epsilon to prevent issues at t=0 and with alpha=1
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)

        # Compute argument for Mittag-Leffler function
        # z = - (t/τ)^(1-α)
        # Using algebraic simplification to avoid overflow in tau calculation:
        # tau = (eta/c_alpha)^(1/(1-alpha))
        # z = - (t * (c_alpha/eta)^(1/(1-alpha)))^(1-alpha)
        # z = - t^(1-alpha) * (c_alpha/eta)
        beta_safe = 1.0 - alpha_safe
        z = -jnp.power(t_safe, beta_safe) * (c_alpha / eta)

        # Compute E_{1-α,1-α}(z)
        ml_value = mittag_leffler_e2(z, alpha=beta_safe, beta=beta_safe)

        # Compute G(t) = c_α * t^(-α) * E(...)
        G_t = c_alpha * jnp.power(t_safe, -alpha_safe) * ml_value

        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep_jax(
        t: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/c_α) t^α E_{1+α,1+α}(-(t/τ)^(1-α))
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)

        # Compute argument for Mittag-Leffler function
        # z = - (t/τ)^(1-α) = - t^(1-alpha) * (c_alpha/eta)
        beta_exp = 1.0 - alpha_safe
        z = -jnp.power(t_safe, beta_exp) * (c_alpha / eta)

        # Compute E_{1+α,1+α}(z)
        ml_alpha = 1.0 + alpha_safe
        ml_beta = 1.0 + alpha_safe
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Compute J(t)
        # J(t) = (1/c_alpha) * t^alpha * E(...)
        J_t = (1.0 / c_alpha) * jnp.power(t_safe, alpha_safe) * ml_value

        # Ensure monotonicity: creep compliance must increase with time
        # Use cumulative maximum to enforce J(t_i) >= J(t_{i-1})
        J_t_monotonic = jnp.maximum.accumulate(J_t)

        return J_t_monotonic

    @staticmethod
    @jax.jit
    def _predict_oscillation_jax(
        omega: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute beta for the denominator
        beta_safe = 1.0 - alpha_safe

        # Compute safe values
        omega_safe = jnp.maximum(omega, epsilon)

        # (iω)^α = |ω|^α * exp(i α π/2)
        omega_alpha = jnp.power(omega_safe, alpha_safe)
        phase_alpha = jnp.pi * alpha_safe / 2.0
        i_omega_alpha = omega_alpha * (jnp.cos(phase_alpha) + 1j * jnp.sin(phase_alpha))

        # (iωτ)^(1-α) = (iω)^(1-α) * τ^(1-α)
        # τ^(1-α) = [(eta/c_alpha)^(1/(1-alpha))]^(1-alpha) = eta/c_alpha
        # So term is (iω)^(1-α) * (eta/c_alpha)
        omega_beta = jnp.power(omega_safe, beta_safe)
        phase_beta = jnp.pi * beta_safe / 2.0
        i_omega_beta = omega_beta * (jnp.cos(phase_beta) + 1j * jnp.sin(phase_beta))
        denominator_term = i_omega_beta * (eta / c_alpha)

        # Complex modulus: G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))
        # G* = c_alpha * i_omega_alpha / (1 + i_omega_beta * eta/c_alpha)
        #    = i_omega_alpha / (1/c_alpha + i_omega_beta * eta/c_alpha^2) ? No.

        G_star = c_alpha * i_omega_alpha / (1.0 + denominator_term)

        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellGel:
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

        with log_fit(logger, model="FractionalMaxwellGel", data_shape=X.shape) as ctx:
            try:
                logger.info(
                    "Starting Fractional Maxwell Gel model fit",
                    test_mode=test_mode,
                    n_points=len(X),
                )

                logger.debug(
                    "Input data statistics",
                    x_range=(float(np.min(np.abs(X))), float(np.max(np.abs(X)))),
                    y_range=(float(np.min(np.abs(y))), float(np.max(np.abs(y)))),
                )

                ctx["test_mode"] = test_mode

                # Smart initialization for oscillation mode (Issue #9)
                if test_mode == "oscillation":
                    try:
                        from rheojax.utils.initialization import (
                            initialize_fractional_maxwell_gel,
                        )

                        success = initialize_fractional_maxwell_gel(
                            np.array(X), np.array(y), self.parameters
                        )
                        if success:
                            logger.debug(
                                "Smart initialization applied from frequency-domain features",
                                c_alpha=self.parameters.get_value("c_alpha"),
                                alpha=self.parameters.get_value("alpha"),
                                eta=self.parameters.get_value("eta"),
                            )
                    except Exception as e:
                        # Silent fallback to defaults - don't break if initialization fails
                        logger.debug(
                            "Smart initialization failed, using defaults",
                            error=str(e),
                        )

                # Create objective function with stateless predictions
                def model_fn(x, params):
                    """Model function for optimization (stateless)."""
                    c_alpha, alpha, eta = params[0], params[1], params[2]

                    # Direct prediction based on test mode (stateless, calls _jax methods)
                    if test_mode == "relaxation":
                        return self._predict_relaxation_jax(x, c_alpha, alpha, eta)
                    elif test_mode == "creep":
                        return self._predict_creep_jax(x, c_alpha, alpha, eta)
                    elif test_mode == "oscillation":
                        return self._predict_oscillation_jax(x, c_alpha, alpha, eta)
                    else:
                        raise ValueError(f"Unsupported test mode: {test_mode}")

                objective = create_least_squares_objective(
                    model_fn, x_data, y_data, normalize=True
                )

                logger.debug(
                    "Running NLSQ optimization",
                    use_jax=kwargs.get("use_jax", True),
                    method=kwargs.get("method", "auto"),
                    max_iter=kwargs.get("max_iter", 1000),
                )

                # Optimize using NLSQ (JAX enabled by default)
                result = nlsq_optimize(
                    objective,
                    self.parameters,
                    use_jax=kwargs.get("use_jax", True),
                    method=kwargs.get("method", "auto"),
                    max_iter=kwargs.get("max_iter", 1000),
                )

                # Validate optimization succeeded
                if not result.success:
                    logger.error(
                        "Optimization failed",
                        message=result.message,
                        n_iterations=getattr(result, "nfev", None),
                    )
                    raise RuntimeError(
                        f"Optimization failed: {result.message}. "
                        f"Try adjusting initial values, bounds, or max_iter."
                    )

                # Log final parameters
                c_alpha_val = self.parameters.get_value("c_alpha")
                alpha_val = self.parameters.get_value("alpha")
                eta_val = self.parameters.get_value("eta")
                assert c_alpha_val is not None and alpha_val is not None
                tau_val = self._compute_tau(c_alpha_val, alpha_val)

                ctx["c_alpha"] = c_alpha_val
                ctx["alpha"] = alpha_val
                ctx["eta"] = eta_val
                ctx["tau"] = tau_val
                ctx["cost"] = float(result.fun) if hasattr(result, "fun") else None

                logger.info(
                    "Fractional Maxwell Gel model fit completed",
                    c_alpha=c_alpha_val,
                    alpha=alpha_val,
                    eta=eta_val,
                    tau=tau_val,
                    cost=ctx["cost"],
                )

                self.fitted_ = True
                return self

            except Exception as e:
                logger.error(
                    "Fractional Maxwell Gel model fit failed",
                    test_mode=test_mode,
                    error=str(e),
                    exc_info=True,
                )
                raise

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
            return self.predict_rheodata(X)  # type: ignore[return-value]

        # Handle raw array input
        from rheojax.core.test_modes import TestMode

        x = jnp.asarray(X)
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        eta = self.parameters.get_value("eta")

        test_mode = getattr(self, "_test_mode", None) or kwargs.get("test_mode")
        if test_mode in ("oscillation", TestMode.OSCILLATION):
            result = self._predict_oscillation_jax(x, c_alpha, alpha, eta)
        elif test_mode in ("creep", TestMode.CREEP):
            result = self._predict_creep_jax(x, c_alpha, alpha, eta)
        else:
            result = self._predict_relaxation_jax(x, c_alpha, alpha, eta)
        return np.array(result)

    def model_function(self, X, params, test_mode=None, **kwargs):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c_alpha, alpha, eta]

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
            return self._predict_oscillation_jax(X, **params_dict)
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
        if not isinstance(test_mode, str) or not test_mode:
            test_mode = rheo_data.test_mode

        # Get parameters
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        eta = self.parameters.get_value("eta")

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            y_pred = self._predict_relaxation_jax(x, c_alpha, alpha, eta)
        elif test_mode == "creep":
            y_pred = self._predict_creep_jax(x, c_alpha, alpha, eta)
        elif test_mode == "oscillation":
            y_pred_stacked = self._predict_oscillation_jax(x, c_alpha, alpha, eta)
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

    def predict(self, X, test_mode: str | None = None, **kwargs):  # type: ignore[override]
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
            c_alpha = self.parameters.get_value("c_alpha")
            alpha = self.parameters.get_value("alpha")
            eta = self.parameters.get_value("eta")
            x = jnp.asarray(X)

            # Normalize test_mode to string
            mode = test_mode or "relaxation"
            if hasattr(mode, "value"):
                mode = mode.value

            # Route to appropriate prediction method based on test_mode
            if mode == "relaxation":
                result = self._predict_relaxation_jax(x, c_alpha, alpha, eta)
            elif mode == "creep":
                result = self._predict_creep_jax(x, c_alpha, alpha, eta)
            elif mode == "oscillation":
                result = self._predict_oscillation_jax(x, c_alpha, alpha, eta)
            else:
                raise ValueError(
                    f"Unknown test mode: {mode}. "
                    f"Must be 'relaxation', 'creep', or 'oscillation'"
                )
            return np.array(result)
