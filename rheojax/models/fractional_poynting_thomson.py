"""Fractional Poynting-Thomson (FPT) Model.

This model consists of a Fractional Kelvin-Voigt element in series with a spring,
similar to FKVZ but with different parameter interpretation emphasizing
the instantaneous modulus.

Theory
------
The FPT model consists of:
- Spring (G_e - instantaneous modulus) in series with
- Fractional Kelvin-Voigt element (spring + SpringPot in parallel)

Creep compliance:
    J(t) = 1/G_e + (1/G_k) * (1 - E_α(-(t/τ)^α))

Complex modulus:
    G*(ω) = [1/G_e + (1/G_k)/(1 + (iωτ)^α)]^(-1)

where E_α is the one-parameter Mittag-Leffler function.

Parameters
----------
Ge : float
    Instantaneous modulus (Pa), bounds [1e-3, 1e9]
Gk : float
    Retarded modulus (Pa), bounds [1e-3, 1e9]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau : float
    Retardation time (s), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Two springs in series
- alpha → 1: Classical Poynting-Thomson (standard linear solid)

Note
----
FPT and FKVZ have identical mathematical forms but different physical
interpretations. FPT emphasizes stress relaxation, while FKVZ
emphasizes strain retardation.

References
----------
- Mainardi, F. (2010). Fractional Calculus and Waves in Linear Viscoelasticity
- Poynting, J.H. & Thomson, J.J. (1902). Properties of Matter
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e

logger = get_logger(__name__)


@ModelRegistry.register("fractional_poynting_thomson")
class FractionalPoyntingThomson(BaseModel):
    """Fractional Poynting-Thomson model.

    A fractional viscoelastic model emphasizing instantaneous
    elastic response with fractional retardation.

    Test Modes
    ----------
    - Relaxation: Supported
    - Creep: Supported (primary mode)
    - Oscillation: Supported
    - Rotation: Not supported (no steady-state flow)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalPoyntingThomson
    >>>
    >>> # Create model
    >>> model = FractionalPoyntingThomson()
    >>>
    >>> # Set parameters
    >>> model.set_params(Ge=1500.0, Gk=500.0, alpha=0.5, tau=1.0)
    >>>
    >>> # Predict creep compliance
    >>> t = jnp.logspace(-2, 2, 50)
    >>> J_t = model.predict(t)
    """

    def __init__(self):
        """Initialize Fractional Poynting-Thomson model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="Ge",
            value=1500.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Instantaneous modulus",
        )
        self.parameters.add(
            name="Gk",
            value=500.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Retarded modulus",
        )
        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s",
            description="Retardation time",
        )

    @staticmethod
    @jax.jit
    def _predict_creep(
        t: jnp.ndarray,
        Ge: float,
        Gk: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        J(t) = 1/G_e + (1/G_k) * (1 - E_α(-(t/τ)^α))

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Instantaneous modulus (Pa)
        Gk : float
            Retarded modulus (Pa)
        alpha : float
            Fractional order
        tau : float
            Retardation time (s)

        Returns
        -------
        jnp.ndarray
            Creep compliance J(t) (1/Pa)
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        tau_safe = tau + epsilon
        # Instantaneous compliance (elastic response)
        J_inst = 1.0 / (Ge + epsilon)
        # Retarded compliance amplitude
        J_retard_amp = 1.0 / (Gk + epsilon)
        # Compute argument: z = -(t/τ)^α
        z = -jnp.power(t / tau_safe, alpha_safe)
        # Mittag-Leffler function E_α(z) with concrete alpha
        ml_term = mittag_leffler_e(z, alpha=alpha_safe)
        # J(t) = 1/G_e + (1/G_k) * (1 - E_α(-(t/τ)^α))
        J_t = J_inst + J_retard_amp * (1.0 - ml_term)

        return J_t

    @staticmethod
    @jax.jit
    def _predict_relaxation(
        t: jnp.ndarray,
        Ge: float,
        Gk: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        G(t) exhibits stress relaxation from instantaneous to
        equilibrium modulus.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Instantaneous modulus (Pa)
        Gk : float
            Retarded modulus (Pa)
        alpha : float
            Fractional order
        tau : float
            Retardation time (s)

        Returns
        -------
        jnp.ndarray
            Relaxation modulus G(t) (Pa)
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        tau_safe = tau + epsilon
        # Compute transition function
        z = -jnp.power(t / tau_safe, alpha_safe)
        ml_term = mittag_leffler_e(z, alpha=alpha_safe)
        # Instantaneous modulus
        G_inst = Ge
        # Equilibrium modulus (series combination)
        G_eq = (Ge * Gk) / (Ge + Gk + epsilon)
        # Interpolate using Mittag-Leffler decay
        # G(t) = G_eq + (G_inst - G_eq) * E_α(-(t/τ)^α)
        G_t = G_eq + (G_inst - G_eq) * ml_term

        return G_t

    @staticmethod
    @jax.jit
    def _predict_oscillation(
        omega: jnp.ndarray,
        Ge: float,
        Gk: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        Convert from complex compliance:
        J*(ω) = 1/G_e + (1/G_k) / (1 + (iωτ)^α)
        G*(ω) = 1 / J*(ω)

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        Ge : float
            Instantaneous modulus (Pa)
        Gk : float
            Retarded modulus (Pa)
        alpha : float
            Fractional order
        tau : float
            Retardation time (s)

        Returns
        -------
        jnp.ndarray
            Complex modulus array with shape (..., 2) where [:, 0] is G' and [:, 1] is G''
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        tau_safe = tau + epsilon
        # Compute (iωτ)^α
        omega_tau_alpha = jnp.power(omega * tau_safe, alpha_safe)
        phase = jnp.pi * alpha_safe / 2.0
        i_omega_tau_alpha = omega_tau_alpha * (jnp.cos(phase) + 1j * jnp.sin(phase))
        # Complex compliance
        J_inst = 1.0 / (Ge + epsilon)
        J_kv = (1.0 / (Gk + epsilon)) / (1.0 + i_omega_tau_alpha)
        J_star = J_inst + J_kv
        # Complex modulus (inverse of compliance)
        G_star = 1.0 / (J_star + epsilon)
        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalPoyntingThomson:
        """Fit model to data using NLSQ TRF optimization.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable (time or frequency)
        y : jnp.ndarray
            Dependent variable (modulus or compliance)
        **kwargs : dict
            Additional fitting options

        Returns
        -------
        self
            Fitted model instance
        """
        from rheojax.core.test_modes import TestMode
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Detect test mode
        test_mode_str = kwargs.get("test_mode", "creep")

        # Convert string to TestMode enum
        if isinstance(test_mode_str, str):
            test_mode_map = {
                "relaxation": TestMode.RELAXATION,
                "creep": TestMode.CREEP,
                "oscillation": TestMode.OSCILLATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.CREEP)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        logger.info(
            "Starting FractionalPoyntingThomson fit",
            test_mode=(
                test_mode.value if hasattr(test_mode, "value") else str(test_mode)
            ),
            data_shape=X.shape,
        )

        with log_fit(
            logger, model="FractionalPoyntingThomson", data_shape=X.shape
        ) as ctx:
            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == TestMode.OSCILLATION:
                try:
                    import numpy as np

                    from rheojax.utils.initialization import (
                        initialize_fractional_poynting_thomson,
                    )

                    logger.debug(
                        "Attempting smart initialization for oscillation mode",
                        data_points=len(X),
                    )
                    success = initialize_fractional_poynting_thomson(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            Ge=self.parameters.get_value("Ge"),
                            Gk=self.parameters.get_value("Gk"),
                            alpha=self.parameters.get_value("alpha"),
                            tau=self.parameters.get_value("tau"),
                        )
                except Exception as e:
                    # Silent fallback to defaults - don't break if initialization fails
                    logger.debug(
                        "Smart initialization failed, using defaults",
                        error=str(e),
                        exc_info=True,
                    )

            # Create stateless model function for optimization
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                Ge, Gk, alpha, tau = params[0], params[1], params[2], params[3]

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, Ge, Gk, alpha, tau)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, Ge, Gk, alpha, tau)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, Ge, Gk, alpha, tau)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            # Create objective function
            logger.debug("Creating least squares objective function")
            objective = create_least_squares_objective(
                model_fn, jnp.array(X), jnp.array(y), normalize=True
            )

            # Optimize using NLSQ TRF
            logger.debug(
                "Starting NLSQ optimization",
                method=kwargs.get("method", "auto"),
                max_iter=kwargs.get("max_iter", 1000),
            )
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
                    "NLSQ optimization failed",
                    message=result.message,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True
            ctx["success"] = True
            ctx["fitted_params"] = {
                "Ge": self.parameters.get_value("Ge"),
                "Gk": self.parameters.get_value("Gk"),
                "alpha": self.parameters.get_value("alpha"),
                "tau": self.parameters.get_value("tau"),
            }

        logger.info(
            "FractionalPoyntingThomson fit completed",
            Ge=self.parameters.get_value("Ge"),
            Gk=self.parameters.get_value("Gk"),
            alpha=self.parameters.get_value("alpha"),
            tau=self.parameters.get_value("tau"),
        )
        return self

    def _predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict response for given input.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable

        Returns
        -------
        jnp.ndarray
            Predicted values
        """
        # Get parameters
        params = self.parameters.to_dict()
        Ge = params["Ge"]
        Gk = params["Gk"]
        alpha = params["alpha"]
        tau = params["tau"]

        # Auto-detect test mode
        if jnp.all(X > 0) and len(X) > 1:
            log_range = jnp.log10(jnp.max(X)) - jnp.log10(jnp.min(X) + 1e-12)
            if log_range > 3:
                return self._predict_oscillation(X, Ge, Gk, alpha, tau)

        # Default to creep (primary mode for FPT)
        return self._predict_creep(X, Ge, Gk, alpha, tau)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Ge, Gk, alpha, tau]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        Ge = params[0]
        Gk = params[1]
        alpha = params[2]
        tau = params[3]

        # Use test_mode from last fit if available, otherwise default to CREEP
        # Use explicit test_mode parameter (closure-captured in fit_bayesian)

        # Fall back to self._test_mode only for backward compatibility

        if test_mode is None:

            test_mode = getattr(self, "_test_mode", TestMode.CREEP)

        # Normalize test_mode to handle both string and TestMode enum
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        logger.debug(
            "model_function evaluation",
            test_mode=str(test_mode),
            alpha=float(alpha) if hasattr(alpha, "item") else alpha,
            input_shape=X.shape if hasattr(X, "shape") else len(X),
        )

        # Call appropriate prediction function based on test mode
        if test_mode == TestMode.RELAXATION:
            logger.debug("Computing relaxation modulus with Mittag-Leffler evaluation")
            return self._predict_relaxation(X, Ge, Gk, alpha, tau)
        elif test_mode == TestMode.CREEP:
            logger.debug("Computing creep compliance with Mittag-Leffler evaluation")
            return self._predict_creep(X, Ge, Gk, alpha, tau)
        elif test_mode == TestMode.OSCILLATION:
            logger.debug("Computing complex modulus for oscillation mode")
            return self._predict_oscillation(X, Ge, Gk, alpha, tau)
        else:
            # Default to creep mode for FPT model
            logger.debug("Default to creep mode prediction")
            return self._predict_creep(X, Ge, Gk, alpha, tau)


# Convenience alias
FPT = FractionalPoyntingThomson

__all__ = ["FractionalPoyntingThomson", "FPT"]
