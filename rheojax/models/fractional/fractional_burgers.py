"""Fractional Burgers Model (FBM).

This model combines Maxwell and Kelvin-Voigt elements in series with
fractional derivatives, providing four relaxation mechanisms for
complex viscoelastic behavior.

Theory
------
The FBM model consists of:
- Maxwell element (spring + dashpot) in series with
- Fractional Kelvin-Voigt element (spring + SpringPot)

Creep compliance:
    J(t) = J_g + (t^α)/(η_1 * Γ(1+α)) + J_k * (1 - E_α(-(t/τ_k)^α))

where:
- J_g: Glassy compliance (instantaneous)
- η_1: Viscosity (Maxwell arm)
- J_k: Kelvin compliance
- τ_k: Retardation time

Parameters
----------
Jg : float
    Glassy compliance (1/Pa), bounds [1e-9, 1e3]
eta1 : float
    Viscosity (Pa·s), bounds [1e-6, 1e12]
Jk : float
    Kelvin compliance (1/Pa), bounds [1e-9, 1e3]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau_k : float
    Retardation time (s), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Classical Burgers model with Newtonian flow
- alpha → 1: Burgers model with power-law flow

References
----------
- Mainardi, F. (2010). Fractional Calculus and Waves in Linear Viscoelasticity
- Bagley, R.L. & Torvik, P.J. (1986). J. Rheol. 30, 133-155
- Schiessel, H. & Blumen, A. (1993). J. Phys. A: Math. Gen. 26, 5057
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()

from jax.scipy.special import gamma as jax_gamma

from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e

logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_burgers",
    protocols=[
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.OSCILLATION,
    ],
)
class FractionalBurgersModel(BaseModel):
    """Fractional Burgers model.

    A four-parameter fractional viscoelastic model combining
    instantaneous compliance, viscous flow, and retardation.

    Test Modes
    ----------
    - Relaxation: Supported (via inversion)
    - Creep: Supported (primary mode)
    - Oscillation: Supported
    - Rotation: Partial support (viscous flow at low frequencies)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalBurgersModel
    >>>
    >>> # Create model
    >>> model = FractionalBurgersModel()
    >>>
    >>> # Set parameters
    >>> model.set_params(Jg=1e-6, eta1=1000.0, Jk=5e-6, alpha=0.5, tau_k=1.0)
    >>>
    >>> # Predict creep compliance
    >>> t = jnp.logspace(-2, 2, 50)
    >>> J_t = model.predict(t)
    """

    def __init__(self):
        """Initialize Fractional Burgers model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="Jg",
            value=1e-6,
            bounds=(1e-9, 1e3),
            units="1/Pa",
            description="Glassy compliance",
        )
        self.parameters.add(
            name="eta1",
            value=1000.0,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Viscosity (Maxwell arm)",
        )
        self.parameters.add(
            name="Jk",
            value=1e-5,
            bounds=(1e-9, 1e3),
            units="1/Pa",
            description="Kelvin compliance",
        )
        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau_k",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s",
            description="Retardation time",
        )

    @staticmethod
    @jax.jit
    def _predict_creep(
        t: jnp.ndarray,
        Jg: float,
        eta1: float,
        Jk: float,
        alpha: float,
        tau_k: float,
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        J(t) = J_g + t^α/(η_1 * Γ(1+α)) + J_k * (1 - E_α(-(t/τ_k)^α))

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Jg : float
            Glassy compliance (1/Pa)
        eta1 : float
            Viscosity (Pa·s)
        Jk : float
            Kelvin compliance (1/Pa)
        alpha : float
            Fractional order
        tau_k : float
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

        tau_k_safe = tau_k + epsilon
        eta1_safe = eta1 + epsilon
        # Instantaneous compliance (elastic response)
        J_instant = Jg
        # Fractional viscous flow term: t^α / (η_1 * Γ(1+α))
        gamma_term = jax_gamma(1.0 + alpha_safe)
        J_flow = jnp.power(t, alpha_safe) / (eta1_safe * gamma_term)
        # Retardation term: J_k * (1 - E_α(-(t/τ_k)^α))
        z = -jnp.power(t / tau_k_safe, alpha_safe)
        ml_term = mittag_leffler_e(z, alpha=alpha_safe)
        J_retard = Jk * (1.0 - ml_term)
        # Total creep compliance
        J_t = J_instant + J_flow + J_retard

        return J_t

    @staticmethod
    @jax.jit
    def _predict_relaxation(
        t: jnp.ndarray,
        Jg: float,
        eta1: float,
        Jk: float,
        alpha: float,
        tau_k: float,
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Note: Analytical relaxation modulus requires numerical inversion.
        This provides an approximation.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Jg : float
            Glassy compliance (1/Pa)
        eta1 : float
            Viscosity (Pa·s)
        Jk : float
            Kelvin compliance (1/Pa)
        alpha : float
            Fractional order
        tau_k : float
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

        tau_k_safe = tau_k + epsilon
        # Approximate using inverse relationship
        # G(0) ≈ 1/J_g (instantaneous modulus)
        G_inst = 1.0 / (Jg + epsilon)
        # Long-time decay (fractional Maxwell-like)
        # G(t) ~ t^(-α) at intermediate times
        G_decay = G_inst * jnp.power(t / tau_k_safe, -alpha_safe)
        # Smooth transition
        z = -jnp.power(t / tau_k_safe, alpha_safe)
        ml_term = mittag_leffler_e(z, alpha=alpha_safe)
        # Combine terms
        G_t = G_inst * ml_term + G_decay * (1.0 - ml_term)

        return G_t

    @staticmethod
    @jax.jit
    def _predict_oscillation(
        omega: jnp.ndarray,
        Jg: float,
        eta1: float,
        Jk: float,
        alpha: float,
        tau_k: float,
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        Computed from complex compliance:
        J*(ω) = J_g + (iω)^(-α)/(η_1*Γ(1-α)) + J_k/(1 + (iωτ_k)^α)
        G*(ω) = 1/J*(ω)

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        Jg : float
            Glassy compliance (1/Pa)
        eta1 : float
            Viscosity (Pa·s)
        Jk : float
            Kelvin compliance (1/Pa)
        alpha : float
            Fractional order
        tau_k : float
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

        tau_k_safe = tau_k + epsilon
        eta1_safe = eta1 + epsilon
        # Instantaneous compliance
        J_inst = Jg
        # Fractional viscous term: (iω)^(-α) / (η_1 * Γ(1-α))
        omega_neg_alpha = jnp.power(omega, -alpha_safe)
        phase = -jnp.pi * alpha_safe / 2.0
        i_omega_neg_alpha = omega_neg_alpha * (jnp.cos(phase) + 1j * jnp.sin(phase))
        gamma_term = jax_gamma(1.0 - alpha_safe)
        J_flow = i_omega_neg_alpha / (eta1_safe * gamma_term)
        # Retardation term: J_k / (1 + (iωτ_k)^α)
        omega_tau_alpha = jnp.power(omega * tau_k_safe, alpha_safe)
        phase_alpha = jnp.pi * alpha_safe / 2.0
        i_omega_tau_alpha = omega_tau_alpha * (
            jnp.cos(phase_alpha) + 1j * jnp.sin(phase_alpha)
        )
        J_retard = Jk / (1.0 + i_omega_tau_alpha)
        # Total complex compliance
        J_star = J_inst + J_flow + J_retard
        # Complex modulus (inverse)
        G_star = 1.0 / (J_star + epsilon)
        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray, **kwargs) -> FractionalBurgersModel:
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
                "creep": TestMode.CREEP,
                "relaxation": TestMode.RELAXATION,
                "oscillation": TestMode.OSCILLATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.CREEP)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        logger.info(
            "Starting FractionalBurgersModel fit",
            test_mode=(
                test_mode.value if hasattr(test_mode, "value") else str(test_mode)
            ),
            data_shape=X.shape,
        )

        with log_fit(logger, model="FractionalBurgersModel", data_shape=X.shape) as ctx:
            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == TestMode.OSCILLATION:
                try:
                    import numpy as np

                    from rheojax.utils.initialization import (
                        initialize_fractional_burgers,
                    )

                    logger.debug(
                        "Attempting smart initialization for oscillation mode",
                        data_points=len(X),
                    )
                    success = initialize_fractional_burgers(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            Jg=self.parameters.get_value("Jg"),
                            eta1=self.parameters.get_value("eta1"),
                            Jk=self.parameters.get_value("Jk"),
                            alpha=self.parameters.get_value("alpha"),
                            tau_k=self.parameters.get_value("tau_k"),
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
                Jg, eta1, Jk, alpha, tau_k = (
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                )

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, Jg, eta1, Jk, alpha, tau_k)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, Jg, eta1, Jk, alpha, tau_k)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, Jg, eta1, Jk, alpha, tau_k)
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
                "Jg": self.parameters.get_value("Jg"),
                "eta1": self.parameters.get_value("eta1"),
                "Jk": self.parameters.get_value("Jk"),
                "alpha": self.parameters.get_value("alpha"),
                "tau_k": self.parameters.get_value("tau_k"),
            }

        logger.info(
            "FractionalBurgersModel fit completed",
            Jg=self.parameters.get_value("Jg"),
            eta1=self.parameters.get_value("eta1"),
            Jk=self.parameters.get_value("Jk"),
            alpha=self.parameters.get_value("alpha"),
            tau_k=self.parameters.get_value("tau_k"),
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
        Jg = params["Jg"]
        eta1 = params["eta1"]
        Jk = params["Jk"]
        alpha = params["alpha"]
        tau_k = params["tau_k"]

        # Auto-detect test mode
        if jnp.all(X > 0) and len(X) > 1:
            log_range = jnp.log10(jnp.max(X)) - jnp.log10(jnp.min(X) + 1e-12)
            if log_range > 3:
                return self._predict_oscillation(X, Jg, eta1, Jk, alpha, tau_k)

        # Default to creep (primary mode for Burgers)
        return self._predict_creep(X, Jg, eta1, Jk, alpha, tau_k)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Jg, eta1, Jk, alpha, tau_k]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        Jg = params[0]
        eta1 = params[1]
        Jk = params[2]
        alpha = params[3]
        tau_k = params[4]

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
            return self._predict_relaxation(X, Jg, eta1, Jk, alpha, tau_k)
        elif test_mode == TestMode.CREEP:
            logger.debug("Computing creep compliance with Mittag-Leffler evaluation")
            return self._predict_creep(X, Jg, eta1, Jk, alpha, tau_k)
        elif test_mode == TestMode.OSCILLATION:
            logger.debug("Computing complex modulus for oscillation mode")
            return self._predict_oscillation(X, Jg, eta1, Jk, alpha, tau_k)
        else:
            # Default to creep mode for Burgers model
            logger.debug("Default to creep mode prediction")
            return self._predict_creep(X, Jg, eta1, Jk, alpha, tau_k)


# Convenience alias
FBM = FractionalBurgersModel

__all__ = ["FractionalBurgersModel", "FBM"]
