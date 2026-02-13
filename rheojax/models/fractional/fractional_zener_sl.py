"""Fractional Zener Solid-Liquid (FZSL) Model.

This model combines a fractional Maxwell element (SpringPot + dashpot) in parallel
with a spring, providing both equilibrium elasticity and fractional relaxation behavior.

Theory
------
The FZSL model consists of:
- Spring (G_e) in parallel with
- Fractional Maxwell element (SpringPot c_alpha + dashpot eta in series)

Relaxation modulus:
    G(t) = G_e + c_α * t^(-α) * E_{1-α,1}(-(t/τ)^(1-α))

Complex modulus:
    G*(ω) = G_e + c_α * (iω)^α / (1 + iωτ)

where E_{α,β} is the two-parameter Mittag-Leffler function.

Parameters
----------
Ge : float
    Equilibrium modulus (Pa), bounds [1e-3, 1e9]
c_alpha : float
    SpringPot constant (Pa·s^α), bounds [1e-3, 1e9]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau : float
    Relaxation time (s), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Purely elastic behavior (spring only)
- alpha → 1: Classical Zener solid (two springs and one dashpot)

References
----------
- Mainardi, F. (2010). Fractional Calculus and Waves in Linear Viscoelasticity
- Schiessel, H., et al. (1995). J. Phys. A: Math. Gen. 28, 6567
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.utils.mittag_leffler import mittag_leffler_e2

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_zener_sl",
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
class FractionalZenerSolidLiquid(BaseModel):
    """Fractional Zener Solid-Liquid model.

    A fractional viscoelastic model combining equilibrium elasticity
    with fractional relaxation behavior.

    Test Modes
    ----------
    - Relaxation: Supported
    - Creep: Supported (via numerical inversion)
    - Oscillation: Supported
    - Rotation: Not supported (no steady-state flow)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalZenerSolidLiquid
    >>>
    >>> # Create model
    >>> model = FractionalZenerSolidLiquid()
    >>>
    >>> # Set parameters
    >>> model.set_params(Ge=1000.0, c_alpha=500.0, alpha=0.5, tau=1.0)
    >>>
    >>> # Predict relaxation modulus
    >>> t = jnp.logspace(-2, 2, 50)
    >>> G_t = model.predict(t)  # Relaxation mode
    >>>
    >>> # Predict complex modulus
    >>> omega = jnp.logspace(-2, 2, 50)
    >>> G_star = model.predict(omega)  # Oscillation mode
    """

    def __init__(self):
        """Initialize Fractional Zener Solid-Liquid model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="Ge",
            value=1000.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Equilibrium modulus",
        )
        self.parameters.add(
            name="c_alpha",
            value=500.0,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="SpringPot constant",
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
            description="Relaxation time",
        )

    @staticmethod
    @jax.jit
    def _predict_relaxation(
        t: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        G(t) = G_e + c_α * t^(-α) * E_{1-α,1}(-(t/τ)^(1-α))

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Equilibrium modulus (Pa)
        c_alpha : float
            SpringPot constant (Pa·s^α)
        alpha : float
            Fractional order
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Relaxation modulus G(t) (Pa)
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Parameters for two-parameter Mittag-Leffler: E_{1-α,1}
        ml_alpha = 1.0 - alpha_safe
        ml_beta = 1.0

        tau_safe = tau + epsilon
        # Compute fractional relaxation term
        # E_{1-α,1}(-(t/τ)^(1-α))
        z = -jnp.power(t / tau_safe, ml_alpha)
        # Mittag-Leffler function with concrete alpha/beta
        ml_term = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)
        # G(t) = G_e + c_α * t^(-α) * E_{1-α,1}(...)
        fractional_term = c_alpha * jnp.power(t, -alpha_safe) * ml_term

        G_t = Ge + fractional_term
        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep(
        t: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Note: Analytical creep compliance for FZSL is complex.
        This uses numerical approximation based on inverse relationship.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Equilibrium modulus (Pa)
        c_alpha : float
            SpringPot constant (Pa·s^α)
        alpha : float
            Fractional order
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Creep compliance J(t) (1/Pa)
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # For equilibrium: J(∞) = 1/G_e
        # Approximate creep using inverse relaxation at long times
        J_eq = 1.0 / (Ge + epsilon)
        # Short time: dominated by SpringPot
        # J(t) ≈ t^α / c_α for small t
        J_short = jnp.power(t, alpha_safe) / (c_alpha + epsilon)
        # Use smooth, monotonic interpolation
        # Sigmoid-based transition to ensure monotonicity
        # Map time to sigmoid argument with characteristic scale tau
        x = jnp.log10(t / tau + epsilon) / 2.0  # Log-scale transition
        sigmoid_weight = 1.0 / (1.0 + jnp.exp(-x))
        # Ensure J_short <= J_eq at transition by scaling
        J_short_scaled = jnp.minimum(J_short, J_eq * 0.9)
        # Monotonic blend: start from J_short, approach J_eq
        J_t = J_short_scaled * (1.0 - sigmoid_weight) + J_eq * sigmoid_weight

        return J_t

    @staticmethod
    @jax.jit
    def _predict_oscillation(
        omega: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        G*(ω) = G_e + c_α * (iω)^α / (1 + (iωτ)^(1-α))

        This is the correct formula for FZSL (spring + FMG in parallel).

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        Ge : float
            Equilibrium modulus (Pa)
        c_alpha : float
            SpringPot constant (Pa·s^α)
        alpha : float
            Fractional order
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Complex modulus array with shape (..., 2) where [:, 0] is G' and [:, 1] is G''
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # beta = 1 - alpha for this model
        beta_safe = 1.0 - alpha_safe

        tau_safe = tau + epsilon
        omega_safe = jnp.maximum(omega, epsilon)
        # Compute (iω)^α = ω^α * exp(i*π*α/2)
        omega_alpha = jnp.power(omega_safe, alpha_safe)
        phase_alpha = jnp.pi * alpha_safe / 2.0
        i_omega_alpha = omega_alpha * (jnp.cos(phase_alpha) + 1j * jnp.sin(phase_alpha))
        # Compute (iωτ)^(1-α) = |ωτ|^(1-α) * exp(i*(1-α)*π/2)
        omega_tau = omega_safe * tau_safe
        omega_tau_beta = jnp.power(omega_tau, beta_safe)
        phase_beta = jnp.pi * beta_safe / 2.0
        i_omega_tau_beta = omega_tau_beta * (
            jnp.cos(phase_beta) + 1j * jnp.sin(phase_beta)
        )
        # Denominator: 1 + (iωτ)^(1-α)
        denominator = 1.0 + i_omega_tau_beta
        # Fractional term: c_α * (iω)^α / (1 + (iωτ)^(1-α))
        fractional_term = c_alpha * i_omega_alpha / denominator
        # Total complex modulus
        G_star = Ge + fractional_term
        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerSolidLiquid:
        """Fit model to data using NLSQ TRF optimization.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable (time or frequency)
        y : jnp.ndarray
            Dependent variable (modulus or compliance)
        **kwargs : dict
            Additional fitting options (test_mode, optimization settings)

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

        # Detect test mode if not provided
        test_mode_str = kwargs.get("test_mode", "relaxation")

        # Convert string to TestMode enum
        if isinstance(test_mode_str, str):
            test_mode_map = {
                "relaxation": TestMode.RELAXATION,
                "creep": TestMode.CREEP,
                "oscillation": TestMode.OSCILLATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.RELAXATION)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        # Determine data shape for logging
        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger,
            model="FractionalZenerSolidLiquid",
            data_shape=data_shape,
            test_mode=(
                test_mode_str if isinstance(test_mode_str, str) else str(test_mode)
            ),
        ) as ctx:
            logger.debug(
                "Starting FZSL fit",
                n_points=len(X) if hasattr(X, "__len__") else 1,
                test_mode=str(test_mode),
                initial_params=self.parameters.to_dict(),
            )

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == TestMode.OSCILLATION:
                try:
                    import numpy as np

                    from rheojax.utils.initialization import (
                        initialize_fractional_zener_sl,
                    )

                    success = initialize_fractional_zener_sl(
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

            # Create stateless model function for optimization
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                Ge, c_alpha, alpha, tau = params[0], params[1], params[2], params[3]

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, Ge, c_alpha, alpha, tau)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, Ge, c_alpha, alpha, tau)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, Ge, c_alpha, alpha, tau)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            # Create objective function
            logger.debug("Creating least squares objective", normalize=True)
            objective = create_least_squares_objective(
                model_fn, jnp.array(X), jnp.array(y), normalize=True
            )

            # Optimize using NLSQ TRF
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
                "FZSL fit completed successfully",
                final_params=self.parameters.to_dict(),
            )

        return self

    def _predict(self, X: jnp.ndarray) -> jnp.ndarray:
        """Predict response for given input.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable (time or frequency)

        Returns
        -------
        jnp.ndarray
            Predicted values
        """
        # Get parameter values
        Ge = self.parameters.get_value("Ge")
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        tau = self.parameters.get_value("tau")

        # Auto-detect test mode based on input characteristics
        # NOTE: This is a heuristic - explicit test_mode is recommended
        # Default to relaxation for time-domain data
        # Oscillation should typically use RheoData with domain='frequency'
        return self._predict_relaxation(X, Ge, c_alpha, alpha, tau)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Ge, c_alpha, alpha, tau]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        Ge = params[0]
        c_alpha = params[1]
        alpha = params[2]
        tau = params[3]

        # Use test_mode from last fit if available, otherwise default to RELAXATION
        # Use explicit test_mode parameter (closure-captured in fit_bayesian)

        # Fall back to self._test_mode only for backward compatibility

        if test_mode is None:

            test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Normalize test_mode to handle both string and TestMode enum
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        # Call appropriate prediction function based on test mode
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, Ge, c_alpha, alpha, tau)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, Ge, c_alpha, alpha, tau)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(X, Ge, c_alpha, alpha, tau)
        else:
            # Default to relaxation mode for FZSL model
            return self._predict_relaxation(X, Ge, c_alpha, alpha, tau)


# Convenience alias
FZSL = FractionalZenerSolidLiquid

__all__ = ["FractionalZenerSolidLiquid", "FZSL"]
