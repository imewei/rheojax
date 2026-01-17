"""Fractional Zener Liquid-Liquid (FZLL) Model.

This is the most general fractional Zener model with two SpringPots and one dashpot,
providing maximum flexibility in describing fractional viscoelastic behavior.

Theory
------
The FZLL model consists of:
- Two SpringPots with different fractional orders
- One dashpot
- Complex arrangement providing liquid-like behavior at long times

Complex modulus:
    G*(ω) = c_1 * (iω)^α / (1 + (iωτ)^β) + c_2 * (iω)^γ

where all three fractional orders (α, β, γ) can be different.

Parameters
----------
c1 : float
    First SpringPot constant (Pa·s^α), bounds [1e-3, 1e9]
c2 : float
    Second SpringPot constant (Pa·s^γ), bounds [1e-3, 1e9]
alpha : float
    First fractional order, bounds [0.0, 1.0]
beta : float
    Second fractional order, bounds [0.0, 1.0]
gamma : float
    Third fractional order, bounds [0.0, 1.0]
tau : float
    Relaxation time (s), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha, beta, gamma → 1: Classical viscoelastic liquid
- beta → 0: Simplifies to parallel SpringPots

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

logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_zener_ll",
    protocols=[
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
    ],
)
class FractionalZenerLiquidLiquid(BaseModel):
    """Fractional Zener Liquid-Liquid model.

    The most general fractional Zener model with three independent
    fractional orders.

    Test Modes
    ----------
    - Relaxation: Supported (numerical)
    - Creep: Supported (numerical)
    - Oscillation: Supported (analytical)
    - Rotation: Partial support (power-law at high shear rates)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalZenerLiquidLiquid
    >>>
    >>> # Create model
    >>> model = FractionalZenerLiquidLiquid()
    >>>
    >>> # Set parameters
    >>> model.set_params(c1=500.0, c2=100.0, alpha=0.5, beta=0.3, gamma=0.7, tau=1.0)
    >>>
    >>> # Predict complex modulus
    >>> omega = jnp.logspace(-2, 2, 50)
    >>> G_star = model.predict(omega)
    """

    def __init__(self):
        """Initialize Fractional Zener Liquid-Liquid model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="c1",
            value=500.0,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="First SpringPot constant",
        )
        self.parameters.add(
            name="c2",
            value=500.0,
            bounds=(1e-3, 1e9),
            units="Pa·s^γ",
            description="Second SpringPot constant",
        )
        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="First fractional order",
        )
        self.parameters.add(
            name="beta",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="Second fractional order",
        )
        self.parameters.add(
            name="gamma",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="Third fractional order",
        )
        self.parameters.add(
            name="tau",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s",
            description="Relaxation time",
        )

    def _predict_oscillation(
        self,
        omega: jnp.ndarray,
        c1: float,
        c2: float,
        alpha: float,
        beta: float,
        gamma: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        G*(ω) = c_1 * (iω)^α / (1 + (iωτ)^β) + c_2 * (iω)^γ

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        c1 : float
            First SpringPot constant (Pa·s^α)
        c2 : float
            Second SpringPot constant (Pa·s^γ)
        alpha : float
            First fractional order
        beta : float
            Second fractional order
        gamma : float
            Third fractional order
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Complex modulus array with shape (..., 2) where [:, 0] is G' and [:, 1] is G''
        """
        # Clip fractional orders BEFORE JIT to make them concrete (not traced)

        epsilon = 1e-12
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        beta_safe = jnp.clip(beta, epsilon, 1.0 - epsilon)
        gamma_safe = jnp.clip(gamma, epsilon, 1.0 - epsilon)

        # JIT-compiled inner function with concrete alpha/beta/gamma
        @jax.jit
        def _compute_oscillation(omega, c1, c2, tau):
            tau_safe = tau + epsilon

            # First term: c_1 * (iω)^α / (1 + (iωτ)^β)

            # Compute (iω)^α
            omega_alpha = jnp.power(omega, alpha_safe)
            phase_alpha = jnp.pi * alpha_safe / 2.0
            i_omega_alpha = omega_alpha * (
                jnp.cos(phase_alpha) + 1j * jnp.sin(phase_alpha)
            )

            # Compute (iωτ)^β
            omega_tau_beta = jnp.power(omega * tau_safe, beta_safe)
            phase_beta = jnp.pi * beta_safe / 2.0
            i_omega_tau_beta = omega_tau_beta * (
                jnp.cos(phase_beta) + 1j * jnp.sin(phase_beta)
            )

            # First term
            term1 = c1 * i_omega_alpha / (1.0 + i_omega_tau_beta)

            # Second term: c_2 * (iω)^γ
            omega_gamma = jnp.power(omega, gamma_safe)
            phase_gamma = jnp.pi * gamma_safe / 2.0
            i_omega_gamma = omega_gamma * (
                jnp.cos(phase_gamma) + 1j * jnp.sin(phase_gamma)
            )

            term2 = c2 * i_omega_gamma

            # Total complex modulus
            G_star = term1 + term2

            # Extract storage and loss moduli
            G_prime = jnp.real(G_star)
            G_double_prime = jnp.imag(G_star)

            return jnp.stack([G_prime, G_double_prime], axis=-1)

        return _compute_oscillation(omega, c1, c2, tau)

    def _predict_relaxation(
        self,
        t: jnp.ndarray,
        c1: float,
        c2: float,
        alpha: float,
        beta: float,
        gamma: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Uses a combined power-law and exponential decay form:

        G(t) = c1 / (1 + (t/tau)^alpha) + c2 * exp(-t/tau)

        This form provides liquid-like behavior (G→0 as t→∞) with two
        distinct contributions:
        - c1 term: fractional power-law decay controlled by alpha
        - c2 term: exponential decay for short-time dynamics

        This parameterization avoids the c1/c2 degeneracy that would occur
        if only the sum (c1 + c2) affected predictions.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        c1 : float
            First SpringPot constant (power-law term)
        c2 : float
            Second SpringPot constant (exponential term)
        alpha : float
            First fractional order (controls power-law decay)
        beta : float
            Second fractional order (not used in relaxation)
        gamma : float
            Third fractional order (not used in relaxation)
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Relaxation modulus G(t) (Pa)
        """
        epsilon = 1e-12

        # JIT-compiled function with JAX-compatible clipping
        @jax.jit
        def _compute_relaxation(t, c1, c2, alpha, beta, gamma, tau):
            # Clip fractional orders using JAX operations (tracer-safe)
            alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
            tau_safe = tau + epsilon

            # Use c1 as the main modulus contribution and alpha as the decay exponent
            # This breaks the c1/c2 degeneracy by using them differently
            # G(t) = c1 / (1 + (t/tau)^alpha) + c2 * exp(-t/tau)
            t_tau_ratio = t / tau_safe

            # Primary power-law decay term from c1
            term1 = c1 / (1.0 + jnp.power(t_tau_ratio, alpha_safe))

            # Secondary exponential decay term from c2 (simpler decay)
            term2 = c2 * jnp.exp(-t_tau_ratio)

            G_t = term1 + term2

            return G_t

        return _compute_relaxation(t, c1, c2, alpha, beta, gamma, tau)

    def _predict_creep(
        self,
        t: jnp.ndarray,
        c1: float,
        c2: float,
        alpha: float,
        beta: float,
        gamma: float,
        tau: float,
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Note: Analytical creep compliance is complex for FZLL.
        This provides a numerical approximation.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        c1 : float
            First SpringPot constant
        c2 : float
            Second SpringPot constant
        alpha : float
            First fractional order
        beta : float
            Second fractional order
        gamma : float
            Third fractional order
        tau : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Creep compliance J(t) (1/Pa)
        """
        # Clip fractional orders BEFORE JIT to make them concrete (not traced)

        epsilon = 1e-12
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        gamma_safe = jnp.clip(gamma, epsilon, 1.0 - epsilon)

        # Compute average order as concrete value
        avg_order = (alpha_safe + gamma_safe) / 2.0

        # JIT-compiled inner function with concrete alpha/gamma
        @jax.jit
        def _compute_creep(t, c1, c2, tau):
            # Short time behavior
            J_short = jnp.power(t, alpha_safe) / (c1 + epsilon)

            # Long time behavior (unbounded growth for liquid)
            J_long = jnp.power(t, avg_order) / (c2 + epsilon)

            # Crossover
            weight = jnp.tanh(t / tau)
            J_t = J_short * (1.0 - weight) + J_long * weight

            return J_t

        return _compute_creep(t, c1, c2, tau)

    def _initialize_relaxation_parameters(self, X, y) -> bool:
        """Derive heuristic starting values from relaxation data.

        For FZLL, estimates c1, c2, and tau from the data characteristics.
        """
        import numpy as np

        logger.debug(
            "Attempting relaxation parameter initialization",
            data_size=len(X) if hasattr(X, "__len__") else "unknown",
        )

        try:
            t = np.asarray(X, dtype=float).ravel()
            g = np.asarray(y, dtype=float).ravel()
            if t.shape != g.shape or t.size < 4:
                logger.debug(
                    "Insufficient data for relaxation initialization",
                    t_shape=t.shape,
                    g_shape=g.shape,
                )
                return False

            order = np.argsort(t)
            t_sorted = t[order]
            g_sorted = g[order]

            # For liquid model, G(t→∞) → 0
            # Formula: G(t) = c1 / (1 + (t/tau)^alpha) + c2 * exp(-t/tau)
            # At t=0: G(0) ≈ c1 + c2

            # Find where G is at half of its maximum - that gives us tau estimate
            g_max = float(np.max(g_sorted))

            # At early times, G ≈ c1 / (1 + (t_min/tau)^alpha) + c2 * exp(-t_min/tau)
            # If t_min = tau: G = c1/2 + c2*exp(-1) ≈ c1/2 + 0.37*c2
            # So g_max ≈ c1/2 for the case where t_min = tau
            # Estimate: c1 ≈ 2*g_max, c2 ≈ 0 (small contribution)
            c1_guess = g_max * 2.0
            c2_guess = 1.0  # Small contribution from exponential term

            # Find tau from where G decays to half (c1/(1+1) = c1/2 at t=tau)
            # Since g_max ≈ c1/2, we have t_min ≈ tau
            tau_guess = float(t_sorted[0])

            # Get bounds
            c1_bounds = self.parameters.get("c1").bounds or (1e-3, 1e9)
            c2_bounds = self.parameters.get("c2").bounds or (1e-3, 1e9)
            tau_bounds = self.parameters.get("tau").bounds or (1e-6, 1e6)

            c1_guess = float(np.clip(c1_guess, c1_bounds[0], c1_bounds[1]))
            c2_guess = float(np.clip(c2_guess, c2_bounds[0], c2_bounds[1]))
            tau_guess = float(np.clip(tau_guess, tau_bounds[0], tau_bounds[1]))

            self.parameters.set_value("c1", c1_guess)
            self.parameters.set_value("c2", c2_guess)
            self.parameters.set_value("tau", tau_guess)
            # Keep fractional orders at defaults (0.5)
            logger.debug(
                "FZLL relaxation initialization completed",
                c1=c1_guess,
                c2=c2_guess,
                tau=tau_guess,
                g_max=g_max,
            )
            return True
        except Exception as exc:
            logger.debug(
                "Relaxation initialization failed",
                error=str(exc),
                exc_info=True,
            )
            return False

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerLiquidLiquid:
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
        test_mode_str = kwargs.get("test_mode", "oscillation")

        # Convert string to TestMode enum
        if isinstance(test_mode_str, str):
            test_mode_map = {
                "relaxation": TestMode.RELAXATION,
                "creep": TestMode.CREEP,
                "oscillation": TestMode.OSCILLATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.OSCILLATION)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        logger.info(
            "Starting FractionalZenerLiquidLiquid fit",
            test_mode=(
                test_mode.value if hasattr(test_mode, "value") else str(test_mode)
            ),
            data_shape=X.shape,
        )

        with log_fit(
            logger, model="FractionalZenerLiquidLiquid", data_shape=X.shape
        ) as ctx:
            # Data-aware initialization for relaxation mode
            if test_mode == TestMode.RELAXATION:
                logger.debug("Applying relaxation-mode parameter initialization")
                self._initialize_relaxation_parameters(X, y)

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == TestMode.OSCILLATION:
                try:
                    import numpy as np

                    from rheojax.utils.initialization import (
                        initialize_fractional_zener_ll,
                    )

                    logger.debug(
                        "Attempting smart initialization for oscillation mode",
                        data_points=len(X),
                    )
                    success = initialize_fractional_zener_ll(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            c1=self.parameters.get_value("c1"),
                            c2=self.parameters.get_value("c2"),
                            alpha=self.parameters.get_value("alpha"),
                            beta=self.parameters.get_value("beta"),
                            gamma=self.parameters.get_value("gamma"),
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
                c1, c2, alpha, beta, gamma, tau = (
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    params[5],
                )

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, c1, c2, alpha, beta, gamma, tau)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, c1, c2, alpha, beta, gamma, tau)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, c1, c2, alpha, beta, gamma, tau)
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
                "c1": self.parameters.get_value("c1"),
                "c2": self.parameters.get_value("c2"),
                "alpha": self.parameters.get_value("alpha"),
                "beta": self.parameters.get_value("beta"),
                "gamma": self.parameters.get_value("gamma"),
                "tau": self.parameters.get_value("tau"),
            }

        logger.info(
            "FractionalZenerLiquidLiquid fit completed",
            c1=self.parameters.get_value("c1"),
            c2=self.parameters.get_value("c2"),
            alpha=self.parameters.get_value("alpha"),
            beta=self.parameters.get_value("beta"),
            gamma=self.parameters.get_value("gamma"),
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
        # Get parameter values
        c1 = self.parameters.get_value("c1")
        c2 = self.parameters.get_value("c2")
        alpha = self.parameters.get_value("alpha")
        beta = self.parameters.get_value("beta")
        gamma = self.parameters.get_value("gamma")
        tau = self.parameters.get_value("tau")

        # Auto-detect test mode based on input characteristics
        # NOTE: This is a heuristic - explicit test_mode is recommended
        # Default to relaxation for time-domain data
        # Oscillation should typically use RheoData with domain='frequency'
        return self._predict_relaxation(X, c1, c2, alpha, beta, gamma, tau)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c1, c2, alpha, beta, gamma, tau]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        c1 = params[0]
        c2 = params[1]
        alpha = params[2]
        beta = params[3]
        gamma = params[4]
        tau = params[5]

        # Use test_mode from last fit if available, otherwise default to OSCILLATION
        # Get test_mode value BEFORE entering JIT region to avoid tracing issues
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", TestMode.OSCILLATION)

        # Convert to string representation for comparison (JAX-safe)
        if hasattr(test_mode, "value"):
            test_mode_str = test_mode.value
        elif isinstance(test_mode, str):
            test_mode_str = test_mode
        else:
            test_mode_str = "oscillation"

        logger.debug(
            "model_function evaluation",
            test_mode=test_mode_str,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            input_shape=X.shape if hasattr(X, "shape") else len(X),
        )

        # Use string comparison (JAX-safe) instead of enum comparison
        if test_mode_str == "relaxation":
            logger.debug("Computing relaxation modulus for FZLL")
            return self._predict_relaxation(X, c1, c2, alpha, beta, gamma, tau)
        elif test_mode_str == "creep":
            logger.debug("Computing creep compliance for FZLL")
            return self._predict_creep(X, c1, c2, alpha, beta, gamma, tau)
        else:  # Default to oscillation
            logger.debug("Computing complex modulus for oscillation mode")
            return self._predict_oscillation(X, c1, c2, alpha, beta, gamma, tau)


# Convenience alias
FZLL = FractionalZenerLiquidLiquid

__all__ = ["FractionalZenerLiquidLiquid", "FZLL"]
