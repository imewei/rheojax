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

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry


@ModelRegistry.register("fractional_zener_ll")
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
            value=None,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="First SpringPot constant",
        )
        self.parameters.add(
            name="c2",
            value=None,
            bounds=(1e-3, 1e9),
            units="Pa·s^γ",
            description="Second SpringPot constant",
        )
        self.parameters.add(
            name="alpha",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="First fractional order",
        )
        self.parameters.add(
            name="beta",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Second fractional order",
        )
        self.parameters.add(
            name="gamma",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Third fractional order",
        )
        self.parameters.add(
            name="tau",
            value=None,
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
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        beta_safe = float(np.clip(beta, epsilon, 1.0 - epsilon))
        gamma_safe = float(np.clip(gamma, epsilon, 1.0 - epsilon))

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

        Note: Analytical relaxation modulus is complex for FZLL.
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
            Relaxation modulus G(t) (Pa)
        """
        # Clip fractional orders BEFORE JIT to make them concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        gamma_safe = float(np.clip(gamma, epsilon, 1.0 - epsilon))

        # Compute max/min orders as concrete values
        max_order = max(alpha_safe, gamma_safe)
        min(alpha_safe, gamma_safe)

        # JIT-compiled inner function with concrete alpha/gamma
        @jax.jit
        def _compute_relaxation(t, c1, c2, tau):
            tau_safe = tau + epsilon

            # Short time: dominated by highest fractional order
            G_short = (c1 + c2) * jnp.power(t, -max_order)

            # Long time: power-law decay with characteristic time
            G_long = c2 * jnp.power(t, -gamma_safe)

            # Crossover around tau
            weight = jnp.tanh(t / tau_safe)
            G_t = G_short * (1.0 - weight) + G_long * weight

            return G_t

        return _compute_relaxation(t, c1, c2, tau)

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
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        gamma_safe = float(np.clip(gamma, epsilon, 1.0 - epsilon))

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

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerLiquidLiquid:
        """Fit model to data.

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
        from rheojax.core.parameters import ParameterOptimizer

        # Detect test mode
        test_mode = kwargs.get("test_mode", "oscillation")

        # Select prediction function
        if test_mode == "relaxation":
            predict_fn = self._predict_relaxation
        elif test_mode == "creep":
            predict_fn = self._predict_creep
        elif test_mode == "oscillation":
            predict_fn = self._predict_oscillation
        else:
            raise ValueError(f"Test mode '{test_mode}' not supported for FZLL model")

        # Set up optimizer
        optimizer = ParameterOptimizer(
            parameters=self.parameters, predict_fn=predict_fn, loss="mse"
        )

        # Fit parameters
        result = optimizer.fit(X, y, **kwargs)

        # Update parameters
        for name, value in result.items():
            self.parameters.set_value(name, value)

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

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c1, c2, alpha, beta, gamma, tau]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        c1 = params[0]
        c2 = params[1]
        alpha = params[2]
        beta = params[3]
        gamma = params[4]
        tau = params[5]

        # Fractional models default to relaxation mode
        # Call the _jax method directly
        return self._predict_relaxation_jax(X, c1, c2, alpha, beta, gamma, tau)

# Convenience alias
FZLL = FractionalZenerLiquidLiquid

__all__ = ["FractionalZenerLiquidLiquid", "FZLL"]
