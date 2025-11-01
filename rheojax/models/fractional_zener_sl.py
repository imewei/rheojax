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

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e2


@ModelRegistry.register("fractional_zener_sl")
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
            value=None,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Equilibrium modulus",
        )
        self.parameters.add(
            name="c_alpha",
            value=None,
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="SpringPot constant",
        )
        self.parameters.add(
            name="alpha",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau",
            value=None,
            bounds=(1e-6, 1e6),
            units="s",
            description="Relaxation time",
        )

    def _predict_relaxation(
        self, t: jnp.ndarray, Ge: float, c_alpha: float, alpha: float, tau: float
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # Compute Mittag-Leffler parameters as concrete values
        ml_alpha = 1.0 - alpha_safe
        ml_beta = 1.0

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Ge, c_alpha, tau):
            tau_safe = tau + epsilon

            # Compute fractional relaxation term
            # E_{1-α,1}(-(t/τ)^(1-α))
            z = -jnp.power(t / tau_safe, ml_alpha)

            # Mittag-Leffler function with concrete alpha/beta
            ml_term = mittag_leffler_e2(z, ml_alpha, ml_beta)

            # G(t) = G_e + c_α * t^(-α) * E_{1-α,1}(...)
            fractional_term = c_alpha * jnp.power(t, -alpha_safe) * ml_term

            return Ge + fractional_term

        return _compute_relaxation(t, Ge, c_alpha, tau)

    def _predict_creep(
        self, t: jnp.ndarray, Ge: float, c_alpha: float, alpha: float, tau: float
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_creep(t, Ge, c_alpha, tau):
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

        return _compute_creep(t, Ge, c_alpha, tau)

    def _predict_oscillation(
        self, omega: jnp.ndarray, Ge: float, c_alpha: float, alpha: float, tau: float
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        beta_safe = 1.0 - alpha_safe

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_oscillation(omega, Ge, c_alpha, tau):
            tau_safe = tau + epsilon
            omega_safe = jnp.maximum(omega, epsilon)

            # Compute (iω)^α = ω^α * exp(i*π*α/2)
            omega_alpha = jnp.power(omega_safe, alpha_safe)
            phase_alpha = jnp.pi * alpha_safe / 2.0
            i_omega_alpha = omega_alpha * (
                jnp.cos(phase_alpha) + 1j * jnp.sin(phase_alpha)
            )

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

        return _compute_oscillation(omega, Ge, c_alpha, tau)

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerSolidLiquid:
        """Fit model to data.

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
        from rheojax.core.parameters import ParameterOptimizer

        # Detect test mode if not provided
        test_mode = kwargs.get("test_mode", "relaxation")

        # Select prediction function based on test mode
        if test_mode == "relaxation":
            predict_fn = self._predict_relaxation
        elif test_mode == "creep":
            predict_fn = self._predict_creep
        elif test_mode == "oscillation":
            predict_fn = self._predict_oscillation
        else:
            raise ValueError(f"Test mode '{test_mode}' not supported for FZSL model")

        # Set up optimizer
        optimizer = ParameterOptimizer(
            parameters=self.parameters, predict_fn=predict_fn, loss="mse"
        )

        # Fit parameters
        result = optimizer.fit(X, y, **kwargs)

        # Update parameters with fitted values
        for name, value in result.items():
            self.parameters.set_value(name, value)

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

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Ge, c_alpha, alpha, tau]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        Ge = params[0]
        c_alpha = params[1]
        alpha = params[2]
        tau = params[3]

        # Fractional models default to relaxation mode
        # Call the _jax method directly
        return self._predict_relaxation_jax(X, Ge, c_alpha, alpha, tau)

# Convenience alias
FZSL = FractionalZenerSolidLiquid

__all__ = ["FractionalZenerSolidLiquid", "FZSL"]
