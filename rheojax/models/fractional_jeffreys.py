"""Fractional Jeffreys Model (FJM).

This model combines two dashpots and one SpringPot in a parallel-series
arrangement, providing viscous flow with fractional relaxation behavior.

Theory
------
The FJM model consists of:
- Dashpot (η_1) in parallel with
- Series combination of dashpot (η_2) and SpringPot

Relaxation modulus:
    G(t) = (η_1/τ_1) * t^(-α) * E_{1-α,1-α}(-(t/τ_1)^(1-α))

where:
- τ_1 = η_2 / characteristic_modulus (relaxation time)
- E_{α,β} is the two-parameter Mittag-Leffler function

Complex modulus:
    G*(ω) = η_1(iω) * [1 + (iωτ_2)^α] / [1 + (iωτ_1)^α]

Parameters
----------
eta1 : float
    First viscosity (Pa·s), bounds [1e-6, 1e12]
eta2 : float
    Second viscosity (Pa·s), bounds [1e-6, 1e12]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau1 : float
    Relaxation time (s), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Two dashpots in parallel (Newtonian fluid)
- alpha → 1: Classical Jeffreys model (viscoelastic liquid)

References
----------
- Mainardi, F. (2010). Fractional Calculus and Waves in Linear Viscoelasticity
- Jeffreys, H. (1929). The Earth
- Friedrich, C. (1991). Rheol. Acta 30, 151-158
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()

from jax.scipy.special import gamma as jax_gamma

from rheojax.core.base import BaseModel
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e2


@ModelRegistry.register("fractional_jeffreys")
class FractionalJeffreysModel(BaseModel):
    """Fractional Jeffreys model.

    A fractional viscoelastic liquid model combining viscous flow
    with fractional relaxation behavior.

    Test Modes
    ----------
    - Relaxation: Supported
    - Creep: Supported
    - Oscillation: Supported
    - Rotation: Supported (viscous flow at low frequencies)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalJeffreysModel
    >>>
    >>> # Create model
    >>> model = FractionalJeffreysModel()
    >>>
    >>> # Set parameters
    >>> model.set_params(eta1=1000.0, eta2=500.0, alpha=0.5, tau1=1.0)
    >>>
    >>> # Predict relaxation modulus
    >>> t = jnp.logspace(-2, 2, 50)
    >>> G_t = model.predict(t)
    """

    def __init__(self):
        """Initialize Fractional Jeffreys model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="eta1",
            value=None,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="First viscosity",
        )
        self.parameters.add(
            name="eta2",
            value=None,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Second viscosity",
        )
        self.parameters.add(
            name="alpha",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau1",
            value=None,
            bounds=(1e-6, 1e6),
            units="s",
            description="Relaxation time",
        )

    def _predict_relaxation(
        self, t: jnp.ndarray, eta1: float, eta2: float, alpha: float, tau1: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        G(t) = (η_1/τ_1) * t^(-α) * E_{1-α,1-α}(-(t/τ_1)^(1-α))

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        eta1 : float
            First viscosity (Pa·s)
        eta2 : float
            Second viscosity (Pa·s)
        alpha : float
            Fractional order
        tau1 : float
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
        ml_beta = 1.0 - alpha_safe

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, eta1, eta2, tau1):
            tau1_safe = tau1 + epsilon
            eta1_safe = eta1 + epsilon

            # Compute fractional relaxation term
            # E_{1-α,1-α}(-(t/τ_1)^(1-α))
            z = -jnp.power(t / tau1_safe, ml_alpha)

            # Two-parameter Mittag-Leffler function with concrete α and β
            ml_term = mittag_leffler_e2(z, ml_alpha, ml_beta)

            # G(t) = (η_1/τ_1) * t^(-α) * E_{1-α,1-α}(-(t/τ_1)^(1-α))
            prefactor = eta1_safe / tau1_safe
            G_t = prefactor * jnp.power(t, -alpha_safe) * ml_term

            return G_t

        return _compute_relaxation(t, eta1, eta2, tau1)

    def _predict_creep(
        self, t: jnp.ndarray, eta1: float, eta2: float, alpha: float, tau1: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        For Jeffreys model, creep shows unbounded flow behavior.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        eta1 : float
            First viscosity (Pa·s)
        eta2 : float
            Second viscosity (Pa·s)
        alpha : float
            Fractional order
        tau1 : float
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
        def _compute_creep(t, eta1, eta2, tau1):
            tau1_safe = tau1 + epsilon
            eta1_safe = eta1 + epsilon
            eta2_safe = eta2 + epsilon

            # For liquid-like behavior: J(t) ~ t/η_eff at long times
            # Effective viscosity combines both dashpots
            eta_eff = (eta1_safe * eta2_safe) / (eta1_safe + eta2_safe)

            # Short time: elastic-like response
            # Approximate using SpringPot behavior
            J_short = (
                jnp.power(t, alpha_safe)
                * jax_gamma(1.0 + alpha_safe)
                / (eta1_safe * tau1_safe**alpha_safe)
            )

            # Long time: Newtonian flow
            J_long = t / eta_eff

            # Crossover around tau1
            weight = 1.0 - jnp.exp(-t / tau1_safe)
            J_t = J_short * (1.0 - weight) + J_long * weight

            return J_t

        return _compute_creep(t, eta1, eta2, tau1)

    def _predict_oscillation(
        self, omega: jnp.ndarray, eta1: float, eta2: float, alpha: float, tau1: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        G*(ω) = η_1(iω) * [1 + (iωτ_2)^α] / [1 + (iωτ_1)^α]

        where τ_2 = η_2/η_1 * τ_1

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        eta1 : float
            First viscosity (Pa·s)
        eta2 : float
            Second viscosity (Pa·s)
        alpha : float
            Fractional order
        tau1 : float
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

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_oscillation(omega, eta1, eta2, tau1):
            tau1_safe = tau1 + epsilon
            eta1_safe = eta1 + epsilon
            eta2_safe = eta2 + epsilon

            # Second time constant
            tau2 = (eta2_safe / eta1_safe) * tau1_safe

            # Compute (iω)
            i_omega = 1j * omega

            # Compute (iωτ_1)^α
            omega_tau1_alpha = jnp.power(omega * tau1_safe, alpha_safe)
            phase1 = jnp.pi * alpha_safe / 2.0
            i_omega_tau1_alpha = omega_tau1_alpha * (
                jnp.cos(phase1) + 1j * jnp.sin(phase1)
            )

            # Compute (iωτ_2)^α
            omega_tau2_alpha = jnp.power(omega * tau2, alpha_safe)
            phase2 = jnp.pi * alpha_safe / 2.0
            i_omega_tau2_alpha = omega_tau2_alpha * (
                jnp.cos(phase2) + 1j * jnp.sin(phase2)
            )

            # Complex modulus: G*(ω) = η_1(iω) * [1 + (iωτ_2)^α] / [1 + (iωτ_1)^α]
            numerator = 1.0 + i_omega_tau2_alpha
            denominator = 1.0 + i_omega_tau1_alpha

            G_star = eta1_safe * i_omega * (numerator / denominator)

            # Extract storage and loss moduli
            G_prime = jnp.real(G_star)
            G_double_prime = jnp.imag(G_star)

            return jnp.stack([G_prime, G_double_prime], axis=-1)

        return _compute_oscillation(omega, eta1, eta2, tau1)

    def _predict_rotation(
        self,
        gamma_dot: jnp.ndarray,
        eta1: float,
        eta2: float,
        alpha: float,
        tau1: float,
    ) -> jnp.ndarray:
        """Predict steady shear viscosity η(γ̇).

        For Jeffreys model at steady state:
        η = η_1 (approximately, since it's a liquid)

        Parameters
        ----------
        gamma_dot : jnp.ndarray
            Shear rate array (1/s)
        eta1 : float
            First viscosity (Pa·s)
        eta2 : float
            Second viscosity (Pa·s)
        alpha : float
            Fractional order
        tau1 : float
            Relaxation time (s)

        Returns
        -------
        jnp.ndarray
            Viscosity η (Pa·s)
        """
        # At steady state, Jeffreys model behaves as Newtonian
        # with effective viscosity dominated by parallel dashpot
        eta1 = eta1 + 1e-12

        # Constant viscosity (Newtonian behavior)
        eta = jnp.full_like(gamma_dot, eta1)

        return eta

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray, **kwargs) -> FractionalJeffreysModel:
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
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )
        from rheojax.core.test_modes import TestMode

        # Detect test mode
        test_mode_str = kwargs.get("test_mode", "relaxation")

        # Convert string to TestMode enum
        if isinstance(test_mode_str, str):
            test_mode_map = {
                "relaxation": TestMode.RELAXATION,
                "creep": TestMode.CREEP,
                "oscillation": TestMode.OSCILLATION,
                "rotation": TestMode.ROTATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.RELAXATION)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        # Create stateless model function for optimization
        def model_fn(x, params):
            """Model function for optimization (stateless)."""
            eta1, eta2, alpha, tau1 = params[0], params[1], params[2], params[3]

            # Direct prediction based on test mode (stateless)
            if test_mode == TestMode.RELAXATION:
                return self._predict_relaxation(x, eta1, eta2, alpha, tau1)
            elif test_mode == TestMode.CREEP:
                return self._predict_creep(x, eta1, eta2, alpha, tau1)
            elif test_mode == TestMode.OSCILLATION:
                return self._predict_oscillation(x, eta1, eta2, alpha, tau1)
            elif test_mode == TestMode.ROTATION:
                return self._predict_rotation(x, eta1, eta2, alpha, tau1)
            else:
                raise ValueError(f"Unsupported test mode: {test_mode}")

        # Create objective function
        objective = create_least_squares_objective(
            model_fn, jnp.array(X), jnp.array(y), normalize=True
        )

        # Optimize using NLSQ TRF
        nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 1000),
        )

        self.fitted_ = True
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
        eta1 = params["eta1"]
        eta2 = params["eta2"]
        alpha = params["alpha"]
        tau1 = params["tau1"]

        # Auto-detect test mode
        if jnp.all(X > 0) and len(X) > 1:
            log_range = jnp.log10(jnp.max(X)) - jnp.log10(jnp.min(X) + 1e-12)
            if log_range > 3:
                return self._predict_oscillation(X, eta1, eta2, alpha, tau1)

        # Default to relaxation
        return self._predict_relaxation(X, eta1, eta2, alpha, tau1)

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [eta1, eta2, alpha, tau1]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        eta1 = params[0]
        eta2 = params[1]
        alpha = params[2]
        tau1 = params[3]

        # Use test_mode from last fit if available, otherwise default to RELAXATION
        test_mode = getattr(self, '_test_mode', TestMode.RELAXATION)

        # Call appropriate prediction function based on test mode
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, eta1, eta2, alpha, tau1)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, eta1, eta2, alpha, tau1)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(X, eta1, eta2, alpha, tau1)
        elif test_mode == TestMode.ROTATION:
            return self._predict_rotation(X, eta1, eta2, alpha, tau1)
        else:
            # Default to relaxation mode for Jeffreys model
            return self._predict_relaxation(X, eta1, eta2, alpha, tau1)

# Convenience alias
FJM = FractionalJeffreysModel

__all__ = ["FractionalJeffreysModel", "FJM"]
