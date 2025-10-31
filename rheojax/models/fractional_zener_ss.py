"""Fractional Zener Solid-Solid (FZSS) Model.

This model combines two springs and one SpringPot, providing both
instantaneous and equilibrium elasticity with fractional relaxation.

Theory
------
The FZSS model consists of:
- Spring (G_e) in parallel with
- Series combination of spring (G_m) and SpringPot

Relaxation modulus:
    G(t) = G_e + G_m * E_α(-(t/τ_α)^α)

Complex modulus:
    G*(ω) = G_e + G_m / (1 + (iωτ_α)^(-α))

where E_α is the one-parameter Mittag-Leffler function.

Parameters
----------
Ge : float
    Equilibrium modulus (Pa), bounds [1e-3, 1e9]
Gm : float
    Maxwell arm modulus (Pa), bounds [1e-3, 1e9]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau_alpha : float
    Relaxation time (s^α), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Two springs in parallel (G = G_e + G_m)
- alpha → 1: Classical Zener solid with G(t) = G_e + G_m*exp(-t/τ)

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
from rheojax.utils.mittag_leffler import mittag_leffler_e


@ModelRegistry.register("fractional_zener_ss")
class FractionalZenerSolidSolid(BaseModel):
    """Fractional Zener Solid-Solid model.

    A fractional viscoelastic model with both instantaneous and
    equilibrium elasticity.

    Test Modes
    ----------
    - Relaxation: Supported
    - Creep: Supported
    - Oscillation: Supported
    - Rotation: Not supported (no steady-state flow)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalZenerSolidSolid
    >>>
    >>> # Create model
    >>> model = FractionalZenerSolidSolid()
    >>>
    >>> # Set parameters
    >>> model.set_params(Ge=1000.0, Gm=500.0, alpha=0.5, tau_alpha=1.0)
    >>>
    >>> # Predict relaxation modulus
    >>> t = jnp.logspace(-2, 2, 50)
    >>> G_t = model.predict(t)
    """

    def __init__(self):
        """Initialize Fractional Zener Solid-Solid model."""
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
            name="Gm",
            value=None,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Maxwell arm modulus",
        )
        self.parameters.add(
            name="alpha",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau_alpha",
            value=None,
            bounds=(1e-6, 1e6),
            units="s^α",
            description="Relaxation time",
        )

    def _predict_relaxation(
        self, t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        G(t) = G_e + G_m * E_α(-(t/τ_α)^α)

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Equilibrium modulus (Pa)
        Gm : float
            Maxwell arm modulus (Pa)
        alpha : float
            Fractional order
        tau_alpha : float
            Relaxation time (s^α)

        Returns
        -------
        jnp.ndarray
            Relaxation modulus G(t) (Pa)
        """
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Ge, Gm, tau_alpha):
            tau_alpha_safe = tau_alpha + epsilon

            # Compute argument: z = -(t/τ_α)^α
            z = -jnp.power(t / tau_alpha_safe, alpha_safe)

            # Mittag-Leffler function E_α(z) with concrete alpha
            ml_term = mittag_leffler_e(z, alpha_safe)

            # G(t) = G_e + G_m * E_α(-(t/τ_α)^α)
            return Ge + Gm * ml_term

        return _compute_relaxation(t, Ge, Gm, tau_alpha)

    def _predict_creep(
        self, t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        For FZSS, creep compliance is:
        J(t) = 1/(G_e + G_m) + (1/G_e - 1/(G_e + G_m)) * (1 - E_α(-(t/τ_α)^α))

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        Ge : float
            Equilibrium modulus (Pa)
        Gm : float
            Maxwell arm modulus (Pa)
        alpha : float
            Fractional order
        tau_alpha : float
            Relaxation time (s^α)

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
        def _compute_creep(t, Ge, Gm, tau_alpha):
            tau_alpha_safe = tau_alpha + epsilon

            # Instantaneous and equilibrium compliances
            G_total = Ge + Gm + epsilon
            J_inst = 1.0 / G_total
            J_eq = 1.0 / (Ge + epsilon)

            # Compute argument: z = -(t/τ_α)^α
            z = -jnp.power(t / tau_alpha_safe, alpha_safe)

            # Mittag-Leffler function with concrete alpha
            ml_term = mittag_leffler_e(z, alpha_safe)

            # J(t) = J_inst + (J_eq - J_inst) * (1 - E_α(-t^α/τ_α))
            return J_inst + (J_eq - J_inst) * (1.0 - ml_term)

        return _compute_creep(t, Ge, Gm, tau_alpha)

    def _predict_oscillation(
        self, omega: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        G*(ω) = G_e + G_m / (1 + (iωτ_α)^(-α))

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        Ge : float
            Equilibrium modulus (Pa)
        Gm : float
            Maxwell arm modulus (Pa)
        alpha : float
            Fractional order
        tau_alpha : float
            Relaxation time (s^α)

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
        def _compute_oscillation(omega, Ge, Gm, tau_alpha):
            tau_alpha_safe = tau_alpha + epsilon

            # Compute (iω)^(-α) = ω^(-α) * exp(-i*π*α/2)
            omega_neg_alpha = jnp.power(omega, -alpha_safe)
            phase = -jnp.pi * alpha_safe / 2.0

            # (iω)^(-α) in complex form
            i_omega_neg_alpha = omega_neg_alpha * (jnp.cos(phase) + 1j * jnp.sin(phase))

            # Denominator: 1 + (iωτ_α)^(-α) = 1 + τ_α^(-α) * (iω)^(-α)
            tau_neg_alpha = jnp.power(tau_alpha_safe, -alpha_safe)
            denominator = 1.0 + tau_neg_alpha * i_omega_neg_alpha

            # Maxwell arm contribution: G_m / (1 + (iωτ_α)^(-α))
            maxwell_term = Gm / denominator

            # Total complex modulus
            G_star = Ge + maxwell_term

            # Extract storage and loss moduli
            G_prime = jnp.real(G_star)
            G_double_prime = jnp.imag(G_star)

            return jnp.stack([G_prime, G_double_prime], axis=-1)

        return _compute_oscillation(omega, Ge, Gm, tau_alpha)

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerSolidSolid:
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
        test_mode = kwargs.get("test_mode", "relaxation")

        # Select prediction function
        if test_mode == "relaxation":
            predict_fn = self._predict_relaxation
        elif test_mode == "creep":
            predict_fn = self._predict_creep
        elif test_mode == "oscillation":
            predict_fn = self._predict_oscillation
        else:
            raise ValueError(f"Test mode '{test_mode}' not supported for FZSS model")

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
        Ge = self.parameters.get_value("Ge")
        Gm = self.parameters.get_value("Gm")
        alpha = self.parameters.get_value("alpha")
        tau_alpha = self.parameters.get_value("tau_alpha")

        # Auto-detect test mode based on input characteristics
        # NOTE: This is a heuristic - explicit test_mode is recommended
        # Default to relaxation for time-domain data
        # Oscillation should typically use RheoData with domain='frequency'
        return self._predict_relaxation(X, Ge, Gm, alpha, tau_alpha)


# Convenience alias
FZSS = FractionalZenerSolidSolid

__all__ = ["FractionalZenerSolidSolid", "FZSS"]
