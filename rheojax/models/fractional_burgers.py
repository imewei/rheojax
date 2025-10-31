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

jax, jnp = safe_import_jax()

from jax.scipy.special import gamma as jax_gamma

from rheojax.core.base import BaseModel
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e


@ModelRegistry.register("fractional_burgers")
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
            value=None,
            bounds=(1e-9, 1e3),
            units="1/Pa",
            description="Glassy compliance",
        )
        self.parameters.add(
            name="eta1",
            value=None,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Viscosity (Maxwell arm)",
        )
        self.parameters.add(
            name="Jk",
            value=None,
            bounds=(1e-9, 1e3),
            units="1/Pa",
            description="Kelvin compliance",
        )
        self.parameters.add(
            name="alpha",
            value=None,
            bounds=(0.0, 1.0),
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau_k",
            value=None,
            bounds=(1e-6, 1e6),
            units="s",
            description="Retardation time",
        )

    def _predict_creep(
        self,
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_creep(t, Jg, eta1, Jk, tau_k):
            tau_k_safe = tau_k + epsilon
            eta1_safe = eta1 + epsilon

            # Instantaneous compliance (elastic response)
            J_instant = Jg

            # Fractional viscous flow term: t^α / (η_1 * Γ(1+α))
            gamma_term = jax_gamma(1.0 + alpha_safe)
            J_flow = jnp.power(t, alpha_safe) / (eta1_safe * gamma_term)

            # Retardation term: J_k * (1 - E_α(-(t/τ_k)^α))
            z = -jnp.power(t / tau_k_safe, alpha_safe)
            ml_term = mittag_leffler_e(z, alpha_safe)
            J_retard = Jk * (1.0 - ml_term)

            # Total creep compliance
            J_t = J_instant + J_flow + J_retard

            return J_t

        return _compute_creep(t, Jg, eta1, Jk, tau_k)

    def _predict_relaxation(
        self,
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Jg, eta1, Jk, tau_k):
            tau_k_safe = tau_k + epsilon

            # Approximate using inverse relationship
            # G(0) ≈ 1/J_g (instantaneous modulus)
            G_inst = 1.0 / (Jg + epsilon)

            # Long-time decay (fractional Maxwell-like)
            # G(t) ~ t^(-α) at intermediate times
            G_decay = G_inst * jnp.power(t / tau_k_safe, -alpha_safe)

            # Smooth transition
            z = -jnp.power(t / tau_k_safe, alpha_safe)
            ml_term = mittag_leffler_e(z, alpha_safe)

            # Combine terms
            G_t = G_inst * ml_term + G_decay * (1.0 - ml_term)

            return G_t

        return _compute_relaxation(t, Jg, eta1, Jk, tau_k)

    def _predict_oscillation(
        self,
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np

        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_oscillation(omega, Jg, eta1, Jk, tau_k):
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

        return _compute_oscillation(omega, Jg, eta1, Jk, tau_k)

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray, **kwargs) -> FractionalBurgersModel:
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
        test_mode = kwargs.get("test_mode", "creep")

        # Select prediction function
        if test_mode == "relaxation":
            predict_fn = self._predict_relaxation
        elif test_mode == "creep":
            predict_fn = self._predict_creep
        elif test_mode == "oscillation":
            predict_fn = self._predict_oscillation
        else:
            raise ValueError(f"Test mode '{test_mode}' not supported for FBM")

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


# Convenience alias
FBM = FractionalBurgersModel

__all__ = ["FractionalBurgersModel", "FBM"]
