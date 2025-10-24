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

import jax
import jax.numpy as jnp

from rheo.core.base import BaseModel
from rheo.core.parameters import Parameter, ParameterSet
from rheo.core.registry import ModelRegistry
from rheo.utils.mittag_leffler import mittag_leffler_e


@ModelRegistry.register('fractional_poynting_thomson')
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
    >>> from rheo.models import FractionalPoyntingThomson
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
        self.parameters.add(Parameter(
            name='Ge',
            value=None,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Instantaneous modulus'
        ))
        self.parameters.add(Parameter(
            name='Gk',
            value=None,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Retarded modulus'
        ))
        self.parameters.add(Parameter(
            name='alpha',
            value=None,
            bounds=(0.0, 1.0),
            units='',
            description='Fractional order'
        ))
        self.parameters.add(Parameter(
            name='tau',
            value=None,
            bounds=(1e-6, 1e6),
            units='s',
            description='Retardation time'
        ))

    def _predict_creep(self, t: jnp.ndarray, Ge: float, Gk: float,
                      alpha: float, tau: float) -> jnp.ndarray:
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_creep(t, Ge, Gk, tau):
            tau_safe = tau + epsilon

            # Instantaneous compliance (elastic response)
            J_inst = 1.0 / (Ge + epsilon)

            # Retarded compliance amplitude
            J_retard_amp = 1.0 / (Gk + epsilon)

            # Compute argument: z = -(t/τ)^α
            z = -jnp.power(t / tau_safe, alpha_safe)

            # Mittag-Leffler function E_α(z) with concrete alpha
            ml_term = mittag_leffler_e(z, alpha_safe)

            # J(t) = 1/G_e + (1/G_k) * (1 - E_α(-(t/τ)^α))
            J_t = J_inst + J_retard_amp * (1.0 - ml_term)

            return J_t

        return _compute_creep(t, Ge, Gk, tau)

    def _predict_relaxation(self, t: jnp.ndarray, Ge: float, Gk: float,
                           alpha: float, tau: float) -> jnp.ndarray:
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Ge, Gk, tau):
            tau_safe = tau + epsilon

            # Compute transition function
            z = -jnp.power(t / tau_safe, alpha_safe)
            ml_term = mittag_leffler_e(z, alpha_safe)

            # Instantaneous modulus
            G_inst = Ge

            # Equilibrium modulus (series combination)
            G_eq = (Ge * Gk) / (Ge + Gk + epsilon)

            # Interpolate using Mittag-Leffler decay
            # G(t) = G_eq + (G_inst - G_eq) * E_α(-(t/τ)^α)
            G_t = G_eq + (G_inst - G_eq) * ml_term

            return G_t

        return _compute_relaxation(t, Ge, Gk, tau)

    def _predict_oscillation(self, omega: jnp.ndarray, Ge: float, Gk: float,
                            alpha: float, tau: float) -> jnp.ndarray:
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
        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        epsilon = 1e-12
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_oscillation(omega, Ge, Gk, tau):
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

        return _compute_oscillation(omega, Ge, Gk, tau)

    def _fit(self, X: jnp.ndarray, y: jnp.ndarray, **kwargs) -> FractionalPoyntingThomson:
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
        from rheo.core.parameters import ParameterOptimizer

        # Detect test mode
        test_mode = kwargs.get('test_mode', 'creep')

        # Select prediction function
        if test_mode == 'relaxation':
            predict_fn = self._predict_relaxation
        elif test_mode == 'creep':
            predict_fn = self._predict_creep
        elif test_mode == 'oscillation':
            predict_fn = self._predict_oscillation
        else:
            raise ValueError(f"Test mode '{test_mode}' not supported for FPT model")

        # Set up optimizer
        optimizer = ParameterOptimizer(
            parameters=self.parameters,
            predict_fn=predict_fn,
            loss='mse'
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
        Ge = params['Ge']
        Gk = params['Gk']
        alpha = params['alpha']
        tau = params['tau']

        # Auto-detect test mode
        if jnp.all(X > 0) and len(X) > 1:
            log_range = jnp.log10(jnp.max(X)) - jnp.log10(jnp.min(X) + 1e-12)
            if log_range > 3:
                return self._predict_oscillation(X, Ge, Gk, alpha, tau)

        # Default to creep (primary mode for FPT)
        return self._predict_creep(X, Ge, Gk, alpha, tau)


# Convenience alias
FPT = FractionalPoyntingThomson

__all__ = ['FractionalPoyntingThomson', 'FPT']
