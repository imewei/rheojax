"""Zener (Standard Linear Solid) viscoelastic model.

The Zener model, also known as the Standard Linear Solid (SLS), consists of
a Maxwell element (spring G_m and dashpot eta in series) in parallel with
an equilibrium spring G_e.

Theory:
    - Total modulus: G_total = G_e + G_m
    - Relaxation modulus: G(t) = G_e + G_m * exp(-t/tau) where tau = eta/G_m
    - Complex modulus: G*(omega) = G_e + G_m*(omega*tau)^2/(1+(omega*tau)^2) + i*G_m*omega*tau/(1+(omega*tau)^2)
    - Creep compliance: J(t) = 1/(G_e+G_m) + (G_m/(G_e*(G_e+G_m))) * (1 - exp(-t/tau_c))
      where tau_c = eta * (G_e + G_m) / (G_e * G_m)
    - Steady shear viscosity: eta(gamma_dot) = eta (constant)

References:
    - Ferry, J. D. (1980). Viscoelastic properties of polymers.
    - Tschoegl, N. W. (1989). The phenomenological theory of linear viscoelastic behavior.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional

from rheo.core.base import BaseModel
from rheo.core.parameters import Parameter, ParameterSet
from rheo.core.data import RheoData
from rheo.core.test_modes import TestMode, detect_test_mode
from rheo.core.registry import ModelRegistry


@ModelRegistry.register('zener')
class Zener(BaseModel):
    """Zener (Standard Linear Solid) viscoelastic model.

    The Zener model consists of a Maxwell element (spring G_m and dashpot eta)
    in parallel with an equilibrium spring G_e. This provides both instantaneous
    elastic response and time-dependent relaxation to a finite equilibrium modulus.

    Parameters:
        Ge (float): Equilibrium modulus in Pa, range [1e-3, 1e9], default 1e4
        Gm (float): Maxwell modulus in Pa, range [1e-3, 1e9], default 1e5
        eta (float): Viscosity in Pa·s, range [1e-6, 1e12], default 1e3

    Supported test modes:
        - Relaxation: Stress relaxation under constant strain
        - Creep: Strain development under constant stress
        - Oscillation: Small amplitude oscillatory shear (SAOS)
        - Rotation: Steady shear flow

    Example:
        >>> from rheo.models.zener import Zener
        >>> from rheo.core.data import RheoData
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = Zener()
        >>> model.parameters.set_value('Ge', 1e4)
        >>> model.parameters.set_value('Gm', 1e5)
        >>> model.parameters.set_value('eta', 1e3)
        >>>
        >>> # Predict relaxation
        >>> t = jnp.linspace(0.01, 10, 100)
        >>> data = RheoData(x=t, y=jnp.zeros_like(t), domain='time')
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize Zener model with default parameters."""
        super().__init__()

        # Define parameters with physical bounds
        self.parameters = ParameterSet()
        self.parameters.add(
            name='Ge',
            value=1e4,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Equilibrium modulus'
        )
        self.parameters.add(
            name='Gm',
            value=1e5,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Maxwell modulus'
        )
        self.parameters.add(
            name='eta',
            value=1e3,
            bounds=(1e-6, 1e12),
            units='Pa·s',
            description='Viscosity'
        )

        self.fitted_ = False

    def _fit(self, X, y, **kwargs):
        """Fit Zener model to data.

        Args:
            X: RheoData object or independent variable array
            y: Dependent variable array (if X is not RheoData)
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        from rheo.utils.optimization import nlsq_optimize, create_least_squares_objective

        # Handle RheoData input
        if isinstance(X, RheoData):
            rheo_data = X
            x_data = jnp.array(rheo_data.x)
            y_data = jnp.array(rheo_data.y)
            test_mode = rheo_data.test_mode
        else:
            x_data = jnp.array(X)
            y_data = jnp.array(y)
            test_mode = kwargs.get('test_mode', TestMode.RELAXATION)

        # Create objective function with stateless predictions
        def model_fn(x, params):
            """Model function for optimization (stateless)."""
            Ge, Gm, eta = params[0], params[1], params[2]

            # Direct prediction based on test mode (stateless)
            if test_mode == TestMode.RELAXATION:
                return self._predict_relaxation(x, Ge, Gm, eta)
            elif test_mode == TestMode.CREEP:
                return self._predict_creep(x, Ge, Gm, eta)
            elif test_mode == TestMode.OSCILLATION:
                return self._predict_oscillation(x, Ge, Gm, eta)
            elif test_mode == TestMode.ROTATION:
                return self._predict_rotation(x, Ge, Gm, eta)
            else:
                raise ValueError(f"Unsupported test mode: {test_mode}")

        objective = create_least_squares_objective(model_fn, x_data, y_data, normalize=True)

        # Optimize
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get('use_jax', True),
            method=kwargs.get('method', 'auto'),
            max_iter=kwargs.get('max_iter', 1000)
        )

        self.fitted_ = True
        return self

    def _predict(self, X):
        """Predict response based on input data.

        Args:
            X: RheoData object or independent variable array

        Returns:
            Predicted values as JAX array
        """
        # Handle RheoData input
        if isinstance(X, RheoData):
            rheo_data = X
            test_mode = detect_test_mode(rheo_data)
            x_data = jnp.array(rheo_data.x)
        else:
            x_data = jnp.array(X)
            test_mode = TestMode.RELAXATION  # Default

        # Get parameter values
        Ge = self.parameters.get_value('Ge')
        Gm = self.parameters.get_value('Gm')
        eta = self.parameters.get_value('eta')

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(x_data, Ge, Gm, eta)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(x_data, Ge, Gm, eta)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(x_data, Ge, Gm, eta)
        elif test_mode == TestMode.ROTATION:
            return self._predict_rotation(x_data, Ge, Gm, eta)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    @staticmethod
    @jax.jit
    def _predict_relaxation(t: jnp.ndarray, Ge: float, Gm: float, eta: float) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Theory: G(t) = G_e + G_m * exp(-t/tau) where tau = eta/G_m

        Args:
            t: Time array (s)
            Ge: Equilibrium modulus (Pa)
            Gm: Maxwell modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Relaxation modulus G(t) in Pa
        """
        tau = eta / Gm  # Relaxation time
        return Ge + Gm * jnp.exp(-t / tau)

    @staticmethod
    @jax.jit
    def _predict_creep(t: jnp.ndarray, Ge: float, Gm: float, eta: float) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Theory: J(t) = 1/(G_e+G_m) + (G_m/(G_e*(G_e+G_m))) * (1 - exp(-t/tau_c))
        where tau_c = eta * (G_e + G_m) / (G_e * G_m) is the creep retardation time

        Args:
            t: Time array (s)
            Ge: Equilibrium modulus (Pa)
            Gm: Maxwell modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Creep compliance J(t) in 1/Pa
        """
        G_total = Ge + Gm
        J_inf = 1.0 / G_total  # Instantaneous compliance
        tau_c = eta * G_total / (Ge * Gm)  # Retardation time

        # Creep compliance
        return J_inf + (Gm / (Ge * G_total)) * (1.0 - jnp.exp(-t / tau_c))

    @staticmethod
    @jax.jit
    def _predict_oscillation(omega: jnp.ndarray, Ge: float, Gm: float, eta: float) -> jnp.ndarray:
        """Predict complex modulus G*(omega).

        Theory:
            G'(omega) = G_e + G_m * (omega*tau)^2 / (1 + (omega*tau)^2)
            G''(omega) = G_m * omega*tau / (1 + (omega*tau)^2)
            G*(omega) = G'(omega) + i*G''(omega)

        Args:
            omega: Angular frequency array (rad/s)
            Ge: Equilibrium modulus (Pa)
            Gm: Maxwell modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Complex modulus G*(omega) in Pa
        """
        tau = eta / Gm  # Relaxation time
        omega_tau = omega * tau
        omega_tau_sq = omega_tau ** 2

        # Storage modulus G'
        G_prime = Ge + Gm * omega_tau_sq / (1.0 + omega_tau_sq)

        # Loss modulus G''
        G_double_prime = Gm * omega_tau / (1.0 + omega_tau_sq)

        # Complex modulus
        return G_prime + 1j * G_double_prime

    @staticmethod
    @jax.jit
    def _predict_rotation(gamma_dot: jnp.ndarray, Ge: float, Gm: float, eta: float) -> jnp.ndarray:
        """Predict steady shear viscosity eta(gamma_dot).

        Theory: eta(gamma_dot) = eta (constant, Newtonian behavior)

        Args:
            gamma_dot: Shear rate array (1/s)
            Ge: Equilibrium modulus (Pa) - not used but kept for interface consistency
            Gm: Maxwell modulus (Pa) - not used but kept for interface consistency
            eta: Viscosity (Pa·s)

        Returns:
            Viscosity eta in Pa·s (constant array)
        """
        return eta * jnp.ones_like(gamma_dot)

    def get_relaxation_time(self) -> float:
        """Get characteristic relaxation time tau = eta/G_m.

        Returns:
            Relaxation time in seconds
        """
        Gm = self.parameters.get_value('Gm')
        eta = self.parameters.get_value('eta')
        return eta / Gm

    def get_retardation_time(self) -> float:
        """Get characteristic retardation time for creep.

        Theory: tau_c = eta * (G_e + G_m) / (G_e * G_m)

        Returns:
            Retardation time in seconds
        """
        Ge = self.parameters.get_value('Ge')
        Gm = self.parameters.get_value('Gm')
        eta = self.parameters.get_value('eta')
        return eta * (Ge + Gm) / (Ge * Gm)

    def __repr__(self) -> str:
        """String representation of Zener model."""
        Ge = self.parameters.get_value('Ge')
        Gm = self.parameters.get_value('Gm')
        eta = self.parameters.get_value('eta')
        tau = self.get_relaxation_time()
        return f"Zener(Ge={Ge:.2e} Pa, Gm={Gm:.2e} Pa, eta={eta:.2e} Pa·s, tau={tau:.2e} s)"


__all__ = ['Zener']
