"""Fractional Kelvin-Voigt Model (FKV).

This model consists of a spring and a SpringPot element in parallel. It describes
materials with solid-like behavior with power-law creep, typical of filled polymers
and soft solids.

Mathematical Description:
    Relaxation Modulus: G(t) = G_e + c_α t^(-α) / Γ(1-α)
    Complex Modulus: G*(ω) = G_e + c_α (iω)^α
    Creep Compliance: J(t) = (1/G_e) (1 - E_α(-(t/τ_ε)^α))

where τ_ε = (c_α/G_e)^(1/α) is a characteristic retardation time.

Parameters:
    Ge (float): Equilibrium modulus (Pa), bounds [1e-3, 1e9]
    c_alpha (float): SpringPot constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): Fractional order, bounds [0.0, 1.0]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Bagley, R. L., & Torvik, P. J. (1983). A theoretical basis for the application
      of fractional calculus to viscoelasticity. Journal of Rheology, 27(3), 201-210.
    - Makris, N., & Constantinou, M. C. (1991). Fractional-derivative Maxwell model
      for viscous dampers. Journal of Structural Engineering, 117(9), 2708-2724.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gamma as jax_gamma

from rheo.core.base import BaseModel, Parameter, ParameterSet
from rheo.core.data import RheoData
from rheo.core.registry import ModelRegistry
from rheo.utils.mittag_leffler import mittag_leffler_e


@ModelRegistry.register('fractional_kelvin_voigt')
class FractionalKelvinVoigt(BaseModel):
    """Fractional Kelvin-Voigt model: Spring and SpringPot in parallel.

    This model describes solid-like materials with power-law creep behavior,
    typical of filled polymers and soft solids.

    Attributes:
        parameters: ParameterSet with Ge, c_alpha, alpha

    Examples:
        >>> from rheo.models import FractionalKelvinVoigt
        >>> from rheo.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalKelvinVoigt()
        >>> model.parameters.set_value('Ge', 1e6)
        >>> model.parameters.set_value('c_alpha', 1e4)
        >>> model.parameters.set_value('alpha', 0.5)
        >>>
        >>> # Predict relaxation modulus
        >>> t = np.logspace(-3, 3, 50)
        >>> data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        >>> data.metadata['test_mode'] = 'relaxation'
        >>> G_t = model.predict(data)
        >>>
        >>> # Predict complex modulus
        >>> omega = np.logspace(-2, 2, 50)
        >>> data = RheoData(x=omega, y=np.zeros_like(omega), domain='frequency')
        >>> G_star = model.predict(data)
    """

    def __init__(self):
        """Initialize Fractional Kelvin-Voigt model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name='Ge',
            value=1e6,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Equilibrium modulus'
        )

        self.parameters.add(
            name='c_alpha',
            value=1e4,
            bounds=(1e-3, 1e9),
            units='Pa·s^α',
            description='SpringPot constant'
        )

        self.parameters.add(
            name='alpha',
            value=0.5,
            bounds=(0.0, 1.0),
            units='dimensionless',
            description='Fractional order'
        )

        self.fitted_ = False

    def _compute_tau_epsilon(self, Ge: float, c_alpha: float, alpha: float) -> float:
        """Compute characteristic retardation time.

        Args:
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Characteristic time τ_ε = (c_α/G_e)^(1/α)
        """
        epsilon = 1e-12
        alpha_safe = max(alpha, epsilon)
        return (c_alpha / Ge) ** (1.0 / alpha_safe)

    def _predict_relaxation_jax(
        self,
        t: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = G_e + c_α t^(-α) / Γ(1-α)

        Args:
            t: Time array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Ge, c_alpha):
            t_safe = jnp.maximum(t, epsilon)

            # Elastic part
            G_elastic = Ge

            # Viscous part: c_α t^(-α) / Γ(1-α)
            gamma_term = jax_gamma(1.0 - alpha_safe)
            G_viscous = c_alpha * (t_safe ** (-alpha_safe)) / gamma_term

            # Total relaxation modulus
            G_t = G_elastic + G_viscous

            return G_t

        return _compute_relaxation(t, Ge, c_alpha)

    def _predict_creep_jax(
        self,
        t: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/G_e) (1 - E_α(-(t/τ_ε)^α))

        where τ_ε = (c_α/G_e)^(1/α)

        Args:
            t: Time array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Creep compliance array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_creep(t, Ge, c_alpha):
            t_safe = jnp.maximum(t, epsilon)

            # Characteristic retardation time
            tau_epsilon = (c_alpha / Ge) ** (1.0 / alpha_safe)

            # Argument for Mittag-Leffler function
            z = -((t_safe / tau_epsilon) ** alpha_safe)

            # Compute E_α(z) with concrete alpha
            ml_value = mittag_leffler_e(z, alpha=alpha_safe)

            # Creep compliance
            J_t = (1.0 / Ge) * (1.0 - ml_value)

            return J_t

        return _compute_creep(t, Ge, c_alpha)

    def _predict_oscillation_jax(
        self,
        omega: jnp.ndarray,
        Ge: float,
        c_alpha: float,
        alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = G_e + c_α (iω)^α

        Args:
            omega: Angular frequency array
            Ge: Equilibrium modulus
            c_alpha: SpringPot constant
            alpha: Fractional order

        Returns:
            Complex modulus array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_oscillation(omega, Ge, c_alpha):
            omega_safe = jnp.maximum(omega, epsilon)

            # Elastic part
            G_elastic = Ge

            # SpringPot part: c_α (iω)^α = c_α |ω|^α exp(i α π/2)
            G_springpot = c_alpha * (omega_safe ** alpha_safe) * jnp.exp(1j * alpha_safe * jnp.pi / 2.0)

            # Complex modulus
            G_star = G_elastic + G_springpot

            return G_star

        return _compute_oscillation(omega, Ge, c_alpha)

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalKelvinVoigt:
        """Fit model parameters to data.

        Args:
            X: Input features (time or frequency)
            y: Target values (modulus or compliance)
            **kwargs: Additional fitting options

        Returns:
            self
        """
        # Placeholder for optimization implementation
        raise NotImplementedError("Parameter fitting will be implemented in optimization module")

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation.

        Args:
            X: RheoData object or array of x-values

        Returns:
            Predicted values
        """
        # Handle RheoData input
        if isinstance(X, RheoData):
            return self.predict_rheodata(X)

        # Handle raw array input (assume relaxation mode)
        x = jnp.asarray(X)
        Ge = self.parameters.get_value('Ge')
        c_alpha = self.parameters.get_value('c_alpha')
        alpha = self.parameters.get_value('alpha')

        result = self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
        return np.array(result)

    def predict_rheodata(self, rheo_data: RheoData, test_mode: Optional[str] = None) -> RheoData:
        """Predict response for RheoData.

        Args:
            rheo_data: Input RheoData with x values
            test_mode: Test mode ('relaxation', 'creep', 'oscillation')
                      If None, auto-detect from rheo_data

        Returns:
            RheoData with predicted y values
        """
        # Auto-detect test mode if not provided
        if test_mode is None:
            test_mode = rheo_data.test_mode

        # Get parameters
        Ge = self.parameters.get_value('Ge')
        c_alpha = self.parameters.get_value('c_alpha')
        alpha = self.parameters.get_value('alpha')

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == 'relaxation':
            y_pred = self._predict_relaxation_jax(x, Ge, c_alpha, alpha)
        elif test_mode == 'creep':
            y_pred = self._predict_creep_jax(x, Ge, c_alpha, alpha)
        elif test_mode == 'oscillation':
            y_pred = self._predict_oscillation_jax(x, Ge, c_alpha, alpha)
        else:
            raise ValueError(
                f"Unknown test mode: {test_mode}. "
                f"Must be 'relaxation', 'creep', or 'oscillation'"
            )

        # Create output RheoData
        result = RheoData(
            x=np.array(x),
            y=np.array(y_pred),
            x_units=rheo_data.x_units,
            y_units=rheo_data.y_units,
            domain=rheo_data.domain,
            metadata=rheo_data.metadata.copy()
        )

        return result

    def predict(self, X, test_mode: Optional[str] = None):
        """Predict response.

        Args:
            X: RheoData object or array of x-values
            test_mode: Test mode for prediction

        Returns:
            Predicted values (RheoData if input is RheoData, else array)
        """
        if isinstance(X, RheoData):
            return self.predict_rheodata(X, test_mode=test_mode)
        else:
            return self._predict(X)
