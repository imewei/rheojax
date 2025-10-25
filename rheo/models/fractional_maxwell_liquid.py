"""Fractional Maxwell Liquid (FML) model.

This model consists of a spring in series with a SpringPot element. It captures
the behavior of materials with elastic response at short times and power-law
relaxation at long times, typical of polymer melts and concentrated solutions.

Mathematical Description:
    Relaxation Modulus: G(t) = G_m t^(-α) E_{1-α,1-α}(-t^(1-α)/τ_α)
    Complex Modulus: G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)
    Creep Compliance: J(t) = (1/G_m) + (t^α)/(G_m τ_α^α) E_{α,1+α}(-(t/τ_α)^α)

Parameters:
    Gm (float): Maxwell modulus (Pa), bounds [1e-3, 1e9]
    alpha (float): Power-law exponent, bounds [0.0, 1.0]
    tau_alpha (float): Relaxation time (s^α), bounds [1e-6, 1e6]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Friedrich, C. (1991). Relaxation and retardation functions of the Maxwell model
      with fractional derivatives. Rheologica Acta, 30(2), 151-158.
    - Schiessel, H., Metzler, R., Blumen, A., & Nonnenmacher, T. F. (1995). Generalized
      viscoelastic models: their fractional equations with solutions. Journal of Physics
      A: Mathematical and General, 28(23), 6567.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from rheo.core.base import BaseModel, Parameter, ParameterSet
from rheo.core.data import RheoData
from rheo.core.registry import ModelRegistry
from rheo.utils.mittag_leffler import mittag_leffler_e2


@ModelRegistry.register('fractional_maxwell_liquid')
class FractionalMaxwellLiquid(BaseModel):
    """Fractional Maxwell Liquid model: Spring in series with SpringPot.

    This model describes materials with elastic response at short times and
    power-law relaxation at long times, such as polymer melts.

    Attributes:
        parameters: ParameterSet with Gm, alpha, tau_alpha

    Examples:
        >>> from rheo.models import FractionalMaxwellLiquid
        >>> from rheo.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalMaxwellLiquid()
        >>> model.parameters.set_value('Gm', 1e6)
        >>> model.parameters.set_value('alpha', 0.7)
        >>> model.parameters.set_value('tau_alpha', 1.0)
        >>>
        >>> # Predict relaxation modulus
        >>> t = np.logspace(-3, 3, 50)
        >>> data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        >>> data.metadata['test_mode'] = 'relaxation'
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize Fractional Maxwell Liquid model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name='Gm',
            value=1e6,
            bounds=(1e-3, 1e9),
            units='Pa',
            description='Maxwell modulus'
        )

        self.parameters.add(
            name='alpha',
            value=0.5,
            bounds=(0.0, 1.0),
            units='dimensionless',
            description='Power-law exponent'
        )

        self.parameters.add(
            name='tau_alpha',
            value=1.0,
            bounds=(1e-6, 1e6),
            units='s^α',
            description='Relaxation time'
        )

        self.fitted_ = False

    def _predict_relaxation_jax(
        self,
        t: jnp.ndarray,
        Gm: float,
        alpha: float,
        tau_alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = G_m t^(-α) E_{1-α,1-α}(-t^(1-α)/τ_α)

        Args:
            t: Time array
            Gm: Maxwell modulus
            alpha: Power-law exponent
            tau_alpha: Relaxation time

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # Compute Mittag-Leffler parameters as concrete values
        ml_alpha = 1.0 - alpha_safe
        ml_beta = 1.0 - alpha_safe

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_relaxation(t, Gm, tau_alpha):
            t_safe = jnp.maximum(t, epsilon)
            tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

            # Compute argument for Mittag-Leffler function
            z = -(t_safe ** (1.0 - alpha_safe)) / tau_alpha_safe

            # Compute E_{1-α,1-α}(z) with concrete alpha/beta
            ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

            # Compute G(t)
            G_t = Gm * (t_safe ** (-alpha_safe)) * ml_value

            return G_t

        return _compute_relaxation(t, Gm, tau_alpha)

    def _predict_creep_jax(
        self,
        t: jnp.ndarray,
        Gm: float,
        alpha: float,
        tau_alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/G_m) + (t^α)/(G_m τ_α^α) E_{α,1+α}(-(t/τ_α)^α)

        Args:
            t: Time array
            Gm: Maxwell modulus
            alpha: Power-law exponent
            tau_alpha: Relaxation time

        Returns:
            Creep compliance array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha BEFORE JIT to make it concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))

        # Compute Mittag-Leffler parameters as concrete values
        ml_alpha = alpha_safe
        ml_beta = 1.0 + alpha_safe

        # JIT-compiled inner function with concrete alpha
        @jax.jit
        def _compute_creep(t, Gm, tau_alpha):
            t_safe = jnp.maximum(t, epsilon)
            tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

            # Instantaneous compliance (elastic part)
            J_instant = 1.0 / Gm

            # Time-dependent part with Mittag-Leffler function
            z = -((t_safe / tau_alpha_safe) ** alpha_safe)

            # Compute E_{α,1+α}(z) with concrete alpha/beta
            ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

            # Creep compliance
            J_t = J_instant + (t_safe ** alpha_safe) / (Gm * (tau_alpha_safe ** alpha_safe)) * ml_value

            return J_t

        return _compute_creep(t, Gm, tau_alpha)

    def _predict_oscillation_jax(
        self,
        omega: jnp.ndarray,
        Gm: float,
        alpha: float,
        tau_alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)

        Args:
            omega: Angular frequency array
            Gm: Maxwell modulus
            alpha: Power-law exponent
            tau_alpha: Relaxation time

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
        def _compute_oscillation(omega, Gm, tau_alpha):
            omega_safe = jnp.maximum(omega, epsilon)
            tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

            # Complex frequency term
            i_omega = 1j * omega_safe

            # (iωτ_α)^α = |ωτ_α|^α * exp(i α π/2)
            omega_tau = omega_safe * tau_alpha_safe
            i_omega_tau_alpha = (omega_tau ** alpha_safe) * jnp.exp(1j * alpha_safe * jnp.pi / 2.0)

            # Complex modulus
            G_star = Gm * i_omega_tau_alpha / (1.0 + i_omega_tau_alpha)

            return G_star

        return _compute_oscillation(omega, Gm, tau_alpha)

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellLiquid:
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
        Gm = self.parameters.get_value('Gm')
        alpha = self.parameters.get_value('alpha')
        tau_alpha = self.parameters.get_value('tau_alpha')

        result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
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
            # Check for explicit test_mode in metadata first
            if 'test_mode' in rheo_data.metadata:
                test_mode = rheo_data.metadata['test_mode']
            else:
                test_mode = rheo_data.test_mode

        # Get parameters
        Gm = self.parameters.get_value('Gm')
        alpha = self.parameters.get_value('alpha')
        tau_alpha = self.parameters.get_value('tau_alpha')

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == 'relaxation':
            y_pred = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
        elif test_mode == 'creep':
            y_pred = self._predict_creep_jax(x, Gm, alpha, tau_alpha)
        elif test_mode == 'oscillation':
            y_pred = self._predict_oscillation_jax(x, Gm, alpha, tau_alpha)
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
