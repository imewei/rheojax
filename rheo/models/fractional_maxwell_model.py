"""Fractional Maxwell Model (FMM).

This is the most general fractional Maxwell model with two SpringPots in series,
each with independent fractional orders. It provides maximum flexibility in
describing viscoelastic materials with fractional dynamics.

Mathematical Description:
    Relaxation Modulus: G(t) = c_1 t^(-α) E_{1-α}(-(t/τ)^β)
    Complex Modulus: G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

where α and β are independent fractional orders.

Parameters:
    c1 (float): Material constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): First fractional order, bounds [0.0, 1.0]
    beta (float): Second fractional order, bounds [0.0, 1.0]
    tau (float): Relaxation time (s), bounds [1e-6, 1e6]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Schiessel, H., & Blumen, A. (1993). Hierarchical analogues to fractional relaxation
      equations. Journal of Physics A: Mathematical and General, 26(19), 5057.
    - Heymans, N., & Bauwens, J. C. (1994). Fractal rheological models and fractional
      differential equations for viscoelastic behavior. Rheologica Acta, 33(3), 210-219.
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
from rheo.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2


@ModelRegistry.register('fractional_maxwell_model')
class FractionalMaxwellModel(BaseModel):
    """Fractional Maxwell Model: Two SpringPots in series with independent orders.

    This is the most general fractional Maxwell model, allowing for complex
    viscoelastic behavior with two independent fractional orders.

    Attributes:
        parameters: ParameterSet with c1, alpha, beta, tau

    Examples:
        >>> from rheo.models import FractionalMaxwellModel
        >>> from rheo.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalMaxwellModel()
        >>> model.parameters.set_value('c1', 1e5)
        >>> model.parameters.set_value('alpha', 0.5)
        >>> model.parameters.set_value('beta', 0.7)
        >>> model.parameters.set_value('tau', 1.0)
        >>>
        >>> # Predict relaxation modulus
        >>> t = np.logspace(-3, 3, 50)
        >>> data = RheoData(x=t, y=np.zeros_like(t), domain='time')
        >>> data.metadata['test_mode'] = 'relaxation'
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize Fractional Maxwell Model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name='c1',
            value=1e5,
            bounds=(1e-3, 1e9),
            units='Pa·s^α',
            description='Material constant'
        )

        self.parameters.add(
            name='alpha',
            value=0.5,
            bounds=(0.0, 1.0),
            units='dimensionless',
            description='First fractional order'
        )

        self.parameters.add(
            name='beta',
            value=0.5,
            bounds=(0.0, 1.0),
            units='dimensionless',
            description='Second fractional order'
        )

        self.parameters.add(
            name='tau',
            value=1.0,
            bounds=(1e-6, 1e6),
            units='s',
            description='Relaxation time'
        )

        self.fitted_ = False

    def _predict_relaxation_jax(
        self,
        t: jnp.ndarray,
        c1: float,
        alpha: float,
        beta: float,
        tau: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = c_1 t^(-α) E_{1-α}(-(t/τ)^β)

        Args:
            t: Time array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha and beta BEFORE JIT to make them concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        beta_safe = float(np.clip(beta, epsilon, 1.0 - epsilon))

        # Compute Mittag-Leffler parameter as concrete value
        ml_alpha = 1.0 - alpha_safe

        # JIT-compiled inner function with concrete alpha/beta
        @jax.jit
        def _compute_relaxation(t, c1, tau):
            t_safe = jnp.maximum(t, epsilon)
            tau_safe = jnp.maximum(tau, epsilon)

            # Compute argument for Mittag-Leffler function
            z = -((t_safe / tau_safe) ** beta_safe)

            # Compute E_{1-α}(z) with concrete alpha
            ml_value = mittag_leffler_e(z, alpha=ml_alpha)

            # Compute G(t)
            G_t = c1 * (t_safe ** (-alpha_safe)) * ml_value

            return G_t

        return _compute_relaxation(t, c1, tau)

    def _predict_creep_jax(
        self,
        t: jnp.ndarray,
        c1: float,
        alpha: float,
        beta: float,
        tau: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        For the general FMM, the creep compliance is obtained through
        the relationship J(s) = 1/[s²G(s)] in Laplace domain.

        Args:
            t: Time array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Creep compliance array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha and beta BEFORE JIT to make them concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        beta_safe = float(np.clip(beta, epsilon, 1.0 - epsilon))

        # Compute Mittag-Leffler parameters as concrete values
        ml_alpha = alpha_safe
        ml_beta = 1.0 + alpha_safe

        # JIT-compiled inner function with concrete alpha/beta
        @jax.jit
        def _compute_creep(t, c1, tau):
            t_safe = jnp.maximum(t, epsilon)
            tau_safe = jnp.maximum(tau, epsilon)

            # For general FMM, creep is more complex
            # Approximate using J(t) ≈ (1/c1) t^α E_{α,1+α}((t/τ)^β)
            z = (t_safe / tau_safe) ** beta_safe

            # Compute E_{α,1+α}(z) with concrete alpha/beta
            ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

            # Creep compliance
            J_t = (1.0 / c1) * (t_safe ** alpha_safe) * ml_value

            return J_t

        return _compute_creep(t, c1, tau)

    def _predict_oscillation_jax(
        self,
        omega: jnp.ndarray,
        c1: float,
        alpha: float,
        beta: float,
        tau: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = c_1 (iω)^α / (1 + (iωτ)^β)

        Args:
            omega: Angular frequency array
            c1: Material constant
            alpha: First fractional order
            beta: Second fractional order
            tau: Relaxation time

        Returns:
            Complex modulus array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha and beta BEFORE JIT to make them concrete (not traced)
        import numpy as np
        alpha_safe = float(np.clip(alpha, epsilon, 1.0 - epsilon))
        beta_safe = float(np.clip(beta, epsilon, 1.0 - epsilon))

        # JIT-compiled inner function with concrete alpha/beta
        @jax.jit
        def _compute_oscillation(omega, c1, tau):
            omega_safe = jnp.maximum(omega, epsilon)
            tau_safe = jnp.maximum(tau, epsilon)

            # (iω)^α = |ω|^α * exp(i α π/2)
            i_omega_alpha = (omega_safe ** alpha_safe) * jnp.exp(1j * alpha_safe * jnp.pi / 2.0)

            # (iωτ)^β = |ωτ|^β * exp(i β π/2)
            omega_tau = omega_safe * tau_safe
            i_omega_tau_beta = (omega_tau ** beta_safe) * jnp.exp(1j * beta_safe * jnp.pi / 2.0)

            # Complex modulus
            G_star = c1 * i_omega_alpha / (1.0 + i_omega_tau_beta)

            return G_star

        return _compute_oscillation(omega, c1, tau)

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellModel:
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
        c1 = self.parameters.get_value('c1')
        alpha = self.parameters.get_value('alpha')
        beta = self.parameters.get_value('beta')
        tau = self.parameters.get_value('tau')

        result = self._predict_relaxation_jax(x, c1, alpha, beta, tau)
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
        c1 = self.parameters.get_value('c1')
        alpha = self.parameters.get_value('alpha')
        beta = self.parameters.get_value('beta')
        tau = self.parameters.get_value('tau')

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == 'relaxation':
            y_pred = self._predict_relaxation_jax(x, c1, alpha, beta, tau)
        elif test_mode == 'creep':
            y_pred = self._predict_creep_jax(x, c1, alpha, beta, tau)
        elif test_mode == 'oscillation':
            y_pred = self._predict_oscillation_jax(x, c1, alpha, beta, tau)
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
