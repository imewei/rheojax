"""SpringPot fractional viscoelastic element.

The SpringPot (also called fractional element or Scott-Blair element) is a
power-law viscoelastic element that interpolates between pure elastic (alpha=1)
and pure viscous (alpha=0) behavior.

Theory:
    - Relaxation modulus: G(t) = c_alpha * t^(-alpha) / Gamma(1-alpha)
    - Complex modulus: G*(omega) = c_alpha * (i*omega)^alpha
    - Creep compliance: J(t) = (1/c_alpha) * t^alpha / Gamma(1+alpha)
    - Uses Mittag-Leffler functions for accurate fractional calculus

References:
    - Scott Blair, G. W. (1947). The role of psychophysics in rheology.
    - Bagley, R. L., & Torvik, P. J. (1983). Fractional calculus model of viscoelastic behavior.
    - Schiessel, H., et al. (1995). Generalized viscoelastic models.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


from jax.scipy.special import gamma as jax_gamma

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode


@ModelRegistry.register("springpot")
class SpringPot(BaseModel):
    """SpringPot fractional viscoelastic element.

    The SpringPot represents a power-law viscoelastic material that exhibits
    fractional behavior between pure solid (alpha=1) and pure fluid (alpha=0).

    Parameters:
        c_alpha (float): Material constant in Pa·s^alpha, range [1e-3, 1e9], default 1e5
        alpha (float): Power-law exponent (0=fluid, 1=solid), range [0.0, 1.0], default 0.5

    Supported test modes:
        - Relaxation: Stress relaxation under constant strain
        - Creep: Strain development under constant stress
        - Oscillation: Small amplitude oscillatory shear (SAOS)
        - Rotation: NOT SUPPORTED (SpringPot is linear viscoelastic)

    Example:
        >>> from rheojax.models.springpot import SpringPot
        >>> from rheojax.core.data import RheoData
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = SpringPot()
        >>> model.parameters.set_value('c_alpha', 1e5)
        >>> model.parameters.set_value('alpha', 0.5)
        >>>
        >>> # Predict relaxation
        >>> t = jnp.linspace(0.01, 10, 100)
        >>> data = RheoData(x=t, y=jnp.zeros_like(t), domain='time')
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize SpringPot model with default parameters."""
        super().__init__()

        # Define parameters with physical bounds
        self.parameters = ParameterSet()
        self.parameters.add(
            name="c_alpha",
            value=1e5,
            bounds=(1e-3, 1e9),
            units="Pa·s^alpha",
            description="Material constant",
        )
        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=(0.0, 1.0),
            units="dimensionless",
            description="Power-law exponent (0=fluid, 1=solid)",
        )

        self.fitted_ = False
        self._test_mode = TestMode.RELAXATION  # Store test mode for model_function

    def _fit(self, X, y, **kwargs):
        """Fit SpringPot model to data.

        Args:
            X: RheoData object or independent variable array
            y: Dependent variable array (if X is not RheoData)
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Handle RheoData input
        if isinstance(X, RheoData):
            rheo_data = X
            x_data = jnp.array(rheo_data.x)
            y_data = jnp.array(rheo_data.y)
            test_mode = rheo_data.test_mode
        else:
            x_data = jnp.array(X)
            y_data = jnp.array(y)
            test_mode = kwargs.get("test_mode", TestMode.RELAXATION)

        # Validate test mode
        if test_mode == TestMode.ROTATION:
            raise ValueError(
                "SpringPot model does not support steady shear (rotation) test mode"
            )

        # Store test mode for model_function
        self._test_mode = test_mode

        # Create objective function with stateless predictions
        def model_fn(x, params):
            """Model function for optimization (stateless)."""
            c_alpha, alpha = params[0], params[1]

            # Direct prediction based on test mode (stateless)
            if test_mode == TestMode.RELAXATION:
                return self._predict_relaxation(x, c_alpha, alpha)
            elif test_mode == TestMode.CREEP:
                return self._predict_creep(x, c_alpha, alpha)
            elif test_mode == TestMode.OSCILLATION:
                return self._predict_oscillation(x, c_alpha, alpha)
            else:
                raise ValueError(f"Unsupported test mode: {test_mode}")

        objective = create_least_squares_objective(
            model_fn, x_data, y_data, normalize=True
        )

        # Optimize
        nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 1000),
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
            # Use test_mode from last fit if available, otherwise default to RELAXATION
            test_mode = getattr(self, '_test_mode', TestMode.RELAXATION)

        # Validate test mode
        if test_mode == TestMode.ROTATION:
            raise ValueError(
                "SpringPot model does not support steady shear (rotation) test mode"
            )

        # Get parameter values
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(x_data, c_alpha, alpha)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(x_data, c_alpha, alpha)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(x_data, c_alpha, alpha)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c_alpha, alpha]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        c_alpha = params[0]
        alpha = params[1]

        # Use stored test mode from last fit, or default to RELAXATION
        test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, c_alpha, alpha)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, c_alpha, alpha)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(X, c_alpha, alpha)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    @staticmethod
    @jax.jit
    def _predict_relaxation(
        t: jnp.ndarray, c_alpha: float, alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Theory: G(t) = c_alpha * t^(-alpha) / Gamma(1-alpha)

        For alpha=0 (pure viscous): G(t) = c_alpha / t
        For alpha=1 (pure elastic): G(t) = c_alpha (constant)

        Args:
            t: Time array (s)
            c_alpha: Material constant (Pa·s^alpha)
            alpha: Power-law exponent (0=fluid, 1=solid)

        Returns:
            Relaxation modulus G(t) in Pa
        """
        # Handle special cases
        # alpha -> 0: pure viscous (G -> c_alpha/t)
        # alpha -> 1: pure elastic (G -> c_alpha)

        # General formula: G(t) = c_alpha * t^(-alpha) / Gamma(1-alpha)
        gamma_factor = jax_gamma(1.0 - alpha)
        return c_alpha * jnp.power(t, -alpha) / gamma_factor

    @staticmethod
    @jax.jit
    def _predict_creep(t: jnp.ndarray, c_alpha: float, alpha: float) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Theory: J(t) = (1/c_alpha) * t^alpha / Gamma(1+alpha)

        For alpha=0 (pure viscous): J(t) = t/c_alpha
        For alpha=1 (pure elastic): J(t) = 1/c_alpha

        Args:
            t: Time array (s)
            c_alpha: Material constant (Pa·s^alpha)
            alpha: Power-law exponent (0=fluid, 1=solid)

        Returns:
            Creep compliance J(t) in 1/Pa
        """
        # General formula: J(t) = (1/c_alpha) * t^alpha / Gamma(1+alpha)
        gamma_factor = jax_gamma(1.0 + alpha)
        return (1.0 / c_alpha) * jnp.power(t, alpha) / gamma_factor

    @staticmethod
    @jax.jit
    def _predict_oscillation(
        omega: jnp.ndarray, c_alpha: float, alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(omega).

        Theory: G*(omega) = c_alpha * (i*omega)^alpha

        This can be written as:
            G*(omega) = c_alpha * omega^alpha * (cos(pi*alpha/2) + i*sin(pi*alpha/2))

        Therefore:
            G'(omega) = c_alpha * omega^alpha * cos(pi*alpha/2)
            G''(omega) = c_alpha * omega^alpha * sin(pi*alpha/2)

        Args:
            omega: Angular frequency array (rad/s)
            c_alpha: Material constant (Pa·s^alpha)
            alpha: Power-law exponent (0=fluid, 1=solid)

        Returns:
            Complex modulus G*(omega) in Pa
        """
        # Compute (i*omega)^alpha = omega^alpha * exp(i*pi*alpha/2)
        omega_alpha = jnp.power(omega, alpha)
        phase = jnp.pi * alpha / 2.0

        # Storage modulus G'
        G_prime = c_alpha * omega_alpha * jnp.cos(phase)

        # Loss modulus G''
        G_double_prime = c_alpha * omega_alpha * jnp.sin(phase)

        # Complex modulus
        return G_prime + 1j * G_double_prime

    def get_characteristic_time(self, reference_value: float = 1.0) -> float:
        """Get characteristic time scale for the material.

        For SpringPot, there's no single characteristic time, but we can define
        a reference time where G(t) = reference_value.

        From G(t) = c_alpha * t^(-alpha) / Gamma(1-alpha) = reference_value:
            t = (c_alpha / (reference_value * Gamma(1-alpha)))^(1/alpha)

        Args:
            reference_value: Reference modulus value (Pa), default 1.0

        Returns:
            Characteristic time in seconds
        """
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")

        # Avoid division by zero for alpha=0
        if alpha < 1e-10:
            return float("inf")

        gamma_factor = float(jax_gamma(1.0 - alpha))
        return (c_alpha / (reference_value * gamma_factor)) ** (1.0 / alpha)

    def __repr__(self) -> str:
        """String representation of SpringPot model."""
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        return f"SpringPot(c_alpha={c_alpha:.2e} Pa·s^{alpha:.2f}, alpha={alpha:.2f})"


__all__ = ["SpringPot"]
