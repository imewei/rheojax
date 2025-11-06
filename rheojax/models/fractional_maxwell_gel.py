"""Fractional Maxwell Gel (FMG) model.

This model consists of a SpringPot element (power-law viscoelastic element) in series
with a dashpot (Newtonian viscous element). It captures the transition from power-law
viscoelastic behavior to terminal flow.

Mathematical Description:
    Relaxation Modulus: G(t) = c_α t^(-α) E_{1-α,1-α}(-t^(1-α)/τ)
    Complex Modulus: G*(ω) = c_α (iω)^α · (iωτ) / (1 + iωτ)
    Creep Compliance: J(t) = (1/c_α) t^α E_{1+α,1+α}(-(t/τ)^(1-α))

where τ = η / c_α^(1/(1-α)) is a characteristic relaxation time.

Parameters:
    c_alpha (float): Material constant (Pa·s^α), bounds [1e-3, 1e9]
    alpha (float): Power-law exponent, bounds [0.0, 1.0]
    eta (float): Viscosity (Pa·s), bounds [1e-6, 1e12]

Test Modes: Relaxation, Creep, Oscillation

References:
    - Blair, G. S., Veinoglou, B. C., & Caffyn, J. E. (1947). Limitations of the Newtonian
      time scale in relation to non-equilibrium rheological states and a theory of
      quasi-properties. Proc. R. Soc. Lond. A, 189(1016), 69-87.
    - Friedrich, C., & Braun, H. (1992). Generalized Cole-Cole behavior and its rheological
      relevance. Rheologica Acta, 31(4), 309-322.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e2


@ModelRegistry.register("fractional_maxwell_gel")
class FractionalMaxwellGel(BaseModel):
    """Fractional Maxwell Gel model: SpringPot in series with dashpot.

    This model describes the rheology of materials transitioning from power-law
    viscoelastic behavior to terminal flow, such as polymer solutions and gels.

    Attributes:
        parameters: ParameterSet with c_alpha, alpha, eta

    Examples:
        >>> from rheojax.models import FractionalMaxwellGel
        >>> from rheojax.core.data import RheoData
        >>> import numpy as np
        >>>
        >>> # Create model with parameters
        >>> model = FractionalMaxwellGel()
        >>> model.parameters.set_value('c_alpha', 1e5)
        >>> model.parameters.set_value('alpha', 0.5)
        >>> model.parameters.set_value('eta', 1e3)
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
        """Initialize Fractional Maxwell Gel model."""
        super().__init__()
        self.parameters = ParameterSet()

        self.parameters.add(
            name="c_alpha",
            value=10.0,  # Chosen to keep tau numerically stable across alpha ∈ [0,1]
            bounds=(1e-3, 1e9),
            units="Pa·s^α",
            description="SpringPot material constant",
        )

        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=(0.0, 1.0),
            units="dimensionless",
            description="Power-law exponent",
        )

        self.parameters.add(
            name="eta",
            value=1e4,  # Chosen to keep tau~O(1) for alpha=0.5 with c_alpha=100
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Dashpot viscosity",
        )

        self.fitted_ = False

    def _compute_tau(self, c_alpha: float, alpha: float) -> float:
        """Compute characteristic relaxation time.

        Args:
            c_alpha: SpringPot constant
            alpha: Power-law exponent

        Returns:
            Characteristic time τ = η / c_α^(1/(1-α))
        """
        eta = self.parameters.get_value("eta")
        # Add small epsilon to prevent division by zero
        epsilon = 1e-12
        return eta / (c_alpha ** (1.0 / (1.0 - alpha + epsilon)))

    def _predict_relaxation_jax(
        self, t: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = c_α t^(-α) E_{1-α,1-α}(-t^(1-α)/τ)

        Args:
            t: Time array
            c_alpha: SpringPot constant
            alpha: Power-law exponent
            eta: Viscosity

        Returns:
            Relaxation modulus array
        """
        # Add small epsilon to prevent issues at t=0 and with alpha=1
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)

        # Compute characteristic time
        tau = eta / (c_alpha ** (1.0 / (1.0 - alpha_safe)))

        # Compute argument for Mittag-Leffler function
        z = -(t_safe ** (1.0 - alpha_safe)) / tau

        # Compute E_{1-α,1-α}(z) (requires concrete alpha/beta)
        ml_alpha = 1.0 - alpha_safe
        ml_beta = 1.0 - alpha_safe
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Compute G(t)
        G_t = c_alpha * (t_safe ** (-alpha_safe)) * ml_value

        return G_t

    def _predict_creep_jax(
        self, t: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/c_α) t^α E_{1+α,1+α}(-(t/τ)^(1-α))

        Args:
            t: Time array
            c_alpha: SpringPot constant
            alpha: Power-law exponent
            eta: Viscosity

        Returns:
            Creep compliance array
        """
        # Add small epsilon to prevent issues
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute safe values
        t_safe = jnp.maximum(t, epsilon)

        # Compute characteristic time
        tau = eta / (c_alpha ** (1.0 / (1.0 - alpha_safe)))

        # Compute argument for Mittag-Leffler function
        z = -((t_safe / tau) ** (1.0 - alpha_safe))

        # Compute E_{1+α,1+α}(z) (requires concrete alpha/beta)
        ml_alpha = 1.0 + alpha_safe
        ml_beta = 1.0 + alpha_safe
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Compute J(t)
        J_t = (1.0 / c_alpha) * (t_safe**alpha_safe) * ml_value

        # Ensure monotonicity: creep compliance must increase with time
        # Use cumulative maximum to enforce J(t_i) >= J(t_{i-1})
        J_t_monotonic = jnp.maximum.accumulate(J_t)

        return J_t_monotonic

    def _predict_oscillation_jax(
        self, omega: jnp.ndarray, c_alpha: float, alpha: float, eta: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))

        This is the correct formula for SpringPot in series with dashpot.

        Args:
            omega: Angular frequency array
            c_alpha: SpringPot constant
            alpha: Power-law exponent
            eta: Viscosity

        Returns:
            Complex modulus array
        """
        # Add small epsilon
        epsilon = 1e-12

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute beta for the denominator
        beta_safe = 1.0 - alpha_safe

        # Compute safe values
        omega_safe = jnp.maximum(omega, epsilon)
        tau_safe = jnp.maximum(
            eta / (c_alpha ** (1.0 / (1.0 - alpha_safe + epsilon))), epsilon
        )

        # (iω)^α = |ω|^α * exp(i α π/2)
        i_omega_alpha = (omega_safe**alpha_safe) * jnp.exp(
            1j * alpha_safe * jnp.pi / 2.0
        )

        # (iωτ)^(1-α) = |ωτ|^(1-α) * exp(i (1-α) π/2)
        omega_tau = omega_safe * tau_safe
        i_omega_tau_beta = (omega_tau**beta_safe) * jnp.exp(
            1j * beta_safe * jnp.pi / 2.0
        )

        # Complex modulus: G*(ω) = c_α (iω)^α / (1 + (iωτ)^(1-α))
        G_star = c_alpha * i_omega_alpha / (1.0 + i_omega_tau_beta)

        return G_star

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellGel:
        """Fit model parameters to data.

        Args:
            X: Independent variable (time or frequency)
            y: Dependent variable (modulus or compliance)
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
            test_mode = kwargs.get("test_mode", "relaxation")

        # Create objective function with stateless predictions
        def model_fn(x, params):
            """Model function for optimization (stateless)."""
            c_alpha, alpha, eta = params[0], params[1], params[2]

            # Direct prediction based on test mode (stateless, calls _jax methods)
            if test_mode == "relaxation":
                return self._predict_relaxation_jax(x, c_alpha, alpha, eta)
            elif test_mode == "creep":
                return self._predict_creep_jax(x, c_alpha, alpha, eta)
            elif test_mode == "oscillation":
                return self._predict_oscillation_jax(x, c_alpha, alpha, eta)
            else:
                raise ValueError(f"Unsupported test mode: {test_mode}")

        objective = create_least_squares_objective(
            model_fn, x_data, y_data, normalize=True
        )

        # Optimize using NLSQ (JAX enabled by default)
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 1000),
        )

        # Validate optimization succeeded
        if not result.success:
            raise RuntimeError(
                f"Optimization failed: {result.message}. "
                f"Try adjusting initial values, bounds, or max_iter."
            )

        self.fitted_ = True
        return self

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
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        eta = self.parameters.get_value("eta")

        result = self._predict_relaxation_jax(x, c_alpha, alpha, eta)
        return np.array(result)

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [c_alpha, alpha, eta]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        c_alpha = params[0]
        alpha = params[1]
        eta = params[2]

        # Fractional models default to relaxation mode
        # Call the _jax method directly
        return self._predict_relaxation_jax(X, c_alpha, alpha, eta)

    def predict_rheodata(
        self, rheo_data: RheoData, test_mode: str | None = None
    ) -> RheoData:
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
        c_alpha = self.parameters.get_value("c_alpha")
        alpha = self.parameters.get_value("alpha")
        eta = self.parameters.get_value("eta")

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            y_pred = self._predict_relaxation_jax(x, c_alpha, alpha, eta)
        elif test_mode == "creep":
            y_pred = self._predict_creep_jax(x, c_alpha, alpha, eta)
        elif test_mode == "oscillation":
            y_pred = self._predict_oscillation_jax(x, c_alpha, alpha, eta)
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
            metadata=rheo_data.metadata.copy(),
        )

        return result

    def predict(self, X, test_mode: str | None = None):
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
