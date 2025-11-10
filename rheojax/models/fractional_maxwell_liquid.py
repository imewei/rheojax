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

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.registry import ModelRegistry
from rheojax.utils.mittag_leffler import mittag_leffler_e2


@ModelRegistry.register("fractional_maxwell_liquid")
class FractionalMaxwellLiquid(BaseModel):
    """Fractional Maxwell Liquid model: Spring in series with SpringPot.

    This model describes materials with elastic response at short times and
    power-law relaxation at long times, such as polymer melts.

    Attributes:
        parameters: ParameterSet with Gm, alpha, tau_alpha

    Examples:
        >>> from rheojax.models import FractionalMaxwellLiquid
        >>> from rheojax.core.data import RheoData
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
            name="Gm",
            value=1e6,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Maxwell modulus",
        )

        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=(0.0, 1.0),
            units="dimensionless",
            description="Power-law exponent",
        )

        self.parameters.add(
            name="tau_alpha",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s^α",
            description="Relaxation time",
        )

        self.fitted_ = False

    def _predict_relaxation_jax(
        self, t: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
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

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute Mittag-Leffler parameters
        ml_alpha = 1.0 - alpha_safe
        ml_beta = 1.0 - alpha_safe

        # Compute relaxation modulus
        t_safe = jnp.maximum(t, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # Compute argument for Mittag-Leffler function
        z = -(t_safe ** (1.0 - alpha_safe)) / tau_alpha_safe

        # Compute E_{1-α,1-α}(z) with Python float alpha/beta
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Compute G(t)
        G_t = Gm * (t_safe ** (-alpha_safe)) * ml_value

        return G_t

    def _predict_creep_jax(
        self, t: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
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

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute Mittag-Leffler parameters
        ml_alpha = alpha_safe
        ml_beta = 1.0 + alpha_safe

        # Compute creep compliance
        t_safe = jnp.maximum(t, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # Instantaneous compliance (elastic part)
        J_instant = 1.0 / Gm

        # Time-dependent part with Mittag-Leffler function
        z = -((t_safe / tau_alpha_safe) ** alpha_safe)

        # Compute E_{α,1+α}(z) with Python float alpha/beta
        ml_value = mittag_leffler_e2(z, alpha=ml_alpha, beta=ml_beta)

        # Creep compliance
        J_t = (
            J_instant
            + (t_safe**alpha_safe) / (Gm * (tau_alpha_safe**alpha_safe)) * ml_value
        )

        return J_t

    def _predict_oscillation_jax(
        self, omega: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
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

        # Clip alpha to safe range (now works with JAX tracers)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute oscillation response
        omega_safe = jnp.maximum(omega, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # (iωτ_α)^α = |ωτ_α|^α * exp(i α π/2)
        omega_tau = omega_safe * tau_alpha_safe
        i_omega_tau_alpha = (omega_tau**alpha_safe) * jnp.exp(
            1j * alpha_safe * jnp.pi / 2.0
        )

        # Complex modulus
        G_star = Gm * i_omega_tau_alpha / (1.0 + i_omega_tau_alpha)

        return G_star

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> FractionalMaxwellLiquid:
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

        # Smart initialization for oscillation mode (Issue #9)
        if test_mode == "oscillation":
            try:
                import numpy as np

                from rheojax.utils.initialization import (
                    initialize_fractional_maxwell_liquid,
                )

                success = initialize_fractional_maxwell_liquid(
                    np.array(X), np.array(y), self.parameters
                )
                if success:
                    import logging

                    logging.debug(
                        "Smart initialization applied from frequency-domain features"
                    )
            except Exception as e:
                # Silent fallback to defaults - don't break if initialization fails
                import logging

                logging.debug(f"Smart initialization failed, using defaults: {e}")

        # Create objective function with stateless predictions
        def model_fn(x, params):
            """Model function for optimization (stateless)."""
            Gm, alpha, tau_alpha = params[0], params[1], params[2]

            # Direct prediction based on test mode (stateless, calls _jax methods)
            if test_mode == "relaxation":
                return self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
            elif test_mode == "creep":
                return self._predict_creep_jax(x, Gm, alpha, tau_alpha)
            elif test_mode == "oscillation":
                return self._predict_oscillation_jax(x, Gm, alpha, tau_alpha)
            else:
                raise ValueError(f"Unsupported test mode: {test_mode}")

        # Extract optimization strategy from kwargs (set by BaseModel.fit)
        use_log_residuals = kwargs.get("use_log_residuals", False)
        use_multi_start = kwargs.get("use_multi_start", False)
        n_starts = kwargs.get("n_starts", 5)
        perturb_factor = kwargs.get("perturb_factor", 0.3)

        objective = create_least_squares_objective(
            model_fn,
            x_data,
            y_data,
            normalize=True,
            use_log_residuals=use_log_residuals,
        )

        # Choose optimization strategy
        if use_multi_start:
            from rheojax.utils.optimization import nlsq_multistart_optimize

            result = nlsq_multistart_optimize(
                objective,
                self.parameters,
                n_starts=n_starts,
                perturb_factor=perturb_factor,
                use_jax=kwargs.get("use_jax", True),
                method=kwargs.get("method", "auto"),
                max_iter=kwargs.get("max_iter", 1000),
                verbose=kwargs.get("verbose", False),
            )
        else:
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
        Gm = self.parameters.get_value("Gm")
        alpha = self.parameters.get_value("alpha")
        tau_alpha = self.parameters.get_value("tau_alpha")

        result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
        return np.array(result)

    def model_function(self, X, params):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Gm, alpha, tau_alpha]

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        Gm = params[0]
        alpha = params[1]
        tau_alpha = params[2]

        # Fractional models default to relaxation mode
        # Call the _jax method directly
        return self._predict_relaxation_jax(X, Gm, alpha, tau_alpha)

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
            # Check for explicit test_mode in metadata first
            if "test_mode" in rheo_data.metadata:
                test_mode = rheo_data.metadata["test_mode"]
            else:
                test_mode = rheo_data.test_mode

        # Get parameters
        Gm = self.parameters.get_value("Gm")
        alpha = self.parameters.get_value("alpha")
        tau_alpha = self.parameters.get_value("tau_alpha")

        # Convert input to JAX
        x = jnp.asarray(rheo_data.x)

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            y_pred = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
        elif test_mode == "creep":
            y_pred = self._predict_creep_jax(x, Gm, alpha, tau_alpha)
        elif test_mode == "oscillation":
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
