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

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode, detect_test_mode
from rheojax.logging import get_logger, log_fit

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "zener",
    protocols=[
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
        Protocol.FLOW_CURVE,
    ],
)
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
        >>> from rheojax.models.zener import Zener
        >>> from rheojax.core.data import RheoData
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
            name="Ge",
            value=1e4,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Equilibrium modulus",
        )
        self.parameters.add(
            name="Gm",
            value=1e5,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Maxwell modulus",
        )
        self.parameters.add(
            name="eta",
            value=1e3,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Viscosity",
        )

        self.fitted_ = False
        self._test_mode = TestMode.RELAXATION  # Store test mode for model_function

    def _fit(self, X, y, **kwargs):
        """Fit Zener model to data.

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

        # Store test mode for model_function
        self._test_mode = test_mode

        # Determine test mode string for logging
        test_mode_str = (
            test_mode.value if hasattr(test_mode, "value") else str(test_mode)
        )
        data_shape = (int(x_data.shape[0]),) if hasattr(x_data, "shape") else None

        with log_fit(
            logger, model="Zener", data_shape=data_shape, test_mode=test_mode_str
        ) as ctx:
            logger.debug(
                "Starting Zener model fit",
                test_mode=test_mode_str,
                n_points=data_shape[0] if data_shape else None,
                initial_Ge=self.parameters.get_value("Ge"),
                initial_Gm=self.parameters.get_value("Gm"),
                initial_eta=self.parameters.get_value("eta"),
            )

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

            logger.debug("Creating least squares objective", normalize=True)
            objective = create_least_squares_objective(
                model_fn, x_data, y_data, normalize=True
            )

            # Optimize
            logger.debug(
                "Starting NLSQ optimization",
                use_jax=kwargs.get("use_jax", True),
                method=kwargs.get("method", "auto"),
                max_iter=kwargs.get("max_iter", 1000),
            )
            try:
                result = nlsq_optimize(
                    objective,
                    self.parameters,
                    use_jax=kwargs.get("use_jax", True),
                    method=kwargs.get("method", "auto"),
                    max_iter=kwargs.get("max_iter", 1000),
                )
            except Exception as e:
                logger.error(
                    "NLSQ optimization raised exception",
                    error_type=type(e).__name__,
                    error_message=str(e),
                    exc_info=True,
                )
                raise

            # Validate optimization succeeded
            if not result.success:
                logger.error(
                    "Optimization failed",
                    message=result.message,
                    test_mode=test_mode_str,
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True

            # Log fitted parameters
            fitted_Ge = self.parameters.get_value("Ge")
            fitted_Gm = self.parameters.get_value("Gm")
            fitted_eta = self.parameters.get_value("eta")
            fitted_tau = fitted_eta / fitted_Gm if fitted_Gm > 0 else float("inf")

            logger.debug(
                "Zener fit completed successfully",
                fitted_Ge=fitted_Ge,
                fitted_Gm=fitted_Gm,
                fitted_eta=fitted_eta,
                relaxation_time=fitted_tau,
            )

            ctx["Ge"] = fitted_Ge
            ctx["Gm"] = fitted_Gm
            ctx["eta"] = fitted_eta
            ctx["tau"] = fitted_tau

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
            test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Get parameter values
        Ge = self.parameters.get_value("Ge")
        Gm = self.parameters.get_value("Gm")
        eta = self.parameters.get_value("eta")

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

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time, frequency, or shear rate)
            params: Array of parameter values [Ge, Gm, eta]
            test_mode: Test mode for predictions (relaxation, creep, oscillation, rotation)

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array (in order they were added to ParameterSet)
        Ge = params[0]
        Gm = params[1]
        eta = params[2]

        # Use provided test_mode, or fallback to stored test mode or default
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, Ge, Gm, eta)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, Ge, Gm, eta)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(X, Ge, Gm, eta)
        elif test_mode == TestMode.ROTATION:
            return self._predict_rotation(X, Ge, Gm, eta)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    @staticmethod
    @jax.jit
    def _predict_relaxation(
        t: jnp.ndarray, Ge: float, Gm: float, eta: float
    ) -> jnp.ndarray:
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
    def _predict_oscillation(
        omega: jnp.ndarray, Ge: float, Gm: float, eta: float
    ) -> jnp.ndarray:
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
        omega_tau_sq = omega_tau**2

        # Storage modulus G'
        G_prime = Ge + Gm * omega_tau_sq / (1.0 + omega_tau_sq)

        # Loss modulus G''
        G_double_prime = Gm * omega_tau / (1.0 + omega_tau_sq)

        # Complex modulus
        return G_prime + 1j * G_double_prime

    @staticmethod
    @jax.jit
    def _predict_rotation(
        gamma_dot: jnp.ndarray, Ge: float, Gm: float, eta: float
    ) -> jnp.ndarray:
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
        Gm = self.parameters.get_value("Gm")
        eta = self.parameters.get_value("eta")
        return eta / Gm

    def get_retardation_time(self) -> float:
        """Get characteristic retardation time for creep.

        Theory: tau_c = eta * (G_e + G_m) / (G_e * G_m)

        Returns:
            Retardation time in seconds
        """
        Ge = self.parameters.get_value("Ge")
        Gm = self.parameters.get_value("Gm")
        eta = self.parameters.get_value("eta")
        return eta * (Ge + Gm) / (Ge * Gm)

    def __repr__(self) -> str:
        """String representation of Zener model."""
        Ge = self.parameters.get_value("Ge")
        Gm = self.parameters.get_value("Gm")
        eta = self.parameters.get_value("eta")
        tau = self.get_relaxation_time()
        return f"Zener(Ge={Ge:.2e} Pa, Gm={Gm:.2e} Pa, eta={eta:.2e} Pa·s, tau={tau:.2e} s)"


__all__ = ["Zener"]
