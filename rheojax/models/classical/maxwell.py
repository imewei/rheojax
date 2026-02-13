"""Maxwell viscoelastic model.

The Maxwell model consists of a spring (G0) and dashpot (eta) in series,
representing the simplest linear viscoelastic behavior with stress relaxation.

Theory:
    - Relaxation modulus: G(t) = G0 * exp(-t/tau) where tau = eta/G0
    - Complex modulus: G*(omega) = G0*(omega*tau)^2/(1+(omega*tau)^2) + i*G0*omega*tau/(1+(omega*tau)^2)
    - Creep compliance: J(t) = 1/G0 + t/eta
    - Steady shear viscosity: eta(gamma_dot) = eta (constant)

References:
    - Ferry, J. D. (1980). Viscoelastic properties of polymers.
    - Tschoegl, N. W. (1989). The phenomenological theory of linear viscoelastic behavior.
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode, TestMode, detect_test_mode
from rheojax.logging import get_logger, log_fit

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "maxwell",
    protocols=[
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
        Protocol.FLOW_CURVE,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class Maxwell(BaseModel):
    """Maxwell viscoelastic model (spring and dashpot in series).

    The Maxwell model is the simplest viscoelastic model, consisting of a linear
    spring (elastic modulus G0) in series with a linear dashpot (viscosity eta).

    Parameters:
        G0 (float): Elastic modulus in Pa, range [1e-3, 1e9], default 1e5
        eta (float): Viscosity in Pa·s, range [1e-6, 1e12], default 1e3

    Supported test modes:
        - Relaxation: Stress relaxation under constant strain
        - Creep: Strain development under constant stress
        - Oscillation: Small amplitude oscillatory shear (SAOS)
        - Rotation: Steady shear flow

    Example:
        >>> from rheojax.models.maxwell import Maxwell
        >>> from rheojax.core.data import RheoData
        >>> import jax.numpy as jnp
        >>>
        >>> # Create model
        >>> model = Maxwell()
        >>> model.parameters.set_value('G0', 1e5)
        >>> model.parameters.set_value('eta', 1e3)
        >>>
        >>> # Predict relaxation
        >>> t = jnp.linspace(0.01, 10, 100)
        >>> data = RheoData(x=t, y=jnp.zeros_like(t), domain='time')
        >>> G_t = model.predict(data)
    """

    def __init__(self):
        """Initialize Maxwell model with default parameters."""
        super().__init__()

        # Define parameters with physical bounds
        self.parameters = ParameterSet()
        self.parameters.add(
            name="G0",
            value=1e5,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Elastic modulus",
        )
        self.parameters.add(
            name="eta",
            value=1e3,
            bounds=(1e-6, 1e12),
            units="Pa·s",
            description="Viscosity",
        )

        self.fitted_ = False
        self._relaxation_offset = 0.0
        self._test_mode = TestMode.RELAXATION  # Store test mode for model_function

    def _fit(self, X, y, **kwargs):
        """Fit Maxwell model to data.

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
        def _to_array(values):
            arr = np.asarray(values)
            if np.iscomplexobj(arr):
                return arr.astype(np.complex128)
            return arr.astype(float)

        if isinstance(X, RheoData):
            rheo_data = X
            x_np = np.asarray(rheo_data.x, dtype=float)
            y_np = _to_array(rheo_data.y)
            test_mode = rheo_data.test_mode
        else:
            x_np = np.asarray(X, dtype=float)
            y_np = _to_array(y)
            supplied_mode = kwargs.get("test_mode")
            if supplied_mode is None and np.iscomplexobj(y_np):
                test_mode = TestMode.OSCILLATION
            else:
                test_mode = supplied_mode or TestMode.RELAXATION

        if isinstance(test_mode, str):
            try:
                test_mode = TestMode[test_mode.upper()]
            except KeyError:
                test_mode = TestMode.RELAXATION

        # Determine test_mode string for logging
        test_mode_str = test_mode.name if hasattr(test_mode, "name") else str(test_mode)

        with log_fit(
            logger,
            self.__class__.__name__,
            data_shape=x_np.shape,
            test_mode=test_mode_str,
        ) as ctx:
            logger.debug(
                "Processing input data",
                x_range=(float(x_np.min()), float(x_np.max())),
                y_range=(float(np.real(y_np).min()), float(np.real(y_np).max())),
                is_complex=np.iscomplexobj(y_np),
            )

            # Store test mode for model_function
            self._test_mode = test_mode
            self._relaxation_offset = 0.0

            if test_mode == TestMode.RELAXATION:
                tail = max(3, y_np.size // 6)
                offset = float(np.median(y_np[-tail:]))
                y_np = y_np - offset
                self._relaxation_offset = offset
                logger.debug(
                    "Applied relaxation offset correction",
                    offset=offset,
                    tail_points=tail,
                )

            x_data = jnp.array(x_np)
            y_data = jnp.array(y_np)

            # Provide simple heuristics for relaxation data to improve deterministic fits
            if test_mode == TestMode.RELAXATION:
                init_success = self._initialize_relaxation_parameters(x_data, y_data)
                logger.debug(
                    "Relaxation parameter initialization",
                    success=init_success,
                    G0_init=self.parameters.get_value("G0"),
                    eta_init=self.parameters.get_value("eta"),
                )

            # Create objective function with stateless predictions
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                G0, eta = params[0], params[1]

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, G0, eta)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, G0, eta)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, G0, eta)
                elif test_mode == TestMode.ROTATION:
                    return self._predict_rotation(x, G0, eta)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            objective = create_least_squares_objective(
                model_fn, x_data, y_data, normalize=True
            )

            logger.debug(
                "Starting NLSQ optimization",
                method=kwargs.get("method", "auto"),
                max_iter=kwargs.get("max_iter", 1000),
                use_jax=kwargs.get("use_jax", True),
            )

            # Optimize
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
                    iterations=getattr(result, "nit", None),
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True

            # Log fitted parameters and result metrics
            G0_fitted = self.parameters.get_value("G0")
            eta_fitted = self.parameters.get_value("eta")
            tau_fitted = eta_fitted / G0_fitted

            ctx["G0"] = G0_fitted
            ctx["eta"] = eta_fitted
            ctx["tau"] = tau_fitted
            ctx["iterations"] = getattr(result, "nit", None)
            ctx["cost"] = getattr(result, "fun", None)

            logger.debug(
                "Optimization completed successfully",
                G0=G0_fitted,
                eta=eta_fitted,
                tau=tau_fitted,
                iterations=getattr(result, "nit", None),
                final_cost=getattr(result, "fun", None),
            )

        return self

    def _initialize_relaxation_parameters(self, X, y) -> bool:
        """Estimate G0 and eta from relaxation data for faster convergence."""
        try:
            t = np.asarray(X, dtype=float).ravel()
            g = np.asarray(y, dtype=float).ravel()
            if t.shape != g.shape or t.size < 3:
                logger.debug(
                    "Initialization skipped: insufficient data",
                    t_shape=t.shape,
                    g_shape=g.shape,
                )
                return False

            order = np.argsort(t)
            t_sorted = t[order]
            g_sorted = g[order]

            tail = max(3, t_sorted.size // 6)
            baseline = float(np.median(g_sorted[-tail:]))
            transient = g_sorted - baseline

            g0_bounds = self.parameters.get("G0").bounds or (1e-3, 1e9) # type: ignore[union-attr]
            eta_bounds = self.parameters.get("eta").bounds or (1e-6, 1e12) # type: ignore[union-attr]

            # Attempt to estimate parameters from the first two signal-dominant points
            positive_mask = transient > 0
            signal_floor = max(float(np.max(transient)), 1e-12) * 1e-3
            idx_candidates = np.where(positive_mask & (transient > signal_floor))[0]
            if idx_candidates.size < 2:
                idx_candidates = np.where(positive_mask)[0]
            if idx_candidates.size < 2:
                logger.debug(
                    "Initialization skipped: insufficient positive transient points",
                    n_candidates=idx_candidates.size,
                )
                return False

            i0, i1 = idx_candidates[0], idx_candidates[1]
            t0, t1 = t_sorted[i0], t_sorted[i1]
            y0, y1 = transient[i0], transient[i1]
            if not (y0 > 0 and y1 > 0 and t1 > t0 and y1 != y0):
                logger.debug(
                    "Initialization skipped: invalid transient values",
                    y0=y0,
                    y1=y1,
                    t0=t0,
                    t1=t1,
                )
                return False

            ratio = y1 / y0
            if ratio <= 0 or ratio < 1e-3:
                logger.debug(
                    "Initialization skipped: invalid decay ratio",
                    ratio=ratio,
                )
                return False
            with np.errstate(divide="ignore"):
                tau_estimate = -(t1 - t0) / np.log(ratio)
            if not (np.isfinite(tau_estimate) and tau_estimate > 0):
                logger.debug(
                    "Initialization skipped: invalid tau estimate",
                    tau_estimate=tau_estimate,
                )
                return False

            g0_estimate = float(y0 * np.exp(t0 / tau_estimate))
            g0_guess = float(np.clip(g0_estimate, g0_bounds[0], g0_bounds[1]))
            eta_guess = float(
                np.clip(g0_guess * tau_estimate, eta_bounds[0], eta_bounds[1])
            )

            self.parameters.set_value("G0", g0_guess)
            self.parameters.set_value("eta", eta_guess)
            logger.debug(
                "Maxwell relaxation initialization successful",
                G0=g0_guess,
                eta=eta_guess,
                tau_estimate=tau_estimate,
                baseline=baseline,
            )
            return True
        except Exception as exc:  # pragma: no cover - heuristic best effort
            logger.debug(
                "Maxwell relaxation initialization failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                exc_info=True,
            )
            return False

    def _predict(self, X, **kwargs):
        """Predict response based on input data.

        Args:
            X: RheoData object or independent variable array
            **kwargs: Additional arguments (ignored for Maxwell)

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
        G0 = self.parameters.get_value("G0")
        eta = self.parameters.get_value("eta")

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(x_data, G0, eta) + getattr(
                self, "_relaxation_offset", 0.0
            )
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(x_data, G0, eta)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(x_data, G0, eta)
        elif test_mode in (TestMode.ROTATION, TestMode.FLOW_CURVE):
            return self._predict_rotation(x_data, G0, eta)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        CRITICAL: test_mode is now passed as parameter (NOT read from self._test_mode)
        to ensure correct posteriors in Bayesian inference (v0.4.0 fix).

        Args:
            X: Independent variable (time, frequency, or shear rate)
            params: Array of parameter values [G0, eta]
            test_mode: Explicit test mode for predictions. If None, falls back
                to self._test_mode for backward compatibility.

        Returns:
            Model predictions as JAX array
        """
        # Extract parameters from array
        G0 = params[0]
        eta = params[1]

        # Use explicit test_mode parameter (closure-captured in fit_bayesian)
        # Fall back to self._test_mode only for backward compatibility
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Dispatch to appropriate prediction method
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, G0, eta) + getattr(
                self, "_relaxation_offset", 0.0
            )
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, G0, eta)
        elif test_mode == TestMode.OSCILLATION:
            return self._predict_oscillation(X, G0, eta)
        elif test_mode == TestMode.ROTATION:
            return self._predict_rotation(X, G0, eta)
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")

    @staticmethod
    @jax.jit
    def _predict_relaxation(t: jnp.ndarray, G0: float, eta: float) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Theory: G(t) = G0 * exp(-t/tau) where tau = eta/G0

        Args:
            t: Time array (s)
            G0: Elastic modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Relaxation modulus G(t) in Pa
        """
        tau = eta / G0  # Relaxation time
        return G0 * jnp.exp(-t / tau)

    @staticmethod
    @jax.jit
    def _predict_creep(t: jnp.ndarray, G0: float, eta: float) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Theory: J(t) = 1/G0 + t/eta

        Args:
            t: Time array (s)
            G0: Elastic modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Creep compliance J(t) in 1/Pa
        """
        return (1.0 / G0) + (t / eta)

    @staticmethod
    @jax.jit
    def _predict_oscillation(omega: jnp.ndarray, G0: float, eta: float) -> jnp.ndarray:
        """Predict complex modulus G*(omega).

        Theory:
            G'(omega) = G0 * (omega*tau)^2 / (1 + (omega*tau)^2)
            G''(omega) = G0 * omega*tau / (1 + (omega*tau)^2)
            G*(omega) = G'(omega) + i*G''(omega)

        Args:
            omega: Angular frequency array (rad/s)
            G0: Elastic modulus (Pa)
            eta: Viscosity (Pa·s)

        Returns:
            Complex modulus G*(omega) in Pa
        """
        tau = eta / G0  # Relaxation time
        omega_tau = omega * tau
        omega_tau_sq = omega_tau**2

        # Storage modulus G'
        G_prime = G0 * omega_tau_sq / (1.0 + omega_tau_sq)

        # Loss modulus G''
        G_double_prime = G0 * omega_tau / (1.0 + omega_tau_sq)

        # Complex modulus
        return G_prime + 1j * G_double_prime

    @staticmethod
    @jax.jit
    def _predict_rotation(gamma_dot: jnp.ndarray, G0: float, eta: float) -> jnp.ndarray:
        """Predict steady shear viscosity eta(gamma_dot).

        Theory: eta(gamma_dot) = eta (constant, Newtonian behavior)

        Args:
            gamma_dot: Shear rate array (1/s)
            G0: Elastic modulus (Pa) - not used but kept for interface consistency
            eta: Viscosity (Pa·s)

        Returns:
            Viscosity eta in Pa·s (constant array)
        """
        return eta * jnp.ones_like(gamma_dot)

    def get_relaxation_time(self) -> float:
        """Get characteristic relaxation time tau = eta/G0.

        Returns:
            Relaxation time in seconds
        """
        G0 = self.parameters.get_value("G0")
        eta = self.parameters.get_value("eta")
        return eta / G0 # type: ignore[operator]

    def __repr__(self) -> str:
        """String representation of Maxwell model."""
        G0 = self.parameters.get_value("G0")
        eta = self.parameters.get_value("eta")
        tau = self.get_relaxation_time()
        return f"Maxwell(G0={G0:.2e} Pa, eta={eta:.2e} Pa·s, tau={tau:.2e} s)"


__all__ = ["Maxwell"]
