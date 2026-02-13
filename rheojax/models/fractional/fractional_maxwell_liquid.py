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
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


JAX_ARRAY_TYPES = tuple(
    t for t in (getattr(jax, "Array", None), jax.core.Tracer) if t is not None
)


import numpy as np

from rheojax.core.base import BaseModel, ParameterSet
from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.utils.mittag_leffler import mittag_leffler_e2

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_maxwell_liquid",
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
            bounds=FRACTIONAL_ORDER_BOUNDS,
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

    def bayesian_prior_factory(
        self, param_name: str, lower: float | None, upper: float | None
    ):
        """Provide custom priors that stay near realistic data-informed scales."""
        return None  # Disable custom priors for stability

        # stats = getattr(self, "_fml_bayes_stats", {})
        # if lower is None or upper is None:
        #     return None
        #
        # def _log_normal(target: float, scale: float = 0.6):
        #     if not np.isfinite(target) or target <= 0:
        #         return None
        #     low = float(max(lower, 1e-12))
        #     high = float(max(upper, low * 1.01))
        #     log_low = np.log(low)
        #     log_high = np.log(high)
        #     loc = float(np.clip(np.log(target), log_low + 1e-6, log_high - 1e-6))
        #     base = dist.TruncatedNormal(
        #         loc=loc, scale=scale, low=log_low, high=log_high
        #     )
        #     return dist.TransformedDistribution(base, dist_transforms.ExpTransform())
        #
        # if param_name == "Gm" and "gm_target" in stats:
        #     return _log_normal(stats["gm_target"], scale=0.7)
        # if param_name == "tau_alpha" and "tau_target" in stats:
        #     return _log_normal(stats["tau_target"], scale=0.7)
        # return None

    def bayesian_parameter_bounds(
        self,
        bounds: dict[str, tuple[float | None, float | None]],
        X: np.ndarray,
        y: np.ndarray,
        test_mode,
    ) -> dict[str, tuple[float | None, float | None]]:
        """Tighten tau bounds based on data scale to avoid pathological samples."""
        return bounds  # Disable bounds tightening for stability

        # stats: dict[str, float] = {}
        #
        # if "tau_alpha" in bounds and X.size > 0:
        #     positive_times = np.asarray(X, dtype=float)
        #     positive_times = positive_times[positive_times > 0]
        #     if positive_times.size:
        #         t_min = float(np.min(positive_times))
        #         t_max = float(np.max(positive_times))
        #         lower, upper = bounds["tau_alpha"]
        #         new_lower = max(lower or 0.0, t_min * 0.2, 1e-5)
        #         new_upper = min(upper or np.inf, t_max * 5.0)
        #         if new_upper <= new_lower:
        #             new_upper = new_lower * 10.0
        #         bounds["tau_alpha"] = (new_lower, new_upper)
        #
        #         tau_geo = float(np.exp(0.5 * (np.log(new_lower) + np.log(new_upper))))
        #         stats["tau_target"] = tau_geo
        #
        # y_abs = np.asarray(y, dtype=float)
        # if y_abs.size:
        #     gm_lower, gm_upper = bounds.get("Gm", (None, None))
        #     gm_lower = gm_lower if gm_lower is not None else 1e-3
        #     gm_upper = gm_upper if gm_upper is not None else y_abs.max() * 10.0
        #     median_scale = float(np.median(np.abs(y_abs)))
        #     gm_target = float(np.clip(median_scale, gm_lower * 1.1, gm_upper * 0.9))
        #     stats["gm_target"] = gm_target
        #
        # if stats:
        #     self._fml_bayes_stats = stats
        #
        # return bounds

    def bayesian_nuts_kwargs(self) -> dict:
        """Prefer conservative NUTS settings for the stiff Mittag-Leffler kernel."""

        return {"target_accept_prob": 0.999, "max_tree_depth": 12}

    @staticmethod
    @jax.jit
    def _predict_relaxation_jax(
        t: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = G_m E_{α,1}(-(t/τ)^α)

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

        # Clip alpha but allow traced values when running inside JAX
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute relaxation modulus
        t_safe = jnp.maximum(t, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # Compute argument for Mittag-Leffler function
        # z = - (t/τ)^α
        z = -jnp.power(t_safe / tau_alpha_safe, alpha_safe)

        # Compute E_{α,1}(z)
        ml_value = mittag_leffler_e2(z, alpha=alpha_safe, beta=1.0)

        # Compute G(t)
        # G(t) = Gm * E(...)
        G_t = Gm * ml_value

        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep_jax(
        t: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        J(t) = (1/G_m) + (t^α)/(G_m τ^α Γ(1+α))

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

        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute creep compliance
        t_safe = jnp.maximum(t, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # Instantaneous compliance (elastic part)
        J_instant = 1.0 / Gm

        # Viscous/Fractional part
        # J_frac = t^α / (G_m * τ^α * Γ(1+α))
        num = jnp.power(t_safe, alpha_safe)
        denom = (
            Gm
            * jnp.power(tau_alpha_safe, alpha_safe)
            * jax.scipy.special.gamma(1.0 + alpha_safe)
        )

        J_t = J_instant + num / denom

        return J_t

    @staticmethod
    @jax.jit
    def _predict_oscillation_jax(
        omega: jnp.ndarray, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = G_m (iωτ_α)^α / (1 + (iωτ_α)^α)

        Args:
            omega: Angular frequency array
            Gm: Maxwell modulus
            alpha: Power-law exponent
            tau_alpha: Relaxation time

        Returns:
            Complex modulus array [G', G'']
        """
        # Add small epsilon
        epsilon = 1e-12

        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)

        # Compute oscillation response
        omega_safe = jnp.maximum(omega, epsilon)
        tau_alpha_safe = jnp.maximum(tau_alpha, epsilon)

        # (iωτ_α)^α = |ωτ_α|^α * exp(i α π/2)
        omega_tau = omega_safe * tau_alpha_safe
        omega_tau_alpha = jnp.power(omega_tau, alpha_safe)
        phase_alpha = jnp.pi * alpha_safe / 2.0

        cos_phase = jnp.cos(phase_alpha)
        sin_phase = jnp.sin(phase_alpha)

        i_omega_tau_alpha = omega_tau_alpha * (cos_phase + 1j * sin_phase)

        # Complex modulus
        # G* = Gm * X / (1 + X) where X = (iωτ)^α
        G_star = Gm * i_omega_tau_alpha / (1.0 + i_omega_tau_alpha)

        return jnp.stack([jnp.real(G_star), jnp.imag(G_star)], axis=-1)

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

        # Determine data shape for logging
        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger,
            model="FractionalMaxwellLiquid",
            data_shape=data_shape,
            test_mode=test_mode if isinstance(test_mode, str) else str(test_mode),
        ) as ctx:
            logger.debug(
                "Starting FML fit",
                n_points=len(X) if hasattr(X, "__len__") else 1,
                test_mode=str(test_mode),
                initial_params=self.parameters.to_dict(),
            )

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == "oscillation":
                try:
                    from rheojax.utils.initialization import (
                        initialize_fractional_maxwell_liquid,
                    )

                    success = initialize_fractional_maxwell_liquid(
                        np.array(X), np.array(y), self.parameters
                    )
                    if success:
                        logger.debug(
                            "Smart initialization applied from frequency-domain features",
                            initialized_params=self.parameters.to_dict(),
                        )
                except Exception as e:
                    logger.debug(
                        "Smart initialization failed, using defaults",
                        error=str(e),
                    )

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

            logger.debug(
                "Creating least squares objective",
                normalize=True,
                use_log_residuals=use_log_residuals,
            )
            objective = create_least_squares_objective(
                model_fn,
                x_data,
                y_data,
                normalize=True,
                use_log_residuals=use_log_residuals,
            )

            # Choose optimization strategy
            try:
                if use_multi_start:
                    from rheojax.utils.optimization import nlsq_multistart_optimize

                    logger.debug(
                        "Starting multi-start NLSQ optimization",
                        n_starts=n_starts,
                        perturb_factor=perturb_factor,
                        method=kwargs.get("method", "auto"),
                        max_iter=kwargs.get("max_iter", 1000),
                    )
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
                    logger.debug(
                        "Starting NLSQ optimization",
                        method=kwargs.get("method", "auto"),
                        max_iter=kwargs.get("max_iter", 1000),
                    )
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
                    error=str(e),
                    exc_info=True,
                )
                raise

            # Validate optimization succeeded
            if not result.success:
                logger.error(
                    "Optimization failed",
                    message=result.message,
                    final_params=self.parameters.to_dict(),
                )
                raise RuntimeError(
                    f"Optimization failed: {result.message}. "
                    f"Try adjusting initial values, bounds, or max_iter."
                )

            self.fitted_ = True
            ctx["final_params"] = self.parameters.to_dict()
            ctx["success"] = True
            logger.debug(
                "FML fit completed successfully",
                final_params=self.parameters.to_dict(),
            )

        return self

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Internal predict implementation.

        Args:
            X: RheoData object or array of x-values
            **kwargs: Additional arguments (test_mode handled via self._test_mode)

        Returns:
            Predicted values
        """
        # Handle RheoData input
        if isinstance(X, RheoData):
            return self.predict_rheodata(X)

        # Handle raw array input
        from rheojax.core.test_modes import TestMode

        x = jnp.asarray(X)
        Gm = self.parameters.get_value("Gm")
        alpha = self.parameters.get_value("alpha")
        tau_alpha = self.parameters.get_value("tau_alpha")

        test_mode = getattr(self, "_test_mode", None) or kwargs.get("test_mode")
        if test_mode in ("oscillation", TestMode.OSCILLATION):
            result = self._predict_oscillation_jax(x, Gm, alpha, tau_alpha)
        elif test_mode in ("creep", TestMode.CREEP):
            result = self._predict_creep_jax(x, Gm, alpha, tau_alpha)
        elif test_mode in ("flow_curve", "rotation", TestMode.FLOW_CURVE, TestMode.ROTATION):
            # Flow curve: use relaxation-based prediction
            result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
        else:
            result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
        return np.array(result)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Gm, alpha, tau_alpha]

        Returns:
            Model predictions as JAX array
        """
        # Use explicit test_mode parameter (closure-captured in fit_bayesian)
        # Fall back to self._test_mode only for backward compatibility
        if test_mode is None:
            test_mode = getattr(self, "_test_mode", "relaxation")

        # Normalize test_mode to string
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        # Extract parameter names from function signature
        params_dict = {name: params[i] for i, name in enumerate(self.parameters.keys())}

        # Dispatch to appropriate prediction method
        if test_mode == "relaxation":
            return self._predict_relaxation_jax(X, **params_dict)
        elif test_mode == "creep":
            return self._predict_creep_jax(X, **params_dict)
        elif test_mode == "oscillation":
            # Return complex array for oscillation mode
            complex_result = self._predict_oscillation_jax(X, **params_dict)
            return complex_result[..., 0] + 1j * complex_result[..., 1]
        else:
            # Default to relaxation for unknown modes
            return self._predict_relaxation_jax(X, **params_dict)

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
        if not isinstance(test_mode, str) or not test_mode:
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
            # Return complex array for RheoData [G' + iG'']
            y_pred_stacked = self._predict_oscillation_jax(x, Gm, alpha, tau_alpha)
            y_pred = y_pred_stacked[..., 0] + 1j * y_pred_stacked[..., 1]
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

    def predict(self, X, test_mode: str | None = None, **kwargs):
        """Predict response.

        Args:
            X: RheoData object or array of x-values
            test_mode: Test mode for prediction ('relaxation', 'creep', 'oscillation',
                       'flow_curve'). If None, defaults to 'relaxation'.
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Predicted values (RheoData if input is RheoData, else array)
        """
        if isinstance(X, RheoData):
            return self.predict_rheodata(X, test_mode=test_mode)
        else:
            # Get parameters
            Gm = self.parameters.get_value("Gm")
            alpha = self.parameters.get_value("alpha")
            tau_alpha = self.parameters.get_value("tau_alpha")
            x = jnp.asarray(X)

            # Normalize test_mode to string
            mode = test_mode or "relaxation"
            if hasattr(mode, "value"):
                mode = mode.value

            # Route to appropriate prediction method based on test_mode
            if mode == "relaxation":
                result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
            elif mode == "creep":
                result = self._predict_creep_jax(x, Gm, alpha, tau_alpha)
            elif mode == "oscillation":
                result = self._predict_oscillation_jax(x, Gm, alpha, tau_alpha)
            elif mode in ("flow_curve", "rotation"):
                result = self._predict_relaxation_jax(x, Gm, alpha, tau_alpha)
            else:
                raise ValueError(
                    f"Unknown test mode: {mode}. "
                    f"Must be 'relaxation', 'creep', 'oscillation', or 'flow_curve'"
                )
            return np.array(result)
