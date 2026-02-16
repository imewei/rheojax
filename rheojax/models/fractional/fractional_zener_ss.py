"""Fractional Zener Solid-Solid (FZSS) Model.

This model combines two springs and one SpringPot, providing both
instantaneous and equilibrium elasticity with fractional relaxation.

Theory
------
The FZSS model consists of:
- Spring (G_e) in parallel with
- Series combination of spring (G_m) and SpringPot

Relaxation modulus:
    G(t) = G_e + G_m * E_α(-(t/τ_α)^α)

Complex modulus:
    G*(ω) = G_e + G_m / (1 + (iωτ_α)^(-α))

where E_α is the one-parameter Mittag-Leffler function.

Parameters
----------
Ge : float
    Equilibrium modulus (Pa), bounds [1e-3, 1e9]
Gm : float
    Maxwell arm modulus (Pa), bounds [1e-3, 1e9]
alpha : float
    Fractional order, bounds [0.0, 1.0]
tau_alpha : float
    Relaxation time (s^α), bounds [1e-6, 1e6]

Limit Cases
-----------
- alpha → 0: Two springs in parallel (G = G_e + G_m)
- alpha → 1: Classical Zener solid with G(t) = G_e + G_m*exp(-t/τ)

References
----------
- Mainardi, F. (2010). Fractional Calculus and Waves in Linear Viscoelasticity
- Schiessel, H., et al. (1995). J. Phys. A: Math. Gen. 28, 6567
"""

from __future__ import annotations

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger, log_fit
from rheojax.models.fractional.fractional_mixin import FRACTIONAL_ORDER_BOUNDS

jax, jnp = safe_import_jax()


from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.utils.compatibility import format_compatibility_message
from rheojax.utils.mittag_leffler import mittag_leffler_e

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "fractional_zener_ss",
    protocols=[
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.OSCILLATION,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class FractionalZenerSolidSolid(BaseModel):
    """Fractional Zener Solid-Solid model.

    A fractional viscoelastic model with both instantaneous and
    equilibrium elasticity.

    Test Modes
    ----------
    - Relaxation: Supported
    - Creep: Supported
    - Oscillation: Supported
    - Rotation: Not supported (no steady-state flow)

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from rheojax.models import FractionalZenerSolidSolid
    >>>
    >>> # Create model
    >>> model = FractionalZenerSolidSolid()
    >>>
    >>> # Set parameters
    >>> model.set_params(Ge=1000.0, Gm=500.0, alpha=0.5, tau_alpha=1.0)
    >>>
    >>> # Predict relaxation modulus
    >>> t = jnp.logspace(-2, 2, 50)
    >>> G_t = model.predict(t)
    """

    def __init__(self):
        """Initialize Fractional Zener Solid-Solid model."""
        super().__init__()

        # Define parameters with bounds and descriptions
        self.parameters = ParameterSet()
        self.parameters.add(
            name="Ge",
            value=1000.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Equilibrium modulus",
        )
        self.parameters.add(
            name="Gm",
            value=1000.0,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Maxwell arm modulus",
        )
        self.parameters.add(
            name="alpha",
            value=0.5,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="",
            description="Fractional order",
        )
        self.parameters.add(
            name="tau_alpha",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="s^α",
            description="Relaxation time",
        )

    @staticmethod
    @jax.jit
    def _predict_relaxation_jax(
        t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t) using JAX.

        G(t) = G_e + G_m * E_α(-(t/τ_α)^α)
        """
        epsilon = 1e-12
        # Clip alpha using JAX operations (tracer-safe)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        tau_alpha_safe = tau_alpha + epsilon

        # Compute argument: z = -(t/τ_α)^α
        z = -jnp.power(t / tau_alpha_safe, alpha_safe)

        # Mittag-Leffler function E_α(z)
        ml_term = mittag_leffler_e(z, alpha_safe)

        # G(t) = G_e + G_m * E_α(-(t/τ_α)^α)
        return Ge + Gm * ml_term

    def _predict_relaxation(
        self, t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict relaxation modulus G(t).

        Wrapper for JIT-compiled implementation.
        """
        return self._predict_relaxation_jax(t, Ge, Gm, alpha, tau_alpha)

    @staticmethod
    @jax.jit
    def _predict_creep_jax(
        t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t) using JAX.

        For FZSS, creep compliance is:
        J(t) = 1/(G_e + G_m) + (1/G_e - 1/(G_e + G_m)) * (1 - E_α(-(t/τ_α)^α))
        """
        epsilon = 1e-12
        # Clip alpha using JAX operations (tracer-safe)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        tau_alpha_safe = tau_alpha + epsilon

        # Instantaneous and equilibrium compliances
        G_total = Ge + Gm + epsilon
        J_inst = 1.0 / G_total
        J_eq = 1.0 / (Ge + epsilon)

        # Compute argument: z = -(t/τ_α)^α
        z = -jnp.power(t / tau_alpha_safe, alpha_safe)

        # Mittag-Leffler function
        ml_term = mittag_leffler_e(z, alpha_safe)

        # J(t) = J_inst + (J_eq - J_inst) * (1 - E_α(-t^α/τ_α))
        return J_inst + (J_eq - J_inst) * (1.0 - ml_term)

    def _predict_creep(
        self, t: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict creep compliance J(t).

        Wrapper for JIT-compiled implementation.
        """
        return self._predict_creep_jax(t, Ge, Gm, alpha, tau_alpha)

    @staticmethod
    @jax.jit
    def _predict_oscillation_jax(
        omega: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) using JAX.

        G*(ω) = G_e + G_m / (1 + (iωτ_α)^(-α))
        """
        epsilon = 1e-12
        # Clip alpha using JAX operations (tracer-safe)
        alpha_safe = jnp.clip(alpha, epsilon, 1.0 - epsilon)
        tau_alpha_safe = tau_alpha + epsilon

        # Compute (iω)^(-α) = ω^(-α) * exp(-i*π*α/2)
        omega_neg_alpha = jnp.power(omega, -alpha_safe)
        phase = -jnp.pi * alpha_safe / 2.0

        # (iω)^(-α) in complex form
        i_omega_neg_alpha = omega_neg_alpha * (jnp.cos(phase) + 1j * jnp.sin(phase))

        # Denominator: 1 + (iωτ_α)^(-α) = 1 + τ_α^(-α) * (iω)^(-α)
        tau_neg_alpha = jnp.power(tau_alpha_safe, -alpha_safe)
        denominator = 1.0 + tau_neg_alpha * i_omega_neg_alpha

        # Maxwell arm contribution: G_m / (1 + (iωτ_α)^(-α))
        maxwell_term = Gm / denominator

        # Total complex modulus
        G_star = Ge + maxwell_term

        # Extract storage and loss moduli
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)

        return jnp.stack([G_prime, G_double_prime], axis=-1)

    def _predict_oscillation(
        self, omega: jnp.ndarray, Ge: float, Gm: float, alpha: float, tau_alpha: float
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω).

        Wrapper for JIT-compiled implementation.
        """
        return self._predict_oscillation_jax(omega, Ge, Gm, alpha, tau_alpha)

    def _fit(
        self, X: jnp.ndarray, y: jnp.ndarray, **kwargs
    ) -> FractionalZenerSolidSolid:
        """Fit model to data using NLSQ TRF optimization.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable (time or frequency)
        y : jnp.ndarray
            Dependent variable (modulus or compliance)
        **kwargs : dict
            Additional fitting options

        Returns
        -------
        self
            Fitted model instance
        """
        from rheojax.core.test_modes import TestMode
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Detect test mode
        test_mode_str = kwargs.get("test_mode", "relaxation")

        # Convert string to TestMode enum
        if isinstance(test_mode_str, str):
            test_mode_map = {
                "relaxation": TestMode.RELAXATION,
                "creep": TestMode.CREEP,
                "oscillation": TestMode.OSCILLATION,
            }
            test_mode = test_mode_map.get(test_mode_str, TestMode.RELAXATION)
        else:
            test_mode = test_mode_str

        # Store test mode for model_function
        self._test_mode = test_mode

        # Provide simple data-aware initialization for relaxation fits
        compatibility_guard = kwargs.pop("compatibility_guard", False)
        incompat_confidence = kwargs.pop("compatibility_confidence_threshold", 0.65)
        compatibility_report = None

        # Determine data shape for logging
        data_shape = (len(X),) if hasattr(X, "__len__") else None

        with log_fit(
            logger,
            model="FractionalZenerSolidSolid",
            data_shape=data_shape,
            test_mode=(
                test_mode_str if isinstance(test_mode_str, str) else str(test_mode)
            ),
        ) as ctx:
            logger.debug(
                "Starting FZSS fit",
                n_points=len(X) if hasattr(X, "__len__") else 1,
                test_mode=str(test_mode),
                initial_params=self.parameters.to_dict(),
                compatibility_guard=compatibility_guard,
            )

            if test_mode == TestMode.RELAXATION:
                self._initialize_relaxation_parameters(X, y)
                logger.debug(
                    "Relaxation parameters initialized",
                    initialized_params=self.parameters.to_dict(),
                )

                if compatibility_guard:
                    compatibility_report = _compute_relaxation_compatibility(self, X, y)
                    if compatibility_report and _should_block_relaxation_fit(
                        compatibility_report, incompat_confidence
                    ):
                        message = format_compatibility_message(compatibility_report)
                        logger.error(
                            "Data incompatible with FZSS model",
                            compatibility_report=compatibility_report,
                        )
                        raise RuntimeError(
                            "Optimization failed: data is incompatible with "
                            "FractionalZenerSolidSolid.\n"
                            f"Model-data compatibility:\n{message}"
                        )

            # Smart initialization for oscillation mode (Issue #9)
            if test_mode == TestMode.OSCILLATION:
                try:
                    import numpy as np

                    from rheojax.utils.initialization import (
                        initialize_fractional_zener_ss,
                    )

                    success = initialize_fractional_zener_ss(
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

            # Create stateless model function for optimization
            def model_fn(x, params):
                """Model function for optimization (stateless)."""
                Ge, Gm, alpha, tau_alpha = params[0], params[1], params[2], params[3]

                # Direct prediction based on test mode (stateless)
                if test_mode == TestMode.RELAXATION:
                    return self._predict_relaxation(x, Ge, Gm, alpha, tau_alpha)
                elif test_mode == TestMode.CREEP:
                    return self._predict_creep(x, Ge, Gm, alpha, tau_alpha)
                elif test_mode == TestMode.OSCILLATION:
                    return self._predict_oscillation(x, Ge, Gm, alpha, tau_alpha)
                else:
                    raise ValueError(f"Unsupported test mode: {test_mode}")

            # Create objective function
            logger.debug("Creating least squares objective", normalize=True)
            objective = create_least_squares_objective(
                model_fn, jnp.array(X), jnp.array(y), normalize=True
            )

            # Optimize using NLSQ TRF
            logger.debug(
                "Starting NLSQ optimization",
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

            # Detect incompatible relaxation data even when optimization converges
            if test_mode == TestMode.RELAXATION and compatibility_guard:
                if compatibility_report is None:
                    compatibility_report = _compute_relaxation_compatibility(self, X, y)

                if compatibility_report and _should_block_relaxation_fit(
                    compatibility_report, incompat_confidence
                ):
                    message = format_compatibility_message(compatibility_report)
                    logger.error(
                        "Post-fit compatibility check failed",
                        compatibility_report=compatibility_report,
                    )
                    raise RuntimeError(
                        "Optimization failed: data is incompatible with "
                        "FractionalZenerSolidSolid.\n"
                        f"Model-data compatibility:\n{message}"
                    )

            self.fitted_ = True
            ctx["final_params"] = self.parameters.to_dict()
            ctx["success"] = True
            logger.debug(
                "FZSS fit completed successfully",
                final_params=self.parameters.to_dict(),
            )

        return self

    def _initialize_relaxation_parameters(self, X, y) -> bool:
        """Derive heuristic starting values from relaxation data."""
        import logging

        import numpy as np

        try:
            t = np.asarray(X, dtype=float).ravel()
            g = np.asarray(y, dtype=float).ravel()
            if t.shape != g.shape or t.size < 4:
                return False

            order = np.argsort(t)
            t_sorted = t[order]
            g_sorted = g[order]

            tail = max(3, t_sorted.size // 6)
            head = max(3, t_sorted.size // 6)
            ge_guess = float(np.median(g_sorted[-tail:]))
            gm_guess = float(np.median(g_sorted[:head]) - ge_guess)
            gm_guess = max(gm_guess, 1e-3)

            ge_param = self.parameters.get("Ge")
            gm_param = self.parameters.get("Gm")
            tau_param = self.parameters.get("tau_alpha")
            alpha_param = self.parameters.get("alpha")
            assert ge_param is not None and ge_param.bounds is not None
            assert gm_param is not None and gm_param.bounds is not None
            assert tau_param is not None and tau_param.bounds is not None
            assert alpha_param is not None and alpha_param.bounds is not None
            ge_bounds = ge_param.bounds
            gm_bounds = gm_param.bounds
            tau_bounds = tau_param.bounds
            alpha_bounds = alpha_param.bounds

            ge_guess = float(np.clip(ge_guess, ge_bounds[0], ge_bounds[1]))
            gm_guess = float(np.clip(gm_guess, gm_bounds[0], gm_bounds[1]))

            normalized = np.clip((g_sorted - ge_guess) / (gm_guess + 1e-9), 0.0, 1.0)
            target = np.exp(-1.0)
            idx = int(np.argmin(np.abs(normalized - target)))
            tau_guess = float(np.clip(t_sorted[idx], tau_bounds[0], tau_bounds[1]))

            alpha_guess = float(np.clip(0.6, alpha_bounds[0], alpha_bounds[1]))

            self.parameters.set_value("Ge", ge_guess)
            self.parameters.set_value("Gm", gm_guess)
            self.parameters.set_value("tau_alpha", tau_guess)
            self.parameters.set_value("alpha", alpha_guess)
            logging.debug(
                "FZSS relaxation init | Ge=%.3g Gm=%.3g tau_alpha=%.3g alpha=%.2f",
                ge_guess,
                gm_guess,
                tau_guess,
                alpha_guess,
            )
            return True
        except Exception as exc:  # pragma: no cover - fallback only
            logging.debug(f"Relaxation initialization failed: {exc}")
            return False

    def _predict(self, X: jnp.ndarray) -> jnp.ndarray:  # type: ignore[override]
        """Predict response for given input.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable

        Returns
        -------
        jnp.ndarray
            Predicted values
        """
        # Get parameter values
        Ge = self.parameters.get_value("Ge")
        Gm = self.parameters.get_value("Gm")
        alpha = self.parameters.get_value("alpha")
        tau_alpha = self.parameters.get_value("tau_alpha")
        assert Ge is not None and Gm is not None and alpha is not None and tau_alpha is not None

        # Auto-detect test mode based on input characteristics
        # NOTE: This is a heuristic - explicit test_mode is recommended
        # Default to relaxation for time-domain data
        # Oscillation should typically use RheoData with domain='frequency'
        return self._predict_relaxation(X, Ge, Gm, alpha, tau_alpha)

    def model_function(self, X, params, test_mode=None, **kwargs):
        """Model function for Bayesian inference.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [Ge, Gm, alpha, tau_alpha]

        Returns:
            Model predictions as JAX array
        """
        from rheojax.core.test_modes import TestMode

        # Extract parameters from array (in order they were added to ParameterSet)
        Ge = params[0]
        Gm = params[1]
        alpha = params[2]
        tau_alpha = params[3]

        # Use test_mode from last fit if available, otherwise default to RELAXATION
        # Use explicit test_mode parameter (closure-captured in fit_bayesian)

        # Fall back to self._test_mode only for backward compatibility

        if test_mode is None:

            test_mode = getattr(self, "_test_mode", TestMode.RELAXATION)

        # Normalize test_mode to handle both string and TestMode enum
        if hasattr(test_mode, "value"):
            test_mode = test_mode.value

        # Call appropriate prediction function based on test mode
        if test_mode == TestMode.RELAXATION:
            return self._predict_relaxation(X, Ge, Gm, alpha, tau_alpha)
        elif test_mode == TestMode.CREEP:
            return self._predict_creep(X, Ge, Gm, alpha, tau_alpha)
        elif test_mode == TestMode.OSCILLATION:
            # _predict_oscillation returns stacked (n, 2) for NLSQ fitting,
            # but Bayesian inference needs complex array for gradient compatibility
            stacked = self._predict_oscillation(X, Ge, Gm, alpha, tau_alpha)
            # Convert [G', G''] stacked array to complex G* = G' + i*G''
            return stacked[:, 0] + 1j * stacked[:, 1]
        else:
            # Default to relaxation mode for FZSS model
            return self._predict_relaxation(X, Ge, Gm, alpha, tau_alpha)


def _should_block_relaxation_fit(compat: dict, minimum_confidence: float) -> bool:
    """Return True when compatibility analysis flags obvious mismatches."""

    if compat.get("compatible", True):
        return False

    if compat.get("confidence", 0.0) < minimum_confidence:
        return False

    try:
        from rheojax.utils.compatibility import DecayType, MaterialType
    except Exception:  # pragma: no cover - defensive guard
        return False

    decay_type = compat.get("decay_type")
    material_type = compat.get("material_type")

    return (
        decay_type == DecayType.EXPONENTIAL
        or material_type == MaterialType.VISCOELASTIC_LIQUID
    )


def _compute_relaxation_compatibility(model, X, y) -> dict | None:
    """Best-effort compatibility evaluation for relaxation data."""

    try:
        import numpy as np

        from rheojax.utils.compatibility import check_model_compatibility

        return check_model_compatibility(
            model,
            t=np.asarray(X, dtype=float),
            G_t=np.asarray(y, dtype=float),
            test_mode="relaxation",
        )
    except Exception:  # pragma: no cover - compatibility is heuristic
        return None


# Convenience alias
FZSS = FractionalZenerSolidSolid

__all__ = ["FractionalZenerSolidSolid", "FZSS"]
