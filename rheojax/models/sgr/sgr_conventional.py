"""Soft Glassy Rheology (SGR) Conventional Model.

This module implements the conventional SGR model from Sollich 1998, describing
the rheology of soft glassy materials including foams, emulsions, pastes, and
colloidal suspensions through a statistical mechanics framework.

The SGR model is characterized by:
- Exponential distribution of energy traps: rho(E) ~ exp(-E)
- Effective noise temperature x controlling material phase:
  * x < 1: Glass phase with yield stress (solid-like)
  * 1 < x < 2: Power-law viscoelastic fluid with G' ~ G'' ~ omega^(x-1)
  * x >= 2: Newtonian viscous liquid
- Mesoscopic elements undergoing thermally activated hopping between traps
- Characteristic power-law rheological responses

Constitutive Equations:
    Equilibrium modulus:
        G0(x) = integral_0^inf rho(E) * E * (1 - exp(-E/x)) dE

    Frequency-dependent complex modulus:
        G*(omega) = G_scale * Gp(x, omega*tau0)
        where Gp(x, z) = integral_0^inf rho(E) * E * [i*z] / [i*z + exp(E/x)] dE

Model Parameters:
    x (dimensionless): Effective noise temperature, range (0.5, 3.0), default 1.5
        Controls material phase transition and power-law exponent
    G0 (Pa): Modulus scale, range (1e-3, 1e9), default 1e3
        Sets absolute magnitude of elastic response
    tau0 (s): Attempt time, range (1e-9, 1e3), default 1e-3
        Characteristic microscopic relaxation timescale

Physical Interpretation:
    The noise temperature x quantifies the ratio of thermal fluctuations to
    the energy scale of structural rearrangements. At x=1, the material undergoes
    a glass transition from solid-like (x<1) to fluid-like (x>1) behavior. The
    power-law exponent (x-1) directly reflects the breadth of the distribution
    of relaxation times arising from the exponential trap distribution.

References:
    - P. Sollich, Rheological constitutive equation for a model of soft glassy
      materials, Physical Review E, 1998, 58(1), 738-759
    - P. Sollich et al., Rheology of Soft Glassy Materials, Physical Review
      Letters, 1997, 78(10), 2020-2023
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import diffrax
import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode
from rheojax.logging import get_logger, log_fit
from rheojax.utils.sgr_kernels import G0, Gp

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp_typing
else:
    jnp_typing = np

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "sgr_conventional",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.STARTUP,
        Protocol.OSCILLATION,
        Protocol.LAOS,
    ],
)
class SGRConventional(BaseModel):
    """Soft Glassy Rheology (SGR) Conventional Model.

    Statistical mechanics model for soft glassy materials (foams, emulsions,
    pastes, colloidal suspensions) based on Sollich 1998. The model describes
    rheological behavior through a population of mesoscopic elements trapped
    in energy wells with exponential distribution rho(E) ~ exp(-E), undergoing
    thermally activated hopping.

    The effective noise temperature x controls material phase:
    - x < 1: Glass with yield stress (solid-like)
    - 1 < x < 2: Power-law viscoelastic fluid (G' ~ G'' ~ omega^(x-1))
    - x >= 2: Newtonian liquid

    Parameters:
        x: Effective noise temperature (dimensionless), controls phase transition
        G0: Modulus scale (Pa), sets absolute magnitude of elastic response
        tau0: Attempt time (s), characteristic microscopic relaxation timescale

    Attributes:
        parameters: ParameterSet containing x, G0, tau0

    Example:
        >>> from rheojax.models.sgr_conventional import SGRConventional
        >>> import numpy as np
        >>> model = SGRConventional()
        >>> omega = np.logspace(-2, 2, 50)
        >>> # Fitting oscillation data
        >>> model.fit(omega, G_star, test_mode='oscillation')
        >>> G_star_pred = model.predict(omega)
        >>> # Bayesian inference
        >>> result = model.fit_bayesian(omega, G_star, num_samples=2000)
        >>> intervals = model.get_credible_intervals(result.posterior_samples)

    Notes:
        - Inherits from BaseModel (includes BayesianMixin for NumPyro NUTS)
        - Mode-aware Bayesian inference (stores test_mode for correct predictions)
        - JAX-accelerated kernel functions for GPU computation
        - Float64 precision critical for numerical stability near x=1
    """

    def __init__(self, dynamic_x: bool = False):
        """Initialize SGR Conventional Model.

        Creates ParameterSet with:
        - x (noise temperature): bounds (0.5, 3.0), default 1.5
        - G0 (modulus scale): bounds (1e-3, 1e9), default 1e3
        - tau0 (attempt time): bounds (1e-9, 1e3), default 1e-3

        When dynamic_x=True, additional parameters for x(t) evolution:
        - x_eq: Equilibrium noise temperature at rest
        - alpha_aging: Aging rate (drives x toward x_eq)
        - beta_rejuv: Rejuvenation rate (shear-induced increase in x)
        - x_ss_A: Steady-state amplitude parameter
        - x_ss_n: Steady-state power-law exponent

        Args:
            dynamic_x: If True, x evolves via dx/dt equation. If False, x is constant.

        The model is ready for fitting after instantiation.
        """
        super().__init__()

        # Create parameter set
        self.parameters = ParameterSet()

        # x: Effective noise temperature (dimensionless)
        # Range: 0.5 to 3.0 covers glass -> power-law fluid -> Newtonian
        # Default: 1.5 (middle of power-law regime)
        self.parameters.add(
            name="x",
            value=1.5,
            bounds=(0.5, 3.0),
            units="dimensionless",
            description="Effective noise temperature (glass transition at x=1)",
        )

        # G0: Modulus scale (Pa)
        # Range: 1e-3 to 1e9 Pa covers soft gels to stiff pastes
        # Default: 1e3 Pa (typical soft gel)
        self.parameters.add(
            name="G0",
            value=1e3,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Modulus scale (absolute magnitude of elastic response)",
        )

        # tau0: Attempt time (s)
        # Range: 1e-9 to 1e3 s covers molecular to macroscopic timescales
        # Default: 1e-3 s (millisecond, typical for colloidal relaxation)
        self.parameters.add(
            name="tau0",
            value=1e-3,
            bounds=(1e-9, 1e3),
            units="s",
            description="Attempt time (microscopic relaxation timescale)",
        )

        # Dynamic x(t) mode flag
        self.dynamic_x = dynamic_x

        # Add dynamic x parameters if enabled
        if dynamic_x:
            # x_eq: Equilibrium noise temperature at rest
            self.parameters.add(
                name="x_eq",
                value=1.0,
                bounds=(0.5, 2.5),
                units="dimensionless",
                description="Equilibrium noise temperature at rest (aging target)",
            )

            # alpha_aging: Aging rate coefficient
            self.parameters.add(
                name="alpha_aging",
                value=0.1,
                bounds=(0.0, 10.0),
                units="1/s",
                description="Aging rate coefficient (drives x toward x_eq)",
            )

            # beta_rejuv: Rejuvenation rate coefficient
            self.parameters.add(
                name="beta_rejuv",
                value=0.5,
                bounds=(0.0, 10.0),
                units="dimensionless",
                description="Rejuvenation rate coefficient (shear-induced x increase)",
            )

            # x_ss_A: Steady-state amplitude parameter
            self.parameters.add(
                name="x_ss_A",
                value=0.5,
                bounds=(0.0, 2.0),
                units="dimensionless",
                description="Steady-state amplitude: x_ss = x_eq + A*(gamma_dot*tau0)^n",
            )

            # x_ss_n: Steady-state power-law exponent
            self.parameters.add(
                name="x_ss_n",
                value=0.3,
                bounds=(0.0, 1.0),
                units="dimensionless",
                description="Steady-state power-law exponent",
            )

        # Storage for x(t) trajectory in dynamic mode
        self._x_trajectory: np.ndarray | None = None
        self._t_trajectory: np.ndarray | None = None

        # Store test mode for mode-aware Bayesian inference
        self._test_mode: TestMode | str | None = None

        # LAOS attributes
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Fit SGR model to data using NLSQ optimization.

        Routes to appropriate fitting method based on test_mode. This is the
        internal implementation required by BaseModel.

        Args:
            X: Independent variable (frequency for oscillation, time for relaxation)
            y: Dependent variable (complex modulus, relaxation modulus, etc.)
            test_mode: Test mode ('oscillation', 'relaxation', 'creep', 'steady_shear')
            **kwargs: NLSQ optimizer arguments (max_iter, ftol, xtol, gtol)

        Raises:
            ValueError: If test_mode not provided or invalid
            RuntimeError: If optimization fails to converge

        Note:
            NLSQ provides 5-270x speedup over scipy. Fitted parameters serve
            as warm-start for Bayesian inference via fit_bayesian().
        """
        # Detect test mode
        if test_mode is None:
            raise ValueError("test_mode must be specified for SGR fitting")

        with log_fit(logger, model="SGRConventional", data_shape=X.shape) as ctx:
            try:
                logger.info(
                    "Starting SGR Conventional model fit",
                    test_mode=test_mode,
                    n_points=len(X),
                    dynamic_x=self.dynamic_x,
                )

                logger.debug(
                    "Input data statistics",
                    x_range=(float(np.min(X)), float(np.max(X))),
                    y_range=(float(np.min(np.abs(y))), float(np.max(np.abs(y)))),
                )

                # Store test mode for mode-aware Bayesian inference
                self._test_mode = test_mode
                ctx["test_mode"] = test_mode

                # Route to appropriate fitting method
                if test_mode == "oscillation":
                    self._fit_oscillation_mode(X, y, **kwargs)
                elif test_mode == "relaxation":
                    self._fit_relaxation_mode(X, y, **kwargs)
                elif test_mode == "creep":
                    self._fit_creep_mode(X, y, **kwargs)
                elif test_mode == "steady_shear":
                    self._fit_steady_shear_mode(X, y, **kwargs)
                elif test_mode == "laos":
                    self._fit_laos_mode(X, y, **kwargs)
                elif test_mode == "startup":
                    self._fit_startup_mode(X, y, **kwargs)
                else:
                    raise ValueError(
                        f"Unsupported test_mode: {test_mode}. "
                        f"SGR model supports 'oscillation', 'relaxation', 'creep', "
                        f"'steady_shear', 'laos', and 'startup'."
                    )

                # Log final parameters
                x_val = self.parameters.get_value("x")
                G0_val = self.parameters.get_value("G0")
                tau0_val = self.parameters.get_value("tau0")

                ctx["x"] = x_val
                ctx["G0"] = G0_val
                ctx["tau0"] = tau0_val
                ctx["phase_regime"] = self.get_phase_regime()

                logger.info(
                    "SGR Conventional model fit completed",
                    x=x_val,
                    G0=G0_val,
                    tau0=tau0_val,
                    phase_regime=self.get_phase_regime(),
                )

            except Exception as e:
                logger.error(
                    "SGR Conventional model fit failed",
                    test_mode=test_mode,
                    error=str(e),
                    exc_info=True,
                )
                raise

    def _fit_oscillation_mode(
        self,
        omega: np.ndarray,
        G_star: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to complex modulus data (oscillation mode).

        Uses NLSQ-accelerated optimization (5-270x faster than scipy) to fit
        SGR parameters [x, G0, tau0] to complex modulus data G*(omega).

        Args:
            omega: Angular frequency array (rad/s)
            G_star: Complex modulus data. Accepted formats:
                - Complex array (M,) where G* = G' + i*G''
                - Real array (M, 2) where columns are [G', G'']
            **kwargs: NLSQ optimizer arguments:
                - max_iter: Maximum iterations (default: 1000)
                - ftol: Function tolerance (default: 1e-6)
                - xtol: Parameter tolerance (default: 1e-6)
                - gtol: Gradient tolerance (default: 1e-6)

        Raises:
            RuntimeError: If optimization fails to converge

        Note:
            After fitting, use fit_bayesian() for uncertainty quantification.
            The fitted parameters serve as warm-start for NUTS sampling.
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        omega_jax = jnp.asarray(omega, dtype=jnp.float64)

        # Handle G_star format: can be (M, 2) or complex (M,)
        G_star_np = np.asarray(G_star)
        if np.iscomplexobj(G_star_np):
            # Convert complex to (M, 2) [G', G'']
            G_star_2d = np.column_stack([np.real(G_star_np), np.imag(G_star_np)])
        elif G_star_np.ndim == 2 and G_star_np.shape[1] == 2:
            G_star_2d = G_star_np
        elif G_star_np.ndim == 2 and G_star_np.shape[0] == 2:
            # Transposed format (2, M) -> (M, 2)
            G_star_2d = G_star_np.T
        else:
            raise ValueError(
                f"G_star must be complex (M,) or real (M, 2), got shape {G_star_np.shape}"
            )

        G_star_jax = jnp.asarray(G_star_2d, dtype=jnp.float64)

        # Create model function for NLSQ: takes (x_data, params) -> predictions
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            """Stateless model function for optimization."""
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_oscillation_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function
        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR oscillation fitting failed: {result.message}. "
                "Try adjusting initial values, bounds, or use use_log_residuals=True."
            )

        logger.debug(
            f"SGR oscillation fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_relaxation_mode(
        self,
        t: np.ndarray,
        G_t: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to relaxation modulus data (relaxation mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to relaxation modulus data G(t).

        Args:
            t: Time array (s)
            G_t: Relaxation modulus array (Pa)
            **kwargs: NLSQ optimizer arguments:
                - max_iter: Maximum iterations (default: 1000)
                - ftol: Function tolerance (default: 1e-6)
                - xtol: Parameter tolerance (default: 1e-6)
                - gtol: Gradient tolerance (default: 1e-6)
                - use_log_residuals: Use log-space residuals (default: True)

        Raises:
            RuntimeError: If optimization fails to converge

        Note:
            Log-space residuals are recommended for relaxation data due to
            the power-law decay spanning many orders of magnitude.
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        G_t_jax = jnp.asarray(G_t, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            """Stateless model function for optimization."""
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_relaxation_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space recommended for power-law decay)
        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            G_t_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR relaxation fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR relaxation fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_creep_mode(
        self,
        t: np.ndarray,
        J_t: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to creep compliance data (creep mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to creep compliance data J(t).

        Args:
            t: Time array (s)
            J_t: Creep compliance array (1/Pa)
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        J_t_jax = jnp.asarray(J_t, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            """Stateless model function for optimization."""
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_creep_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space for compliance spanning decades)
        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            J_t_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR creep fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR creep fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_steady_shear_mode(
        self,
        gamma_dot: np.ndarray,
        eta: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to steady shear viscosity data (steady_shear mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to flow curve data eta(gamma_dot).

        Args:
            gamma_dot: Shear rate array (1/s)
            eta: Viscosity array (Pa.s)
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        eta_jax = jnp.asarray(eta, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            """Stateless model function for optimization."""
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_steady_shear_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space for viscosity spanning decades)
        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            eta_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR steady shear fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR steady shear fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_laos_mode(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to LAOS data.

        Uses Monte Carlo simulation for time-domain stress prediction,
        then optimizes parameters to match measured stress.

        Args:
            X: Time array (s)
            y: Stress array (Pa)
            **kwargs: Required kwargs:
                - gamma_0: Strain amplitude
                - omega: Angular frequency (rad/s)
                Optional kwargs:
                - n_particles: Monte Carlo particle count (default 5000)
                - max_iter: Optimizer max iterations (default 50)
                - seed: Random seed for reproducibility (default 42)

        Raises:
            ValueError: If gamma_0 or omega not provided
        """
        gamma_0 = kwargs.get("gamma_0")
        omega = kwargs.get("omega")

        if gamma_0 is None or omega is None:
            raise ValueError("LAOS fitting requires gamma_0 and omega in kwargs")

        if gamma_0 is not None:
            self._gamma_0 = gamma_0
        if omega is not None:
            self._omega_laos = omega

        n_particles = kwargs.get("n_particles", 5000)

        # Use SAOS approximation for small amplitude
        if gamma_0 < 0.1:
            logger.info(
                f"Small strain amplitude gamma_0={gamma_0}. Using SAOS approximation."
            )
            # Use JAX-native FFT for JAX-First compliance
            sigma = y
            t = X
            sigma_fft = jnp.fft.fft(jnp.asarray(sigma))
            n = len(sigma)
            fundamental_idx = int(omega * (t[-1] - t[0]) / (2 * np.pi))
            fundamental_idx = max(1, min(fundamental_idx, n // 2 - 1))

            G_star_amplitude = (
                2.0 * float(jnp.abs(sigma_fft[fundamental_idx])) / (n * gamma_0)
            )
            phase = float(jnp.angle(sigma_fft[fundamental_idx]))

            G_prime = G_star_amplitude * np.cos(phase)
            G_double_prime = G_star_amplitude * np.sin(phase)

            omega_single = np.array([omega])
            G_star_single = np.array([[G_prime, G_double_prime]])

            self._fit_oscillation_mode(omega_single, G_star_single, **kwargs)
        else:
            # Full MC-based LAOS fitting
            self._fit_laos_mc(X, y, gamma_0, omega, n_particles, **kwargs)

    def _fit_laos_mc(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_0: float,
        omega: float,
        n_particles: int,
        **kwargs,
    ) -> None:
        """Full Monte Carlo-based LAOS fitting.

        Runs MC simulations within optimization loop to match time-domain stress.

        Args:
            t: Time array (s)
            sigma: Measured stress array (Pa)
            gamma_0: Strain amplitude
            omega: Angular frequency (rad/s)
            n_particles: Number of MC particles
            **kwargs: Optimizer arguments

        Note:
            Uses scipy.optimize.minimize (L-BFGS-B) because the objective function
            calls Monte Carlo simulations which are stochastic and not JAX-traceable.
            This is acceptable per Technical Guidelines as it's used only for large-
            amplitude LAOS fitting, not the primary oscillation/relaxation modes.
        """
        from scipy.optimize import minimize

        from rheojax.utils.sgr_monte_carlo import simulate_oscillatory

        logger.info(
            f"Full MC-based LAOS fitting: {n_particles} particles, "
            f"gamma_0={gamma_0}, omega={omega:.3f} rad/s"
        )

        # Determine simulation parameters from data
        period = 2.0 * np.pi / omega
        t_total = t[-1] - t[0]
        n_cycles = max(1, int(t_total / period))
        points_per_cycle = max(10, len(t) // n_cycles)

        # Warm-start: estimate parameters from stress amplitude
        sigma_max = np.max(np.abs(sigma))
        G0_init = sigma_max / gamma_0
        x_init = self.parameters.get_value("x")
        tau0_init = self.parameters.get_value("tau0")

        # Normalize target stress for residual calculation
        sigma_norm = sigma / (sigma_max + 1e-12)

        # Fixed random seed for reproducibility
        seed = kwargs.get("seed", 42)

        def objective(params):
            """Compute residual between MC stress and measured stress."""
            x_val, log_G0, log_tau0 = params
            G0_val = np.exp(log_G0)
            tau0_val = np.exp(log_tau0)

            x_val = np.clip(x_val, 0.5, 2.5)

            try:
                key = jax.random.PRNGKey(seed)
                _, _, sigma_mc = simulate_oscillatory(
                    key=key,
                    gamma_0=gamma_0,
                    omega=omega,
                    n_cycles=n_cycles,
                    points_per_cycle=points_per_cycle,
                    x=x_val,
                    n_particles=n_particles,
                    k=G0_val,
                    Gamma0=1.0 / tau0_val,
                    xg=1.0,
                )

                t_mc = np.linspace(0, t_total, len(sigma_mc))
                sigma_mc_interp = np.interp(t - t[0], t_mc, np.array(sigma_mc))

                sigma_mc_max = np.max(np.abs(sigma_mc_interp)) + 1e-12
                sigma_mc_norm = sigma_mc_interp / sigma_mc_max

                residual = np.sum((sigma_mc_norm - sigma_norm) ** 2)
                return residual

            except Exception as e:
                logger.warning(f"MC simulation failed: {e}")
                return 1e10

        x0 = np.array([x_init, np.log(G0_init), np.log(tau0_init)])

        bounds = [
            (0.5, 2.5),
            (np.log(1e-3), np.log(1e9)),
            (np.log(1e-9), np.log(1e3)),
        ]

        max_iter = kwargs.get("max_iter", 50)
        logger.info(f"Starting MC-LAOS optimization (max {max_iter} iterations)...")

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )

        x_opt, log_G0_opt, log_tau0_opt = result.x
        self.parameters.set_value("x", float(x_opt))
        self.parameters.set_value("G0", float(np.exp(log_G0_opt)))
        self.parameters.set_value("tau0", float(np.exp(log_tau0_opt)))

        if result.success:
            logger.info(
                f"MC-LAOS fit converged: x={x_opt:.4f}, "
                f"G0={np.exp(log_G0_opt):.2e}, tau0={np.exp(log_tau0_opt):.2e}, "
                f"cost={result.fun:.3e}"
            )
        else:
            logger.warning(
                f"MC-LAOS fit did not fully converge: {result.message}. "
                f"Best: x={x_opt:.4f}, G0={np.exp(log_G0_opt):.2e}"
            )

        self.fitted_ = True

    def _fit_startup_mode(
        self,
        t: np.ndarray,
        eta_plus: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR to startup flow data (stress growth coefficient).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to stress growth coefficient η⁺(t) data from startup flow experiments.

        Args:
            t: Time array (s)
            eta_plus: Stress growth coefficient array (Pa·s).
                      If stress data is provided, must also pass gamma_dot in kwargs.
            **kwargs: NLSQ optimizer arguments, plus:
                - gamma_dot: Applied shear rate (required if y is stress, not η⁺)
                - is_stress: If True, treat y as stress and divide by gamma_dot

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Check if we need to convert stress to eta_plus
        gamma_dot = kwargs.get("gamma_dot", 1.0)
        is_stress = kwargs.get("is_stress", False)

        if is_stress:
            # Convert stress to stress growth coefficient
            eta_plus_data = eta_plus / gamma_dot
        else:
            eta_plus_data = eta_plus

        # Store gamma_dot for prediction
        self._startup_gamma_dot = gamma_dot

        # Convert inputs to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        eta_plus_jax = jnp.asarray(eta_plus_data, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            """Stateless model function for optimization."""
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_startup_jit(
                x_data, x_param, G0_param, tau0_param, gamma_dot
            )

        # Create residual function (log-space for spanning decades)
        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            eta_plus_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR startup fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR startup fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    @staticmethod
    @jax.jit
    def _predict_oscillation_jit(
        omega: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled oscillation prediction: G'(omega), G''(omega).

        Computes complex modulus from SGR kernels:
            G'(omega) = G0_scale * G_prime(x, omega*tau0)
            G''(omega) = G0_scale * G_double_prime(x, omega*tau0)

        Args:
            omega: Angular frequency array (rad/s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Complex modulus [G', G''] with shape (M, 2)

        Notes:
            - Uses Gp kernel from rheojax.utils.sgr_kernels
            - Power-law scaling G' ~ G'' ~ omega^(x-1) for 1 < x < 2
        """
        # Compute dimensionless frequency
        omega_tau0 = omega * tau0

        # Call Gp kernel (returns G_prime, G_double_prime)
        G_prime, G_double_prime = Gp(x, omega_tau0)

        # Scale by G0
        G_prime_scaled = G0_scale * G_prime
        G_double_prime_scaled = G0_scale * G_double_prime

        # Stack into (M, 2) array
        G_star = jnp.stack([G_prime_scaled, G_double_prime_scaled], axis=1)

        return G_star

    @staticmethod
    @jax.jit
    def _predict_relaxation_jit(
        t: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled relaxation prediction: G(t).

        Computes relaxation modulus with power-law decay:
            G(t) ~ G0 at short times (t << tau0)
            G(t) ~ t^(x-2) at long times (t >> tau0) for 1 < x < 2

        The relaxation modulus is related to the frequency-domain response
        via Fourier transform. For SGR, we use the inverse relationship:
            G(t) = G0 * integral of G_prime(omega) * cos(omega*t) d(omega)

        For computational efficiency, we use an analytical approximation
        based on the power-law scaling.

        Args:
            t: Time array (s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Relaxation modulus G(t) with shape (M,)

        Notes:
            - Power-law decay G(t) ~ t^(x-2) for large t
            - Plateau G(t) = G0 at short times
        """
        # Dimensionless time
        t_scaled = t / tau0

        # Compute equilibrium modulus factor (dimensionless)
        G0_dim = G0(x)

        # Relaxation modulus using power-law form
        # Power-law decay exponent: x - 2.0 (embedded in formula below)
        # At short times: G(t) -> G0_dim * G0_scale
        # At long times: G(t) ~ (t/tau0)^(x-2) * G0_scale

        # Use a smooth transition formula
        # G(t) = G0_scale * G0_dim * (1 + t_scaled)^(x-2)
        # This gives correct short-time and long-time limits

        epsilon = 1e-12  # Prevent division by zero
        t_safe = jnp.maximum(t_scaled, epsilon)

        # Power-law form with smooth crossover
        # For x < 2: decay, for x >= 2: approaches constant
        G_t = G0_scale * G0_dim / jnp.power(1.0 + t_safe, 2.0 - x)

        return G_t

    @staticmethod
    @jax.jit
    def _predict_creep_jit(
        t: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled creep prediction: J(t).

        Computes creep compliance as approximate inverse of G(t):
            J(t) = 1 / G(t)

        For more accurate results, uses proper compliance formula:
            J(t) ~ t^(2-x) / G0 for large t (1 < x < 2)

        Args:
            t: Time array (s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Creep compliance J(t) with shape (M,)

        Notes:
            - Monotonicity enforced via jnp.maximum.accumulate()
            - Power-law growth J(t) ~ t^(2-x) for large t
        """
        # Dimensionless time
        t_scaled = t / tau0

        # Compute equilibrium modulus factor
        G0_dim = G0(x)

        # Creep compliance power-law exponent
        growth_exp = 2.0 - x

        epsilon = 1e-12
        t_safe = jnp.maximum(t_scaled, epsilon)

        # Creep compliance with power-law growth
        # J(t) = (1 / (G0_scale * G0_dim)) * (1 + t_scaled)^(2-x)
        J_t = jnp.power(1.0 + t_safe, growth_exp) / (G0_scale * G0_dim)

        # Enforce monotonicity: J(t_i) >= J(t_{i-1})
        J_t_monotonic = jnp.maximum.accumulate(J_t)

        return J_t_monotonic

    @staticmethod
    @jax.jit
    def _predict_steady_shear_jit(
        gamma_dot: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled steady shear prediction: eta(gamma_dot).

        Computes viscosity as function of shear rate:
            eta ~ gamma_dot^(x-2) for 1 < x < 2 (shear-thinning)
            eta = const for x >= 2 (Newtonian)
            sigma_y > 0 for x < 1 (yield stress, glass phase)

        Args:
            gamma_dot: Shear rate array (1/s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Viscosity eta(gamma_dot) with shape (M,)

        Notes:
            - Shear-thinning exponent: x - 2
            - Uses relationship: eta ~ G0 * tau0 * (gamma_dot * tau0)^(x-2)
        """
        # Dimensionless shear rate
        gamma_dot_scaled = gamma_dot * tau0

        epsilon = 1e-12
        gamma_dot_safe = jnp.maximum(gamma_dot_scaled, epsilon)

        # Compute equilibrium modulus factor
        G0_dim = G0(x)

        # Viscosity power-law exponent
        visc_exp = x - 2.0

        # Viscosity formula
        # eta = G0_scale * tau0 * G0_dim * (gamma_dot * tau0)^(x-2)
        # For x = 2: eta = const (Newtonian)
        # For x < 2: eta decreases with gamma_dot (shear-thinning)

        eta = G0_scale * tau0 * G0_dim * jnp.power(gamma_dot_safe, visc_exp)

        return eta

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict based on fitted test mode.

        Routes to appropriate prediction method based on stored test_mode.

        Args:
            X: Independent variable (frequency or time)

        Returns:
            Predicted values (complex modulus or relaxation modulus)

        Raises:
            ValueError: If test_mode not set (model not fitted)
        """
        if self._test_mode is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Route to appropriate prediction method
        if self._test_mode == "oscillation":
            return self._predict_oscillation(X)
        elif self._test_mode == "relaxation":
            return self._predict_relaxation(X)
        elif self._test_mode == "creep":
            return self._predict_creep(X)
        elif self._test_mode == "steady_shear":
            return self._predict_steady_shear(X)
        elif self._test_mode == "laos":
            return self._predict_laos(X)
        elif self._test_mode == "startup":
            return self._predict_startup(X)
        else:
            raise ValueError(f"Unknown test_mode: {self._test_mode}")

    def _predict_oscillation(self, omega: np.ndarray) -> np.ndarray:
        """Predict complex modulus in oscillation mode.

        Args:
            omega: Angular frequency array (rad/s)

        Returns:
            Complex modulus [G', G''] with shape (M, 2)
        """
        # Get parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Convert to JAX arrays
        omega_jax = jnp.asarray(omega)

        # Call JIT-compiled prediction
        G_star_jax = self._predict_oscillation_jit(omega_jax, x, G0_scale, tau0)

        # Convert back to numpy
        return np.array(G_star_jax)

    def _predict_relaxation(self, t: np.ndarray) -> np.ndarray:
        """Predict relaxation modulus in relaxation mode.

        Args:
            t: Time array (s)

        Returns:
            Relaxation modulus array (Pa)
        """
        # Get parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Convert to JAX arrays
        t_jax = jnp.asarray(t)

        # Call JIT-compiled prediction
        G_t_jax = self._predict_relaxation_jit(t_jax, x, G0_scale, tau0)

        # Convert back to numpy
        return np.array(G_t_jax)

    def _predict_creep(self, t: np.ndarray) -> np.ndarray:
        """Predict creep compliance in creep mode.

        Args:
            t: Time array (s)

        Returns:
            Creep compliance array (1/Pa)
        """
        # Get parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Convert to JAX arrays
        t_jax = jnp.asarray(t)

        # Call JIT-compiled prediction
        J_t_jax = self._predict_creep_jit(t_jax, x, G0_scale, tau0)

        # Convert back to numpy
        return np.array(J_t_jax)

    def _predict_steady_shear(self, gamma_dot: np.ndarray) -> np.ndarray:
        """Predict viscosity in steady shear mode.

        Args:
            gamma_dot: Shear rate array (1/s)

        Returns:
            Viscosity array (Pa.s)
        """
        # Get parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Convert to JAX arrays
        gamma_dot_jax = jnp.asarray(gamma_dot)

        # Call JIT-compiled prediction
        eta_jax = self._predict_steady_shear_jit(gamma_dot_jax, x, G0_scale, tau0)

        # Convert back to numpy
        return np.array(eta_jax)

    def _predict_laos(self, X: np.ndarray) -> np.ndarray:
        """Predict LAOS response.

        Args:
            X: Time array or strain array

        Returns:
            Stress response array
        """
        if self._gamma_0 is None or self._omega_laos is None:
            raise ValueError(
                "LAOS prediction requires gamma_0 and omega. "
                "Set via fit() with test_mode='laos' or simulate_laos()."
            )

        strain, stress = self.simulate_laos(
            self._gamma_0, self._omega_laos, n_cycles=1, n_points_per_cycle=len(X)
        )
        return stress

    @staticmethod
    @jax.jit
    def _predict_startup_jit(
        t: jnp.ndarray, x: float, G0_scale: float, tau0: float, gamma_dot: float
    ) -> jnp.ndarray:
        """JIT-compiled startup flow prediction: eta_plus(t).

        Computes stress growth coefficient η⁺(t) = σ(t)/γ̇ = ∫₀ᵗ G(s) ds.

        For SGR's power-law relaxation G(t) ~ G₀ · (1 + t/τ₀)^(x-2):
            η⁺(t) = ∫₀ᵗ G(s) ds = G₀ · τ₀ · G₀(x) · [(1+t/τ₀)^(x-1) - 1] / (x-1)

        Special case for x=1:
            η⁺(t) = G₀ · τ₀ · G₀(x) · ln(1 + t/τ₀)

        Args:
            t: Time array (s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)
            gamma_dot: Applied shear rate (1/s)

        Returns:
            Stress growth coefficient η⁺(t) with shape (M,)
            Multiply by gamma_dot to get stress σ(t).

        Notes:
            - At short times: η⁺(t) → G₀ · t (elastic response)
            - At long times: η⁺(t) → η₀ (zero-shear viscosity)
            - For x < 1 (glass): η⁺ → ∞ as t → ∞ (no steady state)
        """
        from rheojax.utils.sgr_kernels import G0 as G0_func

        # Dimensionless time
        t_scaled = t / tau0

        # Compute equilibrium modulus factor (dimensionless)
        G0_dim = G0_func(x)

        epsilon = 1e-12
        t_safe = jnp.maximum(t_scaled, epsilon)

        # Stress growth exponent: x - 1
        exp = x - 1.0

        # Integral of G(t) from 0 to t
        # ∫ G₀·G₀(x)·(1+s/τ₀)^(x-2) ds = G₀·G₀(x)·τ₀ · [(1+t/τ₀)^(x-1) - 1]/(x-1)

        # Handle special case x ≈ 1 separately
        def x_near_one(_):
            # ln(1 + t/tau0) for x = 1
            return G0_scale * G0_dim * tau0 * jnp.log(1.0 + t_safe)

        def x_not_one(_):
            # [(1+t/tau0)^(x-1) - 1] / (x-1)
            return (
                G0_scale * G0_dim * tau0 * ((jnp.power(1.0 + t_safe, exp) - 1.0) / exp)
            )

        # Use lax.cond for JIT-compatible branching
        eta_plus = jax.lax.cond(
            jnp.abs(exp) < 1e-6,
            x_near_one,
            x_not_one,
            operand=None,
        )

        return eta_plus

    def _predict_startup(self, t: np.ndarray) -> np.ndarray:
        """Predict stress growth coefficient in startup flow mode.

        Args:
            t: Time array (s)

        Returns:
            Stress growth coefficient η⁺(t) array (Pa·s)
            To get stress: σ(t) = γ̇ · η⁺(t)
        """
        # Get parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Get stored shear rate (set during fit)
        gamma_dot = getattr(self, "_startup_gamma_dot", 1.0)

        # Convert to JAX arrays
        t_jax = jnp.asarray(t)

        # Call JIT-compiled prediction
        eta_plus_jax = self._predict_startup_jit(t_jax, x, G0_scale, tau0, gamma_dot)

        # Convert back to numpy
        return np.array(eta_plus_jax)

    def get_phase_regime(self) -> str:
        """Determine material phase regime from noise temperature x.

        The SGR model exhibits three distinct phase regimes based on x:
        - Glass phase (x < 1): Solid-like with yield stress
        - Power-law fluid (1 <= x < 2): Viscoelastic with G' ~ G'' ~ omega^(x-1)
        - Newtonian liquid (x >= 2): Viscous with constant viscosity

        Returns:
            Phase regime string: 'glass', 'power-law', or 'newtonian'

        Example:
            >>> model = SGRConventional()
            >>> model.parameters.set_value('x', 0.8)
            >>> model.get_phase_regime()
            'glass'
        """
        x = self.parameters.get_value("x")

        if x < 1.0:
            return "glass"
        elif x < 2.0:
            return "power-law"
        else:
            return "newtonian"

    def _compute_x_ss(self, gamma_dot: float, tau0: float) -> float:
        """Compute steady-state effective temperature x_ss(gamma_dot).

        The steady-state noise temperature increases with shear rate following:
            x_ss(gamma_dot) = x_eq + A * (gamma_dot * tau0)^n

        This represents the balance between aging and rejuvenation at constant
        shear rate, following Fielding et al.

        Args:
            gamma_dot: Shear rate (1/s)
            tau0: Attempt time (s)

        Returns:
            Steady-state noise temperature x_ss

        Raises:
            ValueError: If dynamic_x mode is not enabled
        """
        if not self.dynamic_x:
            raise ValueError("x_ss computation requires dynamic_x=True mode")

        x_eq = self.parameters.get_value("x_eq")
        A = self.parameters.get_value("x_ss_A")
        n = self.parameters.get_value("x_ss_n")

        # Dimensionless shear rate
        gamma_dot_dim = gamma_dot * tau0

        # Steady-state: x_ss = x_eq + A * (gamma_dot * tau0)^n
        x_ss = x_eq + A * (gamma_dot_dim**n)

        return float(x_ss)

    @staticmethod
    @jax.jit
    def _dx_dt_jit(
        x: float,
        gamma_dot: float,
        x_eq: float,
        x_ss: float,
        alpha_aging: float,
        beta_rejuv: float,
    ) -> float:
        """JIT-compiled evolution equation for dx/dt.

        The effective temperature x evolves according to:
            dx/dt = -alpha_aging * (x - x_eq) + beta_rejuv * gamma_dot * (x_ss - x)

        Terms:
        - Aging term: -alpha_aging * (x - x_eq)
          Drives x toward equilibrium x_eq at rest (gamma_dot = 0)
        - Rejuvenation term: beta_rejuv * gamma_dot * (x_ss - x)
          Shear-induced increase in x toward steady-state x_ss

        Args:
            x: Current effective temperature
            gamma_dot: Current shear rate (1/s)
            x_eq: Equilibrium temperature at rest
            x_ss: Steady-state temperature at given shear rate
            alpha_aging: Aging rate coefficient
            beta_rejuv: Rejuvenation rate coefficient

        Returns:
            Time derivative dx/dt
        """
        # Aging term (drives x -> x_eq)
        aging = -alpha_aging * (x - x_eq)

        # Rejuvenation term (shear drives x -> x_ss)
        rejuvenation = beta_rejuv * gamma_dot * (x_ss - x)

        return aging + rejuvenation

    def evolve_x(
        self,
        t: np.ndarray,
        gamma_dot: np.ndarray,
        x_initial: float,
    ) -> np.ndarray:
        """Evolve effective temperature x(t) via ODE integration.

        Integrates the evolution equation:
            dx/dt = -alpha_aging * (x - x_eq) + beta_rejuv * gamma_dot(t) * (x_ss(t) - x)

        Uses JAX ODE integration (jax.experimental.ode.odeint) for stability
        and compatibility with JAX transformations.

        Args:
            t: Time array (s)
            gamma_dot: Shear rate array (1/s), same length as t
            x_initial: Initial effective temperature

        Returns:
            x_trajectory: Effective temperature x(t) at each time point

        Raises:
            ValueError: If dynamic_x mode is not enabled or array shapes mismatch

        Example:
            >>> model = SGRConventional(dynamic_x=True)
            >>> t = np.linspace(0, 10, 100)
            >>> gamma_dot = np.ones_like(t) * 5.0  # Constant shear
            >>> x_t = model.evolve_x(t, gamma_dot, x_initial=1.0)
        """
        if not self.dynamic_x:
            raise ValueError("evolve_x requires dynamic_x=True mode")

        if t.shape != gamma_dot.shape:
            raise ValueError(
                f"Time and shear rate arrays must have same shape: "
                f"t.shape={t.shape}, gamma_dot.shape={gamma_dot.shape}"
            )

        # Get parameters
        x_eq = self.parameters.get_value("x_eq")
        alpha_aging = self.parameters.get_value("alpha_aging")
        beta_rejuv = self.parameters.get_value("beta_rejuv")
        tau0 = self.parameters.get_value("tau0")

        # Convert to JAX arrays
        t_jax = jnp.asarray(t)
        gamma_dot_jax = jnp.asarray(gamma_dot)

        # Define ODE vector field for diffrax
        # Signature: vector_field(t, y, args) -> dy/dt
        def vector_field(t_val, x_val, args):
            """ODE vector field: dx/dt = f(t, x)."""
            # Interpolate gamma_dot at current time t_val
            # Use linear interpolation
            gamma_dot_current = jnp.interp(t_val, t_jax, gamma_dot_jax)

            # Compute x_ss at current shear rate
            gamma_dot_dim = gamma_dot_current * tau0
            x_ss_current = x_eq + self.parameters.get_value("x_ss_A") * (
                gamma_dot_dim ** self.parameters.get_value("x_ss_n")
            )

            # Compute dx/dt
            # Ensure x_val is scalar extract if needed, though diffrax passes arrays
            return self._dx_dt_jit(
                x_val, gamma_dot_current, x_eq, x_ss_current, alpha_aging, beta_rejuv
            )

        # Solve ODE using Diffrax
        # Use Tsit5 (Runge-Kutta 5(4)) which is generally efficient for non-stiff problems
        # Use PIDController for adaptive step size (similar to odeint)
        term = diffrax.ODETerm(vector_field)
        solver = diffrax.Tsit5()
        t0 = t_jax[0]
        t1 = t_jax[-1]
        dt0 = (t1 - t0) / len(t_jax) if len(t_jax) > 1 else 0.001

        # Save solution at specified time points
        saveat = diffrax.SaveAt(ts=t_jax)

        # Step size controller
        stepsize_controller = diffrax.PIDController(
            rtol=1.4e-8, atol=1.4e-8
        )  # Match standard precision

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0=x_initial,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=100000,  # Safety limit
        )

        # Extract trajectory
        # sol.ys has shape (num_save_points, num_states) -> (N, 1) or (N,) depending on y0
        x_trajectory_jax = sol.ys

        # Ensure correct shape (N,)
        if x_trajectory_jax.ndim > 1:
            x_trajectory_jax = x_trajectory_jax.squeeze()

        # Convert back to numpy
        x_trajectory = np.array(x_trajectory_jax)

        # Store trajectory for analysis
        self._x_trajectory = x_trajectory
        self._t_trajectory = t

        return x_trajectory

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference with NumPyro NUTS.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes SGR predictions given input X and a parameter array.

        Args:
            X: Independent variable (frequency for oscillation, time for relaxation)
            params: Array of parameter values [x, G0, tau0]
                Length: 3
            test_mode: Optional test mode override. If None, uses stored self._test_mode

        Returns:
            Model predictions as JAX array
            - For oscillation: Complex modulus [G', G''] with shape (M, 2)
            - For relaxation: Relaxation modulus with shape (M,)
            - For creep: Creep compliance with shape (M,)
            - For steady_shear: Viscosity with shape (M,)
            - For startup: Stress growth coefficient with shape (M,)

        Note:
            Uses stored test_mode from last fit() call for mode-aware inference.
            This ensures correct kernel functions are used during NUTS sampling.

        Example:
            >>> # During Bayesian inference, NumPyro calls:
            >>> predictions = model.model_function(omega, params_sample)
            >>> # Where params_sample = [x_sample, G0_sample, tau0_sample]
        """
        # Extract parameters from array
        x = params[0]
        G0_scale = params[1]
        tau0 = params[2]

        # Use stored test mode from last fit (critical for mode-aware inference)
        mode = test_mode if test_mode is not None else self._test_mode

        if mode is None:
            mode = "oscillation"  # Default fallback

        # Convert X to JAX array
        X_jax = jnp.asarray(X)

        # Route to appropriate prediction method based on test mode
        if mode == "oscillation":
            return self._predict_oscillation_jit(X_jax, x, G0_scale, tau0)
        elif mode == "relaxation":
            return self._predict_relaxation_jit(X_jax, x, G0_scale, tau0)
        elif mode == "creep":
            return self._predict_creep_jit(X_jax, x, G0_scale, tau0)
        elif mode == "steady_shear":
            return self._predict_steady_shear_jit(X_jax, x, G0_scale, tau0)
        elif mode == "laos":
            # For LAOS Bayesian inference, use oscillation response
            return self._predict_oscillation_jit(X_jax, x, G0_scale, tau0)
        elif mode == "startup":
            gamma_dot = getattr(self, "_startup_gamma_dot", 1.0)
            return self._predict_startup_jit(X_jax, x, G0_scale, tau0, gamma_dot)
        else:
            raise ValueError(f"Unsupported test mode: {mode}")

    # =========================================================================
    # LAOS (Large Amplitude Oscillatory Shear) Methods - Task Group 6
    # =========================================================================

    def simulate_laos(
        self,
        gamma_0: float,
        omega: float,
        n_cycles: int = 2,
        n_points_per_cycle: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate LAOS response for given strain amplitude and frequency.

        Generates time-domain stress response to sinusoidal strain input:
            gamma(t) = gamma_0 * sin(omega * t)

        For SGR model, the stress response is computed using the complex modulus
        in the linear viscoelastic approximation, with nonlinearity arising from
        strain-dependent softening at large amplitudes.

        Args:
            gamma_0: Strain amplitude (dimensionless)
            omega: Angular frequency (rad/s)
            n_cycles: Number of oscillation cycles to simulate
            n_points_per_cycle: Number of time points per cycle

        Returns:
            strain: Strain array gamma(t)
            stress: Stress array sigma(t)

        Example:
            >>> model = SGRConventional()
            >>> model.parameters.set_value("x", 1.5)
            >>> strain, stress = model.simulate_laos(gamma_0=0.1, omega=1.0)
        """
        # Store LAOS parameters
        self._gamma_0 = gamma_0
        self._omega_laos = omega

        # Get model parameters
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        # Time array
        period = 2.0 * np.pi / omega
        t_max = n_cycles * period
        n_points = n_cycles * n_points_per_cycle
        t = np.linspace(0, t_max, n_points, endpoint=False)

        # Strain: gamma(t) = gamma_0 * sin(omega * t)
        strain = gamma_0 * np.sin(omega * t)

        # Strain rate: gamma_dot(t) = gamma_0 * omega * cos(omega * t)
        strain_rate = gamma_0 * omega * np.cos(omega * t)

        # Get complex modulus at this frequency
        omega_arr = np.array([omega])
        G_star = self._predict_oscillation_jit(
            jnp.asarray(omega_arr), x, G0_scale, tau0
        )
        G_prime = float(G_star[0, 0])
        G_double_prime = float(G_star[0, 1])

        # In linear viscoelastic regime:
        # sigma(t) = G' * gamma(t) + (G'' / omega) * gamma_dot(t)
        # sigma(t) = G' * gamma_0 * sin(omega*t) + G'' * gamma_0 * cos(omega*t)
        stress = G_prime * strain + (G_double_prime / omega) * strain_rate

        # Add weak nonlinearity based on SGR physics for large amplitudes
        # Large strains can cause local yielding events in SGR
        # Approximate this by adding strain-softening at large |gamma|
        if gamma_0 > 0.1:  # Only for larger amplitudes
            # Strain-softening factor: reduces stress at high strains
            softening = 1.0 - 0.1 * (np.abs(strain) / gamma_0) ** 2
            stress = stress * softening

        return strain, stress

    def extract_laos_harmonics(
        self,
        stress: np.ndarray,
        n_points_per_cycle: int = 256,
    ) -> dict:
        """Extract Fourier harmonics from LAOS stress response.

        Performs FFT analysis to extract harmonic amplitudes and phases:
            sigma(t) = sum_n I_n * sin(n*omega*t + phi_n)

        For LAOS, odd harmonics (n = 1, 3, 5, ...) dominate due to symmetry.

        Args:
            stress: Stress time series from simulate_laos()
            n_points_per_cycle: Points per oscillation cycle

        Returns:
            Dictionary containing:
            - I_1, I_3, I_5, ...: Harmonic amplitudes
            - phi_1, phi_3, phi_5, ...: Phase angles
            - I_3_I_1, I_5_I_1, ...: Relative intensities

        Example:
            >>> strain, stress = model.simulate_laos(gamma_0=0.5, omega=1.0)
            >>> harmonics = model.extract_laos_harmonics(stress)
            >>> print(f"Third harmonic ratio: {harmonics['I_3_I_1']:.4f}")
        """
        # Use last complete cycle for steady-state analysis
        stress_cycle = stress[-n_points_per_cycle:]

        # FFT of stress signal
        stress_fft = np.fft.fft(stress_cycle)
        n = len(stress_cycle)

        # Frequency indices for harmonics
        # Fundamental is at index 1 (one complete cycle in the window)
        fundamental_idx = 1

        # Extract harmonic amplitudes (magnitude) and phases
        harmonics = {}

        # Fundamental (n=1)
        I_1 = 2.0 * np.abs(stress_fft[fundamental_idx]) / n
        phi_1 = np.angle(stress_fft[fundamental_idx])
        harmonics["I_1"] = I_1
        harmonics["phi_1"] = phi_1

        # Third harmonic (n=3)
        idx_3 = 3 * fundamental_idx
        if idx_3 < n // 2:
            I_3 = 2.0 * np.abs(stress_fft[idx_3]) / n
            phi_3 = np.angle(stress_fft[idx_3])
        else:
            I_3 = 0.0
            phi_3 = 0.0
        harmonics["I_3"] = I_3
        harmonics["phi_3"] = phi_3

        # Fifth harmonic (n=5)
        idx_5 = 5 * fundamental_idx
        if idx_5 < n // 2:
            I_5 = 2.0 * np.abs(stress_fft[idx_5]) / n
            phi_5 = np.angle(stress_fft[idx_5])
        else:
            I_5 = 0.0
            phi_5 = 0.0
        harmonics["I_5"] = I_5
        harmonics["phi_5"] = phi_5

        # Seventh harmonic (n=7)
        idx_7 = 7 * fundamental_idx
        if idx_7 < n // 2:
            I_7 = 2.0 * np.abs(stress_fft[idx_7]) / n
            phi_7 = np.angle(stress_fft[idx_7])
        else:
            I_7 = 0.0
            phi_7 = 0.0
        harmonics["I_7"] = I_7
        harmonics["phi_7"] = phi_7

        # Relative intensities
        if I_1 > 0:
            harmonics["I_3_I_1"] = I_3 / I_1
            harmonics["I_5_I_1"] = I_5 / I_1
            harmonics["I_7_I_1"] = I_7 / I_1
        else:
            harmonics["I_3_I_1"] = 0.0
            harmonics["I_5_I_1"] = 0.0
            harmonics["I_7_I_1"] = 0.0

        return harmonics

    def compute_chebyshev_coefficients(
        self,
        strain: np.ndarray,
        stress: np.ndarray,
        gamma_0: float,
        omega: float,
        n_points_per_cycle: int = 256,
    ) -> dict:
        """Compute Chebyshev decomposition of LAOS response.

        Decomposes stress into elastic and viscous Chebyshev contributions:
            sigma(gamma, gamma_dot) = sum_n e_n * T_n(gamma/gamma_0)
                                    + sum_n v_n * T_n(gamma_dot/gamma_dot_0)

        where T_n are Chebyshev polynomials of the first kind.

        Physical interpretation:
        - e_n: Elastic (in-phase with strain) Chebyshev coefficients
        - v_n: Viscous (out-of-phase with strain) Chebyshev coefficients
        - e_3/e_1 > 0: Strain stiffening
        - e_3/e_1 < 0: Strain softening
        - v_3/v_1 > 0: Shear thickening
        - v_3/v_1 < 0: Shear thinning

        Args:
            strain: Strain array from simulate_laos()
            stress: Stress array from simulate_laos()
            gamma_0: Strain amplitude
            omega: Angular frequency
            n_points_per_cycle: Points per oscillation cycle

        Returns:
            Dictionary containing:
            - e_1, e_3, e_5: Elastic Chebyshev coefficients
            - v_1, v_3, v_5: Viscous Chebyshev coefficients
            - e_3_e_1, v_3_v_1: Normalized coefficients

        Example:
            >>> strain, stress = model.simulate_laos(gamma_0=0.5, omega=1.0)
            >>> chebyshev = model.compute_chebyshev_coefficients(
            ...     strain, stress, gamma_0=0.5, omega=1.0
            ... )
            >>> print(f"Strain stiffening ratio: {chebyshev['e_3_e_1']:.4f}")
        """
        # Use last complete cycle
        strain_cycle = strain[-n_points_per_cycle:]
        stress_cycle = stress[-n_points_per_cycle:]

        # Normalize strain to [-1, 1] for Chebyshev basis
        gamma_norm = strain_cycle / gamma_0

        # Compute strain rate
        dt = 2.0 * np.pi / (omega * n_points_per_cycle)
        gamma_dot = np.gradient(strain_cycle, dt)
        gamma_dot_0 = gamma_0 * omega
        gamma_dot_norm = gamma_dot / gamma_dot_0

        # Chebyshev polynomials T_n(x)
        # T_0(x) = 1
        # T_1(x) = x
        # T_2(x) = 2x^2 - 1
        # T_3(x) = 4x^3 - 3x
        # T_5(x) = 16x^5 - 20x^3 + 5x

        def T_1(x):
            return x

        def T_3(x):
            return 4 * x**3 - 3 * x

        def T_5(x):
            return 16 * x**5 - 20 * x**3 + 5 * x

        # Decompose stress into elastic (in-phase) and viscous (out-of-phase) parts
        # Using orthogonality of Chebyshev polynomials

        # Elastic part: stress component in-phase with strain
        # sigma_elastic proportional to strain
        # Viscous part: stress component in-phase with strain rate
        # sigma_viscous proportional to strain rate (90 degrees out of phase)

        # Project stress onto Chebyshev basis
        # Use numerical integration (trapezoidal rule)

        # Weight function for Chebyshev: w(x) = 1/sqrt(1-x^2)
        # But for LAOS, we use uniform weighting over the cycle

        # Elastic coefficients (project onto strain-dependent basis)
        e_1 = 2.0 * np.mean(stress_cycle * T_1(gamma_norm))
        e_3 = 2.0 * np.mean(stress_cycle * T_3(gamma_norm))
        e_5 = 2.0 * np.mean(stress_cycle * T_5(gamma_norm))

        # Viscous coefficients (project onto strain-rate-dependent basis)
        v_1 = 2.0 * np.mean(stress_cycle * T_1(gamma_dot_norm))
        v_3 = 2.0 * np.mean(stress_cycle * T_3(gamma_dot_norm))
        v_5 = 2.0 * np.mean(stress_cycle * T_5(gamma_dot_norm))

        # Build result dictionary
        chebyshev = {
            "e_1": e_1,
            "e_3": e_3,
            "e_5": e_5,
            "v_1": v_1,
            "v_3": v_3,
            "v_5": v_5,
        }

        # Normalized coefficients (standard LAOS metrics)
        if abs(e_1) > 1e-12:
            chebyshev["e_3_e_1"] = e_3 / e_1
            chebyshev["e_5_e_1"] = e_5 / e_1
        else:
            chebyshev["e_3_e_1"] = 0.0
            chebyshev["e_5_e_1"] = 0.0

        if abs(v_1) > 1e-12:
            chebyshev["v_3_v_1"] = v_3 / v_1
            chebyshev["v_5_v_1"] = v_5 / v_1
        else:
            chebyshev["v_3_v_1"] = 0.0
            chebyshev["v_5_v_1"] = 0.0

        return chebyshev

    def get_lissajous_curve(
        self,
        gamma_0: float,
        omega: float,
        n_points: int = 256,
        normalized: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Lissajous curve (stress vs strain) for LAOS.

        Args:
            gamma_0: Strain amplitude
            omega: Angular frequency (rad/s)
            n_points: Number of points in curve
            normalized: If True, normalize strain and stress

        Returns:
            strain: Strain array (one period)
            stress: Stress array (one period)

        Example:
            >>> strain, stress = model.get_lissajous_curve(gamma_0=0.1, omega=1.0)
            >>> plt.plot(strain, stress)  # Elastic Lissajous
        """
        # Simulate two cycles
        strain, stress = self.simulate_laos(
            gamma_0, omega, n_cycles=2, n_points_per_cycle=n_points
        )

        # Use last cycle for steady-state
        strain_cycle = strain[-n_points:]
        stress_cycle = stress[-n_points:]

        if normalized:
            strain_cycle = strain_cycle / gamma_0
            stress_max = np.max(np.abs(stress_cycle))
            if stress_max > 0:
                stress_cycle = stress_cycle / stress_max

        return strain_cycle, stress_cycle

    # =========================================================================
    # Thixotropy Extension Methods
    # =========================================================================

    def enable_thixotropy(
        self,
        k_build: float = 0.1,
        k_break: float = 0.5,
        n_struct: float = 2.0,
    ) -> None:
        """Enable thixotropy modeling with structural parameter lambda(t).

        Adds thixotropy kinetics parameters to the model. The structural
        parameter lambda represents the state of internal microstructure:
        - lambda = 1: Fully built structure
        - lambda = 0: Fully broken structure

        Evolution equation:
            d(lambda)/dt = k_build * (1 - lambda) - k_break * gamma_dot * lambda

        The effective modulus is coupled to lambda:
            G_eff(t) = G0 * lambda(t)^n_struct

        Args:
            k_build: Structure build-up rate (1/s), default 0.1
            k_break: Structure breakdown rate (dimensionless), default 0.5
            n_struct: Structural coupling exponent, default 2.0

        Example:
            >>> model = SGRConventional()
            >>> model.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)
            >>> # Now model can predict stress transients with thixotropy
        """
        # Add thixotropy parameters if not already present
        if "k_build" not in self.parameters.keys():
            self.parameters.add(
                name="k_build",
                value=k_build,
                bounds=(0.0, 10.0),
                units="1/s",
                description="Structure build-up rate (1/s)",
            )

        if "k_break" not in self.parameters.keys():
            self.parameters.add(
                name="k_break",
                value=k_break,
                bounds=(0.0, 10.0),
                units="dimensionless",
                description="Structure breakdown rate (shear-dependent)",
            )

        if "n_struct" not in self.parameters.keys():
            self.parameters.add(
                name="n_struct",
                value=n_struct,
                bounds=(0.1, 5.0),
                units="dimensionless",
                description="Structural coupling exponent",
            )

        # Flag for thixotropy mode
        self._thixotropy_enabled = True

        # Storage for lambda trajectory
        self._lambda_trajectory: np.ndarray | None = None

    def evolve_lambda(
        self,
        t: np.ndarray,
        gamma_dot: np.ndarray,
        lambda_initial: float = 1.0,
    ) -> np.ndarray:
        """Evolve structural parameter lambda(t) for given shear history.

        Integrates the thixotropy kinetics equation:
            d(lambda)/dt = k_build * (1 - lambda) - k_break * gamma_dot * lambda

        Args:
            t: Time array (s)
            gamma_dot: Shear rate array (1/s), same shape as t
            lambda_initial: Initial structural parameter [0, 1], default 1.0

        Returns:
            lambda_t: Structural parameter evolution, same shape as t

        Raises:
            ValueError: If thixotropy not enabled or array shapes mismatch

        Example:
            >>> model = SGRConventional()
            >>> model.enable_thixotropy()
            >>> t = np.linspace(0, 10, 100)
            >>> gamma_dot = np.ones_like(t) * 10.0  # Constant shear
            >>> lambda_t = model.evolve_lambda(t, gamma_dot, lambda_initial=1.0)
        """
        if not getattr(self, "_thixotropy_enabled", False):
            raise ValueError("Thixotropy not enabled. Call enable_thixotropy() first.")

        if t.shape != gamma_dot.shape:
            raise ValueError(
                f"Time and shear rate arrays must have same shape: "
                f"t.shape={t.shape}, gamma_dot.shape={gamma_dot.shape}"
            )

        # Get thixotropy parameters
        k_build = self.parameters.get_value("k_build")
        k_break = self.parameters.get_value("k_break")

        # Integrate using Euler method
        dt = np.diff(t)
        dt = np.concatenate([[0], dt])

        lambda_t = np.zeros_like(t)
        lambda_t[0] = lambda_initial

        for i in range(1, len(t)):
            dlambda_dt = (
                k_build * (1.0 - lambda_t[i - 1])
                - k_break * gamma_dot[i] * lambda_t[i - 1]
            )
            lambda_t[i] = lambda_t[i - 1] + dlambda_dt * dt[i]
            # Clamp to [0, 1]
            lambda_t[i] = np.clip(lambda_t[i], 0.0, 1.0)

        # Store trajectory
        self._lambda_trajectory = lambda_t

        return lambda_t

    def predict_thixotropic_stress(
        self,
        t: np.ndarray,
        gamma_dot: np.ndarray,
        lambda_t: np.ndarray | None = None,
        lambda_initial: float = 1.0,
    ) -> np.ndarray:
        """Predict stress response with thixotropic modulus.

        The effective modulus is coupled to the structural parameter:
            G_eff(t) = G0 * lambda(t)^n_struct

        Args:
            t: Time array (s)
            gamma_dot: Shear rate array (1/s)
            lambda_t: Pre-computed lambda trajectory, or None to compute
            lambda_initial: Initial lambda if computing [0, 1], default 1.0

        Returns:
            sigma: Stress response (Pa)

        Example:
            >>> model = SGRConventional()
            >>> model.enable_thixotropy()
            >>> t = np.linspace(0, 10, 100)
            >>> gamma_dot = np.ones_like(t) * 10.0
            >>> sigma = model.predict_thixotropic_stress(t, gamma_dot)
        """
        if not getattr(self, "_thixotropy_enabled", False):
            raise ValueError("Thixotropy not enabled. Call enable_thixotropy() first.")

        # Compute lambda trajectory if not provided
        if lambda_t is None:
            lambda_t = self.evolve_lambda(t, gamma_dot, lambda_initial)

        # Get parameters
        G0_val = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")
        x = self.parameters.get_value("x")
        n_struct = self.parameters.get_value("n_struct")

        # Effective modulus from structure
        G_eff = G0_val * np.power(lambda_t, n_struct)

        # Viscosity from power-law (SGR-like)
        gamma_dot_safe = np.maximum(np.abs(gamma_dot), 1e-12)
        eta_factor = np.power(gamma_dot_safe * tau0, x - 2.0)

        # Stress = G_eff * gamma_dot * tau0 * eta_factor
        sigma = G_eff * gamma_dot * tau0 * eta_factor

        return sigma

    def predict_stress_transient(
        self,
        t: np.ndarray,
        gamma_dot: np.ndarray,
        lambda_initial: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict stress transient (overshoot/undershoot) for shear step protocol.

        For step-up in shear rate: Initially high stress (intact structure)
        followed by decay as structure breaks down (overshoot).

        For step-down in shear rate: Initially low stress (broken structure)
        followed by increase as structure rebuilds (undershoot).

        Args:
            t: Time array (s)
            gamma_dot: Shear rate array (1/s), can include steps
            lambda_initial: Initial structural parameter [0, 1]

        Returns:
            sigma: Stress response (Pa)
            lambda_t: Structural parameter evolution

        Example:
            >>> model = SGRConventional()
            >>> model.enable_thixotropy()
            >>> t = np.linspace(0, 10, 100)
            >>> gamma_dot = np.ones_like(t)
            >>> gamma_dot[t >= 5] = 10.0  # Step up at t=5
            >>> sigma, lambda_t = model.predict_stress_transient(t, gamma_dot)
        """
        # Evolve lambda
        lambda_t = self.evolve_lambda(t, gamma_dot, lambda_initial)

        # Compute stress
        sigma = self.predict_thixotropic_stress(t, gamma_dot, lambda_t)

        return sigma, lambda_t

    # =========================================================================
    # Shear Banding Detection Methods
    # =========================================================================

    def detect_shear_banding(
        self,
        gamma_dot: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        n_points: int = 100,
        gamma_dot_range: tuple[float, float] = (1e-2, 1e2),
    ) -> tuple[bool, dict | None]:
        """Detect shear banding from constitutive curve.

        Computes the steady-state flow curve and checks for non-monotonicity
        (d sigma / d gamma_dot < 0) which indicates shear banding instability.

        Args:
            gamma_dot: Shear rate array (1/s). If None, uses gamma_dot_range.
            sigma: Stress array (Pa). If None, computes from model.
            n_points: Number of points if computing flow curve
            gamma_dot_range: Range for computing flow curve if gamma_dot is None

        Returns:
            is_banding: True if shear banding detected
            banding_info: Dict with banding region info, or None

        Example:
            >>> model = SGRConventional()
            >>> model.parameters.set_value("x", 0.8)  # Glass regime
            >>> is_banding, info = model.detect_shear_banding()
        """
        # Import detection function
        from rheojax.transforms.srfs import detect_shear_banding as _detect_banding

        # Compute flow curve if not provided
        if gamma_dot is None:
            gamma_dot = np.logspace(
                np.log10(gamma_dot_range[0]),
                np.log10(gamma_dot_range[1]),
                n_points,
            )

        if sigma is None:
            # Compute viscosity from model
            self._test_mode = "steady_shear"
            eta = self.predict(gamma_dot)
            sigma = eta * gamma_dot

        # Detect shear banding
        is_banding, banding_info = _detect_banding(gamma_dot, sigma, warn=True)

        return is_banding, banding_info

    def predict_banded_flow(
        self,
        gamma_dot_applied: float,
        gamma_dot: np.ndarray | None = None,
        sigma: np.ndarray | None = None,
        n_points: int = 100,
    ) -> dict | None:
        """Predict flow in shear banding regime with lever rule.

        When shear banding occurs, the material splits into bands with
        different local shear rates. This method computes the band
        fractions and the composite stress.

        Args:
            gamma_dot_applied: Applied average shear rate (1/s)
            gamma_dot: Shear rate array for flow curve. If None, computed.
            sigma: Stress array for flow curve. If None, computed.
            n_points: Number of points if computing flow curve

        Returns:
            coexistence: Dict with band coexistence info, or None

        Example:
            >>> model = SGRConventional()
            >>> model.parameters.set_value("x", 0.8)
            >>> coex = model.predict_banded_flow(gamma_dot_applied=1.0)
            >>> if coex:
            ...     print(f"Low band: {coex['fraction_low']:.2%}")
            ...     print(f"High band: {coex['fraction_high']:.2%}")
        """
        from rheojax.transforms.srfs import compute_shear_band_coexistence

        # Compute flow curve if not provided
        if gamma_dot is None:
            gamma_dot = np.logspace(-2, 3, n_points)

        if sigma is None:
            self._test_mode = "steady_shear"
            eta = self.predict(gamma_dot)
            sigma = eta * gamma_dot

        # Compute coexistence
        coexistence = compute_shear_band_coexistence(
            gamma_dot, sigma, gamma_dot_applied
        )

        return coexistence
