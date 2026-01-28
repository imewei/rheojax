"""Single-mode Transient Network Theory (TNT) model.

This module implements `TNTSingleMode`, a composable constitutive model for
associative polymers and physical gels with reversible crosslinks.

Composability
-------------
The single-mode TNT model supports multiple variant combinations via
constructor parameters:

- ``breakage``: Breakage rate function ("constant", "bell", "power_law",
  "stretch_creation")
- ``stress_type``: Stress formula ("linear", "fene")
- ``xi``: Slip parameter (0 = upper-convected, >0 = Gordon-Schowalter)

These can be combined freely, e.g.::

    TNTSingleMode(breakage="bell", stress_type="fene", xi=0.3)

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear (analytical for constant, numerical otherwise)
- OSCILLATION: Small-amplitude oscillatory shear (analytical, Maxwell-like)
- STARTUP: Transient stress growth at constant rate (ODE)
- RELAXATION: Stress decay after cessation (analytical or ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE)

Example
-------
>>> from rheojax.models.tnt import TNTSingleMode
>>> import numpy as np
>>>
>>> # Basic (Tanaka-Edwards) model
>>> model = TNTSingleMode()
>>>
>>> # Flow curve (analytical)
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # Bell force-dependent variant
>>> model_bell = TNTSingleMode(breakage="bell")
>>>
>>> # FENE + non-affine composition
>>> model_full = TNTSingleMode(stress_type="fene", xi=0.2)

References
----------
- Green, M.S. & Tobolsky, A.V. (1946). J. Chem. Phys. 14, 80-92.
- Tanaka, F. & Edwards, S.F. (1992). Macromolecules 25, 1516-1523.
- Bell, G.I. (1978). Science 200, 618-627.
"""

from __future__ import annotations

import logging
from typing import Literal

import diffrax
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.models.tnt._base import TNTBase
from rheojax.models.tnt._kernels import (
    build_tnt_creep_ode_rhs,
    build_tnt_laos_ode_rhs,
    build_tnt_ode_rhs,
    build_tnt_relaxation_ode_rhs,
    tnt_base_relaxation_vec,
    tnt_base_steady_conformation,
    tnt_base_steady_n1_vec,
    tnt_base_steady_stress_vec,
    tnt_saos_moduli_vec,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)

# Breakage type alias
BreakageType = Literal["constant", "bell", "power_law", "stretch_creation"]
StressType = Literal["linear", "fene"]


@ModelRegistry.register(
    "tnt_single_mode",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
)
@ModelRegistry.register(
    "tnt",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
)
class TNTSingleMode(TNTBase):
    """Single-mode Transient Network Theory model.

    Implements the Green-Tobolsky / Tanaka-Edwards transient network model
    with composable physics variants. The conformation tensor S tracks the
    average chain configuration between reversible crosslinks.

    The constitutive equation is::

        dS/dt = L·S + S·L^T + g₀·I - β(S)·S

    Stress is computed from S via σ = G·f(S)·(S - I) + η_s·γ̇.

    Parameters
    ----------
    breakage : str, default "constant"
        Breakage rate function:
        - "constant": β = 1/τ_b (Tanaka-Edwards, UCM-like)
        - "bell": β = (1/τ_b)·exp(ν·(stretch-1)) (force-dependent)
        - "power_law": β = (1/τ_b)·stretch^m
        - "stretch_creation": β = (1/τ_b), g₀ = (1+κ·stretch)/τ_b
    stress_type : str, default "linear"
        Stress formula:
        - "linear": σ = G·(S - I) (Gaussian chains)
        - "fene": σ = G·f(tr(S))·(S - I) (finitely extensible)
    xi : float, default 0.0
        Gordon-Schowalter slip parameter (0=upper-convected, 1=corotational)

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters
    fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    Basic Tanaka-Edwards model:

    >>> model = TNTSingleMode()
    >>> gamma_dot = np.logspace(-2, 2, 50)
    >>> sigma = model.predict(gamma_dot, test_mode='flow_curve')

    Bell force-dependent breakage:

    >>> model = TNTSingleMode(breakage="bell")
    >>> # Now has additional parameter 'nu' (force sensitivity)

    See Also
    --------
    TNTLoopBridge : Two-species loop-bridge kinetics
    TNTCates : Living polymer (wormlike micelle) model
    """

    def __init__(
        self,
        breakage: BreakageType = "constant",
        stress_type: StressType = "linear",
        xi: float = 0.0,
    ):
        """Initialize single-mode TNT model.

        Parameters
        ----------
        breakage : str, default "constant"
            Breakage rate function type
        stress_type : str, default "linear"
            Stress formula type
        xi : float, default 0.0
            Slip parameter for Gordon-Schowalter derivative
        """
        # Store variant flags before calling super().__init__
        self._breakage = breakage
        self._stress_type = stress_type
        self._xi = xi

        super().__init__()
        self._setup_parameters()
        self._build_variant_ode_functions()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with TNT parameters.

        Core parameters (always present):
        - G: Network modulus (Pa)
        - tau_b: Bond lifetime (s)
        - eta_s: Solvent viscosity (Pa·s)

        Conditional parameters (variant-dependent):
        - nu: Force sensitivity (Bell breakage)
        - m_break: Breakage exponent (power_law breakage)
        - kappa: Creation enhancement (stretch_creation breakage)
        - L_max: Maximum extensibility (FENE stress)
        """
        self.parameters = ParameterSet()

        # Core parameters
        self.parameters.add(
            name="G",
            value=1e3,
            bounds=(1e0, 1e8),
            units="Pa",
            description="Network modulus (elastic contribution from active chains)",
        )
        self.parameters.add(
            name="tau_b",
            value=1.0,
            bounds=(1e-6, 1e4),
            units="s",
            description="Bond lifetime (mean time between detachment events)",
        )
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian background contribution)",
        )

        # Conditional: Bell breakage
        if self._breakage == "bell":
            self.parameters.add(
                name="nu",
                value=1.0,
                bounds=(0.01, 20.0),
                units="dimensionless",
                description="Force sensitivity (Bell model, higher = more shear-thinning)",
            )

        # Conditional: Power-law breakage
        if self._breakage == "power_law":
            self.parameters.add(
                name="m_break",
                value=2.0,
                bounds=(0.5, 10.0),
                units="dimensionless",
                description="Breakage power-law exponent",
            )

        # Conditional: Stretch-creation breakage
        if self._breakage == "stretch_creation":
            self.parameters.add(
                name="kappa",
                value=0.5,
                bounds=(0.0, 5.0),
                units="dimensionless",
                description="Creation rate enhancement from chain stretch",
            )

        # Conditional: FENE stress
        if self._stress_type == "fene":
            self.parameters.add(
                name="L_max",
                value=10.0,
                bounds=(2.0, 100.0),
                units="dimensionless",
                description="Maximum chain extensibility (FENE-P spring)",
            )

    # =========================================================================
    # Property Accessors
    # =========================================================================

    @property
    def G(self) -> float:
        """Get network modulus G (Pa)."""
        val = self.parameters.get_value("G")
        return float(val) if val is not None else 0.0

    @property
    def tau_b(self) -> float:
        """Get bond lifetime τ_b (s)."""
        val = self.parameters.get_value("tau_b")
        return float(val) if val is not None else 0.0

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        val = self.parameters.get_value("eta_s")
        return float(val) if val is not None else 0.0

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = G·τ_b + η_s (Pa·s)."""
        return self.G * self.tau_b + self.eta_s

    @property
    def breakage(self) -> str:
        """Get breakage type."""
        return self._breakage

    @property
    def stress_type(self) -> str:
        """Get stress type."""
        return self._stress_type

    @property
    def xi(self) -> float:
        """Get slip parameter ξ."""
        return self._xi

    @property
    def _is_basic(self) -> bool:
        """Whether this is the basic (constant/linear/UC) variant."""
        return (
            self._breakage == "constant"
            and self._stress_type == "linear"
            and self._xi == 0.0
        )

    # =========================================================================
    # Variant ODE Infrastructure
    # =========================================================================

    def _build_variant_ode_functions(self):
        """Build and cache variant-specific ODE RHS functions.

        Called once in __init__. Each variant combination traces to a
        separate JAX-compiled function.
        """
        use_fene = self._stress_type == "fene"
        use_gs = self._xi > 0

        self._variant_ode = build_tnt_ode_rhs(
            self._breakage, use_fene, use_gs
        )
        self._variant_creep_ode = build_tnt_creep_ode_rhs(
            self._breakage, use_fene, use_gs
        )
        self._variant_laos_ode = build_tnt_laos_ode_rhs(
            self._breakage, use_fene, use_gs
        )
        self._variant_relax_ode = build_tnt_relaxation_ode_rhs(
            self._breakage, use_fene, use_gs
        )

    def _get_variant_args(self) -> dict:
        """Build args dict with variant parameters from self.parameters.

        Used by public simulation methods that read parameter values
        from the fitted ParameterSet.
        """
        args = {
            "nu": 0.0,
            "m_break": 0.0,
            "kappa": 0.0,
            "L_max": 10.0,
            "xi": self._xi,
        }
        if self._breakage == "bell":
            val = self.parameters.get_value("nu")
            args["nu"] = float(val) if val is not None else 0.0
        elif self._breakage == "power_law":
            val = self.parameters.get_value("m_break")
            args["m_break"] = float(val) if val is not None else 0.0
        elif self._breakage == "stretch_creation":
            val = self.parameters.get_value("kappa")
            args["kappa"] = float(val) if val is not None else 0.0
        if self._stress_type == "fene":
            val = self.parameters.get_value("L_max")
            args["L_max"] = float(val) if val is not None else 10.0
        return args

    def _unpack_variant_params(self, params) -> dict:
        """Unpack variant parameters from a JAX params array.

        Used by model_function where G, tau_b, eta_s and variant params
        come from the traced parameter array (for NLSQ/NUTS).

        Returns dict with all variant param values (dummy values for
        inactive variants).
        """
        result = {
            "nu": 0.0,
            "m_break": 0.0,
            "kappa": 0.0,
            "L_max": 10.0,
            "xi": self._xi,
        }
        idx = 3  # After G, tau_b, eta_s
        if self._breakage == "bell":
            result["nu"] = params[idx]
            idx += 1
        elif self._breakage == "power_law":
            result["m_break"] = params[idx]
            idx += 1
        elif self._breakage == "stretch_creation":
            result["kappa"] = params[idx]
            idx += 1
        if self._stress_type == "fene":
            result["L_max"] = params[idx]
            idx += 1
        return result

    def _get_ode_solver(self):
        """Return appropriate ODE solver for this variant.

        Uses Dopri5 (5th-order Dormand-Prince) for non-basic variants
        which may be mildly stiff, Tsit5 for basic (constant breakage).

        Note: Implicit solvers (Kvaerno5) are incompatible with current
        diffrax/lineax versions due to TracerBoolConversionError in LU.
        """
        if self._is_basic:
            return diffrax.Tsit5()
        return diffrax.Dopri5()

    def _compute_stress_from_conformation(
        self, S_xx, S_yy, S_zz, S_xy, G, eta_s, gamma_dot, vp
    ):
        """Compute total shear stress from conformation tensor components.

        Handles both linear and FENE-P stress types.
        """
        if self._stress_type == "fene":
            tr_S = S_xx + S_yy + S_zz
            L2 = vp["L_max"] * vp["L_max"]
            f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
            sigma_el = G * f * S_xy
        else:
            sigma_el = G * S_xy
        return sigma_el + eta_s * gamma_dot

    # =========================================================================
    # Core Interface Methods
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization.

        Parameters
        ----------
        x : array-like
            Independent variable (shear rate, frequency, or time)
        y : array-like
            Dependent variable (stress, modulus, or strain)
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        self
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        self._test_mode = test_mode

        x_jax = jnp.asarray(x, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # Store protocol-specific inputs
        self._gamma_dot_applied = kwargs.get("gamma_dot")
        self._sigma_applied = kwargs.get("sigma_applied")
        self._gamma_0 = kwargs.get("gamma_0")
        self._omega_laos = kwargs.get("omega")

        # Smart initialization based on protocol
        if test_mode in ["flow_curve", "steady_shear", "rotation"]:
            self.initialize_from_flow_curve(
                np.asarray(x), np.asarray(y)
            )
        elif test_mode == "oscillation":
            self.initialize_from_saos(
                np.asarray(x), np.real(np.asarray(y)), np.imag(np.asarray(y))
            )

        # Define model function for fitting
        def model_fn(x_fit, params):
            return self.model_function(x_fit, params, test_mode=test_mode)

        # Create objective and optimize
        objective = create_least_squares_objective(
            model_fn,
            x_jax,
            y_jax,
            use_log_residuals=kwargs.get(
                "use_log_residuals", test_mode == "flow_curve"
            ),
        )

        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            method=kwargs.get("method", "auto"),
            max_iter=kwargs.get("max_iter", 2000),
        )

        self.fitted_ = True
        self._nlsq_result = result

        logger.info(
            f"Fitted TNTSingleMode ({self._breakage}): "
            f"G={self.G:.2e}, τ_b={self.tau_b:.2e}, η_s={self.eta_s:.2e}"
        )

        return self

    def _predict(self, x, **kwargs):
        """Predict response using protocol-aware dispatch.

        Parameters
        ----------
        x : array-like
            Independent variable
        **kwargs
            Additional arguments including test_mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        # Build parameter array from ParameterSet (ordering matters)
        param_values = [
            float(self.parameters.get_value(name))
            for name in self.parameters.keys()
        ]
        params = jnp.array(param_values)
        return self.model_function(x_jax, params, test_mode=test_mode)

    def model_function(self, X, params, test_mode=None):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode. This is the
        stateless function used for both NLSQ optimization and Bayesian
        inference.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in ParameterSet order: [G, tau_b, eta_s, ...]
        test_mode : str, optional
            Override stored test mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack core parameters (always first 3)
        G = params[0]
        tau_b = params[1]
        eta_s = params[2]

        # Unpack variant parameters (dummy values for inactive variants)
        vp = self._unpack_variant_params(params)

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            if self._is_basic:
                return tnt_base_steady_stress_vec(X_jax, G, tau_b, eta_s)
            return self._variant_flow_curve_internal(
                X_jax, G, tau_b, eta_s, vp
            )

        elif mode == "oscillation":
            # All TNT variants linearize to Maxwell in SAOS
            G_prime, G_double_prime = tnt_saos_moduli_vec(
                X_jax, G, tau_b, eta_s
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, G, tau_b, eta_s, gamma_dot, vp
            )

        elif mode == "relaxation":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError(
                    "relaxation mode requires gamma_dot (pre-shear rate)"
                )
            return self._simulate_relaxation_internal(
                X_jax, G, tau_b, eta_s, gamma_dot, vp
            )

        elif mode == "creep":
            sigma_applied = self._sigma_applied
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G, tau_b, eta_s, sigma_applied, vp
            )

        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G, tau_b, eta_s, self._gamma_0, self._omega_laos, vp
            )
            return stress

        else:
            logger.warning(
                f"Unknown test_mode '{mode}', defaulting to flow_curve"
            )
            if self._is_basic:
                return tnt_base_steady_stress_vec(X_jax, G, tau_b, eta_s)
            return self._variant_flow_curve_internal(
                X_jax, G, tau_b, eta_s, vp
            )

    # =========================================================================
    # Variant Flow Curve (ODE-to-steady-state)
    # =========================================================================

    def _variant_flow_curve_internal(
        self,
        gamma_dot_arr: jnp.ndarray,
        G: float,
        tau_b: float,
        eta_s: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Compute variant flow curve by running ODE to steady state.

        For non-constant breakage, the steady-state conformation cannot be
        solved analytically. Instead, we integrate the ODE for ~50·τ_b at
        each shear rate and extract the final stress.
        """
        variant_ode = self._variant_ode
        is_fene = self._stress_type == "fene"

        def solve_single(gdot):
            def ode_fn(ti, yi, args):
                return variant_ode(
                    ti, yi, args["gdot"], args["G"], args["tau_b"],
                    args["nu"], args["m_break"], args["kappa"],
                    args["L_max"], args["xi"],
                )

            args = {"gdot": gdot, "G": G, "tau_b": tau_b, **vp}
            y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
            t_end = 50.0 * tau_b
            dt0 = tau_b / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term, solver, 0.0, t_end, dt0, y0,
                args=args, saveat=saveat,
                stepsize_controller=controller, max_steps=500_000,
            )

            S_final = sol.ys[0]
            if is_fene:
                tr_S = S_final[0] + S_final[1] + S_final[2]
                L2 = vp["L_max"] * vp["L_max"]
                f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
                sigma_el = G * f * S_final[3]
            else:
                sigma_el = G * S_final[3]

            return sigma_el + eta_s * gdot

        return jax.vmap(solve_single)(gamma_dot_arr)

    def _variant_steady_conformation(
        self,
        gamma_dot_arr: jnp.ndarray,
        G: float,
        tau_b: float,
        vp: dict,
    ) -> jnp.ndarray:
        """Compute variant steady-state conformation via ODE.

        Returns array of shape (N, 4) with [S_xx, S_yy, S_zz, S_xy]
        for each shear rate.
        """
        variant_ode = self._variant_ode

        def solve_single(gdot):
            def ode_fn(ti, yi, args):
                return variant_ode(
                    ti, yi, args["gdot"], args["G"], args["tau_b"],
                    args["nu"], args["m_break"], args["kappa"],
                    args["L_max"], args["xi"],
                )

            args = {"gdot": gdot, "G": G, "tau_b": tau_b, **vp}
            y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
            t_end = 50.0 * tau_b
            dt0 = tau_b / 20.0

            term = diffrax.ODETerm(ode_fn)
            solver = diffrax.Tsit5()
            controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)
            saveat = diffrax.SaveAt(ts=jnp.array([t_end]))

            sol = diffrax.diffeqsolve(
                term, solver, 0.0, t_end, dt0, y0,
                args=args, saveat=saveat,
                stepsize_controller=controller, max_steps=500_000,
            )
            return sol.ys[0]

        return jax.vmap(solve_single)(gamma_dot_arr)

    # =========================================================================
    # Analytical / Hybrid Predictions
    # =========================================================================

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict steady shear stress and viscosity.

        For constant breakage: analytical (UCM-like, no shear thinning).
        For non-constant breakage: ODE-to-steady-state (shear thinning).

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return (sigma, eta, N1)

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ (Pa), or (σ, η, N₁) if return_components=True
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)

        if self._is_basic:
            sigma = tnt_base_steady_stress_vec(
                gd, self.G, self.tau_b, self.eta_s
            )
            if return_components:
                eta = sigma / jnp.maximum(gd, 1e-20)
                N1 = tnt_base_steady_n1_vec(gd, self.G, self.tau_b)
                return np.asarray(sigma), np.asarray(eta), np.asarray(N1)
            return np.asarray(sigma)

        # Variant: compute via ODE-to-steady-state
        vp = self._get_variant_args()
        sigma = self._variant_flow_curve_internal(
            gd, self.G, self.tau_b, self.eta_s, vp
        )

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            # N1 from steady-state conformation
            S_ss = self._variant_steady_conformation(
                gd, self.G, self.tau_b, vp
            )
            if self._stress_type == "fene":
                tr_S = S_ss[:, 0] + S_ss[:, 1] + S_ss[:, 2]
                L2 = vp["L_max"] ** 2
                f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
                N1 = self.G * f * (S_ss[:, 0] - S_ss[:, 1])
            else:
                N1 = self.G * (S_ss[:, 0] - S_ss[:, 1])
            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)

        return np.asarray(sigma)

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        In the linear regime, TNT reduces to single-mode Maxwell:
        G'(ω) = G·(ωτ_b)²/(1+(ωτ_b)²)
        G''(ω) = G·(ωτ_b)/(1+(ωτ_b)²) + η_s·ω

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array (rad/s)
        return_components : bool, default True
            If True, return (G', G'')

        Returns
        -------
        tuple or np.ndarray
            (G', G'') if return_components=True, else |G*|
        """
        w = jnp.asarray(omega, dtype=jnp.float64)
        G_prime, G_double_prime = tnt_saos_moduli_vec(
            w, self.G, self.tau_b, self.eta_s
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)

        G_star_mag = jnp.sqrt(G_prime**2 + G_double_prime**2)
        return np.asarray(G_star_mag)

    def predict_normal_stresses(
        self,
        gamma_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict first and second normal stress differences.

        Basic TNT (constant/linear/UC): N₁ = 2G·(τ_b·γ̇)², N₂ = 0.
        Gordon-Schowalter (ξ > 0): N₂ ≠ 0.
        FENE-P: N₁ enhanced by Peterlin factor f(trS).

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (N₁, N₂) in Pa
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)

        if self._is_basic:
            N1 = tnt_base_steady_n1_vec(gd, self.G, self.tau_b)
            N2 = jnp.zeros_like(N1)
            return np.asarray(N1), np.asarray(N2)

        # Variant: compute from steady-state conformation
        vp = self._get_variant_args()
        S_ss = self._variant_steady_conformation(
            gd, self.G, self.tau_b, vp
        )

        if self._stress_type == "fene":
            tr_S = S_ss[:, 0] + S_ss[:, 1] + S_ss[:, 2]
            L2 = vp["L_max"] ** 2
            f = L2 / jnp.maximum(L2 - tr_S, 1e-10)
            N1 = self.G * f * (S_ss[:, 0] - S_ss[:, 1])
            N2 = self.G * f * (S_ss[:, 1] - S_ss[:, 2])
        else:
            N1 = self.G * (S_ss[:, 0] - S_ss[:, 1])
            N2 = self.G * (S_ss[:, 1] - S_ss[:, 2])

        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # ODE-Based Internal Simulations (for model_function)
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        eta_s: float,
        gamma_dot: float,
        vp: dict | None = None,
    ) -> jnp.ndarray:
        """Internal startup simulation for model_function.

        Returns total shear stress σ_xy(t).
        """
        if vp is None:
            vp = {"nu": 0.0, "m_break": 0.0, "kappa": 0.0,
                  "L_max": 10.0, "xi": 0.0}

        variant_ode = self._variant_ode

        def ode_fn(ti, yi, args):
            return variant_ode(
                ti, yi, args["gamma_dot"], args["G"], args["tau_b"],
                args["nu"], args["m_break"], args["kappa"],
                args["L_max"], args["xi"],
            )

        args = {"gamma_dot": gamma_dot, "G": G, "tau_b": tau_b, **vp}
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        return self._compute_stress_from_conformation(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G, eta_s, gamma_dot, vp,
        )

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        eta_s: float,
        gamma_dot_preshear: float,
        vp: dict | None = None,
    ) -> jnp.ndarray:
        """Internal relaxation simulation for model_function.

        For constant breakage, relaxation is analytical (single exponential).
        For non-constant breakage, uses ODE integration.
        """
        if vp is None:
            vp = {"nu": 0.0, "m_break": 0.0, "kappa": 0.0,
                  "L_max": 10.0, "xi": 0.0}

        if self._breakage == "constant" and self._stress_type == "linear":
            # Analytical: σ(t) = G·τ_b·γ̇·exp(-t/τ_b)
            sigma_0 = G * tau_b * gamma_dot_preshear
            return tnt_base_relaxation_vec(t, sigma_0, tau_b)

        # ODE-based relaxation for non-constant breakage
        # First find steady-state conformation from pre-shear
        S_xx_0, S_yy_0, S_zz_0, S_xy_0 = tnt_base_steady_conformation(
            gamma_dot_preshear, tau_b
        )
        y0 = jnp.array([S_xx_0, S_yy_0, S_zz_0, S_xy_0], dtype=jnp.float64)

        variant_relax_ode = self._variant_relax_ode

        def ode_fn(ti, yi, args):
            return variant_relax_ode(
                ti, yi, args["G"], args["tau_b"],
                args["nu"], args["m_break"], args["kappa"],
                args["L_max"], args["xi"],
            )

        args = {"G": G, "tau_b": tau_b, **vp}

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        return self._compute_stress_from_conformation(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G, eta_s, 0.0, vp,
        )

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        eta_s: float,
        sigma_applied: float,
        vp: dict | None = None,
    ) -> jnp.ndarray:
        """Internal creep simulation for model_function.

        Returns accumulated strain γ(t).
        """
        if vp is None:
            vp = {"nu": 0.0, "m_break": 0.0, "kappa": 0.0,
                  "L_max": 10.0, "xi": 0.0}

        variant_creep_ode = self._variant_creep_ode

        def ode_fn(ti, yi, args):
            return variant_creep_ode(
                ti, yi, args["sigma_applied"], args["G"], args["tau_b"],
                args["eta_s"], args["nu"], args["m_break"],
                args["kappa"], args["L_max"], args["xi"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G": G, "tau_b": tau_b, "eta_s": eta_s,
            **vp,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        return sol.ys[:, 4]  # γ (strain)

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G: float,
        tau_b: float,
        eta_s: float,
        gamma_0: float,
        omega: float,
        vp: dict | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function.

        Returns (strain, stress) arrays.
        """
        if vp is None:
            vp = {"nu": 0.0, "m_break": 0.0, "kappa": 0.0,
                  "L_max": 10.0, "xi": 0.0}

        variant_laos_ode = self._variant_laos_ode

        def ode_fn(ti, yi, args):
            return variant_laos_ode(
                ti, yi, args["gamma_0"], args["omega"],
                args["G"], args["tau_b"],
                args["nu"], args["m_break"], args["kappa"],
                args["L_max"], args["xi"],
            )

        args = {
            "gamma_0": gamma_0, "omega": omega,
            "G": G, "tau_b": tau_b,
            **vp,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        strain = gamma_0 * jnp.sin(omega * t)
        gamma_dot_t = gamma_0 * omega * jnp.cos(omega * t)
        stress = self._compute_stress_from_conformation(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            G, eta_s, gamma_dot_t, vp,
        )

        return strain, stress

    # =========================================================================
    # Public Simulation Methods (return numpy arrays)
    # =========================================================================

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return full conformation tensor components

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ(t), or (S_xx, S_yy, S_xy, S_zz) if return_full
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        vp = self._get_variant_args()
        variant_ode = self._variant_ode

        def ode_fn(ti, yi, args):
            return variant_ode(
                ti, yi, args["gamma_dot"], args["G"], args["tau_b"],
                args["nu"], args["m_break"], args["kappa"],
                args["L_max"], args["xi"],
            )

        args = {
            "gamma_dot": gamma_dot, "G": self.G, "tau_b": self.tau_b,
            **vp,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = self._get_ode_solver()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t_jax[0]
        t1 = t_jax[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "S_xx": np.asarray(sol.ys[:, 0]),
            "S_yy": np.asarray(sol.ys[:, 1]),
            "S_zz": np.asarray(sol.ys[:, 2]),
            "S_xy": np.asarray(sol.ys[:, 3]),
        }

        if return_full:
            return (
                np.asarray(sol.ys[:, 0]),
                np.asarray(sol.ys[:, 1]),
                np.asarray(sol.ys[:, 3]),
                np.asarray(sol.ys[:, 2]),
            )

        # Total stress: σ = G·f(S)·S_xy + η_s·γ̇ (f=1 for linear, FENE-P otherwise)
        sigma = self._compute_stress_from_conformation(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            self.G, self.eta_s, gamma_dot, vp,
        )
        return np.asarray(sigma)

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate stress relaxation after cessation of steady shear.

        For constant breakage + linear stress, relaxation is analytical:
        σ(t) = G·S_xy(0)·exp(-t/τ_b).
        For non-constant breakage or FENE stress, ODE integration is used.

        Parameters
        ----------
        t : np.ndarray
            Time array (s), starting from t=0 (cessation)
        gamma_dot_preshear : float
            Shear rate before cessation (1/s)
        return_full : bool, default False
            If True, return full conformation tensor components

        Returns
        -------
        np.ndarray or tuple
            Relaxing stress σ(t), or (S_xx, S_yy, S_xy, S_zz) if return_full
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        vp = self._get_variant_args()

        use_ode = return_full or not self._is_basic

        if not use_ode:
            # Analytical relaxation for constant breakage + linear stress
            sigma_0 = self.G * self.tau_b * gamma_dot_preshear
            sigma = tnt_base_relaxation_vec(t_jax, sigma_0, self.tau_b)
            return np.asarray(sigma)

        # ODE-based relaxation (variant or return_full)
        # Find steady-state conformation from pre-shear
        if self._is_basic:
            S_xx_0, S_yy_0, S_zz_0, S_xy_0 = tnt_base_steady_conformation(
                gamma_dot_preshear, self.tau_b
            )
        else:
            # Run startup ODE to steady state for variant breakage
            t_ss = jnp.linspace(0.0, 50.0 * self.tau_b, 2000)
            _ = self._simulate_startup_internal(
                t_ss, self.G, self.tau_b, self.eta_s,
                gamma_dot_preshear, vp,
            )
            # Re-run to get conformation (use trajectory if available)
            S_xx_0, S_yy_0, S_zz_0, S_xy_0 = (
                tnt_base_steady_conformation(gamma_dot_preshear, self.tau_b)
            )
            # Override with ODE steady state via _variant_steady_conformation
            ss = self._variant_steady_conformation(
                jnp.array([gamma_dot_preshear]), self.G, self.tau_b, vp
            )
            S_xx_0, S_yy_0, S_zz_0, S_xy_0 = (
                ss[0, 0], ss[0, 1], ss[0, 2], ss[0, 3]
            )

        y0 = jnp.array(
            [S_xx_0, S_yy_0, S_zz_0, S_xy_0], dtype=jnp.float64
        )

        variant_relax_ode = self._variant_relax_ode

        def ode_fn(ti, yi, args):
            return variant_relax_ode(
                ti, yi, args["G"], args["tau_b"],
                args["nu"], args["m_break"], args["kappa"],
                args["L_max"], args["xi"],
            )

        args = {"G": self.G, "tau_b": self.tau_b, **vp}

        term = diffrax.ODETerm(ode_fn)
        solver = self._get_ode_solver()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t_jax[0]
        t1 = t_jax[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
        )

        self._trajectory = {
            "t": np.asarray(t_jax),
            "S_xx": np.asarray(sol.ys[:, 0]),
            "S_yy": np.asarray(sol.ys[:, 1]),
            "S_zz": np.asarray(sol.ys[:, 2]),
            "S_xy": np.asarray(sol.ys[:, 3]),
        }

        if return_full:
            return (
                np.asarray(sol.ys[:, 0]),
                np.asarray(sol.ys[:, 1]),
                np.asarray(sol.ys[:, 3]),
                np.asarray(sol.ys[:, 2]),
            )

        # Stress from conformation: σ = G·f(S)·S_xy + η_s·γ̇ (γ̇=0 in relaxation)
        sigma = self._compute_stress_from_conformation(
            sol.ys[:, 0], sol.ys[:, 1], sol.ys[:, 2], sol.ys[:, 3],
            self.G, self.eta_s, 0.0, vp,
        )
        return np.asarray(sigma)

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_applied: float,
        return_rate: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate creep deformation under constant stress.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float
            Applied constant stress (Pa)
        return_rate : bool, default False
            If True, also return shear rate γ̇(t)

        Returns
        -------
        np.ndarray or tuple
            Strain γ(t), or (γ, γ̇) if return_rate=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        vp = self._get_variant_args()
        variant_creep_ode = self._variant_creep_ode

        def ode_fn(ti, yi, args):
            return variant_creep_ode(
                ti, yi, args["sigma_applied"], args["G"], args["tau_b"],
                args["eta_s"], args["nu"], args["m_break"],
                args["kappa"], args["L_max"], args["xi"],
            )

        args = {
            "sigma_applied": sigma_applied,
            "G": self.G, "tau_b": self.tau_b, "eta_s": self.eta_s,
            **vp,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = self._get_ode_solver()

        # Creep with nonlinear breakage/stress is stiff — use relaxed
        # tolerances and very small initial dt for non-basic variants.
        if self._is_basic:
            rtol, atol = 1e-6, 1e-8
        else:
            rtol, atol = 1e-4, 1e-6
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

        t0 = t_jax[0]
        t1 = t_jax[-1]
        n_steps_hint = max(len(t), 1000)
        if not self._is_basic:
            n_steps_hint *= 10  # Smaller initial dt for stiff variants
        dt0 = (t1 - t0) / n_steps_hint

        saveat = diffrax.SaveAt(ts=t_jax)

        sol = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, y0,
            args=args, saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

        gamma = np.asarray(sol.ys[:, 4])
        S_xy = np.asarray(sol.ys[:, 3])

        self._trajectory = {
            "t": np.asarray(t_jax),
            "gamma": gamma,
            "S_xy": S_xy,
        }

        if return_rate:
            eta_s_reg = max(self.eta_s, 1e-10 * self.G * self.tau_b)
            # For FENE: σ_elastic = G·f(trS)·S_xy, for linear: G·S_xy
            if self._stress_type == "fene":
                L2 = vp["L_max"] ** 2
                tr_S = (np.asarray(sol.ys[:, 0])
                        + np.asarray(sol.ys[:, 1])
                        + np.asarray(sol.ys[:, 2]))
                f_pet = L2 / np.maximum(L2 - tr_S, 1e-10)
                sigma_elastic = self.G * f_pet * S_xy
            else:
                sigma_elastic = self.G * S_xy
            gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg
            return gamma, gamma_dot

        return gamma

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
        n_cycles: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Simulate Large-Amplitude Oscillatory Shear (LAOS).

        Parameters
        ----------
        t : np.ndarray
            Time array (s), or None to auto-generate
        gamma_0 : float
            Strain amplitude (dimensionless)
        omega : float
            Angular frequency (rad/s)
        n_cycles : int, optional
            Number of oscillation cycles (overrides t)

        Returns
        -------
        dict
            Dictionary with keys: 't', 'strain', 'stress', 'strain_rate'
        """
        if n_cycles is not None:
            T = 2 * np.pi / omega
            t = np.linspace(0, n_cycles * T, n_cycles * 200)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        vp = self._get_variant_args()

        strain, stress = self._simulate_laos_internal(
            t_jax, self.G, self.tau_b, self.eta_s, gamma_0, omega, vp
        )

        strain_rate = gamma_0 * omega * jnp.cos(omega * t_jax)

        self._trajectory = {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
        }

        return {
            "t": np.asarray(t_jax),
            "strain": np.asarray(strain),
            "stress": np.asarray(stress),
            "strain_rate": np.asarray(strain_rate),
        }

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_overshoot_ratio(
        self,
        gamma_dot: float,
        t_max: float | None = None,
    ) -> tuple[float, float]:
        """Compute stress overshoot ratio in startup flow.

        For constant breakage, there is no overshoot (UCM behavior).
        Overshoot requires Bell or stretch-dependent breakage.

        Parameters
        ----------
        gamma_dot : float
            Shear rate (1/s)
        t_max : float, optional
            Maximum simulation time (default: 20·τ_b)

        Returns
        -------
        tuple[float, float]
            (overshoot_ratio, strain_at_overshoot)
        """
        if t_max is None:
            t_max = 20 * self.tau_b

        t = np.linspace(0, t_max, 1000)
        sigma = self.simulate_startup(t, gamma_dot)

        peak_idx = np.argmax(sigma)
        sigma_max = sigma[peak_idx]
        strain_at_peak = gamma_dot * t[peak_idx]

        sigma_ss = sigma[-1]
        overshoot_ratio = sigma_max / sigma_ss if sigma_ss > 0 else 1.0

        return overshoot_ratio, strain_at_peak

    def get_relaxation_spectrum(
        self,
        t: np.ndarray | None = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get relaxation modulus G(t).

        For single-mode TNT: G(t) = G·exp(-t/τ_b)

        Parameters
        ----------
        t : np.ndarray, optional
            Time array (default: logspace from 0.01·τ_b to 100·τ_b)
        n_points : int, default 100
            Number of points if t not provided

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, G(t))
        """
        if t is None:
            t = np.logspace(
                np.log10(0.01 * self.tau_b),
                np.log10(100 * self.tau_b),
                n_points,
            )

        G_t = self.G * np.exp(-t / self.tau_b)
        return t, G_t

    def extract_laos_harmonics(
        self,
        laos_result: dict[str, np.ndarray],
        n_harmonics: int = 5,
    ) -> dict[str, np.ndarray]:
        """Extract Fourier harmonics from LAOS stress response.

        Parameters
        ----------
        laos_result : dict
            Result from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Dictionary with 'n', 'sigma_prime', 'sigma_double_prime',
            'intensity', 'I3_I1'
        """
        t = laos_result["t"]
        stress = laos_result["stress"]
        strain = laos_result["strain"]

        fft_strain = np.fft.fft(strain)
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        omega = 2 * np.pi * np.abs(
            freqs[np.argmax(np.abs(fft_strain[1:])) + 1]
        )

        harmonics = [2 * i + 1 for i in range(n_harmonics)]
        sigma_prime_list: list[float] = []
        sigma_double_prime_list: list[float] = []

        for n in harmonics:
            sin_basis = np.sin(n * omega * t)
            cos_basis = np.cos(n * omega * t)

            dt = t[1] - t[0]
            sigma_n_prime = (
                2 * np.trapezoid(stress * sin_basis, dx=dt) / (t[-1] - t[0])
            )
            sigma_n_double_prime = (
                2 * np.trapezoid(stress * cos_basis, dx=dt) / (t[-1] - t[0])
            )

            sigma_prime_list.append(sigma_n_prime)
            sigma_double_prime_list.append(sigma_n_double_prime)

        sigma_prime = np.array(sigma_prime_list)
        sigma_double_prime = np.array(sigma_double_prime_list)
        intensity = np.sqrt(sigma_prime**2 + sigma_double_prime**2)

        return {
            "n": np.array(harmonics),
            "sigma_prime": sigma_prime,
            "sigma_double_prime": sigma_double_prime,
            "intensity": intensity,
            "I3_I1": (
                intensity[1] / intensity[0] if intensity[0] > 0 else 0.0
            ),
        }
