"""Base class for FIKH (Fractional IKH) models.

This module provides the FIKHBase class that encapsulates shared functionality
for both FIKH (single-mode) and FMLIKH (multi-layer) fractional thixotropic
elasto-viscoplastic models.

Key features:
- Caputo fractional derivative for structure evolution (power-law memory)
- Full thermokinematic coupling (Arrhenius viscosity, thermal yield stress)
- Support for 6 protocols: flow_curve, startup, relaxation, creep, oscillation, LAOS

Inheritance:
    BaseModel + FractionalModelMixin → FIKHBase → FIKH, FMLIKH
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode
from rheojax.logging import get_logger
from rheojax.models.fractional.fractional_mixin import (
    FRACTIONAL_ORDER_BOUNDS,
    FractionalModelMixin,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


class FIKHBase(BaseModel, FractionalModelMixin):
    """Base class for Fractional Isotropic-Kinematic Hardening models.

    FIKH models combine:
    1. Maxwell viscoelastic element (stress relaxation)
    2. Armstrong-Frederick kinematic hardening (backstress evolution)
    3. Fractional thixotropy (Caputo derivative for structure evolution)
    4. Optional thermokinematic coupling (Arrhenius + heat generation)

    State Evolution:
        - Stress: dσ/dt = G(γ̇ - γ̇ᵖ) - σ/τ
        - Backstress: dα = C·dγᵖ - γ_dyn·|α|^(m-1)·α·|dγᵖ|
        - Structure: D^α λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|
        - Temperature: ρc_p·dT/dt = χ·σ·γ̇ᵖ - h·(T-T_env)

    Supported Protocols:
        - FLOW_CURVE: Steady-state stress vs shear rate
        - STARTUP: Constant rate startup (stress overshoot)
        - RELAXATION: Stress relaxation at fixed strain
        - CREEP: Constant stress creep (delayed yielding)
        - OSCILLATION: SAOS (G', G'')
        - LAOS: Large amplitude oscillatory shear

    Attributes:
        include_thermal: Whether thermal coupling is enabled.
        alpha_structure: Fractional order for structure evolution.
        n_history: Number of history points for Caputo derivative.
    """

    # Protocols supported by FIKH models
    # Note: LAOS is handled as a special case of OSCILLATION
    SUPPORTED_PROTOCOLS = [
        TestMode.FLOW_CURVE,
        TestMode.STARTUP,
        TestMode.RELAXATION,
        TestMode.CREEP,
        TestMode.OSCILLATION,
    ]

    def __init__(
        self,
        include_thermal: bool = True,
        include_isotropic_hardening: bool = False,
        alpha_structure: float = 0.5,
        n_history: int = 100,
    ):
        """Initialize FIKHBase model.

        Args:
            include_thermal: Whether to include thermal coupling.
            include_isotropic_hardening: Whether to include isotropic hardening R.
            alpha_structure: Fractional order for structure evolution (0 < α < 1).
            n_history: Number of history points for Caputo derivative.
        """
        super().__init__()
        self.include_thermal = include_thermal
        self.include_isotropic_hardening = include_isotropic_hardening
        self.alpha_structure = alpha_structure
        self.n_history = n_history
        self._test_mode: str | None = None  # For Bayesian closure

        # Setup parameters
        self._setup_base_parameters()
        if include_thermal:
            self._setup_thermal_parameters()
        if include_isotropic_hardening:
            self._setup_isotropic_hardening_parameters()

        logger.debug(
            "Initialized FIKHBase",
            include_thermal=include_thermal,
            include_isotropic_hardening=include_isotropic_hardening,
            alpha_structure=alpha_structure,
            n_history=n_history,
        )

    def _setup_base_parameters(self) -> None:
        """Setup base FIKH parameters (mechanical + fractional)."""
        # Elasticity
        self.parameters.add(
            "G",
            value=1e3,
            bounds=(1e-1, 1e9),
            units="Pa",
            description="Shear modulus",
        )
        self.parameters.add(
            "eta",
            value=1e6,
            bounds=(1e-3, 1e12),
            units="Pa s",
            description="Maxwell viscosity (relaxation time = eta/G)",
        )

        # Kinematic Hardening (Armstrong-Frederick)
        self.parameters.add(
            "C",
            value=5e2,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Kinematic hardening modulus",
        )
        self.parameters.add(
            "gamma_dyn",
            value=1.0,
            bounds=(0.0, 1e4),
            units="-",
            description="Dynamic recovery parameter",
        )
        self.parameters.add(
            "m",
            value=1.0,
            bounds=(0.5, 3.0),
            units="-",
            description="AF recovery exponent",
        )

        # Yield Stress
        self.parameters.add(
            "sigma_y0",
            value=10.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Minimal yield stress (destructured)",
        )
        self.parameters.add(
            "delta_sigma_y",
            value=50.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Structural yield stress contribution",
        )

        # Thixotropy
        self.parameters.add(
            "tau_thix",
            value=1.0,
            bounds=(1e-6, 1e12),
            units="s",
            description="Thixotropic rebuilding time scale",
        )
        self.parameters.add(
            "Gamma",
            value=0.5,
            bounds=(0.0, 1e4),
            units="-",
            description="Structural breakdown coefficient",
        )

        # Fractional order
        self.parameters.add(
            "alpha_structure",
            value=self.alpha_structure,
            bounds=FRACTIONAL_ORDER_BOUNDS,
            units="-",
            description="Fractional order for structure evolution",
        )

        # Viscosity
        self.parameters.add(
            "eta_inf",
            value=0.1,
            bounds=(0.0, 1e9),
            units="Pa s",
            description="High-shear (solvent) viscosity",
        )
        self.parameters.add(
            "mu_p",
            value=1e-3,
            bounds=(1e-9, 1e3),
            units="Pa s",
            description="Plastic viscosity (Perzyna regularization)",
        )

    def _setup_thermal_parameters(self) -> None:
        """Setup thermal coupling parameters."""
        # Reference temperature
        self.parameters.add(
            "T_ref",
            value=298.15,
            bounds=(200.0, 500.0),
            units="K",
            description="Reference temperature",
        )

        # Activation energies
        self.parameters.add(
            "E_a",
            value=5e4,
            bounds=(0.0, 2e5),
            units="J/mol",
            description="Viscosity activation energy",
        )
        self.parameters.add(
            "E_y",
            value=3e4,
            bounds=(0.0, 2e5),
            units="J/mol",
            description="Yield stress activation energy",
        )

        # Structure-temperature coupling
        self.parameters.add(
            "m_y",
            value=1.0,
            bounds=(0.5, 2.0),
            units="-",
            description="Structure exponent for yield stress",
        )

        # Thermal properties
        self.parameters.add(
            "rho_cp",
            value=4e6,
            bounds=(1e5, 1e8),
            units="J/(m³·K)",
            description="Volumetric heat capacity",
        )
        self.parameters.add(
            "chi",
            value=0.9,
            bounds=(0.0, 1.0),
            units="-",
            description="Taylor-Quinney coefficient",
        )
        self.parameters.add(
            "h",
            value=100.0,
            bounds=(0.0, 1e6),
            units="W/(m²·K)",
            description="Heat transfer coefficient",
        )
        self.parameters.add(
            "T_env",
            value=298.15,
            bounds=(200.0, 500.0),
            units="K",
            description="Environmental temperature",
        )

    def _setup_isotropic_hardening_parameters(self) -> None:
        """Setup isotropic hardening parameters."""
        self.parameters.add(
            "Q_iso",
            value=0.0,
            bounds=(0.0, 1e9),
            units="Pa",
            description="Isotropic hardening saturation",
        )
        self.parameters.add(
            "b_iso",
            value=1.0,
            bounds=(0.0, 100.0),
            units="-",
            description="Isotropic hardening rate",
        )

    def _get_params_dict(self) -> dict[str, Any]:
        """Get current parameters as dictionary."""
        return dict(
            zip(
                self.parameters.keys(),
                self.parameters.get_values(),
                strict=False,
            )
        )

    def _extract_time_strain(
        self, X: ArrayLike, **kwargs
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract time and strain arrays from input.

        Args:
            X: Input data - can be:
                - RheoData object with .time and .strain attributes
                - (2, N) array of [time, strain]
                - Time array with strain in kwargs

        Returns:
            Tuple of (times, strains) as JAX arrays.

        Raises:
            ValueError: If strain data cannot be extracted.
        """
        # Check for RheoData
        if hasattr(X, "time") and hasattr(X, "strain"):
            return jnp.asarray(X.time), jnp.asarray(X.strain)

        X_arr = jnp.asarray(X)

        # (2, N) array
        if X_arr.ndim == 2 and X_arr.shape[0] == 2:
            return X_arr[0], X_arr[1]

        # Time array with strain in kwargs
        if "strain" in kwargs:
            return X_arr, jnp.asarray(kwargs["strain"])

        raise ValueError(
            "FIKH models require both time and strain history. "
            "Pass RheoData, X of shape (2, N), or X=time with strain kwarg."
        )

    def _get_initial_state(
        self,
        mode: str,
        params: dict[str, Any],
        T_init: float | None = None,
        sigma_0: float | None = None,
        lambda_0: float = 1.0,
    ) -> jnp.ndarray:
        """Get initial state vector for ODE integration.

        Args:
            mode: Protocol mode.
            params: Parameter dictionary.
            T_init: Initial temperature (if thermal enabled).
            sigma_0: Initial stress (for relaxation).
            lambda_0: Initial structure parameter.

        Returns:
            Initial state array.
        """
        T_ref = params.get("T_ref", 298.15)
        T_0 = T_init if T_init is not None else T_ref

        if self.include_thermal:
            # State: [σ, α, T, γᵖ, λ]
            if mode == "relaxation" and sigma_0 is not None:
                return jnp.array([sigma_0, 0.0, T_0, 0.0, lambda_0])
            elif mode == "creep":
                # State: [γ, α, T, γᵖ, λ]
                return jnp.array([0.0, 0.0, T_0, 0.0, lambda_0])
            else:
                return jnp.array([0.0, 0.0, T_0, 0.0, lambda_0])
        else:
            # State: [σ, α, γᵖ, λ]
            if mode == "relaxation" and sigma_0 is not None:
                return jnp.array([sigma_0, 0.0, 0.0, lambda_0])
            elif mode == "creep":
                # State: [γ, α, γᵖ, λ]
                return jnp.array([0.0, 0.0, 0.0, lambda_0])
            else:
                return jnp.array([0.0, 0.0, 0.0, lambda_0])

    def _simulate_transient(
        self,
        t: jnp.ndarray,
        params: dict[str, Any],
        mode: str,
        gamma_dot: float | None = None,
        sigma_applied: float | None = None,
        sigma_0: float | None = None,
        T_init: float | None = None,
    ) -> jnp.ndarray:
        """Simulate transient response using Diffrax ODE integration.

        Args:
            t: Time array.
            params: Parameter dictionary.
            mode: Protocol mode ('startup', 'relaxation', 'creep').
            gamma_dot: Applied shear rate (for startup).
            sigma_applied: Applied stress (for creep).
            sigma_0: Initial stress (for relaxation).
            T_init: Initial temperature.

        Returns:
            Primary output array (stress for startup/relaxation, strain for creep).
        """
        import diffrax

        from rheojax.models.fikh._kernels import (
            fikh_creep_ode_rhs,
            fikh_maxwell_ode_rhs,
        )

        # Build args
        args = dict(params)
        args["include_thermal"] = self.include_thermal

        # Select ODE function and initial state
        if mode == "creep":
            ode_fn = fikh_creep_ode_rhs
            args["sigma_applied"] = (
                sigma_applied if sigma_applied is not None else 100.0
            )
            y0 = self._get_initial_state(mode, params, T_init, lambda_0=1.0)
        elif mode == "relaxation":
            ode_fn = fikh_maxwell_ode_rhs
            args["gamma_dot"] = 0.0
            sigma_init = (
                sigma_0
                if sigma_0 is not None
                else params.get("sigma_y0", 10.0) + params.get("delta_sigma_y", 50.0)
            )
            y0 = self._get_initial_state(
                mode, params, T_init, sigma_0=sigma_init, lambda_0=0.5
            )
        else:  # startup
            ode_fn = fikh_maxwell_ode_rhs
            args["gamma_dot"] = gamma_dot if gamma_dot is not None else 1.0
            y0 = self._get_initial_state(mode, params, T_init, lambda_0=1.0)

        # Diffrax setup
        term = diffrax.ODETerm(lambda ti, yi, args_i: ode_fn(ti, yi, args_i))
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t0, t1 = t[0], t[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0,
            t1,
            dt0,
            y0,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=10_000_000,
            throw=False,
        )

        # Extract primary variable
        result = sol.ys[:, 0]

        # Handle solver failures
        result = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            result,
            jnp.nan * jnp.ones_like(result),
        )

        # Add viscous background for startup
        if mode == "startup" and params.get("eta_inf", 0.0) > 0:
            result = result + params["eta_inf"] * args["gamma_dot"]

        return result

    def _validate_test_mode(self, test_mode: str | TestMode | None) -> TestMode:
        """Validate and convert test mode.

        Args:
            test_mode: Test mode string or enum.

        Returns:
            Validated TestMode enum.

        Raises:
            ValueError: If test mode is not supported.
        """
        if test_mode is None:
            test_mode = self._test_mode or "startup"

        # Handle string conversion
        if isinstance(test_mode, str):
            mode_str = test_mode.lower()
            if mode_str == "laos":
                # LAOS uses return mapping (like startup), not oscillation
                # Map to STARTUP for routing purposes
                return TestMode.STARTUP
            elif mode_str == "saos":
                return TestMode.OSCILLATION
            test_mode = TestMode(mode_str)

        if test_mode not in self.SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"Test mode {test_mode} not supported by FIKH. "
                f"Supported: {[p.value for p in self.SUPPORTED_PROTOCOLS]} + ['laos']"
            )

        return test_mode
