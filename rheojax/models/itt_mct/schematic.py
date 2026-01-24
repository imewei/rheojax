"""ITT-MCT Schematic F₁₂ Model.

The F₁₂ schematic model is a simplified Mode-Coupling Theory that captures
the essential physics of the glass transition with minimal parameters:

- Glass transition at v₂ = 4 (for v₁ = 0)
- Yield stress in glass state (ε > 0)
- Shear thinning from cage breaking
- Two-step relaxation (β and α processes)

Parameters
----------
v1 : float
    Linear vertex coefficient (typically 0)
v2 : float
    Quadratic vertex coefficient (glass transition at v₂_c = 4)
Gamma : float
    Bare relaxation rate (1/s)
gamma_c : float
    Critical strain for cage breaking (dimensionless)
G_inf : float
    High-frequency modulus (Pa)
epsilon : float
    Separation parameter ε = (v₂ - v₂_c)/v₂_c

References
----------
Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids", Chapter 4
Fuchs M. & Cates M.E. (2002) Phys. Rev. Lett. 89, 248304
"""

from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from scipy.integrate import solve_ivp

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger
from rheojax.models.itt_mct._base import ITTMCTBase
from rheojax.models.itt_mct._kernels import (
    compute_complex_modulus_from_correlator,
    extract_laos_harmonics,
    f12_equilibrium_correlator_rhs,
    f12_memory,
    f12_steady_state_stress,
    f12_volterra_creep_rhs,
    f12_volterra_flow_curve_rhs,
    f12_volterra_laos_rhs,
    f12_volterra_relaxation_rhs,
    f12_volterra_startup_rhs,
    strain_decorrelation,
)
from rheojax.utils.mct_kernels import (
    glass_transition_criterion,
    prony_decompose_memory,
)

# Try to import diffrax-based solvers for fast ODE integration
try:
    from rheojax.models.itt_mct._kernels_diffrax import (
        is_diffrax_available,
        precompile_flow_curve_solver,
        solve_flow_curve_batch,
    )

    _HAS_DIFFRAX = is_diffrax_available()
except ImportError:
    _HAS_DIFFRAX = False

    def precompile_flow_curve_solver(*args, **kwargs):
        """Stub when diffrax not available."""
        return 0.0

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


@ModelRegistry.register(
    "itt_mct_schematic",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.LAOS,
    ],
)
class ITTMCTSchematic(ITTMCTBase):
    """ITT-MCT Schematic F₁₂ Model.

    The F₁₂ model uses a quadratic memory kernel m(Φ) = v₁Φ + v₂Φ²
    to describe the cage effect in dense colloidal suspensions.

    The glass transition occurs when the non-ergodicity parameter f
    (long-time limit of Φ) becomes non-zero, which happens at v₂ = v₂_c = 4
    for v₁ = 0.

    Parameters
    ----------
    epsilon : float, optional
        Separation parameter. If provided, v₂ is set to achieve this ε.
        ε < 0: fluid state
        ε = 0: critical point
        ε > 0: glass state
    v2 : float, optional
        Quadratic vertex coefficient. Alternative to epsilon.
    integration_method : {"volterra", "history"}, default "volterra"
        Integration method for memory kernel
    n_prony_modes : int, default 10
        Number of Prony modes for Volterra integration
    decorrelation_form : {"gaussian", "lorentzian"}, default "gaussian"
        Strain decorrelation function form:
        - "gaussian": h(γ) = exp(-(γ/γ_c)²) - faster exponential decay
        - "lorentzian": h(γ) = 1/(1+(γ/γ_c)²) - slower algebraic decay
    memory_form : {"simplified", "full"}, default "simplified"
        Memory kernel form:
        - "simplified": single decorrelation m(Φ) = h[γ_acc] × (v₁Φ + v₂Φ²)
        - "full": two-time decorrelation m(t,s,t₀) = h[γ(t,t₀)] × h[γ(t,s)] × (v₁Φ + v₂Φ²)
    stress_form : {"schematic", "microscopic"}, default "schematic"
        Stress computation form:
        - "schematic": σ = G_∞ × γ̇ × ∫ Φ² × h(γ) dt (standard schematic)
        - "microscopic": σ = (k_BT/60π²) × ∫dk k⁴ [S'/S²]² Φ² (structure factor weighted)
    phi_volume : float, optional
        Volume fraction for Percus-Yevick S(k). Required if stress_form="microscopic".
    k_BT : float, default 1.0
        Thermal energy k_B × T in Joules. Default 1.0 gives dimensionless stress.

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters with the following:
        - v1: Linear vertex (default 0)
        - v2: Quadratic vertex (default 2.0, fluid state)
        - Gamma: Bare relaxation rate (default 1.0 s⁻¹)
        - gamma_c: Critical strain (default 0.1)
        - G_inf: High-frequency modulus (default 1e6 Pa)
    memory_form : str
        The memory kernel form ("simplified" or "full")
    stress_form : str
        The stress computation form ("schematic" or "microscopic")

    Examples
    --------
    >>> model = ITTMCTSchematic(epsilon=-0.1)  # Fluid state
    >>> model.get_glass_transition_info()
    {'is_glass': False, 'epsilon': -0.1, ...}

    >>> model = ITTMCTSchematic(epsilon=0.05)  # Glass state
    >>> sigma = model.predict(np.logspace(-3, 2, 50), test_mode='flow_curve')
    >>> # Shows yield stress at low rates

    >>> # Use Lorentzian decorrelation for materials with extended yielding
    >>> model = ITTMCTSchematic(epsilon=0.05, decorrelation_form="lorentzian")

    >>> # Use full two-time memory kernel (Fuchs & Cates 2002)
    >>> model = ITTMCTSchematic(epsilon=0.05, memory_form="full")

    >>> # Use microscopic stress with Percus-Yevick S(k)
    >>> model = ITTMCTSchematic(
    ...     epsilon=0.05,
    ...     stress_form="microscopic",
    ...     phi_volume=0.5,
    ...     k_BT=4.11e-21,  # Room temperature
    ... )
    """

    def __init__(
        self,
        epsilon: Optional[float] = None,
        v2: Optional[float] = None,
        integration_method: Literal["volterra", "history"] = "volterra",
        n_prony_modes: int = 10,
        decorrelation_form: Literal["gaussian", "lorentzian"] = "gaussian",
        memory_form: Literal["simplified", "full"] = "simplified",
        stress_form: Literal["schematic", "microscopic"] = "schematic",
        phi_volume: Optional[float] = None,
        k_BT: float = 1.0,
    ):
        """Initialize F₁₂ Schematic Model.

        Parameters
        ----------
        epsilon : float, optional
            Separation parameter ε = (v₂ - v₂_c)/v₂_c.
            Mutually exclusive with v2.
        v2 : float, optional
            Direct vertex coefficient. Mutually exclusive with epsilon.
        integration_method : str, default "volterra"
            Integration method for memory kernel
        n_prony_modes : int, default 10
            Number of Prony modes
        decorrelation_form : {"gaussian", "lorentzian"}, default "gaussian"
            Form of the strain decorrelation function h(γ):
            - "gaussian": h(γ) = exp(-(γ/γ_c)²) - faster decay (default, Fuchs & Cates 2002)
            - "lorentzian": h(γ) = 1/(1+(γ/γ_c)²) - slower algebraic decay (Brader et al. 2008)
        memory_form : {"simplified", "full"}, default "simplified"
            Memory kernel form:
            - "simplified": single decorrelation m(Φ) = h[γ_acc] × (v₁Φ + v₂Φ²)
            - "full": two-time decorrelation m(t,s,t₀) = h[γ(t,t₀)] × h[γ(t,s)] × (v₁Φ + v₂Φ²)
        stress_form : {"schematic", "microscopic"}, default "schematic"
            Stress computation form:
            - "schematic": σ = G_∞ × γ̇ × ∫ Φ² × h(γ) dt (standard schematic)
            - "microscopic": σ = (k_BT/60π²) × ∫dk k⁴ [S'/S²]² Φ² (structure factor weighted)
        phi_volume : float, optional
            Volume fraction for Percus-Yevick S(k). Required if stress_form="microscopic".
        k_BT : float, default 1.0
            Thermal energy k_B × T in Joules. Default 1.0 gives dimensionless stress.
            Use 4.11e-21 J for T=298K with real units.
        """
        # Store initialization parameters before parent __init__
        self._init_epsilon = epsilon
        self._init_v2 = v2

        # Validate decorrelation form
        if decorrelation_form not in ("gaussian", "lorentzian"):
            raise ValueError(
                f"decorrelation_form must be 'gaussian' or 'lorentzian', got {decorrelation_form!r}"
            )
        self._use_lorentzian = decorrelation_form == "lorentzian"
        self._decorrelation_form = decorrelation_form

        # Validate memory form
        if memory_form not in ("simplified", "full"):
            raise ValueError(
                f"memory_form must be 'simplified' or 'full', got {memory_form!r}"
            )
        self._memory_form = memory_form

        # Validate stress form
        if stress_form not in ("schematic", "microscopic"):
            raise ValueError(
                f"stress_form must be 'schematic' or 'microscopic', got {stress_form!r}"
            )
        if stress_form == "microscopic" and phi_volume is None:
            raise ValueError(
                "phi_volume is required when stress_form='microscopic'"
            )
        self._stress_form = stress_form
        self._phi_volume = phi_volume
        self._k_BT = k_BT

        # Pre-compute microscopic stress prefactor if needed
        self._microscopic_stress_prefactor = None
        if stress_form == "microscopic":
            from rheojax.utils.mct_kernels import get_microscopic_stress_prefactor
            self._microscopic_stress_prefactor = get_microscopic_stress_prefactor(
                phi_volume, k_BT=k_BT
            )

        super().__init__(
            integration_method=integration_method,
            n_prony_modes=n_prony_modes,
        )

        # Set v2 from epsilon or direct value
        if epsilon is not None and v2 is not None:
            raise ValueError("Specify either epsilon or v2, not both")

        v1 = self.parameters.get_value("v1")
        v2_critical = self._get_v2_critical(v1)

        if epsilon is not None:
            v2_value = v2_critical * (1 + epsilon)
            self.parameters.set_value("v2", v2_value)
        elif v2 is not None:
            self.parameters.set_value("v2", v2)

    def _setup_parameters(self) -> None:
        """Initialize F₁₂ model parameters."""
        self.parameters = ParameterSet()

        # Vertex coefficients
        self.parameters.add(
            name="v1",
            value=0.0,
            bounds=(0.0, 5.0),
            units="-",
            description="Linear vertex coefficient (typically 0 for F₁₂)",
        )

        self.parameters.add(
            name="v2",
            value=2.0,  # Default: fluid state
            bounds=(0.5, 10.0),
            units="-",
            description="Quadratic vertex coefficient (glass at v₂ > 4)",
        )

        # Dynamics
        self.parameters.add(
            name="Gamma",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="1/s",
            description="Bare relaxation rate",
        )

        # Strain decorrelation
        self.parameters.add(
            name="gamma_c",
            value=0.1,
            bounds=(0.01, 0.5),
            units="-",
            description="Critical strain for cage breaking",
        )

        # Modulus
        self.parameters.add(
            name="G_inf",
            value=1e6,
            bounds=(1.0, 1e12),
            units="Pa",
            description="High-frequency elastic modulus",
        )

    def _get_v2_critical(self, v1: float) -> float:
        """Get critical v₂ value for glass transition.

        Parameters
        ----------
        v1 : float
            Linear vertex coefficient

        Returns
        -------
        float
            Critical v₂ value
        """
        # For F₁₂ with v₁ = 0: v₂_c = 4
        if abs(v1) < 1e-10:
            return 4.0
        else:
            # Approximate for non-zero v₁
            return (4.0 - 2.0 * v1) / (1.0 - v1 / 4.0) if v1 < 4.0 else 4.0

    def _compute_equilibrium_correlator(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute equilibrium (quiescent) correlator Φ_eq(t).

        Solves the MCT equation without shear:
        ∂Φ/∂t + Γ[Φ + ∫₀^t m(Φ) ∂Φ/∂s ds] = 0

        Parameters
        ----------
        t : jnp.ndarray
            Time array

        Returns
        -------
        jnp.ndarray
            Equilibrium correlator Φ_eq(t)
        """
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")

        t_np = np.array(t)
        t_max = t_np.max()

        # Get or initialize Prony modes
        if self._prony_amplitudes is None:
            # Use simple exponential decay for initial Prony estimate
            tau_modes = np.logspace(-3, np.log10(t_max), self.n_prony_modes)
            g_modes = np.ones(self.n_prony_modes) / self.n_prony_modes
            self._prony_amplitudes = g_modes
            self._prony_times = tau_modes

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Initial state: [Φ, K₁, K₂, ..., Kₙ]
        state0 = np.zeros(1 + self.n_prony_modes)
        state0[0] = 1.0  # Φ(0) = 1

        def rhs_numpy(t_val, state):
            """Numpy wrapper for ODE solver."""
            state_jax = jnp.array(state)
            deriv = f12_equilibrium_correlator_rhs(
                state_jax, t_val, v1, v2, Gamma, g, tau, self.n_prony_modes
            )
            return np.array(deriv)

        # Solve ODE
        sol = solve_ivp(
            rhs_numpy,
            [0, t_max],
            state0,
            t_eval=t_np,
            method="RK45",
            rtol=1e-8,
            atol=1e-10,
        )

        # Extract correlator
        phi_eq = jnp.array(sol.y[0, :])

        # Ensure physical bounds
        phi_eq = jnp.clip(phi_eq, 0.0, 1.0)

        return phi_eq

    def _compute_memory_kernel(
        self,
        phi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute memory kernel m(Φ) = v₁Φ + v₂Φ².

        Parameters
        ----------
        phi : jnp.ndarray
            Correlator values

        Returns
        -------
        jnp.ndarray
            Memory kernel values
        """
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        return f12_memory(phi, v1, v2)

    def get_glass_transition_info(self) -> Dict[str, Any]:
        """Get information about the glass transition state.

        Returns
        -------
        dict
            Glass transition properties:
            - is_glass: bool
            - epsilon: separation parameter
            - v2_critical: critical v₂ value
            - f_neq: non-ergodicity parameter
            - lambda_exponent: MCT exponent parameter
        """
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        return glass_transition_criterion(v1, v2)

    @property
    def epsilon(self) -> float:
        """Get separation parameter ε = (v₂ - v₂_c)/v₂_c."""
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        v2_c = self._get_v2_critical(v1)
        return (v2 - v2_c) / v2_c

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        """Set separation parameter and update v₂ accordingly."""
        v1 = self.parameters.get_value("v1")
        v2_c = self._get_v2_critical(v1)
        v2_new = v2_c * (1 + value)
        self.parameters.set_value("v2", v2_new)

    # =========================================================================
    # Protocol Implementations
    # =========================================================================

    def _predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        use_diffrax: Optional[bool] = None,
        **kwargs,
    ) -> np.ndarray:
        """Predict steady-state flow curve σ(γ̇).

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        use_diffrax : bool, optional
            Force use of diffrax (True) or scipy (False).
            If None (default), uses diffrax when available.

        Returns
        -------
        np.ndarray
            Steady-state stress σ (Pa)
        """
        gamma_dot = np.asarray(gamma_dot)

        # Get parameters
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        # Initialize Prony modes if needed
        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = self._prony_amplitudes
        tau = self._prony_times

        # Determine which solver to use
        should_use_diffrax = use_diffrax if use_diffrax is not None else _HAS_DIFFRAX

        if should_use_diffrax and _HAS_DIFFRAX:
            return self._predict_flow_curve_diffrax(
                gamma_dot, v1, v2, Gamma, gamma_c, G_inf, g, tau
            )
        else:
            return self._predict_flow_curve_scipy(
                gamma_dot, v1, v2, Gamma, gamma_c, G_inf, g, tau
            )

    def _predict_flow_curve_diffrax(
        self,
        gamma_dot: np.ndarray,
        v1: float,
        v2: float,
        Gamma: float,
        gamma_c: float,
        G_inf: float,
        g: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Fast flow curve prediction using diffrax + vmap.

        First call triggers JIT compilation (~5-10s), subsequent calls
        are very fast (<0.5s for 50 points).
        """
        # Handle zero shear rates separately (yield stress)
        mask_zero = gamma_dot < 1e-15
        mask_nonzero = ~mask_zero

        # Use microscopic prefactor if stress_form is microscopic
        G_eff = G_inf
        if self._stress_form == "microscopic" and self._microscopic_stress_prefactor is not None:
            G_eff = self._microscopic_stress_prefactor

        sigma = np.zeros_like(gamma_dot)

        # Zero shear rate: yield stress if glass
        if np.any(mask_zero):
            info = self.get_glass_transition_info()
            if info["is_glass"]:
                f_neq = info["f_neq"]
                sigma[mask_zero] = G_eff * gamma_c * f_neq
            # else: sigma stays 0

        # Non-zero shear rates: batched diffrax solve
        if np.any(mask_nonzero):
            gamma_dot_nonzero = gamma_dot[mask_nonzero]

            # Call batched diffrax solver with memory_form
            sigma_nonzero = solve_flow_curve_batch(
                jnp.asarray(gamma_dot_nonzero),
                v1,
                v2,
                Gamma,
                gamma_c,
                G_eff,  # Use effective modulus (G_inf or microscopic)
                jnp.asarray(g),
                jnp.asarray(tau),
                self.n_prony_modes,
                self._use_lorentzian,
                self._memory_form,
            )

            # Add yield stress contribution for glass
            info = self.get_glass_transition_info()
            if info["is_glass"]:
                f_neq = info["f_neq"]
                # Approximate: yield stress diminishes with shear
                gamma_eff = gamma_dot_nonzero / Gamma
                # Use model's decorrelation form
                if self._use_lorentzian:
                    h_gamma = 1.0 / (1.0 + (gamma_eff / gamma_c) ** 2)
                else:
                    h_gamma = np.exp(-(gamma_eff / gamma_c) ** 2)
                sigma_y = G_eff * gamma_c * f_neq * (1 - h_gamma)
                sigma_nonzero = np.asarray(sigma_nonzero) + sigma_y

            sigma[mask_nonzero] = np.asarray(sigma_nonzero)

        return sigma

    def _predict_flow_curve_scipy(
        self,
        gamma_dot: np.ndarray,
        v1: float,
        v2: float,
        Gamma: float,
        gamma_c: float,
        G_inf: float,
        g: np.ndarray,
        tau: np.ndarray,
    ) -> np.ndarray:
        """Slow flow curve prediction using scipy (fallback).

        Warning: This is ~100x slower than diffrax version.
        """
        # Use microscopic prefactor if stress_form is microscopic
        G_eff = G_inf
        if self._stress_form == "microscopic" and self._microscopic_stress_prefactor is not None:
            G_eff = self._microscopic_stress_prefactor

        sigma = np.zeros_like(gamma_dot)

        for i, gd in enumerate(gamma_dot):
            if gd < 1e-15:
                # Zero shear rate: yield stress if glass, zero if fluid
                info = self.get_glass_transition_info()
                if info["is_glass"]:
                    # Approximate yield stress
                    f_neq = info["f_neq"]
                    sigma[i] = G_eff * gamma_c * f_neq
                else:
                    sigma[i] = 0.0
            else:
                # Integrate to steady state
                sigma[i] = self._compute_steady_state_stress(gd)

        return sigma

    def _compute_steady_state_stress(
        self,
        gamma_dot: float,
        t_max: Optional[float] = None,
    ) -> float:
        """Compute steady-state stress at a single shear rate.

        The stress is computed as the time integral:
            σ = G_eff * γ̇ * ∫₀^∞ Φ(t) * h(γ(t)) dt

        Parameters
        ----------
        gamma_dot : float
            Shear rate
        t_max : float, optional
            Maximum integration time. If None, uses adaptive time.

        Returns
        -------
        float
            Steady-state stress
        """
        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        # Use microscopic prefactor if stress_form is microscopic
        G_eff = G_inf
        if self._stress_form == "microscopic" and self._microscopic_stress_prefactor is not None:
            G_eff = self._microscopic_stress_prefactor

        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Adaptive integration time
        if t_max is None:
            tau_bare = 1.0 / Gamma
            tau_shear = gamma_c / max(gamma_dot, 1e-10)
            tau_eff = min(tau_bare, tau_shear)
            t_max = 50.0 * tau_eff
            t_max = max(10.0, min(t_max, 500.0))

        # Initial state: [Φ, K₁..Kₙ, γ, σ_integral]
        state0 = np.zeros(3 + self.n_prony_modes)
        state0[0] = 1.0  # Φ(0) = 1
        # γ(0) = 0, σ_integral(0) = 0

        use_full_memory = self._memory_form == "full"

        def rhs_numpy(t_val, state):
            # Extract state
            phi = state[0]
            K = state[1 : 1 + self.n_prony_modes]
            gamma_acc = state[1 + self.n_prony_modes]

            # Strain decorrelation (use model's decorrelation form)
            if self._use_lorentzian:
                h_gamma = 1.0 / (1.0 + (gamma_acc / gamma_c) ** 2)
            else:
                h_gamma = np.exp(-(gamma_acc / gamma_c) ** 2)
            phi_advected = phi * h_gamma

            # Memory kernel
            m_phi = v1 * phi_advected + v2 * phi_advected * phi_advected

            # Memory integral from Prony modes
            memory_integral = np.sum(K)

            # MCT equation
            dphi_dt = -Gamma * (phi + memory_integral)

            # Prony mode evolution with memory form
            if use_full_memory:
                # Full two-time: mode-specific decorrelation
                gamma_mode = gamma_dot * np.asarray(tau)
                if self._use_lorentzian:
                    h_mode = 1.0 / (1.0 + (gamma_mode / gamma_c) ** 2)
                else:
                    h_mode = np.exp(-(gamma_mode / gamma_c) ** 2)
                dK_dt = -K / np.asarray(tau) + np.asarray(g) * m_phi * h_mode * dphi_dt
            else:
                dK_dt = -K / np.asarray(tau) + np.asarray(g) * m_phi * dphi_dt

            # Strain accumulation
            dgamma_dt = gamma_dot

            # Stress integrand: d(σ_integral)/dt = G_eff * γ̇ * Φ * h(γ)
            dsigma_dt = G_eff * gamma_dot * phi_advected

            return np.concatenate([[dphi_dt], dK_dt, [dgamma_dt, dsigma_dt]])

        # Integrate ODE
        t_span = [0, t_max]
        sol = solve_ivp(rhs_numpy, t_span, state0, method="RK45", rtol=1e-5, atol=1e-7)

        # Extract stress integral from final state
        sigma = sol.y[2 + self.n_prony_modes, -1]

        return float(sigma)

    def _predict_oscillation(
        self,
        omega: np.ndarray,
        return_components: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Predict linear viscoelastic moduli G*(ω).

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency (rad/s)
        return_components : bool, default False
            If True, return (G', G'') as shape (n, 2)

        Returns
        -------
        np.ndarray
            Complex modulus or components
        """
        omega = np.asarray(omega)
        G_inf = self.parameters.get_value("G_inf")

        # Need equilibrium correlator over sufficient time range
        omega_min = omega.min()
        t_max = 100.0 / omega_min  # Cover several periods of slowest frequency
        t = np.logspace(-4, np.log10(t_max), 2000)

        # Compute equilibrium correlator
        phi_eq = np.array(self._compute_equilibrium_correlator(jnp.array(t)))

        # Compute G*(ω) via Fourier transform
        G_prime, G_double_prime = compute_complex_modulus_from_correlator(
            jnp.array(omega),
            jnp.array(t),
            jnp.array(phi_eq),
            G_inf,
        )

        G_prime = np.array(G_prime)
        G_double_prime = np.array(G_double_prime)

        if return_components:
            return np.column_stack([G_prime, G_double_prime])
        else:
            return np.sqrt(G_prime**2 + G_double_prime**2)

    def _predict_startup(
        self,
        t: np.ndarray,
        gamma_dot: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Predict stress growth in startup flow.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float, default 1.0
            Applied shear rate (1/s)

        Returns
        -------
        np.ndarray
            Stress response σ(t) (Pa)
        """
        t = np.asarray(t)

        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Initial state: [Φ, K₁..Kₙ, γ, σ]
        state0 = np.zeros(3 + self.n_prony_modes)
        state0[0] = 1.0  # Φ(0) = 1
        state0[-2] = 0.0  # γ(0) = 0
        state0[-1] = 0.0  # σ(0) = 0

        def rhs_numpy(t_val, state):
            state_jax = jnp.array(state)
            deriv = f12_volterra_startup_rhs(
                state_jax,
                t_val,
                gamma_dot,
                v1,
                v2,
                Gamma,
                gamma_c,
                G_inf,
                g,
                tau,
                self.n_prony_modes,
                self._use_lorentzian,
            )
            return np.array(deriv)

        sol = solve_ivp(
            rhs_numpy,
            [0, t.max()],
            state0,
            t_eval=t,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        # Extract stress
        sigma = sol.y[-1, :]

        return sigma

    def _predict_creep(
        self,
        t: np.ndarray,
        sigma_applied: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Predict creep compliance J(t).

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_applied : float, default 1.0
            Applied stress (Pa)

        Returns
        -------
        np.ndarray
            Creep compliance J(t) = γ(t)/σ₀ (1/Pa)
        """
        t = np.asarray(t)

        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Initial state: [Φ, K₁..Kₙ, γ, γ̇]
        state0 = np.zeros(3 + self.n_prony_modes)
        state0[0] = 1.0  # Φ(0) = 1
        state0[-2] = 0.0  # γ(0) = 0
        state0[-1] = sigma_applied / G_inf  # Initial γ̇ estimate

        def rhs_numpy(t_val, state):
            state_jax = jnp.array(state)
            deriv = f12_volterra_creep_rhs(
                state_jax,
                t_val,
                sigma_applied,
                v1,
                v2,
                Gamma,
                gamma_c,
                G_inf,
                g,
                tau,
                self.n_prony_modes,
                self._use_lorentzian,
            )
            return np.array(deriv)

        sol = solve_ivp(
            rhs_numpy,
            [0, t.max()],
            state0,
            t_eval=t,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        # Extract strain and compute compliance
        gamma = sol.y[-2, :]
        J = gamma / sigma_applied

        return J

    def _predict_relaxation(
        self,
        t: np.ndarray,
        gamma_pre: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        """Predict stress relaxation after flow cessation.

        Parameters
        ----------
        t : np.ndarray
            Time array (s) after stopping
        gamma_pre : float, default 0.01
            Pre-shear strain before relaxation

        Returns
        -------
        np.ndarray
            Relaxing stress σ(t) (Pa)
        """
        t = np.asarray(t)

        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Initial state after pre-shear: [Φ, K₁..Kₙ, σ]
        # Use model's decorrelation form
        if self._use_lorentzian:
            h_gamma = 1.0 / (1.0 + (gamma_pre / gamma_c) ** 2)
        else:
            h_gamma = np.exp(-(gamma_pre / gamma_c) ** 2)
        state0 = np.zeros(2 + self.n_prony_modes)
        state0[0] = h_gamma  # Φ affected by pre-shear
        state0[-1] = G_inf * gamma_pre * h_gamma  # Initial stress

        def rhs_numpy(t_val, state):
            state_jax = jnp.array(state)
            deriv = f12_volterra_relaxation_rhs(
                state_jax,
                t_val,
                gamma_pre,
                v1,
                v2,
                Gamma,
                gamma_c,
                G_inf,
                g,
                tau,
                self.n_prony_modes,
                self._use_lorentzian,
            )
            return np.array(deriv)

        sol = solve_ivp(
            rhs_numpy,
            [0, t.max()],
            state0,
            t_eval=t,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        # Extract stress
        sigma = sol.y[-1, :]

        # Add residual stress for glass state
        info = self.get_glass_transition_info()
        if info["is_glass"]:
            f_neq = info["f_neq"]
            sigma_residual = G_inf * gamma_pre * f_neq * h_gamma
            sigma = np.maximum(sigma, sigma_residual)

        return sigma

    def _predict_laos(
        self,
        t: np.ndarray,
        gamma_0: float = 0.1,
        omega: float = 1.0,
        **kwargs,
    ) -> np.ndarray:
        """Predict LAOS stress response.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_0 : float, default 0.1
            Strain amplitude
        omega : float, default 1.0
            Angular frequency (rad/s)

        Returns
        -------
        np.ndarray
            Stress response σ(t) (Pa)
        """
        t = np.asarray(t)

        v1 = self.parameters.get_value("v1")
        v2 = self.parameters.get_value("v2")
        Gamma = self.parameters.get_value("Gamma")
        gamma_c = self.parameters.get_value("gamma_c")
        G_inf = self.parameters.get_value("G_inf")

        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        g = jnp.array(self._prony_amplitudes)
        tau = jnp.array(self._prony_times)

        # Initial state: [Φ, K₁..Kₙ, γ_acc, σ]
        state0 = np.zeros(3 + self.n_prony_modes)
        state0[0] = 1.0  # Φ(0) = 1
        state0[-2] = 0.0  # γ_acc(0) = 0
        state0[-1] = 0.0  # σ(0) = 0

        def rhs_numpy(t_val, state):
            state_jax = jnp.array(state)
            deriv = f12_volterra_laos_rhs(
                state_jax,
                t_val,
                gamma_0,
                omega,
                v1,
                v2,
                Gamma,
                gamma_c,
                G_inf,
                g,
                tau,
                self.n_prony_modes,
                self._use_lorentzian,
            )
            return np.array(deriv)

        sol = solve_ivp(
            rhs_numpy,
            [0, t.max()],
            state0,
            t_eval=t,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )

        # Extract stress
        sigma = sol.y[-1, :]

        return sigma

    def get_laos_harmonics(
        self,
        t: np.ndarray,
        gamma_0: float = 0.1,
        omega: float = 1.0,
        n_harmonics: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Fourier harmonics from LAOS response.

        Parameters
        ----------
        t : np.ndarray
            Time array covering at least one full period
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency
        n_harmonics : int, default 5
            Number of odd harmonics to extract

        Returns
        -------
        sigma_prime_n : np.ndarray
            In-phase coefficients [σ'₁, σ'₃, σ'₅, ...]
        sigma_double_prime_n : np.ndarray
            Out-of-phase coefficients [σ''₁, σ''₃, σ''₅, ...]
        """
        # Compute LAOS response
        sigma = self._predict_laos(t, gamma_0=gamma_0, omega=omega)

        # Extract harmonics
        sigma_prime, sigma_double_prime = extract_laos_harmonics(
            jnp.array(t),
            jnp.array(sigma),
            omega,
            n_harmonics=n_harmonics,
        )

        return np.array(sigma_prime), np.array(sigma_double_prime)

    def model_function(
        self,
        X: np.ndarray,
        v1: float,
        v2: float,
        Gamma: float,
        gamma_c: float,
        G_inf: float,
        test_mode: str = "flow_curve",
        **kwargs,
    ) -> np.ndarray:
        """Static model function for Bayesian inference.

        This function allows the model to be used with NumPyro sampling
        by accepting parameters as arguments.

        Parameters
        ----------
        X : np.ndarray
            Independent variable
        v1, v2, Gamma, gamma_c, G_inf : float
            Model parameters
        test_mode : str, default "flow_curve"
            Protocol type
        **kwargs
            Additional protocol-specific parameters

        Returns
        -------
        np.ndarray
            Model predictions
        """
        # Temporarily set parameters
        old_values = {
            "v1": self.parameters.get_value("v1"),
            "v2": self.parameters.get_value("v2"),
            "Gamma": self.parameters.get_value("Gamma"),
            "gamma_c": self.parameters.get_value("gamma_c"),
            "G_inf": self.parameters.get_value("G_inf"),
        }

        self.parameters.set_value("v1", v1)
        self.parameters.set_value("v2", v2)
        self.parameters.set_value("Gamma", Gamma)
        self.parameters.set_value("gamma_c", gamma_c)
        self.parameters.set_value("G_inf", G_inf)

        # Reset Prony modes for new parameters
        self._prony_amplitudes = None
        self._prony_times = None

        try:
            result = self._predict(X, test_mode=test_mode, **kwargs)
        finally:
            # Restore original values
            for name, value in old_values.items():
                self.parameters.set_value(name, value)
            self._prony_amplitudes = None
            self._prony_times = None

        return result

    def precompile(self) -> float:
        """Pre-compile the diffrax ODE solver for fast subsequent calls.

        Triggers JIT compilation with dummy data so the first real prediction
        doesn't incur the compilation cost. Useful when predictable timing
        is important (e.g., in interactive applications or benchmarks).

        Returns
        -------
        float
            Compilation time in seconds (0.0 if diffrax not available)

        Examples
        --------
        >>> model = ITTMCTSchematic(epsilon=0.05)
        >>> compile_time = model.precompile()
        >>> print(f"Compilation took {compile_time:.1f}s")
        >>> # Now flow curve predictions will be fast
        >>> sigma = model.predict(gamma_dot, test_mode='flow_curve')

        Notes
        -----
        First call to flow curve prediction triggers JIT compilation which
        can take 30-90 seconds. This method triggers that compilation upfront.

        Only affects diffrax-based flow curve solver. Other protocols
        (oscillation, startup, etc.) use scipy and don't need precompilation.
        """
        if not _HAS_DIFFRAX:
            logger.warning("diffrax not available, precompilation skipped")
            return 0.0

        # Initialize Prony modes if needed
        if self._prony_amplitudes is None:
            self.initialize_prony_modes()

        return precompile_flow_curve_solver(
            n_modes=self.n_prony_modes,
            use_lorentzian=self._use_lorentzian,
            memory_form=self._memory_form,
        )

    @property
    def decorrelation_form(self) -> str:
        """Get the strain decorrelation function form."""
        return self._decorrelation_form

    @property
    def memory_form(self) -> str:
        """Get the memory kernel form.

        Returns
        -------
        str
            "simplified" for single decorrelation m(Φ) = h[γ_acc] × (v₁Φ + v₂Φ²)
            "full" for two-time decorrelation m(t,s,t₀) = h[γ(t,t₀)] × h[γ(t,s)] × (v₁Φ + v₂Φ²)
        """
        return self._memory_form

    @property
    def stress_form(self) -> str:
        """Get the stress computation form.

        Returns
        -------
        str
            "schematic" for σ = G_∞ × γ̇ × ∫ Φ² × h(γ) dt
            "microscopic" for σ = (k_BT/60π²) × ∫dk k⁴ [S'/S²]² Φ²
        """
        return self._stress_form

    def __repr__(self) -> str:
        """Return string representation."""
        info = self.get_glass_transition_info()
        state = "glass" if info["is_glass"] else "fluid"
        return (
            f"ITTMCTSchematic("
            f"ε={info['epsilon']:.3f} [{state}], "
            f"v₂={self.parameters.get_value('v2'):.2f}, "
            f"h(γ)={self._decorrelation_form}, "
            f"m={self._memory_form}, "
            f"σ={self._stress_form}, "
            f"G_inf={self.parameters.get_value('G_inf'):.2e} Pa)"
        )
