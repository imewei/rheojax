"""ITT-MCT Isotropically Sheared Model (ISM).

The ISM is the full Mode-Coupling Theory with k-resolved correlators
and explicit structure factor S(k) dependence. It provides quantitative
predictions for dense colloidal suspensions.

Key differences from F₁₂ schematic:
- k-resolved correlator Φ(k,t) for each wave vector
- MCT vertex V(k,q) computed from S(k)
- More parameters but more quantitative predictions
- Requires structure factor input (Percus-Yevick or experimental)

Parameters
----------
phi : float
    Volume fraction (0.1 to 0.64 for hard spheres)
sigma_d : float
    Particle diameter (m)
D0 : float
    Bare diffusion coefficient (m²/s)
kBT : float
    Thermal energy (J)
n_k : int
    Number of k-grid points

References
----------
Fuchs M. & Cates M.E. (2009) J. Rheol. 53, 957
Brader J.M. et al. (2009) Proc. Natl. Acad. Sci. 106, 15186
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import scipy.integrate
from scipy.optimize import nnls

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger
from rheojax.models.itt_mct._base import ITTMCTBase
from rheojax.models.itt_mct._kernels import extract_laos_harmonics
from rheojax.utils.structure_factor import (
    hard_sphere_properties,
    interpolate_sk,
    mct_vertex_isotropic,
    percus_yevick_sk,
)

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


@ModelRegistry.register(
    "itt_mct_isotropic",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class ITTMCTIsotropic(ITTMCTBase):
    """ITT-MCT Isotropically Sheared Model with k-resolved correlators.

    The ISM computes density correlators Φ(k,t) for an array of wave vectors,
    using the static structure factor S(k) to compute the MCT memory kernel.

    The model can use:
    - Built-in Percus-Yevick S(k) for hard spheres (default)
    - User-provided S(k) data

    Parameters
    ----------
    phi : float, optional
        Volume fraction. If provided with Percus-Yevick, determines S(k).
    sk_source : {"percus_yevick", "user_provided"}, default "percus_yevick"
        Source of structure factor data
    k_data : np.ndarray, optional
        Wave vectors for user-provided S(k)
    sk_data : np.ndarray, optional
        Structure factor values for user-provided S(k)
    n_k : int, default 100
        Number of k-grid points
    integration_method : str, default "volterra"
        Integration method for memory kernel

    Attributes
    ----------
    k_grid : np.ndarray
        Wave vector array (1/m or dimensionless)
    S_k : np.ndarray
        Structure factor at k_grid points
    vertex : np.ndarray
        MCT vertex matrix V(k,q)

    Examples
    --------
    >>> # Using Percus-Yevick for hard spheres
    >>> model = ITTMCTIsotropic(phi=0.55)
    >>> model.get_glass_transition_info()
    {'is_glass': True, 'phi': 0.55, 'phi_mct': 0.516, ...}

    >>> # Using user-provided S(k)
    >>> model = ITTMCTIsotropic(
    ...     sk_source="user_provided",
    ...     k_data=k_experimental,
    ...     sk_data=sk_experimental
    ... )
    """

    def __init__(
        self,
        phi: float | None = None,
        sk_source: Literal["percus_yevick", "user_provided"] = "percus_yevick",
        k_data: np.ndarray | None = None,
        sk_data: np.ndarray | None = None,
        n_k: int = 100,
        integration_method: Literal["volterra", "history"] = "volterra",
        n_prony_modes: int = 10,
    ):
        """Initialize ISM model.

        Parameters
        ----------
        phi : float, optional
            Volume fraction for Percus-Yevick S(k)
        sk_source : str, default "percus_yevick"
            Source of structure factor
        k_data, sk_data : np.ndarray, optional
            User-provided structure factor data
        n_k : int, default 100
            Number of k-grid points
        integration_method : str, default "volterra"
            Integration method
        n_prony_modes : int, default 10
            Number of Prony modes
        """
        self._init_phi = phi
        self._sk_source = sk_source
        self._user_k_data = k_data
        self._user_sk_data = sk_data
        self._n_k = n_k

        super().__init__(
            integration_method=integration_method,
            n_prony_modes=n_prony_modes,
        )

        # Set phi if provided
        if phi is not None:
            self.parameters.set_value("phi", phi)

        # Initialize k-grid and S(k)
        self._initialize_structure_factor()

    def _setup_parameters(self) -> None:
        """Initialize ISM parameters."""
        self.parameters = ParameterSet()

        # Volume fraction / density
        self.parameters.add(
            name="phi",
            value=0.55,
            bounds=(0.1, 0.64),
            units="-",
            description="Volume fraction (glass at φ ≈ 0.516 for hard spheres)",
        )

        # Particle properties
        self.parameters.add(
            name="sigma_d",
            value=1e-6,
            bounds=(1e-9, 1e-3),
            units="m",
            description="Particle diameter",
        )

        # Dynamics
        self.parameters.add(
            name="D0",
            value=1e-12,
            bounds=(1e-18, 1e-6),
            units="m²/s",
            description="Bare short-time diffusion coefficient",
        )

        self.parameters.add(
            name="kBT",
            value=4.1e-21,  # 300K
            bounds=(1e-24, 1e-18),
            units="J",
            description="Thermal energy k_B T",
        )

        # Strain decorrelation
        self.parameters.add(
            name="gamma_c",
            value=0.1,
            bounds=(0.01, 0.5),
            units="-",
            description="Critical strain for cage breaking",
        )

    def _initialize_structure_factor(self) -> None:
        """Initialize k-grid and compute/interpolate S(k)."""
        phi = self.parameters.get_value("phi")
        sigma_d = self.parameters.get_value("sigma_d")

        assert phi is not None
        assert sigma_d is not None

        # Create k-grid (dimensionless, k*σ)
        # Cover range from 0.1 to 50 in k*σ (peak at ~7)
        self.k_grid = np.linspace(0.1, 50.0, self._n_k) / sigma_d

        if self._sk_source == "percus_yevick":
            self.S_k = percus_yevick_sk(self.k_grid, phi, sigma=sigma_d)
        elif self._sk_source == "user_provided":
            if self._user_k_data is None or self._user_sk_data is None:
                raise ValueError(
                    "Must provide k_data and sk_data for user_provided source"
                )
            self.S_k = interpolate_sk(
                self._user_k_data,
                self._user_sk_data,
                self.k_grid,
            )
        else:
            raise ValueError(f"Unknown sk_source: {self._sk_source}")

        # Compute MCT vertex
        self._compute_vertex()

        logger.debug(
            f"Initialized S(k) with {self._n_k} k-points, "
            f"S_max={self.S_k.max():.2f} at k={self.k_grid[self.S_k.argmax()]:.2f}"
        )

    def _compute_vertex(self) -> None:
        """Compute MCT vertex function V(k,q)."""
        phi = self.parameters.get_value("phi")
        sigma_d = self.parameters.get_value("sigma_d")

        assert phi is not None
        assert sigma_d is not None

        # R10-ISM-004: pass sigma=sigma_d so S(k) in the vertex uses the correct
        # physical particle diameter. Without this, percus_yevick_sk defaults to
        # sigma=1 (dimensionless) while k_grid is in physical units (1/m), causing
        # a dimensional mismatch that shifts the S(k) peak by orders of magnitude.
        self.vertex = mct_vertex_isotropic(
            self.k_grid,
            self.k_grid,
            phi,
            sk_func=lambda k: percus_yevick_sk(k, phi, sigma=sigma_d),
        )

    def update_structure_factor(
        self,
        phi: float | None = None,
        k_data: np.ndarray | None = None,
        sk_data: np.ndarray | None = None,
    ) -> None:
        """Update structure factor (e.g., after parameter change).

        Parameters
        ----------
        phi : float, optional
            New volume fraction (for Percus-Yevick)
        k_data, sk_data : np.ndarray, optional
            New user-provided S(k) data
        """
        if phi is not None:
            self.parameters.set_value("phi", phi)

        if k_data is not None and sk_data is not None:
            self._user_k_data = k_data
            self._user_sk_data = sk_data
            self._sk_source = "user_provided"

        self._initialize_structure_factor()

    def _compute_equilibrium_correlator(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute equilibrium k-resolved correlator Φ(k,t) via Volterra ODE.

        Solves the MCT Volterra integral equation:

            dΦ(k,t)/dt + Γ(k)[Φ(k,t) + ∫₀ᵗ m(k,t-s) · dΦ(k,s)/ds ds] = 0

        with memory kernel m(k,t) = Σ_q V(k,q) · Φ(q,t)² using the
        precomputed vertex matrix self.vertex.

        The Volterra integral is converted to an ODE system via Prony
        decomposition with M auxiliary variables per k-mode:

            dΦ(k)/dt = -Γ(k) × [Φ(k) + Σᵢ Kᵢ(k)]
            m(k)     = Σ_q V(k,q) × Φ(q)²
            dKᵢ(k)/dt = -Kᵢ(k)/τᵢ + gᵢ × m(k) × dΦ(k)/dt

        State vector: [Φ(k₀),...,Φ(k_{N-1}), K₁(k₀),...,K₁(k_{N-1}),
                        ..., K_M(k₀),...,K_M(k_{N-1})]
        Total state size: N_k × (1 + N_prony)

        The ODE is solved in dimensionless units (σ=1, D₀=1, t̃=t·D₀/σ²)
        for numerical stability. The vertex is recomputed in dimensionless
        units to avoid the large magnitudes arising from physical k-units
        (1/m). Results are interpolated back to the requested physical time.

        Parameters
        ----------
        t : jnp.ndarray
            Time array in physical units (s)

        Returns
        -------
        jnp.ndarray
            Equilibrium correlator Φ(k,t) with shape (n_t, n_k)

        Notes
        -----
        The dimensionless vertex is computed from the dimensionless k-grid
        k̃ = k·σ_d, satisfying Ṽ(k̃,q̃) which avoids large physical-unit
        magnitudes. Bare dimensionless relaxation rates are Γ̃(k̃) = k̃²/S(k̃).

        Prony weights (g) are determined by NNLS fitting of a normalised
        exponential proxy for the memory kernel decay. The time constants
        (τ) are log-spaced from 0.1/Γ̃_max to max(2·t_max, 10/Γ̃_min).
        Fallback to equal weights gᵢ = 1/N if NNLS returns a zero solution.

        A fallback to bare-exponential correlators is used if solve_ivp
        fails, preserving backward-compatible qualitative behavior.

        References
        ----------
        Fuchs M. & Cates M.E. (2009) J. Rheol. 53, 957
        Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids"
        """
        D0 = self.parameters.get_value("D0")
        sigma_d = self.parameters.get_value("sigma_d")
        phi = self.parameters.get_value("phi")

        assert D0 is not None
        assert sigma_d is not None
        assert phi is not None

        t_np = np.asarray(t, dtype=np.float64)
        n_k = len(self.k_grid)
        n_prony = self.n_prony_modes

        # ---------------------------------------------------------------
        # Dimensionless units: k̃ = k·σ_d, t̃ = t·D₀/σ_d²
        # Prevents the physical-unit vertex from causing overflow in the
        # ODE solver (physical V entries are O(10¹²) vs dimensionless O(1)).
        # ---------------------------------------------------------------
        k_dim = self.k_grid * sigma_d  # dimensionless wave vectors
        S_k_dim = self.S_k  # S(k) is dimensionless

        # Dimensionless bare relaxation rates Γ̃(k̃) = k̃² / S(k̃)
        Gamma_dim = k_dim**2 / np.maximum(S_k_dim, 1e-12)

        # Dimensionless vertex V̈(k̃,q̃): recomputed with σ=1 so entries are
        # O(1)–O(100), compatible with the ODE step-size constraints.
        def sk_func_dim(k_arg):
            return percus_yevick_sk(k_arg, phi, sigma=1.0)

        V_dim = mct_vertex_isotropic(k_dim, k_dim, phi, sk_func=sk_func_dim, sigma=1.0)

        # ---------------------------------------------------------------
        # Prony parameters: log-spaced τ covering requested time range
        # τ_min = 0.1/Γ̃_max  (shorter than fastest mode)
        # τ_max = max(2·t̃_req, 10/Γ̃_min)  (longer than slowest mode)
        # ---------------------------------------------------------------
        Gamma_min = max(float(Gamma_dim.min()), 1e-10)
        Gamma_max = max(float(Gamma_dim.max()), 1e-10)
        tau_min = max(0.1 / Gamma_max, 1e-10)

        # Physical t_max converted to dimensionless
        t_dim_max = float(np.max(t_np)) * D0 / sigma_d**2
        tau_max = max(t_dim_max * 2.0, 10.0 / Gamma_min)
        tau_max = max(tau_max, tau_min * 1e5)

        tau_arr = np.logspace(np.log10(tau_min), np.log10(tau_max), n_prony)

        # Prony weights via NNLS: fit normalised exponential proxy
        # Proxy: exp(-Γ_eff · t̃) where Γ_eff is the geometric-mean rate
        Gamma_eff = float(np.exp(0.5 * (np.log(Gamma_min) + np.log(Gamma_max))))
        n_fit_pts = max(50, 5 * n_prony)
        t_fit = np.logspace(np.log10(tau_min), np.log10(tau_max * 0.5), n_fit_pts)
        m_proxy = np.exp(-Gamma_eff * t_fit)
        A_fit = np.exp(-t_fit[:, None] / tau_arr[None, :])
        g_raw, _ = nnls(A_fit, m_proxy)
        g_sum = float(g_raw.sum())
        if g_sum < 1e-12:
            g_arr = np.ones(n_prony) / n_prony  # fallback: equal weights
        else:
            g_arr = g_raw / g_sum  # normalise: Σᵢ gᵢ = 1

        logger.debug(
            "ISM correlator: n_k=%d, n_prony=%d, tau=[%.2e, %.2e], g_nnz=%d/%d",
            n_k,
            n_prony,
            tau_arr[0],
            tau_arr[-1],
            int(np.count_nonzero(g_arr > 1e-14 * g_arr.max())),
            n_prony,
        )

        # ---------------------------------------------------------------
        # Volterra ODE system in dimensionless time
        # ---------------------------------------------------------------
        state_size = n_k * (1 + n_prony)

        def _volterra_rhs(t_var: float, y: np.ndarray) -> np.ndarray:
            """ODE right-hand side for k-resolved MCT Volterra equation."""
            phi_k = np.clip(y[:n_k], 0.0, 1.0)

            # Prony auxiliary variables: K[i, k], shape (n_prony, n_k)
            K_mat = y[n_k:].reshape(n_prony, n_k)
            K_sum = K_mat.sum(axis=0)  # (n_k,)

            # Memory kernel: m(k) = Σ_q V(k,q) · Φ(q)²
            m_k = V_dim @ (phi_k * phi_k)  # (n_k,)

            # Correlator evolution: dΦ/dt = -Γ̃(k) × [Φ(k) + Σᵢ Kᵢ(k)]
            dphi_dt = -Gamma_dim * (phi_k + K_sum)  # (n_k,)

            dy = np.empty(state_size)
            dy[:n_k] = dphi_dt

            # Prony mode evolution: dKᵢ/dt = -Kᵢ/τᵢ + gᵢ × m(k) × dΦ/dt
            for i in range(n_prony):
                offset = n_k * (1 + i)
                dy[offset : offset + n_k] = (
                    -K_mat[i] / tau_arr[i] + g_arr[i] * m_k * dphi_dt
                )

            return dy

        # Initial conditions: Φ(k, 0) = 1 (fully correlated), Kᵢ(k, 0) = 0
        y0 = np.zeros(state_size)
        y0[:n_k] = 1.0

        # Integration span in dimensionless time
        t_dim_end = max(t_dim_max * 1.001, tau_max * 1.01)

        # t_eval: convert physical time to dimensionless, restrict to (0, t_dim_end)
        t_eval_dim = t_np * (D0 / sigma_d**2)
        mask_positive = t_eval_dim > 0.0
        t_eval_valid = np.unique(
            np.clip(t_eval_dim[mask_positive], 0.0, t_dim_end * 0.9999)
        )
        if len(t_eval_valid) == 0:
            t_eval_valid = np.array([t_dim_end * 0.5])

        # Solve using Radau (L-stable implicit Runge-Kutta), appropriate for
        # stiff systems such as MCT near the glass transition.
        # Suppress overflow warnings from the numerical Jacobian estimation:
        # Radau uses finite differences to build the Jacobian matrix. For
        # large state values (e.g. the initial memory kernel at t≈0), the
        # finite-difference step can temporarily exceed the float64 range in
        # the "factor increase" heuristic. These overflows are benign —
        # the solver self-corrects on the next step.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sol = scipy.integrate.solve_ivp(
                    _volterra_rhs,
                    (0.0, t_dim_end),
                    y0,
                    method="Radau",
                    t_eval=t_eval_valid,
                    rtol=1e-5,
                    atol=1e-8,
                    dense_output=False,
                )
            solver_ok = sol.success
        except Exception as exc:
            logger.warning(
                "ISM Volterra ODE raised an exception (%s). "
                "Using bare-exponential fallback.",
                exc,
            )
            sol = None
            solver_ok = False

        if not solver_ok:
            msg = (
                getattr(sol, "message", "solver exception")
                if sol is not None
                else "solver exception"
            )
            logger.warning(
                "ISM Volterra solver did not converge (%s). "
                "Falling back to bare-exponential (Brownian) correlator. "
                "Increase n_prony_modes or check vertex for numerical issues.",
                msg,
            )
            # Fallback: bare Brownian dynamics (no MCT memory)
            Gamma_phys = self.k_grid**2 * D0 / np.maximum(S_k_dim, 1e-12)
            phi_fallback = np.exp(
                -t_np[:, None] * Gamma_phys[None, :]
            )  # shape (n_t, n_k)
            return jnp.array(phi_fallback)

        # ---------------------------------------------------------------
        # Reconstruct correlator on the full requested time grid
        # t=0 → Φ(k,0) = 1 by initial condition
        # t>0 → interpolate from ODE solution
        # ---------------------------------------------------------------
        phi_out = np.ones((len(t_np), n_k))

        if len(sol.t) > 0:
            for ik in range(n_k):
                phi_interp = np.interp(
                    t_eval_dim[mask_positive],
                    sol.t,
                    sol.y[ik],
                )
                phi_out[mask_positive, ik] = phi_interp

        phi_out = np.clip(phi_out, 0.0, 1.0)

        return jnp.array(phi_out)

    def _compute_nonergodicity_parameter(self, k: float) -> float:
        """Compute non-ergodicity parameter f(k) for glass state.

        Parameters
        ----------
        k : float
            Wave vector

        Returns
        -------
        float
            Non-ergodicity parameter f(k) ∈ [0, 1]
        """
        phi = self.parameters.get_value("phi")
        sigma_d = self.parameters.get_value("sigma_d")

        assert phi is not None
        assert sigma_d is not None

        # S(k) at this wave vector
        S_k_val = percus_yevick_sk(np.array([k]), phi, sigma=sigma_d)[0]

        # Simplified f(k) estimate from MCT
        # f(k) ≈ 1 - 1/S(k)_max for k near peak
        phi_mct = 0.516
        if phi > phi_mct:
            # Glass: f(k) depends on distance from transition
            epsilon = (phi - phi_mct) / phi_mct
            f_k = 0.3 * S_k_val / self.S_k.max() * min(1.0, epsilon * 10)
        else:
            f_k = 0.0

        return float(f_k)

    def _compute_memory_kernel(
        self,
        phi_k: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute memory kernel from k-resolved correlator.

        Parameters
        ----------
        phi_k : jnp.ndarray
            Correlator values at each k

        Returns
        -------
        jnp.ndarray
            Memory kernel m(k) for each wave vector
        """
        # m(k) = Σ_q V(k,q) Φ(q) Φ(|k-q|)
        # Simplified: use precomputed vertex
        m_k = jnp.dot(self.vertex, phi_k * phi_k)
        return m_k

    def get_glass_transition_info(self) -> dict[str, Any]:
        """Get information about the glass transition state.

        Returns
        -------
        dict
            Glass transition properties including:
            - is_glass: bool
            - phi: current volume fraction
            - phi_mct: MCT glass transition (≈0.516)
            - S_max: peak of S(k)
        """
        phi = self.parameters.get_value("phi")
        assert phi is not None
        properties = hard_sphere_properties(phi)
        return {
            "is_glass": properties["is_glassy"],
            "phi": phi,
            "phi_mct": 0.516,
            "phi_rcp": 0.64,
            "S_max": self.S_k.max(),
            "k_peak": self.k_grid[self.S_k.argmax()],
        }

    def get_sk_info(self) -> dict[str, Any]:
        """Get information about current S(k).

        Returns
        -------
        dict
            S(k) properties
        """
        return {
            "source": self._sk_source,
            "n_k": self._n_k,
            "k_range": (self.k_grid.min(), self.k_grid.max()),
            "S_max": self.S_k.max(),
            "S_max_position": self.k_grid[self.S_k.argmax()],
            "S_0": self.S_k[0],
        }

    # =========================================================================
    # Protocol Implementations
    # =========================================================================

    def _predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Predict steady-state flow curve σ(γ̇).

        For ISM, the stress is computed from k-resolved correlators:
        σ = (kBT/6π²) ∫ dk k⁴ S(k)² [∂lnS/∂lnk]² ∫ dτ Φ(k,τ)² h(γ̇τ)

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        np.ndarray
            Steady-state stress σ (Pa)
        """
        gamma_dot = np.asarray(gamma_dot)

        sigma_d = self.parameters.get_value("sigma_d")
        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        gamma_c = self.parameters.get_value("gamma_c")

        assert sigma_d is not None
        assert D0 is not None
        assert kBT is not None
        assert gamma_c is not None

        # Bare relaxation rates: Γ(k) = k²D₀/S(k)
        Gamma_k = self.k_grid**2 * D0 / self.S_k
        tau_k_arr = 1.0 / np.maximum(Gamma_k, 1e-30)

        # Modulus scale
        G_scale = kBT / sigma_d**3

        # R10-ISM-001: hoist glass check and f(k) once, shared across all γ̇ values.
        info = self.get_glass_transition_info()
        is_glass = info["is_glass"]
        if is_glass:
            f_k_arr = np.array(
                [self._compute_nonergodicity_parameter(k) for k in self.k_grid]
            )
        else:
            f_k_arr = np.zeros(len(self.k_grid))

        sigma = np.zeros_like(gamma_dot)

        # R10-ISM-003: correct ITT-MCT flow curve uses full time integral
        # σ(γ̇) = γ̇ · ∫₀^∞ G(t) · h(γ̇t) dt
        # where G(t) = G_scale · Σ_k k⁴ S(k)² · Φ(k,t)²
        # and h(γ) = exp(-(γ/γ_c)²) is the strain decorrelation.
        # Using a single t_eff = 1/γ̇ was an O(1)-point approximation
        # that misses the entire shape of G(t) and underestimates the
        # integral by orders of magnitude near the glass transition.
        #
        # Use t_max = 100 * tau_k_max to cover full correlator decay.
        # The h(γ̇t) factor suppresses contributions beyond t ~ gamma_c/gamma_dot,
        # so the integral naturally converges even for the glass state.
        t_max_global = float(100.0 * tau_k_arr.max())

        # Linear time grid is more stable for the exponential × Gaussian integrand
        t_int = np.linspace(0.0, t_max_global, 500)

        # Pre-compute G(t) on the shared grid (vectorized over k using broadcasting)
        # phi_k shape: (n_t, n_k)
        exp_decay = np.exp(-t_int[:, None] / tau_k_arr[None, :])  # (n_t, n_k)
        if is_glass:
            phi_k_eq = f_k_arr[None, :] + (1.0 - f_k_arr[None, :]) * exp_decay
        else:
            phi_k_eq = exp_decay  # (n_t, n_k)

        # G_k weights: k⁴ S(k)²
        G_k_weights = self.k_grid**4 * self.S_k**2  # (n_k,)

        # G(t) = G_scale · Σ_k G_k · Φ(k,t)²  shape: (n_t,)
        G_t = G_scale * np.sum(G_k_weights[None, :] * phi_k_eq**2, axis=1)

        for i, gd in enumerate(gamma_dot):
            if gd < 1e-15:
                # Zero shear rate: yield stress for glass, zero for fluid.
                # Estimate: σ_y ≈ G_scale · gamma_c · Σ_k G_k·f_k² / Σ_k G_k (weighted plateau)
                if is_glass:
                    G_k_total = float(np.sum(G_k_weights))
                    if G_k_total > 0:
                        sigma[i] = (
                            G_scale
                            * gamma_c
                            * float(np.sum(G_k_weights * f_k_arr**2))
                            / G_k_total
                            * 0.1
                        )
                else:
                    sigma[i] = 0.0
                continue

            # Strain decorrelation on the time grid: h(γ̇t)
            gamma_t = gd * t_int
            h_t = np.exp(-((gamma_t / gamma_c) ** 2))

            # σ = γ̇ · ∫ G(t) · h(γ̇t) dt
            integrand = G_t * h_t
            sigma[i] = gd * np.trapezoid(integrand, t_int)

        return sigma

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
            Complex modulus G* = G' + iG'' by default.
            If return_components=True, returns (n, 2) array [G', G''].
        """
        omega = np.asarray(omega)

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None

        # Bare relaxation rates: Γ(k) = k²D₀/S(k)
        Gamma_k = self.k_grid**2 * D0 / self.S_k
        tau_k_arr = 1.0 / np.maximum(Gamma_k, 1e-30)

        G_scale = kBT / sigma_d**3

        # R10-ISM-001: hoist glass check once.
        info = self.get_glass_transition_info()
        is_glass = info["is_glass"]
        if is_glass:
            f_k_arr = np.array(
                [self._compute_nonergodicity_parameter(k) for k in self.k_grid]
            )
        else:
            f_k_arr = np.zeros(len(self.k_grid))

        # R10-ISM-004/005: replace the Maxwell approximation
        # G'(ω) = Σ_k G_k·[f_k + (1-f_k)·ω²τ²/(1+ω²τ²)] with the correct
        # Fourier transform of G(t) = G_scale·Σ_k G_k·Φ(k,t)².
        #
        # The Maxwell formula misses the f_k² plateau contribution and the
        # cross-term 2f_k(1-f_k) in Φ². With Φ(k,t) = f_k + (1-f_k)e^{-t/τ_k}:
        # Φ² = f_k² + 2f_k(1-f_k)e^{-t/τ_k} + (1-f_k)²e^{-2t/τ_k}
        # FT gives three contributions: G_e plateau, mode at τ_k, mode at τ_k/2.
        # We compute the Fourier integral numerically on a dense time grid so all
        # contributions (including the elastic plateau from f_k²) are captured.
        #
        # Numerical strategy for each omega: use a per-frequency linear grid that
        # covers the correlator decay (t_max = 100*tau_k_max) with enough points
        # to resolve the oscillation period 2π/ω (at least 20 points per period).
        # This avoids Nyquist aliasing when omega is large.
        G_k_weights = self.k_grid**4 * self.S_k**2  # (n_k,)
        G_prime = np.zeros_like(omega)
        G_double_prime = np.zeros_like(omega)

        for idx, w in enumerate(omega):
            t_max_w = float(100.0 * tau_k_arr.max())
            # Ensure >= 20 points per oscillation period (Nyquist × 10 safety margin)
            T_period = 2.0 * np.pi / w
            n_t_w = max(2000, int(np.ceil(20.0 * t_max_w / T_period)))
            n_t_w = min(n_t_w, 50000)  # Cap to avoid excessive memory use

            t_w = np.linspace(0.0, t_max_w, n_t_w)

            # G(t) = G_scale · Σ_k k⁴ S(k)² · Φ(k,t)²  (vectorized over k)
            exp_decay = np.exp(-t_w[:, None] / tau_k_arr[None, :])
            if is_glass:
                phi_k_w = f_k_arr[None, :] + (1.0 - f_k_arr[None, :]) * exp_decay
            else:
                phi_k_w = exp_decay
            G_t_w = G_scale * np.sum(G_k_weights[None, :] * phi_k_w**2, axis=1)

            # Numerical Fourier transform:
            # G'(ω) = ω ∫₀^∞ G(t) sin(ωt) dt
            # G''(ω) = ω ∫₀^∞ G(t) cos(ωt) dt
            sin_wt = np.sin(w * t_w)
            cos_wt = np.cos(w * t_w)
            G_prime[idx] = w * np.trapezoid(G_t_w * sin_wt, t_w)
            G_double_prime[idx] = w * np.trapezoid(G_t_w * cos_wt, t_w)

        if return_components:
            return np.column_stack([G_prime, G_double_prime])
        return G_prime + 1j * G_double_prime

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

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")
        gamma_c = self.parameters.get_value("gamma_c")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None
        assert gamma_c is not None

        Gamma_k = self.k_grid**2 * D0 / self.S_k
        tau_k_arr = 1.0 / np.maximum(Gamma_k, 1e-30)
        G_scale = kBT / sigma_d**3

        # R10-ISM-001: hoist glass check and f(k) out of any inner loop.
        info = self.get_glass_transition_info()
        is_glass = info["is_glass"]
        if is_glass:
            f_k_arr = np.array(
                [self._compute_nonergodicity_parameter(k) for k in self.k_grid]
            )
        else:
            f_k_arr = np.zeros(len(self.k_grid))

        # R10-ISM-002: replace triple-nested Python loop with vectorized computation.
        # Old code: O(n_t × n_sub × n_k) Python iterations where n_sub varied per t.
        # New code: O(1) numpy calls. We precompute G(t) on a shared fine grid,
        # then integrate σ(t) = γ̇ · ∫₀ᵗ G(s) · h(γ̇s) ds via cumulative trapezoid.
        t_max_val = float(np.max(t))
        n_fine = max(500, 2 * len(t))
        t_fine = np.linspace(0.0, t_max_val, n_fine)

        # phi_k_fine shape: (n_fine, n_k) — vectorized over k via broadcasting
        exp_decay = np.exp(-t_fine[:, None] / tau_k_arr[None, :])
        if is_glass:
            phi_k_fine = f_k_arr[None, :] + (1.0 - f_k_arr[None, :]) * exp_decay
        else:
            phi_k_fine = exp_decay

        # G_k weights: k⁴ S(k)²
        G_k_weights = self.k_grid**4 * self.S_k**2  # (n_k,)

        # G(t_fine) = G_scale · Σ_k G_k · Φ(k,t)²   shape: (n_fine,)
        G_t_fine = G_scale * np.sum(G_k_weights[None, :] * phi_k_fine**2, axis=1)

        # Strain decorrelation on the fine grid
        gamma_t_fine = gamma_dot * t_fine
        h_t_fine = np.exp(-((gamma_t_fine / gamma_c) ** 2))

        # Cumulative integral: σ(t) = γ̇ · ∫₀ᵗ G(s) · h(γ̇s) ds
        integrand_fine = G_t_fine * h_t_fine
        sigma_cumulative = gamma_dot * scipy.integrate.cumulative_trapezoid(
            integrand_fine, t_fine, initial=0.0
        )

        # Interpolate cumulative stress to the requested time points
        sigma = np.interp(t, t_fine, sigma_cumulative)

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
            Creep compliance J(t) (1/Pa)
        """
        t = np.asarray(t)

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None

        Gamma_k = self.k_grid**2 * D0 / self.S_k
        G_scale = kBT / sigma_d**3

        # Simplified creep: J(t) ≈ t/η + 1/G_∞ for fluid
        info = self.get_glass_transition_info()

        if info["is_glass"]:
            # Glass: bounded compliance approaching J_∞
            J_inf = 1.0 / (G_scale * 1e3)  # Approximate
            tau_eff = 1.0 / np.mean(Gamma_k)
            J = J_inf * (1 - np.exp(-t / tau_eff))
        else:
            # Fluid: viscous flow
            eta_eff = G_scale / np.mean(Gamma_k)
            J_0 = 1.0 / G_scale
            J = J_0 + t / eta_eff

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
            Time array (s)
        gamma_pre : float
            Pre-shear strain

        Returns
        -------
        np.ndarray
            Relaxing stress σ(t) (Pa)
        """
        t = np.asarray(t)

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")
        gamma_c = self.parameters.get_value("gamma_c")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None
        assert gamma_c is not None

        Gamma_k = self.k_grid**2 * D0 / self.S_k
        G_scale = kBT / sigma_d**3

        h_pre = np.exp(-((gamma_pre / gamma_c) ** 2))

        sigma = np.zeros_like(t)

        # R11-ISM-001: TODO — Vectorize over (t, k) dimensions using jnp broadcasting
        # for better performance. Current double Python loop is O(n_t * n_k).
        for i, t_val in enumerate(t):
            stress_k = np.zeros(len(self.k_grid))

            for j, k in enumerate(self.k_grid):
                tau_k = 1.0 / Gamma_k[j]
                f_k = self._compute_nonergodicity_parameter(k)

                # Relaxing correlator
                phi_t = f_k + (1 - f_k) * np.exp(-t_val / tau_k)
                phi_t *= h_pre

                G_k = k**4 * self.S_k[j] ** 2
                stress_k[j] = G_k * phi_t * phi_t

            sigma[i] = G_scale * gamma_pre * np.trapezoid(stress_k, self.k_grid)

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
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        np.ndarray
            Stress response σ(t) (Pa)
        """
        t = np.asarray(t)

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")
        gamma_c = self.parameters.get_value("gamma_c")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None
        assert gamma_c is not None

        Gamma_k = self.k_grid**2 * D0 / self.S_k
        G_scale = kBT / sigma_d**3

        sigma = np.zeros_like(t)

        # R11-ISM-001: TODO — Vectorize over (t, k) dimensions using jnp broadcasting
        # for better performance. Current double Python loop is O(n_t * n_k).
        for i, t_val in enumerate(t):
            gamma_t = gamma_0 * np.sin(omega * t_val)
            gamma_dot_t = gamma_0 * omega * np.cos(omega * t_val)

            h = np.exp(-((gamma_t / gamma_c) ** 2))

            stress_k = np.zeros(len(self.k_grid))

            for j, k in enumerate(self.k_grid):
                tau_k = 1.0 / Gamma_k[j]
                f_k = self._compute_nonergodicity_parameter(k)

                # Simplified LAOS response
                wt = omega * tau_k
                G_k = k**4 * self.S_k[j] ** 2

                # In-phase and out-of-phase contributions
                G_prime_k = G_k * (f_k + (1 - f_k) * wt**2 / (1 + wt**2))
                G_double_prime_k = G_k * (1 - f_k) * wt / (1 + wt**2)

                # Nonlinear correction from h(γ)
                stress_k[j] = h * (
                    G_prime_k * gamma_t + G_double_prime_k * gamma_dot_t / omega
                )

            sigma[i] = G_scale * np.trapezoid(stress_k, self.k_grid)

        return sigma

    def get_laos_harmonics(
        self,
        t: np.ndarray,
        gamma_0: float = 0.1,
        omega: float = 1.0,
        n_harmonics: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            In-phase coefficients [sigma'_1, sigma'_3, sigma'_5, ...]
        sigma_double_prime_n : np.ndarray
            Out-of-phase coefficients [sigma''_1, sigma''_3, sigma''_5, ...]
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

    def __repr__(self) -> str:
        """Return string representation."""
        info = self.get_glass_transition_info()
        state = "glass" if info["is_glass"] else "fluid"
        return (
            f"ITTMCTIsotropic("
            f"φ={info['phi']:.3f} [{state}], "
            f"n_k={self._n_k}, "
            f"sk_source='{self._sk_source}')"
        )
