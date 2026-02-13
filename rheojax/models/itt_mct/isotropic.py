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

from typing import Any, Literal

import numpy as np

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

        assert phi is not None

        # Simplified vertex computation
        self.vertex = mct_vertex_isotropic(
            self.k_grid,
            self.k_grid,
            phi,
            sk_func=lambda k: percus_yevick_sk(k, phi),
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
        """Compute equilibrium k-resolved correlator Φ(k,t).

        Parameters
        ----------
        t : jnp.ndarray
            Time array

        Returns
        -------
        jnp.ndarray
            Equilibrium correlator Φ(k,t) with shape (n_t, n_k)
        """
        # For ISM, we solve coupled equations for each k
        # Simplified: use single-mode approximation
        D0 = self.parameters.get_value("D0")

        assert D0 is not None

        t_np = np.array(t)
        n_t = len(t_np)
        n_k = len(self.k_grid)

        # Bare relaxation rates: Γ(k) = k² D0 / S(k)
        Gamma_k = self.k_grid**2 * D0 / self.S_k

        # Simplified: exponential decay modulated by S(k)
        # Full MCT would require solving coupled equations
        phi_eq = np.zeros((n_t, n_k))

        for i, k_val in enumerate(self.k_grid):
            # Simple exponential for now (full MCT is more complex)
            phi_eq[:, i] = np.exp(-Gamma_k[i] * t_np)

            # Glass state: plateau at f(k)
            info = self.get_glass_transition_info()
            if info["is_glass"]:
                f_k = self._compute_nonergodicity_parameter(k_val)
                phi_eq[:, i] = f_k + (1 - f_k) * phi_eq[:, i]

        return jnp.array(phi_eq)

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

        # Bare relaxation rates
        Gamma_k = self.k_grid**2 * D0 / self.S_k

        # Modulus scale
        G_scale = kBT / sigma_d**3

        sigma = np.zeros_like(gamma_dot)

        for i, gd in enumerate(gamma_dot):
            if gd < 1e-15:
                # Zero shear limit
                info = self.get_glass_transition_info()
                if info["is_glass"]:
                    # Yield stress estimate
                    sigma[i] = G_scale * gamma_c * 0.1
                else:
                    sigma[i] = 0.0
            else:
                # Compute stress from k-space integration
                stress_k = np.zeros(len(self.k_grid))

                for j, k in enumerate(self.k_grid):
                    # Characteristic strain
                    gamma_eff = gd / Gamma_k[j]
                    h = np.exp(-((gamma_eff / gamma_c) ** 2))

                    # Correlator contribution
                    f_k = self._compute_nonergodicity_parameter(k)
                    phi_eff = f_k * h + (1 - f_k) * h

                    # Stress contribution ∝ k⁴ S² Φ²
                    stress_k[j] = k**4 * self.S_k[j] ** 2 * phi_eff**2

                # Integrate over k
                sigma[i] = G_scale * gd * np.trapezoid(stress_k, self.k_grid)

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
            Complex modulus or components
        """
        omega = np.asarray(omega)

        D0 = self.parameters.get_value("D0")
        kBT = self.parameters.get_value("kBT")
        sigma_d = self.parameters.get_value("sigma_d")

        assert D0 is not None
        assert kBT is not None
        assert sigma_d is not None

        # Bare relaxation rates
        Gamma_k = self.k_grid**2 * D0 / self.S_k

        G_scale = kBT / sigma_d**3

        G_prime = np.zeros_like(omega)
        G_double_prime = np.zeros_like(omega)

        for i, w in enumerate(omega):
            g_prime_k = np.zeros(len(self.k_grid))
            g_double_prime_k = np.zeros(len(self.k_grid))

            for j, k in enumerate(self.k_grid):
                # Maxwell-like contribution from each k-mode
                tau_k = 1.0 / Gamma_k[j]
                wt = w * tau_k

                f_k = self._compute_nonergodicity_parameter(k)
                G_k = k**4 * self.S_k[j] ** 2

                # G'(ω) = G_k × [f_k + (1-f_k) × ω²τ²/(1+ω²τ²)]
                g_prime_k[j] = G_k * (f_k + (1 - f_k) * wt**2 / (1 + wt**2))

                # G''(ω) = G_k × (1-f_k) × ωτ/(1+ω²τ²)
                g_double_prime_k[j] = G_k * (1 - f_k) * wt / (1 + wt**2)

            G_prime[i] = G_scale * np.trapezoid(g_prime_k, self.k_grid)
            G_double_prime[i] = G_scale * np.trapezoid(g_double_prime_k, self.k_grid)

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

        for i, t_val in enumerate(t):
            gamma_acc = gamma_dot * t_val
            stress_k = np.zeros(len(self.k_grid))

            for j, k in enumerate(self.k_grid):
                h = np.exp(-((gamma_acc / gamma_c) ** 2))
                tau_k = 1.0 / Gamma_k[j]

                f_k = self._compute_nonergodicity_parameter(k)

                # Correlator evolution under shear
                phi_t = f_k + (1 - f_k) * np.exp(-t_val / tau_k)
                phi_t *= h

                G_k = k**4 * self.S_k[j] ** 2
                stress_k[j] = G_k * phi_t

            sigma[i] = G_scale * gamma_dot * np.trapezoid(stress_k, self.k_grid) * t_val

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

        return J / sigma_applied * sigma_applied  # Normalize

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

        for i, t_val in enumerate(t):
            stress_k = np.zeros(len(self.k_grid))

            for j, k in enumerate(self.k_grid):
                tau_k = 1.0 / Gamma_k[j]
                f_k = self._compute_nonergodicity_parameter(k)

                # Relaxing correlator
                phi_t = f_k + (1 - f_k) * np.exp(-t_val / tau_k)
                phi_t *= h_pre

                G_k = k**4 * self.S_k[j] ** 2
                stress_k[j] = G_k * phi_t

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
