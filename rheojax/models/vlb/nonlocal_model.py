"""Nonlocal VLB model with tensor diffusion for shear banding.

This module implements `VLBNonlocal`, a spatially-resolved (1D) extension
of the VLB framework where the distribution tensor mu varies across the
gap of a Couette geometry.

The PDE governing the distribution tensor is:

    dmu/dt = k_d(I - mu) + L·mu + mu·L^T + D_mu * nabla^2(mu)

where D_mu is the distribution tensor diffusivity (m²/s), a material
constant that sets the cooperativity length xi = sqrt(D_mu / k_d_0).

Shear banding arises when the Bell breakage rate creates a non-monotonic
constitutive curve (S-shaped sigma vs gamma_dot). The nonlocal diffusion
term regularizes the banding interface and sets its width.

Parameters
----------
breakage : str
    "constant" or "bell"
stress_type : str
    "linear" or "fene"
n_points : int
    Spatial grid points across gap (default 51)
gap_width : float
    Gap width in meters (default 1e-3)

References
----------
- Vernerey, F.J., Long, R. & Brighenti, R. (2017). JMPS 107, 1-20.
- Dhont, J.K.G. (1999). PRE 60, 4534.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.vlb._base import VLBBase
from rheojax.models.vlb._kernels import (
    laplacian_1d_neumann_vlb,
    vlb_breakage_bell,
    vlb_stress_fene_xy,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)

BreakageType = Literal["constant", "bell"]
StressType = Literal["linear", "fene"]


@ModelRegistry.register(
    "vlb_nonlocal",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.CREEP,
    ],
    deformation_modes=[DeformationMode.SHEAR],
)
class VLBNonlocal(VLBBase):
    """Nonlocal VLB with tensor diffusion for shear banding.

    Solves a 1D PDE across the gap of a Couette cell. The state at each
    spatial point is (mu_xx, mu_yy, mu_zz, mu_xy), plus a single wall
    stress Sigma (spatially uniform at low Reynolds number).

    Shear banding occurs when the Bell breakage rate creates a non-monotonic
    flow curve. The diffusion term D_mu * nabla^2(mu) regularizes the
    interface with width ~ xi = sqrt(D_mu / k_d_0).

    Parameters
    ----------
    breakage : str, default "constant"
        "constant" or "bell"
    stress_type : str, default "linear"
        "linear" or "fene"
    n_points : int, default 51
        Spatial grid resolution
    gap_width : float, default 1e-3
        Gap width (m)
    """

    def __init__(
        self,
        breakage: BreakageType = "constant",
        stress_type: StressType = "linear",
        n_points: int = 51,
        gap_width: float = 1e-3,
    ):
        """Initialize VLBNonlocal model."""
        self._breakage = breakage
        self._stress_type = stress_type
        self.n_points = n_points
        self.gap_width = gap_width

        super().__init__()
        self._setup_parameters()

        # Spatial grid
        self.y = np.linspace(0, gap_width, n_points)
        self.dy = gap_width / (n_points - 1)

        self._test_mode = None

        logger.info(
            f"VLBNonlocal initialized: breakage={breakage}, "
            f"stress={stress_type}, n_points={n_points}"
        )

    # =========================================================================
    # Parameters
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet for nonlocal model."""
        self.parameters = ParameterSet()

        # Core parameters
        self.parameters.add(
            name="G0", value=1e3, bounds=(1e0, 1e8),
            units="Pa", description="Network modulus",
        )
        self.parameters.add(
            name="k_d_0", value=1.0, bounds=(1e-6, 1e6),
            units="1/s", description="Unstressed dissociation rate",
        )
        self.parameters.add(
            name="eta_s", value=0.0, bounds=(0.0, 1e4),
            units="Pa·s", description="Solvent viscosity",
        )

        # Nonlocal parameter
        self.parameters.add(
            name="D_mu", value=1e-8, bounds=(1e-14, 1e-4),
            units="m²/s", description="Distribution tensor diffusivity",
        )

        # Bell parameter
        if self._breakage == "bell":
            self.parameters.add(
                name="nu", value=3.0, bounds=(0.0, 20.0),
                units="dimensionless",
                description="Force sensitivity (Bell model)",
            )

        # FENE parameter
        if self._stress_type == "fene":
            self.parameters.add(
                name="L_max", value=10.0, bounds=(1.5, 1000.0),
                units="dimensionless",
                description="Maximum chain extensibility (FENE-P spring)",
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def G0(self) -> float:
        val = self.parameters.get_value("G0")
        return float(val) if val is not None else 1e3

    @property
    def k_d_0(self) -> float:
        val = self.parameters.get_value("k_d_0")
        return float(val) if val is not None else 1.0

    def get_cooperativity_length(self) -> float:
        """Cooperativity length xi = sqrt(D_mu / k_d_0).

        This sets the shear band interface width.

        Returns
        -------
        float
            Cooperativity length (m)
        """
        D_mu = float(self.parameters.get_value("D_mu") or 1e-8)
        return np.sqrt(D_mu / self.k_d_0)

    # =========================================================================
    # PDE Integration Core
    # =========================================================================

    def _build_pde_rhs(self):
        """Build PDE RHS function for diffrax integration.

        State: [Sigma, mu_xx[0:N], mu_yy[0:N], mu_zz[0:N], mu_xy[0:N]]
        Total state size: 1 + 4*N
        """
        breakage = self._breakage
        stress_type = self._stress_type
        n = self.n_points  # Closed over as static (required for JAX tracing)

        @jax.jit
        def pde_rhs(t, state, args):
            G0 = args["G0"]
            k_d_0 = args["k_d_0"]
            eta_s = args["eta_s"]
            D_mu = args["D_mu"]
            nu = args["nu"]
            L_max = args["L_max"]
            dy = args["dy"]
            gamma_dot_avg = args["gamma_dot_avg"]

            # Unpack state (n is closed over as static)
            Sigma = state[0]
            mu_xx = state[1:1 + n]
            mu_yy = state[1 + n:1 + 2 * n]
            mu_zz = state[1 + 2 * n:1 + 3 * n]
            mu_xy = state[1 + 3 * n:1 + 4 * n]

            # Local dissociation rate
            if breakage == "bell":
                k_d = jax.vmap(
                    lambda xx, yy, zz: vlb_breakage_bell(xx, yy, zz, k_d_0, nu)
                )(mu_xx, mu_yy, mu_zz)
            else:
                k_d = jnp.full(n, k_d_0)

            # Local elastic stress
            if stress_type == "fene":
                sigma_elastic = jax.vmap(
                    lambda xx, yy, zz, xy: vlb_stress_fene_xy(xx, yy, zz, xy, G0, L_max)
                )(mu_xx, mu_yy, mu_zz, mu_xy)
            else:
                sigma_elastic = G0 * mu_xy

            # Local shear rate from stress balance
            # Sigma = sigma_elastic + eta_s * gamma_dot
            # For eta_s = 0, regularize with small fraction of network viscosity
            eta_eff = jnp.maximum(eta_s, 1e-2 * G0 / jnp.maximum(k_d_0, 1e-30))
            gamma_dot = (Sigma - sigma_elastic) / eta_eff

            # mu evolution (local kinetics)
            dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
            dmu_yy = k_d * (1.0 - mu_yy)
            dmu_zz = k_d * (1.0 - mu_zz)
            dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

            # Add diffusion
            dmu_xx = dmu_xx + D_mu * laplacian_1d_neumann_vlb(mu_xx, dy)
            dmu_yy = dmu_yy + D_mu * laplacian_1d_neumann_vlb(mu_yy, dy)
            dmu_zz = dmu_zz + D_mu * laplacian_1d_neumann_vlb(mu_zz, dy)
            dmu_xy = dmu_xy + D_mu * laplacian_1d_neumann_vlb(mu_xy, dy)

            # Stress feedback: enforce average shear rate = imposed value
            K = 10.0 * G0
            mean_gd = jnp.mean(gamma_dot)
            dSigma = K * (gamma_dot_avg - mean_gd)

            return jnp.concatenate([
                jnp.array([dSigma]),
                dmu_xx, dmu_yy, dmu_zz, dmu_xy,
            ])

        return pde_rhs

    def _build_initial_state(self, perturbation: float = 0.01) -> jnp.ndarray:
        """Build initial state with small perturbation for symmetry breaking.

        Parameters
        ----------
        perturbation : float
            Amplitude of spatial noise (relative)

        Returns
        -------
        jnp.ndarray
            Initial state vector, shape (1 + 4*n_points,)
        """
        n = self.n_points
        # Initial guess for wall stress
        Sigma_0 = self.G0 * 1.0  # Arbitrary initial stress

        # Uniform equilibrium with small noise for symmetry breaking
        key = jax.random.PRNGKey(42)
        noise = perturbation * jax.random.normal(key, shape=(n,))

        mu_xx_0 = jnp.ones(n) + noise
        mu_yy_0 = jnp.ones(n)
        mu_zz_0 = jnp.ones(n)
        mu_xy_0 = jnp.zeros(n)

        return jnp.concatenate([
            jnp.array([Sigma_0]),
            mu_xx_0, mu_yy_0, mu_zz_0, mu_xy_0,
        ])

    def _unpack_state(self, state: jnp.ndarray) -> dict:
        """Unpack state vector into named fields."""
        n = self.n_points
        return {
            "Sigma": state[0],
            "mu_xx": state[1:1 + n],
            "mu_yy": state[1 + n:1 + 2 * n],
            "mu_zz": state[1 + 2 * n:1 + 3 * n],
            "mu_xy": state[1 + 3 * n:1 + 4 * n],
        }

    def _compute_gamma_dot_profile(self, state_fields: dict) -> jnp.ndarray:
        """Compute local shear rate profile from state."""
        G0 = self.G0
        k_d_0 = self.k_d_0
        eta_s = float(self.parameters.get_value("eta_s") or 0.0)
        Sigma = state_fields["Sigma"]
        mu_xy = state_fields["mu_xy"]

        if self._stress_type == "fene":
            L_max = float(self.parameters.get_value("L_max") or 10.0)
            sigma_elastic = jax.vmap(
                lambda xx, yy, zz, xy: vlb_stress_fene_xy(xx, yy, zz, xy, G0, L_max)
            )(state_fields["mu_xx"], state_fields["mu_yy"],
              state_fields["mu_zz"], mu_xy)
        else:
            sigma_elastic = G0 * mu_xy

        eta_eff = max(eta_s, 1e-2 * G0 / max(k_d_0, 1e-30))
        return (Sigma - sigma_elastic) / eta_eff

    # =========================================================================
    # Simulation Methods
    # =========================================================================

    def simulate_steady_shear(
        self,
        gamma_dot_avg: float,
        t_end: float = 100.0,
        dt: float = 0.1,
        perturbation: float = 0.01,
    ) -> dict:
        """Simulate approach to steady state under imposed average shear rate.

        Parameters
        ----------
        gamma_dot_avg : float
            Imposed average shear rate (1/s)
        t_end : float
            Simulation end time (s)
        dt : float
            Output time step (s)
        perturbation : float
            Initial spatial noise amplitude

        Returns
        -------
        dict
            't': time array
            'y': spatial grid
            'mu_xy': mu_xy profiles (N_t, N_y)
            'gamma_dot': shear rate profiles (N_t, N_y)
            'stress': wall stress Sigma(t)
        """
        n = self.n_points
        params = self.get_parameter_dict()

        nu = params.get("nu", 0.0)
        L_max = params.get("L_max", 10.0)

        args = {
            "G0": jnp.float64(params["G0"]),
            "k_d_0": jnp.float64(params["k_d_0"]),
            "eta_s": jnp.float64(params["eta_s"]),
            "D_mu": jnp.float64(params["D_mu"]),
            "nu": jnp.float64(nu),
            "L_max": jnp.float64(L_max),
            "dy": jnp.float64(self.dy),
            "gamma_dot_avg": jnp.float64(gamma_dot_avg),
        }

        pde_rhs = self._build_pde_rhs()
        y0 = self._build_initial_state(perturbation)

        # Set initial stress to expected level
        eta_0 = params["G0"] / params["k_d_0"]
        y0 = y0.at[0].set(eta_0 * gamma_dot_avg)

        n_steps = int(t_end / dt)
        t_save = jnp.linspace(0.0, t_end, n_steps + 1)

        term = diffrax.ODETerm(pde_rhs)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

        sol = diffrax.diffeqsolve(
            term, solver, 0.0, t_end, dt / 10.0, y0,
            args=args, saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=controller,
            max_steps=5_000_000, throw=False,
        )

        # Extract profiles at each time
        t_out = np.asarray(t_save)
        stress_out = np.asarray(sol.ys[:, 0])

        mu_xy_profiles = np.asarray(sol.ys[:, 1 + 3 * n:1 + 4 * n])

        # Compute gamma_dot profiles
        gamma_dot_profiles = []
        for i in range(len(t_out)):
            fields = self._unpack_state(sol.ys[i])
            gd = self._compute_gamma_dot_profile(fields)
            gamma_dot_profiles.append(np.asarray(gd))

        return {
            "t": t_out,
            "y": self.y,
            "mu_xy": mu_xy_profiles,
            "gamma_dot": np.array(gamma_dot_profiles),
            "stress": stress_out,
        }

    def simulate_startup(
        self,
        gamma_dot_avg: float,
        t_end: float = 50.0,
        dt: float = 0.05,
    ) -> dict:
        """Simulate startup from rest with banding evolution.

        Parameters
        ----------
        gamma_dot_avg : float
            Imposed average shear rate (1/s)
        t_end : float
            End time (s)
        dt : float
            Output time step (s)

        Returns
        -------
        dict
            Same format as simulate_steady_shear
        """
        return self.simulate_steady_shear(
            gamma_dot_avg, t_end=t_end, dt=dt, perturbation=0.01
        )

    def simulate_creep(
        self,
        sigma_0: float,
        t_end: float = 100.0,
        dt: float = 0.1,
    ) -> dict:
        """Simulate stress-controlled creep with spatial resolution.

        In creep, the stress Sigma is held fixed (no feedback).

        Parameters
        ----------
        sigma_0 : float
            Applied stress (Pa)
        t_end : float
            End time (s)
        dt : float
            Output time step (s)

        Returns
        -------
        dict
            't', 'y', 'gamma_dot', 'mu_xy', 'velocity'
        """
        n = self.n_points
        params = self.get_parameter_dict()
        nu = params.get("nu", 0.0)
        L_max = params.get("L_max", 10.0)
        breakage = self._breakage
        stress_type = self._stress_type
        dy = self.dy

        # Creep PDE: Sigma is constant, no feedback
        @jax.jit
        def creep_rhs(t, state, args):
            G0 = args["G0"]
            k_d_0 = args["k_d_0"]
            eta_s = args["eta_s"]
            D_mu = args["D_mu"]
            nu_val = args["nu"]
            L_max_val = args["L_max"]
            dy_val = args["dy"]
            Sigma = args["Sigma"]

            mu_xx = state[0:n]
            mu_yy = state[n:2 * n]
            mu_zz = state[2 * n:3 * n]
            mu_xy = state[3 * n:4 * n]

            if breakage == "bell":
                k_d = jax.vmap(
                    lambda xx, yy, zz: vlb_breakage_bell(xx, yy, zz, k_d_0, nu_val)
                )(mu_xx, mu_yy, mu_zz)
            else:
                k_d = jnp.full(n, k_d_0)

            if stress_type == "fene":
                sigma_elastic = jax.vmap(
                    lambda xx, yy, zz, xy: vlb_stress_fene_xy(
                        xx, yy, zz, xy, G0, L_max_val
                    )
                )(mu_xx, mu_yy, mu_zz, mu_xy)
            else:
                sigma_elastic = G0 * mu_xy

            eta_eff = jnp.maximum(eta_s, 1e-2 * G0 / jnp.maximum(k_d_0, 1e-30))
            gamma_dot = (Sigma - sigma_elastic) / eta_eff

            dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
            dmu_yy = k_d * (1.0 - mu_yy)
            dmu_zz = k_d * (1.0 - mu_zz)
            dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy

            dmu_xx = dmu_xx + D_mu * laplacian_1d_neumann_vlb(mu_xx, dy_val)
            dmu_yy = dmu_yy + D_mu * laplacian_1d_neumann_vlb(mu_yy, dy_val)
            dmu_zz = dmu_zz + D_mu * laplacian_1d_neumann_vlb(mu_zz, dy_val)
            dmu_xy = dmu_xy + D_mu * laplacian_1d_neumann_vlb(mu_xy, dy_val)

            return jnp.concatenate([dmu_xx, dmu_yy, dmu_zz, dmu_xy])

        args = {
            "G0": jnp.float64(params["G0"]),
            "k_d_0": jnp.float64(params["k_d_0"]),
            "eta_s": jnp.float64(params["eta_s"]),
            "D_mu": jnp.float64(params["D_mu"]),
            "nu": jnp.float64(nu),
            "L_max": jnp.float64(L_max),
            "dy": jnp.float64(dy),
            "Sigma": jnp.float64(sigma_0),
        }

        # Initial state: equilibrium + noise
        key = jax.random.PRNGKey(42)
        noise = 0.01 * jax.random.normal(key, shape=(n,))
        y0 = jnp.concatenate([
            jnp.ones(n) + noise,  # mu_xx
            jnp.ones(n),          # mu_yy
            jnp.ones(n),          # mu_zz
            jnp.zeros(n),         # mu_xy
        ])

        n_steps = int(t_end / dt)
        t_save = jnp.linspace(0.0, t_end, n_steps + 1)

        term = diffrax.ODETerm(creep_rhs)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-4, atol=1e-6)

        sol = diffrax.diffeqsolve(
            term, solver, 0.0, t_end, dt / 10.0, y0,
            args=args, saveat=diffrax.SaveAt(ts=t_save),
            stepsize_controller=controller,
            max_steps=5_000_000, throw=False,
        )

        t_out = np.asarray(t_save)
        mu_xy_profiles = np.asarray(sol.ys[:, 3 * n:4 * n])

        # Compute gamma_dot and velocity profiles
        gamma_dot_profiles = []
        velocity_profiles = []
        for i in range(len(t_out)):
            mu_xy_i = sol.ys[i, 3 * n:4 * n]
            if self._stress_type == "fene":
                mu_xx_i = sol.ys[i, :n]
                mu_yy_i = sol.ys[i, n:2 * n]
                mu_zz_i = sol.ys[i, 2 * n:3 * n]
                sigma_el = jax.vmap(
                    lambda xx, yy, zz, xy: vlb_stress_fene_xy(
                        xx, yy, zz, xy, params["G0"], L_max
                    )
                )(mu_xx_i, mu_yy_i, mu_zz_i, mu_xy_i)
            else:
                sigma_el = params["G0"] * mu_xy_i

            eta_eff = max(params["eta_s"], 1e-2 * params["G0"] / max(params["k_d_0"], 1e-30))
            gd = np.asarray((sigma_0 - sigma_el) / eta_eff)
            gamma_dot_profiles.append(gd)

            # Velocity from integrating shear rate
            v = np.concatenate([[0.0], np.cumsum(gd[:-1]) * self.dy])
            velocity_profiles.append(v)

        return {
            "t": t_out,
            "y": self.y,
            "mu_xy": mu_xy_profiles,
            "gamma_dot": np.array(gamma_dot_profiles),
            "velocity": np.array(velocity_profiles),
            "stress": np.full(len(t_out), sigma_0),
        }

    # =========================================================================
    # Banding Detection
    # =========================================================================

    def detect_banding(
        self, result: dict, threshold: float = 0.1
    ) -> dict:
        """Detect shear banding from steady-state profiles.

        Parameters
        ----------
        result : dict
            Result from simulate_steady_shear()
        threshold : float
            Relative variation threshold for banding detection

        Returns
        -------
        dict
            'is_banding': bool
            'band_contrast': max/min shear rate ratio
            'band_width': approximate band width (m)
            'band_location': center of high-shear band (m)
        """
        gamma_dot_final = result["gamma_dot"][-1]

        mean_gd = np.mean(gamma_dot_final)
        std_gd = np.std(gamma_dot_final)
        relative_variation = std_gd / max(mean_gd, 1e-10)

        is_banding = relative_variation > threshold

        band_contrast = np.max(gamma_dot_final) / max(np.min(gamma_dot_final), 1e-10)

        # Find band location
        high_shear_mask = gamma_dot_final > mean_gd + std_gd
        if np.any(high_shear_mask):
            band_indices = np.where(high_shear_mask)[0]
            band_width = (band_indices[-1] - band_indices[0]) * self.dy
            band_location = self.y[band_indices].mean()
        else:
            band_width = self.gap_width
            band_location = self.gap_width / 2

        return {
            "is_banding": is_banding,
            "relative_variation": relative_variation,
            "band_contrast": band_contrast,
            "band_width": band_width,
            "band_width_fraction": band_width / self.gap_width,
            "band_location": band_location,
            "gamma_dot_profile": gamma_dot_final,
        }

    def get_velocity_profile(self, result: dict) -> np.ndarray:
        """Compute velocity profile from final shear rate profile.

        v(y) = integral_0^y gamma_dot(y') dy'

        Parameters
        ----------
        result : dict
            Result from simulate_steady_shear()

        Returns
        -------
        np.ndarray
            Velocity profile v(y)
        """
        gamma_dot_final = result["gamma_dot"][-1]
        v = np.concatenate([[0.0], np.cumsum(gamma_dot_final[:-1]) * self.dy])
        return v

    # =========================================================================
    # Visualization
    # =========================================================================

    def plot_profiles(self, result: dict, ax=None):
        """Plot spatial profiles (shear rate and mu_xy).

        Parameters
        ----------
        result : dict
            Result from simulate_steady_shear()
        ax : matplotlib axes, optional
            If None, creates new figure

        Returns
        -------
        matplotlib figure
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        else:
            fig = ax[0].get_figure()
            axes = ax

        y_mm = self.y * 1e3  # Convert to mm

        # Shear rate profile
        axes[0].plot(y_mm, result["gamma_dot"][-1])
        axes[0].set_xlabel("Position y (mm)")
        axes[0].set_ylabel("Shear rate (1/s)")
        axes[0].set_title("Shear Rate Profile")

        # mu_xy profile
        axes[1].plot(y_mm, result["mu_xy"][-1])
        axes[1].set_xlabel("Position y (mm)")
        axes[1].set_ylabel(r"$\mu_{xy}$")
        axes[1].set_title("Distribution Tensor Profile")

        # Stress evolution
        axes[2].plot(result["t"], result["stress"])
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Stress (Pa)")
        axes[2].set_title("Stress Evolution")

        plt.tight_layout()
        return fig

    # =========================================================================
    # Fit/Predict (minimal implementation)
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit is not supported for nonlocal models (use simulate methods)."""
        raise NotImplementedError(
            "VLBNonlocal does not support _fit(). Use simulate_steady_shear() "
            "or simulate_startup() for direct simulation."
        )

    def _predict(self, X, **kwargs):
        """Predict is not directly supported for nonlocal models."""
        raise NotImplementedError(
            "VLBNonlocal does not support _predict(). Use simulate_steady_shear() "
            "for flow curve predictions."
        )

    # =========================================================================
    # Repr
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"VLBNonlocal(breakage={self._breakage!r}, "
            f"stress={self._stress_type!r}, "
            f"n_points={self.n_points}, gap={self.gap_width:.1e}m)"
        )
