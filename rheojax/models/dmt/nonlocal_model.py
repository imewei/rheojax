"""Nonlocal (1D) de Souza Mendes-Thompson (DMT) model.

Implements the spatially-resolved DMT model for shear banding analysis.
Adds fluidity diffusion to regularize the local model and allow for
heterogeneous flow profiles.

This model is appropriate for:
- Shear banding detection and characterization
- Gap-dependent rheology
- Cooperativity length scale estimation
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger
from rheojax.models.dmt._base import DMTBase
from rheojax.models.dmt._kernels import (
    elastic_modulus,
    structure_evolution,
    viscosity_exponential,
    viscosity_herschel_bulkley_regularized,
)

# Safe JAX import
jax, jnp = safe_import_jax()

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register(
    "dmt_nonlocal",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.STARTUP,
        Protocol.CREEP,
    ],
)
class DMTNonlocal(DMTBase):
    """Nonlocal (1D) DMT model for shear banding analysis.

    Extends the local DMT model with spatial structure diffusion:

    ∂λ/∂t = (1-λ)/t_eq - a·λ·|γ̇|^c/t_eq + D_λ·∂²λ/∂y²

    The diffusion term introduces a cooperativity length scale:
    ξ ~ √(D_λ · t_eq)

    which regularizes the problem and sets the minimum width of
    shear bands.

    This model solves for:
    - λ(y, t): Structure field across the gap
    - v(y, t): Velocity profile (from momentum balance)
    - γ̇(y, t): Local shear rate

    Parameters
    ----------
    closure : {"exponential", "herschel_bulkley"}, default "exponential"
        Viscosity closure type.
    include_elasticity : bool, default True
        Include Maxwell viscoelastic backbone.
    n_points : int, default 51
        Number of spatial grid points across the gap.
    gap_width : float, default 1e-3
        Gap width H [m] (e.g., for Couette cell).

    Attributes
    ----------
    n_points : int
        Spatial grid resolution
    gap_width : float
        Physical gap width [m]
    y : array
        Spatial coordinate array [m]

    Examples
    --------
    >>> from rheojax.models.dmt import DMTNonlocal
    >>>
    >>> # Create nonlocal model for banding analysis
    >>> model = DMTNonlocal(
    ...     closure="herschel_bulkley",
    ...     n_points=101,
    ...     gap_width=1e-3
    ... )
    >>>
    >>> # Simulate steady shear with banding
    >>> result = model.simulate_steady_shear(
    ...     gamma_dot_avg=10.0, t_end=1000.0
    ... )
    >>>
    >>> # Check for banding
    >>> banding_info = model.detect_banding(result)

    See Also
    --------
    DMTLocal : Local (0D) variant for homogeneous flow
    FluidityNonlocal : Simpler nonlocal fluidity model

    References
    ----------
    Coussot, P. et al. (2002). "Viscosity bifurcation in thixotropic,
        yielding fluids." J. Rheol. 46, 573-589.
    """

    def __init__(
        self,
        closure: Literal["exponential", "herschel_bulkley"] = "exponential",
        include_elasticity: bool = True,
        n_points: int = 51,
        gap_width: float = 1e-3,
    ):
        """Initialize DMTNonlocal model."""
        self.n_points = n_points
        self.gap_width = gap_width

        super().__init__(closure=closure, include_elasticity=include_elasticity)

        # Add nonlocal-specific parameters
        self._add_nonlocal_parameters()

        # Spatial grid
        self.y = np.linspace(0, gap_width, n_points)
        self.dy = gap_width / (n_points - 1)

        logger.info(
            "DMTNonlocal initialized",
            closure=closure,
            include_elasticity=include_elasticity,
            n_points=n_points,
            gap_width=gap_width,
        )

    def _add_nonlocal_parameters(self):
        """Add parameters specific to nonlocal model."""
        # D_λ: Structure diffusion coefficient
        self.parameters.add(
            name="D_lambda",
            value=1e-6,
            bounds=(1e-10, 1e-2),
            units="m²/s",
            description="Structure diffusion coefficient (cooperativity)",
        )

    def get_cooperativity_length(self) -> float:
        """Compute cooperativity length scale.

        ξ = √(D_λ · t_eq)

        Returns
        -------
        float
            Cooperativity length [m]
        """
        D_lambda = self.parameters.get_value("D_lambda")
        t_eq = self.parameters.get_value("t_eq")
        return np.sqrt(D_lambda * t_eq)

    # =========================================================================
    # Required Abstract Methods
    # =========================================================================

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "DMTNonlocal":
        """Fit model to data."""
        test_mode = kwargs.get("test_mode", "flow_curve")

        if test_mode in ("flow_curve", "rotation"):
            return self._fit_flow_curve(X, y, **kwargs)
        elif test_mode == "startup":
            return self._fit_startup(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict model response."""
        test_mode = kwargs.get("test_mode", "flow_curve")

        if test_mode in ("flow_curve", "rotation"):
            return self._predict_flow_curve(X, **kwargs)
        else:
            raise ValueError(f"Unknown test_mode for prediction: {test_mode}")

    # =========================================================================
    # Spatial Operators
    # =========================================================================

    def _laplacian(self, field: jnp.ndarray) -> jnp.ndarray:
        """Compute 1D Laplacian with Neumann BCs (∂field/∂y = 0 at walls).

        Uses second-order central differences.

        Parameters
        ----------
        field : array
            Field values at grid points (shape: n_points)

        Returns
        -------
        array
            Laplacian ∂²field/∂y² at each point
        """
        dy_sq = self.dy ** 2

        # Interior points: central difference
        lap = jnp.zeros_like(field)
        lap = lap.at[1:-1].set(
            (field[:-2] - 2 * field[1:-1] + field[2:]) / dy_sq
        )

        # Neumann BCs: ∂field/∂y = 0 at y=0 and y=H
        # Ghost point approach: field[-1] = field[1], field[N] = field[N-2]
        lap = lap.at[0].set(2 * (field[1] - field[0]) / dy_sq)
        lap = lap.at[-1].set(2 * (field[-2] - field[-1]) / dy_sq)

        return lap

    def _compute_shear_rate_from_velocity(
        self, v: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute shear rate from velocity profile.

        γ̇(y) = dv/dy

        Parameters
        ----------
        v : array
            Velocity profile v(y)

        Returns
        -------
        array
            Shear rate profile γ̇(y)
        """
        gamma_dot = jnp.zeros_like(v)

        # Central difference for interior
        gamma_dot = gamma_dot.at[1:-1].set(
            (v[2:] - v[:-2]) / (2 * self.dy)
        )

        # One-sided for boundaries
        gamma_dot = gamma_dot.at[0].set((v[1] - v[0]) / self.dy)
        gamma_dot = gamma_dot.at[-1].set((v[-1] - v[-2]) / self.dy)

        return gamma_dot

    # =========================================================================
    # Steady Shear Simulation
    # =========================================================================

    def simulate_steady_shear(
        self,
        gamma_dot_avg: float,
        t_end: float,
        dt: float = 0.1,
        lam_init: float | np.ndarray = 1.0,
    ) -> dict[str, np.ndarray]:
        """Simulate approach to steady state under imposed average shear rate.

        For planar Couette: v(0) = 0, v(H) = V_wall = γ̇_avg · H

        The stress is uniform (σ(y) = Σ) at low Reynolds number.
        The local shear rate γ̇(y) varies to satisfy the local constitutive
        relation with the uniform stress.

        Parameters
        ----------
        gamma_dot_avg : float
            Average (imposed) shear rate [1/s]
        t_end : float
            Simulation end time [s]
        dt : float
            Time step [s]
        lam_init : float or array
            Initial structure (scalar for uniform, array for profile)

        Returns
        -------
        dict
            't': time array
            'lam': structure profiles λ(y, t_i)
            'gamma_dot': shear rate profiles γ̇(y, t_i)
            'velocity': velocity profiles v(y, t_i)
            'stress': wall stress Σ(t)
        """
        n_steps = int(t_end / dt)
        params = self.get_parameter_dict()

        # Wall velocity for imposed average shear rate
        V_wall = gamma_dot_avg * self.gap_width

        # Initial conditions
        if isinstance(lam_init, float):
            lam = jnp.ones(self.n_points) * lam_init
        else:
            lam = jnp.array(lam_init)

        # Initialize velocity (linear profile for homogeneous flow)
        v = jnp.linspace(0, V_wall, self.n_points)

        # Storage for trajectory
        t_list = []
        lam_list = []
        gamma_dot_list = []
        v_list = []
        stress_list = []

        # Time stepping
        for step in range(n_steps):
            t = step * dt
            t_list.append(t)

            # Compute local shear rate
            gamma_dot = self._compute_shear_rate_from_velocity(v)
            gamma_dot_list.append(np.array(gamma_dot))

            # Compute local viscosity
            if self.closure == "exponential":
                eta = jax.vmap(
                    lambda l, gd: viscosity_exponential(l, params["eta_0"], params["eta_inf"])
                )(lam, gamma_dot)
            else:
                eta = jax.vmap(
                    lambda l, gd: viscosity_herschel_bulkley_regularized(
                        l, gd, params["tau_y0"], params["K0"],
                        params["n_flow"], params["eta_inf"],
                        params["m1"], params["m2"]
                    )
                )(lam, gamma_dot)

            # Compute stress (uniform for low Re)
            stress = jnp.mean(eta * gamma_dot)
            stress_list.append(float(stress))

            # Store profiles
            lam_list.append(np.array(lam))
            v_list.append(np.array(v))

            # Update structure: local evolution + diffusion
            # dλ/dt = (local) + D_λ ∂²λ/∂y²
            local_rate = jax.vmap(
                lambda l, gd: structure_evolution(l, gd, params["t_eq"], params["a"], params["c"])
            )(lam, gamma_dot)

            diffusion = params["D_lambda"] * self._laplacian(lam)
            dlam_dt = local_rate + diffusion

            lam = jnp.clip(lam + dt * dlam_dt, 0.0, 1.0)

            # Update velocity (stress-driven approach)
            # In steady Couette, we adjust velocity to maintain V_wall BC
            # while distributing shear according to local viscosity
            # v(y) = ∫₀^y (Σ/η(y')) dy'
            # For simplicity, use iterative approach

            # Compute target shear rate from uniform stress
            if self.closure == "exponential":
                gamma_dot_target = stress / eta
            else:
                # Iterative for HB
                gamma_dot_target = stress / jnp.maximum(eta, 1e-10)

            # Reconstruct velocity from shear rate
            v_new = jnp.concatenate([
                jnp.array([0.0]),
                jnp.cumsum(gamma_dot_target[:-1]) * self.dy
            ])

            # Rescale to match wall velocity
            v = v_new * V_wall / jnp.maximum(v_new[-1], 1e-10)

        return {
            "t": np.array(t_list),
            "y": self.y,
            "lam": np.array(lam_list),
            "gamma_dot": np.array(gamma_dot_list),
            "velocity": np.array(v_list),
            "stress": np.array(stress_list),
        }

    # =========================================================================
    # Banding Detection
    # =========================================================================

    def detect_banding(
        self,
        result: dict,
        threshold: float = 0.1,
    ) -> dict:
        """Detect shear banding from steady-state profiles.

        A shear band is detected when the shear rate profile shows
        significant spatial variation (standard deviation / mean > threshold).

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
            'band_width': approximate band width [m]
            'band_location': center of high-shear band [m]
            'gamma_dot_profile': final shear rate profile
            'lam_profile': final structure profile
        """
        # Use final profiles
        gamma_dot_final = result["gamma_dot"][-1]
        lam_final = result["lam"][-1]

        # Compute variation metrics
        mean_gd = np.mean(gamma_dot_final)
        std_gd = np.std(gamma_dot_final)
        relative_variation = std_gd / max(mean_gd, 1e-10)

        is_banding = relative_variation > threshold

        # Band contrast
        band_contrast = np.max(gamma_dot_final) / max(np.min(gamma_dot_final), 1e-10)

        # Find band location and width
        # High-shear band is where γ̇ > mean + std
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
            "lam_profile": lam_final,
        }

    # =========================================================================
    # Flow Curve
    # =========================================================================

    def _fit_flow_curve(
        self, gamma_dot: np.ndarray, stress: np.ndarray, **kwargs
    ) -> "DMTNonlocal":
        """Fit to steady-state flow curve.

        For nonlocal model, this fits to the apparent (average) flow curve.
        """
        # Use local model fit as approximation
        from rheojax.models.dmt.local import DMTLocal

        local_model = DMTLocal(
            closure=self.closure,
            include_elasticity=self.include_elasticity
        )
        local_model._fit_flow_curve(gamma_dot, stress, **kwargs)

        # Copy parameters
        for name in local_model.parameters.keys():
            if name in self.parameters.keys():
                self.parameters.set_value(name, local_model.parameters.get_value(name))

        self._fitted = True
        return self

    def _predict_flow_curve(self, gamma_dot: np.ndarray, **kwargs) -> np.ndarray:
        """Predict steady-state flow curve.

        Can either use:
        - Homogeneous (local) approximation
        - Full nonlocal simulation at each point (slow)
        """
        use_local = kwargs.get("use_local_approximation", True)

        if use_local:
            # Use local approximation
            from rheojax.models.dmt.local import DMTLocal

            local_model = DMTLocal(
                closure=self.closure,
                include_elasticity=self.include_elasticity
            )
            # Copy parameters
            for name in self.parameters.keys():
                if name in local_model.parameters.keys():
                    local_model.parameters.set_value(
                        name, self.parameters.get_value(name)
                    )
            return local_model._predict_flow_curve(gamma_dot)
        else:
            # Full nonlocal simulation
            stress = []
            for gd in gamma_dot:
                result = self.simulate_steady_shear(
                    gamma_dot_avg=float(gd),
                    t_end=kwargs.get("t_equilibrate", 1000.0),
                    dt=kwargs.get("dt", 0.1),
                )
                stress.append(result["stress"][-1])
            return np.array(stress)

    def _fit_startup(self, t: np.ndarray, stress: np.ndarray, **kwargs) -> "DMTNonlocal":
        """Fit to startup transient."""
        raise NotImplementedError("Startup fitting for nonlocal model not implemented")

    # =========================================================================
    # Visualization Helpers
    # =========================================================================

    def plot_profiles(
        self,
        result: dict,
        time_indices: list[int] | None = None,
        figsize: tuple = (12, 4),
    ):
        """Plot structure and shear rate profiles.

        Parameters
        ----------
        result : dict
            Result from simulate_steady_shear()
        time_indices : list, optional
            Indices of time points to plot (default: [0, -1])
        figsize : tuple
            Figure size

        Returns
        -------
        fig, axes
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt

        if time_indices is None:
            time_indices = [0, len(result["t"]) // 2, -1]

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        y_mm = result["y"] * 1000  # Convert to mm

        for i, idx in enumerate(time_indices):
            t_val = result["t"][idx]
            color = plt.cm.viridis(i / (len(time_indices) - 1))

            # Structure profile
            axes[0].plot(
                y_mm, result["lam"][idx],
                color=color, label=f"t = {t_val:.1f} s"
            )

            # Shear rate profile
            axes[1].plot(
                y_mm, result["gamma_dot"][idx],
                color=color, label=f"t = {t_val:.1f} s"
            )

            # Velocity profile
            axes[2].plot(
                y_mm, result["velocity"][idx] * 1000,  # Convert to mm/s
                color=color, label=f"t = {t_val:.1f} s"
            )

        axes[0].set_xlabel("y [mm]")
        axes[0].set_ylabel("λ [-]")
        axes[0].set_title("Structure Profile")
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        axes[1].set_xlabel("y [mm]")
        axes[1].set_ylabel("γ̇ [1/s]")
        axes[1].set_title("Shear Rate Profile")
        axes[1].legend()

        axes[2].set_xlabel("y [mm]")
        axes[2].set_ylabel("v [mm/s]")
        axes[2].set_title("Velocity Profile")
        axes[2].legend()

        plt.tight_layout()
        return fig, axes
