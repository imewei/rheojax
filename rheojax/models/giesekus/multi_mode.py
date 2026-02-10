"""Multi-mode Giesekus nonlinear viscoelastic model.

This module implements `GiesekusMultiMode`, an extension of the single-mode
Giesekus model with N parallel relaxation modes.

Multi-Mode Superposition
------------------------
The total stress is the sum of N polymer modes plus a Newtonian solvent::

    σ_total = σ_s + Σᵢ σ_p,i

where each mode i has its own parameters (η_p,i, λ_i, α_i).

For SAOS (linear regime)::

    G'(ω) = Σᵢ G_i·(ωλ_i)² / (1 + (ωλ_i)²)
    G''(ω) = Σᵢ G_i·(ωλ_i) / (1 + (ωλ_i)²) + η_s·ω

where G_i = η_p,i / λ_i.

Example
-------
>>> from rheojax.models.giesekus import GiesekusMultiMode
>>> import numpy as np
>>>
>>> # Create 3-mode model
>>> model = GiesekusMultiMode(n_modes=3)
>>>
>>> # Set mode parameters
>>> model.set_mode_params(0, eta_p=100.0, lambda_1=10.0, alpha=0.3)
>>> model.set_mode_params(1, eta_p=50.0, lambda_1=1.0, alpha=0.2)
>>> model.set_mode_params(2, eta_p=20.0, lambda_1=0.1, alpha=0.1)
>>>
>>> # Predict SAOS
>>> omega = np.logspace(-2, 2, 50)
>>> G_prime, G_double_prime = model.predict_saos(omega)

References
----------
- Giesekus, H. (1982). J. Non-Newtonian Fluid Mech. 11, 69-109.
- Bird, R.B. et al. (1987). Dynamics of Polymeric Liquids, Vol. 1.
"""

from __future__ import annotations

import logging

import diffrax
import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry
from rheojax.models.giesekus._kernels import (
    giesekus_multimode_ode_rhs,
    giesekus_multimode_saos_moduli,
    giesekus_steady_shear_stress_vec,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "giesekus_multi",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
    ],
)
@ModelRegistry.register(
    "giesekus_multimode",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
    ],
)
class GiesekusMultiMode(BaseModel):
    """Multi-mode Giesekus nonlinear viscoelastic model.

    This model extends the single-mode Giesekus with N parallel Maxwell
    modes, each with its own relaxation time, viscosity, and mobility
    factor.

    The constitutive equation for each mode is::

        τᵢ + λᵢ∇̂τᵢ + (αᵢλᵢ/η_p,i)τᵢ·τᵢ = 2η_p,i D

    Total stress: σ = η_s·γ̇ + Σᵢ τᵢ

    Parameters
    ----------
    n_modes : int
        Number of relaxation modes (N ≥ 1). Default: 3

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters including per-mode values
    fitted_ : bool
        Whether the model has been fitted

    Notes
    -----
    The multi-mode model is particularly useful for:

    1. Fitting broad SAOS spectra that single-mode cannot capture
    2. Representing polydisperse polymer systems
    3. Capturing multiple relaxation processes

    Each mode can have different α_i values, allowing different
    molecular weight fractions to exhibit different anisotropy.

    See Also
    --------
    GiesekusSingleMode : Single relaxation time variant
    GeneralizedMaxwell : Linear multi-mode Maxwell model
    """

    def __init__(self, n_modes: int = 3):
        """Initialize multi-mode Giesekus model.

        Parameters
        ----------
        n_modes : int, default 3
            Number of relaxation modes (must be ≥ 1)

        Raises
        ------
        ValueError
            If n_modes < 1
        """
        super().__init__()

        if n_modes < 1:
            raise ValueError(f"n_modes must be ≥ 1, got {n_modes}")

        self._n_modes = n_modes
        self._test_mode = None
        self._setup_parameters()

        # Protocol-specific inputs
        self._gamma_dot_applied: float | None = None
        self._sigma_applied: float | None = None
        self._gamma_0: float | None = None
        self._omega_laos: float | None = None

        # Internal storage
        self._trajectory: dict[str, np.ndarray] | None = None

    def _setup_parameters(self):
        """Initialize ParameterSet with multi-mode parameters.

        Creates parameters:
        - eta_s: Shared solvent viscosity
        - eta_p_i: Polymer viscosity for mode i
        - lambda_i: Relaxation time for mode i
        - alpha_i: Mobility factor for mode i
        """
        self.parameters = ParameterSet()

        # Shared solvent viscosity
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian contribution)",
        )

        # Per-mode parameters with logarithmically spaced defaults
        for i in range(self._n_modes):
            # Viscosity (decreasing with mode number)
            eta_default = 100.0 / (i + 1)
            self.parameters.add(
                name=f"eta_p_{i}",
                value=eta_default,
                bounds=(1e-6, 1e6),
                units="Pa·s",
                description=f"Polymer viscosity for mode {i}",
            )

            # Relaxation time (logarithmically spaced)
            lambda_default = 10.0 ** (1 - i)  # 10, 1, 0.1, ...
            self.parameters.add(
                name=f"lambda_{i}",
                value=lambda_default,
                bounds=(1e-8, 1e4),
                units="s",
                description=f"Relaxation time for mode {i}",
            )

            # Mobility factor (same default for all modes)
            self.parameters.add(
                name=f"alpha_{i}",
                value=0.3,
                bounds=(0.0, 0.5),
                units="dimensionless",
                description=f"Mobility factor for mode {i}",
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_modes(self) -> int:
        """Get number of modes."""
        return self._n_modes

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        return float(self.parameters.get_value("eta_s"))

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = η_s + Σ η_p,i (Pa·s)."""
        eta_p_total = sum(
            self.parameters.get_value(f"eta_p_{i}") for i in range(self._n_modes)
        )
        return self.eta_s + eta_p_total

    def get_mode_params(self, mode_idx: int) -> dict[str, float]:
        """Get parameters for a specific mode.

        Parameters
        ----------
        mode_idx : int
            Mode index (0 to n_modes-1)

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'eta_p', 'lambda_1', 'alpha'
        """
        if mode_idx < 0 or mode_idx >= self._n_modes:
            raise IndexError(f"Mode index {mode_idx} out of range [0, {self._n_modes})")

        return {
            "eta_p": float(self.parameters.get_value(f"eta_p_{mode_idx}")),
            "lambda_1": float(self.parameters.get_value(f"lambda_{mode_idx}")),
            "alpha": float(self.parameters.get_value(f"alpha_{mode_idx}")),
        }

    def set_mode_params(
        self,
        mode_idx: int,
        eta_p: float | None = None,
        lambda_1: float | None = None,
        alpha: float | None = None,
    ) -> None:
        """Set parameters for a specific mode.

        Parameters
        ----------
        mode_idx : int
            Mode index (0 to n_modes-1)
        eta_p : float, optional
            Polymer viscosity (Pa·s)
        lambda_1 : float, optional
            Relaxation time (s)
        alpha : float, optional
            Mobility factor (0 ≤ α ≤ 0.5)
        """
        if mode_idx < 0 or mode_idx >= self._n_modes:
            raise IndexError(f"Mode index {mode_idx} out of range [0, {self._n_modes})")

        if eta_p is not None:
            self.parameters.set_value(f"eta_p_{mode_idx}", eta_p)
        if lambda_1 is not None:
            self.parameters.set_value(f"lambda_{mode_idx}", lambda_1)
        if alpha is not None:
            self.parameters.set_value(f"alpha_{mode_idx}", alpha)

    def get_mode_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Get all mode parameters as JAX arrays.

        Uses vectorized extraction via get_values() + slicing for ~3x speedup
        over N individual get_value() calls.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (eta_p_modes, lambda_modes, alpha_modes), each shape (n_modes,)
        """
        # Get all parameter values at once (single dict lookup traversal)
        all_values = self.parameters.get_values()  # shape: (1 + 3*n_modes,)

        # Parameter layout: [eta_s, eta_p_0, lambda_0, alpha_0, eta_p_1, ...]
        # Extract mode arrays using NumPy slicing (faster than list comprehension)
        # eta_p: indices 1, 4, 7, ... (stride 3 starting from 1)
        # lambda: indices 2, 5, 8, ... (stride 3 starting from 2)
        # alpha: indices 3, 6, 9, ... (stride 3 starting from 3)
        eta_p = jnp.asarray(all_values[1::3][:self._n_modes], dtype=jnp.float64)
        lambda_vals = jnp.asarray(all_values[2::3][:self._n_modes], dtype=jnp.float64)
        alpha = jnp.asarray(all_values[3::3][:self._n_modes], dtype=jnp.float64)

        return eta_p, lambda_vals, alpha

    # =========================================================================
    # Core Interface Methods
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data.

        Parameters
        ----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
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

        test_mode = kwargs.get("test_mode", self._test_mode or "oscillation")
        self._test_mode = test_mode

        x_jax = jnp.asarray(x, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # Define model function for fitting (follows ParameterSet ordering)
        def model_fn(x_fit, params):
            """Stateless model function for optimization."""
            return self.model_function(x_fit, params, test_mode=test_mode)

        # Create objective and optimize using ParameterSet
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

        logger.info(f"Fitted {self._n_modes}-mode Giesekus: η₀={self.eta_0:.2e} Pa·s")

        return self

    def _predict(self, x, **kwargs):
        """Predict response.

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
        test_mode = kwargs.get("test_mode", self._test_mode or "oscillation")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        param_names = list(self.parameters.keys())
        params = jnp.array(
            [self.parameters.get_value(n) for n in param_names], dtype=jnp.float64
        )

        # Forward kwargs (gamma_dot, sigma_applied, etc.) to model_function
        predict_kwargs = {k: v for k, v in kwargs.items() if k != "test_mode"}
        return self.model_function(x_jax, params, test_mode=test_mode, **predict_kwargs)

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            All parameter values in order
        test_mode : str, optional
            Override stored test mode
        **kwargs : dict
            Protocol-specific arguments (gamma_dot, sigma_applied, etc.)

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        mode = test_mode or self._test_mode or "oscillation"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Parse parameters - interleaved order:
        # [eta_s, eta_p_0, lambda_0, alpha_0, eta_p_1, lambda_1, alpha_1, ...]
        eta_s = params[0]

        # Use stride-3 slicing matching _setup_parameters() order
        eta_p_modes = params[1::3][: self._n_modes]
        lambda_modes = params[2::3][: self._n_modes]
        alpha_modes = params[3::3][: self._n_modes]

        if mode == "oscillation":
            G_prime, G_double_prime = self._predict_saos_internal(
                X_jax, eta_p_modes, lambda_modes, eta_s
            )
            # Return components for fitting to [G', G''] data
            return jnp.column_stack([G_prime, G_double_prime])

        elif mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._predict_flow_curve_internal(
                X_jax, eta_p_modes, lambda_modes, alpha_modes, eta_s
            )

        elif mode == "startup":
            # Get gamma_dot from kwargs or instance attribute
            gamma_dot = kwargs.get("gamma_dot") or self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, eta_p_modes, lambda_modes, alpha_modes, gamma_dot
            )

        else:
            logger.warning(f"Unknown test_mode '{mode}', using oscillation")
            G_prime, G_double_prime = self._predict_saos_internal(
                X_jax, eta_p_modes, lambda_modes, eta_s
            )
            return jnp.column_stack([G_prime, G_double_prime])

    # =========================================================================
    # Analytical Predictions
    # =========================================================================

    def _predict_saos_internal(
        self,
        omega: jnp.ndarray,
        eta_p_modes: jnp.ndarray,
        lambda_modes: jnp.ndarray,
        eta_s: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal SAOS prediction."""

        # Vectorize over frequency
        def saos_at_omega(w):
            return giesekus_multimode_saos_moduli(w, eta_p_modes, lambda_modes, eta_s)

        G_prime, G_double_prime = jax.vmap(saos_at_omega)(omega)
        return G_prime, G_double_prime

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

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
        omega_jax = jnp.asarray(omega, dtype=jnp.float64)
        eta_p_modes, lambda_modes, _ = self.get_mode_arrays()

        G_prime, G_double_prime = self._predict_saos_internal(
            omega_jax, eta_p_modes, lambda_modes, self.eta_s
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)

        G_star_mag = jnp.sqrt(G_prime**2 + G_double_prime**2)
        return np.asarray(G_star_mag)

    def _predict_flow_curve_internal(
        self,
        gamma_dot: jnp.ndarray,
        eta_p_modes: jnp.ndarray,
        lambda_modes: jnp.ndarray,
        alpha_modes: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Internal flow curve prediction (steady shear).

        For multi-mode Giesekus, we sum the contributions from each mode.
        This is approximate for nonlinear superposition.
        """

        # Sum contributions from each mode
        # Note: This is exact only in linear regime; nonlinear coupling is neglected
        def mode_contribution(i):
            eta_p = eta_p_modes[i]
            lambda_1 = lambda_modes[i]
            alpha = alpha_modes[i]
            return giesekus_steady_shear_stress_vec(
                gamma_dot, eta_p, lambda_1, alpha, 0.0
            )

        stress_contributions = jax.vmap(mode_contribution)(jnp.arange(len(eta_p_modes)))
        polymer_stress = jnp.sum(stress_contributions, axis=0)

        # Add solvent contribution
        total_stress = polymer_stress + eta_s * gamma_dot

        return total_stress

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict steady shear stress.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return (sigma, eta)

        Returns
        -------
        np.ndarray or tuple
            Shear stress σ (Pa), or (σ, η) if return_components=True
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)
        eta_p_modes, lambda_modes, alpha_modes = self.get_mode_arrays()

        sigma = self._predict_flow_curve_internal(
            gd, eta_p_modes, lambda_modes, alpha_modes, self.eta_s
        )

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            return np.asarray(sigma), np.asarray(eta)

        return np.asarray(sigma)

    # =========================================================================
    # ODE-Based Simulations
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        eta_p_modes: jnp.ndarray,
        lambda_modes: jnp.ndarray,
        alpha_modes: jnp.ndarray,
        gamma_dot: float,
    ) -> jnp.ndarray:
        """Internal startup simulation."""
        # State: [τ_xx^0, τ_yy^0, τ_xy^0, τ_zz^0, ..., τ_xx^N-1, ...]
        # Total size: 4 * n_modes
        y0 = jnp.zeros(4 * self._n_modes, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_multimode_ode_rhs(
                ti,
                yi,
                args["gamma_dot"],
                args["eta_p"],
                args["lambda"],
                args["alpha"],
            )

        args = {
            "gamma_dot": gamma_dot,
            "eta_p": eta_p_modes,
            "lambda": lambda_modes,
            "alpha": alpha_modes,
        }

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t[0]
        t1 = t[-1]
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
            max_steps=100_000,
            throw=False,
        )

        # Sum τ_xy from all modes (index 2 in each mode's 4-element block)
        tau_xy_total = jnp.zeros(len(t), dtype=jnp.float64)
        for i in range(self._n_modes):
            tau_xy_total = tau_xy_total + sol.ys[:, 4 * i + 2]

        # Add solvent contribution
        total_stress = tau_xy_total + self.eta_s * gamma_dot

        # Handle solver failures
        total_stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            total_stress,
            jnp.nan * jnp.ones_like(total_stress),
        )

        return total_stress

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return per-mode stresses

        Returns
        -------
        np.ndarray or dict
            Total shear stress, or dict with per-mode stresses
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        eta_p_modes, lambda_modes, alpha_modes = self.get_mode_arrays()

        # State: 4 * n_modes
        y0 = jnp.zeros(4 * self._n_modes, dtype=jnp.float64)

        def ode_fn(ti, yi, args):
            return giesekus_multimode_ode_rhs(
                ti,
                yi,
                args["gamma_dot"],
                args["eta_p"],
                args["lambda"],
                args["alpha"],
            )

        args = {
            "gamma_dot": gamma_dot,
            "eta_p": eta_p_modes,
            "lambda": lambda_modes,
            "alpha": alpha_modes,
        }

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = t_jax[0]
        t1 = t_jax[-1]
        dt0 = (t1 - t0) / max(len(t), 1000)

        saveat = diffrax.SaveAt(ts=t_jax)

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
            max_steps=100_000,
            throw=False,
        )

        if return_full:
            result = {"t": np.asarray(t_jax)}
            tau_xy_total = np.zeros(len(t))

            for i in range(self._n_modes):
                tau_xy_i = np.asarray(sol.ys[:, 4 * i + 2])
                result[f"tau_xy_{i}"] = tau_xy_i
                tau_xy_total += tau_xy_i

            tau_xy_total_final = tau_xy_total + self.eta_s * gamma_dot

            # Handle solver failures
            tau_xy_total_final = np.where(
                sol.result == diffrax.RESULTS.successful,
                tau_xy_total_final,
                np.nan * np.ones_like(tau_xy_total_final),
            )
            result["tau_xy_total"] = tau_xy_total_final
            return result

        # Sum τ_xy from all modes
        tau_xy_total = np.zeros(len(t))
        for i in range(self._n_modes):
            tau_xy_total += np.asarray(sol.ys[:, 4 * i + 2])

        total_stress = tau_xy_total + self.eta_s * gamma_dot

        # Handle solver failures
        total_stress = np.where(
            sol.result == diffrax.RESULTS.successful,
            total_stress,
            np.nan * np.ones_like(total_stress),
        )

        return total_stress

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def get_relaxation_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Get discrete relaxation spectrum.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (lambda_i, G_i) where G_i = η_p,i / λ_i
        """
        eta_p_modes, lambda_modes, _ = self.get_mode_arrays()
        G_modes = eta_p_modes / lambda_modes

        # Sort by relaxation time (descending)
        sort_idx = jnp.argsort(lambda_modes)[::-1]

        return np.asarray(lambda_modes[sort_idx]), np.asarray(G_modes[sort_idx])

    def get_continuous_spectrum(
        self,
        t: np.ndarray | None = None,
        n_points: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get continuous relaxation modulus G(t).

        Parameters
        ----------
        t : np.ndarray, optional
            Time array
        n_points : int, default 200
            Number of points if t not provided

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, G(t))
        """
        eta_p_modes, lambda_modes, _ = self.get_mode_arrays()

        if t is None:
            lambda_min = float(jnp.min(lambda_modes))
            lambda_max = float(jnp.max(lambda_modes))
            t = np.logspace(
                np.log10(0.01 * lambda_min),
                np.log10(100 * lambda_max),
                n_points,
            )

        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # G(t) = Σ G_i exp(-t/λ_i)
        G_modes = eta_p_modes / lambda_modes

        def G_at_t(t_val):
            return jnp.sum(G_modes * jnp.exp(-t_val / lambda_modes))

        G_t = jax.vmap(G_at_t)(t_jax)

        return np.asarray(t), np.asarray(G_t)

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"n_modes={self._n_modes}, "
            f"η₀={self.eta_0:.2e} Pa·s, "
            f"η_s={self.eta_s:.2e} Pa·s)"
        )
