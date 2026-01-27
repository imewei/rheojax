"""Multi-species Transient Network Theory (TNT) model.

This module implements `TNTMultiSpecies`, a constitutive model for networks
with N distinct bond types that relax independently.

Key Physics
-----------
Multi-species TNT describes networks with heterogeneous physical crosslinks:

- N independent bond populations (species)
- Each species i has its own modulus G_i and lifetime τ_b_i
- Each species has its own conformation tensor S_i that evolves independently
- Total stress is the sum over all species: σ = Σ G_i·S_xy_i + η_s·γ̇

The constitutive equations for each species i::

    dS_i/dt = L·S_i + S_i·L^T + (1/τ_b_i)·I - (1/τ_b_i)·S_i

This represents a superposition of N Maxwell modes (multi-mode UCM),
commonly used to model:
- Polydisperse systems with broad relaxation spectra
- Multicomponent associative networks
- Complex hierarchical structures

Supported Protocols
-------------------
- FLOW_CURVE: Steady shear (analytical)
- OSCILLATION: Small-amplitude oscillatory shear (analytical)
- STARTUP: Transient stress growth (ODE)
- RELAXATION: Stress decay after cessation (analytical/ODE)
- CREEP: Strain evolution under constant stress (ODE)
- LAOS: Large-amplitude oscillatory shear (ODE)

Example
-------
>>> from rheojax.models.tnt import TNTMultiSpecies
>>> import numpy as np
>>>
>>> # Two-species network
>>> model = TNTMultiSpecies(n_species=2)
>>>
>>> # Flow curve (analytical superposition)
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # SAOS (analytical Maxwell superposition)
>>> omega = np.logspace(-2, 2, 50)
>>> G_star = model.predict(omega, test_mode='oscillation',
>>>                         return_components=True)
>>>
>>> # Startup (ODE with 2N state variables)
>>> t = np.linspace(0, 10, 200)
>>> model._gamma_dot_applied = 1.0
>>> sigma_t = model.predict(t, test_mode='startup')

References
----------
- Likhtman, A.E. & Graham, R.S. (2003). J. Non-Newt. Fluid Mech. 114, 1-12.
- Graham, R.S., Likhtman, A.E., McLeish, T.C.B. & Milner, S.T. (2003).
  J. Rheol. 47, 1171-1200.
"""

from __future__ import annotations

import logging
from typing import Literal

import diffrax
import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.models.tnt._base import TNTBase
from rheojax.models.tnt._kernels import (
    tnt_base_steady_stress,
    tnt_multimode_ode_rhs,
    tnt_multimode_relaxation,
    tnt_multimode_relaxation_vec,
    tnt_multimode_saos_moduli,
    tnt_multimode_saos_moduli_vec,
    tnt_single_mode_creep_ode_rhs,
    tnt_single_mode_ode_rhs,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "tnt_multi_species",
    protocols=["flow_curve", "oscillation", "startup", "relaxation", "creep", "laos"],
)
class TNTMultiSpecies(TNTBase):
    """Multi-species Transient Network Theory model.

    Implements a network with N independent bond populations, each with
    its own modulus G_i and lifetime τ_b_i. The total stress is a
    superposition of N Maxwell modes.

    This is equivalent to a generalized Maxwell model where each mode
    represents a distinct physical crosslink species rather than a
    mathematical decomposition.

    Parameters
    ----------
    n_species : int, default 2
        Number of distinct bond species (N ≥ 1)

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters: [G_0, tau_b_0, G_1, tau_b_1, ..., G_{N-1},
        tau_b_{N-1}, eta_s]
    fitted_ : bool
        Whether the model has been fitted
    _n_species : int
        Number of species

    Notes
    -----
    The state vector has 4*N components:
        [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xy_{N-1}]

    Each species evolves independently via the upper-convected derivative
    with constant breakage/creation rates.

    See Also
    --------
    TNTSingleMode : Single-mode TNT with variant breakage rates
    GeneralizedMaxwell : Mathematical multi-mode decomposition
    """

    def __init__(self, n_species: int = 2):
        """Initialize multi-species TNT model.

        Parameters
        ----------
        n_species : int, default 2
            Number of distinct bond species (must be ≥ 1)
        """
        if n_species < 1:
            raise ValueError(f"n_species must be ≥ 1, got {n_species}")

        self._n_species = n_species
        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with 2N+1 parameters.

        Parameters are organized as:
            [G_0, tau_b_0, G_1, tau_b_1, ..., G_{N-1}, tau_b_{N-1}, eta_s]

        Default values:
            - G_i = 1000.0 / N (equal distribution)
            - tau_b_i = 10^(-i) (logarithmic spacing: 1.0, 0.1, 0.01, ...)
            - eta_s = 0.0 (no solvent viscosity)
        """
        self.parameters = ParameterSet()

        for i in range(self._n_species):
            # Mode modulus
            self.parameters.add(
                name=f"G_{i}",
                value=1000.0 / self._n_species,
                bounds=(1e0, 1e8),
                units="Pa",
                description=f"Network modulus for bond species {i}",
            )

            # Mode lifetime (logarithmically spaced by default)
            default_tau = 10.0 ** (-i)
            self.parameters.add(
                name=f"tau_b_{i}",
                value=default_tau,
                bounds=(1e-6, 1e4),
                units="s",
                description=f"Bond lifetime for species {i}",
            )

        # Global solvent viscosity
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian background)",
        )

    # =========================================================================
    # Property Accessors
    # =========================================================================

    @property
    def n_species(self) -> int:
        """Get number of bond species N."""
        return self._n_species

    def _get_mode_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get G_modes and tau_modes as JAX arrays.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (G_modes, tau_modes) with shape (N,)
        """
        G_modes = jnp.array(
            [float(self.parameters.get_value(f"G_{i}")) for i in range(self._n_species)],
            dtype=jnp.float64,
        )
        tau_modes = jnp.array(
            [float(self.parameters.get_value(f"tau_b_{i}")) for i in range(self._n_species)],
            dtype=jnp.float64,
        )
        return G_modes, tau_modes

    @property
    def eta_s(self) -> float:
        """Get solvent viscosity η_s (Pa·s)."""
        return float(self.parameters.get_value("eta_s"))

    @property
    def G_total(self) -> float:
        """Get total modulus G_total = Σ G_i (Pa)."""
        G_modes, _ = self._get_mode_arrays()
        return float(jnp.sum(G_modes))

    @property
    def eta_0(self) -> float:
        """Get zero-shear viscosity η₀ = Σ G_i·τ_b_i + η_s (Pa·s)."""
        G_modes, tau_modes = self._get_mode_arrays()
        return float(jnp.sum(G_modes * tau_modes) + self.eta_s)

    # =========================================================================
    # Equilibrium State
    # =========================================================================

    def get_equilibrium_conformation_multimode(self) -> jnp.ndarray:
        """Return equilibrium conformation for all N modes.

        Returns
        -------
        jnp.ndarray
            Equilibrium state [1, 1, 1, 0, ..., 1, 1, 1, 0] with shape (4N,)
        """
        single_mode_eq = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
        return jnp.tile(single_mode_eq, self._n_species)

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
            # Initialize from flow curve (only sets G_0, tau_b_0, eta_s)
            self.initialize_from_flow_curve(np.asarray(x), np.asarray(y))
        elif test_mode == "oscillation":
            # Initialize from SAOS (only sets G_0, tau_b_0, eta_s)
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

        G_modes, tau_modes = self._get_mode_arrays()
        logger.info(
            f"Fitted TNTMultiSpecies (N={self._n_species}): "
            f"G_total={float(jnp.sum(G_modes)):.2e} Pa, "
            f"tau_range=[{float(jnp.min(tau_modes)):.2e}, {float(jnp.max(tau_modes)):.2e}] s, "
            f"η_s={self.eta_s:.2e} Pa·s"
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

        # Build parameter array from ParameterSet (ordering: G_0, tau_b_0, ..., eta_s)
        param_values = [
            float(self.parameters.get_value(name)) for name in self.parameters.keys()
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
            Parameter values: [G_0, tau_b_0, G_1, tau_b_1, ..., G_{N-1},
            tau_b_{N-1}, eta_s]
            Total length: 2*N + 1
        test_mode : str, optional
            Override stored test mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack parameters
        N = self._n_species
        G_modes = params[0 : 2 * N : 2]  # G_0, G_1, ..., G_{N-1}
        tau_modes = params[1 : 2 * N : 2]  # tau_b_0, tau_b_1, ..., tau_b_{N-1}
        eta_s = params[2 * N]

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._predict_flow_curve_internal(X_jax, G_modes, tau_modes, eta_s)

        elif mode == "oscillation":
            G_prime, G_double_prime = tnt_multimode_saos_moduli_vec(
                X_jax, G_modes, tau_modes, eta_s
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._simulate_startup_internal(
                X_jax, G_modes, tau_modes, eta_s, gamma_dot
            )

        elif mode == "relaxation":
            gamma_dot = self._gamma_dot_applied
            if gamma_dot is None:
                raise ValueError("relaxation mode requires gamma_dot (pre-shear rate)")
            return self._simulate_relaxation_internal(
                X_jax, G_modes, tau_modes, eta_s, gamma_dot
            )

        elif mode == "creep":
            sigma_applied = self._sigma_applied
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G_modes, tau_modes, eta_s, sigma_applied
            )

        elif mode == "laos":
            if self._gamma_0 is None or self._omega_laos is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G_modes, tau_modes, eta_s, self._gamma_0, self._omega_laos
            )
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return self._predict_flow_curve_internal(X_jax, G_modes, tau_modes, eta_s)

    # =========================================================================
    # Analytical Predictions
    # =========================================================================

    def _predict_flow_curve_internal(
        self,
        gamma_dot: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_modes: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Analytical steady shear stress for multi-species TNT.

        σ = Σ G_i·τ_b_i·γ̇ + η_s·γ̇ = η₀·γ̇

        For constant breakage (UCM), the conformation tensor steady state
        gives S_xy = τ·γ̇, hence σ_xy = G·τ·γ̇. This is Newtonian
        (no shear thinning), consistent with single-mode TNT.

        Parameters
        ----------
        gamma_dot : jnp.ndarray
            Shear rate array (1/s)
        G_modes : jnp.ndarray
            Mode moduli, shape (N,)
        tau_modes : jnp.ndarray
            Mode relaxation times, shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Shear stress σ (Pa)
        """
        # η₀ = Σ G_i·τ_b_i + η_s
        eta_0 = jnp.sum(G_modes * tau_modes) + eta_s
        return eta_0 * gamma_dot

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict steady shear stress and viscosity.

        Analytical superposition: σ = Σ G_i·τ_b_i·γ̇ / (1 + (τ_b_i·γ̇)²) + η_s·γ̇

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
        G_modes, tau_modes = self._get_mode_arrays()

        sigma = self._predict_flow_curve_internal(gd, G_modes, tau_modes, self.eta_s)

        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            # N1 = Σ 2·G_i·(τ_b_i·γ̇)² / (1 + (τ_b_i·γ̇)²)²
            wi = tau_modes[:, None] * gd[None, :]  # (N, M)
            wi2 = wi * wi
            denom2 = (1.0 + wi2) ** 2
            N1 = jnp.sum(2.0 * G_modes[:, None] * wi2 / denom2, axis=0)
            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)

        return np.asarray(sigma)

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        Analytical superposition:
            G'(ω) = Σ G_i·(ωτ_i)² / (1 + (ωτ_i)²)
            G''(ω) = Σ G_i·(ωτ_i) / (1 + (ωτ_i)²) + η_s·ω

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
        G_modes, tau_modes = self._get_mode_arrays()

        G_prime, G_double_prime = tnt_multimode_saos_moduli_vec(
            w, G_modes, tau_modes, self.eta_s
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

        For multi-mode UCM (conformation tensor):
            N₁ = Σ 2·G_i·(τ_b_i·γ̇)²
            N₂ = 0 (upper-convected derivative)

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
        G_modes, tau_modes = self._get_mode_arrays()

        # N1 = Σ 2·G_i·(τ_b_i·γ̇)²  (UCM conformation: S_xx - S_yy = 2(τγ̇)²)
        wi = tau_modes[:, None] * gd[None, :]  # (N, M)
        wi2 = wi * wi
        N1 = jnp.sum(2.0 * G_modes[:, None] * wi2, axis=0)
        N2 = jnp.zeros_like(N1)

        return np.asarray(N1), np.asarray(N2)

    # =========================================================================
    # ODE-Based Internal Simulations (for model_function)
    # =========================================================================

    def _simulate_startup_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_modes: jnp.ndarray,
        eta_s: float,
        gamma_dot: float,
    ) -> jnp.ndarray:
        """Internal startup simulation for model_function.

        Returns total shear stress σ_xy(t) = Σ G_i·S_xy_i + η_s·γ̇.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_modes : jnp.ndarray
            Mode relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)
        gamma_dot : float
            Applied shear rate (1/s)

        Returns
        -------
        jnp.ndarray
            Shear stress σ(t) (Pa)
        """

        def ode_fn(ti, yi, args):
            return tnt_multimode_ode_rhs(
                ti, yi, args["gamma_dot"], args["G_modes"], args["tau_modes"]
            )

        args = {"gamma_dot": gamma_dot, "G_modes": G_modes, "tau_modes": tau_modes}

        # Initial state: all modes at equilibrium [1,1,1,0, ...]
        y0 = self.get_equilibrium_conformation_multimode()

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
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
            max_steps=500_000,
        )

        # Extract S_xy_i for each mode (index 3, 7, 11, ...)
        N = self._n_species
        S_xy_modes = sol.ys[:, 3::4]  # (T, N)

        # Total stress: σ = Σ G_i·S_xy_i + η_s·γ̇
        sigma = jnp.sum(G_modes[None, :] * S_xy_modes, axis=1) + eta_s * gamma_dot
        return sigma

    def _simulate_relaxation_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_modes: jnp.ndarray,
        eta_s: float,
        gamma_dot_preshear: float,
    ) -> jnp.ndarray:
        """Internal relaxation simulation for model_function.

        Analytical multi-mode relaxation:
            σ(t) = Σ σ₀_i·exp(-t/τ_b_i)
        where σ₀_i = G_i·τ_b_i·γ̇ / (1 + (τ_b_i·γ̇)²)

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        G_modes : jnp.ndarray
            Mode moduli, shape (N,)
        tau_modes : jnp.ndarray
            Mode relaxation times, shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)
        gamma_dot_preshear : float
            Pre-shear rate (1/s)

        Returns
        -------
        jnp.ndarray
            Relaxing stress σ(t) (Pa)
        """
        # Initial stress per mode at steady state
        wi = tau_modes * gamma_dot_preshear
        sigma_0_modes = G_modes * wi / (1.0 + wi * wi)

        return tnt_multimode_relaxation_vec(t, sigma_0_modes, tau_modes)

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_modes: jnp.ndarray,
        eta_s: float,
        sigma_applied: float,
    ) -> jnp.ndarray:
        """Internal creep simulation for model_function.

        State: [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xy_{N-1}, γ] = 4N+1

        The applied stress is held constant:
            σ = Σ G_i·S_xy_i + η_s·γ̇
        so the shear rate is:
            γ̇ = (σ - Σ G_i·S_xy_i) / η_s

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        G_modes : jnp.ndarray
            Mode moduli, shape (N,)
        tau_modes : jnp.ndarray
            Mode relaxation times, shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)
        sigma_applied : float
            Applied constant stress (Pa)

        Returns
        -------
        jnp.ndarray
            Strain γ(t)
        """

        def ode_fn(ti, yi, args):
            # Unpack state: [S_modes..., gamma]
            N = args["G_modes"].shape[0]
            S_state = yi[: 4 * N]
            gamma = yi[4 * N]

            # Compute elastic stress contribution from each mode
            S_xy_modes = S_state[3::4]  # Extract S_xy_i
            sigma_elastic = jnp.sum(args["G_modes"] * S_xy_modes)

            # Compute shear rate from stress constraint
            eta_s_reg = jnp.maximum(
                args["eta_s"], 1e-10 * jnp.max(args["G_modes"] * args["tau_modes"])
            )
            gamma_dot = (args["sigma_applied"] - sigma_elastic) / eta_s_reg

            # Conformation evolution (multimode)
            d_S = tnt_multimode_ode_rhs(
                ti, S_state, gamma_dot, args["G_modes"], args["tau_modes"]
            )

            # Strain evolution
            d_gamma = gamma_dot

            return jnp.concatenate([d_S, jnp.array([d_gamma])])

        args = {
            "sigma_applied": sigma_applied,
            "G_modes": G_modes,
            "tau_modes": tau_modes,
            "eta_s": eta_s,
        }

        # Initial state: all modes at equilibrium + zero strain
        y0_conf = self.get_equilibrium_conformation_multimode()
        y0 = jnp.concatenate([y0_conf, jnp.array([0.0])])

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
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
            max_steps=500_000,
        )

        return sol.ys[:, -1]  # γ (last component)

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_modes: jnp.ndarray,
        eta_s: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function.

        State: [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xy_{N-1}] = 4N

        Oscillatory shear: γ̇(t) = γ₀·ω·cos(ωt)

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        G_modes : jnp.ndarray
            Mode moduli, shape (N,)
        tau_modes : jnp.ndarray
            Mode relaxation times, shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            (strain, stress) arrays
        """

        def ode_fn(ti, yi, args):
            gamma_dot = args["gamma_0"] * args["omega"] * jnp.cos(args["omega"] * ti)
            return tnt_multimode_ode_rhs(
                ti, yi, gamma_dot, args["G_modes"], args["tau_modes"]
            )

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "G_modes": G_modes,
            "tau_modes": tau_modes,
        }

        y0 = self.get_equilibrium_conformation_multimode()

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t[0])
        t1 = float(t[-1])
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
            max_steps=500_000,
        )

        # Extract S_xy_i for each mode
        S_xy_modes = sol.ys[:, 3::4]  # (T, N)

        # Compute strain and stress
        strain = gamma_0 * jnp.sin(omega * t)
        gamma_dot_t = gamma_0 * omega * jnp.cos(omega * t)
        stress = jnp.sum(G_modes[None, :] * S_xy_modes, axis=1) + eta_s * gamma_dot_t

        return strain, stress

    # =========================================================================
    # Public Simulation Methods (return numpy arrays)
    # =========================================================================

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
            If True, return full conformation tensor for all modes

        Returns
        -------
        np.ndarray or dict
            Shear stress σ(t), or dict with 'S_xx', 'S_yy', 'S_xy', 'S_zz'
            (each shape (T, N)) if return_full=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        G_modes, tau_modes = self._get_mode_arrays()

        def ode_fn(ti, yi, args):
            return tnt_multimode_ode_rhs(
                ti, yi, args["gamma_dot"], args["G_modes"], args["tau_modes"]
            )

        args = {"gamma_dot": gamma_dot, "G_modes": G_modes, "tau_modes": tau_modes}
        y0 = self.get_equilibrium_conformation_multimode()

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
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
            max_steps=500_000,
        )

        # Extract mode components: S_xx_i at indices 0,4,8,...
        N = self._n_species
        S_xx_modes = sol.ys[:, 0::4]  # (T, N)
        S_yy_modes = sol.ys[:, 1::4]
        S_zz_modes = sol.ys[:, 2::4]
        S_xy_modes = sol.ys[:, 3::4]

        self._trajectory = {
            "t": np.asarray(t_jax),
            "S_xx": np.asarray(S_xx_modes),
            "S_yy": np.asarray(S_yy_modes),
            "S_zz": np.asarray(S_zz_modes),
            "S_xy": np.asarray(S_xy_modes),
        }

        if return_full:
            return {
                "S_xx": np.asarray(S_xx_modes),
                "S_yy": np.asarray(S_yy_modes),
                "S_xy": np.asarray(S_xy_modes),
                "S_zz": np.asarray(S_zz_modes),
            }

        # Total stress: σ = Σ G_i·S_xy_i + η_s·γ̇
        sigma = jnp.sum(G_modes[None, :] * S_xy_modes, axis=1) + self.eta_s * gamma_dot
        return np.asarray(sigma)

    def simulate_relaxation(
        self,
        t: np.ndarray,
        gamma_dot_preshear: float,
    ) -> np.ndarray:
        """Simulate stress relaxation after cessation of steady shear.

        Analytical multi-mode relaxation:
            σ(t) = Σ σ₀_i·exp(-t/τ_b_i)

        Parameters
        ----------
        t : np.ndarray
            Time array (s), starting from t=0 (cessation)
        gamma_dot_preshear : float
            Shear rate before cessation (1/s)

        Returns
        -------
        np.ndarray
            Relaxing stress σ(t) (Pa)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        G_modes, tau_modes = self._get_mode_arrays()

        # Initial stress per mode from steady-state
        wi = tau_modes * gamma_dot_preshear
        sigma_0_modes = G_modes * wi / (1.0 + wi * wi)

        sigma = tnt_multimode_relaxation_vec(t_jax, sigma_0_modes, tau_modes)

        self._trajectory = {
            "t": np.asarray(t_jax),
            "sigma": np.asarray(sigma),
        }

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
        G_modes, tau_modes = self._get_mode_arrays()

        def ode_fn(ti, yi, args):
            N = args["G_modes"].shape[0]
            S_state = yi[: 4 * N]
            gamma = yi[4 * N]

            # Elastic stress
            S_xy_modes = S_state[3::4]
            sigma_elastic = jnp.sum(args["G_modes"] * S_xy_modes)

            # Shear rate
            eta_s_reg = jnp.maximum(
                args["eta_s"], 1e-10 * jnp.max(args["G_modes"] * args["tau_modes"])
            )
            gamma_dot = (args["sigma_applied"] - sigma_elastic) / eta_s_reg

            # Conformation evolution
            d_S = tnt_multimode_ode_rhs(
                ti, S_state, gamma_dot, args["G_modes"], args["tau_modes"]
            )

            return jnp.concatenate([d_S, jnp.array([gamma_dot])])

        args = {
            "sigma_applied": sigma_applied,
            "G_modes": G_modes,
            "tau_modes": tau_modes,
            "eta_s": self.eta_s,
        }

        y0_conf = self.get_equilibrium_conformation_multimode()
        y0 = jnp.concatenate([y0_conf, jnp.array([0.0])])

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        t0 = float(t_jax[0])
        t1 = float(t_jax[-1])
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
            max_steps=500_000,
        )

        gamma = np.asarray(sol.ys[:, -1])

        self._trajectory = {
            "t": np.asarray(t_jax),
            "gamma": gamma,
        }

        if return_rate:
            # Extract S_xy modes
            N = self._n_species
            S_xy_modes = sol.ys[:, 3::4]
            sigma_elastic = jnp.sum(G_modes[None, :] * S_xy_modes, axis=1)
            eta_s_reg = max(self.eta_s, 1e-10 * float(jnp.max(G_modes * tau_modes)))
            gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg
            return gamma, np.asarray(gamma_dot)

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
        G_modes, tau_modes = self._get_mode_arrays()

        strain, stress = self._simulate_laos_internal(
            t_jax, G_modes, tau_modes, self.eta_s, gamma_0, omega
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

    def get_relaxation_spectrum(
        self,
        t: np.ndarray | None = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get relaxation modulus G(t).

        For multi-species TNT: G(t) = Σ G_i·exp(-t/τ_b_i)

        Parameters
        ----------
        t : np.ndarray, optional
            Time array (default: logspace from 0.01·min(τ) to 100·max(τ))
        n_points : int, default 100
            Number of points if t not provided

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (t, G(t))
        """
        G_modes, tau_modes = self._get_mode_arrays()

        if t is None:
            t_min = 0.01 * float(jnp.min(tau_modes))
            t_max = 100 * float(jnp.max(tau_modes))
            t = np.logspace(np.log10(t_min), np.log10(t_max), n_points)

        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # G(t) = Σ G_i·exp(-t/τ_i)
        G_t = jnp.sum(
            G_modes[:, None] * jnp.exp(-t_jax[None, :] / tau_modes[:, None]), axis=0
        )

        return t, np.asarray(G_t)

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
        omega = 2 * np.pi * np.abs(freqs[np.argmax(np.abs(fft_strain[1:])) + 1])

        harmonics = [2 * i + 1 for i in range(n_harmonics)]
        sigma_prime = []
        sigma_double_prime = []

        for n in harmonics:
            sin_basis = np.sin(n * omega * t)
            cos_basis = np.cos(n * omega * t)

            dt = t[1] - t[0]
            sigma_n_prime = 2 * np.trapezoid(stress * sin_basis, dx=dt) / (t[-1] - t[0])
            sigma_n_double_prime = (
                2 * np.trapezoid(stress * cos_basis, dx=dt) / (t[-1] - t[0])
            )

            sigma_prime.append(sigma_n_prime)
            sigma_double_prime.append(sigma_n_double_prime)

        sigma_prime = np.array(sigma_prime)
        sigma_double_prime = np.array(sigma_double_prime)
        intensity = np.sqrt(sigma_prime**2 + sigma_double_prime**2)

        return {
            "n": np.array(harmonics),
            "sigma_prime": sigma_prime,
            "sigma_double_prime": sigma_double_prime,
            "intensity": intensity,
            "I3_I1": intensity[1] / intensity[0] if intensity[0] > 0 else 0.0,
        }

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        G_modes, tau_modes = self._get_mode_arrays()
        G_total = float(jnp.sum(G_modes))
        tau_min = float(jnp.min(tau_modes))
        tau_max = float(jnp.max(tau_modes))
        return (
            f"TNTMultiSpecies(n_species={self._n_species}, "
            f"G_total={G_total:.2e} Pa, "
            f"tau_range=[{tau_min:.2e}, {tau_max:.2e}] s, "
            f"η_s={self.eta_s:.2e} Pa·s)"
        )
