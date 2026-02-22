"""Sticky Rouse model for associative polymers with sticker dynamics.

This module implements the Sticky Rouse model, which describes polymers with
multiple "sticker" groups along the chain that reversibly associate to form
temporary crosslinks.

Key Physics
-----------
The Sticky Rouse model combines Rouse chain dynamics with sticker-mediated
associations:

- Each Rouse mode k has a natural relaxation time τ_R_k
- Sticker association imposes a lifetime τ_s (the sticker lifetime)
- Effective mode relaxation: τ_eff_k = max(τ_R_k, τ_s)

This creates a characteristic plateau in G(t) at intermediate times when
τ_s dominates mode relaxation. Fast modes (τ_R_k < τ_s) are slowed by
sticker lifetime, while slow modes (τ_R_k > τ_s) relax at their natural rate.

The model is essentially a multi-mode Maxwell with mode-dependent effective
relaxation times constrained by sticker kinetics.

Physical Motivation
-------------------
Associative polymers include:
- Ionomers with ionic stickers
- Supramolecular polymers with hydrogen bonds
- Hydrogels with multiple crosslink types

The sticker lifetime τ_s sets a minimum relaxation time floor. Rouse modes
faster than sticker opening cannot fully relax — they are frozen by sticker
association until breakage events allow chain rearrangement.

Mathematical Framework
----------------------
Multi-mode Maxwell constitutive equation for each mode k::

    dS_k/dt = L·S_k + S_k·L^T + (1/τ_eff_k)·I - (1/τ_eff_k)·S_k

where τ_eff_k = max(τ_R_k, τ_s).

Total stress is the sum over all modes plus solvent contribution::

    σ = Σ G_k·S_xy_k + η_s·γ̇

State Vector
------------
For N modes, the state vector has 4*N components::

    [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xx_{N-1}, ..., S_xy_{N-1}]

Equilibrium: all modes at S = I → [1, 1, 1, 0, 1, 1, 1, 0, ..., 1, 1, 1, 0]

Parameters
----------
For N modes, we have 2*N + 2 parameters:

Per-mode (k = 0 to N-1):
    - G_k: Mode modulus (Pa)
    - tau_R_k: Rouse relaxation time (s)

Global:
    - tau_s: Sticker lifetime (s)
    - eta_s: Solvent viscosity (Pa·s)

Derived quantities:
    - tau_eff_k = max(tau_R_k, tau_s) for each mode

Default Mode Spacing
--------------------
By default, Rouse times are logarithmically spaced::

    tau_R_k = 10.0^(1-k) for k = 0, 1, 2, ...

So for n_modes=3:
    - Mode 0: tau_R_0 = 10.0 s
    - Mode 1: tau_R_1 = 1.0 s
    - Mode 2: tau_R_2 = 0.1 s

The sticker lifetime tau_s (default 0.1 s) then determines which modes
experience the plateau.

Supported Protocols
-------------------
1. **flow_curve**: Analytical steady-state shear stress
2. **oscillation**: Analytical SAOS moduli (G', G'')
3. **startup**: ODE-based transient startup to steady shear
4. **relaxation**: Analytical multi-exponential stress relaxation
5. **creep**: ODE-based stress-controlled creep compliance
6. **laos**: ODE-based large-amplitude oscillatory shear

References
----------
- Leibler, L., Rubinstein, M., & Colby, R.H. (1991). Macromolecules 24, 4701.
- Rubinstein, M. & Semenov, A.N. (2001). Macromolecules 34, 1058-1068.
- Semenov, A.N. & Rubinstein, M. (1998). Macromolecules 31, 1373-1385.
"""

from __future__ import annotations

import logging

import numpy as np

from rheojax.core.jax_config import lazy_import, safe_import_jax

diffrax = lazy_import("diffrax")
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.tnt._base import TNTBase
from rheojax.models.tnt._kernels import (
    tnt_multimode_ode_rhs,
    tnt_multimode_relaxation_vec,
    tnt_multimode_saos_moduli_vec,
)
from rheojax.utils.optimization import nlsq_curve_fit

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)

_MISSING = object()


@ModelRegistry.register(
    "tnt_sticky_rouse",
    protocols=["flow_curve", "oscillation", "startup", "relaxation", "creep", "laos"],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class TNTStickyRouse(TNTBase):
    """Sticky Rouse model for associative polymers.

    Multi-mode Maxwell model where sticker dynamics impose a relaxation time
    floor: τ_eff_k = max(τ_R_k, τ_s).

    Creates a plateau in G(t) at intermediate times (sticker-dominated regime)
    before terminal relaxation (slowest Rouse mode).

    Parameters
    ----------
    n_modes : int, default 3
        Number of Rouse modes

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters:
        - G_0, G_1, ..., G_{N-1}: Mode moduli (Pa)
        - tau_R_0, tau_R_1, ..., tau_R_{N-1}: Rouse relaxation times (s)
        - tau_s: Sticker lifetime (s)
        - eta_s: Solvent viscosity (Pa·s)

    Notes
    -----
    The model reduces to standard multi-mode Maxwell when tau_s → 0.
    For tau_s → ∞, all modes relax at tau_s (single network behavior).

    Examples
    --------
    >>> # 3-mode sticky Rouse
    >>> model = TNTStickyRouse(n_modes=3)
    >>> model.fit(omega, G_star, test_mode='oscillation')
    >>>
    >>> # Predict plateau modulus
    >>> G_plateau = model.predict_plateau_modulus()
    >>>
    >>> # Predict startup with stress overshoot
    >>> t = np.linspace(0, 10, 200)
    >>> sigma = model.predict(t, test_mode='startup', gamma_dot=1.0)
    >>>
    >>> # Extract effective relaxation times
    >>> tau_eff = model.get_effective_times()
    """

    def __init__(self, n_modes: int = 3):
        """Initialize Sticky Rouse model.

        Parameters
        ----------
        n_modes : int, default 3
            Number of Rouse modes (must be >= 1)
        """
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")

        self._n_modes = n_modes
        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_modes(self) -> int:
        """Number of Rouse modes."""
        return self._n_modes

    @property
    def tau_s(self) -> float:
        """Sticker lifetime (s)."""
        val = self.parameters.get_value("tau_s")
        return float(val) if val is not None else 0.0

    @property
    def eta_s(self) -> float:
        """Solvent viscosity (Pa·s)."""
        val = self.parameters.get_value("eta_s")
        return float(val) if val is not None else 0.0

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize parameters for N-mode Sticky Rouse model.

        Creates 2*N + 2 parameters:
        - G_k: Mode moduli (1e3/N Pa default, bounds [1e0, 1e8])
        - tau_R_k: Rouse times (10^(1-k) s default, bounds [1e-6, 1e4])
        - tau_s: Sticker lifetime (0.1 s default, bounds [1e-6, 1e4])
        - eta_s: Solvent viscosity (0.0 Pa·s default, bounds [0.0, 1e4])
        """
        self.parameters = ParameterSet()

        # Default modulus per mode (equal weight by default)
        G_default = 1e3 / self._n_modes

        # Mode parameters: G_k and tau_R_k interleaved for k = 0, ..., N-1
        for k in range(self._n_modes):
            # Rouse time: logarithmic spacing (10^(1-k))
            tau_R_default = 10.0 ** (1 - k)

            self.parameters.add(
                name=f"G_{k}",
                value=G_default,
                bounds=(1e0, 1e8),
                description=f"Mode {k} modulus (Pa)",
            )
            self.parameters.add(
                name=f"tau_R_{k}",
                value=tau_R_default,
                bounds=(1e-6, 1e4),
                description=f"Mode {k} Rouse relaxation time (s)",
            )

        # Global sticker parameters
        self.parameters.add(
            name="tau_s",
            value=0.1,
            bounds=(1e-6, 1e4),
            description="Sticker lifetime (s)",
        )
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            description="Solvent viscosity (Pa·s)",
        )

    # =========================================================================
    # Helper: Extract Mode Arrays
    # =========================================================================

    def _get_mode_arrays(self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Extract mode parameters as JAX arrays.

        Returns
        -------
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_R : jnp.ndarray
            Rouse relaxation times (s), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        """
        G_vals = []
        for k in range(self._n_modes):
            val = self.parameters.get_value(f"G_{k}")
            G_vals.append(float(val) if val is not None else 0.0)
        G_modes = jnp.array(G_vals)
        tau_vals = []
        for k in range(self._n_modes):
            val = self.parameters.get_value(f"tau_R_{k}")
            tau_vals.append(float(val) if val is not None else 0.0)
        tau_R = jnp.array(tau_vals)
        tau_s_val = self.parameters.get_value("tau_s")
        tau_s = float(tau_s_val) if tau_s_val is not None else 0.0
        tau_eff = jnp.maximum(tau_R, tau_s)
        return G_modes, tau_R, tau_eff

    def get_effective_times(self) -> np.ndarray:
        """Get effective relaxation times for all modes.

        Returns
        -------
        np.ndarray
            Effective times τ_eff_k = max(τ_R_k, τ_s), shape (N,)
        """
        _, _, tau_eff = self._get_mode_arrays()
        return np.asarray(tau_eff)

    # =========================================================================
    # Model Function
    # =========================================================================

    def model_function(
        self,
        X: jnp.ndarray,
        params: jnp.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> jnp.ndarray:
        """Compute model prediction for given parameters.

        Parameters
        ----------
        X : jnp.ndarray
            Independent variable (time, frequency, or shear rate)
        params : jnp.ndarray
            Parameter array [G_0, tau_R_0, G_1, tau_R_1, ..., tau_s, eta_s]
            Length: 2*N + 2
        test_mode : str or None
            Protocol: 'oscillation', 'relaxation', 'flow_curve', 'startup',
            'creep', or 'laos'

        Returns
        -------
        jnp.ndarray
            Predicted response (protocol-dependent)
        """
        N = self._n_modes

        # Extract parameters
        G_modes = params[0 : 2 * N : 2]
        tau_R_modes = params[1 : 2 * N : 2]
        tau_s = params[2 * N]
        eta_s = params[2 * N + 1]

        # Effective times: floor at sticker lifetime
        tau_eff = jnp.maximum(tau_R_modes, tau_s)

        # Resolve test mode with fallback
        mode = test_mode or self._test_mode or "flow_curve"
        # Use sentinel pattern to avoid swallowing falsy values (e.g. gamma_dot=0.0)
        _gd = kwargs.get("gamma_dot", _MISSING)
        gamma_dot = _gd if _gd is not _MISSING else getattr(self, "_gamma_dot_applied", None)
        _sa = kwargs.get("sigma_applied", _MISSING)
        sigma_applied = _sa if _sa is not _MISSING else getattr(self, "_sigma_applied", None)
        _g0 = kwargs.get("gamma_0", _MISSING)
        gamma_0 = _g0 if _g0 is not _MISSING else getattr(self, "_gamma_0", None)
        _om = kwargs.get("omega", _MISSING)
        omega = _om if _om is not _MISSING else getattr(self, "_omega_laos", None)

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Dispatch to protocol-specific prediction
        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._predict_flow_curve_vec(X_jax, G_modes, tau_eff, eta_s)
        elif mode == "oscillation":
            # Return |G*| magnitude for fitting/Bayesian inference
            # (complex values not supported by JAX grad)
            G_star = self._predict_oscillation_vec(X_jax, G_modes, tau_eff, eta_s)
            return jnp.column_stack([jnp.real(G_star), jnp.imag(G_star)])
        elif mode == "relaxation":
            # Need initial stress per mode (from fitting context)
            if not hasattr(self, "_sigma_0_modes") or self._sigma_0_modes is None:
                # Default: equal stress per mode
                sigma_0 = 1e3  # Pa
                sigma_0_modes = jnp.ones(N) * (sigma_0 / N)
            else:
                sigma_0_modes = self._sigma_0_modes
            return self._predict_relaxation_vec(X_jax, sigma_0_modes, tau_eff)
        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return self._predict_startup(X_jax, gamma_dot, G_modes, tau_eff, eta_s)
        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._predict_creep(
                X_jax, sigma_applied, G_modes, tau_eff, eta_s
            )
        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            return self._predict_laos(
                X_jax, gamma_0, omega, G_modes, tau_eff, eta_s
            )
        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return self._predict_flow_curve_vec(X_jax, G_modes, tau_eff, eta_s)

    # =========================================================================
    # Analytical Predictions
    # =========================================================================

    def _predict_oscillation_vec(
        self,
        omega: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Predict complex modulus G*(ω) for SAOS (vectorized).

        Parameters
        ----------
        omega : jnp.ndarray
            Angular frequency array (rad/s)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Complex modulus G' + 1j*G'', shape (len(omega),)
        """
        G_prime_arr, G_double_prime_arr = tnt_multimode_saos_moduli_vec(
            omega, G_modes, tau_eff, eta_s
        )
        return G_prime_arr + 1j * G_double_prime_arr

    def _predict_relaxation_vec(
        self,
        t: jnp.ndarray,
        sigma_0_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predict stress relaxation σ(t) (vectorized).

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        sigma_0_modes : jnp.ndarray
            Initial stress per mode (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)

        Returns
        -------
        jnp.ndarray
            Relaxing stress σ(t) (Pa), shape (len(t),)
        """
        return tnt_multimode_relaxation_vec(t, sigma_0_modes, tau_eff)

    def _predict_flow_curve_vec(
        self,
        gamma_dot: jnp.ndarray,
        G_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Predict steady shear stress σ(γ̇) (vectorized).

        For UCM conformation tensor formulation, the steady-state shear stress
        is Newtonian: σ = η₀·γ̇ where η₀ = Σ G_k·τ_eff_k + η_s.

        Parameters
        ----------
        gamma_dot : jnp.ndarray
            Shear rate array (1/s)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Steady shear stress (Pa), shape (len(gamma_dot),)
        """
        eta_0 = jnp.sum(G_modes * tau_eff) + eta_s
        return eta_0 * gamma_dot

    # =========================================================================
    # Fitting
    # =========================================================================

    def _fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> TNTStickyRouse:
        """Fit model to data using NLSQ optimization.

        Parameters
        ----------
        x : np.ndarray
            Independent variable (time, frequency, or shear rate)
        y : np.ndarray
            Dependent variable (stress, modulus, or complex modulus)
        **kwargs : dict
            Optional keyword arguments:
            - test_mode: Protocol ('oscillation', 'relaxation', 'flow_curve')
        """
        test_mode = kwargs.get("test_mode", self._test_mode)
        if test_mode is None:
            raise ValueError("test_mode must be specified for fitting")

        # Store protocol-specific inputs
        self._gamma_dot_applied = kwargs.get("gamma_dot")
        self._sigma_applied = kwargs.get("sigma_applied")
        self._gamma_0 = kwargs.get("gamma_0")
        self._omega_laos = kwargs.get("omega")

        # Convert to JAX arrays
        x_jax = jnp.asarray(X, dtype=jnp.float64)
        y_jax = jnp.asarray(y, dtype=jnp.float64)

        # For relaxation, store initial stress distribution
        if test_mode == "relaxation":
            sigma_0 = float(y[0])  # Initial stress
            G_modes, _, _ = self._get_mode_arrays()
            # Equal stress per mode initially
            self._sigma_0_modes = jnp.ones(self._n_modes) * (
                sigma_0 / jnp.sum(G_modes) * G_modes
            )

        # Build objective function
        def objective(params_array):
            y_pred = self.model_function(x_jax, params_array, test_mode=test_mode)
            if jnp.iscomplexobj(y_jax):
                # Complex fitting (oscillation)
                residuals = jnp.concatenate(
                    [jnp.real(y_pred - y_jax), jnp.imag(y_pred - y_jax)]
                )
            else:
                residuals = y_pred - y_jax
            return residuals

        # Run NLSQ optimization
        result = nlsq_curve_fit(
            self.model_function,
            x_jax,
            y_jax,
            self.parameters,
            test_mode=test_mode,
        )

        # nlsq_curve_fit already updates ParameterSet in-place

        # Store fit statistics
        self._fit_result = result
        self._r_squared = result.r_squared

        logger.info(
            f"Sticky Rouse fit complete: R²={self._r_squared:.4f}, "
            f"n_modes={self._n_modes}"
        )
        return self

    # =========================================================================
    # Prediction
    # =========================================================================

    def _predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict response for given input.

        Parameters
        ----------
        x : np.ndarray
            Independent variable (time, frequency, or shear rate)
        **kwargs : dict
            Optional keyword arguments:
            - test_mode: Protocol
            - gamma_dot: Shear rate for startup (1/s)
            - sigma_applied: Applied stress for creep (Pa)
            - gamma_0: Strain amplitude for LAOS
            - omega: Angular frequency for LAOS (rad/s)

        Returns
        -------
        np.ndarray
            Predicted response
        """
        test_mode = kwargs.get("test_mode", self._test_mode)
        if test_mode is None:
            raise ValueError("test_mode must be specified for prediction")

        # Get mode parameters
        G_modes, tau_R, tau_eff = self._get_mode_arrays()
        eta_s = self.eta_s

        x_jax = jnp.asarray(X, dtype=jnp.float64)

        # Dispatch by protocol
        if test_mode == "oscillation":
            result = self._predict_oscillation_vec(x_jax, G_modes, tau_eff, eta_s)
        elif test_mode == "flow_curve":
            result = self._predict_flow_curve_vec(x_jax, G_modes, tau_eff, eta_s)
        elif test_mode == "relaxation":
            # Initial stress distribution
            if not hasattr(self, "_sigma_0_modes") or self._sigma_0_modes is None:
                sigma_0 = kwargs.get("sigma_0", 1e3)
                sigma_0_modes = jnp.ones(self._n_modes) * (
                    sigma_0 / jnp.sum(G_modes) * G_modes
                )
            else:
                sigma_0_modes = self._sigma_0_modes
            result = self._predict_relaxation_vec(x_jax, sigma_0_modes, tau_eff)
        elif test_mode == "startup":
            _gd = kwargs.get("gamma_dot", _MISSING)
            gamma_dot = _gd if _gd is not _MISSING else getattr(self, "_gamma_dot_applied", None)
            if gamma_dot is None:
                raise ValueError("gamma_dot must be provided for startup")
            self._gamma_dot_applied = gamma_dot
            result = self._predict_startup(x_jax, gamma_dot, G_modes, tau_eff, eta_s)
        elif test_mode == "creep":
            _sa = kwargs.get("sigma_applied", _MISSING)
            sigma_applied = _sa if _sa is not _MISSING else getattr(self, "_sigma_applied", None)
            if sigma_applied is None:
                raise ValueError("sigma_applied must be provided for creep")
            self._sigma_applied = sigma_applied
            result = self._predict_creep(x_jax, sigma_applied, G_modes, tau_eff, eta_s)
        elif test_mode == "laos":
            _g0 = kwargs.get("gamma_0", _MISSING)
            gamma_0 = _g0 if _g0 is not _MISSING else getattr(self, "_gamma_0", None)
            _om = kwargs.get("omega", _MISSING)
            omega = _om if _om is not _MISSING else getattr(self, "_omega_laos", None)
            if gamma_0 is None or omega is None:
                raise ValueError("gamma_0 and omega must be provided for LAOS")
            self._gamma_0 = gamma_0
            self._omega_laos = omega
            result = self._predict_laos(x_jax, gamma_0, omega, G_modes, tau_eff, eta_s)
        else:
            raise ValueError(f"Unsupported test_mode: {test_mode}")

        return np.asarray(result)

    # =========================================================================
    # ODE-Based Transient Protocols
    # =========================================================================

    def _predict_startup(
        self,
        t: jnp.ndarray,
        gamma_dot: float,
        G_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Predict startup shear stress σ(t) via ODE integration.

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Transient shear stress σ(t) (Pa), shape (len(t),)
        """
        N = self._n_modes

        # Initial state: all modes at equilibrium S = I
        y0 = jnp.tile(jnp.array([1.0, 1.0, 1.0, 0.0]), N)

        # ODE RHS
        def ode_rhs(t_val, state, args):
            return tnt_multimode_ode_rhs(t_val, state, gamma_dot, G_modes, tau_eff)

        # Solve ODE
        term = diffrax.ODETerm(ode_rhs)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=t)
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
            throw=False,
        )

        # Extract stress from solution
        states = solution.ys  # Shape: (len(t), 4*N)
        states_reshaped = states.reshape((len(t), N, 4))

        # Stress: σ = Σ G_k·S_xy_k + η_s·γ̇
        S_xy_modes = states_reshaped[:, :, 3]  # Shape: (len(t), N)
        sigma = jnp.sum(G_modes * S_xy_modes, axis=1) + eta_s * gamma_dot

        # Handle solver failures
        sigma = jnp.where(
            solution.result == diffrax.RESULTS.successful,
            sigma,
            jnp.nan * jnp.ones_like(sigma),
        )

        # Store trajectory
        self._trajectory = {
            "time": np.asarray(t),
            "stress": np.asarray(sigma),
            "S_xy": np.asarray(S_xy_modes),
        }

        return sigma

    def _predict_creep(
        self,
        t: jnp.ndarray,
        sigma_applied: float,
        G_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Predict creep compliance γ(t) via ODE integration.

        State: [S_xx_0, S_yy_0, S_zz_0, S_xy_0, ..., S_xy_{N-1}, γ]
        Total: 4*N + 1 components

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        sigma_applied : float
            Applied constant stress (Pa)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            Creep strain γ(t), shape (len(t),)
        """
        N = self._n_modes

        # Initial state: all modes at equilibrium, zero strain
        y0 = jnp.concatenate(
            [jnp.tile(jnp.array([1.0, 1.0, 1.0, 0.0]), N), jnp.array([0.0])]
        )

        # ODE RHS
        def ode_rhs(t_val, state, args):
            # Unpack state
            conf_state = state[: 4 * N]
            _gamma = state[4 * N]

            # Compute gamma_dot from stress constraint
            conf_reshaped = conf_state.reshape((N, 4))
            S_xy_modes = conf_reshaped[:, 3]
            sigma_elastic = jnp.sum(G_modes * S_xy_modes)

            eta_s_reg = jnp.maximum(eta_s, 1e-10 * jnp.sum(G_modes * tau_eff))
            gamma_dot = (sigma_applied - sigma_elastic) / eta_s_reg

            # Conformation evolution
            d_conf = tnt_multimode_ode_rhs(
                t_val, conf_state, gamma_dot, G_modes, tau_eff
            )

            # Strain evolution
            d_gamma = gamma_dot

            return jnp.concatenate([d_conf, jnp.array([d_gamma])])

        # Solve ODE
        term = diffrax.ODETerm(ode_rhs)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=t)
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
            throw=False,
        )

        # Extract strain
        gamma = solution.ys[:, 4 * N]

        # Handle solver failures
        gamma = jnp.where(
            solution.result == diffrax.RESULTS.successful,
            gamma,
            jnp.nan * jnp.ones_like(gamma),
        )

        # Store trajectory
        self._trajectory = {
            "time": np.asarray(t),
            "strain": np.asarray(gamma),
        }

        return gamma

    def _predict_laos(
        self,
        t: jnp.ndarray,
        gamma_0: float,
        omega: float,
        G_modes: jnp.ndarray,
        tau_eff: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Predict LAOS stress σ(t) via ODE integration.

        γ̇(t) = γ₀·ω·cos(ωt)

        Parameters
        ----------
        t : jnp.ndarray
            Time array (s)
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)
        G_modes : jnp.ndarray
            Mode moduli (Pa), shape (N,)
        tau_eff : jnp.ndarray
            Effective relaxation times (s), shape (N,)
        eta_s : float
            Solvent viscosity (Pa·s)

        Returns
        -------
        jnp.ndarray
            LAOS stress σ(t) (Pa), shape (len(t),)
        """
        N = self._n_modes

        # Initial state: all modes at equilibrium
        y0 = jnp.tile(jnp.array([1.0, 1.0, 1.0, 0.0]), N)

        # ODE RHS with oscillatory shear rate
        def ode_rhs(t_val, state, args):
            gamma_dot = gamma_0 * omega * jnp.cos(omega * t_val)
            return tnt_multimode_ode_rhs(t_val, state, gamma_dot, G_modes, tau_eff)

        # Solve ODE
        term = diffrax.ODETerm(ode_rhs)
        solver = diffrax.Tsit5()
        saveat = diffrax.SaveAt(ts=t)
        stepsize_controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t[0],
            t1=t[-1],
            dt0=None,
            y0=y0,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            max_steps=500_000,
            throw=False,
        )

        # Extract stress
        states = solution.ys  # Shape: (len(t), 4*N)
        states_reshaped = states.reshape((len(t), N, 4))

        S_xy_modes = states_reshaped[:, :, 3]
        gamma_dot_arr = gamma_0 * omega * jnp.cos(omega * t)
        sigma = jnp.sum(G_modes * S_xy_modes, axis=1) + eta_s * gamma_dot_arr

        # Handle solver failures
        sigma = jnp.where(
            solution.result == diffrax.RESULTS.successful,
            sigma,
            jnp.nan * jnp.ones_like(sigma),
        )

        # Store trajectory
        self._trajectory = {
            "time": np.asarray(t),
            "stress": np.asarray(sigma),
            "strain": np.asarray(gamma_0 * jnp.sin(omega * t)),
        }

        return sigma

    # =========================================================================
    # Post-Processing and Analysis
    # =========================================================================

    def predict_plateau_modulus(self) -> float:
        """Compute plateau modulus G_N = Σ G_k for modes with τ_R_k < τ_s.

        The plateau modulus is the sum of mode moduli for modes dominated
        by sticker lifetime (fast Rouse modes).

        Returns
        -------
        float
            Plateau modulus G_N (Pa)
        """
        G_modes, tau_R, _ = self._get_mode_arrays()
        tau_s = self.tau_s

        # Modes with tau_R < tau_s contribute to plateau
        plateau_mask = tau_R < tau_s
        G_plateau = float(jnp.sum(jnp.where(plateau_mask, G_modes, 0.0)))

        return G_plateau

    def predict_zero_shear_viscosity(self) -> float:
        """Compute zero-shear viscosity η₀ = Σ G_k·τ_eff_k + η_s.

        Returns
        -------
        float
            Zero-shear viscosity η₀ (Pa·s)
        """
        G_modes, _, tau_eff = self._get_mode_arrays()
        eta_s = self.eta_s
        eta_0 = float(jnp.sum(G_modes * tau_eff) + eta_s)
        return eta_0

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict SAOS storage and loss moduli.

        Analytical superposition for multi-mode Maxwell:
            G'(ω) = Σ G_k·(ωτ_eff_k)² / (1 + (ωτ_eff_k)²)
            G''(ω) = Σ G_k·(ωτ_eff_k) / (1 + (ωτ_eff_k)²) + η_s·ω

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
        G_modes, _, tau_eff = self._get_mode_arrays()

        G_prime, G_double_prime = tnt_multimode_saos_moduli_vec(
            w, G_modes, tau_eff, self.eta_s
        )

        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)

        G_star_mag = jnp.sqrt(jnp.maximum(G_prime**2 + G_double_prime**2, 1e-30))
        return np.asarray(G_star_mag)

    def predict_terminal_time(self) -> float:
        """Return longest effective relaxation time (terminal mode).

        Returns
        -------
        float
            Terminal time τ_terminal = max(τ_eff_k) (s)
        """
        _, _, tau_eff = self._get_mode_arrays()
        return float(jnp.max(tau_eff))

    def predict_normal_stress_difference(
        self, gamma_dot: float | np.ndarray
    ) -> np.ndarray:
        """Predict first normal stress difference N₁(γ̇).

        N₁ = Σ 2·G_k·τ_eff_k²·γ̇² / (1 + (τ_eff_k·γ̇)²)

        Parameters
        ----------
        gamma_dot : float or np.ndarray
            Shear rate (1/s)

        Returns
        -------
        np.ndarray
            N₁ (Pa)
        """
        G_modes, _, tau_eff = self._get_mode_arrays()

        gamma_dot = jnp.asarray(gamma_dot, dtype=jnp.float64)

        def compute_n1(gd):
            wi = tau_eff * gd
            wi2 = wi * wi
            return jnp.sum(2.0 * G_modes * wi2 / (1.0 + wi2))

        if np.ndim(gamma_dot) == 0:
            result = compute_n1(gamma_dot)
        else:
            result = jax.vmap(compute_n1)(gamma_dot)

        return np.asarray(result)

    def get_trajectory(self) -> dict[str, np.ndarray] | None:
        """Get stored ODE trajectory from last prediction.

        Returns
        -------
        dict or None
            Dictionary with keys like 'time', 'stress', 'strain', 'S_xy'
        """
        return self._trajectory

    # =========================================================================
    # Initialization from Data
    # =========================================================================

    def initialize_from_saos(
        self,
        omega: np.ndarray,
        G_prime: np.ndarray,
        G_double_prime: np.ndarray,
    ) -> None:
        """Initialize parameters from SAOS data.

        Uses crossover frequency to estimate sticker lifetime and plateau
        modulus to distribute mode strengths.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency array (rad/s)
        G_prime : np.ndarray
            Storage modulus G' (Pa)
        G_double_prime : np.ndarray
            Loss modulus G'' (Pa)
        """
        omega = np.asarray(omega)
        G_prime = np.asarray(G_prime)

        # Estimate plateau modulus from high-frequency G'
        G_plateau = np.max(G_prime)

        # Estimate terminal time from low-frequency crossover
        super().initialize_from_saos(omega, G_prime, G_double_prime)

        # Distribute modulus across modes
        G_per_mode = G_plateau / self._n_modes
        for k in range(self._n_modes):
            self.parameters.set_value(f"G_{k}", G_per_mode)

        # Estimate sticker lifetime from plateau frequency
        # (where G' starts to plateau)
        plateau_idx = np.argmax(G_prime > 0.9 * G_plateau)
        if plateau_idx > 0:
            omega_plateau = omega[plateau_idx]
            tau_s_est = 1.0 / omega_plateau
            self.parameters.set_value("tau_s", np.clip(tau_s_est, 1e-6, 1e4))

        logger.debug(
            f"SAOS initialization: G_plateau={G_plateau:.3e} Pa, "
            f"tau_s={self.tau_s:.3e} s"
        )

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        """Return string representation."""
        G_modes, _, tau_eff = self._get_mode_arrays()
        return (
            f"TNTStickyRouse(n_modes={self._n_modes}, "
            f"tau_s={self.tau_s:.3e} s, "
            f"G_plateau={float(jnp.sum(G_modes)):.3e} Pa)"
        )
