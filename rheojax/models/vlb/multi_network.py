"""Multi-network VLB (Vernerey-Long-Brighenti) transient network model.

This module implements `VLBMultiNetwork`, a constitutive model for polymers
with N distinct populations of dynamic crosslinks, each with its own
modulus G_i and dissociation rate k_d_i, plus an optional permanent network
and solvent viscosity.

Key Physics
-----------
Multi-network VLB describes heterogeneous networks where:

- N independent transient populations with moduli G_i and rates k_d_i
- Each population has its own distribution tensor mu_i evolving independently
- Optional permanent network (k_d = 0, modulus G_e) for cross-linked gels
- Optional Newtonian solvent (viscosity eta_s)

Total stress:

    sigma = sum_i G_i * (mu_i - I) + G_e * (F.F^T - I) + eta_s * D

This represents a superposition of Maxwell elements with molecular basis:
each mode corresponds to a physical population of crosslinks, not a
mathematical decomposition.

Supported Protocols
-------------------
- FLOW_CURVE: Newtonian (constant k_d), analytical superposition
- OSCILLATION: Multi-mode Maxwell G'(omega), G''(omega) (analytical)
- STARTUP: Analytical superposition of transient terms
- RELAXATION: Prony series G(t) = G_e + sum G_i * exp(-k_d_i * t)
- CREEP: ODE for multi-mode; analytical for 1 mode + permanent (SLS)
- LAOS: Multi-mode ODE integration via diffrax

Example
-------
>>> from rheojax.models.vlb import VLBMultiNetwork
>>> import numpy as np
>>>
>>> # Two transient networks + permanent
>>> model = VLBMultiNetwork(n_modes=2, include_permanent=True)
>>>
>>> # SAOS (multi-mode Maxwell)
>>> omega = np.logspace(-2, 2, 50)
>>> G_star = model.predict(omega, test_mode='oscillation')

References
----------
- Vernerey, F.J., Long, R. & Brighenti, R. (2017). JMPS 107, 1-20.
"""

from __future__ import annotations

import logging

import diffrax
import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.vlb._base import VLBBase
from rheojax.models.vlb._kernels import (
    vlb_creep_compliance_dual_vec,
    vlb_multi_relaxation_vec,
    vlb_multi_saos_vec,
    vlb_multi_startup_stress_vec,
    vlb_multi_steady_viscosity,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "vlb_multi_network",
    protocols=[
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.RELAXATION,
        Protocol.CREEP,
        Protocol.LAOS,
    ],
    deformation_modes=[
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    ],
)
class VLBMultiNetwork(VLBBase):
    """Multi-network VLB model: M transient + optional permanent + solvent.

    Implements a network with N independent transient crosslink populations,
    each with modulus G_i and dissociation rate k_d_i. The total stress is
    a superposition of N Maxwell modes.

    Parameters
    ----------
    n_modes : int, default 2
        Number of distinct transient network populations (N >= 1)
    include_permanent : bool, default False
        Whether to include a permanent (elastic) network (G_e)

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters: [G_0, k_d_0, G_1, k_d_1, ..., eta_s, (G_e)]
    fitted_ : bool
        Whether the model has been fitted
    _n_modes : int
        Number of transient modes

    Notes
    -----
    Parameter ordering: [G_0, k_d_0, G_1, k_d_1, ..., G_{N-1}, k_d_{N-1}, eta_s, (G_e)]
    Total parameter count: 2N + 1 (without permanent) or 2N + 2 (with permanent)

    See Also
    --------
    VLBLocal : Single transient network (2 parameters)
    """

    def __init__(
        self,
        n_modes: int = 2,
        include_permanent: bool = False,
    ):
        """Initialize multi-network VLB model.

        Parameters
        ----------
        n_modes : int, default 2
            Number of transient network populations (must be >= 1)
        include_permanent : bool, default False
            Include permanent elastic network
        """
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}")

        self._n_modes = n_modes
        self._include_permanent = include_permanent

        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with multi-network parameters.

        Parameters are organized as:
            [G_0, k_d_0, G_1, k_d_1, ..., G_{N-1}, k_d_{N-1}, eta_s, (G_e)]

        Default values:
            - G_i = 1000.0 / N (equal distribution)
            - k_d_i = 10^(i) (logarithmic spacing: 1.0, 10.0, 100.0, ...)
            - eta_s = 0.0 (no solvent viscosity)
            - G_e = 0.0 (no permanent network, if included)
        """
        self.parameters = ParameterSet()

        for i in range(self._n_modes):
            # Mode modulus
            self.parameters.add(
                name=f"G_{i}",
                value=1000.0 / self._n_modes,
                bounds=(1e0, 1e8),
                units="Pa",
                description=f"Network modulus for transient population {i}",
            )

            # Mode dissociation rate (logarithmically spaced)
            default_kd = 10.0**i
            self.parameters.add(
                name=f"k_d_{i}",
                value=default_kd,
                bounds=(1e-6, 1e6),
                units="1/s",
                description=f"Dissociation rate for population {i}",
            )

        # Solvent viscosity
        self.parameters.add(
            name="eta_s",
            value=0.0,
            bounds=(0.0, 1e4),
            units="Pa·s",
            description="Solvent viscosity (Newtonian background)",
        )

        # Optional permanent network
        if self._include_permanent:
            self.parameters.add(
                name="G_e",
                value=0.0,
                bounds=(0.0, 1e8),
                units="Pa",
                description="Permanent (elastic) network modulus",
            )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def n_modes(self) -> int:
        """Number of transient network modes."""
        return self._n_modes

    @property
    def include_permanent(self) -> bool:
        """Whether a permanent network is included."""
        return self._include_permanent

    @property
    def G_e(self) -> float:
        """Permanent network modulus (Pa). 0 if not included."""
        if not self._include_permanent:
            return 0.0
        val = self.parameters.get_value("G_e")
        return float(val) if val is not None else 0.0

    @property
    def eta_s(self) -> float:
        """Solvent viscosity (Pa*s)."""
        val = self.parameters.get_value("eta_s")
        return float(val) if val is not None else 0.0

    @property
    def G_total(self) -> float:
        """Total modulus: sum G_i + G_e."""
        G, _ = self._get_mode_arrays_numpy()
        return float(np.sum(G)) + self.G_e

    @property
    def eta_0(self) -> float:
        """Zero-shear viscosity: sum G_i/k_d_i + eta_s."""
        G, kd = self._get_mode_arrays_numpy()
        return float(np.sum(G / kd)) + self.eta_s

    # =========================================================================
    # Mode Array Helpers
    # =========================================================================

    def _get_mode_arrays_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Get mode arrays as numpy arrays.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            (G_modes, kd_modes) each shape (N,)
        """
        G_arr = np.array([
            float(self.parameters.get_value(f"G_{i}"))
            for i in range(self._n_modes)
        ])
        kd_arr = np.array([
            float(self.parameters.get_value(f"k_d_{i}"))
            for i in range(self._n_modes)
        ])
        return G_arr, kd_arr

    def _unpack_mode_params(
        self, params: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, float, float]:
        """Unpack mode parameters from a flat JAX array.

        Parameters
        ----------
        params : jnp.ndarray
            Flat parameter array [G_0, k_d_0, G_1, k_d_1, ..., eta_s, (G_e)]

        Returns
        -------
        tuple
            (G_modes, kd_modes, eta_s, G_e) where G_modes and kd_modes
            are shape (N,) arrays
        """
        N = self._n_modes
        G_modes = params[0: 2 * N: 2]  # G_0, G_1, ...
        kd_modes = params[1: 2 * N: 2]  # k_d_0, k_d_1, ...
        eta_s = params[2 * N]
        G_e = params[2 * N + 1] if self._include_permanent else 0.0
        return G_modes, kd_modes, eta_s, G_e

    # =========================================================================
    # Fitting
    # =========================================================================

    def _fit(self, x, y, **kwargs):
        """Fit model to data using protocol-aware optimization.

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

        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        self._test_mode = test_mode

        x_jax = jnp.asarray(x, dtype=jnp.float64)

        # For oscillation mode with complex G* data, convert to |G*|
        # since model_function returns |G*| for oscillation
        y_np = np.asarray(y)
        if test_mode == "oscillation" and np.iscomplexobj(y_np):
            y_jax = jnp.asarray(np.abs(y_np), dtype=jnp.float64)
        else:
            y_jax = jnp.asarray(y_np, dtype=jnp.float64)

        # Store protocol-specific inputs
        self._gamma_dot_applied = kwargs.get("gamma_dot")
        self._sigma_applied = kwargs.get("sigma_applied")
        self._gamma_0 = kwargs.get("gamma_0")
        self._omega_laos = kwargs.get("omega")

        # Smart initialization
        if test_mode == "oscillation":
            self.initialize_from_saos(
                np.asarray(x), np.real(np.asarray(y)), np.imag(np.asarray(y))
            )

        # Define model function for fitting (exclude test_mode from kwargs to avoid duplicates)
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("test_mode", "use_log_residuals", "use_jax", "method",
                         "max_iter", "use_multi_start", "n_starts", "perturb_factor")
        }

        def model_fn(x_fit, params):
            return self.model_function(x_fit, params, test_mode=test_mode, **fwd_kwargs)

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
            f"Fitted VLBMultiNetwork ({self._n_modes} modes): "
            f"eta_0={self.eta_0:.2e}"
        )

        return self

    # =========================================================================
    # Prediction
    # =========================================================================

    def _predict(self, x, **kwargs):
        """Predict response using protocol-aware dispatch.

        Parameters
        ----------
        x : array-like
            Independent variable
        **kwargs
            Additional arguments

        Returns
        -------
        jnp.ndarray
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        if "gamma_dot" in kwargs:
            self._gamma_dot_applied = kwargs["gamma_dot"]
        if "sigma_applied" in kwargs:
            self._sigma_applied = kwargs["sigma_applied"]
        if "gamma_0" in kwargs:
            self._gamma_0 = kwargs["gamma_0"]
        if "omega" in kwargs:
            self._omega_laos = kwargs["omega"]

        param_values = [
            float(self.parameters.get_value(name))
            for name in self.parameters.keys()
        ]
        params = jnp.array(param_values)

        # Remove test_mode from kwargs to avoid duplicate
        fwd_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("test_mode", "deformation_mode", "poisson_ratio")
        }
        return self.model_function(x_jax, params, test_mode=test_mode, **fwd_kwargs)

    # =========================================================================
    # Model Function (NLSQ / NumPyro)
    # =========================================================================

    def model_function(self, X, params, test_mode=None, **kwargs):
        """NumPyro/BayesianMixin model function.

        Routes to appropriate prediction based on test_mode.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values: [G_0, k_d_0, ..., G_{N-1}, k_d_{N-1}, eta_s, (G_e)]
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific parameters

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        G_modes, kd_modes, eta_s, G_e = self._unpack_mode_params(params)

        mode = test_mode or self._test_mode or "flow_curve"
        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)

        X_jax = jnp.asarray(X, dtype=jnp.float64)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return self._predict_flow_curve_internal(X_jax, G_modes, kd_modes, eta_s)

        elif mode == "oscillation":
            G_prime, G_double_prime = vlb_multi_saos_vec(
                X_jax, G_modes, kd_modes, G_e, eta_s
            )
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return vlb_multi_startup_stress_vec(
                X_jax, gamma_dot, G_modes, kd_modes, G_e, eta_s
            )

        elif mode == "relaxation":
            return vlb_multi_relaxation_vec(X_jax, G_modes, kd_modes, G_e)

        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            return self._simulate_creep_internal(
                X_jax, G_modes, kd_modes, eta_s, G_e, sigma_applied
            )

        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G_modes, kd_modes, eta_s, gamma_0, omega
            )
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return self._predict_flow_curve_internal(X_jax, G_modes, kd_modes, eta_s)

    # =========================================================================
    # Analytical Predictions
    # =========================================================================

    def _predict_flow_curve_internal(
        self,
        gamma_dot: jnp.ndarray,
        G_modes: jnp.ndarray,
        kd_modes: jnp.ndarray,
        eta_s: float,
    ) -> jnp.ndarray:
        """Analytical steady shear stress for multi-network VLB.

        sigma = (sum G_i / k_d_i + eta_s) * gamma_dot = eta_0 * gamma_dot

        Newtonian for constant k_d (no shear thinning).
        """
        eta_0 = vlb_multi_steady_viscosity(G_modes, kd_modes, eta_s)
        return eta_0 * gamma_dot

    # =========================================================================
    # Creep Internal (ODE-based for general case)
    # =========================================================================

    def _simulate_creep_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        kd_modes: jnp.ndarray,
        eta_s: float,
        G_e: float,
        sigma_0: float,
    ) -> jnp.ndarray:
        """Internal creep simulation.

        For N=1 + permanent (SLS), uses analytical solution.
        For general case, uses ODE integration.

        Returns strain array gamma(t).
        """
        N = self._n_modes
        has_perm = G_e > 1e-30
        has_solvent = eta_s > 1e-30

        # Special case: 1 mode + permanent, no solvent -> SLS analytical
        if N == 1 and has_perm and not has_solvent:
            J = vlb_creep_compliance_dual_vec(t, G_modes[0], kd_modes[0], G_e)
            return sigma_0 * J

        # General case: ODE integration
        # State: [gamma, mu_xy_0, mu_xy_1, ..., mu_xy_{N-1}]
        # For each mode i: d(mu_xy_i)/dt = -k_d_i * mu_xy_i + gamma_dot * 1
        # (mu_yy = 1 since creep is small strain initially)
        # Stress balance: sigma_0 = sum G_i * mu_xy_i + G_e * gamma + eta_s * gdot

        def ode_fn(ti, yi, args):
            gamma = yi[0]
            mu_xy = yi[1:]

            # Stress balance -> solve for gamma_dot
            elastic = jnp.sum(args["G_modes"] * mu_xy) + args["G_e"] * gamma
            remaining = args["sigma_0"] - elastic

            # If eta_s > 0: gdot = remaining / eta_s
            # If eta_s = 0: gdot must come from mode balance directly
            gdot = jnp.where(
                args["eta_s"] > 1e-30,
                remaining / args["eta_s"],
                remaining / 1e-10,  # Fallback, shouldn't reach here if properly set
            )

            dgamma = gdot
            dmu_xy = -args["kd_modes"] * mu_xy + gdot * jnp.ones_like(mu_xy)

            return jnp.concatenate([jnp.array([dgamma]), dmu_xy])

        # Initial conditions: elastic jump
        G_total = jnp.sum(G_modes) + G_e
        gamma_0 = sigma_0 / jnp.maximum(G_total, 1e-30)
        mu_xy_0 = sigma_0 / jnp.maximum(G_total, 1e-30) * jnp.ones(N)

        y0 = jnp.concatenate([jnp.array([gamma_0]), mu_xy_0])

        args = {
            "G_modes": G_modes,
            "kd_modes": kd_modes,
            "eta_s": jnp.maximum(eta_s, 1e-10),  # Small regularization
            "G_e": G_e,
            "sigma_0": sigma_0,
        }

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

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
            stepsize_controller=controller,
            max_steps=500_000,
            throw=False,
        )

        gamma = sol.ys[:, 0]
        gamma = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            gamma,
            jnp.nan * jnp.ones_like(gamma),
        )
        return gamma

    # =========================================================================
    # LAOS Internal (ODE-based)
    # =========================================================================

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G_modes: jnp.ndarray,
        kd_modes: jnp.ndarray,
        eta_s: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for multi-network.

        State vector: [mu_xy_0, ..., mu_xy_{N-1}, mu_xx_0, ..., mu_xx_{N-1}]
        (mu_yy stays at 1 for each mode since yy decouples)

        Returns (strain, stress) arrays.
        """
        N = self._n_modes

        def ode_fn(ti, yi, args):
            mu_xy = yi[:N]
            mu_xx = yi[N:]

            gdot = args["gamma_0"] * args["omega"] * jnp.cos(args["omega"] * ti)

            # mu_yy = 1 always (decoupled, relaxes to 1 independently)
            dmu_xy = -args["kd_modes"] * mu_xy + gdot * jnp.ones(N)
            dmu_xx = args["kd_modes"] * (1.0 - mu_xx) + 2.0 * gdot * mu_xy

            return jnp.concatenate([dmu_xy, dmu_xx])

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "kd_modes": kd_modes,
        }

        # Initial state: equilibrium
        y0 = jnp.concatenate([
            jnp.zeros(N),  # mu_xy = 0
            jnp.ones(N),   # mu_xx = 1
        ])

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)

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
            stepsize_controller=controller,
            max_steps=500_000,
            throw=False,
        )

        mu_xy_all = sol.ys[:, :N]  # shape (T, N)
        gamma_dot_t = gamma_0 * omega * jnp.cos(omega * t)

        strain = gamma_0 * jnp.sin(omega * t)
        stress = jnp.sum(G_modes[None, :] * mu_xy_all, axis=1) + eta_s * gamma_dot_t
        stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            stress,
            jnp.nan * jnp.ones_like(stress),
        )

        return strain, stress

    # =========================================================================
    # Public Methods
    # =========================================================================

    def predict_saos(
        self,
        omega: np.ndarray,
        return_components: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict multi-mode SAOS moduli.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency (rad/s)
        return_components : bool, default True
            If True, return (G', G'')

        Returns
        -------
        tuple or np.ndarray
            (G', G'') or |G*|
        """
        w = jnp.asarray(omega, dtype=jnp.float64)
        G_np, kd_np = self._get_mode_arrays_numpy()
        G_modes = jnp.asarray(G_np, dtype=jnp.float64)
        kd_modes = jnp.asarray(kd_np, dtype=jnp.float64)

        G_p, G_pp = vlb_multi_saos_vec(w, G_modes, kd_modes, self.G_e, self.eta_s)
        if return_components:
            return np.asarray(G_p), np.asarray(G_pp)
        return np.asarray(jnp.sqrt(G_p**2 + G_pp**2))

    def get_relaxation_spectrum(self) -> list[tuple[float, float]]:
        """Get relaxation spectrum as list of (G, tau) pairs.

        Returns
        -------
        list[tuple[float, float]]
            [(G_i, 1/k_d_i)] for each transient mode
        """
        G_np, kd_np = self._get_mode_arrays_numpy()
        return [(float(G_np[i]), 1.0 / float(kd_np[i])) for i in range(self._n_modes)]

    def __repr__(self) -> str:
        """Return string representation."""
        perm_str = "+permanent" if self._include_permanent else ""
        return (
            f"VLBMultiNetwork(n_modes={self._n_modes}{perm_str}, "
            f"eta_0={self.eta_0:.2e})"
        )
