"""Single-network VLB (Vernerey-Long-Brighenti) transient network model.

This module implements `VLBLocal`, a constitutive model for polymers with
dynamic crosslinks, based on a single transient network with constant
dissociation rate k_d.

Key Physics
-----------
The VLB local model tracks the distribution tensor mu:

    dmu/dt = k_d*(I - mu) + L·mu + mu·L^T

With constant k_d, this is mathematically equivalent to the upper-convected
Maxwell model (UCM). However, the VLB derivation provides:

- A molecular-statistical foundation (from chain distribution phi(r,t))
- Clear path to force-dependent extensions (Bell, Langevin)
- Natural multi-network superposition via independent populations
- Physical interpretation: G0 = c*k_B*T (chain density × thermal energy)

Parameters
----------
The model has 2 free parameters:
- G0: Network modulus (Pa) = chain density × k_B*T
- k_d: Dissociation rate (1/s), inverse relaxation time

Supported Protocols
-------------------
- FLOW_CURVE: Newtonian sigma = G0*gamma_dot/k_d (analytical)
- OSCILLATION: Maxwell G'(omega), G''(omega) (analytical)
- STARTUP: sigma(t) = G0*gamma_dot*t_R*(1-exp(-t/t_R)) (analytical)
- RELAXATION: G(t) = G0*exp(-k_d*t) (analytical)
- CREEP: J(t) = (1+k_d*t)/G0 (analytical)
- LAOS: Full tensor ODE integration via diffrax

Example
-------
>>> from rheojax.models.vlb import VLBLocal
>>> import numpy as np
>>>
>>> model = VLBLocal()
>>> model.parameters.set_value("G0", 1000.0)
>>> model.parameters.set_value("k_d", 1.0)
>>>
>>> # Flow curve (analytical, Newtonian)
>>> gamma_dot = np.logspace(-2, 2, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # SAOS (Maxwell)
>>> omega = np.logspace(-2, 2, 50)
>>> G_star = model.predict(omega, test_mode='oscillation')

References
----------
- Vernerey, F.J., Long, R. & Brighenti, R. (2017). JMPS 107, 1-20.
"""

from __future__ import annotations

import logging

import numpy as np

from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import lazy_import, safe_import_jax
diffrax = lazy_import("diffrax")
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode
from rheojax.models.vlb._base import VLBBase
from rheojax.models.vlb._kernels import (
    vlb_creep_compliance_single_vec,
    vlb_relaxation_modulus_vec,
    vlb_saos_moduli_vec,
    vlb_startup_n1_vec,
    vlb_startup_stress_vec,
    vlb_steady_n1_vec,
    vlb_steady_shear_vec,
    vlb_trouton_ratio_vec,
    vlb_uniaxial_steady_vec,
    vlb_uniaxial_transient_vec,
)

jax, jnp = safe_import_jax()

logger = logging.getLogger(__name__)


@ModelRegistry.register(
    "vlb_local",
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
@ModelRegistry.register(
    "vlb",
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
class VLBLocal(VLBBase):
    """Single transient network VLB model (2 params: G0, k_d).

    Implements the VLB framework for a single population of dynamic
    crosslinks with constant dissociation rate. Analytically equivalent
    to the Maxwell model but with molecular-statistical foundations.

    The distribution tensor mu has equilibrium mu = I and evolves via:

        dmu/dt = k_d*(I - mu) + L·mu + mu·L^T

    Cauchy stress: sigma = G0*(mu - I)

    Parameters
    ----------
    G0 : float
        Network modulus (Pa), physically G0 = c*k_B*T where c is chain density
    k_d : float
        Dissociation rate (1/s), inverse of relaxation time t_R = 1/k_d

    Attributes
    ----------
    parameters : ParameterSet
        Model parameters (G0, k_d)
    fitted_ : bool
        Whether the model has been fitted

    See Also
    --------
    VLBMultiNetwork : Multi-network VLB with N transient + permanent + solvent
    """

    def __init__(self):
        """Initialize single-network VLB model."""
        super().__init__()
        self._setup_parameters()
        self._test_mode = None

    # =========================================================================
    # Parameter Setup
    # =========================================================================

    def _setup_parameters(self):
        """Initialize ParameterSet with VLB local parameters.

        Parameters:
        - G0: Network modulus (Pa)
        - k_d: Dissociation rate (1/s)
        """
        self.parameters = ParameterSet()

        self.parameters.add(
            name="G0",
            value=1000.0,
            bounds=(1e0, 1e8),
            units="Pa",
            description="Network modulus (chain density * k_B * T)",
        )
        self.parameters.add(
            name="k_d",
            value=1.0,
            bounds=(1e-6, 1e6),
            units="1/s",
            description="Dissociation rate (inverse relaxation time)",
        )

    # =========================================================================
    # Property Accessors
    # =========================================================================

    @property
    def G0(self) -> float:
        """Get network modulus G0 (Pa)."""
        val = self.parameters.get_value("G0")
        return float(val) if val is not None else 0.0

    @property
    def k_d(self) -> float:
        """Get dissociation rate k_d (1/s)."""
        val = self.parameters.get_value("k_d")
        return float(val) if val is not None else 0.0

    @property
    def relaxation_time(self) -> float:
        """Get relaxation time t_R = 1/k_d (s)."""
        return 1.0 / max(self.k_d, 1e-30)

    @property
    def viscosity(self) -> float:
        """Get zero-shear viscosity eta_0 = G0/k_d (Pa*s)."""
        return self.G0 / max(self.k_d, 1e-30)

    # =========================================================================
    # Fitting
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

        # Smart initialization based on protocol
        if test_mode in ["flow_curve", "steady_shear", "rotation"]:
            self.initialize_from_flow_curve(np.asarray(x), np.asarray(y))
        elif test_mode == "oscillation":
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
            f"Fitted VLBLocal: G0={self.G0:.2e}, k_d={self.k_d:.2e}"
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
            Additional arguments including test_mode

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        test_mode = kwargs.get("test_mode", self._test_mode or "flow_curve")
        x_jax = jnp.asarray(x, dtype=jnp.float64)

        # Extract and store protocol-specific parameters from kwargs
        if "gamma_dot" in kwargs:
            self._gamma_dot_applied = kwargs["gamma_dot"]
        if "sigma_applied" in kwargs:
            self._sigma_applied = kwargs["sigma_applied"]
        if "gamma_0" in kwargs:
            self._gamma_0 = kwargs["gamma_0"]
        if "omega" in kwargs:
            self._omega_laos = kwargs["omega"]

        # Build parameter array from ParameterSet (ordering matters)
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

        Routes to appropriate prediction based on test_mode. This is the
        stateless function used for both NLSQ optimization and Bayesian
        inference.

        Parameters
        ----------
        X : array-like
            Independent variable
        params : array-like
            Parameter values in ParameterSet order: [G0, k_d]
        test_mode : str, optional
            Override stored test mode
        **kwargs
            Protocol-specific parameters: gamma_dot, sigma_applied, gamma_0, omega

        Returns
        -------
        jnp.ndarray
            Predicted response
        """
        # Unpack core parameters
        G0 = params[0]
        k_d = params[1]

        mode = test_mode or self._test_mode or "flow_curve"
        X_jax = jnp.asarray(X, dtype=jnp.float64)

        # Extract protocol parameters
        gamma_dot = kwargs.get("gamma_dot", self._gamma_dot_applied)
        sigma_applied = kwargs.get("sigma_applied", self._sigma_applied)
        gamma_0 = kwargs.get("gamma_0", self._gamma_0)
        omega = kwargs.get("omega", self._omega_laos)

        if mode in ["flow_curve", "steady_shear", "rotation"]:
            return vlb_steady_shear_vec(X_jax, G0, k_d)

        elif mode == "oscillation":
            G_prime, G_double_prime = vlb_saos_moduli_vec(X_jax, G0, k_d)
            return jnp.sqrt(G_prime**2 + G_double_prime**2)

        elif mode == "startup":
            if gamma_dot is None:
                raise ValueError("startup mode requires gamma_dot")
            return vlb_startup_stress_vec(X_jax, gamma_dot, G0, k_d)

        elif mode == "relaxation":
            return vlb_relaxation_modulus_vec(X_jax, G0, k_d)

        elif mode == "creep":
            if sigma_applied is None:
                raise ValueError("creep mode requires sigma_applied")
            # Return strain gamma = sigma_0 * J(t)
            J = vlb_creep_compliance_single_vec(X_jax, G0, k_d)
            return sigma_applied * J

        elif mode == "laos":
            if gamma_0 is None or omega is None:
                raise ValueError("LAOS mode requires gamma_0 and omega")
            _, stress = self._simulate_laos_internal(
                X_jax, G0, k_d, gamma_0, omega
            )
            return stress

        else:
            logger.warning(f"Unknown test_mode '{mode}', defaulting to flow_curve")
            return vlb_steady_shear_vec(X_jax, G0, k_d)

    # =========================================================================
    # LAOS Internal Simulation (ODE via diffrax)
    # =========================================================================

    def _simulate_laos_internal(
        self,
        t: jnp.ndarray,
        G0: float,
        k_d: float,
        gamma_0: float,
        omega: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Internal LAOS simulation for model_function.

        Returns (strain, stress) arrays.
        """

        def ode_fn(ti, yi, args):
            mu_xx, mu_yy, mu_zz, mu_xy = yi[0], yi[1], yi[2], yi[3]
            gdot = args["gamma_0"] * args["omega"] * jnp.cos(args["omega"] * ti)
            dmu_xx = args["k_d"] * (1.0 - mu_xx) + 2.0 * gdot * mu_xy
            dmu_yy = args["k_d"] * (1.0 - mu_yy)
            dmu_zz = args["k_d"] * (1.0 - mu_zz)
            dmu_xy = -args["k_d"] * mu_xy + gdot * mu_yy
            return jnp.array([dmu_xx, dmu_yy, dmu_zz, dmu_xy])

        args = {
            "gamma_0": gamma_0,
            "omega": omega,
            "G0": G0,
            "k_d": k_d,
        }
        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

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
            max_steps=500_000,
            throw=False,
        )

        strain = gamma_0 * jnp.sin(omega * t)
        stress = G0 * sol.ys[:, 3]  # sigma = G0 * mu_xy
        stress = jnp.where(
            sol.result == diffrax.RESULTS.successful,
            stress,
            jnp.nan * jnp.ones_like(stress),
        )

        return strain, stress

    # =========================================================================
    # Public Simulation Methods
    # =========================================================================

    def predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        return_components: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict steady shear stress (and optionally viscosity, N1).

        Newtonian: sigma = G0*gamma_dot/k_d.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        return_components : bool, default False
            If True, return (sigma, eta, N1)

        Returns
        -------
        np.ndarray or tuple
            Stress, or (stress, viscosity, N1) if return_components=True
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)
        sigma = vlb_steady_shear_vec(gd, self.G0, self.k_d)
        if return_components:
            eta = sigma / jnp.maximum(gd, 1e-20)
            N1 = vlb_steady_n1_vec(gd, self.G0, self.k_d)
            return np.asarray(sigma), np.asarray(eta), np.asarray(N1)
        return np.asarray(sigma)

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
        w = jnp.asarray(omega, dtype=jnp.float64)
        G_prime, G_double_prime = vlb_saos_moduli_vec(w, self.G0, self.k_d)
        if return_components:
            return np.asarray(G_prime), np.asarray(G_double_prime)
        G_star = jnp.sqrt(G_prime**2 + G_double_prime**2)
        return np.asarray(G_star)

    def predict_normal_stresses(
        self,
        gamma_dot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict steady-state first and second normal stress differences.

        N1 = 2*G0*(gamma_dot/k_d)^2, N2 = 0 (upper-convected Maxwell).

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (N1, N2) in Pa. N2 is always zero for upper-convected VLB.
        """
        gd = jnp.asarray(gamma_dot, dtype=jnp.float64)
        N1 = vlb_steady_n1_vec(gd, self.G0, self.k_d)
        N2 = jnp.zeros_like(N1)
        return np.asarray(N1), np.asarray(N2)

    def simulate_startup(
        self,
        t: np.ndarray,
        gamma_dot: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate startup flow at constant shear rate.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        gamma_dot : float
            Applied shear rate (1/s)
        return_full : bool, default False
            If True, return (sigma, N1, strain)

        Returns
        -------
        np.ndarray or tuple
            Stress, or (stress, N1, strain) if return_full=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sigma = vlb_startup_stress_vec(t_jax, gamma_dot, self.G0, self.k_d)
        if return_full:
            N1 = vlb_startup_n1_vec(t_jax, gamma_dot, self.G0, self.k_d)
            strain = gamma_dot * t_jax
            return np.asarray(sigma), np.asarray(N1), np.asarray(strain)
        return np.asarray(sigma)

    def simulate_relaxation(
        self,
        t: np.ndarray,
    ) -> np.ndarray:
        """Simulate stress relaxation G(t) = G0*exp(-k_d*t).

        Parameters
        ----------
        t : np.ndarray
            Time after cessation of flow (s)

        Returns
        -------
        np.ndarray
            Relaxation modulus G(t) (Pa)
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        G_t = vlb_relaxation_modulus_vec(t_jax, self.G0, self.k_d)
        return np.asarray(G_t)

    def simulate_creep(
        self,
        t: np.ndarray,
        sigma_0: float,
        return_full: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Simulate creep under constant applied stress.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma_0 : float
            Applied stress (Pa)
        return_full : bool, default False
            If True, return (strain, compliance)

        Returns
        -------
        np.ndarray or tuple
            Strain gamma(t), or (gamma, J) if return_full=True
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        J = vlb_creep_compliance_single_vec(t_jax, self.G0, self.k_d)
        gamma = sigma_0 * J
        if return_full:
            return np.asarray(gamma), np.asarray(J)
        return np.asarray(gamma)

    def simulate_laos(
        self,
        t: np.ndarray,
        gamma_0: float,
        omega: float,
    ) -> dict:
        """Simulate large-amplitude oscillatory shear (LAOS).

        Parameters
        ----------
        t : np.ndarray
            Time array (s), should span at least 3-5 full cycles
        gamma_0 : float
            Strain amplitude
        omega : float
            Angular frequency (rad/s)

        Returns
        -------
        dict
            Keys: 'time', 'strain', 'stress', 'N1', 'gamma_dot'
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)

        # Use diffrax ODE integration
        def ode_fn(ti, yi, args):
            mu_xx, mu_yy, mu_zz, mu_xy = yi[0], yi[1], yi[2], yi[3]
            gdot = gamma_0 * omega * jnp.cos(omega * ti)
            dmu_xx = self.k_d * (1.0 - mu_xx) + 2.0 * gdot * mu_xy
            dmu_yy = self.k_d * (1.0 - mu_yy)
            dmu_zz = self.k_d * (1.0 - mu_zz)
            dmu_xy = -self.k_d * mu_xy + gdot * mu_yy
            return jnp.array([dmu_xx, dmu_yy, dmu_zz, dmu_xy])

        y0 = jnp.array([1.0, 1.0, 1.0, 0.0], dtype=jnp.float64)

        term = diffrax.ODETerm(ode_fn)
        solver = diffrax.Tsit5()
        controller = diffrax.PIDController(rtol=1e-6, atol=1e-8)
        dt0 = (t_jax[-1] - t_jax[0]) / max(len(t_jax), 1000)
        saveat = diffrax.SaveAt(ts=t_jax)

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t_jax[0],
            t_jax[-1],
            dt0,
            y0,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=500_000,
            throw=False,
        )

        strain = gamma_0 * np.sin(omega * np.asarray(t))
        gamma_dot_arr = gamma_0 * omega * np.cos(omega * np.asarray(t))
        stress = np.asarray(self.G0 * sol.ys[:, 3])
        N1 = np.asarray(self.G0 * (sol.ys[:, 0] - sol.ys[:, 1]))

        self._trajectory = {
            "time": np.asarray(t),
            "strain": strain,
            "stress": stress,
            "N1": N1,
            "gamma_dot": gamma_dot_arr,
        }

        return self._trajectory

    def predict_uniaxial_extension(
        self,
        epsilon_dot: np.ndarray,
        return_trouton: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict steady-state uniaxial extensional stress.

        Parameters
        ----------
        epsilon_dot : np.ndarray
            Extensional strain rate (1/s)
        return_trouton : bool, default False
            If True, also return Trouton ratio

        Returns
        -------
        np.ndarray or tuple
            Extensional stress, or (stress, Trouton_ratio)
        """
        ed = jnp.asarray(epsilon_dot, dtype=jnp.float64)
        sigma_E = vlb_uniaxial_steady_vec(ed, self.G0, self.k_d)

        # Warn about singularity
        crit_rate = self.k_d / 2.0
        if np.any(np.asarray(epsilon_dot) > 0.9 * crit_rate):
            logger.warning(
                f"Extensional rates near singularity at eps_dot = k_d/2 = {crit_rate:.2e} 1/s. "
                "Results may be unreliable."
            )

        if return_trouton:
            Tr = vlb_trouton_ratio_vec(ed, self.G0, self.k_d)
            return np.asarray(sigma_E), np.asarray(Tr)
        return np.asarray(sigma_E)

    def simulate_uniaxial_extension(
        self,
        t: np.ndarray,
        epsilon_dot: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate transient uniaxial extension.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        epsilon_dot : float
            Extensional strain rate (1/s)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (extensional_stress, extensional_viscosity) as functions of time
        """
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sigma_E = vlb_uniaxial_transient_vec(t_jax, epsilon_dot, self.G0, self.k_d)
        eta_E = sigma_E / max(abs(epsilon_dot), 1e-20)
        return np.asarray(sigma_E), np.asarray(eta_E)

    def get_relaxation_spectrum(self) -> list[tuple[float, float]]:
        """Get relaxation spectrum as list of (G, tau) pairs.

        Returns
        -------
        list[tuple[float, float]]
            [(G0, 1/k_d)] — single mode
        """
        return [(self.G0, self.relaxation_time)]

    def extract_laos_harmonics(
        self,
        laos_result: dict,
        n_harmonics: int = 5,
    ) -> dict:
        """Extract Fourier harmonics from LAOS results.

        Parameters
        ----------
        laos_result : dict
            Output from simulate_laos()
        n_harmonics : int, default 5
            Number of harmonics to extract

        Returns
        -------
        dict
            Keys: 'harmonic_index', 'sigma_harmonics', 'N1_harmonics'
        """
        t = laos_result["time"]
        stress = laos_result["stress"]
        N1 = laos_result["N1"]

        # Use last 2 cycles for steady-state harmonics
        period = 2 * np.pi / self._omega_laos if self._omega_laos else 1.0
        t_range = t[-1] - t[0]
        n_cycles = int(t_range / period)
        if n_cycles >= 2:
            start_idx = np.searchsorted(t, t[-1] - 2 * period)
        else:
            start_idx = 0

        t_ss = t[start_idx:]
        stress_ss = stress[start_idx:]
        N1_ss = N1[start_idx:]

        # FFT
        n = len(stress_ss)
        sigma_fft = np.fft.rfft(stress_ss) / n * 2
        N1_fft = np.fft.rfft(N1_ss) / n * 2
        freqs = np.fft.rfftfreq(n, d=(t_ss[-1] - t_ss[0]) / n)

        # Find fundamental frequency index
        if self._omega_laos:
            f0 = self._omega_laos / (2 * np.pi)
            fund_idx = np.argmin(np.abs(freqs - f0))
        else:
            fund_idx = np.argmax(np.abs(sigma_fft[1:])) + 1

        harmonics_idx = [fund_idx * (i + 1) for i in range(n_harmonics)]
        harmonics_idx = [i for i in harmonics_idx if i < len(sigma_fft)]

        return {
            "harmonic_index": list(range(1, len(harmonics_idx) + 1)),
            "sigma_harmonics": np.abs(sigma_fft[harmonics_idx]),
            "N1_harmonics": np.abs(N1_fft[harmonics_idx]),
        }
