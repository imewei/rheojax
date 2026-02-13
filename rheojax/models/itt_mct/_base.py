"""Base class for ITT-MCT (Integration Through Transients Mode-Coupling Theory) models.

This module provides the abstract base class for ITT-MCT implementations,
handling protocol dispatch, Volterra ODE integration, and history management.

Classes
-------
ITTMCTBase
    Abstract base class for ITT-MCT models (F₁₂ schematic and ISM)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.inventory import Protocol
from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


# Type alias for protocol strings
ProtocolType = Literal[
    "flow_curve",
    "oscillation",
    "startup",
    "creep",
    "relaxation",
    "laos",
]


class ITTMCTBase(BaseModel):
    """Abstract base class for ITT-MCT (Mode-Coupling Theory) models.

    ITT-MCT describes the nonlinear rheology of dense colloidal suspensions
    and glassy materials through the "cage effect" - particles trapped by
    their neighbors. The theory predicts:

    - Glass transition at a critical volume fraction
    - Yield stress in the glass state
    - Shear thinning from cage breaking
    - Stress overshoot in startup flows
    - History-dependent creep compliance

    Subclasses must implement:
    - `_setup_parameters()`: Define model parameters
    - `_predict_protocol_*()`: Protocol-specific predictions
    - `_compute_equilibrium_correlator()`: Quiescent correlator Φ_eq(t)

    Attributes
    ----------
    integration_method : {"volterra", "history"}
        Integration approach for memory kernel.
        "volterra": Prony decomposition + ODE (O(N) per step, default)
        "history": Full history integration (O(N²), more accurate near glass)
    n_prony_modes : int
        Number of Prony modes for Volterra integration
    parameters : ParameterSet
        Model parameters

    Notes
    -----
    The MCT equation of motion for the density correlator Φ(t) is:

        ∂Φ/∂t + Γ[Φ + ∫₀^t m(Φ) ∂Φ/∂s ds] = 0

    where m(Φ) is the memory kernel and Γ is the bare relaxation rate.

    Under shear, the advected correlator Φ(t,t') additionally depends on
    accumulated strain through a decorrelation function h(γ).

    References
    ----------
    Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids"
    Fuchs M. & Cates M.E. (2002) Phys. Rev. Lett. 89, 248304
    """

    # Supported protocols for ITT-MCT models
    SUPPORTED_PROTOCOLS = [
        Protocol.FLOW_CURVE,
        Protocol.OSCILLATION,
        Protocol.STARTUP,
        Protocol.CREEP,
        Protocol.RELAXATION,
        Protocol.LAOS,
    ]

    def __init__(
        self,
        integration_method: Literal["volterra", "history"] = "volterra",
        n_prony_modes: int = 10,
    ):
        """Initialize ITT-MCT base model.

        Parameters
        ----------
        integration_method : {"volterra", "history"}, default "volterra"
            Integration method for memory kernel:
            - "volterra": Prony series decomposition for O(N) integration
            - "history": Full history storage for O(N²) but higher accuracy
        n_prony_modes : int, default 10
            Number of Prony modes if using Volterra integration
        """
        super().__init__()

        self.integration_method = integration_method
        self.n_prony_modes = n_prony_modes

        # Initialize parameters (implemented by subclass)
        self._setup_parameters()

        # Internal state
        self._equilibrium_correlator: np.ndarray | None = None
        self._prony_amplitudes: np.ndarray | None = None
        self._prony_times: np.ndarray | None = None
        self._history_buffer: np.ndarray | None = None

        # Protocol-specific storage
        self._last_protocol: ProtocolType | None = None
        self._trajectory: dict[str, np.ndarray] | None = None

    @abstractmethod
    def _setup_parameters(self) -> None:
        """Initialize model parameters.

        Subclasses must implement this to add parameters to self.parameters.

        Example implementation:
        ```python
        self.parameters = ParameterSet()
        self.parameters.add("v2", value=2.0, bounds=(0.5, 10), units="-",
                           description="Quadratic vertex coefficient")
        ```
        """
        pass

    @abstractmethod
    def _compute_equilibrium_correlator(
        self,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute equilibrium (quiescent) correlator Φ_eq(t).

        Parameters
        ----------
        t : jnp.ndarray
            Time array

        Returns
        -------
        jnp.ndarray
            Equilibrium correlator values Φ_eq(t)
        """
        pass

    @abstractmethod
    def _compute_memory_kernel(
        self,
        phi: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute memory kernel m(Φ) from correlator values.

        Parameters
        ----------
        phi : jnp.ndarray
            Correlator values

        Returns
        -------
        jnp.ndarray
            Memory kernel m(Φ)
        """
        pass

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> ITTMCTBase:
        """Internal fit implementation.

        Parameters
        ----------
        X : np.ndarray
            Independent variable (depends on protocol):
            - flow_curve: shear rate γ̇
            - oscillation: angular frequency ω
            - startup/creep/relaxation/laos: time t
        y : np.ndarray
            Dependent variable (depends on protocol):
            - flow_curve: stress σ or viscosity η
            - oscillation: G* or (G', G'')
            - startup/relaxation: stress σ(t)
            - creep: strain γ(t) or compliance J(t)
            - laos: stress σ(t)
        test_mode : str, optional
            Protocol: "flow_curve", "oscillation", "startup",
            "creep", "relaxation", "laos"
        **kwargs
            Additional protocol-specific parameters

        Returns
        -------
        ITTMCTBase
            self for method chaining
        """
        # Detect or validate protocol
        if test_mode is None:
            test_mode = self._detect_protocol(X, y, **kwargs)
        self._last_protocol = test_mode  # type: ignore[assignment]

        # Dispatch to protocol-specific fitting
        protocol_dispatch = {
            "flow_curve": self._fit_flow_curve,
            "oscillation": self._fit_oscillation,
            "startup": self._fit_startup,
            "creep": self._fit_creep,
            "relaxation": self._fit_relaxation,
            "laos": self._fit_laos,
        }

        fit_method = protocol_dispatch.get(test_mode)
        if fit_method is None:
            raise ValueError(
                f"Unknown test_mode '{test_mode}'. "
                f"Supported: {list(protocol_dispatch.keys())}"
            )

        return fit_method(X, y, **kwargs)  # type: ignore[operator]

    def _predict(
        self,
        X: np.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Internal predict implementation.

        Parameters
        ----------
        X : np.ndarray
            Independent variable array
        test_mode : str, optional
            Protocol. Uses last fit protocol if not specified.
        **kwargs
            Protocol-specific parameters

        Returns
        -------
        np.ndarray
            Model predictions
        """
        if test_mode is None:
            test_mode = self._last_protocol
        if test_mode is None:
            raise ValueError("No test_mode specified and model has not been fit.")

        # Dispatch to protocol-specific prediction
        protocol_dispatch = {
            "flow_curve": self._predict_flow_curve,
            "oscillation": self._predict_oscillation,
            "startup": self._predict_startup,
            "creep": self._predict_creep,
            "relaxation": self._predict_relaxation,
            "laos": self._predict_laos,
        }

        predict_method = protocol_dispatch.get(test_mode)
        if predict_method is None:
            raise ValueError(f"Unknown test_mode '{test_mode}'")

        return predict_method(X, **kwargs)  # type: ignore[operator]

    def _detect_protocol(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> str:
        """Attempt to detect protocol from data characteristics.

        Parameters
        ----------
        X : np.ndarray
            Independent variable
        y : np.ndarray
            Dependent variable
        **kwargs
            May contain protocol hints

        Returns
        -------
        str
            Detected protocol name
        """
        # Check for explicit hints in kwargs
        if "gamma_dot" in kwargs or "shear_rate" in kwargs:
            return "flow_curve"
        if "omega" in kwargs or "frequency" in kwargs:
            return "oscillation"
        if "gamma_0" in kwargs and "omega" in kwargs:
            return "laos"
        if "sigma_applied" in kwargs:
            return "creep"

        # Heuristics based on data shape
        if y.ndim == 2 and y.shape[1] == 2:
            # (G', G'') pair suggests oscillation
            return "oscillation"

        # Default to flow_curve
        logger.warning(
            "Could not auto-detect protocol, defaulting to 'flow_curve'. "
            "Specify test_mode explicitly for other protocols."
        )
        return "flow_curve"

    # =========================================================================
    # Protocol-specific fit methods (default implementations, can be overridden)
    # =========================================================================

    def _fit_flow_curve(
        self,
        gamma_dot: np.ndarray,
        sigma: np.ndarray,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to steady-state flow curve data.

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)
        sigma : np.ndarray
            Steady-state stress array (Pa)

        Returns
        -------
        ITTMCTBase
            self
        """
        # Default implementation uses NLSQ optimization
        # Subclasses can override for specialized fitting
        from rheojax.utils.optimization import fit_with_nlsq

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures gamma_dot, sigma, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            y_pred = self._predict_flow_curve(gamma_dot)
            return sigma - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    def _fit_oscillation(
        self,
        omega: np.ndarray,
        G_star: np.ndarray,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to SAOS (G', G'') data.

        Parameters
        ----------
        omega : np.ndarray
            Angular frequency (rad/s)
        G_star : np.ndarray
            Complex modulus. If 1D, interpreted as |G*|.
            If 2D with shape (n, 2), interpreted as (G', G'').

        Returns
        -------
        ITTMCTBase
            self
        """
        from rheojax.utils.optimization import fit_with_nlsq

        # Parse G_star format
        if G_star.ndim == 2:
            G_prime = G_star[:, 0]
            G_double_prime = G_star[:, 1]
            y_combined = np.concatenate([G_prime, G_double_prime])
        else:
            # Assume |G*|
            y_combined = G_star

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures omega, y_combined, G_star, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            G_pred = self._predict_oscillation(omega, return_components=True)
            if G_star.ndim == 2:
                y_pred = np.concatenate([G_pred[:, 0], G_pred[:, 1]])
            else:
                y_pred = np.sqrt(G_pred[:, 0] ** 2 + G_pred[:, 1] ** 2)
            return y_combined - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    def _fit_startup(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_dot: float = 1.0,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to startup flow data (stress growth).

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma : np.ndarray
            Stress response σ(t) (Pa)
        gamma_dot : float, default 1.0
            Applied shear rate (1/s)

        Returns
        -------
        ITTMCTBase
            self
        """
        from rheojax.utils.optimization import fit_with_nlsq

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures t, sigma, gamma_dot, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            y_pred = self._predict_startup(t, gamma_dot=gamma_dot)
            return sigma - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    def _fit_creep(
        self,
        t: np.ndarray,
        J: np.ndarray,
        sigma_applied: float = 1.0,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to creep compliance data.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        J : np.ndarray
            Creep compliance J(t) = γ(t)/σ₀ (1/Pa)
        sigma_applied : float, default 1.0
            Applied stress (Pa)

        Returns
        -------
        ITTMCTBase
            self
        """
        from rheojax.utils.optimization import fit_with_nlsq

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures t, J, sigma_applied, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            y_pred = self._predict_creep(t, sigma_applied=sigma_applied)
            return J - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    def _fit_relaxation(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_pre: float = 0.01,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to stress relaxation data.

        Parameters
        ----------
        t : np.ndarray
            Time array (s) after flow cessation
        sigma : np.ndarray
            Relaxing stress σ(t) (Pa)
        gamma_pre : float, default 0.01
            Pre-shear strain before relaxation

        Returns
        -------
        ITTMCTBase
            self
        """
        from rheojax.utils.optimization import fit_with_nlsq

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures t, sigma, gamma_pre, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            y_pred = self._predict_relaxation(t, gamma_pre=gamma_pre)
            return sigma - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    def _fit_laos(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_0: float = 0.1,
        omega: float = 1.0,
        **kwargs,
    ) -> ITTMCTBase:
        """Fit to LAOS data.

        Parameters
        ----------
        t : np.ndarray
            Time array (s)
        sigma : np.ndarray
            Stress response σ(t) (Pa)
        gamma_0 : float, default 0.1
            Strain amplitude
        omega : float, default 1.0
            Angular frequency (rad/s)

        Returns
        -------
        ITTMCTBase
            self
        """
        from rheojax.utils.optimization import fit_with_nlsq

        param_names = list(self.parameters.keys())
        initial_values = np.array([self.parameters.get_value(p) for p in param_names])
        # Convert bounds from list of tuples to (lower, upper) arrays
        param_bounds = self.parameters.get_bounds()
        lower = np.array([b[0] if b[0] is not None else -np.inf for b in param_bounds])
        upper = np.array([b[1] if b[1] is not None else np.inf for b in param_bounds])
        bounds = (lower, upper)

        # Closure captures t, sigma, gamma_0, omega, param_names, and bounds
        def residual_func(params):
            # Clip params to bounds to handle numerical precision issues
            params_clipped = jnp.clip(params, lower, upper)
            param_dict = dict(zip(param_names, params_clipped, strict=True))
            self.parameters.set_values(param_dict)
            y_pred = self._predict_laos(t, gamma_0=gamma_0, omega=omega)
            return sigma - y_pred

        result = fit_with_nlsq(
            residual_func,
            initial_values,
            bounds=bounds,
            **kwargs,
        )

        self.parameters.set_values(dict(zip(param_names, result.x, strict=True)))
        self._nlsq_result = result
        return self

    # =========================================================================
    # Protocol-specific predict methods (abstract - must be implemented)
    # =========================================================================

    @abstractmethod
    def _predict_flow_curve(
        self,
        gamma_dot: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Predict steady-state flow curve σ(γ̇).

        Parameters
        ----------
        gamma_dot : np.ndarray
            Shear rate array (1/s)

        Returns
        -------
        np.ndarray
            Steady-state stress σ (Pa)
        """
        pass

    @abstractmethod
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
            If False, return |G*|

        Returns
        -------
        np.ndarray
            Complex modulus or components
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
            Pre-shear strain

        Returns
        -------
        np.ndarray
            Relaxing stress σ(t) (Pa)
        """
        pass

    @abstractmethod
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
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_glass_transition_info(self) -> dict[str, Any]:
        """Get information about the glass transition state.

        Returns
        -------
        dict
            Glass transition properties including:
            - is_glass: whether current parameters are in glass state
            - epsilon: separation parameter
            - f_neq: non-ergodicity parameter (plateau height)
        """
        # Subclasses implement this based on their vertex parameters
        raise NotImplementedError("Subclass must implement get_glass_transition_info")

    def initialize_prony_modes(
        self,
        t_max: float = 1000.0,
        n_points: int = 1000,
    ) -> None:
        """Initialize Prony modes for Volterra integration.

        Computes equilibrium correlator and fits memory kernel to
        Prony series for efficient ODE integration.

        Parameters
        ----------
        t_max : float, default 1000.0
            Maximum time for correlator computation
        n_points : int, default 1000
            Number of time points
        """
        from rheojax.utils.mct_kernels import prony_decompose_memory

        # Compute equilibrium correlator on fine time grid
        t = np.logspace(-3, np.log10(t_max), n_points)
        t_jax = jnp.array(t)
        phi_eq = np.array(self._compute_equilibrium_correlator(t_jax))

        # Compute memory kernel from correlator
        m_t = np.array(self._compute_memory_kernel(jnp.array(phi_eq)))

        # Fit Prony series
        self._prony_amplitudes, self._prony_times = prony_decompose_memory(
            t, m_t, n_modes=self.n_prony_modes
        )

        # Store equilibrium correlator
        self._equilibrium_correlator = phi_eq

        logger.debug(
            f"Initialized {self.n_prony_modes} Prony modes, "
            f"tau range: [{self._prony_times.min():.2e}, {self._prony_times.max():.2e}]"
        )

    def get_prony_modes(self) -> tuple[np.ndarray, np.ndarray]:
        """Get Prony mode amplitudes and relaxation times.

        Returns
        -------
        g : np.ndarray
            Mode amplitudes
        tau : np.ndarray
            Mode relaxation times
        """
        if self._prony_amplitudes is None:
            self.initialize_prony_modes()
        assert self._prony_amplitudes is not None
        assert self._prony_times is not None
        return self._prony_amplitudes, self._prony_times

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"integration_method='{self.integration_method}', "
            f"n_prony_modes={self.n_prony_modes})"
        )
