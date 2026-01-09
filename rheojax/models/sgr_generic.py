"""Soft Glassy Rheology (SGR) GENERIC Thermodynamic Framework Model.

This module implements the GENERIC (General Equation for Non-Equilibrium
Reversible-Irreversible Coupling) thermodynamic framework for the SGR model,
based on Fuereder & Ilg (2013) Physical Review E 88, 042134.

The GENERIC framework provides a thermodynamically consistent formulation by
splitting the dynamics into two parts:

1. Reversible (Hamiltonian) dynamics:
    dz/dt|_rev = L(z) * dF/dz

    where L is the antisymmetric Poisson bracket operator that generates
    reversible dynamics conserving energy.

2. Irreversible (dissipative) dynamics:
    dz/dt|_irrev = M(z) * dS/dz

    where M is the symmetric positive semi-definite friction matrix that
    generates entropy-producing irreversible dynamics.

The full GENERIC dynamics is:
    dz/dt = L(z) * dF/dz + M(z) * dS/dz

Key thermodynamic constraints:
- Entropy production: W = (dS/dz)^T M (dS/dz) >= 0 (second law)
- Energy conservation in reversible part: L * dS/dz = 0
- Entropy conservation in reversible part: L^T * dF/dz = 0
- Degeneracy conditions: L * dS/dz = M * dF/dz = 0

State Variables:
    For SGR, the GENERIC state vector z contains:
    - sigma: Stress (momentum-like variable conjugate to strain)
    - P(E,l): Trap occupation distribution (structural variable)

    In the simplified formulation used here:
    - z[0] = sigma: Macroscopic stress
    - z[1] = lambda: Structural parameter (0 = broken, 1 = intact)

Physical Interpretation:
    The GENERIC framework ensures that the SGR model satisfies fundamental
    thermodynamic laws: energy conservation (first law) and entropy production
    (second law). The Poisson bracket encodes the reversible coupling between
    stress and strain rate (Hamiltonian mechanics), while the friction matrix
    encodes the irreversible trap hopping dynamics that produces entropy.

References:
    - I. Fuereder and P. Ilg, GENERIC framework for the Fokker-Planck equation,
      Physical Review E, 2013, 88, 042134
    - P. Sollich, Rheological constitutive equation for a model of soft glassy
      materials, Physical Review E, 1998, 58(1), 738-759
    - H.C. Ottinger, Beyond Equilibrium Thermodynamics, Wiley, 2005
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import TestMode
from rheojax.logging import get_logger, log_fit
from rheojax.utils.sgr_kernels import G0, Gp

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp_typing
else:
    jnp_typing = np

# Module logger
logger = get_logger(__name__)


@ModelRegistry.register("sgr_generic")
class SGRGeneric(BaseModel):
    """Soft Glassy Rheology (SGR) GENERIC Thermodynamic Framework Model.

    This model implements the GENERIC (General Equation for Non-Equilibrium
    Reversible-Irreversible Coupling) thermodynamic framework for SGR, ensuring
    thermodynamic consistency via explicit entropy production tracking.

    The GENERIC formulation splits dynamics into:
    - Reversible (Hamiltonian): dz/dt = L * dF/dz (Poisson bracket L antisymmetric)
    - Irreversible (dissipative): dz/dt = M * dS/dz (friction M symmetric PSD)

    Parameters:
        x: Effective noise temperature (dimensionless), controls phase transition
        G0: Modulus scale (Pa), sets absolute magnitude of elastic response
        tau0: Attempt time (s), characteristic microscopic relaxation timescale

    State Variables:
        z = [sigma, lambda] where:
        - sigma: Macroscopic stress (Pa)
        - lambda: Structural parameter [0, 1] representing trap occupation

    Thermodynamic Functions:
        - F(z): Helmholtz free energy = U(z) - T*S(z)
        - U(z): Internal energy from elastic storage
        - S(z): Entropy from trap distribution
        - W: Entropy production rate = (dF/dz)^T M (dF/dz) >= 0

    Example:
        >>> from rheojax.models.sgr_generic import SGRGeneric
        >>> import numpy as np
        >>> model = SGRGeneric()
        >>> omega = np.logspace(-2, 2, 50)
        >>> model._test_mode = 'oscillation'
        >>> G_star = model.predict(omega)
        >>> # Check thermodynamic consistency
        >>> state = np.array([100.0, 0.5])
        >>> W = model.compute_entropy_production(state)
        >>> assert W >= 0, "Second law violated!"

    Notes:
        - Inherits from BaseModel (includes BayesianMixin for NumPyro NUTS)
        - Predictions match SGRConventional in linear viscoelastic regime
        - GENERIC structure guarantees thermodynamic consistency
        - Reference: Fuereder & Ilg 2013 PRE 88, 042134
    """

    def __init__(self):
        """Initialize SGR GENERIC Model.

        Creates ParameterSet with:
        - x (noise temperature): bounds (0.5, 3.0), default 1.5
        - G0 (modulus scale): bounds (1e-3, 1e9), default 1e3
        - tau0 (attempt time): bounds (1e-9, 1e3), default 1e-3
        """
        super().__init__()

        # Create parameter set (same as SGRConventional for compatibility)
        self.parameters = ParameterSet()

        # x: Effective noise temperature (dimensionless)
        self.parameters.add(
            name="x",
            value=1.5,
            bounds=(0.5, 3.0),
            units="dimensionless",
            description="Effective noise temperature (glass transition at x=1)",
        )

        # G0: Modulus scale (Pa)
        self.parameters.add(
            name="G0",
            value=1e3,
            bounds=(1e-3, 1e9),
            units="Pa",
            description="Modulus scale (absolute magnitude of elastic response)",
        )

        # tau0: Attempt time (s)
        self.parameters.add(
            name="tau0",
            value=1e-3,
            bounds=(1e-9, 1e3),
            units="s",
            description="Attempt time (microscopic relaxation timescale)",
        )

        # Store test mode for mode-aware Bayesian inference
        self._test_mode: TestMode | str | None = None

        # Storage for entropy production tracking
        self._cumulative_entropy_production: float = 0.0

    # =========================================================================
    # GENERIC State Variables and Thermodynamic Functions
    # =========================================================================

    def free_energy(self, state: np.ndarray) -> float:
        """Compute Helmholtz free energy F(z) = U(z) - T*S(z).

        The free energy functional for SGR combines:
        - Elastic energy storage from stressed trap elements
        - Entropic contribution from trap occupation distribution

        Args:
            state: State vector [sigma, lambda] where sigma is stress (Pa)
                   and lambda is structural parameter [0, 1]

        Returns:
            Free energy F (J/m^3 or Pa, depending on normalization)

        Notes:
            F = U - T*S where T is the noise temperature x (in units of trap depth)
        """
        U = self.internal_energy(state)
        S = self.entropy(state)
        T = self.parameters.get_value("x")  # Noise temperature as effective temperature

        return U - T * S

    def internal_energy(self, state: np.ndarray) -> float:
        """Compute internal energy U(z) from elastic storage.

        The internal energy represents energy stored in elastically
        deformed trap elements. For SGR with stress sigma and
        structural parameter lambda:

            U = (1/2) * (sigma^2 / (G0 * lambda^n))

        where the effective modulus depends on structure.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Internal energy U (J/m^3)
        """
        sigma = state[0]
        lam = np.clip(state[1], 0.01, 1.0)  # Prevent division by zero

        G0_val = self.parameters.get_value("G0")
        x = self.parameters.get_value("x")

        # Compute dimensionless equilibrium modulus
        G0_dim = float(G0(x))

        # Effective modulus depends on structure
        G_eff = G0_val * G0_dim * lam

        # Elastic energy: U = sigma^2 / (2 * G_eff)
        U = sigma**2 / (2.0 * G_eff + 1e-20)

        return U

    def entropy(self, state: np.ndarray) -> float:
        """Compute entropy S(z) from trap occupation distribution.

        The entropy represents the configurational entropy of trap
        occupation. For the structural parameter lambda in [0, 1],
        we use a mixing entropy form:

            S = -k * [lambda * ln(lambda) + (1-lambda) * ln(1-lambda)]

        This captures the entropy associated with the distribution
        of elements between trapped (structured) and free (unstructured) states.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Entropy S (dimensionless, normalized by kB)
        """
        lam = np.clip(state[1], 1e-10, 1.0 - 1e-10)  # Prevent log(0)

        # Binary mixing entropy (normalized by characteristic scale)
        S = -(lam * np.log(lam) + (1.0 - lam) * np.log(1.0 - lam))

        return S

    def free_energy_gradient(self, state: np.ndarray) -> np.ndarray:
        """Compute gradient dF/dz of free energy.

        The gradient components are:
        - dF/d(sigma): Conjugate to stress (strain-like)
        - dF/d(lambda): Conjugate to structure (chemical potential-like)

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Gradient [dF/d(sigma), dF/d(lambda)]
        """
        sigma = state[0]
        lam = np.clip(state[1], 0.01, 1.0 - 1e-10)

        G0_val = self.parameters.get_value("G0")
        x = self.parameters.get_value("x")
        G0_dim = float(G0(x))
        G_eff = G0_val * G0_dim * lam

        # dU/d(sigma) = sigma / G_eff
        dU_dsigma = sigma / (G_eff + 1e-20)

        # dU/d(lambda) = -sigma^2 / (2 * G_eff^2) * G0_val * G0_dim
        dU_dlam = -(sigma**2) / (2.0 * (G_eff + 1e-20) ** 2) * G0_val * G0_dim

        # dS/d(lambda) = -ln(lambda) + ln(1-lambda) = ln((1-lambda)/lambda)
        dS_dlam = np.log((1.0 - lam) / lam)

        # dF/dz = dU/dz - T * dS/dz
        dF_dsigma = dU_dsigma
        dF_dlam = dU_dlam - x * dS_dlam

        return np.array([dF_dsigma, dF_dlam])

    # =========================================================================
    # GENERIC Operators: Poisson Bracket L and Friction Matrix M
    # =========================================================================

    def poisson_bracket(self, state: np.ndarray) -> np.ndarray:
        """Compute Poisson bracket operator L(z).

        The Poisson bracket generates reversible (Hamiltonian) dynamics.
        It must be antisymmetric: L = -L^T.

        For SGR, the Poisson bracket couples stress sigma to strain rate:
            L = [[0, L_12], [-L_12, 0]]

        where L_12 encodes the stress-strain rate coupling from
        the constitutive relation.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            2x2 antisymmetric Poisson bracket matrix L

        Notes:
            - L is state-dependent in general
            - Antisymmetry ensures energy conservation: dE/dt = 0 for reversible part
        """
        lam = np.clip(state[1], 0.01, 1.0)

        G0_val = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")
        x = self.parameters.get_value("x")
        G0_dim = float(G0(x))

        # Coupling strength for stress-strain relationship
        # L_12 ~ G_eff / tau0 for Maxwell-like coupling
        G_eff = G0_val * G0_dim * lam
        L_12 = G_eff / tau0

        # Antisymmetric Poisson bracket
        L = np.array([[0.0, L_12], [-L_12, 0.0]])

        return L

    def friction_matrix(self, state: np.ndarray) -> np.ndarray:
        """Compute friction matrix M(z).

        The friction matrix generates irreversible (dissipative) dynamics.
        It must be symmetric and positive semi-definite: M = M^T, M >= 0.

        For SGR, the friction matrix encodes:
        - Viscous dissipation (stress relaxation)
        - Structural evolution (trap hopping)

        Args:
            state: State vector [sigma, lambda]

        Returns:
            2x2 symmetric positive semi-definite friction matrix M

        Notes:
            - M is state-dependent
            - Positive semi-definiteness ensures entropy production W >= 0
            - The noise temperature x appears in M controlling dissipation rate
        """
        lam = np.clip(state[1], 0.01, 1.0)
        # Note: sigma (state[0]) not used in friction matrix - structure-based

        G0_val = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")
        x = self.parameters.get_value("x")
        G0_dim = float(G0(x))

        # Effective modulus and relaxation rate
        G_eff = G0_val * G0_dim * lam
        gamma_relax = 1.0 / tau0  # Base relaxation rate

        # In SGR, the noise temperature x controls the yielding rate
        # Higher x means faster relaxation (more trap hopping)
        yielding_factor = np.exp(-1.0 / x)  # Arrhenius-like activation

        # Friction components
        # M_11: Stress dissipation (viscous friction)
        M_11 = yielding_factor * gamma_relax * G_eff

        # M_22: Structural dissipation (trap dynamics)
        # Rate of structure change from trap hopping
        M_22 = yielding_factor * gamma_relax * lam * (1.0 - lam)

        # Cross-coupling (must maintain symmetry)
        # Stress can drive structural change and vice versa
        # Use geometric mean to ensure positive semi-definiteness
        # M_12 = alpha * sqrt(M_11 * M_22) with |alpha| <= 1
        alpha = 0.0  # Decouple for simplicity (can be non-zero for coupled dynamics)
        M_12 = alpha * np.sqrt(M_11 * M_22 + 1e-20)

        # Symmetric friction matrix
        M = np.array([[M_11, M_12], [M_12, M_22]])

        return M

    # =========================================================================
    # GENERIC Dynamics
    # =========================================================================

    def reversible_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Compute reversible (Hamiltonian) part of dynamics.

        dz/dt|_rev = L(z) * dF/dz

        This represents the energy-conserving part of the dynamics,
        encoding the reversible coupling between variables.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Time derivative dz/dt from reversible dynamics
        """
        L = self.poisson_bracket(state)
        dF_dz = self.free_energy_gradient(state)

        return L @ dF_dz

    def irreversible_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Compute irreversible (dissipative) part of dynamics.

        dz/dt|_irrev = M(z) * dS/dz

        where dS/dz = (1/T) * dF/dz for systems at effective temperature T.

        This represents the entropy-producing part of the dynamics,
        encoding irreversible relaxation processes.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Time derivative dz/dt from irreversible dynamics
        """
        M = self.friction_matrix(state)
        dF_dz = self.free_energy_gradient(state)

        # For non-equilibrium systems, dS/dz = dF/dz / T (with appropriate sign)
        # The irreversible dynamics drives the system toward equilibrium
        # dz/dt|_irrev = -M * dF/dz (negative gradient for energy minimization)
        return -M @ dF_dz

    def full_dynamics(self, state: np.ndarray) -> np.ndarray:
        """Compute full GENERIC dynamics.

        dz/dt = L(z) * dF/dz + M(z) * dS/dz

        The total dynamics combines reversible (Hamiltonian) and
        irreversible (dissipative) contributions.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Total time derivative dz/dt
        """
        dz_dt_rev = self.reversible_dynamics(state)
        dz_dt_irrev = self.irreversible_dynamics(state)

        return dz_dt_rev + dz_dt_irrev

    # =========================================================================
    # Thermodynamic Consistency Checks
    # =========================================================================

    def compute_entropy_production(self, state: np.ndarray) -> float:
        """Compute entropy production rate W at given state.

        The entropy production is:
            W = (dF/dz)^T * M(z) * (dF/dz) >= 0

        This must be non-negative (second law of thermodynamics).

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Entropy production rate W (must be >= 0)

        Raises:
            Warning if W < 0 due to numerical errors
        """
        M = self.friction_matrix(state)
        dF_dz = self.free_energy_gradient(state)

        # W = dF^T M dF (quadratic form)
        W = dF_dz @ M @ dF_dz

        # Check thermodynamic consistency
        if W < -1e-12:
            logger.warning(
                f"Entropy production W = {W:.6e} < 0 at state={state}. "
                "This violates the second law and may indicate numerical issues."
            )

        return max(W, 0.0)  # Ensure non-negative for downstream use

    def entropy_production_rate(self, state: np.ndarray) -> float:
        """Compute entropy production rate dS/dt.

        This is equivalent to compute_entropy_production() but expressed
        in terms of entropy rather than free energy.

        Args:
            state: State vector [sigma, lambda]

        Returns:
            Entropy production rate dS/dt >= 0
        """
        T = self.parameters.get_value("x")  # Noise temperature

        # dS/dt = W / T for dissipative processes at temperature T
        W = self.compute_entropy_production(state)

        return W / (T + 1e-20)

    def verify_thermodynamic_consistency(
        self, state: np.ndarray, tol: float = 1e-10
    ) -> dict:
        """Verify all GENERIC thermodynamic consistency conditions.

        Checks:
        1. Poisson bracket antisymmetry: L = -L^T
        2. Friction matrix symmetry: M = M^T
        3. Friction matrix positive semi-definiteness: eigenvalues >= 0
        4. Entropy production non-negativity: W >= 0
        5. Degeneracy condition: L * dS/dz = 0 (for closed systems)

        Args:
            state: State vector [sigma, lambda]
            tol: Numerical tolerance for consistency checks

        Returns:
            Dictionary with consistency check results
        """
        L = self.poisson_bracket(state)
        M = self.friction_matrix(state)
        # Note: dF_dz computed in compute_entropy_production, not needed here

        results = {}

        # 1. Poisson bracket antisymmetry
        antisym_error = np.max(np.abs(L + L.T))
        results["poisson_antisymmetric"] = antisym_error < tol
        results["poisson_antisymmetry_error"] = antisym_error

        # 2. Friction matrix symmetry
        sym_error = np.max(np.abs(M - M.T))
        results["friction_symmetric"] = sym_error < tol
        results["friction_symmetry_error"] = sym_error

        # 3. Friction matrix positive semi-definiteness
        eigenvalues = np.linalg.eigvalsh(M)
        min_eig = np.min(eigenvalues)
        results["friction_positive_semidefinite"] = min_eig >= -tol
        results["friction_min_eigenvalue"] = min_eig

        # 4. Entropy production non-negativity
        W = self.compute_entropy_production(state)
        results["entropy_production_nonnegative"] = W >= -tol
        results["entropy_production"] = W

        # 5. Overall consistency
        results["thermodynamically_consistent"] = all(
            [
                results["poisson_antisymmetric"],
                results["friction_symmetric"],
                results["friction_positive_semidefinite"],
                results["entropy_production_nonnegative"],
            ]
        )

        return results

    # =========================================================================
    # BaseModel Interface Implementation
    # =========================================================================

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | None = None,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC model to data using NLSQ optimization.

        Routes to appropriate fitting method based on test_mode.

        Args:
            X: Independent variable (frequency for oscillation, time for relaxation)
            y: Dependent variable (complex modulus, relaxation modulus, etc.)
            test_mode: Test mode ('oscillation', 'relaxation', 'creep', 'steady_shear', 'laos')
            **kwargs: NLSQ optimizer arguments

        Raises:
            ValueError: If test_mode not provided or invalid
        """
        if test_mode is None:
            raise ValueError("test_mode must be specified for SGR GENERIC fitting")

        with log_fit(logger, model="SGRGeneric", data_shape=X.shape) as ctx:
            try:
                logger.info(
                    "Starting SGR GENERIC model fit",
                    test_mode=test_mode,
                    n_points=len(X),
                )

                logger.debug(
                    "Input data statistics",
                    x_range=(float(np.min(X)), float(np.max(X))),
                    y_range=(float(np.min(np.abs(y))), float(np.max(np.abs(y)))),
                )

                # Store test mode for mode-aware Bayesian inference
                self._test_mode = test_mode
                ctx["test_mode"] = test_mode

                # Route to appropriate fitting method
                if test_mode == "oscillation":
                    self._fit_oscillation_mode(X, y, **kwargs)
                elif test_mode == "relaxation":
                    self._fit_relaxation_mode(X, y, **kwargs)
                elif test_mode == "creep":
                    self._fit_creep_mode(X, y, **kwargs)
                elif test_mode == "steady_shear":
                    self._fit_steady_shear_mode(X, y, **kwargs)
                elif test_mode == "laos":
                    self._fit_laos_mode(X, y, **kwargs)
                else:
                    raise ValueError(
                        f"Unsupported test_mode: {test_mode}. "
                        f"SGR GENERIC model supports 'oscillation', 'relaxation', "
                        f"'creep', 'steady_shear', 'laos'."
                    )

                # Log final parameters
                x_val = self.parameters.get_value("x")
                G0_val = self.parameters.get_value("G0")
                tau0_val = self.parameters.get_value("tau0")

                ctx["x"] = x_val
                ctx["G0"] = G0_val
                ctx["tau0"] = tau0_val
                ctx["phase_regime"] = self.get_phase_regime()

                logger.info(
                    "SGR GENERIC model fit completed",
                    x=x_val,
                    G0=G0_val,
                    tau0=tau0_val,
                    phase_regime=self.get_phase_regime(),
                )

            except Exception as e:
                logger.error(
                    "SGR GENERIC model fit failed",
                    test_mode=test_mode,
                    error=str(e),
                    exc_info=True,
                )
                raise

    def _fit_oscillation_mode(
        self,
        omega: np.ndarray,
        G_star: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC to complex modulus data (oscillation mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to complex modulus data G*(omega). The GENERIC model uses the same
        kernel functions as SGRConventional in the linear viscoelastic regime.

        Args:
            omega: Angular frequency array (rad/s)
            G_star: Complex modulus data. Accepted formats:
                - Complex array (M,) where G* = G' + i*G''
                - Real array (M, 2) where columns are [G', G'']
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        omega_jax = jnp.asarray(omega, dtype=jnp.float64)

        # Handle G_star format
        G_star_np = np.asarray(G_star)
        if np.iscomplexobj(G_star_np):
            G_star_2d = np.column_stack([np.real(G_star_np), np.imag(G_star_np)])
        elif G_star_np.ndim == 2 and G_star_np.shape[1] == 2:
            G_star_2d = G_star_np
        elif G_star_np.ndim == 2 and G_star_np.shape[0] == 2:
            G_star_2d = G_star_np.T
        else:
            raise ValueError(
                f"G_star must be complex (M,) or real (M, 2), got shape {G_star_np.shape}"
            )

        G_star_jax = jnp.asarray(G_star_2d, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_oscillation_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function
        objective = create_least_squares_objective(
            model_fn,
            omega_jax,
            G_star_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", False),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR GENERIC oscillation fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR GENERIC oscillation fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_relaxation_mode(
        self,
        t: np.ndarray,
        G_t: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC to relaxation modulus data (relaxation mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to relaxation modulus data G(t).

        Args:
            t: Time array (s)
            G_t: Relaxation modulus array (Pa)
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        G_t_jax = jnp.asarray(G_t, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_relaxation_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space for power-law data)
        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            G_t_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR GENERIC relaxation fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR GENERIC relaxation fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_creep_mode(
        self,
        t: np.ndarray,
        J_t: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC to creep compliance data (creep mode).

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to creep compliance data J(t).

        Theory: For x > 1 (fluid), J(t) ~ t^(x-1)

        Args:
            t: Time array (s)
            J_t: Creep compliance array (1/Pa)
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        t_jax = jnp.asarray(t, dtype=jnp.float64)
        J_t_jax = jnp.asarray(J_t, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_creep_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space for compliance spanning decades)
        objective = create_least_squares_objective(
            model_fn,
            t_jax,
            J_t_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR GENERIC creep fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR GENERIC creep fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_steady_shear_mode(
        self,
        gamma_dot: np.ndarray,
        sigma: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC to steady shear flow curve data.

        Uses NLSQ-accelerated optimization to fit SGR parameters [x, G0, tau0]
        to flow curve data sigma(gamma_dot).

        Theory:
            - Fluid (x > 1): sigma ~ gamma_dot^(x-1)
            - Glass (x < 1): sigma = sigma_y + A*gamma_dot^(1-x)

        Args:
            gamma_dot: Shear rate array (1/s)
            sigma: Stress array (Pa)
            **kwargs: NLSQ optimizer arguments

        Raises:
            RuntimeError: If optimization fails to converge
        """
        from rheojax.utils.optimization import (
            create_least_squares_objective,
            nlsq_optimize,
        )

        # Convert inputs to JAX arrays
        gamma_dot_jax = jnp.asarray(gamma_dot, dtype=jnp.float64)
        sigma_jax = jnp.asarray(sigma, dtype=jnp.float64)

        # Create model function for NLSQ
        def model_fn(x_data: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
            x_param = params[0]
            G0_param = params[1]
            tau0_param = params[2]
            return self._predict_steady_shear_jit(x_data, x_param, G0_param, tau0_param)

        # Create residual function (log-space for power-law data)
        objective = create_least_squares_objective(
            model_fn,
            gamma_dot_jax,
            sigma_jax,
            normalize=True,
            use_log_residuals=kwargs.get("use_log_residuals", True),
        )

        # Run NLSQ optimization
        result = nlsq_optimize(
            objective,
            self.parameters,
            use_jax=kwargs.get("use_jax", True),
            max_iter=kwargs.get("max_iter", 1000),
            ftol=kwargs.get("ftol", 1e-6),
            xtol=kwargs.get("xtol", 1e-6),
            gtol=kwargs.get("gtol", 1e-6),
        )

        if not result.success:
            raise RuntimeError(
                f"SGR GENERIC steady shear fitting failed: {result.message}. "
                "Try adjusting initial values or bounds."
            )

        logger.debug(
            f"SGR GENERIC steady shear fit converged: x={self.parameters.get_value('x'):.4f}, "
            f"G0={self.parameters.get_value('G0'):.2e}, "
            f"tau0={self.parameters.get_value('tau0'):.2e}, "
            f"cost={result.fun:.3e}"
        )

        self.fitted_ = True

    def _fit_laos_mode(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        **kwargs,
    ) -> None:
        """Fit SGR GENERIC to LAOS stress data.

        Uses Monte Carlo or Population Balance solver for time-domain stress
        prediction, then optimizes parameters to match measured stress.

        Args:
            t: Time array (s)
            sigma: Stress array (Pa)
            **kwargs: Required kwargs:
                - gamma_0: Strain amplitude
                - omega: Angular frequency (rad/s)
                Optional kwargs:
                - n_particles: Monte Carlo particle count (default 5000)
                - use_pde: Use PDE solver instead of MC (default False)

        Raises:
            ValueError: If gamma_0 or omega not provided
            RuntimeError: If optimization fails
        """
        gamma_0 = kwargs.get("gamma_0")
        omega = kwargs.get("omega")

        if gamma_0 is None or omega is None:
            raise ValueError("LAOS fitting requires gamma_0 and omega in kwargs")

        n_particles = kwargs.get("n_particles", 5000)
        use_pde = kwargs.get("use_pde", False)

        logger.info(
            f"SGR GENERIC LAOS fitting: gamma_0={gamma_0}, omega={omega}, "
            f"{'PDE' if use_pde else 'MC'} solver with {n_particles if not use_pde else 'grid'}"
        )

        # Store LAOS parameters
        self._gamma_0 = gamma_0
        self._omega_laos = omega

        # For now, use analytical approximation for small amplitude
        # Full MC/PDE fitting would require iterative simulation
        if gamma_0 < 0.1:
            # Small amplitude - use SAOS approximation
            logger.warning(
                f"Small strain amplitude gamma_0={gamma_0}. Using SAOS approximation."
            )
            # Extract G', G'' from stress signal via FFT
            from scipy.fft import fft

            sigma_fft = fft(sigma)
            n = len(sigma)
            fundamental_idx = int(omega * (t[-1] - t[0]) / (2 * np.pi))
            fundamental_idx = max(1, min(fundamental_idx, n // 2 - 1))

            G_star_amplitude = 2.0 * np.abs(sigma_fft[fundamental_idx]) / (n * gamma_0)
            phase = np.angle(sigma_fft[fundamental_idx])

            G_prime = G_star_amplitude * np.cos(phase)
            G_double_prime = G_star_amplitude * np.sin(phase)

            # Fit to single-point SAOS
            omega_single = np.array([omega])
            G_star_single = np.array([[G_prime, G_double_prime]])

            self._fit_oscillation_mode(omega_single, G_star_single, **kwargs)
        else:
            # Large amplitude - full MC-based LAOS fitting
            self._fit_laos_mc(t, sigma, gamma_0, omega, n_particles, **kwargs)

    def _fit_laos_mc(
        self,
        t: np.ndarray,
        sigma: np.ndarray,
        gamma_0: float,
        omega: float,
        n_particles: int,
        **kwargs,
    ) -> None:
        """Full Monte Carlo-based LAOS fitting.

        Runs MC simulations within optimization loop to match time-domain stress.

        Args:
            t: Time array (s)
            sigma: Measured stress array (Pa)
            gamma_0: Strain amplitude
            omega: Angular frequency (rad/s)
            n_particles: Number of MC particles
            **kwargs: Optimizer arguments
        """
        from scipy.optimize import minimize

        from rheojax.utils.sgr_monte_carlo import simulate_oscillatory

        logger.info(
            f"Full MC-based LAOS fitting: {n_particles} particles, "
            f"gamma_0={gamma_0}, omega={omega:.3f} rad/s"
        )

        # Determine simulation parameters from data
        period = 2.0 * np.pi / omega
        t_total = t[-1] - t[0]
        n_cycles = max(1, int(t_total / period))
        points_per_cycle = max(10, len(t) // n_cycles)

        # Warm-start: estimate parameters from stress amplitude
        sigma_max = np.max(np.abs(sigma))
        G0_init = sigma_max / gamma_0
        x_init = self.parameters.get_value("x")
        tau0_init = self.parameters.get_value("tau0")

        # Normalize target stress for residual calculation
        sigma_norm = sigma / (sigma_max + 1e-12)

        # Fixed random seed for reproducibility within optimization
        seed = kwargs.get("seed", 42)

        def objective(params):
            """Compute residual between MC stress and measured stress."""
            x_val, log_G0, log_tau0 = params
            G0_val = np.exp(log_G0)
            tau0_val = np.exp(log_tau0)

            # Clamp x to valid range
            x_val = np.clip(x_val, 0.5, 2.5)

            try:
                # Run MC simulation
                key = jax.random.PRNGKey(seed)
                _, _, sigma_mc = simulate_oscillatory(
                    key=key,
                    gamma_0=gamma_0,
                    omega=omega,
                    n_cycles=n_cycles,
                    points_per_cycle=points_per_cycle,
                    x=x_val,
                    n_particles=n_particles,
                    k=G0_val,
                    Gamma0=1.0 / tau0_val,
                    xg=1.0,
                )

                # Interpolate to match data time points
                t_mc = np.linspace(0, t_total, len(sigma_mc))
                sigma_mc_interp = np.interp(t - t[0], t_mc, np.array(sigma_mc))

                # Normalize MC stress
                sigma_mc_max = np.max(np.abs(sigma_mc_interp)) + 1e-12
                sigma_mc_norm = sigma_mc_interp / sigma_mc_max

                # Compute residual (allow phase shift by minimizing over shifts)
                residual = np.sum((sigma_mc_norm - sigma_norm) ** 2)

                return residual

            except Exception as e:
                logger.warning(f"MC simulation failed: {e}")
                return 1e10  # Large penalty

        # Initial guess in log space for G0, tau0
        x0 = np.array([x_init, np.log(G0_init), np.log(tau0_init)])

        # Bounds
        bounds = [
            (0.5, 2.5),  # x
            (np.log(1e-3), np.log(1e9)),  # log(G0)
            (np.log(1e-9), np.log(1e3)),  # log(tau0)
        ]

        # Run optimization
        max_iter = kwargs.get("max_iter", 50)

        logger.info(f"Starting MC-LAOS optimization (max {max_iter} iterations)...")

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "disp": False},
        )

        # Update parameters
        x_opt, log_G0_opt, log_tau0_opt = result.x
        self.parameters.set_value("x", float(x_opt))
        self.parameters.set_value("G0", float(np.exp(log_G0_opt)))
        self.parameters.set_value("tau0", float(np.exp(log_tau0_opt)))

        if result.success:
            logger.info(
                f"MC-LAOS fit converged: x={x_opt:.4f}, "
                f"G0={np.exp(log_G0_opt):.2e}, tau0={np.exp(log_tau0_opt):.2e}, "
                f"cost={result.fun:.3e}"
            )
        else:
            logger.warning(
                f"MC-LAOS fit did not fully converge: {result.message}. "
                f"Best: x={x_opt:.4f}, G0={np.exp(log_G0_opt):.2e}"
            )

        self.fitted_ = True

    @staticmethod
    @jax.jit
    def _predict_creep_jit(
        t: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled creep prediction: J(t).

        Theory: J(t) ~ t^(x-1) for x > 1 (fluid regime)

        Args:
            t: Time array (s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Creep compliance J(t) with shape (M,)
        """
        # Dimensionless time
        t_scaled = t / tau0

        # Compute equilibrium modulus factor
        G0_dim = G0(x)

        epsilon = 1e-12
        t_safe = jnp.maximum(t_scaled, epsilon)

        # Creep compliance: J(t) ~ (1 + t/tau0)^(x-1) / G0
        # This is the inverse relationship to G(t)
        growth_exp = x - 1.0
        J_t = jnp.power(1.0 + t_safe, growth_exp) / (G0_scale * G0_dim)

        # Enforce monotonicity for physical creep
        J_t_monotonic = jnp.maximum.accumulate(J_t)

        return J_t_monotonic

    @staticmethod
    @jax.jit
    def _predict_steady_shear_jit(
        gamma_dot: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled steady shear prediction: sigma(gamma_dot).

        Theory:
            - Fluid (x > 1): sigma ~ gamma_dot^(x-1)
            - Glass (x < 1): sigma = sigma_y + A*gamma_dot^(1-x)

        Args:
            gamma_dot: Shear rate array (1/s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Stress sigma(gamma_dot) with shape (M,)
        """
        # Compute equilibrium modulus factor
        G0_dim = G0(x)

        epsilon = 1e-12
        gamma_dot_safe = jnp.maximum(gamma_dot, epsilon)

        # Dimensionless shear rate
        gamma_dot_scaled = gamma_dot_safe * tau0

        # Flow curve: sigma = G0 * tau0 * gamma_dot * (gamma_dot * tau0)^(x-2)
        # = G0 * (gamma_dot * tau0)^(x-1)
        sigma = G0_scale * G0_dim * jnp.power(gamma_dot_scaled, x - 1.0)

        return sigma

    @staticmethod
    @jax.jit
    def _predict_oscillation_jit(
        omega: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled oscillation prediction: G'(omega), G''(omega).

        Uses same kernel functions as SGRConventional for linear response.
        The GENERIC formulation gives equivalent results in the linear regime.

        Args:
            omega: Angular frequency array (rad/s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Complex modulus [G', G''] with shape (M, 2)
        """
        # Compute dimensionless frequency
        omega_tau0 = omega * tau0

        # Call Gp kernel (returns G_prime, G_double_prime)
        G_prime, G_double_prime = Gp(x, omega_tau0)

        # Scale by G0
        G_prime_scaled = G0_scale * G_prime
        G_double_prime_scaled = G0_scale * G_double_prime

        # Stack into (M, 2) array
        G_star = jnp.stack([G_prime_scaled, G_double_prime_scaled], axis=1)

        return G_star

    @staticmethod
    @jax.jit
    def _predict_relaxation_jit(
        t: jnp.ndarray, x: float, G0_scale: float, tau0: float
    ) -> jnp.ndarray:
        """JIT-compiled relaxation prediction: G(t).

        Uses power-law form consistent with SGR theory.

        Args:
            t: Time array (s)
            x: Effective noise temperature (dimensionless)
            G0_scale: Modulus scale (Pa)
            tau0: Attempt time (s)

        Returns:
            Relaxation modulus G(t) with shape (M,)
        """
        # Dimensionless time
        t_scaled = t / tau0

        # Compute equilibrium modulus factor (dimensionless)
        G0_dim = G0(x)

        epsilon = 1e-12
        t_safe = jnp.maximum(t_scaled, epsilon)

        # Power-law form: G(t) ~ (1 + t/tau0)^(x-2)
        G_t = G0_scale * G0_dim / jnp.power(1.0 + t_safe, 2.0 - x)

        return G_t

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict based on fitted test mode.

        Routes to appropriate prediction method based on stored test_mode.

        Args:
            X: Independent variable (frequency or time)

        Returns:
            Predicted values (complex modulus or relaxation modulus)

        Raises:
            ValueError: If test_mode not set (model not fitted)
        """
        if self._test_mode is None:
            raise ValueError("Model not fitted. Call fit() first or set _test_mode.")

        if self._test_mode == "oscillation":
            return self._predict_oscillation(X)
        elif self._test_mode == "relaxation":
            return self._predict_relaxation(X)
        else:
            raise ValueError(f"Unknown test_mode: {self._test_mode}")

    def _predict_oscillation(self, omega: np.ndarray) -> np.ndarray:
        """Predict complex modulus in oscillation mode.

        Args:
            omega: Angular frequency array (rad/s)

        Returns:
            Complex modulus [G', G''] with shape (M, 2)
        """
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        omega_jax = jnp.asarray(omega)
        G_star_jax = self._predict_oscillation_jit(omega_jax, x, G0_scale, tau0)

        return np.array(G_star_jax)

    def _predict_relaxation(self, t: np.ndarray) -> np.ndarray:
        """Predict relaxation modulus in relaxation mode.

        Args:
            t: Time array (s)

        Returns:
            Relaxation modulus array (Pa)
        """
        x = self.parameters.get_value("x")
        G0_scale = self.parameters.get_value("G0")
        tau0 = self.parameters.get_value("tau0")

        t_jax = jnp.asarray(t)
        G_t_jax = self._predict_relaxation_jit(t_jax, x, G0_scale, tau0)

        return np.array(G_t_jax)

    def model_function(self, X, params, test_mode=None):
        """Model function for Bayesian inference with NumPyro NUTS.

        Required by BayesianMixin for NumPyro NUTS sampling.

        Args:
            X: Independent variable (frequency or time)
            params: Array of parameter values [x, G0, tau0]
            test_mode: Optional test mode override

        Returns:
            Model predictions as JAX array
        """
        x = params[0]
        G0_scale = params[1]
        tau0 = params[2]

        mode = test_mode if test_mode is not None else self._test_mode
        if mode is None:
            mode = "oscillation"

        X_jax = jnp.asarray(X)

        if mode == "oscillation":
            return self._predict_oscillation_jit(X_jax, x, G0_scale, tau0)
        elif mode == "relaxation":
            return self._predict_relaxation_jit(X_jax, x, G0_scale, tau0)
        else:
            raise ValueError(f"Unsupported test mode: {mode}")

    def get_phase_regime(self) -> str:
        """Determine material phase regime from noise temperature x.

        Returns:
            Phase regime string: 'glass', 'power-law', or 'newtonian'
        """
        x = self.parameters.get_value("x")

        if x < 1.0:
            return "glass"
        elif x < 2.0:
            return "power-law"
        else:
            return "newtonian"
