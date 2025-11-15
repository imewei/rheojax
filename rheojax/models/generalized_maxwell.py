"""Generalized Maxwell Model (Prony series) for multi-mode viscoelastic relaxation.

The Generalized Maxwell Model (GMM) extends the single Maxwell element to N modes,
providing a flexible framework for capturing complex relaxation spectra:

    E(t) = E_∞ + Σᵢ₌₁ᴺ Eᵢ exp(-t/τᵢ)

Key features:
- Tri-mode equality: relaxation, oscillation, and creep predictions
- Two-step NLSQ fitting with softmax penalty for physical constraints
- Transparent element minimization (auto-optimize N)
- Bayesian inference via NumPyro NUTS with warm-start
- Tiered Bayesian prior safety mechanism (fail-fast on bad NLSQ convergence)
- JIT-compiled predictions for GPU acceleration

References:
    - Park, S. W., & Schapery, R. A. (1999). Methods of interconversion between
      linear viscoelastic material functions. Part I—A numerical method based on
      Prony series. International Journal of Solids and Structures, 36(11), 1653-1675.
    - pyvisco: https://github.com/saintsfan342000/pyvisco
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, cast

import nlsq
import numpy as np

from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.utils.optimization import OptimizationResult
from rheojax.utils.prony import (
    compute_r_squared,
    create_prony_parameter_set,
    select_optimal_n,
    softmax_penalty,
)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp_typing
else:
    jnp_typing = np

# Get module logger
logger = logging.getLogger(__name__)


@ModelRegistry.register("generalized_maxwell")
class GeneralizedMaxwell(BaseModel):
    """Generalized Maxwell Model with N exponential relaxation modes.

    The GMM uses Prony series representation for tri-mode viscoelastic behavior:

    **Relaxation mode:**
        E(t) = E_∞ + Σᵢ₌₁ᴺ Eᵢ exp(-t/τᵢ)

    **Oscillation mode (closed-form Fourier transform):**
        E'(ω) = E_∞ + Σᵢ Eᵢ (ωτᵢ)²/(1+(ωτᵢ)²)
        E"(ω) = Σᵢ Eᵢ (ωτᵢ)/(1+(ωτᵢ)²)

    **Creep mode (numerical simulation):**
        J(t) = ε(t)/σ₀ via backward-Euler integration

    Parameters:
        n_modes: Number of relaxation modes (N)
        modulus_type: 'shear' (G) or 'tensile' (E)

    Attributes:
        parameters: ParameterSet containing E_inf, E_i, tau_i (or G equivalents)

    Example:
        >>> from rheojax.models.generalized_maxwell import GeneralizedMaxwell
        >>> import numpy as np
        >>> model = GeneralizedMaxwell(n_modes=3, modulus_type='shear')
        >>> t = np.logspace(-3, 2, 50)
        >>> G_data = ...  # Relaxation modulus data
        >>> model.fit(t, G_data, test_mode='relaxation', optimization_factor=1.5)
        >>> G_pred = model.predict(t)
    """

    def __init__(self, n_modes: int = 3, modulus_type: str = "shear"):
        """Initialize Generalized Maxwell Model.

        Args:
            n_modes: Number of exponential relaxation modes (N ≥ 1)
            modulus_type: 'shear' for G (default) or 'tensile' for E

        Raises:
            ValueError: If n_modes < 1 or modulus_type invalid
        """
        super().__init__()

        if n_modes < 1:
            raise ValueError(f"n_modes must be ≥ 1, got {n_modes}")

        if modulus_type not in ["shear", "tensile"]:
            raise ValueError(
                f"modulus_type must be 'shear' or 'tensile', got '{modulus_type}'"
            )

        self._n_modes = n_modes
        self._modulus_type = modulus_type
        self._test_mode: str | None = None

        # Create Prony parameter set
        self.parameters = create_prony_parameter_set(n_modes, modulus_type)

        # Store NLSQ result for warm-start and diagnostics
        self._nlsq_result: OptimizationResult | None = None

        # Store element minimization diagnostics
        self._element_minimization_diagnostics: dict[str, object] | None = None

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_mode: str | None = None,
        optimization_factor: float | None = 1.5,
        **kwargs,
    ) -> None:
        """Fit GMM to data using NLSQ optimization.

        Args:
            X: Independent variable (time or frequency)
            y: Dependent variable (modulus or compliance)
            test_mode: Test mode ('relaxation', 'oscillation', 'creep')
            optimization_factor: R² threshold multiplier for element minimization (None to disable)
            **kwargs: NLSQ optimizer arguments (max_iter, ftol, xtol, gtol)

        Raises:
            ValueError: If test_mode not provided or invalid
        """
        # Detect test mode
        if test_mode is None:
            raise ValueError("test_mode must be specified for GMM fitting")

        self._test_mode = test_mode

        # Route to appropriate fitting method
        if test_mode == "relaxation":
            self._fit_relaxation_mode(
                X, y, optimization_factor=optimization_factor, **kwargs
            )
        elif test_mode == "oscillation":
            self._fit_oscillation_mode(
                X, y, optimization_factor=optimization_factor, **kwargs
            )
        elif test_mode == "creep":
            self._fit_creep_mode(
                X, y, optimization_factor=optimization_factor, **kwargs
            )
        else:
            raise ValueError(f"Unknown test_mode: {test_mode}")

    def _nlsq_fit(
        self, objective, x0, bounds, max_nfev=1000, ftol=1e-6, xtol=1e-6, gtol=1e-6
    ) -> OptimizationResult:
        """NLSQ wrapper for consistent fitting across modes.

        Args:
            objective: Residual function
            x0: Initial parameter guess
            bounds: (lower, upper) parameter bounds
            max_nfev: Maximum function evaluations
            ftol: Function tolerance
            xtol: Parameter tolerance
            gtol: Gradient tolerance

        Returns:
            OptimizationResult with fitted parameters and diagnostics
        """
        ls = nlsq.LeastSquares()

        try:
            nlsq_result = ls.least_squares(
                objective,
                x0=np.asarray(x0),
                bounds=bounds,
                method="trf",
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
                max_nfev=max_nfev,
                verbose=0,
            )
        except ValueError as e:
            # Handle infeasible initial guess
            raise RuntimeError(
                f"NLSQ optimization failed with error: {e}\n"
                "This may indicate:\n"
                "  1. Data is unsuitable for GMM fitting (e.g., constant values)\n"
                "  2. Initial parameter guess is outside bounds\n"
                "  3. Too many modes for the available data"
            ) from e

        # Convert to OptimizationResult
        result = OptimizationResult(
            x=np.asarray(nlsq_result.x),
            fun=nlsq_result.cost,
            jac=np.asarray(nlsq_result.jac) if nlsq_result.jac is not None else None,
            success=nlsq_result.success,
            message=nlsq_result.message,
            nit=nlsq_result.nfev,
            nfev=nlsq_result.nfev,
            njev=nlsq_result.njev if hasattr(nlsq_result, "njev") else 0,
            optimality=(
                nlsq_result.optimality if hasattr(nlsq_result, "optimality") else None
            ),
            active_mask=(
                nlsq_result.active_mask if hasattr(nlsq_result, "active_mask") else None
            ),
            cost=nlsq_result.cost,
            grad=(
                np.asarray(nlsq_result.grad)
                if hasattr(nlsq_result, "grad") and nlsq_result.grad is not None
                else None
            ),
            nlsq_result=nlsq_result,
        )

        return result

    def _fit_relaxation_mode(
        self,
        t: np.ndarray,
        E_t: np.ndarray,
        optimization_factor: float | None = 1.5,
        **kwargs,
    ) -> None:
        """Fit GMM to relaxation modulus data.

        Args:
            t: Time array
            E_t: Relaxation modulus array
            optimization_factor: R² threshold multiplier for element minimization
            **kwargs: NLSQ optimizer arguments
        """
        # Extract kwargs
        max_iter = kwargs.get("max_iter", 1000)
        ftol = kwargs.get("ftol", 1e-6)
        xtol = kwargs.get("xtol", 1e-6)
        gtol = kwargs.get("gtol", 1e-6)

        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Define objective function
        def objective(params):
            """Residual for relaxation modulus."""
            E_inf = params[0]
            E_i = params[1 : 1 + self._n_modes]
            tau_i = params[1 + self._n_modes :]

            # Predict relaxation modulus
            E_pred = self._predict_relaxation_jit(jnp.asarray(t), E_inf, E_i, tau_i)

            return E_pred - E_t

        # Initial parameter guess
        E_inf_guess = jnp.min(E_t)  # Equilibrium modulus
        E_sum_guess = jnp.max(E_t) - E_inf_guess
        E_i_guess = jnp.full(self._n_modes, E_sum_guess / self._n_modes)
        tau_i_guess = jnp.logspace(-2, 2, self._n_modes)

        x0 = jnp.concatenate([jnp.array([E_inf_guess]), E_i_guess, tau_i_guess])

        # Parameter bounds
        bounds_lower = jnp.concatenate(
            [
                jnp.array([0.0]),
                jnp.full(self._n_modes, 1e-12),
                jnp.full(self._n_modes, 1e-6),
            ]
        )
        bounds_upper = jnp.concatenate(
            [
                jnp.array([jnp.max(E_t) * 10]),
                jnp.full(self._n_modes, jnp.max(E_t) * 10),
                jnp.full(self._n_modes, 1e6),
            ]
        )

        # Step 1: Fit with softmax penalty
        def objective_step1(params):
            """Objective with softmax penalty."""
            E_i = params[1 : 1 + self._n_modes]
            residual = objective(params)
            penalty = softmax_penalty(E_i, scale=1e-3)
            return jnp.concatenate([residual, jnp.array([penalty])])

        result_step1 = self._nlsq_fit(
            objective_step1,
            x0,
            bounds=(bounds_lower, bounds_upper),
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        # Check for negative Eᵢ
        params_opt = result_step1.x
        E_i_opt = params_opt[1 : 1 + self._n_modes]

        if jnp.any(E_i_opt < 0):
            logger.warning(
                "Negative Eᵢ detected in relaxation fit. Refitting with hard bounds."
            )

            # Step 2: Refit with hard bounds
            result_step2 = self._nlsq_fit(
                objective,
                params_opt,
                bounds=(bounds_lower, bounds_upper),
                max_nfev=max_iter,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )

            result_final = result_step2
            params_opt = result_final.x
        else:
            result_final = result_step1

        # Store NLSQ result
        self._nlsq_result = result_final

        # Set fitted parameters
        E_inf_opt = params_opt[0]
        E_i_opt = params_opt[1 : 1 + self._n_modes]
        tau_i_opt = params_opt[1 + self._n_modes :]

        self.parameters.set_value(f"{symbol}_inf", float(E_inf_opt))
        for i in range(self._n_modes):
            self.parameters.set_value(f"{symbol}_{i+1}", float(E_i_opt[i]))
            self.parameters.set_value(f"tau_{i+1}", float(tau_i_opt[i]))

        # Element minimization
        if optimization_factor is not None and self._n_modes > 1:
            self._apply_element_minimization(t, E_t, optimization_factor, **kwargs)

    def _apply_element_minimization(
        self, X: np.ndarray, y: np.ndarray, optimization_factor: float, **kwargs
    ) -> None:
        """Apply element minimization to reduce number of modes.

        Args:
            X: Independent variable (time or frequency)
            y: Dependent variable (modulus or compliance)
                - For relaxation/creep: 1D array of shape (M,)
                - For oscillation: 1D concatenated array [G', G"] of shape (2*M,)
            optimization_factor: R² threshold multiplier (e.g., 1.5 means N_opt where R²_N ≥ 1.5 * R²_min)
            **kwargs: NLSQ optimizer arguments
        """
        # Store initial n_modes for diagnostics
        n_initial = self._n_modes

        # Compute R² for current fit
        # For oscillation mode, need to flatten predictions to match 1D y
        y_pred_current = self.predict(X)
        if self._test_mode == "oscillation":
            # predict() returns (M, 2), flatten to (2*M,) to match y
            y_pred_current = (
                y_pred_current.T.flatten()
            )  # Transpose then flatten: [G', G"]

        # Iterative N reduction
        fit_results = {}
        for n in range(self._n_modes, 0, -1):
            # Create model with n modes
            model_n = GeneralizedMaxwell(n_modes=n, modulus_type=self._modulus_type)

            try:
                # Fit with n modes
                model_n.fit(
                    X, y, test_mode=self._test_mode, optimization_factor=None, **kwargs
                )

                # Compute R²
                y_pred_n = model_n.predict(X)
                if self._test_mode == "oscillation":
                    # Flatten predictions to match 1D y
                    y_pred_n = y_pred_n.T.flatten()

                r2_n = compute_r_squared(y, y_pred_n)

                fit_results[n] = {"r2": r2_n, "model": model_n}
            except (RuntimeError, ValueError):
                # Fitting failed for this N, skip
                logger.warning(f"Element minimization: fitting failed for n_modes={n}")
                break

        # Select optimal N
        r2_values = {n: cast(float, result["r2"]) for n, result in fit_results.items()}
        n_optimal = select_optimal_n(r2_values, optimization_factor=optimization_factor)

        # Store diagnostics with all required keys
        # Convert dict to arrays for test compatibility
        n_modes_list = sorted(r2_values.keys())
        r2_list = [r2_values[n] for n in n_modes_list]

        self._element_minimization_diagnostics = {
            "n_initial": n_initial,
            "r2": r2_list,  # R² values as list
            "n_modes": n_modes_list,  # Corresponding n_modes as list
            "n_optimal": n_optimal,
            "optimization_factor": optimization_factor,
        }

        # Update model if optimal N is different
        if n_optimal < self._n_modes:
            logger.info(
                f"Element minimization: reducing from {self._n_modes} to {n_optimal} modes"
            )

            # Copy parameters from optimal model
            optimal_model = cast(GeneralizedMaxwell, fit_results[n_optimal]["model"])
            self._n_modes = n_optimal
            self.parameters = optimal_model.parameters
            self._nlsq_result = optimal_model._nlsq_result

    def _fit_oscillation_mode(
        self,
        omega: np.ndarray,
        E_star: np.ndarray,
        optimization_factor: float | None = 1.5,
        **kwargs,
    ) -> None:
        """Fit GMM to complex modulus data.

        Args:
            omega: Angular frequency array
            E_star: Complex modulus [E', E"] - can be (2, M) or (M, 2)
            optimization_factor: R² threshold multiplier for element minimization
            **kwargs: NLSQ optimizer arguments
        """
        # Extract kwargs
        max_iter = kwargs.get("max_iter", 1000)
        ftol = kwargs.get("ftol", 1e-6)
        xtol = kwargs.get("xtol", 1e-6)
        gtol = kwargs.get("gtol", 1e-6)

        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Standardize input shape to (2, M)
        E_star = np.asarray(E_star)
        if E_star.ndim == 1:
            # Handle 1D concatenated [G', G"] from element minimization
            M = len(E_star) // 2
            E_prime = E_star[:M]
            E_double_prime = E_star[M:]
        elif E_star.shape[0] == 2:
            # Input is (2, M), extract directly
            E_prime = E_star[0]
            E_double_prime = E_star[1]  # FIX: Added missing assignment
        elif E_star.shape[1] == 2:
            # Input is (M, 2), transpose to (2, M)
            E_prime = E_star[:, 0]
            E_double_prime = E_star[:, 1]
        else:
            raise ValueError(
                f"E_star must have shape (2, M), (M, 2), or be 1D concatenated [G', G\"], got {E_star.shape}"
            )

        # Define objective function
        def objective(params):
            """Residual for complex modulus."""
            E_inf = params[0]
            E_i = params[1 : 1 + self._n_modes]
            tau_i = params[1 + self._n_modes :]

            # Predict complex modulus (returns (2, M))
            E_star_pred = self._predict_oscillation_jit(
                jnp.asarray(omega), E_inf, E_i, tau_i
            )
            E_prime_pred = E_star_pred[0]  # Extract G' from (2, M)
            E_double_prime_pred = E_star_pred[1]  # Extract G" from (2, M)

            # Combined residual
            residual_prime = E_prime_pred - E_prime
            residual_double_prime = E_double_prime_pred - E_double_prime

            return jnp.concatenate([residual_prime, residual_double_prime])

        # Initial parameter guess
        E_inf_guess = jnp.min(E_prime)  # Low-frequency plateau
        E_i_guess = jnp.full(
            self._n_modes, (jnp.max(E_prime) - E_inf_guess) / self._n_modes
        )
        tau_i_guess = jnp.logspace(-2, 2, self._n_modes)

        x0 = jnp.concatenate([jnp.array([E_inf_guess]), E_i_guess, tau_i_guess])

        # Parameter bounds
        bounds_lower = jnp.concatenate(
            [
                jnp.array([0.0]),
                jnp.full(self._n_modes, 1e-12),
                jnp.full(self._n_modes, 1e-6),
            ]
        )
        bounds_upper = jnp.concatenate(
            [
                jnp.array([jnp.max(E_prime) * 10]),
                jnp.full(self._n_modes, jnp.max(E_prime) * 10),
                jnp.full(self._n_modes, 1e6),
            ]
        )

        # Step 1: Fit with softmax penalty
        def objective_step1(params):
            """Objective with softmax penalty."""
            E_i = params[1 : 1 + self._n_modes]
            residual = objective(params)
            penalty = softmax_penalty(E_i, scale=1e-3)
            return jnp.concatenate([residual, jnp.array([penalty])])

        result_step1 = self._nlsq_fit(
            objective_step1,
            x0,
            bounds=(bounds_lower, bounds_upper),
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        # Check for negative Eᵢ
        params_opt = result_step1.x
        E_i_opt = params_opt[1 : 1 + self._n_modes]

        if jnp.any(E_i_opt < 0):
            logger.warning(
                "Negative Eᵢ detected in oscillation fit. Refitting with hard bounds."
            )

            # Step 2: Refit with hard bounds
            result_step2 = self._nlsq_fit(
                objective,
                params_opt,
                bounds=(bounds_lower, bounds_upper),
                max_nfev=max_iter,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )

            result_final = result_step2
            params_opt = result_final.x
        else:
            result_final = result_step1

        # Store NLSQ result
        self._nlsq_result = result_final

        # Set fitted parameters
        E_inf_opt = params_opt[0]
        E_i_opt = params_opt[1 : 1 + self._n_modes]
        tau_i_opt = params_opt[1 + self._n_modes :]

        self.parameters.set_value(f"{symbol}_inf", float(E_inf_opt))
        for i in range(self._n_modes):
            self.parameters.set_value(f"{symbol}_{i+1}", float(E_i_opt[i]))
            self.parameters.set_value(f"tau_{i+1}", float(tau_i_opt[i]))

        # Element minimization
        if optimization_factor is not None and self._n_modes > 1:
            # Reconstruct combined data for minimization (flatten to 1D)
            combined_data = np.concatenate([E_prime, E_double_prime])
            self._apply_element_minimization(
                omega, combined_data, optimization_factor, **kwargs
            )

    def _fit_creep_mode(
        self,
        t: np.ndarray,
        J_t: np.ndarray,
        optimization_factor: float | None = 1.5,
        **kwargs,
    ) -> None:
        """Fit GMM to creep compliance data.

        Args:
            t: Time array
            J_t: Creep compliance array
            optimization_factor: R² threshold multiplier for element minimization
            **kwargs: NLSQ optimizer arguments
        """
        # Extract kwargs
        max_iter = kwargs.get("max_iter", 1000)
        ftol = kwargs.get("ftol", 1e-6)
        xtol = kwargs.get("xtol", 1e-6)
        gtol = kwargs.get("gtol", 1e-6)

        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Define objective function (predict creep from GMM simulation)
        def objective(params):
            """Residual for creep compliance."""
            E_inf = params[0]
            E_i = params[1 : 1 + self._n_modes]
            tau_i = params[1 + self._n_modes :]

            # Predict creep compliance via GMM simulation
            # Apply step stress σ₀ = 1, solve for strain ε(t), compute J(t) = ε(t)/σ₀
            J_pred = self._predict_creep_internal(t, E_inf, E_i, tau_i)

            return J_pred - J_t

        # Initial parameter guess
        # For creep: J_0 = 1/(E_∞ + ΣEᵢ), J_∞ = 1/E_∞
        J_0 = jnp.min(J_t)  # Initial compliance (instant response)
        J_inf = jnp.max(J_t)  # Final compliance (long-time)

        # E_∞ corresponds to long-time equilibrium: J_∞ = 1/E_∞
        E_inf_guess = 1.0 / J_inf

        # Total instant modulus: J_0 = 1/(E_∞ + ΣEᵢ)
        E_total_guess = 1.0 / J_0
        E_sum_guess = max(E_total_guess - E_inf_guess, 1e-12)

        E_i_guess = jnp.full(self._n_modes, E_sum_guess / self._n_modes)
        tau_i_guess = jnp.logspace(-2, 2, self._n_modes)

        x0 = jnp.concatenate(
            [jnp.array([max(E_inf_guess, 1e-12)]), E_i_guess, tau_i_guess]
        )

        # Parameter bounds
        bounds_lower = jnp.concatenate(
            [
                jnp.array([0.0]),
                jnp.full(self._n_modes, 1e-12),
                jnp.full(self._n_modes, 1e-6),
            ]
        )
        bounds_upper = jnp.concatenate(
            [
                jnp.array([1.0 / J_0 * 10]),
                jnp.full(self._n_modes, 1.0 / J_0 * 10),
                jnp.full(self._n_modes, 1e6),
            ]
        )

        # Step 1: Fit with softmax penalty
        def objective_step1(params):
            """Objective with softmax penalty."""
            E_i = params[1 : 1 + self._n_modes]
            residual = objective(params)
            penalty = softmax_penalty(E_i, scale=1e-3)
            return jnp.concatenate([residual, jnp.array([penalty])])

        result_step1 = self._nlsq_fit(
            objective_step1,
            x0,
            bounds=(bounds_lower, bounds_upper),
            max_nfev=max_iter,
            ftol=ftol,
            xtol=xtol,
            gtol=gtol,
        )

        # Check for negative Eᵢ
        params_opt = result_step1.x
        E_i_opt = params_opt[1 : 1 + self._n_modes]

        if jnp.any(E_i_opt < 0):
            logger.warning(
                "Negative Eᵢ detected in creep fit. Refitting with hard bounds."
            )

            # Step 2: Refit with hard bounds
            result_step2 = self._nlsq_fit(
                objective,
                params_opt,
                bounds=(bounds_lower, bounds_upper),
                max_nfev=max_iter,
                ftol=ftol,
                xtol=xtol,
                gtol=gtol,
            )

            result_final = result_step2
            params_opt = result_final.x
        else:
            result_final = result_step1

        # Store NLSQ result
        self._nlsq_result = result_final

        # Set fitted parameters
        E_inf_opt = params_opt[0]
        E_i_opt = params_opt[1 : 1 + self._n_modes]
        tau_i_opt = params_opt[1 + self._n_modes :]

        self.parameters.set_value(f"{symbol}_inf", float(E_inf_opt))
        for i in range(self._n_modes):
            self.parameters.set_value(f"{symbol}_{i+1}", float(E_i_opt[i]))
            self.parameters.set_value(f"tau_{i+1}", float(tau_i_opt[i]))

        # Element minimization
        if optimization_factor is not None and self._n_modes > 1:
            self._apply_element_minimization(t, J_t, optimization_factor, **kwargs)

    def _predict_creep_internal(
        self,
        t: np.ndarray | jnp_typing.ndarray,
        E_inf: float,
        E_i: jnp_typing.ndarray,
        tau_i: jnp_typing.ndarray,
        sigma_0: float = 1.0,
    ) -> jnp_typing.ndarray:
        """Internal creep prediction for optimization.

        Args:
            t: Time array
            E_inf: Equilibrium modulus
            E_i: Prony coefficients (N,)
            tau_i: Relaxation times (N,)
            sigma_0: Applied stress (default 1.0)

        Returns:
            Creep compliance J(t)
        """
        # Call JIT-compiled creep prediction
        J_t = self._predict_creep_jit(jnp.asarray(t), E_inf, E_i, tau_i, sigma_0)
        return J_t

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Predict based on fitted test mode.

        Args:
            X: Independent variable (time or frequency)

        Returns:
            Predicted values (modulus or compliance)

        Raises:
            ValueError: If test_mode not set (model not fitted)
        """
        if self._test_mode is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Route to appropriate prediction method
        if self._test_mode == "relaxation":
            return self._predict_relaxation(X)
        elif self._test_mode == "oscillation":
            return self._predict_oscillation(X)
        elif self._test_mode == "creep":
            return self._predict_creep(X)
        else:
            raise ValueError(f"Unknown test_mode: {self._test_mode}")

    @staticmethod
    @jax.jit
    def _predict_relaxation_jit(
        t: jnp_typing.ndarray,
        E_inf: float,
        E_i: jnp_typing.ndarray,
        tau_i: jnp_typing.ndarray,
    ) -> jnp_typing.ndarray:
        """JIT-compiled relaxation prediction.

        Args:
            t: Time array
            E_inf: Equilibrium modulus
            E_i: Prony coefficients (N,)
            tau_i: Relaxation times (N,)

        Returns:
            Relaxation modulus E(t)
        """
        # E(t) = E_∞ + Σᵢ Eᵢ exp(-t/τᵢ)
        E_t = E_inf + jnp.sum(
            E_i[:, None] * jnp.exp(-t[None, :] / tau_i[:, None]), axis=0
        )
        return E_t

    def _predict_relaxation(self, t: np.ndarray | jnp_typing.ndarray) -> np.ndarray:
        """Predict relaxation modulus E(t).

        Args:
            t: Time array

        Returns:
            Relaxation modulus array
        """
        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Extract parameters
        E_inf = self.parameters.get_value(f"{symbol}_inf")
        E_i = jnp.array(
            [self.parameters.get_value(f"{symbol}_{i+1}") for i in range(self._n_modes)]
        )
        tau_i = jnp.array(
            [self.parameters.get_value(f"tau_{i+1}") for i in range(self._n_modes)]
        )

        # Convert input to JAX array
        t_jax = jnp.asarray(t)

        # Call JIT-compiled prediction
        E_t = self._predict_relaxation_jit(t_jax, E_inf, E_i, tau_i)

        return np.asarray(E_t)

    @staticmethod
    @jax.jit
    def _predict_oscillation_jit(
        omega: jnp_typing.ndarray,
        E_inf: float,
        E_i: jnp_typing.ndarray,
        tau_i: jnp_typing.ndarray,
    ) -> jnp_typing.ndarray:
        """JIT-compiled oscillation prediction.

        Args:
            omega: Angular frequency array
            E_inf: Equilibrium modulus
            E_i: Prony coefficients (N,)
            tau_i: Relaxation times (N,)

        Returns:
            Complex modulus [E', E"] (2, M)
        """
        # Closed-form Fourier transform
        omega_tau = omega[None, :] * tau_i[:, None]
        omega_tau_sq = omega_tau**2

        # E'(ω) = E_∞ + Σᵢ Eᵢ (ωτᵢ)²/(1+(ωτᵢ)²)
        E_prime = E_inf + jnp.sum(
            E_i[:, None] * omega_tau_sq / (1 + omega_tau_sq), axis=0
        )

        # E"(ω) = Σᵢ Eᵢ (ωτᵢ)/(1+(ωτᵢ)²)
        E_double_prime = jnp.sum(E_i[:, None] * omega_tau / (1 + omega_tau_sq), axis=0)

        # Return as (2, M) for standard complex modulus convention
        return jnp.stack([E_prime, E_double_prime], axis=0)

    def _predict_oscillation(
        self, omega: np.ndarray | jnp_typing.ndarray
    ) -> np.ndarray:
        """Predict complex modulus in oscillation mode.

        Args:
            omega: Angular frequency array

        Returns:
            Complex modulus [E', E"] (M, 2)
        """
        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Extract parameters
        E_inf = self.parameters.get_value(f"{symbol}_inf")
        E_i = jnp.array(
            [self.parameters.get_value(f"{symbol}_{i+1}") for i in range(self._n_modes)]
        )
        tau_i = jnp.array(
            [self.parameters.get_value(f"tau_{i+1}") for i in range(self._n_modes)]
        )

        # Convert input to JAX array
        omega_jax = jnp.asarray(omega)

        # Call JIT-compiled prediction (returns (2, M))
        E_star = self._predict_oscillation_jit(omega_jax, E_inf, E_i, tau_i)

        # Transpose to (M, 2) for user-facing API
        return np.asarray(E_star).T

    @staticmethod
    @jax.jit
    def _predict_creep_jit(
        t: jnp_typing.ndarray,
        E_inf: float,
        E_i: jnp_typing.ndarray,
        tau_i: jnp_typing.ndarray,
        sigma_0: float = 1.0,
    ) -> jnp_typing.ndarray:
        """JIT-compiled creep prediction via backward-Euler.

        Args:
            t: Time array
            E_inf: Equilibrium modulus
            E_i: Prony coefficients (N,)
            tau_i: Relaxation times (N,)
            sigma_0: Applied stress (default 1.0)

        Returns:
            Creep compliance J(t)
        """
        # Backward-Euler scheme for unconditional stability
        # GMM ODEs: dσᵢ/dt = -σᵢ/τᵢ + Eᵢ dε/dt
        # Total stress: σ = E_∞ ε + Σᵢ σᵢ
        # Apply step stress σ₀, solve for ε(t), compute J(t) = ε(t)/σ₀

        n_steps = len(t)
        n_modes = len(E_i)

        # Initialize arrays
        epsilon = jnp.zeros(n_steps)

        # Time step (assume uniform spacing for now, handle variable later)
        dt = jnp.diff(t, prepend=0.0)

        # Backward-Euler update loop
        def update_step(carry, inputs):
            """Update internal variables and strain."""
            eps_prev, sig_i_prev = carry
            t_curr, dt_curr = inputs

            # Protect against zero dt at first step
            dt_safe = jnp.maximum(dt_curr, 1e-12)

            # Solve for new strain from total stress balance
            # σ₀ = E_∞ εⁿ⁺¹ + Σᵢ σᵢⁿ⁺¹
            # σᵢⁿ⁺¹ = (σᵢⁿ + Eᵢ Δε) / (1 + Δt/τᵢ)
            # Substitute and solve for Δε

            # Coefficients for backward-Euler
            alpha_i = jnp.exp(-dt_safe / tau_i)  # Exact exponential integration
            beta_i = E_i * tau_i * (1 - alpha_i) / dt_safe

            # Total effective modulus
            E_eff = E_inf + jnp.sum(beta_i)

            # Solve for strain increment
            stress_from_prev = jnp.sum(alpha_i * sig_i_prev)
            d_eps = (sigma_0 - stress_from_prev) / E_eff
            eps_new = eps_prev + d_eps

            # Update internal stresses
            sig_i_new = alpha_i * sig_i_prev + beta_i * d_eps

            return (eps_new, sig_i_new), eps_new

        # Initialize
        eps_init = 0.0
        sig_i_init = jnp.zeros(n_modes)

        # Scan over time steps
        _, epsilon = jax.lax.scan(update_step, (eps_init, sig_i_init), (t, dt))

        # Compute compliance
        J_t = epsilon / sigma_0

        return J_t

    def _predict_creep(self, t: np.ndarray | jnp_typing.ndarray) -> np.ndarray:
        """Predict creep compliance J(t).

        Args:
            t: Time array

        Returns:
            Creep compliance array
        """
        symbol = "E" if self._modulus_type == "tensile" else "G"

        # Extract parameters
        E_inf = self.parameters.get_value(f"{symbol}_inf")
        E_i = jnp.array(
            [self.parameters.get_value(f"{symbol}_{i+1}") for i in range(self._n_modes)]
        )
        tau_i = jnp.array(
            [self.parameters.get_value(f"tau_{i+1}") for i in range(self._n_modes)]
        )

        # Convert input to JAX array
        t_jax = jnp.asarray(t)

        # Call JIT-compiled prediction
        J_t = self._predict_creep_jit(t_jax, E_inf, E_i, tau_i, sigma_0=1.0)

        return np.asarray(J_t)

    def _extract_nlsq_diagnostics(self, nlsq_result) -> dict:
        """Extract diagnostics from NLSQ OptimizationResult.

        Args:
            nlsq_result: OptimizationResult from nlsq_optimize()

        Returns:
            Dictionary with diagnostic metrics
        """
        # Extract convergence flag
        convergence_flag = nlsq_result.success

        # Extract gradient norm (optimality metric)
        gradient_norm = (
            nlsq_result.optimality if nlsq_result.optimality is not None else np.inf
        )

        # Estimate Hessian condition number from Jacobian
        # For least-squares: Hessian ≈ J^T J
        if nlsq_result.jac is not None:
            jac = np.asarray(nlsq_result.jac)
            # Compute approximate Hessian
            hessian_approx = jac.T @ jac
            # Compute condition number (ratio of largest/smallest singular values)
            try:
                cond_number = np.linalg.cond(hessian_approx)
            except np.linalg.LinAlgError:
                cond_number = np.inf
        else:
            cond_number = np.inf

        # Estimate parameter uncertainties from diagonal of covariance matrix
        # Cov ≈ inv(J^T J) if well-conditioned
        param_uncertainties = {}
        symbol = "E" if self._modulus_type == "tensile" else "G"

        if nlsq_result.jac is not None and cond_number < 1e10:
            try:
                # Compute covariance matrix
                cov_matrix = np.linalg.inv(hessian_approx)
                std_devs = np.sqrt(np.abs(np.diag(cov_matrix)))

                # Map to parameter names
                param_names = [f"{symbol}_inf"]
                param_names += [f"{symbol}_{i+1}" for i in range(self._n_modes)]
                param_names += [f"tau_{i+1}" for i in range(self._n_modes)]

                for i, name in enumerate(param_names):
                    if i < len(std_devs):
                        param_uncertainties[name] = float(std_devs[i])
            except (np.linalg.LinAlgError, ValueError):
                # Covariance matrix computation failed
                pass

        # Check proximity to bounds
        params_near_bounds = {}
        for param_name in self.parameters.keys():
            value = self.parameters.get_value(param_name)
            bounds = self.parameters.get(param_name).bounds
            lower, upper = bounds

            # Check if within 10% of bounds
            bound_range = upper - lower
            if abs(value - lower) < 0.1 * bound_range:
                params_near_bounds[param_name] = "lower"
            elif abs(value - upper) < 0.1 * bound_range:
                params_near_bounds[param_name] = "upper"

        return {
            "convergence_flag": convergence_flag,
            "gradient_norm": gradient_norm,
            "hessian_condition": cond_number,
            "param_uncertainties": param_uncertainties,
            "params_near_bounds": params_near_bounds,
        }

    def _classify_nlsq_convergence(self, diagnostics: dict) -> str:
        """Classify NLSQ convergence quality.

        Args:
            diagnostics: Dictionary from _extract_nlsq_diagnostics()

        Returns:
            Classification: 'hard_failure', 'suspicious', or 'good'
        """
        # Hard failure conditions
        if not diagnostics["convergence_flag"]:
            return "hard_failure"

        # GMM-specific: High Hessian condition and params near bounds are often acceptable
        # Only classify as suspicious if BOTH conditions are true AND uncertainties are high

        # Check if any uncertainties are > 100% of parameter value (very unreliable)
        high_uncertainty_count = 0
        for param_name, std_dev in diagnostics["param_uncertainties"].items():
            value = self.parameters.get_value(param_name)
            if abs(value) > 1e-12 and std_dev / abs(value) > 1.0:
                high_uncertainty_count += 1

        # Suspicious if: (high condition OR many params near bounds) AND high uncertainties
        if (
            high_uncertainty_count > self._n_modes
        ):  # More than half the parameters are highly uncertain
            if (
                diagnostics["hessian_condition"] > 1e10
                or len(diagnostics["params_near_bounds"]) > self._n_modes
            ):
                return "suspicious"

        # Good convergence if optimizer says so
        return "good"

    def _construct_bayesian_priors(
        self,
        classification: str,
        prior_mode: str = "warn",
        allow_fallback_priors: bool = False,
    ) -> dict:
        """Construct Bayesian priors based on NLSQ convergence classification.

        Args:
            classification: 'hard_failure', 'suspicious', or 'good'
            prior_mode: 'strict', 'warn', or 'auto_widen'
            allow_fallback_priors: Enable generic priors on hard failure

        Returns:
            Dictionary of priors for NumPyro: {param_name: {'mean': float, 'std': float}}

        Raises:
            ValueError: If hard failure and prior_mode='strict' or allow_fallback_priors=False
        """
        priors = {}

        if classification == "hard_failure":
            # Hard failure: raise error or use fallback priors
            if prior_mode == "strict" or not allow_fallback_priors:
                raise ValueError(
                    "NLSQ optimization failed or did not converge properly. "
                    "Cannot construct reliable priors from failed fit. "
                    "Please:\n"
                    "  1. Check model suitability for your data\n"
                    "  2. Adjust initial values or bounds\n"
                    "  3. Increase max_iter if optimization terminated early\n"
                    "  4. Provide manual priors via fit_bayesian(priors={...})\n"
                    "  5. Set allow_fallback_priors=True for generic weakly informative priors (not recommended)"
                )

            # Fallback: generic weakly informative priors
            warnings.warn(
                "WARNING: NLSQ optimization failed. Using generic weakly informative priors. "
                "Results may not be reliable. Consider manual prior specification.",
                UserWarning,
                stacklevel=2,
            )

            # Use parameter bounds as guides for generic priors
            for param_name in self.parameters.keys():
                bounds = self.parameters.get(param_name).bounds
                lower, upper = bounds
                mean = (lower + upper) / 2
                std = (upper - lower) / 4  # Wide prior covering ~95% of bounds
                priors[param_name] = {"mean": mean, "std": std}

        elif classification == "suspicious":
            # Suspicious: use safer priors, optionally widen
            if prior_mode == "auto_widen":
                warnings.warn(
                    "Suspicious NLSQ convergence detected (high Hessian condition, params near bounds, or high uncertainty). "
                    "Using inflated priors centered at NLSQ estimates.",
                    UserWarning,
                    stacklevel=2,
                )

                # Center at NLSQ, inflate std
                for param_name in self.parameters.keys():
                    value = self.parameters.get_value(param_name)
                    bounds = self.parameters.get(param_name).bounds
                    lower, upper = bounds

                    # Inflate std to 50% of estimate or 10% of bounds, whichever is larger
                    std_from_estimate = 0.5 * abs(value)
                    std_from_bounds = 0.1 * (upper - lower)
                    std = max(std_from_estimate, std_from_bounds)

                    priors[param_name] = {"mean": value, "std": std}
            else:
                # Warn mode: decouple from Hessian, use wider priors
                logger.warning(
                    "Suspicious NLSQ convergence. Using safer priors decoupled from Hessian."
                )

                for param_name in self.parameters.keys():
                    value = self.parameters.get_value(param_name)
                    bounds = self.parameters.get(param_name).bounds
                    lower, upper = bounds

                    # Use 20% of bounds range as std
                    std = 0.2 * (upper - lower)

                    priors[param_name] = {"mean": value, "std": std}

        else:  # Good convergence
            # Use NLSQ estimates and covariance for prior construction
            diagnostics = self._extract_nlsq_diagnostics(self._nlsq_result)

            for param_name in self.parameters.keys():
                value = self.parameters.get_value(param_name)

                # Get uncertainty from Hessian if available
                if param_name in diagnostics["param_uncertainties"]:
                    std = diagnostics["param_uncertainties"][param_name]
                    # Cap minimum std to avoid delta-like distributions
                    min_std = 0.01 * abs(value) if abs(value) > 1e-12 else 1e-6
                    std = max(std, min_std)
                else:
                    # Fallback: use 5% of parameter value or 5% of bounds
                    bounds = self.parameters.get(param_name).bounds
                    lower, upper = bounds
                    std = max(0.05 * abs(value), 0.05 * (upper - lower))

                priors[param_name] = {"mean": value, "std": std}

        return priors

    def get_relaxation_spectrum(self) -> dict:
        """Get discrete relaxation spectrum (E_i, τ_i).

        Returns:
            Dictionary with 'E_inf', 'E_i', 'tau_i'
        """
        symbol = "E" if self._modulus_type == "tensile" else "G"

        E_inf = self.parameters.get_value(f"{symbol}_inf")
        E_i = np.array(
            [self.parameters.get_value(f"{symbol}_{i+1}") for i in range(self._n_modes)]
        )
        tau_i = np.array(
            [self.parameters.get_value(f"tau_{i+1}") for i in range(self._n_modes)]
        )

        return {f"{symbol}_inf": E_inf, f"{symbol}_i": E_i, "tau_i": tau_i}

    def get_element_minimization_diagnostics(self) -> dict | None:
        """Get element minimization diagnostics.

        Returns:
            Dictionary with .n_initial., .r2., .n_modes., .n_optimal., .optimization_factor. or None if not run
        """
        return self._element_minimization_diagnostics

    def model_function(self, X, params):
        """Model function for Bayesian inference with NumPyro NUTS.

        This method is required by BayesianMixin for NumPyro NUTS sampling.
        It computes GMM predictions given input X and a parameter array.

        Args:
            X: Independent variable (time or frequency)
            params: Array of parameter values [E_inf, E_1, ..., E_N, tau_1, ..., tau_N]
                    Length: 1 + 2*n_modes

        Returns:
            Model predictions as JAX array

        Note:
            Uses self._test_mode (set during fit()) to route to appropriate prediction method.
            For oscillation mode, returns complex modulus [G', G"] with shape (M, 2).
        """
        # Extract parameters from array
        E_inf = params[0]
        E_i = params[1 : 1 + self._n_modes]
        tau_i = params[1 + self._n_modes :]

        # Use stored test mode from last fit
        test_mode = getattr(self, "_test_mode", "relaxation")

        # Route to appropriate prediction method
        if test_mode == "relaxation":
            return self._predict_relaxation_jit(jnp.asarray(X), E_inf, E_i, tau_i)
        elif test_mode == "oscillation":
            # Return complex modulus as (M, 2) for Bayesian likelihood
            E_star = self._predict_oscillation_jit(jnp.asarray(X), E_inf, E_i, tau_i)
            # E_star is now (M, 2) from the updated _predict_oscillation_jit
            return E_star
        elif test_mode == "creep":
            return self._predict_creep_jit(
                jnp.asarray(X), E_inf, E_i, tau_i, sigma_0=1.0
            )
        else:
            raise ValueError(f"Unsupported test mode: {test_mode}")
