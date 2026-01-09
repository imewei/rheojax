"""Prony series utilities for Generalized Maxwell Model parameter identification.

This module provides utilities for working with Prony series representations of
viscoelastic relaxation moduli:

    E(t) = E_∞ + Σᵢ₌₁ᴺ Eᵢ exp(-t/τᵢ)

Key capabilities:
- Parameter validation and bounds checking
- Dynamic ParameterSet creation for N modes
- Log-space transforms for wide time-scale ranges
- Element minimization (optimal N selection)
- R² goodness-of-fit metric computation
- Softmax penalty for constrained optimization

References:
    - Park, S. W., & Schapery, R. A. (1999). Methods of interconversion between
      linear viscoelastic material functions. Part I—A numerical method based on
      Prony series. International Journal of Solids and Structures, 36(11), 1653-1675.
    - pyvisco: https://github.com/saintsfan342000/pyvisco
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:  # pragma: no cover
    import jax.numpy as jnp_typing
else:
    jnp_typing = np


type ArrayLike = np.ndarray | jnp_typing.ndarray


def validate_prony_parameters(
    E_inf: float, E_i: ArrayLike, tau_i: ArrayLike
) -> tuple[bool, str]:
    """Validate Prony series parameters for physical consistency.

    Checks:
    - E_inf ≥ 0 (equilibrium modulus non-negative)
    - All Eᵢ > 0 (positive mode strengths)
    - All τᵢ > 0 (positive relaxation times)
    - Same number of Eᵢ and τᵢ elements

    Args:
        E_inf: Equilibrium modulus (Pa)
        E_i: Array of mode strengths (Pa)
        tau_i: Array of relaxation times (s)

    Returns:
        (valid, message): Tuple of validation status and error message

    Example:
        >>> E_inf = 1e3
        >>> E_i = np.array([1e5, 1e4, 1e3])
        >>> tau_i = np.array([1e-2, 1e-1, 1.0])
        >>> valid, msg = validate_prony_parameters(E_inf, E_i, tau_i)
        >>> print(valid)
        True
    """
    logger.debug(
        "Validating Prony series parameters",
        E_inf=E_inf,
        n_modes=len(E_i) if hasattr(E_i, "__len__") else 1,
    )

    # Convert to numpy arrays for consistent handling
    E_i_arr = np.asarray(E_i)
    tau_i_arr = np.asarray(tau_i)

    # Check E_inf non-negative
    if E_inf < 0:
        logger.debug(
            "Prony validation failed: negative E_inf",
            E_inf=E_inf,
        )
        return False, f"E_inf must be non-negative, got {E_inf}"

    # Check array lengths match
    if len(E_i_arr) != len(tau_i_arr):
        logger.debug(
            "Prony validation failed: length mismatch",
            len_E_i=len(E_i_arr),
            len_tau_i=len(tau_i_arr),
        )
        return (
            False,
            f"E_i and tau_i must have same length, got {len(E_i_arr)} and {len(tau_i_arr)}",
        )

    # Check all Eᵢ > 0
    if np.any(E_i_arr <= 0):
        neg_indices = np.where(E_i_arr <= 0)[0]
        logger.debug(
            "Prony validation failed: non-positive mode strengths",
            neg_indices=neg_indices.tolist(),
        )
        return (
            False,
            f"All E_i must be positive, found non-positive at indices {neg_indices.tolist()}",
        )

    # Check all τᵢ > 0
    if np.any(tau_i_arr <= 0):
        neg_indices = np.where(tau_i_arr <= 0)[0]
        logger.debug(
            "Prony validation failed: non-positive relaxation times",
            neg_indices=neg_indices.tolist(),
        )
        return (
            False,
            f"All tau_i must be positive, found non-positive at indices {neg_indices.tolist()}",
        )

    logger.debug("Prony parameters validation passed", n_modes=len(E_i_arr))
    return True, ""


def create_prony_parameter_set(
    n_modes: int, modulus_type: str = "shear"
) -> ParameterSet:
    """Create ParameterSet for N-mode Prony series.

    Dynamically generates parameters:
    - E_inf (or G_inf for shear): Equilibrium modulus
    - E_1...E_N (or G_1...G_N): Mode strengths
    - tau_1...tau_N: Relaxation times

    Args:
        n_modes: Number of Maxwell modes (N ≥ 1)
        modulus_type: 'shear' for G(t) or 'tensile' for E(t)

    Returns:
        ParameterSet with 2N+1 parameters configured for Prony series

    Raises:
        ValueError: If n_modes < 1 or modulus_type invalid

    Example:
        >>> params = create_prony_parameter_set(n_modes=3, modulus_type='shear')
        >>> list(params.keys())
        ['G_inf', 'G_1', 'G_2', 'G_3', 'tau_1', 'tau_2', 'tau_3']
    """
    logger.debug(
        "Creating Prony ParameterSet",
        n_modes=n_modes,
        modulus_type=modulus_type,
    )

    if n_modes < 1:
        logger.error(
            "Invalid n_modes for Prony ParameterSet",
            n_modes=n_modes,
        )
        raise ValueError(f"n_modes must be ≥ 1, got {n_modes}")

    if modulus_type not in ["shear", "tensile"]:
        logger.error(
            "Invalid modulus_type for Prony ParameterSet",
            modulus_type=modulus_type,
            valid_types=["shear", "tensile"],
        )
        raise ValueError(
            f"modulus_type must be 'shear' or 'tensile', got {modulus_type}"
        )

    param_set = ParameterSet()

    # Choose symbol based on modulus type
    symbol = "G" if modulus_type == "shear" else "E"
    units = "Pa"

    # Add equilibrium modulus (can be zero for liquids)
    param_set.add(
        name=f"{symbol}_inf",
        value=1e3,
        bounds=(0.0, 1e9),
        units=units,
        description=f"Equilibrium {modulus_type} modulus",
    )

    # Add mode strengths (must be positive)
    for i in range(1, n_modes + 1):
        param_set.add(
            name=f"{symbol}_{i}",
            value=1e5,
            bounds=(1e-3, 1e9),
            units=units,
            description=f"Mode {i} strength",
        )

    # Add relaxation times (wide range to handle diverse timescales)
    for i in range(1, n_modes + 1):
        param_set.add(
            name=f"tau_{i}",
            value=10.0 ** (i - 1 - n_modes / 2),  # Logarithmic spacing
            bounds=(1e-6, 1e6),
            units="s",
            description=f"Mode {i} relaxation time",
        )

    logger.info(
        "Created Prony ParameterSet",
        n_modes=n_modes,
        modulus_type=modulus_type,
        n_parameters=len(param_set),
    )
    return param_set


def tau_to_log_tau(tau_i: ArrayLike) -> ArrayLike:
    """Transform relaxation times to log-space.

    Useful for optimization over wide time-scale ranges (e.g., 1e-6 to 1e6 s).
    Log-space optimization provides more uniform parameter sensitivity.

    Args:
        tau_i: Array of relaxation times (s)

    Returns:
        log10(tau_i): Log-transformed relaxation times

    Example:
        >>> tau = np.array([1e-3, 1e-1, 1e1, 1e3])
        >>> log_tau = tau_to_log_tau(tau)
        >>> print(log_tau)
        [-3. -1.  1.  3.]
    """
    tau_arr = jnp.asarray(tau_i)
    return jnp.log10(tau_arr)


def log_tau_to_tau(log_tau_i: ArrayLike) -> ArrayLike:
    """Transform log-space relaxation times back to linear space.

    Inverse of tau_to_log_tau().

    Args:
        log_tau_i: Array of log10(tau) values

    Returns:
        tau_i: Relaxation times (s)

    Example:
        >>> log_tau = np.array([-3., -1., 1., 3.])
        >>> tau = log_tau_to_tau(log_tau)
        >>> print(tau)
        [1.e-03 1.e-01 1.e+01 1.e+03]
    """
    log_tau_arr = jnp.asarray(log_tau_i)
    return jnp.power(10.0, log_tau_arr)


def compute_r_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute R² coefficient of determination.

    R² = 1 - SS_res / SS_tot
    where SS_res = Σ(y_true - y_pred)², SS_tot = Σ(y_true - mean(y_true))²

    R² ∈ (-∞, 1], with R²=1 being perfect fit.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² coefficient (1.0 = perfect fit, 0.0 = mean baseline, <0 = worse than mean)

    Example:
        >>> y_true = np.array([1., 2., 3., 4., 5.])
        >>> y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        >>> r2 = compute_r_squared(y_true, y_pred)
        >>> print(f"{r2:.4f}")
        0.9960
    """
    y_true_arr = jnp.asarray(y_true)
    y_pred_arr = jnp.asarray(y_pred)

    # Residual sum of squares
    ss_res = jnp.sum((y_true_arr - y_pred_arr) ** 2)

    # Total sum of squares
    ss_tot = jnp.sum((y_true_arr - jnp.mean(y_true_arr)) ** 2)

    # Handle edge case where all y_true are identical
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    return float(r2)


def iterative_n_reduction(fit_results_dict: dict[int, float]) -> dict[str, ArrayLike]:
    """Track R² vs N for element minimization visualization.

    Args:
        fit_results_dict: Dictionary mapping n_modes → R² value
            Example: {10: 0.998, 9: 0.997, 8: 0.995, ...}

    Returns:
        Dictionary with keys:
        - 'n_modes': Array of N values (sorted ascending)
        - 'r2': Array of R² values corresponding to each N
        - 'r2_min': Minimum R² across all fits
        - 'r2_max': Maximum R² across all fits

    Example:
        >>> results = {10: 0.998, 8: 0.995, 6: 0.990, 4: 0.980, 2: 0.950}
        >>> diagnostics = iterative_n_reduction(results)
        >>> print(diagnostics['n_modes'])
        [ 2  4  6  8 10]
        >>> print(diagnostics['r2'])
        [0.95  0.98  0.99  0.995 0.998]
    """
    if not fit_results_dict:
        raise ValueError("fit_results_dict cannot be empty")

    # Sort by n_modes
    n_values = sorted(fit_results_dict.keys())
    r2_values = [fit_results_dict[n] for n in n_values]

    return {
        "n_modes": np.array(n_values),
        "r2": np.array(r2_values),
        "r2_min": float(np.min(r2_values)),
        "r2_max": float(np.max(r2_values)),
    }


def select_optimal_n(
    r2_values: dict[int, float], optimization_factor: float = 1.5
) -> int:
    """Select optimal number of modes using R² threshold criterion.

    Algorithm:
    1. Find maximum R² across all N: R²_max (best achievable fit)
    2. Compute R² degradation tolerance: ΔR² = (1 - R²_max) × (optimization_factor - 1.0)
    3. Set threshold: R²_threshold = R²_max - ΔR²
    4. Select smallest N where R²_N ≥ R²_threshold

    Interpretation:
    - optimization_factor = 1.0: Require R² ≥ R²_max (maximum parsimony, only accept best)
    - optimization_factor = 1.5: Allow 50% of max degradation (balance quality/parsimony)
    - optimization_factor = 2.0: Allow 100% of max degradation (maximum parsimony)

    For optimization_factor > 1, this allows some degradation from the best fit in exchange
    for fewer parameters. Higher factor = more tolerant of degradation = simpler model.

    Args:
        r2_values: Dictionary mapping n_modes → R² value
        optimization_factor: Parsimony factor (≥ 1.0)
            - 1.0: No degradation allowed (require best R²)
            - 1.5 (default): Allow 50% of max possible degradation
            - 2.0: Allow 100% degradation (maximum simplicity)

    Returns:
        Optimal number of modes (N_opt)

    Raises:
        ValueError: If optimization_factor < 1.0 or r2_values empty

    Example:
        >>> r2 = {5: 0.998, 3: 0.995, 2: 0.980, 1: 0.900}
        >>> # R²_max = 0.998, degradation room = 1 - 0.998 = 0.002
        >>> # factor=1.5: ΔR² = 0.002 × 0.5 = 0.001, threshold = 0.997
        >>> # Smallest N with R² ≥ 0.997: N=3
        >>> n_opt = select_optimal_n(r2, optimization_factor=1.5)
        >>> print(n_opt)
        3
        >>> # factor=1.0: ΔR² = 0, threshold = 0.998, need N=5
        >>> n_opt = select_optimal_n(r2, optimization_factor=1.0)
        >>> print(n_opt)
        5
    """
    logger.debug(
        "Selecting optimal number of modes",
        n_candidates=len(r2_values),
        optimization_factor=optimization_factor,
    )

    if optimization_factor < 1.0:
        logger.error(
            "Invalid optimization_factor",
            optimization_factor=optimization_factor,
        )
        raise ValueError(
            f"optimization_factor must be ≥ 1.0, got {optimization_factor}"
        )

    if not r2_values:
        logger.error("Empty r2_values dictionary")
        raise ValueError("r2_values cannot be empty")

    # Find maximum R² (best fit)
    r2_max = max(r2_values.values())

    # Compute degradation tolerance
    # degradation_room = how much R² can degrade from perfect (1.0 - r2_max)
    # we allow (optimization_factor - 1.0) × degradation_room loss
    degradation_room = 1.0 - r2_max
    allowed_degradation = degradation_room * (optimization_factor - 1.0)

    # Set threshold
    r2_threshold = r2_max - allowed_degradation

    logger.debug(
        "Computed R2 threshold for mode selection",
        r2_max=r2_max,
        r2_threshold=r2_threshold,
        allowed_degradation=allowed_degradation,
    )

    # Find smallest N satisfying threshold
    # Sort by N (ascending) to find minimum N first
    n_sorted = sorted(r2_values.keys())

    for n in n_sorted:
        if r2_values[n] >= r2_threshold:
            logger.info(
                "Selected optimal number of modes",
                n_opt=n,
                r2_at_n_opt=r2_values[n],
                r2_max=r2_max,
                r2_threshold=r2_threshold,
            )
            return n

    # If no N satisfies threshold (shouldn't happen), return smallest N
    n_opt = min(n_sorted)
    logger.info(
        "Selected minimum N (no N satisfied threshold)",
        n_opt=n_opt,
        r2_at_n_opt=r2_values[n_opt],
    )
    return n_opt


def softmax_penalty(E_i: ArrayLike, scale: float = 1.0):
    """Compute softmax penalty for negative moduli in Step 1 fitting.

    This differentiable penalty encourages positive Eᵢ values during
    unconstrained optimization. It approaches zero when all Eᵢ >> 0,
    and increases smoothly for negative values.

    Penalty = scale × Σᵢ log(1 + exp(-Eᵢ/scale))

    Args:
        E_i: Array of mode strengths (Pa)
        scale: Smoothness parameter (default 1.0). Larger values give
            smoother penalty but weaker enforcement.

    Returns:
        Penalty value (≥ 0, differentiable, JAX array or scalar)

    Note:
        Returns JAX array for gradient compatibility. Do not convert
        to Python float() when used in JAX-traced functions.

    Example:
        >>> E_i = np.array([1e5, 1e4, -1e3])  # One negative mode
        >>> penalty = softmax_penalty(E_i, scale=1e3)
        >>> print(f"{penalty:.2f}")
        693.15  # Penalty for negative value
        >>> E_i_pos = np.array([1e5, 1e4, 1e3])  # All positive
        >>> penalty_pos = softmax_penalty(E_i_pos, scale=1e3)
        >>> print(f"{penalty_pos:.2e}")
        3.13e+02  # Small penalty for finite positive values
    """
    E_arr = jnp.asarray(E_i)
    penalty = scale * jnp.sum(jnp.log(1.0 + jnp.exp(-E_arr / scale)))
    # Return JAX array (do not convert to Python float for gradient compatibility)
    return penalty


def warm_start_from_n_modes(
    params_n: ArrayLike, n_target: int, modulus_type: str = "shear"
) -> ArrayLike:
    """Extract warm-start parameters for reduced-mode fit from N-mode solution.

    Used in element minimization to initialize N-1 mode fit from N mode solution.
    Provides intelligent parameter extraction for faster convergence in successive
    NLSQ fits during element search optimization.

    Algorithm:
    1. Extract E_inf, E_i, tau_i from N-mode params
    2. If n_target < N: Truncate to first n_target modes (keep strongest modes)
    3. If n_target > N: Pad with zeros/default values (edge case, typically not used)
    4. If n_target == N: Return params unchanged

    Parameter Layout:
    - params_n format: [E_inf, E_1, E_2, ..., E_N, tau_1, tau_2, ..., tau_N]
    - Total length: 2*N + 1

    Args:
        params_n: Fitted parameters from N-mode optimization
            Shape: (2*N + 1,) where N is current number of modes
        n_target: Target number of modes for next fit (typically N-1)
        modulus_type: 'shear' (G) or 'tensile' (E) - currently not used,
            but kept for API consistency

    Returns:
        Initial parameters for n_target-mode fit
        Shape: (2*n_target + 1,)

    Raises:
        ValueError: If n_target < 1 or params_n has invalid length

    Example:
        >>> # 5-mode fit result
        >>> params_5 = np.array([1e3, 1e6, 5e5, 2e5, 8e4, 3e4,  # E_inf, E_1..E_5
        ...                      1e-2, 1e-1, 1.0, 1e1, 1e2])    # tau_1..tau_5
        >>> # Warm-start for 4-mode fit (truncate weakest mode E_5)
        >>> params_4 = warm_start_from_n_modes(params_5, n_target=4)
        >>> print(params_4.shape)
        (9,)  # 2*4 + 1 = 9 parameters
        >>> # E_inf, E_1..E_4, tau_1..tau_4
        >>> print(params_4)
        [1.e+03 1.e+06 5.e+05 2.e+05 8.e+04 1.e-02 1.e-01 1.e+00 1.e+01]

    Notes:
        - Truncation assumes modes are ordered by importance (strongest first)
        - For GMM fitting, this ordering is typically achieved by sorting by E_i
        - Warm-start can provide 2-5x speedup in element minimization
        - Compilation reuse provides additional speedup when combined with this
    """
    logger.debug(
        "Extracting warm-start parameters",
        n_target=n_target,
        params_length=len(params_n) if hasattr(params_n, "__len__") else 1,
        modulus_type=modulus_type,
    )

    if n_target < 1:
        logger.error("Invalid n_target for warm-start", n_target=n_target)
        raise ValueError(f"n_target must be ≥ 1, got {n_target}")

    params_arr = np.asarray(params_n)

    # Infer current N from params length: 2*N + 1
    if (len(params_arr) - 1) % 2 != 0:
        logger.error(
            "Invalid parameter array format",
            params_length=len(params_arr),
            expected_format="2*N+1",
        )
        raise ValueError(
            f"Invalid params_n length {len(params_arr)}, expected 2*N+1 format"
        )

    n_current = (len(params_arr) - 1) // 2

    # Extract components
    E_inf = params_arr[0]
    E_i = params_arr[1 : 1 + n_current]
    tau_i = params_arr[1 + n_current :]

    # Case 1: Reduce modes (typical for element minimization)
    if n_target < n_current:
        # Truncate to first n_target modes
        E_i_target = E_i[:n_target]
        tau_i_target = tau_i[:n_target]
        logger.debug(
            "Warm-start: reducing modes",
            n_current=n_current,
            n_target=n_target,
            operation="truncate",
        )

    # Case 2: Same number of modes (no-op)
    elif n_target == n_current:
        E_i_target = E_i
        tau_i_target = tau_i
        logger.debug(
            "Warm-start: same number of modes",
            n_current=n_current,
            operation="passthrough",
        )

    # Case 3: Increase modes (edge case, pad with small values)
    else:
        # Pad with small positive values
        n_pad = n_target - n_current
        E_i_target = np.concatenate([E_i, np.full(n_pad, 1e3)])
        tau_i_target = np.concatenate([tau_i, np.logspace(-2, 2, n_pad)])
        logger.debug(
            "Warm-start: increasing modes",
            n_current=n_current,
            n_target=n_target,
            n_pad=n_pad,
            operation="pad",
        )

    # Reconstruct parameter array
    params_target = np.concatenate([[E_inf], E_i_target, tau_i_target])

    logger.debug(
        "Warm-start parameters extracted",
        output_length=len(params_target),
        E_inf=E_inf,
    )
    return params_target
