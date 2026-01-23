"""
JAX-compatible Mittag-Leffler function implementations.

This module provides efficient, JAX-compatible implementations of the Mittag-Leffler
function using a hybrid strategy:
1. Taylor series for small arguments (|z| < 8)
2. Asymptotic expansions for large arguments (|z| > 8)
   - Exponential expansion for positive z (Creep mode growth)
   - Inverse power law expansion for negative z (Relaxation mode decay)

This approach avoids the numerical instability of Padé approximations near alpha=beta
and correctly models the exponential growth for positive arguments.

References
----------
- R. Garrappa, Numerical evaluation of two and three parameter Mittag-Leffler
  functions, SIAM Journal of Numerical Analysis, 2015, 53(3), 1350-1369
- Haubold, H. J., Mathai, A. M., & Saxena, R. K. (2011). Mittag-Leffler functions
  and their applications. Journal of applied mathematics, 2011.
"""

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

logger = get_logger(__name__)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
from jax.scipy.special import gamma as jax_gamma


def mittag_leffler_e(z: float | jnp.ndarray, alpha: float) -> float | jnp.ndarray:
    """
    One-parameter Mittag-Leffler function E_α(z).

    E_α(z) = E_{α,1}(z)

    Parameters
    ----------
    z : float or jnp.ndarray
        Argument(s) of the Mittag-Leffler function.
    alpha : float
        Order parameter, must be real and positive (0 < alpha <= 2).

    Returns
    -------
    float or jnp.ndarray
        Value(s) of E_α(z).
    """
    # Validate alpha when not traced (static values only)
    if not isinstance(alpha, (jax.core.Tracer, jnp.ndarray)):
        if not (0 < alpha <= 2):
            logger.error(
                "Invalid alpha parameter for Mittag-Leffler function",
                alpha=alpha,
                valid_range="(0, 2]",
            )
            raise ValueError(f"alpha must satisfy 0 < alpha <= 2, got alpha={alpha}")

    return mittag_leffler_e2(z, alpha, beta=1.0)


@jax.jit
def mittag_leffler_e2(
    z: float | jnp.ndarray, alpha: float, beta: float
) -> float | jnp.ndarray:
    """
    Two-parameter Mittag-Leffler function E_{α,β}(z).

    Uses a hybrid evaluation strategy:
    - |z| <= 8: Taylor Series (Kahan summation)
    - z > 8: Positive Asymptotic Expansion (Exponential growth)
    - z < -8: Negative Asymptotic Expansion (Algebraic decay)
    - Smooth blending at boundaries for gradient stability.

    Parameters
    ----------
    z : float or jnp.ndarray
        Argument(s) of the Mittag-Leffler function.
    alpha : float
        First parameter (0 < alpha <= 2).
    beta : float
        Second parameter.

    Returns
    -------
    float or jnp.ndarray
        Value(s) of E_{α,β}(z).
    """
    # Validate alpha when not traced (static values only)
    # Note: We check against Tracer and Array to allow JIT-compiled calls to pass through
    if not isinstance(alpha, (jax.core.Tracer, jnp.ndarray)):
        if not (0 < alpha <= 2):
            logger.error(
                "Invalid alpha parameter for Mittag-Leffler function",
                alpha=alpha,
                beta=beta,
                valid_range="(0, 2]",
            )
            raise ValueError(f"alpha must satisfy 0 < alpha <= 2, got alpha={alpha}")

    # Convert scalar to array for consistency
    z_arr = jnp.asarray(z)
    is_scalar = z_arr.ndim == 0
    z_arr = jnp.atleast_1d(z_arr)

    # Use float64 for precision
    z_f64 = (
        z_arr.astype(jnp.float64)
        if jnp.isrealobj(z_arr)
        else z_arr.astype(jnp.complex128)
    )

    # Vectorized computation
    result = _mittag_leffler_hybrid(z_f64, alpha, beta)

    # Cast back to original dtype if needed (e.g. if input was float32)
    if jnp.issubdtype(z_arr.dtype, jnp.floating):
        result = result.astype(z_arr.dtype)

    if is_scalar:
        return result[0]
    return result


def _ml_taylor(z, alpha, beta, n_iter=300):
    """
    Taylor series expansion: E_{a,b}(z) = sum_{k=0}^N z^k / Gamma(a*k + b)
    Using Kahan summation for reduced cancellation error.
    """

    def body(k, state):
        sum_val, c_val, z_pow = state

        # Calculate term
        term = z_pow / jax_gamma(alpha * k + beta)

        # Kahan summation step
        y = term - c_val
        t = sum_val + y
        c_new = (t - sum_val) - y
        sum_new = t

        # Update z_power
        z_pow_new = z_pow * z

        return sum_new, c_new, z_pow_new

    init_state = (jnp.zeros_like(z), jnp.zeros_like(z), jnp.ones_like(z))
    total, _, _ = jax.lax.fori_loop(0, n_iter, body, init_state)
    return total


def _ml_asymptotic_pos(z, alpha, beta):
    """
    Asymptotic expansion for large positive z (Creep mode).
    E_{a,b}(z) ~ (1/a) * z^((1-b)/a) * exp(z^(1/a))
    """
    inv_alpha = 1.0 / alpha
    exponent = z**inv_alpha
    # Avoid overflow in z^((1-beta)/alpha) by checking sign
    power_term = z ** ((1.0 - beta) * inv_alpha)
    prefactor = inv_alpha * power_term
    return prefactor * jnp.exp(exponent)


def _safe_rgamma(x):
    """
    Computes 1/Gamma(x) safely, returning 0 at poles (negative integers) with correct gradients.

    Uses reflection formula for x < 0.5:
    1/Gamma(z) = Gamma(1-z) * sin(pi*z) / pi
    """

    # Reflection formula is valid everywhere but numerically better for x < 0.5
    # and handles poles at 0, -1, -2... where sin(pi*z) = 0.
    def _reflection(z):
        return (jax_gamma(1.0 - z) * jnp.sin(jnp.pi * z)) / jnp.pi

    def _standard(z):
        return 1.0 / jax_gamma(z)

    # Use reflection for z < 0.5 to avoid poles in standard gamma
    return jax.lax.cond(x < 0.5, _reflection, _standard, operand=x)


def _ml_asymptotic_neg(z, alpha, beta, n_terms=20):
    """
    Asymptotic expansion for large negative z (Relaxation mode).
    E_{a,b}(z) ~ - sum_{k=1}^N z^(-k) / Gamma(beta - alpha*k)
    """
    inv_z = 1.0 / z

    def body(k, val):
        # k goes from 1 to n_terms
        # Term = z^(-k) / Gamma(beta - alpha*k)
        # Use safe reciprocal gamma to handle poles
        rgamma_val = _safe_rgamma(beta - alpha * k)
        term = (inv_z**k) * rgamma_val
        # Series is - sum(...)
        return val - term

    return jax.lax.fori_loop(1, n_terms + 1, body, jnp.zeros_like(z))


def _sigmoid_blend(x, transition, width=1.0):
    """Smooth sigmoid transition from 0 to 1 around transition point."""
    return jax.nn.sigmoid((x - transition) / width)


def _smooth_blend(val1, val2, z, threshold, width=0.5):
    """
    Smoothly blend between val1 (z < threshold) and val2 (z > threshold).

    Parameters
    ----------
    val1 : scalar
        Value for z < threshold.
    val2 : scalar
        Value for z > threshold.
    z : scalar
        Control variable.
    threshold : float
        Transition point.
    width : float
        Width of the transition region.

    Returns
    -------
    scalar
        Blended value.
    """
    weight = jax.nn.sigmoid((z - threshold) / width)
    return (1.0 - weight) * val1 + weight * val2


def _mittag_leffler_hybrid(z, alpha, beta):
    """Hybrid implementation using vmap + blended regions for smoothness."""

    # Thresholds & Widths
    # Tuned for smoothness and stability with n_iter=300 for Taylor series
    THRESH_POS = 6.0
    WIDTH_POS = 0.5

    # Safe cutoff for positive pure branch
    CUTOFF_POS = 10.0

    # Define the scalar kernel
    def _kernel(z_val, a_val, b_val):
        # Dynamic Negative Threshold based on alpha
        # For small alpha (e.g. 0.01), Taylor explodes quickly for z < -1.0.
        # We need to switch to Asymptotic much earlier.
        # Empirical fit:
        # alpha=0.01 -> thresh ~ -0.97
        # alpha=0.99 -> thresh ~ -7.93
        thresh_neg = -0.9 - 7.1 * a_val
        width_neg = 0.1 + 0.4 * a_val

        # Cutoff for pure negative branch (4 sigma)
        # safe_cutoff = thresh - 4 * width
        cutoff_neg = thresh_neg - 4.0 * width_neg

        def _pure_pos(_):
            return _ml_asymptotic_pos(z_val, a_val, b_val)

        def _pure_neg(_):
            return _ml_asymptotic_neg(z_val, a_val, b_val)

        def _blended_region(_):
            # Taylor series is computed everywhere in the blended region
            # n_iter=300 ensures accuracy up to z=10.0
            # For negative z, the dynamic threshold ensures we don't evaluate
            # deep in the unstable region where Taylor explodes.
            val_taylor = _ml_taylor(z_val, a_val, b_val, n_iter=300)

            # Positive Asymptotic (guarded)
            # Safety floor of 1.0 avoids domain errors
            z_pos_safe = jnp.maximum(z_val, 1.0)
            val_pos = _ml_asymptotic_pos(z_pos_safe, a_val, b_val)

            # Negative Asymptotic (guarded)
            # Use dynamic threshold as ceiling to avoid evaluating asymptotic series
            # in its divergent region (small |z|).
            z_neg_safe = jnp.minimum(z_val, thresh_neg)
            val_neg = _ml_asymptotic_neg(z_neg_safe, a_val, b_val)

            # Blend Neg <-> Taylor (Left transition)
            # z < thresh_neg: Neg dominates
            # z > thresh_neg: Taylor dominates
            res = _smooth_blend(val_neg, val_taylor, z_val, thresh_neg, width_neg)

            # Blend Result <-> Pos (Right transition)
            # z < THRESH_POS: Result (Taylor/Neg) dominates
            # z > THRESH_POS: Pos dominates
            res = _smooth_blend(res, val_pos, z_val, THRESH_POS, WIDTH_POS)

            return res

        # Main branch logic with optimization for far-field
        return jax.lax.cond(
            z_val > CUTOFF_POS,
            _pure_pos,
            lambda _: jax.lax.cond(
                z_val < cutoff_neg, _pure_neg, _blended_region, operand=None
            ),
            operand=None,
        )

    # Prepare for broadcasting
    z_arr = jnp.asarray(z)
    a_arr = jnp.asarray(alpha)
    b_arr = jnp.asarray(beta)

    # Broadcast shapes to handle potentially array-valued alpha/beta (though rare)
    # If alpha/beta are scalars, this is cheap.
    # We want to support: z=(N,), alpha=scalar, beta=scalar
    # Or z=(N,), alpha=(N,), beta=(N,)
    # simple broadcasting:
    z_b, a_b, b_b = jnp.broadcast_arrays(z_arr, a_arr, b_arr)

    # Apply vmap over the broadcasted arrays
    # This maps the scalar kernel over all elements
    res = jax.vmap(_kernel)(z_b, a_b, b_b)

    # DEBUG: Check for NaNs
    # Note: This will trigger synchronization, only for debugging!
    # if jnp.any(jnp.isnan(res)):
    #     jax.debug.print("NaN detected in ML hybrid! z={}, a={}, b={}", z_b, a_b, b_b)

    return res


# Convenience aliases
ml_e = mittag_leffler_e
ml_e2 = mittag_leffler_e2

__all__ = [
    "mittag_leffler_e",
    "mittag_leffler_e2",
    "ml_e",
    "ml_e2",
]
