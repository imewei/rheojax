r"""
JAX-compatible Mittag-Leffler function implementations.

This module provides efficient, JAX-compatible implementations of the Mittag-Leffler
function using a hybrid strategy:

1. Taylor series for small arguments (\|z\| < 8)
2. Asymptotic expansions for large arguments (\|z\| > 8)

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

# ML-CONST: Module-level constant for Taylor iteration indices.
# Hoisted out of _ml_taylor to avoid re-materializing the array on every call
# (which happens once per vmapped element when using vmap over z).
_ML_TAYLOR_K = jnp.arange(300, dtype=jnp.float64)


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
    r"""
    Two-parameter Mittag-Leffler function E_{α,β}(z).

    Uses a hybrid evaluation strategy:

    - \|z\| <= 8: Taylor Series (Kahan summation)
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
    r"""Taylor series: E_{a,b}(z) = \sum_{k=0}^{N} z^k / \Gamma(a k + b).

    For real z, uses vectorized log-space computation via ``gammaln`` to ensure
    clean JAX gradients (the ``fori_loop`` overflow clamp produced inf in the
    unused ``jnp.where`` branch, corrupting the backward pass — see KRN-011).

    For complex z, falls back to iterative Kahan summation with overflow clamp
    (gradient w.r.t. complex z is not required by current use cases).
    """
    if jnp.iscomplexobj(z):
        return _ml_taylor_complex(z, alpha, beta, n_iter)

    # --- Real z: vectorized log-space computation ---
    # ML-CONST: reuse the module-level arange; slice to n_iter if caller passes
    # a value smaller than 300 (rare, but preserves the original API contract).
    k = _ML_TAYLOR_K if n_iter == 300 else jnp.arange(n_iter, dtype=jnp.float64)

    # ML-02: Fuse intermediate allocations.
    # Compute log|z^k| directly without a separate safe_abs_z variable.
    abs_z = jnp.abs(z)
    log_abs_z = jnp.log(jnp.maximum(abs_z, 1e-300))

    # log|term_k| = k*log|z| - gammaln(a*k + b), k=0 → log_zpow=0
    # Fused into a single expression: avoids separate log_zpow and log_gamma arrays.
    log_abs_terms = jnp.where(k == 0, 0.0, k * log_abs_z) - jax.scipy.special.gammaln(
        alpha * k + beta
    )

    # Clamp to avoid exp overflow, then exponentiate once.
    abs_terms = jnp.exp(jnp.minimum(log_abs_terms, 700.0))

    # Zero out when z ≈ 0 and k > 0 (z^k → 0, but log-space gives artefacts)
    abs_terms = jnp.where((k > 0) & (abs_z < 1e-300), 0.0, abs_terms)

    # R11-ML-001: Zero out negligible terms to avoid unnecessary computation
    abs_terms = jnp.where(abs_terms < 1e-30 * jnp.max(abs_terms), 0.0, abs_terms)

    # Sign: z^k = |z|^k for z >= 0,  (-1)^k |z|^k for z < 0.
    # Fused sign computation — no separate neg_sign variable.
    sign = jnp.where(z >= 0, 1.0, jnp.where(k % 2 == 0, 1.0, -1.0))

    return jnp.sum(sign * abs_terms)


def _ml_taylor_complex(z, alpha, beta, n_iter=300):
    """Iterative Taylor series for complex z (Kahan summation + overflow clamp)."""

    def body(k, state):
        sum_val, c_val, z_pow = state

        term = z_pow / jax_gamma(alpha * k + beta)

        # Kahan summation step
        y = term - c_val
        t = sum_val + y
        c_new = (t - sum_val) - y
        sum_new = t

        # Update z_power with overflow clamp (KRN-011)
        z_pow_raw = z_pow * z
        abs_val = jnp.abs(z_pow_raw)
        scale = jnp.where(abs_val > 1e300, 1e300 / jnp.maximum(abs_val, 1e-300), 1.0)
        z_pow_new = z_pow_raw * scale

        return sum_new, c_new, z_pow_new

    init_state = (jnp.zeros_like(z), jnp.zeros_like(z), jnp.ones_like(z))
    total, _, _ = jax.lax.fori_loop(0, n_iter, body, init_state)
    return total


def _ml_asymptotic_pos(z, alpha, beta):
    """
    Asymptotic expansion for large positive z (Creep mode).
    E_{a,b}(z) ~ (1/a) * z^((1-b)/a) * exp(z^(1/a))

    KRN-005: Uses log-space evaluation with overflow cap to prevent inf
    for small alpha (< 0.5) at moderate z values.
    """
    inv_alpha = 1.0 / alpha
    # Compute in log-space to avoid overflow
    log_exponent = inv_alpha * jnp.log(jnp.maximum(z, 1e-30))
    log_power = (1.0 - beta) * inv_alpha * jnp.log(jnp.maximum(z, 1e-30))
    log_prefactor = jnp.log(inv_alpha) + log_power
    # Cap the total log-result at 709 (exp(709) ≈ 8.2e307, near float64 max)
    log_result = log_prefactor + log_exponent
    log_result = jnp.minimum(log_result, 709.0)
    return jnp.exp(log_result)


def _safe_rgamma(x):
    """Compute 1/Gamma(x) safely, returning 0 at poles (negative integers).

    Uses "safe-where" pattern (guarded inputs, no lax.cond) so that JAX
    auto-diff produces finite gradients in BOTH branches even though only
    one branch's value is selected.  Ref: DLMF 5.2 — 1/Γ(z).

    For x < 0.5  (reflection):  1/Γ(z) = sin(πz) · Γ(1−z) / π
        Since z < 0.5 ⟹ 1−z > 0.5, Γ(1−z) has no poles.
    For x ≥ 0.5 (standard):  1/Γ(z) directly, no poles for z > 0.
    """
    is_reflection = x < 0.5

    # --- Standard branch: 1/Gamma(x) for x >= 0.5 ---
    # Guard: when reflection is active, use x=1.0 (safe) to avoid NaN grads
    x_std = jnp.where(is_reflection, 1.0, x)
    x_std = jnp.clip(x_std, 1e-10, 170.0)
    g_std = jax_gamma(x_std)
    val_std = 1.0 / jnp.maximum(g_std, 1e-300)

    # --- Reflection branch: sin(πz) * Gamma(1-z) / π for x < 0.5 ---
    # Guard: when standard is active, use refl_arg=1.0 (safe) to avoid NaN grads
    refl_arg = jnp.where(is_reflection, 1.0 - x, 1.0)
    refl_arg = jnp.clip(refl_arg, 0.5, 170.0)  # 1-x > 0.5 when x < 0.5
    g_refl = jax_gamma(refl_arg)
    sin_val = jnp.sin(jnp.pi * jnp.where(is_reflection, x, 0.0))
    val_refl = sin_val * g_refl / jnp.pi

    return jnp.where(is_reflection, val_refl, val_std)


def _ml_asymptotic_neg(z, alpha, beta, n_terms=20):
    """
    Asymptotic expansion for large negative z (Relaxation mode).
    E_{a,b}(z) ~ - sum_{k=1}^N z^(-k) / Gamma(beta - alpha*k)

    ML-03: Precompute all gamma_args as a vector, then apply vmap(_safe_rgamma)
    once and do a single vectorized dot product.  Eliminates the fori_loop and
    reduces the number of sequential kernel dispatches from n_terms to 1.
    """
    inv_z = 1.0 / z
    # k = 1 .. n_terms as a static vector (shape (n_terms,))
    ks = jnp.arange(1, n_terms + 1, dtype=jnp.float64)

    # Precompute all gamma arguments in one shot
    gamma_args = beta - alpha * ks  # shape (n_terms,)

    # Vectorised reciprocal-gamma over all 20 arguments at once
    rgamma_vals = jax.vmap(_safe_rgamma)(gamma_args)  # shape (n_terms,)

    # inv_z^k = exp(k * log(inv_z)) — more numerically stable than pow iteration
    # inv_z is a scalar (negative, so take absolute value first then restore sign)
    # Note: z < 0 and k is integer → sign of inv_z^k follows (-1)^k
    abs_inv_z = jnp.abs(inv_z)
    pow_abs = jnp.exp(ks * jnp.log(jnp.maximum(abs_inv_z, 1e-300)))
    inv_z_pow = jnp.where(ks % 2 == 0, pow_abs, -pow_abs)  # restores sign of inv_z^k

    # Signed terms and sum; series is -sum(...)
    terms = inv_z_pow * rgamma_vals
    return -jnp.sum(terms)


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
    # Tuned for smoothness and stability with n_iter=300 for Taylor series.
    # WIDTH_POS=0.2 ensures sigmoid leakage at z=0 is < 1e-17, preventing
    # the positive asymptotic branch from corrupting small-z accuracy.
    THRESH_POS = 8.0
    WIDTH_POS = 0.2

    # Safe cutoff for positive pure branch (> THRESH_POS + 4*WIDTH_POS)
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
            # For alpha >= 1, the negative asymptotic expansion diverges;
            # fall back to Taylor series which converges for all finite z.
            val = _ml_asymptotic_neg(z_val, a_val, b_val)
            val_taylor = _ml_taylor(z_val, a_val, b_val, n_iter=300)
            return jnp.where(a_val < 1.0, val, val_taylor)

        def _blended_region(_):
            # Taylor series is computed everywhere in the blended region
            # n_iter=300 ensures accuracy up to z=10.0
            # For negative z, the dynamic threshold ensures we don't evaluate
            # deep in the unstable region where Taylor explodes.
            #
            # ML-01: Compute val_taylor exactly once and reuse it for both the
            # alpha>=1 fallback in val_neg and the blend targets below.
            # Previously _ml_taylor was called twice per element in this branch.
            val_taylor = _ml_taylor(z_val, a_val, b_val, n_iter=300)

            # Positive Asymptotic (guarded)
            # Safety floor of 1.0 avoids domain errors
            z_pos_safe = jnp.maximum(z_val, 1.0)
            val_pos = _ml_asymptotic_pos(z_pos_safe, a_val, b_val)

            # Negative Asymptotic (guarded)
            # Use dynamic threshold as ceiling to avoid evaluating asymptotic series
            # in its divergent region (small |z|).
            z_neg_safe = jnp.minimum(z_val, thresh_neg)
            val_neg_raw = _ml_asymptotic_neg(z_neg_safe, a_val, b_val)

            # The negative asymptotic expansion diverges for alpha >= 1
            # (1/Gamma(b-ak) grows factorially while z^{-k} decays exponentially).
            # Reuse the already-computed val_taylor — no second call needed (ML-01).
            val_neg = jnp.where(a_val < 1.0, val_neg_raw, val_taylor)

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
