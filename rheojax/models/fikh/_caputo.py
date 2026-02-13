"""Caputo fractional derivative utilities for FIKH models.

This module provides JAX-compatible implementations of the Caputo fractional
derivative using the L1 scheme, along with history buffer management for
memory-efficient computation.

The Caputo derivative is defined as:
    D^α f(t) = (1/Γ(1-α)) ∫₀ᵗ f'(s)/(t-s)^α ds

For structure evolution in FIKH:
    D^α λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

References:
    - Diethelm, K. (2010). The Analysis of Fractional Differential Equations.
    - Li, C., & Zeng, F. (2015). Numerical Methods for Fractional Calculus.
"""

from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


@partial(jax.jit, static_argnums=(1,))
def compute_gl_weights(alpha: float, n: int) -> jnp.ndarray:
    """Compute Grünwald-Letnikov weights for fractional derivative.

    The GL weights are defined recursively:
        w_0 = 1
        w_k = (1 - (1+α)/k) * w_{k-1}

    These weights sum to zero as n → ∞ for 0 < α < 1.

    Args:
        alpha: Fractional order (0 < α < 1).
        n: Number of weights to compute.

    Returns:
        Array of GL weights [w_0, w_1, ..., w_{n-1}].
    """

    def scan_fn(carry, k):
        w_prev = carry
        # w_k = (1 - (1 + alpha) / k) * w_{k-1}
        w_k = (1.0 - (1.0 + alpha) / k) * w_prev
        return w_k, w_k

    w_0 = jnp.array(1.0)
    # Scan over k = 1, 2, ..., n-1
    _, weights_rest = jax.lax.scan(scan_fn, w_0, jnp.arange(1, n))

    # Prepend w_0
    weights = jnp.concatenate([jnp.array([1.0]), weights_rest])
    return weights


@partial(jax.jit, static_argnums=(1,))
def compute_l1_coefficients(alpha: float, n: int) -> jnp.ndarray:
    """Compute L1 scheme coefficients for Caputo derivative.

    The L1 scheme approximates D^α f(t_n) as:
        D^α f ≈ (1/(Γ(2-α)·dt^α)) Σ_{k=0}^{n-1} b_k·(f_{n-k} - f_{n-k-1})

    where b_k = (k+1)^{1-α} - k^{1-α}

    Args:
        alpha: Fractional order (0 < α < 1).
        n: Number of coefficients to compute.

    Returns:
        Array of L1 coefficients [b_0, b_1, ..., b_{n-1}].
    """
    k = jnp.arange(n)
    one_minus_alpha = 1.0 - alpha
    b_k = jnp.power(k + 1, one_minus_alpha) - jnp.power(k, one_minus_alpha)
    return b_k


@jax.jit
def caputo_derivative_l1(
    f_history: jnp.ndarray,
    dt: float,
    alpha: float,
    b_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute Caputo fractional derivative using L1 scheme.

    D^α f(t_n) ≈ (1/(Γ(2-α)·dt^α)) Σ_{k=0}^{n-1} b_k·(f_{n-k} - f_{n-k-1})

    The history buffer stores values in chronological order:
        f_history = [f_{n-N+1}, f_{n-N+2}, ..., f_{n-1}, f_n]

    Args:
        f_history: History buffer of function values, shape (n_history,) or
            (n_history, state_dim). Most recent value at index -1.
        dt: Time step size.
        alpha: Fractional order (0 < α < 1).
        b_coeffs: Pre-computed L1 coefficients from compute_l1_coefficients.

    Returns:
        Caputo derivative approximation, same shape as f_history[0].
    """
    # Compute differences: f_{n-k} - f_{n-k-1}
    # With history = [f_{n-N+1}, ..., f_{n-1}, f_n], we need:
    # f_n - f_{n-1}, f_{n-1} - f_{n-2}, ..., f_{n-N+2} - f_{n-N+1}
    diffs = f_history[1:] - f_history[:-1]  # Shape: (n_hist-1,) or (n_hist-1, d)

    # Reverse to match L1 indexing: newest first
    diffs_reversed = jnp.flip(diffs, axis=0)

    # Truncate coefficients to match available history
    n_diffs = diffs_reversed.shape[0]
    b_truncated = b_coeffs[:n_diffs]

    # Weighted sum
    if diffs_reversed.ndim == 1:
        weighted_sum = jnp.dot(b_truncated, diffs_reversed)
    else:
        # For multi-dimensional state
        weighted_sum = jnp.einsum("k,k...->...", b_truncated, diffs_reversed)

    # Normalization factor
    gamma_factor = jax.scipy.special.gamma(2.0 - alpha)
    dt_alpha = jnp.power(dt, alpha)
    normalization = 1.0 / (gamma_factor * dt_alpha)

    return normalization * weighted_sum


@partial(jax.jit, static_argnums=(0, 1))
def create_history_buffer(n_history: int, state_dim: int = 1) -> jnp.ndarray:
    """Create an initialized history buffer.

    Args:
        n_history: Number of history points to store.
        state_dim: Dimension of state (default 1 for scalar λ).

    Returns:
        Zero-initialized history buffer of shape (n_history,) if state_dim=1,
        or (n_history, state_dim) otherwise.
    """
    if state_dim == 1:
        return jnp.zeros(n_history)
    return jnp.zeros((n_history, state_dim))


@jax.jit
def update_history_buffer(
    buffer: jnp.ndarray,
    new_value: jnp.ndarray,
) -> jnp.ndarray:
    """Update history buffer with new value using ring buffer pattern.

    Shifts buffer left (dropping oldest) and appends new value at end.
    This maintains chronological order: oldest at index 0, newest at index -1.

    Args:
        buffer: Current history buffer, shape (n_history,) or (n_history, d).
        new_value: New value to append, scalar or shape (d,).

    Returns:
        Updated buffer with new_value at the end.
    """
    # Ensure new_value is a scalar for 1D buffer
    if buffer.ndim == 1:
        # Flatten new_value to scalar if needed
        new_val_scalar = jnp.atleast_1d(new_value).flatten()[0]
        shifted = jnp.concatenate([buffer[1:], jnp.array([new_val_scalar])])
    else:
        # For multi-dimensional buffer
        new_val_1d = jnp.atleast_1d(new_value).flatten()
        shifted = jnp.concatenate([buffer[1:], new_val_1d[None, :]], axis=0)
    return shifted


@jax.jit
def initialize_history_with_value(
    buffer: jnp.ndarray,
    initial_value: jnp.ndarray,
) -> jnp.ndarray:
    """Initialize history buffer with a constant initial value.

    This is useful for starting the simulation with λ(t<0) = λ_0.

    Args:
        buffer: History buffer to initialize.
        initial_value: Value to fill the buffer with.

    Returns:
        Buffer filled with initial_value.
    """
    if buffer.ndim == 1:
        return jnp.full_like(buffer, initial_value)
    return jnp.broadcast_to(initial_value, buffer.shape)


@partial(jax.jit, static_argnums=(2,))
def fractional_derivative_with_short_memory(
    f_current: jnp.ndarray,
    f_history: jnp.ndarray,
    alpha: float,
    dt: float,
    b_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute fractional derivative with short-memory truncation.

    This is the main entry point for computing Caputo derivatives in FIKH.
    It combines the current value with history for the L1 scheme.

    Short-memory principle: For practical computation, we truncate the
    history to N points, introducing O(dt) error but enabling efficient
    JAX scan operations.

    Args:
        f_current: Current function value at t_n.
        f_history: History buffer of previous values [f_{n-N}, ..., f_{n-1}].
        alpha: Fractional order.
        dt: Time step.
        b_coeffs: Pre-computed L1 coefficients.

    Returns:
        Approximation of D^α f(t_n).
    """
    # Append current value to history for derivative computation
    if f_history.ndim == 1:
        full_history = jnp.concatenate([f_history, f_current[None]])
    else:
        full_history = jnp.concatenate([f_history, f_current[None, :]], axis=0)

    return caputo_derivative_l1(full_history, dt, alpha, b_coeffs)


# Convenience function for computing fractional structure evolution
@jax.jit
def fractional_structure_derivative(
    lam: jnp.ndarray,
    lam_history: jnp.ndarray,
    gamma_dot_p_abs: jnp.ndarray,
    tau_thix: float,
    Gamma: float,
    alpha: float,
    dt: float,
    b_coeffs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the RHS for fractional structure evolution.

    The fractional thixotropic evolution equation is:
        D^α λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

    This function returns the RHS, which when integrated gives the
    structure parameter evolution.

    Args:
        lam: Current structure parameter.
        lam_history: History buffer for λ.
        gamma_dot_p_abs: Absolute plastic shear rate.
        tau_thix: Thixotropic time scale.
        Gamma: Breakdown coefficient.
        alpha: Fractional order.
        dt: Time step.
        b_coeffs: Pre-computed L1 coefficients.

    Returns:
        RHS of fractional evolution equation.
    """
    # Build-up term
    build_up = (1.0 - lam) / jnp.maximum(tau_thix, 1e-12)

    # Breakdown term
    break_down = Gamma * lam * gamma_dot_p_abs

    return build_up - break_down
