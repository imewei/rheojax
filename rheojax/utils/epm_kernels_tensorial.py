"""Tensorial kernels for Elasto-Plastic Models (EPM).

This module implements the full tensorial (3-component) stress formulation for EPM
simulations. It tracks the stress tensor [σ_xx, σ_yy, σ_xy] in 2D plane strain,
enabling prediction of normal stress differences (N₁, N₂), anisotropic flow behavior,
and kinematic hardening.

Key Components:
- Tensorial Eshelby propagator G_ij(q) for elastic stress redistribution
- Yield criteria: von Mises (isotropic) and Hill (anisotropic)
- Component-wise Prandtl-Reuss flow rule for plastic strain evolution
- Full tensorial EPM time-stepping kernel
"""

from collections.abc import Callable
from functools import partial

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


@partial(jax.jit, static_argnames=("L",))
def make_tensorial_propagator_q(L: int, nu: float, mu: float = 1.0) -> jax.Array:
    """Create the tensorial Eshelby propagator in Fourier space for plane strain.

    The propagator G_ij(q) couples stress component i to plastic strain component j.
    For 2D plane strain with σ_zz = ν(σ_xx + σ_yy), we track [σ_xx, σ_yy, σ_xy].

    The Eshelby tensor for a circular inclusion in plane strain gives:
    G_ij(q) = C_ijkl * q_k * q_l / |q|²

    where C_ijkl is the elastic stiffness tensor.

    Args:
        L: Lattice size (assumes square lattice L x L).
        nu: Poisson's ratio (plane strain constraint).
        mu: Shear modulus.

    Returns:
        4D array of shape (3, 3, L, L // 2 + 1) representing the propagator
        in Fourier space using rfft2 convention.
        Components are ordered as [xx, yy, xy] for both stress and strain.
    """
    # Create wave vectors
    qx = jnp.fft.fftfreq(L) * 2 * jnp.pi
    qy = jnp.fft.rfftfreq(L) * 2 * jnp.pi

    # Meshgrid with 'ij' indexing for matrix coordinates
    QX, QY = jnp.meshgrid(qx, qy, indexing="ij")
    Q2 = QX**2 + QY**2

    # Handle q=0 singularity
    valid_mask = Q2 > 0
    safe_Q2 = jnp.where(valid_mask, Q2, 1.0)

    # Normalized wave vector components
    qx_hat = QX / jnp.sqrt(safe_Q2)
    qy_hat = QY / jnp.sqrt(safe_Q2)

    # Plane strain elastic constants
    # For plane strain: E' = E/(1-ν²), ν' = ν/(1-ν)
    # Lame parameters for plane strain
    lambda_ps = 2 * mu * nu / (1 - 2 * nu)

    # Initialize propagator tensor (3, 3, L, L//2+1)
    propagator = jnp.zeros((3, 3, L, L // 2 + 1))

    # Compute Eshelby tensor components
    # For plane strain, the Eshelby propagator couples stress to strain
    # G_ij = -(2μ) * [δ_ij + (λ+μ)/(λ+2μ) * q̂_i * q̂_j]

    # Common factor
    prefactor = -2.0 * mu
    coupling_factor = (lambda_ps + mu) / (lambda_ps + 2 * mu)

    # Component mapping: 0=xx, 1=yy, 2=xy

    # G_xxxx: Normal-normal coupling
    G_xxxx = prefactor * (1.0 + coupling_factor * qx_hat**2)

    # G_xxyy: Cross normal coupling
    G_xxyy = prefactor * coupling_factor * qx_hat * qy_hat

    # G_yyxx: Symmetric to G_xxyy
    G_yyxx = G_xxyy

    # G_yyyy: Normal-normal coupling
    G_yyyy = prefactor * (1.0 + coupling_factor * qy_hat**2)

    # G_xyxy: Shear-shear coupling (quadrupolar)
    # For shear: 2 * (qx * qy)² / q⁴ pattern
    G_xyxy = -4.0 * mu * (QX**2 * QY**2) / (safe_Q2**2)

    # G_xxxy: Normal-shear coupling
    G_xxxy = prefactor * coupling_factor * qx_hat * qy_hat

    # G_yyxy: Normal-shear coupling
    G_yyxy = prefactor * coupling_factor * qy_hat * qx_hat

    # Assemble the propagator (ensuring symmetry)
    # Apply valid_mask to zero out invalid entries
    propagator = propagator.at[0, 0].set(jnp.where(valid_mask, G_xxxx, 0.0))
    propagator = propagator.at[0, 1].set(jnp.where(valid_mask, G_xxyy, 0.0))
    propagator = propagator.at[1, 0].set(jnp.where(valid_mask, G_yyxx, 0.0))
    propagator = propagator.at[1, 1].set(jnp.where(valid_mask, G_yyyy, 0.0))
    propagator = propagator.at[2, 2].set(jnp.where(valid_mask, G_xyxy, 0.0))
    propagator = propagator.at[0, 2].set(jnp.where(valid_mask, G_xxxy, 0.0))
    propagator = propagator.at[2, 0].set(jnp.where(valid_mask, G_xxxy, 0.0))
    propagator = propagator.at[1, 2].set(jnp.where(valid_mask, G_yyxy, 0.0))
    propagator = propagator.at[2, 1].set(jnp.where(valid_mask, G_yyxy, 0.0))

    # Explicitly enforce zero at q=0 for all components
    propagator = propagator.at[:, :, 0, 0].set(0.0)

    return propagator


@jax.jit
def compute_von_mises_stress(stress_tensor: jax.Array, nu: float) -> jax.Array:
    """Compute von Mises effective stress for plane strain.

    For plane strain: σ_zz = ν(σ_xx + σ_yy)

    σ_eff = sqrt[(σ_xx - σ_yy)² + (σ_yy - σ_zz)² + (σ_zz - σ_xx)² + 6σ_xy²] / sqrt(2)

    Args:
        stress_tensor: Array of shape (3,) or (..., 3) with [σ_xx, σ_yy, σ_xy].
        nu: Poisson's ratio for plane strain constraint.

    Returns:
        Von Mises effective stress (scalar or array matching input batch shape).
    """
    sigma_xx = stress_tensor[..., 0]
    sigma_yy = stress_tensor[..., 1]
    sigma_xy = stress_tensor[..., 2]

    # Plane strain constraint
    sigma_zz = nu * (sigma_xx + sigma_yy)

    # von Mises formula
    diff_xx_yy = sigma_xx - sigma_yy
    diff_yy_zz = sigma_yy - sigma_zz
    diff_zz_xx = sigma_zz - sigma_xx

    sigma_eff = jnp.sqrt(
        (diff_xx_yy**2 + diff_yy_zz**2 + diff_zz_xx**2 + 6 * sigma_xy**2) / 2.0
    )

    return sigma_eff


@jax.jit
def compute_hill_stress(
    stress_tensor: jax.Array, hill_H: float, hill_N: float, nu: float = 0.3
) -> jax.Array:
    """Compute Hill anisotropic yield stress for plane strain.

    For plane strain with σ_zz = ν(σ_xx + σ_yy), the Hill criterion is:

    σ_eff² = H[(σ_xx - σ_yy)² + (σ_yy - σ_zz)² + (σ_zz - σ_xx)²] + 2N·σ_xy²

    This reduces to von Mises when H=1/3, N=1.5.

    For the simplified 2D form often used:
    σ_eff = sqrt[H(σ_xx - σ_yy)² + 2N·σ_xy²]

    This matches von Mises pure shear when H=0.5, N=1.5.

    Args:
        stress_tensor: Array of shape (3,) or (..., 3) with [σ_xx, σ_yy, σ_xy].
        hill_H: Anisotropy parameter H.
        hill_N: Anisotropy parameter N.
        nu: Poisson's ratio for plane strain (used for full formulation).

    Returns:
        Hill effective stress (scalar or array matching input batch shape).
    """
    sigma_xx = stress_tensor[..., 0]
    sigma_yy = stress_tensor[..., 1]
    sigma_xy = stress_tensor[..., 2]

    # Plane strain constraint
    sigma_zz = nu * (sigma_xx + sigma_yy)

    # Full Hill formulation for plane strain
    diff_xx_yy = sigma_xx - sigma_yy
    diff_yy_zz = sigma_yy - sigma_zz
    diff_zz_xx = sigma_zz - sigma_xx

    sigma_eff = jnp.sqrt(
        hill_H * (diff_xx_yy**2 + diff_yy_zz**2 + diff_zz_xx**2)
        + 2 * hill_N * sigma_xy**2
    )

    return sigma_eff


def get_yield_criterion(name: str) -> Callable:
    """Factory function to get yield criterion function by name.

    Args:
        name: Yield criterion name ("von_mises" or "hill").

    Returns:
        Callable yield criterion function.

    Raises:
        ValueError: If criterion name is unknown.
    """
    criteria = {
        "von_mises": compute_von_mises_stress,
        "hill": compute_hill_stress,
    }

    if name not in criteria:
        raise ValueError(
            f"Unknown yield criterion: {name}. " f"Available: {list(criteria.keys())}"
        )

    return criteria[name]


@jax.jit
def compute_plastic_strain_rate(
    stress_tensor: jax.Array,
    sigma_eff: jax.Array,
    tau_pl_shear: float,
    tau_pl_normal: float,
    yield_mask: jax.Array,
) -> jax.Array:
    """Compute component-wise plastic strain rate using Prandtl-Reuss flow rule.

    Flow rule:
    - Shear: ε̇ᵖ_xy = (σ_xy / σ_eff) · (1 / tau_pl_shear) · yield_mask
    - Normal: ε̇ᵖ_xx = (σ'_xx / σ_eff) · (1 / tau_pl_normal) · yield_mask
              ε̇ᵖ_yy = (σ'_yy / σ_eff) · (1 / tau_pl_normal) · yield_mask

    where σ'_ii = σ_ii - (σ_xx + σ_yy)/2 is the deviatoric stress.

    Args:
        stress_tensor: Stress components [σ_xx, σ_yy, σ_xy], shape (..., 3).
        sigma_eff: Effective stress (from yield criterion), shape (...,).
        tau_pl_shear: Plastic relaxation time for shear.
        tau_pl_normal: Plastic relaxation time for normal stresses.
        yield_mask: Binary or smooth mask (0=elastic, 1=yielding), shape (...,).

    Returns:
        Plastic strain rate [ε̇ᵖ_xx, ε̇ᵖ_yy, ε̇ᵖ_xy], shape (..., 3).
    """
    sigma_xx = stress_tensor[..., 0]
    sigma_yy = stress_tensor[..., 1]
    sigma_xy = stress_tensor[..., 2]

    # Deviatoric normal stresses (incompressible plastic flow)
    mean_stress = (sigma_xx + sigma_yy) / 2.0
    dev_xx = sigma_xx - mean_stress
    dev_yy = sigma_yy - mean_stress

    # Avoid division by zero when sigma_eff = 0
    safe_sigma_eff = jnp.where(sigma_eff > 1e-12, sigma_eff, 1.0)

    # Normal components
    eps_dot_p_xx = (dev_xx / safe_sigma_eff) * (1.0 / tau_pl_normal) * yield_mask
    eps_dot_p_yy = (dev_yy / safe_sigma_eff) * (1.0 / tau_pl_normal) * yield_mask

    # Shear component
    eps_dot_p_xy = (sigma_xy / safe_sigma_eff) * (1.0 / tau_pl_shear) * yield_mask

    # Zero out plastic flow when not yielding (sigma_eff ≈ 0)
    no_stress_mask = sigma_eff > 1e-12
    eps_dot_p_xx = jnp.where(no_stress_mask, eps_dot_p_xx, 0.0)
    eps_dot_p_yy = jnp.where(no_stress_mask, eps_dot_p_yy, 0.0)
    eps_dot_p_xy = jnp.where(no_stress_mask, eps_dot_p_xy, 0.0)

    # Stack components
    eps_dot_p = jnp.stack([eps_dot_p_xx, eps_dot_p_yy, eps_dot_p_xy], axis=-1)

    return eps_dot_p


@jax.jit
def apply_tensorial_propagator(
    propagator: jax.Array, eps_dot_p: jax.Array
) -> jax.Array:
    """Apply the tensorial propagator to compute elastic stress redistribution.

    Performs: σ̇_i = G_ij * ε̇ᵖ_j (convolution in Fourier space).

    Args:
        propagator: Tensorial propagator G_ij(q), shape (3, 3, L, L//2+1).
        eps_dot_p: Plastic strain rate field [ε̇ᵖ_xx, ε̇ᵖ_yy, ε̇ᵖ_xy], shape (3, L, L).

    Returns:
        Elastic stress redistribution rate [σ̇_xx, σ̇_yy, σ̇_xy], shape (3, L, L).
    """
    L = eps_dot_p.shape[1]

    # Transform plastic strain rates to Fourier space
    eps_dot_p_q = jnp.fft.rfft2(eps_dot_p, axes=(1, 2))  # Shape: (3, L, L//2+1)

    # Apply propagator: σ̇_i(q) = G_ij(q) * ε̇ᵖ_j(q)
    # Using einsum for tensor contraction over component index j
    stress_dot_q = jnp.einsum("ijkl,jkl->ikl", propagator, eps_dot_p_q)

    # Transform back to real space
    stress_dot = jnp.fft.irfft2(stress_dot_q, s=(L, L), axes=(1, 2))

    return stress_dot


@partial(jax.jit, static_argnames=("smooth", "yield_criterion"))
def tensorial_epm_step(
    stress: jax.Array,
    thresholds: jax.Array,
    strain_rate: float,
    dt: float,
    propagator: jax.Array,
    params: dict[str, float],
    smooth: bool = False,
    yield_criterion: str = "von_mises",
) -> jax.Array:
    """Perform one full tensorial EPM time step.

    Dynamics:
    σ̇_ij = 2μ ε̇_ij - 2μ ε̇ᵖ_ij + G_ij * ε̇ᵖ_j

    Args:
        stress: Current stress tensor field [σ_xx, σ_yy, σ_xy], shape (3, L, L).
        thresholds: Local yield thresholds, shape (L, L).
        strain_rate: Imposed macroscopic shear rate γ̇.
        dt: Time step size.
        propagator: Precomputed tensorial propagator, shape (3, 3, L, L//2+1).
        params: Dictionary with keys:
            - mu: Shear modulus
            - nu: Poisson's ratio
            - tau_pl_shear: Plastic relaxation time for shear
            - tau_pl_normal: Plastic relaxation time for normal stresses
            - smoothing_width: Width for smooth yielding (if smooth=True)
            - hill_H, hill_N: Hill anisotropy parameters (if yield_criterion="hill")
        smooth: Use smooth (tanh) yielding instead of hard (step) yielding.
        yield_criterion: "von_mises" or "hill".

    Returns:
        Updated stress tensor field, shape (3, L, L).
    """
    mu = params.get("mu", 1.0)
    nu = params.get("nu", 0.3)
    tau_pl_shear = params.get("tau_pl_shear", 1.0)
    tau_pl_normal = params.get("tau_pl_normal", 1.0)

    # 1. Compute effective stress using selected criterion
    if yield_criterion == "von_mises":
        # stress has shape (3, L, L), need (..., 3) for compute_von_mises_stress
        stress_reshaped = jnp.moveaxis(stress, 0, -1)  # (L, L, 3)
        sigma_eff = compute_von_mises_stress(stress_reshaped, nu)  # (L, L)
    elif yield_criterion == "hill":
        stress_reshaped = jnp.moveaxis(stress, 0, -1)
        hill_H = params.get("hill_H", 0.5)
        hill_N = params.get("hill_N", 1.5)
        sigma_eff = compute_hill_stress(stress_reshaped, hill_H, hill_N, nu)
    else:
        raise ValueError(f"Unknown yield criterion: {yield_criterion}")

    # 2. Determine yielding mask
    if smooth:
        # Smooth yielding with tanh activation
        smoothing_width = params.get("smoothing_width", 0.1)
        yield_mask = 0.5 * (1.0 + jnp.tanh((sigma_eff - thresholds) / smoothing_width))
    else:
        # Hard yielding (step function)
        yield_mask = (sigma_eff > thresholds).astype(stress.dtype)

    # 3. Compute plastic strain rate
    eps_dot_p = compute_plastic_strain_rate(
        stress_reshaped, sigma_eff, tau_pl_shear, tau_pl_normal, yield_mask
    )  # (L, L, 3)

    # Reshape back to (3, L, L) for propagator application
    eps_dot_p = jnp.moveaxis(eps_dot_p, -1, 0)

    # 4. Compute stress rates
    # Elastic loading: For simple shear γ̇, only σ_xy increases
    # ε̇_xy = γ̇/2, so σ̇_xy = 2μ ε̇_xy = μ γ̇
    loading_rate = jnp.zeros_like(stress)
    loading_rate = loading_rate.at[2].set(mu * strain_rate)  # Only shear component

    # Plastic relaxation: -2μ ε̇ᵖ_ij
    relaxation_rate = -2.0 * mu * eps_dot_p

    # Elastic redistribution: G_ij * ε̇ᵖ_j
    redistribution_rate = apply_tensorial_propagator(propagator, eps_dot_p)

    # Total stress rate
    total_stress_rate = loading_rate + relaxation_rate + redistribution_rate

    # 5. Update stress (Euler integration)
    new_stress = stress + total_stress_rate * dt

    return new_stress
