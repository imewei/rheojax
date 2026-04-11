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

References
----------
Budrikis & Zapperi (2013) Universality and Localization in Classical and Quantum EPM.
Phys. Rev. E 88, 062403.
Eq. (1): tensorial Eshelby propagator G_ij(q) for 2D plane-strain redistribution.
Eq. (3): von Mises yield criterion J_2 = (1/2)*s_ij*s_ij >= sigma_c^2/3.
Eq. (5): Prandtl-Reuss plastic flow depsilon_ij = d_lambda * s_ij / (2*sigma_c).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

from rheojax.core.jax_config import safe_import_jax

if TYPE_CHECKING:
    import jax

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
    lambda_ps = 2 * mu * nu / jnp.maximum(1 - 2 * nu, 1e-10)

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

    # G_xxxy: Normal-shear coupling (σ_xx ↔ ε_xy)
    # From Eshelby tensor G_ijkl with i=x, j=x, k=x, l=y:
    #   G_xxxy ∝ C_xxkl * qk * ql / |q|² → qx_hat² * qy_hat
    # (Budrikis & Zapperi 2013, Eq. 1; Picard et al. 2004)
    G_xxxy = prefactor * coupling_factor * qx_hat * qx_hat * qy_hat

    # G_yyxy: Normal-shear coupling (σ_yy ↔ ε_xy)
    # From Eshelby tensor G_ijkl with i=y, j=y, k=x, l=y:
    #   G_yyxy ∝ C_yykl * qk * ql / |q|² → qy_hat² * qx_hat
    G_yyxy = prefactor * coupling_factor * qy_hat * qy_hat * qx_hat

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
        (diff_xx_yy**2 + diff_yy_zz**2 + diff_zz_xx**2 + 6 * sigma_xy**2) / 2.0 + 1e-30
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
        + 1e-30
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


@partial(jax.jit, static_argnames=("fluidity_form",))
def compute_plastic_strain_rate(
    stress_tensor: jax.Array,
    sigma_eff: jax.Array,
    tau_pl_shear: float,
    tau_pl_normal: float,
    yield_mask: jax.Array,
    sigma_c_mean: float = 1.0,
    n_fluid: float = 1.0,
    nu: float = 0.5,
    fluidity_form: str = "overstress",
) -> jax.Array:
    r"""Compute component-wise plastic strain rate using Prandtl-Reuss flow rule.

    Three constitutive laws are available via ``fluidity_form``:

    1. **"linear"** (Bingham, scalar-consistent):
       g_eff = σ_eff
       ε̇ᵖ_ij = (σ'_ij / σ_eff) · σ_eff · (1 / τ_pl_ij) · yield_mask
             = σ'_ij · (1 / τ_pl_ij) · yield_mask
       High-rate asymptote: σ_ij ∝ γ̇ · τ_pl (pure Bingham, no additive yield).

    2. **"power"** (power-law fluidity, soft-glassy):
       g_eff = (σ_eff / σ_c_mean)^n_fluid · σ_c_mean
       ε̇ᵖ_ij = (σ'_ij / σ_eff) · g_eff · (1 / τ_pl_ij) · yield_mask
       High-rate asymptote: σ_ij ∝ γ̇^(1/n_fluid) (shear-thinning, no additive yield).

    3. **"overstress"** (Herschel-Bulkley, default):
       g_eff = (σ_eff - σ_c_mean)_+^n_fluid · σ_c_mean^(1 - n_fluid)
       ε̇ᵖ_ij = (σ'_ij / σ_eff) · g_eff · (1 / τ_pl_ij) · yield_mask
       High-rate asymptote: σ_ij = σ_c_mean/√3 + constant · γ̇^(1/n_fluid) for pure shear.
       This is the full HB form σ = σ_y + K·γ̇^n_HB with σ_y = σ_c_mean/√3 (von Mises
       pure-shear plateau) and n_HB = 1/n_fluid. **Recommended default** for HB-like
       yield-stress fluids (emulsions, gels, foams, pastes).

    The Prandtl-Reuss flow direction ``σ'_ij / σ_eff`` is unchanged across forms — only
    the magnitude factor ``g_eff`` changes. The shear and normal channels use
    ``tau_pl_shear`` and ``tau_pl_normal`` respectively.

    Args:
        stress_tensor: Stress components [σ_xx, σ_yy, σ_xy], shape (..., 3).
        sigma_eff: Effective stress (from yield criterion), shape (...,).
        tau_pl_shear: Plastic relaxation time for shear.
        tau_pl_normal: Plastic relaxation time for normal stresses.
        yield_mask: Binary or smooth mask (0=elastic, 1=yielding), shape (...,).
        sigma_c_mean: Mean yield threshold — used as the stress scale for the power-law
            and overstress forms. Default 1.0. Ignored for the "linear" form.
        n_fluid: Power-law / HB exponent. Default 1.0. The implied HB flow exponent is
            n_HB = 1/n_fluid. Ignored for the "linear" form.
        nu: Poisson's ratio for plane-strain σ_zz = ν(σ_xx + σ_yy). Default 0.5
            (incompressible).
        fluidity_form: One of "linear", "power", or "overstress". Default "overstress".

    Returns:
        Plastic strain rate [ε̇ᵖ_xx, ε̇ᵖ_yy, ε̇ᵖ_xy], shape (..., 3).
    """
    sigma_xx = stress_tensor[..., 0]
    sigma_yy = stress_tensor[..., 1]
    sigma_xy = stress_tensor[..., 2]

    # Plane-strain: sigma_zz = nu * (sigma_xx + sigma_yy)
    sigma_zz = nu * (sigma_xx + sigma_yy)
    mean_stress = (sigma_xx + sigma_yy + sigma_zz) / 3.0
    dev_xx = sigma_xx - mean_stress
    dev_yy = sigma_yy - mean_stress

    # Avoid division by zero when sigma_eff = 0
    safe_sigma_eff = jnp.where(sigma_eff > 1e-12, sigma_eff, 1.0)

    # Compute the form-specific magnitude factor g_eff. The Prandtl-Reuss direction
    # (sigma'_ij / sigma_eff) is multiplied by g_eff to get the plastic-flow magnitude.
    if fluidity_form == "linear":
        # Bingham: plastic rate ∝ sigma'_ij directly.
        # (direction * g_eff) = (sigma'_ij / sigma_eff) * sigma_eff = sigma'_ij.
        g_eff = sigma_eff
    elif fluidity_form == "power":
        # Power-law fluidity: plastic rate ∝ (sigma_eff / sigma_c)^n_fluid.
        inv_scm = 1.0 / jnp.maximum(sigma_c_mean, 1e-8)
        sigma_eff_safe = jnp.maximum(sigma_eff, 1e-8)
        g_eff = (sigma_eff_safe * inv_scm) ** n_fluid * sigma_c_mean
    elif fluidity_form == "overstress":
        # Herschel-Bulkley: only sigma_eff ABOVE threshold drives plastic flow.
        # Matches scalar kernel's overstress branch; eps softening keeps the derivative
        # well-behaved at the yield surface for gradient-based fitting.
        eps = 1e-6
        overstress = jnp.maximum(sigma_eff - sigma_c_mean, eps)
        g_eff = overstress ** n_fluid * (sigma_c_mean ** (1.0 - n_fluid))
    else:
        raise ValueError(
            f"Unknown fluidity_form={fluidity_form!r}; "
            "must be 'linear', 'power', or 'overstress'."
        )

    # Component rates: direction * g_eff * (1 / tau_pl_ij) * yield_mask
    eps_dot_p_xx = (dev_xx / safe_sigma_eff) * g_eff * (1.0 / tau_pl_normal) * yield_mask
    eps_dot_p_yy = (dev_yy / safe_sigma_eff) * g_eff * (1.0 / tau_pl_normal) * yield_mask
    eps_dot_p_xy = (sigma_xy / safe_sigma_eff) * g_eff * (1.0 / tau_pl_shear) * yield_mask

    # Zero out plastic flow when there is no stress at all (sigma_eff ≈ 0)
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


@partial(jax.jit, static_argnames=("smooth", "yield_criterion", "fluidity_form"))
def tensorial_epm_step(
    stress: jax.Array,
    thresholds: jax.Array,
    strain_rate: float,
    dt: float,
    propagator: jax.Array,
    params: dict[str, float],
    smooth: bool = False,
    yield_criterion: str = "von_mises",
    fluidity_form: str = "overstress",
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
            - sigma_c_mean: Mean yield threshold (used by power/overstress forms)
            - n_fluid: Power-law / HB exponent (used by power/overstress forms)
            - smoothing_width: Width for smooth yielding (if smooth=True)
            - hill_H, hill_N: Hill anisotropy parameters (if yield_criterion="hill")
        smooth: Use smooth (tanh) yielding instead of hard (step) yielding.
        yield_criterion: "von_mises" or "hill".
        fluidity_form: Constitutive law for the plastic strain rate —
            "linear", "power", or "overstress" (default). See
            ``compute_plastic_strain_rate`` for the mathematical forms.

    Returns:
        Updated stress tensor field, shape (3, L, L).
    """
    mu = params.get("mu", 1.0)
    nu = params.get("nu", 0.3)
    tau_pl_shear = params.get("tau_pl_shear", 1.0)
    tau_pl_normal = params.get("tau_pl_normal", 1.0)
    sigma_c_mean = params.get("sigma_c_mean", 1.0)
    n_fluid = params.get("n_fluid", 1.0)

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

    # 3. Compute plastic strain rate (form-dispatched: linear/power/overstress)
    eps_dot_p = compute_plastic_strain_rate(
        stress_reshaped,
        sigma_eff,
        tau_pl_shear,
        tau_pl_normal,
        yield_mask,
        sigma_c_mean=sigma_c_mean,
        n_fluid=n_fluid,
        nu=nu,
        fluidity_form=fluidity_form,
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
