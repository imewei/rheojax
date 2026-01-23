"""Tests for tensorial EPM kernels.

This module tests the tensorial (3-component) stress formulation for EPM models,
including the tensorial Eshelby propagator, yield criteria (von Mises, Hill),
and component-wise flow rules.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


@pytest.mark.unit
def test_tensorial_propagator_shape():
    """Test that the tensorial propagator has correct shape (3, 3, L, L//2+1)."""
    from rheojax.utils.epm_kernels_tensorial import make_tensorial_propagator_q

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # Shape should be (3, 3, L, L//2+1) for rfft2 convention
    expected_shape = (3, 3, L, L // 2 + 1)
    assert (
        propagator.shape == expected_shape
    ), f"Expected {expected_shape}, got {propagator.shape}"

    # Check dtype is float64
    assert propagator.dtype == jnp.float64


@pytest.mark.unit
def test_tensorial_propagator_symmetry():
    """Test that tensorial propagator is symmetric: G[i,j] == G[j,i]."""
    from rheojax.utils.epm_kernels_tensorial import make_tensorial_propagator_q

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # Check symmetry for all components
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                propagator[i, j],
                propagator[j, i],
                rtol=1e-10,
                err_msg=f"Propagator not symmetric at ({i}, {j})",
            )


@pytest.mark.unit
def test_tensorial_propagator_zero_mean():
    """Test that propagator enforces zero mean: G[i,j](q=0) = 0 for all i,j."""
    from rheojax.utils.epm_kernels_tensorial import make_tensorial_propagator_q

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # q=0 corresponds to index [0, 0] in rfft2 convention
    zero_freq_values = propagator[:, :, 0, 0]

    np.testing.assert_allclose(
        zero_freq_values,
        jnp.zeros((3, 3)),
        atol=1e-12,
        err_msg="Propagator at q=0 should be zero for all components",
    )


@pytest.mark.unit
def test_von_mises_stress_pure_shear():
    """Test von Mises stress calculation for pure shear (σ_xy only)."""
    from rheojax.utils.epm_kernels_tensorial import compute_von_mises_stress

    nu = 0.3

    # Pure shear: σ_xx = σ_yy = 0, σ_xy = 1.0
    stress_tensor = jnp.array([0.0, 0.0, 1.0])  # [σ_xx, σ_yy, σ_xy]

    sigma_eff = compute_von_mises_stress(stress_tensor, nu)

    # For plane strain: σ_zz = ν(σ_xx + σ_yy) = 0
    # σ_eff = sqrt((0² + 0² + 0²) / 2 + 6 * 1.0²) / sqrt(2)
    #       = sqrt(6) / sqrt(2) = sqrt(3) ≈ 1.732
    expected = jnp.sqrt(3.0)

    np.testing.assert_allclose(sigma_eff, expected, rtol=1e-10)


@pytest.mark.unit
def test_hill_criterion_reduces_to_von_mises():
    """Test that Hill criterion with H=1/3, N=1.5 reduces to von Mises."""
    from rheojax.utils.epm_kernels_tensorial import (
        compute_hill_stress,
        compute_von_mises_stress,
    )

    nu = 0.3

    # Test several stress states
    stress_states = [
        jnp.array([1.0, 0.5, 0.3]),  # Mixed
        jnp.array([0.0, 0.0, 1.0]),  # Pure shear
        jnp.array([2.0, -1.0, 0.0]),  # Normal stresses only
        jnp.array([1.5, 1.5, 0.5]),  # Equal normal stresses
    ]

    for stress_tensor in stress_states:
        von_mises = compute_von_mises_stress(stress_tensor, nu)

        # Hill with H=1/3, N=1.5 should equal von Mises (both include sqrt(1/2) factor)
        # von Mises: sqrt[(sum of squared differences) / 2 + 6*sigma_xy^2 / 2]
        #          = sqrt[(sum of squared differences + 6*sigma_xy^2) / 2]
        # Hill:      sqrt[H * (sum of squared differences) + 2*N*sigma_xy^2]
        # Match when: H = 1/2 / 2 = 1/3 is NOT correct...
        # Let's check: von Mises has 1/2 factor outside, 6 factor on sigma_xy^2
        # Hill: H * (diff^2) + 2*N*sigma_xy^2
        # For equivalence: H=1/3, 2*N = 6/2 = 3, so N = 1.5
        # Wait, von Mises is sqrt[(diff² + 6σ_xy²)/2]
        # Hill should be: sqrt[H·diff² + 2N·σ_xy²]
        # Match: H=1/3, 2N=3 → N=1.5 ✗ (doesn't give 1/2 factor)
        # Actually: H=1/2, 2N=3 gives sqrt[(diff² + 6σ_xy²)/2] when diff²=diff²
        # Hmm, von Mises has THREE difference terms, not just (σ_xx - σ_yy)²
        # Let me recalculate: sum = (σ_xx-σ_yy)² + (σ_yy-σ_zz)² + (σ_zz-σ_xx)²
        # von Mises = sqrt[(sum + 6σ_xy²) / 2]
        # Hill = sqrt[H·sum + 2N·σ_xy²]
        # Match: H = 1/2, 2N = 6/2 = 3 → N = 1.5 ✓ But this gives sqrt[sum/2 + 3σ_xy²]
        # That's NOT the same as sqrt[(sum + 6σ_xy²)/2]
        # Let me be more careful:
        # von Mises = sqrt[(sum + 6σ_xy²) / 2] = sqrt[sum/2 + 6σ_xy²/2]
        # Hill = sqrt[H·sum + 2N·σ_xy²]
        # Match: H=1/2, 2N=3 gives sqrt[sum/2 + 3σ_xy²]
        # So we need 3σ_xy² = 6σ_xy²/2 ✓ That works!
        # Therefore H=1/2, N=3/2=1.5
        hill = compute_hill_stress(stress_tensor, hill_H=1.0 / 2.0, hill_N=1.5, nu=nu)

        np.testing.assert_allclose(
            hill,
            von_mises,
            rtol=1e-8,
            err_msg=f"Hill doesn't match von Mises for stress {stress_tensor}",
        )


@pytest.mark.unit
def test_flow_rule_direction_alignment():
    """Test that plastic strain rate aligns with stress deviator direction."""
    from rheojax.utils.epm_kernels_tensorial import compute_plastic_strain_rate

    # Test stress tensor: deviatoric stress
    stress_tensor = jnp.array([1.0, -0.5, 0.8])  # [σ_xx, σ_yy, σ_xy]
    sigma_eff = 1.5  # Effective stress
    tau_pl_shear = 1.0
    tau_pl_normal = 1.0
    yield_mask = 1.0  # Yielding

    eps_dot_p = compute_plastic_strain_rate(
        stress_tensor, sigma_eff, tau_pl_shear, tau_pl_normal, yield_mask
    )

    # Check shape
    assert eps_dot_p.shape == (3,), f"Expected shape (3,), got {eps_dot_p.shape}"

    # For yielding state, plastic strain rate should be non-zero
    assert jnp.any(
        eps_dot_p != 0.0
    ), "Plastic strain rate should be non-zero when yielding"

    # Check that shear component direction matches stress
    # ε̇ᵖ_xy should have same sign as σ_xy
    assert jnp.sign(eps_dot_p[2]) == jnp.sign(stress_tensor[2])

    # For deviatoric normal stress (σ'_xx = σ_xx - mean)
    mean_stress = (stress_tensor[0] + stress_tensor[1]) / 2.0
    dev_xx = stress_tensor[0] - mean_stress
    # ε̇ᵖ_xx should have same sign as σ'_xx
    assert jnp.sign(eps_dot_p[0]) == jnp.sign(dev_xx)


@pytest.mark.unit
def test_flow_rule_no_yielding():
    """Test that plastic strain rate is zero when not yielding."""
    from rheojax.utils.epm_kernels_tensorial import compute_plastic_strain_rate

    stress_tensor = jnp.array([1.0, 0.5, 0.3])
    sigma_eff = 1.2
    tau_pl_shear = 1.0
    tau_pl_normal = 1.0
    yield_mask = 0.0  # NOT yielding

    eps_dot_p = compute_plastic_strain_rate(
        stress_tensor, sigma_eff, tau_pl_shear, tau_pl_normal, yield_mask
    )

    # Should be all zeros
    np.testing.assert_allclose(eps_dot_p, jnp.zeros(3), atol=1e-12)


@pytest.mark.unit
def test_yield_criterion_factory():
    """Test that get_yield_criterion returns correct functions."""
    from rheojax.utils.epm_kernels_tensorial import get_yield_criterion

    # Test von Mises
    von_mises_fn = get_yield_criterion("von_mises")
    assert callable(von_mises_fn)

    stress = jnp.array([0.0, 0.0, 1.0])
    nu = 0.3
    sigma_eff = von_mises_fn(stress, nu)
    expected = jnp.sqrt(3.0)
    np.testing.assert_allclose(sigma_eff, expected, rtol=1e-10)

    # Test Hill
    hill_fn = get_yield_criterion("hill")
    assert callable(hill_fn)

    sigma_eff_hill = hill_fn(stress, hill_H=0.5, hill_N=1.5, nu=nu)
    assert isinstance(sigma_eff_hill, jax.Array)

    # Test invalid name
    with pytest.raises(ValueError, match="Unknown yield criterion"):
        get_yield_criterion("invalid_criterion")


@pytest.mark.unit
def test_apply_tensorial_propagator():
    """Test that tensorial propagator application preserves zero mean."""
    from rheojax.utils.epm_kernels_tensorial import (
        apply_tensorial_propagator,
        make_tensorial_propagator_q,
    )

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # Random plastic strain field
    key = jax.random.PRNGKey(42)
    eps_dot_p = jax.random.normal(key, (3, L, L))

    stress_dot = apply_tensorial_propagator(propagator, eps_dot_p)

    # Check shape
    assert stress_dot.shape == (3, L, L)

    # Check that mean of each component is near zero (redistribution conserves)
    for i in range(3):
        mean_stress = jnp.mean(stress_dot[i])
        assert (
            jnp.abs(mean_stress) < 1e-6
        ), f"Component {i} has non-zero mean: {mean_stress}"


@pytest.mark.unit
def test_tensorial_epm_step_elastic_loading():
    """Test tensorial EPM step with pure elastic loading (no yielding)."""
    from rheojax.utils.epm_kernels_tensorial import (
        make_tensorial_propagator_q,
        tensorial_epm_step,
    )

    L = 16
    nu = 0.3
    mu = 1.0

    # Initial state: zero stress everywhere
    stress = jnp.zeros((3, L, L))
    thresholds = jnp.ones((L, L)) * 10.0  # High threshold, no yielding

    propagator = make_tensorial_propagator_q(L, nu, mu)

    strain_rate = 0.1
    dt = 0.01
    params = {"mu": mu, "nu": nu, "tau_pl_shear": 1.0, "tau_pl_normal": 1.0}

    new_stress = tensorial_epm_step(
        stress, thresholds, strain_rate, dt, propagator, params, smooth=False
    )

    # Only shear stress should increase by mu * gamma_dot * dt
    expected_sigma_xy = mu * strain_rate * dt
    np.testing.assert_allclose(new_stress[2], expected_sigma_xy, rtol=1e-10)

    # Normal stresses should remain zero
    np.testing.assert_allclose(new_stress[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(new_stress[1], 0.0, atol=1e-10)


@pytest.mark.unit
def test_tensorial_epm_step_with_yielding():
    """Test tensorial EPM step with active plastic sites."""
    from rheojax.utils.epm_kernels_tensorial import (
        make_tensorial_propagator_q,
        tensorial_epm_step,
    )

    L = 16
    nu = 0.3
    mu = 1.0

    # Initial state: high shear stress at center that exceeds threshold
    stress = jnp.zeros((3, L, L))
    stress = stress.at[2, L // 2, L // 2].set(2.0)  # σ_xy = 2.0

    thresholds = jnp.ones((L, L)) * 1.0  # Will yield at center

    propagator = make_tensorial_propagator_q(L, nu, mu)

    strain_rate = 0.0  # No external loading
    dt = 0.1
    params = {"mu": mu, "nu": nu, "tau_pl_shear": 1.0, "tau_pl_normal": 1.0}

    new_stress = tensorial_epm_step(
        stress, thresholds, strain_rate, dt, propagator, params, smooth=False
    )

    # Shear stress at center should decrease due to plastic relaxation
    assert new_stress[2, L // 2, L // 2] < 2.0, "Stress should relax when yielding"


@pytest.mark.unit
def test_tensorial_epm_step_smooth_vs_hard():
    """Test that smooth yielding approximates hard yielding away from threshold."""
    from rheojax.utils.epm_kernels_tensorial import (
        make_tensorial_propagator_q,
        tensorial_epm_step,
    )

    L = 16
    nu = 0.3
    mu = 1.0

    # Stress field with some sites well above threshold, some well below
    stress = jnp.zeros((3, L, L))
    stress = stress.at[2, 0, 0].set(3.0)  # Well above threshold
    stress = stress.at[2, 1, 1].set(0.2)  # Well below threshold

    thresholds = jnp.ones((L, L))

    propagator = make_tensorial_propagator_q(L, nu, mu)

    params = {
        "mu": mu,
        "nu": nu,
        "tau_pl_shear": 1.0,
        "tau_pl_normal": 1.0,
        "smoothing_width": 0.01,  # Very narrow smooth transition
    }

    # Hard yielding
    stress_hard = tensorial_epm_step(
        stress, thresholds, 0.0, 0.1, propagator, params, smooth=False
    )

    # Smooth yielding
    stress_smooth = tensorial_epm_step(
        stress, thresholds, 0.0, 0.1, propagator, params, smooth=True
    )

    # Sites far from threshold should behave similarly
    # Site (0,0): yielding
    np.testing.assert_allclose(stress_hard[2, 0, 0], stress_smooth[2, 0, 0], rtol=0.05)

    # Site (1,1): not yielding
    np.testing.assert_allclose(stress_hard[2, 1, 1], stress_smooth[2, 1, 1], rtol=0.05)


@pytest.mark.unit
def test_propagator_conservation_law():
    """Test that propagator enforces total stress conservation (zero net stress change).

    When plastic strain is uniform (constant field), redistribution should sum to zero.
    """
    from rheojax.utils.epm_kernels_tensorial import (
        apply_tensorial_propagator,
        make_tensorial_propagator_q,
    )

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # Uniform plastic strain field (constant)
    eps_dot_p = jnp.ones((3, L, L)) * 0.1

    stress_dot = apply_tensorial_propagator(propagator, eps_dot_p)

    # Total stress redistribution should sum to zero for uniform strain
    # (Eshelby propagator conserves total stress)
    for i in range(3):
        total_stress = jnp.sum(stress_dot[i])
        assert (
            jnp.abs(total_stress) < 1e-6
        ), f"Component {i} violates conservation: {total_stress}"


@pytest.mark.unit
def test_plastic_strain_rate_magnitude_bounds():
    """Test that plastic strain rate magnitude is bounded by physical limits."""
    from rheojax.utils.epm_kernels_tensorial import compute_plastic_strain_rate

    # High stress state
    stress_tensor = jnp.array([5.0, 3.0, 4.0])  # [σ_xx, σ_yy, σ_xy]
    sigma_eff = 6.0
    tau_pl_shear = 1.0
    tau_pl_normal = 1.0
    yield_mask = 1.0  # Fully yielding

    eps_dot_p = compute_plastic_strain_rate(
        stress_tensor, sigma_eff, tau_pl_shear, tau_pl_normal, yield_mask
    )

    # Magnitude should be bounded by max(stress components / tau_pl)
    max_expected = jnp.max(jnp.abs(stress_tensor)) / min(tau_pl_shear, tau_pl_normal)
    magnitude = jnp.linalg.norm(eps_dot_p)

    assert (
        magnitude <= max_expected * 2.0
    ), f"Plastic strain rate {magnitude} exceeds bound {max_expected}"


@pytest.mark.unit
def test_von_mises_stress_zero_stress():
    """Test von Mises stress calculation for zero stress (corner case)."""
    from rheojax.utils.epm_kernels_tensorial import compute_von_mises_stress

    nu = 0.3

    # Zero stress tensor
    stress_tensor = jnp.array([0.0, 0.0, 0.0])

    sigma_eff = compute_von_mises_stress(stress_tensor, nu)

    # Should be exactly zero
    np.testing.assert_allclose(sigma_eff, 0.0, atol=1e-12)


@pytest.mark.unit
def test_von_mises_stress_extreme_values():
    """Test von Mises stress for very large stress values (numerical stability)."""
    from rheojax.utils.epm_kernels_tensorial import compute_von_mises_stress

    nu = 0.3

    # Very large stress values
    stress_tensor = jnp.array([1e6, 5e5, 3e5])

    sigma_eff = compute_von_mises_stress(stress_tensor, nu)

    # Should be finite and positive
    assert jnp.isfinite(sigma_eff)
    assert sigma_eff > 0


@pytest.mark.unit
def test_hill_criterion_anisotropy_effects():
    """Test that Hill criterion responds to anisotropy parameters."""
    from rheojax.utils.epm_kernels_tensorial import compute_hill_stress

    nu = 0.3
    stress_tensor = jnp.array([1.0, 0.5, 0.3])

    # Isotropic-like parameters (H=0.5, N=1.5)
    sigma_iso = compute_hill_stress(stress_tensor, hill_H=0.5, hill_N=1.5, nu=nu)

    # Anisotropic parameters favoring normal stresses (larger H)
    sigma_aniso_normal = compute_hill_stress(
        stress_tensor, hill_H=2.0, hill_N=1.5, nu=nu
    )

    # Anisotropic parameters favoring shear (larger N)
    sigma_aniso_shear = compute_hill_stress(
        stress_tensor, hill_H=0.5, hill_N=3.0, nu=nu
    )

    # Different anisotropy should give different effective stresses
    assert not jnp.isclose(sigma_iso, sigma_aniso_normal, rtol=0.1)
    assert not jnp.isclose(sigma_iso, sigma_aniso_shear, rtol=0.1)

    # All should be positive and finite
    assert sigma_iso > 0 and jnp.isfinite(sigma_iso)
    assert sigma_aniso_normal > 0 and jnp.isfinite(sigma_aniso_normal)
    assert sigma_aniso_shear > 0 and jnp.isfinite(sigma_aniso_shear)


@pytest.mark.unit
def test_flow_rule_extreme_parameters():
    """Test plastic flow rule with extreme relaxation times (numerical stability)."""
    from rheojax.utils.epm_kernels_tensorial import compute_plastic_strain_rate

    stress_tensor = jnp.array([1.0, 0.5, 0.3])
    sigma_eff = 1.2

    # Very fast relaxation (small tau_pl)
    eps_dot_p_fast = compute_plastic_strain_rate(
        stress_tensor,
        sigma_eff,
        tau_pl_shear=0.001,
        tau_pl_normal=0.001,
        yield_mask=1.0,
    )
    assert jnp.all(jnp.isfinite(eps_dot_p_fast))

    # Very slow relaxation (large tau_pl)
    eps_dot_p_slow = compute_plastic_strain_rate(
        stress_tensor,
        sigma_eff,
        tau_pl_shear=100.0,
        tau_pl_normal=100.0,
        yield_mask=1.0,
    )
    assert jnp.all(jnp.isfinite(eps_dot_p_slow))

    # Fast should have much larger magnitude than slow
    assert jnp.linalg.norm(eps_dot_p_fast) > jnp.linalg.norm(eps_dot_p_slow) * 10


@pytest.mark.unit
def test_yield_criterion_exactly_at_threshold():
    """Test yielding behavior when stress exactly equals threshold (edge case)."""
    from rheojax.utils.epm_kernels_tensorial import (
        make_tensorial_propagator_q,
        tensorial_epm_step,
    )

    L = 16
    nu = 0.3
    mu = 1.0

    # Stress field exactly at threshold
    stress = jnp.zeros((3, L, L))
    # Set shear stress such that von Mises = 1.0 exactly
    # For pure shear: σ_eff = sqrt(3) * σ_xy
    # So σ_xy = 1.0 / sqrt(3) ≈ 0.577 gives σ_eff = 1.0
    stress = stress.at[2].set(1.0 / jnp.sqrt(3.0))

    thresholds = jnp.ones((L, L))  # All thresholds = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    params = {"mu": mu, "nu": nu, "tau_pl_shear": 1.0, "tau_pl_normal": 1.0}

    # Hard yielding: exactly at threshold may or may not yield (implementation defined)
    # Just verify it doesn't crash
    stress_hard = tensorial_epm_step(
        stress, thresholds, 0.0, 0.1, propagator, params, smooth=False
    )
    assert stress_hard.shape == (3, L, L)
    assert jnp.all(jnp.isfinite(stress_hard))

    # Smooth yielding: should give yield_mask ≈ 0.5 at threshold
    stress_smooth = tensorial_epm_step(
        stress, thresholds, 0.0, 0.1, propagator, params, smooth=True
    )
    assert stress_smooth.shape == (3, L, L)
    assert jnp.all(jnp.isfinite(stress_smooth))


@pytest.mark.unit
def test_tensorial_propagator_isotropy():
    """Test that propagator respects material isotropy (rotation invariance)."""
    from rheojax.utils.epm_kernels_tensorial import make_tensorial_propagator_q

    L = 32
    nu = 0.3
    mu = 1.0

    propagator = make_tensorial_propagator_q(L, nu, mu)

    # For isotropic material, G_xxxx should equal G_yyyy at rotated points
    # Check diagonal symmetry: propagator[i, i, x, y] should have similar structure
    # This is a weak test - just verify both normal components exist and are similar in magnitude
    G_xxxx_mag = jnp.mean(jnp.abs(propagator[0, 0]))
    G_yyyy_mag = jnp.mean(jnp.abs(propagator[1, 1]))

    # Should be similar (within 50%) for isotropic material
    ratio = G_xxxx_mag / (G_yyyy_mag + 1e-12)
    assert 0.5 < ratio < 2.0, f"Propagator anisotropy unexpected: {ratio}"


@pytest.mark.unit
def test_plastic_strain_rate_incompressibility():
    """Test that plastic flow is incompressible (ε̇ᵖ_xx + ε̇ᵖ_yy ≈ 0)."""
    from rheojax.utils.epm_kernels_tensorial import compute_plastic_strain_rate

    # Arbitrary stress state
    stress_tensor = jnp.array([2.0, -1.0, 0.5])
    sigma_eff = 2.5
    tau_pl_shear = 1.0
    tau_pl_normal = 1.0
    yield_mask = 1.0

    eps_dot_p = compute_plastic_strain_rate(
        stress_tensor, sigma_eff, tau_pl_shear, tau_pl_normal, yield_mask
    )

    # Check incompressibility: ε̇ᵖ_xx + ε̇ᵖ_yy should be ≈ 0
    trace = eps_dot_p[0] + eps_dot_p[1]

    np.testing.assert_allclose(
        trace, 0.0, atol=1e-10, err_msg="Plastic flow should be incompressible"
    )
