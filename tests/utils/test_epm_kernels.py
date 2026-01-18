"""Tests for EPM kernels."""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.utils.epm_kernels import (
    compute_plastic_strain_rate,
    make_propagator_q,
    solve_elastic_propagator,
    epm_step,
    update_yield_thresholds
)

jax, jnp = safe_import_jax()


@pytest.mark.unit
def test_propagator_symmetry():
    """Test that the propagator has the correct quadrupolar symmetry."""
    L = 33  # Odd size to have a clear center
    propagator_q = make_propagator_q(L, L)

    # Create a source plastic strain rate at the center
    plastic_strain_rate = jnp.zeros((L, L))
    plastic_strain_rate = plastic_strain_rate.at[L // 2, L // 2].set(1.0)

    # Solve for stress redistribution rate
    stress_redist = solve_elastic_propagator(plastic_strain_rate, propagator_q)

    # Check 4-fold symmetry (rotation by 90 degrees)
    # The quadrupolar field is invariant under 90 degree rotation + flipping?
    # Actually, xy quadrupole sin(2theta) or cos(4theta)?
    # Standard EPM is typically -cos(4theta).
    # Rotating by 90 degrees (pi/2) -> cos(4(theta + pi/2)) = cos(4theta + 2pi) = cos(4theta).
    # So it should be invariant under 90 deg rotation.
    stress_rot90 = jnp.rot90(stress_redist)

    np.testing.assert_allclose(stress_redist, stress_rot90, atol=1e-5)

    # Also check 180 degree rotation (always true for centrosymmetric kernels)
    stress_rot180 = jnp.rot90(stress_redist, 2)
    np.testing.assert_allclose(stress_redist, stress_rot180, atol=1e-5)


@pytest.mark.unit
def test_propagator_zero_mean():
    """Test that redistribution strictly preserves mean zero stress."""
    L = 32
    propagator_q = make_propagator_q(L, L)

    # Random plastic strain field
    key = jax.random.PRNGKey(0)
    plastic_strain_rate = jax.random.normal(key, (L, L))

    stress_redist = solve_elastic_propagator(plastic_strain_rate, propagator_q)

    mean_stress = jnp.mean(stress_redist)
    assert jnp.abs(mean_stress) < 1e-6


@pytest.mark.unit
def test_smooth_vs_hard_yielding():
    """Test that smooth yielding approximates hard yielding."""
    stress = jnp.array([-2.0, -1.2, -0.8, 0.0, 0.8, 1.2, 2.0])
    thresholds = jnp.ones_like(stress)

    # Hard mode
    rate_hard = compute_plastic_strain_rate(stress, thresholds, smooth=False)
    # Expected: [-2.0, -1.2, 0, 0, 0, 1.2, 2.0]
    expected_hard = jnp.where(jnp.abs(stress) > 1.0, stress, 0.0)
    np.testing.assert_allclose(rate_hard, expected_hard)

    # Smooth mode
    rate_smooth = compute_plastic_strain_rate(
        stress, thresholds, smooth=True, smoothing_width=0.01
    )

    # Check that values far from threshold match hard mode
    mask_far = jnp.abs(jnp.abs(stress) - 1.0) > 0.1
    np.testing.assert_allclose(
        rate_smooth[mask_far], expected_hard[mask_far], atol=1e-2
    )

@pytest.mark.unit
def test_epm_step_hard():
    """Test one EPM step in hard mode."""
    L = 16
    stress = jnp.zeros((L, L))
    thresholds = jnp.ones((L, L))
    strain = 0.0
    key = jax.random.PRNGKey(42)
    state = (stress, thresholds, strain, key)

    propagator_q = make_propagator_q(L, L)
    shear_rate = 0.1
    dt = 0.01
    params = {"mu": 1.0, "tau_pl": 1.0}

    # Run one step (Elastic loading only since stress=0)
    new_state = epm_step(state, propagator_q, shear_rate, dt, params, smooth=False)
    new_stress = new_state[0]
    new_strain = new_state[2]

    # Should increase by mu * gdot * dt
    expected_stress = stress + 1.0 * 0.1 * 0.01
    np.testing.assert_allclose(new_stress, expected_stress)
    assert new_strain == 0.1 * 0.01

@pytest.mark.unit
def test_epm_step_plastic():
    """Test EPM step with active plastic sites."""
    L = 16
    # Create stress > threshold
    stress = jnp.zeros((L, L))
    stress = stress.at[0, 0].set(2.0)
    thresholds = jnp.ones((L, L))
    strain = 0.0
    key = jax.random.PRNGKey(42)
    state = (stress, thresholds, strain, key)

    propagator_q = make_propagator_q(L, L)

    # With gdot=0, site should relax
    new_state = epm_step(state, propagator_q, 0.0, 0.1, {"mu": 1.0, "tau_pl": 1.0}, smooth=False)

    # Site (0,0) should decrease (relax)
    assert new_state[0][0, 0] < 2.0

    # Threshold at (0,0) should have renewed (changed) because it yielded
    assert new_state[1][0, 0] != 1.0
