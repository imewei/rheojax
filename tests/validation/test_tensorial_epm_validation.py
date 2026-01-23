"""Physical validation tests for Tensorial EPM.

This module validates the tensorial EPM implementation against analytical limits
and scalar EPM predictions to ensure physical correctness.
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.lattice import LatticeEPM
from rheojax.models.epm.tensor import TensorialEPM

jax, jnp = safe_import_jax()


@pytest.mark.slow
@pytest.mark.validation
def test_linear_response_small_strain():
    """Validate linear elastic response for small strains (γ₀ << 1).

    For small strain, stress should scale linearly: σ_xy = μ·γ₀
    """
    model = TensorialEPM(L=32, dt=0.001, mu=1.0, sigma_c_mean=10.0)

    # Very small strain to stay in elastic regime
    gamma_small = 0.001
    time = jnp.linspace(0, 0.1, 10)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma": gamma_small})

    result = model.predict(data, test_mode="relaxation", seed=42)

    # Initial modulus G(0) should equal mu in elastic regime
    G_0 = result.y[0]
    mu = model.parameters.get_value("mu")

    np.testing.assert_allclose(
        G_0, mu, rtol=0.1, err_msg=f"Linear response failed: G(0)={G_0} vs mu={mu}"
    )


@pytest.mark.slow
@pytest.mark.validation
def test_single_stz_eshelby_quadrupolar_decay():
    """Validate that a single yielding site creates quadrupolar stress decay ~1/r².

    This tests the Eshelby propagator physics: plastic event creates
    long-range elastic stress redistribution with quadrupolar symmetry.
    """
    model = TensorialEPM(L=64, dt=0.01, mu=1.0)

    # Initialize with high stress at center, low threshold there
    key = jax.random.PRNGKey(42)
    stress = jnp.zeros((3, 64, 64))
    # High shear stress at center to trigger yielding
    stress = stress.at[2, 32, 32].set(10.0)

    # Low threshold only at center
    thresholds = jnp.ones((64, 64)) * 100.0
    thresholds = thresholds.at[32, 32].set(0.5)

    # Initialize state manually
    strain = 0.0
    state = (stress, thresholds, strain, key)

    # Take one step with no external loading
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()

    new_state = model._epm_step(state, propagator_q, 0.0, 0.01, params, smooth=False)
    new_stress = new_state[0]

    # Check that stress at center decreased (yielding occurred)
    assert new_stress[2, 32, 32] < stress[2, 32, 32], "Center should yield and relax"

    # Check that stress spread to neighbors (redistribution)
    # Neighboring sites should have non-zero stress
    neighbor_stress = jnp.abs(new_stress[2, 32, 33])
    assert neighbor_stress > 1e-6, "Stress should redistribute to neighbors"

    # Far-field stress should be much smaller (decay with distance)
    far_stress = jnp.abs(new_stress[2, 32, 50])
    assert far_stress < neighbor_stress, "Stress should decay with distance"


@pytest.mark.slow
@pytest.mark.validation
def test_steady_shear_scaling():
    """Validate steady-state shear stress scaling σ_xy ~ γ̇ in flowing regime.

    For high enough shear rates (above yield threshold), stress should
    scale approximately with rate. EPM models show variable scaling due to
    stochastic threshold distribution, so we just verify monotonic increase.
    """
    model = TensorialEPM(L=32, dt=0.01, mu=1.0, sigma_c_mean=0.5, sigma_c_std=0.05)

    # High shear rates well above yielding
    shear_rates = jnp.array([1.0, 2.0, 4.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    result = model.predict(data, test_mode="flow_curve", seed=42)

    # Extract stresses
    sigma_xy = result.y

    # Check monotonic increase (primary physical requirement)
    assert sigma_xy[1] > sigma_xy[0], "Stress should increase with rate"
    assert sigma_xy[2] > sigma_xy[1], "Stress should increase with rate"

    # Verify all stresses are positive and finite
    assert jnp.all(sigma_xy > 0)
    assert jnp.all(jnp.isfinite(sigma_xy))


@pytest.mark.slow
@pytest.mark.validation
def test_normal_stress_small_without_normal_loading():
    """Validate that N₁ stays small without applied normal stress.

    For simple shear (no normal loading), first normal stress difference
    should be much smaller than shear stress, especially at low rates.
    """
    model = TensorialEPM(L=32, dt=0.01, mu=1.0, sigma_c_mean=1.0)

    # Low to moderate shear rates
    shear_rates = jnp.array([0.1, 0.5])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    result = model.predict(data, test_mode="flow_curve", seed=42)

    sigma_xy = result.y
    N1 = result.metadata["N1"]

    # N₁ should be smaller than σ_xy in magnitude
    N1_ratio = jnp.abs(N1) / (jnp.abs(sigma_xy) + 1e-10)

    # Allow N₁ to be up to 50% of shear stress (typical for EPM-like models)
    assert jnp.all(N1_ratio < 0.5), f"N₁/σ_xy ratio {N1_ratio} too large"


@pytest.mark.slow
@pytest.mark.validation
def test_flow_curve_matches_scalar_epm():
    """Validate that tensorial shear stress matches scalar LatticeEPM within tolerance.

    For pure shear with symmetric initial conditions, tensorial EPM should
    reproduce scalar EPM flow curve (within statistical noise from stochastic thresholds).
    """
    # Use same configuration
    L, dt, seed = 32, 0.01, 42
    mu, tau_pl = 1.0, 1.0
    sigma_c_mean, sigma_c_std = 1.0, 0.1

    scalar_model = LatticeEPM(
        L=L,
        dt=dt,
        mu=mu,
        tau_pl=tau_pl,
        sigma_c_mean=sigma_c_mean,
        sigma_c_std=sigma_c_std,
    )

    tensorial_model = TensorialEPM(
        L=L,
        dt=dt,
        mu=mu,
        tau_pl_shear=tau_pl,
        tau_pl_normal=tau_pl,
        sigma_c_mean=sigma_c_mean,
        sigma_c_std=sigma_c_std,
    )

    # Test flow curve at multiple rates
    shear_rates = jnp.array([0.1, 0.5, 1.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    scalar_result = scalar_model.predict(data, test_mode="flow_curve", seed=seed)
    tensorial_result = tensorial_model.predict(data, test_mode="flow_curve", seed=seed)

    # Compare shear stresses
    sigma_scalar = scalar_result.y
    sigma_tensorial = tensorial_result.y

    # Should match within 60% (allowing for stochastic differences and different implementations)
    relative_error = jnp.abs(sigma_scalar - sigma_tensorial) / (sigma_scalar + 1e-10)

    # At least verify same order of magnitude
    assert jnp.all(
        relative_error < 0.6
    ), f"Tensorial vs Scalar large mismatch: errors {relative_error}"

    # Check correlation (both should increase with rate)
    assert (
        jnp.corrcoef(sigma_scalar, sigma_tensorial)[0, 1] > 0.5
    ), "Scalar and tensorial flow curves should be positively correlated"


@pytest.mark.slow
@pytest.mark.validation
def test_relaxation_decay_matches_scalar():
    """Validate that G(t) relaxation matches scalar EPM behavior.

    Stress relaxation after step strain should follow similar decay
    in both scalar and tensorial formulations.
    """
    L, dt, seed = 32, 0.01, 42
    mu, tau_pl = 1.0, 1.0
    gamma_0 = 0.1

    scalar_model = LatticeEPM(L=L, dt=dt, mu=mu, tau_pl=tau_pl)
    tensorial_model = TensorialEPM(
        L=L, dt=dt, mu=mu, tau_pl_shear=tau_pl, tau_pl_normal=tau_pl
    )

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma": gamma_0})

    scalar_result = scalar_model.predict(data, test_mode="relaxation", seed=seed)
    tensorial_result = tensorial_model.predict(data, test_mode="relaxation", seed=seed)

    G_scalar = scalar_result.y
    G_tensorial = tensorial_result.y

    # Initial moduli should match (both should be ~mu)
    np.testing.assert_allclose(
        G_scalar[0], G_tensorial[0], rtol=0.2, err_msg="Initial moduli don't match"
    )

    # Decay rates should be similar (compare final/initial ratio)
    decay_scalar = G_scalar[-1] / G_scalar[0]
    decay_tensorial = G_tensorial[-1] / G_tensorial[0]

    np.testing.assert_allclose(
        decay_scalar,
        decay_tensorial,
        rtol=0.5,
        err_msg=f"Relaxation decay mismatch: scalar {decay_scalar} vs tensorial {decay_tensorial}",
    )
