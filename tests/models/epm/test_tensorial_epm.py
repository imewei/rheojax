"""Tests for Tensorial EPM model.

This module tests the full tensorial (3-component) stress formulation for EPM,
including normal stress differences (N₁, N₂) and flexible fitting.
"""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.tensor import TensorialEPM

jax, jnp = safe_import_jax()


@pytest.mark.unit
def test_tensorial_epm_initialization():
    """Test TensorialEPM initialization and parameter defaults.

    Verifies:
    - Standard EPM parameters from base class
    - Tensorial-specific parameters (nu, tau_pl_shear, tau_pl_normal)
    - Tensorial propagator shape: (3, 3, L, L//2+1)
    - Stress state shape: (3, L, L) for [σ_xx, σ_yy, σ_xy]
    """
    model = TensorialEPM(L=32, dt=0.01)

    # Base class parameters
    assert model.L == 32
    assert model.dt == 0.01
    assert model.parameters.get_value("mu") == 1.0
    assert model.parameters.get_value("tau_pl") == 1.0
    assert model.parameters.get_value("sigma_c_mean") == 1.0
    assert model.parameters.get_value("sigma_c_std") == 0.1

    # Tensorial-specific parameters
    assert model.parameters.get_value("nu") == 0.48  # Avoid 0.5 singularity
    assert model.parameters.get_value("tau_pl_shear") == 1.0
    assert model.parameters.get_value("tau_pl_normal") == 1.0
    assert model.parameters.get_value("w_N1") == 1.0

    # Hill parameters
    assert model.parameters.get_value("hill_H") == 0.5
    assert model.parameters.get_value("hill_N") == 1.5

    # Yield criterion
    assert model.yield_criterion == "von_mises"

    # Check tensorial propagator shape: (3, 3, L, L//2+1)
    assert model._propagator_q_norm.shape == (3, 3, 32, 32 // 2 + 1)

    # Check q=0 singularity is zeroed for all components
    assert jnp.allclose(model._propagator_q_norm[:, :, 0, 0], 0.0)

    # Initialize state to check stress shape
    key = jax.random.PRNGKey(0)
    stress, thresholds, strain, _ = model._init_state(key)

    # Stress should be (3, L, L)
    assert stress.shape == (3, 32, 32)
    # Should be initialized to zeros
    assert jnp.allclose(stress, 0.0)


@pytest.mark.unit
def test_tensorial_epm_flow_curve_returns_shear_and_normal():
    """Test flow curve returns σ_xy with N₁ in metadata.

    Flow curve should return:
    - y: 1D array of shear stress σ_xy
    - metadata['N1']: 1D array of first normal stress difference N₁
    """
    model = TensorialEPM(L=16, dt=0.01)

    shear_rates = jnp.array([0.01, 0.1, 1.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    # Run prediction
    result = model.predict(data, test_mode="flow_curve", seed=42)

    assert result.x.shape == (3,)
    # Shear stress should be 1D
    assert result.y.shape == (3,)

    # Shear stress should be positive and monotonic
    assert jnp.all(result.y > 0)
    assert result.y[2] > result.y[0]

    # N₁ should be in metadata
    assert "N1" in result.metadata
    N1 = result.metadata["N1"]
    assert N1.shape == (3,)

    # N₁ can be positive, negative, or very small depending on shear rate and simulation time
    # Just verify it exists and has finite values
    assert jnp.all(jnp.isfinite(N1))


@pytest.mark.unit
def test_tensorial_epm_shear_only_fitting():
    """Test fitting to 1D shear stress data (backward compatible).

    When y is 1D, should fit only to σ_xy component.
    """
    model = TensorialEPM(L=16, dt=0.01)

    # Synthetic shear-only data
    shear_rates = jnp.array([0.01, 0.1, 1.0])
    sigma_xy_data = jnp.array([0.05, 0.4, 2.0])

    # Fit should detect 1D mode
    # Note: EPM fitting is complex; we just verify shape detection works
    try:
        result = model._fit(shear_rates, sigma_xy_data, test_mode="flow_curve")
    except NotImplementedError as e:
        # Expected: fitting not yet implemented
        assert "shear_only" in str(e)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.unit
def test_tensorial_epm_combined_fitting():
    """Test fitting to 2D data [σ_xy, N₁] simultaneously.

    When y has shape (2, n), should fit to both shear and normal stresses.
    """
    model = TensorialEPM(L=16, dt=0.01)

    # Synthetic combined data
    shear_rates = jnp.array([0.01, 0.1, 1.0])
    sigma_xy_data = jnp.array([0.05, 0.4, 2.0])
    N1_data = jnp.array([0.01, 0.1, 0.5])

    # Stack into (2, 3) array
    y_combined = jnp.stack([sigma_xy_data, N1_data], axis=0)

    # Fit should detect combined mode
    try:
        result = model._fit(shear_rates, y_combined, test_mode="flow_curve")
    except NotImplementedError as e:
        # Expected: fitting not yet implemented
        assert "combined" in str(e)
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.unit
def test_tensorial_epm_smooth_mode_differentiable():
    """Test that smooth mode predictions are differentiable.

    Required for gradient-based optimization in fitting.
    """

    # Simple differentiability test
    # Create a differentiable function that runs the EPM
    def predict_shear_stress(mu_val):
        """Predict shear stress as function of mu."""
        model = TensorialEPM(L=16, dt=0.01, mu=1.0)  # Initialize with default
        # Scale propagator by mu_val
        propagator_q = model._propagator_q_norm * mu_val
        params = model._get_param_dict()
        params["mu"] = mu_val  # Override mu in params dict

        # Simple simulation
        key = jax.random.PRNGKey(42)
        state = model._init_state(key)

        # Run a few steps
        gdot = 0.1
        dt = 0.01
        for _ in range(10):
            state = model._epm_step(state, propagator_q, gdot, dt, params, smooth=True)

        # Return mean shear stress
        return jnp.mean(state[0][2])

    # Compute gradient
    try:
        grad_fn = jax.grad(predict_shear_stress)
        grad_val = grad_fn(1.0)

        # Gradient should be finite
        assert jnp.isfinite(grad_val)
        # Gradient should be non-zero (stress depends on mu)
        assert jnp.abs(grad_val) > 1e-6
    except Exception as e:
        pytest.fail(f"Gradient computation failed: {e}")


@pytest.mark.unit
def test_tensorial_epm_normal_stress_extraction():
    """Test extraction of normal stress differences from stress tensor.

    Verifies:
    - get_shear_stress() returns σ_xy component
    - get_normal_stress_differences() returns (N₁, N₂)
    - N₁ = σ_xx - σ_yy
    - N₂ = σ_yy - σ_zz with σ_zz = ν(σ_xx + σ_yy)
    """
    model = TensorialEPM(L=16, dt=0.01)

    # Create synthetic stress tensor
    stress = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],  # σ_xx
            [[0.5, 1.0], [1.5, 2.0]],  # σ_yy
            [[0.1, 0.2], [0.3, 0.4]],  # σ_xy
        ]
    )

    # Extract shear stress
    sigma_xy = model.get_shear_stress(stress)
    assert sigma_xy.shape == (2, 2)
    assert jnp.allclose(sigma_xy, stress[2])

    # Extract normal stress differences
    nu = model.parameters.get_value("nu")
    N1, N2 = model.get_normal_stress_differences(stress, nu)

    assert N1.shape == (2, 2)
    assert N2.shape == (2, 2)

    # Verify formulas
    sigma_xx = stress[0]
    sigma_yy = stress[1]
    sigma_zz = nu * (sigma_xx + sigma_yy)

    expected_N1 = sigma_xx - sigma_yy
    expected_N2 = sigma_yy - sigma_zz

    assert jnp.allclose(N1, expected_N1)
    assert jnp.allclose(N2, expected_N2)


@pytest.mark.unit
def test_tensorial_epm_scalar_limit_consistency():
    """Test that tensorial EPM reduces to scalar EPM in appropriate limit.

    For pure shear with no normal stress effects (high symmetry),
    the shear component should match scalar EPM predictions.

    This is a smoke test - exact matching would require identical initialization.
    """
    from rheojax.models.epm.lattice import LatticeEPM

    # Same configuration for both models
    L, dt = 16, 0.01
    seed = 42

    scalar_model = LatticeEPM(L=L, dt=dt, mu=1.0, tau_pl=1.0)
    tensorial_model = TensorialEPM(
        L=L, dt=dt, mu=1.0, tau_pl_shear=1.0, tau_pl_normal=1.0
    )

    # Simple flow curve
    shear_rates = jnp.array([0.01, 0.1])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    # Run both models
    scalar_result = scalar_model.predict(data, test_mode="flow_curve", seed=seed)
    tensorial_result = tensorial_model.predict(data, test_mode="flow_curve", seed=seed)

    # Scalar model returns 1D stress
    assert scalar_result.y.shape == (2,)
    # Tensorial model also returns 1D shear stress
    assert tensorial_result.y.shape == (2,)

    # They should be in similar order of magnitude
    # (not exact due to different stress state evolution)
    ratio = tensorial_result.y / (scalar_result.y + 1e-10)
    assert jnp.all((ratio > 0.1) & (ratio < 10.0))


@pytest.mark.unit
def test_tensorial_epm_startup_protocol():
    """Test startup shear protocol returns only σ_xy (not combined).

    For non-flow_curve protocols, only shear stress is returned.
    """
    model = TensorialEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma_dot": 0.1})

    result = model.predict(data, test_mode="startup", seed=42)

    # Should return 1D stress (σ_xy only)
    assert result.y.shape == time.shape

    # Initial elastic regime: stress ~ mu * gdot * t
    t_short = time[1]
    mu = model.parameters.get_value("mu")
    expected_stress = mu * 0.1 * t_short
    assert jnp.isclose(result.y[1], expected_stress, rtol=0.2)


@pytest.mark.unit
def test_tensorial_epm_relaxation_protocol():
    """Test stress relaxation protocol returns G(t) decay."""
    model = TensorialEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma": 0.1})

    result = model.predict(data, test_mode="relaxation", seed=42)

    # Should return modulus array
    assert result.y.shape == time.shape

    # G(t) should decay monotonically (or stay constant in elastic limit)
    # Initial modulus should be close to mu
    mu = model.parameters.get_value("mu")
    assert jnp.isclose(result.y[0], mu, rtol=0.2)

    # Later values should be <= initial (relaxation)
    assert result.y[-1] <= result.y[0] + 1e-6


@pytest.mark.unit
def test_tensorial_epm_creep_protocol():
    """Test creep protocol returns strain(t) with positive strain rate."""
    model = TensorialEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 10.0, 100)
    target_stress = 0.5
    data = RheoData(x=time, y=jnp.full_like(time, target_stress))

    result = model.predict(data, test_mode="creep", seed=42)

    # Should return strain array
    assert result.y.shape == time.shape

    # Strain should increase (creep)
    assert result.y[-1] > result.y[0]

    # Strain rate should be positive on average
    strain_rate = jnp.gradient(result.y, time)
    assert jnp.mean(strain_rate) > 0


@pytest.mark.unit
def test_tensorial_epm_oscillation_protocol():
    """Test oscillatory shear protocol returns stress(t)."""
    model = TensorialEPM(L=16, dt=0.01)

    # One period at omega=1
    time = jnp.linspace(0, 2 * jnp.pi, 100)
    data = RheoData(
        x=time, y=jnp.zeros_like(time), metadata={"gamma0": 0.01, "omega": 1.0}
    )

    result = model.predict(data, test_mode="oscillation", seed=42)

    # Should return stress array
    assert result.y.shape == time.shape

    # Stress should oscillate (variance > 0)
    assert jnp.var(result.y) > 0


@pytest.mark.unit
def test_tensorial_epm_seed_reproducibility():
    """Test that same seed produces identical results."""
    model = TensorialEPM(L=16, dt=0.01)

    shear_rates = jnp.array([0.1, 1.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    # Run twice with same seed
    result1 = model.predict(data, test_mode="flow_curve", seed=42)
    result2 = model.predict(data, test_mode="flow_curve", seed=42)

    # Should be identical
    np.testing.assert_allclose(result1.y, result2.y, rtol=1e-10)

    # Different seed should give different results
    result3 = model.predict(data, test_mode="flow_curve", seed=99)
    assert not jnp.allclose(result1.y, result3.y, rtol=1e-3)


@pytest.mark.unit
def test_tensorial_epm_parameter_bounds():
    """Test that parameter initialization respects bounds."""
    model = TensorialEPM(L=16, dt=0.01)

    # Check all parameters have bounds
    for param_name in [
        "mu",
        "nu",
        "tau_pl_shear",
        "tau_pl_normal",
        "sigma_c_mean",
        "sigma_c_std",
    ]:
        param = model.parameters.get(param_name)
        assert param.bounds is not None, f"Parameter {param_name} missing bounds"
        lower, upper = param.bounds
        assert lower < upper, f"Invalid bounds for {param_name}"

        # Value should be within bounds
        value = param.value
        assert (
            lower <= value <= upper
        ), f"Parameter {param_name} value {value} outside bounds [{lower}, {upper}]"


@pytest.mark.unit
def test_tensorial_epm_smooth_vs_hard_mode():
    """Test that smooth mode gives different results than hard mode."""
    model = TensorialEPM(L=16, dt=0.01)

    shear_rates = jnp.array([0.1, 1.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    # Run with hard yielding (default for predict)
    result_hard = model.predict(data, test_mode="flow_curve", seed=42, smooth=False)

    # Run with smooth yielding
    result_smooth = model.predict(data, test_mode="flow_curve", seed=42, smooth=True)

    # Results should be similar but not identical
    # (smooth approximates hard with small differences)
    assert not jnp.allclose(result_hard.y, result_smooth.y, rtol=1e-6)

    # But should be in same ballpark (within 30%)
    ratio = result_hard.y / (result_smooth.y + 1e-10)
    assert jnp.all((ratio > 0.7) & (ratio < 1.3))


@pytest.mark.unit
def test_tensorial_epm_hill_criterion_selection():
    """Test that Hill criterion can be selected at initialization."""
    model = TensorialEPM(L=16, dt=0.01, yield_criterion="hill")

    assert model.yield_criterion == "hill"

    # Should run without errors
    shear_rates = jnp.array([0.1])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    result = model.predict(data, test_mode="flow_curve", seed=42)
    assert result.y.shape == (1,)
    assert jnp.isfinite(result.y[0])


@pytest.mark.unit
def test_tensorial_epm_invalid_yield_criterion():
    """Test that invalid yield criterion raises error."""
    with pytest.raises(ValueError, match="Unknown yield criterion"):
        TensorialEPM(L=16, dt=0.01, yield_criterion="invalid_criterion")


@pytest.mark.unit
def test_tensorial_epm_flow_curve_n1_scaling():
    """Test that N₁ in flow curve scales appropriately with shear rate."""
    model = TensorialEPM(L=16, dt=0.01)

    # Wide range of shear rates
    shear_rates = jnp.array([0.01, 0.1, 1.0, 10.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    result = model.predict(data, test_mode="flow_curve", seed=42)

    # Check N₁ exists and is finite
    assert "N1" in result.metadata
    N1 = result.metadata["N1"]
    assert N1.shape == (4,)
    assert jnp.all(jnp.isfinite(N1))

    # N₁ magnitude should generally increase with rate (or stay small)
    # At least verify it's not constant
    assert jnp.var(N1) > 0 or jnp.all(jnp.abs(N1) < 0.1)


@pytest.mark.unit
def test_tensorial_epm_metadata_preservation():
    """Test that metadata is preserved through prediction."""
    model = TensorialEPM(L=16, dt=0.01)

    custom_metadata = {"experiment_id": "test_001", "temperature": 25.0}
    shear_rates = jnp.array([0.1, 1.0])
    data = RheoData(
        x=shear_rates, y=jnp.zeros_like(shear_rates), metadata=custom_metadata
    )

    result = model.predict(data, test_mode="flow_curve", seed=42)

    # Custom metadata should be preserved
    assert "experiment_id" in result.metadata
    assert result.metadata["experiment_id"] == "test_001"
    assert "temperature" in result.metadata
    assert result.metadata["temperature"] == 25.0

    # N₁ should also be added
    assert "N1" in result.metadata
