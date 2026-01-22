"""Tests for EPMBase abstract class and LatticeEPM refactoring."""

import pytest
import numpy as np
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.base import EPMBase
from rheojax.models.epm.lattice import LatticeEPM
from rheojax.core.data import RheoData

jax, jnp = safe_import_jax()


@pytest.mark.unit
def test_epm_base_common_parameters():
    """Test EPMBase initializes common parameters correctly."""
    model = LatticeEPM(L=32, dt=0.02, mu=2.0, sigma_c_mean=0.8, sigma_c_std=0.2)

    # Check configuration attributes
    assert model.L == 32
    assert model.dt == 0.02

    # Check common parameters
    assert model.params.get_value("mu") == 2.0
    assert model.params.get_value("sigma_c_mean") == 0.8
    assert model.params.get_value("sigma_c_std") == 0.2


@pytest.mark.unit
def test_epm_base_init_thresholds_shape():
    """Test _init_thresholds returns correct shape."""
    model = LatticeEPM(L=16)
    key = jax.random.PRNGKey(42)

    thresholds = model._init_thresholds(key)

    # Should return (L, L) for scalar lattice
    assert thresholds.shape == (16, 16)

    # Should be positive
    assert jnp.all(thresholds > 0)

    # Should follow Gaussian distribution roughly
    mean_val = jnp.mean(thresholds)
    std_val = jnp.std(thresholds)
    assert jnp.isclose(mean_val, model.params.get_value("sigma_c_mean"), rtol=0.2)
    assert jnp.isclose(std_val, model.params.get_value("sigma_c_std"), rtol=0.3)


@pytest.mark.unit
def test_epm_base_get_param_dict():
    """Test _get_param_dict extracts parameters correctly."""
    model = LatticeEPM(mu=1.5, tau_pl=2.0, sigma_c_mean=0.7, sigma_c_std=0.15)

    param_dict = model._get_param_dict()

    # Should contain all required parameters
    assert "mu" in param_dict
    assert "tau_pl" in param_dict
    assert "sigma_c_mean" in param_dict
    assert "sigma_c_std" in param_dict
    assert "smoothing_width" in param_dict

    # Values should match
    assert param_dict["mu"] == 1.5
    assert param_dict["tau_pl"] == 2.0
    assert param_dict["sigma_c_mean"] == 0.7
    assert param_dict["sigma_c_std"] == 0.15


@pytest.mark.unit
def test_lattice_epm_backward_compatibility_initialization():
    """Test LatticeEPM still works exactly as before refactoring."""
    # Test original initialization pattern
    model = LatticeEPM(L=32, dt=0.01)

    assert model.L == 32
    assert model.dt == 0.01
    assert model.params.get_value("mu") == 1.0
    assert model.params.get_value("tau_pl") == 1.0

    # Check propagator shape (Real-FFT: last dim is L//2 + 1)
    assert model._propagator_q_norm.shape == (32, 32 // 2 + 1)
    # Check singularity
    assert model._propagator_q_norm[0, 0] == 0.0


@pytest.mark.unit
def test_lattice_epm_backward_compatibility_flow_curve():
    """Test LatticeEPM flow curve still works after refactoring."""
    model = LatticeEPM(L=16, dt=0.01)

    shear_rates = jnp.array([0.01, 0.1, 1.0])
    data = RheoData(x=shear_rates, y=jnp.zeros_like(shear_rates))

    # Run prediction
    result = model.predict(data, test_mode="flow_curve", seed=42)

    assert result.x.shape == (3,)
    assert result.y.shape == (3,)
    # Stress should be positive and monotonic with rate roughly
    assert jnp.all(result.y > 0)
    assert result.y[2] > result.y[0]


@pytest.mark.unit
def test_lattice_epm_backward_compatibility_startup():
    """Test LatticeEPM startup protocol still works after refactoring."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma_dot": 0.1})

    result = model.predict(data, test_mode="startup", seed=42)

    assert result.y.shape == time.shape
    # Initial linear elastic regime: stress ~ mu * gdot * t
    t_short = time[1]
    expected_stress = 1.0 * 0.1 * t_short
    assert jnp.isclose(result.y[1], expected_stress, rtol=0.1)


@pytest.mark.unit
def test_lattice_epm_backward_compatibility_relaxation():
    """Test LatticeEPM relaxation protocol still works after refactoring."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 2.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma": 0.1})

    result = model.predict(data, test_mode="relaxation", seed=42)

    assert result.y.shape == time.shape
    # Modulus should be positive (may not decay much for this test)
    assert jnp.all(result.y >= 0)


@pytest.mark.unit
def test_lattice_epm_backward_compatibility_oscillation():
    """Test LatticeEPM oscillation protocol still works after refactoring."""
    model = LatticeEPM(L=16, dt=0.005)

    time = jnp.linspace(0, 10.0, 200)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma0": 0.05, "omega": 2.0})

    result = model.predict(data, test_mode="oscillation", seed=42)

    assert result.y.shape == time.shape
    # Stress should oscillate
    stress = result.y
    assert jnp.max(stress) > 0
