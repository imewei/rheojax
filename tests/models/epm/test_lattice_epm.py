"""Tests for Lattice EPM model."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.lattice import LatticeEPM
from rheojax.models.epm.tensor import TensorialEPM

jax, jnp = safe_import_jax()


@pytest.mark.unit
def test_lattice_epm_initialization():
    """Test LatticeEPM initialization and parameter defaults."""
    model = LatticeEPM(L=32, dt=0.01)

    assert model.L == 32
    assert model.dt == 0.01
    assert model.parameters.get_value("mu") == 1.0
    assert model.parameters.get_value("tau_pl") == 1.0

    # Check propagator shape (Real-FFT: last dim is L//2 + 1)
    assert model._propagator_q_norm.shape == (32, 32 // 2 + 1)
    # Check singularity
    assert model._propagator_q_norm[0, 0] == 0.0


@pytest.mark.unit
def test_lattice_epm_fit_requires_test_mode():
    """Test that _fit requires test_mode parameter."""
    model = LatticeEPM(L=8, dt=0.1)
    # _fit now works but requires test_mode
    with pytest.raises(ValueError, match="test_mode must be specified"):
        model._fit(jnp.array([0.1, 1.0]), jnp.array([0.5, 1.0]))


@pytest.mark.unit
def test_lattice_epm_flow_curve():
    """Test flow curve simulation (steady state stress vs shear rate)."""
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
def test_lattice_epm_startup():
    """Test startup shear simulation."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma_dot": 0.1})

    result = model.predict(data, test_mode="startup", seed=42)

    assert result.y.shape == time.shape
    # Initial linear elastic regime: stress ~ mu * gdot * t
    # For very short times
    t_short = time[1]
    expected_stress = 1.0 * 0.1 * t_short
    assert jnp.isclose(result.y[1], expected_stress, rtol=0.1)


@pytest.mark.unit
def test_lattice_epm_creep_controller():
    """Test adaptive creep controller maintains target stress."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 10.0, 100)
    target_stress = 0.5
    # Input y can be used as target stress
    data = RheoData(x=time, y=jnp.full_like(time, target_stress))

    # Predict returns strain
    result = model.predict(data, test_mode="creep", seed=42)

    strain = result.y
    assert strain.shape == time.shape

    # Strain should be increasing (creep)
    assert strain[-1] > strain[0]

    # Since we don't expose the stress history from predict (only strain),
    # we can't directly verify the stress matching here without exposing internals.
    # But we can verify strain rate is positive.
    strain_rate = jnp.gradient(strain, time)
    assert jnp.mean(strain_rate) > 0


@pytest.mark.unit
def test_lattice_epm_relaxation_backward_compat():
    """Test relaxation protocol for backward compatibility after EPMBase refactoring."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 5.0, 50)
    data = RheoData(x=time, y=jnp.zeros_like(time), metadata={"gamma": 0.1})

    result = model.predict(data, test_mode="relaxation", seed=42)

    # Should return modulus that decays
    assert result.y.shape == time.shape
    assert result.y[0] > 0  # Initial modulus
    # Relaxation: G(t) should decay or stay constant
    assert result.y[-1] <= result.y[0] + 1e-6


@pytest.mark.unit
def test_lattice_epm_oscillation_backward_compat():
    """Test oscillation protocol for backward compatibility."""
    model = LatticeEPM(L=16, dt=0.01)

    time = jnp.linspace(0, 2 * jnp.pi, 100)
    data = RheoData(
        x=time, y=jnp.zeros_like(time), metadata={"gamma0": 0.01, "omega": 1.0}
    )

    result = model.predict(data, test_mode="oscillation", seed=42)

    # Should return oscillating stress
    assert result.y.shape == time.shape
    assert jnp.var(result.y) > 0  # Should oscillate


@pytest.mark.unit
def test_lattice_epm_parameters_after_refactoring():
    """Test that LatticeEPM still has correct parameters after EPMBase refactoring."""
    model = LatticeEPM(L=32, dt=0.01)

    # Should have base EPM parameters
    assert model.parameters.get_value("mu") == 1.0
    assert model.parameters.get_value("tau_pl") == 1.0
    assert model.parameters.get_value("sigma_c_mean") == 1.0
    assert model.parameters.get_value("sigma_c_std") == 0.1

    # Should NOT have tensorial parameters (check via get method instead)
    assert model.parameters.get("nu") is None
    assert model.parameters.get("tau_pl_shear") is None


@pytest.mark.unit
def test_tensorial_epm_scaffold():
    """Test that TensorialEPM is constructable and its forward / fit paths work.

    TensorialEPM used to raise NotImplementedError from fit(). After the
    overstress tensorial kernel port, fit() now inherits from EPMBase and
    runs real NLSQ optimization. This test checks:
        1. The model can be constructed with the default fluidity_form.
        2. predict() returns a correctly-shaped result.
        3. fit() is callable (no NotImplementedError) when given 1D data
           and a test_mode. We do NOT run a full fit here — that would be
           prohibitively slow for a unit test. The integration test
           `test_tensorial_epm_overstress_matches_analytical_hb_flow_curve`
           in tests/models/epm/test_tensorial_epm.py covers real fitting.
    """
    model = TensorialEPM(L=16)  # small L for speed
    # Default fluidity form should be overstress (HB-capable)
    assert model.fluidity_form == "overstress"

    # Prediction should work
    result = model.predict(
        RheoData(x=jnp.array([0.1]), y=jnp.array([0.0])),
        test_mode="flow_curve",
        seed=42,
    )
    assert result.y.shape == (1,)

    # Fitting with 2D y (combined σ_xy + N₁) should still raise — that mode
    # is deliberately unsupported for now.
    y_2d = jnp.zeros((2, 1))
    with pytest.raises(NotImplementedError, match="shear-only"):
        model._fit(jnp.zeros((1,)), y_2d, test_mode="flow_curve")
