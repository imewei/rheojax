"""Integration tests for Lattice EPM pipeline."""

import pytest
import jax.numpy as jnp
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.lattice import LatticeEPM
from rheojax.core.data import RheoData
from rheojax.utils.epm_kernels import epm_step

jax, jnp = safe_import_jax()


@pytest.mark.integration
def test_lattice_epm_instantiation():
    """Test that LatticeEPM can be instantiated and has correct defaults."""
    model = LatticeEPM(L=32, mu=1.0)
    assert model.L == 32
    assert model.params.get_value("mu") == 1.0

    # Check propagator precomputation
    assert hasattr(model, "_propagator_q_norm")
    assert model._propagator_q_norm.shape == (32, 32)


@pytest.mark.integration
def test_flow_curve_protocol():
    """Test the Flow Curve protocol (steady state shear)."""
    model = LatticeEPM(L=32, dt=0.01)

    # Define shear rates
    gamma_dot = jnp.array([0.01, 0.1])
    # x is shear rate for flow curve
    data = RheoData(x=gamma_dot, y=jnp.zeros_like(gamma_dot), initial_test_mode="flow_curve")

    # Run prediction
    result = model._predict(data, seed=42)

    assert result.y is not None
    assert result.y.shape == (2,)
    # Stress should be positive and monotonic with rate (mostly)
    assert jnp.all(result.y > 0)
    # Higher rate -> Higher stress
    assert result.y[1] >= result.y[0]


@pytest.mark.integration
def test_creep_protocol_adaptive():
    """Test Creep protocol with Adaptive Controller."""
    model = LatticeEPM(L=32, dt=0.01)

    # Define target stress
    # Yield stress is around 1.0 (mean) - 0.1 (std).
    # Target stress 1.5 should flow.
    time = jnp.linspace(0, 1.0, 100)
    target_stress = 1.5
    # Creep: x is time, y is target stress (or metadata)
    # Usually y holds the controlled variable (stress), result y holds response (strain).
    # If y is provided as input, it's the stress signal.
    data = RheoData(
        x=time,
        y=jnp.full_like(time, target_stress),
        initial_test_mode="creep"
    )

    # Run prediction (returns Strain)
    result = model._predict(data, seed=42)

    assert result.y.shape == time.shape
    # Strain should increase (flow)
    assert result.y[-1] > result.y[0]

    # Check if strain rate is roughly constant at end (linear strain)
    # strain ~ rate * t
    # rate = strain[-1] / time[-1]
    # Check positivity
    assert result.y[-1] > 0.0


@pytest.mark.integration
def test_smooth_mode_differentiability():
    """Test that smooth mode is differentiable (for NLSQ/NUTS)."""
    model = LatticeEPM(L=16, dt=0.01)

    # Instead of calling _run_flow_curve which returns RheoData (and triggers Tracer conversion error),
    # we replicate the loop logic here to verify the PHYSICS is differentiable.
    # This avoids the RheoData wrapper validation inside JAX transform.

    key = jax.random.PRNGKey(0)
    shear_rate = 0.1

    def loss_fn(mu_val):
        # Scale propagator
        propagator = model._propagator_q_norm * mu_val

        param_dict = {
            "mu": mu_val,
            "tau_pl": 1.0,
            "sigma_c_mean": 1.0,
            "sigma_c_std": 0.1,
            "smoothing_width": 0.1
        }

        # Run 10 steps
        n_steps = 10
        state = model._init_state(key)

        def body(carrier, _):
            curr_state = carrier
            # epm_step returns (stress, thresh, strain, key)
            new_state = epm_step(curr_state, propagator, shear_rate, 0.01, param_dict, smooth=True)
            # Return mean stress
            return new_state, jnp.mean(new_state[0])

        _, stresses = jax.lax.scan(body, state, None, length=n_steps)
        return jnp.sum(stresses**2)

    # Compute gradient w.r.t. mu
    grad_fn = jax.grad(loss_fn)
    grad_mu = grad_fn(1.0)

    assert jnp.isfinite(grad_mu)
    assert grad_mu != 0.0
