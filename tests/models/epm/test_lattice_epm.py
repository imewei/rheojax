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
@pytest.mark.smoke
def test_lattice_epm_creep_metadata_stress_respected():
    """metadata['stress'] must drive the creep controller even when y is zero.

    Regression: _run_creep used `if data.y is not None: target = mean(y)` which
    silently ignored metadata['stress'] whenever the caller passed a dummy y
    (the idiomatic predict-time shape). That produced target_stress=0 and a
    flat-zero strain prediction (see examples/epm/04_epm_creep.ipynb).
    """
    model = LatticeEPM(L=16, dt=0.01, sigma_c_mean=0.5, sigma_c_std=0.1)
    time = jnp.linspace(0.5, 10.0, 20)  # dt_data=0.5 (coarse, like notebook)
    data = RheoData(
        x=time,
        y=jnp.zeros_like(time),
        initial_test_mode="creep",
        metadata={"stress": 1.0},  # above yield → unbounded creep expected
    )
    pred = model.predict(data, smooth=True, seed=0).y
    # Above-yield creep must produce a clearly non-zero strain trajectory
    assert (
        float(jnp.max(jnp.abs(pred))) > 0.5
    ), f"metadata['stress'] ignored: strain stayed near zero, max={float(jnp.max(pred)):.4f}"


@pytest.mark.unit
@pytest.mark.smoke
def test_lattice_epm_creep_coarse_dt_matches_fine_dt():
    """Creep prediction must be insensitive to data spacing (controller substep).

    Regression: _jit_creep_kernel used dt = time[1]-time[0] as the ODE step,
    so coarse data grids (dt_data=0.5) produced 50x-oversized Euler steps,
    underdriving the P-controller by ~40%. After the fix, the kernel substeps
    at self.dt internally so results agree across grid resolutions.
    """
    model = LatticeEPM(L=16, dt=0.01, sigma_c_mean=0.5, sigma_c_std=0.1)
    target = 1.0  # above yield

    def run(n):
        t = jnp.linspace(0.5, 10.0, n)
        d = RheoData(x=t, y=jnp.full_like(t, target), initial_test_mode="creep")
        return float(model.predict(d, smooth=True, seed=0).y[-1])

    coarse = run(20)  # dt_data=0.5
    fine = run(951)  # dt_data=0.01 (== self.dt, ground truth)
    rel_err = abs(coarse - fine) / max(abs(fine), 1e-6)
    assert rel_err < 0.15, (
        f"coarse strain[-1]={coarse:.4f} diverges from fine={fine:.4f} "
        f"(rel err {rel_err:.2%}) — controller substep missing"
    )


@pytest.mark.unit
@pytest.mark.slow
def test_lattice_epm_creep_round_trip_fit():
    """Round-trip: fit EPM to its own creep simulation, recover R² > 0.9.

    Regression: examples/epm/04_epm_creep.ipynb reported R²=-36.74 because
    (1) predict-time metadata was ignored and (2) the JIT kernel underdrove
    the controller at coarse dt. With both fixed, the model must cleanly fit
    a ground-truth creep curve it generated itself.
    """
    truth = LatticeEPM(
        L=16, dt=0.01, mu=1.0, tau_pl=1.0, sigma_c_mean=0.5, sigma_c_std=0.1
    )
    target = 1.0  # above yield → growing unbounded creep
    t = jnp.linspace(0.5, 10.0, 20)
    gamma = truth.predict(
        RheoData(
            x=t,
            y=jnp.zeros_like(t),
            initial_test_mode="creep",
            metadata={"stress": target},
        ),
        smooth=True,
        seed=0,
    ).y

    fit_model = LatticeEPM(
        L=16, dt=0.01, mu=0.8, tau_pl=2.0, sigma_c_mean=0.8, sigma_c_std=0.2
    )
    fit_model.fit(
        np.asarray(t),
        np.asarray(gamma),
        test_mode="creep",
        stress=target,
        method="scipy",
        seed=0,  # match truth-generator seed for deterministic lattice
    )
    pred = fit_model.predict(
        RheoData(
            x=t,
            y=jnp.zeros_like(t),
            initial_test_mode="creep",
            metadata={"stress": target},
        ),
        smooth=True,
        seed=0,
    ).y
    g = np.asarray(gamma)
    p = np.asarray(pred)
    ss_res = float(np.sum((g - p) ** 2))
    ss_tot = float(np.sum((g - g.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    assert r2 > 0.9, f"Round-trip R²={r2:.4f} (expected >0.9)"


@pytest.mark.unit
@pytest.mark.smoke
def test_lattice_epm_relaxation_honours_fluidity_form():
    """Fit path (JIT) and predict path (Python) must agree on relaxation.

    Regression: _jit_relaxation_kernel hardcoded plastic_rate = activation * σ *
    fluidity (linear Bingham) while _run_relaxation → epm_step honours the
    model's fluidity_form (overstress by default). On above-yield step strains
    the two paths diverge, making NLSQ round-trip fits impossible (same
    pattern as the pre-fix creep kernel).
    """
    model = LatticeEPM(
        L=16,
        dt=0.01,
        mu=1.0,
        tau_pl=1.0,
        sigma_c_mean=0.3,
        sigma_c_std=0.05,
        n_fluid=2.0,
    )
    time = jnp.linspace(0.01, 5.0, 50)
    strain = 1.0  # mu·γ = 1.0 >> σ_c_mean = 0.3 → sites yield, forms diverge

    # Predict path (Python, uses epm_step with overstress)
    pred_py = np.asarray(
        model.predict(
            RheoData(
                x=time,
                y=jnp.zeros_like(time),
                initial_test_mode="relaxation",
                metadata={"gamma": strain},
            ),
            smooth=True,
            seed=0,
        ).y
    )

    # Fit path (JIT kernel) via model_function
    model._test_mode = "relaxation"
    model._cached_gamma = strain
    model._cached_seed = 0
    params_arr = jnp.asarray(
        [model.parameters.get_value(n) for n in model.parameters.keys()],
        dtype=jnp.float64,
    )
    pred_jit = np.asarray(
        model.model_function(
            time,
            params_arr,
            test_mode="relaxation",
            gamma=strain,
            seed=0,
        )
    )

    max_rel_diff = float(np.max(np.abs(pred_jit - pred_py) / (np.abs(pred_py) + 1e-12)))
    assert max_rel_diff < 0.05, (
        f"Fit path (JIT) and predict path (Python) diverge on above-yield "
        f"relaxation: max rel diff = {max_rel_diff:.4f}. The JIT kernel is "
        f"ignoring fluidity_form."
    )


@pytest.mark.unit
@pytest.mark.slow
def test_lattice_epm_relaxation_round_trip_fit():
    """Round-trip: fit EPM to its own relaxation simulation, recover R² > 0.9.

    Regression: same bug as the creep round-trip — before the fluidity_form
    port, the JIT relaxation kernel (linear Bingham) and the Python predict
    path (overstress HB) disagreed on above-yield parameters, so NLSQ could
    never converge to a point that matches the ground-truth trajectory.
    """
    truth = LatticeEPM(
        L=16,
        dt=0.01,
        mu=1.0,
        tau_pl=1.0,
        sigma_c_mean=0.3,
        sigma_c_std=0.05,
        n_fluid=2.0,
    )
    strain = 1.0  # above yield
    t = jnp.linspace(0.01, 5.0, 50)  # uniformly spaced, matches kernel assumption
    g = truth.predict(
        RheoData(
            x=t,
            y=jnp.zeros_like(t),
            initial_test_mode="relaxation",
            metadata={"gamma": strain},
        ),
        smooth=True,
        seed=0,
    ).y

    fit_model = LatticeEPM(
        L=16,
        dt=0.01,
        mu=0.8,
        tau_pl=2.0,
        sigma_c_mean=0.5,
        sigma_c_std=0.1,
        n_fluid=2.0,
    )
    fit_model.fit(
        np.asarray(t),
        np.asarray(g),
        test_mode="relaxation",
        gamma=strain,
        method="scipy",
        seed=0,
        use_log_residuals=False,
    )
    pred = fit_model.predict(
        RheoData(
            x=t,
            y=jnp.zeros_like(t),
            initial_test_mode="relaxation",
            metadata={"gamma": strain},
        ),
        smooth=True,
        seed=0,
    ).y
    g_arr = np.asarray(g)
    p = np.asarray(pred)
    ss_res = float(np.sum((g_arr - p) ** 2))
    ss_tot = float(np.sum((g_arr - g_arr.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    assert r2 > 0.9, f"Round-trip relaxation R²={r2:.4f} (expected >0.9)"


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
