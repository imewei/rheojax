"""Tests for EPMBase abstract class and LatticeEPM refactoring."""

import types

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.models.epm.base import EPMBase
from rheojax.models.epm.lattice import LatticeEPM
from rheojax.models.epm.tensor import TensorialEPM

jax, jnp = safe_import_jax()


# Small, fast configuration shared across the coverage tests below: an 8x8
# lattice with a short Bayesian-step budget keeps every simulation well under
# a second while still exercising the full scan machinery.
def _small_lattice(fluidity_form="overstress"):
    return LatticeEPM(L=8, dt=0.01, n_bayesian_steps=20, fluidity_form=fluidity_form)


def _small_tensorial(fluidity_form="overstress"):
    return TensorialEPM(L=8, dt=0.01, n_bayesian_steps=20, fluidity_form=fluidity_form)


def _params_in_order(model):
    """Parameter values as a flat list in ``model.parameters`` order."""
    return [model.parameters.get_value(k) for k in model.parameters.keys()]


def _finite(arr):
    return bool(jnp.all(jnp.isfinite(jnp.asarray(arr))))


# Protocol -> (test_mode, model_function protocol kwargs) for the time-series
# protocols. flow_curve is handled separately (takes shear rates, not time).
_TIME_PROTOCOLS = {
    "startup": ("startup", {"gamma_dot": 0.1}),
    "relaxation": ("relaxation", {"gamma": 0.1}),
    "creep": ("creep", {"stress": 1.0}),
    "oscillation": ("oscillation", {"gamma0": 0.05, "omega": 1.0}),
}


@pytest.mark.unit
def test_epm_base_common_parameters():
    """Test EPMBase initializes common parameters correctly."""
    model = LatticeEPM(L=32, dt=0.02, mu=2.0, sigma_c_mean=0.8, sigma_c_std=0.2)

    # Check configuration attributes
    assert model.L == 32
    assert model.dt == 0.02

    # Check common parameters
    assert model.parameters.get_value("mu") == 2.0
    assert model.parameters.get_value("sigma_c_mean") == 0.8
    assert model.parameters.get_value("sigma_c_std") == 0.2


@pytest.mark.unit
def test_smoothing_width_constructor_kwarg():
    """smoothing_width is documented as configurable; both variants must
    accept it as a constructor kwarg instead of only the hardcoded 0.1.
    """
    lattice = LatticeEPM(smoothing_width=0.05)
    tensorial = TensorialEPM(smoothing_width=0.2)

    assert lattice.parameters.get_value("smoothing_width") == 0.05
    assert tensorial.parameters.get_value("smoothing_width") == 0.2


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
    assert jnp.isclose(mean_val, model.parameters.get_value("sigma_c_mean"), rtol=0.2)
    assert jnp.isclose(std_val, model.parameters.get_value("sigma_c_std"), rtol=0.3)


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
    assert model.parameters.get_value("mu") == 1.0
    assert model.parameters.get_value("tau_pl") == 1.0

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
    data = RheoData(
        x=time, y=jnp.zeros_like(time), metadata={"gamma0": 0.05, "omega": 2.0}
    )

    result = model.predict(data, test_mode="oscillation", seed=42)

    assert result.y.shape == time.shape
    # Stress should oscillate
    stress = result.y
    assert jnp.max(stress) > 0


# =============================================================================
# fluidity_form constitutive-law branches (JIT scalar kernels)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("fluidity_form", ["linear", "power", "overstress"])
def test_model_function_flow_curve_all_fluidity_forms(fluidity_form):
    """model_function flow curve is finite for every constitutive law.

    Exercises the linear/power/overstress branches of ``_jit_flow_curve_single``.
    """
    model = _small_lattice(fluidity_form)
    params = _params_in_order(model)
    shear_rates = jnp.array([0.1, 0.5, 1.0])

    y = model.model_function(shear_rates, params, test_mode="flow_curve")

    assert y.shape == shear_rates.shape
    assert _finite(y)
    # Stress must be positive for positive shear rates under any fluidity form.
    assert jnp.all(y > 0)


@pytest.mark.unit
@pytest.mark.parametrize("fluidity_form", ["linear", "power", "overstress"])
@pytest.mark.parametrize("protocol", list(_TIME_PROTOCOLS))
def test_model_function_time_protocols_all_fluidity_forms(fluidity_form, protocol):
    """Each time-domain JIT kernel runs finitely for every fluidity form.

    Covers the linear/power/overstress branches of the startup, relaxation,
    creep and oscillation JIT kernels.
    """
    model = _small_lattice(fluidity_form)
    params = _params_in_order(model)
    time = jnp.linspace(0.0, 1.0, 12)

    test_mode, kwargs = _TIME_PROTOCOLS[protocol]
    y = model.model_function(time, params, test_mode=test_mode, **kwargs)

    assert y.shape == time.shape
    assert _finite(y)


@pytest.mark.unit
def test_invalid_fluidity_form_rejected_by_constructor():
    """Constructor validates the fluidity_form selector."""
    with pytest.raises(ValueError, match="fluidity_form must be"):
        LatticeEPM(L=8, fluidity_form="bogus")


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_mode,kwargs",
    [
        ("flow_curve", {}),
        ("startup", {"gamma_dot": 0.1}),
        ("relaxation", {"gamma": 0.1}),
        ("creep", {"stress": 1.0}),
        ("oscillation", {"gamma0": 0.05, "omega": 1.0}),
    ],
)
def test_unknown_fluidity_form_raises_in_kernel(test_mode, kwargs):
    """A fluidity_form the kernels don't recognise raises at trace time.

    The constructor guards the selector, so we bypass it to reach the
    defensive ``else`` branch inside each JIT kernel body.
    """
    model = _small_lattice("overstress")
    model.fluidity_form = "not_a_real_form"  # bypass constructor validation
    params = _params_in_order(model)
    x = jnp.array([0.1, 1.0]) if test_mode == "flow_curve" else jnp.linspace(0, 1, 6)

    with pytest.raises(ValueError, match="Unknown fluidity_form"):
        model.model_function(x, params, test_mode=test_mode, **kwargs)


# =============================================================================
# Unknown test-mode dispatch guards
# =============================================================================


@pytest.mark.unit
def test_unknown_test_mode_scalar_path_raises():
    """Scalar model dispatch rejects unknown test modes."""
    model = _small_lattice()
    params = _params_in_order(model)
    with pytest.raises(ValueError, match="Unknown test mode"):
        model.model_function(jnp.array([0.1, 1.0]), params, test_mode="nonsense")


@pytest.mark.unit
def test_unknown_test_mode_general_path_raises():
    """Tensorial (general) model dispatch rejects unknown test modes."""
    model = _small_tensorial()
    params = _params_in_order(model)
    with pytest.raises(ValueError, match="Unknown test mode"):
        model.model_function(jnp.array([0.1, 1.0]), params, test_mode="nonsense")


# =============================================================================
# precompile()
# =============================================================================


@pytest.mark.unit
def test_precompile_triggers_jit_and_sets_flag():
    """precompile() compiles kernels, returns a wall time, and flips the flag."""
    model = _small_lattice()
    assert model._precompiled is False

    elapsed = model.precompile(n_points=3, verbose=False)

    assert isinstance(elapsed, float)
    assert elapsed >= 0.0
    assert model._precompiled is True


@pytest.mark.unit
def test_precompile_verbose_logging_path():
    """precompile(verbose=True) also compiles and returns a wall time."""
    model = _small_lattice()
    elapsed = model.precompile(n_points=3, verbose=True)
    assert isinstance(elapsed, float)
    assert model._precompiled is True


# =============================================================================
# _fit (NLSQ) — smoke coverage of the fitting pipeline
# =============================================================================


@pytest.mark.unit
def test_fit_flow_curve_smoke():
    """_fit runs the NLSQ flow-curve pipeline and marks the model fitted."""
    model = _small_lattice()
    X = np.array([0.1, 0.5, 1.0])
    y = np.array([1.1, 1.4, 1.8])

    model.fit(X, y, test_mode="flow_curve", max_iter=3, seed=0)

    assert model.fitted_ is True
    assert model._test_mode == "flow_curve"


@pytest.mark.unit
def test_fit_creep_caches_data_cadence():
    """Creep _fit caches the data cadence (dt) used by the substepped controller."""
    model = _small_lattice()
    X = np.linspace(0.0, 1.0, 6)
    y = np.linspace(0.0, 0.5, 6)

    model.fit(X, y, test_mode="creep", max_iter=2, seed=0, stress=1.0)

    assert model.fitted_ is True
    # _fit computes the creep cadence outside any JIT trace as a Python float.
    expected_dt = float(X[1] - X[0])
    assert np.isclose(model._creep_dt_data, expected_dt)


# =============================================================================
# General (non-JIT) model functions — tensorial path
# =============================================================================


@pytest.mark.unit
def test_tensorial_model_function_flow_curve():
    """TensorialEPM routes flow_curve through the general model function."""
    model = _small_tensorial()
    params = _params_in_order(model)
    shear_rates = jnp.array([0.1, 0.5, 1.0])

    y = model.model_function(shear_rates, params, test_mode="flow_curve")

    assert y.shape == shear_rates.shape
    assert _finite(y)
    assert jnp.all(y > 0)


@pytest.mark.unit
@pytest.mark.parametrize("protocol", list(_TIME_PROTOCOLS))
def test_tensorial_model_function_time_protocols(protocol):
    """TensorialEPM time protocols run finitely through the general path."""
    model = _small_tensorial()
    params = _params_in_order(model)
    time = jnp.linspace(0.0, 1.0, 12)

    test_mode, kwargs = _TIME_PROTOCOLS[protocol]
    y = model.model_function(time, params, test_mode=test_mode, **kwargs)

    assert y.shape == time.shape
    assert _finite(y)


# =============================================================================
# General (non-JIT) model functions — direct calls on the scalar lattice
# =============================================================================


@pytest.mark.unit
def test_general_model_functions_scalar_direct():
    """Directly exercise the scalar (ndim==2) branches of the _model_* methods."""
    model = _small_lattice()
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    time = jnp.linspace(0.0, 1.0, 10)
    shear_rates = jnp.array([0.1, 1.0])

    flow = model._model_flow_curve(shear_rates, key, propagator_q, params)
    startup = model._model_startup(time, key, propagator_q, params, 0.1)
    relax = model._model_relaxation(time, key, propagator_q, params, 0.1)
    creep = model._model_creep(time, key, propagator_q, params, 1.0)
    osc = model._model_oscillation(time, key, propagator_q, params, 0.05, 1.0)

    assert flow.shape == shear_rates.shape and _finite(flow)
    assert startup.shape == time.shape and _finite(startup)
    assert relax.shape == time.shape and _finite(relax)
    assert creep.shape == time.shape and _finite(creep)
    assert osc.shape == time.shape and _finite(osc)


@pytest.mark.unit
def test_general_model_functions_single_point_branches():
    """len(time)==1 hits the n_steps==0 fall-through in each _model_* method."""
    model = _small_lattice()
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    t1 = jnp.array([0.0])

    assert model._model_startup(t1, key, propagator_q, params, 0.1).shape == (1,)
    assert model._model_relaxation(t1, key, propagator_q, params, 0.1).shape == (1,)
    assert model._model_creep(t1, key, propagator_q, params, 1.0).shape == (1,)
    assert model._model_oscillation(t1, key, propagator_q, params, 0.05, 1.0).shape == (
        1,
    )


@pytest.mark.unit
def test_model_creep_coarse_dt_matches_fine_dt():
    """_model_creep (fit-path; TensorialEPM routes creep NLSQ/Bayesian calls
    here via _model_function_general) must substep the P-controller like
    _model_creep_jit/_run_creep, so results stop depending on the caller's
    data cadence. Mirrors
    test_lattice_epm.py::test_lattice_epm_creep_coarse_dt_matches_fine_dt.
    """
    model = LatticeEPM(L=8, dt=0.01, sigma_c_mean=0.5, sigma_c_std=0.1)
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    target = 1.0  # above yield

    def run(n):
        t = jnp.linspace(0.5, 10.0, n)
        model._creep_dt_data = float(t[1] - t[0])
        return float(model._model_creep(t, key, propagator_q, params, target)[-1])

    coarse = run(20)  # dt_data=0.5
    fine = run(951)  # dt_data=0.01 (== self.dt, ground truth)
    rel_err = abs(coarse - fine) / max(abs(fine), 1e-6)
    assert rel_err < 0.15, (
        f"coarse strain[-1]={coarse:.4f} diverges from fine={fine:.4f} "
        f"(rel err {rel_err:.2%}) — controller substep missing in _model_creep"
    )


@pytest.mark.unit
def test_mean_shear_stress_scalar_and_tensorial():
    """_mean_shear_stress reduces scalar fields and picks sigma_xy for tensors."""
    scalar_field = jnp.ones((4, 4)) * 2.0
    assert jnp.isclose(EPMBase._mean_shear_stress(scalar_field), 2.0)

    # Tensorial (3, L, L): only index [2] (sigma_xy) should be averaged.
    tensor_field = jnp.zeros((3, 4, 4))
    tensor_field = tensor_field.at[0].set(5.0)  # sigma_xx — must be ignored
    tensor_field = tensor_field.at[2].set(3.0)  # sigma_xy — the shear stress
    assert jnp.isclose(EPMBase._mean_shear_stress(tensor_field), 3.0)


# =============================================================================
# Protocol-runner edge cases (single point + missing time array)
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "test_mode,metadata",
    [
        ("startup", {"gamma_dot": 0.1}),
        ("relaxation", {"gamma": 0.1}),
        ("creep", {"stress": 1.0}),
        ("oscillation", {"gamma0": 0.05, "omega": 1.0}),
    ],
)
def test_predict_single_time_point(test_mode, metadata):
    """A single-sample time array exercises the n_steps==0 runner branches."""
    model = _small_lattice()
    t1 = jnp.array([0.0])
    data = RheoData(x=t1, y=jnp.zeros(1), metadata=metadata)

    result = model.predict(data, test_mode=test_mode, seed=0)

    assert result.y.shape == (1,)
    assert _finite(result.y)


@pytest.mark.unit
@pytest.mark.parametrize(
    "runner_name",
    ["_run_startup", "_run_relaxation", "_run_creep", "_run_oscillation"],
)
def test_run_protocols_reject_none_time(runner_name):
    """Every time-domain runner rejects a RheoData with no x (time) array."""
    model = _small_lattice()
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    # SimpleNamespace stands in for a RheoData whose x slot is None.
    none_data = types.SimpleNamespace(x=None, metadata={})

    runner = getattr(model, runner_name)
    with pytest.raises(ValueError, match="must not be None"):
        runner(none_data, key, propagator_q, params, False)


@pytest.mark.unit
def test_run_creep_target_stress_fallbacks():
    """Creep target stress falls back to mean(y), then to 1.0 when y is ~0."""
    model = _small_lattice()
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    time = jnp.linspace(0.0, 1.0, 8)

    # No metadata['stress'] and a non-zero y -> target = mean(y).
    data_nonzero = RheoData(x=time, y=jnp.full_like(time, 2.0), metadata={})
    res_nonzero = model._run_creep(data_nonzero, key, propagator_q, params, False)
    assert res_nonzero.y.shape == time.shape and _finite(res_nonzero.y)

    # No metadata['stress'] and an all-zero y -> target falls back to 1.0.
    data_zero = RheoData(x=time, y=jnp.zeros_like(time), metadata={})
    res_zero = model._run_creep(data_zero, key, propagator_q, params, False)
    assert res_zero.y.shape == time.shape and _finite(res_zero.y)


@pytest.mark.unit
def test_run_creep_guards_nonpositive_dt_data():
    """A duplicated leading timestamp (dt_data<=0) must fall back to self.dt
    rather than collapsing the P-controller substep to dt_sub=0.

    Regression: without the ``dt_data > 0`` guard (present in the matching
    fit-path helper ``_model_creep_jit``), dt_sub == 0 for the whole
    trajectory and strain never accumulates despite a nonzero shear rate.
    """
    model = _small_lattice()
    key = jax.random.PRNGKey(0)
    propagator_q = model._propagator_q_norm * model.parameters.get_value("mu")
    params = model._get_param_dict()
    # Duplicated leading timestamps -> dt_data = time[1] - time[0] == 0.
    time = jnp.array([0.0, 0.0, 0.5, 1.0, 1.5, 2.0])
    data = RheoData(x=time, y=jnp.full_like(time, 2.0), metadata={"stress": 2.0})

    result = model._run_creep(data, key, propagator_q, params, True)

    assert _finite(result.y)
    assert float(result.y[-1]) > float(result.y[0])


# =============================================================================
# _get_param_dict validation
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing",
    ["mu", "tau_pl", "sigma_c_mean", "sigma_c_std", "n_fluid", "smoothing_width"],
)
def test_get_param_dict_raises_on_unset_parameter(missing):
    """_get_param_dict raises a clear error when any parameter value is unset."""
    model = _small_lattice()
    # Force the value to None, bypassing set_value's bounds validation.
    model.parameters.get(missing).value = None

    with pytest.raises(ValueError, match=f"'{missing}' must be set"):
        model._get_param_dict()
