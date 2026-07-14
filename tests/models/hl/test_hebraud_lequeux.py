import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheojax.models.hl import HebraudLequeux


class TestHebraudLequeux:
    @pytest.fixture
    def model(self):
        """Create a standard HL model instance."""
        m = HebraudLequeux()
        # Set to glassy phase
        m.parameters.set_value("alpha", 0.3)
        m.parameters.set_value("tau", 1.0)
        m.parameters.set_value("sigma_c", 1.0)
        return m

    def test_instantiation(self, model):
        """Test model initialization and parameter defaults."""
        assert model.parameters.get_value("alpha") == 0.3
        assert model.parameters.get_value("tau") == 1.0
        assert model.parameters.get_value("sigma_c") == 1.0
        assert model.get_phase_state() == "glass"

    def test_phase_state(self, model):
        """Test phase classification."""
        model.parameters.set_value("alpha", 0.6)
        assert model.get_phase_state() == "fluid"
        model.parameters.set_value("alpha", 0.3)
        assert model.get_phase_state() == "glass"

    @pytest.mark.slow
    def test_flow_curve_prediction(self, model):
        """Test steady shear flow curve prediction."""
        # Use higher shear rates for faster test (fewer time steps needed)
        gdot = np.logspace(0, 1, 5)  # 1.0 to 10.0
        # Use synthetic data for mock fit to allow convergence
        synthetic_stress = np.array([0.5, 0.6, 0.8, 1.2, 2.0])  # Rough HB shape
        model.fit(gdot, synthetic_stress, test_mode="steady_shear", max_iter=2)
        stress = model.predict(gdot)

        assert stress.shape == gdot.shape
        assert np.all(np.isfinite(stress))
        # Stress should increase with shear rate
        assert np.all(np.diff(stress) > 0)
        # Check yield stress existence (stress > 0)
        assert stress[0] > 0.0

    def test_creep_prediction(self, model):
        """Test creep compliance prediction."""
        # Short duration for test speed
        t = np.linspace(0, 1, 10)
        # Mock fit to set mode and context
        synthetic_compliance = t * 0.1  # Mock linear compliance
        model.fit(
            t, synthetic_compliance, test_mode="creep", stress_target=0.5, max_iter=2
        )
        compliance = model.predict(t)

        assert compliance.shape == t.shape
        assert np.all(np.isfinite(compliance))
        # Compliance should be non-negative
        assert np.all(compliance >= 0)

    def test_relaxation_prediction(self, model):
        """Test stress relaxation prediction."""
        # Short duration
        t = np.linspace(0, 1, 10)
        # Mock fit
        synthetic_modulus = np.exp(-t)
        model.fit(t, synthetic_modulus, test_mode="relaxation", gamma0=0.1, max_iter=2)
        G_t = model.predict(t)

        assert G_t.shape == t.shape
        assert np.all(np.isfinite(G_t))
        assert G_t[0] > 0

    def test_startup_prediction(self, model):
        """Test startup stress prediction."""
        t = np.linspace(0, 1, 10)
        synthetic_stress = 1.0 - np.exp(-t)
        model.fit(t, synthetic_stress, test_mode="startup", gdot=1.0, max_iter=2)
        stress = model.predict(t)

        assert stress.shape == t.shape
        assert np.all(np.isfinite(stress))
        assert stress[0] == 0  # Starts at 0

    def test_laos_prediction(self, model):
        """Test LAOS stress prediction."""
        t = np.linspace(0, 1, 20)
        synthetic_stress = np.sin(t)
        model.fit(
            t, synthetic_stress, test_mode="laos", gamma0=1.0, omega=10.0, max_iter=2
        )
        stress = model.predict(t)

        assert stress.shape == t.shape
        assert np.all(np.isfinite(stress))

    def test_grid_scaling(self, model):
        """Test that grid adapts to large sigma_c."""
        model.parameters.set_value("sigma_c", 100.0)
        gdot = np.array([0.1, 1.0])

        # We just want to verify prediction works and returns large values
        # No need to fit, just set the test mode and protocol kwargs manually
        model._test_mode = "steady_shear"
        model._last_fit_kwargs = {
            "_sigma_max": max(5.0, model.grid_sigma_factor * 100.0),
            "_n_bins": 501,
        }

        # Predict
        stress = model.predict(gdot)

        # With sigma_c=100, stress should be > 10
        # If clipped to grid [-5, 5], it would be ~5
        assert stress[0] > 10.0
        assert np.all(np.isfinite(stress))

    def test_bayesian_interface(self, model):
        """Test model_function for Bayesian inference compatibility."""
        gdot = np.array([0.1, 1.0, 10.0])
        params = jnp.array([0.3, 1.0, 1.0])  # alpha, tau, sigma_c

        # Manually setup context to avoid running expensive/unstable fit
        model._test_mode = "steady_shear"
        model._last_fit_kwargs = {}
        # Metadata needed for some modes, but steady_shear might be fine without t_max
        model._fit_data_metadata = {"t_max": 10.0, "len_X": 3}

        # Call model_function
        pred = model.model_function(gdot, params)
        assert pred.shape == gdot.shape


class TestHLValidation:
    """Validation / error-path coverage for the HL fitting dispatcher."""

    @pytest.fixture
    def model(self):
        return HebraudLequeux()

    def test_fit_requires_test_mode(self, model):
        """fit() with no test_mode raises (dispatcher guard)."""
        X = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="test_mode"):
            model.fit(X, X, max_iter=1)

    def test_fit_unsupported_test_mode(self, model):
        """Unknown test_mode is rejected by the dispatcher."""
        X = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="Unsupported test mode"):
            model.fit(X, X, test_mode="not_a_real_mode", max_iter=1)

    def test_creep_requires_stress_target(self, model):
        t = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="stress_target"):
            model.fit(t, t * 0.1, test_mode="creep", max_iter=1)

    def test_relaxation_requires_gamma0(self, model):
        t = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="gamma0"):
            model.fit(t, np.exp(-t), test_mode="relaxation", max_iter=1)

    def test_startup_requires_gdot(self, model):
        t = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="gdot"):
            model.fit(t, 1.0 - np.exp(-t), test_mode="startup", max_iter=1)

    def test_laos_requires_gamma0_and_omega(self, model):
        t = np.linspace(0, 1, 5)
        with pytest.raises(ValueError, match="gamma0 and omega"):
            model.fit(t, np.sin(t), test_mode="laos", gamma0=1.0, max_iter=1)

    def test_predict_before_fit_raises(self, model):
        """_predict guards against a missing test_mode."""
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(np.linspace(0, 1, 5))

    def test_oscillation_bad_g_star_shape(self, model):
        """SAOS fit rejects G* that is neither complex nor (M, 2)."""
        omega = np.logspace(-1, 1, 4)
        bad = np.ones((4, 3))  # wrong second dimension
        with pytest.raises(ValueError, match=r"complex array or \(M, 2\)"):
            model.fit(omega, bad, test_mode="oscillation", max_iter=1)


class TestHLModelFunction:
    """Direct exercise of model_function protocol branches (Bayesian path).

    These bypass the optimizer: we set _test_mode + protocol kwargs manually
    and call model_function once per protocol. This is the cheap way to cover
    the per-mode kernel dispatch, the _kw fallback helper, and
    get_dt_and_n_steps without running a full NUTS trace.
    """

    @pytest.fixture
    def model(self):
        m = HebraudLequeux()
        m.parameters.set_value("alpha", 0.3)
        m.parameters.set_value("tau", 1.0)
        m.parameters.set_value("sigma_c", 1.0)
        # Small grid keeps each kernel evaluation fast.
        return m

    @staticmethod
    def _params():
        return jnp.array([0.3, 1.0, 1.0])

    def test_model_function_requires_mode(self, model):
        model._test_mode = None
        with pytest.raises(ValueError, match="test_mode required"):
            model.model_function(np.array([0.0, 1.0]), self._params())

    def test_model_function_unknown_mode(self, model):
        with pytest.raises(ValueError, match="Unknown test mode"):
            model.model_function(
                np.array([0.0, 1.0]), self._params(), test_mode="bogus"
            )

    def test_model_function_creep(self, model):
        model._test_mode = "creep"
        model._last_fit_kwargs = {
            "stress_target": 0.5,
            "_sigma_max": 10.0,
            "_n_bins": 21,
        }
        t = np.linspace(0, 1, 8)
        out = np.asarray(model.model_function(t, self._params()))
        assert out.shape == t.shape
        assert np.all(np.isfinite(out))

    def test_model_function_relaxation(self, model):
        model._test_mode = "relaxation"
        model._last_fit_kwargs = {
            "gamma0": 0.1,
            "_G0_relax": 1.0,
            "_sigma_max": 5.0,
            "_n_bins": 21,
        }
        t = np.linspace(0, 1, 8)
        out = np.asarray(model.model_function(t, self._params()))
        assert out.shape == t.shape
        assert np.all(np.isfinite(out))

    def test_model_function_startup(self, model):
        model._test_mode = "startup"
        model._last_fit_kwargs = {"gdot": 1.0, "_sigma_max": 10.0, "_n_bins": 21}
        t = np.linspace(0, 1, 8)
        out = np.asarray(model.model_function(t, self._params()))
        assert out.shape == t.shape
        assert np.all(np.isfinite(out))

    def test_model_function_laos(self, model):
        model._test_mode = "laos"
        model._last_fit_kwargs = {
            "gamma0": 1.0,
            "omega": 5.0,
            "_sigma_max": 10.0,
            "_n_bins": 21,
        }
        t = np.linspace(0, 1, 12)
        out = np.asarray(model.model_function(t, self._params()))
        assert out.shape == t.shape
        assert np.all(np.isfinite(out))

    def test_model_function_saos(self, model):
        model._test_mode = "oscillation"
        model._last_fit_kwargs = {
            "n_cycles": 2,
            "gamma0": 0.01,
            "_sigma_max": 5.0,
            "_n_bins": 21,
        }
        omega = np.array([1.0, 5.0])
        out = np.asarray(model.model_function(omega, self._params()))
        # SAOS returns (M, 2) columns [G', G'']
        assert out.shape == (omega.shape[0], 2)
        assert np.all(np.isfinite(out))

    def test_model_function_creep_metadata_fallback(self, model):
        """Empty X falls back to _fit_data_metadata t_max for n_steps."""
        model._test_mode = "creep"
        model._last_fit_kwargs = {
            "stress_target": 0.5,
            "_sigma_max": 10.0,
            "_n_bins": 21,
        }
        model._fit_data_metadata = {"t_max": 1.0}
        # Empty X: float(X[-1]) raises -> n_steps recovered from metadata t_max.
        empty = np.array([], dtype=np.float64)
        out = np.asarray(model.model_function(empty, self._params()))
        assert out.shape == (0,)

    def test_model_function_kwarg_override(self, model):
        """Explicit protocol kwargs take precedence over _last_fit_kwargs (_kw)."""
        model._test_mode = "creep"
        model._last_fit_kwargs = {"_sigma_max": 10.0, "_n_bins": 21}
        t = np.linspace(0, 1, 6)
        # stress_target supplied only via explicit kwarg exercises the _kw
        # kwargs branch (not the _last_fit_kwargs fallback).
        out = np.asarray(model.model_function(t, self._params(), stress_target=0.5))
        assert out.shape == t.shape
        assert np.all(np.isfinite(out))

    def test_predict_unknown_mode_raises(self, model):
        """_predict rejects an unrecognized stored test_mode."""
        model._test_mode = "bogus_mode"
        with pytest.raises(ValueError, match="Unknown test mode"):
            model.predict(np.linspace(0, 1, 5))

    def test_model_function_flow_curve_fallback(self, model, monkeypatch):
        """Flow-curve Bayesian path falls back when the primary solve raises."""
        import rheojax.models.hl.hebraud_lequeux as hl_mod

        calls = {"n": 0}

        def flaky_flow_curve(
            gdot, alpha, tau, sigma_c, dt, sigma_max, n_bins, per_rate_schedule=None
        ):
            calls["n"] += 1
            if per_rate_schedule is not None:
                # Primary (scheduled) path raises -> triggers fallback branch.
                raise RuntimeError("forced primary failure")
            g = np.abs(np.asarray(gdot, dtype=np.float64))
            return jnp.asarray(0.5 * sigma_c + 0.1 * np.sqrt(g))

        monkeypatch.setattr(hl_mod, "run_flow_curve", flaky_flow_curve)

        model._test_mode = "flow_curve"
        model._last_fit_kwargs = {
            "_stress_scale": 1.0,
            "_sigma_max_min_norm": 5.0,
            "_n_bins_fit": 51,
            "_tau_est": 1.0,
            "_sigma_c_est": 1.0,
            "_precomputed_schedule": None,
        }
        gdot = np.array([0.1, 1.0])
        out = np.asarray(model.model_function(gdot, self._params()))
        assert out.shape == gdot.shape
        assert np.all(np.isfinite(out))
        # Both the failing primary and the succeeding fallback ran.
        assert calls["n"] == 2


class TestHLOscillation:
    """SAOS/oscillation fit + predict path (exercises _fit_oscillation)."""

    @pytest.fixture
    def model(self):
        m = HebraudLequeux()
        # Coarse grid so the multi-start SAOS optimizer stays fast.
        m.grid_n_bins = 21
        return m

    @pytest.mark.slow
    def test_oscillation_fit_and_predict_complex(self, model):
        omega = np.array([0.5, 2.0])
        # Synthetic G* with G' > G'' (solid-like), complex input path.
        G_star = np.array([1.0 + 0.2j, 1.3 + 0.4j])
        model.fit(
            omega,
            G_star,
            test_mode="oscillation",
            n_cycles=2,
            gamma0=0.01,
        )
        assert model.fitted_
        G_pred = model.predict(omega)
        assert G_pred.shape == omega.shape
        assert np.iscomplexobj(G_pred)
        assert np.all(np.isfinite(G_pred))
        # Storage modulus (real part) should be positive for a solid-like fit.
        assert np.all(np.real(G_pred) > 0)

    @pytest.mark.slow
    def test_oscillation_fit_accepts_m2_array(self, model):
        """(M, 2) [G', G''] input format is accepted by _fit_oscillation."""
        omega = np.array([0.5, 2.0])
        G_star = np.array([[1.0, 0.2], [1.3, 0.4]])
        model.fit(omega, G_star, test_mode="oscillation", n_cycles=2, gamma0=0.01)
        assert model.fitted_
        # Fitted parameters land inside their declared bounds.
        assert 0.01 <= model.parameters.get_value("alpha") <= 0.99
        assert model.parameters.get_value("sigma_c") > 0


class TestHLSteadyShear:
    """Steady-shear (flow-curve) fit path (_fit_steady_shear).

    The real HL PDE flow-curve solver is far too slow to run through the
    hardcoded 5-start x 200-eval Nelder-Mead loop inside a test timeout
    (the pre-existing slow ``test_flow_curve_prediction`` times out for the
    same reason). We stub ``run_flow_curve`` with a cheap monotonic surrogate
    so the fit/predict *orchestration* (normalization, multi-start loop,
    parameter clipping, schedule precompute, predict rescaling) is exercised
    without paying for the PDE integration.
    """

    def test_steady_shear_fit_and_predict(self, monkeypatch):
        import rheojax.models.hl.hebraud_lequeux as hl_mod

        def fake_flow_curve(
            gdot, alpha, tau, sigma_c, dt, sigma_max, n_bins, per_rate_schedule=None
        ):
            g = np.abs(np.asarray(gdot, dtype=np.float64))
            # Herschel-Bulkley-like surrogate: finite yield + sqrt-rate tail.
            return jnp.asarray(0.5 * sigma_c + 0.1 * np.sqrt(g))

        monkeypatch.setattr(hl_mod, "run_flow_curve", fake_flow_curve)

        model = HebraudLequeux()
        gdot = np.array([1.0, 3.16, 10.0])
        stress = np.array([0.6, 0.9, 1.6])
        model.fit(gdot, stress, test_mode="flow_curve")
        assert model.fitted_
        # Multi-start stored the normalization/schedule bookkeeping.
        assert "_stress_scale" in model._last_fit_kwargs
        assert "_precomputed_schedule" in model._last_fit_kwargs
        assert len(model._last_fit_kwargs["_precomputed_schedule"]) == len(gdot)

        pred = model.predict(gdot)
        assert pred.shape == gdot.shape
        assert np.all(np.isfinite(pred))
        # Surrogate is monotonic in shear rate; verifies predict rescaling.
        assert np.all(np.diff(pred) > 0)
        assert pred[0] > 0.0
        # Fitted params respect their declared bounds.
        assert 0.01 <= model.parameters.get_value("alpha") <= 0.99
        assert model.parameters.get_value("sigma_c") > 0

    def test_steady_shear_model_function(self, monkeypatch):
        """Bayesian flow-curve branch (normalized units) with stubbed kernel."""
        import rheojax.models.hl.hebraud_lequeux as hl_mod

        def fake_flow_curve(
            gdot, alpha, tau, sigma_c, dt, sigma_max, n_bins, per_rate_schedule=None
        ):
            g = np.abs(np.asarray(gdot, dtype=np.float64))
            return jnp.asarray(0.5 * sigma_c + 0.1 * np.sqrt(g))

        monkeypatch.setattr(hl_mod, "run_flow_curve", fake_flow_curve)

        model = HebraudLequeux()
        model._test_mode = "flow_curve"
        model._last_fit_kwargs = {
            "_stress_scale": 2.0,
            "_sigma_max_min_norm": 5.0,
            "_n_bins_fit": 51,
            "_tau_est": 1.0,
            "_sigma_c_est": 2.0,
            "_precomputed_schedule": None,
        }
        gdot = np.array([0.1, 1.0, 10.0])
        params = jnp.array([0.3, 1.0, 2.0])
        out = np.asarray(model.model_function(gdot, params))
        assert out.shape == gdot.shape
        assert np.all(np.isfinite(out))


class TestHLPhaseAndGrid:
    """Phase-state boundary and grid-parameter helpers."""

    def test_phase_state_boundary(self):
        model = HebraudLequeux()
        # alpha exactly 0.5 is the critical point: classified as fluid.
        model.parameters.set_value("alpha", 0.5)
        assert model.get_phase_state() == "fluid"
        model.parameters.set_value("alpha", 0.4999)
        assert model.get_phase_state() == "glass"

    def test_grid_params_floor(self):
        """sigma_max never drops below the 5.0 floor for small sigma_c."""
        model = HebraudLequeux()
        sigma_max, n_bins = model._get_grid_params(0.001)
        assert sigma_max == 5.0
        assert n_bins == model.grid_n_bins

    def test_grid_params_scales_with_sigma_c(self):
        model = HebraudLequeux()
        sigma_max, _ = model._get_grid_params(100.0)
        assert sigma_max == 100.0 * model.grid_sigma_factor

    def test_adaptive_dt_caps_scan_length(self):
        model = HebraudLequeux()
        dt, n_steps = model._adaptive_dt(1e6)
        # dt grows with t_max so n_steps stays within the scan cap.
        assert n_steps <= model._max_scan_steps + 1
        assert dt >= model._min_dt


class TestHLCreepWarmStart:
    """Regression for _fit_creep sigma_c warm-start (root-cause fix).

    Without warm-starting sigma_c from stress_target, the default
    sigma_c=1.0 leaves sigma_max~5.0 (via _get_grid_params), far below a
    physical stress_target, so the servo controller can never represent
    yielding near the true stress scale.
    """

    def test_creep_warm_starts_sigma_c_from_stress_target(self, monkeypatch):
        import rheojax.utils.optimization as opt_mod

        class _FakeResult:
            success = True
            message = "ok"

        def fake_nlsq_optimize(objective, parameters, **kwargs):
            return _FakeResult()

        monkeypatch.setattr(opt_mod, "nlsq_optimize", fake_nlsq_optimize)

        model = HebraudLequeux()  # default sigma_c = 1.0
        t = np.linspace(0, 1, 5)
        compliance = t * 0.01
        model._fit_creep(t, compliance, stress_target=50.0, max_iter=1)

        # sigma_c must be warm-started to the stress_target scale, not left
        # at the default (which would cap sigma_max at 5.0).
        assert model.parameters.get_value("sigma_c") == pytest.approx(50.0)
        assert model._last_fit_kwargs["_sigma_max"] >= 50.0

    def test_creep_does_not_override_explicit_sigma_c(self, monkeypatch):
        """Warm-start only kicks in for the untouched default (<=1.0)."""
        import rheojax.utils.optimization as opt_mod

        class _FakeResult:
            success = True
            message = "ok"

        def fake_nlsq_optimize(objective, parameters, **kwargs):
            return _FakeResult()

        monkeypatch.setattr(opt_mod, "nlsq_optimize", fake_nlsq_optimize)

        model = HebraudLequeux()
        model.parameters.set_value("sigma_c", 75.0)
        t = np.linspace(0, 1, 5)
        compliance = t * 0.01
        model._fit_creep(t, compliance, stress_target=50.0, max_iter=1)

        assert model.parameters.get_value("sigma_c") == pytest.approx(75.0)


class TestHLFlowCurveNormalization:
    """Regression for the flow-curve stress_scale epsilon-guard (root-cause fix).

    The normalization must use the *magnitude* of the low-rate stress, like
    every other epsilon-guard in the file, so a small negative low-rate
    stress sample doesn't collapse stress_scale to ~1e-12.
    """

    def test_negative_low_rate_stress_normalizes_by_magnitude(self, monkeypatch):
        import rheojax.models.hl.hebraud_lequeux as hl_mod

        def fake_flow_curve(
            gdot, alpha, tau, sigma_c, dt, sigma_max, n_bins, per_rate_schedule=None
        ):
            g = np.abs(np.asarray(gdot, dtype=np.float64))
            return jnp.asarray(0.5 * sigma_c + 0.1 * np.sqrt(g))

        monkeypatch.setattr(hl_mod, "run_flow_curve", fake_flow_curve)

        model = HebraudLequeux()
        gdot = np.array([-1e-3, 1.0, 3.16, 10.0])
        # Low-rate sample (nearest gdot=0) has a small negative stress, as
        # can happen from measurement noise near the yield point.
        stress = np.array([-0.05, 0.6, 0.9, 1.6])
        model.fit(gdot, stress, test_mode="flow_curve", max_iter=1)

        stress_scale = model._last_fit_kwargs["_stress_scale"]
        assert stress_scale == pytest.approx(0.05)
        # Without the fix stress_scale would collapse to 1e-12.
        assert stress_scale > 1e-6
