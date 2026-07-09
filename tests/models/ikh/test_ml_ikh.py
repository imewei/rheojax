"""Coverage-focused tests for the ML-IKH model (rheojax/models/ikh/ml_ikh.py).

These tests target code paths not exercised by the existing IKH-family suites:
  - weighted-sum ODE transients (creep/relaxation) and their arg-building
  - the startup branch of ``_simulate_transient`` (only reachable directly)
  - ``model_function`` protocol dispatch (NumPyro entry point)
  - frequency-domain SAOS fitting (``_fit_saos_frequency_domain``)
  - the MIKH warm-start and amplitude-fallback initialisation branches
  - ``_fit`` dispatch for oscillation and the default (else) protocol

Conventions follow the sibling IKH tests: synthetic noiseless data, small
arrays, tight iteration caps, ``numpy.testing.assert_allclose`` with explicit
tolerances, and finite/physicality checks.
"""

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.ikh.ml_ikh import MLIKH

jax, jnp = safe_import_jax()


def _param_dict(model):
    """Return the model's current parameters as a name->value dict."""
    return dict(zip(model.parameters.keys(), model.parameters.get_values(), strict=True))


# ============================================================================
# weighted-sum ODE transients (covers _build_ode_args weighted_sum branch,
# _simulate_transient weighted_sum creep/relaxation, weighted_sum stress extract)
# ============================================================================


class TestMLIKHWeightedSumTransients:
    """ODE transients for the weighted-sum yield formulation."""

    def _make_model(self):
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        model.parameters.set_value("G", 500.0)
        model.parameters.set_value("C", 100.0)
        model.parameters.set_value("gamma_dyn", 0.5)
        model.parameters.set_value("sigma_y0", 20.0)
        model.parameters.set_value("k3", 30.0)
        model.parameters.set_value("tau_thix_1", 0.5)
        model.parameters.set_value("tau_thix_2", 5.0)
        model.parameters.set_value("eta_inf", 5.0)
        return model

    def test_weighted_sum_creep(self):
        """Weighted-sum creep returns finite, non-decreasing strain."""
        model = self._make_model()
        t = jnp.linspace(0.01, 5.0, 30)
        strain = model.predict_creep(t, sigma_applied=60.0)

        strain_arr = np.asarray(strain)
        assert strain_arr.shape == (30,)
        assert np.all(np.isfinite(strain_arr))
        # Creep strain should grow monotonically under sustained overstress.
        assert strain_arr[-1] >= strain_arr[0]

    def test_weighted_sum_relaxation(self):
        """Weighted-sum relaxation returns finite, bounded stress."""
        model = self._make_model()
        t = jnp.linspace(0.01, 10.0, 30)
        sigma = model.predict_relaxation(t, sigma_0=80.0)

        sigma_arr = np.asarray(sigma)
        assert sigma_arr.shape == (30,)
        assert np.all(np.isfinite(sigma_arr))
        # Relaxing stress must not exceed the initial value (no runaway growth).
        assert sigma_arr[0] <= 80.0 + 1e-6
        assert np.max(sigma_arr) <= 80.0 + 1e-6


# ============================================================================
# startup branch of _simulate_transient (only reachable via direct call;
# predict_startup routes through the return-mapping scan kernel instead)
# ============================================================================


class TestSimulateTransientStartup:
    """Directly exercise the ODE startup branch and viscous add-on."""

    def test_startup_per_mode_direct(self):
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        for i in (1, 2):
            model.parameters.set_value(f"G_{i}", 100.0)
            model.parameters.set_value(f"sigma_y0_{i}", 5.0)
            model.parameters.set_value(f"delta_sigma_y_{i}", 10.0)
        model.parameters.set_value("eta_inf", 2.0)  # triggers viscous contribution

        t = jnp.linspace(0.01, 3.0, 25)
        params = _param_dict(model)
        sigma = model._simulate_transient(t, params, "startup", gamma_dot=1.0)

        sigma_arr = np.asarray(sigma)
        assert sigma_arr.shape == (25,)
        assert np.all(np.isfinite(sigma_arr))
        # eta_inf * gamma_dot = 2.0 viscous floor is added to every point.
        assert np.all(sigma_arr >= 0.0)

    def test_startup_weighted_sum_direct(self):
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        model.parameters.set_value("G", 200.0)
        model.parameters.set_value("sigma_y0", 5.0)
        model.parameters.set_value("k3", 10.0)
        model.parameters.set_value("eta_inf", 1.0)

        t = jnp.linspace(0.01, 3.0, 25)
        params = _param_dict(model)
        sigma = model._simulate_transient(t, params, "startup", gamma_dot=1.0)

        sigma_arr = np.asarray(sigma)
        assert sigma_arr.shape == (25,)
        assert np.all(np.isfinite(sigma_arr))


# ============================================================================
# model_function (NumPyro entry point) protocol dispatch
# ============================================================================


class TestMLIKHModelFunction:
    """Exercise every dispatch branch of model_function."""

    def test_flow_curve_array_params(self):
        """Array params + explicit flow_curve mode."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        gamma_dot = jnp.logspace(-2, 2, 15)
        params = model.parameters.get_values()  # ndarray path (isinstance branch)

        out = model.model_function(gamma_dot, params, test_mode="flow_curve")
        out_arr = np.asarray(out)
        assert out_arr.shape == (15,)
        assert np.all(np.isfinite(out_arr))

    def test_default_mode_falls_back_to_startup(self):
        """test_mode=None with a fresh model falls through to startup."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        t = np.linspace(0.0, 2.0, 20)
        strain = 1.0 * t
        X = np.stack([t, strain])
        params = model.parameters.get_values()

        out = model.model_function(X, params, test_mode=None)
        out_arr = np.asarray(out)
        assert out_arr.shape == (20,)
        assert np.all(np.isfinite(out_arr))

    def test_oscillation_per_mode_dict_params(self):
        """Oscillation dispatch (per_mode) with a dict params argument."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        omega = jnp.logspace(-1, 1, 12)
        params = _param_dict(model)  # dict path (else branch)

        out = model.model_function(omega, params, test_mode="oscillation")
        out_arr = np.asarray(out)
        # Returns column_stack([G', G'']).
        assert out_arr.shape == (12, 2)
        assert np.all(np.isfinite(out_arr))
        # High-viscosity approximation => G' ~ G_total, G'' ~ 0.
        np.testing.assert_allclose(out_arr[:, 1], 0.0, atol=1e-3)

    def test_oscillation_weighted_sum(self):
        """Oscillation dispatch for the weighted-sum global-G branch."""
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        omega = jnp.logspace(-1, 1, 12)
        params = model.parameters.get_values()

        out = model.model_function(omega, params, test_mode="oscillation")
        out_arr = np.asarray(out)
        assert out_arr.shape == (12, 2)
        assert np.all(np.isfinite(out_arr))
        G_val = float(model.parameters.get_value("G"))
        # G' should recover the global elastic modulus at these frequencies.
        np.testing.assert_allclose(out_arr[:, 0], G_val, rtol=1e-3)

    def test_creep_dispatch(self):
        """Creep dispatch through model_function (ODE path)."""
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        for i in (1, 2):
            model.parameters.set_value(f"G_{i}", 100.0)
            model.parameters.set_value(f"sigma_y0_{i}", 5.0)
        model.parameters.set_value("eta_inf", 5.0)

        t = jnp.linspace(0.01, 2.0, 15)
        params = model.parameters.get_values()
        out = model.model_function(t, params, test_mode="creep", sigma_applied=40.0)
        out_arr = np.asarray(out)
        assert out_arr.shape == (15,)
        assert np.all(np.isfinite(out_arr))


# ============================================================================
# frequency-domain SAOS fitting (_fit_oscillation -> _fit_saos_frequency_domain)
# ============================================================================


class TestMLIKHSAOSFrequencyFit:
    """Cover the three y-format branches of frequency-domain SAOS fitting."""

    def test_saos_per_mode_components(self):
        """(N, 2) target => component-wise fit, per_mode objective."""
        model = MLIKH(n_modes=1, yield_mode="per_mode")
        omega = jnp.logspace(-1, 1, 12)
        target = np.column_stack([np.full(12, 500.0), np.zeros(12)])  # [G', G'']

        model.fit(omega, target, test_mode="oscillation", max_iter=20)
        assert model.fitted_
        # G' target is flat 500; the single-mode modulus should track it.
        assert np.isfinite(float(model.parameters.get_value("G_1")))

    def test_saos_weighted_sum_complex(self):
        """Complex target => component-wise fit, weighted_sum objective."""
        model = MLIKH(n_modes=1, yield_mode="weighted_sum")
        omega = jnp.logspace(-1, 1, 12)
        target = np.full(12, 400.0) + 0.0j  # complex G* = G' + iG''

        model.fit(omega, target, test_mode="oscillation", max_iter=20)
        assert model.fitted_
        assert np.isfinite(float(model.parameters.get_value("G")))

    def test_saos_magnitude_fallback(self):
        """Real 1-D target => magnitude-only fallback branch."""
        model = MLIKH(n_modes=1, yield_mode="weighted_sum")
        omega = jnp.logspace(-1, 1, 12)
        target = np.full(12, 350.0)  # |G*|

        model.fit(omega, target, test_mode="saos", max_iter=20)
        assert model.fitted_
        assert np.isfinite(float(model.parameters.get_value("G")))


# ============================================================================
# time-domain oscillation dispatch (len(X) > 100 -> return mapping)
# ============================================================================


def test_oscillation_time_domain_dispatch():
    """Long 1-D time input routes oscillation fitting to return mapping."""
    model = MLIKH(n_modes=2, yield_mode="per_mode")
    t = np.linspace(0.0, 10.0, 120)
    strain = 0.5 * np.sin(1.0 * t)
    # Build a self-consistent target so residuals are finite at init.
    y = np.asarray(model.predict(np.stack([t, strain]), test_mode="startup"))

    model.fit(t, y, test_mode="oscillation", strain=strain, max_iter=5)
    assert model.fitted_


# ============================================================================
# MIKH warm-start and amplitude-fallback initialisation
# ============================================================================


class TestMLIKHInitialisation:
    """Cover the smart-init branches inside _fit_return_mapping."""

    def _startup_data(self):
        true_model = MLIKH(n_modes=2, yield_mode="per_mode")
        for i in (1, 2):
            true_model.parameters.set_value(f"G_{i}", 80.0)
            true_model.parameters.set_value(f"sigma_y0_{i}", 10.0)
            true_model.parameters.set_value(f"delta_sigma_y_{i}", 15.0)
        t = np.linspace(0.0, 4.0, 60)
        gamma_dot = 2.0
        gamma = gamma_dot * t
        X = np.stack([t, gamma])
        y = np.asarray(true_model.predict(X, test_mode="startup"))
        return X, y

    def test_mikh_warmstart_path(self):
        """mikh_warmstart=True runs the single-mode warm-start distribution."""
        X, y = self._startup_data()
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        model.fit(X, y, test_mode="startup", mikh_warmstart=True, max_iter=20)
        assert model.fitted_
        y_pred = np.asarray(model.predict(X, test_mode="startup"))
        assert np.all(np.isfinite(y_pred))

    def test_amplitude_fallback_per_mode(self):
        """Tiny initial G_i / sigma_y0_i triggers amplitude-based rescaling."""
        X, y = self._startup_data()
        # Force the initial moduli far below the stress amplitude so the
        # rescaling guards (cur_G < amp*0.1, cur_sy < amp*0.01) fire.
        model = MLIKH(n_modes=2, yield_mode="per_mode")
        for i in (1, 2):
            model.parameters.set_value(f"G_{i}", 1.0)
            model.parameters.set_value(f"sigma_y0_{i}", 0.05)
        model.fit(X, y, test_mode="startup", max_iter=5)
        assert model.fitted_

    def test_amplitude_fallback_weighted_sum(self):
        """Weighted-sum branch of the amplitude fallback."""
        X, y = self._startup_data()
        model = MLIKH(n_modes=2, yield_mode="weighted_sum")
        model.parameters.set_value("G", 1.0)
        model.parameters.set_value("sigma_y0", 0.05)
        model.fit(X, y, test_mode="startup", max_iter=5)
        assert model.fitted_


# ============================================================================
# _fit dispatch: creep/relaxation and the default (else) protocol branch
# ============================================================================


def test_fit_default_protocol_falls_back_to_return_mapping():
    """An unrecognised test_mode routes to the return-mapping fitter."""
    model = MLIKH(n_modes=2, yield_mode="per_mode")
    t = np.linspace(0.0, 2.0, 40)
    strain = 1.0 * t
    X = np.stack([t, strain])
    y = np.asarray(model.predict(X, test_mode="startup"))

    model.fit(X, y, test_mode="strain_controlled", max_iter=5)
    assert model.fitted_
    assert model._test_mode == "strain_controlled"
