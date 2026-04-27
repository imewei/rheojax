"""Unit tests for FMLIKH (multi-layer FIKH) model.

Tests cover:
- Multi-mode initialization
- Per-mode parameter handling
- Shared vs per-mode fractional order
- Multi-mode predictions
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fikh import FMLIKH

jax, jnp = safe_import_jax()


class TestFMLIKHInitialization:
    """Test FMLIKH model initialization."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test default FMLIKH initialization."""
        model = FMLIKH()
        assert model.n_modes == 3
        assert model.shared_alpha is True
        assert model.include_thermal is True

    @pytest.mark.smoke
    def test_custom_n_modes(self):
        """Test FMLIKH with custom number of modes."""
        for n in [1, 2, 4, 5]:
            model = FMLIKH(n_modes=n, include_thermal=False)
            assert model.n_modes == n

    def test_invalid_n_modes(self):
        """Test FMLIKH rejects invalid n_modes."""
        with pytest.raises(ValueError):
            FMLIKH(n_modes=0)

        with pytest.raises(ValueError):
            FMLIKH(n_modes=-1)

    def test_isothermal_initialization(self):
        """Test isothermal FMLIKH."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        assert "T_ref" not in model.parameters

    def test_repr(self):
        """Test string representation."""
        model = FMLIKH(n_modes=3, include_thermal=False)
        repr_str = repr(model)
        assert "FMLIKH" in repr_str
        assert "n_modes=3" in repr_str


class TestFMLIKHParameters:
    """Test FMLIKH parameter handling."""

    @pytest.mark.smoke
    def test_per_mode_parameters_exist(self):
        """Test per-mode parameters are created."""
        model = FMLIKH(n_modes=3, include_thermal=False)

        for i in range(3):
            assert f"G_{i}" in model.parameters
            assert f"eta_{i}" in model.parameters
            assert f"C_{i}" in model.parameters
            assert f"gamma_dyn_{i}" in model.parameters

    def test_single_mode_params_removed(self):
        """Test single-mode G, eta, C, gamma_dyn are removed."""
        model = FMLIKH(n_modes=2, include_thermal=False)

        assert "G" not in model.parameters
        assert "eta" not in model.parameters
        assert "C" not in model.parameters
        assert "gamma_dyn" not in model.parameters

    def test_shared_alpha_single_parameter(self):
        """Test shared_alpha=True gives single alpha parameter."""
        model = FMLIKH(n_modes=3, shared_alpha=True, include_thermal=False)

        assert "alpha_structure" in model.parameters
        assert "alpha_0" not in model.parameters
        assert "alpha_1" not in model.parameters

    def test_per_mode_alpha_parameters(self):
        """Test shared_alpha=False gives per-mode alpha."""
        model = FMLIKH(n_modes=3, shared_alpha=False, include_thermal=False)

        assert "alpha_structure" not in model.parameters
        for i in range(3):
            assert f"alpha_{i}" in model.parameters

    def test_shared_params_preserved(self):
        """Test shared parameters are preserved."""
        model = FMLIKH(n_modes=2, include_thermal=False)

        shared_params = [
            "sigma_y0",
            "delta_sigma_y",
            "tau_thix",
            "Gamma",
            "eta_inf",
            "mu_p",
            "m",
        ]
        for param in shared_params:
            assert param in model.parameters, f"Missing shared param: {param}"

    def test_get_mode_params(self):
        """Test _get_mode_params extracts correct values."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        params = model._get_params_dict()

        mode_0_params = model._get_mode_params(params, 0)
        mode_1_params = model._get_mode_params(params, 1)

        # Check mode-specific params differ
        assert mode_0_params["G"] == params["G_0"]
        assert mode_1_params["G"] == params["G_1"]

        # Check shared params are same
        assert mode_0_params["sigma_y0"] == mode_1_params["sigma_y0"]


class TestFMLIKHPredictions:
    """Test FMLIKH model predictions."""

    @pytest.fixture
    def model(self):
        """Create 2-mode isothermal FMLIKH."""
        return FMLIKH(n_modes=2, include_thermal=False)

    @pytest.mark.smoke
    def test_startup_prediction(self, model):
        """Test startup prediction with multiple modes."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t
        stress = model._predict_from_params(t, strain, model._get_params_dict())

        assert stress.shape == t.shape
        assert jnp.isfinite(stress).all()

    @pytest.mark.smoke
    def test_flow_curve_prediction(self, model):
        """Test flow curve prediction with multiple modes."""
        gamma_dot = jnp.logspace(-2, 2, 20)
        stress = model._predict(gamma_dot, test_mode="flow_curve")

        assert stress.shape == gamma_dot.shape
        assert jnp.isfinite(stress).all()

    @pytest.mark.smoke
    def test_startup_fit_with_steady_state_warmstart(self):
        """FMLIKH startup fit must reach R^2 > 0.9 when warm-started.

        Regression for the NB08 calibration bug (2026-04-13): the factory
        default G_0=1e3, eta_0=1e6 makes the per-step elastic kick
        G*dt*gamma_dot ~ 1200 Pa dominate the yield stress on typical
        startup data (dt ~ 1 s, gamma_dot ~ 1 s^-1), trapping NLSQ in a
        basin where the fit oscillates (R^2 on the order of -1e4).
        Anchoring (sigma_y0, delta_sigma_y) to the measured steady-state
        stress and collapsing modal stiffnesses unlocks R^2 > 0.9.
        """
        # Small synthetic startup: monotonic rise to a ~25 Pa plateau.
        # dt=0.5 s x 40 points x n_sub=50 (since stable_dt=0.01) keeps
        # the dense grid at ~2000 pts — enough to exercise the
        # default-init trap without blowing the test budget.
        t = jnp.linspace(0.005, 20.0, 40)
        strain = 1.0 * t
        sigma_ss = 25.0
        # Simple saturating curve (no oscillation) — well within FMLIKH's
        # representational capacity on a 2-mode fit.
        stress_np = sigma_ss * (1.0 - np.exp(-np.asarray(t) / 2.0))
        stress = jnp.asarray(np.maximum(stress_np, 0.1))

        m = FMLIKH(n_modes=2, include_thermal=False, shared_alpha=True)
        sig_ss = float(jnp.mean(stress[-10:]))
        m.parameters.set_value("sigma_y0", 0.5 * sig_ss)
        m.parameters.set_value("delta_sigma_y", 0.5 * sig_ss)
        m.parameters.set_value("eta_inf", 0.1)
        m.parameters.set_value("mu_p", 0.01)
        for i, (G, eta, C) in enumerate([(1.0, 0.5, 0.5), (5.0, 2.0, 2.0)]):
            m.parameters.set_value(f"G_{i}", G)
            m.parameters.set_value(f"eta_{i}", eta)
            m.parameters.set_value(f"C_{i}", C)
            m.parameters.set_value(f"gamma_dyn_{i}", 0.1)

        # Default NLSQ trust-region (no method override) produces the best
        # R^2 on this scan-kernel protocol; method='scipy' trades ~0.06 R^2
        # for robustness on ODE-diffrax protocols but is unneeded here.
        m.fit(t, stress, test_mode="startup", strain=strain)
        pred = m.predict(t, test_mode="startup", strain=strain)

        ss_res = float(jnp.sum((stress - pred) ** 2))
        ss_tot = float(jnp.sum((stress - jnp.mean(stress)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.9, (
            f"FMLIKH startup warm-start regressed: R^2 = {r2:.3f} (expected > 0.9). "
            "Default-init startup fit is known to trap NLSQ at R^2 << 0."
        )

    @pytest.mark.smoke
    def test_flow_curve_fit_with_hb_warmstart(self):
        """FMLIKH flow-curve fit must reach R^2 > 0.9 when warm-started.

        Regression for the NB07 calibration bug (2026-04-13): fitting from
        the factory-default G_i = 1e3..10, eta_i = 1e6..1e4 onto a
        near-Newtonian shear-thinning dataset traps NLSQ in a flat
        sigma ~ 4 Pa plateau (R^2 ~ 0.43). Warm-starting from a
        Herschel-Bulkley fit (sigma = sigma_y + K*gamma_dot^n) unlocks
        the correct basin of attraction and should land R^2 >> 0.9.
        """
        from scipy.optimize import curve_fit

        # Synthetic near-Newtonian HB data (mimics the NB07 dataset shape
        # without shipping external data into the unit test).
        gamma_dot = jnp.logspace(-2, 2, 21)
        sigma_y_true, K_true, n_true = 0.9, 0.8, 0.9
        rng = np.random.default_rng(42)
        stress = (
            sigma_y_true
            + K_true * gamma_dot**n_true
            + 0.02 * rng.standard_normal(gamma_dot.shape)
        )
        # Clip to positive to avoid log issues in the objective.
        stress = jnp.asarray(np.maximum(np.asarray(stress), 0.1))

        # HB warm start
        hb = lambda g, sy, K, n: sy + K * g**n  # noqa: E731
        popt, _ = curve_fit(
            hb,
            np.asarray(gamma_dot),
            np.asarray(stress),
            p0=[0.5, 4.0, 0.5],
            maxfev=5000,
        )
        sy_hb, K_hb, _ = popt

        m = FMLIKH(n_modes=3, include_thermal=False, shared_alpha=True)
        m.parameters.set_value("sigma_y0", float(sy_hb))
        m.parameters.set_value("delta_sigma_y", 0.1)
        m.parameters.set_value("eta_inf", float(K_hb))
        m.parameters.set_value("mu_p", 0.01)
        for i, (G, eta, C) in enumerate(
            [(1.0, 0.1, 0.5), (5.0, 0.5, 2.0), (10.0, 2.0, 5.0)]
        ):
            m.parameters.set_value(f"G_{i}", G)
            m.parameters.set_value(f"eta_{i}", eta)
            m.parameters.set_value(f"C_{i}", C)
            m.parameters.set_value(f"gamma_dyn_{i}", 0.1)

        m.fit(gamma_dot, stress, test_mode="flow_curve")
        pred = m.predict(gamma_dot, test_mode="flow_curve")

        # R^2 against the mean baseline.
        ss_res = float(jnp.sum((stress - pred) ** 2))
        ss_tot = float(jnp.sum((stress - jnp.mean(stress)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        assert r2 > 0.9, (
            f"FMLIKH HB warm-start regressed: R^2 = {r2:.3f} (expected > 0.9). "
            "Default-init flow-curve fit is known to trap NLSQ at R^2 ~ 0.43."
        )

    def test_more_modes_gives_different_response(self):
        """Test that more modes changes the response."""
        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        model_2 = FMLIKH(n_modes=2, include_thermal=False, shared_alpha=True)
        model_4 = FMLIKH(n_modes=4, include_thermal=False, shared_alpha=True)

        stress_2 = model_2._predict_from_params(t, strain, model_2._get_params_dict())
        stress_4 = model_4._predict_from_params(t, strain, model_4._get_params_dict())

        # Responses should be different (default parameters differ)
        assert not jnp.allclose(stress_2, stress_4)

    def test_single_mode_matches_fikh(self):
        """Test single-mode FMLIKH is similar to FIKH."""
        from rheojax.models.fikh import FIKH

        t = jnp.linspace(0, 10, 100)
        strain = 0.01 * t

        # FIKH
        fikh = FIKH(include_thermal=False, alpha_structure=0.5)

        # FMLIKH with 1 mode and matching parameters
        fmlikh = FMLIKH(n_modes=1, include_thermal=False, alpha_structure=0.5)

        # Set FMLIKH params to match FIKH
        fmlikh_params = fmlikh._get_params_dict()
        fikh_params = fikh._get_params_dict()

        # Update FMLIKH mode-0 params to match FIKH
        fmlikh_params["G_0"] = fikh_params["G"]
        fmlikh_params["eta_0"] = fikh_params["eta"]
        fmlikh_params["C_0"] = fikh_params["C"]
        fmlikh_params["gamma_dyn_0"] = fikh_params["gamma_dyn"]

        stress_fikh = fikh._predict_from_params(t, strain, fikh_params)
        stress_fmlikh = fmlikh._predict_from_params(t, strain, fmlikh_params)

        # Should be identical (or very close)
        assert jnp.allclose(stress_fikh, stress_fmlikh, rtol=1e-4)


class TestFMLIKHModeInfo:
    """Test FMLIKH mode information utilities."""

    def test_get_mode_info(self):
        """Test get_mode_info returns correct structure."""
        model = FMLIKH(n_modes=3, include_thermal=False, shared_alpha=True)
        info = model.get_mode_info()

        assert info["n_modes"] == 3
        assert info["shared_alpha"] is True
        assert len(info["modes"]) == 3

        for i, mode in enumerate(info["modes"]):
            assert mode["mode"] == i
            assert "G" in mode
            assert "eta" in mode
            assert "tau" in mode
            assert "C" in mode

    def test_get_mode_info_per_mode_alpha(self):
        """Test get_mode_info with per-mode alpha."""
        model = FMLIKH(n_modes=2, include_thermal=False, shared_alpha=False)
        info = model.get_mode_info()

        assert info["shared_alpha"] is False
        assert "alpha_shared" not in info

        for mode in info["modes"]:
            assert "alpha" in mode


class TestFMLIKHModelFunction:
    """Test FMLIKH model_function for Bayesian inference."""

    def test_model_function_startup(self):
        """Test model_function for startup."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        model._test_mode = "startup"

        t = jnp.linspace(0, 5, 50)
        strain = 0.01 * t
        X = jnp.stack([t, strain], axis=0)

        params = model._get_params_dict()
        result = model.model_function(X, params)

        assert result.shape == t.shape

    def test_model_function_flow_curve(self):
        """Test model_function for flow curve."""
        model = FMLIKH(n_modes=2, include_thermal=False)
        model._test_mode = "flow_curve"

        gamma_dot = jnp.logspace(-2, 2, 20)
        params = model._get_params_dict()

        result = model.model_function(gamma_dot, params)
        assert result.shape == gamma_dot.shape
