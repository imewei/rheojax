"""Tests for rheojax.utils.protocol_preprocessing."""

import numpy as np
import pytest

from rheojax.utils.protocol_preprocessing import (
    PreprocessingResult,
    check_kramers_kronig,
    estimate_eta0,
    fit_gel_point,
    preprocess_for_protocol,
)


@pytest.mark.smoke
class TestCheckKramersKronig:
    """Tests for Kramers-Kronig consistency check."""

    def test_consistent_data_passes(self):
        """Power-law data should pass KK test."""
        omega = np.logspace(-2, 2, 50)
        # G' ~ omega^0.5, G'' ~ omega^0.5 (gel-like, slope < 2)
        G_prime = 1000.0 * omega**0.5
        G_double_prime = 500.0 * omega**0.5
        passes, max_slope = check_kramers_kronig(omega, G_prime, G_double_prime)
        assert isinstance(passes, bool)
        assert isinstance(max_slope, float)

    def test_steep_data_fails(self):
        """Very steep data (slope > tolerance) should fail."""
        omega = np.logspace(-2, 2, 50)
        # G' ~ omega^3, slope = 3 > 2.5 default tolerance
        G_prime = omega**3
        G_double_prime = omega**2
        passes, max_slope = check_kramers_kronig(omega, G_prime, G_double_prime)
        assert max_slope > 2.5

    def test_custom_tolerance(self):
        omega = np.logspace(-2, 2, 50)
        G_prime = omega**1.5
        G_double_prime = omega**1.0
        passes, _ = check_kramers_kronig(omega, G_prime, G_double_prime, tolerance=1.0)
        # slope ~1.5 > tolerance 1.0
        assert passes is False


@pytest.mark.smoke
class TestEstimateEta0:
    """Tests for zero-shear viscosity estimation."""

    def test_newtonian_fluid(self):
        gamma_dot = np.logspace(-3, 3, 50)
        eta = np.full_like(gamma_dot, 100.0)
        eta0 = estimate_eta0(gamma_dot, eta=eta)
        np.testing.assert_allclose(eta0, 100.0, rtol=0.01)

    def test_from_sigma(self):
        gamma_dot = np.logspace(-3, 3, 50)
        eta_val = 100.0
        sigma = eta_val * gamma_dot
        eta0 = estimate_eta0(gamma_dot, sigma=sigma)
        np.testing.assert_allclose(eta0, eta_val, rtol=0.1)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            estimate_eta0(np.array([]))

    def test_no_eta_no_sigma_raises(self):
        with pytest.raises(ValueError, match="Either eta or sigma"):
            estimate_eta0(np.array([1.0, 2.0]))


@pytest.mark.smoke
class TestFitGelPoint:
    """Tests for gel-point fitting."""

    def test_known_power_law(self):
        """G(t) = S * t^(-n) with known S and n."""
        S_true = 500.0
        n_true = 0.5
        t = np.logspace(-2, 2, 50)
        G_t = S_true * t ** (-n_true)
        S, n = fit_gel_point(t, G_t)
        np.testing.assert_allclose(S, S_true, rtol=0.01)
        np.testing.assert_allclose(n, n_true, rtol=0.01)

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="At least two"):
            fit_gel_point(np.array([1.0]), np.array([100.0]))

    def test_negative_values_filtered(self):
        t = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        G = np.array([-5.0, 0.0, 100.0, 50.0, 33.3])
        S, n = fit_gel_point(t, G)
        assert S > 0
        assert n > 0


@pytest.mark.smoke
class TestPreprocessingResult:
    """Test PreprocessingResult dataclass."""

    def test_construction(self):
        result = PreprocessingResult(
            X=np.array([1.0, 2.0]),
            y=np.array([3.0, 4.0]),
        )
        assert len(result.X) == 2
        assert len(result.warnings) == 0
        assert len(result.applied) == 0


# ---------------------------------------------------------------------------
# Additional standalone-function edge cases
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestCheckKramersKronigEdges:
    """Edge cases for the KK slope test."""

    def test_duplicate_frequencies_returns_true_zero(self):
        """All-duplicate omega -> no valid slopes -> (True, 0.0)."""
        omega = np.full(10, 5.0)
        G_prime = np.full(10, 100.0)
        G_double_prime = np.full(10, 50.0)
        passes, max_slope = check_kramers_kronig(omega, G_prime, G_double_prime)
        assert passes is True
        assert max_slope == 0.0

    def test_scale_invariance_of_slope(self):
        """Log-log slope is invariant under overall amplitude scaling of G'."""
        omega = np.logspace(-2, 2, 40)
        G_prime = omega**1.2
        G_double_prime = omega**1.0
        _, slope_a = check_kramers_kronig(omega, G_prime, G_double_prime)
        _, slope_b = check_kramers_kronig(omega, 1000.0 * G_prime, G_double_prime)
        np.testing.assert_allclose(slope_a, slope_b, rtol=1e-9)

    def test_nonpositive_values_clamped_no_nan(self):
        """Zero/negative moduli are clamped, not turned into NaN/Inf."""
        omega = np.logspace(-2, 2, 20)
        G_prime = np.zeros_like(omega)
        G_double_prime = np.zeros_like(omega)
        passes, max_slope = check_kramers_kronig(omega, G_prime, G_double_prime)
        assert np.isfinite(max_slope)
        assert isinstance(passes, bool)


@pytest.mark.smoke
class TestEstimateEta0Edges:
    """Edge cases for zero-shear viscosity estimation."""

    def test_eta_takes_precedence_over_sigma(self):
        """When both eta and sigma given, eta is used."""
        gamma_dot = np.logspace(-3, 1, 30)
        eta = np.full_like(gamma_dot, 42.0)
        sigma = 999.0 * gamma_dot  # would give eta=999 if used
        eta0 = estimate_eta0(gamma_dot, eta=eta, sigma=sigma)
        np.testing.assert_allclose(eta0, 42.0, rtol=1e-9)

    def test_single_point_uses_that_point(self):
        """n_low clamps to array length for tiny inputs."""
        eta0 = estimate_eta0(np.array([2.0]), eta=np.array([7.0]))
        np.testing.assert_allclose(eta0, 7.0, rtol=1e-9)

    def test_unsorted_input_sorted_internally(self):
        """Lowest shear rates selected regardless of input order."""
        gamma_dot = np.array([10.0, 0.1, 5.0, 0.2, 0.3])
        eta = np.array([1.0, 100.0, 2.0, 90.0, 80.0])  # high eta at low gamma
        eta0 = estimate_eta0(gamma_dot, eta=eta)
        # lowest-3 gamma (0.1,0.2,0.3) -> eta (100,90,80) -> median 90
        np.testing.assert_allclose(eta0, 90.0, rtol=1e-9)


@pytest.mark.smoke
class TestFitGelPointEdges:
    """Edge cases for gel-point fitting."""

    def test_recovers_various_exponents(self):
        t = np.logspace(-1, 2, 40)
        for S_true, n_true in [(10.0, 0.2), (250.0, 0.75)]:
            G_t = S_true * t ** (-n_true)
            S, n = fit_gel_point(t, G_t)
            np.testing.assert_allclose(S, S_true, rtol=1e-6)
            np.testing.assert_allclose(n, n_true, rtol=1e-6)

    def test_all_nonpositive_raises(self):
        with pytest.raises(ValueError, match="At least two"):
            fit_gel_point(np.array([-1.0, 0.0]), np.array([-3.0, 0.0]))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessForProtocol:
    """Tests for the preprocess_for_protocol dispatcher."""

    def test_unknown_mode_passthrough(self):
        """Unknown test_mode returns data unchanged, no diagnostics."""
        X = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = preprocess_for_protocol(X, y, "not_a_protocol")
        np.testing.assert_array_equal(result.X, X)
        np.testing.assert_array_equal(result.y, y)
        assert result.diagnostics == {}
        assert result.warnings == []

    @pytest.mark.parametrize(
        "mode",
        ["relaxation", "creep", "oscillation", "flow_curve", "startup", "laos"],
    )
    def test_all_modes_dispatch(self, mode):
        """Each known mode returns a PreprocessingResult without raising."""
        x = np.linspace(0.1, 10.0, 40)
        if mode == "oscillation":
            y = (1000.0 * x**0.5) + 1j * (500.0 * x**0.5)
        else:
            y = np.exp(-x) + 1.0
        result = preprocess_for_protocol(x, y, mode)
        assert isinstance(result, PreprocessingResult)
        assert result.X.shape[0] == x.shape[0]

    def test_input_coerced_to_float(self):
        """Integer X is coerced to float64."""
        result = preprocess_for_protocol(
            np.arange(30), np.ones(30), "not_a_protocol"
        )
        assert result.X.dtype == np.float64


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessRelaxation:
    """Tests for _preprocess_relaxation via the dispatcher."""

    def test_clean_decay_no_warnings(self):
        """Smooth monotonic decay: no ringing, strong decay."""
        t = np.linspace(0.01, 10.0, 100)
        G_t = 1000.0 * np.exp(-t / 0.5)
        result = preprocess_for_protocol(t, G_t, "relaxation")
        assert result.diagnostics["short_time_sign_changes"] == 0
        # strong decay -> late/early ratio tiny, no weak-decay warning
        assert result.diagnostics["late_to_early_ratio"] < 0.5
        assert result.applied == []

    def test_short_array_skips_ringing_and_plateau(self):
        """len(t) <= 10 skips ringing block; <= 20 skips plateau block."""
        t = np.linspace(0.0, 1.0, 8)
        G_t = np.exp(-t)
        result = preprocess_for_protocol(t, G_t, "relaxation")
        assert "short_time_sign_changes" not in result.diagnostics
        assert "late_to_early_ratio" not in result.diagnostics

    def test_ringing_detected_and_warned(self):
        """Oscillatory short-time data triggers ringing warning + diagnostics."""
        t = np.linspace(0.0, 5.0, 60)
        # Strong oscillation in the first 20 points, then decay
        base = 1000.0 * np.exp(-t)
        ripple = np.zeros_like(t)
        ripple[:20] = 300.0 * np.sin(np.arange(20) * 2.0)
        G_t = base + ripple
        result = preprocess_for_protocol(t, G_t, "relaxation")
        assert result.diagnostics["short_time_sign_changes"] > 3
        assert "ringing_end_idx" in result.diagnostics
        assert any("ringing" in w for w in result.warnings)
        # Diagnostics only: data untouched without apply_cutoff
        assert result.applied == []
        assert result.X.shape[0] == t.shape[0]

    def test_ringing_cutoff_applied(self):
        """apply_cutoff=True trims the ringing region."""
        t = np.linspace(0.0, 5.0, 60)
        base = 1000.0 * np.exp(-t)
        ripple = np.zeros_like(t)
        ripple[:20] = 300.0 * np.sin(np.arange(20) * 2.0)
        G_t = base + ripple
        result = preprocess_for_protocol(t, G_t, "relaxation", apply_cutoff=True)
        assert "cutoff_applied_idx" in result.diagnostics
        assert any("cutoff" in a for a in result.applied)
        assert result.X.shape[0] < t.shape[0]

    def test_weak_decay_warns(self):
        """G(t) that barely decays triggers weak-decay warning."""
        t = np.linspace(0.01, 10.0, 100)
        G_t = 1000.0 - 5.0 * t  # ends at ~0.95 of start, monotone, no ringing
        result = preprocess_for_protocol(t, G_t, "relaxation")
        assert result.diagnostics["late_to_early_ratio"] > 0.5
        assert any("weak" in w for w in result.warnings)

    def test_complex_modulus_uses_real_part(self):
        """Complex G_t: real part drives diagnostics, output stays complex."""
        t = np.linspace(0.01, 10.0, 100)
        G_t = (1000.0 * np.exp(-t)) + 1j * np.full_like(t, 10.0)
        result = preprocess_for_protocol(t, G_t, "relaxation")
        assert np.iscomplexobj(result.y)
        assert np.isfinite(result.diagnostics["late_to_early_ratio"])


# ---------------------------------------------------------------------------
# Creep
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessCreep:
    """Tests for _preprocess_creep via the dispatcher."""

    def test_viscous_flow_detected(self):
        """Linearly growing J(t) at long times flags viscous flow."""
        t = np.linspace(0.0, 10.0, 100)
        J_t = 0.1 + 2.0 * t  # strong late-time linear growth
        result = preprocess_for_protocol(t, J_t, "creep")
        assert result.diagnostics["has_viscous_flow"] is True
        assert result.diagnostics["late_slope"] > 0

    def test_plateau_no_viscous_flow(self):
        """Saturating compliance does not flag viscous flow."""
        t = np.linspace(0.0, 10.0, 100)
        J_t = 5.0 * (1.0 - np.exp(-t))  # plateaus at 5
        result = preprocess_for_protocol(t, J_t, "creep")
        assert result.diagnostics["has_viscous_flow"] is False

    def test_nonmonotonic_warns(self):
        """Many decreasing steps trigger the non-monotonic warning."""
        rng = np.random.default_rng(0)
        t = np.linspace(0.0, 10.0, 100)
        J_t = t + rng.normal(0.0, 3.0, size=t.shape)  # noisy -> many dips
        result = preprocess_for_protocol(t, J_t, "creep")
        assert result.diagnostics["n_decreasing_steps"] > 0
        if result.diagnostics["n_decreasing_steps"] > (len(t) - 1) * 0.1:
            assert any("Non-monotonic" in w for w in result.warnings)

    def test_short_array_no_flow_diagnostics(self):
        """len(t) <= 20 skips the viscous-flow block but still checks monotonicity."""
        t = np.linspace(0.0, 1.0, 10)
        J_t = np.linspace(0.0, 1.0, 10)
        result = preprocess_for_protocol(t, J_t, "creep")
        assert "late_slope" not in result.diagnostics
        assert "n_decreasing_steps" in result.diagnostics

    def test_complex_compliance_uses_real_part(self):
        t = np.linspace(0.0, 10.0, 100)
        J_t = (0.1 + 2.0 * t) + 1j * np.ones_like(t)
        result = preprocess_for_protocol(t, J_t, "creep")
        assert np.iscomplexobj(result.y)
        assert np.isfinite(result.diagnostics["late_slope"])


# ---------------------------------------------------------------------------
# Oscillation
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessOscillation:
    """Tests for _preprocess_oscillation via the dispatcher."""

    def test_complex_gstar_kk_and_tandelta(self):
        """Complex G* yields KK diagnostics and tan(delta) range."""
        omega = np.logspace(-2, 2, 50)
        G_star = (1000.0 * omega**0.5) + 1j * (500.0 * omega**0.5)
        result = preprocess_for_protocol(omega, G_star, "oscillation")
        assert "kk_passes" in result.diagnostics
        assert "max_log_slope_G_prime" in result.diagnostics
        lo, hi = result.diagnostics["tan_delta_range"]
        # tan(delta) = G''/G' = 0.5 constant here
        np.testing.assert_allclose(lo, 0.5, rtol=1e-6)
        np.testing.assert_allclose(hi, 0.5, rtol=1e-6)

    def test_two_column_gstar(self):
        """(N,2) real array: column 0 = G', column 1 = G''."""
        omega = np.logspace(-2, 2, 50)
        G_star = np.column_stack([1000.0 * omega**0.5, 500.0 * omega**0.5])
        result = preprocess_for_protocol(omega, G_star, "oscillation")
        assert "kk_passes" in result.diagnostics
        lo, hi = result.diagnostics["tan_delta_range"]
        np.testing.assert_allclose([lo, hi], [0.5, 0.5], rtol=1e-6)

    def test_kk_violation_warns(self):
        """Steep G' triggers the KK-consistency warning."""
        omega = np.logspace(-2, 2, 50)
        G_star = (omega**3) + 1j * (omega**2)
        result = preprocess_for_protocol(omega, G_star, "oscillation")
        assert result.diagnostics["kk_passes"] is False
        assert any("Kramers-Kronig" in w for w in result.warnings)

    def test_ambiguous_shape_early_return(self):
        """1-D real G* that isn't complex/(N,2) returns with empty diagnostics."""
        omega = np.logspace(-2, 2, 50)
        G_star = 1000.0 * omega**0.5  # plain 1-D real
        result = preprocess_for_protocol(omega, G_star, "oscillation")
        assert result.diagnostics == {}
        np.testing.assert_array_equal(result.y, G_star)

    def test_short_array_skips_kk_keeps_tandelta(self):
        """len(omega) <= 5 skips KK block but still reports tan(delta)."""
        omega = np.logspace(-2, 0, 4)
        G_star = (100.0 * omega) + 1j * (50.0 * omega)
        result = preprocess_for_protocol(omega, G_star, "oscillation")
        assert "kk_passes" not in result.diagnostics
        assert "tan_delta_range" in result.diagnostics


# ---------------------------------------------------------------------------
# Flow curve
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessFlowCurve:
    """Tests for _preprocess_flow_curve via the dispatcher."""

    def test_yield_stress_detected(self):
        """Low-rate stress plateau flags a yield stress."""
        gamma_dot = np.logspace(-2, 2, 50)
        # Herschel-Bulkley-like: sigma_y + K*gd^n, strong plateau at low rate
        sigma = 50.0 + 2.0 * gamma_dot**0.5
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert result.diagnostics["has_yield_stress"] is True
        assert result.diagnostics["yield_stress_estimate"] > 0
        assert "eta_0" in result.diagnostics

    def test_newtonian_no_yield_stress(self):
        """Sigma ~ gamma_dot (slope 1) has no yield stress."""
        gamma_dot = np.logspace(-2, 2, 50)
        sigma = 10.0 * gamma_dot
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert result.diagnostics["has_yield_stress"] is False
        np.testing.assert_allclose(
            result.diagnostics["low_rate_slope"], 1.0, atol=0.05
        )

    def test_shear_banding_flag(self):
        """Non-monotonic stress flags shear banding + warning."""
        gamma_dot = np.logspace(-2, 2, 50)
        sigma = 10.0 * gamma_dot.copy()
        sigma[25:30] = sigma[24]  # dip: force decreasing steps
        sigma[26] = sigma[24] * 0.5
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert result.diagnostics["shear_banding_flag"] is True
        assert result.diagnostics["n_stress_decreases"] > 0
        assert any("shear banding" in w.lower() for w in result.warnings)

    def test_monotonic_no_banding(self):
        gamma_dot = np.logspace(-2, 2, 50)
        sigma = 10.0 * gamma_dot
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert result.diagnostics["shear_banding_flag"] is False
        assert result.diagnostics["n_stress_decreases"] == 0

    def test_short_array_skips_slope_block(self):
        """len(gamma_dot) <= 5 skips the slope/yield/banding block."""
        gamma_dot = np.logspace(-2, 0, 4)
        sigma = 10.0 * gamma_dot
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert "min_log_slope" not in result.diagnostics
        # eta_0 still attempted
        assert "eta_0" in result.diagnostics

    def test_complex_stress_uses_real_part(self):
        gamma_dot = np.logspace(-2, 2, 50)
        sigma = (10.0 * gamma_dot) + 1j * np.ones_like(gamma_dot)
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        assert np.iscomplexobj(result.y)
        assert np.isfinite(result.diagnostics["min_log_slope"])

    def test_all_duplicate_shear_rates_zero_slope(self):
        """All-duplicate gamma_dot -> no valid log-slopes -> slopes fall back to 0."""
        gamma_dot = np.full(10, 3.0)
        sigma = np.full(10, 25.0)
        result = preprocess_for_protocol(gamma_dot, sigma, "flow_curve")
        np.testing.assert_allclose(result.diagnostics["min_log_slope"], 0.0)
        # zero slope reads as a low-rate plateau -> yield stress flagged
        assert result.diagnostics["has_yield_stress"] is True

    def test_empty_input_eta0_failure_swallowed(self):
        """Empty arrays: slope block skipped, estimate_eta0 raises and is caught."""
        result = preprocess_for_protocol(
            np.array([]), np.array([]), "flow_curve"
        )
        assert "eta_0" not in result.diagnostics
        assert result.diagnostics == {}
        assert result.warnings == []


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestPreprocessStartup:
    """Tests for _preprocess_startup via the dispatcher."""

    def test_overshoot_ratio(self):
        """Stress overshoot then decay to steady state gives ratio > 1."""
        t = np.linspace(0.0, 10.0, 100)
        # Rise to peak near t=1, decay to plateau ~1.0
        sigma = 2.5 * t * np.exp(-t) + 1.0
        result = preprocess_for_protocol(t, sigma, "startup")
        peak = result.diagnostics["sigma_peak"]
        ss = result.diagnostics["sigma_steady_state"]
        assert peak >= ss
        assert result.diagnostics["overshoot_ratio"] >= 1.0
        np.testing.assert_allclose(
            result.diagnostics["overshoot_ratio"], peak / ss, rtol=1e-9
        )

    def test_peak_at_end_no_steady_state(self):
        """Monotone-rising stress: peak is the last point, no steady-state block."""
        t = np.linspace(0.0, 10.0, 50)
        sigma = t.copy()  # argmax is last index
        result = preprocess_for_protocol(t, sigma, "startup")
        np.testing.assert_allclose(result.diagnostics["t_peak"], t[-1], rtol=1e-9)
        assert "sigma_steady_state" not in result.diagnostics

    def test_complex_stress_uses_real_part(self):
        t = np.linspace(0.0, 10.0, 100)
        sigma = (2.5 * t * np.exp(-t) + 1.0) + 1j * np.ones_like(t)
        result = preprocess_for_protocol(t, sigma, "startup")
        assert np.iscomplexobj(result.y)
        assert np.isfinite(result.diagnostics["sigma_peak"])


# ---------------------------------------------------------------------------
# LAOS
# ---------------------------------------------------------------------------


def _laos_signal(n=256, third=0.1, gamma_0_cycles=1):
    """One-window fundamental + scaled third harmonic on [0,1)."""
    t = np.linspace(0.0, 1.0, n, endpoint=False)
    response = np.sin(2 * np.pi * t) + third * np.sin(2 * np.pi * 3 * t)
    return t, response


@pytest.mark.smoke
class TestPreprocessLaos:
    """Tests for _preprocess_laos via the dispatcher."""

    def test_harmonic_ratio_recovered(self):
        """I3/I1 recovers the injected third-harmonic amplitude ratio."""
        t, response = _laos_signal(third=0.1)
        result = preprocess_for_protocol(t, response, "laos")
        ew = result.diagnostics["ewoldt_classification"]
        np.testing.assert_allclose(ew["harmonic_ratio_I3_I1"], 0.1, rtol=1e-3)
        assert result.diagnostics["n_points"] == t.shape[0]
        assert "response_range" in result.diagnostics
        assert result.diagnostics["estimated_cycles"] >= 1

    def test_missing_gamma0_warns_and_raw_ratio(self):
        """Without gamma_0, Q0 is the raw ratio and a warning is emitted."""
        t, response = _laos_signal(third=0.05)
        result = preprocess_for_protocol(t, response, "laos")
        assert result.diagnostics["Q0_formula"].startswith("I3_I1")
        assert any("gamma_0" in w for w in result.warnings)
        np.testing.assert_allclose(result.diagnostics["Q0"], 0.05, rtol=1e-3)

    def test_gamma0_provided_scales_q0(self):
        """With gamma_0, Q0 = (I3/I1)/gamma_0^2."""
        t, response = _laos_signal(third=0.2)
        gamma_0 = 2.0
        result = preprocess_for_protocol(t, response, "laos", gamma_0=gamma_0)
        assert result.diagnostics["Q0_formula"] == "I3_I1 / gamma_0^2"
        expected = 0.2 / gamma_0**2
        np.testing.assert_allclose(result.diagnostics["Q0"], expected, rtol=1e-3)

    def test_strong_nonlinearity_warns(self):
        """Large harmonic ratio triggers the strong-nonlinearity warning."""
        t, response = _laos_signal(third=0.3)  # raw Q0 = 0.3 > 0.1
        result = preprocess_for_protocol(t, response, "laos")
        assert result.diagnostics["Q0"] > 0.1
        assert any("Strong nonlinearity" in w for w in result.warnings)

    def test_short_array_only_basic_stats(self):
        """len <= 10 skips Fourier analysis; only basic stats present."""
        t = np.linspace(0.0, 1.0, 8, endpoint=False)
        response = np.sin(2 * np.pi * t)
        result = preprocess_for_protocol(t, response, "laos")
        assert "ewoldt_classification" not in result.diagnostics
        assert result.diagnostics["n_points"] == 8
        assert "zero_crossings" in result.diagnostics

    def test_complex_response_uses_real_part(self):
        t, response = _laos_signal(third=0.1)
        result = preprocess_for_protocol(
            t, response + 1j * np.ones_like(response), "laos"
        )
        assert np.iscomplexobj(result.y)
        assert "ewoldt_classification" in result.diagnostics

    def test_ewoldt_classification_labels(self):
        """Classification fields take one of the documented labels."""
        t, response = _laos_signal(third=0.15)
        result = preprocess_for_protocol(t, response, "laos")
        ew = result.diagnostics["ewoldt_classification"]
        assert ew["elastic_behavior"] in ("strain_stiffening", "strain_softening")
        assert ew["viscous_behavior"] in ("shear_thickening", "shear_thinning")


# ---------------------------------------------------------------------------
# Property-style: diagnostics-only invariance
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize(
    "mode",
    ["relaxation", "creep", "oscillation", "flow_curve", "startup", "laos"],
)
def test_default_call_never_mutates_data(mode):
    """Without opt-in cleaning, X/y are returned unchanged (diagnostics-only)."""
    x = np.linspace(0.1, 10.0, 60)
    if mode == "oscillation":
        y = (1000.0 * x**0.5) + 1j * (500.0 * x**0.5)
    elif mode == "laos":
        y = np.sin(2 * np.pi * np.linspace(0, 1, 60, endpoint=False))
    else:
        y = np.exp(-x) + 1.0
    x_ref = x.copy()
    y_ref = y.copy()
    result = preprocess_for_protocol(x, y, mode)
    assert result.applied == []
    np.testing.assert_array_equal(result.X, x_ref)
    np.testing.assert_array_equal(result.y, y_ref)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "mode",
    ["relaxation", "creep", "oscillation", "flow_curve", "startup", "laos"],
)
def test_diagnostics_contain_no_nan_or_inf(mode):
    """Numeric diagnostics stay finite on well-formed inputs."""
    x = np.linspace(0.1, 10.0, 80)
    if mode == "oscillation":
        y = (1000.0 * x**0.5) + 1j * (500.0 * x**0.5)
    elif mode == "laos":
        y = np.sin(2 * np.pi * np.linspace(0, 1, 80, endpoint=False))
    else:
        y = np.exp(-x) + 1.0
    result = preprocess_for_protocol(x, y, mode)

    def _check(v):
        if isinstance(v, float):
            assert np.isfinite(v), f"{mode}: non-finite diagnostic {v}"
        elif isinstance(v, (tuple, list)):
            for item in v:
                _check(item)
        elif isinstance(v, dict):
            for item in v.values():
                _check(item)

    for value in result.diagnostics.values():
        _check(value)
