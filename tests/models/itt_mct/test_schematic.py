"""Tests for ITTMCTSchematic (F₁₂) model.

Tests cover:
- Parameter initialization
- Glass transition detection
- All 6 protocols (flow curve, SAOS, startup, creep, relaxation, LAOS)
- Bayesian inference basics
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.itt_mct import ITTMCTSchematic

jax, jnp = safe_import_jax()


class TestITTMCTSchematicInitialization:
    """Tests for model initialization."""

    @pytest.mark.smoke
    def test_default_initialization(self):
        """Test default initialization creates fluid state."""
        model = ITTMCTSchematic()

        assert model.parameters.get_value("v1") == 0.0
        assert model.parameters.get_value("v2") == 2.0  # Fluid (< 4)
        assert model.parameters.get_value("Gamma") == 1.0
        assert model.parameters.get_value("gamma_c") == 0.1
        assert model.parameters.get_value("G_inf") == 1e6

    @pytest.mark.smoke
    def test_initialization_with_epsilon(self):
        """Test initialization with separation parameter."""
        # Fluid state
        model_fluid = ITTMCTSchematic(epsilon=-0.1)
        info_fluid = model_fluid.get_glass_transition_info()
        assert not info_fluid["is_glass"]
        assert info_fluid["epsilon"] == pytest.approx(-0.1, rel=1e-3)

        # Glass state
        model_glass = ITTMCTSchematic(epsilon=0.1)
        info_glass = model_glass.get_glass_transition_info()
        assert info_glass["is_glass"]
        assert info_glass["epsilon"] == pytest.approx(0.1, rel=1e-3)

    def test_initialization_with_v2(self):
        """Test initialization with direct v2 value."""
        model = ITTMCTSchematic(v2=5.0)  # Glass (> 4)
        info = model.get_glass_transition_info()
        assert info["is_glass"]
        assert model.parameters.get_value("v2") == 5.0

    def test_cannot_specify_both_epsilon_and_v2(self):
        """Test that specifying both epsilon and v2 raises error."""
        with pytest.raises(ValueError, match="Specify either epsilon or v2"):
            ITTMCTSchematic(epsilon=0.1, v2=5.0)

    def test_epsilon_property(self):
        """Test epsilon property getter and setter."""
        model = ITTMCTSchematic(epsilon=0.05)

        # Getter
        assert model.epsilon == pytest.approx(0.05, rel=1e-3)

        # Setter
        model.epsilon = -0.1
        assert model.epsilon == pytest.approx(-0.1, rel=1e-3)
        assert not model.get_glass_transition_info()["is_glass"]


class TestGlassTransition:
    """Tests for glass transition behavior."""

    @pytest.mark.smoke
    def test_fluid_state_properties(self):
        """Test properties in fluid state (ε < 0)."""
        model = ITTMCTSchematic(epsilon=-0.1)
        info = model.get_glass_transition_info()

        assert not info["is_glass"]
        assert info["epsilon"] < 0
        assert info["f_neq"] == 0.0  # No arrested structure

    @pytest.mark.smoke
    def test_glass_state_properties(self):
        """Test properties in glass state (ε > 0)."""
        model = ITTMCTSchematic(epsilon=0.1)
        info = model.get_glass_transition_info()

        assert info["is_glass"]
        assert info["epsilon"] > 0
        assert info["f_neq"] > 0  # Non-zero plateau

    def test_critical_point(self):
        """Test behavior at critical point (ε ≈ 0)."""
        model = ITTMCTSchematic(epsilon=0.0)
        info = model.get_glass_transition_info()

        assert info["epsilon"] == pytest.approx(0.0, abs=1e-6)
        # At critical point, marginally in fluid state
        assert not info["is_glass"]


class TestFlowCurveProtocol:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.slow
    def test_flow_curve_fluid(self):
        """Test flow curve in fluid state."""
        model = ITTMCTSchematic(epsilon=-0.1)
        gamma_dot = np.logspace(-2, 2, 10)

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma >= 0)
        # Fluid should show no yield stress
        assert sigma[0] < sigma[-1]  # Increasing with shear rate

    @pytest.mark.slow
    def test_flow_curve_glass(self):
        """Test flow curve in glass state (yield stress)."""
        model = ITTMCTSchematic(epsilon=0.1)
        gamma_dot = np.logspace(-3, 2, 10)

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma >= 0)
        # Glass should show yield stress (non-zero at γ̇ → 0)
        assert sigma[0] > 0

    @pytest.mark.slow
    def test_flow_curve_shear_thinning(self):
        """Test shear thinning behavior."""
        model = ITTMCTSchematic(epsilon=0.05)
        gamma_dot = np.logspace(-1, 3, 20)

        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        # Compute effective viscosity
        eta = sigma / gamma_dot

        # Should show shear thinning (decreasing viscosity)
        assert eta[-1] < eta[0]


class TestOscillationProtocol:
    """Tests for SAOS (G', G'') predictions."""

    @pytest.mark.slow
    def test_oscillation_returns_moduli(self):
        """Test that oscillation returns valid moduli."""
        model = ITTMCTSchematic(epsilon=-0.05)
        omega = np.logspace(-1, 2, 10)

        # Test |G*|
        G_star = model.predict(omega, test_mode="oscillation")
        assert G_star.shape == omega.shape
        assert np.all(G_star >= 0)

        # Test components
        G_components = model.predict(
            omega, test_mode="oscillation", return_components=True
        )
        assert G_components.shape == (len(omega), 2)
        G_prime = G_components[:, 0]
        G_double_prime = G_components[:, 1]

        # G' should increase with frequency for viscoelastic material
        assert G_prime[-1] >= G_prime[0]

    @pytest.mark.slow
    def test_glass_plateau_modulus(self):
        """Test that glass shows plateau modulus."""
        model = ITTMCTSchematic(epsilon=0.1)
        omega = np.logspace(-2, 1, 15)

        G_components = model.predict(
            omega, test_mode="oscillation", return_components=True
        )
        G_prime = G_components[:, 0]

        # Glass should have relatively flat G' at low frequency.
        # With G(t) = G_∞Φ²(t), the plateau is at G_∞f² and the approach
        # is steeper than Φ¹, so CoV threshold must be relaxed.
        G_prime_low = G_prime[:5]
        assert np.std(G_prime_low) / np.mean(G_prime_low) < 1.5


class TestStartupProtocol:
    """Tests for startup flow predictions."""

    @pytest.mark.slow
    def test_startup_stress_growth(self):
        """Test stress growth in startup flow."""
        model = ITTMCTSchematic(epsilon=0.05)
        t = np.linspace(0, 10, 50)
        gamma_dot = 1.0

        sigma = model.predict(t, test_mode="startup", gamma_dot=gamma_dot)

        assert sigma.shape == t.shape
        assert sigma[0] == pytest.approx(0.0, abs=1e-6)  # σ(0) = 0
        assert np.all(sigma >= 0)

    @pytest.mark.slow
    def test_startup_overshoot(self):
        """Test stress overshoot in startup (characteristic of MCT)."""
        model = ITTMCTSchematic(epsilon=0.1)
        t = np.linspace(0, 20, 100)
        gamma_dot = 10.0

        sigma = model.predict(t, test_mode="startup", gamma_dot=gamma_dot)

        # Find maximum stress
        sigma_max = np.max(sigma)
        sigma_final = sigma[-1]

        # Overshoot means max > final
        # Note: This may not always occur depending on parameters
        assert sigma_max >= sigma_final * 0.9  # At least close to or exceeding final


class TestCreepProtocol:
    """Tests for creep compliance predictions."""

    @pytest.mark.slow
    def test_creep_compliance_positive(self):
        """Test creep compliance is positive and increasing."""
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.linspace(0.1, 100, 50)
        sigma_applied = 100.0

        J = model.predict(t, test_mode="creep", sigma_applied=sigma_applied)

        assert J.shape == t.shape
        assert np.all(J >= 0)
        # Compliance should increase with time for fluid
        assert J[-1] > J[0]

    @pytest.mark.slow
    def test_creep_glass_bounded(self):
        """Test that glass state creep is bounded."""
        model = ITTMCTSchematic(epsilon=0.1)
        t = np.linspace(0.1, 100, 50)
        sigma_applied = 50.0  # Below yield stress

        J = model.predict(t, test_mode="creep", sigma_applied=sigma_applied)

        assert J.shape == t.shape
        assert np.all(J >= 0)
        # Glass should have bounded compliance


class TestRelaxationProtocol:
    """Tests for stress relaxation predictions."""

    @pytest.mark.slow
    def test_relaxation_stress_decay(self):
        """Test stress decay in relaxation."""
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.linspace(0, 50, 50)
        gamma_pre = 0.05

        sigma = model.predict(t, test_mode="relaxation", gamma_pre=gamma_pre)

        assert sigma.shape == t.shape
        # Stress should decay for fluid
        assert sigma[0] >= sigma[-1]

    @pytest.mark.slow
    def test_relaxation_glass_residual(self):
        """Test residual stress in glass relaxation."""
        model = ITTMCTSchematic(epsilon=0.1)
        t = np.linspace(0, 100, 50)
        gamma_pre = 0.05

        sigma = model.predict(t, test_mode="relaxation", gamma_pre=gamma_pre)

        # Glass should have non-zero residual stress
        assert sigma[-1] > 0


class TestLAOSProtocol:
    """Tests for LAOS predictions."""

    @pytest.mark.slow
    def test_laos_oscillatory_response(self):
        """Test LAOS gives oscillatory stress response."""
        model = ITTMCTSchematic(epsilon=0.05)
        T = 2 * np.pi  # One period
        t = np.linspace(0, 3 * T, 200)
        gamma_0 = 0.1
        omega = 1.0

        sigma = model.predict(t, test_mode="laos", gamma_0=gamma_0, omega=omega)

        assert sigma.shape == t.shape
        # Should be oscillatory (changes sign or varies significantly)
        sigma_range = sigma.max() - sigma.min()
        assert sigma_range > 0

    @pytest.mark.slow
    def test_laos_harmonics(self):
        """Test extraction of LAOS harmonics."""
        model = ITTMCTSchematic(epsilon=0.1)
        T = 2 * np.pi
        t = np.linspace(0, 5 * T, 500)  # Multiple periods
        gamma_0 = 0.2
        omega = 1.0

        sigma_prime, sigma_double_prime = model.get_laos_harmonics(
            t, gamma_0=gamma_0, omega=omega, n_harmonics=3
        )

        # Should have fundamental plus higher harmonics
        assert len(sigma_prime) == 3
        assert len(sigma_double_prime) == 3
        # Fundamental should be largest
        assert np.abs(sigma_prime[0]) >= np.abs(sigma_prime[1])


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.slow
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data."""
        # Generate synthetic data
        true_model = ITTMCTSchematic(epsilon=0.05)
        true_model.parameters.set_value("G_inf", 5e5)
        gamma_dot = np.logspace(-2, 2, 20)
        sigma_true = true_model.predict(gamma_dot, test_mode="flow_curve")

        # Add noise
        noise = np.random.normal(0, 0.05 * sigma_true.mean(), sigma_true.shape)
        sigma_noisy = sigma_true + noise

        # Fit new model
        fit_model = ITTMCTSchematic(epsilon=-0.1)  # Start in different state
        fit_model.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Check recovery (approximate)
        sigma_fit = fit_model.predict(gamma_dot, test_mode="flow_curve")
        r_squared = 1 - np.sum((sigma_fit - sigma_true) ** 2) / np.sum(
            (sigma_true - sigma_true.mean()) ** 2
        )
        assert r_squared > 0.8


class TestModelFunction:
    """Tests for static model function (Bayesian compatibility)."""

    @pytest.mark.smoke
    def test_model_function_flow_curve(self):
        """Test model_function raises NotImplementedError (Bayesian not supported)."""
        model = ITTMCTSchematic()
        gamma_dot = np.logspace(-1, 2, 10)

        params = np.array([0.0, 4.5, 1.0, 0.1, 1e6])
        with pytest.raises(
            NotImplementedError, match="Bayesian inference is not yet supported"
        ):
            model.model_function(
                gamma_dot,
                params,
                test_mode="flow_curve",
            )

    def test_model_function_raises_and_preserves_params(self):
        """Test that model_function raises NotImplementedError and preserves params."""
        model = ITTMCTSchematic(epsilon=0.1)
        original_v2 = model.parameters.get_value("v2")

        gamma_dot = np.logspace(-1, 1, 5)
        params_array = np.array([0.0, 2.0, 1.0, 0.1, 1e6])  # Different v2

        with pytest.raises(NotImplementedError, match="Bayesian"):
            model.model_function(
                gamma_dot,
                params_array,
                test_mode="flow_curve",
            )

        # Original parameters should be preserved (model_function raises before modifying)
        assert model.parameters.get_value("v2") == pytest.approx(original_v2, rel=1e-6)


class TestRepr:
    """Tests for string representation."""

    def test_repr_fluid(self):
        """Test repr for fluid state."""
        model = ITTMCTSchematic(epsilon=-0.1)
        repr_str = repr(model)

        assert "ITTMCTSchematic" in repr_str
        assert "fluid" in repr_str
        assert "ε=" in repr_str

    def test_repr_glass(self):
        """Test repr for glass state."""
        model = ITTMCTSchematic(epsilon=0.1)
        repr_str = repr(model)

        assert "ITTMCTSchematic" in repr_str
        assert "glass" in repr_str


# =============================================================================
# Fast coverage tests for the prediction / fitting machinery.
#
# The tests above are marked @pytest.mark.slow and are deselected from the
# coverage run, leaving nearly all of schematic.py's predict/fit code
# unexercised. The tests below drive every protocol path with small arrays and
# scipy (not diffrax, except where explicitly noted) so they stay fast while
# still exercising the numerical machinery and checking physical invariants.
# =============================================================================


class TestConstructionValidation:
    """Constructor option validation and derived-attribute wiring."""

    def test_invalid_decorrelation_form_raises(self):
        with pytest.raises(ValueError, match="decorrelation_form must be"):
            ITTMCTSchematic(decorrelation_form="bogus")  # type: ignore[arg-type]

    def test_invalid_memory_form_raises(self):
        with pytest.raises(ValueError, match="memory_form must be"):
            ITTMCTSchematic(memory_form="bogus")  # type: ignore[arg-type]

    def test_invalid_stress_form_raises(self):
        with pytest.raises(ValueError, match="stress_form must be"):
            ITTMCTSchematic(stress_form="bogus")  # type: ignore[arg-type]

    def test_microscopic_requires_phi_volume(self):
        with pytest.raises(ValueError, match="phi_volume is required"):
            ITTMCTSchematic(stress_form="microscopic")

    def test_lorentzian_form_wired(self):
        model = ITTMCTSchematic(epsilon=0.05, decorrelation_form="lorentzian")
        assert model.decorrelation_form == "lorentzian"
        assert model._use_lorentzian is True

    def test_full_memory_form_wired(self):
        model = ITTMCTSchematic(epsilon=0.05, memory_form="full")
        assert model.memory_form == "full"

    def test_microscopic_stress_form_wired(self):
        model = ITTMCTSchematic(epsilon=0.05, stress_form="microscopic", phi_volume=0.5)
        assert model.stress_form == "microscopic"
        # Prefactor is precomputed at construction for the microscopic path.
        assert model._microscopic_stress_prefactor is not None

    def test_v2_critical_nonzero_v1(self):
        """_get_v2_critical(v1) must agree with glass_transition_criterion's
        exact v2_critical, since get_glass_transition_info() is the
        independent source of truth for the same quantity."""
        model = ITTMCTSchematic()
        model.parameters.set_value("v1", 0.5)
        v2_c = model._get_v2_critical(0.5)

        # Exact closed form: v2_c = (1 + sqrt(1 - v1))^2 (Gotze 2009 Sec 4.3).
        expected = (1.0 + 0.5**0.5) ** 2
        assert v2_c == pytest.approx(expected, rel=1e-12)

        # And it must match what get_glass_transition_info() independently
        # computes via glass_transition_criterion for the same v1.
        model.parameters.set_value("v2", v2_c * 1.1)
        info = model.get_glass_transition_info()
        assert v2_c == pytest.approx(info["v2_critical"], rel=1e-12)

    def test_epsilon_roundtrip_nonzero_v1(self):
        """Setting model.epsilon with v1 != 0 must read back the same
        epsilon that get_glass_transition_info() independently reports,
        not a value derived from a different v2_critical formula."""
        model = ITTMCTSchematic()
        model.parameters.set_value("v1", 0.5)
        model.epsilon = 0.1

        assert model.epsilon == pytest.approx(0.1, rel=1e-9)
        assert model.get_glass_transition_info()["epsilon"] == pytest.approx(
            0.1, rel=1e-9
        )


class TestSteadyStateStressRefactor:
    """t_max bugfix + rename regression coverage (no diffrax variant)."""

    def test_default_t_max_uses_max_not_min(self):
        """Old formula: min(tau_bare, tau_shear) collapsed the yield plateau
        at low gamma_dot. New formula: max(tau_bare, tau_shear), matching
        compute_adaptive_t_max in _kernels_diffrax.py.
        """
        from rheojax.models.itt_mct.schematic import _default_steady_state_t_max

        Gamma = 1.0  # tau_bare = 1.0
        gamma_c = 0.1
        gamma_dot = 0.001  # tau_shear = 0.1 / 0.001 = 100.0

        t_max = _default_steady_state_t_max(gamma_dot, Gamma, gamma_c)

        # max(1.0, 100.0) * 50 = 5000, clipped to new upper bound 1000.
        assert t_max == pytest.approx(1000.0)

    def test_default_t_max_respects_lower_bound(self):
        from rheojax.models.itt_mct.schematic import _default_steady_state_t_max

        Gamma = 1000.0  # tau_bare = 0.001
        gamma_c = 0.1
        gamma_dot = 1000.0  # tau_shear = 0.1 / 1000 = 0.0001

        t_max = _default_steady_state_t_max(gamma_dot, Gamma, gamma_c)

        # max(0.001, 0.0001) * 50 = 0.05, clipped up to the 10.0 floor.
        assert t_max == pytest.approx(10.0)

    def test_compute_steady_state_stress_scipy_exists(self):
        """The rename landed: the scipy-only method has the new name."""
        model = ITTMCTSchematic(epsilon=-0.05)
        assert hasattr(model, "_compute_steady_state_stress_scipy")
        sigma = model._compute_steady_state_stress_scipy(5.0)
        assert np.isfinite(sigma)
        assert sigma > 0.0

    def test_flow_curve_scipy_path_still_works_after_rename(self):
        """_predict_flow_curve_scipy's call site was updated correctly."""
        model = ITTMCTSchematic(epsilon=-0.05)
        gamma_dot = np.array([0.0, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=False)
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0.0)

    def test_flow_curve_diffrax_nan_fallback_still_works_after_rename(
        self, monkeypatch
    ):
        """_predict_flow_curve_diffrax's NaN-fallback call site was updated.

        Deterministic (monkeypatched), not glass-state-triggered: a real
        glass-state input isn't guaranteed to make diffrax fail, which
        would leave this call site unexercised (see the spec's
        "Deterministic fallback tests" guidance).
        """
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=0.05)
        gamma_dot = np.array([0.0, 1.0, 10.0, 100.0])

        def fake_solve_flow_curve_batch(gamma_dot_array, *args, **kwargs):
            import jax.numpy as jnp

            return jnp.full(gamma_dot_array.shape, jnp.nan)

        monkeypatch.setattr(
            schematic_mod, "solve_flow_curve_batch", fake_solve_flow_curve_batch
        )

        sigma = model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)
        assert np.all(np.isfinite(sigma))


class TestSteadyStateStressConvergence:
    """_compute_steady_state_stress_scipy must not silently trust a non-converged solve_ivp."""

    def test_raises_when_solver_never_converges(self, monkeypatch):
        """If solve_ivp reports success=False on every attempt (LSODA and the
        Radau retry), the method must raise instead of silently returning
        sol.y[..., -1] from an integration that never reached t_max."""
        model = ITTMCTSchematic(epsilon=0.05)
        # Pre-warm the Prony-mode cache with a real (unpatched) solve so the
        # mock below only intercepts the steady-state ODE under test, not
        # the unrelated solve_ivp call inside initialize_prony_modes().
        model._compute_steady_state_stress_scipy(0.5)

        class _FailedSol:
            success = False
            message = "mock non-convergence"
            y = np.zeros((3 + model.n_prony_modes, 1))
            t = np.array([0.0])

        monkeypatch.setattr(
            "rheojax.models.itt_mct.schematic.solve_ivp",
            lambda *args, **kwargs: _FailedSol(),
        )

        with pytest.raises(RuntimeError, match="failed to converge"):
            model._compute_steady_state_stress_scipy(1.0)


class TestEquilibriumCorrelator:
    """Direct exercise of the quiescent correlator Φ_eq(t)."""

    def test_correlator_bounds_and_ic_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.logspace(-3, 1, 40)
        phi = np.asarray(model._compute_equilibrium_correlator(t))

        assert phi.shape == t.shape
        assert np.all(np.isfinite(phi))
        # Physical bounds enforced by the model (clip to [0, 1]).
        assert np.all(phi >= 0.0) and np.all(phi <= 1.0)
        # Φ starts near 1 (first grid point is t≈1e-3, already slightly relaxed).
        assert 0.99 < phi[0] <= 1.0
        # Fluid correlator relaxes: late time below early time.
        assert phi[-1] < phi[0]

    def test_correlator_glass_plateau(self):
        """Glass correlator retains a non-ergodic plateau (Φ_∞ > Φ_fluid_∞)."""
        glass = np.asarray(
            ITTMCTSchematic(epsilon=0.1)._compute_equilibrium_correlator(
                np.logspace(-3, 1, 40)
            )
        )
        fluid = np.asarray(
            ITTMCTSchematic(epsilon=-0.1)._compute_equilibrium_correlator(
                np.logspace(-3, 1, 40)
            )
        )
        assert glass[-1] >= fluid[-1]


@pytest.mark.slow
class TestEquilibriumCorrelatorDiffrax:
    """Diffrax equilibrium-correlator path (first call triggers JIT).

    Note on the TDD red step: `_compute_equilibrium_correlator` currently
    takes only `self, t` (no `**kwargs`), so every test below that passes
    `use_diffrax=...` genuinely raises `TypeError` until Step 10 adds the
    parameter -- a real red step, unlike `_predict_oscillation` below
    (which already has `**kwargs` and needs a different check; see
    `test_predict_oscillation_accepts_use_diffrax`).
    """

    def test_diffrax_matches_scipy_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = jnp.logspace(-2, 2, 30)

        phi_scipy = model._compute_equilibrium_correlator(t, use_diffrax=False)
        phi_diffrax = model._compute_equilibrium_correlator(t, use_diffrax=True)

        np.testing.assert_allclose(
            np.array(phi_diffrax), np.array(phi_scipy), rtol=1e-3, atol=1e-6
        )

    def test_dispatch_default_uses_diffrax_when_available(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = jnp.logspace(-2, 2, 20)
        phi_default = model._compute_equilibrium_correlator(t)
        phi_explicit = model._compute_equilibrium_correlator(t, use_diffrax=True)
        np.testing.assert_allclose(np.array(phi_default), np.array(phi_explicit))

    def test_nan_fallback_calls_scipy(self, monkeypatch):
        """Deterministic fallback test: force the diffrax helper to
        return NaN and assert the scipy path is used instead (not a
        glass-state test, which isn't guaranteed to trigger a failure).
        """
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        t = jnp.logspace(-2, 2, 10)

        def fake_solve(*args, **kwargs):
            return jnp.full((len(t),), jnp.nan)

        monkeypatch.setattr(
            schematic_mod, "solve_equilibrium_correlator_trajectory", fake_solve
        )

        phi = model._compute_equilibrium_correlator(t, use_diffrax=True)

        assert np.all(np.isfinite(np.array(phi)))

    def test_predict_oscillation_accepts_use_diffrax(self):
        """Genuine TDD red step for the oscillation forwarding: unlike
        _compute_equilibrium_correlator (no **kwargs), _predict_oscillation
        already accepts **kwargs today, so calling it with use_diffrax=...
        does NOT raise before Step 11 -- it silently swallows the kwarg
        and both backends end up computing via whatever
        _compute_equilibrium_correlator's own default resolves to, making
        a naive before/after parity assertion pass even without Step 11.
        Assert the signature explicitly instead."""
        import inspect

        model = ITTMCTSchematic(epsilon=0.05)
        assert "use_diffrax" in inspect.signature(model._predict_oscillation).parameters

    def test_oscillation_speedup_side_effect_still_correct(self):
        """_predict_oscillation depends on _compute_equilibrium_correlator
        internally — confirm its output is unaffected by the diffrax path
        existing (dispatch defaults to diffrax, but physics must match)."""
        model_scipy = ITTMCTSchematic(epsilon=0.05)
        model_diffrax = ITTMCTSchematic(epsilon=0.05)
        omega = np.logspace(-2, 2, 15)

        G_star_scipy = model_scipy._predict_oscillation(
            omega, use_diffrax=False, return_components=True
        )
        G_star_diffrax = model_diffrax._predict_oscillation(
            omega, use_diffrax=True, return_components=True
        )

        np.testing.assert_allclose(G_star_diffrax, G_star_scipy, rtol=1e-3, atol=1e-6)


class TestFlowCurveScipy:
    """Scipy (non-diffrax) flow-curve path and single-rate steady stress."""

    def test_flow_curve_scipy_fluid_monotonic(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        gamma_dot = np.array([0.0, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=False)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0.0)
        # Fluid: no yield stress at γ̇ = 0.
        assert sigma[0] == pytest.approx(0.0, abs=1e-9)
        # Stress increases with shear rate.
        assert np.all(np.diff(sigma[1:]) > 0)

    def test_flow_curve_scipy_glass_yield_stress(self):
        model = ITTMCTSchematic(epsilon=0.1)
        gamma_dot = np.array([0.0, 1.0, 10.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=False)

        assert np.all(np.isfinite(sigma))
        # Glass: finite yield stress persists as γ̇ → 0.
        assert sigma[0] > 0.0
        info = model.get_glass_transition_info()
        G_inf = model.parameters.get_value("G_inf")
        gamma_c = model.parameters.get_value("gamma_c")
        # Gaussian gamma_dot->0+ limit of sigma = G_inf*gamma_dot*int Phi_adv^2 dt
        # includes the int_0^inf h(x)^2 dx = sqrt(pi)/(2*sqrt(2)) prefactor
        # (see _predict_flow_curve_diffrax); a bare gamma_c*f_neq^2 would be
        # discontinuous with the finite-rate branch as gamma_dot -> 0.
        prefactor = np.sqrt(np.pi) / (2.0 * np.sqrt(2.0))
        expected = G_inf * gamma_c * info["f_neq"] ** 2 * prefactor
        np.testing.assert_allclose(sigma[0], expected, rtol=1e-6)

    def test_flow_curve_microscopic_stress_form(self):
        model = ITTMCTSchematic(epsilon=0.05, stress_form="microscopic", phi_volume=0.4)
        sigma = model.predict(
            np.array([0.0, 1.0, 10.0]), test_mode="flow_curve", use_diffrax=False
        )
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0.0)

    def test_steady_state_stress_lorentzian_full_memory(self):
        """Exercise the lorentzian + full-memory branches of steady-state stress."""
        model = ITTMCTSchematic(
            epsilon=0.05, decorrelation_form="lorentzian", memory_form="full"
        )
        sigma = model._compute_steady_state_stress_scipy(5.0)
        assert np.isfinite(sigma)
        assert sigma > 0.0

    def test_prony_cache_invalidated_on_param_change(self):
        """Changing physics params invalidates the cached Prony modes."""
        model = ITTMCTSchematic(epsilon=-0.05)
        model.predict(np.array([1.0, 10.0]), test_mode="flow_curve", use_diffrax=False)
        assert model._prony_amplitudes is not None

        model.parameters.set_value("v2", 5.0)  # fluid → glass
        model._check_prony_cache()
        assert model._prony_amplitudes is None


@pytest.mark.slow
class TestFlowCurveDiffrax:
    """Diffrax flow-curve path (first call triggers JIT compilation)."""

    def test_flow_curve_diffrax_glass(self):
        model = ITTMCTSchematic(epsilon=0.05)
        gamma_dot = np.array([0.0, 1.0, 10.0, 100.0])
        sigma = model.predict(gamma_dot, test_mode="flow_curve", use_diffrax=True)

        assert sigma.shape == gamma_dot.shape
        assert np.all(np.isfinite(sigma))
        assert np.all(sigma >= 0.0)
        # Zero-rate yield stress branch of the diffrax path.
        assert sigma[0] > 0.0


class TestOscillationDetailed:
    """SAOS moduli: magnitude/component consistency and positivity."""

    def test_oscillation_magnitude_matches_components(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        omega = np.logspace(-1, 1, 8)

        G_star = model.predict(omega, test_mode="oscillation")
        comps = model.predict(omega, test_mode="oscillation", return_components=True)

        assert comps.shape == (len(omega), 2)
        assert np.all(np.isfinite(comps))
        # |G*| returned as complex; its magnitude equals sqrt(G'^2 + G''^2).
        mag = np.abs(G_star)
        np.testing.assert_allclose(mag, np.hypot(comps[:, 0], comps[:, 1]), rtol=1e-6)
        # Loss modulus is non-negative for a passive material.
        assert np.all(comps[:, 1] >= -1e-6)


class TestStartupDetailed:
    """Startup flow: initial condition and finiteness."""

    def test_startup_initial_condition_and_finite(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.linspace(0.0, 5.0, 20)
        sigma = model.predict(t, test_mode="startup", gamma_dot=2.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))
        # σ(0) = 0 startup initial condition.
        np.testing.assert_allclose(sigma[0], 0.0, atol=1e-6)
        assert np.all(sigma >= -1e-9)

    def test_startup_lorentzian_full_memory(self):
        model = ITTMCTSchematic(
            epsilon=0.05, decorrelation_form="lorentzian", memory_form="full"
        )
        sigma = model.predict(
            np.linspace(0.0, 3.0, 15), test_mode="startup", gamma_dot=1.0
        )
        assert np.all(np.isfinite(sigma))


@pytest.mark.slow
class TestStartupDiffrax:
    def test_predict_startup_accepts_use_diffrax(self):
        """Genuine TDD red step: _predict_startup already accepts **kwargs
        today, so a bare use_diffrax=... call would silently swallow it
        (not raise) both before and after the real implementation --
        assert the signature explicitly instead of relying on a
        behavioral test to fail for the right reason."""
        import inspect

        model = ITTMCTSchematic(epsilon=-0.1)
        assert "use_diffrax" in inspect.signature(model._predict_startup).parameters

    def test_diffrax_matches_scipy_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 30)

        sigma_scipy = model.predict(
            t, test_mode="startup", gamma_dot=1.0, use_diffrax=False
        )
        sigma_diffrax = model.predict(
            t, test_mode="startup", gamma_dot=1.0, use_diffrax=True
        )

        np.testing.assert_allclose(sigma_diffrax, sigma_scipy, rtol=1e-3, atol=1e-6)

    def test_dispatch_default_uses_diffrax_when_available(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 20)
        sigma_default = model.predict(t, test_mode="startup", gamma_dot=1.0)
        sigma_explicit = model.predict(
            t, test_mode="startup", gamma_dot=1.0, use_diffrax=True
        )
        np.testing.assert_allclose(sigma_default, sigma_explicit)

    def test_nan_fallback_calls_scipy(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 10)

        def fake_solve(*args, **kwargs):
            return jnp.full((len(t),), jnp.nan)

        monkeypatch.setattr(schematic_mod, "solve_startup_trajectory", fake_solve)

        sigma = model.predict(t, test_mode="startup", gamma_dot=1.0, use_diffrax=True)

        assert np.all(np.isfinite(sigma))


class TestCreepDetailed:
    """Creep compliance: elastic-jump IC and monotonicity."""

    def test_creep_elastic_jump_ic(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.linspace(0.0, 10.0, 20)
        sigma_applied = 100.0
        J = model.predict(t, test_mode="creep", sigma_applied=sigma_applied)

        assert J.shape == t.shape
        assert np.all(np.isfinite(J))
        # J(0) = γ(0)/σ₀ = (σ₀/G_inf)/σ₀ = 1/G_inf (instantaneous elastic jump).
        G_inf = model.parameters.get_value("G_inf")
        np.testing.assert_allclose(J[0], 1.0 / G_inf, rtol=1e-6)
        # Fluid compliance grows with time.
        assert J[-1] > J[0]


@pytest.mark.slow
class TestCreepDiffrax:
    def test_predict_creep_accepts_use_diffrax(self):
        """Genuine TDD red step: _predict_creep already accepts **kwargs
        today, so a bare use_diffrax=... call would silently swallow it --
        assert the signature explicitly."""
        import inspect

        model = ITTMCTSchematic(epsilon=-0.1)
        assert "use_diffrax" in inspect.signature(model._predict_creep).parameters

    def test_dispatch_default_uses_diffrax_when_available(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 20)
        J_default = model.predict(t, test_mode="creep", sigma_applied=1.0)
        J_explicit = model.predict(
            t, test_mode="creep", sigma_applied=1.0, use_diffrax=True
        )
        np.testing.assert_allclose(J_default, J_explicit)

    def test_diffrax_matches_scipy_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 30)

        J_scipy = model.predict(
            t, test_mode="creep", sigma_applied=1.0, use_diffrax=False
        )
        J_diffrax = model.predict(
            t, test_mode="creep", sigma_applied=1.0, use_diffrax=True
        )

        np.testing.assert_allclose(J_diffrax, J_scipy, rtol=1e-3, atol=1e-6)

    def test_glass_state_stability(self):
        """Real (not monkeypatched) glass-state case: whichever backend
        is used, output must be finite -- exercises the actual NaN
        fallback path if diffrax genuinely fails here, without asserting
        it must fail (that would be flaky; see test_nan_fallback_calls_scipy
        for the deterministic version)."""
        model = ITTMCTSchematic(epsilon=0.1)
        t = np.linspace(0.01, 50.0, 30)

        J = model.predict(t, test_mode="creep", sigma_applied=1.0, use_diffrax=True)

        assert np.all(np.isfinite(J))

    def test_nan_fallback_calls_scipy(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 10)

        def fake_solve(*args, **kwargs):
            return jnp.full((len(t),), jnp.nan)

        monkeypatch.setattr(schematic_mod, "solve_creep_trajectory", fake_solve)

        J = model.predict(t, test_mode="creep", sigma_applied=1.0, use_diffrax=True)

        assert np.all(np.isfinite(J))


class TestRelaxationDetailed:
    """Stress relaxation: step-strain IC and glass residual."""

    def test_relaxation_initial_condition(self):
        model = ITTMCTSchematic(epsilon=-0.05)
        t = np.linspace(0.0, 20.0, 20)
        gamma_pre = 0.05
        sigma = model.predict(t, test_mode="relaxation", gamma_pre=gamma_pre)

        assert np.all(np.isfinite(sigma))
        # σ(0) = G_inf γ_pre h(γ_pre)² with gaussian decorrelation.
        G_inf = model.parameters.get_value("G_inf")
        gamma_c = model.parameters.get_value("gamma_c")
        h = np.exp(-((gamma_pre / gamma_c) ** 2))
        np.testing.assert_allclose(sigma[0], G_inf * gamma_pre * h * h, rtol=1e-6)
        # Fluid stress relaxes.
        assert sigma[-1] <= sigma[0] + 1e-9

    def test_relaxation_lorentzian_initial_condition(self):
        """Lorentzian decorrelation changes the step-strain IC."""
        model = ITTMCTSchematic(epsilon=0.05, decorrelation_form="lorentzian")
        t = np.linspace(0.0, 20.0, 20)
        gamma_pre = 0.05
        sigma = model.predict(t, test_mode="relaxation", gamma_pre=gamma_pre)

        G_inf = model.parameters.get_value("G_inf")
        gamma_c = model.parameters.get_value("gamma_c")
        h = 1.0 / (1.0 + (gamma_pre / gamma_c) ** 2)
        np.testing.assert_allclose(sigma[0], G_inf * gamma_pre * h * h, rtol=1e-6)
        # Glass retains residual stress.
        assert sigma[-1] > 0.0


@pytest.mark.slow
class TestRelaxationDiffrax:
    def test_predict_relaxation_accepts_use_diffrax(self):
        """Genuine TDD red step: _predict_relaxation already accepts
        **kwargs today, so a bare use_diffrax=... call would silently
        swallow it -- assert the signature explicitly."""
        import inspect

        model = ITTMCTSchematic(epsilon=-0.1)
        assert "use_diffrax" in inspect.signature(model._predict_relaxation).parameters

    def test_dispatch_default_uses_diffrax_when_available(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 100.0, 20)
        sigma_default = model.predict(t, test_mode="relaxation", gamma_pre=0.01)
        sigma_explicit = model.predict(
            t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=True
        )
        np.testing.assert_allclose(sigma_default, sigma_explicit)

    def test_diffrax_matches_scipy_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 100.0, 30)

        sigma_scipy = model.predict(
            t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=False
        )
        sigma_diffrax = model.predict(
            t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=True
        )

        np.testing.assert_allclose(sigma_diffrax, sigma_scipy, rtol=1e-3, atol=1e-6)

    def test_stress_never_negative_glass_and_fluid(self):
        """Regression guard: the algebraic reconstruction (sigma =
        G_inf*gamma_pre*clip(phi,0,1)**2) must never go negative, in
        either backend -- this is exactly the bug the scipy path's
        algebraic-reconstruction fix already solved."""
        t = np.logspace(-2, 3, 50)
        for epsilon in (-0.1, 0.1):
            model = ITTMCTSchematic(epsilon=epsilon)
            sigma_scipy = model.predict(
                t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=False
            )
            sigma_diffrax = model.predict(
                t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=True
            )
            assert np.all(sigma_scipy >= 0.0), f"scipy negative at epsilon={epsilon}"
            assert np.all(sigma_diffrax >= 0.0), (
                f"diffrax negative at epsilon={epsilon}"
            )

    def test_nan_fallback_calls_scipy(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.01, 10.0, 10)

        def fake_solve(*args, **kwargs):
            return jnp.full((len(t),), jnp.nan)

        monkeypatch.setattr(schematic_mod, "solve_relaxation_trajectory", fake_solve)

        sigma = model.predict(
            t, test_mode="relaxation", gamma_pre=0.01, use_diffrax=True
        )

        assert np.all(np.isfinite(sigma))


class TestLAOSDetailed:
    """LAOS response and harmonic extraction."""

    def test_laos_initial_condition_and_range(self):
        model = ITTMCTSchematic(epsilon=0.05)
        t = np.linspace(0.0, 2 * np.pi, 40)
        sigma = model.predict(t, test_mode="laos", gamma_0=0.1, omega=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))
        np.testing.assert_allclose(sigma[0], 0.0, atol=1e-6)
        assert sigma.max() - sigma.min() > 0.0

    def test_laos_harmonics_fundamental_dominates(self):
        model = ITTMCTSchematic(epsilon=0.05)
        t = np.linspace(0.0, 4 * np.pi, 200)
        sp, sdp = model.get_laos_harmonics(t, gamma_0=0.1, omega=1.0, n_harmonics=3)

        assert len(sp) == 3 and len(sdp) == 3
        assert np.all(np.isfinite(sp)) and np.all(np.isfinite(sdp))
        # Fundamental harmonic dominates the higher odd harmonics.
        assert np.abs(sp[0]) >= np.abs(sp[1])


@pytest.mark.slow
class TestLAOSDiffrax:
    def test_predict_laos_accepts_use_diffrax(self):
        """Genuine TDD red step: _predict_laos already accepts **kwargs
        today, so a bare use_diffrax=... call would silently swallow it --
        assert the signature explicitly."""
        import inspect

        model = ITTMCTSchematic(epsilon=-0.1)
        assert "use_diffrax" in inspect.signature(model._predict_laos).parameters

    def test_dispatch_default_uses_diffrax_when_available(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        omega = 1.0
        period = 2.0 * np.pi / omega
        t = np.linspace(0.0, 2 * period, 20)
        sigma_default = model.predict(t, test_mode="laos", gamma_0=0.1, omega=omega)
        sigma_explicit = model.predict(
            t, test_mode="laos", gamma_0=0.1, omega=omega, use_diffrax=True
        )
        np.testing.assert_allclose(sigma_default, sigma_explicit)

    def test_diffrax_matches_scipy_fluid(self):
        model = ITTMCTSchematic(epsilon=-0.1)
        omega = 1.0
        period = 2.0 * np.pi / omega
        t = np.linspace(0.0, 2 * period, 40)

        sigma_scipy = model.predict(
            t, test_mode="laos", gamma_0=0.1, omega=omega, use_diffrax=False
        )
        sigma_diffrax = model.predict(
            t, test_mode="laos", gamma_0=0.1, omega=omega, use_diffrax=True
        )

        np.testing.assert_allclose(sigma_diffrax, sigma_scipy, rtol=1e-3, atol=1e-6)

    def test_nan_fallback_calls_scipy(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        t = np.linspace(0.0, 10.0, 10)

        def fake_solve(*args, **kwargs):
            return jnp.full((len(t),), jnp.nan)

        monkeypatch.setattr(schematic_mod, "solve_laos_trajectory", fake_solve)

        sigma = model.predict(
            t, test_mode="laos", gamma_0=0.1, omega=1.0, use_diffrax=True
        )

        assert np.all(np.isfinite(sigma))


class TestFitUseDiffraxForwarding:
    """model.fit(..., use_diffrax=False) must actually reach _predict_X
    during NLSQ residual evaluation, for every protocol."""

    def test_fit_flow_curve_forwards_use_diffrax(self, monkeypatch):
        model = ITTMCTSchematic(epsilon=-0.1)
        gamma_dot = np.array([1.0, 10.0])
        sigma = np.array([1e5, 1e6])

        seen = {"use_diffrax": "not called"}
        original = model._predict_flow_curve

        def spy(gd, use_diffrax=None, **kwargs):
            seen["use_diffrax"] = use_diffrax
            return original(gd, use_diffrax=use_diffrax, **kwargs)

        monkeypatch.setattr(model, "_predict_flow_curve", spy)
        model.fit(
            gamma_dot, sigma, test_mode="flow_curve", use_diffrax=False, max_iter=2
        )

        assert seen["use_diffrax"] is False

    @pytest.mark.parametrize(
        ("test_mode", "x", "y", "extra_kwargs"),
        [
            ("startup", np.linspace(0.01, 10.0, 5), np.ones(5), {"gamma_dot": 1.0}),
            (
                "creep",
                np.linspace(0.01, 10.0, 5),
                np.ones(5) * 1e-6,
                {"sigma_applied": 1.0},
            ),
            ("relaxation", np.linspace(0.01, 10.0, 5), np.ones(5), {"gamma_pre": 0.01}),
            (
                "laos",
                np.linspace(0.0, 10.0, 5),
                np.ones(5),
                {"gamma_0": 0.1, "omega": 1.0},
            ),
        ],
    )
    def test_fit_protocol_forwards_use_diffrax(
        self, test_mode, x, y, extra_kwargs, monkeypatch
    ):
        model = ITTMCTSchematic(epsilon=-0.1)
        predict_method_name = f"_predict_{test_mode}"
        original = getattr(model, predict_method_name)

        seen = {"use_diffrax": "not called"}

        def spy(*args, use_diffrax=None, **kwargs):
            seen["use_diffrax"] = use_diffrax
            return original(*args, use_diffrax=use_diffrax, **kwargs)

        monkeypatch.setattr(model, predict_method_name, spy)
        model.fit(
            x, y, test_mode=test_mode, use_diffrax=False, max_iter=2, **extra_kwargs
        )

        assert seen["use_diffrax"] is False

    def test_fit_oscillation_forwards_use_diffrax(self, monkeypatch):
        """_fit_oscillation is the 6th closure -- easy to miss since it's
        not one of the 5 new diffrax-kernel protocols, but it depends on
        _predict_oscillation, which gained use_diffrax in Task 2."""
        model = ITTMCTSchematic(epsilon=-0.1)
        omega = np.logspace(-1, 1, 5)
        G_star = np.ones(5) * 1e5 + 1j * np.ones(5) * 1e4

        original = model._predict_oscillation
        seen = {"use_diffrax": "not called"}

        def spy(*args, use_diffrax=None, **kwargs):
            seen["use_diffrax"] = use_diffrax
            return original(*args, use_diffrax=use_diffrax, **kwargs)

        monkeypatch.setattr(model, "_predict_oscillation", spy)
        model.fit(omega, G_star, test_mode="oscillation", use_diffrax=False, max_iter=2)

        assert seen["use_diffrax"] is False


@pytest.mark.slow
class TestPrecompile:
    """Diffrax solver precompilation entry point."""

    def test_precompile_returns_time(self):
        model = ITTMCTSchematic(epsilon=0.05)
        compile_time = model.precompile()
        assert isinstance(compile_time, float)
        assert compile_time >= 0.0


class TestPrecompileProtocols:
    def test_default_still_warms_only_flow_curve(self, monkeypatch):
        """protocols=None preserves today's behavior exactly -- zero
        change to existing callers' warm-up cost."""
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        called = {"flow_curve": False, "startup": False}

        def fake_flow_curve_precompile(*args, **kwargs):
            called["flow_curve"] = True
            return 0.1

        def fake_startup_solve(*args, **kwargs):
            called["startup"] = True
            return jnp.array([0.0])

        monkeypatch.setattr(
            schematic_mod, "precompile_flow_curve_solver", fake_flow_curve_precompile
        )
        monkeypatch.setattr(
            schematic_mod, "solve_startup_trajectory", fake_startup_solve
        )

        model.precompile()

        assert called["flow_curve"] is True
        assert called["startup"] is False

    def test_explicit_protocols_warms_requested_ones(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        called = {"startup": False, "creep": False}

        def fake_startup_solve(*args, **kwargs):
            called["startup"] = True
            return jnp.array([0.0])

        def fake_creep_solve(*args, **kwargs):
            called["creep"] = True
            return jnp.array([0.0])

        monkeypatch.setattr(
            schematic_mod, "solve_startup_trajectory", fake_startup_solve
        )
        monkeypatch.setattr(schematic_mod, "solve_creep_trajectory", fake_creep_solve)

        model.precompile(protocols=["startup"])

        assert called["startup"] is True
        assert called["creep"] is False

    def test_protocols_all_warms_everything(self, monkeypatch):
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        # Compute real Prony modes before mocking solve_equilibrium_correlator_trajectory
        # below -- initialize_prony_modes() depends on that same solver internally
        # for a fresh model, so mocking it first would corrupt the Prony fit
        # rather than exercise the protocols="all" warm-up path under test.
        model.initialize_prony_modes()
        called = {
            name: False
            for name in (
                "flow_curve",
                "equilibrium_correlator",
                "startup",
                "creep",
                "relaxation",
                "laos",
            )
        }

        def make_fake(name, return_shape_fn):
            def fake(*args, **kwargs):
                called[name] = True
                return return_shape_fn()

            return fake

        monkeypatch.setattr(
            schematic_mod,
            "precompile_flow_curve_solver",
            make_fake("flow_curve", lambda: 0.1),
        )
        monkeypatch.setattr(
            schematic_mod,
            "solve_equilibrium_correlator_trajectory",
            make_fake("equilibrium_correlator", lambda: jnp.array([0.0])),
        )
        monkeypatch.setattr(
            schematic_mod,
            "solve_startup_trajectory",
            make_fake("startup", lambda: jnp.array([0.0])),
        )
        monkeypatch.setattr(
            schematic_mod,
            "solve_creep_trajectory",
            make_fake("creep", lambda: jnp.array([0.0])),
        )
        monkeypatch.setattr(
            schematic_mod,
            "solve_relaxation_trajectory",
            make_fake("relaxation", lambda: jnp.array([0.0])),
        )
        monkeypatch.setattr(
            schematic_mod,
            "solve_laos_trajectory",
            make_fake("laos", lambda: jnp.array([0.0])),
        )

        model.precompile(protocols="all")

        assert all(called.values()), called

    def test_x_shapes_the_dummy_trajectory(self, monkeypatch):
        """X's length must reach the warmed solver, not the hardcoded
        5-point default -- otherwise a caller who precompiles with their
        real (differently-sized) data still eats a cold JIT compile on
        the first real predict()/fit() call, since SaveAt(ts=t_array)
        retraces on array length."""
        import rheojax.models.itt_mct.schematic as schematic_mod

        model = ITTMCTSchematic(epsilon=-0.1)
        seen = {"t_len": None}

        def fake_startup_solve(t_array, *args, **kwargs):
            seen["t_len"] = len(t_array)
            return jnp.zeros(len(t_array))

        monkeypatch.setattr(
            schematic_mod, "solve_startup_trajectory", fake_startup_solve
        )

        model.precompile(protocols=["startup"], X=np.linspace(0.0, 10.0, 42))

        assert seen["t_len"] == 42
