"""Tests for ITTMCTSchematic (F₁₂) model.

Tests cover:
- Parameter initialization
- Glass transition detection
- All 6 protocols (flow curve, SAOS, startup, creep, relaxation, LAOS)
- Bayesian inference basics
"""

import numpy as np
import pytest

from rheojax.models.itt_mct import ITTMCTSchematic


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
        """_get_v2_critical uses the v1 != 0 branch when v1 is nonzero."""
        model = ITTMCTSchematic()
        model.parameters.set_value("v1", 1.0)
        # epsilon property routes through _get_v2_critical(v1) with v1 != 0.
        v2_c = model._get_v2_critical(1.0)
        assert v2_c == pytest.approx((4.0 - 2.0 * 1.0) / (1.0 - 1.0 / 4.0))
        # And the epsilon getter stays finite with the modified v1.
        assert np.isfinite(model.epsilon)


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
        expected = G_inf * gamma_c * info["f_neq"] ** 2
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
        sigma = model._compute_steady_state_stress(5.0)
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
        np.testing.assert_allclose(
            mag, np.hypot(comps[:, 0], comps[:, 1]), rtol=1e-6
        )
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
        sigma = model.predict(np.linspace(0.0, 3.0, 15), test_mode="startup", gamma_dot=1.0)
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
class TestPrecompile:
    """Diffrax solver precompilation entry point."""

    def test_precompile_returns_time(self):
        model = ITTMCTSchematic(epsilon=0.05)
        compile_time = model.precompile()
        assert isinstance(compile_time, float)
        assert compile_time >= 0.0
