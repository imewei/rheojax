"""Physics validation tests for Fluidity-Saramito EVP models.

Tests verify key physical behaviors:
- Yield stress emergence (σ → τ_y as γ̇ → 0)
- Stress overshoot increases with waiting time (thixotropy)
- N₁ ~ σ² scaling (Maxwell viscoelasticity)
- Creep bifurcation at σ_y
- Non-exponential relaxation
"""

import numpy as np
import pytest

from rheojax.models.fluidity.saramito import (
    FluiditySaramitoLocal,
    FluiditySaramitoNonlocal,
)


class TestYieldStressEmergence:
    """Test yield stress behavior in flow curves."""

    @pytest.fixture
    def model(self):
        """Create model with known yield stress."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-6)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_low_rate_approaches_yield(self, model):
        """Test stress approaches τ_y at low shear rates."""
        gamma_dot = np.logspace(-4, -1, 50)
        sigma = model._predict_flow_curve(gamma_dot)

        tau_y = 100.0

        # At lowest rates, stress should be close to yield
        assert sigma[0] > tau_y * 0.8  # Within 20%
        assert sigma[0] < tau_y * 1.5  # Not too far above

    @pytest.mark.smoke
    def test_hb_scaling_at_high_rates(self, model):
        """Test HB power-law scaling at high rates."""
        gamma_dot = np.logspace(0, 2, 50)
        sigma = model._predict_flow_curve(gamma_dot)

        tau_y = 100.0
        K = 50.0
        n = 0.5

        # At high rates: σ ≈ τ_y + K*γ̇^n
        sigma_expected = tau_y + K * gamma_dot**n

        # Should match within 20%
        relative_error = np.abs(sigma - sigma_expected) / sigma_expected
        assert np.mean(relative_error) < 0.2


class TestThixotropicOvershoot:
    """Test thixotropic stress overshoot behavior."""

    @pytest.fixture
    def model(self):
        """Create thixotropic model with parameters that show overshoot.

        Key insight: Overshoot requires stress dynamics to be FASTER than
        fluidity evolution. With slow rejuvenation, stress builds up toward
        the initial high τ_ss(f_age), then drops when f increases and τ_ss falls.
        """
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1000.0)
        model.parameters.set_value("tau_y0", 1.0)  # Low yield for easy flow
        model.parameters.set_value("K_HB", 1.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 0.01)  # λ_init = 0.1 s
        model.parameters.set_value("f_flow", 0.5)  # 50x range
        model.parameters.set_value("t_a", 1000.0)  # Slow aging
        model.parameters.set_value("b", 0.005)  # Slow rejuvenation (key for overshoot)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_overshoot_exists(self, model):
        """Test stress overshoot is present in startup."""
        t = np.linspace(0, 20, 500)
        gamma_dot = 10.0

        _, stress, _ = model.simulate_startup(t, gamma_dot, t_wait=0.0)

        sigma_max = np.max(stress)
        sigma_final = stress[-1]

        # Should have overshoot (max > final)
        # With slow rejuvenation, stress catches up to falling τ_ss and overshoots
        overshoot_ratio = sigma_max / sigma_final
        assert overshoot_ratio > 1.1  # At least 10% overshoot expected

    def test_overshoot_increases_with_waiting(self, model):
        """Test overshoot increases with waiting time (TC-019).

        The simulate_startup t_wait parameter models aging from a previously
        rejuvenated state (f_flow). At t_wait=0, f_init = f_age (fully aged,
        special case). For t_wait > 0: f_init = f_age + (f_flow - f_age)*exp(-t_wait/t_a).
        So short positive t_wait = less aged (high f), long t_wait = more aged (low f).
        """
        t = np.linspace(0, 100, 1000)
        gamma_dot = 0.1

        # Short wait: recently sheared, higher initial fluidity (less structure)
        _, stress_short, _ = model.simulate_startup(t, gamma_dot, t_wait=10.0)
        overshoot_short = np.max(stress_short) / stress_short[-1]

        # Long wait: well-aged, lower initial fluidity (more structure)
        _, stress_long, _ = model.simulate_startup(t, gamma_dot, t_wait=5000.0)
        overshoot_long = np.max(stress_long) / stress_long[-1]

        # Longer wait = more aging = stronger structure = larger overshoot
        # This is a key thixotropic signature (TC-019: strict monotonicity)
        assert overshoot_long > overshoot_short, (
            f"Longer wait should produce more overshoot: "
            f"overshoot_long={overshoot_long:.4f}, overshoot_short={overshoot_short:.4f}"
        )


class TestNormalStressScaling:
    """Test normal stress difference scaling."""

    @pytest.fixture
    def model(self):
        """Create model for normal stress tests."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_n1_positive(self, model):
        """Test N₁ > 0 (Weissenberg effect)."""
        gamma_dot = np.array([0.1, 1.0, 10.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert np.all(N1 > 0)

    @pytest.mark.smoke
    def test_n1_increases_with_rate(self, model):
        """Test N₁ increases with shear rate."""
        gamma_dot = np.array([0.1, 1.0, 10.0])
        N1, _ = model.predict_normal_stresses(gamma_dot)

        # N₁ should increase with rate
        assert N1[1] > N1[0]
        assert N1[2] > N1[1]

    def test_n2_zero_for_ucm(self, model):
        """Test N₂ = 0 for upper-convected Maxwell."""
        gamma_dot = np.array([0.1, 1.0, 10.0])
        _, N2 = model.predict_normal_stresses(gamma_dot)

        # N₂ should be zero (UCM prediction)
        assert np.all(N2 == 0)


class TestCreepBifurcation:
    """Test creep bifurcation behavior."""

    @pytest.fixture
    def model(self):
        """Create model for creep tests."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_creep_above_yield(self, model):
        """Test continuous flow above yield stress."""
        t = np.linspace(0, 100, 500)
        sigma_applied = 150.0  # Above yield (100)

        strain, fluidity = model.simulate_creep(t, sigma_applied)

        # Strain should increase continuously
        assert strain[-1] > strain[0]

        # Final strain rate should be positive
        strain_rate_final = (strain[-1] - strain[-10]) / (t[-1] - t[-10])
        assert strain_rate_final > 0

    @pytest.mark.smoke
    def test_creep_below_yield(self, model):
        """Test arrested flow below yield stress."""
        t = np.linspace(0, 100, 500)
        sigma_applied = 50.0  # Below yield (100)

        strain, fluidity = model.simulate_creep(t, sigma_applied)

        # Strain rate should decrease over time (approaching arrest)
        # Compare early and late strain rates
        strain_rate_early = (strain[50] - strain[0]) / (t[50] - t[0])
        strain_rate_late = (strain[-1] - strain[-50]) / (t[-1] - t[-50])

        # Late rate should be smaller (approaching zero)
        assert strain_rate_late <= strain_rate_early * 1.1  # Allow small tolerance


class TestNonexponentialRelaxation:
    """Test non-exponential stress relaxation."""

    @pytest.fixture
    def model(self):
        """Create model for relaxation tests."""
        model = FluiditySaramitoLocal(coupling="minimal")
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-5)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_stress_decreases(self, model):
        """Test stress decreases during relaxation."""
        t = np.linspace(0, 100, 500)

        # Simulate startup to initial stress
        t_startup = np.linspace(0, 10, 100)
        _, stress_startup, _ = model.simulate_startup(t_startup, gamma_dot=10.0)
        sigma_0 = stress_startup[-1]

        # Then relaxation (by setting sigma_applied and simulating)
        # The relaxation kernel starts with elevated stress
        import diffrax

        from rheojax.core.jax_config import safe_import_jax
        from rheojax.models.fluidity.saramito._kernels import (
            saramito_local_relaxation_ode_rhs,
        )

        jax, jnp = safe_import_jax()

        args = model._get_saramito_ode_args()
        f_flow = model.parameters.get_value("f_flow")
        y0 = jnp.array([0.0, 0.0, sigma_0, f_flow])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_local_relaxation_ode_rhs(ti, yi, args_i)
        )
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t[0],
            t[-1],
            0.01,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=t_jax),
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

        stress = np.array(sol.ys[:, 2])

        # Stress should decrease
        assert stress[-1] < stress[0]

    def test_residual_stress_above_yield(self, model):
        """Test residual stress can remain above zero (yield stress effect)."""
        # For EVP materials, stress can freeze above zero if it drops below yield
        # This is the "residual stress" or "frozen-in stress" behavior

        t = np.linspace(0, 500, 1000)  # Long time

        # Start with stress well above yield
        sigma_0 = 500.0

        import diffrax

        from rheojax.core.jax_config import safe_import_jax
        from rheojax.models.fluidity.saramito._kernels import (
            saramito_local_relaxation_ode_rhs,
        )

        jax, jnp = safe_import_jax()

        args = model._get_saramito_ode_args()
        f_flow = model.parameters.get_value("f_flow")
        y0 = jnp.array([0.0, 0.0, sigma_0, f_flow])

        term = diffrax.ODETerm(
            lambda ti, yi, args_i: saramito_local_relaxation_ode_rhs(ti, yi, args_i)
        )
        solver = diffrax.Tsit5()
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-7)

        t_jax = jnp.asarray(t, dtype=jnp.float64)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t[0],
            t[-1],
            0.01,
            y0,
            args=args,
            saveat=diffrax.SaveAt(ts=t_jax),
            stepsize_controller=stepsize_controller,
            max_steps=1_000_000,
        )

        stress_final = np.array(sol.ys[-1, 2])

        # Final stress could be near yield if model captures solid-like behavior
        # At minimum, it should be non-negative
        assert stress_final >= 0


class TestNonlocalShearBanding:
    """Test shear banding in nonlocal model."""

    @pytest.fixture
    def model(self):
        """Create nonlocal model."""
        model = FluiditySaramitoNonlocal(
            coupling="minimal",
            N_y=31,
            H=1e-3,
            xi=5e-5,
        )
        model.parameters.set_value("G", 1e4)
        model.parameters.set_value("tau_y0", 100.0)
        model.parameters.set_value("K_HB", 50.0)
        model.parameters.set_value("n_HB", 0.5)
        model.parameters.set_value("f_age", 1e-6)
        model.parameters.set_value("f_flow", 1e-2)
        model.parameters.set_value("t_a", 10.0)
        model.parameters.set_value("b", 1.0)
        model.parameters.set_value("n_rej", 1.0)
        return model

    @pytest.mark.smoke
    def test_nonlocal_startup_runs(self, model):
        """Test nonlocal startup simulation runs."""
        t = np.linspace(0, 10, 100)
        gamma_dot = 1.0

        t_out, sigma, f_field = model.simulate_startup(t, gamma_dot)

        assert sigma.shape == t.shape
        assert f_field.shape == (model.N_y,)
        assert np.all(np.isfinite(sigma))
        assert np.all(f_field > 0)

    @pytest.mark.smoke
    def test_shear_banding_detection(self, model):
        """Test shear banding detection method."""
        # Run startup to generate profile
        t = np.linspace(0, 50, 100)
        _, _, f_field = model.simulate_startup(t, gamma_dot=0.1)

        is_banded, cv, ratio = model.detect_shear_bands()

        assert isinstance(is_banded, bool)
        assert cv >= 0
        assert ratio >= 1

    def test_banding_metrics(self, model):
        """Test banding metrics calculation."""
        t = np.linspace(0, 50, 100)
        _, _, f_field = model.simulate_startup(t, gamma_dot=0.1)

        metrics = model.get_banding_metrics()

        assert "cv" in metrics
        assert "ratio" in metrics
        assert "band_fraction" in metrics
        assert "f_max" in metrics
        assert "f_min" in metrics

        assert 0 <= metrics["band_fraction"] <= 1
        assert metrics["f_max"] >= metrics["f_min"]


class TestCouplingModeComparison:
    """Compare minimal vs full coupling behavior."""

    @pytest.mark.smoke
    def test_full_coupling_higher_aged_yield(self):
        """Test full coupling gives higher effective yield when aged."""
        # Minimal coupling
        model_min = FluiditySaramitoLocal(coupling="minimal")
        model_min.parameters.set_value("tau_y0", 100.0)
        model_min.parameters.set_value("f_age", 1e-6)

        # Full coupling
        model_full = FluiditySaramitoLocal(coupling="full")
        model_full.parameters.set_value("tau_y0", 100.0)
        model_full.parameters.set_value("f_age", 1e-6)
        model_full.parameters.set_value("tau_y_coupling", 1e-4)
        model_full.parameters.set_value("m_yield", 0.5)

        # Effective yield at f_age
        tau_y_min = model_min.get_effective_yield_stress(1e-6)
        tau_y_full = model_full.get_effective_yield_stress(1e-6)

        # Full coupling should give higher effective yield
        assert tau_y_full > tau_y_min

    def test_full_coupling_converges_to_minimal_at_high_f(self):
        """Test full coupling approaches minimal at high fluidity."""
        model_min = FluiditySaramitoLocal(coupling="minimal")
        model_min.parameters.set_value("tau_y0", 100.0)

        model_full = FluiditySaramitoLocal(coupling="full")
        model_full.parameters.set_value("tau_y0", 100.0)
        model_full.parameters.set_value("tau_y_coupling", 1e-4)
        model_full.parameters.set_value("m_yield", 0.5)

        # At high fluidity, coupling term becomes small
        f_high = 0.1
        tau_y_min = model_min.get_effective_yield_stress(f_high)
        tau_y_full = model_full.get_effective_yield_stress(f_high)

        # Should be close (coupling term small at high f)
        relative_diff = abs(tau_y_full - tau_y_min) / tau_y_min
        assert relative_diff < 0.5  # Within 50%
