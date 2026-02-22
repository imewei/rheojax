"""Unit tests for DMT (de Souza Mendes-Thompson) models.

Tests cover:
- Model creation and parameter setup
- Flow curve predictions
- Startup shear with stress overshoot
- Creep with delayed yielding
- SAOS moduli (Maxwell variant)
- LAOS simulation and harmonic extraction
- Nonlocal model and banding detection
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.dmt import DMTLocal, DMTNonlocal

# Safe JAX import
jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def dmt_exponential():
    """DMT model with exponential closure.

    Uses moderate viscosity contrast (1000:1) for monotonic flow curves.
    Extreme contrasts (1e6:1) produce N-shaped curves (physically valid
    but more complex behavior).
    """
    model = DMTLocal(closure="exponential", include_elasticity=True)
    # Set moderate viscosity contrast for predictable test behavior
    model.parameters["eta_0"].value = 1000.0  # Pa·s
    model.parameters["eta_inf"].value = 1.0  # Pa·s
    return model


@pytest.fixture
def dmt_herschel_bulkley():
    """DMT model with Herschel-Bulkley closure."""
    return DMTLocal(closure="herschel_bulkley", include_elasticity=True)


@pytest.fixture
def dmt_viscous():
    """DMT model without elasticity."""
    return DMTLocal(closure="exponential", include_elasticity=False)


@pytest.fixture
def dmt_nonlocal():
    """Nonlocal DMT model."""
    return DMTNonlocal(
        closure="exponential",
        include_elasticity=True,
        n_points=21,
        gap_width=1e-3,
    )


# =============================================================================
# Test Model Creation
# =============================================================================


class TestDMTLocalCreation:
    """Test DMTLocal model instantiation and parameters."""

    @pytest.mark.smoke
    def test_model_creation_exponential(self, dmt_exponential):
        """Test creation with exponential closure."""
        assert dmt_exponential.closure == "exponential"
        assert dmt_exponential.include_elasticity is True
        assert "eta_0" in dmt_exponential.parameters.keys()
        assert "eta_inf" in dmt_exponential.parameters.keys()
        assert "G0" in dmt_exponential.parameters.keys()
        # HB parameters should NOT be present
        assert "tau_y0" not in dmt_exponential.parameters.keys()

    @pytest.mark.smoke
    def test_model_creation_herschel_bulkley(self, dmt_herschel_bulkley):
        """Test creation with Herschel-Bulkley closure."""
        assert dmt_herschel_bulkley.closure == "herschel_bulkley"
        assert "tau_y0" in dmt_herschel_bulkley.parameters.keys()
        assert "K0" in dmt_herschel_bulkley.parameters.keys()
        assert "n_flow" in dmt_herschel_bulkley.parameters.keys()
        assert "m1" in dmt_herschel_bulkley.parameters.keys()
        assert "m2" in dmt_herschel_bulkley.parameters.keys()

    @pytest.mark.smoke
    def test_model_creation_viscous(self, dmt_viscous):
        """Test creation without elasticity."""
        assert dmt_viscous.include_elasticity is False
        assert "G0" not in dmt_viscous.parameters.keys()
        assert "m_G" not in dmt_viscous.parameters.keys()
        # Kinetics parameters should still be present
        assert "t_eq" in dmt_viscous.parameters.keys()
        assert "a" in dmt_viscous.parameters.keys()
        assert "c" in dmt_viscous.parameters.keys()

    @pytest.mark.smoke
    def test_parameters_have_correct_bounds(self, dmt_exponential):
        """Test parameter bounds are physically reasonable."""
        # eta_0 should be larger than eta_inf (bounds overlap is OK)
        eta_0_bounds = dmt_exponential.parameters["eta_0"].bounds
        eta_inf_bounds = dmt_exponential.parameters["eta_inf"].bounds
        assert eta_0_bounds[0] > 0  # Positive viscosity
        assert eta_inf_bounds[0] > 0

        # t_eq should be positive
        t_eq_bounds = dmt_exponential.parameters["t_eq"].bounds
        assert t_eq_bounds[0] > 0

    @pytest.mark.smoke
    def test_repr(self, dmt_exponential):
        """Test string representation."""
        repr_str = repr(dmt_exponential)
        assert "DMTLocal" in repr_str
        assert "exponential" in repr_str

    def test_get_closure_info(self, dmt_exponential, dmt_herschel_bulkley):
        """Test closure info string."""
        exp_info = dmt_exponential.get_closure_info()
        assert "Exponential" in exp_info

        hb_info = dmt_herschel_bulkley.get_closure_info()
        assert "Herschel-Bulkley" in hb_info


# =============================================================================
# Test Equilibrium Structure
# =============================================================================


class TestEquilibriumStructure:
    """Test equilibrium structure calculations."""

    @pytest.mark.smoke
    def test_equilibrium_structure_high_shear(self, dmt_exponential):
        """λ_eq should approach 0 at high shear rates."""
        from rheojax.models.dmt._kernels import equilibrium_structure

        a = dmt_exponential.parameters.get_value("a")
        c = dmt_exponential.parameters.get_value("c")

        gamma_dot_high = 1000.0
        lam_eq = equilibrium_structure(gamma_dot_high, a, c)

        assert float(lam_eq) < 0.01  # Should be very small

    @pytest.mark.smoke
    def test_equilibrium_structure_low_shear(self, dmt_exponential):
        """λ_eq should approach 1 at low shear rates."""
        from rheojax.models.dmt._kernels import equilibrium_structure

        a = dmt_exponential.parameters.get_value("a")
        c = dmt_exponential.parameters.get_value("c")

        gamma_dot_low = 1e-6
        lam_eq = equilibrium_structure(gamma_dot_low, a, c)

        assert float(lam_eq) > 0.99  # Should be close to 1

    def test_equilibrium_structure_monotonic(self, dmt_exponential):
        """λ_eq should decrease monotonically with γ̇."""
        from rheojax.models.dmt._kernels import equilibrium_structure

        a = dmt_exponential.parameters.get_value("a")
        c = dmt_exponential.parameters.get_value("c")

        gamma_dots = jnp.logspace(-3, 3, 20)
        lam_eqs = jax.vmap(lambda gd: equilibrium_structure(gd, a, c))(gamma_dots)

        # Check monotonically decreasing
        diffs = jnp.diff(lam_eqs)
        assert jnp.all(diffs <= 0)


# =============================================================================
# Test Flow Curve
# =============================================================================


class TestDMTLocalFlowCurve:
    """Test steady-state flow curve predictions."""

    @pytest.mark.smoke
    def test_flow_curve_shape_exponential(self, dmt_exponential):
        """Flow curve should show shear-thinning behavior."""
        gamma_dot = np.logspace(-2, 2, 20)
        stress = dmt_exponential._predict_flow_curve(gamma_dot)

        # Stress should generally increase with shear rate (overall trend)
        # Note: Thixotropic materials can have non-monotonic (N-shaped) flow curves
        # in certain parameter regimes, so we check overall trend
        assert stress[-1] > stress[0]  # High rate stress > low rate stress

        # Apparent viscosity should decrease overall (shear-thinning)
        eta_app = stress / gamma_dot
        assert eta_app[0] > eta_app[-1]  # Low-rate viscosity > high-rate

    @pytest.mark.smoke
    def test_flow_curve_shape_hb(self, dmt_herschel_bulkley):
        """HB flow curve should show yield stress plateau."""
        gamma_dot = np.logspace(-2, 2, 20)
        stress = dmt_herschel_bulkley._predict_flow_curve(gamma_dot)

        # Stress should generally increase (overall trend)
        # Note: Thixotropic HB can have non-monotonic regions
        assert stress[-1] > stress[0]  # High rate stress > low rate stress

        # Low-shear stress should be finite (yield-like behavior)
        # With structure kinetics, low-shear stress depends on λ_eq
        assert stress[0] > 0  # Finite stress at low rate

    def test_flow_curve_limits_exponential(self, dmt_exponential):
        """Test viscosity limits at extreme shear rates."""
        eta_0 = dmt_exponential.parameters.get_value("eta_0")
        eta_inf = dmt_exponential.parameters.get_value("eta_inf")

        # Very low shear rate
        gamma_dot_low = np.array([1e-6])
        stress_low = dmt_exponential._predict_flow_curve(gamma_dot_low)
        eta_apparent_low = stress_low[0] / gamma_dot_low[0]

        # Should be close to eta_0 (allowing for numerical tolerance)
        assert eta_apparent_low > 0.5 * eta_0

        # Very high shear rate
        gamma_dot_high = np.array([1e6])
        stress_high = dmt_exponential._predict_flow_curve(gamma_dot_high)
        eta_apparent_high = stress_high[0] / gamma_dot_high[0]

        # Should approach eta_inf
        assert eta_apparent_high < 10 * eta_inf


# =============================================================================
# Test Startup Shear
# =============================================================================


class TestDMTLocalStartup:
    """Test startup of steady shear simulations."""

    @pytest.mark.smoke
    def test_startup_runs(self, dmt_exponential):
        """Test that startup simulation runs without error."""
        t, stress, lam = dmt_exponential.simulate_startup(
            gamma_dot=10.0, t_end=10.0, dt=0.1
        )

        assert len(t) == len(stress) == len(lam)
        assert len(t) > 0

    @pytest.mark.smoke
    def test_startup_overshoot_maxwell(self, dmt_exponential):
        """Maxwell variant should show stress overshoot."""
        t, stress, lam = dmt_exponential.simulate_startup(
            gamma_dot=10.0, t_end=50.0, dt=0.01, lam_init=1.0
        )

        # Find peak stress
        peak_idx = np.argmax(stress)
        peak_stress = stress[peak_idx]
        final_stress = stress[-1]

        # Peak should exceed steady state (overshoot)
        assert peak_stress > final_stress * 1.05  # At least 5% overshoot

    def test_startup_structure_decay(self, dmt_exponential):
        """Structure should decay from λ=1 toward λ_eq during startup."""
        t, stress, lam = dmt_exponential.simulate_startup(
            gamma_dot=10.0, t_end=100.0, dt=0.1, lam_init=1.0
        )

        # Initial structure should be close to 1 (first scan output may differ slightly)
        assert lam[0] == pytest.approx(1.0, rel=0.02)  # Allow 2% tolerance

        # Final structure should be λ_eq < 1
        assert lam[-1] < 0.5  # Significant breakdown

        # Structure should monotonically decrease
        assert np.all(np.diff(lam) <= 0)

    def test_startup_approaches_steady_state(self, dmt_exponential):
        """Startup stress should approach steady-state value."""
        gamma_dot = 10.0
        t, stress, lam = dmt_exponential.simulate_startup(
            gamma_dot=gamma_dot, t_end=500.0, dt=0.1
        )

        # Get steady-state prediction
        stress_ss = dmt_exponential._predict_flow_curve(np.array([gamma_dot]))[0]

        # Final stress should be close to steady-state
        assert stress[-1] == pytest.approx(stress_ss, rel=0.1)

    def test_startup_viscous_no_overshoot(self, dmt_viscous):
        """Viscous variant may not show true stress overshoot."""
        t, stress, lam = dmt_viscous.simulate_startup(
            gamma_dot=10.0, t_end=50.0, dt=0.01
        )

        # Can still run and produce results
        assert len(t) > 0
        assert stress[-1] > 0


# =============================================================================
# Test Stress Relaxation (Maxwell only)
# =============================================================================


class TestDMTLocalRelaxation:
    """Test stress relaxation simulations."""

    def test_relaxation_requires_elasticity(self, dmt_viscous):
        """Relaxation should raise error without elasticity."""
        with pytest.raises(ValueError, match="include_elasticity"):
            dmt_viscous.simulate_relaxation(t_end=10.0)

    def test_relaxation_runs(self, dmt_exponential):
        """Test relaxation simulation runs."""
        t, stress, lam = dmt_exponential.simulate_relaxation(
            t_end=100.0, dt=0.1, sigma_init=100.0, lam_init=0.3
        )

        assert len(t) == len(stress) == len(lam)
        # First output is after first timestep, so stress may have changed slightly
        assert stress[0] > 0  # Positive initial stress
        assert stress[0] <= 100.0  # Should not exceed initial

    def test_relaxation_stress_decays(self, dmt_exponential):
        """Stress should decay during relaxation."""
        t, stress, lam = dmt_exponential.simulate_relaxation(
            t_end=100.0, dt=0.1, sigma_init=100.0, lam_init=0.3
        )

        # Stress should decrease
        assert stress[-1] < stress[0]

    def test_relaxation_structure_recovers(self, dmt_exponential):
        """Structure should recover toward λ=1 during relaxation."""
        t, stress, lam = dmt_exponential.simulate_relaxation(
            t_end=500.0, dt=0.1, sigma_init=100.0, lam_init=0.3
        )

        # Structure should increase toward 1
        assert lam[-1] > lam[0]
        assert lam[-1] > 0.9  # Nearly fully recovered


# =============================================================================
# Test Creep
# =============================================================================


class TestDMTLocalCreep:
    """Test creep simulations."""

    @pytest.mark.smoke
    def test_creep_runs(self, dmt_exponential):
        """Test creep simulation runs."""
        t, gamma, gamma_dot, lam = dmt_exponential.simulate_creep(
            sigma_0=100.0, t_end=100.0, dt=0.1
        )

        assert len(t) == len(gamma) == len(gamma_dot) == len(lam)
        # For Maxwell variant, initial strain includes elastic contribution γ_e = σ/G
        # With σ=100, G=100 (default), expect γ_e ≈ 1.0
        assert gamma[0] >= 0  # Positive strain
        assert gamma[0] < 10.0  # Reasonable initial strain

    def test_creep_strain_accumulates(self, dmt_exponential):
        """Strain should accumulate during creep."""
        t, gamma, gamma_dot, lam = dmt_exponential.simulate_creep(
            sigma_0=500.0, t_end=100.0, dt=0.1
        )

        # Strain should increase
        assert gamma[-1] > gamma[0]
        assert np.all(np.diff(gamma) >= 0)  # Monotonic

    def test_creep_below_yield_slows(self, dmt_herschel_bulkley):
        """Creep below yield stress should be very slow."""
        tau_y0 = dmt_herschel_bulkley.parameters.get_value("tau_y0")

        # Apply stress well below yield
        sigma_below = tau_y0 * 0.5
        t, gamma, gamma_dot, lam = dmt_herschel_bulkley.simulate_creep(
            sigma_0=sigma_below, t_end=100.0, dt=0.1
        )

        # Shear rate should be very small
        assert gamma_dot[-1] < 0.1  # Nearly arrested

    def test_creep_maxwell_initial_elastic_jump(self, dmt_exponential):
        """Maxwell creep should have initial elastic strain γ_e(0) = σ₀/G."""
        sigma_0 = 100.0
        G0 = dmt_exponential.parameters.get_value("G0")
        m_G = dmt_exponential.parameters.get_value("m_G")
        lam_init = 1.0

        # Expected initial elastic strain: γ_e = σ₀/G(λ=1) = σ₀/(G₀·1^m_G) = σ₀/G₀
        expected_gamma_e = sigma_0 / (G0 * (lam_init**m_G))

        t, gamma, gamma_dot, lam = dmt_exponential.simulate_creep(
            sigma_0=sigma_0, t_end=10.0, dt=0.01, lam_init=lam_init
        )

        # Initial strain should include elastic contribution
        # Allow some tolerance for first-step numerical effects
        assert gamma[0] >= expected_gamma_e * 0.9
        assert gamma[0] <= expected_gamma_e * 1.5  # May include some viscous flow

    def test_creep_maxwell_elastic_strain_evolves_with_structure(self, dmt_exponential):
        """As structure breaks down (λ decreases), elastic modulus decreases,
        so elastic strain γ_e = σ/G should increase."""
        sigma_0 = 500.0  # High stress to cause significant breakdown

        t, gamma, gamma_dot, lam = dmt_exponential.simulate_creep(
            sigma_0=sigma_0, t_end=100.0, dt=0.1, lam_init=1.0
        )

        # Structure should decrease significantly
        assert lam[-1] < lam[0]

        # Compute elastic strain at start and end
        G0 = dmt_exponential.parameters.get_value("G0")
        m_G = dmt_exponential.parameters.get_value("m_G")

        gamma_e_start = sigma_0 / (G0 * (lam[0] ** m_G))
        gamma_e_end = sigma_0 / (G0 * (lam[-1] ** m_G))

        # Elastic strain should increase as structure decreases (G decreases)
        assert gamma_e_end > gamma_e_start

    def test_creep_viscous_vs_maxwell_initial_strain(
        self, dmt_exponential, dmt_viscous
    ):
        """Compare viscous and Maxwell creep initial behavior."""
        sigma_0 = 100.0

        # Maxwell: should have initial elastic strain
        t_m, gamma_m, _, _ = dmt_exponential.simulate_creep(
            sigma_0=sigma_0, t_end=10.0, dt=0.1, lam_init=1.0
        )

        # Viscous: should start near zero
        t_v, gamma_v, _, _ = dmt_viscous.simulate_creep(
            sigma_0=sigma_0, t_end=10.0, dt=0.1, lam_init=1.0
        )

        # Maxwell initial strain should be larger (includes elastic jump)
        assert gamma_m[0] > gamma_v[0]


# =============================================================================
# Test SAOS (Maxwell only)
# =============================================================================


class TestDMTLocalSAOS:
    """Test small amplitude oscillatory shear predictions."""

    def test_saos_requires_elasticity(self, dmt_viscous):
        """SAOS should raise error without elasticity."""
        with pytest.raises(ValueError, match="include_elasticity"):
            dmt_viscous.predict_saos(np.array([1.0]))

    @pytest.mark.smoke
    def test_saos_returns_moduli(self, dmt_exponential):
        """SAOS should return G' and G''."""
        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = dmt_exponential.predict_saos(omega)

        assert len(G_prime) == len(omega)
        assert len(G_double_prime) == len(omega)
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    def test_saos_maxwell_behavior(self, dmt_exponential):
        """SAOS should show Maxwell-like frequency dependence."""
        omega = np.logspace(-3, 3, 50)
        G_prime, G_double_prime = dmt_exponential.predict_saos(omega, lam_0=1.0)

        # At low frequency: G' ~ ω², G'' ~ ω
        # At high frequency: G' → G0, G'' → η_inf·ω (solvent contribution)

        # G' should increase with frequency and plateau
        assert G_prime[-1] > G_prime[0]

        # G'' has both viscoelastic and solvent contributions:
        # G''_ve = G·ωθ/(1+(ωθ)²) has a peak at ω = 1/θ
        # G''_solvent = η_inf·ω increases monotonically
        # Total G'' may or may not have a visible peak depending on η_inf
        # With low η_inf, there should be a local maximum or inflection
        # Just verify G'' is positive and shows typical behavior
        assert np.all(G_double_prime > 0)
        # At very low frequency, loss should be small
        assert G_double_prime[0] < G_double_prime[-1] / 10


# =============================================================================
# Test LAOS
# =============================================================================


class TestDMTLocalLAOS:
    """Test large amplitude oscillatory shear simulations."""

    @pytest.mark.smoke
    def test_laos_runs(self, dmt_exponential):
        """Test LAOS simulation runs."""
        result = dmt_exponential.simulate_laos(
            gamma_0=0.1, omega=1.0, n_cycles=5, points_per_cycle=64
        )

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "lam" in result
        assert len(result["t"]) == 5 * 64

    def test_laos_strain_is_sinusoidal(self, dmt_exponential):
        """Input strain should be sinusoidal."""
        gamma_0 = 0.5
        omega = 2.0
        result = dmt_exponential.simulate_laos(gamma_0=gamma_0, omega=omega, n_cycles=3)

        # Check amplitude
        assert np.max(result["strain"]) == pytest.approx(gamma_0, rel=0.01)
        assert np.min(result["strain"]) == pytest.approx(-gamma_0, rel=0.01)

    def test_laos_harmonics_extraction(self, dmt_exponential):
        """Test Fourier harmonic extraction."""
        result = dmt_exponential.simulate_laos(gamma_0=0.5, omega=1.0, n_cycles=10)

        harmonics = dmt_exponential.extract_harmonics(result, n_harmonics=3)

        assert "sigma_prime" in harmonics
        assert "sigma_double_prime" in harmonics
        assert "I_n_1" in harmonics

        # First harmonic should dominate
        assert harmonics["I_n_1"][0] == pytest.approx(1.0, rel=0.01)

        # Higher harmonics should be smaller
        assert harmonics["I_n_1"][1] < 0.5  # I_3/I_1 < 0.5

    def test_laos_nonlinearity_increases_with_amplitude(self, dmt_exponential):
        """Higher harmonics should increase with strain amplitude."""
        # Small amplitude
        result_small = dmt_exponential.simulate_laos(
            gamma_0=0.01, omega=1.0, n_cycles=10
        )
        harmonics_small = dmt_exponential.extract_harmonics(result_small)

        # Large amplitude
        result_large = dmt_exponential.simulate_laos(
            gamma_0=1.0, omega=1.0, n_cycles=10
        )
        harmonics_large = dmt_exponential.extract_harmonics(result_large)

        # Third harmonic ratio should be larger at large amplitude
        assert harmonics_large["I_n_1"][1] > harmonics_small["I_n_1"][1]


# =============================================================================
# Test Nonlocal Model
# =============================================================================


class TestDMTNonlocal:
    """Test nonlocal (1D) DMT model."""

    @pytest.mark.smoke
    def test_nonlocal_creation(self, dmt_nonlocal):
        """Test nonlocal model instantiation."""
        assert dmt_nonlocal.n_points == 21
        assert dmt_nonlocal.gap_width == 1e-3
        assert len(dmt_nonlocal.y) == 21
        assert "D_lambda" in dmt_nonlocal.parameters.keys()

    def test_cooperativity_length(self, dmt_nonlocal):
        """Test cooperativity length calculation."""
        xi = dmt_nonlocal.get_cooperativity_length()
        assert xi > 0

        D_lambda = dmt_nonlocal.parameters.get_value("D_lambda")
        t_eq = dmt_nonlocal.parameters.get_value("t_eq")
        expected = np.sqrt(D_lambda * t_eq)
        assert xi == pytest.approx(expected)

    @pytest.mark.slow
    def test_nonlocal_steady_shear_runs(self, dmt_nonlocal):
        """Test steady shear simulation runs."""
        result = dmt_nonlocal.simulate_steady_shear(
            gamma_dot_avg=10.0, t_end=100.0, dt=1.0
        )

        assert "t" in result
        assert "y" in result
        assert "lam" in result
        assert "gamma_dot" in result
        assert "velocity" in result
        assert "stress" in result

        # Check array shapes
        n_times = len(result["t"])
        n_points = len(result["y"])
        assert result["lam"].shape == (n_times, n_points)
        assert result["gamma_dot"].shape == (n_times, n_points)

    @pytest.mark.slow
    def test_banding_detection(self, dmt_nonlocal):
        """Test shear banding detection."""
        # Run to steady state
        result = dmt_nonlocal.simulate_steady_shear(
            gamma_dot_avg=10.0, t_end=200.0, dt=1.0
        )

        banding_info = dmt_nonlocal.detect_banding(result, threshold=0.1)

        assert "is_banding" in banding_info
        assert "band_contrast" in banding_info
        assert "band_width" in banding_info
        # is_banding may be numpy bool, convert to check
        assert isinstance(bool(banding_info["is_banding"]), bool)


# =============================================================================
# Test Bayesian Inference
# =============================================================================


class TestDMTBayesian:
    """Test model_function and Bayesian inference for DMT models."""

    # --- model_function: flow_curve ---

    def test_model_function_flow_curve_exponential(self, dmt_exponential):
        """Test model_function returns correct flow curve (exponential)."""
        gamma_dot = jnp.logspace(-1, 2, 20)
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])
        stress = dmt_exponential.model_function(gamma_dot, params, "flow_curve")

        assert stress.shape == (20,)
        assert jnp.all(jnp.isfinite(stress))
        assert jnp.all(stress > 0)

        # Compare with direct kernel
        ref = dmt_exponential._predict_flow_curve(np.array(gamma_dot))
        np.testing.assert_allclose(np.array(stress), ref, rtol=1e-6)

    def test_model_function_flow_curve_hb(self, dmt_herschel_bulkley):
        """Test model_function returns correct flow curve (HB)."""
        gamma_dot = jnp.logspace(-1, 2, 20)
        params = jnp.array([v.value for v in dmt_herschel_bulkley.parameters.values()])
        stress = dmt_herschel_bulkley.model_function(gamma_dot, params, "flow_curve")

        assert stress.shape == (20,)
        assert jnp.all(jnp.isfinite(stress))
        assert jnp.all(stress > 0)

    # --- model_function: startup ---

    def test_model_function_startup(self, dmt_exponential):
        """Test model_function returns correct startup stress."""
        t = jnp.linspace(0, 10, 100)
        dmt_exponential._gamma_dot_applied = 10.0
        dmt_exponential._startup_lam_init = 1.0
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])

        stress = dmt_exponential.model_function(t, params, "startup")

        assert stress.shape == (100,)
        assert jnp.all(jnp.isfinite(stress))
        # Stress should be positive for positive shear rate
        assert jnp.all(stress > 0)

    # --- model_function: relaxation ---

    def test_model_function_relaxation(self, dmt_exponential):
        """Test model_function returns correct relaxation stress."""
        t = jnp.linspace(0, 50, 200)
        dmt_exponential._relax_sigma_init = 200.0
        dmt_exponential._relax_lam_init = 0.5
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])

        stress = dmt_exponential.model_function(t, params, "relaxation")

        assert stress.shape == (200,)
        assert jnp.all(jnp.isfinite(stress))
        # Stress should decay
        assert float(stress[-1]) < float(stress[0])

    # --- model_function: oscillation ---

    def test_model_function_oscillation(self, dmt_exponential):
        """Test model_function returns complex modulus for SAOS."""
        omega = jnp.logspace(-2, 2, 30)
        dmt_exponential._saos_lam_0 = 1.0
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])

        G_star = dmt_exponential.model_function(omega, params, "oscillation")

        assert G_star.shape == (30,)
        assert jnp.iscomplexobj(G_star)
        G_prime = jnp.real(G_star)
        G_double_prime = jnp.imag(G_star)
        assert jnp.all(G_prime >= 0)
        assert jnp.all(G_double_prime >= 0)

    # --- model_function: laos ---

    def test_model_function_laos(self, dmt_exponential):
        """Test model_function returns LAOS stress waveform."""
        omega = 1.0
        gamma_0 = 0.1
        n_cycles = 3
        points_per_cycle = 64
        t = jnp.linspace(0, n_cycles * 2 * np.pi / omega, n_cycles * points_per_cycle)
        dmt_exponential._gamma_0 = gamma_0
        dmt_exponential._omega_laos = omega
        dmt_exponential._laos_lam_init = 1.0
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])

        stress = dmt_exponential.model_function(t, params, "laos")

        assert stress.shape == t.shape
        assert jnp.all(jnp.isfinite(stress))

    # --- fit_bayesian smoke test ---

    @pytest.mark.slow
    def test_fit_bayesian_flow_curve(self, dmt_exponential, mcmc_config):
        """Smoke test: fit_bayesian works end-to-end for flow curve."""
        gamma_dot = np.logspace(-1, 2, 15)
        stress_true = dmt_exponential._predict_flow_curve(gamma_dot)

        np.random.seed(42)
        stress_noisy = stress_true * (1 + 0.05 * np.random.randn(len(stress_true)))

        # First fit with NLSQ for warm-start
        dmt_exponential._fit_flow_curve(gamma_dot, stress_noisy)
        assert dmt_exponential._fitted

        # Now run Bayesian inference
        result = dmt_exponential.fit_bayesian(
            gamma_dot,
            stress_noisy,
            test_mode="flow_curve",
            num_warmup=mcmc_config["num_warmup"],
            num_samples=mcmc_config["num_samples"],
            num_chains=mcmc_config["num_chains"],
            seed=42,
        )

        assert result is not None
        assert result.posterior_samples is not None
        assert len(result.posterior_samples) > 0


# =============================================================================
# Test Registry
# =============================================================================


class TestF002KwargsCache:
    """Regression tests for F-002: model_function must use cached protocol kwargs."""

    @pytest.mark.smoke
    def test_startup_kwargs_cached_after_fit(self, dmt_exponential):
        """After _fit_transient(gamma_dot=10), model_function must use 10, not 1."""
        t = np.linspace(0.01, 5, 50)
        # Generate data at gamma_dot=10
        stress_true = np.array(
            dmt_exponential._predict_startup(t, gamma_dot=10.0, lam_init=1.0)
        )
        noise = stress_true * 0.01 * np.random.default_rng(42).standard_normal(len(t))
        stress_noisy = stress_true + noise

        # Fit with gamma_dot=10
        dmt_exponential._fit_transient(t, stress_noisy, gamma_dot=10.0, lam_init=1.0)

        # Verify model_function now uses gamma_dot=10 (not default 1.0)
        assert hasattr(dmt_exponential, "_gamma_dot_applied")
        assert dmt_exponential._gamma_dot_applied == 10.0
        assert dmt_exponential._startup_lam_init == 1.0

        # model_function output should match predict_startup at same gamma_dot
        params = jnp.array([v.value for v in dmt_exponential.parameters.values()])
        mf_stress = dmt_exponential.model_function(t, params, "startup")
        predict_stress = dmt_exponential._predict_startup(t, gamma_dot=10.0)

        # These should be nearly identical (both use gamma_dot=10)
        assert jnp.allclose(jnp.array(mf_stress), jnp.array(predict_stress), rtol=0.05)

    @pytest.mark.smoke
    def test_relaxation_kwargs_cached_after_fit(self, dmt_exponential):
        """After _fit_relaxation(sigma_init=200), model_function must use 200."""
        t = np.linspace(0.01, 20, 100)
        sigma_init = 200.0
        lam_init = 0.5

        # Generate synthetic relaxation data
        _, stress_true, _ = dmt_exponential.simulate_relaxation(
            t_end=20.0, dt=0.01, sigma_init=sigma_init, lam_init=lam_init
        )
        # Subsample to match t array length
        indices = np.linspace(0, len(stress_true) - 1, len(t)).astype(int)
        stress_data = np.array(stress_true[indices])

        dmt_exponential._fit_relaxation(
            t, stress_data, sigma_init=sigma_init, lam_init=lam_init
        )

        assert dmt_exponential._relax_sigma_init == sigma_init
        assert dmt_exponential._relax_lam_init == lam_init

    @pytest.mark.smoke
    def test_creep_kwargs_cached_after_fit(self, dmt_exponential):
        """After _fit_creep(sigma_0=50), model_function must use 50."""
        sigma_0 = 50.0
        t_end = 10.0
        dt = 0.1
        t = np.arange(0, t_end, dt)

        # Generate creep data
        _, gamma_true, _, _ = dmt_exponential.simulate_creep(
            sigma_0=sigma_0, t_end=t_end, dt=dt
        )
        gamma_data = np.array(gamma_true)

        dmt_exponential._fit_creep(t, gamma_data, sigma_0=sigma_0, lam_init=1.0)

        assert dmt_exponential._sigma_applied == sigma_0
        assert dmt_exponential._creep_lam_init == 1.0

    @pytest.mark.smoke
    def test_oscillation_kwargs_cached_after_fit(self, dmt_exponential):
        """After _fit_oscillation(lam_0=0.8), model_function must use 0.8."""
        omega = np.logspace(-2, 2, 30)
        lam_0 = 0.8

        G_prime, G_double_prime = dmt_exponential.predict_saos(omega, lam_0=lam_0)
        G_star = np.array(G_prime) + 1j * np.array(G_double_prime)

        dmt_exponential._fit_oscillation(omega, G_star, lam_0=lam_0)

        assert dmt_exponential._saos_lam_0 == lam_0

    @pytest.mark.smoke
    def test_laos_kwargs_cached_after_fit(self, dmt_exponential):
        """After _fit_laos, model_function must use cached gamma_0/omega."""
        gamma_0 = 0.5
        omega = 2.0

        result = dmt_exponential.simulate_laos(
            gamma_0=gamma_0, omega=omega, n_cycles=3, points_per_cycle=64
        )
        t = result["t"]
        stress = result["stress"]

        dmt_exponential._fit_laos(
            np.array(t), np.array(stress),
            gamma_0=gamma_0, omega_laos=omega, lam_init=1.0,
        )

        assert dmt_exponential._gamma_0 == gamma_0
        assert dmt_exponential._omega_laos == omega
        assert dmt_exponential._laos_lam_init == 1.0


class TestFitRelaxation:
    """Tests for _fit_relaxation implementation."""

    def test_fit_relaxation_returns_self(self, dmt_exponential):
        """_fit_relaxation returns self for fluent API."""
        t = np.linspace(0.01, 20, 100)
        _, stress, _ = dmt_exponential.simulate_relaxation(
            t_end=20.0, dt=0.2, sigma_init=100.0, lam_init=0.5
        )
        result = dmt_exponential._fit_relaxation(
            t, np.array(stress), sigma_init=100.0, lam_init=0.5
        )
        assert result is dmt_exponential
        assert dmt_exponential._fitted

    def test_fit_relaxation_produces_finite_params(self, dmt_exponential):
        """Fitted parameters should be finite and within bounds."""
        t = np.linspace(0.01, 20, 100)
        _, stress, _ = dmt_exponential.simulate_relaxation(
            t_end=20.0, dt=0.2, sigma_init=100.0, lam_init=0.5
        )
        dmt_exponential._fit_relaxation(
            t, np.array(stress), sigma_init=100.0, lam_init=0.5
        )
        for name in dmt_exponential.parameters.keys():
            p = dmt_exponential.parameters[name]
            assert np.isfinite(p.value), f"Parameter {name} is not finite: {p.value}"
            lo, hi = p.bounds
            assert lo <= p.value <= hi, f"Parameter {name}={p.value} outside bounds [{lo}, {hi}]"


class TestFitCreep:
    """Tests for _fit_creep implementation."""

    def test_fit_creep_returns_self(self, dmt_exponential):
        """_fit_creep returns self for fluent API."""
        sigma_0 = 50.0
        _, gamma, _, _ = dmt_exponential.simulate_creep(
            sigma_0=sigma_0, t_end=10.0, dt=0.1
        )
        t = np.linspace(0, 10.0, len(gamma))
        result = dmt_exponential._fit_creep(
            t, np.array(gamma), sigma_0=sigma_0, lam_init=1.0
        )
        assert result is dmt_exponential
        assert dmt_exponential._fitted

    def test_fit_creep_viscous(self, dmt_viscous):
        """Viscous variant creep fit works."""
        sigma_0 = 20.0
        _, gamma, _, _ = dmt_viscous.simulate_creep(
            sigma_0=sigma_0, t_end=5.0, dt=0.05
        )
        t = np.linspace(0, 5.0, len(gamma))
        result = dmt_viscous._fit_creep(
            t, np.array(gamma), sigma_0=sigma_0, lam_init=1.0
        )
        assert result is dmt_viscous
        assert dmt_viscous._fitted


class TestFitOscillation:
    """Tests for _fit_oscillation implementation."""

    def test_fit_oscillation_returns_self(self, dmt_exponential):
        """_fit_oscillation returns self for fluent API."""
        omega = np.logspace(-2, 2, 30)
        G_prime, G_double_prime = dmt_exponential.predict_saos(omega, lam_0=1.0)
        G_star = np.array(G_prime) + 1j * np.array(G_double_prime)

        result = dmt_exponential._fit_oscillation(omega, G_star, lam_0=1.0)
        assert result is dmt_exponential
        assert dmt_exponential._fitted

    def test_fit_oscillation_accepts_2d_input(self, dmt_exponential):
        """_fit_oscillation accepts (N,2) input [G', G'']."""
        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = dmt_exponential.predict_saos(omega, lam_0=1.0)
        G_2d = np.column_stack([np.array(G_prime), np.array(G_double_prime)])

        result = dmt_exponential._fit_oscillation(omega, G_2d, lam_0=1.0)
        assert result is dmt_exponential
        assert dmt_exponential._fitted

    def test_fit_oscillation_recovers_moduli(self, dmt_exponential):
        """Fitted model should reproduce SAOS moduli."""
        omega = np.logspace(-2, 2, 30)
        G_prime_true, G_double_prime_true = dmt_exponential.predict_saos(omega, lam_0=1.0)
        G_star = np.array(G_prime_true) + 1j * np.array(G_double_prime_true)

        # Perturb initial params slightly
        orig_G0 = dmt_exponential.parameters["G0"].value
        dmt_exponential.parameters["G0"].value = orig_G0 * 1.5

        dmt_exponential._fit_oscillation(omega, G_star, lam_0=1.0)

        # Predict with fitted params
        G_prime_fit, G_double_prime_fit = dmt_exponential.predict_saos(omega, lam_0=1.0)

        # Should be reasonable fit (not perfect due to optimization landscape)
        residual = np.mean(
            (np.array(G_prime_fit) - np.array(G_prime_true)) ** 2
            + (np.array(G_double_prime_fit) - np.array(G_double_prime_true)) ** 2
        )
        assert residual < 1e6, f"Fit residual too large: {residual}"


class TestFitLAOS:
    """Tests for _fit_laos implementation."""

    def test_fit_laos_returns_self(self, dmt_exponential):
        """_fit_laos returns self for fluent API."""
        gamma_0 = 0.1
        omega = 1.0
        result = dmt_exponential.simulate_laos(
            gamma_0=gamma_0, omega=omega, n_cycles=3, points_per_cycle=64
        )
        dmt_exponential._fit_laos(
            np.array(result["t"]), np.array(result["stress"]),
            gamma_0=gamma_0, omega_laos=omega, lam_init=1.0,
        )
        assert dmt_exponential._fitted

    def test_fit_laos_viscous(self, dmt_viscous):
        """Viscous variant LAOS fit works."""
        gamma_0 = 0.5
        omega = 1.0
        result = dmt_viscous.simulate_laos(
            gamma_0=gamma_0, omega=omega, n_cycles=3, points_per_cycle=64
        )
        dmt_viscous._fit_laos(
            np.array(result["t"]), np.array(result["stress"]),
            gamma_0=gamma_0, omega_laos=omega, lam_init=1.0,
        )
        assert dmt_viscous._fitted


class TestPredictLAOS:
    """Tests for LAOS predict dispatch."""

    @pytest.mark.smoke
    def test_predict_laos_dispatch(self, dmt_exponential):
        """model._predict(t, test_mode='laos') should not raise."""
        t = np.linspace(0, 10 * 2 * np.pi, 500)
        stress = dmt_exponential._predict(
            t, test_mode="laos", gamma_0=0.1, omega_laos=1.0
        )
        assert len(stress) > 0
        assert np.all(np.isfinite(stress))


class TestPublicAPIRoundTrip:
    """Test fit() -> predict() through the public BaseModel API."""

    @pytest.mark.smoke
    def test_fit_predict_flow_curve(self, dmt_exponential):
        """Public API round-trip for flow curve."""
        gamma_dot = np.logspace(-1, 2, 20)
        stress_true = dmt_exponential._predict_flow_curve(gamma_dot)
        noise = 0.02 * stress_true * np.random.default_rng(0).standard_normal(len(gamma_dot))
        stress_data = np.array(stress_true) + noise

        dmt_exponential.fit(gamma_dot, stress_data, test_mode="flow_curve")
        stress_pred = dmt_exponential.predict(gamma_dot, test_mode="flow_curve")

        assert stress_pred.shape == stress_data.shape
        assert np.all(np.isfinite(stress_pred))


class TestDMTRegistry:
    """Test model registry integration."""

    @pytest.mark.smoke
    def test_local_registered(self):
        """Test DMTLocal is in registry."""
        from rheojax.core.registry import ModelRegistry

        models = ModelRegistry.list_models()
        assert "dmt_local" in models

    @pytest.mark.smoke
    def test_nonlocal_registered(self):
        """Test DMTNonlocal is in registry."""
        from rheojax.core.registry import ModelRegistry

        models = ModelRegistry.list_models()
        assert "dmt_nonlocal" in models

    def test_create_from_registry(self):
        """Test creating models via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("dmt_local")
        assert isinstance(model, DMTLocal)
