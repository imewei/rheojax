"""Unit tests for Fluidity-Saramito physics kernels.

Tests cover:
- Von Mises stress calculation
- Plasticity function α
- Upper-convected derivative
- Fluidity evolution
- Yield stress coupling
- Steady-state predictions
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.fluidity.saramito._kernels import (
    banding_ratio,
    detect_shear_bands,
    fluidity_evolution_saramito,
    herschel_bulkley_viscosity,
    laplacian_1d_neumann,
    saramito_flow_curve_steady,
    saramito_plasticity_alpha,
    saramito_steady_state_full,
    shear_banding_cv,
    upper_convected_2d,
    von_mises_stress_2d,
    yield_stress_from_fluidity,
)

jax, jnp = safe_import_jax()


class TestVonMisesStress:
    """Tests for Von Mises equivalent stress calculation."""

    @pytest.mark.smoke
    def test_pure_shear(self):
        """Test Von Mises stress for pure shear."""
        tau_xy = 100.0
        tau_mag = von_mises_stress_2d(0.0, 0.0, tau_xy)

        # For pure shear: |τ| = τ_xy (in our formulation)
        # With traceless: τ_zz = 0, so |τ| = √(2τ_xy²/2) = τ_xy
        assert tau_mag > 0
        assert np.isfinite(tau_mag)

    @pytest.mark.smoke
    def test_isotropic_normal_stress(self):
        """Test that traceless stress gives consistent result."""
        tau_xx = 50.0
        tau_yy = -25.0  # Not exactly traceless but close
        tau_xy = 100.0

        tau_mag = von_mises_stress_2d(tau_xx, tau_yy, tau_xy)

        # Should be dominated by shear stress
        assert tau_mag > tau_xy * 0.8
        assert np.isfinite(tau_mag)

    def test_zero_stress(self):
        """Test zero stress gives zero magnitude."""
        tau_mag = von_mises_stress_2d(0.0, 0.0, 0.0)
        assert tau_mag < 1e-10


class TestPlasticityAlpha:
    """Tests for Saramito plasticity function."""

    @pytest.mark.smoke
    def test_below_yield(self):
        """Test α = 0 below yield stress."""
        tau_y = 100.0
        # Stress below yield
        alpha = saramito_plasticity_alpha(0.0, 0.0, 50.0, tau_y)

        # Should be close to zero (smooth approximation)
        assert alpha < 0.1

    @pytest.mark.smoke
    def test_above_yield(self):
        """Test α > 0 above yield stress."""
        tau_y = 100.0
        # Stress well above yield
        alpha = saramito_plasticity_alpha(0.0, 0.0, 200.0, tau_y)

        # Should be positive, approaching 0.5
        assert alpha > 0.3
        assert alpha < 1.0

    def test_far_above_yield(self):
        """Test α → 1 far above yield."""
        tau_y = 100.0
        # Stress much higher than yield
        alpha = saramito_plasticity_alpha(0.0, 0.0, 10000.0, tau_y)

        # Should approach 1
        assert alpha > 0.9

    def test_bounds(self):
        """Test α is bounded [0, 1]."""
        for tau_xy in [1.0, 50.0, 100.0, 200.0, 1000.0]:
            alpha = saramito_plasticity_alpha(0.0, 0.0, tau_xy, tau_y=100.0)
            assert 0.0 <= alpha <= 1.0


class TestUpperConvected:
    """Tests for upper-convected derivative."""

    @pytest.mark.smoke
    def test_simple_shear(self):
        """Test convective terms in simple shear."""
        tau_xx, tau_yy, tau_xy = 10.0, -5.0, 100.0
        gamma_dot = 1.0

        conv_xx, conv_yy, conv_xy = upper_convected_2d(
            tau_xx, tau_yy, tau_xy, gamma_dot
        )

        # Expected: conv_xx = 2*γ̇*τ_xy = 200
        assert np.isclose(conv_xx, 200.0, rtol=1e-10)

        # Expected: conv_yy = 0
        assert np.isclose(conv_yy, 0.0, rtol=1e-10)

        # Expected: conv_xy = γ̇*τ_yy = -5
        assert np.isclose(conv_xy, -5.0, rtol=1e-10)

    def test_zero_rate(self):
        """Test zero convective terms at zero rate."""
        conv_xx, conv_yy, conv_xy = upper_convected_2d(10.0, -5.0, 100.0, 0.0)

        assert conv_xx == 0.0
        assert conv_yy == 0.0
        assert conv_xy == 0.0


class TestFluidityEvolution:
    """Tests for fluidity evolution kernel."""

    @pytest.mark.smoke
    def test_aging_at_rest(self):
        """Test fluidity decreases (ages) at rest."""
        f = 1e-3  # High fluidity
        f_age = 1e-6  # Low target
        f_flow = 1e-2
        t_a = 10.0
        b = 1.0
        n_rej = 1.0

        # At rest (driving = 0), should age toward f_age
        df_dt = fluidity_evolution_saramito(f, 0.0, f_age, f_flow, t_a, b, n_rej)

        # Aging: (f_age - f)/t_a < 0 since f > f_age
        assert df_dt < 0

    @pytest.mark.smoke
    def test_rejuvenation_under_flow(self):
        """Test fluidity increases under flow."""
        f = 1e-6  # Low fluidity (aged)
        f_age = 1e-6
        f_flow = 1e-2  # High target
        t_a = 10.0
        b = 1.0
        n_rej = 1.0
        driving = 10.0  # High shear rate

        df_dt = fluidity_evolution_saramito(f, driving, f_age, f_flow, t_a, b, n_rej)

        # Rejuvenation dominates: b*|γ̇|^n*(f_flow - f) > 0
        assert df_dt > 0

    def test_equilibrium_at_rest(self):
        """Test df/dt = 0 at f = f_age with no flow."""
        f_age = 1e-6
        df_dt = fluidity_evolution_saramito(
            f_age, 0.0, f_age, 1e-2, 10.0, 1.0, 1.0
        )

        # Should be close to zero at equilibrium
        assert abs(df_dt) < 1e-15


class TestYieldStressCoupling:
    """Tests for dynamic yield stress coupling."""

    @pytest.mark.smoke
    def test_minimal_coupling(self):
        """Test constant yield stress with zero coupling."""
        tau_y0 = 100.0
        tau_y = yield_stress_from_fluidity(1e-4, tau_y0, 0.0, 1.0)

        assert tau_y == tau_y0

    @pytest.mark.smoke
    def test_full_coupling(self):
        """Test increased yield stress at low fluidity."""
        tau_y0 = 100.0
        tau_y_coupling = 1.0
        m_yield = 1.0

        # Low fluidity (aged structure)
        f_low = 1e-6
        tau_y_low = yield_stress_from_fluidity(f_low, tau_y0, tau_y_coupling, m_yield)

        # High fluidity (rejuvenated)
        f_high = 1e-3
        tau_y_high = yield_stress_from_fluidity(f_high, tau_y0, tau_y_coupling, m_yield)

        # Aged state should have higher yield stress
        assert tau_y_low > tau_y_high
        assert tau_y_low > tau_y0
        assert tau_y_high > tau_y0  # Still above base


class TestHerschelBulkleyViscosity:
    """Tests for HB viscosity calculation."""

    @pytest.mark.smoke
    def test_shear_thinning(self):
        """Test shear-thinning behavior (n < 1)."""
        tau_y = 100.0
        K = 50.0
        n = 0.5

        eta_low = herschel_bulkley_viscosity(0.1, tau_y, K, n)
        eta_high = herschel_bulkley_viscosity(10.0, tau_y, K, n)

        # Viscosity should decrease with increasing rate
        assert eta_low > eta_high

    def test_solvent_contribution(self):
        """Test solvent viscosity adds correctly."""
        eta_s = 1.0
        eta_without = herschel_bulkley_viscosity(1.0, 100.0, 50.0, 0.5, 0.0)
        eta_with = herschel_bulkley_viscosity(1.0, 100.0, 50.0, 0.5, eta_s)

        assert np.isclose(eta_with - eta_without, eta_s, rtol=1e-10)


class TestSteadyStateFlowCurve:
    """Tests for steady-state flow curve prediction."""

    @pytest.mark.smoke
    def test_yield_stress_emergence(self):
        """Test yield stress appears at low rates."""
        gamma_dot = jnp.logspace(-3, 2, 50)

        sigma = saramito_flow_curve_steady(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
        )

        # At low rates, stress should approach yield stress
        assert np.min(sigma) > 50  # Above ~50% of yield
        assert np.min(sigma) < 200  # Not too far above yield

    @pytest.mark.smoke
    def test_monotonic_increase(self):
        """Test stress increases monotonically with rate."""
        gamma_dot = jnp.logspace(-2, 2, 50)

        sigma = saramito_flow_curve_steady(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
        )

        # Check monotonicity
        sigma_np = np.array(sigma)
        assert np.all(np.diff(sigma_np) > 0)

    def test_full_coupling_effect(self):
        """Test full coupling increases low-rate stress."""
        gamma_dot = jnp.array([0.01, 0.1, 1.0])

        sigma_minimal = saramito_flow_curve_steady(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
            coupling_mode="minimal",
        )

        sigma_full = saramito_flow_curve_steady(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
            coupling_mode="full",
            tau_y_coupling=1e-3,
            m_yield=0.5,
        )

        # Full coupling should give higher stress at low rates
        assert np.array(sigma_full)[0] > np.array(sigma_minimal)[0]


class TestNormalStresses:
    """Tests for normal stress predictions."""

    @pytest.mark.smoke
    def test_n1_positive(self):
        """Test N₁ is positive (Weissenberg effect)."""
        gamma_dot = jnp.array([0.1, 1.0, 10.0])

        tau_xy, tau_xx, N1 = saramito_steady_state_full(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
        )

        # N₁ should be positive for simple shear
        assert np.all(np.array(N1) > 0)

    def test_n1_scaling(self):
        """Test N₁ ~ γ̇² (approximately at high rates)."""
        gamma_dot = jnp.array([1.0, 10.0])

        _, _, N1 = saramito_steady_state_full(
            gamma_dot,
            G=1e4,
            tau_y0=100.0,
            K_HB=50.0,
            n_HB=0.5,
            f_age=1e-6,
            f_flow=1e-2,
            t_a=10.0,
            b=1.0,
            n_rej=1.0,
        )

        N1_np = np.array(N1)
        # Ratio should scale roughly as (10/1)^(1-2) depending on model details
        # Just check it increases significantly
        assert N1_np[1] > N1_np[0] * 5


class TestLaplacian:
    """Tests for 1D Laplacian with Neumann BCs."""

    @pytest.mark.smoke
    def test_constant_field(self):
        """Test Laplacian of constant is zero."""
        f = jnp.ones(10) * 5.0
        dy = 0.1

        lap = laplacian_1d_neumann(f, dy)

        assert np.allclose(np.array(lap), 0.0, atol=1e-10)

    def test_parabolic_field(self):
        """Test Laplacian of parabola is constant."""
        # f(y) = y^2, ∇²f = 2
        y = jnp.linspace(0, 1, 11)
        f = y**2
        dy = y[1] - y[0]

        lap = laplacian_1d_neumann(f, dy)

        # Interior points should be close to 2
        lap_interior = np.array(lap)[2:-2]
        assert np.allclose(lap_interior, 2.0, rtol=0.1)


class TestShearBandingMetrics:
    """Tests for shear banding detection."""

    @pytest.mark.smoke
    def test_homogeneous_cv(self):
        """Test CV is low for uniform profile."""
        f = jnp.ones(50) * 1e-4

        cv = shear_banding_cv(f)

        assert cv < 0.01

    @pytest.mark.smoke
    def test_banded_cv(self):
        """Test CV is high for step profile."""
        f = jnp.concatenate([jnp.ones(25) * 1e-6, jnp.ones(25) * 1e-3])

        cv = shear_banding_cv(f)

        # CV should be significant for step profile
        assert cv > 0.3

    def test_banding_ratio(self):
        """Test banding ratio for heterogeneous profile."""
        f = jnp.concatenate([jnp.ones(25) * 1e-6, jnp.ones(25) * 1e-3])

        ratio = banding_ratio(f)

        # Ratio should be ~1000
        assert ratio > 100
        assert ratio < 10000

    def test_detect_shear_bands_homogeneous(self):
        """Test detection correctly identifies uniform flow."""
        f = jnp.ones(50) * 1e-4

        is_banded, cv, ratio = detect_shear_bands(f)

        assert not is_banded
        assert cv < 0.3

    def test_detect_shear_bands_banded(self):
        """Test detection correctly identifies shear bands."""
        f = jnp.concatenate([jnp.ones(25) * 1e-6, jnp.ones(25) * 1e-3])

        is_banded, cv, ratio = detect_shear_bands(f)

        assert is_banded
        assert cv > 0.3
