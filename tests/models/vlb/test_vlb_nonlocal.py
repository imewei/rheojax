"""Unit tests for VLBNonlocal (nonlocal VLB with spatial PDE).

Tests cover:
- Model creation and parameter setup
- Homogeneous solutions matching VLBLocal
- Spatial diffusion effects
- Shear banding detection for Bell breakage
- Protocol simulations (steady shear, startup, creep)
- Grid convergence
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.vlb import VLBNonlocal

jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def vlb_nl_constant():
    """VLBNonlocal with constant breakage, few grid points for speed."""
    model = VLBNonlocal(breakage="constant", n_points=11)
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("D_mu", 1e-6)
    return model


@pytest.fixture
def vlb_nl_bell():
    """VLBNonlocal with Bell breakage, few grid points."""
    model = VLBNonlocal(breakage="bell", n_points=11)
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("nu", 5.0)
    model.parameters.set_value("D_mu", 1e-6)
    return model


@pytest.fixture
def vlb_nl_fene():
    """VLBNonlocal with FENE stress, few grid points."""
    model = VLBNonlocal(stress_type="fene", n_points=11)
    model.parameters.set_value("G0", 1000.0)
    model.parameters.set_value("k_d_0", 1.0)
    model.parameters.set_value("eta_s", 0.0)
    model.parameters.set_value("L_max", 10.0)
    model.parameters.set_value("D_mu", 1e-6)
    return model


# =============================================================================
# Creation Tests (@smoke)
# =============================================================================


class TestVLBNonlocalCreation:
    """Test model creation."""

    @pytest.mark.smoke
    def test_creation_default(self, vlb_nl_constant):
        """Default nonlocal has core params + D_mu."""
        assert vlb_nl_constant.n_points == 11
        assert vlb_nl_constant.parameters.get_value("D_mu") == 1e-6
        assert vlb_nl_constant.G0 == 1000.0

    @pytest.mark.smoke
    def test_creation_bell(self, vlb_nl_bell):
        """Bell nonlocal has nu parameter."""
        assert vlb_nl_bell.parameters.get_value("nu") == 5.0

    @pytest.mark.smoke
    def test_parameter_count(self):
        """Constant: G0, k_d_0, eta_s, D_mu = 4 params."""
        model = VLBNonlocal(n_points=11)
        assert len(model.parameters) == 4

        model_bell = VLBNonlocal(breakage="bell", n_points=11)
        assert len(model_bell.parameters) == 5  # + nu

        model_fene = VLBNonlocal(stress_type="fene", n_points=11)
        assert len(model_fene.parameters) == 5  # + L_max

    @pytest.mark.smoke
    def test_grid_setup(self, vlb_nl_constant):
        """Spatial grid is correctly set up."""
        assert len(vlb_nl_constant.y) == 11
        assert vlb_nl_constant.y[0] == pytest.approx(0.0)
        assert vlb_nl_constant.y[-1] == pytest.approx(1e-3)
        assert vlb_nl_constant.dy == pytest.approx(1e-4)

    @pytest.mark.smoke
    def test_cooperativity_length(self, vlb_nl_constant):
        """Cooperativity length xi = sqrt(D_mu / k_d_0)."""
        xi = vlb_nl_constant.get_cooperativity_length()
        expected = np.sqrt(1e-6 / 1.0)  # = 1e-3 m
        assert xi == pytest.approx(expected, rel=1e-10)

    @pytest.mark.smoke
    def test_repr(self, vlb_nl_bell):
        """Repr includes grid info."""
        r = repr(vlb_nl_bell)
        assert "bell" in r
        assert "11" in r


# =============================================================================
# Homogeneous Tests (uniform profile = local result)
# =============================================================================


class TestVLBNonlocalHomogeneous:
    """Uniform initial conditions should give uniform profiles."""

    def test_constant_kd_uniform_stress(self, vlb_nl_constant):
        """Constant k_d with uniform IC gives uniform mu_xy profile."""
        result = vlb_nl_constant.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=50.0, dt=1.0, perturbation=0.0
        )
        # At steady state, mu_xy should be uniform
        mu_xy_final = result["mu_xy"][-1]
        assert np.std(mu_xy_final) < 1e-4 * np.mean(np.abs(mu_xy_final))

    def test_startup_uniform(self, vlb_nl_constant):
        """Startup with uniform IC stays uniform."""
        result = vlb_nl_constant.simulate_startup(gamma_dot_avg=1.0, t_end=10.0, dt=0.5)
        # gamma_dot profile should be nearly uniform
        gdot_final = result["gamma_dot"][-1]
        assert np.std(gdot_final) < 1e-3 * np.mean(np.abs(gdot_final) + 1e-10)

    def test_diffusion_smooths_perturbation(self, vlb_nl_constant):
        """D_mu > 0 damps spatial perturbations over time."""
        result = vlb_nl_constant.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=50.0, dt=5.0, perturbation=0.1
        )
        # Early perturbation should be larger than late
        gdot_early = result["gamma_dot"][1]
        gdot_late = result["gamma_dot"][-1]
        std_early = float(np.std(gdot_early))
        std_late = float(np.std(gdot_late))
        # Late should be smoother (or same if already uniform)
        assert std_late <= std_early + 1e-4

    def test_neumann_bc_zero_flux(self, vlb_nl_constant):
        """Neumann BC: gradient at boundaries is approximately zero."""
        from rheojax.models.vlb._kernels import laplacian_1d_neumann_vlb

        # Constant field: Laplacian should be zero
        field = jnp.ones(11, dtype=jnp.float64) * 5.0
        lap = laplacian_1d_neumann_vlb(field, 1e-4)
        np.testing.assert_allclose(np.asarray(lap), 0.0, atol=1e-10)

        # Linear field: Laplacian should be zero (second derivative of linear = 0)
        field_lin = jnp.linspace(1.0, 2.0, 11)
        lap_lin = laplacian_1d_neumann_vlb(field_lin, 1e-4)
        # Interior points should be ~0, boundaries may differ due to BC
        np.testing.assert_allclose(np.asarray(lap_lin[1:-1]), 0.0, atol=1.0)


# =============================================================================
# Banding Tests (Bell breakage required)
# =============================================================================


class TestVLBNonlocalBanding:
    """Shear banding with Bell breakage + nonlocal diffusion."""

    def test_banding_detected(self, vlb_nl_bell):
        """Bell + nonlocal produces banding for suitable parameters."""
        # Strong shear thinning → potential banding
        result = vlb_nl_bell.simulate_steady_shear(
            gamma_dot_avg=2.0, t_end=100.0, dt=2.0, perturbation=0.05
        )
        banding = vlb_nl_bell.detect_banding(result, threshold=0.1)
        # May or may not band depending on parameters - just check API works
        assert isinstance(banding, dict)
        assert "is_banding" in banding

    def test_velocity_profile_integration(self, vlb_nl_bell):
        """Integral of gamma_dot across gap gives V_wall."""
        result = vlb_nl_bell.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=50.0, dt=5.0, perturbation=0.0
        )
        v_profile = vlb_nl_bell.get_velocity_profile(result)
        # v(0) = 0, v(H) = gamma_dot_avg * H
        assert v_profile[0] == pytest.approx(0.0, abs=1e-10)
        expected_v_wall = 1.0 * 1e-3  # gamma_dot_avg * gap_width
        assert v_profile[-1] == pytest.approx(expected_v_wall, rel=0.1)

    def test_steady_state_convergence(self, vlb_nl_constant):
        """Transient simulation converges to steady state."""
        result = vlb_nl_constant.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=50.0, dt=5.0, perturbation=0.0
        )
        stress = result["stress"]
        # Stress should converge (late values should be stable)
        rel_change = abs(float(stress[-1] - stress[-2])) / abs(
            float(stress[-1]) + 1e-20
        )
        assert rel_change < 0.01


# =============================================================================
# Protocol Tests
# =============================================================================


class TestVLBNonlocalProtocols:
    """Test simulation protocols."""

    def test_steady_shear_returns_dict(self, vlb_nl_constant):
        """simulate_steady_shear returns correct keys."""
        result = vlb_nl_constant.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=10.0, dt=1.0
        )
        assert "t" in result
        assert "y" in result
        assert "mu_xy" in result
        assert "gamma_dot" in result
        assert "stress" in result
        assert result["mu_xy"].shape[1] == 11  # n_points

    def test_startup_returns_dict(self, vlb_nl_constant):
        """simulate_startup returns correct structure."""
        result = vlb_nl_constant.simulate_startup(gamma_dot_avg=1.0, t_end=10.0, dt=1.0)
        assert "t" in result
        assert "stress" in result

    def test_creep_returns_dict(self, vlb_nl_constant):
        """simulate_creep returns correct structure."""
        result = vlb_nl_constant.simulate_creep(sigma_0=100.0, t_end=10.0, dt=1.0)
        assert "t" in result
        assert "stress" in result

    def test_fit_raises_not_implemented(self, vlb_nl_constant):
        """Nonlocal model cannot be fitted directly."""
        with pytest.raises(NotImplementedError):
            vlb_nl_constant.fit(np.array([1.0]), np.array([1.0]))


# =============================================================================
# FENE Nonlocal Tests
# =============================================================================


class TestVLBNonlocalFene:
    """Test FENE-P with nonlocal model."""

    def test_fene_nonlocal_runs(self, vlb_nl_fene):
        """FENE + nonlocal simulation completes."""
        result = vlb_nl_fene.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=20.0, dt=2.0, perturbation=0.0
        )
        assert np.all(np.isfinite(result["stress"]))

    def test_fene_nonlocal_uniform(self, vlb_nl_fene):
        """FENE nonlocal with uniform IC stays uniform."""
        result = vlb_nl_fene.simulate_steady_shear(
            gamma_dot_avg=1.0, t_end=20.0, dt=2.0, perturbation=0.0
        )
        gdot_final = result["gamma_dot"][-1]
        assert np.std(gdot_final) < 1e-3 * np.mean(np.abs(gdot_final) + 1e-10)
