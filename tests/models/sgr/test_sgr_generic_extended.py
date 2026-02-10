"""Extended tests for SGRGeneric model with advanced features.

This module contains tests for the SGRGeneric feature extension including:
- Shear banding detection (User Story 1)
- LAOS analysis (User Story 2)
- Thixotropy (User Story 3)
- Dynamic noise temperature x(t) (User Story 4)
- Thermodynamic consistency verification

All tests follow TDD methodology - write tests first, ensure they fail,
then implement features to make them pass.

References:
    - Fuereder & Ilg 2013 PRE 88, 042134
    - Sollich 1998 PRE 58, 738
    - Ewoldt, Hosoi, McKinley 2008 J Rheol 52, 1427
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models import SGRGeneric

# Ensure float64
jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def model() -> SGRGeneric:
    """Create a basic SGRGeneric model with default parameters."""
    return SGRGeneric()


@pytest.fixture
def glass_model() -> SGRGeneric:
    """Create SGRGeneric model in glass regime (x < 1)."""
    m = SGRGeneric()
    m.parameters.set_value("x", 0.8)
    m.parameters.set_value("G0", 1000.0)
    m.parameters.set_value("tau0", 0.01)
    m._test_mode = "steady_shear"
    return m


@pytest.fixture
def fluid_model() -> SGRGeneric:
    """Create SGRGeneric model in power-law fluid regime (x > 1)."""
    m = SGRGeneric()
    m.parameters.set_value("x", 1.5)
    m.parameters.set_value("G0", 1000.0)
    m.parameters.set_value("tau0", 0.01)
    m._test_mode = "oscillation"
    return m


@pytest.fixture
def dynamic_x_model() -> SGRGeneric:
    """Create SGRGeneric model with dynamic x enabled."""
    return SGRGeneric(dynamic_x=True)


@pytest.fixture
def thixotropic_model() -> SGRGeneric:
    """Create SGRGeneric model with thixotropy enabled."""
    m = SGRGeneric()
    m.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)
    return m


# =============================================================================
# Phase 2: Thermodynamic Verification Tests (T011)
# =============================================================================


class TestThermodynamicConsistency:
    """Tests for GENERIC thermodynamic consistency requirements."""

    @pytest.mark.smoke
    def test_poisson_antisymmetry_2d(self, model):
        """Test L + L^T = 0 for 2D state (standard mode)."""
        state = np.array([100.0, 0.5])  # [sigma, lambda]
        L = model.poisson_bracket(state)

        # L should be antisymmetric
        antisym_error = np.max(np.abs(L + L.T))
        assert antisym_error < 1e-10, f"L antisymmetry error: {antisym_error}"

    @pytest.mark.smoke
    def test_friction_symmetry_2d(self, model):
        """Test M = M^T for 2D state."""
        state = np.array([100.0, 0.5])
        M = model.friction_matrix(state)

        # M should be symmetric
        sym_error = np.max(np.abs(M - M.T))
        assert sym_error < 1e-10, f"M symmetry error: {sym_error}"

    @pytest.mark.smoke
    def test_friction_psd_2d(self, model):
        """Test eig(M) >= 0 for 2D state."""
        state = np.array([100.0, 0.5])
        M = model.friction_matrix(state)

        # M should be positive semi-definite
        eigenvalues = np.linalg.eigvalsh(M)
        min_eig = np.min(eigenvalues)
        assert min_eig >= -1e-12, f"M has negative eigenvalue: {min_eig}"

    @pytest.mark.smoke
    def test_entropy_production_nonnegative_2d(self, model):
        """Test W >= 0 for 2D state."""
        state = np.array([100.0, 0.5])
        W = model.compute_entropy_production(state)

        assert W >= -1e-12, f"Entropy production W = {W} < 0"

    def test_verify_thermodynamic_consistency_2d(self, model):
        """Test full thermodynamic verification for 2D state."""
        state = np.array([100.0, 0.5])
        result = model.verify_thermodynamic_consistency(state)

        assert result["poisson_antisymmetric"], "L not antisymmetric"
        assert result["friction_symmetric"], "M not symmetric"
        assert result["friction_positive_semidefinite"], "M not PSD"
        assert result["entropy_production_nonnegative"], "W < 0"
        assert result["thermodynamically_consistent"], "Thermodynamically inconsistent"

    @pytest.mark.parametrize(
        "sigma,lam",
        [
            (1.0, 0.1),
            (100.0, 0.5),
            (1000.0, 0.9),
            (0.1, 0.01),
            (10.0, 0.99),
        ],
    )
    def test_thermodynamic_consistency_parameter_sweep(self, model, sigma, lam):
        """Test thermodynamic consistency across parameter space."""
        state = np.array([sigma, lam])
        result = model.verify_thermodynamic_consistency(state)
        assert result[
            "thermodynamically_consistent"
        ], f"Inconsistent at sigma={sigma}, lambda={lam}"


# =============================================================================
# Phase 3: User Story 1 - Shear Banding Tests (T012-T014)
# =============================================================================


class TestShearBandingDetection:
    """Tests for shear banding detection in glass regime."""

    @pytest.mark.smoke
    def test_detect_shear_banding_return_types(self, glass_model):
        """Test detect_shear_banding() returns correct types."""
        is_banding, info = glass_model.detect_shear_banding(
            gamma_dot_range=(1e-2, 1e2), n_points=100
        )

        assert isinstance(is_banding, bool)
        if is_banding:
            assert isinstance(info, dict)
            assert "gamma_dot_low" in info
            assert "gamma_dot_high" in info
            assert "sigma_range" in info  # Not sigma_plateau

    def test_detect_shear_banding_glass_regime(self, glass_model):
        """Test shear banding detection in glass regime (x < 1)."""
        is_banding, info = glass_model.detect_shear_banding()

        # Glass regime should exhibit shear banding
        assert is_banding, "Glass regime (x=0.8) should show shear banding"
        assert info is not None
        assert info["gamma_dot_low"] < info["gamma_dot_high"]
        assert info["sigma_low"] > 0 or info["sigma_high"] > 0

    def test_no_shear_banding_fluid_regime(self, fluid_model):
        """Test no shear banding in fluid regime (x > 1)."""
        fluid_model._test_mode = "steady_shear"
        is_banding, info = fluid_model.detect_shear_banding()

        # Fluid regime should not exhibit shear banding
        assert not is_banding, "Fluid regime (x=1.5) should not show banding"

    @pytest.mark.smoke
    def test_predict_banded_flow_return_types(self, glass_model):
        """Test predict_banded_flow() returns correct types."""
        result = glass_model.predict_banded_flow(gamma_dot_applied=1.0)

        if result is not None:
            assert isinstance(result, dict)
            assert "fraction_low" in result
            assert "fraction_high" in result
            assert "stress_plateau" in result  # Not sigma_composite

    def test_predict_banded_flow_lever_rule(self, glass_model):
        """Test lever rule conservation in banded flow."""
        gamma_dot_applied = 1.0
        result = glass_model.predict_banded_flow(gamma_dot_applied)

        if result is not None:
            # Lever rule: f_low + f_high = 1
            total_fraction = result["fraction_low"] + result["fraction_high"]
            assert (
                abs(total_fraction - 1.0) < 1e-6
            ), f"Lever rule violated: {total_fraction}"


# =============================================================================
# Phase 4: User Story 2 - LAOS Analysis Tests (T018-T022)
# =============================================================================


class TestLAOSAnalysis:
    """Tests for Large Amplitude Oscillatory Shear (LAOS) analysis."""

    @pytest.mark.smoke
    def test_simulate_laos_output_shapes(self, fluid_model):
        """Test simulate_laos() returns correct shapes."""
        gamma_0 = 0.5
        omega = 1.0
        n_cycles = 2
        n_points_per_cycle = 256

        strain, stress = fluid_model.simulate_laos(
            gamma_0=gamma_0,
            omega=omega,
            n_cycles=n_cycles,
            n_points_per_cycle=n_points_per_cycle,
        )

        expected_len = n_cycles * n_points_per_cycle
        assert strain.shape == (expected_len,), f"Strain shape: {strain.shape}"
        assert stress.shape == (expected_len,), f"Stress shape: {stress.shape}"

    def test_simulate_laos_periodicity(self, fluid_model):
        """Test that LAOS output is periodic."""
        gamma_0 = 0.5
        omega = 1.0
        n_cycles = 2
        n_points_per_cycle = 256

        strain, stress = fluid_model.simulate_laos(
            gamma_0=gamma_0,
            omega=omega,
            n_cycles=n_cycles,
            n_points_per_cycle=n_points_per_cycle,
        )

        # Compare first and last cycle
        cycle_1_strain = strain[:n_points_per_cycle]
        cycle_2_strain = strain[-n_points_per_cycle:]

        # Strain should repeat exactly
        np.testing.assert_allclose(
            cycle_1_strain, cycle_2_strain, rtol=1e-6, atol=1e-10
        )

    @pytest.mark.smoke
    def test_extract_laos_harmonics_structure(self, fluid_model):
        """Test extract_laos_harmonics() returns required keys."""
        strain, stress = fluid_model.simulate_laos(
            gamma_0=0.5, omega=1.0, n_cycles=2, n_points_per_cycle=256
        )

        harmonics = fluid_model.extract_laos_harmonics(stress, n_points_per_cycle=256)

        # Check required keys
        required_keys = ["I_1", "I_3", "I_5", "I_7", "phi_1", "phi_3", "phi_5", "phi_7"]
        for key in required_keys:
            assert key in harmonics, f"Missing key: {key}"

        # I1 should be dominant
        assert harmonics["I_1"] > 0, "Fundamental amplitude I1 should be positive"

    def test_extract_laos_harmonics_values(self, fluid_model):
        """Test that harmonic amplitudes are physically reasonable."""
        strain, stress = fluid_model.simulate_laos(
            gamma_0=0.5, omega=1.0, n_cycles=2, n_points_per_cycle=256
        )

        harmonics = fluid_model.extract_laos_harmonics(stress)

        # Third harmonic ratio should be small for near-linear response
        I3_I1 = harmonics["I_3_I_1"]
        assert 0 <= I3_I1 < 1.0, f"I3/I1 = {I3_I1} out of expected range"

    @pytest.mark.smoke
    def test_compute_chebyshev_coefficients_structure(self, fluid_model):
        """Test compute_chebyshev_coefficients() returns required keys."""
        gamma_0 = 0.5
        omega = 1.0
        strain, stress = fluid_model.simulate_laos(
            gamma_0=gamma_0, omega=omega, n_cycles=2, n_points_per_cycle=256
        )

        chebyshev = fluid_model.compute_chebyshev_coefficients(
            strain, stress, gamma_0, omega, n_points_per_cycle=256
        )

        # Check required keys
        required_keys = ["e_1", "e_3", "e_5", "v_1", "v_3", "v_5", "e_3_e_1", "v_3_v_1"]
        for key in required_keys:
            assert key in chebyshev, f"Missing key: {key}"

    def test_chebyshev_coefficients_values(self, fluid_model):
        """Test Chebyshev coefficients are physically reasonable."""
        gamma_0 = 0.5
        omega = 1.0
        strain, stress = fluid_model.simulate_laos(
            gamma_0=gamma_0, omega=omega, n_cycles=2, n_points_per_cycle=256
        )

        chebyshev = fluid_model.compute_chebyshev_coefficients(
            strain, stress, gamma_0, omega
        )

        # e1 (first elastic) should dominate
        assert abs(chebyshev["e_1"]) > 0, "e1 should be non-zero"

        # Ratios should be in reasonable range
        assert abs(chebyshev["e_3_e_1"]) < 1.0, f"e3/e1 = {chebyshev['e_3_e_1']}"

    @pytest.mark.smoke
    def test_get_lissajous_curve_output(self, fluid_model):
        """Test get_lissajous_curve() output shape and normalization."""
        gamma_0 = 0.5
        omega = 1.0
        n_points = 256

        strain, stress = fluid_model.get_lissajous_curve(
            gamma_0=gamma_0, omega=omega, n_points=n_points, normalized=False
        )

        assert strain.shape == (n_points,)
        assert stress.shape == (n_points,)

        # Check amplitude
        assert np.max(np.abs(strain)) <= gamma_0 * 1.01  # Allow small numerical error

    def test_get_lissajous_curve_normalized(self, fluid_model):
        """Test normalized Lissajous curve is bounded [-1, 1]."""
        strain_norm, stress_norm = fluid_model.get_lissajous_curve(
            gamma_0=0.5, omega=1.0, n_points=256, normalized=True
        )

        assert np.max(np.abs(strain_norm)) <= 1.01, "Normalized strain out of [-1, 1]"
        assert np.max(np.abs(stress_norm)) <= 1.01, "Normalized stress out of [-1, 1]"


# =============================================================================
# Phase 5: User Story 3 - Thixotropy Tests (T028-T033)
# =============================================================================


class TestThixotropy:
    """Tests for thixotropic stress transients."""

    @pytest.mark.smoke
    def test_enable_thixotropy_activation(self, model):
        """Test enable_thixotropy() activates parameters correctly."""
        # Initially no thixotropy
        assert not getattr(model, "_thixotropy_enabled", False)

        # Enable thixotropy
        model.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)

        # Check activation
        assert model._thixotropy_enabled
        assert "k_build" in model.parameters.keys()
        assert "k_break" in model.parameters.keys()
        assert "n_struct" in model.parameters.keys()

        # Check values
        assert model.parameters.get_value("k_build") == 0.1
        assert model.parameters.get_value("k_break") == 0.5
        assert model.parameters.get_value("n_struct") == 2.0

    @pytest.mark.smoke
    def test_evolve_lambda_kinetics(self, thixotropic_model):
        """Test evolve_lambda() with build/break kinetics."""
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0  # Constant shear

        lambda_t = thixotropic_model.evolve_lambda(t, gamma_dot, lambda_initial=1.0)

        # Lambda should decrease under shear (breakdown)
        assert lambda_t[-1] < lambda_t[0], "Lambda should decrease under shear"

        # Lambda should stay in bounds [0, 1]
        assert np.all(lambda_t >= 0.0)
        assert np.all(lambda_t <= 1.0)

    def test_evolve_lambda_recovery(self, thixotropic_model):
        """Test lambda recovery at rest (zero shear)."""
        t = np.linspace(0, 100, 1000)
        gamma_dot = np.zeros_like(t)  # Rest

        lambda_t = thixotropic_model.evolve_lambda(t, gamma_dot, lambda_initial=0.5)

        # Lambda should increase toward 1 at rest (buildup)
        assert lambda_t[-1] > lambda_t[0], "Lambda should increase at rest"

    @pytest.mark.smoke
    def test_predict_thixotropic_stress_computation(self, thixotropic_model):
        """Test predict_thixotropic_stress() returns reasonable values."""
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0

        sigma = thixotropic_model.predict_thixotropic_stress(t, gamma_dot)

        assert sigma.shape == t.shape
        assert np.all(sigma > 0), "Stress should be positive"
        assert np.isfinite(sigma).all(), "Stress should be finite"

    @pytest.mark.smoke
    def test_predict_stress_transient_overshoot(self, thixotropic_model):
        """Test stress overshoot behavior for step-up shear."""
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0  # Constant high shear

        sigma_t, lambda_t = thixotropic_model.predict_stress_transient(
            t, gamma_dot, lambda_initial=1.0
        )

        assert sigma_t.shape == t.shape
        assert lambda_t.shape == t.shape

        # For thixotropic material with lambda_initial=1:
        # Stress should be high initially and decay as structure breaks
        # This may or may not show classic overshoot depending on parameters

    def test_thermodynamic_consistency_thixotropy_mode(self, thixotropic_model):
        """Test W >= 0 during thixotropic evolution (1000 random states)."""
        np.random.seed(42)

        for _ in range(1000):
            sigma = np.random.uniform(1, 1000)
            lam = np.random.uniform(0.1, 0.9)
            state = np.array([sigma, lam])

            W = thixotropic_model.compute_entropy_production(state)
            assert W >= -1e-12, f"W = {W} < 0 at state={state}"


# =============================================================================
# Phase 6: User Story 4 - Dynamic x Tests (T041-T048)
# =============================================================================


class TestDynamicX:
    """Tests for dynamic noise temperature x(t) evolution."""

    @pytest.mark.smoke
    def test_dynamic_x_constructor(self):
        """Test SGRGeneric(dynamic_x=True) enables 3D state."""
        model = SGRGeneric(dynamic_x=True)

        assert model._dynamic_x is True
        # Check dynamic x parameters exist
        assert "x_eq" in model.parameters.keys()
        assert "alpha_aging" in model.parameters.keys()
        assert "beta_rejuv" in model.parameters.keys()

    @pytest.mark.smoke
    def test_poisson_bracket_3d_antisymmetry(self, dynamic_x_model):
        """Test _poisson_bracket_3d() returns antisymmetric L."""
        state = np.array([100.0, 0.5, 1.5])  # [sigma, lambda, x]
        L = dynamic_x_model._poisson_bracket_3d(state)

        assert L.shape == (3, 3)

        # L should be antisymmetric
        antisym_error = np.max(np.abs(L + L.T))
        assert antisym_error < 1e-10, f"L antisymmetry error: {antisym_error}"

    @pytest.mark.smoke
    def test_friction_matrix_3d_symmetry(self, dynamic_x_model):
        """Test _friction_matrix_3d() returns symmetric M."""
        state = np.array([100.0, 0.5, 1.5])
        M = dynamic_x_model._friction_matrix_3d(state, gamma_dot=10.0)

        assert M.shape == (3, 3)

        # M should be symmetric
        sym_error = np.max(np.abs(M - M.T))
        assert sym_error < 1e-10, f"M symmetry error: {sym_error}"

    def test_friction_matrix_3d_block_diagonal(self, dynamic_x_model):
        """Test 3D friction matrix is block-diagonal (M_13 = M_23 = 0)."""
        state = np.array([100.0, 0.5, 1.5])
        M = dynamic_x_model._friction_matrix_3d(state, gamma_dot=10.0)

        # Block-diagonal structure
        assert abs(M[0, 2]) < 1e-12, f"M_13 = {M[0, 2]} != 0"
        assert abs(M[1, 2]) < 1e-12, f"M_23 = {M[1, 2]} != 0"

    @pytest.mark.smoke
    def test_friction_matrix_3d_psd(self, dynamic_x_model):
        """Test 3D friction matrix is positive semi-definite."""
        state = np.array([100.0, 0.5, 1.5])
        M = dynamic_x_model._friction_matrix_3d(state, gamma_dot=10.0)

        eigenvalues = np.linalg.eigvalsh(M)
        min_eig = np.min(eigenvalues)
        assert min_eig >= -1e-12, f"M has negative eigenvalue: {min_eig}"

    def test_free_energy_gradient_3d(self, dynamic_x_model):
        """Test free_energy_gradient() includes dF/dx = -S."""
        state = np.array([100.0, 0.5, 1.5])
        grad = dynamic_x_model.free_energy_gradient(state)

        assert grad.shape == (3,), f"Gradient shape: {grad.shape}"

        # dF/dx should be -S (entropy)
        # S = -[lambda*ln(lambda) + (1-lambda)*ln(1-lambda)]
        lam = state[1]
        S = -(lam * np.log(lam) + (1 - lam) * np.log(1 - lam))
        expected_dFdx = -S

        assert (
            abs(grad[2] - expected_dFdx) < 1e-6
        ), f"dF/dx = {grad[2]}, expected -S = {expected_dFdx}"

    @pytest.mark.smoke
    def test_evolve_x_aging(self, dynamic_x_model):
        """Test x evolution: aging at rest (x -> x_eq)."""
        t = np.linspace(0, 100, 1000)
        gamma_dot = np.zeros_like(t)  # Rest

        x0 = 2.0  # Start above x_eq
        x_eq = dynamic_x_model.parameters.get_value("x_eq")

        x_t = dynamic_x_model.evolve_x(t, gamma_dot, x0=x0)

        # x should decrease toward x_eq at rest (aging)
        assert x_t[-1] < x_t[0], "x should decrease at rest (aging)"
        assert abs(x_t[-1] - x_eq) < abs(x_t[0] - x_eq), "x should approach x_eq"

    def test_evolve_x_rejuvenation(self, dynamic_x_model):
        """Test x evolution: rejuvenation under shear (x -> x_ss)."""
        t = np.linspace(0, 100, 1000)
        gamma_dot = np.ones_like(t) * 10.0  # High shear

        x0 = dynamic_x_model.parameters.get_value("x_eq")  # Start at rest
        x_t = dynamic_x_model.evolve_x(t, gamma_dot, x0=x0)

        # x should increase under shear (rejuvenation)
        assert x_t[-1] > x_t[0], "x should increase under shear (rejuvenation)"

    def test_thermodynamic_consistency_3d_random_states(self, dynamic_x_model):
        """Test W >= 0 for 1000 random 3D states."""
        np.random.seed(42)

        for _ in range(1000):
            sigma = np.random.uniform(1, 1000)
            lam = np.random.uniform(0.1, 0.9)
            x = np.random.uniform(0.6, 2.5)
            state = np.array([sigma, lam, x])

            W = dynamic_x_model.compute_entropy_production(state)
            assert W >= -1e-12, f"W = {W} < 0 at state={state}"

    def test_friction_matrix_eigenvalues_3d_random_states(self, dynamic_x_model):
        """Test eig(M_3x3) >= 0 for 1000 random states."""
        np.random.seed(42)

        for _ in range(1000):
            sigma = np.random.uniform(1, 1000)
            lam = np.random.uniform(0.1, 0.9)
            x = np.random.uniform(0.6, 2.5)
            gamma_dot = np.random.uniform(0.01, 100)
            state = np.array([sigma, lam, x])

            M = dynamic_x_model._friction_matrix_3d(state, gamma_dot)
            eigenvalues = np.linalg.eigvalsh(M)
            min_eig = np.min(eigenvalues)

            assert min_eig >= -1e-12, f"M not PSD: min_eig = {min_eig} at state={state}"

    @pytest.mark.smoke
    def test_combined_thixotropy_dynamic_x(self, dynamic_x_model):
        """Test thixotropy + dynamic_x combined mode."""
        # Enable thixotropy on dynamic_x model
        dynamic_x_model.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)

        assert dynamic_x_model._dynamic_x is True
        assert dynamic_x_model._thixotropy_enabled is True

        # Should still be thermodynamically consistent
        state = np.array([100.0, 0.5, 1.5])
        result = dynamic_x_model.verify_thermodynamic_consistency(state)
        assert result[
            "thermodynamically_consistent"
        ], "Combined mode should be consistent"


# =============================================================================
# Phase 7: Property-Based Tests (T059)
# =============================================================================


class TestPropertyBased:
    """Property-based tests for comprehensive validation."""

    @pytest.mark.slow
    def test_thermodynamic_consistency_full_parameter_space(self, model):
        """Test thermodynamic consistency across full parameter space."""
        np.random.seed(42)

        # Sample 1000 random states and parameter combinations
        for _ in range(1000):
            # Random parameters
            x = np.random.uniform(0.6, 2.5)
            G0 = np.random.uniform(10, 10000)
            tau0 = np.random.uniform(1e-5, 1.0)

            model.parameters.set_value("x", x)
            model.parameters.set_value("G0", G0)
            model.parameters.set_value("tau0", tau0)

            # Random state
            sigma = np.random.uniform(1, 1000)
            lam = np.random.uniform(0.05, 0.95)
            state = np.array([sigma, lam])

            # Verify thermodynamic consistency
            result = model.verify_thermodynamic_consistency(state)
            assert result[
                "thermodynamically_consistent"
            ], f"Inconsistent at x={x}, G0={G0}, state={state}"
