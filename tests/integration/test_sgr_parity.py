"""Parity tests: SGRGeneric vs SGRConventional feature comparison.

This module verifies that SGRGeneric produces equivalent results to
SGRConventional for all extended features, ensuring GENERIC framework
correctness matches the established reference implementation.

Test categories:
- Shear banding detection parity (User Story 1)
- LAOS harmonics parity (User Story 2)
- Thixotropic stress transient parity (User Story 3)

References:
    - Fuereder & Ilg 2013 PRE 88, 042134
    - Sollich 1998 PRE 58, 738
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Safe import to enforce float64
jax, jnp = safe_import_jax()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sgr_conventional():
    """Create SGRConventional model."""
    from rheojax.models import SGRConventional

    return SGRConventional()


@pytest.fixture
def sgr_generic():
    """Create SGRGeneric model."""
    from rheojax.models import SGRGeneric

    return SGRGeneric()


@pytest.fixture
def shared_params():
    """Common parameters for both models."""
    return {
        "x": 0.8,  # Glass regime for shear banding
        "G0": 1000.0,
        "tau0": 0.01,
    }


@pytest.fixture
def laos_params():
    """Common parameters for LAOS tests."""
    return {
        "x": 1.5,  # Power-law fluid regime
        "G0": 1000.0,
        "tau0": 0.01,
    }


def set_params(model, params: dict):
    """Set parameters on a model."""
    for key, value in params.items():
        model.parameters.set_value(key, value)


# =============================================================================
# User Story 1: Shear Banding Parity Tests (T014)
# =============================================================================


class TestShearBandingParity:
    """Parity tests for shear banding detection."""

    @pytest.mark.smoke
    def test_shear_banding_detection_parity(
        self, sgr_conventional, sgr_generic, shared_params
    ):
        """Test shear banding detection matches between models."""
        set_params(sgr_conventional, shared_params)
        set_params(sgr_generic, shared_params)

        # Set test mode
        sgr_conventional._test_mode = "steady_shear"
        sgr_generic._test_mode = "steady_shear"

        # Detect shear banding
        is_banding_conv, info_conv = sgr_conventional.detect_shear_banding(
            gamma_dot_range=(1e-2, 1e2), n_points=100
        )
        is_banding_gen, info_gen = sgr_generic.detect_shear_banding(
            gamma_dot_range=(1e-2, 1e2), n_points=100
        )

        # Should agree on detection
        assert is_banding_conv == is_banding_gen, (
            f"Banding detection mismatch: conv={is_banding_conv}, gen={is_banding_gen}"
        )

        if is_banding_conv and is_banding_gen:
            # Check banding region agreement (5% tolerance)
            rtol = 0.05

            np.testing.assert_allclose(
                info_conv["gamma_dot_low"],
                info_gen["gamma_dot_low"],
                rtol=rtol,
                err_msg="gamma_dot_low mismatch",
            )
            np.testing.assert_allclose(
                info_conv["gamma_dot_high"],
                info_gen["gamma_dot_high"],
                rtol=rtol,
                err_msg="gamma_dot_high mismatch",
            )
            # Compare sigma_range instead of sigma_plateau (detect_shear_banding)
            np.testing.assert_allclose(
                info_conv["sigma_range"][0],
                info_gen["sigma_range"][0],
                rtol=rtol,
                err_msg="sigma_range[0] mismatch",
            )
            np.testing.assert_allclose(
                info_conv["sigma_range"][1],
                info_gen["sigma_range"][1],
                rtol=rtol,
                err_msg="sigma_range[1] mismatch",
            )

    def test_predict_banded_flow_parity(
        self, sgr_conventional, sgr_generic, shared_params
    ):
        """Test banded flow prediction matches between models."""
        set_params(sgr_conventional, shared_params)
        set_params(sgr_generic, shared_params)

        sgr_conventional._test_mode = "steady_shear"
        sgr_generic._test_mode = "steady_shear"

        gamma_dot_applied = 1.0

        result_conv = sgr_conventional.predict_banded_flow(gamma_dot_applied)
        result_gen = sgr_generic.predict_banded_flow(gamma_dot_applied)

        # Both should return result or both None
        assert (result_conv is None) == (result_gen is None)

        if result_conv is not None and result_gen is not None:
            rtol = 0.05

            np.testing.assert_allclose(
                result_conv["fraction_low"],
                result_gen["fraction_low"],
                rtol=rtol,
                err_msg="fraction_low mismatch",
            )
            np.testing.assert_allclose(
                result_conv["fraction_high"],
                result_gen["fraction_high"],
                rtol=rtol,
                err_msg="fraction_high mismatch",
            )


# =============================================================================
# User Story 2: LAOS Harmonics Parity Tests (T022)
# =============================================================================


class TestLAOSParity:
    """Parity tests for LAOS analysis."""

    @pytest.mark.smoke
    def test_laos_harmonics_I3_I1_parity(
        self, sgr_conventional, sgr_generic, laos_params
    ):
        """Test I3/I1 harmonic ratio matches within 2% tolerance."""
        set_params(sgr_conventional, laos_params)
        set_params(sgr_generic, laos_params)

        gamma_0 = 0.5
        omega = 1.0
        n_cycles = 2
        n_points_per_cycle = 256

        # Simulate LAOS
        strain_conv, stress_conv = sgr_conventional.simulate_laos(
            gamma_0=gamma_0,
            omega=omega,
            n_cycles=n_cycles,
            n_points_per_cycle=n_points_per_cycle,
        )
        strain_gen, stress_gen = sgr_generic.simulate_laos(
            gamma_0=gamma_0,
            omega=omega,
            n_cycles=n_cycles,
            n_points_per_cycle=n_points_per_cycle,
        )

        # Extract harmonics
        harm_conv = sgr_conventional.extract_laos_harmonics(stress_conv)
        harm_gen = sgr_generic.extract_laos_harmonics(stress_gen)

        # Compare I3/I1 ratio (2% tolerance as specified in SC-009)
        I3_I1_conv = harm_conv["I_3_I_1"]
        I3_I1_gen = harm_gen["I_3_I_1"]

        # Use absolute tolerance since values can be small
        atol = 0.02  # 2% absolute tolerance
        diff = abs(I3_I1_conv - I3_I1_gen)

        assert diff < atol, (
            f"I3/I1 parity failed: conv={I3_I1_conv:.4f}, gen={I3_I1_gen:.4f}, "
            f"diff={diff:.4f} > {atol}"
        )

    def test_laos_fundamental_amplitude_parity(
        self, sgr_conventional, sgr_generic, laos_params
    ):
        """Test fundamental amplitude I1 matches between models."""
        set_params(sgr_conventional, laos_params)
        set_params(sgr_generic, laos_params)

        gamma_0 = 0.5
        omega = 1.0

        strain_conv, stress_conv = sgr_conventional.simulate_laos(
            gamma_0=gamma_0, omega=omega
        )
        strain_gen, stress_gen = sgr_generic.simulate_laos(
            gamma_0=gamma_0, omega=omega
        )

        harm_conv = sgr_conventional.extract_laos_harmonics(stress_conv)
        harm_gen = sgr_generic.extract_laos_harmonics(stress_gen)

        # I1 should match within 5%
        np.testing.assert_allclose(
            harm_conv["I_1"],
            harm_gen["I_1"],
            rtol=0.05,
            err_msg="I_1 amplitude mismatch",
        )

    def test_chebyshev_coefficients_parity(
        self, sgr_conventional, sgr_generic, laos_params
    ):
        """Test Chebyshev coefficients match between models."""
        set_params(sgr_conventional, laos_params)
        set_params(sgr_generic, laos_params)

        gamma_0 = 0.5
        omega = 1.0

        strain_conv, stress_conv = sgr_conventional.simulate_laos(
            gamma_0=gamma_0, omega=omega
        )
        strain_gen, stress_gen = sgr_generic.simulate_laos(
            gamma_0=gamma_0, omega=omega
        )

        cheb_conv = sgr_conventional.compute_chebyshev_coefficients(
            strain_conv, stress_conv, gamma_0, omega
        )
        cheb_gen = sgr_generic.compute_chebyshev_coefficients(
            strain_gen, stress_gen, gamma_0, omega
        )

        # Compare ratios
        rtol = 0.05

        np.testing.assert_allclose(
            cheb_conv["e_3_e_1"],
            cheb_gen["e_3_e_1"],
            rtol=rtol,
            atol=0.01,
            err_msg="e_3/e_1 mismatch",
        )
        np.testing.assert_allclose(
            cheb_conv["v_3_v_1"],
            cheb_gen["v_3_v_1"],
            rtol=rtol,
            atol=0.01,
            err_msg="v_3/v_1 mismatch",
        )


# =============================================================================
# User Story 3: Thixotropy Parity Tests (T033)
# =============================================================================


class TestThixotropyParity:
    """Parity tests for thixotropic stress transients."""

    @pytest.mark.smoke
    def test_stress_transient_parity(self, sgr_conventional, sgr_generic, shared_params):
        """Test stress transient matches within 5% tolerance."""
        # Use power-law regime for thixotropy
        params = {"x": 1.3, "G0": 1000.0, "tau0": 0.01}
        set_params(sgr_conventional, params)
        set_params(sgr_generic, params)

        # Enable thixotropy with same parameters
        thixo_params = {"k_build": 0.1, "k_break": 0.5, "n_struct": 2.0}
        sgr_conventional.enable_thixotropy(**thixo_params)
        sgr_generic.enable_thixotropy(**thixo_params)

        # Time and shear rate
        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0  # Constant shear

        # Predict stress transients
        sigma_conv, lambda_conv = sgr_conventional.predict_stress_transient(
            t, gamma_dot, lambda_initial=1.0
        )
        sigma_gen, lambda_gen = sgr_generic.predict_stress_transient(
            t, gamma_dot, lambda_initial=1.0
        )

        # Compare stress (5% tolerance as specified)
        rtol = 0.05

        np.testing.assert_allclose(
            sigma_conv,
            sigma_gen,
            rtol=rtol,
            err_msg="Stress transient mismatch",
        )

        np.testing.assert_allclose(
            lambda_conv,
            lambda_gen,
            rtol=rtol,
            err_msg="Lambda evolution mismatch",
        )

    def test_lambda_evolution_parity(self, sgr_conventional, sgr_generic):
        """Test lambda evolution matches between models."""
        params = {"x": 1.3, "G0": 1000.0, "tau0": 0.01}
        set_params(sgr_conventional, params)
        set_params(sgr_generic, params)

        thixo_params = {"k_build": 0.1, "k_break": 0.5, "n_struct": 2.0}
        sgr_conventional.enable_thixotropy(**thixo_params)
        sgr_generic.enable_thixotropy(**thixo_params)

        t = np.linspace(0, 10, 100)
        gamma_dot = np.ones_like(t) * 10.0

        lambda_conv = sgr_conventional.evolve_lambda(t, gamma_dot, lambda_initial=1.0)
        lambda_gen = sgr_generic.evolve_lambda(t, gamma_dot, lambda_initial=1.0)

        np.testing.assert_allclose(
            lambda_conv,
            lambda_gen,
            rtol=0.05,
            err_msg="Lambda evolution mismatch",
        )


# =============================================================================
# Cross-Feature Validation
# =============================================================================


class TestCrossFeatureValidation:
    """Tests for cross-feature validation."""

    @pytest.mark.smoke
    def test_oscillation_prediction_parity(
        self, sgr_conventional, sgr_generic, laos_params
    ):
        """Test basic oscillation predictions match (baseline sanity check)."""
        set_params(sgr_conventional, laos_params)
        set_params(sgr_generic, laos_params)

        omega = np.logspace(-2, 2, 50)

        # Both should be able to predict oscillation
        sgr_conventional._test_mode = "oscillation"
        sgr_generic._test_mode = "oscillation"

        G_star_conv = sgr_conventional.predict(omega)
        G_star_gen = sgr_generic.predict(omega)

        # Should match within 1%
        np.testing.assert_allclose(
            G_star_conv,
            G_star_gen,
            rtol=0.01,
            err_msg="Oscillation prediction mismatch",
        )

    def test_relaxation_prediction_parity(
        self, sgr_conventional, sgr_generic, laos_params
    ):
        """Test relaxation predictions match."""
        set_params(sgr_conventional, laos_params)
        set_params(sgr_generic, laos_params)

        t = np.logspace(-4, 2, 50)

        sgr_conventional._test_mode = "relaxation"
        sgr_generic._test_mode = "relaxation"

        G_t_conv = sgr_conventional.predict(t)
        G_t_gen = sgr_generic.predict(t)

        # Should match within 1%
        np.testing.assert_allclose(
            G_t_conv,
            G_t_gen,
            rtol=0.01,
            err_msg="Relaxation prediction mismatch",
        )
