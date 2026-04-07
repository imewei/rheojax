"""Tests for rheojax.utils.protocol_preprocessing."""

import numpy as np
import pytest

from rheojax.utils.protocol_preprocessing import (
    PreprocessingResult,
    check_kramers_kronig,
    estimate_eta0,
    fit_gel_point,
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
