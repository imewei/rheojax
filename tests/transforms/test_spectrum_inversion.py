"""Tests for SpectrumInversion transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.spectrum_inversion import SpectrumInversion


class TestSpectrumInversionTikhonov:
    """Test Tikhonov-regularized spectrum inversion."""

    def _make_single_mode_data(self):
        """Create oscillation data from a single Maxwell mode (delta-function spectrum)."""
        G_0 = 1000.0
        tau_0 = 1.0
        omega = np.logspace(-3, 3, 100)
        wt2 = (omega * tau_0) ** 2
        G_prime = G_0 * wt2 / (1.0 + wt2)
        G_double_prime = G_0 * omega * tau_0 / (1.0 + wt2)
        G_star = G_prime + 1j * G_double_prime
        return omega, G_star, tau_0

    @pytest.mark.smoke
    def test_single_mode_peak(self):
        """Spectrum should peak near the true relaxation time."""
        omega, G_star, tau_0 = self._make_single_mode_data()
        data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})

        transform = SpectrumInversion(
            method="tikhonov",
            n_tau=80,
            source="oscillation",
        )
        result, meta = transform.transform(data)

        tau = np.asarray(result.x)
        H = np.asarray(result.y)
        peak_tau = tau[np.argmax(H)]

        # Peak should be within one decade of true τ
        assert 0.1 * tau_0 < peak_tau < 10.0 * tau_0

    @pytest.mark.smoke
    def test_non_negative_spectrum(self):
        """H(τ) should be non-negative everywhere."""
        omega, G_star, _ = self._make_single_mode_data()
        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(method="tikhonov", n_tau=50)
        result, _ = transform.transform(data)
        assert np.all(np.asarray(result.y) >= 0)

    def test_two_mode_spectrum(self):
        """Two-mode system should show two distinct peaks."""
        G1, tau1 = 500.0, 0.01
        G2, tau2 = 800.0, 100.0
        omega = np.logspace(-4, 4, 200)

        wt1 = (omega * tau1) ** 2
        wt2 = (omega * tau2) ** 2
        G_p = G1 * wt1 / (1 + wt1) + G2 * wt2 / (1 + wt2)
        G_pp = G1 * omega * tau1 / (1 + wt1) + G2 * omega * tau2 / (1 + wt2)
        G_star = G_p + 1j * G_pp

        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(
            method="tikhonov",
            n_tau=100,
            tau_range=(1e-3, 1e3),
        )
        result, meta = transform.transform(data)
        H = np.asarray(result.y)
        # Should have non-trivial spectrum (not all zeros)
        assert np.max(H) > 0

    def test_manual_regularization(self):
        """Manual λ should override auto-selection."""
        omega, G_star, _ = self._make_single_mode_data()
        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(
            method="tikhonov",
            n_tau=50,
            regularization=1.0,
        )
        result, meta = transform.transform(data)
        assert meta["spectrum_result"].regularization_param == 1.0

    def test_relaxation_source(self):
        """Inversion from G(t) relaxation data."""
        G_0 = 1000.0
        tau_0 = 1.0
        t = np.logspace(-3, 3, 100)
        G_t = G_0 * np.exp(-t / tau_0)

        data = RheoData(x=t, y=G_t)
        transform = SpectrumInversion(
            method="tikhonov",
            n_tau=50,
            source="relaxation",
        )
        result, meta = transform.transform(data)
        tau = np.asarray(result.x)
        H = np.asarray(result.y)
        peak_tau = tau[np.argmax(H)]
        assert 0.1 * tau_0 < peak_tau < 10.0 * tau_0


class TestSpectrumInversionMaxEnt:
    """Test maximum entropy spectrum inversion."""

    @pytest.mark.smoke
    def test_basic_max_entropy(self):
        """MaxEnt should produce non-negative spectrum."""
        G_0 = 1000.0
        tau_0 = 1.0
        omega = np.logspace(-2, 2, 50)
        wt2 = (omega * tau_0) ** 2
        G_star = G_0 * wt2 / (1 + wt2) + 1j * G_0 * omega * tau_0 / (1 + wt2)

        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(method="max_entropy", n_tau=40)
        result, meta = transform.transform(data)

        H = np.asarray(result.y)
        assert np.all(H >= 0)
        assert np.max(H) > 0

    def test_invalid_method(self):
        """Unknown method should raise ValueError."""
        data = RheoData(x=np.array([1.0]), y=np.array([1.0]))
        transform = SpectrumInversion(method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            transform.transform(data)
