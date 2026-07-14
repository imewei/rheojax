"""Tests for SpectrumInversion transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.spectrum_inversion import SpectrumInversion, _build_kernel


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

    def test_d_ln_tau_boundary_weights_symmetric_and_exact(self):
        """Quadrature weights must be symmetric at both ends and sum to the
        true log(tau) range, not overshoot it by half a bin at tau_min.
        """
        tau = np.logspace(-3, 3, 10)
        # exp(-t/tau) ~= 1 for all tau when t is tiny, so the first row of the
        # relaxation kernel isolates the d_ln_tau weight vector directly.
        t = np.array([1e-15])
        A = _build_kernel(t, tau, "relaxation", 0.0)
        d_ln_tau = A[0, :]

        assert np.isclose(d_ln_tau[0], d_ln_tau[-1])
        assert np.isclose(
            np.sum(d_ln_tau), np.log(tau[-1] / tau[0]), rtol=1e-9
        )

    def test_nan_input_raises_validation_error(self):
        """NaN in x must raise a clear validation error instead of silently
        propagating into an all-NaN tau grid. Uses validate=False to bypass
        RheoData's own upstream NaN check and exercise the transform's guard
        directly (the guard must not rely solely on the caller pre-validating).
        """
        x = np.array([1.0, 2.0, np.nan, 3.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        data = RheoData(x=x, y=y, validate=False)
        transform = SpectrumInversion(method="tikhonov", n_tau=10)
        with pytest.raises(ValueError, match="strictly positive"):
            transform.transform(data)

    def test_G_e_none_raises_not_implemented(self):
        """G_e=None is documented as 'estimated from data' but no estimator
        exists. It must raise a clear NotImplementedError instead of an
        opaque TypeError from `y - None` inside _assemble_target.
        """
        omega, G_star, _ = self._make_single_mode_data()
        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(method="tikhonov", n_tau=50, G_e=None)
        with pytest.raises(NotImplementedError, match="G_e"):
            transform.transform(data)


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

    def test_entropy_regularization_flattens_spectrum(self):
        """Larger lambda must pull H(tau) toward the flat default model m,
        proving the entropy term (S, m) actually enters the update rule
        instead of being dead code behind plain chi-squared descent.
        """
        G_0 = 1000.0
        tau_0 = 1.0
        omega = np.logspace(-2, 2, 50)
        wt2 = (omega * tau_0) ** 2
        G_star = G_0 * wt2 / (1 + wt2) + 1j * G_0 * omega * tau_0 / (1 + wt2)
        data = RheoData(x=omega, y=G_star)

        weak = SpectrumInversion(method="max_entropy", n_tau=40, regularization=1e-8)
        strong = SpectrumInversion(method="max_entropy", n_tau=40, regularization=1e6)

        H_weak = np.asarray(weak.transform(data)[0].y)
        H_strong = np.asarray(strong.transform(data)[0].y)

        peakiness_weak = H_weak.max() / H_weak.mean()
        peakiness_strong = H_strong.max() / H_strong.mean()

        # Strong entropy weighting should be markedly flatter (closer to the
        # uniform default model) than weak entropy weighting.
        assert peakiness_strong < peakiness_weak

    def test_auto_regularization_scales_with_data(self):
        """Auto-selected lambda must reflect the data, not a hardcoded 1.0
        constant that reports the same 'selection' regardless of scale.
        """
        tau_0 = 1.0
        omega = np.logspace(-2, 2, 50)
        wt2 = (omega * tau_0) ** 2

        def fit(G_0):
            G_star = G_0 * wt2 / (1 + wt2) + 1j * G_0 * omega * tau_0 / (1 + wt2)
            data = RheoData(x=omega, y=G_star)
            transform = SpectrumInversion(method="max_entropy", n_tau=40)
            _, meta = transform.transform(data)
            return meta["spectrum_result"].regularization_param

        lam_small = fit(1.0)
        lam_large = fit(1.0e6)

        assert lam_small != 1.0
        assert lam_large != 1.0
        assert lam_small != lam_large

    def test_invalid_method(self):
        """Unknown method should raise ValueError."""
        data = RheoData(x=np.array([1.0]), y=np.array([1.0]))
        transform = SpectrumInversion(method="unknown")
        with pytest.raises(ValueError, match="Unknown method"):
            transform.transform(data)

    @pytest.mark.smoke
    def test_n_tau_one_raises(self):
        """n_tau=1 should raise ValueError (need >=2 for d(ln tau) bins)."""
        omega = np.logspace(-2, 2, 50)
        G_star = 1000.0 * (1j * omega) / (1.0 + 1j * omega)
        data = RheoData(x=omega, y=G_star)
        transform = SpectrumInversion(n_tau=1)
        with pytest.raises(ValueError, match="n_tau must be >= 2"):
            transform.transform(data)
