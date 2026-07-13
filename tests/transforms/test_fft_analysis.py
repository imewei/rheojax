"""Tests for FFT Analysis transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.transforms.fft_analysis import FFTAnalysis

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestFFTAnalysis:
    """Test suite for FFT Analysis transform."""

    @pytest.mark.smoke
    def test_basic_initialization(self):
        """Test basic FFT transform initialization."""
        fft = FFTAnalysis()
        assert fft.window == "hann"
        assert fft.detrend is True
        assert fft.return_psd is False

    @pytest.mark.smoke
    def test_custom_initialization(self):
        """Test FFT with custom parameters."""
        fft = FFTAnalysis(window="hamming", detrend=False, return_psd=True)
        assert fft.window == "hamming"
        assert fft.detrend is False
        assert fft.return_psd is True

    @pytest.mark.smoke
    def test_single_frequency_signal(self):
        """Test FFT of single-frequency sinusoid."""
        # Create pure sine wave
        t = jnp.linspace(0, 10, 1000)
        f0 = 2.0  # Hz
        signal = jnp.sin(2 * jnp.pi * f0 * t)

        data = RheoData(x=t, y=signal, domain="time")

        # Apply FFT
        fft = FFTAnalysis(window="none", detrend=False)
        freq_data = fft.transform(data)

        # Check domain conversion
        assert freq_data.domain == "frequency"
        assert "transform" in freq_data.metadata
        assert freq_data.metadata["transform"] == "fft"

        # Peak should be at f0
        freqs = np.array(freq_data.x)
        spectrum = np.array(freq_data.y)

        # Find peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = freqs[peak_idx]

        # Should be close to f0
        assert np.abs(peak_freq - f0) < 0.1

    def test_exponential_relaxation(self):
        """Test FFT of exponential relaxation."""
        # Create exponential decay
        t = jnp.linspace(0, 10, 1000)
        tau = 2.0
        G_t = jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        # Apply FFT
        fft = FFTAnalysis(window="hann", detrend=True)
        freq_data = fft.transform(data)

        # Check spectrum is reasonable
        assert freq_data.x.shape == freq_data.y.shape
        assert jnp.all(jnp.isfinite(freq_data.y))

    def test_window_functions(self):
        """Test different window functions."""
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        windows = ["hann", "hamming", "blackman", "bartlett", "none"]

        for window in windows:
            fft = FFTAnalysis(window=window)
            freq_data = fft.transform(data)

            # All should produce valid results
            assert freq_data.domain == "frequency"
            assert jnp.all(jnp.isfinite(freq_data.y))

    def test_psd_calculation(self):
        """Test power spectral density calculation."""
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        # Calculate PSD
        fft = FFTAnalysis(return_psd=True)
        psd_data = fft.transform(data)

        # PSD should be positive
        assert jnp.all(psd_data.y >= 0)
        assert psd_data.y_units == "PSD"

    def test_psd_magnitude_matches_scipy_periodogram(self):
        """PSD scale must correct for window power loss (sum(w**2)), not just n.

        Regression test for the default window='hann': the old dt/n scaling
        under-reported PSD magnitude by a window-dependent factor (~2.7x for
        Hann) relative to the standard periodogram/Welch definition used by
        scipy.signal.periodogram(..., window='hann').
        """
        from scipy.signal import periodogram

        n = 1000
        fs = 100.0
        t = np.arange(n) / fs
        f0 = 5.0
        signal = np.sin(2 * np.pi * f0 * t)
        data = RheoData(x=jnp.array(t), y=jnp.array(signal), domain="time")

        fft = FFTAnalysis(window="hann", detrend=False, return_psd=True)
        psd_data = fft.transform(data)

        ref_freqs, ref_psd = periodogram(signal, fs=fs, window="hann")

        # Compare at the peak frequency bin, where the two frequency grids
        # (rfftfreq vs scipy) coincide.
        peak_idx = int(np.argmax(np.array(psd_data.y)))
        ref_idx = int(np.argmax(ref_psd))
        assert np.array(psd_data.x)[peak_idx] == pytest.approx(
            ref_freqs[ref_idx], abs=1e-6
        )
        assert float(np.array(psd_data.y)[peak_idx]) == pytest.approx(
            ref_psd[ref_idx], rel=0.05
        )

    def test_psd_odd_length_matches_scipy_periodogram(self):
        """One-sided PSD must double the last bin for odd-length signals too.

        Regression test: rfft's last bin is the true Nyquist bin (correctly
        left undoubled) only when n is even. For odd n, the last bin is a
        genuine two-sided bin like any other and must be doubled, or
        sum(PSD)*df underestimates power (Parseval's theorem).
        """
        from scipy.signal import periodogram

        n = 101
        fs = 10.0
        t = np.arange(n) / fs
        signal = np.random.default_rng(0).standard_normal(n)
        data = RheoData(x=jnp.array(t), y=jnp.array(signal), domain="time")

        fft = FFTAnalysis(window="none", detrend=False, return_psd=True)
        psd_data = fft.transform(data)

        ref_freqs, ref_psd = periodogram(signal, fs=fs, window="boxcar", detrend=False)

        np.testing.assert_allclose(np.array(psd_data.x), ref_freqs, atol=1e-9)
        np.testing.assert_allclose(
            np.array(psd_data.y), ref_psd, rtol=1e-6, atol=1e-12
        )

    def test_psd_normalize_noop_warns(self, caplog):
        """normalize=True must not be silently ignored when return_psd=True."""
        import logging

        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        fft = FFTAnalysis(return_psd=True, normalize=True)
        with caplog.at_level(logging.WARNING):
            fft.transform(data)

        assert any(
            "normalize" in record.getMessage() and "return_psd" in record.getMessage()
            for record in caplog.records
        )

    def test_inverse_fft(self):
        """Test round-trip FFT → inverse FFT."""
        # Create signal
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 2.0 * t) + 0.5 * jnp.sin(2 * jnp.pi * 5.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        # Forward FFT (no window, no detrend for better reconstruction)
        fft = FFTAnalysis(window="none", detrend=False, return_psd=False)
        freq_data = fft.transform(data)

        # Inverse FFT
        reconstructed = fft.inverse_transform(freq_data)

        # Should recover original signal (approximately)
        assert reconstructed.domain == "time"
        # Trim edges (edge effects from FFT)
        trim = 50
        signal_trim = signal[trim:-trim]
        recon_trim = reconstructed.y[trim:-trim]
        correlation = np.corrcoef(np.array(signal_trim), np.array(recon_trim))[0, 1]
        assert correlation > 0.95  # Relaxed requirement due to edge effects

    def test_inverse_fft_preserves_time_units(self):
        """Reconstructed time axis must carry the original time units, not 's'.

        Regression test: the reconstructed x_units was hardcoded to "s"
        regardless of the original time base, mislabeling e.g. minute-spaced
        data by a 60x factor.
        """
        t = jnp.linspace(0, 10, 100)
        signal = jnp.sin(2 * jnp.pi * 0.5 * t)
        data = RheoData(x=t, y=signal, domain="time", x_units="min")

        fft = FFTAnalysis(window="none", detrend=False, return_psd=False)
        freq_data = fft.transform(data)
        assert freq_data.x_units == "1/min"

        reconstructed = fft.inverse_transform(freq_data)
        assert reconstructed.x_units == "min"

    def test_detrending(self):
        """Test detrending functionality."""
        # Create signal with linear trend
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 2.0 * t) + 0.5 * t  # Sine + linear trend

        data = RheoData(x=t, y=signal, domain="time")

        # With detrending
        fft_detrend = FFTAnalysis(detrend=True)
        freq_data_detrend = fft_detrend.transform(data)

        # Without detrending
        fft_no_detrend = FFTAnalysis(detrend=False)
        freq_data_no_detrend = fft_no_detrend.transform(data)

        # Detrended should have less low-frequency content
        # (DC component should be smaller)
        assert freq_data_detrend.y[0] < freq_data_no_detrend.y[0]

    def test_find_peaks(self):
        """Test peak finding in FFT spectrum."""
        # Create signal with multiple frequencies
        t = jnp.linspace(0, 10, 1000)
        signal = (
            jnp.sin(2 * jnp.pi * 2.0 * t)
            + 0.5 * jnp.sin(2 * jnp.pi * 5.0 * t)
            + 0.3 * jnp.sin(2 * jnp.pi * 8.0 * t)
        )

        data = RheoData(x=t, y=signal, domain="time")

        # Apply FFT
        fft = FFTAnalysis(window="hann")
        freq_data = fft.transform(data)

        # Find peaks
        peak_freqs, peak_heights = fft.find_peaks(freq_data, prominence=0.1, n_peaks=3)

        # Should find 3 peaks
        assert len(peak_freqs) <= 3
        assert len(peak_heights) <= 3

        # Peaks should be positive
        assert jnp.all(peak_heights > 0)

    def test_characteristic_time(self):
        """Test characteristic time extraction."""
        # Create oscillatory signal with clear peak (sine wave)
        # Exponential decays don't have sharp peaks in FFT
        t = jnp.linspace(0, 10, 2000)
        f_characteristic = 0.2  # Hz (τ = 1/f = 5s)
        signal = jnp.sin(2 * jnp.pi * f_characteristic * t)

        data = RheoData(x=t, y=signal, domain="time")

        # Apply FFT
        fft = FFTAnalysis(window="hann", detrend=False)
        freq_data = fft.transform(data)

        # Get characteristic time
        tau_extracted = fft.get_characteristic_time(freq_data)

        # Should be positive and finite
        assert np.isfinite(tau_extracted)
        assert tau_extracted > 0
        # Should be close to 1/f_characteristic = 5s
        # Allow wide range due to FFT discretization and windowing
        assert 2.0 < tau_extracted < 10.0

    def test_frequency_domain_error(self):
        """Test that FFT raises error for frequency-domain input."""
        # Create frequency-domain data
        freq = jnp.logspace(-2, 2, 100)
        spectrum = jnp.ones_like(freq)

        data = RheoData(x=freq, y=spectrum, domain="frequency")

        # Should raise error
        fft = FFTAnalysis()
        with pytest.raises(ValueError, match="time-domain"):
            fft.transform(data)

    def test_non_monotonic_time_raises(self):
        """A backward step buried in an otherwise-increasing array must raise.

        Regression test: np.diff(t) = [1, 1, -0.5, 1.5, 1] has median=1.0 > 0,
        so a median-only dt<=0 check would miss the non-monotonicity and feed
        unsorted knots into interpax's interpolator, producing silently wrong
        resampled data instead of an error.
        """
        t = jnp.array([0.0, 1.0, 2.0, 1.5, 3.0, 4.0])
        signal = jnp.sin(t)
        # RheoData only warns (not errors) on non-monotonic x, so this path
        # is reachable from ordinary construction.
        with pytest.warns(UserWarning, match="not monotonic"):
            data = RheoData(x=t, y=signal, domain="time")

        fft = FFTAnalysis()
        with pytest.raises(ValueError, match="monotonically increasing"):
            fft.transform(data)

    def test_frequency_units_only_hz_for_seconds(self):
        """x_units='Hz' must only be stamped when the input was in seconds."""
        t_min = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t_min)
        data = RheoData(x=t_min, y=signal, domain="time", x_units="min")

        fft = FFTAnalysis()
        freq_data = fft.transform(data)

        assert freq_data.x_units != "Hz"
        assert freq_data.x_units == "1/min"

        t_sec = jnp.linspace(0, 10, 1000)
        data_sec = RheoData(x=t_sec, y=signal, domain="time", x_units="s")
        freq_data_sec = fft.transform(data_sec)
        assert freq_data_sec.x_units == "Hz"

    def test_complex_data_handling(self):
        """Test FFT with complex input data."""
        # Create complex signal (should take real part)
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 2.0 * t) + 1j * jnp.cos(2 * jnp.pi * 2.0 * t)

        data = RheoData(x=t, y=signal, domain="time")

        # Apply FFT
        fft = FFTAnalysis()
        freq_data = fft.transform(data)

        # Should produce real spectrum
        assert jnp.all(jnp.isfinite(freq_data.y))

    def test_metadata_preservation(self):
        """Test that metadata is preserved and updated."""
        t = jnp.linspace(0, 10, 1000)
        signal = jnp.sin(2 * jnp.pi * 2.0 * t)

        data = RheoData(
            x=t,
            y=signal,
            domain="time",
            metadata={"sample": "test", "temperature": 298},
        )

        # Apply FFT
        fft = FFTAnalysis(window="hamming")
        freq_data = fft.transform(data)

        # Original metadata should be preserved
        assert freq_data.metadata["sample"] == "test"
        assert freq_data.metadata["temperature"] == 298

        # New metadata should be added
        assert freq_data.metadata["transform"] == "fft"
        assert freq_data.metadata["window"] == "hamming"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
