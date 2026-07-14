"""Tests for OWChirp transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.transforms.owchirp import OWChirp

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestOWChirp:
    """Test suite for OWChirp transform."""

    @pytest.mark.smoke
    def test_basic_initialization(self):
        """Test basic OWChirp initialization."""
        ow = OWChirp()
        assert ow.n_frequencies == 100
        assert ow.frequency_range == (1e-2, 1e2)
        assert ow.wavelet_width == 5.0
        assert ow.extract_harmonics is True

    @pytest.mark.smoke
    def test_custom_initialization(self):
        """Test OWChirp with custom parameters."""
        ow = OWChirp(
            n_frequencies=50,
            frequency_range=(0.1, 10),
            wavelet_width=3.0,
            extract_harmonics=False,
            max_harmonic=5,
        )
        assert ow.n_frequencies == 50
        assert ow.frequency_range == (0.1, 10)
        assert ow.wavelet_width == 3.0
        assert ow.extract_harmonics is False
        assert ow.max_harmonic == 5

    @pytest.mark.smoke
    def test_single_frequency_signal(self):
        """Test OWChirp on single-frequency oscillation."""
        # Create pure sine wave
        t = jnp.linspace(0, 100, 5000)
        f0 = 1.0  # Hz
        omega = 2 * jnp.pi * f0
        signal = jnp.sin(omega * t)

        data = RheoData(x=t, y=signal, domain="time")

        # Apply OWChirp
        ow = OWChirp(n_frequencies=50, frequency_range=(0.1, 10))
        spectrum = ow.transform(data)

        # Check output
        assert spectrum.domain == "frequency"
        assert len(spectrum.x) == 50
        assert len(spectrum.y) == 50

        # Peak should be near f0
        freqs = np.array(spectrum.x)
        spec = np.array(spectrum.y)
        peak_freq = freqs[np.argmax(spec)]

        # Should be close to f0
        assert 0.5 < peak_freq < 2.0

    def test_harmonic_extraction(self):
        """Test extraction of harmonics from nonlinear signal."""
        # Create signal with fundamental and 3rd harmonic
        t = jnp.linspace(0, 100, 5000)
        omega = 2 * jnp.pi * 1.0  # 1 Hz
        signal = jnp.sin(omega * t) + 0.3 * jnp.sin(3 * omega * t)

        data = RheoData(x=t, y=signal, domain="time")

        # Extract harmonics
        ow = OWChirp(
            n_frequencies=100, frequency_range=(0.1, 10), extract_harmonics=True
        )
        harmonics = ow.get_harmonics(data, fundamental_freq=1.0)

        # Should find fundamental and third harmonic
        assert "fundamental" in harmonics
        assert "third" in harmonics

        fundamental_freq, fundamental_amp = harmonics["fundamental"]
        third_freq, third_amp = harmonics["third"]

        # Frequencies should be correct
        assert 0.8 < fundamental_freq < 1.2
        assert 2.5 < third_freq < 3.5

        # Third harmonic should be weaker
        assert third_amp < fundamental_amp

    def test_laos_signal(self):
        """Test OWChirp on LAOS-like signal with multiple harmonics."""
        # Simulate LAOS response with odd harmonics
        t = jnp.linspace(0, 50, 5000)
        omega = 2 * jnp.pi * 2.0  # 2 Hz
        signal = (
            jnp.sin(omega * t)
            + 0.2 * jnp.sin(3 * omega * t)
            + 0.1 * jnp.sin(5 * omega * t)
        )

        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(extract_harmonics=True, max_harmonic=7)
        spectrum = ow.transform(data)

        # Should produce valid spectrum
        assert jnp.all(jnp.isfinite(spectrum.y))
        assert jnp.all(spectrum.y >= 0)

    def test_time_frequency_map(self):
        """Test full time-frequency map extraction."""
        # Create chirp-like signal
        t = jnp.linspace(0, 20, 2000)
        # Frequency increases with time
        phase = 2 * jnp.pi * (0.5 * t + 0.05 * t**2)
        signal = jnp.sin(phase)

        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(n_frequencies=30, frequency_range=(0.1, 5))
        times, freqs, coeffs = ow.get_time_frequency_map(data)

        # Check dimensions
        assert len(times) == len(t)
        assert len(freqs) == 30
        assert coeffs.shape == (30, len(t))

        # Coefficients should be complex
        assert jnp.iscomplexobj(coeffs)

    def test_automatic_fundamental_detection(self):
        """Test automatic detection of fundamental frequency."""
        # Create signal with clear fundamental
        t = jnp.linspace(0, 50, 5000)
        f0 = 1.5  # Hz
        omega = 2 * jnp.pi * f0
        signal = jnp.sin(omega * t) + 0.1 * jnp.sin(3 * omega * t)

        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(extract_harmonics=True)
        harmonics = ow.get_harmonics(data)  # No fundamental_freq provided

        # Should auto-detect fundamental
        fundamental_freq, _ = harmonics["fundamental"]

        # Should be close to f0
        assert 1.0 < fundamental_freq < 2.0

    def test_wavelet_transform_consistency(self):
        """Test that wavelet transform is consistent."""
        t = jnp.linspace(0, 20, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        # Apply transform twice
        ow = OWChirp(n_frequencies=20)
        spectrum1 = ow.transform(data)
        spectrum2 = ow.transform(data)

        # Should be identical
        assert jnp.allclose(spectrum1.y, spectrum2.y)

    def test_frequency_domain_error(self):
        """Test that OWChirp raises error for frequency-domain input."""
        freq = jnp.logspace(-2, 2, 100)
        spectrum = jnp.ones_like(freq)
        data = RheoData(x=freq, y=spectrum, domain="frequency")

        ow = OWChirp()
        with pytest.raises(ValueError, match="time-domain"):
            ow.transform(data)

    def test_complex_data_handling(self):
        """Test handling of complex input data."""
        t = jnp.linspace(0, 20, 1000)
        # Complex signal (should take real part)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t) + 1j * jnp.cos(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp()
        spectrum = ow.transform(data)

        # Should produce valid real spectrum
        assert jnp.all(jnp.isfinite(spectrum.y))

    def test_wavelet_width_effect(self):
        """Test effect of wavelet width on resolution."""
        t = jnp.linspace(0, 50, 2000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        # Narrow wavelet (better frequency resolution)
        ow_narrow = OWChirp(wavelet_width=2.0, n_frequencies=50)
        spectrum_narrow = ow_narrow.transform(data)

        # Wide wavelet (better time resolution)
        ow_wide = OWChirp(wavelet_width=10.0, n_frequencies=50)
        spectrum_wide = ow_wide.transform(data)

        # Both should be valid
        assert jnp.all(jnp.isfinite(spectrum_narrow.y))
        assert jnp.all(jnp.isfinite(spectrum_wide.y))

    def test_nonzero_time_origin_not_all_zero(self):
        """Regression: a signal with a nonzero absolute time origin must not
        produce an all-zero spectrum (was caused by the wavelet being built
        at a fixed t_center=0.0 on the raw physical time array)."""
        t = jnp.linspace(1000, 1100, 5000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(n_frequencies=50, frequency_range=(0.1, 10))
        spectrum = ow.transform(data)

        assert float(jnp.max(spectrum.y)) > 0.0

        freqs = np.array(spectrum.x)
        spec = np.array(spectrum.y)
        peak_freq = freqs[np.argmax(spec)]
        assert 0.5 < peak_freq < 2.0

    def test_high_frequency_peak_not_mis_centered(self):
        """Regression: a 5 Hz signal must peak near 5 Hz, not collapse to the
        frequency_range minimum (caused by the mis-centered, causal-clipped
        wavelet envelope that narrows and loses energy as frequency grows)."""
        t = jnp.linspace(0, 100, 5000)
        signal = jnp.sin(2 * jnp.pi * 5.0 * t)
        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(n_frequencies=50, frequency_range=(0.1, 10))
        spectrum = ow.transform(data)

        freqs = np.array(spectrum.x)
        spec = np.array(spectrum.y)
        peak_freq = freqs[np.argmax(spec)]
        assert 4.0 < peak_freq < 6.5

    def test_amplitude_calibration_across_frequencies(self):
        """Regression: unit-amplitude sinusoids at different frequencies must
        yield comparable peak spectrum magnitude (was >100x apart due to the
        unnormalized Gaussian envelope combined with a mismatched /sqrt(f)
        compensation)."""
        t = jnp.linspace(0, 100, 5000)
        peaks = []
        for f in (0.1, 1.0, 10.0):
            signal = jnp.sin(2 * jnp.pi * f * t)
            data = RheoData(x=t, y=signal, domain="time")
            ow = OWChirp(n_frequencies=80, frequency_range=(0.01, 100))
            spectrum = ow.transform(data)
            peaks.append(float(jnp.max(spectrum.y)))

        # Previously ranged from 16.81 to 0.13 (>100x). Now should be tight.
        assert max(peaks) / min(peaks) < 2.0

    def test_harmonic_amplitude_ratio_matches_true_nonlinearity(self):
        """Regression: I3/I1 extracted via get_harmonics must match the true
        injected nonlinearity ratio, not be skewed by frequency-dependent
        normalization defects."""
        t = jnp.linspace(0, 100, 5000)
        omega = 2 * jnp.pi * 1.0
        signal = jnp.sin(omega * t) + 0.3 * jnp.sin(3 * omega * t)
        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(
            n_frequencies=80, frequency_range=(0.1, 10), extract_harmonics=True
        )
        harmonics = ow.get_harmonics(data, fundamental_freq=1.0)

        _, fundamental_amp = harmonics["fundamental"]
        _, third_amp = harmonics["third"]
        ratio = third_amp / fundamental_amp

        # True ratio is 0.3; previously measured 0.079 (off by 3.8x).
        assert 0.2 < ratio < 0.4

    def test_frequency_range_lower_bound_must_be_positive(self):
        """Regression: frequency_range[0] <= 0 must raise, not silently
        collapse most of the log-spaced frequency array to 0.0."""
        with pytest.raises(ValueError, match="frequency_range"):
            OWChirp(n_frequencies=10, frequency_range=(0.0, 10.0))

        with pytest.raises(ValueError, match="frequency_range"):
            OWChirp(frequency_range=(-1.0, 10.0))

    def test_max_harmonic_above_ninth_is_not_silently_dropped(self):
        """Regression: max_harmonic > 9 (e.g. 11) must appear in the returned
        dict instead of being computed and discarded."""
        t = jnp.linspace(0, 100, 5000)
        omega = 2 * jnp.pi * 1.0
        signal = jnp.sin(omega * t) + 0.1 * jnp.sin(11 * omega * t)
        data = RheoData(x=t, y=signal, domain="time")

        ow = OWChirp(
            n_frequencies=100,
            frequency_range=(0.1, 15),
            extract_harmonics=True,
            max_harmonic=11,
        )
        harmonics = ow.get_harmonics(data, fundamental_freq=1.0)

        assert any(
            freq_amp[0] == pytest.approx(11.0) for freq_amp in harmonics.values()
        )

    def test_direct_and_fft_wavelet_paths_agree(self):
        """Regression: the private direct/vmap implementation (_wavelet_transform)
        must agree with the production FFT-based path (_optimized_wavelet_transform)
        so it cannot silently diverge if ever wired up or reused."""
        t = jnp.linspace(0, 20, 200)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)
        signal = signal - jnp.mean(signal)
        frequencies = jnp.array([0.5, 1.0, 2.0])

        ow = OWChirp()
        direct = ow._wavelet_transform(t, signal, frequencies)
        fft_based = ow._optimized_wavelet_transform(t, signal, frequencies)

        direct_spectrum = jnp.mean(jnp.abs(direct), axis=1)
        fft_spectrum = jnp.mean(jnp.abs(fft_based), axis=1)

        assert jnp.allclose(direct_spectrum, fft_spectrum, rtol=0.15)

    def test_non_second_x_units_are_not_mislabeled_as_hz(self):
        """Regression: a signal stored with a non-second time axis (e.g.
        minutes) must not have its output frequency axis mislabeled "Hz"
        with the wrong numeric scale (was ~60x off for minutes)."""
        f0_hz = 1.0
        t_seconds = jnp.linspace(0, 120, 6000)  # 2 physical minutes @ 50 Hz
        signal = jnp.sin(2 * jnp.pi * f0_hz * t_seconds)
        t_minutes = t_seconds / 60.0

        data = RheoData(x=t_minutes, y=signal, x_units="min", domain="time")

        ow = OWChirp(n_frequencies=50, frequency_range=(0.1, 10))
        spectrum = ow.transform(data)

        assert spectrum.x_units == "Hz"

        freqs = np.array(spectrum.x)
        spec = np.array(spectrum.y)
        peak_freq = freqs[np.argmax(spec)]
        assert 0.5 < peak_freq < 2.0

    def test_metadata_preservation(self):
        """Test metadata preservation."""
        t = jnp.linspace(0, 20, 1000)
        signal = jnp.sin(2 * jnp.pi * 1.0 * t)

        data = RheoData(
            x=t,
            y=signal,
            domain="time",
            metadata={"sample": "gel", "strain_amplitude": 1.0},
        )

        ow = OWChirp()
        spectrum = ow.transform(data)

        # Original metadata preserved
        assert spectrum.metadata["sample"] == "gel"
        assert spectrum.metadata["strain_amplitude"] == 1.0

        # Transform metadata added
        assert "transform" in spectrum.metadata
        assert spectrum.metadata["transform"] == "owchirp"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
