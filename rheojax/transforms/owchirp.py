"""Optimally Windowed Chirp (OWChirp) transform for LAOS analysis.

This module implements the OWChirp transform for analyzing Large Amplitude
Oscillatory Shear (LAOS) data, providing time-frequency analysis and nonlinear
viscoelastic parameter extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.inventory import TransformType
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Array alias for runtime isinstance checks (via safe_import_jax handle)
Array = jax.Array

# Module logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from rheojax.core.data import RheoData


@TransformRegistry.register("owchirp", type=TransformType.ANALYSIS)
class OWChirp(BaseTransform):
    """Optimally Windowed Chirp transform for LAOS data analysis.

    The OWChirp transform uses chirp wavelets to perform time-frequency
    analysis of Large Amplitude Oscillatory Shear (LAOS) data, extracting
    nonlinear viscoelastic parameters and higher harmonics.

    This is particularly useful for:
    - Analyzing frequency-dependent nonlinear response
    - Extracting time-varying moduli during LAOS
    - Identifying structural changes during oscillatory deformation
    - Higher harmonic analysis (3rd, 5th, 7th harmonics)

    The transform uses a Morlet-like chirp wavelet that is optimally windowed
    to balance time and frequency resolution.

    Parameters
    ----------
    n_frequencies : int, default=100
        Number of frequency points for analysis
    frequency_range : tuple, default=(1e-2, 1e2)
        Frequency range (f_min, f_max) in Hz
    wavelet_width : float, default=5.0
        Width parameter for wavelet (controls time-frequency resolution)
    extract_harmonics : bool, default=True
        Whether to extract higher harmonics (3ω, 5ω, etc.)
    max_harmonic : int, default=7
        Maximum harmonic to extract (odd harmonics only)

    Examples
    --------
    Basic usage:

    >>> from rheojax.core.data import RheoData
    >>> from rheojax.transforms.owchirp import OWChirp
    >>>
    >>> # LAOS stress response data
    >>> t = jnp.linspace(0, 100, 10000)
    >>> omega = 1.0  # rad/s
    >>> # Nonlinear stress: includes 3rd harmonic
    >>> stress = jnp.sin(omega * t) + 0.2 * jnp.sin(3 * omega * t)
    >>> data = RheoData(x=t, y=stress, domain='time',
    ...                 metadata={'test_mode': 'oscillation'})
    >>>
    >>> # Apply OWChirp transform
    >>> owchirp = OWChirp(n_frequencies=50, extract_harmonics=True)
    >>> spectrum = owchirp.transform(data)
    >>>
    >>> # Extract nonlinear parameters
    >>> harmonics = owchirp.get_harmonics(data)
    """

    def __init__(
        self,
        n_frequencies: int = 100,
        frequency_range: tuple[float, float] = (1e-2, 1e2),
        wavelet_width: float = 5.0,
        extract_harmonics: bool = True,
        max_harmonic: int = 7,
    ):
        """Initialize OWChirp transform.

        Parameters
        ----------
        n_frequencies : int
            Number of frequency points
        frequency_range : tuple
            (f_min, f_max) in Hz
        wavelet_width : float
            Wavelet width parameter
        extract_harmonics : bool
            Extract higher harmonics
        max_harmonic : int
            Maximum harmonic order
        """
        super().__init__()
        self.n_frequencies = n_frequencies
        self.frequency_range = frequency_range
        self.wavelet_width = wavelet_width
        self.extract_harmonics = extract_harmonics
        self.max_harmonic = max_harmonic

    def _chirp_wavelet(
        self, t: Array, t_center: float, frequency: float, width: float
    ) -> Array:
        """Generate chirp wavelet at given frequency.

        The chirp wavelet is a Morlet-like wavelet with a Gaussian envelope:
            ψ(t) = exp(-((t-t_c)/σ)²) * exp(2πi*f*t)

        Parameters
        ----------
        t : Array
            Time array
        t_center : float
            Center time of wavelet
        frequency : float
            Frequency in Hz
        width : float
            Width parameter (controls localization)

        Returns
        -------
        Array
            Complex wavelet coefficients
        """
        # R7-OWC-001: Guard against frequency=0 to prevent division by zero
        # in sigma computation. Use a small epsilon floor.
        freq_safe = jnp.maximum(frequency, 1e-30)

        # Gaussian envelope width
        sigma = width / (2.0 * jnp.pi * freq_safe)

        # Gaussian envelope
        envelope = jnp.exp(-0.5 * (((t - t_center) / sigma) ** 2))

        # Complex exponential (chirp)
        omega = 2.0 * jnp.pi * frequency
        chirp = jnp.exp(1j * omega * t)

        return envelope * chirp

    def _wavelet_transform(self, t: Array, signal: Array, frequencies: Array) -> Array:
        """Compute wavelet transform of signal.

        Uses vectorized JAX operations (vmap) instead of nested Python loops
        for O(n_freqs * n_times) computation without Python-level overhead.

        Parameters
        ----------
        t : Array
            Time array
        signal : Array
            Input signal
        frequencies : Array
            Frequency array

        Returns
        -------
        Array
            Wavelet coefficients (n_frequencies, n_times)
        """
        # TRANS-001: Compute dt once (invariant to freq, t_center)
        # R11-OWC-003: Use median dt for robustness to non-uniform sampling.
        # Warn if spacing is non-uniform (>5% variation) — reduces CWT accuracy.
        if len(t) > 1:
            dt_arr = jnp.diff(t)
            _dt_med = float(jnp.median(dt_arr))
            _dt_std = float(jnp.std(dt_arr))
            if _dt_med > 0 and (_dt_std / _dt_med) > 0.05:
                import warnings as _warnings

                _warnings.warn(
                    f"OWChirp (vmap path): non-uniform time spacing detected "
                    f"(std/median = {_dt_std / _dt_med:.2f}). "
                    "Results may be inaccurate.",
                    UserWarning,
                    stacklevel=2,
                )
        dt = jnp.where(len(t) > 1, jnp.median(jnp.diff(t)), 1.0)

        # Vectorize over (freq, t_center) pairs using vmap
        def compute_coeff(freq, t_center):
            wavelet = self._chirp_wavelet(t, t_center, freq, self.wavelet_width)
            return jnp.sum(signal * jnp.conj(wavelet)) * dt

        # vmap over t_center (inner), then over freq (outer)
        compute_row = jax.vmap(compute_coeff, in_axes=(None, 0))  # over t_centers
        compute_all = jax.vmap(compute_row, in_axes=(0, None))  # over freqs

        coefficients = compute_all(jnp.asarray(frequencies), t)
        return coefficients

    def _optimized_wavelet_transform(
        self, t: Array, signal: Array, frequencies: Array
    ) -> Array:
        """Optimized wavelet transform using FFT convolution.

        This is much faster than the direct method for long signals.

        Parameters
        ----------
        t : Array
            Time array
        signal : Array
            Input signal
        frequencies : Array
            Frequency array

        Returns
        -------
        Array
            Wavelet coefficients
        """
        if len(t) < 2:
            raise ValueError("Wavelet transform requires at least 2 time points")
        dt_arr = jnp.diff(t)
        dt = float(jnp.median(dt_arr))
        # R11-OWC-001: Use median dt for robustness to non-uniform sampling.
        # Warn when spacing varies more than 5% — non-uniform dt reduces CWT accuracy.
        dt_std = float(jnp.std(dt_arr))
        if dt > 0 and (dt_std / dt) > 0.05:
            import warnings as _warnings

            _warnings.warn(
                f"OWChirp: non-uniform time spacing detected "
                f"(std/median = {dt_std / dt:.2f}). "
                "FFT-based CWT assumes uniform dt — results may be inaccurate. "
                "Interpolate to a uniform grid before transforming.",
                UserWarning,
                stacklevel=2,
            )

        # R10-OWC-002: Zero-pad to 2× length for linear (non-circular) correlation.
        # The FFT-based cross-correlation is circular by default; zero-padding to at
        # least 2N prevents wrap-around aliasing in the time domain.
        n_orig = len(t)
        n_pad = 2 * n_orig

        # TR-01: Vectorized batched FFT — replaces the Python for-loop over
        # frequencies (which issued 200 sequential FFT calls) with 3 batched
        # operations: one fft on the wavelet matrix, one pointwise multiply, one
        # ifft.  Shape legend: F = n_frequencies, N = n_orig, P = n_pad.

        # Build wavelet matrix (F, P) — all wavelets zero-padded in one shot.
        # vmap over frequencies; each call produces a length-n_orig complex array
        # that is then zero-padded to n_pad.
        def _make_wavelet_row(freq: Array) -> Array:
            """Return zero-padded wavelet for a single frequency (shape: (n_pad,))."""
            # Center at t=0 per R10-OWC-002 convention.
            wavelet = self._chirp_wavelet(t, 0.0, freq, self.wavelet_width)
            return jnp.pad(wavelet, (0, n_pad - n_orig))

        # wavelet_matrix: (F, P)
        wavelet_matrix = jax.vmap(_make_wavelet_row)(jnp.asarray(frequencies))

        # Single batched FFT of all wavelets: (F, P)
        wavelet_fft_matrix = jnp.fft.fft(wavelet_matrix, axis=-1)

        # Signal: pad once and FFT once → (P,)
        signal_padded = jnp.pad(signal, (0, n_pad - n_orig))
        signal_fft = jnp.fft.fft(signal_padded)  # (P,)

        # Cross-correlation in frequency domain; broadcast signal_fft over F axis.
        # signal_fft[None, :] is (1, P); result is (F, P).
        conv_fft = signal_fft[None, :] * jnp.conj(wavelet_fft_matrix)

        # Single batched IFFT: (F, P) → trim to (F, N)
        conv_full = jnp.fft.ifft(conv_fft, axis=-1)
        conv_trimmed = conv_full[:, :n_orig]  # (F, N)

        # Apply 1/√f scale normalization (standard L² CWT normalization).
        # TR-02: Use jnp.maximum instead of Python max() — avoids device→host
        # transfer when frequencies is a JAX array.
        # R7-OWC-002: Guard against freq=0 (logspace guarantees positive values
        # but defend against edge cases in direct calls).
        freq_safe = jnp.maximum(jnp.asarray(frequencies), 1e-30)  # (F,)
        scale = jnp.sqrt(freq_safe)[:, None]  # (F, 1) for broadcasting over N

        coefficients = conv_trimmed / scale  # (F, N)

        return coefficients * dt

    def _transform(self, data: RheoData) -> RheoData:
        """Apply OWChirp transform to LAOS data.

        Parameters
        ----------
        data : RheoData
            Time-domain LAOS data (stress or strain)

        Returns
        -------
        RheoData
            Time-frequency spectrum

        Raises
        ------
        ValueError
            If data is not time-domain
        """
        from rheojax.core.data import RheoData

        logger.info(
            "Starting OWChirp transform",
            n_frequencies=self.n_frequencies,
            frequency_range=self.frequency_range,
            wavelet_width=self.wavelet_width,
            extract_harmonics=self.extract_harmonics,
        )

        # Validate domain
        if data.domain != "time":
            logger.error(
                "Invalid domain for OWChirp",
                expected="time",
                got=data.domain,
            )
            raise ValueError("OWChirp requires time-domain data")

        # Get time and signal
        t = data.x
        signal = data.y

        logger.debug(
            "Input data extracted",
            data_points=len(t),
            domain=data.domain,
        )

        # Convert to JAX arrays
        if not isinstance(t, Array):
            t = jnp.array(t)
        if not isinstance(signal, Array):
            signal = jnp.array(signal)

        # Handle complex data
        if jnp.iscomplexobj(signal):
            logger.debug("Converting complex signal to real part")
            signal = jnp.real(signal)

        # R11-OWC-002: Remove DC offset to prevent spurious low-frequency peak
        signal = signal - jnp.mean(signal)

        # Generate frequency array (log-spaced)
        logger.debug(
            "Generating frequency array",
            f_min=self.frequency_range[0],
            f_max=self.frequency_range[1],
            n_frequencies=self.n_frequencies,
        )
        frequencies = jnp.logspace(
            jnp.log10(self.frequency_range[0]),
            jnp.log10(self.frequency_range[1]),
            self.n_frequencies,
        )

        # Compute wavelet transform (use optimized FFT method)
        logger.debug("Computing optimized wavelet transform using FFT convolution")
        coefficients = self._optimized_wavelet_transform(t, signal, frequencies)

        # Compute magnitude spectrum (average over time)
        # R8-OWC-001: averaging over time axis discards time-localization information;
        # for time-resolved analysis, use the full 2D coefficients array directly.
        logger.debug("Computing magnitude spectrum")
        logger.info(
            "Wavelet coefficients averaged over time axis. "
            "For time-resolved analysis, use the raw coefficients from "
            "_optimized_wavelet_transform() before averaging."
        )
        spectrum = jnp.mean(jnp.abs(coefficients), axis=1)

        # Create metadata
        new_metadata = (data.metadata or {}).copy()
        new_metadata.update(
            {
                "transform": "owchirp",
                "wavelet_width": self.wavelet_width,
                "n_frequencies": self.n_frequencies,
                "frequency_range": self.frequency_range,
                # R9-OWC-001: Only the time-averaged spectrum is returned by
                # _transform(). The full 2D map requires get_time_frequency_map().
                "time_frequency_map": False,
            }
        )

        logger.info(
            "OWChirp transform completed",
            output_frequencies=len(frequencies),
            spectrum_shape=spectrum.shape,
        )

        # Return frequency-domain data (averaged)
        return RheoData(
            x=frequencies,
            y=spectrum,
            x_units="Hz",
            y_units="magnitude",
            domain="frequency",
            metadata=new_metadata,
            validate=False,
        )

    def get_time_frequency_map(self, data: RheoData) -> tuple[Array, Array, Array]:
        """Get full time-frequency map (spectrogram).

        Parameters
        ----------
        data : RheoData
            Time-domain LAOS data

        Returns
        -------
        times : Array
            Time array
        frequencies : Array
            Frequency array
        coefficients : Array
            Complex wavelet coefficients (n_frequencies, n_times)
        """
        # Get time and signal
        t = data.x
        signal = data.y

        # Convert to JAX arrays
        if not isinstance(t, Array):
            t = jnp.array(t)
        if not isinstance(signal, Array):
            signal = jnp.array(signal)

        # Handle complex
        if jnp.iscomplexobj(signal):
            signal = jnp.real(signal)

        # R11-OWC-002: Remove DC offset to prevent spurious low-frequency peak
        signal = signal - jnp.mean(signal)

        # Generate frequencies
        frequencies = jnp.logspace(
            jnp.log10(self.frequency_range[0]),
            jnp.log10(self.frequency_range[1]),
            self.n_frequencies,
        )

        # Compute wavelet transform
        coefficients = self._optimized_wavelet_transform(t, signal, frequencies)

        return t, frequencies, coefficients

    def get_harmonics(
        self, data: RheoData, fundamental_freq: float | None = None
    ) -> dict:
        """Extract harmonic content from LAOS data.

        Parameters
        ----------
        data : RheoData
            Time-domain LAOS data
        fundamental_freq : float, optional
            Fundamental frequency in Hz. If None, auto-detect from FFT peak.

        Returns
        -------
        dict
            Dictionary with harmonic amplitudes::

                {'fundamental': (freq, amplitude),
                 'third': (3*freq, amplitude),
                 'fifth': (5*freq, amplitude),
                 ...}
        """
        logger.info(
            "Extracting harmonics",
            fundamental_freq=fundamental_freq,
            max_harmonic=self.max_harmonic,
        )

        # Get frequency spectrum
        freq_data = self.transform(data)
        freqs = freq_data.x
        spectrum = freq_data.y

        # Convert to numpy for peak detection
        if isinstance(freqs, Array):
            freqs = np.array(freqs)
        if isinstance(spectrum, Array):
            spectrum = np.array(spectrum)

        # Find fundamental frequency if not provided
        if fundamental_freq is None:
            logger.debug("Auto-detecting fundamental frequency from FFT peak")
            # Find peak in spectrum
            from scipy.signal import find_peaks

            # Use lower prominence threshold (1% of max) to detect peaks more reliably
            peaks, properties = find_peaks(spectrum, prominence=0.01 * np.max(spectrum))

            if len(peaks) == 0:
                # Fallback: use the frequency with maximum amplitude
                logger.debug(
                    "No peaks detected with prominence threshold, using max amplitude"
                )
                max_idx = np.argmax(spectrum)
                fundamental_freq = float(freqs[max_idx])
                logger.debug(
                    "Fundamental frequency from max amplitude",
                    fundamental_freq=fundamental_freq,
                )
            else:
                # Fundamental is typically the strongest peak
                strongest_peak = peaks[np.argmax(spectrum[peaks])]
                fundamental_freq = float(freqs[strongest_peak])
                logger.debug(
                    "Fundamental frequency detected",
                    fundamental_freq=fundamental_freq,
                    n_peaks_found=len(peaks),
                )

        # Extract harmonics
        harmonics = {}
        harmonics["fundamental"] = (
            fundamental_freq,
            self._get_amplitude_at_freq(freqs, spectrum, fundamental_freq),
        )

        if self.extract_harmonics:
            logger.debug(
                "Extracting odd harmonics",
                max_harmonic=self.max_harmonic,
            )
            # Extract odd harmonics up to max_harmonic
            for n in range(3, self.max_harmonic + 1, 2):
                harmonic_freq = n * fundamental_freq
                amplitude = self._get_amplitude_at_freq(freqs, spectrum, harmonic_freq)

                harmonic_name = {3: "third", 5: "fifth", 7: "seventh", 9: "ninth"}
                if n in harmonic_name:
                    harmonics[harmonic_name[n]] = (harmonic_freq, amplitude)

        logger.info(
            "Harmonic extraction completed",
            n_harmonics=len(harmonics),
            fundamental_freq=fundamental_freq,
        )

        return harmonics

    def _get_amplitude_at_freq(
        self,
        freqs: np.ndarray,
        spectrum: np.ndarray,
        target_freq: float,
        window: float = 0.1,
    ) -> float:
        """Get amplitude at specific frequency (with averaging window).

        Parameters
        ----------
        freqs : np.ndarray
            Frequency array
        spectrum : np.ndarray
            Spectrum values
        target_freq : float
            Target frequency
        window : float
            Fractional window for averaging (e.g., 0.1 = ±10%)

        Returns
        -------
        float
            Amplitude at target frequency
        """
        # Find frequencies within window
        f_min = target_freq * (1 - window)
        f_max = target_freq * (1 + window)

        mask = (freqs >= f_min) & (freqs <= f_max)

        if np.sum(mask) == 0:
            return 0.0

        # Return maximum in window
        return float(np.max(spectrum[mask]))


__all__ = ["OWChirp"]
