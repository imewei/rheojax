"""Optimally Windowed Chirp (OWChirp) transform for LAOS analysis.

This module implements the OWChirp transform for analyzing Large Amplitude
Oscillatory Shear (LAOS) data, providing time-frequency analysis and nonlinear
viscoelastic parameter extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.base import BaseTransform
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import TransformRegistry
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Import Array for runtime isinstance checks
from jax import Array

# Module logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from rheojax.core.data import RheoData


@TransformRegistry.register("owchirp")
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
        # Gaussian envelope width
        sigma = width / (2.0 * jnp.pi * frequency)

        # Gaussian envelope
        envelope = jnp.exp(-(((t - t_center) / sigma) ** 2))

        # Complex exponential (chirp)
        omega = 2.0 * jnp.pi * frequency
        chirp = jnp.exp(1j * omega * t)

        return envelope * chirp

    def _wavelet_transform(self, t: Array, signal: Array, frequencies: Array) -> Array:
        """Compute wavelet transform of signal.

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
        n_times = len(t)
        n_freqs = len(frequencies)

        # Initialize coefficient array
        coefficients = jnp.zeros((n_freqs, n_times), dtype=jnp.complex64)

        # Compute wavelet transform at each frequency
        for i, freq in enumerate(frequencies):
            # For each time point, compute wavelet convolution
            # Use sliding window approach
            for j, t_center in enumerate(t):
                wavelet = self._chirp_wavelet(t, t_center, freq, self.wavelet_width)
                # Inner product
                coeff = jnp.sum(signal * jnp.conj(wavelet)) * (t[1] - t[0])
                coefficients = coefficients.at[i, j].set(coeff)

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
        dt = t[1] - t[0]
        len(t)

        coefficients_list = []

        for freq in frequencies:
            # Create wavelet centered at middle
            t_center = (t[0] + t[-1]) / 2.0
            wavelet = self._chirp_wavelet(t, t_center, freq, self.wavelet_width)

            # FFT-based convolution
            signal_fft = jnp.fft.fft(signal)
            wavelet_fft = jnp.fft.fft(wavelet)
            conv = jnp.fft.ifft(signal_fft * jnp.conj(wavelet_fft))

            coefficients_list.append(conv)

        coefficients = jnp.stack(coefficients_list, axis=0)

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
        logger.debug("Computing magnitude spectrum")
        spectrum = jnp.mean(jnp.abs(coefficients), axis=1)

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update(
            {
                "transform": "owchirp",
                "wavelet_width": self.wavelet_width,
                "n_frequencies": self.n_frequencies,
                "frequency_range": self.frequency_range,
                "time_frequency_map": True,  # Full 2D map available
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

            peaks, properties = find_peaks(spectrum, prominence=0.1 * np.max(spectrum))

            if len(peaks) == 0:
                logger.error("Could not detect fundamental frequency in spectrum")
                raise ValueError("Could not detect fundamental frequency")

            # Fundamental is typically the strongest peak
            strongest_peak = peaks[np.argmax(spectrum[peaks])]
            fundamental_freq = freqs[strongest_peak]
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
