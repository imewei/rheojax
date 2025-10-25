"""FFT-based frequency analysis transform for rheological data.

This module provides FFT analysis to convert time-domain rheological data
(relaxation, creep) to frequency domain for spectral analysis and feature extraction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Literal

import jax.numpy as jnp
import numpy as np

from rheo.core.base import BaseTransform
from rheo.core.registry import TransformRegistry

if TYPE_CHECKING:
    from rheo.core.data import RheoData


WindowType = Literal['hann', 'hamming', 'blackman', 'bartlett', 'none']


@TransformRegistry.register('fft_analysis')
class FFTAnalysis(BaseTransform):
    """Transform time-domain rheological data to frequency domain using FFT.

    This transform applies Fast Fourier Transform to convert time-domain signals
    to frequency domain, enabling analysis of characteristic frequencies, relaxation
    time distributions, and spectral features.

    Features:
    - Multiple window functions (Hann, Hamming, Blackman, Bartlett)
    - Optional detrending to remove DC offset
    - Power spectral density (PSD) calculation
    - Peak detection for characteristic frequencies
    - JAX-accelerated computation

    Parameters
    ----------
    window : WindowType, default='hann'
        Window function to apply before FFT. Options: 'hann', 'hamming',
        'blackman', 'bartlett', 'none'
    detrend : bool, default=True
        Whether to remove linear trend before FFT
    return_psd : bool, default=False
        If True, return power spectral density instead of FFT magnitude
    normalize : bool, default=True
        Whether to normalize the FFT result

    Examples
    --------
    >>> from rheo.core.data import RheoData
    >>> from rheo.transforms.fft_analysis import FFTAnalysis
    >>>
    >>> # Create time-domain relaxation data
    >>> t = jnp.linspace(0, 10, 1000)
    >>> G_t = jnp.exp(-t/2.0)  # Exponential relaxation
    >>> data = RheoData(x=t, y=G_t, domain='time')
    >>>
    >>> # Apply FFT analysis
    >>> fft = FFTAnalysis(window='hann', detrend=True)
    >>> freq_data = fft.transform(data)
    >>>
    >>> # freq_data.x contains frequencies, freq_data.y contains spectrum
    """

    def __init__(
        self,
        window: WindowType = 'hann',
        detrend: bool = True,
        return_psd: bool = False,
        normalize: bool = True
    ):
        """Initialize FFT Analysis transform.

        Parameters
        ----------
        window : WindowType
            Window function to apply
        detrend : bool
            Whether to detrend data before FFT
        return_psd : bool
            Return power spectral density instead of magnitude
        normalize : bool
            Normalize FFT output
        """
        super().__init__()
        self.window = window
        self.detrend = detrend
        self.return_psd = return_psd
        self.normalize = normalize

    def _get_window(self, n: int) -> jnp.ndarray:
        """Get window function of length n.

        Parameters
        ----------
        n : int
            Length of window

        Returns
        -------
        jnp.ndarray
            Window coefficients
        """
        if self.window == 'hann':
            return jnp.hanning(n)
        elif self.window == 'hamming':
            return jnp.hamming(n)
        elif self.window == 'blackman':
            return jnp.blackman(n)
        elif self.window == 'bartlett':
            return jnp.bartlett(n)
        elif self.window == 'none':
            return jnp.ones(n)
        else:
            raise ValueError(f"Unknown window type: {self.window}")

    def _detrend_data(self, y: jnp.ndarray) -> jnp.ndarray:
        """Remove linear trend from data.

        Parameters
        ----------
        y : jnp.ndarray
            Input data

        Returns
        -------
        jnp.ndarray
            Detrended data
        """
        # Fit linear trend: y = a*x + b
        n = len(y)
        x = jnp.arange(n)

        # Linear regression
        x_mean = jnp.mean(x)
        y_mean = jnp.mean(y)

        slope = jnp.sum((x - x_mean) * (y - y_mean)) / jnp.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean

        # Remove trend
        trend = slope * x + intercept
        return y - trend

    def _transform(self, data: RheoData) -> RheoData:
        """Apply FFT transform to time-domain data.

        Parameters
        ----------
        data : RheoData
            Input time-domain data

        Returns
        -------
        RheoData
            Frequency-domain data with FFT spectrum

        Raises
        ------
        ValueError
            If data is already in frequency domain
        """
        from rheo.core.data import RheoData

        # Validate domain
        if data.domain == 'frequency':
            raise ValueError("FFT analysis requires time-domain data")

        # Get time and signal data
        t = data.x
        y = data.y

        # Convert to JAX arrays for computation
        if not isinstance(t, jnp.ndarray):
            t = jnp.array(t)
        if not isinstance(y, jnp.ndarray):
            y = jnp.array(y)

        # Handle complex data by taking real part
        if jnp.iscomplexobj(y):
            y = jnp.real(y)

        # Detrend if requested
        if self.detrend:
            y = self._detrend_data(y)

        # Apply window function
        window = self._get_window(len(y))
        y_windowed = y * window

        # Compute FFT
        # Use rfft for real signals (more efficient)
        fft_result = jnp.fft.rfft(y_windowed)

        # Compute frequencies
        n = len(t)
        dt = (t[-1] - t[0]) / (n - 1)  # Average sampling interval
        freqs = jnp.fft.rfftfreq(n, d=dt)

        # Compute magnitude or PSD
        if self.return_psd:
            # Power spectral density
            spectrum = jnp.abs(fft_result) ** 2 / (n * dt)
        else:
            # Magnitude spectrum
            spectrum = jnp.abs(fft_result)

        # Normalize if requested
        if self.normalize and not self.return_psd:
            spectrum = spectrum / jnp.max(spectrum)

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update({
            'transform': 'fft',
            'window': self.window,
            'detrended': self.detrend,
            'psd': self.return_psd,
            'original_domain': 'time',
            'n_points': len(t),
            'dt': float(dt),
            # Store complex coefficients for inverse transform
            'fft_complex': fft_result
        })

        # Create new RheoData in frequency domain
        return RheoData(
            x=freqs,
            y=spectrum,
            x_units='Hz' if data.x_units else None,
            y_units='PSD' if self.return_psd else 'magnitude',
            domain='frequency',
            metadata=new_metadata,
            validate=False
        )

    def _inverse_transform(self, data: RheoData) -> RheoData:
        """Apply inverse FFT to return to time domain.

        Parameters
        ----------
        data : RheoData
            Frequency-domain data

        Returns
        -------
        RheoData
            Time-domain data

        Raises
        ------
        ValueError
            If data is not in frequency domain or missing required metadata
        """
        from rheo.core.data import RheoData

        if data.domain != 'frequency':
            raise ValueError("Inverse FFT requires frequency-domain data")

        if 'transform' not in data.metadata or data.metadata['transform'] != 'fft':
            raise ValueError("Data was not created by FFT transform")

        # Get original parameters
        n_points = data.metadata.get('n_points')
        dt = data.metadata.get('dt')
        fft_complex = data.metadata.get('fft_complex')

        if n_points is None or dt is None:
            raise ValueError("Missing metadata for inverse FFT (n_points, dt)")

        if fft_complex is None:
            raise ValueError("Missing complex FFT coefficients for inverse transform")

        # Use the stored complex coefficients for accurate reconstruction
        # Apply inverse FFT
        y_reconstructed = jnp.fft.irfft(fft_complex, n=n_points)

        # Reconstruct time array
        t = jnp.arange(n_points) * dt

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update({
            'transform': 'ifft',
            'original_domain': 'frequency'
        })

        return RheoData(
            x=t,
            y=y_reconstructed,
            x_units='s' if data.x_units else None,
            y_units='reconstructed',
            domain='time',
            metadata=new_metadata,
            validate=False
        )

    def find_peaks(
        self,
        freq_data: RheoData,
        prominence: float = 0.1,
        n_peaks: int = 5
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Find characteristic frequency peaks in FFT spectrum.

        Parameters
        ----------
        freq_data : RheoData
            Frequency-domain data from FFT
        prominence : float, default=0.1
            Minimum prominence for peak detection (relative to max)
        n_peaks : int, default=5
            Maximum number of peaks to return

        Returns
        -------
        peak_freqs : jnp.ndarray
            Frequencies of detected peaks
        peak_heights : jnp.ndarray
            Heights of detected peaks
        """
        freqs = freq_data.x
        spectrum = freq_data.y

        # Convert to numpy for scipy peak detection
        if isinstance(spectrum, jnp.ndarray):
            spectrum = np.array(spectrum)
        if isinstance(freqs, jnp.ndarray):
            freqs = np.array(freqs)

        # Simple peak detection: find local maxima
        from scipy.signal import find_peaks as scipy_find_peaks

        # Normalize spectrum for prominence calculation
        spectrum_norm = spectrum / np.max(spectrum)

        # Find peaks
        peak_indices, properties = scipy_find_peaks(
            spectrum_norm,
            prominence=prominence
        )

        # Sort by prominence and take top n_peaks
        if len(peak_indices) > n_peaks:
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-n_peaks:]
            peak_indices = peak_indices[top_indices]

        peak_freqs = freqs[peak_indices]
        peak_heights = spectrum[peak_indices]

        # Convert back to JAX
        return jnp.array(peak_freqs), jnp.array(peak_heights)

    def get_characteristic_time(self, freq_data: RheoData) -> float:
        """Extract characteristic time from FFT peak frequency.

        Parameters
        ----------
        freq_data : RheoData
            Frequency-domain data from FFT

        Returns
        -------
        float
            Characteristic time (1 / peak_frequency)
        """
        peak_freqs, peak_heights = self.find_peaks(freq_data, n_peaks=1)

        if len(peak_freqs) == 0:
            # No peak found, return NaN
            return float('nan')

        # Characteristic time is inverse of dominant frequency
        return 1.0 / float(peak_freqs[0])


__all__ = ['FFTAnalysis']
