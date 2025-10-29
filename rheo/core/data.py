"""RheoData class - piblin Measurement wrapper with JAX support.

This module provides the RheoData abstraction that wraps piblin.Measurement
while adding JAX array support and additional rheological data handling features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Union

import jax.numpy as jnp
import numpy as np

try:
    import piblin

    HAS_PIBLIN = True
except ImportError:
    HAS_PIBLIN = False
    warnings.warn(
        "piblin is not installed. Some features may be limited.", ImportWarning, stacklevel=2
    )


ArrayLike = Union[np.ndarray, jnp.ndarray, list, tuple]


@dataclass
class RheoData:
    """Wrapper around piblin.Measurement with JAX support and rheological features.

    This class provides a unified interface for rheological data that maintains
    full compatibility with piblin.Measurement while adding support for JAX arrays
    and additional features needed for rheological analysis.

    Attributes:
        x: Independent variable data (e.g., time, frequency)
        y: Dependent variable data (e.g., stress, strain, modulus)
        x_units: Units for x-axis data
        y_units: Units for y-axis data
        domain: Data domain ('time' or 'frequency')
        metadata: Dictionary of additional metadata
        validate: Whether to validate data on creation
    """

    x: ArrayLike | None = None
    y: ArrayLike | None = None
    x_units: str | None = None
    y_units: str | None = None
    domain: str = "time"
    metadata: dict[str, Any] = field(default_factory=dict)
    validate: bool = True
    _measurement: Any | None = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize and validate RheoData."""
        if self.x is None or self.y is None:
            if self._measurement is None:
                raise ValueError("x and y data must be provided")
            else:
                # Extract from piblin measurement
                self.x = np.array(self._measurement.x)
                self.y = np.array(self._measurement.y)
                if hasattr(self._measurement, "metadata"):
                    self.metadata.update(self._measurement.metadata)

        # Convert to arrays
        self.x = self._ensure_array(self.x)
        self.y = self._ensure_array(self.y)

        # Validate shapes
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have the same shape. Got x: {self.x.shape}, y: {self.y.shape}"
            )

        # Validate data if requested
        if self.validate:
            self._validate_data()

        # Create piblin measurement if not provided
        if self._measurement is None and HAS_PIBLIN:
            self._create_piblin_measurement()

    def _ensure_array(self, data: ArrayLike) -> np.ndarray | jnp.ndarray:
        """Ensure data is a proper array."""
        if isinstance(data, (np.ndarray, jnp.ndarray)):
            return data
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            return np.array(data)

    def _validate_data(self):
        """Validate data for common issues."""
        # Check for NaN values first (NaN is also non-finite)
        if isinstance(self.x, np.ndarray):
            if np.any(np.isnan(self.x)):
                raise ValueError("x data contains NaN values")
            if not np.all(np.isfinite(self.x)):
                raise ValueError("x data contains non-finite values")
        elif isinstance(self.x, jnp.ndarray):
            if jnp.any(jnp.isnan(self.x)):
                raise ValueError("x data contains NaN values")
            if not jnp.all(jnp.isfinite(self.x)):
                raise ValueError("x data contains non-finite values")

        if isinstance(self.y, np.ndarray):
            if np.any(np.isnan(self.y)):
                raise ValueError("y data contains NaN values")
            if not np.all(np.isfinite(self.y)):
                raise ValueError("y data contains non-finite values")
        elif isinstance(self.y, jnp.ndarray):
            if jnp.any(jnp.isnan(self.y)):
                raise ValueError("y data contains NaN values")
            if not jnp.all(jnp.isfinite(self.y)):
                raise ValueError("y data contains non-finite values")

        # Check for monotonic x-axis
        if len(self.x) > 1:
            if isinstance(self.x, np.ndarray):
                diffs = np.diff(self.x)
                if not (np.all(diffs > 0) or np.all(diffs < 0)):
                    warnings.warn("x-axis is not monotonic", UserWarning, stacklevel=2)
            elif isinstance(self.x, jnp.ndarray):
                diffs = jnp.diff(self.x)
                if not (jnp.all(diffs > 0) or jnp.all(diffs < 0)):
                    warnings.warn("x-axis is not monotonic", UserWarning, stacklevel=2)

        # Check for negative values in frequency domain
        if self.domain == "frequency":
            if isinstance(self.y, np.ndarray):
                if np.any(np.real(self.y) < 0):
                    warnings.warn(
                        "y data contains negative values in frequency domain",
                        UserWarning, stacklevel=2,
                    )
            elif isinstance(self.y, jnp.ndarray):
                if jnp.any(jnp.real(self.y) < 0):
                    warnings.warn(
                        "y data contains negative values in frequency domain",
                        UserWarning, stacklevel=2,
                    )

    def _create_piblin_measurement(self):
        """Create internal piblin Measurement."""
        if HAS_PIBLIN:
            # Convert to numpy for piblin
            x_np = np.array(self.x) if isinstance(self.x, jnp.ndarray) else self.x
            y_np = np.array(self.y) if isinstance(self.y, jnp.ndarray) else self.y

            self._measurement = piblin.Measurement(
                x=x_np,
                y=y_np,
                x_units=self.x_units,
                y_units=self.y_units,
                metadata=self.metadata,
            )

    @classmethod
    def from_piblin(cls, measurement: Any) -> RheoData:
        """Create RheoData from piblin Measurement.

        Args:
            measurement: piblin.Measurement object

        Returns:
            RheoData instance wrapping the measurement
        """
        return cls(_measurement=measurement)

    def to_piblin(self) -> Any:
        """Convert to piblin Measurement.

        Returns:
            piblin.Measurement object
        """
        if self._measurement is not None:
            return self._measurement

        if not HAS_PIBLIN:
            raise ImportError("piblin is required for this operation")

        # Convert to numpy for piblin
        x_np = np.array(self.x) if isinstance(self.x, jnp.ndarray) else self.x
        y_np = np.array(self.y) if isinstance(self.y, jnp.ndarray) else self.y

        return piblin.Measurement(
            x=x_np,
            y=y_np,
            x_units=self.x_units,
            y_units=self.y_units,
            metadata=self.metadata,
        )

    def to_jax(self) -> RheoData:
        """Convert arrays to JAX arrays.

        Returns:
            New RheoData with JAX arrays
        """
        return RheoData(
            x=jnp.array(self.x),
            y=jnp.array(self.y),
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def to_numpy(self) -> RheoData:
        """Convert arrays to NumPy arrays.

        Returns:
            New RheoData with NumPy arrays
        """
        x_np = np.array(self.x) if isinstance(self.x, jnp.ndarray) else self.x
        y_np = np.array(self.y) if isinstance(self.y, jnp.ndarray) else self.y

        return RheoData(
            x=x_np,
            y=y_np,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def copy(self) -> RheoData:
        """Create a copy of the RheoData.

        Returns:
            Copy of the RheoData instance
        """
        return RheoData(
            x=self.x.copy() if hasattr(self.x, "copy") else self.x,
            y=self.y.copy() if hasattr(self.y, "copy") else self.y,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def update_metadata(self, metadata: dict[str, Any]):
        """Update metadata dictionary.

        Args:
            metadata: Dictionary of metadata to add/update
        """
        self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with data and metadata
        """
        x_data = self.x.tolist() if hasattr(self.x, "tolist") else list(self.x)
        y_data = self.y.tolist() if hasattr(self.y, "tolist") else list(self.y)

        return {
            "x": x_data,
            "y": y_data,
            "x_units": self.x_units,
            "y_units": self.y_units,
            "domain": self.domain,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> RheoData:
        """Create from dictionary representation.

        Args:
            data_dict: Dictionary with data and metadata

        Returns:
            RheoData instance
        """
        return cls(
            x=np.array(data_dict["x"]),
            y=np.array(data_dict["y"]),
            x_units=data_dict.get("x_units"),
            y_units=data_dict.get("y_units"),
            domain=data_dict.get("domain", "time"),
            metadata=data_dict.get("metadata", {}),
            validate=False,
        )

    # NumPy-like interface
    @property
    def shape(self) -> tuple:
        """Shape of the y data."""
        return self.y.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of y data."""
        return self.y.ndim

    @property
    def size(self) -> int:
        """Size of y data."""
        return self.y.size

    @property
    def dtype(self):
        """Data type of y data."""
        return self.y.dtype

    @property
    def is_complex(self) -> bool:
        """Check if y data is complex."""
        return np.iscomplexobj(self.y)

    @property
    def modulus(self) -> np.ndarray | None:
        """Get modulus of complex data."""
        if self.is_complex:
            return np.abs(self.y)
        return None

    @property
    def phase(self) -> np.ndarray | None:
        """Get phase of complex data."""
        if self.is_complex:
            return np.angle(self.y)
        return None

    @property
    def test_mode(self) -> str:
        """Automatically detect or retrieve test mode.

        The test mode is detected based on data characteristics and cached
        in metadata. If already detected, returns the cached value. If
        explicitly set in metadata['test_mode'], returns that value.

        Returns:
            Test mode string (relaxation, creep, oscillation, rotation, unknown)
        """
        # Check if already detected and cached
        if "detected_test_mode" in self.metadata:
            return self.metadata["detected_test_mode"]

        # Lazy import to avoid circular dependency
        from rheo.core.test_modes import detect_test_mode

        # Detect test mode
        mode = detect_test_mode(self)

        # Cache the result
        self.metadata["detected_test_mode"] = mode

        return mode

    def __getitem__(self, idx):
        """Support indexing and slicing."""
        if isinstance(idx, (int, np.integer)):
            return (self.x[idx], self.y[idx])
        else:
            return RheoData(
                x=self.x[idx],
                y=self.y[idx],
                x_units=self.x_units,
                y_units=self.y_units,
                domain=self.domain,
                metadata=self.metadata.copy(),
                validate=False,
            )

    def __add__(self, other):
        """Add two RheoData objects or scalar."""
        if isinstance(other, RheoData):
            if not np.array_equal(self.x, other.x):
                raise ValueError("x-axes must match for addition")
            return RheoData(
                x=self.x,
                y=self.y + other.y,
                x_units=self.x_units,
                y_units=self.y_units,
                domain=self.domain,
                metadata=self.metadata.copy(),
                validate=False,
            )
        else:
            return RheoData(
                x=self.x,
                y=self.y + other,
                x_units=self.x_units,
                y_units=self.y_units,
                domain=self.domain,
                metadata=self.metadata.copy(),
                validate=False,
            )

    def __sub__(self, other):
        """Subtract two RheoData objects or scalar."""
        if isinstance(other, RheoData):
            if not np.array_equal(self.x, other.x):
                raise ValueError("x-axes must match for subtraction")
            return RheoData(
                x=self.x,
                y=self.y - other.y,
                x_units=self.x_units,
                y_units=self.y_units,
                domain=self.domain,
                metadata=self.metadata.copy(),
                validate=False,
            )
        else:
            return RheoData(
                x=self.x,
                y=self.y - other,
                x_units=self.x_units,
                y_units=self.y_units,
                domain=self.domain,
                metadata=self.metadata.copy(),
                validate=False,
            )

    def __mul__(self, other):
        """Multiply by scalar or another RheoData."""
        if isinstance(other, RheoData):
            if not np.array_equal(self.x, other.x):
                raise ValueError("x-axes must match for multiplication")
            y_result = self.y * other.y
        else:
            y_result = self.y * other

        return RheoData(
            x=self.x,
            y=y_result,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    # Data operations
    def interpolate(self, new_x: ArrayLike) -> RheoData:
        """Interpolate data to new x values.

        Args:
            new_x: New x values for interpolation

        Returns:
            Interpolated RheoData
        """
        new_x = self._ensure_array(new_x)

        if isinstance(self.x, jnp.ndarray) or isinstance(self.y, jnp.ndarray):
            # Use JAX interpolation
            new_y = jnp.interp(new_x, self.x, self.y)
        else:
            # Use NumPy interpolation
            new_y = np.interp(new_x, self.x, self.y)

        return RheoData(
            x=new_x,
            y=new_y,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def resample(self, n_points: int) -> RheoData:
        """Resample data to specified number of points.

        Args:
            n_points: Number of points to resample to

        Returns:
            Resampled RheoData
        """
        if self.domain == "frequency":
            # Log-spaced for frequency domain
            new_x = np.logspace(
                np.log10(self.x.min()), np.log10(self.x.max()), n_points
            )
        else:
            # Linear-spaced for time domain
            new_x = np.linspace(self.x.min(), self.x.max(), n_points)

        return self.interpolate(new_x)

    def smooth(self, window_size: int = 5) -> RheoData:
        """Smooth data using moving average.

        Args:
            window_size: Size of smoothing window

        Returns:
            Smoothed RheoData
        """
        if window_size % 2 == 0:
            window_size += 1  # Make odd for symmetric window

        # Simple moving average
        kernel = np.ones(window_size) / window_size

        if isinstance(self.y, jnp.ndarray):
            # Use JAX convolution
            smoothed_y = jnp.convolve(self.y, kernel, mode="same")
        else:
            # Use NumPy convolution
            smoothed_y = np.convolve(self.y, kernel, mode="same")

        return RheoData(
            x=self.x,
            y=smoothed_y,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def derivative(self) -> RheoData:
        """Compute numerical derivative.

        Returns:
            RheoData with derivative values
        """
        if isinstance(self.x, jnp.ndarray) or isinstance(self.y, jnp.ndarray):
            # Use JAX gradient
            dy_dx = jnp.gradient(self.y, self.x)
        else:
            # Use NumPy gradient
            dy_dx = np.gradient(self.y, self.x)

        return RheoData(
            x=self.x,
            y=dy_dx,
            x_units=self.x_units,
            y_units=(
                f"d({self.y_units})/d({self.x_units})"
                if self.y_units and self.x_units
                else None
            ),
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    def integral(self) -> RheoData:
        """Compute numerical integral.

        Returns:
            RheoData with integrated values
        """
        if isinstance(self.x, jnp.ndarray) or isinstance(self.y, jnp.ndarray):
            # Use JAX cumulative trapezoid
            from jax.scipy.integrate import cumulative_trapezoid

            integrated = cumulative_trapezoid(self.y, self.x, initial=0)
        else:
            # Use NumPy/SciPy cumulative trapezoid
            from scipy.integrate import cumulative_trapezoid

            integrated = cumulative_trapezoid(self.y, self.x, initial=0)

        return RheoData(
            x=self.x,
            y=integrated,
            x_units=self.x_units,
            y_units=(
                f"∫{self.y_units}·d{self.x_units}"
                if self.y_units and self.x_units
                else None
            ),
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )

    # Domain conversion placeholders
    def to_frequency_domain(self) -> RheoData:
        """Convert time domain data to frequency domain.

        Returns:
            Frequency domain RheoData
        """
        if self.domain != "time":
            warnings.warn("Data is already in frequency domain", UserWarning, stacklevel=2)
            return self.copy()

        # This would use FFT transform when implemented
        raise NotImplementedError(
            "Frequency domain conversion will be implemented with transforms"
        )

    def to_time_domain(self) -> RheoData:
        """Convert frequency domain data to time domain.

        Returns:
            Time domain RheoData
        """
        if self.domain != "frequency":
            warnings.warn("Data is already in time domain", UserWarning, stacklevel=2)
            return self.copy()

        # This would use inverse FFT transform when implemented
        raise NotImplementedError(
            "Time domain conversion will be implemented with transforms"
        )

    # piblin compatibility methods
    def slice(
        self, start: float | None = None, end: float | None = None
    ) -> RheoData:
        """Slice data between x values (piblin compatibility).

        Args:
            start: Start x value
            end: End x value

        Returns:
            Sliced RheoData
        """
        mask = np.ones_like(self.x, dtype=bool)

        if start is not None:
            mask &= self.x >= start
        if end is not None:
            mask &= self.x <= end

        return self[mask]
