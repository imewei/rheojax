"""RheoData class - JAX-native rheological data container.

This module provides the RheoData abstraction for rheological data that supports
both NumPy and JAX arrays with additional features for rheological analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing helper only
    import jax.numpy as jnp_typing
else:
    jnp_typing = np

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
HAS_JAX = True

# Module-level logger
logger = get_logger(__name__)


type ArrayLike = np.ndarray | jnp_typing.ndarray | list | tuple


def _coerce_ndarray(data: ArrayLike | jnp_typing.ndarray | None) -> np.ndarray:
    """Convert any array-like input to a NumPy array for scalar ops."""
    if data is None:
        logger.error("Array data is None during conversion", exc_info=True)
        raise ValueError("Array data must be initialized before conversion")
    if isinstance(data, np.ndarray):
        return data
    if HAS_JAX and isinstance(data, jnp.ndarray):
        logger.debug(
            "Converting JAX array to NumPy",
            from_type="jax.ndarray",
            to_type="np.ndarray",
        )
        return np.asarray(data)
    logger.debug(
        "Converting array-like to NumPy",
        from_type=type(data).__name__,
        to_type="np.ndarray",
    )
    return np.asarray(data)


@dataclass
class RheoData:
    """JAX-native container for rheological data with NumPy/JAX array support.

    This class provides a unified interface for rheological data that supports
    both NumPy and JAX arrays with additional features needed for rheological
    analysis including automatic test mode detection, data validation, and
    domain-specific operations.

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
    # Optional explicit test mode passed during initialization (e.g., relaxation/creep/oscillation)
    initial_test_mode: InitVar[str | None] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    validate: bool = True
    _explicit_test_mode: str | None = field(default=None, repr=False, init=False)

    def __post_init__(self, initial_test_mode: str | None):
        """Initialize and validate RheoData."""
        logger.debug(
            "Creating RheoData",
            domain=self.domain,
            test_mode=initial_test_mode,
            validate=self.validate,
        )

        # Normalize metadata container
        if self.metadata is None:
            self.metadata = {}

        # Persist explicitly provided test mode into metadata and internal cache
        if initial_test_mode is not None:
            self._explicit_test_mode = initial_test_mode
            self.metadata.setdefault("test_mode", initial_test_mode)
            self.metadata.setdefault("detected_test_mode", initial_test_mode)
        elif self.metadata and "test_mode" in self.metadata:
            self._explicit_test_mode = self.metadata.get("test_mode")

        if self.x is None or self.y is None:
            logger.error("x and y data must be provided", exc_info=True)
            raise ValueError("x and y data must be provided")

        # Convert to arrays
        self.x = self._ensure_array(self.x)
        self.y = self._ensure_array(self.y)

        x_array = _coerce_ndarray(self.x)
        y_array = _coerce_ndarray(self.y)

        # Log creation details after array conversion
        logger.debug(
            "RheoData arrays created",
            x_shape=x_array.shape,
            y_shape=y_array.shape,
            x_dtype=str(x_array.dtype),
            y_dtype=str(y_array.dtype),
        )

        # Validate shapes
        if x_array.shape != y_array.shape:
            logger.error(
                "Shape mismatch between x and y data",
                x_shape=x_array.shape,
                y_shape=y_array.shape,
                exc_info=True,
            )
            raise ValueError(
                f"x and y must have the same shape. Got x: {x_array.shape}, y: {y_array.shape}"
            )

        # Validate data if requested
        if self.validate:
            self._validate_data()

    def _ensure_array(self, data: ArrayLike) -> np.ndarray:
        """Ensure data is a proper array."""
        if isinstance(data, (np.ndarray, jnp.ndarray)):
            return data
        elif isinstance(data, (list, tuple)):
            logger.debug(
                "Converting list/tuple to array", from_type=type(data).__name__
            )
            return np.array(data)
        else:
            logger.debug("Converting to array", from_type=type(data).__name__)
            return np.array(data)

    def _validate_data(self):
        """Validate data for common issues."""
        logger.debug(
            "Validating data",
            checks=["nan", "finite", "monotonic", "negative_frequency"],
        )

        # Check for NaN values first (NaN is also non-finite)
        if isinstance(self.x, np.ndarray):
            if np.any(np.isnan(self.x)):
                logger.error("x data contains NaN values", exc_info=True)
                raise ValueError("x data contains NaN values")
            if not np.all(np.isfinite(self.x)):
                logger.error("x data contains non-finite values", exc_info=True)
                raise ValueError("x data contains non-finite values")
        elif isinstance(self.x, jnp.ndarray):
            if jnp.any(jnp.isnan(self.x)):
                logger.error("x data contains NaN values", exc_info=True)
                raise ValueError("x data contains NaN values")
            if not jnp.all(jnp.isfinite(self.x)):
                logger.error("x data contains non-finite values", exc_info=True)
                raise ValueError("x data contains non-finite values")

        if isinstance(self.y, np.ndarray):
            if np.any(np.isnan(self.y)):
                logger.error("y data contains NaN values", exc_info=True)
                raise ValueError("y data contains NaN values")
            if not np.all(np.isfinite(self.y)):
                logger.error("y data contains non-finite values", exc_info=True)
                raise ValueError("y data contains non-finite values")
        elif isinstance(self.y, jnp.ndarray):
            if jnp.any(jnp.isnan(self.y)):
                logger.error("y data contains NaN values", exc_info=True)
                raise ValueError("y data contains NaN values")
            if not jnp.all(jnp.isfinite(self.y)):
                logger.error("y data contains non-finite values", exc_info=True)
                raise ValueError("y data contains non-finite values")

        # Check for monotonic x-axis
        if len(self.x) > 1:
            if isinstance(self.x, np.ndarray):
                diffs = np.diff(self.x)
                if not (np.all(diffs > 0) or np.all(diffs < 0)):
                    logger.debug("x-axis is not monotonic")
                    warnings.warn("x-axis is not monotonic", UserWarning, stacklevel=2)
            elif isinstance(self.x, jnp.ndarray):
                diffs = jnp.diff(self.x)
                if not (jnp.all(diffs > 0) or jnp.all(diffs < 0)):
                    logger.debug("x-axis is not monotonic")
                    warnings.warn("x-axis is not monotonic", UserWarning, stacklevel=2)

        # Check for negative values in frequency domain
        if self.domain == "frequency":
            if isinstance(self.y, np.ndarray):
                if np.any(np.real(self.y) < 0):
                    logger.debug("y data contains negative values in frequency domain")
                    warnings.warn(
                        "y data contains negative values in frequency domain",
                        UserWarning,
                        stacklevel=2,
                    )
            elif isinstance(self.y, jnp.ndarray):
                if jnp.any(jnp.real(self.y) < 0):
                    logger.debug("y data contains negative values in frequency domain")
                    warnings.warn(
                        "y data contains negative values in frequency domain",
                        UserWarning,
                        stacklevel=2,
                    )

        logger.debug("Data validation completed successfully")

    def to_jax(self) -> RheoData:
        """Convert arrays to JAX arrays.

        Returns:
            New RheoData with JAX arrays
        """
        logger.debug(
            "Converting RheoData to JAX arrays", from_type="numpy", to_type="jax"
        )
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

        Uses np.asarray() for zero-copy conversion when possible, providing
        10-30% memory savings for large arrays (>100k points).

        Returns:
            New RheoData with NumPy arrays
        """
        logger.debug(
            "Converting RheoData to NumPy arrays", from_type="jax", to_type="numpy"
        )
        # Use asarray for zero-copy when array is already NumPy-compatible
        # Preserve dtype (handles both float64 and complex128)
        x_np = np.asarray(self.x)
        y_np = np.asarray(self.y)

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
        logger.debug("Creating copy of RheoData")
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
        logger.debug("Updating metadata", keys=list(metadata.keys()))
        self.metadata.update(metadata)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with data and metadata
        """
        logger.debug("Converting RheoData to dictionary")
        x_data = self.x.tolist() if hasattr(self.x, "tolist") else list(self.x)
        y_data = self.y.tolist() if hasattr(self.y, "tolist") else list(self.y)

        data_dict = {
            "x": x_data,
            "y": y_data,
            "x_units": self.x_units,
            "y_units": self.y_units,
            "domain": self.domain,
            "metadata": self.metadata,
        }

        if self._explicit_test_mode is not None:
            data_dict["test_mode"] = self._explicit_test_mode

        return data_dict

    @classmethod
    def from_dict(cls, data_dict: dict[str, Any]) -> RheoData:
        """Create from dictionary representation.

        Args:
            data_dict: Dictionary with data and metadata

        Returns:
            RheoData instance
        """
        logger.debug("Creating RheoData from dictionary")
        metadata = data_dict.get("metadata", {}) or {}
        test_mode = data_dict.get("test_mode")

        return cls(
            x=np.array(data_dict["x"]),
            y=np.array(data_dict["y"]),
            x_units=data_dict.get("x_units"),
            y_units=data_dict.get("y_units"),
            domain=data_dict.get("domain", "time"),
            metadata=dict(metadata),
            initial_test_mode=test_mode,
            validate=False,
        )

    # NumPy-like interface
    @property
    def shape(self) -> tuple:
        """Shape of the y data."""
        return _coerce_ndarray(self.y).shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of y data."""
        return _coerce_ndarray(self.y).ndim

    @property
    def size(self) -> int:
        """Size of y data."""
        return int(_coerce_ndarray(self.y).size)

    @property
    def dtype(self):
        """Data type of y data."""
        return _coerce_ndarray(self.y).dtype

    @property
    def is_complex(self) -> bool:
        """Check if y data is complex."""
        return np.iscomplexobj(_coerce_ndarray(self.y))

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
    def y_real(self) -> np.ndarray:
        """Get real component of y data.

        For complex modulus data (G* = G' + i·G''), this returns the storage
        modulus (G'). For real data, returns y unchanged.

        Returns:
            Real component of y data (G' for complex modulus)

        Example:
            >>> data = read_trios('frequency_sweep.txt')  # Returns complex G*
            >>> G_prime = data[0].y_real  # Storage modulus (G')
            >>> plt.loglog(data[0].x, G_prime, label="G'")
        """
        if self.is_complex:
            if isinstance(self.y, jnp.ndarray):
                return jnp.real(self.y)
            return np.real(self.y)
        return self.y

    @property
    def y_imag(self) -> np.ndarray:
        """Get imaginary component of y data.

        For complex modulus data (G* = G' + i·G''), this returns the loss
        modulus (G''). For real data, returns zeros.

        Returns:
            Imaginary component of y data (G'' for complex modulus)

        Example:
            >>> data = read_trios('frequency_sweep.txt')  # Returns complex G*
            >>> G_double_prime = data[0].y_imag  # Loss modulus (G'')
            >>> plt.loglog(data[0].x, G_double_prime, label='G"')
        """
        if self.is_complex:
            if isinstance(self.y, jnp.ndarray):
                return jnp.imag(self.y)
            return np.imag(self.y)
        if isinstance(self.y, jnp.ndarray):
            return jnp.zeros_like(self.y)
        return np.zeros_like(self.y)

    @property
    def storage_modulus(self) -> np.ndarray | None:
        """Get storage modulus (G') from complex modulus data.

        Alias for y_real that makes rheological intent explicit.

        Returns:
            Storage modulus (G') if data is complex, None otherwise

        Example:
            >>> data = read_trios('frequency_sweep.txt')
            >>> G_prime = data[0].storage_modulus
        """
        if self.is_complex:
            return self.y_real
        return None

    @property
    def loss_modulus(self) -> np.ndarray | None:
        """Get loss modulus (G'') from complex modulus data.

        Alias for y_imag that makes rheological intent explicit.

        Returns:
            Loss modulus (G'') if data is complex, None otherwise

        Example:
            >>> data = read_trios('frequency_sweep.txt')
            >>> G_double_prime = data[0].loss_modulus
        """
        if self.is_complex:
            return self.y_imag
        return None

    @property
    def tan_delta(self) -> np.ndarray | None:
        """Get loss tangent (tan δ = G''/G') from complex modulus data.

        The loss tangent quantifies the ratio of viscous to elastic response:
        - tan δ < 1: Elastic-dominant (solid-like)
        - tan δ > 1: Viscous-dominant (liquid-like)
        - tan δ = 1: Equal elastic and viscous contributions

        Returns:
            Loss tangent (dimensionless) if data is complex, None otherwise

        Example:
            >>> data = read_trios('frequency_sweep.txt')
            >>> tan_d = data[0].tan_delta
            >>> print(f"Material type: {'solid-like' if tan_d.mean() < 1 else 'liquid-like'}")
        """
        if self.is_complex:
            G_prime = self.y_real
            G_double_prime = self.y_imag
            # Avoid division by zero
            if isinstance(G_prime, jnp.ndarray):
                return jnp.where(G_prime > 0, G_double_prime / G_prime, jnp.nan)
            return np.where(G_prime > 0, G_double_prime / G_prime, np.nan)
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
        # Prefer explicitly provided test mode
        if self._explicit_test_mode is not None:
            return self._explicit_test_mode

        # Check if already set in metadata (explicit or previously detected)
        if "test_mode" in self.metadata:
            return self.metadata["test_mode"]
        if "detected_test_mode" in self.metadata:
            return self.metadata["detected_test_mode"]

        # Lazy import to avoid circular dependency
        from rheojax.core.test_modes import detect_test_mode

        # Detect test mode
        logger.debug("Detecting test mode from data characteristics")
        mode = detect_test_mode(self)

        # Cache the result
        self.metadata["detected_test_mode"] = mode
        self._explicit_test_mode = mode

        logger.debug("Test mode detected", test_mode=mode)

        return mode

    def __getitem__(self, idx):
        """Support indexing and slicing."""
        if isinstance(idx, (int, np.integer)):
            return (self.x[idx], self.y[idx])
        else:
            logger.debug("Slicing RheoData", index_type=type(idx).__name__)
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
                logger.error("x-axes must match for addition", exc_info=True)
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
                logger.error("x-axes must match for subtraction", exc_info=True)
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
                logger.error("x-axes must match for multiplication", exc_info=True)
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
        logger.debug(
            "Interpolating data",
            n_new_points=len(new_x) if hasattr(new_x, "__len__") else 1,
        )
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
        logger.debug("Resampling data", n_points=n_points, domain=self.domain)
        x_array = _coerce_ndarray(self.x)

        if self.domain == "frequency":
            # Log-spaced for frequency domain
            new_x = np.logspace(
                np.log10(x_array.min()), np.log10(x_array.max()), n_points
            )
        else:
            # Linear-spaced for time domain
            new_x = np.linspace(x_array.min(), x_array.max(), n_points)

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

        logger.debug("Smoothing data", window_size=window_size)

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
        logger.debug("Computing numerical derivative")
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
        logger.debug("Computing numerical integral")
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
            logger.debug("Data is already in frequency domain")
            warnings.warn(
                "Data is already in frequency domain", UserWarning, stacklevel=2
            )
            return self.copy()

        logger.error("Frequency domain conversion not yet implemented", exc_info=True)
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
            logger.debug("Data is already in time domain")
            warnings.warn("Data is already in time domain", UserWarning, stacklevel=2)
            return self.copy()

        logger.error("Time domain conversion not yet implemented", exc_info=True)
        # This would use inverse FFT transform when implemented
        raise NotImplementedError(
            "Time domain conversion will be implemented with transforms"
        )

    # Data slicing methods
    def slice(self, start: float | None = None, end: float | None = None) -> RheoData:
        """Slice data between x values.

        Args:
            start: Start x value
            end: End x value

        Returns:
            Sliced RheoData
        """
        logger.debug("Slicing data by x range", start=start, end=end)
        x_array = _coerce_ndarray(self.x)
        y_array = _coerce_ndarray(self.y)

        mask = np.ones_like(x_array, dtype=bool)

        if start is not None:
            mask &= x_array >= start
        if end is not None:
            mask &= x_array <= end

        sliced_x = x_array[mask]
        sliced_y = y_array[mask]

        logger.debug("Sliced data", original_size=len(x_array), new_size=len(sliced_x))

        return RheoData(
            x=sliced_x,
            y=sliced_y,
            x_units=self.x_units,
            y_units=self.y_units,
            domain=self.domain,
            metadata=self.metadata.copy(),
            validate=False,
        )
