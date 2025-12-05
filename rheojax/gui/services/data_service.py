"""
Data Service
===========

Service for loading, validating, and managing rheological data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.io import auto_load

logger = logging.getLogger(__name__)


class DataService:
    """Service for rheological data operations.

    Features:
        - Multi-format data loading (TRIOS, CSV, Excel, Anton Paar)
        - Data validation and quality checks
        - Test mode auto-detection
        - Data preprocessing and filtering
        - Column mapping suggestions

    Example
    -------
    >>> service = DataService()
    >>> rheo_data = service.load_file('data.csv', x_col='time', y_col='stress')
    >>> test_mode = service.detect_test_mode(rheo_data)
    >>> warnings = service.validate_data(rheo_data)
    """

    def __init__(self) -> None:
        """Initialize data service."""
        self._supported_formats = [
            ".csv",
            ".txt",
            ".xlsx",
            ".xls",
            ".dat",
            ".tri",
            ".rdf",
        ]

    def load_file(
        self,
        file_path: str | Path,
        x_col: str | None = None,
        y_col: str | None = None,
        test_mode: str | None = None,
        **kwargs: Any,
    ) -> RheoData:
        """Load data from file using rheojax.io.auto_load.

        Parameters
        ----------
        file_path : str | Path
            Path to data file
        x_col : str, optional
            X-axis column name (for CSV/Excel files)
        y_col : str, optional
            Y-axis column name (for CSV/Excel files)
        test_mode : str, optional
            Test mode ('relaxation', 'creep', 'oscillation', 'flow')
        **kwargs
            Additional loader arguments

        Returns
        -------
        RheoData
            Loaded rheological data

        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If file format is unsupported or data is invalid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self._supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self._supported_formats)}"
            )

        try:
            # Use auto_load to detect format and load
            if x_col and y_col:
                kwargs["x_col"] = x_col
                kwargs["y_col"] = y_col

            data = auto_load(str(file_path), **kwargs)

            # Ensure we have RheoData
            if not isinstance(data, RheoData):
                # Convert to RheoData if needed
                if hasattr(data, "x") and hasattr(data, "y"):
                    data = RheoData(x=data.x, y=data.y)
                else:
                    raise ValueError("Failed to load data as RheoData")

            # Set test mode if provided
            if test_mode:
                data.metadata["test_mode"] = test_mode

            logger.info(f"Successfully loaded {len(data.x)} data points from {file_path}")
            return data

        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
            raise ValueError(f"Failed to load file: {e}") from e

    def detect_test_mode(self, data: RheoData) -> str:
        """Auto-detect test mode from data characteristics.

        Analyzes column names, data ranges, and characteristics to determine
        the most likely test mode.

        Parameters
        ----------
        data : RheoData
            Rheological data

        Returns
        -------
        str
            Test mode ('relaxation', 'creep', 'oscillation', 'flow', 'unknown')
        """
        # Check metadata first
        if data.metadata and "test_mode" in data.metadata:
            return data.metadata["test_mode"]

        # Check domain
        if data.domain == "frequency":
            return "oscillation"

        # Analyze data characteristics
        x = np.asarray(data.x)
        y = np.asarray(data.y)

        # Check for complex data (oscillation)
        if np.iscomplexobj(y):
            return "oscillation"

        # Check x-axis characteristics
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min

        # Very small values suggest frequency (rad/s or Hz)
        if x_min > 0.01 and x_max < 1000 and x_range > 1:
            # Could be frequency sweep
            if len(x) > 10:
                # Check if logarithmically spaced (typical for frequency sweeps)
                log_x = np.log10(x[x > 0])
                if len(log_x) > 5:
                    log_spacing = np.diff(log_x)
                    if np.std(log_spacing) < 0.5:  # Relatively uniform in log space
                        return "oscillation"

        # Check y-axis trends
        # Relaxation: decreasing modulus over time
        if np.all(np.diff(y) <= 0):  # Monotonically decreasing
            if x_min >= 0 and x_max > 1:  # Time-like
                return "relaxation"

        # Creep: increasing strain over time
        if np.all(np.diff(y) >= 0):  # Monotonically increasing
            if x_min >= 0 and x_max > 1:  # Time-like
                return "creep"

        # Flow: shear rate vs viscosity/stress
        # Typically shows power-law or plateau behavior
        if x_min > 0 and len(x) > 5:
            # Check for power-law relationship
            try:
                log_x = np.log10(x[x > 0])
                log_y = np.log10(y[y > 0])
                if len(log_x) == len(log_y) and len(log_x) > 5:
                    correlation = np.corrcoef(log_x, log_y)[0, 1]
                    if abs(correlation) > 0.9:  # Strong log-log correlation
                        return "flow"
            except Exception:
                pass

        logger.warning("Could not auto-detect test mode, defaulting to 'unknown'")
        return "unknown"

    def validate_data(self, data: RheoData) -> list[str]:
        """Validate data quality and characteristics.

        Parameters
        ----------
        data : RheoData
            Rheological data

        Returns
        -------
        list[str]
            List of validation warnings/errors
        """
        warnings = []

        x = np.asarray(data.x)
        y = np.asarray(data.y)

        # Check for NaN or Inf
        if np.any(~np.isfinite(x)):
            warnings.append("X-axis contains NaN or Inf values")
        if np.any(~np.isfinite(y)):
            warnings.append("Y-axis contains NaN or Inf values")

        # Check for negative values where inappropriate
        if np.any(x < 0):
            warnings.append("X-axis contains negative values (check if appropriate)")
        if np.any(y < 0):
            warnings.append("Y-axis contains negative values (check if appropriate)")

        # Check for sufficient data points
        if len(x) < 5:
            warnings.append(f"Insufficient data points: {len(x)} (recommend at least 5)")

        # Check for duplicates
        if len(np.unique(x)) < len(x):
            warnings.append("X-axis contains duplicate values")

        # Check for zero values (can cause log issues)
        if np.any(x == 0):
            warnings.append("X-axis contains zero values (may cause issues with log plots)")
        if np.any(y == 0):
            warnings.append("Y-axis contains zero values (may cause issues with log plots)")

        # Check data range
        x_range = np.ptp(x)
        y_range = np.ptp(y)

        if x_range == 0:
            warnings.append("X-axis has no variation (constant values)")
        if y_range == 0:
            warnings.append("Y-axis has no variation (constant values)")

        # Check for outliers (simple IQR method)
        if len(y) > 4:
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = np.sum((y < q1 - 3 * iqr) | (y > q3 + 3 * iqr))
                if outliers > 0:
                    warnings.append(f"Detected {outliers} potential outliers")

        # Check for monotonicity (sometimes required)
        if not np.all(np.diff(x) > 0):
            warnings.append("X-axis is not monotonically increasing")

        return warnings

    def get_column_suggestions(self, file_path: Path) -> dict[str, list[str]]:
        """Suggest column mappings based on file headers.

        Parameters
        ----------
        file_path : Path
            Path to data file

        Returns
        -------
        dict
            Dictionary with 'x_suggestions' and 'y_suggestions' lists
        """
        import pandas as pd

        suggestions = {"x_suggestions": [], "y_suggestions": []}

        try:
            # Try to read first few rows
            if file_path.suffix.lower() in [".csv", ".txt", ".dat"]:
                df = pd.read_csv(file_path, nrows=5)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, nrows=5)
            else:
                return suggestions

            columns = df.columns.tolist()

            # Common x-axis patterns
            x_patterns = [
                "time",
                "t",
                "freq",
                "frequency",
                "omega",
                "angular",
                "shear_rate",
                "rate",
                "strain",
            ]

            # Common y-axis patterns
            y_patterns = [
                "stress",
                "modulus",
                "g_prime",
                "g_double_prime",
                "g*",
                "viscosity",
                "eta",
                "compliance",
            ]

            # Match patterns (case-insensitive)
            for col in columns:
                col_lower = col.lower()
                for pattern in x_patterns:
                    if pattern in col_lower:
                        suggestions["x_suggestions"].append(col)
                        break

                for pattern in y_patterns:
                    if pattern in col_lower:
                        suggestions["y_suggestions"].append(col)
                        break

        except Exception as e:
            logger.warning(f"Failed to read column headers: {e}")

        return suggestions

    def convert_units(
        self, data: RheoData, x_units: str | None = None, y_units: str | None = None
    ) -> RheoData:
        """Convert data units.

        Parameters
        ----------
        data : RheoData
            Input data
        x_units : str, optional
            Target x-axis units
        y_units : str, optional
            Target y-axis units

        Returns
        -------
        RheoData
            Data with converted units

        Notes
        -----
        Currently supports basic conversions. More comprehensive unit
        conversion requires pint or similar library.
        """
        x = np.asarray(data.x)
        y = np.asarray(data.y)

        # Simple conversion factors (extend as needed)
        time_conversions = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0}

        pressure_conversions = {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "psi": 6894.76}

        # Convert x-axis
        if x_units and data.x_units and x_units != data.x_units:
            if data.x_units in time_conversions and x_units in time_conversions:
                factor = time_conversions[data.x_units] / time_conversions[x_units]
                x = x * factor
                logger.info(f"Converted x-axis from {data.x_units} to {x_units}")

        # Convert y-axis
        if y_units and data.y_units and y_units != data.y_units:
            if data.y_units in pressure_conversions and y_units in pressure_conversions:
                factor = pressure_conversions[data.y_units] / pressure_conversions[y_units]
                y = y * factor
                logger.info(f"Converted y-axis from {data.y_units} to {y_units}")

        # Create new RheoData with converted values
        return RheoData(
            x=x,
            y=y,
            x_units=x_units or data.x_units,
            y_units=y_units or data.y_units,
            domain=data.domain,
            metadata=data.metadata.copy(),
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file formats.

        Returns
        -------
        list[str]
            File extensions (e.g., ['.csv', '.xlsx', '.txt'])
        """
        return self._supported_formats.copy()

    def preprocess_data(
        self,
        data: RheoData,
        remove_outliers: bool = False,
        smooth: bool = False,
        outlier_threshold: float = 3.0,
        **kwargs: Any,
    ) -> RheoData:
        """Preprocess data with filtering and smoothing.

        Parameters
        ----------
        data : RheoData
            Input data
        remove_outliers : bool, default=False
            Remove statistical outliers using IQR method
        smooth : bool, default=False
            Apply smoothing filter
        outlier_threshold : float, default=3.0
            IQR multiplier for outlier detection
        **kwargs
            Additional preprocessing options

        Returns
        -------
        RheoData
            Preprocessed data
        """
        x = np.asarray(data.x).copy()
        y = np.asarray(data.y).copy()

        # Remove outliers
        if remove_outliers and len(y) > 4:
            q1, q3 = np.percentile(y, [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                mask = (y >= q1 - outlier_threshold * iqr) & (
                    y <= q3 + outlier_threshold * iqr
                )
                x = x[mask]
                y = y[mask]
                removed = np.sum(~mask)
                if removed > 0:
                    logger.info(f"Removed {removed} outliers")

        # Smooth data
        if smooth:
            from scipy.signal import savgol_filter

            window_length = kwargs.get("window_length", min(11, len(y) // 3))
            if window_length % 2 == 0:
                window_length += 1  # Must be odd
            window_length = max(3, window_length)  # Minimum 3

            polyorder = kwargs.get("polyorder", min(2, window_length - 1))

            if len(y) >= window_length:
                y = savgol_filter(y, window_length, polyorder)
                logger.info("Applied Savitzky-Golay smoothing")

        return RheoData(
            x=x,
            y=y,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata=data.metadata.copy(),
        )
