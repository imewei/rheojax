"""
Data Service
===========

Service for loading, validating, and managing rheological data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.io import auto_load
from rheojax.io.readers.csv_reader import detect_csv_delimiter
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
        logger.debug("Initializing DataService")
        self._supported_formats = [
            ".csv",
            ".txt",
            ".xlsx",
            ".xls",
            ".dat",
            ".tri",
            ".rdf",
        ]
        logger.debug(
            "DataService initialized",
            supported_formats=self._supported_formats,
        )

    def load_file(
        self,
        file_path: str | Path,
        x_col: str | None = None,
        y_col: str | None = None,
        y2_col: str | None = None,
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
        y2_col : str, optional
            Secondary Y-axis column name (e.g., G'' for oscillatory data)
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
        logger.debug(
            "load_file called",
            file_path=str(file_path),
            x_col=x_col,
            y_col=y_col,
            y2_col=y2_col,
            test_mode=test_mode,
        )
        file_path = Path(file_path)
        logger.info("Loading data", filepath=str(file_path))

        if not file_path.exists():
            logger.error("File not found", filepath=str(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix.lower() not in self._supported_formats:
            logger.error(
                "Unsupported file format",
                filepath=str(file_path),
                suffix=file_path.suffix,
                supported_formats=self._supported_formats,
            )
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                f"Supported formats: {', '.join(self._supported_formats)}"
            )

        try:
            # Use auto_load to detect format and load
            if x_col and y_col:
                kwargs["x_col"] = x_col
                kwargs["y_col"] = y_col
            if y2_col:
                kwargs["y2_col"] = y2_col

            logger.debug("Calling auto_load", filepath=str(file_path), kwargs=kwargs)
            data = auto_load(str(file_path), **kwargs)

            # Ensure we have RheoData
            if not isinstance(data, RheoData):
                # Convert to RheoData if needed
                if hasattr(data, "x") and hasattr(data, "y"):
                    logger.debug("Converting loaded data to RheoData")
                    data = RheoData(x=data.x, y=data.y)
                else:
                    logger.error(
                        "Failed to load data as RheoData",
                        filepath=str(file_path),
                        data_type=type(data).__name__,
                    )
                    raise ValueError("Failed to load data as RheoData")

            # Set test mode if provided
            if test_mode:
                data.metadata["test_mode"] = test_mode
                logger.debug("Test mode set", test_mode=test_mode)

            # Light unit conversions for common CSV/Excel cases
            data = self._convert_units(data)

            n_records = len(data.x)
            logger.info(
                "Data loaded",
                filepath=str(file_path),
                n_records=n_records,
            )
            logger.debug(
                "load_file completed",
                filepath=str(file_path),
                n_records=n_records,
                x_range=(float(np.min(data.x)), float(np.max(data.x))),
                y_range=(float(np.min(data.y)), float(np.max(data.y))),
            )
            return data

        except FileNotFoundError:
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(
                "Failed to load file",
                filepath=str(file_path),
                error=str(e),
                exc_info=True,
            )
            raise ValueError(f"Failed to load file: {e}") from e

    def preview_file(
        self, file_path: str | Path, max_rows: int = 100
    ) -> dict[str, Any]:
        """Return a lightweight preview of a data file.

        The preview keeps parsing simple to avoid the heavier auto_load path and
        trims rows to keep UI rendering responsive.
        """
        logger.debug(
            "preview_file called",
            file_path=str(file_path),
            max_rows=max_rows,
        )

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error("File not found for preview", filepath=str(file_path))
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        data: list[list[Any]]
        headers: list[str]
        metadata: dict[str, Any] = {}

        try:
            if suffix in {".csv", ".txt", ".dat"}:
                logger.debug("Previewing CSV/text file", filepath=str(file_path))
                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=max_rows)
                headers = [str(col) for col in df.columns]
                data = df.fillna("").values.tolist()
                metadata["rows"] = len(df.index)
                metadata["columns"] = len(df.columns)
            elif suffix in {".xlsx", ".xls"}:
                logger.debug("Previewing Excel file", filepath=str(file_path))
                df = pd.read_excel(file_path, nrows=max_rows)
                headers = [str(col) for col in df.columns]
                data = df.fillna("").values.tolist()
                metadata["rows"] = len(df.index)
                metadata["columns"] = len(df.columns)
            else:
                # Fallback: use auto_load and build a simple table from RheoData
                logger.debug(
                    "Previewing via auto_load fallback",
                    filepath=str(file_path),
                )
                data_obj = auto_load(str(file_path))
                if not isinstance(data_obj, RheoData):
                    logger.error(
                        "Unsupported preview format",
                        filepath=str(file_path),
                        data_type=type(data_obj).__name__,
                    )
                    raise ValueError("Unsupported preview format")

                headers = ["x", "y"]
                rows = min(max_rows, len(data_obj.x))
                if hasattr(data_obj, "y2") and data_obj.y2 is not None:
                    headers.append("y2")
                    data = [
                        [data_obj.x[i], data_obj.y[i], data_obj.y2[i]]
                        for i in range(rows)
                    ]
                else:
                    data = [[data_obj.x[i], data_obj.y[i]] for i in range(rows)]

                metadata.update({"rows": len(data_obj.x), "columns": len(headers)})

            logger.debug(
                "preview_file completed",
                filepath=str(file_path),
                n_headers=len(headers),
                n_rows=len(data),
            )
            return {"data": data, "headers": headers, "metadata": metadata}

        except Exception as e:
            logger.error(
                "Failed to preview file",
                filepath=str(file_path),
                error=str(e),
                exc_info=True,
            )
            raise

    def _convert_units(self, data: RheoData) -> RheoData:
        """Convert common units to canonical values.

        - Frequency: Hz -> rad/s
        - Time: ms/min -> s
        - Modulus: kPa/MPa -> Pa
        """
        logger.debug(
            "_convert_units called",
            x_units=data.x_units,
            y_units=data.y_units,
        )

        x_units = (data.x_units or "").lower()
        y_units = (data.y_units or "").lower()

        x = np.asarray(data.x)
        y = np.asarray(data.y)

        x_converted = False
        y_converted = False
        original_x_units = x_units
        original_y_units = y_units

        if x_units == "hz":
            x = x * 2 * np.pi
            x_units = "rad/s"
            x_converted = True
        elif x_units == "ms":
            x = x / 1000.0
            x_units = "s"
            x_converted = True
        elif x_units in ("min", "mins", "minutes"):
            x = x * 60.0
            x_units = "s"
            x_converted = True

        if y_units == "kpa":
            y = y * 1e3
            y_units = "pa"
            y_converted = True
        elif y_units == "mpa":
            y = y * 1e6
            y_units = "pa"
            y_converted = True

        if x_converted:
            logger.debug(
                "X-axis units converted",
                from_units=original_x_units,
                to_units=x_units,
            )
        if y_converted:
            logger.debug(
                "Y-axis units converted",
                from_units=original_y_units,
                to_units=y_units,
            )

        logger.debug("_convert_units completed")
        return RheoData(
            x=x,
            y=y,
            x_units=x_units or None,
            y_units=y_units or None,
            domain=data.domain,
            metadata=data.metadata,
            initial_test_mode=data.metadata.get("test_mode"),
            validate=False,
        )

    def detect_test_mode(self, data: RheoData) -> str:
        """Auto-detect test mode from data characteristics.

        Analyzes column names, data ranges, and characteristics to determine
        the most likely test mode.

        Detection priority:
        1. Explicit metadata
        2. Domain indicator (frequency -> oscillation)
        3. Complex data (oscillation)
        4. Monotonic trends (relaxation/creep) - checked BEFORE flow
        5. Power-law relationship (flow)
        6. Log-spaced x-axis (oscillation frequency sweep)

        Parameters
        ----------
        data : RheoData
            Rheological data

        Returns
        -------
        str
            Test mode ('relaxation', 'creep', 'oscillation', 'flow', 'unknown')
        """
        logger.debug("detect_test_mode called", n_points=len(data.x))

        # Check metadata first
        if data.metadata and "test_mode" in data.metadata:
            mode = data.metadata["test_mode"]
            logger.debug("Test mode found in metadata", test_mode=mode)
            return mode

        # Check domain
        if data.domain == "frequency":
            logger.debug("Detected oscillation from domain=frequency")
            return "oscillation"

        # Analyze data characteristics
        x = np.asarray(data.x)
        y = np.asarray(data.y)

        # Check for complex data (oscillation)
        if np.iscomplexobj(y):
            logger.debug("Detected oscillation from complex y-data")
            return "oscillation"

        # Get basic statistics
        x_min, x_max = np.min(x), np.max(x)

        # Check y-axis trends FIRST (before flow detection)
        # This prevents exponential decay from being misclassified as flow
        y_diff = np.diff(y)

        # Relaxation: decreasing modulus over time (monotonically decreasing)
        # Typical: G(t) = G0 * exp(-t/tau)
        if len(y) > 3 and np.all(y_diff <= 0):
            if x_min >= 0:  # Time-like x-axis (non-negative)
                # Additional check: y values should span significant range
                y_range_ratio = np.max(y) / (np.min(y) + 1e-10)
                if y_range_ratio > 2:  # At least 2x decay
                    logger.debug(
                        "Detected relaxation: monotonic decrease",
                        y_range_ratio=float(y_range_ratio),
                    )
                    return "relaxation"

        # Creep: increasing compliance/strain over time (monotonically increasing)
        # Typical: J(t) = J0 * (1 - exp(-t/tau))
        if len(y) > 3 and np.all(y_diff >= 0):
            if x_min >= 0:  # Time-like x-axis (non-negative)
                # Additional check: y values should show significant growth
                y_range_ratio = np.max(y) / (np.min(y) + 1e-10)
                if y_range_ratio > 1.5:  # At least 50% growth
                    logger.debug(
                        "Detected creep: monotonic increase",
                        y_range_ratio=float(y_range_ratio),
                    )
                    return "creep"

        # Flow: shear rate vs viscosity/stress (power-law relationship)
        # Only check if data is NOT monotonic (already handled above)
        if x_min > 0 and len(x) > 5:
            # Check for power-law relationship in log-log space
            try:
                # Filter positive values for log
                mask = (x > 0) & (y > 0)
                if np.sum(mask) > 5:
                    log_x = np.log10(x[mask])
                    log_y = np.log10(y[mask])
                    correlation = np.corrcoef(log_x, log_y)[0, 1]
                    # Only classify as flow if there's variation in y (not monotonic)
                    if abs(correlation) > 0.9 and not (
                        np.all(y_diff >= 0) or np.all(y_diff <= 0)
                    ):
                        logger.debug(
                            "Detected flow: power-law correlation",
                            correlation=float(correlation),
                        )
                        return "flow"
            except Exception as exc:
                logger.debug(
                    "Flow detection failed on log-log correlation",
                    error=str(exc),
                )

        # Oscillation: frequency sweep with log-spaced x-axis
        if x_min > 0.001 and x_max < 10000:
            if len(x) > 10:
                # Check if logarithmically spaced (typical for frequency sweeps)
                try:
                    log_x = np.log10(x[x > 0])
                    if len(log_x) > 5:
                        log_spacing = np.diff(log_x)
                        if np.std(log_spacing) < 0.3:  # Uniform in log space
                            logger.debug("Detected oscillation: log-spaced x-axis")
                            return "oscillation"
                except Exception as exc:
                    logger.debug("Oscillation detection failed", error=str(exc))

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
        logger.debug("validate_data called", n_points=len(data.x))
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
            warnings.append(
                f"Insufficient data points: {len(x)} (recommend at least 5)"
            )

        # Check for duplicates
        if len(np.unique(x)) < len(x):
            warnings.append("X-axis contains duplicate values")

        # Check for zero values (can cause log issues)
        if np.any(x == 0):
            warnings.append(
                "X-axis contains zero values (may cause issues with log plots)"
            )
        if np.any(y == 0):
            warnings.append(
                "Y-axis contains zero values (may cause issues with log plots)"
            )

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

        logger.debug(
            "validate_data completed",
            n_warnings=len(warnings),
            warnings=warnings,
        )
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
            Dictionary with 'x_suggestions', 'y_suggestions', and 'y2_suggestions' lists
        """
        logger.debug("get_column_suggestions called", filepath=str(file_path))
        import re

        import pandas as pd

        suggestions = {"x_suggestions": [], "y_suggestions": [], "y2_suggestions": []}

        def normalize_col(col: str) -> str:
            """Normalize column name for pattern matching."""
            # Convert to lowercase, remove underscores/spaces, handle primes
            normalized = col.lower()
            normalized = normalized.replace("_", "").replace(" ", "").replace("-", "")
            # Handle prime notation: G' -> gprime, G'' -> gdoubleprime
            normalized = normalized.replace("''", "doubleprime").replace("'", "prime")
            return normalized

        def matches_pattern(
            col: str, patterns: list[str], exclude_patterns: list[str] | None = None
        ) -> bool:
            """Check if column matches any pattern (with word boundaries for short patterns)."""
            col_norm = normalize_col(col)
            col_lower = col.lower()

            # Check exclusions first
            if exclude_patterns:
                for excl in exclude_patterns:
                    if excl in col_norm:
                        return False

            for pattern in patterns:
                pattern_norm = pattern.lower().replace("_", "").replace(" ", "")
                # For very short patterns (1-2 chars), use word boundary matching
                if len(pattern) <= 2:
                    # Match at word boundaries or start/end
                    if re.search(
                        rf"(^|[^a-z]){re.escape(pattern_norm)}($|[^a-z])", col_lower
                    ):
                        return True
                else:
                    # For longer patterns, substring match is fine
                    if pattern_norm in col_norm:
                        return True
            return False

        try:
            # Try to read first few rows
            if file_path.suffix.lower() in [".csv", ".txt", ".dat"]:
                delimiter = detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, sep=delimiter, nrows=5)
            elif file_path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path, nrows=5)
            else:
                logger.debug(
                    "get_column_suggestions: unsupported format",
                    filepath=str(file_path),
                )
                return suggestions

            columns = df.columns.tolist()
            logger.debug(
                "Read column headers",
                filepath=str(file_path),
                columns=columns,
            )

            # X-axis patterns (frequency, time, rate)
            # Exclude temperature which contains 't'
            x_patterns = [
                "time",
                "freq",
                "frequency",
                "omega",
                "angular",
                "shear_rate",
                "shearrate",
                "rate",
                "strain",
                "rad/s",
                "hz",
            ]
            x_exclude = ["temp", "temperature"]

            # Y-axis patterns (storage modulus, stress, viscosity)
            # Exclude loss modulus patterns
            y_patterns = [
                "stress",
                "modulus",
                "gprime",
                "g_prime",
                "g'",
                "gp",
                "storage",
                "viscosity",
                "eta",
                "compliance",
                "j",
            ]
            y_exclude = ["doubleprime", "loss", "g''", "gpp", "gdouble"]

            # Y2-axis patterns (loss modulus G'')
            y2_patterns = [
                "gdoubleprime",
                "g_double_prime",
                "g''",
                "gpp",
                "gdouble",
                "loss",
                "lossy",
            ]

            # Match columns to patterns
            for col in columns:
                # Check X patterns (exclude temperature)
                if matches_pattern(col, x_patterns, x_exclude):
                    if col not in suggestions["x_suggestions"]:
                        suggestions["x_suggestions"].append(col)
                    continue  # Don't check Y patterns if matched X

                # Check Y2 patterns first (more specific than Y)
                if matches_pattern(col, y2_patterns):
                    if col not in suggestions["y2_suggestions"]:
                        suggestions["y2_suggestions"].append(col)
                    continue

                # Check Y patterns (exclude loss modulus)
                if matches_pattern(col, y_patterns, y_exclude):
                    if col not in suggestions["y_suggestions"]:
                        suggestions["y_suggestions"].append(col)

            logger.debug(
                "get_column_suggestions completed",
                filepath=str(file_path),
                suggestions=suggestions,
            )

        except Exception as e:
            logger.warning(
                "Failed to read column headers",
                filepath=str(file_path),
                error=str(e),
            )

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
        logger.debug(
            "convert_units called",
            source_x_units=data.x_units,
            source_y_units=data.y_units,
            target_x_units=x_units,
            target_y_units=y_units,
        )
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
                logger.info(
                    "Converted x-axis units",
                    from_units=data.x_units,
                    to_units=x_units,
                )

        # Convert y-axis
        if y_units and data.y_units and y_units != data.y_units:
            if data.y_units in pressure_conversions and y_units in pressure_conversions:
                factor = (
                    pressure_conversions[data.y_units] / pressure_conversions[y_units]
                )
                y = y * factor
                logger.info(
                    "Converted y-axis units",
                    from_units=data.y_units,
                    to_units=y_units,
                )

        logger.debug("convert_units completed")
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
        logger.debug("get_supported_formats called")
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
        logger.debug(
            "preprocess_data called",
            n_points=len(data.x),
            remove_outliers=remove_outliers,
            smooth=smooth,
            outlier_threshold=outlier_threshold,
        )
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
                    logger.info("Removed outliers", n_removed=removed)

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
                logger.info(
                    "Applied Savitzky-Golay smoothing",
                    window_length=window_length,
                    polyorder=polyorder,
                )

        logger.debug(
            "preprocess_data completed",
            n_points_out=len(x),
        )
        return RheoData(
            x=x,
            y=y,
            x_units=data.x_units,
            y_units=data.y_units,
            domain=data.domain,
            metadata=data.metadata.copy(),
        )
