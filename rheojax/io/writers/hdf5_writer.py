"""HDF5 writer for rheological data."""

from __future__ import annotations

import enum
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)

# Types that HDF5 can natively store as attributes
_HDF5_SCALAR_TYPES = (str, int, float, bool, np.integer, np.floating, np.bool_)

# Sentinel for None values in HDF5 attributes
_NONE_SENTINEL = "__rheojax_None__"


def save_hdf5(
    data: RheoData,
    filepath: str | Path,
    compression: bool = True,
    compression_level: int = 4,
    **kwargs,
) -> None:
    """Save RheoData to HDF5 file.

    HDF5 is the recommended format for archiving rheological data. It provides:
    - Efficient storage with compression
    - Preservation of all metadata
    - Fast read/write performance
    - Cross-platform compatibility

    Args:
        data: RheoData object to save
        filepath: Output file path
        compression: Enable gzip compression (default: True)
        compression_level: Compression level 0-9 (default: 4)
        **kwargs: Additional arguments passed to h5py

    Raises:
        ImportError: If h5py not installed
        ValueError: If data is invalid
        IOError: If file cannot be written
    """
    try:
        import h5py
    except ImportError as exc:
        logger.error(
            "h5py import failed",
            error_type="ImportError",
            suggestion="pip install h5py",
            exc_info=True,
        )
        raise ImportError(
            "h5py is required for HDF5 writing. Install with: pip install h5py"
        ) from exc

    if not (0 <= compression_level <= 9):
        raise ValueError(f"compression_level must be 0-9, got {compression_level}")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine compression settings
    compression_algorithm: str | None = None
    compression_opts = None
    if compression:
        compression_algorithm = "gzip"
        compression_opts = compression_level
        logger.debug(
            "Compression settings configured",
            algorithm=compression_algorithm,
            compression_level=compression_opts,
        )

    with log_io(logger, "write", filepath=str(filepath)) as ctx:
        # Atomic write: write to a temp file in the same directory, then rename.
        # This prevents corrupt files from interrupted writes.
        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(dir=filepath.parent, suffix=".h5.tmp")
            os.close(tmp_fd)
            tmp_fd = None

            with h5py.File(tmp_path, "w") as f:
                # Store x and y data with explicit float64 preservation
                x_arr = np.asarray(data.x, dtype=np.float64)
                y_arr = np.asarray(data.y)
                # Preserve complex dtype; ensure real arrays are float64
                if not np.issubdtype(y_arr.dtype, np.complexfloating):
                    y_arr = np.asarray(y_arr, dtype=np.float64)

                logger.debug(
                    "Writing data arrays",
                    x_shape=x_arr.shape,
                    x_dtype=str(x_arr.dtype),
                    y_shape=y_arr.shape,
                    y_dtype=str(y_arr.dtype),
                    compression=compression_algorithm,
                )
                f.create_dataset(
                    "x",
                    data=x_arr,
                    compression=compression_algorithm,
                    compression_opts=compression_opts,
                )

                f.create_dataset(
                    "y",
                    data=y_arr,
                    compression=compression_algorithm,
                    compression_opts=compression_opts,
                )

                # Store units as attributes
                if data.x_units is not None:
                    f["x"].attrs["units"] = data.x_units
                if data.y_units is not None:
                    f["y"].attrs["units"] = data.y_units
                logger.debug(
                    "Units stored",
                    x_units=data.x_units,
                    y_units=data.y_units,
                )

                # Store domain
                f.attrs["domain"] = data.domain
                logger.debug("Domain stored", domain=data.domain)

                # Store test_mode and deformation_mode as top-level attrs
                # (belt-and-suspenders: also in metadata dict)
                test_mode = data.test_mode
                if test_mode:
                    f.attrs["test_mode"] = str(test_mode)
                deformation_mode = data.deformation_mode
                if deformation_mode:
                    f.attrs["deformation_mode"] = str(deformation_mode)

                # Store metadata
                if data.metadata:
                    metadata_group = f.create_group("metadata")
                    dropped = _write_metadata_recursive(metadata_group, data.metadata)
                    logger.debug(
                        "Metadata written",
                        metadata_keys=list(data.metadata.keys()),
                    )
                    if dropped:
                        logger.warning(
                            "Some metadata keys could not be serialized "
                            "and were dropped from the HDF5 file",
                            dropped_keys=dropped,
                        )

                # Store rheojax version
                try:
                    import rheojax

                    f.attrs["rheojax_version"] = rheojax.__version__
                    logger.debug("Version stored", rheojax_version=rheojax.__version__)
                except ImportError:
                    pass

            # Atomic rename: only overwrites target after successful write
            os.replace(tmp_path, filepath)
            tmp_path = None  # Prevent cleanup since rename succeeded

        finally:
            # Clean up temp file if rename didn't happen (write failed)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        ctx["data_points"] = len(data.x)
        ctx["compression"] = compression
        ctx["has_metadata"] = bool(data.metadata)


def _write_metadata_recursive(
    group: Any,
    metadata: dict[str, Any],
    _path: str = "",
) -> list[str]:
    """Recursively write metadata to HDF5 group.

    Args:
        group: HDF5 group
        metadata: Metadata dictionary
        _path: Internal path prefix for logging (do not set externally)

    Returns:
        List of metadata key paths that could not be serialized.
    """
    dropped_keys: list[str] = []

    for key, value in metadata.items():
        full_key = f"{_path}/{key}" if _path else key

        if value is None:
            # None is not HDF5-storable; store as sentinel string
            group.attrs[key] = _NONE_SENTINEL
            continue

        # Convert enum values to their underlying Python type
        # (h5py can't serialize str-enum subclasses directly)
        if isinstance(value, enum.Enum):
            value = value.value

        if isinstance(value, dict):
            subgroup = group.create_group(key)
            dropped_keys.extend(
                _write_metadata_recursive(subgroup, value, _path=full_key)
            )
            continue

        if isinstance(value, (list, tuple)):
            try:
                if value and all(isinstance(v, str) for v in value):
                    import h5py

                    group.attrs.create(key, value, dtype=h5py.string_dtype())
                else:
                    group.attrs[key] = np.array(value)
            except (TypeError, ValueError):
                # Lists of mixed types — fall back to string
                group.attrs[key] = str(value)
                logger.debug(
                    "Metadata stored as string (mixed-type list)",
                    key=full_key,
                )
            continue

        if isinstance(value, np.ndarray):
            # HDF5 attributes have a 64 KB size limit; store large arrays
            # as datasets within the metadata group instead.
            if value.nbytes > 60_000:
                group.create_dataset(key, data=value)
            else:
                group.attrs[key] = value
            continue

        if isinstance(value, _HDF5_SCALAR_TYPES):
            group.attrs[key] = value
            continue

        # Last resort: stringify
        try:
            group.attrs[key] = str(value)
            logger.debug(
                "Metadata stringified for storage",
                key=full_key,
                original_type=type(value).__name__,
            )
        except (TypeError, ValueError, OSError):
            dropped_keys.append(full_key)
            logger.warning(
                "Could not serialize metadata key — value dropped",
                key=full_key,
                value_type=type(value).__name__,
            )

    return dropped_keys


def save_fit_result_hdf5(
    result: Any,
    filepath: str | Path,
    compression: bool = True,
    compression_level: int = 4,
) -> None:
    """Save a FitResult to HDF5 file.

    Stores model parameters, statistics, fitted curve, and metadata
    in a structured HDF5 layout.

    Args:
        result: A FitResult instance (from rheojax.core.fit_result).
        filepath: Output file path.
        compression: Enable gzip compression (default: True).
        compression_level: Compression level 0-9 (default: 4).

    Raises:
        ImportError: If h5py not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 writing. Install with: pip install h5py"
        ) from exc

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    comp_algo: str | None = "gzip" if compression else None
    comp_opts = compression_level if compression else None

    with h5py.File(filepath, "w") as f:
        f.attrs["rheojax_type"] = "FitResult"
        f.attrs["model_name"] = result.model_name or ""
        f.attrs["model_class_name"] = result.model_class_name or ""
        f.attrs["protocol"] = result.protocol or ""
        f.attrs["n_params"] = result.n_params

        # Store scalar statistics
        for attr_name in ("r_squared", "aic", "bic", "aicc", "rmse", "mae"):
            val = getattr(result, attr_name, None)
            if val is not None and np.isfinite(val):
                f.attrs[attr_name] = float(val)

        # Parameters
        params_grp = f.create_group("params")
        for name, value in result.params.items():
            params_grp.attrs[name] = float(value)

        # Parameter units
        if result.params_units:
            units_grp = f.create_group("params_units")
            for name, unit in result.params_units.items():
                units_grp.attrs[name] = str(unit)

        # Fitted curve
        if result.fitted_curve is not None:
            arr = np.asarray(result.fitted_curve)
            f.create_dataset(
                "fitted_curve", data=arr,
                compression=comp_algo, compression_opts=comp_opts,
            )

        # Input data
        if result.X is not None:
            f.create_dataset(
                "input_x", data=np.asarray(result.X),
                compression=comp_algo, compression_opts=comp_opts,
            )
        if result.y is not None:
            f.create_dataset(
                "input_y", data=np.asarray(result.y),
                compression=comp_algo, compression_opts=comp_opts,
            )

        # Timestamp
        if result.timestamp:
            f.attrs["timestamp"] = result.timestamp

    logger.info(
        "Saved FitResult to HDF5",
        filepath=str(filepath),
        model_name=result.model_name,
    )


def load_hdf5(filepath: str | Path) -> RheoData:
    """Load RheoData from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        RheoData object

    Raises:
        ImportError: If h5py not installed
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        import h5py
    except ImportError as exc:
        logger.error(
            "h5py import failed",
            error_type="ImportError",
            suggestion="pip install h5py",
            exc_info=True,
        )
        raise ImportError(
            "h5py is required for HDF5 reading. Install with: pip install h5py"
        ) from exc

    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(
            "File not found",
            filepath=str(filepath),
            error_type="FileNotFoundError",
        )
        raise FileNotFoundError(f"File not found: {filepath}")

    with log_io(logger, "read", filepath=str(filepath)) as ctx:
        with h5py.File(filepath, "r") as f:
            # Load data
            x = f["x"][:]
            y = f["y"][:]
            logger.debug(
                "Data arrays loaded",
                x_shape=x.shape,
                y_shape=y.shape,
            )

            # Load units
            # R6-HDF5-001: h5py may return bytes instead of str on some
            # platforms/backends. Decode to str for downstream compatibility.
            x_units = f["x"].attrs.get("units", None)
            if isinstance(x_units, bytes):
                x_units = x_units.decode("utf-8")
            y_units = f["y"].attrs.get("units", None)
            if isinstance(y_units, bytes):
                y_units = y_units.decode("utf-8")
            logger.debug(
                "Units loaded",
                x_units=x_units,
                y_units=y_units,
            )

            # Load domain
            # R6-HDF5-002: Decode bytes for top-level string attrs.
            domain = f.attrs.get("domain", "time")
            if isinstance(domain, bytes):
                domain = domain.decode("utf-8")
            logger.debug("Domain loaded", domain=domain)

            # Load metadata
            metadata = {}
            if "metadata" in f:
                metadata = _read_metadata_recursive(f["metadata"])
                logger.debug(
                    "Metadata loaded",
                    metadata_keys=list(metadata.keys()),
                )

            # Restore test_mode/deformation_mode from top-level attrs
            # into metadata (belt-and-suspenders with metadata dict)
            # R6-HDF5-003: Decode bytes for top-level string attrs.
            test_mode = f.attrs.get("test_mode", None)
            if isinstance(test_mode, bytes):
                test_mode = test_mode.decode("utf-8")
            if test_mode and "test_mode" not in metadata:
                metadata["test_mode"] = test_mode
            deformation_mode = f.attrs.get("deformation_mode", None)
            if isinstance(deformation_mode, bytes):
                deformation_mode = deformation_mode.decode("utf-8")
            if deformation_mode and "deformation_mode" not in metadata:
                metadata["deformation_mode"] = deformation_mode

            ctx["data_points"] = len(x)
            ctx["has_metadata"] = bool(metadata)
            ctx["domain"] = domain

            return RheoData(
                x=x,
                y=y,
                x_units=x_units,
                y_units=y_units,
                domain=domain,
                initial_test_mode=metadata.get("test_mode"),
                metadata=metadata,
                validate=True,
            )


def _read_metadata_recursive(group: Any) -> dict[str, Any]:
    """Recursively read metadata from HDF5 group.

    Args:
        group: HDF5 group

    Returns:
        Metadata dictionary
    """
    metadata: dict[str, Any] = {}

    # Read attributes
    for key, value in group.attrs.items():
        # h5py may return bytes instead of str on some platforms
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        # R8-IO-003: decode numpy arrays of bytes from h5py string_dtype()
        elif hasattr(value, "dtype") and hasattr(value, "tolist"):
            try:
                items = value.tolist()
                if not isinstance(items, list):
                    # 0-d array: tolist() returns a scalar — unwrap to Python native type
                    if isinstance(items, bytes):
                        value = items.decode("utf-8")
                    elif isinstance(items, str):
                        value = items
                    else:
                        # int, float, bool, None — unwrap from 0-d numpy array
                        value = items
                elif items and isinstance(items[0], bytes):
                    value = [
                        v.decode("utf-8") if isinstance(v, bytes) else str(v)
                        for v in items
                    ]
                elif items and isinstance(items[0], str):
                    value = items  # already decoded, convert from numpy to list
            except (AttributeError, UnicodeDecodeError):
                pass
        # Restore None values from sentinel (backward-compatible with old "__None__")
        if isinstance(value, str) and value in (_NONE_SENTINEL, "__None__"):
            metadata[key] = None
        else:
            metadata[key] = value

    # VIS-HDF-001: Move import to top of function (not inside for loop).
    # Python caches modules so repeated imports are cheap, but placing import
    # inside a loop is misleading and signals incomplete refactoring.
    import h5py

    # Read subgroups and datasets.
    # HDF5-READ-001: attrs are loaded first; skip any dataset/subgroup whose
    # name collides with an already-loaded attribute.  Without this guard the
    # dataset loop would silently overwrite the attribute value, corrupting
    # metadata that was intentionally stored as a scalar attribute (e.g.
    # test_mode, deformation_mode stored as belt-and-suspenders duplicates).
    for key in group.keys():
        if key in metadata:
            # Attribute with the same name already loaded — skip the
            # dataset/subgroup to preserve the attribute's value.
            logger.debug(
                "Skipping HDF5 dataset/subgroup — name collides with "
                "previously loaded attribute; attribute value is kept",
                key=key,
            )
            continue
        if isinstance(group[key], h5py.Group):
            metadata[key] = _read_metadata_recursive(group[key])
        else:
            metadata[key] = group[key][:]

    return metadata
