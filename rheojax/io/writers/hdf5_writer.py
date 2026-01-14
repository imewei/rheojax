"""HDF5 writer for rheological data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)


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
        # Write to HDF5
        with h5py.File(filepath, "w") as f:
            # Store x and y data
            logger.debug(
                "Writing data arrays",
                x_shape=data.x.shape,
                y_shape=data.y.shape,
                compression=compression_algorithm,
            )
            f.create_dataset(
                "x",
                data=np.array(data.x),
                compression=compression_algorithm,
                compression_opts=compression_opts,
            )

            f.create_dataset(
                "y",
                data=np.array(data.y),
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

            # Store metadata
            if data.metadata:
                metadata_group = f.create_group("metadata")
                _write_metadata_recursive(metadata_group, data.metadata)
                logger.debug(
                    "Metadata written",
                    metadata_keys=list(data.metadata.keys()),
                )

            # Store rheojax version
            try:
                import rheojax

                f.attrs["rheojax_version"] = rheojax.__version__
                logger.debug("Version stored", rheojax_version=rheojax.__version__)
            except ImportError:
                pass

        ctx["data_points"] = len(data.x)
        ctx["compression"] = compression
        ctx["has_metadata"] = bool(data.metadata)


def _write_metadata_recursive(group: Any, metadata: dict[str, Any]) -> None:
    """Recursively write metadata to HDF5 group.

    Args:
        group: HDF5 group
        metadata: Metadata dictionary
    """
    for key, value in metadata.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_metadata_recursive(subgroup, value)
            continue

        if isinstance(value, (list, tuple)):
            group.attrs[key] = np.array(value)
            continue

        if isinstance(value, np.ndarray):
            group.attrs[key] = value
            continue

        if isinstance(value, (str, int, float, bool)):
            group.attrs[key] = value
            continue

        try:
            group.attrs[key] = str(value)
        except Exception:
            pass  # Skip values that can't be serialized


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
            x_units = f["x"].attrs.get("units", None)
            y_units = f["y"].attrs.get("units", None)
            logger.debug(
                "Units loaded",
                x_units=x_units,
                y_units=y_units,
            )

            # Load domain
            domain = f.attrs.get("domain", "time")
            logger.debug("Domain loaded", domain=domain)

            # Load metadata
            metadata = {}
            if "metadata" in f:
                metadata = _read_metadata_recursive(f["metadata"])
                logger.debug(
                    "Metadata loaded",
                    metadata_keys=list(metadata.keys()),
                )

            ctx["data_points"] = len(x)
            ctx["has_metadata"] = bool(metadata)
            ctx["domain"] = domain

            return RheoData(
                x=x,
                y=y,
                x_units=x_units,
                y_units=y_units,
                domain=domain,
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
    metadata = {}

    # Read attributes
    for key, value in group.attrs.items():
        metadata[key] = value

    # Read subgroups
    for key in group.keys():
        if isinstance(group[key], type(group)):  # It's a group
            metadata[key] = _read_metadata_recursive(group[key])
        else:  # It's a dataset
            metadata[key] = group[key][:]

    return metadata
