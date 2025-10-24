"""HDF5 writer for rheological data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

from rheo.core.data import RheoData


def save_hdf5(
    data: RheoData,
    filepath: str,
    compression: bool = True,
    compression_level: int = 4,
    **kwargs
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
    except ImportError:
        raise ImportError("h5py is required for HDF5 writing. Install with: pip install h5py")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine compression settings
    compression_opts = None
    if compression:
        compression = 'gzip'
        compression_opts = compression_level

    # Write to HDF5
    with h5py.File(filepath, 'w') as f:
        # Store x and y data
        f.create_dataset(
            'x',
            data=np.array(data.x),
            compression=compression if compression else None,
            compression_opts=compression_opts
        )

        f.create_dataset(
            'y',
            data=np.array(data.y),
            compression=compression if compression else None,
            compression_opts=compression_opts
        )

        # Store units as attributes
        if data.x_units is not None:
            f['x'].attrs['units'] = data.x_units
        if data.y_units is not None:
            f['y'].attrs['units'] = data.y_units

        # Store domain
        f.attrs['domain'] = data.domain

        # Store metadata
        if data.metadata:
            metadata_group = f.create_group('metadata')
            _write_metadata_recursive(metadata_group, data.metadata)

        # Store rheo version
        try:
            import rheo
            f.attrs['rheo_version'] = rheo.__version__
        except:
            pass


def _write_metadata_recursive(group: Any, metadata: Dict[str, Any]) -> None:
    """Recursively write metadata to HDF5 group.

    Args:
        group: HDF5 group
        metadata: Metadata dictionary
    """
    for key, value in metadata.items():
        if isinstance(value, dict):
            subgroup = group.create_group(key)
            _write_metadata_recursive(subgroup, value)
        elif isinstance(value, (list, tuple)):
            # Convert to numpy array
            group.attrs[key] = np.array(value)
        elif isinstance(value, (str, int, float, bool, np.ndarray)):
            group.attrs[key] = value
        else:
            # Try to convert to string
            try:
                group.attrs[key] = str(value)
            except:
                pass  # Skip values that can't be serialized


def load_hdf5(filepath: str) -> RheoData:
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
    except ImportError:
        raise ImportError("h5py is required for HDF5 reading. Install with: pip install h5py")

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with h5py.File(filepath, 'r') as f:
        # Load data
        x = f['x'][:]
        y = f['y'][:]

        # Load units
        x_units = f['x'].attrs.get('units', None)
        y_units = f['y'].attrs.get('units', None)

        # Load domain
        domain = f.attrs.get('domain', 'time')

        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata = _read_metadata_recursive(f['metadata'])

        return RheoData(
            x=x,
            y=y,
            x_units=x_units,
            y_units=y_units,
            domain=domain,
            metadata=metadata,
            validate=True
        )


def _read_metadata_recursive(group: Any) -> Dict[str, Any]:
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
