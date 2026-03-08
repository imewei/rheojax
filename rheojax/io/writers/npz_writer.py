"""NumPy .npz writer/reader for RheoData objects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.logging import get_logger

logger = get_logger(__name__)

__all__ = ["save_npz", "load_npz"]


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar and array types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.complexfloating):
            return {"real": float(obj.real), "imag": float(obj.imag)}
        return super().default(obj)


def _encode_str(s: str | None) -> np.ndarray:
    """Encode a string (or None) as a uint8 byte array for npz storage."""
    encoded = (s or "").encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


def _decode_str(arr: np.ndarray) -> str | None:
    """Decode a uint8 byte array back to a string (None if empty)."""
    s = arr.tobytes().decode("utf-8")
    return s if s else None


def save_npz(
    data: RheoData,
    filepath: str | Path,
    compressed: bool = True,
) -> None:
    """Save a RheoData object to a NumPy .npz archive.

    Strings and metadata are stored as UTF-8 encoded uint8 byte arrays —
    no pickle is used.

    Args:
        data: RheoData object to save.
        filepath: Destination path (np.savez appends .npz if not present).
        compressed: If True (default), use np.savez_compressed. If False,
            use np.savez (larger file, faster write).

    Raises:
        OSError: If the file cannot be written.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Serialise metadata to JSON bytes (handles numpy types via custom encoder)
    metadata_bytes = json.dumps(
        data.metadata or {}, cls=_NumpyEncoder, allow_nan=False
    ).encode("utf-8")

    arrays: dict[str, np.ndarray] = {
        "x": np.asarray(data.x),
        "y": np.asarray(data.y),
        "_metadata_json": np.frombuffer(metadata_bytes, dtype=np.uint8),
        "_x_units": _encode_str(data.x_units),
        "_y_units": _encode_str(data.y_units),
        "_domain": _encode_str(data.domain),
        "_initial_test_mode": _encode_str(data._explicit_test_mode),
    }

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(filepath, **arrays)

    logger.info(
        "Saved RheoData to npz",
        filepath=str(filepath),
        compressed=compressed,
        n_points=len(data.x),
    )


def load_npz(filepath: str | Path) -> RheoData:
    """Load a RheoData object from a NumPy .npz archive.

    Args:
        filepath: Path to the .npz file (with or without .npz extension).

    Returns:
        Reconstructed RheoData object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid RheoData npz archive.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        with_suffix = filepath.with_suffix(".npz")
        if with_suffix.exists():
            filepath = with_suffix
        else:
            raise FileNotFoundError(f"File not found: {filepath}")

    try:
        npz = np.load(filepath, allow_pickle=False)
    except Exception as e:
        raise ValueError(f"Failed to load npz archive: {filepath}: {e}") from e

    x = npz["x"]
    y = npz["y"]

    # Validate array shape compatibility before constructing RheoData
    if x.ndim >= 1 and y.ndim >= 1 and len(x) != len(y):
        raise ValueError(
            f"Corrupt npz archive: x has {len(x)} points but y has {len(y)}"
        )

    # Parse metadata from UTF-8 bytes
    try:
        metadata: dict = json.loads(npz["_metadata_json"].tobytes().decode("utf-8"))
    except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
        logger.warning("Could not parse metadata JSON from npz, using empty dict")
        metadata = {}

    x_units = _decode_str(npz["_x_units"]) if "_x_units" in npz else None
    y_units = _decode_str(npz["_y_units"]) if "_y_units" in npz else None
    domain = _decode_str(npz["_domain"]) if "_domain" in npz else None
    initial_test_mode = (
        _decode_str(npz["_initial_test_mode"]) if "_initial_test_mode" in npz else None
    )

    logger.info(
        "Loaded RheoData from npz",
        filepath=str(filepath),
        n_points=len(x),
        domain=domain,
        initial_test_mode=initial_test_mode,
    )

    return RheoData(
        x=x,
        y=y,
        x_units=x_units,
        y_units=y_units,
        domain=domain,
        initial_test_mode=initial_test_mode,
        metadata=metadata,
        validate=True,
    )
