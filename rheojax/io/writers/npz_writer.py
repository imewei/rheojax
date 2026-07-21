"""NumPy .npz writer/reader for RheoData objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.io.json_encoder import NumpyJSONEncoder as _NumpyEncoder
from rheojax.logging import get_logger

logger = get_logger(__name__)

__all__ = ["save_npz", "load_npz", "save_fit_result_npz"]


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
        data.metadata or {}, cls=_NumpyEncoder, allow_nan=True
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
    save_fn(filepath, **arrays)  # type: ignore[arg-type]

    logger.info(
        "Saved RheoData to npz",
        filepath=str(filepath),
        compressed=compressed,
        n_points=len(data.x),  # type: ignore[arg-type]
    )


def save_fit_result_npz(
    result: Any,
    filepath: str | Path,
    compressed: bool = True,
) -> None:
    """Save a FitResult to a NumPy .npz archive (no pickle, safe serialization).

    Uses the same unprefixed key schema as ``FitResult._save_npz`` / read by
    ``FitResult._load_npz`` (the only FitResult npz reader), so files written
    here load back through ``FitResult.load(...)``.

    Args:
        result: A FitResult instance (from rheojax.core.fit_result).
        filepath: Destination path.
        compressed: Use compressed npz (default: True).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    param_names = list(result.params.keys())
    param_values = np.array([result.params[n] for n in param_names], dtype=np.float64)

    # Build arrays dict — all values are numpy arrays (no pickle used).
    # Key names match FitResult._load_npz's expectations exactly.
    arrays: dict[str, np.ndarray] = {
        "model_name": _encode_str(result.model_name or ""),
        "model_class_name": _encode_str(result.model_class_name or ""),
        "protocol": _encode_str(result.protocol or ""),
        "n_params": np.array(result.n_params),
        "timestamp": _encode_str(result.timestamp or ""),
        "param_names": _encode_str(json.dumps(param_names)),
        "param_values": param_values,
        "success": np.array(result.success),
        "n_data": np.array(result.n_data),
    }
    if result.optimization_result is not None:
        arrays["_is_complex_split"] = np.array(
            result.optimization_result._is_complex_split
        )

    # Units as JSON-encoded string (informational; not read back by
    # FitResult._load_npz, which always sets params_units={}).
    if result.params_units:
        arrays["_params_units"] = _encode_str(json.dumps(result.params_units))

    # Fitted curve
    if result.fitted_curve is not None:
        arrays["fitted_curve"] = np.asarray(result.fitted_curve)

    # Input data
    if result.X is not None:
        arrays["X"] = np.asarray(result.X)
    if result.y is not None:
        arrays["y"] = np.asarray(result.y)

    # Statistics as JSON-encoded string (informational; not read back by
    # FitResult._load_npz, which recomputes stats from y/fitted_curve).
    stats = {}
    for attr_name in ("r_squared", "aic", "bic", "rmse", "mae"):
        val = getattr(result, attr_name, None)
        if val is not None:
            stats[attr_name] = float(val)
    if stats:
        arrays["_stats"] = _encode_str(json.dumps(stats))

    save_fn = np.savez_compressed if compressed else np.savez
    save_fn(filepath, **arrays)  # type: ignore[arg-type]

    logger.info(
        "Saved FitResult to npz",
        filepath=str(filepath),
        model_name=result.model_name,
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

    with npz:
        x = npz["x"]
        y = npz["y"]

        # Validate array shape compatibility before constructing RheoData
        if x.ndim >= 1 and y.ndim >= 1 and len(x) != len(y):
            raise ValueError(
                f"Corrupt npz archive: x has {len(x)} points but y has {len(y)}"
            )

        # Parse metadata from UTF-8 bytes
        try:
            metadata: dict = json.loads(
                npz["_metadata_json"].tobytes().decode("utf-8")
            )
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            logger.warning("Could not parse metadata JSON from npz, using empty dict")
            metadata = {}

        x_units = _decode_str(npz["_x_units"]) if "_x_units" in npz else None
        y_units = _decode_str(npz["_y_units"]) if "_y_units" in npz else None
        domain = _decode_str(npz["_domain"]) if "_domain" in npz else None
        initial_test_mode = (
            _decode_str(npz["_initial_test_mode"])
            if "_initial_test_mode" in npz
            else None
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
        domain=domain or "time",
        initial_test_mode=initial_test_mode,
        metadata=metadata,
        validate=True,
    )
