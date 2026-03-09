"""Shared JSON encoder for numpy/JAX types.

Used by both analysis_exporter and npz_writer to serialize numpy scalars,
arrays, and JAX arrays into JSON-compatible Python types.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/JAX scalar and array types.

    Handles:
    - numpy ndarrays → lists
    - numpy integer/floating/bool scalars → Python int/float/bool
    - numpy complex → dict with real/imag
    - JAX arrays (scalar or shaped) → Python scalar or list
    - Non-serializable types → str fallback
    """

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
            return {"__complex__": True, "real": float(obj.real), "imag": float(obj.imag)}
        # JAX arrays
        if hasattr(obj, "item") and hasattr(obj, "shape"):
            try:
                if obj.shape == () or obj.size == 1:
                    return obj.item()
                return np.asarray(obj).tolist()
            except (TypeError, ValueError):
                pass
        # Fallback: stringify non-serializable types (Path, datetime, enum, etc.)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
