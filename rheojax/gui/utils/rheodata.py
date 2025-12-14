"""Helpers for converting GUI state into core RheoData."""

from __future__ import annotations

from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.state.store import DatasetState


def rheodata_from_dataset_state(dataset: DatasetState) -> RheoData:
    """Convert a GUI DatasetState into a core RheoData.

    Notes
    -----
    - For oscillation datasets, data may arrive as separate real arrays for
      storage (G') and loss (G'') moduli via ``y_data`` and ``y2_data``.
      In that case, this helper combines them into a complex modulus
      ``y = G' + 1j*G''`` so downstream code (plotting/inference) can
      treat oscillation consistently.
    """
    x = getattr(dataset, "x_data", None)
    y = getattr(dataset, "y_data", None)
    y2 = getattr(dataset, "y2_data", None)
    if x is None or y is None:
        raise ValueError("DatasetState is missing x/y data")

    metadata = dict(getattr(dataset, "metadata", {}) or {})
    test_mode = dataset.test_mode or metadata.get("test_mode")
    metadata.setdefault("test_mode", test_mode)

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    if (test_mode or "") == "oscillation" and y2 is not None and not np.iscomplexobj(y_arr):
        y2_arr = np.asarray(y2)
        # Combine into complex modulus G* = G' + iG''.
        if y2_arr.shape == y_arr.shape:
            y_arr = y_arr.astype(float) + 1j * y2_arr.astype(float)

    return RheoData(
        x=x_arr,
        y=y_arr,
        x_units=metadata.get("x_units"),
        y_units=metadata.get("y_units"),
        domain=metadata.get("domain", "time"),
        metadata=metadata,
        initial_test_mode=test_mode,
        validate=False,
    )


def rheodata_from_any(data: Any) -> RheoData:
    """Best-effort conversion of a DatasetState/RheoData-like object into RheoData."""
    if isinstance(data, DatasetState):
        return rheodata_from_dataset_state(data)
    if isinstance(data, RheoData):
        return data
    if hasattr(data, "x") and hasattr(data, "y"):
        return RheoData(
            x=np.asarray(getattr(data, "x")),
            y=np.asarray(getattr(data, "y")),
            metadata=dict(getattr(data, "metadata", {}) or {}),
            initial_test_mode=getattr(data, "initial_test_mode", None),
            validate=False,
        )
    raise TypeError(f"Unsupported data type for rheodata conversion: {type(data)}")

