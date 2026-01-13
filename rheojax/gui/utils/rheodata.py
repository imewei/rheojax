"""Helpers for converting GUI state into core RheoData."""

from __future__ import annotations

from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.gui.state.store import DatasetState
from rheojax.logging import get_logger

logger = get_logger(__name__)


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
    logger.debug(
        "Entering rheodata_from_dataset_state", dataset_type=type(dataset).__name__
    )
    x = getattr(dataset, "x_data", None)
    y = getattr(dataset, "y_data", None)
    y2 = getattr(dataset, "y2_data", None)
    if x is None or y is None:
        logger.error(
            "DatasetState is missing x/y data",
            has_x=x is not None,
            has_y=y is not None,
            exc_info=True,
        )
        raise ValueError("DatasetState is missing x/y data")

    metadata = dict(getattr(dataset, "metadata", {}) or {})
    test_mode = dataset.test_mode or metadata.get("test_mode")
    metadata.setdefault("test_mode", test_mode)
    logger.debug(
        "Extracted dataset info",
        test_mode=test_mode,
        has_y2=y2 is not None,
        metadata_keys=list(metadata.keys()),
    )

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    logger.debug(
        "Converted arrays",
        x_shape=x_arr.shape,
        y_shape=y_arr.shape,
        y_dtype=str(y_arr.dtype),
    )

    if (
        (test_mode or "") == "oscillation"
        and y2 is not None
        and not np.iscomplexobj(y_arr)
    ):
        y2_arr = np.asarray(y2)
        # Combine into complex modulus G* = G' + iG''.
        if y2_arr.shape == y_arr.shape:
            logger.debug(
                "Combining G' and G'' into complex modulus",
                y_shape=y_arr.shape,
                y2_shape=y2_arr.shape,
            )
            y_arr = y_arr.astype(float) + 1j * y2_arr.astype(float)

    rheo_data = RheoData(
        x=x_arr,
        y=y_arr,
        x_units=metadata.get("x_units"),
        y_units=metadata.get("y_units"),
        domain=metadata.get("domain", "time"),
        metadata=metadata,
        initial_test_mode=test_mode,
        validate=False,
    )
    logger.debug(
        "Created RheoData",
        x_shape=rheo_data.x.shape,
        y_shape=rheo_data.y.shape,
        test_mode=test_mode,
    )
    return rheo_data


def rheodata_from_any(data: Any) -> RheoData:
    """Best-effort conversion of a DatasetState/RheoData-like object into RheoData."""
    logger.debug("Entering rheodata_from_any", data_type=type(data).__name__)
    if isinstance(data, DatasetState):
        logger.debug("Data is DatasetState, delegating to rheodata_from_dataset_state")
        return rheodata_from_dataset_state(data)
    if isinstance(data, RheoData):
        logger.debug("Data is already RheoData, returning as-is")
        return data
    if hasattr(data, "x") and hasattr(data, "y"):
        logger.debug("Data has x and y attributes, creating RheoData")
        rheo_data = RheoData(
            x=np.asarray(data.x),
            y=np.asarray(data.y),
            metadata=dict(getattr(data, "metadata", {}) or {}),
            initial_test_mode=getattr(data, "initial_test_mode", None),
            validate=False,
        )
        logger.debug(
            "Created RheoData from x/y attributes",
            x_shape=rheo_data.x.shape,
            y_shape=rheo_data.y.shape,
        )
        return rheo_data
    logger.error(
        "Unsupported data type for rheodata conversion",
        data_type=type(data).__name__,
        exc_info=True,
    )
    raise TypeError(f"Unsupported data type for rheodata conversion: {type(data)}")
