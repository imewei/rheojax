"""Lightweight validation helpers shared across public boundaries."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# This module is the single rejection boundary for legacy public keyword
# arguments removed in v0.7.0. Production code must never forward these names.
REMOVED_OPTION_NAMES = ("deformation_mode", "poisson_ratio")


def reject_removed_options(options: Mapping[str, object]) -> None:
    """Reject removed DMTA options without modifying ``options``.

    Args:
        options: Named options supplied to a public API boundary.

    Raises:
        TypeError: If a removed DMTA option is present.
    """
    present = [key for key in REMOVED_OPTION_NAMES if key in options]
    if not present:
        return
    names = ", ".join(f"'{key}'" for key in present)
    raise TypeError(
        f"Removed option(s) {names} no longer supported; RheoJAX is shear-only "
        f"(DMTA/tensile support was removed). Remove them."
    )


def validate_predict_input(X: object, *, name: str = "X") -> None:
    """Validate a predict() input array at the public API boundary.

    Checks for NaNs and, for a 1-D axis (time/frequency), monotonic
    non-decreasing ordering. Runs as a host-side NumPy check (eager, not
    inside JIT) so it raises a clear error before bad data ever reaches an
    ODE solver, where a NaN silently produces garbage instead of an error.

    Args:
        X: Input array-like (NumPy or JAX array).
        name: Name used in the raised error message.

    Raises:
        ValueError: If X contains NaN, is empty, or if a 1-D X is not
            monotonically non-decreasing.
    """
    arr = np.asarray(X)
    if arr.size == 0:
        raise ValueError(f"{name} is empty; cannot predict on zero data points.")
    if np.issubdtype(arr.dtype, np.number) and np.isnan(arr).any():
        raise ValueError(
            f"{name} contains NaN value(s); check input data for missing/invalid entries."
        )
    if arr.ndim == 1 and arr.size > 1 and np.issubdtype(arr.dtype, np.number):
        if np.any(np.diff(arr) < 0):
            raise ValueError(
                f"{name} must be monotonically non-decreasing (e.g. a time/frequency axis)."
            )
