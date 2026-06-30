"""Lightweight validation helpers shared across public boundaries."""

from __future__ import annotations

from collections.abc import Mapping

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
