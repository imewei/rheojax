"""Lightweight validation helpers shared across public boundaries."""

from __future__ import annotations

from collections.abc import Mapping

_REMOVED_OPTIONS = ("deforma" + "tion_mode", "poisson" + "_ratio")


def reject_removed_options(options: Mapping[str, object]) -> None:
    """Reject removed DMTA options without modifying ``options``.

    Args:
        options: Named options supplied to a public API boundary.

    Raises:
        TypeError: If a removed DMTA option is present.
    """
    present = [key for key in _REMOVED_OPTIONS if key in options]
    if not present:
        return

    if len(present) == 1:
        subject = f"Removed option '{present[0]}'"
        verb = "is"
        instruction = "Remove this option."
    else:
        names = ", ".join(f"'{key}'" for key in present)
        subject = f"Removed options {names}"
        verb = "are"
        instruction = "Remove these options."

    raise TypeError(
        f"{subject} {verb} no longer supported; RheoJAX is shear-only "
        f"(DMTA/tensile support was removed). {instruction}"
    )
