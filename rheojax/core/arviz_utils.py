"""Utilities for safely importing optional ArviZ dependency."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterable
from types import ModuleType


def import_arviz(*, required: Iterable[str] | None = None) -> ModuleType:
    """Return the ArviZ module or raise ImportError/RuntimeError as needed.

    Args:
        required: Optional iterable of attribute names that must exist on the
            ArviZ module (e.g., "plot_pair", "plot_forest", "from_numpyro").

    Raises:
        ImportError: When ArviZ itself cannot be imported or has been disabled.
        RuntimeError: When ArviZ imports but is missing required helpers.
    """

    if sys.modules.get("arviz", object()) is None:
        # Explicitly disabled (e.g., via monkeypatch during testing)
        raise ImportError(
            "ArviZ appears to be disabled. Re-import the module to restore plotting capabilities."
        )

    try:
        arviz = importlib.import_module("arviz")
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised when ArviZ absent
        raise ImportError(
            "ArviZ is required for Bayesian diagnostics. Install it with `pip install arviz`."
        ) from exc

    if required:
        missing = sorted(name for name in required if not hasattr(arviz, name))
        if missing:
            missing_csv = ", ".join(missing)
            raise RuntimeError(
                "ArviZ is installed but missing required component(s) "
                f"[{missing_csv}]. Reinstall ArviZ with optional plotting extras "
                "(`pip install arviz[plots]`) to enable Bayesian diagnostics."
            )

    return arviz
