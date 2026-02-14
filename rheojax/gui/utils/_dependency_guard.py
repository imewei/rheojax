"""Optional-dependency import guards for the GUI layer."""

from __future__ import annotations

from rheojax.logging import get_logger

logger = get_logger(__name__)


def require_dependency(
    module: str,
    feature: str,
    install: str | None = None,
) -> None:
    """Raise ImportError with a helpful message if *module* is not installed.

    Parameters
    ----------
    module : str
        Python package name (e.g. ``"arviz"``).
    feature : str
        Human-readable feature that needs the dependency.
    install : str | None
        pip install command hint.  Defaults to ``pip install <module>``.
    """
    install_cmd = install or f"pip install {module}"
    msg = f"'{module}' is required for {feature}. Install with: {install_cmd}"
    logger.error(msg)
    raise ImportError(msg)
