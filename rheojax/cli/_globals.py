"""Shared argparse parent parser and global flag handling for the RheoJAX CLI.

All CLI subcommands should use ``create_global_parser()`` as their
``parents`` entry so that common flags are available everywhere without
duplication.
"""

from __future__ import annotations

import argparse
import logging

from rheojax.logging import configure_logging, get_logger

logger = get_logger(__name__)

# Map --log-level strings to stdlib logging levels
_LOG_LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def create_global_parser() -> argparse.ArgumentParser:
    """Return a parent parser that carries all shared CLI flags.

    Intended to be passed as an element of ``parents=`` when constructing
    a subcommand parser.  ``add_help=False`` is set so that the parent
    parser does not emit its own ``-h`` / ``--help`` flag — the child
    parser owns that.

    Returns:
        Configured :class:`argparse.ArgumentParser` with shared flags.

    Example:
        >>> parent = create_global_parser()
        >>> parser = argparse.ArgumentParser(parents=[parent])
    """
    parent = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    verbosity = parent.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity.  Pass multiple times for more detail (-vv, -vvv).",
    )
    verbosity.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress all non-error output.",
    )

    parent.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        dest="no_color",
        help="Disable ANSI color output.",
    )

    parent.add_argument(
        "--log-level",
        choices=list(_LOG_LEVEL_MAP.keys()),
        default=None,
        dest="log_level",
        metavar="LEVEL",
        help="Set log level explicitly (DEBUG, INFO, WARNING, ERROR).  "
        "Overrides --verbose / --quiet.",
    )

    return parent


def apply_globals(args: argparse.Namespace) -> None:
    """Configure logging from parsed global flags.

    Should be called once per command invocation, immediately after
    ``parser.parse_args()``.

    Args:
        args: The parsed namespace returned by ``parser.parse_args()``.

    Example:
        >>> parsed = parser.parse_args()
        >>> apply_globals(parsed)
    """
    if args.log_level:
        level = args.log_level
    elif getattr(args, "quiet", False):
        level = "ERROR"
    else:
        verbosity = getattr(args, "verbose", 0)
        if verbosity >= 2:
            level = "DEBUG"
        elif verbosity == 1:
            level = "INFO"
        else:
            level = "WARNING"

    configure_logging(level=level)
    logger.debug("Global flags applied", log_level=level)
