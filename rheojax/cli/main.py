"""RheoJAX CLI main entry point.

This module provides the main CLI dispatcher for rheojax commands.

Usage:
    rheojax <command> [options]

Commands:
    spp     - SPP (Sequence of Physical Processes) analysis
    info    - Display package information
"""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING

from rheojax.logging import configure_logging, get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_main_parser() -> argparse.ArgumentParser:
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog="rheojax",
        description="RheoJAX: JAX-accelerated rheological analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  spp       SPP (Sequence of Physical Processes) analysis for LAOS data
  info      Display package version and configuration info

Examples:
  rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5
  rheojax spp batch data_dir/ --omega 1.0
  rheojax info

For command-specific help:
  rheojax <command> --help
        """,
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # SPP subcommand
    subparsers.add_parser(
        "spp",
        help="SPP (Sequence of Physical Processes) analysis",
        add_help=False,  # Let the spp module handle help
    )

    # Info subcommand
    subparsers.add_parser(
        "info",
        help="Display package information",
    )

    return parser


def show_info() -> int:
    """Display package information."""
    logger.info("Running CLI command", command="info")

    import rheojax
    from rheojax.core.jax_config import safe_import_jax

    jax, jnp = safe_import_jax()

    float64_enabled = jnp.array([1.0]).dtype == jnp.float64
    devices = jax.devices()

    logger.debug(
        "Package configuration",
        version=rheojax.__version__,
        jax_version=jax.__version__,
        float64=float64_enabled,
        devices=str(devices),
    )

    print("RheoJAX Package Information")
    print("=" * 40)
    print(f"Version:     {rheojax.__version__}")
    print(f"JAX version: {jax.__version__}")
    print(f"Float64:     {float64_enabled}")
    print(f"Devices:     {devices}")
    print("=" * 40)

    return 0


def main(args: list[str] | None = None) -> int:
    """Main CLI entry point."""
    # Configure logging if not already configured
    configure_logging()

    if args is None:
        args = sys.argv[1:]

    logger.debug("CLI invoked", raw_args=args)

    # If no args, show help
    if not args:
        logger.debug("No arguments provided, showing help")
        parser = create_main_parser()
        parser.print_help()
        return 0

    # Check for version flag
    if args[0] in ("--version", "-V"):
        import rheojax

        logger.info("Running CLI command", command="version")
        print(f"rheojax {rheojax.__version__}")
        return 0

    # Dispatch to subcommand
    command = args[0]
    logger.debug("Dispatching to subcommand", command=command, subargs=args[1:])

    if command == "spp":
        logger.info("Running CLI command", command="spp")
        from rheojax.cli.spp import main as spp_main

        return spp_main(args[1:])

    elif command == "info":
        return show_info()

    elif command in ("--help", "-h"):
        logger.debug("Help requested")
        parser = create_main_parser()
        parser.print_help()
        return 0

    else:
        logger.error("Unknown command", command=command)
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Use 'rheojax --help' for available commands", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
