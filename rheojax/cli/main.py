"""RheoJAX CLI main entry point.

This module provides the main CLI dispatcher for rheojax commands.

Usage:
    rheojax <command> [options]

Commands:
    fit       - NLSQ model fitting
    bayesian  - Bayesian inference (NUTS sampling)
    spp       - SPP (Sequence of Physical Processes) analysis
    info      - Display package information
    inventory - List available models and transforms
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
  fit       NLSQ model fitting
  bayesian  Bayesian inference (NUTS sampling)
  spp       SPP (Sequence of Physical Processes) analysis for LAOS data
  info      Display package version and configuration info
  inventory List available models and transforms with protocol support

Examples:
  rheojax fit data.csv --model maxwell --x-col time --y-col G_t
  rheojax bayesian data.csv --model maxwell --x-col time --y-col G_t --warm-start
  rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5
  rheojax spp batch data_dir/ --omega 1.0
  rheojax info
  rheojax inventory
  rheojax inventory --protocol laos

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

    # Fit subcommand
    subparsers.add_parser(
        "fit",
        help="NLSQ model fitting",
        add_help=False,  # Let the fit module handle help
    )

    # Bayesian subcommand
    subparsers.add_parser(
        "bayesian",
        help="Bayesian inference (NUTS sampling)",
        add_help=False,  # Let the bayesian module handle help
    )

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

    # Inventory subcommand
    inventory_parser = subparsers.add_parser(
        "inventory",
        help="List available models and transforms",
    )
    inventory_parser.add_argument(
        "--protocol",
        type=str,
        help="Filter models by protocol (e.g., laos, creep)",
    )
    inventory_parser.add_argument(
        "--type",
        type=str,
        dest="transform_type",
        help="Filter transforms by type (e.g., spectral, superposition)",
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


def show_inventory(args: list[str] | None = None) -> int:
    """Display model and transform inventory."""
    logger.info("Running CLI command", command="inventory")

    from rheojax.core.registry import Registry

    # Parse args if passed directly (though main parser usually handles this)
    # Here we assume args are already parsed by main parser if we had access to the namespace
    # But since main() dispatch passes raw args list, we might need to parse again or just use Registry
    # Actually main() dispatch structure doesn't pass parsed args, so we rely on Registry capabilities.
    # Wait, main() passes `args[1:]` to subcommands. But `inventory` parser was defined in main parser.
    # Let's just use the Registry directly. If filtering is needed, we'd need the parsed args.

    # To keep it simple and consistent with existing `spp` pattern:
    # We will re-parse the sub-arguments for inventory here.
    parser = argparse.ArgumentParser(prog="rheojax inventory")
    parser.add_argument(
        "--protocol",
        type=str,
        help="Filter models by protocol",
    )
    parser.add_argument(
        "--type",
        type=str,
        dest="transform_type",
        help="Filter transforms by type",
    )
    parsed_args = parser.parse_args(args or [])

    registry = Registry.get_instance()

    # Force discovery of all models if not already loaded
    # In a real package this might happen automatically via imports
    # For now, let's assume models are imported. If not, we might need explicit discovery.
    # We can try to import the main package to trigger registration.
    import rheojax.models  # noqa
    import rheojax.transforms  # noqa

    inv = registry.inventory()

    print("\nRheoJAX Inventory")
    print("=================")

    # Models
    print("\nModels")
    print("------")
    models = inv["all_models"]

    if parsed_args.protocol:
        proto = parsed_args.protocol.lower()
        models = [m for m in models if proto in m["protocols"]]
        print(f"(Filtered by protocol: {proto})")

    if not models:
        print("  No models found.")
    else:
        # Determine column widths
        name_width = max(len(m["name"]) for m in models) + 2
        proto_width = 40

        print(f"{'Name':<{name_width}} {'Protocols':<{proto_width}} {'Description'}")
        print(f"{'-'*name_width} {'-'*proto_width} {'-'*30}")

        for m in sorted(models, key=lambda x: x["name"]):
            name = m["name"]
            protos = ", ".join(m["protocols"])
            if len(protos) > proto_width - 3:
                protos = protos[: proto_width - 3] + "..."
            desc = m["description"] or ""
            if len(desc) > 50:
                desc = desc[:47] + "..."

            print(f"{name:<{name_width}} {protos:<{proto_width}} {desc}")

    # Transforms
    print("\nTransforms")
    print("----------")
    transforms = inv["all_transforms"]

    if parsed_args.transform_type:
        ttype = parsed_args.transform_type.lower()
        transforms = [t for t in transforms if t["type"] == ttype]
        print(f"(Filtered by type: {ttype})")

    if not transforms:
        print("  No transforms found.")
    else:
        name_width = max(len(t["name"]) for t in transforms) + 2
        type_width = 15

        print(f"{'Name':<{name_width}} {'Type':<{type_width}} {'Description'}")
        print(f"{'-'*name_width} {'-'*type_width} {'-'*30}")

        for t in sorted(transforms, key=lambda x: x["name"]):
            name = t["name"]
            ttype = t["type"] or "N/A"
            desc = t["description"] or ""
            if len(desc) > 50:
                desc = desc[:47] + "..."

            print(f"{name:<{name_width}} {ttype:<{type_width}} {desc}")

    print("\n")
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

    if command == "fit":
        logger.info("Running CLI command", command="fit")
        from rheojax.cli.fit import main as fit_main

        return fit_main(args[1:])

    elif command == "bayesian":
        logger.info("Running CLI command", command="bayesian")
        from rheojax.cli.bayesian import main as bayesian_main

        return bayesian_main(args[1:])

    elif command == "spp":
        logger.info("Running CLI command", command="spp")
        from rheojax.cli.spp import main as spp_main

        return spp_main(args[1:])

    elif command == "info":
        return show_info()

    elif command == "inventory":
        return show_inventory(args[1:])

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
