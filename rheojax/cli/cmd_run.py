"""CLI subcommand for executing YAML pipeline configurations.

Delegates to rheojax.cli._yaml_runner for the actual execution logic,
keeping this module as a thin argument-parsing wrapper.

Usage:
    rheojax run pipeline.yaml
    rheojax run pipeline.yaml --override model=springpot --override max_iter=2000
    rheojax run pipeline.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for run subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax run",
        description="Execute a YAML pipeline configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Execute a pipeline config
  rheojax run my_pipeline.yaml

  # Override a config value at runtime
  rheojax run my_pipeline.yaml --override model=springpot

  # Override multiple values
  rheojax run pipeline.yaml --override model=maxwell --override max_iter=5000

  # Preview without executing
  rheojax run pipeline.yaml --dry-run

  # Generate a template to start from
  rheojax pipeline init --template basic --output pipeline.yaml
        """,
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to the YAML pipeline configuration file",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config key at runtime, e.g. --override model=maxwell "
            "(repeatable, dot-notation supported: steps.0.model=maxwell)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config and planned steps without executing",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """Execute a YAML pipeline config."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    logger.info(
        "CLI run command",
        config=str(parsed.config),
        dry_run=parsed.dry_run,
        overrides=parsed.override,
    )

    if not parsed.config.exists():
        print(f"Error: Config file not found: {parsed.config}", file=sys.stderr)
        return 1

    if parsed.config.suffix not in (".yaml", ".yml"):
        print(
            f"Warning: Expected a .yaml/.yml config file, got '{parsed.config.suffix}'",
            file=sys.stderr,
        )

    try:
        from rheojax.cli._yaml_runner import run_pipeline

        return run_pipeline(
            config_path=parsed.config,
            overrides=parsed.override,
            dry_run=parsed.dry_run,
        )
    except ImportError:
        print(
            "Error: YAML runner not available. "
            "Ensure 'rheojax.cli._yaml_runner' is installed.",
            file=sys.stderr,
        )
        logger.error("_yaml_runner module not found")
        return 1
    except Exception as e:
        print(f"Error running pipeline: {e}", file=sys.stderr)
        logger.error("Pipeline run failed", config=str(parsed.config))
        logger.debug("Pipeline run traceback", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
