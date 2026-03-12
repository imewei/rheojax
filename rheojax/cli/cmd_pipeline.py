"""CLI subcommand for pipeline management (init, validate, show).

Provides utilities for creating, validating, and inspecting YAML pipeline
configurations used by 'rheojax run'.

Usage:
    rheojax pipeline init --template basic --output pipeline.yaml
    rheojax pipeline validate pipeline.yaml
    rheojax pipeline show pipeline.yaml
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

try:
    from rheojax.cli._templates import TEMPLATES as _TEMPLATES

    _TEMPLATE_CHOICES = tuple(sorted(_TEMPLATES.keys()))
except Exception:
    # Fallback if _templates is unavailable at import time.
    # Keep in sync with TEMPLATES dict in rheojax/cli/_templates.py.
    _TEMPLATE_CHOICES = (
        "basic",
        "bayesian",
        "batch",
        "creep",
        "mastercurve",
        "oscillation",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for pipeline subcommand."""
    from rheojax.cli._globals import create_global_parser

    parser = argparse.ArgumentParser(
        prog="rheojax pipeline",
        description="Pipeline management: init, validate, and show YAML configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[create_global_parser()],
        epilog="""
Examples:
  # Generate a basic pipeline template
  rheojax pipeline init --template basic --output pipeline.yaml

  # List available templates
  rheojax pipeline init --list

  # Validate a pipeline config
  rheojax pipeline validate pipeline.yaml

  # Show a human-readable summary of a config
  rheojax pipeline show pipeline.yaml

  # Then execute the config
  rheojax run pipeline.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # init subcommand
    init_parser = subparsers.add_parser(
        "init",
        help="Write a template YAML pipeline config",
    )
    init_parser.add_argument(
        "--template",
        "-t",
        type=str,
        default="basic",
        choices=_TEMPLATE_CHOICES,
        help=f"Template type: {', '.join(_TEMPLATE_CHOICES)} (default: basic)",
    )
    init_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("pipeline.yaml"),
        help="Output file path (default: pipeline.yaml)",
    )
    init_parser.add_argument(
        "--list",
        action="store_true",
        dest="list_templates",
        help="List available templates and exit",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists",
    )

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a pipeline YAML config against the schema",
    )
    validate_parser.add_argument(
        "config",
        type=Path,
        help="Path to the pipeline YAML config file",
    )

    # show subcommand
    show_parser = subparsers.add_parser(
        "show",
        help="Print a human-readable summary of a pipeline config",
    )
    show_parser.add_argument(
        "config",
        type=Path,
        help="Path to the pipeline YAML config file",
    )

    return parser


def run_init(args: argparse.Namespace) -> int:
    """Write a template YAML pipeline config."""
    # List templates mode
    if args.list_templates:
        try:
            from rheojax.cli._templates import list_templates

            templates = list_templates()
        except ImportError:
            templates = [{"name": n, "description": ""} for n in _TEMPLATE_CHOICES]
        print("Available pipeline templates:")
        for t in templates:
            desc = t.get("description", "")
            if desc:
                print(f"  {t['name']} — {desc}")
            else:
                print(f"  {t['name']}")
        return 0

    # Check for existing output file
    if args.output.exists() and not args.force:
        print(
            f"Error: Output file '{args.output}' already exists. "
            "Use --force to overwrite.",
            file=sys.stderr,
        )
        return 1

    try:
        from rheojax.cli._templates import write_template

        write_template(args.template, args.output)
        logger.info(
            "Template written",
            template=args.template,
            output=str(args.output),
        )
        print(f"Template '{args.template}' written to: {args.output}")
        print(f"Edit the file, then run: rheojax run {args.output}")
        return 0
    except ImportError:
        print(
            "Error: Template system not available. "
            "Ensure 'rheojax.cli._templates' is installed.",
            file=sys.stderr,
        )
        return 1
    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error writing template: {e}", file=sys.stderr)
        logger.error("Template write failed")
        logger.debug("Template write traceback", exc_info=True)
        return 1


def run_validate(args: argparse.Namespace) -> int:
    """Validate a pipeline YAML config against the schema."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        from rheojax.cli._yaml_schema import load_config

        load_config(args.config)
        # load_config raises ValueError with all error messages if validation
        # fails, so reaching this line means the config is valid.
    except ImportError:
        print(
            "Error: YAML schema validator not available. "
            "Ensure 'rheojax.cli._yaml_schema' is installed.",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error loading config '{args.config}': {e}", file=sys.stderr)
        logger.error("Config load failed", config=str(args.config))
        logger.debug("Config load traceback", exc_info=True)
        return 1

    print(f"Config is valid: {args.config}")
    return 0


def run_show(args: argparse.Namespace) -> int:
    """Print a human-readable summary of a pipeline config."""
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        import yaml  # type: ignore[import-untyped]

        with open(args.config) as fh:
            config = yaml.safe_load(fh)
    except ImportError:
        print(
            "Error: PyYAML is required. Install with: pip install pyyaml",
            file=sys.stderr,
        )
        return 1
    except Exception as e:
        print(f"Error reading config '{args.config}': {e}", file=sys.stderr)
        return 1

    if not isinstance(config, dict):
        print(
            f"Error: Config must be a YAML mapping, got {type(config).__name__}",
            file=sys.stderr,
        )
        return 1

    # Plain summary from raw YAML dict (current schema only)
    print(f"Pipeline config: {args.config}")
    print("-" * 50)

    name = config.get("name", "(unnamed)")
    version = config.get("version", "?")
    print(f"Name:    {name}")
    print(f"Version: {version}")

    defaults = config.get("defaults") or {}
    if defaults:
        print(f"Defaults: {defaults}")

    steps = config.get("steps") or []
    if steps:
        print(f"Steps:   {len(steps)}")
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                step_type = step.get("type", "(unknown)")
                print(f"  {i}. {step_type}")
            else:
                print(f"  {i}. {step}")
    else:
        print("Steps:   (none defined)")

    # Non-blocking validation: show warnings but don't fail the command.
    try:
        from rheojax.cli._yaml_schema import PipelineConfig, validate_config

        pc = PipelineConfig(
            version=str(config.get("version", "")),
            name=str(config.get("name", "")),
            defaults=dict(config.get("defaults") or {}),
            steps=list(config.get("steps") or []),
        )
        errors = validate_config(pc)
        if errors:
            print()
            print("Validation warnings:")
            for err in errors:
                print(f"  - {err}")
    except Exception:
        pass  # Validation is best-effort in show mode

    return 0


def main(args: list[str] | None = None) -> int:
    """Dispatch pipeline management subcommands."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    from rheojax.cli._globals import apply_globals

    apply_globals(parsed)

    if parsed.command is None:
        parser.print_help()
        return 0

    if parsed.command == "init":
        return run_init(parsed)
    elif parsed.command == "validate":
        return run_validate(parsed)
    elif parsed.command == "show":
        return run_show(parsed)
    else:
        print(f"Unknown pipeline subcommand: {parsed.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
