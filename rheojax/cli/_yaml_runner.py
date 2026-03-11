"""YAML pipeline config execution engine.

Translates a validated :class:`~rheojax.cli._yaml_schema.PipelineConfig`
into a :class:`~rheojax.pipeline.builder.PipelineBuilder` call-chain and
executes it.

Typical usage::

    exit_code = run_pipeline("pipeline.yaml", overrides=["defaults.test_mode=oscillation"])
"""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from rheojax.cli._output import create_progress, get_console, print_error, print_success
from rheojax.cli._yaml_schema import PipelineConfig, load_config, validate_config
from rheojax.logging import get_logger
from rheojax.pipeline.builder import PipelineBuilder

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Config -> Builder translation
# ------------------------------------------------------------------

# Per-step-type keys that may be populated from config.defaults.
# Keys not listed here are silently dropped when merging defaults into a step.
_STEP_ALLOWED_DEFAULTS: dict[str, set[str]] = {
    "load": {"test_mode", "deformation_mode", "poisson_ratio", "format", "x_col", "y_col", "y_cols"},
    "transform": set(),  # transforms take their own kwargs, not protocol defaults
    "fit": {"test_mode", "deformation_mode", "poisson_ratio", "method", "max_iter"},
    "bayesian": {
        "test_mode",
        "deformation_mode",
        "poisson_ratio",
        "num_warmup",
        "num_samples",
        "num_chains",
        "seed",
        "warm_start",
    },
    "export": set(),  # export has no rheological defaults
}


def config_to_builder(config: PipelineConfig) -> PipelineBuilder:
    """Convert a :class:`PipelineConfig` into a :class:`PipelineBuilder`.

    Defaults defined in ``config.defaults`` are merged into each step's
    kwargs before forwarding to the builder, with explicit step values
    taking precedence.

    Args:
        config: Validated pipeline configuration.

    Returns:
        Populated :class:`PipelineBuilder` ready to call ``.build()``.

    Raises:
        ValueError: If a step type is unrecognised (should not happen after
            validation, but guards against programmatic misuse).

    Example:
        >>> builder = config_to_builder(config)
        >>> pipeline = builder.build()
    """
    builder = PipelineBuilder()

    for step in config.steps:
        step_type: str = step.get("type", "")

        # Filter defaults to only keys allowed for this step type, then merge.
        # Explicit step values always win over defaults.
        allowed_defaults = _STEP_ALLOWED_DEFAULTS.get(step_type, set())
        filtered_defaults = {k: v for k, v in config.defaults.items() if k in allowed_defaults}
        merged: dict[str, Any] = {**filtered_defaults, **step}
        merged.pop("type")  # Remove the "type" sentinel; already captured above

        logger.debug("Translating step", step_type=step_type, kwargs=list(merged.keys()))

        if step_type == "load":
            file_path = merged.pop("file")
            fmt = merged.pop("format", "auto")
            builder.add_load_step(file_path=file_path, format=fmt, **merged)

        elif step_type == "transform":
            transform_name = merged.pop("name")
            builder.add_transform_step(transform_name, **merged)

        elif step_type == "fit":
            model_name = merged.pop("model")
            builder.add_fit_step(model_name, **merged)

        elif step_type == "bayesian":
            builder.add_bayesian_step(**merged)

        elif step_type == "export":
            output_path = merged.pop("output")
            builder.add_export_step(output_path, **merged)

        else:
            raise ValueError(f"Unrecognised step type: '{step_type}'")

    return builder


# ------------------------------------------------------------------
# Override application
# ------------------------------------------------------------------

# Matches dotted path segments like "steps.2.model" or "defaults.test_mode"
_OVERRIDE_RE = re.compile(r"^([a-zA-Z_][\w.]*)\s*=\s*(.*)$")


def apply_overrides(config: PipelineConfig, overrides: list[str]) -> PipelineConfig:
    """Apply CLI ``--override key=value`` pairs to a :class:`PipelineConfig`.

    Override paths use dot notation:

    * ``defaults.test_mode=oscillation`` — sets a top-level default.
    * ``steps.2.model=zener`` — mutates the ``model`` key of step index 2
      (0-based).

    Args:
        config: Original config (will not be mutated; a deep-copy is returned).
        overrides: List of ``"path=value"`` override strings.

    Returns:
        New :class:`PipelineConfig` with overrides applied.

    Raises:
        ValueError: If an override path is malformed or the target does not
            exist.

    Example:
        >>> config = apply_overrides(config, ["defaults.test_mode=creep", "steps.1.model=zener"])
    """
    if not overrides:
        return config

    # Work on a deep copy so the original remains unchanged
    steps_copy = deepcopy(config.steps)
    defaults_copy = deepcopy(config.defaults)

    for override in overrides:
        match = _OVERRIDE_RE.match(override)
        if not match:
            raise ValueError(
                f"Invalid override format: '{override}'. "
                "Expected 'path.to.key=value'."
            )

        path_str, raw_value = match.group(1), match.group(2)
        value: Any = _coerce_value(raw_value)

        parts = path_str.split(".")

        if parts[0] == "defaults":
            if len(parts) < 2:
                raise ValueError(
                    f"Override path '{path_str}' must specify a key under 'defaults'."
                )
            _nested_set(defaults_copy, parts[1:], value)

        elif parts[0] == "steps":
            if len(parts) < 3:
                raise ValueError(
                    f"Override path '{path_str}' must be 'steps.<index>.<key>'."
                )
            try:
                step_idx = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Step index in override '{path_str}' must be an integer."
                ) from exc
            if step_idx < 0:
                raise ValueError(
                    f"Step index must be non-negative, got {step_idx} in override '{path_str}'."
                )
            if step_idx >= len(steps_copy):
                raise ValueError(
                    f"Override path '{path_str}': step index {step_idx} is out of "
                    f"range (pipeline has {len(steps_copy)} steps)."
                )
            _nested_set(steps_copy[step_idx], parts[2:], value)

        else:
            raise ValueError(
                f"Override path '{path_str}' must start with 'defaults' or 'steps'."
            )

        logger.debug("Applied override", path=path_str, value=value)

    return PipelineConfig(
        version=config.version,
        name=config.name,
        defaults=defaults_copy,
        steps=steps_copy,
    )


def _coerce_value(raw: str) -> Any:
    """Attempt to coerce a string override value to a Python scalar."""
    raw = raw.strip()
    # Boolean
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    # None
    if raw.lower() in ("null", "none", "~"):
        return None
    # Integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Float
    try:
        return float(raw)
    except ValueError:
        pass
    # String fallback (strip surrounding quotes if present)
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    return raw


_STEP_ALLOWED_KEYS: dict[str, set[str]] = {
    "load": {"file", "format", "x_col", "y_col", "y_cols", "test_mode", "deformation_mode", "poisson_ratio"},
    "fit": {"model", "method", "max_iter", "test_mode", "deformation_mode", "poisson_ratio", "use_jax", "params"},
    "bayesian": {"num_warmup", "num_samples", "num_chains", "seed", "warm_start", "target_accept_prob", "test_mode"},
    "export": {"output", "format"},
    # "transform" is intentionally absent — open kwargs by design.
}


def _nested_set(target: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value in a nested dict via a list of key segments.

    When the first key targets a step dict (has a ``"type"`` key), the
    override key is validated against the step-type allowlist to catch
    typos and accidental injections early.
    """
    # Validate override key against step-type allowlist.
    if "type" in target and len(keys) == 1:
        step_type = target.get("type", "")
        allowed = _STEP_ALLOWED_KEYS.get(step_type)
        if allowed is not None and keys[0] not in allowed:
            logger.warning(
                "Override injects unrecognised key into %s step: '%s'. "
                "Allowed: %s",
                step_type,
                keys[0],
                sorted(allowed),
            )

    for key in keys[:-1]:
        if key not in target or not isinstance(target[key], dict):
            target[key] = {}
        target = target[key]
    target[keys[-1]] = value


# ------------------------------------------------------------------
# Dry-run display
# ------------------------------------------------------------------


def dry_run_pipeline(config: PipelineConfig) -> None:
    """Print what a pipeline would do without executing it.

    Args:
        config: Validated pipeline configuration to describe.

    Example:
        >>> dry_run_pipeline(config)
    """
    console = get_console()
    console.print(f"\n[header]Pipeline:[/header] {config.name}  (version {config.version})")
    if config.defaults:
        console.print(f"[muted]Defaults:[/muted] {config.defaults}")

    console.print(f"\n[header]{len(config.steps)} steps:[/header]")
    for idx, step in enumerate(config.steps):
        step_type = step.get("type", "?")
        extra = {k: v for k, v in step.items() if k != "type"}
        console.print(f"  [info]{idx + 1}.[/info] [header]{step_type}[/header]  {extra}")
    console.print()


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------


def run_pipeline(
    config_path: str | Path,
    overrides: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Load, optionally override, and execute a YAML pipeline config.

    Args:
        config_path: Path to the YAML pipeline config file.
        overrides: Optional list of ``"path=value"`` override strings.
        dry_run: When ``True`` prints the plan and exits without executing.

    Returns:
        Exit code — ``0`` on success, ``1`` on error.

    Example:
        >>> exit_code = run_pipeline("pipeline.yaml", overrides=["steps.1.model=zener"])
    """
    # Load -----------------------------------------------------------
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        logger.error("Failed to load pipeline config", error=str(exc))
        return 1

    # Overrides ------------------------------------------------------
    if overrides:
        try:
            config = apply_overrides(config, overrides)
        except ValueError as exc:
            print_error(str(exc))
            logger.error("Failed to apply overrides", error=str(exc))
            return 1

        # Re-validate after overrides
        errors = validate_config(config)
        if errors:
            for err in errors:
                print_error(err)
            return 1

    # Dry run --------------------------------------------------------
    if dry_run:
        dry_run_pipeline(config)
        return 0

    # Build & execute ------------------------------------------------
    try:
        builder = config_to_builder(config)
    except ValueError as exc:
        print_error(f"Pipeline construction failed: {exc}")
        logger.error("Pipeline construction failed", error=str(exc))
        return 1

    logger.info("Executing pipeline", name=config.name, num_steps=len(config.steps))

    with create_progress() as progress:
        task = progress.add_task(f"Running pipeline '{config.name}'…", total=None)
        try:
            builder.build(validate=True)
            progress.update(task, description="Done")
        except Exception as exc:
            print_error(f"Pipeline execution failed: {exc}")
            logger.error("Pipeline execution failed", error=str(exc))
            logger.debug("Pipeline execution traceback", exc_info=True)
            return 1

    print_success(f"Pipeline '{config.name}' completed successfully.")
    return 0
