"""Pipeline serialization for GUI <-> CLI YAML round-trip.

Provides bidirectional conversion between the GUI's
:class:`~rheojax.gui.state.store.VisualPipelineState` and the CLI's
YAML pipeline format, ensuring both interfaces share the same config.

The four public functions cover every conversion path:

* :func:`to_yaml` -- GUI steps -> YAML string (for ``rheojax run``)
* :func:`from_yaml` -- YAML string -> GUI steps
* :func:`to_pipeline_builder` -- GUI steps -> PipelineBuilder
* :func:`from_pipeline_builder` -- PipelineBuilder -> GUI state

Key field mappings
------------------
+------------+---------------------------------+-------------------------------------+
| Step type  | GUI ``config`` dict keys        | YAML keys (excluding ``type``)      |
+============+=================================+=====================================+
| load       | file, format, ...               | file, format, ...                   |
+------------+---------------------------------+-------------------------------------+
| transform  | name, ...                       | name, ...                           |
+------------+---------------------------------+-------------------------------------+
| fit        | model, method, ...              | model, method, ...                  |
+------------+---------------------------------+-------------------------------------+
| bayesian   | num_warmup, num_samples, ...    | num_warmup, num_samples, ...        |
+------------+---------------------------------+-------------------------------------+
| export     | output, format, ...             | output, format, ...                 |
+------------+---------------------------------+-------------------------------------+

Note that the GUI uses ``output`` (matching the YAML key) for the export
destination, while :class:`~rheojax.pipeline.builder.PipelineBuilder` uses
``output_path`` internally.  This module handles the translation transparently.
"""

from __future__ import annotations

import uuid
from typing import Any

from rheojax.gui.state.store import (
    PipelineStepConfig,
    StepStatus,
    VisualPipelineState,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DISPLAY_NAMES: dict[str, str] = {
    "load": "Load Data",
    "transform": "Transform",
    "fit": "Fit",
    "bayesian": "Bayesian Inference",
    "export": "Export Results",
}


def _display_name(step_type: str, config: dict[str, Any]) -> str:
    """Build a human-readable display name for a pipeline step.

    Args:
        step_type: One of the five recognised step type strings.
        config: Step configuration dict.

    Returns:
        Display name string.
    """
    if step_type == "transform":
        transform_name = config.get("name", "")
        return f"Transform: {transform_name}" if transform_name else "Transform"
    if step_type == "fit":
        model_name = config.get("model", "")
        return f"Fit: {model_name}" if model_name else "Fit"
    return _DISPLAY_NAMES.get(step_type, step_type.capitalize())


def _make_step(
    step_type: str,
    config: dict[str, Any],
    position: int,
) -> PipelineStepConfig:
    """Construct a :class:`PipelineStepConfig` with a fresh UUID.

    Args:
        step_type: Step type identifier.
        config: Step-specific configuration dict.
        position: Zero-based position in the step list.

    Returns:
        New :class:`PipelineStepConfig` with ``PENDING`` status.
    """
    return PipelineStepConfig(
        id=str(uuid.uuid4()),
        step_type=step_type,
        name=_display_name(step_type, config),
        config=config,
        status=StepStatus.PENDING,
        position=position,
    )


def _get_yaml() -> Any:
    """Import and return the ``yaml`` module, raising a clear error if absent.

    Returns:
        The ``yaml`` module.

    Raises:
        ImportError: If PyYAML is not installed.
    """
    try:
        import yaml  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for pipeline serialization.  "
            "Install it with: uv add pyyaml"
        ) from exc
    return yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def to_yaml(
    steps: list[PipelineStepConfig],
    pipeline_name: str = "Untitled Pipeline",
    defaults: dict[str, Any] | None = None,
) -> str:
    """Convert GUI pipeline steps to a YAML string.

    The resulting string is compatible with ``rheojax run <config.yaml>``.

    Args:
        steps: Ordered list of :class:`PipelineStepConfig` instances.
        pipeline_name: Human-readable name written to the ``name`` field.
        defaults: Optional mapping written to the top-level ``defaults``
            block.  Pass ``None`` or an empty dict to omit the block.

    Returns:
        YAML string representing the pipeline.

    Example:
        >>> yaml_str = to_yaml(state.steps, pipeline_name=state.pipeline_name)
        >>> Path("pipeline.yaml").write_text(yaml_str)
    """
    yaml = _get_yaml()

    yaml_steps: list[dict[str, Any]] = []
    for step in steps:
        cfg = dict(step.config)  # shallow copy — do not mutate original
        step_type = step.step_type

        # Build step dict: type first, then config keys.
        step_dict: dict[str, Any] = {"type": step_type}

        step_dict.update(cfg)

        yaml_steps.append(step_dict)

    doc: dict[str, Any] = {
        "version": "1",
        "name": pipeline_name,
        "steps": yaml_steps,
    }
    if defaults:
        # Insert defaults between name and steps for readability.
        doc = {
            "version": "1",
            "name": pipeline_name,
            "defaults": defaults,
            "steps": yaml_steps,
        }

    result: str = yaml.dump(
        doc, default_flow_style=False, allow_unicode=True, sort_keys=False
    )
    logger.debug(
        "Serialized pipeline to YAML",
        pipeline_name=pipeline_name,
        num_steps=len(steps),
    )
    return result


def from_yaml(yaml_str: str) -> tuple[list[PipelineStepConfig], str]:
    """Parse a YAML string into GUI pipeline steps.

    Args:
        yaml_str: YAML pipeline configuration string (as produced by
            :func:`to_yaml` or the ``rheojax run`` CLI).

    Returns:
        A ``(steps, pipeline_name)`` tuple where *steps* is an ordered list
        of :class:`PipelineStepConfig` instances and *pipeline_name* is the
        string from the YAML ``name`` field.

    Raises:
        ValueError: If the YAML is structurally invalid or missing required
            fields.
        ImportError: If PyYAML is not installed.

    Example:
        >>> steps, name = from_yaml(Path("pipeline.yaml").read_text())
    """
    yaml = _get_yaml()

    raw = yaml.safe_load(yaml_str)
    if not isinstance(raw, dict):
        raise ValueError(
            "Pipeline YAML must be a mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    missing = [k for k in ("version", "name", "steps") if k not in raw]
    if missing:
        raise ValueError(f"Pipeline YAML is missing required top-level keys: {missing}")

    pipeline_name: str = str(raw["name"])
    raw_steps = raw.get("steps", [])
    if not isinstance(raw_steps, list):
        raise ValueError("'steps' must be a YAML sequence.")

    steps: list[PipelineStepConfig] = []
    for idx, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"Step {idx + 1}: each step must be a YAML mapping.")

        step_type = raw_step.get("type")
        if not step_type:
            raise ValueError(f"Step {idx + 1}: missing required 'type' key.")

        # Build config dict: everything except the "type" key.
        config: dict[str, Any] = {k: v for k, v in raw_step.items() if k != "type"}

        steps.append(_make_step(step_type, config, position=idx))

    logger.debug(
        "Deserialized pipeline from YAML",
        pipeline_name=pipeline_name,
        num_steps=len(steps),
    )
    return steps, pipeline_name


def to_pipeline_builder(steps: list[PipelineStepConfig]) -> Any:
    """Convert GUI pipeline steps to a PipelineBuilder.

    Uses the same field mapping as
    :func:`~rheojax.cli._yaml_runner.config_to_builder` so that pipelines
    built from the GUI and from the CLI behave identically.

    Args:
        steps: Ordered list of :class:`PipelineStepConfig` instances.

    Returns:
        A :class:`~rheojax.pipeline.builder.PipelineBuilder` populated with
        the provided steps.  Call ``.build()`` to execute.

    Raises:
        ValueError: If an unrecognised step type is encountered.

    Example:
        >>> builder = to_pipeline_builder(state.steps)
        >>> pipeline = builder.build()
    """
    from rheojax.pipeline.builder import PipelineBuilder  # noqa: PLC0415

    builder = PipelineBuilder()

    for step in steps:
        cfg = dict(step.config)  # shallow copy — avoid mutating stored config
        step_type = step.step_type

        if step_type == "load":
            file_path = cfg.pop("file")
            fmt = cfg.pop("format", "auto")
            builder.add_load_step(file_path=file_path, format=fmt, **cfg)

        elif step_type == "transform":
            transform_name = cfg.pop("name")
            builder.add_transform_step(transform_name, **cfg)

        elif step_type == "fit":
            model_name = cfg.pop("model")
            builder.add_fit_step(model_name, **cfg)

        elif step_type == "bayesian":
            builder.add_bayesian_step(**cfg)

        elif step_type == "export":
            output_path = cfg.pop("output")
            builder.add_export_step(output_path, **cfg)

        else:
            raise ValueError(
                f"Unrecognised pipeline step type: '{step_type}'.  "
                "Expected one of: load, transform, fit, bayesian, export."
            )

    logger.debug(
        "Built PipelineBuilder from GUI steps",
        num_steps=len(steps),
    )
    return builder


def from_pipeline_builder(
    builder: Any,
    pipeline_name: str = "Untitled Pipeline",
) -> VisualPipelineState:
    """Convert a :class:`~rheojax.pipeline.builder.PipelineBuilder` to a GUI state.

    Translates the builder's internal ``(step_type, kwargs)`` tuples back into
    :class:`PipelineStepConfig` instances, reversing the key mapping applied by
    :func:`to_pipeline_builder`.

    Args:
        builder: A :class:`~rheojax.pipeline.builder.PipelineBuilder` instance
            whose ``.steps`` attribute contains ``(step_type, kwargs)`` tuples.
        pipeline_name: Name to assign to the returned
            :class:`VisualPipelineState`.

    Returns:
        :class:`VisualPipelineState` populated from the builder steps.

    Example:
        >>> state = from_pipeline_builder(builder, pipeline_name="My Analysis")
        >>> yaml_str = to_yaml(state.steps, pipeline_name=state.pipeline_name)
    """
    gui_steps: list[PipelineStepConfig] = []

    for position, (step_type, kwargs) in enumerate(builder.steps):
        cfg: dict[str, Any] = dict(kwargs)  # shallow copy

        if step_type == "load":
            # PipelineBuilder stores as file_path; GUI/YAML use file.
            if "file_path" in cfg:
                cfg["file"] = cfg.pop("file_path")

        elif step_type == "export":
            # PipelineBuilder stores as output_path; GUI/YAML use output.
            if "output_path" in cfg:
                cfg["output"] = cfg.pop("output_path")

        # transform and fit: builder already uses "name" and "model" keys
        # (set by add_transform_step / add_fit_step), so no remapping needed.
        # bayesian: all kwargs are passed through as-is.

        gui_steps.append(_make_step(step_type, cfg, position=position))

    logger.debug(
        "Converted PipelineBuilder to VisualPipelineState",
        pipeline_name=pipeline_name,
        num_steps=len(gui_steps),
    )
    return VisualPipelineState(
        steps=gui_steps,
        pipeline_name=pipeline_name,
    )


__all__ = [
    "PipelineStepConfig",
    "VisualPipelineState",
    "from_pipeline_builder",
    "from_yaml",
    "to_pipeline_builder",
    "to_yaml",
]
