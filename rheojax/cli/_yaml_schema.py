"""YAML pipeline config schema definition and validation.

A pipeline config YAML file has the following top-level structure::

    version: "1"
    name: "My Analysis"
    defaults:
      test_mode: relaxation
    steps:
      - type: load
        file: data.csv
      - type: fit
        model: maxwell

Use :func:`load_config` to parse a file and :func:`validate_config` to
check it for structural errors before executing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rheojax.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

VALID_STEP_TYPES: frozenset[str] = frozenset(
    {"load", "transform", "fit", "bayesian", "export"}
)

# Per-step required and optional keys (excluding the universal "type" key).
STEP_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "load": {
        "required": ["file"],
        "optional": ["format", "x_col", "y_col", "y_cols", "test_mode"],
    },
    "transform": {
        "required": ["name"],
        "optional": [],  # transform-specific kwargs are allowed freely
    },
    "fit": {
        "required": ["model"],
        "optional": [
            "method",
            "max_iter",
            "test_mode",
            "deformation_mode",
            "poisson_ratio",
        ],
    },
    "bayesian": {
        "required": [],
        "optional": [
            "num_warmup",
            "num_samples",
            "num_chains",
            "seed",
            "warm_start",
        ],
    },
    "export": {
        "required": ["output"],
        "optional": ["format"],
    },
}

# Step types whose extra kwargs are passed through without validation.
_OPEN_KWARGS_STEPS: frozenset[str] = frozenset({"transform"})


# ------------------------------------------------------------------
# Dataclass
# ------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Validated representation of a YAML pipeline config file.

    Attributes:
        version: Config schema version (currently ``"1"``).
        name: Human-readable pipeline name.
        defaults: Default values merged into each step at execution time.
        steps: Ordered list of step dicts, each containing at minimum a
            ``"type"`` key.
    """

    version: str
    name: str
    defaults: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        Parsed :class:`PipelineConfig`.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the YAML structure is invalid or missing required keys.

    Example:
        >>> config = load_config("pipeline.yaml")
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load pipeline configs.  "
            "Install it with: uv add pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    logger.debug("Loading pipeline config", path=str(path))

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw).__name__}.")

    # Required top-level keys
    missing_top = [k for k in ("version", "name", "steps") if k not in raw]
    if missing_top:
        raise ValueError(f"Config is missing required top-level keys: {missing_top}")

    steps = raw.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("'steps' must be a YAML sequence.")

    config = PipelineConfig(
        version=str(raw["version"]),
        name=str(raw["name"]),
        defaults=dict(raw.get("defaults") or {}),
        steps=steps,
    )

    errors = validate_config(config)
    if errors:
        bullet_list = "\n  - ".join(errors)
        raise ValueError(f"Pipeline config validation failed:\n  - {bullet_list}")

    logger.debug(
        "Pipeline config loaded",
        name=config.name,
        version=config.version,
        num_steps=len(config.steps),
    )
    return config


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------


def validate_config(config: PipelineConfig) -> list[str]:
    """Validate a :class:`PipelineConfig`, returning error messages.

    An empty list means the config is valid.

    Args:
        config: Config to validate.

    Returns:
        List of human-readable error strings; empty when config is valid.

    Example:
        >>> errors = validate_config(config)
        >>> if errors:
        ...     for e in errors:
        ...         print(e)
    """
    errors: list[str] = []

    if config.version not in ("1", 1):
        errors.append(f"Unsupported config version '{config.version}' (expected '1').")

    if not config.name or not config.name.strip():
        errors.append("Config 'name' must be a non-empty string.")

    if not config.steps:
        errors.append("Pipeline must have at least one step.")
        return errors  # Nothing more to validate

    # First step must be a load step
    first_type = config.steps[0].get("type", "")
    if first_type != "load":
        errors.append(
            f"The first pipeline step must be of type 'load', got '{first_type}'."
        )

    has_fit = False
    for idx, step in enumerate(config.steps):
        step_label = f"Step {idx + 1}"

        if not isinstance(step, dict):
            errors.append(f"{step_label}: Each step must be a YAML mapping.")
            continue

        step_type = step.get("type")
        if step_type is None:
            errors.append(f"{step_label}: Missing required 'type' key.")
            continue

        if step_type not in VALID_STEP_TYPES:
            errors.append(
                f"{step_label}: Unknown step type '{step_type}'. "
                f"Valid types: {sorted(VALID_STEP_TYPES)}."
            )
            continue

        schema = STEP_SCHEMAS[step_type]
        step_keys = set(step.keys()) - {"type"}

        # Check required keys
        for req in schema["required"]:
            if req not in step:
                errors.append(
                    f"{step_label} ({step_type}): Missing required key '{req}'."
                )

        # For steps with fixed allowed keys, warn about unknown keys
        if step_type not in _OPEN_KWARGS_STEPS:
            allowed = set(schema["required"]) | set(schema["optional"])
            unknown = step_keys - allowed
            if unknown:
                errors.append(
                    f"{step_label} ({step_type}): Unknown keys {sorted(unknown)}. "
                    f"Allowed: {sorted(allowed)}."
                )

        if step_type == "fit":
            has_fit = True

        if step_type == "bayesian" and not has_fit:
            errors.append(
                f"{step_label} (bayesian): A 'fit' step must precede any 'bayesian' step."
            )

    # Validate file/output path safety — reject absolute paths and '..' traversal
    _PATH_KEYS = {"file", "output"}
    for idx, step in enumerate(config.steps):
        if not isinstance(step, dict):
            continue
        for key in _PATH_KEYS:
            value = step.get(key)
            if value is None or not isinstance(value, str):
                continue
            path = Path(value)
            if path.is_absolute():
                errors.append(
                    f"Step {idx + 1}: '{key}' must be a relative path, "
                    f"got absolute path '{value}'."
                )
            if ".." in path.parts:
                errors.append(
                    f"Step {idx + 1}: '{key}' must not contain '..' segments, "
                    f"got '{value}'."
                )

    return errors
