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

# Bayesian sampling parameter limits — shared by validate_config() and the
# GUI's PipelineExecutionService._coerce_bayesian_int() to prevent OOM.
BAYESIAN_PARAM_LIMITS: dict[str, tuple[int, int]] = {
    "num_warmup": (1, 50_000),
    "num_samples": (1, 50_000),
    "num_chains": (1, 16),
}

# Per-step required and optional keys (excluding the universal "type" key).
STEP_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "load": {
        "required": ["file"],
        "optional": [
            "format",
            "x_col",
            "y_col",
            "y_cols",
            "test_mode",
            "deformation_mode",
            "poisson_ratio",
        ],
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
            "use_jax",  # SER-003: PipelineBuilder passes use_jax to fit steps
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
            "target_accept_prob",
            "test_mode",
            "deformation_mode",
            "poisson_ratio",
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
        raise ValueError(
            f"Config file must be a YAML mapping, got {type(raw).__name__}."
        )

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
    from rheojax.cli._yaml_runner import _STEP_ALLOWED_DEFAULTS  # noqa: PLC0415

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
        # Merge only the defaults that are relevant to this step type,
        # so that e.g. test_mode from defaults is not flagged as unknown
        # on export steps.
        _allowed_defaults = _STEP_ALLOWED_DEFAULTS.get(step_type, set())
        _filtered_defaults = {
            k: v for k, v in config.defaults.items() if k in _allowed_defaults
        }
        effective = {**_filtered_defaults, **step}
        step_keys = set(effective.keys()) - {"type"}

        # Check required keys against the effective (defaults-merged) view
        for req in schema["required"]:
            if req not in effective:
                errors.append(
                    f"{step_label} ({step_type}): Missing required key '{req}'."
                )

        # For steps with fixed allowed keys, warn about unknown keys.
        # Treated as a warning (not an error) because callers legitimately pass
        # extra kwargs such as target_accept_prob, seed, custom_priors, ftol, etc.
        if step_type not in _OPEN_KWARGS_STEPS:
            allowed = set(schema["required"]) | set(schema["optional"])
            unknown = step_keys - allowed
            if unknown:
                logger.warning(
                    "Unknown keys in pipeline step",
                    step=step_label,
                    step_type=step_type,
                    unknown=sorted(unknown),
                    allowed=sorted(allowed),
                )

        # P3-3: Log a warning if the transform name isn't in the registry.
        # This is best-effort — custom transforms or aliases may not be
        # registered at validation time, so we warn rather than error.
        if step_type == "transform":
            transform_name = step.get("name")
            if transform_name:
                try:
                    from rheojax.core.registry import TransformRegistry

                    registered = TransformRegistry.list_transforms()
                    if registered and transform_name not in registered:
                        logger.warning(
                            "Transform name not found in registry",
                            step=step_label,
                            transform=transform_name,
                            available=registered,
                        )
                except Exception:
                    pass  # Registry not available at validation time

        if step_type == "fit":
            has_fit = True

        # Validate Bayesian sampling parameter ranges to prevent OOM.
        if step_type == "bayesian":
            for key, (lo, hi) in BAYESIAN_PARAM_LIMITS.items():
                val = step.get(key)
                if val is not None:
                    try:
                        ival = int(val)
                    except (TypeError, ValueError):
                        errors.append(
                            f"{step_label} (bayesian): '{key}' must be an integer, "
                            f"got {val!r}."
                        )
                        continue
                    if not (lo <= ival <= hi):
                        errors.append(
                            f"{step_label} (bayesian): '{key}' must be between "
                            f"{lo} and {hi}, got {val}."
                        )

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
