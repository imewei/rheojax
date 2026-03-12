"""YAML pipeline templates for ``rheojax pipeline init``.

Each template is a complete, valid pipeline config that can be written
to disk and edited by the user.  Templates are selected by name via
:func:`get_template` and enumerated via :func:`list_templates`.
"""

from __future__ import annotations

from pathlib import Path

from rheojax.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
# Template library
# ------------------------------------------------------------------

TEMPLATES: dict[str, dict[str, str]] = {
    "basic": {
        "description": "Basic NLSQ fitting workflow",
        "yaml": """\
version: "1"
name: "Basic NLSQ Fit"

defaults:
  test_mode: relaxation

steps:
  - type: load
    file: data.csv
    x_col: time
    y_col: G_t
    format: auto

  - type: fit
    model: maxwell
    method: auto
    max_iter: 5000

  - type: export
    output: results/
    format: directory
""",
    },
    "bayesian": {
        "description": "NLSQ warm-start followed by Bayesian NUTS inference",
        "yaml": """\
version: "1"
name: "NLSQ + Bayesian Inference"

defaults:
  test_mode: relaxation

steps:
  - type: load
    file: data.csv
    x_col: time
    y_col: G_t
    format: auto

  - type: fit
    model: maxwell
    method: auto
    max_iter: 5000

  - type: bayesian
    num_warmup: 1000
    num_samples: 2000
    num_chains: 4
    seed: 42
    warm_start: true

  - type: export
    output: results/
    format: directory
""",
    },
    "oscillation": {
        "description": "SAOS oscillation fitting (G' and G'')",
        "yaml": """\
version: "1"
name: "SAOS Oscillation Fit"

defaults:
  test_mode: oscillation

steps:
  - type: load
    file: saos.csv
    x_col: omega
    y_cols:
      - G_prime
      - G_double_prime
    format: auto

  - type: fit
    model: maxwell
    method: auto
    max_iter: 5000
    test_mode: oscillation

  - type: export
    output: results/
    format: directory
""",
    },
    "mastercurve": {
        "description": "Time-temperature superposition (master curve) analysis",
        "yaml": """\
version: "1"
name: "Master Curve (TTS)"

defaults:
  test_mode: oscillation

steps:
  - type: load
    file: tts_data.csv
    x_col: omega
    y_col: G_star
    format: auto

  - type: transform
    name: mastercurve
    reference_temperature: 25.0
    fit_wlf: true

  - type: fit
    model: generalized_maxwell
    method: auto
    max_iter: 5000
    test_mode: oscillation

  - type: export
    output: results/mastercurve/
    format: directory
""",
    },
    "batch": {
        "description": "Batch processing multiple files from a directory",
        "yaml": """\
version: "1"
name: "Batch NLSQ Processing"

# Note: this template processes a single file.
# To process multiple files with glob patterns (e.g. data/*.csv),
# use the 'rheojax batch' command instead of 'rheojax run'.

defaults:
  test_mode: relaxation

steps:
  - type: load
    file: data/sample.csv
    x_col: time
    y_col: G_t
    format: auto

  - type: fit
    model: maxwell
    method: auto
    max_iter: 5000

  - type: export
    output: results/batch/
    format: directory
""",
    },
    "creep": {
        "description": "Creep and creep recovery analysis",
        "yaml": """\
version: "1"
name: "Creep Analysis"

defaults:
  test_mode: creep

steps:
  - type: load
    file: creep.csv
    x_col: time
    y_col: gamma
    format: auto

  - type: fit
    model: maxwell
    method: auto
    max_iter: 5000
    test_mode: creep

  - type: bayesian
    num_warmup: 500
    num_samples: 1000
    num_chains: 4
    seed: 0
    warm_start: true

  - type: export
    output: results/creep/
    format: directory
""",
    },
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def get_template(name: str) -> str:
    """Return the YAML string for a named template.

    Args:
        name: Template identifier (e.g. ``"basic"``, ``"bayesian"``).

    Returns:
        YAML string ready to write to a file.

    Raises:
        KeyError: If *name* is not a registered template.

    Example:
        >>> yaml_str = get_template("basic")
    """
    if name not in TEMPLATES:
        available = sorted(TEMPLATES.keys())
        raise KeyError(f"Template '{name}' not found.  Available: {available}")
    logger.debug("Retrieved template", name=name)
    return TEMPLATES[name]["yaml"]


def list_templates() -> list[dict[str, str]]:
    """Return metadata for all available templates.

    Returns:
        List of dicts, each with ``"name"`` and ``"description"`` keys,
        sorted alphabetically by name.

    Example:
        >>> for t in list_templates():
        ...     print(t["name"], "-", t["description"])
    """
    return [
        {"name": name, "description": info["description"]}
        for name, info in sorted(TEMPLATES.items())
    ]


def write_template(name: str, output_path: str | Path) -> None:
    """Write a named template YAML to *output_path*.

    Args:
        name: Template identifier.
        output_path: Destination file path.  Parent directories are created
            if they do not exist.

    Raises:
        KeyError: If *name* is not a registered template.
        OSError: If the file cannot be written.

    Example:
        >>> write_template("bayesian", "my_pipeline.yaml")
    """
    yaml_str = get_template(name)  # raises KeyError if unknown
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml_str, encoding="utf-8")
    logger.info("Template written", name=name, path=str(dest))
