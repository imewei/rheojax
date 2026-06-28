"""Regression tests for removed DMTA-specific CLI and YAML options."""

from __future__ import annotations

import pytest

from rheojax.cli._yaml_schema import PipelineConfig, validate_config


@pytest.mark.parametrize(
    "parser_factory, positional",
    [
        ("rheojax.cli.fit:create_parser", "data.csv"),
        ("rheojax.cli.bayesian:create_parser", "data.csv"),
        ("rheojax.cli.cmd_batch:create_parser", "*.csv"),
    ],
)
@pytest.mark.parametrize(
    "removed_flag, value",
    [("--deformation-mode", "tension"), ("--poisson-ratio", "0.5")],
)
def test_removed_cli_flag_is_rejected(
    parser_factory: str,
    positional: str,
    removed_flag: str,
    value: str,
) -> None:
    module_name, factory_name = parser_factory.split(":")
    module = __import__(module_name, fromlist=[factory_name])
    parser = getattr(module, factory_name)()

    with pytest.raises(SystemExit):
        parser.parse_args([positional, "--model", "maxwell", removed_flag, value])


@pytest.mark.parametrize("removed_key", ["deformation_mode", "poisson_ratio"])
def test_yaml_validator_rejects_removed_step_key(removed_key: str) -> None:
    config = PipelineConfig(
        version="1",
        name="Removed key",
        steps=[{"type": "load", "file": "data.csv", removed_key: "tension"}],
    )

    errors = validate_config(config)

    assert any(removed_key in error for error in errors)
