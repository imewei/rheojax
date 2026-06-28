"""Regression tests for removed DMTA-specific CLI and YAML options."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from rheojax.cli._yaml_runner import apply_overrides, config_to_builder
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


@pytest.mark.parametrize("removed_key", ["deformation_mode", "poisson_ratio"])
def test_yaml_validator_rejects_removed_default_key(removed_key: str) -> None:
    config = PipelineConfig(
        version="1",
        name="Removed default",
        defaults={removed_key: "tension"},
        steps=[{"type": "load", "file": "data.csv"}],
    )

    errors = validate_config(config)

    assert any("defaults" in error.lower() and removed_key in error for error in errors)


@pytest.mark.parametrize("removed_key", ["deformation_mode", "poisson_ratio"])
def test_yaml_validator_rejects_removed_default_from_override(removed_key: str) -> None:
    config = PipelineConfig(
        version="1",
        name="Removed override",
        steps=[{"type": "load", "file": "data.csv"}],
    )
    overridden = apply_overrides(config, [f"defaults.{removed_key}=tension"])

    errors = validate_config(overridden)

    assert any("defaults" in error.lower() and removed_key in error for error in errors)


@pytest.mark.parametrize(
    "step, passthrough_keys",
    [
        (
            {
                "type": "fit",
                "model": "maxwell",
                "ftol": 1e-8,
                "xtol": 1e-8,
            },
            ("ftol", "xtol"),
        ),
        (
            {"type": "bayesian", "custom_priors": {"G": "lognormal"}},
            ("custom_priors",),
        ),
    ],
)
def test_yaml_validator_accepts_supported_passthrough_keys(
    step: dict,
    passthrough_keys: tuple[str, ...],
) -> None:
    steps = [{"type": "load", "file": "data.csv"}]
    if step["type"] == "bayesian":
        steps.append({"type": "fit", "model": "maxwell"})
    steps.append(step)
    config = PipelineConfig(version="1", name="Passthrough", steps=steps)

    errors = validate_config(config)
    builder = config_to_builder(config)
    builder_kwargs = next(
        kwargs for step_type, kwargs in builder.steps if step_type == step["type"]
    )

    assert not any(
        key in error for key in passthrough_keys for error in errors
    )
    assert all(builder_kwargs[key] == step[key] for key in passthrough_keys)


def test_yaml_validator_warns_but_accepts_unrelated_unknown_key() -> None:
    config = PipelineConfig(
        version="1",
        name="Unknown key",
        steps=[
            {"type": "load", "file": "data.csv"},
            {"type": "fit", "model": "maxwell", "future_fit_option": True},
        ],
    )

    with patch("rheojax.cli._yaml_schema.logger.warning") as warning:
        errors = validate_config(config)

    assert not any("future_fit_option" in error for error in errors)
    warning.assert_called_once()
    assert warning.call_args.args[0] == "Unknown keys in pipeline step"
    assert warning.call_args.kwargs["unknown"] == ["future_fit_option"]
