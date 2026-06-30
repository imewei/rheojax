"""Tests for lightweight shared option validation."""

from types import MappingProxyType

import pytest

from rheojax.core._validation import reject_removed_options


@pytest.mark.parametrize(
    "removed_key",
    ("deformation_mode", "poisson_ratio"),
)
def test_reject_removed_options_names_single_offending_option(removed_key):
    with pytest.raises(TypeError) as exc_info:
        reject_removed_options({removed_key: object()})

    message = str(exc_info.value)
    assert f"Removed option(s) '{removed_key}'" in message
    assert "shear-only" in message
    assert "Remove them." in message


def test_reject_removed_options_orders_multiple_offending_options():
    options = {"poisson_ratio": 0.5, "deformation_mode": "tension"}

    with pytest.raises(TypeError) as exc_info:
        reject_removed_options(options)

    message = str(exc_info.value)
    assert "'deformation_mode'" in message
    assert "'poisson_ratio'" in message
    assert "shear-only" in message
    assert "Remove them." in message


def test_reject_removed_options_accepts_mapping_without_mutating_it():
    options = {"method": "scipy", "max_iter": 23}
    readonly_options = MappingProxyType(options)

    reject_removed_options(readonly_options)

    assert options == {"method": "scipy", "max_iter": 23}
