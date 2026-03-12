"""Tests for unified unit normalization (Phase 1)."""

import numpy as np
import pytest

from rheojax.io.readers._utils import (
    UNIFIED_UNIT_CONVERSIONS,
    normalize_temperature,
    normalize_units,
)


@pytest.mark.smoke
def test_celsius_to_kelvin():
    assert normalize_temperature(0, "C") == pytest.approx(273.15)
    assert normalize_temperature(100, "°C") == pytest.approx(373.15)


@pytest.mark.smoke
def test_fahrenheit_to_kelvin():
    assert normalize_temperature(32, "F") == pytest.approx(273.15)
    assert normalize_temperature(212, "°F") == pytest.approx(373.15)


@pytest.mark.smoke
def test_kelvin_passthrough():
    assert normalize_temperature(300, "K") == pytest.approx(300.0)


@pytest.mark.smoke
def test_invalid_unit():
    with pytest.raises(ValueError):
        normalize_temperature(0, "X")


@pytest.mark.smoke
def test_normalize_units_hz():
    values = np.array([1.0])
    result, unit = normalize_units(values, "Hz")
    np.testing.assert_allclose(result, np.array([2 * np.pi]))
    assert unit == "rad/s"


@pytest.mark.smoke
def test_normalize_units_kpa():
    values = np.array([1.0, 2.0])
    result, unit = normalize_units(values, "kPa")
    np.testing.assert_allclose(result, np.array([1000.0, 2000.0]))
    assert unit == "Pa"


@pytest.mark.smoke
def test_normalize_units_gpa():
    values = np.array([1.0])
    result, unit = normalize_units(values, "GPa")
    np.testing.assert_allclose(result, np.array([1e9]))
    assert unit == "Pa"


@pytest.mark.smoke
def test_normalize_units_percent():
    values = np.array([50.0])
    result, unit = normalize_units(values, "%")
    np.testing.assert_allclose(result, np.array([0.5]))
    assert unit == "dimensionless"


@pytest.mark.smoke
def test_normalize_units_celsius_array():
    values = np.array([0.0, 100.0])
    result, unit = normalize_units(values, "°C")
    np.testing.assert_allclose(result, np.array([273.15, 373.15]))
    assert unit == "K"


@pytest.mark.smoke
def test_normalize_units_fahrenheit_array():
    values = np.array([32.0, 212.0])
    result, unit = normalize_units(values, "°F")
    np.testing.assert_allclose(result, np.array([273.15, 373.15]))
    assert unit == "K"


@pytest.mark.smoke
def test_normalize_units_passthrough():
    values = np.array([1.0])
    result, unit = normalize_units(values, "Pa")
    np.testing.assert_allclose(result, values)
    assert unit == "Pa"


@pytest.mark.smoke
def test_unified_dict_has_expected_keys():
    expected_keys = {
        "hz",
        "1/hz",
        "ms",
        "min",
        "mins",
        "minutes",
        "kpa",
        "mpa",
        "gpa",
        "mpa·s",
        "mpa.s",
        "rpm",
        "rev/min",
        "rev/s",
        "%",
        "°c",
        "°f",
        "c",
        "f",
    }
    assert expected_keys.issubset(UNIFIED_UNIT_CONVERSIONS.keys())
