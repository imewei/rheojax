"""Tests for modulus conversion utilities (DMTA/DMA support).

Tests the E* <-> G* conversion for all deformation modes and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import DeformationMode
from rheojax.utils.modulus_conversion import (
    POISSON_PRESETS,
    _conversion_factor,
    _validate_poisson_ratio,
    convert_modulus,
    convert_rheodata,
)

jax, jnp = safe_import_jax()


# ── Poisson's ratio validation ──────────────────────────────────────────


@pytest.mark.smoke
def test_validate_poisson_ratio_valid():
    """Valid Poisson's ratios should not raise."""
    _validate_poisson_ratio(0.5)
    _validate_poisson_ratio(0.0)
    _validate_poisson_ratio(0.35)
    _validate_poisson_ratio(-0.5)


@pytest.mark.smoke
def test_validate_poisson_ratio_invalid_high():
    """Poisson's ratio > 0.5 should raise ValueError."""
    with pytest.raises(ValueError, match="must be in"):
        _validate_poisson_ratio(0.51)


@pytest.mark.smoke
def test_validate_poisson_ratio_invalid_low():
    """Poisson's ratio <= -1 should raise ValueError."""
    with pytest.raises(ValueError, match="must be in"):
        _validate_poisson_ratio(-1.0)


@pytest.mark.smoke
def test_validate_poisson_ratio_type_error():
    """Non-numeric Poisson's ratio should raise TypeError."""
    with pytest.raises(TypeError, match="must be a number"):
        _validate_poisson_ratio("0.5")  # type: ignore[arg-type]


# ── Conversion factor ───────────────────────────────────────────────────


@pytest.mark.smoke
def test_conversion_factor_rubber():
    """For rubber (v=0.5), factor should be 3.0."""
    assert _conversion_factor(0.5) == pytest.approx(3.0)


@pytest.mark.smoke
def test_conversion_factor_glassy():
    """For glassy polymer (v=0.35), factor should be 2.7."""
    assert _conversion_factor(0.35) == pytest.approx(2.7)


# ── Array-level conversion ──────────────────────────────────────────────


@pytest.mark.smoke
def test_convert_modulus_tension_to_shear_real():
    """E -> G conversion with real arrays and rubber Poisson's ratio."""
    E_prime = np.array([3e9, 6e9, 9e9])
    G_prime = convert_modulus(E_prime, "tension", "shear", poisson_ratio=0.5)
    expected = E_prime / 3.0
    np.testing.assert_allclose(G_prime, expected, rtol=1e-10)


@pytest.mark.smoke
def test_convert_modulus_shear_to_tension_complex():
    """G* -> E* conversion with complex arrays."""
    G_star = np.array([1e6 + 1e5j, 2e6 + 2e5j])
    E_star = convert_modulus(G_star, "shear", "tension", poisson_ratio=0.5)
    expected = G_star * 3.0
    np.testing.assert_allclose(E_star, expected, rtol=1e-10)


@pytest.mark.smoke
def test_convert_modulus_roundtrip():
    """E -> G -> E roundtrip should recover original data."""
    E_original = np.array([1e8 + 5e6j, 3e8 + 1e7j, 5e8 + 2e7j])
    nu = 0.35
    G = convert_modulus(E_original, "tension", "shear", poisson_ratio=nu)
    E_recovered = convert_modulus(G, "shear", "tension", poisson_ratio=nu)
    np.testing.assert_allclose(E_recovered, E_original, rtol=1e-10)


@pytest.mark.smoke
def test_convert_modulus_same_mode_noop():
    """Converting between same modes should return data unchanged."""
    data = np.array([1.0, 2.0, 3.0])
    result = convert_modulus(data, "shear", "shear")
    np.testing.assert_array_equal(result, data)


@pytest.mark.smoke
def test_convert_modulus_tensile_to_tensile_noop():
    """Converting between two tensile modes should be a no-op."""
    data = np.array([1.0, 2.0, 3.0])
    result = convert_modulus(data, "tension", "bending")
    np.testing.assert_array_equal(result, data)


@pytest.mark.smoke
def test_convert_modulus_jax_array():
    """Conversion should work with JAX arrays."""
    E_star = jnp.array([3e9, 6e9])
    G_star = convert_modulus(E_star, "tension", "shear", poisson_ratio=0.5)
    expected = E_star / 3.0
    np.testing.assert_allclose(np.array(G_star), np.array(expected), rtol=1e-10)


@pytest.mark.smoke
def test_convert_modulus_enum_input():
    """Conversion should accept DeformationMode enum values."""
    data = np.array([3e9])
    result = convert_modulus(
        data, DeformationMode.TENSION, DeformationMode.SHEAR, poisson_ratio=0.5
    )
    np.testing.assert_allclose(result, data / 3.0, rtol=1e-10)


# ── RheoData-level conversion ───────────────────────────────────────────


@pytest.mark.smoke
def test_convert_rheodata_tension_to_shear():
    """Convert RheoData from tension to shear."""
    from rheojax.core.data import RheoData

    omega = np.logspace(-2, 2, 50)
    E_star = 3e6 * (1j * omega * 0.1) / (1 + 1j * omega * 0.1)

    rheo = RheoData(
        x=omega,
        y=E_star,
        domain="frequency",
        x_units="rad/s",
        y_units="E* (Pa)",
        metadata={"deformation_mode": "tension"},
        validate=False,
    )

    shear_data = convert_rheodata(rheo, "shear", poisson_ratio=0.5)

    # Check y-data converted
    np.testing.assert_allclose(
        np.array(shear_data.y), np.array(E_star / 3.0), rtol=1e-10
    )
    # Check metadata updated
    assert shear_data.metadata["deformation_mode"] == "shear"
    assert shear_data.metadata["poisson_ratio"] == 0.5
    assert shear_data.metadata["converted_from"] == "tension"
    # Check units relabeled
    assert "G" in shear_data.y_units


@pytest.mark.smoke
def test_convert_rheodata_units_preserve_si_prefix():
    """Unit relabeling should NOT corrupt SI prefixes like G in GPa."""
    from rheojax.core.data import RheoData

    omega = np.logspace(-2, 2, 10)
    G_star = 1e9 * (1j * omega * 0.1) / (1 + 1j * omega * 0.1)

    rheo = RheoData(
        x=omega,
        y=G_star,
        domain="frequency",
        x_units="rad/s",
        y_units="G* (GPa)",
        metadata={"deformation_mode": "shear"},
        validate=False,
    )

    tensile_data = convert_rheodata(rheo, "tension", poisson_ratio=0.5)
    # Should become "E* (GPa)", NOT "E* (EPa)"
    assert "E*" in tensile_data.y_units
    assert "GPa" in tensile_data.y_units
    assert "EPa" not in tensile_data.y_units


# ── Presets ──────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_poisson_presets_valid():
    """All preset Poisson's ratios should be physically valid."""
    for material, nu in POISSON_PRESETS.items():
        _validate_poisson_ratio(nu)
        assert -1.0 < nu <= 0.5, f"Invalid preset for {material}: {nu}"
