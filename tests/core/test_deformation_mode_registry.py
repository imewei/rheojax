"""Tests for deformation mode classification in the model registry.

Validates that all models are correctly classified by supported deformation modes.
"""

from __future__ import annotations

import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.core.test_modes import DeformationMode

jax, jnp = safe_import_jax()

# Trigger registration of all models (lazy imports require explicit loading)
import rheojax.models as _rjm  # noqa: E402

for _name in list(_rjm._LAZY_IMPORTS):
    getattr(_rjm, _name)


# ── Classification correctness ───────────────────────────────────────────


@pytest.mark.smoke
def test_all_models_have_deformation_modes():
    """Every registered model should have at least one deformation mode."""
    # Exclude test fixture models registered in test files
    test_models = {"simple_test_model"}
    all_models = ModelRegistry.list_models()
    for name in all_models:
        if name in test_models:
            continue
        info = ModelRegistry.get_info(name)
        modes = getattr(info, "deformation_modes", [])
        assert len(modes) > 0, f"Model '{name}' has no deformation_modes"


@pytest.mark.smoke
def test_flow_models_shear_only():
    """Flow curve models should be SHEAR-ONLY."""
    shear_only_models = [
        "bingham",
        "carreau",
        "carreau_yasuda",
        "cross",
        "herschel_bulkley",
        "power_law",
    ]
    for name in shear_only_models:
        info = ModelRegistry.get_info(name)
        modes = getattr(info, "deformation_modes", [])
        assert modes == [
            DeformationMode.SHEAR
        ], f"Flow model '{name}' should be SHEAR-only, got {modes}"


@pytest.mark.smoke
def test_classical_models_dmta_compatible():
    """Classical models should support all 4 deformation modes."""
    dmta_models = ["maxwell", "zener", "springpot"]
    expected_modes = {
        DeformationMode.SHEAR,
        DeformationMode.TENSION,
        DeformationMode.BENDING,
        DeformationMode.COMPRESSION,
    }
    for name in dmta_models:
        info = ModelRegistry.get_info(name)
        modes = set(getattr(info, "deformation_modes", []))
        assert (
            modes == expected_modes
        ), f"Classical model '{name}' should have all 4 modes, got {modes}"


@pytest.mark.smoke
def test_nonlocal_models_shear_only():
    """Nonlocal PDE models (no oscillation) should be SHEAR-only."""
    shear_only_nonlocal = [
        "dmt_nonlocal",
        "vlb_nonlocal",
        "fluidity_saramito_nonlocal",
    ]
    for name in shear_only_nonlocal:
        if name in ModelRegistry.list_models():
            info = ModelRegistry.get_info(name)
            modes = getattr(info, "deformation_modes", [])
            assert modes == [
                DeformationMode.SHEAR
            ], f"Nonlocal model '{name}' should be SHEAR-only, got {modes}"


@pytest.mark.smoke
def test_spp_shear_only():
    """SPP yield stress decomposition is inherently shear-only."""
    info = ModelRegistry.get_info("spp_yield_stress")
    modes = getattr(info, "deformation_modes", [])
    assert modes == [DeformationMode.SHEAR]


# ── find_compatible queries ──────────────────────────────────────────────


@pytest.mark.smoke
def test_find_compatible_tension():
    """ModelRegistry.find(deformation_mode='tension') returns DMTA models."""
    from rheojax.core.inventory import Protocol

    results = ModelRegistry.find(
        protocol=Protocol.OSCILLATION,
        deformation_mode=DeformationMode.TENSION,
    )
    # find() returns list of model name strings
    result_names = set(results)
    assert "maxwell" in result_names
    assert "zener" in result_names
    assert "fractional_zener_ss" in result_names
    assert "generalized_maxwell" in result_names
    # Should NOT include flow-only or nonlocal models
    assert "bingham" not in result_names
    assert "power_law" not in result_names
    assert "spp_yield_stress" not in result_names


@pytest.mark.smoke
def test_find_compatible_shear_returns_all():
    """ModelRegistry.find(deformation_mode='shear') returns all models with shear."""
    results = ModelRegistry.find(deformation_mode=DeformationMode.SHEAR)
    result_names = set(results)
    # Every model should support shear
    assert "maxwell" in result_names
    assert "bingham" in result_names
    assert "spp_yield_stress" in result_names
