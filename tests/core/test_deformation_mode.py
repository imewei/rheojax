"""Tests for deformation mode integration in BaseModel and RheoData.

Tests the E* <-> G* conversion at the BaseModel boundary for DMTA support.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import DeformationMode

jax, jnp = safe_import_jax()


# ── DeformationMode enum ────────────────────────────────────────────────


@pytest.mark.smoke
def test_deformation_mode_values():
    """DeformationMode enum has correct values."""
    assert DeformationMode.SHEAR.value == "shear"
    assert DeformationMode.TENSION.value == "tension"
    assert DeformationMode.BENDING.value == "bending"
    assert DeformationMode.COMPRESSION.value == "compression"


@pytest.mark.smoke
def test_deformation_mode_is_tensile():
    """is_tensile() returns True for tension/bending/compression."""
    assert not DeformationMode.SHEAR.is_tensile()
    assert DeformationMode.TENSION.is_tensile()
    assert DeformationMode.BENDING.is_tensile()
    assert DeformationMode.COMPRESSION.is_tensile()


@pytest.mark.smoke
def test_deformation_mode_from_string():
    """DeformationMode can be created from string."""
    assert DeformationMode("shear") == DeformationMode.SHEAR
    assert DeformationMode("tension") == DeformationMode.TENSION


# ── RheoData deformation_mode property ──────────────────────────────────


@pytest.mark.smoke
def test_rheodata_default_shear():
    """RheoData.deformation_mode defaults to 'shear'."""
    from rheojax.core.data import RheoData

    data = RheoData(x=np.array([1.0, 2.0]), y=np.array([3.0, 4.0]), validate=False)
    assert data.deformation_mode == "shear"
    assert data.storage_modulus_label == "G'"


@pytest.mark.smoke
def test_rheodata_tension_labels():
    """RheoData with tension mode uses E' labels."""
    from rheojax.core.data import RheoData

    data = RheoData(
        x=np.array([1.0, 2.0]),
        y=np.array([3.0, 4.0]),
        metadata={"deformation_mode": "tension"},
        validate=False,
    )
    assert data.deformation_mode == "tension"
    assert data.storage_modulus_label == "E'"
    assert data.loss_modulus_label == 'E"'


# ── BaseModel fit/predict with deformation_mode ─────────────────────────


@pytest.mark.smoke
def test_maxwell_fit_with_tension():
    """Maxwell fit with deformation_mode='tension' converts E* to G*."""
    from rheojax.models.classical.maxwell import Maxwell

    # Generate synthetic G* data, then convert to E*
    omega = np.logspace(-1, 2, 50)
    G0 = 1e6
    eta = 1e4
    tau = eta / G0
    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)
    G_star = G_prime + 1j * G_double_prime

    # Convert to E* with nu=0.5 (rubber): E = 3G
    E_star = G_star * 3.0

    # Fit in tension mode — should internally convert back to G*
    model = Maxwell()
    model.fit(
        omega,
        E_star,
        test_mode="oscillation",
        deformation_mode="tension",
        poisson_ratio=0.5,
    )

    # Fitted parameters should be in G-space (not E-space)
    fitted_G0 = model.parameters.get_value("G0")
    fitted_eta = model.parameters.get_value("eta")
    # G0 should be ~1e6, not ~3e6
    assert fitted_G0 == pytest.approx(G0, rel=0.1)
    assert fitted_eta == pytest.approx(eta, rel=0.1)


@pytest.mark.smoke
def test_predict_returns_tensile_after_tensile_fit():
    """After fitting in tension mode, predict returns E* (not G*)."""
    from rheojax.models.classical.maxwell import Maxwell

    omega = np.logspace(-1, 2, 50)
    G0 = 1e6
    eta = 1e4
    tau = eta / G0
    omega_tau = omega * tau
    G_star = G0 * (
        omega_tau**2 / (1 + omega_tau**2) + 1j * omega_tau / (1 + omega_tau**2)
    )
    E_star = G_star * 3.0  # nu=0.5

    model = Maxwell()
    model.fit(
        omega,
        E_star,
        test_mode="oscillation",
        deformation_mode="tension",
        poisson_ratio=0.5,
    )

    # Predict should return E*-space result
    pred = model.predict(omega, test_mode="oscillation")
    # The predicted values should be in E*-space (factor of 3)
    pred_G = model._predict(omega)
    np.testing.assert_allclose(np.abs(pred), np.abs(pred_G) * 3.0, rtol=1e-6)


@pytest.mark.smoke
def test_predict_shear_override():
    """predict(deformation_mode='shear') overrides stored tensile mode."""
    from rheojax.models.classical.maxwell import Maxwell

    omega = np.logspace(-1, 2, 50)
    G0 = 1e6
    eta = 1e4
    G_star = G0 * (1j * omega * eta / G0) / (1 + 1j * omega * eta / G0)
    E_star = G_star * 3.0

    model = Maxwell()
    model.fit(
        omega,
        E_star,
        test_mode="oscillation",
        deformation_mode="tension",
        poisson_ratio=0.5,
    )

    # Override to get G* directly
    pred_G = model.predict(omega, test_mode="oscillation", deformation_mode="shear")
    pred_internal = model._predict(omega)
    np.testing.assert_allclose(np.abs(pred_G), np.abs(pred_internal), rtol=1e-6)


@pytest.mark.smoke
def test_fit_without_deformation_mode_is_shear():
    """Default fit (no deformation_mode) works as pure shear."""
    from rheojax.models.classical.maxwell import Maxwell

    omega = np.logspace(-1, 2, 50)
    G0 = 1e6
    eta = 1e4
    tau = eta / G0
    omega_tau = omega * tau
    G_star = G0 * (
        omega_tau**2 / (1 + omega_tau**2) + 1j * omega_tau / (1 + omega_tau**2)
    )

    model = Maxwell()
    model.fit(omega, G_star, test_mode="oscillation")

    # No conversion should happen
    assert model._deformation_mode is None
    fitted_G0 = model.parameters.get_value("G0")
    assert fitted_G0 == pytest.approx(G0, rel=0.1)


@pytest.mark.smoke
def test_rheodata_autodetect_deformation_mode():
    """BaseModel.fit() auto-detects deformation_mode from RheoData metadata."""
    from rheojax.core.data import RheoData
    from rheojax.models.classical.maxwell import Maxwell

    omega = np.logspace(-1, 2, 50)
    G0 = 1e6
    eta = 1e4
    tau = eta / G0
    omega_tau = omega * tau
    G_star = G0 * (
        omega_tau**2 / (1 + omega_tau**2) + 1j * omega_tau / (1 + omega_tau**2)
    )
    E_star = G_star * 3.0

    rheo = RheoData(
        x=omega,
        y=E_star,
        domain="frequency",
        metadata={"deformation_mode": "tension", "test_mode": "oscillation"},
        validate=False,
    )

    model = Maxwell()
    model.fit(rheo, E_star, test_mode="oscillation")

    # Should auto-detect tension mode from RheoData metadata
    assert model._deformation_mode == DeformationMode.TENSION
    fitted_G0 = model.parameters.get_value("G0")
    assert fitted_G0 == pytest.approx(G0, rel=0.1)


@pytest.mark.smoke
def test_maxwell_tension_params_equal_3x_shear():
    """For nu=0.5, fitting E* should give G-space params = E-params/3."""
    from rheojax.models.classical.maxwell import Maxwell

    omega = np.logspace(-1, 2, 100)
    G0 = 5e5
    eta = 2e3
    tau = eta / G0
    omega_tau = omega * tau
    G_star = G0 * (
        omega_tau**2 / (1 + omega_tau**2) + 1j * omega_tau / (1 + omega_tau**2)
    )

    # Fit in G-space
    model_shear = Maxwell()
    model_shear.fit(omega, G_star, test_mode="oscillation")
    G0_shear = model_shear.parameters.get_value("G0")

    # Fit same data as E* (3x) in tension mode
    E_star = G_star * 3.0
    model_tension = Maxwell()
    model_tension.fit(
        omega,
        E_star,
        test_mode="oscillation",
        deformation_mode="tension",
        poisson_ratio=0.5,
    )
    G0_tension = model_tension.parameters.get_value("G0")

    # Both should give same G-space parameters
    assert G0_tension == pytest.approx(G0_shear, rel=0.01)
