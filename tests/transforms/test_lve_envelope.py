"""Tests for LVEEnvelope transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.lve_envelope import LVEEnvelope, lve_envelope


class TestLVEEnvelopeAnalytical:
    """Test analytical LVE envelope computation."""

    def test_single_mode_short_time(self):
        """At short times, σ_LVE⁺(t) ≈ γ̇₀ G₀ t (linear growth)."""
        G_i = np.array([1000.0])
        tau_i = np.array([10.0])
        t = np.array([0.001, 0.01])
        shear_rate = 1.0

        sigma = lve_envelope(t, G_i, tau_i, 0.0, shear_rate)
        expected = shear_rate * G_i[0] * t  # linear at short times
        np.testing.assert_allclose(sigma, expected, rtol=0.01)

    def test_single_mode_long_time(self):
        """At long times, σ_LVE⁺(t) → γ̇₀ G₀ τ (steady state viscosity)."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        t = np.array([100.0])
        shear_rate = 1.0

        sigma = lve_envelope(t, G_i, tau_i, 0.0, shear_rate)
        expected = shear_rate * G_i[0] * tau_i[0]  # η₀ = G₀τ
        np.testing.assert_allclose(sigma, expected, rtol=1e-5)

    def test_equilibrium_modulus_contribution(self):
        """G_e adds a linear growth: G_e * t."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        G_e = 100.0
        t = np.array([10.0])
        shear_rate = 1.0

        sigma = lve_envelope(t, G_i, tau_i, G_e, shear_rate)
        sigma_no_ge = lve_envelope(t, G_i, tau_i, 0.0, shear_rate)
        assert sigma[0] > sigma_no_ge[0]
        np.testing.assert_allclose(
            sigma[0] - sigma_no_ge[0], shear_rate * G_e * t[0], rtol=1e-10
        )

    def test_shear_rate_scaling(self):
        """σ scales linearly with γ̇₀."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        t = np.logspace(-2, 2, 50)

        sigma_1 = lve_envelope(t, G_i, tau_i, 0.0, 1.0)
        sigma_10 = lve_envelope(t, G_i, tau_i, 0.0, 10.0)
        np.testing.assert_allclose(sigma_10, 10.0 * sigma_1, rtol=1e-10)


@pytest.mark.smoke
class TestLVEEnvelopeTransform:
    """Test LVEEnvelope as a BaseTransform."""

    def test_basic_transform(self):
        """Basic transform with explicit Prony parameters."""
        G_i = np.array([500.0, 300.0])
        tau_i = np.array([0.1, 10.0])
        transform = LVEEnvelope(shear_rate=1.0, G_i=G_i, tau_i=tau_i)
        result, meta = transform.transform(None)

        assert result.x is not None
        assert result.y is not None
        assert len(result.y) > 0
        assert np.all(result.y >= 0)
        assert "lve_result" in meta

    def test_params_from_metadata(self):
        """Read Prony params from data metadata."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        data = RheoData(
            x=np.logspace(-2, 2, 50),
            y=np.zeros(50),
            metadata={"G_i": G_i, "tau_i": tau_i},
        )
        transform = LVEEnvelope(shear_rate=0.5)
        result, _ = transform.transform(data)
        assert result.y is not None

    def test_missing_params_raises(self):
        """Should raise ValueError when no Prony params available."""
        transform = LVEEnvelope(shear_rate=1.0)
        with pytest.raises(ValueError, match="G_i and tau_i must be provided"):
            transform.transform(None)

    def test_mismatched_lengths_raises(self):
        """G_i and tau_i must have same length."""
        transform = LVEEnvelope(
            shear_rate=1.0,
            G_i=np.array([1000.0, 500.0]),
            tau_i=np.array([1.0]),
        )
        with pytest.raises(ValueError, match="same length"):
            transform.transform(None)
