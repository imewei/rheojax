"""Tests for PronyConversion transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.transforms.prony_conversion import (
    PronyConversion,
    _prony_to_frequency,
    _prony_to_time,
)


class TestPronyConversionAnalytical:
    """Test analytical Prony functions with known solutions."""

    def test_single_mode_time(self):
        """Single Maxwell mode: G(t) = G_0 exp(-t/tau)."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        t = np.logspace(-2, 2, 100)
        G_t = _prony_to_time(G_i, tau_i, 0.0, t)
        expected = 1000.0 * np.exp(-t)
        np.testing.assert_allclose(G_t, expected, rtol=1e-10)

    def test_single_mode_frequency(self):
        """G'(ω) = G_0 ω²τ²/(1+ω²τ²), G''(ω) = G_0 ωτ/(1+ω²τ²)."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        omega = np.logspace(-2, 2, 100)
        G_p, G_pp = _prony_to_frequency(G_i, tau_i, 0.0, omega)

        wt2 = omega**2
        expected_Gp = 1000.0 * wt2 / (1.0 + wt2)
        expected_Gpp = 1000.0 * omega / (1.0 + wt2)

        np.testing.assert_allclose(G_p, expected_Gp, rtol=1e-10)
        np.testing.assert_allclose(G_pp, expected_Gpp, rtol=1e-10)

    def test_equilibrium_modulus(self):
        """G_e shifts G'(ω) at low frequencies."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        G_e = 100.0
        omega = np.array([0.001])
        G_p, _ = _prony_to_frequency(G_i, tau_i, G_e, omega)
        assert G_p[0] > 100.0  # Should be G_e + small contribution


@pytest.mark.smoke
class TestPronyConversionTransform:
    """Test PronyConversion as a BaseTransform."""

    def test_time_to_freq(self):
        """Round-trip: generate G(t) from known Prony, convert to freq."""
        G_i = np.array([500.0, 300.0])
        tau_i = np.array([0.1, 10.0])
        t = np.logspace(-3, 3, 200)
        G_t = _prony_to_time(G_i, tau_i, 0.0, t)

        data = RheoData(x=t, y=G_t, metadata={"test_mode": "relaxation"})
        transform = PronyConversion(direction="time_to_freq", n_modes=5)
        result, meta = transform.transform(data)

        assert result.x is not None
        assert result.y is not None
        assert "prony_result" in meta

    def test_freq_to_time(self):
        """Convert oscillation data to relaxation."""
        omega = np.logspace(-2, 2, 100)
        G_i = np.array([500.0, 300.0])
        tau_i = np.array([0.1, 10.0])
        G_p, G_pp = _prony_to_frequency(G_i, tau_i, 0.0, omega)
        G_star = G_p + 1j * G_pp  # Note: convention G* = G' + iG''

        data = RheoData(x=omega, y=G_star, metadata={"test_mode": "oscillation"})
        transform = PronyConversion(direction="freq_to_time", n_modes=5)
        result, meta = transform.transform(data)

        assert result.x is not None
        assert result.y is not None
        assert result.metadata["test_mode"] == "relaxation"

    def test_round_trip_fidelity(self):
        """G(t) → G'(ω) → G(t) should approximately recover original."""
        G_i = np.array([1000.0])
        tau_i = np.array([1.0])
        t_orig = np.logspace(-2, 2, 200)
        G_t_orig = _prony_to_time(G_i, tau_i, 0.0, t_orig)

        # Forward: time → freq
        data1 = RheoData(x=t_orig, y=G_t_orig)
        fwd = PronyConversion(direction="time_to_freq", n_modes=10)
        freq_data, _ = fwd.transform(data1)

        # Backward: freq → time
        bwd = PronyConversion(direction="freq_to_time", n_modes=10, t_out=t_orig)
        time_data, _ = bwd.transform(freq_data)

        G_t_recovered = np.asarray(time_data.y)
        # Allow 20% relative error due to Prony fitting approximation
        mask = G_t_orig > G_t_orig.max() * 0.01  # Ignore near-zero tail
        rel_error = np.abs(G_t_recovered[mask] - G_t_orig[mask]) / G_t_orig[mask]
        assert np.median(rel_error) < 0.2

    def test_invalid_direction(self):
        data = RheoData(x=np.array([1.0]), y=np.array([1.0]))
        transform = PronyConversion(direction="invalid")
        with pytest.raises(ValueError, match="Invalid direction"):
            transform.transform(data)
