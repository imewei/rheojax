"""Tests for Mutation Number transform."""

import jax.numpy as jnp
import numpy as np
import pytest

from rheo.core.data import RheoData
from rheo.core.test_modes import TestMode
from rheo.transforms.mutation_number import MutationNumber


class TestMutationNumber:
    """Test suite for Mutation Number transform."""

    def test_basic_initialization(self):
        """Test basic mutation number initialization."""
        mn = MutationNumber()
        assert mn.integration_method == "trapz"
        assert mn.extrapolate is False

    def test_custom_initialization(self):
        """Test mutation number with custom parameters."""
        mn = MutationNumber(
            integration_method="simpson",
            extrapolate=True,
            extrapolation_model="powerlaw",
        )
        assert mn.integration_method == "simpson"
        assert mn.extrapolate is True
        assert mn.extrapolation_model == "powerlaw"

    def test_exponential_relaxation(self):
        """Test mutation number for exponential relaxation."""
        # G(t) = G_0 * exp(-t/tau)
        # For pure exponential: Δ should be close to 1
        t = jnp.linspace(0, 50, 1000)
        tau = 5.0
        G_0 = 1000.0
        G_t = G_0 * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time", metadata={"test_mode": "relaxation"})

        # Calculate mutation number
        mn = MutationNumber(integration_method="trapz")
        delta = mn.calculate(data)

        # For exponential decay, Δ ≈ 1 (viscous)
        # Actual value depends on integration limits
        assert 0.5 < delta < 1.5

    def test_maxwell_relaxation(self):
        """Test mutation number for Maxwell element."""
        # Maxwell: G(t) = G_0 * exp(-t/tau)
        t = jnp.linspace(0.01, 100, 2000)
        tau = 10.0
        G_0 = 1000.0
        G_t = G_0 * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        # Should auto-detect as relaxation
        mn = MutationNumber()
        delta = mn.calculate(data)

        # Mutation number should be finite and positive
        assert 0 < delta < 2

    def test_zener_relaxation(self):
        """Test mutation number for Zener (SLS) model."""
        # Zener: G(t) = G_eq + (G_0 - G_eq) * exp(-t/tau)
        t = jnp.linspace(0, 50, 1000)
        G_0 = 1000.0
        G_eq = 200.0  # Non-zero equilibrium modulus
        tau = 5.0
        G_t = G_eq + (G_0 - G_eq) * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        delta = mn.calculate(data)

        # Should be less than pure Maxwell (has equilibrium modulus)
        assert 0 < delta < 1.0

    def test_integration_methods(self):
        """Test different integration methods."""
        t = jnp.linspace(0, 50, 1000)
        tau = 5.0
        G_t = 1000.0 * jnp.exp(-t / tau)
        data = RheoData(x=t, y=G_t, domain="time")

        methods = ["trapz", "simpson", "cumulative"]
        deltas = []

        for method in methods:
            mn = MutationNumber(integration_method=method)
            delta = mn.calculate(data)
            deltas.append(delta)

            # Should be positive and finite
            assert 0 < delta < 2

        # All methods should give similar results
        deltas = np.array(deltas)
        assert np.std(deltas) / np.mean(deltas) < 0.1  # <10% variation

    def test_transform_method(self):
        """Test transform method returns RheoData."""
        t = jnp.linspace(0, 50, 1000)
        G_t = 1000.0 * jnp.exp(-t / 5.0)
        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        result = mn.transform(data)

        # Should return RheoData
        assert isinstance(result, RheoData)

        # Should be scalar (single value)
        assert len(result.y) == 1

        # Mutation number should be in metadata
        assert "mutation_number" in result.metadata

    def test_relaxation_time_calculation(self):
        """Test average relaxation time calculation."""
        # Known relaxation time
        t = jnp.linspace(0, 50, 1000)
        tau = 5.0
        G_0 = 1000.0
        G_t = G_0 * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        tau_avg = mn.get_relaxation_time(data)

        # Should be close to actual tau
        # (May differ due to finite integration range)
        assert 3.0 < tau_avg < 7.0

    def test_equilibrium_modulus_estimation(self):
        """Test equilibrium modulus estimation."""
        # Zener model with known equilibrium modulus
        t = jnp.linspace(0, 100, 1000)
        G_eq = 200.0
        G_0 = 1000.0
        tau = 5.0
        G_t = G_eq + (G_0 - G_eq) * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        G_eq_est = mn.get_equilibrium_modulus(data)

        # Should be close to actual G_eq
        assert 150 < G_eq_est < 250

    def test_extrapolation_exponential(self):
        """Test extrapolation to infinite time (exponential)."""
        # Create data that doesn't fully decay
        t = jnp.linspace(0, 10, 500)  # Short time window
        tau = 20.0  # Long relaxation time
        G_t = 1000.0 * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        # Without extrapolation
        mn_no_extrap = MutationNumber(extrapolate=False)
        delta_no_extrap = mn_no_extrap.calculate(data)

        # With extrapolation
        mn_extrap = MutationNumber(extrapolate=True, extrapolation_model="exponential")
        delta_extrap = mn_extrap.calculate(data)

        # Extrapolated should be larger (captures tail)
        assert delta_extrap > delta_no_extrap

    def test_non_relaxation_error(self):
        """Test error for non-relaxation data."""
        # Create monotonically increasing data (creep-like)
        t = jnp.linspace(0, 10, 100)
        J_t = t  # Linear creep

        data = RheoData(x=t, y=J_t, domain="time")

        mn = MutationNumber()
        with pytest.raises(ValueError, match="RELAXATION"):
            mn.calculate(data)

    def test_complex_data_handling(self):
        """Test handling of complex data."""
        t = jnp.linspace(0, 50, 1000)
        G_real = 1000.0 * jnp.exp(-t / 5.0)
        G_complex = G_real + 1j * G_real * 0.1  # Add small imaginary part

        data = RheoData(x=t, y=G_complex, domain="time")

        mn = MutationNumber()
        delta = mn.calculate(data)

        # Should handle complex by taking real part
        assert np.isfinite(delta)
        assert delta > 0

    def test_edge_case_zero_modulus(self):
        """Test error handling for zero initial modulus."""
        t = jnp.linspace(0, 10, 100)
        G_t = jnp.zeros_like(t)

        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        with pytest.raises(ValueError, match="positive"):
            mn.calculate(data)

    def test_metadata_preservation(self):
        """Test metadata preservation."""
        t = jnp.linspace(0, 50, 1000)
        G_t = 1000.0 * jnp.exp(-t / 5.0)

        data = RheoData(
            x=t,
            y=G_t,
            domain="time",
            metadata={"sample": "polymer", "temperature": 298},
        )

        mn = MutationNumber()
        result = mn.transform(data)

        # Original metadata preserved
        assert result.metadata["sample"] == "polymer"
        assert result.metadata["temperature"] == 298

        # Transform metadata added
        assert "transform" in result.metadata
        assert result.metadata["transform"] == "mutation_number"
        assert "integration_method" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
