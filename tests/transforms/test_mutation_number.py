"""Tests for Mutation Number transform."""

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.test_modes import TestMode
from rheojax.transforms.mutation_number import MutationNumber

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestMutationNumber:
    """Test suite for Mutation Number transform."""

    @pytest.mark.smoke
    def test_basic_initialization(self):
        """Test basic mutation number initialization."""
        mn = MutationNumber()
        assert mn.integration_method == "trapz"
        assert mn.extrapolate is False

    @pytest.mark.smoke
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

    @pytest.mark.smoke
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
        # The relaxing part G_relax = (G_0 - G_eq) * exp(-t/tau) is pure exponential
        # For exponential decay, the mutation number formula gives Δ ≈ 1.0
        t = jnp.linspace(0, 50, 1000)
        G_0 = 1000.0
        G_eq = 200.0  # Non-zero equilibrium modulus (20% retention)
        tau = 5.0
        G_t = G_eq + (G_0 - G_eq) * jnp.exp(-t / tau)

        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber()
        delta = mn.calculate(data)

        # For Zener with exponential relaxation, Δ ≈ 1.0
        # (The relaxing part decays completely, like a viscous fluid)
        assert 0.9 < delta <= 1.0

    def test_integration_methods(self):
        """Test different integration methods."""
        t = jnp.linspace(0, 50, 1000)
        tau = 5.0
        G_t = 1000.0 * jnp.exp(-t / tau)
        data = RheoData(x=t, y=G_t, domain="time")

        methods = ["trapz", "simpson"]
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

    def test_cumulative_integration_not_implemented(self):
        """integration_method='cumulative' must raise, not silently behave
        like 'trapz' (P3 regression: the branch was byte-identical to trapz
        with no indication the option had no effect)."""
        t = jnp.linspace(0, 50, 1000)
        G_t = 1000.0 * jnp.exp(-t / 5.0)
        data = RheoData(x=t, y=G_t, domain="time")

        mn = MutationNumber(integration_method="cumulative")
        with pytest.raises(NotImplementedError):
            mn.calculate(data)

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

    def test_relaxation_time_converges_for_nonzero_geq(self):
        """get_relaxation_time must subtract G_eq so tau_avg converges to the
        true relaxation time regardless of the measurement window, instead of
        growing roughly linearly with the window length (P1 regression:
        previously integrated raw G(t) without subtracting G_eq)."""
        G_0 = 1000.0
        G_eq = 200.0
        tau = 5.0

        mn = MutationNumber()
        taus = []
        for t_max in (50.0, 200.0, 500.0):
            t = jnp.linspace(0, t_max, 2000)
            G_t = G_eq + (G_0 - G_eq) * jnp.exp(-t / tau)
            data = RheoData(x=t, y=G_t, domain="time")
            taus.append(mn.get_relaxation_time(data))

        for tau_avg in taus:
            assert 4.5 < tau_avg < 5.5

        # Must not blow up with the measurement window like the old
        # (buggy) behavior, where tau_avg roughly tripled per window extension.
        assert max(taus) - min(taus) < 0.5

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

        # Both should give Δ ≈ 1 for exponential decay
        # Extrapolation might capture slightly more tail contribution
        assert delta_no_extrap >= 0.9
        assert delta_extrap >= delta_no_extrap  # Can be equal if both hit 1.0 ceiling

    def test_extrapolate_tG_integral_respects_powerlaw_model(self):
        """_extrapolate_tG_integral must use self.extrapolation_model instead
        of always fitting an exponential tail (P1 regression: previously the
        ∫t×G dt tail correction ignored extrapolation_model entirely, so
        'powerlaw' mode silently got an exponential-fit tail mismatched with
        the powerlaw-fit ∫G dt tail from _extrapolate_tail)."""
        t = jnp.linspace(0.5, 5.0, 50)
        G_relax = 1000.0 * jnp.power(t + 1.0, -3.0)  # exact powerlaw decay

        mn_exp = MutationNumber(extrapolation_model="exponential")
        mn_pow = MutationNumber(extrapolation_model="powerlaw")

        tail_tG_exp = mn_exp._extrapolate_tG_integral(t, G_relax)
        tail_tG_pow = mn_pow._extrapolate_tG_integral(t, G_relax)

        assert np.isfinite(tail_tG_exp) and tail_tG_exp > 0
        assert np.isfinite(tail_tG_pow) and tail_tG_pow > 0

        # Before the fix, both configurations produced the exact same value
        # because the powerlaw setting was ignored. After the fix, the
        # powerlaw-consistent tail must differ materially from the
        # exponential-fit tail for genuine powerlaw-decay data.
        assert tail_tG_pow != pytest.approx(tail_tG_exp, rel=0.05)

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
