"""Integration-style tests for the SPPDecomposer transform."""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.transforms.spp_decomposer import SPPDecomposer

jax, jnp = safe_import_jax()


def _make_rheodata(gamma_0: float = 0.5, G: float = 80.0, third: float = 5.0):
    omega = 1.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1500)
    strain = gamma_0 * jnp.sin(omega * t)
    stress = G * strain + third * jnp.sin(3 * omega * t)
    return RheoData(
        x=np.asarray(t, dtype=float),
        y=np.asarray(stress, dtype=float),
        domain="time",
        metadata={
            "test_mode": "oscillation",
            "omega": omega,
            "gamma_0": gamma_0,
            "strain": strain,
        },
    )


def test_decomposer_extracts_yield_metrics():
    data = _make_rheodata()
    decomposer = SPPDecomposer(
        omega=data.metadata["omega"], gamma_0=data.metadata["gamma_0"], n_harmonics=3
    )

    result = decomposer.transform(data)

    assert "sigma_sy" in decomposer.results_
    assert "sigma_dy" in decomposer.results_
    assert decomposer.results_["sigma_sy"] > 0
    assert "spp_results" in result.metadata
    assert result.metadata["spp_results"]["sigma_sy"] == decomposer.results_["sigma_sy"]

    # MATLAB-compatible outputs
    spp_out = decomposer.results_["spp_data_out"]
    assert spp_out.shape[1] == 15
    fsf_out = decomposer.results_["fsf_data_out"]
    assert fsf_out.shape[1] == 9
    assert "ft_out" in decomposer.results_


def test_decomposer_lissajous_metrics_present():
    data = _make_rheodata(third=10.0)
    decomposer = SPPDecomposer(
        omega=data.metadata["omega"], gamma_0=data.metadata["gamma_0"], n_harmonics=5
    )

    result = decomposer.transform(data)
    assert "G_L" in decomposer.results_
    assert "S_factor" in decomposer.results_
    # Metadata copy-through
    assert "spp_results" in result.metadata


@pytest.mark.smoke
def test_invalid_domain_raises():
    data = RheoData(
        x=[0, 1], y=[0, 1], domain="frequency", metadata={"test_mode": "oscillation"}
    )
    decomposer = SPPDecomposer(omega=1.0, gamma_0=0.1)

    with pytest.raises(ValueError):
        decomposer.transform(data)


# ============================================================================
# Tests for cycle selection and numerical method
# ============================================================================


def _make_multi_cycle_data(n_cycles: int = 5, gamma_0: float = 0.5, G: float = 80.0):
    """Generate multiple LAOS cycles."""
    omega = 1.0
    t = jnp.linspace(0, n_cycles * 2 * jnp.pi / omega, n_cycles * 500)
    strain = gamma_0 * jnp.sin(omega * t)
    stress = G * strain
    return RheoData(
        x=np.asarray(t, dtype=float),
        y=np.asarray(stress, dtype=float),
        domain="time",
        metadata={
            "test_mode": "oscillation",
            "omega": omega,
            "gamma_0": gamma_0,
            "strain": strain,
        },
    )


def test_cycle_selection_start_cycle():
    """Test that start_cycle parameter skips initial cycles."""
    data = _make_multi_cycle_data(n_cycles=5)
    omega = data.metadata["omega"]
    gamma_0 = data.metadata["gamma_0"]

    # Analyze all cycles
    decomposer_all = SPPDecomposer(omega=omega, gamma_0=gamma_0)
    decomposer_all.transform(data)

    # Analyze starting from cycle 2
    decomposer_skip = SPPDecomposer(omega=omega, gamma_0=gamma_0, start_cycle=2)
    decomposer_skip.transform(data)

    # Both should produce valid results
    assert decomposer_all.results_["sigma_sy"] > 0
    assert decomposer_skip.results_["sigma_sy"] > 0

    # Cycle selection should be recorded in results
    assert decomposer_skip.results_["cycles_analyzed"] == (2, 5)


def test_cycle_selection_end_cycle():
    """Test that end_cycle parameter limits analyzed cycles."""
    data = _make_multi_cycle_data(n_cycles=5)
    omega = data.metadata["omega"]
    gamma_0 = data.metadata["gamma_0"]

    # Analyze only cycles 1-3
    decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0, start_cycle=1, end_cycle=3)
    decomposer.transform(data)

    assert decomposer.results_["cycles_analyzed"] == (1, 3)
    assert decomposer.results_["sigma_sy"] > 0


def test_numerical_method_produces_additional_results():
    """Test that use_numerical_method=True produces additional outputs."""
    data = _make_rheodata()
    omega = data.metadata["omega"]
    gamma_0 = data.metadata["gamma_0"]

    # With numerical method
    decomposer = SPPDecomposer(
        omega=omega,
        gamma_0=gamma_0,
        use_numerical_method=True,
        step_size=1,
    )
    result = decomposer.transform(data)

    # Should have numerical results
    assert "numerical" in decomposer.results_
    assert "Gp_t" in decomposer.results_["numerical"]
    assert "Gpp_t" in decomposer.results_["numerical"]
    assert "G_star_t" in decomposer.results_["numerical"]

    # Should have mean values
    assert "Gp_t_mean" in decomposer.results_
    assert "Gpp_t_mean" in decomposer.results_

    # Metadata should record numerical method usage
    assert result.metadata["use_numerical_method"] is True


def test_numerical_method_linear_material_moduli():
    """Test that numerical method recovers correct moduli for linear material."""
    omega = 1.0
    gamma_0 = 0.5
    G_prime = 100.0
    G_double_prime = 30.0

    # Generate linear viscoelastic response
    t = jnp.linspace(0, 4 * jnp.pi / omega, 2000, endpoint=False)
    strain = gamma_0 * jnp.sin(omega * t)
    strain_rate = gamma_0 * omega * jnp.cos(omega * t)
    stress = G_prime * strain + G_double_prime / omega * strain_rate

    data = RheoData(
        x=np.asarray(t, dtype=float),
        y=np.asarray(stress, dtype=float),
        domain="time",
        metadata={
            "test_mode": "oscillation",
            "omega": omega,
            "gamma_0": gamma_0,
            "strain": strain,
            "strain_rate": strain_rate,
        },
    )

    decomposer = SPPDecomposer(
        omega=omega,
        gamma_0=gamma_0,
        use_numerical_method=True,
        step_size=1,
    )
    decomposer.transform(data)

    # For linear material, mean Gp should be close to G'
    # Allow 20% tolerance for numerical methods
    np.testing.assert_allclose(
        decomposer.results_["Gp_t_mean"],
        G_prime,
        rtol=0.20,
    )


def test_step_size_parameter():
    """Test that step_size parameter is respected."""
    data = _make_rheodata()
    omega = data.metadata["omega"]
    gamma_0 = data.metadata["gamma_0"]

    # With different step sizes
    decomposer_k1 = SPPDecomposer(
        omega=omega,
        gamma_0=gamma_0,
        use_numerical_method=True,
        step_size=1,
    )
    decomposer_k1.transform(data)

    decomposer_k3 = SPPDecomposer(
        omega=omega,
        gamma_0=gamma_0,
        use_numerical_method=True,
        step_size=3,
    )
    decomposer_k3.transform(data)

    # Both should produce valid results
    assert "Gp_t_mean" in decomposer_k1.results_
    assert "Gp_t_mean" in decomposer_k3.results_

    # Larger step size should produce smoother results (lower variance)
    Gp_k1 = decomposer_k1.results_["numerical"]["Gp_t"]
    Gp_k3 = decomposer_k3.results_["numerical"]["Gp_t"]

    var_k1 = float(jnp.nanvar(Gp_k1))
    var_k3 = float(jnp.nanvar(Gp_k3))

    # k=3 should have lower variance (more smoothing)
    assert var_k3 <= var_k1 * 1.1  # Allow some tolerance


def test_rogers_defaults_and_delta_present():
    omega = 2.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 600)
    stress = 100.0 * jnp.sin(omega * t)
    strain = gamma_0 * jnp.sin(omega * t)

    data = RheoData(
        x=t,
        y=stress,
        domain="time",
        metadata={"omega": omega, "gamma_0": gamma_0, "strain": strain},
    )

    decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0)
    decomposer.transform(data)

    assert decomposer.step_size == 8
    assert decomposer.num_mode == 2
    assert decomposer.n_harmonics == 39
    assert "Delta" in decomposer.results_
    assert decomposer.results_["spp_params"][1] == 39
