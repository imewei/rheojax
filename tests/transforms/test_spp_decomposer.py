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


def test_wrap_strain_rate_affects_fourier_core_outputs():
    """Regression: wrap_strain_rate must propagate into the Fourier core path.

    Before the fix, spp_fourier_analysis always recomputed its internal
    strain rate with periodic wrapping (looped=True) regardless of the
    constructor's wrap_strain_rate setting, so Gp_t/Gpp_t/etc. were
    identical whether wrap_strain_rate was True or False.
    """
    # A coarse single-cycle signal makes the 8-point-stencil edge zone a
    # sizeable fraction of the array, so periodic-wrap vs edge-aware strain
    # rate inference diverge well above floating-point noise.
    omega = 1.5
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 64, endpoint=False)
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 80.0 * strain + 5.0 * jnp.sin(3 * omega * t)
    data = RheoData(
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

    decomposer_wrap = SPPDecomposer(
        omega=omega, gamma_0=gamma_0, n_harmonics=3, wrap_strain_rate=True
    )
    decomposer_wrap.transform(data)

    decomposer_nowrap = SPPDecomposer(
        omega=omega, gamma_0=gamma_0, n_harmonics=3, wrap_strain_rate=False
    )
    decomposer_nowrap.transform(data)

    Gp_wrap = decomposer_wrap.results_["core"]["Gp_t"]
    Gp_nowrap = decomposer_nowrap.results_["core"]["Gp_t"]
    assert not np.allclose(Gp_wrap, Gp_nowrap, rtol=1e-4, atol=1e-6)


def test_get_cycle_mask_rounds_not_truncates_cycle_count():
    """Regression: floating-point drift just under N periods must round to
    N cycles, not floor to N-1 (spp_decomposer.py _get_cycle_mask).
    """
    omega = 1.0
    T_period = 2 * np.pi / omega
    dt = T_period / 500.0
    n_periods = 5
    # Cumulative summation of a non-exact dt accumulates floating-point
    # error, landing total_time/T_period just under 5.0 (~4.998).
    t = np.cumsum(np.full(n_periods * 500, dt)) - dt
    gamma_0 = 0.5
    strain = gamma_0 * np.sin(omega * t)
    stress = 80.0 * strain

    data = RheoData(
        x=t,
        y=stress,
        domain="time",
        metadata={"omega": omega, "gamma_0": gamma_0, "strain": strain},
    )

    decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0, start_cycle=2)
    decomposer.transform(data)

    assert decomposer.results_["cycles_analyzed"] == (2, 5)


def test_invalid_cycle_range_raises():
    """Regression: end_cycle before start_cycle after clamping must raise,
    not silently fall back to analyzing the entire dataset.
    """
    data = _make_multi_cycle_data(n_cycles=5)
    decomposer = SPPDecomposer(
        omega=data.metadata["omega"],
        gamma_0=data.metadata["gamma_0"],
        start_cycle=3,
        end_cycle=1,
    )

    with pytest.raises(ValueError):
        decomposer.transform(data)


def test_scrambled_time_raises_instead_of_silent_corruption():
    """Regression: unsorted/non-uniform time must raise, not silently produce
    finite-but-wrong output from FFT/derivative kernels that assume ordered,
    uniformly-spaced samples.
    """
    omega = 1.0
    t = jnp.linspace(0, 4 * jnp.pi / omega, 200)
    strain = jnp.sin(omega * t)
    stress = 100.0 * strain + 50.0 * jnp.cos(omega * t)

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(t))
    t_scrambled = np.asarray(t)[perm]
    stress_scrambled = np.asarray(stress)[perm]

    data = RheoData(
        x=t_scrambled,
        y=stress_scrambled,
        domain="time",
        metadata={"omega": omega, "gamma_0": 1.0},
        validate=False,
    )
    decomposer = SPPDecomposer(omega=omega, gamma_0=1.0, use_numerical_method=True)

    with pytest.raises(ValueError):
        decomposer.transform(data)


def test_harmonic_reconstruction_uses_resolved_omega():
    """Regression: harmonic_reconstruction must use the omega resolved from
    dataset metadata, not the stale constructor omega, when they differ.
    """
    true_omega = 2.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 4 * jnp.pi / true_omega, 2000, endpoint=False)
    stress = 100.0 * jnp.sin(true_omega * t) + 20.0 * jnp.sin(3 * true_omega * t + 0.1)

    data = RheoData(
        x=np.asarray(t, dtype=float),
        y=np.asarray(stress, dtype=float),
        domain="time",
        metadata={"omega": true_omega, "gamma_0": gamma_0},
    )

    # Constructor omega is intentionally stale relative to the dataset.
    decomposer = SPPDecomposer(omega=1.0, gamma_0=gamma_0, n_harmonics=3)
    decomposer.transform(data)

    amps = decomposer.results_["harmonic_amplitudes"]
    np.testing.assert_allclose(amps[0], 100.0, rtol=0.05)
    np.testing.assert_allclose(amps[1], 20.0, rtol=0.05)


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
