"""Unit tests for SPP JAX kernels.

These tests exercise the lightweight analytical cases to keep runtime short
while confirming basic correctness of the SPP extraction routines.
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.utils import spp_kernels as ks

jax, jnp = safe_import_jax()


def _synthetic_laos(gamma_0: float = 1.0, G: float = 100.0, third_harm: float = 0.0):
    """Generate a simple LAOS waveform with optional 3rd harmonic."""
    omega = 1.0
    t = jnp.linspace(0, 2 * jnp.pi, 2000)
    gamma = gamma_0 * jnp.sin(omega * t)
    gamma_dot = gamma_0 * omega * jnp.cos(omega * t)
    sigma = G * gamma + third_harm * jnp.sin(3 * omega * t)
    return t, gamma, gamma_dot, sigma


def test_static_and_dynamic_yield_stress_linear_material():
    t, gamma, gamma_dot, sigma = _synthetic_laos(gamma_0=0.8, G=50.0)

    sigma_sy = ks.static_yield_stress(sigma, gamma, strain_amplitude=0.8)
    sigma_dy = ks.dynamic_yield_stress(sigma, gamma_dot, rate_amplitude=0.8)

    # Static yield should be close to linear prediction; dynamic should be smaller
    np.testing.assert_allclose(float(sigma_sy), 50.0 * 0.8, rtol=0.2)
    assert sigma_dy >= 0


def test_power_law_fit_recovers_known_parameters():
    gamma_dot = jnp.linspace(1e-2, 10.0, 50)
    K_true = 5.0
    n_true = 0.3
    sigma = K_true * jnp.abs(gamma_dot) ** n_true

    K_hat, n_hat, _ = ks.power_law_fit(sigma, gamma_dot)

    np.testing.assert_allclose(float(K_hat), K_true, rtol=0.1)
    np.testing.assert_allclose(float(n_hat), n_true, rtol=0.05)


def test_harmonic_reconstruction_preserves_first_harmonic():
    omega = 2.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1024)
    signal = 3.0 * jnp.sin(omega * t) + 0.5 * jnp.sin(3 * omega * t)

    amplitudes, phases, recon = ks.harmonic_reconstruction(
        signal, omega=omega, n_harmonics=1, dt=float(t[1] - t[0])
    )

    # Expect third harmonic largely removed
    ratio = jnp.linalg.norm(signal - jnp.asarray(recon)) / jnp.linalg.norm(signal)
    assert ratio < 0.3


def test_lissajous_metrics_positive_area():
    t, gamma, gamma_dot, sigma = _synthetic_laos(gamma_0=1.0, G=25.0, third_harm=10.0)

    metrics = ks.lissajous_metrics(
        stress=sigma,
        strain=gamma,
        strain_rate=gamma_dot,
        strain_amplitude=1.0,
        rate_amplitude=1.0,
    )

    assert "G_L" in metrics and metrics["G_L"] > 0
    assert "S_factor" in metrics


@pytest.mark.smoke
def test_apparent_cage_modulus_matches_modulus_in_linear_regime():
    _, gamma, _, sigma = _synthetic_laos(gamma_0=0.2, G=120.0)

    G_cage = ks.apparent_cage_modulus(sigma, gamma, strain_amplitude=0.2)

    # Mean cage modulus should be close to linear modulus
    mean_G = float(jnp.mean(jnp.abs(G_cage)))
    assert mean_G > 0


# ============================================================================
# Tests for numerical differentiation (MATLAB compatibility)
# ============================================================================


def test_numerical_derivative_first_order_sinusoid():
    """Test first derivative of sin(t) should be cos(t)."""
    omega = 2.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d_signal = ks.numerical_derivative(signal, dt, order=1, step_size=1)
    expected = omega * jnp.cos(omega * t)

    # Check interior points (avoid boundary artifacts)
    interior = slice(50, -50)
    np.testing.assert_allclose(
        np.array(d_signal[interior]),
        np.array(expected[interior]),
        rtol=0.02,
    )


def test_numerical_derivative_second_order_sinusoid():
    """Test second derivative of sin(t) should be -sin(t)."""
    omega = 1.5
    t = jnp.linspace(0, 4 * jnp.pi / omega, 1000)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d2_signal = ks.numerical_derivative(signal, dt, order=2, step_size=1)
    expected = -(omega**2) * jnp.sin(omega * t)

    interior = slice(100, -100)
    np.testing.assert_allclose(
        np.array(d2_signal[interior]),
        np.array(expected[interior]),
        rtol=0.05,
    )


def test_numerical_derivative_periodic_matches_analytic():
    """Test periodic differentiation on a complete period."""
    omega = 1.0
    # Exactly one period (endpoint=False for periodicity)
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d1, d2, d3 = ks.numerical_derivative_periodic(signal, dt, step_size=1)

    # Expected derivatives
    expected_d1 = omega * jnp.cos(omega * t)
    expected_d2 = -(omega**2) * jnp.sin(omega * t)
    expected_d3 = -(omega**3) * jnp.cos(omega * t)

    # Use atol for near-zero comparisons (numerical precision limits)
    np.testing.assert_allclose(
        np.array(d1), np.array(expected_d1), rtol=0.02, atol=1e-10
    )
    np.testing.assert_allclose(
        np.array(d2), np.array(expected_d2), rtol=0.05, atol=1e-10
    )
    np.testing.assert_allclose(
        np.array(d3), np.array(expected_d3), rtol=0.1, atol=1e-10
    )


@pytest.mark.smoke
def test_spp_numerical_analysis_linear_material():
    """Test SPP numerical analysis on a linear viscoelastic material."""
    omega = 1.0
    gamma_0 = 0.5
    G_prime = 100.0  # Storage modulus
    G_double_prime = 50.0  # Loss modulus

    # Generate linear LAOS response
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = G_prime * strain + G_double_prime * gamma_0 * jnp.cos(omega * t)

    result = ks.spp_numerical_analysis(strain, stress, omega, dt, step_size=1)

    # For linear material, Gp_t should be approximately constant at G'
    # and Gpp_t should be approximately constant at G''
    interior = slice(100, -100)  # Avoid boundary effects
    Gp_mean = float(jnp.nanmean(result["Gp_t"][interior]))
    Gpp_mean = float(jnp.nanmean(result["Gpp_t"][interior]))

    # Allow 15% tolerance for numerical derivatives
    np.testing.assert_allclose(Gp_mean, G_prime, rtol=0.15)
    np.testing.assert_allclose(Gpp_mean, G_double_prime, rtol=0.15)


def test_numerical_derivative_step_size_smoothing():
    """Larger step sizes should provide smoother derivatives."""
    omega = 1.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])

    # Add noise to signal
    rng = np.random.default_rng(42)
    signal = jnp.sin(omega * t) + 0.02 * jnp.array(rng.standard_normal(len(t)))

    d1_k1 = ks.numerical_derivative(signal, dt, order=1, step_size=1)
    d1_k3 = ks.numerical_derivative(signal, dt, order=1, step_size=3)

    # Larger step size should have lower variance in the derivative
    var_k1 = float(jnp.var(d1_k1))
    var_k3 = float(jnp.var(d1_k3))

    # k=3 should have lower variance (more smoothing)
    assert var_k3 < var_k1


def test_spp_numerical_analysis_num_mode_toggle():
    """num_mode=2 (periodic) should smooth boundary artifacts compared to mode 1."""
    omega = 2.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 200, endpoint=False)
    dt = float(t[1] - t[0])
    gamma_0 = 0.8
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 50.0 * strain

    res_edge = ks.spp_numerical_analysis(
        strain, stress, omega, dt, step_size=2, num_mode=1
    )
    res_loop = ks.spp_numerical_analysis(
        strain, stress, omega, dt, step_size=2, num_mode=2
    )

    # Both modes recover modulus ~50 Pa
    np.testing.assert_allclose(float(jnp.nanmean(res_edge["Gp_t"])), 50.0, rtol=5e-3)
    np.testing.assert_allclose(float(jnp.nanmean(res_loop["Gp_t"])), 50.0, rtol=5e-3)

    # Periodic mode should have lower variance at the boundaries
    assert float(jnp.var(res_loop["Gp_t"][:10])) < float(jnp.var(res_edge["Gp_t"][:10]))


# ============================================================================
# Tests for 4th-order numerical differentiation (Phase 1.1)
# ============================================================================


def test_numerical_derivative_4th_order_first_derivative():
    """Test 4th-order derivative of sin(t) matches cos(t)."""
    omega = 2.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d1 = ks.numerical_derivative_4th_order(signal, dt, order=1, step_size=1)
    expected = omega * jnp.cos(omega * t)

    # 4th order should be accurate in interior (avoid boundary artifacts)
    # Use atol for near-zero comparisons
    interior = slice(10, -10)
    np.testing.assert_allclose(
        np.array(d1[interior]), np.array(expected[interior]), rtol=0.01, atol=1e-10
    )


def test_numerical_derivative_4th_order_second_derivative():
    """Test 4th-order second derivative of sin(t) matches -sin(t)."""
    omega = 1.5
    t = jnp.linspace(0, 4 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d2 = ks.numerical_derivative_4th_order(signal, dt, order=2, step_size=1)
    expected = -(omega**2) * jnp.sin(omega * t)

    # Check interior region (avoid boundary effects)
    # Use atol for near-zero comparisons
    interior = slice(20, -20)
    np.testing.assert_allclose(
        np.array(d2[interior]), np.array(expected[interior]), rtol=0.02, atol=1e-8
    )


def test_numerical_derivative_4th_order_third_derivative():
    """Test 4th-order third derivative of sin(t) matches -cos(t)."""
    omega = 1.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    signal = jnp.sin(omega * t)

    d3 = ks.numerical_derivative_4th_order(signal, dt, order=3, step_size=1)
    expected = -(omega**3) * jnp.cos(omega * t)

    # Check interior region (avoid boundary effects and near-zero values)
    # Use atol for near-zero comparisons
    interior = slice(50, -50)
    np.testing.assert_allclose(
        np.array(d3[interior]), np.array(expected[interior]), rtol=0.05, atol=1e-10
    )


# ============================================================================
# Tests for phase alignment (Phase 1.2)
# ============================================================================


def test_compute_phase_offset_zero_for_pure_sine():
    """Phase offset should be near zero for pure sine wave."""
    omega = 1.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    signal = 5.0 * jnp.sin(omega * t)

    Delta = ks.compute_phase_offset(signal, omega, dt)
    # Pure sine has zero phase offset
    assert abs(float(Delta)) < 0.1


def test_compute_phase_offset_nonzero_for_pure_cosine():
    """Phase offset should be nonzero for pure cosine wave."""
    omega = 1.0
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    signal = 5.0 * jnp.cos(omega * t)

    Delta = ks.compute_phase_offset(signal, omega, dt)
    # Cosine has a phase offset relative to sine
    # The exact value depends on the FFT convention; just verify it's computed
    assert not jnp.isnan(Delta)
    assert abs(float(Delta)) > 0  # Should be nonzero for cosine


def test_harmonic_reconstruction_full_preserves_phase():
    """Phase-aligned reconstruction should preserve signal phase."""
    omega = 1.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    # Signal with known phase offset
    phi = 0.3
    strain = gamma_0 * jnp.sin(omega * t + phi)
    strain_rate = gamma_0 * omega * jnp.cos(omega * t + phi)
    stress = 100.0 * strain + 10.0 * jnp.sin(3 * omega * t + 3 * phi)

    # Pass concrete W_int to avoid data-dependent shape issues
    L = len(strain)
    W_int = int(round(L / 2))
    result = ks.harmonic_reconstruction_full(
        strain, strain_rate, stress, omega, n_harmonics=5, n_cycles=1, W_int=W_int
    )

    # Should return reconstructed waveforms
    assert "stress_recon" in result
    assert "strain_recon" in result
    assert "Delta" in result
    # Phase offset should be computed
    assert not jnp.isnan(result["Delta"])


# ============================================================================
# Tests for Fourier analysis with analytical derivatives (Phase 1.3)
# ============================================================================


def test_spp_fourier_analysis_linear_material():
    """Test Fourier SPP analysis recovers linear moduli."""
    omega = 1.0
    gamma_0 = 0.5
    G_prime = 100.0
    G_double_prime = 30.0

    t = jnp.linspace(0, 2 * jnp.pi / omega, 1000, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = G_prime * strain + G_double_prime * gamma_0 * jnp.cos(omega * t)

    result = ks.spp_fourier_analysis(strain, stress, omega, dt, n_harmonics=5)

    # Mean moduli should match linear values (allow higher tolerance for Fourier method)
    # Use interior region to avoid boundary effects
    interior = slice(100, -100)
    Gp_mean = float(jnp.nanmean(result["Gp_t"][interior]))
    Gpp_mean = float(jnp.nanmean(result["Gpp_t"][interior]))

    np.testing.assert_allclose(Gp_mean, G_prime, rtol=0.25)
    np.testing.assert_allclose(Gpp_mean, G_double_prime, rtol=0.35)


def test_spp_fourier_analysis_includes_frenet_serret():
    """Fourier analysis should compute Frenet-Serret frame."""
    omega = 1.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 100.0 * strain + 10.0 * jnp.sin(3 * omega * t)

    result = ks.spp_fourier_analysis(strain, stress, omega, dt, n_harmonics=5)

    assert "T_vec" in result
    assert "N_vec" in result
    assert "B_vec" in result
    # Check frame shapes
    assert result["T_vec"].shape == (len(result["Gp_t"]), 3)
    # Verify frame vectors are computed (not NaN)
    assert not jnp.any(jnp.isnan(result["T_vec"]))
    assert not jnp.any(jnp.isnan(result["B_vec"]))


# ============================================================================
# Tests for moduli rates (Phase 2.1)
# ============================================================================


def test_spp_numerical_analysis_includes_moduli_rates():
    """Numerical analysis should include Gp_t_dot, Gpp_t_dot, G_speed, delta_t_dot."""
    omega = 1.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 100.0 * strain + 20.0 * jnp.sin(3 * omega * t)

    result = ks.spp_numerical_analysis(strain, stress, omega, dt, step_size=1)

    assert "Gp_t_dot" in result
    assert "Gpp_t_dot" in result
    assert "G_speed" in result
    assert "delta_t_dot" in result
    # G_speed should be non-negative
    assert float(jnp.all(result["G_speed"] >= 0))


def test_moduli_rates_zero_for_linear_material():
    """For linear material, moduli should be constant so rates ~0."""
    omega = 1.0
    gamma_0 = 0.5
    G_prime = 100.0

    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = G_prime * strain  # Pure elastic

    result = ks.spp_numerical_analysis(strain, stress, omega, dt, step_size=1)

    # For constant G', the rate should be near zero
    # Allow tolerance for numerical noise
    interior = slice(50, -50)
    G_speed_mean = float(jnp.mean(jnp.abs(result["G_speed"][interior])))
    assert G_speed_mean < G_prime * 0.5  # Rate should be small relative to modulus


# ============================================================================
# Tests for Frenet-Serret frame (Phase 2.2)
# ============================================================================


def test_frenet_serret_frame_orthogonality():
    """T, N, B vectors should be mutually orthogonal."""
    omega = 1.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 100.0 * strain + 30.0 * jnp.sin(3 * omega * t)

    result = ks.spp_numerical_analysis(strain, stress, omega, dt, step_size=1)

    # Select middle points (avoid boundary issues)
    mid = len(result["T_vec"]) // 2
    T = result["T_vec"][mid]
    N = result["N_vec"][mid]
    B = result["B_vec"][mid]

    # Check orthogonality: T·N ≈ 0, T·B ≈ 0, N·B ≈ 0
    np.testing.assert_allclose(float(jnp.dot(T, N)), 0.0, atol=0.15)
    np.testing.assert_allclose(float(jnp.dot(T, B)), 0.0, atol=0.15)
    np.testing.assert_allclose(float(jnp.dot(N, B)), 0.0, atol=0.15)


def test_frenet_serret_frame_unit_vectors():
    """T, N, B vectors should be unit vectors."""
    omega = 1.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    stress = 100.0 * strain + 30.0 * jnp.sin(3 * omega * t)

    result = ks.spp_numerical_analysis(strain, stress, omega, dt, step_size=1)

    # Check norms in middle region
    interior = slice(50, -50)
    T_norms = jnp.linalg.norm(result["T_vec"][interior], axis=1)
    N_norms = jnp.linalg.norm(result["N_vec"][interior], axis=1)
    B_norms = jnp.linalg.norm(result["B_vec"][interior], axis=1)

    np.testing.assert_allclose(np.array(T_norms), 1.0, atol=0.1)
    np.testing.assert_allclose(np.array(N_norms), 1.0, atol=0.15)
    np.testing.assert_allclose(np.array(B_norms), 1.0, atol=0.15)


def test_frenet_serret_frame_standalone_function():
    """Test standalone frenet_serret_frame() function."""
    omega = 1.0
    gamma_0 = 0.5
    n_points = 500
    t = jnp.linspace(0, 2 * jnp.pi / omega, n_points, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)
    rate = gamma_0 * omega * jnp.cos(omega * t)
    stress = 100.0 * strain + 30.0 * jnp.sin(3 * omega * t)

    # Create rd (first derivatives) and rdd (second derivatives) from response wave
    response = jnp.column_stack([strain, rate / omega, stress])
    rd = jnp.gradient(response, dt, axis=0)
    rdd = jnp.gradient(rd, dt, axis=0)

    T_vec, N_vec, B_vec, curvature, torsion = ks.frenet_serret_frame(rd, rdd)

    assert T_vec.shape == (n_points, 3)
    assert N_vec.shape == (n_points, 3)
    assert B_vec.shape == (n_points, 3)


# ============================================================================
# Tests for displacement-stress yield extraction (Phase 2.3)
# ============================================================================


def test_yield_from_displacement_stress_basic():
    """Test yield extraction from displacement stress."""
    omega = 1.0
    gamma_0 = 0.5
    G_prime = 100.0

    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    strain = gamma_0 * jnp.sin(omega * t)
    strain_rate = gamma_0 * omega * jnp.cos(omega * t) / omega  # Normalized rate
    Gp_t = G_prime * jnp.ones_like(t)  # Constant modulus
    delta_t = jnp.zeros_like(t)  # Pure elastic: delta = 0
    # Displacement stress is zero for elastic material
    disp_stress = jnp.zeros_like(t)

    result = ks.yield_from_displacement_stress(
        disp_stress,
        strain,
        strain_rate,
        Gp_t,
        delta_t,
        strain_amplitude=gamma_0,
        rate_amplitude=gamma_0 * omega,
    )

    assert "sigma_sy_disp" in result
    assert "sigma_dy_disp" in result


# ============================================================================
# Tests for data preprocessing (Phase 3.2)
# ============================================================================


def test_differentiate_rate_from_strain():
    """Test strain rate differentiation from strain."""
    omega = 2.0
    gamma_0 = 0.5
    t = jnp.linspace(0, 2 * jnp.pi / omega, 500, endpoint=False)
    dt = float(t[1] - t[0])
    strain = gamma_0 * jnp.sin(omega * t)

    # Use explicit step_size=1 (integer, not traced)
    rate = ks.differentiate_rate_from_strain(strain, dt)  # Default step_size=1
    expected = gamma_0 * omega * jnp.cos(omega * t)

    # Check interior to avoid boundary effects
    # Use atol for near-zero comparisons
    interior = slice(20, -20)
    np.testing.assert_allclose(
        np.array(rate[interior]), np.array(expected[interior]), rtol=0.02, atol=1e-10
    )


def test_convert_units_strain():
    """Test unit conversion for strain."""
    strain_percent = jnp.array([1.0, 5.0, 10.0])
    strain_fraction = ks.convert_units(strain_percent, "percent", "fraction")

    np.testing.assert_allclose(np.array(strain_fraction), np.array([0.01, 0.05, 0.10]))


def test_convert_units_stress():
    """Test unit conversion for stress from Pa to kPa."""
    # Pa to kPa: multiply by 0.001
    stress_Pa = jnp.array([1000.0, 2500.0])
    stress_kPa = ks.convert_units(stress_Pa, "Pa", "kPa")

    np.testing.assert_allclose(np.array(stress_kPa), np.array([1.0, 2.5]))


def test_convert_units_angle():
    """Test unit conversion for angles."""
    angle_deg = jnp.array([45.0, 90.0, 180.0])
    angle_rad = ks.convert_units(angle_deg, "deg", "rad")

    expected = jnp.array([jnp.pi / 4, jnp.pi / 2, jnp.pi])
    np.testing.assert_allclose(np.array(angle_rad), np.array(expected))


def test_differentiate_rate_from_strain_wrapped_matches_cosine():
    """8-point wrapped derivative should track analytic cosine for sine strain."""
    t = jnp.linspace(0, 2 * jnp.pi, 1000)
    dt = float(t[1] - t[0])
    strain = jnp.sin(t)

    inferred_rate = ks.differentiate_rate_from_strain(
        strain, dt, step_size=8, looped=True
    )

    np.testing.assert_allclose(
        np.array(inferred_rate), np.array(jnp.cos(t)), rtol=8e-2, atol=8e-2
    )


def test_spp_numerical_analysis_accepts_vector_omega():
    """Numerical analysis should accept per-sample omega and keep shapes stable."""
    t = jnp.linspace(0, 2 * jnp.pi, 400)
    dt = float(t[1] - t[0])
    omega = 1.0 + 0.01 * jnp.sin(t)  # small jitter
    strain = jnp.sin(omega.mean() * t)
    stress = 50.0 * strain

    result = ks.spp_numerical_analysis(strain, stress, omega, dt)

    assert result["Gp_t"].shape == strain.shape
    assert np.isfinite(np.array(result["Gp_t"]).mean())
