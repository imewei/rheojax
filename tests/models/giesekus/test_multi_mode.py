"""Tests for GiesekusMultiMode model.

Tests cover:
- Instantiation with different n_modes
- Per-mode parameter management
- SAOS predictions (multi-mode superposition)
- Flow curve predictions
- ODE-based startup simulation
- Relaxation spectrum
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.giesekus import GiesekusMultiMode

jax, jnp = safe_import_jax()


class TestInstantiation:
    """Tests for multi-mode instantiation."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default 3 modes."""
        model = GiesekusMultiMode()

        assert model.n_modes == 3
        assert model.eta_s == 0.0

    @pytest.mark.smoke
    def test_custom_n_modes(self):
        """Test instantiation with custom number of modes."""
        for n in [1, 2, 5, 10]:
            model = GiesekusMultiMode(n_modes=n)
            assert model.n_modes == n

    def test_invalid_n_modes(self):
        """Test error for invalid n_modes."""
        with pytest.raises(ValueError):
            GiesekusMultiMode(n_modes=0)

        with pytest.raises(ValueError):
            GiesekusMultiMode(n_modes=-1)

    @pytest.mark.smoke
    def test_parameter_creation(self):
        """Test parameters are created for each mode."""
        model = GiesekusMultiMode(n_modes=3)

        # Check per-mode parameters exist
        for i in range(3):
            assert f"eta_p_{i}" in model.parameters.keys()
            assert f"lambda_{i}" in model.parameters.keys()
            assert f"alpha_{i}" in model.parameters.keys()

        # Check shared solvent viscosity
        assert "eta_s" in model.parameters.keys()


class TestParameterManagement:
    """Tests for per-mode parameter management."""

    @pytest.mark.smoke
    def test_get_mode_params(self):
        """Test getting mode parameters."""
        model = GiesekusMultiMode(n_modes=3)

        params = model.get_mode_params(0)

        assert "eta_p" in params
        assert "lambda_1" in params
        assert "alpha" in params

    @pytest.mark.smoke
    def test_set_mode_params(self):
        """Test setting mode parameters."""
        model = GiesekusMultiMode(n_modes=3)

        model.set_mode_params(1, eta_p=75.0, lambda_1=5.0, alpha=0.2)

        params = model.get_mode_params(1)
        assert params["eta_p"] == 75.0
        assert params["lambda_1"] == 5.0
        assert params["alpha"] == 0.2

    def test_mode_index_bounds(self):
        """Test error for out-of-bounds mode index."""
        model = GiesekusMultiMode(n_modes=3)

        with pytest.raises(IndexError):
            model.get_mode_params(3)

        with pytest.raises(IndexError):
            model.set_mode_params(-1, eta_p=100.0)

    @pytest.mark.smoke
    def test_get_mode_arrays(self):
        """Test getting all mode parameters as arrays."""
        model = GiesekusMultiMode(n_modes=3)
        model.set_mode_params(0, eta_p=100.0, lambda_1=10.0, alpha=0.3)
        model.set_mode_params(1, eta_p=50.0, lambda_1=1.0, alpha=0.2)
        model.set_mode_params(2, eta_p=20.0, lambda_1=0.1, alpha=0.1)

        eta_p, lambda_vals, alpha = model.get_mode_arrays()

        assert len(eta_p) == 3
        assert len(lambda_vals) == 3
        assert len(alpha) == 3

        assert np.isclose(eta_p[0], 100.0)
        assert np.isclose(lambda_vals[1], 1.0)
        assert np.isclose(alpha[2], 0.1)

    def test_eta_0_property(self):
        """Test total zero-shear viscosity."""
        model = GiesekusMultiMode(n_modes=3)
        model.parameters.set_value("eta_s", 5.0)
        model.set_mode_params(0, eta_p=100.0)
        model.set_mode_params(1, eta_p=50.0)
        model.set_mode_params(2, eta_p=20.0)

        # η₀ = η_s + Σ η_p,i = 5 + 100 + 50 + 20 = 175
        assert np.isclose(model.eta_0, 175.0)


class TestSAOSPredictions:
    """Tests for multi-mode SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction."""
        model = GiesekusMultiMode(n_modes=3)

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    @pytest.mark.smoke
    def test_saos_superposition(self):
        """Test SAOS is sum of Maxwell modes."""
        model = GiesekusMultiMode(n_modes=2)
        model.set_mode_params(0, eta_p=100.0, lambda_1=10.0)
        model.set_mode_params(1, eta_p=50.0, lambda_1=1.0)
        model.parameters.set_value("eta_s", 0.0)

        omega = 1.0
        G_prime, G_double_prime = model.predict_saos(np.array([omega]))

        # Manual calculation
        G1 = 100.0 / 10.0  # = 10
        wl1 = omega * 10.0
        Gp1 = G1 * wl1**2 / (1 + wl1**2)
        Gpp1 = G1 * wl1 / (1 + wl1**2)

        G2 = 50.0 / 1.0  # = 50
        wl2 = omega * 1.0
        Gp2 = G2 * wl2**2 / (1 + wl2**2)
        Gpp2 = G2 * wl2 / (1 + wl2**2)

        expected_Gp = Gp1 + Gp2
        expected_Gpp = Gpp1 + Gpp2

        assert np.isclose(G_prime[0], expected_Gp, rtol=0.01)
        assert np.isclose(G_double_prime[0], expected_Gpp, rtol=0.01)

    def test_saos_broad_spectrum(self):
        """Test multi-mode captures broad frequency range."""
        model = GiesekusMultiMode(n_modes=5)

        # Set logarithmically spaced relaxation times
        for i in range(5):
            model.set_mode_params(i, eta_p=100.0 / (i + 1), lambda_1=10.0 ** (2 - i))

        omega = np.logspace(-3, 3, 50)
        G_prime, G_double_prime = model.predict_saos(omega)

        # Should have reasonable moduli across 6 decades
        assert np.all(G_prime > 0)
        assert np.all(G_double_prime > 0)

        # Check transitions visible
        log_omega = np.log10(omega)
        log_Gp = np.log10(G_prime)

        # G' should increase with frequency overall
        assert log_Gp[-1] > log_Gp[0]


class TestFlowCurvePredictions:
    """Tests for multi-mode flow curve predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction."""
        model = GiesekusMultiMode(n_modes=3)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)

    @pytest.mark.smoke
    def test_flow_curve_via_predict(self):
        """Test flow curve via predict() method."""
        model = GiesekusMultiMode(n_modes=3)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape

    def test_flow_curve_with_viscosity(self):
        """Test flow curve with viscosity return."""
        model = GiesekusMultiMode(n_modes=3)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma, eta = model.predict_flow_curve(gamma_dot, return_components=True)

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert np.all(eta > 0)


class TestStartupSimulation:
    """Tests for multi-mode startup simulation."""

    @pytest.mark.slow
    def test_simulate_startup(self):
        """Test startup simulation runs.

        Note: Multi-mode ODE has JIT compatibility issues with diffrax.
        Marked as slow pending investigation.
        """
        model = GiesekusMultiMode(n_modes=2)
        model.set_mode_params(0, eta_p=100.0, lambda_1=1.0, alpha=0.3)
        model.set_mode_params(1, eta_p=50.0, lambda_1=0.1, alpha=0.2)

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_startup(t, gamma_dot=1.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.slow
    def test_startup_reaches_steady_state(self):
        """Test startup approaches steady state."""
        model = GiesekusMultiMode(n_modes=2)
        model.set_mode_params(0, eta_p=100.0, lambda_1=1.0, alpha=0.3)
        model.set_mode_params(1, eta_p=50.0, lambda_1=0.1, alpha=0.2)

        t = np.linspace(0, 20, 200)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        # Stress should approach steady value
        # (variance of last 10% should be small)
        final_stress = sigma[int(0.9 * len(sigma)) :]
        cv = np.std(final_stress) / np.mean(final_stress)

        assert cv < 0.05  # Less than 5% coefficient of variation

    def test_startup_full_return(self):
        """Test startup with per-mode stress return."""
        model = GiesekusMultiMode(n_modes=2)

        t = np.linspace(0, 5, 50)
        result = model.simulate_startup(t, gamma_dot=10.0, return_full=True)

        assert "t" in result
        assert "tau_xy_0" in result
        assert "tau_xy_1" in result
        assert "tau_xy_total" in result


class TestRelaxationSpectrum:
    """Tests for relaxation spectrum analysis."""

    @pytest.mark.smoke
    def test_get_relaxation_spectrum(self):
        """Test discrete spectrum retrieval."""
        model = GiesekusMultiMode(n_modes=3)
        model.set_mode_params(0, eta_p=100.0, lambda_1=10.0)
        model.set_mode_params(1, eta_p=50.0, lambda_1=1.0)
        model.set_mode_params(2, eta_p=20.0, lambda_1=0.1)

        lambdas, G_i = model.get_relaxation_spectrum()

        assert len(lambdas) == 3
        assert len(G_i) == 3

        # G_i = η_p,i / λ_i
        assert np.isclose(G_i[0], 100.0 / 10.0)  # After sorting

    def test_continuous_spectrum(self):
        """Test continuous G(t) computation."""
        model = GiesekusMultiMode(n_modes=3)
        model.set_mode_params(0, eta_p=100.0, lambda_1=10.0)
        model.set_mode_params(1, eta_p=50.0, lambda_1=1.0)
        model.set_mode_params(2, eta_p=20.0, lambda_1=0.1)

        t, G_t = model.get_continuous_spectrum(n_points=100)

        assert len(t) == 100
        assert len(G_t) == 100

        # G(t) should decay
        assert G_t[0] > G_t[-1]


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("giesekus_multi")
        assert isinstance(model, GiesekusMultiMode)

        model2 = ModelRegistry.create("giesekus_multimode")
        assert isinstance(model2, GiesekusMultiMode)


class TestModelFunction:
    """Tests for model_function (Bayesian interface)."""

    @pytest.mark.smoke
    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = GiesekusMultiMode(n_modes=2)

        # Build params array: [eta_s, eta_p_0, eta_p_1, lambda_0, lambda_1, alpha_0, alpha_1]
        params = jnp.array([5.0, 100.0, 50.0, 10.0, 1.0, 0.3, 0.2])

        omega = jnp.logspace(-1, 1, 10)
        y = model.model_function(omega, params, test_mode="oscillation")

        assert y.shape == (len(omega), 2)  # G_prime and G_double_prime
        assert np.all(y > 0)
