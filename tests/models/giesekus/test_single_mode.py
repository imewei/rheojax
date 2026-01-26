"""Tests for GiesekusSingleMode model.

Tests cover:
- Instantiation and parameter management
- Flow curve predictions
- SAOS predictions
- ODE-based simulations (startup, relaxation, creep, LAOS)
- Registry integration
- BayesianMixin interface
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.models.giesekus import GiesekusSingleMode

jax, jnp = safe_import_jax()


class TestInstantiation:
    """Tests for model instantiation and parameters."""

    @pytest.mark.smoke
    def test_default_instantiation(self):
        """Test model instantiates with default parameters."""
        model = GiesekusSingleMode()

        assert model.eta_p == 100.0
        assert model.lambda_1 == 1.0
        assert model.alpha == 0.3
        assert model.eta_s == 0.0

    @pytest.mark.smoke
    def test_parameter_setting(self):
        """Test parameters can be set."""
        model = GiesekusSingleMode()

        model.parameters.set_value("eta_p", 200.0)
        model.parameters.set_value("lambda_1", 2.0)
        model.parameters.set_value("alpha", 0.4)
        model.parameters.set_value("eta_s", 50.0)

        assert model.eta_p == 200.0
        assert model.lambda_1 == 2.0
        assert model.alpha == 0.4
        assert model.eta_s == 50.0

    @pytest.mark.smoke
    def test_derived_properties(self):
        """Test derived properties are computed correctly."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 2.0)
        model.parameters.set_value("eta_s", 10.0)

        # η₀ = η_p + η_s
        assert model.eta_0 == 110.0

        # G = η_p / λ
        assert model.G == 50.0

        # Relaxation time alias
        assert model.relaxation_time == 2.0

    def test_dimensionless_numbers(self):
        """Test Weissenberg and Deborah number computations."""
        model = GiesekusSingleMode()
        model.parameters.set_value("lambda_1", 2.0)

        assert model.weissenberg_number(5.0) == 10.0  # Wi = λγ̇
        assert model.deborah_number(0.5) == 1.0  # De = λω

    def test_normal_stress_ratio(self):
        """Test theoretical N₂/N₁ ratio."""
        model = GiesekusSingleMode()

        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            model.parameters.set_value("alpha", alpha)
            assert model.get_normal_stress_ratio() == -alpha / 2


class TestFlowCurve:
    """Tests for flow curve (steady shear) predictions."""

    @pytest.mark.smoke
    def test_predict_flow_curve(self):
        """Test flow curve prediction via predict()."""
        model = GiesekusSingleMode()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict(gamma_dot, test_mode="flow_curve")

        assert sigma.shape == gamma_dot.shape
        assert np.all(sigma > 0)
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_predict_flow_curve_method(self):
        """Test direct predict_flow_curve method."""
        model = GiesekusSingleMode()

        gamma_dot = np.logspace(-2, 2, 20)
        sigma = model.predict_flow_curve(gamma_dot)

        assert sigma.shape == gamma_dot.shape

    def test_flow_curve_with_components(self):
        """Test flow curve with viscosity and N1."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        gamma_dot = np.logspace(-2, 2, 20)
        sigma, eta, N1 = model.predict_flow_curve(gamma_dot, return_components=True)

        assert sigma.shape == gamma_dot.shape
        assert eta.shape == gamma_dot.shape
        assert N1.shape == gamma_dot.shape
        assert np.all(eta > 0)
        assert np.all(N1 > 0)

    @pytest.mark.smoke
    def test_shear_thinning(self):
        """Test viscosity decreases with shear rate."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        gamma_dot = np.logspace(-2, 2, 20)
        _, eta, _ = model.predict_flow_curve(gamma_dot, return_components=True)

        # Viscosity should be monotonically decreasing
        for i in range(len(eta) - 1):
            assert eta[i] >= eta[i + 1] * 0.99  # Allow tiny numerical tolerance


class TestSAOS:
    """Tests for SAOS predictions."""

    @pytest.mark.smoke
    def test_predict_saos(self):
        """Test SAOS prediction."""
        model = GiesekusSingleMode()

        omega = np.logspace(-2, 2, 20)
        G_prime, G_double_prime = model.predict_saos(omega)

        assert G_prime.shape == omega.shape
        assert G_double_prime.shape == omega.shape
        assert np.all(G_prime >= 0)
        assert np.all(G_double_prime >= 0)

    @pytest.mark.smoke
    def test_saos_crossover(self):
        """Test G' = G'' at crossover frequency."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)
        model.parameters.set_value("eta_s", 0.0)

        omega_c = 1.0  # Crossover at ω = 1/λ
        G_prime, G_double_prime = model.predict_saos(np.array([omega_c]))

        assert np.isclose(G_prime[0], G_double_prime[0], rtol=0.01)

    def test_saos_magnitude(self):
        """Test |G*| prediction via test_mode='oscillation'."""
        model = GiesekusSingleMode()

        omega = np.logspace(-2, 2, 20)
        G_star = model.predict(omega, test_mode="oscillation")

        G_prime, G_double_prime = model.predict_saos(omega)
        expected = np.sqrt(G_prime**2 + G_double_prime**2)

        assert np.allclose(G_star, expected)


class TestNormalStresses:
    """Tests for normal stress predictions."""

    @pytest.mark.smoke
    def test_predict_normal_stresses(self):
        """Test N1 and N2 predictions."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        gamma_dot = np.array([1.0, 10.0, 100.0])
        N1, N2 = model.predict_normal_stresses(gamma_dot)

        assert N1.shape == gamma_dot.shape
        assert N2.shape == gamma_dot.shape
        assert np.all(N1 > 0)  # N1 always positive
        assert np.all(N2 < 0)  # N2 always negative

    @pytest.mark.smoke
    def test_n2_n1_ratio_prediction(self):
        """Test predicted N₂/N₁ equals -α/2."""
        model = GiesekusSingleMode()

        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
            model.parameters.set_value("alpha", alpha)

            gamma_dot = np.array([10.0])
            N1, N2 = model.predict_normal_stresses(gamma_dot)

            ratio = N2[0] / N1[0]
            expected = -alpha / 2

            assert np.isclose(ratio, expected, rtol=0.01)


class TestStartupSimulation:
    """Tests for startup flow simulation."""

    @pytest.mark.smoke
    def test_simulate_startup(self):
        """Test startup simulation runs."""
        model = GiesekusSingleMode()

        t = np.linspace(0, 5, 100)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_startup_overshoot(self):
        """Test stress overshoot in startup."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        t = np.linspace(0, 10, 200)
        sigma = model.simulate_startup(t, gamma_dot=10.0)

        # Find peak
        peak_idx = np.argmax(sigma)
        sigma_max = sigma[peak_idx]
        sigma_ss = sigma[-1]

        # Should have overshoot
        assert sigma_max > sigma_ss * 1.1

        # Peak should occur before steady state
        assert peak_idx < len(t) - 10

    def test_startup_full_return(self):
        """Test full stress tensor return."""
        model = GiesekusSingleMode()

        t = np.linspace(0, 5, 100)
        tau_xx, tau_yy, tau_xy, tau_zz = model.simulate_startup(
            t, gamma_dot=10.0, return_full=True
        )

        assert tau_xx.shape == t.shape
        assert tau_yy.shape == t.shape
        assert tau_xy.shape == t.shape
        assert tau_zz.shape == t.shape

    def test_startup_via_predict(self):
        """Test startup via predict() method."""
        model = GiesekusSingleMode()
        model._gamma_dot_applied = 10.0

        t = np.linspace(0, 5, 100)
        sigma = model.predict(t, test_mode="startup", gamma_dot=10.0)

        assert sigma.shape == t.shape


class TestRelaxationSimulation:
    """Tests for stress relaxation simulation."""

    @pytest.mark.smoke
    def test_simulate_relaxation(self):
        """Test relaxation simulation runs."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.2)  # Lower alpha for stability

        # Use moderate Wi to avoid overly stiff initial conditions
        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)  # Wi = 1

        assert sigma.shape == t.shape
        assert np.all(np.isfinite(sigma))

    @pytest.mark.smoke
    def test_relaxation_decay(self):
        """Test stress decays during relaxation."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.2)

        t = np.linspace(0, 5, 50)
        sigma = model.simulate_relaxation(t, gamma_dot_preshear=1.0)  # Wi = 1

        # Stress should decay monotonically
        assert sigma[0] > sigma[-1]

        # Final stress should approach zero (5 relaxation times)
        assert sigma[-1] < sigma[0] * 0.01


class TestCreepSimulation:
    """Tests for creep simulation."""

    @pytest.mark.smoke
    def test_simulate_creep(self):
        """Test creep simulation runs."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_s", 10.0)  # Need solvent for creep

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        assert gamma.shape == t.shape
        assert np.all(np.isfinite(gamma))

    @pytest.mark.smoke
    def test_creep_strain_increases(self):
        """Test strain increases during creep."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma = model.simulate_creep(t, sigma_applied=50.0)

        # Strain should be monotonically increasing
        assert gamma[-1] > gamma[0]

    def test_creep_with_rate(self):
        """Test creep with rate return."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_s", 10.0)

        t = np.linspace(0, 10, 100)
        gamma, gamma_dot = model.simulate_creep(t, sigma_applied=50.0, return_rate=True)

        assert gamma.shape == t.shape
        assert gamma_dot.shape == t.shape


class TestLAOSSimulation:
    """Tests for LAOS simulation."""

    @pytest.mark.smoke
    def test_simulate_laos(self):
        """Test LAOS simulation runs."""
        model = GiesekusSingleMode()

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)

        assert "t" in result
        assert "strain" in result
        assert "stress" in result
        assert "strain_rate" in result

    @pytest.mark.smoke
    def test_laos_periodicity(self):
        """Test LAOS response is periodic after transient."""
        model = GiesekusSingleMode()

        result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=5)

        # Check last two cycles have similar stress amplitude
        stress = result["stress"]
        n_per_cycle = len(stress) // 5

        cycle_4_max = np.max(np.abs(stress[3 * n_per_cycle : 4 * n_per_cycle]))
        cycle_5_max = np.max(np.abs(stress[4 * n_per_cycle :]))

        assert np.isclose(cycle_4_max, cycle_5_max, rtol=0.05)

    def test_laos_harmonics(self):
        """Test LAOS harmonic extraction."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        result = model.simulate_laos(t=None, gamma_0=1.0, omega=1.0, n_cycles=5)

        harmonics = model.extract_laos_harmonics(result, n_harmonics=3)

        assert "n" in harmonics
        assert "intensity" in harmonics
        assert len(harmonics["n"]) == 3
        assert harmonics["n"][0] == 1  # First harmonic
        assert harmonics["n"][1] == 3  # Third harmonic


class TestRegistryIntegration:
    """Tests for model registry integration."""

    @pytest.mark.smoke
    def test_registry_create(self):
        """Test model creation via registry."""
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create("giesekus")
        assert isinstance(model, GiesekusSingleMode)

        model2 = ModelRegistry.create("giesekus_single")
        assert isinstance(model2, GiesekusSingleMode)


class TestBayesianInterface:
    """Tests for BayesianMixin compatibility."""

    @pytest.mark.smoke
    def test_model_function(self):
        """Test model_function for BayesianMixin."""
        model = GiesekusSingleMode()

        X = jnp.logspace(-2, 2, 10)
        params = jnp.array([100.0, 1.0, 0.3, 0.0])  # eta_p, lambda_1, alpha, eta_s

        y = model.model_function(X, params, test_mode="flow_curve")

        assert y.shape == X.shape
        assert np.all(np.isfinite(y))

    def test_model_function_saos(self):
        """Test model_function for SAOS."""
        model = GiesekusSingleMode()

        X = jnp.logspace(-1, 2, 10)
        params = jnp.array([100.0, 1.0, 0.3, 10.0])

        y = model.model_function(X, params, test_mode="oscillation")

        assert y.shape == X.shape
        assert np.all(y > 0)


class TestFitting:
    """Tests for model fitting."""

    @pytest.mark.smoke
    def test_fit_flow_curve(self):
        """Test fitting to flow curve data."""
        # Create synthetic data
        model_true = GiesekusSingleMode()
        model_true.parameters.set_value("eta_p", 150.0)
        model_true.parameters.set_value("lambda_1", 0.5)
        model_true.parameters.set_value("alpha", 0.25)
        model_true.parameters.set_value("eta_s", 5.0)

        gamma_dot = np.logspace(-1, 2, 20)
        sigma_true = model_true.predict(gamma_dot, test_mode="flow_curve")

        # Add small noise
        np.random.seed(42)
        sigma_noisy = sigma_true * (1 + 0.02 * np.random.randn(len(sigma_true)))

        # Fit
        model_fit = GiesekusSingleMode()
        model_fit.fit(gamma_dot, sigma_noisy, test_mode="flow_curve")

        # Check fit quality
        sigma_pred = model_fit.predict(gamma_dot, test_mode="flow_curve")
        r2 = 1 - np.sum((sigma_noisy - sigma_pred) ** 2) / np.sum(
            (sigma_noisy - np.mean(sigma_noisy)) ** 2
        )

        assert r2 > 0.95

    @pytest.mark.slow
    def test_fit_saos(self):
        """Test fitting to SAOS data."""
        # Create synthetic data
        model_true = GiesekusSingleMode()
        model_true.parameters.set_value("eta_p", 100.0)
        model_true.parameters.set_value("lambda_1", 1.0)
        model_true.parameters.set_value("eta_s", 10.0)

        omega = np.logspace(-1, 2, 20)
        G_star_true = model_true.predict(omega, test_mode="oscillation")

        # Fit
        model_fit = GiesekusSingleMode()
        model_fit.fit(omega, G_star_true, test_mode="oscillation")

        # Check
        G_star_pred = model_fit.predict(omega, test_mode="oscillation")
        r2 = 1 - np.sum((G_star_true - G_star_pred) ** 2) / np.sum(
            (G_star_true - np.mean(G_star_true)) ** 2
        )

        assert r2 > 0.95


class TestAnalysisMethods:
    """Tests for analysis helper methods."""

    @pytest.mark.smoke
    def test_overshoot_ratio(self):
        """Test overshoot ratio calculation."""
        model = GiesekusSingleMode()
        model.parameters.set_value("alpha", 0.3)

        overshoot, strain_at_peak = model.get_overshoot_ratio(gamma_dot=10.0)

        # Should have overshoot > 1
        assert overshoot > 1.0
        # Strain at peak should be positive
        assert strain_at_peak > 0

    def test_relaxation_spectrum(self):
        """Test relaxation spectrum G(t)."""
        model = GiesekusSingleMode()
        model.parameters.set_value("eta_p", 100.0)
        model.parameters.set_value("lambda_1", 1.0)

        t, G_t = model.get_relaxation_spectrum(n_points=50)

        assert len(t) == 50
        assert len(G_t) == 50
        assert G_t[0] > G_t[-1]  # Should decay
