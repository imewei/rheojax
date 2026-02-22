import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheojax.models.hl import HebraudLequeux


class TestHebraudLequeux:
    @pytest.fixture
    def model(self):
        """Create a standard HL model instance."""
        m = HebraudLequeux()
        # Set to glassy phase
        m.parameters.set_value("alpha", 0.3)
        m.parameters.set_value("tau", 1.0)
        m.parameters.set_value("sigma_c", 1.0)
        return m

    def test_instantiation(self, model):
        """Test model initialization and parameter defaults."""
        assert model.parameters.get_value("alpha") == 0.3
        assert model.parameters.get_value("tau") == 1.0
        assert model.parameters.get_value("sigma_c") == 1.0
        assert model.get_phase_state() == "glass"

    def test_phase_state(self, model):
        """Test phase classification."""
        model.parameters.set_value("alpha", 0.6)
        assert model.get_phase_state() == "fluid"
        model.parameters.set_value("alpha", 0.3)
        assert model.get_phase_state() == "glass"

    def test_flow_curve_prediction(self, model):
        """Test steady shear flow curve prediction."""
        # Use higher shear rates for faster test (fewer time steps needed)
        gdot = np.logspace(0, 1, 5)  # 1.0 to 10.0
        # Use synthetic data for mock fit to allow convergence
        synthetic_stress = np.array([0.5, 0.6, 0.8, 1.2, 2.0])  # Rough HB shape
        model.fit(gdot, synthetic_stress, test_mode="steady_shear", max_iter=2)
        stress = model.predict(gdot)

        assert stress.shape == gdot.shape
        assert np.all(np.isfinite(stress))
        # Stress should increase with shear rate
        assert np.all(np.diff(stress) > 0)
        # Check yield stress existence (stress > 0)
        assert stress[0] > 0.0

    def test_creep_prediction(self, model):
        """Test creep compliance prediction."""
        # Short duration for test speed
        t = np.linspace(0, 1, 10)
        # Mock fit to set mode and context
        synthetic_compliance = t * 0.1  # Mock linear compliance
        model.fit(
            t, synthetic_compliance, test_mode="creep", stress_target=0.5, max_iter=2
        )
        compliance = model.predict(t)

        assert compliance.shape == t.shape
        assert np.all(np.isfinite(compliance))
        # Compliance should be non-negative
        assert np.all(compliance >= 0)

    def test_relaxation_prediction(self, model):
        """Test stress relaxation prediction."""
        # Short duration
        t = np.linspace(0, 1, 10)
        # Mock fit
        synthetic_modulus = np.exp(-t)
        model.fit(t, synthetic_modulus, test_mode="relaxation", gamma0=0.1, max_iter=2)
        G_t = model.predict(t)

        assert G_t.shape == t.shape
        assert np.all(np.isfinite(G_t))
        assert G_t[0] > 0

    def test_startup_prediction(self, model):
        """Test startup stress prediction."""
        t = np.linspace(0, 1, 10)
        synthetic_stress = 1.0 - np.exp(-t)
        model.fit(t, synthetic_stress, test_mode="startup", gdot=1.0, max_iter=2)
        stress = model.predict(t)

        assert stress.shape == t.shape
        assert np.all(np.isfinite(stress))
        assert stress[0] == 0  # Starts at 0

    def test_laos_prediction(self, model):
        """Test LAOS stress prediction."""
        t = np.linspace(0, 1, 20)
        synthetic_stress = np.sin(t)
        model.fit(
            t, synthetic_stress, test_mode="laos", gamma0=1.0, omega=10.0, max_iter=2
        )
        stress = model.predict(t)

        assert stress.shape == t.shape
        assert np.all(np.isfinite(stress))

    def test_grid_scaling(self, model):
        """Test that grid adapts to large sigma_c."""
        model.parameters.set_value("sigma_c", 100.0)
        gdot = np.array([0.1, 1.0])

        # We just want to verify prediction works and returns large values
        # No need to fit, just set the test mode and protocol kwargs manually
        model._test_mode = "steady_shear"
        model._last_fit_kwargs = {
            "_sigma_max": max(5.0, model.grid_sigma_factor * 100.0),
            "_n_bins": 501,
        }

        # Predict
        stress = model.predict(gdot)

        # With sigma_c=100, stress should be > 10
        # If clipped to grid [-5, 5], it would be ~5
        assert stress[0] > 10.0
        assert np.all(np.isfinite(stress))

    def test_bayesian_interface(self, model):
        """Test model_function for Bayesian inference compatibility."""
        gdot = np.array([0.1, 1.0, 10.0])
        params = jnp.array([0.3, 1.0, 1.0])  # alpha, tau, sigma_c

        # Manually setup context to avoid running expensive/unstable fit
        model._test_mode = "steady_shear"
        model._last_fit_kwargs = {}
        # Metadata needed for some modes, but steady_shear might be fine without t_max
        model._fit_data_metadata = {"t_max": 10.0, "len_X": 3}

        # Call model_function
        pred = model.model_function(gdot, params)
        assert pred.shape == gdot.shape
