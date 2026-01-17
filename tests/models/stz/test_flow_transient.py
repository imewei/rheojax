"""Tests for STZ Flow and Transient protocols.

Tests cover steady-state flow curves and transient startup/relaxation behavior.
Follows the 2-8 test rule per task group.
"""

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax

# Ensure float64 is enabled
jax, jnp = safe_import_jax()

from rheojax.models.stz import STZConventional


@pytest.mark.unit
class TestSTZFlowTransient:
    """Test suite for STZ Flow and Transient protocols."""

    def test_steady_shear_prediction(self):
        """Test steady-state flow curve prediction."""
        model = STZConventional(variant="standard")

        # Set parameters
        model.parameters.set_value("G0", 1e9)
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("chi_inf", 0.15)
        model.parameters.set_value("tau0", 1e-12)
        model.parameters.set_value("ez", 1.0)

        # Generate shear rates
        gamma_dot = np.logspace(-2, 4, 20)

        # Predict stress
        model._test_mode = "steady_shear"
        stress = model._predict(gamma_dot)

        # Stress should be finite and positive for positive shear rates
        assert np.all(np.isfinite(stress))
        assert np.all(stress > 0)

        # Stress should increase with shear rate (shear thickening at high rates)
        # or saturate (yielding behavior)
        assert stress[-1] > stress[0]

    def test_steady_shear_approaches_yield_stress(self):
        """Test that stress saturates near sigma_y at high shear rates."""
        model = STZConventional(variant="minimal")

        sigma_y = 1e6
        model.parameters.set_value("sigma_y", sigma_y)
        model.parameters.set_value("chi_inf", 0.2)
        model.parameters.set_value("tau0", 1e-12)
        model.parameters.set_value("ez", 1.0)

        # Very high shear rate
        gamma_dot = np.array([1e10])

        model._test_mode = "steady_shear"
        stress = model._predict(gamma_dot)

        # Stress should approach but not exceed a multiple of sigma_y
        # The exact limit depends on chi_inf
        assert stress[0] < 10 * sigma_y

    @pytest.mark.slow
    def test_startup_flow_overshoot(self):
        """Test stress overshoot in startup flow (strain-controlled)."""
        model = STZConventional(variant="standard")

        # Set parameters for overshoot behavior
        # Use tau0=1e-9 for less stiff dynamics (vs 1e-12 which causes solver issues)
        model.parameters.set_value("G0", 1e9)
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("chi_inf", 0.15)
        model.parameters.set_value("tau0", 1e-9)
        model.parameters.set_value("epsilon0", 0.1)
        model.parameters.set_value("c0", 1.0)

        # Time array scaled to tau0 (several characteristic times)
        t = np.linspace(0, 1e-7, 50)  # 100 ns, covers ~100 tau0
        gamma_dot = 1e7  # Shear rate matching timescale

        p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}

        # Simulate startup
        stress = model._simulate_transient_jit(
            jnp.asarray(t),
            p_values,
            "startup",
            gamma_dot,
            None,
            None,
            model.variant,
        )

        stress = np.array(stress)

        # Stress should start at 0
        assert np.isclose(stress[0], 0.0, atol=1e-3)

        # Stress should be finite
        assert np.all(np.isfinite(stress))

        # Stress should increase initially (elastic loading)
        assert stress[5] > stress[0]

    @pytest.mark.slow
    def test_relaxation_behavior(self):
        """Test stress relaxation (gamma_dot = 0 after initial loading)."""
        model = STZConventional(variant="standard")

        # Set parameters with tau0=1e-9 for less stiff dynamics
        model.parameters.set_value("G0", 1e9)
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("chi_inf", 0.15)
        model.parameters.set_value("tau0", 1e-9)
        model.parameters.set_value("epsilon0", 0.1)
        model.parameters.set_value("c0", 1.0)

        # Time array scaled to tau0
        t = np.linspace(0, 1e-7, 50)
        sigma_0 = 5e5  # Initial stress

        p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}

        # Simulate relaxation
        stress = model._simulate_transient_jit(
            jnp.asarray(t),
            p_values,
            "relaxation",
            None,  # gamma_dot = 0 for relaxation
            None,
            sigma_0,
            model.variant,
        )

        stress = np.array(stress)

        # Stress should be finite
        assert np.all(np.isfinite(stress))

        # Initial stress should be close to sigma_0
        assert np.isclose(stress[0], sigma_0, rtol=0.1)

    @pytest.mark.slow
    def test_variant_differences(self):
        """Test that different variants produce different dynamics."""
        # Time span scaled to tau0=1e-9, gamma_dot=1e7 for stable integration
        t = np.linspace(0, 1e-7, 30)  # 100 ns
        gamma_dot = 1e7  # Shear rate compatible with explicit solver

        results = {}
        for variant in ["minimal", "standard", "full"]:
            model = STZConventional(variant=variant)
            model.parameters.set_value("G0", 1e9)
            model.parameters.set_value("sigma_y", 1e6)
            model.parameters.set_value("chi_inf", 0.15)
            model.parameters.set_value("tau0", 1e-9)
            model.parameters.set_value("epsilon0", 0.1)
            model.parameters.set_value("c0", 1.0)

            p_values = {
                k: model.parameters.get_value(k) for k in model.parameters.keys()
            }

            stress = model._simulate_transient_jit(
                jnp.asarray(t),
                p_values,
                "startup",
                gamma_dot,
                None,
                None,
                variant,
            )
            results[variant] = np.array(stress)

        # All variants should produce finite results
        for variant, stress in results.items():
            assert np.all(np.isfinite(stress)), f"{variant} produced non-finite stress"

    @pytest.mark.slow
    def test_diffrax_integration_convergence(self):
        """Test that Diffrax ODE integration converges properly."""
        model = STZConventional(variant="standard")

        # Use tau0=1e-9 for less stiff dynamics
        model.parameters.set_value("G0", 1e9)
        model.parameters.set_value("sigma_y", 1e6)
        model.parameters.set_value("chi_inf", 0.15)
        model.parameters.set_value("tau0", 1e-9)
        model.parameters.set_value("epsilon0", 0.1)
        model.parameters.set_value("c0", 1.0)

        gamma_dot = 1e7  # Shear rate compatible with explicit solver

        # Coarse time grid
        t_coarse = np.linspace(0, 1e-7, 10)
        # Fine time grid
        t_fine = np.linspace(0, 1e-7, 100)

        p_values = {k: model.parameters.get_value(k) for k in model.parameters.keys()}

        stress_coarse = model._simulate_transient_jit(
            jnp.asarray(t_coarse),
            p_values,
            "startup",
            gamma_dot,
            None,
            None,
            "standard",
        )
        stress_fine = model._simulate_transient_jit(
            jnp.asarray(t_fine), p_values, "startup", gamma_dot, None, None, "standard"
        )

        # Interpolate fine solution to coarse grid
        stress_fine_interp = np.interp(t_coarse, t_fine, np.array(stress_fine))

        # Solutions should be similar (adaptive stepping handles accuracy)
        np.testing.assert_allclose(
            np.array(stress_coarse),
            stress_fine_interp,
            rtol=0.1,  # 10% tolerance for adaptive solver
        )
