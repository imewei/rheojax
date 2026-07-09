"""
Tests for SGRGeneric model using GENERIC thermodynamic framework.

This test module validates the SGRGeneric model implementation based on
Fuereder & Ilg (2013) Physical Review E 88, 042134, ensuring thermodynamic
consistency and correct GENERIC structure (Poisson bracket + friction matrix).

The GENERIC framework splits dynamics into:
- Reversible (Hamiltonian): dz/dt = L * dF/dz  (Poisson bracket L antisymmetric)
- Irreversible (dissipative): dz/dt = M * dS/dz (friction M symmetric positive semi-definite)

Key thermodynamic constraints tested:
- Entropy production W = (dF/dz)^T M (dF/dz) >= 0 (second law)
- Poisson bracket antisymmetry: L = -L^T
- Friction matrix symmetry and positive semi-definiteness: M = M^T, eigenvalues >= 0
- Energy conservation in reversible part
- Entropy balance dS/dt >= 0
"""

import numpy as np
import pytest

from rheojax.core.parameters import ParameterSet
from rheojax.core.registry import ModelRegistry
from rheojax.models import SGRGeneric


class TestSGRGenericThermodynamics:
    """Test suite for GENERIC thermodynamic consistency."""

    @pytest.mark.smoke
    def test_thermodynamic_consistency_entropy_production_nonnegative(self):
        """Test that entropy production W >= 0 always (second law compliance)."""
        model = SGRGeneric()

        # Set parameters in power-law regime
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test various state configurations
        # State vector contains momentum (stress-related) and structural variables
        test_states = [
            np.array([0.0, 0.5]),  # Zero stress, partial structure
            np.array([100.0, 0.8]),  # Moderate stress, high structure
            np.array([1000.0, 0.2]),  # High stress, low structure
            np.array([-500.0, 0.6]),  # Negative stress (compression)
            np.array([0.0, 1.0]),  # Zero stress, full structure
            np.array([50.0, 0.01]),  # Low stress, near-empty structure
        ]

        for state in test_states:
            # Compute entropy production
            W = model.compute_entropy_production(state)

            # Second law: W >= 0
            assert W >= -1e-12, (
                f"Entropy production W={W:.6e} < 0 violates second law for state={state}"
            )

            # Check for finite value
            assert np.isfinite(W), f"Entropy production not finite for state={state}"

    def test_poisson_bracket_antisymmetry(self):
        """Test Poisson bracket L is antisymmetric: L = -L^T."""
        model = SGRGeneric()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test at various states
        test_states = [
            np.array([0.0, 0.5]),
            np.array([100.0, 0.8]),
            np.array([1000.0, 0.2]),
        ]

        for state in test_states:
            # Get Poisson bracket matrix
            L = model.poisson_bracket(state)

            # Check antisymmetry: L = -L^T
            L_T = L.T
            antisymmetry_error = np.max(np.abs(L + L_T))

            assert antisymmetry_error < 1e-12, (
                f"Poisson bracket not antisymmetric: max|L + L^T| = {antisymmetry_error:.6e} for state={state}"
            )

            # Diagonal should be zero for antisymmetric matrix
            diag_error = np.max(np.abs(np.diag(L)))
            assert diag_error < 1e-12, (
                f"Poisson bracket diagonal not zero: max|diag(L)| = {diag_error:.6e}"
            )

    def test_friction_matrix_symmetry_and_positive_semidefinite(self):
        """Test friction matrix M is symmetric and positive semi-definite."""
        model = SGRGeneric()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test at various states
        test_states = [
            np.array([0.0, 0.5]),
            np.array([100.0, 0.8]),
            np.array([1000.0, 0.2]),
            np.array([-500.0, 0.6]),
        ]

        for state in test_states:
            # Get friction matrix
            M = model.friction_matrix(state)

            # Check symmetry: M = M^T
            M_T = M.T
            symmetry_error = np.max(np.abs(M - M_T))
            assert symmetry_error < 1e-12, (
                f"Friction matrix not symmetric: max|M - M^T| = {symmetry_error:.6e} for state={state}"
            )

            # Check positive semi-definiteness via eigenvalues
            eigenvalues = np.linalg.eigvalsh(M)  # Real eigenvalues for symmetric matrix

            # All eigenvalues must be >= 0
            min_eigenvalue = np.min(eigenvalues)
            assert min_eigenvalue >= -1e-12, (
                f"Friction matrix not positive semi-definite: min eigenvalue = {min_eigenvalue:.6e} for state={state}"
            )

    def test_energy_conservation_reversible_dynamics(self):
        """Test energy conservation: dE/dt = power input for reversible part."""
        model = SGRGeneric()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Initial state
        state = np.array([100.0, 0.5])

        # Compute free energy at current state
        F_initial = model.free_energy(state)

        # Get reversible dynamics (Hamiltonian part)
        dz_dt_rev = model.reversible_dynamics(state)

        # For isolated system (no external power input), reversible dynamics
        # should conserve energy: dF/dt = (dF/dz)^T * dz_dt_rev = 0
        dF_dz = model.free_energy_gradient(state)
        energy_rate_rev = np.dot(dF_dz, dz_dt_rev)

        # Energy should be conserved (rate = 0) in reversible part
        # Allow small numerical tolerance (numerical precision limits)
        assert np.abs(energy_rate_rev) < 1e-9, (
            f"Reversible dynamics not energy-conserving: dF/dt = {energy_rate_rev:.6e}"
        )

    def test_entropy_balance_positive_production(self):
        """Test entropy balance: dS/dt >= 0 (entropy increases or stays constant)."""
        model = SGRGeneric()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test various non-equilibrium states
        test_states = [
            np.array([100.0, 0.5]),  # Non-equilibrium
            np.array([500.0, 0.3]),  # Stressed state
            np.array([0.0, 0.8]),  # Relaxed but structured
        ]

        for state in test_states:
            # Compute entropy rate from irreversible dynamics
            dS_dt = model.entropy_production_rate(state)

            # Second law: dS/dt >= 0
            assert dS_dt >= -1e-12, (
                f"Entropy production rate dS/dt = {dS_dt:.6e} < 0 violates second law for state={state}"
            )


class TestSGRGenericModelComparison:
    """Test suite for SGRGeneric vs SGRConventional equivalence."""

    def test_equivalence_to_sgr_conventional_linear_regime(self):
        """Test SGRGeneric matches SGRConventional predictions in linear regime."""
        from rheojax.models import SGRConventional

        # Create both models with same parameters
        generic = SGRGeneric()
        conventional = SGRConventional()

        # Set identical parameters
        x_val = 1.5
        G0_val = 1e3
        tau0_val = 1e-3

        generic.parameters.set_value("x", x_val)
        generic.parameters.set_value("G0", G0_val)
        generic.parameters.set_value("tau0", tau0_val)

        conventional.parameters.set_value("x", x_val)
        conventional.parameters.set_value("G0", G0_val)
        conventional.parameters.set_value("tau0", tau0_val)

        # Test oscillation mode predictions
        generic._test_mode = "oscillation"
        conventional._test_mode = "oscillation"

        omega = np.logspace(0, 4, 20)

        G_star_generic = generic.predict(omega)
        G_star_conventional = conventional.predict(omega)

        # Check shapes match
        assert G_star_generic.shape == G_star_conventional.shape, (
            f"Shape mismatch: generic={G_star_generic.shape}, conventional={G_star_conventional.shape}"
        )

        # Check predictions match within tolerance
        # GENERIC formulation should give same linear response as conventional
        relative_error_G_prime = np.max(
            np.abs(np.real(G_star_generic) - np.real(G_star_conventional))
            / (np.abs(np.real(G_star_conventional)) + 1e-10)
        )
        relative_error_G_double_prime = np.max(
            np.abs(np.imag(G_star_generic) - np.imag(G_star_conventional))
            / (np.abs(np.imag(G_star_conventional)) + 1e-10)
        )

        # Allow 10% tolerance for different numerical implementations
        assert relative_error_G_prime < 0.1, (
            f"G' mismatch: max relative error = {relative_error_G_prime:.4f}"
        )
        assert relative_error_G_double_prime < 0.1, (
            f"G'' mismatch: max relative error = {relative_error_G_double_prime:.4f}"
        )


class TestSGRGenericFreeEnergy:
    """Test suite for free energy functional F(z)."""

    def test_free_energy_computation(self):
        """Test free energy F(state) computation is well-defined."""
        model = SGRGeneric()

        # Set parameters
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 1e3)
        model.parameters.set_value("tau0", 1e-3)

        # Test various states
        test_states = [
            np.array([0.0, 0.5]),
            np.array([100.0, 0.8]),
            np.array([1000.0, 0.2]),
            np.array([-500.0, 0.6]),
        ]

        for state in test_states:
            # Compute free energy
            F = model.free_energy(state)

            # Check finite value
            assert np.isfinite(F), f"Free energy not finite for state={state}"

            # Free energy should be real
            assert np.isreal(F), f"Free energy not real for state={state}"

        # Test free energy decomposition: F = U - T*S
        state = np.array([100.0, 0.5])
        F = model.free_energy(state)
        U = model.internal_energy(state)
        S = model.entropy(state)
        T = model.parameters.get_value("x")  # Noise temperature

        # Check decomposition
        F_expected = U - T * S
        decomposition_error = np.abs(F - F_expected)
        assert decomposition_error < 1e-10, (
            f"Free energy decomposition error: |F - (U - T*S)| = {decomposition_error:.6e}"
        )


class TestSGRGenericModelInterface:
    """Test suite for BaseModel interface compliance."""

    def test_model_registration_and_base_model_interface(self):
        """Test model registration and BaseModel interface."""
        from rheojax.core.base import BaseModel

        # Check model is registered
        registered_models = ModelRegistry.list_models()
        assert "sgr_generic" in registered_models, (
            "SGRGeneric not registered in ModelRegistry"
        )

        # Check factory pattern works
        model = ModelRegistry.create("sgr_generic")
        assert isinstance(model, SGRGeneric), (
            f"Factory created wrong type: {type(model)}"
        )

        # Check BaseModel inheritance
        assert isinstance(model, BaseModel), "SGRGeneric should inherit from BaseModel"

        # Check ParameterSet creation
        assert isinstance(model.parameters, ParameterSet), (
            "model.parameters should be ParameterSet"
        )

        # Check required parameters exist
        assert "x" in model.parameters.keys(), "Missing parameter 'x'"
        assert "G0" in model.parameters.keys(), "Missing parameter 'G0'"
        assert "tau0" in model.parameters.keys(), "Missing parameter 'tau0'"

        # Check parameter bounds
        x_param = model.parameters.get("x")
        assert x_param.bounds == (0.5, 3.0), f"x bounds incorrect: {x_param.bounds}"

        # Check BaseModel interface methods exist
        assert hasattr(model, "fit"), "Missing fit() method"
        assert hasattr(model, "predict"), "Missing predict() method"
        assert hasattr(model, "_fit"), "Missing _fit() method"
        assert hasattr(model, "_predict"), "Missing _predict() method"
        assert hasattr(model, "model_function"), "Missing model_function() method"

        # Check BayesianMixin integration
        assert hasattr(model, "fit_bayesian"), "Missing fit_bayesian() method"

        # Test prediction interface works
        model._test_mode = "oscillation"
        omega = np.logspace(0, 3, 10)
        G_star = model.predict(omega)

        assert G_star.shape == (10,), f"Prediction shape incorrect: {G_star.shape}"
        assert np.iscomplexobj(G_star), "Predictions should be complex G*"
        assert not np.any(np.isnan(G_star)), "Predictions contain NaN"
        assert not np.any(np.isinf(G_star)), "Predictions contain Inf"


class TestSGRGenericFitting:
    """Test suite for SGRGeneric NLSQ fitting (Task Group 3)."""

    def test_oscillation_mode_fitting_basic(self):
        """Test basic oscillation mode fitting with synthetic data."""
        # Generate synthetic data
        model_true = SGRGeneric()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "oscillation"

        omega = np.logspace(-2, 2, 50)
        G_star = model_true.predict(omega)

        # Fit model
        model_fit = SGRGeneric()
        model_fit.parameters.set_value("x", 2.0)
        model_fit.parameters.set_value("G0", 500.0)
        model_fit.parameters.set_value("tau0", 0.01)

        model_fit.fit(omega, G_star, test_mode="oscillation")

        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "oscillation"

    def test_relaxation_mode_fitting_basic(self):
        """Test basic relaxation mode fitting with synthetic data."""
        # Generate synthetic data
        model_true = SGRGeneric()
        model_true.parameters.set_value("x", 1.5)
        model_true.parameters.set_value("G0", 1000.0)
        model_true.parameters.set_value("tau0", 0.001)
        model_true._test_mode = "relaxation"

        t = np.logspace(-3, 2, 50)
        G_t = model_true.predict(t)

        # Fit model
        model_fit = SGRGeneric()
        model_fit.fit(t, G_t, test_mode="relaxation")

        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "relaxation"

    def test_fitting_stores_test_mode_for_bayesian(self):
        """Test that fitting stores test_mode for mode-aware Bayesian inference."""
        model = SGRGeneric()
        assert model._test_mode is None

        # Generate data and fit
        omega = np.logspace(-1, 1, 20)
        model.parameters.set_value("x", 1.5)
        model._test_mode = "oscillation"
        G_star = model.predict(omega)

        model2 = SGRGeneric()
        model2.fit(omega, G_star, test_mode="oscillation")

        # test_mode should be stored
        assert model2._test_mode == "oscillation"

    def test_fitting_invalid_test_mode_raises(self):
        """Test that invalid test_mode raises ValueError."""
        model = SGRGeneric()
        omega = np.logspace(-1, 1, 20)
        G_star = np.ones((20, 2))

        with pytest.raises(ValueError, match="test_mode must be specified"):
            model.fit(omega, G_star)

        with pytest.raises(ValueError, match="Unsupported test_mode"):
            model.fit(omega, G_star, test_mode="invalid_mode")


def _make_model(x=1.5, G0=1000.0, tau0=1e-3):
    """Helper: SGRGeneric with power-law-fluid parameters."""
    m = SGRGeneric()
    m.parameters.set_value("x", x)
    m.parameters.set_value("G0", G0)
    m.parameters.set_value("tau0", tau0)
    return m


class TestSGRGenericAdditionalFitModes:
    """Fitting in creep, steady-shear, and startup protocols (noiseless data)."""

    def test_creep_fit_and_predict_roundtrip(self):
        model_true = _make_model()
        model_true._test_mode = "creep"
        t = np.logspace(-3, 2, 40)
        J = model_true.predict(t)

        # J(t) ~ (1 + t/tau0)^(x-1): positive, monotonically increasing, finite
        assert np.all(J > 0)
        assert np.all(np.diff(J) >= 0)
        assert np.all(np.isfinite(J))

        model_fit = _make_model(x=1.2, G0=500.0, tau0=1e-2)
        model_fit.fit(t, J, test_mode="creep")
        assert model_fit.fitted_ is True
        assert model_fit._test_mode == "creep"
        np.testing.assert_allclose(model_fit.predict(t), J, rtol=1e-3, atol=1e-8)

    def test_steady_shear_fit_and_predict_roundtrip(self):
        model_true = _make_model()
        model_true._test_mode = "steady_shear"
        gamma_dot = np.logspace(-2, 2, 40)
        sigma = model_true.predict(gamma_dot)

        # Returns real stress (not complex), sigma ~ gamma_dot^(x-1) increasing for x>1
        assert not np.iscomplexobj(sigma)
        assert np.all(sigma > 0)
        assert np.all(np.diff(sigma) >= 0)

        model_fit = _make_model(x=1.2, G0=500.0, tau0=1e-2)
        model_fit.fit(gamma_dot, sigma, test_mode="steady_shear")
        assert model_fit.fitted_ is True
        np.testing.assert_allclose(
            model_fit.predict(gamma_dot), sigma, rtol=1e-3, atol=1e-8
        )

    def test_flow_curve_alias_routes_to_steady_shear(self):
        model = _make_model()
        gamma_dot = np.logspace(-2, 2, 20)
        model._test_mode = "steady_shear"
        sigma = model.predict(gamma_dot)

        model_fit = _make_model()
        model_fit.fit(gamma_dot, sigma, test_mode="flow_curve")
        assert model_fit.fitted_ is True

    def test_startup_fit_and_predict_roundtrip(self):
        model_true = _make_model()
        model_true._startup_gamma_dot = 1.0
        model_true._test_mode = "startup"
        t = np.logspace(-3, 2, 40)
        eta_plus = model_true.predict(t)

        assert np.all(np.isfinite(eta_plus))
        assert np.all(eta_plus >= 0)

        model_fit = _make_model()
        model_fit.fit(t, eta_plus, test_mode="startup", gamma_dot=1.0)
        assert model_fit.fitted_ is True
        assert model_fit._startup_gamma_dot == 1.0
        np.testing.assert_allclose(
            model_fit.predict(t), eta_plus, rtol=1e-3, atol=1e-8
        )

    def test_startup_fit_from_stress_input(self):
        """is_stress=True divides applied stress by gamma_dot before fitting."""
        model_true = _make_model()
        model_true._startup_gamma_dot = 2.0
        model_true._test_mode = "startup"
        t = np.logspace(-3, 1, 30)
        eta_plus = model_true.predict(t)
        stress = eta_plus * 2.0  # sigma = eta_plus * gamma_dot

        model_fit = _make_model()
        model_fit.fit(t, stress, test_mode="startup", gamma_dot=2.0, is_stress=True)
        assert model_fit.fitted_ is True


class TestSGRGenericFitValidation:
    """Input validation raised before optimization begins."""

    def test_oscillation_bad_gstar_shape_raises(self):
        model = SGRGeneric()
        with pytest.raises(ValueError, match="G_star must be complex"):
            model.fit(np.logspace(0, 2, 10), np.ones((10, 3)), test_mode="oscillation")

    def test_laos_fit_requires_gamma0_and_omega(self):
        model = SGRGeneric()
        t = np.linspace(0, 4 * np.pi, 100)
        with pytest.raises(ValueError, match="requires gamma_0 and omega"):
            model.fit(t, np.sin(t), test_mode="laos")

    def test_laos_fit_rejects_nonpositive_gamma0(self):
        model = SGRGeneric()
        t = np.linspace(0, 4 * np.pi, 100)
        with pytest.raises(ValueError, match="must be positive"):
            model.fit(t, np.sin(t), test_mode="laos", gamma_0=-1.0, omega=1.0)


class TestSGRGenericLAOSFitBugs:
    """LAOS fitting via the public fit() API is currently broken (bug report).

    Both branches of _fit_laos_mode forward **kwargs that still contain the
    positional argument names (omega / gamma_0 / n_particles), so the inner
    call raises TypeError('got multiple values for argument ...').
    """

    @pytest.mark.xfail(
        reason="_fit_laos_mode passes omega in **kwargs to _fit_oscillation_mode "
        "(positional collision); small-amplitude LAOS fit is broken.",
        strict=True,
    )
    def test_laos_saos_branch_small_amplitude(self):
        model = _make_model()
        model._test_mode = "oscillation"
        omega = 1.0
        gamma_0 = 0.05  # < 0.1 -> SAOS branch
        t = np.linspace(0, 4 * np.pi, 400)
        strain = gamma_0 * np.sin(omega * t)
        stress = 500.0 * strain
        model.fit(t, stress, test_mode="laos", gamma_0=gamma_0, omega=omega)
        assert model.fitted_ is True

    @pytest.mark.xfail(
        reason="_fit_laos_mode passes gamma_0/omega/n_particles in **kwargs to "
        "_fit_laos_mc (positional collision); MC LAOS fit is broken.",
        strict=True,
    )
    def test_laos_mc_branch_large_amplitude(self):
        model = _make_model()
        omega = 2.0
        gamma_0 = 0.5  # >= 0.1 -> MC branch
        t = np.linspace(0, 2 * 2 * np.pi / omega, 200)
        stress = 500.0 * gamma_0 * np.sin(omega * t)
        model.fit(
            t, stress, test_mode="laos", gamma_0=gamma_0, omega=omega,
            n_particles=50, max_iter=1,
        )
        assert model.fitted_ is True


class TestSGRGenericPredictJIT:
    """Direct JIT predictors and the LAOS predict route."""

    def test_predict_viscosity_jit_shear_thinning(self):
        model = _make_model()
        gamma_dot = np.logspace(-2, 2, 20)
        from rheojax.core.jax_config import safe_import_jax

        _, jnp = safe_import_jax()
        eta = np.asarray(
            model._predict_viscosity_jit(jnp.asarray(gamma_dot), 1.5, 1000.0, 1e-3)
        )
        # x=1.5 < 2 -> shear-thinning: eta decreases with gamma_dot
        assert np.all(np.isfinite(eta))
        assert np.all(eta > 0)
        assert np.all(np.diff(eta) <= 0)

    def test_predict_laos_mode_via_predict(self):
        model = _make_model()
        model._test_mode = "laos"
        model._gamma_0 = 0.2
        model._omega_laos = 1.0
        X = np.linspace(0, 4 * np.pi, 50)
        stress = model.predict(X)
        assert stress.shape == (50,)
        assert np.all(np.isfinite(stress))


class TestSGRGenericModelFunction:
    """model_function() routing for Bayesian inference across modes."""

    @pytest.mark.parametrize("mode", ["oscillation", "relaxation", "steady_shear", "creep"])
    def test_model_function_basic_modes(self, mode):
        model = SGRGeneric()
        params = np.array([1.5, 1000.0, 1e-3])
        X = np.logspace(-2, 2, 25)
        out = np.asarray(model.model_function(X, params, test_mode=mode))
        assert np.all(np.isfinite(out))
        if mode == "oscillation":
            assert out.shape == (25, 2)
        else:
            assert out.shape == (25,)

    def test_model_function_defaults_to_oscillation(self):
        model = SGRGeneric()
        assert model._test_mode is None
        params = np.array([1.5, 1000.0, 1e-3])
        out = np.asarray(model.model_function(np.logspace(0, 2, 10), params))
        assert out.shape == (10, 2)

    def test_model_function_startup_requires_gamma_dot(self):
        model = SGRGeneric()
        params = np.array([1.5, 1000.0, 1e-3])
        with pytest.raises(RuntimeError, match="gamma_dot not provided"):
            model.model_function(np.logspace(-2, 1, 10), params, test_mode="startup")

    def test_model_function_startup_with_gamma_dot(self):
        model = SGRGeneric()
        params = np.array([1.5, 1000.0, 1e-3])
        out = np.asarray(
            model.model_function(
                np.logspace(-2, 1, 10), params, test_mode="startup", gamma_dot=2.0
            )
        )
        assert out.shape == (10,)
        assert np.all(np.isfinite(out))

    def test_model_function_laos_not_implemented(self):
        model = SGRGeneric()
        params = np.array([1.5, 1000.0, 1e-3])
        with pytest.raises(NotImplementedError, match="LAOS"):
            model.model_function(np.logspace(0, 2, 10), params, test_mode="laos")

    def test_model_function_unsupported_mode_raises(self):
        model = SGRGeneric()
        params = np.array([1.5, 1000.0, 1e-3])
        with pytest.raises(ValueError, match="Unsupported test mode"):
            model.model_function(np.logspace(0, 2, 10), params, test_mode="nope")


class TestSGRGenericPredictErrors:
    """Prediction-side error handling."""

    def test_predict_without_test_mode_raises(self):
        model = SGRGeneric()
        with pytest.raises(ValueError, match="test_mode must be specified"):
            model.predict(np.array([1.0, 2.0]))

    def test_predict_unknown_test_mode_raises(self):
        model = _make_model()
        model._test_mode = "not_a_real_mode"
        with pytest.raises(ValueError, match="Unknown test_mode"):
            model.predict(np.logspace(0, 2, 10))

    def test_predict_startup_unfitted_raises(self):
        model = _make_model()
        model._test_mode = "startup"  # but no _startup_gamma_dot cached
        with pytest.raises(RuntimeError, match="_startup_gamma_dot"):
            model.predict(np.logspace(-2, 1, 10))


class TestSGRGenericPhaseRegime:
    """Phase-regime classification from noise temperature x."""

    @pytest.mark.parametrize(
        "x,expected",
        [(0.8, "glass"), (1.0, "power-law"), (1.5, "power-law"), (2.0, "newtonian"), (2.5, "newtonian")],
    )
    def test_phase_regime(self, x, expected):
        model = SGRGeneric()
        model.parameters.set_value("x", x)
        assert model.get_phase_regime() == expected


class TestSGRGenericDynamicsSplit:
    """Reversible / irreversible / full GENERIC dynamics."""

    def test_dynamics_components_finite(self):
        model = _make_model()
        for state in (np.array([100.0, 0.5]), np.array([-500.0, 0.2]), np.array([0.0, 0.9])):
            rev = model.reversible_dynamics(state)
            irr = model.irreversible_dynamics(state)
            full = model.full_dynamics(state)
            assert np.all(np.isfinite(rev))
            assert np.all(np.isfinite(irr))
            np.testing.assert_allclose(full, rev + irr, rtol=1e-12, atol=1e-12)

    def test_entropy_production_rate_nonnegative(self):
        model = _make_model()
        for state in (np.array([100.0, 0.5]), np.array([500.0, 0.3])):
            assert model.entropy_production_rate(state) >= -1e-12


class TestSGRGenericThixotropyEdgeCases:
    """Error paths and re-configuration for thixotropy."""

    def test_enable_thixotropy_twice_updates_values(self):
        model = SGRGeneric()
        model.enable_thixotropy(k_build=0.1, k_break=0.5, n_struct=2.0)
        # Second call must overwrite existing parameter values (else-branches)
        model.enable_thixotropy(k_build=0.2, k_break=0.6, n_struct=1.5)
        assert model.parameters.get_value("k_build") == 0.2
        assert model.parameters.get_value("k_break") == 0.6
        assert model.parameters.get_value("n_struct") == 1.5

    def test_evolve_lambda_requires_enabled(self):
        model = SGRGeneric()
        with pytest.raises(ValueError, match="Thixotropy not enabled"):
            model.evolve_lambda(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_evolve_lambda_shape_mismatch(self):
        model = SGRGeneric()
        model.enable_thixotropy()
        with pytest.raises(ValueError, match="same shape"):
            model.evolve_lambda(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

    def test_predict_thixotropic_stress_requires_enabled(self):
        model = SGRGeneric()
        with pytest.raises(ValueError, match="Thixotropy not enabled"):
            model.predict_thixotropic_stress(
                np.array([0.0, 1.0]), np.array([0.0, 1.0])
            )


class TestSGRGenericDynamicXErrors:
    """Error paths for dynamic noise-temperature evolution."""

    def test_evolve_x_requires_dynamic_x(self):
        model = SGRGeneric()  # dynamic_x=False
        with pytest.raises(ValueError, match="Dynamic x not enabled"):
            model.evolve_x(np.array([0.0, 1.0]), np.array([0.0, 1.0]))

    def test_evolve_x_shape_mismatch(self):
        model = SGRGeneric(dynamic_x=True)
        with pytest.raises(ValueError, match="same shape"):
            model.evolve_x(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))


class TestSGRGenericLAOSAnalysisEdgeCases:
    """Edge branches in harmonic / Chebyshev decomposition."""

    def test_harmonics_small_window_zeros_high_harmonics(self):
        """Window too short to resolve 5th/7th harmonics -> zeroed out."""
        model = _make_model()
        model._test_mode = "oscillation"
        _, stress = model.simulate_laos(
            gamma_0=0.5, omega=1.0, n_cycles=2, n_points_per_cycle=10
        )
        harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=10)
        # n=10 -> n//2=5, so idx_5=5 and idx_7=7 are not < 5 -> else-branch
        assert harmonics["I_5"] == 0.0
        assert harmonics["I_7"] == 0.0

    def test_harmonics_tiny_window_zeros_third_harmonic(self):
        """n_points_per_cycle=4 -> n//2=2 so even idx_3=3 is unresolved."""
        model = _make_model()
        model._test_mode = "oscillation"
        _, stress = model.simulate_laos(
            gamma_0=0.5, omega=1.0, n_cycles=2, n_points_per_cycle=4
        )
        harmonics = model.extract_laos_harmonics(stress, n_points_per_cycle=4)
        assert harmonics["I_3"] == 0.0

    def test_harmonics_zero_stress_zero_ratios(self):
        model = _make_model()
        harmonics = model.extract_laos_harmonics(np.zeros(256))
        # I_1 == 0 triggers the ratio else-branch
        assert harmonics["I_3_I_1"] == 0.0
        assert harmonics["I_5_I_1"] == 0.0
        assert harmonics["I_7_I_1"] == 0.0

    def test_chebyshev_zero_stress_zero_ratios(self):
        model = _make_model()
        strain = np.zeros(256)
        stress = np.zeros(256)
        cheb = model.compute_chebyshev_coefficients(strain, stress, gamma_0=0.5, omega=1.0)
        # e_1 and v_1 near zero trigger the else-branches
        assert cheb["e_3_e_1"] == 0.0
        assert cheb["e_5_e_1"] == 0.0
        assert cheb["v_3_v_1"] == 0.0
        assert cheb["v_5_v_1"] == 0.0
