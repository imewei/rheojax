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
from rheojax.models.sgr_generic import SGRGeneric


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
            assert (
                W >= -1e-12
            ), f"Entropy production W={W:.6e} < 0 violates second law for state={state}"

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

            assert (
                antisymmetry_error < 1e-12
            ), f"Poisson bracket not antisymmetric: max|L + L^T| = {antisymmetry_error:.6e} for state={state}"

            # Diagonal should be zero for antisymmetric matrix
            diag_error = np.max(np.abs(np.diag(L)))
            assert (
                diag_error < 1e-12
            ), f"Poisson bracket diagonal not zero: max|diag(L)| = {diag_error:.6e}"

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
            assert (
                symmetry_error < 1e-12
            ), f"Friction matrix not symmetric: max|M - M^T| = {symmetry_error:.6e} for state={state}"

            # Check positive semi-definiteness via eigenvalues
            eigenvalues = np.linalg.eigvalsh(M)  # Real eigenvalues for symmetric matrix

            # All eigenvalues must be >= 0
            min_eigenvalue = np.min(eigenvalues)
            assert (
                min_eigenvalue >= -1e-12
            ), f"Friction matrix not positive semi-definite: min eigenvalue = {min_eigenvalue:.6e} for state={state}"

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
        assert (
            np.abs(energy_rate_rev) < 1e-9
        ), f"Reversible dynamics not energy-conserving: dF/dt = {energy_rate_rev:.6e}"

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
            assert (
                dS_dt >= -1e-12
            ), f"Entropy production rate dS/dt = {dS_dt:.6e} < 0 violates second law for state={state}"


class TestSGRGenericModelComparison:
    """Test suite for SGRGeneric vs SGRConventional equivalence."""

    def test_equivalence_to_sgr_conventional_linear_regime(self):
        """Test SGRGeneric matches SGRConventional predictions in linear regime."""
        from rheojax.models.sgr_conventional import SGRConventional

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
        assert (
            G_star_generic.shape == G_star_conventional.shape
        ), f"Shape mismatch: generic={G_star_generic.shape}, conventional={G_star_conventional.shape}"

        # Check predictions match within tolerance
        # GENERIC formulation should give same linear response as conventional
        relative_error_G_prime = np.max(
            np.abs(G_star_generic[:, 0] - G_star_conventional[:, 0])
            / (np.abs(G_star_conventional[:, 0]) + 1e-10)
        )
        relative_error_G_double_prime = np.max(
            np.abs(G_star_generic[:, 1] - G_star_conventional[:, 1])
            / (np.abs(G_star_conventional[:, 1]) + 1e-10)
        )

        # Allow 10% tolerance for different numerical implementations
        assert (
            relative_error_G_prime < 0.1
        ), f"G' mismatch: max relative error = {relative_error_G_prime:.4f}"
        assert (
            relative_error_G_double_prime < 0.1
        ), f"G'' mismatch: max relative error = {relative_error_G_double_prime:.4f}"


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
        assert (
            decomposition_error < 1e-10
        ), f"Free energy decomposition error: |F - (U - T*S)| = {decomposition_error:.6e}"


class TestSGRGenericModelInterface:
    """Test suite for BaseModel interface compliance."""

    def test_model_registration_and_base_model_interface(self):
        """Test model registration and BaseModel interface."""
        from rheojax.core.base import BaseModel

        # Check model is registered
        registered_models = ModelRegistry.list_models()
        assert (
            "sgr_generic" in registered_models
        ), "SGRGeneric not registered in ModelRegistry"

        # Check factory pattern works
        model = ModelRegistry.create("sgr_generic")
        assert isinstance(
            model, SGRGeneric
        ), f"Factory created wrong type: {type(model)}"

        # Check BaseModel inheritance
        assert isinstance(model, BaseModel), "SGRGeneric should inherit from BaseModel"

        # Check ParameterSet creation
        assert isinstance(
            model.parameters, ParameterSet
        ), "model.parameters should be ParameterSet"

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

        assert G_star.shape == (10, 2), f"Prediction shape incorrect: {G_star.shape}"
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
            model.fit(omega, G_star, test_mode="creep")
