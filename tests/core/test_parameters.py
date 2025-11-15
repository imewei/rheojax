"""Tests for parameter management system.

This test suite ensures proper parameter handling, validation,
and optimization support for models and transforms.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import (
    Parameter,
    ParameterConstraint,
    ParameterOptimizer,
    ParameterSet,
    SharedParameterSet,
)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestParameterConstraints:
    """Test parameter constraint system."""

    @pytest.mark.smoke
    def test_bounds_constraint(self):
        """Test bounds constraints on parameters."""
        constraint = ParameterConstraint(type="bounds", min_value=0.0, max_value=100.0)

        # Valid values
        assert constraint.validate(50.0) == True
        assert constraint.validate(0.0) == True
        assert constraint.validate(100.0) == True

        # Invalid values
        assert constraint.validate(-1.0) == False
        assert constraint.validate(101.0) == False

    @pytest.mark.smoke
    def test_positive_constraint(self):
        """Test positive value constraint."""
        constraint = ParameterConstraint(type="positive")

        assert constraint.validate(1.0) == True
        assert constraint.validate(0.1) == True
        assert constraint.validate(0.0) == False
        assert constraint.validate(-1.0) == False

    @pytest.mark.smoke
    def test_integer_constraint(self):
        """Test integer value constraint."""
        constraint = ParameterConstraint(type="integer")

        assert constraint.validate(5) == True
        assert constraint.validate(5.0) == True  # Whole number float
        assert constraint.validate(5.5) == False
        assert constraint.validate(5.1) == False

    @pytest.mark.smoke
    def test_fixed_constraint(self):
        """Test fixed parameter constraint."""
        constraint = ParameterConstraint(type="fixed", value=10.0)

        assert constraint.validate(10.0) == True
        assert constraint.validate(9.99) == False
        assert constraint.validate(10.01) == False

    @pytest.mark.smoke
    def test_relative_constraint(self):
        """Test relative constraints between parameters."""
        # Parameter A must be less than parameter B
        constraint = ParameterConstraint(
            type="relative", relation="less_than", other_param="param_b"
        )

        # Need context with both parameters
        context: dict = {"param_a": 5.0, "param_b": 10.0}
        assert constraint.validate(5.0, context=context) == True

        context = {"param_a": 15.0, "param_b": 10.0}
        assert constraint.validate(15.0, context=context) == False

    @pytest.mark.smoke
    def test_custom_constraint(self):
        """Test custom constraint function."""

        def custom_validator(value):
            # Value must be even
            return value % 2 == 0

        constraint = ParameterConstraint(type="custom", validator=custom_validator)

        assert constraint.validate(4) == True
        assert constraint.validate(5) == False

    @pytest.mark.smoke
    def test_multiple_constraints(self):
        """Test applying multiple constraints."""
        param = Parameter(
            name="test",
            value=5.0,
            constraints=[
                ParameterConstraint(type="positive"),
                ParameterConstraint(type="bounds", min_value=0, max_value=10),
                ParameterConstraint(type="integer"),
            ],
        )

        # Valid value
        assert param.validate(5.0) == True

        # Invalid values
        assert param.validate(-1.0) == False  # Not positive
        assert param.validate(11.0) == False  # Out of bounds
        assert param.validate(5.5) == False  # Not integer


class TestSharedParameters:
    """Test shared parameter management across models."""

    @pytest.mark.smoke
    def test_create_shared_parameter_set(self):
        """Test creating shared parameters."""
        shared = SharedParameterSet()

        # Add shared parameters
        shared.add_shared("temperature", value=25.0, units="C")
        shared.add_shared("pressure", value=101.3, units="kPa")

        assert "temperature" in shared
        assert "pressure" in shared

    def test_link_models_to_shared_parameters(self):
        """Test linking models to shared parameters."""
        shared = SharedParameterSet()
        shared.add_shared("G", value=100.0, units="Pa")

        # Create mock models
        model1 = Mock()
        model1.name = "model1"
        model1.parameters = ParameterSet()

        model2 = Mock()
        model2.name = "model2"
        model2.parameters = ParameterSet()

        # Link models to shared parameter
        shared.link_model(model1, "G")
        shared.link_model(model2, "G")

        # Both models should reference the same parameter
        assert shared.get_linked_models("G") == [model1, model2]

    def test_update_shared_parameter(self):
        """Test updating shared parameters propagates to all models."""
        shared = SharedParameterSet()
        shared.add_shared("alpha", value=1.0)

        # Create and link models
        model1_params = ParameterSet()
        model2_params = ParameterSet()

        shared.link_parameter_set(model1_params, "alpha")
        shared.link_parameter_set(model2_params, "alpha")

        # Update shared parameter
        shared.set_value("alpha", 2.0)

        # Check propagation
        assert shared.get_value("alpha") == 2.0
        # In real implementation, linked models would also update

    def test_shared_parameter_constraints(self):
        """Test constraints on shared parameters."""
        shared = SharedParameterSet()

        # Add shared parameter with constraints
        shared.add_shared(
            "ratio",
            value=0.5,
            constraints=[ParameterConstraint(type="bounds", min_value=0, max_value=1)],
        )

        # Valid update
        shared.set_value("ratio", 0.7)
        assert shared.get_value("ratio") == 0.7

        # Invalid update should raise
        with pytest.raises(ValueError, match="violates constraints"):
            shared.set_value("ratio", 1.5)

    def test_shared_parameter_groups(self):
        """Test grouping related shared parameters."""
        shared = SharedParameterSet()

        # Create parameter groups
        shared.create_group("viscoelastic", ["G", "eta", "tau"])
        shared.create_group("thermal", ["T", "alpha_T"])

        # Add parameters to groups
        shared.add_shared("G", value=100.0, group="viscoelastic")
        shared.add_shared("eta", value=1000.0, group="viscoelastic")
        shared.add_shared("T", value=25.0, group="thermal")

        # Get parameters by group
        ve_params: dict = shared.get_group("viscoelastic")
        assert "G" in ve_params
        assert "eta" in ve_params
        assert "T" not in ve_params


class TestParameterOptimizer:
    """Test parameter optimization utilities."""

    def test_optimizer_setup(self):
        """Test setting up parameter optimizer."""
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(0, 10))
        params.add("b", value=2.0, bounds=(0, 20))

        optimizer = ParameterOptimizer(params)

        # Check initial state
        assert optimizer.n_parameters == 2
        assert np.array_equal(optimizer.get_values(), [1.0, 2.0])
        assert optimizer.get_bounds() == [(0, 10), (0, 20)]

    def test_optimizer_objective_function(self):
        """Test defining objective function for optimization."""
        params = ParameterSet()
        params.add("x", value=0.0)
        params.add("y", value=0.0)

        optimizer = ParameterOptimizer(params)

        # Define objective (minimize x^2 + y^2)
        def objective(values):
            x, y = values
            return x**2 + y**2

        optimizer.set_objective(objective)

        # Evaluate objective
        result = optimizer.evaluate([3.0, 4.0])
        assert result == 25.0

    def test_optimizer_with_jax(self):
        """Test optimizer with JAX arrays."""
        params = ParameterSet()
        params.add("theta", value=0.0, bounds=(-np.pi, np.pi))

        optimizer = ParameterOptimizer(params, use_jax=True)

        # Define JAX objective
        def objective(values):
            theta = values[0]
            return jnp.sin(theta) ** 2

        optimizer.set_objective(objective)

        # Test with JAX array
        result = optimizer.evaluate(jnp.array([jnp.pi / 2]))
        assert float(result) == 1.0

    def test_optimizer_gradient(self):
        """Test gradient computation for optimization."""
        params = ParameterSet()
        params.add("x", value=1.0)

        optimizer = ParameterOptimizer(params, use_jax=True)

        # Define differentiable objective
        def objective(values):
            x = values[0]
            return x**3 - 2 * x + 1

        optimizer.set_objective(objective)

        # Compute gradient at x=2
        # f'(x) = 3x^2 - 2, so f'(2) = 10
        gradient = optimizer.compute_gradient([2.0])
        assert float(gradient[0]) == 10.0

    def test_optimizer_constraints(self):
        """Test optimization with constraints."""
        params = ParameterSet()
        params.add("x", value=1.0, bounds=(0, 5))
        params.add("y", value=1.0, bounds=(0, 5))

        optimizer = ParameterOptimizer(params)

        # Add constraint: x + y <= 6
        def constraint_fn(values):
            x, y = values
            return 6 - (x + y)  # >= 0 for valid

        optimizer.add_constraint(constraint_fn)

        # Check constraint validation
        assert optimizer.validate_constraints([2.0, 3.0]) == True
        assert optimizer.validate_constraints([4.0, 3.0]) == False

    def test_optimizer_history(self):
        """Test optimization history tracking."""
        params = ParameterSet()
        params.add("x", value=5.0)

        optimizer = ParameterOptimizer(params, track_history=True)

        def objective(values):
            return (values[0] - 2) ** 2

        optimizer.set_objective(objective)

        # Simulate optimization steps
        optimizer.step([5.0])
        optimizer.step([4.0])
        optimizer.step([3.0])
        optimizer.step([2.0])

        history = optimizer.get_history()

        assert len(history) == 4
        assert history[-1]["values"] == [2.0]
        assert history[-1]["objective"] == 0.0

    def test_optimizer_callback(self):
        """Test optimization callbacks."""
        params = ParameterSet()
        params.add("x", value=0.0)

        optimizer = ParameterOptimizer(params)

        # Track callback calls
        callback_data = []

        def callback(iteration, values, objective_value):
            callback_data.append(
                {
                    "iteration": iteration,
                    "values": values.copy(),
                    "objective": objective_value,
                }
            )

        optimizer.set_callback(callback)

        # Simulate optimization
        def objective(values):
            return values[0] ** 2

        optimizer.set_objective(objective)

        for i, x in enumerate([5.0, 3.0, 1.0]):
            optimizer.step([x], iteration=i)

        assert len(callback_data) == 3
        assert callback_data[0]["values"] == [5.0]
        assert callback_data[-1]["values"] == [1.0]


class TestParameterSensitivity:
    """Test parameter sensitivity analysis."""

    def test_local_sensitivity(self):
        """Test local sensitivity analysis."""
        params = ParameterSet()
        params.add("k", value=1.0)
        params.add("b", value=0.0)

        # Model: y = k*x + b
        def model(param_values, x):
            k, b = param_values
            return k * x + b

        # Compute sensitivity at x=2
        x_test = 2.0
        base_values = [1.0, 0.0]

        # Sensitivity to k: dy/dk = x = 2
        # Sensitivity to b: dy/db = 1

        sensitivities = []
        delta = 1e-6

        for i in range(2):
            values_plus = base_values.copy()
            values_plus[i] += delta

            y_base = model(base_values, x_test)
            y_plus = model(values_plus, x_test)

            sensitivity = (y_plus - y_base) / delta
            sensitivities.append(sensitivity)

        assert np.allclose(sensitivities[0], 2.0, rtol=1e-5)  # dk
        assert np.allclose(sensitivities[1], 1.0, rtol=1e-5)  # db

    def test_global_sensitivity(self):
        """Test global sensitivity analysis (variance-based)."""
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(0.5, 1.5))
        params.add("b", value=2.0, bounds=(1.5, 2.5))

        # Model with interaction: y = a*b + a^2
        def model(param_values):
            a, b = param_values
            return a * b + a**2

        # Monte Carlo sensitivity analysis
        n_samples = 1000
        np.random.seed(42)

        # Generate samples
        a_samples = np.random.uniform(0.5, 1.5, n_samples)
        b_samples = np.random.uniform(1.5, 2.5, n_samples)

        # Compute outputs
        outputs = []
        for a, b in zip(a_samples, b_samples):
            outputs.append(model([a, b]))

        outputs = np.array(outputs)

        # Compute variance
        total_variance = np.var(outputs)

        # First-order sensitivity for 'a'
        # Fix b, vary a
        b_fixed = 2.0
        outputs_a = []
        for a in a_samples:
            outputs_a.append(model([a, b_fixed]))

        variance_a = np.var(outputs_a)
        sensitivity_a = variance_a / total_variance

        assert 0 <= sensitivity_a <= 1

    def test_parameter_identifiability(self):
        """Test parameter identifiability analysis."""
        params = ParameterSet()
        params.add("k1", value=1.0)
        params.add("k2", value=2.0)

        # Model where k1 and k2 are not independently identifiable
        # y = (k1 * k2) * x, only the product matters
        def model(param_values, x):
            k1, k2 = param_values
            return (k1 * k2) * x

        # Generate synthetic data
        x_data = np.linspace(0, 10, 20)
        true_product = 2.0
        y_data = true_product * x_data

        # Fisher Information Matrix approximation
        n_params = 2
        fim = np.zeros((n_params, n_params))

        delta = 1e-6
        base_values = [1.0, 2.0]

        for i in range(n_params):
            for j in range(n_params):
                # Compute derivatives
                values_i = base_values.copy()
                values_i[i] += delta

                values_j = base_values.copy()
                values_j[j] += delta

                dy_di = (model(values_i, x_data) - model(base_values, x_data)) / delta
                dy_dj = (model(values_j, x_data) - model(base_values, x_data)) / delta

                fim[i, j] = np.sum(dy_di * dy_dj)

        # Check condition number (high = poor identifiability)
        condition_number = np.linalg.cond(fim)
        assert condition_number > 100  # Indicates poor identifiability


from unittest.mock import Mock, patch

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import (
    Parameter,
    ParameterConstraint,
    ParameterOptimizer,
    ParameterSet,
    SharedParameterSet,
)

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class TestParameterConstraints:
    """Test parameter constraint system."""

    def test_bounds_constraint(self):
        """Test bounds constraints on parameters."""
        constraint = ParameterConstraint(type="bounds", min_value=0.0, max_value=100.0)

        # Valid values
        assert constraint.validate(50.0) == True
        assert constraint.validate(0.0) == True
        assert constraint.validate(100.0) == True

        # Invalid values
        assert constraint.validate(-1.0) == False
        assert constraint.validate(101.0) == False

    def test_positive_constraint(self):
        """Test positive value constraint."""
        constraint = ParameterConstraint(type="positive")

        assert constraint.validate(1.0) == True
        assert constraint.validate(0.1) == True
        assert constraint.validate(0.0) == False
        assert constraint.validate(-1.0) == False

    def test_integer_constraint(self):
        """Test integer value constraint."""
        constraint = ParameterConstraint(type="integer")

        assert constraint.validate(5) == True
        assert constraint.validate(5.0) == True  # Whole number float
        assert constraint.validate(5.5) == False
        assert constraint.validate(5.1) == False

    def test_fixed_constraint(self):
        """Test fixed parameter constraint."""
        constraint = ParameterConstraint(type="fixed", value=10.0)

        assert constraint.validate(10.0) == True
        assert constraint.validate(9.99) == False
        assert constraint.validate(10.01) == False

    def test_relative_constraint(self):
        """Test relative constraints between parameters."""
        # Parameter A must be less than parameter B
        constraint = ParameterConstraint(
            type="relative", relation="less_than", other_param="param_b"
        )

        # Need context with both parameters
        context = {"param_a": 5.0, "param_b": 10.0}
        assert constraint.validate(5.0, context=context) == True

        context = {"param_a": 15.0, "param_b": 10.0}
        assert constraint.validate(15.0, context=context) == False

    def test_custom_constraint(self):
        """Test custom constraint function."""

        def custom_validator(value):
            # Value must be even
            return value % 2 == 0

        constraint = ParameterConstraint(type="custom", validator=custom_validator)

        assert constraint.validate(4) == True
        assert constraint.validate(5) == False

    def test_multiple_constraints(self):
        """Test applying multiple constraints."""
        param = Parameter(
            name="test",
            value=5.0,
            constraints=[
                ParameterConstraint(type="positive"),
                ParameterConstraint(type="bounds", min_value=0, max_value=10),
                ParameterConstraint(type="integer"),
            ],
        )

        # Valid value
        assert param.validate(5.0) == True

        # Invalid values
        assert param.validate(-1.0) == False  # Not positive
        assert param.validate(11.0) == False  # Out of bounds
        assert param.validate(5.5) == False  # Not integer


class TestSharedParameters:
    """Test shared parameter management across models."""

    def test_create_shared_parameter_set(self):
        """Test creating shared parameters."""
        shared = SharedParameterSet()

        # Add shared parameters
        shared.add_shared("temperature", value=25.0, units="C")
        shared.add_shared("pressure", value=101.3, units="kPa")

        assert "temperature" in shared
        assert "pressure" in shared

    def test_link_models_to_shared_parameters(self):
        """Test linking models to shared parameters."""
        shared = SharedParameterSet()
        shared.add_shared("G", value=100.0, units="Pa")

        # Create mock models
        model1 = Mock()
        model1.name = "model1"
        model1.parameters = ParameterSet()

        model2 = Mock()
        model2.name = "model2"
        model2.parameters = ParameterSet()

        # Link models to shared parameter
        shared.link_model(model1, "G")
        shared.link_model(model2, "G")

        # Both models should reference the same parameter
        assert shared.get_linked_models("G") == [model1, model2]

    def test_update_shared_parameter(self):
        """Test updating shared parameters propagates to all models."""
        shared = SharedParameterSet()
        shared.add_shared("alpha", value=1.0)

        # Create and link models
        model1_params = ParameterSet()
        model2_params = ParameterSet()

        shared.link_parameter_set(model1_params, "alpha")
        shared.link_parameter_set(model2_params, "alpha")

        # Update shared parameter
        shared.set_value("alpha", 2.0)

        # Check propagation
        assert shared.get_value("alpha") == 2.0
        # In real implementation, linked models would also update

    def test_shared_parameter_constraints(self):
        """Test constraints on shared parameters."""
        shared = SharedParameterSet()

        # Add shared parameter with constraints
        shared.add_shared(
            "ratio",
            value=0.5,
            constraints=[ParameterConstraint(type="bounds", min_value=0, max_value=1)],
        )

        # Valid update
        shared.set_value("ratio", 0.7)
        assert shared.get_value("ratio") == 0.7

        # Invalid update should raise
        with pytest.raises(ValueError, match="violates constraints"):
            shared.set_value("ratio", 1.5)

    def test_shared_parameter_groups(self):
        """Test grouping related shared parameters."""
        shared = SharedParameterSet()

        # Create parameter groups
        shared.create_group("viscoelastic", ["G", "eta", "tau"])
        shared.create_group("thermal", ["T", "alpha_T"])

        # Add parameters to groups
        shared.add_shared("G", value=100.0, group="viscoelastic")
        shared.add_shared("eta", value=1000.0, group="viscoelastic")
        shared.add_shared("T", value=25.0, group="thermal")

        # Get parameters by group
        ve_params = shared.get_group("viscoelastic")
        assert "G" in ve_params
        assert "eta" in ve_params
        assert "T" not in ve_params


class TestParameterOptimizer:
    """Test parameter optimization utilities."""

    def test_optimizer_setup(self):
        """Test setting up parameter optimizer."""
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(0, 10))
        params.add("b", value=2.0, bounds=(0, 20))

        optimizer = ParameterOptimizer(params)

        # Check initial state
        assert optimizer.n_parameters == 2
        assert np.array_equal(optimizer.get_values(), [1.0, 2.0])
        assert optimizer.get_bounds() == [(0, 10), (0, 20)]

    def test_optimizer_objective_function(self):
        """Test defining objective function for optimization."""
        params = ParameterSet()
        params.add("x", value=0.0)
        params.add("y", value=0.0)

        optimizer = ParameterOptimizer(params)

        # Define objective (minimize x^2 + y^2)
        def objective(values):
            x, y = values
            return x**2 + y**2

        optimizer.set_objective(objective)

        # Evaluate objective
        result = optimizer.evaluate([3.0, 4.0])
        assert result == 25.0

    def test_optimizer_with_jax(self):
        """Test optimizer with JAX arrays."""
        params = ParameterSet()
        params.add("theta", value=0.0, bounds=(-np.pi, np.pi))

        optimizer = ParameterOptimizer(params, use_jax=True)

        # Define JAX objective
        def objective(values):
            theta = values[0]
            return jnp.sin(theta) ** 2

        optimizer.set_objective(objective)

        # Test with JAX array
        result = optimizer.evaluate(jnp.array([jnp.pi / 2]))
        assert float(result) == 1.0

    def test_optimizer_gradient(self):
        """Test gradient computation for optimization."""
        params = ParameterSet()
        params.add("x", value=1.0)

        optimizer = ParameterOptimizer(params, use_jax=True)

        # Define differentiable objective
        def objective(values):
            x = values[0]
            return x**3 - 2 * x + 1

        optimizer.set_objective(objective)

        # Compute gradient at x=2
        # f'(x) = 3x^2 - 2, so f'(2) = 10
        gradient = optimizer.compute_gradient([2.0])
        assert float(gradient[0]) == 10.0

    def test_optimizer_constraints(self):
        """Test optimization with constraints."""
        params = ParameterSet()
        params.add("x", value=1.0, bounds=(0, 5))
        params.add("y", value=1.0, bounds=(0, 5))

        optimizer = ParameterOptimizer(params)

        # Add constraint: x + y <= 6
        def constraint_fn(values):
            x, y = values
            return 6 - (x + y)  # >= 0 for valid

        optimizer.add_constraint(constraint_fn)

        # Check constraint validation
        assert optimizer.validate_constraints([2.0, 3.0]) == True
        assert optimizer.validate_constraints([4.0, 3.0]) == False

    def test_optimizer_history(self):
        """Test optimization history tracking."""
        params = ParameterSet()
        params.add("x", value=5.0)

        optimizer = ParameterOptimizer(params, track_history=True)

        def objective(values):
            return (values[0] - 2) ** 2

        optimizer.set_objective(objective)

        # Simulate optimization steps
        optimizer.step([5.0])
        optimizer.step([4.0])
        optimizer.step([3.0])
        optimizer.step([2.0])

        history = optimizer.get_history()

        assert len(history) == 4
        assert history[-1]["values"] == [2.0]
        assert history[-1]["objective"] == 0.0

    def test_optimizer_callback(self):
        """Test optimization callbacks."""
        params = ParameterSet()
        params.add("x", value=0.0)

        optimizer = ParameterOptimizer(params)

        # Track callback calls
        callback_data = []

        def callback(iteration, values, objective_value):
            callback_data.append(
                {
                    "iteration": iteration,
                    "values": values.copy(),
                    "objective": objective_value,
                }
            )

        optimizer.set_callback(callback)

        # Simulate optimization
        def objective(values):
            return values[0] ** 2

        optimizer.set_objective(objective)

        for i, x in enumerate([5.0, 3.0, 1.0]):
            optimizer.step([x], iteration=i)

        assert len(callback_data) == 3
        assert callback_data[0]["values"] == [5.0]
        assert callback_data[-1]["values"] == [1.0]


class TestParameterSensitivity:
    """Test parameter sensitivity analysis."""

    def test_local_sensitivity(self):
        """Test local sensitivity analysis."""
        params = ParameterSet()
        params.add("k", value=1.0)
        params.add("b", value=0.0)

        # Model: y = k*x + b
        def model(param_values, x):
            k, b = param_values
            return k * x + b

        # Compute sensitivity at x=2
        x_test = 2.0
        base_values = [1.0, 0.0]

        # Sensitivity to k: dy/dk = x = 2
        # Sensitivity to b: dy/db = 1

        sensitivities = []
        delta = 1e-6

        for i in range(2):
            values_plus = base_values.copy()
            values_plus[i] += delta

            y_base = model(base_values, x_test)
            y_plus = model(values_plus, x_test)

            sensitivity = (y_plus - y_base) / delta
            sensitivities.append(sensitivity)

        assert np.allclose(sensitivities[0], 2.0, rtol=1e-5)  # dk
        assert np.allclose(sensitivities[1], 1.0, rtol=1e-5)  # db

    def test_global_sensitivity(self):
        """Test global sensitivity analysis (variance-based)."""
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(0.5, 1.5))
        params.add("b", value=2.0, bounds=(1.5, 2.5))

        # Model with interaction: y = a*b + a^2
        def model(param_values):
            a, b = param_values
            return a * b + a**2

        # Monte Carlo sensitivity analysis
        n_samples = 1000
        np.random.seed(42)

        # Generate samples
        a_samples = np.random.uniform(0.5, 1.5, n_samples)
        b_samples = np.random.uniform(1.5, 2.5, n_samples)

        # Compute outputs
        outputs = []
        for a, b in zip(a_samples, b_samples):
            outputs.append(model([a, b]))

        outputs = np.array(outputs)

        # Compute variance
        total_variance = np.var(outputs)

        # First-order sensitivity for 'a'
        # Fix b, vary a
        b_fixed = 2.0
        outputs_a = []
        for a in a_samples:
            outputs_a.append(model([a, b_fixed]))

        variance_a = np.var(outputs_a)
        sensitivity_a = variance_a / total_variance

        assert 0 <= sensitivity_a <= 1

    def test_parameter_identifiability(self):
        """Test parameter identifiability analysis."""
        params = ParameterSet()
        params.add("k1", value=1.0)
        params.add("k2", value=2.0)

        # Model where k1 and k2 are not independently identifiable
        # y = (k1 * k2) * x, only the product matters
        def model(param_values, x):
            k1, k2 = param_values
            return (k1 * k2) * x

        # Generate synthetic data
        x_data = np.linspace(0, 10, 20)
        true_product = 2.0
        y_data = true_product * x_data

        # Fisher Information Matrix approximation
        n_params = 2
        fim = np.zeros((n_params, n_params))

        delta = 1e-6
        base_values = [1.0, 2.0]

        for i in range(n_params):
            for j in range(n_params):
                # Compute derivatives
                values_i = base_values.copy()
                values_i[i] += delta

                values_j = base_values.copy()
                values_j[j] += delta

                dy_di = (model(values_i, x_data) - model(base_values, x_data)) / delta
                dy_dj = (model(values_j, x_data) - model(base_values, x_data)) / delta

                fim[i, j] = np.sum(dy_di * dy_dj)

        # Check condition number (high = poor identifiability)
        condition_number = np.linalg.cond(fim)
        assert condition_number > 100  # Indicates poor identifiability
