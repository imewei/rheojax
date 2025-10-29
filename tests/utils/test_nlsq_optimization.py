"""Tests for NLSQ optimization integration.

This test suite validates that the nlsq_optimize() function works correctly
with NLSQ optimizer, maintaining float64 precision and proper ParameterSet
integration.
"""

import numpy as np
import pytest

from rheo.core.parameters import Parameter, ParameterSet
from rheo.utils.optimization import OptimizationResult, nlsq_optimize


class TestNLSQOptimizer:
    """Test NLSQ optimizer implementation."""

    def test_nlsq_optimize_function_signature(self):
        """Test nlsq_optimize() function signature and parameter handling."""
        # Set up simple parameters
        params = ParameterSet()
        params.add(name="x", value=1.0, bounds=(0.0, 10.0))

        # Define simple objective function
        def objective(values):
            x = values[0]
            return (x - 5.0) ** 2

        # Call nlsq_optimize with all expected parameters
        result = nlsq_optimize(
            objective=objective,
            parameters=params,
            method="auto",
            use_jax=True,
            max_iter=100,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )

        # Verify result is OptimizationResult
        assert isinstance(result, OptimizationResult)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")

    def test_bounds_extraction_from_parameterset(self):
        """Test that bounds are correctly extracted from ParameterSet."""
        # Create ParameterSet with various bound types
        params = ParameterSet()
        params.add(name="a", value=1.0, bounds=(0.0, 10.0))
        params.add(name="b", value=2.0, bounds=(-5.0, 5.0))
        params.add(name="c", value=3.0, bounds=(-100.0, 100.0))  # Effectively unbounded

        # Extract bounds
        bounds = params.get_bounds()

        # Verify bounds structure
        assert len(bounds) == 3
        assert bounds[0] == (0.0, 10.0)
        assert bounds[1] == (-5.0, 5.0)
        assert bounds[2] == (-100.0, 100.0)

        # Define objective
        def objective(values):
            return sum((values - np.array([5.0, 0.0, 7.0])) ** 2)

        # Optimize with bounds
        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify bounds were respected
        assert 0.0 <= result.x[0] <= 10.0
        assert -5.0 <= result.x[1] <= 5.0
        assert -100.0 <= result.x[2] <= 100.0

    def test_optimization_result_structure(self):
        """Test OptimizationResult dataclass structure and fields."""
        # Create simple optimization problem
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            return values[0] ** 2

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify OptimizationResult has all required fields
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "nit")
        assert hasattr(result, "nfev")

        # Verify types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, (float, np.floating))
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.nit, int)
        assert isinstance(result.nfev, int)

    def test_float64_precision_maintained(self):
        """Test that float64 precision is maintained throughout optimization."""
        # Create parameters with float64 values
        params = ParameterSet()
        params.add(name="x", value=1.0, bounds=(0.0, 10.0))
        params.add(name="y", value=1.0, bounds=(0.0, 10.0))

        # Objective requiring high precision
        def objective(values):
            # Use JAX arrays to maintain precision
            from rheo.core.jax_config import safe_import_jax

            jax, jnp = safe_import_jax()

            x, y = values[0], values[1]
            result = (x - 5.123456789) ** 2 + (y - 3.987654321) ** 2
            return result

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify result maintains float64 precision
        assert result.x.dtype == np.float64, f"Expected float64, got {result.x.dtype}"

        # Verify parameter values are float64
        for name in params:
            param = params.get(name)
            value = param.value
            # Convert to array to check dtype
            arr_value = np.array(value)
            assert arr_value.dtype == np.float64 or isinstance(
                value, float
            ), f"Parameter {name} value is not float64: {type(value)}"

    def test_convergence_simple_problem(self):
        """Test convergence for simple optimization problem."""
        # Classic quadratic problem: minimize (x - 5)^2 + (y - 3)^2
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 5.0) ** 2 + (y - 3.0) ** 2

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True, max_iter=1000)

        # Verify convergence
        assert result.success, f"Optimization failed: {result.message}"

        # Verify optimal values (within tolerance)
        assert abs(result.x[0] - 5.0) < 1e-4, f"x = {result.x[0]}, expected 5.0"
        assert abs(result.x[1] - 3.0) < 1e-4, f"y = {result.x[1]}, expected 3.0"

        # Verify objective value is near zero
        assert result.fun < 1e-6, f"Final objective value: {result.fun}"

        # Verify parameters were updated
        assert abs(params.get_value("x") - 5.0) < 1e-4
        assert abs(params.get_value("y") - 3.0) < 1e-4

    def test_nlsq_updates_parameterset(self):
        """Test that nlsq_optimize() updates ParameterSet with optimal values."""
        # Create parameters with initial values
        params = ParameterSet()
        params.add(name="a", value=10.0, bounds=(0.0, 20.0))
        params.add(name="b", value=10.0, bounds=(0.0, 20.0))

        # Store initial values
        initial_a = params.get_value("a")
        initial_b = params.get_value("b")

        def objective(values):
            return (values[0] - 7.0) ** 2 + (values[1] - 4.0) ** 2

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify parameters were updated
        final_a = params.get_value("a")
        final_b = params.get_value("b")

        assert final_a != initial_a, "Parameter 'a' was not updated"
        assert final_b != initial_b, "Parameter 'b' was not updated"

        # Verify updated values are close to optimal
        assert abs(final_a - 7.0) < 1e-3
        assert abs(final_b - 4.0) < 1e-3

        # Verify result.x matches ParameterSet values
        assert np.allclose(result.x, params.get_values())

    def test_nlsq_with_tight_bounds(self):
        """Test NLSQ optimizer handles tight bounds correctly."""
        # Create problem where optimal is at boundary
        # Use feasible initial value within bounds
        params = ParameterSet()
        params.add(
            name="x", value=1.5, bounds=(0.0, 3.0)
        )  # Initial value within bounds

        def objective(values):
            return (values[0] - 5.0) ** 2  # Optimal would be 5, but bounded to 3

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify result respects bounds
        assert result.success, f"Optimization failed: {result.message}"
        assert 0.0 <= result.x[0] <= 3.0

        # Should converge to upper bound
        assert abs(result.x[0] - 3.0) < 1e-3


class TestOptimizationResultDataclass:
    """Test OptimizationResult dataclass functionality."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult can be created with required fields."""
        result = OptimizationResult(
            x=np.array([1.0, 2.0]),
            fun=0.5,
            jac=np.array([0.0, 0.0]),
            success=True,
            message="Optimization converged",
            nit=10,
            nfev=50,
            njev=10,
        )

        assert result.x.shape == (2,)
        assert result.fun == 0.5
        assert result.success is True
        assert result.nit == 10


class TestNLSQErrorHandling:
    """Test error handling in NLSQ optimization."""

    def test_invalid_objective_raises_error(self):
        """Test that non-callable objective raises ValueError."""
        params = ParameterSet()
        params.add(name="x", value=1.0)

        with pytest.raises(ValueError, match="objective must be callable"):
            nlsq_optimize("not a function", params)

    def test_invalid_parameters_raises_error(self):
        """Test that non-ParameterSet parameters raises ValueError."""

        def objective(x):
            return x[0] ** 2

        with pytest.raises(ValueError, match="parameters must be ParameterSet"):
            nlsq_optimize(objective, "not a ParameterSet")
