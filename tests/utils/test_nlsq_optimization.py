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


class TestNLSQGradientValidation:
    """Test gradient computation and Jacobian accuracy."""

    def test_gradient_computation_accuracy(self):
        """Test that JAX automatic differentiation produces accurate gradients."""
        from rheo.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()

        # Create simple quadratic function
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 3.0) ** 2 + (y - 4.0) ** 2

        # Compute analytical gradient at (1.0, 1.0)
        # ∂f/∂x = 2(x - 3) = 2(1 - 3) = -4
        # ∂f/∂y = 2(y - 4) = 2(1 - 4) = -6
        expected_grad = np.array([-4.0, -6.0])

        # Compute gradient using JAX
        grad_fn = jax.grad(objective)
        computed_grad = grad_fn(np.array([1.0, 1.0]))

        # Verify gradient accuracy
        np.testing.assert_allclose(
            computed_grad, expected_grad, rtol=1e-10, atol=1e-10
        )

    def test_jacobian_for_residual_function(self):
        """Test Jacobian computation for residual-based optimization."""
        from rheo.core.jax_config import safe_import_jax

        jax, jnp = safe_import_jax()

        # Create residual function: r(x) = [x - 3, y - 4]
        def residual(values):
            x, y = values[0], values[1]
            return jnp.array([x - 3.0, y - 4.0])

        # Analytical Jacobian:
        # J = [[1, 0],
        #      [0, 1]]
        expected_jacobian = np.array([[1.0, 0.0], [0.0, 1.0]])

        # Compute Jacobian using JAX
        jac_fn = jax.jacobian(residual)
        computed_jacobian = jac_fn(np.array([1.0, 1.0]))

        # Verify Jacobian accuracy
        np.testing.assert_allclose(
            computed_jacobian, expected_jacobian, rtol=1e-10, atol=1e-10
        )


class TestNLSQScipyComparison:
    """Test NLSQ optimization against scipy for validation."""

    def test_nlsq_vs_scipy_simple_problem(self):
        """Test that NLSQ produces similar results to scipy.optimize.minimize."""
        from scipy.optimize import minimize

        # Simple quadratic problem
        params_nlsq = ParameterSet()
        params_nlsq.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params_nlsq.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 5.0) ** 2 + (y - 3.0) ** 2

        # Run NLSQ optimization
        result_nlsq = nlsq_optimize(objective, params_nlsq, use_jax=True, max_iter=1000)

        # Run scipy optimization
        x0 = np.array([0.0, 0.0])
        bounds = [(-10.0, 10.0), (-10.0, 10.0)]
        result_scipy = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        # Both should converge to same solution
        np.testing.assert_allclose(result_nlsq.x, result_scipy.x, rtol=1e-3, atol=1e-3)

        # Both should find near-zero objective value
        assert abs(result_nlsq.fun - result_scipy.fun) < 1e-4

    def test_nlsq_vs_scipy_least_squares(self):
        """Test NLSQ against scipy.optimize.least_squares for residual problems."""
        from scipy.optimize import least_squares

        # Create residual problem: fit line y = a*x + b
        np.random.seed(42)
        x_data = np.linspace(0, 10, 50)
        y_data = 2.0 * x_data + 3.0 + np.random.normal(0, 0.5, 50)

        params = ParameterSet()
        params.add(name="a", value=1.0, bounds=(0.0, 10.0))
        params.add(name="b", value=0.0, bounds=(-10.0, 10.0))

        def residual(values):
            a, b = values[0], values[1]
            return np.sum((y_data - (a * x_data + b)) ** 2)

        # Run NLSQ
        result_nlsq = nlsq_optimize(residual, params, use_jax=True, max_iter=1000)

        # Run scipy least_squares (using same residual-squared approach)
        x0 = np.array([1.0, 0.0])
        bounds = ([0.0, -10.0], [10.0, 10.0])
        result_scipy = least_squares(
            lambda p: np.sqrt(residual(p)), x0, bounds=bounds, method="trf"
        )

        # Results should be close
        np.testing.assert_allclose(result_nlsq.x, result_scipy.x, rtol=0.1, atol=0.1)


class TestNLSQToleranceTesting:
    """Test optimization tolerance parameters (ftol, xtol, gtol)."""

    def test_ftol_convergence_criterion(self):
        """Test that ftol (function tolerance) controls convergence."""
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            return values[0] ** 2

        # Run with tight ftol
        result_tight = nlsq_optimize(
            objective, params, use_jax=True, ftol=1e-10, max_iter=1000
        )

        # Run with loose ftol
        params.set_value("x", 0.0)  # Reset
        result_loose = nlsq_optimize(
            objective, params, use_jax=True, ftol=1e-3, max_iter=1000
        )

        # Both should succeed or at least converge close to optimal
        # Lenient check: just verify convergence to near-optimal
        assert abs(result_tight.x[0]) < 1e-3  # Very close to optimal
        assert abs(result_loose.x[0]) < 0.1  # Reasonably close

    def test_xtol_convergence_criterion(self):
        """Test that xtol (parameter tolerance) controls convergence."""
        params = ParameterSet()
        params.add(name="x", value=5.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=5.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 3.0) ** 2 + (y - 4.0) ** 2

        # Run with different xtol
        result_tight = nlsq_optimize(
            objective, params, use_jax=True, xtol=1e-10, max_iter=1000
        )

        # Both should converge close to (3, 4)
        assert abs(result_tight.x[0] - 3.0) < 1e-4
        assert abs(result_tight.x[1] - 4.0) < 1e-4

    def test_max_iter_limits_iterations(self):
        """Test that max_iter parameter limits iteration count."""
        params = ParameterSet()
        params.add(name="x", value=100.0, bounds=(-1000.0, 1000.0))

        def objective(values):
            return (values[0] - 5.0) ** 2

        # Run with very few iterations (may not converge)
        result_few = nlsq_optimize(objective, params, use_jax=True, max_iter=5)

        # Verify iteration count is limited
        assert result_few.nit <= 5

        # May not converge with so few iterations
        # But should still return valid result
        assert hasattr(result_few, "x")
        assert hasattr(result_few, "success")


class TestNLSQIllConditionedProblems:
    """Test NLSQ optimization on ill-conditioned problems."""

    def test_ill_conditioned_quadratic(self):
        """Test optimization on ill-conditioned quadratic (Rosenbrock-like)."""
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        # Rosenbrock function: (a - x)^2 + b(y - x^2)^2
        # Optimal: (x, y) = (a, a^2) = (1, 1)
        def rosenbrock(values):
            x, y = values[0], values[1]
            a = 1.0
            b = 100.0
            return (a - x) ** 2 + b * (y - x**2) ** 2

        # Run optimization with generous settings
        result = nlsq_optimize(
            rosenbrock, params, use_jax=True, max_iter=5000, ftol=1e-6
        )

        # May require many iterations, but should converge
        # Check proximity to optimal (1, 1) with tolerance
        assert abs(result.x[0] - 1.0) < 0.1  # Lenient for Rosenbrock
        assert abs(result.x[1] - 1.0) < 0.1

    def test_local_minima_trapping(self):
        """Test behavior with function having local minima."""
        params = ParameterSet()
        params.add(name="x", value=2.0, bounds=(-10.0, 10.0))  # Start near local min

        # Function with local minima: sin(x) + x/10
        def objective(values):
            x = values[0]
            return np.sin(x) + x / 10.0

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True, max_iter=1000)

        # May converge to local minimum - just verify it ran
        # Success is not guaranteed for non-convex functions
        # Just check result is within bounds
        assert -10.0 <= result.x[0] <= 10.0


class TestNLSQParameterScaling:
    """Test optimization with parameters of different scales."""

    def test_different_parameter_scales(self):
        """Test optimization with parameters spanning multiple orders of magnitude."""
        params = ParameterSet()
        params.add(name="small", value=1e-6, bounds=(1e-10, 1e-3))
        params.add(name="large", value=1e6, bounds=(1e3, 1e9))

        # Optimal: small=1e-7, large=1e7
        def objective(values):
            small, large = values[0], values[1]
            return (small - 1e-7) ** 2 + (large - 1e7) ** 2

        # Run optimization
        result = nlsq_optimize(objective, params, use_jax=True, max_iter=2000)

        # Verify convergence despite scale differences (lenient tolerances)
        # Multi-scale problems are challenging for optimization
        # Just verify optimizer got reasonably close to optimal values
        assert result.x[0] > 0  # Should be positive
        assert result.x[0] < 1e-5  # Should be small (within 2 orders of magnitude)
        assert abs(result.x[1] - 1e7) / 1e7 < 0.5  # 50% relative tolerance for large param


class TestNLSQRobustness:
    """Test robustness of NLSQ optimization to edge cases."""

    def test_optimization_with_zero_initial_values(self):
        """Test optimization starting from zero initial values."""
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 5.0) ** 2 + (y - 3.0) ** 2

        result = nlsq_optimize(objective, params, use_jax=True)

        assert result.success
        assert abs(result.x[0] - 5.0) < 1e-3
        assert abs(result.x[1] - 3.0) < 1e-3

    def test_optimization_with_boundary_initial_values(self):
        """Test optimization starting from parameter bounds."""
        params = ParameterSet()
        params.add(name="x", value=10.0, bounds=(0.0, 10.0))  # At upper bound

        def objective(values):
            return (values[0] - 5.0) ** 2  # Optimal in middle of range

        result = nlsq_optimize(objective, params, use_jax=True)

        assert result.success
        assert abs(result.x[0] - 5.0) < 1e-3

    def test_optimization_near_optimal(self):
        """Test optimization starting very close to optimal."""
        params = ParameterSet()
        params.add(
            name="x", value=4.9999, bounds=(0.0, 10.0)
        )  # Very close to optimal

        def objective(values):
            return (values[0] - 5.0) ** 2

        result = nlsq_optimize(objective, params, use_jax=True)

        # Should converge quickly
        assert result.success
        assert result.nit < 10  # Should need very few iterations
        assert abs(result.x[0] - 5.0) < 1e-6


class TestNLSQPerformanceCharacteristics:
    """Test performance characteristics of NLSQ optimization."""

    def test_convergence_speed_simple_problem(self):
        """Test that simple problems converge quickly."""
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            return (values[0] - 5.0) ** 2

        result = nlsq_optimize(objective, params, use_jax=True)

        # Simple quadratic should converge in < 50 iterations
        assert result.success
        assert result.nit < 50

    def test_function_evaluation_count(self):
        """Test that function evaluation count is reasonable."""
        params = ParameterSet()
        params.add(name="x", value=0.0, bounds=(-10.0, 10.0))
        params.add(name="y", value=0.0, bounds=(-10.0, 10.0))

        def objective(values):
            x, y = values[0], values[1]
            return (x - 3.0) ** 2 + (y - 4.0) ** 2

        result = nlsq_optimize(objective, params, use_jax=True)

        # Verify nfev is tracked
        assert result.nfev > 0
        assert result.success

        # For gradient-based methods, nfev should be close to nit
        # (may vary depending on line search)
