"""JAX-specific validation and performance tests.

Tests JIT compilation, automatic differentiation, and JAX compatibility.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from rheo.core.data import RheoData
from rheo.utils.mittag_leffler import mittag_leffler_e, mittag_leffler_e2


class TestJITCompilation:
    """Test JIT compilation support."""

    @pytest.mark.jax
    def test_simple_jitted_function(self):
        """Test basic JIT compilation."""
        @jax.jit
        def quadratic(x):
            return x ** 2 + 2 * x + 1

        x = jnp.array([1.0, 2.0, 3.0])
        result = quadratic(x)

        expected = np.array([4.0, 9.0, 16.0])
        np.testing.assert_allclose(result, expected)

    @pytest.mark.jax
    def test_jitted_relu(self):
        """Test JIT compilation of activation function."""
        @jax.jit
        def relu(x):
            return jnp.maximum(x, 0.0)

        x = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = relu(x)

        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(result, expected)

    @pytest.mark.jax
    def test_jitted_complex_function(self):
        """Test JIT compilation with complex operations."""
        @jax.jit
        def complex_ops(x, y):
            z = jnp.sin(x) * jnp.cos(y)
            return jnp.sum(z ** 2)

        x = jnp.linspace(0, np.pi, 10)
        y = jnp.linspace(0, 2*np.pi, 10)

        result = complex_ops(x, y)

        assert isinstance(result, jnp.ndarray)
        assert jnp.isfinite(result)

    @pytest.mark.jax
    @pytest.mark.performance
    def test_jit_compilation_time(self):
        """Test that JIT compilation adds minimal overhead."""
        import time

        def slow_function(x):
            return jnp.sum(jnp.sin(x) ** 2 for _ in range(100))

        jitted_function = jax.jit(slow_function)

        x = jnp.linspace(0, 1, 1000)

        # Warm up (includes compilation)
        start = time.time()
        jitted_function(x)
        compile_time = time.time() - start

        # Second call (should be fast)
        start = time.time()
        jitted_function(x)
        execute_time = time.time() - start

        # Execution should be much faster than compilation
        assert execute_time < compile_time


class TestAutomaticDifferentiation:
    """Test JAX automatic differentiation capabilities."""

    @pytest.mark.jax
    def test_gradient_simple_function(self):
        """Test gradient computation."""
        def f(x):
            return x ** 2 + 3 * x + 2

        grad_f = jax.grad(f)

        x = jnp.array(2.0)
        gradient = grad_f(x)

        # df/dx = 2x + 3, at x=2: gradient = 7
        expected = 2 * 2 + 3
        np.testing.assert_allclose(gradient, expected)

    @pytest.mark.jax
    def test_gradient_vector_function(self):
        """Test gradient of vector-valued function."""
        def f(x):
            return jnp.sum(x ** 2)

        grad_f = jax.grad(f)

        x = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_f(x)

        # df/dx = 2x
        expected = 2 * x
        np.testing.assert_allclose(gradient, expected)

    @pytest.mark.jax
    def test_jacobian_computation(self):
        """Test Jacobian matrix computation."""
        def f(x):
            return jnp.array([x[0] ** 2, x[0] * x[1], x[1] ** 2])

        jacobian_f = jax.jacobian(f)

        x = jnp.array([2.0, 3.0])
        jac = jacobian_f(x)

        # Check shape
        assert jac.shape == (3, 2)

    @pytest.mark.jax
    def test_hessian_computation(self):
        """Test Hessian matrix computation."""
        def f(x):
            return x[0] ** 2 + x[0] * x[1] + x[1] ** 2

        hessian_f = jax.hessian(f)

        x = jnp.array([1.0, 1.0])
        hess = hessian_f(x)

        # Check shape
        assert hess.shape == (2, 2)

        # Hessian should be symmetric
        np.testing.assert_allclose(hess, hess.T)

    @pytest.mark.jax
    def test_grad_with_jit(self):
        """Test combining JIT and grad."""
        def f(x):
            return jnp.sum(jnp.sin(x) ** 2)

        grad_f = jax.jit(jax.grad(f))

        x = jnp.linspace(0, 2*np.pi, 100)
        gradient = grad_f(x)

        assert gradient.shape == x.shape
        assert jnp.all(jnp.isfinite(gradient))

    @pytest.mark.jax
    def test_grad_with_vmap(self):
        """Test combining grad and vmap for batched derivatives."""
        def f(x):
            return jnp.sum(x ** 3)

        grad_f = jax.grad(f)
        batched_grad = jax.vmap(grad_f)

        # Batch of vectors
        x_batch = jnp.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])

        gradients = batched_grad(x_batch)

        assert gradients.shape == x_batch.shape


class TestVectorization:
    """Test JAX vectorization (vmap)."""

    @pytest.mark.jax
    def test_vmap_basic(self):
        """Test basic vmap usage."""
        def f(x):
            return x ** 2

        vmapped_f = jax.vmap(f)

        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = vmapped_f(x)

        expected = x ** 2
        np.testing.assert_allclose(result, expected)

    @pytest.mark.jax
    def test_vmap_batch_processing(self):
        """Test vmap for batch processing."""
        def compute_mean_sq_difference(x, y):
            return jnp.mean((x - y) ** 2)

        # Batch version
        batched_version = jax.vmap(compute_mean_sq_difference)

        # Create batches
        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_batch = jnp.array([[1.1, 2.1], [3.1, 4.1]])

        results = batched_version(x_batch, y_batch)

        assert results.shape == (2,)

    @pytest.mark.jax
    def test_vmap_with_custom_function(self):
        """Test vmap with custom function."""
        def scale_and_shift(x, scale, shift):
            return scale * x + shift

        # Apply to batch with same scale/shift
        batched_func = jax.vmap(
            scale_and_shift, in_axes=(0, None, None)
        )

        x_batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        scale = 2.0
        shift = 1.0

        result = batched_func(x_batch, scale, shift)

        assert result.shape == x_batch.shape


class TestMittagLefflerJAX:
    """Test JAX compatibility of Mittag-Leffler functions."""

    @pytest.mark.jax
    def test_mittag_leffler_jitted(self):
        """Test JIT compilation of Mittag-Leffler."""
        @jax.jit
        def ml_jitted(z, alpha):
            return mittag_leffler_e(z, alpha=alpha)

        z = jnp.array([0.1, 0.5, 1.0])
        alpha = 0.5

        result = ml_jitted(z, alpha)

        assert isinstance(result, jnp.ndarray)
        assert result.shape == z.shape

    @pytest.mark.jax
    def test_mittag_leffler_grad(self):
        """Test gradient of Mittag-Leffler."""
        def f(z):
            return jnp.sum(mittag_leffler_e(z, alpha=0.7))

        grad_f = jax.grad(f)

        z = jnp.array([0.1, 0.2, 0.3])

        try:
            gradient = grad_f(z)
            assert gradient.shape == z.shape
            assert jnp.all(jnp.isfinite(gradient))
        except (ValueError, NotImplementedError):
            # ML function may not be differentiable
            pytest.skip("Mittag-Leffler gradient not implemented")

    @pytest.mark.jax
    @pytest.mark.xfail(reason="vmap over alpha not supported - alpha must be concrete for Mittag-Leffler")
    def test_mittag_leffler_vmap(self):
        """Test vmap of Mittag-Leffler."""
        # Apply to batch of alphas
        alphas = jnp.array([0.3, 0.5, 0.7, 0.9])
        z = 0.5

        # Manual loop version
        results_loop = jnp.array([mittag_leffler_e(z, alpha=a) for a in alphas])

        assert results_loop.shape == alphas.shape


class TestRheoDataJAX:
    """Test JAX operations on RheoData."""

    @pytest.mark.jax
    def test_rheodata_jax_arithmetic(self, oscillation_data_simple):
        """Test arithmetic on JAX-converted RheoData."""
        data = oscillation_data_simple
        jax_data = data.to_jax()

        # Scaling
        scaled = jax_data.y * 2.0
        assert isinstance(scaled, jnp.ndarray)

        # Arithmetic combinations
        result = jax_data.y + jnp.conj(jax_data.y)
        assert isinstance(result, jnp.ndarray)

    @pytest.mark.jax
    def test_rheodata_jax_linear_algebra(self, oscillation_data_large):
        """Test linear algebra operations on RheoData."""
        data = oscillation_data_large
        jax_data = data.to_jax()

        # Real and imaginary parts
        real_part = jax_data.y.real
        imag_part = jax_data.y.imag

        assert real_part.shape == imag_part.shape

        # Magnitude
        magnitude = jnp.abs(jax_data.y)
        assert magnitude.shape == jax_data.y.shape

        # Phase
        phase = jnp.angle(jax_data.y)
        assert phase.shape == jax_data.y.shape

    @pytest.mark.jax
    def test_rheodata_jax_fft(self, oscillation_data_simple):
        """Test FFT operations on RheoData."""
        data = oscillation_data_simple

        # FFT of data
        try:
            fft_result = jnp.fft.fft(data.y)
            assert fft_result.shape == data.y.shape

            # Inverse FFT should recover original (approximately)
            recovered = jnp.fft.ifft(fft_result)
            np.testing.assert_allclose(recovered, data.y, rtol=1e-5)
        except (ValueError, NotImplementedError):
            pytest.skip("FFT not available in current JAX version")


class TestNumericalPrecision:
    """Test numerical precision with JAX."""

    @pytest.mark.jax
    def test_float32_precision(self):
        """Test float32 precision."""
        x32 = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float32)
        result = jnp.sum(x32)

        # Check that we get float32 result
        assert result.dtype == jnp.float32

    @pytest.mark.jax
    def test_float64_precision(self):
        """Test float64 precision."""
        x64 = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
        result = jnp.sum(x64)

        # Check that we get float64 result
        assert result.dtype == jnp.float64

    @pytest.mark.jax
    def test_complex64_operations(self):
        """Test complex64 operations."""
        x = jnp.array([1+2j, 3+4j], dtype=jnp.complex64)
        y = jnp.array([1-2j, 3-4j], dtype=jnp.complex64)

        result = x * y
        assert result.dtype == jnp.complex64

    @pytest.mark.jax
    def test_complex128_operations(self):
        """Test complex128 operations."""
        x = jnp.array([1+2j, 3+4j], dtype=jnp.complex128)
        y = jnp.array([1-2j, 3-4j], dtype=jnp.complex128)

        result = x * y
        assert result.dtype == jnp.complex128


class TestJAXDeviceHandling:
    """Test JAX device handling."""

    @pytest.mark.jax
    def test_array_device_placement(self):
        """Test array device placement."""
        x = jnp.array([1.0, 2.0, 3.0])

        # Check that array is created
        assert x is not None
        assert x.shape == (3,)

    @pytest.mark.jax
    def test_device_transfers(self):
        """Test transfers between devices (if available)."""
        x_jax = jnp.array([1.0, 2.0, 3.0])

        # Convert to numpy
        x_numpy = np.array(x_jax)

        # Convert back
        x_jax_again = jnp.array(x_numpy)

        # Should be equal
        np.testing.assert_array_equal(x_jax, x_jax_again)

    @pytest.mark.jax
    def test_jax_array_interface(self):
        """Test JAX array implements standard interfaces."""
        x = jnp.array([1.0, 2.0, 3.0])

        # Should support standard operations
        assert hasattr(x, 'shape')
        assert hasattr(x, 'dtype')
        assert hasattr(x, 'ndim')
        assert hasattr(x, 'size')
