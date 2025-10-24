"""Tests for Smooth Derivative transform."""

import pytest
import numpy as np
import jax.numpy as jnp

from rheo.core.data import RheoData
from rheo.transforms.smooth_derivative import SmoothDerivative


class TestSmoothDerivative:
    """Test suite for Smooth Derivative transform."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        deriv = SmoothDerivative()
        assert deriv.method == 'savgol'
        assert deriv.window_length == 11
        assert deriv.polyorder == 3
        assert deriv.deriv == 1

    def test_custom_initialization(self):
        """Test custom parameters."""
        deriv = SmoothDerivative(
            method='finite_diff',
            window_length=21,
            polyorder=5,
            deriv=2
        )
        assert deriv.method == 'finite_diff'
        assert deriv.window_length == 21
        assert deriv.polyorder == 5
        assert deriv.deriv == 2

    def test_initialization_validation(self):
        """Test parameter validation."""
        # Even window length should raise error
        with pytest.raises(ValueError, match="odd"):
            SmoothDerivative(window_length=10)

        # polyorder >= window_length should raise error
        with pytest.raises(ValueError, match="less than"):
            SmoothDerivative(window_length=11, polyorder=11)

        # deriv < 1 should raise error
        with pytest.raises(ValueError, match="at least 1"):
            SmoothDerivative(deriv=0)

    def test_linear_function_derivative(self):
        """Test derivative of linear function."""
        # dy/dx of (y = 2x + 3) should be 2
        x = jnp.linspace(0, 10, 100)
        y = 2 * x + 3

        data = RheoData(x=x, y=y, domain='time')

        deriv = SmoothDerivative(method='savgol', window_length=11, polyorder=2)
        dy_dx = deriv.transform(data)

        # Derivative should be approximately 2
        assert jnp.allclose(dy_dx.y, 2.0, atol=0.01)

    def test_quadratic_function_derivative(self):
        """Test derivative of quadratic function."""
        # dy/dx of (y = x²) should be 2x
        x = jnp.linspace(0, 10, 200)
        y = x ** 2

        data = RheoData(x=x, y=y, domain='time')

        deriv = SmoothDerivative(method='savgol', window_length=11, polyorder=3)
        dy_dx = deriv.transform(data)

        # Derivative should be approximately 2x
        expected = 2 * x
        assert jnp.allclose(dy_dx.y, expected, atol=0.1)

    def test_second_derivative(self):
        """Test second derivative calculation."""
        # d²y/dx² of (y = x²) should be 2
        x = jnp.linspace(0, 10, 200)
        y = x ** 2

        data = RheoData(x=x, y=y, domain='time')

        deriv = SmoothDerivative(method='savgol', window_length=11, polyorder=4, deriv=2)
        d2y_dx2 = deriv.transform(data)

        # Second derivative should be approximately 2
        assert jnp.allclose(d2y_dx2.y, 2.0, atol=0.1)

    def test_exponential_derivative(self):
        """Test derivative of exponential function."""
        # dy/dx of (y = exp(x)) should be exp(x)
        x = jnp.linspace(0, 5, 200)
        y = jnp.exp(x)

        data = RheoData(x=x, y=y, domain='time')

        deriv = SmoothDerivative(method='savgol', window_length=11, polyorder=3)
        dy_dx = deriv.transform(data)

        # Derivative should be approximately exp(x)
        expected = jnp.exp(x)

        # Check correlation (handles edge effects)
        correlation = np.corrcoef(np.array(dy_dx.y), np.array(expected))[0, 1]
        assert correlation > 0.99

    def test_noisy_data_smoothing(self):
        """Test derivative on noisy data."""
        # Create noisy linear data
        np.random.seed(42)
        x = jnp.linspace(0, 10, 200)
        y_true = 2 * x + 3
        noise = 0.5 * np.random.randn(len(x))
        y_noisy = y_true + noise

        data = RheoData(x=x, y=y_noisy, domain='time')

        # Savitzky-Golay should smooth noise
        deriv = SmoothDerivative(method='savgol', window_length=21, polyorder=3)
        dy_dx = deriv.transform(data)

        # Derivative should be close to 2 (with some noise tolerance)
        assert jnp.abs(jnp.mean(dy_dx.y) - 2.0) < 0.2
        assert jnp.std(dy_dx.y) < 0.5  # Reduced noise

    def test_different_methods(self):
        """Test different differentiation methods."""
        x = jnp.linspace(0, 10, 200)
        y = x ** 2
        data = RheoData(x=x, y=y, domain='time')

        methods = ['savgol', 'finite_diff', 'spline']

        for method in methods:
            if method == 'savgol':
                deriv = SmoothDerivative(method=method, window_length=11, polyorder=3)
            else:
                deriv = SmoothDerivative(method=method)

            dy_dx = deriv.transform(data)

            # All should give reasonable results
            expected = 2 * x
            correlation = np.corrcoef(np.array(dy_dx.y), np.array(expected))[0, 1]
            assert correlation > 0.95

    def test_inverse_transform_integration(self):
        """Test inverse transform (integration)."""
        # Create derivative data
        x = jnp.linspace(0, 10, 200)
        dy_dx = 2 * jnp.ones_like(x)  # Constant derivative

        data = RheoData(x=x, y=dy_dx, domain='time')

        deriv = SmoothDerivative()
        integrated = deriv.inverse_transform(data)

        # Integrated result should be linear (2x + C)
        # Check that it's approximately linear
        fitted_slope = (integrated.y[-1] - integrated.y[0]) / (x[-1] - x[0])
        assert jnp.abs(fitted_slope - 2.0) < 0.1

    def test_pre_smoothing(self):
        """Test pre-smoothing option."""
        np.random.seed(42)
        x = jnp.linspace(0, 10, 200)
        y_true = 2 * x
        noise = 1.0 * np.random.randn(len(x))
        y_noisy = y_true + noise

        data = RheoData(x=x, y=y_noisy, domain='time')

        # With pre-smoothing
        deriv_smooth = SmoothDerivative(smooth_before=True, smooth_window=11)
        dy_dx_smooth = deriv_smooth.transform(data)

        # Without pre-smoothing
        deriv_no_smooth = SmoothDerivative(smooth_before=False)
        dy_dx_no_smooth = deriv_no_smooth.transform(data)

        # Pre-smoothed should have less noise
        assert jnp.std(dy_dx_smooth.y) < jnp.std(dy_dx_no_smooth.y)

    def test_post_smoothing(self):
        """Test post-smoothing option."""
        np.random.seed(42)
        x = jnp.linspace(0, 10, 200)
        y_true = x ** 2
        noise = 2.0 * np.random.randn(len(x))
        y_noisy = y_true + noise

        data = RheoData(x=x, y=y_noisy, domain='time')

        # With post-smoothing
        deriv_smooth = SmoothDerivative(smooth_after=True, smooth_window=11)
        dy_dx_smooth = deriv_smooth.transform(data)

        # Without post-smoothing
        deriv_no_smooth = SmoothDerivative(smooth_after=False)
        dy_dx_no_smooth = deriv_no_smooth.transform(data)

        # Post-smoothed should have smoother derivative
        assert jnp.std(dy_dx_smooth.y) < jnp.std(dy_dx_no_smooth.y)

    def test_complex_data_handling(self):
        """Test handling of complex data."""
        x = jnp.linspace(0, 10, 200)
        y_complex = (2 * x + 3) + 1j * x  # Complex linear

        data = RheoData(x=x, y=y_complex, domain='time')

        deriv = SmoothDerivative()
        dy_dx = deriv.transform(data)

        # Should take real part and differentiate
        assert jnp.all(jnp.isfinite(dy_dx.y))

    def test_non_uniform_spacing(self):
        """Test derivative with non-uniform spacing."""
        # Non-uniform spacing
        x = jnp.concatenate([
            jnp.linspace(0, 1, 20),
            jnp.linspace(1, 10, 180)
        ])
        y = 2 * x + 3

        data = RheoData(x=x, y=y, domain='time')

        deriv = SmoothDerivative(method='finite_diff')
        dy_dx = deriv.transform(data)

        # Should still give reasonable derivative
        assert jnp.abs(jnp.mean(dy_dx.y) - 2.0) < 0.2

    def test_noise_estimation(self):
        """Test noise level estimation."""
        np.random.seed(42)
        x = jnp.linspace(0, 10, 200)
        y_true = 2 * x
        noise_std = 0.5
        noise = noise_std * np.random.randn(len(x))
        y_noisy = y_true + noise

        data = RheoData(x=x, y=y_noisy, domain='time')

        deriv = SmoothDerivative()
        estimated_noise = deriv.estimate_noise_level(data)

        # Should be in the right ballpark
        assert 0.1 < estimated_noise < 2.0

    def test_y_units_update(self):
        """Test that y_units are correctly updated."""
        x = jnp.linspace(0, 10, 100)
        y = 2 * x

        data = RheoData(x=x, y=y, x_units='s', y_units='Pa', domain='time')

        # First derivative
        deriv1 = SmoothDerivative(deriv=1)
        dy_dx = deriv1.transform(data)

        assert 'Pa' in dy_dx.y_units
        assert 's' in dy_dx.y_units

        # Second derivative
        deriv2 = SmoothDerivative(deriv=2)
        d2y_dx2 = deriv2.transform(data)

        assert '^2' in d2y_dx2.y_units or 'order_2' in d2y_dx2.y_units

    def test_metadata_preservation(self):
        """Test metadata preservation."""
        x = jnp.linspace(0, 10, 100)
        y = 2 * x

        data = RheoData(
            x=x,
            y=y,
            domain='time',
            metadata={'sample': 'polymer', 'temperature': 298}
        )

        deriv = SmoothDerivative()
        dy_dx = deriv.transform(data)

        # Original metadata preserved
        assert dy_dx.metadata['sample'] == 'polymer'
        assert dy_dx.metadata['temperature'] == 298

        # Transform metadata added
        assert 'transform' in dy_dx.metadata
        assert dy_dx.metadata['transform'] == 'derivative'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
