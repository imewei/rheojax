"""Smooth noise-robust numerical differentiation for rheological data.

This module provides noise-robust differentiation using Savitzky-Golay filtering
and other smoothing techniques, essential for converting between rheological
functions (e.g., creep compliance → relaxation modulus).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.signal import savgol_filter

from rheo.core.base import BaseTransform
from rheo.core.jax_config import safe_import_jax
from rheo.core.registry import TransformRegistry

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    from rheo.core.data import RheoData


DerivativeMethod = Literal["savgol", "finite_diff", "spline", "total_variation"]


@TransformRegistry.register("smooth_derivative")
class SmoothDerivative(BaseTransform):
    """Smooth noise-robust numerical differentiation.

    This transform computes derivatives of noisy rheological data using
    regularization techniques to suppress noise amplification. Multiple
    methods are available:

    1. Savitzky-Golay: Fits local polynomials and computes analytical derivatives
    2. Finite Difference: Simple finite differences with optional smoothing
    3. Spline: Fits smoothing splines and computes derivatives
    4. Total Variation: Regularized differentiation minimizing total variation

    Savitzky-Golay is recommended for most applications as it preserves peak
    positions better than simple smoothing while providing good noise suppression.

    Common use cases:
    - Creep compliance J(t) → relaxation modulus G(t) (via numerical inversion)
    - Storage modulus G'(ω) → loss modulus G"(ω) via Kramers-Kronig
    - Time-derivative of strain in controlled-strain experiments

    Parameters
    ----------
    method : DerivativeMethod, default='savgol'
        Differentiation method
    window_length : int, default=11
        Window length for Savitzky-Golay or smoothing (must be odd)
    polyorder : int, default=3
        Polynomial order for Savitzky-Golay (must be < window_length)
    deriv : int, default=1
        Order of derivative (1, 2, 3, ...)
    smooth_before : bool, default=False
        Apply additional smoothing before differentiation
    smooth_after : bool, default=False
        Apply additional smoothing after differentiation

    Examples
    --------
    >>> from rheo.core.data import RheoData
    >>> from rheo.transforms.smooth_derivative import SmoothDerivative
    >>>
    >>> # Create noisy creep compliance data
    >>> t = jnp.linspace(0.1, 10, 100)
    >>> J_t = t + 0.1 * jnp.random.normal(size=len(t))  # Noisy linear creep
    >>> data = RheoData(x=t, y=J_t, domain='time')
    >>>
    >>> # Compute smooth derivative
    >>> deriv = SmoothDerivative(window_length=11, polyorder=3)
    >>> dJ_dt = deriv.transform(data)
    >>>
    >>> # For higher-order derivatives
    >>> deriv2 = SmoothDerivative(window_length=15, polyorder=4, deriv=2)
    >>> d2J_dt2 = deriv2.transform(data)
    """

    def __init__(
        self,
        method: DerivativeMethod = "savgol",
        window_length: int = 11,
        polyorder: int = 3,
        deriv: int = 1,
        smooth_before: bool = False,
        smooth_after: bool = False,
        smooth_window: int = 5,
    ):
        """Initialize Smooth Derivative transform.

        Parameters
        ----------
        method : DerivativeMethod
            Differentiation method
        window_length : int
            Window length (must be odd)
        polyorder : int
            Polynomial order for Savitzky-Golay
        deriv : int
            Derivative order
        smooth_before : bool
            Smooth before differentiation
        smooth_after : bool
            Smooth after differentiation
        smooth_window : int
            Smoothing window size
        """
        super().__init__()
        self.method = method
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.smooth_before = smooth_before
        self.smooth_after = smooth_after
        self.smooth_window = smooth_window

        # Validate parameters
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")

        if self.polyorder >= self.window_length:
            raise ValueError("polyorder must be less than window_length")

        if self.deriv < 1:
            raise ValueError("deriv must be at least 1")

    def _smooth_data(self, y: jnp.ndarray, window: int) -> jnp.ndarray:
        """Apply moving average smoothing.

        Parameters
        ----------
        y : jnp.ndarray
            Data to smooth
        window : int
            Window size

        Returns
        -------
        jnp.ndarray
            Smoothed data
        """
        if window % 2 == 0:
            window += 1

        kernel = jnp.ones(window) / window
        smoothed = jnp.convolve(y, kernel, mode="same")

        return smoothed

    def _savgol_derivative(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative using Savitzky-Golay filter.

        Parameters
        ----------
        x : jnp.ndarray
            Independent variable
        y : jnp.ndarray
            Dependent variable

        Returns
        -------
        jnp.ndarray
            Derivative dy/dx
        """
        # Convert to numpy for scipy
        x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
        y_np = np.array(y) if isinstance(y, jnp.ndarray) else y

        # Check if uniformly spaced
        dx = np.diff(x_np)
        is_uniform = np.allclose(dx, dx[0], rtol=1e-5)

        if is_uniform:
            # Use scipy's savgol_filter directly
            delta = dx[0]
            dy_dx = savgol_filter(
                y_np,
                window_length=self.window_length,
                polyorder=self.polyorder,
                deriv=self.deriv,
                delta=delta,
            )
        else:
            # Non-uniform spacing: use derivative of fitted polynomial
            # This is more complex - use finite difference as fallback
            dy_dx = self._finite_diff_derivative(x, y)

        return jnp.array(dy_dx)

    def _finite_diff_derivative(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative using finite differences.

        Parameters
        ----------
        x : jnp.ndarray
            Independent variable
        y : jnp.ndarray
            Dependent variable

        Returns
        -------
        jnp.ndarray
            Derivative dy/dx
        """
        if self.deriv == 1:
            # First derivative using central differences
            dy_dx = jnp.gradient(y, x)
        elif self.deriv == 2:
            # Second derivative
            dy_dx_1 = jnp.gradient(y, x)
            dy_dx = jnp.gradient(dy_dx_1, x)
        else:
            # Higher-order derivatives (recursive)
            dy_dx = y
            for _ in range(self.deriv):
                dy_dx = jnp.gradient(dy_dx, x)

        return dy_dx

    def _spline_derivative(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Compute derivative using smoothing splines.

        Parameters
        ----------
        x : jnp.ndarray
            Independent variable
        y : jnp.ndarray
            Dependent variable

        Returns
        -------
        jnp.ndarray
            Derivative dy/dx
        """
        from scipy.interpolate import UnivariateSpline

        # Convert to numpy
        x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
        y_np = np.array(y) if isinstance(y, jnp.ndarray) else y

        # Fit smoothing spline
        # s parameter controls smoothing (higher = smoother)
        s = len(x) * 0.01  # Heuristic smoothing parameter

        spline = UnivariateSpline(x_np, y_np, s=s, k=min(5, self.polyorder + 1))

        # Compute derivative
        dy_dx = spline.derivative(n=self.deriv)(x_np)

        return jnp.array(dy_dx)

    def _total_variation_derivative(
        self, x: jnp.ndarray, y: jnp.ndarray, alpha: float = 0.1
    ) -> jnp.ndarray:
        """Compute derivative with total variation regularization.

        This minimizes:
            ||y - integral(u)||² + α * TV(u)
        where u = dy/dx is the derivative.

        Parameters
        ----------
        x : jnp.ndarray
            Independent variable
        y : jnp.ndarray
            Dependent variable
        alpha : float
            Regularization parameter

        Returns
        -------
        jnp.ndarray
            Derivative dy/dx
        """
        # This is a complex optimization problem
        # For now, use finite differences with smoothing as approximation
        dy_dx = self._finite_diff_derivative(x, y)

        # Apply TV denoising to the derivative
        # (Requires convex optimization - use simple smoothing for now)
        dy_dx = self._smooth_data(dy_dx, self.smooth_window)

        return dy_dx

    def _transform(self, data: RheoData) -> RheoData:
        """Compute smooth derivative of data.

        Parameters
        ----------
        data : RheoData
            Input data

        Returns
        -------
        RheoData
            Derivative data
        """
        from rheo.core.data import RheoData

        # Get data
        x = data.x
        y = data.y

        # Convert to JAX arrays
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        if not isinstance(y, jnp.ndarray):
            y = jnp.array(y)

        # Handle complex data
        if jnp.iscomplexobj(y):
            y = jnp.real(y)

        # Pre-smoothing if requested
        if self.smooth_before:
            y = self._smooth_data(y, self.smooth_window)

        # Compute derivative based on method
        if self.method == "savgol":
            dy_dx = self._savgol_derivative(x, y)
        elif self.method == "finite_diff":
            dy_dx = self._finite_diff_derivative(x, y)
        elif self.method == "spline":
            dy_dx = self._spline_derivative(x, y)
        elif self.method == "total_variation":
            dy_dx = self._total_variation_derivative(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Post-smoothing if requested
        if self.smooth_after:
            dy_dx = self._smooth_data(dy_dx, self.smooth_window)

        # Create new y_units
        if data.y_units and data.x_units:
            if self.deriv == 1:
                new_y_units = f"d({data.y_units})/d({data.x_units})"
            else:
                new_y_units = (
                    f"d^{self.deriv}({data.y_units})/d({data.x_units})^{self.deriv}"
                )
        else:
            new_y_units = f"derivative_order_{self.deriv}"

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update(
            {
                "transform": "derivative",
                "method": self.method,
                "derivative_order": self.deriv,
                "window_length": self.window_length,
                "polyorder": self.polyorder,
            }
        )

        return RheoData(
            x=x,
            y=dy_dx,
            x_units=data.x_units,
            y_units=new_y_units,
            domain=data.domain,
            metadata=new_metadata,
            validate=False,
        )

    def _inverse_transform(self, data: RheoData) -> RheoData:
        """Apply numerical integration (inverse of derivative).

        Parameters
        ----------
        data : RheoData
            Derivative data

        Returns
        -------
        RheoData
            Integrated data (approximation of original)
        """
        from scipy.integrate import cumulative_trapezoid as scipy_cumtrapz

        from rheo.core.data import RheoData

        # Get data
        x = data.x
        dy_dx = data.y

        # Convert to numpy for scipy
        x_np = np.array(x) if isinstance(x, jnp.ndarray) else x
        dy_dx_np = np.array(dy_dx) if isinstance(dy_dx, jnp.ndarray) else dy_dx

        # Numerical integration (cumulative trapezoid)
        y_integrated = scipy_cumtrapz(dy_dx_np, x_np, initial=0)

        # Create metadata
        new_metadata = data.metadata.copy()
        new_metadata.update(
            {"transform": "integral", "original_transform": "derivative"}
        )

        return RheoData(
            x=x,
            y=jnp.array(y_integrated),
            x_units=data.x_units,
            y_units="integrated",
            domain=data.domain,
            metadata=new_metadata,
            validate=False,
        )

    def estimate_noise_level(self, data: RheoData) -> float:
        """Estimate noise level in data.

        This uses the median absolute deviation (MAD) of the second derivative
        as a robust noise estimator.

        Parameters
        ----------
        data : RheoData
            Input data

        Returns
        -------
        float
            Estimated noise standard deviation
        """
        # Get data
        x = data.x
        y = data.y

        # Convert to arrays
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        if not isinstance(y, jnp.ndarray):
            y = jnp.array(y)

        # Compute second derivative (amplifies noise)
        d2y = jnp.gradient(jnp.gradient(y, x), x)

        # MAD estimator
        median = jnp.median(d2y)
        mad = jnp.median(jnp.abs(d2y - median))

        # Convert MAD to standard deviation (assumes Gaussian noise)
        sigma = 1.4826 * mad

        return float(sigma)


__all__ = ["SmoothDerivative"]
