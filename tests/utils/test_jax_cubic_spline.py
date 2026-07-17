"""Tests for rheojax.utils.jax_cubic_spline.

interpax is no longer a rheojax dependency (see pyproject.toml), so these
tests validate against scipy.interpolate (a real dependency) rather than
interpax directly.
"""

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from rheojax.utils.jax_cubic_spline import NotAKnotCubicSpline, local_cubic_eval


@pytest.mark.smoke
class TestNotAKnotCubicSpline:
    """C2 not-a-knot spline: parity against scipy.interpolate.CubicSpline."""

    def test_exact_cubic_reproduction(self):
        """A global cubic is reproduced exactly (0th-3rd derivatives)."""
        x = np.array([0.0, 0.7, 1.5, 2.6, 4.0, 5.5, 7.0])
        y = x**3 - 2 * x**2 + 3 * x - 1
        xq = np.linspace(x[0], x[-1], 37)

        spline = NotAKnotCubicSpline(x, y)
        np.testing.assert_allclose(
            np.asarray(spline(xq)), xq**3 - 2 * xq**2 + 3 * xq - 1, atol=1e-8
        )
        np.testing.assert_allclose(
            np.asarray(spline.derivative(nu=1)(xq)), 3 * xq**2 - 4 * xq + 3, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(spline.derivative(nu=2)(xq)), 6 * xq - 4, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(spline.derivative(nu=3)(xq)), np.full_like(xq, 6.0), atol=1e-5
        )

    @pytest.mark.parametrize("n", [4, 5, 10, 50])
    def test_matches_scipy_not_a_knot(self, n):
        rng = np.random.default_rng(42 + n)
        x = np.sort(rng.uniform(0, 10, n))
        y = rng.normal(size=n)
        xq = rng.uniform(x[0], x[-1], 100)

        ref = CubicSpline(x, y, bc_type="not-a-knot")
        mine = NotAKnotCubicSpline(x, y)

        np.testing.assert_allclose(np.asarray(mine(xq)), ref(xq), atol=1e-8)
        np.testing.assert_allclose(
            np.asarray(mine.derivative(nu=1)(xq)), ref(xq, 1), atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(mine.derivative(nu=2)(xq)), ref(xq, 2), atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(mine.derivative(nu=3)(xq)), ref(xq, 3), atol=1e-4
        )

    def test_n_equals_2(self):
        """n=2 degenerates to a straight line."""
        x = np.array([1.0, 4.0])
        y = np.array([2.0, 11.0])
        ref = CubicSpline(x, y, bc_type="not-a-knot")
        mine = NotAKnotCubicSpline(x, y)
        xq = np.linspace(1.0, 4.0, 5)
        np.testing.assert_allclose(np.asarray(mine(xq)), ref(xq), atol=1e-10)

    def test_n_equals_3(self):
        """n=3 uses the parabola special case."""
        x = np.array([0.0, 1.0, 3.0])
        y = np.array([0.0, 1.0, 9.0])
        ref = CubicSpline(x, y, bc_type="not-a-knot")
        mine = NotAKnotCubicSpline(x, y)
        xq = np.linspace(0.0, 3.0, 11)
        np.testing.assert_allclose(np.asarray(mine(xq)), ref(xq), atol=1e-10)

    def test_duplicate_x_raises(self):
        """Matches scipy.interpolate.CubicSpline: rejects duplicate x with a
        clear error instead of silently returning NaN/Inf from a singular
        not-a-knot system."""
        x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 1.0, 4.0, 9.0])
        with pytest.raises(ValueError, match="strictly increasing"):
            NotAKnotCubicSpline(x, y)

    def test_extrapolation_is_finite_not_nan(self):
        """Documented behavioral difference from interpax: finite extrapolation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 4.0, 9.0])
        spline = NotAKnotCubicSpline(x, y)
        out = np.asarray(spline(np.array([-1.0, 4.0])))
        assert np.all(np.isfinite(out))


@pytest.mark.smoke
class TestLocalCubicEval:
    """C1 local Catmull-Rom-style Hermite spline."""

    @staticmethod
    def _reference(x, y, xq, nu=0):
        """Independent reference: same tangent rule, scipy's Hermite basis."""
        dx = np.diff(x)
        slope = np.diff(y) / dx
        interior = 0.5 * (slope[:-1] + slope[1:])
        tangents = np.concatenate([slope[:1], interior, slope[-1:]])
        return CubicHermiteSpline(x, y, tangents)(xq, nu)

    @pytest.mark.parametrize("n", [4, 5, 10, 50])
    def test_matches_reference_hermite(self, n):
        rng = np.random.default_rng(7 + n)
        x = np.sort(rng.uniform(0, 10, n))
        y = rng.normal(size=n)
        xq = rng.uniform(x[0], x[-1], 100)

        ref = self._reference(x, y, xq)
        mine = np.asarray(local_cubic_eval(x, y, xq))
        np.testing.assert_allclose(mine, ref, atol=1e-8)

    def test_duplicate_x_yields_finite_output(self):
        x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 1.0, 4.0, 9.0])
        xq = np.linspace(0.0, 3.0, 20)
        out = np.asarray(local_cubic_eval(x, y, xq))
        assert np.all(np.isfinite(out))

    def test_derivative_orders(self):
        x = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
        y = np.array([0.0, 1.0, 2.0, 1.5, 3.0])
        xq = np.linspace(x[0], x[-1], 40)
        for nu in (0, 1, 2, 3):
            ref = self._reference(x, y, xq, nu=nu)
            mine = np.asarray(local_cubic_eval(x, y, xq, nu=nu))
            np.testing.assert_allclose(mine, ref, atol=1e-6)

    def test_invalid_nu_raises(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 4.0])
        with pytest.raises(ValueError, match="nu must be"):
            local_cubic_eval(x, y, np.array([0.5]), nu=4)
