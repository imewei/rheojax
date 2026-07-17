"""Pure-JAX cubic spline interpolation, replacing interpax.

interpax pins ``jax<0.11`` in its own dependencies (unreleased upstream fix as
of 2026-07-17), which blocks raising rheojax's own jax ceiling. This module
reproduces the exact numerics of the two interpax schemes rheojax used, so no
call site's output changes:

- :func:`local_cubic_eval` matches ``interpax.Interpolator1D(method="cubic")``
  / ``interpax.interp1d(method="cubic")``: a C1 Catmull-Rom-style cubic
  Hermite spline whose node tangents are the average of adjacent secant
  slopes (see ``interpax._fd_derivs._cubic1``). No linear solve -- O(n)
  closed form.
- :class:`NotAKnotCubicSpline` matches ``interpax.CubicSpline`` (default
  ``bc_type="not-a-knot"``), which is itself the same algorithm as
  ``scipy.interpolate.CubicSpline(bc_type="not-a-knot")``. The tridiagonal
  system below is transcribed directly from
  ``scipy.interpolate._cubic.CubicSpline.__init__`` and solved with a JAX
  Thomas algorithm (O(n), not a dense O(n^3) solve).

Numerically validated in ``tests/utils/test_jax_cubic_spline.py`` (random
arrays, edge cases n=2/3, derivative orders 0-3, exact cubic reproduction)
against ``scipy.interpolate.CubicSpline``/``CubicHermiteSpline`` -- interpax
itself is no longer a dependency, so scipy is the committed oracle. Matches
to ~1e-8 for values/1st derivatives, looser (~1e-4) for the noisier 3rd
derivative.

One deliberate behavioral difference from interpax: query points outside
``[x[0], x[-1]]`` are cubically extrapolated here (clipped to the boundary
segment), whereas interpax's default ``extrap=False`` returns NaN outside the
domain. No current call site queries out-of-bounds points, so this is latent;
a future caller relying on NaN-on-out-of-bounds should clamp/mask explicitly.
"""

from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()


def _thomas_solve(sub, diag, sup, rhs):
    """Solve a tridiagonal system via the Thomas algorithm.

    ``sub[i]``/``sup[i]`` are the coefficients of ``x[i-1]``/``x[i+1]`` in
    row ``i``; ``sub[0]`` and ``sup[-1]`` are unused padding.

    # ponytail: unpivoted (unlike scipy's solve_banded), valid for the
    # diagonally-dominant systems well-conditioned monotone grids produce;
    # add partial pivoting if ever fed extreme non-uniform spacing.
    """
    sup0 = sup[0] / diag[0]
    rhs0 = rhs[0] / diag[0]

    def forward(carry, row):
        sup_prev, rhs_prev = carry
        sub_i, diag_i, sup_i, rhs_i = row
        w = 1.0 / (diag_i - sub_i * sup_prev)
        sup_star = sup_i * w
        rhs_star = (rhs_i - sub_i * rhs_prev) * w
        return (sup_star, rhs_star), (sup_star, rhs_star)

    _, (sup_star_rest, rhs_star_rest) = jax.lax.scan(
        forward, (sup0, rhs0), (sub[1:], diag[1:], sup[1:], rhs[1:])
    )
    sup_star = jnp.concatenate([sup0[None], sup_star_rest])
    rhs_star = jnp.concatenate([rhs0[None], rhs_star_rest])

    def backward(x_next, row):
        sup_star_i, rhs_star_i = row
        x_i = rhs_star_i - sup_star_i * x_next
        return x_i, x_i

    x_last = rhs_star[-1]
    _, x_rest_rev = jax.lax.scan(
        backward, x_last, (sup_star[:-1], rhs_star[:-1]), reverse=True
    )
    return jnp.concatenate([x_rest_rev, x_last[None]])


def _safe_slope(x, y):
    """``diff(y)/diff(x)``, matching interpax's zero-spacing guard.

    interpax's own ``_cubic1``/``_cubic2`` use
    ``jnp.where(dx == 0, 0, 1/dx)`` rather than dividing directly, so a
    duplicate x value yields a finite (zero) slope instead of inf/NaN.
    """
    dx = jnp.diff(x)
    dxi = jnp.where(dx == 0, 0.0, 1.0 / dx)
    return jnp.diff(y) * dxi


def _local_cubic_slopes(x, y):
    """Node tangents matching interpax's ``_cubic1`` (centered secant average)."""
    slope = _safe_slope(x, y)
    interior = 0.5 * (slope[:-1] + slope[1:])
    return jnp.concatenate([slope[:1], interior, slope[-1:]])


def _not_a_knot_slopes(x, y):
    """Node slopes for the not-a-knot C2 cubic spline.

    Transcribed from ``scipy.interpolate._cubic.CubicSpline.__init__``
    (BSD-3-Clause), restricted to the ``bc_type="not-a-knot"`` (both ends)
    case that interpax's ``CubicSpline`` default uses.
    """
    n = x.shape[0]
    dx = jnp.diff(x)
    slope = _safe_slope(x, y)

    if n == 2:
        return jnp.array([slope[0], slope[0]])

    if n == 3:
        a = jnp.array(
            [
                [1.0, 1.0, 0.0],
                [dx[1], 2 * (dx[0] + dx[1]), dx[0]],
                [0.0, 1.0, 1.0],
            ]
        )
        b = jnp.array(
            [
                2 * slope[0],
                3 * (dx[0] * slope[1] + dx[1] * slope[0]),
                2 * slope[1],
            ]
        )
        return jnp.linalg.solve(a, b)

    sub = jnp.zeros(n)
    diag = jnp.zeros(n)
    sup = jnp.zeros(n)
    rhs = jnp.zeros(n)

    sub = sub.at[1:-1].set(dx[1:])
    diag = diag.at[1:-1].set(2 * (dx[:-1] + dx[1:]))
    sup = sup.at[1:-1].set(dx[:-1])
    rhs = rhs.at[1:-1].set(3 * (dx[1:] * slope[:-1] + dx[:-1] * slope[1:]))

    d0 = x[2] - x[0]
    diag = diag.at[0].set(dx[1])
    sup = sup.at[0].set(d0)
    rhs = rhs.at[0].set(
        ((dx[0] + 2 * d0) * dx[1] * slope[0] + dx[0] ** 2 * slope[1]) / d0
    )

    dn = x[-1] - x[-3]
    sub = sub.at[-1].set(dn)
    diag = diag.at[-1].set(dx[-2])
    rhs = rhs.at[-1].set(
        (dx[-1] ** 2 * slope[-2] + (2 * dn + dx[-1]) * dx[-2] * slope[-1]) / dn
    )

    return _thomas_solve(sub, diag, sup, rhs)


def _hermite_eval(x, y, s, xq, nu=0):
    """Evaluate a cubic Hermite spline (or its ``nu``-th derivative) at ``xq``."""
    n = x.shape[0]
    idx = jnp.clip(jnp.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    m0, m1 = s[idx], s[idx + 1]
    dx = x1 - x0
    t = (xq - x0) / dx

    if nu == 0:
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        return h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1
    elif nu == 1:
        h00 = 6 * t**2 - 6 * t
        h10 = 3 * t**2 - 4 * t + 1
        h01 = -6 * t**2 + 6 * t
        h11 = 3 * t**2 - 2 * t
        return (h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1) / dx
    elif nu == 2:
        h00 = 12 * t - 6
        h10 = 6 * t - 4
        h01 = -12 * t + 6
        h11 = 6 * t - 2
        return (h00 * y0 + h10 * dx * m0 + h01 * y1 + h11 * dx * m1) / dx**2
    elif nu == 3:
        return (12.0 * y0 + 6.0 * dx * m0 - 12.0 * y1 + 6.0 * dx * m1) / dx**3
    else:
        raise ValueError(f"nu must be 0, 1, 2, or 3, got {nu}")


def local_cubic_eval(x, y, xq, nu=0):
    """C1 local cubic Hermite spline, matching ``interpax`` ``method="cubic"``."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    s = _local_cubic_slopes(x, y)
    return _hermite_eval(x, y, s, jnp.asarray(xq), nu=nu)


class NotAKnotCubicSpline:
    """C2 not-a-knot cubic spline, matching ``interpax.CubicSpline`` default."""

    def __init__(self, x, y):
        self.x = jnp.asarray(x)
        self.y = jnp.asarray(y)
        # Matches scipy.interpolate.CubicSpline: a duplicate/non-monotonic x
        # makes the not-a-knot boundary rows singular (e.g. diag[0] = dx[1]
        # is exactly zero for a duplicate at index 1), so reject upfront
        # rather than silently returning NaN/Inf from a singular solve.
        if not bool(jnp.all(jnp.diff(self.x) > 0)):
            raise ValueError(
                "NotAKnotCubicSpline requires strictly increasing x (no "
                "duplicate or non-monotonic values); the not-a-knot system "
                "is singular otherwise."
            )
        self.s = _not_a_knot_slopes(self.x, self.y)

    def __call__(self, xq, nu=0):
        return _hermite_eval(self.x, self.y, self.s, jnp.asarray(xq), nu=nu)

    def derivative(self, nu=1):
        return _BoundDerivative(self, nu)


class _BoundDerivative:
    """Callable returned by :meth:`NotAKnotCubicSpline.derivative`."""

    def __init__(self, spline, nu):
        self._spline = spline
        self._nu = nu

    def __call__(self, xq):
        return self._spline(xq, nu=self._nu)
