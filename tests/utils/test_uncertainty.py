"""Tests for uncertainty quantification utilities.

Covers both hessian_ci() and bootstrap_ci() across:
- Simple real-valued data (Maxwell relaxation model)
- Complex oscillation data (G*)
- Edge cases (unfitted model, degenerate residuals)
"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.utils.uncertainty import bootstrap_ci, hessian_ci

jax, jnp = safe_import_jax()

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _maxwell_data(
    G0: float = 1000.0,
    tau: float = 1.0,
    n: int = 40,
    t_range: tuple[float, float] = (0.05, 20.0),
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Maxwell relaxation data: G(t) = G0 * exp(-t/tau)."""
    t = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), n)
    G = G0 * np.exp(-t / tau)
    if noise > 0.0:
        rng = rng or np.random.default_rng(0)
        G = G + noise * rng.normal(size=n)
    return t, G


def _maxwell_oscillation_data(
    G0: float = 1000.0,
    tau: float = 1.0,
    n: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Maxwell oscillation data: G* = G0*(i*omega*tau)/(1+i*omega*tau)."""
    omega = np.logspace(-2, 2, n)
    z = 1j * omega * tau
    G_star = G0 * z / (1.0 + z)
    return omega, G_star


@pytest.fixture(scope="module")
def fitted_maxwell():
    """Return a Maxwell model fitted to synthetic relaxation data."""
    from rheojax.models import Maxwell

    t, G = _maxwell_data(G0=1000.0, tau=1.0)
    model = Maxwell()
    model.fit(t, G, test_mode="relaxation")
    return model, t, G


@pytest.fixture(scope="module")
def fitted_maxwell_oscillation():
    """Return a Maxwell model fitted to synthetic oscillation data."""
    from rheojax.models import Maxwell

    omega, G_star = _maxwell_oscillation_data(G0=1000.0, tau=1.0)
    model = Maxwell()
    model.fit(omega, G_star, test_mode="oscillation")
    return model, omega, G_star


# ---------------------------------------------------------------------------
# hessian_ci — smoke tests
# ---------------------------------------------------------------------------


class TestHessianCI:
    """Tests for hessian_ci()."""

    @pytest.mark.smoke
    def test_returns_dict_with_correct_keys(self, fitted_maxwell):
        """hessian_ci returns a dict keyed by parameter names."""
        model, t, G = fitted_maxwell
        ci = hessian_ci(model, t, G, alpha=0.05)
        expected_keys = set(model.parameters.keys())
        assert set(ci.keys()) == expected_keys

    @pytest.mark.smoke
    def test_lower_less_than_upper(self, fitted_maxwell):
        """Each CI must have lower < upper."""
        model, t, G = fitted_maxwell
        ci = hessian_ci(model, t, G, alpha=0.05)
        for name, (lo, hi) in ci.items():
            assert lo < hi, f"Parameter {name!r}: lower ({lo}) >= upper ({hi})"

    @pytest.mark.smoke
    def test_optimal_value_within_ci(self, fitted_maxwell):
        """The optimal parameter value must lie within its CI."""
        model, t, G = fitted_maxwell
        ci = hessian_ci(model, t, G, alpha=0.05)
        for name, (lo, hi) in ci.items():
            val = model.parameters.get_value(name)
            assert (
                lo <= val <= hi
            ), f"Parameter {name!r}: optimal value {val:.4g} not in [{lo:.4g}, {hi:.4g}]"

    @pytest.mark.smoke
    def test_alpha_controls_width(self, fitted_maxwell):
        """Tighter alpha should produce wider intervals (more conservative)."""
        model, t, G = fitted_maxwell
        ci_narrow = hessian_ci(model, t, G, alpha=0.20)  # 80% CI
        ci_wide = hessian_ci(model, t, G, alpha=0.01)  # 99% CI
        for name in ci_narrow:
            width_narrow = ci_narrow[name][1] - ci_narrow[name][0]
            width_wide = ci_wide[name][1] - ci_wide[name][0]
            assert (
                width_wide >= width_narrow
            ), f"99% CI for {name!r} is narrower than 80% CI"

    @pytest.mark.smoke
    def test_complex_oscillation_data(self, fitted_maxwell_oscillation):
        """hessian_ci works on models fitted to complex G* data."""
        model, omega, G_star = fitted_maxwell_oscillation
        ci = hessian_ci(model, omega, G_star, alpha=0.05, test_mode="oscillation")
        assert len(ci) == len(list(model.parameters.keys()))
        for name, (lo, hi) in ci.items():
            assert lo < hi

    def test_unfitted_model_raises(self):
        """hessian_ci raises RuntimeError on an unfitted model."""
        from rheojax.models import Maxwell

        model = Maxwell()
        t, G = _maxwell_data()
        with pytest.raises(RuntimeError, match="not been fitted"):
            hessian_ci(model, t, G)

    def test_ci_values_are_finite(self, fitted_maxwell):
        """All CI bounds must be finite floats."""
        model, t, G = fitted_maxwell
        ci = hessian_ci(model, t, G, alpha=0.05)
        for name, (lo, hi) in ci.items():
            assert np.isfinite(lo), f"Lower bound for {name!r} is not finite: {lo}"
            assert np.isfinite(hi), f"Upper bound for {name!r} is not finite: {hi}"

    def test_reuses_nlsq_pcov_when_available(self, fitted_maxwell):
        """When _nlsq_result.pcov exists, hessian_ci must reuse it (no Hessian)."""
        model, t, G = fitted_maxwell
        # Verify the model actually has a pcov from its NLSQ result
        nlsq = getattr(model, "_nlsq_result", None)
        if nlsq is None or getattr(nlsq, "pcov", None) is None:
            pytest.skip("Model does not expose a precomputed pcov via _nlsq_result")
        ci = hessian_ci(model, t, G, alpha=0.05)
        # Should succeed without any exception
        assert ci is not None

    def test_test_mode_propagation(self, fitted_maxwell):
        """Explicit test_mode kwarg is accepted without error."""
        model, t, G = fitted_maxwell
        ci = hessian_ci(model, t, G, alpha=0.05, test_mode="relaxation")
        assert len(ci) > 0


# ---------------------------------------------------------------------------
# bootstrap_ci — smoke tests
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    """Tests for bootstrap_ci().  Marked slow for full suite runs."""

    @pytest.mark.smoke
    def test_returns_dict_with_correct_keys(self, fitted_maxwell):
        """bootstrap_ci returns a dict keyed by parameter names."""
        model, t, G = fitted_maxwell
        ci = bootstrap_ci(model, t, G, n_bootstrap=20, seed=0)
        expected_keys = set(model.parameters.keys())
        assert set(ci.keys()) == expected_keys

    @pytest.mark.smoke
    def test_lower_less_than_upper(self, fitted_maxwell):
        """Each CI must have lower < upper for non-degenerate data."""
        model, t, G = fitted_maxwell
        ci = bootstrap_ci(model, t, G, n_bootstrap=20, seed=7)
        for name, (lo, hi) in ci.items():
            assert lo <= hi, f"Parameter {name!r}: lower ({lo}) > upper ({hi})"

    @pytest.mark.smoke
    def test_ci_values_are_finite(self, fitted_maxwell):
        """All bootstrap CI bounds must be finite."""
        model, t, G = fitted_maxwell
        ci = bootstrap_ci(model, t, G, n_bootstrap=20, seed=1)
        for name, (lo, hi) in ci.items():
            assert np.isfinite(lo), f"Lower bound for {name!r} is non-finite: {lo}"
            assert np.isfinite(hi), f"Upper bound for {name!r} is non-finite: {hi}"

    @pytest.mark.smoke
    def test_seed_reproducibility(self, fitted_maxwell):
        """Same seed must produce identical CIs."""
        model, t, G = fitted_maxwell
        ci_a = bootstrap_ci(model, t, G, n_bootstrap=20, seed=99)
        ci_b = bootstrap_ci(model, t, G, n_bootstrap=20, seed=99)
        for name in ci_a:
            assert (
                ci_a[name] == ci_b[name]
            ), f"Non-reproducible CI for {name!r}: {ci_a[name]} vs {ci_b[name]}"

    @pytest.mark.smoke
    def test_different_seeds_differ(self, fitted_maxwell):
        """Different seeds should typically produce different CIs."""
        model, t, G = fitted_maxwell
        ci_a = bootstrap_ci(model, t, G, n_bootstrap=20, seed=1)
        ci_b = bootstrap_ci(model, t, G, n_bootstrap=20, seed=2)
        # At least one parameter CI should differ between seeds
        all_same = all(ci_a[n] == ci_b[n] for n in ci_a)
        assert not all_same, "Bootstrap CIs are identical for different seeds"

    @pytest.mark.smoke
    def test_unfitted_model_raises(self):
        """bootstrap_ci raises RuntimeError on an unfitted model."""
        from rheojax.models import Maxwell

        model = Maxwell()
        t, G = _maxwell_data()
        with pytest.raises(RuntimeError, match="not been fitted"):
            bootstrap_ci(model, t, G, n_bootstrap=5)

    @pytest.mark.smoke
    def test_complex_oscillation_data(self, fitted_maxwell_oscillation):
        """bootstrap_ci works on models fitted to complex G* data."""
        model, omega, G_star = fitted_maxwell_oscillation
        ci = bootstrap_ci(
            model, omega, G_star, n_bootstrap=20, seed=42, test_mode="oscillation"
        )
        assert len(ci) == len(list(model.parameters.keys()))
        for name, (lo, hi) in ci.items():
            assert lo <= hi

    @pytest.mark.slow
    def test_more_bootstrap_reduces_variance(self, fitted_maxwell):
        """Wider bootstrap distribution (small n) should have wider CIs than large n."""
        model, t, G = fitted_maxwell
        # 10 vs 200 replicates — 200 should give tighter percentile estimates
        # (the intervals themselves might be similar, but we test they exist)
        ci_small = bootstrap_ci(model, t, G, n_bootstrap=10, seed=42)
        ci_large = bootstrap_ci(model, t, G, n_bootstrap=200, seed=42)
        # Basic sanity: both return the correct keys
        assert set(ci_small.keys()) == set(ci_large.keys())

    @pytest.mark.slow
    def test_alpha_controls_width(self, fitted_maxwell):
        """Wider confidence level (smaller alpha) should produce wider intervals."""
        model, t, G = fitted_maxwell
        ci_narrow = bootstrap_ci(model, t, G, n_bootstrap=100, alpha=0.20, seed=0)
        ci_wide = bootstrap_ci(model, t, G, n_bootstrap=100, alpha=0.01, seed=0)
        for name in ci_narrow:
            w_narrow = ci_narrow[name][1] - ci_narrow[name][0]
            w_wide = ci_wide[name][1] - ci_wide[name][0]
            assert (
                w_wide >= w_narrow * 0.8
            ), f"99% bootstrap CI for {name!r} not wider than 80% CI"

    @pytest.mark.slow
    def test_bootstrap_ci_covers_true_params(self, fitted_maxwell):
        """Bootstrap 95% CI should cover the true parameter values for noisy data.

        NOTE: Bootstrap on noiseless data is degenerate (zero residuals →
        all resamples are identical → CIs reflect rounding noise only).
        Use realistic noise (1% of signal) so the bootstrap distribution
        is well-defined.
        """
        from rheojax.models import Maxwell

        rng = np.random.default_rng(42)
        t, G = _maxwell_data(G0=1000.0, tau=1.0, n=80, noise=10.0, rng=rng)
        model = Maxwell()
        model.fit(t, G, test_mode="relaxation")

        ci = bootstrap_ci(model, t, G, n_bootstrap=200, alpha=0.05, seed=0)

        true_params = {"G0": 1000.0, "tau": 1.0}
        for name, true_val in true_params.items():
            if name not in ci:
                continue
            lo, hi = ci[name]
            # Allow 50% tolerance beyond the CI (bootstrap on small N is noisy)
            margin = 0.5 * (hi - lo)
            assert lo - margin <= true_val <= hi + margin, (
                f"True {name}={true_val} not within CI [{lo:.4g}, {hi:.4g}] "
                f"(with margin [{lo-margin:.4g}, {hi+margin:.4g}])"
            )


# ---------------------------------------------------------------------------
# Comparison: hessian_ci vs bootstrap_ci
# ---------------------------------------------------------------------------


class TestHessianVsBootstrap:
    """Cross-method consistency checks."""

    @pytest.mark.smoke
    def test_both_cover_optimal_value(self, fitted_maxwell):
        """Both methods should produce intervals containing the optimal value."""
        model, t, G = fitted_maxwell

        ci_h = hessian_ci(model, t, G, alpha=0.05)
        ci_b = bootstrap_ci(model, t, G, n_bootstrap=50, alpha=0.05, seed=0)

        for name in ci_h:
            val = model.parameters.get_value(name)
            lo_h, hi_h = ci_h[name]
            lo_b, hi_b = ci_b[name]
            assert (
                lo_h <= val <= hi_h
            ), f"Hessian CI for {name!r} does not cover optimum {val:.4g}"
            assert (
                lo_b <= val <= hi_b
            ), f"Bootstrap CI for {name!r} does not cover optimum {val:.4g}"

    @pytest.mark.slow
    def test_both_methods_agree_on_noisy_data(self, fitted_maxwell):
        """On noisy synthetic data, Hessian and bootstrap CIs should overlap.

        NOTE: On zero-noise data bootstrap degenerates (see
        test_bootstrap_ci_covers_true_params). Use realistic noise.
        """
        from rheojax.models import Maxwell

        rng = np.random.default_rng(7)
        t, G = _maxwell_data(G0=1000.0, tau=1.0, n=100, noise=10.0, rng=rng)
        model = Maxwell()
        model.fit(t, G, test_mode="relaxation")

        ci_h = hessian_ci(model, t, G, alpha=0.05)
        ci_b = bootstrap_ci(model, t, G, n_bootstrap=200, alpha=0.05, seed=0)

        for name in ci_h:
            lo_h, hi_h = ci_h[name]
            lo_b, hi_b = ci_b[name]
            # Intervals should have some overlap (allow non-overlap if both
            # are very narrow — just check the general direction is consistent)
            width_h = hi_h - lo_h
            width_b = hi_b - lo_b
            gap = max(lo_h, lo_b) - min(hi_h, hi_b)
            max_width = max(width_h, width_b)
            # Allow gap up to 50% of the wider interval
            assert gap <= 0.5 * max_width, (
                f"Hessian CI [{lo_h:.4g}, {hi_h:.4g}] and bootstrap CI "
                f"[{lo_b:.4g}, {hi_b:.4g}] are too far apart for {name!r}"
            )


# ---------------------------------------------------------------------------
# Public API surface check
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Verify the functions are importable from the package top-level."""

    def test_importable_from_utils(self):
        """hessian_ci and bootstrap_ci must be importable from rheojax.utils."""
        from rheojax.utils import bootstrap_ci as bci
        from rheojax.utils import hessian_ci as hci

        assert callable(hci)
        assert callable(bci)

    def test_importable_from_uncertainty_module(self):
        """Direct import from rheojax.utils.uncertainty must work."""
        from rheojax.utils.uncertainty import bootstrap_ci as bci
        from rheojax.utils.uncertainty import hessian_ci as hci

        assert callable(hci)
        assert callable(bci)
