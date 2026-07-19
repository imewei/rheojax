"""Regression tests for Round 9 RCA (2026-02-25) prevention.

Guards against the 5 regression patterns introduced by previous AI agents:
  a. Always-true conditionals (if x or not x:)
  b. Removed dispatch calls without verifying replacement path
  c. Strain arrays used as time arrays without checking return semantics
  d. finally cleanup without success guards
  e. Overly broad exception swallowing

"""

from __future__ import annotations

import numpy as np
import pytest

from rheojax.core.bayesian import BayesianMixin
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import ParameterSet

jax, jnp = safe_import_jax()


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------


class _SimpleBayesianModel(BayesianMixin):
    """Minimal model for Bayesian regression tests."""

    def __init__(self):
        self.parameters = ParameterSet()
        self.parameters.add("a", value=2.0, bounds=(0.1, 10.0))
        self.parameters.add("b", value=1.0, bounds=(0.1, 10.0))

    def model_function(self, X, params, test_mode=None, **kwargs):
        a, b = params
        return a * X + b


class _FailingBayesianModel(BayesianMixin):
    """Model whose fit_bayesian always raises during MCMC."""

    def __init__(self):
        self.parameters = ParameterSet()
        self.parameters.add("a", value=1.0, bounds=(0.1, 5.0))

    def model_function(self, X, params, test_mode=None, **kwargs):
        raise RuntimeError("Intentional MCMC failure")


# ═══════════════════════════════════════════════════════════════════════════
# Category 1: ITT-MCT oscillation return_components shape contract
#   Regression pattern (a): always-true conditional
# ═══════════════════════════════════════════════════════════════════════════


class TestITTMCTOscillationShape:
    """Guard: `if return_components:` must NOT be always-true."""

    @pytest.mark.slow
    def test_schematic_default_returns_1d(self):
        """Schematic predict(oscillation) without return_components → (N,)."""
        from rheojax.models.itt_mct import ITTMCTSchematic

        model = ITTMCTSchematic(epsilon=0.05)
        omega = np.logspace(-1, 1, 5)
        result = model.predict(omega, test_mode="oscillation")
        assert result.ndim == 1, (
            f"Default oscillation predict must return 1-D |G*|, got shape {result.shape}"
        )
        assert result.shape == (5,)

    @pytest.mark.slow
    def test_schematic_components_returns_2d(self):
        """Schematic predict(oscillation, return_components=True) → (N, 2)."""
        from rheojax.models.itt_mct import ITTMCTSchematic

        model = ITTMCTSchematic(epsilon=0.05)
        omega = np.logspace(-1, 1, 5)
        result = model.predict(omega, test_mode="oscillation", return_components=True)
        assert result.ndim == 2, (
            f"return_components=True must return 2-D [G', G''], got shape {result.shape}"
        )
        assert result.shape == (5, 2)

    @pytest.mark.smoke
    def test_isotropic_default_returns_1d(self):
        """ISM predict(oscillation) without return_components → (N,)."""
        from rheojax.models.itt_mct import ITTMCTIsotropic

        model = ITTMCTIsotropic(phi=0.55, n_k=20)
        omega = np.logspace(-1, 1, 5)
        result = model.predict(omega, test_mode="oscillation")
        assert result.ndim == 1, (
            f"Default oscillation predict must return 1-D |G*|, got shape {result.shape}"
        )

    @pytest.mark.smoke
    def test_isotropic_components_returns_2d(self):
        """ISM predict(oscillation, return_components=True) → (N, 2)."""
        from rheojax.models.itt_mct import ITTMCTIsotropic

        model = ITTMCTIsotropic(phi=0.55, n_k=20)
        omega = np.logspace(-1, 1, 5)
        result = model.predict(omega, test_mode="oscillation", return_components=True)
        assert result.ndim == 2
        assert result.shape == (5, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Category 2: Bayesian _test_mode persistence after successful fit
#   Regression pattern (d): finally cleanup without success guard
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianTestModePersistence:
    """Guard: _test_mode must persist after successful fit_bayesian."""

    @pytest.mark.smoke
    def test_test_mode_survives_successful_fit(self):
        """After successful fit_bayesian, _test_mode must NOT be deleted."""
        model = _SimpleBayesianModel()
        X = np.linspace(0.1, 5, 30)
        y = 2.0 * X + 1.0

        # Fresh model: no _test_mode yet
        assert not hasattr(model, "_test_mode")

        result = model.fit_bayesian(
            X,
            y,
            num_warmup=20,
            num_samples=30,
            num_chains=1,
            seed=42,
            test_mode="relaxation",
        )
        assert result is not None

        # _test_mode must survive the finally block
        assert hasattr(model, "_test_mode"), (
            "_test_mode was deleted by the finally block after a successful fit"
        )

    @pytest.mark.smoke
    def test_test_mode_reverted_on_failure(self):
        """After failed fit_bayesian, state must be reverted."""
        model = _FailingBayesianModel()
        # Pre-set a _test_mode that should be restored on failure
        model._test_mode = "oscillation"
        model._last_fit_kwargs = {"foo": "bar"}
        saved_kwargs = model._last_fit_kwargs.copy()

        X = np.linspace(0.1, 5, 10)
        y = np.ones(10)

        with pytest.raises(Exception):
            model.fit_bayesian(
                X,
                y,
                num_warmup=5,
                num_samples=5,
                num_chains=1,
                seed=0,
                test_mode="relaxation",
            )

        # Original state must be restored
        assert model._test_mode == "oscillation", (
            "_test_mode was not reverted after failed fit_bayesian"
        )
        assert model._last_fit_kwargs == saved_kwargs


# ═══════════════════════════════════════════════════════════════════════════
# Category 3: SGR LAOS _predict uses monotonic time, not strain
#   Regression pattern (c): strain arrays used as time arrays
# ═══════════════════════════════════════════════════════════════════════════


class TestSGRLAOSTimeMonotonicity:
    """Guard: SGR LAOS _predict must build monotonic time grid for interp."""

    @pytest.mark.smoke
    def test_laos_predict_returns_finite(self):
        """SGR LAOS predict should return finite stress (not NaN from bad interp)."""
        from rheojax.models.sgr.sgr_generic import SGRGeneric

        model = SGRGeneric()
        # Set minimal params for LAOS
        model.parameters.set_value("x", 1.5)
        model.parameters.set_value("G0", 100.0)
        model.parameters.set_value("tau0", 1.0)
        model._test_mode = "laos"
        model._gamma_0 = 0.1
        model._omega_laos = 1.0
        model._n_cycles = 2

        omega = 1.0
        period = 2.0 * np.pi / omega
        n_pts = 20
        t_query = np.linspace(0, 2 * period, n_pts, endpoint=False)

        result = model.predict(
            t_query, test_mode="laos", gamma_0=0.1, omega_laos=1.0, n_cycles=2
        )
        assert result.shape == (n_pts,), f"Expected ({n_pts},), got {result.shape}"
        assert np.all(np.isfinite(result)), (
            "LAOS predict returned NaN — likely using non-monotonic strain as interp x-axis"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Category 4: Equal-bounds handled as deterministic (not Uniform crash)
#   Regression pattern (e): tolerance that incorrectly identified valid params as fixed
# ═══════════════════════════════════════════════════════════════════════════


class TestEqualBoundsDeterministic:
    """Guard: params with equal bounds → numpyro.deterministic, not crash."""

    @pytest.mark.smoke
    def test_equal_bounds_does_not_crash(self):
        """Param with bounds (5.0, 5.0) must be treated as deterministic."""
        model = _SimpleBayesianModel()
        # Pin param 'a' by setting equal bounds
        model.parameters["a"].bounds = (5.0, 5.0)

        X = np.linspace(0.1, 5, 20)
        y = 5.0 * X + 1.0

        result = model.fit_bayesian(
            X,
            y,
            num_warmup=20,
            num_samples=30,
            num_chains=1,
            seed=42,
            test_mode="relaxation",
        )
        assert result is not None

        # The fixed param should have constant posterior
        a_samples = result.posterior_samples.get("a")
        if a_samples is not None:
            unique = np.unique(np.asarray(a_samples))
            assert len(unique) == 1, (
                f"Equal-bounds param 'a' should be deterministic, got {len(unique)} unique values"
            )
            assert np.isclose(unique[0], 5.0, atol=1e-6)

    @pytest.mark.smoke
    def test_small_but_valid_range_not_treated_as_equal(self):
        """Param with bounds (0, 1e-6) must NOT be treated as fixed.

        The equal-bounds epsilon is 1e-9, so a range of 1e-6 is clearly
        a valid sampling range and must produce varied posterior samples.
        """
        model = _SimpleBayesianModel()
        # Range 1e-6 is well above the 1e-9 epsilon
        model.parameters["a"].bounds = (0.0, 1e-6)

        X = np.linspace(0.1, 5, 20)
        y = 1e-7 * X + 1.0

        result = model.fit_bayesian(
            X,
            y,
            num_warmup=10,
            num_samples=20,
            num_chains=1,
            seed=42,
            test_mode="relaxation",
        )
        assert result is not None
        a_samples = result.posterior_samples.get("a")
        if a_samples is not None:
            # Should have variation (not deterministic)
            assert not np.all(np.asarray(a_samples) == np.asarray(a_samples)[0]), (
                "Param with bounds (0, 1e-6) should NOT be treated as deterministic"
            )
