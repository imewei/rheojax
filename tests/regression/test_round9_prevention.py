"""Regression tests for Round 9 RCA (2026-02-25) prevention.

Guards against the 5 regression patterns introduced by previous AI agents:
  a. Always-true conditionals (if x or not x:)
  b. Removed dispatch calls without verifying replacement path
  c. Strain arrays used as time arrays without checking return semantics
  d. finally cleanup without success guards
  e. Overly broad exception swallowing

Plus systemic gap:
  f. Pipeline never forwarding deformation_mode to models (DMTA)
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from rheojax.core.bayesian import BayesianMixin
from rheojax.core.data import RheoData
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

    @pytest.mark.smoke
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

    @pytest.mark.smoke
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


# ═══════════════════════════════════════════════════════════════════════════
# Category 5: DMTA Pipeline deformation_mode propagation
#   Systemic gap: Pipeline never forwarded deformation_mode
# ═══════════════════════════════════════════════════════════════════════════


class TestPipelineDMTAPropagation:
    """Guard: Pipeline must forward deformation_mode from data.metadata."""

    @pytest.fixture
    def dmta_data(self):
        """RheoData with DMTA metadata (deformation_mode + poisson_ratio)."""
        omega = np.logspace(-1, 2, 30)
        E_star = 3e9 * omega**2 / (1 + omega**2) + 1j * 3e9 * omega / (1 + omega**2)
        return RheoData(
            x=omega,
            y=np.abs(E_star),
            x_units="rad/s",
            y_units="Pa",
            domain="frequency",
            metadata={
                "test_mode": "oscillation",
                "deformation_mode": "tension",
                "poisson_ratio": 0.5,
            },
            validate=False,
        )

    @pytest.mark.smoke
    def test_pipeline_fit_propagates_deformation_mode(self, dmta_data):
        """Pipeline.fit() must pass deformation_mode from data metadata to model.fit()."""
        from rheojax.core.base import BaseModel
        from rheojax.core.registry import ModelRegistry
        from rheojax.core.test_modes import DeformationMode
        from rheojax.pipeline import Pipeline

        class _DmtaTracer(BaseModel):
            """Model that records kwargs it receives in fit()."""

            def __init__(self):
                super().__init__()
                self.parameters.add("a", value=1.0, bounds=(0.01, 1e12))

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return np.ones_like(X)

        ModelRegistry.register("_dmta_tracer")(_DmtaTracer)
        try:
            p = Pipeline()
            p.data = dmta_data
            p.fit("_dmta_tracer")
            model = p._last_model
            # deformation_mode is consumed at BaseModel.fit() boundary and stored as model attr
            assert hasattr(model, "_deformation_mode"), (
                "Pipeline.fit() did not propagate deformation_mode — model has no _deformation_mode"
            )
            assert model._deformation_mode == DeformationMode.TENSION, (
                f"Expected TENSION, got {model._deformation_mode}"
            )
            assert model._poisson_ratio == 0.5, (
                "Pipeline.fit() did not propagate poisson_ratio from data.metadata"
            )
        finally:
            ModelRegistry.unregister("_dmta_tracer")

    @pytest.mark.smoke
    def test_pipeline_fit_does_not_override_explicit_deformation(self, dmta_data):
        """Explicit deformation_mode kwarg must take precedence over metadata."""
        from rheojax.core.base import BaseModel
        from rheojax.core.registry import ModelRegistry
        from rheojax.core.test_modes import DeformationMode
        from rheojax.pipeline import Pipeline

        class _DmtaTracer2(BaseModel):
            def __init__(self):
                super().__init__()
                self.parameters.add("a", value=1.0, bounds=(0.01, 1e12))

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return np.ones_like(X)

        ModelRegistry.register("_dmta_tracer2")(_DmtaTracer2)
        try:
            p = Pipeline()
            p.data = dmta_data
            p.fit("_dmta_tracer2", deformation_mode="bending")
            model = p._last_model
            assert model._deformation_mode == DeformationMode.BENDING, (
                "Explicit deformation_mode kwarg was overridden by metadata"
            )
        finally:
            ModelRegistry.unregister("_dmta_tracer2")

    @pytest.mark.smoke
    def test_bayesian_pipeline_nlsq_propagates_deformation(self, dmta_data):
        """BayesianPipeline.fit_nlsq() must propagate deformation_mode."""
        from rheojax.core.base import BaseModel
        from rheojax.core.registry import ModelRegistry
        from rheojax.core.test_modes import DeformationMode
        from rheojax.pipeline.bayesian import BayesianPipeline

        class _DmtaTracer3(BaseModel):
            def __init__(self):
                super().__init__()
                self.parameters.add("a", value=1.0, bounds=(0.01, 1e12))

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return np.ones_like(X)

        ModelRegistry.register("_dmta_tracer3")(_DmtaTracer3)
        try:
            bp = BayesianPipeline()
            bp.data = dmta_data
            bp.fit_nlsq("_dmta_tracer3")
            model = bp._last_model
            assert hasattr(model, "_deformation_mode"), (
                "BayesianPipeline.fit_nlsq() did not propagate deformation_mode"
            )
            assert model._deformation_mode == DeformationMode.TENSION, (
                f"Expected TENSION, got {model._deformation_mode}"
            )
        finally:
            ModelRegistry.unregister("_dmta_tracer3")


# ═══════════════════════════════════════════════════════════════════════════
# Category 6: Registry discover() only swallows "already registered"
#   Regression pattern (e): overly broad exception swallowing
# ═══════════════════════════════════════════════════════════════════════════


class TestRegistryDiscoverExceptionScope:
    """Guard: discover() must not swallow non-registration errors."""

    @pytest.mark.smoke
    def test_discover_propagates_real_valueerror(self):
        """ValueError not matching 'already registered' must propagate."""
        from rheojax.core.registry import Registry

        registry = Registry.get_instance()

        def _register_that_raises(*args, **kwargs):
            raise ValueError("Corrupt model definition")

        with patch.object(registry, "register", side_effect=_register_that_raises):
            # discover() should let non-"already registered" ValueError propagate
            with pytest.raises(ValueError, match="Corrupt model definition"):
                registry.discover("rheojax.models.classical")

    @pytest.mark.smoke
    def test_discover_swallows_already_registered(self):
        """ValueError with 'already registered' must be silently caught."""
        from rheojax.core.registry import Registry

        registry = Registry.get_instance()

        def _register_already(*args, **kwargs):
            raise ValueError("Model 'Maxwell' is already registered")

        with patch.object(registry, "register", side_effect=_register_already):
            # Should NOT raise — "already registered" is expected during discovery
            registry.discover("rheojax.models.classical")
