"""Tests for FitOrchestrator: the shared execute() pipeline behind fit().

Covers regressions found in a targeted audit of fit_orchestrator.py:
- uncertainty silently discarded when return_result=False
- _build_fit_result silently downgrading its return type on internal failure
- post-fit bookkeeping misreported as fit failure
- RheoData passed as y (argument-order misuse)
- missing shape/NaN validation on raw-array fit() calls
- DEBUG R2 log reimplementing ss_res/ss_tot instead of delegating to
  OptimizationResult.r_squared
- "Uncertainty computed" info log firing even when compute_uncertainty()
  silently failed and returned None
- missing frequency-domain negative-value check for raw-array oscillation fits
- auto_p0 partial-failure misreported as full success
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.core.fit_orchestrator import FitOrchestrator
from rheojax.core.fit_result import FitResult
from rheojax.core.post_fit_validator import PostFitValidator
from rheojax.models.classical.maxwell import Maxwell


def _relaxation_data(n=30):
    t = np.logspace(-2, 1, n)
    G0, tau = 1e5, 1.0
    G = G0 * np.exp(-t / tau)
    return t, G


@pytest.mark.smoke
class TestUncertaintyAttachment:
    """Bug 1: uncertainty result must not be discarded when return_result=False."""

    def test_uncertainty_attached_without_return_result(self):
        model = Maxwell()
        t, G = _relaxation_data()
        model.fit(t, G, uncertainty="hessian")  # return_result defaults to False

        assert getattr(model, "uncertainty_", None) is not None
        assert model.uncertainty_method_ == "hessian"

    def test_uncertainty_still_in_fit_result_metadata(self):
        model = Maxwell()
        t, G = _relaxation_data()
        result = model.fit(t, G, uncertainty="hessian", return_result=True)

        assert isinstance(result, FitResult)
        assert "uncertainty" in result.metadata
        # Both paths should agree
        assert model.uncertainty_ is result.metadata["uncertainty"]


@pytest.mark.smoke
class TestBuildFitResultContract:
    """Bug 2: return_result=True must always yield a FitResult, even on
    internal build_fit_result() failure."""

    def test_build_fit_result_failure_yields_error_fit_result(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with patch(
            "rheojax.utils.model_selection.build_fit_result",
            side_effect=RuntimeError("boom"),
        ):
            result = FitOrchestrator()._build_fit_result(
                model, t, G, "relaxation", None, None
            )

        assert isinstance(result, FitResult)
        assert result.metadata["error"] == "boom"
        assert result.metadata["error_type"] == "RuntimeError"


@pytest.mark.smoke
class TestPostFitBookkeepingIsolation:
    """Bug 3: a post-fit logging/bookkeeping failure must not be reported as
    a fit failure, and fitted_ must remain True."""

    def test_log_fit_completion_failure_does_not_undo_fit(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with patch.object(
            Maxwell, "get_params", side_effect=RuntimeError("logging exploded")
        ):
            # get_params() is called inside _log_fit_completion's debug-log
            # tail. It must NOT be reported as a fit failure.
            result = model.fit(t, G)

        assert result is model
        assert model.fitted_ is True

    def test_get_params_failure_inside_log_completion_is_swallowed(self):
        model = Maxwell()
        t, G = _relaxation_data()
        model.fit(t, G)  # real fit succeeds first

        with patch.object(
            Maxwell, "get_params", side_effect=RuntimeError("get_params broke")
        ):
            # Directly exercise the now-guarded tail of _log_fit_completion.
            FitOrchestrator._log_fit_completion(model, t, G, t.shape)
        # No exception propagated, model still reports fitted.
        assert model.fitted_ is True


@pytest.mark.smoke
class TestRheoDataAsYMisuse:
    """Bug 4: passing a RheoData as y (argument-order confusion) must raise
    a clear error, not flow unconverted into _fit()."""

    def test_rheodata_as_y_raises_value_error(self):
        model = Maxwell()
        t, G = _relaxation_data()
        rheo_y = RheoData(x=t, y=G, initial_test_mode="relaxation")

        with pytest.raises(ValueError, match="y must be a plain array"):
            model.fit(t, rheo_y)


@pytest.mark.smoke
class TestRawArrayValidation:
    """Bug 6: raw-array fit() calls must get the same shape/NaN checks that
    RheoData-wrapped calls already receive."""

    def test_nan_in_y_raises(self):
        model = Maxwell()
        t, _ = _relaxation_data()
        y = np.full_like(t, np.nan)
        with pytest.raises(ValueError, match="NaN"):
            model.fit(t, y)

    def test_shape_mismatch_raises(self):
        model = Maxwell()
        t = np.linspace(0.1, 10, 10)
        y = np.linspace(1, 2, 5)
        with pytest.raises(ValueError, match="same first dimension"):
            model.fit(t, y)


@pytest.mark.smoke
class TestLogFitCompletionR2:
    """DEBUG R2 log must delegate to OptimizationResult.r_squared instead of
    reimplementing ss_res/ss_tot (which breaks use_log_residuals dimensional
    consistency)."""

    def test_delegates_to_nlsq_result_r_squared(self):
        model = Maxwell()
        t, G = _relaxation_data()
        model.fit(t, G)
        # Stand in for the real OptimizationResult with a duck-typed object
        # exposing only `.r_squared` (no `.fun`) -- if _log_fit_completion
        # still hand-rolled ss_res from `.fun`, this would raise internally
        # and fall back to R2=None instead of using this sentinel.
        model._nlsq_result = SimpleNamespace(r_squared=0.4242)

        with patch("rheojax.core.fit_orchestrator.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = True
            FitOrchestrator._log_fit_completion(model, t, G, t.shape)

        info_kwargs = mock_logger.info.call_args_list[0].kwargs
        assert info_kwargs["R2"] == 0.4242


@pytest.mark.smoke
class TestUncertaintyFailureLogging:
    """The 'Uncertainty computed' info log must not fire when
    compute_uncertainty() silently failed and returned None."""

    def test_no_success_log_when_uncertainty_computation_fails(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with patch.object(PostFitValidator, "compute_uncertainty", return_value=None):
            with patch("rheojax.core.fit_orchestrator.logger") as mock_logger:
                mock_logger.isEnabledFor.return_value = True
                model.fit(t, G, uncertainty="hessian")

        assert model.uncertainty_ is None
        info_messages = [c.args[0] for c in mock_logger.info.call_args_list]
        assert not any("Uncertainty computed" in m for m in info_messages)
        assert any("Uncertainty computation failed" in m for m in info_messages)


@pytest.mark.smoke
class TestFrequencyDomainValidation:
    """Raw-array fit() validation must detect frequency domain (oscillation
    test_mode, or complex y) so RheoData's negative-frequency-value warning
    actually fires, matching RheoData-wrapped calls."""

    def _oscillation_data_with_negative_value(self):
        omega = np.logspace(-2, 2, 10)
        g_star = np.linspace(1, 10, 10) * (1 + 1j)
        g_star[0] = -1 + 1j  # negative real part
        return omega, g_star

    def test_oscillation_test_mode_warns_on_negative_value(self):
        omega, g_star = self._oscillation_data_with_negative_value()
        with pytest.warns(UserWarning, match="negative values in frequency domain"):
            FitOrchestrator._validate_fit_data(omega, g_star, test_mode="oscillation")

    def test_complex_y_without_test_mode_warns_on_negative_value(self):
        omega, g_star = self._oscillation_data_with_negative_value()
        with pytest.warns(UserWarning, match="negative values in frequency domain"):
            FitOrchestrator._validate_fit_data(omega, g_star)


@pytest.mark.smoke
class TestNonStandardShapeConventions:
    """Regression: this generic I/O-boundary gate wraps X/y in a 1-D-x-axis
    RheoData, which is stricter than some models' own documented input
    conventions -- it used to reject them before the model's own (already
    correct) handling ever ran. Both conventions must pass through."""

    def test_ikh_packed_time_strain_x_is_not_rejected(self):
        """IKH/FIKH pack X as (2, N) [time, strain] for startup/LAOS (see
        ikh/_base.py::_extract_time_strain) -- not a 1-D x-axis."""
        t = np.linspace(0, 5.0, 20)
        strain = 2.0 * t
        X = np.stack([t, strain])
        y = np.linspace(1.0, 5.0, 20)
        FitOrchestrator._validate_fit_data(X, y, test_mode="startup")

    def test_ikh_packed_time_strain_length_mismatch_still_raises(self):
        X = np.stack([np.linspace(0, 5.0, 20), np.linspace(0, 10.0, 20)])
        y = np.linspace(1.0, 5.0, 19)  # wrong length
        with pytest.raises(ValueError, match="X and y must have the same length"):
            FitOrchestrator._validate_fit_data(X, y, test_mode="startup")

    def test_ikh_packed_time_strain_nan_still_raises(self):
        X = np.stack([np.linspace(0, 5.0, 20), np.linspace(0, 10.0, 20)])
        y = np.linspace(1.0, 5.0, 20)
        y[0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            FitOrchestrator._validate_fit_data(X, y, test_mode="startup")

    def test_transposed_g_star_2xm_is_not_rejected(self):
        """Some oscillation models (e.g. SGRConventional) accept a
        transposed (2, M) G'/G'' array and auto-transpose it to (M, 2)
        internally; the gate must transpose it too before delegating."""
        omega = np.logspace(-2, 2, 10)
        g_prime = np.linspace(1e3, 1e4, 10)
        g_double_prime = np.linspace(1e2, 1e3, 10)
        G_2xM = np.vstack([g_prime, g_double_prime])
        FitOrchestrator._validate_fit_data(omega, G_2xM, test_mode="oscillation")

    def test_genuinely_bad_shape_still_raises(self):
        omega = np.logspace(-2, 2, 10)
        bad = np.ones((10, 3))
        with pytest.raises(ValueError, match=r"2-dimensional with shape \(N, 2\)"):
            FitOrchestrator._validate_fit_data(omega, bad, test_mode="oscillation")


@pytest.mark.smoke
class TestAutoInitFailureReporting:
    """auto_p0 partial failures must be tracked and reported, not
    misrepresented as full success via n_params_set=len(p0)."""

    def test_partial_failure_tracked_not_reported_as_full_success(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with patch(
            "rheojax.utils.initialization.auto_p0.auto_p0",
            return_value={"G0": 1e5, "not_a_real_param": 999.0},
        ):
            with patch("rheojax.core.fit_orchestrator.logger") as mock_logger:
                FitOrchestrator._run_auto_init(model, t, G, None)

        # Bug 5: a partial failure must be visible at WARNING (not just
        # INFO, which is routinely filtered out in production), so it is
        # never logged at INFO-only-success level.
        assert mock_logger.info.call_count == 0
        warning_kwargs = mock_logger.warning.call_args_list[0].kwargs
        assert warning_kwargs["n_params_set"] == 1
        assert warning_kwargs["n_params_failed"] == 1
        assert warning_kwargs["failed_params"] == ["not_a_real_param"]

    def test_full_success_still_logs_at_info(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with patch(
            "rheojax.utils.initialization.auto_p0.auto_p0",
            return_value={"G0": 1e5},
        ):
            with patch("rheojax.core.fit_orchestrator.logger") as mock_logger:
                FitOrchestrator._run_auto_init(model, t, G, None)

        assert mock_logger.warning.call_count == 0
        info_kwargs = mock_logger.info.call_args_list[0].kwargs
        assert info_kwargs["n_params_failed"] == 0


@pytest.mark.smoke
class TestInvalidUncertaintyMethod:
    """Bug: an invalid uncertainty method must raise before the fit commits,
    not after model.fitted_ is already True."""

    def test_invalid_uncertainty_raises_before_fit_commits(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            model.fit(t, G, uncertainty="monte_carlo", return_result=True)

        # The fit must never have run/committed.
        assert model.fitted_ is False

    def test_invalid_uncertainty_raises_regardless_of_return_result(self):
        model = Maxwell()
        t, G = _relaxation_data()

        with pytest.raises(ValueError, match="Unknown uncertainty method"):
            model.fit(t, G, uncertainty="monte_carlo", return_result=False)


@pytest.mark.smoke
class TestRuntimeErrorEnhancementWithReturnResult:
    """Bug: return_result=True must get the same compatibility-enhanced
    error message as the return_result=False (raising) path."""

    def test_enhancement_applied_when_return_result_true(self):
        model = Maxwell()
        t, G = _relaxation_data()

        from rheojax.utils.compatibility import DecayType, MaterialType

        compat = {
            "compatible": False,
            "confidence": 0.9,
            "decay_type": DecayType.UNKNOWN,
            "material_type": MaterialType.UNKNOWN,
            "warnings": ["mismatch"],
            "recommendations": ["try another model"],
        }

        with patch.object(
            Maxwell,
            "_fit",
            side_effect=RuntimeError("Optimization failed: did not converge"),
        ):
            with patch.object(Maxwell, "_check_compatibility", return_value=compat):
                result = model.fit(t, G, return_result=True)

        assert isinstance(result, FitResult)
        assert "Model-data compatibility issue detected" in result.metadata["error"]


@pytest.mark.smoke
class TestNonMonotonicXSorted:
    """Bug: non-monotonic x must be sorted (not just warned about) before
    reaching model._fit(), since many models assume ascending x."""

    def test_unsorted_x_is_sorted_before_fit(self):
        t, G = _relaxation_data()
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(t))
        t_shuffled, G_shuffled = t[perm], G[perm]

        model = Maxwell()
        with pytest.warns(UserWarning, match="not monotonic"):
            model.fit(t_shuffled, G_shuffled)

        assert np.all(np.diff(model.X_data) > 0)
        # y must have been permuted in lockstep with x, not left shuffled.
        order = np.argsort(t_shuffled)
        np.testing.assert_allclose(model.X_data, t_shuffled[order])
        np.testing.assert_allclose(model.y_data, G_shuffled[order])

    def test_already_sorted_x_is_a_no_op(self):
        t, G = _relaxation_data()
        X_out, y_out = FitOrchestrator._sort_by_x(t, G)
        assert X_out is t
        assert y_out is G
