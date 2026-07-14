"""FitOrchestrator: encapsulates the full fit() workflow.

Extracted from BaseModel.fit() (formerly ~370 lines, cyclomatic complexity 44)
to reduce BaseModel to a thin delegation layer.

The orchestrator handles:
- RheoData unpacking and test_mode propagation
- Auto-initialization (auto_p0)
- Optimization strategy detection (log-residuals, multi-start)
- Compatibility checks
- Method remapping (auto_global -> auto + workflow)
- Delegation to model._fit()
- Post-fit cleanup (strip optimization kwargs)
- R-squared computation
- Physics validation
- Uncertainty quantification
- FitResult construction
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.core.post_fit_validator import PostFitValidator
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    type ArrayLike = np.ndarray

logger = get_logger(__name__)

_validator = PostFitValidator()


class FitOrchestrator:
    """Orchestrates the full fit() workflow on behalf of BaseModel."""

    def execute(
        self,
        model: Any,
        X: ArrayLike,
        y: ArrayLike | None,
        *,
        method: str = "nlsq",
        check_compatibility: bool = False,
        use_log_residuals: bool | None = None,
        use_multi_start: bool | None = None,
        n_starts: int = 5,
        perturb_factor: float = 0.3,
        auto_init: bool = False,
        return_result: bool = False,
        check_physics: bool = False,
        uncertainty: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the full fit pipeline.

        All parameters mirror BaseModel.fit() — see its docstring for details.

        Returns:
            ``model`` for method chaining, or ``FitResult`` if return_result=True.
        """
        # --- logging preamble ---
        _shape = getattr(X, "shape", None)
        data_shape = (
            _shape
            if _shape is not None
            else (len(X) if hasattr(X, "__len__") else (1,))
        )
        logger.debug(
            "Entering fit",
            model=model.__class__.__name__,
            data_shape=data_shape,
            method=method,
        )

        # Reset protocol state from any prior fit/fit_bayesian call
        model._last_fit_kwargs = {}

        # --- RheoData unpacking ---
        X, y, kwargs = self._unpack_rheodata(X, y, kwargs)
        if y is None:
            raise ValueError(
                "fit() requires y data: pass y explicitly, or pass a RheoData "
                "for X with a non-None y."
            )

        # --- I/O boundary validation (shape/NaN/monotonicity) ---
        # Reuses RheoData's own validation (shape-match + NaN/finite raise,
        # non-monotonic x warns) so raw-array fit() calls get the same checks
        # a RheoData-wrapped call already received via __post_init__.
        self._validate_fit_data(X, y, kwargs.get("test_mode"))

        # --- fail fast on an invalid uncertainty method ---
        # Must happen before model._fit() runs: a bad method name is a caller
        # error and should never be discovered only after an expensive fit
        # already committed (model.fitted_ = True) further down.
        if uncertainty is not None and uncertainty not in ("hessian", "bootstrap"):
            raise ValueError(
                f"Unknown uncertainty method: {uncertainty!r}. "
                "Expected 'hessian' or 'bootstrap'."
            )

        # --- sort into ascending x order ---
        # Non-monotonic x only warns above; many models (cumulative/ODE
        # integration schemes) assume ascending x and would silently produce
        # a wrong-but-converged fit otherwise. Mirrors the sort
        # RheoData.interpolate() already performs for the same reason.
        X, y = self._sort_by_x(X, y)

        # --- store data for Bayesian warm-start ---
        model.X_data = X
        model.y_data = y

        # --- optimization strategy auto-detection ---
        use_log_residuals, use_multi_start = model._detect_optimization_strategy(
            X, use_log_residuals, use_multi_start, n_starts
        )
        kwargs["use_log_residuals"] = use_log_residuals
        kwargs["use_multi_start"] = use_multi_start
        kwargs["n_starts"] = n_starts
        kwargs["perturb_factor"] = perturb_factor

        # --- auto-initialization ---
        test_mode = kwargs.get("test_mode", None)
        if auto_init:
            self._run_auto_init(model, X, y, test_mode)

        # --- optional compatibility check ---
        if check_compatibility:
            self._run_compatibility_check(model, X, y, test_mode)

        # --- method remapping (auto_global -> auto + workflow) ---
        if method == "auto_global":
            kwargs.setdefault("workflow", "auto_global")
            method = "auto"

        # --- delegate to model._fit() ---
        # NOTE: only model._fit() is wrapped here. Post-fit bookkeeping
        # (fitted_ flag, kwarg stripping, completion logging) runs
        # unconditionally below so a bug in that bookkeeping is never
        # misreported as a fit failure after the optimizer already succeeded.
        try:
            model._fit(X, y, method=method, **kwargs)
        except RuntimeError as e:
            logger.error(
                "Fit failed with RuntimeError",
                model=model.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            enhanced = model._enhance_error_with_compatibility(e, X, y, test_mode)
            if return_result:
                return model._make_error_result(test_mode, enhanced)
            if enhanced is not e:
                raise enhanced from e
            raise
        except Exception as e:
            logger.error(
                "Fit failed with unexpected error",
                model=model.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            if return_result:
                return model._make_error_result(test_mode, e)
            raise

        # --- post-fit bookkeeping (never conflated with _fit() failures) ---
        model.fitted_ = True
        self._strip_optimization_kwargs(model)
        self._log_fit_completion(model, X, y, data_shape)

        # --- post-fit physics check ---
        if check_physics:
            _validator.check_physics(model)

        # --- post-fit uncertainty quantification ---
        _uncertainty_result = None
        if uncertainty is not None:
            _uncertainty_result = _validator.compute_uncertainty(
                model, X, y, uncertainty, test_mode
            )
            # Attach regardless of return_result so the (possibly expensive)
            # computation is never silently discarded.
            model.uncertainty_ = _uncertainty_result
            model.uncertainty_method_ = uncertainty
            if not return_result:
                if _uncertainty_result is not None:
                    logger.info(
                        "Uncertainty computed but return_result=False; "
                        "result stored on model.uncertainty_ (not in a FitResult)",
                        model=model.__class__.__name__,
                        method=uncertainty,
                    )
                else:
                    # compute_uncertainty() already logged a WARNING on
                    # failure; don't also claim success here (model.uncertainty_
                    # is None, not a usable result).
                    logger.info(
                        "Uncertainty computation failed; model.uncertainty_ is None",
                        model=model.__class__.__name__,
                        method=uncertainty,
                    )

        # --- build FitResult if requested ---
        if return_result:
            return self._build_fit_result(
                model, X, y, test_mode, _uncertainty_result, uncertainty
            )

        return model

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unpack_rheodata(
        X: ArrayLike,
        y: ArrayLike | None,
        kwargs: dict,
    ) -> tuple[ArrayLike, ArrayLike | None, dict]:
        """Extract arrays and test_mode from RheoData."""
        from rheojax.core.data import RheoData

        if isinstance(y, RheoData):
            raise ValueError(
                "y must be a plain array; RheoData should be passed as X "
                "(with y=None), not as y."
            )
        if isinstance(X, RheoData):
            _metadata = X.metadata
            # R10-BASE-001: propagate test_mode from RheoData
            if "test_mode" in _metadata and "test_mode" not in kwargs:
                kwargs["test_mode"] = _metadata["test_mode"]
            if y is None:
                if X.y is None:
                    raise ValueError("RheoData.y is None; cannot use as fit target")
                y = np.asarray(X.y)
            if X.x is None:
                raise ValueError("RheoData.x is None; cannot use as fit input")
            X = np.asarray(X.x)
        return X, y, kwargs

    @staticmethod
    def _validate_fit_data(
        X: ArrayLike, y: ArrayLike, test_mode: str | None = None
    ) -> None:
        """Validate shape/NaN/monotonicity on the unpacked fit arrays.

        Constructs a throwaway :class:`RheoData` to reuse its existing
        validation logic (shape mismatch and NaN/non-finite values raise
        ``ValueError``; non-monotonic x only warns). This closes the gap
        where raw-array ``fit(X, y)`` calls previously received none of the
        I/O-boundary checks that RheoData-wrapped calls already get via
        ``RheoData.__post_init__``.

        ``domain`` is resolved the same way ``_standard_nlsq_fit`` resolves
        oscillation mode (explicit ``test_mode`` kwarg, else complex ``y``),
        so the frequency-domain negative-value check in
        ``RheoData._validate_data`` also runs for raw-array oscillation
        fits (``domain`` otherwise defaults to "time" and skips that check).
        """
        from rheojax.core.data import RheoData, _check_dtype

        x_arr = np.asarray(X)
        y_arr = np.asarray(y)

        # Some models accept protocol-encoded X (e.g. stacked (time, strain)
        # rows for startup/LAOS, shape (2, N)) or a transposed complex-modulus
        # y (shape (2, N) real/imag rows) — both fall outside RheoData's
        # plain (x: 1D, y: 1D/complex/(N, 2)) convention. Those models already
        # run their own shape validation with model-specific error messages,
        # so defer to them here instead of raising RheoData's generic one;
        # still run the same dtype/NaN/non-finite checks directly so those
        # guarantees (in particular rejecting bool/object dtypes, which
        # would otherwise silently pass the finite check below) aren't lost.
        if x_arr.ndim > 1 or (y_arr.ndim == 2 and y_arr.shape[1] != 2):
            _check_dtype(x_arr, "X")
            _check_dtype(y_arr, "y", allow_complex=True)
            for name, arr in (("X", x_arr), ("y", y_arr)):
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name} data contains NaN or non-finite values")
            return

        domain = (
            "frequency"
            if test_mode == "oscillation" or np.iscomplexobj(np.asarray(y))
            else "time"
        )
        RheoData(x=X, y=y, domain=domain, validate=True)

    @staticmethod
    def _sort_by_x(X: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Sort X (and y in lockstep) into strictly ascending order.

        No-op if X is already ascending (the common case) or not a plain
        1-D sequence. See RheoData.interpolate() (data.py) for the same
        sort, needed for the same reason: order-sensitive numerics silently
        produce wrong results on unsorted input.
        """
        x_np = np.asarray(X)
        if x_np.ndim != 1 or len(x_np) < 2 or np.all(np.diff(x_np) > 0):
            return X, y
        order = np.argsort(x_np)
        return np.asarray(X)[order], np.asarray(y)[order]

    @staticmethod
    def _run_auto_init(
        model: Any, X: ArrayLike, y: ArrayLike, test_mode: str | None
    ) -> None:
        try:
            from rheojax.utils.initialization.auto_p0 import auto_p0 as _auto_p0

            p0 = _auto_p0(X, y, model, test_mode=test_mode)
            failed_params = []
            for name, value in p0.items():
                try:
                    model.parameters.set_value(name, value)
                except (KeyError, ValueError):
                    failed_params.append(name)
            _log = logger.warning if failed_params else logger.info
            _log(
                "auto_p0 initialized parameters",
                model=model.__class__.__name__,
                n_params_set=len(p0) - len(failed_params),
                n_params_failed=len(failed_params),
                failed_params=failed_params,
            )
        except Exception as exc:
            logger.warning(
                "auto_p0 failed, using default initial values",
                model=model.__class__.__name__,
                error=str(exc),
            )

    @staticmethod
    def _run_compatibility_check(
        model: Any, X: ArrayLike, y: ArrayLike, test_mode: str | None
    ) -> None:
        compatibility = model._check_compatibility(X, y, test_mode)
        if compatibility and not compatibility.get("compatible", True):
            try:
                from rheojax.utils.compatibility import format_compatibility_message

                message = format_compatibility_message(compatibility)
                logger.warning(
                    "Model compatibility check failed",
                    model=model.__class__.__name__,
                    message=message,
                )
            except Exception as exc:
                logger.debug(
                    "Failed to format compatibility message",
                    error=str(exc),
                )

    @staticmethod
    def _strip_optimization_kwargs(model: Any) -> None:
        """Remove optimization-only kwargs so they don't leak to model_function."""
        _opt_keys = (
            "use_log_residuals",
            "use_multi_start",
            "n_starts",
            "perturb_factor",
        )
        _lfk = getattr(model, "_last_fit_kwargs", None)
        if isinstance(_lfk, dict):
            for _ok in _opt_keys:
                _lfk.pop(_ok, None)

    @staticmethod
    def _log_fit_completion(
        model: Any, X: ArrayLike, y: ArrayLike, data_shape: Any
    ) -> None:
        """Compute R-squared (DEBUG only) and log fit completion."""
        r2 = None
        if logger.isEnabledFor(logging.DEBUG):
            try:
                # X is always a plain array here: execute() unpacks any
                # RheoData into a plain array (via _unpack_rheodata) before
                # this is ever called, and this is its only caller.
                _X_score = X
                _y_score = y

                _nlsq_result = getattr(model, "_nlsq_result", None)
                if _nlsq_result is not None:
                    # Delegate to the canonical OptimizationResult.r_squared
                    # property instead of re-deriving ss_res/ss_tot here: it
                    # already handles the use_log_residuals dimensional
                    # consistency correction, complex-data component split,
                    # and normalization-weight un-normalization.
                    r2 = getattr(_nlsq_result, "r_squared", None)
                else:
                    r2 = model.score(_X_score, _y_score)
            except Exception as exc:
                logger.debug(
                    "R-squared computation failed after fit",
                    model=model.__class__.__name__,
                    error=str(exc),
                )

        try:
            logger.info(
                "Fit completed",
                model=model.__class__.__name__,
                fitted=model.fitted_,
                R2=r2,
                data_shape=data_shape,
            )
            logger.debug(
                "Exiting fit",
                model=model.__class__.__name__,
                parameters=model.get_params(),
            )
        except Exception as exc:
            # A fit that already succeeded must never be reported as failed
            # just because completion logging/bookkeeping raised (e.g. a
            # future get_params() override). See R2 computation above for
            # the same catch-and-log-warning convention.
            logger.warning(
                "Post-fit completion logging failed",
                model=model.__class__.__name__,
                error=str(exc),
            )

    @staticmethod
    def _build_fit_result(
        model: Any,
        X: ArrayLike,
        y: ArrayLike,
        test_mode: str | None,
        uncertainty_result: dict | None,
        uncertainty_method: str | None,
    ) -> Any:
        """Construct a FitResult, attaching uncertainty if available."""
        try:
            from rheojax.utils.model_selection import build_fit_result

            fit_result = build_fit_result(model, X, y, test_mode=test_mode)
            if uncertainty_result is not None:
                fit_result.metadata["uncertainty"] = uncertainty_result
                fit_result.metadata["uncertainty_method"] = uncertainty_method
            return fit_result
        except Exception as exc:
            logger.warning(
                "FitResult construction failed, returning error FitResult",
                model=model.__class__.__name__,
                error=str(exc),
            )
            # Preserve the return_result=True -> FitResult contract even when
            # FitResult construction itself fails, mirroring how _fit()
            # failures are already handled via _make_error_result().
            return model._make_error_result(test_mode, exc)
