"""FitOrchestrator: encapsulates the full fit() workflow.

Extracted from BaseModel.fit() (formerly ~370 lines, cyclomatic complexity 44)
to reduce BaseModel to a thin delegation layer.

The orchestrator handles:
- RheoData unpacking and test_mode propagation
- Deformation mode conversion (E* -> G*)
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

from rheojax.core.deformation_converter import DeformationModeConverter
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.post_fit_validator import PostFitValidator
from rheojax.core.test_modes import DeformationMode
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

if TYPE_CHECKING:
    type ArrayLike = np.ndarray

logger = get_logger(__name__)

_converter = DeformationModeConverter()
_validator = PostFitValidator()


class FitOrchestrator:
    """Orchestrates the full fit() workflow on behalf of BaseModel."""

    def execute(
        self,
        model: Any,
        X: ArrayLike,
        y: ArrayLike,
        *,
        method: str = "nlsq",
        check_compatibility: bool = False,
        use_log_residuals: bool | None = None,
        use_multi_start: bool | None = None,
        n_starts: int = 5,
        perturb_factor: float = 0.3,
        deformation_mode: str | DeformationMode | None = None,
        poisson_ratio: float = 0.5,
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
        X, y, deformation_mode, kwargs = self._unpack_rheodata(
            X, y, deformation_mode, kwargs
        )

        # --- deformation mode conversion (E* -> G*) ---
        resolved_dm = _converter.resolve_deformation_mode(deformation_mode)
        if resolved_dm is not None:
            model._deformation_mode = resolved_dm
            model._poisson_ratio = poisson_ratio
            y = _converter.convert_to_shear(
                y, resolved_dm, poisson_ratio, model.__class__.__name__
            )
        else:
            # Clear stale tensile mode (R10-BASE-003)
            model._deformation_mode = None

        # --- store data for Bayesian warm-start ---
        model.X_data = X
        model.y_data = y
        self._normalize_stored_data(model)

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
        try:
            model._fit(X, y, method=method, **kwargs)
            model.fitted_ = True
            self._strip_optimization_kwargs(model)
            self._log_fit_completion(model, X, y, data_shape)
        except RuntimeError as e:
            logger.error(
                "Fit failed with RuntimeError",
                model=model.__class__.__name__,
                error=str(e),
                exc_info=True,
            )
            if return_result:
                return model._make_error_result(test_mode, e)
            enhanced = model._enhance_error_with_compatibility(e, X, y, test_mode)
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

        # --- post-fit physics check ---
        if check_physics:
            _validator.check_physics(model)

        # --- post-fit uncertainty quantification ---
        _uncertainty_result = None
        if uncertainty is not None:
            _uncertainty_result = _validator.compute_uncertainty(
                model, X, y, uncertainty, test_mode
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
        deformation_mode: str | DeformationMode | None,
        kwargs: dict,
    ) -> tuple[ArrayLike, ArrayLike, str | DeformationMode | None, dict]:
        """Extract arrays, test_mode, and deformation_mode from RheoData."""
        from rheojax.core.data import RheoData

        if isinstance(X, RheoData):
            _metadata = X.metadata
            if deformation_mode is None:
                deformation_mode = _metadata.get("deformation_mode", None)
            # R10-BASE-001: propagate test_mode from RheoData
            if "test_mode" in _metadata and "test_mode" not in kwargs:
                kwargs["test_mode"] = _metadata["test_mode"]
            if y is None:
                y = X.y
            X = X.x
        return X, y, deformation_mode, kwargs

    @staticmethod
    def _normalize_stored_data(model: Any) -> None:
        """Ensure model.X_data / y_data are raw arrays, not RheoData."""
        from rheojax.core.data import RheoData as _RheoData

        if isinstance(model.X_data, _RheoData):
            model.X_data = model.X_data.x
        if isinstance(model.y_data, _RheoData):
            model.y_data = model.y_data.y

    @staticmethod
    def _run_auto_init(
        model: Any, X: ArrayLike, y: ArrayLike, test_mode: str | None
    ) -> None:
        try:
            from rheojax.utils.initialization.auto_p0 import auto_p0 as _auto_p0

            p0 = _auto_p0(X, y, model, test_mode=test_mode)
            for name, value in p0.items():
                try:
                    model.parameters.set_value(name, value)
                except (KeyError, ValueError):
                    pass
            logger.info(
                "auto_p0 initialized parameters",
                model=model.__class__.__name__,
                n_params_set=len(p0),
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
                from rheojax.core.data import RheoData as _RheoData

                _X_score = X.x if isinstance(X, _RheoData) else X
                _y_score = y

                if (
                    getattr(model, "_nlsq_result", None) is not None
                    and model._nlsq_result.fun is not None
                ):
                    fun = np.asarray(model._nlsq_result.fun)
                    if fun.ndim == 0:
                        ss_res = float(np.abs(fun))
                    else:
                        ss_res = float(np.sum(np.abs(fun) ** 2))
                    y_arr = np.asarray(_y_score)
                    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
                    if ss_tot > 0:
                        r2 = 1.0 - (ss_res / ss_tot)
                    else:
                        r2 = 1.0 if ss_res == 0.0 else None
                else:
                    r2 = model.score(_X_score, _y_score)
            except Exception as exc:
                logger.debug(
                    "R-squared computation failed after fit",
                    model=model.__class__.__name__,
                    error=str(exc),
                )

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
                "FitResult construction failed, returning model",
                model=model.__class__.__name__,
                error=str(exc),
            )
            return model
