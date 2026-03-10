"""Model selection and comparison utilities for RheoJAX.

Provides :func:`compare_models` to fit multiple models to the same rheological
dataset and rank them by an information criterion (AIC, BIC, or AICc).

Example::

    from rheojax.utils.model_selection import compare_models

    comparison = compare_models(
        t, G_data,
        models=["maxwell", "zener", "springpot"],
        test_mode="relaxation",
        criterion="aic",
    )
    print(comparison.summary())

The :func:`build_fit_result` helper is designed to be reusable from
``BaseModel.fit()`` in a future phase so that every fit automatically
produces a structured :class:`~rheojax.core.fit_result.FitResult`.
"""

from __future__ import annotations

import signal
import time
from typing import Any

import numpy as np

from rheojax.core.fit_result import FitResult, ModelComparison
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger

jax, jnp = safe_import_jax()

logger = get_logger(__name__)

__all__ = [
    "build_fit_result",
    "compare_models",
]


# ---------------------------------------------------------------------------
# build_fit_result
# ---------------------------------------------------------------------------


def build_fit_result(
    model: Any,
    X: Any,
    y: Any,
    test_mode: str | None = None,
    model_name: str | None = None,
) -> FitResult:
    """Build a :class:`~rheojax.core.fit_result.FitResult` from a fitted model.

    Reads parameter values and units directly from the model's
    :class:`~rheojax.core.parameters.ParameterSet`, then generates the fitted
    curve via ``model.predict(X)``.  The underlying ``OptimizationResult``
    (stored as ``model._nlsq_result``) is attached when available so that AIC,
    BIC, confidence intervals, and prediction intervals are fully functional.

    This helper is intentionally stateless with respect to the model — it
    reads public attributes only and does not mutate the model instance.

    Args:
        model: A fitted ``BaseModel`` instance.
        X: Input array used for fitting (passed through to ``FitResult.X``).
        y: Target array used for fitting (passed through to ``FitResult.Y``).
        test_mode: Protocol string (e.g. ``"relaxation"``).  When ``None`` the
            function tries to read ``model._test_mode`` as a fallback.
        model_name: Registry name override.  When ``None`` the function probes
            ``ModelRegistry`` by class name to locate the canonical name, then
            falls back to the lower-cased class name.

    Returns:
        Populated :class:`~rheojax.core.fit_result.FitResult`.
    """
    # ------------------------------------------------------------------
    # Resolve model_name
    # ------------------------------------------------------------------
    if model_name is None:
        # Try to find the registry name for this class.
        cls_name = model.__class__.__name__
        try:
            # Scan the live registry for a matching plugin class
            from rheojax.core.registry import Registry

            reg = Registry.get_instance()
            found_name: str | None = None
            for rname, info in reg._models.items():
                if info.plugin_class is model.__class__:
                    found_name = rname
                    break
            model_name = found_name if found_name is not None else cls_name.lower()
        except Exception:
            model_name = model.__class__.__name__.lower()

    model_class_name: str = model.__class__.__name__

    # ------------------------------------------------------------------
    # Resolve test_mode
    # ------------------------------------------------------------------
    if test_mode is None:
        # Try the attribute stored by BaseModel.fit()
        _raw_tm = getattr(model, "_test_mode", None)
        test_mode = str(_raw_tm) if _raw_tm is not None else None

    # ------------------------------------------------------------------
    # Extract parameter values and units
    # ------------------------------------------------------------------
    try:
        param_names: list[str] = list(model.parameters.keys())
    except Exception:
        param_names = []

    params: dict[str, float] = {}
    params_units: dict[str, str] = {}
    for pname in param_names:
        try:
            params[pname] = float(model.parameters.get_value(pname))
        except Exception:
            params[pname] = float("nan")
        try:
            unit = getattr(model.parameters[pname], "units", "") or ""
            params_units[pname] = str(unit)
        except Exception:
            params_units[pname] = ""

    n_params: int = len(param_names)

    # ------------------------------------------------------------------
    # Attached OptimizationResult + Fitted curve
    # (single predict() call to avoid redundant ODE solves)
    # ------------------------------------------------------------------
    opt_result = getattr(model, "_nlsq_result", None)
    fitted_curve: np.ndarray | None = None
    _cached_pred: np.ndarray | None = None

    # Get prediction once — reused for both opt_result and fitted_curve
    if getattr(model, "fitted_", False):
        try:
            pred = model.predict(X)
            _cached_pred = np.asarray(pred)
            fitted_curve = _cached_pred
        except Exception as exc:
            logger.warning(
                "build_fit_result: predict() failed — fitted_curve omitted",
                model=model_class_name,
                error=str(exc),
            )

    # Many models (e.g. classical family) don't store _nlsq_result.
    # Build a minimal OptimizationResult from cached prediction residuals
    # so that AIC/BIC/R² are always available in the FitResult.
    if opt_result is None and _cached_pred is not None and y is not None:
        try:
            from rheojax.utils.optimization import OptimizationResult

            pred_arr = _cached_pred
            y_arr = np.asarray(y)

            if np.iscomplexobj(y_arr):
                residuals = np.concatenate([
                    y_arr.real - pred_arr.real,
                    y_arr.imag - pred_arr.imag,
                ])
                _is_complex = True
            else:
                residuals = y_arr - pred_arr
                _is_complex = False

            rss = float(np.sum(residuals**2))
            opt_result = OptimizationResult(
                x=np.array(list(params.values())),
                fun=rss,
                success=True,
                message="Reconstructed from predict()",
                residuals=residuals,
                y_data=y_arr,
                n_data=len(y_arr),
                _is_complex_split=_is_complex,
            )
        except Exception as exc:
            logger.debug(
                "build_fit_result: could not reconstruct OptimizationResult",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Assemble FitResult
    # ------------------------------------------------------------------
    return FitResult(
        model_name=model_name,
        model_class_name=model_class_name,
        protocol=test_mode,
        params=params,
        params_units=params_units,
        n_params=n_params,
        optimization_result=opt_result,
        fitted_curve=fitted_curve,
        X=np.asarray(X) if X is not None else None,
        y=np.asarray(y) if y is not None else None,
    )


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------


def compare_models(
    X: Any,
    y: Any,
    models: list[str | Any] | None = None,
    test_mode: str | None = None,
    criterion: str = "aic",
    return_results: bool = False,
    per_model_timeout: float | None = 300.0,
    **fit_kwargs: Any,
) -> ModelComparison:
    """Fit multiple models to the same data and rank them by an information criterion.

    Each model is fitted sequentially (parallel vmap is not feasible because
    models have different parameter counts).  Failed fits are logged as warnings
    and skipped rather than raising.

    Args:
        X: Input array (e.g. time, frequency, shear rate).
        y: Target array (e.g. G(t), G*, viscosity).
        models: List of model identifiers.  Each entry is either a registry
            name string (e.g. ``"maxwell"``) or an already-instantiated
            ``BaseModel`` instance.  String entries are created via
            :meth:`~rheojax.core.registry.ModelRegistry.create`.
            When ``None``, auto-discovers all models registered for
            *test_mode* via :meth:`ModelRegistry.find`.
        test_mode: Protocol string forwarded to ``model.fit()``.  Pass ``None``
            to let each model auto-detect the protocol from the data shape.
        criterion: Information criterion used for ranking.  One of ``"aic"``,
            ``"bic"``, or ``"aicc"``.  Defaults to ``"aic"``.
        return_results: Unused — kept for forward compatibility with a future
            API where the function returns ``(ModelComparison, list[FitResult])``.
            Currently the full list of :class:`~rheojax.core.fit_result.FitResult`
            objects is always accessible as ``ModelComparison.results``.
        per_model_timeout: Maximum wall-clock seconds per model fit. ``None``
            disables the timeout.  Defaults to 300 s (5 min).
        **fit_kwargs: Additional keyword arguments forwarded to every
            ``model.fit()`` call (e.g. ``max_iter``, ``workflow``).

    Returns:
        :class:`~rheojax.core.fit_result.ModelComparison` with rankings,
        Δ-criterion values, and Akaike weights.

    Raises:
        ValueError: If ``criterion`` is not one of the supported values, or
            if ``models`` is ``None`` and ``test_mode`` is also ``None``.

    Example::

        comparison = compare_models(
            t, G_data,
            models=["maxwell", "zener", "springpot"],
            test_mode="relaxation",
            criterion="aicc",
            max_iter=2000,
        )
        print(comparison.summary())
        fig = comparison.plot()

        # Auto-discover all models for a protocol:
        comparison = compare_models(t, G_data, test_mode="relaxation")
    """
    _valid_criteria = {"aic", "bic", "aicc"}
    if criterion not in _valid_criteria:
        raise ValueError(
            f"criterion must be one of {sorted(_valid_criteria)!r}, got {criterion!r}"
        )

    # Auto-discover models from registry when not explicitly provided
    if models is None:
        if test_mode is None:
            raise ValueError(
                "test_mode is required when models is None "
                "(needed for auto-discovery via ModelRegistry.find)"
            )
        models = ModelRegistry.find(protocol=test_mode)
        logger.info(
            "compare_models: auto-discovered models from registry",
            protocol=test_mode,
            n_models=len(models),
            models=models,
        )

    fit_results: list[FitResult] = []

    for idx, model_entry in enumerate(models):
        # ------------------------------------------------------------------
        # Resolve model instance
        # ------------------------------------------------------------------
        resolved_name: str | None = None

        if isinstance(model_entry, str):
            resolved_name = model_entry
            logger.debug(
                "compare_models: creating model from registry",
                name=resolved_name,
                index=idx,
            )
            try:
                model = ModelRegistry.create(resolved_name)
            except Exception as exc:
                logger.warning(
                    "compare_models: could not create model — skipping",
                    name=resolved_name,
                    error=str(exc),
                )
                continue
        else:
            model = model_entry
            resolved_name = None  # will be resolved inside build_fit_result

        model_label: str = resolved_name or model.__class__.__name__

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------
        logger.info(
            "compare_models: fitting model",
            model=model_label,
            progress=f"{idx + 1}/{len(models)}",
        )

        fit_kwargs_with_mode: dict[str, Any] = dict(fit_kwargs)
        if test_mode is not None:
            fit_kwargs_with_mode["test_mode"] = test_mode

        t0 = time.monotonic()
        try:
            # Use SIGALRM-based timeout on Unix; wall-clock fallback elsewhere
            if per_model_timeout is not None and hasattr(signal, "SIGALRM"):
                def _timeout_handler(
                    signum: int,
                    frame: Any,
                    _label: str = model_label,
                    _t: float = per_model_timeout,
                ) -> None:
                    raise TimeoutError(
                        f"Model '{_label}' exceeded {_t}s timeout"
                    )
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(max(1, int(per_model_timeout)))
                try:
                    model.fit(X, y, **fit_kwargs_with_mode)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                model.fit(X, y, **fit_kwargs_with_mode)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.warning(
                "compare_models: fit failed — skipping model",
                model=model_label,
                elapsed_s=round(elapsed, 1),
                error=str(exc),
            )
            continue

        # ------------------------------------------------------------------
        # Build FitResult
        # ------------------------------------------------------------------
        try:
            fr = build_fit_result(
                model=model,
                X=X,
                y=y,
                test_mode=test_mode,
                model_name=resolved_name,
            )
        except Exception as exc:
            logger.warning(
                "compare_models: build_fit_result failed — skipping model",
                model=model_label,
                error=str(exc),
            )
            continue

        # Attach the fitted model instance so callers (e.g. Pipeline) can
        # reuse it without re-fitting.
        fr._fitted_model = model

        fit_results.append(fr)
        logger.debug(
            "compare_models: collected result",
            model=fr.model_name,
            criterion=criterion,
            value=getattr(fr, criterion, None),
        )

    if not fit_results:
        logger.warning(
            "compare_models: no models fitted successfully — returning empty comparison"
        )

    return ModelComparison(results=fit_results, criterion=criterion)
