"""Uncertainty quantification for RheoJAX model fits.

This module provides post-fit uncertainty estimation using two methods:

1. **Hessian-based CI** (fast, approximate): Uses the Cramér-Rao bound.
   At the MLE, the inverse Hessian of the negative log-likelihood approximates
   the parameter covariance matrix. When NLSQ already computed a Jacobian,
   the Jacobian-based pcov from scipy is reused directly (no extra computation).

2. **Bootstrap CI** (robust, computationally intensive): Residual bootstrap —
   resamples residuals from the converged fit, generates synthetic datasets,
   refits the model to each, and reports percentile intervals over the resulting
   parameter distribution. Works for any model including those with ODE internals.

Both functions handle complex oscillation data (G* = G' + iG'') transparently:
bootstrap resampling uses the same indices for real and imaginary parts so that
the G'(ω)/G''(ω) pairing is never broken.

Example::

    from rheojax.models import Maxwell
    from rheojax.utils.uncertainty import hessian_ci, bootstrap_ci

    model = Maxwell()
    model.fit(t, G_relax)

    ci_fast   = hessian_ci(model, t, G_relax, alpha=0.05)
    ci_robust = bootstrap_ci(model, t, G_relax, n_bootstrap=500, alpha=0.05)

    for name, (lo, hi) in ci_fast.items():
        print(f"{name}: [{lo:.4g}, {hi:.4g}]")
"""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import — MUST come after NLSQ to preserve float64
jax, jnp = safe_import_jax()

logger = get_logger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from rheojax.core.base import BaseModel

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_fitted(model: BaseModel) -> None:
    """Raise RuntimeError if *model* has not been fitted yet."""
    if not getattr(model, "fitted_", False):
        raise RuntimeError(
            f"Model {model.__class__.__name__} has not been fitted. "
            "Call model.fit() before computing confidence intervals."
        )


def _param_names(model: BaseModel) -> list[str]:
    """Return ordered list of parameter names from model.parameters."""
    return list(model.parameters.keys())


def _param_values(model: BaseModel) -> np.ndarray:
    """Return current optimal parameter values as a float64 numpy array."""
    return np.asarray(model.parameters.get_values(), dtype=np.float64)


def _n_obs(y: np.ndarray) -> int:
    """Effective number of observations for AICc / degrees-of-freedom.

    Complex arrays contribute 2N real data points (G' and G'' fitted
    independently), matching the residual vector length used by NLSQ.
    """
    if np.iscomplexobj(y):
        return 2 * int(y.shape[0])
    return int(y.ravel().shape[0])


def _t_critical(alpha: float, n: int, k: int) -> float:
    """Two-sided t-critical value at significance *alpha*.

    Uses scipy.stats.t.ppf(1 - alpha/2, dof) where dof = max(n - k, 1).
    Falls back to the normal quantile (scipy-free environments) when scipy
    is unavailable.
    """
    dof = max(n - k, 1)
    try:
        from scipy.stats import t as t_dist

        return float(t_dist.ppf(1.0 - alpha / 2.0, dof))
    except ImportError:  # pragma: no cover
        # Normal approximation — valid as n → ∞
        from math import erfinv, sqrt

        return float(sqrt(2.0) * erfinv(1.0 - alpha))


def _predict_safe(model: BaseModel, X: np.ndarray) -> np.ndarray:
    """Call model.predict(X), stripping keyword arguments unsupported by older
    _predict() signatures.  Returns a numpy array."""
    y_pred = model.predict(X)
    return np.asarray(y_pred)


def _residuals_to_real(residuals: np.ndarray) -> np.ndarray:
    """Convert residuals to a single real float64 array.

    Complex residuals are split into [real_part | imag_part] (length 2N),
    which matches the convention used by OptimizationResult / scipy's TRF.
    Real (N, 2) arrays are ravelled; real (N,) arrays pass through unchanged.
    """
    if np.iscomplexobj(residuals):
        return np.concatenate(
            [np.real(residuals).ravel(), np.imag(residuals).ravel()]
        ).astype(np.float64)
    return np.asarray(residuals, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def hessian_ci(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
    test_mode: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute confidence intervals via Hessian inversion (Cramér-Rao bound).

    Strategy (in order of preference):

    1. **Reuse NLSQ pcov** — When `model._nlsq_result.pcov` is available
       (computed by scipy's TRF Jacobian), it is used directly. This is the
       cheapest path and is consistent with the optimizer's own uncertainty
       estimate.

    2. **JAX Hessian** — When pcov is not available, constructs a loss
       ``L(θ) = 0.5 ‖f(X; θ) - y‖²`` and evaluates `jax.hessian(L)` at the
       optimal parameters. The inverse Hessian approximates the covariance
       under the Cramér-Rao bound.

    Standard errors are ``se_i = sqrt(pcov[i, i])`` and the CI half-width is
    ``t_{α/2, n-k} · se_i`` where *n* is the number of observations and *k*
    the number of free parameters.

    Parameters
    ----------
    model:
        A fitted RheoJAX model (BaseModel subclass). Must have `fitted_=True`.
    X:
        Independent variable array used for fitting (same shape as at fit time).
    y:
        Dependent variable array (may be complex for oscillation data).
    alpha:
        Significance level.  0.05 produces 95 % confidence intervals.
    test_mode:
        Passed to the model's prediction when recomputing residuals (required
        for Hessian path). If None, the stored `_test_mode` attribute on the
        model is used when available.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping ``parameter_name → (lower_bound, upper_bound)`` at the
        ``(1 - alpha)`` confidence level.

    Raises
    ------
    RuntimeError
        If the model has not been fitted.
    ValueError
        If neither pcov nor a valid Hessian can be obtained.

    Examples
    --------
    >>> ci = hessian_ci(model, omega, G_star, alpha=0.05)
    >>> for name, (lo, hi) in ci.items():
    ...     print(f"{name}: [{lo:.4g}, {hi:.4g}]")
    """
    _assert_fitted(model)

    param_names = _param_names(model)
    param_vals = _param_values(model)
    k = len(param_names)
    n = _n_obs(np.asarray(y))

    logger.debug(
        "hessian_ci called",
        model=model.__class__.__name__,
        n_params=k,
        n_obs=n,
        alpha=alpha,
    )

    pcov: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Path 1: reuse existing pcov from NLSQ result (preferred)
    # ------------------------------------------------------------------
    nlsq_result = getattr(model, "_nlsq_result", None)
    if nlsq_result is not None and getattr(nlsq_result, "pcov", None) is not None:
        candidate = np.asarray(nlsq_result.pcov, dtype=np.float64)
        if candidate.shape == (k, k) and np.all(np.isfinite(candidate)):
            pcov = candidate
            logger.debug("Using existing NLSQ pcov for confidence intervals")

    # ------------------------------------------------------------------
    # Path 2: JAX Hessian at optimal parameters
    # ------------------------------------------------------------------
    if pcov is None:
        logger.debug("NLSQ pcov not available — computing JAX Hessian")
        resolved_test_mode = test_mode or getattr(model, "_test_mode", None)

        y_arr = np.asarray(y)
        is_complex = np.iscomplexobj(y_arr)

        # Only models with model_function(X, params, test_mode) expose a
        # JAX-differentiable loss.  For models without model_function, the
        # Hessian path is not viable — skip directly to Jacobian fallback.
        model_fn = getattr(model, "model_function", None)
        if model_fn is None:
            logger.debug(
                "Model has no model_function — skipping Hessian, "
                "using Jacobian fallback directly"
            )
            pcov = _jacobian_fallback_pcov(model, X, y, param_vals, n)
        else:
            _resolved_tm = resolved_test_mode

            def loss_fn(params_arr: jnp.ndarray) -> jnp.ndarray:
                y_pred_jnp = model_fn(X, params_arr, _resolved_tm)
                y_true_jnp = jnp.asarray(y_arr)
                if is_complex:
                    r_real = jnp.real(y_pred_jnp) - jnp.real(y_true_jnp)
                    r_imag = jnp.imag(y_pred_jnp) - jnp.imag(y_true_jnp)
                    return 0.5 * (jnp.sum(r_real**2) + jnp.sum(r_imag**2))
                return 0.5 * jnp.sum(
                    (jnp.asarray(y_pred_jnp) - y_true_jnp) ** 2
                )

            try:
                params_jax = jnp.asarray(param_vals)
                H = jax.hessian(loss_fn)(params_jax)
                H_np = np.asarray(H, dtype=np.float64)

                if not np.all(np.isfinite(H_np)):
                    raise ValueError("Hessian contains non-finite values")

                # Regularise near-singular Hessians with a small diagonal ridge
                ridge = 1e-10 * np.mean(np.abs(np.diag(H_np)))
                H_reg = H_np + ridge * np.eye(k)

                try:
                    pcov = np.linalg.inv(H_reg)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(f"Hessian inversion failed: {exc}") from exc

                if not np.all(np.isfinite(pcov)):
                    raise ValueError(
                        "Inverted Hessian contains non-finite values — "
                        "the loss surface may be flat or unbounded near the optimum."
                    )

                logger.debug(
                    "JAX Hessian computed and inverted",
                    condition_number=float(
                        np.linalg.cond(H_reg) if np.all(np.isfinite(H_reg)) else np.inf
                    ),
                )

            except Exception as exc:
                # --------------------------------------------------------------
                # Path 3: Jacobian fallback (scipy finite-difference Jacobian)
                # --------------------------------------------------------------
                logger.warning(
                    "JAX Hessian computation failed — attempting Jacobian fallback",
                    error=str(exc),
                )
                pcov = _jacobian_fallback_pcov(model, X, y, param_vals, n)

    if pcov is None:
        raise ValueError(
            "Could not obtain a parameter covariance matrix for "
            f"{model.__class__.__name__}. "
            "Neither NLSQ pcov, JAX Hessian, nor Jacobian fallback succeeded. "
            "Try fitting with method='scipy' to enable Jacobian-based pcov."
        )

    # ------------------------------------------------------------------
    # Build confidence intervals from pcov diagonal
    # ------------------------------------------------------------------
    t_crit = _t_critical(alpha, n, k)
    se = np.sqrt(np.maximum(np.diag(pcov), 0.0))

    ci: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(param_names):
        centre = float(param_vals[i])
        half_width = float(t_crit * se[i])
        ci[name] = (centre - half_width, centre + half_width)

    logger.debug(
        "hessian_ci completed",
        t_critical=t_crit,
        se_range=(float(se.min()), float(se.max())),
    )
    return ci


def _jacobian_fallback_pcov(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    param_vals: np.ndarray,
    n: int,
) -> np.ndarray | None:
    """Compute pcov via finite-difference Jacobian using scipy.

    This is a last-resort fallback when neither the cached NLSQ pcov nor the
    JAX Hessian are available.  It uses scipy.optimize.approx_fprime to build
    an approximate Jacobian of the residual vector with respect to the
    parameters, then delegates to the existing SVD-based covariance helper.

    Parameters
    ----------
    model:
        Fitted model instance.
    X:
        Independent variable array.
    y:
        Dependent variable (possibly complex).
    param_vals:
        Current optimal parameter values.
    n:
        Number of observations (complex counts as N, not 2N).

    Returns
    -------
    np.ndarray or None
        Covariance matrix (k×k) or None if computation fails.
    """
    try:
        from scipy.optimize import approx_fprime

        from rheojax.utils.optimization import compute_covariance_from_jacobian

        y_arr = np.asarray(y)

        def residual_vec(params: np.ndarray) -> np.ndarray:
            """Real residual vector for Jacobian estimation."""
            # Temporarily store params in model parameters and predict
            orig_vals = _param_values(model)
            for i, name in enumerate(_param_names(model)):
                model.parameters.set_value(name, float(params[i]))
            try:
                y_pred = _predict_safe(model, X)
                residuals = y_pred - y_arr
                return _residuals_to_real(residuals)
            finally:
                # Always restore original values
                for i, name in enumerate(_param_names(model)):
                    model.parameters.set_value(name, float(orig_vals[i]))

        eps = np.sqrt(np.finfo(np.float64).eps)
        # approx_fprime returns shape (k,) for a scalar — use Jacobian row by row
        k = len(param_vals)
        base_res = residual_vec(param_vals)
        m = len(base_res)
        jac = np.zeros((m, k), dtype=np.float64)
        for j in range(k):
            eps_vec = np.zeros(k)
            eps_vec[j] = eps * max(abs(param_vals[j]), 1.0)
            jac[:, j] = approx_fprime(param_vals, residual_vec, eps_vec)

        pcov = compute_covariance_from_jacobian(jac, base_res, n_data=n)
        if pcov is not None:
            logger.debug("Jacobian fallback pcov computed successfully")
        return pcov

    except Exception as exc:
        logger.warning(
            "Jacobian fallback pcov computation failed",
            error=str(exc),
        )
        return None


def bootstrap_ci(
    model: BaseModel,
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 200,
    alpha: float = 0.05,
    seed: int = 42,
    test_mode: str | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute confidence intervals via residual bootstrap.

    Residual bootstrap procedure:

    1. Compute residuals ``r = y - ŷ`` from the current fit.
    2. For each of *n_bootstrap* iterations:

       a. Resample residual indices with replacement (JAX PRNG, reproducible).
       b. Construct synthetic data ``y_boot = ŷ + r[idx]``.  For complex y,
          the **same** index set is applied to both G' and G'' so that the
          frequency pairing is preserved.
       c. Deepcopy the model and refit it to ``(X, y_boot)`` with the current
          parameters as the warm-start.
       d. Record the fitted parameter vector.

    3. Compute percentile CIs: ``[alpha/2, 1 - alpha/2]`` quantiles over the
       bootstrap parameter distribution.

    Parameters
    ----------
    model:
        A fitted RheoJAX model (BaseModel subclass). Must have `fitted_=True`.
    X:
        Independent variable array (same as used at fit time).
    y:
        Dependent variable array (may be complex for oscillation data).
    n_bootstrap:
        Number of bootstrap replicates.  200 is adequate for rough estimates;
        use 1000+ for publication-quality percentile intervals.
    alpha:
        Significance level.  0.05 produces 95 % confidence intervals.
    seed:
        Integer seed for the JAX PRNG used to draw resample indices.
        Fixed seed → fully reproducible intervals.
    test_mode:
        Protocol identifier forwarded to `model.fit()` for each bootstrap
        replicate. If None, the stored `_test_mode` on the model is used.

    Returns
    -------
    dict[str, tuple[float, float]]
        Mapping ``parameter_name → (lower_bound, upper_bound)`` at the
        ``(1 - alpha)`` confidence level.

    Raises
    ------
    RuntimeError
        If the model has not been fitted, or if fewer than 2 bootstrap
        replicates converge successfully.

    Notes
    -----
    *Performance*: each replicate deep-copies the model and calls ``.fit()``.
    For ODE-based models (EPM, VLB, HVM, STZ, …) this can be slow.  Consider
    reducing *n_bootstrap* or using :func:`hessian_ci` for a fast approximation.

    *Memory*: bootstrap does not hold all synthetic datasets in memory
    simultaneously; only the parameter vectors are accumulated.

    Examples
    --------
    >>> ci = bootstrap_ci(model, t, G_relax, n_bootstrap=500, seed=0)
    >>> for name, (lo, hi) in ci.items():
    ...     print(f"{name}: [{lo:.4g}, {hi:.4g}]")
    """
    _assert_fitted(model)

    param_names = _param_names(model)
    n_params = len(param_names)
    n_obs = _n_obs(np.asarray(y))

    resolved_test_mode: str | None = test_mode or getattr(model, "_test_mode", None)

    logger.info(
        "bootstrap_ci started",
        model=model.__class__.__name__,
        n_bootstrap=n_bootstrap,
        n_obs=n_obs,
        n_params=n_params,
        alpha=alpha,
        seed=seed,
    )

    # ------------------------------------------------------------------
    # Baseline prediction and residuals
    # ------------------------------------------------------------------
    y_arr = np.asarray(y)
    is_complex = np.iscomplexobj(y_arr)

    y_pred_base = _predict_safe(model, X)
    y_pred_base = np.asarray(y_pred_base)

    if is_complex:
        # Ensure predictions are complex so residuals stay complex
        y_pred_base = y_pred_base.astype(np.complex128)
        y_arr = y_arr.astype(np.complex128)

    residuals_base = y_arr - y_pred_base  # shape (N,), possibly complex

    n_r = len(residuals_base)  # same as n_obs for complex arrays

    logger.debug(
        "Baseline residuals computed",
        residual_shape=residuals_base.shape,
        residual_dtype=str(residuals_base.dtype),
        is_complex=is_complex,
        rss=float(np.sum(np.abs(residuals_base) ** 2)),
    )

    # ------------------------------------------------------------------
    # Bootstrap loop
    # ------------------------------------------------------------------
    # Pre-generate all resample index sets using JAX PRNG for reproducibility.
    # Converting to numpy immediately so the loop body stays pure numpy/scipy.
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_bootstrap)
    # Shape: (n_bootstrap, n_r) — integer indices into residuals_base
    idx_all = np.asarray(
        jax.vmap(lambda k: jax.random.randint(k, (n_r,), 0, n_r))(keys),
        dtype=np.intp,
    )

    boot_params: list[np.ndarray] = []
    n_failed = 0

    # Collect kwargs that should be forwarded to each bootstrap refit.
    # We reproduce the protocol kwargs the original fit received by inspecting
    # _last_fit_kwargs on the model (if available).
    base_fit_kwargs: dict[str, Any] = {}
    if resolved_test_mode is not None:
        base_fit_kwargs["test_mode"] = resolved_test_mode
    _lfk = getattr(model, "_last_fit_kwargs", None)
    if _lfk is not None and isinstance(_lfk, dict):
        # Only forward protocol kwargs — never optimization meta-kwargs
        _SKIP = {
            "use_log_residuals",
            "use_multi_start",
            "n_starts",
            "perturb_factor",
            "max_iter",
            "ftol",
            "xtol",
            "gtol",
            "workflow",
            "fallback",
        }
        for k_name, v in _lfk.items():
            if k_name not in _SKIP:
                base_fit_kwargs.setdefault(k_name, v)

    for i in range(n_bootstrap):
        idx = idx_all[i]

        # Build synthetic dataset by resampling residuals (same idx for Re/Im)
        r_boot = residuals_base[idx]
        y_boot = y_pred_base + r_boot

        # Deep-copy model — preserves parameters, _test_mode, _last_fit_kwargs,
        # and all protocol caches without touching the original model.
        try:
            model_copy = copy.deepcopy(model)
        except Exception as exc:
            logger.warning(
                "deepcopy failed for bootstrap replicate",
                iteration=i,
                error=str(exc),
            )
            n_failed += 1
            continue

        # Refit the copy to the synthetic data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_copy.fit(X, y_boot, **base_fit_kwargs)

            fitted_vals = _param_values(model_copy)
            boot_params.append(fitted_vals)

        except Exception as exc:
            logger.debug(
                "Bootstrap replicate failed",
                iteration=i,
                error=str(exc),
            )
            n_failed += 1
            continue

    n_success = len(boot_params)
    logger.info(
        "bootstrap_ci completed",
        n_success=n_success,
        n_failed=n_failed,
        n_bootstrap=n_bootstrap,
    )

    if n_success < 2:
        raise RuntimeError(
            f"Bootstrap CI failed: only {n_success}/{n_bootstrap} replicates "
            f"converged successfully for {model.__class__.__name__}. "
            "Check model fit quality and consider using hessian_ci() instead."
        )

    if n_failed > 0:
        logger.warning(
            "Some bootstrap replicates failed",
            n_failed=n_failed,
            n_success=n_success,
            failure_rate=f"{100 * n_failed / n_bootstrap:.1f}%",
        )

    # ------------------------------------------------------------------
    # Percentile confidence intervals
    # ------------------------------------------------------------------
    params_matrix = np.stack(boot_params, axis=0)  # (n_success, n_params)

    lo_pct = 100.0 * alpha / 2.0
    hi_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lo = np.percentile(params_matrix, lo_pct, axis=0)
    ci_hi = np.percentile(params_matrix, hi_pct, axis=0)

    ci: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(param_names):
        ci[name] = (float(ci_lo[j]), float(ci_hi[j]))

    logger.debug(
        "Percentile intervals computed",
        alpha=alpha,
        lo_pct=lo_pct,
        hi_pct=hi_pct,
    )
    return ci
