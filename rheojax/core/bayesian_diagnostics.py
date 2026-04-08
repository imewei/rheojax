"""Bayesian convergence diagnostics: R-hat, ESS, divergence counting.

This module provides functions for computing MCMC convergence diagnostics
from posterior samples. Used by BayesianMixin._compute_diagnostics().
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.bayesian_result import DiagnosticsDict
from rheojax.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from numpyro.infer import MCMC


def _import_numpyro_diagnostics():
    """Lazy-import NumPyro diagnostics functions."""
    import numpyro.diagnostics

    return numpyro.diagnostics


def compute_per_param_diagnostic(
    posterior_samples: dict[str, np.ndarray],
    num_chains: int,
    num_samples: int,
    diagnostic_fn: Callable,
    label: str,
) -> tuple[dict[str, float], bool]:
    """Compute a per-parameter diagnostic (R-hat or ESS).

    Reshapes flat posterior samples to (num_chains, num_samples) and applies
    the given NumPyro diagnostic function to each parameter.

    Args:
        posterior_samples: Parameter name -> flat samples array.
        num_chains: Number of MCMC chains.
        num_samples: Samples per chain.
        diagnostic_fn: NumPyro function (e.g. split_gelman_rubin, effective_sample_size).
        label: Human-readable label for log messages (e.g. "R-hat", "ESS").

    Returns:
        (result_dict, all_succeeded) where result_dict maps parameter names
        to diagnostic values (NaN on failure).
    """
    result_dict: dict[str, float] = {}
    all_succeeded = True
    for name in posterior_samples:
        try:
            samples_arr = posterior_samples[name]
            # If already shaped (num_chains, num_samples), use directly
            if samples_arr.ndim == 2:
                samples_shaped = samples_arr
            elif num_chains >= 2:
                # Use -1 to infer actual draws (may differ from num_samples
                # due to thinning, warmup inclusion, or early stopping)
                samples_shaped = samples_arr.reshape(num_chains, -1)
            else:
                samples_shaped = samples_arr.reshape(1, -1)
            result_dict[name] = float(diagnostic_fn(samples_shaped))
        except Exception as exc:
            # R12-B-008: log array size and chain count to help diagnose
            # reshape failures caused by unexpected sample counts (e.g., thinning
            # or early stopping causing total_samples % num_chains != 0).
            logger.debug(
                "Posterior reshape failed for parameter",
                parameter=name,
                array_size=getattr(samples_arr, "size", None),
                num_chains=num_chains,
            )
            logger.warning(
                f"{label} computation failed for parameter",
                parameter=name,
                error=str(exc),
            )
            # R12-B-010: NaN fallback is correct here — it signals "diagnostic
            # unavailable" to the caller without hiding genuine failures.
            # Downstream code filters NaN values when computing summary
            # statistics (max R-hat, min ESS), so a NaN for one parameter
            # does not silently corrupt the aggregate.
            result_dict[name] = float("nan")
            all_succeeded = False
    return result_dict, all_succeeded


def compute_diagnostics(
    mcmc: MCMC,
    posterior_samples: dict[str, np.ndarray],
    num_samples: int,
    num_chains: int,
) -> DiagnosticsDict:
    """Compute convergence diagnostics from MCMC samples.

    Args:
        mcmc: NumPyro MCMC object after sampling
        posterior_samples: Dictionary of posterior samples
        num_samples: Number of samples per chain
        num_chains: Number of MCMC chains

    Returns:
        Dictionary with diagnostic information:
            - r_hat: R-hat (Gelman-Rubin) statistic per parameter (NaN if failed)
            - ess: Effective sample size per parameter (NaN if failed)
            - divergences: Number of divergent transitions (-1 if unknown)
            - diagnostics_valid: Whether all diagnostics computed successfully
    """
    numpyro_diag = _import_numpyro_diagnostics()

    diagnostics: DiagnosticsDict = {}
    all_valid = True

    try:
        r_hat_dict, r_hat_ok = compute_per_param_diagnostic(
            posterior_samples,
            num_chains,
            num_samples,
            numpyro_diag.split_gelman_rubin,
            "R-hat",
        )
        diagnostics["r_hat"] = r_hat_dict
        all_valid = all_valid and r_hat_ok

        ess_dict, ess_ok = compute_per_param_diagnostic(
            posterior_samples,
            num_chains,
            num_samples,
            numpyro_diag.effective_sample_size,
            "ESS",
        )
        diagnostics["ess"] = ess_dict
        all_valid = all_valid and ess_ok

        try:
            try:
                divergences = mcmc.get_extra_fields(group_by_chain=True)["diverging"]
            except TypeError:
                divergences = mcmc.get_extra_fields()["diverging"]
            num_divergences = int(np.sum(divergences))
        except (KeyError, AttributeError):
            logger.warning(
                "Divergence information not available from MCMC extra fields. "
                "Divergence count is unknown — inspect trace plots manually."
            )
            num_divergences = -1
            all_valid = False

        diagnostics["divergences"] = num_divergences
        diagnostics["total_samples"] = int(num_samples * num_chains)
        diagnostics["num_chains"] = int(num_chains)
        diagnostics["num_samples_per_chain"] = int(num_samples)

    except Exception as e:
        logger.error(
            "Diagnostics computation failed entirely",
            error=str(e),
            exc_info=True,
        )
        warnings.warn(
            f"MCMC diagnostics computation failed: {e}. "
            "R-hat, ESS, and divergence values are unavailable. "
            "Posterior samples may be unreliable — inspect trace plots manually.",
            RuntimeWarning,
            stacklevel=3,
        )
        diagnostics["r_hat"] = dict.fromkeys(posterior_samples.keys(), float("nan"))
        diagnostics["ess"] = dict.fromkeys(posterior_samples.keys(), float("nan"))
        diagnostics["divergences"] = -1
        diagnostics["total_samples"] = int(num_samples * num_chains)
        diagnostics["num_chains"] = int(num_chains)
        diagnostics["num_samples_per_chain"] = int(num_samples)
        diagnostics["error"] = str(e)
        all_valid = False

    diagnostics["diagnostics_valid"] = all_valid

    actual_divergences = diagnostics.get("divergences", -1)
    if actual_divergences > 0:
        logger.warning(
            "Divergent transitions detected",
            divergences=actual_divergences,
            total_samples=num_samples * num_chains,
            hint="Try: different seed=, increased num_warmup, or higher target_accept_prob",
        )

    return diagnostics


__all__ = [
    "compute_diagnostics",
    "compute_per_param_diagnostic",
]
