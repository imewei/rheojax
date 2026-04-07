"""BayesianResult dataclass and related types for Bayesian inference output.

This module contains the result container returned by BayesianMixin.fit_bayesian(),
including ArviZ InferenceData conversion with fast-path xarray assembly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np

from rheojax.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from numpyro.infer import MCMC


class DiagnosticsDict(TypedDict, total=False):
    """Typed structure for Bayesian convergence diagnostics."""

    r_hat: dict[str, float]
    ess: dict[str, float]
    divergences: int
    diagnostics_valid: bool
    total_samples: int
    num_chains: int
    num_samples_per_chain: int
    error: str


@dataclass
class BayesianResult:
    """Results from Bayesian inference with NUTS sampling.

    This dataclass stores the complete results of NumPyro NUTS sampling,
    including posterior samples, summary statistics, convergence diagnostics,
    and placeholders for future model comparison metrics.

    Attributes:
        posterior_samples: Dictionary mapping parameter names to arrays of
            posterior samples (shape: [num_samples * num_chains, ]). All arrays are float64.
        summary: Dictionary with summary statistics for each parameter.
            Contains nested dicts with 'mean', 'std', and quantiles.
        diagnostics: Dictionary with convergence diagnostics including:
            - r_hat: Gelman-Rubin statistic for each parameter (dict)
            - ess: Effective sample size for each parameter (dict)
            - divergences: Number of divergent transitions (int)
        num_samples: Number of posterior samples per chain (after warmup).
        num_chains: Number of MCMC chains used in sampling.
        mcmc: NumPyro MCMC object containing full sampling information including
            NUTS-specific diagnostics (energy, divergences, tree depth).
            Required for ArviZ visualization with full diagnostics.
        model_comparison: Dictionary for model comparison metrics (WAIC, LOO).
            Currently a placeholder for future implementation.
        _inference_data: Cached ArviZ InferenceData object. Automatically
            created on first access via to_inference_data(). Do not set manually.

    Example:
        >>> result = model.fit_bayesian(X, y)
        >>> print(result.summary["a"]["mean"])
        >>> print(result.diagnostics["r_hat"]["a"])
        >>> # Convert to ArviZ InferenceData for advanced plotting
        >>> idata = result.to_inference_data()
    """

    posterior_samples: dict[str, np.ndarray]
    summary: dict[str, dict[str, float]]
    diagnostics: DiagnosticsDict
    num_samples: int
    num_chains: int
    mcmc: MCMC | None = None
    model_comparison: dict[str, float] = field(default_factory=dict)
    _inference_data: Any | None = field(default=None, repr=False)
    _inference_data_ll: Any | None = field(default=None, repr=False)

    def __post_init__(self):
        """Validate result after initialization."""
        logger.debug(
            "Initializing BayesianResult",
            num_parameters=len(self.posterior_samples),
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        # Ensure posterior_samples are float64 numpy arrays
        # BAY-05: skip copy when already float64 NumPy to avoid eager allocation
        for name, samples in self.posterior_samples.items():
            arr = np.asarray(samples)
            if arr.dtype != np.float64:
                arr = arr.astype(np.float64)
            self.posterior_samples[name] = arr
        logger.debug(
            "BayesianResult initialized",
            parameter_names=list(self.posterior_samples.keys()),
        )

    def to_inference_data(self, log_likelihood: bool = False) -> Any:
        """Convert to ArviZ InferenceData format for advanced visualization.

        Converts the NumPyro MCMC result to ArviZ InferenceData format, which
        enables access to ArviZ's comprehensive plotting and diagnostic tools.
        The conversion preserves all NUTS-specific diagnostics including energy,
        divergences, and tree depth information.

        The InferenceData object is cached after first conversion to avoid
        repeated conversion overhead. The ``log_likelihood=False`` and
        ``log_likelihood=True`` variants are cached independently.

        Args:
            log_likelihood: If True, compute pointwise log-likelihood for
                WAIC/LOO model comparison (az.waic(), az.loo()). This
                re-evaluates the model for all samples (~600-800ms slower).
                Default False for faster conversion when only plotting.

        Returns:
            ArviZ InferenceData object containing:
                - posterior: Posterior samples for all parameters
                - sample_stats: NUTS diagnostics (energy, divergences, etc.)
                - log_likelihood: Only when ``log_likelihood=True``
                - Additional groups as available from NumPyro

        Raises:
            ImportError: If arviz is not installed
            ValueError: If MCMC object was not stored (older results)

        Example:
            >>> result = model.fit_bayesian(X, y)
            >>> idata = result.to_inference_data()  # Fast: no log-lik
            >>> az.plot_trace(idata)
            >>>
            >>> # For model comparison (slower):
            >>> idata_ll = result.to_inference_data(log_likelihood=True)
            >>> az.waic(idata_ll)

        Note:
            Requires arviz package: pip install arviz
            The MCMC object must be present (automatically stored by fit_bayesian).
        """

        logger.debug(
            "Converting BayesianResult to InferenceData",
            log_likelihood=log_likelihood,
        )

        # Return cached version if available
        if log_likelihood and self._inference_data_ll is not None:
            logger.debug("Returning cached InferenceData (with log_likelihood)")
            return self._inference_data_ll
        if not log_likelihood and self._inference_data is not None:
            logger.debug("Returning cached InferenceData (without log_likelihood)")
            return self._inference_data

        # Ensure MCMC object is available
        if self.mcmc is None:
            logger.error("MCMC object not available for InferenceData conversion")
            raise ValueError(
                "MCMC object not available for conversion. "
                "This may be a result from an older version. "
                "Re-run fit_bayesian() to generate a compatible result."
            )

        # Import arviz (lazy import)
        from rheojax.core.arviz_utils import import_arviz

        try:
            az = import_arviz(required=("InferenceData",))
            logger.debug("ArviZ imported successfully")
        except ImportError as exc:
            logger.error("ArviZ import failed", exc_info=True)
            raise ImportError(
                "ArviZ is required for InferenceData conversion. "
                "Install it with: pip install arviz"
            ) from exc

        if log_likelihood:
            # log_likelihood=True requires model re-evaluation (slow path).
            # Delegate to az.from_numpyro which traces the model to extract
            # pointwise log-likelihoods for WAIC/LOO computation.
            logger.debug(
                "Creating InferenceData from MCMC object (with log_likelihood)",
            )
            try:
                az_full = import_arviz(required=("from_numpyro",))
                idata = az_full.from_numpyro(self.mcmc, log_likelihood=True)
            except ImportError as exc:
                raise ImportError(
                    "ArviZ is required for log-likelihood computation. "
                    "Install it with: pip install arviz"
                ) from exc
            logger.info(
                "InferenceData created successfully (with log_likelihood)",
                num_chains=self.num_chains,
                num_samples=self.num_samples,
            )
            self._inference_data_ll = idata
            return idata

        # Fast path (log_likelihood=False): build InferenceData directly from
        # numpy arrays already present in BayesianResult, bypassing the
        # az.from_numpyro model-trace that triggers XLA recompilation
        # (~500-1500ms on first call). This reduces conversion to <5ms.
        #
        # ArviZ's from_numpyro calls numpyro.handlers.trace().get_trace() in
        # NumPyroConverter.__init__ to discover observed sites for the
        # observed_data / log_likelihood groups. When log_likelihood=False we
        # don't need those groups, so we can skip the trace entirely and
        # assemble the two groups we do need (posterior, sample_stats) from
        # data that is already on the host.
        logger.debug("Building InferenceData directly (fast path, no model trace)")

        import xarray as xr

        # --- posterior group ---
        # Use get_samples(group_by_chain=True) to include ALL sampled sites
        # (model parameters + deterministic sites like num_nonfinite) so the
        # result matches what az.from_numpyro would return.
        # self.posterior_samples only contains param_names + sigma params;
        # deterministic sites (numpyro.deterministic) are excluded from it.
        num_chains = self.num_chains
        posterior_dict: dict[str, xr.DataArray] = {}
        try:
            # Prefer group_by_chain samples — already (num_chains, num_draws)
            _raw_samples = self.mcmc.get_samples(group_by_chain=True)
            if hasattr(_raw_samples, "_asdict"):
                _raw_samples = _raw_samples._asdict()
            for name, arr in _raw_samples.items():
                np_arr = np.asarray(arr)
                if np_arr.ndim >= 2:
                    # Shape is already (num_chains, num_draws[, ...])
                    posterior_dict[name] = xr.DataArray(
                        np_arr, dims=("chain", "draw") + tuple(
                            f"dim_{i}" for i in range(np_arr.ndim - 2)
                        )
                    )
                else:
                    # Fallback: reshape flat array
                    try:
                        shaped = np_arr.reshape(num_chains, -1)
                    except ValueError:
                        shaped = np_arr[np.newaxis, :]
                    posterior_dict[name] = xr.DataArray(shaped, dims=("chain", "draw"))
        except Exception as exc:
            # Fall back to posterior_samples (already numpy, already on host)
            logger.debug(
                "get_samples(group_by_chain=True) failed, using posterior_samples",
                error=str(exc),
            )
            for name, flat_arr in self.posterior_samples.items():
                try:
                    shaped = flat_arr.reshape(num_chains, -1)
                except ValueError:
                    shaped = flat_arr[np.newaxis, :]
                posterior_dict[name] = xr.DataArray(shaped, dims=("chain", "draw"))

        posterior_ds = xr.Dataset(posterior_dict)

        # --- sample_stats group ---
        # Mirrors ArviZ's NumPyroConverter.sample_stats_to_xarray() rename map.
        _stat_rename = {
            "potential_energy": "lp",
            "adapt_state.step_size": "step_size",
            "num_steps": "n_steps",
            "accept_prob": "acceptance_rate",
        }
        stats_dict: dict[str, xr.DataArray] = {}
        try:
            try:
                extra_fields = self.mcmc.get_extra_fields(group_by_chain=True)
            except TypeError:
                extra_fields = self.mcmc.get_extra_fields()

            if isinstance(extra_fields, dict):
                for stat, value in extra_fields.items():
                    if isinstance(value, (dict, tuple)):
                        continue
                    arr = np.asarray(value)
                    # Ensure (chain, draw) shape
                    if arr.ndim == 1:
                        try:
                            arr = arr.reshape(num_chains, -1)
                        except ValueError:
                            arr = arr[np.newaxis, :]
                    elif arr.ndim != 2:
                        continue  # Skip unexpected shapes

                    # Match ArviZ's NumPyroConverter: potential_energy → negated "lp".
                    # Also synthesise "energy" from potential_energy so that
                    # ArviZ's plot_energy() works out of the box.  True HMC
                    # energy = potential + kinetic, but potential_energy alone is
                    # the standard proxy when kinetic energy is not stored.
                    if stat == "potential_energy":
                        stats_dict["lp"] = xr.DataArray(-arr, dims=("chain", "draw"))
                        stats_dict["energy"] = xr.DataArray(arr, dims=("chain", "draw"))
                    else:
                        dest_name = _stat_rename.get(stat, stat)
                        stats_dict[dest_name] = xr.DataArray(arr, dims=("chain", "draw"))
        except Exception as exc:
            logger.debug(
                "Failed to extract sample_stats from MCMC extra fields",
                error=str(exc),
            )

        sample_stats_ds = xr.Dataset(stats_dict)

        idata = az.InferenceData(
            posterior=posterior_ds,
            sample_stats=sample_stats_ds,
        )

        logger.info(
            "InferenceData created successfully (fast path)",
            num_chains=self.num_chains,
            num_samples=self.num_samples,
        )

        self._inference_data = idata
        return idata


__all__ = [
    "BayesianResult",
    "DiagnosticsDict",
]
