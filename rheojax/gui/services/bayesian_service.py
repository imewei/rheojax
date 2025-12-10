"""
Bayesian Service
===============

Service for Bayesian inference with NumPyro and ArviZ diagnostics.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.registry import Registry

logger = logging.getLogger(__name__)


@dataclass
class BayesianResult:
    """Result from Bayesian inference.

    Attributes
    ----------
    model_name : str
        Name of the model
    posterior_samples : dict
        Posterior samples for each parameter
    diagnostics : dict
        R-hat, ESS, divergences, etc.
    metadata : dict
        MCMC configuration and metadata
    """

    model_name: str
    posterior_samples: dict[str, np.ndarray]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]


class BayesianService:
    """Service for Bayesian inference operations.

    Features:
        - NUTS sampling with NumPyro
        - NLSQ warm-start initialization
        - ArviZ diagnostic integration
        - Credible intervals and posterior analysis
        - Prior specification and validation

    Example
    -------
    >>> service = BayesianService()
    >>> result = service.run_mcmc('maxwell', data, num_warmup=1000, num_samples=2000)
    >>> intervals = service.get_credible_intervals(result, prob=0.95)
    """

    def __init__(self) -> None:
        """Initialize Bayesian service."""
        self._registry = Registry.get_instance()

    def get_default_priors(self, model_name: str) -> dict[str, dict]:
        """Get default prior distributions for model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        dict
            Prior specifications for each parameter
            Format: {'param_name': {'dist': 'uniform', 'args': [lower, upper]}}
        """
        try:
            model = self._registry.create_instance(model_name, plugin_type="model")

            priors = {}
            for param_name, param in model.parameters.items():
                lower, upper = param.bounds
                # Default to uniform priors over parameter bounds
                priors[param_name] = {
                    "distribution": "uniform",
                    "lower": float(lower),
                    "upper": float(upper),
                }

            return priors

        except Exception as e:
            logger.error(f"Failed to get priors for {model_name}: {e}")
            return {}

    def run_mcmc(
        self,
        model_name: str,
        data: RheoData,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        warm_start: dict[str, float] | None = None,
        test_mode: str | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
        **kwargs: Any,
    ) -> BayesianResult:
        """Run NUTS sampling for Bayesian inference.

        Parameters
        ----------
        model_name : str
            Name of the model
        data : RheoData
            Rheological data
        num_warmup : int, default=1000
            Number of warmup/burn-in iterations
        num_samples : int, default=2000
            Number of posterior samples per chain
        num_chains : int, default=4
            Number of MCMC chains
        warm_start : dict, optional
            Initial parameter values (e.g., from NLSQ fit)
        test_mode : str, optional
            Test mode (relaxation, creep, oscillation)
        progress_callback : Callable[[int, int, int], None], optional
            Progress callback: callback(chain, iteration, total_iterations)
        **kwargs
            Additional NumPyro MCMC options

        Returns
        -------
        BayesianResult
            Posterior samples and diagnostics
        """
        try:
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Set initial values if provided
            if warm_start:
                for name, value in warm_start.items():
                    if name in model.parameters:
                        model.parameters[name].value = value

            # Extract data
            x = np.asarray(data.x)
            y = np.asarray(data.y)

            # Determine test mode
            if test_mode is None:
                test_mode = data.metadata.get("test_mode", "oscillation")

            logger.info(
                f"Running MCMC for {model_name}: "
                f"{num_warmup} warmup + {num_samples} samples × {num_chains} chains"
            )

            # Run Bayesian inference
            # Emit initial progress
            if progress_callback:
                progress_callback(0, 1, "warmup")
                kwargs.setdefault("progress_callback", progress_callback)

            result = model.fit_bayesian(
                x,
                y,
                test_mode=test_mode,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                **kwargs,
            )

            if progress_callback:
                progress_callback(100, 100, "complete")

            # Extract posterior samples
            posterior_samples = result.posterior_samples

            # Calculate diagnostics
            diagnostics = self.get_diagnostics(result)

            # Metadata
            metadata = {
                "model_name": model_name,
                "test_mode": test_mode,
                "num_warmup": num_warmup,
                "num_samples": num_samples,
                "num_chains": num_chains,
                "warm_start_used": warm_start is not None,
            }

            return BayesianResult(
                model_name=model_name,
                posterior_samples=posterior_samples,
                diagnostics=diagnostics,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"MCMC failed for {model_name}: {e}")
            raise RuntimeError(f"MCMC failed: {e}") from e

    def get_diagnostics(self, result: BayesianResult | Any) -> dict[str, Any]:
        """Calculate MCMC diagnostics using ArviZ.

        Parameters
        ----------
        result : BayesianResult or MCMC result
            Bayesian inference result

        Returns
        -------
        dict
            Diagnostics including R-hat, ESS, divergences
        """
        try:
            import arviz as az

            # Convert to ArviZ InferenceData if needed
            if isinstance(result, BayesianResult):
                posterior_samples = result.posterior_samples
            elif hasattr(result, "posterior_samples"):
                posterior_samples = result.posterior_samples
            else:
                logger.warning("Unable to extract posterior samples for diagnostics")
                return {}

            # Convert to ArviZ format
            # Reshape samples: (n_chains, n_draws)
            idata_dict = {}
            # Try to infer chain count from result metadata when available
            num_chains = None
            if isinstance(result, BayesianResult):
                num_chains = result.metadata.get("num_chains") if result.metadata else None
            elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                num_chains = result.metadata.get("num_chains")

            for param_name, samples in posterior_samples.items():
                arr = np.asarray(samples)
                if arr.ndim == 1 and num_chains and arr.size % num_chains == 0:
                    idata_dict[param_name] = arr.reshape(num_chains, -1)
                elif arr.ndim == 1:
                    # Fallback: assume single chain
                    idata_dict[param_name] = arr.reshape(1, -1)
                else:
                    idata_dict[param_name] = arr

            idata = az.from_dict(idata_dict)

            # Calculate R-hat (Gelman-Rubin statistic)
            rhat = az.rhat(idata)
            rhat_dict = {k: float(v.values) for k, v in rhat.data_vars.items()}

            # Calculate ESS (Effective Sample Size)
            ess = az.ess(idata)
            ess_dict = {k: float(v.values) for k, v in ess.data_vars.items()}

            # Check for divergences (if available)
            divergences = 0
            if hasattr(result, "diagnostics") and "divergences" in result.diagnostics:
                divergences = int(result.diagnostics["divergences"])

            diagnostics = {
                "rhat": rhat_dict,
                "ess": ess_dict,
                "divergences": divergences,
                "max_rhat": max(rhat_dict.values()) if rhat_dict else None,
                "min_ess": min(ess_dict.values()) if ess_dict else None,
            }

            # Add warnings
            warnings = []
            if diagnostics["max_rhat"] and diagnostics["max_rhat"] > 1.1:
                warnings.append(f"High R-hat detected: {diagnostics['max_rhat']:.3f} > 1.1")
            if diagnostics["min_ess"] and diagnostics["min_ess"] < 400:
                warnings.append(f"Low ESS detected: {diagnostics['min_ess']:.0f} < 400")
            if divergences > 0:
                warnings.append(f"{divergences} divergent transitions detected")

            diagnostics["warnings"] = warnings

            return diagnostics

        except Exception as e:
            logger.error(f"Diagnostic calculation failed: {e}")
            return {"error": str(e)}

    def get_credible_intervals(
        self,
        result: BayesianResult,
        prob: float = 0.95,
    ) -> dict[str, tuple[float, float, float]]:
        """Get credible intervals from posterior.

        Parameters
        ----------
        result : BayesianResult
            Bayesian inference result
        prob : float, default=0.95
            Credibility level (0-1)

        Returns
        -------
        dict
            Credible intervals for each parameter
            Format: {'param': (lower, median, upper)}
        """
        intervals = {}

        alpha = 1 - prob
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)

        for param_name, samples in result.posterior_samples.items():
            # Flatten samples across chains if needed
            samples_flat = samples.flatten() if samples.ndim > 1 else samples

            lower = float(np.percentile(samples_flat, lower_percentile))
            median = float(np.percentile(samples_flat, 50))
            upper = float(np.percentile(samples_flat, upper_percentile))

            intervals[param_name] = (lower, median, upper)

        return intervals

    def get_posterior_means(self, result: BayesianResult) -> dict[str, float]:
        """Get posterior mean for each parameter.

        Parameters
        ----------
        result : BayesianResult
            Bayesian inference result

        Returns
        -------
        dict
            Posterior means
        """
        means = {}
        for param_name, samples in result.posterior_samples.items():
            means[param_name] = float(np.mean(samples))
        return means

    def compare_models(
        self,
        results: list[BayesianResult],
        criterion: str = "waic",
    ) -> dict[str, float]:
        """Compare models using information criteria.

        Parameters
        ----------
        results : list[BayesianResult]
            List of Bayesian results to compare
        criterion : str, default='waic'
            Information criterion ('waic', 'loo', 'dic')

        Returns
        -------
        dict
            Model comparison scores
        """
        try:
            import arviz as az

            comparison = {}

            for result in results:
                model_name = result.model_name

                # Convert to InferenceData
                idata_dict = {
                    k: v.reshape(1, -1) if v.ndim == 1 else v
                    for k, v in result.posterior_samples.items()
                }
                idata = az.from_dict(idata_dict)

                # Calculate criterion
                if criterion.lower() == "waic":
                    score = az.waic(idata)
                    comparison[model_name] = float(score.waic)
                elif criterion.lower() == "loo":
                    score = az.loo(idata)
                    comparison[model_name] = float(score.loo)
                else:
                    logger.warning(f"Unknown criterion: {criterion}")

            return comparison

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {}

    def validate_priors(
        self,
        model_name: str,
        priors: dict[str, dict],
    ) -> list[str]:
        """Validate prior specifications.

        Parameters
        ----------
        model_name : str
            Name of the model
        priors : dict
            Prior specifications

        Returns
        -------
        list[str]
            Validation warnings
        """
        warnings = []

        try:
            model = self._registry.create_instance(model_name, plugin_type="model")

            for param_name in model.parameters:
                if param_name not in priors:
                    warnings.append(f"No prior specified for {param_name}")
                    continue

                prior = priors[param_name]
                dist = prior.get("distribution", "").lower()

                if dist not in ["uniform", "normal", "lognormal", "gamma", "beta"]:
                    warnings.append(f"Unknown distribution for {param_name}: {dist}")

                # Check bounds
                if dist == "uniform":
                    lower = prior.get("lower")
                    upper = prior.get("upper")
                    if lower is None or upper is None:
                        warnings.append(f"Uniform prior for {param_name} missing bounds")
                    elif lower >= upper:
                        warnings.append(f"Invalid bounds for {param_name}: {lower} >= {upper}")

        except Exception as e:
            warnings.append(f"Prior validation failed: {e}")

        return warnings
