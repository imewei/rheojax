"""
Bayesian Service
===============

Service for Bayesian inference with NumPyro and ArviZ diagnostics.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.registry import Registry
from rheojax.gui.services.model_service import normalize_model_name
from rheojax.gui.state.store import DatasetState
from rheojax.gui.utils.rheodata import rheodata_from_dataset_state
from rheojax.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BayesianResult:
    """Result from Bayesian inference.

    Attributes
    ----------
    model_name : str
        Name of the model
    posterior_samples : dict
        Posterior samples for each parameter
    summary : dict
        Summary statistics (mean, std, median, quantiles) for each parameter
    diagnostics : dict
        R-hat, ESS, divergences, etc.
    metadata : dict
        MCMC configuration and metadata
    inference_data : Any, optional
        Full ArviZ InferenceData with sample_stats for energy plots
    """

    model_name: str
    posterior_samples: dict[str, np.ndarray]
    summary: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]
    inference_data: Any | None = None


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
        logger.debug("Initializing BayesianService")
        self._registry = Registry.get_instance()
        logger.debug("BayesianService initialized successfully")

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
        logger.debug("Entering get_default_priors", model=model_name)
        try:
            model_name = normalize_model_name(model_name)
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

            logger.debug(
                "Exiting get_default_priors",
                model=model_name,
                num_priors=len(priors),
            )
            return priors

        except Exception as e:
            logger.error(
                f"Failed to get priors for {model_name}: {e}",
                exc_info=True,
            )
            return {}

    def run_mcmc(
        self,
        model_name: str,
        data: RheoData | DatasetState,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        warm_start: dict[str, float] | None = None,
        test_mode: str | None = None,
        progress_callback: Callable[[str, int, int, int], None] | None = None,
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
        progress_callback : Callable[[str, int, int, int], None], optional
            Progress callback: callback(stage, chain, iteration, total_iterations)
        **kwargs
            Additional NumPyro MCMC options

        Returns
        -------
        BayesianResult
            Posterior samples and diagnostics
        """
        logger.debug(
            "Entering run_mcmc",
            model=model_name,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            warm_start_provided=warm_start is not None,
            test_mode=test_mode,
        )
        try:
            model_name = normalize_model_name(model_name)
            # Create model instance
            model = self._registry.create_instance(model_name, plugin_type="model")

            # Normalize DatasetState -> RheoData
            if isinstance(data, DatasetState):
                data = rheodata_from_dataset_state(data)
                metadata = dict(data.metadata or {})
                test_mode = test_mode or metadata.get("test_mode")

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
                "Starting Bayesian inference",
                model=model_name,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                test_mode=test_mode,
                data_points=len(x),
                warm_start_used=warm_start is not None,
            )

            # Run Bayesian inference
            # Wrap progress callback to NumPyro/worker signature (stage, chain, iteration, total)
            if progress_callback:
                total_iterations = max(num_chains * (num_warmup + num_samples), 1)

                def _wrapped_callback(
                    stage: str, chain: int, iteration: int, total: int
                ):
                    progress_callback(stage, chain, iteration, total)

                progress_callback("warmup", 1, 0, total_iterations)
                # NumPyro NUTS does not accept a progress_callback arg; keep it GUI-side only.

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
                progress_callback(
                    "sampling",
                    num_chains,
                    num_samples,
                    max(num_chains * (num_warmup + num_samples), 1),
                )

            # Extract posterior samples
            posterior_samples = result.posterior_samples

            # Calculate diagnostics
            diagnostics = self.get_diagnostics(result)

            # Get full InferenceData with sample_stats for energy plots
            inference_data = None
            try:
                inference_data = result.to_inference_data()
            except Exception as e:
                logger.warning(f"Could not get InferenceData: {e}")

            # Metadata
            metadata = {
                "model_name": model_name,
                "test_mode": test_mode,
                "num_warmup": num_warmup,
                "num_samples": num_samples,
                "num_chains": num_chains,
                "warm_start_used": warm_start is not None,
            }

            # Extract summary diagnostics for logging
            max_r_hat = diagnostics.get("max_rhat")
            min_ess = diagnostics.get("min_ess")
            divergences = diagnostics.get("divergences", 0)

            logger.info(
                "Bayesian inference complete",
                model=model_name,
                r_hat=max_r_hat,
                ess=min_ess,
                divergences=divergences,
                num_parameters=len(posterior_samples),
            )

            logger.debug(
                "Exiting run_mcmc",
                model=model_name,
                success=True,
            )

            return BayesianResult(
                model_name=model_name,
                posterior_samples=posterior_samples,
                summary=result.summary,
                diagnostics=diagnostics,
                metadata=metadata,
                inference_data=inference_data,
            )

        except Exception as e:
            logger.error(
                f"MCMC failed for {model_name}: {e}",
                exc_info=True,
            )
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
        logger.debug("Entering get_diagnostics")
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
                num_chains = (
                    result.metadata.get("num_chains") if result.metadata else None
                )
            elif hasattr(result, "metadata") and isinstance(result.metadata, dict):
                num_chains = result.metadata.get("num_chains")

            # Fall back to core BayesianResult.num_chains when metadata is absent
            if num_chains is None and hasattr(result, "num_chains"):
                try:
                    num_chains = int(result.num_chains)
                except Exception:
                    num_chains = None

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
                "r_hat": rhat_dict,  # alias for GUI/state consumers
                "ess": ess_dict,
                "divergences": divergences,
                "max_rhat": max(rhat_dict.values()) if rhat_dict else None,
                "max_r_hat": max(rhat_dict.values()) if rhat_dict else None,
                "min_ess": min(ess_dict.values()) if ess_dict else None,
            }

            # Add warnings
            warnings = []
            if diagnostics["max_rhat"] and diagnostics["max_rhat"] > 1.1:
                warnings.append(
                    f"High R-hat detected: {diagnostics['max_rhat']:.3f} > 1.1"
                )
            if diagnostics["min_ess"] and diagnostics["min_ess"] < 400:
                warnings.append(f"Low ESS detected: {diagnostics['min_ess']:.0f} < 400")
            if divergences > 0:
                warnings.append(f"{divergences} divergent transitions detected")

            diagnostics["warnings"] = warnings

            logger.debug(
                "Exiting get_diagnostics",
                max_rhat=diagnostics["max_rhat"],
                min_ess=diagnostics["min_ess"],
                divergences=divergences,
                num_warnings=len(warnings),
            )

            return diagnostics

        except Exception as e:
            logger.error(f"Diagnostic calculation failed: {e}", exc_info=True)
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
        logger.debug(
            "Entering get_credible_intervals",
            model=result.model_name,
            prob=prob,
        )
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

        logger.debug(
            "Exiting get_credible_intervals",
            num_parameters=len(intervals),
        )
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
        logger.debug("Entering get_posterior_means", model=result.model_name)
        means = {}
        for param_name, samples in result.posterior_samples.items():
            means[param_name] = float(np.mean(samples))
        logger.debug("Exiting get_posterior_means", num_parameters=len(means))
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
        logger.debug(
            "Entering compare_models",
            num_models=len(results),
            criterion=criterion,
        )
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

            logger.debug(
                "Exiting compare_models",
                num_models=len(comparison),
                criterion=criterion,
            )
            return comparison

        except Exception as e:
            logger.error(f"Model comparison failed: {e}", exc_info=True)
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
        logger.debug(
            "Entering validate_priors",
            model=model_name,
            num_priors=len(priors),
        )
        warnings = []

        try:
            model_name = normalize_model_name(model_name)
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
                        warnings.append(
                            f"Uniform prior for {param_name} missing bounds"
                        )
                    elif lower >= upper:
                        warnings.append(
                            f"Invalid bounds for {param_name}: {lower} >= {upper}"
                        )

        except Exception as e:
            logger.error(f"Prior validation failed: {e}", exc_info=True)
            warnings.append(f"Prior validation failed: {e}")

        logger.debug(
            "Exiting validate_priors",
            num_warnings=len(warnings),
        )
        return warnings
