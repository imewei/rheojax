"""Specialized pipeline for Bayesian workflows.

This module provides the BayesianPipeline class for orchestrating the complete
NLSQ → NumPyro NUTS workflow with a fluent API.

Example:
    >>> from rheo.pipeline.bayesian import BayesianPipeline
    >>> pipeline = BayesianPipeline()
    >>> result = (pipeline
    ...     .load('data.csv')
    ...     .fit_nlsq('maxwell')
    ...     .fit_bayesian(num_samples=2000)
    ...     .plot_posterior()
    ...     .save('results.hdf5'))
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rheo.core.base import BaseModel
from rheo.core.jax_config import safe_import_jax
from rheo.core.registry import ModelRegistry
from rheo.pipeline.base import Pipeline

# Safe JAX import (verifies NLSQ was imported first)
jax, jnp = safe_import_jax()


class BayesianPipeline(Pipeline):
    """Specialized pipeline for Bayesian rheological analysis workflows.

    This class extends the base Pipeline to provide a fluent API for the
    NLSQ → NumPyro NUTS workflow. It supports:
    - NLSQ optimization for fast point estimation
    - Bayesian inference with automatic warm-start from NLSQ
    - Convergence diagnostics (R-hat, ESS, divergences)
    - Posterior visualization (distributions and trace plots)

    All methods return self to enable method chaining.

    Attributes:
        data: Current RheoData state (inherited from Pipeline)
        _last_model: Last fitted model (inherited from Pipeline)
        _nlsq_result: Stored NLSQ optimization result
        _bayesian_result: Stored Bayesian inference result
        _diagnostics: Stored convergence diagnostics

    Example:
        >>> pipeline = BayesianPipeline()
        >>> pipeline.load('data.csv') \\
        ...     .fit_nlsq('maxwell') \\
        ...     .fit_bayesian(num_samples=2000) \\
        ...     .plot_posterior() \\
        ...     .save('results.hdf5')
    """

    def __init__(self, data=None):
        """Initialize Bayesian pipeline.

        Args:
            data: Optional initial RheoData. If None, must call load() first.
        """
        super().__init__(data=data)
        self._nlsq_result = None
        self._bayesian_result = None
        self._diagnostics = None

    def fit_nlsq(self, model: str | BaseModel, **nlsq_kwargs) -> BayesianPipeline:
        """Fit model using NLSQ optimization for point estimation.

        This method performs fast GPU-accelerated nonlinear least squares
        optimization to obtain point estimates of model parameters. The
        optimization result is stored for potential warm-starting of
        Bayesian inference.

        Args:
            model: Model name (string) or Model instance to fit
            **nlsq_kwargs: Additional arguments passed to NLSQ optimizer
                (e.g., max_iter, ftol, xtol, gtol)

        Returns:
            self for method chaining

        Raises:
            ValueError: If data not loaded

        Example:
            >>> pipeline.fit_nlsq('maxwell')
            >>> # or with instance
            >>> from rheo.models import Maxwell
            >>> pipeline.fit_nlsq(Maxwell(), max_iter=1000)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Create model if string
        if isinstance(model, str):
            model_obj = ModelRegistry.create(model)
            model_name = model
        else:
            model_obj = model
            model_name = model_obj.__class__.__name__

        # Fit using model's fit method (uses NLSQ by default)
        X = self.data.x
        y = self.data.y

        # Convert to numpy for fitting
        if isinstance(X, jnp.ndarray):
            X = np.array(X)
        if isinstance(y, jnp.ndarray):
            y = np.array(y)

        model_obj.fit(X, y, method="nlsq", **nlsq_kwargs)

        # Store fitted model
        self._last_model = model_obj
        self.steps.append(("fit_nlsq", model_obj))
        self.history.append(("fit_nlsq", model_name, model_obj.score(X, y)))

        # Store NLSQ result from model
        self._nlsq_result = model_obj.get_nlsq_result()

        return self

    def fit_bayesian(
        self, num_samples: int = 2000, num_warmup: int = 1000, **nuts_kwargs
    ) -> BayesianPipeline:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        This method runs NUTS (No-U-Turn Sampler) for Bayesian parameter
        estimation. If a model has been previously fitted with fit_nlsq(),
        the NLSQ point estimates are automatically used for warm-starting
        the sampler, leading to faster convergence.

        Args:
            num_samples: Number of posterior samples to collect (default: 2000)
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            **nuts_kwargs: Additional arguments passed to NUTS sampler

        Returns:
            self for method chaining

        Raises:
            ValueError: If no model has been fitted with fit_nlsq()

        Example:
            >>> pipeline.fit_nlsq('maxwell').fit_bayesian(num_samples=2000)
            >>> # With custom NUTS parameters
            >>> pipeline.fit_bayesian(
            ...     num_samples=3000,
            ...     num_warmup=1500,
            ...     target_accept_prob=0.9
            ... )
        """
        if self._last_model is None:
            raise ValueError(
                "No model fitted. Call fit_nlsq() first to fit a model "
                "before running Bayesian inference."
            )

        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Get data
        X = self.data.x
        y = self.data.y

        # Convert to numpy
        if isinstance(X, jnp.ndarray):
            X = np.array(X)
        if isinstance(y, jnp.ndarray):
            y = np.array(y)

        # Extract initial values from NLSQ fit for warm-start
        initial_values = None
        if self._last_model.fitted_:
            initial_values = {
                name: self._last_model.parameters.get_value(name)
                for name in self._last_model.parameters
            }

        # Run Bayesian inference
        result = self._last_model.fit_bayesian(
            X,
            y,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=1,  # Single chain for now
            initial_values=initial_values,
            **nuts_kwargs,
        )

        # Store results
        self._bayesian_result = result
        self._diagnostics = result.diagnostics

        # Add to history
        self.history.append(
            (
                "fit_bayesian",
                num_samples,
                num_warmup,
                result.diagnostics.get("divergences", 0),
            )
        )

        return self

    def get_diagnostics(self) -> dict[str, Any]:
        """Get convergence diagnostics from Bayesian inference.

        Returns diagnostics including R-hat (Gelman-Rubin statistic),
        effective sample size (ESS), and number of divergent transitions.

        Returns:
            Dictionary with diagnostic information:
                - r_hat: R-hat for each parameter (dict)
                - ess: Effective sample size for each parameter (dict)
                - divergences: Number of divergent transitions (int)

        Raises:
            ValueError: If Bayesian inference has not been run

        Example:
            >>> diagnostics = pipeline.get_diagnostics()
            >>> print(f"R-hat: {diagnostics['r_hat']}")
            >>> print(f"ESS: {diagnostics['ess']}")
            >>> print(f"Divergences: {diagnostics['divergences']}")
        """
        if self._bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        return self._diagnostics

    def get_posterior_summary(self) -> pd.DataFrame:
        """Get formatted posterior summary statistics.

        Returns a pandas DataFrame with summary statistics for each
        parameter including mean, standard deviation, median, and
        quantiles (5%, 25%, 75%, 95%).

        Returns:
            DataFrame with parameters as rows and statistics as columns

        Raises:
            ValueError: If Bayesian inference has not been run

        Example:
            >>> summary = pipeline.get_posterior_summary()
            >>> print(summary)
                     mean       std    median       q05       q25       q75       q95
            a    5.123   0.245     5.110     4.721     4.962     5.285     5.531
            b    0.487   0.032     0.485     0.435     0.465     0.509     0.542
        """
        if self._bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        # Convert summary dict to DataFrame
        summary_data = {}
        for param_name, stats in self._bayesian_result.summary.items():
            summary_data[param_name] = stats

        df = pd.DataFrame(summary_data).T
        return df

    def plot_posterior(
        self, param_name: str | None = None, **plot_kwargs
    ) -> BayesianPipeline:
        """Plot posterior distributions.

        Generates histogram plots of posterior distributions for model
        parameters. If param_name is None, plots all parameters in
        separate subplots.

        Args:
            param_name: Name of specific parameter to plot. If None,
                plots all parameters (default: None)
            **plot_kwargs: Additional arguments passed to matplotlib
                (e.g., bins, alpha, color)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run

        Example:
            >>> # Plot all parameters
            >>> pipeline.plot_posterior()
            >>> # Plot specific parameter
            >>> pipeline.plot_posterior('a', bins=50, alpha=0.7)
        """
        if self._bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        import matplotlib.pyplot as plt

        posterior_samples = self._bayesian_result.posterior_samples

        # Determine which parameters to plot
        if param_name is not None:
            if param_name not in posterior_samples:
                raise ValueError(
                    f"Parameter '{param_name}' not found in posterior samples. "
                    f"Available parameters: {list(posterior_samples.keys())}"
                )
            params_to_plot = [param_name]
        else:
            params_to_plot = list(posterior_samples.keys())

        # Create subplots
        n_params = len(params_to_plot)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle single parameter case
        if n_params == 1:
            axes = np.array([axes])

        axes_flat = axes.flatten() if n_params > 1 else axes

        # Plot each parameter
        for idx, param in enumerate(params_to_plot):
            ax = axes_flat[idx]
            samples = posterior_samples[param]

            # Plot histogram
            bins = plot_kwargs.pop("bins", 30)
            alpha = plot_kwargs.pop("alpha", 0.7)
            ax.hist(samples, bins=bins, alpha=alpha, **plot_kwargs)

            # Add summary statistics
            mean = self._bayesian_result.summary[param]["mean"]
            median = self._bayesian_result.summary[param]["median"]

            ax.axvline(mean, color="red", linestyle="--", linewidth=2, label="Mean")
            ax.axvline(
                median, color="blue", linestyle="--", linewidth=2, label="Median"
            )

            ax.set_xlabel(f"{param}")
            ax.set_ylabel("Frequency")
            ax.set_title(f"Posterior: {param}")
            ax.legend()
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for idx in range(n_params, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

        self.history.append(("plot_posterior", param_name if param_name else "all"))
        return self

    def plot_trace(
        self, param_name: str | None = None, **plot_kwargs
    ) -> BayesianPipeline:
        """Plot MCMC trace plots.

        Generates trace plots showing parameter values across MCMC iterations.
        Useful for diagnosing convergence issues. If param_name is None,
        plots all parameters.

        Args:
            param_name: Name of specific parameter to plot. If None,
                plots all parameters (default: None)
            **plot_kwargs: Additional arguments passed to matplotlib
                (e.g., alpha, linewidth)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run

        Example:
            >>> # Plot all trace plots
            >>> pipeline.plot_trace()
            >>> # Plot specific parameter
            >>> pipeline.plot_trace('a', alpha=0.5)
        """
        if self._bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        import matplotlib.pyplot as plt

        posterior_samples = self._bayesian_result.posterior_samples

        # Determine which parameters to plot
        if param_name is not None:
            if param_name not in posterior_samples:
                raise ValueError(
                    f"Parameter '{param_name}' not found in posterior samples. "
                    f"Available parameters: {list(posterior_samples.keys())}"
                )
            params_to_plot = [param_name]
        else:
            params_to_plot = list(posterior_samples.keys())

        # Create subplots
        n_params = len(params_to_plot)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 3 * n_params))

        # Handle single parameter case
        if n_params == 1:
            axes = [axes]

        # Plot each parameter
        for idx, param in enumerate(params_to_plot):
            ax = axes[idx]
            samples = posterior_samples[param]

            # Plot trace
            alpha = plot_kwargs.pop("alpha", 0.7)
            ax.plot(samples, alpha=alpha, **plot_kwargs)

            # Add mean line
            mean = self._bayesian_result.summary[param]["mean"]
            ax.axhline(mean, color="red", linestyle="--", linewidth=2, label="Mean")

            ax.set_xlabel("Iteration")
            ax.set_ylabel(f"{param}")
            ax.set_title(f"Trace: {param}")
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

        self.history.append(("plot_trace", param_name if param_name else "all"))
        return self

    def reset(self) -> BayesianPipeline:
        """Reset pipeline to initial state.

        Clears all data, models, and results including NLSQ and Bayesian
        inference results.

        Returns:
            self for method chaining

        Example:
            >>> pipeline.reset()
        """
        super().reset()
        self._nlsq_result = None
        self._bayesian_result = None
        self._diagnostics = None
        return self

    def __repr__(self) -> str:
        """String representation of Bayesian pipeline."""
        n_steps = len(self.history)
        has_data = self.data is not None
        has_model = self._last_model is not None
        has_nlsq = self._nlsq_result is not None
        has_bayesian = self._bayesian_result is not None

        return (
            f"BayesianPipeline(steps={n_steps}, "
            f"has_data={has_data}, "
            f"has_model={has_model}, "
            f"has_nlsq={has_nlsq}, "
            f"has_bayesian={has_bayesian})"
        )


__all__ = ["BayesianPipeline"]
