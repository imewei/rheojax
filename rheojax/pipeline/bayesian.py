"""Specialized pipeline for Bayesian workflows.

This module provides the BayesianPipeline class for orchestrating the complete
NLSQ → NumPyro NUTS workflow with a fluent API.

Example:
    >>> from rheojax.pipeline.bayesian import BayesianPipeline
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

from rheojax.core.arviz_utils import import_arviz
from rheojax.core.base import BaseModel
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.pipeline.base import Pipeline

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
            >>> from rheojax.models import Maxwell
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
        self,
        num_samples: int = 2000,
        num_warmup: int = 1000,
        num_chains: int = 4,
        **nuts_kwargs,
    ) -> BayesianPipeline:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        This method runs NUTS (No-U-Turn Sampler) for Bayesian parameter
        estimation. If a model has been previously fitted with fit_nlsq(),
        the NLSQ point estimates are automatically used for warm-starting
        the sampler, leading to faster convergence.

        Multi-chain sampling is enabled by default (num_chains=4) to provide
        reliable convergence diagnostics (R-hat, ESS) and parallel execution
        on multi-GPU systems.

        Args:
            num_samples: Number of posterior samples per chain (default: 2000)
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            num_chains: Number of MCMC chains (default: 4). Multiple chains
                enable proper R-hat computation and parallel execution.
                Chain method is auto-selected: 'parallel' on multi-GPU,
                'vectorized' on single GPU/CPU.
            **nuts_kwargs: Additional arguments passed to NUTS sampler
                (e.g., target_accept_prob, chain_method)

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
            ...     num_chains=4,
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

        # Get test_mode from data metadata if available
        test_mode = None
        if hasattr(self.data, "metadata") and self.data.metadata is not None:
            test_mode = self.data.metadata.get("test_mode")

        # Run Bayesian inference with multi-chain parallelization
        result = self._last_model.fit_bayesian(
            X,
            y,
            test_mode=test_mode,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
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
        self, param_name: str | None = None, show: bool = True, **plot_kwargs
    ) -> BayesianPipeline:
        """Plot posterior distributions.

        Generates histogram plots of posterior distributions for model
        parameters. If param_name is None, plots all parameters in
        separate subplots.

        Args:
            param_name: Name of specific parameter to plot. If None,
                plots all parameters (default: None)
            show: Whether to call plt.show() (default: True)
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
            >>> # Plot without showing (for save_figure)
            >>> pipeline.plot_posterior(show=False).save_figure('posterior.pdf')
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

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_posterior", param_name if param_name else "all"))
        return self

    def plot_trace(
        self, param_name: str | None = None, show: bool = True, **plot_kwargs
    ) -> BayesianPipeline:
        """Plot MCMC trace plots.

        Generates trace plots showing parameter values across MCMC iterations.
        Useful for diagnosing convergence issues. If param_name is None,
        plots all parameters.

        Args:
            param_name: Name of specific parameter to plot. If None,
                plots all parameters (default: None)
            show: Whether to call plt.show() (default: True)
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
            >>> # Plot without showing (for save_figure)
            >>> pipeline.plot_trace(show=False).save_figure('trace.pdf')
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

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_trace", param_name if param_name else "all"))
        return self

    def _get_inference_data(self) -> Any:
        """Get or create ArviZ InferenceData from Bayesian result.

        Helper method that retrieves the InferenceData object from the
        BayesianResult, converting it on first access. The InferenceData
        is cached for subsequent calls.

        Returns:
            ArviZ InferenceData object

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> idata = pipeline._get_inference_data()
        """
        if self._bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        return self._bayesian_result.to_inference_data()

    def plot_pair(
        self,
        var_names: list[str] | None = None,
        kind: str = "scatter",
        divergences: bool = True,
        show: bool = True,
        **plot_kwargs,
    ) -> BayesianPipeline:
        """Plot pairwise relationships between parameters (pair plot).

        Creates a matrix of scatter or KDE plots showing correlations between
        parameters. This is critical for identifying parameter dependencies,
        non-identifiability issues, and understanding the joint posterior
        structure. Divergent transitions are highlighted by default to identify
        problematic posterior geometry.

        Args:
            var_names: List of parameter names to plot. If None, plots all
                parameters (default: None)
            kind: Type of pair plot - "scatter", "kde", or "hexbin"
                (default: "scatter")
            divergences: Whether to highlight divergent transitions in red
                (default: True). Useful for identifying problematic regions.
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_pair()
                (e.g., marginals, point_estimate_marker_style)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot all parameters with divergences highlighted
            >>> pipeline.plot_pair()
            >>>
            >>> # Plot specific parameters as KDE
            >>> pipeline.plot_pair(var_names=["G0", "eta"], kind="kde")
            >>>
            >>> # Save without showing
            >>> pipeline.plot_pair(show=False).save_figure("pair.pdf")

        Note:
            Pair plots are essential for diagnosing:
            - Parameter correlations (indicates non-identifiability)
            - Funnel geometry (divergences concentrated in specific regions)
            - Multimodal posteriors (multiple clusters)
        """
        try:
            az = import_arviz(required=("plot_pair",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for pair plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create pair plot
        axes = az.plot_pair(
            idata,
            var_names=var_names,
            kind=kind,
            divergences=divergences,
            **plot_kwargs,
        )

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif hasattr(axes, "ravel"):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_pair", var_names if var_names else "all"))
        return self

    def plot_forest(
        self,
        var_names: list[str] | None = None,
        combined: bool = True,
        hdi_prob: float = 0.95,
        show: bool = True,
        **plot_kwargs,
    ) -> BayesianPipeline:
        """Plot forest plot with credible intervals for parameters.

        Creates a forest plot showing parameter estimates with credible intervals
        (highest density intervals). Excellent for comparing parameter magnitudes
        and uncertainties at a glance. Each parameter is shown as a point estimate
        with error bars representing the credible interval.

        Args:
            var_names: List of parameter names to plot. If None, plots all
                parameters (default: None)
            combined: Whether to combine multiple chains (default: True)
            hdi_prob: Probability mass for credible interval (default: 0.95).
                Common values: 0.68 (1σ), 0.95 (2σ), 0.997 (3σ)
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_forest()
                (e.g., rope, ref_val, colors)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot all parameters with 95% CI
            >>> pipeline.plot_forest()
            >>>
            >>> # Plot specific parameters with 68% CI
            >>> pipeline.plot_forest(var_names=["G0", "eta"], hdi_prob=0.68)
            >>>
            >>> # Save without showing
            >>> pipeline.plot_forest(show=False).save_figure("forest.pdf")

        Note:
            Forest plots are useful for:
            - Quickly comparing parameter magnitudes
            - Assessing parameter uncertainty
            - Identifying parameters with poor estimation (wide intervals)
        """
        try:
            az = import_arviz(required=("plot_forest",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for forest plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create forest plot
        axes = az.plot_forest(
            idata,
            var_names=var_names,
            combined=combined,
            hdi_prob=hdi_prob,
            **plot_kwargs,
        )

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_forest", var_names if var_names else "all"))
        return self

    def plot_energy(self, show: bool = True, **plot_kwargs) -> BayesianPipeline:
        """Plot NUTS energy diagnostic plot.

        Creates an energy plot showing the distribution of energy transitions
        during NUTS sampling. This is a NUTS-specific diagnostic that helps
        identify problematic posterior geometry such as heavy tails, funnels,
        or multimodal distributions. Energy transitions that differ between
        the marginal and transition distributions indicate sampling problems.

        Args:
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_energy()

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot energy diagnostic
            >>> pipeline.plot_energy()
            >>>
            >>> # Save without showing
            >>> pipeline.plot_energy(show=False).save_figure("energy.pdf")

        Note:
            Energy diagnostics help identify:
            - Heavy-tailed posteriors (energy dist has fat tails)
            - Funnel geometry (energy varies dramatically)
            - Problematic parameterizations
            Good NUTS sampling shows similar marginal and transition energy distributions.
        """
        try:
            az = import_arviz(required=("plot_energy",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for energy plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create energy plot
        axes = az.plot_energy(idata, **plot_kwargs)

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_energy", None))
        return self

    def plot_autocorr(
        self,
        var_names: list[str] | None = None,
        combined: bool = False,
        show: bool = True,
        **plot_kwargs,
    ) -> BayesianPipeline:
        """Plot autocorrelation diagnostic for MCMC mixing.

        Creates autocorrelation plots showing how correlated consecutive samples
        are in the MCMC chain. High autocorrelation indicates poor mixing and
        suggests more samples are needed for reliable inference. Ideally,
        autocorrelation should decay quickly to zero.

        Args:
            var_names: List of parameter names to plot. If None, plots all
                parameters (default: None)
            combined: Whether to combine multiple chains (default: False)
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_autocorr()
                (e.g., max_lag)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot autocorrelation for all parameters
            >>> pipeline.plot_autocorr()
            >>>
            >>> # Plot specific parameters with longer lag
            >>> pipeline.plot_autocorr(var_names=["G0"], max_lag=100)
            >>>
            >>> # Save without showing
            >>> pipeline.plot_autocorr(show=False).save_figure("autocorr.pdf")

        Note:
            Autocorrelation diagnostics help identify:
            - Poor mixing (high autocorrelation persists)
            - Need for more samples (ESS will be low)
            - Chain length adequacy
            Goal: autocorrelation drops to ~0 within a few dozen lags.
        """
        try:
            az = import_arviz(required=("plot_autocorr",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for autocorrelation plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create autocorrelation plot
        axes = az.plot_autocorr(
            idata,
            var_names=var_names,
            combined=combined,
            **plot_kwargs,
        )

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_autocorr", var_names if var_names else "all"))
        return self

    def plot_rank(
        self,
        var_names: list[str] | None = None,
        show: bool = True,
        **plot_kwargs,
    ) -> BayesianPipeline:
        """Plot rank plot for convergence diagnostics.

        Creates rank plots (also called rank histograms or rank-normalization
        plots) which are a modern alternative to trace plots for diagnosing
        convergence. A uniform rank distribution across chains indicates good
        mixing and convergence. Non-uniformity suggests convergence problems.

        Args:
            var_names: List of parameter names to plot. If None, plots all
                parameters (default: None)
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_rank()

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot rank diagnostic for all parameters
            >>> pipeline.plot_rank()
            >>>
            >>> # Plot specific parameters
            >>> pipeline.plot_rank(var_names=["G0", "eta"])
            >>>
            >>> # Save without showing
            >>> pipeline.plot_rank(show=False).save_figure("rank.pdf")

        Note:
            Rank plots help identify:
            - Non-convergence (non-uniform rank distribution)
            - Chain sticking (vertical bands)
            - Insufficient mixing (patterns in ranks)
            Goal: uniform histogram across all bins.
        """
        try:
            az = import_arviz(required=("plot_rank",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for rank plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create rank plot
        axes = az.plot_rank(
            idata,
            var_names=var_names,
            **plot_kwargs,
        )

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_rank", var_names if var_names else "all"))
        return self

    def plot_ess(
        self,
        var_names: list[str] | None = None,
        kind: str = "local",
        show: bool = True,
        **plot_kwargs,
    ) -> BayesianPipeline:
        """Plot effective sample size (ESS) diagnostic.

        Creates a plot showing the effective sample size for each parameter,
        which quantifies how many independent samples the MCMC chain is
        equivalent to. Low ESS indicates high autocorrelation and suggests
        more samples are needed. ESS values should ideally be > 400.

        Args:
            var_names: List of parameter names to plot. If None, plots all
                parameters (default: None)
            kind: Type of ESS plot - "local", "quantile", or "evolution"
                (default: "local")
            show: Whether to call plt.show() (default: True)
            **plot_kwargs: Additional arguments passed to arviz.plot_ess()
                (e.g., min_ess)

        Returns:
            self for method chaining

        Raises:
            ValueError: If Bayesian inference has not been run
            ImportError: If arviz is not installed

        Example:
            >>> # Plot ESS for all parameters
            >>> pipeline.plot_ess()
            >>>
            >>> # Plot quantile ESS
            >>> pipeline.plot_ess(kind="quantile")
            >>>
            >>> # Save without showing
            >>> pipeline.plot_ess(show=False).save_figure("ess.pdf")

        Note:
            ESS diagnostics help assess:
            - Sampling efficiency (ESS / total samples)
            - Which parameters need more sampling
            - Overall chain quality
            Goal: ESS > 400 for bulk and tail estimates.
        """
        try:
            az = import_arviz(required=("plot_ess",))
        except ImportError as exc:
            raise ImportError(
                "ArviZ is required for ESS plots. Install it with: pip install arviz"
            ) from exc

        # Get InferenceData
        idata = self._get_inference_data()

        # Create ESS plot
        axes = az.plot_ess(
            idata,
            var_names=var_names,
            kind=kind,
            **plot_kwargs,
        )

        # Extract figure from axes
        import matplotlib.pyplot as plt

        if hasattr(axes, "figure"):
            fig = axes.figure
        elif isinstance(axes, np.ndarray):
            fig = axes.ravel()[0].figure
        else:
            fig = plt.gcf()

        # Store figure for save_figure() method
        self._current_figure = fig

        if show:
            plt.show()

        self.history.append(("plot_ess", var_names if var_names else "all"))
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
