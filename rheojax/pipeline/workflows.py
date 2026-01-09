"""Specialized pipeline classes for common rheological workflows.

This module provides pre-configured pipelines for standard analysis workflows
like mastercurve construction, model comparison, and data conversion.

Example:
    >>> from rheojax.pipeline.workflows import ModelComparisonPipeline
    >>> pipeline = ModelComparisonPipeline(['maxwell', 'kelvin_voigt', 'zener'])
    >>> pipeline.run(data)
    >>> best = pipeline.get_best_model()
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.logging import get_logger
from rheojax.pipeline.base import Pipeline

if TYPE_CHECKING:
    from rheojax.models.spp_yield_stress import SPPYieldStress

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Module-level logger
logger = get_logger(__name__)


class MastercurvePipeline(Pipeline):
    """Pipeline for time-temperature superposition analysis.

    This pipeline automates the construction of mastercurves from
    multi-temperature rheological data using horizontal shift factors.

    Attributes:
        reference_temp: Reference temperature for mastercurve
        shift_factors: Dictionary of temperature -> shift factor

    Example:
        >>> pipeline = MastercurvePipeline(reference_temp=298.15)
        >>> pipeline.run(file_paths, temperatures)
        >>> mastercurve = pipeline.get_result()
    """

    def __init__(self, reference_temp: float = 298.15):
        """Initialize mastercurve pipeline.

        Args:
            reference_temp: Reference temperature in Kelvin (default: 298.15 K)
        """
        super().__init__()
        self.reference_temp = reference_temp
        self.shift_factors: dict[float, float] = {}

    def run(
        self,
        file_paths: list[str],
        temperatures: list[float],
        format: str = "auto",
        **load_kwargs,
    ) -> MastercurvePipeline:
        """Execute mastercurve workflow.

        Args:
            file_paths: List of data file paths (one per temperature)
            temperatures: List of temperatures (in Kelvin)
            format: File format for loading
            **load_kwargs: Additional arguments passed to load (e.g., x_col, y_col)

        Returns:
            self for method chaining

        Raises:
            ValueError: If file_paths and temperatures have different lengths
        """
        if len(file_paths) != len(temperatures):
            raise ValueError(
                f"Number of files ({len(file_paths)}) must match "
                f"number of temperatures ({len(temperatures)})"
            )

        logger.info(
            "Starting mastercurve construction",
            n_datasets=len(file_paths),
            reference_temp=self.reference_temp,
        )
        start_time = time.perf_counter()

        # Load all datasets
        datasets = []
        for i, file_path in enumerate(file_paths):
            dataset_start = time.perf_counter()
            try:
                temp_pipeline = Pipeline()
                temp_pipeline.load(file_path, format=format, **load_kwargs)
                datasets.append(temp_pipeline.get_result())
                dataset_elapsed = time.perf_counter() - dataset_start
                logger.debug(
                    "Dataset loaded",
                    dataset=i,
                    file_path=file_path,
                    temperature=temperatures[i],
                    elapsed=dataset_elapsed,
                )
            except Exception as e:
                logger.error(
                    "Failed to load dataset",
                    dataset=i,
                    file_path=file_path,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Merge datasets with temperature metadata
        merged_data = self._merge_datasets(datasets, temperatures)

        # Apply mastercurve transform if available
        # For now, we'll implement a simple version
        self.data = merged_data
        self._apply_mastercurve_shift()

        self.history.append(
            ("mastercurve", str(len(file_paths)), str(self.reference_temp))
        )

        total_time = time.perf_counter() - start_time
        logger.info(
            "Mastercurve construction complete",
            n_datasets=len(file_paths),
            n_shift_factors=len(self.shift_factors),
            total_time=total_time,
        )
        return self

    def _merge_datasets(
        self, datasets: list[RheoData], temperatures: list[float]
    ) -> RheoData:
        """Merge multiple datasets with temperature metadata.

        Args:
            datasets: List of RheoData objects
            temperatures: Corresponding temperatures

        Returns:
            Merged RheoData
        """
        # Add temperature metadata to each dataset
        for data, temp in zip(datasets, temperatures, strict=False):
            data.metadata["temperature"] = temp

        # For simplicity, concatenate all data
        # In practice, this would be more sophisticated
        all_x = np.concatenate([np.array(d.x) for d in datasets])
        all_y = np.concatenate([np.array(d.y) for d in datasets])
        all_temps = np.concatenate(
            [
                np.full(len(d.x), temp)
                for d, temp in zip(datasets, temperatures, strict=False)
            ]
        )

        return RheoData(
            x=all_x,
            y=all_y,
            x_units=datasets[0].x_units,
            y_units=datasets[0].y_units,
            domain=datasets[0].domain,
            metadata={
                "type": "mastercurve",
                "reference_temp": self.reference_temp,
                "temperatures": all_temps.tolist(),
            },
            validate=False,
        )

    def _apply_mastercurve_shift(self):
        """Apply horizontal shift to create mastercurve.

        This implements a simplified WLF-based shift.
        In production, this would use the mastercurve transform.
        """
        if self.data is None:
            return

        temps = np.array(self.data.metadata.get("temperatures", []))
        if len(temps) == 0:
            return

        # Calculate shift factors using simplified WLF equation
        # log(a_T) = -C1(T - Tref) / (C2 + T - Tref)
        # Using typical values: C1=17.44, C2=51.6
        C1, C2 = 17.44, 51.6

        for temp in np.unique(temps):
            if temp == self.reference_temp:
                shift = 1.0
            else:
                log_shift = (
                    -C1
                    * (temp - self.reference_temp)
                    / (C2 + temp - self.reference_temp)
                )
                shift = 10**log_shift

            self.shift_factors[float(temp)] = shift

        # Apply shifts to x data
        shifted_x = self.data.x.copy()
        for i, temp in enumerate(temps):
            shift = self.shift_factors[float(temp)]
            shifted_x = (
                shifted_x.at[i].set(shifted_x[i] / shift)
                if isinstance(shifted_x, jnp.ndarray)
                else shifted_x
            )
            if isinstance(shifted_x, np.ndarray):
                shifted_x[i] = shifted_x[i] / shift

        self.data.x = shifted_x

    def get_shift_factors(self) -> dict[float, float]:
        """Get computed shift factors.

        Returns:
            Dictionary mapping temperature to shift factor
        """
        return self.shift_factors.copy()


class ModelComparisonPipeline(Pipeline):
    """Pipeline for comparing multiple models on the same data.

    This pipeline fits multiple models to the same dataset and
    computes comparison metrics (RMSE, R², AIC, etc.).

    Attributes:
        models: List of model names to compare
        results: Dictionary of model_name -> metrics

    Example:
        >>> pipeline = ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
        >>> pipeline.run(data)
        >>> best = pipeline.get_best_model()
        >>> print(pipeline.get_comparison_table())
    """

    def __init__(self, models: list[str]):
        """Initialize model comparison pipeline.

        Args:
            models: List of model names to compare
        """
        super().__init__()
        self.models = models
        self.results: dict[str, dict[str, Any]] = {}

    def run(self, data: RheoData, **fit_kwargs) -> ModelComparisonPipeline:
        """Fit multiple models and compare.

        Args:
            data: RheoData to fit
            **fit_kwargs: Additional arguments passed to fit

        Returns:
            self for method chaining
        """
        self.data = data
        X = np.array(data.x)
        y = np.array(data.y)

        logger.info(
            "Starting model comparison",
            n_models=len(self.models),
            data_shape=X.shape,
        )
        start_time = time.perf_counter()

        for model_name in self.models:
            model_start = time.perf_counter()
            try:
                # Create and fit model
                model = ModelRegistry.create(model_name)
                model.fit(X, y, **fit_kwargs)

                # Generate predictions
                y_pred = model.predict(X)

                # Handle complex modulus (oscillation mode)
                # Case 1: Complex predictions (G* = G' + iG")
                if np.iscomplexobj(y_pred):
                    y_pred_magnitude = np.abs(y_pred)
                # Case 2: 2D array [G', G"] format
                elif y_pred.ndim == 2 and y_pred.shape[1] == 2:
                    y_pred_magnitude = np.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2)
                # Case 3: Real predictions
                else:
                    y_pred_magnitude = y_pred

                # Calculate metrics using magnitude (real values)
                residuals = y - y_pred_magnitude

                # Try to use NLSQ result properties (NLSQ 0.6.0 CurveFitResult compatible)
                # Falls back to manual computation if result not available
                nlsq_result = (
                    model.get_nlsq_result()
                    if hasattr(model, "get_nlsq_result")
                    else None
                )

                if nlsq_result is not None and nlsq_result.rmse is not None:
                    # Use NLSQ 0.6.0 CurveFitResult-compatible properties
                    rmse = nlsq_result.rmse
                    r_squared = nlsq_result.r_squared or 0.0
                    aic = nlsq_result.aic
                    bic = nlsq_result.bic
                else:
                    # Fallback: Calculate metrics manually
                    rmse = np.sqrt(np.mean(residuals**2))

                    # Calculate R² manually (avoid calling model.score())
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                    # Calculate AIC/BIC manually
                    n = len(y)
                    k = len(model.parameters) if hasattr(model, "parameters") else 0
                    if n > 0 and rmse > 0:
                        rss = np.sum(residuals**2)
                        aic = 2 * k + n * np.log(rss / n)
                        bic = k * np.log(n) + n * np.log(rss / n)
                    else:
                        aic = np.inf
                        bic = np.inf

                # Calculate relative RMSE
                rel_rmse = rmse / np.mean(np.abs(y))

                # Store results
                n_params = len(model.parameters) if hasattr(model, "parameters") else 0
                self.results[model_name] = {
                    "model": model,
                    "parameters": model.get_params(),
                    "predictions": y_pred_magnitude,  # Always real-valued, plottable magnitudes
                    "residuals": residuals,
                    "rmse": float(rmse),
                    "rel_rmse": float(rel_rmse),
                    "r_squared": float(r_squared),
                    "n_params": n_params,
                    "aic": float(aic) if aic is not None else np.inf,
                    "bic": float(bic) if bic is not None else np.inf,
                }

                self.history.append(("fit_compare", model_name, str(r_squared)))

                model_elapsed = time.perf_counter() - model_start
                logger.debug(
                    "Model fitted",
                    model=model_name,
                    r_squared=float(r_squared),
                    rmse=float(rmse),
                    elapsed=model_elapsed,
                )

            except Exception as e:
                logger.error(
                    "Failed to fit model",
                    model=model_name,
                    error=str(e),
                    exc_info=True,
                )
                warnings.warn(f"Failed to fit model {model_name}: {e}", stacklevel=2)
                continue

        total_time = time.perf_counter() - start_time
        logger.info(
            "Model comparison complete",
            n_models=len(self.models),
            n_successful=len(self.results),
            total_time=total_time,
        )
        return self

    def get_best_model(self, metric: str = "rmse", minimize: bool = True) -> str:
        """Return name of best-fitting model.

        Args:
            metric: Metric to use for comparison ('rmse', 'r_squared', 'aic', 'bic')
            minimize: If True, lower values are better (e.g., RMSE, AIC, BIC)

        Returns:
            Name of best model

        Example:
            >>> best = pipeline.get_best_model(metric='aic')
        """
        if not self.results:
            raise ValueError("No models fitted. Call run() first.")

        if minimize:
            return min(self.results.items(), key=lambda x: x[1].get(metric, np.inf))[0]
        else:
            return max(self.results.items(), key=lambda x: x[1].get(metric, -np.inf))[0]

    def get_comparison_table(self) -> dict[str, dict[str, float]]:
        """Get comparison table of all models.

        Returns:
            Dictionary of model_name -> metrics

        Example:
            >>> table = pipeline.get_comparison_table()
            >>> for model, metrics in table.items():
            ...     print(f"{model}: R²={metrics['r_squared']:.4f}")
        """
        return {
            name: {
                "rmse": result["rmse"],
                "rel_rmse": result["rel_rmse"],
                "r_squared": result["r_squared"],
                "aic": result.get("aic", np.nan),
                "bic": result.get("bic", np.nan),
                "n_params": result["n_params"],
            }
            for name, result in self.results.items()
        }

    def get_model_result(self, model_name: str) -> dict[str, Any]:
        """Get detailed results for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model, parameters, and metrics

        Example:
            >>> result = pipeline.get_model_result('maxwell')
            >>> params = result['parameters']
        """
        if model_name not in self.results:
            raise KeyError(f"Model {model_name} not in results")
        return self.results[model_name]


class CreepToRelaxationPipeline(Pipeline):
    """Convert creep compliance data to relaxation modulus.

    This pipeline performs the numerical conversion from J(t) to G(t)
    using regularized numerical inversion techniques.

    Example:
        >>> pipeline = CreepToRelaxationPipeline()
        >>> pipeline.run(creep_data)
        >>> relaxation_data = pipeline.get_result()
    """

    def run(
        self, creep_data: RheoData, method: str = "approximate"
    ) -> CreepToRelaxationPipeline:
        """Execute conversion workflow.

        Args:
            creep_data: RheoData with creep compliance J(t)
            method: Conversion method ('approximate', 'exact')

        Returns:
            self for method chaining

        Raises:
            ValueError: If input is not creep data
        """
        self.data = creep_data

        logger.info(
            "Starting creep to relaxation conversion",
            method=method,
            data_points=len(creep_data.x),
        )
        start_time = time.perf_counter()

        # Validate test mode
        test_mode = creep_data.metadata.get("test_mode", "").lower()
        if test_mode and test_mode != "creep":
            warnings.warn(
                f"Input appears to be {test_mode} data, not creep. "
                "Results may be inaccurate.",
                stacklevel=2,
            )

        try:
            if method == "approximate":
                self._approximate_conversion()
            elif method == "exact":
                self._exact_conversion()
            else:
                raise ValueError(f"Unknown method: {method}")

            self.history.append(("creep_to_relaxation", method))

            total_time = time.perf_counter() - start_time
            logger.info(
                "Creep to relaxation conversion complete",
                method=method,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(
                "Creep to relaxation conversion failed",
                method=method,
                error=str(e),
                exc_info=True,
            )
            raise

        return self

    def _approximate_conversion(self):
        """Apply approximate conversion G(t) ≈ 1/J(t).

        This is valid for small strains and elastic-dominant materials.
        """
        if self.data is None:
            return

        J_t = np.array(self.data.y)

        # Avoid division by zero
        J_t = np.maximum(J_t, 1e-20)

        G_t = 1.0 / J_t

        self.data = RheoData(
            x=self.data.x,
            y=G_t,
            x_units=self.data.x_units,
            y_units="Pa" if not self.data.y_units else self.data.y_units,
            domain=self.data.domain,
            metadata={
                **self.data.metadata,
                "test_mode": "relaxation",
                "conversion_method": "approximate",
            },
            validate=False,
        )

    def _exact_conversion(self):
        """Apply exact conversion using Laplace transform inversion.

        This is more accurate but computationally intensive.
        For now, we use a simplified numerical approach.
        """
        if self.data is None:
            return

        # This would use a proper Laplace transform inversion
        # For now, fall back to approximate
        warnings.warn(
            "Exact conversion not fully implemented. Using approximate method.",
            stacklevel=2,
        )
        self._approximate_conversion()
        self.data.metadata["conversion_method"] = "exact_approximate"


class FrequencyToTimePipeline(Pipeline):
    """Convert frequency domain data to time domain.

    This pipeline converts dynamic modulus G*(ω) to relaxation modulus G(t)
    using Fourier transform techniques.

    Example:
        >>> pipeline = FrequencyToTimePipeline()
        >>> pipeline.run(frequency_data)
        >>> time_data = pipeline.get_result()
    """

    def run(
        self,
        frequency_data: RheoData,
        time_range: tuple | None = None,
        n_points: int = 100,
    ) -> FrequencyToTimePipeline:
        """Execute frequency to time conversion.

        Args:
            frequency_data: RheoData in frequency domain
            time_range: Optional (t_min, t_max) for time range
            n_points: Number of time points to generate

        Returns:
            self for method chaining
        """
        self.data = frequency_data

        logger.info(
            "Starting frequency to time conversion",
            n_points=n_points,
            input_points=len(frequency_data.x),
        )
        start_time = time.perf_counter()

        if frequency_data.domain != "frequency":
            warnings.warn("Input data may not be in frequency domain", stacklevel=2)

        try:
            # Generate time points
            if time_range is None:
                # Auto-generate from frequency range
                w_min = np.min(np.array(frequency_data.x))
                w_max = np.max(np.array(frequency_data.x))
                t_min = 1.0 / w_max
                t_max = 1.0 / w_min
            else:
                t_min, t_max = time_range

            t = np.logspace(np.log10(t_min), np.log10(t_max), n_points)

            # Simplified conversion using inverse Fourier transform approximation
            # In practice, this would use proper numerical FFT
            omega = np.array(frequency_data.x)
            G_star = np.array(frequency_data.y)

            # Placeholder: proper implementation would use FFT
            # For now, use simple numerical integration
            G_t = self._approximate_inverse_transform(t, omega, G_star)

            self.data = RheoData(
                x=t,
                y=G_t,
                x_units="s",
                y_units=frequency_data.y_units,
                domain="time",
                metadata={
                    **frequency_data.metadata,
                    "conversion": "frequency_to_time",
                    "original_domain": "frequency",
                },
                validate=False,
            )

            self.history.append(("frequency_to_time", str(n_points)))

            total_time = time.perf_counter() - start_time
            logger.info(
                "Frequency to time conversion complete",
                n_points=n_points,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(
                "Frequency to time conversion failed",
                error=str(e),
                exc_info=True,
            )
            raise

        return self

    def _approximate_inverse_transform(
        self, t: np.ndarray, omega: np.ndarray, G_star: np.ndarray
    ) -> np.ndarray:
        """Inverse Fourier transform from G*(ω) to G(t).

        Uses numerical integration of the inverse Fourier transform:
        G(t) = (2/π) ∫ G'(ω) cos(ωt) dω

        Args:
            t: Time points
            omega: Angular frequency points
            G_star: Complex modulus (G' + iG'' or just G')

        Returns:
            Relaxation modulus at time points
        """
        from scipy.integrate import trapezoid

        # Extract real part (storage modulus G')
        if np.iscomplexobj(G_star):
            G_prime = np.real(G_star)
        elif G_star.ndim == 2 and G_star.shape[1] == 2:
            G_prime = G_star[:, 0]
        else:
            G_prime = G_star

        # Sort by frequency for proper integration
        sort_idx = np.argsort(omega)
        omega_sorted = omega[sort_idx]
        G_prime_sorted = G_prime[sort_idx]

        # Compute G(t) via numerical integration of inverse transform
        G_t = np.zeros_like(t)

        for i, t_i in enumerate(t):
            # G(t) = (2/π) ∫ G'(ω) cos(ωt) dω
            integrand = G_prime_sorted * np.cos(omega_sorted * t_i)
            G_t[i] = (2.0 / np.pi) * trapezoid(integrand, omega_sorted)

        # Ensure non-negative (physical constraint)
        G_t = np.maximum(G_t, 0.0)

        return G_t


class SPPAmplitudeSweepPipeline(Pipeline):
    """Pipeline for SPP analysis of amplitude sweep LAOS data.

    This pipeline performs SPP (Sequence of Physical Processes) analysis
    on amplitude sweep LAOS data to extract yield stress parameters and
    nonlinear viscoelastic metrics.

    Workflow:
        1. Load amplitude sweep data (multiple γ_0 values)
        2. Apply SPP decomposition at each amplitude
        3. Extract yield stresses (static and dynamic)
        4. Fit power-law scaling to yield stress vs amplitude
        5. Optionally fit Bayesian SPPYieldStress model

    Attributes:
        omega: Angular frequency of oscillation (rad/s)
        results: Dictionary of SPP metrics per amplitude
        model: Fitted SPPYieldStress model (after fit_model)

    Example:
        >>> pipeline = SPPAmplitudeSweepPipeline(omega=1.0)
        >>> pipeline.run(amplitude_data_list)
        >>> pipeline.fit_model(bayesian=True)
        >>> print(pipeline.get_yield_stresses())
    """

    def __init__(
        self,
        omega: float = 1.0,
        n_harmonics: int = 39,
        step_size: int = 8,
        num_mode: int = 2,
        wrap_strain_rate: bool = True,
        use_numerical_method: bool | None = None,
    ):
        """Initialize SPP amplitude sweep pipeline.

        Args:
            omega: Angular frequency in rad/s (default: 1.0)
            n_harmonics: Number of harmonics for SPP decomposition (default: 39)
            step_size: Differentiation step size k (default: 8, Rogers parity)
            num_mode: Numerical differentiation mode (default: 2 periodic)
            wrap_strain_rate: Whether to use wrapped differentiation when rate missing
            use_numerical_method: Force numerical path; None keeps default from transform
        """
        super().__init__()
        self.omega = omega
        self.n_harmonics = n_harmonics
        self.step_size = step_size
        self.num_mode = num_mode
        self.wrap_strain_rate = wrap_strain_rate
        self.use_numerical_method = use_numerical_method
        self.results: dict[float, dict] = {}  # gamma_0 -> SPP results
        self.model: SPPYieldStress | None = None
        self._gamma_0_values: list[float] = []
        self._sigma_sy_values: list[float] = []
        self._sigma_dy_values: list[float] = []

    def run(
        self,
        stress_data: list[RheoData],
        gamma_0_values: list[float] | None = None,
    ) -> SPPAmplitudeSweepPipeline:
        """Execute SPP analysis on amplitude sweep data.

        Args:
            stress_data: List of RheoData objects, one per amplitude
            gamma_0_values: Strain amplitudes corresponding to each dataset.
                           If None, extracted from RheoData metadata.

        Returns:
            self for method chaining

        Raises:
            ValueError: If gamma_0_values not provided and not in metadata
        """
        from rheojax.transforms.spp_decomposer import SPPDecomposer

        # Extract gamma_0 values if not provided
        if gamma_0_values is None:
            gamma_0_values = []
            for data in stress_data:
                if "gamma_0" in data.metadata:
                    gamma_0_values.append(data.metadata["gamma_0"])
                else:
                    raise ValueError(
                        "gamma_0_values must be provided or present in metadata"
                    )

        if len(stress_data) != len(gamma_0_values):
            raise ValueError(
                f"Number of datasets ({len(stress_data)}) must match "
                f"number of amplitudes ({len(gamma_0_values)})"
            )

        logger.info(
            "Starting SPP amplitude sweep analysis",
            n_datasets=len(stress_data),
            omega=self.omega,
        )
        start_time = time.perf_counter()
        n_successful = 0

        # Process each amplitude
        for i, (gamma_0, data) in enumerate(
            zip(gamma_0_values, stress_data, strict=False)
        ):
            amplitude_start = time.perf_counter()

            # Ensure required metadata is present for downstream transforms/models
            if data.metadata is None:
                data.metadata = {}
            data.metadata.setdefault("test_mode", "oscillation")
            data.metadata.setdefault("gamma_0", gamma_0)
            data.metadata.setdefault("omega", self.omega)

            # Apply SPP decomposition
            decomposer = SPPDecomposer(
                omega=self.omega,
                gamma_0=gamma_0,
                n_harmonics=self.n_harmonics,
                step_size=self.step_size,
                num_mode=self.num_mode,
                wrap_strain_rate=self.wrap_strain_rate,
                use_numerical_method=(
                    self.use_numerical_method
                    if self.use_numerical_method is not None
                    else False
                ),
            )

            try:
                decomposer.transform(data)
                results = decomposer.get_results()
                self.results[float(gamma_0)] = results

                self._gamma_0_values.append(float(gamma_0))
                self._sigma_sy_values.append(results["sigma_sy"])
                self._sigma_dy_values.append(results["sigma_dy"])

                self.history.append(("spp_analyze", str(gamma_0), "success"))
                n_successful += 1

                amplitude_elapsed = time.perf_counter() - amplitude_start
                logger.debug(
                    "SPP decomposition completed",
                    dataset=i,
                    gamma_0=gamma_0,
                    sigma_sy=results["sigma_sy"],
                    sigma_dy=results["sigma_dy"],
                    elapsed=amplitude_elapsed,
                )

            except Exception as e:
                logger.error(
                    "SPP analysis failed",
                    dataset=i,
                    gamma_0=gamma_0,
                    error=str(e),
                    exc_info=True,
                )
                warnings.warn(
                    f"SPP analysis failed at γ_0 = {gamma_0}: {e}", stacklevel=2
                )
                self.history.append(("spp_analyze", str(gamma_0), f"failed: {e}"))

        # Sort by amplitude
        sort_idx = np.argsort(self._gamma_0_values)
        self._gamma_0_values = [self._gamma_0_values[i] for i in sort_idx]
        self._sigma_sy_values = [self._sigma_sy_values[i] for i in sort_idx]
        self._sigma_dy_values = [self._sigma_dy_values[i] for i in sort_idx]

        total_time = time.perf_counter() - start_time
        logger.info(
            "SPP amplitude sweep analysis complete",
            n_datasets=len(stress_data),
            n_successful=n_successful,
            total_time=total_time,
        )

        return self

    def fit_model(
        self,
        bayesian: bool = False,
        yield_type: str = "static",
        **fit_kwargs,
    ) -> SPPAmplitudeSweepPipeline:
        """Fit SPPYieldStress model to extracted yield stresses.

        Args:
            bayesian: Whether to use Bayesian inference (default: False)
            yield_type: Which yield stress to fit ('static' or 'dynamic')
            **fit_kwargs: Additional arguments passed to fit or fit_bayesian

        Returns:
            self for method chaining
        """
        from rheojax.models.spp_yield_stress import SPPYieldStress

        if not self._gamma_0_values:
            raise RuntimeError("No data available. Call run() first.")

        logger.info(
            "Starting SPP model fitting",
            bayesian=bayesian,
            yield_type=yield_type,
            n_points=len(self._gamma_0_values),
        )
        start_time = time.perf_counter()

        gamma_0_array = np.array(self._gamma_0_values)
        if yield_type == "static":
            sigma_array = np.array(self._sigma_sy_values)
        else:
            sigma_array = np.array(self._sigma_dy_values)

        self.model = SPPYieldStress()

        try:
            if bayesian:
                self.model.fit_bayesian(
                    gamma_0_array,
                    sigma_array,
                    test_mode="oscillation",
                    **fit_kwargs,
                )
                self.history.append(("fit_bayesian", yield_type, "complete"))
            else:
                self.model.fit(
                    gamma_0_array,
                    sigma_array,
                    test_mode="oscillation",
                    yield_type=yield_type,
                    **fit_kwargs,
                )
                self.history.append(("fit_nlsq", yield_type, "complete"))

            total_time = time.perf_counter() - start_time
            logger.info(
                "SPP model fitting complete",
                bayesian=bayesian,
                yield_type=yield_type,
                total_time=total_time,
            )
        except Exception as e:
            logger.error(
                "SPP model fitting failed",
                bayesian=bayesian,
                yield_type=yield_type,
                error=str(e),
                exc_info=True,
            )
            raise

        return self

    def get_yield_stresses(self) -> dict[str, np.ndarray]:
        """Get extracted yield stresses from amplitude sweep.

        Returns:
            Dictionary with:
            - gamma_0: strain amplitudes
            - sigma_sy: static yield stresses
            - sigma_dy: dynamic yield stresses
        """
        return {
            "gamma_0": np.array(self._gamma_0_values),
            "sigma_sy": np.array(self._sigma_sy_values),
            "sigma_dy": np.array(self._sigma_dy_values),
        }

    def get_amplitude_results(self, gamma_0: float) -> dict:
        """Get full SPP results for a specific amplitude.

        Args:
            gamma_0: Strain amplitude to retrieve

        Returns:
            Dictionary of SPP metrics for that amplitude

        Raises:
            KeyError: If amplitude not in results
        """
        if gamma_0 not in self.results:
            raise KeyError(f"No results for γ_0 = {gamma_0}")
        return self.results[gamma_0].copy()

    def get_model(self) -> Any:
        """Get fitted SPPYieldStress model.

        Returns:
            Fitted model or None if not fitted
        """
        return self.model

    def get_nonlinearity_metrics(self) -> dict[float, dict]:
        """Get nonlinearity metrics (I3/I1, S, T) for each amplitude.

        Returns:
            Dictionary mapping gamma_0 to nonlinearity metrics
        """
        return {
            gamma_0: {
                "I3_I1_ratio": results.get("I3_I1_ratio", 0.0),
                "S_factor": results.get("S_factor", 0.0),
                "T_factor": results.get("T_factor", 0.0),
            }
            for gamma_0, results in self.results.items()
        }


__all__ = [
    "MastercurvePipeline",
    "ModelComparisonPipeline",
    "CreepToRelaxationPipeline",
    "FrequencyToTimePipeline",
    "SPPAmplitudeSweepPipeline",
]
