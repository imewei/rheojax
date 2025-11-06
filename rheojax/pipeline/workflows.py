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

import warnings
from typing import Any

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry
from rheojax.pipeline.base import Pipeline

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


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

        # Load all datasets
        datasets = []
        for file_path in file_paths:
            temp_pipeline = Pipeline()
            temp_pipeline.load(file_path, format=format, **load_kwargs)
            datasets.append(temp_pipeline.get_result())

        # Merge datasets with temperature metadata
        merged_data = self._merge_datasets(datasets, temperatures)

        # Apply mastercurve transform if available
        # For now, we'll implement a simple version
        self.data = merged_data
        self._apply_mastercurve_shift()

        self.history.append(
            ("mastercurve", str(len(file_paths)), str(self.reference_temp))
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

        for model_name in self.models:
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
                rmse = np.sqrt(np.mean(residuals**2))

                # Calculate R² manually (avoid calling model.score() which predicts again)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                # Calculate relative RMSE
                rel_rmse = rmse / np.mean(np.abs(y))

                # Store results
                self.results[model_name] = {
                    "model": model,
                    "parameters": model.get_params(),
                    "predictions": y_pred,  # Original predictions (may be 2D for oscillation)
                    "predictions_magnitude": y_pred_magnitude,  # Scalar magnitude for metrics
                    "residuals": residuals,
                    "rmse": float(rmse),
                    "rel_rmse": float(rel_rmse),
                    "r_squared": float(r_squared),
                    "n_params": (
                        len(model.parameters) if hasattr(model, "parameters") else 0
                    ),
                }

                # Calculate AIC (Akaike Information Criterion)
                n = len(y)
                k = self.results[model_name]["n_params"]
                if n > 0 and rmse > 0:
                    aic = n * np.log(rmse**2) + 2 * k
                    self.results[model_name]["aic"] = float(aic)

                    # Calculate BIC (Bayesian Information Criterion)
                    bic = n * np.log(rmse**2) + k * np.log(n)
                    self.results[model_name]["bic"] = float(bic)
                else:
                    self.results[model_name]["aic"] = np.inf
                    self.results[model_name]["bic"] = np.inf

                self.history.append(("fit_compare", model_name, str(r_squared)))

            except Exception as e:
                warnings.warn(f"Failed to fit model {model_name}: {e}", stacklevel=2)
                continue

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

        # Validate test mode
        test_mode = creep_data.metadata.get("test_mode", "").lower()
        if test_mode and test_mode != "creep":
            warnings.warn(
                f"Input appears to be {test_mode} data, not creep. "
                "Results may be inaccurate.",
                stacklevel=2,
            )

        if method == "approximate":
            self._approximate_conversion()
        elif method == "exact":
            self._exact_conversion()
        else:
            raise ValueError(f"Unknown method: {method}")

        self.history.append(("creep_to_relaxation", method))
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

        if frequency_data.domain != "frequency":
            warnings.warn("Input data may not be in frequency domain", stacklevel=2)

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
        return self

    def _approximate_inverse_transform(
        self, t: np.ndarray, omega: np.ndarray, G_star: np.ndarray
    ) -> np.ndarray:
        """Approximate inverse Fourier transform.

        Args:
            t: Time points
            omega: Angular frequency points
            G_star: Complex modulus

        Returns:
            Relaxation modulus at time points
        """
        # Very simplified approximation
        # Proper implementation would use numerical inverse Laplace transform
        G_t = np.zeros_like(t)

        for i, t_i in enumerate(t):
            # Approximate G(t) from G*(ω) using cos transform
            G_t[i] = np.trapezoid(np.real(G_star) * np.cos(omega * t_i), omega) * (
                2 / np.pi
            )

        return G_t


__all__ = [
    "MastercurvePipeline",
    "ModelComparisonPipeline",
    "CreepToRelaxationPipeline",
    "FrequencyToTimePipeline",
]
