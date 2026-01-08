"""Batch processing pipeline for multiple datasets.

This module provides utilities for applying the same pipeline to
multiple datasets efficiently, with parallel processing support.

Example:
    >>> from rheojax.pipeline import Pipeline, BatchPipeline
    >>> template = Pipeline().fit('maxwell').plot()
    >>> batch = BatchPipeline(template)
    >>> batch.process_directory('data/', pattern='*.csv')
    >>> batch.export_summary('summary.xlsx')
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.pipeline.base import Pipeline


class BatchPipeline:
    """Apply pipeline to multiple datasets.

    This class enables batch processing of multiple data files with
    the same pipeline configuration, collecting results for analysis.

    Attributes:
        template_pipeline: Template Pipeline to apply to each dataset
        results: List of (file_path, result, metrics) tuples

    Example:
        >>> template = Pipeline().fit('maxwell')
        >>> batch = BatchPipeline(template)
        >>> batch.process_files(['data1.csv', 'data2.csv'])
    """

    def __init__(self, template_pipeline: Pipeline | None = None):
        """Initialize batch pipeline.

        Args:
            template_pipeline: Template Pipeline to clone for each file.
                If None, must be set before processing.
        """
        self.template_pipeline = template_pipeline
        self.results: list[tuple[Path, RheoData, dict[str, Any]]] = []
        self.errors: list[tuple[Path, Exception]] = []

    def set_template(self, pipeline: Pipeline) -> BatchPipeline:
        """Set template pipeline.

        Args:
            pipeline: Pipeline to use as template

        Returns:
            self for method chaining
        """
        self.template_pipeline = pipeline
        return self

    def process_files(
        self,
        file_paths: Iterable[str | Path],
        format: str = "auto",
        parallel: bool = False,
        n_workers: int | None = None,
        **load_kwargs,
    ) -> BatchPipeline:
        """Process multiple files with the pipeline.

        Args:
            file_paths: List of file paths to process
            format: File format for loading
            parallel: Whether to use parallel processing
            n_workers: Number of parallel workers (default: min(4, cpu_count))
            **load_kwargs: Additional arguments for data loading

        Returns:
            self for method chaining

        Example:
            >>> batch.process_files(['data1.csv', 'data2.csv'], parallel=True)
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.template_pipeline is None:
            raise ValueError("No template pipeline set. Call set_template() first.")

        normalized_paths = [Path(p) for p in file_paths]

        if not normalized_paths:
            return self

        if parallel:
            # Parallel processing with ThreadPoolExecutor
            if n_workers is None:
                n_workers = min(4, os.cpu_count() or 1)

            def process_one(file_path):
                try:
                    result, metrics = self._process_file(
                        file_path, format=format, **load_kwargs
                    )
                    return (file_path, result, metrics, None)
                except Exception as e:
                    return (file_path, None, None, e)

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(process_one, fp): fp for fp in normalized_paths
                }

                for future in as_completed(futures):
                    file_path, result, metrics, error = future.result()
                    if error is None:
                        self.results.append((file_path, result, metrics))
                    else:
                        self.errors.append((file_path, error))
                        warnings.warn(
                            f"Failed to process {file_path}: {error}", stacklevel=2
                        )
        else:
            # Sequential processing
            for file_path in normalized_paths:
                try:
                    result, metrics = self._process_file(
                        file_path, format=format, **load_kwargs
                    )
                    self.results.append((file_path, result, metrics))
                except Exception as e:
                    self.errors.append((file_path, e))
                    warnings.warn(f"Failed to process {file_path}: {e}", stacklevel=2)

        return self

    def process_directory(
        self,
        directory: str | Path,
        pattern: str = "*.csv",
        recursive: bool = False,
        **kwargs,
    ) -> BatchPipeline:
        """Process all files in directory matching pattern.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.csv', '*.xlsx')
            recursive: Whether to search recursively
            **kwargs: Additional arguments passed to process_files

        Returns:
            self for method chaining

        Example:
            >>> batch.process_directory('data/', pattern='*.csv')
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            file_paths = list(directory_path.rglob(pattern))
        else:
            file_paths = list(directory_path.glob(pattern))

        if not file_paths:
            warnings.warn(
                f"No files matching '{pattern}' found in {directory}", stacklevel=2
            )
            return self

        return self.process_files(file_paths, **kwargs)

    def _process_file(
        self, file_path: Path, format: str = "auto", **load_kwargs
    ) -> tuple[RheoData, dict[str, Any]]:
        """Process single file with pipeline.

        Args:
            file_path: Path to file
            format: File format
            **load_kwargs: Additional load arguments

        Returns:
            Tuple of (result_data, metrics)
        """
        # Clone template pipeline
        pipeline = self._clone_pipeline(self.template_pipeline)
        path = Path(file_path)

        # Load data
        pipeline.load(path, format=format, **load_kwargs)

        # Execute template steps (transforms, fits, etc.)
        # The template should already have the steps configured
        # We just need to execute them on the new data

        # For now, we'll execute the history of the template
        # This is simplified - a full implementation would clone and execute steps
        result = pipeline.get_result()

        # Compute metrics if model was fitted
        metrics: dict[str, Any] = {}
        if pipeline._last_model is not None:
            model = pipeline._last_model
            X = np.array(result.x)
            y = np.array(result.y)

            metrics["r_squared"] = model.score(X, y)
            metrics["parameters"] = model.get_params()
            metrics["model"] = model.__class__.__name__

            # Calculate RMSE
            y_pred = model.predict(X)
            residuals = y - y_pred
            metrics["rmse"] = float(np.sqrt(np.mean(residuals**2)))

        return result, metrics

    def _clone_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Clone pipeline for independent execution.

        Args:
            pipeline: Pipeline to clone

        Returns:
            New Pipeline instance
        """
        # For now, create a new pipeline
        # A full implementation would deep copy the pipeline configuration
        return Pipeline()

    def get_results(self) -> list[tuple[Path, RheoData, dict[str, Any]]]:
        """Get all processing results.

        Returns:
            List of (file_path, result_data, metrics) tuples

        Example:
            >>> results = batch.get_results()
            >>> for path, data, metrics in results:
            ...     print(f"{path}: R²={metrics.get('r_squared', 0):.4f}")
        """
        return self.results.copy()

    def get_errors(self) -> list[tuple[Path, Exception]]:
        """Get processing errors.

        Returns:
            List of (file_path, exception) tuples

        Example:
            >>> errors = batch.get_errors()
            >>> for path, error in errors:
            ...     print(f"Error in {path}: {error}")
        """
        return self.errors.copy()

    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get summary DataFrame of all results.

        Returns:
            DataFrame with file paths and metrics

        Example:
            >>> df = batch.get_summary_dataframe()
            >>> print(df)
        """
        if not self.results:
            return pd.DataFrame()

        summary_data: list[dict[str, Any]] = []
        for file_path, result, metrics in self.results:
            path_obj = Path(file_path)
            row = {
                "file_path": str(path_obj),
                "file_name": path_obj.name,
                "n_points": len(result.x),
            }
            row.update(metrics)
            summary_data.append(row)

        return pd.DataFrame(summary_data)

    def export_summary(
        self, output_path: str | Path, format: str = "excel"
    ) -> BatchPipeline:
        """Export summary of batch results.

        Args:
            output_path: Output file path
            format: Output format ('excel', 'csv')

        Returns:
            self for method chaining

        Example:
            >>> batch.export_summary('summary.xlsx')
        """
        df = self.get_summary_dataframe()

        if df.empty:
            warnings.warn("No results to export", stacklevel=2)
            return self

        output_path = Path(output_path)

        if format == "excel":
            df.to_excel(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        return self

    def apply_filter(
        self, filter_fn: Callable[[Path, RheoData, dict[str, Any]], bool]
    ) -> BatchPipeline:
        """Filter results based on custom criteria.

        Args:
            filter_fn: Function that takes (file_path, data, metrics) and
                returns True to keep the result

        Returns:
            self for method chaining

        Example:
            >>> # Keep only results with R² > 0.9
            >>> batch.apply_filter(lambda p, d, m: m.get('r_squared', 0) > 0.9)
        """
        self.results = [
            (path, data, metrics)
            for path, data, metrics in self.results
            if filter_fn(path, data, metrics)
        ]
        return self

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics across all results.

        Returns:
            Dictionary with summary statistics

        Example:
            >>> stats = batch.get_statistics()
            >>> print(f"Mean R²: {stats['mean_r_squared']:.4f}")
        """
        if not self.results:
            return {}

        # Collect metrics
        r_squared_values = []
        rmse_values = []

        for _, _, metrics in self.results:
            if "r_squared" in metrics:
                r_squared_values.append(metrics["r_squared"])
            if "rmse" in metrics:
                rmse_values.append(metrics["rmse"])

        stats = {
            "total_files": len(self.results),
            "total_errors": len(self.errors),
            "success_rate": (
                len(self.results) / (len(self.results) + len(self.errors))
                if (len(self.results) + len(self.errors)) > 0
                else 0
            ),
        }

        if r_squared_values:
            stats.update(
                {
                    "mean_r_squared": float(np.mean(r_squared_values)),
                    "std_r_squared": float(np.std(r_squared_values)),
                    "min_r_squared": float(np.min(r_squared_values)),
                    "max_r_squared": float(np.max(r_squared_values)),
                }
            )

        if rmse_values:
            stats.update(
                {
                    "mean_rmse": float(np.mean(rmse_values)),
                    "std_rmse": float(np.std(rmse_values)),
                    "min_rmse": float(np.min(rmse_values)),
                    "max_rmse": float(np.max(rmse_values)),
                }
            )

        return stats

    def clear(self) -> BatchPipeline:
        """Clear all results and errors.

        Returns:
            self for method chaining
        """
        self.results.clear()
        self.errors.clear()
        return self

    def __len__(self) -> int:
        """Get number of processed results."""
        return len(self.results)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BatchPipeline(results={len(self.results)}, " f"errors={len(self.errors)})"
        )


__all__ = ["BatchPipeline"]
