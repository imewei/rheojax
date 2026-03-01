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

import copy
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData
from rheojax.logging import get_logger, log_fit, log_pipeline_stage
from rheojax.pipeline.base import Pipeline

logger = get_logger(__name__)


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
        logger.debug(
            "BatchPipeline initialized",
            has_template=template_pipeline is not None,
        )

    def set_template(self, pipeline: Pipeline) -> BatchPipeline:
        """Set template pipeline.

        Args:
            pipeline: Pipeline to use as template

        Returns:
            self for method chaining
        """
        self.template_pipeline = pipeline
        logger.debug("Template pipeline set", pipeline_type=type(pipeline).__name__)
        return self

    def process_files(
        self,
        file_paths: Iterable[str | Path],
        format: str = "auto",
        parallel: bool = False,
        parallel_io: bool = True,
        n_workers: int | None = None,
        **load_kwargs,
    ) -> BatchPipeline:
        """Process multiple files with the pipeline.

        Args:
            file_paths: List of file paths to process
            format: File format for loading
            parallel: Whether to use parallel processing for the full pipeline.
                Default False: JAX JIT cache is not thread-safe with concurrent
                ThreadPoolExecutor. Set True only for I/O-bound pipelines without
                JAX JIT calls (e.g., loading + simple numpy transforms).
            parallel_io: Whether to load files in parallel using threads.
                Default True: file I/O is thread-safe and benefits from
                parallelism. Loading phase runs in threads, pipeline replay
                runs sequentially.
            n_workers: Number of parallel workers (default: min(4, cpu_count))
            **load_kwargs: Additional arguments for data loading

        Returns:
            self for method chaining

        Example:
            >>> batch.process_files(['data1.csv', 'data2.csv'])
            >>> # Parallel mode (use with caution — JAX JIT not thread-safe):
            >>> batch.process_files(['data1.csv', 'data2.csv'], parallel=True)
        """
        import os
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if self.template_pipeline is None:
            logger.error("No template pipeline set")
            raise ValueError("No template pipeline set. Call set_template() first.")

        normalized_paths = [Path(p) for p in file_paths]

        if not normalized_paths:
            logger.debug("No files to process")
            return self

        logger.info(
            "Starting batch processing",
            n_files=len(normalized_paths),
            parallel=parallel,
            n_workers=n_workers if parallel else 1,
        )

        if parallel:
            if parallel_io:
                logger.debug(
                    "parallel_io is ignored when parallel=True "
                    "(full pipeline runs in threads, including I/O)"
                )
            import warnings as _batch_warnings

            has_fit_steps = any(
                step_action in ("fit", "fit_nlsq")
                for step_action, _ in self.template_pipeline.steps
            )
            if has_fit_steps:
                _batch_warnings.warn(
                    "parallel=True with a fitting pipeline may cause JAX JIT compilation "
                    "races between threads. Set parallel=False for pipelines that call "
                    "model.fit().",
                    UserWarning,
                    stacklevel=2,
                )

            # Parallel processing with ThreadPoolExecutor
            if n_workers is None:
                n_workers = min(4, os.cpu_count() or 1)

            def process_one(file_path):
                try:
                    logger.debug("Processing file", filepath=str(file_path))
                    result, metrics = self._process_file(
                        file_path, format=format, **load_kwargs
                    )
                    logger.debug(
                        "File processed successfully",
                        filepath=str(file_path),
                        n_points=len(result.x) if result else 0,
                    )
                    return (file_path, result, metrics, None)
                except Exception as e:
                    logger.error(
                        "Failed to process file",
                        filepath=str(file_path),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )
                    return (file_path, None, None, e)

            # NOTE: This uses concurrent.futures.ThreadPoolExecutor (not Qt threads).
            # Designed for headless/pipeline use only. If called from the GUI,
            # the calling thread blocks at as_completed(). Use WorkerPool for
            # GUI integration.
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
            # Phase 1: Optionally pre-load files in parallel (I/O only, thread-safe)
            preloaded: dict[Path, RheoData] = {}
            if parallel_io and len(normalized_paths) > 1:
                io_workers = n_workers or min(len(normalized_paths), 8)
                preloaded = self._parallel_preload(
                    normalized_paths,
                    format=format,
                    n_workers=io_workers,
                    **load_kwargs,
                )

            # Phase 2: Sequential pipeline replay (JAX-safe)
            for file_path in normalized_paths:
                try:
                    logger.debug("Processing file", filepath=str(file_path))
                    result, metrics = self._process_file(
                        file_path,
                        format=format,
                        preloaded_data=preloaded.get(file_path),
                        **load_kwargs,
                    )
                    self.results.append((file_path, result, metrics))
                    logger.debug(
                        "File processed successfully",
                        filepath=str(file_path),
                        n_points=len(result.x) if result else 0,
                    )
                except Exception as e:
                    self.errors.append((file_path, e))
                    logger.error(
                        "Failed to process file",
                        filepath=str(file_path),
                        error_type=type(e).__name__,
                        error_message=str(e),
                        exc_info=True,
                    )
                    warnings.warn(f"Failed to process {file_path}: {e}", stacklevel=2)

        logger.info(
            "Batch processing completed",
            n_success=len(self.results),
            n_errors=len(self.errors),
        )
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
        logger.debug(
            "Scanning directory",
            directory=str(directory_path),
            pattern=pattern,
            recursive=recursive,
        )

        if not directory_path.exists():
            logger.error("Directory not found", directory=str(directory))
            raise FileNotFoundError(f"Directory not found: {directory}")

        if recursive:
            file_paths = list(directory_path.rglob(pattern))
        else:
            file_paths = list(directory_path.glob(pattern))

        logger.debug(
            "Directory scan completed",
            directory=str(directory_path),
            n_files_found=len(file_paths),
        )

        if not file_paths:
            logger.warning(
                "No files matching pattern found",
                directory=str(directory),
                pattern=pattern,
            )
            warnings.warn(
                f"No files matching '{pattern}' found in {directory}", stacklevel=2
            )
            return self

        return self.process_files(file_paths, **kwargs)

    def _parallel_preload(
        self,
        file_paths: list[Path],
        format: str = "auto",
        n_workers: int = 8,
        **load_kwargs,
    ) -> dict[Path, RheoData]:
        """Pre-load files in parallel using threads (I/O only, thread-safe).

        Returns a dict mapping file_path -> RheoData for successfully loaded files.
        Failures are logged but do not raise (handled later in _process_file).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from rheojax.io import auto_load

        loaded: dict[Path, RheoData] = {}

        def _load_one(fp: Path) -> tuple[Path, RheoData | None, Exception | None]:
            try:
                data = auto_load(fp, format=format, **load_kwargs)
                return (fp, data, None)
            except Exception as e:
                return (fp, None, e)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_load_one, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                fp, data, err = future.result()
                if err is None and data is not None:
                    loaded[fp] = data
                elif err is not None:
                    logger.debug(
                        "Parallel preload failed for file",
                        filepath=str(fp),
                        error=str(err),
                    )

        logger.debug(
            "Parallel preload completed",
            n_loaded=len(loaded),
            n_total=len(file_paths),
        )
        return loaded

    def _process_file(
        self,
        file_path: Path,
        format: str = "auto",
        preloaded_data: RheoData | None = None,
        **load_kwargs,
    ) -> tuple[RheoData, dict[str, Any]]:
        """Process single file with pipeline.

        Args:
            file_path: Path to file
            format: File format
            preloaded_data: Pre-loaded RheoData (skips I/O if provided)
            **load_kwargs: Additional load arguments

        Returns:
            Tuple of (result_data, metrics)
        """
        # Clone template pipeline
        pipeline = self._clone_pipeline(self.template_pipeline)
        path = Path(file_path)

        # R11-BATCH-001: Clear template-copied steps before load+replay to avoid
        # duplicates.  Reset before load() so the load step itself is not mixed
        # with stale template steps.
        pipeline.steps = []
        pipeline._last_model = None

        # Load data — use preloaded_data if available (from parallel I/O phase)
        if preloaded_data is not None:
            pipeline.data = preloaded_data
        else:
            with log_pipeline_stage(logger, "load", filepath=str(path)):
                pipeline.load(path, format=format, **load_kwargs)

        # R12-E-006: pre-initialize metrics so transform replay errors can be
        # recorded inside the loop below before fit metrics are appended.
        metrics: dict[str, Any] = {}

        # R10-BATCH-001: Replay template steps on the newly loaded data.
        # Steps are recorded as ("fit", model_obj) or ("transform", transform_obj)
        # tuples. For each step we create a fresh model/transform of the same class
        # and re-fit/re-transform on the new dataset, preserving fit kwargs that were
        # stored in _last_fit_kwargs by the model itself.
        fit_kwargs_replay: dict[str, Any] = {}
        for step_action, step_obj in self.template_pipeline.steps:
            if step_action in ("fit", "fit_nlsq"):
                model_cls = type(step_obj)
                new_model = model_cls()
                X = np.array(pipeline.data.x)
                y = np.array(pipeline.data.y)
                _lfk = getattr(step_obj, "_last_fit_kwargs", None)
                fit_kwargs_replay = dict(_lfk) if _lfk is not None else {}
                # Strip internal tracking keys and protocol-specific kwargs
                # that should not be replayed from the template to new datasets.
                _batch_strip_keys = {
                    # NOTE: "method" is intentionally NOT stripped — ODE models
                    # that require method="scipy" must preserve this in replay.
                    "gamma_dot",
                    "sigma_init",
                    "lam_init",
                    "sigma_0",
                    "lam_0",
                    "gamma_0",
                    "omega_laos",
                    "n_cycles",
                    "points_per_cycle",
                }
                for _k in _batch_strip_keys:
                    fit_kwargs_replay.pop(_k, None)
                # R12-E-003: forward deformation_mode and poisson_ratio from
                # the template model so DMTA fits are replayed correctly.
                _deformation_mode = getattr(step_obj, "_deformation_mode", None)
                if _deformation_mode is not None:
                    fit_kwargs_replay.setdefault("deformation_mode", _deformation_mode)
                _poisson_ratio = getattr(step_obj, "_poisson_ratio", None)
                if _poisson_ratio is not None:
                    fit_kwargs_replay.setdefault("poisson_ratio", _poisson_ratio)
                new_model.fit(X, y, **fit_kwargs_replay)
                pipeline._last_model = new_model
                pipeline.steps.append((step_action, new_model))
                logger.debug(
                    "Replayed fit step",
                    model=model_cls.__name__,
                    filepath=str(path),
                )
            elif step_action == "transform":
                # Re-apply the transform to the pipeline's current data.
                # NOTE: deepcopy carries fitted state from the template (e.g.
                # shift_factors in Mastercurve).  This is intentional for
                # transforms that should reuse the template's fitted params,
                # but wrong for transforms that must re-fit per dataset.
                try:
                    transform_cls = type(step_obj)
                    new_transform = copy.deepcopy(step_obj)
                    transform_result = new_transform.transform(pipeline.data)
                    # Handle transforms that return (data, extra) tuples
                    if isinstance(transform_result, tuple):
                        pipeline.data = transform_result[0]
                    else:
                        pipeline.data = transform_result
                    # Propagate test_mode from data metadata into replay kwargs
                    # so that a subsequent fit step picks it up correctly.
                    if pipeline.data is not None and hasattr(pipeline.data, "metadata"):
                        _tm = (pipeline.data.metadata or {}).get("test_mode")
                        if _tm is not None and "test_mode" not in fit_kwargs_replay:
                            fit_kwargs_replay["test_mode"] = _tm
                    pipeline.steps.append((step_action, new_transform))
                    logger.debug(
                        "Replayed transform step",
                        transform=transform_cls.__name__,
                        filepath=str(path),
                    )
                except Exception as _te:
                    # R12-E-006: elevate to ERROR — downstream fit uses unprocessed data
                    logger.error(
                        "Transform replay failed; skipping — downstream fit uses unprocessed data",
                        transform=type(step_obj).__name__,
                        error=str(_te),
                    )
                    metrics["transform_replay_failed"] = True

        result = pipeline.get_result()

        # Compute metrics if model was fitted
        if pipeline._last_model is not None:
            model = pipeline._last_model
            X = np.array(result.x)
            y = np.array(result.y)

            with log_fit(
                logger,
                model=model.__class__.__name__,
                data_shape=X.shape,
            ) as ctx:
                metrics["r_squared"] = model.score(X, y)
                metrics["parameters"] = model.get_params()
                metrics["model"] = model.__class__.__name__

                # Calculate RMSE
                # R8-PIPE-005: handle complex oscillation data in RMSE
                y_pred = model.predict(X)
                residuals = np.asarray(y) - np.asarray(y_pred)
                metrics["rmse"] = float(np.sqrt(np.mean(np.abs(residuals) ** 2)))

                ctx["r_squared"] = metrics["r_squared"]
                ctx["rmse"] = metrics["rmse"]

        return result, metrics

    def _clone_pipeline(self, pipeline: Pipeline) -> Pipeline:
        """Clone pipeline for independent execution.

        Args:
            pipeline: Pipeline to clone

        Returns:
            New Pipeline instance
        """
        # R8-PIPE-002: implement clone instead of returning empty Pipeline
        return copy.deepcopy(pipeline)

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
            logger.warning("No results to export")
            warnings.warn("No results to export", stacklevel=2)
            return self

        output_path = Path(output_path)

        logger.info(
            "Exporting batch summary",
            output_path=str(output_path),
            format=format,
            n_results=len(df),
        )

        if format == "excel":
            df.to_excel(output_path, index=False)
        elif format == "csv":
            df.to_csv(output_path, index=False)
        else:
            logger.error("Unknown export format", format=format)
            raise ValueError(f"Unknown format: {format}")

        logger.debug("Export completed", output_path=str(output_path))
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
        original_count = len(self.results)
        self.results = [
            (path, data, metrics)
            for path, data, metrics in self.results
            if filter_fn(path, data, metrics)
        ]
        logger.debug(
            "Filter applied",
            original_count=original_count,
            filtered_count=len(self.results),
            removed_count=original_count - len(self.results),
        )
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
        n_results = len(self.results)
        n_errors = len(self.errors)
        self.results.clear()
        self.errors.clear()
        logger.debug(
            "BatchPipeline cleared",
            cleared_results=n_results,
            cleared_errors=n_errors,
        )
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
