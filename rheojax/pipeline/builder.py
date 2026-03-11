"""Pipeline builder for programmatic pipeline construction.

This module provides a builder pattern for creating and validating
complex pipelines programmatically with fluent API.

Example:
    >>> from rheojax.pipeline.builder import PipelineBuilder
    >>> pipeline = (PipelineBuilder()
    ...     .add_load_step('data.csv', x_col='time', y_col='stress')
    ...     .add_transform_step('smooth', window_size=5)
    ...     .add_fit_step('maxwell')
    ...     .add_plot_step(style='publication')
    ...     .build())
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.logging import get_logger
from rheojax.pipeline.base import Pipeline

# Module-level logger
logger = get_logger(__name__)


class PipelineBuilder:
    """Build and validate pipelines programmatically.

    This class provides a fluent API for constructing pipelines with
    validation of step order and dependencies.

    Example:
        >>> builder = PipelineBuilder()
        >>> builder.add_load_step('data.csv')
        >>> builder.add_fit_step('maxwell')
        >>> pipeline = builder.build()
    """

    def __init__(self):
        """Initialize pipeline builder."""
        self.steps: list[tuple[str, dict[str, Any]]] = []

    def add_load_step(
        self, file_path: str | Path, format: str = "auto", **kwargs
    ) -> PipelineBuilder:
        """Add data loading step.

        Args:
            file_path: Path to data file
            format: File format ('auto', 'csv', 'excel', etc.)
            **kwargs: Additional arguments for loader

        Returns:
            self for method chaining

        Example:
            >>> builder.add_load_step('data.csv', x_col='time', y_col='stress')
        """
        file_path_str = str(file_path)
        self.steps.append(
            ("load", {"file_path": file_path_str, "format": format, **kwargs})
        )
        return self

    def add_transform_step(self, transform_name: str, **kwargs) -> PipelineBuilder:
        """Add transform step.

        Args:
            transform_name: Name of transform to apply
            **kwargs: Arguments for transform constructor

        Returns:
            self for method chaining

        Example:
            >>> builder.add_transform_step('smooth', window_size=5)
        """
        self.steps.append(("transform", {"name": transform_name, **kwargs}))
        return self

    def add_fit_step(
        self, model_name: str, method: str = "auto", use_jax: bool = True, **kwargs
    ) -> PipelineBuilder:
        """Add model fitting step.

        Args:
            model_name: Name of model to fit
            method: Optimization method
            use_jax: Whether to use JAX gradients
            **kwargs: Additional fit arguments

        Returns:
            self for method chaining

        Example:
            >>> builder.add_fit_step('maxwell')
        """
        self.steps.append(
            (
                "fit",
                {"model": model_name, "method": method, "use_jax": use_jax, **kwargs},
            )
        )
        return self

    def add_predict_step(
        self, store_as: str | None = None, **kwargs
    ) -> PipelineBuilder:
        """Add prediction step.

        Args:
            store_as: Optional name to store prediction
            **kwargs: Additional prediction arguments

        Returns:
            self for method chaining

        Example:
            >>> builder.add_predict_step(store_as='prediction')
        """
        self.steps.append(("predict", {"store_as": store_as, **kwargs}))
        return self

    def add_plot_step(
        self, show: bool = False, style: str = "default", **kwargs
    ) -> PipelineBuilder:
        """Add plotting step.

        Args:
            show: Whether to display plot
            style: Plot style
            **kwargs: Additional plot arguments

        Returns:
            self for method chaining

        Example:
            >>> builder.add_plot_step(style='publication', show=True)
        """
        self.steps.append(("plot", {"show": show, "style": style, **kwargs}))
        return self

    def add_bayesian_step(
        self,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 4,
        seed: int = 0,
        warm_start: bool = True,
        **kwargs,
    ) -> PipelineBuilder:
        """Add Bayesian inference step (NUTS sampling).

        Args:
            num_warmup: Number of warmup iterations per chain
            num_samples: Number of posterior samples per chain
            num_chains: Number of MCMC chains
            seed: Random seed for reproducibility
            warm_start: Whether to use NLSQ results as initial values
            **kwargs: Additional arguments for fit_bayesian()

        Returns:
            self for method chaining

        Example:
            >>> builder.add_bayesian_step(num_warmup=500, num_samples=1000)
        """
        self.steps.append(
            (
                "bayesian",
                {
                    "num_warmup": num_warmup,
                    "num_samples": num_samples,
                    "num_chains": num_chains,
                    "seed": seed,
                    "warm_start": warm_start,
                    **kwargs,
                },
            )
        )
        return self

    def add_export_step(
        self,
        output_path: str | Path,
        format: str = "auto",
        **kwargs,
    ) -> PipelineBuilder:
        """Add analysis export step.

        Args:
            output_path: Output directory or file path
            format: Export format ('directory', 'excel', 'hdf5', 'auto')
            **kwargs: Additional arguments for Pipeline.export()

        Returns:
            self for method chaining

        Example:
            >>> builder.add_export_step('./results', format='directory')
        """
        self.steps.append(
            ("export", {"output_path": str(output_path), "format": format, **kwargs})
        )
        return self

    def add_save_step(
        self, file_path: str | Path, format: str = "hdf5", **kwargs
    ) -> PipelineBuilder:
        """Add data saving step.

        Args:
            file_path: Output file path
            format: Output format
            **kwargs: Additional save arguments

        Returns:
            self for method chaining

        Example:
            >>> builder.add_save_step('output.hdf5')
        """
        file_path_str = str(file_path)
        self.steps.append(
            ("save", {"file_path": file_path_str, "format": format, **kwargs})
        )
        return self

    def build(self, validate: bool = True) -> Pipeline:
        """Build and optionally validate pipeline.

        Args:
            validate: Whether to validate pipeline structure

        Returns:
            Constructed Pipeline instance

        Raises:
            ValueError: If validation fails

        Example:
            >>> pipeline = builder.build()
        """
        if validate:
            self._validate_pipeline()

        pipeline = Pipeline()

        if not validate:
            # Skip eager execution — return an unconfigured Pipeline shell so
            # callers can inspect the builder output without requiring valid
            # data / model state.
            return pipeline

        self._execute_steps(pipeline)
        return pipeline

    def _execute_steps(
        self,
        pipeline: Pipeline,
        on_step: "Callable[[int, str], None] | None" = None,
    ) -> None:
        """Execute all builder steps on a Pipeline.

        Parameters
        ----------
        pipeline : Pipeline
            Target pipeline instance.
        on_step : callable, optional
            ``on_step(index, step_type)`` is called before each step executes.
            Used by the CLI for per-step progress reporting.
        """
        for i, (step_type, step_kwargs) in enumerate(self.steps):
            if on_step is not None:
                on_step(i, step_type)
            # Copy to avoid mutating stored dicts (allows repeated build() calls)
            kwargs = step_kwargs.copy()
            if step_type == "load":
                pipeline.load(**kwargs)
            elif step_type == "transform":
                transform_name = kwargs.pop("name")
                pipeline.transform(transform_name, **kwargs)
            elif step_type == "fit":
                model_name = kwargs.pop("model")
                pipeline.fit(model_name, **kwargs)
            elif step_type == "predict":
                # Store prediction but don't break chain
                kwargs.pop("store_as", None)  # Remove for now
                # Predictions are implicit in pipeline
                logger.warning(
                    "Predict step is a no-op — predictions are generated "
                    "automatically during fit/bayesian steps"
                )
            elif step_type == "bayesian":
                # Delegate to Pipeline.fit_bayesian which handles
                # warm_start, test_mode propagation, and result caching
                pipeline.fit_bayesian(**kwargs)
            elif step_type == "export":
                output_path = kwargs.pop("output_path")
                pipeline.export(output_path, **kwargs)
            elif step_type == "plot":
                pipeline.plot(**kwargs)
            elif step_type == "save":
                pipeline.save(**kwargs)

    def _validate_pipeline(self):
        """Validate pipeline construction.

        Raises:
            ValueError: If pipeline structure is invalid
        """
        if not self.steps:
            raise ValueError("Pipeline has no steps")

        # Check that first step is load
        if self.steps[0][0] != "load":
            raise ValueError(
                "Pipeline must start with a load step. "
                f"First step is: {self.steps[0][0]}"
            )

        # Check that data-dependent steps come after load
        has_loaded = False
        for step_type, _ in self.steps:
            if step_type == "load":
                has_loaded = True
            elif step_type in [
                "transform", "fit", "plot", "save", "bayesian", "export",
            ]:
                if not has_loaded:
                    raise ValueError(
                        f"Step '{step_type}' requires data to be loaded first"
                    )

        # Check that fit comes before predict and bayesian
        fit_indices = [
            i for i, (step_type, _) in enumerate(self.steps) if step_type == "fit"
        ]
        predict_indices = [
            i for i, (step_type, _) in enumerate(self.steps) if step_type == "predict"
        ]
        bayesian_indices = [
            i
            for i, (step_type, _) in enumerate(self.steps)
            if step_type == "bayesian"
        ]

        for pred_idx in predict_indices:
            if not any(fit_idx < pred_idx for fit_idx in fit_indices):
                warnings.warn(
                    f"Predict step at index {pred_idx} has no prior fit step",
                    stacklevel=2,
                )

        for bayes_idx in bayesian_indices:
            if not any(fit_idx < bayes_idx for fit_idx in fit_indices):
                raise ValueError(
                    f"Bayesian step at index {bayes_idx} requires a prior fit step"
                )

        # Validate that referenced models/transforms exist
        self._validate_components()

    def _validate_components(self):
        """Validate that models and transforms exist in registry.

        Raises:
            ValueError: If component not found
        """
        # Trigger lazy discovery so registries are populated before listing.
        from rheojax.models import _ensure_all_registered as _ensure_models

        _ensure_models()
        try:
            from rheojax.transforms import _ensure_all_registered as _ensure_transforms

            _ensure_transforms()
        except ImportError:
            pass

        for step_type, step_kwargs in self.steps:
            if step_type == "fit":
                model_name = step_kwargs.get("model")
                if model_name:
                    available = ModelRegistry.list_models()
                    if model_name not in available:
                        raise ValueError(
                            f"Model '{model_name}' not found in registry. "
                            f"Available: {available}"
                        )

            elif step_type == "transform":
                transform_name = step_kwargs.get("name")
                if transform_name:
                    available = TransformRegistry.list_transforms()
                    if transform_name not in available:
                        raise ValueError(
                            f"Transform '{transform_name}' not found in registry. "
                            f"Available: {available}"
                        )

    def clear(self) -> PipelineBuilder:
        """Clear all steps.

        Returns:
            self for method chaining

        Example:
            >>> builder.clear()
        """
        self.steps = []
        return self

    def get_steps(self) -> list[tuple[str, dict[str, Any]]]:
        """Get current pipeline steps.

        Returns:
            List of (step_type, kwargs) tuples

        Example:
            >>> steps = builder.get_steps()
        """
        return self.steps.copy()

    def __len__(self) -> int:
        """Get number of steps."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation."""
        step_types = [step[0] for step in self.steps]
        return f"PipelineBuilder(steps={step_types})"


__all__ = [
    "PipelineBuilder",
]
