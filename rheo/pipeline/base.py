"""Base pipeline class for fluent API workflows.

This module provides the core Pipeline class that enables intuitive method
chaining for common rheological analysis workflows.

Example:
    >>> from rheo.pipeline import Pipeline
    >>> pipeline = Pipeline()
    >>> result = (pipeline
    ...     .load('data.csv')
    ...     .transform('smooth', window_size=5)
    ...     .fit('maxwell')
    ...     .plot()
    ...     .save('result.hdf5')
    ...     .get_result())
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from rheo.core.base import BaseModel, BaseTransform
from rheo.core.data import RheoData
from rheo.core.jax_config import safe_import_jax
from rheo.core.registry import ModelRegistry, TransformRegistry

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()


class Pipeline:
    """Fluent API for rheological analysis workflows.

    This class provides a chainable interface for loading data, applying
    transforms, fitting models, and generating outputs. All methods return
    self to enable method chaining.

    Attributes:
        data: Current RheoData state
        steps: List of (operation, object) tuples for fitted models
        history: List of (operation, details) tuples tracking all operations
        _last_model: Last fitted model for convenience

    Example:
        >>> pipeline = Pipeline()
        >>> pipeline.load('data.csv').fit('maxwell').plot()
    """

    def __init__(self, data: RheoData | None = None):
        """Initialize pipeline.

        Args:
            data: Optional initial RheoData. If None, must call load() first.
        """
        self.data = data
        self.steps: list[tuple[str, Any]] = []
        self.history: list[tuple[str, ...]] = []
        self._last_model: BaseModel | None = None

    def load(self, file_path: str, format: str = "auto", **kwargs) -> Pipeline:
        """Load data from file.

        Args:
            file_path: Path to data file
            format: File format ('auto', 'csv', 'excel', 'trios', 'hdf5')
            **kwargs: Additional arguments passed to reader

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not recognized

        Example:
            >>> pipeline = Pipeline().load('data.csv', x_col='time', y_col='stress')
        """
        from rheo.io import auto_load

        if format == "auto":
            result = auto_load(file_path, **kwargs)
        else:
            # Format-specific loading
            if format == "csv":
                from rheo.io import load_csv

                result = load_csv(file_path, **kwargs)
            elif format == "excel":
                from rheo.io import load_excel

                result = load_excel(file_path, **kwargs)
            elif format == "trios":
                from rheo.io import load_trios

                result = load_trios(file_path, **kwargs)
            elif format == "hdf5":
                from rheo.io import load_hdf5

                result = load_hdf5(file_path, **kwargs)
            else:
                raise ValueError(f"Unknown format: {format}")

        # Handle multiple segments (for TRIOS)
        if isinstance(result, list):
            if len(result) == 1:
                self.data = result[0]
            else:
                warnings.warn(f"Loaded {len(result)} segments. Using first segment.", stacklevel=2)
                self.data = result[0]
        else:
            self.data = result

        self.history.append(("load", file_path, format))
        return self

    def transform(self, transform: str | BaseTransform, **kwargs) -> Pipeline:
        """Apply a transform to the data.

        Args:
            transform: Transform name (string) or Transform instance
            **kwargs: Arguments passed to transform constructor (if string)

        Returns:
            self for method chaining

        Raises:
            ValueError: If data not loaded or transform not found

        Example:
            >>> pipeline.transform('smooth', window_size=5)
            >>> # or with instance
            >>> from rheo.transforms import SmoothTransform
            >>> pipeline.transform(SmoothTransform(window_size=5))
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Create transform if string
        if isinstance(transform, str):
            transform_obj = TransformRegistry.create(transform, **kwargs)
            transform_name = transform
        else:
            transform_obj = transform
            transform_name = transform_obj.__class__.__name__

        # Apply transform to x and y data
        # The transform operates on the y data
        self.data.y = transform_obj.transform(self.data.y)

        self.history.append(("transform", transform_name))
        return self

    def fit(
        self,
        model: str | BaseModel,
        method: str = "auto",
        use_jax: bool = True,
        **fit_kwargs,
    ) -> Pipeline:
        """Fit a model to the data.

        Args:
            model: Model name (string) or Model instance
            method: Optimization method ('auto', 'L-BFGS-B', etc.)
            use_jax: Whether to use JAX gradients for optimization
            **fit_kwargs: Additional arguments passed to optimizer

        Returns:
            self for method chaining

        Raises:
            ValueError: If data not loaded or model not found

        Example:
            >>> pipeline.fit('maxwell')
            >>> # or with instance
            >>> from rheo.models.linear import Maxwell
            >>> pipeline.fit(Maxwell())
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

        # Fit using model's fit method
        X = self.data.x
        y = self.data.y

        # Convert to numpy for fitting
        if isinstance(X, jnp.ndarray):
            X = np.array(X)
        if isinstance(y, jnp.ndarray):
            y = np.array(y)

        model_obj.fit(X, y, **fit_kwargs)

        # Store fitted model
        self._last_model = model_obj
        self.steps.append(("fit", model_obj))
        self.history.append(("fit", model_name, model_obj.score(X, y)))

        return self

    def predict(
        self, model: BaseModel | None = None, X: np.ndarray | None = None
    ) -> RheoData:
        """Generate predictions from fitted model.

        Args:
            model: Model to use for prediction. If None, uses last fitted model.
            X: Input data for prediction. If None, uses current data.x.

        Returns:
            RheoData with predictions

        Raises:
            ValueError: If no model has been fitted

        Example:
            >>> predictions = pipeline.predict()
        """
        if model is None:
            model = self._last_model

        if model is None:
            raise ValueError("No model fitted. Call fit() first.")

        if X is None:
            if self.data is None:
                raise ValueError("No data available for prediction.")
            X = self.data.x

        # Convert to numpy for prediction
        if isinstance(X, jnp.ndarray):
            X = np.array(X)

        predictions = model.predict(X)

        return RheoData(
            x=X,
            y=predictions,
            x_units=self.data.x_units if self.data else None,
            y_units=self.data.y_units if self.data else None,
            domain=self.data.domain if self.data else "time",
            metadata={
                **(self.data.metadata if self.data else {}),
                "type": "prediction",
                "model": model.__class__.__name__,
            },
            validate=False,
        )

    def plot(
        self,
        show: bool = True,
        style: str = "default",
        include_prediction: bool = False,
        **plot_kwargs,
    ) -> Pipeline:
        """Plot current data state.

        Args:
            show: Whether to call plt.show()
            style: Plot style ('default', 'publication', 'presentation')
            include_prediction: If True and model fitted, overlay predictions
            **plot_kwargs: Additional arguments passed to plotting function

        Returns:
            self for method chaining

        Example:
            >>> pipeline.plot(style='publication')
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        from rheo.visualization.plotter import plot_rheo_data

        fig, ax = plot_rheo_data(self.data, style=style, **plot_kwargs)

        # Optionally overlay predictions
        if include_prediction and self._last_model is not None:
            predictions = self.predict()
            import matplotlib.pyplot as plt

            # Get the axes (handle both single and multiple axes)
            if isinstance(ax, np.ndarray):
                ax_plot = ax[0]
            else:
                ax_plot = ax

            ax_plot.plot(
                predictions.x,
                predictions.y,
                "--",
                label="Model Prediction",
                linewidth=2,
            )
            ax_plot.legend()

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        self.history.append(("plot", style))
        return self

    def save(self, file_path: str, format: str = "hdf5", **kwargs) -> Pipeline:
        """Save current data to file.

        Args:
            file_path: Output file path
            format: Output format ('hdf5', 'excel', 'csv')
            **kwargs: Additional arguments passed to writer

        Returns:
            self for method chaining

        Example:
            >>> pipeline.save('output.hdf5')
        """
        if self.data is None:
            raise ValueError("No data to save. Call load() first.")

        if format == "hdf5":
            from rheo.io import save_hdf5

            save_hdf5(self.data, file_path, **kwargs)
        elif format == "excel":
            from rheo.io import save_excel

            save_excel(self.data, file_path, **kwargs)
        elif format == "csv":
            # Simple CSV export
            import pandas as pd

            df = pd.DataFrame({"x": np.array(self.data.x), "y": np.array(self.data.y)})
            df.to_csv(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

        self.history.append(("save", file_path, format))
        return self

    def get_result(self) -> RheoData:
        """Get current data state.

        Returns:
            Current RheoData

        Example:
            >>> data = pipeline.get_result()
        """
        if self.data is None:
            raise ValueError("No data available. Call load() first.")
        return self.data

    def get_history(self) -> list[tuple]:
        """Get pipeline execution history.

        Returns:
            List of (operation, details) tuples

        Example:
            >>> history = pipeline.get_history()
            >>> for step in history:
            ...     print(step)
        """
        return self.history.copy()

    def get_last_model(self) -> BaseModel | None:
        """Get the last fitted model.

        Returns:
            Last fitted BaseModel or None

        Example:
            >>> model = pipeline.get_last_model()
            >>> params = model.get_params()
        """
        return self._last_model

    def get_all_models(self) -> list[BaseModel]:
        """Get all fitted models from pipeline.

        Returns:
            List of all fitted models

        Example:
            >>> models = pipeline.get_all_models()
        """
        return [step[1] for step in self.steps if step[0] == "fit"]

    def clone(self) -> Pipeline:
        """Create a copy of the pipeline.

        Returns:
            New Pipeline with copied data and history

        Example:
            >>> pipeline2 = pipeline.clone()
        """
        new_pipeline = Pipeline(data=self.data.copy() if self.data else None)
        new_pipeline.steps = self.steps.copy()
        new_pipeline.history = self.history.copy()
        new_pipeline._last_model = self._last_model
        return new_pipeline

    def reset(self) -> Pipeline:
        """Reset pipeline to initial state.

        Returns:
            self for method chaining

        Example:
            >>> pipeline.reset()
        """
        self.data = None
        self.steps = []
        self.history = []
        self._last_model = None
        return self

    def __repr__(self) -> str:
        """String representation of pipeline."""
        n_steps = len(self.history)
        has_data = self.data is not None
        has_model = self._last_model is not None
        return (
            f"Pipeline(steps={n_steps}, "
            f"has_data={has_data}, "
            f"has_model={has_model})"
        )


__all__ = ["Pipeline"]
