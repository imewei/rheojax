"""Base pipeline class for fluent API workflows.

This module provides the core Pipeline class that enables intuitive method
chaining for common rheological analysis workflows.

Example:
    >>> from rheojax.pipeline import Pipeline
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

import copy
import uuid
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.base import BaseModel, BaseTransform
from rheojax.core.data import RheoData
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.registry import ModelRegistry, TransformRegistry
from rheojax.logging import get_logger, log_pipeline_stage

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Module-level logger
logger = get_logger(__name__)


def _is_jax_array(x: Any) -> bool:
    """Robust check for JAX arrays across JAX versions."""
    return hasattr(x, "devices") and not isinstance(x, np.ndarray)


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
        self.history: list[tuple[Any, ...]] = []
        self._last_model: BaseModel | None = None
        self._last_fit_result: Any = None
        self._last_bayesian_result: Any = None
        self._transform_results: dict[str, tuple[Any, RheoData | None]] = {}
        self._last_transform_name: str | None = None
        self._current_figure: Any = None
        self._diagnostic_results: Any = None
        self._last_comparison: Any = None
        self._id = str(uuid.uuid4())[:8]
        logger.debug(
            "Pipeline initialized",
            pipeline_id=self._id,
            has_initial_data=data is not None,
        )

    def load(
        self,
        file_path: str | Path,
        format: str = "auto",
        *,
        test_mode: str | None = None,
        initial_test_mode: str | None = None,
        **kwargs,
    ) -> Pipeline:
        """Load data from file.

        Args:
            file_path: Path to data file
            format: File format ('auto', 'csv', 'excel', 'trios', 'hdf5')
            test_mode: Optional rheological mode metadata to attach to the
                resulting RheoData (e.g., 'relaxation', 'creep', 'oscillation')
            initial_test_mode: Backwards-compatible alias for test_mode
            **kwargs: Additional arguments passed to reader

        Returns:
            self for method chaining

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format not recognized

        Example:
            >>> pipeline = Pipeline().load('data.csv', x_col='time', y_col='stress')
        """
        from rheojax.io import auto_load

        path = Path(file_path)

        explicit_mode = test_mode if test_mode is not None else initial_test_mode

        with log_pipeline_stage(
            logger, "load", pipeline_id=self._id, file_path=str(path), format=format
        ) as ctx:
            try:
                if format == "auto":
                    result = auto_load(path, **kwargs)
                else:
                    # Format-specific loading
                    if format == "csv":
                        from rheojax.io import load_csv

                        result = load_csv(path, **kwargs)
                    elif format == "excel":
                        from rheojax.io import load_excel

                        result = load_excel(path, **kwargs)
                    elif format == "trios":
                        from rheojax.io import load_trios

                        result = load_trios(path, **kwargs)
                    elif format == "hdf5":
                        from rheojax.io import load_hdf5

                        result = load_hdf5(path, **kwargs)
                    elif format == "npz":
                        from rheojax.io.writers.npz_writer import load_npz

                        result = load_npz(path)
                    else:
                        raise ValueError(f"Unknown format: {format}")

                # Handle multiple segments (for TRIOS)
                if isinstance(result, list):
                    if len(result) == 1:
                        self.data = result[0]
                    else:
                        warnings.warn(
                            f"Loaded {len(result)} segments. Using first segment.",
                            stacklevel=2,
                        )
                        self.data = result[0]
                    ctx["n_segments"] = len(result)
                else:
                    self.data = result

                self._apply_test_mode_metadata(self.data, explicit_mode)
                ctx["n_points"] = len(self.data.x) if self.data else 0
                ctx["test_mode"] = explicit_mode

            except Exception as e:
                logger.error(
                    "Failed to load data",
                    pipeline_id=self._id,
                    file_path=str(path),
                    format=format,
                    error=str(e),
                    exc_info=True,
                )
                raise

        self.history.append(("load", str(path), format))
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
            >>> from rheojax.transforms import SmoothTransform
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

        logger.debug(
            "Creating transform",
            pipeline_id=self._id,
            transform=transform_name,
        )

        with log_pipeline_stage(
            logger, "transform", pipeline_id=self._id, transform=transform_name
        ) as ctx:
            try:
                # Apply transform to full RheoData (not raw y array)
                # Transforms expect RheoData with x, y, metadata, domain
                ctx["input_shape"] = len(self.data.x)
                pre_transform_data = self.data
                result = transform_obj.transform(self.data)

                # Cache full result + pre-transform data for plot_transform()
                self._transform_results[transform_name] = (result, pre_transform_data)
                self._last_transform_name = transform_name

                if isinstance(result, tuple):
                    self.data = result[0]
                else:
                    self.data = result
            except Exception as e:
                logger.error(
                    "Transform failed",
                    pipeline_id=self._id,
                    transform=transform_name,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # R12-E-002: append to steps so batch replay can replay transforms
        self.steps.append(("transform", transform_obj))
        self.history.append(("transform", transform_name))
        return self

    def _apply_test_mode_metadata(
        self, data: RheoData | None, mode: str | None
    ) -> None:
        """Attach explicit test mode information to loaded data."""

        if data is None or mode is None:
            return

        if data.metadata is None:
            data.metadata = {}

        data.metadata["test_mode"] = mode
        data.metadata.setdefault("detected_test_mode", mode)

        # Persist explicit annotation for downstream helpers that rely on it
        if hasattr(data, "_explicit_test_mode"):
            data._explicit_test_mode = mode

    def fit(
        self,
        model: str | BaseModel,
        method: str = "auto",
        **fit_kwargs,
    ) -> Pipeline:
        """Fit a model to the data.

        Args:
            model: Model name (string) or Model instance
            method: Optimization method passed to model.fit() ('nlsq', 'scipy', 'auto').
                Default 'auto' lets the model choose.
            **fit_kwargs: Additional arguments passed to optimizer

        Returns:
            self for method chaining

        Raises:
            ValueError: If data not loaded or model not found

        Example:
            >>> pipeline.fit('maxwell')
            >>> # or with instance
            >>> from rheojax.models.linear import Maxwell
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

        logger.debug(
            "Creating model for fitting",
            pipeline_id=self._id,
            model=model_name,
        )

        # Fit using model's fit method
        X = self.data.x
        y = self.data.y

        # Convert to numpy for fitting
        if _is_jax_array(X):
            X = np.array(X)
        if _is_jax_array(y):
            y = np.array(y)

        with log_pipeline_stage(
            logger,
            "fit",
            pipeline_id=self._id,
            model=model_name,
            data_shape=X.shape,  # type: ignore[union-attr]
        ) as ctx:
            try:
                # PB-001: auto-propagate test_mode from loaded data metadata
                if hasattr(self, "data") and self.data is not None:
                    _meta = getattr(self.data, "metadata", None)
                    if _meta is not None:
                        if "test_mode" not in fit_kwargs:
                            _tm = _meta.get("test_mode")
                            if _tm is not None:
                                fit_kwargs["test_mode"] = _tm
                        # R9-PIPE-DMT: propagate deformation_mode for DMTA data
                        if "deformation_mode" not in fit_kwargs:
                            _dm = _meta.get("deformation_mode")
                            if _dm is not None:
                                fit_kwargs["deformation_mode"] = _dm
                        if "poisson_ratio" not in fit_kwargs:
                            _pr = _meta.get("poisson_ratio")
                            if _pr is not None:
                                fit_kwargs["poisson_ratio"] = _pr
                # R12-E-001: forward method kwarg to model.fit()
                fit_kwargs["method"] = method
                model_obj.fit(X, y, **fit_kwargs)
                self._last_model = model_obj
                self._last_fit_result = None  # Lazily built by get_fit_result()
                self.steps.append(("fit", model_obj))
                try:
                    score = model_obj.score(X, y)
                except Exception:
                    score = float("nan")
                ctx["r_squared"] = score
                self.history.append(("fit", model_name, score))
            except Exception as e:
                logger.error(
                    "Model fitting failed",
                    pipeline_id=self._id,
                    model=model_name,
                    error=str(e),
                    exc_info=True,
                )
                raise

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
        if _is_jax_array(X):
            X = np.array(X)

        logger.debug(
            "Generating predictions",
            pipeline_id=self._id,
            model=model.__class__.__name__,
            n_points=len(X),
        )

        predictions = model.predict(X)

        return RheoData(
            x=X,
            y=predictions,
            x_units=self.data.x_units if self.data else None,
            y_units=self.data.y_units if self.data else None,
            domain=self.data.domain if self.data else "time",
            metadata={
                **(
                    self.data.metadata
                    if (self.data and self.data.metadata is not None)
                    else {}
                ),
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

        with log_pipeline_stage(
            logger,
            "plot",
            pipeline_id=self._id,
            style=style,
            include_prediction=include_prediction,
        ) as ctx:
            from rheojax.visualization.plotter import plot_rheo_data

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
                ctx["prediction_overlay"] = True

            if show:
                import matplotlib.pyplot as plt

                plt.show()

            # Store figure for save_figure() method
            self._current_figure = fig

        self.history.append(("plot", style))
        return self

    def save(self, file_path: str | Path, format: str = "hdf5", **kwargs) -> Pipeline:
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

        path = Path(file_path)

        # R12-E-007: include fitted model parameters in data metadata before saving
        if self.steps:
            _last_fit_steps = [s for s in self.steps if s[0] in ("fit", "fit_nlsq")]
            if _last_fit_steps:
                _fit_model = _last_fit_steps[-1][1]
                if hasattr(_fit_model, "parameters"):
                    if self.data.metadata is None:
                        self.data.metadata = {}
                    for _pname in _fit_model.parameters.keys():
                        try:
                            self.data.metadata[f"fitted_{_pname}"] = float(
                                _fit_model.parameters.get_value(_pname)
                            )
                        except (TypeError, ValueError):
                            pass
                    self.data.metadata["fitted_model"] = type(_fit_model).__name__

        with log_pipeline_stage(
            logger,
            "save",
            pipeline_id=self._id,
            file_path=str(path),
            format=format,
        ) as ctx:
            try:
                if format == "hdf5":
                    from rheojax.io import save_hdf5

                    save_hdf5(self.data, path, **kwargs)
                elif format == "excel":
                    from rheojax.io import save_excel

                    # R13-PIPE-XLS-001: The "parameters" key should contain
                    # actual model parameters (name → value), not metadata
                    # labels. Data metadata (units, domain) goes into a
                    # separate "fit_quality" dict for the Fit Quality sheet.
                    parameters: dict[str, Any] = {}
                    _last_fit_steps = [
                        s for s in self.steps if s[0] in ("fit", "fit_nlsq")
                    ]
                    if _last_fit_steps:
                        _fit_model = _last_fit_steps[-1][1]
                        if hasattr(_fit_model, "parameters"):
                            for _pname in _fit_model.parameters.keys():
                                try:
                                    parameters[_pname] = float(
                                        _fit_model.parameters.get_value(_pname)
                                    )
                                except (TypeError, ValueError):
                                    pass
                            parameters["model"] = type(_fit_model).__name__

                    fit_quality: dict[str, Any] = {}
                    if self.data.x_units:
                        fit_quality["x_units"] = self.data.x_units
                    if self.data.y_units:
                        fit_quality["y_units"] = self.data.y_units
                    if self.data.domain:
                        fit_quality["domain"] = self.data.domain

                    excel_payload: dict[str, Any] = {
                        "x": np.array(self.data.x),
                        "predictions": np.array(self.data.y),
                    }
                    if parameters:
                        excel_payload["parameters"] = parameters
                    if fit_quality:
                        excel_payload["fit_quality"] = fit_quality
                    # R13-PIPE-XLS-002: propagate deformation_mode for
                    # correct E'/G' column labels in the Predictions sheet.
                    if self.data.metadata:
                        _dm = self.data.metadata.get("deformation_mode")
                        if _dm is not None:
                            excel_payload["deformation_mode"] = _dm
                    save_excel(excel_payload, path, **kwargs)
                elif format == "csv":
                    # R13-PIPE-CSV-001: Handle complex and 2D y arrays
                    # in CSV export. Complex y is split into real/imag
                    # columns; 2D y is split into numbered columns.
                    import pandas as pd

                    x_arr = np.array(self.data.x)
                    y_arr = np.array(self.data.y)
                    if np.iscomplexobj(y_arr):
                        df = pd.DataFrame(
                            {
                                "x": x_arr,
                                "y_real": np.real(y_arr),
                                "y_imag": np.imag(y_arr),
                            }
                        )
                    elif y_arr.ndim == 2:
                        cols: dict[str, Any] = {"x": x_arr}
                        for ci in range(y_arr.shape[1]):
                            cols[f"y_{ci}"] = y_arr[:, ci]
                        df = pd.DataFrame(cols)
                    else:
                        df = pd.DataFrame({"x": x_arr, "y": y_arr})
                    df.to_csv(path, index=False, **kwargs)
                else:
                    raise ValueError(f"Unknown format: {format}")

                ctx["n_points"] = len(self.data.x)
            except Exception as e:
                logger.error(
                    "Failed to save data",
                    pipeline_id=self._id,
                    file_path=str(path),
                    format=format,
                    error=str(e),
                    exc_info=True,
                )
                raise

        self.history.append(("save", str(path), format))
        return self

    def save_figure(
        self,
        filepath: str | Path,
        format: str | None = None,
        dpi: int = 300,
        **kwargs: Any,
    ) -> Pipeline:
        """
        Save the most recent plot to file.

        Convenience method for exporting plots with publication-quality defaults.
        Wraps rheojax.visualization.plotter.save_figure() to enable fluent API chaining.

        Parameters
        ----------
        filepath : str or Path
            Output file path. Format inferred from extension if not specified.
        format : str, optional
            Output format ('pdf', 'svg', 'png', 'eps'). If None, inferred from filepath.
        dpi : int, default=300
            Resolution for raster formats (PNG).
        **kwargs : dict
            Additional arguments passed to save_figure().
            See rheojax.visualization.plotter.save_figure() for details.

        Returns
        -------
        self : Pipeline
            Returns self to enable method chaining

        Raises
        ------
        ValueError
            If no plot exists (plot() not called yet)
        ValueError
            If format cannot be inferred or is unsupported
        OSError
            If filepath directory doesn't exist

        Examples
        --------
        Basic usage with method chaining:

        >>> pipeline = Pipeline()
        >>> pipeline.load('data.csv').fit('maxwell').plot().save_figure('result.pdf')

        Save multiple formats:

        >>> pipeline.plot(style='publication')
        >>> pipeline.save_figure('figure.pdf')
        >>> pipeline.save_figure('figure.png', dpi=600)
        >>> pipeline.save_figure('figure.svg', transparent=True)

        Explicit format:

        >>> pipeline.plot().save_figure('output', format='pdf')

        See Also
        --------
        plot : Generate plot with automatic type selection
        rheojax.visualization.plotter.save_figure : Core export function

        Notes
        -----
        This method saves the most recent plot generated by plot(). If you call plot()
        multiple times, only the last figure is saved. To save multiple plots, call
        save_figure() after each plot() call.

        The figure is stored internally by plot() and retrieved by save_figure().
        """
        if self._current_figure is None:
            raise ValueError(
                "No figure to save. Call plot() before save_figure(). "
                "Example: pipeline.load('data.csv').fit('maxwell').plot().save_figure('output.pdf')"
            )

        from rheojax.visualization.plotter import save_figure

        path = Path(filepath)

        with log_pipeline_stage(
            logger,
            "save_figure",
            pipeline_id=self._id,
            file_path=str(path),
            format=format,
            dpi=dpi,
        ) as ctx:
            try:
                save_figure(
                    self._current_figure, path, format=format, dpi=dpi, **kwargs
                )
                ctx["saved"] = True
            except Exception as e:
                logger.error(
                    "Failed to save figure",
                    pipeline_id=self._id,
                    file_path=str(path),
                    error=str(e),
                    exc_info=True,
                )
                raise

        self.history.append(("save_figure", str(path)))
        return self

    def fit_bayesian(
        self,
        model: str | BaseModel | None = None,
        seed: int | None = None,
        **bayesian_kwargs,
    ) -> Pipeline:
        """Run Bayesian (NUTS) inference on current data.

        Uses the last fitted model (or a new one) with NLSQ warm-start.

        Args:
            model: Model name, instance, or None to reuse last fitted model.
            seed: Random seed for reproducibility (default: 0).
            **bayesian_kwargs: Arguments forwarded to model.fit_bayesian()
                (num_warmup, num_samples, num_chains, target_accept_prob, etc.)

        Returns:
            self for method chaining

        Example:
            >>> pipeline.fit('maxwell').fit_bayesian(seed=42, num_warmup=1000)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        # Resolve model
        if model is not None:
            if isinstance(model, str):
                model_obj = ModelRegistry.create(model)
            else:
                model_obj = model
        elif self._last_model is not None:
            model_obj = self._last_model
        else:
            raise ValueError("No model available. Call fit() first or provide a model.")

        X = self.data.x
        y = self.data.y
        if _is_jax_array(X):
            X = np.array(X)
        if _is_jax_array(y):
            y = np.array(y)

        # Auto-propagate metadata
        _meta = getattr(self.data, "metadata", None) or {}
        if "test_mode" not in bayesian_kwargs and _meta.get("test_mode"):
            bayesian_kwargs["test_mode"] = _meta["test_mode"]
        if "deformation_mode" not in bayesian_kwargs and _meta.get("deformation_mode"):
            bayesian_kwargs["deformation_mode"] = _meta["deformation_mode"]
        if "poisson_ratio" not in bayesian_kwargs and _meta.get("poisson_ratio"):
            bayesian_kwargs["poisson_ratio"] = _meta["poisson_ratio"]

        if seed is not None:
            bayesian_kwargs["seed"] = seed

        with log_pipeline_stage(
            logger,
            "fit_bayesian",
            pipeline_id=self._id,
            model=model_obj.__class__.__name__,
        ) as ctx:
            try:
                result = model_obj.fit_bayesian(X, y, **bayesian_kwargs)
                self._last_bayesian_result = result
                self._last_model = model_obj
                # Store sampling kwargs on the model so BatchPipeline can
                # replay with the same configuration.  _last_fit_kwargs only
                # contains protocol kwargs from NLSQ — Bayesian sampling
                # params (num_warmup, num_samples, num_chains, seed) are
                # consumed by NumPyro and never stored there.
                _sampling_keys = {
                    "num_warmup", "num_samples", "num_chains", "seed",
                    "target_accept_prob",
                }
                model_obj._last_bayesian_kwargs = {
                    k: v for k, v in bayesian_kwargs.items()
                    if k in _sampling_keys
                }
                self.steps.append(("fit_bayesian", model_obj))
                self.history.append(("fit_bayesian", model_obj.__class__.__name__))
                ctx["num_samples"] = getattr(result, "num_samples", None)
                ctx["num_chains"] = getattr(result, "num_chains", None)
            except Exception as e:
                logger.error(
                    "Bayesian fitting failed",
                    pipeline_id=self._id,
                    error=str(e),
                    exc_info=True,
                )
                raise

        return self

    def plot_fit(
        self,
        confidence: float = 0.95,
        show_residuals: bool = True,
        show_uncertainty: bool = True,
        show: bool = True,
        style: str = "default",
        **kwargs,
    ) -> Pipeline:
        """Plot NLSQ fit with uncertainty band and residuals.

        Requires a prior call to fit(). Uses FitPlotter internally.

        Args:
            confidence: Confidence level for uncertainty band (default: 0.95).
            show_residuals: If True, add residuals subplot.
            show_uncertainty: If True and covariance available, show band.
            show: Whether to call plt.show() (default: True).
            style: Plot style ('default', 'publication', 'presentation').
            **kwargs: Additional arguments forwarded to FitPlotter.plot_nlsq().

        Returns:
            self for method chaining

        Example:
            >>> pipeline.fit('maxwell').plot_fit(confidence=0.95)
        """
        if self._last_model is None:
            raise ValueError("No model fitted. Call fit() first.")
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        from rheojax.visualization.fit_plotter import FitPlotter

        fit_result = self.get_fit_result()
        plotter = FitPlotter()

        X = np.array(self.data.x) if _is_jax_array(self.data.x) else np.asarray(self.data.x)
        y = np.array(self.data.y) if _is_jax_array(self.data.y) else np.asarray(self.data.y)

        # Forward deformation_mode from metadata
        _meta = getattr(self.data, "metadata", None) or {}
        if "deformation_mode" not in kwargs:
            dm = _meta.get("deformation_mode")
            if dm is not None:
                kwargs["deformation_mode"] = dm
        if "test_mode" not in kwargs:
            tm = _meta.get("test_mode")
            if tm is not None:
                kwargs["test_mode"] = tm

        fig, axes = plotter.plot_nlsq(
            X, y, fit_result, self._last_model,
            confidence=confidence,
            show_residuals=show_residuals,
            show_uncertainty=show_uncertainty,
            style=style,
            **kwargs,
        )

        self._current_figure = fig

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        self.history.append(("plot_fit", style))
        return self

    def plot_bayesian(
        self,
        credible_level: float = 0.95,
        max_draws: int = 500,
        show_nlsq_overlay: bool = False,
        show_residuals: bool = False,
        show: bool = True,
        style: str = "default",
        **kwargs,
    ) -> Pipeline:
        """Plot Bayesian posterior predictive with credible interval.

        Requires a prior call to fit_bayesian().

        Args:
            credible_level: Credible interval level (default: 0.95).
            max_draws: Maximum posterior draws for band computation.
            show_nlsq_overlay: If True, overlay NLSQ fit for comparison.
            show_residuals: If True, add residuals subplot.
            show: Whether to call plt.show() (default: True).
            style: Plot style.
            **kwargs: Additional arguments forwarded to FitPlotter.plot_bayesian().

        Returns:
            self for method chaining

        Example:
            >>> pipeline.fit('maxwell').fit_bayesian(seed=42).plot_bayesian()
        """
        if self._last_bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")
        if self._last_model is None:
            raise ValueError("No model available.")
        if self.data is None:
            raise ValueError("No data loaded.")

        from rheojax.visualization.fit_plotter import FitPlotter

        plotter = FitPlotter()
        X = np.array(self.data.x) if _is_jax_array(self.data.x) else np.asarray(self.data.x)
        y = np.array(self.data.y) if _is_jax_array(self.data.y) else np.asarray(self.data.y)

        # Forward metadata
        _meta = getattr(self.data, "metadata", None) or {}
        if "deformation_mode" not in kwargs:
            dm = _meta.get("deformation_mode")
            if dm is not None:
                kwargs["deformation_mode"] = dm
        if "test_mode" not in kwargs:
            tm = _meta.get("test_mode")
            if tm is not None:
                kwargs["test_mode"] = tm

        fit_result = None
        if show_nlsq_overlay:
            try:
                fit_result = self.get_fit_result()
            except ValueError:
                pass

        fig, axes = plotter.plot_bayesian(
            X, y, self._last_bayesian_result, self._last_model,
            credible_level=credible_level,
            max_draws=max_draws,
            show_nlsq_overlay=show_nlsq_overlay,
            fit_result=fit_result,
            show_residuals=show_residuals,
            style=style,
            **kwargs,
        )

        self._current_figure = fig

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        self.history.append(("plot_bayesian", style))
        return self

    def plot_diagnostics(
        self,
        output_dir: str | Path | None = None,
        style: str = "default",
        prefix: str = "mcmc",
        formats: tuple[str, ...] = ("pdf", "png"),
        dpi: int = 300,
        **kwargs,
    ) -> Pipeline:
        """Generate ArviZ MCMC diagnostic suite (6 plots).

        Requires a prior call to fit_bayesian().

        Args:
            output_dir: Directory for saving plots. If None, displays only.
            style: Plot style.
            prefix: Filename prefix for saved plots.
            formats: Output formats (default: ('pdf', 'png')).
            dpi: Resolution for raster formats.
            **kwargs: Additional arguments forwarded to generate_diagnostic_suite().

        Returns:
            self for method chaining

        Example:
            >>> pipeline.fit_bayesian(seed=42).plot_diagnostics(output_dir='./diag')
        """
        if self._last_bayesian_result is None:
            raise ValueError("No Bayesian result available. Call fit_bayesian() first.")

        from rheojax.visualization.fit_plotter import generate_diagnostic_suite

        result = generate_diagnostic_suite(
            self._last_bayesian_result,
            style=style,
            output_dir=output_dir,
            prefix=prefix,
            formats=formats,
            dpi=dpi,
            **kwargs,
        )

        self._diagnostic_results = result

        # Expose the first diagnostic figure for save_figure() chaining.
        # generate_diagnostic_suite returns dict[str, Figure | Path].
        if isinstance(result, dict):
            for fig_or_path in result.values():
                if hasattr(fig_or_path, "savefig"):
                    self._current_figure = fig_or_path
                    break

        self.history.append(("plot_diagnostics", str(output_dir)))
        return self

    def plot_transform(
        self,
        transform_name: str | None = None,
        show_intermediate: bool = True,
        show: bool = True,
        style: str = "default",
        **kwargs,
    ) -> Pipeline:
        """Plot the result of a previously applied transform.

        Uses TransformPlotter for per-transform layout dispatch.

        Args:
            transform_name: Name of the transform to plot. If None, uses the
                most recently applied transform.
            show_intermediate: Whether to show before/after comparison.
            show: Whether to call plt.show() (default: True).
            style: Plot style.
            **kwargs: Additional arguments forwarded to TransformPlotter.

        Returns:
            self for method chaining

        Example:
            >>> pipeline.transform('mastercurve', reference_temp=25.0).plot_transform()
        """
        from rheojax.visualization.transform_plotter import TransformPlotter

        if transform_name is None:
            transform_name = self._last_transform_name

        if transform_name is None or transform_name not in self._transform_results:
            available = list(self._transform_results.keys())
            raise ValueError(
                f"No cached result for transform '{transform_name}'. "
                f"Available transforms: {available}. "
                "Call transform() before plot_transform()."
            )

        cached_result, pre_data = self._transform_results[transform_name]
        plotter = TransformPlotter()

        fig, axes = plotter.plot(
            transform_name,
            cached_result,
            input_data=pre_data if show_intermediate else None,
            show_intermediate=show_intermediate,
            style=style,
            **kwargs,
        )

        self._current_figure = fig

        if show:
            import matplotlib.pyplot as plt

            plt.show()

        self.history.append(("plot_transform", transform_name, style))
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

    def get_history(self) -> list[tuple[Any, ...]]:
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
        return [step[1] for step in self.steps if step[0] in ("fit", "fit_nlsq")]

    def get_fitted_parameters(self) -> dict[str, float]:
        """Get fitted parameters from the last model as a dictionary.

        This is a convenience method that extracts parameter values from
        the last fitted model's ParameterSet.

        Returns:
            Dictionary mapping parameter names to their fitted values

        Raises:
            ValueError: If no model has been fitted yet

        Example:
            >>> pipeline = Pipeline()
            >>> pipeline.load('data.csv').fit('maxwell')
            >>> params = pipeline.get_fitted_parameters()
            >>> print(params)  # {'G0': 100000.0, 'eta': 1000.0}
            >>> G0 = params['G0']
        """
        if self._last_model is None:
            raise ValueError("No model fitted. Call fit() first.")

        # Extract all parameter values from the model's ParameterSet
        return {
            name: self._last_model.parameters.get_value(name)
            for name in self._last_model.parameters.keys()
        }

    def compare_models(
        self,
        models: list[str | BaseModel],
        criterion: str = "aic",
        **fit_kwargs,
    ) -> Pipeline:
        """Compare multiple models on the current data.

        Fits each model and ranks by information criterion.  The best model
        becomes ``_last_model`` and is appended to ``steps``.

        Args:
            models: List of model names (strings) or BaseModel instances.
            criterion: Ranking criterion ('aic', 'aicc', 'bic').
            **fit_kwargs: Extra kwargs forwarded to each ``model.fit()`` call.

        Returns:
            self for method chaining

        Raises:
            ValueError: If no data is loaded.

        Example:
            >>> pipeline.load('data.csv').compare_models(['maxwell', 'zener'])
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        from rheojax.utils.model_selection import compare_models as _compare

        X = self.data.x
        y = self.data.y

        if _is_jax_array(X):
            X = np.array(X)
        if _is_jax_array(y):
            y = np.array(y)

        # Auto-propagate metadata
        _meta = getattr(self.data, "metadata", None) or {}
        if "test_mode" not in fit_kwargs and _meta.get("test_mode"):
            fit_kwargs["test_mode"] = _meta["test_mode"]
        if "deformation_mode" not in fit_kwargs and _meta.get("deformation_mode"):
            fit_kwargs["deformation_mode"] = _meta["deformation_mode"]

        test_mode = fit_kwargs.pop("test_mode", None)

        comparison = _compare(
            X, y,
            models=models,
            test_mode=test_mode,
            criterion=criterion,
            **fit_kwargs,
        )

        self._last_comparison = comparison
        self.history.append(("compare_models", comparison.best_model, criterion))

        # Set the best model as _last_model if available — reuse the
        # already-fitted instance from compare_models() instead of re-fitting.
        if comparison.results:
            best_fr = next(
                (r for r in comparison.results if r.model_name == comparison.best_model),
                None,
            )
            fitted_model = getattr(best_fr, "_fitted_model", None) if best_fr else None
            if fitted_model is not None:
                self._last_model = fitted_model
                self.steps.append(("compare_models", fitted_model))
            else:
                logger.warning(
                    "Best model FitResult has no attached fitted model",
                    model=comparison.best_model,
                )

        return self

    def get_fit_result(self) -> Any:
        """Construct a FitResult from the last fitted model.

        Returns:
            FitResult with model metadata, fitted parameters, and statistics.

        Raises:
            ValueError: If no model has been fitted.

        Example:
            >>> result = pipeline.load('data.csv').fit('maxwell').get_fit_result()
            >>> print(result.summary())
        """
        if self._last_model is None:
            raise ValueError("No model fitted. Call fit() first.")

        from rheojax.utils.model_selection import build_fit_result

        X = self.data.x if self.data is not None else None
        y = self.data.y if self.data is not None else None
        test_mode = None
        if self.data is not None:
            _meta = getattr(self.data, "metadata", None) or {}
            test_mode = _meta.get("test_mode")

        return build_fit_result(
            self._last_model,
            X,
            y,
            test_mode=test_mode,
        )

    def clone(self) -> Pipeline:
        """Create a copy of the pipeline.

        Returns:
            New Pipeline with copied data and history

        Example:
            >>> pipeline2 = pipeline.clone()
        """
        new_pipeline = Pipeline(data=self.data.copy() if self.data else None)
        new_pipeline.steps = copy.deepcopy(self.steps)
        new_pipeline.history = self.history.copy()
        new_pipeline._last_model = (
            copy.deepcopy(self._last_model) if self._last_model is not None else None
        )
        logger.debug(
            "Pipeline cloned",
            original_id=self._id,
            new_id=new_pipeline._id,
        )
        return new_pipeline

    def reset(self) -> Pipeline:
        """Reset pipeline to initial state.

        Returns:
            self for method chaining

        Example:
            >>> pipeline.reset()
        """
        logger.debug("Pipeline reset", pipeline_id=self._id)
        self.data = None
        self.steps = []
        self.history = []
        self._last_model = None
        self._last_fit_result = None
        self._last_bayesian_result = None
        self._transform_results = {}
        self._last_transform_name = None
        self._current_figure = None
        self._diagnostic_results = None
        self._last_comparison = None
        return self

    def export(
        self,
        output: str | Path,
        format: str = "auto",
        *,
        include_data: bool = True,
        include_figures: bool = True,
        include_diagnostics: bool = True,
        figure_formats: tuple[str, ...] = ("pdf", "png"),
        figure_dpi: int = 300,
        **kwargs,
    ) -> Pipeline:
        """Export the full analysis to a directory or file.

        This bundles data, parameters, statistics, figures, transform results,
        and Bayesian diagnostics into a single export.

        Args:
            output: Output path. If a directory (no extension or trailing /),
                exports as structured directory. If .xlsx, exports Excel.
            format: Export format ('auto', 'directory', 'excel').
                'auto' infers from the output path extension.
            include_data: Save raw and transformed data files.
            include_figures: Save generated matplotlib figures.
            include_diagnostics: Save MCMC diagnostic plots.
            figure_formats: Formats for figure files (default: ('pdf', 'png')).
            figure_dpi: Resolution for raster figures (default: 300).
            **kwargs: Additional arguments forwarded to the exporter.

        Returns:
            self for method chaining

        Example:
            >>> pipeline.load('data.csv').fit('maxwell').plot_fit().export('./results')
            >>> pipeline.export('report.xlsx')
        """
        from rheojax.io.analysis_exporter import AnalysisExporter

        output_path = Path(output)
        exporter = AnalysisExporter(
            figure_formats=figure_formats,
            figure_dpi=figure_dpi,
        )

        # Determine format
        if format == "auto":
            if output_path.suffix.lower() == ".xlsx":
                format = "excel"
            else:
                format = "directory"

        with log_pipeline_stage(
            logger,
            "export",
            pipeline_id=self._id,
            output=str(output_path),
            format=format,
        ) as ctx:
            try:
                if format == "directory":
                    exporter.export_directory(
                        self,
                        output_path,
                        include_data=include_data,
                        include_figures=include_figures,
                        include_diagnostics=include_diagnostics,
                        **kwargs,
                    )
                elif format == "excel":
                    exporter.export_excel(
                        self,
                        output_path,
                        include_plots=include_figures,
                        **kwargs,
                    )
                else:
                    raise ValueError(
                        f"Unknown export format: {format}. Use 'directory' or 'excel'."
                    )
                ctx["format"] = format
            except Exception as e:
                logger.error(
                    "Export failed",
                    pipeline_id=self._id,
                    output=str(output_path),
                    error=str(e),
                    exc_info=True,
                )
                raise

        self.steps.append(("export", {"output_path": str(output_path), "format": format}))
        self.history.append(("export", str(output_path), format))
        return self

    def __repr__(self) -> str:
        """String representation of pipeline."""
        n_steps = len(self.history)
        has_data = self.data is not None
        has_model = self._last_model is not None
        return f"Pipeline(steps={n_steps}, has_data={has_data}, has_model={has_model})"


__all__ = ["Pipeline"]
