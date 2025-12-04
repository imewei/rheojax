"""Base classes for models and transforms with JAX support.

This module provides abstract base classes that define consistent interfaces
for all models and transforms in the rheo package, with full JAX support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from rheojax.core.bayesian import BayesianMixin, BayesianResult
from rheojax.core.jax_config import safe_import_jax
from rheojax.core.parameters import Parameter, ParameterSet

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()

# Type alias for arrays (accepts both NumPy and JAX arrays)
# Note: jnp.ndarray is dynamically imported, so we use np.ndarray for type checking
type ArrayLike = np.ndarray


class BaseModel(BayesianMixin, ABC):
    """Abstract base class for all rheological models.

    This class defines the standard interface that all models must implement,
    supporting JAX arrays, multiple API styles (fluent, scikit-learn, piblin),
    and Bayesian inference via NumPyro NUTS.

    All models inherit Bayesian capabilities from BayesianMixin, including:
    - fit_bayesian(): Bayesian parameter estimation using NUTS
    - sample_prior(): Sample from prior distributions
    - get_credible_intervals(): Compute highest density intervals

    The fit() method uses NLSQ optimization by default for fast point estimation,
    which can be used to warm-start Bayesian inference.
    """

    def __init__(self):
        """Initialize base model."""
        self.parameters = ParameterSet()
        self.fitted_ = False
        self._nlsq_result = None  # Store NLSQ optimization result
        self._bayesian_result = None  # Store Bayesian inference result
        self.X_data = None  # Store data for Bayesian inference
        self.y_data = None

    @abstractmethod
    def _fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> BaseModel:
        """Internal fit implementation to be overridden by subclasses.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting options

        Returns:
            self for method chaining
        """
        pass

    @abstractmethod
    def _predict(self, X: ArrayLike) -> ArrayLike:
        """Internal predict implementation to be overridden by subclasses.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        method: str = "nlsq",
        check_compatibility: bool = False,
        use_log_residuals: bool | None = None,
        use_multi_start: bool | None = None,
        n_starts: int = 5,
        perturb_factor: float = 0.3,
        **kwargs,
    ) -> BaseModel:
        """Fit the model to data using NLSQ optimization.

        This method uses NLSQ (GPU-accelerated nonlinear least squares) by default
        for fast point estimation. The optimization result is stored for potential
        warm-starting of Bayesian inference.

        For very wide frequency ranges (>10 decades), multi-start optimization is
        automatically enabled to escape local minima.

        Args:
            X: Input features
            y: Target values
            method: Optimization method ('nlsq' by default for compatibility)
            check_compatibility: Whether to check model-data compatibility before
                fitting. If True, warns when model may not be appropriate for data.
                Default is False for backward compatibility.
            use_log_residuals: Whether to use log-space residuals for fitting.
                Recommended for wide frequency ranges (>8 decades) to prevent
                optimizer bias. If None (default), automatically detected based
                on data range. Explicit True/False overrides auto-detection.
            use_multi_start: Whether to use multi-start optimization to escape
                local minima. Recommended for very wide ranges (>10 decades).
                If None (default), automatically enabled for >10 decades.
            n_starts: Number of random starts for multi-start optimization (default: 5)
            perturb_factor: Perturbation magnitude for multi-start random starts (default: 0.3).
                Parameters are perturbed by ± perturb_factor * (value or range).
                Larger values (0.7-0.9) explore wider parameter space.
            **kwargs: Additional fitting options passed to _fit()

        Returns:
            self for method chaining (scikit-learn style)

        Example:
            >>> model = Maxwell()
            >>> model.fit(t, G_data)  # Uses NLSQ by default
            >>> model.fit(t, G_data, method='nlsq', max_iter=1000)
            >>> model.fit(t, G_data, check_compatibility=True)  # Check compatibility
            >>> model.fit(omega, G_star, use_log_residuals=True)  # Force log-residuals
            >>> model.fit(mastercurve, None, use_multi_start=True, n_starts=10)  # Multi-start
        """
        import logging

        # Store data for potential Bayesian inference
        self.X_data = X
        self.y_data = y

        # Auto-detect wide frequency ranges and configure optimization strategy
        if use_log_residuals is None or use_multi_start is None:
            try:
                from rheojax.core.data import RheoData
                from rheojax.utils.data_quality import detect_data_range_decades

                # Extract x array from RheoData if needed
                if isinstance(X, RheoData):
                    x_array = X.x
                else:
                    x_array = X

                decades = detect_data_range_decades(x_array)

                # Auto-enable log-residuals for wide ranges (>8 decades)
                if use_log_residuals is None:
                    if decades > 8.0:
                        use_log_residuals = True
                        logging.info(
                            f"Auto-enabling log-residuals for wide range ({decades:.1f} decades)"
                        )
                    else:
                        use_log_residuals = False

                # Auto-enable multi-start for very wide ranges (>10 decades)
                if use_multi_start is None:
                    if decades > 10.0:
                        use_multi_start = True
                        logging.info(
                            f"Auto-enabling multi-start optimization for very wide range "
                            f"({decades:.1f} decades, {n_starts} starts)"
                        )
                    else:
                        use_multi_start = False

            except Exception as e:
                logging.debug(f"Range detection failed: {e}")
                if use_log_residuals is None:
                    use_log_residuals = False
                if use_multi_start is None:
                    use_multi_start = False

        # Pass optimization strategy to _fit via kwargs
        kwargs["use_log_residuals"] = use_log_residuals
        kwargs["use_multi_start"] = use_multi_start
        kwargs["n_starts"] = n_starts
        kwargs["perturb_factor"] = perturb_factor

        # Optional compatibility check
        if check_compatibility:
            try:
                from rheojax.utils.compatibility import (
                    check_model_compatibility,
                    format_compatibility_message,
                )

                # Determine test mode if not provided
                test_mode = kwargs.get("test_mode", None)

                # Check compatibility
                compatibility = check_model_compatibility(
                    model=self,
                    t=X if test_mode == "relaxation" else None,
                    G_t=y if test_mode == "relaxation" else None,
                    omega=X if test_mode == "oscillation" else None,
                    G_star=y if test_mode == "oscillation" else None,
                    test_mode=test_mode,
                )

                # Log compatibility results
                if not compatibility["compatible"]:
                    message = format_compatibility_message(compatibility)
                    logging.warning(f"Model compatibility check:\n{message}")

            except Exception as e:
                logging.debug(f"Compatibility check failed: {e}")

        # Call subclass implementation (which uses NLSQ via optimization module)
        try:
            self._fit(X, y, method=method, **kwargs)
            self.fitted_ = True
        except RuntimeError as e:
            # Enhance error message with compatibility information
            error_msg = str(e)

            # Check if this is an optimization failure
            if "Optimization failed" in error_msg or "did not converge" in error_msg:
                # Try to provide more context
                try:
                    from rheojax.utils.compatibility import (
                        check_model_compatibility,
                        format_compatibility_message,
                    )

                    test_mode = kwargs.get("test_mode", None)
                    compatibility = check_model_compatibility(
                        model=self,
                        t=X if test_mode == "relaxation" else None,
                        G_t=y if test_mode == "relaxation" else None,
                        omega=X if test_mode == "oscillation" else None,
                        G_star=y if test_mode == "oscillation" else None,
                        test_mode=test_mode,
                    )

                    if not compatibility["compatible"]:
                        # Provide enhanced error message
                        compat_msg = format_compatibility_message(compatibility)
                        enhanced_msg = (
                            f"{error_msg}\n\n"
                            f"Model-data compatibility issue detected:\n"
                            f"{compat_msg}\n\n"
                            f"Note: This model may not be appropriate for your data. "
                            f"In model comparison pipelines, it's normal for some models "
                            f"to fail when their underlying physics doesn't match the material behavior."
                        )
                        raise RuntimeError(enhanced_msg) from e
                except Exception:
                    # If compatibility check fails, just raise original error
                    pass

            # Re-raise original error if not enhanced
            raise

        # Note: _nlsq_result would be set by subclass _fit implementation
        # if it explicitly stores the OptimizationResult
        return self

    def fit_bayesian(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        num_warmup: int = 1000,
        num_samples: int = 2000,
        num_chains: int = 1,
        initial_values: dict[str, float] | None = None,
        test_mode: str | None = None,
        **nuts_kwargs,
    ) -> BayesianResult:
        """Perform Bayesian inference using NumPyro NUTS sampler.

        This method delegates to BayesianMixin.fit_bayesian() to run NUTS sampling
        for Bayesian parameter estimation. If initial_values is not provided and
        the model has been previously fitted with fit(), the NLSQ point estimates
        are automatically used for warm-starting.

        Args:
            X: Independent variable data (input features) or RheoData object
            y: Dependent variable data (observations to fit). If X is RheoData,
                y is ignored and extracted from X.
            num_warmup: Number of warmup/burn-in iterations (default: 1000)
            num_samples: Number of posterior samples to collect (default: 2000)
            num_chains: Number of MCMC chains (default: 1)
            initial_values: Optional dict of initial parameter values for
                warm-start. If None and model is fitted, uses NLSQ estimates.
            test_mode: Explicit test mode (e.g., 'relaxation', 'creep', 'oscillation').
                If None, inferred from RheoData.metadata['test_mode'] or defaults
                to 'relaxation'. Overrides RheoData metadata if provided.
            **nuts_kwargs: Additional arguments passed to NUTS sampler

        Returns:
            BayesianResult containing posterior samples, summary statistics,
            and convergence diagnostics (R-hat, ESS, divergences)

        Example:
            >>> model = Maxwell()
            >>> # Warm-start from NLSQ with explicit mode
            >>> model.fit(t, G_data, test_mode='relaxation')  # NLSQ optimization
            >>> result = model.fit_bayesian(t, G_data, test_mode='relaxation')
            >>>
            >>> # RheoData with embedded mode (recommended)
            >>> rheo_data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})
            >>> result = model.fit_bayesian(rheo_data)
            >>>
            >>> # Or provide explicit initial values
            >>> result = model.fit_bayesian(
            ...     t, G_data,
            ...     initial_values={'G0': 1e5, 'eta': 1e3},
            ...     test_mode='creep'
            ... )
        """
        # Store data for model_function access
        self.X_data = X
        self.y_data = y

        # Auto warm-start from NLSQ if available and no explicit initial values
        if initial_values is None and self.fitted_:
            # Extract current parameter values as initial values
            initial_values = {
                name: self.parameters.get_value(name) for name in self.parameters
            }

        # Call BayesianMixin implementation with test_mode parameter
        result = super().fit_bayesian(
            X,
            y,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            initial_values=initial_values,
            test_mode=test_mode,
            **nuts_kwargs,
        )

        # Store result for later access
        self._bayesian_result = result

        return result

    def predict(self, X: ArrayLike, test_mode: str | None = None) -> ArrayLike:
        """Make predictions.

        Args:
            X: Input features
            test_mode: Optional test mode ('oscillation', 'relaxation', 'creep', 'flow').
                      If provided, sets model's test_mode before prediction.
                      Useful for data generation without fitting.

        Returns:
            Model predictions
        """
        if not self.fitted_ and len(self.parameters) > 0:
            # Check if we have parameters set manually
            if not any(p.value is None for p in self.parameters._parameters.values()):
                # Parameters are set, consider it fitted
                self.fitted_ = True

        # Set test_mode if provided (for data generation without fitting)
        if test_mode is not None and hasattr(self, "_test_mode"):
            self._test_mode = test_mode

        return self._predict(X)

    def fit_predict(self, X: ArrayLike, y: ArrayLike, **kwargs) -> ArrayLike:
        """Fit model and return predictions.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting options

        Returns:
            Model predictions on training data
        """
        self.fit(X, y, **kwargs)
        return self.predict(X)

    def get_nlsq_result(self):
        """Get stored NLSQ optimization result.

        Returns:
            OptimizationResult from NLSQ fit, or None if not fitted

        Example:
            >>> model.fit(t, G_data)
            >>> result = model.get_nlsq_result()
            >>> if result:
            ...     print(f"Converged: {result.success}")
        """
        return self._nlsq_result

    def get_bayesian_result(self) -> BayesianResult | None:
        """Get stored Bayesian inference result.

        Returns:
            BayesianResult from fit_bayesian(), or None if not run

        Example:
            >>> model.fit_bayesian(t, G_data)
            >>> result = model.get_bayesian_result()
            >>> print(result.diagnostics['r_hat'])
        """
        return self._bayesian_result

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters.

        Args:
            deep: If True, return parameters of sub-objects

        Returns:
            Dictionary of parameter names and values
        """
        if hasattr(self, "parameters") and self.parameters:
            return self.parameters.to_dict()
        return {}

    def set_params(self, **params) -> BaseModel:
        """Set model parameters.

        Args:
            **params: Parameter names and values

        Returns:
            self for method chaining
        """
        if hasattr(self, "parameters"):
            for name, value in params.items():
                if name in self.parameters:
                    self.parameters.set_value(name, value)
        return self

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Compute model score (R² by default).

        Args:
            X: Input features
            y: True target values

        Returns:
            Model score (R² coefficient)
        """
        predictions = self.predict(X)

        # Convert to numpy for scoring
        if isinstance(predictions, jnp.ndarray):
            predictions = np.array(predictions)
        if isinstance(y, jnp.ndarray):
            y = np.array(y)

        # Compute R² score
        # For complex data (e.g., oscillatory shear), use magnitude of residuals
        if np.iscomplexobj(y) or np.iscomplexobj(predictions):
            ss_res = np.sum(np.abs(y - predictions) ** 2)
            ss_tot = np.sum(np.abs(y - np.mean(y)) ** 2)
        else:
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

        # Handle edge cases
        if ss_tot == 0:
            # All y values are the same
            return 1.0 if ss_res == 0 else 0.0

        # Handle NaN case
        r2 = 1 - (ss_res / ss_tot)
        if np.isnan(r2):
            return 0.0

        return float(np.real(r2))

    def to_dict(self) -> dict[str, Any]:
        """Serialize model to dictionary.

        Returns:
            Dictionary representation of model
        """
        return {
            "class": self.__class__.__name__,
            "parameters": self.parameters.to_dict() if self.parameters else {},
            "fitted": self.fitted_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseModel:
        """Create model from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Model instance
        """
        model = cls()
        if "parameters" in data:
            model.parameters = ParameterSet.from_dict(data["parameters"])
        model.fitted_ = data.get("fitted", False)
        return model

    def __repr__(self) -> str:
        """String representation of model."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class BaseTransform(ABC):
    """Abstract base class for all data transforms.

    This class defines the standard interface that all transforms must implement,
    supporting JAX arrays and composable transformations.
    """

    def __init__(self):
        """Initialize base transform."""
        self.fitted_ = False

    @abstractmethod
    def _transform(self, data: ArrayLike) -> ArrayLike:
        """Internal transform implementation to be overridden by subclasses.

        Args:
            data: Input data to transform

        Returns:
            Transformed data
        """
        pass

    def _inverse_transform(self, data: ArrayLike) -> ArrayLike:
        """Internal inverse transform implementation.

        Args:
            data: Transformed data

        Returns:
            Original data

        Raises:
            NotImplementedError: If inverse transform not available
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse transform"
        )

    def transform(self, data: ArrayLike) -> ArrayLike:
        """Transform the data.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        return self._transform(data)

    def inverse_transform(self, data: ArrayLike) -> ArrayLike:
        """Apply inverse transformation.

        Args:
            data: Transformed data

        Returns:
            Original data
        """
        return self._inverse_transform(data)

    def fit(self, data: ArrayLike) -> BaseTransform:
        """Fit the transform to data (learn parameters if needed).

        Args:
            data: Training data

        Returns:
            self for method chaining
        """
        # Default implementation does nothing (stateless transform)
        self.fitted_ = True
        return self

    def fit_transform(self, data: ArrayLike) -> ArrayLike:
        """Fit to data and transform it.

        Args:
            data: Input data

        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)

    def __add__(self, other: BaseTransform) -> TransformPipeline:
        """Compose transforms using + operator.

        Args:
            other: Another transform to compose

        Returns:
            Pipeline of transforms
        """
        if isinstance(other, TransformPipeline):
            return TransformPipeline([self] + other.transforms)
        elif isinstance(other, BaseTransform):
            return TransformPipeline([self, other])
        else:
            raise TypeError(f"Cannot compose with {type(other)}")

    def __repr__(self) -> str:
        """String representation of transform."""
        return f"{self.__class__.__name__}()"


class TransformPipeline(BaseTransform):
    """Pipeline of multiple transforms applied sequentially."""

    def __init__(self, transforms: list[BaseTransform]):
        """Initialize transform pipeline.

        Args:
            transforms: List of transforms to apply in order
        """
        super().__init__()
        self.transforms = transforms

    def _transform(self, data: ArrayLike) -> ArrayLike:
        """Apply all transforms in sequence.

        Args:
            data: Input data

        Returns:
            Transformed data after all transforms
        """
        result = data
        for transform in self.transforms:
            result = transform.transform(result)
        return result

    def _inverse_transform(self, data: ArrayLike) -> ArrayLike:
        """Apply inverse transforms in reverse order.

        Args:
            data: Transformed data

        Returns:
            Original data
        """
        result = data
        for transform in reversed(self.transforms):
            result = transform.inverse_transform(result)
        return result

    def fit(self, data: ArrayLike) -> TransformPipeline:
        """Fit all transforms in the pipeline.

        Args:
            data: Training data

        Returns:
            self for method chaining
        """
        current_data = data
        for transform in self.transforms:
            current_data = transform.fit_transform(current_data)
        self.fitted_ = True
        return self

    def __repr__(self) -> str:
        """String representation of pipeline."""
        transform_names = " → ".join(t.__class__.__name__ for t in self.transforms)
        return f"TransformPipeline([{transform_names}])"


__all__ = [
    "BaseModel",
    "BaseTransform",
    "TransformPipeline",
    "Parameter",
    "ParameterSet",
]
