"""Base classes for models and transforms with JAX support.

This module provides abstract base classes that define consistent interfaces
for all models and transforms in the rheo package, with full JAX support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import jax.numpy as jnp

from .parameters import ParameterSet


ArrayLike = Union[np.ndarray, jnp.ndarray]


class BaseModel(ABC):
    """Abstract base class for all rheological models.

    This class defines the standard interface that all models must implement,
    supporting JAX arrays and multiple API styles (fluent, scikit-learn, piblin).
    """

    def __init__(self):
        """Initialize base model."""
        self.parameters = ParameterSet()
        self.fitted_ = False

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

    def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> BaseModel:
        """Fit the model to data.

        Args:
            X: Input features
            y: Target values
            **kwargs: Additional fitting options

        Returns:
            self for method chaining (scikit-learn style)
        """
        self._fit(X, y, **kwargs)
        self.fitted_ = True
        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        if not self.fitted_ and len(self.parameters) > 0:
            # Check if we have parameters set manually
            if not any(p.value is None for p in self.parameters._parameters.values()):
                # Parameters are set, consider it fitted
                self.fitted_ = True

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

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get model parameters.

        Args:
            deep: If True, return parameters of sub-objects

        Returns:
            Dictionary of parameter names and values
        """
        if hasattr(self, 'parameters') and self.parameters:
            return self.parameters.to_dict()
        return {}

    def set_params(self, **params) -> BaseModel:
        """Set model parameters.

        Args:
            **params: Parameter names and values

        Returns:
            self for method chaining
        """
        if hasattr(self, 'parameters'):
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

        return float(r2)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary.

        Returns:
            Dictionary representation of model
        """
        return {
            "class": self.__class__.__name__,
            "parameters": self.parameters.to_dict() if self.parameters else {},
            "fitted": self.fitted_
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BaseModel:
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
        raise NotImplementedError(f"{self.__class__.__name__} does not support inverse transform")

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


# Import Parameter and ParameterSet for convenience
from .parameters import Parameter, ParameterSet

__all__ = [
    'BaseModel',
    'BaseTransform',
    'TransformPipeline',
    'Parameter',
    'ParameterSet'
]