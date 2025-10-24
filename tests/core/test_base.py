"""Tests for base model and transform classes.

This test suite ensures that BaseModel and BaseTransform provide
consistent interfaces with JAX support and proper parameter management.
"""

import numpy as np
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

from rheo.core.base import BaseModel, BaseTransform, ParameterSet, Parameter


class TestParameterClass:
    """Test Parameter class for parameter management."""

    def test_create_parameter(self):
        """Test creating a parameter."""
        param = Parameter(
            name="modulus",
            value=100.0,
            bounds=(0.0, 1e6),
            units="Pa",
            description="Storage modulus"
        )

        assert param.name == "modulus"
        assert param.value == 100.0
        assert param.bounds == (0.0, 1e6)
        assert param.units == "Pa"
        assert param.description == "Storage modulus"

    def test_parameter_validation(self):
        """Test parameter value validation."""
        param = Parameter(name="test", value=50.0, bounds=(0.0, 100.0))

        # Valid value
        param.value = 75.0
        assert param.value == 75.0

        # Invalid value (out of bounds)
        with pytest.raises(ValueError, match="out of bounds"):
            param.value = 150.0

    def test_parameter_without_bounds(self):
        """Test parameter without bounds."""
        param = Parameter(name="test", value=1000.0)

        # Should accept any value
        param.value = -500.0
        assert param.value == -500.0

        param.value = 1e10
        assert param.value == 1e10

    def test_parameter_to_dict(self):
        """Test parameter serialization to dict."""
        param = Parameter(
            name="test",
            value=100.0,
            bounds=(0.0, 200.0),
            units="Pa"
        )

        param_dict = param.to_dict()

        assert param_dict["name"] == "test"
        assert param_dict["value"] == 100.0
        assert param_dict["bounds"] == (0.0, 200.0)
        assert param_dict["units"] == "Pa"

    def test_parameter_from_dict(self):
        """Test parameter creation from dict."""
        param_dict = {
            "name": "test",
            "value": 50.0,
            "bounds": [0.0, 100.0],
            "units": "Pa"
        }

        param = Parameter.from_dict(param_dict)

        assert param.name == "test"
        assert param.value == 50.0
        assert param.bounds == (0.0, 100.0)
        assert param.units == "Pa"


class TestParameterSet:
    """Test ParameterSet for managing multiple parameters."""

    def test_create_empty_parameter_set(self):
        """Test creating an empty parameter set."""
        params = ParameterSet()

        assert len(params) == 0
        assert params.to_dict() == {}

    def test_add_parameters(self):
        """Test adding parameters to set."""
        params = ParameterSet()

        params.add("G", value=100.0, bounds=(0, 1e6), units="Pa")
        params.add("eta", value=1000.0, bounds=(0, 1e9), units="Pa.s")

        assert len(params) == 2
        assert "G" in params
        assert "eta" in params

    def test_get_parameter(self):
        """Test retrieving parameters."""
        params = ParameterSet()
        params.add("test", value=50.0)

        param = params.get("test")
        assert param.value == 50.0

        # Non-existent parameter
        assert params.get("nonexistent") is None

    def test_set_parameter_value(self):
        """Test setting parameter values."""
        params = ParameterSet()
        params.add("test", value=50.0, bounds=(0, 100))

        params.set_value("test", 75.0)
        assert params.get("test").value == 75.0

        # Invalid value
        with pytest.raises(ValueError):
            params.set_value("test", 150.0)

    def test_get_values_array(self):
        """Test getting parameter values as array."""
        params = ParameterSet()
        params.add("a", value=1.0)
        params.add("b", value=2.0)
        params.add("c", value=3.0)

        values = params.get_values()

        assert np.array_equal(values, [1.0, 2.0, 3.0])

    def test_set_values_array(self):
        """Test setting parameter values from array."""
        params = ParameterSet()
        params.add("a", value=0.0)
        params.add("b", value=0.0)
        params.add("c", value=0.0)

        params.set_values([1.0, 2.0, 3.0])

        assert params.get("a").value == 1.0
        assert params.get("b").value == 2.0
        assert params.get("c").value == 3.0

    def test_get_bounds(self):
        """Test getting parameter bounds."""
        params = ParameterSet()
        params.add("a", value=1.0, bounds=(0, 10))
        params.add("b", value=2.0, bounds=(0, 20))

        bounds = params.get_bounds()

        assert bounds[0] == (0, 10)
        assert bounds[1] == (0, 20)

    def test_parameter_set_to_dict(self):
        """Test serializing parameter set."""
        params = ParameterSet()
        params.add("G", value=100.0, units="Pa")
        params.add("eta", value=1000.0, units="Pa.s")

        params_dict = params.to_dict()

        assert "G" in params_dict
        assert "eta" in params_dict
        assert params_dict["G"]["value"] == 100.0
        assert params_dict["eta"]["value"] == 1000.0

    def test_parameter_set_from_dict(self):
        """Test creating parameter set from dict."""
        params_dict = {
            "G": {"value": 100.0, "units": "Pa"},
            "eta": {"value": 1000.0, "units": "Pa.s"}
        }

        params = ParameterSet.from_dict(params_dict)

        assert len(params) == 2
        assert params.get("G").value == 100.0
        assert params.get("eta").value == 1000.0


class TestBaseModel:
    """Test BaseModel abstract class."""

    def test_base_model_interface(self):
        """Test that BaseModel defines required interface."""
        # Create a concrete implementation
        class ConcreteModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.fitted_ = True
                return self

            def _predict(self, X):
                return X * 2

            def get_params(self, deep=True):
                return {"param1": 1}

            def set_params(self, **params):
                return self

        model = ConcreteModel()

        # Test that interface methods exist
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert hasattr(model, 'get_params')
        assert hasattr(model, 'set_params')

    def test_model_fit_with_numpy(self):
        """Test model fitting with numpy arrays."""
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.coef_ = np.mean(y) / np.mean(X)
                return self

            def _predict(self, X):
                return X * self.coef_

        model = TestModel()
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        model.fit(X, y)

        assert hasattr(model, 'coef_')
        assert model.coef_ == 2.0

    def test_model_fit_with_jax(self):
        """Test model fitting with JAX arrays."""
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.coef_ = jnp.mean(y) / jnp.mean(X)
                return self

            def _predict(self, X):
                return X * self.coef_

        model = TestModel()
        X = jnp.array([1, 2, 3, 4, 5])
        y = jnp.array([2, 4, 6, 8, 10])

        model.fit(X, y)

        assert hasattr(model, 'coef_')
        assert float(model.coef_) == 2.0

    def test_model_predict(self):
        """Test model prediction."""
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.coef_ = 2.0
                return self

            def _predict(self, X):
                return X * self.coef_

        model = TestModel()
        model.fit(np.array([1]), np.array([2]))

        X_test = np.array([1, 2, 3])
        predictions = model.predict(X_test)

        assert np.array_equal(predictions, [2, 4, 6])

    def test_model_parameters(self):
        """Test model parameter management."""
        class TestModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.parameters = ParameterSet()
                self.parameters.add("alpha", value=1.0, bounds=(0, 10))
                self.parameters.add("beta", value=2.0, bounds=(0, 20))

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                alpha = self.parameters.get("alpha").value
                beta = self.parameters.get("beta").value
                return X * alpha + beta

        model = TestModel()

        # Test getting parameters
        params = model.get_params()
        assert "alpha" in params
        assert "beta" in params

        # Test setting parameters
        model.set_params(alpha=5.0, beta=10.0)
        assert model.parameters.get("alpha").value == 5.0
        assert model.parameters.get("beta").value == 10.0

    def test_model_serialization(self):
        """Test model serialization."""
        class TestModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.fitted_ = False

            def _fit(self, X, y, **kwargs):
                self.fitted_ = True
                self.coef_ = 2.0
                return self

            def _predict(self, X):
                return X * self.coef_

            def to_dict(self):
                return {
                    "fitted": self.fitted_,
                    "coef": getattr(self, 'coef_', None)
                }

            @classmethod
            def from_dict(cls, data):
                model = cls()
                model.fitted_ = data["fitted"]
                if data["coef"] is not None:
                    model.coef_ = data["coef"]
                return model

        # Create and fit model
        model = TestModel()
        model.fit(np.array([1]), np.array([2]))

        # Serialize
        model_dict = model.to_dict()

        # Deserialize
        restored = TestModel.from_dict(model_dict)

        assert restored.fitted_ == True
        assert restored.coef_ == 2.0

    def test_model_sklearn_compatibility(self):
        """Test scikit-learn style interface."""
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                self.fitted_ = True
                return self

            def _predict(self, X):
                return np.ones_like(X)

            def score(self, X, y):
                """Compute R² score."""
                predictions = self.predict(X)
                ss_res = np.sum((y - predictions) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot == 0:
                    return 1.0 if ss_res == 0 else 0.0
                return 1 - (ss_res / ss_tot)

        model = TestModel()

        # Test fit returns self (for chaining)
        result = model.fit(np.array([1]), np.array([1]))
        assert result is model

        # Test score method
        X = np.array([1, 2, 3])
        y = np.array([1, 1, 1])
        score = model.score(X, y)
        assert score == 1.0  # Perfect score for constant prediction


class TestBaseTransform:
    """Test BaseTransform abstract class."""

    def test_base_transform_interface(self):
        """Test that BaseTransform defines required interface."""
        class ConcreteTransform(BaseTransform):
            def _transform(self, data):
                return data * 2

            def _inverse_transform(self, data):
                return data / 2

        transform = ConcreteTransform()

        # Test that interface methods exist
        assert hasattr(transform, 'transform')
        assert hasattr(transform, 'inverse_transform')
        assert hasattr(transform, 'fit_transform')

    def test_transform_with_numpy(self):
        """Test transform with numpy arrays."""
        class TestTransform(BaseTransform):
            def _transform(self, data):
                return np.log(data)

            def _inverse_transform(self, data):
                return np.exp(data)

        transform = TestTransform()
        data = np.array([1, 2, 3, 4, 5])

        transformed = transform.transform(data)
        assert np.allclose(transformed, np.log(data))

        restored = transform.inverse_transform(transformed)
        assert np.allclose(restored, data)

    def test_transform_with_jax(self):
        """Test transform with JAX arrays."""
        class TestTransform(BaseTransform):
            def _transform(self, data):
                return jnp.sqrt(data)

            def _inverse_transform(self, data):
                return data ** 2

        transform = TestTransform()
        data = jnp.array([1, 4, 9, 16, 25])

        transformed = transform.transform(data)
        assert jnp.allclose(transformed, jnp.sqrt(data))

        restored = transform.inverse_transform(transformed)
        assert jnp.allclose(restored, data)

    def test_fit_transform(self):
        """Test fit_transform method."""
        class TestTransform(BaseTransform):
            def fit(self, data):
                """Learn parameters from data."""
                self.mean_ = np.mean(data)
                self.std_ = np.std(data)
                return self

            def _transform(self, data):
                return (data - self.mean_) / self.std_

        transform = TestTransform()
        data = np.array([1, 2, 3, 4, 5])

        # fit_transform should fit then transform
        transformed = transform.fit_transform(data)

        assert hasattr(transform, 'mean_')
        assert hasattr(transform, 'std_')
        assert np.allclose(np.mean(transformed), 0.0, atol=1e-10)
        assert np.allclose(np.std(transformed), 1.0)

    def test_transform_parameters(self):
        """Test transform with parameters."""
        class TestTransform(BaseTransform):
            def __init__(self, scale=1.0, offset=0.0):
                super().__init__()
                self.scale = scale
                self.offset = offset

            def _transform(self, data):
                return data * self.scale + self.offset

            def _inverse_transform(self, data):
                return (data - self.offset) / self.scale

        transform = TestTransform(scale=2.0, offset=10.0)
        data = np.array([1, 2, 3])

        transformed = transform.transform(data)
        assert np.array_equal(transformed, [12, 14, 16])

        restored = transform.inverse_transform(transformed)
        assert np.allclose(restored, data)

    def test_transform_validation(self):
        """Test transform input validation."""
        class TestTransform(BaseTransform):
            def _transform(self, data):
                if np.any(data < 0):
                    raise ValueError("Data must be non-negative")
                return np.sqrt(data)

        transform = TestTransform()

        # Valid data
        valid_data = np.array([1, 4, 9])
        result = transform.transform(valid_data)
        assert result is not None

        # Invalid data
        invalid_data = np.array([1, -1, 4])
        with pytest.raises(ValueError, match="must be non-negative"):
            transform.transform(invalid_data)

    def test_transform_chaining(self):
        """Test chaining multiple transforms."""
        class LogTransform(BaseTransform):
            def _transform(self, data):
                return np.log(data)

            def _inverse_transform(self, data):
                return np.exp(data)

        class StandardizeTransform(BaseTransform):
            def fit(self, data):
                self.mean_ = np.mean(data)
                self.std_ = np.std(data)
                return self

            def _transform(self, data):
                return (data - self.mean_) / self.std_

        # Create pipeline of transforms
        log_transform = LogTransform()
        standardize = StandardizeTransform()

        data = np.array([1, 10, 100])

        # Chain transforms
        logged = log_transform.transform(data)
        standardized = standardize.fit_transform(logged)

        assert standardized is not None
        assert np.allclose(np.mean(standardized), 0.0, atol=1e-10)


class TestJAXSupport:
    """Test JAX-specific functionality in base classes."""

    def test_jax_jit_compilation(self):
        """Test that methods can be JIT compiled."""
        import jax

        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return jnp.sum(X ** 2)

        model = TestModel()

        # JIT compile the predict method
        jit_predict = jax.jit(model._predict)

        X = jnp.array([1.0, 2.0, 3.0])
        result = jit_predict(X)

        assert float(result) == 14.0

    def test_jax_grad_support(self):
        """Test that methods support automatic differentiation."""
        import jax

        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return jnp.sum(X ** 2)

        model = TestModel()

        # Compute gradient
        grad_fn = jax.grad(model._predict)

        X = jnp.array([1.0, 2.0, 3.0])
        gradient = grad_fn(X)

        expected = 2 * X  # Gradient of x^2 is 2x
        assert jnp.allclose(gradient, expected)

    def test_jax_vmap_support(self):
        """Test that methods support vectorization."""
        import jax

        class TestTransform(BaseTransform):
            def _transform(self, data):
                return jnp.exp(data)

        transform = TestTransform()

        # Vectorize over batch dimension
        vmap_transform = jax.vmap(transform._transform)

        # Batch of data
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = vmap_transform(batch)

        expected = jnp.exp(batch)
        assert jnp.allclose(result, expected)