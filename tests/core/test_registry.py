"""Tests for plugin registry system.

This test suite ensures the registry system can discover, register,
validate, and manage models and transforms as plugins.
"""

from typing import Any, Dict, Type
from unittest.mock import Mock, patch

import pytest

from rheojax.core.registry import PluginInfo, PluginType, Registry


class TestRegistryCreation:
    """Test registry creation and initialization."""

    def setup_method(self):
        """Save registry state before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_create_empty_registry(self):
        """Test creating an empty registry."""
        registry = Registry()

        assert registry is not None
        assert len(registry) == 0
        assert registry.get_all_models() == []
        assert registry.get_all_transforms() == []

    def test_registry_singleton_pattern(self):
        """Test that registry follows singleton pattern."""
        registry1 = Registry.get_instance()
        registry2 = Registry.get_instance()

        assert registry1 is registry2

    def test_registry_namespaces(self):
        """Test that models and transforms have separate namespaces."""
        registry = Registry()

        # Create mock model and transform with same name
        mock_model = Mock()
        mock_transform = Mock()

        registry.register("test_item", mock_model, plugin_type=PluginType.MODEL)
        registry.register("test_item", mock_transform, plugin_type=PluginType.TRANSFORM)

        # Should be able to retrieve both without conflict
        assert registry.get("test_item", plugin_type=PluginType.MODEL) is mock_model
        assert (
            registry.get("test_item", plugin_type=PluginType.TRANSFORM)
            is mock_transform
        )


class TestRegistryRegistration:
    """Test plugin registration."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_register_model(self):
        """Test registering a model."""
        registry = Registry()

        # Create a mock model class
        class MockModel:
            def fit(self, data):
                pass

            def predict(self, data):
                pass

        # Register the model
        registry.register("maxwell", MockModel, plugin_type=PluginType.MODEL)

        # Verify registration
        assert "maxwell" in registry.get_all_models()
        assert registry.get("maxwell", plugin_type=PluginType.MODEL) is MockModel

    def test_register_transform(self):
        """Test registering a transform."""
        registry = Registry()

        # Create a mock transform class
        class MockTransform:
            def transform(self, data):
                pass

        # Register the transform
        registry.register("fft", MockTransform, plugin_type=PluginType.TRANSFORM)

        # Verify registration
        assert "fft" in registry.get_all_transforms()
        assert registry.get("fft", plugin_type=PluginType.TRANSFORM) is MockTransform

    def test_register_with_metadata(self):
        """Test registering with metadata."""
        registry = Registry()

        class MockModel:
            pass

        metadata = {
            "description": "Maxwell viscoelastic model",
            "parameters": ["G", "eta"],
            "domain": "frequency",
            "version": "1.0.0",
        }

        registry.register(
            "maxwell", MockModel, plugin_type=PluginType.MODEL, metadata=metadata
        )

        # Retrieve plugin info
        info = registry.get_info("maxwell", plugin_type=PluginType.MODEL)

        assert info.name == "maxwell"
        assert info.plugin_class is MockModel
        assert info.metadata == metadata

    def test_register_duplicate_raises_error(self):
        """Test that registering duplicate names raises error."""
        registry = Registry()

        class MockModel:
            pass

        registry.register("test", MockModel, plugin_type=PluginType.MODEL)

        # Attempting to register again should raise error
        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", MockModel, plugin_type=PluginType.MODEL)

    def test_register_with_force_overwrite(self):
        """Test force overwriting existing registration."""
        registry = Registry()

        class MockModel1:
            pass

        class MockModel2:
            pass

        registry.register("test", MockModel1, plugin_type=PluginType.MODEL)
        registry.register("test", MockModel2, plugin_type=PluginType.MODEL, force=True)

        # Should now point to MockModel2
        assert registry.get("test", plugin_type=PluginType.MODEL) is MockModel2

    def test_register_invalid_type_raises_error(self):
        """Test that registering with invalid type raises error."""
        registry = Registry()

        with pytest.raises(ValueError, match="Invalid plugin type"):
            registry.register("test", Mock(), plugin_type="invalid")


class TestRegistryRetrieval:
    """Test plugin retrieval."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_get_registered_model(self):
        """Test retrieving a registered model."""
        registry = Registry()

        class MockModel:
            pass

        registry.register("maxwell", MockModel, plugin_type=PluginType.MODEL)

        retrieved = registry.get("maxwell", plugin_type=PluginType.MODEL)
        assert retrieved is MockModel

    def test_get_nonexistent_returns_none(self):
        """Test retrieving non-existent plugin returns None."""
        registry = Registry()

        result = registry.get("nonexistent", plugin_type=PluginType.MODEL)
        assert result is None

    def test_get_nonexistent_raises_error(self):
        """Test retrieving non-existent plugin can raise error."""
        registry = Registry()

        with pytest.raises(KeyError, match="not found in registry"):
            registry.get(
                "nonexistent", plugin_type=PluginType.MODEL, raise_on_missing=True
            )

    def test_get_all_models(self):
        """Test retrieving all registered models."""
        registry = Registry()

        # Register multiple models
        for name in ["maxwell", "kelvin", "zener"]:
            registry.register(name, Mock(), plugin_type=PluginType.MODEL)

        models = registry.get_all_models()

        assert len(models) == 3
        assert "maxwell" in models
        assert "kelvin" in models
        assert "zener" in models

    def test_get_all_transforms(self):
        """Test retrieving all registered transforms."""
        registry = Registry()

        # Register multiple transforms
        for name in ["fft", "mastercurve", "owchirp"]:
            registry.register(name, Mock(), plugin_type=PluginType.TRANSFORM)

        transforms = registry.get_all_transforms()

        assert len(transforms) == 3
        assert "fft" in transforms
        assert "mastercurve" in transforms
        assert "owchirp" in transforms

    def test_get_info(self):
        """Test retrieving plugin info."""
        registry = Registry()

        class MockModel:
            """Test model documentation."""

            pass

        metadata = {"version": "1.0.0", "author": "test"}
        registry.register(
            "test", MockModel, plugin_type=PluginType.MODEL, metadata=metadata
        )

        info = registry.get_info("test", plugin_type=PluginType.MODEL)

        assert info.name == "test"
        assert info.plugin_class is MockModel
        assert info.plugin_type == PluginType.MODEL
        assert info.metadata == metadata
        assert info.doc == "Test model documentation."


class TestRegistryValidation:
    """Test plugin validation."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_validate_model_interface(self):
        """Test validation of model interface."""
        registry = Registry()

        # Valid model
        class ValidModel:
            def fit(self, data):
                pass

            def predict(self, data):
                pass

        # Invalid model (missing methods)
        class InvalidModel:
            pass

        # Valid model should register successfully
        registry.register(
            "valid", ValidModel, plugin_type=PluginType.MODEL, validate=True
        )

        # Invalid model should raise error with validation
        with pytest.raises(ValueError, match="does not implement required interface"):
            registry.register(
                "invalid", InvalidModel, plugin_type=PluginType.MODEL, validate=True
            )

    def test_validate_transform_interface(self):
        """Test validation of transform interface."""
        registry = Registry()

        # Valid transform
        class ValidTransform:
            def transform(self, data):
                pass

        # Invalid transform
        class InvalidTransform:
            pass

        # Valid transform should register successfully
        registry.register(
            "valid", ValidTransform, plugin_type=PluginType.TRANSFORM, validate=True
        )

        # Invalid transform should raise error
        with pytest.raises(ValueError, match="does not implement required interface"):
            registry.register(
                "invalid",
                InvalidTransform,
                plugin_type=PluginType.TRANSFORM,
                validate=True,
            )

    def test_skip_validation(self):
        """Test that validation can be skipped."""
        registry = Registry()

        # Invalid model
        class InvalidModel:
            pass

        # Should register successfully without validation
        registry.register(
            "test", InvalidModel, plugin_type=PluginType.MODEL, validate=False
        )

        assert registry.get("test", plugin_type=PluginType.MODEL) is InvalidModel


class TestRegistryDiscovery:
    """Test automatic plugin discovery."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_discover_plugins_in_module(self):
        """Test discovering plugins in a module using decorators."""
        registry = Registry()

        # Test decorator-based discovery
        @registry.model("MaxwellModel")
        class MaxwellModel:
            def fit(self, data):
                pass

            def predict(self, data):
                pass

        @registry.transform("FFTTransform")
        class FFTTransform:
            def transform(self, data):
                pass

        # Should have discovered the valid plugins
        assert "MaxwellModel" in registry.get_all_models()
        assert "FFTTransform" in registry.get_all_transforms()

    def test_discover_with_decorators(self):
        """Test plugin discovery using decorators."""
        registry = Registry()

        # Test decorator registration
        @registry.model("decorated_model")
        class DecoratedModel:
            def fit(self, data):
                pass

            def predict(self, data):
                pass

        @registry.transform("decorated_transform")
        class DecoratedTransform:
            def transform(self, data):
                pass

        # Verify registration via decorators
        assert "decorated_model" in registry.get_all_models()
        assert "decorated_transform" in registry.get_all_transforms()

    def test_discover_custom_path(self):
        """Test discovering plugins in custom path."""
        registry = Registry()

        # Mock custom plugin directory
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", return_value=["custom_model.py"]),
            patch("importlib.import_module") as mock_import,
        ):

            # Setup mock module
            mock_module = Mock()
            mock_module.CustomModel = type(
                "CustomModel", (), {"fit": None, "predict": None}
            )
            mock_import.return_value = mock_module

            registry.discover_directory("/custom/plugins")

            # Should discover custom plugins
            assert len(registry.get_all_models()) > 0


class TestRegistryManagement:
    """Test registry management operations."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        registry = Registry()

        class MockModel:
            pass

        registry.register("test", MockModel, plugin_type=PluginType.MODEL)
        assert "test" in registry.get_all_models()

        # Unregister
        registry.unregister("test", plugin_type=PluginType.MODEL)
        assert "test" not in registry.get_all_models()

    def test_clear_registry(self):
        """Test clearing the registry."""
        registry = Registry()

        # Register some plugins
        registry.register("model1", Mock(), plugin_type=PluginType.MODEL)
        registry.register("transform1", Mock(), plugin_type=PluginType.TRANSFORM)

        assert len(registry.get_all_models()) > 0
        assert len(registry.get_all_transforms()) > 0

        # Clear registry
        registry.clear()

        assert len(registry.get_all_models()) == 0
        assert len(registry.get_all_transforms()) == 0

    def test_registry_stats(self):
        """Test getting registry statistics."""
        registry = Registry()

        # Register various plugins
        for i in range(5):
            registry.register(f"model_{i}", Mock(), plugin_type=PluginType.MODEL)
        for i in range(3):
            registry.register(
                f"transform_{i}", Mock(), plugin_type=PluginType.TRANSFORM
            )

        stats = registry.get_stats()

        assert stats["total"] == 8
        assert stats["models"] == 5
        assert stats["transforms"] == 3

    def test_registry_export_import(self):
        """Test exporting and importing registry state."""
        registry = Registry()

        # Register plugins with metadata
        registry.register(
            "model1", Mock, plugin_type=PluginType.MODEL, metadata={"version": "1.0"}
        )
        registry.register(
            "transform1",
            Mock,
            plugin_type=PluginType.TRANSFORM,
            metadata={"author": "test"},
        )

        # Export registry state
        state = registry.export_state()

        # Clear and verify empty
        registry.clear()
        assert len(registry) == 0

        # Import state
        registry.import_state(state)

        # Verify restoration
        assert "model1" in registry.get_all_models()
        assert "transform1" in registry.get_all_transforms()


class TestRegistryCompatibility:
    """Test registry compatibility with plugin system."""

    def setup_method(self):
        """Save and clear registry before each test."""
        self._saved_state = Registry.get_instance().get_all()
        Registry.get_instance().clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        Registry.get_instance().clear()
        for name, (plugin, plugin_type) in self._saved_state.items():
            Registry.get_instance().register(name, plugin, plugin_type=plugin_type)

    def test_create_instance(self):
        """Test creating plugin instances through registry."""
        registry = Registry()

        class MockModel:
            def __init__(self, param1=1, param2=2):
                self.param1 = param1
                self.param2 = param2

        registry.register("test", MockModel, plugin_type=PluginType.MODEL)

        # Create instance through registry
        instance = registry.create_instance(
            "test", PluginType.MODEL, param1=10, param2=20
        )

        assert isinstance(instance, MockModel)
        assert instance.param1 == 10
        assert instance.param2 == 20

    def test_list_compatible_plugins(self):
        """Test listing plugins compatible with certain data."""
        registry = Registry()

        class FrequencyModel:
            domain = "frequency"

        class TimeModel:
            domain = "time"

        registry.register(
            "freq_model",
            FrequencyModel,
            plugin_type=PluginType.MODEL,
            metadata={"domain": "frequency"},
        )
        registry.register(
            "time_model",
            TimeModel,
            plugin_type=PluginType.MODEL,
            metadata={"domain": "time"},
        )

        # Find compatible models for frequency data
        compatible = registry.find_compatible(domain="frequency")

        assert "freq_model" in compatible
        assert "time_model" not in compatible

    def test_plugin_versioning(self):
        """Test plugin versioning support."""
        registry = Registry()

        class ModelV1:
            version = "1.0.0"

        class ModelV2:
            version = "2.0.0"

        # Register different versions
        registry.register(
            "model",
            ModelV1,
            plugin_type=PluginType.MODEL,
            metadata={"version": "1.0.0"},
        )

        # Try to register newer version
        registry.register(
            "model",
            ModelV2,
            plugin_type=PluginType.MODEL,
            metadata={"version": "2.0.0"},
            force=True,
        )

        # Should get the newer version
        model = registry.get("model", plugin_type=PluginType.MODEL)
        assert model is ModelV2
