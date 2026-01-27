"""Focused tests for ModelRegistry and TransformRegistry convenience classes.

This test suite ensures that the ModelRegistry and TransformRegistry classes
provide the expected decorator-based registration, factory methods, and discovery
capabilities as specified in Task Group 6.
"""

import pytest

from rheojax.core.base import BaseModel, BaseTransform
from rheojax.core.registry import ModelRegistry, Registry, TransformRegistry


class TestModelRegistryDecorator:
    """Test ModelRegistry decorator registration (Task 6.2)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_register_model_with_decorator(self):
        """Test @ModelRegistry.register('name') decorator."""

        @ModelRegistry.register("maxwell")
        class Maxwell(BaseModel):
            """Maxwell viscoelastic model."""

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Verify registration
        assert "maxwell" in ModelRegistry.list_models()

        # Verify info retrieval
        info = ModelRegistry.get_info("maxwell")
        assert info is not None
        assert info.name == "maxwell"
        assert info.plugin_class is Maxwell
        assert "Maxwell viscoelastic model" in info.doc

    def test_register_model_with_metadata(self):
        """Test decorator with metadata."""

        @ModelRegistry.register("zener", parameters=["G_s", "G_p", "eta_p"])
        class Zener(BaseModel):
            """Zener standard linear solid model."""

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Verify metadata stored
        info = ModelRegistry.get_info("zener")
        assert info.metadata["parameters"] == ["G_s", "G_p", "eta_p"]

    def test_decorator_returns_class_unchanged(self):
        """Test that decorator returns original class."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Should be able to instantiate directly
        model = TestModel()
        assert isinstance(model, TestModel)
        assert isinstance(model, BaseModel)


class TestModelRegistryFactory:
    """Test ModelRegistry factory method (Task 6.2)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_create_model_by_name(self):
        """Test ModelRegistry.create(name) factory method."""

        @ModelRegistry.register("springpot")
        class SpringPot(BaseModel):
            def __init__(self, V=1.0, alpha=0.5):
                super().__init__()
                self.V = V
                self.alpha = alpha

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Create instance using factory method
        model = ModelRegistry.create("springpot")
        assert isinstance(model, SpringPot)
        assert model.V == 1.0
        assert model.alpha == 0.5

    def test_create_model_with_arguments(self):
        """Test factory method with constructor arguments."""

        @ModelRegistry.register("springpot")
        class SpringPot(BaseModel):
            def __init__(self, V=1.0, alpha=0.5):
                super().__init__()
                self.V = V
                self.alpha = alpha

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Create with custom parameters
        model = ModelRegistry.create("springpot", V=2.5, alpha=0.7)
        assert model.V == 2.5
        assert model.alpha == 0.7

    def test_create_nonexistent_model_raises_error(self):
        """Test that creating non-existent model raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            ModelRegistry.create("nonexistent_model")


class TestModelRegistryDiscovery:
    """Test ModelRegistry.list_models() discovery (Task 6.2)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_list_models_empty(self):
        """Test listing models when none registered."""
        models = ModelRegistry.list_models()
        assert models == []

    def test_list_models_with_multiple_registrations(self):
        """Test ModelRegistry.list_models() discovery."""

        # Register multiple models
        @ModelRegistry.register("maxwell")
        class Maxwell(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        @ModelRegistry.register("zener")
        class Zener(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        @ModelRegistry.register("springpot")
        class SpringPot(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        # Discover all models
        models = ModelRegistry.list_models()
        assert len(models) == 3
        assert "maxwell" in models
        assert "zener" in models
        assert "springpot" in models

    def test_get_info_for_registered_model(self):
        """Test getting info for registered model."""

        @ModelRegistry.register("test_model", version="1.0.0", domain="frequency")
        class TestModel(BaseModel):
            """Test model documentation."""

            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        info = ModelRegistry.get_info("test_model")
        assert info.name == "test_model"
        assert info.plugin_class is TestModel
        assert info.metadata["version"] == "1.0.0"
        assert info.metadata["domain"] == "frequency"
        assert "Test model documentation" in info.doc


class TestTransformRegistryDecorator:
    """Test TransformRegistry decorator registration (Task 6.3)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_register_transform_with_decorator(self):
        """Test @TransformRegistry.register('name') decorator."""

        @TransformRegistry.register("fft_analysis")
        class RheoAnalysis(BaseTransform):
            """FFT-based rheological analysis."""

            def _transform(self, data):
                return data

        # Verify registration
        assert "fft_analysis" in TransformRegistry.list_transforms()

        # Verify info retrieval
        info = TransformRegistry.get_info("fft_analysis")
        assert info is not None
        assert info.name == "fft_analysis"
        assert info.plugin_class is RheoAnalysis
        assert "FFT-based rheological analysis" in info.doc

    def test_register_transform_with_metadata(self):
        """Test decorator with metadata."""

        @TransformRegistry.register("mastercurve", method="wlf", algorithm="ml")
        class AutomatedMasterCurve(BaseTransform):
            """ML-based time-temperature superposition."""

            def _transform(self, data):
                return data

        # Verify metadata stored
        info = TransformRegistry.get_info("mastercurve")
        assert info.metadata["method"] == "wlf"
        assert info.metadata["algorithm"] == "ml"


class TestTransformRegistryFactory:
    """Test TransformRegistry factory method (Task 6.3)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_create_transform_by_name(self):
        """Test TransformRegistry.create(name) factory method."""

        @TransformRegistry.register("mutation_number")
        class MutationNumber(BaseTransform):
            def __init__(self, threshold=0.1):
                super().__init__()
                self.threshold = threshold

            def _transform(self, data):
                return data

        # Create instance using factory method
        transform = TransformRegistry.create("mutation_number")
        assert isinstance(transform, MutationNumber)
        assert transform.threshold == 0.1

    def test_create_transform_with_arguments(self):
        """Test factory method with constructor arguments."""

        @TransformRegistry.register("mutation_number")
        class MutationNumber(BaseTransform):
            def __init__(self, threshold=0.1):
                super().__init__()
                self.threshold = threshold

            def _transform(self, data):
                return data

        # Create with custom parameters
        transform = TransformRegistry.create("mutation_number", threshold=0.5)
        assert transform.threshold == 0.5

    def test_create_nonexistent_transform_raises_error(self):
        """Test that creating non-existent transform raises KeyError."""
        with pytest.raises(KeyError, match="not found in registry"):
            TransformRegistry.create("nonexistent_transform")


class TestTransformRegistryDiscovery:
    """Test TransformRegistry discovery (Task 6.3)."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_list_transforms_empty(self):
        """Test listing transforms when none registered."""
        transforms = TransformRegistry.list_transforms()
        assert transforms == []

    def test_list_transforms_with_multiple_registrations(self):
        """Test TransformRegistry.list_transforms() discovery."""

        # Register multiple transforms
        @TransformRegistry.register("fft_analysis")
        class RheoAnalysis(BaseTransform):
            def _transform(self, data):
                return data

        @TransformRegistry.register("mastercurve")
        class AutomatedMasterCurve(BaseTransform):
            def _transform(self, data):
                return data

        @TransformRegistry.register("owchirp")
        class OWChirpGeneration(BaseTransform):
            def _transform(self, data):
                return data

        # Discover all transforms
        transforms = TransformRegistry.list_transforms()
        assert len(transforms) == 3
        assert "fft_analysis" in transforms
        assert "mastercurve" in transforms
        assert "owchirp" in transforms


class TestRegistryIsolation:
    """Test that ModelRegistry and TransformRegistry share the same singleton."""

    def setup_method(self):
        """Save and clear registry before each test."""
        registry = Registry.get_instance()
        # Save full PluginInfo objects, not just class and type
        self._saved_models = dict(registry._models)
        self._saved_transforms = dict(registry._transforms)
        registry.clear()

    def teardown_method(self):
        """Restore registry state after each test."""
        registry = Registry.get_instance()
        registry.clear()
        # Restore full PluginInfo objects directly
        registry._models.update(self._saved_models)
        registry._transforms.update(self._saved_transforms)

    def test_shared_singleton(self):
        """Test that both registries use the same singleton Registry."""

        @ModelRegistry.register("test_model")
        class TestModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        @TransformRegistry.register("test_transform")
        class TestTransform(BaseTransform):
            def _transform(self, data):
                return data

        # Both should be registered in the same underlying registry
        registry = Registry.get_instance()
        assert "test_model" in registry.get_all_models()
        assert "test_transform" in registry.get_all_transforms()

        # Stats should reflect both
        stats = registry.get_stats()
        assert stats["models"] == 1
        assert stats["transforms"] == 1
        assert stats["total"] == 2

    def test_unregister_via_convenience_classes(self):
        """Test unregistering via ModelRegistry and TransformRegistry."""

        @ModelRegistry.register("temp_model")
        class TempModel(BaseModel):
            def _fit(self, X, y, **kwargs):
                return self

            def _predict(self, X):
                return X

        @TransformRegistry.register("temp_transform")
        class TempTransform(BaseTransform):
            def _transform(self, data):
                return data

        # Verify registration
        assert "temp_model" in ModelRegistry.list_models()
        assert "temp_transform" in TransformRegistry.list_transforms()

        # Unregister via convenience classes
        ModelRegistry.unregister("temp_model")
        TransformRegistry.unregister("temp_transform")

        # Verify unregistration
        assert "temp_model" not in ModelRegistry.list_models()
        assert "temp_transform" not in TransformRegistry.list_transforms()
