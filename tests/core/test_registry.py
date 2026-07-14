"""Tests for plugin registry system.

This test suite ensures the registry system can discover, register,
validate, and manage models and transforms as plugins.
"""

from unittest.mock import Mock, patch

import pytest

from rheojax.core.registry import PluginInfo, PluginType, Registry


class TestRegistryCreation:
    """Test registry creation and initialization."""

    def setup_method(self):
        """Save registry state before each test."""
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

        metadata: dict = {
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

    def test_register_invalid_protocol_string_raises_error(self):
        """A typo'd protocol string must fail loudly at registration time,
        not silently register the model with an empty protocols list."""
        registry = Registry()

        class MockModel:
            pass

        with pytest.raises(ValueError, match="Invalid protocol"):
            registry.register(
                "maxwell",
                MockModel,
                plugin_type=PluginType.MODEL,
                protocols=["relaxaton"],
            )

        # The model must not have been silently registered either.
        assert "maxwell" not in registry.get_all_models()

    def test_register_invalid_transform_type_raises_error(self):
        """A typo'd transform_type string must fail loudly at registration
        time, not silently register the transform with transform_type=None."""
        registry = Registry()

        class MockTransform:
            pass

        with pytest.raises(ValueError, match="Invalid transform_type"):
            registry.register(
                "fft",
                MockTransform,
                plugin_type=PluginType.TRANSFORM,
                transform_type="not_a_real_type",
            )

        assert "fft" not in registry.get_all_transforms()


class TestRegistryRetrieval:
    """Test plugin retrieval."""

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

        metadata: dict = {"version": "1.0.0", "author": "test"}
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

    def test_validate_model_interface_rejects_incomplete_abc_subclass(self):
        """hasattr(fit)/hasattr(predict) alone can't catch an incomplete ABC
        subclass, since abstract stubs and inherited concrete wrappers are
        still present via attribute lookup. validate=True must additionally
        reject a class that still has unimplemented abstract methods."""
        import abc

        registry = Registry()

        class AbstractModel(abc.ABC):
            @abc.abstractmethod
            def fit(self, data): ...

            @abc.abstractmethod
            def predict(self, data): ...

        class IncompleteModel(AbstractModel):
            """Overrides fit but never implements predict."""

            def fit(self, data):
                pass

        # hasattr checks alone would pass (predict is still present as the
        # inherited abstract stub), so this only fails via the abstractness
        # check.
        assert hasattr(IncompleteModel, "predict")

        with pytest.raises(ValueError, match="abstract"):
            registry.register(
                "incomplete",
                IncompleteModel,
                plugin_type=PluginType.MODEL,
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
            # discover() now only registers classes actually defined in the
            # scanned module (obj.__module__ == module_name), so the mock
            # class must claim membership in "custom_model" to be picked up.
            mock_module.CustomModel.__module__ = "custom_model"
            mock_import.return_value = mock_module

            registry.discover_directory("/custom/plugins")

            # Should discover custom plugins
            assert len(registry.get_all_models()) > 0

    def test_discover_skips_classes_imported_from_elsewhere(self):
        """discover() must not register classes merely imported into the
        scanned module's namespace (e.g. a base class), only ones actually
        defined there - otherwise e.g. BaseModel pollutes get_all_models()."""
        registry = Registry()

        class BaseModelLike:
            """Stand-in for a base class defined in another module."""

            def fit(self, data):
                pass

            def predict(self, data):
                pass

        # Simulate importing BaseModelLike into the scanned module: its
        # __module__ still points at its real (different) origin module.
        BaseModelLike.__module__ = "some.other.module"

        class LocalModel:
            def fit(self, data):
                pass

            def predict(self, data):
                pass

        LocalModel.__module__ = "fake_scanned_module"

        fake_module = Mock()
        fake_module.BaseModelLike = BaseModelLike
        fake_module.LocalModel = LocalModel

        with patch("importlib.import_module", return_value=fake_module):
            registry.discover("fake_scanned_module")

        assert "LocalModel" in registry.get_all_models()
        assert "BaseModelLike" not in registry.get_all_models()

    def test_discover_logs_warning_on_import_error(self, caplog):
        """A broken module path must be logged, not silently swallowed."""
        registry = Registry()

        with (
            patch(
                "importlib.import_module", side_effect=ImportError("no such module")
            ),
            caplog.at_level("WARNING"),
        ):
            registry.discover("nonexistent.module.path")

        assert len(caplog.records) >= 1
        assert any(
            "nonexistent.module.path" in str(record.__dict__)
            for record in caplog.records
        )


class TestRegistryManagement:
    """Test registry management operations."""

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

    def test_get_all_returns_models_and_transforms(self):
        """get_all() must return every registered plugin with its type."""
        registry = Registry()

        class MockModel:
            pass

        class MockTransform:
            pass

        registry.register("some_model", MockModel, plugin_type=PluginType.MODEL)
        registry.register(
            "some_transform", MockTransform, plugin_type=PluginType.TRANSFORM
        )

        all_plugins = registry.get_all()
        assert all_plugins["some_model"] == (MockModel, PluginType.MODEL)
        assert all_plugins["some_transform"] == (MockTransform, PluginType.TRANSFORM)

    def test_get_all_warns_on_cross_type_name_collision(self, caplog):
        """A name registered as both a model and a transform is valid (see
        test_registry_namespaces - separate namespaces by design), but
        get_all() can only return one entry per name. It must log a
        warning instead of silently dropping the model entry."""
        registry = Registry()

        class MockModel:
            pass

        class MockTransform:
            pass

        registry.register("dual", MockModel, plugin_type=PluginType.MODEL)
        registry.register("dual", MockTransform, plugin_type=PluginType.TRANSFORM)

        with caplog.at_level("WARNING"):
            all_plugins = registry.get_all()

        # Both registrations remain independently retrievable via get().
        assert registry.get("dual", PluginType.MODEL) is MockModel
        assert registry.get("dual", PluginType.TRANSFORM) is MockTransform
        # get_all() can only surface one; make sure the collision is logged.
        assert all_plugins["dual"] == (MockTransform, PluginType.TRANSFORM)
        assert any("dual" in str(record.__dict__) for record in caplog.records)

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
        state: dict = registry.export_state()

        # Clear and verify empty
        registry.clear()
        assert len(registry) == 0

        # Import state
        registry.import_state(state)

        # Verify restoration
        assert "model1" in registry.get_all_models()
        assert "transform1" in registry.get_all_transforms()

    def test_import_state_logs_warning_on_missing_class(self, caplog):
        """A renamed/moved class must be logged, not silently dropped."""
        registry = Registry()

        state = {
            "models": {
                "ghost_model": {
                    "class_name": "NoLongerExists",
                    "module": "unittest.mock",
                    "metadata": {},
                    "protocols": [],
                }
            },
            "transforms": {},
        }

        with caplog.at_level("WARNING"):
            registry.import_state(state)

        assert "ghost_model" not in registry.get_all_models()
        assert len(caplog.records) >= 1
        assert any(
            "ghost_model" in str(record.__dict__) for record in caplog.records
        )

    def test_import_state_skips_malformed_entries_without_crashing(self, caplog):
        """Entries missing required keys (or not dict-shaped) must be skipped
        per-entry, not crash import_state and abort remaining restoration."""
        registry = Registry()

        state = {
            "models": {
                # Missing "module" key entirely -> KeyError on info["module"]
                "no_module": {"class_name": "Mock"},
                # Not a dict at all -> TypeError on info["module"]
                "not_a_dict": None,
                # Well-formed entry that should still restore successfully
                "good_model": {
                    "class_name": "Mock",
                    "module": "unittest.mock",
                    "metadata": {},
                    "protocols": [],
                },
            },
            "transforms": {
                "no_class_name": {"module": "unittest.mock"},
                "also_not_a_dict": ["oops"],
            },
        }

        with caplog.at_level("WARNING"):
            registry.import_state(state)

        assert "no_module" not in registry.get_all_models()
        assert "not_a_dict" not in registry.get_all_models()
        assert "no_class_name" not in registry.get_all_transforms()
        assert "also_not_a_dict" not in registry.get_all_transforms()
        # The well-formed entry must still be restored despite the bad ones.
        assert "good_model" in registry.get_all_models()


class TestRegistryCompatibility:
    """Test registry compatibility with plugin system."""

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

    def test_find_compatible_plugin_type_scopes_search(self):
        """find_compatible(plugin_type=...) must not mix models and transforms.

        Without a protocol/transform_type filter, a bare find_compatible()
        call still scans both namespaces (back-compat default), but callers
        that scope with plugin_type must get only that namespace back.
        """
        registry = Registry()

        class MyModel:
            pass

        class MyTransform:
            pass

        registry.register(
            "mymodel",
            MyModel,
            plugin_type=PluginType.MODEL,
            metadata={"domain": "x"},
        )
        registry.register(
            "mytransform",
            MyTransform,
            plugin_type=PluginType.TRANSFORM,
            metadata={"domain": "x"},
        )

        # Bare call (no scoping) still searches both, preserving existing behavior.
        both = registry.find_compatible(domain="x")
        assert set(both) == {"mymodel", "mytransform"}

        # Scoped calls must not leak across namespaces.
        models_only = registry.find_compatible(
            domain="x", plugin_type=PluginType.MODEL
        )
        assert models_only == ["mymodel"]

        transforms_only = registry.find_compatible(
            domain="x", plugin_type=PluginType.TRANSFORM
        )
        assert transforms_only == ["mytransform"]

    def test_find_compatible_invalid_protocol_returns_no_matches(self):
        """An unrecognized protocol string must not raise, just match nothing."""
        registry = Registry()

        class SomeModel:
            pass

        registry.register(
            "some_model",
            SomeModel,
            plugin_type=PluginType.MODEL,
            protocols=["relaxation"],
        )

        # "rotation"/"unknown" are real RheoData.test_mode values that are not
        # members of the Protocol enum (see TestModeEnum) - must not crash.
        compatible = registry.find_compatible(protocol="unknown")
        assert compatible == []

    def test_find_compatible_invalid_transform_type_returns_no_matches(self):
        """An unrecognized transform_type string must not raise, just match
        nothing (mirrors the protocol-side graceful-degradation contract)."""
        registry = Registry()

        class SomeTransform:
            pass

        registry.register(
            "some_transform",
            SomeTransform,
            plugin_type=PluginType.TRANSFORM,
            transform_type="spectral",
        )

        compatible = registry.find_compatible(transform_type="not_a_real_type")
        assert compatible == []

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
