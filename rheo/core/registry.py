"""Plugin registry system for models and transforms.

This module provides a registry system for discovering, registering, and managing
models and transforms as plugins, enabling extensibility of the rheo package.
"""

from __future__ import annotations

import importlib
import inspect
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union


class PluginType(Enum):
    """Types of plugins that can be registered."""
    MODEL = "model"
    TRANSFORM = "transform"


@dataclass
class PluginInfo:
    """Information about a registered plugin."""
    name: str
    plugin_class: Type
    plugin_type: PluginType
    metadata: Dict[str, Any]
    doc: Optional[str] = None

    def __post_init__(self):
        """Extract documentation from plugin class."""
        if self.doc is None and self.plugin_class:
            self.doc = inspect.getdoc(self.plugin_class)


class Registry:
    """Central registry for models and transforms.

    This class manages plugin registration, discovery, and retrieval
    for all models and transforms in the rheo package.
    """

    _instance: Optional[Registry] = None
    _models: Dict[str, PluginInfo]
    _transforms: Dict[str, PluginInfo]

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._models = {}
            cls._instance._transforms = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> Registry:
        """Get the singleton registry instance.

        Returns:
            The global Registry instance
        """
        return cls()

    def register(
        self,
        name: str,
        plugin_class: Type,
        plugin_type: Union[PluginType, str],
        metadata: Optional[Dict[str, Any]] = None,
        validate: bool = False,
        force: bool = False
    ):
        """Register a plugin in the registry.

        Args:
            name: Unique name for the plugin
            plugin_class: The plugin class to register
            plugin_type: Type of plugin (MODEL or TRANSFORM)
            metadata: Optional metadata dictionary
            validate: Whether to validate the plugin interface
            force: Whether to overwrite existing registration

        Raises:
            ValueError: If plugin is already registered (and force=False) or invalid
        """
        # Convert string to enum if needed
        if isinstance(plugin_type, str):
            try:
                plugin_type = PluginType(plugin_type.lower())
            except ValueError:
                raise ValueError(f"Invalid plugin type: {plugin_type}. Must be 'model' or 'transform'")

        # Select appropriate registry
        if plugin_type == PluginType.MODEL:
            registry = self._models
        elif plugin_type == PluginType.TRANSFORM:
            registry = self._transforms
        else:
            raise ValueError(f"Invalid plugin type: {plugin_type}")

        # Check if already registered
        if name in registry and not force:
            raise ValueError(f"Plugin '{name}' is already registered as a {plugin_type.value}")

        # Validate interface if requested
        if validate:
            self._validate_plugin(plugin_class, plugin_type)

        # Create plugin info
        info = PluginInfo(
            name=name,
            plugin_class=plugin_class,
            plugin_type=plugin_type,
            metadata=metadata or {}
        )

        # Register the plugin
        registry[name] = info

    def _validate_plugin(self, plugin_class: Type, plugin_type: PluginType):
        """Validate that a plugin implements the required interface.

        Args:
            plugin_class: The plugin class to validate
            plugin_type: Expected plugin type

        Raises:
            ValueError: If plugin doesn't implement required interface
        """
        if plugin_type == PluginType.MODEL:
            # Check for required model methods
            required_methods = ['fit', 'predict']
            for method in required_methods:
                if not hasattr(plugin_class, method):
                    raise ValueError(
                        f"Model plugin does not implement required interface: missing '{method}' method"
                    )

        elif plugin_type == PluginType.TRANSFORM:
            # Check for required transform methods
            if not hasattr(plugin_class, 'transform'):
                raise ValueError(
                    "Transform plugin does not implement required interface: missing 'transform' method"
                )

    def get(
        self,
        name: str,
        plugin_type: Union[PluginType, str],
        raise_on_missing: bool = False
    ) -> Optional[Type]:
        """Retrieve a registered plugin class.

        Args:
            name: Name of the plugin
            plugin_type: Type of plugin
            raise_on_missing: Whether to raise error if not found

        Returns:
            The plugin class, or None if not found

        Raises:
            KeyError: If plugin not found and raise_on_missing=True
        """
        # Convert string to enum if needed
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type.lower())

        # Select appropriate registry
        if plugin_type == PluginType.MODEL:
            registry = self._models
        elif plugin_type == PluginType.TRANSFORM:
            registry = self._transforms
        else:
            return None

        # Retrieve plugin
        if name in registry:
            return registry[name].plugin_class
        elif raise_on_missing:
            raise KeyError(f"Plugin '{name}' not found in registry for type {plugin_type.value}")
        else:
            return None

    def get_info(self, name: str, plugin_type: Union[PluginType, str]) -> Optional[PluginInfo]:
        """Get full information about a registered plugin.

        Args:
            name: Name of the plugin
            plugin_type: Type of plugin

        Returns:
            PluginInfo object or None if not found
        """
        # Convert string to enum if needed
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type.lower())

        # Select appropriate registry
        if plugin_type == PluginType.MODEL:
            registry = self._models
        elif plugin_type == PluginType.TRANSFORM:
            registry = self._transforms
        else:
            return None

        return registry.get(name)

    def get_all_models(self) -> List[str]:
        """Get list of all registered model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())

    def get_all_transforms(self) -> List[str]:
        """Get list of all registered transform names.

        Returns:
            List of transform names
        """
        return list(self._transforms.keys())

    def unregister(self, name: str, plugin_type: Union[PluginType, str]):
        """Remove a plugin from the registry.

        Args:
            name: Name of the plugin to remove
            plugin_type: Type of plugin
        """
        # Convert string to enum if needed
        if isinstance(plugin_type, str):
            plugin_type = PluginType(plugin_type.lower())

        # Select appropriate registry
        if plugin_type == PluginType.MODEL:
            registry = self._models
        elif plugin_type == PluginType.TRANSFORM:
            registry = self._transforms
        else:
            return

        # Remove if exists
        if name in registry:
            del registry[name]

    def clear(self):
        """Clear all registered plugins."""
        self._models.clear()
        self._transforms.clear()

    def __len__(self) -> int:
        """Get total number of registered plugins.

        Returns:
            Total count of models and transforms
        """
        return len(self._models) + len(self._transforms)

    def __contains__(self, name: str) -> bool:
        """Check if a plugin name is registered.

        Args:
            name: Plugin name to check

        Returns:
            True if registered as either model or transform
        """
        return name in self._models or name in self._transforms

    def get_stats(self) -> Dict[str, int]:
        """Get registration statistics.

        Returns:
            Dictionary with counts of registered plugins
        """
        return {
            "total": len(self),
            "models": len(self._models),
            "transforms": len(self._transforms)
        }

    def discover(self, module_name: str):
        """Discover and register plugins from a module.

        Args:
            module_name: Name of the module to import and scan
        """
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return

        # Scan module for plugins
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # Check if it's a model
                if hasattr(obj, 'fit') and hasattr(obj, 'predict'):
                    self.register(name, obj, PluginType.MODEL, validate=False)
                # Check if it's a transform
                elif hasattr(obj, 'transform'):
                    self.register(name, obj, PluginType.TRANSFORM, validate=False)

    def discover_directory(self, path: str):
        """Discover plugins in a directory.

        Args:
            path: Path to directory containing plugin modules
        """
        if not os.path.exists(path):
            return

        # Add path to Python path temporarily
        import sys
        sys.path.insert(0, path)

        try:
            # Scan for Python files
            for filename in os.listdir(path):
                if filename.endswith('.py') and not filename.startswith('_'):
                    module_name = filename[:-3]
                    self.discover(module_name)
        finally:
            # Remove temporary path
            sys.path.pop(0)

    def create_instance(
        self,
        name: str,
        plugin_type: Union[PluginType, str],
        *args,
        **kwargs
    ) -> Any:
        """Create an instance of a registered plugin.

        Args:
            name: Name of the plugin
            plugin_type: Type of plugin
            *args: Positional arguments for plugin constructor
            **kwargs: Keyword arguments for plugin constructor

        Returns:
            Instance of the plugin class

        Raises:
            KeyError: If plugin not found
        """
        plugin_class = self.get(name, plugin_type, raise_on_missing=True)
        return plugin_class(*args, **kwargs)

    def find_compatible(self, **criteria) -> List[str]:
        """Find plugins matching certain criteria.

        Args:
            **criteria: Criteria to match against plugin metadata

        Returns:
            List of plugin names matching all criteria
        """
        compatible = []

        # Check models
        for name, info in self._models.items():
            if self._matches_criteria(info.metadata, criteria):
                compatible.append(name)

        # Check transforms
        for name, info in self._transforms.items():
            if self._matches_criteria(info.metadata, criteria):
                compatible.append(name)

        return compatible

    def _matches_criteria(self, metadata: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
        """Check if metadata matches all criteria.

        Args:
            metadata: Plugin metadata
            criteria: Criteria to match

        Returns:
            True if all criteria match
        """
        for key, value in criteria.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def export_state(self) -> Dict[str, Any]:
        """Export registry state for serialization.

        Returns:
            Dictionary representation of registry state
        """
        return {
            "models": {
                name: {
                    "class_name": info.plugin_class.__name__,
                    "module": info.plugin_class.__module__,
                    "metadata": info.metadata
                }
                for name, info in self._models.items()
            },
            "transforms": {
                name: {
                    "class_name": info.plugin_class.__name__,
                    "module": info.plugin_class.__module__,
                    "metadata": info.metadata
                }
                for name, info in self._transforms.items()
            }
        }

    def import_state(self, state: Dict[str, Any]):
        """Import registry state from serialization.

        Args:
            state: Dictionary representation of registry state
        """
        # Import models
        for name, info in state.get("models", {}).items():
            try:
                module = importlib.import_module(info["module"])
                plugin_class = getattr(module, info["class_name"])
                self.register(name, plugin_class, PluginType.MODEL,
                            metadata=info.get("metadata", {}), force=True)
            except (ImportError, AttributeError):
                continue

        # Import transforms
        for name, info in state.get("transforms", {}).items():
            try:
                module = importlib.import_module(info["module"])
                plugin_class = getattr(module, info["class_name"])
                self.register(name, plugin_class, PluginType.TRANSFORM,
                            metadata=info.get("metadata", {}), force=True)
            except (ImportError, AttributeError):
                continue

    # Decorator methods for easy registration
    def model(self, name: Optional[str] = None, **metadata):
        """Decorator for registering a model.

        Args:
            name: Optional name for the model (uses class name if not provided)
            **metadata: Additional metadata for the model

        Returns:
            Decorator function
        """
        def decorator(cls):
            model_name = name or cls.__name__
            self.register(model_name, cls, PluginType.MODEL, metadata=metadata)
            return cls
        return decorator

    def transform(self, name: Optional[str] = None, **metadata):
        """Decorator for registering a transform.

        Args:
            name: Optional name for the transform (uses class name if not provided)
            **metadata: Additional metadata for the transform

        Returns:
            Decorator function
        """
        def decorator(cls):
            transform_name = name or cls.__name__
            self.register(transform_name, cls, PluginType.TRANSFORM, metadata=metadata)
            return cls
        return decorator


class ModelRegistry:
    """Convenient interface for model registration and creation.

    This class provides a simplified API specifically for models,
    delegating to the main Registry singleton.

    Example:
        >>> @ModelRegistry.register('maxwell')
        >>> class Maxwell(BaseModel):
        ...     pass
        >>>
        >>> model = ModelRegistry.create('maxwell')
        >>> models = ModelRegistry.list_models()
    """

    _registry = None

    @classmethod
    def _get_registry(cls) -> Registry:
        """Get the global registry instance."""
        if cls._registry is None:
            cls._registry = Registry.get_instance()
        return cls._registry

    @classmethod
    def register(cls, name: str, **metadata):
        """Decorator for registering a model.

        Args:
            name: Name for the model
            **metadata: Additional metadata for the model

        Returns:
            Decorator function

        Example:
            >>> @ModelRegistry.register('maxwell')
            >>> class Maxwell(BaseModel):
            ...     pass
        """
        def decorator(model_class):
            registry = cls._get_registry()
            registry.register(name, model_class, PluginType.MODEL, metadata=metadata)
            return model_class
        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        """Create a model instance by name (factory method).

        Args:
            name: Name of the model to create
            *args: Positional arguments for model constructor
            **kwargs: Keyword arguments for model constructor

        Returns:
            Instance of the model class

        Raises:
            KeyError: If model not found

        Example:
            >>> model = ModelRegistry.create('maxwell')
        """
        registry = cls._get_registry()
        return registry.create_instance(name, PluginType.MODEL, *args, **kwargs)

    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names (discovery).

        Returns:
            List of registered model names

        Example:
            >>> models = ModelRegistry.list_models()
            >>> print(models)
            ['maxwell', 'zener', 'springpot', ...]
        """
        registry = cls._get_registry()
        return registry.get_all_models()

    @classmethod
    def get_info(cls, name: str) -> Optional[PluginInfo]:
        """Get information about a registered model.

        Args:
            name: Name of the model

        Returns:
            PluginInfo object with model details

        Example:
            >>> info = ModelRegistry.get_info('maxwell')
            >>> print(info.doc)
            Maxwell viscoelastic model...
        """
        registry = cls._get_registry()
        return registry.get_info(name, PluginType.MODEL)

    @classmethod
    def unregister(cls, name: str):
        """Unregister a model.

        Args:
            name: Name of the model to remove
        """
        registry = cls._get_registry()
        registry.unregister(name, PluginType.MODEL)


class TransformRegistry:
    """Convenient interface for transform registration and creation.

    This class provides a simplified API specifically for transforms,
    delegating to the main Registry singleton.

    Example:
        >>> @TransformRegistry.register('fft_analysis')
        >>> class RheoAnalysis(BaseTransform):
        ...     pass
        >>>
        >>> transform = TransformRegistry.create('fft_analysis')
        >>> transforms = TransformRegistry.list_transforms()
    """

    _registry = None

    @classmethod
    def _get_registry(cls) -> Registry:
        """Get the global registry instance."""
        if cls._registry is None:
            cls._registry = Registry.get_instance()
        return cls._registry

    @classmethod
    def register(cls, name: str, **metadata):
        """Decorator for registering a transform.

        Args:
            name: Name for the transform
            **metadata: Additional metadata for the transform

        Returns:
            Decorator function

        Example:
            >>> @TransformRegistry.register('fft_analysis')
            >>> class RheoAnalysis(BaseTransform):
            ...     pass
        """
        def decorator(transform_class):
            registry = cls._get_registry()
            registry.register(name, transform_class, PluginType.TRANSFORM, metadata=metadata)
            return transform_class
        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> Any:
        """Create a transform instance by name (factory method).

        Args:
            name: Name of the transform to create
            *args: Positional arguments for transform constructor
            **kwargs: Keyword arguments for transform constructor

        Returns:
            Instance of the transform class

        Raises:
            KeyError: If transform not found

        Example:
            >>> transform = TransformRegistry.create('fft_analysis')
        """
        registry = cls._get_registry()
        return registry.create_instance(name, PluginType.TRANSFORM, *args, **kwargs)

    @classmethod
    def list_transforms(cls) -> List[str]:
        """List all registered transform names (discovery).

        Returns:
            List of registered transform names

        Example:
            >>> transforms = TransformRegistry.list_transforms()
            >>> print(transforms)
            ['fft_analysis', 'mastercurve', 'owchirp', ...]
        """
        registry = cls._get_registry()
        return registry.get_all_transforms()

    @classmethod
    def get_info(cls, name: str) -> Optional[PluginInfo]:
        """Get information about a registered transform.

        Args:
            name: Name of the transform

        Returns:
            PluginInfo object with transform details

        Example:
            >>> info = TransformRegistry.get_info('fft_analysis')
            >>> print(info.doc)
            FFT-based rheological analysis...
        """
        registry = cls._get_registry()
        return registry.get_info(name, PluginType.TRANSFORM)

    @classmethod
    def unregister(cls, name: str):
        """Unregister a transform.

        Args:
            name: Name of the transform to remove
        """
        registry = cls._get_registry()
        registry.unregister(name, PluginType.TRANSFORM)