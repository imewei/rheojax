"""Core abstractions and common functionality for the rheo package.

This module provides:
- Base classes for models and transforms
- RheoData wrapper around piblin Measurement
- Parameter management system
- Plugin registry for extensibility
"""

from .base import BaseModel, BaseTransform, TransformPipeline
from .data import RheoData
from .parameters import (
    Parameter,
    ParameterSet,
    ParameterConstraint,
    SharedParameterSet,
    ParameterOptimizer
)
from .registry import Registry, PluginType, PluginInfo, ModelRegistry, TransformRegistry
from .test_modes import TestMode, detect_test_mode, validate_test_mode

__all__ = [
    # Base classes
    "BaseModel",
    "BaseTransform",
    "TransformPipeline",

    # Data wrapper
    "RheoData",

    # Parameters
    "Parameter",
    "ParameterSet",
    "ParameterConstraint",
    "SharedParameterSet",
    "ParameterOptimizer",

    # Registry
    "Registry",
    "PluginType",
    "PluginInfo",
    "ModelRegistry",
    "TransformRegistry",

    # Test modes
    "TestMode",
    "detect_test_mode",
    "validate_test_mode",
]

# Initialize global registry
_registry = Registry.get_instance()