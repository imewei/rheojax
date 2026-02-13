"""Core abstractions and common functionality for the rheojax package.

This module provides:
- Base classes for models and transforms
- RheoData container for rheological data
- Parameter management system
- Plugin registry for extensibility
"""

from .base import BaseModel, BaseTransform, TransformPipeline
from .data import RheoData
from .parameters import (
    Parameter,
    ParameterConstraint,
    ParameterOptimizer,
    ParameterSet,
    SharedParameterSet,
)
from .registry import ModelRegistry, PluginInfo, PluginType, Registry, TransformRegistry
from .test_modes import (
    DeformationMode,
    TestMode,
    detect_test_mode,
    validate_test_mode,
)

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
    # Test modes and deformation geometry
    "DeformationMode",
    "TestMode",
    "detect_test_mode",
    "validate_test_mode",
]

# Initialize global registry
_registry = Registry.get_instance()
