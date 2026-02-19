"""Parameter management system for models and transforms.

This module provides classes for managing parameters, constraints,
and optimization support for rheological models.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
HAS_JAX = True

# Module-level logger
logger = get_logger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing helper only
    import jax.numpy as jnp_typing
else:
    jnp_typing = np


type ArrayLike = np.ndarray | jnp_typing.ndarray | list | tuple


def _coerce_array(values: ArrayLike) -> np.ndarray:
    """Convert array-like inputs to NumPy arrays without altering callers."""
    if isinstance(values, np.ndarray):
        return values
    if HAS_JAX and isinstance(values, jnp.ndarray):
        return np.asarray(values)
    return np.asarray(values)


@dataclass
class ParameterConstraint:
    """Constraint on a parameter value."""

    type: str  # 'bounds', 'positive', 'integer', 'fixed', 'relative', 'custom'
    min_value: float | None = None
    max_value: float | None = None
    value: float | None = None  # For fixed constraints
    relation: str | None = None  # For relative constraints
    other_param: str | None = None  # For relative constraints
    validator: Callable | None = None  # For custom constraints

    def validate(self, value: float, context: dict[str, float] | None = None) -> bool:
        """Check if value satisfies the constraint.

        Args:
            value: Value to validate
            context: Context with other parameter values (for relative constraints)

        Returns:
            True if constraint is satisfied
        """
        # NaN/Inf bypass IEEE 754 comparisons — reject unconditionally
        if not np.isfinite(value):
            return False

        if self.type == "bounds":
            if self.min_value is not None and value < self.min_value:
                logger.debug(
                    "Bound check failed: value below minimum",
                    constraint_type=self.type,
                    value=value,
                    min_value=self.min_value,
                )
                return False
            if self.max_value is not None and value > self.max_value:
                logger.debug(
                    "Bound check failed: value above maximum",
                    constraint_type=self.type,
                    value=value,
                    max_value=self.max_value,
                )
                return False
            return True

        elif self.type == "positive":
            return value > 0

        elif self.type == "integer":
            return float(value).is_integer()

        elif self.type == "fixed":
            return value == self.value

        elif self.type == "relative" and context:
            if self.other_param not in context:
                return True  # Can't validate without context

            other_value = context[self.other_param]

            if self.relation == "less_than":
                return value < other_value
            elif self.relation == "greater_than":
                return value > other_value
            elif self.relation == "equal":
                return value == other_value

        elif self.type == "custom" and self.validator:
            return self.validator(value)

        return True


class Parameter:
    """Single parameter with value, bounds, and metadata.

    A Parameter represents a model parameter with support for bounds validation,
    units tracking, and constraint enforcement. Parameters can be used in both
    NLSQ optimization and Bayesian inference workflows.

    Attributes:
        name: Parameter identifier used for lookup and serialization.
        value: Current parameter value (may be None if unset).
        bounds: Lower and upper bounds as tuple (min, max).
        units: Physical units string for display (e.g., "Pa", "s").
        description: Human-readable description.
        constraints: List of ParameterConstraint objects for validation.

    Example:
        >>> param = Parameter("G0", value=1e5, bounds=(1e3, 1e9), units="Pa")
        >>> param.value = 2e5  # Validated against bounds
        >>> param.validate()
        True
    """

    __slots__ = (
        "name",
        "_bounds",
        "units",
        "description",
        "constraints",
        "_value",
        "_clamp_on_set",
        "_was_clamped",
    )

    def __init__(
        self,
        name: str,
        value: float | None = None,
        bounds: tuple[float, float] | None = None,
        units: str | None = None,
        description: str | None = None,
        constraints: list[ParameterConstraint] | None = None,
    ) -> None:
        self.name = name
        self._bounds: tuple[float, float] | None = bounds
        self.units = units
        self.description = description
        self.constraints = list(constraints) if constraints else []
        self._value: float | None = None
        self._clamp_on_set = False
        self._was_clamped = False
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Creating parameter",
                parameter=name,
                bounds=bounds,
                units=units,
            )
        self._initialize(value)

    @property
    def bounds(self) -> tuple[float, float] | None:
        """Get parameter bounds."""
        return self._bounds

    @bounds.setter
    def bounds(self, new_bounds: tuple[float, float] | None) -> None:
        """Set parameter bounds and sync any bounds constraint."""
        self._bounds = new_bounds
        # Sync bounds constraint if constraints list exists
        if hasattr(self, "constraints"):
            for c in self.constraints:
                if hasattr(c, "type") and c.type == "bounds":
                    if new_bounds is not None:
                        c.min_value = new_bounds[0]
                        c.max_value = new_bounds[1]
                    break

    def _initialize(self, value: float | None) -> None:
        """Validate parameter after initialization."""
        if self.bounds is not None:
            lower, upper = self.bounds
            lower = float(lower)
            upper = float(upper)
            if lower > upper:
                logger.error(
                    "Invalid bounds: lower > upper",
                    parameter=self.name,
                    bounds=(lower, upper),
                    exc_info=True,
                )
                raise ValueError(
                    f"Invalid bounds for parameter '{self.name}': {(lower, upper)}"
                )
            self.bounds = (lower, upper)

        # Add bounds as constraint if specified and not already present
        if self.bounds:
            has_bounds_constraint = any(c.type == "bounds" for c in self.constraints)
            if not has_bounds_constraint:
                self.constraints.insert(
                    0,
                    ParameterConstraint(
                        type="bounds",
                        min_value=self.bounds[0],
                        max_value=self.bounds[1],
                    ),
                )

        if value is not None:
            self._clamp_on_set = True
            self.value = value
            self._clamp_on_set = False

    @property
    def value(self) -> float | None:
        """Get parameter value."""
        return self._value

    @value.setter
    def value(self, val: float | None) -> None:
        """Set parameter value with validation."""
        if val is None:
            self._value = None
            self._was_clamped = False
            return

        try:
            numeric_val = float(val)
        except (TypeError, ValueError) as exc:
            logger.error(
                "Failed to convert value to numeric",
                parameter=self.name,
                value=val,
                exc_info=True,
            )
            raise ValueError(
                f"Parameter '{self.name}' requires a numeric value"
            ) from exc

        if not np.isfinite(numeric_val):
            logger.error(
                "Non-finite value received",
                parameter=self.name,
                value=numeric_val,
                exc_info=True,
            )
            raise ValueError(f"Parameter '{self.name}' received non-finite value")

        clamped_during_init = False
        if self.bounds:
            lower, upper = self.bounds
            _debug = logger.isEnabledFor(10)  # logging.DEBUG == 10
            if _debug:
                logger.debug(
                    "Bound check",
                    parameter=self.name,
                    value=numeric_val,
                    bounds=self.bounds,
                )
            if self._clamp_on_set:
                if numeric_val < lower:
                    warnings.warn(
                        f"Parameter '{self.name}' initialized below bounds; clamped to {lower}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if _debug:
                        logger.debug(
                            "Value clamped to lower bound",
                            parameter=self.name,
                            original_value=numeric_val,
                            clamped_value=lower,
                        )
                    numeric_val = lower
                    clamped_during_init = True
                elif numeric_val > upper:
                    warnings.warn(
                        f"Parameter '{self.name}' initialized above bounds; clamped to {upper}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if _debug:
                        logger.debug(
                            "Value clamped to upper bound",
                            parameter=self.name,
                            original_value=numeric_val,
                            clamped_value=upper,
                        )
                    numeric_val = upper
                    clamped_during_init = True
            elif numeric_val < lower or numeric_val > upper:
                logger.error(
                    "Value out of bounds",
                    parameter=self.name,
                    value=numeric_val,
                    bounds=self.bounds,
                    exc_info=True,
                )
                raise ValueError(f"Value {numeric_val} out of bounds {self.bounds}")

        if self._clamp_on_set:
            self._was_clamped = clamped_during_init
        else:
            self._was_clamped = False

        self._value = numeric_val

    @property
    def was_clamped(self) -> bool:
        """Return True if the last assignment clamped the value."""
        return self._was_clamped

    def validate(self, value: float, context: dict[str, float] | None = None) -> bool:
        """Validate value against all constraints.

        Args:
            value: Value to validate
            context: Context with other parameter values

        Returns:
            True if all constraints are satisfied
        """
        for constraint in self.constraints:
            if not constraint.validate(value, context):
                logger.debug(
                    "Constraint validation failed",
                    parameter=self.name,
                    value=value,
                    constraint_type=constraint.type,
                )
                return False
        return True

    def __hash__(self) -> int:
        """Make Parameter hashable for use as dict keys.

        Returns:
            Hash based on name, value, bounds, and units
        """
        return hash((self.name, self.value, self.bounds, self.units))

    def __eq__(self, other: object) -> bool:
        """Check equality with another Parameter.

        Args:
            other: Object to compare with

        Returns:
            True if parameters are equal
        """
        if not isinstance(other, Parameter):
            return NotImplemented
        return (
            self.name == other.name
            and self.value == other.value
            and self.bounds == other.bounds
            and self.units == other.units
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        d = {
            "name": self.name,
            "value": self.value,
            "bounds": self.bounds,
            "units": self.units,
            "description": self.description,
        }
        if self.constraints:
            d["constraints"] = [
                {
                    "type": c.type,
                    "min_value": getattr(c, "min_value", None),
                    "max_value": getattr(c, "max_value", None),
                }
                for c in self.constraints
            ]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Parameter:
        """Create from dictionary representation."""
        param = cls(
            name=data["name"],
            value=data.get("value"),
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            units=data.get("units"),
            description=data.get("description"),
        )
        if "constraints" in data:
            for c_data in data["constraints"]:
                param.constraints.append(
                    ParameterConstraint(
                        type=c_data["type"],
                        min_value=c_data.get("min_value"),
                        max_value=c_data.get("max_value"),
                    )
                )
        return param


class ParameterSet:
    """Collection of parameters for a model or transform.

    A ParameterSet manages multiple Parameter objects with dict-like access,
    batch operations, and serialization support. It is the primary interface
    for working with model parameters in RheoJAX.

    Key Features:
        - Dict-like access: ``params["G0"]`` or ``params.get("G0")``
        - Batch operations: ``get_values()``, ``set_values()``, ``get_bounds()``
        - Unpack helper: ``G0, eta = params.unpack("G0", "eta")``
        - Serialization: ``to_dict()`` / ``from_dict()`` for JSON/HDF5

    Example:
        >>> params = ParameterSet()
        >>> params.add("G0", value=1e5, bounds=(1e3, 1e9), units="Pa")
        >>> params.add("eta", value=1e3, bounds=(1e-3, 1e9), units="Pa*s")
        >>> G0, eta = params.unpack("G0", "eta")
        >>> print(f"G0={G0:.2e}, eta={eta:.2e}")
        G0=1.00e+05, eta=1.00e+03

    See Also:
        Parameter: Individual parameter class.
        SharedParameterSet: For multi-model parameter sharing.
    """

    __slots__ = ("_parameters", "_order", "_has_relative_constraints")

    def __init__(self):
        """Initialize empty parameter set."""
        self._parameters: dict[str, Parameter] = {}
        self._order: list[str] = []
        self._has_relative_constraints: bool = False
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug("ParameterSet created")

    def add(
        self,
        name: str,
        value: float | None = None,
        bounds: tuple[float, float] | None = None,
        units: str | None = None,
        description: str | None = None,
        constraints: list[ParameterConstraint] | None = None,
    ) -> Parameter:
        """Add a parameter to the set.

        Args:
            name: Parameter name
            value: Initial value
            bounds: Value bounds (min, max)
            units: Parameter units
            description: Parameter description
            constraints: List of constraints

        Returns:
            The created Parameter object
        """
        _debug = logger.isEnabledFor(10)  # logging.DEBUG == 10
        if _debug:
            logger.debug(
                "Adding parameter to set",
                operation="add",
                parameter=name,
                value=value,
                bounds=bounds,
            )
        param = Parameter(
            name=name,
            value=value,
            bounds=bounds,
            units=units,
            description=description,
            constraints=constraints or [],
        )

        self._parameters[name] = param
        if name not in self._order:
            self._order.append(name)

        # Track whether any relative constraints exist (for set_value optimization)
        if constraints:
            if any(c.type == "relative" for c in constraints):
                self._has_relative_constraints = True

        if _debug:
            logger.debug(
                "Parameter added",
                operation="add",
                params=list(self._parameters.keys()),
            )
        return param

    def get(self, name: str) -> Parameter | None:
        """Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter object or None if not found
        """
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Getting parameter",
                operation="get",
                parameter=name,
            )
        return self._parameters.get(name)

    def set_value(self, name: str, value: float):
        """Set parameter value.

        Args:
            name: Parameter name
            value: New value

        Raises:
            KeyError: If parameter not found
            ValueError: If value violates constraints
        """
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Setting parameter value",
                operation="set_value",
                parameter=name,
                value=value,
            )
        if name not in self._parameters:
            logger.error(
                "Parameter not found",
                parameter=name,
                available_params=list(self._parameters.keys()),
                exc_info=True,
            )
            raise KeyError(f"Parameter '{name}' not found")

        param = self._parameters[name]

        # Build context dict only when relative constraints exist (>99% of models skip this)
        context: dict[str, float] | None = None
        if self._has_relative_constraints:
            context = {
                p.name: p.value for p in self._parameters.values() if p.value is not None
            }
        if not param.validate(value, context):
            logger.error(
                "Value violates constraints",
                parameter=name,
                value=value,
                exc_info=True,
            )
            raise ValueError(
                f"Value {value} violates constraints for parameter '{name}'"
            )

        param.value = value

    def set_bounds(self, name: str, bounds: tuple[float, float]):
        """Set bounds for a parameter.

        Args:
            name: Parameter name
            bounds: Tuple of (min, max) values

        Raises:
            KeyError: If parameter not found
            ValueError: If bounds are invalid
        """
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Setting parameter bounds",
                operation="set_bounds",
                parameter=name,
                bounds=bounds,
            )
        if name not in self._parameters:
            logger.error(
                "Parameter not found",
                parameter=name,
                available_params=list(self._parameters.keys()),
                exc_info=True,
            )
            raise KeyError(f"Parameter '{name}' not found")

        min_val, max_val = bounds
        if min_val >= max_val:
            logger.error(
                "Invalid bounds: min >= max",
                parameter=name,
                min_val=min_val,
                max_val=max_val,
                exc_info=True,
            )
            raise ValueError(
                f"Invalid bounds: min ({min_val}) must be < max ({max_val})"
            )

        param = self._parameters[name]
        # bounds.setter auto-syncs the associated bounds constraint
        param.bounds = bounds

    def get_values(self) -> np.ndarray:
        """Get all parameter values as array.

        Returns:
            Array of parameter values in order
        """
        values = []
        for name in self._order:
            param = self._parameters[name]
            if param.value is not None:
                values.append(param.value)
            else:
                logger.warning(
                    "Parameter has no value set, defaulting to 0.0",
                    parameter=name,
                    bounds=param.bounds,
                )
                values.append(0.0)
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Getting all parameter values",
                operation="get_values",
                params=list(self._parameters.keys()),
                num_params=len(values),
            )
        return np.array(values, dtype=np.float64)

    def set_values(self, values: ArrayLike | dict[str, float]):
        """Set parameter values from array or dictionary.

        Args:
            values: Array of values in order, or dict mapping names to values

        Raises:
            ValueError: If wrong number of values (array) or unknown parameter (dict)
        """
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Setting multiple parameter values",
                operation="set_values",
                params=list(self._parameters.keys()),
            )
        if isinstance(values, dict):
            for name, value in values.items():
                if name not in self._parameters:
                    logger.error(
                        "Unknown parameter in dict",
                        parameter=name,
                        available_params=list(self._parameters.keys()),
                        exc_info=True,
                    )
                    raise ValueError(f"Unknown parameter: {name}")
                self.set_value(name, float(value))
        else:
            values = np.atleast_1d(values)
            if len(values) != len(self._order):
                logger.error(
                    "Wrong number of values",
                    expected=len(self._order),
                    got=len(values),
                    exc_info=True,
                )
                raise ValueError(
                    f"Expected {len(self._order)} values, got {len(values)}"
                )
            for name, value in zip(self._order, values, strict=False):
                self.set_value(name, float(value))

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get bounds for all parameters.

        Returns:
            List of (min, max) tuples
        """
        bounds: list[tuple[float | None, float | None]] = []
        for name in self._order:
            param = self._parameters[name]
            if param.bounds:
                bounds.append(param.bounds)
            else:
                bounds.append((None, None))
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Getting all parameter bounds",
                operation="get_bounds",
                params=list(self._parameters.keys()),
                num_params=len(bounds),
            )
        return bounds

    def get_value(self, name: str) -> float | None:
        """Get value of a specific parameter.

        Args:
            name: Parameter name

        Returns:
            Parameter value or None
        """
        param = self.get(name)
        return param.value if param else None

    def unpack(self, *names: str) -> tuple[float | None, ...]:
        """Extract multiple parameter values in a single call.

        This method provides a concise way to extract several parameter values
        at once, reducing boilerplate in model implementations.

        Args:
            *names: Parameter names to extract

        Returns:
            Tuple of parameter values in the same order as requested.
            Returns None for parameters with None values.

        Raises:
            KeyError: If any parameter name is not found. The error message
                includes the missing name and lists available parameters.

        Examples:
            Basic usage - extract multiple parameters in one line:

            >>> params = ParameterSet()
            >>> _ = params.add('x', value=1.5)
            >>> _ = params.add('G0', value=100.0)
            >>> _ = params.add('tau0', value=0.01)
            >>> x, G0, tau0 = params.unpack('x', 'G0', 'tau0')
            >>> x
            1.5
            >>> G0
            100.0

            Before (verbose)::

                x = params.get_value('x')
                G0 = params.get_value('G0')
                tau0 = params.get_value('tau0')

            After (concise)::

                x, G0, tau0 = params.unpack('x', 'G0', 'tau0')
        """
        values = []
        for name in names:
            if name not in self._parameters:
                available = list(self._parameters.keys())
                raise KeyError(
                    f"Parameter '{name}' not found. "
                    f"Available parameters: {available}"
                )
            values.append(self.get_value(name))
        return tuple(values)

    def __len__(self) -> int:
        """Number of parameters."""
        return len(self._parameters)

    def __contains__(self, name: str) -> bool:
        """Check if parameter exists."""
        return name in self._parameters

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self._order)

    def keys(self):
        """Return an iterator over parameter names (dict-like interface).

        Returns:
            Iterator over parameter names in order

        Examples:
            >>> params = ParameterSet()
            >>> params.add('alpha', value=0.5)
            >>> params.add('beta', value=1.0)
            >>> list(params.keys())
            ['alpha', 'beta']
        """
        return iter(self._order)

    def values(self):
        """Return an iterator over Parameter objects (dict-like interface).

        Returns:
            Iterator over Parameter objects in order

        Examples:
            >>> params = ParameterSet()
            >>> params.add('alpha', value=0.5, units='')
            >>> for param in params.values():
            ...     print(f"{param.name}: {param.value}")
            alpha: 0.5
        """
        for name in self._order:
            yield self._parameters[name]

    def items(self):
        """Return an iterator over (name, Parameter) tuples (dict-like interface).

        Returns:
            Iterator over (name, Parameter) tuples in order

        Examples:
            >>> params = ParameterSet()
            >>> params.add('alpha', value=0.5)
            >>> for name, param in params.items():
            ...     print(f"{name}: {param.value}")
            alpha: 0.5
        """
        for name in self._order:
            yield name, self._parameters[name]

    def __getitem__(self, key: str) -> Parameter:
        """Get parameter by name using subscript notation.

        Args:
            key: Parameter name

        Returns:
            Parameter object

        Raises:
            KeyError: If parameter not found

        Examples:
            >>> params = ParameterSet()
            >>> params.add('alpha', value=0.5)
            >>> param = params['alpha']  # Get parameter object
            >>> value = params['alpha'].value  # Get value
        """
        if key not in self._parameters:
            logger.error(
                "Parameter not found in subscript access",
                parameter=key,
                available_params=list(self._parameters.keys()),
                exc_info=True,
            )
            raise KeyError(f"Parameter '{key}' not found in ParameterSet")
        return self._parameters[key]

    def __setitem__(self, key: str, value: float | Parameter):
        """Set parameter value using subscript notation.

        Args:
            key: Parameter name
            value: New value (float) or Parameter object

        Raises:
            KeyError: If parameter not found and value is float
            ValueError: If value violates constraints

        Examples:
            >>> params = ParameterSet()
            >>> params.add('alpha', value=0.5, bounds=(0, 1))
            >>> params['alpha'] = 0.7  # Set value
            >>> # Or replace entire parameter:
            >>> params['alpha'] = Parameter('alpha', value=0.8, bounds=(0, 1))
        """
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Setting parameter via subscript",
                operation="__setitem__",
                parameter=key,
            )
        if isinstance(value, Parameter):
            # Replace entire parameter
            self._parameters[key] = value
            if key not in self._order:
                self._order.append(key)
        else:
            # Set value only
            if key not in self._parameters:
                logger.error(
                    "Parameter not found for subscript assignment",
                    parameter=key,
                    available_params=list(self._parameters.keys()),
                    exc_info=True,
                )
                raise KeyError(
                    f"Parameter '{key}' not found. Use add() to create new parameters."
                )
            self.set_value(key, float(value))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to dictionary representation."""
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Converting ParameterSet to dict",
                operation="to_dict",
                params=list(self._parameters.keys()),
            )
        return {name: self._parameters[name].to_dict() for name in self._order}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSet:
        """Create from dictionary representation."""
        if logger.isEnabledFor(10):  # logging.DEBUG == 10
            logger.debug(
                "Creating ParameterSet from dict",
                operation="from_dict",
                params=list(data.keys()),
            )
        params = cls()
        for name, param_data in data.items():
            if isinstance(param_data, dict):
                params.add(
                    name=name,
                    value=param_data.get("value"),
                    bounds=(
                        tuple(param_data["bounds"])
                        if param_data.get("bounds")
                        else None
                    ),
                    units=param_data.get("units"),
                    description=param_data.get("description"),
                )
        return params


class SharedParameterSet:
    """Manages parameters shared across multiple models."""

    def __init__(self):
        """Initialize shared parameter set."""
        self._shared: dict[str, Parameter] = {}
        self._links: dict[str, list[Any]] = {}  # Parameter -> list of linked objects
        self._groups: dict[str, list[str]] = {}  # Group name -> parameter names
        logger.debug("SharedParameterSet created")

    def add_shared(
        self,
        name: str,
        value: float | None = None,
        bounds: tuple[float, float] | None = None,
        units: str | None = None,
        constraints: list[ParameterConstraint] | None = None,
        group: str | None = None,
    ) -> Parameter:
        """Add a shared parameter.

        Args:
            name: Parameter name
            value: Initial value
            bounds: Value bounds
            units: Parameter units
            constraints: Parameter constraints
            group: Optional group name

        Returns:
            The created Parameter
        """
        logger.debug(
            "Adding shared parameter",
            operation="add_shared",
            parameter=name,
            value=value,
            bounds=bounds,
            group=group,
        )
        param = Parameter(
            name=name,
            value=value,
            bounds=bounds,
            units=units,
            constraints=constraints or [],
        )

        self._shared[name] = param
        self._links[name] = []

        if group:
            if group not in self._groups:
                self._groups[group] = []
            self._groups[group].append(name)

        logger.debug(
            "Shared parameter added",
            operation="add_shared",
            params=list(self._shared.keys()),
        )
        return param

    def link_model(self, model: Any, param_name: str):
        """Link a model to a shared parameter.

        Args:
            model: Model to link
            param_name: Name of shared parameter
        """
        logger.debug(
            "Linking model to shared parameter",
            operation="link_model",
            parameter=param_name,
        )
        if param_name not in self._shared:
            logger.error(
                "Shared parameter not found for linking",
                parameter=param_name,
                available_params=list(self._shared.keys()),
                exc_info=True,
            )
            raise KeyError(f"Shared parameter '{param_name}' not found")

        if model not in self._links[param_name]:
            self._links[param_name].append(model)

    def link_parameter_set(self, param_set: ParameterSet, param_name: str):
        """Link a parameter set to a shared parameter.

        Args:
            param_set: ParameterSet to link
            param_name: Name of shared parameter
        """
        logger.debug(
            "Linking ParameterSet to shared parameter",
            operation="link_parameter_set",
            parameter=param_name,
        )
        if param_name not in self._shared:
            logger.error(
                "Shared parameter not found for linking",
                parameter=param_name,
                available_params=list(self._shared.keys()),
                exc_info=True,
            )
            raise KeyError(f"Shared parameter '{param_name}' not found")

        if param_set not in self._links[param_name]:
            self._links[param_name].append(param_set)

    def set_value(self, name: str, value: float):
        """Set shared parameter value.

        Args:
            name: Parameter name
            value: New value

        Raises:
            ValueError: If value violates constraints
        """
        logger.debug(
            "Setting shared parameter value",
            operation="set_value",
            parameter=name,
            value=value,
        )
        if name not in self._shared:
            logger.error(
                "Shared parameter not found",
                parameter=name,
                available_params=list(self._shared.keys()),
                exc_info=True,
            )
            raise KeyError(f"Shared parameter '{name}' not found")

        param = self._shared[name]

        # Validate
        if not param.validate(value):
            logger.error(
                "Value violates constraints for shared parameter",
                parameter=name,
                value=value,
                exc_info=True,
            )
            raise ValueError(
                f"Value {value} violates constraints for parameter '{name}'"
            )

        param.value = value

        # Update linked models/parameter sets
        for linked in self._links.get(name, []):
            if (
                hasattr(linked, "set_value")
                and hasattr(linked, "__contains__")
                and name in linked
            ):
                # This is a ParameterSet with the parameter
                linked.set_value(name, value)
            elif hasattr(linked, "parameters") and name in linked.parameters:
                # This is a model with parameters
                linked.parameters.set_value(name, value)

    def get_value(self, name: str) -> float | None:
        """Get shared parameter value.

        Args:
            name: Parameter name

        Returns:
            Parameter value or None
        """
        param = self._shared.get(name)
        return param.value if param else None

    def get_linked_models(self, param_name: str) -> list[Any]:
        """Get models linked to a parameter.

        Args:
            param_name: Parameter name

        Returns:
            List of linked models
        """
        return self._links.get(param_name, [])

    def create_group(self, group_name: str, param_names: list[str]):
        """Create a parameter group.

        Args:
            group_name: Name for the group
            param_names: Parameter names to include
        """
        logger.debug(
            "Creating parameter group",
            operation="create_group",
            group=group_name,
            params=param_names,
        )
        self._groups[group_name] = param_names

    def get_group(self, group_name: str) -> list[str]:
        """Get parameters in a group.

        Args:
            group_name: Group name

        Returns:
            List of parameter names in group
        """
        return self._groups.get(group_name, [])

    def __contains__(self, name: str) -> bool:
        """Check if shared parameter exists."""
        return name in self._shared


class ParameterOptimizer:
    """Optimizer for parameter fitting."""

    def __init__(
        self,
        parameters: ParameterSet,
        use_jax: bool = False,
        track_history: bool = False,
    ):
        """Initialize parameter optimizer.

        Args:
            parameters: ParameterSet to optimize
            use_jax: Whether to use JAX for optimization
            track_history: Whether to track optimization history
        """
        self.parameters = parameters
        self.use_jax = use_jax and HAS_JAX
        self.track_history = track_history
        self.history: list[dict[str, Any]] = []
        self.objective: Callable | None = None
        self.constraints: list[Callable] = []
        self.callback: Callable | None = None
        logger.debug(
            "ParameterOptimizer created",
            num_params=len(parameters),
            use_jax=self.use_jax,
            track_history=track_history,
        )

    @property
    def n_parameters(self) -> int:
        """Number of parameters."""
        return len(self.parameters)

    def get_values(self) -> np.ndarray:
        """Get current parameter values."""
        return self.parameters.get_values()

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get parameter bounds."""
        return self.parameters.get_bounds()

    def set_objective(self, objective: Callable):
        """Set objective function to minimize.

        Args:
            objective: Function that takes parameter values and returns scalar
        """
        logger.debug(
            "Setting objective function",
            operation="set_objective",
        )
        self.objective = objective

    def evaluate(self, values: ArrayLike) -> float:
        """Evaluate objective at given values.

        Args:
            values: Parameter values

        Returns:
            Objective function value
        """
        if self.objective is None:
            logger.error(
                "No objective function set",
                exc_info=True,
            )
            raise ValueError("No objective function set")

        result = self.objective(values)

        # Convert to float if needed
        if isinstance(result, (np.ndarray, jnp.ndarray)):
            result = float(result)

        return result

    def compute_gradient(self, values: ArrayLike) -> np.ndarray:
        """Compute gradient of objective.

        Args:
            values: Parameter values

        Returns:
            Gradient vector
        """
        logger.debug(
            "Computing gradient",
            operation="compute_gradient",
            use_jax=self.use_jax,
        )
        if not self.use_jax or not HAS_JAX:
            # Numerical gradient
            eps = 1e-8
            values_array = _coerce_array(values)
            n = len(values_array)
            grad = np.zeros(n)

            for i in range(n):
                values_plus = values_array.copy()
                values_plus[i] += eps

                f_plus = self.evaluate(values_plus)
                f = self.evaluate(values_array)

                grad[i] = (f_plus - f) / eps

            return grad
        else:
            # JAX automatic differentiation
            grad_fn = jax.grad(self.objective)
            return np.array(grad_fn(jnp.array(values)))

    def add_constraint(self, constraint: Callable):
        """Add optimization constraint.

        Args:
            constraint: Function that returns >= 0 for valid values
        """
        logger.debug(
            "Adding optimization constraint",
            operation="add_constraint",
            num_constraints=len(self.constraints) + 1,
        )
        self.constraints.append(constraint)

    def validate_constraints(self, values: ArrayLike) -> bool:
        """Check if constraints are satisfied.

        Args:
            values: Parameter values

        Returns:
            True if all constraints satisfied
        """
        values_array = _coerce_array(values)
        for constraint in self.constraints:
            if constraint(values_array) < 0:
                logger.debug(
                    "Constraint validation failed",
                    operation="validate_constraints",
                )
                return False
        return True

    def set_callback(self, callback: Callable):
        """Set optimization callback.

        Args:
            callback: Function called after each iteration
        """
        logger.debug(
            "Setting optimization callback",
            operation="set_callback",
        )
        self.callback = callback

    def step(self, values: ArrayLike, iteration: int | None = None):
        """Perform one optimization step.

        Args:
            values: Current parameter values
            iteration: Current iteration number
        """
        # Update parameters
        coerced_values = _coerce_array(values)
        self.parameters.set_values(coerced_values)

        # Evaluate objective
        obj_value = self.evaluate(coerced_values)

        # Track history
        if self.track_history:
            self.history.append(
                {
                    "iteration": iteration or len(self.history),
                    "values": coerced_values.copy(),
                    "objective": obj_value,
                }
            )

        # Call callback
        if self.callback:
            self.callback(iteration or 0, coerced_values, obj_value)

    def get_history(self) -> list[dict[str, Any]]:
        """Get optimization history.

        Returns:
            List of history dictionaries
        """
        return self.history
