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

# Safe JAX import (enforces float64)
jax, jnp = safe_import_jax()
HAS_JAX = True


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
        if self.type == "bounds":
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
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
    """Single parameter with value, bounds, and metadata."""

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
        self.bounds = bounds
        self.units = units
        self.description = description
        self.constraints = list(constraints) if constraints else []
        self._value: float | None = None
        self._clamp_on_set = False
        self._was_clamped = False
        self._initialize(value)

    def _initialize(self, value: float | None) -> None:
        """Validate parameter after initialization."""
        if self.bounds is not None:
            lower, upper = self.bounds
            lower = float(lower)
            upper = float(upper)
            if lower > upper:
                raise ValueError(
                    f"Invalid bounds for parameter '{self.name}': {(lower, upper)}"
                )
            self.bounds = (lower, upper)

        # Add bounds as constraint if specified
        if self.bounds:
            self.constraints.insert(
                0,
                ParameterConstraint(
                    type="bounds", min_value=self.bounds[0], max_value=self.bounds[1]
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
            raise ValueError(
                f"Parameter '{self.name}' requires a numeric value"
            ) from exc

        if not np.isfinite(numeric_val):
            raise ValueError(f"Parameter '{self.name}' received non-finite value")

        clamped_during_init = False
        if self.bounds:
            lower, upper = self.bounds
            if self._clamp_on_set:
                if numeric_val < lower:
                    warnings.warn(
                        f"Parameter '{self.name}' initialized below bounds; clamped to {lower}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    numeric_val = lower
                    clamped_during_init = True
                elif numeric_val > upper:
                    warnings.warn(
                        f"Parameter '{self.name}' initialized above bounds; clamped to {upper}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    numeric_val = upper
                    clamped_during_init = True
            elif numeric_val < lower or numeric_val > upper:
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
        return {
            "name": self.name,
            "value": self.value,
            "bounds": self.bounds,
            "units": self.units,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Parameter:
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            value=data.get("value"),
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            units=data.get("units"),
            description=data.get("description"),
        )


class ParameterSet:
    """Collection of parameters for a model or transform."""

    def __init__(self):
        """Initialize empty parameter set."""
        self._parameters: dict[str, Parameter] = {}
        self._order: list[str] = []

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

        return param

    def get(self, name: str) -> Parameter | None:
        """Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter object or None if not found
        """
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
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")

        param = self._parameters[name]

        # Validate against constraints
        context = {
            p.name: p.value for p in self._parameters.values() if p.value is not None
        }
        if not param.validate(value, context):
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
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not found")

        min_val, max_val = bounds
        if min_val >= max_val:
            raise ValueError(
                f"Invalid bounds: min ({min_val}) must be < max ({max_val})"
            )

        param = self._parameters[name]
        param.bounds = bounds

        # Update or add bounds constraint
        # Remove existing bounds constraints
        param.constraints = [c for c in param.constraints if c.type != "bounds"]

        # Add new bounds constraint
        param.constraints.insert(
            0,
            ParameterConstraint(
                type="bounds",
                min_value=min_val,
                max_value=max_val,
            ),
        )

    def get_values(self) -> np.ndarray:
        """Get all parameter values as array.

        Returns:
            Array of parameter values in order
        """
        values = []
        for name in self._order:
            param = self._parameters[name]
            values.append(param.value if param.value is not None else 0.0)
        return np.array(values)

    def set_values(self, values: ArrayLike):
        """Set all parameter values from array.

        Args:
            values: Array of values in order

        Raises:
            ValueError: If wrong number of values
        """
        values = np.atleast_1d(values)
        if len(values) != len(self._order):
            raise ValueError(f"Expected {len(self._order)} values, got {len(values)}")

        for name, value in zip(self._order, values, strict=False):
            self.set_value(name, float(value))

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get bounds for all parameters.

        Returns:
            List of (min, max) tuples
        """
        bounds = []
        for name in self._order:
            param = self._parameters[name]
            if param.bounds:
                bounds.append(param.bounds)
            else:
                bounds.append((None, None))
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
        if isinstance(value, Parameter):
            # Replace entire parameter
            self._parameters[key] = value
            if key not in self._order:
                self._order.append(key)
        else:
            # Set value only
            if key not in self._parameters:
                raise KeyError(
                    f"Parameter '{key}' not found. Use add() to create new parameters."
                )
            self.set_value(key, float(value))

    def to_dict(self) -> dict[str, dict[str, Any]]:
        """Convert to dictionary representation."""
        return {name: self._parameters[name].to_dict() for name in self._order}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParameterSet:
        """Create from dictionary representation."""
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

        return param

    def link_model(self, model: Any, param_name: str):
        """Link a model to a shared parameter.

        Args:
            model: Model to link
            param_name: Name of shared parameter
        """
        if param_name not in self._shared:
            raise KeyError(f"Shared parameter '{param_name}' not found")

        if model not in self._links[param_name]:
            self._links[param_name].append(model)

    def link_parameter_set(self, param_set: ParameterSet, param_name: str):
        """Link a parameter set to a shared parameter.

        Args:
            param_set: ParameterSet to link
            param_name: Name of shared parameter
        """
        if param_name not in self._shared:
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
        if name not in self._shared:
            raise KeyError(f"Shared parameter '{name}' not found")

        param = self._shared[name]

        # Validate
        if not param.validate(value):
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
        self.objective = objective

    def evaluate(self, values: ArrayLike) -> float:
        """Evaluate objective at given values.

        Args:
            values: Parameter values

        Returns:
            Objective function value
        """
        if self.objective is None:
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
                return False
        return True

    def set_callback(self, callback: Callable):
        """Set optimization callback.

        Args:
            callback: Function called after each iteration
        """
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
