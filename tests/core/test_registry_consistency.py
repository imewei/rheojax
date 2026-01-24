"""Registry consistency tests using static analysis.

This module uses introspection to verify that models declaring protocols have
corresponding implementation evidence, without actually running the models.

Test Design:
    - Uses inspect.getsource() to analyze _predict method implementations
    - Searches for test_mode dispatch patterns (if/elif chains, match statements)
    - Checks for dedicated predict_* methods (predict_flow_curve, etc.)
    - Warns (not fails) on missing evidence to handle edge cases gracefully

Markers:
    - @pytest.mark.smoke: Fast introspection-only tests

Usage:
    pytest tests/core/test_registry_consistency.py -v
"""

import inspect
import re
import warnings

import pytest

# Force registration of all models
import rheojax.models  # noqa: F401
from rheojax.core.inventory import Protocol
from rheojax.core.registry import ModelRegistry


def get_predict_source(model_class: type) -> str | None:
    """Get the source code of a model's _predict method.

    Args:
        model_class: The model class to inspect

    Returns:
        Source code string or None if not available
    """
    # Try to get _predict method
    method = getattr(model_class, "_predict", None)
    if method is None:
        return None

    try:
        return inspect.getsource(method)
    except (OSError, TypeError):
        # Source not available (e.g., built-in, C extension)
        return None


def find_test_mode_patterns(source: str) -> set[str]:
    """Find test_mode values referenced in source code.

    Searches for patterns like:
        - test_mode == "relaxation"
        - test_mode == 'creep'
        - test_mode in ["relaxation", "creep"]
        - test_mode in ("relaxation", "creep")
        - match test_mode: case "relaxation":

    Args:
        source: Source code string

    Returns:
        Set of test_mode values found
    """
    modes = set()

    # Pattern: test_mode == "value" or test_mode == 'value'
    pattern_eq = r'test_mode\s*==\s*["\'](\w+)["\']'
    modes.update(re.findall(pattern_eq, source))

    # Pattern: test_mode in ["val1", "val2"] or test_mode in ("val1", "val2")
    pattern_in = r"test_mode\s+in\s*[\[\(]([^\]\)]+)[\]\)]"
    for match in re.findall(pattern_in, source):
        # Extract quoted strings from the list
        modes.update(re.findall(r'["\'](\w+)["\']', match))

    # Pattern: case "value": (for match statements)
    pattern_case = r'case\s+["\'](\w+)["\']:'
    modes.update(re.findall(pattern_case, source))

    # Pattern: elif mode == "value" (common shorthand)
    pattern_mode = r'mode\s*==\s*["\'](\w+)["\']'
    modes.update(re.findall(pattern_mode, source))

    # Pattern: if "value" in test_mode (reverse check)
    pattern_reverse = r'["\'](\w+)["\']\s+in\s+test_mode'
    modes.update(re.findall(pattern_reverse, source))

    return modes


def has_predict_method(model_class: type, protocol: Protocol) -> bool:
    """Check if model has a dedicated predict_* method for the protocol.

    Args:
        model_class: The model class to check
        protocol: The protocol to look for

    Returns:
        True if a predict_<protocol_value> method exists
    """
    method_name = f"predict_{protocol.value}"
    return hasattr(model_class, method_name) and callable(
        getattr(model_class, method_name)
    )


def get_implementation_evidence(
    model_class: type, protocol: Protocol
) -> tuple[bool, str]:
    """Check for evidence that a protocol is implemented.

    Args:
        model_class: The model class to analyze
        protocol: The protocol to check

    Returns:
        Tuple of (has_evidence, evidence_type)
        evidence_type is one of: "dedicated_method", "test_mode_dispatch",
                                 "inherited", "none"
    """
    # Check 1: Dedicated predict_* method
    if has_predict_method(model_class, protocol):
        return True, "dedicated_method"

    # Check 2: test_mode dispatch in _predict source
    source = get_predict_source(model_class)
    if source:
        modes_found = find_test_mode_patterns(source)
        if protocol.value in modes_found:
            return True, "test_mode_dispatch"

    # Check 3: Inherited implementation (check MRO)
    for base_class in model_class.__mro__[1:]:
        if base_class.__name__ in ("object", "BaseModel", "BayesianMixin"):
            continue

        # Check if base class has dedicated method
        if has_predict_method(base_class, protocol):
            return True, "inherited"

        # Check if base class has test_mode dispatch
        base_source = get_predict_source(base_class)
        if base_source:
            modes_found = find_test_mode_patterns(base_source)
            if protocol.value in modes_found:
                return True, "inherited"

    return False, "none"


@pytest.mark.smoke
class TestRegistryConsistency:
    """Verify registry declarations match implementation evidence."""

    def test_all_models_have_protocols_declared(self):
        """Every registered model should declare at least one protocol."""
        models_without_protocols = []

        for model_name in ModelRegistry.list_models():
            info = ModelRegistry.get_info(model_name)
            if not info or not info.protocols:
                models_without_protocols.append(model_name)

        # This is a hard requirement - all models must declare protocols
        assert not models_without_protocols, (
            f"Models without declared protocols: {models_without_protocols}\n"
            "All models must declare supported protocols in @ModelRegistry.register()"
        )

    @pytest.mark.parametrize("model_name", ModelRegistry.list_models())
    def test_protocol_implementation_evidence(self, model_name: str):
        """Check that declared protocols have implementation evidence.

        This test uses static analysis to verify models have code paths
        for their declared protocols. It WARNS (not fails) on missing
        evidence to handle:
            - Models with generic _predict that handles all modes
            - Models with complex dispatch logic
            - Edge cases we haven't anticipated
        """
        info = ModelRegistry.get_info(model_name)
        if not info or not info.protocols:
            pytest.skip(f"No protocols declared for {model_name}")

        model_class = info.plugin_class
        missing_evidence = []

        for protocol in info.protocols:
            has_evidence, evidence_type = get_implementation_evidence(
                model_class, protocol
            )
            if not has_evidence:
                missing_evidence.append(protocol.value)

        # Warn instead of fail - some models may use generic dispatch
        if missing_evidence:
            warnings.warn(
                f"{model_name}: No explicit implementation evidence found for "
                f"protocols: {missing_evidence}. This may be fine if the model "
                f"uses generic dispatch logic.",
                UserWarning,
                stacklevel=1,
            )

    def test_registered_models_count(self):
        """Verify expected number of models are registered."""
        models = ModelRegistry.list_models()

        # Based on models/__init__.py, expect 27+ models
        assert len(models) >= 27, (
            f"Expected at least 27 registered models, found {len(models)}.\n"
            f"Registered: {sorted(models)}"
        )

    def test_protocol_coverage(self):
        """Check that all Protocol enum values have at least one model."""
        protocol_models: dict[Protocol, list[str]] = {p: [] for p in Protocol}

        for model_name in ModelRegistry.list_models():
            info = ModelRegistry.get_info(model_name)
            if info and info.protocols:
                for protocol in info.protocols:
                    protocol_models[protocol].append(model_name)

        # All protocols should have at least one model
        empty_protocols = [
            p.value for p, models in protocol_models.items() if not models
        ]

        # Warn rather than fail - some protocols may be legitimately unused
        if empty_protocols:
            warnings.warn(
                f"Protocols with no registered models: {empty_protocols}",
                UserWarning,
                stacklevel=1,
            )


@pytest.mark.smoke
class TestIntrospectionHelpers:
    """Test the introspection helper functions themselves."""

    def test_find_test_mode_patterns_equality(self):
        """Test detection of == patterns."""
        source = """
        if test_mode == "relaxation":
            return self._predict_relaxation(X)
        elif test_mode == 'creep':
            return self._predict_creep(X)
        """
        modes = find_test_mode_patterns(source)
        assert "relaxation" in modes
        assert "creep" in modes

    def test_find_test_mode_patterns_in_list(self):
        """Test detection of 'in [...]' patterns."""
        source = """
        if test_mode in ["relaxation", "creep"]:
            return self._predict_transient(X)
        elif test_mode in ("oscillation", "laos"):
            return self._predict_oscillatory(X)
        """
        modes = find_test_mode_patterns(source)
        assert "relaxation" in modes
        assert "creep" in modes
        assert "oscillation" in modes
        assert "laos" in modes

    def test_find_test_mode_patterns_mode_shorthand(self):
        """Test detection of 'mode ==' shorthand patterns."""
        source = """
        mode = test_mode or "relaxation"
        if mode == "flow_curve":
            return self._flow_curve(X)
        """
        modes = find_test_mode_patterns(source)
        assert "flow_curve" in modes

    def test_get_predict_source_real_model(self):
        """Test getting source from a real model."""
        from rheojax.models import Maxwell

        source = get_predict_source(Maxwell)
        # Maxwell should have _predict method with source
        assert source is not None
        assert "def _predict" in source or "test_mode" in source

    def test_has_predict_method(self):
        """Test detection of dedicated predict_* methods."""
        from rheojax.models.ikh import MLIKH

        # MLIKH has predict_flow_curve, predict_startup, etc.
        assert has_predict_method(MLIKH, Protocol.FLOW_CURVE)
        assert has_predict_method(MLIKH, Protocol.STARTUP)
        assert has_predict_method(MLIKH, Protocol.RELAXATION)
