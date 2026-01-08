"""Internal utilities for rheological data readers.

This module provides shared functionality for csv_reader and excel_reader:
- Unit extraction from column headers
- Domain detection (time/frequency)
- Test mode detection from column patterns
- Transform validation
- Complex modulus construction
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = [
    "VALID_TEST_MODES",
    "VALID_TRANSFORMS",
    "TRANSFORM_REQUIREMENTS",
    "extract_unit_from_header",
    "detect_domain",
    "detect_test_mode_from_columns",
    "validate_transform",
    "construct_complex_modulus",
]

# =============================================================================
# Constants
# =============================================================================

VALID_TEST_MODES: frozenset[str] = frozenset(
    {
        "relaxation",
        "creep",
        "oscillation",
        "rotation",
    }
)

VALID_TRANSFORMS: frozenset[str] = frozenset(
    {
        "mastercurve",
        "srfs",
        "owchirp",
        "spp",
        "fft",
        "mutation",
        "derivative",
    }
)

TRANSFORM_REQUIREMENTS: dict[str, dict] = {
    "mastercurve": {"required": ["temperature"], "domain": "frequency"},
    "srfs": {"required": ["reference_gamma_dot"], "domain": None},
    "owchirp": {"required": [], "domain": "time"},
    "spp": {"required": ["omega", "gamma_0"], "domain": "time"},
    "fft": {"required": [], "domain": "time"},
    "mutation": {"required": [], "domain": "time"},
    "derivative": {"required": [], "domain": None},
}

# Pre-compiled patterns for test mode detection (performance optimization)
_TEST_MODE_PATTERNS: dict[str, list[re.Pattern]] = {
    "oscillation": [
        re.compile(p, re.IGNORECASE)
        for p in [r"G['\"]", r"G\*", r"omega", r"frequency", r"angular"]
    ],
    "relaxation": [
        re.compile(p, re.IGNORECASE) for p in [r"G\s*\(\s*t\s*\)", r"relaxation"]
    ],
    "creep": [
        re.compile(p, re.IGNORECASE)
        for p in [r"J\s*\(\s*t\s*\)", r"compliance", r"creep"]
    ],
    "rotation": [
        re.compile(p, re.IGNORECASE)
        for p in [r"shear\s*[-_]?\s*rate", r"viscosity", r"eta", r"γ̇", r"gamma[-_]?dot"]
    ],
}

# Pre-compiled pattern for G'/G'' detection in domain detection
_MODULUS_PATTERN = re.compile(r"G['\"]", re.IGNORECASE)

# Unit pattern for extraction
_UNIT_PATTERN = re.compile(r"^(.+?)\s*\(([^)]+)\)$")

# Domain detection patterns
_FREQUENCY_UNITS = {"rad/s", "hz", "1/s"}
_TIME_UNITS = {"s", "sec", "min", "ms"}
_FREQUENCY_KEYWORDS = {"omega", "frequency", "freq", "angular"}
_TIME_KEYWORDS = {"time", "t"}


# =============================================================================
# Unit Extraction
# =============================================================================


def extract_unit_from_header(header: str) -> tuple[str, str | None]:
    """Extract name and unit from 'name (unit)' format.

    Args:
        header: Column header string

    Returns:
        Tuple of (name, unit) where unit may be None if not found

    Examples:
        >>> extract_unit_from_header("omega (rad/s)")
        ('omega', 'rad/s')
        >>> extract_unit_from_header("G' (Pa)")
        ("G'", 'Pa')
        >>> extract_unit_from_header("viscosity")
        ('viscosity', None)
        >>> extract_unit_from_header("  time (s)  ")
        ('time', 's')
    """
    header = header.strip()
    match = _UNIT_PATTERN.match(header)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return header, None


# =============================================================================
# Domain Detection
# =============================================================================


def detect_domain(
    x_header: str,
    x_units: str | None,
    y_headers: list[str] | None = None,
) -> str:
    """Detect data domain from column information.

    Detection priority:
        1. x_units containing 'rad/s' or 'Hz' -> 'frequency'
        2. x_units containing 's' or 'min' -> 'time'
        3. x_header containing 'omega'/'frequency' -> 'frequency'
        4. x_header containing 'time' -> 'time'
        5. Default -> 'time'

    Args:
        x_header: X column header
        x_units: X units (explicit or extracted)
        y_headers: Y column headers for additional context (optional)

    Returns:
        'time' or 'frequency'
    """
    # Check units first (most reliable)
    if x_units:
        x_units_lower = x_units.lower()
        for freq_unit in _FREQUENCY_UNITS:
            if freq_unit in x_units_lower:
                return "frequency"
        for time_unit in _TIME_UNITS:
            # Check for exact match or unit at boundary to avoid false positives
            # e.g., "1/s" should not match time domain
            if x_units_lower == time_unit or x_units_lower.endswith(f" {time_unit}"):
                return "time"
            # Handle standalone 's' or 'min'
            if time_unit in {"s", "sec", "min", "ms"} and x_units_lower in {time_unit}:
                return "time"

    # Check column name patterns
    x_header_lower = x_header.lower()
    for freq_keyword in _FREQUENCY_KEYWORDS:
        if freq_keyword in x_header_lower:
            return "frequency"
    for time_keyword in _TIME_KEYWORDS:
        if time_keyword in x_header_lower:
            return "time"

    # Check y_headers for oscillation indicators (G', G'')
    if y_headers:
        for yh in y_headers:
            if _MODULUS_PATTERN.search(yh):
                return "frequency"

    # Default to time domain
    return "time"


# =============================================================================
# Test Mode Detection
# =============================================================================


def detect_test_mode_from_columns(
    x_header: str,
    y_headers: list[str],
    x_units: str | None = None,
    y_units: str | None = None,
) -> str | None:
    """Detect test mode from column names and units.

    Detection patterns:
        - G', G'', G*, omega, frequency -> 'oscillation'
        - G(t), relaxation -> 'relaxation'
        - J(t), compliance, creep -> 'creep'
        - shear rate, viscosity, eta -> 'rotation'

    Args:
        x_header: X column header
        y_headers: Y column header(s)
        x_units: X units (optional)
        y_units: Y units (optional)

    Returns:
        Test mode string or None if cannot determine
    """
    # Combine all headers for pattern matching
    all_headers = [x_header] + y_headers
    all_text = " ".join(all_headers).lower()

    # Check each test mode's patterns (uses pre-compiled regex)
    for mode, patterns in _TEST_MODE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(all_text):
                return mode

    # Additional unit-based detection
    if x_units:
        x_units_lower = x_units.lower()
        if "rad/s" in x_units_lower or "hz" in x_units_lower:
            return "oscillation"
        if "1/s" in x_units_lower and y_units:
            y_units_lower = y_units.lower()
            if "pa" in y_units_lower and ("s" in y_units_lower or "*" in y_units_lower):
                return "rotation"  # viscosity units like Pa*s or Pa·s

    return None


# =============================================================================
# Transform Validation
# =============================================================================


def validate_transform(
    intended_transform: str,
    domain: str,
    metadata: dict,
    test_mode: str | None = None,
) -> list[str]:
    """Validate transform requirements and return warning messages.

    Checks:
        1. Transform type is valid
        2. Required metadata fields are present
        3. Domain is compatible with transform
        4. test_mode is consistent with transform (if provided)

    Args:
        intended_transform: The intended transform type
        domain: Detected/explicit domain
        metadata: Current metadata dict
        test_mode: Explicit test mode if provided

    Returns:
        List of warning messages (empty if all valid)
    """
    warnings_list: list[str] = []

    # Validate transform type
    transform_lower = intended_transform.lower()
    if transform_lower not in VALID_TRANSFORMS:
        warnings_list.append(
            f"Unknown intended_transform '{intended_transform}'. "
            f"Valid options: {sorted(VALID_TRANSFORMS)}"
        )
        return warnings_list

    requirements = TRANSFORM_REQUIREMENTS[transform_lower]

    # Check required metadata fields
    required_fields = requirements["required"]
    missing_fields = [f for f in required_fields if f not in metadata]
    if missing_fields:
        warnings_list.append(
            f"intended_transform '{transform_lower}' requires {missing_fields} in metadata"
        )

    # Check domain compatibility
    required_domain = requirements["domain"]
    if required_domain and domain != required_domain:
        warnings_list.append(
            f"intended_transform '{transform_lower}' expects domain='{required_domain}', "
            f"got '{domain}'"
        )

    # Check test_mode/transform consistency
    if test_mode:
        # Define expected test modes for transforms
        transform_test_modes = {
            "mastercurve": "oscillation",
            "srfs": "rotation",
            "owchirp": "oscillation",
            "spp": "oscillation",
            "fft": None,  # Can be any
            "mutation": None,  # Can be any
            "derivative": None,  # Can be any
        }
        expected_mode = transform_test_modes.get(transform_lower)
        if expected_mode and test_mode != expected_mode:
            warnings_list.append(
                f"test_mode '{test_mode}' may conflict with intended_transform "
                f"'{transform_lower}' (typically used with '{expected_mode}')"
            )

    return warnings_list


# =============================================================================
# Complex Modulus Construction
# =============================================================================


def construct_complex_modulus(
    g_prime: NDArray[np.floating],
    g_double_prime: NDArray[np.floating],
) -> NDArray[np.complexfloating]:
    """Construct complex modulus G* = G' + i*G''.

    Args:
        g_prime: Storage modulus array (real) - G'
        g_double_prime: Loss modulus array (real) - G''

    Returns:
        Complex array G* = G' + i*G''

    Raises:
        ValueError: If arrays have different lengths
    """
    g_prime = np.asarray(g_prime, dtype=np.float64)
    g_double_prime = np.asarray(g_double_prime, dtype=np.float64)

    if g_prime.shape != g_double_prime.shape:
        raise ValueError(
            f"G' and G'' arrays must have the same shape. "
            f"Got G'.shape={g_prime.shape}, G''.shape={g_double_prime.shape}"
        )

    return g_prime + 1j * g_double_prime
