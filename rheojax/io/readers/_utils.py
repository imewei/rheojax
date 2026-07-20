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

from rheojax.logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from numpy.typing import NDArray

__all__ = [
    "VALID_TEST_MODES",
    "VALID_TRANSFORMS",
    "TRANSFORM_REQUIREMENTS",
    "extract_unit_from_header",
    "infer_y_unit_from_name",
    "detect_domain",
    "detect_test_mode_from_columns",
    "check_tensile_guard",
    "check_file_for_unsupported_data",
    "validate_transform",
    "construct_complex_modulus",
    "UNIFIED_UNIT_CONVERSIONS",
    "normalize_units",
    "normalize_temperature",
    "find_column_by_pattern",
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
        "startup",
        "flow_curve",
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
        for p in [
            r"[EG]['\"]",
            r"[EG]\*",
            r"E[-_]?stor",
            r"E[-_]?loss",
            r"omega",
            r"frequency",
            r"angular",
        ]
    ],
    "relaxation": [
        re.compile(p, re.IGNORECASE) for p in [r"[EG]\s*\(\s*t\s*\)", r"relaxation"]
    ],
    "creep": [
        re.compile(p, re.IGNORECASE)
        for p in [r"J\s*\(\s*t\s*\)", r"compliance", r"creep"]
    ],
    "rotation": [
        re.compile(p, re.IGNORECASE)
        for p in [
            r"shear\s*[-_]?\s*rate",
            r"viscosity",
            r"(?<![a-z])eta(?![a-z])",  # word-bounded: avoids "theta", "beta", "zeta"
            r"η",  # Greek eta (Unicode viscosity symbol)
            r"γ̇",
            r"gamma[-_]?dot",
        ]
    ],
}

# Pre-compiled pattern for G'/G'' or E'/E'' detection in domain detection
_MODULUS_PATTERN = re.compile(r"[EG]['\"]", re.IGNORECASE)

# Pre-compiled patterns for deformation mode detection (DMTA/DMA)
_TENSILE_MODULUS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"E['\"]",  # E' or E''
        r"E\*",  # E*
        r"E[-_]?stor",  # E_stor, Estor, E-stor
        r"E[-_]?loss",  # E_loss, Eloss, E-loss
        r"Young",  # Young's modulus
        r"tensile",  # Tensile modulus
        r"storage\s+modulus\s*[\(:]?\s*E['\"*]",  # "Storage Modulus E'"/"(E')"
        r"loss\s+modulus\s*[\(:]?\s*E['\"*]",  # "Loss Modulus E'"/"(E')"
    ]
]

_SHEAR_MODULUS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"G['\"]",  # G' or G''
        r"G\*",  # G*
        r"G[-_]?stor",  # G_stor
        r"G[-_]?loss",  # G_loss
        r"shear\s+modulus",  # Shear modulus
        r"storage\s+modulus.*G",  # "Storage Modulus G'"
        r"loss\s+modulus.*G",  # "Loss Modulus G'"
    ]
]

_BENDING_MODULUS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"E[-_]?bend",  # E_bending, E-bend
        r"\bbending\b",  # bending (keyword, word-bounded)
        r"modulus[-_\s]*bend",  # modulus_bending
        r"\bflexural\b",  # Flexural modulus (synonym, word-bounded)
    ]
]

_COMPRESSION_MODULUS_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bE[-_]?comp",  # E_compression, E-comp (not ActiveCompression)
        r"\bcompression\b",  # compression (keyword, word-bounded)
        r"modulus[-_\s]*comp",  # modulus_compression
        r"\bcompressive\b",  # Compressive modulus (word-bounded)
    ]
]

# Unit pattern for extraction
_UNIT_PATTERN = re.compile(r"^(.+?)\s*\(([^)]+)\)$")

# Domain detection patterns
# Note: "1/s" is deliberately excluded -- it is the SI unit for shear rate
# (see UNIFIED_UNIT_CONVERSIONS' rpm/rev/min/rev/s targets), not just frequency.
# Data with x_units="1/s" falls through to header-keyword checks and then
# defaults to "time" domain, matching RheoData's own default.
_FREQUENCY_UNITS = {"rad/s", "hz"}
_TIME_UNITS = {"s", "sec", "min", "ms"}
_FREQUENCY_KEYWORDS = {"omega", "frequency", "freq", "angular"}
_TIME_KEYWORDS = {"time", "t"}


def find_column_by_pattern(
    columns: list[str], columns_lower: list[str], patterns: list[str]
) -> str | None:
    """Find the column whose name matches one of *patterns*, preferring an
    exact match over a substring match and refusing to guess when multiple
    distinct columns are equally plausible candidates.

    An exact match (the whole column name, lowercased, equals a pattern) is
    unambiguous and always wins. A substring/token match (the pattern
    appears as a whole word inside a longer header, e.g. "Time (s)" for
    pattern "time") is used only when it uniquely identifies one column.

    Checking patterns in flat priority order used to mean a column like
    "Relaxation Time" (a derived/metadata value, not the swept axis) could
    win over "Angular Frequency" (the file's actual x-axis) just because
    "time" happened to be earlier in the pattern list than "frequency" --
    silently importing the wrong independent variable. If no column is an
    exact match and more than one column plausibly matches a different
    pattern, that ambiguity is real: return None so the caller raises
    "could not auto-detect" instead of guessing (a pattern must match as a
    whole token -- bounded by non-alphanumeric characters rather than
    ``\\b``, since ``\\b`` fails after symbolic suffixes like "g'"/"g''"
    whose last character is already non-word -- same approach as
    ColumnMapperDialog's auto-detect).

    Args:
        columns: Original-case column names.
        columns_lower: Same columns, lowercased (index-aligned with columns).
        patterns: Substrings to search for.

    Returns:
        The original-case column name of the unambiguous match, or None.
    """
    exact_matches: list[str] = []
    substring_matches: list[str] = []
    seen_exact: set[str] = set()
    seen_substring: set[str] = set()
    for pattern in patterns:
        regex = re.compile(r"(?<![a-z0-9])" + re.escape(pattern) + r"(?![a-z0-9])")
        for col, col_lower in zip(columns, columns_lower, strict=True):
            if col_lower == pattern:
                if col not in seen_exact:
                    exact_matches.append(col)
                    seen_exact.add(col)
            elif regex.search(col_lower) and col not in seen_substring:
                substring_matches.append(col)
                seen_substring.add(col)

    if len(exact_matches) == 1:
        return exact_matches[0]
    if exact_matches:
        return None
    if len(substring_matches) == 1:
        return substring_matches[0]
    return None


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
    logger.debug("Extracting unit from header", header=header)
    header = header.strip()
    match = _UNIT_PATTERN.match(header)
    if match:
        name, unit = match.group(1).strip(), match.group(2).strip()
        logger.debug("Unit extracted successfully", name=name, unit=unit)
        return name, unit
    logger.debug("No unit found in header", name=header)
    return header, None


def infer_y_unit_from_name(name: str) -> str | None:
    """Infer a default SI y-unit from a bare flow-curve column name.

    Flow-curve exports frequently label the y-column "Viscosity" or "Stress"
    with no bracketed unit (``extract_unit_from_header`` returns ``None`` for
    these). Stress and viscosity have the same order of magnitude at low
    shear rates, so leaving the unit unset lets a viscosity column silently
    be fit as stress (or vice versa) with no error -- only a bad fit.

    Args:
        name: Column header string (already unit-stripped, or bare).

    Returns:
        "Pa.s" for a viscosity-labeled column, "Pa" for a stress-labeled
        column, or None if the name doesn't match either.
    """
    normalized = name.strip().lower()
    if "viscosity" in normalized:
        return "Pa.s"
    if "stress" in normalized:
        return "Pa"
    return None


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
    logger.debug(
        "Detecting domain from column info",
        x_header=x_header,
        x_units=x_units,
        y_headers=y_headers,
    )

    # Check units first (most reliable)
    if x_units:
        x_units_lower = x_units.lower()
        for freq_unit in _FREQUENCY_UNITS:
            if freq_unit in x_units_lower:
                logger.debug(
                    "Domain detected from frequency unit",
                    domain="frequency",
                    matched_unit=freq_unit,
                )
                return "frequency"
        for time_unit in _TIME_UNITS:
            # Check for exact match or unit at boundary to avoid false positives
            # e.g., "1/s" should not match time domain
            if x_units_lower == time_unit or x_units_lower.endswith(f" {time_unit}"):
                logger.debug(
                    "Domain detected from time unit",
                    domain="time",
                    matched_unit=time_unit,
                )
                return "time"

    # Check column name patterns
    x_header_lower = x_header.lower()
    for freq_keyword in _FREQUENCY_KEYWORDS:
        if freq_keyword in x_header_lower:
            logger.debug(
                "Domain detected from header keyword",
                domain="frequency",
                matched_keyword=freq_keyword,
            )
            return "frequency"
    for time_keyword in _TIME_KEYWORDS:
        if time_keyword in x_header_lower:
            logger.debug(
                "Domain detected from header keyword",
                domain="time",
                matched_keyword=time_keyword,
            )
            return "time"

    # Check y_headers for oscillation indicators (G', G'')
    if y_headers:
        for yh in y_headers:
            if _MODULUS_PATTERN.search(yh):
                logger.debug(
                    "Domain detected from y_header modulus pattern",
                    domain="frequency",
                    matched_header=yh,
                )
                return "frequency"

    # Default to time domain
    logger.debug("Domain defaulting to time (no pattern matched)", domain="time")
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
    logger.debug(
        "Detecting test mode from columns",
        x_header=x_header,
        y_headers=y_headers,
        x_units=x_units,
        y_units=y_units,
    )

    # Combine all headers for pattern matching
    all_headers = [x_header] + y_headers
    all_text = " ".join(all_headers).lower()

    # Check each test mode's patterns (uses pre-compiled regex)
    for mode, patterns in _TEST_MODE_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(all_text):
                logger.debug(
                    "Test mode detected from pattern",
                    test_mode=mode,
                    matched_pattern=pattern.pattern,
                )
                return mode

    # Additional unit-based detection
    if x_units:
        x_units_lower = x_units.lower()
        if "rad/s" in x_units_lower or "hz" in x_units_lower:
            logger.debug(
                "Test mode detected from x_units",
                test_mode="oscillation",
                x_units=x_units,
            )
            return "oscillation"
        if "1/s" in x_units_lower and y_units:
            y_units_lower = y_units.lower()
            if "pa" in y_units_lower and ("s" in y_units_lower or "*" in y_units_lower):
                logger.debug(
                    "Test mode detected from unit combination",
                    test_mode="rotation",
                    x_units=x_units,
                    y_units=y_units,
                )
                return "rotation"  # viscosity units like Pa*s or Pa·s

    logger.debug("Test mode could not be determined from columns")
    return None


# =============================================================================
# Tensile Guard (DMTA/DMA Rejection)
# =============================================================================


def check_tensile_guard(
    headers: list[str] | set[str] | pd.Index,
    units: str | None = None,
    source: str = "columns",
) -> None:
    """Scan headers and units for tensile/E* and other unsupported deformation modes (bending, compression).

    Raises UnsupportedDataError if any matches are found.
    """
    from rheojax.io._exceptions import UnsupportedDataError

    headers_list = [str(h) for h in headers]
    all_text = " ".join(headers_list).lower()

    # Check column names/text for tensile/E*
    for p in _TENSILE_MODULUS_PATTERNS:
        if p.search(all_text):
            raise UnsupportedDataError(
                f"Tensile/E* data detected: matching pattern '{p.pattern}' in {source}. "
                "RheoJAX only supports shear deformation mode."
            )

    # Check for other unsupported modes: bending
    for p in _BENDING_MODULUS_PATTERNS:
        if p.search(all_text):
            raise UnsupportedDataError(
                f"Bending data detected: matching pattern '{p.pattern}' in {source}. "
                "RheoJAX only supports shear deformation mode."
            )

    # Check for other unsupported modes: compression
    for p in _COMPRESSION_MODULUS_PATTERNS:
        if p.search(all_text):
            raise UnsupportedDataError(
                f"Compression data detected: matching pattern '{p.pattern}' in {source}. "
                "RheoJAX only supports shear deformation mode."
            )

    # Check units string
    if units:
        units_str = str(units)
        # E', E'', E* in units
        if re.search(r"\bE['\"\*]", units_str):
            raise UnsupportedDataError(
                f"Tensile/E* data detected in units: '{units_str}'. "
                "RheoJAX only supports shear deformation mode."
            )


def check_file_for_unsupported_data(filepath: Path) -> None:
    """Read the start of the file or sheet headers and raise UnsupportedDataError if tensile/E* data is detected."""
    from rheojax.io._exceptions import UnsupportedDataError

    suffix = filepath.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            import pandas as pd

            xl = pd.ExcelFile(filepath)
            for sheet in xl.sheet_names:
                df = xl.parse(sheet, nrows=5)
                # Check column headers
                headers = list(df.columns)
                # Also check values in first row if they might contain headers/metadata
                for row_idx in range(min(len(df), 3)):
                    headers.extend(
                        [str(val) for val in df.iloc[row_idx] if pd.notna(val)]
                    )
                check_tensile_guard(headers, source=f"Excel sheet '{sheet}'")
        except Exception as e:
            if type(e).__name__ == "UnsupportedDataError" or isinstance(
                e, UnsupportedDataError
            ):
                raise
            logger.warning(
                "Tensile-data pre-scan of Excel file could not complete; "
                "the safety guard was skipped for this file",
                filepath=str(filepath),
                error=str(e),
                exc_info=True,
            )
    else:
        # Text files (CSV, TSV, TXT, JSON)
        try:
            # Check file content (only first 200 lines to avoid loading large files completely)
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                lines = [f.readline() for _ in range(200)]
                content = "".join(lines)

            # Check for patterns in the raw content
            for p in _TENSILE_MODULUS_PATTERNS:
                if p.search(content):
                    raise UnsupportedDataError(
                        f"Unsupported tensile data detected: matching pattern '{p.pattern}' in file content. "
                        "RheoJAX only supports shear deformation mode."
                    )
            for p in _BENDING_MODULUS_PATTERNS:
                if p.search(content):
                    raise UnsupportedDataError(
                        f"Unsupported bending data detected: matching pattern '{p.pattern}' in file content. "
                        "RheoJAX only supports shear deformation mode."
                    )
            for p in _COMPRESSION_MODULUS_PATTERNS:
                if p.search(content):
                    raise UnsupportedDataError(
                        f"Unsupported compression data detected: matching pattern '{p.pattern}' in file content. "
                        "RheoJAX only supports shear deformation mode."
                    )
            # Check for units as well
            if re.search(r"\bE['\"\*]", content):
                raise UnsupportedDataError(
                    "Unsupported tensile data detected in file content. "
                    "RheoJAX only supports shear deformation mode."
                )
        except Exception as e:
            if type(e).__name__ == "UnsupportedDataError" or isinstance(
                e, UnsupportedDataError
            ):
                raise
            logger.warning(
                "Tensile-data pre-scan of text file could not complete; "
                "the safety guard was skipped for this file",
                filepath=str(filepath),
                error=str(e),
                exc_info=True,
            )


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
    logger.debug(
        "Validating transform requirements",
        intended_transform=intended_transform,
        domain=domain,
        test_mode=test_mode,
        metadata_keys=list(metadata.keys()),
    )

    warnings_list: list[str] = []

    # Validate transform type
    transform_lower = intended_transform.lower()
    if transform_lower not in VALID_TRANSFORMS:
        warnings_list.append(
            f"Unknown intended_transform '{intended_transform}'. "
            f"Valid options: {sorted(VALID_TRANSFORMS)}"
        )
        logger.debug(
            "Transform validation failed: unknown transform",
            intended_transform=intended_transform,
            valid_transforms=sorted(VALID_TRANSFORMS),
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
        logger.debug(
            "Transform validation: missing required fields",
            transform=transform_lower,
            missing_fields=missing_fields,
        )

    # Check domain compatibility
    required_domain = requirements["domain"]
    if required_domain and domain != required_domain:
        warnings_list.append(
            f"intended_transform '{transform_lower}' expects domain='{required_domain}', "
            f"got '{domain}'"
        )
        logger.debug(
            "Transform validation: domain mismatch",
            transform=transform_lower,
            expected_domain=required_domain,
            actual_domain=domain,
        )

    # Check test_mode/transform consistency
    if test_mode is not None:
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
            logger.debug(
                "Transform validation: test_mode conflict",
                transform=transform_lower,
                expected_mode=expected_mode,
                actual_mode=test_mode,
            )

    if not warnings_list:
        logger.debug(
            "Transform validation passed",
            transform=transform_lower,
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
    logger.debug(
        "Constructing complex modulus",
        g_prime_shape=getattr(g_prime, "shape", None),
        g_double_prime_shape=getattr(g_double_prime, "shape", None),
    )

    g_prime = np.asarray(g_prime, dtype=np.float64)
    g_double_prime = np.asarray(g_double_prime, dtype=np.float64)

    if g_prime.shape != g_double_prime.shape:
        logger.error(
            "Shape mismatch when constructing complex modulus",
            g_prime_shape=g_prime.shape,
            g_double_prime_shape=g_double_prime.shape,
            exc_info=True,
        )
        raise ValueError(
            f"G' and G'' arrays must have the same shape. "
            f"Got G'.shape={g_prime.shape}, G''.shape={g_double_prime.shape}"
        )

    logger.debug(
        "Complex modulus constructed successfully",
        result_shape=g_prime.shape,
        result_dtype="complex128",
    )
    return g_prime + 1j * g_double_prime


# =============================================================================
# Unified Unit Conversions
# =============================================================================

# Merge of Anton Paar UNIT_CONVERSIONS + TRIOS TRIOS_UNIT_CONVERSIONS + extensions.
# Each entry maps source_unit (lowercase) -> (target_si_unit, factor_or_None).
# factor=None signals an additive conversion (temperature) handled in normalize_units().
UNIFIED_UNIT_CONVERSIONS: dict[str, tuple[str, float | None]] = {
    # Frequency
    "hz": ("rad/s", 2 * np.pi),
    "1/hz": ("rad/s", 2 * np.pi),
    # Time
    "ms": ("s", 0.001),
    "min": ("s", 60.0),
    "mins": ("s", 60.0),
    "minutes": ("s", 60.0),
    # Pressure / Modulus
    "kpa": ("Pa", 1000.0),
    "mpa": ("Pa", 1e6),  # megapascal
    "gpa": ("Pa", 1e9),  # gigapascal (NEW)
    # Viscosity — note mPa here means millipascal (0.001 Pa)
    "mpa·s": ("Pa.s", 0.001),
    "mpa.s": ("Pa.s", 0.001),
    # Rotation
    "rpm": ("1/s", 1.0 / 60.0),
    "rev/min": ("1/s", 1.0 / 60.0),
    "rev/s": ("1/s", 1.0),
    # Dimensionless
    "%": ("dimensionless", 0.01),
    # Temperature (additive — factor is None, handled specially)
    "°c": ("K", None),
    "°f": ("K", None),
    "c": ("K", None),
    "f": ("K", None),
}

# SI prefix case matters for Pa: "M" (mega, 1e6) vs "m" (milli, 1e-3) collide
# onto the same "mpa" key once lowercased. Checked before the case-insensitive
# UNIFIED_UNIT_CONVERSIONS lookup so a literal "mPa" header (millipascal,
# plausible for soft/low-modulus samples) isn't silently misread as megapascal
# (mirrors anton_paar.py's _CASE_SENSITIVE_UNIT_CONVERSIONS).
_CASE_SENSITIVE_UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    "MPa": ("Pa", 1e6),
    "mPa": ("Pa", 0.001),
}


def normalize_temperature(value: float, unit: str = "C") -> float:
    """Convert a temperature value to Kelvin.

    Args:
        value: Temperature value.
        unit: Source unit — "C", "°C", "F", "°F", or "K".

    Returns:
        Temperature in Kelvin.

    Raises:
        ValueError: If unit is not recognized.
    """
    unit_lower = unit.strip().lower()
    if unit_lower in ("c", "°c", "celsius"):
        return value + 273.15
    if unit_lower in ("f", "°f", "fahrenheit"):
        return (value - 32) * 5 / 9 + 273.15
    if unit_lower in ("k", "kelvin"):
        return float(value)
    raise ValueError(
        f"Unrecognized temperature unit '{unit}'. Expected 'C', '°C', 'F', '°F', or 'K'."
    )


def normalize_units(
    values: NDArray[np.floating], source_unit: str
) -> tuple[NDArray[np.floating], str]:
    """Convert values to SI units.

    Handles both multiplicative conversions (kPa -> Pa) and additive
    conversions (°C -> K).

    Args:
        values: Array of values to convert.
        source_unit: Source unit string (case-insensitive lookup).

    Returns:
        Tuple of (converted_values, target_unit_string).
        If no conversion found, returns (values, source_unit) unchanged.
    """
    stripped = source_unit.strip()
    if stripped in _CASE_SENSITIVE_UNIT_CONVERSIONS:
        target_unit, factor = _CASE_SENSITIVE_UNIT_CONVERSIONS[stripped]
        values = np.asarray(values, dtype=np.float64)
        return values * factor, target_unit

    key = source_unit.lower()
    if key not in UNIFIED_UNIT_CONVERSIONS:
        return values, source_unit

    target_unit, unit_factor = UNIFIED_UNIT_CONVERSIONS[key]
    values = np.asarray(values, dtype=np.float64)

    if unit_factor is not None:
        return values * unit_factor, target_unit

    # Additive temperature conversion
    if key in ("°c", "c"):
        return values + 273.15, target_unit
    if key in ("°f", "f"):
        return (values - 32) * 5 / 9 + 273.15, target_unit

    return values, source_unit
