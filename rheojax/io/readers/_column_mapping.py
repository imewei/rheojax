"""Consolidated canonical column field registry for all I/O readers.

Merges patterns from Anton Paar COLUMN_MAPPINGS, TRIOS TRIOS_COLUMN_MAPPINGS,
and _utils.py regex patterns into a single canonical source.
"""
from __future__ import annotations

from dataclasses import dataclass

from rheojax.io.readers._utils import extract_unit_from_header
from rheojax.logging import get_logger

logger = get_logger(__name__)

__all__ = ["CanonicalField", "CANONICAL_FIELDS", "match_column", "match_columns"]


@dataclass
class CanonicalField:
    """Canonical field descriptor for a rheological measurement column."""

    canonical_name: str
    patterns: list[str]
    si_unit: str
    applicable_modes: list[str]
    is_x_candidate: bool = False
    is_y_candidate: bool = False
    priority: int = 100


CANONICAL_FIELDS: dict[str, CanonicalField] = {
    "time": CanonicalField(
        canonical_name="time",
        patterns=[
            r"^time$",
            r"^t$",
            r"^zeit$",
            r"^step\s*time$",
        ],
        si_unit="s",
        applicable_modes=["creep", "relaxation", "oscillation", "rotation"],
        is_x_candidate=True,
        priority=10,
    ),
    "angular_frequency": CanonicalField(
        canonical_name="angular_frequency",
        patterns=[
            r"^angular[\s_]?frequency$",
            r"^frequency$",
            r"^omega$",
            r"^ω$",
        ],
        si_unit="rad/s",
        applicable_modes=["oscillation"],
        is_x_candidate=True,
        priority=5,
    ),
    "shear_rate": CanonicalField(
        canonical_name="shear_rate",
        patterns=[
            r"^shear[\s_]?rate$",
            r"^γ̇$",
            r"^gamma[\s_]?dot$",
        ],
        si_unit="1/s",
        applicable_modes=["rotation"],
        is_x_candidate=True,
        priority=5,
    ),
    "storage_modulus": CanonicalField(
        canonical_name="storage_modulus",
        patterns=[
            r"^storage[\s_]?modulus$",
            r"^g'$",
            r"^g_prime$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    "loss_modulus": CanonicalField(
        canonical_name="loss_modulus",
        patterns=[
            r"^loss[\s_]?modulus$",
            r"^g''$",
            r'^g"$',
            r"^g_double_prime$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    "complex_modulus": CanonicalField(
        canonical_name="complex_modulus",
        patterns=[
            r"^complex[\s_]?modulus$",
            r"^g\*$",
            r"^\|g\*\|$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=10,
    ),
    "tensile_storage_modulus": CanonicalField(
        canonical_name="tensile_storage_modulus",
        patterns=[
            r"^e'$",
            r"^e_prime$",
            r"^e_stor$",
            r"^tensile[\s_]?storage[\s_]?modulus$",
            r"^young'?s?[\s_]?storage[\s_]?modulus$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    "tensile_loss_modulus": CanonicalField(
        canonical_name="tensile_loss_modulus",
        patterns=[
            r"^e''$",
            r'^e"$',
            r"^e_double_prime$",
            r"^e_loss$",
            r"^tensile[\s_]?loss[\s_]?modulus$",
            r"^young'?s?[\s_]?loss[\s_]?modulus$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=5,
    ),
    "compliance": CanonicalField(
        canonical_name="compliance",
        patterns=[
            r"^compliance$",
            r"^j\(?t\)?$",
        ],
        si_unit="1/Pa",
        applicable_modes=["creep"],
        is_y_candidate=True,
        priority=5,
    ),
    "relaxation_modulus": CanonicalField(
        canonical_name="relaxation_modulus",
        patterns=[
            r"^relaxation[\s_]?modulus$",
            r"^g\(?t\)?$",
        ],
        si_unit="Pa",
        applicable_modes=["relaxation"],
        is_y_candidate=True,
        priority=5,
    ),
    "viscosity": CanonicalField(
        canonical_name="viscosity",
        patterns=[
            r"^viscosity$",
            r"^η$",
            r"^eta$",
        ],
        si_unit="Pa.s",
        applicable_modes=["rotation"],
        is_y_candidate=True,
        priority=5,
    ),
    "complex_viscosity": CanonicalField(
        canonical_name="complex_viscosity",
        patterns=[
            r"^complex[\s_]?viscosity$",
            r"^η\*$",
            r"^eta\*$",
        ],
        si_unit="Pa.s",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=10,
    ),
    "shear_stress": CanonicalField(
        canonical_name="shear_stress",
        patterns=[
            r"^shear[\s_]?stress$",
            r"^stress$",
            r"^τ$",
            r"^tau$",
        ],
        si_unit="Pa",
        applicable_modes=["creep", "relaxation", "rotation"],
        is_y_candidate=True,
        priority=20,
    ),
    "shear_strain": CanonicalField(
        canonical_name="shear_strain",
        patterns=[
            r"^shear[\s_]?strain$",
            r"^strain$",
            r"^γ$",
            r"^gamma$",
        ],
        si_unit="dimensionless",
        applicable_modes=["creep", "relaxation"],
        is_y_candidate=True,
        priority=20,
    ),
    "phase_angle": CanonicalField(
        canonical_name="phase_angle",
        patterns=[
            r"^phase[\s_]?angle$",
            r"^δ$",
            r"^delta$",
        ],
        si_unit="deg",
        applicable_modes=["oscillation"],
        priority=100,
    ),
    "temperature": CanonicalField(
        canonical_name="temperature",
        patterns=[
            r"^temperature$",
            r"^temp$",
        ],
        si_unit="°C",
        applicable_modes=["creep", "relaxation", "oscillation", "rotation"],
        priority=100,
    ),
    "normal_force": CanonicalField(
        canonical_name="normal_force",
        patterns=[
            r"^normal[\s_]?force$",
        ],
        si_unit="N",
        applicable_modes=["creep", "relaxation", "oscillation", "rotation"],
        priority=100,
    ),
    "torque": CanonicalField(
        canonical_name="torque",
        patterns=[
            r"^torque$",
        ],
        si_unit="N.m",
        applicable_modes=["rotation"],
        priority=100,
    ),
    "strain_amplitude": CanonicalField(
        canonical_name="strain_amplitude",
        patterns=[
            r"^strain[\s_]?amplitude$",
        ],
        si_unit="dimensionless",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=100,
    ),
    "stress_amplitude": CanonicalField(
        canonical_name="stress_amplitude",
        patterns=[
            r"^stress[\s_]?amplitude$",
        ],
        si_unit="Pa",
        applicable_modes=["oscillation"],
        is_y_candidate=True,
        priority=100,
    ),
}

# Pre-compiled patterns sorted by priority (lower number = higher priority)
import re as _re  # deferred to avoid circular import at module level

_compiled_patterns: dict[str, list[_re.Pattern]] = {
    name: [_re.compile(p, _re.IGNORECASE) for p in cf.patterns]
    for name, cf in CANONICAL_FIELDS.items()
}

# Sorted canonical field names by priority for match_column iteration
_priority_order: list[str] = sorted(
    CANONICAL_FIELDS.keys(), key=lambda n: CANONICAL_FIELDS[n].priority
)

def match_column(
    header: str, instrument: str | None = None
) -> CanonicalField | None:
    """Match a column header string to a CanonicalField.

    Uses :func:`~rheojax.io.readers._utils.extract_unit_from_header` to strip
    parenthesized unit suffixes (e.g. ``"omega (rad/s)"`` → ``"omega"``),
    ensuring consistent unit extraction across the I/O subsystem.

    Parameters
    ----------
    header:
        Raw column header string (may include a parenthesized unit suffix).
    instrument:
        Optional instrument name for future instrument-specific filtering.
        Currently unused; reserved for Phase 2 extension.

    Returns
    -------
    CanonicalField or None
        The first matching canonical field (ordered by priority), or None if
        no field matches.
    """
    # Reuse the canonical unit extraction from _utils to strip "(unit)" suffixes.
    # This avoids a duplicate regex and ensures slash-containing names like
    # "1/s" are not incorrectly truncated.
    name_part, _ = extract_unit_from_header(header)

    for field_name in _priority_order:
        for pattern in _compiled_patterns[field_name]:
            if pattern.match(name_part):
                logger.debug(
                    "Column %r matched canonical field %r", header, field_name
                )
                return CANONICAL_FIELDS[field_name]

    logger.debug("Column %r had no canonical match", header)
    return None


def match_columns(
    headers: list[str], instrument: str | None = None
) -> dict[str, CanonicalField]:
    """Match a list of column headers to canonical fields.

    Parameters
    ----------
    headers:
        List of raw column header strings.
    instrument:
        Optional instrument name passed through to :func:`match_column`.

    Returns
    -------
    dict mapping header -> CanonicalField for every header that matched.
    """
    result: dict[str, CanonicalField] = {}
    for header in headers:
        cf = match_column(header, instrument=instrument)
        if cf is not None:
            result[header] = cf
    return result
