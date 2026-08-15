"""Custom exceptions and warnings for the RheoJAX package.

This module defines the exception and warning hierarchy used across
the I/O subsystem and the fitting/transform workflows.
"""

# ---------------------------------------------------------------------------
# I/O exceptions
# ---------------------------------------------------------------------------


class RheoJaxFormatError(ValueError):
    """Raised when no reader can parse a file.

    Subclasses ValueError so downstream ``except ValueError`` still catches it.
    """


class RheoJaxValidationWarning(UserWarning):
    """Emitted for data quality issues detected during loading or validation.

    Subclasses UserWarning so standard warning filters apply.
    """


# ---------------------------------------------------------------------------
# Fitting exceptions & warnings
# ---------------------------------------------------------------------------


class RheoJaxInitWarning(UserWarning):
    """Emitted when auto_p0 estimation partially fails.

    Some parameters may have been estimated successfully while others fell
    back to defaults. The warning message identifies which parameters could
    not be estimated.
    """


class RheoJaxPhysicsWarning(UserWarning):
    """Emitted for post-fit physics violations.

    Examples: negative moduli, fractional orders outside [0, 1],
    negative relaxation times, or thermodynamically inconsistent parameters.
    """


class UnsupportedDataError(ValueError):
    """Raised for unsupported measurement geometry. RheoJAX is shear-only;
    tensile/bending/compression (DMTA/DMA, E*) data is not supported."""
