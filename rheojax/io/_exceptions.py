"""Custom exceptions and warnings for the RheoJAX I/O subsystem."""


class RheoJaxFormatError(ValueError):
    """Raised when no reader can parse a file.

    Subclasses ValueError so downstream ``except ValueError`` still catches it.
    """


class RheoJaxValidationWarning(UserWarning):
    """Emitted for data quality issues detected during loading or validation.

    Subclasses UserWarning so standard warning filters apply.
    """
