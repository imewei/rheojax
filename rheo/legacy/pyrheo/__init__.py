"""PyRheo compatibility layer.

This module provides backward compatibility for users migrating from pyRheo.
Import paths from pyRheo will be mapped to the new rheo structure.
"""

import warnings

warnings.warn(
    "The rheo.legacy.pyrheo module is provided for backward compatibility only. "
    "Please migrate to the new rheo API for better performance and features.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # Compatibility imports will be added as modules are implemented
]
