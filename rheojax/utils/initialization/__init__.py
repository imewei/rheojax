"""Smart parameter initialization for fractional models in oscillation mode.

This module provides data-driven initialization strategies to improve optimization
convergence for fractional viscoelastic models when fitting frequency-domain data.

The initialization extracts features from the complex modulus G*(ω) such as:
- Low-frequency plateau (equilibrium modulus)
- High-frequency plateau (total modulus)
- Transition frequency (characteristic relaxation time)
- Slope in transition region (fractional order)

These features provide much better starting points than arbitrary default values,
helping the optimizer avoid local minima in the non-convex landscape created by
Mittag-Leffler functions.

**Architecture (v0.2.0):**
This module has been refactored to use the Template Method design pattern:
- `BaseInitializer` (in `initialization/base.py`): Abstract class enforcing 5-step algorithm
- 11 concrete initializers (in `initialization/fractional_*.py`): Model-specific implementations
- This module (`initialization/__init__.py`): Facade providing backward-compatible API

Benefits:
- Eliminates code duplication (49% reduction: 932 → 471 lines)
- Enforces consistent initialization algorithm across all 11 fractional models
- Near-zero performance overhead (<0.01% of total fitting time)
- 100% backward compatibility: all public functions preserved

References
----------
- Issue #9: Fractional models fail to optimize in oscillation mode due to local minima
- Template Method pattern documented in CLAUDE.md under "Key Design Patterns"
"""

# Import base class and utility functions
from rheojax.utils.initialization.base import (
    BaseInitializer,
    extract_frequency_features,
)

# Import concrete initializers
from rheojax.utils.initialization.fractional_burgers import (
    FractionalBurgersInitializer,
)
from rheojax.utils.initialization.fractional_jeffreys import (
    FractionalJeffreysInitializer,
)
from rheojax.utils.initialization.fractional_kelvin_voigt import (
    FractionalKelvinVoigtInitializer,
)
from rheojax.utils.initialization.fractional_kv_zener import (
    FractionalKVZenerInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_gel import (
    FractionalMaxwellGelInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_liquid import (
    FractionalMaxwellLiquidInitializer,
)
from rheojax.utils.initialization.fractional_maxwell_model import (
    FractionalMaxwellModelInitializer,
)
from rheojax.utils.initialization.fractional_poynting_thomson import (
    FractionalPoyntingThomsonInitializer,
)
from rheojax.utils.initialization.fractional_zener_ll import (
    FractionalZenerLLInitializer,
)
from rheojax.utils.initialization.fractional_zener_sl import (
    FractionalZenerSLInitializer,
)
from rheojax.utils.initialization.fractional_zener_ss import (
    FractionalZenerSSInitializer,
)

# Create wrapper functions for backward compatibility
# These delegate to the concrete initializers


def initialize_fractional_zener_ss(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalZenerSSInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_maxwell_liquid(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalMaxwellLiquidInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_maxwell_gel(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalMaxwellGelInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_zener_ll(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalZenerLLInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_zener_sl(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalZenerSLInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_kelvin_voigt(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalKelvinVoigtInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_maxwell_model(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalMaxwellModelInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_kv_zener(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalKVZenerInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_poynting_thomson(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalPoyntingThomsonInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_jeffreys(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalJeffreysInitializer()
    return initializer.initialize(omega, G_star, param_set)


def initialize_fractional_burgers(omega, G_star, param_set):
    """Wrapper for backward compatibility. See initialization.py for full docs."""
    initializer = FractionalBurgersInitializer()
    return initializer.initialize(omega, G_star, param_set)


__all__ = [
    # Public API functions (backward compatible)
    "extract_frequency_features",
    "initialize_fractional_burgers",
    "initialize_fractional_jeffreys",
    "initialize_fractional_kelvin_voigt",
    "initialize_fractional_kv_zener",
    "initialize_fractional_maxwell_gel",
    "initialize_fractional_maxwell_liquid",
    "initialize_fractional_maxwell_model",
    "initialize_fractional_poynting_thomson",
    "initialize_fractional_zener_ll",
    "initialize_fractional_zener_sl",
    "initialize_fractional_zener_ss",
    # Base class
    "BaseInitializer",
    # Concrete initializers
    "FractionalBurgersInitializer",
    "FractionalJeffreysInitializer",
    "FractionalKelvinVoigtInitializer",
    "FractionalKVZenerInitializer",
    "FractionalMaxwellGelInitializer",
    "FractionalMaxwellLiquidInitializer",
    "FractionalMaxwellModelInitializer",
    "FractionalPoyntingThomsonInitializer",
    "FractionalZenerLLInitializer",
    "FractionalZenerSLInitializer",
    "FractionalZenerSSInitializer",
]
