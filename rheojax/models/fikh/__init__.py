"""FIKH (Fractional IKH) model family.

This module provides Fractional Isotropic-Kinematic Hardening models
for thixotropic elasto-viscoplastic materials with memory effects.

Models:
    FIKH: Single-mode fractional IKH with optional thermal coupling.
    FMLIKH: Multi-layer (multi-mode) variant with per-mode parameters.

Key Features:
    - Caputo fractional derivative for structure evolution (power-law memory)
    - Full thermokinematic coupling (Arrhenius viscosity, thermal yield)
    - Armstrong-Frederick kinematic hardening
    - 6 protocols: flow_curve, startup, relaxation, creep, oscillation, LAOS

Example:
    >>> from rheojax.models.fikh import FIKH, FMLIKH
    >>>
    >>> # Single-mode FIKH with thermal coupling
    >>> model = FIKH(include_thermal=True, alpha_structure=0.5)
    >>> model.fit(t, stress, test_mode='startup', strain=strain)
    >>>
    >>> # Multi-mode FMLIKH for broad relaxation spectrum
    >>> model = FMLIKH(n_modes=3, include_thermal=False)
    >>> model.fit(omega, G_star, test_mode='oscillation')

Mathematical Background:
    The fractional structure evolution follows:
        D^α_C λ = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

    where D^α_C is the Caputo fractional derivative:
        D^α_C f(t) = (1/Γ(1-α)) ∫₀ᵗ f'(s)/(t-s)^α ds

    Key property: α → 1 recovers classical IKH (exponential relaxation)
                  α → 0 gives strong memory (power-law relaxation)
"""

from rheojax.models.fikh.fikh import FIKH
from rheojax.models.fikh.fmlikh import FMLIKH

__all__ = ["FIKH", "FMLIKH"]
