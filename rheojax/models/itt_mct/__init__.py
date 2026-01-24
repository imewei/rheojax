"""ITT-MCT (Integration Through Transients Mode-Coupling Theory) models.

This package provides Mode-Coupling Theory implementations for dense
colloidal suspensions and glassy materials:

Models
------
ITTMCTSchematic
    F₁₂ schematic model with scalar correlator. Fast and efficient for
    all six rheological protocols. Captures glass transition, yield stress,
    shear thinning, and stress overshoot.

ITTMCTIsotropic
    Full isotropically sheared model with k-resolved correlators and S(k).
    More accurate but computationally intensive. Requires structure factor input.

Supported Protocols
-------------------
- Flow curve (steady shear): σ(γ̇)
- SAOS (oscillation): G'(ω), G''(ω)
- Startup: σ(t) at constant γ̇
- Creep: J(t) at constant σ
- Stress relaxation: σ(t) after flow cessation
- LAOS: σ(t) at γ = γ₀sin(ωt)

Example
-------
>>> from rheojax.models.itt_mct import ITTMCTSchematic
>>>
>>> # Create model in glass state
>>> model = ITTMCTSchematic(epsilon=0.05)  # ε > 0 → glass
>>>
>>> # Predict flow curve (shows yield stress)
>>> gamma_dot = np.logspace(-3, 3, 50)
>>> sigma = model.predict(gamma_dot, test_mode='flow_curve')
>>>
>>> # Predict linear viscoelasticity
>>> omega = np.logspace(-2, 2, 50)
>>> G_star = model.predict(omega, test_mode='oscillation')

References
----------
Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids"
Fuchs M. & Cates M.E. (2002) Phys. Rev. Lett. 89, 248304
Brader J.M. et al. (2008) J. Phys.: Condens. Matter 20, 494243
"""

from rheojax.models.itt_mct._base import ITTMCTBase
from rheojax.models.itt_mct.isotropic import ITTMCTIsotropic
from rheojax.models.itt_mct.schematic import ITTMCTSchematic

__all__ = [
    "ITTMCTBase",
    "ITTMCTSchematic",
    "ITTMCTIsotropic",
]
