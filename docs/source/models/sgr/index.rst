Soft Glassy Rheology (SGR) Models
=================================

This section documents the Soft Glassy Rheology (SGR) family of models for
disordered soft materials exhibiting glassy dynamics.

.. include:: /_includes/glass_transition_physics.rst


Quick Reference
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`sgr_conventional`
     - 3 (:math:`x`, :math:`G_0`, :math:`\tau_0`)
     - Soft glassy materials, aging, yield stress fluids
   * - :doc:`sgr_generic`
     - 4-5
     - Thermodynamically consistent extension (GENERIC framework)


Overview
--------

The Soft Glassy Rheology (SGR) model is a mesoscopic constitutive framework
for **soft glassy materials (SGMs)**—systems exhibiting structural disorder and
metastability similar to glasses but with interaction energies of order :math:`k_B T`:

- **Foams** (shaving cream, bread dough)
- **Dense emulsions** (mayonnaise, salad cream)
- **Pastes** (toothpaste, hair gel)
- **Colloidal glasses** (paints, ceramic slips)
- **Polymer gels** (physical gels, block copolymers)

**Key physics captured:**

- **Noise-activated hopping**: Elements escape energy traps via effective noise temperature :math:`x`
- **Glass transition**: Phase transition at :math:`x = 1` (fluid :math:`\leftrightarrow` glass)
- **Aging**: Time-dependent evolution of trapped state distribution
- **Power-law rheology**: :math:`G' \sim G'' \sim \omega^{x-1}` in fluid regime
- **Yield stress**: Emerges in glass regime (:math:`x < 1`)

The model was developed by Sollich, Lequeux, Hébraud, and Cates based on
Bouchaud's trap model for structural glasses.


Model Hierarchy
---------------

::

   SGR Family
   │
   ├── SGR Conventional (Sollich 1998)
   │   └── Trap model with Arrhenius hopping
   │   └── Exponential trap depth distribution ρ(E) = e^(-E)
   │   └── Strain-warped time Z(t,t') for flow coupling
   │   └── 3 core parameters: x, G_0, τ_0
   │
   └── SGR GENERIC (Fuereder & Ilg 2013)
       └── Thermodynamically consistent extension
       └── GENERIC framework (reversible + irreversible)
       └── Proper dissipation and entropy production
       └── Improved nonlinear response predictions


When to Use Which Model
-----------------------

.. list-table::
   :widths: 35 30 35
   :header-rows: 1

   * - Feature / Use Case
     - SGR Conventional
     - SGR GENERIC
   * - Linear oscillatory (SAOS)
     - ✓ Standard choice
     - ✓ Equivalent
   * - Aging and rejuvenation
     - ✓ Full support
     - ✓ Full support
   * - Large amplitude (LAOS)
     - Qualitative
     - ✓ Better nonlinear
   * - Thermodynamic consistency
     - ~
     - ✓ Guaranteed
   * - Steady flow curves
     - ✓ Good
     - ✓ Better at high rates
   * - Computational cost
     - 1× (faster)
     - 2-3× (more expensive)
   * - Simple interpretation
     - ✓ Standard
     - More complex

**Decision Guide:**

- **Start with SGR Conventional** for standard characterization (SAOS, flow curves)
- **Use SGR GENERIC** when thermodynamic consistency matters (nonlinear, LAOS)
  or when conventional model shows systematic deviations


SGR Phase Diagram
-----------------

The SGR model exhibits a genuine phase transition controlled by the effective
noise temperature :math:`x`:

::

   x (noise temperature)
   │
   │  x > 2     Newtonian Fluid
   │            G' ~ ω^2, G'' ~ ω
   │            Classical liquid behavior
   │
   │  1 < x < 2 Power-Law Fluid
   │            G' ~ G'' ~ ω^(x-1) 
   │            Flat loss tangent: tan δ = tan(πx/2)
   │            Broad relaxation spectrum
   │
   │  x = 1     Glass Transition (Critical Point)
   │            Logarithmic aging, critical slowing
   │
   │  x < 1     Soft Glass
   │            Yield stress emerges
   │            G' >> G'', weak frequency dependence
   │            Aging without equilibration
   │
   └─────────────────────────────────────────────

**Physical interpretation of** :math:`x` **:**

- :math:`x` **represents the ratio of "noise energy" to typical trap depth**
- High :math:`x`: Frequent hopping, equilibrium attained, liquid-like
- Low :math:`x`: Rare hopping, aging dominates, solid-like
- :math:`x \approx 1`: Marginal stability, critical dynamics


Key Parameters
--------------

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Physical Meaning
   * - Noise temperature
     - :math:`x`
     - 0.5–3
     - Controls phase: :math:`x < 1` (glass), :math:`x > 1` (fluid)
   * - Modulus scale
     - :math:`G_0`
     - :math:`10\text{--}10^4` Pa
     - Sets magnitude of :math:`G'`, :math:`G''`
   * - Attempt time
     - :math:`\tau_0`
     - :math:`10^{-6}`–:math:`10^{-2}` s
     - Microscopic timescale for trap escape


Quick Start
-----------

**SGR Conventional model:**

.. code-block:: python

   from rheojax.models import SGRConventional
   import numpy as np

   # Create model
   model = SGRConventional()

   # Set parameters for a soft glassy material
   model.parameters.set_value('x', 1.3)      # Power-law fluid regime
   model.parameters.set_value('G0', 1000.0)  # Pa
   model.parameters.set_value('tau0', 1e-4)  # s

   # Fit to oscillatory data
   omega = np.logspace(-2, 2, 50)
   model.fit(omega, G_star_data, test_mode='oscillation')

   # Check if material is in glass or fluid regime
   x = model.parameters.get_value('x')
   if x < 1:
       print(f"Glass regime (x = {x:.2f}): Yield stress expected")
   else:
       print(f"Fluid regime (x = {x:.2f}): Power-law G' ~ G'' ~ ω^{x-1:.2f}")

**Bayesian inference:**

.. code-block:: python

   # Bayesian with NLSQ warm-start
   result = model.fit_bayesian(
       omega, G_star_data,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Credible interval for noise temperature
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"x: [{intervals['x'][0]:.2f}, {intervals['x'][1]:.2f}]")

**GENERIC formulation:**

.. code-block:: python

   from rheojax.models import SGRGeneric

   # Thermodynamically consistent version
   model = SGRGeneric()
   model.fit(omega, G_star_data, test_mode='oscillation')


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   sgr_conventional
   sgr_generic


See Also
--------

- :doc:`/models/hl/index` — Hébraud-Lequeux: mean-field limit of trap dynamics
- :doc:`/models/epm/index` — EPM: spatially-resolved plasticity
- :doc:`/models/stz/index` — STZ: shear transformation zones
- :doc:`/models/fluidity/index` — Fluidity models for yield stress fluids
- :doc:`/transforms/srfs` — Strain-rate frequency superposition (SGR analog of TTS)
- :doc:`/user_guide/soft_glassy_materials` — Introduction to soft glassy rheology


References
----------

1. Sollich, P., Lequeux, F., Hébraud, P., & Cates, M.E. (1997). "Rheology of soft
   glassy materials." *Phys. Rev. Lett.*, 78, 2020–2023.
   https://doi.org/10.1103/PhysRevLett.78.2020

2. Sollich, P. (1998). "Rheological constitutive equation for a model of soft glassy
   materials." *Phys. Rev. E*, 58, 738–759.
   https://doi.org/10.1103/PhysRevE.58.738

3. Fielding, S.M., Sollich, P., & Cates, M.E. (2000). "Aging and rheology in soft
   materials." *J. Rheol.*, 44, 323–369.
   https://doi.org/10.1122/1.551088

4. Fuereder, I. & Ilg, P. (2013). "Nonequilibrium thermodynamics of the soft glassy
   rheology model." *Phys. Rev. E*, 88, 042134.
   https://doi.org/10.1103/PhysRevE.88.042134

5. Sollich, P. & Cates, M.E. (2012). "Thermodynamic interpretation of soft glassy
   rheology models." *Phys. Rev. E*, 85, 031127.
   https://doi.org/10.1103/PhysRevE.85.031127

6. Cates, M.E. & Sollich, P. (2004). "Tensorial constitutive models for disordered
   foams, dense emulsions, and other soft nonergodic materials."
   *J. Rheol.*, 48, 193–207. https://doi.org/10.1122/1.1634985

7. Bouchaud, J.P. (1992). "Weak ergodicity breaking and aging in disordered systems."
   *J. Phys. I France*, 2, 1705–1713.
   https://doi.org/10.1051/jp1:1992238
