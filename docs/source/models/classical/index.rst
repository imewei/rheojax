Classical Viscoelastic Models
=============================

This section documents the classical linear viscoelastic models that form
the foundation of rheological analysis.


Quick Reference
---------------

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`maxwell`
     - 2 (G, η)
     - Stress relaxation, simple viscoelastic liquids
   * - :doc:`zener`
     - 3 (G₁, G₂, η)
     - Solids with single relaxation time, standard linear solid
   * - :doc:`springpot`
     - 2 (c_α, α)
     - Power-law behavior, fractional element, broad spectra


Overview
--------

Classical viscoelastic models represent the historical foundation of rheology,
built from combinations of ideal mechanical elements:

- **Spring** (Hookean): Instantaneous elastic response, :math:`\sigma = G\gamma`
- **Dashpot** (Newtonian): Viscous flow, :math:`\sigma = \eta\dot{\gamma}`
- **SpringPot** (Fractional): Power-law intermediate behavior

These models provide closed-form analytical solutions for standard rheological
tests and serve as building blocks for more complex constitutive equations.


Model Hierarchy
---------------

::

   Classical Models
   │
   ├── Maxwell (Series)
   │   └── Spring ── Dashpot
   │   └── Liquid-like (terminal flow)
   │   └── Single relaxation time τ = η/G
   │
   ├── Zener (Standard Linear Solid)
   │   └── Spring ── [Spring ∥ Dashpot]
   │   └── Solid-like (equilibrium modulus)
   │   └── Kelvin-Voigt element + spring
   │
   └── SpringPot (Fractional Element)
       └── Intermediate between spring and dashpot
       └── Power-law kernel: G(t) ∼ t^(-α)
       └── Foundation for fractional models


When to Use Which Model
-----------------------

.. list-table::
   :widths: 30 20 20 30
   :header-rows: 1

   * - Material Behavior
     - Maxwell
     - Zener
     - SpringPot
   * - Single exponential relaxation
     - ✓ Best choice
     - ✓ With plateau
     - Overkill
   * - Terminal flow (liquid)
     - ✓ Best choice
     - ✗
     - ✗
   * - Equilibrium modulus (solid)
     - ✗
     - ✓ Best choice
     - ✗
   * - Power-law relaxation
     - ✗
     - ✗
     - ✓ Best choice
   * - Broad relaxation spectrum
     - Poor fit
     - Poor fit
     - ✓ Best choice
   * - Simple teaching example
     - ✓ Best choice
     - ✓ Good
     - More complex

**Decision Guide:**

- **Start with Maxwell** for viscoelastic liquids (polymer melts, solutions)
- **Use Zener** for viscoelastic solids (rubbers, gels with crosslinks)
- **Use SpringPot** when log-log plots show power-law slopes (polymeric glasses,
  biological tissues, complex fluids)


Key Parameters
--------------

.. list-table::
   :widths: 15 10 20 55
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - Shear modulus
     - G
     - Pa
     - Elastic stiffness (energy storage)
   * - Viscosity
     - η
     - Pa·s
     - Resistance to flow (energy dissipation)
   * - Relaxation time
     - τ
     - s
     - τ = η/G, characteristic timescale
   * - SpringPot constant
     - c_α
     - Pa·s^α
     - Quasi-property (fractional element)
   * - Fractional order
     - α
     - —
     - 0 = elastic, 1 = viscous, 0.5 = critical gel


Quick Start
-----------

**Maxwell model:**

.. code-block:: python

   from rheojax.models import Maxwell
   import numpy as np

   model = Maxwell()
   model.fit(t, G_t, test_mode='relaxation')

   # Get fitted relaxation time
   tau = model.parameters.get_value('eta') / model.parameters.get_value('G')

**Zener model:**

.. code-block:: python

   from rheojax.models import Zener
   import numpy as np

   model = Zener()
   model.fit(omega, G_star, test_mode='oscillation')

   # Equilibrium modulus
   G_eq = model.parameters.get_value('G2')

**SpringPot element:**

.. code-block:: python

   from rheojax.models import SpringPot

   model = SpringPot()
   model.fit(omega, G_star, test_mode='oscillation')

   # Fractional order indicates spectrum breadth
   alpha = model.parameters.get_value('alpha')


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   maxwell
   zener
   springpot


See Also
--------

- :doc:`/models/fractional/index` — Extended models with fractional calculus
- :doc:`/models/multi_mode/generalized_maxwell` — Multiple Maxwell elements for broad spectra
- :doc:`/transforms/mastercurve` — Time-temperature superposition
- :doc:`/user_guide/model_selection` — Comprehensive model selection guide


References
----------

1. Maxwell, J.C. (1867). "On the dynamical theory of gases."
   *Philosophical Transactions*, 157, 49-88. https://www.jstor.org/stable/108968

2. Zener, C.M. (1948). *Elasticity and Anelasticity of Metals*.
   University of Chicago Press. https://doi.org/10.1002/9781118661275

3. Ferry, J.D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed.
   John Wiley & Sons. ISBN: 978-0471048947.

4. Tschoegl, N.W. (1989). *The Phenomenological Theory of Linear Viscoelastic
   Behavior*. Springer-Verlag. https://doi.org/10.1007/978-3-642-73602-5

5. Koeller, R.C. (1984). "Applications of fractional calculus to the theory of
   viscoelasticity." *J. Appl. Mech.*, 51, 299-307. https://doi.org/10.1115/1.3167616
