Fractional Viscoelastic Models
==============================

This section documents the fractional calculus-based viscoelastic models
that capture power-law relaxation and broad spectral behavior.


Quick Reference
---------------

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`fractional_maxwell_gel`
     - 3
     - Gels with terminal flow, SpringPot + dashpot
   * - :doc:`fractional_maxwell_liquid`
     - 3
     - Viscoelastic liquids, spring + SpringPot
   * - :doc:`fractional_maxwell_model`
     - 4
     - Two-order generalized, hierarchical relaxation
   * - :doc:`fractional_kelvin_voigt`
     - 3
     - Solids with bounded creep, spring :math:`\parallel` SpringPot
   * - :doc:`fractional_zener_ss`
     - 4
     - Solid-Solid Zener, low-frequency plateau
   * - :doc:`fractional_zener_sl`
     - 4
     - Solid-Liquid Zener, terminal flow
   * - :doc:`fractional_zener_ll`
     - 4
     - Liquid-Liquid Zener, double flow
   * - :doc:`fractional_kv_zener`
     - 5
     - Complex retardation, bounded creep + plateau
   * - :doc:`fractional_jeffreys`
     - 5
     - Polymer solutions, two relaxation modes
   * - :doc:`fractional_burgers`
     - 6
     - Primary creep, four-element
   * - :doc:`fractional_poynting_thomson`
     - 5
     - Solid with multiple timescales


Overview
--------

Fractional viscoelastic models replace integer-order derivatives in classical
constitutive equations with fractional-order derivatives, enabling:

- **Power-law relaxation**: :math:`G(t) \sim t^{-\alpha}` for broad spectra
- **Parsimonious fitting**: Fewer parameters than multi-mode models
- **Physical insight**: Fractional order :math:`\alpha` relates to structural heterogeneity

The **SpringPot** element is the fundamental building block, interpolating
between ideal spring (:math:`\alpha` = 0) and dashpot (:math:`\alpha` = 1) behavior.

.. include:: /_includes/fractional_seealso.rst


Model Hierarchy
---------------

::

   Fractional Models
   │
   ├── Maxwell Family (Series)
   │   ├── FractionalMaxwellGel (FMG)
   │   │   └── SpringPot ── Dashpot
   │   │   └── Gel with terminal flow
   │   │
   │   ├── FractionalMaxwellLiquid (FML)
   │   │   └── Spring ── SpringPot
   │   │   └── Viscoelastic liquid
   │   │
   │   └── FractionalMaxwellModel (Two-Order)
   │       └── SpringPot(α) ── SpringPot(β)
   │       └── Hierarchical relaxation
   │
   ├── Kelvin-Voigt Family (Parallel)
   │   ├── FractionalKelvinVoigt
   │   │   └── Spring \parallel SpringPot
   │   │   └── Solid with bounded creep
   │   │
   │   └── FractionalKVZener
   │       └── Spring ── [Spring \parallel SpringPot]
   │       └── Complex retardation
   │
   ├── Zener Family (Combined)
   │   ├── FractionalZenerSS
   │   │   └── Spring ── [SpringPot \parallel Spring]
   │   │   └── Solid-Solid, plateau at both limits
   │   │
   │   ├── FractionalZenerSL
   │   │   └── Spring ── [SpringPot \parallel Dashpot]
   │   │   └── Solid-Liquid, terminal flow
   │   │
   │   └── FractionalZenerLL
   │       └── Dashpot ── [SpringPot \parallel Dashpot]
   │       └── Liquid-Liquid, double flow
   │
   └── Extended Models
       ├── FractionalJeffreys
       │   └── Two relaxation modes
       │
       ├── FractionalBurgers
       │   └── Four-element, primary creep
       │
       └── FractionalPoyntingThomson
           └── Multiple timescales


When to Use Which Model
-----------------------

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Material Type
     - Recommended Model
     - Alternatives
     - Key Indicator
   * - Gel (terminal flow)
     - FMG
     - FZSL
     - G'' > G' at low :math:`\omega`
   * - Polymer melt
     - FML
     - FMG, FZSL
     - G'' crosses G' once
   * - Crosslinked gel
     - FKV, FZSS
     - —
     - G' plateau both limits
   * - Biological tissue
     - FKV
     - FZSS
     - Bounded compliance
   * - Hierarchical material
     - Two-Order FM
     - FBurgers
     - Two power-law slopes
   * - Critical gel (gel point)
     - SpringPot
     - FMG (:math:`\alpha` ≈ 0.5)
     - tan :math:`\delta` ≈ const

**Decision Flowchart:**

1. Does material flow at long times (G'' > G' as :math:`\omega \to 0`)?
   - **Yes** :math:`\to` Maxwell family (FMG, FML, FZSL, FZLL)
   - **No** :math:`\to` Kelvin-Voigt family or FZSS

2. Is there a high-frequency plateau in G'?
   - **Yes** :math:`\to` Models with spring in series (FML, FZSS, FZSL)
   - **No** :math:`\to` Models starting with SpringPot (FMG, FKV)

3. Are two power-law regimes visible?
   - **Yes** :math:`\to` Two-Order FM or FBurgers
   - **No** :math:`\to` Single-order models


Key Parameters
--------------

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - Fractional order
     - :math:`\alpha`
     - —
     - 0 = solid, 1 = liquid, 0.5 = critical gel
   * - SpringPot constant
     - :math:`c_\alpha`
     - Pa·s\ :math:`^{\alpha}`
     - Sets magnitude (unusual units)
   * - Shear modulus
     - G
     - Pa
     - Elastic plateau stiffness
   * - Viscosity
     - :math:`\eta`
     - Pa·s
     - Terminal viscosity (when present)
   * - Relaxation time
     - :math:`\tau`
     - s
     - Crossover frequency :math:`\omega \approx 1/\tau`

**Physical interpretation of** :math:`\alpha`:

- :math:`\alpha \to 0`: Nearly elastic, broad relaxation spectrum
- :math:`\alpha \to 0.3` **–0.5**: Typical for soft solids, gels
- :math:`\alpha \to 0.5`: Critical gel, self-similar structure
- :math:`\alpha \to 0.7` **–0.9**: Approaching Newtonian behavior
- :math:`\alpha \to 1`: Classical dashpot (Newtonian)


Quick Start
-----------

**Fractional Maxwell Gel (soft gels):**

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   import numpy as np

   model = FractionalMaxwellGel()
   omega = np.logspace(-2, 2, 50)

   # Fit to oscillatory data
   model.fit(omega, G_star, test_mode='oscillation')

   # Fractional order indicates spectrum breadth
   alpha = model.parameters.get_value('alpha')
   print(f"Fractional order: {alpha:.2f}")

**Fractional Kelvin-Voigt (bounded creep):**

.. code-block:: python

   from rheojax.models import FractionalKelvinVoigt

   model = FractionalKelvinVoigt()
   model.fit(t, J_t, test_mode='creep')

   # Equilibrium compliance
   Ge = model.parameters.get_value('Ge')
   J_eq = 1 / Ge

**Bayesian inference:**

.. code-block:: python

   # Bayesian with warm-start from NLSQ
   result = model.fit_bayesian(
       omega, G_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Credible intervals for fractional order
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"alpha: [{intervals['alpha'][0]:.2f}, {intervals['alpha'][1]:.2f}]")


Model Documentation
-------------------

**Maxwell Family:**

.. toctree::
   :maxdepth: 1

   fractional_maxwell_gel
   fractional_maxwell_liquid
   fractional_maxwell_model

**Kelvin-Voigt Family:**

.. toctree::
   :maxdepth: 1

   fractional_kelvin_voigt
   fractional_kv_zener

**Zener Family:**

.. toctree::
   :maxdepth: 1

   fractional_zener_ss
   fractional_zener_sl
   fractional_zener_ll

**Extended Models:**

.. toctree::
   :maxdepth: 1

   fractional_jeffreys
   fractional_burgers
   fractional_poynting_thomson


See Also
--------

- :doc:`/models/classical/index` — Integer-order building blocks
- :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations
- :doc:`/models/sgr/index` — Power-law from disordered structure (SGR approach)
- :doc:`/transforms/mastercurve` — Time-temperature superposition
- :doc:`/examples/advanced/04-fractional-models-deep-dive` — Comparison notebook


References
----------

1. Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press. https://doi.org/10.1142/p614

2. Schiessel, H., Metzler, R., Blumen, A., & Nonnenmacher, T.F. (1995).
   "Generalized viscoelastic models: their fractional equations with solutions."
   *J. Phys. A*, 28, 6567–6584. https://doi.org/10.1088/0305-4470/28/23/012

3. Bagley, R.L. & Torvik, P.J. (1983). "A theoretical basis for the application
   of fractional calculus to viscoelasticity." *J. Rheol.*, 27, 201–210.
   https://doi.org/10.1122/1.549724

4. Jaishankar, A. & McKinley, G.H. (2013). "Power-law rheology in the bulk
   and at the interface." *Proc. R. Soc. A*, 469, 20120284.
   https://doi.org/10.1098/rspa.2012.0284

5. Friedrich, C. (1991). "Relaxation and retardation functions of the Maxwell
   model with fractional derivatives." *Rheol. Acta*, 30, 151–158.
   https://doi.org/10.1007/BF01134604

6. Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.
   ISBN: 978-0125588409

7. Gorenflo, R. et al. (2014). *Mittag-Leffler Functions, Related Topics and
   Applications*. Springer. https://doi.org/10.1007/978-3-662-43930-2
