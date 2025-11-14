.. _model_families:

Model Families Overview
========================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Distinguish between classical, fractional, and flow model families
   2. Identify which family applies to your experimental data
   3. Understand the advantages and limitations of each family
   4. Select candidate models within each family
   5. Recognize when to upgrade from simple to complex models

.. admonition:: Prerequisites
   :class: important

   - :doc:`../01_fundamentals/material_classification` — Liquids, solids, gels
   - :doc:`../01_fundamentals/test_modes` — SAOS, relaxation, creep, flow
   - :doc:`getting_started` — Basic fitting workflow

The Three Model Families
-------------------------

RheoJAX provides **20 rheological models** organized into three families:

1. **Classical Viscoelastic** (3 models) — Single or discrete relaxation times
2. **Fractional Viscoelastic** (11 models) — Power-law relaxation, distributed timescales
3. **Flow Models** (6 models) — Nonlinear steady shear behavior

Each family targets different material types and test modes.

Quick Reference Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 30 25

   * - Family
     - Test Modes
     - Material Types
     - Key Feature
   * - Classical
     - SAOS, Relaxation, Creep
     - Simple liquids/solids
     - Exponential decay
   * - Fractional
     - SAOS, Relaxation, Creep
     - Complex materials
     - Power-law relaxation
   * - Flow
     - Steady Shear
     - Nonlinear fluids
     - Shear thinning/thickening

Family 1: Classical Viscoelastic Models
----------------------------------------

**Who they're for**: Simple polymers, dilute solutions, basic characterization

**Test modes**: SAOS, stress relaxation, creep

**Key characteristic**: **Exponential relaxation** (single or discrete timescales)

The Models
~~~~~~~~~~

1. **Maxwell** (2 parameters)

   - **Type**: Viscoelastic liquid
   - **Equation**: G(t) = G₀ exp(-t/τ)
   - **Use for**: Single relaxation time, simple liquids
   - **Example**: Low molecular weight polymer melts

2. **Zener** (3 parameters)

   - **Type**: Viscoelastic solid (Standard Linear Solid)
   - **Equation**: G(t) = Ge + Gm exp(-t/τ)
   - **Use for**: Materials with equilibrium modulus (crosslinked)
   - **Example**: Lightly crosslinked rubbers

3. **SpringPot** (2 parameters)

   - **Type**: Pure fractional element (bridges solid and liquid)
   - **Equation**: G(t) ~ t^(-α)
   - **Use for**: Power-law behavior, conceptual studies
   - **Example**: Critical gels

When to Use Classical Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Use classical models when**:

- Material is simple (monodisperse polymer, dilute solution)
- Relaxation appears exponential in log-log plot
- You need quick characterization
- Material is well-described by single timescale

❌ **Avoid classical models when**:

- Relaxation curve is not straight in semi-log plot (exponential)
- Material has broad molecular weight distribution
- Gel-like or power-law behavior observed
- Fitting fails or requires many Maxwell elements in parallel

Example: Fitting Maxwell Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.maxwell import Maxwell

   # Stress relaxation data
   model = Maxwell()
   model.fit(time, G_t, test_mode='relaxation')

   # Parameters
   G0 = model.parameters.get_value('G0')    # Modulus (Pa)
   eta = model.parameters.get_value('eta')  # Viscosity (Pa·s)
   tau = eta / G0                            # Relaxation time (s)

**For detailed equations**: :doc:`/models/classical/index`

Family 2: Fractional Viscoelastic Models
-----------------------------------------

**Who they're for**: Complex materials with broad relaxation spectra

**Test modes**: SAOS, stress relaxation, creep

**Key characteristic**: **Power-law relaxation** via fractional derivatives

Why Fractional Models?
~~~~~~~~~~~~~~~~~~~~~~~

Real materials often have **distributions of relaxation times** due to:

- Polydispersity (molecular weight distribution)
- Branching and entanglements
- Fractal or heterogeneous structure

**Fractional calculus** captures these distributions with a single parameter α (fractional order).

**Power-law signatures**:

- Relaxation: G(t) ~ t^(-α)
- Oscillation: G', G" ~ ω^α (parallel scaling)

The 11 Fractional Models
~~~~~~~~~~~~~~~~~~~~~~~~~

**Liquids** (zero equilibrium modulus):

1. **Fractional Maxwell Liquid (FML)** — Most common for polymer melts
2. **Fractional Maxwell Model (FMM)** — Generalized Maxwell
3. **Fractional Jeffreys** — Retardation + relaxation

**Solids** (finite equilibrium modulus):

4. **Fractional Zener Solid-Solid (FZSS)** — Two solid elements
5. **Fractional Kelvin-Voigt (FKV)** — Solid with fractional damping
6. **Fractional KV-Zener** — Extended Kelvin-Voigt
7. **Fractional Zener Liquid-Liquid (FZLL)** — Two liquid elements
8. **Fractional Zener Solid-Liquid (FZSL)** — Hybrid

**Gels** (power-law across all frequencies):

9. **Fractional Maxwell Gel (FMG)** — Critical gel
10. **Fractional Burgers** — Complex gel behavior
11. **Fractional Poynting-Thomson** — Extended gel model

When to Use Fractional Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Use fractional models when**:

- Log-log plot shows parallel G' and G" lines (power-law)
- Material has broad molecular weight distribution
- Relaxation is NOT purely exponential
- Classical models fail to capture curvature
- You observe gel-like behavior

❌ **Avoid fractional models when**:

- Material is simple (use classical first)
- You have very few data points (< 10)
- Single exponential fits perfectly well

The Fractional Order Parameter (α)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**α** characterizes the breadth of the relaxation spectrum:

- **α → 1**: Narrow spectrum (nearly liquid-like, approaches Maxwell)
- **α ≈ 0.5**: Broad spectrum (gel-like, power-law)
- **α → 0**: Narrow spectrum (nearly solid-like, approaches elastic)

**Typical values**:

- Monodisperse polymers: α = 0.8 - 0.95
- Polydisperse polymers: α = 0.5 - 0.7
- Gels: α = 0.3 - 0.6
- Soft glassy materials: α = 0.1 - 0.3

Example: Fitting Fractional Zener Solid-Solid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

   # SAOS frequency sweep (omega, [G', G"])
   model = FractionalZenerSolidSolid()
   model.fit(omega, G_star, test_mode='oscillation')

   # Parameters (automatic smart initialization in v0.2.0)
   Ge = model.parameters.get_value('Ge')           # Equilibrium modulus
   alpha = model.parameters.get_value('alpha')     # Fractional order
   tau = model.parameters.get_value('tau_alpha')   # Characteristic time

   print(f"Gel-like character (α): {alpha:.3f}")

**For detailed equations**: :doc:`/models/fractional/index`

**For fractional calculus background**: :doc:`../03_advanced_topics/fractional_viscoelasticity_reference`

Family 3: Flow Models
----------------------

**Who they're for**: Processing, formulation, quality control

**Test modes**: Steady shear flow (rotation)

**Key characteristic**: **Nonlinear viscosity** η(γ̇)

Why Flow Models?
~~~~~~~~~~~~~~~~

Most complex fluids exhibit **shear-dependent viscosity**:

- **Shear thinning**: η decreases with shear rate (polymers, suspensions)
- **Shear thickening**: η increases with shear rate (dense suspensions)
- **Yield stress**: Material doesn't flow below critical stress (pastes, gels)

**These are nonlinear effects** not captured by linear viscoelastic models.

The 6 Flow Models
~~~~~~~~~~~~~~~~~

**Newtonian and Shear Thinning**:

1. **PowerLaw** (2 parameters) — Simple shear thinning

   - η(γ̇) = K γ̇^(n-1)
   - n < 1: Shear thinning
   - n = 1: Newtonian

2. **Carreau** (4 parameters) — Shear thinning with Newtonian plateaus

   - η(γ̇) = η∞ + (η₀ - η∞) [1 + (λ γ̇)²]^((n-1)/2)
   - Most flexible for polymers

3. **Carreau-Yasuda** (5 parameters) — Extended Carreau

4. **Cross** (4 parameters) — Alternative to Carreau

**Yield Stress**:

5. **Bingham** (2 parameters) — Newtonian above yield stress

   - σ = σ_y + η_p γ̇ (for σ > σ_y)
   - Use for: Drilling fluids, simple pastes

6. **Herschel-Bulkley** (3 parameters) — Power-law above yield stress

   - σ = σ_y + K γ̇^n
   - Use for: Soft solids, yield-stress fluids

When to Use Flow Models
~~~~~~~~~~~~~~~~~~~~~~~~

✅ **Use flow models when**:

- You have steady shear data (shear rate vs. viscosity or stress)
- Material exhibits shear thinning or shear thickening
- You need to predict processing behavior (extrusion, pumping)
- Material has yield stress

❌ **Avoid flow models when**:

- You have oscillation or relaxation data (use viscoelastic models)
- Material is in linear regime
- You need to predict elastic behavior (G')

Example: Fitting PowerLaw Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.power_law import PowerLaw

   # Steady shear data (shear rate, viscosity)
   model = PowerLaw()
   model.fit(shear_rate, viscosity, test_mode='rotation')

   # Parameters
   K = model.parameters.get_value('K')     # Consistency (Pa·s^n)
   n = model.parameters.get_value('n')     # Flow index

   if n < 1:
       print(f"Shear thinning: n = {n:.3f}")
   elif n > 1:
       print(f"Shear thickening: n = {n:.3f}")
   else:
       print("Newtonian fluid")

**For detailed equations**: :doc:`/models/flow/index`

Model Selection Flowchart
--------------------------

.. code-block:: text

   START: What type of data do you have?
      │
      ├─→ SAOS / Relaxation / Creep (linear viscoelasticity)
      │     │
      │     ├─→ Simple exponential decay?
      │     │      └─→ YES: Classical (Maxwell, Zener)
      │     │
      │     └─→ Power-law / gel-like / complex?
      │            └─→ YES: Fractional (FML, FZSS, FMG)
      │
      └─→ Steady Shear Flow (nonlinear)
            │
            ├─→ Shear thinning without yield stress?
            │      └─→ YES: PowerLaw, Carreau
            │
            └─→ Yield stress present?
                   └─→ YES: Bingham, Herschel-Bulkley

Complexity Ladder: When to Upgrade Models
------------------------------------------

Start simple, add complexity only if needed:

**Level 1: Classical** (2-3 parameters)

- Try first for all linear viscoelastic data
- If fit is poor, move to Level 2

**Level 2: Fractional** (3-4 parameters)

- Better for complex materials
- If still poor, check data quality or try Level 3

**Level 3: Multi-mode** (6+ parameters)

- Discrete spectrum (multiple Maxwell/Zener elements)
- Only if you have high-quality data over wide range

**Red flags for over-fitting**:

- More parameters than data points / 3
- Parameters vary wildly with small data changes
- Excellent fit but unphysical parameter values

Key Concepts
------------

.. admonition:: Main Takeaways
   :class: tip

   1. **Classical models** (Maxwell, Zener): Exponential relaxation, simple materials

   2. **Fractional models** (FML, FZSS, FMG): Power-law relaxation, complex materials

   3. **Flow models** (PowerLaw, Carreau, HB): Nonlinear viscosity, steady shear

   4. **Start simple**: Try classical first, upgrade to fractional if needed

   5. **Test mode determines family**: Linear (classical/fractional) vs. nonlinear (flow)

.. admonition:: Self-Check Questions
   :class: tip

   1. **You observe G' and G" scaling as ω^0.4 (parallel lines). Which family?**

      Hint: Power-law scaling → fractional

   2. **Your material shows exponential stress relaxation. Start with classical or fractional?**

      Hint: Exponential → classical (simpler)

   3. **Can you use a Maxwell model for steady shear flow data?**

      Hint: Maxwell is linear viscoelastic, not for flow

   4. **What does α = 0.9 tell you about the material?**

      Hint: Close to 1 → narrow relaxation spectrum, nearly liquid-like

   5. **You fit PowerLaw and get n = 0.6. What does this mean?**

      Hint: n < 1 → shear thinning

Further Reading
---------------

**Within this documentation**:

- :doc:`model_selection` — Decision flowcharts and compatibility checking
- :doc:`fitting_strategies` — Initialization and validation

**Model handbooks** (full equations and theory):

- :doc:`/models/classical/index` — 3 classical models
- :doc:`/models/fractional/index` — 11 fractional models
- :doc:`/models/flow/index` — 6 flow models

**Advanced theory**:

- :doc:`../03_advanced_topics/fractional_viscoelasticity_reference` — Fractional calculus

Summary
-------

RheoJAX's 20 models span three families: **classical** (exponential, simple), **fractional** (power-law, complex), and **flow** (nonlinear shear). Classical models are best for simple materials with single relaxation times. Fractional models capture broad relaxation spectra in complex materials. Flow models describe shear-dependent viscosity in nonlinear regime. Always start simple and add complexity only when necessary.

Next Steps
----------

Proceed to: :doc:`model_selection`

Learn to use decision flowcharts and compatibility checking to choose the right model for your data.
