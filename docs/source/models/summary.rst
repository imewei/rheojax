Models Summary & Selection Guide
==================================

This page serves as a comprehensive quick-reference guide for all 23 rheological models in RheoJAX. Use the comparison matrices and decision flowcharts below to select the appropriate model for your experimental data and material system.

.. contents:: Page Contents
   :local:
   :depth: 2


Complete Model Comparison Matrix
---------------------------------

The table below provides a comprehensive overview of all models across key characteristics for rapid model selection.

.. list-table:: Comprehensive Model Comparison
   :header-rows: 1
   :widths: 16 10 6 12 12 10 8 8 18
   :class: longtable

   * - Model
     - Family
     - Params
     - Test Modes
     - Material Type
     - Equilibrium Modulus
     - Complexity
     - α Range
     - Best For
   * - :doc:`Maxwell </models/classical/maxwell>`
     - Classical
     - 2
     - R, C, O, Rot
     - Liquid
     - No (G∞=0)
     - ★☆☆☆☆
     - N/A
     - Simple viscoelastic liquids, polymer melts with single relaxation
   * - :doc:`Zener </models/classical/zener>`
     - Classical
     - 3
     - R, C, O
     - Solid
     - Yes (Ge>0)
     - ★★☆☆☆
     - N/A
     - Soft solids, elastomers with exponential relaxation
   * - :doc:`SpringPot </models/classical/springpot>`
     - Fractional
     - 2
     - R, O
     - Gel
     - No
     - ★★☆☆☆
     - 0-1
     - Power-law gels, critical gels (Scott-Blair element)
   * - :doc:`Fractional Maxwell Gel </models/fractional/fractional_maxwell_gel>`
     - Fractional
     - 3
     - R, C, O
     - Gel
     - No
     - ★★★☆☆
     - 0-1
     - Gels with elastic plateau + power-law tail
   * - :doc:`Fractional Maxwell Liquid </models/fractional/fractional_maxwell_liquid>`
     - Fractional
     - 3
     - R, C, O
     - Liquid
     - No (flows)
     - ★★★☆☆
     - 0-1
     - Liquid-like materials with fractional memory effects
   * - :doc:`Fractional Maxwell Model </models/fractional/fractional_maxwell_model>`
     - Fractional
     - 4
     - R, O
     - Variable
     - Configurable
     - ★★★★☆
     - 0-1 (two)
     - Wideband fitting, materials with multiple fractional processes
   * - :doc:`Fractional Kelvin-Voigt </models/fractional/fractional_kelvin_voigt>`
     - Fractional
     - 3-4
     - C, O
     - Solid
     - Yes
     - ★★★☆☆
     - 0-1
     - Solid-like with slow fractional relaxation, creep-dominated
   * - :doc:`Fractional Zener SL </models/fractional/fractional_zener_sl>`
     - Fractional
     - 4
     - R, C, O
     - Solid
     - Yes (Gs>0)
     - ★★★★☆
     - 0-1
     - Solid with fractional liquid leg, intermediate behavior
   * - :doc:`Fractional Zener SS </models/fractional/fractional_zener_ss>`
     - Fractional
     - 4
     - R, C, O
     - Solid
     - Yes (Ge>0)
     - ★★★★☆
     - 0-1
     - Dual elastic plateaus with fractional transition (most common)
   * - :doc:`Fractional Zener LL </models/fractional/fractional_zener_ll>`
     - Fractional
     - 4
     - R, C, O
     - Liquid
     - No
     - ★★★★☆
     - 0-1
     - Liquid-biased Zener, complex liquids with memory
   * - :doc:`Fractional KV Zener </models/fractional/fractional_kv_zener>`
     - Fractional
     - 4
     - C, O
     - Solid
     - Yes
     - ★★★★☆
     - 0-1
     - Fractional KV block in series with spring, creep applications
   * - :doc:`Fractional Burgers </models/fractional/fractional_burgers>`
     - Fractional
     - 5
     - R, C, O
     - Solid/Liquid
     - Configurable
     - ★★★★★
     - 0-1
     - Captures creep AND relaxation simultaneously, versatile
   * - :doc:`Fractional Poynting-Thomson </models/fractional/fractional_poynting_thomson>`
     - Fractional
     - 5
     - R, C, O
     - Solid
     - Yes
     - ★★★★★
     - 0-1
     - Multi-plateau solids, alternate formulation for complex behavior
   * - :doc:`Fractional Jeffreys </models/fractional/fractional_jeffreys>`
     - Fractional
     - 4
     - R, C, O
     - Liquid
     - No
     - ★★★★☆
     - 0-1
     - Liquid-like with fractional damping, two dashpots + springpot
   * - :doc:`Power Law </models/flow/power_law>`
     - Flow
     - 2
     - Rot
     - Fluid
     - N/A
     - ★☆☆☆☆
     - N/A
     - Shear-thinning/thickening fluids (Ostwald-de Waele)
   * - :doc:`Carreau </models/flow/carreau>`
     - Flow
     - 4
     - Rot
     - Fluid
     - N/A
     - ★★★☆☆
     - N/A
     - Polymer solutions with Newtonian → power-law transition
   * - :doc:`Carreau-Yasuda </models/flow/carreau_yasuda>`
     - Flow
     - 5
     - Rot
     - Fluid
     - N/A
     - ★★★★☆
     - N/A
     - Adjustable transition sharpness, concentrated polymers
   * - :doc:`Cross </models/flow/cross>`
     - Flow
     - 4
     - Rot
     - Fluid
     - N/A
     - ★★★☆☆
     - N/A
     - Alternative interpolation for polymer solutions
   * - :doc:`Herschel-Bulkley </models/flow/herschel_bulkley>`
     - Flow
     - 3
     - Rot
     - Viscoplastic
     - N/A
     - ★★☆☆☆
     - N/A
     - Yield stress fluids with power-law post-yield (gels, slurries)
   * - :doc:`Bingham </models/flow/bingham>`
     - Flow
     - 2
     - Rot
     - Viscoplastic
     - N/A
     - ★★☆☆☆
     - N/A
     - Linear viscoplastic (yield stress + constant viscosity)
   * - :doc:`SGR Conventional </models/sgr/sgr_conventional>`
     - SGR
     - 3
     - R, C, O
     - Soft Glass
     - No (flows)
     - ★★★★☆
     - x: 0.5-3
     - Foams, emulsions, pastes, colloidal suspensions (Sollich 1998)
   * - :doc:`SGR GENERIC </models/sgr/sgr_generic>`
     - SGR
     - 3
     - R, C, O
     - Soft Glass
     - No (flows)
     - ★★★★★
     - x: 0.5-3
     - Thermodynamically consistent SGR (Fuereder & Ilg 2013)

**Legend:**

* **Test Modes:** R = Relaxation, C = Creep, O = Oscillation, Rot = Rotation (steady shear)
* **Complexity:** ★☆☆☆☆ = Simplest, ★★★★★ = Most complex
* **α Range:** Fractional order range for fractional models; N/A for classical/flow models
* **Equilibrium Modulus:** Whether model predicts finite G∞ at long times (solid-like)


Model Selection Decision Flowchart
-----------------------------------

For a comprehensive decision flowchart based on your experimental data, see:
:doc:`/user_guide/model_selection`.

**Quick Selection Guide:**

.. list-table:: Quick Model Selection by Data Type
   :header-rows: 1
   :widths: 25 40 35

   * - Data Type
     - Data Characteristics
     - Recommended Models
   * - **Oscillation (G', G")**
     - Two plateaus visible
     - FZSS ★★★★☆ (most common)
   * -
     - One plateau (low-ω)
     - FML ★★★☆☆
   * -
     - Power-law (no plateaus)
     - FMG ★★★☆☆, SpringPot ★★☆☆☆
   * - **Relaxation (G(t))**
     - Exponential decay → 0
     - Maxwell ★☆☆☆☆
   * -
     - Exponential decay → plateau
     - Zener ★★☆☆☆
   * -
     - Power-law decay
     - FZSS ★★★★☆, FMG ★★★☆☆
   * - **Creep (J(t))**
     - Bounded compliance
     - Zener ★★☆☆☆, FZSS ★★★★☆
   * -
     - Unbounded compliance
     - Maxwell ★☆☆☆☆, FML ★★★☆☆
   * - **Flow (η vs γ̇)**
     - Yield stress + linear
     - Bingham ★★☆☆☆
   * -
     - Yield stress + power-law
     - Herschel-Bulkley ★★☆☆☆
   * -
     - Shear thinning (no yield)
     - Power Law ★☆☆☆☆, Carreau ★★★☆☆


Model Families Overview
------------------------

Classical Models (3 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Exponential decay/recovery, simple viscoelastic behavior, single relaxation time.

**Advantages:**

* Fewest parameters (2-3)
* Fast fitting and physically interpretable
* Well-established theory and validation
* Good for teaching and simple materials

**Limitations:**

* Cannot capture power-law behavior
* Single relaxation time unrealistic for most polymers
* Poor fit for broad relaxation spectra

**Upgrade path to fractional:**

* Maxwell → Fractional Maxwell Liquid (add fractional memory)
* Zener → Fractional Zener SS (add fractional relaxation)

**Models:**

* **Maxwell** (2 params): Simplest liquid, single relaxation
* **Zener** (3 params): Solid with equilibrium modulus
* **SpringPot** (2 params): Pure power-law element (bridge to fractional)


Fractional Models (11 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Power-law relaxation, broad relaxation spectra, non-exponential behavior, soft matter.

**Advantages:**

* Capture power-law dynamics naturally
* Fewer parameters than multi-mode Maxwell
* Physical interpretation via fractional order α
* Excellent for polymers, gels, biological materials

**Fractional order (α) interpretation:**

.. list-table:: Fractional Order Interpretation Guide
   :header-rows: 1
   :widths: 15 25 60

   * - α Value
     - Physical Meaning
     - Material Examples
   * - α → 0
     - Elastic-dominated
     - Stiff gels, crosslinked elastomers (spring-like)
   * - α ≈ 0.3-0.5
     - Balanced viscoelasticity
     - Soft gels, entangled polymers, biological tissues
   * - α ≈ 0.5
     - Critical gel
     - Gel point, sol-gel transition
   * - α → 1
     - Viscous-dominated
     - Polymer melts, concentrated solutions (dashpot-like)

**Typical α ranges by material:**

* **Soft gels:** α = 0.2 - 0.4
* **Polymer melts:** α = 0.6 - 0.9
* **Biological tissues:** α = 0.3 - 0.5
* **Emulsions:** α = 0.4 - 0.7

**Model selection within fractional family:**

* **Most common starting point:** Fractional Zener SS (FZSS) - dual plateaus, versatile
* **For liquids:** Fractional Maxwell Liquid (FML) or Fractional Zener LL
* **For gels:** Fractional Maxwell Gel (FMG) or SpringPot
* **For creep:** Fractional Kelvin-Voigt (FKV) or Fractional Burgers
* **For complex materials:** Fractional Burgers (5 params) or Fractional Maxwell Model (4 params)


Flow Models (6 models)
~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Steady shear flow, viscosity vs shear rate, non-Newtonian fluids, process design.


Soft Glassy Rheology Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Soft glassy materials (foams, emulsions, pastes, colloidal suspensions), aging systems, power-law fluids near glass transition.

**Advantages:**

* Statistical mechanics foundation (trap model)
* Single noise temperature parameter x captures material state
* Natural aging dynamics for x < 1
* Power-law rheology emerges from microscopic physics
* Bayesian inference support for uncertainty quantification

**Noise temperature (x) interpretation:**

.. list-table:: Noise Temperature Interpretation Guide
   :header-rows: 1
   :widths: 15 25 60

   * - x Value
     - Physical Meaning
     - Material Examples
   * - x < 1
     - Glass (aging)
     - Aged colloidal suspensions, dense pastes (non-ergodic)
   * - x ≈ 1
     - Glass transition
     - Critical point, rheological singularity
   * - 1 < x < 2
     - Power-law fluid
     - Foams, emulsions, soft gels (SGM regime)
   * - x ≥ 2
     - Newtonian liquid
     - Dilute suspensions, simple fluids

**Model selection within SGR family:**

* **SGR Conventional** (Sollich 1998): Standard trap model, simpler formulation
* **SGR GENERIC** (Fuereder & Ilg 2013): Thermodynamically consistent, better stability near x → 1

**Connection to SRFS Transform:**

The noise temperature x from SGR models directly relates to SRFS shift factors:
a(γ̇) ~ (γ̇)^(2-x), enabling complementary analysis of oscillatory and flow data.

**Non-Newtonian classification:**

1. **Shear-thinning (pseudoplastic):** Viscosity decreases with shear rate

   * Most common: polymer solutions, paints, food products
   * Models: Power Law (n<1), Carreau, Cross, Herschel-Bulkley (n<1)

2. **Shear-thickening (dilatant):** Viscosity increases with shear rate

   * Less common: concentrated suspensions, cornstarch
   * Models: Power Law (n>1), Herschel-Bulkley (n>1)

3. **Viscoplastic (yield stress):** Requires minimum stress to flow

   * Examples: toothpaste, gels, slurries, drilling muds
   * Models: Bingham, Herschel-Bulkley

**Industrial applications:**

.. list-table:: Flow Models by Industry
   :header-rows: 1
   :widths: 25 35 40

   * - Industry
     - Common Models
     - Typical Materials
   * - Polymer Processing
     - Carreau, Cross, Power Law
     - Polymer melts, concentrated solutions
   * - Food & Cosmetics
     - Herschel-Bulkley, Bingham
     - Ketchup, toothpaste, yogurt, creams
   * - Oil & Gas
     - Herschel-Bulkley, Power Law
     - Drilling muds, crude oil
   * - Coatings & Paints
     - Carreau, Herschel-Bulkley
     - Paints, inks, adhesives
   * - Pharmaceuticals
     - Bingham, Carreau-Yasuda
     - Suspensions, gels, ointments


Quick Selection Guide
----------------------

By Material Type
~~~~~~~~~~~~~~~~

.. list-table:: Material-to-Model Quick Reference
   :header-rows: 1
   :widths: 25 35 40

   * - Material Type
     - Recommended Models
     - Notes
   * - Polymer Melts
     - FML, FZSS, Carreau (flow)
     - α typically 0.6-0.9; use flow models for processing
   * - Soft Gels
     - FZSS, FMG, SpringPot
     - α typically 0.2-0.4; check for yield stress
   * - Elastomers
     - FZSS, Zener
     - Two plateaus common; classical may suffice
   * - Biological Tissues
     - FZSS, FML, Fractional Burgers
     - α typically 0.3-0.5; complex behavior common
   * - Emulsions/Suspensions
     - FZSS (oscillation), Herschel-Bulkley (flow)
     - Check for yield stress in flow
   * - Critical Gels
     - SpringPot, FMG
     - α ≈ 0.5; power-law across all frequencies
   * - Polymer Solutions
     - Carreau, Cross (flow); FML (oscillation)
     - Shear-thinning dominant
   * - Viscoplastic Materials
     - Bingham, Herschel-Bulkley
     - Yield stress present; toothpaste, gels, slurries
   * - Foams/Emulsions
     - SGR Conventional, SGR GENERIC
     - Soft glassy materials; x parameter captures state
   * - Colloidal Suspensions
     - SGR Conventional, FZSS
     - Aging systems (x<1) or power-law fluids (1<x<2)
   * - Pastes/Dense Suspensions
     - SGR GENERIC, Herschel-Bulkley
     - Near glass transition; use GENERIC for x→1

By Application
~~~~~~~~~~~~~~

.. list-table:: Application-Based Model Selection
   :header-rows: 1
   :widths: 20 30 25 25

   * - Application
     - Primary Goal
     - Recommended Models
     - Complexity
   * - Research
     - Physical insight, publication
     - Fractional models (FZSS, FML, Burgers)
     - ★★★★☆
   * - Industrial QC
     - Fast screening, reproducibility
     - Maxwell, Zener, Power Law, Bingham
     - ★★☆☆☆
   * - Process Design
     - Predict flow behavior
     - Carreau, Herschel-Bulkley, Cross
     - ★★★☆☆
   * - Material Development
     - Structure-property relationships
     - Fractional models, multi-technique
     - ★★★★★
   * - Teaching
     - Conceptual understanding
     - Maxwell, Zener, Power Law
     - ★☆☆☆☆

By Data Quality
~~~~~~~~~~~~~~~

.. list-table:: Data Quality Considerations
   :header-rows: 1
   :widths: 20 35 45

   * - Data Characteristics
     - Model Recommendation
     - Rationale
   * - Limited data (<20 points)
     - 2-3 parameter models (Maxwell, Zener, Power Law)
     - Avoid overfitting with simpler models
   * - Moderate data (20-50 points)
     - 3-4 parameter models (FZSS, FML, Carreau)
     - Balanced complexity and fit quality
   * - Extensive data (>50 points)
     - Complex models (Burgers, Carreau-Yasuda, FMM)
     - Sufficient data to constrain 5+ parameters
   * - High noise
     - Classical models first
     - Fractional models sensitive to noise; pre-smooth data
   * - Narrow frequency range
     - Avoid multi-parameter models
     - Limited information → simpler models
   * - Multi-technique data
     - Advanced fractional models
     - Combined relaxation + oscillation → Burgers, FZSS


Parameter Count Comparison
---------------------------

**2-Parameter Models (Simplest):**

* **Maxwell**: G₀, η - Liquid with single relaxation
* **PowerLaw**: K, n - Shear-thinning/thickening
* **Bingham**: τ₀, ηpl - Linear viscoplastic
* **SpringPot**: V, α - Pure power-law element

**3-Parameter Models:**

* **Zener**: Gs, Gp, ηp - Classical solid with plateau
* **FML**: V, α, η - Fractional liquid
* **FMG**: Gs, V, α - Fractional gel
* **Herschel-Bulkley**: τ₀, K, n - Yield + power-law

**4-Parameter Models:**

* **FZSS**: Ge, Gm, α, τα - Most common fractional solid
* **FZSL**: Gs, ηs, V, α - Fractional solid-liquid Zener
* **FZLL**: ηs, ηp, V, α - Fractional liquid-liquid Zener
* **FKV**: Gp, V, α, (ηp optional) - Fractional Kelvin-Voigt
* **Carreau**: η₀, η∞, λ, n - Flow with plateaus
* **Cross**: K, m, η₀, η∞ - Alternative flow interpolation
* **Fractional Maxwell Model**: V₁, V₂, α₁, α₂ - Dual springpots
* **Fractional Jeffreys**: Two dashpots + springpot parameters

**5-Parameter Models (Most Complex):**

* **Fractional Burgers**: Maxwell + FKV (5 params) - Creep + relaxation
* **Fractional Poynting-Thomson**: Multi-plateau solid (5 params)
* **Carreau-Yasuda**: η₀, η∞, λ, n, a - Adjustable transition


Bayesian Inference Support
---------------------------

**All 23 models support complete Bayesian workflows** via NumPyro NUTS sampling:

* `.fit()` - Fast NLSQ point estimation
* `.fit_bayesian()` - Full posterior sampling with MCMC
* `.sample_prior()` - Prior predictive checks
* `.get_credible_intervals()` - Uncertainty quantification

**Recommended workflow:** NLSQ → NUTS warm-start for 2-5x faster convergence.

See :doc:`/user_guide/bayesian_inference` for comprehensive Bayesian analysis guide.


Next Steps
----------

* **Detailed model documentation:** See :doc:`/models/index` for individual model handbooks
* **Multi-technique fitting:** :doc:`/user_guide/multi_technique_fitting`
* **Model selection workflow:** :doc:`/user_guide/model_selection`
* **Compatibility checking:** :doc:`/user_guide/core_concepts` (automatic detection of model-data mismatches)
* **SGR models:** :doc:`/models/sgr/sgr_conventional` and :doc:`/models/sgr/sgr_generic`
* **SRFS transform:** :doc:`/transforms/srfs` for strain-rate frequency superposition
* **Example notebooks:** 27 examples in ``examples/`` directory

**Need a model not listed?** Open an issue via :doc:`/developer/contributing`.
