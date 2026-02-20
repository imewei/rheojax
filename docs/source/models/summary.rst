Models Summary & Selection Guide
==================================

This page serves as a comprehensive quick-reference guide for all **53 rheological models** in RheoJAX. Use the comparison matrices and decision flowcharts below to select the appropriate model for your experimental data and material system.


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
     - :math:`\alpha` Range
     - Best For
   * - :doc:`Maxwell </models/classical/maxwell>`
     - Classical
     - 2
     - R, C, O, Rot
     - Liquid
     - No (:math:`G_\infty = 0`)
     - ★☆☆☆☆
     - N/A
     - Simple viscoelastic liquids, polymer melts with single relaxation
   * - :doc:`Zener </models/classical/zener>`
     - Classical
     - 3
     - R, C, O
     - Solid
     - Yes (:math:`G_e > 0`)
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
   * - :doc:`Generalized Maxwell </models/multi_mode/generalized_maxwell>`
     - Multi-Mode
     - 2N+1
     - R, C, O
     - Variable
     - Configurable
     - ★★★★☆
     - N/A
     - Prony series, broadband fitting, industrial master curves
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
     - Yes (:math:`G_s > 0`)
     - ★★★★☆
     - 0-1
     - Solid with fractional liquid leg, intermediate behavior
   * - :doc:`Fractional Zener SS </models/fractional/fractional_zener_ss>`
     - Fractional
     - 4
     - R, C, O
     - Solid
     - Yes (:math:`G_e > 0`)
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
   * - :doc:`Giesekus Single Mode </models/giesekus/giesekus>`
     - Giesekus
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Polymer
     - No
     - ★★★★☆
     - :math:`\alpha`: 0-0.5
     - Nonlinear viscoelastic with shear thinning, :math:`N_1, N_2` predictions
   * - :doc:`Giesekus Multi Mode </models/giesekus/giesekus>`
     - Giesekus
     - 4N
     - O, Flow, Startup
     - Polymer
     - No
     - ★★★★★
     - :math:`\alpha_i`: 0-0.5
     - Multi-mode nonlinear viscoelastic, broadband spectra with normal stresses
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
     - R, C, O, Flow, Startup, LAOS
     - Soft Glass
     - No (flows)
     - ★★★★★
     - x: 0.5-3
     - Thermodynamically consistent SGR (Fuereder & Ilg 2013)
   * - :doc:`Fluidity Local </models/fluidity/fluidity_local>`
     - Fluidity
     - 2-3
     - O, Flow
     - Cooperative
     - No
     - ★★★☆☆
     - N/A
     - Local fluidity dynamics, simple cooperative flow
   * - :doc:`Fluidity Nonlocal </models/fluidity/fluidity_nonlocal>`
     - Fluidity
     - 3-4
     - O, Flow
     - Cooperative
     - No
     - ★★★★☆
     - N/A
     - Nonlocal fluidity with cooperativity length
   * - :doc:`Fluidity-Saramito Local </models/fluidity/saramito_evp>`
     - Saramito EVP
     - 10-12
     - Flow, Startup, Creep, R, O, LAOS
     - EVP Thixotropic
     - Configurable
     - ★★★★★
     - N/A
     - Tensorial EVP with fluidity coupling, :math:`N_1` predictions
   * - :doc:`Fluidity-Saramito Nonlocal </models/fluidity/saramito_evp>`
     - Saramito EVP
     - 11-13
     - Flow, Startup, Creep, R, O, LAOS
     - EVP Thixotropic
     - Configurable
     - ★★★★★
     - N/A
     - Nonlocal EVP for shear banding, cooperativity length
   * - :doc:`Lattice EPM </models/epm/lattice_epm>`
     - EPM
     - 4+
     - R, C, Startup, Flow
     - Elasto-plastic
     - Configurable
     - ★★★★★
     - N/A
     - Lattice elasto-plastic model, plastic rearrangements
   * - :doc:`Tensorial EPM </models/epm/tensorial_epm>`
     - EPM
     - 4+
     - R, C, Startup, Flow
     - Elasto-plastic
     - Configurable
     - ★★★★★
     - N/A
     - Full tensorial EPM for complex loading
   * - :doc:`MIKH </models/ikh/mikh>`
     - IKH
     - 4-5
     - R, C, O
     - Thixotropic
     - Configurable
     - ★★★★☆
     - N/A
     - Modified IKH for thixotropic materials
   * - :doc:`MLIKH </models/ikh/ml_ikh>`
     - IKH
     - 4+
     - R, C, O
     - Thixotropic
     - Configurable
     - ★★★★★
     - N/A
     - ML-enhanced IKH with neural network augmentation
   * - :doc:`FIKH </models/fikh/fikh>`
     - FIKH
     - 5-6
     - R, C, O, Flow, Startup, LAOS
     - Thixotropic
     - Configurable
     - ★★★★★
     - :math:`\alpha`: 0-1
     - Fractional IKH with Caputo structure kinetics
   * - :doc:`FMLIKH </models/fikh/fmlikh>`
     - FIKH
     - 6+
     - R, C, O, Flow, Startup, LAOS
     - Thixotropic
     - Configurable
     - ★★★★★
     - :math:`\alpha`: 0-1
     - Fractional multi-layer IKH, multiple yield surfaces
   * - :doc:`DMT Local </models/dmt/dmt>`
     - DMT
     - 5-7
     - R, C, O, Flow, Startup, LAOS
     - Thixotropic
     - Configurable
     - ★★★★☆
     - N/A
     - Structural kinetics with exponential or HB viscosity closure
   * - :doc:`DMT Nonlocal </models/dmt/dmt>`
     - DMT
     - 6-8
     - R, C, O, Flow, Startup, LAOS
     - Thixotropic
     - Configurable
     - ★★★★★
     - N/A
     - Spatially-resolved thixotropy with structure diffusion, shear banding
   * - :doc:`Hébraud-Lequeux </models/hl/hebraud_lequeux>`
     - HL
     - 3-4
     - R, C, O, Flow, Startup, LAOS
     - Soft matter
     - No
     - ★★★★☆
     - N/A
     - Mean-field model for soft glassy materials
   * - :doc:`STZ Conventional </models/stz/stz_conventional>`
     - STZ
     - 4+
     - R, O, Flow, Startup
     - Amorphous
     - No
     - ★★★★★
     - N/A
     - Shear transformation zone model (Falk-Langer)
   * - :doc:`ITT-MCT Schematic </models/itt_mct/itt_mct_schematic>`
     - ITT-MCT
     - 6
     - R, C, O, Flow, Startup, LAOS
     - Colloidal Glass
     - Configurable
     - ★★★★★
     - :math:`\varepsilon`: -0.5 to 0.5
     - Dense colloidal suspensions, glass transition (:math:`F_{12}` schematic)
   * - :doc:`ITT-MCT Isotropic </models/itt_mct/itt_mct_isotropic>`
     - ITT-MCT
     - 5+
     - R, C, O, Flow, Startup, LAOS
     - Colloidal Glass
     - Configurable
     - ★★★★★
     - :math:`\phi`: 0.1 to 0.64
     - Hard-sphere colloids with S(k), full MCT physics
   * - :doc:`SPP Yield Stress </models/spp/spp_yield_stress>`
     - SPP
     - 3+
     - LAOS
     - Yield stress
     - Yes
     - ★★★★☆
     - N/A
     - LAOS-based yield stress analysis (Rogers et al.)

   * - :doc:`TNT Tanaka-Edwards </models/tnt/tnt_tanaka_edwards>`
     - TNT
     - 3
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★☆☆☆
     - N/A
     - Baseline transient network (Maxwell via conformation tensor)
   * - :doc:`TNT Bell </models/tnt/tnt_bell>`
     - TNT
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★★☆☆
     - :math:`\nu`: 0.01-20
     - Force-dependent bond breakage, shear-thinning networks
   * - :doc:`TNT FENE-P </models/tnt/tnt_fene_p>`
     - TNT
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★★☆☆
     - :math:`L_{max}`: 2-100
     - Finite extensibility, strain hardening at large deformations
   * - :doc:`TNT Non-Affine </models/tnt/tnt_non_affine>`
     - TNT
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★★☆☆
     - :math:`\xi`: 0-1
     - Non-affine chain slip, non-zero :math:`N_2`
   * - :doc:`TNT Stretch-Creation </models/tnt/tnt_stretch_creation>`
     - TNT
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★★☆☆
     - :math:`\kappa`: 0-5
     - Flow-enhanced bond formation, shear thickening
   * - :doc:`TNT Loop-Bridge </models/tnt/tnt_loop_bridge>`
     - TNT
     - 6
     - R, C, O, Flow, Startup, LAOS
     - Telechelic
     - No
     - ★★★★☆
     - N/A
     - Two-species kinetics (loops + bridges), telechelic polymers
   * - :doc:`TNT Sticky Rouse </models/tnt/tnt_sticky_rouse>`
     - TNT
     - 4-6
     - R, C, O, Flow, Startup, LAOS
     - Multi-sticker
     - No
     - ★★★★☆
     - N/A
     - Multi-mode sticker dynamics, broad relaxation spectrum
   * - :doc:`TNT Cates </models/tnt/tnt_cates>`
     - TNT
     - 4
     - R, C, O, Flow, Startup, LAOS
     - Micelles
     - No
     - ★★★☆☆
     - N/A
     - Living polymers, wormlike micelles (:math:`\tau_d = \sqrt{\tau_{rep} \cdot \tau_{break}}`)
   * - :doc:`TNT Multi-Species </models/tnt/tnt_multi_species>`
     - TNT
     - 2N+1
     - R, C, O, Flow, Startup, LAOS
     - Mixed Network
     - No
     - ★★★★☆
     - N/A
     - Heterogeneous networks with multiple bond types

   * - :doc:`VLB Local </models/vlb/vlb>`
     - VLB
     - 2
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★☆☆☆☆
     - N/A
     - Single transient network (Maxwell via distribution tensor)
   * - :doc:`VLB Multi-Network </models/vlb/vlb>`
     - VLB
     - 2N+1
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - Configurable
     - ★★★☆☆
     - N/A
     - Multi-network generalized Maxwell with molecular basis
   * - :doc:`VLB Variant </models/vlb/vlb_variant>`
     - VLB
     - 2-6
     - R, C, O, Flow, Startup, LAOS
     - Assoc. Polymer
     - No
     - ★★★☆☆
     - N/A
     - Bell shear thinning, FENE bounded extension, Arrhenius temperature
   * - :doc:`VLB Nonlocal </models/vlb/vlb_nonlocal>`
     - VLB
     - 4-6
     - Flow, Startup, Creep
     - Assoc. Polymer
     - No
     - ★★★★☆
     - N/A
     - Spatially-resolved shear banding with tensor diffusion

   * - :doc:`HVM Local </models/hvm/hvm>`
     - HVM
     - 6-10
     - R, C, O, Flow, Startup, LAOS
     - Vitrimer
     - Yes (:math:`G_P`)
     - ★★★★★
     - N/A
     - Hybrid vitrimer: permanent + exchangeable (BER/TST) + dissociative networks

   * - :doc:`HVNM Local </models/hvnm/hvnm>`
     - HVNM
     - 13-25
     - R, C, O, Flow, Startup, LAOS
     - Filled Vitrimer
     - Yes (:math:`G_P X`)
     - ★★★★★
     - N/A
     - NP-filled vitrimer: 4 subnetworks, dual TST, Guth-Gold amplification

**Legend:**

* **Test Modes:** R = Relaxation, C = Creep, O = Oscillation, Rot = Rotation (steady shear), Flow = Flow curve, Startup = Startup shear, LAOS = Large-amplitude oscillatory
* **Complexity:** ★☆☆☆☆ = Simplest, ★★★★★ = Most complex
* :math:`\alpha` **Range:** Fractional order range for fractional models; for ITT-MCT: :math:`\varepsilon` = separation parameter (glass transition), :math:`\phi` = volume fraction; N/A for non-fractional models
* **Equilibrium Modulus:** Whether model predicts finite :math:`G_\infty` at long times (solid-like)


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
   * - **Oscillation** (:math:`G'`, :math:`G''`)
     - Two plateaus visible
     - FZSS ★★★★☆ (most common)
   * -
     - One plateau (low-:math:`\omega`)
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
   * - **Flow (** :math:`\eta vs \dot{\gamma}` **)**
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
* Physical interpretation via fractional order :math:`\alpha`
* Excellent for polymers, gels, biological materials

**Fractional order (** :math:`\alpha` **) interpretation:**

.. list-table:: Fractional Order Interpretation Guide
   :header-rows: 1
   :widths: 15 25 60

   * - :math:`\alpha` Value
     - Physical Meaning
     - Material Examples
   * - :math:`\alpha \to 0`
     - Elastic-dominated
     - Stiff gels, crosslinked elastomers (spring-like)
   * - :math:`\alpha \approx 0.3\text{--}0.5`
     - Balanced viscoelasticity
     - Soft gels, entangled polymers, biological tissues
   * - :math:`\alpha \approx 0.5`
     - Critical gel
     - Gel point, sol-gel transition
   * - :math:`\alpha \to 1`
     - Viscous-dominated
     - Polymer melts, concentrated solutions (dashpot-like)

**Typical** :math:`\alpha` **ranges by material:**

* **Soft gels:** :math:`\alpha` = 0.2 - 0.4
* **Polymer melts:** :math:`\alpha` = 0.6 - 0.9
* **Biological tissues:** :math:`\alpha` = 0.3 - 0.5
* **Emulsions:** :math:`\alpha` = 0.4 - 0.7

**Model selection within fractional family:**

* **Most common starting point:** Fractional Zener SS (FZSS) - dual plateaus, versatile
* **For liquids:** Fractional Maxwell Liquid (FML) or Fractional Zener LL
* **For gels:** Fractional Maxwell Gel (FMG) or SpringPot
* **For creep:** Fractional Kelvin-Voigt (FKV) or Fractional Burgers
* **For complex materials:** Fractional Burgers (5 params) or Fractional Maxwell Model (4 params)


Generalized Maxwell (Multi-Mode) (1 model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Prony-series fitting of broadband relaxation or oscillatory data, industrial master curve analysis, when no single relaxation time captures the spectrum.

**Advantages:**

* Systematically covers broad relaxation spectra via N Maxwell modes
* Automatic mode reduction via ``optimization_factor`` — starts from N modes and prunes unnecessary ones
* Directly connects to Prony series widely used in industry
* Supports relaxation, creep, and oscillation protocols
* JIT-compiled element search for fast multi-start optimization

**Model selection:**

* **GeneralizedMaxwell (N=2–3)**: Quick broadband fit, moderate complexity
* **GeneralizedMaxwell (N=5–10)**: Publication-quality master curve decomposition
* **GeneralizedMaxwell (optimization_factor=1.5)**: Auto-reduce from N=10 to optimal mode count

**Key physics:**

* Parallel Maxwell elements: :math:`G(t) = G_e + \sum_{i=1}^N G_i \exp(-t/\tau_i)`
* Oscillation: :math:`G'(\omega) = G_e + \sum G_i \frac{\omega^2 \tau_i^2}{1 + \omega^2 \tau_i^2}`
* Element search warm-starts from N+1, re-uses JIT compilation (2-5x speedup)

**Typical applications:** Polymer master curves, broadband industrial QC, relaxation spectra decomposition, viscoelastic material databases.


Flow Models (6 models)
~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Steady shear flow, viscosity vs shear rate, non-Newtonian fluids, process design.


Giesekus Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Polymer melts and solutions exhibiting shear thinning, nonlinear normal stress differences, stress overshoot in startup, and LAOS response. Ideal when both :math:`N_1` and :math:`N_2` predictions are required.

**Advantages:**

* Quadratic stress term gives physically motivated shear thinning
* Predicts both :math:`N_1 > 0` and :math:`N_2 < 0` with fixed ratio :math:`N_2/N_1 = -\alpha/2`
* Mobility factor :math:`\alpha` directly measurable from normal stress ratio
* ODE-based: full support for flow curve, SAOS, startup, relaxation, creep, LAOS
* Multi-mode variant for broadband spectra with mode-dependent :math:`\alpha_i`

**Mobility factor (** :math:`\alpha` **) interpretation:**

.. list-table:: Giesekus Mobility Factor Guide
   :header-rows: 1
   :widths: 15 25 60

   * - :math:`\alpha` Value
     - Physical Meaning
     - Material Examples
   * - :math:`\alpha = 0`
     - UCM limit (no shear thinning)
     - Dilute polymer solutions, Boger fluids
   * - :math:`\alpha \approx 0.1\text{--}0.3`
     - Moderate shear thinning
     - Polymer melts, semidilute solutions
   * - :math:`\alpha \approx 0.5`
     - Maximum anisotropy
     - Strongly shear-thinning polymer melts

**Model selection within Giesekus family:**

* **GiesekusSingleMode**: 4 params (:math:`\eta_p, \lambda, \alpha, \eta_s`), single relaxation time, all 6 protocols
* **GiesekusMultiMode**: N modes with independent :math:`\alpha_i`, broadband spectra, flow curve + SAOS + startup

**Key physics:**

* Constitutive equation: :math:`\boldsymbol{\tau} + \lambda \overset{\nabla}{\boldsymbol{\tau}} + \frac{\alpha \lambda}{\eta_p} \boldsymbol{\tau} \cdot \boldsymbol{\tau} = 2\eta_p \mathbf{D}`
* Conformation tensor: :math:`\mathbf{c} = \mathbf{I} + (\lambda/\eta_p)\boldsymbol{\tau}`, quadratic term drives anisotropic relaxation
* Analytical flow curve: :math:`\eta(\dot{\gamma})` from cubic equation at steady state
* Cox-Merz rule: :math:`\eta(\dot{\gamma}) \approx |\eta^*(\omega)|` for moderate :math:`\alpha`

**Typical applications:** Polymer melts (PE, PP, PS), concentrated solutions, wormlike micelles, liquid crystals, any system needing :math:`N_1, N_2` predictions.


Fluidity Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Thixotropic yield-stress fluids, materials with time-dependent viscosity, fluidity-based structure kinetics, shear banding via cooperative diffusion.

**Advantages:**

* Scalar fluidity parameter :math:`f` tracks microstructural state
* Coupled aging–rejuvenation kinetics for thixotropy
* Simple yet effective: connects naturally to soft glassy rheology
* Nonlocal variant adds cooperativity length for shear banding resolution
* Supports flow curve, startup, creep, and LAOS protocols

**Model selection within Fluidity family:**

* **FluidityLocal**: Homogeneous flow, scalar fluidity evolution, fast fitting
* **FluidityNonlocal**: PDE-based spatially resolved flow, banding detection, cooperativity length :math:`\xi`

**Key physics:**

* Fluidity evolution: :math:`df/dt = (f_{eq} - f)/\tau_f + D_f \nabla^2 f` (nonlocal)
* Flow rule: :math:`\sigma = \eta(f) \dot{\gamma}` with :math:`\eta = 1/f`
* Cooperativity length :math:`\xi` sets minimum shear band width

**Typical applications:** Colloidal gels, bentonite suspensions, Laponite, Carbopol, foams, soft glassy materials.


Fluidity-Saramito EVP Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Yield-stress fluids with combined elastic, viscous, and plastic behavior; thixotropic materials requiring
stress overshoot prediction; systems needing normal stress difference (:math:`N_1`) predictions; shear banding analysis.

**Advantages:**

* Full tensorial stress state: [:math:`\tau_{xx}, \tau_{yy}, \tau_{xy}`] for normal stress predictions
* Von Mises yield criterion with Herschel-Bulkley plastic flow
* Thixotropic fluidity evolution (aging + rejuvenation)
* Predicts stress overshoot in startup shear (key thixotropic signature)
* Supports 6 protocols: flow curve, startup, creep, relaxation, oscillation, LAOS
* Nonlocal variant captures shear banding via cooperativity length

**Model selection within Saramito family:**

* **FluiditySaramitoLocal (minimal)**: Simplest, :math:`\lambda` = 1/f only, homogeneous flow
* **FluiditySaramitoLocal (full)**: :math:`\tau_y(f)` coupling, aging yield stress
* **FluiditySaramitoNonlocal (minimal)**: Shear banding capable with :math:`D_f \nabla^2 f`
* **FluiditySaramitoNonlocal (full)**: Full thixotropic banding

**Key physics:**

* Upper-convected Maxwell viscoelasticity: :math:`\lambda(d\tau/dt - \mathbf{L} \cdot \tau - \tau \cdot \mathbf{L}^T) + \alpha(\tau)\tau = 2\eta_p \mathbf{D}`
* Plasticity parameter: :math:`\alpha = \max(0, 1 - \tau_y / |\tau|)` (Von Mises)
* Fluidity evolution: :math:`df/dt = (f_{\text{age}} - f)/t_a + b|\dot{\gamma}|^n(f_{\text{flow}} - f)`

**Typical applications:** Carbopol gels, cement pastes, drilling muds, mayonnaise, blood, cosmetic creams.


Soft Glassy Rheology Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Soft glassy materials (foams, emulsions, pastes, colloidal suspensions), aging systems, power-law fluids near glass transition.

**Advantages:**

* Statistical mechanics foundation (trap model)
* Single noise temperature parameter x captures material state
* Natural aging dynamics for :math:`x < 1`
* Power-law rheology emerges from microscopic physics
* Bayesian inference support for uncertainty quantification

**Noise temperature (** :math:`x` **) interpretation:**

.. list-table:: Noise Temperature Interpretation Guide
   :header-rows: 1
   :widths: 15 25 60

   * - x Value
     - Physical Meaning
     - Material Examples
   * - :math:`x < 1`
     - Glass (aging)
     - Aged colloidal suspensions, dense pastes (non-ergodic)
   * - :math:`x \approx 1`
     - Glass transition
     - Critical point, rheological singularity
   * - :math:`1 < x < 2`
     - Power-law fluid
     - Foams, emulsions, soft gels (SGM regime)
   * - :math:`x \geq 2`
     - Newtonian liquid
     - Dilute suspensions, simple fluids

**Model selection within SGR family:**

* **SGR Conventional** (Sollich 1998): Standard trap model, simpler formulation
* **SGR GENERIC** (Fuereder & Ilg 2013): Thermodynamically consistent, better stability near :math:`x \to 1`

**Connection to SRFS Transform:**

The noise temperature :math:`x` from SGR models directly relates to SRFS shift factors:
:math:`a(\dot{\gamma}) \sim \dot{\gamma}^{(2-x)}`, enabling complementary analysis of oscillatory and flow data.


ITT-MCT Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Dense colloidal suspensions near the glass transition, hard-sphere systems, microscopic rheological theory, yielding and flow of glassy materials.

**Advantages:**

* Microscopic theory based on Mode-Coupling Theory
* Quantitative predictions for hard-sphere colloids
* Captures glass transition physics (cage effect)
* Full nonlinear rheology including LAOS harmonics
* Two-time correlators for non-equilibrium response
* Strain decorrelation naturally emerges from advection

**Separation parameter (** :math:`\varepsilon` **) interpretation:**

.. list-table:: Glass Transition Parameter Guide
   :header-rows: 1
   :widths: 15 25 60

   * - :math:`\varepsilon` Value
     - Physical Meaning
     - Material Examples
   * - :math:`\varepsilon` < 0
     - Glass state
     - Dense suspensions below :math:`\phi_c`, kinetically arrested
   * - :math:`\varepsilon \approx 0`
     - Glass transition
     - Critical point, diverging relaxation time
   * - :math:`\varepsilon` > 0
     - Fluid state
     - Mobile suspensions, ergodic dynamics

**Model selection within ITT-MCT family:**

* **ITTMCTSchematic (** :math:`F_{12}` **)**: Simplified scalar correlator, 6 parameters, fast fitting
* **ITTMCTIsotropic (ISM)**: Full k-resolved correlators with S(k) input, quantitative predictions

**Key physics:**

* Memory kernel: :math:`m(\Phi) = v_1 \Phi + v_2 \Phi^2` (schematic) or k-integral (isotropic)
* Glass transition criterion: :math:`v_{2,c} = 4` (for :math:`v_1 = 0`)
* Strain decorrelation: :math:`h(\gamma) = \exp(-(\gamma/\gamma_c)^2)`
* Integration through transients (ITT) for nonlinear flow

**Typical applications:** PMMA hard-sphere colloids, silica suspensions, concentrated emulsions, microgel pastes.

**Comparison with SGR:**

* **SGR**: Phenomenological trap model, noise temperature x, simpler physics
* **ITT-MCT**: Microscopic derivation, volume fraction :math:`\phi`, full correlator dynamics
* Both capture yielding, but ITT-MCT provides quantitative predictions from structure


DMT Thixotropic Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Thixotropic materials with time-dependent rheology, stress overshoot in startup, delayed yielding, materials with structural buildup at rest.

**Advantages:**

* Scalar structure parameter :math:`\lambda \in [0, 1]` tracks microstructure
* Clear separation of buildup (aging) and breakdown (shear) kinetics
* Two viscosity closures: exponential (smooth) or Herschel-Bulkley (yield stress)
* Optional Maxwell backbone for stress overshoot and SAOS
* Nonlocal variant captures shear banding via structure diffusion

**Structure parameter (** :math:`\lambda` **) interpretation:**

.. list-table:: Structure Parameter Guide
   :header-rows: 1
   :widths: 15 25 60

   * - :math:`\lambda` Value
     - Physical Meaning
     - Material State
   * - :math:`\lambda` = 1
     - Fully structured
     - At rest (aged), maximum viscosity, colloidal network intact
   * - 0 < :math:`\lambda` < 1
     - Partially broken
     - Under shear, intermediate microstructure
   * - :math:`\lambda` = 0
     - Fully broken
     - High shear (rejuvenated), minimum viscosity, network destroyed

**Model selection within DMT family:**

* **DMTLocal (exponential)**: Smooth viscosity transition, no yield stress, simple
* **DMTLocal (herschel_bulkley)**: Explicit yield stress, structure-dependent :math:`\tau_y` and K
* **DMTLocal + elasticity**: Maxwell backbone for stress overshoot and SAOS
* **DMTNonlocal**: Shear banding via structure diffusion (:math:`D_{\lambda} \nabla^2 \lambda`)

**Key physics:**

* Structure kinetics: :math:`d\lambda/dt = (1-\lambda)/t_eq - a\lambda|\dot{\gamma}|^c/t_eq`
* Equilibrium structure: :math:`\lambda_{eq} = 1/(1 + a|\dot{\gamma}|^c)`
* Exponential viscosity: :math:`\eta(\lambda) = \eta_{\infty}(\eta_0/\eta_{\infty})^{\lambda}`
* Maxwell stress: :math:`d\sigma/dt = G\dot{\gamma} - \sigma/\theta(\lambda)`

**Typical applications:** Drilling muds, waxy crude oils, cement pastes, mayonnaise, ketchup, paints, concentrated suspensions.


Isotropic Kinematic Hardening Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Materials with yield-stress evolution under deformation history, cyclic loading with Bauschinger effect, isotropic + kinematic hardening, metal-like rheology in complex fluids.

**Advantages:**

* Combined isotropic and kinematic hardening captures evolving yield surfaces
* Multi-layer variant (MLIKH) for progressive yielding
* ODE-based: startup, creep, relaxation, oscillation, LAOS
* Strain-rate-dependent yield for soft materials

**Model selection within IKH family:**

* **MIKH**: Modified IKH with single yield surface — simpler, 6-8 parameters
* **MLIKH**: Multi-layer IKH with N yield surfaces — progressive yielding, N×3 + base parameters

**Key physics:**

* Yield function: :math:`f = |\sigma - \alpha| - (\sigma_y + R)` (kinematic + isotropic)
* Back-stress evolution: :math:`\dot{\alpha} = C \dot{\varepsilon}^p - \gamma_k \alpha |\dot{\varepsilon}^p|`
* Isotropic hardening: :math:`\dot{R} = b(Q - R) |\dot{\varepsilon}^p|`

**Typical applications:** Structured fluids under cyclic loading, waxy crude oils, soft solids, gel fracture.


Fractional IKH Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Same as IKH but with power-law memory effects; materials requiring fractional-order structure kinetics, long-time memory in yielding behavior.

**Advantages:**

* Caputo fractional derivative in structure kinetics — bridges IKH and fractional viscoelasticity
* Order :math:`\alpha \in (0, 1]` interpolates between integer (IKH) and maximally non-local memory
* Inherits all IKH protocols plus fractional relaxation spectra
* Multi-layer fractional variant (FMLIKH) for progressive yielding with memory

**Model selection within FIKH family:**

* **FIKH**: Fractional IKH with single yield surface + Caputo memory, 5-6 parameters
* **FMLIKH**: Fractional multi-layer IKH — N yield surfaces with fractional kinetics

**Key physics:**

* Fractional structure kinetics: :math:`{}^C D_t^{\alpha} \lambda = \text{aging} - \text{shear breakdown}`
* Caputo derivative :math:`{}^C D_t^{\alpha}` provides long-range temporal memory
* Reduces to integer IKH when :math:`\alpha \to 1`

**Typical applications:** Materials with long-time memory effects, thixotropic systems with power-law recovery, structured fluids under complex loading histories.


Hébraud-Lequeux Model (1 model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Dense amorphous materials (emulsions, foams, granular media) where mesoscopic rearrangement events (T1 events) control rheology; mean-field fluidity approach for amorphous solids.

**Advantages:**

* Mean-field kinetic model for mesoscopic stress redistribution
* Predicts flow curves, creep, and oscillatory response from microscopic rearrangements
* PDE-based stress probability distribution — captures heterogeneity
* Connects to SGR at the mesoscale but with explicit stress redistribution

**Key physics:**

* Stress probability distribution :math:`P(\sigma, t)` evolves via advection + diffusion + rearrangement
* Rearrangement rate: :math:`\Gamma = \Gamma_0 \Theta(|\sigma| - \sigma_c)` (above critical stress)
* Mean-field coupling: rearrangement events redistribute stress to neighbors
* Diffusion coefficient :math:`D_\sigma \propto \alpha \Gamma` from collective rearrangements

**Typical applications:** Concentrated emulsions, wet foams, colloidal glasses, granular media near jamming.


STZ Model (1 model)
~~~~~~~~~~~~~~~~~~~~~

**When to use:** Amorphous solids undergoing plastic deformation via shear transformation zones, metallic glasses, bulk metallic glass forming liquids, granular materials.

**Advantages:**

* Physical basis in localized shear transformation zones
* Temperature-dependent transition rates (Arrhenius activated)
* Captures strain rate sensitivity and rate-dependent yield stress
* ODE-based: 8 parameters, all physically interpretable
* Supports flow curve, startup, creep, and relaxation

**Key physics:**

* STZ creation/annihilation: :math:`\dot{\Lambda} = R_0 [e^{-\Delta F / k_B T} \cosh(\Omega \sigma / k_B T)]`
* Effective disorder temperature :math:`\chi` evolves with plastic work
* Strain rate: :math:`\dot{\varepsilon}^{pl} = 2 \epsilon_0 \Lambda e^{-\Delta F / k_B T} \sinh(\Omega \sigma / k_B T)`
* Steady-state flow stress is rate- and temperature-dependent

**Typical applications:** Metallic glasses, amorphous polymers below :math:`T_g`, granular shear, simulation benchmarks for amorphous plasticity.


Elasto-Plastic Models (2 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Yield-stress materials modeled as ensembles of mesoscopic elastoplastic elements; lattice-based models for heterogeneous yielding; full tensorial stress for anisotropic plasticity.

**Advantages:**

* Mesoscale ensemble approach: many elements sample the stress distribution
* Lattice variant adds spatial correlations (Eshelby-like stress propagation)
* Tensorial variant for full 3D stress state and anisotropic yield surfaces
* SAOS from element-level Maxwell response with yield threshold
* Flow curve from element statistics with configurable disorder

**Model selection within EPM family:**

* **LatticeEPM**: Lattice-based, L×L grid, Eshelby kernel, spatial correlations
* **TensorialEPM**: Full tensor, 3D stress state, anisotropic yield, off-lattice

**Key physics:**

* Element mechanics: :math:`\sigma_i = G(\gamma - \gamma_i^{pl})` with local yield :math:`\sigma_c`
* Yield criterion: :math:`|\sigma_i| > \sigma_c` triggers plastic rearrangement
* Stress redistribution: Eshelby kernel (lattice) or mean-field (tensorial)
* Disorder: :math:`\sigma_c` drawn from configurable distribution (Gaussian, Weibull)

**Typical applications:** Soft glasses, amorphous solids, yield stress fluids with heterogeneous microstructure, earthquake fault mechanics analogues.


Transient Network Theory Models (9 variants across 5 classes)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Associating polymers, physical gels, telechelic networks, wormlike micelles, living polymers, bio-networks with reversible crosslinks, any material with bond-mediated viscoelasticity.

**Advantages:**

* Molecular-level physics: conformation tensor tracks chain stretch and orientation
* Composable variants: Bell + FENE + slip can be combined in a single model
* Full protocol support: all 6 test modes (flow curve, SAOS, startup, relaxation, creep, LAOS)
* GPU-accelerated ODE integration via Diffrax with JAX JIT compilation
* Complete Bayesian inference pipeline (NLSQ → NUTS)

**Key physics:**

* Conformation tensor :math:`\mathbf{S}` evolves via upper-convected derivative + breakage
* Stress: :math:`\boldsymbol{\sigma} = G \cdot f(\mathbf{S}) + 2\eta_s \mathbf{D}`
* Bond lifetime :math:`\tau_b` can be constant (Tanaka-Edwards) or force-dependent (Bell)
* Single mode recovers Maxwell behavior; multi-mode gives broad spectra

**Model selection within TNT family:**

* **Start here:** TNTSingleMode (constant breakage) — 3 params, Maxwell-like baseline
* **Shear thinning:** TNTSingleMode(breakage="bell") — force-dependent breakage
* **Strain hardening:** TNTSingleMode(stress_type="fene") — finite extensibility
* **Telechelic networks:** TNTLoopBridge — loop-bridge population kinetics
* **Multi-sticker polymers:** TNTStickyRouse — hierarchical Rouse + sticker relaxation
* **Wormlike micelles:** TNTCates — living polymer scission/recombination
* **Heterogeneous networks:** TNTMultiSpecies — discrete relaxation spectrum

**Typical applications:** HEUR telechelics, PEG-PEO associating polymers, fibrin and collagen bio-networks, CTAB/CPCl wormlike micelles, PVA-borax gels, supramolecular polymer networks, vitrimers.


VLB Transient Network Models (4 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Associating polymers, physical gels, hydrogels, vitrimers, self-healing polymers, any material with reversible cross-links where a molecular-statistical foundation is desired.

**Advantages:**

* Molecular-statistical foundation via distribution tensor :math:`\boldsymbol{\mu}`
* All-analytical single-network predictions (2 parameters, Maxwell behavior)
* Multi-network extension for broad relaxation spectra
* Uniaxial extension predictions (Trouton ratio, extensional viscosity)
* Bell breakage for shear thinning, stress overshoot, nonlinear LAOS
* FENE-P for bounded extensional viscosity and strain hardening
* Arrhenius temperature dependence
* Nonlocal PDE for shear banding with tensor diffusion
* Full Bayesian inference pipeline (NLSQ → NUTS)

**Key physics:**

* Distribution tensor :math:`\boldsymbol{\mu} = \langle \mathbf{r}\mathbf{r} \rangle / \langle r_0^2 \rangle` from chain statistics
* Stress: :math:`\boldsymbol{\sigma} = G_0(\boldsymbol{\mu} - \mathbf{I})`
* Bond kinetics: :math:`\dot{\boldsymbol{\mu}} = k_d(\mathbf{I} - \boldsymbol{\mu}) + \mathbf{L} \cdot \boldsymbol{\mu} + \boldsymbol{\mu} \cdot \mathbf{L}^T`
* Single network recovers Maxwell; multi-network gives generalized Maxwell
* Bell breakage: :math:`k_d(\mu) = k_d^0 \exp(\nu(\lambda_c - 1))`
* FENE-P: :math:`\sigma = G_0 f(\text{tr}(\mu))(\mu - I)` with bounded extensibility
* Nonlocal PDE: :math:`+ D_\mu \nabla^2 \mu` for cooperative rearrangements

**Model selection within VLB family:**

* **Start here:** VLBLocal — 2 params (:math:`G_0, k_d`), analytical everywhere
* **Broad spectrum:** VLBMultiNetwork — N modes + optional permanent network + solvent
* **Nonlinear:** VLBVariant — Bell shear thinning, FENE bounded extension, temperature
* **Shear banding:** VLBNonlocal — spatially-resolved PDE with banding detection

**Typical applications:** PVA-borax hydrogels, boronate ester gels, vitrimers, telechelic polymers, supramolecular networks, shear-banding wormlike micelles.

**Comparison with TNT:**

* Mathematically equivalent to TNT at constant :math:`k_d` (both give Maxwell)
* VLB now has Bell + FENE-P variants (matching TNT's nonlinear extensions)
* VLB preferred for molecular extensions (Langevin, entropic :math:`k_d`)
* TNT additionally offers non-affine and loop-bridge variants


Hybrid Vitrimer Model (1 model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Vitrimers (covalent adaptable networks) with associative bond exchange, materials with permanent + exchangeable crosslinks, temperature-dependent topology rearrangement.

**Advantages:**

* 3-subnetwork architecture: permanent (P) + exchangeable vitrimer (E) + dissociative physical (D)
* Evolving natural-state tensor :math:`\mu^E_{nat}` — the vitrimer hallmark (BER rearranges topology)
* TST kinetics: stress- or stretch-activated bond exchange rates
* Arrhenius temperature dependence with topology freezing transition :math:`T_v`
* Factory methods for 5 limiting cases (neo-Hookean, Maxwell, Zener, pure/partial vitrimer)
* Full protocol support: flow curve, SAOS, startup, relaxation, creep, LAOS

**Key physics:**

* Bond exchange reaction: :math:`k_{BER} = \nu_0 \exp(-E_a/RT) \cosh(V_{act} \sigma_{VM}/RT)`
* Factor-of-2: :math:`\tau_E^{eff} = 1/(2 k_{BER,0})` — both :math:`\mu^E` and :math:`\mu^E_{nat}` relax toward each other
* Stress :math:`\sigma_E \to 0` at steady state (natural state fully tracks deformation)
* 11-component ODE state integrated via Diffrax Tsit5

**Typical applications:** Epoxy vitrimers, polyester CANs, silicone vitrimers, polyurethane vitrimers, self-healing networks.


Hybrid Vitrimer Nanocomposite Model (1 model)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Nanoparticle-filled vitrimers, nanocomposites with interfacial bond exchange, materials where filler–matrix interphase contributes distinct relaxation.

**Advantages:**

* 4-subnetwork architecture: P + E + D + interphase (I) around nanoparticles
* Guth-Gold strain amplification: :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2`
* Dual TST kinetics: independent matrix (:math:`k_{BER}^{mat}`) and interphase (:math:`k_{BER}^{int}`) exchange
* :math:`\phi = 0` recovers HVM exactly (verified to machine precision)
* Factory methods for 5 configurations: unfilled vitrimer, filled elastomer, partial NC, etc.

**Key physics:**

* Interphase reinforcement: :math:`G_I = \beta_I \cdot G_E` scales with NP surface area
* Separate Arrhenius activation for matrix and interphase exchange
* Feature flags for interfacial damage, diffusion, and degradation
* 17-18 component ODE state depending on configuration

**Typical applications:** Silica-epoxy vitrimer nanocomposites, CNT-vitrimer networks, graphene-polymer CANs, functional nanocomposites with adaptable bonds.


SPP LAOS Model (1 model)
~~~~~~~~~~~~~~~~~~~~~~~~~~

**When to use:** Large amplitude oscillatory shear (LAOS) analysis, yield stress extraction from oscillatory data, intracycle nonlinear characterization, model validation against SPP trajectories.

**Advantages:**

* Instantaneous moduli :math:`G'_t, G''_t` resolve intracycle viscoelastic transitions
* Cole-Cole trajectory reveals sequence of physical processes during nonlinear deformation
* Robust yield stress determination from trajectory features
* Model-experiment comparison via trajectory mismatch metric
* Complementary to Fourier-based LAOS (FT-Rheology)

**Key physics:**

* Instantaneous storage: :math:`G'_t = \dot{\sigma}/\dot{\gamma}` (elastic contribution)
* Instantaneous loss: :math:`G''_t = (1/\omega)(d\sigma/d\gamma)|_{\dot{\gamma}=\text{const}}` (viscous contribution)
* Cole-Cole trajectory: :math:`G'_t` vs :math:`G''_t` traces physical process sequence
* Yield identification: kink/cusp (Type I) or smooth maximum (Type II)

**Typical applications:** Yield stress fluids (Carbopol, cement), soft glasses, colloidal gels, biological hydrogels, any material requiring intracycle LAOS analysis.


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
     - Giesekus, FML, FZSS, Carreau (flow)
     - Giesekus for :math:`N_1, N_2` and startup; :math:`\alpha` typically 0.6-0.9 for fractional
   * - Soft Gels
     - FZSS, FMG, SpringPot
     - :math:`\alpha` typically 0.2-0.4; check for yield stress
   * - Elastomers
     - FZSS, Zener
     - Two plateaus common; classical may suffice
   * - Biological Tissues
     - FZSS, FML, Fractional Burgers
     - :math:`\alpha` typically 0.3-0.5; complex behavior common
   * - Emulsions/Suspensions
     - FZSS (oscillation), Herschel-Bulkley (flow)
     - Check for yield stress in flow
   * - Critical Gels
     - SpringPot, FMG
     - :math:`\alpha \approx 0.5`; power-law across all frequencies
   * - Polymer Solutions
     - Giesekus, Carreau, Cross (flow); FML (oscillation)
     - Giesekus for nonlinear + :math:`N_1`; Carreau/Cross for viscosity only
   * - Viscoplastic Materials
     - Bingham, Herschel-Bulkley
     - Yield stress present; toothpaste, gels, slurries
   * - Foams/Emulsions
     - SGR Conventional, SGR GENERIC
     - Soft glassy materials; x parameter captures state
   * - Colloidal Suspensions
     - SGR Conventional, ITTMCTSchematic, FZSS
     - Aging systems (:math:`x < 1`), hard-sphere (MCT), or power-law fluids
   * - Hard-Sphere Colloids
     - ITTMCTSchematic, ITTMCTIsotropic
     - Near glass transition; use ISM for quantitative S(k) predictions
   * - Pastes/Dense Suspensions
     - SGR GENERIC, Herschel-Bulkley
     - Near glass transition; use GENERIC for :math:`x \to 1`
   * - Thixotropic Yield Stress
     - FluiditySaramitoLocal, Herschel-Bulkley
     - Stress overshoot, aging; use Saramito for :math:`N_1`
   * - Shear Banding Materials
     - FluiditySaramitoNonlocal, FluidityNonlocal
     - Spatially resolved flow, cooperativity length
   * - Associating Polymers
     - TNTSingleMode, TNTStickyRouse
     - Reversible crosslinks; Bell variant for shear thinning
   * - Wormlike Micelles
     - TNTCates, TNTSingleMode(bell)
     - Living polymers; :math:`\tau_d = \sqrt{\tau_{rep} \cdot \tau_{break}}`
   * - Telechelic Networks
     - TNTLoopBridge, TNTSingleMode
     - Loop-bridge kinetics; end-functionalized polymers
   * - Self-Healing Gels
     - VLBLocal, VLBMultiNetwork
     - Molecular-statistical foundation; 2 params for Maxwell-like networks
   * - Vitrimers/CANs
     - HVMLocal, VLBMultiNetwork
     - Evolving natural state, BER/TST kinetics, Arrhenius :math:`k_{BER}`
   * - NP-Filled Vitrimers
     - HVNMLocal, HVMLocal (unfilled)
     - Dual TST kinetics, Guth-Gold amplification, Payne effect
   * - DMTA/DMA Specimens
     - FZSS, GeneralizedMaxwell, Zener
     - Set ``deformation_mode='tension'``; auto E*↔G* conversion

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

* **Maxwell**: :math:`G_0, \eta` - Liquid with single relaxation
* **PowerLaw**: K, n - Shear-thinning/thickening
* **Bingham**: :math:`\tau_0, \eta_{pl}` - Linear viscoplastic
* **SpringPot**: V, :math:`\alpha` - Pure power-law element

**3-Parameter Models:**

* **Zener**: Gs, Gp, :math:`\eta_p` - Classical solid with plateau
* **FML**: V, :math:`\alpha, \eta` - Fractional liquid
* **FMG**: Gs, V, :math:`\alpha` - Fractional gel
* **Herschel-Bulkley**: :math:`\tau_0`, K, n - Yield + power-law

**4-Parameter Models:**

* **FZSS**: Ge, Gm, :math:`\alpha, \tau\alpha` - Most common fractional solid
* **FZSL**: Gs, :math:`\eta_s, V, \alpha` - Fractional solid-liquid Zener
* **FZLL**: :math:`\eta_s, \eta_p, V, \alpha` - Fractional liquid-liquid Zener
* **FKV**: Gp, V, :math:`\alpha`, (:math:`\eta_p` optional) - Fractional Kelvin-Voigt
* **Carreau**: :math:`\eta_0, \eta_{\infty}, \lambda`, n - Flow with plateaus
* **Cross**: K, m, :math:`\eta_0, \eta_{\infty}` - Alternative flow interpolation
* **Fractional Maxwell Model**: :math:`V_1, V_2, \alpha_1, \alpha_2` - Dual springpots
* **Fractional Jeffreys**: Two dashpots + springpot parameters

**5-Parameter Models (Most Complex):**

* **Fractional Burgers**: Maxwell + FKV (5 params) - Creep + relaxation
* **Fractional Poynting-Thomson**: Multi-plateau solid (5 params)
* **Carreau-Yasuda**: :math:`\eta_0, \eta_{\infty}, \lambda`, n, a - Adjustable transition


Bayesian Inference Support
---------------------------

**All 53 models support complete Bayesian workflows** via NumPyro NUTS sampling:

* `.fit()` - Fast NLSQ point estimation
* `.fit_bayesian()` - Full posterior sampling with MCMC
* `.sample_prior()` - Prior predictive checks
* `.get_credible_intervals()` - Uncertainty quantification

**Recommended workflow:** NLSQ → NUTS warm-start for 2-5x faster convergence.

See :doc:`/user_guide/bayesian_inference` for comprehensive Bayesian analysis guide.


DMTA / DMA Support
-------------------

**All 49 oscillation-capable models support DMTA data** through automatic :math:`E^* \leftrightarrow G^*`
conversion at the ``BaseModel`` boundary:

* **Tensile modulus conversion:** :math:`E^* = 2(1 + \nu) G^*` applied automatically when ``deformation_mode='tension'``
* **Poisson ratio presets:** rubber (0.5), glassy polymer (0.35), semicrystalline (0.40)
* **Transparent workflow:** Model parameters stay in shear space; conversion at fit/predict boundary
* **CSV auto-detection:** Columns named ``E'``, ``E''``, or ``E*`` automatically set ``deformation_mode='tension'``

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()
   model.fit(omega, E_star, test_mode='oscillation',
             deformation_mode='tension', poisson_ratio=0.5)
   E_pred = model.predict(omega)  # Returns E* automatically

See :doc:`/models/dmta/index` for DMTA theory, model compatibility, and workflow guides.


Next Steps
----------

* **Detailed model documentation:** See :doc:`/models/index` for individual model handbooks
* **Multi-technique fitting:** :doc:`/user_guide/multi_technique_fitting`
* **Model selection workflow:** :doc:`/user_guide/model_selection`
* **Compatibility checking:** :doc:`/user_guide/core_concepts` (automatic detection of model-data mismatches)
* **Giesekus models:** :doc:`/models/giesekus/giesekus` for nonlinear viscoelastic polymer melts and solutions
* **SGR models:** :doc:`/models/sgr/sgr_conventional` and :doc:`/models/sgr/sgr_generic`
* **ITT-MCT models:** :doc:`/models/itt_mct/itt_mct_schematic` and :doc:`/models/itt_mct/itt_mct_isotropic` for colloidal glasses
* **TNT models:** :doc:`/models/tnt/index` for transient network theory (associating polymers, micelles)
* **VLB models:** :doc:`/models/vlb/index` for VLB transient networks (hydrogels, vitrimers, self-healing polymers)
* **HVM models:** :doc:`/models/hvm/index` for hybrid vitrimer constitutive models
* **HVNM models:** :doc:`/models/hvnm/index` for vitrimer nanocomposite models
* **DMTA support:** :doc:`/models/dmta/index` for tensile modulus conversion and DMA workflows
* **SRFS transform:** :doc:`/transforms/srfs` for strain-rate frequency superposition
* **Example notebooks:** 246 examples across 20+ directories in ``examples/``

**Need a model not listed?** Open an issue via :doc:`/developer/contributing`.
