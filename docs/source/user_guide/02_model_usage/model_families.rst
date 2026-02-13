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

The Model Families
-------------------

RheoJAX provides **53 rheological models** organized into families based on their theoretical foundations:

.. list-table:: Model Families Overview
   :header-rows: 1
   :widths: 20 30 50

   * - Category
     - Model Family
     - Physical Basis
   * - ``classical``
     - Maxwell, Zener, SpringPot
     - Linear viscoelastic elements
   * - ``fractional_maxwell``
     - FractionalMaxwellGel, etc.
     - Fractional calculus extensions (Maxwell-based)
   * - ``fractional_zener``
     - FractionalZenerSS, etc.
     - Fractional calculus extensions (Zener-based)
   * - ``fractional_advanced``
     - Burgers, Jeffreys, Poynting-Thomson
     - Multi-element fractional models
   * - ``flow``
     - PowerLaw, Carreau, Bingham, HB
     - Viscosity vs. shear rate
   * - ``multi_mode``
     - GeneralizedMaxwell
     - Discrete relaxation spectrum
   * - ``sgr``
     - SGRConventional, SGRGeneric
     - Trap model for amorphous materials
   * - ``fluidity``
     - FluidityLocal, FluidityNonlocal
     - Cooperative flow models
   * - ``fluidity_saramito``
     - FluiditySaramitoLocal, FluiditySaramitoNonlocal
     - Tensorial EVP with thixotropic fluidity
   * - ``epm``
     - LatticeEPM, TensorialEPM
     - Lattice/Tensorial elasto-plasticity
   * - ``ikh``
     - MIKH, MLIKH
     - Isotropic kinematic hardening (thixotropy)
   * - ``hl``
     - HebraudLequeux
     - Mean-field soft matter model
   * - ``stz``
     - STZConventional
     - Shear transformation zone theory
   * - ``spp_laos``
     - SPPYieldStress
     - Sequence of Physical Processes (LAOS)
   * - ``giesekus``
     - GiesekusSingleMode, GiesekusMultiMode
     - Nonlinear viscoelastic polymer (tensor ODE)
   * - ``fikh``
     - FIKHLocal, FMLIKHLocal
     - Fractional isotropic-kinematic hardening
   * - ``dmt``
     - DMTLocal, DMTNonlocal
     - Structure-parameter thixotropy (de Souza Mendes)
   * - ``itt_mct``
     - ITTMCTSchematic, ITTMCTIsotropic
     - Mode-coupling theory (dense suspensions)
   * - ``tnt``
     - SingleMode, Cates, LoopBridge, MultiSpecies, StickyRouse
     - Transient network theory (5 variants)
   * - ``vlb``
     - VLBLocal, MultiNetwork, Variant, Nonlocal
     - Distribution tensor network (4 variants)
   * - ``hvm``
     - HVMLocal
     - Hybrid Vitrimer Model (3 subnetworks)
   * - ``hvnm``
     - HVNMLocal
     - Vitrimer Nanocomposite (4 subnetworks)

Quick Reference Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 30

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
   * - SGR
     - SAOS, Relaxation
     - Soft glasses
     - Noise temperature :math:`x`
   * - Fluidity
     - SAOS, Flow
     - Cooperative systems
     - Spatial correlations
   * - EPM
     - Relaxation, Creep, Startup
     - Elasto-plastic solids
     - Plastic rearrangements
   * - IKH
     - SAOS, Relaxation, Creep
     - Thixotropic materials
     - Kinematic hardening
   * - HL
     - SAOS, Relaxation
     - Soft matter
     - Mean-field dynamics
   * - STZ
     - SAOS, Relaxation, Flow
     - Amorphous solids
     - Shear transformation
   * - SPP
     - LAOS
     - Yield stress fluids
     - Physical process decomposition
   * - Giesekus
     - All 6 protocols
     - Polymer solutions/melts
     - Nonlinear shear thinning + :math:`N_1`
   * - Saramito
     - All 6 protocols
     - EVP fluids
     - Tensorial yield + thixotropy
   * - FIKH
     - All 6 protocols
     - Complex thixotropic
     - Fractional memory + hardening
   * - DMT
     - All 6 protocols
     - Structured fluids
     - Structure parameter :math:`\lambda`
   * - ITT-MCT
     - All 6 protocols
     - Dense colloids, glasses
     - Memory kernel, glass transition
   * - TNT
     - All 6 protocols
     - Associative polymers
     - Network attachment/detachment
   * - VLB
     - All 6 protocols
     - Transient networks
     - Distribution tensor :math:`\boldsymbol{\mu}`
   * - HVM
     - All 6 protocols
     - Vitrimers
     - Bond exchange reactions
   * - HVNM
     - All 6 protocols
     - Filled vitrimers
     - Interphase + Guth-Gold

Family 1: Classical Viscoelastic Models
----------------------------------------

**Who they're for**: Simple polymers, dilute solutions, basic characterization

**Test modes**: SAOS, stress relaxation, creep

**Key characteristic**: **Exponential relaxation** (single or discrete timescales)

The Models
~~~~~~~~~~

1. **Maxwell** (2 parameters)

   - **Type**: Viscoelastic liquid
   - **Equation**: :math:`G(t) = G_0 \exp(-t/\tau)`
   - **Use for**: Single relaxation time, simple liquids
   - **Example**: Low molecular weight polymer melts

2. **Zener** (3 parameters)

   - **Type**: Viscoelastic solid (Standard Linear Solid)
   - **Equation**: :math:`G(t) = G_e + G_m \exp(-t/\tau)`
   - **Use for**: Materials with equilibrium modulus (crosslinked)
   - **Example**: Lightly crosslinked rubbers

3. **SpringPot** (2 parameters)

   - **Type**: Pure fractional element (bridges solid and liquid)
   - **Equation**: :math:`G(t) \sim t^{-\alpha}`
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

   from rheojax.models import Maxwell

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

**Fractional calculus** captures these distributions with a single parameter :math:`\alpha` (fractional order).

**Power-law signatures**:

- Relaxation: :math:`G(t) \sim t^{-\alpha}`
- Oscillation: :math:`G', G'' \sim \omega^\alpha` (parallel scaling)

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

- Log-log plot shows parallel :math:`G'` and :math:`G''` lines (power-law)
- Material has broad molecular weight distribution
- Relaxation is NOT purely exponential
- Classical models fail to capture curvature
- You observe gel-like behavior

❌ **Avoid fractional models when**:

- Material is simple (use classical first)
- You have very few data points (< 10)
- Single exponential fits perfectly well

The Fractional Order Parameter (:math:`\alpha`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\alpha` characterizes the breadth of the relaxation spectrum:

- :math:`\alpha \to 1`: Narrow spectrum (nearly liquid-like, approaches Maxwell)
- :math:`\alpha \approx 0.5`: Broad spectrum (gel-like, power-law)
- :math:`\alpha \to 0`: Narrow spectrum (nearly solid-like, approaches elastic)

**Typical values**:

- Monodisperse polymers: :math:`\alpha = 0.8 - 0.95`
- Polydisperse polymers: :math:`\alpha = 0.5 - 0.7`
- Gels: :math:`\alpha = 0.3 - 0.6`
- Soft glassy materials: :math:`\alpha = 0.1 - 0.3`

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

**Key characteristic**: **Nonlinear viscosity** :math:`\eta(\dot{\gamma})`

Why Flow Models?
~~~~~~~~~~~~~~~~

Most complex fluids exhibit **shear-dependent viscosity**:

- **Shear thinning**: :math:`\eta` decreases with shear rate (polymers, suspensions)
- **Shear thickening**: :math:`\eta` increases with shear rate (dense suspensions)
- **Yield stress**: Material doesn't flow below critical stress (pastes, gels)

**These are nonlinear effects** not captured by linear viscoelastic models.

The 6 Flow Models
~~~~~~~~~~~~~~~~~

**Newtonian and Shear Thinning**:

1. **PowerLaw** (2 parameters) — Simple shear thinning

   - :math:`\eta(\dot{\gamma}) = K \dot{\gamma}^{n-1}`
   - :math:`n < 1`: Shear thinning
   - :math:`n = 1`: Newtonian

2. **Carreau** (4 parameters) — Shear thinning with Newtonian plateaus

   - :math:`\eta(\dot{\gamma}) = \eta_\infty + (\eta_0 - \eta_\infty) [1 + (\lambda \dot{\gamma})^2]^{(n-1)/2}`
   - Most flexible for polymers

3. **Carreau-Yasuda** (5 parameters) — Extended Carreau

4. **Cross** (4 parameters) — Alternative to Carreau

**Yield Stress**:

5. **Bingham** (2 parameters) — Newtonian above yield stress

   - :math:`\sigma = \sigma_y + \eta_p \dot{\gamma}` (for :math:`\sigma > \sigma_y`)
   - Use for: Drilling fluids, simple pastes

6. **Herschel-Bulkley** (3 parameters) — Power-law above yield stress

   - :math:`\sigma = \sigma_y + K \dot{\gamma}^n`
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
- You need to predict elastic behavior (:math:`G'`)

Example: Fitting PowerLaw Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import PowerLaw

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

Family 4: Soft Glassy Rheology (SGR) Models
--------------------------------------------

**Who they're for**: Soft glassy materials (foams, emulsions, pastes, colloidal suspensions)

**Test modes**: SAOS, stress relaxation

**Key characteristic**: **Noise temperature** :math:`x` parameterizes material state

The SGR models are based on Sollich's trap model (1998), treating soft glassy materials
as ensembles of mesoscopic elements that hop between energy traps.

The 2 SGR Models
~~~~~~~~~~~~~~~~~

1. **SGR Conventional** (3 parameters: :math:`x, G_0, \tau_0`)

   - Original Sollich formulation
   - :math:`x < 1`: Glass (aging), :math:`x > 1`: Ergodic (flows)
   - Use for: Foams, emulsions, pastes

2. **SGR GENERIC** (3 parameters)

   - Thermodynamically consistent (Fuereder & Ilg 2013)
   - Better stability near :math:`x \to 1`
   - Use for: Systems near glass transition

**For detailed equations**: :doc:`/models/sgr/sgr_conventional`

Family 5: Fluidity Models
--------------------------

**Who they're for**: Materials exhibiting cooperative flow

**Test modes**: SAOS, steady shear flow

**Key characteristic**: **Fluidity field** describes local flow propensity

The 2 Fluidity Models
~~~~~~~~~~~~~~~~~~~~~~

1. **Fluidity Local** (2-3 parameters)

   - Local fluidity dynamics
   - Use for: Simple shear-thinning materials

2. **Fluidity Nonlocal** (3-4 parameters)

   - Includes spatial correlations via cooperativity length
   - Use for: Systems with cooperative rearrangements

**For detailed equations**: :doc:`/models/fluidity/fluidity_local`

Family 6: Elasto-Plastic Models (EPM)
--------------------------------------

**Who they're for**: Yield stress fluids, amorphous solids, metallic glasses

**Test modes**: Stress relaxation, creep, startup shear, flow curves

**Key characteristic**: **Plastic rearrangements** via local yielding events

The 2 EPM Models
~~~~~~~~~~~~~~~~~

1. **Lattice EPM** (multiple parameters)

   - Discrete lattice of mesoscopic blocks
   - Tracks stress redistribution after plastic events
   - Use for: Granular materials, dense suspensions

2. **Tensorial EPM** (multiple parameters)

   - Full tensorial stress formulation
   - Use for: Complex loading conditions

**For detailed equations**: :doc:`/models/epm/lattice_epm`

Family 7: Isotropic Kinematic Hardening (IKH)
----------------------------------------------

**Who they're for**: Thixotropic materials with structural recovery

**Test modes**: SAOS, relaxation, creep

**Key characteristic**: **Structure parameter** evolves with deformation

The 2 IKH Models
~~~~~~~~~~~~~~~~~

1. **MIKH** (Modified IKH) — Standard thixotropic model

   - Structure builds up at rest, breaks down under shear
   - Use for: Drilling muds, waxy crude oils

2. **MLIKH** (ML-enhanced IKH) — Machine learning augmented

   - Neural network-enhanced constitutive relations
   - Use for: Complex thixotropic behavior

**For detailed equations**: :doc:`/models/ikh/mikh`

Family 8: Hébraud-Lequeux (HL) Model
-------------------------------------

**Who they're for**: Soft glassy materials, dense suspensions

**Test modes**: SAOS, relaxation

**Key characteristic**: **Mean-field dynamics** for plastic flow

The HL Model (3-4 parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Mean-field approach to soft matter flow
- Describes stress distribution evolution
- Use for: Dense colloidal suspensions, soft glasses

**For detailed equations**: :doc:`/models/hl/hebraud_lequeux`

Family 9: Shear Transformation Zone (STZ) Models
-------------------------------------------------

**Who they're for**: Metallic glasses, amorphous solids

**Test modes**: SAOS, relaxation, flow

**Key characteristic**: **STZ density** evolves with deformation

The STZ Model (4+ parameters)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Based on Falk-Langer STZ theory
- Tracks shear transformation zone population
- Use for: Metallic glasses, bulk amorphous materials

**For detailed equations**: :doc:`/models/stz/stz_conventional`

Family 10: SPP Models (LAOS Analysis)
--------------------------------------

**Who they're for**: Yield stress fluids characterized via LAOS

**Test modes**: Large-amplitude oscillatory shear (LAOS)

**Key characteristic**: **Sequence of Physical Processes** decomposition

The SPP Model
~~~~~~~~~~~~~~

1. **SPP Yield Stress** — LAOS-based yield stress analysis

   - Decomposes stress into elastic storage + plastic dissipation
   - Extracts yield stress from oscillatory data
   - Use for: Gels, pastes, structured fluids

**For detailed equations**: :doc:`/models/spp/spp_yield_stress`

Family 11: Giesekus Models
---------------------------

**Who they're for**: Polymer solutions and melts exhibiting shear thinning and normal stresses

**Test modes**: All 6 (flow curve, SAOS, startup, relaxation, creep, LAOS)

**Key characteristic**: **Nonlinear tensor ODE** with mobility parameter :math:`\alpha`

The 2 Giesekus Models
~~~~~~~~~~~~~~~~~~~~~~

1. **Giesekus Single-Mode** (3 parameters: :math:`G, \lambda_1, \alpha`)

   - Upper-convected Maxwell + anisotropic drag (:math:`\alpha` controls nonlinearity)
   - :math:`\alpha = 0`: recovers upper-convected Maxwell; :math:`\alpha = 0.5`: maximum shear thinning
   - Predicts first and second normal stress differences (:math:`N_1, N_2`)
   - Use for: Polymer solutions, dilute/semi-dilute systems

2. **Giesekus Multi-Mode** (:math:`3N` parameters)

   - N parallel Giesekus elements with solvent viscosity
   - Use for: Polydisperse polymers, entangled melts

**For detailed equations**: :doc:`/models/giesekus/index`

**For tutorial**: :doc:`../03_advanced_topics/constitutive_ode_models`

Family 12: Fluidity-Saramito EVP Models
-----------------------------------------

**Who they're for**: Elastoviscoplastic materials with thixotropy

**Test modes**: All 6 protocols

**Key characteristic**: **Tensorial stress** with Von Mises yield + thixotropic fluidity evolution

The 2 Saramito-Fluidity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fluidity-Saramito Local** (7-8 parameters)

   - Tensorial stress :math:`[\tau_{xx}, \tau_{yy}, \tau_{xy}]` with Von Mises yielding :math:`\alpha = \max(0, 1 - \tau_y/|\tau|)`
   - Coupling modes: "minimal" (:math:`\lambda = 1/f`) or "full" (:math:`\lambda + \tau_y(f)` aging yield)
   - Predicts normal stresses :math:`N_1`, stress overshoot, creep bifurcation
   - Use for: Yield stress fluids, carbopol gels, soft solids

2. **Fluidity-Saramito Nonlocal** (8-9 parameters + n_points)

   - Includes spatial cooperativity for shear banding
   - Use for: Systems showing heterogeneous flow

**For detailed equations**: :doc:`/models/fluidity/saramito_evp`

**For tutorial**: :doc:`../03_advanced_topics/constitutive_ode_models`

Family 13: FIKH (Fractional IKH) Models
-----------------------------------------

**Who they're for**: Complex thixotropic materials with long-term memory

**Test modes**: All 6 protocols

**Key characteristic**: **Fractional derivatives** + kinematic hardening for power-law memory

The 2 FIKH Models
~~~~~~~~~~~~~~~~~

1. **FIKH Local** — Fractional IKH

   - Combines structure-parameter thixotropy with fractional Caputo derivatives
   - Power-law memory captures long-time relaxation tails
   - Use for: Waxy crude oils, drilling muds with complex aging

2. **FMLIKH Local** — Fractional Modified Leonov IKH

   - Extended fractional variant with Leonov constitutive equation
   - Use for: Highly nonlinear thixotropic materials

**For detailed equations**: :doc:`/models/fikh/fmlikh`

**For tutorial**: :doc:`../03_advanced_topics/constitutive_ode_models`

Family 14: DMT (de Souza Mendes-Thompson) Models
--------------------------------------------------

**Who they're for**: Thixotropic materials with structure buildup and breakdown

**Test modes**: All 6 protocols

**Key characteristic**: **Structure parameter** :math:`\lambda \in [0, 1]` with viscosity closure

The 2 DMT Models
~~~~~~~~~~~~~~~~~

1. **DMT Local** (5-7 parameters)

   - Structure kinetics: :math:`d\lambda/dt = (1-\lambda)/t_{eq} - a\lambda|\dot{\gamma}|^c/t_{eq}`
   - Two closures: "exponential" (smooth) or "herschel_bulkley" (explicit yield)
   - Optional Maxwell elasticity for stress overshoot and SAOS
   - Use for: Drilling muds, waxy crude oils, food products

2. **DMT Nonlocal** (6-8 parameters + n_points)

   - Structure diffusion :math:`D_\lambda \nabla^2 \lambda` for spatial heterogeneity
   - Use for: Shear-banding thixotropic materials

**For detailed equations**: :doc:`/models/dmt/index`

**For tutorial**: :doc:`../03_advanced_topics/thixotropy_yielding`

Family 15: ITT-MCT (Mode-Coupling Theory) Models
--------------------------------------------------

**Who they're for**: Dense colloidal suspensions and glass-forming systems

**Test modes**: All 6 protocols

**Key characteristic**: **Memory kernel** from microscopic pair correlations; glass transition

The 2 ITT-MCT Models
~~~~~~~~~~~~~~~~~~~~~

1. **ITT-MCT Schematic** (:math:`F_{12}`) (5-6 parameters)

   - Memory kernel :math:`m(\Phi) = v_1 \Phi + v_2 \Phi^2` with strain decorrelation
   - Glass transition at :math:`v_2 = 4`: :math:`\epsilon = (v_2 - 4)/4`
   - Semi-quantitative; first JIT compile takes 30-90s
   - Use for: Generic glass-forming systems, fast exploration

2. **ITT-MCT Isotropic** (4-5 parameters)

   - Uses Percus-Yevick structure factor :math:`S(k)` for quantitative predictions
   - Volume fraction :math:`\varphi` as primary control parameter
   - Use for: Hard-sphere colloids, quantitative comparison with experiments

**For detailed equations**: :doc:`/models/itt_mct/index`

**For tutorial**: :doc:`../03_advanced_topics/dense_suspensions_glasses`

Family 16: TNT (Transient Network Theory) Models
--------------------------------------------------

**Who they're for**: Associative polymers, wormlike micelles, telechelic polymers, biological gels

**Test modes**: All 6 protocols

**Key characteristic**: **Dynamic crosslinks** with attachment/detachment kinetics

The 5 TNT Models
~~~~~~~~~~~~~~~~~

1. **TNT SingleMode** (3 params) — Simplest transient network
2. **TNT Cates** (3-4 params) — Living polymers (wormlike micelles)
3. **TNT LoopBridge** (4-5 params) — Telechelic loop↔bridge dynamics
4. **TNT MultiSpecies** (:math:`2N+1` params) — Multiple chain populations
5. **TNT StickyRouse** (4-5 params) — Rouse chains with sticky associations

**For detailed equations**: :doc:`/models/tnt/index`

**For tutorial**: :doc:`../03_advanced_topics/polymer_network_models`

Family 17: VLB (Vernerey-Long-Brighenti) Models
-------------------------------------------------

**Who they're for**: Transient polymer networks with distribution tensor formulation

**Test modes**: All 6 protocols

**Key characteristic**: **Chain distribution tensor** :math:`\boldsymbol{\mu}` tracks end-to-end vector statistics

The 4 VLB Models
~~~~~~~~~~~~~~~~~

1. **VLB Local** (3-4 params) — Basic distribution tensor network
2. **VLB MultiNetwork** (:math:`3N` params) — Multiple interacting networks
3. **VLB Variant** (5-6 params) — Bell force sensitivity + FENE extensibility
4. **VLB Nonlocal** (4-5 params + n_points) — PDE for shear banding

**For detailed equations**: :doc:`/models/vlb/index`

**For tutorial**: :doc:`../03_advanced_topics/polymer_network_models`

Family 18: HVM (Hybrid Vitrimer Model)
----------------------------------------

**Who they're for**: Vitrimers — polymers with covalent + exchangeable crosslinks

**Test modes**: All 6 protocols

**Key characteristic**: **3 subnetworks** (Permanent + Exchangeable + Dissociative) with TST kinetics

- Bond Exchange Reactions (BER) via transition state theory
- Factor-of-2 relaxation: :math:`\tau_E = 1/(2k_{BER})` — both :math:`\boldsymbol{\mu}` and :math:`\boldsymbol{\mu}_{nat}` relax
- :math:`\sigma_E \to 0` at steady state (natural state tracks deformation)
- 5 factory methods for limiting cases (neo-Hookean, Maxwell, Zener, etc.)

**For detailed equations**: :doc:`/models/hvm/index`

**For tutorial**: :doc:`../03_advanced_topics/vitrimer_models`

Family 19: HVNM (Hybrid Vitrimer Nanocomposite Model)
------------------------------------------------------

**Who they're for**: Nanoparticle-filled vitrimers with interphase reinforcement

**Test modes**: All 6 protocols

**Key characteristic**: **4 subnetworks** (P + E + D + Interphase) with Guth-Gold amplification

- Extends HVM with a 4th interphase subnetwork at NP-polymer interface
- :math:`X(\varphi) = 1 + 2.5\varphi + 14.1\varphi^2` modulus amplification
- Dual TST kinetics: independent matrix and interphase exchange rates
- :math:`\varphi = 0` recovers HVM exactly (machine precision verified)
- 5 factory methods for limiting cases

**For detailed equations**: :doc:`/models/hvnm/index`

**For tutorial**: :doc:`../03_advanced_topics/vitrimer_models`

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
      │     ├─→ Power-law / gel-like / complex?
      │     │      └─→ YES: Fractional (FML, FZSS, FMG)
      │     │
      │     └─→ Multiple relaxation processes?
      │            └─→ YES: TNT MultiSpecies, VLB MultiNetwork, GeneralizedMaxwell
      │
      ├─→ Steady Shear Flow (nonlinear)
      │     │
      │     ├─→ Shear thinning without yield stress?
      │     │      └─→ YES: PowerLaw, Carreau, Giesekus
      │     │
      │     ├─→ Yield stress present?
      │     │      └─→ YES: Bingham, Herschel-Bulkley, Saramito
      │     │
      │     └─→ Thixotropic (time-dependent)?
      │            └─→ YES: DMT, Fluidity, IKH
      │
      ├─→ Transient (Startup / Stress Overshoot)
      │     │
      │     ├─→ Polymer solution/melt?
      │     │      └─→ YES: Giesekus, TNT
      │     │
      │     ├─→ Thixotropic fluid?
      │     │      └─→ YES: DMT, IKH, Fluidity-Saramito
      │     │
      │     └─→ Vitrimer / adaptive material?
      │            └─→ YES: HVM, HVNM
      │
      └─→ Dense suspension / Glass?
            │
            ├─→ Colloidal glass?
            │      └─→ YES: ITT-MCT, SGR
            │
            ├─→ Metallic glass / amorphous solid?
            │      └─→ YES: STZ, EPM
            │
            └─→ Soft glass (foam, emulsion)?
                   └─→ YES: SGR, HL

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

   4. **Nonlinear viscoelastic** (Giesekus, Saramito): Tensor ODE, normal stresses, yielding

   5. **Thixotropic** (DMT, IKH/FIKH, Fluidity): Structure parameter, time-dependent viscosity

   6. **Soft matter physics** (SGR, HL, ITT-MCT): Statistical mechanics, glass transition

   7. **Elasto-plastic** (EPM, STZ): Amorphous solid yielding dynamics

   8. **Transient networks** (TNT, VLB): Dynamic crosslink attachment/detachment

   9. **Vitrimers** (HVM, HVNM): Bond exchange reactions, nanocomposite reinforcement

   10. **LAOS analysis** (SPP): Nonlinear oscillatory characterization

   11. **Start simple**: Try classical first, upgrade only if needed

   12. **Physics determines family**: Linear (classical/fractional) → nonlinear (flow/Giesekus) → thixotropic (DMT/IKH) → network (TNT/VLB) → vitrimer (HVM/HVNM)

.. admonition:: Self-Check Questions
   :class: tip

   1. **You observe** :math:`G'` **and** :math:`G''` **scaling as** :math:`\omega^{0.4}` **(parallel lines). Which family?**

      Hint: Power-law scaling indicates fractional

   2. **Your material shows exponential stress relaxation. Start with classical or fractional?**

      Hint: Exponential indicates classical (simpler)

   3. **Can you use a Maxwell model for steady shear flow data?**

      Hint: Maxwell is linear viscoelastic, not for flow

   4. **What does** :math:`\alpha = 0.9` **tell you about the material?**

      Hint: Close to 1 means narrow relaxation spectrum, nearly liquid-like

   5. **You fit PowerLaw and get** :math:`n = 0.6`. **What does this mean?**

      Hint: :math:`n < 1` indicates shear thinning

Further Reading
---------------

**Within this documentation**:

- :doc:`model_selection` — Decision flowcharts and compatibility checking
- :doc:`fitting_strategies` — Initialization and validation

**Model handbooks** (full equations and theory):

- :doc:`/models/classical/index` — 3 classical models
- :doc:`/models/fractional/index` — 11 fractional models
- :doc:`/models/flow/index` — 6 flow models
- :doc:`/models/giesekus/index` — 2 Giesekus models
- :doc:`/models/sgr/index` — 2 SGR models
- :doc:`/models/fluidity/index` — Fluidity + Saramito models
- :doc:`/models/epm/index` — 2 EPM models
- :doc:`/models/ikh/index` — 2 IKH models
- :doc:`/models/fikh/index` — 2 FIKH models
- :doc:`/models/dmt/index` — 2 DMT models
- :doc:`/models/hl/index` — HL model
- :doc:`/models/stz/index` — STZ model
- :doc:`/models/spp/index` — SPP model
- :doc:`/models/itt_mct/index` — 2 ITT-MCT models
- :doc:`/models/tnt/index` — 5 TNT models
- :doc:`/models/vlb/index` — 4 VLB models
- :doc:`/models/hvm/index` — HVM model
- :doc:`/models/hvnm/index` — HVNM model

**Advanced tutorials**:

- :doc:`../03_advanced_topics/fractional_viscoelasticity_reference` — Fractional calculus
- :doc:`../03_advanced_topics/sgr_analysis` — Soft glassy rheology
- :doc:`../03_advanced_topics/constitutive_ode_models` — Giesekus, IKH, Saramito
- :doc:`../03_advanced_topics/thixotropy_yielding` — DMT, Fluidity, HL, STZ, EPM
- :doc:`../03_advanced_topics/dense_suspensions_glasses` — ITT-MCT
- :doc:`../03_advanced_topics/polymer_network_models` — TNT and VLB
- :doc:`../03_advanced_topics/vitrimer_models` — HVM and HVNM
- :doc:`../03_advanced_topics/transforms_complete` — All 7 data transforms

Summary
-------

RheoJAX provides **53 rheological models** across 22 families (20 listed below; Fractional
Maxwell, Fractional Zener, and Fractional Advanced are sub-families counted separately in
some contexts):

- **Classical** (3): Maxwell, Zener, SpringPot — exponential relaxation, simple materials
- **Fractional** (11): FML, FZSS, FMG, Burgers, Jeffreys, etc. — power-law relaxation
- **Flow** (6): PowerLaw, Carreau, HB, Bingham, Cross — nonlinear viscosity
- **Multi-Mode** (1): Generalized Maxwell — discrete relaxation spectrum
- **Giesekus** (2): Single/multi-mode — nonlinear viscoelastic tensor ODE
- **SGR** (2): Conventional, GENERIC — soft glassy materials
- **Fluidity** (2): Local, Nonlocal — cooperative flow
- **Saramito** (2): Local, Nonlocal — tensorial EVP + thixotropy
- **EPM** (2): Lattice, Tensorial — elasto-plastic yielding
- **IKH** (2): MIKH, MLIKH — isotropic-kinematic hardening
- **FIKH** (2): FIKH, FMLIKH — fractional kinematic hardening
- **DMT** (2): Local, Nonlocal — structure parameter thixotropy
- **HL** (1): Hébraud-Lequeux — mean-field soft matter
- **STZ** (1): Shear transformation zones — amorphous solids
- **SPP** (1): LAOS yield stress analysis
- **ITT-MCT** (2): Schematic, Isotropic — mode-coupling theory
- **TNT** (5): SingleMode, Cates, LoopBridge, MultiSpecies, StickyRouse
- **VLB** (4): Local, MultiNetwork, Variant, Nonlocal
- **HVM** (1): Hybrid vitrimer — 3 subnetworks + TST kinetics
- **HVNM** (1): Vitrimer nanocomposite — 4 subnetworks + Guth-Gold

**DMTA/DMA Support:** All 41+ oscillation-capable models also support tensile modulus (E*)
via automatic E* ↔ G* conversion when ``deformation_mode='tension'`` is specified.
See :doc:`/models/dmta/index` for details.

Always start simple and add complexity only when necessary.

Next Steps
----------

Proceed to: :doc:`model_selection`

Learn to use decision flowcharts and compatibility checking to choose the right model for your data.
