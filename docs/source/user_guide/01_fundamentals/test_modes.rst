.. _test_modes:

Test Modes in Rheology
=======================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Identify and describe four major rheological test modes (SAOS, relaxation, creep, flow)
   2. Choose the appropriate test mode for a given material and research question
   3. Interpret raw experimental data from each test mode
   4. Understand advantages and limitations of each technique
   5. Recognize which models are appropriate for each test mode

.. admonition:: Prerequisites
   :class: important

   - :doc:`what_is_rheology` — Understanding of stress, strain, viscoelasticity
   - :doc:`material_classification` — Material types (liquid, solid, gel)

Overview: Why Multiple Test Modes?
-----------------------------------

Different experimental techniques probe different aspects of material behavior:

- **Frequency domain vs. time domain**: SAOS vs. relaxation/creep
- **Linear vs. nonlinear**: Small strain (SAOS/relaxation/creep) vs. large strain/rate (flow)
- **Storage vs. dissipation**: G' vs. G" vs. η
- **Timescale range**: Fast (high ω) vs. slow (low ω)

No single test mode provides complete characterization—each reveals complementary information.

Test Mode Summary Table
------------------------

.. list-table:: RheoJAX Test Modes Reference
   :header-rows: 1
   :widths: 18 22 60

   * - TestModeEnum
     - Protocol
     - Description
   * - ``RELAXATION``
     - Stress relaxation G(t)
     - Step strain, measure stress decay over time
   * - ``CREEP``
     - Creep compliance J(t)
     - Step stress, measure strain growth over time
   * - ``OSCILLATION``
     - SAOS G*(ω)
     - Small-amplitude oscillatory shear, complex modulus
   * - ``FLOW_CURVE``
     - Steady-state η(γ̇)
     - Viscosity vs shear rate at equilibrium
   * - ``STARTUP``
     - Transient σ(t,γ̇)
     - Stress overshoot/undershoot at fixed shear rate
   * - ``ROTATION``
     - Legacy steady shear
     - Deprecated, use ``FLOW_CURVE`` instead

The Six Test Modes
-------------------

1. Small-Amplitude Oscillatory Shear (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: Apply sinusoidal strain, measure sinusoidal stress response

**Input**: γ(t) = γ₀ sin(ωt)

**Output**: σ(t) = γ₀[G'(ω) sin(ωt) + G"(ω) cos(ωt)]

**Measured quantities**:

- G'(ω) — Storage modulus (elastic component, in-phase)
- G"(ω) — Loss modulus (viscous component, out-of-phase)
- tan(δ) = G"/G' — Loss tangent (damping ratio)
- η*(ω) = √[(G')² + (G")²] / ω — Complex viscosity

**Frequency sweep**: Vary ω from ~0.01 to ~100 rad/s

**Why it's powerful**:

- Probes linear viscoelasticity (non-destructive)
- Direct access to G' and G" across timescales
- Easiest to model mathematically (Fourier transform of relaxation)
- Most common test in rheology

**Limitations**:

- Limited to small strains (γ₀ typically < 1%)
- May miss nonlinear behavior
- Requires stable oscillation (not all instruments can do low frequencies well)

**When to use**:

- Characterizing material structure (crosslinking, gelation)
- Model fitting for viscoelastic parameters
- Quality control and formulation optimization
- Frequency-dependent behavior (mastercurves)

**Example applications**:

- Polymer melts: Molecular weight distribution from G' and G" curves
- Gels: Gelation monitoring (crossover of G' and G")
- Suspensions: Particle network structure

2. Stress Relaxation
~~~~~~~~~~~~~~~~~~~~~

**What it is**: Apply step strain, measure stress decay over time

**Input**: γ(t) = γ₀ H(t) — Step strain at t=0

**Output**: σ(t) = G(t) γ₀ — Stress decays as material relaxes

**Measured quantity**:

- G(t) — Relaxation modulus (stress/strain as function of time)

**Time range**: Typically 0.01 s to 1000 s

**Why it's powerful**:

- Direct measurement of relaxation spectrum
- Time-domain data (easier to interpret physically)
- Can access very long timescales
- Simple experimental protocol

**Limitations**:

- Requires fast strain application (instrument rise time < 0.01 s)
- Inertial artifacts at short times
- Sample slippage or edge fracture at long times
- Limited to materials that don't flow significantly

**When to use**:

- Materials with long relaxation times (polymer melts, elastomers)
- Studying molecular relaxation mechanisms
- Validating viscoelastic models
- Gelation studies (loss of relaxation as gel forms)

**Example applications**:

- Polymers: Reptation dynamics, entanglement networks
- Rubbers: Viscoelastic damping in elastomers
- Biological tissues: Stress relaxation in cartilage, skin

3. Creep (Compliance)
~~~~~~~~~~~~~~~~~~~~~

**What it is**: Apply constant stress, measure strain increase over time

**Input**: σ(t) = σ₀ H(t) — Step stress at t=0

**Output**: γ(t) = J(t) σ₀ — Strain increases as material creeps

**Measured quantity**:

- J(t) — Creep compliance (strain/stress as function of time)

**Time range**: Typically 0.1 s to 10,000 s (hours)

**Why it's powerful**:

- Probes long-term deformation under constant load
- Can distinguish viscous flow from elastic deformation
- Sensitive to weak network structures
- Physically intuitive (mimics real loading conditions)

**Limitations**:

- Long experimental time
- Difficult to apply truly constant stress (instrument drift)
- Sample may flow out of geometry
- Less common than SAOS or relaxation

**When to use**:

- Studying long-term material stability (sagging, settling)
- Weak gels and soft solids
- Materials near yield stress
- Validating linear viscoelastic models (creep and relaxation are related)

**Example applications**:

- Asphalt: Long-term deformation under road load
- Food products: Spreadability, flow under gravity
- Soft tissues: Load-bearing capacity

4. Steady Shear Flow (Flow Curve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: Apply constant shear rate, measure viscosity at equilibrium

**Input**: γ̇ = constant — Shear rate (s⁻¹)

**Output**: σ = η(γ̇) · γ̇ — Shear stress

**Measured quantity**:

- η(γ̇) — Viscosity as a function of shear rate

**Shear rate range**: Typically 0.01 to 1000 s⁻¹

**RheoJAX test mode**: ``test_mode='flow_curve'`` (or legacy ``test_mode='rotation'``)

**Why it's powerful**:

- Directly measures flow behavior
- Reveals nonlinear effects (shear thinning, shear thickening, yield stress)
- Mimics processing conditions (pumping, mixing, extrusion)
- Simple physical interpretation

**Limitations**:

- Nonlinear regime (can't predict from linear viscoelasticity)
- Edge effects, wall slip, sample expulsion
- May structurally damage sample
- Not directly related to G' and G" (except at very small γ̇)

**When to use**:

- Processing design (extrusion, coating, pumping)
- Formulation optimization (pumpability, spreadability)
- Quality control (viscosity specs)
- Studying flow instabilities

**Example applications**:

- Paints and coatings: Shear thinning for easy application
- Food: Mouthfeel, pourability
- Inks: Printing behavior
- Blood: Cardiovascular fluid dynamics

5. Startup Shear
~~~~~~~~~~~~~~~~~

**What it is**: Apply constant shear rate, measure transient stress evolution

**Input**: γ̇ = constant (suddenly applied at t=0)

**Output**: σ(t) — Stress as function of time at fixed shear rate

**Measured quantity**:

- σ(t, γ̇) — Transient stress response (often shows overshoot/undershoot)

**Time range**: Typically 0.01 to 100 s

**RheoJAX test mode**: ``test_mode='startup'``

**Why it's powerful**:

- Reveals thixotropic behavior (stress overshoot, undershoot)
- Probes microstructural evolution during flow
- Critical for understanding yielding dynamics
- Used for elasto-plastic model validation (EPM, IKH)

**Limitations**:

- Transient data harder to model than steady-state
- Requires fast instrument response
- Sample history-dependent (must control pre-shear)
- May require multiple shear rates for complete characterization

**When to use**:

- Studying thixotropy and shear rejuvenation
- Validating constitutive models (EPM, IKH, STZ)
- Understanding yield stress fluids and soft glasses
- Investigating shear banding and flow instabilities

**Example applications**:

- Colloidal gels: Stress overshoot indicates structural breakdown
- Emulsions: Yielding dynamics in mayonnaise, cosmetics
- Thixotropic fluids: Drilling muds, paints
- Soft glassy materials: Foams, pastes

6. Large-Amplitude Oscillatory Shear (LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**What it is**: Apply sinusoidal strain at large amplitudes, analyze nonlinear stress response

**Input**: γ(t) = γ₀ sin(ωt) with γ₀ >> linear limit

**Output**: σ(t) — Non-sinusoidal stress waveform

**Measured quantities**:

- Higher harmonics: σ₃/σ₁, σ₅/σ₁
- Chebyshev coefficients (e₁, e₃, v₁, v₃)
- Lissajous-Bowditch curves (σ vs γ, σ vs γ̇)
- SPP decomposition (sequence of physical processes)

**RheoJAX test mode**: ``test_mode='oscillation'`` with SPP models

**Why it's powerful**:

- Probes nonlinear viscoelasticity within single test
- Fingerprints material microstructure
- Distinguishes between similar linear rheology materials
- Rich information in single experiment

**Limitations**:

- Complex interpretation (multiple analysis frameworks)
- Requires specialized instruments and software
- Computationally intensive analysis
- No universal standards for reporting

**When to use**:

- Distinguishing materials with similar G', G"
- Studying yielding and flow transitions
- Material fingerprinting and quality control
- Research on nonlinear constitutive behavior

**Example applications**:

- Gels: Yield stress determination from Lissajous curves
- Polymer melts: Strain-hardening/softening characterization
- Complex fluids: Microstructural evolution during deformation

Visual Comparison of Test Modes
--------------------------------

.. code-block:: text

   SAOS (Frequency Sweep)
   ───────────────────────
   Input:  γ(t) = γ₀ sin(ωt)
           ┌───┐     ┌───┐
           │   │     │   │
   ────────┘   └─────┘   └────

   Output: σ(t) (phase-shifted)
           ┌───┐     ┌───┐
         ┌─┘   └───┬─┘   └──
   ──────┘         └────────

   Measure: G'(ω), G"(ω)


   STRESS RELAXATION
   ─────────────────
   Input:  γ(t) = γ₀ (step)
                 ┌──────────
           γ₀ ───┤
                 │
   ──────────────┘

   Output: σ(t) (decays)
                 ╱────
           σ₀ ──╱
               ╱
   ───────────╱

   Measure: G(t)


   CREEP
   ─────
   Input:  σ(t) = σ₀ (step)
                 ┌──────────
           σ₀ ───┤
                 │
   ──────────────┘

   Output: γ(t) (increases)
                       ╱─────
                     ╱
                   ╱
   ───────────────╱

   Measure: J(t)


   STEADY SHEAR FLOW
   ─────────────────
   Input:  γ̇ = const (various rates)
           │ ╱╱╱╱╱╱╱╱╱
           │╱╱╱╱╱╱╱╱╱
           ╱╱╱╱╱╱╱╱╱

   Output: σ (steady state)
           │     ●
           │   ●
           │ ●
           └────────── γ̇

   Measure: η(γ̇)

Relationships Between Test Modes
---------------------------------

Linear Viscoelasticity: SAOS ↔ Relaxation ↔ Creep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the **linear regime**, all three are related by Fourier transform:

**SAOS to Relaxation**:

.. math::

   G'(\omega) = \omega \int_0^\infty G(t) \sin(\omega t) \, dt

   G"(\omega) = \omega \int_0^\infty G(t) \cos(\omega t) \, dt

**Relaxation to SAOS** (inverse transform):

.. math::

   G(t) = \frac{2}{\pi} \int_0^\infty \frac{G"(\omega)}{\omega} \cos(\omega t) \, d\omega

**Creep and Relaxation** (Laplace space):

.. math::

   \tilde{J}(s) \cdot \tilde{G}(s) = \frac{1}{s^2}

**Practical implication**: If you fit a model to SAOS data, you can predict relaxation and creep behavior (and vice versa).

Flow vs. Linear Viscoelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cox-Merz rule** (empirical, often holds for polymers):

.. math::

   \eta(\dot{\gamma}) \approx \eta^*(\omega) \quad \text{at } \dot{\gamma} = \omega

**Limitation**: Only valid for some materials, breaks down for structured fluids (suspensions, gels).

Choosing the Right Test Mode
-----------------------------

Decision Flowchart
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   [What do you want to know?]
      │
      ├─→ "Frequency-dependent viscoelasticity (G', G")"
      │      └─→ SAOS (frequency sweep)
      │
      ├─→ "Long-term deformation under load"
      │      └─→ CREEP
      │
      ├─→ "Relaxation timescales and spectrum"
      │      └─→ STRESS RELAXATION
      │
      ├─→ "Flow behavior, processing conditions"
      │      └─→ STEADY SHEAR FLOW
      │
      └─→ "Comprehensive characterization"
             └─→ Combine SAOS + Relaxation + Flow

Practical Guidelines
~~~~~~~~~~~~~~~~~~~~

**Use SAOS when**:

- You need G' and G" for modeling
- Material is stable over long time
- You want non-destructive testing
- You're monitoring gelation or curing

**Use Stress Relaxation when**:

- You need time-domain data
- Material has long relaxation times
- You're studying molecular mechanisms
- You want to validate SAOS-derived models

**Use Creep when**:

- You're studying long-term stability (sagging, settling)
- Material is near yield stress
- You have very long experimental time available
- You want to separate viscous flow from elastic deformation

**Use Steady Shear Flow when**:

- You're designing processing equipment
- You need viscosity at specific shear rates
- You're studying shear thinning/thickening
- You need to detect yield stress

Model Compatibility with Test Modes
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 10 10 10 10 10 10

   * - Model Family
     - SAOS
     - Relax
     - Creep
     - Flow
     - Startup
     - LAOS
   * - Classical (Maxwell, Zener)
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
   * - Fractional Models
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
   * - Flow (PowerLaw, Carreau, HB)
     - ✗
     - ✗
     - ✗
     - ✓
     - ✗
     - ✗
   * - SGR (Soft Glassy Rheology)
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
   * - Fluidity (Local, Nonlocal)
     - ✓
     - ✗
     - ✗
     - ✓
     - ✗
     - ✗
   * - EPM (Lattice, Tensorial)
     - ✗
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
   * - IKH (MIKH, MLIKH)
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
   * - HL (Hébraud-Lequeux)
     - ✓
     - ✓
     - ✗
     - ✗
     - ✗
     - ✗
   * - STZ (Shear Transformation)
     - ✓
     - ✓
     - ✗
     - ✓
     - ✓
     - ✗
   * - SPP (LAOS Analysis)
     - ✗
     - ✗
     - ✗
     - ✗
     - ✗
     - ✓

**Key distinctions**:

- **Linear viscoelastic** (Classical, Fractional, IKH): SAOS, relaxation, creep
- **Flow models** (PowerLaw, Carreau, HB): Nonlinear steady shear only
- **Soft matter physics** (SGR, HL, Fluidity): Statistical mechanics approaches
- **Elasto-plastic** (EPM, STZ): Startup transients and flow curves
- **Nonlinear oscillatory** (SPP): LAOS analysis and yield stress

Worked Example: Multi-Mode Characterization
--------------------------------------------

**Material**: Polymer melt (polystyrene)

**Goal**: Complete rheological characterization

**Experimental protocol**:

1. **SAOS frequency sweep** (10⁻² to 10² rad/s)

   - Result: G' ~ ω², G" ~ ω at low ω (liquid-like)
   - Crossover at ω_c ≈ 1 rad/s → τ ≈ 1 s
   - Fit: Fractional Maxwell Liquid (FML)

2. **Stress relaxation** (t = 0.01 to 100 s)

   - Result: G(t) decays from 10⁵ Pa to <100 Pa
   - Confirms liquid-like behavior (no plateau)
   - Validates FML fit from SAOS

3. **Steady shear flow** (γ̇ = 10⁻² to 10³ s⁻¹)

   - Result: Shear thinning (η decreases with γ̇)
   - Zero-shear viscosity η₀ ≈ 10⁵ Pa·s
   - Fit: Carreau model for processing predictions

**Outcome**: Complete characterization for both linear viscoelasticity (SAOS/relaxation) and nonlinear flow (steady shear).

Key Concepts
------------

.. admonition:: Main Takeaways
   :class: tip

   1. **SAOS**: Frequency-dependent G' and G", most common test, linear regime

   2. **Stress Relaxation**: Time-domain G(t), direct measurement of relaxation spectrum

   3. **Creep**: Long-term deformation J(t), good for weak gels and stability

   4. **Steady Shear Flow**: Nonlinear viscosity η(γ̇), for processing design

   5. **Linear tests are related** via Fourier/Laplace transforms—fitting one predicts others

.. admonition:: Self-Check Questions
   :class: tip

   1. **You need to predict how a material flows during extrusion. Which test mode is most relevant?**

      Hint: Think about processing conditions (shear rate)

   2. **Why can't you use a flow curve (η vs γ̇) to predict SAOS behavior (G' vs ω)?**

      Hint: Linear vs. nonlinear regimes

   3. **A material has G' = G" = 1000 Pa at 1 rad/s. Can you predict G(t) at t = 1 s exactly?**

      Hint: Need full frequency sweep, not single point

   4. **You observe stress relaxation G(t) from 10⁵ Pa to 10³ Pa over 100 s. Is this a liquid or solid?**

      Hint: Check if it plateaus or continues decaying

   5. **Why is SAOS preferred over creep for routine characterization?**

      Hint: Consider experimental time and frequency range

Further Reading
---------------

**Within this documentation**:

- :doc:`parameter_interpretation` — Physical meaning of measured quantities
- :doc:`../02_model_usage/model_families` — Which models apply to which test modes
- :doc:`../04_practical_guides/data_io` — Loading experimental data from instruments

**Textbook chapters**:

- Macosko, *Rheology*, Chapter 7 — Experimental methods
- Ferry, *Viscoelastic Properties of Polymers*, Chapter 3 — Dynamic mechanical properties

Summary
-------

Four major test modes probe different aspects of rheology: **SAOS** (frequency-dependent G', G"), **stress relaxation** (time-domain G(t)), **creep** (long-term J(t)), and **steady shear flow** (nonlinear η(γ̇)). Linear viscoelastic tests (SAOS, relaxation, creep) are mathematically related, while flow tests probe nonlinear behavior. Choose test modes based on your material and research question.

Next Steps
----------

Proceed to: :doc:`parameter_interpretation`

Learn the physical meaning of rheological parameters like G', G", τ, and α.
