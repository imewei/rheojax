.. _model-tnt-sticky-rouse:

===========================================================
TNT Sticky Rouse (Multi-Mode Sticker Dynamics) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

----

Quick Reference
===============

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Use when:**
     - Multi-sticker associating polymers with hierarchical relaxation (multiple stickers per chain, Rouse-like sub-chain dynamics between stickers)
   * - **Materials:**
     - Multi-sticker ionomers (sulfonated polystyrene), HEUR with multiple hydrophobes, beta-cyclodextrin/adamantane complexes, supramolecular polymers with multiple binding sites
   * - **Parameters:**
     - 2N+2 parameters: :math:`G_k` (N moduli), :math:`\tau_{R,k}` (N Rouse times), :math:`\tau_s` (sticker lifetime), :math:`\eta_s` (solvent viscosity), where N = n_modes
   * - **Key equation:**
     - Multi-mode ODE: :math:`\frac{dS_k}{dt} = \kappa \cdot S_k + S_k \cdot \kappa^T - \frac{1}{\tau_{eff,k}}(S_k - I)` with :math:`\tau_{eff,k} = \tau_{R,k} + \tau_s`
   * - **Test modes:**
     - FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS (all 6 protocols)
   * - **Signature:**
     - Broad relaxation spectrum, power-law stress relaxation at intermediate times, Rouse-like :math:`G' \sim \omega^{1/2}` scaling, sticky plateau at low frequencies
   * - **Typical range:**
     - :math:`\tau_s`: 1e-6 to 1e6 s; :math:`G_k`: 1e-2 to 1e6 Pa; :math:`\tau_{R,k} = \tau_{R,1}/k^2` (Rouse scaling)
   * - **Related models:**
     - :ref:`model-tnt-tanaka-edwards`, :ref:`model-tnt-multi-species`, :ref:`model-generalized-maxwell`, Rouse model

----

Notation Guide
==============

.. list-table:: Primary Symbols
   :widths: 15 60 25
   :header-rows: 1

   * - Symbol
     - Definition
     - Units
   * - :math:`G_k`
     - Modulus of Rouse mode k (k = 1..N)
     - Pa
   * - :math:`\tau_{R,k}`
     - Rouse relaxation time for mode k
     - s
   * - :math:`\tau_s`
     - Sticker (association) lifetime
     - s
   * - :math:`\eta_s`
     - Solvent viscosity
     - Pa·s
   * - :math:`\tau_{eff,k}`
     - Effective relaxation time for mode k: :math:`\tau_{R,k} + \tau_s`
     - s
   * - :math:`S_k`
     - Conformation tensor for mode k (3x3 symmetric)
     - dimensionless
   * - :math:`N`
     - Number of Rouse modes (n_modes)
     - dimensionless
   * - :math:`N_s`
     - Number of stickers per chain (physical parameter)
     - dimensionless
   * - :math:`\kappa`
     - Velocity gradient tensor :math:`(\nabla v)^T`
     - 1/s
   * - :math:`D`
     - Rate of deformation tensor :math:`(D = (\kappa + \kappa^T)/2)`
     - 1/s
   * - :math:`\sigma`
     - Extra stress tensor
     - Pa
   * - :math:`I`
     - Identity tensor
     - dimensionless
   * - :math:`p`
     - Rouse mode index (alternative to k)
     - dimensionless

.. list-table:: Derived Quantities
   :widths: 15 60 25
   :header-rows: 1

   * - Symbol
     - Definition
     - Units
   * - :math:`\eta_0`
     - Zero-shear viscosity: :math:`\sum_k G_k \tau_{eff,k} + \eta_s`
     - Pa·s
   * - :math:`G_N^{(0)}`
     - Plateau modulus: :math:`\sum_k G_k`
     - Pa
   * - :math:`\tau_{max}`
     - Longest relaxation time: :math:`\max(\tau_{eff,k})`
     - s
   * - :math:`\lambda`
     - Terminal relaxation time (sticky-limited)
     - s
   * - :math:`\omega_c`
     - Characteristic (crossover) frequency
     - rad/s

----

Overview
========

Physical Picture
----------------

The **TNT Sticky Rouse** model describes the viscoelastic response of unentangled polymer chains bearing multiple reversible association sites ("stickers"). It extends the classical Rouse model to incorporate hierarchical relaxation: fast Rouse dynamics of sub-chain segments between stickers, combined with slow sticker exchange kinetics.

**Historical Context:**

- **Leibler, Rubinstein, Colby (1991):** Introduced sticky reptation model for entangled associating polymers
- **Rubinstein & Semenov (1998):** Developed thermoreversible gelation theory for multi-sticker networks
- **Chen, Liang, Colby (2013):** Experimental validation of sticky Rouse dynamics in ionomers

**Multi-Sticker Architecture:**

Consider a flexible polymer chain with :math:`N_s` regularly-spaced stickers along its backbone. Each sticker can reversibly bind to complementary sites (from other chains or matrix). Between consecutive stickers, the chain segment behaves as a Rouse sub-chain with its own relaxation spectrum.

**Hierarchical Relaxation:**

1. **Short times** (:math:`t \ll \tau_s`): Stickers remain associated; chain appears permanently crosslinked; Rouse modes relax subject to sticker constraints
2. **Intermediate times** (:math:`t \sim \tau_s`): Sticker exchange begins; interplay between Rouse dynamics and sticker unbinding
3. **Long times** (:math:`t \gg \tau_s`): All stickers have exchanged; terminal relaxation governed by longest effective time :math:`\tau_{eff,1} = \tau_{R,1} + \tau_s`

**Key Signature:**

The superposition of multiple Rouse modes with sticker-renormalized relaxation times creates a **broad relaxation spectrum** that manifests as:

- Power-law stress relaxation :math:`G(t) \sim t^{-1/2}` at intermediate times
- Characteristic :math:`G' \sim \omega^{1/2}` scaling in SAOS (Rouse regime)
- Sticky plateau at frequencies :math:`1/\tau_s < \omega < 1/\tau_{R,N}`

Material Examples
-----------------

**Sulfonated Polystyrene Ionomers:**

- Multiple ionic groups along backbone
- Reversible ionic clusters act as transient crosslinks
- Exhibits sticky Rouse behavior below entanglement threshold

**HEUR (Hydrophobically-modified Ethoxylated Urethanes):**

- Multiple hydrophobic end-groups per chain
- Hydrophobic association creates reversible network
- Broad relaxation spectrum from hierarchical dynamics

**Beta-Cyclodextrin/Adamantane Complexes:**

- Polymers with multiple adamantane guest groups
- Beta-cyclodextrin hosts provide reversible binding
- Tunable sticker lifetime via pH, temperature

**Supramolecular Polymers:**

- Hydrogen-bonded assemblies with multiple binding sites
- Metal-ligand coordination polymers
- Pi-stacking based reversible networks

Relationship to Rouse Model
----------------------------

The classical **Rouse model** describes unentangled polymer melts as bead-spring chains with Gaussian statistics. Each mode :math:`p` has:

- Modulus: :math:`G_p \approx G_N^{(0)}/N` (equal mode strength)
- Relaxation time: :math:`\tau_p = \tau_R/p^2` (harmonic spacing)
- SAOS prediction: :math:`G'(\omega) \sim G''(\omega) \sim \omega^{1/2}` in Rouse regime

The **Sticky Rouse** model modifies this by adding a sticker contribution to each mode's relaxation:

.. math::

   \tau_{eff,p} = \tau_{R,p} + \tau_s = \frac{\tau_{R,1}}{p^2} + \tau_s

This shifts the mode spectrum and creates new regimes depending on the ratio :math:`\tau_s/\tau_{R,1}`.

----

Physical Foundations
====================

Rouse Dynamics
--------------

For a flexible polymer chain with :math:`N_s` beads (no hydrodynamic interactions), the Rouse model predicts:

**Normal Mode Decomposition:**

The chain's configuration is decomposed into :math:`N_s` normal modes with eigenvalues:

.. math::

   \lambda_p = 4 \sin^2\left(\frac{\pi p}{2N_s}\right) \quad p = 1, 2, \ldots, N_s

**Relaxation Times:**

Each mode has characteristic time:

.. math::

   \tau_p = \frac{\zeta b^2}{3\pi^2 k_B T} \cdot \frac{N_s^2}{p^2} = \frac{\tau_{R,1}}{p^2}

where :math:`\zeta` is bead friction, :math:`b` is segment length.

**Stress Contribution:**

Mode :math:`p` contributes:

.. math::

   \sigma_p(t) = G_p \langle S_p(t) - I \rangle

with :math:`G_p = \frac{\nu k_B T}{N_s}` (polymer number density :math:`\nu`).

Sticker Kinetics
----------------

**Association/Dissociation:**

Each sticker undergoes reversible binding:

.. math::

   \text{Bound} \xrightleftharpoons[k_{on}]{k_{off}} \text{Free}

with sticker lifetime:

.. math::

   \tau_s = \frac{1}{k_{off}}

**Renewal Assumption:**

When a sticker detaches, the sub-chain segment immediately relaxes its orientation via Rouse dynamics. This "renewal" process couples sticker exchange to Rouse modes.

Effective Relaxation Time
--------------------------

The key insight is that mode :math:`k` can only fully relax after:

1. Stickers on both sides of the sub-chain have detached (time :math:`\sim \tau_s`)
2. Rouse relaxation of the freed segment (time :math:`\sim \tau_{R,k}`)

This gives:

.. math::

   \tau_{eff,k} = \tau_{R,k} + \tau_s

**Physical Interpretation:**

- If :math:`\tau_s \ll \tau_{R,k}`: Stickers exchange rapidly; mode relaxes at Rouse time
- If :math:`\tau_s \gg \tau_{R,k}`: Sticker exchange rate-limiting; mode relaxes at :math:`\tau_s`
- If :math:`\tau_s \sim \tau_{R,k}`: Cooperative effect; effective time is sum

Multi-Sticker Coupling
-----------------------

For :math:`N_s` stickers dividing the chain into :math:`N_s+1` segments:

- **Independent Modes:** Each Rouse mode "sees" the sticker network as a collection of independent obstacles
- **Spectrum Broadening:** The range of :math:`\tau_{eff,k}` values spans from :math:`\tau_{R,N} + \tau_s` to :math:`\tau_{R,1} + \tau_s`
- **Sticky Plateau:** At frequencies :math:`1/\tau_s < \omega < 1/\tau_{R,N}`, stickers are effectively permanent crosslinks; :math:`G' \approx G_N^{(0)}`

Scaling Predictions
-------------------

**High-Frequency Rouse Regime** (:math:`\omega \tau_{R,1} \gg 1`):

.. math::

   G'(\omega) \approx G''(\omega) \approx G_N^{(0)} \left(\frac{\omega \tau_{R,1}}{N}\right)^{1/2}

**Intermediate Sticky Regime** (:math:`1/\tau_s < \omega < 1/\tau_{R,N}`):

.. math::

   G'(\omega) \approx G_N^{(0)} \quad \text{(plateau)}

**Terminal Regime** (:math:`\omega \tau_s \ll 1`):

.. math::

   G'(\omega) \approx \left(\sum_k G_k \tau_{eff,k}\right) \omega^2, \quad G''(\omega) \approx \eta_0 \omega

Leibler-Rubinstein-Colby (LRC) Scaling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The original LRC theory (Leibler, Rubinstein, Colby 1991) provides scaling relations for
sticky Rouse dynamics:

**Terminal relaxation time:**

.. math::

   \tau_{\text{term}} = \tau_s \left(\frac{N}{N_s}\right)^2

where :math:`\tau_s` is the sticker lifetime and :math:`N_s` is the number of monomers
between stickers.

**Zero-shear viscosity:**

.. math::

   \eta_0 \sim \tau_s \, N_s^2 \, N^3

The :math:`N^3` scaling (rather than :math:`N^{3.4}` for entangled melts) reflects the
Rouse-like dynamics between sticker release events.

Sticky Reptation Crossover
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For entangled sticky polymers, there is a crossover between sticky Rouse and sticky
reptation dynamics:

.. math::

   \tau_{\text{term}} = \max\!\left(\tau_{\text{rep}}, \, \tau_s \cdot n_e\right)

where :math:`n_e` is the number of entanglements per sticker. When :math:`\tau_s \cdot n_e
> \tau_{\text{rep}}`, sticker dynamics dominate over reptation — the **sticky regime**.
When reptation is faster, the system crosses over to standard entangled dynamics.

----

Governing Equations
===================

Multi-Mode Conformation Tensor Evolution
-----------------------------------------

For each Rouse mode :math:`k = 1, 2, \ldots, N`, the conformation tensor :math:`S_k` (3x3 symmetric) evolves according to:

.. math::

   \frac{dS_k}{dt} = \kappa \cdot S_k + S_k \cdot \kappa^T - \frac{1}{\tau_{eff,k}} (S_k - I)

where:

- :math:`\kappa = (\nabla v)^T` is the velocity gradient tensor
- :math:`I` is the identity tensor
- :math:`\tau_{eff,k} = \tau_{R,k} + \tau_s` is the effective relaxation time for mode :math:`k`

**Tensor Components:**

In 2D shear flow (:math:`\kappa_{xy} = \dot{\gamma}`, others zero):

.. math::

   \frac{dS_{xx,k}}{dt} &= 2\dot{\gamma} S_{xy,k} - \frac{1}{\tau_{eff,k}}(S_{xx,k} - 1) \\
   \frac{dS_{yy,k}}{dt} &= - \frac{1}{\tau_{eff,k}}(S_{yy,k} - 1) \\
   \frac{dS_{zz,k}}{dt} &= - \frac{1}{\tau_{eff,k}}(S_{zz,k} - 1) \\
   \frac{dS_{xy,k}}{dt} &= \dot{\gamma} S_{yy,k} - \frac{1}{\tau_{eff,k}} S_{xy,k}

Total Stress
------------

The extra stress tensor is the sum over all modes plus solvent contribution:

.. math::

   \sigma = \sum_{k=1}^{N} G_k (S_k - I) + 2\eta_s D

where :math:`D = (\kappa + \kappa^T)/2` is the rate of deformation tensor.

**Shear Stress:**

.. math::

   \sigma_{xy} = \sum_{k=1}^{N} G_k S_{xy,k} + \eta_s \dot{\gamma}

**Normal Stress Differences:**

.. math::

   N_1 = \sigma_{xx} - \sigma_{yy} = \sum_{k=1}^{N} G_k (S_{xx,k} - S_{yy,k})

State Vector
------------

The model tracks :math:`4N` degrees of freedom (4 independent components per mode):

.. math::

   \mathbf{y} = [S_{xx,1}, S_{yy,1}, S_{zz,1}, S_{xy,1}, \ldots, S_{xx,N}, S_{yy,N}, S_{zz,N}, S_{xy,N}]^T

**Equilibrium State:**

.. math::

   S_k = I \quad \forall k \implies \mathbf{y}_{eq} = [1, 1, 1, 0, \ldots, 1, 1, 1, 0]^T

Analytical Solutions
--------------------

**Small-Amplitude Oscillatory Shear (SAOS):**

For :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, linearization gives:

.. math::

   G'(\omega) = \sum_{k=1}^{N} G_k \frac{(\omega \tau_{eff,k})^2}{1 + (\omega \tau_{eff,k})^2}

.. math::

   G''(\omega) = \sum_{k=1}^{N} G_k \frac{\omega \tau_{eff,k}}{1 + (\omega \tau_{eff,k})^2} + \omega \eta_s

**Stress Relaxation:**

For step strain :math:`\gamma_0`, the relaxation modulus is:

.. math::

   G(t) = \sum_{k=1}^{N} G_k \exp\left(-\frac{t}{\tau_{eff,k}}\right)

**Flow Curve (Approximate):**

At steady shear rate :math:`\dot{\gamma}`, assuming mode decoupling:

.. math::

   \eta(\dot{\gamma}) \approx \sum_{k=1}^{N} \frac{G_k \tau_{eff,k}}{1 + (\tau_{eff,1} \dot{\gamma})^2} + \eta_s

(Note: This neglects nonlinear coupling; full solution requires ODE integration.)

**Startup Shear:**

Requires numerical integration of the multi-mode ODE system with initial condition :math:`\mathbf{y}(0) = \mathbf{y}_{eq}`.

**Creep:**

For constant stress :math:`\sigma_0`, strain evolution requires coupled ODE solution (no closed form).

**LAOS:**

For :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with large :math:`\gamma_0`, full nonlinear ODE integration is necessary; harmonics extracted via Fourier analysis.

----

Parameter Table
===============

.. list-table:: TNT Sticky Rouse Parameters
   :widths: 20 15 25 15 25
   :header-rows: 1

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Physical Meaning
   * - Mode moduli
     - :math:`G_k`
     - [varied]
     - (1e-2, 1e6) Pa
     - Contribution of Rouse mode k to total modulus
   * - Rouse times
     - :math:`\tau_{R,k}`
     - [varied]
     - (1e-6, 1e4) s
     - Relaxation time of mode k without stickers
   * - Sticker lifetime
     - :math:`\tau_s`
     - 1.0
     - (1e-6, 1e6) s
     - Average duration of sticker association
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0
     - (0.0, 1e4) Pa·s
     - Background viscosity (monomeric friction)

**Derived Parameters:**

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Symbol
     - Definition
     - Units
   * - :math:`\tau_{eff,k}`
     - :math:`\tau_{R,k} + \tau_s`
     - s
   * - :math:`G_N^{(0)}`
     - :math:`\sum_{k=1}^{N} G_k`
     - Pa
   * - :math:`\eta_0`
     - :math:`\sum_{k=1}^{N} G_k \tau_{eff,k} + \eta_s`
     - Pa·s
   * - :math:`\lambda`
     - :math:`\max_k(\tau_{eff,k})`
     - s

**Typical Constraints:**

For ideal Rouse behavior:

1. **Equal mode strengths:** :math:`G_k = G_N^{(0)}/N`
2. **Harmonic time spacing:** :math:`\tau_{R,k} = \tau_{R,1}/k^2`

These can be relaxed for real materials, but imposing them reduces parameter count from :math:`2N+2` to :math:`4` (:math:`G_N^{(0)}`, :math:`\tau_{R,1}`, :math:`\tau_s`, :math:`\eta_s`).

----

Parameter Interpretation
========================

Sticker Lifetime (:math:`\tau_s`)
----------------------------------

**Physical Meaning:**

Average time a sticker remains bound before dissociating. Controlled by:

- Binding energy: :math:`\tau_s \sim \exp(\Delta E_{bind}/k_B T)`
- Sticker chemistry (H-bonds, ionic, hydrophobic)
- Temperature, pH, ionic strength

**Regimes:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Condition
     - Behavior
   * - :math:`\tau_s \gg \tau_{R,1}`
     - Sticker-dominated; all modes relax at :math:`\sim \tau_s`; narrow spectrum; single-time Maxwell-like
   * - :math:`\tau_s \ll \tau_{R,N}`
     - Rouse-dominated; stickers irrelevant; pure Rouse spectrum :math:`G'(\omega) \sim \omega^{1/2}`
   * - :math:`\tau_{R,N} \ll \tau_s \ll \tau_{R,1}`
     - Crossover regime; broad spectrum; sticky plateau visible at :math:`\omega \sim 1/\tau_s`

**Experimental Determination:**

- Onset of sticky plateau in :math:`G'(\omega)` occurs near :math:`\omega \approx 1/\tau_s`
- Terminal relaxation time :math:`\lambda \approx \tau_{R,1} + \tau_s` (from :math:`G(t)` or :math:`G''` peak)

Rouse Times (:math:`\tau_{R,k}`)
---------------------------------

**Physical Meaning:**

Relaxation time of mode :math:`k` in the absence of stickers. Determined by:

- Segment friction :math:`\zeta`
- Molecular weight distribution
- Solvent quality (affects :math:`b`, :math:`\zeta`)

**Ideal Scaling:**

For monodisperse chains:

.. math::

   \tau_{R,k} = \frac{\tau_{R,1}}{k^2}, \quad \tau_{R,1} = \frac{\zeta N_s^2 b^2}{3\pi^2 k_B T}

**Polydispersity Effects:**

Real materials may deviate from :math:`1/k^2` scaling due to:

- Molecular weight distribution
- Chain branching
- Non-ideal solvent conditions

**Constraints in Fitting:**

To reduce parameter count, enforce :math:`\tau_{R,k} = \tau_{R,1}/k^2` and fit only :math:`\tau_{R,1}`.

Mode Moduli (:math:`G_k`)
--------------------------

**Physical Meaning:**

Contribution of mode :math:`k` to plateau modulus :math:`G_N^{(0)} = \sum_k G_k`. Related to chain density and mode entropy.

**Ideal Rouse:**

.. math::

   G_k = \frac{G_N^{(0)}}{N} \quad \forall k

**Real Materials:**

- Mode strengths may vary (non-ideal spectrum)
- Typically :math:`G_k` decreases slightly with :math:`k` due to friction distribution

**Fitting Strategy:**

- Unconstrained: Fit all :math:`N` values of :math:`G_k` independently (2N+2 total parameters)
- Constrained: Set :math:`G_k = G_N^{(0)}/N` and fit only :math:`G_N^{(0)}` (4 total parameters)

Solvent Viscosity (:math:`\eta_s`)
-----------------------------------

**Physical Meaning:**

Background viscosity from solvent or unassociated monomers. Provides high-frequency dissipation floor.

**Impact on Rheology:**

- Adds constant contribution to :math:`G''(\omega)`: :math:`\omega \eta_s`
- Shifts :math:`\tan\delta = G''/G'` at high :math:`\omega`
- Negligible for polymer melts; important for solutions

**Typical Values:**

- Melt: :math:`\eta_s \approx 0` Pa·s
- Dilute solution: :math:`\eta_s \approx \eta_{solvent}` (e.g., 0.001 Pa·s for water)
- Semi-dilute: :math:`\eta_s = \phi \eta_{solvent}` (volume fraction :math:`\phi`)

----

Validity and Assumptions
========================

Underlying Assumptions
----------------------

1. **Unentangled Regime:**

   - Chain length :math:`N_s < N_e` (entanglement threshold)
   - Rouse dynamics (no tube constraints)
   - Violated for high-MW associating polymers (use sticky reptation instead)

2. **Gaussian Statistics:**

   - Chains obey Gaussian elasticity (small to moderate deformations)
   - Breaks down for :math:`\gamma > 1` (finite extensibility)
   - FENE corrections needed for large :math:`\gamma_0` in LAOS

3. **Homogeneous Stickers:**

   - All stickers identical (same :math:`\tau_s`, binding energy)
   - No sticker-sticker variation along chain
   - Real systems may have binding site heterogeneity

4. **Independent Sticker Renewal:**

   - Sticker dissociation events uncorrelated
   - No cooperative unbinding (e.g., zipper-like dissociation)
   - Valid for dilute sticker networks

5. **Mean-Field Binding:**

   - Sticker rebinding is instantaneous to available sites
   - Neglects spatial correlations in binding site distribution
   - Assumes well-mixed environment

6. **No Excluded Volume:**

   - Ideal chain statistics
   - Violated in good solvent conditions (swollen coils)

7. **Affine Deformation:**

   - Chain deforms with the flow (no slip)
   - Valid for homogeneous shear; breaks down in extensional flows with chain tumbling

Material Applicability
----------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Material Class
     - When Appropriate
     - When Inappropriate
   * - Ionomers
     - Low MW (unentangled), dilute ionic groups
     - High MW (entangled), dense ionic clusters
   * - Supramolecular polymers
     - Weak H-bonds, multiple sites
     - Strong coordination bonds (lifetime distribution)
   * - Hydrogels
     - Unentangled precursors, reversible crosslinks
     - Chemical crosslinks, entangled networks
   * - Associating solutions
     - Dilute/semi-dilute, multiple hydrophobes
     - Concentrated (overlap), micellar aggregation

Comparison with Other Models
-----------------------------

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Model
     - Key Difference
     - When to Use Sticky Rouse Instead
   * - Generalized Maxwell
     - No physical mode structure
     - Need mechanistic interpretation, Rouse scaling validation
   * - TNT Tanaka-Edwards
     - Single-mode, simpler
     - Multiple relaxation times observed, broader spectrum
   * - Sticky Reptation
     - Entangled regime
     - Unentangled polymers (MW < entanglement threshold)
   * - GENERIC Fluidity
     - Thixotropic structure parameter
     - Thixotropy negligible, sticker exchange dominant

----

Regimes and Behavior
====================

Frequency-Domain Map
--------------------

**High-Frequency Rouse Regime** (:math:`\omega \tau_{R,1} \gg 1`):

.. math::

   G'(\omega) \approx G''(\omega) \approx G_N^{(0)} \sqrt{\frac{\omega \tau_{R,1}}{N}}

- Characteristic :math:`\omega^{1/2}` scaling
- Moduli roughly equal (:math:`\tan\delta \approx 1`)
- Polymer segments undergoing sub-Rouse relaxation

**Sticky Plateau Regime** (:math:`1/\tau_s < \omega < 1/\tau_{R,N}`):

.. math::

   G'(\omega) \approx G_N^{(0)}, \quad G''(\omega) \ll G'

- Stickers effectively permanent
- Temporary network behavior
- Width of plateau scales as :math:`\log(\tau_s/\tau_{R,N})` in frequency space

**Terminal Relaxation Regime** (:math:`\omega \tau_s \ll 1`):

.. math::

   G'(\omega) \approx \eta_0 \lambda \omega^2, \quad G''(\omega) \approx \eta_0 \omega

- Liquid-like terminal flow
- :math:`G'' > G'` (viscous dissipation dominates)
- Longest time :math:`\lambda = \tau_{R,1} + \tau_s`

Intermediate Frequency Signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The sticky Rouse model predicts a characteristic **half-power-law** scaling at
intermediate frequencies:

.. math::

   G'(\omega) \sim \omega^{1/2} \quad \text{for} \quad 1/\tau_{\text{term}} \ll \omega \ll 1/\tau_s

This :math:`\omega^{1/2}` dependence is the Rouse scaling, arising from the
self-similar relaxation of chain segments between stickers. It appears as a
characteristic slope in the log-log plot of :math:`G'` vs :math:`\omega`.

**Diagnostic value:** The :math:`\omega^{1/2}` intermediate regime distinguishes sticky
Rouse from:

- **Single Maxwell** (TNTSingleMode): Sharp transition from :math:`\omega^2` to plateau
- **Multi-species** (TNTMultiSpecies): Discrete steps between modes
- **Cates**: Near-perfect Maxwell with single crossover

Plateau Identification
^^^^^^^^^^^^^^^^^^^^^^^

For entangled sticky polymers, two plateaus may be visible in :math:`G'(\omega)`:

1. **High-frequency plateau** (:math:`G_N^0`): Entanglement plateau — reflects
   topological constraints between chain entanglements
2. **Low-frequency plateau** (:math:`G_e`): Sticker network plateau — reflects the
   elastic contribution of sticker-sticker associations

The ratio :math:`G_e / G_N^0` gives the fraction of stress carried by the sticker
network relative to entanglements.

Time-Domain Signatures
----------------------

**Stress Relaxation after Step Strain:**

.. math::

   G(t) = \sum_{k=1}^{N} G_k \exp\left(-\frac{t}{\tau_{eff,k}}\right)

- Multi-exponential decay
- At :math:`t \ll \tau_s`: Rapid Rouse decay (:math:`\sim t^{-1/2}` envelope)
- At :math:`t \sim \tau_s`: Crossover to slower decay
- At :math:`t \gg \tau_{R,1}`: Final exponential tail :math:`\sim \exp(-t/\lambda)`

**Startup Shear Flow:**

For constant :math:`\dot{\gamma}`, stress grows as:

1. Initial elastic response (fast modes)
2. Stress overshoot if :math:`\dot{\gamma} \tau_s > 1` (sticker network stretches before yielding)
3. Steady-state flow at :math:`\sigma_{ss} = \eta(\dot{\gamma}) \dot{\gamma}`

**Creep under Constant Stress:**

.. math::

   \gamma(t) \sim t^{1/2} \quad \text{at } t \ll \tau_s \quad \text{(sub-diffusive Rouse creep)}

.. math::

   \gamma(t) \sim t \quad \text{at } t \gg \tau_s \quad \text{(viscous flow)}

Nonlinear Flow Regimes
----------------------

**Shear Rate Parameter:**

Define Weissenberg number for mode :math:`k`:

.. math::

   Wi_k = \dot{\gamma} \tau_{eff,k}

**Regimes:**

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - :math:`Wi_1` Range
     - Behavior
   * - :math:`Wi_1 \ll 1`
     - Linear viscoelastic (Newtonian plateau); :math:`\eta \approx \eta_0`
   * - :math:`Wi_1 \sim 1`
     - Longest mode becomes nonlinear; stress overshoot in startup
   * - :math:`Wi_1 \gg 1`
     - Shear thinning; :math:`\eta \sim \dot{\gamma}^{-1}` (power-law from mode superposition)

**LAOS Nonlinearity:**

For oscillatory strain :math:`\gamma_0 \sin(\omega t)`:

- **Deborah number:** :math:`De_k = \omega \tau_{eff,k}`
- **Strain amplitude:** :math:`\gamma_0`

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - :math:`De_1`
     - :math:`\gamma_0`
     - Response
   * - :math:`\ll 1`
     - Small
     - Terminal regime; linear viscous dissipation
   * - :math:`\sim 1`
     - Small
     - Viscoelastic transition; :math:`G' \approx G''`
   * - :math:`\gg 1`
     - Small
     - Elastic regime; :math:`G' \gg G''`
   * - Any
     - :math:`> 1`
     - Nonlinear LAOS; higher harmonics appear; chain stretching

Sticky vs. Rouse Crossover
---------------------------

The relative importance of stickers vs. Rouse dynamics is governed by the ratio:

.. math::

   R = \frac{\tau_s}{\tau_{R,1}}

**Rouse-Dominated** (:math:`R \ll 1`):

- Stickers exchange much faster than Rouse relaxation
- Effectively no stickers; pure Rouse model applicable
- :math:`\tau_{eff,k} \approx \tau_{R,k} \propto 1/k^2`

**Sticky-Dominated** (:math:`R \gg 1`):

- Sticker exchange rate-limits all modes
- :math:`\tau_{eff,k} \approx \tau_s` for all :math:`k`
- Narrow spectrum; single-mode Maxwell-like behavior

**Crossover Regime** (:math:`R \sim 1`):

- Broad spectrum with :math:`\tau_{eff,k}` ranging from :math:`\tau_s` (slow modes) to :math:`\tau_{R,N} + \tau_s` (fast modes)
- Richest rheological behavior; power-law relaxation
- Sticky plateau visible in :math:`G'(\omega)`

----

What You Can Learn from This Model
===================================

Extracting Sticker Lifetime
----------------------------

**Method 1: Sticky Plateau Onset**

In SAOS data, identify frequency :math:`\omega_s` where :math:`G'` begins to plateau (transition from terminal to sticky regime):

.. math::

   \tau_s \approx \frac{1}{\omega_s}

**Method 2: Terminal Relaxation Time**

From stress relaxation :math:`G(t)`, fit the long-time tail:

.. math::

   G(t \to \infty) \sim \exp(-t/\lambda), \quad \lambda = \tau_{R,1} + \tau_s

If :math:`\tau_{R,1}` known (from mode fitting), extract :math:`\tau_s = \lambda - \tau_{R,1}`.

**Method 3: Peak in :math:`G''(\omega)`**

The terminal peak in :math:`G''` occurs near :math:`\omega \approx 1/\lambda`, providing another estimate of :math:`\tau_s`.

Determining Number of Modes
----------------------------

**Spectral Width:**

The breadth of the relaxation spectrum correlates with the number of distinguishable Rouse modes. Compare:

- Frequency span of :math:`G'` plateau: :math:`\Delta \log\omega \sim \log(N)`
- Number of inflection points in :math:`G'(\omega)` or :math:`G''(\omega)`

**Parsimonious Fitting:**

Start with :math:`N = 3`, increase until fit quality plateaus (adjusted :math:`R^2`, AIC). Overfitting risk if :math:`N > 10` for typical experimental noise.

Validating Rouse Scaling
-------------------------

**Test 1: Harmonic Time Spacing**

Plot fitted :math:`\tau_{R,k}` vs. :math:`k` on log-log axes. Expect slope :math:`-2` if ideal Rouse:

.. math::

   \log(\tau_{R,k}) = \log(\tau_{R,1}) - 2\log(k)

**Test 2: High-Frequency Scaling**

In Rouse regime (:math:`\omega \tau_{R,1} \gg 1`), verify:

.. math::

   \log G'(\omega) \sim \frac{1}{2} \log\omega + \text{const}

Slope of 0.5 on log-log plot confirms Rouse dynamics.

**Test 3: Equal Mode Strengths**

Check if :math:`G_k \approx G_N^{(0)}/N` for all modes. Deviation indicates non-ideal distribution (polydispersity, branching).

Molecular Weight Estimation
----------------------------

From Rouse theory, the longest Rouse time scales as:

.. math::

   \tau_{R,1} \sim M_w^2

where :math:`M_w` is weight-average molecular weight. If :math:`\tau_{R,1}` known:

.. math::

   M_w \propto \sqrt{\tau_{R,1}}

(Requires calibration with known standards.)

Sticker Binding Energy
-----------------------

If :math:`\tau_s` measured at multiple temperatures :math:`T`:

.. math::

   \tau_s(T) = \tau_0 \exp\left(\frac{\Delta E_{bind}}{k_B T}\right)

Arrhenius plot of :math:`\log\tau_s` vs. :math:`1/T` yields binding energy :math:`\Delta E_{bind}` from slope.

Discriminating Material Classes
--------------------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Observable
     - Sticky Rouse
     - Alternative Mechanism
   * - :math:`G'(\omega) \sim \omega^{1/2}` at high :math:`\omega`
     - Rouse modes active
     - Glassy/entangled: :math:`G' \sim \omega^0` (plateau)
   * - :math:`G'` plateau at intermediate :math:`\omega`
     - Sticky network
     - Chemical gel: permanent plateau
   * - Stress overshoot in startup
     - Sticker stretching (:math:`Wi_1 > 1`)
     - Shear banding, yield stress (no overshoot)
   * - Multi-exponential :math:`G(t)`
     - Multiple modes
     - Single Maxwell: mono-exponential

----

Experimental Design
===================

Optimal Test Protocols
-----------------------

**Primary: Small-Amplitude Oscillatory Shear (SAOS)**

Cover frequency range :math:`10^{-3}` to :math:`10^2` rad/s (at least 5 decades):

- **Low** :math:`\omega`: Terminal regime (:math:`G' \sim \omega^2`, :math:`G'' \sim \omega`)
- **Intermediate** :math:`\omega`: Sticky plateau (:math:`G' \approx G_N^{(0)}`)
- **High** :math:`\omega`: Rouse regime (:math:`G' \sim \omega^{1/2}`)

**Strain Amplitude:** :math:`\gamma_0 = 0.01-0.1` (confirm linear regime via amplitude sweep).

**Secondary: Stress Relaxation**

Step strain :math:`\gamma_0 = 0.1-0.5`, measure :math:`G(t)` from :math:`10^{-2}` to :math:`10^4` s:

- Validates multi-exponential spectrum
- Direct access to :math:`\tau_{eff,k}` via exponential fitting
- Complementary to SAOS (covers same time scales in different representation)

**Tertiary: Steady Shear Flow Curve**

Measure :math:`\eta(\dot{\gamma})` from :math:`10^{-3}` to :math:`10^2` 1/s:

- Probes nonlinear regime (:math:`Wi_1 > 1`)
- Validates shear thinning predictions
- Tests multi-mode consistency (must match SAOS via Cox-Merz rule at low :math:`\dot{\gamma}`)

**Advanced: LAOS**

Strain sweeps at fixed :math:`\omega` (e.g., :math:`\omega = 1/\tau_s`):

- :math:`\gamma_0` from 0.1 to 10
- Extract :math:`G'_1, G''_1` (fundamental), :math:`G'_3, G''_3` (third harmonic)
- Quantifies nonlinear elasticity (chain stretching effects)

Time-Temperature Superposition
-------------------------------

**Applicability:**

Sticky Rouse is thermorheologically simple if:

1. :math:`\tau_s(T)` and :math:`\tau_{R,k}(T)` have the same activation energy
2. :math:`G_k` temperature-independent (or weakly dependent)

**Procedure:**

Measure :math:`G'(\omega, T)`, :math:`G''(\omega, T)` at multiple :math:`T` (e.g., 10-60°C in 10°C steps).

Shift horizontally to reference :math:`T_{ref}` using shift factor :math:`a_T`:

.. math::

   G'(\omega, T) \to G'(a_T \omega, T_{ref})

If successful, reveals extended frequency range (e.g., 8 decades from 5 temperatures).

**Extract Activation Energy:**

.. math::

   \log a_T = \frac{\Delta E_{a}}{R} \left(\frac{1}{T} - \frac{1}{T_{ref}}\right)

Sample Requirements
-------------------

**Geometry:**

- **Cone-plate** (preferred): Homogeneous shear, small sample volume, gap angle 0.04-0.1 rad
- **Parallel plates:** Large normal forces, edge effects at high :math:`\gamma_0`
- **Couette:** High-viscosity samples, but difficult LAOS interpretation

**Volume:** 0.5-2 mL (cone-plate), 5-10 mL (parallel plates)

**Loading:** Avoid air bubbles, ensure complete wetting of geometry

**Temperature Control:** ±0.1°C stability for TTS measurements

**Equilibration:** 5-10 minutes at each temperature before measurement

Data Quality Checks
--------------------

**Linearity Verification:**

Perform strain amplitude sweep at fixed :math:`\omega`:

- :math:`G', G''` should be constant for :math:`\gamma_0 < \gamma_{LVE}`
- Typical :math:`\gamma_{LVE} \sim 0.1-1` for sticky Rouse systems

**Instrument Compliance:**

At high :math:`\omega`, check for artifacts:

- :math:`G''` should not exceed :math:`\omega \eta_s + G''_{max}` (solvent limit)
- Spurious peaks in :math:`G''` indicate inertia effects

**Torque Range:**

Ensure measured torque :math:`> 10\%` of instrument minimum for accurate data.

**Repeatability:**

Replicate SAOS at reference condition; coefficient of variation should be :math:`< 5\%`.

----

Computational Implementation
=============================

State Vector and ODE System
----------------------------

For :math:`N` modes, the state vector has dimension :math:`4N`:

.. math::

   \mathbf{y} = [S_{xx,1}, S_{yy,1}, S_{zz,1}, S_{xy,1}, \ldots, S_{xx,N}, S_{yy,N}, S_{zz,N}, S_{xy,N}]^T

**ODE Right-Hand Side:**

For mode :math:`k`, with velocity gradient :math:`\kappa`:

.. math::

   \frac{d\mathbf{y}_k}{dt} = \mathbf{f}_k(\mathbf{y}_k, \kappa, \tau_{eff,k})

where :math:`\mathbf{y}_k = [S_{xx,k}, S_{yy,k}, S_{zz,k}, S_{xy,k}]^T` and:

.. math::

   \mathbf{f}_k = \begin{bmatrix}
   2\kappa_{xy} S_{xy,k} - (S_{xx,k} - 1)/\tau_{eff,k} \\
   - (S_{yy,k} - 1)/\tau_{eff,k} \\
   - (S_{zz,k} - 1)/\tau_{eff,k} \\
   \kappa_{xy} S_{yy,k} - S_{xy,k}/\tau_{eff,k}
   \end{bmatrix}

**Vectorization via vmap:**

Use JAX `vmap` to parallelize over modes:

.. code-block:: python

   def ode_single_mode(y_k, kappa, tau_eff_k):
       S_xx, S_yy, S_zz, S_xy = y_k
       dS_xx = 2*kappa_xy*S_xy - (S_xx - 1)/tau_eff_k
       dS_yy = - (S_yy - 1)/tau_eff_k
       dS_zz = - (S_zz - 1)/tau_eff_k
       dS_xy = kappa_xy*S_yy - S_xy/tau_eff_k
       return jnp.array([dS_xx, dS_yy, dS_zz, dS_xy])

   ode_all_modes = jax.vmap(ode_single_mode, in_axes=(0, None, 0))

Then call `ode_all_modes(y, kappa, tau_eff)` where `y` is (N, 4), `tau_eff` is (N,).

Stress Calculation
------------------

**Total Shear Stress:**

.. code-block:: python

   def compute_stress(y, G, eta_s, gamma_dot):
       S_xy = y[:, 3]  # Shape (N,)
       sigma_xy = jnp.sum(G * S_xy) + eta_s * gamma_dot
       return sigma_xy

**Normal Stress Differences:**

.. code-block:: python

   def compute_N1(y, G):
       S_xx = y[:, 0]
       S_yy = y[:, 1]
       N1 = jnp.sum(G * (S_xx - S_yy))
       return N1

SAOS Implementation
-------------------

Use analytical expressions for efficiency:

.. code-block:: python

   def saos(omega, G, tau_eff, eta_s):
       # G, tau_eff are arrays of length N
       omega_tau = omega[:, None] * tau_eff[None, :]  # (len(omega), N)
       G_prime = jnp.sum(G * omega_tau**2 / (1 + omega_tau**2), axis=1)
       G_double_prime = jnp.sum(G * omega_tau / (1 + omega_tau**2), axis=1) + omega * eta_s
       return G_prime, G_double_prime

Relaxation Modulus
------------------

.. code-block:: python

   def relaxation_modulus(t, G, tau_eff):
       exp_terms = jnp.exp(-t[:, None] / tau_eff[None, :])  # (len(t), N)
       G_t = jnp.sum(G * exp_terms, axis=1)
       return G_t

Startup Shear Simulation
-------------------------

.. code-block:: python

   def simulate_startup(gamma_dot, t_end, G, tau_eff, eta_s):
       y0 = jnp.tile(jnp.array([1.0, 1.0, 1.0, 0.0]), N)  # Equilibrium
       kappa = jnp.array([[0, gamma_dot], [0, 0]])

       def rhs(t, y):
           y_reshaped = y.reshape(N, 4)
           dy = ode_all_modes(y_reshaped, kappa, tau_eff)
           return dy.ravel()

       t_eval = jnp.linspace(0, t_end, 1000)
       solution = odeint(rhs, y0, t_eval)

       sigma_xy = jax.vmap(lambda y: compute_stress(y.reshape(N, 4), G, eta_s, gamma_dot))(solution)
       return t_eval, sigma_xy

Performance Optimization
------------------------

**JIT Compilation:**

Decorate all functions with `@jax.jit` for 10-100x speedups:

.. code-block:: python

   @jax.jit
   def ode_all_modes(y, kappa, tau_eff):
       ...

**Avoid Python Loops:**

Use `vmap`, `lax.scan`, or `lax.fori_loop` instead of explicit for-loops over modes.

**Precompute Constants:**

Calculate :math:`\tau_{eff,k} = \tau_{R,k} + \tau_s` once at initialization, not during ODE integration.

----

Fitting Guidance
================

Primary Data: SAOS
-------------------

**Why SAOS is Ideal:**

1. Analytical solution (no ODE integration)
2. Direct access to all modes via frequency sweep
3. High signal-to-noise ratio
4. Well-defined linear regime

**Objective Function:**

Minimize log-space error to balance :math:`G'` and :math:`G''`:

.. math::

   \mathcal{L} = \sum_i \left[\left(\log G'_{pred}(\omega_i) - \log G'_{data}(\omega_i)\right)^2 + \left(\log G''_{pred}(\omega_i) - \log G''_{data}(\omega_i)\right)^2\right]

**Parameter Bounds:**

- :math:`G_k \in (0.01 \cdot G''_{max}, 100 \cdot G''_{max})`
- :math:`\tau_{R,k} \in (0.01/\omega_{max}, 100/\omega_{min})`
- :math:`\tau_s \in (0.01/\omega_{max}, 100/\omega_{min})`
- :math:`\eta_s \in (0, 10 \cdot G''_{max}/\omega_{max})`

Constrained vs. Unconstrained Fitting
--------------------------------------

**Unconstrained (2N+2 parameters):**

Fit all :math:`G_k, \tau_{R,k}` independently plus :math:`\tau_s, \eta_s`.

- **Pros:** Maximum flexibility; captures non-ideal spectra
- **Cons:** High parameter count; risk of overfitting; non-unique solutions

**Constrained (4 parameters):**

Impose Rouse scaling:

.. math::

   G_k = \frac{G_N^{(0)}}{N}, \quad \tau_{R,k} = \frac{\tau_{R,1}}{k^2}

Fit only :math:`G_N^{(0)}, \tau_{R,1}, \tau_s, \eta_s`.

- **Pros:** Parsimonious; physically motivated; stable fits
- **Cons:** May not capture polydispersity or non-ideal behavior

**Recommended Strategy:**

1. Start with constrained fit (N=3-5)
2. If fit poor (:math:`R^2 < 0.95`), relax to unconstrained
3. Validate by checking if fitted :math:`\tau_{R,k}` obeys :math:`1/k^2` scaling

Initialization Strategy
-----------------------

**Step 1: Estimate Plateau Modulus**

.. math::

   G_N^{(0)} \approx \min_{\omega} G'(\omega) \quad \text{(sticky plateau value)}

**Step 2: Estimate Sticker Lifetime**

From peak in :math:`G''(\omega)`:

.. math::

   \tau_s \approx \frac{1}{\omega_{G''_{peak}}}

**Step 3: Estimate Longest Rouse Time**

From terminal slope in :math:`G'`:

.. math::

   \tau_{R,1} \approx \frac{1}{\omega_{terminal}} - \tau_s

**Step 4: Set Mode Strengths**

.. math::

   G_k = \frac{G_N^{(0)}}{N}, \quad \tau_{R,k} = \frac{\tau_{R,1}}{k^2}

**Step 5: Estimate Solvent Viscosity**

.. math::

   \eta_s \approx \frac{G''(\omega_{max})}{\omega_{max}}

Regularization and Constraints
-------------------------------

**Monotonicity:**

Enforce :math:`\tau_{eff,1} > \tau_{eff,2} > \cdots > \tau_{eff,N}` to prevent mode crossing.

**Positivity:**

All :math:`G_k, \tau_{R,k}, \tau_s, \eta_s > 0` (built into bounds).

**Smoothness Penalty:**

For unconstrained fits, add regularization term:

.. math::

   \mathcal{L}_{reg} = \mathcal{L} + \alpha \sum_{k=1}^{N-1} (G_{k+1} - G_k)^2

to discourage erratic mode strength variations.

Multi-Start Optimization
-------------------------

Due to multi-modal likelihood surface, use multiple initial guesses:

1. Random sampling within bounds (10-20 starts)
2. Latin hypercube sampling for parameter space coverage
3. Select solution with lowest :math:`\mathcal{L}` and physical consistency

Secondary Data: Relaxation
---------------------------

If :math:`G(t)` available, fit directly:

.. math::

   \mathcal{L} = \sum_i \left(\log G(t_i) - \log G_{pred}(t_i)\right)^2

**Advantages:**

- Analytical solution (fast)
- Exponentials easier to resolve than SAOS peaks

**Disadvantages:**

- Requires high dynamic range in :math:`G(t)` (6+ decades)
- Experimental drift at long times
- Edge effects in step strain

Validation Tests
----------------

**After fitting, check:**

1. **R-squared:** :math:`R^2 > 0.95` (0.99 for good fit)
2. **Residual Randomness:** Plot residuals vs. :math:`\omega`; should show no trends
3. **Rouse Scaling:** Plot :math:`\tau_{R,k}` vs. :math:`k` on log-log; expect slope -2
4. **Mode Strength Distribution:** :math:`G_k` should be similar order of magnitude
5. **Physical Bounds:** :math:`\eta_0 = \sum G_k \tau_{eff,k} + \eta_s` should match steady-shear viscosity
6. **Cross-Validation:** Predict startup shear using fitted parameters; compare to experiment

----

Usage Examples
==============

Basic SAOS Fitting
------------------

.. code-block:: python

   from rheojax.models.tnt import TNTStickyRouse
   from rheojax.core import RheoData
   import jax.numpy as jnp

   # Experimental SAOS data
   omega = jnp.logspace(-2, 2, 50)  # rad/s
   G_prime_data = ...  # Pa
   G_double_prime_data = ...  # Pa
   G_star = G_prime_data + 1j * G_double_prime_data

   # Create model with 5 Rouse modes
   model = TNTStickyRouse(n_modes=5)

   # Fit to SAOS data
   rheo_data = RheoData(x=omega, y=G_star, test_mode='oscillation')
   result = model.fit(rheo_data)

   print(f"R-squared: {result.r_squared:.4f}")
   print(f"Fitted parameters: {result.parameters}")

   # Extract sticker lifetime
   tau_s = result.parameters['tau_s']
   print(f"Sticker lifetime: {tau_s:.2e} s")

Constrained Rouse Scaling
--------------------------

.. code-block:: python

   # Enforce ideal Rouse mode structure
   model = TNTStickyRouse(n_modes=5, constrain_rouse_scaling=True)

   # Now only 4 free parameters: G_N0, tau_R1, tau_s, eta_s
   result = model.fit(rheo_data)

   # Check if constraint was beneficial
   print(f"Constrained R^2: {result.r_squared:.4f}")

Predicting Startup Shear
-------------------------

.. code-block:: python

   # After fitting to SAOS, predict startup shear response
   gamma_dot = 1.0  # 1/s
   t_startup = jnp.linspace(0, 100, 500)  # s

   # Simulate startup
   sigma_xy = model.predict(
       t_startup,
       test_mode='startup',
       gamma_dot=gamma_dot
   )

   # Plot stress growth
   import matplotlib.pyplot as plt
   plt.plot(t_startup, sigma_xy)
   plt.xlabel('Time (s)')
   plt.ylabel('Shear Stress (Pa)')
   plt.title(f'Startup Shear at gamma_dot = {gamma_dot} 1/s')
   plt.show()

Stress Relaxation
-----------------

.. code-block:: python

   # Predict relaxation modulus after step strain
   t_relax = jnp.logspace(-3, 3, 100)  # s
   G_t = model.predict(t_relax, test_mode='relaxation')

   # Plot on log-log scale
   plt.loglog(t_relax, G_t)
   plt.xlabel('Time (s)')
   plt.ylabel('G(t) (Pa)')
   plt.title('Stress Relaxation Modulus')
   plt.grid(which='both', alpha=0.3)
   plt.show()

LAOS Simulation
---------------

.. code-block:: python

   # Large-amplitude oscillatory shear
   gamma_0 = 1.0  # Strain amplitude
   omega_laos = 1.0  # rad/s
   n_cycles = 10

   t_laos = jnp.linspace(0, 2*jnp.pi*n_cycles/omega_laos, 1000)

   sigma_laos = model.predict(
       t_laos,
       test_mode='laos',
       gamma_0=gamma_0,
       omega=omega_laos
   )

   # Extract harmonics via FFT
   from rheojax.utils import extract_harmonics
   harmonics = extract_harmonics(t_laos, sigma_laos, omega_laos, n_harmonics=5)

   print(f"G'_1: {harmonics['G1_prime']:.2f} Pa")
   print(f"G'_3: {harmonics['G3_prime']:.2f} Pa")

Bayesian Inference
------------------

.. code-block:: python

   # Propagate uncertainty in fitted parameters
   result_bayesian = model.fit_bayesian(
       rheo_data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(
       result_bayesian.posterior_samples,
       credibility=0.95
   )

   print("95% Credible Intervals:")
   for param, (low, high) in intervals.items():
       print(f"  {param}: [{low:.2e}, {high:.2e}]")

   # Plot posterior distributions
   import arviz as az
   az.plot_pair(result_bayesian.posterior_samples, divergences=True)

Multi-Temperature TTS
---------------------

.. code-block:: python

   # Fit at multiple temperatures, extract activation energy
   from rheojax.transforms import Mastercurve

   temps = [20, 30, 40, 50, 60]  # °C
   datasets = [load_saos_data(T) for T in temps]

   # Apply TTS
   mc = Mastercurve(reference_temp=40, auto_shift=True)
   master_data, shift_factors = mc.transform(datasets)

   # Fit sticky Rouse to master curve
   model = TNTStickyRouse(n_modes=5)
   result = model.fit(master_data)

   # Extract activation energy from shift factors
   import numpy as np
   T_kelvin = np.array(temps) + 273.15
   log_aT = np.log(shift_factors)

   # Arrhenius fit: log(a_T) = E_a/R * (1/T - 1/T_ref)
   from scipy.stats import linregress
   slope, intercept, r_value, p_value, std_err = linregress(1/T_kelvin, log_aT)
   E_a = slope * 8.314  # J/mol (R = 8.314 J/(mol·K))

   print(f"Activation energy: {E_a/1000:.1f} kJ/mol")

----

Failure Mode: Terminal Flow
----------------------------

The sticky Rouse model always predicts eventual viscous flow at sufficiently long times
or low frequencies. When all stickers have released at least once (time :math:`\gg
\tau_{\text{term}}`), the chain loses all memory of its initial configuration and flows
as a viscous liquid.

**Physical signatures:**

- :math:`G'(\omega) \sim \omega^2` and :math:`G''(\omega) \sim \omega` at :math:`\omega
  \ll 1/\tau_{\text{term}}`
- Steady-state creep rate :math:`\dot{\gamma}_{ss} = \sigma_0/\eta_0`
- No residual elasticity (unlike multi-species with permanent bonds)

**Distinction from gel behavior:** If the material shows a low-frequency elastic plateau
(:math:`G'` does not decrease to zero), the sticky Rouse model is inappropriate. Consider
:ref:`model-tnt-multi-species` with a permanent bond component, or a yield-stress model.

----

See Also
========

**TNT Shared Reference:**

- :doc:`tnt_protocols` — Full protocol equations and numerical methods
- :doc:`tnt_knowledge_extraction` — Model identification and fitting guidance

**TNT Base Model:**

- :ref:`model-tnt-tanaka-edwards` — Base model (single-mode limit)

**Related TNT Variants:**

- :ref:`model-tnt-multi-species` — Discrete multi-mode comparison (arbitrary :math:`G_k, \tau_k`)
- :ref:`model-tnt-loop-bridge` — Two-species topology comparison

**Alternative Models:**

- :ref:`model-tnt-cates` — Living polymers (single effective mode rather than broad spectrum)

----

API Reference
=============

.. autoclass:: rheojax.models.tnt.TNTStickyRouse
   :members:
   :undoc-members:
   :show-inheritance:

----

References
==========

Foundational Papers
-------------------

1. **Leibler, L., Rubinstein, M., & Colby, R. H. (1991).**
   "Dynamics of reversible networks."
   *Macromolecules*, 24(16), 4701-4707.
   DOI: 10.1021/ma00016a034

   - Original sticky reptation model
   - Introduced concept of renormalized Rouse time

2. **Rouse, P. E. (1953).**
   "A theory of the linear viscoelastic properties of dilute solutions of coiling polymers."
   *Journal of Chemical Physics*, 21(7), 1272-1280.
   DOI: 10.1063/1.1699180

   - Classical Rouse model
   - Harmonic mode spacing :math:`\tau_p \propto 1/p^2`

3. **Rubinstein, M., & Semenov, A. N. (1998).**
   "Thermoreversible gelation in solutions of associating polymers. 2. Linear dynamics."
   *Macromolecules*, 31(4), 1386-1397.
   DOI: 10.1021/ma970617+

   - Multi-sticker network dynamics
   - Hierarchical relaxation theory

Experimental Validation
-----------------------

4. **Chen, Q., Tudryn, G. J., & Colby, R. H. (2013).**
   "Ionomer dynamics and the sticky Rouse model."
   *Journal of Rheology*, 57(5), 1441-1462.
   DOI: 10.1122/1.4818868

   - Sticky Rouse behavior in ionomer solutions
   - Power-law relaxation validation

5. **Baxandall, L. G. (1989).**
   "Dynamics of reversibly crosslinked chains."
   *Macromolecules*, 22(4), 1982-1988.
   DOI: 10.1021/ma00194a076

   - Early theoretical treatment of transient networks
   - Crosslink kinetics coupling to Rouse modes

Review Articles
---------------

6. **Rubinstein, M., & Colby, R. H. (2003).**
   *Polymer Physics.*
   Oxford University Press.
   ISBN: 978-0198520597

   - Chapter 9: Rouse model (pages 372-399)
   - Chapter 10: Sticky reptation (pages 431-450)

7. **Tanaka, F., & Edwards, S. F. (1992).**
   "Viscoelastic properties of physically crosslinked networks."
   *Macromolecules*, 25(5), 1516-1523.
   DOI: 10.1021/ma00031a024

   - Green-Tobolsky network theory
   - Transient crosslink statistics

Computational Methods
---------------------

8. **Padding, J. T., & Briels, W. J. (2001).**
   "Uncrossability constraints in mesoscopic polymer melt simulations."
   *Journal of Chemical Physics*, 115(6), 2846-2859.
   DOI: 10.1063/1.1385162

   - Numerical integration of multi-mode constitutive equations
   - Stability analysis for stiff ODE systems

9. **Morrison, F. A. (2001).**
   *Understanding Rheology.*
   Oxford University Press.
   ISBN: 978-0195141665

   - Chapter 8: Multi-mode models (pages 441-488)
   - SAOS vs. transient flow predictions

Applications
------------

10. **Annable, T., Buscall, R., Ettelaie, R., & Whittlestone, D. (1993).**
    "The rheology of solutions of associating polymers."
    *Journal of Rheology*, 37(4), 695-726.
    DOI: 10.1122/1.550391

    - HEUR associating polymer rheology
    - Multi-mode sticky Rouse fits to experimental data
