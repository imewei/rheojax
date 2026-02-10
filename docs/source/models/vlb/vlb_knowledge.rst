.. _vlb_knowledge:

======================================
VLB — What You Can Learn
======================================

This guide explains how to extract physical insights from VLB model parameters,
how to interpret fitting results across protocols, and how to use the model for
material design and quality control.

For notation and governing equations, see :doc:`vlb`.


Parameter Interpretation
========================

Network Modulus :math:`G_0`
---------------------------

The network modulus encodes the density of elastically active chains:

.. math::

   G_0 = c k_B T

where :math:`c` is the number density of active chains and :math:`k_B T` is
the thermal energy (:math:`\approx 4.1 \times 10^{-21}` J at room temperature).

**Calculating chain density:**

.. math::

   c = \frac{G_0}{k_B T} \approx \frac{G_0}{4.1 \times 10^{-21}} \text{ chains/m}^3

.. list-table:: Typical :math:`G_0` by Material
   :widths: 30 20 50
   :header-rows: 1

   * - Material
     - :math:`G_0` (Pa)
     - Physical Interpretation
   * - Dilute hydrogels
     - 10 - 100
     - Low cross-link density; soft, swollen networks
   * - PVA-borax gels
     - 100 - 1000
     - Moderate density, dynamic boronates
   * - Telechelic polymers
     - 1000 - 10\ :sup:`4`
     - Concentrated end-functionalized chains
   * - Vitrimers
     - 10\ :sup:`5` - 10\ :sup:`6`
     - High cross-link density, stiff network
   * - Elastomers (comparison)
     - 10\ :sup:`5` - 10\ :sup:`7`
     - Permanent networks (VLB limit :math:`k_d \to 0`)


.. _vlb-kd-regimes:

Dissociation Rate :math:`k_d`
-------------------------------

The dissociation rate controls the network lifetime and relaxation:

.. list-table:: Kinetic Regimes
   :widths: 22 22 56
   :header-rows: 1

   * - :math:`k_d` (1/s)
     - Bond Lifetime
     - Regime
   * - :math:`< 10^{-4}`
     - > hours
     - Quasi-permanent.  Effectively elastic on experimental timescales.  Vitrimer-like.
   * - :math:`10^{-4}` — :math:`10^{-1}`
     - seconds to hours
     - Slow exchange.  Self-healing with long recovery times.  Thermo-activated exchange.
   * - :math:`10^{-1}` — :math:`10^1`
     - 0.1 - 10 s
     - Dynamic gel.  Active bond exchange.  Typical associating polymers.
   * - :math:`10^1` — :math:`10^3`
     - ms
     - Fast exchange.  Liquid-like terminal behavior.  Concentrated micelles.
   * - :math:`> 10^3`
     - < ms
     - Ultra-fast exchange.  Near-Newtonian.  Dilute associating solutions.


Derived Quantities
------------------

**Zero-shear viscosity** :math:`\eta_0 = G_0/k_d`:

Sensitive to both parameters. An increase in :math:`G_0` or decrease in
:math:`k_d` raises viscosity. Compare with rotational viscometry for
consistency.

**Crossover frequency** :math:`\omega_c = k_d`:

The SAOS crossover gives a direct read of :math:`k_d` without fitting — simply
identify the frequency where :math:`G' = G''`.

**Weissenberg number** :math:`\text{Wi} = \dot{\gamma}/k_d`:

Quantifies relative importance of deformation vs relaxation.  :math:`\text{Wi} > 1`
means the material is being deformed faster than it can relax.

**Pressure (normal force data):**

In incompressible materials the pressure :math:`p` is a Lagrange multiplier
determined by boundary conditions.  For confined geometries (parallel plates,
cone-and-plate), the thrust force :math:`F_N` is related to the first normal
stress difference:

.. math::

   F_N = \frac{\pi R^2}{2} N_1 = \frac{\pi R^2}{2} G_0(\mu_{xx} - \mu_{yy})

where :math:`R` is the plate radius.  If normal force data is available, this
provides an independent check on :math:`G_0` and the steady-state
Weissenberg number.


Diagnostic Signatures
=====================

Single-Exponential Relaxation
-----------------------------

**Expected:** :math:`\ln G(t)` is linear with slope :math:`-k_d`.

**If deviation:**

- **Upward curvature** (slower-than-exponential at long times):
  residual permanent network (:math:`G_e > 0`), use VLBMultiNetwork
- **Downward curvature** (faster-than-exponential):
  multiple relaxation times, use VLBMultiNetwork with M > 1
- **Power-law tail**: not Maxwell-like; consider fractional models (FMG, FZSS)

Newtonian Flow Curve
---------------------

**Expected:** :math:`\sigma \propto \dot{\gamma}` (constant viscosity).

**If deviation:**

- **Shear thinning**: :math:`k_d` increases with stress → need force-dependent
  :math:`k_d` (Bell model, see :doc:`vlb_advanced`)
- **Shear thickening**: formation-enhanced kinetics
  (stretch-creation model, TNT family)
- **Yield stress**: permanent network component or DMT/Fluidity model needed

SAOS Semicircle
---------------

**Expected:** Cole-Cole plot (:math:`G'` vs :math:`G''`) is a semicircle.

**If deviation:**

- **Depressed semicircle**: broadened relaxation spectrum,
  use VLBMultiNetwork
- **Multiple arcs**: well-separated relaxation modes
- **High-frequency uptick**: solvent contribution (:math:`\eta_s > 0`)

Monotonic Startup
-----------------

**Expected:** :math:`\sigma_{12}(t)` monotonically increases to steady state.

**If stress overshoot observed:**

- Constant-:math:`k_d` VLB cannot produce overshoot
- Indicates force-dependent breakage (Bell model)
- Or structure-dependent kinetics (DMT model)


Multi-Network Spectrum Analysis
================================

When fitting VLBMultiNetwork, the relaxation spectrum
:math:`\{(G_I, k_{d,I})\}` reveals network structure:

Spectrum Interpretation
-----------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Spectrum Feature
     - Physical Interpretation
   * - Single dominant mode
     - Narrow bond lifetime distribution; well-defined network
   * - Two well-separated modes
     - Two distinct bond types (e.g., hydrogen bonds + ionic cross-links)
   * - Broad spectrum (many modes)
     - Polydisperse bond lifetimes; heterogeneous network
   * - :math:`G_e > 0`
     - Permanent cross-links alongside reversible ones
   * - Large :math:`\eta_s`
     - Significant un-networked polymer contribution

Mode Assignment Strategy
------------------------

1. **Sort modes by** :math:`k_d`:  fastest :math:`k_d` = most labile bonds
2. **Compare** :math:`G_I`:  largest :math:`G_I` = most abundant bond population
3. **Cross-reference** with chemistry:  match timescales to known bond kinetics
   (e.g., boronate esters: :math:`k_d \sim 0.1` s\ :sup:`-1`;
   H-bonds: :math:`k_d \sim 10^3` s\ :sup:`-1`)


Application Examples
====================

Hydrogel Design
---------------

**Goal:** Tune self-healing rate and modulus.

**Approach:**

1. Fit SAOS data with VLBLocal to extract :math:`G_0` and :math:`k_d`
2. :math:`G_0` controls stiffness → adjust cross-linker concentration
3. :math:`k_d` controls healing time → modify cross-linker chemistry
   (e.g., catechol-metal: slow, boronate: moderate, host-guest: fast)

**Quality metric:** Self-healing efficiency
:math:`\propto k_d \cdot t_{heal}` → higher :math:`k_d` means faster healing
but lower toughness (trade-off).

Vitrimer Characterization
--------------------------

**Goal:** Determine exchange kinetics from rheology.

**Approach:**

1. Perform stress relaxation at multiple temperatures
2. Fit VLBLocal at each :math:`T` to extract :math:`k_d(T)`
3. Plot :math:`\ln k_d` vs :math:`1/T` → Arrhenius activation energy

.. math::

   k_d(T) = k_d^0 \exp\!\left(-\frac{E_a}{k_B T}\right)

The activation energy :math:`E_a` characterizes the bond exchange mechanism.

Telechelic Network Diagnostics
-------------------------------

**Goal:** Distinguish loop fraction from bridge fraction.

**Approach:**

1. Fit SAOS with VLBMultiNetwork (2 modes)
2. Faster mode → loop relaxation (non-load-bearing)
3. Slower mode → bridge relaxation (load-bearing)
4. :math:`G_{bridge}/G_{total}` estimates the bridge fraction

This is complementary to TNTLoopBridge, which models loop-bridge kinetics
explicitly.

Batch Quality Control
---------------------

**Goal:** Detect batch-to-batch variations in cross-link density.

**Approach:**

1. Establish baseline :math:`G_0^{ref}, k_d^{ref}` from a reference batch
2. Fit each new batch with VLBLocal
3. Flag deviations:

   - :math:`G_0/G_0^{ref} < 0.9`: under-crosslinked
   - :math:`G_0/G_0^{ref} > 1.1`: over-crosslinked
   - :math:`k_d/k_d^{ref} > 1.5`: accelerated degradation
   - :math:`k_d/k_d^{ref} < 0.5`: kinetic trapping


.. _vlb-cross-protocol-validation:

Cross-Protocol Validation Workflow
====================================

A robust characterization uses multiple protocols to validate the model:

**Step 1: SAOS (primary)**

- Extract :math:`G_0, k_d` from crossover
- Verify Cole-Cole semicircle

**Step 2: Stress relaxation (validation)**

- Verify :math:`G(0) = G_0` from SAOS
- Verify exponential decay with slope :math:`-k_d`

**Step 3: Startup shear (validation)**

- Verify :math:`\sigma^{ss} = \eta_0 \dot{\gamma}` matches flow curve
- No stress overshoot (constant :math:`k_d`)

**Step 4: Creep (optional)**

- Verify :math:`J(0) = 1/G_0`
- Verify :math:`dJ/dt = 1/\eta_0`

**Consistency check:**

.. math::

   \eta_{SAOS} = \lim_{\omega \to 0} \frac{G''}{\omega} \stackrel{?}{=}
   \eta_{flow} = \frac{\sigma}{\dot{\gamma}} \stackrel{?}{=}
   \eta_{creep} = \frac{1}{G_0 \cdot dJ/dt}

If all three agree, the constant-:math:`k_d` VLB model is appropriate.
Discrepancies indicate rate-dependent structure or non-Maxwell behavior.


When VLB Is Not Enough
======================

The constant-:math:`k_d` VLB model has clear limitations.  Here is how to
recognize them and which extension to consider:

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1

   * - Observation
     - VLB Prediction
     - Reality
     - Next Step
   * - Shear thinning
     - Newtonian
     - :math:`\eta \propto \dot{\gamma}^{n-1}`
     - Force-dependent :math:`k_d` (Bell)
   * - Stress overshoot
     - Monotonic
     - Overshoot at high Wi
     - Bell :math:`k_d` or DMT
   * - LAOS harmonics in :math:`\sigma_{12}`
     - :math:`I_3/I_1 = 0`
     - :math:`I_3/I_1 > 0`
     - Nonlinear :math:`k_d` or FENE
   * - Extensional hardening
     - Singularity at :math:`\dot{\varepsilon} = k_d/2`
     - Bounded growth
     - Langevin finite extensibility
   * - Aging
     - Time-independent
     - Properties change at rest
     - DMT or Fluidity-Saramito
   * - Power-law relaxation
     - Single exponential
     - :math:`G(t) \propto t^{-\alpha}`
     - Fractional models (FMG, FZSS)
   * - Shear banding
     - Homogeneous
     - Banded profiles
     - :class:`~rheojax.models.vlb.VLBNonlocal`
   * - Vitrimer BER kinetics
     - N/A (no evolving natural state)
     - Associative exchange, topology rearrangement
     - :doc:`/models/hvm/index` (HVM)
   * - NP-filled vitrimer
     - N/A (no filler effects)
     - Payne effect, dual freezing temperatures
     - :doc:`/models/hvnm/index` (HVNM)
