.. _model-tnt-fene-p:

===========================================================
TNT FENE-P (Finite Extensibility) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

----

Quick Reference
===============

.. list-table:: TNT FENE-P at a Glance
   :widths: 25 75
   :header-rows: 0

   * - **Use when**
     - Networks where chains have finite extensibility (strain stiffening at large deformations, rubber-like materials, elastomers with physical crosslinks)
   * - **Parameters**
     - 4 parameters: :math:`G` (modulus, Pa), :math:`\tau_b` (bond lifetime, s), :math:`L_{\text{max}}` (maximum extensibility, dimensionless), :math:`\eta_s` (solvent viscosity, Pa·s)
   * - **Key equation**
     - FENE-P stress: :math:`\boldsymbol{\sigma} = G \, f(\mathbf{S}) \, (\mathbf{S} - \mathbf{I}) + 2\eta_s \mathbf{D}` where :math:`f(\mathbf{S}) = L_{\text{max}}^2 / (L_{\text{max}}^2 - \text{tr}(\mathbf{S}))`
   * - **Test modes**
     - All 6 protocols: FLOW_CURVE, OSCILLATION, STARTUP, CREEP, RELAXATION, LAOS
   * - **Material examples**
     - Rubber networks, highly extensible hydrogels, FENE polymers, silicone elastomers, polyacrylamide hydrogels, PDMS networks
   * - **Key characteristics**
     - Strain stiffening, bounded chain extension, stress saturation at high strain rates, non-Gaussian chain statistics

**When to choose FENE-P:**

- Strain stiffening observed in startup or LAOS experiments
- Materials with finite maximum chain extensibility
- Rubber-like networks with physical or chemical crosslinks
- Need to prevent unphysical infinite extension of chains

**When NOT to use:**

- Small-strain linear viscoelasticity only (use base Tanaka-Edwards)
- Rigid or semi-flexible polymers (assumptions of Gaussian chains break down)
- Materials showing only strain softening (consider non-affine or Bell variants)

----

Notation Guide
==============

.. list-table:: Mathematical Symbols
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Type
     - Meaning
   * - :math:`\mathbf{S}`
     - Tensor
     - Conformation tensor (dimensionless, measures chain end-to-end vector)
   * - :math:`G`
     - Scalar
     - Network modulus (Pa)
   * - :math:`\tau_b`
     - Scalar
     - Bond lifetime / relaxation time (s)
   * - :math:`L_{\text{max}}`
     - Scalar
     - Maximum chain extensibility (dimensionless, :math:`L_{\text{max}}^2 = N_K`)
   * - :math:`f(\mathbf{S})`
     - Scalar
     - FENE-P spring function (dimensionless)
   * - :math:`\text{tr}(\mathbf{S})`
     - Scalar
     - Trace of conformation tensor :math:`S_{xx} + S_{yy} + S_{zz}`
   * - :math:`\eta_s`
     - Scalar
     - Solvent viscosity (Pa·s)
   * - :math:`R_{\text{max}}`
     - Scalar
     - Maximum chain end-to-end distance (m)
   * - :math:`R_0`
     - Scalar
     - Equilibrium chain end-to-end distance (m)
   * - :math:`N_K`
     - Scalar
     - Number of Kuhn segments per chain (dimensionless)
   * - :math:`\boldsymbol{\kappa}`
     - Tensor
     - Velocity gradient tensor :math:`\nabla \mathbf{v}`
   * - :math:`\mathbf{D}`
     - Tensor
     - Rate-of-strain tensor :math:`(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2`
   * - :math:`\boldsymbol{\sigma}`
     - Tensor
     - Cauchy stress tensor (Pa)
   * - :math:`\gamma_0`
     - Scalar
     - Strain amplitude (dimensionless)
   * - :math:`\omega`
     - Scalar
     - Angular frequency (rad/s)
   * - :math:`\text{Wi}`
     - Scalar
     - Weissenberg number :math:`\text{Wi} = \tau_b \dot{\gamma}`

----

Overview
========

Physical Motivation
-------------------

The **TNT FENE-P** model extends the classical Tanaka-Edwards transient network theory by incorporating **finite chain extensibility** through the FENE-P (Finitely Extensible Nonlinear Elastic with Peterlin closure) spring law. Real polymer chains have a finite contour length and cannot extend indefinitely — as the chain approaches its maximum extension, the restoring force diverges, preventing unphysical behavior.

**Historical development:**

1. **Warner (1972)**: Introduced the FENE spring model to describe finite extensibility in polymer kinetic theory.
2. **Peterlin (1966)**: Developed the Peterlin closure approximation for treating non-Gaussian chain statistics.
3. **Bird, Dotson, Johnson (1980)**: Formulated the FENE-P constitutive equation for dilute polymer solutions.
4. **Tanaka & Edwards (1992)**: Transient network theory for physical gels and associating polymers.
5. **Integration**: FENE-P closure combined with transient network framework to model extensible networks.

Key Distinctions from Base Model
---------------------------------

The base Tanaka-Edwards model assumes **Hookean (Gaussian) springs**, valid only at small chain extensions. The FENE-P variant introduces:

- **Nonlinear spring function** :math:`f(\mathbf{S})` that diverges as :math:`\text{tr}(\mathbf{S}) \to L_{\text{max}}^2`
- **Strain stiffening** at large deformations (stress increases faster than linear)
- **Bounded chain extension** preventing unphysical infinite stretch
- **Stress saturation** at high strain rates due to maximum chain extension

**Comparison:**

.. list-table:: Gaussian vs FENE-P Spring Laws
   :widths: 30 35 35
   :header-rows: 1

   * - Property
     - Gaussian (Base)
     - FENE-P
   * - Spring function
     - :math:`f = 1`
     - :math:`f = L_{\text{max}}^2 / (L_{\text{max}}^2 - \text{tr}(\mathbf{S}))`
   * - Small extension
     - Linear
     - Linear (recovers Gaussian)
   * - Large extension
     - Unbounded
     - Diverges (stiffens)
   * - Stress
     - :math:`\sigma \propto S`
     - :math:`\sigma \propto f(S) \cdot S`
   * - Material
     - Weak gels
     - Rubber networks, elastomers

----

Physical Foundations
====================

Gaussian vs Non-Gaussian Chain Statistics
------------------------------------------

A freely-jointed chain with :math:`N_K` Kuhn segments of length :math:`b_K` has:

- **Equilibrium end-to-end distance**: :math:`R_0 = b_K \sqrt{N_K}`
- **Maximum contour length**: :math:`R_{\text{max}} = N_K b_K`
- **Extensibility parameter**: :math:`L_{\text{max}}^2 = R_{\text{max}}^2 / R_0^2 = N_K`

At small extensions :math:`R \ll R_{\text{max}}`, the chain behaves as a **Hookean spring** (Gaussian statistics):

.. math::

   F \approx \frac{3 k_B T}{R_0^2} R

As :math:`R \to R_{\text{max}}`, the exact force (from inverse Langevin function :math:`\mathcal{L}^{-1}`) diverges:

.. math::

   F = \frac{k_B T}{b_K} \mathcal{L}^{-1}\left(\frac{R}{R_{\text{max}}}\right)

The **FENE-P approximation** replaces this with a tractable form:

.. math::

   F \approx \frac{3 k_B T}{R_0^2} \frac{L_{\text{max}}^2}{L_{\text{max}}^2 - R^2/R_0^2} R

This is the Peterlin closure — it captures the divergence at maximum extension while remaining analytically tractable.

Interpretation of :math:`L_{\text{max}}`
-----------------------------------------

The dimensionless extensibility parameter :math:`L_{\text{max}}` has direct physical meaning:

.. math::

   L_{\text{max}}^2 = \frac{R_{\text{max}}^2}{R_0^2} = N_K

where :math:`N_K` is the **number of Kuhn segments** in the chain.

**Examples:**

- **Short chains** (:math:`N_K \sim 10`, :math:`L_{\text{max}} \sim 3`): Strongly non-Gaussian, early strain stiffening
- **Moderate flexibility** (:math:`N_K \sim 100`, :math:`L_{\text{max}} \sim 10`): Typical for flexible polymers
- **Long chains** (:math:`N_K \sim 1000`, :math:`L_{\text{max}} \sim 30`): Nearly Gaussian until very high strains

From molecular weight:

.. math::

   L_{\text{max}} \approx \sqrt{\frac{M_w}{M_K}}

where :math:`M_w` is the chain molecular weight and :math:`M_K` is the Kuhn monomer mass.

Mechanical Analog
-----------------

The FENE-P spring can be visualized as a **nonlinear spring with hardening**:

- At rest, the spring constant is :math:`k_0 = 3 k_B T / R_0^2`
- As extension :math:`R` increases, the effective spring constant increases: :math:`k_{\text{eff}} = k_0 f(R)`
- At maximum extension, :math:`k_{\text{eff}} \to \infty` (infinite stiffness prevents further extension)

This is analogous to a "rubber band" that becomes progressively harder to stretch.

Representative Materials
-------------------------

.. list-table:: Materials Exhibiting FENE-P Behavior
   :widths: 30 70
   :header-rows: 1

   * - Material Class
     - Examples
   * - Natural rubber
     - Vulcanized rubber, latex networks
   * - Silicone elastomers
     - PDMS networks, silicone gels
   * - Hydrogels
     - Polyacrylamide (PAAm), alginate gels with physical crosslinks
   * - Thermoplastic elastomers
     - Styrene-butadiene-styrene (SBS), triblock copolymers
   * - Associating polymers
     - PEO with hydrophobic associating groups, telechelic polymers

----

Governing Equations
===================

FENE-P Spring Function
-----------------------

The central feature of the FENE-P model is the **nonlinear spring function**:

.. math::

   f(\mathbf{S}) = \frac{L_{\text{max}}^2}{L_{\text{max}}^2 - \text{tr}(\mathbf{S})}

where :math:`\text{tr}(\mathbf{S}) = S_{xx} + S_{yy} + S_{zz}` is the trace of the conformation tensor.

**Properties:**

- :math:`f(\mathbf{I}) = L_{\text{max}}^2 / (L_{\text{max}}^2 - 3)` at equilibrium (isotropic state :math:`\mathbf{S} = \mathbf{I}`)
- :math:`f(\mathbf{S}) > 1` for :math:`\text{tr}(\mathbf{S}) > 3` (extension)
- :math:`f(\mathbf{S}) \to \infty` as :math:`\text{tr}(\mathbf{S}) \to L_{\text{max}}^2` (singularity at maximum extension)
- :math:`f(\mathbf{S}) \to 1` as :math:`L_{\text{max}} \to \infty` (recovers Gaussian limit)

Conformation Tensor Evolution
------------------------------

The conformation tensor :math:`\mathbf{S}` evolves according to:

.. math::

   \frac{D\mathbf{S}}{Dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   - \frac{1}{\tau_b} \left( f(\mathbf{S}) \mathbf{S} - \mathbf{I} \right)

where:

- :math:`D/Dt` is the material derivative
- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` is the velocity gradient tensor
- The term :math:`\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T` represents **affine deformation**
- The term :math:`-(1/\tau_b)(f(\mathbf{S})\mathbf{S} - \mathbf{I})` represents **bond breakage and reformation**

**Key difference from Gaussian model:** The breakage term is multiplied by :math:`f(\mathbf{S})`, making the effective relaxation time **strain-dependent**:

.. math::

   \tau_{\text{eff}}(\mathbf{S}) = \frac{\tau_b}{f(\mathbf{S})}

At high extension, :math:`f(\mathbf{S})` increases, so :math:`\tau_{\text{eff}}` decreases — chains relax faster when highly stretched.

Stress Tensor
-------------

The Cauchy stress tensor is:

.. math::

   \boldsymbol{\sigma} = G \, f(\mathbf{S}) \, (\mathbf{S} - \mathbf{I}) + 2\eta_s \mathbf{D} - p \mathbf{I}

where:

- :math:`G` is the network modulus (Pa)
- :math:`f(\mathbf{S})(\mathbf{S} - \mathbf{I})` is the **elastic stress** from chain extension
- :math:`2\eta_s \mathbf{D}` is the **solvent viscosity contribution**
- :math:`p` is the hydrostatic pressure (determined by incompressibility)

For **simple shear**, the shear stress is:

.. math::

   \sigma_{xy} = G \, f(\mathbf{S}) \, S_{xy} + \eta_s \dot{\gamma}

For **oscillatory shear**, only the elastic contribution matters (assuming :math:`\eta_s = 0`):

.. math::

   \sigma_{xy}(t) = G \, f(\mathbf{S}(t)) \, S_{xy}(t)

Steady Shear Flow
-----------------

In steady simple shear at rate :math:`\dot{\gamma}`, the components of :math:`\mathbf{S}` satisfy:

.. math::

   \dot{\gamma} S_{yy} - \frac{1}{\tau_b} (f S_{xy} - 0) &= 0 \\
   \dot{\gamma} (S_{xx} + S_{yy}) - \frac{1}{\tau_b} (f S_{xx} - 1) &= 0 \\
   -\frac{1}{\tau_b} (f S_{yy} - 1) &= 0 \\
   -\frac{1}{\tau_b} (f S_{zz} - 1) &= 0

where :math:`f = L_{\text{max}}^2 / (L_{\text{max}}^2 - (S_{xx} + S_{yy} + S_{zz}))`.

Unlike the Gaussian case, these equations are **implicit** and must be solved numerically (no closed-form solution exists).

**Limiting behavior:**

- **Low Wi** (:math:`\text{Wi} = \tau_b \dot{\gamma} \ll 1`): :math:`f \approx f_{\text{eq}} = L_{\text{max}}^2/(L_{\text{max}}^2 - 3)`, recovers Newtonian :math:`\eta \approx G \tau_b f_{\text{eq}}`
- **High Wi** (:math:`\text{Wi} \gg 1`): :math:`S_{xx}` grows, :math:`f` increases, **stress saturates** near :math:`G L_{\text{max}}^2`

Small Amplitude Oscillatory Shear (SAOS)
-----------------------------------------

For infinitesimal strain :math:`\gamma_0 \to 0`, linearize around equilibrium :math:`\mathbf{S} = \mathbf{I}`:

.. math::

   f_{\text{eq}} = \frac{L_{\text{max}}^2}{L_{\text{max}}^2 - 3}

The linearized ODE gives:

.. math::

   G'(\omega) &= G \, f_{\text{eq}} \, \frac{(\omega \tau_b)^2}{1 + (\omega \tau_b)^2} \\
   G''(\omega) &= G \, f_{\text{eq}} \, \frac{\omega \tau_b}{1 + (\omega \tau_b)^2}

**Effect of finite extensibility:**

- Moduli are scaled by :math:`f_{\text{eq}} > 1`
- For :math:`L_{\text{max}} = 10`: :math:`f_{\text{eq}} = 100/97 \approx 1.03` (3% increase)
- For :math:`L_{\text{max}} = 3`: :math:`f_{\text{eq}} = 9/6 = 1.5` (50% increase)
- Relaxation time remains :math:`\tau_b` (to leading order)

Startup of Shear Flow
----------------------

Following a step imposition of shear rate :math:`\dot{\gamma}` at :math:`t = 0`, the stress evolves as:

.. math::

   \sigma_{xy}(t) = G \, f(\mathbf{S}(t)) \, S_{xy}(t) + \eta_s \dot{\gamma}

where :math:`\mathbf{S}(t)` is obtained by numerically integrating the ODE.

**Characteristic features:**

1. **Initial linear growth**: :math:`\sigma \approx G f_{\text{eq}} \dot{\gamma} t` for :math:`t \ll \tau_b`
2. **Strain stiffening**: As :math:`\text{tr}(\mathbf{S})` increases, :math:`f` increases, leading to an **upturn** in stress
3. **Overshoot**: For :math:`\text{Wi} > 1`, stress peaks at :math:`t \sim \tau_b`, then relaxes
4. **Steady state**: Approaches steady-shear solution with :math:`f > f_{\text{eq}}`

The **FENE-P overshoot** can be more pronounced than Gaussian due to stiffening before the peak.

Stress Relaxation
-----------------

After cessation of flow from steady state at :math:`\text{Wi} = \text{Wi}_0`, the stress relaxes:

.. math::

   \sigma_{xy}(t) = \sigma_0 \exp\left(-\int_0^t \frac{f(\mathbf{S}(s))}{\tau_b} ds\right)

where :math:`\mathbf{S}(t)` evolves under :math:`\boldsymbol{\kappa} = 0`:

.. math::

   \frac{d\mathbf{S}}{dt} = -\frac{1}{\tau_b} (f(\mathbf{S}) \mathbf{S} - \mathbf{I})

**Key feature:** The relaxation is **faster than exponential** initially (while :math:`f > f_{\text{eq}}`), then slows down as :math:`\mathbf{S} \to \mathbf{I}`.

Effective relaxation time:

.. math::

   \tau_{\text{eff}}(t) = \frac{\tau_b}{f(\mathbf{S}(t))}

FENE Damping Function
^^^^^^^^^^^^^^^^^^^^^^

The step-strain damping function for the FENE-P TNT model is:

.. math::

   h(\gamma_0) \approx \frac{L_{\max}^2 - 3}{L_{\max}^2 - (2 + \gamma_0^2)}

For small strains (:math:`\gamma_0 \ll L_{\max}`), :math:`h \to 1` (linear regime). As
:math:`\gamma_0 \to \sqrt{L_{\max}^2 - 2}`, the damping function diverges — reflecting
the stress stiffening as chains approach maximum extension.

**Initial FENE factor after step strain:**

.. math::

   f_0 = \frac{L_{\max}^2}{L_{\max}^2 - (2 + \gamma_0^2)}

This produces a very fast initial relaxation for large :math:`\gamma_0` approaching
:math:`L_{\max}`, because the high initial FENE force drives rapid stress decay.

Creep and Recovery
------------------

Under constant stress :math:`\sigma_0`, the system evolves via 5 coupled ODEs (for :math:`S_{xx}, S_{yy}, S_{zz}, S_{xy}`, plus :math:`\gamma` from kinematics).

**Elastic contribution** (instantaneous jump):

.. math::

   \gamma_e(0^+) = \frac{\sigma_0}{G f_{\text{eq}}}

**Viscous flow:** As :math:`\mathbf{S}` evolves, the strain rate :math:`\dot{\gamma}(t) = \sigma_0 / \eta(t)` depends on the instantaneous viscosity:

.. math::

   \eta(t) = G \, f(\mathbf{S}(t)) \, \tau_b

Unlike the Gaussian case (constant :math:`\eta = G\tau_b`), the FENE-P viscosity **increases** with extension.

Creep Compliance Saturation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The FENE stress function prevents infinite chain extension, which manifests in creep as
**compliance saturation**: the creep compliance :math:`J(t)` levels off sharply as chains
approach :math:`L_{\max}`. Unlike the Hookean model where :math:`\gamma \to \infty` under
constant stress, FENE chains have a maximum recoverable strain set by :math:`L_{\max}`.

At high applied stress, the creep rate :math:`\dot{\gamma}(t)` shows a transient
acceleration followed by saturation — the FENE spring hardens and resists further
extension.

Large Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

For :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0 \gg 1`, the FENE-P nonlinearity generates **rich harmonic content**:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots} [G'_n(\gamma_0, \omega) \sin(n\omega t) + G''_n(\gamma_0, \omega) \cos(n\omega t)]

**Sources of nonlinearity:**

1. :math:`f(\mathbf{S})` varies cyclically with :math:`\mathbf{S}(t)`
2. Asymmetry between extension and compression (chains stiffen more in extension)

**Lissajous curves** (:math:`\sigma` vs :math:`\gamma`) show characteristic **stiffening loops** at large :math:`\gamma_0`.

----

Parameter Table
===============

.. list-table:: TNT FENE-P Parameters
   :widths: 15 15 20 15 35
   :header-rows: 1

   * - Symbol
     - Default
     - Bounds
     - Units
     - Description
   * - :math:`G`
     - 1000
     - :math:`(10^0, 10^8)`
     - Pa
     - Network modulus (plateau modulus)
   * - :math:`\tau_b`
     - 1.0
     - :math:`(10^{-6}, 10^4)`
     - s
     - Bond lifetime / relaxation time
   * - :math:`L_{\text{max}}`
     - 10.0
     - :math:`(2.0, 100)`
     - dimensionless
     - Maximum chain extensibility (:math:`\sqrt{N_K}`)
   * - :math:`\eta_s`
     - 0.0
     - :math:`(0, 10^4)`
     - Pa·s
     - Solvent viscosity (optional)

**Notes:**

- Lower bound :math:`L_{\text{max}} \ge 2` ensures FENE-P closure validity (minimum 4 Kuhn segments)
- For :math:`L_{\text{max}} > 50`, behavior is nearly Gaussian (little practical difference)
- :math:`\eta_s` is often negligible for dense networks, but important for dilute solutions

----

Parameter Interpretation
=========================

Effect of :math:`L_{\text{max}}`
---------------------------------

.. list-table:: Extensibility Regimes
   :widths: 20 30 50
   :header-rows: 1

   * - :math:`L_{\text{max}}`
     - Regime
     - Behavior
   * - :math:`\to \infty`
     - Gaussian limit
     - Recovers base Tanaka-Edwards model, no strain stiffening
   * - :math:`50 - 100`
     - Nearly Gaussian
     - Stiffening only at very high strains (:math:`\gamma > 10`)
   * - :math:`10 - 30`
     - Moderate extensibility
     - Typical for flexible polymers, stiffening at :math:`\gamma \sim 1-5`
   * - :math:`3 - 10`
     - Strongly non-Gaussian
     - Early onset of stiffening, significant FENE effects
   * - :math:`2 - 3`
     - Highly extensible limit
     - Dramatic stiffening, low chain flexibility

Connection to Chain Structure
------------------------------

From polymer physics:

.. math::

   L_{\text{max}}^2 = N_K = \frac{M_w}{M_K}

where:

- :math:`N_K` = number of Kuhn segments per chain
- :math:`M_w` = chain molecular weight (g/mol)
- :math:`M_K` = Kuhn monomer mass (g/mol)

**Example (polyacrylamide):**

- Kuhn length :math:`b_K \approx 2` nm
- Molecular weight :math:`M_w = 10^5` g/mol
- Monomer mass :math:`M_0 = 71` g/mol
- Kuhn monomer :math:`M_K \approx 2 \times M_0 = 142` g/mol
- :math:`N_K \approx 10^5 / 142 \approx 700`
- :math:`L_{\text{max}} \approx \sqrt{700} \approx 26`

Relationship Between Parameters
--------------------------------

**Plateau modulus and network density:**

.. math::

   G = \nu k_B T = \frac{\rho R T}{M_c}

where :math:`\nu` is the network strand density (chains/m³) and :math:`M_c` is the molecular weight between crosslinks.

**Zero-shear viscosity (Gaussian limit):**

.. math::

   \eta_0 = G \tau_b f_{\text{eq}} + \eta_s \approx G \tau_b \left(1 + \frac{3}{L_{\text{max}}^2 - 3}\right)

For large :math:`L_{\text{max}}`, :math:`f_{\text{eq}} \to 1`, giving :math:`\eta_0 \approx G\tau_b`.

**Strain at onset of stiffening:**

Heuristic estimate from :math:`f(\mathbf{S}) \approx 1.5 f_{\text{eq}}`:

.. math::

   \gamma_{\text{stiff}} \sim \frac{L_{\text{max}}}{\sqrt{3}}

For :math:`L_{\text{max}} = 10`, :math:`\gamma_{\text{stiff}} \sim 6`.

Dimensionless Groups
--------------------

.. list-table:: Key Dimensionless Parameters
   :widths: 30 30 40
   :header-rows: 1

   * - Group
     - Definition
     - Physical Meaning
   * - Weissenberg number
     - :math:`\text{Wi} = \tau_b \dot{\gamma}`
     - Ratio of relaxation to deformation timescales
   * - Deborah number
     - :math:`\text{De} = \tau_b \omega`
     - Ratio of relaxation to observation timescales
   * - Extensibility
     - :math:`L_{\text{max}}^2 = N_K`
     - Number of Kuhn segments
   * - Normalized trace
     - :math:`\text{tr}(\mathbf{S}) / L_{\text{max}}^2`
     - Fraction of maximum extension (must be < 1)

----

Validity and Assumptions
=========================

Peterlin Closure Approximation
-------------------------------

The FENE-P model uses the **Peterlin approximation**:

.. math::

   \langle \mathbf{R} \otimes \mathbf{R} \rangle \approx \frac{\langle \mathbf{R} \rangle \otimes \langle \mathbf{R} \rangle}{\langle R^2 \rangle / R_{\text{max}}^2}

This decouples the orientation and extension fluctuations. **Limitations:**

- Underestimates stress at very high extensions (compared to exact FENE or bead-spring simulations)
- Error increases as :math:`L_{\text{max}}` decreases
- For :math:`L_{\text{max}} \ge 10`, approximation is typically accurate to within 10-20%

Affine Deformation Assumption
------------------------------

The model assumes **affine deformation**: the conformation tensor :math:`\mathbf{S}` deforms with the macroscopic flow field.

**Breaks down when:**

- Network heterogeneity (non-affine rearrangements)
- Slip at crosslinks (see non-affine variant)
- Highly entangled systems

Mean-Field Averaging
--------------------

All chains are assumed to have the same conformation tensor :math:`\mathbf{S}`. In reality, there is a distribution of chain extensions.

**Consequences:**

- Overestimates the rate of stress increase in startup (real networks have dispersion)
- Cannot capture chain-length polydispersity effects

Valid for Flexible Polymers
----------------------------

The FENE-P model assumes **flexible, Gaussian-like chains** with many Kuhn segments.

**Not valid for:**

- Rigid rods (use wormlike chain models)
- Semi-flexible polymers with persistence length :math:`\ell_p \sim L` (use WLC-based models)
- Colloidal particles (use Brownian dynamics)

Constant Breakage Rate
-----------------------

Bond breakage rate :math:`1/\tau_b` is assumed **independent of force**.

**To incorporate force-dependent kinetics**, combine with the Bell variant:

.. math::

   \tau_b \to \tau_b(\mathbf{S}) = \tau_{b0} \exp\left(\frac{F(\mathbf{S})}{F_b}\right)

where :math:`F(\mathbf{S}) \propto f(\mathbf{S}) \sqrt{\text{tr}(\mathbf{S})}`.

----

Regimes and Behavior
====================

Small Strain Regime
-------------------

**Condition:** :math:`\text{tr}(\mathbf{S}) - 3 \ll L_{\text{max}}^2`

In this regime:

.. math::

   f(\mathbf{S}) \approx f_{\text{eq}} = \frac{L_{\text{max}}^2}{L_{\text{max}}^2 - 3}

The stress response is **nearly Gaussian**:

.. math::

   \sigma_{xy} \approx G f_{\text{eq}} S_{xy}

**Observable:** SAOS moduli are :math:`G' \approx G f_{\text{eq}} (\omega\tau_b)^2 / (1 + (\omega\tau_b)^2)`.

Moderate Strain Regime
----------------------

**Condition:** :math:`\text{tr}(\mathbf{S}) \sim L_{\text{max}}^2 / 2`

The FENE function begins to increase noticeably:

.. math::

   f(\mathbf{S}) \approx 2 f_{\text{eq}}

**Observable:** Onset of **strain stiffening** in startup experiments. The stress grows faster than :math:`t` (upward curvature in :math:`\sigma(t)`).

Near Maximum Extension Regime
------------------------------

**Condition:** :math:`L_{\text{max}}^2 - \text{tr}(\mathbf{S}) \sim O(1)`

The FENE function diverges:

.. math::

   f(\mathbf{S}) \sim \frac{L_{\text{max}}^2}{\epsilon}, \quad \epsilon \ll 1

**Stress saturation:**

.. math::

   \sigma_{\text{max}} \sim G L_{\text{max}}^2

**Observable:** Stress plateaus at high :math:`\text{Wi}` in steady shear. This is a **signature prediction** of finite extensibility.

**Warning:** In practice, sample failure (chain scission, network rupture) often occurs before reaching this limit.

Extensional Viscosity Bound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Unlike the upper-convected Maxwell (UCM) model which predicts divergent extensional
viscosity at :math:`Wi = 1/2`, the FENE-P model gives a **bounded** extensional viscosity:

.. math::

   \eta_E \leq 3\eta_0 \frac{L_{\max}^2}{L_{\max}^2 - 3}

This bound is approached as :math:`\dot{\epsilon} \to \infty`. For typical values
:math:`L_{\max} = 10`, the maximum extensional viscosity is approximately :math:`3.1 \eta_0`
— a physically realistic prediction.

Steady Shear Regimes
--------------------

.. list-table:: Flow Curve Regimes
   :widths: 20 40 40
   :header-rows: 1

   * - :math:`\text{Wi}`
     - Conformation
     - Viscosity
   * - :math:`\ll 1`
     - Isotropic (:math:`\mathbf{S} \approx \mathbf{I}`)
     - Newtonian :math:`\eta \approx G\tau_b f_{\text{eq}}`
   * - :math:`\sim 1`
     - Moderate extension
     - Shear thinning begins
   * - :math:`\gg 1`
     - High extension (:math:`S_{xx} \gg 1`)
     - Bounded :math:`\eta \sim G\tau_b L_{\text{max}}^2 / \text{Wi}`

**Key difference from Gaussian:** The FENE-P viscosity **saturates** at high :math:`\text{Wi}` rather than decreasing indefinitely.

Startup Regimes
---------------

For :math:`\text{Wi} > 1`, startup shows:

1. **Linear growth** (:math:`t \ll \tau_b`): :math:`\sigma \approx G f_{\text{eq}} \dot{\gamma} t`
2. **Strain stiffening** (:math:`t \sim \tau_b`): Stress curves upward as :math:`f` increases
3. **Overshoot** (:math:`t \sim 2\tau_b`): Stress peaks at :math:`\sigma_{\text{max}} > \sigma_{\text{ss}}`
4. **Relaxation to steady state** (:math:`t \gg \tau_b`): Stress decays to steady-shear value

**FENE-P overshoot** can be **larger** than Gaussian due to stiffening before the peak.

LAOS Regimes
------------

.. list-table:: LAOS Behavior vs Strain Amplitude
   :widths: 20 40 40
   :header-rows: 1

   * - :math:`\gamma_0`
     - Response
     - Harmonics
   * - :math:`\ll 1`
     - Linear (SAOS)
     - Only :math:`n=1`
   * - :math:`\sim 1`
     - Weakly nonlinear
     - Weak :math:`n=3,5`
   * - :math:`\gg 1`
     - Strongly nonlinear
     - Strong higher harmonics, stiffening loops

**Lissajous signature:** Clockwise loops in :math:`\sigma` vs :math:`\gamma` due to **strain stiffening** (elastic dominance).

LAOS Strain-Stiffening Signature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In LAOS, FENE stress function produces a distinctive **strain-stiffening** signature:

- **Lissajous curves** become rectangular (box-like) at large :math:`\gamma_0` — high
  stress at peak strain due to the FENE divergence as :math:`\text{tr}(\mathbf{S})
  \to L_{\max}^2`
- **Intra-cycle stiffening**: The tangent modulus :math:`G'_L` (large-strain modulus from
  Ewoldt decomposition) exceeds :math:`G'_M` (minimum-strain modulus)
- Higher harmonics grow with :math:`\gamma_0`, but the dominant signature is
  **stiffening** rather than the softening seen in Bell variants

----

What You Can Learn
==================

From SAOS Data
--------------

.. list-table:: Information from SAOS
   :widths: 40 60
   :header-rows: 1

   * - Observable
     - Extracted Parameter
   * - Plateau modulus :math:`G_0`
     - :math:`G \times f_{\text{eq}}` (need to decouple :math:`L_{\text{max}}` from nonlinear data)
   * - Crossover frequency :math:`\omega_c`
     - Relaxation time :math:`\tau_b \approx 1/\omega_c`
   * - Slope of :math:`\tan\delta` at low :math:`\omega`
     - No direct :math:`L_{\text{max}}` information (linearized model)

**Limitation:** :math:`L_{\text{max}}` is **not identifiable** from SAOS alone (only appears via :math:`f_{\text{eq}} \approx 1` correction).

From Startup and Flow Curves
-----------------------------

.. list-table:: Information from Nonlinear Shear
   :widths: 40 60
   :header-rows: 1

   * - Observable
     - Extracted Parameter
   * - Strain at stiffening onset
     - :math:`L_{\text{max}}` (via :math:`\gamma_{\text{stiff}} \sim L_{\text{max}}/\sqrt{3}`)
   * - Stress overshoot magnitude
     - Combined :math:`G, \tau_b, L_{\text{max}}` (stronger overshoot for low :math:`L_{\text{max}}`)
   * - Steady-state stress saturation
     - :math:`G \times L_{\text{max}}^2` (maximum network stress)
   * - High-Wi viscosity
     - Bounded value confirms FENE-P vs Gaussian

From LAOS Data
--------------

.. list-table:: Information from LAOS
   :widths: 40 60
   :header-rows: 1

   * - Observable
     - Extracted Parameter
   * - Strain amplitude for nonlinearity
     - :math:`L_{\text{max}}` (via :math:`\gamma_0 \sim L_{\text{max}}`)
   * - Lissajous loop shape
     - Stiffening vs softening (FENE-P gives stiffening)
   * - Higher harmonics :math:`I_{3/1}, I_{5/1}`
     - Degree of non-Gaussian behavior (larger for low :math:`L_{\text{max}}`)

Chain Architecture from :math:`L_{\text{max}}`
-----------------------------------------------

Once :math:`L_{\text{max}}` is known, infer chain structure:

.. math::

   N_K = L_{\text{max}}^2

If Kuhn length :math:`b_K` is known (from literature or scattering):

.. math::

   R_{\text{max}} = L_{\text{max}} \times b_K \sqrt{N_K} = L_{\text{max}}^2 b_K

If molecular weight is known:

.. math::

   M_K \approx \frac{M_w}{L_{\text{max}}^2}

**Example:** :math:`L_{\text{max}} = 10` from startup → :math:`N_K = 100` Kuhn segments.

Stress Saturation Level
------------------------

The maximum stress the network can sustain:

.. math::

   \sigma_{\text{max}} \approx G L_{\text{max}}^2

**Practical use:**

- Predict failure stress (if chains break before full extension)
- Estimate safe operating strain limits
- Design formulations to avoid rupture

----

Experimental Design
===================

Best Protocols for Determining :math:`L_{\text{max}}`
------------------------------------------------------

.. list-table:: Protocol Recommendations
   :widths: 30 30 40
   :header-rows: 1

   * - Protocol
     - :math:`\text{Wi}` or :math:`\gamma_0`
     - Observable
   * - Startup
     - :math:`\text{Wi} > 5`
     - Strain stiffening upturn before overshoot
   * - LAOS
     - :math:`\gamma_0 > 1`
     - Stiffening loops, higher harmonics
   * - Flow curve
     - :math:`\text{Wi} > 10`
     - Stress saturation at high :math:`\dot{\gamma}`

**Optimal strategy:** Combine SAOS (for :math:`G, \tau_b`) + Startup (for :math:`L_{\text{max}}`) + LAOS (for validation).

SAOS First
----------

**Purpose:** Obtain :math:`G` and :math:`\tau_b` from linear response.

**Protocol:**

- Frequency sweep: :math:`\omega = 10^{-2}` to :math:`10^2` rad/s
- Strain amplitude: :math:`\gamma_0 = 0.01` (linear regime)
- Extract :math:`G_0 \approx G f_{\text{eq}}` and :math:`\tau_b` from :math:`\omega_c`

Startup at Multiple Wi
-----------------------

**Purpose:** Capture strain stiffening and determine :math:`L_{\text{max}}`.

**Protocol:**

- Step shear rates: :math:`\dot{\gamma} = 0.1, 1, 10, 100` s⁻¹ (adjust to span :math:`\text{Wi} = 0.1` to :math:`100`)
- Duration: :math:`t = 10 \tau_b` (long enough for steady state)
- Look for **upward curvature** before overshoot

**Analysis:**

- Fit early-time slope to get :math:`G f_{\text{eq}}`
- Fit stiffening region to extract :math:`L_{\text{max}}`

LAOS for Validation
-------------------

**Purpose:** Confirm FENE-P nonlinearity, check for additional physics.

**Protocol:**

- Strain amplitudes: :math:`\gamma_0 = 0.1, 0.5, 1, 5, 10`
- Frequency: :math:`\omega = 1/\tau_b` (Deborah number :math:`\sim 1`)
- Extract Lissajous curves and higher harmonics

**Check:**

- Stiffening (clockwise) loops confirm FENE-P
- Softening (counterclockwise) loops suggest additional physics (non-affine, yielding)

Experimental Artifacts
----------------------

.. list-table:: Common Artifacts
   :widths: 30 70
   :header-rows: 1

   * - Artifact
     - Mitigation
   * - Edge fracture
     - Use cone-plate or parallel-plate with edge trim
   * - Sample expulsion
     - Lower strain amplitude, use sandblasted plates
   * - Chain scission
     - Limit maximum :math:`\gamma` or :math:`\dot{\gamma}`, check for hysteresis
   * - Wall slip
     - Use roughened surfaces, check with gap-dependent tests
   * - Inertia
     - Keep :math:`\rho \omega R^2 / \eta \ll 1` (low Reynolds number)

Complementary Techniques
-------------------------

.. list-table:: Complementary Measurements
   :widths: 30 70
   :header-rows: 1

   * - Technique
     - Information
   * - SANS/SAXS
     - Direct measurement of :math:`R_0, R_{\text{max}}`; validation of :math:`L_{\text{max}}`
   * - Birefringence
     - Chain orientation, validates :math:`\mathbf{S}` predictions
   * - Microscopy
     - Network heterogeneity, defects
   * - DLS
     - Relaxation time :math:`\tau_b`, diffusion coefficient

----

Computational Implementation
=============================

JAX Implementation
------------------

The FENE-P stress kernel is implemented in ``rheojax/models/tnt/_kernels.py`` as:

.. code-block:: python

   def stress_fene(S: Array, params: Array) -> Array:
       """FENE-P stress function.

       Args:
           S: Conformation tensor (3x3)
           params: [G, L_max]

       Returns:
           Stress tensor (3x3)
       """
       G, L_max = params
       tr_S = jnp.trace(S)

       # Clamp trace to prevent singularity
       epsilon = 1e-6
       tr_S_safe = jnp.minimum(tr_S, L_max**2 - epsilon)

       f = L_max**2 / (L_max**2 - tr_S_safe)
       sigma = G * f * (S - jnp.eye(3))
       return sigma

FENE Singularity Handling
--------------------------

The key numerical challenge is the **singularity** :math:`f \to \infty` as :math:`\text{tr}(\mathbf{S}) \to L_{\text{max}}^2`.

**Clamping strategy:**

.. code-block:: python

   epsilon = 1e-6  # Safety margin
   tr_S_safe = jnp.minimum(tr_S, L_max**2 - epsilon)

This prevents division by zero while allowing :math:`f` to become very large (up to :math:`\sim L_{\text{max}}^2 / \epsilon`).

**Alternative:** Use a **smoothed FENE** function:

.. math::

   f_{\text{smooth}}(\mathbf{S}) = \frac{L_{\text{max}}^2}{L_{\text{max}}^2 - \text{tr}(\mathbf{S}) + \epsilon}

This avoids sharp cutoffs but changes the physics slightly at high extension.

ODE Stiffness
-------------

Near the FENE singularity, the ODE becomes **stiff** due to rapid changes in :math:`f(\mathbf{S})`.

**Solution:** Use adaptive step-size control in Diffrax:

.. code-block:: python

   from diffrax import diffeqsolve, Dopri5, PIDController

   solver = Dopri5()
   stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

   solution = diffeqsolve(
       terms, solver, t0, t1, dt0, y0,
       stepsize_controller=stepsize_controller
   )

The PID controller automatically reduces step size when :math:`f` increases rapidly.

JIT Compilation
---------------

The stress kernel is **JIT-compiled** for performance:

.. code-block:: python

   from jax import jit

   @jit
   def residuals_fene(params, x, y, test_mode):
       # ... compute predictions
       return y_pred - y

**Dispatch at Python level:** The stress type ("gaussian", "fene", "bell") is selected before JIT compilation, avoiding dynamic branches inside the JIT region.

Memory Considerations
---------------------

For time-series predictions (startup, LAOS), the full trajectory :math:`\mathbf{S}(t)` is stored:

- **Memory usage:** :math:`O(N_{\text{time}} \times 9)` (9 components of :math:`\mathbf{S}`)
- **Optimization:** Use ``jax.lax.scan`` instead of ``jax.lax.fori_loop`` to reduce memory

Example:

.. code-block:: python

   def step(S, t):
       S_new = integrate_one_step(S, t, dt)
       return S_new, S_new

   _, S_trajectory = jax.lax.scan(step, S_initial, t_array)

This stores only the final result in memory during compilation.

----

Fitting Guidance
================

Two-Stage Fitting Strategy
---------------------------

**Recommended workflow:**

1. **Stage 1: SAOS** → Fit :math:`G, \tau_b` using linearized model (fix :math:`L_{\text{max}} = \infty` or large value)
2. **Stage 2: Nonlinear** → Fit startup/LAOS data with :math:`G, \tau_b` fixed, optimize :math:`L_{\text{max}}`

**Rationale:** :math:`L_{\text{max}}` is weakly constrained by SAOS (only via :math:`f_{\text{eq}} \approx 1` correction), but strongly constrained by large-strain data.

.. code-block:: python

   # Stage 1: SAOS
   model_saos = TNTSingleMode(stress_type="gaussian")  # Gaussian approximation
   model_saos.fit(omega, G_star, test_mode='oscillation')
   G_fit, tau_b_fit = model_saos.params['G'].value, model_saos.params['tau_b'].value

   # Stage 2: Startup
   model_fene = TNTSingleMode(stress_type="fene")
   model_fene.params['G'].value = G_fit
   model_fene.params['tau_b'].value = tau_b_fit
   model_fene.params['G'].fixed = True
   model_fene.params['tau_b'].fixed = True
   model_fene.fit(t, sigma_startup, test_mode='startup', gamma_dot=10.0)
   L_max_fit = model_fene.params['L_max'].value

Bayesian Inference
------------------

**Recommended priors:**

.. code-block:: python

   import numpyro.distributions as dist

   priors = {
       'G': dist.LogNormal(jnp.log(1e3), 1.0),
       'tau_b': dist.LogNormal(jnp.log(1.0), 1.0),
       'L_max': dist.LogNormal(jnp.log(10.0), 0.5),  # Mode at 10
       'eta_s': dist.HalfNormal(1e2)
   }

**Why LogNormal for :math:`L_{\text{max}}`:**

- Ensures :math:`L_{\text{max}} > 0`
- Allows wide range (e.g., 5-20) while concentrating prior mass near typical values
- Reflects multiplicative uncertainty (factor of 2-3 uncertainty is common)

**Posterior checks:**

- Verify :math:`L_{\text{max}}` posterior is unimodal (multimodality indicates non-identifiability)
- Check :math:`\hat{R} < 1.01` for convergence
- Ensure ESS > 400 for reliable credible intervals

Identifiability of :math:`L_{\text{max}}`
------------------------------------------

:math:`L_{\text{max}}` is **only identifiable from nonlinear data** where :math:`\text{tr}(\mathbf{S})` becomes large.

**Symptoms of non-identifiability:**

- Flat posterior over wide range of :math:`L_{\text{max}}`
- High correlation between :math:`G` and :math:`L_{\text{max}}` (both scale stress)
- Poor convergence (low ESS, high :math:`\hat{R}`)

**Solution:** Include data at high :math:`\text{Wi}` or :math:`\gamma_0` where FENE effects are pronounced.

NLSQ Convergence Issues
------------------------

**Common failure modes:**

1. **Singularity divergence:** Optimizer drives :math:`\text{tr}(\mathbf{S}) \to L_{\text{max}}^2`, causing :math:`f \to \infty`
2. **Poor initialization:** Starting :math:`L_{\text{max}}` too small leads to early saturation

**Mitigations:**

.. code-block:: python

   # Initialize L_max larger than observed max trace
   tr_S_max = compute_max_trace_from_data(t, sigma)
   L_max_init = jnp.sqrt(tr_S_max) * 1.5  # 50% safety margin

   model.params['L_max'].value = L_max_init

   # Use bounded optimization
   from rheojax.utils.optimization import nlsq_curve_fit
   result = nlsq_curve_fit(
       model_fn, x, y, params,
       bounds={'L_max': (2.0, 100.0)}  # Hard bounds
   )

Multi-Start Optimization
-------------------------

For complex data (multiple overshoot peaks, noise), use **multi-start**:

.. code-block:: python

   from rheojax.utils.optimization import nlsq_optimize

   L_max_guesses = [3.0, 10.0, 30.0]
   results = []
   for L_max_init in L_max_guesses:
       model.params['L_max'].value = L_max_init
       result = nlsq_optimize(residuals, params)
       results.append(result)

   # Select best fit
   best = min(results, key=lambda r: r.cost)

----

Usage Examples
==============

Basic Fitting to Startup Data
------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   from rheojax.core import RheoData
   import jax.numpy as jnp

   # Generate synthetic startup data
   t = jnp.linspace(0, 10, 200)
   gamma_dot = 10.0

   # True parameters
   model_true = TNTSingleMode(stress_type="fene")
   model_true.params['G'].value = 1000.0
   model_true.params['tau_b'].value = 1.0
   model_true.params['L_max'].value = 10.0
   stress_true = model_true.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Add noise
   stress_noisy = stress_true + jax.random.normal(jax.random.PRNGKey(0), stress_true.shape) * 10

   # Fit model
   model = TNTSingleMode(stress_type="fene")
   rheo_data = RheoData(x=t, y=stress_noisy, test_mode='startup', gamma_dot=gamma_dot)
   model.fit(rheo_data)

   print(f"Fitted G: {model.params['G'].value:.1f} Pa")
   print(f"Fitted tau_b: {model.params['tau_b'].value:.3f} s")
   print(f"Fitted L_max: {model.params['L_max'].value:.2f}")

   # Predict and plot
   stress_pred = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)
   import matplotlib.pyplot as plt
   plt.plot(t, stress_noisy, 'o', label='Data')
   plt.plot(t, stress_pred, '-', label='FENE-P fit')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.show()

Bayesian Inference with Warm-Start
-----------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   import numpyro.distributions as dist

   # Step 1: NLSQ point estimate
   model = TNTSingleMode(stress_type="fene")
   model.fit(t, stress, test_mode='startup', gamma_dot=10.0)

   # Step 2: Bayesian with informative priors from NLSQ
   G_nlsq = model.params['G'].value
   tau_b_nlsq = model.params['tau_b'].value
   L_max_nlsq = model.params['L_max'].value

   priors = {
       'G': dist.LogNormal(jnp.log(G_nlsq), 0.2),
       'tau_b': dist.LogNormal(jnp.log(tau_b_nlsq), 0.2),
       'L_max': dist.LogNormal(jnp.log(L_max_nlsq), 0.3),
       'eta_s': dist.HalfNormal(10.0)
   }

   result = model.fit_bayesian(
       t, stress, test_mode='startup', gamma_dot=10.0,
       priors=priors,
       num_warmup=1000, num_samples=2000, num_chains=4
   )

   # Extract credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print("95% Credible Intervals:")
   for param, (lower, upper) in intervals.items():
       print(f"  {param}: [{lower:.3f}, {upper:.3f}]")

   # Diagnostic plots
   import arviz as az
   idata = az.from_dict(result.posterior_samples)
   az.plot_trace(idata)
   plt.tight_layout()
   plt.show()

Simulating Startup Response
----------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   import jax.numpy as jnp

   model = TNTSingleMode(stress_type="fene")
   model.params['G'].value = 1000.0
   model.params['tau_b'].value = 1.0
   model.params['L_max'].value = 10.0

   # Startup at multiple Wi
   gamma_dots = [0.1, 1.0, 10.0, 100.0]
   t = jnp.linspace(0, 10, 500)

   import matplotlib.pyplot as plt
   for gamma_dot in gamma_dots:
       stress = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)
       Wi = model.params['tau_b'].value * gamma_dot
       plt.plot(t, stress, label=f'Wi = {Wi:.1f}')

   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.title('FENE-P Startup: Strain Stiffening at High Wi')
   plt.show()

LAOS Simulation and Harmonic Analysis
--------------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   import jax.numpy as jnp
   import numpy as np
   from scipy.fft import fft

   model = TNTSingleMode(stress_type="fene")
   model.params['G'].value = 1000.0
   model.params['tau_b'].value = 1.0
   model.params['L_max'].value = 10.0

   # LAOS parameters
   gamma_0 = 2.0  # Large amplitude
   omega = 1.0    # De ~ 1
   n_cycles = 10
   t = jnp.linspace(0, 2 * jnp.pi * n_cycles / omega, 2000)

   # Simulate
   stress = model.predict(t, test_mode='laos', gamma_0=gamma_0, omega=omega)

   # Extract last cycle
   t_cycle = t[-200:]
   stress_cycle = stress[-200:]
   gamma_cycle = gamma_0 * jnp.sin(omega * t_cycle)

   # Fourier analysis
   stress_fft = fft(np.array(stress_cycle))
   n_harmonics = 5
   harmonics = np.abs(stress_fft[:n_harmonics])

   print("Harmonic intensities:")
   for n in range(1, n_harmonics):
       ratio = harmonics[n] / harmonics[1]
       print(f"  I_{n}/I_1 = {ratio:.4f}")

   # Lissajous plot
   plt.figure(figsize=(10, 4))
   plt.subplot(1, 2, 1)
   plt.plot(gamma_cycle, stress_cycle)
   plt.xlabel('Strain')
   plt.ylabel('Stress (Pa)')
   plt.title('Lissajous Curve (Stiffening)')

   plt.subplot(1, 2, 2)
   plt.stem(range(1, n_harmonics), harmonics[1:n_harmonics] / harmonics[1])
   plt.xlabel('Harmonic n')
   plt.ylabel('I_n / I_1')
   plt.title('Harmonic Spectrum')
   plt.tight_layout()
   plt.show()

Comparing Gaussian vs FENE-P
-----------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode
   import jax.numpy as jnp

   # Shared parameters
   G = 1000.0
   tau_b = 1.0
   gamma_dot = 10.0
   t = jnp.linspace(0, 5, 300)

   # Gaussian model
   model_gaussian = TNTSingleMode(stress_type="gaussian")
   model_gaussian.params['G'].value = G
   model_gaussian.params['tau_b'].value = tau_b
   stress_gaussian = model_gaussian.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # FENE-P model
   model_fene = TNTSingleMode(stress_type="fene")
   model_fene.params['G'].value = G
   model_fene.params['tau_b'].value = tau_b
   model_fene.params['L_max'].value = 10.0
   stress_fene = model_fene.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Plot comparison
   import matplotlib.pyplot as plt
   plt.plot(t, stress_gaussian, '--', label='Gaussian (base)')
   plt.plot(t, stress_fene, '-', label='FENE-P (L_max=10)')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.title('Strain Stiffening in FENE-P vs Gaussian')
   plt.show()

----

Composition with Other Variants
================================

FENE + Bell (Force-Dependent Breakage)
---------------------------------------

Combine finite extensibility with **force-dependent bond kinetics**:

.. code-block:: python

   from rheojax.models.tnt import TNTSingleMode

   model = TNTSingleMode(breakage="bell", stress_type="fene")
   # Parameters: G, tau_b0, L_max, F_b, eta_s

**Physical scenario:** Rubber networks where bonds break faster under high tension (e.g., physical crosslinks with thermal activation).

**Effect:** Stress saturation occurs at **lower** levels than pure FENE-P (bonds break before reaching maximum extension).

FENE + Non-Affine (Finite Extension with Slip)
-----------------------------------------------

.. code-block:: python

   model = TNTSingleMode(non_affine=True, stress_type="fene")
   # Parameters: G, tau_b, L_max, xi, eta_s

**Physical scenario:** Elastomers with imperfect crosslinks, allowing partial slip while still exhibiting strain stiffening.

**Effect:** Reduces stress overshoot magnitude (slip dissipates energy) but retains stiffening at high strain.

FENE + Stretch-Dependent Creation
----------------------------------

.. code-block:: python

   model = TNTSingleMode(creation="stretch", stress_type="fene")
   # Parameters: G, tau_b, L_max, tau_c, alpha_c, eta_s

**Physical scenario:** Self-healing networks where new bonds form more readily under extension.

**Effect:** Enhances stress hardening (reformation counteracts breakage).

Multi-Variant Composition
--------------------------

Combine all three variants:

.. code-block:: python

   model = TNTSingleMode(
       breakage="bell",
       stress_type="fene",
       non_affine=True,
       creation="stretch"
   )

**Use case:** Complex materials (e.g., dynamic covalent networks, vitrimers) requiring multiple physics.

----

Failure Mode: Chain Snap
=========================

The FENE-P stress function diverges as :math:`\text{tr}(\mathbf{S}) \to L_{\max}^2`:

.. math::

   f(\mathbf{S}) = \frac{L_{\max}^2}{L_{\max}^2 - \text{tr}(\mathbf{S})} \to \infty

This represents **catastrophic chain extension** near the maximum extensibility limit.
Physically, this corresponds to chains being pulled taut between crosslinks, with the
entropic spring force diverging as the end-to-end distance approaches the contour length.

**Practical implications:**

- Numerical integration requires clipping: :math:`\text{tr}(\mathbf{S}) < 0.99 L_{\max}^2`
- Very high :math:`Wi` or :math:`\gamma_0` near :math:`L_{\max}` causes ODE stiffness
- Physical chains would rupture or detach before reaching this limit (combine with Bell
  breakage for realistic behavior)

----

See Also
========

**TNT Shared Reference:**

- :doc:`tnt_protocols` — Full protocol equations, cohort formulation, and numerical methods
- :doc:`tnt_knowledge_extraction` — Model identification and fitting guidance

**TNT Base Model:**

- :ref:`model-tnt-tanaka-edwards` — Base model (constant breakage, Hookean chains)

**Complementary Extensions (combine with FENE):**

- :ref:`model-tnt-bell` — Add force-dependent breakage to finite extensibility (Bell+FENE)
- :ref:`model-tnt-stretch-creation` — Strain-enhanced creation for shear thickening with hardening

**Related TNT Variants:**

- :ref:`model-tnt-non-affine` — Alternative nonlinear effect (:math:`N_2 \neq 0`)
- :ref:`model-tnt-loop-bridge` — Two-species topology
- :ref:`model-tnt-cates` — Living polymers

----

API Reference
=============

.. autoclass:: rheojax.models.tnt.TNTSingleMode
   :members: fit, predict, simulate_startup, simulate_laos, fit_bayesian, get_credible_intervals
   :noindex:

**Key Methods:**

- ``fit(rheo_data)``: NLSQ optimization to data
- ``predict(x, test_mode, **kwargs)``: Forward prediction
- ``simulate_startup(gamma_dot, t_end, n_points)``: Startup flow simulation
- ``simulate_laos(gamma_0, omega, n_cycles, points_per_cycle)``: LAOS simulation
- ``fit_bayesian(rheo_data, priors, num_warmup, num_samples, num_chains)``: Bayesian inference

**Parameters:**

.. code-block:: python

   model.params['G']       # Modulus (Pa)
   model.params['tau_b']   # Bond lifetime (s)
   model.params['L_max']   # Extensibility (dimensionless)
   model.params['eta_s']   # Solvent viscosity (Pa·s)

----

References
==========

.. [Warner1972] Warner, H. R. (1972). "Kinetic Theory and Rheology of Dilute Suspensions of Finitely Extendible Dumbbells". *Industrial & Engineering Chemistry Fundamentals*, 11(3), 379-387. https://doi.org/10.1021/i160043a017

.. [Peterlin1966] Peterlin, A. (1966). "Hydrodynamics of Linear Macromolecules". *Pure and Applied Chemistry*, 12(1-4), 563-586. https://doi.org/10.1351/pac196612010563

.. [Bird1980] Bird, R. B., Dotson, P. J., & Johnson, N. L. (1980). "Polymer Solution Rheology Based on a Finitely Extensible Bead-Spring Chain Model". *Journal of Non-Newtonian Fluid Mechanics*, 7(2-3), 213-235. https://doi.org/10.1016/0377-0257(80)85007-5

.. [TanakaEdwards1992] Tanaka, F., & Edwards, S. F. (1992). "Viscoelastic Properties of Physically Crosslinked Networks". *Macromolecules*, 25(5), 1516-1523. https://doi.org/10.1021/ma00031a024

.. [Keunings1997] Keunings, R. (1997). "On the Peterlin Approximation for Finitely Extensible Dumbbells". *Journal of Non-Newtonian Fluid Mechanics*, 68(1), 85-100. https://doi.org/10.1016/S0377-0257(96)01497-8

.. [Wiest1989] Wiest, J. M. (1989). "A Differential Constitutive Equation for Polymer Melts". *Rheologica Acta*, 28(1), 4-12. https://doi.org/10.1007/BF01354763

.. [DoiEdwards1986] Doi, M., & Edwards, S. F. (1986). *The Theory of Polymer Dynamics*. Oxford University Press. ISBN: 978-0198519768

.. [Bird1987] Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric Liquids, Volume 1: Fluid Mechanics* (2nd ed.). Wiley. ISBN: 978-0471802457

**Recommended Reading Order:**

1. Warner (1972) — FENE spring foundation
2. Peterlin (1966) — Closure approximation
3. Bird et al. (1980) — FENE-P constitutive equation
4. Tanaka & Edwards (1992) — Transient network base theory
5. Keunings (1997) — FENE-P accuracy and limitations

----

.. note::

   **Practical Tips:**

   - Always fit SAOS data first to get :math:`G` and :math:`\tau_b`
   - :math:`L_{\text{max}}` requires nonlinear data (startup or LAOS)
   - Clamp :math:`\text{tr}(\mathbf{S}) < L_{\text{max}}^2 - \epsilon` to avoid singularity
   - Use multi-start optimization if fits fail to converge
   - Validate FENE-P assumptions with complementary scattering data

.. warning::

   **Numerical Stability:**

   - Near :math:`\text{tr}(\mathbf{S}) \to L_{\text{max}}^2`, ODEs become stiff
   - Use adaptive step size control (Diffrax PIDController)
   - Initialize :math:`L_{\text{max}} > \sqrt{\text{max observed } \text{tr}(\mathbf{S})}`
   - Check for optimizer divergence (cost → ∞) indicating singularity issues

.. seealso::

   **External Resources:**

   - PolymerPhysics.org FENE Tutorial (hypothetical link)
   - RheoHub FENE-P Calculator (hypothetical link)
   - NumPyro documentation for Bayesian inference: https://num.pyro.ai/

----

*Last updated: 2026-01-26*
