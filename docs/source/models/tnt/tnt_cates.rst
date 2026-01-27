.. _model-tnt-cates:

===========================================================
TNT Cates (Living Polymers / Wormlike Micelles) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
===============

**Use When:**

- Wormlike micelles (e.g., CTAB/NaSal, CPyCl/NaSal, SDS/LAPB)
- Living polymer systems with reversible scission
- Surfactant solutions showing single-mode Maxwell behavior
- Systems with perfect semicircular Cole-Cole plots
- Materials exhibiting shear banding in flow curves

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Symbol
     - Default
     - Units
     - Description
   * - :math:`G_0`
     - 100
     - Pa
     - Plateau modulus
   * - :math:`\tau_\text{rep}`
     - 10.0
     - s
     - Reptation time
   * - :math:`\tau_\text{break}`
     - 0.1
     - s
     - Mean breaking time
   * - :math:`\eta_s`
     - 0.0
     - Pa·s
     - Solvent viscosity

**Key Equations:**

Effective relaxation time (fast-breaking limit):

.. math::

   \tau_d = \sqrt{\tau_\text{rep} \cdot \tau_\text{break}}

Breaking parameter:

.. math::

   \zeta = \frac{\tau_\text{break}}{\tau_\text{rep}}

Zero-shear viscosity:

.. math::

   \eta_0 = G_0 \tau_d

**Test Modes:**

All six protocols supported:

- OSCILLATION (SAOS): :math:`G'(\omega)`, :math:`G''(\omega)`
- FLOW_CURVE: :math:`\sigma(\dot{\gamma})`, shear banding prediction
- STARTUP: Transient stress overshoot
- RELAXATION: Monoexponential stress decay
- CREEP: Single-mode compliance
- LAOS: Nonlinear oscillatory response

**Material Examples:**

- CTAB/NaSal wormlike micelles (cetyl trimethylammonium bromide / sodium salicylate)
- CPyCl/NaSal (cetyl pyridinium chloride / sodium salicylate)
- SDS/LAPB (sodium dodecyl sulfate / lauryl amido propyl betaine)
- Ionic surfactant solutions above critical micelle concentration
- Living polymer melts with reversible cross-linking
- Telechelic polymers with sticky ends

**Key Characteristics:**

- Single Maxwell-like relaxation in fast-breaking limit (:math:`\zeta \ll 1`)
- Perfect semicircular Cole-Cole plot (:math:`G''` vs :math:`G'`)
- Monoexponential stress relaxation
- Non-monotonic flow curve (constitutive instability)
- Shear banding for :math:`\text{Wi}_d > 1`
- Crossover frequency :math:`\omega_c = 1/\tau_d`

Notation Guide
==============

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Symbol
     - Units
     - Description
   * - :math:`G_0`
     - Pa
     - Plateau modulus (related to mesh size)
   * - :math:`\tau_\text{rep}`
     - s
     - Reptation time (curvilinear diffusion along tube)
   * - :math:`\tau_\text{break}`
     - s
     - Mean breaking time (Poisson scission)
   * - :math:`\tau_d`
     - s
     - Effective relaxation time = :math:`\sqrt{\tau_\text{rep} \tau_\text{break}}`
   * - :math:`\zeta`
     - --
     - Breaking parameter = :math:`\tau_\text{break}/\tau_\text{rep}`
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity
   * - :math:`\eta_0`
     - Pa·s
     - Zero-shear viscosity = :math:`G_0 \tau_d`
   * - :math:`S`
     - --
     - Conformation tensor (end-to-end vector average)
   * - :math:`\boldsymbol{\kappa}`
     - :math:`s^{-1}`
     - Velocity gradient tensor
   * - :math:`D`
     - :math:`s^{-1}`
     - Rate of deformation tensor = :math:`(\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2`
   * - :math:`\boldsymbol{\sigma}`
     - Pa
     - Stress tensor
   * - :math:`\text{Wi}_d`
     - --
     - Weissenberg number = :math:`\tau_d \dot{\gamma}`
   * - :math:`L`
     - nm
     - Mean micelle contour length
   * - :math:`\xi`
     - nm
     - Mesh size (entanglement length scale)
   * - :math:`\omega_c`
     - rad/s
     - Crossover frequency = :math:`1/\tau_d`
   * - :math:`k_B T`
     - J
     - Thermal energy
   * - :math:`E_\text{scission}`
     - J/mol
     - Activation energy for scission

Overview
========

Physical Background
-------------------

The TNT Cates model describes the rheology of **living polymers**, systems where polymeric chains can reversibly break and recombine on timescales comparable to their stress relaxation. The most prominent experimental realization is **wormlike micelles**: long, flexible, cylindrical surfactant aggregates that form in concentrated surfactant solutions.

Unlike conventional polymers with permanent covalent bonds, wormlike micelles continuously undergo:

1. **Scission**: Random breaking at any point along the contour
2. **Recombination**: End-to-end fusion when micelle tips meet
3. **Reversibility**: Breaking and recombination rates are balanced at equilibrium

The model was developed by M.E. Cates in 1987-1990 and represents one of the most successful theories in surfactant rheology.

Historical Development
----------------------

**1987 - Cates (Macromolecules):**

- Extended reptation theory to living polymers
- Showed that reversible scission fundamentally alters stress relaxation
- Predicted single-mode Maxwell behavior in fast-breaking limit

**1990 - Cates (J Phys Chem):**

- Nonlinear rheology and flow curve predictions
- Constitutive instability leading to shear banding
- Connection to experimental observations

**1990 - Cates and Candau (J Phys Condens Matter):**

- Comprehensive review of statics and dynamics
- Scaling laws for micelle length and relaxation times

**1991 - Turner and Cates (Langmuir):**

- Linear viscoelasticity in detail
- Cole-Cole plot predictions

**1991 - Rehage and Hoffmann (Mol Phys):**

- Experimental verification with CTAB/NaSal
- Perfect Maxwell behavior and shear banding

Why This Model Matters
-----------------------

1. **Explains Maxwell behavior in surfactants**: Conventional polymers show broad spectra (many modes); wormlike micelles show single-mode behavior
2. **Predictive power**: Quantitatively explains linear and nonlinear rheology with just 3 parameters
3. **Shear banding mechanism**: First model to predict flow curve instability from microscopic dynamics
4. **Industrial relevance**: Wormlike micelles are used in consumer products (shampoos, detergents), enhanced oil recovery, drag reduction
5. **Theoretical foundation**: Connects reptation theory to reversible kinetics

Physical Foundations
====================

Reptation Theory
----------------

**De Gennes (1971), Doi-Edwards (1978):**

Entangled polymers are confined to a "tube" formed by neighboring chains. Stress relaxation occurs via **curvilinear diffusion** (reptation) along the tube axis. The reptation time scales as:

.. math::

   \tau_\text{rep} \sim \frac{L^3}{\pi^2 D}

where :math:`L` is the contour length and :math:`D` is the curvilinear diffusion coefficient.

For permanent polymers, :math:`\tau_\text{rep}` is the dominant relaxation time. The stress relaxes via a spectrum of modes:

.. math::

   G(t) = G_0 \sum_{p \text{ odd}} \frac{8}{\pi^2 p^2} \exp\left(-\frac{p^2 t}{\tau_\text{rep}}\right)

Reversible Scission
-------------------

**Cates addition (1987):**

Wormlike micelles break at random positions with Poisson statistics. The mean scission time for a micelle of length :math:`L` is:

.. math::

   \tau_\text{break}(L) = \frac{\tau_\text{break}^0}{L/L_0}

where :math:`\tau_\text{break}^0` is the breaking time of a reference length :math:`L_0`.

**Key insight:** Breaking randomizes the tube position. If :math:`\tau_\text{break} \ll \tau_\text{rep}`, the micelle breaks many times before reptating out of its original tube. This **scrambles the memory** of the initial conformation.

Fast-Breaking Limit
-------------------

**Condition:**

.. math::

   \zeta = \frac{\tau_\text{break}}{\tau_\text{rep}} \ll 1

**Consequence:**

The effective stress relaxation becomes **single-mode** with a geometric mean relaxation time:

.. math::

   \tau_d = \sqrt{\tau_\text{rep} \cdot \tau_\text{break}}

**Physical picture:**

- Reptation requires diffusion over length :math:`L`
- Breaking cuts the micelle into pieces of size approximately :math:`L/2` every :math:`\tau_\text{break}`
- The micelle escapes its tube when the diffusion length :math:`\sqrt{D t}` equals the breaking length :math:`\sim \sqrt{D \tau_\text{break}}`
- Solving :math:`L \sim \sqrt{D \tau_\text{break}}` with :math:`\tau_\text{rep} \sim L^3/D` gives :math:`\tau_d \sim \sqrt{\tau_\text{rep} \tau_\text{break}}`

**Scaling:**

.. math::

   \tau_d \sim L \quad \text{(linear in length)}

compared to :math:`\tau_\text{rep} \sim L^3` for unbreakable chains.

Recombination and Equilibrium
------------------------------

At equilibrium, the scission rate equals the recombination rate:

.. math::

   k_\text{break} n_\text{micelles} = k_\text{recomb} n_\text{ends}^2

where:

- :math:`k_\text{break}` is the scission rate constant
- :math:`k_\text{recomb}` is the recombination rate constant
- :math:`n_\text{micelles}` is the number of micelles
- :math:`n_\text{ends}` is the number of free ends

This gives an equilibrium micelle length distribution. For simplicity, the TNT Cates model assumes a **mean-field** description with average properties.

Tube Model Mapping
------------------

The conformation tensor :math:`S` represents the average end-to-end vector orientation. In the tube model:

.. math::

   S = \langle \mathbf{u} \otimes \mathbf{u} \rangle

where :math:`\mathbf{u}` is the unit tangent vector along the tube.

The stress is:

.. math::

   \boldsymbol{\sigma} = G_0 (S - I) + 2 \eta_s D

where :math:`G_0 \sim k_B T / \xi^3` is the plateau modulus (:math:`\xi` is the mesh size).

Governing Equations
===================

Conformation Tensor Evolution
------------------------------

The fast-breaking Cates model reduces to a single-mode upper-convected Maxwell (UCM) constitutive equation with relaxation time :math:`\tau_d`:

.. math::

   \frac{DS}{Dt} - \boldsymbol{\kappa} \cdot S - S \cdot \boldsymbol{\kappa}^T = -\frac{1}{\tau_d}(S - I)

where:

- :math:`\frac{D}{Dt}` is the material derivative
- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` is the velocity gradient tensor
- :math:`I` is the identity tensor

**Expanded form:**

.. math::

   \frac{\partial S}{\partial t} + \mathbf{v} \cdot \nabla S - \boldsymbol{\kappa} \cdot S - S \cdot \boldsymbol{\kappa}^T = -\frac{1}{\tau_d}(S - I)

For homogeneous flows (:math:`\nabla S = 0`):

.. math::

   \frac{dS}{dt} = \boldsymbol{\kappa} \cdot S + S \cdot \boldsymbol{\kappa}^T - \frac{1}{\tau_d}(S - I)

Stress Tensor
-------------

.. math::

   \boldsymbol{\sigma} = G_0 (S - I) + 2 \eta_s D

where:

- :math:`G_0` is the plateau modulus
- :math:`\eta_s` is the solvent viscosity
- :math:`D = (\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2` is the rate of deformation tensor

**Total stress:**

.. math::

   \boldsymbol{\sigma}_\text{total} = -p I + \boldsymbol{\sigma}

where :math:`p` is the pressure (isotropic part).

Effective Relaxation Time
--------------------------

**Computed internally:**

.. math::

   \tau_d = \sqrt{\tau_\text{rep} \cdot \tau_\text{break}}

This is **not** a fitted parameter. The model fits :math:`\tau_\text{rep}` and :math:`\tau_\text{break}` separately, and :math:`\tau_d` is derived.

**Physical interpretation:**

- :math:`\tau_d` is the **observable** relaxation time in SAOS (crossover frequency :math:`\omega_c = 1/\tau_d`)
- :math:`\tau_\text{rep}` and :math:`\tau_\text{break}` are **microscopic** timescales
- Requires temperature-dependent or concentration-dependent data to separate :math:`\tau_\text{rep}` and :math:`\tau_\text{break}`

Steady Shear Flow
-----------------

**Velocity field:**

.. math::

   \mathbf{v} = (\dot{\gamma} y, 0, 0)

**Velocity gradient:**

.. math::

   \boldsymbol{\kappa} = \begin{pmatrix} 0 & \dot{\gamma} & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

**Steady-state solution (analytical):**

Weissenberg number:

.. math::

   \text{Wi}_d = \tau_d \dot{\gamma}

Shear stress:

.. math::

   \sigma_{xy} = \frac{G_0 \text{Wi}_d}{1 + \text{Wi}_d^2} + \eta_s \dot{\gamma}

Normal stress differences:

.. math::

   N_1 = \sigma_{xx} - \sigma_{yy} = \frac{2 G_0 \text{Wi}_d^2}{1 + \text{Wi}_d^2}

.. math::

   N_2 = 0 \quad \text{(UCM model)}

**Flow curve instability:**

The shear stress is **non-monotonic**: it increases for :math:`\text{Wi}_d < 1`, reaches a maximum at :math:`\text{Wi}_d = 1`, then decreases for :math:`\text{Wi}_d > 1`.

Maximum shear stress:

.. math::

   \sigma_{xy}^\text{max} = \frac{G_0}{2} + \eta_s \dot{\gamma}_\text{max}

where :math:`\dot{\gamma}_\text{max} = 1/\tau_d`.

**Constitutive instability:**

For :math:`\text{Wi}_d > 1`, the flow curve has **negative slope** :math:`d\sigma/d\dot{\gamma} < 0`. This is mechanically unstable and leads to **shear banding**: coexistence of high and low shear rate bands.

Small Amplitude Oscillatory Shear (SAOS)
-----------------------------------------

**Input:**

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t)

**Complex modulus:**

.. math::

   G^*(\omega) = G'(\omega) + i G''(\omega)

**Storage modulus:**

.. math::

   G'(\omega) = \frac{G_0 (\omega \tau_d)^2}{1 + (\omega \tau_d)^2}

**Loss modulus:**

.. math::

   G''(\omega) = \frac{G_0 (\omega \tau_d)}{1 + (\omega \tau_d)^2} + \omega \eta_s

**Limiting behavior:**

Low frequency (:math:`\omega \tau_d \ll 1`):

.. math::

   G' \sim G_0 \omega^2 \tau_d^2, \quad G'' \sim G_0 \omega \tau_d + \omega \eta_s

High frequency (:math:`\omega \tau_d \gg 1`):

.. math::

   G' \to G_0, \quad G'' \sim \frac{G_0}{\omega \tau_d}

**Crossover frequency:**

.. math::

   \omega_c = \frac{1}{\tau_d} \quad \text{where } G'(\omega_c) = G''(\omega_c) - \omega_c \eta_s

**Loss tangent:**

.. math::

   \tan \delta = \frac{G''}{G'} = \frac{1}{\omega \tau_d} + \frac{\eta_s}{G_0} \frac{1}{(\omega \tau_d)^2}

Cole-Cole Plot
--------------

**Signature of single-mode Maxwell:**

Plot :math:`G''` vs :math:`G'` (parametric in :math:`\omega`). For a single Maxwell mode with :math:`\eta_s = 0`:

.. math::

   \left(G' - \frac{G_0}{2}\right)^2 + (G'')^2 = \left(\frac{G_0}{2}\right)^2

This is a **perfect semicircle** with:

- Center at :math:`(G_0/2, 0)`
- Radius :math:`G_0/2`
- Passes through origin :math:`(0, 0)` at :math:`\omega \to 0`
- Passes through :math:`(G_0, 0)` at :math:`\omega \to \infty`

**Experimental test:**

If wormlike micelles truly follow the Cates model (fast-breaking limit), the Cole-Cole plot should be a perfect semicircle. Deviations indicate:

- :math:`\zeta` not small enough (intermediate breaking)
- Branching (Y-junctions)
- Polydispersity in micelle length
- Multiple relaxation modes

Startup Flow
------------

**Step shear rate:**

.. math::

   \dot{\gamma}(t) = \begin{cases} 0 & t < 0 \\ \dot{\gamma}_0 & t \geq 0 \end{cases}

**ODE system:**

.. math::

   \frac{dS_{xx}}{dt} = 2 \dot{\gamma}_0 S_{xy} - \frac{1}{\tau_d}(S_{xx} - 1)

.. math::

   \frac{dS_{xy}}{dt} = \dot{\gamma}_0 S_{yy} - \frac{1}{\tau_d} S_{xy}

.. math::

   \frac{dS_{yy}}{dt} = -\frac{1}{\tau_d}(S_{yy} - 1)

**Initial condition:**

.. math::

   S(0) = I \quad \text{(isotropic state)}

**Transient shear stress:**

.. math::

   \sigma_{xy}(t) = G_0 S_{xy}(t) + \eta_s \dot{\gamma}_0

**Analytical solution (for :math:`\eta_s = 0`):**

.. math::

   \sigma_{xy}(t) = \frac{G_0 \text{Wi}_d}{1 + \text{Wi}_d^2} \left[ 1 - e^{-t/\tau_d} (1 + \text{Wi}_d^2) + \text{Wi}_d^2 e^{-t/\tau_d} \cos\left(\frac{\text{Wi}_d t}{\tau_d}\right) \right]

For :math:`\text{Wi}_d > 1`, the stress exhibits **damped oscillations** before reaching steady state. There is **no stress overshoot** in the UCM model (unlike shear-thinning models).

Stress Relaxation
-----------------

**Protocol:**

1. Apply steady shear :math:`\dot{\gamma}_0` until steady state
2. At :math:`t = 0`, set :math:`\dot{\gamma} = 0` and monitor stress decay

**Relaxation:**

.. math::

   \sigma_{xy}(t) = \sigma_{xy}(0) e^{-t/\tau_d}

**Monoexponential decay** with time constant :math:`\tau_d`.

Creep
-----

**Protocol:**

Apply constant stress :math:`\sigma_0` at :math:`t = 0` and measure strain :math:`\gamma(t)`.

**Compliance:**

.. math::

   J(t) = \frac{\gamma(t)}{\sigma_0}

**Analytical solution:**

.. math::

   J(t) = \frac{1}{G_0} \left[ 1 - e^{-t/\tau_d} \right] + \frac{t}{\eta_0}

where :math:`\eta_0 = G_0 \tau_d`.

**Limits:**

- Short time: :math:`J(t) \sim t/(G_0 \tau_d) = t/\eta_0` (viscous flow)
- Long time: :math:`J(t) \to 1/G_0 + t/\eta_0` (steady-state flow)

Large Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

**Input:**

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t)

**Nonlinearity parameter:**

.. math::

   \text{Wi}_\text{LAOS} = \gamma_0 \omega \tau_d

**Fourier decomposition:**

For :math:`\gamma_0 \omega \tau_d > 1`, the stress waveform contains odd harmonics:

.. math::

   \sigma(t) = \sum_{n \text{ odd}} \sigma_n \sin(n \omega t + \delta_n)

**Lissajous curves:**

Stress vs strain and stress vs strain rate curves become ellipses distorted by nonlinearity.

Parameter Table
===============

.. list-table::
   :header-rows: 1
   :widths: 15 15 20 15 35

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Physical Meaning
   * - Plateau modulus
     - :math:`G_0`
     - 100 Pa
     - (1, 1e6) Pa
     - Elastic modulus at high frequency
   * - Reptation time
     - :math:`\tau_\text{rep}`
     - 10.0 s
     - (1e-3, 1e6) s
     - Relaxation time for unbreakable chain to reptate out of tube
   * - Breaking time
     - :math:`\tau_\text{break}`
     - 0.1 s
     - (1e-6, 1e4) s
     - Mean time between scission events for a micelle
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0 Pa·s
     - (0, 1e4) Pa·s
     - Viscosity of solvent (water, glycerol mixtures, etc.)

Derived Quantities
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Quantity
     - Formula
     - Meaning
   * - Effective relaxation time
     - :math:`\tau_d = \sqrt{\tau_\text{rep} \tau_\text{break}}`
     - Observable relaxation time in SAOS
   * - Breaking parameter
     - :math:`\zeta = \tau_\text{break}/\tau_\text{rep}`
     - Fast-breaking if :math:`\zeta \ll 1`
   * - Zero-shear viscosity
     - :math:`\eta_0 = G_0 \tau_d + \eta_s`
     - Viscosity at :math:`\dot{\gamma} \to 0`
   * - Crossover frequency
     - :math:`\omega_c = 1/\tau_d`
     - Frequency where :math:`G' = G''`
   * - Critical shear rate
     - :math:`\dot{\gamma}_c = 1/\tau_d`
     - Onset of shear thinning

Parameter Interpretation
========================

Plateau Modulus
---------------

Related to the mesh size :math:`\xi`:

.. math::

   G_0 \sim \frac{k_B T}{\xi^3}

Typical values:

- Dilute micelles: 1-10 Pa
- Semi-dilute: 10-100 Pa
- Concentrated: 100-1000 Pa

Concentration dependence:

.. math::

   G_0 \sim c^{2.3}

where :math:`c` is surfactant concentration.

Reptation Time
--------------

Time for a micelle to diffuse curvilinearly along its tube over a distance equal to its contour length :math:`L`.

Scaling:

.. math::

   \tau_\text{rep} \sim \frac{L^3}{D}

Length dependence:

.. math:`

   \tau_\text{rep} \sim L^3

Concentration dependence:

.. math::

   \tau_\text{rep} \sim c^{1.5}

Breaking Time
-------------

Mean time between scission events. Related to scission energy barrier:

.. math::

   \tau_\text{break} \sim \exp\left(\frac{E_\text{scission}}{k_B T}\right)

Temperature dependence (Arrhenius):

.. math::

   \tau_\text{break}(T) = \tau_\text{break}^0 \exp\left(\frac{E_\text{scission}}{k_B T}\right)

Length dependence:

.. math::

   \tau_\text{break} \sim \frac{1}{L}

Breaking Parameter
------------------

.. math::

   \zeta = \frac{\tau_\text{break}}{\tau_\text{rep}}

Regimes:

- Fast-breaking: :math:`\zeta \ll 1` (single-mode Maxwell)
- Intermediate: :math:`\zeta \sim 1` (multi-mode spectrum)
- Unbreakable: :math:`\zeta \gg 1` (pure reptation)

Critical value: :math:`\zeta \lesssim 0.1` for single-mode approximation.

Effective Relaxation Time
--------------------------

.. math::

   \tau_d = \sqrt{\tau_\text{rep} \cdot \tau_\text{break}}

Scaling with length:

.. math::

   \tau_d \sim L

Observable in SAOS crossover frequency: :math:`\omega_c = 1/\tau_d`.

Decomposition challenge: Measuring :math:`\tau_d` alone is not enough. Need additional information from temperature series, concentration series, or scattering.

Validity and Assumptions
========================

Core Assumptions
----------------

**1. Fast-breaking limit:**

.. math::

   \zeta = \frac{\tau_\text{break}}{\tau_\text{rep}} \ll 1

**2. Mean-field:** Ignores spatial heterogeneity.

**3. Linear chains:** No branching (Y-junctions), no ring closure.

**4. Reversible scission:** Breaking and recombination are reversible.

**5. Equilibrium structure:** Micelle length distribution at thermodynamic equilibrium.

When the Model Applies
-----------------------

Ideal systems:

- CTAB/NaSal
- CPyCl/NaSal
- Solutions with :math:`\zeta < 0.1`

Indicators of validity:

- Perfect semicircular Cole-Cole plot
- Monoexponential stress relaxation
- Single crossover in :math:`G'`, :math:`G''`

When the Model Breaks Down
---------------------------

**1. Slow breaking (:math:`\zeta \gtrsim 1`):** Multi-mode spectrum.

**2. Branching:** Y-junctions change topology.

**3. Very concentrated solutions:** Gel-like structures.

**4. Non-equilibrium:** Transient networks.

Regimes and Behavior
====================

Linear Viscoelastic Regime
---------------------------

Condition: :math:`\gamma_0 \ll 1` or :math:`\text{Wi}_d \ll 1`

Behavior:

- Single Maxwell mode with :math:`\tau_d`
- Perfect semicircular Cole-Cole plot
- :math:`G'(\omega) \sim \omega^2` at low :math:`\omega`

Nonlinear Regime
----------------

Condition: :math:`\text{Wi}_d = \tau_d \dot{\gamma} \sim 1`

Shear thinning:

.. math::

   \eta(\dot{\gamma}) = \frac{G_0 \tau_d}{1 + (\tau_d \dot{\gamma})^2} + \eta_s

Flow curve maximum at:

.. math::

   \dot{\gamma}_\text{max} = \frac{1}{\tau_d}, \quad \sigma_\text{max} = \frac{G_0}{2}

Shear Banding Regime
--------------------

For :math:`\text{Wi}_d > 1`, negative slope in flow curve leads to shear banding.

What You Can Learn
==================

From SAOS Data
--------------

1. Effective relaxation time: :math:`\tau_d = 1/\omega_c`
2. Plateau modulus: :math:`G_0 = \lim_{\omega \to \infty} G'(\omega)`
3. Zero-shear viscosity: :math:`\eta_0 = \lim_{\omega \to 0} G''(\omega)/\omega`
4. Breaking parameter estimate from Cole-Cole plot

From Temperature Series
-----------------------

Arrhenius plot of :math:`\ln \tau_d` vs :math:`1/T` yields scission energy.

From Concentration Series
--------------------------

Scaling:

.. math::

   G_0 \sim c^{2.3}, \quad \tau_d \sim c^{0.5}

Experimental Design
===================

Primary Technique: SAOS
------------------------

**Why start here:** Non-destructive, reveals full linear spectrum.

**Protocol:**

1. Frequency sweep: 0.01 to 100 rad/s
2. Strain amplitude: 0.01 to 0.1 (linear regime)
3. Cole-Cole plot validation

Secondary: Flow Curves
----------------------

**Why:** Test nonlinear predictions, identify shear banding.

**Protocol:**

1. Steady shear sweep: 0.001 to 1000 1/s
2. Wait > 10 tau_d for equilibration
3. Look for stress plateau

Computational Implementation
============================

Numerical Integration
---------------------

ODE solver for conformation tensor evolution using adaptive Runge-Kutta.

Effective Relaxation Time
--------------------------

Computed internally:

.. code-block:: python

   tau_d = jnp.sqrt(tau_rep * tau_break)

Analytical Solutions
--------------------

**Steady shear:**

.. code-block:: python

   Wi_d = tau_d * gamma_dot
   sigma_xy = G_0 * Wi_d / (1 + Wi_d**2) + eta_s * gamma_dot

**SAOS:**

.. code-block:: python

   G_prime = G_0 * (omega * tau_d)**2 / (1 + (omega * tau_d)**2)
   G_double_prime = G_0 * (omega * tau_d) / (1 + (omega * tau_d)**2)

Fitting Guidance
================

Step-by-Step Protocol
---------------------

**Step 1:** Fit SAOS to single Maxwell (G_0, tau_d, eta_s).

**Step 2:** Validate with Cole-Cole plot.

**Step 3:** Decompose tau_d using temperature or concentration series.

**Step 4:** Fit flow curve (optional validation).

Usage Examples
==============

Basic Fitting
-------------

.. code-block:: python

   from rheojax.models.tnt import TNTCates
   import jax.numpy as jnp

   model = TNTCates()
   omega = jnp.logspace(-2, 2, 50)
   result = model.fit(omega, G_star, test_mode='oscillation')

See Also
========

Related Models
--------------

- :ref:`model-tnt-tanaka-edwards`
- :ref:`model-tnt-multi-species`
- :ref:`model-giesekus`

API Reference
=============

.. autoclass:: rheojax.models.tnt.TNTCates
   :members:
   :undoc-members:
   :show-inheritance:

References
==========

1. Cates (1987) Macromolecules 20:2289-2296
2. Cates (1990) J Phys Chem 94:371-375
3. Cates and Candau (1990) J Phys Condens Matter 2:6869-6892
4. Turner and Cates (1991) Langmuir 7:1590-1594
5. Rehage and Hoffmann (1991) Mol Phys 74:933-973
6. Berret (2006) Molecular Gels, Springer
7. Fielding (2007) Soft Matter 3:1262-1279
8. Doi and Edwards (1986) Theory of Polymer Dynamics, Oxford
