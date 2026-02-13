.. _model-tnt-loop-bridge:

===========================================================
TNT Loop-Bridge (Two-Species Kinetics) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

----

1. Quick Reference
===========================================================

**Use When:**

Telechelic polymers with two distinct chain populations:

- **Bridges**: Load-bearing chains connecting two different junctions
- **Loops**: Non-load-bearing chains with both ends on the same junction

**Material Examples:**

- HEUR (Hydrophobically-modified Ethoxylated URethane) thickeners
- PEG-hydrophobe associating polymers
- Telechelic ionomers (e.g., Surlyn)
- Flower micelle networks
- Triblock copolymer solutions (ABA with associating end-blocks)

**Key Characteristics:**

- Non-monotonic viscosity with shear rate
- Shear thickening at moderate rates (loop-to-bridge conversion)
- Shear thinning at high rates (bridge breakage dominates)
- Stress overshoot in startup flows with possible initial thickening

**Parameters (6):**

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 55

   * - Parameter
     - Default
     - Units
     - Description
   * - :math:`G`
     - 1000
     - Pa
     - Network modulus (all bridges active)
   * - :math:`\tau_b`
     - 1.0
     - s
     - Bridge lifetime
   * - :math:`\tau_a`
     - 0.1
     - s
     - Loop-to-bridge association time
   * - :math:`\nu`
     - 1.0
     - dimensionless
     - Force sensitivity (Bell exponent)
   * - :math:`f_{B,eq}`
     - 0.5
     - dimensionless
     - Equilibrium bridge fraction
   * - :math:`\eta_s`
     - 0.0
     - Pa·s
     - Solvent viscosity

**Test Modes:**

All six protocols supported:

- ``FLOW_CURVE``: Non-monotonic viscosity with thickening then thinning
- ``STARTUP``: Stress overshoot with transient thickening
- ``CREEP``: Population evolution under constant stress
- ``RELAXATION``: Bridge-to-loop conversion during stress decay
- ``OSCILLATION``: Linear viscoelasticity (SAOS)
- ``LAOS``: Nonlinear oscillatory response (population cycling)

**Key Equation:**

.. math::

   \frac{df_B}{dt} = \frac{1 - f_B}{\tau_a} - f_B \cdot k_{off}(\mathbf{S})

where :math:`k_{off}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]`.

Only bridges contribute to stress:

.. math::

   \boldsymbol{\sigma} = G \cdot f_B \cdot (\mathbf{S} - \mathbf{I}) + 2\eta_s \mathbf{D}

----

2. Notation Guide
===========================================================

**State Variables:**

.. list-table::
   :widths: 20 80

   * - :math:`f_B`
     - Bridge fraction (dimensionless, 0 to 1)
   * - :math:`f_L`
     - Loop fraction, :math:`f_L = 1 - f_B`
   * - :math:`\mathbf{S}`
     - Bridge conformation tensor (dimensionless, 3x3 symmetric)
   * - :math:`\text{tr}(\mathbf{S})`
     - Trace of conformation tensor, :math:`S_{xx} + S_{yy} + S_{zz}`

**Material Parameters:**

.. list-table::
   :widths: 20 80

   * - :math:`G`
     - Network modulus (Pa), sets stress magnitude
   * - :math:`\tau_b`
     - Bridge lifetime (s), reciprocal detachment rate
   * - :math:`\tau_a`
     - Loop-to-bridge association time (s)
   * - :math:`\nu`
     - Force sensitivity (dimensionless), Bell activation exponent
   * - :math:`f_{B,eq}`
     - Equilibrium bridge fraction, :math:`\tau_b/(\tau_a + \tau_b)`
   * - :math:`\eta_s`
     - Solvent viscosity (Pa·s)

**Kinetic Rates:**

.. list-table::
   :widths: 20 80

   * - :math:`k_{on}`
     - Loop-to-bridge association rate, :math:`1/\tau_a`
   * - :math:`k_{off}(\mathbf{S})`
     - Force-dependent bridge dissociation rate
   * - :math:`k_{off,0}`
     - Equilibrium detachment rate, :math:`1/\tau_b`

**Tensors:**

.. list-table::
   :widths: 20 80

   * - :math:`\boldsymbol{\kappa}`
     - Velocity gradient tensor, :math:`\nabla \mathbf{v}`
   * - :math:`\mathbf{D}`
     - Rate-of-deformation tensor, symmetric part of :math:`\boldsymbol{\kappa}`
   * - :math:`\mathbf{I}`
     - Identity tensor

**Dimensionless Numbers:**

.. list-table::
   :widths: 20 80

   * - :math:`\text{Wi}_b`
     - Weissenberg number for bridges, :math:`\dot{\gamma} \tau_b`
   * - :math:`\text{Wi}_a`
     - Weissenberg number for association, :math:`\dot{\gamma} \tau_a`

----

3. Overview
===========================================================

Physical Picture
-----------------------------------------------------------

The TNT Loop-Bridge model describes telechelic polymers — linear chains with associating groups (stickers) at both ends. In solution, these stickers aggregate into micelles or junction zones. Each chain can exist in one of two states:

1. **Bridge**: Both ends attached to different junctions, chain is stretched and carries network stress
2. **Loop**: Both ends attached to the same junction, chain is relaxed and contributes no stress

At equilibrium, the bridge fraction :math:`f_{B,eq}` is determined by the balance between:

- Loop-to-bridge conversion (rate :math:`k_{on} = 1/\tau_a`)
- Bridge-to-loop relaxation (rate :math:`k_{off,0} = 1/\tau_b`)

Under flow, this balance is perturbed:

- **Flow convection** stretches bridges, increasing :math:`k_{off}` (force-dependent detachment)
- **Strain** can promote loop-to-bridge conversion (geometric effect)
- **Non-monotonic viscosity** results: thickening at moderate rates (more bridges), thinning at high rates (bridge breakage dominates)

Historical Context
-----------------------------------------------------------

The model was introduced by:

- **Annable, Buscall, Ettelaie, and Whittlestone (1993)** in *Journal of Rheology* as a two-species extension of the Tanaka-Edwards transient network framework
- Built on earlier work by **Tanaka & Edwards (1992)** on transient networks
- Inspired by **Marrucci, Bhatt, and Ball (1993)** on telechelic dynamics
- Extended by **Vaccaro & Marrucci (2000)** with detailed stress-population coupling
- Experimentally validated by **Tripathi, Tam, and McKinley (2006)** for HEUR solutions

The model is particularly important for understanding:

- Shear thickening in associating polymers (e.g., paint thickeners)
- Flow-induced structuring in telechelic networks
- Stress overshoot and transient thickening
- Nonlinear rheology of flower micelle networks

Key Features
-----------------------------------------------------------

**Two-Species Kinetics:**

Unlike single-species transient networks (e.g., basic TNT), this model tracks the bridge fraction :math:`f_B(t)` as a dynamic variable. Only bridges contribute to stress, so the effective modulus is :math:`G_{\text{eff}} = G \cdot f_B`.

**Force-Dependent Dissociation:**

Bridge detachment follows Bell's theory:

.. math::

   k_{off}(\mathbf{S}) = k_{off,0} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]

where :math:`\text{tr}(\mathbf{S}) - 3` measures chain extension (since :math:`\text{tr}(\mathbf{I}) = 3`). High extension accelerates breakage.

**Stress-Population Coupling:**

Population kinetics and conformation evolution are fully coupled:

- :math:`f_B` controls the effective modulus
- :math:`\mathbf{S}` controls :math:`k_{off}` (through force dependence)
- Both evolve simultaneously under flow

**Non-Monotonic Viscosity:**

At low rates: :math:`\dot{\gamma} \tau_a \sim 1` promotes loop-to-bridge conversion → viscosity increases.

At high rates: :math:`\dot{\gamma} \tau_b \sim 1` and :math:`\text{tr}(\mathbf{S}) \gg 3` → bridge breakage dominates → viscosity decreases.

The viscosity maximum occurs at :math:`\dot{\gamma}^* \sim \frac{1}{\nu \tau_b}`.

----

4. Physical Foundations
===========================================================

Telechelic Architecture
-----------------------------------------------------------

Telechelic polymers are linear chains with two functional end-groups (telechelics = "distant attractors" in Greek). Common examples:

- **HEUR**: PEG backbone with hydrophobic end-caps (C16-C18 alkanes)
- **Ionomers**: Polyolefin chains with ionic end-groups (e.g., Surlyn)
- **ABA triblocks**: Polystyrene-PEO-Polystyrene in selective solvent

In aqueous solution, hydrophobic end-groups aggregate into micelles (typically 20-100 stickers per micelle). Each chain can bridge two micelles or form a loop within one micelle.

Bridge-Loop Equilibrium
-----------------------------------------------------------

At rest, the bridge fraction is determined by entropic and enthalpic balance:

- **Loop formation**: High entropy (chain is relaxed), but only one micelle occupied
- **Bridge formation**: Low entropy (chain is stretched), but connects network

The equilibrium condition :math:`\mu_B = \mu_L` (equal chemical potentials) gives:

.. math::

   f_{B,eq} = \frac{\tau_b}{\tau_a + \tau_b} = \frac{k_{off,0}}{k_{on} + k_{off,0}}

This is independent of :math:`G` (which only sets stress magnitude) and :math:`\nu` (which requires extension for activation).

Typical values:

- :math:`\tau_a < \tau_b` → :math:`f_{B,eq} > 0.5` (most chains are bridges)
- :math:`\tau_a > \tau_b` → :math:`f_{B,eq} < 0.5` (most chains are loops)

Three-State Chain Population
-----------------------------------------------------------

The full loop-bridge framework distinguishes three chain states:

- **Bridges**: Both ends attached to *different* junction points — these are the only
  chains that carry stress (elastically active)
- **Loops**: Both ends attached to the *same* junction — carry no stress (act as dangling
  ends contributing viscous drag)
- **Danglers**: One end free (detached) — carry no stress

The conservation constraint is:

.. math::

   f_B + f_L + f_D = 1

where :math:`f_B`, :math:`f_L`, :math:`f_D` are the fractions of bridges, loops, and
danglers respectively. In the simplified two-state model implemented in RheoJAX,
danglers are absorbed into the loop population: :math:`f_B + f_L = 1`.

Stress-Bearing Mechanism
-----------------------------------------------------------

**Only bridges contribute to stress.** This is the key distinction from single-species networks.

Loops are relaxed: they exert no force on junctions. Bridges are stretched: they exert force :math:`\sim k_B T \cdot (\mathbf{R} - \mathbf{R}_0)` on their end junctions.

In mean-field approximation, all bridges have the same average conformation :math:`\mathbf{S}`, giving stress:

.. math::

   \boldsymbol{\sigma}_{\text{bridge}} = G \cdot f_B \cdot (\mathbf{S} - \mathbf{I})

where:

- :math:`G = n_{chain} k_B T` (total chain density × thermal energy)
- :math:`f_B` is the fraction carrying stress
- :math:`(\mathbf{S} - \mathbf{I})` is the extension beyond equilibrium

Solvent contributes Newtonian viscosity :math:`2\eta_s \mathbf{D}`.

Shear-Induced Association
-----------------------------------------------------------

Flow stretches bridges, which has two effects:

1. **Force-dependent detachment**: High :math:`\text{tr}(\mathbf{S})` increases :math:`k_{off}` exponentially (Bell's theory)
2. **Geometric promotion of bridging**: Shear separates micelles, increasing the probability that a detached chain reattaches to a different micelle (not yet captured in mean-field model)

At moderate Weissenberg numbers (:math:`\dot{\gamma} \tau_a \sim 1`), the second effect dominates → :math:`f_B` increases above :math:`f_{B,eq}` → shear thickening.

At high Weissenberg numbers (:math:`\dot{\gamma} \tau_b \sim 1` and :math:`\text{tr}(\mathbf{S}) \gg 3`), force-dependent detachment dominates → :math:`f_B` decreases → shear thinning.

Force-Dependent Dissociation (Bell Theory)
-----------------------------------------------------------

Bond breakage under force follows Kramers theory for thermally activated escape over a barrier:

.. math::

   k_{off}(F) = k_{off,0} \exp\left[\frac{F \cdot \delta}{k_B T}\right]

where :math:`\delta` is the activation distance. For Gaussian chains, force is proportional to extension:

.. math::

   F \sim k_B T \cdot \left(\sqrt{\text{tr}(\mathbf{S})} - \sqrt{3}\right)

giving:

.. math::

   k_{off}(\mathbf{S}) = k_{off,0} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]

where :math:`\nu \sim \delta / a` (activation distance over monomer size).

Typical values: :math:`\nu \sim 0.5` to :math:`5` (weak to strong force sensitivity).

Mean-Field Approximation
-----------------------------------------------------------

The model assumes all bridges have the same conformation :math:`\mathbf{S}`. This neglects:

- Distribution of chain extensions (polydispersity effects)
- Spatial heterogeneity (micelle clustering)
- Correlations between bridge orientation and stress

These approximations are reasonable for:

- Moderate extensions (:math:`\text{tr}(\mathbf{S}) < 10`)
- Homogeneous flows (no spatial gradients)
- Fast equilibration within each population

For highly stretched states or shear banding, more detailed models are needed.

----

5. Governing Equations
===========================================================

Population Kinetics
-----------------------------------------------------------

The bridge fraction evolves according to:

.. math::

   \frac{df_B}{dt} = k_{on} (1 - f_B) - k_{off}(\mathbf{S}) f_B

where:

- :math:`k_{on} = 1/\tau_a`: Loop-to-bridge association rate (constant)
- :math:`k_{off}(\mathbf{S}) = \frac{1}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}) - 3\right)\right]`: Force-dependent bridge dissociation

At equilibrium (:math:`df_B/dt = 0`, :math:`\mathbf{S} = \mathbf{I}`):

.. math::

   f_{B,eq} = \frac{k_{on}}{k_{on} + k_{off,0}} = \frac{\tau_b}{\tau_a + \tau_b}

This provides a consistency check: :math:`f_{B,eq}` should match the ratio of timescales.

Analytical Bridge Fraction Solution
-----------------------------------------------------------

For constant shear rate :math:`\dot{\gamma}_0`, the bridge fraction evolves as:

.. math::

   f_B(t) = f_B^{\text{eq}}(\dot{\gamma}_0) + \left[f_B(0) - f_B^{\text{eq}}(\dot{\gamma}_0)\right]
   \exp\!\left[-(k_{LB} + k_{BL})\,t\right]

where :math:`f_B^{\text{eq}}(\dot{\gamma}_0) = k_{LB} / (k_{LB} + k_{BL})` is the
equilibrium bridge fraction at shear rate :math:`\dot{\gamma}_0`. The approach to
equilibrium occurs on the **kinetic timescale** :math:`\tau_{\text{kin}} = 1/(k_{LB} + k_{BL})`,
which is independent of the chain relaxation time :math:`\tau_b`.

Two Characteristic Timescales
-----------------------------------------------------------

The loop-bridge model has two distinct timescales:

1. **Chain relaxation** :math:`\tau_b`: Time for individual chain conformation to relax
   (sets the viscoelastic response of bridges)
2. **Bridge kinetics** :math:`\tau_{\text{kin}} = 1/(k_{LB} + k_{BL})`: Time for the
   bridge fraction to equilibrate (sets the population dynamics)

When :math:`\tau_b \ll \tau_{\text{kin}}`, chains relax fast but the bridge fraction
changes slowly — producing a **two-step relaxation** visible in G(t). When
:math:`\tau_b \gg \tau_{\text{kin}}`, population dynamics are fast but chain relaxation
is slow — the effective modulus adjusts quickly to :math:`G_{\text{eff}} = G \cdot f_B^{\text{eq}}`.

Bridge Conformation Evolution
-----------------------------------------------------------

The bridge conformation tensor evolves via upper-convected derivative with dissociation:

.. math::

   \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   - k_{off}(\mathbf{S}) (\mathbf{S} - \mathbf{I})

**Physical interpretation:**

- :math:`\boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T`: Convective stretching (upper-convected)
- :math:`-k_{off}(\mathbf{S}) (\mathbf{S} - \mathbf{I})`: Dissociation returns :math:`\mathbf{S} \to \mathbf{I}` (newly attached bridges are unstretched)

**Key assumption:** When a bridge detaches and reattaches, it reforms with equilibrium conformation :math:`\mathbf{I}` (instantaneous equilibration of loops).

Stress Tensor
-----------------------------------------------------------

Total stress is the sum of bridge elastic stress and solvent viscous stress:

.. math::

   \boldsymbol{\sigma} = G f_B (\mathbf{S} - \mathbf{I}) + 2\eta_s \mathbf{D}

**Critical point:** Only the bridge fraction contributes to elastic stress. The effective modulus is :math:`G_{\text{eff}}(t) = G \cdot f_B(t)`, which varies with flow history.

For simple shear (:math:`\kappa_{xy} = \dot{\gamma}`, :math:`D_{xy} = \dot{\gamma}/2`):

.. math::

   \sigma_{xy} = G f_B S_{xy} + \eta_s \dot{\gamma}

Five-State ODE System (Rate-Controlled)
-----------------------------------------------------------

For steady or transient shear at imposed :math:`\dot{\gamma}`, the system is:

.. math::

   \frac{df_B}{dt} &= \frac{1 - f_B}{\tau_a} - \frac{f_B}{\tau_b} \exp\left[\nu \left(S_{xx} + S_{yy} + S_{zz} - 3\right)\right] \\
   \frac{dS_{xx}}{dt} &= 2\dot{\gamma} S_{xy} - k_{off} (S_{xx} - 1) \\
   \frac{dS_{yy}}{dt} &= -k_{off} (S_{yy} - 1) \\
   \frac{dS_{zz}}{dt} &= -k_{off} (S_{zz} - 1) \\
   \frac{dS_{xy}}{dt} &= \dot{\gamma} S_{yy} - k_{off} S_{xy}

where :math:`k_{off} = \frac{1}{\tau_b} \exp\left[\nu \left(S_{xx} + S_{yy} + S_{zz} - 3\right)\right]`.

**Initial conditions:** :math:`f_B(0) = f_{B,eq}`, :math:`\mathbf{S}(0) = \mathbf{I}`.

**Steady state:** Solve :math:`df_B/dt = 0` and :math:`d\mathbf{S}/dt = \mathbf{0}` simultaneously (typically requires root-finding).

Six-State ODE System (Creep)
-----------------------------------------------------------

For imposed stress :math:`\sigma_0`, add strain as a state variable:

.. math::

   \frac{d\gamma}{dt} &= \frac{\sigma_0 - G f_B S_{xy}}{\eta_s} \\
   \dot{\gamma} &= \frac{d\gamma}{dt}

with the same five equations above, now coupled through :math:`\dot{\gamma}(t)`.

This requires implicit solution at each timestep (creep compliance).

Linearized Equations (SAOS)
-----------------------------------------------------------

For small-amplitude oscillatory shear :math:`\gamma(t) = \gamma_0 e^{i\omega t}`, linearize around :math:`f_B = f_{B,eq}`, :math:`\mathbf{S} = \mathbf{I}`:

.. math::

   f_B(t) &= f_{B,eq} + \delta f_B e^{i\omega t} \\
   \mathbf{S}(t) &= \mathbf{I} + \delta \mathbf{S} e^{i\omega t}

Substituting and keeping first-order terms:

.. math::

   i\omega \delta f_B &= -\frac{\delta f_B}{\tau_a} - \frac{\delta f_B}{\tau_b}
   - \frac{\nu f_{B,eq}}{\tau_b} \text{tr}(\delta \mathbf{S}) \\
   i\omega \delta S_{xy} &= \dot{\gamma}_0 - \frac{\delta S_{xy}}{\tau_b}

Solving gives complex modulus:

.. math::

   G^*(\omega) = G f_{B,eq} \frac{i\omega \tau_b}{1 + i\omega \tau_b}
   \left(1 + \frac{\nu f_{B,eq}}{1 + i\omega \tau_{\text{pop}}}\right)

where :math:`\tau_{\text{pop}} = \frac{\tau_a \tau_b}{\tau_a + \tau_b}` is the population relaxation time.

**Prediction:** Two relaxation processes:

1. Bridge conformation relaxation (time :math:`\tau_b`)
2. Population redistribution (time :math:`\tau_{\text{pop}}`)

At high frequencies (:math:`\omega \tau_b \gg 1`), :math:`G' \to G f_{B,eq}` (plateau modulus).

Flow Curve (Steady Shear)
-----------------------------------------------------------

Steady-state bridge fraction and conformation satisfy:

.. math::

   0 &= \frac{1 - f_B^{ss}}{\tau_a} - \frac{f_B^{ss}}{\tau_b} \exp\left[\nu \left(\text{tr}(\mathbf{S}^{ss}) - 3\right)\right] \\
   0 &= \boldsymbol{\kappa} \cdot \mathbf{S}^{ss} + \mathbf{S}^{ss} \cdot \boldsymbol{\kappa}^T
   - k_{off}^{ss} (\mathbf{S}^{ss} - \mathbf{I})

For simple shear, these are five coupled nonlinear equations in :math:`(f_B^{ss}, S_{xx}^{ss}, S_{yy}^{ss}, S_{zz}^{ss}, S_{xy}^{ss})`.

Viscosity is:

.. math::

   \eta(\dot{\gamma}) = \frac{\sigma_{xy}^{ss}}{\dot{\gamma}} = \frac{G f_B^{ss} S_{xy}^{ss}}{\dot{\gamma}} + \eta_s

**Non-monotonic behavior:**

- Low :math:`\dot{\gamma}`: :math:`f_B^{ss} > f_{B,eq}` (shear thickening)
- High :math:`\dot{\gamma}`: :math:`f_B^{ss} < f_{B,eq}` (shear thinning from force-dependent detachment)
- Maximum at :math:`\dot{\gamma}^* \sim \frac{1}{\nu \tau_b}`

Startup Transient
-----------------------------------------------------------

For step strain rate :math:`\dot{\gamma}` applied at :math:`t = 0`:

.. math::

   \sigma_{xy}(t) = G f_B(t) S_{xy}(t) + \eta_s \dot{\gamma}

**Stress overshoot:** Occurs when :math:`d\sigma_{xy}/dt = 0` before steady state.

**Transient thickening:** If :math:`f_B(t)` increases faster than :math:`S_{xy}(t)` initially, stress can exceed the final steady-state value before the overshoot.

This is a signature of loop-to-bridge conversion under flow.

Bridge Recovery During Relaxation
-----------------------------------------------------------

After cessation of flow, the bridge fraction :math:`f_B` recovers toward its quiescent
equilibrium value. If flow had depleted bridges (i.e., :math:`f_B < f_B^{\text{eq,0}}`),
the effective modulus may **increase** during relaxation as bridges reform — a
counter-intuitive feature where the material appears to stiffen while stress is relaxing.

This produces a characteristic **non-monotonic** stress relaxation: initial rapid decay
(chain relaxation) followed by partial stress recovery (bridge reformation), then final
decay to zero.

----

6. Parameter Table
===========================================================

.. list-table::
   :header-rows: 1
   :widths: 15 12 12 15 15 31

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Units
     - Physical Meaning
   * - Network modulus
     - :math:`G`
     - 1000
     - (1, 1e8)
     - Pa
     - Elastic stress per unit bridge fraction
   * - Bridge lifetime
     - :math:`\tau_b`
     - 1.0
     - (1e-6, 1e4)
     - s
     - Inverse detachment rate at equilibrium
   * - Association time
     - :math:`\tau_a`
     - 0.1
     - (1e-6, 1e4)
     - s
     - Time for loop to become bridge
   * - Force sensitivity
     - :math:`\nu`
     - 1.0
     - (0.01, 20)
     - dimensionless
     - Bell exponent, controls thinning onset
   * - Equilibrium bridge fraction
     - :math:`f_{B,eq}`
     - 0.5
     - (0.01, 0.99)
     - dimensionless
     - Bridge fraction at rest
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0
     - (0.0, 1e4)
     - Pa·s
     - Background viscosity

**Consistency Constraint:**

The parameter :math:`f_{B,eq}` is not independent: it should satisfy

.. math::

   f_{B,eq} = \frac{\tau_b}{\tau_a + \tau_b}

within experimental uncertainty. Fitting can either:

1. Fix :math:`f_{B,eq}` to this relation (5-parameter fit)
2. Allow independent variation to test the model (6-parameter fit with consistency check)

**Parameter Scaling:**

For numerical stability, timescales should satisfy:

.. math::

   10^{-6} \leq \frac{\tau_a}{\tau_b} \leq 10^6

Extreme ratios can cause stiffness in the ODE system.

----

7. Parameter Interpretation
===========================================================

Network Modulus (:math:`G`)
-----------------------------------------------------------

**Physical meaning:** Elastic modulus if all chains were bridges (:math:`f_B = 1`).

Related to chain density and temperature:

.. math::

   G = n_{\text{chain}} k_B T

where :math:`n_{\text{chain}}` is the number density of polymer chains.

**Fitting:** Determined from the plateau modulus in SAOS:

.. math::

   G_0' = G f_{B,eq}

If :math:`f_{B,eq}` is known independently, :math:`G` is directly obtained.

**Typical values:**

- Dilute HEUR (1 wt%): :math:`G \sim 10` to :math:`100` Pa
- Semi-dilute (5 wt%): :math:`G \sim 100` to :math:`1000` Pa
- Concentrated (>10 wt%): :math:`G \sim 1000` to :math:`10000` Pa

Bridge Lifetime (:math:`\tau_b`)
-----------------------------------------------------------

**Physical meaning:** Average time a bridge remains attached before detaching (at equilibrium, :math:`\mathbf{S} = \mathbf{I}`).

Controlled by:

- Sticker-micelle interaction energy :math:`\Delta G_{\text{bind}}` (higher → longer :math:`\tau_b`)
- Temperature (higher → shorter :math:`\tau_b`, Arrhenius)
- Micelle aggregation number (larger → more stable)

**Fitting:** Terminal relaxation time from SAOS:

.. math::

   \tau_b \approx \frac{1}{\omega_c}

where :math:`\omega_c` is the crossover frequency (:math:`G' = G''`).

**Typical values:**

- Fast dynamics (low molecular weight): :math:`\tau_b \sim 0.01` to :math:`0.1` s
- Moderate: :math:`\tau_b \sim 0.1` to :math:`10` s
- Slow (strong association): :math:`\tau_b \sim 10` to :math:`1000` s

Association Time (:math:`\tau_a`)
-----------------------------------------------------------

**Physical meaning:** Time for a loop to convert into a bridge.

Controlled by:

- Sticker diffusion rate (how fast a free end finds a new micelle)
- Micelle density (higher → shorter :math:`\tau_a`)
- Chain flexibility (stiffer → longer :math:`\tau_a`)

**Fitting:** Determines the rate of shear thickening. Compare transient startup at different :math:`\dot{\gamma}`:

- If thickening occurs at :math:`t \sim \tau_a`, association is being probed
- Faster thickening → shorter :math:`\tau_a`

Also affects the width of the thickening regime in the flow curve.

**Typical values:**

- Fast reassociation: :math:`\tau_a \sim 0.001` to :math:`0.01` s
- Moderate: :math:`\tau_a \sim 0.01` to :math:`1` s
- Slow (diffusion-limited): :math:`\tau_a \sim 1` to :math:`100` s

**Ratio interpretation:**

- :math:`\tau_a / \tau_b < 1` → Most chains are bridges at rest (:math:`f_{B,eq} > 0.5`)
- :math:`\tau_a / \tau_b > 1` → Most chains are loops at rest (:math:`f_{B,eq} < 0.5`), large thickening potential

Force Sensitivity (:math:`\nu`)
-----------------------------------------------------------

**Physical meaning:** Strength of force-dependent dissociation (Bell exponent).

Related to the activation distance :math:`\delta` over which the sticker must move to escape the micelle:

.. math::

   \nu \sim \frac{\delta}{a}

where :math:`a` is the monomer size.

**Fitting:** Controls the onset of shear thinning:

- High :math:`\nu` → steep increase in :math:`k_{off}` with extension → early thinning
- Low :math:`\nu` → weak force dependence → extended thickening regime

**Experimental signature:** Slope of :math:`\log(\eta)` vs :math:`\log(\dot{\gamma})` in the thinning regime:

.. math::

   \frac{d \log \eta}{d \log \dot{\gamma}} \propto -\nu

**Typical values:**

- Weak force sensitivity: :math:`\nu \sim 0.1` to :math:`0.5`
- Moderate: :math:`\nu \sim 0.5` to :math:`2`
- Strong: :math:`\nu \sim 2` to :math:`10`

Equilibrium Bridge Fraction (:math:`f_{B,eq}`)
-----------------------------------------------------------

**Physical meaning:** Fraction of chains in bridge state at rest.

Determined by free energy balance:

.. math::

   f_{B,eq} = \frac{\tau_b}{\tau_a + \tau_b}

**Independent measurement:** Estimate from plateau modulus if :math:`G` is known:

.. math::

   f_{B,eq} = \frac{G_0'}{G}

**Fitting strategy:**

1. Fit :math:`\tau_a`, :math:`\tau_b` from rheology
2. Compute :math:`f_{B,eq}` from ratio
3. Check consistency with :math:`G_0' / G`

Discrepancies indicate:

- Model inadequacy (e.g., dangling ends, super-bridges)
- Parameter correlation (need more experimental constraints)

**Typical values:**

- Loop-dominated: :math:`f_{B,eq} \sim 0.1` to :math:`0.4`
- Balanced: :math:`f_{B,eq} \sim 0.4` to :math:`0.6`
- Bridge-dominated: :math:`f_{B,eq} \sim 0.6` to :math:`0.9`

Solvent Viscosity (:math:`\eta_s`)
-----------------------------------------------------------

**Physical meaning:** Background viscosity from solvent and unassociated chains.

For aqueous HEUR: :math:`\eta_s \approx \eta_{\text{water}} \sim 0.001` Pa·s (often negligible).

For concentrated solutions: :math:`\eta_s` includes polymer overlap contributions.

**Fitting:** High-shear limiting viscosity:

.. math::

   \eta_{\infty} = \eta_s

as :math:`f_B \to 0` and :math:`\mathbf{S} \to \mathbf{I}` at :math:`\dot{\gamma} \to \infty`.

**Typical values:**

- Dilute aqueous: :math:`\eta_s \sim 0.001` to :math:`0.01` Pa·s
- Semi-dilute: :math:`\eta_s \sim 0.01` to :math:`1` Pa·s
- Polymer melt: :math:`\eta_s \sim 1` to :math:`1000` Pa·s

----

8. Validity and Assumptions
===========================================================

The TNT Loop-Bridge model makes the following simplifications:

**1. Two-Species Approximation**

Assumes only two states: bridges and loops. Neglects:

- **Dangling ends** (one sticker attached, one free)
- **Free chains** (both stickers detached)
- **Super-bridges** (one chain bridging multiple micelles)
- **Multi-loop configurations**

**Valid when:** Sticker-micelle binding energy is high (:math:`\Delta G_{\text{bind}} \gg k_B T`), so free ends are rare.

**2. Mean-Field Conformation**

All bridges have the same average conformation :math:`\mathbf{S}`. Neglects:

- Distribution of chain extensions
- Spatial heterogeneity
- Correlation between orientation and local stress

**Valid when:** Moderate extensions (:math:`\text{tr}(\mathbf{S}) < 10`), homogeneous flows, fast equilibration.

**3. Instantaneous Loop Reformation**

When a bridge detaches, the loop immediately equilibrates to :math:`\mathbf{S} = \mathbf{I}`. Neglects:

- Memory of previous conformation
- Transient stretching of loops

**Valid when:** Loop relaxation time :math:`\ll \tau_b` (loops are much faster than bridges).

**4. Affine Deformation**

Chain conformation follows the macroscopic velocity gradient (upper-convected derivative). Neglects:

- Chain slip
- Non-affine motion near junctions
- Brownian fluctuations

**Valid when:** :math:`\text{Wi} < 10` to :math:`100` (moderate Weissenberg number).

**5. No Hydrodynamic Interactions**

Micelles are treated as fixed junctions; solvent flow around micelles is ignored. Neglects:

- Micelle mobility
- Cooperative rearrangements

**Valid when:** Micelle size :math:`\ll` chain contour length, dilute to semi-dilute regime.

**6. Homogeneous Flow**

Assumes no spatial gradients (0D model). Cannot describe:

- Shear banding
- Flow instabilities
- Spatial heterogeneity

**Valid when:** Gap width :math:`\ll` characteristic length scale of structure evolution.

**7. Bell-Type Dissociation**

Force-dependent detachment follows exponential form. Neglects:

- Non-Gaussian chain statistics (finite extensibility)
- Stick-slip dynamics
- Multiple activation pathways

**Valid when:** Extensions are moderate (:math:`\text{tr}(\mathbf{S}) < 10` to :math:`20`).

**When to Use This Model:**

- HEUR solutions in dilute to semi-dilute regime
- Telechelic ionomers with well-defined end-groups
- Flower micelle networks with two-state kinetics
- Shear thickening followed by thinning in flow curves
- Stress overshoot in startup flows

**When NOT to Use:**

- High extensions (:math:`\text{tr}(\mathbf{S}) > 20`) → need finite extensibility (FENE-type)
- Shear banding → need spatial (1D) model
- Multiple relaxation processes → need more species or modes
- Chain entanglements dominate → need reptation-based model

----

9. Regimes and Behavior
===========================================================

Linear Viscoelastic Regime (:math:`\text{Wi} \ll 1`)
-----------------------------------------------------------

**Behavior:** Single Maxwell-like relaxation with effective modulus.

**Modulus:**

.. math::

   G_0' = G f_{B,eq}

**Relaxation time:** :math:`\tau_b` (bridge detachment controls terminal relaxation).

**Population:** :math:`f_B \approx f_{B,eq}` (small perturbations).

**Key prediction:** Storage modulus plateau at high frequency reflects the equilibrium bridge fraction.

Shear Thickening Regime (:math:`\text{Wi}_a \sim 1`, :math:`\text{Wi}_b < 1`)
-----------------------------------------------------------

**Conditions:** :math:`\dot{\gamma} \tau_a \sim 1`, extensions are still moderate.

**Mechanism:** Flow promotes loop-to-bridge conversion faster than force-dependent detachment.

**Bridge fraction:** :math:`f_B > f_{B,eq}` (excess bridges).

**Viscosity:** Increases above zero-shear value:

.. math::

   \eta(\dot{\gamma}) > \eta_0 = \frac{G f_{B,eq} \tau_b}{1 + \eta_s / (G f_{B,eq} \tau_b)}

**Experimental signature:** Positive slope in :math:`\log \eta` vs :math:`\log \dot{\gamma}` plot.

**Material examples:** HEUR at 5-10 wt%, moderate shear rates (10-100 1/s).

Critical Shear Rate (:math:`\dot{\gamma}^*`)
-----------------------------------------------------------

**Definition:** Shear rate at which viscosity is maximum.

**Scaling estimate:**

.. math::

   \dot{\gamma}^* \sim \frac{1}{\nu \tau_b}

**Physical picture:** Balance between:

- Thickening (loop-to-bridge conversion, rate :math:`\sim 1/\tau_a`)
- Thinning (force-dependent detachment, rate :math:`\sim (1/\tau_b) e^{\nu (\text{tr}(\mathbf{S}) - 3)}`)

At :math:`\dot{\gamma}^*`, extensions are :math:`\text{tr}(\mathbf{S}) \sim 5` to :math:`10`.

**Parameter dependence:**

- Large :math:`\nu` → low :math:`\dot{\gamma}^*` (early onset of thinning)
- Large :math:`\tau_b` → low :math:`\dot{\gamma}^*` (longer-lived bridges are more sensitive to force)

Shear Thinning Regime (:math:`\text{Wi}_b \gg 1`)
-----------------------------------------------------------

**Conditions:** :math:`\dot{\gamma} \tau_b \gg 1`, high extensions :math:`\text{tr}(\mathbf{S}) \gg 3`.

**Mechanism:** Force-dependent detachment dominates, :math:`k_{off} \gg k_{on}`.

**Bridge fraction:** :math:`f_B < f_{B,eq}` (depletion of bridges).

**Viscosity:** Power-law decrease:

.. math::

   \eta(\dot{\gamma}) \sim \dot{\gamma}^{-p}

where :math:`p \sim 0.5` to :math:`1` (depends on :math:`\nu`).

**Experimental signature:** Negative slope in :math:`\log \eta` vs :math:`\log \dot{\gamma}`, stress plateau or weak increase.

**Material examples:** HEUR at high shear (>1000 1/s), approaching :math:`\eta_s`.

Startup Transient Behavior
-----------------------------------------------------------

**Stress overshoot:** Occurs when :math:`d\sigma/dt = 0` before steady state.

**Overshoot time:** :math:`t_{\text{peak}} \sim \tau_b` (bridge relaxation).

**Overshoot magnitude:** Depends on :math:`\text{Wi}_a`:

- Low :math:`\text{Wi}_a` (:math:`< 0.1`): Weak overshoot, :math:`\sigma_{\text{peak}} \approx 1.1 \sigma_{ss}`
- Moderate :math:`\text{Wi}_a` (:math:`\sim 1`): Strong overshoot with transient thickening, :math:`\sigma_{\text{peak}} > 1.5 \sigma_{ss}`
- High :math:`\text{Wi}_a` (:math:`> 10`): Overshoot returns to steady-state thinning

**Transient thickening:** If :math:`f_B(t)` increases faster than :math:`S_{xy}(t)` grows, stress can exceed the final value before relaxing to steady state.

This is a signature of loop-to-bridge conversion and distinguishes telechelic networks from simple Maxwell models.

Creep Compliance
-----------------------------------------------------------

**Step stress:** :math:`\sigma_0` applied at :math:`t = 0`.

**Short-time behavior:** Elastic response :math:`\gamma \sim \sigma_0 / (G f_{B,eq})`.

**Intermediate time:** Population redistribution, :math:`f_B` adjusts to new stress.

**Long-time behavior:** Viscous flow :math:`\gamma \sim t / \eta_s` if :math:`\eta_s > 0`.

**Compliance:**

.. math::

   J(t) = \frac{\gamma(t)}{\sigma_0}

Shows double-exponential relaxation (bridge relaxation + population adjustment) followed by linear flow.

Creep Rupture Via Bridge Collapse
-----------------------------------------------------------

Under sustained applied stress, the bridge fraction can progressively decrease:

1. Stress stretches bridge chains, increasing their breakage rate
2. Reduced :math:`f_B` decreases the effective modulus :math:`G_{\text{eff}} = G \cdot f_B`
3. Lower modulus means higher strain for the same stress → more chain stretch
4. Positive feedback: above a critical stress, :math:`f_B \to 0` (all chains become loops)

This **bridge collapse** mechanism produces creep rupture — delayed catastrophic failure
under sustained load. The critical stress depends on the ratio :math:`k_{BL}/k_{LB}` and
the force sensitivity of the bridge-to-loop transition.

Oscillatory Shear (SAOS and LAOS)
-----------------------------------------------------------

**SAOS (small amplitude):** Two relaxation processes:

1. Bridge conformation (time :math:`\tau_b`)
2. Population redistribution (time :math:`\tau_{\text{pop}} = \tau_a \tau_b / (\tau_a + \tau_b)`)

**Cole-Cole plot:** :math:`G''` vs :math:`G'` shows deviation from single Maxwell (two arcs).

**LAOS (large amplitude):** Population cycling — :math:`f_B(t)` oscillates due to periodic stretching/relaxation.

Higher harmonics appear when :math:`k_{off}(t)` becomes nonlinear (high strain amplitudes).

----

10. What You Can Learn from Fitting This Model
===========================================================

Equilibrium Bridge Fraction
-----------------------------------------------------------

**From:** Plateau modulus in SAOS, :math:`G_0'`.

**Relation:**

.. math::

   f_{B,eq} = \frac{G_0'}{G}

**Interpretation:** Fraction of chains actively contributing to network stress at rest.

**Application:** Design of thickeners (maximize :math:`f_{B,eq}` for high viscosity at rest).

Bridge Lifetime and Association Time
-----------------------------------------------------------

**From:** Terminal relaxation frequency :math:`\omega_c` (where :math:`G' = G''`) and flow curve thickening regime.

**Relations:**

.. math::

   \tau_b &\sim \frac{1}{\omega_c} \\
   \tau_a &\sim \frac{1}{\dot{\gamma}_{\text{thick}}}

where :math:`\dot{\gamma}_{\text{thick}}` is the shear rate where thickening begins.

**Interpretation:**

- :math:`\tau_b`: Sticker-micelle binding strength
- :math:`\tau_a`: Chain diffusion and reassociation rate

**Application:** Tuning molecular architecture (e.g., end-group hydrophobicity) to control timescales.

Force Sensitivity (Bell Exponent)
-----------------------------------------------------------

**From:** Slope of thinning regime in flow curve, onset of thinning :math:`\dot{\gamma}^*`.

**Relation:**

.. math::

   \nu \sim \frac{\log(k_{off,\text{high}} / k_{off,0})}{\text{tr}(\mathbf{S}_{\text{high}}) - 3}

**Interpretation:** Sensitivity of detachment to chain extension (activation barrier shape).

**Application:** Predicting failure under high-rate deformation (e.g., extrusion, spraying).

Shear Thickening Amplitude
-----------------------------------------------------------

**From:** Maximum viscosity :math:`\eta_{\text{max}}` compared to zero-shear :math:`\eta_0`.

**Relation:**

.. math::

   \frac{\eta_{\text{max}}}{\eta_0} \propto \frac{f_{B,\text{max}}}{f_{B,eq}}

where :math:`f_{B,\text{max}}` is the peak bridge fraction.

**Interpretation:** Amount of excess bridging achievable under flow.

**Dependence:**

- Large :math:`\tau_a / \tau_b` ratio → large thickening (more loops available for conversion)
- Small :math:`\nu` → sustained thickening (weak force dependence)

**Application:** Optimizing thickening efficiency in coatings, adhesives.

Onset of Shear Thinning
-----------------------------------------------------------

**From:** Critical shear rate :math:`\dot{\gamma}^*` where :math:`\eta(\dot{\gamma})` is maximum.

**Relation:**

.. math::

   \dot{\gamma}^* \sim \frac{1}{\nu \tau_b}

**Interpretation:** Transition from association-dominated to detachment-dominated regime.

**Application:** Processing windows for molding, extrusion (avoid thinning).

Stress Overshoot and Transient Thickening
-----------------------------------------------------------

**From:** Startup experiments at various :math:`\dot{\gamma}`.

**Metrics:**

- Overshoot magnitude: :math:`\sigma_{\text{peak}} / \sigma_{ss}`
- Overshoot time: :math:`t_{\text{peak}}`
- Transient thickening: Does :math:`\sigma(t) > \sigma_{ss}` occur?

**Interpretation:**

- Large overshoot → significant population redistribution
- Transient thickening → loop-to-bridge conversion is fast relative to conformation evolution

**Application:** Predicting behavior in rapid deformations (printing, dispensing).

Network Connectivity
-----------------------------------------------------------

**From:** Comparison of :math:`f_{B,eq}` (from :math:`G_0'`) with :math:`\tau_a / (\tau_a + \tau_b)` (from timescales).

**Consistency check:** Should match within 10-20%.

**Discrepancies indicate:**

- Presence of dangling ends (reduces :math:`G_0'`)
- Multi-species populations (more than loops/bridges)
- Spatial heterogeneity (micelle clustering)

**Application:** Validating model assumptions, guiding molecular design.

----

11. Experimental Design and Data Requirements
===========================================================

Small-Amplitude Oscillatory Shear (SAOS)
-----------------------------------------------------------

**Objective:** Measure :math:`G'(\omega)` and :math:`G''(\omega)` to extract:

- Plateau modulus :math:`G_0'` → :math:`f_{B,eq}` (if :math:`G` known)
- Terminal relaxation time :math:`\tau_b`
- Population relaxation time :math:`\tau_{\text{pop}}`

**Protocol:**

1. Strain sweep to identify linear regime (typically :math:`\gamma_0 < 0.1`)
2. Frequency sweep from :math:`\omega = 10^{-2}` to :math:`10^2` rad/s
3. Check for low-frequency plateau (terminal behavior) and high-frequency plateau (:math:`G_0'`)

**Expected features:**

- Two relaxation processes (double-Maxwellian)
- :math:`G'` plateau at high :math:`\omega` → :math:`G_0' = G f_{B,eq}`
- Crossover at :math:`\omega_c \sim 1/\tau_b`

**Data quality:**

- At least 10 points per decade in :math:`\omega`
- Signal-to-noise ratio >100:1 for :math:`G'`, :math:`G''`
- Temperature control :math:`\pm 0.1` K (association is thermally activated)

Steady Shear Flow Curve
-----------------------------------------------------------

**Objective:** Measure :math:`\sigma(\dot{\gamma})` or :math:`\eta(\dot{\gamma})` to observe:

- Shear thickening at moderate :math:`\dot{\gamma}`
- Viscosity maximum at :math:`\dot{\gamma}^*`
- Shear thinning at high :math:`\dot{\gamma}`

**Protocol:**

1. Shear rate sweep from :math:`\dot{\gamma} = 10^{-3}` to :math:`10^3` 1/s (log-spaced)
2. Allow steady state at each point (wait :math:`> 5 \tau_b`)
3. Check reversibility (up-down sweep) to detect thixotropy

**Expected features:**

- Zero-shear plateau :math:`\eta_0` at low :math:`\dot{\gamma}`
- Viscosity peak at :math:`\dot{\gamma}^*`
- Power-law thinning at high :math:`\dot{\gamma}`
- High-shear plateau :math:`\eta_\infty = \eta_s`

**Data quality:**

- At least 15-20 points covering the full curve
- Steady-state criterion: :math:`|\dot{\sigma}/\sigma| < 0.01` over time :math:`\sim \tau_b`
- Check for edge fracture, slip at high :math:`\dot{\gamma}`

Startup Shear Flow
-----------------------------------------------------------

**Objective:** Measure transient :math:`\sigma(t)` after step :math:`\dot{\gamma}` to extract:

- Stress overshoot magnitude and time
- Transient thickening (if :math:`\sigma(t) > \sigma_{ss}` before overshoot)
- Approach to steady state

**Protocol:**

1. Pre-shear at low :math:`\dot{\gamma}` to equilibrate
2. Rest for time :math:`> 10 \tau_b` (return to :math:`f_B = f_{B,eq}`)
3. Step to target :math:`\dot{\gamma}`, measure :math:`\sigma(t)` for :math:`t = 0` to :math:`10 \tau_b`
4. Repeat for multiple :math:`\dot{\gamma}` spanning thickening and thinning regimes

**Expected features:**

- Overshoot at :math:`t \sim \tau_b`
- Transient thickening for :math:`\dot{\gamma} \sim 1/\tau_a`
- Steady-state approach as :math:`\sigma(t) \to \sigma_{ss}`

**Data quality:**

- Sampling rate :math:`> 10 / \tau_b` (resolve overshoot peak)
- Repeat 3-5 times to check reproducibility
- Monitor strain: :math:`\gamma(t) = \int \dot{\gamma} dt` should be linear

Creep Compliance
-----------------------------------------------------------

**Objective:** Measure :math:`\gamma(t)` after step stress :math:`\sigma_0` to extract:

- Initial compliance :math:`J_0 = 1/(G f_{B,eq})`
- Retardation times :math:`\tau_b`, :math:`\tau_{\text{pop}}`
- Long-time viscous flow :math:`\gamma \sim t/\eta_s`

**Protocol:**

1. Apply constant stress :math:`\sigma_0` (within linear regime if possible)
2. Measure :math:`\gamma(t)` for :math:`t = 0` to :math:`100 \tau_b`
3. Repeat for multiple :math:`\sigma_0` to check linearity

**Expected features:**

- Instantaneous compliance jump :math:`J_0`
- Retardation (slow relaxation) at intermediate times
- Linear flow :math:`J(t) \sim t/\eta_s` if :math:`\eta_s > 0`

**Data quality:**

- High-resolution strain measurement (resolution :math:`< 0.001`)
- Check for instrument compliance (subtract out)
- Temperature drift :math:`< 0.1` K over measurement

Large-Amplitude Oscillatory Shear (LAOS)
-----------------------------------------------------------

**Objective:** Measure higher harmonics in :math:`\sigma(t)` during oscillation to probe:

- Population cycling :math:`f_B(t)`
- Nonlinear force-dependent detachment
- Intracycle shear thickening-thinning

**Protocol:**

1. Apply sinusoidal strain :math:`\gamma(t) = \gamma_0 \sin(\omega t)`
2. Vary :math:`\gamma_0` from 0.1 (linear) to 5 (highly nonlinear)
3. Measure stress waveform :math:`\sigma(t)` at high resolution (>100 points per cycle)
4. Fourier decompose to extract harmonics :math:`G_1'(\omega), G_1''(\omega), G_3'(\omega), G_3''(\omega), \ldots`

**Expected features:**

- Odd harmonics dominate (material symmetry)
- Third harmonic :math:`G_3'` increases with :math:`\gamma_0` (nonlinear elastic response)
- Intracycle thickening-thinning visible in Lissajous plots

**Data quality:**

- Waveform sampling >200 points/cycle
- Multiple cycles (10+) to check periodicity
- Fourier truncation error <1%

Recommended Test Matrix
-----------------------------------------------------------

**Minimum for model validation:**

1. SAOS frequency sweep (1 test)
2. Flow curve (1 test)
3. Startup at 3-5 shear rates (spanning regimes)

**Comprehensive characterization:**

1. SAOS frequency sweep
2. Flow curve (up-down sweep to check reversibility)
3. Startup at 10 shear rates (log-spaced)
4. Creep at 3-5 stress levels
5. LAOS at 3 strain amplitudes

**Total time:** 4-8 hours for comprehensive suite.

Temperature Control
-----------------------------------------------------------

Association kinetics are thermally activated:

.. math::

   \tau_b(T) \sim \exp\left[\frac{\Delta G_{\text{bind}}}{k_B T}\right]

**Requirement:** Temperature control :math:`\pm 0.1` K to avoid drift in :math:`\tau_a`, :math:`\tau_b`.

**Protocol:** Equilibrate sample for 10-15 minutes at target temperature before testing.

----

12. Computational Implementation
===========================================================

State Variables
-----------------------------------------------------------

For rate-controlled flows (imposed :math:`\dot{\gamma}`):

.. math::

   \mathbf{y} = [f_B, S_{xx}, S_{yy}, S_{zz}, S_{xy}]^T

For stress-controlled flows (creep):

.. math::

   \mathbf{y} = [f_B, S_{xx}, S_{yy}, S_{zz}, S_{xy}, \gamma]^T

ODE Right-Hand Side (Rate-Controlled)
-----------------------------------------------------------

Implemented in ``rheojax.models.tnt._kernels.tnt_loop_bridge_ode_rhs``:

.. code-block:: python

   def tnt_loop_bridge_ode_rhs(y, t, gamma_dot, params):
       """
       y = [f_B, S_xx, S_yy, S_zz, S_xy]
       params = [G, tau_b, tau_a, nu, f_B_eq, eta_s]
       """
       f_B, S_xx, S_yy, S_zz, S_xy = y
       G, tau_b, tau_a, nu, f_B_eq, eta_s = params

       # Trace of conformation tensor
       tr_S = S_xx + S_yy + S_zz

       # Force-dependent detachment rate
       k_off = (1.0 / tau_b) * jnp.exp(nu * (tr_S - 3.0))

       # Population kinetics
       df_B_dt = (1.0 - f_B) / tau_a - f_B * k_off

       # Conformation evolution (upper-convected)
       dS_xx_dt = 2.0 * gamma_dot * S_xy - k_off * (S_xx - 1.0)
       dS_yy_dt = -k_off * (S_yy - 1.0)
       dS_zz_dt = -k_off * (S_zz - 1.0)
       dS_xy_dt = gamma_dot * S_yy - k_off * S_xy

       return jnp.array([df_B_dt, dS_xx_dt, dS_yy_dt, dS_zz_dt, dS_xy_dt])

**JAX compilation:** Use ``jax.jit`` for 10-50x speedup:

.. code-block:: python

   ode_rhs_jit = jax.jit(tnt_loop_bridge_ode_rhs, static_argnums=(2,))

ODE Right-Hand Side (Creep)
-----------------------------------------------------------

For imposed stress :math:`\sigma_0`, add implicit equation for :math:`\dot{\gamma}(t)`:

.. code-block:: python

   def tnt_loop_bridge_creep_rhs(y, t, sigma_0, params):
       """
       y = [f_B, S_xx, S_yy, S_zz, S_xy, gamma]
       """
       f_B, S_xx, S_yy, S_zz, S_xy, gamma = y
       G, tau_b, tau_a, nu, f_B_eq, eta_s = params

       # Current stress (implicit equation)
       sigma_xy = G * f_B * S_xy + eta_s * gamma_dot

       # Solve for gamma_dot: sigma_xy = sigma_0
       gamma_dot = (sigma_0 - G * f_B * S_xy) / eta_s

       # Rest is same as rate-controlled
       tr_S = S_xx + S_yy + S_zz
       k_off = (1.0 / tau_b) * jnp.exp(nu * (tr_S - 3.0))

       df_B_dt = (1.0 - f_B) / tau_a - f_B * k_off
       dS_xx_dt = 2.0 * gamma_dot * S_xy - k_off * (S_xx - 1.0)
       dS_yy_dt = -k_off * (S_yy - 1.0)
       dS_zz_dt = -k_off * (S_zz - 1.0)
       dS_xy_dt = gamma_dot * S_yy - k_off * S_xy
       d_gamma_dt = gamma_dot

       return jnp.array([df_B_dt, dS_xx_dt, dS_yy_dt, dS_zz_dt, dS_xy_dt, d_gamma_dt])

Initial Conditions
-----------------------------------------------------------

For all transient protocols, start from equilibrium:

.. code-block:: python

   y0 = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0])  # [f_B, S_xx, S_yy, S_zz, S_xy]

For creep, add :math:`\gamma(0) = 0`:

.. code-block:: python

   y0_creep = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0, 0.0])

Steady-State Root-Finding (Flow Curve)
-----------------------------------------------------------

For flow curve, solve :math:`dy/dt = 0` using Newton-Raphson:

.. code-block:: python

   from jax.scipy.optimize import root_scalar

   def residual(y_ss, gamma_dot, params):
       dy_dt = tnt_loop_bridge_ode_rhs(y_ss, 0.0, gamma_dot, params)
       return dy_dt

   # Initial guess: equilibrium
   y_guess = jnp.array([f_B_eq, 1.0, 1.0, 1.0, 0.0])

   # Solve for each gamma_dot
   y_ss = root_scalar(residual, y_guess, args=(gamma_dot, params))

**Bracketing:** For non-monotonic viscosity, root-finding may fail. Use continuation from low to high :math:`\dot{\gamma}`, using previous solution as initial guess.

SAOS Linearization
-----------------------------------------------------------

For small-amplitude oscillatory shear, linearize around equilibrium:

.. math::

   G^*(\omega) = G f_{B,eq} \frac{i\omega \tau_b}{1 + i\omega \tau_b}
   \left(1 + \frac{\nu f_{B,eq}}{1 + i\omega \tau_{\text{pop}}}\right)

where :math:`\tau_{\text{pop}} = \tau_a \tau_b / (\tau_a + \tau_b)`.

Implemented analytically (no ODE solve needed):

.. code-block:: python

   def saos_modulus(omega, params):
       G, tau_b, tau_a, nu, f_B_eq, eta_s = params
       tau_pop = (tau_a * tau_b) / (tau_a + tau_b)

       # Bridge relaxation
       G_bridge = G * f_B_eq * (1j * omega * tau_b) / (1 + 1j * omega * tau_b)

       # Population correction
       G_pop_corr = (nu * f_B_eq) / (1 + 1j * omega * tau_pop)

       G_star = G_bridge * (1 + G_pop_corr)
       return G_star.real, G_star.imag  # (G', G'')

LAOS Simulation
-----------------------------------------------------------

For large-amplitude oscillatory shear :math:`\gamma(t) = \gamma_0 \sin(\omega t)`:

.. code-block:: python

   # Time-dependent strain rate
   def gamma_dot_laos(t, gamma_0, omega):
       return gamma_0 * omega * jnp.cos(omega * t)

   # Solve ODE with time-varying gamma_dot
   def ode_rhs_laos(y, t, gamma_0, omega, params):
       gamma_dot = gamma_dot_laos(t, gamma_0, omega)
       return tnt_loop_bridge_ode_rhs(y, t, gamma_dot, params)

   # Integrate for 10 cycles
   t_end = 10 * (2 * jnp.pi / omega)
   sol = odeint(ode_rhs_laos, y0, t_span, args=(gamma_0, omega, params))

Extract harmonics via FFT after discarding transient cycles (first 5).

Numerical Stability
-----------------------------------------------------------

**Stiffness:** When :math:`\tau_a \ll \tau_b` or :math:`\nu` is large, the system becomes stiff (fast and slow timescales).

**Solution:** Use implicit solver (e.g., ``jax-scipy.integrate.solve_ivp`` with ``method='BDF'``).

**Adaptive timestepping:** Essential for capturing overshoot peak and steady-state approach.

**Tolerances:** Set ``rtol=1e-6``, ``atol=1e-8`` for accurate stress predictions.

**Vectorization:** Use ``jax.vmap`` to solve for multiple :math:`\dot{\gamma}` in parallel:

.. code-block:: python

   solve_startup_vmap = jax.vmap(solve_startup, in_axes=(None, 0, None))
   # Vectorize over gamma_dot array
   sigma_array = solve_startup_vmap(y0, gamma_dot_array, params)

----

13. Fitting Guidance and Best Practices
===========================================================

Parameter Initialization
-----------------------------------------------------------

**Step 1: Estimate** :math:`G` **and** :math:`f_{B,eq}` **from SAOS:**

Plateau modulus at high frequency:

.. math::

   G_0' \approx G f_{B,eq}

If :math:`f_{B,eq}` is unknown, assume :math:`f_{B,eq} \approx 0.5` (balanced) as initial guess.

**Step 2: Estimate** :math:`\tau_b` **from terminal relaxation:**

Crossover frequency :math:`\omega_c` where :math:`G' = G''`:

.. math::

   \tau_b \approx \frac{1}{\omega_c}

**Step 3: Estimate** :math:`\tau_a` **from consistency:**

If :math:`f_{B,eq}` is known:

.. math::

   \tau_a = \tau_b \left(\frac{1 - f_{B,eq}}{f_{B,eq}}\right)

**Step 4: Estimate** :math:`\nu` **from flow curve slope:**

Onset of thinning :math:`\dot{\gamma}^*`:

.. math::

   \nu \sim \frac{1}{\tau_b \dot{\gamma}^*}

**Step 5:** Set :math:`\eta_s = 0` initially (negligible solvent viscosity).

Fitting Strategy (Hierarchical)
-----------------------------------------------------------

**Stage 1: SAOS only** (fix :math:`\nu`, :math:`\eta_s = 0`)

Fit :math:`G`, :math:`\tau_b`, :math:`\tau_a` (or :math:`f_{B,eq}`) to :math:`G'(\omega)`, :math:`G''(\omega)`.

**Objective:** Minimize

.. math::

   \chi^2 = \sum_i \left[\frac{G'_{\text{pred}}(\omega_i) - G'_{\text{data}}(\omega_i)}{\sigma_{G'}}\right]^2
   + \sum_i \left[\frac{G''_{\text{pred}}(\omega_i) - G''_{\text{data}}(\omega_i)}{\sigma_{G''}}\right]^2

**Stage 2: Add flow curve** (release :math:`\nu`)

Fix :math:`G`, :math:`\tau_b`, :math:`\tau_a` from Stage 1. Fit :math:`\nu` to :math:`\eta(\dot{\gamma})`.

**Objective:** Minimize

.. math::

   \chi^2 = \sum_j \left[\frac{\eta_{\text{pred}}(\dot{\gamma}_j) - \eta_{\text{data}}(\dot{\gamma}_j)}{\sigma_{\eta}}\right]^2

**Stage 3: Global refinement** (all parameters)

Use Stage 1-2 results as initial guess. Fit all parameters simultaneously to SAOS + flow curve.

**Stage 4: Add transient data** (validate)

Fix all parameters from Stage 3. Predict startup :math:`\sigma(t)` and compare to data (no fitting).

If discrepancies exist, refine :math:`\tau_a` or :math:`\nu` slightly.

Parameter Bounds
-----------------------------------------------------------

Always enforce physical bounds:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Parameter
     - Lower Bound
     - Upper Bound
   * - :math:`G`
     - 1 Pa
     - :math:`10^8` Pa
   * - :math:`\tau_b`
     - :math:`10^{-6}` s
     - :math:`10^4` s
   * - :math:`\tau_a`
     - :math:`10^{-6}` s
     - :math:`10^4` s
   * - :math:`\nu`
     - 0.01
     - 20
   * - :math:`f_{B,eq}`
     - 0.01
     - 0.99
   * - :math:`\eta_s`
     - 0 Pa·s
     - :math:`10^4` Pa·s

Weighting of Data
-----------------------------------------------------------

**Log-space weighting:** For quantities spanning decades (e.g., :math:`\eta`, :math:`\omega`), use log-space residuals:

.. math::

   \chi^2 = \sum_i \left[\log(\eta_{\text{pred}}) - \log(\eta_{\text{data}})\right]^2

This prevents high-viscosity points from dominating the fit.

**Uncertainty weighting:** If experimental uncertainties :math:`\sigma_i` are known, use inverse-variance weighting:

.. math::

   \chi^2 = \sum_i \frac{(y_{\text{pred},i} - y_{\text{data},i})^2}{\sigma_i^2}

Consistency Checks
-----------------------------------------------------------

**1. Equilibrium bridge fraction:**

Compare :math:`f_{B,eq}` from fit with :math:`\tau_b / (\tau_a + \tau_b)`. Should agree within 10%.

**2. Plateau modulus:**

Check :math:`G f_{B,eq} \approx G_0'` from SAOS.

**3. Timescale separation:**

Verify :math:`\tau_{\text{pop}} = \tau_a \tau_b / (\tau_a + \tau_b) < \tau_b`.

**4. Critical shear rate:**

Check :math:`\dot{\gamma}^* \sim 1 / (\nu \tau_b)` matches observed viscosity maximum.

Common Pitfalls
-----------------------------------------------------------

**1. Parameter correlation:**

:math:`G` and :math:`f_{B,eq}` are highly correlated in SAOS (only product :math:`G f_{B,eq}` is constrained). Need flow curve or independent :math:`f_{B,eq}` measurement.

**2. Overfitting transient data:**

Startup curves can be fit well with incorrect parameters (e.g., too high :math:`\nu` compensated by too low :math:`\tau_a`). Always validate against multiple protocols.

**3. Ignoring solvent viscosity:**

If high-shear viscosity does not plateau at :math:`\eta_s \approx 0`, :math:`\eta_s > 0` is required. Neglecting it biases :math:`G`, :math:`\tau_b` estimates.

**4. Non-unique solutions:**

Non-monotonic viscosity can have multiple parameter sets giving similar fits. Use physical constraints (e.g., :math:`f_{B,eq} < 0.9` for loop-dominated systems).

Recommended Optimization Algorithm
-----------------------------------------------------------

Use Levenberg-Marquardt (least-squares with Jacobian):

.. code-block:: python

   from scipy.optimize import least_squares

   result = least_squares(
       residual_function,
       x0=initial_guess,
       bounds=(lower_bounds, upper_bounds),
       method='trf',  # Trust-region reflective
       ftol=1e-8,
       xtol=1e-8,
       max_nfev=1000
   )

For global search (if local minima suspected), use basin-hopping or differential evolution first.

----

15. Usage Examples
===========================================================

Example 1: Basic SAOS Prediction
-----------------------------------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTLoopBridge
   import jax.numpy as jnp
   import matplotlib.pyplot as plt

   # Create model with default parameters
   model = TNTLoopBridge()

   # SAOS prediction
   omega = jnp.logspace(-2, 2, 100)
   G_prime, G_double_prime = model.predict(omega, test_mode='oscillation')

   # Plot
   plt.loglog(omega, G_prime, 'o-', label="G'")
   plt.loglog(omega, G_double_prime, 's-', label="G''")
   plt.xlabel('Frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Example 2: Flow Curve with Non-Monotonic Viscosity
-----------------------------------------------------------

.. code-block:: python

   # Predict flow curve
   gamma_dot = jnp.logspace(-3, 3, 50)
   viscosity = model.predict(gamma_dot, test_mode='flow_curve')

   # Plot
   plt.loglog(gamma_dot, viscosity, 'o-')
   plt.xlabel('Shear rate (1/s)')
   plt.ylabel('Viscosity (Pa·s)')
   plt.title('Non-Monotonic Viscosity')
   plt.grid(True, alpha=0.3)
   plt.show()

Example 3: Startup with Transient Thickening
-----------------------------------------------------------

.. code-block:: python

   # Startup at moderate shear rate
   t = jnp.linspace(0, 20, 500)
   gamma_dot = 1.0  # Wi_a ~ 0.1

   stress = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Plot
   plt.plot(t, stress, '-')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.title(f'Startup at gamma_dot = {gamma_dot} 1/s')
   plt.grid(True, alpha=0.3)
   plt.show()

Example 4: Fitting to Experimental Data
-----------------------------------------------------------

.. code-block:: python

   from rheojax.core.data import RheoData

   # Load experimental SAOS data
   omega_exp = ...  # rad/s
   G_star_exp = ...  # Pa (complex)

   # Fit model
   rheo_data = RheoData(x=omega_exp, y=G_star_exp, test_mode='oscillation')
   result = model.fit(rheo_data)

   # Print fitted parameters
   params = model.get_parameter_values()
   print(f"G = {params['G']:.1f} Pa")
   print(f"tau_b = {params['tau_b']:.3f} s")
   print(f"tau_a = {params['tau_a']:.3f} s")
   print(f"nu = {params['nu']:.2f}")
   print(f"f_B_eq = {params['f_B_eq']:.2f}")

Example 5: Bayesian Inference
-----------------------------------------------------------

.. code-block:: python

   # Bayesian inference with NumPyro
   result_bayes = model.fit_bayesian(
       rheo_data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Check convergence
   print(f"R-hat: {result_bayes.diagnostics['r_hat']}")
   print(f"ESS: {result_bayes.diagnostics['ess']}")

   # Credible intervals
   intervals = model.get_credible_intervals(
       result_bayes.posterior_samples,
       credibility=0.95
   )
   print(f"tau_a: [{intervals['tau_a'][0]:.3f}, {intervals['tau_a'][1]:.3f}] s")

Example 6: Parameter Sensitivity
-----------------------------------------------------------

.. code-block:: python

   # Vary nu to see effect on flow curve
   nu_values = [0.5, 1.0, 2.0, 5.0]
   gamma_dot = jnp.logspace(-2, 3, 50)

   for nu in nu_values:
       model.set_parameter('nu', nu)
       eta = model.predict(gamma_dot, test_mode='flow_curve')
       plt.loglog(gamma_dot, eta, label=f'nu = {nu}')

   plt.xlabel('Shear rate (1/s)')
   plt.ylabel('Viscosity (Pa·s)')
   plt.legend()
   plt.title('Effect of Force Sensitivity')
   plt.grid(True, alpha=0.3)
   plt.show()

----

14. Failure Mode: Loop Saturation
===========================================================

Under extreme flow conditions, all bridge chains convert to loops:

.. math::

   f_B \to 0, \quad f_L \to 1

In this limit, the network loses all elasticity (:math:`G_{\text{eff}} \to 0`) and
behaves as a viscous fluid. This **loop saturation** represents the complete destruction
of the stress-bearing network.

**Physical signatures:**

- Flow curve shows a dramatic viscosity drop at high rates
- Startup stress overshoot followed by near-complete stress collapse
- Recovery after flow cessation governed by :math:`\tau_{\text{kin}}` (bridge reformation time)

----

16. See Also
===========================================================

**TNT Shared Reference:**

- :doc:`tnt_protocols` — Full protocol equations and numerical methods
- :doc:`tnt_knowledge_extraction` — Model identification and fitting guidance

**TNT Base Model:**

- :ref:`model-tnt-tanaka-edwards` — Base model (constant breakage, single species)

**Generalizations:**

- :ref:`model-tnt-multi-species` — Generalization to N bond types (LoopBridge is 2-species special case)
- :ref:`model-tnt-sticky-rouse` — Multi-mode alternative for broad relaxation spectra

**Alternative Models for Similar Materials:**

- :ref:`model-tnt-cates` — Living polymer alternative for micellar systems
- :ref:`model-tnt-bell` — Force-dependent breakage (complementary mechanism)

----

17. API Reference
===========================================================

.. autoclass:: rheojax.models.tnt.TNTLoopBridge
   :members:
   :inherited-members:
   :show-inheritance:
   :exclude-members: parameters

**Key Methods:**

- ``fit(rheo_data)``: Fit model to SAOS, flow curve, or transient data using NLSQ
- ``predict(x, test_mode, **kwargs)``: Predict stress, viscosity, or modulus
- ``fit_bayesian(rheo_data, num_warmup, num_samples)``: Bayesian inference with NumPyro NUTS
- ``get_parameter_values()``: Return dict of fitted parameters
- ``set_parameter(name, value)``: Update parameter value

**Test Mode Support:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Test Mode
     - Description
   * - ``FLOW_CURVE``
     - Steady shear: :math:`\eta(\dot{\gamma})` with non-monotonic behavior
   * - ``STARTUP``
     - Transient shear: :math:`\sigma(t)` with overshoot and possible thickening
   * - ``CREEP``
     - Constant stress: :math:`\gamma(t)` with retardation and flow
   * - ``RELAXATION``
     - Stress relaxation: :math:`\sigma(t)` after step strain (bridge-to-loop conversion)
   * - ``OSCILLATION``
     - SAOS: :math:`G'(\omega)`, :math:`G''(\omega)` with two relaxations
   * - ``LAOS``
     - Large amplitude: :math:`\sigma(t)` with higher harmonics

----

18. References
===========================================================

Foundational Papers
-----------------------------------------------------------

1. **Annable, T., Buscall, R., Ettelaie, R., & Whittlestone, D. (1993)**
   "The rheology of solutions of associating polymers: Comparison of experimental behavior with transient network theory"
   *Journal of Rheology*, 37(4), 695-726.
   DOI: 10.1122/1.550391

   *Original TNT loop-bridge model for HEUR solutions.*

2. **Tanaka, F., & Edwards, S. F. (1992)**
   "Viscoelastic properties of physically cross-linked networks: Transient network theory"
   *Macromolecules*, 25(5), 1516-1523.
   DOI: 10.1021/ma00031a024

   *Foundation of transient network theory.*

3. **Marrucci, G., Bhargava, S., & Cooper, S. L. (1993)**
   "Models of shear-thickening behavior in physically crosslinked networks"
   *Macromolecules*, 26(24), 6483-6488.
   DOI: 10.1021/ma00076a027

   *Theory of flow-induced association in telechelic polymers.*

Force-Dependent Kinetics
-----------------------------------------------------------

4. **Bell, G. I. (1978)**
   "Models for the specific adhesion of cells to cells"
   *Science*, 200(4342), 618-627.
   DOI: 10.1126/science.347575

   *Bell's theory of force-dependent bond dissociation.*

5. **Vaccaro, A., & Marrucci, G. (2000)**
   "A model for the nonlinear rheology of associating polymers"
   *Journal of Non-Newtonian Fluid Mechanics*, 92(2-3), 261-273.
   DOI: 10.1016/S0377-0257(00)00095-1

   *Stress-population coupling in telechelic networks.*

Experimental Validation
-----------------------------------------------------------

6. **Tripathi, A., Tam, K. C., & McKinley, G. H. (2006)**
   "Rheology and dynamics of associative polymers in shear and extension: Theory and experiments"
   *Macromolecules*, 39(5), 1981-1999.
   DOI: 10.1021/ma051614x

   *Comprehensive rheological study of HEUR solutions, validation of loop-bridge kinetics.*

7. **Pellens, L., Corrales, R. G., & Mewis, J. (2004)**
   "General nonlinear rheological behavior of associative polymers"
   *Journal of Rheology*, 48(2), 379-393.
   DOI: 10.1122/1.1645516

   *Shear thickening and thinning in associating polymers.*

Reviews and Extensions
-----------------------------------------------------------

8. **Rubinstein, M., & Semenov, A. N. (2001)**
   "Dynamics of entangled solutions of associating polymers"
   *Macromolecules*, 34(4), 1058-1068.
   DOI: 10.1021/ma0013049

   *Theory of dynamics in telechelic networks with entanglements.*

9. **Berret, J. F. (2006)**
   "Rheology of wormlike micelles: Equilibrium properties and shear banding transitions"
   *Molecular Gels*, 667-720.
   DOI: 10.1007/1-4020-3689-2_20

   *Related systems: wormlike micelles with loop-bridge-like kinetics.*

10. **Green, M. S., & Tobolsky, A. V. (1946)**
    "A new approach to the theory of relaxing polymeric media"
    *Journal of Chemical Physics*, 14(2), 80-92.
    DOI: 10.1063/1.1724109

    *Classic transient network theory (Green-Tobolsky model).*

Numerical Methods
-----------------------------------------------------------

11. **Fang, J., Kröger, M., & Öttinger, H. C. (2000)**
    "A thermodynamically admissible reptation model for fast flows of entangled polymers. II. Model predictions for shear and extensional flows"
    *Journal of Rheology*, 44(6), 1293-1317.
    DOI: 10.1122/1.1308522

    *Numerical methods for stiff differential equations in rheology.*

Related Material Systems
-----------------------------------------------------------

12. **Hourdet, D., L'alloret, F., & Audebert, R. (1994)**
    "Reversible thermothickening of aqueous polymer solutions"
    *Polymer*, 35(12), 2624-2630.
    DOI: 10.1016/0032-3861(94)90390-5

    *Temperature-dependent association in HEUR (thermal activation of* :math:`\tau_a`, :math:`\tau_b`).

13. **Chassenieux, C., Nicolai, T., & Benyahia, L. (2011)**
    "Rheology of associative polymer solutions"
    *Current Opinion in Colloid & Interface Science*, 16(1), 18-26.
    DOI: 10.1016/j.cocis.2010.07.007

    *Recent review of associating polymer rheology.*
