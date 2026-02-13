.. _tnt-knowledge-extraction:

===========================================================
TNT Knowledge Extraction Guide
===========================================================

This guide shows how to extract physical knowledge from fitted TNT (Transient Network Theory) model parameters. Learn how to map fitted parameters to physical properties, classify materials, apply scaling laws, and use diagnostic decision trees to select the right TNT variant.

.. contents:: Table of Contents
   :local:
   :depth: 3

--------
Overview
--------

What Knowledge Can Be Extracted
================================

TNT models provide access to fundamental network physics through fitted parameters:

**Structural Properties**

- Chain number density from plateau modulus
- Entanglement density from network connectivity
- Chain length from extensibility limits
- Network topology from bridge/loop fractions

**Kinetic Properties**

- Activation energies from temperature dependence
- Force sensitivity from shear-rate dependence
- Bond lifetime distributions from relaxation spectra
- Mechanochemical coupling strengths

**Material Classification**

- Network type: physical vs chemical crosslinks
- Polymer architecture: linear, telechelic, miktoarm
- Flow regime: dilute, semi-dilute, concentrated
- Transition identification: gel point, glass transition

**Predictive Capabilities**

- Nonlinear rheology from linear viscoelastic data
- Temperature-rate superposition
- Concentration scaling
- Processing window optimization

How to Use This Guide
======================

1. **Fit your TNT model** using :doc:`tnt_protocols`
2. **Extract parameter values** (point estimates or posteriors)
3. **Apply parameter-to-physics maps** (Section 2) to get physical properties
4. **Classify your material** using decision trees (Sections 3, 5)
5. **Validate** using scaling laws (Section 4)
6. **Compare models** if needed (Section 6)
7. **Report results** following Section 11

Cross-Model Comparison Strategy
================================

TNT family has 10 variants. Decision process:

1. Start with :class:`~rheojax.models.TNTSingleMode` (simplest)
2. Check residuals and physical reasonableness
3. If systematic deviations, add complexity:

   - Shear thinning → :class:`~rheojax.models.TNTBell` (:ref:`model-tnt-bell`)
   - Strain hardening → :class:`~rheojax.models.TNTFENE` (:ref:`model-tnt-fene-p`)
   - Multiple relaxations → :class:`~rheojax.models.TNTMultiSpecies` (:ref:`model-tnt-multi-species`)
   - Shear thickening → :class:`~rheojax.models.TNTLoopBridge` (:ref:`model-tnt-loop-bridge`)

4. Compare via WAIC/BIC (Bayesian) or AIC (NLSQ)
5. Use simplest model within 2 WAIC units of best

-------------------------
Parameter-to-Physics Map
-------------------------

This section provides explicit formulas to convert fitted TNT parameters into physical material properties.

Primary TNT Parameters
=======================

.. list-table:: Core TNT Parameter Interpretations
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Plateau Modulus
     - :math:`G`
     - Network stiffness, entropic elasticity
     - 10 Pa (dilute) to 100 kPa (gel)
   * - Breakage Time
     - :math:`\tau_b`
     - Bond lifetime at zero stress
     - :math:`10^{-6}` s (micelles) to :math:`10^3` s (reversible gels)
   * - Solvent Viscosity
     - :math:`\eta_s`
     - Matrix contribution to viscosity
     - 0.001 Pa·s (water) to 10 Pa·s (polymer melt)

Derived Physical Properties
============================

Chain Number Density
--------------------

From rubber elasticity theory:

.. math::

   n_{\text{chains}} = \frac{G}{k_B T}

where :math:`k_B = 1.380649 \times 10^{-23}` J/K is Boltzmann constant and :math:`T` is absolute temperature (Kelvin).

**Example**: :math:`G = 1000` Pa at :math:`T = 298` K gives :math:`n_{\text{chains}} = 2.43 \times 10^{23}` m\ :sup:`-3`.

**Molar Concentration**:

.. math::

   c_{\text{chains}} = \frac{n_{\text{chains}}}{N_A} = \frac{G}{k_B T N_A} = \frac{G}{RT}

where :math:`R = 8.314` J/(mol·K) and :math:`c_{\text{chains}}` is in mol/m\ :sup:`3`.

Activation Energy
-----------------

If :math:`\tau_b` measured at multiple temperatures, use Arrhenius:

.. math::

   \tau_b(T) = \tau_0 \exp\left(\frac{E_a}{k_B T}\right)

Solve for activation energy:

.. math::

   E_a = k_B T \ln\left(\frac{\tau_b}{\tau_0}\right)

where :math:`\tau_0 \sim 10^{-9}` s is molecular attempt time (phonon frequency).

**Example**: :math:`\tau_b = 0.1` s at :math:`T = 298` K gives :math:`E_a \approx 0.97` eV.

**Alternative**: Plot :math:`\ln(\tau_b)` vs :math:`1/T` to get slope :math:`E_a/k_B`.

Bell Model Extensions
=====================

*See also:* :ref:`model-tnt-bell` for full variant handbook.

Force Sensitivity Parameter
----------------------------

.. list-table:: Bell Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Bell Parameter
     - :math:`\nu`
     - Force sensitivity of bond breakage
     - 0.1 (weak) to 10 (strong)
   * - Characteristic Stress
     - :math:`\sigma_c = G`
     - Stress scale for shear thinning
     - Same as :math:`G`

**Barrier Distance**:

.. math::

   d_b = \frac{\nu k_B T}{F_{\text{char}}}

where :math:`F_{\text{char}} = \sigma_c / n_{\text{chains}}^{1/3}` is characteristic force per chain.

**Physical Interpretation**:

- :math:`\nu < 1`: bonds weakly force-sensitive (small barrier distance)
- :math:`\nu \sim 1`: typical hydrogen bond
- :math:`\nu > 5`: strongly force-sensitive (large conformational change)

FENE Model Extensions
======================

*See also:* :ref:`model-tnt-fene-p` for full variant handbook.

Chain Extensibility
-------------------

.. list-table:: FENE Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Max Extension
     - :math:`L_{\text{max}}`
     - Maximum chain stretch
     - 1.5 (slight) to 20 (highly extensible)

**Number of Kuhn Segments**:

.. math::

   N_K = L_{\text{max}}^2

**Contour Length**: For known Kuhn length :math:`b_K`:

.. math::

   L_c = N_K b_K = L_{\text{max}}^2 b_K

**Example**: :math:`L_{\text{max}} = 10` gives :math:`N_K = 100` segments. With :math:`b_K = 1` nm, :math:`L_c = 100` nm.

**Onset of Stiffening**: Strain at which FENE effects appear:

.. math::

   \gamma_{\text{onset}} \approx \frac{1}{L_{\text{max}}}

Non-Affine Model Extensions
============================

*See also:* :ref:`model-tnt-non-affine` for full variant handbook.

Entanglement Coupling
---------------------

.. list-table:: Non-Affine Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Non-Affine Parameter
     - :math:`\xi`
     - Degree of non-affine deformation
     - -0.5 (slip) to 0.5 (affine)

**Second Normal Stress Coefficient**:

.. math::

   \Psi_2 = -\frac{G \tau_b^2 \xi}{2}

**Interpretation**:

- :math:`\xi = 0`: affine (standard TNT, :math:`N_2 = 0`)
- :math:`\xi > 0`: chain slip suppressed (:math:`N_2 < 0`)
- :math:`\xi < 0`: enhanced slip (:math:`|N_2|` larger)

Stretch-Creation Model Extensions
==================================

*See also:* :ref:`model-tnt-stretch-creation` for full variant handbook.

Mechanochemical Coupling
-------------------------

.. list-table:: Stretch-Creation Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Coupling Strength
     - :math:`\kappa`
     - Stretch-induced bond creation rate
     - 0 (no coupling) to 5 (strong)

**Creation Enhancement**:

.. math::

   k_c(\lambda) = k_c^0 \exp(\kappa \lambda)

where :math:`\lambda` is chain stretch and :math:`k_c^0 = 1/\tau_c` is baseline creation rate.

**Physical Meaning**:

- :math:`\kappa = 0`: no mechanochemical coupling
- :math:`\kappa > 0`: strain stiffening via bond creation
- :math:`\kappa < 0`: strain softening (rare)

Loop-Bridge Model Extensions
=============================

*See also:* :ref:`model-tnt-loop-bridge` for full variant handbook.

Network Topology
----------------

.. list-table:: Loop-Bridge Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Association Time
     - :math:`\tau_a`
     - Time for chain end to find binding site
     - :math:`10^{-3}` to :math:`10^2` s
   * - Equilibrium Bridge Fraction
     - :math:`f_{B,\text{eq}}`
     - Fraction of chains forming bridges at rest
     - 0.1 (mostly loops) to 0.9 (mostly bridges)

**Network Connectivity**:

.. math::

   \nu_{\text{eff}} = f_{B,\text{eq}} \nu_{\text{total}}

where :math:`\nu_{\text{total}} = G/(k_B T)` is total chain density.

**Shear Thickening Onset**:

.. math::

   \dot{\gamma}_{\text{thick}} \sim \frac{1}{\tau_a}

Cates Model Extensions
======================

*See also:* :ref:`model-tnt-cates` for full variant handbook.

Living Polymer Parameters
-------------------------

.. list-table:: Cates Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Reptation Time
     - :math:`\tau_{\text{rep}}`
     - Time for chain to diffuse its length
     - :math:`10^{-2}` to :math:`10^2` s
   * - Breakage Time
     - :math:`\tau_{\text{break}}`
     - Time for chain scission
     - :math:`10^{-4}` to 1 s

**Effective Relaxation Time** (geometric mean):

.. math::

   \tau_d = \sqrt{\tau_{\text{rep}} \tau_{\text{break}}}

**Average Micelle Length**:

.. math::

   L_{\text{avg}} \sim \sqrt{\frac{\tau_{\text{rep}}}{\tau_{\text{break}}}}

**Scission Energy**: From temperature dependence of :math:`\tau_{\text{break}}`.

Sticky Rouse Extensions
========================

*See also:* :ref:`model-tnt-sticky-rouse` for full variant handbook.

Sticker Dynamics
----------------

.. list-table:: Sticky Rouse Parameter Interpretation
   :widths: 15 20 30 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Physical Meaning
     - Typical Range
   * - Rouse Times
     - :math:`\tau_{R,p}`
     - Rouse mode :math:`p` relaxation time
     - :math:`10^{-6}` to 0.1 s
   * - Sticker Time
     - :math:`\tau_s`
     - Sticker dissociation time
     - :math:`10^{-4}` to 10 s

**Effective Mode Relaxation**:

.. math::

   \tau_{p,\text{eff}} = \tau_{R,p} + \tau_s

**Number of Kuhn Segments** from Rouse time:

.. math::

   N \sim \left(\frac{\tau_R \eta_s}{k_B T}\right)^{1/2}

-----------------------
Material Classification
-----------------------

This section provides decision trees to identify material type from fitted TNT parameters.

Network Density Classification
===============================

Based on plateau modulus :math:`G`:

.. code-block:: text

   Plateau Modulus Decision Tree
   ==============================

   G < 100 Pa
   ├─> Dilute Network
   │   ├─> Likely: Simple associating polymer, transient gel
   │   ├─> Examples: Low-concentration HEUR, weakly crosslinked
   │   └─> Check: c < c* (overlap concentration)

   100 Pa ≤ G < 10,000 Pa
   ├─> Semi-Dilute to Concentrated
   │   ├─> Likely: HEUR, telechelic polymers, colloidal gels
   │   ├─> Examples: PEO-PPO-PEO, hydrophobically modified polymers
   │   └─> Check: c* < c < c** (entanglement concentration)

   G ≥ 10,000 Pa
   ├─> Dense Network or Gel
   │   ├─> Likely: Reversible elastomer, dense colloidal gel
   │   ├─> Examples: Vitrimers, dense microgels
   │   └─> Check: Gel point passed, G' > G'' at all frequencies

**Quantitative Criteria**:

- **Dilute**: :math:`c < c^* \sim 1/[N b^3]` (overlap)
- **Semi-dilute**: :math:`c^* < c < c^{**} \sim 1/N_e b^3` (entanglement)
- **Concentrated**: :math:`c > c^{**}`

where :math:`N` is degree of polymerization, :math:`b` is segment length, :math:`N_e` is entanglement length.

Relaxation Time Classification
===============================

Based on breakage time :math:`\tau_b`:

.. code-block:: text

   Breakage Time Decision Tree
   ============================

   τ_b < 0.001 s
   ├─> Fast Dynamics
   │   ├─> Likely: Wormlike micelles, weak hydrogen bonds
   │   ├─> Examples: CTAB/NaSal, PEO/tannic acid
   │   └─> Accessible frequency range: f > 1000 Hz (SAOS challenging)

   0.001 s ≤ τ_b < 1 s
   ├─> Intermediate Dynamics
   │   ├─> Likely: Typical telechelic, moderate H-bonds
   │   ├─> Examples: HEUR in water, supramolecular polymers
   │   └─> Accessible: Standard SAOS (0.01 - 100 Hz)

   1 s ≤ τ_b < 1000 s
   ├─> Slow Dynamics
   │   ├─> Likely: Reversible gels, strong physical crosslinks
   │   ├─> Examples: Vitrimers, ionomers
   │   └─> Accessible: Creep, stress relaxation preferred

   τ_b ≥ 1000 s
   ├─> Very Slow or Permanent
   │   ├─> Likely: Near-permanent network, glassy
   │   └─> Check: Is network truly reversible? May need chemical gel model

Shear Response Classification
==============================

Based on flow curve shape (from :ref:`Bell <model-tnt-bell>`, :ref:`FENE-P <model-tnt-fene-p>`, or :ref:`loop-bridge <model-tnt-loop-bridge>` fits):

.. code-block:: text

   Flow Curve Decision Tree
   =========================

   Shear Thinning (η decreases with γ̇)
   ├─> Bell Parameter ν > 1
   │   ├─> Likely: Force-sensitive bonds (H-bonds, metal-ligand)
   │   ├─> Power-law region: η ~ γ̇^(-α), α = ν/(ν+1)
   │   └─> Onset: γ̇ ~ 1/τ_b

   Shear Thickening (η increases with γ̇)
   ├─> Loop-Bridge f_B,eq increases
   │   ├─> Likely: Telechelic with free ends (loops → bridges)
   │   ├─> Onset: γ̇ ~ 1/τ_a
   │   └─> Check: Does thickening saturate or diverge?

   Strain Hardening (σ increases with γ at fixed γ̇)
   ├─> FENE L_max finite
   │   ├─> Likely: Extensible chains (not infinitely stretchable)
   │   ├─> Onset: γ ~ 1/L_max
   │   └─> Alternative: Stretch-creation κ > 0

Oscillatory Response Classification
====================================

Based on SAOS Cole-Cole plot (:math:`G''` vs :math:`G'`):

.. code-block:: text

   SAOS Decision Tree
   ==================

   Perfect Semicircle
   ├─> Cates Living Polymer
   │   ├─> Single effective relaxation time τ_d
   │   ├─> τ_rep and τ_break coupled
   │   └─> G_max'' = G/2 at ω = 1/τ_d

   Partial Semicircle (truncated at low ω)
   ├─> Multi-Species or Sticky Rouse
   │   ├─> Multiple relaxation times
   │   ├─> Fit spectrum of {G_i, τ_i}
   │   └─> Check: Is there a slow mode not fully relaxed?

   Skewed Arc
   ├─> Non-Affine (ξ ≠ 0)
   │   ├─> Entanglements affect loss modulus
   │   └─> Or: Distributed relaxation times (polydispersity)

   No Semicircle (G'' increases monotonically)
   ├─> Not Single-Mode TNT
   │   ├─> Try: Fractional models (KWW, Cole-Cole)
   │   └─> Or: Broad distribution (multi-mode)

.. tip::

   **Variant Handbooks:** For detailed physics, equations, protocol predictions, and failure modes of each variant, see:

   - :ref:`model-tnt-bell` (force-dependent breakage)
   - :ref:`model-tnt-fene-p` (finite extensibility)
   - :ref:`model-tnt-non-affine` (Gordon-Schowalter slip)
   - :ref:`model-tnt-stretch-creation` (enhanced reformation)
   - :ref:`model-tnt-loop-bridge` (two-species topology)
   - :ref:`model-tnt-sticky-rouse` (multi-mode sticker dynamics)
   - :ref:`model-tnt-cates` (living polymers)
   - :ref:`model-tnt-multi-species` (multiple bond types)

Polymer Architecture Inference
===============================

.. list-table:: Architecture from TNT Parameters
   :widths: 25 40 35
   :header-rows: 1

   * - Architecture
     - TNT Signature
     - Example Systems
   * - Linear Associating
     - Single-mode, moderate :math:`\tau_b`
     - PEO with end groups
   * - Telechelic
     - :ref:`Loop-bridge <model-tnt-loop-bridge>`, :math:`f_{B,\text{eq}} < 1`, shear thickening
     - HEUR, PEO-PPO-PEO
   * - Miktoarm Star
     - :ref:`Multi-species <model-tnt-multi-species>` (different arms)
     - ABC star copolymers
   * - Wormlike Micelle
     - :ref:`Cates <model-tnt-cates>`, :math:`\tau_d \ll \tau_{\text{rep}}`
     - CTAB/NaSal
   * - Reversible Gel
     - High :math:`G`, slow :math:`\tau_b`, :ref:`FENE <model-tnt-fene-p>` or :ref:`stretch-creation <model-tnt-stretch-creation>`
     - Vitrimers, ionomers
   * - Supramolecular
     - :ref:`Sticky Rouse <model-tnt-sticky-rouse>`, multiple :math:`\tau_s`
     - Multi-H-bond systems

------------
Scaling Laws
------------

This section lists physical scaling relationships to validate fitted parameters and make predictions.

Fundamental Scaling Relations
==============================

Rubber Elasticity
-----------------

Plateau modulus scales with chain density:

.. math::

   G \sim n_{\text{chains}} k_B T \sim c_{\text{polymer}} T

**Validation**: :math:`G/T` should be roughly constant across temperatures (for entropic networks).

**Concentration Scaling**:

- Dilute: :math:`G \sim c^{2.3}` (scaling theory)
- Semi-dilute: :math:`G \sim c^{2.0}` (mean-field)
- Concentrated: :math:`G \sim c^{2.0 - 2.3}`

Arrhenius Kinetics
------------------

Breakage time temperature dependence:

.. math::

   \tau_b(T) = \tau_0 \exp\left(\frac{E_a}{k_B T}\right)

**Validation**: Plot :math:`\ln(\tau_b)` vs :math:`1/T` to verify linearity and measure :math:`E_a`.

**WLF Alternative** (near :math:`T_g`):

.. math::

   \log\left(\frac{\tau_b(T)}{\tau_b(T_0)}\right) = \frac{-C_1 (T - T_0)}{C_2 + T - T_0}

Use for glass-forming systems.

Viscosity Relations
===================

Zero-Shear Viscosity
--------------------

.. math::

   \eta_0 = G \tau_b + \eta_s

**Validation**: Measure :math:`\eta_0` from flow curve plateau. Should match :math:`G \tau_b + \eta_s` from SAOS fit.

**Concentration Scaling**:

.. math::

   \eta_0 \sim c^{3.0 - 3.9}

in semi-dilute regime (varies by polymer type).

Relaxation Time Scaling
------------------------

For sticky Rouse:

.. math::

   \tau_R \sim N^2 \frac{\eta_s b^2}{k_B T}

where :math:`N` is number of segments, :math:`b` is segment size.

**Validation**: If :math:`N` known, check :math:`\tau_R` vs :math:`N^2` scaling.

Normal Stress Relations
========================

First Normal Stress Difference
-------------------------------

In weak shear (Weissenberg number :math:`\text{Wi} \ll 1`):

.. math::

   N_1 \approx 2 G \tau_b^2 \dot{\gamma}^2

**Validation**: Plot :math:`N_1 / \dot{\gamma}^2` vs :math:`\dot{\gamma}` at low rates. Should plateau at :math:`2 G \tau_b^2`.

**Normal Stress Coefficient**:

.. math::

   \Psi_1 = \frac{N_1}{\dot{\gamma}^2} = 2 G \tau_b^2

Second Normal Stress Difference
--------------------------------

For non-affine model:

.. math::

   N_2 = -G \tau_b^2 \xi \dot{\gamma}^2

**Ratio**:

.. math::

   \frac{N_2}{N_1} = -\frac{\xi}{2}

**Typical Values**: :math:`|N_2/N_1| \sim 0.1 - 0.3` for polymer melts.

Oscillatory Shear Relations
============================

Storage and Loss Moduli
------------------------

Single-mode TNT:

.. math::

   G'(\omega) = G \frac{(\omega \tau_b)^2}{1 + (\omega \tau_b)^2}, \quad
   G''(\omega) = G \frac{\omega \tau_b}{1 + (\omega \tau_b)^2}

**Crossover Frequency**: :math:`G' = G''` at

.. math::

   \omega_c = \frac{1}{\tau_b}

**Validation**: Measure :math:`\omega_c` from SAOS. Should match :math:`1/\tau_b` from fit.

**High-Frequency Limit**:

.. math::

   \lim_{\omega \to \infty} G'(\omega) = G

**Low-Frequency Limit**:

.. math::

   \lim_{\omega \to 0} G''(\omega) / \omega = \eta_0 = G \tau_b + \eta_s

Complex Viscosity
-----------------

.. math::

   |\eta^*(\omega)| = \frac{\sqrt{G'^2 + G''^2}}{\omega}

**Cox-Merz Rule** (empirical, often holds for TNT):

.. math::

   \eta(\dot{\gamma}) \approx |\eta^*(\omega)| \quad \text{at } \dot{\gamma} = \omega

**Validation**: Overlay flow curve :math:`\eta(\dot{\gamma})` with :math:`|\eta^*(\omega)|`. Deviations indicate Cox-Merz breakdown.

Cates Model Scaling
===================

*See also:* :ref:`model-tnt-cates` for the full Cates variant handbook.

Living Polymer Relations
------------------------

**Effective Relaxation Time**:

.. math::

   \tau_d = \sqrt{\tau_{\text{rep}} \tau_{\text{break}}}

**Plateau Modulus**: Standard rubber elasticity:

.. math::

   G = \frac{k_B T}{a^3} = c_{\text{polymer}} \frac{RT}{M_e}

where :math:`M_e` is entanglement molecular weight.

**Crossover**: When :math:`\tau_{\text{break}} \ll \tau_{\text{rep}}`:

.. math::

   \tau_d \ll \tau_{\text{rep}} \quad \Rightarrow \quad \text{Rouse-like (no entanglements)}

When :math:`\tau_{\text{break}} \gg \tau_{\text{rep}}`:

.. math::

   \tau_d \approx \tau_{\text{rep}} \quad \Rightarrow \quad \text{Reptation-like (long chains)}

Sticky Rouse Scaling
=====================

*See also:* :ref:`model-tnt-sticky-rouse` for the full Sticky Rouse variant handbook.

Effective Mode Times
--------------------

For mode :math:`p`:

.. math::

   \tau_{p,\text{eff}} = \tau_{R,p} + \tau_s

**Rouse Time Scaling**:

.. math::

   \tau_{R,p} = \frac{\tau_R}{p^2}, \quad \tau_R = \frac{N^2 b^2 \eta_s}{3 \pi^2 k_B T}

**Slowest Mode** (:math:`p = 1`):

.. math::

   \tau_1 = \tau_R + \tau_s

**Fastest Mode** (:math:`p \gg 1`):

.. math::

   \tau_{p,\text{eff}} \approx \tau_s \quad \text{(sticker-limited)}

Bell Model Scaling
===================

*See also:* :ref:`model-tnt-bell` for the full Bell variant handbook.

Shear Thinning Power Law
-------------------------

At intermediate shear rates :math:`1/\tau_b \ll \dot{\gamma} \ll \exp(\nu)/\tau_b`:

.. math::

   \eta(\dot{\gamma}) \sim \eta_0 \left(\frac{\dot{\gamma}}{\dot{\gamma}_0}\right)^{-\alpha}

where

.. math::

   \alpha = \frac{\nu}{\nu + 1}, \quad \dot{\gamma}_0 = \frac{1}{\tau_b}

**Validation**: Fit flow curve to power law, extract :math:`\alpha`, solve for :math:`\nu = \alpha/(1 - \alpha)`.

**Limiting Cases**:

- :math:`\nu \to 0`: :math:`\alpha \to 0` (Newtonian)
- :math:`\nu \to \infty`: :math:`\alpha \to 1` (strong shear thinning)

FENE Model Scaling
==================

*See also:* :ref:`model-tnt-fene-p` for the full FENE-P variant handbook.

Strain Hardening Onset
-----------------------

Stress upturn at strain:

.. math::

   \gamma_{\text{onset}} \sim \frac{1}{L_{\text{max}}}

**Validation**: In startup shear, identify :math:`\gamma` where stress deviates from linear (constant-stress) regime.

**Stress Enhancement**:

.. math::

   \frac{\sigma(\gamma)}{\sigma_{\text{linear}}} \sim 1 + \left(\frac{\gamma}{1/L_{\text{max}}}\right)^2

for :math:`\gamma \lesssim 1/L_{\text{max}}`.

-------------------------
Diagnostic Decision Tree
-------------------------

This section provides a step-by-step flowchart to select the appropriate TNT variant based on experimental observations.

Master Decision Tree
====================

.. code-block:: text

   TNT Variant Selection Flowchart
   ================================

   START: You have rheological data (SAOS, flow curve, startup, or creep)

   ┌───────────────────────────────────────────────────────────────────┐
   │ Step 1: MEASURE SAOS (Small-Amplitude Oscillatory Shear)         │
   │ ----------------------------------------------------------------- │
   │ Plot Cole-Cole: G'' vs G'                                        │
   └───────────────────────────────────────────────────────────────────┘
              ↓
   ┌─────────────────────────────────────────────┐
   │ Q1: Is the Cole-Cole plot a semicircle?    │
   └─────────────────────────────────────────────┘
              ↓
        ┌─────┴─────┐
        │           │
       YES          NO
        │           │
        ↓           ↓
   ┌─────────┐  ┌──────────────────────────┐
   │ CATES   │  │ Continue to Step 2       │
   │ Model   │  │ (Not living polymer)     │
   └─────────┘  └──────────────────────────┘
   (Living         ↓
   polymers)
                ┌───────────────────────────────────────────────────────┐
                │ Step 2: MEASURE FLOW CURVE                            │
                │ ----------------------------------------------------- │
                │ η vs γ̇ (steady shear viscosity)                      │
                └───────────────────────────────────────────────────────┘
                   ↓
                ┌──────────────────────────────────────────┐
                │ Q2: Is there shear thickening?           │
                │ (η increases with γ̇ at high rates)      │
                └──────────────────────────────────────────┘
                   ↓
              ┌────┴────┐
              │         │
             YES        NO
              │         │
              ↓         ↓
         ┌─────────┐  ┌──────────────────────────┐
         │ LOOP-   │  │ Continue to Step 3       │
         │ BRIDGE  │  │ (No shear thickening)    │
         └─────────┘  └──────────────────────────┘
         (Telechelic)    ↓

                ┌───────────────────────────────────────────────────────┐
                │ Step 3: MEASURE STARTUP SHEAR (Multiple Rates)       │
                │ ----------------------------------------------------- │
                │ σ vs γ at different γ̇                                │
                └───────────────────────────────────────────────────────┘
                   ↓
                ┌──────────────────────────────────────────┐
                │ Q3: Is there strain stiffening?          │
                │ (σ increases faster than linear)         │
                └──────────────────────────────────────────┘
                   ↓
              ┌────┴─────┐
              │          │
             YES         NO
              │          │
              ↓          ↓
         ┌─────────┐  ┌──────────────────────────┐
         │ Q3a:    │  │ Continue to Step 4       │
         │ Rate-   │  │ (No strain stiffening)   │
         │ depend? │  └──────────────────────────┘
         └─────────┘     ↓
              ↓
        ┌─────┴─────┐
        │           │
       YES          NO
        │           │
        ↓           ↓
   ┌──────────┐ ┌────────┐
   │ STRETCH- │ │ FENE   │
   │ CREATION │ │ Model  │
   └──────────┘ └────────┘
   (Mechanochem) (Chain
                 extension)

                ┌───────────────────────────────────────────────────────┐
                │ Step 4: FIT BASIC TNT                                 │
                │ ----------------------------------------------------- │
                │ TNTSingleMode with constant breakage                  │
                └───────────────────────────────────────────────────────┘
                   ↓
                ┌──────────────────────────────────────────┐
                │ Q4: Does constant-breakage fit well?     │
                │ (Check R² > 0.95, random residuals)      │
                └──────────────────────────────────────────┘
                   ↓
              ┌────┴────┐
              │         │
             YES        NO
              │         │
              ↓         ↓
         ┌─────────┐  ┌──────────────────────────┐
         │ TANAKA- │  │ Q4a: What fails?         │
         │ EDWARDS │  │                          │
         │ (Basic) │  └──────────────────────────┘
         └─────────┘     ↓
         (Simplest    ┌────────────┬─────────────┐
          model)      │            │             │
                   SHEAR-THIN  MULTIPLE    SECOND-
                      ↓        RELAXATIONS NORMAL
                   ┌──────┐      ↓           ↓
                   │ BELL │  ┌────────┐  ┌─────────┐
                   │Model │  │ MULTI- │  │NON-AFFINE│
                   └──────┘  │SPECIES │  │ Model   │
                   (Force-   └────────┘  └─────────┘
                   sensitive) (Polydisperse) (Entangle)

.. tip::

   **Variant Handbooks:** For detailed physics, equations, protocol predictions, and failure modes of each variant, see:

   - :ref:`model-tnt-bell` (force-dependent breakage)
   - :ref:`model-tnt-fene-p` (finite extensibility)
   - :ref:`model-tnt-non-affine` (Gordon-Schowalter slip)
   - :ref:`model-tnt-stretch-creation` (enhanced reformation)
   - :ref:`model-tnt-loop-bridge` (two-species topology)
   - :ref:`model-tnt-sticky-rouse` (multi-mode sticker dynamics)
   - :ref:`model-tnt-cates` (living polymers)
   - :ref:`model-tnt-multi-species` (multiple bond types)

Protocol-Specific Diagnostic Signatures
=========================================

This section shows how each experimental protocol discriminates between TNT variants.
Use this table to plan which experiments will be most informative for identifying
your material's dominant physics.

.. list-table:: Protocol-Specific Diagnostic Signatures
   :widths: 10 11 11 11 11 11 12 12 11
   :header-rows: 1

   * - Protocol
     - :ref:`Bell <model-tnt-bell>`
     - :ref:`FENE-P <model-tnt-fene-p>`
     - :ref:`Non-Affine <model-tnt-non-affine>`
     - :ref:`Stretch-Creation <model-tnt-stretch-creation>`
     - :ref:`Loop-Bridge <model-tnt-loop-bridge>`
     - :ref:`Sticky Rouse <model-tnt-sticky-rouse>`
     - :ref:`Cates <model-tnt-cates>`
     - :ref:`Multi-Species <model-tnt-multi-species>`
   * - **SAOS**
     - Same Maxwell
     - Same Maxwell
     - Same Maxwell
     - Same Maxwell
     - Reduced :math:`G'` plateau
     - Multi-mode spectrum
     - Cole-Cole semicircle
     - Multi-peak :math:`G''`
   * - **Flow Curve**
     - Power-law thinning
     - Thinning + saturation
     - :math:`N_2 \neq 0`, same :math:`\sigma(\dot{\gamma})`
     - Shear thickening
     - :math:`f_B`-dependent thinning
     - Cox-Merz failure
     - Non-monotonic :math:`\to` banding
     - Staged thinning (staircase)
   * - **Startup**
     - Overshoot at :math:`\gamma \approx 1/\sqrt{\nu}`
     - Strain stiffening (super-linear)
     - Reduced :math:`N_1`, :math:`N_2 \neq 0`
     - Super-linear stress rise
     - Two timescales in :math:`\sigma(t)`
     - Multi-mode relaxation
     - Large overshoot :math:`\to` plateau
     - Sequential yielding
   * - **Relaxation**
     - Strain-dependent :math:`\tau_{\text{eff}}`
     - :math:`f`-dependent initial decay
     - Same as base TE
     - Slower decay (hardening)
     - Bridge recovery (stress :math:`\uparrow`)
     - Multi-exponential
     - Stretched exponential
     - Multi-exponential
   * - **Creep**
     - Eventual rupture at :math:`\sigma > \sigma_c`
     - Strain saturates at :math:`L_{\text{max}}`
     - Faster creep (slip)
     - Creep ringing / arrest
     - :math:`f_B` collapse :math:`\to` rupture
     - Multi-stage compliance
     - Viscosity bifurcation
     - Staged compliance
   * - **LAOS**
     - Strong odd harmonics
     - Box-like Lissajous
     - :math:`N_2` oscillates at :math:`2\omega`
     - Enhanced odd harmonics
     - Asymmetric Lissajous
     - Complex multi-harmonic
     - Stress plateau
     - Double-yielding

.. note::

   Diagnostic power is **highest** when comparing multiple protocols. A single protocol rarely
   uniquely identifies a variant. Plan experiments to cover at least two rows of this table
   for unambiguous variant selection.

Master Experimental Fingerprints
=================================

The following table maps observable experimental signatures to their diagnostic TNT variant
and the confirmatory test needed to validate the identification.

.. list-table:: Master Experimental Fingerprints
   :widths: 35 25 40
   :header-rows: 1

   * - Observable Signature
     - Diagnostic Variant
     - Confirmatory Test
   * - Power-law shear thinning with rate-dependent relaxation
     - :ref:`Bell <model-tnt-bell>`
     - Step strain: :math:`\tau_{\text{eff}}(\gamma_0)` decreases with :math:`\gamma_0`
   * - Strain stiffening at large extensions
     - :ref:`FENE-P <model-tnt-fene-p>`
     - Extensional viscosity bounded, Lissajous becomes box-like
   * - Non-zero :math:`N_2` (negative, proportional to :math:`\xi`)
     - :ref:`Non-Affine <model-tnt-non-affine>`
     - Lodge-Meissner violation in step strain
   * - Shear thickening (viscosity increases with rate)
     - :ref:`Stretch-Creation <model-tnt-stretch-creation>`
     - Startup shows super-linear stress growth
   * - Concentration-dependent viscosity with two timescales
     - :ref:`Loop-Bridge <model-tnt-loop-bridge>`
     - Bridge fraction recovery after flow cessation
   * - :math:`G'(\omega) \sim \omega^{1/2}` at intermediate frequencies
     - :ref:`Sticky Rouse <model-tnt-sticky-rouse>`
     - Multiple plateau regions in modulus
   * - Cole-Cole semicircle in :math:`G''` vs :math:`G'`
     - :ref:`Cates <model-tnt-cates>`
     - Non-monotonic flow curve :math:`\to` shear banding
   * - Multiple peaks in :math:`G''(\omega)`
     - :ref:`Multi-Species <model-tnt-multi-species>`
     - Sequential yielding in LAOS, staged flow curve
   * - Non-zero equilibrium stress after relaxation
     - :ref:`Multi-Species <model-tnt-multi-species>` (permanent + transient)
     - Creep saturation: :math:`\gamma \to \gamma_\infty`

---------------------------
Cross-Model Comparison
---------------------------

This section relates TNT parameters to other rheological model families.

TNT vs Maxwell Models
=====================

Relationship
------------

Single-mode TNT is identical to Maxwell in linear viscoelasticity:

.. math::

   G_{\text{TNT}} = G_{\text{Maxwell}}, \quad \tau_{\text{TNT}} = \tau_{\text{Maxwell}}

**Difference**: TNT provides physical interpretation (bond breakage) and extends to nonlinear (:ref:`Bell <model-tnt-bell>`, :ref:`FENE-P <model-tnt-fene-p>`).

**When to Use**:

- Use Maxwell for pure phenomenology
- Use TNT when network physics is relevant

TNT vs Giesekus Model
=====================

Shear Thinning Comparison
--------------------------

Both :ref:`TNT-Bell <model-tnt-bell>` and Giesekus produce shear thinning, but via different mechanisms:

.. list-table:: Shear Thinning Mechanisms
   :widths: 25 35 40
   :header-rows: 1

   * - Model
     - Mechanism
     - Parameter
   * - TNT-Bell
     - Force-accelerated bond breakage
     - :math:`\nu` (force sensitivity)
   * - Giesekus
     - Anisotropic drag (mobility tensor)
     - :math:`\alpha` (anisotropy)

**Power-Law Exponent**:

- TNT-Bell: :math:`\eta \sim \dot{\gamma}^{-\nu/(\nu+1)}`
- Giesekus: :math:`\eta \sim \dot{\gamma}^{-1/2}` (fixed exponent at high :math:`\alpha`)

TNT vs Thixotropic Models (DMT, Fluidity)
==========================================

Structural vs Kinetic Approaches
---------------------------------

.. list-table:: TNT vs Structure-Kinetics Models
   :widths: 25 35 40
   :header-rows: 1

   * - Aspect
     - TNT
     - DMT / Fluidity
   * - State Variable
     - Bond density (implicit)
     - Structure parameter :math:`\lambda` or fluidity :math:`f`
   * - Dynamics
     - Bond breakage rate :math:`k_b(\sigma)`
     - Structure evolution :math:`d\lambda/dt`
   * - Yield Stress
     - Emergent from network (high :math:`G`, slow :math:`\tau_b`)
     - Explicit :math:`\tau_y(\lambda)` closure
   * - Thixotropy
     - Via :ref:`multi-species <model-tnt-multi-species>` (different bond types)
     - Native (aging vs rejuvenation)
   * - Strength
     - Network physics, normal stresses
     - Explicit history-dependence

-----------------------------------------------
Cohort Method: Alternative Numerical Approach
-----------------------------------------------

The **cohort (history integral) method** provides an alternative to the standard ODE-based
approach for computing TNT model predictions. Instead of evolving the conformation tensor
:math:`\mathbf{S}(t)` forward in time via a differential equation, the cohort method tracks
individual **cohorts** of chains born at time :math:`t'` and sums their contributions to the
total stress.

Integral Formulation
=====================

Each cohort of chains created at time :math:`t'` contributes to the stress at time :math:`t`
proportionally to their birth rate :math:`\beta(t')`, their survival probability
:math:`\mathcal{S}(t,t')`, and the deformation they have accumulated since birth. The total
stress is the integral over all cohorts:

.. math::

   \boldsymbol{\sigma}(t) = \int_{-\infty}^{t} \beta(t') \, \mathcal{S}(t,t') \, G\left[\mathbf{B}(t,t') - \mathbf{I}\right] \, dt' + 2\eta_s \mathbf{D}(t)

where :math:`\beta(t')` is the birth rate, :math:`\mathcal{S}(t,t') = \exp\left(-\int_{t'}^{t} k_d(s) \, ds\right)` is the survival probability, and :math:`\mathbf{B}(t,t')` is the Finger strain tensor.

Each individual cohort contributes:

.. math::

   d\boldsymbol{\sigma}(t) = G \cdot \beta(t') \cdot \mathbf{S}(t,t') \cdot \exp\left(-\int_{t'}^{t} k_d(s) \, ds\right) \cdot dt'

where :math:`k_d(s)` is the destruction rate at time :math:`s`, which may depend on the local
stress or strain rate (as in the :ref:`Bell <model-tnt-bell>` or
:ref:`stretch-creation <model-tnt-stretch-creation>` variants).

Mathematical Equivalence
=========================

The ODE (conformation tensor) and integral (cohort) approaches are **mathematically equivalent** ---
they give identical predictions. The choice between them is purely one of numerical convenience
for a given problem. The ODE form evolves a single state variable forward in time, while the
integral form explicitly sums over the deformation history.

Advantages of the Cohort Method
================================

- **Complex deformation histories**: Naturally handles step-and-hold sequences, multi-rate
  protocols, and arbitrary time-dependent flows without special treatment at discontinuities.

- **Embarrassingly parallel on GPU**: Each cohort is independent --- the survival probability
  and strain accumulation for cohort :math:`t'` can be computed without reference to any other
  cohort. This maps directly to GPU parallelism.

- **Direct access to age distribution**: The cohort formulation gives immediate access to the
  age distribution of surviving chains, :math:`P(\text{age}) = \mathcal{S}(t, t-\text{age})`,
  which is a useful diagnostic for non-equilibrium states.

- **Numerical stability for large strain steps**: The integral form avoids the stiffness issues
  that can arise in the ODE form when the deformation gradient changes abruptly.

Disadvantages of the Cohort Method
====================================

- **Memory grows with time steps**: All cohort weights must be stored, leading to
  :math:`O(N_t)` memory where :math:`N_t` is the number of time steps. For long simulations,
  this can become prohibitive.

- **Less efficient for steady state**: At steady state, the ODE form reaches a fixed point
  directly, while the cohort form must still integrate over the full history (or truncate at
  a sufficiently large age).

- **Deformation gradient computation**: Each cohort pair :math:`(t, t')` requires computing
  the Finger tensor :math:`\mathbf{B}(t,t')`, which involves the full deformation gradient
  :math:`\mathbf{F}(t,t')` from :math:`t'` to :math:`t`.

When to Use Each Approach
==========================

- **ODE (conformation tensor)**: Preferred for steady-state calculations, simple flow histories
  (constant rate startup, steady shear), and when memory is limited.

- **Cohort (history integral)**: Preferred for complex multi-step protocols (e.g., pre-shear
  followed by relaxation followed by startup), when age-distribution information is needed,
  or when GPU parallelism can be exploited to offset the memory cost.

-------------------------------
Bayesian Knowledge Extraction
-------------------------------

Using posterior distributions to extract physical insights beyond point estimates.

Parameter Correlations
======================

Physical Coupling from Posteriors
----------------------------------

Posterior correlations reveal which parameters are physically coupled:

**Strong Correlation** (:math:`|\rho| > 0.7`):

- :math:`G` vs :math:`\tau_b`: High correlation → data only constrains :math:`\eta_0 = G\tau_b`
- Solution: Use prior knowledge or multi-protocol fitting (SAOS + flow curve)

**Moderate Correlation** (:math:`0.3 < |\rho| < 0.7`):

- :math:`\nu` vs :math:`\tau_b` (:ref:`Bell <model-tnt-bell>`): Force sensitivity affects apparent lifetime
- :math:`L_{\text{max}}` vs :math:`G` (:ref:`FENE-P <model-tnt-fene-p>`): Stiffening strain related to modulus

**No Correlation** (:math:`|\rho| < 0.3`):

- Parameters independently constrained
- High identifiability

Credible Intervals
------------------

Report 95% highest density intervals (HDI) for all parameters with uncertainties on derived quantities.

-----------------------
Temperature Dependence
-----------------------

How to extract activation energies and characterize temperature effects.

Arrhenius Analysis
==================

Multi-Temperature Fitting
--------------------------

1. Measure SAOS at temperatures :math:`T_1, T_2, \ldots, T_n`
2. Fit TNT model at each :math:`T_i` to extract :math:`\tau_b(T_i)`
3. Plot :math:`\ln(\tau_b)` vs :math:`1/T`
4. Fit linear regression: slope = :math:`E_a / k_B`

**Expected Ranges**:

- H-bonds: 0.1 - 0.5 eV
- Metal-ligand: 0.5 - 1.5 eV
- Covalent: > 2 eV

---------------------------
Concentration Dependence
---------------------------

How parameters scale with polymer concentration to validate network physics.

Modulus Scaling
===============

Power-Law Regimes
-----------------

**Dilute** (:math:`c < c^*`):

.. math::

   G \sim c^{2.3}

**Semi-Dilute** (:math:`c^* < c < c^{**}`):

.. math::

   G \sim c^{2.0 - 2.25}

**Concentrated** (:math:`c > c^{**}`):

.. math::

   G \sim c^{2.0}

-------------------
Practical Recipes
-------------------

Step-by-step guides for common analysis tasks.

Recipe 1: Determine if Material is a Living Polymer
====================================================

1. Perform SAOS: Frequency sweep from 0.01 to 100 rad/s
2. Plot Cole-Cole: :math:`G''` vs :math:`G'`
3. Check semicircle: Is :math:`G''_{\text{max}} \approx G/2`?
4. Fit :ref:`Cates model <model-tnt-cates>`
5. Validate: Calculate :math:`\tau_{\text{rep}}` and :math:`\tau_{\text{break}}`

Recipe 2: Measure Force Sensitivity of Crosslinks
==================================================

1. Startup shear: Measure :math:`\sigma` vs :math:`\gamma` at multiple rates
2. Fit :ref:`Bell model <model-tnt-bell>` to extract :math:`\nu`
3. Calculate barrier distance :math:`d_b`
4. Interpret force sensitivity level

---------------------
Reporting Guidelines
---------------------

What to include in publications and technical reports.

Essential Reporting Elements
=============================

**Best-Fit Parameters**: Report with uncertainties (standard errors or credible intervals)

**Model Selection Criteria**: WAIC, AIC, BIC, goodness-of-fit statistics

**Physical Interpretation Table**: Map parameters to physical quantities

**Comparison with Literature**: Similar systems, parameter ranges

**Data Quality Assessment**: Number of points, noise level, replicates

--------
See Also
--------

**TNT Model Handbooks**:

- :ref:`model-tnt-tanaka-edwards`
- :ref:`model-tnt-bell`
- :ref:`model-tnt-fene-p`
- :ref:`model-tnt-non-affine`
- :ref:`model-tnt-stretch-creation`
- :ref:`model-tnt-loop-bridge`
- :ref:`model-tnt-cates`
- :ref:`model-tnt-multi-species`
- :ref:`model-tnt-sticky-rouse`

**Protocols and Workflows**:

- :ref:`tnt-protocols` (:doc:`tnt_protocols`)

----------
References
----------

1. Tanaka, F., & Edwards, S. F. (1992). Viscoelastic properties of physically crosslinked networks. *Macromolecules*, 25, 1516-1523.
   DOI: 10.1021/ma00031a024
2. Bell, G. I. (1978). Models for the specific adhesion of cells to cells. *Science*, 200(4342), 618-627.
   DOI: 10.1126/science.347575
3. Warner, H. R. (1972). Kinetic theory and rheology of dilute suspensions of finitely extendible dumbbells. *Industrial & Engineering Chemistry Fundamentals*, 11(3), 379-387.
   DOI: 10.1021/i160043a017
4. Cates, M. E. (1987). Reptation of living polymers. *Macromolecules*, 20(9), 2289-2296.
   DOI: 10.1021/ma00175a038
5. Leibler, L., Rubinstein, M., & Colby, R. H. (1991). Dynamics of reversible networks. *Macromolecules*, 24(16), 4701-4707.
   DOI: 10.1021/ma00016a034
6. Rubinstein, M., & Colby, R. H. (2003). *Polymer Physics*. Oxford University Press.
   ISBN: 978-0198520597
7. Annable, T., et al. (1993). The rheology of solutions of associating polymers. *Journal of Rheology*, 37(4), 695-726.
   DOI: 10.1122/1.550391
8. Evans, E., & Ritchie, K. (1997). Dynamic strength of molecular adhesion bonds. *Biophysical Journal*, 72(4), 1541-1555.
   DOI: 10.1016/S0006-3495(97)78802-7
9. McLeish, T. C. B. (2002). Tube theory of entangled polymer dynamics. *Advances in Physics*, 51(6), 1379-1527.
   DOI: 10.1080/00018730210153216
