Multi-Lambda Isotropic-Kinematic Hardening (ML-IKH)
=====================================================

Quick Reference
---------------

- **Use when:** Multi-timescale thixotropy, stretched-exponential relaxation, distributed structure kinetics

- **Parameters:** 7N+1 (per_mode) or 6+3N (weighted_sum)

- **Key equation:** :math:`\sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i` (weighted-sum) or :math:`\sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}` (per-mode)

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

- **Material examples:** Complex thixotropic fluids with multi-scale structural dynamics

.. currentmodule:: rheojax.models.ikh.ml_ikh

.. autoclass:: MLIKH
   :members:
   :show-inheritance:


Overview
--------

The **ML-IKH** (Multi-Lambda IKH) model extends the :doc:`mikh` model to N parallel
modes, capturing materials with distributed thixotropic timescales. This is analogous
to the relationship between a single Maxwell element and a Generalized Maxwell Model.

Physical motivation:

- Many thixotropic materials exhibit **stretched-exponential recovery** that cannot
  be captured by a single timescale
- Structural elements (particles, droplets, polymer chains) may have **hierarchical
  organization** with different restructuring rates
- Multi-mode models provide better fit with fewer total parameters than increasing
  complexity of single-mode equations

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - N
     - —
     - Number of structural modes
   * - λᵢ
     - —
     - Structural parameter for mode i (0 = destructured, 1 = structured)
   * - τ_thix,i
     - s
     - Rebuilding timescale for mode i
   * - Γᵢ
     - —
     - Breakdown coefficient for mode i
   * - wᵢ
     - —
     - Weight of mode i (weighted_sum formulation only)
   * - σᵢ
     - Pa
     - Stress contribution from mode i (per_mode formulation only)
   * - αᵢ
     - Pa
     - Backstress for mode i (per_mode formulation only)
   * - γ̇ᵖᵢ
     - 1/s
     - Plastic strain rate for mode i (per_mode formulation only)


Theoretical Background
----------------------

The Need for Multiple Structural Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-timescale thixotropic models often fail to capture the complex structural
dynamics observed in real materials. Experimental evidence shows:

**1. Stretched-Exponential Recovery:**

When a thixotropic material is allowed to recover at rest after pre-shearing,
the yield stress often recovers not as a simple exponential:

.. math::

   \sigma_y(t) = \sigma_{y,\infty} - (\sigma_{y,\infty} - \sigma_{y,0}) e^{-t/\tau}

but rather as a stretched exponential (Kohlrausch-Williams-Watts form):

.. math::

   \sigma_y(t) = \sigma_{y,\infty} - (\sigma_{y,\infty} - \sigma_{y,0}) e^{-(t/\tau)^\beta}

where β < 1 indicates a distribution of timescales. The ML-IKH model captures this
behavior naturally through its N structural modes.


Stretched Exponential Decomposition (KWW)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Kohlrausch-Williams-Watts (KWW)** stretched exponential function:

.. math::

   \phi(t) = \exp\left[-(t/\tau_c)^\beta\right]

where :math:`\beta \in (0, 1]` is the stretch exponent, can be mathematically
decomposed into a sum of pure exponentials:

.. math::

   \exp\left[-(t/\tau_c)^\beta\right] = \int_0^\infty \rho(\tau) \, e^{-t/\tau} \, d\tau

where :math:`\rho(\tau)` is the continuous relaxation time distribution.

**Discrete Mode Approximation:**

For practical computation, this integral is discretized:

.. math::

   \phi(t) \approx \sum_{r=1}^{N} w_r \exp(-t/\tau_r)

The weights :math:`w_r` and timescales :math:`\tau_r` are chosen to minimize
approximation error over the experimental time window.

**Mode Selection Rule:**

A fundamental result from the theory of stretched exponentials is that the
number of modes N required for accurate representation scales as:

.. math::

   \boxed{N \sim \left(\frac{1}{\beta}\right)^2}

This provides practical guidance for model complexity:

.. list-table:: Mode Selection Based on Stretch Exponent
   :widths: 20 20 40 20
   :header-rows: 1

   * - β
     - Physical Behavior
     - Interpretation
     - N Required
   * - 1.0
     - Pure exponential
     - Single timescale
     - 1 (use MIKH)
   * - 0.7
     - Mild stretching
     - Narrow distribution
     - 2
   * - 0.5
     - Moderate stretching
     - Moderate distribution
     - 4
   * - 0.3
     - Strong stretching
     - Broad distribution
     - 9-11

**Determining β from Experimental Data:**

The stretch exponent β can be extracted from recovery experiments:

1. Pre-shear material to destructure (λ → 0)
2. Stop shearing and monitor yield stress recovery σ_y(t)
3. Fit: :math:`\ln[-\ln(\Delta\sigma_y(t)/\Delta\sigma_{y,max})] = \beta \ln(t/\tau_c)`
4. Slope gives β, intercept gives τ_c

A plot of the left-hand side vs :math:`\ln(t)` should be linear for KWW behavior.


**2. Hierarchical Microstructure:**

Complex fluids often have structure at multiple length scales:

- **Waxy crude oils**: Primary wax crystals, crystal clusters, space-spanning networks
- **Colloidal gels**: Primary particles, aggregates, aggregate networks
- **Polymer solutions**: Chain segments, entanglements, transient networks

Each structural level may have distinct kinetics, leading to multiple characteristic
timescales for buildup and breakdown.

**3. Wei, Solomon & Larson (2018) ML-IKH Framework:**

Wei et al. generalized the single-mode IKH model to N modes, demonstrating that:

- N = 2-3 modes typically capture experimental data across a wide range of
  shear rates and time scales
- Mode timescales should span the experimental frequency/time window
- Physical interpretation: different modes represent different structural
  "populations" with distinct kinetics

Mathematical Justification
~~~~~~~~~~~~~~~~~~~~~~~~~~

The multi-mode formulation is mathematically justified by viewing the structural
parameter as arising from a distribution of relaxation times. If structure recovery
is governed by a spectrum of timescales P(τ), then:

.. math::

   \lambda(t) = \int_0^\infty P(\tau) \lambda_\tau(t) \, d\tau

where λ_τ(t) is the contribution from structures with timescale τ.

For practical computation, this integral is discretized into N modes:

.. math::

   \lambda(t) \approx \sum_{i=1}^{N} w_i \lambda_i(t)

with weights wᵢ and timescales τᵢ chosen to span the relevant range.

Physical Foundations
--------------------

Distributed Structure Kinetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real thixotropic materials often have **multiple structural populations** with different kinetic timescales:

- **Waxy crude oils**: Primary wax crystals (fast), crystal clusters (medium), space-spanning networks (slow)
- **Colloidal gels**: Primary bonds (fast), aggregates (medium), network structure (slow)
- **Emulsions**: Droplet-droplet contacts (fast), floc formation (medium), phase separation (slow)

Each population may build up and break down at different rates, leading to **stretched-exponential recovery**:

.. math::

   \lambda(t) \approx 1 - (1 - \lambda_0) \exp\left[-(t/\tau)^\beta\right]

where β < 1 indicates a distribution of timescales. The ML-IKH model captures this by superposing N exponential modes:

.. math::

   \lambda(t) = \sum_{i=1}^{N} w_i \lambda_i(t)


Timescale Distribution: Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distribution of recovery timescales in multi-mode models has concrete
physical origins in the hierarchical nature of soft material microstructure.

**Fast Modes (τ ~ 0.1–1 s):**

- **Physical mechanism**: Local bond reformation, nearest-neighbor particle rearrangement
- **Structural scale**: Individual particle contacts, primary bonds
- **Activation energy**: Low (thermal fluctuations sufficient)
- **Experimental signature**: Rapid initial stress recovery after flow cessation

**Intermediate Modes (τ ~ 1–10 s):**

- **Physical mechanism**: Cluster reorganization, aggregate restructuring
- **Structural scale**: Multi-particle aggregates (10–100 particles)
- **Activation energy**: Moderate (cooperative rearrangements)
- **Experimental signature**: "Shoulder" in recovery curves, non-exponential character

**Slow Modes (τ ~ 10–1000 s):**

- **Physical mechanism**: Network-scale rearrangement, large-scale healing
- **Structural scale**: Percolating network, sample-spanning structure
- **Activation energy**: High (requires coordinated motion of many particles)
- **Experimental signature**: Long-time logarithmic aging, incomplete recovery

**Connection to Aging Dynamics:**

The slowest modes often exhibit power-law rather than exponential kinetics:

.. math::

   \lambda_{slow}(t) \sim t^\mu \quad \text{for} \quad t \gg \tau_{max}

This suggests connections to glassy dynamics and soft glassy rheology
(see :doc:`/models/sgr/index`).


Analogy to Prony Series
~~~~~~~~~~~~~~~~~~~~~~~~

The multi-mode structure is directly analogous to **Prony series** for viscoelasticity:

- **Viscoelasticity**: G(t) = Σ Gᵢ·exp(-t/τᵢ) (Generalized Maxwell)
- **Thixotropy**: λ(t) = Σ wᵢ·λᵢ(t) where λᵢ evolves with τ_thix,i (ML-IKH)

Both use **discrete mode approximations** to represent continuous relaxation spectra.

Governing Equations
-------------------

The ML-IKH model has two formulations (see Mathematical Formulation for complete equations):

**Per-Mode Formulation** (N independent yield surfaces):

- Each mode i has state (σᵢ, αᵢ, λᵢ)
- Total stress: σ_total = Σ σᵢ + η_∞·γ̇
- Use when: Distinct mechanical populations (e.g., bimodal particle size)

**Weighted-Sum Formulation** (single yield surface):

- Single state (σ, α) with N structure parameters λᵢ
- Yield stress: σ_y = σ_y,0 + k₃·Σ wᵢλᵢ
- Use when: Single mechanical response with distributed recovery kinetics

Connection to Generalized Maxwell Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ML-IKH model relates to the MIKH model exactly as the Generalized Maxwell Model
(Prony series) relates to the single Maxwell element:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Property
     - Single Mode
     - Multi-Mode
   * - Viscoelasticity
     - Maxwell element
     - Generalized Maxwell (Prony series)
   * - Thixotropy
     - MIKH (single λ)
     - ML-IKH (N λᵢ)
   * - Recovery
     - Exponential
     - Stretched exponential / multi-exponential
   * - Parameters
     - 11
     - 7N+1 (per_mode) or 6+3N (weighted_sum)


Physical Interpretation
-----------------------

Per-Mode Yield Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the **per_mode** formulation, each mode i represents a distinct **structural population**
with its own mechanical properties and kinetics:

.. math::

   \sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}

This is a **parallel connection** of N independent IKH elements, similar to a
Generalized Maxwell Model:

::

   Mode 1: [Spring G₁] ─ [Dashpot] ─ [Yield σ_y,1(λ₁)] ─ [Hardening α₁]
   Mode 2: [Spring G₂] ─ [Dashpot] ─ [Yield σ_y,2(λ₂)] ─ [Hardening α₂]
   ...
   Mode N: [Spring Gₙ] ─ [Dashpot] ─ [Yield σ_y,N(λₙ)] ─ [Hardening αₙ]
   ─────────────────────────────────────────────────────────────────────
   + [Solvent viscosity η_∞]

**Physical scenarios for per_mode:**

1. **Bimodal particle size distribution**: Large and small particles with different
   aggregation kinetics
2. **Multiple gel networks**: e.g., polymer gel + colloidal gel in same suspension
3. **Hierarchical structure**: Primary bonds (fast kinetics) + secondary networks
   (slow kinetics)

Weighted-Sum Yield Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the **weighted_sum** formulation, all modes share a single mechanical response
but contribute to a common yield stress through their structural parameters:

.. math::

   \sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i

This represents **distributed kinetics** affecting a single yield mechanism:

::

   Single mechanical element: [Spring G] ─ [Dashpot] ─ [Yield σ_y(λ₁...λₙ)] ─ [Hardening α]

   With structure from multiple populations:
   λ₁ (fast): τ₁ ~ 0.1 s     ─┐
   λ₂ (medium): τ₂ ~ 1 s     ─┼─> weighted sum → σ_y(t)
   λ₃ (slow): τ₃ ~ 10 s      ─┘

**Physical scenarios for weighted_sum:**

1. **Single particle population with distributed kinetics**: Same particles, different
   local environments
2. **Stretched-exponential recovery**: Apparent stretched exponential arises from
   superposition of exponentials
3. **Memory effects**: Different timescales represent different "memory depths"
   in the material's thixotropic history


Two Yield Mode Options
----------------------

The ML-IKH model supports two formulations selected via ``yield_mode``:

Per-Mode Yield (Default)
~~~~~~~~~~~~~~~~~~~~~~~~

Each mode has an **independent yield surface**. Total stress is the sum of mode stresses.

- **Use when:**

- Material has distinct yielding events at different strains
- Different structural components have different mechanical properties
- Parallel mechanical connection is appropriate

**Governing equations:**

.. math::

   \sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}

Each mode follows independent MIKH equations.

- **Parameters:** 7 per mode + 1 global = 7N + 1 total

Weighted-Sum Yield
~~~~~~~~~~~~~~~~~~

**Single global yield surface** with structure contribution from all modes.

.. math::

   \sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i

- **Use when:**

- Material has single mechanical response but distributed recovery kinetics
- You want parsimonious model with fewer parameters
- Physical intuition suggests "average" structure controls yielding

- **Parameters:** 6 global + 3 per mode = 6 + 3N total


Mathematical Formulation
------------------------

Per-Mode Formulation
~~~~~~~~~~~~~~~~~~~~

State for each mode i: (σᵢ, αᵢ, λᵢ)

**Yield condition per mode:**

.. math::

   f_i = |\sigma_i - \alpha_i| - \sigma_{y,i}(\lambda_i) \leq 0

**Yield stress per mode:**

.. math::

   \sigma_{y,i} = \sigma_{y0,i} + \Delta\sigma_{y,i} \cdot \lambda_i

**Structure evolution per mode:**

.. math::

   \dot{\lambda}_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p_i|

**Backstress evolution per mode:**

.. math::

   \dot{\alpha}_i = C_i \dot{\gamma}^p_i - \gamma_{dyn,i} |\alpha_i|^{m-1} \alpha_i |\dot{\gamma}^p_i|

**Total stress:**

.. math::

   \sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}

Note that each mode has its own plastic strain rate γ̇ᵖᵢ determined by its own
yield condition. The total strain rate γ̇ is the same for all modes (parallel
connection assumption).

Weighted-Sum Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

Single state: (σ, α) with multiple λᵢ

**Single yield condition:**

.. math::

   f = |\sigma - \alpha| - \sigma_y \leq 0

**Weighted yield stress:**

.. math::

   \sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i

**Multiple structure evolution equations:**

.. math::

   \dot{\lambda}_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|

Each λᵢ evolves independently with its own (τᵢ, Γᵢ), but all share the same
plastic strain rate γ̇ᵖ.

**Backstress evolution:**

.. math::

   \dot{\alpha} = C \dot{\gamma}^p - \gamma_{dyn} |\alpha|^{m-1} \alpha |\dot{\gamma}^p|

**Total stress:**

.. math::

   \sigma_{total} = \sigma + \eta_{\infty} \dot{\gamma}

Steady-State Analysis
~~~~~~~~~~~~~~~~~~~~~

**Per-mode steady state:**

Each mode reaches its own steady state:

.. math::

   \lambda_{ss,i} = \frac{1/\tau_{thix,i}}{1/\tau_{thix,i} + \Gamma_i |\dot{\gamma}|}

The total stress is:

.. math::

   \sigma_{ss} = \sum_{i=1}^{N} \left( \sigma_{y0,i} + \Delta\sigma_{y,i} \lambda_{ss,i} \right) + \eta_{\infty} |\dot{\gamma}|

**Weighted-sum steady state:**

All λᵢ reach steady state with the global plastic strain rate:

.. math::

   \lambda_{ss,i} = \frac{1/\tau_{thix,i}}{1/\tau_{thix,i} + \Gamma_i |\dot{\gamma}^p|}

The yield stress becomes:

.. math::

   \sigma_{y,ss} = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_{ss,i}


Validity and Assumptions
------------------------

**Valid for:**

- **Multi-timescale thixotropy**: Materials with stretched-exponential or multi-exponential recovery
- **Hierarchical microstructure**: Multiple structural length/time scales
- **Complex thixotropic loops**: History-dependent behavior not captured by single-mode models

**Assumptions:**

- **Discrete mode approximation**: Continuous relaxation spectrum approximated by N modes
- **Independent mode kinetics**: No coupling between structural populations (each λᵢ evolves independently)
- **Affine deformation**: Homogeneous flow (no spatial gradients)
- **Isothermal**: No temperature dependence

**Not appropriate for:**

- **Simple thixotropy**: Use :doc:`mikh` instead (simpler, fewer parameters)
- **Shear banding**: Requires spatial extension
- **Temperature-dependent kinetics**: Would require additional modes or temperature-dependent parameters

What You Can Learn
------------------

From fitting Multi-Lambda IKH to experimental data, you can extract insights about timescale distributions, structural mode coupling, and complex thixotropic behavior.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**λ_i (Multiple Structure Parameters)**:
   Set of N dimensionless internal variables (0 ≤ λ_i ≤ 1) tracking distinct structural populations or timescales.
   *For graduate students*: Each λ_i evolves independently: dλ_i/dt = (1-λ_i)/τ_thix,i - Γ_i·λ_i|γ̇^p|. In weighted_sum mode, σ_y = σ_y,0 + k_3·Σw_i·λ_i represents distributed kinetics affecting single yield mechanism. In per_mode, each λ_i has independent yield surface. Recovers stretched-exponential via superposition: λ(t) ≈ Σw_i·exp(-t/τ_i) ~ exp[-(t/τ_c)^β].
   *For practitioners*: Fast mode (τ_1 ~ 0.1-1 s) = local bond reformation. Slow mode (τ_N ~ 100-1000 s) = network reorganization. Measure via multi-step recovery tests at different rest times.

**τ_thix,i (Recovery Timescale Spectrum)**:
   Set of N characteristic rebuilding times spanning 2-4 decades.
   *For graduate students*: Logarithmic spacing: τ_i = τ_min·(τ_max/τ_min)^((i-1)/(N-1)). Span τ_max/τ_min quantifies timescale dispersion. Broader span (>100) requires more modes (N ≥ 3-5). Connects to Kohlrausch-Williams-Watts stretch exponent β via N ~ (1/β)².
   *For practitioners*: Extract from recovery data fitting. If single-mode MIKH R² improvement > 10% with ML-IKH, complex thixotropy confirmed. Typical: N=2-3 sufficient for most soft materials.

**w_i (Mode Weights, weighted_sum only)**:
   Normalized weights (Σw_i = 1) quantifying relative importance of each structural timescale.
   *For graduate students*: In weighted_sum: σ_y = σ_y,0 + k_3·Σw_i·λ_i. Dominant mode: argmax(w_i·k_3). Weights relate to distribution of structural elements (e.g., particle size distribution in bimodal colloids).
   *For practitioners*: w_1 = 0.6, w_2 = 0.3, w_3 = 0.1 means fast mode dominates yield stress evolution. Adjust weights if recovery curves show multi-exponential character.

**Γ_i (Mode-Specific Breakdown Coefficients)**:
   Efficiency of shear-induced destructuring for each structural population.
   *For graduate students*: Controls mode-specific shear-thinning: λ_ss,i = 1/(1 + Γ_i·τ_thix,i|γ̇|). Different Γ_i values enable modeling materials where some structure (weak bonds) breaks easily while other structure (strong crosslinks) persists.
   *For practitioners*: Fit from flow curve multi-regime behavior. Example: Γ_1 >> Γ_2 means fast mode breaks down at low shear, slow mode only at high shear.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Multi-Lambda IKH Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - N = 2, τ_max/τ_min < 10
     - Simple bimodal thixotropy
     - Bidisperse colloids, two-network gels
     - Moderate complexity, 2-exponential recovery
   * - N = 3-5, τ_max/τ_min = 10-100
     - Complex thixotropy
     - Waxy crude oils, cement pastes, dense emulsions
     - Stretched-exponential recovery, history-dependent
   * - N > 5, τ_max/τ_min > 100
     - Extreme timescale dispersion
     - Aging soft glasses, hierarchical gels
     - Requires long-time measurements, non-Fickian
   * - Per-mode: G_1 << G_2
     - Bimodal elasticity
     - Filled polymers, composite gels
     - Distinct mechanical populations
   * - Weighted-sum: w_fast > 0.7
     - Fast-mode dominated
     - Quickly recovering suspensions
     - Rapid restructuring after flow

Timescale Distribution Strategies
---------------------------------

Logarithmic Distribution
~~~~~~~~~~~~~~~~~~~~~~~~

For general-purpose fitting, distribute timescales logarithmically:

.. math::

   \tau_i = \tau_{min} \cdot \left( \frac{\tau_{max}}{\tau_{min}} \right)^{(i-1)/(N-1)}

For N = 3 with τ_min = 0.1 s and τ_max = 100 s:

- τ₁ = 0.1 s (fast mode)
- τ₂ = 3.16 s (intermediate mode)
- τ₃ = 100 s (slow mode)

Experimentally-Motivated Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose timescales based on observed relaxation data:

1. Fit stretched exponential to recovery data
2. Extract characteristic time τ_c and stretch exponent β
3. Use N modes spanning τ_c · 10^(±2/β) approximately

Prony-Series Approach
~~~~~~~~~~~~~~~~~~~~~

Use automatic determination as in Generalized Maxwell fitting:

.. code-block:: python

   import numpy as np

   # Example: fit N modes to recovery data
   # Generate synthetic recovery data
   t_recovery = np.linspace(0, 100, 200)
   lambda_recovery = 1.0 - 0.8 * np.exp(-t_recovery / 10.0) - 0.2 * np.exp(-t_recovery / 50.0)

   # For automatic mode selection, use logarithmic distribution
   n_modes = 3
   tau_min, tau_max = 1.0, 100.0
   tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), n_modes)
   weights = np.ones(n_modes) / n_modes  # Equal weights as initial guess


Industrial Applications
-----------------------

The ML-IKH model is designed for materials with **multi-timescale thixotropy**
that single-mode models cannot capture. This section provides guidance for
industrial materials exhibiting stretched-exponential recovery or hierarchical
microstructure.

Complex Waxy Crude Oils
~~~~~~~~~~~~~~~~~~~~~~~

Waxy crude oils with broad wax crystal size distributions exhibit stretched-exponential
recovery that requires multiple structural modes.

**When to use ML-IKH over MIKH:**

- Recovery experiments show β < 0.8 (stretched exponential fit)
- Yield stress recovery spans >2 decades of time
- Different temperature histories produce different recovery profiles

**Mode selection for waxy crudes:**

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Wax Content
     - Recommended N
     - Physical Interpretation
   * - Low (<5%)
     - 2
     - Primary crystals + weak network
   * - Medium (5-15%)
     - 3
     - Primary + secondary aggregates + network
   * - High (>15%)
     - 4-5
     - Full hierarchical structure

**Typical timescale distribution:**

.. code-block:: python

   from rheojax.models import MLIKH

   # High-wax crude with hierarchical structure
   model = MLIKH(n_modes=4, yield_mode='weighted_sum')

   # Timescales spanning crystal → network scales
   timescales = [1.0, 10.0, 100.0, 1000.0]  # seconds
   weights = [0.15, 0.25, 0.35, 0.25]       # Network-dominated

   for i, (tau, w) in enumerate(zip(timescales, weights), 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)
       model.parameters.set_value(f"w_{i}", w)

**Pipeline restart implications:**

Multi-mode recovery means restart pressure depends strongly on shutdown duration:

- Short shutdown (t < τ_1): Only fast modes recover, moderate restart pressure
- Long shutdown (t > τ_N): All modes recover, maximum restart pressure
- Intermediate: Non-linear pressure increase with rest time

Bidisperse Colloidal Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bidisperse (two particle size) colloidal suspensions naturally produce
two-timescale thixotropy from different aggregation kinetics.

**Per-mode formulation recommended:**

Each particle population has distinct mechanical properties:

.. code-block:: python

   # Bidisperse colloid: small + large particles
   model = MLIKH(n_modes=2, yield_mode='per_mode')

   # Small particles: fast kinetics, lower modulus
   model.parameters.set_value("G_1", 200.0)
   model.parameters.set_value("tau_thix_1", 0.5)
   model.parameters.set_value("sigma_y0_1", 5.0)

   # Large particles: slow kinetics, higher modulus
   model.parameters.set_value("G_2", 800.0)
   model.parameters.set_value("tau_thix_2", 50.0)
   model.parameters.set_value("sigma_y0_2", 15.0)

**Identifying bidisperse behavior:**

- Flow curve shows two distinct shear-thinning regimes
- Startup stress shows double overshoot or shoulder
- Recovery curve is clearly bi-exponential (not stretched)

Drilling Fluids with Hierarchical Clay Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Water-based drilling fluids contain clay platelets that organize at multiple
length scales: face-face contacts (fast), edge-face networks (slow).

**Typical parameters:**

.. list-table::
   :widths: 20 25 25 30
   :header-rows: 1

   * - Mode
     - τ_thix (s)
     - Physical Structure
     - Weight
   * - Fast
     - 0.1-1
     - Face-face contacts
     - 0.3-0.4
   * - Medium
     - 1-10
     - Edge-face bonds
     - 0.3-0.4
   * - Slow
     - 10-100
     - House-of-cards network
     - 0.2-0.3

**API rheology connection:**

The multi-mode structure explains why API rheology readings at different
times after mixing give different values:

- 10-second gel: Dominated by fast modes
- 10-minute gel: Includes slow mode contribution
- The ratio (10-min gel)/(10-sec gel) indicates timescale dispersion

Dense Emulsions and Foams
~~~~~~~~~~~~~~~~~~~~~~~~~

Concentrated emulsions exhibit multi-timescale thixotropy from droplet
rearrangements at different length scales.

**Weighted-sum formulation recommended:**

Single mechanical response (droplet deformation) with distributed recovery:

.. code-block:: python

   # Dense emulsion (φ > 0.7)
   model = MLIKH(n_modes=3, yield_mode='weighted_sum')

   # Single mechanical modulus (droplet elasticity)
   model.parameters.set_value("G", 500.0)
   model.parameters.set_value("sigma_y0", 30.0)
   model.parameters.set_value("k3", 50.0)

   # Distributed recovery from droplet rearrangements
   model.parameters.set_value("tau_thix_1", 0.1)   # Local contacts
   model.parameters.set_value("tau_thix_2", 1.0)   # Cluster rearrangement
   model.parameters.set_value("tau_thix_3", 10.0)  # Network healing

Mode Selection for Industrial Materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Practical guidelines for choosing N:**

1. **Start with N=2** and check if fit improves significantly with N=3
2. **Use the β rule**: If stretched exponential fit gives β, then N ~ (1/β)²
3. **Match experimental timescales**: Ensure τ_min < t_experiment,min and τ_max > t_experiment,max
4. **Check for overfitting**: AIC/BIC should decrease with added modes

**β (stretch exponent) → N mapping:**

.. list-table::
   :widths: 15 25 30 30
   :header-rows: 1

   * - β
     - Behavior
     - N Required
     - Example Materials
   * - 0.9-1.0
     - Near-exponential
     - 1 (use MIKH)
     - Simple gels
   * - 0.7-0.9
     - Mild stretching
     - 2
     - Most drilling fluids
   * - 0.5-0.7
     - Moderate stretching
     - 3-4
     - Waxy crudes, emulsions
   * - 0.3-0.5
     - Strong stretching
     - 5-9
     - Aging glasses, cements

**Data quality requirements:**

Multi-mode fitting requires high-quality recovery data:

- **Time range**: At least 2 decades spanning τ_min to τ_max
- **Data density**: 10+ points per decade of time
- **Noise level**: Signal-to-noise ratio >20 for reliable mode separation
- **Protocol**: Pre-shear to consistent initial state before recovery


Parameters
----------

Per-Mode Parameters (yield_mode='per_mode')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``G_i``
     - Mode i shear modulus [Pa]
   * - ``C_i``
     - Mode i kinematic hardening modulus [Pa]
   * - ``gamma_dyn_i``
     - Mode i dynamic recovery parameter [-]
   * - ``sigma_y0_i``
     - Mode i minimal yield stress [Pa]
   * - ``delta_sigma_y_i``
     - Mode i structural yield contribution [Pa]
   * - ``tau_thix_i``
     - Mode i rebuilding timescale [s]
   * - ``Gamma_i``
     - Mode i breakdown coefficient [-]
   * - ``eta_inf``
     - Global high-shear viscosity [Pa·s]

Weighted-Sum Parameters (yield_mode='weighted_sum')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Description
   * - ``G``
     - Global shear modulus [Pa]
   * - ``C``
     - Global kinematic hardening modulus [Pa]
   * - ``gamma_dyn``
     - Global dynamic recovery [-]
   * - ``m``
     - AF exponent [-]
   * - ``sigma_y0``
     - Base yield stress [Pa]
   * - ``k3``
     - Structure-yield coupling [Pa]
   * - ``tau_thix_i``
     - Mode i rebuilding timescale [s]
   * - ``Gamma_i``
     - Mode i breakdown coefficient [-]
   * - ``w_i``
     - Mode i structure weight [-]
   * - ``eta_inf``
     - Global high-shear viscosity [Pa·s]


Fitting Guidance
----------------

Choosing Number of Modes
~~~~~~~~~~~~~~~~~~~~~~~~

1. Start with N=2 and check if fit improves significantly with N=3
2. Use AIC/BIC for model selection
3. Typical: N=2-4 is sufficient for most materials

**Akaike Information Criterion (AIC):**

.. math::

   AIC = 2k - 2\ln(\hat{L})

where k is the number of parameters and L̂ is the maximum likelihood.
Choose the model with the lowest AIC.

**Rule of thumb:** Add a mode only if it reduces residual sum of squares by
more than 5-10%.

Initializing Timescales
~~~~~~~~~~~~~~~~~~~~~~~

Distribute τᵢ logarithmically across expected range:

.. code-block:: python

   # For N modes spanning 0.1s to 100s:
   tau_values = np.logspace(-1, 2, n_modes)
   for i, tau in enumerate(tau_values, 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)

**Best practice:** Make sure the timescale range encompasses:

- The shortest characteristic time in your data (e.g., fastest startup)
- The longest experimental time (e.g., recovery experiments)

Per-Mode vs Weighted-Sum Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Criterion
     - Per-Mode
     - Weighted-Sum
   * - Distinct yield events
     - ✓ Better
     - –
   * - Single yield stress
     - –
     - ✓ Better
   * - Parameter economy
     - More params (7N+1)
     - Fewer params (6+3N)
   * - Physical interpretation
     - Parallel elements
     - Distributed kinetics

Fitting Protocol
~~~~~~~~~~~~~~~~

**For per_mode:**

1. Fit each mode's parameters separately to data at different timescales
2. Use long-time recovery data for slow modes (large τᵢ)
3. Use fast startup data for fast modes (small τᵢ)
4. Global optimization to fine-tune

**For weighted_sum:**

1. Fit global parameters (G, C, σ_y0) from startup/flow curve
2. Fix mechanical parameters
3. Fit kinetic parameters (τᵢ, Γᵢ, wᵢ) from recovery data
4. Constrain Σwᵢ = 1 for physical interpretation


Parameter Estimation Methods
----------------------------

Multi-mode models present unique parameter estimation challenges due to
mode-mode correlations and potential overfitting. This section provides
advanced methods for reliable ML-IKH parameter estimation.

Mode Number Selection (AIC/BIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selecting the optimal number of modes N requires balancing fit quality
against model complexity.

**Information Criteria:**

.. math::

   AIC &= 2k - 2\ln(\hat{L}) \\
   BIC &= k\ln(n) - 2\ln(\hat{L})

where k is the number of parameters, n is the number of data points,
and L̂ is the maximum likelihood.

**Practical workflow:**

.. code-block:: python

   import numpy as np
   from rheojax.models import MLIKH

   def compute_aic_bic(model, X, y_data):
       """Compute AIC and BIC for fitted model."""
       y_pred = model.predict(X)
       n = len(y_data)
       k = model.parameters.n_free  # Number of free parameters

       # Residual sum of squares
       rss = np.sum((y_data - y_pred)**2)

       # Log-likelihood (assuming Gaussian errors)
       sigma2 = rss / n
       log_likelihood = -n/2 * (np.log(2*np.pi*sigma2) + 1)

       aic = 2*k - 2*log_likelihood
       bic = k*np.log(n) - 2*log_likelihood

       return aic, bic

   # Compare N=2, 3, 4 modes
   results = []
   for n_modes in [2, 3, 4]:
       model = MLIKH(n_modes=n_modes, yield_mode='weighted_sum')
       model.fit(X, y_data, test_mode='startup')
       aic, bic = compute_aic_bic(model, X, y_data)
       results.append({'n_modes': n_modes, 'AIC': aic, 'BIC': bic})

   # Select model with lowest BIC (more conservative than AIC)
   best_n = min(results, key=lambda x: x['BIC'])['n_modes']

**Decision rules:**

- **ΔAIC < 2**: Models essentially equivalent
- **ΔAIC = 2-10**: Some evidence for lower-AIC model
- **ΔAIC > 10**: Strong evidence for lower-AIC model
- **BIC preferred** when sample size is moderate (n > 40) for parsimony

Timescale Initialization from Recovery Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Good initial timescale estimates dramatically improve convergence.

**Method 1: Logarithmic derivative analysis**

The logarithmic derivative of recovery data reveals characteristic timescales:

.. code-block:: python

   import numpy as np

   def estimate_timescales_from_recovery(t, lambda_data, n_modes):
       """Estimate timescales from recovery curve shape."""
       # Compute logarithmic derivative
       d_log_lambda = np.gradient(np.log(1 - lambda_data + 1e-10), np.log(t + 1e-10))

       # Find peaks/shoulders in derivative (indicate timescales)
       from scipy.signal import find_peaks
       peaks, _ = find_peaks(-d_log_lambda, prominence=0.1)

       if len(peaks) >= n_modes:
           tau_estimates = t[peaks[:n_modes]]
       else:
           # Fall back to logarithmic distribution
           tau_estimates = np.logspace(
               np.log10(t[1]), np.log10(t[-1]), n_modes
           )

       return tau_estimates

**Method 2: Stretched exponential fit**

Extract β first, then distribute timescales:

.. code-block:: python

   from scipy.optimize import curve_fit

   def stretched_exp(t, tau_c, beta):
       return 1 - np.exp(-(t/tau_c)**beta)

   # Fit stretched exponential to recovery
   popt, _ = curve_fit(stretched_exp, t_recovery, lambda_recovery,
                       p0=[10.0, 0.7], bounds=([0.1, 0.1], [1000, 1.0]))
   tau_c, beta = popt

   # Distribute timescales around τ_c
   n_modes = max(2, int(np.ceil((1/beta)**2)))
   tau_range = tau_c * 10**(2/beta)  # Span factor
   tau_values = np.logspace(
       np.log10(tau_c / np.sqrt(tau_range)),
       np.log10(tau_c * np.sqrt(tau_range)),
       n_modes
   )

Per-Mode vs Weighted-Sum Fitting Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two ML-IKH formulations require different fitting approaches.

**Per-mode strategy (independent yield surfaces):**

Each mode can be fit semi-independently:

.. code-block:: python

   from rheojax.models import MLIKH

   model = MLIKH(n_modes=2, yield_mode='per_mode')

   # Stage 1: Fit fast mode to short-time data
   mask_fast = t_data < 1.0  # Short times
   model.parameters.freeze_except(['G_1', 'tau_thix_1', 'sigma_y0_1', 'Gamma_1'])
   model.fit(X_data[:, mask_fast], y_data[mask_fast], test_mode='startup')

   # Stage 2: Fit slow mode to long-time data
   mask_slow = t_data > 10.0  # Long times
   model.parameters.unfreeze_all()
   model.parameters.freeze_except(['G_2', 'tau_thix_2', 'sigma_y0_2', 'Gamma_2'])
   model.fit(X_data[:, mask_slow], y_data[mask_slow], test_mode='startup')

   # Stage 3: Global refinement with all parameters
   model.parameters.unfreeze_all()
   model.fit(X_data, y_data, test_mode='startup')

**Weighted-sum strategy (single yield surface):**

Fit mechanical parameters first, then kinetic:

.. code-block:: python

   model = MLIKH(n_modes=3, yield_mode='weighted_sum')

   # Stage 1: Mechanical parameters from flow curve
   model.parameters.freeze(['tau_thix_1', 'tau_thix_2', 'tau_thix_3',
                           'Gamma_1', 'Gamma_2', 'Gamma_3',
                           'w_1', 'w_2', 'w_3'])
   model.fit(gamma_dot, sigma_ss, test_mode='flow_curve')

   # Stage 2: Kinetic parameters from recovery
   model.parameters.unfreeze_all()
   model.parameters.freeze(['G', 'C', 'gamma_dyn', 'sigma_y0', 'k3'])
   model.fit(t_recovery, lambda_recovery, test_mode='relaxation')

   # Stage 3: Global refinement
   model.parameters.unfreeze_all()
   model.fit(X_combined, y_combined)

Regularization for Correlated Mode Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mode weights wᵢ are often correlated, especially when timescales overlap.

**Weight normalization constraint:**

Enforce Σwᵢ = 1 during optimization:

.. code-block:: python

   import jax.numpy as jnp

   def normalize_weights(w_raw):
       """Softmax normalization ensures sum=1, all positive."""
       return jnp.exp(w_raw) / jnp.sum(jnp.exp(w_raw))

   # Use log-weights as free parameters
   # w_i = softmax(log_w_i)

**Timescale separation constraint:**

Prevent modes from collapsing to same timescale:

.. code-block:: python

   def timescale_separation_penalty(tau_values, min_ratio=3.0):
       """Penalty for timescales that are too close."""
       tau_sorted = jnp.sort(tau_values)
       ratios = tau_sorted[1:] / tau_sorted[:-1]
       penalty = jnp.sum(jnp.maximum(0, min_ratio - ratios)**2)
       return penalty

**Bayesian regularization via priors:**

Use informative priors to regularize mode parameters:

.. code-block:: python

   import numpyro
   import numpyro.distributions as dist

   def ml_ikh_bayesian_model(X, y_obs, n_modes):
       # Log-timescales with ordering constraint
       log_tau_base = numpyro.sample('log_tau_base', dist.Normal(1.0, 1.0))
       log_tau_increments = numpyro.sample(
           'log_tau_increments',
           dist.HalfNormal(0.5).expand([n_modes - 1])
       )
       log_tau = jnp.cumsum(jnp.concatenate([
           jnp.array([log_tau_base]),
           log_tau_increments
       ]))
       tau_values = jnp.exp(log_tau)

       # Dirichlet prior for weights (encourages diversity)
       weights = numpyro.sample('weights',
                                dist.Dirichlet(jnp.ones(n_modes)))

       # ... rest of model

This parameterization ensures τ₁ < τ₂ < ... < τₙ automatically.

Bayesian Inference for Multi-Mode Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian inference provides uncertainty quantification for mode parameters:

.. code-block:: python

   from rheojax.models import MLIKH

   model = MLIKH(n_modes=3, yield_mode='weighted_sum')

   # Point estimate first (critical for MCMC initialization)
   model.fit(X, y, test_mode='startup')

   # Bayesian inference
   result = model.fit_bayesian(
       X, y,
       num_warmup=1500,      # More warmup for multi-modal posteriors
       num_samples=3000,     # More samples for mode weight uncertainty
       num_chains=4,
       seed=42
   )

   # Check mode-specific convergence
   for i in range(1, 4):
       print(f"Mode {i}:")
       print(f"  τ_thix_{i}: {result.posterior_samples[f'tau_thix_{i}'].mean():.2f} "
             f"± {result.posterior_samples[f'tau_thix_{i}'].std():.2f}")
       print(f"  w_{i}: {result.posterior_samples[f'w_{i}'].mean():.3f} "
             f"± {result.posterior_samples[f'w_{i}'].std():.3f}")

**Diagnosing mode identifiability:**

- High posterior correlation between wᵢ and wⱼ → modes may be redundant
- Wide posterior for τᵢ → data doesn't constrain this timescale
- Multimodal posterior → consider reducing N or using ordered parameterization


JAX-First Numerical Implementation
-----------------------------------

The ML-IKH model uses a JAX-accelerated ODE integration strategy for all protocols.
This section describes the internal state vector structure and numerical approach.

State Vector Structure
~~~~~~~~~~~~~~~~~~~~~~

For ML-IKH with N modes, the state vector is:

.. code-block:: text

   y = [σ, A, λ_1, λ_2, ..., λ_N]

   Dimension: 2 + N
   ─────────────────
   y[0] = σ       : deviatoric stress [Pa]
   y[1] = A       : backstress internal variable (α = C·A) [-]
   y[2:2+N] = λ_r : structure parameters for modes 1...N [-]

For the **per_mode** formulation with N independent yield surfaces:

.. code-block:: text

   y = [σ_1, σ_2, ..., σ_N, A_1, A_2, ..., A_N, λ_1, λ_2, ..., λ_N]

   Dimension: 3N
   ─────────────────
   y[0:N]     = σ_i : stress for each mode [Pa]
   y[N:2N]    = A_i : backstress variable for each mode [-]
   y[2N:3N]   = λ_i : structure parameter for each mode [-]

ODE System (Rate-Controlled)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The governing equations for the weighted-sum formulation:

.. code-block:: python

   def rhs_mlikh(t, y, gdot, params):
       """ML-IKH right-hand side for ODE integration.

       Args:
           t: Current time
           y: State vector [σ, A, λ_1, ..., λ_N]
           gdot: Applied shear rate γ̇(t)
           params: Model parameters (G, η, C, q, m, k3, w, k1, k2)

       Returns:
           dy/dt: Time derivatives of state vector
       """
       sigma = y[0]
       A = y[1]
       lam = y[2:]  # Shape: (N,)

       # Backstress and effective stress
       sigma_back = params.C * A
       sigma_eff = sigma - sigma_back

       # Weighted yield stress from all modes
       sigma_y = params.k3 * jnp.sum(params.w * lam)

       # Plastic flow rate (Perzyna regularization)
       overstress = jnp.maximum(jnp.abs(sigma_eff) - sigma_y, 0.0)
       gdot_p = (overstress / params.mu_p) * jnp.sign(sigma_eff)

       # Stress evolution (Maxwell element)
       dsigma = params.G * (gdot - gdot_p) - (params.G / params.eta) * sigma

       # Backstress evolution (Armstrong-Frederick)
       fA = (params.q * jnp.abs(A))**params.m * jnp.sign(A)
       dA = gdot_p - fA * jnp.abs(gdot_p)

       # Structure evolution (each mode independent)
       dlam = params.k1 * (1.0 - lam) - params.k2 * lam * jnp.abs(gdot_p)

       return jnp.concatenate([jnp.array([dsigma, dA]), dlam])

Integration Strategy
~~~~~~~~~~~~~~~~~~~~

The model uses **RK4** integration with ``jax.lax.scan`` for efficient compilation:

.. code-block:: python

   @jax.jit
   def simulate_rate_control(rhs, t, u_t, y0, params):
       """Integrate ML-IKH under rate control using scan.

       Args:
           rhs: Right-hand side function
           t: Time array
           u_t: Shear rate history γ̇(t)
           y0: Initial state [σ_0, A_0, λ_1,0, ..., λ_N,0]
           params: Model parameters

       Returns:
           y_hist: State history, shape (len(t), 2+N)
       """
       dt = t[1] - t[0]

       def step(carry, inputs):
           ti, ui = inputs
           y_next = rk4_step(rhs, ti, carry, dt, ui, params)
           return y_next, y_next

       _, y_hist = jax.lax.scan(step, y0, (t, u_t))
       return y_hist

**Key advantages of JAX implementation:**

1. **JIT compilation**: First call compiles, subsequent calls are fast
2. **Automatic differentiation**: Enables gradient-based fitting and Bayesian inference
3. **Vectorization via vmap**: Efficient batch processing over multiple shear rates
4. **GPU acceleration**: Seamless transfer to GPU for large-scale computations

Protocol-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Protocol
     - Implementation Notes
   * - Flow curve
     - For each γ̇, set ``u_t = γ̇ * ones_like(t)``, integrate to steady state
   * - Startup
     - Set ``u_t = γ̇_0 * ones_like(t)``, track full σ(t) for overshoot
   * - Relaxation
     - Initial ``σ_0 = G·γ_0`` from step strain, set ``u_t = 0``
   * - Creep
     - Use stress-controlled wrapper with feedback: ``γ̇_{n+1} = γ̇_n + κ(σ_0 - σ_n)``
   * - LAOS
     - Set ``u_t = γ_0·ω·cos(ω·t)``, extract harmonics from steady-state cycles


Usage
-----

The ML-IKH model is available via:

.. code-block:: python

   from rheojax.models import MLIKH

**Common workflows**:

1. **Recovery data fitting**: Determine N modes and timescales from rest-time recovery
2. **Flow curve + startup**: Fit mechanical and kinetic parameters jointly
3. **Model selection**: Compare N=2 vs N=3 via AIC/BIC
4. **Bayesian inference**: Quantify uncertainty in mode weights and timescales

**Integration with Pipeline**:

.. code-block:: python

   from rheojax.pipeline import BayesianPipeline

   # Multi-mode thixotropic analysis
   (BayesianPipeline()
    .load('recovery_data.csv', x_col='time', y_col='yield_stress')
    .fit_nlsq('ml_ikh', n_modes=3, yield_mode='weighted_sum')
    .fit_bayesian(num_samples=2000)
    .plot_pair()  # Check mode correlations
    .save('ml_ikh_3mode_results.hdf5'))

Usage Examples
--------------

Per-Mode Initialization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import MLIKH

   # 2-mode model with per-mode yield surfaces
   model = MLIKH(n_modes=2, yield_mode='per_mode')

   # Fast mode (short τ)
   model.parameters.set_value("G_1", 500.0)
   model.parameters.set_value("tau_thix_1", 0.5)

   # Slow mode (long τ)
   model.parameters.set_value("G_2", 500.0)
   model.parameters.set_value("tau_thix_2", 10.0)

Weighted-Sum Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 3-mode model with single global yield surface
   model = MLIKH(n_modes=3, yield_mode='weighted_sum')

   # Global mechanical parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)
   model.parameters.set_value("k3", 40.0)

   # Distributed timescales
   for i, tau in enumerate([0.1, 1.0, 10.0], 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)
       model.parameters.set_value(f"w_{i}", 1/3)  # Equal weights

Prediction
~~~~~~~~~~

.. code-block:: python

   import numpy as np

   t = np.linspace(0, 20, 200)
   gamma = 1.0 * t  # Startup shear
   X = np.stack([t, gamma])

   stress = model.predict(X)

Fitting
~~~~~~~

.. code-block:: python

   import numpy as np

   # Generate synthetic startup data
   t_data = np.linspace(0, 20, 100)
   gamma_data = 1.0 * t_data  # Constant shear rate
   X_data = np.stack([t_data, gamma_data])
   stress_data = 50.0 * (1.0 - np.exp(-t_data / 5.0)) + 10.0 * t_data

   # Fit to data
   model.fit(X_data, stress_data, max_iter=500)

   # Bayesian inference
   result = model.fit_bayesian(
       X_data, stress_data,
       num_warmup=500, num_samples=1000,
       test_mode="startup"
   )


Recovery Behavior Comparison
----------------------------

To illustrate the advantage of multi-mode models, consider structure recovery
at rest:

**Single mode (MIKH):**

.. math::

   \lambda(t) = 1 - (1 - \lambda_0) e^{-t/\tau}

Pure exponential recovery.

**Two modes (ML-IKH weighted_sum):**

.. math::

   \lambda_{avg}(t) = w_1 \lambda_1(t) + w_2 \lambda_2(t)

where:

.. math::

   \lambda_i(t) = 1 - (1 - \lambda_{0,i}) e^{-t/\tau_i}

This produces bi-exponential recovery, which can approximate stretched exponentials.

**N modes:**

The sum of N exponentials can approximate stretched exponential recovery for
0.5 ≤ β ≤ 1 with good accuracy using N = 3-5 modes.


Relation to MIKH
----------------

ML-IKH with N=1 modes is equivalent to MIKH for the per_mode formulation.
For weighted_sum, the mapping is:

- ``sigma_y0`` (ML-IKH) ↔ ``sigma_y0`` (MIKH)
- ``k3`` (ML-IKH) ↔ ``delta_sigma_y`` (MIKH)
- ``w_1`` = 1.0 implicitly


Limitations and Considerations
------------------------------

**Computational cost:** ML-IKH with N modes requires tracking 3N state variables
(σᵢ, αᵢ, λᵢ for per_mode) versus 3 for MIKH. Computational cost scales roughly
linearly with N due to JAX vmap optimization.

**Parameter identifiability:** With many modes, parameters may become poorly
identifiable. Use regularization or constrain weights/timescales based on
physical intuition.

**Physical interpretation:** While multi-mode models fit data well, direct
physical interpretation of individual modes requires caution. The modes
represent a mathematical decomposition that may not correspond to distinct
physical structures.


References
----------

.. [1] Wei, Y., Solomon, M. J., and Larson, R. G. "A multimode structural kinetics
   constitutive equation for the transient rheology of thixotropic elasto-viscoplastic
   fluids." *Journal of Rheology*, 62(1), 321-342 (2018).
   https://doi.org/10.1122/1.4996752

.. [2] Dimitriou, C. J. and McKinley, G. H. "A comprehensive constitutive law for
   waxy crude oil: a thixotropic yield stress fluid." *Soft Matter*, 10(35),
   6619-6644 (2014). https://doi.org/10.1039/c4sm00578c

.. [3] Geri, M., Venkatesan, R., Sambath, K., and McKinley, G. H. "Thermokinematic
   memory and the thixotropic elasto-viscoplasticity of waxy crude oils."
   *Journal of Rheology*, 61(3), 427-454 (2017). https://doi.org/10.1122/1.4978259

.. [4] Fielding, S. M., et al. "Aging and rheology in soft materials."
   *Journal of Rheology*, 53(1), 39-64 (2009). https://doi.org/10.1122/1.3018902

.. [5] Mewis, J. and Wagner, N. J. "Thixotropy." *Advances in Colloid and Interface
   Science*, 147-148, 214-227 (2009). https://doi.org/10.1016/j.cis.2008.09.005

.. [6] Kohlrausch, R. "Theorie des elektrischen Rückstandes in der Leidener Flasche."
   *Annalen der Physik*, 167(2), 179-214 (1854). https://doi.org/10.1002/andp.18541670203

.. [7] Williams, G. and Watts, D. C. "Non-symmetrical dielectric relaxation behaviour
   arising from a simple empirical decay function." *Transactions of the Faraday Society*,
   66, 80-85 (1970). https://doi.org/10.1039/tf9706600080

.. [8] de Souza Mendes, P. R. and Thompson, R. L. "A critical overview of elasto-viscoplastic
   thixotropic modeling." *Journal of Non-Newtonian Fluid Mechanics*, 187-188, 8-15 (2012).
   https://doi.org/10.1016/j.jnnfm.2012.08.006

.. [9] Dullaert, K. and Mewis, J. "A structural kinetics model for thixotropy."
   *Journal of Non-Newtonian Fluid Mechanics*, 139(1-2), 21-30 (2006).
   https://doi.org/10.1016/j.jnnfm.2006.06.002

.. [10] Larson, R. G. and Wei, Y. "A review of thixotropy and its rheological modeling."
   *Journal of Rheology*, 63(3), 477-501 (2019). https://doi.org/10.1122/1.5055031

See Also
--------

- :doc:`mikh` — Single-mode thixotropic IKH model (simpler baseline)
- :doc:`/models/dmt/dmt_local` — Alternative multi-timescale thixotropy approach
- :doc:`/models/generalized_maxwell` — Multi-mode viscoelasticity (Prony series)
- :doc:`/user_guide/03_advanced_topics/index` — Advanced multi-mode fitting strategies
