Multi-Lambda Isotropic-Kinematic Hardening (ML-IKH)
=====================================================

Quick Reference
---------------

**Use when:** Multi-timescale thixotropy, stretched-exponential relaxation, distributed structure kinetics

**Parameters:** 7N+1 (per_mode) or 6+3N (weighted_sum)

**Key equation:** :math:`\sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i` (weighted-sum) or :math:`\sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}` (per-mode)

**Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

**Material examples:** Complex thixotropic fluids with multi-scale structural dynamics

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

**Use when:**

- Material has distinct yielding events at different strains
- Different structural components have different mechanical properties
- Parallel mechanical connection is appropriate

**Governing equations:**

.. math::

   \sigma_{total} = \sum_{i=1}^{N} \sigma_i + \eta_{\infty} \dot{\gamma}

Each mode follows independent MIKH equations.

**Parameters:** 7 per mode + 1 global = 7N + 1 total

Weighted-Sum Yield
~~~~~~~~~~~~~~~~~~

**Single global yield surface** with structure contribution from all modes.

.. math::

   \sigma_y = \sigma_{y,0} + k_3 \sum_{i=1}^{N} w_i \lambda_i

**Use when:**

- Material has single mechanical response but distributed recovery kinetics
- You want parsimonious model with fewer parameters
- Physical intuition suggests "average" structure controls yielding

**Parameters:** 6 global + 3 per mode = 6 + 3N total


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
