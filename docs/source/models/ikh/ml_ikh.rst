Multi-Lambda Isotropic-Kinematic Hardening (ML-IKH)
=====================================================

.. admonition:: Quick Reference
   :class: hint

   **Use when:** Multi-timescale thixotropy, stretched-exponential relaxation, distributed structure kinetics

   **Parameters:** 7N+1 (per_mode) or 6+3N (weighted_sum)

   **Key feature:** Distributed thixotropic timescales for complex recovery

   **yield_mode:** ``'per_mode'`` (default) or ``'weighted_sum'``

   **Materials:** Complex thixotropic fluids with multi-scale structural dynamics

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

   from rheojax.utils.prony import fit_prony_series

   # Fit N modes to recovery data
   tau_values, weights = fit_prony_series(
       t_recovery, lambda_recovery, n_modes=3
   )


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


Fitting Strategies
------------------

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

1. Wei, Y., Solomon, M.J., & Larson, R.G. (2018). "A multimode structural kinetics
   constitutive equation for the transient rheology of thixotropic elasto‐viscoplastic
   fluids." *J. Rheol.*, 62(1), 321-342. DOI: 10.1122/1.4996752

2. Dimitriou, C.J. & McKinley, G.H. (2014). "A comprehensive constitutive law for
   waxy crude oil." *Soft Matter*, 10(35), 6619-6644.

3. Geri, M., Venkatesan, R., Sambath, K., & McKinley, G.H. (2017). "Thermokinematic
   memory and the thixotropic elasto‐viscoplasticity of waxy crude oils."
   *J. Rheol.*, 61(3), 427-454. DOI: 10.1122/1.4978259

4. Fielding, S.M. et al. (2009). "Aging and rheology in soft materials."
   *J. Rheol.*, 53(1), 39-64.

5. Mewis, J. & Wagner, N.J. (2009). "Thixotropy."
   *Adv. Colloid Interface Sci.*, 147-148, 214-227.
