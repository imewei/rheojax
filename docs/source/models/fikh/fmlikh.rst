Fractional Multi-Lambda IKH (FMLIKH)
=====================================

Quick Reference
---------------

- **Use when:** Hierarchical thixotropic materials with both distributed timescales AND power-law memory per mode

- **Parameters:** 6 + 5N (shared :math:`\alpha`) or 6 + 6N (per-mode :math:`\alpha`)

- **Key equation:** :math:`D_t^\alpha \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|` (N fractional structure equations)

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

- **Material examples:** Complex waxy crude oils, hierarchical colloidal gels, cement pastes with aging

.. currentmodule:: rheojax.models.fikh.fmlikh

.. autoclass:: FMLIKH
   :members:
   :show-inheritance:


Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`N`
     - —
     - Number of structural modes
   * - :math:`D^\alpha` or :math:`D^{\alpha_i}`
     - —
     - Caputo fractional derivative (shared or per-mode order)
   * - :math:`\alpha` or :math:`\alpha_i`
     - —
     - Fractional order for mode i (0 < :math:`\alpha` < 1)
   * - :math:`\lambda_i`
     - —
     - Structural parameter for mode i (0 = destructured, 1 = structured)
   * - :math:`\tau_{thix,i}`
     - s
     - Rebuilding timescale for mode i
   * - :math:`\Gamma_i`
     - —
     - Breakdown coefficient for mode i
   * - :math:`w_i`
     - —
     - Weight of mode i in total yield stress (:math:`\sum w_i = 1`)
   * - :math:`G_i`
     - Pa
     - Shear modulus for mode i
   * - :math:`\sigma`
     - Pa
     - Total deviatoric stress
   * - :math:`A`
     - —
     - Backstress internal variable


Overview
--------

The **FMLIKH** (Fractional Multi-Lambda IKH) model extends the :doc:`fikh` model
to N parallel structural modes, combining two complementary mechanisms for
capturing complex relaxation dynamics:

1. **Multi-mode structure**: N distinct structural populations with different
   recovery timescales (like :doc:`../ikh/ml_ikh`)
2. **Fractional kinetics**: Power-law memory within each mode via Caputo derivatives

This "double spectrum" architecture provides exceptional flexibility for materials
with hierarchical microstructure where:

- Different structural levels recover on different timescales (captured by :math:`\tau_thix,i`)
- Each level exhibits power-law relaxation (captured by :math:`\alpha or \alpha_i`)

**Physical motivation:**

- **Waxy crude oils**: Primary crystals (fast, :math:`\alpha_1` ≈ 0.7), crystal clusters (medium, :math:`\alpha_2` ≈ 0.5),
  space-spanning networks (slow, :math:`\alpha_3` ≈ 0.4)
- **Colloidal gels**: Particle-particle bonds, aggregate structure, network connectivity
- **Cement pastes**: C-S-H gel formation, ettringite crystals, portlandite network

**Shared vs Per-Mode Fractional Order:**

- ``shared_alpha=True`` (default): Single :math:`\alpha` applies to all modes (fewer parameters)
- ``shared_alpha=False``: Each mode has its own :math:`\alpha_i` (maximum flexibility)


Theoretical Background
----------------------

The Need for Multi-Mode Fractional Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-mode FIKH captures power-law memory but assumes a single structural population.
Real hierarchical materials often exhibit:

1. **Multiple distinct timescales** from different structural levels
2. **Different memory characteristics** at each level

FMLIKH addresses both by superposing N fractional modes:

.. math::

   \sigma_y = \sigma_{y,0} + \Delta\sigma_y \sum_{i=1}^{N} w_i \lambda_i

where each :math:`\lambda_i` evolves via fractional kinetics:

.. math::

   D_t^{\alpha_i} \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|

Stretched Exponential vs Fractional Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Integer-order multi-mode (ML-IKH)** approximates stretched exponential recovery
through superposition of exponentials:

.. math::

   \lambda(t) = \sum_i w_i \left(1 - e^{-t/\tau_i}\right) \approx 1 - e^{-(t/\tau_c)^\beta}

**Fractional multi-mode (FMLIKH)** provides an alternative with power-law tails
within each mode:

.. math::

   \lambda(t) = \sum_i w_i \left(1 - E_{\alpha_i}\left(-\left(\frac{t}{\tau_i}\right)^{\alpha_i}\right)\right)

**When to use which:**

- **ML-IKH**: Recovery clearly follows sum of exponentials (distinct time constants visible)
- **FMLIKH**: Recovery shows power-law tails that persist beyond any exponential fit
- **FMLIKH with shared** :math:`\alpha`: Hierarchical structure with similar memory at each level
- **FMLIKH with per-mode** :math:`\alpha`: Different structural levels have distinct memory characteristics

Connection to Mode-Coupling Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FMLIKH structure parallels developments in Mode-Coupling Theory (MCT) for
glass-forming systems:

- **MCT** :math:`\beta`\ **-relaxation**: Fast cage rattling (analogous to fast FMLIKH modes)
- **MCT** :math:`\alpha`\ **-relaxation**: Slow structural relaxation (analogous to slow FMLIKH modes)
- **Fractional kinetics**: Captures the stretched/power-law character of glass relaxation

For materials near glass transition, FMLIKH provides a phenomenological approach
that captures MCT-like behavior without the full microscopic theory.

Shared vs Per-Mode Fractional Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shared** :math:`\alpha` **(``shared_alpha=True``):**

All modes share a single fractional order:

.. math::

   D_t^\alpha \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p| \quad \forall i

*Physical interpretation*: The microstructure has a universal memory character
(e.g., all levels are governed by similar molecular/thermal fluctuations).

*Advantages*:

- Fewer parameters (N-1 fewer than per-mode)
- Easier parameter identification
- Appropriate when different structural levels share similar physics

**Per-mode** :math:`\alpha` **(``shared_alpha=False``):**

Each mode has its own fractional order :math:`\alpha_i`:

.. math::

   D_t^{\alpha_i} \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|

*Physical interpretation*: Different structural levels have different memory
characteristics (e.g., fast modes are more Markovian, slow modes are glassy).

*Advantages*:

- Maximum flexibility
- Can capture hierarchical systems with fundamentally different dynamics at each level
- Needed when :math:`\alpha` clearly varies with timescale


Physical Foundations
--------------------

Distributed Fractional Kinetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In FMLIKH, the total structural parameter is a weighted sum:

.. math::

   \lambda_{total} = \sum_{i=1}^{N} w_i \lambda_i

Each :math:`\lambda_i` represents a distinct **structural population** with:

- Its own recovery timescale :math:`\tau_thix,i`
- Its own breakdown coefficient :math:`\Gamma_i`
- Its own (or shared) fractional order :math:`\alpha_i`

**Physical examples:**

- **Fast mode** (:math:`\tau_1` ~ 0.1-1 s, :math:`\alpha_1` ~ 0.7-0.9): Local bond reformation, surface contacts
- **Intermediate mode** (:math:`\tau_2` ~ 1-10 s, :math:`\alpha_2` ~ 0.5-0.7): Aggregate restructuring
- **Slow mode** (:math:`\tau_3` ~ 10-1000 s, :math:`\alpha_3` ~ 0.3-0.5): Network-scale reorganization, aging

Hierarchical Microstructure Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many complex fluids have structure at multiple length scales:

::

   Length scale →
   ┌──────────────────────────────────────────────────┐
   │ Primary         Aggregates       Network         │
   │ particles    →  of particles  →  structure       │
   │ (λ_1, fast)      (λ_2, medium)     (λ_3, slow)      │
   └──────────────────────────────────────────────────┘

Each level contributes to mechanical properties differently:

- **Primary particles**: Fast kinetics, weak memory (:math:`\alpha` → 1)
- **Aggregates**: Intermediate kinetics, moderate memory
- **Network**: Slow kinetics, strong memory (:math:`\alpha` → 0)

The weighted sum yield stress reflects how each level contributes to
macroscopic yielding.

Mode Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~

**How many modes N?**

1. **Start with N=2** and check improvement with N=3
2. **Use time-domain data**: Count distinct recovery timescales
3. **Use frequency-domain data**: Count shoulders/features in Cole-Cole plot
4. :math:`\beta` **rule from ML-IKH**: N ~ (1/:math:`\beta`)\ :math:`^2 where \beta` is stretch exponent

**When to use shared vs per-mode** :math:`\alpha` **?**

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Observation
     - Recommendation
     - Rationale
   * - All modes show similar power-law tails
     - shared_alpha=True
     - Universal memory mechanism
   * - Fast modes recover exponentially, slow modes show power-law
     - shared_alpha=False
     - Different physics at each scale
   * - Cole-Cole shows uniform depression
     - shared_alpha=True
     - Single :math:`\alpha` characterizes spectrum
   * - Cole-Cole shows scale-dependent depression
     - shared_alpha=False
     - :math:`\alpha` varies with timescale


Mathematical Formulation
------------------------

State Vector Structure
~~~~~~~~~~~~~~~~~~~~~~

For FMLIKH with N modes:

.. code-block:: text

   y = [σ, A, λ_1, λ_2, ..., λ_N]

   Dimension: 2 + N
   ─────────────────
   y[0]     = σ     : deviatoric stress [Pa]
   y[1]     = A     : backstress internal variable [-]
   y[2:2+N] = λ_i   : structure parameters for modes 1...N [-]

Plus N **history buffers** for fractional derivatives:

.. code-block:: text

   λ_history_i = [λ_i^{n-N_h+1}, ..., λ_i^{n-1}, λ_i^n]  # Shape: (n_history, N)

Yield Stress Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

**Weighted-sum yield stress:**

.. math::

   \sigma_y = \sigma_{y,0} + \Delta\sigma_y \sum_{i=1}^{N} w_i \lambda_i

where :math:`\Sigmaw_i` = 1 (normalization).

**Structure-dependent modulus** (optional):

Some materials also exhibit structure-dependent elasticity:

.. math::

   G_{eff} = G_0 + \Delta G \sum_{i=1}^{N} w_i \lambda_i

Per-Mode Evolution Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each structural mode evolves via fractional kinetics:

.. math::

   D_t^{\alpha_i} \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|

The fractional derivative is computed via L1 scheme with mode-specific
history buffers.

Stress and Backstress
~~~~~~~~~~~~~~~~~~~~~

The stress and backstress follow standard FIKH equations:

**Stress evolution:**

.. math::

   \frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma

**Backstress evolution:**

.. math::

   \frac{dA}{dt} = \dot{\gamma}^p - q|A|^{m-1}A|\dot{\gamma}^p|

**Plastic flow rate:**

.. math::

   \dot{\gamma}^p = \frac{\langle |\sigma - C \cdot A| - \sigma_y(\{\lambda_i\}) \rangle}{\mu_p} \cdot \text{sign}(\sigma - C \cdot A)

Steady-State Analysis
~~~~~~~~~~~~~~~~~~~~~

At steady state (D\ :math:`^{\alpha \lambda}` = 0 for constant :math:`\lambda`):

.. math::

   \lambda_{ss,i} = \frac{1}{1 + \Gamma_i \tau_{thix,i} |\dot{\gamma}|}

The weighted yield stress becomes:

.. math::

   \sigma_{y,ss} = \sigma_{y,0} + \Delta\sigma_y \sum_{i=1}^{N} \frac{w_i}{1 + \Gamma_i \tau_{thix,i} |\dot{\gamma}|}

**Note**: Steady-state flow curve is independent of :math:`\alpha_i` (fractional effects
only appear in transients).


Governing Equations
-------------------

The complete FMLIKH system:

**Stress evolution:**

.. math::

   \frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma

**Backstress evolution:**

.. math::

   \frac{dA}{dt} = \dot{\gamma}^p - q|A|^{m-1}A|\dot{\gamma}^p|

**Fractional structure evolution (for each mode i):**

.. math::

   D_t^{\alpha_i} \lambda_i = \frac{1-\lambda_i}{\tau_{thix,i}} - \Gamma_i \lambda_i |\dot{\gamma}^p|

**Weighted yield stress:**

.. math::

   \sigma_y = \sigma_{y,0} + \Delta\sigma_y \sum_{i=1}^{N} w_i \lambda_i

**Plastic flow rate:**

.. math::

   \dot{\gamma}^p = \frac{\langle |\sigma - C \cdot A| - \sigma_y \rangle}{\mu_p} \cdot \text{sign}(\sigma - C \cdot A)


What You Can Learn
------------------

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Per-Mode Timescales** :math:`\tau_thix,i` **:**

- Span the characteristic recovery times for different structural levels
- Logarithmic distribution typical: :math:`\tau_1 < \tau_2 < ... < \tau_N`
- :math:`\tau_max/\tau_min` ratio indicates breadth of timescale distribution
- Compare to experimental timescales to ensure adequate coverage

**Mode Weights w_i:**

- Relative contribution of each structural level to yield stress
- High w_i for fast modes → rapid initial recovery dominates
- High w_i for slow modes → long-time aging dominates
- Relate to microstructural composition (e.g., particle size distribution)

**Fractional Orders** :math:`\alpha_i` **(per-mode):**

- :math:`\alpha_i` → 1: Mode i behaves like exponential (Markovian)
- :math:`\alpha_i` < 0.5: Mode i has strong memory (glassy)
- Typically: :math:`\alpha` decreases with increasing :math:`\tau` (slow modes are more glassy)

**Breakdown Coefficients** :math:`\Gamma_i` **:**

- Mode-specific shear sensitivity
- High :math:`\Gamma_i`: Structure i breaks easily under shear
- Low :math:`\Gamma_i`: Structure i is shear-resistant
- Critical shear rate for mode i: :math:`\dot{\gamma}_crit,i = 1/(\Gamma_i \cdot \tau_thix,i)`

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: FMLIKH Material Classification
   :header-rows: 1
   :widths: 25 20 25 30

   * - Parameter Pattern
     - Behavior
     - Typical Materials
     - Recommendations
   * - N=2, :math:`\alpha` shared ≈ 0.6
     - Simple hierarchical
     - Bidisperse colloids
     - Good starting point
   * - N=3, :math:`\alpha` shared ≈ 0.5
     - Complex hierarchical
     - Waxy crude oils
     - Standard FMLIKH
   * - N=3, :math:`\alpha_i` decreasing
     - Scale-dependent memory
     - Aging glasses, cements
     - Use per-mode :math:`\alpha`
   * - N=2, :math:`\alpha_1 \approx 0.9, \alpha_2` ≈ 0.4
     - Fast exponential + slow glassy
     - Soft glasses
     - Per-mode :math:`\alpha` critical


Industrial Applications
-----------------------

Complex Waxy Crude Oils
~~~~~~~~~~~~~~~~~~~~~~~

Waxy crude oils with broad crystal size distributions and hierarchical
structure benefit from FMLIKH's dual-spectrum approach:

**Mode assignment:**

.. code-block:: python

   from rheojax.models.fikh import FMLIKH

   # 3-mode model for complex waxy crude
   model = FMLIKH(n_modes=3, shared_alpha=True, include_thermal=True)

   # Mode 1: Primary crystal contacts (fast)
   model.parameters.set_value("tau_thix_1", 1.0)
   model.parameters.set_value("Gamma_1", 2.0)
   model.parameters.set_value("w_1", 0.2)

   # Mode 2: Crystal clusters (medium)
   model.parameters.set_value("tau_thix_2", 10.0)
   model.parameters.set_value("Gamma_2", 1.0)
   model.parameters.set_value("w_2", 0.3)

   # Mode 3: Space-spanning network (slow)
   model.parameters.set_value("tau_thix_3", 100.0)
   model.parameters.set_value("Gamma_3", 0.5)
   model.parameters.set_value("w_3", 0.5)

   # Shared fractional order
   model.parameters.set_value("alpha_structure", 0.55)

**Pipeline restart implications:**

- Mode 1 recovers quickly → initial startup feasible
- Mode 3 recovers slowly → full gelation takes hours
- Fractional kinetics → restart pressure grows as power-law of rest time

Hierarchical Colloidal Gels
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Colloidal gels with particles aggregating at multiple scales:

**Per-mode** :math:`\alpha` **recommended** when bond energies vary with scale:

.. code-block:: python

   # Per-mode α for scale-dependent memory
   model = FMLIKH(n_modes=3, shared_alpha=False)

   # Fast mode: weak van der Waals (nearly exponential)
   model.parameters.set_value("alpha_structure_1", 0.85)
   model.parameters.set_value("tau_thix_1", 0.5)

   # Medium mode: moderate attraction
   model.parameters.set_value("alpha_structure_2", 0.6)
   model.parameters.set_value("tau_thix_2", 5.0)

   # Slow mode: strong covalent/depletion (glassy)
   model.parameters.set_value("alpha_structure_3", 0.4)
   model.parameters.set_value("tau_thix_3", 50.0)

Cement and Concrete Pastes
~~~~~~~~~~~~~~~~~~~~~~~~~~

Cement hydration creates hierarchical structure with different aging dynamics:

- **C-S-H gel**: Fast formation, moderate memory
- **Ettringite**: Medium timescale, structure-dependent
- **Portlandite network**: Slow formation, strong aging (low :math:`\alpha`)

**Application**: Predicting workability loss during placement.


Parameters
----------

Global Parameters
~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``G``
     - 1000
     - (0.1, :math:`10^9`)
     - Pa
     - Global shear modulus
   * - ``eta``
     - :math:`10^6`
     - (10\ :math:`^{-3, 10^1^2}`)
     - Pa·s
     - Maxwell viscosity (:math:`\tau = \eta/G`)
   * - ``C``
     - 500
     - (0, :math:`10^9`)
     - Pa
     - Kinematic hardening modulus
   * - ``gamma_dyn``
     - 1.0
     - (0, :math:`10^4`)
     - —
     - Dynamic recovery parameter
   * - ``m``
     - 1.0
     - (0.5, 3)
     - —
     - AF recovery exponent
   * - ``sigma_y0``
     - 10
     - (0, :math:`10^9`)
     - Pa
     - Minimal yield stress (all modes destructured)
   * - ``delta_sigma_y``
     - 50
     - (0, :math:`10^9`)
     - Pa
     - Total structural yield contribution
   * - ``eta_inf``
     - 0.1
     - (0, :math:`10^9`)
     - Pa·s
     - High-shear (solvent) viscosity
   * - ``mu_p``
     - :math:`10 \times 10^{-3}`
     - (10\ :math:`^{-9, 10^3}`)
     - Pa·s
     - Plastic viscosity (Perzyna)

Shared Fractional Order (``shared_alpha=True``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``alpha_structure``
     - 0.5
     - (0.05, 0.99)
     - —
     - Shared fractional order for all modes

Per-Mode Parameters
~~~~~~~~~~~~~~~~~~~

For each mode i = 1, 2, ..., N:

.. list-table::
   :widths: 18 12 15 12 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``tau_thix_i``
     - varies
     - (10\ :math:`^{-6, 10^1^2}`)
     - s
     - Mode i rebuilding timescale
   * - ``Gamma_i``
     - 0.5
     - (0, :math:`10^4`)
     - —
     - Mode i breakdown coefficient
   * - ``w_i``
     - 1/N
     - (0, 1)
     - —
     - Mode i weight in yield stress
   * - ``alpha_structure_i``
     - 0.5
     - (0.05, 0.99)
     - —
     - Mode i fractional order (if ``shared_alpha=False``)

**Note**: Weights are internally normalized: :math:`\Sigmaw_i` = 1.


Fitting Guidance
----------------

Choosing Number of Modes
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with N=2** and check if fit improves significantly with N=3
2. **Use AIC/BIC** for model selection (same approach as ML-IKH)
3. **Match experimental timescales**: Ensure :math:`\tau_min` < t_exp,min and :math:`\tau_max` > t_exp,max
4. **Typical**: N=2-4 sufficient for most materials

**Rule of thumb for combined** :math:`\beta and \alpha` **effects:**

If stretched exponential fit gives :math:`\beta_eff` < 0.5, you may need either:

- Higher N with shared :math:`\alpha \approx \beta_eff`
- Lower N with per-mode :math:`\alpha_i < \beta_eff` for slow modes

Shared vs Per-Mode :math:`\alpha` Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Criterion
     - Recommendation
   * - Simple system, similar bonds at all scales
     - ``shared_alpha=True``
   * - Hierarchical with different bond physics
     - ``shared_alpha=False``
   * - Limited data, need parsimony
     - ``shared_alpha=True``
   * - Rich data, need flexibility
     - ``shared_alpha=False``
   * - Fast modes exponential, slow modes glassy
     - ``shared_alpha=False`` with :math:`\alpha_1 > \alpha_2 > ... > \alpha_N`

Initializing Parameters
~~~~~~~~~~~~~~~~~~~~~~~

**Timescale initialization:**

.. code-block:: python

   import numpy as np

   # Logarithmic distribution spanning experimental window
   n_modes = 3
   tau_min, tau_max = 0.1, 100.0
   tau_values = np.logspace(np.log10(tau_min), np.log10(tau_max), n_modes)

   for i, tau in enumerate(tau_values, 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)

**Weight initialization:**

.. code-block:: python

   # Equal weights as starting point
   for i in range(1, n_modes + 1):
       model.parameters.set_value(f"w_{i}", 1.0 / n_modes)

**Per-mode** :math:`\alpha` **initialization (if used):**

.. code-block:: python

   # Decreasing α with increasing timescale (typical pattern)
   alpha_values = [0.8, 0.6, 0.4]  # Fast → slow modes

   for i, alpha in enumerate(alpha_values, 1):
       model.parameters.set_value(f"alpha_structure_{i}", alpha)

Sequential Fitting Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stage 1: Flow curve (steady state)**

Fit global parameters (:math:`\sigma_y0, \Delta\sigma_y, \eta_inf`) and :math:`\Gamma_i \cdot \tau_thix,i` products:

.. code-block:: python

   model.parameters.freeze(['G', 'C', 'gamma_dyn', 'eta', 'mu_p'])
   if shared_alpha:
       model.parameters.freeze(['alpha_structure'])
   else:
       for i in range(1, n_modes + 1):
           model.parameters.freeze([f'alpha_structure_{i}'])

   model.fit(gamma_dot, sigma_ss, test_mode='flow_curve')

**Stage 2: Startup transients (** :math:`\alpha` **matters)**

Unfreeze elastic and fractional parameters:

.. code-block:: python

   model.parameters.unfreeze(['G', 'C', 'gamma_dyn'])
   if shared_alpha:
       model.parameters.unfreeze(['alpha_structure'])

   model.fit(t_startup, sigma_startup, test_mode='startup')

**Stage 3: Recovery data (mode separation)**

Use long-time recovery to separate mode contributions:

.. code-block:: python

   model.parameters.unfreeze_all()
   model.fit(t_recovery, lambda_recovery, test_mode='relaxation')

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Issue
     - Solution
   * - Modes collapse to same :math:`\tau`
     - Use logarithmic initialization; add separation constraint
   * - Weights go to 0 or 1
     - Check data spans all timescales; consider reducing N
   * - Per-mode :math:`\alpha` all similar
     - Switch to ``shared_alpha=True``
   * - Slow convergence
     - Fix :math:`\alpha` values first; fit kinetic params only
   * - Memory overflow
     - Reduce n_history; use ``shared_alpha=True``


Parameter Estimation Methods
----------------------------

Mode Number Selection (AIC/BIC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same approach as ML-IKH:

.. code-block:: python

   def compute_aic_bic(model, X, y_data):
       """Compute AIC and BIC for fitted model."""
       y_pred = model.predict(X)
       n = len(y_data)
       k = model.parameters.n_free

       rss = np.sum((y_data - y_pred)**2)
       sigma2 = rss / n
       log_likelihood = -n/2 * (np.log(2*np.pi*sigma2) + 1)

       aic = 2*k - 2*log_likelihood
       bic = k*np.log(n) - 2*log_likelihood

       return aic, bic

   # Compare models
   for n_modes in [2, 3, 4]:
       for shared in [True, False]:
           model = FMLIKH(n_modes=n_modes, shared_alpha=shared)
           model.fit(X, y)
           aic, bic = compute_aic_bic(model, X, y)
           print(f"N={n_modes}, shared={shared}: AIC={aic:.1f}, BIC={bic:.1f}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

For uncertainty quantification:

.. code-block:: python

   from rheojax.models.fikh import FMLIKH

   model = FMLIKH(n_modes=3, shared_alpha=True)

   # Point estimate first
   model.fit(X, y, test_mode='startup')

   # Bayesian inference
   result = model.fit_bayesian(
       X, y,
       num_warmup=1500,      # More for multi-mode
       num_samples=3000,
       num_chains=4,
       seed=42
   )

   # Check per-mode convergence
   for i in range(1, 4):
       print(f"Mode {i}:")
       print(f"  τ_thix_{i}: {result.posterior_samples[f'tau_thix_{i}'].mean():.2f}")
       print(f"  w_{i}: {result.posterior_samples[f'w_{i}'].mean():.3f}")


Numerical Implementation
------------------------

Multi-Mode History Buffer Management
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FMLIKH maintains N separate history buffers for the L1 scheme:

.. code-block:: text

   λ_histories = [
       [λ_1^{n-N_h+1}, ..., λ_1^n],  # Mode 1 history
       [λ_2^{n-N_h+1}, ..., λ_2^n],  # Mode 2 history
       ...
       [λ_N^{n-N_h+1}, ..., λ_N^n],  # Mode N history
   ]
   Shape: (N, n_history)

**Memory scaling**: O(N × n_history) ≈ O(N × 100-500) bytes per state

JAX vmap Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~

The N fractional derivatives are computed in parallel via JAX vmap:

.. code-block:: python

   @jax.jit
   def compute_all_mode_derivatives(lam_histories, dt, alpha, b_coeffs):
       """Vectorized fractional derivative over all modes."""
       # vmap over mode dimension
       return jax.vmap(
           lambda hist: caputo_derivative_l1(hist, dt, alpha, b_coeffs)
       )(lam_histories)

This provides near-linear scaling with N on both CPU and GPU.

Precompilation
~~~~~~~~~~~~~~

For production runs:

.. code-block:: python

   # Trigger JIT compilation
   compile_time = model.precompile()
   print(f"FMLIKH (N={model.n_modes}) compiled in {compile_time:.1f}s")


Usage Examples
--------------

Basic Initialization
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.fikh import FMLIKH

   # 3-mode model with shared α
   model = FMLIKH(n_modes=3, shared_alpha=True)

   # Set global parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)
   model.parameters.set_value("delta_sigma_y", 60.0)
   model.parameters.set_value("alpha_structure", 0.55)

   # Set per-mode parameters
   timescales = [1.0, 10.0, 100.0]
   weights = [0.2, 0.3, 0.5]

   for i, (tau, w) in enumerate(zip(timescales, weights), 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)
       model.parameters.set_value(f"w_{i}", w)
       model.parameters.set_value(f"Gamma_{i}", 0.5)

Per-Mode :math:`\alpha` Setup
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Model with per-mode fractional orders
   model = FMLIKH(n_modes=3, shared_alpha=False)

   # Fast mode: nearly exponential
   model.parameters.set_value("alpha_structure_1", 0.85)
   model.parameters.set_value("tau_thix_1", 0.5)

   # Medium mode: moderate memory
   model.parameters.set_value("alpha_structure_2", 0.6)
   model.parameters.set_value("tau_thix_2", 5.0)

   # Slow mode: strong memory (glassy)
   model.parameters.set_value("alpha_structure_3", 0.35)
   model.parameters.set_value("tau_thix_3", 50.0)

Prediction
~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma_flow = model.predict_flow_curve(gamma_dot)

   # Startup
   t = np.linspace(0, 50, 500)
   sigma_startup = model.predict_startup(t, gamma_dot=1.0)

   # Recovery
   sigma_relax = model.predict_relaxation(t, sigma_0=100.0)

Fitting
~~~~~~~

.. code-block:: python

   # Fit to startup data
   model.fit(t, stress_data, test_mode='startup')

   # Bayesian inference
   result = model.fit_bayesian(
       t, stress_data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Extract mode information
   mode_info = model.get_mode_info()
   print(mode_info)

Comparing Mode Effects
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   t = np.linspace(0, 100, 500)

   plt.figure(figsize=(10, 4))

   # Compare N=2 vs N=3 modes
   for n_modes in [2, 3]:
       model = FMLIKH(n_modes=n_modes, shared_alpha=True)
       model.parameters.set_value("alpha_structure", 0.5)

       # Logarithmic timescale distribution
       taus = np.logspace(0, 2, n_modes)
       for i, tau in enumerate(taus, 1):
           model.parameters.set_value(f"tau_thix_{i}", tau)

       sigma = model.predict_relaxation(t, sigma_0=100.0)
       plt.plot(t, sigma, label=f'N = {n_modes}')

   plt.xlabel('Time [s]')
   plt.ylabel('Stress [Pa]')
   plt.legend()
   plt.title('FMLIKH: Effect of Number of Modes')


Relation to FIKH
----------------

FMLIKH with N=1 is equivalent to FIKH:

.. code-block:: python

   # These are equivalent:
   model_fikh = FIKH(alpha_structure=0.6)
   model_fmlikh = FMLIKH(n_modes=1, shared_alpha=True)
   model_fmlikh.parameters.set_value("alpha_structure", 0.6)
   model_fmlikh.parameters.set_value("w_1", 1.0)

**Parameter mapping:**

- ``sigma_y0`` (FMLIKH) ↔ ``sigma_y0`` (FIKH)
- ``delta_sigma_y`` (FMLIKH) ↔ ``delta_sigma_y`` (FIKH)
- ``alpha_structure`` (shared) ↔ ``alpha_structure`` (FIKH)
- ``tau_thix_1``, ``Gamma_1`` ↔ ``tau_thix``, ``Gamma``
- ``w_1 = 1.0`` implicitly

**When to use which:**

- **FIKH**: Single structural population with power-law memory
- **FMLIKH**: Hierarchical structure with multiple populations


Limitations and Considerations
------------------------------

**Computational cost:**

- N modes require N history buffers → O(N × n_history) memory
- Fractional derivatives computed per mode → O(N × n_history) per step
- With JAX vmap: near-linear scaling, but still more expensive than ML-IKH

**Parameter identifiability:**

- Per-mode :math:`\alpha` with similar timescales can be poorly identified
- Weight-timescale correlations possible
- Use shared :math:`\alpha` when per-mode :math:`\alpha` values are similar

**Physical interpretation:**

- Mode assignment to specific microstructural features requires independent evidence
- The multi-mode fractional decomposition is mathematical; physical correspondence
  should be validated experimentally


References
----------

**FIKH Foundation:**

See :doc:`fikh` references [1-10] for fractional calculus and IKH background.

**Multi-Mode Extensions:**

.. [11] Wei, Y., Solomon, M. J., and Larson, R. G. (2018). "A multimode structural
   kinetics constitutive equation for the transient rheology of thixotropic
   elasto-viscoplastic fluids." *J. Rheol.*, 62(1), 321-342.

.. [12] Fielding, S. M., et al. (2009). "Aging and rheology in soft materials."
   *J. Rheol.*, 53(1), 39-64.

**Stretched Exponentials:**

.. [13] Kohlrausch, R. (1854). "Theorie des elektrischen Rückstandes in der
   Leidener Flasche." *Ann. Phys.*, 167(2), 179-214.

.. [14] Williams, G. and Watts, D. C. (1970). "Non-symmetrical dielectric relaxation
   behaviour arising from a simple empirical decay function." *Trans. Faraday Soc.*,
   66, 80-85.


See Also
--------

- :doc:`fikh` — Single-mode fractional IKH (FMLIKH with N=1)
- :doc:`../ikh/ml_ikh` — Integer-order multi-mode IKH
- :doc:`../ikh/mikh` — Single-mode integer-order IKH (:math:`\alpha` = 1 limit)
- :doc:`../sgr/index` — Soft Glassy Rheology (alternative for aging systems)
