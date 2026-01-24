.. _dmt_models:

======================================
DMT Thixotropic Models
======================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
===============

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Classes**
     - ``DMTLocal``, ``DMTNonlocal``
   * - **Physics**
     - de Souza Mendes-Thompson thixotropic viscoelasticity
   * - **Viscosity Closures**
     - ``"exponential"``, ``"herschel_bulkley"``
   * - **Elasticity**
     - Optional Maxwell backbone (``include_elasticity=True``)
   * - **Protocols**
     - FLOW_CURVE, CREEP, RELAXATION, STARTUP, OSCILLATION, LAOS
   * - **Key Features**
     - Structure kinetics, stress overshoot, delayed yielding, shear banding

**Import:**

.. code-block:: python

   from rheojax.models import DMTLocal, DMTNonlocal

**Basic Usage:**

.. code-block:: python

   # Exponential closure with Maxwell elasticity
   model = DMTLocal(closure="exponential", include_elasticity=True)

   # Herschel-Bulkley closure for yield-stress materials
   model = DMTLocal(closure="herschel_bulkley", include_elasticity=True)

   # Nonlocal variant for shear banding
   model = DMTNonlocal(closure="exponential", n_points=51, gap_width=1e-3)

Notation Guide
==============

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`\lambda`
     - Structure parameter (0 = broken, 1 = fully structured)
     - dimensionless
   * - :math:`\eta_0`
     - Zero-shear viscosity (at :math:`\lambda = 1`)
     - Pa·s
   * - :math:`\eta_\infty`
     - Infinite-shear viscosity (at :math:`\lambda = 0`)
     - Pa·s
   * - :math:`\tau_y`
     - Yield stress
     - Pa
   * - :math:`K`
     - Consistency index
     - Pa·s\ :sup:`n`
   * - :math:`n`
     - Flow index
     - dimensionless
   * - :math:`G`
     - Elastic modulus
     - Pa
   * - :math:`\theta`
     - Relaxation time (:math:`= \eta/G`)
     - s
   * - :math:`t_{eq}`
     - Equilibrium (buildup) timescale
     - s
   * - :math:`a`
     - Breakdown rate coefficient
     - dimensionless
   * - :math:`c`
     - Breakdown rate exponent
     - dimensionless

Overview
========

The de Souza Mendes-Thompson (DMT) model [deSouzaMendes2009]_ [Mendes2011]_ is a
structural-kinetics based thixotropic model that captures time-dependent rheological
behavior through a scalar structure parameter :math:`\lambda \in [0, 1]`.

Key Features
------------

1. **Structure-dependent viscosity**: Material properties depend on microstructural state
   tracked by :math:`\lambda`, with fully structured (:math:`\lambda = 1`) giving high
   viscosity and fully broken (:math:`\lambda = 0`) giving low viscosity.

2. **Structure kinetics**: The structure evolves through competing buildup (aging at rest)
   and breakdown (shear-induced destruction) processes.

3. **Multiple viscosity closures**: Either smooth exponential dependence or
   Herschel-Bulkley form with explicit yield stress.

4. **Optional viscoelasticity**: Maxwell backbone enables stress overshoot in startup
   and elastic recoil.

5. **Spatial extension**: Nonlocal variant captures shear banding through structure
   diffusion.

Physical Foundations
====================

Structure Parameter
-------------------

The structure parameter :math:`\lambda \in [0, 1]` represents the degree of
microstructural organization:

- :math:`\lambda = 1`: Fully structured (at rest, aged)
- :math:`\lambda = 0`: Fully broken (high shear, rejuvenated)

Physical interpretation includes:

- Colloidal networks: Bond connectivity between particles
- Polymer solutions: Entanglement density
- Emulsions/foams: Droplet/bubble deformation state

Structure Kinetics
------------------

The structure evolves according to:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}|^c}{t_{eq}}

where:

- First term: Buildup (aging) drives :math:`\lambda \to 1` at rate :math:`1/t_{eq}`
- Second term: Breakdown destroys structure at rate proportional to :math:`|\dot{\gamma}|^c`

At equilibrium (:math:`d\lambda/dt = 0`):

.. math::

   \lambda_{eq} = \frac{1}{1 + a|\dot{\gamma}|^c}

This gives :math:`\lambda_{eq} \to 1` as :math:`\dot{\gamma} \to 0` and
:math:`\lambda_{eq} \to 0` as :math:`\dot{\gamma} \to \infty`.

Viscosity Closures
==================

Exponential Closure
-------------------

A smooth, monotonic relationship between structure and viscosity:

.. math::

   \eta(\lambda) = \eta_\infty \left(\frac{\eta_0}{\eta_\infty}\right)^\lambda

Properties:

- :math:`\eta(1) = \eta_0` (zero-shear viscosity)
- :math:`\eta(0) = \eta_\infty` (infinite-shear viscosity)
- No explicit yield stress (power-law-like flow curve)

Herschel-Bulkley Closure
------------------------

Structure-dependent yield stress and consistency:

.. math::

   \sigma = \tau_y(\lambda) + K(\lambda) |\dot{\gamma}|^n + \eta_\infty \dot{\gamma}

where:

.. math::

   \tau_y(\lambda) &= \tau_{y0} \lambda^{m_1} \\
   K(\lambda) &= K_0 \lambda^{m_2}

Properties:

- Explicit yield stress :math:`\tau_y` controlled by :math:`\lambda`
- True yield stress behavior (regularized with Papanastasiou)
- Structure-dependent flow index contribution

Maxwell Viscoelasticity
=======================

When ``include_elasticity=True``, a Maxwell element adds elastic response:

.. math::

   \frac{d\sigma}{dt} = G(\lambda) \dot{\gamma} - \frac{G(\lambda)}{\eta(\lambda)} \sigma

where the elastic modulus depends on structure:

.. math::

   G(\lambda) = G_0 \lambda^{m_G}

This gives:

- **Relaxation time**: :math:`\theta(\lambda) = \eta(\lambda) / G(\lambda)`
- **Stress overshoot**: In startup, stress overshoots before reaching steady state
- **Stress relaxation**: After cessation of flow, stress decays exponentially
- **SAOS**: Storage (:math:`G'`) and loss (:math:`G''`) moduli from linear response

Steady-State Flow Curve
=======================

At equilibrium, the structure and stress are uniquely determined by shear rate.

Exponential Closure
-------------------

.. math::

   \sigma_{ss}(\dot{\gamma}) = \eta(\lambda_{eq}(\dot{\gamma})) \cdot \dot{\gamma}

where :math:`\eta` depends on :math:`\lambda_{eq} = 1/(1 + a|\dot{\gamma}|^c)`.

Herschel-Bulkley Closure
------------------------

.. math::

   \sigma_{ss} = \tau_{y0}\lambda_{eq}^{m_1} + K_0\lambda_{eq}^{m_2}|\dot{\gamma}|^n + \eta_\infty\dot{\gamma}

Parameters
==========

Core Viscosity Parameters
-------------------------

.. list-table::
   :widths: 15 45 12 12 16
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``eta_0``
     - Zero-shear viscosity
     - Pa·s
     - 1e5
     - [1e2, 1e8]
   * - ``eta_inf``
     - Infinite-shear viscosity
     - Pa·s
     - 0.1
     - [1e-3, 1e2]

Herschel-Bulkley Parameters (``closure="herschel_bulkley"`` only)
-----------------------------------------------------------------

.. list-table::
   :widths: 15 45 12 12 16
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``tau_y0``
     - Fully-structured yield stress
     - Pa
     - 10.0
     - [0.1, 1e4]
   * - ``K0``
     - Fully-structured consistency
     - Pa·s\ :sup:`n`
     - 5.0
     - [0.1, 1e3]
   * - ``n_flow``
     - Flow index
     - —
     - 0.5
     - [0.1, 1.0]
   * - ``m1``
     - Yield stress exponent
     - —
     - 1.0
     - [0.5, 2.0]
   * - ``m2``
     - Consistency exponent
     - —
     - 1.0
     - [0.5, 2.0]

Elastic Parameters (``include_elasticity=True`` only)
-----------------------------------------------------

.. list-table::
   :widths: 15 45 12 12 16
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``G0``
     - Elastic modulus at :math:`\lambda = 1`
     - Pa
     - 100.0
     - [1e0, 1e6]
   * - ``m_G``
     - Modulus structure exponent
     - —
     - 1.0
     - [0.5, 2.0]

Structure Kinetics Parameters
-----------------------------

.. list-table::
   :widths: 15 45 12 12 16
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``t_eq``
     - Equilibrium (buildup) timescale
     - s
     - 100.0
     - [0.1, 1e4]
   * - ``a``
     - Breakdown rate coefficient
     - —
     - 1.0
     - [1e-3, 1e2]
   * - ``c``
     - Breakdown rate exponent
     - —
     - 1.0
     - [0.1, 2.0]

Nonlocal Parameters (``DMTNonlocal`` only)
------------------------------------------

.. list-table::
   :widths: 15 45 12 12 16
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``D_lambda``
     - Structure diffusion coefficient
     - m²/s
     - 1e-9
     - [1e-12, 1e-6]

API Reference
=============

DMTLocal
--------

.. autoclass:: rheojax.models.dmt.DMTLocal
   :members:
   :undoc-members:
   :show-inheritance:

DMTNonlocal
-----------

.. autoclass:: rheojax.models.dmt.DMTNonlocal
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
==============

Flow Curve Fitting
------------------

.. code-block:: python

   import numpy as np
   from rheojax.models import DMTLocal

   # Experimental flow curve data
   gamma_dot = np.logspace(-2, 2, 20)  # 1/s
   stress = np.array([...])  # Pa

   # Fit with exponential closure
   model = DMTLocal(closure="exponential", include_elasticity=True)
   model.fit(gamma_dot, stress, test_mode="flow_curve")

   # Predict flow curve
   gamma_dot_pred = np.logspace(-3, 3, 100)
   stress_pred = model.predict(gamma_dot_pred, test_mode="flow_curve")

Startup Shear with Stress Overshoot
-----------------------------------

.. code-block:: python

   from rheojax.models import DMTLocal

   model = DMTLocal(closure="exponential", include_elasticity=True)

   # Startup at γ̇ = 10 s⁻¹ from fully-structured state
   t, stress, lam = model.simulate_startup(
       gamma_dot=10.0,
       t_end=100.0,
       dt=0.01,
       lam_init=1.0  # Aged state
   )

   # Find stress overshoot
   peak_idx = np.argmax(stress)
   overshoot_ratio = stress[peak_idx] / stress[-1]

Creep with Delayed Yielding
---------------------------

The creep response differs significantly between viscous and Maxwell variants.

**Viscous Variant** (``include_elasticity=False``):

Pure viscous flow: :math:`\gamma(t) = \int_0^t \sigma_0 / \eta(\lambda(s)) \, ds`

.. code-block:: python

   from rheojax.models import DMTLocal

   model = DMTLocal(closure="herschel_bulkley", include_elasticity=False)

   # Apply constant stress
   t, gamma, gamma_dot, lam = model.simulate_creep(
       sigma_0=50.0,  # Applied stress (Pa)
       t_end=1000.0,
       dt=0.1,
       lam_init=1.0  # Start from aged state
   )

   # Observe delayed yielding: initial slow creep, then acceleration
   # as structure breaks down

**Maxwell Variant** (``include_elasticity=True``):

Total strain includes both elastic and viscous contributions:

.. math::

   \gamma(t) = \underbrace{\frac{\sigma_0}{G(\lambda(t))}}_{\gamma_e(t)} + \underbrace{\int_0^t \frac{\sigma_0}{\eta(\lambda(s))} \, ds}_{\gamma_v(t)}

Key features:

- **Initial elastic jump**: :math:`\gamma(0^+) = \sigma_0 / G(\lambda_0)` — instantaneous response
- **Elastic strain evolution**: As structure breaks down (:math:`\lambda \downarrow`), :math:`G \downarrow`, so :math:`\gamma_e \uparrow`
- **Viscous flow**: Accumulates continuously via :math:`\dot{\gamma}_v = \sigma_0 / \eta(\lambda)`

.. code-block:: python

   from rheojax.models import DMTLocal
   import numpy as np

   model = DMTLocal(closure="exponential", include_elasticity=True)

   # Parameters for initial elastic strain estimate
   G0 = model.parameters.get_value("G0")
   sigma_0 = 100.0  # Pa

   # Expected initial elastic strain: γ_e(0) = σ₀/G₀
   gamma_e_expected = sigma_0 / G0
   print(f"Expected initial elastic strain: {gamma_e_expected:.4f}")

   # Simulate creep
   t, gamma, gamma_dot, lam = model.simulate_creep(
       sigma_0=sigma_0,
       t_end=500.0,
       dt=0.1,
       lam_init=1.0
   )

   # Verify initial strain includes elastic contribution
   print(f"Actual initial strain: {gamma[0]:.4f}")

   # As structure breaks (λ decreases), elastic strain increases
   # because G(λ) = G₀·λ^m_G decreases
   print(f"Structure: {lam[0]:.3f} → {lam[-1]:.3f}")
   print(f"Final strain: {gamma[-1]:.4f}")

SAOS Predictions (Maxwell Variant)
----------------------------------

.. code-block:: python

   import numpy as np
   from rheojax.models import DMTLocal

   model = DMTLocal(closure="exponential", include_elasticity=True)

   omega = np.logspace(-3, 3, 50)  # rad/s
   G_prime, G_double_prime = model.predict_saos(omega, lam_0=1.0)

   # Crossover frequency ω_c where G' = G''
   # Related to relaxation time θ = η₀/G₀

LAOS Analysis
-------------

.. code-block:: python

   from rheojax.models import DMTLocal

   model = DMTLocal(closure="exponential", include_elasticity=True)

   # Simulate LAOS
   result = model.simulate_laos(
       gamma_0=0.5,  # Strain amplitude
       omega=1.0,    # Angular frequency (rad/s)
       n_cycles=10,
       points_per_cycle=128
   )

   # Extract harmonics
   harmonics = model.extract_harmonics(result, n_harmonics=5)
   # harmonics["G1_prime"], harmonics["G3_prime"], etc.

Shear Banding with Nonlocal Model
---------------------------------

.. code-block:: python

   from rheojax.models import DMTNonlocal

   model = DMTNonlocal(
       closure="exponential",
       include_elasticity=True,
       n_points=101,
       gap_width=1e-3  # 1 mm gap
   )

   # Simulate steady shear
   result = model.simulate_steady_shear(
       gamma_dot_avg=10.0,  # Average shear rate
       t_end=500.0,
       dt=1.0
   )

   # Detect banding
   banding_info = model.detect_banding(result, threshold=0.1)
   print(f"Shear banding: {banding_info['is_banding']}")
   print(f"Band contrast: {banding_info['band_contrast']:.2f}")

Numerical Implementation
========================

ODE Integration
---------------

Time-stepping simulations use ``jax.lax.scan`` for efficient compilation:

.. code-block:: python

   def step(state, _):
       lam, sigma = state
       # Update structure
       dlam = structure_evolution(lam, gamma_dot, t_eq, a, c)
       lam_new = clip(lam + dt * dlam, 0, 1)
       # Update stress (Maxwell)
       dsigma = G(lam) * gamma_dot - sigma / theta(lam)
       sigma_new = sigma + dt * dsigma
       return (lam_new, sigma_new), (sigma_new, lam_new)

   _, (stress, lam) = jax.lax.scan(step, init_state, None, length=n_steps)

JIT Compilation
---------------

All core kernels are JIT-compiled for performance:

.. code-block:: python

   @jax.jit
   def equilibrium_structure(gamma_dot, a, c):
       return 1.0 / (1.0 + a * jnp.abs(gamma_dot) ** c)

   @jax.jit
   def viscosity_exponential(lam, eta_0, eta_inf):
       return eta_inf * jnp.power(eta_0 / eta_inf, lam)

First compilation may take 1-2 seconds; subsequent calls are fast.

Papanastasiou Regularization
----------------------------

For numerical stability, the Herschel-Bulkley yield stress is regularized:

.. math::

   \sigma = \tau_y \left(1 - e^{-m|\dot{\gamma}|}\right) + K|\dot{\gamma}|^n + \eta_\infty\dot{\gamma}

where :math:`m` is a large regularization parameter (default: 1000).

Literature References
=====================

.. [deSouzaMendes2009] de Souza Mendes, P. R. (2009). "Modeling the thixotropic behavior
   of structured fluids." *Journal of Non-Newtonian Fluid Mechanics*, 164(1-3), 66-75.

.. [Mendes2011] de Souza Mendes, P. R., & Thompson, R. L. (2012). "A critical overview
   of elasto-viscoplastic thixotropic modeling." *Journal of Non-Newtonian Fluid
   Mechanics*, 187-188, 8-15.

.. [Thompson2014] Thompson, R. L., & de Souza Mendes, P. R. (2014). "Thixotropic behavior
   of elasto-viscoplastic materials." *Physics of Fluids*, 26(2), 023101.
