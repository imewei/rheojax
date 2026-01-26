.. _dmt_models:

======================================
DMT Thixotropic Models
======================================

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

Historical Context
==================

The DMT model represents a sophisticated synthesis of several theoretical traditions
in thixotropic rheology. Understanding this lineage helps clarify the model's
capabilities and limitations.

Jeffreys/Oldroyd-B Backbone
---------------------------

When ``include_elasticity=True``, the DMT model uses a **structure-dependent Maxwell
element in parallel with a Newtonian element**. This yields the classical Jeffreys
(three-element) viscoelastic form:

.. math::

   \tau_1(\lambda)\dot{\sigma} + \sigma = \eta(\lambda)\left(\dot{\gamma} + \tau_2(\lambda)\ddot{\gamma}\right)

where the relaxation time :math:`\tau_1(\lambda)` and retardation time :math:`\tau_2(\lambda)`
depend on the structure parameter. This formulation is numerically equivalent to the
Maxwell-in-parallel representation used in RheoJAX:

.. math::

   \sigma(t) = \sigma_M(t) + \eta_s(\lambda)\dot{\gamma}(t)

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda)} = G(\lambda)\dot{\gamma}

This form avoids :math:`\ddot{\gamma}` and is ideal for protocol replay and fitting.

Evolution from Simple Thixotropy
--------------------------------

The DMT model evolved from simpler structural-kinetics models:

1. **Coussot model** (early 2000s): Introduced the avalanche effect and viscosity
   bifurcation for yield-stress fluids. Pure viscosity-structure coupling without
   elasticity.

2. **Houska model**: Added Bingham-like yield stress with separate thixotropic
   contributions to both yield stress and consistency.

3. **Mujumdar model** (2002): Introduced elastic strain as a state variable with
   a yield function, enabling stress overshoot predictions.

4. **DMT model** (2009-2014): Unified these elements into a comprehensive framework
   with explicit yield stress, optional Maxwell viscoelasticity, and structure-dependent
   material functions.

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

Fluidity Interpretation
-----------------------

The structure parameter :math:`\lambda` has an alternative interpretation in terms
of **fluidity** :math:`\phi = 1/\eta`, the inverse of viscosity:

- High :math:`\lambda` (structured) → low :math:`\phi` (viscous, solid-like)
- Low :math:`\lambda` (broken) → high :math:`\phi` (fluid, liquid-like)

**Fluidity-based kinetics** provide an equivalent formulation:

.. math::

   \frac{d\phi}{dt} = -\frac{\phi - \phi_{min}}{t_{age}} + c|\dot{\gamma}|^m(\phi_{max} - \phi)

This is mathematically equivalent to the :math:`\lambda`-form after a monotone
transformation. The fluidity approach is particularly natural for:

- Spatial extensions (nonlocal fluidity models)
- Connection to soft glassy rheology (SGR)
- Shear banding analysis

**Cooperativity length** for nonlocal models:

.. math::

   \xi \sim \sqrt{D_\lambda \cdot t_{eq}}

where :math:`D_\lambda` is the structure diffusion coefficient. This length scale
sets the minimum shear band width and prevents ill-posed sharp band interfaces.

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

Viscosity Bifurcation
=====================

A defining feature of yield-stress materials is the **viscosity bifurcation**: a
discontinuous transition between solid-like and liquid-like behavior at a critical
stress.

Critical Stress
---------------

For the Herschel-Bulkley closure, the viscosity bifurcation occurs at the yield stress:

- :math:`\sigma < \tau_y(\lambda)`: Effective viscosity diverges (:math:`\eta \to \infty`)
- :math:`\sigma > \tau_y(\lambda)`: Finite viscosity, flow occurs

At equilibrium structure :math:`\lambda_{eq}`:

.. math::

   \sigma_c = \tau_{y0} \lambda_{eq}^{m_1} = \frac{\tau_{y0}}{(1 + a|\dot{\gamma}|^c)^{m_1}}

Avalanche Effect
----------------

Near the critical stress, thixotropic materials exhibit the **avalanche effect**
[Coussot2002]_:

1. **Below yield**: :math:`\sigma_0 < \tau_y`, structure builds up, viscosity increases
2. **Near yield**: :math:`\sigma_0 \approx \tau_y`, metastable state with slow creep
3. **Delayed yielding**: Structure slowly breaks down, then catastrophic flow onset
4. **Above yield**: Immediate structure breakdown and steady flow

This manifests in creep experiments as an initial slow strain accumulation followed
by rapid acceleration when the structure can no longer support the applied stress.

Connection to Herschel-Bulkley
------------------------------

In the limit of fast thixotropic kinetics (:math:`t_{eq} \to 0`), the DMT model
with Herschel-Bulkley closure recovers the classical Herschel-Bulkley constitutive
equation:

.. math::

   \sigma = \tau_y + K|\dot{\gamma}|^n \quad \text{for } \sigma > \tau_y

The key difference is that DMT predicts **time-dependent** transitions between
solid and liquid states, while HB assumes instantaneous equilibrium.

Related Models Comparison
=========================

The DMT model belongs to a family of structural-kinetics thixotropic models. The
table below compares key features:

.. list-table::
   :widths: 16 12 12 12 10 10 28
   :header-rows: 1

   * - Model
     - Yield Stress
     - Elasticity
     - Thixotropy
     - Params
     - Complexity
     - Best For
   * - Coussot
     - Implicit
     - No
     - Yes
     - ~4
     - Low
     - Avalanche effect, simple thixotropy
   * - Houska
     - Explicit
     - No
     - Yes
     - ~7
     - Medium
     - Drilling fluids, concrete
   * - Mujumdar
     - Explicit
     - Yes
     - Yes
     - ~6
     - Medium
     - Stress overshoot, transients
   * - Dullaert-Mewis
     - Implicit
     - Partial
     - Yes
     - ~8
     - Medium
     - Colloidal suspensions
   * - DMT
     - Explicit
     - Optional
     - Yes
     - ~10
     - High
     - Full protocol matching

Coussot Model
-------------

Simple thixotropy without explicit yield stress:

.. math::

   \sigma = \eta(\lambda)\dot{\gamma}, \quad \eta(\lambda) = \eta_0(1 + \lambda^n)

.. math::

   \frac{d\lambda}{dt} = \frac{1}{\theta_0} - \alpha\lambda|\dot{\gamma}|

**Strengths**: Simple, captures avalanche effect. **Limitations**: No elasticity, no stress overshoot.

Houska Model
------------

Bingham-like with thixotropic contributions:

.. math::

   \sigma = (\tau_{y0} + \tau_{yt}\lambda) + (K_0 + K_t\lambda)|\dot{\gamma}|^n

**Strengths**: Explicit yield stress, separate thixotropic contributions.
**Limitations**: No elasticity, many parameters.

Mujumdar Model
--------------

Elastic strain with yield function:

.. math::

   \sigma = G\gamma_e + \eta(\lambda)\dot{\gamma}

.. math::

   \frac{d\gamma_e}{dt} = \dot{\gamma} - k_{rel}\gamma_e f(\sigma)

where :math:`f(\sigma)` is a yield function. **Strengths**: Stress overshoot, elastic recovery.
**Limitations**: Step-function yield behavior.

When to Use Which Model
-----------------------

.. list-table::
   :widths: 35 32 33
   :header-rows: 1

   * - Situation
     - Recommended Model
     - Why
   * - Quick thixotropy fitting
     - Coussot → DMT
     - Start simple, upgrade if needed
   * - Need yield + thixotropy
     - Houska or DMT
     - Explicit yield stress
   * - Stress overshoot important
     - DMT or Mujumdar
     - Requires elasticity
   * - Creep with delayed yield
     - DMT
     - Best bifurcation physics
   * - LAOS analysis
     - DMT (with elasticity)
     - Full nonlinear response
   * - Full protocol matching
     - DMT
     - Most comprehensive

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
     - m^2/s
     - 1e-9
     - [1e-12, 1e-6]

Dimensionless Groups
====================

Several dimensionless numbers characterize DMT model behavior and help classify
rheological regimes.

Deborah Number (De)
-------------------

The ratio of thixotropic timescale to experimental timescale:

.. math::

   De = \frac{t_{eq}}{t_{exp}}

where :math:`t_{exp}` is a characteristic experimental time (e.g., period :math:`2\pi/\omega`
for oscillation, :math:`1/\dot{\gamma}` for steady shear).

- :math:`De \gg 1`: Thixotropy dominates; structure cannot equilibrate during experiment
- :math:`De \ll 1`: Quasi-steady behavior; structure rapidly equilibrates

Weissenberg Number (Wi)
-----------------------

Characterizes the structure breakdown rate:

.. math::

   Wi = \dot{\gamma} \cdot t_{eq}

- :math:`Wi \gg 1`: Strong breakdown, :math:`\lambda \to 0`
- :math:`Wi \ll 1`: Structure preserved, :math:`\lambda \to 1`

Structure Number (Sn)
---------------------

The breakdown efficiency parameter:

.. math::

   Sn = a \cdot |\dot{\gamma}|^c

This appears directly in the equilibrium structure:

.. math::

   \lambda_{eq} = \frac{1}{1 + Sn}

Regime Classification
---------------------

.. list-table::
   :widths: 22 14 14 50
   :header-rows: 1

   * - Regime
     - De
     - Wi
     - Behavior
   * - Linear viscoelastic
     - Low
     - Low
     - Standard Maxwell, :math:`\lambda \approx 1`
   * - Thixotropic
     - High
     - Variable
     - Time-dependent, hysteresis loops
   * - Shear-thinning
     - Low
     - High
     - Steady power-law, :math:`\lambda = \lambda_{eq}`
   * - Glass-like
     - High
     - High
     - Aging + strong breakdown, complex transients

These regimes map to different experimental signatures:

- **Linear VE**: Standard :math:`G'`, :math:`G''` from SAOS
- **Thixotropic**: Hysteresis in flow curves, recovery experiments
- **Shear-thinning**: Rate-dependent steady viscosity
- **Glass-like**: Aging effects, bifurcation, delayed yielding

Protocol-Specific Equations
===========================

This section provides complete mathematical derivations for each rheological protocol.
These equations form the basis for numerical simulations and analytical predictions.

General Governing System
------------------------

For all protocols, the DMT model is governed by a coupled system of differential equations.

**State variables**:

- :math:`\lambda(t)`: Structure parameter
- :math:`\gamma_e(t)`: Elastic strain (when ``include_elasticity=True``)
- :math:`\sigma(t)`: Stress

**Constitutive equation** (generalized Newtonian form):

.. math::

   \sigma(t) = \eta(\lambda(t), \dot{\gamma}(t)) \cdot \dot{\gamma}(t)

**Structure kinetics**:

.. math::

   \frac{d\lambda}{dt} = \underbrace{\frac{1 - \lambda}{t_{eq}}}_{\text{buildup (aging)}}
   - \underbrace{\frac{a \lambda |\dot{\gamma}|^c}{t_{eq}}}_{\text{breakdown (shear)}}

**Maxwell backbone** (when ``include_elasticity=True``):

.. math::

   \sigma(t) = \sigma_M(t) + \eta_s(\lambda) \dot{\gamma}(t)

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda)} = G(\lambda) \dot{\gamma}

where :math:`\tau_1(\lambda) = \eta_M(\lambda) / G(\lambda)` is the structure-dependent
relaxation time.

Rotation / Steady-State Flow Curve
----------------------------------

**Protocol**: Constant shear rate :math:`\dot{\gamma} = \text{const}`, wait for equilibrium.

Equilibrium Condition
~~~~~~~~~~~~~~~~~~~~~

At steady state, :math:`d\lambda/dt = 0`:

.. math::

   0 = \frac{1 - \lambda_{eq}}{t_{eq}} - \frac{a \lambda_{eq} |\dot{\gamma}|^c}{t_{eq}}

Solving for equilibrium structure:

.. math::

   \boxed{\lambda_{eq}(\dot{\gamma}) = \frac{1}{1 + a|\dot{\gamma}|^c}}

**Limiting behaviors**:

- :math:`\dot{\gamma} \to 0`: :math:`\lambda_{eq} \to 1` (fully structured)
- :math:`\dot{\gamma} \to \infty`: :math:`\lambda_{eq} \to 0` (fully broken)

Steady-State Stress
~~~~~~~~~~~~~~~~~~~

**Exponential closure**:

.. math::

   \sigma_{ss}(\dot{\gamma}) = \eta(\lambda_{eq}) \cdot \dot{\gamma}
   = \eta_\infty \left(\frac{\eta_0}{\eta_\infty}\right)^{\lambda_{eq}} \dot{\gamma}

**Herschel-Bulkley closure**:

.. math::

   \sigma_{ss}(\dot{\gamma}) = \tau_{y0} \lambda_{eq}^{m_1} + K_0 \lambda_{eq}^{m_2} |\dot{\gamma}|^n + \eta_\infty \dot{\gamma}

**Maxwell backbone** (steady state, :math:`\dot{\sigma}_M = 0`):

.. math::

   \sigma_M^{ss} = G(\lambda_{eq}) \tau_1(\lambda_{eq}) \dot{\gamma}

.. math::

   \sigma_{ss} = \sigma_M^{ss} + \eta_s(\lambda_{eq}) \dot{\gamma}
   = \left[ G(\lambda_{eq}) \tau_1(\lambda_{eq}) + \eta_s(\lambda_{eq}) \right] \dot{\gamma}

Controlled-Stress Mode
~~~~~~~~~~~~~~~~~~~~~~

When the rheometer controls stress :math:`\sigma` rather than rate :math:`\dot{\gamma}`,
solve the implicit equation:

.. math::

   \sigma = \eta(\lambda_{ss}, \dot{\gamma}) \cdot \dot{\gamma}

together with:

.. math::

   \lambda_{ss} = \frac{1}{1 + a|\dot{\gamma}|^c}

This can yield **multiple solutions** (S-shaped flow curve), representing a route to
**shear banding** in local models when the middle branch is unstable.

Start-up of Steady Shear
------------------------

**Protocol**: :math:`\dot{\gamma}(t) = \dot{\gamma}_0 H(t)` from rest (:math:`\lambda_0 = 1`).

Governing ODEs
~~~~~~~~~~~~~~

For :math:`t > 0`:

**Structure evolution**:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}_0|^c}{t_{eq}}

**Closed-form solution**: Let :math:`r = \frac{1}{t_{eq}} + \frac{a|\dot{\gamma}_0|^c}{t_{eq}}`. Then:

.. math::

   \lambda(t) = \lambda_{ss} + (\lambda_0 - \lambda_{ss}) e^{-rt}

where :math:`\lambda_{ss} = \frac{1}{1 + a|\dot{\gamma}_0|^c}` is the target equilibrium.

**Maxwell stress evolution** (with elasticity):

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = G(\lambda(t)) \dot{\gamma}_0

This ODE has time-varying coefficients through :math:`\lambda(t)`, requiring numerical integration.

Stress Overshoot Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress overshoot characteristic of thixotropic materials arises from the interplay
of elastic loading and structure breakdown:

1. **Initial elastic rise** (:math:`t \ll t_{eq}`): :math:`\sigma \approx G_0 \cdot \gamma = G_0 \dot{\gamma}_0 t`
   (linear elastic loading while :math:`\lambda \approx 1`)

2. **Structure breakdown begins**: :math:`\lambda` starts decreasing toward :math:`\lambda_{ss}`

3. **Viscosity drop**: As :math:`\lambda` decreases, :math:`\eta(\lambda)` drops exponentially
   (exponential closure) or polynomially (HB closure)

4. **Peak stress**: Maximum occurs when the rate of viscosity decrease exceeds the
   rate of strain increase

5. **Approach to steady state**: :math:`\sigma \to \sigma_{ss}`, :math:`\lambda \to \lambda_{ss}`

**Peak strain estimate** (order of magnitude):

.. math::

   \gamma_{peak} \sim \frac{\tau_y}{G_0} \quad \text{(yield strain)}

**Overshoot ratio**: :math:`\sigma_{peak}/\sigma_{ss}` increases with shear rate and
initial structure level :math:`\lambda_0`.

Purely Generalized Newtonian (No Elasticity)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without elasticity (``include_elasticity=False``), the stress follows viscosity directly:

.. math::

   \sigma(t) = \eta(\lambda(t), \dot{\gamma}_0) \cdot \dot{\gamma}_0

In this case, there is **no elastic stress overshoot**. However, transient "viscosity
overshoot/undershoot" can occur depending on how :math:`\eta(\lambda, \dot{\gamma})`
depends on both arguments. For dense glasses with true overshoots, the Maxwell backbone
is required.

Stress Relaxation (Cessation of Flow)
-------------------------------------

**Protocol**: Shear at :math:`\dot{\gamma}_0` until :math:`t = 0`, then :math:`\dot{\gamma}(t > 0) = 0`.

Structure Recovery
~~~~~~~~~~~~~~~~~~

For :math:`t > 0`, with :math:`\dot{\gamma} = 0`:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}}

(pure buildup, no breakdown). Solution:

.. math::

   \lambda(t) = 1 - (1 - \lambda_0) e^{-t/t_{eq}}

where :math:`\lambda_0 = \lambda_{ss}(\dot{\gamma}_0)` is the structure at cessation.

As :math:`t \to \infty`, :math:`\lambda \to 1` (full recovery).

Stress Relaxation (Generalized Newtonian)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without elasticity:

.. math::

   \sigma(t > 0) = \eta(\lambda(t), 0) \cdot 0 = 0

There is **no stress relaxation** in the strict sense because there is no stored elastic
stress. What is predicted is **structural recovery** (thixotropic rebuild).

Stress Relaxation (Maxwell Backbone)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With elasticity, the Maxwell stress relaxes:

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = 0

Solution:

.. math::

   \sigma_M(t) = \sigma_M(0^+) \exp\left( -\int_0^t \frac{ds}{\tau_1(\lambda(s))} \right)

**Key coupling effect**: As structure rebuilds (:math:`\lambda \to 1`), the viscosity
:math:`\eta(\lambda)` increases exponentially. This causes the relaxation time
:math:`\tau_1 = \eta/G` to become very large, effectively **arresting** the relaxation.

This is the signature of **yielding behavior**: a partially relaxed stress becomes
"frozen in" as the material re-solidifies.

Two Timescales
~~~~~~~~~~~~~~

The relaxation exhibits two timescales:

1. **Fast elastic relaxation**: :math:`\tau_1(\lambda_0) = \eta(\lambda_0)/G(\lambda_0)`
   (initial, while structure is still broken)

2. **Slow structural recovery**: :math:`t_{eq}` (controls rebuilding)

The observed relaxation is an interplay of these, with early fast decay transitioning
to arrested relaxation as the material re-gels.

Creep (Step Stress)
-------------------

**Protocol**: At :math:`t = 0`, apply constant stress :math:`\sigma(t) = \sigma_0 H(t)`.

This is the most diagnostic protocol for thixotropic yield-stress materials, revealing
viscosity bifurcation and delayed yielding.

Inversion for Shear Rate
~~~~~~~~~~~~~~~~~~~~~~~~

The constitutive equation must be inverted to find :math:`\dot{\gamma}(t)`:

.. math::

   \sigma_0 = \eta(\lambda(t), \dot{\gamma}(t)) \cdot \dot{\gamma}(t)

**If** :math:`\eta` does not depend on :math:`\dot{\gamma}` (pure structure-dependent viscosity):

.. math::

   \dot{\gamma}(t) = \frac{\sigma_0}{\eta(\lambda(t))}

**If** :math:`\eta` depends on :math:`\dot{\gamma}` (HB or other shear-thinning):
solve the algebraic equation for :math:`\dot{\gamma}(t)` at each time given :math:`\lambda(t)`.

Structure Evolution Under Creep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The kinetics become:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}(t)|^c}{t_{eq}}

with :math:`\dot{\gamma}(t)` determined from the stress constraint.

**Creep compliance**:

.. math::

   J(t) = \frac{\gamma(t)}{\sigma_0} = \frac{1}{\sigma_0} \int_0^t \dot{\gamma}(s) \, ds

Viscosity Bifurcation
~~~~~~~~~~~~~~~~~~~~~

The creep response exhibits **viscosity bifurcation** [Coussot2002]_:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Stress Regime
     - Behavior
   * - :math:`\sigma_0 < \tau_y(\lambda = 1)`
     - **Arrested creep**: Structure builds faster than it breaks.
       :math:`\lambda \to 1`, :math:`\eta \to \eta_0`, flow stops.
       :math:`J(t) \to \text{const}` (solid-like).
   * - :math:`\sigma_0 \approx \tau_y`
     - **Delayed yielding**: Metastable state with slow creep.
       Eventually, catastrophic flow onset.
   * - :math:`\sigma_0 > \tau_y`
     - **Immediate flow**: Breakdown dominates.
       :math:`\lambda` drops, :math:`\eta` drops, :math:`\dot{\gamma}` accelerates.
       :math:`J(t) \sim t` at long times (liquid-like).

This bifurcation is the hallmark of yield-stress fluids: below a critical stress,
:math:`\eta \to \infty` (solid); above it, :math:`\eta \to` finite (liquid).

Maxwell Variant Creep
~~~~~~~~~~~~~~~~~~~~~

With the Maxwell backbone, solve the coupled system:

**Stress constraint**:

.. math::

   \sigma_0 = \sigma_M(t) + \eta_s(\lambda(t)) \dot{\gamma}(t)

Rearranging:

.. math::

   \dot{\gamma}(t) = \frac{\sigma_0 - \sigma_M(t)}{\eta_s(\lambda(t))}

**Maxwell stress evolution**:

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda)} = G(\lambda) \dot{\gamma}(t)

**Structure kinetics**:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}(t)|^c}{t_{eq}}

The Maxwell variant adds an **initial elastic jump**:

.. math::

   \gamma(0^+) = \gamma_e(0) = \frac{\sigma_0}{G(\lambda_0)}

followed by combined elastic and viscous creep.

SAOS (Small Amplitude Oscillatory Shear)
----------------------------------------

**Protocol**: :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0 \ll 1`.

Limitations of Pure DMT
~~~~~~~~~~~~~~~~~~~~~~~

The **pure DMT model** (generalized Newtonian + structure kinetics) is **not a linear
viscoelastic model**. It cannot produce genuine :math:`G'(\omega)` and :math:`G''(\omega)`
in the standard sense because there is no elastic energy storage.

What pure DMT can predict:

- Time-dependent apparent viscosity under oscillatory shear
- Phase shifts only through structural kinetics (not through elastic storage)

For proper SAOS response, the **Maxwell backbone** (``include_elasticity=True``) is required.

Governing Equations (Full Nonlinear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t), \quad \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

.. math::

   \sigma(t) = \eta(\lambda(t), \dot{\gamma}(t)) \cdot \dot{\gamma}(t)

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}(t)|^c}{t_{eq}}

Linear Regime (Small Amplitude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small :math:`\gamma_0` where breakdown is weak, linearize around :math:`\lambda \approx \lambda_0`:

- :math:`|\dot{\gamma}(t)|^c \sim (\gamma_0 \omega)^c |\cos(\omega t)|^c`

This is **not purely sinusoidal**, so even "small amplitude" produces harmonics in
:math:`\lambda` unless further approximations are made.

Maxwell Variant SAOS
~~~~~~~~~~~~~~~~~~~~

With the Maxwell backbone at a fixed structure level :math:`\lambda_0`:

**Complex modulus**:

.. math::

   G^*(\omega) = \frac{i \omega G(\lambda_0) \tau_1(\lambda_0)}{1 + i \omega \tau_1(\lambda_0)}

**Storage modulus**:

.. math::

   G'(\omega) = G(\lambda_0) \frac{(\omega \tau_1)^2}{1 + (\omega \tau_1)^2}

**Loss modulus**:

.. math::

   G''(\omega) = G(\lambda_0) \frac{\omega \tau_1}{1 + (\omega \tau_1)^2} + \eta_s \omega

where :math:`\tau_1 = \tau_1(\lambda_0)`.

**Limiting behaviors**:

- Low frequency: :math:`G' \sim \omega^2`, :math:`G'' \sim \omega` (Maxwell liquid)
- High frequency: :math:`G' \to G(\lambda_0)`, :math:`G'' \to \eta_s \omega` (elastic solid + viscous)
- Crossover: :math:`\omega_c = 1/\tau_1(\lambda_0)`

For fully structured material (:math:`\lambda_0 = 1`):

- :math:`G'(\omega \to 0) \to G_0` (solid-like plateau)
- :math:`G''` shows a minimum (liquid-like at very low :math:`\omega`, solid-like at intermediate :math:`\omega`)

LAOS (Large Amplitude Oscillatory Shear)
----------------------------------------

**Protocol**: :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0` finite (nonlinear).

LAOS is the natural regime for DMT because large amplitude drives substantial
breakdown/rebuild within cycles, making the structure kinetics dominant.

Full Governing System
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t), \quad \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**Maxwell stress evolution**:

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = G(\lambda(t)) \dot{\gamma}(t)

**Total stress**:

.. math::

   \sigma(t) = \sigma_M(t) + \eta_s(\lambda(t)) \dot{\gamma}(t)

**Structure kinetics**:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\gamma_0 \omega \cos(\omega t)|^c}{t_{eq}}

Fourier Decomposition
~~~~~~~~~~~~~~~~~~~~~

The stress response is periodic but non-sinusoidal:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots} \left[ \sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t) \right]

Only **odd harmonics** appear due to symmetry under :math:`\gamma \to -\gamma`.

**First harmonic moduli**:

.. math::

   G'_1 = \frac{\sigma'_1}{\gamma_0}, \quad G''_1 = \frac{\sigma''_1}{\gamma_0}

**Third harmonic ratio** (nonlinearity measure):

.. math::

   I_{3/1} = \frac{\sqrt{(\sigma'_3)^2 + (\sigma''_3)^2}}{\sqrt{(\sigma'_1)^2 + (\sigma''_1)^2}}

Chebyshev Decomposition
~~~~~~~~~~~~~~~~~~~~~~~

An alternative representation using Chebyshev polynomials of the first kind:

.. math::

   \sigma'(\gamma, \dot{\gamma}) = \gamma_0 \sum_n \left[ e_n T_n(x) + v_n T_n(y) \right]

where :math:`x = \gamma/\gamma_0` and :math:`y = \dot{\gamma}/\dot{\gamma}_{max}`.

The coefficients :math:`e_n` (elastic Chebyshev) and :math:`v_n` (viscous Chebyshev)
provide physical interpretation of the nonlinear response.

Intra-Cycle Structure Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In LAOS, the structure parameter :math:`\lambda` oscillates within each cycle:

1. **Near** :math:`|\dot{\gamma}| = \gamma_0 \omega` (maximum shear rate):
   Strong breakdown, :math:`\lambda` decreases

2. **Near** :math:`\dot{\gamma} = 0` (strain extrema):
   Structure recovery, :math:`\lambda` increases (if :math:`t_{eq}` is comparable to period)

The amplitude of :math:`\lambda` oscillation depends on the ratio
:math:`2\pi / (\omega \cdot t_{eq})`:

- :math:`\omega t_{eq} \gg 1`: Structure cannot follow, averages to intermediate value
- :math:`\omega t_{eq} \ll 1`: Structure tracks instantaneous rate closely

LAOS Signatures from Thixotropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DMT-type models predict characteristic LAOS features:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Feature
     - Physical Origin
   * - Strain softening (:math:`G'_1` decreases with :math:`\gamma_0`)
     - Structure breakdown at large :math:`\gamma_0`
   * - Higher harmonics (:math:`I_{3/1} > 0`)
     - Nonlinear structure kinetics
   * - Asymmetric Lissajous curves
     - Different buildup/breakdown rates (:math:`t_{eq}` vs :math:`t_{br}`)
   * - Intra-cycle yielding
     - Stress peak before strain peak
   * - Secondary loops in Lissajous
     - Elastic recoil combined with thixotropy

Pipkin Diagram
~~~~~~~~~~~~~~

The Pipkin diagram maps behavior in (Wi, De) space:

- **Wi** = :math:`\gamma_0 \omega \cdot t_{eq}` (Weissenberg number, strain amplitude effect)
- **De** = :math:`\omega \cdot t_{eq}` (Deborah number, frequency effect)

.. list-table::
   :widths: 25 15 15 45
   :header-rows: 1

   * - Region
     - :math:`\gamma_0`
     - :math:`\omega`
     - Behavior
   * - Linear viscoelastic
     - Low
     - Any
     - :math:`\lambda \approx 1`, standard :math:`G'`, :math:`G''`
   * - Quasi-steady thixotropic
     - High
     - Low
     - Structure equilibrates within cycle
   * - Nonlinear viscoelastic
     - High
     - High
     - Viscoelastic nonlinearity, limited thixotropy
   * - Thixotropic LAOS
     - Intermediate
     - Intermediate
     - Complex coupling of all effects

Fluidity-Maxwell Extension
==========================

This section describes the theoretical foundation for combining DMT structural kinetics
with Maxwell-type viscoelastic stress dynamics. This extension is essential for
capturing true stress relaxation, SAOS moduli, and elastic overshoots.

Jeffreys Form (Original Formulation)
------------------------------------

The original DMT-style formulation uses a **structure-dependent Jeffreys** (three-element)
viscoelastic backbone:

.. math::

   \tau_1(\lambda) \dot{\sigma} + \sigma = \eta(\lambda) \left( \dot{\gamma} + \tau_2(\lambda) \ddot{\gamma} \right)

where:

- :math:`\tau_1(\lambda)`: Structure-dependent **relaxation time**
- :math:`\tau_2(\lambda)`: Structure-dependent **retardation time**
- :math:`\eta(\lambda)`: Structure-dependent viscosity scale

This arises from "a structure-dependent Maxwell element in parallel with a Newtonian
element," yielding the Jeffreys/Oldroyd-B form with times depending on structure.

Maxwell-in-Parallel Representation
----------------------------------

For numerical implementation, the **Maxwell-in-parallel** form is preferred:

**Decomposition**:

.. math::

   \sigma(t) = \sigma_M(t) + \eta_s(\lambda) \dot{\gamma}(t)

where :math:`\sigma_M` is the Maxwell branch stress and :math:`\eta_s` is the
parallel Newtonian viscosity.

**Maxwell branch ODE**:

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda)} = G(\lambda) \dot{\gamma}

**Advantages**:

1. Avoids computing :math:`\ddot{\gamma}` (difficult in rate-controlled experiments)
2. Natural for time-stepping numerical schemes
3. Clear separation of elastic and viscous contributions
4. Easy initialization: :math:`\sigma_M(0) = 0` for stress-free initial state

Material Functions
------------------

The structure-dependent material functions are:

**Relaxation time**:

.. math::

   \tau_1(\lambda) = \frac{\eta_M(\lambda)}{G(\lambda)}

**Common choices**:

1. Linear: :math:`\tau_1(\lambda) = \tau_0 + \tau_{slope} \cdot \lambda`
2. Reciprocal: :math:`\tau_1(\lambda) = \tau_0 / \lambda` (diverges as :math:`\lambda \to 0`)
3. Power-law: :math:`\tau_1(\lambda) = \tau_0 \lambda^{m_\tau}`

**Elastic modulus**:

.. math::

   G(\lambda) = G_0 \lambda^{m_G}

**Parallel viscosity** (often constant):

.. math::

   \eta_s(\lambda) = \eta_{s0}

Protocol Equations with Maxwell Backbone
----------------------------------------

**Flow curve** (steady state, :math:`\dot{\sigma}_M = 0`):

.. math::

   \sigma_{ss} = G(\lambda_{eq}) \tau_1(\lambda_{eq}) \dot{\gamma} + \eta_s(\lambda_{eq}) \dot{\gamma}

where :math:`\lambda_{eq} = 1/(1 + a|\dot{\gamma}|^c)`.

**Start-up** (constant :math:`\dot{\gamma}_0`):

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = G(\lambda(t)) \dot{\gamma}_0

.. math::

   \lambda(t) = \lambda_{ss} + (\lambda_0 - \lambda_{ss}) e^{-rt}

This can produce **overshoot-like behavior** depending on how fast :math:`\tau_1`
collapses under shear (rejuvenation).

**Cessation** (stop shear at :math:`t = 0`):

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = 0, \quad t > 0

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}}

As :math:`\lambda` rebuilds, :math:`\tau_1` increases, causing stress relaxation to **slow down**
and eventually arrest.

**Creep** (constant :math:`\sigma_0`):

Solve for :math:`\dot{\gamma}` from the Maxwell relation:

.. math::

   \dot{\gamma}(t) = \frac{1}{G(\lambda)} \left[ \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda)} \right]

With :math:`\dot{\sigma} = 0` in creep:

.. math::

   \dot{\gamma}(t) = \frac{\sigma_0}{G(\lambda(t)) \tau_1(\lambda(t))}

Then:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - a \lambda \left| \frac{\sigma_0}{G(\lambda) \tau_1(\lambda)} \right|^c \frac{1}{t_{eq}}

**SAOS** (small amplitude, fixed :math:`\lambda \approx \lambda_*`):

.. math::

   G^*(\omega) = \frac{i \omega G(\lambda_*) \tau_1(\lambda_*)}{1 + i \omega \tau_1(\lambda_*)} + i \omega \eta_s

Giving:

.. math::

   G'(\omega) = G(\lambda_*) \frac{(\omega \tau_1)^2}{1 + (\omega \tau_1)^2}

.. math::

   G''(\omega) = G(\lambda_*) \frac{\omega \tau_1}{1 + (\omega \tau_1)^2} + \eta_s \omega

Aging enters through slow time-dependence of :math:`\lambda_*`.

**LAOS** (full nonlinear):

.. math::

   \dot{\sigma}_M + \frac{\sigma_M}{\tau_1(\lambda(t))} = G(\lambda(t)) \gamma_0 \omega \cos(\omega t)

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{eq}} - a \lambda |\gamma_0 \omega \cos(\omega t)|^c \frac{1}{t_{eq}}

Nonlocal Extension for Shear Banding
====================================

This section describes the spatial extension of the DMT model for capturing shear banding
through fluidity/structure diffusion.

Physical Motivation
-------------------

Shear banding occurs when the material separates into coexisting regions of different
shear rates under uniform stress. Local DMT models can predict S-shaped flow curves
(multiple solutions), but the sharp band interfaces are ill-posed without spatial
regularization.

The **nonlocal fluidity model** introduces a diffusive coupling that:

1. Sets a minimum shear band width (cooperativity length)
2. Regularizes the mathematical problem
3. Captures the spatial structure evolution

Governing Equations
-------------------

**Spatial extension**: Promote :math:`\lambda` to a field :math:`\lambda(y, t)` where
:math:`y` is the gap position:

.. math::

   \frac{\partial \lambda}{\partial t} = \underbrace{\frac{1 - \lambda}{t_{eq}} - \frac{a \lambda |\dot{\gamma}|^c}{t_{eq}}}_{\text{local kinetics}}
   + \underbrace{D_\lambda \frac{\partial^2 \lambda}{\partial y^2}}_{\text{structure diffusion}}

**Stress balance** (planar Couette, low Reynolds number):

.. math::

   \sigma(y, t) = \Sigma(t) \quad \text{(uniform across gap)}

The stress is uniform, but :math:`\dot{\gamma}(y, t)` varies spatially according to
the local constitutive relation:

.. math::

   \Sigma(t) = \eta(\lambda(y, t), \dot{\gamma}(y, t)) \cdot \dot{\gamma}(y, t)

**Constraint**: The spatially-averaged shear rate equals the imposed value:

.. math::

   \frac{1}{H} \int_0^H \dot{\gamma}(y, t) \, dy = \dot{\gamma}_{avg}

where :math:`H` is the gap width.

Cooperativity Length
--------------------

The structure diffusion introduces a characteristic length scale:

.. math::

   \xi = \sqrt{D_\lambda \cdot t_{eq}}

This **cooperativity length** sets the minimum shear band width. Typical values:

- :math:`D_\lambda \sim 10^{-9}` to :math:`10^{-12}` m^2/s
- :math:`t_{eq} \sim 1` to :math:`1000` s
- :math:`\xi \sim 1~\mu\text{m}` to :math:`1` mm

**Physical interpretation**: Structure rearrangements are not purely local but involve
cooperative motion of neighboring material elements over distance :math:`\xi`.

Boundary Conditions
-------------------

**No-flux** (common for rheometer walls):

.. math::

   \frac{\partial \lambda}{\partial y}\bigg|_{y=0} = \frac{\partial \lambda}{\partial y}\bigg|_{y=H} = 0

**Fixed structure** (idealized smooth/rough walls):

.. math::

   \lambda(0, t) = \lambda_{wall}, \quad \lambda(H, t) = \lambda_{wall}

The boundary condition choice affects band nucleation location.

Shear Banding Criteria
----------------------

Shear banding occurs when:

1. The **steady-state flow curve** is non-monotonic (S-shaped)
2. The applied stress lies in the **unstable region** (negative slope)
3. The **diffusion length** :math:`\xi` is much smaller than the gap :math:`H`

**Band contrast**: The ratio of shear rates in high-shear vs low-shear bands:

.. math::

   C = \frac{\dot{\gamma}_{high}}{\dot{\gamma}_{low}}

For strong banding, :math:`C \gg 1`.

**Lever rule**: At steady state, the volume fractions of bands satisfy:

.. math::

   f_{high} \dot{\gamma}_{high} + (1 - f_{high}) \dot{\gamma}_{low} = \dot{\gamma}_{avg}

Transient Banding
-----------------

Shear bands can be:

1. **Transient**: Appear during startup but disappear at steady state
2. **Steady-state**: Persist indefinitely at fixed conditions
3. **Oscillatory**: Bands move or oscillate under steady forcing

The DMT nonlocal model can capture all these behaviors depending on parameters.

Connection to Fluidity Models
-----------------------------

The structure parameter :math:`\lambda` and fluidity :math:`\phi` are related by:

.. math::

   \phi = \frac{1}{\eta(\lambda)}

For the exponential closure:

.. math::

   \phi = \frac{1}{\eta_\infty} \left( \frac{\eta_\infty}{\eta_0} \right)^\lambda

The fluidity-based kinetics:

.. math::

   \frac{\partial \phi}{\partial t} = -\frac{\phi - \phi_{min}}{t_{age}} + c |\dot{\gamma}|^m (\phi_{max} - \phi) + D_\phi \nabla^2 \phi

is mathematically equivalent to the :math:`\lambda`-form after appropriate transformation.

Industrial Applications
=======================

The DMT model family is particularly well-suited for industrially relevant materials
that exhibit combined thixotropy, yield stress, and viscoelasticity.

Waxy Crude Oils
---------------

Waxy crude oils form gel structures at temperatures below the wax appearance temperature.
DMT models capture:

- **Gel strength** (yield stress) from wax crystal network
- **Thixotropic breakdown** during pipeline startup
- **Structure recovery** during shut-in periods
- **Restart pressure** prediction for pipeline design

Key parameters: High :math:`\tau_{y0}`, large :math:`t_{eq}` (slow recovery), temperature-dependent.

Drilling Fluids and Muds
------------------------

Water-based and oil-based drilling muds exhibit complex thixotropic behavior:

- **Gel strength** for cuttings suspension during circulation stops
- **Low viscosity** under high shear for efficient pumping
- **Progressive gel** development over time

The HB closure is preferred for explicit yield stress modeling.

Concrete and Cement Slurries
----------------------------

Fresh concrete and cement pastes show pronounced thixotropy:

- **Formwork pressure** prediction requires thixotropic modeling
- **Pumpability** assessment
- **Self-compacting concrete** (SCC) flow design

Cementitious materials often show strong structure recovery (:math:`t_{eq} \sim` minutes).

Food Products
-------------

Many food materials are thixotropic gels:

- **Mayonnaise**: Thixotropic emulsion, important for dispensing
- **Yogurt**: Structure breakdown affects mouthfeel
- **Ketchup**: Classic yield-stress thixotropic material

Cosmetics and Personal Care
---------------------------

- **Lotions and creams**: Must flow during application, then stop
- **Toothpaste**: Yield stress for tube stability, thixotropy for spreading
- **Hair gel**: Structure recovery for styling hold

Connection to Other Model Families
==================================

Soft Glassy Rheology (SGR)
--------------------------

The SGR model [Sollich1997]_ and DMT model address similar physics from different perspectives:

.. list-table::
   :widths: 25 37 38
   :header-rows: 1

   * - Aspect
     - DMT
     - SGR
   * - State variable
     - Structure :math:`\lambda` (scalar)
     - Trap depth distribution :math:`P(E)`
   * - Kinetics
     - Phenomenological buildup/breakdown
     - Activated hopping with effective temperature
   * - Yield stress
     - Explicit via closure
     - Emergent from trap distribution
   * - Aging
     - :math:`\lambda \to 1` as :math:`t \to \infty`
     - Mean trap depth increases
   * - SAOS
     - Requires Maxwell backbone
     - Natural from trap dynamics

Both models predict similar phenomena (aging, rejuvenation, yield stress) through different mechanisms.

Kinetic Elasto-Viscoplastic (KEVP) Models
-----------------------------------------

The DMT model is a member of the broader KEVP class:

- **Saramito model**: Combines Oldroyd-B viscoelasticity with Herschel-Bulkley plasticity
- **Bautista-Manero-Puig (BMP)**: Structure-dependent Maxwell model
- **Mujumdar model**: Elastic strain with yield function

The DMT model is distinguished by its explicit structure-dependent yield stress and
comprehensive treatment of structure kinetics.

API Reference
=============

DMTLocal
--------

.. autoclass:: rheojax.models.dmt.DMTLocal
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

DMTNonlocal
-----------

.. autoclass:: rheojax.models.dmt.DMTNonlocal
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Parameter Estimation Guidance
=============================

Systematic parameter estimation improves model fits and convergence. Below are
recommended strategies for extracting DMT parameters from different experiments.

From Steady-State Flow Curve
----------------------------

The flow curve :math:`\sigma(\dot{\gamma})` provides several key parameters:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Estimation Method
   * - :math:`\tau_{y0}`
     - Extrapolate low-shear plateau; fit Herschel-Bulkley to low :math:`\dot{\gamma}` data
   * - :math:`K_0, n`
     - Power-law fit to intermediate shear rate region above yield
   * - :math:`\eta_\infty`
     - High-shear slope: :math:`\sigma/\dot{\gamma} \to \eta_\infty` as :math:`\dot{\gamma} \to \infty`
   * - :math:`a, c`
     - Fit equilibrium structure: :math:`\lambda_{eq} = 1/(1 + a|\dot{\gamma}|^c)`

The shape of the shear-thinning transition (log-log curvature) constrains :math:`a` and :math:`c`.

From Transient Tests
--------------------

Startup and cessation experiments provide kinetic and elastic parameters:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Estimation Method
   * - :math:`G_0`
     - Initial slope of startup stress: :math:`\sigma(t \ll t_{eq}) \approx G_0 \cdot \gamma`
   * - :math:`t_{eq}`
     - Recovery time constant from structure rebuild after cessation
   * - :math:`a, c`
     - Overshoot decay rate; rate-dependence of peak strain

The stress overshoot ratio :math:`\sigma_{peak}/\sigma_{ss}` increases with shear rate
and initial structure level.

From SAOS
---------

For the Maxwell variant, the linear viscoelastic moduli provide:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Parameter
     - Estimation Method
   * - :math:`G_0`
     - High-frequency plateau: :math:`G'(\omega \to \infty) \to G_0`
   * - :math:`\theta_0`
     - Crossover frequency: :math:`\omega_c` where :math:`G' = G''`; :math:`\theta_0 = 1/\omega_c`
   * - :math:`\eta_0`
     - Low-frequency loss: :math:`G''(\omega)/\omega \to \eta_0` as :math:`\omega \to 0`

Typical Parameter Ranges
------------------------

.. list-table::
   :widths: 15 25 15 45
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Unit
     - Notes
   * - :math:`\tau_{y0}`
     - 1 - 1000
     - Pa
     - Higher for strongly structured gels
   * - :math:`K_0`
     - 0.1 - 100
     - Pa·s\ :sup:`n`
     - Consistency at full structure
   * - :math:`n`
     - 0.2 - 0.8
     - —
     - Shear-thinning index
   * - :math:`\eta_\infty`
     - 0.001 - 1
     - Pa·s
     - Often close to solvent viscosity
   * - :math:`G_0`
     - 10 - 10000
     - Pa
     - Plateau modulus
   * - :math:`m_1, m_2, m_G`
     - 0.5 - 2.0
     - —
     - Structure exponents (often ~1)
   * - :math:`t_{eq}`
     - 1 - 10000
     - s
     - Longer for strong gels
   * - :math:`a`
     - 0.01 - 100
     - —
     - Breakdown intensity
   * - :math:`c`
     - 0.5 - 2.0
     - —
     - Often ~1 for linear kinetics

Fitting Strategy
----------------

Recommended order for parameter estimation:

1. **Flow curve first**: Fix :math:`\tau_{y0}`, :math:`K_0`, :math:`n`, :math:`\eta_\infty`
   from steady-state data
2. **Add transients**: Fit :math:`t_{eq}`, :math:`a`, :math:`c` from startup/cessation
3. **Add elasticity**: Fit :math:`G_0`, :math:`m_G` from SAOS or overshoot magnitude
4. **Refine jointly**: Use Bayesian inference with all protocols simultaneously

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

**Pipkin Diagram** (Wi-De space):

The nonlinear oscillatory response can be classified using the Pipkin diagram,
which maps behavior in the (strain amplitude, frequency) space:

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Region
     - Conditions
     - Behavior
   * - Linear viscoelastic
     - Low :math:`\gamma_0`, any :math:`\omega`
     - :math:`\lambda \approx 1`, standard :math:`G'`, :math:`G''`
   * - Quasi-steady thixotropic
     - High :math:`\gamma_0`, low :math:`\omega`
     - Structure equilibrates within cycle
   * - Nonlinear viscoelastic
     - High :math:`\gamma_0`, high :math:`\omega`
     - Viscoelastic nonlinearity, limited thixotropy
   * - Thixotropic LAOS
     - Intermediate
     - Complex coupling of all effects

**Intra-cycle structure evolution**:

In LAOS, the structure parameter :math:`\lambda` oscillates within each cycle:

- Near :math:`|\dot{\gamma}| = \gamma_0\omega` (maximum shear rate): Strong breakdown,
  :math:`\lambda` decreases
- Near :math:`\dot{\gamma} = 0` (strain extrema): Structure recovery,
  :math:`\lambda` increases

This intra-cycle variation produces:

- **Strain softening**: :math:`G'_1` decreases with :math:`\gamma_0`
- **Higher harmonics**: :math:`I_{3/1} > 0` from nonlinear structure kinetics
- **Asymmetric Lissajous curves**: Different buildup/breakdown rates
- **Secondary loops**: Elastic recoil combined with thixotropy

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

.. [Coussot2002] Coussot, P., Nguyen, Q. D., Huynh, H. T., & Bonn, D. (2002). "Avalanche
   behavior in yield stress fluids." *Physical Review Letters*, 88(17), 175501.

.. [Mujumdar2002] Mujumdar, A., Beris, A. N., & Metzner, A. B. (2002). "Transient phenomena
   in thixotropic systems." *Journal of Non-Newtonian Fluid Mechanics*, 102(2), 157-178.

.. [Dullaert2006] Dullaert, K., & Mewis, J. (2006). "A structural kinetics model for
   thixotropy." *Journal of Non-Newtonian Fluid Mechanics*, 139(1-2), 21-30.

.. [Larson2019] Larson, R. G., & Wei, Y. (2019). "A review of thixotropy and its
   rheological modeling." *Journal of Rheology*, 63(3), 477-501.

.. [Mendes2013] de Souza Mendes, P. R., & Thompson, R. L. (2013). "A unified approach
   to model elasto-viscoplastic thixotropic yield-stress materials and apparent
   yield-stress fluids." *Rheologica Acta*, 52(7), 673-694.

.. [MewisWagner2009] Mewis, J., & Wagner, N. J. (2009). "Thixotropy." *Advances in
   Colloid and Interface Science*, 147-148, 214-227.

.. [Sollich1997] Sollich, P., Lequeux, F., Hébraud, P., & Cates, M. E. (1997).
   "Rheology of soft glassy materials." *Physical Review Letters*, 78(10), 2020-2023.

.. [SoftMatter2011] de Souza Mendes, P. R. (2011). "Thixotropic elasto-viscoplastic
   model for structured fluids." *Soft Matter*, 7(6), 2471-2483.
