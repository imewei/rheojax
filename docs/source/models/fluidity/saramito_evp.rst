.. _saramito_evp:

================================
Fluidity-Saramito EVP Model
================================

Quick Reference
---------------

- **Use when:** Elastoviscoplastic materials with thixotropy

- **Parameters:** :math:`G`, :math:`\tau_y`, :math:`f`, :math:`\eta_p`, :math:`t_a`, :math:`b`

- **Key equation:** :math:`\boldsymbol{\tau} + \lambda \stackrel{\nabla}{\boldsymbol{\tau}} = 2\eta_p \mathbf{D}`

- **Test modes:** flow_curve, startup, creep, oscillation, LAOS

- **Material examples:** Carbopol, hair gel, mayonnaise

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Class**
     - ``FluiditySaramitoLocal``, ``FluiditySaramitoNonlocal``
   * - **Physics**
     - Elastoviscoplastic with thixotropic fluidity
   * - **Coupling Modes**
     - ``"minimal"``, ``"full"``
   * - **Protocols (Local)**
     - FLOW_CURVE, CREEP, RELAXATION, STARTUP, OSCILLATION, LAOS
   * - **Protocols (Nonlocal)**
     - FLOW_CURVE, CREEP, STARTUP
   * - **Key Features**
     - Tensorial stress, Von Mises yield, normal stresses, shear banding

**Import:**

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal, FluiditySaramitoNonlocal

**Basic Usage:**

.. code-block:: python

   # Minimal coupling (simplest, most identifiable)
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Full coupling (aging yield stress)
   model = FluiditySaramitoLocal(coupling="full")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

Notation Guide
==============

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`\boldsymbol{\tau}`
     - Deviatoric stress tensor
     - Pa
   * - :math:`|\boldsymbol{\tau}|`
     - Von Mises equivalent stress
     - Pa
   * - :math:`\tau_y`
     - Yield stress
     - Pa
   * - :math:`f`
     - Fluidity
     - 1/(Pa·s)
   * - :math:`\lambda`
     - Relaxation time (= 1/f)
     - s
   * - :math:`G`
     - Elastic modulus
     - Pa
   * - :math:`\dot{\gamma}`
     - Shear rate
     - 1/s
   * - :math:`\alpha`
     - Plasticity parameter
     - dimensionless
   * - :math:`\xi`
     - Cooperativity length
     - m

Overview
========

The Fluidity-Saramito Elastoviscoplastic (EVP) model combines three key physical mechanisms:

1. **Viscoelasticity**: Upper-convected Maxwell framework with elastic recoil,
   storage modulus :math:`G'`, and first normal stress difference :math:`N_1`.

2. **Viscoplasticity**: True Von Mises yield surface with Herschel-Bulkley
   plastic flow above yield.

3. **Thixotropy**: Time-dependent aging (structural build-up at rest) and
   shear rejuvenation (flow-induced breakdown) via fluidity evolution.

The model captures complex behaviors including:

- Stress overshoot in startup that increases with waiting time
- Creep bifurcation at the yield stress (bounded vs unbounded flow)
- Non-exponential stress relaxation
- Shear banding in spatially-resolved (nonlocal) variant

Physical Foundations
====================

Upper-Convected Maxwell Framework
---------------------------------

The stress evolution follows the upper-convected Maxwell model with plasticity:

.. math::

   \lambda \overset{\nabla}{\boldsymbol{\tau}} + \alpha(\boldsymbol{\tau})\boldsymbol{\tau} = 2\eta_p \mathbf{D}

where:

- :math:`\lambda = 1/f` is the fluidity-dependent relaxation time
- :math:`\overset{\nabla}{\boldsymbol{\tau}}` is the upper-convected derivative
- :math:`\alpha = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|)` is the Von Mises plasticity
- :math:`\eta_p = G/f` is the polymeric viscosity
- :math:`\mathbf{D}` is the rate of deformation tensor

Von Mises Yield Criterion
-------------------------

The plasticity parameter :math:`\alpha` activates plastic flow only when
the Von Mises equivalent stress exceeds the yield stress:

.. math::

   \alpha = \max\left(0, 1 - \frac{\tau_y}{|\boldsymbol{\tau}|}\right)

where the Von Mises stress is:

.. math::

   |\boldsymbol{\tau}| = \sqrt{\frac{1}{2}\boldsymbol{\tau}:\boldsymbol{\tau}}

Fluidity Evolution
------------------

The fluidity evolves via competing aging and rejuvenation:

.. math::

   \frac{df}{dt} = \frac{f_\text{age} - f}{t_a} + b|\dot{\gamma}|^{n_\text{rej}}(f_\text{flow} - f)

where:

- :math:`f_\text{age}`: Equilibrium fluidity at rest (aged state)
- :math:`f_\text{flow}`: High-shear fluidity limit (rejuvenated state)
- :math:`t_a`: Aging timescale
- :math:`b`: Rejuvenation amplitude
- :math:`n_\text{rej}`: Rejuvenation rate exponent

Coupling Modes
--------------

**Minimal Coupling** (``coupling="minimal"``):

- Relaxation time: :math:`\lambda = 1/f`
- Yield stress: :math:`\tau_y = \tau_{y0}` (constant)
- Fewer parameters, easier to identify

**Full Coupling** (``coupling="full"``):

- Relaxation time: :math:`\lambda = 1/f`
- Yield stress: :math:`\tau_y(f) = \tau_{y0} + a_y/f^m`
- Captures aging yield stress (stronger when aged)

Coupling Architecture Design
----------------------------

The Fluidity-Saramito model offers systematic control over how fluidity
couples to mechanical properties. Understanding these coupling choices is
essential for capturing specific physical behaviors.

**Three Coupling Architectures:**

1. **Minimal Coupling** (``coupling="minimal"``):

   - Only relaxation time depends on fluidity: :math:`\lambda = 1/f`
   - Yield stress constant: :math:`\tau_y = \tau_{y0}`
   - Fewest parameters, most identifiable from data
   - Use when: Standard thixotropic EVP, no aging-dependent yield

2. **Aging Yield Coupling** (``coupling="full"``):

   - Relaxation time: :math:`\lambda = 1/f`
   - Yield stress increases with aging (lower fluidity):

     .. math::

        \tau_y(f) = \tau_{y,\min} + \Delta\tau_y \left(\frac{f_*}{f + f_*}\right)^m

   - Captures wait-time dependent yield stress
   - Use when: Materials that strengthen significantly at rest (waxy crude oils,
     cement, greases)

3. **Dissipation-Consistent Driving** (advanced, not yet implemented):

   - Fluidity evolution driven by plastic dissipation :math:`|\boldsymbol{\tau}:\mathbf{D}|`
     instead of kinematic shear rate :math:`|\dot{\gamma}|`:

     .. math::

        \frac{\partial f}{\partial t} = \frac{f_{\rm eq} - f}{\tau_{\rm age}}
        + b \left(\frac{|\boldsymbol{\tau}:\mathbf{D}|}{\tau_*}\right)^n (f_{\rm flow} - f)

   - Thermodynamically consistent formulation
   - Naturally bounds rejuvenation rate during elastic loading
   - Use when: Strong elastic effects, modeling viscoelastic recoil

**Coupling Design Checklist:**

When selecting coupling options, consider these "knobs" in order of priority:

.. list-table::
   :widths: 15 20 35 30
   :header-rows: 1

   * - Knob
     - Parameter
     - Effect
     - Recommendation
   * - **1.** :math:`\lambda(f)`
     - :math:`\lambda = 1/f`
     - Thixotropic viscosity, stress relaxation rate
     - **Always recommended** — core physics
   * - **2.** :math:`\tau_y(f)`
     - Aging yield coupling
     - Wait-time dependent yield stress
     - Use if yield stress varies with rest time
   * - **3. G(f)**
     - Elastic modulus coupling
     - Structural-dependent stiffness
     - **Use sparingly** — hard to identify, often negligible

**Driving Term Options:**

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Driving Term
     - Equation
     - Physical Interpretation
   * - Kinematic (default)
     - :math:`a|\dot{\gamma}|^n(f_{\rm flow} - f)`
     - Rejuvenation from deformation rate
   * - Energetic/Dissipative
     - :math:`b(|\boldsymbol{\tau}:\mathbf{D}|/\tau_*)^n(f_{\rm flow} - f)`
     - Rejuvenation from plastic work
   * - Plastic-only
     - :math:`b\alpha|\boldsymbol{\tau}:\mathbf{D}|^n(f_{\rm flow} - f)`
     - Only above yield (:math:`\alpha > 0`)

**What the Coupled Model Uniquely Predicts:**

Compared to plain Saramito (no fluidity):

- **History-dependent flow curves**: Same :math:`\dot{\gamma}` gives different
  :math:`\sigma` depending on shear history
- **Waiting-time dependence**: Stress overshoot and yield stress increase with rest time
- **Non-monotonic startup**: Peak stress before steady state
- **Creep bifurcation**: Bounded vs unbounded flow at the yield stress

Compared to plain fluidity (scalar stress):

- **Tensorial stress state**: Full :math:`\boldsymbol{\tau}_{ij}` evolution
- **Normal stress difference** :math:`N_1`: Rod climbing, die swell
- **Realistic elastic loading**: Proper rate-dependent elastic response
- **Von Mises yield criterion**: True 3D yielding behavior

Stress-Driven Fluidity Evolution
--------------------------------

An alternative to kinematic (shear-rate) driving is **stress-driven rejuvenation**,
where fluidity evolution depends on the deviatoric stress magnitude rather than
the deformation rate.

**Stress-Driven Evolution Equation:**

.. math::

   \frac{df}{dt} = \frac{f_{\rm age} - f}{t_a}
   + a \left(\frac{|\boldsymbol{\tau}_d|}{\tau_y}\right)^{n_{\rm rej}} (f_{\rm flow} - f)

where :math:`|\boldsymbol{\tau}_d| = \sqrt{\frac{1}{2}\boldsymbol{\tau}_d:\boldsymbol{\tau}_d}`
is the deviatoric stress magnitude.

**Feedback Loop Dynamics:**

Stress-driven rejuvenation creates a self-regulating feedback loop:

1. **Elastic loading phase**: Stress builds up, :math:`|\boldsymbol{\tau}_d|`
   increases, rejuvenation term activates
2. **Fluidity increase**: Higher :math:`f` means faster stress relaxation
   (:math:`\alpha f \boldsymbol{\tau}` term)
3. **Stress saturation**: Relaxation limits further stress growth,
   which in turn limits rejuvenation
4. **Self-limiting equilibrium**: System naturally finds steady state

This contrasts with kinematic driving where :math:`|\dot{\gamma}|` is externally
imposed and can lead to unbounded rejuvenation during rapid startup.

**When to Use Stress-Driven vs Kinematic:**

.. list-table::
   :widths: 35 35 30
   :header-rows: 1

   * - Driving Type
     - Use When
     - Caution
   * - Kinematic :math:`|\dot{\gamma}|`
     - Standard rheometry, rate-controlled tests
     - May over-predict rejuvenation during elastic regime
   * - Stress :math:`|\boldsymbol{\tau}_d|/\tau_y`
     - Strong elastic effects, stress-controlled tests
     - Requires stress evolution equations
   * - Plastic work :math:`|\boldsymbol{\tau}:\mathbf{D}|`
     - Thermodynamic consistency required
     - Most complex implementation

**Physical Interpretation:**

Stress-driven rejuvenation captures the physics that microstructural breakdown
occurs when the material is under stress, not merely when it deforms. This is
particularly relevant for:

- Materials with significant elastic strain before yielding
- Stress-controlled creep tests
- Understanding the viscosity bifurcation near yield stress
- Materials where elastic recoil is important

Temporal Evolution Regimes
--------------------------

Understanding the transient response of the Fluidity-Saramito model requires
tracking the coupled evolution of stress tensor components and fluidity through
distinct temporal regimes.

**Startup Flow Timeline:**

During startup shear from rest (at constant :math:`\dot{\gamma}`), the material
passes through characteristic stages:

1. **Initial state** (:math:`t = 0`):

   - Fluidity at aged value: :math:`f = f_{\rm aged}` (low, from prior rest)
   - Relaxation time large: :math:`\lambda = 1/f \gg 1`
   - Stress tensor: :math:`\boldsymbol{\tau} = \mathbf{0}`

2. **Elastic loading** (:math:`t < t_{\rm yield}`, typically :math:`\gamma < \gamma_y \sim 0.01-0.1`):

   - Stress builds approximately as: :math:`\tau_{xy} \approx G \cdot \gamma = G \dot{\gamma} t`
   - Von Mises stress: :math:`|\boldsymbol{\tau}| \approx \tau_{xy}` (shear-dominated)
   - Plasticity inactive: :math:`\alpha = 0` (below yield)
   - Fluidity unchanged (no flow → no rejuvenation): :math:`df/dt \approx (f_{\rm age} - f)/t_a \approx 0`

3. **Yield onset** (:math:`|\boldsymbol{\tau}| \to \tau_y`):

   - Plasticity activates: :math:`\alpha = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|) > 0`
   - Stress growth rate slows (relaxation now competes with loading)
   - For full coupling: :math:`\tau_y(f_{\rm aged})` may exceed :math:`\tau_{y0}` significantly

4. **Peak overshoot** (:math:`t = t_{\rm peak}`, typically :math:`\gamma \sim 0.1-1`):

   - Maximum stress: :math:`\sigma_{\rm max} = \tau_y(f_{\rm aged}) + G \gamma_y + \text{viscous contribution}`
   - Balance point: elastic loading rate :math:`\approx` plastic dissipation rate
   - Fluidity beginning to increase (rejuvenation activating)

5. **Rejuvenation-dominated decay** (:math:`t > t_{\rm peak}`):

   - Fluidity increases: :math:`df/dt = b|\dot{\gamma}|^{n_{\rm rej}}(f_{\rm flow} - f) > 0`
   - Relaxation time decreases: :math:`\lambda = 1/f \downarrow`
   - Stress decays toward steady state
   - For full coupling: :math:`\tau_y(f)` also decreases

6. **Steady state** (:math:`t \gg t_a, t_{\rm peak}`):

   - Aging = rejuvenation: :math:`(f_{\rm age} - f)/t_a = b|\dot{\gamma}|^{n_{\rm rej}}(f - f_{\rm flow})`
   - Stress: :math:`\sigma_{ss} = \tau_y(f_{ss}) + K_{\rm HB}\dot{\gamma}^{n_{\rm HB}}`

**Creep Bifurcation Dynamics:**

Under constant applied stress :math:`\sigma_{\rm applied}`:

1. **Sub-yield** (:math:`\sigma_{\rm applied} < \tau_y`):

   - Initial elastic strain: :math:`\gamma_e = \sigma_{\rm applied}/G`
   - Plasticity inactive: :math:`\alpha = 0`
   - No rejuvenation (no flow): fluidity decreases (aging)
   - :math:`f \to f_{\rm age}`, :math:`\tau_y(f) \to \tau_y(f_{\rm age}) > \tau_y(f_{\rm flow})`
   - Material stiffens: bounded strain, solid-like arrest

2. **Above yield** (:math:`\sigma_{\rm applied} > \tau_y`):

   - Initial elastic strain plus viscoplastic flow
   - Plasticity active: :math:`\alpha > 0 \Rightarrow` positive :math:`\dot{\gamma}`
   - **Positive feedback loop**:

     - Flow → rejuvenation → :math:`f \uparrow`
     - :math:`f \uparrow \Rightarrow \lambda \downarrow \Rightarrow` faster flow
     - For full coupling: :math:`f \uparrow \Rightarrow \tau_y \downarrow \Rightarrow` easier yielding

   - Result: avalanche-like transition to steady flow

3. **Near yield** (:math:`\sigma_{\rm applied} \approx \tau_y`):

   - **Delayed yielding**: Long induction time before flow accelerates
   - Metastable: small perturbations determine bounded vs unbounded outcome
   - Duration of plateau: :math:`t_{\rm delay} \sim t_a \cdot \ln(\Delta f / \epsilon)`
     where :math:`\epsilon` is initial departure from equilibrium

**Flow Cessation (Relaxation) Dynamics:**

When shear stops (:math:`\dot{\gamma} \to 0`) from steady flow:

1. **Immediate response** (:math:`t = 0^+`):

   - Rejuvenation term vanishes: :math:`b|\dot{\gamma}|^n(f_{\rm flow} - f) \to 0`
   - Stress locked in current configuration
   - Elastic recoil (partial strain recovery) on timescale :math:`\sim \lambda`

2. **Aging phase** (:math:`t > 0`):

   - Pure aging: :math:`df/dt = (f_{\rm age} - f)/t_a < 0`
   - Fluidity decays: :math:`f(t) = f_{\rm age} + (f_{ss} - f_{\rm age})e^{-t/t_a}`
   - Relaxation time grows: :math:`\lambda(t) = 1/f(t) \uparrow`

3. **Stress relaxation:**

   - Below yield (:math:`|\boldsymbol{\tau}| < \tau_y`): Stress frozen (elastic solid)
   - At/above yield: Slow viscoplastic relaxation on growing timescale
   - Non-exponential: :math:`\tau(t) \sim |\boldsymbol{\tau}(0)| \cdot \exp\left(-\int_0^t \alpha(s) f(s) ds\right)`

4. **Long-time behavior:**

   - :math:`f \to f_{\rm age}` exponentially with time constant :math:`t_a`
   - For full coupling: :math:`\tau_y(f) \to \tau_y(f_{\rm age})` (strengthening)
   - Residual stress relaxes slowly if :math:`|\boldsymbol{\tau}| > \tau_y`

**Regime Summary Diagram:**

::

   Startup:
   ───────────────────────────────────────────────────────────────►
   │     Elastic    │   Yield   │    Peak     │  Rejuvenation  │ SS
   │     loading    │   onset   │   stress    │     decay      │
   f: ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔───────────────────────────── → f_ss
   σ:              ╱╲
                 ╱    ╲─────────────────────────────────────────── → σ_ss
               ╱
            ╱
   ▁▁▁▁▁▁╱
   0    γ_y                                                    t →

Mathematical Formulation
========================

Governing Equations
===================

Component Equations (Simple Shear)
----------------------------------

For simple shear with velocity gradient :math:`\mathbf{L} = \dot{\gamma}\mathbf{e}_x\mathbf{e}_y`:

.. math::

   \frac{d\tau_{xx}}{dt} &= 2\dot{\gamma}\tau_{xy} - \alpha f \tau_{xx} \\
   \frac{d\tau_{yy}}{dt} &= -\alpha f \tau_{yy} \\
   \frac{d\tau_{xy}}{dt} &= \dot{\gamma}\tau_{yy} + G\dot{\gamma} - \alpha f \tau_{xy}

The first normal stress difference is:

.. math::

   N_1 = \tau_{xx} - \tau_{yy}

At steady state in simple shear, this scales as :math:`N_1 \sim \lambda \dot{\gamma} \tau_{xy}`.

Steady-State Flow Curve
-----------------------

At steady state, the model reduces to Herschel-Bulkley form:

.. math::

   \sigma = \tau_y + K_\text{HB}\dot{\gamma}^{n_\text{HB}}

with fluidity-dependent parameters when using full coupling.

----

Validity and Assumptions
=========================

Model Assumptions
-----------------

1. **Tensorial stress description**: Uses full stress tensor :math:`\boldsymbol{\tau}`, suitable for 3D flows
2. **Von Mises yield criterion**: Isotropic yielding based on stress magnitude :math:`|\boldsymbol{\tau}|`
3. **Upper-convected Maxwell framework**: Appropriate for entangled polymers and structured fluids
4. **Thixotropic fluidity evolution**: Single scalar internal variable :math:`f` tracks structure
5. **Isothermal**: Temperature effects not explicitly modeled

Applicability
-------------

The model is most appropriate for:

- **Elastoviscoplastic fluids**: Materials exhibiting elastic recoil, yield stress, and time-dependent viscosity
- **Concentrated emulsions**: Mayonnaise, creams, cosmetics
- **Polymer gels**: Carbopol, hydrogels with yield stress
- **Pastes and slurries**: Cement, drilling muds, waxy crude oils
- **Complex loading**: Protocols requiring tensorial stress (extensional flow, normal stresses)

Limitations
-----------

**No shear banding (local model)**:
   The homogeneous (local) model cannot capture spatial heterogeneity. For shear-banded flows, use the :class:`FluiditySaramitoNonlocal` variant.

**Single structural variable**:
   Real materials may have multiple structural timescales or multi-mode thixotropy. Consider multi-lambda extensions for complex aging.

**Isotropic yielding**:
   The Von Mises criterion assumes isotropic yield surface. Anisotropic materials may require tensorial yield criteria.

**Small strain assumption**:
   The UCM framework assumes affine deformation. Not suitable for materials with wall slip or non-affine microstructure.

----

What You Can Learn
==================

From fitting Fluidity-Saramito EVP to experimental data, you can extract insights about elastoviscoplasticity, thixotropic coupling, and normal stress generation in yield-stress materials.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**f (Fluidity)**:
   Time-dependent inverse relaxation time controlling both viscosity :math:`\eta_p = G/f` and relaxation time :math:`\lambda = 1/f` in tensorial UCM framework.
   *For graduate students*: Unlike scalar fluidity models, here f couples to tensorial stress evolution: :math:`\lambda \cdot \overset{\nabla}{\boldsymbol{\tau}} + \alpha(\boldsymbol{\tau}) \boldsymbol{\tau} = 2\eta_p \mathbf{D}` where :math:`\lambda = 1/f`. Evolution: :math:`df/dt = (f_{\text{age}} - f)/t_a + b|\dot{\gamma}|^{n_{\text{rej}}}(f_{\text{flow}} - f)`. Plasticity :math:`\alpha = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|)` activates only when Von Mises stress :math:`|\boldsymbol{\tau}| = \sqrt{\frac{1}{2}\boldsymbol{\tau}:\boldsymbol{\tau}} > \tau_y`. Normal stresses :math:`N_1 \sim \lambda \dot{\gamma} \tau_{xy} \sim \dot{\gamma}/f^2` (aged materials with low f exhibit larger :math:`N_1`).
   *For practitioners*: :math:`f_{\text{age}} \approx 10^{-6}` to :math:`10^{-3}` s\ :sup:`-1` (solid-like), :math:`f_{\text{flow}} \approx 10^{-2}` to 1 s\ :sup:`-1` (liquid-like). Measure via startup overshoot magnitude (larger overshoot = lower initial :math:`f` after aging).

**G (Elastic Modulus)**:
   Elastic stiffness in UCM backbone, controlling both stress buildup and normal stress generation.
   *For graduate students*: Sets stress scale and Weissenberg number :math:`\text{Wi} = \lambda \dot{\gamma} = \dot{\gamma}/f`. Normal stress scaling: :math:`N_1/\tau_{xy} \sim \text{Wi} \sim \dot{\gamma}/f`. For UCM, :math:`N_2 = 0`. Startup overshoot: :math:`\sigma_{\text{peak}} \sim \tau_y + G\gamma_y` where :math:`\gamma_y` is yield strain.
   *For practitioners*: Extract from initial slope in startup or from :math:`G'` plateau in SAOS. Typical: :math:`G = 10^2\text{--}10^4` Pa (soft colloids), :math:`10^4\text{--}10^6` Pa (polymer gels).

:math:`\tau_y` **(Yield Stress)**:
   Von Mises yield criterion threshold. In minimal coupling, constant. In full coupling: :math:`\tau_y(f) = \tau_{y0} + a_y/f^m` (aged materials are stronger).
   *For graduate students*: Plasticity parameter :math:`\alpha = \max(0, 1 - \tau_y/|\tau|)` controls plastic dissipation term. :math:`\alpha = 0` below yield (elastic), :math:`\alpha > 0` above yield (viscoplastic flow). Full coupling exponent :math:`m` typically 0.3-1.0 captures aging-induced hardening.
   *For practitioners*: Measure from flow curve low-shear plateau or creep bifurcation stress. Full coupling appropriate if yield stress increases significantly after aging (wait-time dependent startup tests).

**t_a, b, n_rej (Fluidity Evolution Parameters)**:
   Control aging (:math:`t_a`) and rejuvenation (:math:`b`, :math:`n_{\text{rej}}`) kinetics analogous to local fluidity model.
   *For graduate students*: :math:`df/dt = (f_{\text{age}} - f)/t_a + b|\dot{\gamma}|^{n_{\text{rej}}}(f_{\text{flow}} - f)`. Characteristic shear rate: :math:`\dot{\gamma}_c \sim (1/(bt_a))^{1/n_{\text{rej}}}`. Stress overshoot magnitude and position depend on :math:`t_a` (waiting time scaling), :math:`b` and :math:`n_{\text{rej}}` (breakdown kinetics).
   *For practitioners*: Extract from startup test family at different wait times. Typical: :math:`t_a = 10\text{--}1000` s, :math:`b = 0.1\text{--}10`, :math:`n_{\text{rej}} = 0.5\text{--}1.5`. Longer :math:`t_a` means more pronounced thixotropy.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Fluidity-Saramito Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - :math:`N_1/\tau_{xy} < 0.1`, minimal coupling
     - Weakly elastic EVP
     - Carbopol gels, pastes
     - Dominant viscoplasticity, minimal normal stresses
   * - :math:`N_1/\tau_{xy} = 0.1\text{--}1`, minimal coupling
     - Moderate elasticity
     - Emulsions, soft colloids, cosmetics
     - Significant rod climbing, moderate die swell
   * - :math:`N_1/\tau_{xy} > 1`, minimal coupling
     - Strongly elastic EVP
     - Polymer gels, fiber suspensions
     - Strong Weissenberg effect, edge fracture risk
   * - Full coupling: m > 0.5
     - Strong aging-yield coupling
     - Waxy crude oils, cement
     - Wait-time dependent yield, restart challenges
   * - :math:`t_a > 100` s, full coupling
     - Extreme thixotropy with elasticity
     - Waxy crude oils, drilling muds
     - Long-term aging, complex restart dynamics

----

Parameters
==========

Core Parameters
---------------

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``G``
     - Elastic modulus
     - Pa
     - 1e4
     - [1e1, 1e8]
   * - ``tau_y0``
     - Base yield stress
     - Pa
     - 100
     - [0.1, 1e5]
   * - ``K_HB``
     - HB consistency index
     - Pa·s^n
     - 50
     - [1e-2, 1e5]
   * - ``n_HB``
     - HB flow exponent
     - —
     - 0.5
     - [0.1, 1.5]

Fluidity Parameters
-------------------

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``f_age``
     - Aging fluidity limit
     - 1/(Pa·s)
     - 1e-6
     - [1e-12, 1e-2]
   * - ``f_flow``
     - Flow fluidity limit
     - 1/(Pa·s)
     - 1e-2
     - [1e-6, 1.0]
   * - ``t_a``
     - Aging timescale
     - s
     - 10
     - [0.01, 1e5]
   * - ``b``
     - Rejuvenation amplitude
     - —
     - 1.0
     - [0, 1e3]
   * - ``n_rej``
     - Rejuvenation exponent
     - —
     - 1.0
     - [0.1, 3.0]

Full Coupling Parameters
------------------------

Only active when ``coupling="full"``:

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``tau_y_coupling``
     - Yield stress coupling
     - Pa·(Pa·s)^m
     - 1.0
     - [0, 1e4]
   * - ``m_yield``
     - Yield stress exponent
     - —
     - 0.5
     - [0.1, 2.0]

Nonlocal Parameters
-------------------

Only for ``FluiditySaramitoNonlocal``:

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``xi``
     - Cooperativity length
     - m
     - 1e-5
     - [1e-7, 1e-2]

Parameter Interpretation by Material
------------------------------------

**Concentrated Emulsions** (mayonnaise, cosmetics):

- :math:`\tau_y \sim 10-100` Pa
- :math:`n_\text{HB} \sim 0.3-0.5`
- :math:`t_a \sim 10-100` s

**Polymer Gels** (carbopol, hydrogels):

- :math:`\tau_y \sim 1-50` Pa
- :math:`n_\text{HB} \sim 0.4-0.6`
- :math:`t_a \sim 1-1000` s (depends on concentration)

**Cement/Concrete**:

- :math:`\tau_y \sim 100-1000` Pa
- :math:`n_\text{HB} \sim 0.2-0.4`
- :math:`t_a \sim 100-10000` s (hydration-dependent)

**Drilling Muds**:

- :math:`\tau_y \sim 5-50` Pa
- :math:`n_\text{HB} \sim 0.3-0.7`
- :math:`t_a \sim 10-1000` s

----

Usage
=====

(Usage examples already present in the file)

----

See Also
========

- :doc:`fluidity_local` — Scalar fluidity model without tensorial stress
- :doc:`fluidity_nonlocal` — Nonlocal model for shear banding
- :doc:`../dmt/dmt_local` — DMT structural-kinetics thixotropic model
- :doc:`../flow/herschel_bulkley` — Simpler yield stress model (steady shear only)

----

Fitting Guidance
================

Recommended Workflow
--------------------

1. **Start with flow curve** to get :math:`\tau_y`, :math:`K_{\text{HB}}`, :math:`n_{\text{HB}}`
2. **Add startup** to get :math:`G` and fluidity dynamics (:math:`t_a`, :math:`b`, :math:`n_{\text{rej}}`)
3. **Use creep** to validate :math:`\tau_y` (bifurcation point)
4. **Optionally use LAOS** for nonlinear validation

Step-by-Step Example
--------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np

   # 1. Flow curve fitting
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # 2. Startup fitting (refines G and fluidity parameters)
   model.fit(t, sigma_startup, test_mode='startup', gamma_dot=1.0)

   # 3. Bayesian inference for uncertainty
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
   )

   # 4. Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   for param, stats in intervals.items():
       print(f"{param}: {stats['mean']:.2f} [{stats['lower']:.2f}, {stats['upper']:.2f}]")

Troubleshooting
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Issue
     - Solution
   * - No stress overshoot
     - Increase ``b`` (rejuvenation) or decrease ``t_a`` (aging time)
   * - Overshoot too large
     - Decrease ``b`` or increase ``f_age``
   * - Flow curve too flat
     - Decrease ``n_HB`` (more shear-thinning)
   * - Poor creep fit
     - Check ``tau_y0`` against bifurcation point in data
   * - Bayesian divergences
     - Use NLSQ warm-start, increase ``num_warmup``

Parameter Estimation from Multiple Protocols
--------------------------------------------

For reliable parameter estimation in the Fluidity-Saramito model, a sequential
multi-protocol approach is essential. Each protocol isolates different parameters,
enabling progressive refinement.

**Sequential Fitting Workflow:**

.. list-table::
   :widths: 8 20 35 37
   :header-rows: 1

   * - Step
     - Protocol
     - Parameters Constrained
     - Notes
   * - 1
     - SAOS (small amplitude oscillation)
     - :math:`G` from :math:`G'` plateau; initial :math:`f` from :math:`\tan\delta`
     - Linear regime only. :math:`G = G'(\omega \to \infty)` or plateau
   * - 2
     - Flow curve (steady shear)
     - :math:`\tau_{y0}`, :math:`K_{\rm HB}`, :math:`n_{\rm HB}`
     - Fit Herschel-Bulkley. Fixes steady-state rheology
   * - 3
     - Startup tests (multiple :math:`\dot{\gamma}`)
     - :math:`t_a` (from peak timing), :math:`b`, :math:`n_{\rm rej}` (from peak magnitude)
     - Use waiting time series. Overshoot position → kinetics
   * - 4
     - Creep tests (:math:`\sigma` near :math:`\tau_y`)
     - Validate :math:`\tau_{y0}` from bifurcation point
     - :math:`\sigma < \tau_y`: bounded; :math:`\sigma > \tau_y`: unbounded. Confirms yield

**Step 1: SAOS Analysis**

Extract elastic modulus :math:`G` and estimate initial fluidity:

.. code-block:: python

   # From SAOS data (G', G'' vs omega)
   G_prime_plateau = np.max(G_prime)  # High-frequency plateau
   G = G_prime_plateau

   # Estimate fluidity from loss tangent at intermediate frequency
   # tan(δ) = G''/G' ≈ 1/(ωλ) = f·ω for Maxwell-like response
   omega_mid = omega[len(omega)//2]
   tan_delta_mid = G_double_prime[len(omega)//2] / G_prime[len(omega)//2]
   f_initial_estimate = tan_delta_mid * omega_mid

   print(f"Estimated G: {G:.1f} Pa")
   print(f"Estimated initial f: {f_initial_estimate:.2e} 1/(Pa·s)")

**Step 2: Flow Curve Fitting**

Fix HB steady-state parameters:

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal

   model = FluiditySaramitoLocal(coupling="minimal")

   # Fix G from SAOS, fit HB parameters only
   model.parameters.set_value("G", G, fixed=True)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   tau_y0 = model.parameters.get_value("tau_y0")
   K_HB = model.parameters.get_value("K_HB")
   n_HB = model.parameters.get_value("n_HB")

**Step 3: Startup Test Analysis**

Extract kinetic parameters from stress overshoot:

.. code-block:: python

   # Startup at multiple waiting times
   wait_times = [1, 10, 100, 1000]  # seconds
   gamma_dot_startup = 1.0

   overshoot_data = []
   for t_wait in wait_times:
       _, stress, _ = model.simulate_startup(t, gamma_dot_startup, t_wait=t_wait)
       sigma_max = np.max(stress)
       t_peak = t[np.argmax(stress)]
       sigma_ss = stress[-1]
       overshoot_data.append({
           't_wait': t_wait,
           'sigma_max': sigma_max,
           't_peak': t_peak,
           'overshoot_ratio': sigma_max / sigma_ss
       })

   # t_a controls: log(sigma_max) vs log(t_wait) slope
   # b controls: overall overshoot magnitude
   # n_rej controls: dependence on shear rate

**Parameter Sensitivity Guide:**

.. list-table::
   :widths: 18 40 42
   :header-rows: 1

   * - Parameter
     - Most Sensitive Observables
     - Identifiability Notes
   * - :math:`G`
     - Initial slope in startup; :math:`G'` plateau in SAOS
     - Well-identified from SAOS or early startup
   * - :math:`\tau_{y0}`
     - Flow curve low-shear limit; creep bifurcation
     - Requires low :math:`\dot{\gamma}` data (< 0.01 s\ :sup:`-1`)
   * - :math:`K_{\rm HB}`, :math:`n_{\rm HB}`
     - Flow curve shape
     - Often correlated; fix one if uncertain
   * - :math:`t_a`
     - Overshoot position vs wait time; relaxation timescale
     - Best from wait-time series
   * - :math:`b`
     - Overshoot magnitude; rejuvenation rate
     - Correlated with :math:`n_{\rm rej}`. Fix one first
   * - :math:`n_{\rm rej}`
     - Shear-rate dependence of rejuvenation
     - Often fixed at 1.0 initially
   * - :math:`f_{\rm age}`, :math:`f_{\rm flow}`
     - Aged vs flowing viscosity limits
     - Use extreme conditions (long rest, high shear)

**Spatially-Resolved Data (Nonlocal Variant):**

For the nonlocal model, velocity profiles provide direct access to :math:`\xi`:

1. **Velocity profile measurement**: Ultrasound velocimetry (USV), MRI, or PIV
2. **Shear rate extraction**: :math:`\dot{\gamma}(y) = dv/dy`
3. **Fluidity profile**: :math:`f(y) = \dot{\gamma}(y)/\sigma` (uniform stress)
4. :math:`\xi` **extraction**: Fit boundary layer decay of :math:`f(y)` from walls

.. code-block:: python

   # From velocity profile data: v(y) and known stress Sigma
   gamma_dot_profile = np.gradient(v_profile, y_grid)
   f_profile = gamma_dot_profile / Sigma

   # Fit exponential boundary layer: f(y) = f_bulk + (f_wall - f_bulk)*exp(-y/xi)
   from scipy.optimize import curve_fit

   def boundary_layer(y, f_bulk, f_wall, xi):
       return f_bulk + (f_wall - f_bulk) * np.exp(-y / xi)

   # Fit near wall (y < H/4)
   mask = y_grid < H / 4
   popt, _ = curve_fit(boundary_layer, y_grid[mask], f_profile[mask],
                       p0=[f_profile[-1], f_profile[0], 50e-6])
   xi_extracted = popt[2]

Usage Examples
==============

Basic Flow Curve Fitting
------------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np

   # Generate synthetic data
   gamma_dot = np.logspace(-2, 2, 30)
   sigma_data = 100 + 50 * gamma_dot**0.5  # HB-like

   # Fit model
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma_data, test_mode='flow_curve')

   # Predict
   sigma_pred = model.predict(gamma_dot)

   # Get yield stress
   tau_y = model.parameters.get_value("tau_y0")
   print(f"Fitted yield stress: {tau_y:.1f} Pa")

Startup with Stress Overshoot
-----------------------------

.. code-block:: python

   # Simulate startup
   t = np.linspace(0, 50, 500)
   gamma_dot = 1.0
   t_wait = 100.0  # Waiting time before startup

   strain, stress, fluidity = model.simulate_startup(t, gamma_dot, t_wait=t_wait)

   # Analyze overshoot
   sigma_max = np.max(stress)
   sigma_ss = stress[-1]
   overshoot_ratio = sigma_max / sigma_ss
   print(f"Overshoot ratio: {overshoot_ratio:.2f}")

Creep Bifurcation
-----------------

.. code-block:: python

   t = np.linspace(0, 1000, 500)

   # Below yield - bounded strain
   strain_below, _ = model.simulate_creep(t, sigma_applied=50.0)

   # Above yield - unbounded flow
   strain_above, _ = model.simulate_creep(t, sigma_applied=150.0)

   # Plot shows bifurcation behavior

Normal Stress Predictions
-------------------------

The Fluidity-Saramito model tracks tensorial stress components
(:math:`\tau_{xx}`, :math:`\tau_{yy}`, :math:`\tau_{xy}`) internally via
the upper-convected Maxwell constitutive equation, enabling first normal
stress difference :math:`N_1 = \tau_{xx} - \tau_{yy}` to emerge naturally
during transient simulations (startup, LAOS). No public
``predict_normal_stresses()`` method is currently exposed; instead, access
normal stress data through the tensorial stress output of
``simulate_startup()`` or ``simulate_laos()``.

LAOS Analysis
-------------

Large amplitude oscillatory shear (LAOS) provides a fingerprint of nonlinear
material behavior through Lissajous curve shapes and harmonic decomposition.

**Lissajous Curve Interpretation:**

The Fluidity-Saramito model produces characteristic Lissajous shapes that reveal
the interplay between elasticity, plasticity, and thixotropy:

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Projection
     - Shape Feature
     - Physical Interpretation
   * - **Elastic** (:math:`\sigma` vs :math:`\gamma`)
     - Parallelogram deviation
     - Yielding, plastic flow above :math:`\tau_y`
   * - **Elastic** (:math:`\sigma` vs :math:`\gamma`)
     - Enclosed area
     - Energy dissipation per cycle
   * - **Viscous** (:math:`\sigma` vs :math:`\dot{\gamma}`)
     - Bow-tie/self-intersection
     - Thixotropy, structure kinetics
   * - **Viscous** (:math:`\sigma` vs :math:`\dot{\gamma}`)
     - Asymmetric loops
     - Non-equilibrium fluidity dynamics

**Elastic Projection (** :math:`\sigma` **vs** :math:`\gamma` **):**

- **Linear viscoelastic**: Ellipse with slope :math:`G'` and area :math:`\propto G''`
- **With yielding**: Parallelogram-like corners where :math:`|\tau|` crosses :math:`\tau_y`
- **With thixotropy**: Cycle-to-cycle evolution as fluidity equilibrates
- **Strain softening**: Counterclockwise deviation (intracycle) indicates structure breakdown

**Viscous Projection (** :math:`\sigma` **vs** :math:`\dot{\gamma}` **):**

- **Linear viscoelastic**: Ellipse with slope :math:`\eta'` and area :math:`\propto \eta''`
- **With thixotropy**: Self-intersecting "bow-tie" or "figure-8" patterns
- **Interpretation**: Self-intersection occurs when fluidity lags the deformation,
  causing stress to increase/decrease non-monotonically during each quarter cycle
- **Secondary loops**: Indicate multiple structural relaxation timescales

**Thixotropic Loop Signatures:**

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Pattern
     - Cause
     - Parameter Sensitivity
   * - Self-intersecting viscous loops
     - :math:`f` lags :math:`\dot{\gamma}` by :math:`\pi/2`
     - :math:`\tau_{\text{age}}` vs :math:`1/\omega` ratio
   * - Cycle evolution (softening)
     - :math:`f` not at steady state
     - Number of pre-cycles needed
   * - Asymmetric peaks
     - Non-symmetric :math:`f(t)` waveform
     - :math:`n_{\text{rej}} \neq 1`
   * - Harmonic distortion
     - Nonlinear plasticity
     - :math:`\gamma_0/\gamma_y` ratio

**Harmonic Extraction:**

Fourier decomposition of the stress waveform provides quantitative nonlinearity measures:

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np

   model = FluiditySaramitoLocal(coupling="minimal")
   gamma_0 = 1.0  # 100% strain amplitude
   omega = 1.0    # rad/s

   # Simulate LAOS with sufficient cycles for steady state
   t, strain, stress = model.simulate_laos(gamma_0, omega, n_cycles=10, n_points_per_cycle=100)

   # Extract harmonics from last cycle (steady state)
   last_cycle_idx = -100  # Last 100 points = 1 cycle
   stress_cycle = stress[last_cycle_idx:]

   # FFT for harmonic amplitudes
   fft_stress = np.fft.fft(stress_cycle)
   harmonics = np.abs(fft_stress[:5]) / len(stress_cycle) * 2

   print(f"1st harmonic (G*): {harmonics[1]:.1f} Pa")
   print(f"3rd harmonic (I_3): {harmonics[3]:.1f} Pa")
   print(f"I_3/I_1 ratio: {harmonics[3]/harmonics[1]:.4f}")

   # Chebyshev coefficients for LAOS analysis
   # e_1 (strain-stiffening) and v_1 (shear-thickening)
   from numpy.polynomial import chebyshev
   gamma_norm = strain[last_cycle_idx:] / gamma_0
   cheb_coeffs = chebyshev.chebfit(gamma_norm, stress_cycle, deg=5)

**Standard LAOS Example:**

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np
   import matplotlib.pyplot as plt

   # Large amplitude oscillatory shear
   model = FluiditySaramitoLocal(coupling="minimal")

   gamma_0 = 1.0  # 100% strain
   omega = 1.0    # rad/s

   # Simulate LAOS
   t, strain, stress = model.simulate_laos(gamma_0, omega, n_cycles=3)

   # Plot Lissajous curves
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   # Elastic projection (σ vs γ)
   axes[0].plot(strain, stress)
   axes[0].set_xlabel('Strain γ')
   axes[0].set_ylabel('Stress σ (Pa)')
   axes[0].set_title('Elastic Lissajous (σ vs γ)')
   axes[0].grid(True)

   # Viscous projection (σ vs γ̇)
   gamma_dot = gamma_0 * omega * np.cos(omega * t)
   axes[1].plot(gamma_dot, stress)
   axes[1].set_xlabel('Strain rate γ̇ (1/s)')
   axes[1].set_ylabel('Stress σ (Pa)')
   axes[1].set_title('Viscous Lissajous (σ vs γ̇)')
   axes[1].grid(True)

   plt.tight_layout()

Nonlocal Model with Shear Banding
---------------------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoNonlocal

   # Create nonlocal model
   model = FluiditySaramitoNonlocal(
       coupling="minimal",
       N_y=51,      # Grid points
       H=1e-3,      # Gap width (m)
       xi=1e-5,     # Cooperativity length (m)
   )

   # Simulate startup
   t = np.linspace(0, 50, 200)
   _, sigma, f_field = model.simulate_startup(t, gamma_dot=0.1)

   # Check for shear banding
   is_banded, cv, ratio = model.detect_shear_bands()
   print(f"Shear banding: {is_banded}, CV={cv:.2f}, ratio={ratio:.1f}")

   # Get detailed metrics
   metrics = model.get_banding_metrics()
   print(f"Band fraction: {metrics['band_fraction']:.2f}")

Bayesian Inference
------------------

.. code-block:: python

   # Fit with NLSQ first (warm-start)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Bayesian inference
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,  # Production-ready diagnostics
       seed=42,       # Reproducibility
   )

   # Check diagnostics
   # R-hat should be < 1.01, ESS > 400

   # Plot with ArviZ
   from rheojax.pipeline.bayesian import BayesianPipeline

   pipeline = BayesianPipeline()
   pipeline._idata = result.idata
   pipeline.plot_trace()
   pipeline.plot_pair(divergences=True)

Comparison with Existing Models
===============================

vs FluidityLocal
----------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - FluidityLocal
     - FluiditySaramitoLocal
   * - Stress tensor
     - Scalar :math:`\sigma`
     - Tensorial [:math:`\tau_{xx}`, :math:`\tau_{yy}`, :math:`\tau_{xy}`]
   * - Normal stresses
     - No
     - Yes (:math:`N_1` from UCM)
   * - Yield criterion
     - None (implicit)
     - Von Mises (explicit)
   * - Elastic effects
     - Maxwell-like
     - Upper-convected Maxwell
   * - Parameters
     - 9
     - 10-12

The Saramito model is preferred when:

- Normal stresses (:math:`N_1`) are important
- Tensorial stress state is needed
- True yield criterion is required

vs Standard Saramito (no fluidity)
----------------------------------

The fluidity extension adds:

- Thixotropic time dependence
- Aging/rejuvenation dynamics
- Shear banding capability (nonlocal)

Without fluidity, the standard Saramito model has constant relaxation time
and cannot capture thixotropic behaviors.

vs Other Yield-Stress and Thixotropic Models
--------------------------------------------

The Fluidity-Saramito model occupies a unique position in the landscape of
yield-stress and thixotropic constitutive models. This comprehensive comparison
helps practitioners select the appropriate model.

**Feature Comparison Table:**

.. list-table::
   :widths: 22 16 16 14 16 16
   :header-rows: 1

   * - Feature
     - Fluidity-Saramito
     - STZ
     - HL Trap
     - Giesekus
     - DMT
   * - **Yield stress**
     - Explicit (:math:`\tau_y`)
     - Emergent
     - Emergent
     - None
     - Explicit
   * - **Aging mechanism**
     - Explicit (:math:`f` kinetics)
     - Via :math:`\chi` (effective temp)
     - Via :math:`n(E)` trap distribution
     - None
     - Via :math:`\lambda` kinetics
   * - **Stress tensor**
     - Full tensorial
     - Scalar (typically)
     - Scalar
     - Full tensorial
     - Scalar
   * - **Shear banding**
     - With :math:`D_f` (nonlocal)
     - Needs spatial extension
     - Needs spatial extension
     - No
     - With nonlocal extension
   * - **Normal stresses**
     - Yes (:math:`N_1`)
     - Limited
     - No
     - Yes (:math:`N_1, N_2`)
     - No
   * - **# Parameters**
     - 10-12
     - ~7
     - ~7
     - ~4
     - ~8
   * - **Physical basis**
     - Phenomenological + UCM
     - Statistical mechanics
     - Mean-field kinetic
     - Molecular network
     - Structural kinetics

**Model Selection Guide:**

**When to use Fluidity-Saramito:**

- **Tensorial stress required**: Complex flow geometries, extensional components,
  normal stress predictions for rod climbing, die swell
- **Explicit yield criterion**: Need to track Von Mises stress vs :math:`\tau_y`
- **Clear separation of effects**: Distinct elastic, plastic, and structural
  contributions identifiable from parameters
- **Dense colloids with both aging and elasticity**: Emulsions, polymer gels,
  cosmetics where both :math:`G` and thixotropy matter
- **Validation against velocimetry**: Nonlocal variant directly predicts
  :math:`f(y)` profiles comparable to PIV/USV data

**When to use alternatives:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Model
     - Use When
   * - **STZ**
     - Metallic glasses, granular media; need statistical mechanics foundation;
       yield emerges from disorder physics
   * - **HL Trap**
     - Colloidal glasses near jamming; energy landscape interpretation;
       distribution of relaxation times
   * - **Giesekus**
     - Polymer melts/solutions without yield; need :math:`N_2` prediction;
       fewer parameters, no thixotropy
   * - **DMT**
     - Similar physics to Fluidity-Saramito but with scalar stress and
       structure parameter :math:`\lambda \in [0,1]`; simpler tensorial extension

**Complexity vs. Capability Trade-off:**

::

   Capability (features modeled)
         ▲
         │                                    ┌─────────────────────┐
         │                                    │ Fluidity-Saramito   │
         │                              ┌─────┤ Nonlocal            │
         │                              │     └─────────────────────┘
         │                    ┌─────────┴───┐
         │                    │ Fluidity-   │
         │                    │ Saramito    │
         │          ┌─────────┤ Local       │
         │          │         └─────────────┘
         │    ┌─────┴────┐     ┌──────┐
         │    │ DMT/     │     │ STZ  │
         │    │ Fluidity │     └──────┘
         │    │ Local    │  ┌───────┐
         │    └──────────┘  │ HL    │
         │                  └───────┘
         │  ┌──────────┐
         │  │ HB/      │
         │  │ Bingham  │
         │  └──────────┘
         └───────────────────────────────────────────────────────────► Complexity
                                                               (# parameters)

References
==========

.. [1] Saramito, P. "A new constitutive equation for elastoviscoplastic
   fluid flows." *Journal of Non-Newtonian Fluid Mechanics*, 145, 1-14 (2007).
   https://doi.org/10.1016/j.jnnfm.2006.04.012

.. [2] Saramito, P. "A new elastoviscoplastic model based on the
   Herschel-Bulkley viscoplastic model." *Journal of Non-Newtonian Fluid Mechanics*,
   158, 154-161 (2009). https://doi.org/10.1016/j.jnnfm.2008.12.001

.. [3] Coussot, P., Nguyen, Q. D., Huynh, H. T., and Bonn, D. "Viscosity bifurcation
   in thixotropic, yielding fluids." *Journal of Rheology*, 46(3), 573-589 (2002).
   https://doi.org/10.1122/1.1459447

.. [4] Bocquet, L., Colin, A., and Ajdari, A. "Kinetic theory of plastic flow
   in soft glassy materials." *Physical Review Letters*, 103, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

.. [5] Ovarlez, G., Mahaut, F., Deboeuf, S., Lenoir, N., Hormozi, S., and Chateau, X.
   "Phenomenology and physical origin of shear-localization and shear banding
   in complex fluids." *Journal of Non-Newtonian Fluid Mechanics*, 177-178, 19-28 (2012).
   https://doi.org/10.1016/j.jnnfm.2012.03.011
.. [6] Oldroyd, J. G. "On the formulation of rheological equations of state."
   *Proceedings of the Royal Society A*, **200**, 523-541 (1950).
   https://doi.org/10.1098/rspa.1950.0035

.. [7] Dimitriou, C. J., Casanellas, L., Ober, T. J., & McKinley, G. H. "RheoJAX-PIV of a shear-banding wormlike micellar solution under large amplitude oscillatory shear."
   *Rheologica Acta*, **51**, 395-411 (2012).
   https://doi.org/10.1007/s00397-012-0619-9

.. [8] Fraggedakis, D., Dimakopoulos, Y., & Tsamopoulos, J. "Yielding the yield stress analysis: A thorough comparison of recently proposed elasto-visco-plastic (EVP) fluid models."
   *Journal of Non-Newtonian Fluid Mechanics*, **238**, 170-188 (2016).
   https://doi.org/10.1016/j.jnnfm.2016.11.007

.. [9] de Souza Mendes, P. R. & Thompson, R. L. "A critical overview of elasto-viscoplastic thixotropic modeling."
   *Journal of Non-Newtonian Fluid Mechanics*, **187-188**, 8-15 (2012).
   https://doi.org/10.1016/j.jnnfm.2012.08.006

.. [10] Balmforth, N. J., Frigaard, I. A., & Ovarlez, G. "Yielding to stress: Recent developments in viscoplastic fluid mechanics."
   *Annual Review of Fluid Mechanics*, **46**, 121-146 (2014).
   https://doi.org/10.1146/annurev-fluid-010313-141424


API Reference
=============

.. autoclass:: rheojax.models.fluidity.saramito.FluiditySaramitoLocal
   :members:
   :inherited-members:
   :no-index:

.. autoclass:: rheojax.models.fluidity.saramito.FluiditySaramitoNonlocal
   :members:
   :inherited-members:
   :no-index:
