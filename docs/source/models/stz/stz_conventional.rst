Shear Transformation Zone (STZ)
===============================

Quick Reference
---------------

- **Use when:** Amorphous solids, metallic glasses, colloidal suspensions near jamming, emulsions, granular matter

- **Parameters:** 10 (:math:`G_0, \sigma_y, \chi_{\infty}, \tau_0, \varepsilon_0, c_0, e_Z, \tau_\beta, m_{\infty}, \Gamma_m`)

- **Key equation:** :math:`\dot{\varepsilon}^{pl} = \frac{\varepsilon_0}{\tau_0} \Lambda(\chi) \mathcal{C}(s) \mathcal{T}(s)`

- **Test modes:** flow_curve (steady_shear), startup, relaxation, creep, oscillation (LAOS)

- **Material examples:** Metallic glasses, colloidal glasses, dense emulsions, granular matter

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\chi`
     - Effective temperature (configurational disorder parameter)
   * - :math:`\Lambda(\chi)`
     - STZ density, :math:`\Lambda = \exp(-1/\chi)`
   * - :math:`s`
     - Deviatoric stress (shear stress)
   * - :math:`\sigma_y`
     - Yield stress scale (activation barrier height)
   * - :math:`\dot{\varepsilon}^{pl}`
     - Plastic strain rate (from STZ flips)
   * - :math:`\varepsilon_0`
     - Strain increment per STZ rearrangement (typically 0.1-0.3)
   * - :math:`\tau_0`
     - Molecular attempt time (vibration timescale)
   * - :math:`\mathcal{C}(s)`
     - Rate factor (activation), :math:`\cosh(s/\sigma_y)^q`
   * - :math:`\mathcal{T}(s)`
     - Transition bias, :math:`\tanh(s/\sigma_y)`
   * - :math:`c_0`
     - Effective specific heat (controls rate of :math:`\chi` evolution)
   * - :math:`\chi_\infty`
     - Steady-state effective temperature at high drive
   * - :math:`m`
     - Orientational bias (kinematic hardening, Full variant only)
   * - :math:`e_Z`
     - STZ formation energy (normalized by :math:`k_B T_g`)
   * - :math:`\tau_\beta`
     - Relaxation timescale for STZ density

Overview
--------

The Shear Transformation Zone (STZ) theory provides a physical description of plastic deformation in amorphous materials such as metallic glasses, colloidal suspensions, emulsions, and granular matter. Unlike crystalline materials where plasticity is mediated by dislocations, amorphous solids deform through localized rearrangements of particle clusters known as Shear Transformation Zones.

Historical Development
----------------------

The STZ theory emerged from decades of research on plasticity in disordered materials,
building upon foundational concepts in glass physics and amorphous plasticity.

Origins and Key Contributors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Free Volume Theory (1959)**
   Cohen and Turnbull [Cohen1959]_ proposed that molecular mobility in liquids and glasses
   is controlled by the availability of local "free volume"—excess space beyond dense
   packing that allows molecules to rearrange. They introduced the Boltzmann-like factor
   :math:`\exp(-\text{const}/v_f)` for transition rates, where :math:`v_f` is the free
   volume per particle.

**Flow Defects (1977)**
   Spaepen [7]_ identified localized "flow defects" as the carriers of plastic deformation
   in metallic glasses. These regions of anomalous local structure could undergo shear
   transformations under stress, producing irreversible strain.

**Shear Transformation Zones (1979)**
   Argon [6]_ introduced the term "shear transformation zones" and developed a
   quantitative model for plastic flow in metallic glasses. He proposed that STZs are
   small clusters (~5-10 atoms) that can flip between two stable configurations,
   producing a local shear strain increment :math:`\varepsilon_0 \approx 0.1`.

**Two-State STZ Theory (1998)**
   Falk and Langer [2]_ formalized the STZ concept using molecular dynamics simulations
   of a 2D Lennard-Jones glass. Their key innovations:

   - **Two-state model**: STZs exist in "+" or "−" orientations, allowing directional memory
   - **Jamming**: Once transformed, a STZ cannot transform again in the same direction
   - **Creation/annihilation**: STZs are ephemeral, created and destroyed during plastic work
   - **Rate factor**: Transition rates depend on the strain rate, not just temperature

**Thermodynamic Constraints (2003)**
   Langer and Pechenik [Langer2003]_ used energy balance arguments to derive the form of
   the STZ creation/annihilation rate :math:`\Gamma`. They showed that requiring
   non-negative dissipation (second law) uniquely determines the coupling between
   stress and STZ dynamics.

**Effective Temperature Reformulation (2008)**
   Langer [1]_ introduced the effective temperature :math:`\chi = T_{\text{eff}}/T_Z`
   as the fundamental state variable controlling STZ density via :math:`\Lambda = e^{-1/\chi}`.
   This replaced earlier free-volume formulations and provided a clearer connection
   to nonequilibrium thermodynamics.

Molecular Dynamics Validation
-----------------------------

The two-state STZ model was validated by Falk and Langer [2]_ using molecular dynamics
simulations of a model 2D glass. Their simulations provided direct evidence for localized
plastic rearrangements and quantitative comparison with theoretical predictions.

Simulation Setup
~~~~~~~~~~~~~~~~

**System composition:**
   A 2D binary mixture of soft disks with size ratio 1:1.4, designed to suppress
   crystallization. Systems contained 10,000 to 20,000 particles at number density
   :math:`\rho = 0.85`.

**Interaction potential:**
   Lennard-Jones 6-12 potential with cutoff :math:`r_c = 2.5\sigma`:

   .. math::

      V(r) = 4\varepsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6} \right]

   where :math:`\varepsilon` sets the energy scale and :math:`\sigma` the length scale.
   All quantities are reported in reduced units (:math:`\varepsilon = \sigma = m = 1`).

**Glass preparation:**
   Samples were prepared by rapid quenching from high temperature (:math:`T = 1.0`) to
   low temperature (:math:`T = 0.001 T_0`) at constant density. This produces a
   disordered, kinetically arrested state.

**Deformation protocol:**
   Simple shear at constant strain rate :math:`\dot{\gamma}`, with Lees-Edwards
   periodic boundary conditions. Strain rates ranged from :math:`10^{-4}` to
   :math:`10^{-2}` per unit time.

The :math:`D^2_min` Diagnostic
~~~~~~~~~~~~~~~~~~~~~

Falk and Langer introduced the :math:`D^2_{\min}` diagnostic to identify nonaffine
deformations—particle motions that deviate from homogeneous shear. For each particle
:math:`i`, the local affine strain tensor :math:`\mathbf{J}_i` is computed by minimizing:

.. math::

   D^2_i = \frac{1}{N_i} \sum_{j \in \text{neighbors}} \left| \mathbf{d}_j(t) - \mathbf{J}_i \cdot \mathbf{d}_j(0) \right|^2

where :math:`\mathbf{d}_j(t)` is the displacement of neighbor :math:`j` relative to
particle :math:`i` at time :math:`t`. High values of :math:`D^2_{\min}` identify
particles undergoing plastic rearrangements.

**Key findings from** :math:`D^2_min` **analysis:**

1. **Localization**: Plastic activity is concentrated in localized regions (~5-10
   particles), validating the STZ concept
2. **Two-state behavior**: Regions flip between configurations, consistent with
   the ± orientation model
3. **Transient character**: Active zones are ephemeral, appearing and disappearing
   during plastic flow

Theory vs. MD Comparison
~~~~~~~~~~~~~~~~~~~~~~~~

The MD simulations provided quantitative validation of STZ predictions:

.. list-table:: STZ Theory vs. MD Simulation
   :widths: 30 35 35
   :header-rows: 1

   * - Observable
     - STZ Prediction
     - MD Result
   * - Steady-state flow curve
     - :math:`\sigma \sim \sigma_y + \eta(\chi_\infty) \dot{\gamma}^n`
     - Confirmed; exponent :math:`n \approx 0.5-0.7`
   * - Stress overshoot
     - Present for :math:`\chi_0 < \chi_\infty`
     - Observed; magnitude depends on quench rate
   * - Strain at peak
     - :math:`\gamma_{\text{peak}} \sim 0.1-0.3`
     - Observed; :math:`\gamma_{\text{peak}} \approx 0.1`
   * - Bauschinger effect
     - Predicted from :math:`m` evolution
     - Observed in cyclic loading
   * - Jamming at low :math:`\chi`
     - No flow for :math:`\chi < \chi_c`
     - Confirmed; quenched systems arrested

The **STZ Conventional** model (:class:`rheojax.models.stz.conventional.STZConventional`) implements the effective temperature formulation developed by Langer, Falk, and Bouchbinder (Langer 2008). It captures key nonlinear rheological phenomena including:

*   **Yield Stress**: Emergence of a dynamic yield stress from structural disorder.
*   **Aging & Rejuvenation**: Time-dependent evolution of the structural state (effective temperature).
*   **Transient Overshoot**: Stress peaks during startup flow.
*   **Shear Banding**: (In spatial implementations) Instabilities arising from effective temperature gradients.

Variants
--------

The implementation supports three complexity levels suitable for different applications:

.. list-table:: Model Variants
   :widths: 20 25 15 40
   :header-rows: 1

   * - Variant
     - State Variables
     - Complexity
     - Best For
   * - **Minimal**
     - :math:`s, \chi`
     - Low
     - Steady-state flow curves, simple yield stress fluids.
   * - **Standard**
     - :math:`s, \chi, \Lambda`
     - Medium
     - **Default**. Aging, thixotropy, stress overshoot, transients.
   * - **Full**
     - :math:`s, \chi, \Lambda, m`
     - High
     - LAOS, back-stress, Bauschinger effect, strong anisotropy.

Physical Foundations
--------------------

Amorphous Solids and Localized Plasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike crystalline materials where plastic deformation occurs via dislocation
motion along slip planes, amorphous materials (glasses, colloids, emulsions)
lack long-range order. Instead, plasticity arises from **localized rearrangements**
of small groups of particles.

The STZ concept identifies these rearrangements with mesoscopic regions (5-10
particles) that can flip between two stable configurations under stress. The
flipping is an activated process, with the activation barrier depending on the
local structural disorder (effective temperature :math:`\chi`).

**Key physical picture:**

1. **Low** :math:`\chi` **(annealed glass)**: Deep potential energy minima, high barriers, rare
   STZ flips → solid-like, high yield stress
2. **High** :math:`\chi` **(rejuvenated glass)**: Shallow potential, low barriers, frequent flips
   → fluid-like, low yield stress
3. **Flow-induced heating**: Plastic dissipation increases :math:`\chi` (rejuvenation)
4. **Aging**: Quiescent relaxation decreases :math:`\chi` (annealing)

Thermodynamic Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~

Langer and Pechenik [Langer2003]_ derived the STZ rate equations from thermodynamic
first principles, using energy balance and the second law to constrain the form
of the governing equations.

**Energy Balance (First Law)**

The total energy of the system satisfies:

.. math::

   \frac{dU}{dt} = \sigma \dot{\gamma} - Q_{\text{out}}

where :math:`\sigma \dot{\gamma}` is the mechanical power input and :math:`Q_{\text{out}}`
is the heat flux to the thermal bath. For the configurational subsystem (characterized
by effective temperature :math:`\chi`), the energy balance is:

.. math::

   \frac{dU_C}{dt} = \Gamma - Q_C

where :math:`\Gamma` is the rate at which work is converted to configurational disorder
and :math:`Q_C` is the rate of configurational heat flow to the kinetic subsystem.

**Dissipation and Second Law**

The second law requires non-negative entropy production:

.. math::

   \dot{S}_{\text{irr}} = \frac{\sigma \dot{\gamma}}{T} - \frac{Q_C}{T} + \frac{Q_C}{T_{\text{eff}}} \geq 0

This constraint, combined with energy balance, determines how plastic work is
partitioned between:

1. **Heat** (dissipated to thermal bath)
2. **Configurational energy** (stored as structural disorder)

**Dissipation Rate Formula**

Langer and Pechenik showed that the plastic dissipation rate takes the form:

.. math::

   \dot{Q} = \varepsilon_0 \frac{\Lambda(\chi)}{\tau_0} \left[ \mathcal{C}(s) - s \mathcal{T}(s) / \sigma_y \right] \sigma_y

The term :math:`\mathcal{C}(s) - s \mathcal{T}(s)/\sigma_y` ensures that dissipation is
always positive—more energy is dissipated than stored as recoverable work. This thermodynamic
constraint uniquely determines the coupling between stress and STZ dynamics.

**Work-Heat Partition**

At steady state, the mechanical work is partitioned as:

- **Fraction to heat**: :math:`\approx 1 - \chi/\chi_\infty` (most work → heat at low :math:`\chi`)
- **Fraction to disorder**: :math:`\approx \chi/\chi_\infty` (more stored at high :math:`\chi`)

This explains why rejuvenation is self-limiting: as :math:`\chi \to \chi_\infty`, all
additional work goes to heat rather than further increasing disorder.

Theoretical Background
----------------------

Physical Basis
~~~~~~~~~~~~~~
The central concept of STZ theory is the **Effective Temperature** (:math:`\chi`), which characterizes the configurational disorder of the material's inherent structure.

*   **Low** :math:`\chi`: Deeply annealed, jammed state (solid-like).
*   **High** :math:`\chi`: Rejuvenated, disordered state (liquid-like).

Plastic flow is produced by STZs flipping between two stable configurations (aligned "+" or anti-aligned "-") under the bias of applied stress.

Governing Equations
-------------------

The STZ model is a coupled system of differential equations for stress, effective
temperature, STZ density, and (optionally) orientational bias.

Core Kinetics
~~~~~~~~~~~~~
The plastic strain rate :math:`\dot{\varepsilon}^{pl}` is governed by the density of STZs and the rate of their transitions:

.. math::

   \dot{\varepsilon}^{pl} = \frac{\varepsilon_0}{\tau_0} \Lambda(\chi) \mathcal{C}(s) \mathcal{T}(s)

where:

*   :math:`\Lambda(\chi) = e^{-1/\chi}` is the **STZ Density**.
*   :math:`\mathcal{C}(s) = \cosh(s/\sigma_y)^q` is the **Rate Factor** (activation).
*   :math:`\mathcal{T}(s) = \tanh(s/\sigma_y)` is the **Transition Bias**.
*   :math:`s` is the deviatoric stress.
*   :math:`\sigma_y` is the yield stress scale.

State Evolution Equations
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Effective Temperature Dynamics** (:math:`\chi`)
   Driven by plastic work (rejuvenation) and thermal relaxation (aging):

   .. math::

      \dot{\chi} = \frac{s \dot{\varepsilon}^{pl}}{c_0 \sigma_y} (\chi_\infty - \chi) + \text{Aging}(\chi)

   The term :math:`s \dot{\varepsilon}^{pl}` represents the rate of energy dissipation. :math:`\chi_\infty` is the steady-state effective temperature at high drive.

2. **STZ Density Dynamics** (:math:`\Lambda`)
   (Standard/Full variants) Relaxes toward the equilibrium value :math:`e^{-1/\chi}`:

   .. math::

      \dot{\Lambda} = -\frac{\Lambda - e^{-1/\chi}}{\tau_\beta}

3. **Orientation Dynamics** (:math:`m`)
   (Full variant) Describes the kinematic hardening or back-stress due to STZ alignment:

   .. math::

      \dot{m} = \frac{2 \mathcal{C}(s)}{\tau_0} (\mathcal{T}(s) - m) - \Gamma m

Quasilinear Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For many practical applications, the full nonlinear rate equations can be simplified
using quasilinear approximations [Langer2003]_. These are valid when stress is moderate
and the system is near steady state.

**Linear Stress Function Approximation**

At moderate stress (:math:`|s| \lesssim \sigma_y`), the transition bias simplifies to:

.. math::

   \mathcal{T}(s) = \tanh(s/\sigma_y) \approx s/\sigma_y

This linearization is accurate to within 10% for :math:`|s|/\sigma_y < 0.5`.

**Constant Creation Rate Approximation**

Near steady state, the rate factor can be approximated:

.. math::

   \mathcal{C}(s) = \cosh(s/\sigma_y)^q \approx 1

This holds when the stress-dependent STZ creation rate varies slowly compared to
the relaxation dynamics.

**Resulting Quasilinear Form**

With both approximations, the plastic strain rate becomes:

.. math::

   \dot{\varepsilon}^{pl} \approx \frac{\varepsilon_0}{\tau_0 \sigma_y} \Lambda(\chi) \cdot s

This is a **viscoplastic** constitutive law with stress-dependent viscosity
:math:`\eta_{\text{eff}} = \sigma_y \tau_0 / (\varepsilon_0 \Lambda(\chi))`.

**Validity Conditions**

The quasilinear approximation works well when:

1. :math:`|s|/\sigma_y < 0.5` (moderate stress)
2. :math:`\chi` is approximately constant (near steady state or slow driving)
3. The Weissenberg number :math:`\text{Wi} = \dot{\gamma} \tau_0 < 1`

For transient phenomena (startup, LAOS), the full nonlinear equations should be used.

Yield Stress and Dynamic Stability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key insight from thermodynamic analysis [Langer2003]_ is that the **yield stress
emerges from an exchange of dynamic stability** between jammed and flowing states.

**Jammed State (Low** :math:`\chi` **)**

For :math:`\chi < \chi_c` (critical effective temperature), the system is dynamically
stable in a jammed state:

- STZ density :math:`\Lambda = e^{-1/\chi}` is exponentially small
- No plastic flow occurs: :math:`\dot{\varepsilon}^{pl} \to 0`
- Applied stress is supported elastically up to :math:`\sigma \lesssim \sigma_y`
- The system behaves as a solid

**Flowing State (High** :math:`\chi` **)**

For :math:`\chi > \chi_c`, the system transitions to a flowing state:

- STZ density is appreciable: :math:`\Lambda \sim O(1)`
- Plastic strain rate balances applied strain rate
- Stress reaches a plateau (dynamic yield stress)
- The system behaves as a viscoplastic fluid

**Critical Effective Temperature**

The critical value :math:`\chi_c` can be estimated from the condition that
the flow and aging rates balance:

.. math::

   \chi_c \approx \frac{1}{\ln(\tau_{\text{age}}/\tau_0)}

Typical values: :math:`\chi_c \approx 0.3-0.5` for metallic glasses.

**Bifurcation and Hysteresis**

The transition between jammed and flowing states exhibits:

- **Startup**: At fixed :math:`\dot{\gamma}`, stress overshoots then relaxes to steady state
- **Cessation**: Upon stopping flow, :math:`\chi` decreases (aging) and system re-jams
- **Hysteresis**: Start-up yield stress differs from cessation yield stress

This bifurcation structure explains phenomena like **thixotropic yielding** and
**viscosity bifurcation** in stress-controlled experiments.

Rate Factor and Thermal Activation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The molecular rate factor :math:`\Gamma` [1]_ captures the thermally activated nature
of STZ transitions:

.. math::

   \Gamma = \tau_0^{-1} \exp\left(-\frac{E_Z}{k_B T}\right)

where:

- :math:`\tau_0^{-1}` is the attempt frequency (molecular vibration rate)
- :math:`E_Z` is the activation barrier for STZ rearrangement
- :math:`T` is the thermal (bath) temperature

**Separation of Timescales**

Two fundamental timescales govern STZ dynamics:

1. **Fast timescale** :math:`\tau_0 \sim 10^{-12}` s: molecular vibrations, elastic response
2. **Slow timescale** :math:`\tau_R = \tau_0 e^{E_Z/k_BT}`: structural relaxation

The ratio :math:`\tau_R/\tau_0` diverges at the glass transition:

.. math::

   \frac{\tau_R}{\tau_0} \to \infty \quad \text{as} \quad T \to T_g

This separation underlies the nonequilibrium nature of glasses and the need for
the effective temperature :math:`\chi` as an additional state variable.

**Super-Arrhenius Behavior**

Near the glass transition, relaxation times exhibit super-Arrhenius (Vogel-Fulcher-Tammann)
behavior:

.. math::

   \tau_R = \tau_0 \exp\left(\frac{B}{T - T_0}\right)

where :math:`T_0 < T_g` is the Vogel temperature. This reflects the cooperative nature
of rearrangements as temperature decreases.

**Newtonian Viscosity Limit**

At low stress and strain rate, the STZ model recovers Newtonian viscosity:

.. math::

   \eta = G_0 \tau_R = G_0 \tau_0 \exp\left(\frac{E_Z}{k_B T}\right)

The temperature dependence of viscosity is thus controlled by the STZ activation energy.

Memory Effects and Bauschinger Effect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Full variant of the STZ model includes an **orientational bias** :math:`m` that
captures memory of the deformation history [2]_.

**Orientational Bias**

The variable :math:`m` represents the average orientation of STZs:

- :math:`m = 0`: Random orientation (isotropic)
- :math:`m > 0`: STZs biased in the positive shear direction
- :math:`m < 0`: STZs biased in the negative shear direction

Under steady shear, :math:`m \to m_\infty \cdot \text{sign}(\dot{\gamma})`.

**Bauschinger Effect**

When the shear direction is reversed:

1. Initial response is softer (lower effective yield stress) because existing STZs
   are pre-oriented to flip in the new direction
2. The material re-hardens as :math:`m` reverses sign
3. Asymmetric response in tension/compression cycles

This is the Bauschinger effect, well-known in metallurgy and captured naturally
by the STZ orientation variable.

**Strain Recovery**

After cessation of flow, partial strain recovery can occur:

.. math::

   \gamma_{\text{recovered}} \propto m \cdot \Lambda(\chi) \cdot \Delta t

This is an **anelastic** (delayed elastic) effect arising from the relaxation of
oriented STZ populations.

**Kinematic Hardening**

The back-stress :math:`\sigma_{\text{back}} = \sigma_y \cdot m` acts as a kinematic
hardening term, shifting the center of the yield surface. This is important for:

- Large amplitude oscillatory shear (LAOS)
- Cyclic loading and fatigue
- Start-up after flow reversal

Validity and Assumptions
------------------------

**Model Assumptions:**

1. **Mesoscopic STZ size**: Rearrangements involve ~5-10 particles (coarse-grained)
2. **Effective temperature**: Configurational disorder can be described by a single
   scalar :math:`\chi`
3. **Two-state STZ**: Each zone can flip between "+" and "-" orientations
4. **Local stress bias**: Applied stress biases transitions via :math:`\tanh(s/\sigma_y)`
5. **Separation of timescales**: Fast elastic response (:math:`\tau_0`) vs slow :math:`\chi` evolution

**When the model works well:**

- Amorphous solids below glass transition (:math:`T < T_g`)
- Dense colloidal suspensions (:math:`\phi > 0.55`)
- Metallic glasses under deformation
- Systems where plastic flow is localized (not cooperative)

**Limitations:**

- No spatial coupling (homogeneous model; use nonlocal variants for shear banding)
- Assumes scalar effective temperature (no tensorial disorder)
- No explicit aging kinetics beyond :math:`\chi` relaxation
- Steady-state plasticity may differ from real activated hopping

**Data Requirements:**

- Flow curves (steady shear) for basic fitting
- Startup flow for transient dynamics and :math:`\chi` evolution
- LAOS for nonlinear rheology and back-stress effects (Full variant)

What You Can Learn
------------------

STZ theory provides a microscopic framework for understanding plasticity in amorphous materials through the effective temperature :math:`\chi` and the density of active shear transformation zones :math:`\Lambda(\chi)`.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\chi` **(Effective Temperature)**:
   The configurational disorder parameter, normalized by the glass transition temperature.

   *For graduate students*: :math:`\chi = T_{\text{eff}}/T_g` is the ratio of the effective configurational temperature to the glass transition temperature :math:`T_g`. Unlike thermal temperature :math:`T`, :math:`\chi` quantifies the disorder in the inherent structure (energy landscape minima). In equilibrium, :math:`\chi \to k_B T/T_g`. Under flow, plastic dissipation drives :math:`\chi` above its equilibrium value (rejuvenation). The STZ density :math:`\Lambda = \exp(-e_Z/\chi) \approx \exp(-1/\chi)` controls the rate of plastic events. At :math:`\chi = 1`, the system is at the glass transition; :math:`\chi < 1` is glassy (arrested), :math:`\chi > 1` is liquid-like.

   *For practitioners*: :math:`\chi < 0.5` means deeply annealed glass (high yield stress, brittle), :math:`0.5 < \chi < 1.0` means moderately annealed (moderate yield stress, ductile), :math:`\chi > 1.0` means rejuvenated or liquid-like (low or no yield stress). Fitting :math:`\chi_0` from startup overshoot magnitude and :math:`\chi_{\infty}` from steady-state shear thinning reveals the material's structural evolution under flow.

:math:`\sigma_y` **(Yield Stress Scale)**:
   The stress scale for STZ activation, not the macroscopic yield stress.

   *For graduate students*: :math:`\sigma_y` appears in the activation factors :math:`\mathcal{C}(s) = \cosh(s/\sigma_y)^q` and :math:`\mathcal{T}(s) = \tanh(s/\sigma_y)`. It sets the stress scale at which STZs flip from one orientation to the other. The macroscopic yield stress :math:`\sigma_y^{\text{eff}} \sim \sigma_y \sqrt{\Lambda(\chi)}` depends on the STZ density. Near the glass transition, :math:`\sigma_y` is related to the shear modulus times the STZ size: :math:`\sigma_y \approx G_0\varepsilon_0`.

   *For practitioners*: :math:`\sigma_y` controls the curvature of the flow curve. Larger :math:`\sigma_y` means the material transitions more gradually from solid-like to fluid-like behavior. Fit :math:`\sigma_y` from the stress scale where the flow curve bends (not the low-rate plateau, which depends on :math:`\chi`).

:math:`\varepsilon_0` **(STZ Strain)**:
   The local strain released when a single STZ flips orientation.

   *For graduate students*: :math:`\varepsilon_0` is the typical strain increment per STZ rearrangement event. It represents the local shear transformation of a cluster of ~5-10 particles. The plastic strain rate is :math:`\dot{\varepsilon}^{pl} = \varepsilon_0 \Lambda(\chi) R` where :math:`R` is the STZ flip rate. Typical values :math:`\varepsilon_0 \approx 0.1\text{--}0.3` correspond to a displacement of ~10-30% of the particle diameter.

   *For practitioners*: :math:`\varepsilon_0` is usually fixed (not fitted) at 0.1 or 0.2 based on literature values for similar materials. It controls the absolute magnitude of the plastic strain rate.

:math:`c_0` **(Effective Specific Heat)**:
   The configurational heat capacity controlling the rate of :math:`\chi` evolution.

   *For graduate students*: :math:`c_0` appears in :math:`d\chi/dt = (s\dot{\varepsilon}^{pl}/c_0\sigma_y)(\chi_{\infty} - \chi)`. It represents the density of configurational states per unit energy. Physically, :math:`c_0 \sim (k_B/T_g)(\partial S_{\text{conf}}/\partial E)_V` where :math:`S_{\text{conf}}` is the configurational entropy. Lower :math:`c_0` means the system heats (increases :math:`\chi`) more rapidly under plastic dissipation.

   *For practitioners*: :math:`c_0` controls the width of the stress overshoot in startup. Smaller :math:`c_0` → sharper overshoot. Fit :math:`c_0` from the time to reach peak stress at a given shear rate. Typical values: 0.1-1.0.

:math:`\tau_0` **(Attempt Time)**:
   The microscopic timescale for STZ flip attempts.

   *For graduate students*: :math:`\tau_0` is the inverse attempt frequency, related to phonon vibrations (metallic glasses) or Brownian diffusion (colloids). The plastic strain rate scales as :math:`\dot{\varepsilon}^{pl} \sim \varepsilon_0/\tau_0`. For metallic glasses, :math:`\tau_0 \approx 10^{-12}\text{--}10^{-9}` s (atomic vibrations). For colloids, :math:`\tau_0 \approx \eta_s a^3/(k_B T)` (Brownian time).

   *For practitioners*: :math:`\tau_0` sets the absolute timescale of flow. Fit :math:`\tau_0` from the shear rate scale where the flow curve transitions from yield-dominated to rate-dependent. Typical values: :math:`10^{-9}`-:math:`10^{-6}` s for glasses, :math:`10^{-4}`-:math:`10^{-1}` s for pastes.

:math:`e_Z` **(STZ Formation Energy)**:
   The energy barrier for creating a new STZ, normalized by :math:`k_B T_g`.

   *For graduate students*: :math:`e_Z` appears in the equilibrium STZ density :math:`\Lambda_{\text{eq}} = \exp(-e_Z/\chi)`. It represents the free energy cost of introducing a local rearrangeable region. In the Standard/Full variants, :math:`d\Lambda/dt = -(\Lambda - \exp(-e_Z/\chi))/\tau_\beta` describes the relaxation toward equilibrium. Typical values :math:`e_Z \approx 0.5\text{--}2`.

   *For practitioners*: :math:`e_Z` controls the equilibrium STZ density and thus the long-time aging behavior. Higher :math:`e_Z` means fewer equilibrium STZs and slower aging. Usually fitted from aging experiments (stress growth at rest).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from STZ Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\chi` Range
     - Structural State
     - Typical Materials
     - Flow Behavior
   * - :math:`\chi` **< 0.4**
     - Deeply annealed glass
     - Aged metallic glasses, ultra-strong colloids
     - Very high yield stress (>10 GPa for metals, >1 kPa for colloids), brittle, catastrophic failure, minimal ductility
   * - **0.4 <** :math:`\chi` **< 0.7**
     - Moderately annealed glass
     - As-quenched metallic glasses, carbopol gels, aged emulsions
     - High yield stress (1-10 GPa for metals, 100-1000 Pa for colloids), ductile with large overshoot, significant aging
   * - **0.7 <** :math:`\chi` **< 1.0**
     - Weakly annealed glass
     - Rejuvenated metallic glasses, fresh colloidal suspensions
     - Moderate yield stress (0.1-1 GPa for metals, 10-100 Pa for colloids), small overshoot, weak aging
   * - **1.0 <** :math:`\chi` **< 1.5**
     - Near-transition
     - Glasses near :math:`T_g`, very soft colloids
     - Low or no clear yield stress, strong shear thinning, no aging
   * - :math:`\chi` **> 1.5**
     - Supercooled liquid
     - Above :math:`T_g`, dilute suspensions
     - Newtonian or weakly shear-thinning, no solid-like behavior

Connection to Aging and Rejuvenation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Aging (Quiescent Evolution)**: In the absence of flow, :math:`\chi` decreases via:

.. math::

   \dot{\chi}_{\text{aging}} = -\frac{\chi - \chi_{\text{eq}}}{\tau_{\text{age}}}

- Aging timescale :math:`\tau_{\text{age}}` can be :math:`10^3-10^6` seconds (hours to days)
- Decrease in :math:`\chi` → increase in yield stress over time (thixotropic hardening)
- Measurable via time-dependent stress growth in startup experiments

**Rejuvenation (Flow-Induced Heating)**: During flow, plastic dissipation increases :math:`\chi`:

.. math::

   \dot{\chi}_{\text{rejuv}} = \frac{s \dot{\varepsilon}^{pl}}{c_0 \sigma_y} (\chi_\infty - \chi)

- Rate proportional to :math:`s \dot{\varepsilon}^{pl}` (mechanical power input)
- Higher shear rates → faster rejuvenation → lower effective viscosity
- Explains shear thinning and stress overshoot in startup

**Balance at Steady State**: Flow-induced heating balances structural relaxation

.. math::

   \chi_{ss} = \chi_\infty \left( 1 - e^{-s \dot{\varepsilon}^{pl} / (\text{aging rate})} \right)

Yield Stress from Structural Disorder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unlike phenomenological yield stress models (Herschel-Bulkley), STZ theory connects
the yield stress to microscopic parameters:

.. math::

   \sigma_y^{\text{eff}} \sim \sigma_y \sqrt{\Lambda(\chi)} \sim \sigma_y \exp(-1/2\chi)

**Physical interpretation:**

- **Low** :math:`\chi`: Few STZs available (:math:`\Lambda \to 0`), very high activation barrier
- **High** :math:`\chi`: Many STZs (:math:`\Lambda \to 1`), easy plastic flow

This explains why:

1. **Aging increases yield stress**: :math:`\chi` decreases → :math:`\Lambda` decreases → fewer active STZs
2. **Rejuvenation decreases yield stress**: :math:`\chi` increases → :math:`\Lambda` increases → more active STZs
3. **Temperature dependence**: Near :math:`T_g`, :math:`\chi` is very sensitive to temperature

Transient Stress Overshoot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress overshoot in startup flow arises from competition between:

1. **Elastic loading**: :math:`s` increases as strain accumulates
2. **Structural evolution**: :math:`\chi` increases due to plastic dissipation
3. **Accelerating plasticity**: Higher :math:`\chi` → higher :math:`\Lambda` → faster :math:`\dot{\varepsilon}^{pl}`

**Peak stress location**: Occurs when :math:`d\sigma/dt = 0`, typically at strain :math:`\gamma \sim 0.1\text{--}0.3`

**Overshoot magnitude**: :math:`\sigma_{\text{peak}} / \sigma_{ss}` increases with:

- Lower initial :math:`\chi` (more annealed)
- Higher shear rate (:math:`\text{Wi} > 1`)
- Lower :math:`c_0` (slower :math:`\chi` evolution)

Fitting Strategy
~~~~~~~~~~~~~~~~

From steady-state flow curves, extract:

1. :math:`\sigma_y`: Plateau stress at low :math:`\dot{\gamma}`
2. **Shear thinning slope**: Related to :math:`\chi_{\infty}` and :math:`c_0`

From startup transients, extract:

3. :math:`\chi_0` **(initial state)**: Controls overshoot magnitude
4. :math:`\tau_\beta` or :math:`c_0`: Controls overshoot timing

From aging experiments, extract:

5. **Aging timescale**: Related to :math:`e_Z` and thermal relaxation

Numerical Implementation
------------------------

This implementation leverages **JAX** and **Diffrax** for high-performance simulation:

*   **JIT Compilation**: All physics kernels are JIT-compiled for speed.
*   **Stiff Solvers**: Uses implicit ODE solvers (e.g., Kvaerno5, Tsit5) to handle the fast timescales of STZ flips vs. slow aging.
*   **Protocol Support**:
    *   **Steady Shear**: Algebraic solution (instantaneous).
    *   **Transient**: ODE integration for startup, relaxation, and creep.
    *   **LAOS**: Full cycle integration + FFT for harmonic analysis.

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 60

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``G0``
     - :math:`G_0`
     - Pa
     - High-frequency elastic shear modulus.
   * - ``sigma_y``
     - :math:`\sigma_y`
     - Pa
     - Yield stress scale (activation barrier).
   * - ``chi_inf``
     - :math:`\chi_\infty`
     - -
     - Steady-state effective temperature limit.
   * - ``tau0``
     - :math:`\tau_0`
     - s
     - Molecular vibration timescale (attempt time).
   * - ``epsilon0``
     - :math:`\varepsilon_0`
     - -
     - Strain increment per STZ rearrangement (typically 0.1-0.3).
   * - ``c0``
     - :math:`c_0`
     - -
     - Effective specific heat (controls rate of :math:`\chi` evolution).
   * - ``ez``
     - :math:`e_Z`
     - -
     - STZ formation energy (normalized by :math:`k_B T_g`).
   * - ``tau_beta``
     - :math:`\tau_\beta`
     - s
     - Relaxation timescale for STZ density :math:`\Lambda`.
   * - ``m_inf``
     - :math:`m_\infty`
     - -
     - Saturation value for orientational bias (Full variant).
   * - ``rate_m``
     - :math:`\Gamma_m`
     - -
     - Rate coefficient for orientational bias evolution (Full variant).

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: From flow curve (steady shear)**

Fit :math:`\sigma(\dot{\gamma})` to extract:

- :math:`\sigma_y`: Extrapolate to :math:`\dot{\gamma} \to 0`
- :math:`\chi_{\infty}`: From shear thinning slope (higher slope → higher :math:`\chi_{\infty}`)

**Step 2: From startup overshoot**

Fit :math:`\sigma(t)` at constant :math:`\dot{\gamma}` to extract:

- :math:`\chi_0` (initial :math:`\chi`): Controls overshoot height
- :math:`c_0` or :math:`\tau_\beta`: Controls overshoot width

**Step 3: From LAOS (optional, Full variant)**

Fit Lissajous curves to extract:

- :math:`m_\infty`, :math:`\Gamma_m`: Back-stress and kinematic hardening parameters

Typical Parameter Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Notes
   * - :math:`\chi_0`
     - 0.3-1.0
     - Initial effective temperature (lower = more annealed)
   * - :math:`\chi_{\infty}`
     - 0.5-2.0
     - Steady-state at high drive (higher = more rejuvenated)
   * - :math:`\sigma_y`
     - :math:`10^2-10^6` Pa
     - Material-dependent yield stress scale
   * - :math:`\tau_0`
     - :math:`10^{-9}`--:math:`10^{-6}` s
     - Molecular vibration time (faster for colloids than polymers)
   * - :math:`\varepsilon_0`
     - 0.1-0.3
     - Strain per STZ flip (dimensionless)
   * - :math:`c_0`
     - 0.1-1.0
     - Specific heat (higher = slower :math:`\chi` evolution)

Troubleshooting
~~~~~~~~~~~~~~~

**Problem: No stress overshoot in startup**

- **Solution**: Increase initial :math:`\chi` contrast (lower :math:`\chi_0` or higher :math:`\chi_{\infty}`)
- Or increase shear rate (need :math:`\text{Wi} = \dot{\gamma} \tau_\alpha > 1`)

**Problem: Overshoot too sharp/broad**

- **Solution**: Adjust :math:`c_0` (lower :math:`c_0` means sharper overshoot)
- Or adjust :math:`\tau_\beta` (Standard/Full variant)

**Problem: Wrong steady-state stress**

- **Solution**: Adjust :math:`\sigma_y` and :math:`\chi_{\infty}` simultaneously
- Check if variant is appropriate (Minimal vs Standard vs Full)

Usage
-----

.. code-block:: python

   import numpy as np
   from rheojax.models import STZConventional

   # Initialize model (Standard variant includes Lambda dynamics)
   model = STZConventional(variant="standard")

   # --- 1. Steady State Flow Curve Fitting ---
   # Fit to shear rate vs stress data
   gamma_dot = np.logspace(-3, 1, 20)
   stress_data = ... # Experimental data

   model.fit(gamma_dot, stress_data, test_mode='steady_shear')

   print(model.parameters.get_value("sigma_y"))

   # --- 2. Transient Startup Simulation ---
   # Simulate stress overshoot at constant shear rate
   t = np.linspace(0, 10, 1000)
   stress_overshoot = model.predict(t, test_mode='startup', gamma_dot=1.0)

   # --- 3. LAOS Simulation ---
   # Large Amplitude Oscillatory Shear
   strain, stress = model.simulate_laos(gamma_0=1.0, omega=5.0)

See Also
--------

- :doc:`../sgr/sgr_conventional` — Soft Glassy Rheology (alternative effective temperature model)
- :doc:`../itt_mct/itt_mct_schematic` — Mode-Coupling Theory (cage-based glass transition)
- :doc:`../fluidity/fluidity_saramito_local` — Fluidity models (simpler thixotropic framework)
- :doc:`../dmt/dmt_local` — DMT thixotropic models (structural kinetics approach)

**Choosing between STZ and other models:**

- **Use STZ** if: Amorphous solids, metallic glasses, strong effective temperature effects
- **Use SGR** if: Soft glasses (foams, emulsions), trap-based interpretation preferred
- **Use ITT-MCT** if: Colloidal suspensions, connection to structure factor S(k)
- **Use Fluidity/DMT** if: Simpler thixotropic phenomenology, fewer parameters

Limitations and Extensions
--------------------------

The STZ Conventional model makes several simplifying assumptions that limit its
applicability in certain scenarios.

Known Limitations
~~~~~~~~~~~~~~~~~

**Spatial Homogeneity**

The model assumes homogeneous deformation. In reality, amorphous solids often
exhibit **shear banding**—spatial localization of plastic flow into thin bands.
Shear bands arise from the coupling between:

- Flow-induced heating (:math:`\chi` increases)
- Stress softening (lower viscosity at higher :math:`\chi`)
- Positive feedback leading to localization

*Extension*: Nonlocal STZ models add diffusion of the effective temperature
:math:`D_\chi \nabla^2 \chi` to regularize the localization instability.

**Scalar Effective Temperature**

The model uses a single scalar :math:`\chi` to characterize disorder. More
generally, structural disorder could be:

- Tensorial (different disorder in different directions)
- Multi-valued (multiple length scales of disorder)
- Non-local (correlated over mesoscopic distances)

*Extension*: Tensorial STZ models track orientation-dependent disorder.

**Athermal Limit**

The model assumes thermal activation over barriers. In athermal systems
(e.g., granular matter at :math:`T = 0`), a different mechanism governs
the STZ transition rate:

.. math::

   R \propto |\sigma - \sigma_y|^\beta \Theta(|\sigma| - \sigma_y)

*Extension*: Rate-independent (quasi-static) STZ models for granular plasticity.

**Simple Aging Kinetics**

The model includes relaxation toward equilibrium via :math:`\dot{\chi} \propto
-(\chi - \chi_{\text{eq}})`, but real aging can be:

- Logarithmic in time (not exponential)
- History-dependent (memory effects beyond :math:`m`)
- Sensitive to stress during aging

**Temperature Gradients**

The model assumes isothermal conditions. Under rapid deformation, adiabatic
heating can raise the thermal temperature, coupling :math:`T` and :math:`\chi`
dynamics:

.. math::

   \rho c_p \dot{T} = \beta \sigma \dot{\gamma}^{pl}

where :math:`\beta` is the Taylor-Quinney coefficient.

Active Research Directions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shear Band Dynamics**

Understanding the nucleation, propagation, and arrest of shear bands using
spatially resolved STZ models. Key questions:

- What controls shear band width?
- How do bands interact and coalesce?
- Can band formation be suppressed by tailoring thermal history?

**Polymer Glasses**

Extending STZ theory to polymer glasses, where chain connectivity introduces:

- Entanglement effects at high strain
- Reptation dynamics at long times
- Craze formation vs shear yielding

**Multi-Dimensional Stress States**

Extending beyond simple shear to:

- Triaxial compression (geological applications)
- Combined shear and normal stress
- Pressure sensitivity of yield stress

**Connections to Machine Learning**

Using neural networks to:

- Identify STZ events in MD simulations
- Learn effective constitutive laws from data
- Accelerate multi-scale simulations

References
----------

.. [1] Langer, J. S. "Shear-transformation-zone theory of plastic deformation near the
   glass transition." *Physical Review E*, 77, 021502 (2008).
   https://doi.org/10.1103/PhysRevE.77.021502
   :download:`PDF <../../../reference/langer_2008_stz_dynamics.pdf>`

.. [2] Falk, M. L. and Langer, J. S. "Dynamics of viscoplastic deformation in amorphous
   solids." *Physical Review E*, 57, 7192 (1998).
   https://doi.org/10.1103/PhysRevE.57.7192
   :download:`PDF <../../../reference/falk_langer_1998_stz.pdf>`

.. [3] Bouchbinder, E. and Langer, J. S. "Nonequilibrium thermodynamics of driven
   amorphous materials." *Physical Review E*, 80, 031131, 031132, 031133 (2009).
   https://doi.org/10.1103/PhysRevE.80.031131

.. [Langer2003] Langer, J. S. and Pechenik, L. "Dynamics of shear-transformation zones
   in amorphous plasticity: Energetic constraints in a minimal theory."
   *Physical Review E*, 68, 061507 (2003).
   https://doi.org/10.1103/PhysRevE.68.061507
   :download:`PDF <../../../reference/langer_pechenik_2003_stz_minimal.pdf>`

.. [Cohen1959] Cohen, M. H. and Turnbull, D. "Molecular transport in liquids and glasses."
   *The Journal of Chemical Physics*, 31, 1164-1169 (1959).
   https://doi.org/10.1063/1.1730566

.. [4] Manning, M. L., Langer, J. S., and Carlson, J. M. "Strain localization in a shear
   transformation zone model for amorphous solids." *Physical Review E*, 76, 056106
   (2007). https://doi.org/10.1103/PhysRevE.76.056106

.. [5] Rottler, J. and Robbins, M. O. "Shear yielding of amorphous glassy solids: Effect
   of temperature and strain rate." *Physical Review E*, 68, 011507 (2003).
   https://doi.org/10.1103/PhysRevE.68.011507

.. [6] Argon, A. S. "Plastic deformation in metallic glasses."
   *Acta Metallurgica*, **27**, 47-58 (1979).
   https://doi.org/10.1016/0001-6160(79)90055-5

.. [7] Spaepen, F. "A microscopic mechanism for steady state inhomogeneous flow in metallic glasses."
   *Acta Metallurgica*, **25**, 407-415 (1977).
   https://doi.org/10.1016/0001-6160(77)90232-2

.. [8] Homer, E. R. & Schuh, C. A. "Mesoscale modeling of amorphous metals by shear transformation zone dynamics."
   *Acta Materialia*, **57**, 2823-2833 (2009).
   https://doi.org/10.1016/j.actamat.2009.02.035

.. [9] Nicolas, A., Ferrero, E. E., Martens, K., & Barrat, J.-L. "Deformation and flow of amorphous solids: Insights from elastoplastic models."
   *Reviews of Modern Physics*, **90**, 045006 (2018).
   https://doi.org/10.1103/RevModPhys.90.045006

.. [10] Jagla, E. A. "Shear band dynamics from a mesoscopic modeling of plasticity."
   *Journal of Statistical Mechanics: Theory and Experiment*, **2010**, P12025 (2010).
   https://doi.org/10.1088/1742-5468/2010/12/P12025

API Reference
-------------

.. autoclass:: rheojax.models.stz.conventional.STZConventional
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. autoclass:: rheojax.models.stz._base.STZBase
   :members: get_initial_state
   :undoc-members:
   :no-index:
