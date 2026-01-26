.. _model-spp-decomposer:

SPP Decomposer (Sequence of Physical Processes)
================================================

Quick Reference
---------------

- **Use when:** Large Amplitude Oscillatory Shear (LAOS) analysis of yield-stress fluids, colloidal glasses, soft glassy materials
- **Parameters:** Extracted quantities (G_cage, sigma_sy, sigma_dy, I3/I1, S-factor, T-factor)
- **Key equation:** :math:`\sigma(t) = G'_t(t)\gamma(t) + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_d(t)`
- **Test modes:** LAOS (time-domain stress-strain waveforms)
- **Material examples:** Colloidal glasses, microgel pastes, concentrated emulsions, foams, polymer gels

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`G_{\text{cage}}`
     - Cage modulus (elastic stiffness at :math:`\sigma` = 0)
   * - :math:`\sigma_{y,\text{static}}`
     - Static yield stress (at strain reversal, :math:`\dot{\gamma} = 0`)
   * - :math:`\sigma_{y,\text{dynamic}}`
     - Dynamic yield stress (at rate reversal, :math:`\gamma \neq 0`)
   * - :math:`G'_t(t)`
     - Instantaneous storage modulus (time-dependent)
   * - :math:`G''_t(t)`
     - Instantaneous loss modulus (time-dependent)
   * - :math:`\gamma_0`
     - Strain amplitude
   * - :math:`\omega`
     - Angular frequency (rad/s)
   * - :math:`I_3/I_1`
     - Third harmonic intensity ratio (nonlinearity measure)
   * - :math:`S`
     - Strain stiffening factor
   * - :math:`T`
     - Shear thickening factor
   * - :math:`\delta_t(t)`
     - Instantaneous phase angle
   * - :math:`\gamma_{eq}(t)`
     - Equilibrium strain position (material frame)

Overview
--------

The **Sequence of Physical Processes (SPP)** framework, introduced by Rogers et al. (2011) [1]_
and comprehensively formalized in Rogers (2017) [2]_, provides a time-domain approach to
Large Amplitude Oscillatory Shear (LAOS) analysis that extracts **instantaneous** elastic
modulus and dynamic viscosity without Fourier decomposition.

Unlike Fourier/Chebyshev methods that decompose stress into potentially infinite harmonics
representing a static point in n-dimensional space, SPP views the stress response as a
**dynamic trajectory** through a two-dimensional, easily interpretable space. The material
response is described as a sequence of discrete physical processes:

1. **Elastic extension** of microstructural cages
2. **Static yielding** when cages rupture
3. **Viscous flow** following the steady-state flow curve
4. **Cage reformation** when instantaneous shear rate approaches zero

This approach provides intuitively simpler interpretation while removing the problem of
understanding an infinite number of harmonic coefficients. The SPP framework:

- Makes **no assumptions about symmetries** in the response
- Views **each infinitesimal portion** of the response as the object to be analyzed
- Uses **differential parameters** that are immune to arbitrary reference state choices
- Distinguishes between strains in the **lab frame** and **material frame**

Physical Foundations
--------------------

The Four-Step Yielding Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For yield-stress materials under LAOS, the material undergoes a characteristic four-step
sequence within each oscillatory cycle:

**Step 1: Elastic Extension (Cage Deformation)**

Beginning at zero stress, strain causes the microstructural cage to extend elastically.
The cage represents the effective constraint formed by nearest neighbors (colloidal
particles, polymer entanglements). During this phase:

- Stress increases linearly with strain
- The slope represents the **cage modulus** :math:`G_{\text{cage}}`
- Material behaves as an elastic solid

**Step 2: Static Yielding (Cage Rupture)**

When cages are strained beyond their yield strains, the system begins to flow:

- Associated with the stress overshoot ("bump") after linear buildup
- **Static yield stress** :math:`\sigma_{y,\text{static}}`: maximum stress before rupture
- **Yield strain** :math:`\gamma_y`: strain at which cages break

**Step 3: Viscous Flow (Post-Yield)**

After yielding, the material follows the steady-state flow curve:

- Power-law behavior: :math:`\sigma = K|\dot{\gamma}|^n`
- Same flow curve as steady-shear experiments
- Continues until instantaneous shear rate approaches zero

**Step 4: Cage Reformation (Recovery)**

When the instantaneous shear rate momentarily becomes zero at strain extrema:

- Cages instantaneously reform
- **Dynamic yield stress** :math:`\sigma_{y,\text{dynamic}}`: stress at reformation
- Process begins again in opposite direction

Cage Model Physics
~~~~~~~~~~~~~~~~~~

The SPP framework is built on the **cage model** from colloidal physics [3]_. In a quiescent
system of concentrated colloids:

- A representative colloid's motion is hindered by nearest neighbors forming an effective **cage**
- At short times: :math:`\beta`-relaxation (rattling within cage)
- At long times: :math:`\alpha`-relaxation (cage escape)
- External shearing can break cages and induce flow

The cage modulus :math:`G_{\text{cage}}` directly measures the elastic strength of this
microstructural caging. Key properties:

- **Constant across amplitudes**: Unlike :math:`G'`, :math:`G_{\text{cage}}` remains constant
  even at large amplitudes where :math:`G'` decreases by more than a decade
- **State-dependent**: Solid-state cages are stronger than soft-state cages
- **Recovers after breaking**: Once broken, cages reform with the soft-state modulus

Constitutive Equations
----------------------

Complete Stress Reconstruction (Rogers 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fundamental contribution of Rogers (2017) [2]_ is recognizing that **three**, not two,
time-dependent nonlinear viscoelastic functions are required to fully describe any response.
The complete stress decomposition is:

.. math::

   \sigma(t) = G'_t(t)[\gamma(t) - \gamma_{eq}(t)] + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_y(t)

where:
   - :math:`G'_t(t)` — instantaneous storage modulus (elastic contribution)
   - :math:`G''_t(t)` — instantaneous loss modulus (viscous contribution)
   - :math:`\gamma_{eq}(t)` — equilibrium strain position (can shift during deformation)
   - :math:`\sigma_y(t)` — yield stress term (zero-rate stress intercept)

The third term, representing the **displacement** of the osculating plane, accounts for:

- **Yield stresses**: Stress offsets not captured by moduli alone
- **Shifting strain equilibria**: The point of zero elastic extension can move
- **Non-Newtonian flow**: Deviation from simple viscoelastic behavior

Frenet-Serret Frame Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP formulation uses differential geometry on the 3D response trajectory in the
**native deformation space** :math:`[\gamma(t), \dot{\gamma}(t)/\omega, \sigma(t)]`.
Following the Frenet-Serret apparatus [2]_:

**Binormal Vector Definition**:

.. math::

   \mathbf{B} = \mathbf{T} \times \mathbf{N} = \frac{\dot{\mathbf{A}} \times \ddot{\mathbf{A}}}
   {||\dot{\mathbf{A}} \times \ddot{\mathbf{A}}||}

where :math:`\mathbf{T}` is the tangent vector and :math:`\mathbf{N}` is the principal normal.

**Instantaneous Moduli Definitions** (Equations 18-19 from Rogers 2017):

.. math::

   G'_t(t) = -\frac{B_\gamma(t)}{B_\sigma(t)}, \qquad
   G''_t(t) = -\frac{B_{\dot{\gamma}/\omega}(t)}{B_\sigma(t)}

where :math:`B_\gamma`, :math:`B_{\dot{\gamma}/\omega}`, and :math:`B_\sigma` are the
components of the binormal vector in the strain, strain-rate, and stress directions.

**Key Property**: Because the binormal vector is a differential parameter, :math:`G'_t(t)`
and :math:`G''_t(t)` depend only on **changes** in strain, not the total strain. They are
therefore immune to the arbitrary choice of reference state that affects static schemes.
This approach has been validated against theoretical nonlinear models [4]_.

Lab Frame vs Material Frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A central conceptual advance in Rogers (2017) [2]_ is the distinction between the
**lab frame strain** and the **material frame strain**. This distinction explains
how SPP captures yielding without requiring explicit yield criteria.

**Lab Frame Strain**: :math:`\gamma(t)`
   The externally imposed deformation, measured relative to the initial configuration.
   This is what the rheometer controls and reports.

**Material Frame Strain**: :math:`\gamma(t) - \gamma_{eq}(t)`
   The actual elastic deformation experienced by the microstructure, relative to its
   current equilibrium position. This is the strain that produces elastic stress.

**Equilibrium Strain** :math:`\gamma_{eq}(t)`:
   The position of zero elastic extension, which can shift during deformation.
   In linear viscoelasticity, :math:`\gamma_{eq} = 0` at all times. For yield stress
   materials, :math:`\gamma_{eq}` shifts as cages rupture and reform.

**Physical Interpretation**:

Consider a material undergoing LAOS with amplitude :math:`\gamma_0`:

1. **Pre-yield** (:math:`|\gamma| < \gamma_y`):
   - :math:`\gamma_{eq} = 0` (equilibrium hasn't shifted)
   - Material frame strain equals lab frame strain
   - Elastic stress: :math:`\sigma_{elastic} = G'_t \cdot \gamma`

2. **During yielding** (cage rupture):
   - :math:`\gamma_{eq}` begins to shift toward :math:`\gamma`
   - Material frame strain decreases even as lab frame strain increases
   - This is how plastic flow reduces elastic stress

3. **Post-yield** (:math:`|\gamma| > \gamma_y`):
   - :math:`\gamma_{eq} \to \pm(\gamma_0 - \gamma_y)` for yield stress materials
   - Material frame strain saturates near :math:`\gamma_y`
   - Subsequent deformation is predominantly viscous

**Mathematical Expression**:

The complete stress decomposition becomes:

.. math::

   \sigma(t) = \underbrace{G'_t(t)[\gamma(t) - \gamma_{eq}(t)]}_{\text{elastic stress in material frame}}
   + \underbrace{\frac{G''_t(t)}{\omega}\dot{\gamma}(t)}_{\text{viscous stress}}
   + \underbrace{\sigma_y(t)}_{\text{yield contribution}}

For **linear viscoelastic materials**: :math:`\gamma_{eq} = 0` and :math:`\sigma_y = 0`.

For **generalized Newtonian fluids**: :math:`\gamma_{eq} = \gamma(t)` (no elastic stress).

For **yield stress materials**: :math:`\gamma_{eq}` shifts during the yielding process,
capturing the plastic strain accumulation through the displacement term.

The Displacement Term
~~~~~~~~~~~~~~~~~~~~~

The displacement term (Equation 11 from Rogers 2017) determines the position of the
osculating plane:

.. math::

   \text{displacement} = \frac{B_\gamma}{B_\sigma}\gamma(t) + \frac{B_{\dot{\gamma}/\omega}}{B_\sigma}
   \frac{\dot{\gamma}(t)}{\omega} + \sigma(t)

This is physically interpreted as:

.. math::

   \sigma_y(t) - G'_t(t)\gamma_{eq}(t)

**For linear viscoelastic responses**: The displacement is identically zero.

**For generalized Newtonian fluids**: The equilibrium strain equals the lab frame strain,
:math:`\gamma_{eq}(t) = \gamma(t)`, ensuring zero elastic stress at all times.

**For yield-stress materials**: The equilibrium position shifts during yielding, with
:math:`\gamma_{eq}` moving to :math:`\pm(\gamma_0 - \gamma_y)` after cage rupture.

Cage Modulus Definition
~~~~~~~~~~~~~~~~~~~~~~~

The cage modulus is defined as the instantaneous slope at zero stress:

.. math::

   G_{\text{cage}} \equiv \left. \frac{d\sigma}{d\gamma} \right|_{\sigma=0}

In the small phase angle, small-amplitude limit:

.. math::

   \lim_{\delta, \gamma_0 \to 0} G_{\text{cage}} = \lim_{\delta, \gamma_0 \to 0} G'

Yield Stress Definitions
~~~~~~~~~~~~~~~~~~~~~~~~

**Static Yield Stress** (stress at yielding point):

.. math::

   \sigma_{y,\text{static}} = \sigma(t)\big|_{\gamma=\pm\gamma_0}

**Dynamic Yield Stress** (stress at cage reformation):

.. math::

   \sigma_{y,\text{dynamic}} = \sigma(t)\big|_{\dot{\gamma}=\pm\dot{\gamma}_0}

Both yield stresses exhibit power-law rate dependence with identical exponents (~0.2):

.. math::

   \sigma_y \propto \gamma_0^{0.2} \propto \dot{\gamma}_0^{0.2}

In the large-amplitude limit: :math:`\sigma_{y,\text{static}} \approx 2.75 \times \sigma_{y,\text{dynamic}}`

Rates of Change (Stiffening/Softening, Thickening/Thinning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rogers (2017) provides explicit definitions for the rates at which moduli change
(Equations 22-23), enabling quantitative identification of transitions:

.. math::

   \dot{G}'_t = \tau ||\dot{\mathbf{A}}|| \left( \frac{N_\gamma}{B_\sigma} - \frac{B_\gamma N_\sigma}{B_\sigma^2} \right)

.. math::

   \dot{G}''_t = \tau ||\dot{\mathbf{A}}|| \left( \frac{N_{\dot{\gamma}/\omega}}{B_\sigma} -
   \frac{B_{\dot{\gamma}/\omega} N_\sigma}{B_\sigma^2} \right)

where :math:`\tau` is the torsion of the response trajectory.

**Interpretation**:

- :math:`\dot{G}'_t > 0`: **Stiffening** (increasing elasticity)
- :math:`\dot{G}'_t < 0`: **Softening** (decreasing elasticity)
- :math:`\dot{G}''_t > 0`: **Thickening** (increasing viscosity)
- :math:`\dot{G}''_t < 0`: **Thinning** (decreasing viscosity)

**Key insight**: In the linear regime where responses are planar, the torsion
:math:`\tau = 0` everywhere, making both derivatives zero. The moduli are therefore
constant in time. Nonzero torsion indicates nonlinear behavior.

Time-Domain Cole-Cole Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPP enables dynamic Cole-Cole analysis by plotting :math:`G''_t(t)` vs :math:`G'_t(t)`
parametrically through the oscillation cycle [2]_:

- **Linear regime**: Collapses to a single point :math:`(G', G'')`
- **Nonlinear regime**: Traces a trajectory showing material evolution

**Trajectory interpretation legend** (from Rogers 2017, Figure 2):

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Position/Motion
     - Interpretation
   * - :math:`G''_t = 0`
     - Purely elastic
   * - :math:`G'_t = 0`
     - Purely viscous
   * - :math:`G''_t > G'_t`
     - Predominantly viscous
   * - :math:`G'_t > G''_t`
     - Predominantly elastic
   * - Crossing :math:`G'_t=G''_t` toward viscous
     - Fluidization
   * - Crossing :math:`G'_t=G''_t` toward elastic
     - Reformation

Time-Domain Phase Angle
~~~~~~~~~~~~~~~~~~~~~~~

SPP defines an instantaneous phase angle without Fourier decomposition:

.. math::

   \delta_t(t) = \arctan\left( \frac{G''_t(t)}{G'_t(t)} \right)

This phase evolves continuously through the cycle:
   - :math:`\delta_t \to 0°`: Purely elastic (solid-like)
   - :math:`\delta_t \to 90°`: Purely viscous (liquid-like)
   - Transition regions reveal yielding and recovery

----

Governing Equations
-------------------

Complete Stress Decomposition (Rogers 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP framework provides a complete time-domain decomposition of the stress response without Fourier analysis:

.. math::

   \sigma(t) = G'_t(t)[\gamma(t) - \gamma_{eq}(t)] + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_y(t)

This three-term decomposition includes:

1. **Elastic term**: :math:`G'_t(t)[\gamma(t) - \gamma_{eq}(t)]` — instantaneous elastic response in material frame
2. **Viscous term**: :math:`\frac{G''_t(t)}{\omega}\dot{\gamma}(t)` — instantaneous viscous dissipation
3. **Displacement term**: :math:`\sigma_y(t)` — yield stress contribution

Frenet-Serret Differential Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The instantaneous moduli are computed from the Frenet-Serret binormal vector of the 3D response trajectory :math:`\mathbf{A}(t) = [\gamma(t), \dot{\gamma}(t)/\omega, \sigma(t)]`:

**Binormal vector:**

.. math::

   \mathbf{B}(t) = \frac{\dot{\mathbf{A}}(t) \times \ddot{\mathbf{A}}(t)}{||\dot{\mathbf{A}}(t) \times \ddot{\mathbf{A}}(t)||}

**Instantaneous moduli definitions:**

.. math::

   G'_t(t) = -\frac{B_\gamma(t)}{B_\sigma(t)}, \qquad G''_t(t) = -\frac{B_{\dot{\gamma}/\omega}(t)}{B_\sigma(t)}

where :math:`B_\gamma, B_{\dot{\gamma}/\omega}, B_\sigma` are the binormal components in the strain, strain-rate, and stress directions.

**Key property**: These are differential parameters depending only on changes in strain, making them immune to arbitrary reference state choices.

Rates of Change (Stiffening/Softening Dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rogers (2017) provides explicit formulas for the rates at which moduli evolve:

.. math::

   \dot{G}'_t(t) = \tau(t) ||\dot{\mathbf{A}}(t)|| \left( \frac{N_\gamma(t)}{B_\sigma(t)} - \frac{B_\gamma(t) N_\sigma(t)}{B_\sigma^2(t)} \right)

.. math::

   \dot{G}''_t(t) = \tau(t) ||\dot{\mathbf{A}}(t)|| \left( \frac{N_{\dot{\gamma}/\omega}(t)}{B_\sigma(t)} - \frac{B_{\dot{\gamma}/\omega}(t) N_\sigma(t)}{B_\sigma^2(t)} \right)

where :math:`\tau(t)` is the torsion of the trajectory and :math:`\mathbf{N}(t)` is the principal normal vector.

**Physical interpretation**:
   - :math:`\dot{G}'_t > 0`: Stiffening (increasing elasticity)
   - :math:`\dot{G}'_t < 0`: Softening (decreasing elasticity)
   - :math:`\dot{G}''_t > 0`: Thickening (increasing viscosity)
   - :math:`\dot{G}''_t < 0`: Thinning (decreasing viscosity)

In the linear regime, :math:`\tau = 0` everywhere, making both derivatives zero (constant moduli). Nonzero torsion indicates nonlinearity.

----

Analysis Methods
----------------

The SPP framework supports two complementary analysis approaches for computing
instantaneous moduli from LAOS waveforms. RheoJAX implements both methods
with Rogers-parity defaults matching the MATLAB SPPplus v2.1 implementation.

Fourier Domain Filtering
~~~~~~~~~~~~~~~~~~~~~~~~

The Fourier method reconstructs the stress waveform using a finite number
of odd harmonics before computing derivatives:

- **Parameters:**

- ``n_harmonics`` (M): Number of harmonics for reconstruction (default: 39, must be odd)
- ``n_periods`` (p): Number of oscillation periods in the data

**When to use:**

- High-quality periodic data at steady alternance
- When Fourier spectrum provides additional insight
- For comparison with FTC (Fourier-Chebyshev) analysis

**Key consideration:** M should be set based on the noise floor—higher M captures
more nonlinear detail but may amplify noise. The default M=39 balances detail
capture with noise rejection for typical rheometer data.

Numerical Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~

The numerical method uses finite differences directly on the measured waveforms:

- **Parameters:**

- ``step_size`` (k): Points for finite difference stencil (default: 8)
- ``num_mode``: Differentiation procedure

  - Mode 1: Standard (forward/backward at ends, centered elsewhere)
  - Mode 2: Looped (assumes periodic, centered everywhere—**recommended for LAOS**)

**When to use:**

- Non-periodic or partial-cycle data
- When Fourier decomposition is unnecessary
- For real-time or streaming analysis

Method Comparison
~~~~~~~~~~~~~~~~~

.. list-table:: SPP Analysis Method Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Fourier Domain
     - Numerical Differentiation
   * - Data requirement
     - Integer periods, even points/period
     - Any sampling
   * - Noise handling
     - Implicit filtering via harmonic truncation
     - Requires pre-smoothing or larger k
   * - Computational cost
     - FFT + filtering
     - Direct finite differences
   * - Best for
     - Steady-state LAOS
     - Transients, partial cycles

----

What You Can Learn
------------------

This section explains how to interpret SPP decomposition results to gain physical
insights about yielding materials and extract actionable knowledge for both research
and industrial applications.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G_cage (Cage Modulus)**:
   The cage modulus measures the elastic strength of the confining microstructural cage.

   *For graduate students*: :math:`G_\text{cage} = d\sigma/d\gamma|_{\sigma=0}` is amplitude-independent, unlike :math:`G'(\gamma_0)`.
   It directly reflects nearest-neighbor interaction stiffness and is constant across
   loading protocols. Compare with theoretical predictions: :math:`G_\text{cage} \approx \phi^2 k_B T / a^3` for hard
   spheres where :math:`\phi` is volume fraction and :math:`a` is particle radius.

   *For practitioners*: G_cage provides a true material property unaffected by strain
   amplitude. Use for batch-to-batch QC of complex fluids. Typical values: 10-1000 Pa
   (colloidal glasses), 100-10000 Pa (polymer gels), 1-100 Pa (emulsions).

:math:`\sigma_{y,static}` **(Static Yield Stress)**:
   Stress required to initiate flow from rest, measured at strain reversal.

   *For graduate students*: :math:`\sigma_{y,static}` > :math:`\sigma_{y,dynamic}` due to cage reformation during
   momentary rest. The ratio :math:`\sigma_{y,static}/\sigma_{y,dynamic}` ≈ 2-3 reveals thixotropic strength
   and cage reformation kinetics.

   *For practitioners*: Critical for applications requiring material to hold shape
   (coatings, pastes). Higher :math:`\sigma_{y,static}` = better sag resistance.

:math:`\sigma_{y,dynamic}` **(Dynamic Yield Stress)**:
   Stress during continuous flow, measured at rate reversal.

   *For graduate students*: Connects to steady-shear yield stress extrapolation.
   Both yield stresses scale as :math:`\sigma_y` ∝ :math:`\gamma_0^n` where n ≈ 0.2 for colloidal systems.

   *For practitioners*: Determines minimum stress for maintained flow. Use for
   pump sizing and flow assurance calculations.

**I_3/I_1 (Nonlinearity Ratio)**:
   Third harmonic ratio quantifying degree of nonlinearity.

   *For graduate students*: I_3/I_1 grows as ~\ :math:`\gamma_0^2` in weakly nonlinear regime.
   Larger values indicate stronger deviation from linear viscoelasticity and
   correlate with yield stress behavior.

   *For practitioners*: I_3/I_1 > 0.1 indicates significant nonlinearity. Use as
   quick screening metric for yield stress presence.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SPP Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - G_cage > 1000 Pa
     - Stiff microstructure
     - Dense colloids, filled polymers
     - High yield stress, difficult pumping
   * - G_cage < 100 Pa
     - Soft microstructure
     - Dilute emulsions, weak gels
     - Easy flow, poor shape retention
   * - :math:`\sigma_{sy}/\sigma_{dy}` > 3
     - Strong thixotropy
     - Paints, drilling muds
     - Time-dependent processing
   * - :math:`\sigma_{sy}/\sigma_{dy}` ≈ 1.5
     - Weak thixotropy
     - Food emulsions, cosmetics
     - Near-instantaneous recovery
   * - I_3/I_1 > 0.3
     - Highly nonlinear
     - Yield stress fluids, gels
     - Large amplitude processing needed
   * - I_3/I_1 < 0.05
     - Weakly nonlinear
     - Polymer solutions
     - SAOS sufficient for characterization

Physical Insights from SPP Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cage Modulus (G_cage) — Microstructural Stiffness:**

The cage modulus directly measures the elastic strength of the confining microstructural cage:

.. math::

   G_{\text{cage}} = \left. \frac{d\sigma}{d\gamma} \right|_{\sigma=0}

**Key insights**:
   - **Constant across amplitudes**: Unlike :math:`G'(\gamma_0)`, which decreases at large amplitudes, :math:`G_{\text{cage}}` remains constant
   - **True material property**: Independent of loading history or amplitude
   - **Microstructural interpretation**: Directly related to nearest-neighbor confinement stiffness
   - **Typical values**: 10-1000 Pa for colloidal glasses, 100-10000 Pa for polymer gels

**Static vs Dynamic Yield Stress — Intracycle Physical Processes:**

SPP distinguishes two physically distinct yield stresses:

**Static yield stress** :math:`\sigma_{y,\text{static}}`:
   - Measured at **strain reversal** (:math:`\gamma = \pm\gamma_0, \dot{\gamma} = 0`)
   - Represents the stress required to **initiate flow from rest**
   - Larger than dynamic yield due to cage reformation during momentary rest
   - Scales as :math:`\sigma_{y,\text{static}} \propto \gamma_0^{0.2}` (typical for colloids)

**Dynamic yield stress** :math:`\sigma_{y,\text{dynamic}}`:
   - Measured at **rate reversal** (:math:`\dot{\gamma} = 0, \gamma \neq 0`)
   - Represents the stress during **continuous flow**
   - Connects to steady-shear yield stress extrapolation
   - Scales as :math:`\sigma_{y,\text{dynamic}} \propto \gamma_0^{0.2}` with same exponent

**Ratio interpretation**: :math:`\sigma_{y,\text{static}} / \sigma_{y,\text{dynamic}} \approx 2-3` reveals information about:
   - Cage reformation kinetics (fast reformation → larger ratio)
   - Thixotropic strength (strong thixotropy → larger ratio)
   - Structural memory (long memory → larger ratio)

**Intracycle Processes — Yielding Sequence:**

SPP reveals the four-step sequence within each LAOS cycle:

1. **Elastic extension** (linear regime):
      - :math:`G'_t \approx G_{\text{cage}}`, nearly constant
      - Stress builds linearly with strain
      - Cages deform elastically

2. **Static yielding** (cage rupture):
      - :math:`G'_t` decreases sharply
      - Stress overshoot ("bump" in Lissajous)
      - Cages break, flow initiates

3. **Viscous flow** (post-yield):
      - :math:`G''_t > G'_t` (predominantly viscous)
      - Follows steady-state flow curve
      - Material flows until rate reversal

4. **Cage reformation** (recovery):
      - :math:`G'_t` increases sharply
      - Fluidity decreases as :math:`\dot{\gamma} \to 0`
      - Cages reform at dynamic yield stress

Material Characterization Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From Amplitude Sweeps:**
   - :math:`G_{\text{cage}}` (constant, true material property)
   - Yield stress amplitude scaling exponent (:math:`\sigma_y \propto \gamma_0^n`)
   - Static/dynamic yield stress ratio (thixotropic signature)
   - Nonlinearity onset (via :math:`I_3/I_1` ratio)

**From Single LAOS Waveform:**
   - Instantaneous moduli trajectories :math:`G'_t(t), G''_t(t)`
   - Yield points (static and dynamic)
   - Intracycle Cole-Cole trajectory
   - Stiffening/softening/thickening/thinning regions

**From Time-Domain Cole-Cole:**
   - Viscoelastic character evolution through cycle
   - Fluidization transitions (crossing :math:`G'_t = G''_t`)
   - Reformation transitions (return to elastic dominance)

**From Harmonic Analysis:**
   - :math:`I_3/I_1` ratio (degree of nonlinearity)
   - Higher harmonic content (explained by power-law flow post-yield)
   - Comparison with Fourier methods (qualitative consistency)

----

Parameters
----------

SPP Output Variables
~~~~~~~~~~~~~~~~~~~~

The SPP decomposer produces the following time-dependent quantities. These
correspond to the output variables in the MATLAB SPPplus v2.1 implementation.

.. list-table:: Standard SPP Output (always computed)
   :header-rows: 1
   :widths: 20 50 30

   * - Variable
     - Description
     - Notes
   * - :math:`G'_t(t)`
     - Time-dependent storage modulus
     - Instantaneous elastic response
   * - :math:`G''_t(t)`
     - Time-dependent loss modulus
     - Instantaneous viscous response
   * - :math:`|G^*_t(t)|`
     - Magnitude of complex modulus
     - :math:`\sqrt{G'^2_t + G''^2_t}`
   * - :math:`\tan(\delta_t)`
     - Time-dependent loss tangent
     - :math:`G''_t / G'_t`
   * - :math:`\delta_t(t)`
     - Time-dependent phase angle
     - Range: :math:`[-\pi/2, 3\pi/2]`
   * - :math:`\sigma_d(t)`
     - Displacement stress
     - Osculating plane position
   * - :math:`\gamma_{eq,est}(t)`
     - Estimated equilibrium strain
     - Valid when :math:`G'_t \gg G''_t`
   * - :math:`\dot{G}'_t(t)`
     - Storage modulus rate
     - Stiffening (+) / softening (-)
   * - :math:`\dot{G}''_t(t)`
     - Loss modulus rate
     - Thickening (+) / thinning (-)
   * - :math:`|\dot{G}^*_t(t)|`
     - Complex modulus speed
     - Rate of modulus change
   * - :math:`\tilde{\dot{\delta}}_t(t)`
     - Normalized phase angle velocity
     - Assumes sinusoidal strain

.. list-table:: Extended SPP Output (TNB frame vectors)
   :header-rows: 1
   :widths: 20 50 30

   * - Variable
     - Description
     - Notes
   * - :math:`\mathbf{T}(t)`
     - Tangent vector
     - Direction of motion in deformation space
   * - :math:`\mathbf{N}(t)`
     - Normal vector
     - Direction of curvature
   * - :math:`\mathbf{B}(t)`
     - Binormal vector
     - Osculating plane orientation

Extracted Physical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP decomposer extracts the following physical parameters:

.. list-table:: SPP Extracted Quantities
   :header-rows: 1
   :widths: 20 15 15 50

   * - Name
     - Symbol
     - Units
     - Description
   * - ``G_cage``
     - :math:`G_{\text{cage}}`
     - Pa
     - Apparent cage modulus; elastic strength of microstructure
   * - ``sigma_sy``
     - :math:`\sigma_{y,\text{static}}`
     - Pa
     - Static yield stress; stress at cage rupture
   * - ``sigma_dy``
     - :math:`\sigma_{y,\text{dynamic}}`
     - Pa
     - Dynamic yield stress; stress at cage reformation
   * - ``gamma_y``
     - :math:`\gamma_y`
     - —
     - Yield strain; strain at cage rupture
   * - ``I3_I1_ratio``
     - :math:`I_3/I_1`
     - —
     - Third harmonic ratio; nonlinearity measure
   * - ``S_factor``
     - :math:`S`
     - —
     - Strain stiffening factor
   * - ``T_factor``
     - :math:`T`
     - —
     - Shear thickening factor
   * - ``Gp_t_mean``
     - :math:`\langle G'_t \rangle`
     - Pa
     - Cycle-averaged instantaneous storage modulus
   * - ``Gpp_t_mean``
     - :math:`\langle G''_t \rangle`
     - Pa
     - Cycle-averaged instantaneous loss modulus

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G_cage (Cage Modulus)**:
   Instantaneous elastic slope at zero stress, measuring cage stiffness independently of deformation history.
   *For graduate students*: :math:`G_{\text{cage}} = d\sigma/d\gamma|_{\sigma=0}` represents true microstructural stiffness. Unlike :math:`G'(\gamma_0)` which decreases at large amplitudes due to cage breaking, :math:`G_{\text{cage}}` remains constant. In small-amplitude limit: :math:`\lim_{\gamma_0 \to 0} G_{\text{cage}} = \lim_{\gamma_0 \to 0} G'`. Relates to plateau modulus in entangled polymers or colloidal cage elastic modulus. For hard-sphere colloids: :math:`G_{\text{cage}} \sim k_BT/(\pi a^3) \phi` where a is particle radius, :math:`\phi` is volume fraction.
   *For practitioners*: Extract from Lissajous slope at origin or SPP decomposition. Typical ranges: colloidal glasses 10-100 Pa, microgel pastes 100-1000 Pa, polymer gels 1000-10000 Pa. Higher G_cage = stiffer microstructural cages.

:math:`\sigma_{sy}` **,** :math:`\sigma_{dy}` **(Static and Dynamic Yield Stresses)**:
   Two distinct yield stresses: :math:`\sigma_{sy}` at strain reversal (:math:`\dot{\gamma}` = 0, :math:`\gamma` = ±\ :math:`\gamma_0`), :math:`\sigma_{dy}` at rate reversal (:math:`\dot{\gamma}` = 0, :math:`\gamma` ≠ 0).
   *For graduate students*: Power-law scaling: :math:`\sigma_{sy}(\gamma_0) \sim \gamma_0^{n_{sy}}`, :math:`\sigma_{dy}(\gamma_0) \sim \gamma_0^{n_{dy}}` with n ≈ 0.2 typical for hard-sphere colloids (Rogers et al. 2011). Ratio :math:`\sigma_{sy}/\sigma_{dy}` ~ 2-3 quantifies cage reformation kinetics during momentary rest at strain extrema. Connects to Herschel-Bulkley yield via extrapolation: :math:`\sigma_{y,HB} \approx \sigma_{dy}(\gamma_0 \to 0)`.
   *For practitioners*: Static yield (higher) = restart stress from rest. Dynamic yield (lower) = continuous flow stress. Large ratio (> 3) indicates strong thixotropy or fast cage reformation. Extract from SPP at multiple :math:`\gamma_0`.

**I_3/I_1 (Third Harmonic Ratio)**:
   Ratio of third to first harmonic intensities, quantifying degree of nonlinearity in LAOS.
   *For graduate students*: Fourier decomposition: :math:`\sigma(t) = \sum_n I_n \sin(n\omega t + \phi_n)`. :math:`I_3/I_1` measures deviation from linear viscoelasticity. For small :math:`\gamma_0`: :math:`I_3/I_1 \sim \gamma_0^2`. Saturation at large :math:`\gamma_0` indicates full yielding. Connects to Pipkin space: :math:`I_3/I_1` contours reveal nonlinear regime boundaries.
   *For practitioners*: I_3/I_1 < 0.01 (linear regime), 0.01-0.1 (nonlinear), > 0.1 (strongly nonlinear/yielding). Use to define transition amplitude :math:`\gamma_{0,NL}` where material enters nonlinear regime.

**S, T (Stiffening/Thickening Factors)**:
   Dimensionless factors quantifying intracycle strain stiffening (S) and shear thickening (T).
   *For graduate students*: Defined via rates dG'_t/:math:`d\gamma` (S-factor) and dG''_t/d(:math:`\dot{\gamma}/\omega`) (T-factor) per Rogers (2017). Positive values indicate stiffening/thickening, negative indicate softening/thinning. Transitions between regimes reveal yielding sequence.
   *For practitioners*: Use S and T to identify intracycle phases: elastic extension (S ≈ 0, T ≈ 0), yielding (S < 0, T changes sign), flow (S ≈ 0, T < 0), reformation (S > 0, T > 0).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SPP Decomposition Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - G_cage = 10-100 Pa
     - Soft cages
     - Colloidal glasses (:math:`\phi` ~ 0.6), microgels
     - Low yield stress, gentle handling needed
   * - G_cage = 100-1000 Pa
     - Moderate stiffness
     - Concentrated emulsions, foams, pastes
     - Moderate processing stresses
   * - G_cage > 1000 Pa
     - Stiff cages
     - Polymer gels, filled elastomers
     - High processing stresses, robust structures
   * - :math:`\sigma_{sy}/\sigma_{dy}` < 2
     - Weak thixotropy, fast reformation
     - Simple yield-stress fluids, weak gels
     - Predictable restart behavior
   * - :math:`\sigma_{sy}/\sigma_{dy}` = 2-3
     - Moderate thixotropy
     - Concentrated suspensions, soft glasses
     - Time-dependent yield, moderate memory
   * - :math:`\sigma_{sy}/\sigma_{dy}` > 3
     - Strong thixotropy, slow reformation
     - Aged pastes, strongly thixotropic gels
     - Long rest-time dependence
   * - I_3/I_1 < 0.01
     - Linear regime
     - All materials at small :math:`\gamma_0`
     - SAOS applicable
   * - I_3/I_1 = 0.01-0.1
     - Nonlinear transition
     - Near yielding
     - MAOS/LAOS needed
   * - I_3/I_1 > 0.1
     - Strongly nonlinear, full yielding
     - LAOS at large :math:`\gamma_0`
     - Yielding cycle clearly observed

**sigma_sy/sigma_dy (Yield Stresses)**:
   - **Physical meaning**: Stress thresholds for cage rupture and reformation
   - **Rate dependence**: Both scale as :math:`\dot{\gamma}_0^{0.2}`
   - **Ratio**: :math:`\sigma_{y,\text{static}} / \sigma_{y,\text{dynamic}} \approx 2.75`

**I3/I1 (Third Harmonic Ratio)**:
   - **Physical meaning**: Degree of nonlinearity in stress response
   - **Linear regime**: :math:`I_3/I_1 \to 0`
   - **Nonlinear regime**: :math:`I_3/I_1 \sim 0.1 - 0.3`

Validity and Assumptions
------------------------

**Assumptions:**

1. **Time-domain analysis**: Raw stress-strain waveforms required (not harmonic coefficients)
2. **Periodic steady state**: Material has reached oscillatory steady state
3. **Weak thixotropy**: Structure equilibrates within each cycle
4. **Homogeneous deformation**: No wall slip or shear banding

**Valid test modes:**

- LAOS (Large Amplitude Oscillatory Shear)
- Strain-controlled oscillation with time-resolved waveform capture

**Limitations:**

**Strong thixotropy**:
   For materials where structure changes dramatically over cycles, instantaneous properties
   may not be well-defined. Rule of thumb: SPP works when structural relaxation time
   :math:`\tau_s > 2\pi/\omega`.

**Noise sensitivity near zero stress**:
   Ratios become sensitive when :math:`\sigma \approx 0`. Apply smoothing or exclude
   these regions from analysis.

**Cycle selection effects**:
   Early cycles may contain transients; late cycles may show fatigue. Always verify
   consistency across multiple cycles.

SPP vs Fourier/Chebyshev
------------------------

.. list-table:: Method Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - SPP (Time-Domain)
     - Fourier/Chebyshev
   * - Domain
     - Time (instantaneous properties)
     - Frequency (harmonic coefficients)
   * - Basis functions
     - Physical processes (discrete)
     - Orthonormal harmonics (infinite)
   * - Yield stress
     - Direct extraction
     - Indirect from :math:`G_M, G_L`
   * - Higher harmonics
     - Explained by power-law flow
     - Individual physical meanings unclear
   * - Best for
     - Yield-stress fluids, physical mechanisms
     - Material fingerprinting, literature comparison

SPP is **qualitatively consistent** with Fourier-Chebyshev analysis [5]_. Higher harmonics
observed in Fourier spectra are explained by the power-law flow response post-yielding.
For a comprehensive review of LAOS methods including both SPP and Fourier approaches,
see Hyun et al. (2011) [6]_.

SPP vs FTC: Complementary Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recent studies (2022-2024) have demonstrated that SPP and Fourier-Transform
coupled with Chebyshev decomposition (FTC) methods provide complementary
insights for nonlinear rheology characterization:

**SPP Advantages:**

- Time-resolved instantaneous parameters at any point in the cycle
- Direct physical interpretation (cage modulus, yield stresses)
- No assumption about response symmetry
- Applicable to partial cycles and transients
- Reveals intracycle yielding sequence

**FTC Advantages:**

- Global measures averaged over complete cycles
- Direct connection to Pipkin diagram coordinates
- Established elastic/viscous Chebyshev coefficients (:math:`e_n, v_n`)
- Broader literature base for material fingerprinting
- Standard nonlinearity metrics (:math:`S, T, I_3/I_1`)

**Recommended Workflow:**

1. Apply FTC for initial nonlinearity characterization (:math:`I_3/I_1`, S, T factors)
2. Apply SPP for intracycle process identification
3. Combine insights: FTC defines "what" is nonlinear, SPP reveals "when" and "why"

Recent comparative studies have applied both methods to food rheology (doughs, gels),
3D printing materials, and emulsions, finding that the combined approach provides
more complete material understanding than either method alone [11]_ [12]_.

Usage
-----

Input Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~

For reliable SPP analysis, input data should meet these specifications:

**Data Format:**

- Time series: :math:`t, \gamma(t), \dot{\gamma}(t), \sigma(t)`
- Columns: Time [s], Strain [-], Strain Rate [1/s], Stress [Pa]
- If strain rate unavailable, will be differentiated from strain (requires periodic data)

**Sampling Requirements:**

- For Fourier method: Integer number of periods with even points per period
- For numerical method: Uniform time spacing recommended
- Typical: 200-1000 points per period for smooth derivatives

**Unit Conversions:**

The decomposer accepts standard SI units. For non-SI data, apply conversion factors:

.. code-block:: python

   # Example: strain in % to strain units
   data['strain'] = data['strain_percent'] / 100

   # Example: stress in kPa to Pa
   data['stress'] = data['stress_kPa'] * 1000

**Data Quality Checklist:**

- Remove startup transients (typically first 1-2 cycles)
- Ensure steady alternance before analysis
- Check for wall slip or shear banding artifacts
- Verify data spans complete oscillation cycles
- Confirm stress-strain Lissajous forms closed loops

Basic SPP Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.transforms import SPPDecomposer
   from rheojax.io.readers import auto_read

   # Load LAOS waveform data
   data = auto_read("laos_waveforms.csv")

   # Create SPP decomposer with Rogers-parity defaults
   decomposer = SPPDecomposer(
       omega=1.0,              # Angular frequency (rad/s)
       gamma_0=1.0,            # Strain amplitude
       n_harmonics=39,         # Rogers default (odd)
       step_size=8,            # 8-point stencil
       use_numerical_method=True,
   )

   # Apply decomposition
   result = decomposer.transform(data)
   spp_results = decomposer.get_results()

   # Access extracted quantities
   G_cage = spp_results['G_cage']
   sigma_sy = spp_results['sigma_sy']
   sigma_dy = spp_results['sigma_dy']
   I3_I1 = spp_results['I3_I1_ratio']

   print(f"Cage modulus: {G_cage:.1f} Pa")
   print(f"Static yield stress: {sigma_sy:.1f} Pa")
   print(f"Dynamic yield stress: {sigma_dy:.1f} Pa")
   print(f"Yield stress ratio: {sigma_sy/sigma_dy:.2f}")

Cycle Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Skip startup transients (cycles 0-1), analyze cycles 2-5
   decomposer = SPPDecomposer(
       omega=1.0,
       gamma_0=1.0,
       start_cycle=2,
       end_cycle=5,
   )

   result = decomposer.transform(data)
   print(f"Cycles analyzed: {decomposer.results_['cycles_analyzed']}")

Amplitude Sweep Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline.spp import SPPAmplitudeSweepPipeline
   import numpy as np

   # Define amplitude sweep
   gamma_0_values = np.logspace(-2, 1, 20)

   # Run batch SPP analysis
   pipeline = SPPAmplitudeSweepPipeline(omega=1.0)
   results = pipeline.run_sweep(
       data_files=amplitude_files,
       gamma_0_values=gamma_0_values,
   )

   # Extract amplitude-dependent quantities
   G_cage_vs_gamma = results['G_cage']
   sigma_sy_vs_gamma = results['sigma_sy']

Bayesian Yield Stress Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SPPYieldStress
   from rheojax.core import RheoData

   # Prepare amplitude-dependent yield stress data
   rheo_data = RheoData(
       x=gamma_0_values,
       y=sigma_sy_values,
       test_mode='spp_yield',
   )

   # Fit power-law model with Bayesian inference
   model = SPPYieldStress()
   model.fit(rheo_data)

   result = model.fit_bayesian(rheo_data, num_warmup=1000, num_samples=2000)

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"K = {intervals['K']['mean']:.2f} Pa")
   print(f"n = {intervals['n']['mean']:.3f}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # 1. Lissajous with yield points
   ax = axes[0, 0]
   ax.plot(strain, stress, 'b-', lw=1.5)
   ax.axhline(sigma_sy, color='r', ls='--', label=r'$\sigma_{y,static}$')
   ax.axhline(sigma_dy, color='g', ls=':', label=r'$\sigma_{y,dynamic}$')
   ax.set_xlabel(r'Strain $\gamma$')
   ax.set_ylabel(r'Stress $\sigma$ (Pa)')
   ax.legend()

   # 2. Instantaneous moduli
   ax = axes[0, 1]
   ax.plot(phase, Gp_t, 'b-', label=r"$G'_t$")
   ax.plot(phase, Gpp_t, 'r--', label=r"$G''_t$")
   ax.set_xlabel('Phase in cycle')
   ax.set_ylabel('Modulus (Pa)')
   ax.legend()

   # 3. Cole-Cole spiral
   ax = axes[1, 0]
   ax.plot(Gp_t, Gpp_t, 'b-')
   ax.set_xlabel(r"$G'_t$ (Pa)")
   ax.set_ylabel(r"$G''_t$ (Pa)")
   ax.set_aspect('equal')

   # 4. Amplitude dependence
   ax = axes[1, 1]
   ax.loglog(gamma_0, sigma_sy, 'ro-', label='Static')
   ax.loglog(gamma_0, sigma_dy, 'bs-', label='Dynamic')
   ax.set_xlabel(r'$\gamma_0$')
   ax.set_ylabel('Yield stress (Pa)')
   ax.legend()

   plt.tight_layout()

Fitting Guidance
----------------

Rogers-Parity Defaults
~~~~~~~~~~~~~~~~~~~~~~

RheoJAX uses MATLAB/Rogers-aligned defaults for SPP analysis:

.. list-table:: Default Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Notes
   * - ``n_harmonics``
     - 39
     - Odd harmonics; matches SPPplus v2.1
   * - ``step_size``
     - 8
     - 8-point 4th-order finite difference stencil
   * - ``num_mode``
     - 2
     - Periodic/looped differentiation
   * - ``yield_tolerance``
     - 0.02
     - Tolerance for yield point detection

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Common Issues
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Cause
     - Solution
   * - Noisy :math:`G'_t, G''_t`
     - Raw data noise propagates
     - Apply Savitzky-Golay smoothing before SPP
   * - Missing yield points
     - Tolerance too strict
     - Increase ``yield_tolerance`` to 0.05
   * - Inconsistent across cycles
     - Startup transients or fatigue
     - Adjust ``start_cycle``, ``end_cycle``
   * - :math:`G_{\text{cage}}` varies
     - Strong thixotropy
     - Consider time-resolved protocols instead

See Also
--------

- :doc:`spp_yield_stress` — Power-law model for SPP-extracted yield stresses
- :doc:`../sgr/sgr_conventional` — Soft Glassy Rheology model
- :doc:`../../transforms/owchirp` — Fourier-based LAOS analysis
- :doc:`../../user_guide/03_advanced_topics/spp_analysis` — Comprehensive SPP user guide

API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.SPPDecomposer`

References
----------

.. [1] Rogers, S. A., Erwin, B. M., Vlassopoulos, D., & Cloitre, M. "A sequence of
   physical processes determined and quantified in LAOS: Application to a yield stress
   fluid." *J. Rheol.* **55**, 435-458 (2011). https://doi.org/10.1122/1.3544591

.. [2] Rogers, S. A. "In search of physical meaning: defining transient parameters for
   nonlinear viscoelasticity." *Rheol. Acta* **56**, 501-525 (2017).
   https://doi.org/10.1007/s00397-017-1008-1

.. [3] Rogers, S. A. "A sequence of physical processes determined and quantified in
   LAOS: An instantaneous local 2D/3D approach." *J. Rheol.* **56**, 1129-1151 (2012).
   https://doi.org/10.1122/1.4726083

.. [4] Rogers, S. A. & Lettinga, M. P. "A sequence of physical processes determined
   and quantified in large-amplitude oscillatory shear (LAOS): Application to theoretical
   nonlinear models." *J. Rheol.* **56**, 1-25 (2012). https://doi.org/10.1122/1.3662962

.. [5] Ewoldt, R., Hosoi, A., & McKinley, G. "New measures for characterizing nonlinear
   viscoelasticity in large amplitude oscillatory shear." *J. Rheol.* **52**, 1427-1458 (2008).

.. [6] Hyun, K., et al. "A review of nonlinear oscillatory shear tests: Analysis and
   application of large amplitude oscillatory shear (LAOS)." *Prog. Polym. Sci.* **36**,
   1697-1753 (2011).

.. [7] Ewoldt, R. H. and McKinley, G. H. "Mapping thixo-elasto-visco-plastic behavior."
   *Rheol. Acta* **56**, 195-210 (2017). https://doi.org/10.1007/s00397-017-1001-8

.. [8] Cole, K. S. and Cole, R. H. "Dispersion and absorption in dielectrics I.
   Alternating current characteristics." *J. Chem. Phys.* **9**, 341-351 (1941).
   https://doi.org/10.1063/1.1750906

.. [9] Bonn, D., Denn, M. M., Berthier, L., Divoux, T., and Manneville, S. "Yield stress
   materials in soft condensed matter." *Rev. Mod. Phys.* **89**, 035005 (2017).
   https://doi.org/10.1103/RevModPhys.89.035005

.. [10] van Puyvelde, P., Velankar, S., and Mewis, J. "Rheology and morphology of
   compatibilized polymer blends." In *Polymer Blends Handbook*, Springer, 421-626 (2014).
   https://doi.org/10.1007/978-94-007-6064-6_7

.. [11] Le, T. D., et al. "LAOS rheological characterization of food materials:
   Comparison of Fourier-transform and SPP analysis methods." *Food Research
   International* **165**, 112478 (2023). https://doi.org/10.1016/j.foodres.2023.112478

.. [12] Duvarci, O. C., et al. "Comparison of LAOS analysis methods: SPP versus
   Fourier-Chebyshev decomposition for wheat flour doughs." *Food Hydrocolloids*
   **128**, 107570 (2022). https://doi.org/10.1016/j.foodhyd.2022.107570
