Maxwell-Isotropic-Kinematic Hardening (MIKH)
============================================

Quick Reference
---------------

- **Use when:** Thixotropic elasto-viscoplastic materials with stress overshoot, Bauschinger effect, thixotropic hysteresis

- **Parameters:** 11 (G, η, C, γ_dyn, m, σ_y0, Δσ_y, τ_thix, Γ, η_inf, μ_p)

- **Key equation:** :math:`\frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma` (Maxwell viscoelasticity with plasticity)

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

- **Material examples:** Drilling fluids, greases, waxy crude oil, thixotropic cements, structured emulsions

.. currentmodule:: rheojax.models.ikh.mikh

.. autoclass:: MIKH
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
   * - σ
     - Pa
     - Deviatoric stress (elasto-plastic component)
   * - α
     - Pa
     - Backstress (kinematic hardening variable)
   * - λ
     - –
     - Structural parameter (0 = destructured, 1 = structured)
   * - γ̇
     - 1/s
     - Total shear rate
   * - γ̇ᵖ
     - 1/s
     - Plastic shear rate
   * - σ_y
     - Pa
     - Current yield stress (depends on λ)
   * - ξ
     - Pa
     - Relative stress (ξ = σ - α)


Overview
--------

The **MIKH** (Maxwell-Isotropic-Kinematic Hardening) model is a comprehensive
thixotropic elasto-viscoplastic constitutive equation developed by Dimitriou & McKinley (2014)
for complex fluids like waxy crude oil. It combines:

1. **Maxwell viscoelasticity**: Stress relaxation via η (Maxwell viscosity)
2. **Kinematic hardening**: Backstress evolution (Armstrong-Frederick type)
3. **Isotropic hardening**: Yield stress evolution via structural parameter λ
4. **Viscous background**: High-shear Newtonian contribution (η_inf)

The model captures:

- **Stress overshoot** in startup flow
- **Bauschinger effect** (easier reverse flow after forward loading)
- **Thixotropic loops** (history-dependent stress-strain curves)
- **Yield stress aging** (rest-time dependence)
- **Flow curve** with shear-thinning and yield stress


Theoretical Background
----------------------

Historical Context
~~~~~~~~~~~~~~~~~~

The MIKH model emerges from the synthesis of three traditionally separate fields:

1. **Classical Plasticity**: The theory of plastic deformation in metals, particularly
   the work of Prager and Armstrong-Frederick on kinematic hardening to capture the
   Bauschinger effect.

2. **Thixotropy**: The time-dependent rheology first systematically studied by
   Freundlich and colleagues in the 1920s-30s, formalized through structural kinetics
   approaches (Goodeve 1939, Moore 1959).

3. **Yield Stress Materials**: The Herschel-Bulkley framework and its extensions
   to time-dependent yield stress materials (de Souza Mendes & Thompson 2019).

The unification of these frameworks by Dimitriou & McKinley provides a thermodynamically
consistent model capable of describing the complex behavior of materials like waxy crude
oil, drilling muds, and structured colloidal suspensions.

Material Class: Thixotropic Elasto-Viscoplastic Fluids (TEvp)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TEvp materials exhibit several characteristic behaviors:

**1. Yield Stress:** Below a critical stress σ_y, the material responds elastically
(reversibly). Above σ_y, plastic flow occurs irreversibly.

**2. Thixotropy:** The material's structure—and hence its properties—depend on
mechanical history. Under shear, microstructure breaks down (destructuring);
at rest, it recovers (restructuring). The structural parameter λ ∈ [0, 1] tracks
this state:

- λ = 1: Fully structured (maximum yield stress, maximum elasticity)
- λ = 0: Fully destructured (minimum yield stress)

**3. Viscoelasticity:** Even in the elastic regime, the material exhibits stress
relaxation over time due to microstructural rearrangements.

**4. Kinematic Hardening:** Under cyclic loading, the material exhibits directional
memory—the Bauschinger effect. This is captured through the backstress α, which
shifts the yield surface in stress space.

Physical Interpretation of the Microstructure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In waxy crude oils, the microstructure consists of:

- **Wax crystals** that form a space-spanning network below the gelation temperature
- **Inter-crystalline bonds** (van der Waals forces, crystal interlocking) that
  provide mechanical integrity
- **Continuous oil phase** that acts as the suspending medium

The structural parameter λ represents the **fraction of intact inter-crystalline bonds**.
When sheared, bonds break (destructuring); at rest, thermal fluctuations allow
bonds to reform (restructuring). This microscopic picture motivates the kinetic
equations for λ evolution.

For other TEvp materials:

- **Drilling fluids**: λ represents the organization of clay platelets and polymer chains
- **Colloidal gels**: λ represents the fraction of intact colloidal bonds
- **Greases**: λ represents the organization of thickener fibers


Thermokinematic Memory (FIKH Framework)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Fractal IKH (FIKH)** framework (Geri et al. 2017) extends the MIKH model
to account for temperature-dependent microstructure. This is critical for
materials like waxy crude oils where thermal history determines the precipitated
wax content.

**Effective Volume Fraction:**

The wax volume fraction depends on temperature history:

.. math::

   \phi(T) = \phi_{\max} \cdot f(\Delta T)

where :math:`\Delta T = T - T_{wax}` is the subcooling below the wax appearance
temperature. The function :math:`f(\Delta T)` captures the precipitation kinetics.

**Thermokinematic Memory:**

The material "remembers" its thermal history through:

1. The precipitated wax morphology (cooling rate affects crystal size/shape)
2. The equilibrium connectivity :math:`\xi_{eq}(T)` at each temperature
3. The effective volume fraction :math:`\phi(T)` determining available wax content

This leads to a modified structure evolution equation:

.. math::

   \frac{d\xi}{dt} = k_1(\xi_{eq} - \xi) - k_2 \xi |\dot{\gamma}^p|

where the equilibrium connectivity :math:`\xi_{eq}(T)` replaces the constant target value of 1.

**Parameter Scaling with Microstructure:**

In the FIKH framework, macroscopic parameters depend on both temperature (via φ)
and structure (via ξ):

.. math::

   \mu_p(T, \xi) &= \mu_{p,\infty}(T) \cdot \left[1 - \frac{\xi}{\xi_c}\right]^{-[\eta]\phi} \\
   \sigma_y(\xi) &= \sigma_{y,0} + \Delta\sigma_y \cdot \xi^n \\
   C(\xi) &= C_0 \cdot \xi^p

where :math:`\xi_c` is the critical connectivity for jamming, :math:`[\eta]` is the
intrinsic viscosity, and :math:`n, p` are scaling exponents related to fractal dimension.


Fractal Microstructure Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The structure parameter λ can be interpreted as the **normalized fractal connectivity**
:math:`\xi \in [0, 1]` of the microstructural network.

**Fractal Connectivity:**

In a fractal network (e.g., wax crystal aggregates), the connectivity ξ represents
the fraction of intact bonds relative to a fully percolated network:

.. math::

   \xi = \frac{N_{bonds}}{N_{bonds,max}}

The fractal nature of colloidal aggregates leads to power-law scaling of mechanical
properties with connectivity:

- **Yield stress**: :math:`\sigma_y \propto \xi^n` with :math:`n \approx 2-3`
- **Elastic modulus**: :math:`G \propto \xi^{d_f/(d-d_f)}` where :math:`d_f` is fractal dimension

**The Avalanche Effect:**

The nonlinear coupling between structure (ξ) and yield stress (σ_y) leads to
an important dynamic phenomenon—**delayed yielding** or the "avalanche effect":

1. Under stress :math:`\sigma < \sigma_y(\xi_0)`, the material creeps slowly
2. Slow creep causes gradual structure breakdown: :math:`\xi \downarrow`
3. As ξ decreases, :math:`\sigma_y(\xi)` drops, accelerating breakdown
4. Eventually, catastrophic yielding occurs when :math:`\sigma > \sigma_y(\xi)`

This positive feedback loop explains why thixotropic materials can appear
solid for extended periods before suddenly flowing.

**Relationship to Percolation:**

Near the percolation threshold, the network connectivity follows:

.. math::

   \xi \propto (p - p_c)^\beta

where :math:`p` is the bond occupation probability and :math:`p_c \approx 0.5` for 3D networks.
The critical exponent β depends on network topology.


Thermodynamic Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~

The MIKH model can be derived within the **Gurtin-Fried-Anand** thermomechanical
framework, ensuring:

1. **Frame invariance**: Constitutive equations are objective (independent of observer)
2. **Second law compliance**: Dissipation inequality is satisfied
3. **Energy balance**: Clear separation of stored (elastic) and dissipated energy

The key thermodynamic quantities are:

- **Free energy**: :math:`\psi(\gamma^e, \alpha, \lambda)` storing elastic energy
- **Dissipation**: :math:`\mathcal{D} = \sigma \dot{\gamma}^p + X \dot{\alpha} + Y \dot{\lambda} \geq 0`

where X and Y are thermodynamic forces conjugate to the internal variables.
This framework guarantees that the model respects fundamental physics while
allowing complex phenomenology.


Physical Foundations
--------------------

Maxwell-Like Framework
~~~~~~~~~~~~~~~~~~~~~~

The MIKH model uses a Maxwell-like viscoelastic element as its foundation.
The Maxwell element consists of a spring (modulus G) in series with a dashpot
(viscosity η), giving a relaxation time τ = η/G:

.. math::

   \frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma

The first term represents elastic loading minus plastic flow. The second term
represents viscoelastic relaxation with characteristic time τ = η/G.

**Physical interpretation:**

- At short times (t ≪ τ): Elastic response dominates, σ ≈ G·γ
- At long times (t ≫ τ): Viscous flow, σ → 0 under constant strain
- The Maxwell element captures the liquid-like long-time behavior of structured fluids

Kinematic Hardening (Armstrong-Frederick)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kinematic hardening is a plasticity concept that accounts for the Bauschinger effect:
when a material is deformed plastically in one direction, subsequent yield in the
opposite direction occurs at a lower stress than the initial yield stress.

The backstress α represents the "center" of the yield surface in stress space.
As plastic deformation accumulates, α evolves according to the Armstrong-Frederick
(AF) law:

.. math::

   d\alpha = C \cdot d\gamma^p - \gamma_{dyn} |\alpha|^{m-1} \alpha |d\gamma^p|

**Term 1 (Hardening):** C·dγ_p

- The backstress increases proportionally to plastic strain increment
- C is the kinematic hardening modulus [Pa]
- This creates a "memory" of the plastic deformation direction

**Term 2 (Dynamic Recovery):** -γ_dyn·\|α\|^(m-1)·α·\|dγ_p\|

- Limits backstress saturation (prevents unbounded growth)
- γ_dyn controls recovery rate
- m controls nonlinearity (m = 1 is linear, m > 1 accelerates recovery at high α)
- Recovery is proportional to \|dγ_p\|, so it only occurs during plastic flow

**Steady-state backstress:** At steady plastic flow:

.. math::

   \alpha_{ss} = \frac{C}{\gamma_{dyn}} \cdot \text{sign}(\dot{\gamma}^p)

The ratio C/γ_dyn determines the maximum backstress magnitude.

Isotropic Hardening (Thixotropy)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The yield stress evolves with a structural parameter λ ∈ [0, 1]:

.. math::

   \sigma_y(\lambda) = \sigma_{y,0} + \Delta\sigma_y \cdot \lambda

- σ_y,0: Minimal yield stress when fully destructured (λ = 0)
- Δσ_y: Additional yield stress from structure
- σ_y,max = σ_y,0 + Δσ_y: Maximum yield stress when fully structured (λ = 1)

The structure evolves according to a first-order kinetic equation:

.. math::

   \frac{d\lambda}{dt} = \frac{1-\lambda}{\tau_{thix}} - \Gamma \lambda |\dot{\gamma}^p|

**Term 1 (Buildup):** (1-λ)/τ_thix

- Structure recovers toward λ = 1 with characteristic time τ_thix
- At rest (γ̇ᵖ = 0): λ(t) = 1 - (1 - λ_0)·exp(-t/τ_thix)
- Physical origin: Brownian motion, thermal fluctuations allow bond reformation

**Term 2 (Breakdown):** Γ·λ·\|γ̇ᵖ\|

- Structure breaks down proportionally to plastic strain rate
- Γ is the breakdown efficiency coefficient
- Physical origin: Mechanical work breaks inter-particle bonds

**Steady-state structure:** At constant shear rate:

.. math::

   \lambda_{ss} = \frac{1/\tau_{thix}}{1/\tau_{thix} + \Gamma|\dot{\gamma}|}


Mathematical Formulation
------------------------

Core Equations
~~~~~~~~~~~~~~

**Stress decomposition:**

.. math::

   \sigma_{total} = \sigma + \eta_{\infty} \dot{\gamma}

The total stress consists of the elasto-plastic contribution σ and a purely
viscous background η_∞·γ̇. The latter represents the suspending fluid's viscosity.

**Yield condition:**

.. math::

   f = |\xi| - \sigma_y(\lambda) \leq 0 \quad \text{where} \quad \xi = \sigma - \alpha

The material yields when the relative stress \|ξ\| = \|σ - α\| exceeds the current
yield stress σ_y(λ). The backstress α shifts the yield surface in stress space.

**Plastic flow rule (Perzyna regularization):**

.. math::

   \dot{\gamma}^p = \frac{\langle f \rangle}{\mu_p} \cdot \text{sign}(\xi)

where ⟨·⟩ denotes Macaulay brackets (positive part function):

.. math::

   \langle f \rangle = \max(f, 0) = \begin{cases} f & \text{if } f > 0 \\ 0 & \text{if } f \leq 0 \end{cases}

The Perzyna regularization parameter μ_p [Pa·s] controls how sharply the material
transitions from elastic to plastic behavior. Small μ_p gives rate-independent
plasticity; larger μ_p smooths the transition.

Complete System of ODEs
~~~~~~~~~~~~~~~~~~~~~~~

The MIKH model comprises three coupled ordinary differential equations:

.. math::

   \frac{d\sigma}{dt} &= G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma \\
   \frac{d\alpha}{dt} &= C\dot{\gamma}^p - \gamma_{dyn}|\alpha|^{m-1}\alpha|\dot{\gamma}^p| \\
   \frac{d\lambda}{dt} &= \frac{1-\lambda}{\tau_{thix}} - \Gamma\lambda|\dot{\gamma}^p|

With the plastic flow rate determined by:

.. math::

   \dot{\gamma}^p = \frac{\langle |\sigma - \alpha| - \sigma_y(\lambda) \rangle}{\mu_p} \cdot \text{sign}(\sigma - \alpha)

Two Formulations
~~~~~~~~~~~~~~~~

The MIKH model uses two numerical formulations depending on the experimental protocol:

**1. Maxwell ODE Formulation** (for creep/relaxation)

Suitable for stress-controlled or strain-relaxation experiments where the
full viscoelastic response is needed:

.. code-block:: text

   # State: [σ, α, λ]
   dσ/dt = G(γ̇ - γ̇ᵖ) - (G/η)σ
   dα/dt = C·γ̇ᵖ - γ_dyn·|α|^(m-1)·α·|γ̇ᵖ|
   dλ/dt = (1-λ)/τ_thix - Γ·λ·|γ̇ᵖ|

This formulation uses adaptive ODE integration (Diffrax) for accurate
time-stepping of the coupled system.

**2. Return Mapping Formulation** (for startup/LAOS)

Suitable for strain-driven experiments with incremental time stepping:

.. code-block:: python

   # Given strain increment Δγ:
   1. Elastic predictor: σ_trial = σ_n + G·Δγ
   2. Check yield: f = |σ_trial - α_n| - σ_y(λ_n)
   3. If f > 0: Radial return with AF correction
   4. Update λ AFTER stress (timing-consistent)

The return mapping algorithm provides:

- Exact stress update (radial return to yield surface)
- Implicit treatment of the plastic corrector
- Efficient JAX scan implementation for long time series

**Critical timing fix:** The structure parameter λ is updated AFTER the stress
calculation, using the plastic strain rate from the current step. This ensures
consistency with the physical picture where structure responds to the applied
deformation.

Steady-State Analysis
~~~~~~~~~~~~~~~~~~~~~

At steady state (d/dt = 0), the flow curve follows from the equilibrium conditions.

**Structure balance:**

.. math::

   \lambda_{ss} = \frac{k_1}{k_1 + k_2|\dot{\gamma}|}

where k₁ = 1/τ_thix and k₂ = Γ.

**Steady-state stress:**

.. math::

   \sigma_{ss} = \sigma_{y,0} + \Delta\sigma_y \cdot \lambda_{ss} + \eta_{\infty}|\dot{\gamma}|

Substituting the structure balance:

.. math::

   \sigma_{ss}(\dot{\gamma}) = \sigma_{y,0} + \frac{\Delta\sigma_y}{1 + \Gamma\tau_{thix}|\dot{\gamma}|} + \eta_{\infty}|\dot{\gamma}|

This produces the characteristic shear-thinning flow curve:

- **Low shear rate (γ̇ → 0):** σ → σ_y,0 + Δσ_y (structured yield stress)
- **High shear rate (γ̇ → ∞):** σ → σ_y,0 + η_∞·γ̇ (linear viscous)

Governing Equations
-------------------

The complete MIKH system comprises three coupled ODEs (see Mathematical Formulation for details):

.. math::

   \frac{d\sigma}{dt} &= G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma \\
   \frac{d\alpha}{dt} &= C\dot{\gamma}^p - \gamma_{dyn}|\alpha|^{m-1}\alpha|\dot{\gamma}^p| \\
   \frac{d\lambda}{dt} &= \frac{1-\lambda}{\tau_{thix}} - \Gamma\lambda|\dot{\gamma}^p|

With plastic flow rate:

.. math::

   \dot{\gamma}^p = \frac{\langle |\sigma - \alpha| - \sigma_y(\lambda) \rangle}{\mu_p} \cdot \text{sign}(\sigma - \alpha)

Validity and Assumptions
------------------------

**Valid for:**

- **Thixotropic elasto-viscoplastic fluids**: Waxy crude oils, drilling muds, greases, structured emulsions
- **Materials with Bauschinger effect**: Easier reverse flow after forward loading
- **Yield stress evolution**: Rest-time dependent yield stress
- **Moderate shear rates**: Below onset of turbulence or flow instabilities

**Assumptions:**

- **Single structural parameter λ**: One-dimensional structure kinetics (no multi-scale structure)
- **Isotropic yielding**: von Mises-like yield criterion (no anisotropy)
- **Affine deformation**: No spatial gradients (homogeneous flow)
- **Incompressible**: No density changes
- **Isothermal**: No temperature effects

**Not appropriate for:**

- **Multi-timescale thixotropy**: Use :doc:`ml_ikh` instead
- **Shear banding**: Requires spatial extension (1D or 2D)
- **Viscoelastic effects** dominating over plasticity: Use Maxwell or Oldroyd-B models
- **High-frequency oscillations**: Limited by quasi-static assumption

What You Can Learn
------------------

From fitting Modified IKH to experimental data, you can extract insights about thixotropy, kinematic hardening (Bauschinger effect), and structural evolution in elasto-viscoplastic materials.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**λ (Structure Parameter)**:
   Dimensionless internal variable (0 ≤ λ ≤ 1) quantifying microstructural integrity.
   *For graduate students*: λ represents fraction of intact bonds/aggregates. Evolution: dλ/dt = (1-λ)/τ_thix - Γλ|γ̇^p|. At steady state: λ_ss = 1/(1 + Γτ_thix|γ̇|). Couples to yield stress via σ_y(λ) = σ_y,0 + Δσ_y·λ, capturing aging-induced hardening.
   *For practitioners*: λ = 1 (fully aged, maximum strength) vs λ = 0 (fully broken down, minimum strength). Measure indirectly via yield stress recovery tests. Materials with long τ_thix retain flow history.

**α (Kinematic Backstress)**:
   Internal stress representing directional anisotropy from flow-induced microstructure.
   *For graduate students*: Armstrong-Frederick kinematic hardening: dα/dt = C·γ̇^p - γ_dyn|α|^(m-1)α|γ̇^p|. Produces Bauschinger effect (easier reverse yielding). At steady state: α_ss = C/γ_dyn. Ratio (σ_y - 2α_ss)/σ_y quantifies asymmetry.
   *For practitioners*: Measure via reverse flow tests. High C → strong directional memory, pronounced Bauschinger effect. Typical for waxy crude oils, fiber suspensions.

**τ_thix (Thixotropic Rebuilding Time)**:
   Timescale for structural recovery at rest.
   *For graduate students*: First-order kinetics for aging: λ → 1 with time constant τ_thix. Sets width of hysteresis loops in up-down flow ramps. For thermally-activated processes, τ_thix ~ τ₀exp(ΔE_build/k_BT).
   *For practitioners*: Extract from rest-time dependent startup tests or step-strain recovery. Fast aging (τ_thix < 10 s) vs slow aging (τ_thix > 100 s). Critical for pumping restart protocols.

**Γ (Breakdown Coefficient)**:
   Efficiency of shear-induced destructuring (units: inverse shear rate).
   *For graduate students*: Controls shear-thinning: λ_ss = 1/(1 + Γτ_thix|γ̇|). High Γ → rapid breakdown, low Γ → persistent structure. Connects to flow curve via σ_ss(γ̇) = σ_y,0 + Δσ_y/(1 + Γτ_thix|γ̇|) + η_∞|γ̇|.
   *For practitioners*: Fit from flow curve curvature. Γτ_thix ~ 1 at characteristic shear rate where structure is half-broken.

**C, γ_dyn, m (Kinematic Hardening Parameters)**:
   Control backstress evolution and Bauschinger effect magnitude.
   *For graduate students*: C is hardening modulus, γ_dyn is dynamic recovery rate, m is recovery exponent (m=1 linear, m>1 nonlinear). Armstrong-Frederick model with m=1 widely used. Steady α_ss = C/γ_dyn independent of m.
   *For practitioners*: Identify from cyclic loading or reverse flow tests. C/γ_dyn sets saturation backstress (typ. 10-50% of σ_y).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Modified IKH Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - τ_thix < 10 s, Γτ_thix < 1
     - Fast aging, weak shear-thinning
     - Soft gels, cosmetics, paints
     - Rapid recovery, moderate thixotropy
   * - τ_thix = 10-100 s, Γτ_thix = 1-10
     - Moderate aging, strong shear-thinning
     - Drilling muds, greases, emulsions
     - Pronounced thixotropy, history-dependent
   * - τ_thix > 100 s, Γτ_thix > 10
     - Slow aging, extreme shear-thinning
     - Waxy crude oils, cement pastes
     - Long memory, pumping challenges
   * - C/γ_dyn < 0.1σ_y
     - Weak Bauschinger effect
     - Isotropic gels, simple colloids
     - Symmetric yielding
   * - C/γ_dyn > 0.3σ_y
     - Strong Bauschinger effect
     - Waxy crude oils, fiber suspensions
     - Directional flow history, asymmetric yielding

- **Connection to SAOS**: G ≈ G' (storage modulus) at high frequency

**5. Stress Overshoot Magnitude**

- **Overshoot ratio**: (σ_max - σ_ss) / σ_ss
- Controlled by interplay of G, C, and λ₀ (initial structure)
- **Physical signature**: Thixotropic materials show overshoot; purely viscoplastic do not

**6. Yield Stress Aging**

- **Time dependence**: σ_y(t_rest) = σ_y,0 + Δσ_y·(1 - exp(-t_rest/τ_thix))
- **Aging rate**: 1/τ_thix
- **Maximum recoverable yield stress**: σ_y,0 + Δσ_y

Dimensionless Groups
--------------------

The model behavior can be characterized by several dimensionless numbers:

**Weissenberg Number (Wi):**

.. math::

   Wi = \dot{\gamma} \tau_{thix}

Ratio of shear rate to structure buildup rate. Wi ≫ 1 means structure breaks
down faster than it recovers (destructured regime).

**Deborah Number (De):**

.. math::

   De = \frac{\tau}{\tau_{exp}} = \frac{\eta/G}{t_{exp}}

Ratio of relaxation time to experimental time scale. De ≫ 1 means elastic
response dominates; De ≪ 1 means viscous response dominates.

**Bingham Number (Bi):**

.. math::

   Bi = \frac{\sigma_y}{\eta_{\infty}\dot{\gamma}}

Ratio of yield stress to viscous stress. Bi ≫ 1 means yield-dominated;
Bi ≪ 1 means viscous-dominated.

**Structure Number (Sn):**

.. math::

   Sn = \Gamma \tau_{thix}

Relative efficiency of breakdown versus buildup. Sn ≫ 1 means structure
breaks down efficiently under shear.


Industrial Applications
-----------------------

The MIKH model was developed for and validated against industrial thixotropic
materials. This section provides application-specific guidance with typical
parameter ranges from field studies and laboratory characterization.

Waxy Crude Oil Pipeline Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MIKH model was originally developed for waxy crude oils (Dimitriou & McKinley 2014),
making it the reference model for pipeline flow assurance applications.

**Pipeline Restart After Shutdown:**

When a pipeline shuts down, wax precipitates and forms a gel network.
Key parameter ranges from field applications:

- **τ_thix = 100-10,000 s**: Long aging times for gelled pipelines
- **σ_y,0 + Δσ_y = 50-500 Pa**: Gel strength depends on cooling rate and rest time
- **Γ·τ_thix > 10**: Extreme shear-thinning for pipeline restart

**Engineering implications:**

- Restart pressure scales with σ_y(t_rest) where t_rest can span hours to days
- Monitor thermokinematic memory (FIKH framework) for temperature-cycled systems
- Stress overshoot during restart indicates incomplete gel breakdown

**Cold Flow Assurance:**

For subsea pipelines below wax appearance temperature (WAT):

- Continuous low-shear flow prevents complete gelation
- Target operating shear rate: γ̇ > 1/(Γ·τ_thix) to maintain destructured state
- Thermal cycling protocols require FIKH framework with temperature-dependent φ

Drilling Fluids and Muds
~~~~~~~~~~~~~~~~~~~~~~~~

Water-based drilling fluids exhibit pronounced IKH behavior due to clay platelet
aggregation and polymer interactions.

**Typical parameter ranges:**

- **τ_thix = 1-100 s**: Faster recovery than crude oils due to smaller particles
- **σ_y,0 = 5-15 Pa**: API barite suspension requirements for cutting transport
- **C/γ_dyn ≈ 0.1-0.3 σ_y**: Moderate Bauschinger effect from clay orientation

**Borehole Stability:**

- Gel strength must exceed cutting particle buoyancy: σ_y > Δρ·g·d_particle
- Thixotropic recovery prevents fluid loss into formation during connections
- API 6rpm/300rpm readings map to MIKH parameters via flow curve fitting

**Pump Circulation Restart:**

After pipe connections or trips:

- Initial startup pressure ∝ σ_y(t_connection) where t_connection ~ 30-300 s
- Stress overshoot magnitude indicates gel breakdown efficiency
- Design pumping rate to achieve γ̇ > 1/(Γ·τ_thix) throughout annulus

Greases and Lubricants
~~~~~~~~~~~~~~~~~~~~~~

Grease consistency (NLGI grades) correlates with MIKH parameters through the
yield stress and thixotropic timescales.

**NLGI Grade Correlation:**

.. list-table::
   :widths: 15 25 30 30
   :header-rows: 1

   * - NLGI Grade
     - Application
     - σ_y (Pa)
     - τ_thix (s)
   * - 000-00
     - Centralized systems
     - 50-150
     - 1-10
   * - 0-1
     - Enclosed gears
     - 100-300
     - 5-30
   * - 2
     - General purpose
     - 200-500
     - 10-100
   * - 3-6
     - High-consistency
     - 400-2000
     - 50-500

**Bearing Startup Applications:**

- Stress overshoot magnitude indicates grease breakdown risk under initial loading
- Kinematic hardening (C parameter) critical for reversing loads in oscillating bearings
- Channeling behavior: permanent structure breakdown when γ̇ peak > critical value

**Kinematic Hardening in Reversing Loads:**

The Bauschinger effect (controlled by C/γ_dyn ratio) is particularly important
for greases in oscillating applications:

.. code-block:: python

   # Reverse flow simulation for oscillating bearing
   model = MIKH()
   model.parameters.set_value("C", 100.0)      # Kinematic hardening
   model.parameters.set_value("gamma_dyn", 5.0) # Recovery rate
   # Backstress saturation: α_max = C/γ_dyn = 20 Pa

   # Simulate LAOS to observe Bauschinger effect
   sigma_laos = model.predict_laos(t, gamma_0=0.5, omega=1.0)

Thixotropic Cements and Pastes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Cementitious materials exhibit structure evolution from early hydration and
particle flocculation.

**Pumping and Placement:**

- **τ_thix = 10-1000 s**: Depending on formulation and admixtures
- Structure recovery must match placement window for self-leveling vs. vertical stability
- High Γ values enable rapid breakdown for pumping, but may compromise build-up

**Self-Leveling vs. Non-Sag Behavior:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Application
     - Parameter Requirement
     - Physical Interpretation
   * - Self-leveling floors
     - Low τ_thix, high Γ
     - Fast breakdown, moderate recovery
   * - Vertical surfaces
     - High τ_thix, moderate Γ
     - Slow breakdown, strong recovery
   * - 3D printing
     - Very high σ_y,0 + Δσ_y
     - Immediate yield on deposition

**Yield Stress Aging for Formwork Removal:**

The time-dependent yield stress evolution determines safe formwork removal:

.. math::

   \sigma_y(t_{cure}) = \sigma_{y,0} + \Delta\sigma_y \cdot (1 - e^{-t_{cure}/\tau_{thix}})

For critical structural applications, τ_thix must be characterized at the
curing temperature to predict strength development.


Parameters
----------

.. list-table::
   :widths: 12 10 10 68
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``G``
     - G
     - Pa
     - Elastic shear modulus. Controls initial stiffness and stress overshoot amplitude.
       Typical range: 10² - 10⁶ Pa for structured fluids.
   * - ``eta``
     - η
     - Pa·s
     - Maxwell viscosity. Relaxation time τ = η/G. Large values = elastic solid.
       Setting η → ∞ recovers rate-independent plasticity.
   * - ``C``
     - C
     - Pa
     - Kinematic hardening modulus. Controls backstress buildup rate.
       Larger C = stronger Bauschinger effect.
   * - ``gamma_dyn``
     - γ_dyn
     - –
     - Dynamic recovery parameter. Limits backstress saturation.
       Saturation: α_max = C/γ_dyn.
   * - ``m``
     - m
     - –
     - AF exponent (typically 1.0). Controls nonlinearity of recovery.
       m = 1: linear AF; m > 1: accelerated recovery at high α.
   * - ``sigma_y0``
     - σ_y0
     - Pa
     - Minimal yield stress (fully destructured state, λ=0).
       This is the "static" yield stress after prolonged shearing.
   * - ``delta_sigma_y``
     - Δσ_y
     - Pa
     - Structural yield stress contribution. σ_y,max = σ_y0 + Δσ_y when λ=1.
       Controls strength of aging effect.
   * - ``tau_thix``
     - τ_thix
     - s
     - Thixotropic rebuilding time. Time for structure recovery at rest.
       Typical: 10⁻¹ - 10⁴ s depending on material.
   * - ``Gamma``
     - Γ
     - –
     - Breakdown coefficient. Efficiency of shear-induced destructuring.
       Higher Γ = faster breakdown under shear.
   * - ``eta_inf``
     - η_∞
     - Pa·s
     - High-shear viscosity (solvent contribution).
       Dominates at high shear rates where structure is destroyed.
   * - ``mu_p``
     - μ_p
     - Pa·s
     - Plastic viscosity (Perzyna regularization parameter).
       Small μ_p = sharp yield; large μ_p = smoothed transition.


Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Flow curve first**: Fit σ_y0, Δσ_y, τ_thix, Γ, η_inf from steady-state data
2. **Startup second**: Fix flow curve params, fit G, C, γ_dyn from transient
3. **Relaxation/creep**: Fine-tune η (Maxwell viscosity)

Protocol Selection
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Protocol
     - Best for
   * - ``flow_curve``
     - Steady-state parameters (σ_y0, Δσ_y, η_inf)
   * - ``startup``
     - Elasticity (G) and hardening (C, γ_dyn)
   * - ``relaxation``
     - Maxwell viscosity (η)
   * - ``creep``
     - Combined viscoelastic-plastic response
   * - ``laos``
     - Full nonlinear characterization

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Issue
     - Solution
   * - Poor flow curve fit
     - Check σ_y0 initialization; ensure γ̇ range spans structure transition
   * - No stress overshoot
     - Increase G or decrease Γ (maintain structure during startup)
   * - Overshoot too sharp
     - Increase μ_p (plastic viscosity regularization)
   * - No Bauschinger effect
     - Increase C (hardening) or decrease γ_dyn (less recovery)
   * - Stress doesn't relax
     - Decrease η (Maxwell viscosity); check τ = η/G vs experiment time


Parameter Estimation Methods
----------------------------

The MIKH model's 11 parameters span different experimental timescales and
phenomena. Advanced estimation methods improve identifiability and uncertainty
quantification beyond basic curve fitting.

Sequential Fitting Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sequential approach exploits the separation of timescales in the MIKH model
to improve parameter identifiability:

**Stage 1: Flow Curve (Steady State)**

From flow curve data σ(γ̇), fit the steady-state parameters:

- σ_y,0, Δσ_y (yield stress bounds)
- η_∞ (high-shear viscosity)
- Γ·τ_thix product (controls shear-thinning curvature)

.. code-block:: python

   from rheojax.models import MIKH

   model = MIKH()

   # Fix elastic/hardening params, fit thixotropic
   model.parameters.freeze(['G', 'C', 'gamma_dyn', 'eta', 'mu_p'])
   model.fit(gamma_dot, sigma_ss, test_mode='flow_curve')

   # Extract fitted values
   sigma_y0_fit = model.parameters.get_value('sigma_y0')
   delta_sigma_y_fit = model.parameters.get_value('delta_sigma_y')

**Stage 2: Startup Transients**

From startup stress overshoot σ(t; γ̇_0), fit:

- G (controls initial slope and overshoot magnitude)
- C, γ_dyn (kinematic hardening, Bauschinger effect)
- τ_thix (recovery timescale, now separated from Γ)

.. code-block:: python

   # Unfreeze elastic/hardening parameters
   model.parameters.unfreeze(['G', 'C', 'gamma_dyn'])

   # Fit startup data with flow curve params fixed
   model.fit(t_startup, sigma_startup, test_mode='startup')

**Stage 3: Relaxation/Creep**

From stress relaxation σ(t)|_{γ=const}, fit:

- η (Maxwell viscosity, determines τ_relax = η/G)
- μ_p (Perzyna regularization, yield transition sharpness)

.. code-block:: python

   model.parameters.unfreeze(['eta', 'mu_p'])
   model.fit(t_relax, sigma_relax, test_mode='relaxation')

Multi-Start Global Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For datasets spanning wide parameter ranges or with multiple local minima,
use multi-start optimization:

.. code-block:: python

   # Multi-start with parallel execution
   model.fit(
       X, y,
       use_multi_start=True,
       n_starts=5,           # Number of random initializations
       parallel=True         # ThreadPoolExecutor for 3-5x speedup
   )

**When to use multi-start:**

- Flow curves spanning >3 decades of shear rate
- Combined protocol fitting (flow + startup + relaxation)
- Initial fits show residual structure (systematic over/under-prediction)
- Materials with unusual parameter combinations (e.g., very high τ_thix)

**Global optimization for multi-modal problems:**

.. code-block:: python

   from rheojax.utils.optimization import nlsq_optimize_global

   # Global search for challenging parameter landscapes
   result = nlsq_optimize_global(
       objective_fn,
       initial_params,
       bounds=param_bounds
   )

Bayesian Inference with MCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For uncertainty quantification, use NumPyro NUTS with NLSQ warm-start:

.. code-block:: python

   # Stage 1: Point estimate (fast, provides good initialization)
   model.fit(X, y, test_mode='startup')

   # Stage 2: Bayesian inference (4 chains for reliable R-hat)
   result = model.fit_bayesian(
       X, y,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,         # Production: 4 chains for R-hat diagnostics
       seed=42               # Reproducibility
   )

   # Check convergence diagnostics
   print(f"R-hat: {result.r_hat}")   # Target: <1.01
   print(f"ESS: {result.ess}")       # Target: >400

**Prior Selection Guidance:**

The choice of priors significantly affects Bayesian inference for the MIKH model:

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Parameter
     - Recommended Prior
     - Rationale
   * - τ_thix
     - LogNormal(μ=log(10), σ=1)
     - Spans 1-100 s; positive, heavy-tailed
   * - Γ
     - HalfNormal(σ=10)
     - Positive breakdown coefficient
   * - σ_y,0, Δσ_y
     - TruncatedNormal or Uniform
     - Material-dependent bounds
   * - G
     - LogNormal(μ=log(1000), σ=1)
     - Typical modulus range for soft materials
   * - C/γ_dyn ratio
     - LogNormal(μ=log(10), σ=0.5)
     - Backstress saturation constraint

Regularization for Ill-Conditioned Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When parameters are correlated (common for G-C, τ_thix-Γ pairs), use:

**1. Tikhonov Regularization:**

Add penalty λ‖θ‖² to objective function to stabilize optimization:

.. code-block:: python

   from rheojax.utils.optimization import nlsq_curve_fit

   result = nlsq_curve_fit(
       model_fn, x, y, params,
       regularization='tikhonov',
       lambda_reg=1e-4  # Regularization strength
   )

**2. Bounds Tightening:**

Physically constrain parameter ranges based on material knowledge:

.. code-block:: python

   # Constrain based on material class
   model.parameters.set_bounds('tau_thix', (1.0, 1000.0))  # Drilling fluid
   model.parameters.set_bounds('sigma_y0', (5.0, 50.0))   # API spec range

**3. Combined Protocol Fitting:**

Fitting multiple test modes simultaneously reduces parameter correlation
by providing orthogonal constraints:

.. code-block:: python

   # Combined protocol fitting (pseudo-code pattern)
   # Concatenate datasets with appropriate weighting
   X_combined = combine_protocols(flow_data, startup_data)
   weights = [1.0, 2.0]  # Emphasize transient data

   model.fit(X_combined, y_combined, weights=weights)

Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~

Identify which parameters most influence predictions to guide experimental design:

**Local Sensitivity (Jacobian-based):**

.. code-block:: python

   import jax
   import jax.numpy as jnp

   def compute_sensitivity(model, X, param_names):
       """Compute local parameter sensitivities."""

       def prediction_fn(param_values):
           for name, val in zip(param_names, param_values):
               model.parameters.set_value(name, val)
           return model.predict(X)

       # Get current parameter values
       param_values = jnp.array([
           model.parameters.get_value(name) for name in param_names
       ])

       # Compute Jacobian: ∂σ/∂θ
       jacobian = jax.jacobian(prediction_fn)(param_values)
       return jacobian

**Sensitivity interpretation:**

- High sensitivity: Parameter strongly influences predictions (well-constrained by data)
- Low sensitivity: Parameter weakly influences predictions (may be poorly identifiable)
- Correlated sensitivities: Parameters are coupled (consider reparameterization)

**Practical recommendations:**

1. Compute sensitivities at the fitted parameter values
2. Focus experimental design on regimes where target parameters have high sensitivity
3. For τ_thix: use startup/recovery data at t ~ τ_thix
4. For G: use early-time startup data (t ≪ τ_thix)
5. For Γ: use flow curve data near γ̇ ~ 1/(Γ·τ_thix)


Usage
-----

The MIKH model is available via:

.. code-block:: python

   from rheojax.models import MIKH

**Common workflows**:

1. **Flow curve fitting**: Determine σ_y0, Δσ_y, η_inf from steady-state data
2. **Startup fitting**: Extract G, C, γ_dyn from transient stress overshoot
3. **Creep/relaxation**: Constrain η (Maxwell viscosity) and μ_p (plastic viscosity)
4. **Bayesian inference**: Quantify uncertainty in thixotropic timescales

**Integration with Pipeline**:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Fluent API for complete workflow
   (Pipeline()
    .load('startup_data.csv', x_col='time', y_col='stress')
    .fit_nlsq('mikh', test_mode='startup')
    .fit_bayesian(num_samples=2000)
    .plot_trace()
    .save('mikh_results.hdf5'))

Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import MIKH

   # Initialize model
   model = MIKH()

   # Set parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)
   model.parameters.set_value("eta_inf", 0.1)

Flow Curve
~~~~~~~~~~

.. code-block:: python

   # Predict steady-state flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model.predict_flow_curve(gamma_dot)

Startup Shear
~~~~~~~~~~~~~

.. code-block:: python

   # Predict startup response
   t = np.linspace(0, 10, 200)
   sigma_startup = model.predict_startup(t, gamma_dot=1.0)

Stress Relaxation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict relaxation from initial stress
   t = np.linspace(0, 100, 200)
   sigma_relax = model.predict_relaxation(t, sigma_0=100.0)

Creep
~~~~~

.. code-block:: python

   # Predict creep under constant stress
   t = np.linspace(0, 100, 200)
   strain = model.predict_creep(t, sigma_applied=50.0)

LAOS
~~~~

.. code-block:: python

   # Large amplitude oscillatory shear
   t = np.linspace(0, 20, 500)
   sigma_laos = model.predict_laos(t, gamma_0=1.0, omega=1.0)

Fitting
~~~~~~~

.. code-block:: python

   # Fit to experimental data
   model.fit(gamma_dot, sigma_data, test_mode="flow_curve")

   # Bayesian inference with NLSQ warm-start
   result = model.fit_bayesian(
       X_data, sigma_data,
       num_warmup=1000, num_samples=2000,
       test_mode="startup"
   )


Relation to Other Models
------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model
     - Relationship to MIKH
   * - Herschel-Bulkley
     - MIKH reduces to HB at steady state without kinematic hardening (C=0)
       and with power-law η_inf
   * - Saramito EVP
     - Similar framework but without kinematic hardening; uses Oldroyd-B
       instead of Maxwell for viscoelasticity
   * - de Souza Mendes TEvp
     - Different structure kinetics formulation; uses viscosity-based
       approach rather than yield stress
   * - :doc:`ml_ikh`
     - Multi-mode extension with N parallel structural elements


References
----------

.. [1] Dimitriou, C. J. and McKinley, G. H. "A comprehensive constitutive law for
   waxy crude oil: a thixotropic yield stress fluid." *Soft Matter*, 10(35),
   6619-6644 (2014). https://doi.org/10.1039/c4sm00578c

.. [2] Geri, M., Venkatesan, R., Sambath, K., and McKinley, G. H. "Thermokinematic
   memory and the thixotropic elasto-viscoplasticity of waxy crude oils."
   *Journal of Rheology*, 61(3), 427-454 (2017). https://doi.org/10.1122/1.4978259

.. [3] Saramito, P. "A new elastoviscoplastic model based on the Herschel-Bulkley
   viscoplastic model." *Journal of Non-Newtonian Fluid Mechanics*, 158, 154-161
   (2009). https://doi.org/10.1016/j.jnnfm.2008.08.001

.. [4] de Souza Mendes, P. R. and Thompson, R. L. "Time-dependent yield stress
   materials." *Annual Review of Fluid Mechanics*, 51, 421-449 (2019).
   https://doi.org/10.1146/annurev-fluid-010518-040305

.. [5] Armstrong, P. J. and Frederick, C. O. "A mathematical representation of the
   multiaxial Bauschinger effect." *CEGB Report RD/B/N731* (1966).

.. [6] Mewis, J. and Wagner, N. J. "Thixotropy." *Advances in Colloid and Interface
   Science*, 147-148, 214-227 (2009). https://doi.org/10.1016/j.cis.2008.09.005

.. [7] Chaboche, J. L. "Constitutive equations for cyclic plasticity and cyclic
   viscoplasticity." *International Journal of Plasticity*, 5(3), 247-302 (1989).
   https://doi.org/10.1016/0749-6419(89)90015-6

.. [8] Larson, R. G. and Wei, Y. "A review of thixotropy and its rheological modeling."
   *Journal of Rheology*, 63(3), 477-501 (2019). https://doi.org/10.1122/1.5055031

.. [9] Dullaert, K. and Mewis, J. "A structural kinetics model for thixotropy."
   *Journal of Non-Newtonian Fluid Mechanics*, 139(1-2), 21-30 (2006).
   https://doi.org/10.1016/j.jnnfm.2006.06.002

.. [10] Prager, W. "A new method of analyzing stresses and strains in work-hardening
   plastic solids." *Journal of Applied Mechanics*, 23, 493-496 (1956).
   https://doi.org/10.1115/1.4011389

See Also
--------

- :doc:`ml_ikh` — Multi-mode extension for distributed thixotropic timescales
- :doc:`/models/dmt/dmt_local` — Alternative thixotropic formulation (de Souza Mendes-Thompson)
- :doc:`/models/fluidity/saramito` — Fluidity-Saramito EVP models with thixotropy
- :doc:`/user_guide/03_advanced_topics/index` — Advanced thixotropic modeling workflows
