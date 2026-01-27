.. _model-tnt-stretch-creation:

===========================================================
TNT Stretch-Creation (Enhanced Reformation) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
===============

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Use when:**
     - Networks where bond creation rate depends on chain stretch (strain-enhanced crosslinking, stretch-activated association)
   * - **Parameters:**
     - 4 parameters: :math:`G` (Pa), :math:`\tau_b` (s), :math:`\kappa` (creation enhancement factor, dimensionless 0-5), :math:`\eta_s` (Pa·s)
   * - **Key equation:**
     - Creation rate: :math:`k_{on}(S) = \frac{1}{\tau_b} \left(1 + \kappa \left(\text{tr}(S) - 3\right)\right)`
   * - **Test modes:**
     - All 6: FLOW_CURVE, OSCILLATION, STARTUP, RELAXATION, CREEP, LAOS
   * - **Material examples:**
     - Strain-crystallizing rubbers, mechanophore-activated networks, strain-induced gelation systems, adaptive polymers
   * - **Key characteristics:**
     - Strain hardening through enhanced crosslink formation, positive feedback under extension

Notation Guide
==============

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Symbol
     - Units
     - Meaning
   * - :math:`S`
     - dimensionless
     - Dimensionless conformation tensor (second moment)
   * - :math:`G`
     - Pa
     - Plateau modulus (network elasticity)
   * - :math:`\tau_b`
     - s
     - Breakage timescale (bond lifetime at equilibrium)
   * - :math:`\kappa`
     - dimensionless
     - Creation enhancement factor (0-5 typical range)
   * - :math:`k_{on}(S)`
     - s⁻¹
     - Stretch-dependent bond creation rate
   * - :math:`k_{off}`
     - s⁻¹
     - Bond breakage rate (constant, :math:`1/\tau_b`)
   * - :math:`\text{tr}(S)`
     - dimensionless
     - Trace of conformation tensor (mean-square chain stretch)
   * - :math:`\eta_s`
     - Pa·s
     - Solvent viscosity (Newtonian background)
   * - :math:`\kappa_{flow}`
     - s⁻¹
     - Velocity gradient tensor
   * - :math:`D`
     - s⁻¹
     - Rate-of-strain tensor (symmetric part of :math:`\kappa_{flow}`)
   * - :math:`\sigma`
     - Pa
     - Cauchy stress tensor
   * - :math:`I`
     - dimensionless
     - Identity tensor

Overview
========

The Stretch-Creation variant of the Tanaka-Edwards transient network model introduces a positive feedback mechanism between chain stretch and bond formation rate. Unlike the base model where creation and breakage rates are constant, this variant recognizes that some polymer networks form new crosslinks more readily when chains are already extended.

Physical Motivation
-------------------

Several physical systems exhibit strain-enhanced crosslinking:

1. **Strain crystallization in natural rubber**: Stretched polymer chains align and crystallize, creating additional physical crosslinks. This is the classic mechanism responsible for the remarkable strength and toughness of natural rubber.

2. **Mechanophore-activated networks**: Chemical groups that become reactive when subjected to mechanical force. Chain extension activates binding sites that were previously inaccessible.

3. **Stretch-induced gelation**: Some polymer solutions gel more rapidly under extension as chain stretching promotes intermolecular contacts and association.

4. **Adaptive hydrogels**: Biomimetic networks where mechanical loading triggers crosslink formation, similar to biological tissue remodeling.

Distinction from Elastic Stiffening
------------------------------------

The stretch-creation mechanism is fundamentally different from FENE-like elastic stiffening:

- **FENE models**: Chain force increases nonlinearly with extension due to finite extensibility (elastic stiffening)
- **Stretch-creation**: Number of crosslinks increases with extension (kinetic stiffening)

Both lead to strain hardening, but through different mechanisms. The stretch-creation variant can be combined with FENE to capture both effects simultaneously.

Positive Feedback and Stability
--------------------------------

The coupling :math:`k_{on} \propto (1 + \kappa(\text{tr}(S) - 3))` creates positive feedback:

1. External force stretches chains → :math:`\text{tr}(S)` increases
2. Enhanced creation rate → more crosslinks form
3. More crosslinks → higher stress under same strain
4. Can lead to further stretching if load-controlled

This positive feedback is stabilized by:

- Constant breakage rate :math:`k_{off} = 1/\tau_b`
- Flow-induced relaxation (advection, rotation)
- Finite :math:`\kappa` values prevent runaway

The balance between enhanced creation and constant breakage sets the steady-state network structure.

Physical Foundations
====================

Stretch-Activated Association
------------------------------

The core physical idea is that bond formation probability increases with chain extension. Several microscopic mechanisms can lead to this behavior:

**Entropy-driven exposure**: Coiled chains hide binding sites; stretching exposes them.

**Alignment-induced association**: Extended chains align parallel to each other, promoting intermolecular contacts and hydrogen bonding or van der Waals attraction.

**Force-activated chemistry**: Mechanical force lowers activation barriers for certain chemical reactions (mechanophore activation).

The mean-field coupling :math:`k_{on} \propto \text{tr}(S)` assumes that bond creation rate scales with the average mean-square chain stretch. More sophisticated treatments could use the full distribution of chain stretches, but the mean-field approximation captures the essential physics.

Strain Crystallization Physics
-------------------------------

Natural rubber's exceptional mechanical properties arise from strain-induced crystallization. At rest, polymer chains are amorphous. Under extension:

1. Chains align along the stretching direction
2. Aligned segments pack into crystalline lamellae
3. Crystallites act as additional physical crosslinks
4. Network modulus increases dramatically

Flory (1947) first recognized this mechanism. The stretch-creation variant phenomenologically captures this effect through :math:`\kappa > 0`.

Mechanophore Networks
---------------------

Modern mechanochemistry enables design of polymers with force-sensitive chemical groups (mechanophores). Examples include:

- Spiropyran that isomerizes under tension
- Cyclobutane rings that open to form reactive radicals
- Hidden thiols that become exposed for disulfide exchange

These systems can be engineered so that chain extension activates crosslinking chemistry, directly realizing the stretch-creation coupling.

Mean-Field Coupling Assumption
-------------------------------

The model assumes all chains see the same average stretch :math:`\text{tr}(S)/3`. In reality:

- Chain length polydispersity creates distribution of stretches
- Network heterogeneity (defects, entanglements) causes local variations
- Flow history affects different chain populations differently

Despite these simplifications, the mean-field coupling :math:`\kappa(\text{tr}(S) - 3)` provides a tractable framework that captures qualitative behavior and can guide experimental design.

Stability and Bounds
---------------------

The linear coupling :math:`k_{on} = (1/\tau_b)(1 + \kappa(\text{tr}(S) - 3))` has no built-in saturation. For very large :math:`\kappa` or :math:`\text{tr}(S)`, the creation rate can become arbitrarily large, potentially leading to numerical instability or unphysical predictions.

Practical considerations:

- :math:`\kappa \leq 2` for most fits (mild to moderate enhancement)
- :math:`\kappa > 5` may require additional damping or saturation terms
- Ensure steady-state solutions exist (balance with breakage)
- For extreme extensions, combine with FENE to bound :math:`\text{tr}(S)`

Governing Equations
===================

Modified Conformation Tensor Evolution
---------------------------------------

The key modification is the stretch-dependent creation rate:

.. math::

   k_{on}(S) = \frac{1}{\tau_b} \left(1 + \kappa \left(\text{tr}(S) - 3\right)\right)

The conformation tensor :math:`S` evolves according to:

.. math::

   \frac{dS}{dt} = \kappa_{flow} \cdot S + S \cdot \kappa_{flow}^T - k_{off}(S - S_{eq})

where :math:`k_{off} = 1/\tau_b` and the equilibrium tensor :math:`S_{eq}` is determined by the balance of creation and destruction.

At equilibrium (rest), :math:`dS/dt = 0` implies:

.. math::

   k_{on}(S_{eq}) S_{eq} = k_{off} I

For the base model (:math:`\kappa = 0`), :math:`k_{on} = k_{off}`, so :math:`S_{eq} = I`.

For stretch-creation (:math:`\kappa > 0`), the equilibrium is modified:

.. math::

   \left(1 + \kappa(\text{tr}(S_{eq}) - 3)\right) S_{eq} = I

This is a nonlinear equation for :math:`S_{eq}`. For small :math:`\kappa`, :math:`S_{eq} \approx I` (perturbation). For large :math:`\kappa`, :math:`S_{eq}` can deviate significantly.

However, in the current implementation, the creation term is written as:

.. math::

   \frac{dS}{dt} = \kappa_{flow} \cdot S + S \cdot \kappa_{flow}^T - \frac{1}{\tau_b}(S - I) - \frac{\kappa}{\tau_b}(\text{tr}(S) - 3) I

This form separates the base creation-destruction term :math:`-(1/\tau_b)(S - I)` from the stretch-enhancement correction :math:`-(\kappa/\tau_b)(\text{tr}(S) - 3) I`.

Constitutive Equation (Stress)
-------------------------------

The stress tensor is the sum of network and solvent contributions:

.. math::

   \sigma = G(S - I) + 2 \eta_s D

where:

- :math:`G(S - I)` is the elastic network stress (deformation from equilibrium)
- :math:`2\eta_s D` is the Newtonian solvent stress

The modulus :math:`G` is constant (linear elasticity of Gaussian chains). Nonlinearity enters through the evolution of :math:`S`.

Steady Shear Flow
-----------------

For steady simple shear :math:`\dot{\gamma}`, the velocity gradient is:

.. math::

   \kappa_{flow} = \begin{pmatrix} 0 & \dot{\gamma} & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

At steady state, :math:`dS/dt = 0`. The conformation tensor components :math:`S_{xx}`, :math:`S_{xy}`, :math:`S_{yy}` satisfy:

.. math::

   \dot{\gamma} S_{xy} &= \frac{1}{\tau_b}(S_{xx} - 1) + \frac{\kappa}{\tau_b}(\text{tr}(S) - 3)

.. math::

   \dot{\gamma}(S_{yy} - S_{xx}) &= \frac{1}{\tau_b} S_{xy}

.. math::

   0 &= \frac{1}{\tau_b}(S_{yy} - 1) + \frac{\kappa}{\tau_b}(\text{tr}(S) - 3)

where :math:`\text{tr}(S) = S_{xx} + S_{yy} + S_{zz}` and :math:`S_{zz} = 1 + (\kappa/\tau_b)(\text{tr}(S) - 3)`.

The shear stress is:

.. math::

   \sigma_{xy} = G S_{xy} + \eta_s \dot{\gamma}

This is a nonlinear system coupling :math:`S_{xx}, S_{xy}, S_{yy}` through :math:`\text{tr}(S)`. For :math:`\kappa = 0`, the system decouples and reduces to the base model's analytical solution.

For :math:`\kappa > 0`, numerical root-finding is required. The solution exhibits strain hardening: :math:`\sigma_{xy}` increases more steeply with :math:`\dot{\gamma}` compared to :math:`\kappa = 0`.

Small Amplitude Oscillatory Shear (SAOS)
-----------------------------------------

For linearized perturbations around equilibrium :math:`S = I + \epsilon e^{i\omega t}`, the stretch-creation correction :math:`\kappa(\text{tr}(S) - 3)` is second-order in :math:`\epsilon` and does not affect the linear viscoelastic response.

Thus, the complex modulus for SAOS is identical to the base model:

.. math::

   G^* = G' + iG'' = G \frac{i\omega\tau_b}{1 + i\omega\tau_b}

.. math::

   G' = G \frac{(\omega\tau_b)^2}{1 + (\omega\tau_b)^2}

.. math::

   G'' = G \frac{\omega\tau_b}{1 + (\omega\tau_b)^2}

This is a single Maxwell element. The stretch-creation mechanism is invisible in the linear regime.

Startup of Steady Shear
------------------------

Starting from rest (:math:`S(0) = I`), apply :math:`\dot{\gamma}` and integrate:

.. math::

   \frac{dS}{dt} = \kappa_{flow} \cdot S + S \cdot \kappa_{flow}^T - \frac{1}{\tau_b}(S - I) - \frac{\kappa}{\tau_b}(\text{tr}(S) - 3) I

The stress :math:`\sigma_{xy}(t) = G S_{xy}(t) + \eta_s \dot{\gamma}` initially grows as chains stretch, potentially overshoots if :math:`\kappa` enhances the transient buildup, and then decays to the steady-state value.

The magnitude and timing of the overshoot depend on :math:`\kappa`:

- :math:`\kappa = 0`: Standard Tanaka-Edwards overshoot (moderate)
- :math:`\kappa > 0`: Enhanced overshoot due to faster creation during transient stretching
- :math:`\kappa \gg 1`: Pronounced overshoot, potentially delayed relaxation

Stress Relaxation After Cessation
----------------------------------

After stopping flow from steady state, the conformation tensor relaxes:

.. math::

   \frac{dS}{dt} = -\frac{1}{\tau_b}(S - I) - \frac{\kappa}{\tau_b}(\text{tr}(S) - 3) I

The relaxation is no longer single-exponential due to the coupling between :math:`S` components through :math:`\text{tr}(S)`.

For small :math:`\kappa`, the relaxation is approximately exponential with timescale :math:`\tau_b`. For larger :math:`\kappa`, the decay is slower initially (creation resists relaxation) then faster as :math:`\text{tr}(S) \to 3`.

Creep and Recovery
------------------

Under constant stress :math:`\sigma_0`, the conformation tensor evolves with :math:`\kappa_{flow} = \dot{\gamma}(t) e_x \otimes e_y`, where :math:`\dot{\gamma}(t)` is the instantaneous shear rate determined by:

.. math::

   \sigma_0 = G S_{xy}(t) + \eta_s \dot{\gamma}(t)

This is a differential-algebraic system. The stretch-creation coupling modifies the creep compliance curve, especially at long times where steady-state network structure affects terminal flow rate.

Large Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

Under :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, the conformation tensor and stress evolve according to the full nonlinear ODE. The stretch-creation mechanism generates higher harmonics in the stress response.

The third harmonic :math:`I_3/I_1` is enhanced by :math:`\kappa` since strain-induced creation amplifies the stress at large instantaneous strain.

Parameter Table
===============

.. list-table::
   :widths: 15 15 20 15 35
   :header-rows: 1

   * - Parameter
     - Symbol
     - Default
     - Bounds
     - Description
   * - Modulus
     - :math:`G`
     - 1000 Pa
     - (1, 10⁸) Pa
     - Plateau modulus (network elasticity)
   * - Breakage time
     - :math:`\tau_b`
     - 1.0 s
     - (10⁻⁶, 10⁴) s
     - Characteristic bond lifetime at equilibrium
   * - Creation enhancement
     - :math:`\kappa`
     - 0.5
     - (0.0, 5.0)
     - Stretch-creation coupling strength (dimensionless)
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0 Pa·s
     - (0.0, 10⁴) Pa·s
     - Newtonian background viscosity

Parameter Interpretation
========================

Plateau Modulus :math:`G`
--------------------------

- Directly measurable from :math:`G'` at high frequencies (above :math:`1/\tau_b`)
- Related to network strand density: :math:`G = \nu k_B T` where :math:`\nu` is number density of elastically active strands
- Typical range: 100 Pa (weak gels) to 1 MPa (elastomers)

Breakage Timescale :math:`\tau_b`
----------------------------------

- Sets the characteristic relaxation time in the linear regime
- Inverse of the peak in :math:`G''(\omega)`
- For physical networks: microseconds to hours depending on bond energy
- For chemical networks with dynamic covalent bonds: seconds to days

Creation Enhancement :math:`\kappa`
------------------------------------

This is the key parameter that distinguishes the stretch-creation variant.

**Physical interpretation**: Fractional increase in creation rate per unit excess chain stretch.

- :math:`\kappa = 0`: No stretch-creation coupling (base Tanaka-Edwards)
- :math:`\kappa = 0.1 - 0.5`: Mild enhancement, subtle strain hardening
- :math:`\kappa = 1 - 2`: Moderate enhancement, noticeable strain stiffening
- :math:`\kappa > 2`: Strong coupling, pronounced strain hardening and flow instability

**Typical values**:

- Strain-crystallizing rubber: :math:`\kappa \sim 0.5 - 1.5`
- Mechanophore networks: :math:`\kappa \sim 0.2 - 1.0` (depends on activation force)
- Stretch-induced gelation: :math:`\kappa \sim 1 - 3`

**Fitting strategy**: Start with :math:`\kappa = 0` (base model), fit SAOS to get :math:`G, \tau_b`. Then fit nonlinear startup or flow curve with :math:`\kappa > 0` as the only free parameter.

Solvent Viscosity :math:`\eta_s`
---------------------------------

- Often negligible for polymer melts (:math:`\eta_s = 0`)
- Important for solutions and gels (can dominate at high shear rates)
- Sets Newtonian plateau at high :math:`\dot{\gamma}` (above network relaxation)

Validity and Assumptions
=========================

Assumptions
-----------

1. **Gaussian chain statistics**: No finite extensibility (chains can stretch indefinitely). Combine with FENE variant if chains approach full extension.

2. **Mean-field stretch coupling**: All chains see average :math:`\text{tr}(S)/3`. Ignores distribution of chain extensions.

3. **Linear coupling**: :math:`k_{on} \propto (1 + \kappa(\text{tr}(S) - 3))` is the simplest functional form. Real systems may saturate at large stretch.

4. **Affine deformation**: Chains deform with the continuum (no slip, no reptation). Valid for well-crosslinked networks.

5. **Single relaxation time**: All bonds have the same lifetime :math:`\tau_b`. Polydispersity would require spectrum of :math:`\tau_b`.

6. **Incompressibility**: Trace of stress is determined by pressure (not modeled). Only deviatoric stresses matter.

Validity Regime
---------------

The stretch-creation model is physically justified when:

- Bond formation rate increases measurably with chain extension (mechanophore activation, strain crystallization)
- Network is well-connected (percolated, no dangling ends)
- Deformations are not so large that chains reach full extension (:math:`\text{tr}(S) \ll b^2` where :math:`b` is Kuhn length)
- Timescales are slow enough that local equilibration is faster than bond dynamics

Breakdown Scenarios
-------------------

The model breaks down when:

1. **Extreme extension**: :math:`\text{tr}(S) \to b^2` (Kuhn length). Need FENE correction.

2. **Runaway instability**: If :math:`\kappa` is too large, positive feedback can cause numerical blow-up. Watch for :math:`\text{tr}(S) \to \infty`.

3. **High frequency**: Entanglements or glassy modes faster than :math:`1/\tau_b` are not captured.

4. **Nonaffine deformation**: Loosely crosslinked gels or near gelation point may exhibit nonaffine rearrangements.

5. **Saturation neglected**: Real :math:`k_{on}(S)` likely saturates at large stretch. Linear coupling is first-order approximation.

Regimes and Behavior
====================

Linear Viscoelastic Regime
---------------------------

For :math:`\gamma_0 \ll 1` or :math:`\omega\tau_b \gg 1`:

- :math:`G', G''` identical to base model (single Maxwell element)
- :math:`\kappa` is invisible (second-order in strain)
- Crossover frequency :math:`\omega_c = 1/\tau_b`

**Key point**: SAOS alone cannot determine :math:`\kappa`. Need nonlinear tests.

Moderate Strain Regime
-----------------------

For :math:`\gamma_0 \sim 0.1 - 1` or :math:`\text{Wi} = \dot{\gamma}\tau_b \sim 1`:

- Stress begins to exceed linear prediction
- Strain hardening becomes measurable
- :math:`\kappa` controls magnitude of enhancement
- Startup overshoot is amplified

**Signature**: Flow curve :math:`\sigma(\dot{\gamma})` curves upward relative to base model.

Large Strain Regime
--------------------

For :math:`\gamma_0 > 1` or :math:`\text{Wi} > 1`:

- Significant additional stress from enhanced crosslinking
- :math:`\text{tr}(S)` can be much larger than 3
- Positive feedback becomes strong (creation accelerates)
- Risk of numerical issues if :math:`\kappa` too large

**Behavior**: Stress can increase superlinearly with strain, resembling strain hardening in filled rubbers.

Startup Transients
------------------

Upon imposing :math:`\dot{\gamma}`:

1. **Initial loading** (:math:`t \ll \tau_b`): Elastic response, :math:`S` grows affinely
2. **Overshoot** (:math:`t \sim \tau_b`): Competition between stretching and breakage, enhanced by :math:`\kappa > 0`
3. **Decay to steady state** (:math:`t \gg \tau_b`): Network reaches new equilibrium structure

Effect of :math:`\kappa`:

- :math:`\kappa = 0`: Moderate overshoot
- :math:`\kappa \sim 1`: Enhanced overshoot (more chains created during loading)
- :math:`\kappa > 2`: Pronounced overshoot, delayed relaxation

Steady-State Flow
-----------------

At :math:`t \to \infty`, the network reaches a balance between stretch-enhanced creation and constant breakage.

- Low :math:`\text{Wi}`: Newtonian-like (:math:`\sigma \propto \dot{\gamma}`)
- High :math:`\text{Wi}`: Strain hardening (:math:`\sigma \propto \dot{\gamma}^{1+\alpha}` with :math:`\alpha > 0` depending on :math:`\kappa`)

The flow curve :math:`\sigma_{xy}(\dot{\gamma})` is steeper than the base model, indicating enhanced resistance to flow due to more crosslinks forming under deformation.

What You Can Learn
==================

From SAOS Data
--------------

- **Plateau modulus** :math:`G` from high-frequency :math:`G'`
- **Relaxation time** :math:`\tau_b` from :math:`G''` peak or crossover
- **Solvent viscosity** :math:`\eta_s` from high-frequency :math:`G''` tail

**Cannot determine** :math:`\kappa` from SAOS alone (linear regime insensitive to stretch-creation coupling).

From Startup Shear
------------------

- **Overshoot magnitude** sensitive to :math:`\kappa`
- **Time to peak** modified by creation enhancement
- Compare to base model (:math:`\kappa = 0`) to isolate effect

**Strategy**: Fit base model to SAOS, then fit startup with :math:`\kappa` as single free parameter.

From Flow Curves
----------------

- **Strain hardening exponent** reflects :math:`\kappa` magnitude
- **High-rate plateau** (if present) from :math:`\eta_s`
- **Curvature** in log-log plot indicates nonlinear creation kinetics

**Diagnosis**: Upward curvature in :math:`\sigma(\dot{\gamma})` suggests stretch-creation coupling.

From Creep/Recovery
-------------------

- **Steady-state compliance** affected by :math:`\kappa` through modified network structure
- **Recovery shape** nonexponential due to stretch-dependent relaxation
- **Permanent strain** (if any) indicates irreversible bond rearrangement

From LAOS
---------

- **Higher harmonics** :math:`I_3/I_1, I_5/I_1` enhanced by :math:`\kappa`
- **Pipkin diagram** shows expanded nonlinear region compared to base model
- **Lissajous curves** more elliptical (strain stiffening)

Experimental Design
===================

Recommended Test Sequence
--------------------------

1. **SAOS** (0.001 - 100 rad/s): Determine :math:`G, \tau_b, \eta_s` with base model (:math:`\kappa = 0`)

2. **Startup shear** (3-5 rates spanning :math:`\text{Wi} = 0.1 - 10`): Measure overshoot, compare to base model

3. **Steady flow curve** (logarithmic spacing, :math:`\dot{\gamma} = 0.001 - 100` s⁻¹): Quantify strain hardening

4. **LAOS** (2-3 strains :math:`\gamma_0 = 0.1, 0.5, 2.0` at :math:`\omega = 1/\tau_b`): Check nonlinear signatures

5. **Creep and recovery** (optional): Validate time-dependent predictions

Sample Preparation
------------------

- Ensure network is fully formed and equilibrated before testing
- Avoid pre-shear that might change network structure (unless studying thixotropy)
- Temperature control critical (affects :math:`\tau_b` exponentially via Arrhenius)

Control Samples
---------------

To isolate the stretch-creation effect:

- **Chemically crosslinked network** (no bond dynamics): Should show only elastic response
- **Base transient network** (no mechanophore or strain-crystallization): Compare :math:`\kappa = 0` prediction

Avoiding Artifacts
------------------

- **Wall slip**: Use serrated geometries or small gaps
- **Edge fracture**: Stay below critical strain (~100-300% for elastomers)
- **Strain crystallization melting**: Keep temperature above crystalline melting point unless studying that effect
- **Shear heating**: Use small gaps and low frequencies for viscous samples

Data Quality Checks
-------------------

- SAOS: :math:`G', G''` must satisfy Kramers-Kronig relations
- Startup: Repeatability across cycles (network should be reversible)
- Flow curve: No hysteresis between up and down sweeps (unless thixotropic)
- LAOS: Fourier spectrum should decay monotonically with harmonic number

Computational Implementation
=============================

Numerical Considerations
------------------------

The stretch-creation variant requires ODE integration for all test modes except SAOS (which is analytical). Key challenges:

1. **Nonlinear coupling**: :math:`\text{tr}(S)` couples all components
2. **Positive feedback**: Large :math:`\kappa` can cause stiffness in ODE
3. **Steady-state root-finding**: Implicit equations for flow curve
4. **JIT compilation**: JAX automates differentiation for gradients

Recommended Solver Settings
----------------------------

- **ODE solver**: Dormand-Prince 4(5) adaptive Runge-Kutta (dopri5)
- **Absolute tolerance**: :math:`10^{-8}` for stress, :math:`10^{-6}` for :math:`S` components
- **Relative tolerance**: :math:`10^{-6}`
- **Maximum step**: :math:`0.1 \tau_b` to resolve fast transients

For large :math:`\kappa` (:math:`> 2`), may need:

- Stricter tolerances (:math:`10^{-10}` absolute)
- Implicit solver (BDF) instead of explicit RK
- Smaller maximum step (:math:`0.01 \tau_b`)

JIT Compilation with JAX
-------------------------

The ODE right-hand side :math:`dS/dt` is JIT-compiled for efficiency:

.. code-block:: python

   @jax.jit
   def ode_rhs(S, kappa_flow, kappa):
       trace_S = jnp.trace(S)
       dSdt = (jnp.dot(kappa_flow, S) + jnp.dot(S, kappa_flow.T)
               - (1/tau_b) * (S - jnp.eye(3))
               - (kappa/tau_b) * (trace_S - 3) * jnp.eye(3))
       return dSdt

JAX automatically differentiates this for use in NLSQ fitting (Jacobian-based optimization).

Steady-State Root Finding
--------------------------

For flow curve prediction, solve :math:`dS/dt = 0`:

.. code-block:: python

   from jax.scipy.optimize import root

   def residual(S_flat, gamma_dot, kappa):
       S = S_flat.reshape(3, 3)
       dSdt = ode_rhs(S, kappa_flow(gamma_dot), kappa)
       return dSdt.flatten()

   S_ss = root(residual, S_init.flatten(), gamma_dot, kappa).reshape(3, 3)
   sigma = G * S_ss[0, 1] + eta_s * gamma_dot

For :math:`\kappa > 0`, need good initial guess (e.g., base model solution or previous :math:`\dot{\gamma}` value).

Vectorization for Efficiency
-----------------------------

When predicting over arrays of :math:`\dot{\gamma}` or :math:`\omega`, use `jax.vmap`:

.. code-block:: python

   predict_single = lambda gamma_dot: solve_steady_state(gamma_dot, kappa)
   predict_vectorized = jax.vmap(predict_single)
   sigma_array = predict_vectorized(gamma_dot_array)

This compiles to parallel execution on GPU/TPU.

Handling Numerical Instability
-------------------------------

If :math:`\text{tr}(S) \to \infty` (runaway creation):

1. **Reduce :math:`\kappa`**: Likely unphysical value
2. **Add saturation**: Modify :math:`k_{on}` to plateau at large stretch
3. **Combine with FENE**: Bound chain extension
4. **Check initial conditions**: Ensure :math:`S(0) = I` for rest

Typical symptom: Solver fails with "maximum iterations exceeded" or NaN in stress.

Fitting Guidance
================

Hierarchical Fitting Strategy
------------------------------

**Step 1: Linear viscoelasticity (SAOS)**

Fit base model (:math:`\kappa = 0`) to determine:

- :math:`G` from plateau :math:`G'`
- :math:`\tau_b` from crossover frequency
- :math:`\eta_s` from high-frequency :math:`G''`

**Step 2: Nonlinear startup (validate base)**

Predict startup with :math:`\kappa = 0`, check agreement with data. If overshoot is under-predicted, proceed to Step 3.

**Step 3: Fit :math:`\kappa` from startup or flow curve**

With :math:`G, \tau_b, \eta_s` fixed, optimize :math:`\kappa` to match:

- Startup overshoot magnitude
- Flow curve curvature at high :math:`\text{Wi}`

**Step 4: Joint refinement (optional)**

Re-optimize all 4 parameters simultaneously on combined SAOS + startup + flow data.

Parameter Bounds
----------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Fitting Bounds
   * - :math:`G`
     - 100 - 10⁶ Pa
     - (1, 10⁸) Pa
   * - :math:`\tau_b`
     - 0.001 - 1000 s
     - (10⁻⁶, 10⁴) s
   * - :math:`\kappa`
     - 0.0 - 2.0
     - (0.0, 5.0)
   * - :math:`\eta_s`
     - 0 - 1000 Pa·s
     - (0, 10⁴) Pa·s

Common Pitfalls
---------------

1. **Overfitting :math:`\kappa`**: Without SAOS constraint, can fit noise. Always anchor with linear data first.

2. **Confusing :math:`\kappa` with FENE**: Both cause strain hardening. Check if hardening appears in extension (FENE) vs creation rate (stretch-creation).

3. **Ignoring :math:`\eta_s`**: High-rate plateau mistaken for creation effect. Fit solvent viscosity separately.

4. **Too large :math:`\kappa`**: Values above 3-5 often unphysical, may cause numerical issues.

5. **Wrong initial guess**: Root-finding for steady state needs reasonable :math:`S` initial guess, especially large :math:`\kappa`.

Optimization Settings (NLSQ)
-----------------------------

RheoJAX uses NLSQ (JAX-accelerated Levenberg-Marquardt):

.. code-block:: python

   result = model.fit(
       data,
       test_mode='startup',
       max_iter=5000,
       ftol=1e-8,
       xtol=1e-8,
   )

For stretch-creation variant:

- Increase `max_iter` to 5000-10000 if :math:`\kappa > 1`
- Set `ftol=1e-8` for tight convergence
- Monitor residuals: should decrease monotonically

If fit fails to converge:

- Check data quality (noise, outliers)
- Reduce number of free parameters (fix :math:`\eta_s = 0` if applicable)
- Try different initial guess for :math:`\kappa` (e.g., 0.1, 0.5, 1.0)

Diagnostics
-----------

After fitting, check:

1. **Residual plot**: Should be random scatter, no systematic trends
2. **Predicted vs observed**: :math:`R^2 > 0.95` for good fit
3. **Parameter uncertainties**: Bootstrap or Bayesian inference
4. **Cross-validation**: Predict held-out test (e.g., different :math:`\dot{\gamma}`)

Physical sanity:

- :math:`G` within expected range for material class
- :math:`\tau_b` consistent with bond energy (Arrhenius check)
- :math:`\kappa` positive (creation enhances with stretch)

Usage Examples
==============

Basic Usage
-----------

.. code-block:: python

   from rheojax.models import TNTSingleMode
   from rheojax.core import RheoData
   import numpy as np

   # Create model with stretch-creation breakage
   model = TNTSingleMode(breakage="stretch_creation")

   # Load experimental data (startup shear)
   t = np.linspace(0, 10, 200)  # seconds
   sigma_exp = load_experimental_stress(t)  # Pa

   data = RheoData(x=t, y=sigma_exp, test_mode='startup')

   # Fit model
   result = model.fit(data, gamma_dot=1.0)  # shear rate in s^-1

   # Inspect fitted parameters
   print(f"G = {model.G.value:.2e} Pa")
   print(f"tau_b = {model.tau_b.value:.3f} s")
   print(f"kappa = {model.kappa.value:.3f}")
   print(f"eta_s = {model.eta_s.value:.2e} Pa·s")

   # Predict and plot
   sigma_pred = model.predict(t, test_mode='startup', gamma_dot=1.0)

   import matplotlib.pyplot as plt
   plt.plot(t, sigma_exp, 'o', label='Data')
   plt.plot(t, sigma_pred, '-', label='Fit')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.show()

SAOS Prediction
---------------

.. code-block:: python

   # Small amplitude oscillatory shear
   omega = np.logspace(-2, 2, 50)  # rad/s

   # Set parameters manually or from previous fit
   model.G.value = 1000.0        # Pa
   model.tau_b.value = 1.0       # s
   model.kappa.value = 0.5       # dimensionless
   model.eta_s.value = 0.0       # Pa·s

   # Predict complex modulus (kappa has no effect in linear regime)
   G_complex = model.predict(omega, test_mode='oscillation')

   G_prime = G_complex.real
   G_double_prime = G_complex.imag

   plt.loglog(omega, G_prime, label="G'")
   plt.loglog(omega, G_double_prime, label='G"')
   plt.xlabel('Frequency (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.show()

Flow Curve Prediction
---------------------

.. code-block:: python

   # Steady shear flow
   gamma_dot = np.logspace(-2, 2, 50)  # s^-1

   # Predict steady-state stress
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   plt.loglog(gamma_dot, sigma)
   plt.xlabel('Shear rate (1/s)')
   plt.ylabel('Stress (Pa)')
   plt.title(f'Flow curve (kappa = {model.kappa.value:.2f})')
   plt.show()

   # Compare with base model (kappa = 0)
   model_base = TNTSingleMode(breakage="base")
   model_base.G.value = model.G.value
   model_base.tau_b.value = model.tau_b.value
   model_base.eta_s.value = model.eta_s.value

   sigma_base = model_base.predict(gamma_dot, test_mode='flow_curve')

   plt.loglog(gamma_dot, sigma, label=f'Stretch-creation (κ={model.kappa.value:.2f})')
   plt.loglog(gamma_dot, sigma_base, '--', label='Base (κ=0)')
   plt.xlabel('Shear rate (1/s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()
   plt.show()

Comparing Variants
------------------

.. code-block:: python

   # Create three models: base, Bell, stretch-creation
   model_base = TNTSingleMode(breakage="base")
   model_bell = TNTSingleMode(breakage="bell")
   model_stretch = TNTSingleMode(breakage="stretch_creation")

   # Set common parameters
   for m in [model_base, model_bell, model_stretch]:
       m.G.value = 1000.0
       m.tau_b.value = 1.0
       m.eta_s.value = 0.0

   # Variant-specific parameters
   model_bell.F_0.value = 50.0       # kT
   model_stretch.kappa.value = 1.0   # dimensionless

   # Predict startup
   t = np.linspace(0, 5, 200)
   gamma_dot = 1.0

   sigma_base = model_base.predict(t, test_mode='startup', gamma_dot=gamma_dot)
   sigma_bell = model_bell.predict(t, test_mode='startup', gamma_dot=gamma_dot)
   sigma_stretch = model_stretch.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   plt.plot(t, sigma_base, label='Base')
   plt.plot(t, sigma_bell, label='Bell (force-sensitive)')
   plt.plot(t, sigma_stretch, label='Stretch-creation')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.title('Startup comparison')
   plt.legend()
   plt.show()

Composition with FENE
---------------------

.. code-block:: python

   # Combine stretch-creation with FENE (bounded extensibility)
   from rheojax.models import TNTSingleMode

   model = TNTSingleMode(breakage="stretch_creation", elasticity="fene")

   # Set FENE parameter (finite extensibility)
   model.L.value = 10.0  # Maximum stretch ratio

   # Set stretch-creation parameter
   model.kappa.value = 1.5

   # Other parameters
   model.G.value = 5000.0
   model.tau_b.value = 0.5
   model.eta_s.value = 0.0

   # Predict large-strain startup
   t = np.linspace(0, 3, 300)
   gamma_dot = 5.0  # high rate

   sigma = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   plt.plot(t, sigma)
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.title('Stretch-creation + FENE')
   plt.show()

   # The FENE term prevents S from diverging at large strain
   # The kappa term enhances stress buildup during extension

Bayesian Inference
------------------

.. code-block:: python

   from rheojax.pipeline import BayesianPipeline

   # Load data
   data = RheoData(x=t, y=sigma_exp, test_mode='startup')

   # Create pipeline
   pipeline = BayesianPipeline()

   # Fit with NLSQ first (warm-start)
   pipeline.set_model(model)
   pipeline.set_data(data)
   nlsq_result = pipeline.fit_nlsq(gamma_dot=1.0)

   # Run Bayesian inference (NUTS sampler)
   bayes_result = pipeline.fit_bayesian(
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
   )

   # Plot posterior distributions
   pipeline.plot_pair()      # Pairwise correlations
   pipeline.plot_forest()    # Credible intervals
   pipeline.plot_trace()     # MCMC chains

   # Extract credible intervals
   intervals = model.get_credible_intervals(
       bayes_result.posterior_samples,
       credibility=0.95
   )

   print("95% Credible Intervals:")
   for param, (low, high) in intervals.items():
       print(f"  {param}: [{low:.3e}, {high:.3e}]")

Composition with Other Variants
================================

The stretch-creation variant can be combined with other TNT variants to capture multiple physical effects simultaneously.

With Bell Breakage (Force-Activated Breakage)
----------------------------------------------

Combining stretch-creation and Bell breakage models a network where both creation and destruction are stress-dependent:

.. code-block:: python

   # Not directly supported in single breakage parameter
   # Would require custom model combining both mechanisms
   # Physically: stress-activated breakage + stretch-activated creation

This combination is relevant for dual-responsive networks (e.g., mechanophore activation + mechano-sensitive degradation).

With FENE Elasticity (Finite Extensibility)
--------------------------------------------

As shown above, this is the recommended combination for large strains:

.. code-block:: python

   model = TNTSingleMode(breakage="stretch_creation", elasticity="fene")

- FENE bounds :math:`\text{tr}(S) < 3L^2`, preventing divergence
- Stretch-creation enhances stress within that bound
- Captures both kinetic stiffening (creation) and elastic stiffening (FENE)

Multi-Mode Extension
--------------------

The stretch-creation mechanism can be extended to multi-mode models:

.. math::

   \sigma = \sum_{i=1}^N G_i (S_i - I) + 2\eta_s D

Each mode :math:`i` has its own :math:`\tau_{b,i}` and could have its own :math:`\kappa_i` if creation enhancement varies with timescale.

Typically, a single :math:`\kappa` is used (same mechanism across all modes).

See Also
========

Related Models
--------------

- :ref:`model-tnt-tanaka-edwards`: Standard Tanaka-Edwards (no stretch-creation coupling)
- :ref:`model-tnt-bell`: Force-activated bond breakage (stress-dependent destruction)
- :ref:`model-tnt-fene-p`: Finite extensibility (elastic stiffening)
- :doc:`/models/fluidity/index`: Tensorial EVP with thixotropic fluidity evolution
- :doc:`/models/dmt/index`: Scalar structure parameter with thixotropic kinetics

Related Documentation
---------------------

- :ref:`models-tnt`: General framework for transient network models
- :doc:`/models/tnt/tnt_protocols`: Shared protocol equations for all TNT variants

API Reference
=============

Model Class
-----------

.. code-block:: python

   class TNTSingleMode(BaseModel):
       """
       Single-mode Tanaka-Edwards transient network model
       with stretch-creation variant.

       Parameters
       ----------
       breakage : str
           Breakage mechanism ("base", "bell", "stretch_creation")
       elasticity : str
           Elasticity type ("gaussian", "fene")

       Attributes
       ----------
       G : Parameter
           Plateau modulus (Pa)
       tau_b : Parameter
           Breakage timescale (s)
       kappa : Parameter
           Stretch-creation enhancement factor (dimensionless, 0-5)
       eta_s : Parameter
           Solvent viscosity (Pa·s)
       """

Key Methods
-----------

.. code-block:: python

   def fit(data: RheoData, **kwargs) -> FitResult:
       """
       Fit model to experimental data.

       Parameters
       ----------
       data : RheoData
           Experimental data with test_mode set
       **kwargs
           Protocol-specific parameters (gamma_dot, omega, etc.)

       Returns
       -------
       FitResult
           Fitted parameters, residuals, R^2, convergence info
       """

   def predict(x: ArrayLike, test_mode: str, **kwargs) -> Array:
       """
       Predict rheological response.

       Parameters
       ----------
       x : array_like
           Independent variable (time, frequency, shear rate)
       test_mode : str
           Protocol ("startup", "oscillation", "flow_curve", etc.)
       **kwargs
           Protocol parameters (gamma_dot, gamma_0, omega, etc.)

       Returns
       -------
       y : Array
           Predicted response (stress, modulus, etc.)
       """

   def fit_bayesian(data: RheoData, num_warmup=1000, num_samples=2000,
                    num_chains=4, **kwargs) -> BayesianResult:
       """
       Bayesian parameter inference using NUTS.

       Parameters
       ----------
       data : RheoData
           Experimental data
       num_warmup : int
           NUTS warmup iterations
       num_samples : int
           Posterior samples per chain
       num_chains : int
           Number of MCMC chains

       Returns
       -------
       BayesianResult
           Posterior samples, diagnostics (R-hat, ESS)
       """

Supported Test Modes
--------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Test Mode
     - Description
   * - ``"oscillation"``
     - Small amplitude oscillatory shear (SAOS): :math:`G^*(\omega)`
   * - ``"flow_curve"``
     - Steady shear flow: :math:`\sigma(\dot{\gamma})`
   * - ``"startup"``
     - Startup of steady shear: :math:`\sigma(t)` at constant :math:`\dot{\gamma}`
   * - ``"relaxation"``
     - Stress relaxation after cessation: :math:`\sigma(t)`
   * - ``"creep"``
     - Creep under constant stress: :math:`\gamma(t)` at constant :math:`\sigma_0`
   * - ``"laos"``
     - Large amplitude oscillatory shear: :math:`\sigma(t)` at :math:`\gamma(t) = \gamma_0 \sin(\omega t)`

Parameter Access
----------------

.. code-block:: python

   # Get parameter value
   G_value = model.G.value  # Pa

   # Set parameter value
   model.kappa.value = 1.0

   # Get parameter bounds
   lower, upper = model.kappa.bounds

   # Get all parameters as dict
   params = model.get_parameter_dict()

References
==========

Foundational Theory
-------------------

1. **Tanaka, F., & Edwards, S. F.** (1992). Viscoelastic properties of physically crosslinked networks. 1. Transient network theory. *Macromolecules*, 25(5), 1516-1523.

   - Original transient network theory (base model)

2. **Flory, P. J.** (1947). Thermodynamics of crystallization in high polymers. IV. A theory of crystalline states and fusion in polymers, copolymers, and their mixtures with diluents. *Journal of Chemical Physics*, 15(6), 397-408.

   - Classical theory of strain-induced crystallization

3. **James, H. M., & Guth, E.** (1943). Theory of the elastic properties of rubber. *Journal of Chemical Physics*, 11(10), 455-481.

   - Non-Gaussian network theory (precursor to FENE)

Mechanophore and Adaptive Networks
-----------------------------------

4. **Vernerey, F. J., Long, R., & Brighenti, R.** (2017). A statistically-based continuum theory for polymers with transient networks. *Journal of the Mechanics and Physics of Solids*, 107, 1-20.

   - Modern continuum framework for adaptive networks

5. **Wang, Q., Gossweiler, G. R., Craig, S. L., & Zhao, X.** (2014). Cephalopod-inspired design of electro-mechano-chemically responsive elastomers for on-demand fluorescent patterning. *Nature Communications*, 5, 4899.

   - Mechanophore-crosslinked elastomers (experimental)

6. **Creton, C.** (2017). 50th Anniversary Perspective: Networks and Gels: Soft but Dynamic and Tough. *Macromolecules*, 50(21), 8297-8316.

   - Review of dynamic networks and toughening mechanisms

Strain Crystallization
----------------------

7. **Candau, N., Laghmach, R., Chazeau, L., Chenal, J.-M., Gauthier, C., Biben, T., & Munch, E.** (2014). Strain-induced crystallization of natural rubber and cross-link densities heterogeneities. *Macromolecules*, 47(16), 5815-5824.

   - Modern experimental study of strain crystallization

8. **Tosaka, M.** (2007). Strain-induced crystallization of crosslinked natural rubber as revealed by X-ray diffraction using synchrotron radiation. *Polymer Journal*, 39(12), 1207-1220.

   - In-situ X-ray studies of crystallization kinetics

Numerical Methods
-----------------

9. **Hulsen, M. A., Fattal, R., & Kupferman, R.** (2005). Flow of viscoelastic fluids past a cylinder at high Weissenberg number: Stabilized simulations using matrix logarithms. *Journal of Non-Newtonian Fluid Mechanics*, 127(1), 27-39.

   - Numerical techniques for viscoelastic constitutive equations

10. **Owens, R. G., & Phillips, T. N.** (2002). *Computational Rheology*. Imperial College Press.

    - Comprehensive reference for solving rheological ODEs/PDEs

Related Experimental Techniques
--------------------------------

11. **Hyun, K., Wilhelm, M., Klein, C. O., Cho, K. S., Nam, J. G., Ahn, K. H., Lee, S. J., Ewoldt, R. H., & McKinley, G. H.** (2011). A review of nonlinear oscillatory shear tests: Analysis and application of large amplitude oscillatory shear (LAOS). *Progress in Polymer Science*, 36(12), 1697-1753.

    - LAOS methodology for probing nonlinear response

12. **Negi, A. S., & Osuji, C. O.** (2010). New insights on fumed colloidal rheology—shear thickening and vorticity-aligned structures in flocculating dispersions. *Rheologica Acta*, 49(5), 493-500.

    - Flow-induced structure formation (related phenomenology)
