Maxwell-Isotropic-Kinematic Hardening (MIKH)
============================================

.. admonition:: Quick Reference
   :class: hint

   **Use when:** Thixotropic elasto-viscoplastic materials with stress overshoot, Bauschinger effect, thixotropic hysteresis

   **Parameters:** 11 (G, η, C, γ_dyn, m, σ_y0, Δσ_y, τ_thix, Γ, η_inf, μ_p)

   **Key equations:** dσ/dt = G(γ̇ - γ̇ᵖ) - (G/η)σ, dα = C·dγ_p - γ_dyn·α·|dγ_p|

   **Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

   **Materials:** Drilling fluids, greases, waxy crude oil, thixotropic cements, structured emulsions

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

**Term 2 (Dynamic Recovery):** -γ_dyn·|α|^(m-1)·α·|dγ_p|

- Limits backstress saturation (prevents unbounded growth)
- γ_dyn controls recovery rate
- m controls nonlinearity (m = 1 is linear, m > 1 accelerates recovery at high α)
- Recovery is proportional to |dγ_p|, so it only occurs during plastic flow

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

**Term 2 (Breakdown):** Γ·λ·|γ̇ᵖ|

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

The material yields when the relative stress |ξ| = |σ - α| exceeds the current
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

.. code-block:: python

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

1. Dimitriou, C.J. & McKinley, G.H. (2014). "A comprehensive constitutive law for
   waxy crude oil: a thixotropic yield stress fluid." *Soft Matter*, 10(35), 6619-6644.
   DOI: 10.1039/c4sm00578c

2. Geri, M., Venkatesan, R., Sambath, K., & McKinley, G.H. (2017). "Thermokinematic
   memory and the thixotropic elasto‐viscoplasticity of waxy crude oils."
   *J. Rheol.*, 61(3), 427-454. DOI: 10.1122/1.4978259

3. Saramito, P. (2009). "A new elastoviscoplastic model based on the Herschel–Bulkley
   viscoplastic model." *J. Non-Newtonian Fluid Mech.*, 158, 154-161.

4. de Souza Mendes, P.R. & Thompson, R.L. (2019). "Time-dependent yield stress
   materials." *Annu. Rev. Fluid Mech.*, 51, 421-449.

5. Armstrong, P.J. & Frederick, C.O. (1966). "A mathematical representation of the
   multiaxial Bauschinger effect." *CEGB Report RD/B/N731*.

6. Mewis, J. & Wagner, N.J. (2009). "Thixotropy."
   *Adv. Colloid Interface Sci.*, 147-148, 214-227.
