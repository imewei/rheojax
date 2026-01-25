Shear Transformation Zone (STZ)
===============================

Quick Reference
---------------

**Use when:** Amorphous solids, metallic glasses, colloidal suspensions near jamming, emulsions, granular matter

**Parameters:** 10 (G₀, σ_y, χ_∞, τ₀, ε₀, c₀, e_Z, τ_β, m_∞, Γ_m)

**Key equation:** :math:`\dot{\varepsilon}^{pl} = \frac{\varepsilon_0}{\tau_0} \Lambda(\chi) \mathcal{C}(s) \mathcal{T}(s)`

**Test modes:** flow_curve (steady_shear), startup, relaxation, creep, oscillation (LAOS)

**Material examples:** Metallic glasses, colloidal glasses, dense emulsions, granular matter

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
local structural disorder (effective temperature χ).

**Key physical picture:**

1. **Low χ (annealed glass)**: Deep potential energy minima, high barriers, rare
   STZ flips → solid-like, high yield stress
2. **High χ (rejuvenated glass)**: Shallow potential, low barriers, frequent flips
   → fluid-like, low yield stress
3. **Flow-induced heating**: Plastic dissipation increases χ (rejuvenation)
4. **Aging**: Quiescent relaxation decreases χ (annealing)

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

Validity and Assumptions
------------------------

**Model Assumptions:**

1. **Mesoscopic STZ size**: Rearrangements involve ~5-10 particles (coarse-grained)
2. **Effective temperature**: Configurational disorder can be described by a single
   scalar :math:`\chi`
3. **Two-state STZ**: Each zone can flip between "+" and "-" orientations
4. **Local stress bias**: Applied stress biases transitions via :math:`\tanh(s/\sigma_y)`
5. **Separation of timescales**: Fast elastic response (τ₀) vs slow χ evolution

**When the model works well:**

- Amorphous solids below glass transition (T < T_g)
- Dense colloidal suspensions (φ > 0.55)
- Metallic glasses under deformation
- Systems where plastic flow is localized (not cooperative)

**Limitations:**

- No spatial coupling (homogeneous model; use nonlocal variants for shear banding)
- Assumes scalar effective temperature (no tensorial disorder)
- No explicit aging kinetics beyond χ relaxation
- Steady-state plasticity may differ from real activated hopping

**Data Requirements:**

- Flow curves (steady shear) for basic fitting
- Startup flow for transient dynamics and χ evolution
- LAOS for nonlinear rheology and back-stress effects (Full variant)

What You Can Learn
------------------

STZ theory provides a microscopic framework for understanding plasticity in amorphous materials through the effective temperature χ and the density of active shear transformation zones Λ(χ).

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**χ (Effective Temperature)**:
   The configurational disorder parameter, normalized by the glass transition temperature.

   *For graduate students*: χ = T_eff/T_g is the ratio of the effective configurational temperature to the glass transition temperature T_g. Unlike thermal temperature T, χ quantifies the disorder in the inherent structure (energy landscape minima). In equilibrium, χ → k_BT/T_g. Under flow, plastic dissipation drives χ above its equilibrium value (rejuvenation). The STZ density Λ = exp(-e_Z/χ) ≈ exp(-1/χ) controls the rate of plastic events. At χ = 1, the system is at the glass transition; χ < 1 is glassy (arrested), χ > 1 is liquid-like.

   *For practitioners*: χ < 0.5 means deeply annealed glass (high yield stress, brittle), 0.5 < χ < 1.0 means moderately annealed (moderate yield stress, ductile), χ > 1.0 means rejuvenated or liquid-like (low or no yield stress). Fitting χ₀ from startup overshoot magnitude and χ_∞ from steady-state shear thinning reveals the material's structural evolution under flow.

**σ_y (Yield Stress Scale)**:
   The stress scale for STZ activation, not the macroscopic yield stress.

   *For graduate students*: σ_y appears in the activation factors C(s) = cosh(s/σ_y)^q and T(s) = tanh(s/σ_y). It sets the stress scale at which STZs flip from one orientation to the other. The macroscopic yield stress σ_y^eff ∼ σ_y√Λ(χ) depends on the STZ density. Near the glass transition, σ_y is related to the shear modulus times the STZ size: σ_y ≈ G₀ε₀.

   *For practitioners*: σ_y controls the curvature of the flow curve. Larger σ_y means the material transitions more gradually from solid-like to fluid-like behavior. Fit σ_y from the stress scale where the flow curve bends (not the low-rate plateau, which depends on χ).

**ε₀ (STZ Strain)**:
   The local strain released when a single STZ flips orientation.

   *For graduate students*: ε₀ is the typical strain increment per STZ rearrangement event. It represents the local shear transformation of a cluster of ~5-10 particles. The plastic strain rate is ε̇^pl = ε₀Λ(χ)R where R is the STZ flip rate. Typical values ε₀ ≈ 0.1-0.3 correspond to a displacement of ~10-30% of the particle diameter.

   *For practitioners*: ε₀ is usually fixed (not fitted) at 0.1 or 0.2 based on literature values for similar materials. It controls the absolute magnitude of the plastic strain rate.

**c₀ (Effective Specific Heat)**:
   The configurational heat capacity controlling the rate of χ evolution.

   *For graduate students*: c₀ appears in dχ/dt = (sε̇^pl/c₀σ_y)(χ_∞ - χ). It represents the density of configurational states per unit energy. Physically, c₀ ∼ (k_B/T_g)(∂S_conf/∂E)_V where S_conf is the configurational entropy. Lower c₀ means the system heats (increases χ) more rapidly under plastic dissipation.

   *For practitioners*: c₀ controls the width of the stress overshoot in startup. Smaller c₀ → sharper overshoot. Fit c₀ from the time to reach peak stress at a given shear rate. Typical values: 0.1-1.0.

**τ₀ (Attempt Time)**:
   The microscopic timescale for STZ flip attempts.

   *For graduate students*: τ₀ is the inverse attempt frequency, related to phonon vibrations (metallic glasses) or Brownian diffusion (colloids). The plastic strain rate scales as ε̇^pl ∼ ε₀/τ₀. For metallic glasses, τ₀ ≈ 10⁻¹²-10⁻⁹ s (atomic vibrations). For colloids, τ₀ ≈ η_s a³/k_BT (Brownian time).

   *For practitioners*: τ₀ sets the absolute timescale of flow. Fit τ₀ from the shear rate scale where the flow curve transitions from yield-dominated to rate-dependent. Typical values: 10⁻⁹-10⁻⁶ s for glasses, 10⁻⁴-10⁻¹ s for pastes.

**e_Z (STZ Formation Energy)**:
   The energy barrier for creating a new STZ, normalized by k_BT_g.

   *For graduate students*: e_Z appears in the equilibrium STZ density Λ_eq = exp(-e_Z/χ). It represents the free energy cost of introducing a local rearrangeable region. In the Standard/Full variants, dΛ/dt = -(Λ - exp(-e_Z/χ))/τ_β describes the relaxation toward equilibrium. Typical values e_Z ≈ 0.5-2.

   *For practitioners*: e_Z controls the equilibrium STZ density and thus the long-time aging behavior. Higher e_Z → fewer equilibrium STZs → slower aging. Usually fitted from aging experiments (stress growth at rest).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from STZ Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - χ Range
     - Structural State
     - Typical Materials
     - Flow Behavior
   * - **χ < 0.4**
     - Deeply annealed glass
     - Aged metallic glasses, ultra-strong colloids
     - Very high yield stress (>10 GPa for metals, >1 kPa for colloids), brittle, catastrophic failure, minimal ductility
   * - **0.4 < χ < 0.7**
     - Moderately annealed glass
     - As-quenched metallic glasses, carbopol gels, aged emulsions
     - High yield stress (1-10 GPa for metals, 100-1000 Pa for colloids), ductile with large overshoot, significant aging
   * - **0.7 < χ < 1.0**
     - Weakly annealed glass
     - Rejuvenated metallic glasses, fresh colloidal suspensions
     - Moderate yield stress (0.1-1 GPa for metals, 10-100 Pa for colloids), small overshoot, weak aging
   * - **1.0 < χ < 1.5**
     - Near-transition
     - Glasses near T_g, very soft colloids
     - Low or no clear yield stress, strong shear thinning, no aging
   * - **χ > 1.5**
     - Supercooled liquid
     - Above T_g, dilute suspensions
     - Newtonian or weakly shear-thinning, no solid-like behavior

Connection to Aging and Rejuvenation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Aging (Quiescent Evolution)**: In the absence of flow, χ decreases via:

.. math::

   \dot{\chi}_{\text{aging}} = -\frac{\chi - \chi_{\text{eq}}}{\tau_{\text{age}}}

- Aging timescale :math:`\tau_{\text{age}}` can be 10³-10⁶ seconds (hours to days)
- Decrease in χ → increase in yield stress over time (thixotropic hardening)
- Measurable via time-dependent stress growth in startup experiments

**Rejuvenation (Flow-Induced Heating)**: During flow, plastic dissipation increases χ:

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

- **Low χ**: Few STZs available (:math:`\Lambda \to 0`), very high activation barrier
- **High χ**: Many STZs (:math:`\Lambda \to 1`), easy plastic flow

This explains why:

1. **Aging increases yield stress**: χ decreases → Λ decreases → fewer active STZs
2. **Rejuvenation decreases yield stress**: χ increases → Λ increases → more active STZs
3. **Temperature dependence**: Near T_g, χ is very sensitive to temperature

Transient Stress Overshoot
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress overshoot in startup flow arises from competition between:

1. **Elastic loading**: :math:`s` increases as strain accumulates
2. **Structural evolution**: :math:`\chi` increases due to plastic dissipation
3. **Accelerating plasticity**: Higher :math:`\chi` → higher :math:`\Lambda` → faster :math:`\dot{\varepsilon}^{pl}`

**Peak stress location**: Occurs when :math:`d\sigma/dt = 0`, typically at strain γ ~ 0.1-0.3

**Overshoot magnitude**: :math:`\sigma_{\text{peak}} / \sigma_{ss}` increases with:

- Lower initial χ (more annealed)
- Higher shear rate (Wi > 1)
- Lower c₀ (slower χ evolution)

Fitting Strategy
~~~~~~~~~~~~~~~~

From steady-state flow curves, extract:

1. **σ_y**: Plateau stress at low :math:`\dot{\gamma}`
2. **Shear thinning slope**: Related to χ_∞ and c₀

From startup transients, extract:

3. **χ₀ (initial state)**: Controls overshoot magnitude
4. **τ_β or c₀**: Controls overshoot timing

From aging experiments, extract:

5. **Aging timescale**: Related to e_Z and thermal relaxation

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

- **σ_y**: Extrapolate to :math:`\dot{\gamma} \to 0`
- **χ_∞**: From shear thinning slope (higher slope → higher χ_∞)

**Step 2: From startup overshoot**

Fit :math:`\sigma(t)` at constant :math:`\dot{\gamma}` to extract:

- **χ₀ (initial χ)**: Controls overshoot height
- **c₀ or τ_β**: Controls overshoot width

**Step 3: From LAOS (optional, Full variant)**

Fit Lissajous curves to extract:

- **m_∞, Γ_m**: Back-stress and kinematic hardening parameters

Typical Parameter Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Notes
   * - χ₀
     - 0.3-1.0
     - Initial effective temperature (lower = more annealed)
   * - χ_∞
     - 0.5-2.0
     - Steady-state at high drive (higher = more rejuvenated)
   * - σ_y
     - 10²-10⁶ Pa
     - Material-dependent yield stress scale
   * - τ₀
     - 10⁻⁹-10⁻⁶ s
     - Molecular vibration time (faster for colloids than polymers)
   * - ε₀
     - 0.1-0.3
     - Strain per STZ flip (dimensionless)
   * - c₀
     - 0.1-1.0
     - Specific heat (higher = slower χ evolution)

Troubleshooting
~~~~~~~~~~~~~~~

**Problem: No stress overshoot in startup**

- **Solution**: Increase initial χ contrast (lower χ₀ or higher χ_∞)
- Or increase shear rate (need Wi = :math:`\dot{\gamma} \tau_\alpha > 1`)

**Problem: Overshoot too sharp/broad**

- **Solution**: Adjust c₀ (lower c₀ → sharper overshoot)
- Or adjust τ_β (Standard/Full variant)

**Problem: Wrong steady-state stress**

- **Solution**: Adjust σ_y and χ_∞ simultaneously
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

   model.fit(gamma_dot, stress_data, test_mode="steady_shear")

   print(model.parameters.get_value("sigma_y"))

   # --- 2. Transient Startup Simulation ---
   # Simulate stress overshoot at constant shear rate
   t = np.linspace(0, 10, 1000)
   stress_overshoot = model.predict(t, test_mode="startup", gamma_dot=1.0)

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

References
----------

.. [1] Langer, J. S. "Shear-transformation-zone theory of plastic deformation near the
   glass transition." *Physical Review E*, 77, 021502 (2008).
   https://doi.org/10.1103/PhysRevE.77.021502

.. [2] Falk, M. L. and Langer, J. S. "Dynamics of viscoplastic deformation in amorphous
   solids." *Physical Review E*, 57, 7192 (1998).
   https://doi.org/10.1103/PhysRevE.57.7192

.. [3] Bouchbinder, E. and Langer, J. S. "Nonequilibrium thermodynamics of driven
   amorphous materials." *Physical Review E*, 80, 031131, 031132, 031133 (2009).
   https://doi.org/10.1103/PhysRevE.80.031131

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
