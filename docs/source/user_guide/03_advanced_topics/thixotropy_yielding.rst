.. _thixotropy_yielding:

=============================================
Thixotropy and Yield Stress Analysis
=============================================

**Time-Dependent Viscosity and Structural Breakdown Phenomena**

Overview
========

**Thixotropy** is the time-dependent decrease in viscosity under constant shear, followed by
gradual recovery when the shear is removed. This ubiquitous phenomenon appears in industrial
fluids (drilling muds, paints, food products), biological materials (blood, mucus), and
geological systems (lavas, sediments).

.. admonition:: Key Insight
   :class: tip

   Thixotropy arises from a fundamental competition: **structure buildup at rest** versus
   **structure breakdown under flow**. This competition creates a rich phenomenology—stress
   overshoots, delayed yielding, viscosity bifurcation, shear banding—that cannot be captured
   by time-independent constitutive models (Maxwell, Herschel-Bulkley).

Unlike simple yield-stress fluids that transition instantaneously from solid-like to liquid-like
behavior, **thixotropic materials have memory**: their current viscosity depends on the entire
deformation history, not just the current stress or strain rate.

RheoJAX provides five complementary frameworks for thixotropic and yield-stress analysis, each
grounded in different physical mechanisms:

.. list-table:: Five Thixotropic Frameworks in RheoJAX
   :header-rows: 1
   :widths: 20 30 25 25

   * - Framework
     - Physical Mechanism
     - Key Variable
     - Characteristic Signature
   * - **DMT**
     - Structure parameter kinetics
     - :math:`\lambda \in [0,1]` (structure)
     - Stress overshoot, viscosity bifurcation
   * - **Fluidity**
     - Cooperative flow dynamics
     - :math:`f = 1/\eta` (fluidity)
     - Shear banding, nonlocal effects
   * - **Hébraud-Lequeux**
     - Mean-field stress distribution
     - :math:`P(\sigma, t)` (stress PDF)
     - Yield stress, dense suspension dynamics
   * - **STZ**
     - Disorder temperature
     - :math:`\chi` (effective temperature)
     - Metallic glass physics, :math:`\chi > \chi_0` flow
   * - **EPM**
     - Mesoscopic plastic events
     - Local :math:`\sigma`, :math:`\sigma_y` threshold
     - Avalanches, spatial heterogeneity

When to Use Thixotropic Models
===============================

Thixotropic Models Are Essential For
-------------------------------------

**Time-dependent viscosity:**
   Material viscosity changes significantly during the experiment itself (startup transients,
   delayed yielding, viscosity recovery after cessation).

**Structural evolution:**
   Material contains microstructure (flocs, networks, entanglements) that builds and breaks
   dynamically.

**Yield stress fluids with memory:**
   Material has a yield stress that changes with rest time or shear history.

**Shear banding:**
   Flow localizes into high-shear and low-shear bands with a stress plateau.

**Industrial rheometry:**
   Materials like drilling muds, cement slurries, waxy crude oils, paints, cosmetics,
   food products (yogurt, ketchup, mayonnaise).

**Biological fluids:**
   Blood (rouleaux formation), mucus, saliva, synovial fluid.

Skip Thixotropic Models When
-----------------------------

**Time-independent viscosity:**
   Material reaches steady state quickly (<1 s) and shows no history dependence.

**Simple viscoelasticity:**
   Maxwell, Zener, or fractional models adequately capture dynamics.

**No yield stress:**
   Material is a simple Newtonian or power-law fluid.

**SAOS-only data:**
   Small-amplitude oscillatory shear (SAOS) cannot reveal thixotropic structure evolution;
   you need LAOS, startup, creep, or flow curve data.

Material Classification
-----------------------

Diagnostic questions to identify thixotropic behavior:

.. list-table:: Thixotropy Diagnostic Checklist
   :header-rows: 1
   :widths: 50 25 25

   * - Experimental Signature
     - Thixotropic?
     - Preferred Model
   * - Stress overshoot in startup shear
     - Yes
     - DMT, Fluidity, STZ
   * - Viscosity decreases with time at constant :math:`\dot{\gamma}`
     - Yes
     - All frameworks
   * - Flow curve shows hysteresis (up-down ramps)
     - Yes
     - DMT, HL
   * - Creep shows viscosity bifurcation (flows or arrests)
     - Yes
     - DMT, Fluidity
   * - LAOS intracycle yielding with :math:`G'` collapse
     - Yes
     - SPP + DMT/Fluidity
   * - Shear banding (stress plateau)
     - Yes
     - Fluidity Nonlocal, DMT Nonlocal
   * - Recovery: :math:`\eta` increases after cessation
     - Yes
     - All frameworks
   * - Aging: properties drift with waiting time
     - Yes
     - HL, STZ

Theoretical Foundations
=======================

The Structure Parameter Concept
--------------------------------

Most thixotropic models use a **scalar structure parameter** :math:`\lambda` that tracks the degree of
microstructural order:

.. math::

   \frac{d\lambda}{dt} = \underbrace{\frac{1 - \lambda}{t_{\text{eq}}}}_{\text{buildup at rest}}
                       - \underbrace{a \lambda |\dot{\gamma}|^c / t_{\text{eq}}}_{\text{breakdown under shear}}

where:
   - :math:`\lambda = 1`: fully structured (rest state)
   - :math:`\lambda = 0`: fully broken down (high-shear steady state)
   - :math:`t_\text{eq}`: equilibrium timescale (seconds to minutes)
   - :math:`a, c`: breakdown parameters (material-specific)

**Physical interpretation:**

- At rest (:math:`\dot{\gamma} = 0`): :math:`\lambda \to 1` exponentially with time constant :math:`t_\text{eq}`
- Under steady shear: :math:`\lambda \to \lambda_\infty` where buildup = breakdown
- Transient response: :math:`\lambda` evolves during the experiment itself

This structure parameter then modulates viscosity, yield stress, or modulus.

Viscosity Closures
------------------

Different models connect :math:`\lambda` to macroscopic properties differently:

**DMT Exponential Closure:**

.. math::

   \eta(\lambda) = \eta_\infty \left( \frac{\eta_0}{\eta_\infty} \right)^\lambda

Smooth transition from :math:`\eta_0` (structured) to :math:`\eta_\infty` (broken). No explicit yield stress, but
effective yield emerges from high :math:`\eta` at :math:`\lambda \approx 1`.

**DMT Herschel-Bulkley Closure:**

.. math::

   \eta(\dot{\gamma}, \lambda) = \frac{\tau_y(\lambda)}{|\dot{\gamma}|} + K(\lambda) |\dot{\gamma}|^{n-1}

Explicit yield stress :math:`\tau_y(\lambda)` that decreases as structure breaks (:math:`\lambda \to 0`).

**Fluidity Model:**

.. math::

   f = \frac{1}{\eta}, \quad \frac{df}{dt} = \frac{f_{\text{eq}} - f}{\tau_f} + D \nabla^2 f

Fluidity :math:`f` (inverse viscosity) as the primary variable. Nonlocal variant includes spatial
diffusion :math:`D\nabla^2 f`, enabling shear banding.

Elasticity and Stress Overshoot
--------------------------------

For materials with both thixotropy **and** elasticity (soft solids, gels), a **Maxwell
backbone** is often added:

.. math::

   \dot{\sigma} + \frac{\sigma}{\tau(\lambda)} = G \dot{\gamma}

where the relaxation time :math:`\tau` depends on structure:

.. math::

   \tau(\lambda) = \frac{\eta(\lambda)}{G}

This gives **stress overshoot** in startup shear: stress initially rises elastically, exceeds
steady-state value, then relaxes as structure breaks and viscosity drops.

Five Thixotropic Frameworks
============================

1. DMT (de Souza Mendes-Thompson)
----------------------------------

**Physical basis**: Structural kinetics with scalar parameter :math:`\lambda`

**Import:**

.. code-block:: python

   from rheojax.models import DMTLocal, DMTNonlocal

**Structure kinetics:**

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{t_{\text{eq}}}
                       - a \lambda |\dot{\gamma}|^c / t_{\text{eq}}

**Two viscosity closures:**

1. ``closure="exponential"``: :math:`\eta = \eta_\infty(\eta_0/\eta_\infty)^\lambda` (smooth)
2. ``closure="herschel_bulkley"``: Explicit yield stress :math:`\tau_y(\lambda)`

**Optional elasticity:** ``include_elasticity=True`` adds Maxwell backbone for stress overshoot

**Protocols supported:** FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

**Best for:**
   - Drilling muds, waxy crude oils (petroleum)
   - Cement slurries, drilling fluids (construction)
   - Food products (ketchup, mayonnaise, yogurt)
   - Cosmetics (lotions, creams)

**Key strengths:**
   - Simple, well-established framework (de Souza Mendes 2009)
   - Clear physical interpretation of :math:`\lambda`
   - Captures stress overshoot, viscosity bifurcation, hysteresis
   - Nonlocal variant for shear banding

**Example:**

.. code-block:: python

   from rheojax.models import DMTLocal
   import numpy as np

   # Setup model with exponential closure + elasticity
   model = DMTLocal(closure="exponential", include_elasticity=True)

   # Fit to flow curve
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Simulate startup shear (stress overshoot)
   t, stress, lam = model.simulate_startup(gamma_dot=10.0, t_end=100.0)
   # lam starts at 1 (structured), decays → stress overshoots then relaxes

   # Simulate creep (viscosity bifurcation)
   t, gamma, gamma_dot, lam = model.simulate_creep(sigma_0=50.0, t_end=500.0)
   # Above yield: lam→0, flows continuously
   # Below yield: lam→1, arrested after transient

   # Bayesian inference with uncertainty
   result = model.fit_bayesian(gamma_dot, sigma, test_mode='flow_curve',
                                num_warmup=1000, num_samples=2000)

**See:** :doc:`/models/dmt/index`, :doc:`/examples/dmt/index`

2. Fluidity Models
------------------

**Physical basis**: Cooperative flow with fluidity :math:`f = 1/\eta` as primary variable

**Import:**

.. code-block:: python

   from rheojax.models import FluidityLocal, FluidityNonlocal
   from rheojax.models import FluiditySaramitoLocal, FluiditySaramitoNonlocal

**Fluidity evolution:**

.. math::

   \frac{df}{dt} = \frac{f_{\text{eq}}(\dot{\gamma}) - f}{\tau_f}
                 + D \nabla^2 f \quad \text{(nonlocal)}

**Two families:**

1. **Basic Fluidity**: Simple viscous flow with fluidity evolution
2. **Saramito EVP**: Combines Saramito tensorial viscoelasticity with fluidity thixotropy

**Saramito EVP features:**

- **Tensorial stress**: [:math:`\tau_{xx}`, :math:`\tau_{yy}`, :math:`\tau_{xy}`] for normal stress differences (:math:`N_1`)
- **Von Mises yield criterion**: :math:`\alpha = \max(0, 1 - \tau_y/|\tau|)`
- **Fluidity aging**: :math:`df/dt = \text{aging} + b|\dot{\gamma}|^n \cdot \text{rejuvenation}`
- **Two coupling modes**: "minimal" (:math:`\lambda = 1/f` only) vs "full" (:math:`\lambda + \tau_y(f)` aging yield)

**Protocols supported:** FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

**Best for:**
   - Soft glassy materials (foams, concentrated emulsions)
   - Dense colloidal suspensions
   - Pastes, gels with shear banding
   - Materials showing cooperativity (length scale :math:`\xi`)

**Key strengths:**
   - Nonlocal formulation captures shear banding naturally
   - Cooperativity length :math:`\xi` controls spatial correlations
   - Saramito variant predicts normal stresses
   - Well-suited for spatial gradients in rheometry (Couette, cone-plate)

**Example:**

.. code-block:: python

   from rheojax.models import FluiditySaramitoLocal, FluiditySaramitoNonlocal

   # Local model with minimal coupling
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Predict normal stress differences
   N1, N2 = model.predict_normal_stresses(gamma_dot)

   # Startup with thixotropic stress overshoot
   strain, stress, fluidity = model.simulate_startup(t, gamma_dot, t_wait=100)

   # Nonlocal model for shear banding
   nonlocal = FluiditySaramitoNonlocal(coupling="full", n_points=51)
   result = nonlocal.simulate_steady_shear(gamma_dot_avg=10.0, t_end=500.0)

   # Detect banding from spatial profiles
   banding = nonlocal.detect_banding(result, threshold=0.1)
   if banding['is_banded']:
       print(f"High-rate band fraction: {banding['high_fraction']:.2f}")

**See:** :doc:`/models/fluidity/index`, :doc:`/examples/fluidity/index`

3. Hébraud-Lequeux (HL)
------------------------

**Physical basis**: Mean-field theory of stress distribution :math:`P(\sigma, t)` evolution

**Import:**

.. code-block:: python

   from rheojax.models import HebraudLequeux

**Governing equation (Fokker-Planck):**

.. math::

   \frac{\partial P}{\partial t} = -\frac{\partial}{\partial \sigma} (v P)
                                   + D \frac{\partial^2 P}{\partial \sigma^2}
                                   + \text{yielding terms}

where :math:`P(\sigma, t)` is the probability density of local stress values.

**Mean-field coupling:**
   When an element yields, it creates stress redistribution on neighbors, acting as
   "mechanical noise" that drives other elements toward yielding.

**Protocols supported:** FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

**Best for:**
   - Dense colloidal suspensions (hard spheres)
   - Granular materials, pastes
   - Materials where "avalanche" physics is relevant
   - Systems near jamming transitions

**Key strengths:**
   - Rigorous statistical mechanics foundation (Hébraud & Lequeux 1998)
   - Predicts yield stress from first principles
   - Captures cooperative yielding (mean-field avalanches)
   - Natural connection to mode-coupling theory (MCT)

**Example:**

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   model = HebraudLequeux()

   # Fit to flow curve with yield stress
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Extract yield stress
   sigma_y = model.parameters.get_value('sigma_y')
   print(f"Yield stress: {sigma_y:.1f} Pa")

   # SAOS prediction (G', G'')
   omega = np.logspace(-2, 2, 50)
   G_star = model.predict(omega, test_mode='oscillation', return_components=True)

   # Startup shear with yielding transient
   t = np.linspace(0.1, 100, 300)
   sigma_t = model.predict(t, test_mode='startup', gamma_dot=1.0)

**Warning:**
   HL model requires PDE solver (JAX lax.scan). Memory-intensive for large grids;
   recommend ``n_points`` :math:`\leq` 100 for stability.

**See:** :doc:`/models/hl/index`, :doc:`/examples/hl/index`

4. STZ (Shear Transformation Zone)
-----------------------------------

**Physical basis**: Disorder temperature :math:`\chi` tracks effective thermal excitation

**Import:**

.. code-block:: python

   from rheojax.models import STZConventional

**Three variants:**

- ``variant="minimal"``: :math:`\chi` only (2 parameters)
- ``variant="standard"``: :math:`\chi + \Lambda` (STZ density, 4 parameters)
- ``variant="full"``: :math:`\chi + \Lambda + m` (orientation, 6 parameters)

**Effective temperature evolution:**

.. math::

   \frac{d\chi}{dt} = \frac{1}{\tau_0} \left[ \Delta_0 \dot{\gamma}^2
                    - \frac{\chi - \chi_0}{1 + c \dot{\gamma}^2} \right]

where :math:`\chi_0` is the "configurational temperature" and :math:`\chi > \chi_0` enables flow.

**Protocols supported:** FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

**Best for:**
   - Metallic glasses (bulk, thin films)
   - Amorphous solids (oxide glasses)
   - Dense packings approaching jamming
   - Materials where "disorder" is the key variable

**Key strengths:**
   - Grounded in Falk-Langer theory (Phys. Rev. E 1998)
   - :math:`\chi` has clear physical meaning (disorder)
   - Predicts flow when :math:`\chi > \chi_0` (temperature-activated)
   - Successfully describes metallic glass phenomenology

**Example:**

.. code-block:: python

   from rheojax.models import STZConventional

   # Standard variant (χ + Λ)
   model = STZConventional(variant="standard")

   # Fit to flow curve
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Extract disorder parameters
   chi_inf = model.parameters.get_value('chi_inf')
   chi_0 = model.parameters.get_value('chi_0')
   print(f"Steady-state disorder: χ_∞ = {chi_inf:.3f}")
   print(f"Threshold: χ_0 = {chi_0:.3f}")

   # Check if material can flow at rest
   if chi_inf > chi_0:
       print("Material flows continuously (χ > χ_0)")
   else:
       print("Material is jammed at rest (χ < χ_0)")

**See:** :doc:`/models/stz/index`, :doc:`/examples/stz/index`

5. EPM (Elasto-Plastic Models)
-------------------------------

**Physical basis**: Discrete lattice of blocks with local stress and yield threshold

**Import:**

.. code-block:: python

   from rheojax.models import LatticeEPM

**Mesoscopic picture:**

- **Lattice of blocks**: Each block :math:`i` has local stress :math:`\sigma_i` and yield threshold :math:`\sigma_{y,i}`
- **Eshelby stress redistribution**: When block :math:`i` yields, stress redistributes to neighbors
  via elastic Green's function
- **Plastic events**: :math:`\sigma_i > \sigma_{y,i}` triggers local yielding, creating an avalanche

**FFT-accelerated stress redistribution:**
   Uses FFT convolution for :math:`O(N \log N)` stress propagation instead of :math:`O(N^2)` direct summation.

**Two modes:**

1. **Hard threshold** (simulation): :math:`\sigma_y = \text{constant}`, discrete avalanches
2. **Smooth yielding** (inference): Rate-dependent plasticity for gradient-based fitting

**Protocols supported:** FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

**Best for:**
   - Amorphous solids with spatial heterogeneity
   - Materials showing avalanche statistics
   - Systems where "elastic redistribution" dominates
   - Soft glassy materials with mesoscale structure

**Key strengths:**
   - Captures spatial heterogeneity naturally
   - Avalanche statistics emerge from local rules
   - FFT acceleration enables large lattices (:math:`128 \times 128`)
   - Reproduces power-law avalanche distributions

**Example:**

.. code-block:: python

   from rheojax.models import LatticeEPM

   # Setup lattice (L × L)
   model = LatticeEPM(L=32)

   # Fit to flow curve (smooth mode)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Simulate startup with avalanches
   result = model.simulate_startup(gamma_dot=1e-3, t_end=1000.0)

   # Analyze avalanche statistics
   sizes = result['avalanche_sizes']
   print(f"Largest avalanche: {np.max(sizes)} blocks")

**Warning:**
   EPM simulation mode is stochastic; each run differs. Use ``seed=42`` for reproducibility.

**See:** :doc:`/models/epm/index`, :doc:`/examples/epm/index`

Practical Implementation
========================

Diagnostic Workflow: Identifying Thixotropy
--------------------------------------------

**Step 1: Collect diagnostic data**

.. code-block:: python

   from rheojax.io.readers import auto_read
   import numpy as np

   # Essential tests for thixotropy diagnosis:
   # 1. Flow curve (up-down ramp) for hysteresis
   flow_up = auto_read("flow_up.csv")    # Low → high shear rate
   flow_down = auto_read("flow_down.csv") # High → low shear rate

   # 2. Startup shear for stress overshoot
   startup = auto_read("startup.csv")

   # 3. Creep at different stress levels
   creep_low = auto_read("creep_low_stress.csv")
   creep_high = auto_read("creep_high_stress.csv")

**Step 2: Check for thixotropic signatures**

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 3, figsize=(15, 4))

   # Hysteresis loop
   ax = axes[0]
   ax.loglog(flow_up.x, flow_up.y, 'o-', label='Up ramp')
   ax.loglog(flow_down.x, flow_down.y, 's-', label='Down ramp')
   ax.set_xlabel('γ̇ (s⁻¹)')
   ax.set_ylabel('σ (Pa)')
   ax.legend()
   ax.set_title('Flow Curve Hysteresis')

   # Stress overshoot
   ax = axes[1]
   ax.plot(startup.x, startup.y)
   steady_state = np.mean(startup.y[-10:])  # Last 10 points
   max_stress = np.max(startup.y)
   overshoot_ratio = max_stress / steady_state
   ax.axhline(steady_state, color='r', linestyle='--', label='Steady state')
   ax.set_xlabel('t (s) or strain')
   ax.set_ylabel('σ (Pa)')
   ax.set_title(f'Startup: Overshoot = {overshoot_ratio:.2f}×')
   ax.legend()

   # Viscosity bifurcation in creep
   ax = axes[2]
   gamma_low = creep_low.y  # Strain
   gamma_high = creep_high.y
   ax.plot(creep_low.x, gamma_low, label='σ₁ (below yield)')
   ax.plot(creep_high.x, gamma_high, label='σ₂ (above yield)')
   ax.set_xlabel('t (s)')
   ax.set_ylabel('γ')
   ax.legend()
   ax.set_title('Creep: Viscosity Bifurcation')

   plt.tight_layout()
   plt.savefig('thixotropy_diagnostics.png', dpi=150)

**Interpretation:**

- **Hysteresis area > 10%**: Significant thixotropy
- **Overshoot ratio > 1.2**: Structure + elasticity (use DMT with elasticity)
- **Creep bifurcation**: Clear yield stress (DMT, HL, or Fluidity)

Basic DMT Fitting Workflow
---------------------------

.. code-block:: python

   from rheojax.models import DMTLocal
   from rheojax.io.readers import auto_read
   import numpy as np

   # 1. Load data
   data = auto_read("flow_curve.csv")
   gamma_dot = data.x
   sigma = data.y

   # 2. Choose closure based on data
   # Exponential: smooth viscosity decrease
   # Herschel-Bulkley: clear yield plateau at low γ̇
   model = DMTLocal(closure="exponential", include_elasticity=True)

   # 3. NLSQ point estimation
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # 4. Extract parameters
   eta_0 = model.parameters.get_value('eta_0')
   eta_inf = model.parameters.get_value('eta_inf')
   t_eq = model.parameters.get_value('t_eq')
   a = model.parameters.get_value('a')
   c = model.parameters.get_value('c')

   print(f"Zero-shear viscosity: η₀ = {eta_0:.2e} Pa·s")
   print(f"Infinite-shear viscosity: η_∞ = {eta_inf:.2e} Pa·s")
   print(f"Equilibrium time: t_eq = {t_eq:.1f} s")
   print(f"Breakdown: a = {a:.2f}, c = {c:.2f}")

   # 5. Predict other test modes
   # Startup shear
   t_startup = np.linspace(0.1, 100, 300)
   t, stress_startup, lam = model.simulate_startup(gamma_dot=10.0, t_end=100.0)

   # Creep at different stresses
   sigma_test = [30, 50, 70]  # Pa
   for sig in sigma_test:
       t, gamma, gamma_dot_out, lam = model.simulate_creep(sigma_0=sig, t_end=500.0)
       final_gamma_dot = gamma_dot_out[-1]
       if final_gamma_dot > 1e-6:
           print(f"σ = {sig} Pa: FLOWS (γ̇_final = {final_gamma_dot:.2e} s⁻¹)")
       else:
           print(f"σ = {sig} Pa: ARRESTED (γ̇_final ≈ 0)")

Fluidity Nonlocal: Shear Banding Analysis
------------------------------------------

.. code-block:: python

   from rheojax.models import FluiditySaramitoNonlocal
   import numpy as np

   # Setup nonlocal model with spatial grid
   model = FluiditySaramitoNonlocal(
       coupling="full",
       n_points=51,           # Spatial grid points
       gap_width=1e-3         # Geometry gap (m)
   )

   # Fit to flow curve
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Simulate steady shear with spatial resolution
   result = model.simulate_steady_shear(
       gamma_dot_avg=10.0,    # Applied average shear rate
       t_end=500.0            # Simulation time (s)
   )

   # Extract spatial profiles
   y = result['y']           # Spatial coordinate
   gamma_dot_local = result['gamma_dot_profile']  # Local shear rate
   sigma_local = result['stress_profile']         # Local stress
   f_local = result['fluidity_profile']           # Local fluidity

   # Detect shear banding
   banding = model.detect_banding(result, threshold=0.1)

   if banding['is_banded']:
       print("SHEAR BANDING DETECTED")
       print(f"High-rate band fraction: {banding['high_fraction']:.2%}")
       print(f"Interface position: y = {banding['interface_pos']:.4f} mm")

       # Visualize bands
       import matplotlib.pyplot as plt
       fig, axes = plt.subplots(1, 3, figsize=(15, 4))

       ax = axes[0]
       ax.plot(y * 1e3, gamma_dot_local)  # Convert to mm
       ax.axvline(banding['interface_pos'] * 1e3, color='r', linestyle='--')
       ax.set_xlabel('Position y (mm)')
       ax.set_ylabel('γ̇ (s⁻¹)')
       ax.set_title('Shear Rate Profile')

       ax = axes[1]
       ax.plot(y * 1e3, sigma_local)
       ax.set_xlabel('Position y (mm)')
       ax.set_ylabel('σ (Pa)')
       ax.set_title('Stress Profile (Should Be Constant)')

       ax = axes[2]
       ax.plot(y * 1e3, f_local)
       ax.set_xlabel('Position y (mm)')
       ax.set_ylabel('f (fluidity)')
       ax.set_title('Fluidity Profile')

       plt.tight_layout()
       plt.savefig('shear_banding.png', dpi=150)

Bayesian Inference for Thixotropic Models
==========================================

Thixotropic models have many coupled parameters (typically 5-8), making Bayesian inference
especially valuable for uncertainty quantification.

NLSQ → NUTS Workflow
--------------------

.. code-block:: python

   from rheojax.models import DMTLocal
   from rheojax.io.readers import auto_read

   # 1. Load data
   data = auto_read("flow_curve.csv")
   gamma_dot = data.x
   sigma = data.y

   # 2. NLSQ point estimation (fast warm-start)
   model = DMTLocal(closure="exponential", include_elasticity=False)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   print("NLSQ estimates:")
   for name in model.parameters.keys():
       val = model.parameters.get_value(name)
       print(f"  {name} = {val:.3e}")

   # 3. Bayesian inference with warm-start
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,    # Multiple chains for convergence checks
       seed=42          # Reproducibility
   )

   # 4. Check convergence diagnostics
   print("\nConvergence diagnostics:")
   for param_name, r_hat in result.diagnostics['r_hat'].items():
       ess = result.diagnostics['ess'][param_name]
       print(f"  {param_name}: R-hat = {r_hat:.4f}, ESS = {ess:.0f}")

   # Good convergence: R-hat < 1.01, ESS > 400

   # 5. Get credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )

   print("\n95% Credible Intervals:")
   for param_name, (low, high) in intervals.items():
       median = model.parameters.get_value(param_name)
       print(f"  {param_name}: {median:.3e} [{low:.3e}, {high:.3e}]")

Parameter Correlation Analysis
-------------------------------

Thixotropic models often have correlated parameters (e.g., :math:`t_\text{eq}` and :math:`a` are trade-offs).
Use pair plots to diagnose:

.. code-block:: python

   import arviz as az
   import matplotlib.pyplot as plt

   # Convert to ArviZ InferenceData
   idata = az.from_dict(posterior=result.posterior_samples)

   # Pair plot (correlations)
   az.plot_pair(
       idata,
       var_names=['eta_0', 'eta_inf', 't_eq', 'a', 'c'],
       kind='kde',           # Kernel density estimate
       marginals=True,       # Show 1D marginals
       divergences=True      # Highlight divergent transitions
   )
   plt.suptitle('Parameter Correlations (DMT Model)', y=1.02)
   plt.tight_layout()
   plt.savefig('dmt_pair_plot.png', dpi=150)

   # Forest plot (credible intervals)
   az.plot_forest(
       idata,
       var_names=['eta_0', 'eta_inf', 't_eq', 'a', 'c'],
       combined=True,
       hdi_prob=0.95
   )
   plt.title('95% Credible Intervals (DMT Parameters)')
   plt.tight_layout()
   plt.savefig('dmt_forest_plot.png', dpi=150)

**Interpretation:**

- **Strong correlation** (elliptical contours): Parameters are not independently identifiable;
  need more informative data or stronger priors
- **Divergences** (red points): Sampling difficulties; increase ``num_warmup`` or
  ``target_accept=0.95``

Prediction Uncertainty
----------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Generate predictions from posterior samples
   gamma_dot_pred = np.logspace(-3, 2, 100)
   n_samples = len(result.posterior_samples['eta_0'])

   predictions = []
   for i in range(0, n_samples, 10):  # Subsample for speed
       # Set parameters to this posterior sample
       for param_name in model.parameters.keys():
           value = result.posterior_samples[param_name][i]
           model.parameters.set_value(param_name, value)

       # Predict
       sigma_pred = model.predict(gamma_dot_pred, test_mode='flow_curve')
       predictions.append(sigma_pred)

   predictions = np.array(predictions)

   # Compute percentiles
   median = np.median(predictions, axis=0)
   lower = np.percentile(predictions, 2.5, axis=0)
   upper = np.percentile(predictions, 97.5, axis=0)

   # Plot with uncertainty bands
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(gamma_dot, sigma, 'ko', label='Data', markersize=4)
   ax.loglog(gamma_dot_pred, median, 'r-', label='Posterior median')
   ax.fill_between(gamma_dot_pred, lower, upper, alpha=0.3, color='r',
                    label='95% credible interval')
   ax.set_xlabel('γ̇ (s⁻¹)')
   ax.set_ylabel('σ (Pa)')
   ax.legend()
   ax.set_title('DMT Flow Curve with Bayesian Uncertainty')
   plt.tight_layout()
   plt.savefig('bayesian_prediction.png', dpi=150)

Visualization Best Practices
=============================

Thixotropic Hysteresis Loops
-----------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from rheojax.models import DMTLocal

   # Generate hysteresis data
   model = DMTLocal(closure="exponential", include_elasticity=False)
   model.parameters.set_value('eta_0', 1e3)
   model.parameters.set_value('eta_inf', 1.0)
   model.parameters.set_value('t_eq', 10.0)
   model.parameters.set_value('a', 1.0)
   model.parameters.set_value('c', 1.0)

   # Up ramp: low to high shear rate (structure breaks)
   gamma_dot_up = np.logspace(-3, 2, 50)
   sigma_up = model.predict(gamma_dot_up, test_mode='flow_curve')

   # Down ramp: high to low (structure rebuilds, but not instantaneously)
   gamma_dot_down = np.logspace(2, -3, 50)
   # For true hysteresis, need time-dependent simulation; here we approximate
   sigma_down = model.predict(gamma_dot_down, test_mode='flow_curve') * 0.7

   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(gamma_dot_up, sigma_up, 'ro-', label='Up ramp', linewidth=2)
   ax.loglog(gamma_dot_down, sigma_down, 'bs-', label='Down ramp', linewidth=2)

   # Add arrows to show direction
   n = len(gamma_dot_up)
   for i in [10, 25, 40]:
       ax.annotate('', xy=(gamma_dot_up[i+1], sigma_up[i+1]),
                   xytext=(gamma_dot_up[i], sigma_up[i]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

   # Shade hysteresis area
   ax.fill_between(gamma_dot_up, sigma_up, sigma_down, alpha=0.2, color='gray',
                    label='Hysteresis area')

   ax.set_xlabel('Shear rate γ̇ (s⁻¹)', fontsize=14)
   ax.set_ylabel('Shear stress σ (Pa)', fontsize=14)
   ax.legend(fontsize=12)
   ax.set_title('Thixotropic Hysteresis Loop', fontsize=16)
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('hysteresis_loop.png', dpi=150)

Structure Parameter Evolution
------------------------------

.. code-block:: python

   from rheojax.models import DMTLocal
   import matplotlib.pyplot as plt
   import numpy as np

   model = DMTLocal(closure="exponential", include_elasticity=True)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Simulate step strain rate experiment
   fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

   # Shear rate protocol
   t_total = 300  # seconds
   t1, t2, t3 = 100, 200, 300
   gamma_dot_protocol = np.zeros(t_total)
   gamma_dot_protocol[0:t1] = 0.0    # Rest
   gamma_dot_protocol[t1:t2] = 10.0  # Shear
   gamma_dot_protocol[t2:t3] = 0.0   # Rest again

   # Simulate
   t_sim = np.linspace(0.1, t_total, 1000)
   results = model.simulate_step_protocol(t_sim, gamma_dot_protocol)

   # Plot
   ax = axes[0]
   ax.plot(t_sim, gamma_dot_protocol, 'k-', linewidth=2)
   ax.set_ylabel('γ̇ (s⁻¹)', fontsize=12)
   ax.set_title('Step Shear Rate Protocol', fontsize=14)
   ax.grid(True, alpha=0.3)

   ax = axes[1]
   ax.plot(t_sim, results['lambda'], 'b-', linewidth=2)
   ax.axhline(1.0, color='gray', linestyle='--', label='Fully structured')
   ax.axhline(0.0, color='gray', linestyle='--', label='Fully broken')
   ax.set_ylabel('Structure λ', fontsize=12)
   ax.set_ylim([-0.05, 1.05])
   ax.legend(fontsize=10)
   ax.set_title('Structure Parameter Evolution', fontsize=14)
   ax.grid(True, alpha=0.3)

   ax = axes[2]
   ax.plot(t_sim, results['sigma'], 'r-', linewidth=2)
   ax.set_xlabel('Time t (s)', fontsize=12)
   ax.set_ylabel('σ (Pa)', fontsize=12)
   ax.set_title('Stress Response', fontsize=14)
   ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig('structure_evolution.png', dpi=150)

Model Comparison
================

Choosing Among the Five Frameworks
-----------------------------------

.. list-table:: Framework Selection Guide
   :header-rows: 1
   :widths: 15 20 20 25 20

   * - Framework
     - Best Material Class
     - Key Advantage
     - Key Limitation
     - Typical CPU Time
   * - **DMT**
     - Oilfield fluids, paints, foods
     - Simple, well-established
     - Scalar structure only
     - Fast (< 1 s)
   * - **Fluidity**
     - Soft glasses, emulsions
     - Nonlocal shear banding
     - PDE solver needed
     - Moderate (1-10 s)
   * - **HL**
     - Dense suspensions
     - Mean-field avalanches
     - No spatial resolution
     - Slow (10-60 s)
   * - **STZ**
     - Metallic glasses
     - Disorder physics
     - Many parameters
     - Moderate (1-10 s)
   * - **EPM**
     - Amorphous solids
     - Spatial heterogeneity
     - Stochastic avalanches
     - Slow (10-100 s)

**Decision tree:**

1. **Do you need spatial resolution?**
      - Yes → Fluidity Nonlocal or EPM
      - No → DMT, HL, or STZ

2. **Is the material a dense colloidal suspension?**
      - Yes → HL or Fluidity
      - No → Continue

3. **Is it an industrial fluid (petroleum, food, cosmetic)?**
      - Yes → DMT (best documented for these)
      - No → Continue

4. **Is it a metallic glass or amorphous solid?**
      - Yes → STZ or EPM
      - No → DMT (general-purpose)

5. **Do you need normal stress predictions?**
      - Yes → Fluidity Saramito (tensorial)
      - No → Any framework

Comparison on Same Data
------------------------

.. code-block:: python

   from rheojax.models import DMTLocal, FluidityLocal, HebraudLequeux, STZConventional
   from rheojax.io.readers import auto_read
   import matplotlib.pyplot as plt
   import numpy as np

   # Load data
   data = auto_read("flow_curve.csv")
   gamma_dot = data.x
   sigma = data.y

   # Fit all models
   models = {
       'DMT': DMTLocal(closure="exponential", include_elasticity=False),
       'Fluidity': FluidityLocal(),
       'HL': HebraudLequeux(),
       'STZ': STZConventional(variant="minimal")
   }

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.ravel()

   for i, (name, model) in enumerate(models.items()):
       ax = axes[i]

       # Fit
       try:
           model.fit(gamma_dot, sigma, test_mode='flow_curve')

           # Predict
           sigma_pred = model.predict(gamma_dot, test_mode='flow_curve')

           # Plot
           ax.loglog(gamma_dot, sigma, 'ko', label='Data', markersize=4)
           ax.loglog(gamma_dot, sigma_pred, 'r-', label='Fit', linewidth=2)

           # R²
           ss_res = np.sum((sigma - sigma_pred)**2)
           ss_tot = np.sum((sigma - np.mean(sigma))**2)
           r2 = 1 - ss_res / ss_tot

           ax.set_xlabel('γ̇ (s⁻¹)')
           ax.set_ylabel('σ (Pa)')
           ax.set_title(f'{name} Model (R² = {r2:.4f})')
           ax.legend()
           ax.grid(True, alpha=0.3)

       except Exception as e:
           ax.text(0.5, 0.5, f'{name} failed:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
           ax.set_title(f'{name} Model (FAILED)')

   plt.tight_layout()
   plt.savefig('model_comparison.png', dpi=150)

Limitations and Caveats
========================

Scalar Structure Parameter Limitations
---------------------------------------

Most thixotropic models (DMT, Fluidity, HL minimal, STZ minimal) use a **scalar** structure
parameter :math:`\lambda` or :math:`f`. This is a drastic simplification:

- Real microstructure is **tensorial**: networks orient under shear
- **Anisotropy** not captured: structure breaks differently in different directions
- **Particle-level details** lost: individual bonds, clusters, flocs

For materials where orientation matters (polymer solutions, rod-like colloids, wormlike
micelles), consider:

- Tensorial extensions (Fluidity Saramito with orientation tensor)
- Molecular models (Giesekus, Pom-Pom, FENE)

Mean-Field Approximations
--------------------------

HL and SGR are **mean-field theories**: each element feels the average effect of all others,
neglecting spatial correlations.

**Consequences:**

- **Avalanche statistics** oversimplified (real avalanches have power-law distributions)
- **Shear banding** not captured unless explicitly added (nonlocal extensions)
- **Strain localization** missed

For avalanche-dominated systems, consider EPM (explicit spatial correlations) or mode-coupling
theory extensions.

Timescale Separation Assumption
--------------------------------

Most models assume **instantaneous elastic response** compared to structural relaxation:

.. math::

   \tau_{\text{elastic}} \ll \tau_{\text{structural}} = t_{\text{eq}}

If this separation fails (e.g., :math:`\tau_\text{elastic} \approx t_\text{eq}`), stress and structure evolve on similar
timescales, complicating analysis.

**Diagnostic**: If startup overshoot time :math:`\approx` structural relaxation time, models may struggle.

LAOS Limitations
----------------

**LAOS (Large Amplitude Oscillatory Shear) is challenging for thixotropic models:**

- Structure parameter oscillates every cycle
- May not reach steady state within experiment duration
- Harmonic decomposition (Fourier) obscures transient structure evolution

**Recommendation**: Use **SPP (Sequence of Physical Processes)** analysis instead of
Fourier harmonics for thixotropic LAOS.

.. code-block:: python

   from rheojax.transforms import SPPDecomposer

   # SPP operates in time domain, revealing intracycle structure evolution
   spp = SPPDecomposer(n_harmonics=39)
   result = spp.decompose(t, gamma, stress)

   # Extract cage modulus, yield stress, flow curve parameters
   G_cage = result['cage_modulus']
   sigma_y_static = result['static_yield_stress']
   sigma_y_dynamic = result['dynamic_yield_stress']

See :doc:`spp_analysis` for details.

Tutorial Notebooks
==================

DMT Thixotropic Model Tutorials
--------------------------------

**Basic DMT Workflows** (examples/dmt/01-06):

- ``01_dmt_flow_curve.ipynb``: Flow curve fitting with both closures
- ``02_dmt_startup_shear.ipynb``: Stress overshoot analysis
- ``03_dmt_stress_relaxation.ipynb``: Relaxation with structural recovery
- ``04_dmt_creep.ipynb``: Viscosity bifurcation demonstration
- ``05_dmt_saos.ipynb``: Small-amplitude oscillation (limited thixotropy)
- ``06_dmt_laos.ipynb``: Large-amplitude with structure oscillation

Fluidity Model Tutorials
-------------------------

**Fluidity Local** (examples/fluidity/01-06):

- ``01_fluidity_local_flow_curve.ipynb``
- ``02_fluidity_local_startup.ipynb``
- ``03_fluidity_local_creep.ipynb``
- ``04_fluidity_local_relaxation.ipynb``
- ``05_fluidity_local_saos.ipynb``
- ``06_fluidity_local_laos.ipynb``

**Fluidity Nonlocal** (examples/fluidity/07-12):

- ``07_fluidity_nonlocal_flow_curve.ipynb``: Shear banding detection
- ``08_fluidity_nonlocal_startup.ipynb``: Band formation dynamics
- ``09_fluidity_nonlocal_creep.ipynb``: Spatial heterogeneity in creep
- ``10_fluidity_nonlocal_relaxation.ipynb``
- ``11_fluidity_nonlocal_saos.ipynb``
- ``12_fluidity_nonlocal_laos.ipynb``

**Saramito EVP** (examples/fluidity/13-24):

- ``13-24_saramito_*.ipynb``: Local and nonlocal variants with tensorial stress

Hébraud-Lequeux Tutorials
--------------------------

**HL Model** (examples/hl/01-06):

- ``01_hl_flow_curve.ipynb``: Yield stress from mean-field theory
- ``02_hl_relaxation.ipynb``
- ``03_hl_creep.ipynb``
- ``04_hl_saos.ipynb``
- ``05_hl_startup.ipynb``
- ``06_hl_laos.ipynb``

STZ and EPM Tutorials
----------------------

**STZ** (examples/stz/01-06): Shear transformation zone theory for metallic glasses

**EPM** (examples/epm/01-06): Elastoplastic lattice models with avalanches

References
==========

**DMT Framework:**

- de Souza Mendes, P. R. (2009). "Modeling the thixotropic behavior of structured fluids."
  *J. Non-Newtonian Fluid Mech.* 164, 66-75.
  https://doi.org/10.1016/j.jnnfm.2009.08.005

- de Souza Mendes, P. R. & Thompson, R. L. (2012). "A critical overview of elasto-viscoplastic
  thixotropic modeling." *J. Non-Newtonian Fluid Mech.* 187-188, 8-15.
  https://doi.org/10.1016/j.jnnfm.2012.08.006

**Fluidity Models:**

- Coussot, P., Nguyen, Q. D., Huynh, H. T., & Bonn, D. (2002). "Viscosity bifurcation in
  thixotropic, yielding fluids." *J. Rheol.* 46, 573-589.
  https://doi.org/10.1122/1.1459447

- Saramito, P. (2007). "A new constitutive equation for elastoviscoplastic fluid flows."
  *J. Non-Newtonian Fluid Mech.* 145, 1-14.
  https://doi.org/10.1016/j.jnnfm.2007.04.004

- Saramito, P. (2009). "A new elastoviscoplastic model based on the Herschel-Bulkley viscoplastic
  model." *J. Non-Newtonian Fluid Mech.* 158, 154-161.
  https://doi.org/10.1016/j.jnnfm.2008.12.001

**Hébraud-Lequeux Theory:**

- Hébraud, P. & Lequeux, F. (1998). "Mode-coupling theory for the pasty rheology of soft
  glassy materials." *Phys. Rev. Lett.* 81, 2934-2937.
  https://doi.org/10.1103/PhysRevLett.81.2934

**STZ Theory:**

- Falk, M. L. & Langer, J. S. (1998). "Dynamics of viscoplastic deformation in amorphous
  solids." *Phys. Rev. E* 57, 7192-7205.
  https://doi.org/10.1103/PhysRevE.57.7192

- Langer, J. S. (2008). "Shear-transformation-zone theory of plastic deformation near the
  glass transition." *Phys. Rev. E* 77, 021502.
  https://doi.org/10.1103/PhysRevE.77.021502

**EPM (Elastoplastic Models):**

- Picard, G., Ajdari, A., Lequeux, F., & Bocquet, L. (2005). "Slow flows of yield stress
  fluids: Complex spatiotemporal behavior within a simple elastoplastic model."
  *Phys. Rev. E* 71, 010501(R).
  https://doi.org/10.1103/PhysRevE.71.010501

**Reviews:**

- Mewis, J. & Wagner, N. J. (2009). "Thixotropy." *Adv. Colloid Interface Sci.* 147-148, 214-227.
  https://doi.org/10.1016/j.cis.2008.09.005

- Bonn, D., Denn, M. M., Berthier, L., Divoux, T., & Manneville, S. (2017). "Yield stress
  materials in soft condensed matter." *Rev. Mod. Phys.* 89, 035005.
  https://doi.org/10.1103/RevModPhys.89.035005

- Larson, R. G. & Wei, Y. (2019). "A review of thixotropy and its rheological modeling."
  *J. Rheol.* 63, 477-501.
  https://doi.org/10.1122/1.5055031

See Also
========

- :doc:`/models/dmt/index` — DMT model handbook with detailed equations
- :doc:`/models/fluidity/index` — Fluidity and Saramito EVP models
- :doc:`/models/hl/index` — Hébraud-Lequeux mean-field theory
- :doc:`/models/stz/index` — STZ theory for amorphous solids
- :doc:`/models/epm/index` — Elastoplastic lattice models
- :doc:`spp_analysis` — SPP analysis for thixotropic LAOS
- :doc:`sgr_analysis` — SGR soft glassy rheology (complementary framework)
- :doc:`bayesian_inference` — Uncertainty quantification for thixotropic parameters
- :doc:`/examples/index` — Tutorial notebooks for all five frameworks
