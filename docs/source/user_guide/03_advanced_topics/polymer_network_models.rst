.. _polymer_network_models:

=====================================================
Transient Polymer Network Models (TNT + VLB)
=====================================================

**Dynamic Crosslink Models for Associative Polymers, Micelles, and Gels**

Overview
========

Transient polymer networks are materials where crosslinks dynamically break and reform
on experimental timescales. Unlike permanent networks (rubbers), these materials exhibit
complex time-dependent behavior combining solid-like elasticity and liquid-like flow.

.. admonition:: Key Insight
   :class: tip

   Transient networks are characterized by two competing timescales:
   
   - **Network relaxation time** (:math:`\tau_{\text{net}}`): How quickly crosslinks reform
   - **Applied deformation rate** (:math:`1/\dot{\gamma}`): How quickly the material is deformed

   When :math:`\tau_{\text{net}} \ll 1/\dot{\gamma}`, the network relaxes faster than deformation → **liquid-like**

   When :math:`\tau_{\text{net}} \gg 1/\dot{\gamma}`, the network is frozen on deformation timescale → **solid-like**

RheoJAX provides two complementary theoretical frameworks with 9 total model variants:

**TNT Framework (5 variants)** — Discrete chain populations with kinetic equations

**VLB Framework (4 variants)** — Distribution tensor approach with continuum mechanics

Both frameworks predict identical phenomena (stress overshoot, shear thinning, non-Newtonian
flow) but use different mathematical formalisms. Choose based on your physical intuition
and modeling needs.

When to Use Transient Network Models
-------------------------------------

**Ideal for:**

- **Associative polymers**: Ionomers, hydrogels, supramolecular polymers
- **Wormlike micelles**: Surfactant solutions with reversible micellar scission
- **Telechelic polymers**: End-functionalized chains with reversible sticker associations
- **Vitrimers**: Covalent adaptable networks (see HVM/HVNM for dedicated vitrimer models)
- **Biological gels**: Protein networks with dynamic crosslinks (actin, fibrin)
- **Stress overshoot**: Materials showing transient stress peaks during startup shear
- **Shear thinning**: Viscosity decreasing with shear rate due to bond rupture

**Choose TNT when:**

- You think in terms of **discrete chain populations** (loops, bridges, etc.)
- You need **multiple relaxation modes** (multi-species networks)
- Your material has **distinct kinetic processes** (reptation + breakage in micelles)
- You want **simple analytical expressions** for SAOS

**Choose VLB when:**

- You prefer **tensor formulations** (continuum mechanics intuition)
- You need **force-sensitive kinetics** (Bell model for bond rupture)
- Your material shows **chain extensibility limits** (FENE stiffening)
- You want to model **spatial heterogeneity** (shear banding with nonlocal variant)

Material Classification
-----------------------

.. list-table:: Transient Network Material Types
   :header-rows: 1
   :widths: 25 30 45

   * - Material Class
     - Crosslink Type
     - Example Systems
   * - **Associative Polymers**
     - Reversible physical bonds
     - Ionomers, supramolecular polymers, hydrogels
   * - **Wormlike Micelles**
     - Reversible scission/recombination
     - CTAB, CPyCl surfactant solutions
   * - **Telechelic Polymers**
     - End-group stickers
     - PEO-PPO-PEO (Pluronics), HEUR thickeners
   * - **Vitrimers**
     - Exchangeable covalent bonds
     - Polyester vitrimers (see HVM/HVNM for specialized models)
   * - **Biological Gels**
     - Protein crosslinks
     - Actin networks, fibrin gels

Theoretical Foundations
=======================

The Transient Network Concept
------------------------------

A transient network consists of polymer chains connected by **temporary crosslinks**
that can break and reform. Each crosslink has a characteristic **lifetime** :math:`\tau` before
detaching, and a characteristic **attachment time** before reforming.

.. math::

   \text{Network lifetime: } \tau_{\text{net}} = \frac{1}{k_d} \quad \text{where } k_d = \text{detachment rate}

The network's rheological behavior depends on how :math:`\tau_{\text{net}}` compares to experimental timescales:

**Small Amplitude Oscillatory Shear (SAOS):**

At low frequencies (:math:`\omega \ll k_d`): Material flows like a liquid, :math:`G'' > G'`

At high frequencies (:math:`\omega \gg k_d`): Network is frozen, elastic response, :math:`G' > G''`

Crossover frequency: :math:`\omega_c \approx k_d`

**Steady Shear Flow:**

At low shear rates (:math:`\dot{\gamma} \ll k_d`): Newtonian plateau viscosity

At high shear rates (:math:`\dot{\gamma} \gg k_d`): Shear thinning as bonds break faster than they reform

**Startup Shear:**

Initial elastic response → stress overshoot → steady state flow

Overshoot magnitude controlled by chain stretch before bond rupture

Distribution Tensor Concept (VLB Framework)
-------------------------------------------

The VLB framework uses a **distribution tensor** :math:`\mu(t)` to represent the ensemble-averaged
end-to-end vector distribution of network chains:

.. math::

   \mu_{ij}(t) = \frac{\langle R_i R_j \rangle}{R_0^2}

where :math:`R` is the end-to-end vector and :math:`R_0` is the equilibrium length. The stress is:

.. math::

   \sigma = G_0 (\mu - I)

For an equilibrium network: :math:`\mu = I` (isotropic distribution)

Under deformation: :math:`\mu \neq I` (chains stretch/orient)

**Evolution equation** (stationary chains):

.. math::

   \frac{D\mu}{Dt} = \kappa \cdot \mu + \mu \cdot \kappa^T - k_d(\mu - I) + k_a(I - \mu)

where:
   - :math:`D\mu/Dt`: Material derivative (advection with flow)
   - :math:`\kappa`: Velocity gradient tensor
   - :math:`k_d`: Detachment rate (bond breaking)
   - :math:`k_a`: Attachment rate (bond formation)

Attachment vs Detachment Kinetics
----------------------------------

**Constant Rate Model** (simplest):

.. math::

   k_d = k_{d,0} = \text{constant}, \quad k_a = k_{a,0} = \text{constant}

Suitable for: Weak forces, low strain rates, thermoactivated processes

**Force-Enhanced Detachment (Bell Model)**:

.. math::

   k_d = k_{d,0} \exp\left(\frac{F}{F_{\text{ref}}}\right)

where :math:`F` is the force on the chain. Higher force → faster detachment → shear thinning

Suitable for: Load-bearing networks, materials with force-sensitive bonds

**FENE Extensibility Limit**:

.. math::

   F = \frac{k_B T}{b} \frac{r/r_{\max}}{1 - (r/r_{\max})^2}

Chains stiffen as they approach maximum extension :math:`r_{\max}`

Suitable for: Materials showing strain stiffening at large deformations

TNT Framework (5 Variants)
===========================

TNT models describe the network as **discrete populations** of chains with different
states or kinetics. Total stress is the sum over all populations.

TNT SingleMode — Basic Transient Network
-----------------------------------------

The simplest transient network model: one population of chains with stress-dependent
detachment.

.. code-block:: python

   from rheojax.models import TNTSingleMode
   
   model = TNTSingleMode()
   model.fit(omega, G_star, test_mode='oscillation')
   
   G_0 = model.parameters.get_value('G_0')
   k_d_0 = model.parameters.get_value('k_d_0')
   beta = model.parameters.get_value('beta')
   
   tau_net = 1.0 / k_d_0
   print(f"Network relaxation time: {tau_net:.2f} s")

**Parameters:**

- :math:`G_0` (Pa): Network modulus (number density :math:`\times k_B T`)
- **k_d_0** (:math:`\text{s}^{-1}`): Baseline detachment rate (inverse lifetime)
- **beta** (dimensionless): Nonlinearity parameter (stress-enhanced detachment)

**Physics:**

Chain population :math:`\phi` evolves as:

.. math::

   \frac{d\phi}{dt} = -k_d(\sigma) \phi + k_a (1 - \phi)

where :math:`k_d = k_{d,0} \times (1 + \beta|\sigma/G_0|)`

**Use for:**

- Quick screening of transient network behavior
- Understanding basic network dynamics
- Materials with single dominant relaxation process

**Notebooks**: ``examples/tnt/01-06``

TNT Cates — Wormlike Micelles (Living Polymers)
------------------------------------------------

Specialized model for **living polymers** where chain length fluctuates via reversible
scission and recombination. Key feature: **geometric mean relaxation time**.

.. code-block:: python

   from rheojax.models import TNTCates
   
   model = TNTCates()
   model.fit(omega, G_star, test_mode='oscillation')
   
   tau_rep = model.parameters.get_value('tau_rep')
   tau_break = model.parameters.get_value('tau_break')
   
   # Effective relaxation from competition of processes
   tau_eff = np.sqrt(tau_rep * tau_break)
   print(f"Reptation time: {tau_rep:.2f} s")
   print(f"Breakage time: {tau_break:.2f} s")
   print(f"Effective relaxation: {tau_eff:.2f} s (geometric mean)")

**Parameters:**

- :math:`G_0` (Pa): Plateau modulus
- **tau_rep** (s): Reptation time (chain sliding along tube)
- **tau_break** (s): Breakage time (scission/recombination)

**Physics:**

Chains relax via two competing mechanisms:

1. **Reptation**: Chain diffuses along its contour (slow)
2. **Breakage**: Chain breaks and reforms into shorter segments (fast)

When :math:`\tau_{\text{break}} \ll \tau_{\text{rep}}`: Breakage dominates → :math:`\tau_{\text{eff}} \approx \sqrt{\tau_{\text{rep}} \times \tau_{\text{break}}}`

This gives **single effective mode** despite two processes.

**Characteristic features:**

- Single Maxwell peak in :math:`G'`, :math:`G''` vs :math:`\omega`
- Shear thinning at high rates (breakage accelerated)
- Stress plateau in startup shear

**Use for:**

- CTAB, CPyCl, and other wormlike micellar solutions
- Systems showing single-exponential relaxation despite complex microstructure
- Materials where reptation and scission compete

**Notebooks**: ``examples/tnt/07-12``

.. note::

   The Cates model predicts a **single exponential relaxation** for living polymers,
   unlike typical polymers that show multi-mode spectra. This is a signature prediction
   validated by extensive experimental data on wormlike micelles.

TNT LoopBridge — Telechelic Polymers
-------------------------------------

Model for **telechelic polymers** (end-functionalized chains) where chains can exist as
**loops** (both ends attached to same micelle) or **bridges** (ends on different micelles).
Only bridges contribute to stress.

.. code-block:: python

   from rheojax.models import TNTLoopBridge
   
   model = TNTLoopBridge()
   model.fit(omega, G_star, test_mode='oscillation')
   
   phi_bridge_0 = model.parameters.get_value('phi_bridge_0')
   k_detach = model.parameters.get_value('k_detach')
   k_attach = model.parameters.get_value('k_attach')
   
   print(f"Equilibrium bridge fraction: {phi_bridge_0:.2f}")
   print(f"Bridge lifetime: {1.0/k_detach:.2f} s")

**Parameters:**

- :math:`G_0` (Pa): Modulus scale (bridge contribution)
- **k_detach** (:math:`\text{s}^{-1}`): Bridge → loop detachment rate
- **k_attach** (:math:`\text{s}^{-1}`): Loop → bridge attachment rate
- **phi_bridge_0** (dimensionless): Equilibrium bridge fraction

**Physics:**

Two populations with interconversion:

.. math::

   \text{Bridge} \overset{k_{\text{detach}}}{\underset{k_{\text{attach}}}{\rightleftharpoons}} \text{Loop}

Under flow:

- Bridges stretch → higher stress-assisted detachment
- Loops convert to bridges to replace detached bridges
- Steady state: balance of conversion rates

**Characteristic features:**

- Two-mode relaxation (fast: detachment, slow: loop/bridge exchange)
- Stronger shear thinning than single-mode TNT
- Stress overshoot in startup more pronounced

**Use for:**

- Pluronics (PEO-PPO-PEO triblock copolymers)
- HEUR associative thickeners
- End-functionalized polymers with sticker groups
- Hydrogels with reversible crosslinks

**Notebooks**: ``examples/tnt/13-18``

TNT MultiSpecies — Multi-Population Networks
---------------------------------------------

General framework for :math:`N` **distinct chain populations** with different kinetics. Captures
broad relaxation spectra without assuming specific microstructure.

.. code-block:: python

   from rheojax.models import TNTMultiSpecies
   
   model = TNTMultiSpecies(n_species=3)
   model.fit(omega, G_star, test_mode='oscillation')
   
   for i in range(3):
       G_i = model.parameters.get_value(f'G_{i}')
       k_d_i = model.parameters.get_value(f'k_d_{i}')
       tau_i = 1.0 / k_d_i
       print(f"Species {i}: G = {G_i:.1f} Pa, τ = {tau_i:.2e} s")

**Parameters:**

- **G_i** (Pa): Modulus contribution of species :math:`i`
- **k_d_i** (:math:`\text{s}^{-1}`): Detachment rate of species :math:`i`
- **(Optional) beta** (dimensionless): Shared nonlinearity parameter

Total: :math:`2N + 1` parameters for :math:`N` species

**Physics:**

Each species evolves independently:

.. math::

   \frac{d\phi_i}{dt} = -k_{d,i}(\sigma_i) \phi_i + k_{a,i} (1 - \phi_i)

Total stress: :math:`\sigma_{\text{total}} = \sum G_i \phi_i`

**Use for:**

- Polydisperse networks (distribution of chain lengths)
- Multi-component systems (different polymer types)
- Materials with broad relaxation spectra
- Fitting complex frequency-dependent data

**Notebooks**: ``examples/tnt/19-24``

.. warning::

   Multi-species models have many parameters (:math:`2N+1`). Use Bayesian inference to
   quantify parameter uncertainty and avoid overfitting. Start with :math:`N=2\text{--}3` species
   and increase only if diagnostics show clear improvement.

TNT StickyRouse — Sticky Rouse Chains
--------------------------------------

Combines **Rouse dynamics** (chain internal modes) with **reversible stickers** along
the backbone. Captures both chain relaxation and association/dissociation.

.. code-block:: python

   from rheojax.models import TNTStickyRouse
   
   model = TNTStickyRouse()
   model.fit(omega, G_star, test_mode='oscillation')
   
   G_0 = model.parameters.get_value('G_0')
   tau_R = model.parameters.get_value('tau_R')
   tau_s = model.parameters.get_value('tau_s')
   N_s = model.parameters.get_value('N_s')
   
   print(f"Rouse time: {tau_R:.2e} s")
   print(f"Sticker lifetime: {tau_s:.2e} s")
   print(f"Stickers per chain: {N_s:.0f}")

**Parameters:**

- :math:`G_0` (Pa): Plateau modulus
- **tau_R** (s): Longest Rouse relaxation time
- **tau_s** (s): Sticker association lifetime
- **N_s** (dimensionless): Number of stickers per chain

**Physics:**

Two contributions to stress:

1. **Rouse modes**: Chain internal relaxation (modes :math:`p = 1, 2, \ldots, N_s`)
2. **Sticker network**: Elastic contribution from associations

Relaxation times: :math:`\tau_p = \tau_R / p^2` (Rouse spectrum)

Effective modulus combines Rouse + sticker contributions

**Characteristic features:**

- Multi-mode relaxation spectrum
- Fast modes: chain internal dynamics
- Slow modes: sticker association/dissociation
- Rubbery plateau at intermediate frequencies

**Use for:**

- Ionomers (ion-pair associations along backbone)
- Supramolecular polymers with side-group interactions
- Hydrogels with hydrogen bonding
- Materials showing Rouse-like dynamics with associations

**Notebooks**: ``examples/tnt/25-30``

VLB Framework (4 Variants)
===========================

VLB models use a **distribution tensor** :math:`\mu` to describe chain end-to-end vector statistics.
More mathematically sophisticated than TNT but provides unified treatment of force effects.

VLB Local — Basic Distribution Tensor
--------------------------------------

The foundational VLB model for **homogeneous flow** (no spatial variations). All VLB
variants extend this base model.

.. code-block:: python

   from rheojax.models import VLBLocal
   
   model = VLBLocal()
   model.fit(omega, G_star, test_mode='oscillation')
   
   G_0 = model.parameters.get_value('G_0')
   k_d_0 = model.parameters.get_value('k_d_0')
   k_a_0 = model.parameters.get_value('k_a_0')
   
   tau_d = 1.0 / k_d_0
   tau_a = 1.0 / k_a_0
   print(f"Detachment time: {tau_d:.2f} s")
   print(f"Attachment time: {tau_a:.2f} s")

**Parameters:**

- :math:`G_0` (Pa): Network modulus
- **k_d_0** (:math:`\text{s}^{-1}`): Baseline detachment rate
- **k_a_0** (:math:`\text{s}^{-1}`): Attachment rate
- **(Optional) beta** (dimensionless): Nonlinearity parameter

**Governing equations:**

Distribution tensor evolution:

.. math::

   \frac{D\mu}{Dt} = \kappa \cdot \mu + \mu \cdot \kappa^T - k_d(\mu - I) + k_a(I - \mu)

Stress:

.. math::

   \sigma = G_0 (\mu - I)

For SAOS at frequency :math:`\omega`:

.. math::

   G^*(\omega) = \frac{G_0 i\omega}{i\omega + k_d + k_a}

This gives single Maxwell mode with :math:`\tau = 1/(k_d + k_a)`

**Use for:**

- Homogeneous transient networks
- Comparison with TNT SingleMode (should give similar predictions)
- Foundation for more complex VLB variants
- Materials without force-sensitive kinetics

**Notebooks**: ``examples/vlb/01-06``

.. note::

   VLB Local with constant :math:`k_d`, :math:`k_a` is mathematically equivalent to TNT SingleMode
   with :math:`\beta=0`. The difference is formalism: tensor (VLB) vs population (TNT).

VLB MultiNetwork — Multiple Parallel Networks
----------------------------------------------

Extension to :math:`N` **independent networks**, each with its own distribution tensor. Useful
for polydisperse systems or multi-component materials.

.. code-block:: python

   from rheojax.models import VLBMultiNetwork
   
   model = VLBMultiNetwork(n_networks=3)
   model.fit(omega, G_star, test_mode='oscillation')
   
   for i in range(3):
       G_i = model.parameters.get_value(f'G_0_{i}')
       k_d_i = model.parameters.get_value(f'k_d_0_{i}')
       tau_i = 1.0 / k_d_i
       print(f"Network {i}: G = {G_i:.1f} Pa, τ = {tau_i:.2e} s")

**Parameters:**

- **G_0_i** (Pa): Modulus of network :math:`i`
- **k_d_0_i** (:math:`\text{s}^{-1}`): Detachment rate of network :math:`i`
- **k_a_0_i** (:math:`\text{s}^{-1}`): Attachment rate of network :math:`i`

Total: :math:`3N` parameters for :math:`N` networks

**Physics:**

Each network :math:`i` has independent :math:`\mu_i` evolving according to VLB equation.

Total stress: :math:`\sigma_{\text{total}} = \sum G_{0,i} (\mu_i - I)`

**Use for:**

- Same applications as TNT MultiSpecies
- Multi-modal relaxation spectra
- Polydisperse chain length distributions

**Notebooks**: ``examples/vlb/07`` (included in basic tutorials)

VLB Variant — Force-Sensitive Kinetics
---------------------------------------

Advanced VLB with **Bell force-enhanced detachment** and/or **FENE extensibility limit**.
Captures shear thinning and strain stiffening.

.. code-block:: python

   from rheojax.models import VLBVariant
   
   # Bell model only
   model = VLBVariant(include_bell=True, include_fene=False)
   model.fit(gamma_dot, sigma, test_mode='flow_curve')
   
   F_ref = model.parameters.get_value('F_ref')
   print(f"Bell force scale: {F_ref:.2e} N")
   
   # FENE model only
   model_fene = VLBVariant(include_bell=False, include_fene=True)
   model_fene.fit(gamma_dot, sigma, test_mode='flow_curve')
   
   b = model_fene.parameters.get_value('b')
   print(f"FENE parameter: {b:.1f} (dimensionless)")
   
   # Combined Bell + FENE
   model_full = VLBVariant(include_bell=True, include_fene=True)

**Additional parameters:**

- **F_ref** (N): Bell force scale (force-enhanced detachment)
- **b** (dimensionless): FENE parameter (finite extensibility)

**Bell model physics:**

Force on chain: :math:`F = G_0 \times |\mu - I|`

Detachment rate: :math:`k_d = k_{d,0} \times \exp(F / F_{\text{ref}})`

Higher stretch → higher force → faster detachment → shear thinning

**FENE physics:**

Chain spring force:

.. math::

   F = \frac{k_B T}{b} \frac{\lambda}{1 - \lambda^2}

where :math:`\lambda = r/r_{\max}` is normalized extension

As :math:`\lambda \to 1`: Force diverges (chains cannot extend beyond :math:`r_{\max}`) → strain stiffening

**Characteristic features:**

- **Bell only**: Shear thinning in flow curves, stress plateau at high :math:`\dot{\gamma}`
- **FENE only**: Strain stiffening at large deformations
- **Bell + FENE**: Both shear thinning (low :math:`\dot{\gamma}`) and stiffening (large strain)

**Use for:**

- Load-bearing biological networks (force-sensitive bonds)
- Materials showing pronounced shear thinning
- Systems with finite chain extensibility
- Strain-stiffening gels (FENE)

**Notebooks**: ``examples/vlb/08-09``

VLB Nonlocal — Spatial PDE for Shear Banding
---------------------------------------------

PDE extension with **spatial coupling** to capture heterogeneous flow. Predicts shear
banding when constitutive curve is non-monotonic.

.. code-block:: python

   from rheojax.models import VLBNonlocal
   
   model = VLBNonlocal(n_points=51, gap_width=1e-3)
   
   # Simulate steady shear with banding
   result = model.simulate_steady_shear(
       gamma_dot_avg=5.0,
       t_end=100.0
   )
   
   # Extract spatial profiles
   y_coords = result['y']
   gamma_dot_profile = result['gamma_dot_profile']
   
   # Detect banding
   banding = model.detect_banding(result, threshold=0.1)
   if banding['is_banded']:
       print(f"Shear banding detected!")
       print(f"High-rate band fraction: {banding['high_fraction']:.2f}")

**Additional parameters:**

- **n_points** (int): Spatial discretization (default: 51)
- **D** (:math:`\text{m}^2/\text{s}`): Stress diffusion coefficient
- **gap_width** (m): Gap width for spatial domain

**Physics:**

Distribution tensor varies in space: :math:`\mu(y, t)`

PDE evolution:

.. math::

   \frac{\partial \mu}{\partial t} = \ldots + D \frac{\partial^2 \mu}{\partial y^2}

Stress diffusion term couples neighboring spatial points.

When :math:`d\sigma/d\dot{\gamma} < 0` (non-monotonic constitutive curve): **Instability** → shear banding

Material separates into coexisting bands:
   - Low-rate band: high viscosity
   - High-rate band: low viscosity

**Use for:**

- Materials showing heterogeneous flow under shear
- Wormlike micelles (common banding systems)
- Materials with non-monotonic flow curves
- Spatially-resolved predictions (not just bulk averages)

**Notebooks**: ``examples/vlb/10``

.. warning::

   Shear banding simulations are computationally expensive (PDE solve at each timestep).
   Use coarse spatial discretization (n_points=21-51) for exploratory work, refine
   (n_points=101-201) for quantitative predictions.

Practical Implementation
========================

Multi-Protocol Workflow
-----------------------

Use the same model to predict all rheological tests:

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import numpy as np
   
   model = TNTSingleMode()
   
   # 1. Fit to SAOS data
   omega = np.logspace(-2, 2, 50)
   G_star_data = ...  # Complex modulus data
   model.fit(omega, G_star_data, test_mode='oscillation')
   
   # 2. Predict stress relaxation
   t = np.logspace(-2, 2, 100)
   G_t = model.predict(t, test_mode='relaxation')
   
   # 3. Predict flow curve
   gamma_dot = np.logspace(-3, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')
   
   # 4. Predict startup shear (stress overshoot)
   t_startup = np.linspace(0, 50, 500)
   sigma_startup = model.predict(
       t_startup,
       test_mode='startup',
       gamma_dot=1.0
   )
   
   # 5. Predict LAOS
   t_laos = np.linspace(0, 50, 1000)
   sigma_laos = model.predict(
       t_laos,
       test_mode='laos',
       gamma_0=0.5,
       omega=1.0
   )

Extracting Network Relaxation Time
-----------------------------------

The network relaxation time is a key physical parameter:

.. code-block:: python

   from rheojax.models import VLBLocal
   
   model = VLBLocal()
   model.fit(omega, G_star, test_mode='oscillation')
   
   k_d = model.parameters.get_value('k_d_0')
   k_a = model.parameters.get_value('k_a_0')
   
   # Total relaxation rate
   k_total = k_d + k_a
   tau_net = 1.0 / k_total
   
   print(f"Network relaxation time: {tau_net:.2f} s")
   
   # Compare to crossover frequency from SAOS
   omega_c = k_total
   print(f"Crossover frequency: {omega_c:.2f} rad/s")
   
   # Verify: at ω = ω_c, G' ≈ G''
   G_star_crossover = model.predict(omega_c, test_mode='oscillation')
   print(f"G' = {np.real(G_star_crossover):.1f} Pa")
   print(f"G'' = {np.imag(G_star_crossover):.1f} Pa")

Understanding Stress Overshoot
-------------------------------

Transient networks show **stress overshoot** during startup shear: stress rises initially
(elastic loading), peaks, then decays to steady flow. The overshoot magnitude and time
depend on network kinetics.

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import numpy as np
   import matplotlib.pyplot as plt
   
   model = TNTSingleMode(G_0=1000.0, k_d_0=1.0, beta=0.5)
   
   # Simulate startup at different shear rates
   t = np.linspace(0, 10, 500)
   
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   for gamma_dot in [0.1, 1.0, 10.0]:
       sigma = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)
       
       # Find overshoot
       idx_max = np.argmax(sigma)
       t_overshoot = t[idx_max]
       sigma_max = sigma[idx_max]
       sigma_steady = sigma[-1]
       
       overshoot_ratio = sigma_max / sigma_steady
       
       # Plot stress vs time
       axes[0].plot(t, sigma, label=f'γ̇ = {gamma_dot} s⁻¹')
       axes[0].scatter([t_overshoot], [sigma_max], color='red', s=50, zorder=5)
       
       # Plot stress vs strain (removes time effect)
       strain = gamma_dot * t
       axes[1].plot(strain, sigma, label=f'γ̇ = {gamma_dot} s⁻¹')
       
       print(f"γ̇ = {gamma_dot} s⁻¹: overshoot at t = {t_overshoot:.2f} s, "
             f"ratio = {overshoot_ratio:.2f}")
   
   axes[0].set_xlabel('Time (s)')
   axes[0].set_ylabel('Stress (Pa)')
   axes[0].set_title('Stress Overshoot vs Time')
   axes[0].legend()
   
   axes[1].set_xlabel('Strain')
   axes[1].set_ylabel('Stress (Pa)')
   axes[1].set_title('Stress Overshoot vs Strain (Strain Softening)')
   axes[1].legend()
   
   plt.tight_layout()
   plt.savefig('stress_overshoot_analysis.png', dpi=150)

**Key insights:**

- Overshoot time scales with network relaxation: :math:`t_{\text{overshoot}} \approx \tau_{\text{net}}`
- Higher shear rate → larger overshoot (more chain stretch before breaking)
- In strain-space (not time), overshoot occurs at similar strain across rates

Bayesian Inference for Network Parameters
==========================================

Network parameters (:math:`k_d`, :math:`k_a`, :math:`G_0`) often exhibit correlations. Bayesian inference
quantifies parameter uncertainty and identifies non-identifiability.

Basic Bayesian Workflow
------------------------

.. code-block:: python

   from rheojax.models import VLBLocal
   from rheojax.pipeline.bayesian import BayesianPipeline
   
   # 1. NLSQ point estimation (fast, GPU-accelerated)
   model = VLBLocal()
   model.fit(omega, G_star, test_mode='oscillation')
   
   # 2. Bayesian inference with warm-start
   result = model.fit_bayesian(
       omega, G_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )
   
   # 3. Check convergence
   print(f"R-hat: {result.diagnostics['r_hat']}")
   print(f"ESS: {result.diagnostics['ess']}")
   print(f"Divergences: {result.diagnostics['divergences']}")
   
   # 4. Get credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )
   
   for param, (low, high) in intervals.items():
       mean = np.mean(result.posterior_samples[param])
       print(f"{param}: {mean:.2e} [{low:.2e}, {high:.2e}]")

Using BayesianPipeline for Complete Workflow
---------------------------------------------

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline
   
   pipeline = (BayesianPipeline()
       .load('wormlike_micelles.csv', x_col='omega', y_col='G_star')
       .fit_nlsq('tnt_cates')
       .fit_bayesian(num_samples=2000, num_warmup=1000))
   
   # Generate all diagnostic plots
   (pipeline
       .plot_pair(divergences=True, show=False).save_figure('pair.pdf')
       .plot_forest(hdi_prob=0.95, show=False).save_figure('forest.pdf')
       .plot_autocorr(show=False).save_figure('autocorr.pdf')
       .plot_rank(show=False).save_figure('rank.pdf')
       .plot_ess(kind='local', show=False).save_figure('ess.pdf'))
   
   # Save results
   pipeline.save('bayesian_results.hdf5')

Detecting Parameter Correlations
---------------------------------

Network parameters often show correlations:

.. code-block:: python

   import numpy as np
   
   # Extract posterior samples
   G_0_samples = result.posterior_samples['G_0']
   k_d_samples = result.posterior_samples['k_d_0']
   
   # Compute correlation
   correlation = np.corrcoef(G_0_samples, k_d_samples)[0, 1]
   print(f"Correlation(G_0, k_d_0): {correlation:.3f}")
   
   # Visualize with pair plot
   pipeline.plot_pair(var_names=['G_0', 'k_d_0'], divergences=True)

**Interpretation:**

- :math:`|\rho| < 0.5`: Parameters well-identified
- :math:`0.5 < |\rho| < 0.8`: Moderate correlation (acceptable)
- :math:`|\rho| > 0.8`: Strong correlation (consider reparameterization)

**Common correlations:**

- :math:`G_0` and :math:`k_d`: Both affect modulus magnitude and timescale
- :math:`k_d` and :math:`k_a`: Combined determine network relaxation time

Visualization and Analysis
===========================

Flow Curve Analysis
-------------------

Plot steady-shear flow curves showing shear thinning:

.. code-block:: python

   from rheojax.models import VLBVariant
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Model with Bell force-enhanced detachment
   model = VLBVariant(include_bell=True, include_fene=False)
   model.parameters.set_value('G_0', 1000.0)
   model.parameters.set_value('k_d_0', 1.0)
   model.parameters.set_value('k_a_0', 0.1)
   model.parameters.set_value('F_ref', 1e-10)  # Bell force scale
   
   gamma_dot = np.logspace(-3, 3, 100)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')
   
   # Compute apparent viscosity
   eta_app = sigma / gamma_dot
   
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # Flow curve
   axes[0].loglog(gamma_dot, sigma)
   axes[0].set_xlabel('Shear Rate (s⁻¹)')
   axes[0].set_ylabel('Shear Stress (Pa)')
   axes[0].set_title('Flow Curve (Bell Model)')
   axes[0].grid(True, alpha=0.3)
   
   # Viscosity curve
   axes[1].loglog(gamma_dot, eta_app)
   axes[1].set_xlabel('Shear Rate (s⁻¹)')
   axes[1].set_ylabel('Apparent Viscosity (Pa·s)')
   axes[1].set_title('Shear Thinning')
   axes[1].grid(True, alpha=0.3)
   
   # Add power-law fit at high rates
   idx_high = gamma_dot > 10.0
   log_eta = np.log(eta_app[idx_high])
   log_rate = np.log(gamma_dot[idx_high])
   n_fit = np.polyfit(log_rate, log_eta, 1)[0]
   
   axes[1].text(0.05, 0.95, f'Power-law index: n = {n_fit:.2f}',
                transform=axes[1].transAxes, verticalalignment='top')
   
   plt.tight_layout()
   plt.savefig('flow_curve_analysis.png', dpi=150)

SAOS Master Curves
------------------

Compare multiple TNT/VLB models on same data:

.. code-block:: python

   from rheojax.models import TNTSingleMode, TNTCates, VLBLocal
   import numpy as np
   import matplotlib.pyplot as plt
   
   omega = np.logspace(-2, 2, 50)
   G_star_data = ...  # Your experimental data
   
   models = {
       'TNT SingleMode': TNTSingleMode(),
       'TNT Cates': TNTCates(),
       'VLB Local': VLBLocal()
   }
   
   fig, ax = plt.subplots(figsize=(8, 6))
   
   # Plot data
   ax.loglog(omega, np.real(G_star_data), 'o', label="G' data", color='C0')
   ax.loglog(omega, np.imag(G_star_data), 's', label="G'' data", color='C1')
   
   # Fit and plot each model
   for name, model in models.items():
       model.fit(omega, G_star_data, test_mode='oscillation')
       G_fit = model.predict(omega, test_mode='oscillation')
       
       ax.loglog(omega, np.real(G_fit), '-', label=f"{name} G'", alpha=0.7)
       ax.loglog(omega, np.imag(G_fit), '--', label=f"{name} G''", alpha=0.7)
       
       # Report R²
       r_squared = model._last_fit_result.r_squared
       print(f"{name}: R² = {r_squared:.4f}")
   
   ax.set_xlabel('ω (rad/s)')
   ax.set_ylabel('G\', G\'\' (Pa)')
   ax.set_title('Model Comparison (SAOS)')
   ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
   ax.grid(True, alpha=0.3)
   
   plt.tight_layout()
   plt.savefig('saos_model_comparison.png', dpi=150)

Shear Banding Visualization (Nonlocal)
---------------------------------------

.. code-block:: python

   from rheojax.models import VLBNonlocal
   import numpy as np
   import matplotlib.pyplot as plt
   
   model = VLBNonlocal(n_points=51, gap_width=1e-3)
   
   # Parameters that produce banding (non-monotonic flow curve)
   model.parameters.set_value('G_0', 1000.0)
   model.parameters.set_value('k_d_0', 1.0)
   model.parameters.set_value('k_a_0', 0.5)
   model.parameters.set_value('beta', 1.5)  # Strong nonlinearity
   
   # Simulate steady shear
   result = model.simulate_steady_shear(
       gamma_dot_avg=5.0,
       t_end=200.0
   )
   
   y = result['y']
   gamma_dot_profile = result['gamma_dot_profile']
   sigma_profile = result['stress_profile']
   
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   
   # Shear rate profile
   axes[0].plot(y * 1e3, gamma_dot_profile)
   axes[0].set_xlabel('Position y (mm)')
   axes[0].set_ylabel('Local Shear Rate (s⁻¹)')
   axes[0].set_title('Shear Banding: Spatial Profile')
   axes[0].grid(True, alpha=0.3)
   
   # Stress profile (should be constant)
   axes[1].plot(y * 1e3, sigma_profile)
   axes[1].set_xlabel('Position y (mm)')
   axes[1].set_ylabel('Local Stress (Pa)')
   axes[1].set_title('Stress Profile (Plateau)')
   axes[1].grid(True, alpha=0.3)
   
   # Detect bands
   banding = model.detect_banding(result, threshold=0.1)
   if banding['is_banded']:
       low_idx = banding['low_band_indices']
       high_idx = banding['high_band_indices']
       
       axes[0].axvspan(y[low_idx[0]]*1e3, y[low_idx[-1]]*1e3,
                       alpha=0.2, color='blue', label='Low-rate band')
       axes[0].axvspan(y[high_idx[0]]*1e3, y[high_idx[-1]]*1e3,
                       alpha=0.2, color='red', label='High-rate band')
       axes[0].legend()
       
       print(f"Banding detected!")
       print(f"Low band: {len(low_idx)} points")
       print(f"High band: {len(high_idx)} points")
   
   plt.tight_layout()
   plt.savefig('shear_banding_profile.png', dpi=150)

Model Comparison Table
======================

.. list-table:: TNT vs VLB Framework Comparison
   :header-rows: 1
   :widths: 20 35 35

   * - Feature
     - TNT Framework
     - VLB Framework
   * - **Formalism**
     - Discrete chain populations
     - Distribution tensor (continuum)
   * - **Mathematical Level**
     - ODEs for populations
     - Tensor PDEs
   * - **Physical Intuition**
     - Chain populations (loops, bridges)
     - End-to-end vector distribution
   * - **Force Effects**
     - Stress-dependent rates
     - Bell model, FENE extensibility
   * - **Spatial Variation**
     - Not available
     - Nonlocal PDE variant
   * - **Parameter Count**
     - 3--5 (single mode), :math:`2N+1` (multi)
     - 3-4 (basic), 5-6 (Bell+FENE)
   * - **Best For**
     - Discrete microstructures
     - Continuum mechanics, force sensitivity

.. list-table:: Model Selection by Material Type
   :header-rows: 1
   :widths: 30 35 35

   * - Material Type
     - Recommended Model
     - Key Features Needed
   * - **Wormlike Micelles**
     - TNT Cates
     - Scission/recombination, single mode
   * - **Telechelic Polymers**
     - TNT LoopBridge
     - Loop/bridge interconversion
   * - **Load-Bearing Networks**
     - VLB Variant (Bell)
     - Force-enhanced detachment
   * - **Strain-Stiffening Gels**
     - VLB Variant (FENE)
     - Finite extensibility
   * - **Shear Banding Systems**
     - VLB Nonlocal
     - Spatial coupling, PDE
   * - **Polydisperse Networks**
     - TNT MultiSpecies or VLB MultiNetwork
     - Multiple relaxation times
   * - **Ionomers**
     - TNT StickyRouse
     - Rouse + associations

Limitations and Caveats
========================

Mean-Field Approximations
-------------------------

Both TNT and VLB are **mean-field models**: they average over all chains and neglect
spatial correlations between neighboring chains. This breaks down when:

- **Chain-chain interactions** are strong (e.g., entanglements)
- **Cooperative effects** occur (avalanche dynamics)
- **Spatial heterogeneity** is important (except VLB Nonlocal)

For entangled systems, consider constitutive models like Giesekus or Rolie-Poly.

Single Relaxation Time Assumption
----------------------------------

Most variants (except MultiSpecies/MultiNetwork) assume a **single characteristic
relaxation time**. Real materials often have:

- **Broad relaxation spectra** (chain length polydispersity)
- **Multiple kinetic processes** (fast + slow associations)

Use MultiSpecies/MultiNetwork for materials with broad :math:`G'`, :math:`G''` peaks.

Shear Banding Prediction Limitations
-------------------------------------

VLB Nonlocal predicts banding from **constitutive instability** (non-monotonic flow curve),
but real shear banding involves:

- **Concentration coupling** (density variations)
- **Normal stress effects** (:math:`N_1`, :math:`N_2`)
- **Structural heterogeneity** (pre-existing defects)

Nonlocal VLB is a starting point; compare with experiments for validation.

No Entanglement Effects
------------------------

Transient network models do not include **reptation/entanglement** physics. For
entangled transient networks (e.g., entangled wormlike micelles), use:

- **TNT Cates**: Includes reptation time :math:`\tau_{\text{rep}}` explicitly
- **Hybrid approaches**: Combine transient network + tube model

Parameter Identifiability
--------------------------

Network parameters (:math:`G_0`, :math:`k_d`, :math:`k_a`) can be **correlated**, especially from SAOS data alone.

**Best practices:**

1. Use **multiple test modes** (SAOS + startup + flow curve)
2. Run **Bayesian inference** to quantify uncertainty
3. Check **pair plots** for correlations
4. Use **informative priors** if parameters known approximately

Tutorial Notebooks
==================

RheoJAX provides 36 comprehensive tutorial notebooks for TNT models (30) and VLB models (10):

TNT Framework Tutorials
------------------------

**TNT SingleMode** (6 notebooks): ``examples/tnt/01-06``

1. Flow curve fitting and prediction
2. Startup shear (stress overshoot analysis)
3. Stress relaxation
4. Creep compliance
5. SAOS (frequency sweep)
6. LAOS (nonlinear oscillatory shear)

**TNT Cates** (6 notebooks): ``examples/tnt/07-12``

Same structure as SingleMode, with emphasis on wormlike micelle physics

**TNT LoopBridge** (6 notebooks): ``examples/tnt/13-18``

Same structure, with analysis of loop/bridge populations

**TNT MultiSpecies** (6 notebooks): ``examples/tnt/19-24``

Fitting multi-modal data with :math:`N=2\text{--}4` species

**TNT StickyRouse** (6 notebooks): ``examples/tnt/25-30``

Rouse modes + sticker dynamics, multi-mode spectra

VLB Framework Tutorials
------------------------

**VLB Basic** (7 notebooks): ``examples/vlb/01-07``

1-6: Same protocol structure as TNT
7: MultiNetwork variant (polydisperse systems)

**VLB Advanced** (3 notebooks):

8. Bell model (force-enhanced detachment, shear thinning)
9. FENE model (finite extensibility, strain stiffening)
10. Nonlocal PDE (shear banding prediction)

Each notebook includes:

- **Data loading** (synthetic or experimental)
- **NLSQ fitting** (fast point estimation)
- **Bayesian inference** (uncertainty quantification)
- **Visualization** (diagnostic plots)
- **Physical interpretation** (parameter meaning)

References
==========

**TNT Framework:**

- Tanaka, F., & Edwards, S. F. (1992). "Viscoelastic properties of physically crosslinked
  networks: Transient network theory." *J. Non-Newtonian Fluid Mech.* 43(2-3), 247-271.
  https://doi.org/10.1016/0377-0257(92)80027-U

- Tanaka, F., & Edwards, S. F. (1992). "Viscoelastic properties of physically crosslinked
  networks: Part 3. Time-dependent phenomena." *J. Non-Newtonian Fluid Mech.* 43(2-3), 289-309.
  https://doi.org/10.1016/0377-0257(92)80029-W

**Wormlike Micelles (Cates Model):**

- Cates, M. E. (1987). "Reptation of living polymers: Dynamics of entangled polymers in
  the presence of reversible chain-scission reactions." *Macromolecules* 20(9), 2289-2296.
  https://doi.org/10.1021/ma00175a038

- Cates, M. E. (1990). "Nonlinear viscoelasticity of wormlike micelles (and other
  reversibly breakable polymers)." *J. Phys. Chem.* 94(1), 371-375.
  https://doi.org/10.1021/j100364a063

**Telechelic Polymers (Loop-Bridge):**

- Rubinstein, M., & Semenov, A. N. (1998). "Thermoreversible gelation in solutions of
  associating polymers. 2. Linear dynamics." *Macromolecules* 31(4), 1386-1397.
  https://doi.org/10.1021/ma970617+

**VLB Framework:**

- Vernerey, F. J., Long, R., & Brighenti, R. (2017). "A statistically-based continuum
  theory for polymers with transient networks." *J. Mech. Phys. Solids* 107, 1-20.
  https://doi.org/10.1016/j.jmps.2017.05.016

- Long, R., Mayumi, K., Creton, C., Narita, T., & Hui, C.-Y. (2014). "Time dependent
  behavior of a dual cross-link self-healing gel." *Macromolecules* 47(20), 7243-7250.
  https://doi.org/10.1021/ma501290h

**Force-Sensitive Kinetics:**

- Bell, G. I. (1978). "Models for the specific adhesion of cells to cells." *Science*
  200(4342), 618-627. https://doi.org/10.1126/science.347575

- Evans, E., & Ritchie, K. (1997). "Dynamic strength of molecular adhesion bonds."
  *Biophys. J.* 72(4), 1541-1555.
  https://doi.org/10.1016/S0006-3495(97)78802-7

**Shear Banding:**

- Olmsted, P. D. (2008). "Perspectives on shear banding in complex fluids." *Rheol. Acta*
  47(3), 283-300. https://doi.org/10.1007/s00397-008-0260-9

- Fielding, S. M. (2007). "Complex dynamics of shear banded flows." *Soft Matter* 3(10),
  1262-1279. https://doi.org/10.1039/B707980J

**Bayesian Inference for Rheology:**

- Boudara, V. A. H., Read, D. J., & Ramirez, J. (2019). "Nonlinear rheology of polydisperse blends of
  entangled linear polymers." *J. Rheol.*, 63(1), 71-91. https://doi.org/10.1122/1.5052320

See Also
========

Model Documentation
-------------------

- :doc:`/models/tnt/index` — TNT model family reference
- :doc:`/models/vlb/index` — VLB model family reference
- :doc:`/models/vlb/vlb_variant` — Bell and FENE extensions
- :doc:`/models/vlb/vlb_nonlocal` — Spatial PDE for shear banding

Related Topics
--------------

- :doc:`vitrimer_models` — HVM/HVNM (extends VLB for vitrimers)
- :doc:`constitutive_ode_models` — Giesekus, Rolie-Poly (differential constitutive models)
- :doc:`bayesian_inference` — Uncertainty quantification workflow
- :doc:`multi_technique_fitting` — Combining SAOS + startup + flow curve data

Transforms
----------

- :doc:`/transforms/srfs` — Strain-Rate Frequency Superposition (flow curve mastercurves)

Examples
--------

- :doc:`/examples/tnt/index` — 30 TNT tutorial notebooks
- :doc:`/examples/vlb/index` — 10 VLB tutorial notebooks
