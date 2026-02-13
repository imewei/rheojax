.. _ode_constitutive_models:

================================
ODE-Based Constitutive Models
================================

Overview
========

This tutorial covers constitutive models that solve systems of ordinary differential equations (ODEs)
to predict transient rheological behavior. Unlike analytical models (Maxwell, fractional), these models
use numerical ODE integration via ``diffrax`` to compute time-dependent responses in startup shear,
creep, stress relaxation, and large-amplitude oscillatory shear (LAOS) protocols.

ODE-based models capture rich physics including:

* **Stress overshoot** in startup shear (thixotropic/structural evolution)
* **Normal stress differences** (:math:`N_1`, :math:`N_2`) in shear flow
* **Yield stress behavior** with Von Mises criterion
* **Thixotropic loops** (buildup/breakdown cycles)
* **Nonlinear viscoelasticity** beyond linear response theory

RheoJAX implements three major ODE-based model families:

1. **Giesekus** (nonlinear polymer viscoelasticity)
2. **IKH/FIKH** (isotropic-kinematic hardening with optional fractional memory)
3. **Fluidity-Saramito EVP** (elastoviscoplastic fluids with thixotropic structure evolution)

.. admonition:: Key Insight

   ODE-based models excel at transient protocols where analytical solutions don't exist.
   The trade-off is computational cost: each prediction requires numerical integration,
   making fits 10-100× slower than analytical models. Use NLSQ warm-start before Bayesian inference.


When to Use ODE-Based Models
==============================

Choose ODE-based models when:

* **Transient phenomena dominate**: Stress overshoot, creep acceleration, thixotropic loops
* **Normal stress predictions needed**: Giesekus predicts :math:`N_1/N_2` from shear flow
* **Yield stress with elasticity**: Saramito EVP combines Bingham yield with viscoelastic backbone
* **Fractional memory in thixotropy**: FIKH/FMLIKH for long-time structural memory
* **Shear banding** (nonlocal variants): Spatial heterogeneity in flow

Don't use ODE models when:

* Linear viscoelasticity suffices (use Maxwell, Fractional Maxwell instead)
* Only steady-state flow curves needed (use Carreau-Yasuda, power-law)
* Fast fitting required (>1000 datasets in batch)

.. warning::

   ODE models compile slowly on first call (~30-90s for diffrax JIT).
   Pre-compile with ``model.precompile()`` if running repeated predictions.
   Each NUTS leapfrog step requires forward+backward ODE solves, limiting Bayesian sample counts.


Theoretical Foundations
========================

Constitutive Equations as ODEs
-------------------------------

Classical analytical models have closed-form stress-strain relations:

.. math::

   \sigma(t) = \int_0^t G(t-t') \dot{\gamma}(t') dt' \quad \text{(Maxwell)}

ODE models instead evolve internal state variables :math:`\mathbf{s}(t)` with differential equations:

.. math::

   \frac{d\mathbf{s}}{dt} = \mathcal{F}(\mathbf{s}, \dot{\gamma}, T, \ldots)

where :math:`\mathcal{F}` is a nonlinear operator. The stress :math:`\sigma(t)` is computed
from the current state :math:`\sigma = \mathcal{G}(\mathbf{s})`.

Three Key State Variables
--------------------------

1. **Stress/conformation tensor** :math:`\boldsymbol{\tau}` or :math:`\boldsymbol{\mu}` (Giesekus, Saramito)

   - Tracks anisotropic polymer/network deformation
   - Upper-convected derivative: :math:`\nabla \boldsymbol{\tau} = \partial_t \boldsymbol{\tau} + \mathbf{v} \cdot \nabla \boldsymbol{\tau} - (\nabla \mathbf{v})^T \cdot \boldsymbol{\tau} - \boldsymbol{\tau} \cdot \nabla \mathbf{v}`

2. **Structure parameter** :math:`\lambda \in [0,1]` (IKH, Saramito, DMT)

   - Dimensionless measure of microstructural state (0 = broken, 1 = intact)
   - Evolves via buildup/breakdown kinetics

3. **Backstress tensor** :math:`\boldsymbol{\alpha}` (kinematic hardening, IKH only)

   - Internal stress arising from structural rearrangements
   - Causes thixotropic hysteresis in loading/unloading

.. admonition:: Key Insight

   The dimensionality of the ODE system determines computational cost.
   Giesekus (5 components: 3 tensor + 2 symmetry) is faster than Saramito (4 components + fluidity).
   Multi-mode models multiply the ODE size by the number of modes.


Model Family 1: Giesekus
=========================

Physical Basis
--------------

The Giesekus model describes **nonlinear viscoelastic polymers** (solutions, melts) with:

* Anisotropic drag coefficient (mobility parameter :math:`\alpha`)
* Shear thinning via stress-dependent relaxation
* Normal stress differences (:math:`N_1 > 0`, :math:`N_2 < 0`)

Constitutive equation:

.. math::

   \boldsymbol{\tau} + \lambda \left( \nabla \boldsymbol{\tau} + \frac{\alpha}{G} \boldsymbol{\tau} \cdot \boldsymbol{\tau} \right) = 2\eta_0 \mathbf{D}

where :math:`\mathbf{D}` is the rate-of-deformation tensor, :math:`\lambda` is relaxation time,
:math:`G = \eta_0/\lambda` is modulus, and :math:`\alpha \in [0, 0.5]` controls nonlinearity.

Special case :math:`\alpha = 0`: upper-convected Maxwell (UCM) model.

Parameters and Protocols
-------------------------

**Single-mode parameters** (3):

* ``G``: Elastic modulus (Pa)
* ``lambda_1``: Relaxation time (s)
* ``alpha``: Mobility parameter (0-0.5, dimensionless)

**Multi-mode parameters** (per mode: 3):

* ``G_i``, ``lambda_i``, ``alpha_i`` for mode :math:`i=1,\ldots,N`

**Supported protocols** (6):

* FLOW_CURVE: Steady-state shear stress vs. shear rate
* OSCILLATION: SAOS (:math:`G'`, :math:`G''` vs. frequency)
* STARTUP: Transient stress growth at constant shear rate
* RELAXATION: Stress decay after step strain
* CREEP: Strain growth under constant stress
* LAOS: Large-amplitude oscillatory shear (nonlinear harmonics)

Practical Implementation
-------------------------

Basic SAOS Fitting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import GiesekusSingleMode
   import numpy as np

   # Load experimental data
   omega = np.logspace(-2, 2, 50)  # rad/s
   G_prime = ...  # Pa
   G_double_prime = ...  # Pa
   G_star = G_prime + 1j * G_double_prime

   # Fit Giesekus model
   model = GiesekusSingleMode()
   model.fit(omega, G_star, test_mode='oscillation')

   # Extract parameters
   G = model.parameters.get_value('G')
   lambda_1 = model.parameters.get_value('lambda_1')
   alpha = model.parameters.get_value('alpha')

   print(f"Modulus: {G:.1f} Pa")
   print(f"Relaxation time: {lambda_1:.3f} s")
   print(f"Mobility: {alpha:.3f}")

Startup Shear with Overshoot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict stress growth at constant shear rate
   t = np.linspace(0.01, 50, 300)  # s
   gamma_dot = 1.0  # s⁻¹

   stress = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Find overshoot
   idx_max = np.argmax(stress)
   t_max = t[idx_max]
   sigma_max = stress[idx_max]
   sigma_ss = stress[-1]

   print(f"Overshoot at t={t_max:.2f}s: {sigma_max:.1f} Pa")
   print(f"Steady-state: {sigma_ss:.1f} Pa")
   print(f"Overshoot ratio: {sigma_max/sigma_ss:.2f}")

Normal Stress Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict N₁ and N₂ in steady shear
   gamma_dot_range = np.logspace(-2, 2, 50)
   N1, N2 = model.predict_normal_stresses(gamma_dot_range)

   # First normal stress coefficient
   Psi_1 = N1 / gamma_dot_range**2

   # Check N₁/|N₂| ratio (theory: 1/(2α) - 1)
   ratio = -N1[25] / N2[25]  # Midpoint
   alpha_effective = 1 / (2 * (ratio + 1))
   print(f"N₁/|N₂| ratio: {ratio:.2f}")
   print(f"Implied alpha: {alpha_effective:.3f}")

Multi-Mode Fitting
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import GiesekusMultiMode

   # Fit with 3 relaxation modes
   model = GiesekusMultiMode(n_modes=3)
   model.fit(omega, G_star, test_mode='oscillation')

   # Extract mode parameters
   for i in range(3):
       G_i = model.parameters.get_value(f'G_{i+1}')
       lambda_i = model.parameters.get_value(f'lambda_{i+1}')
       alpha_i = model.parameters.get_value(f'alpha_{i+1}')
       print(f"Mode {i+1}: G={G_i:.1f} Pa, λ={lambda_i:.3f} s, α={alpha_i:.3f}")

.. note::

   Giesekus startup predictions require ODE integration of the conformation tensor
   :math:`\boldsymbol{\mu}`. First call compiles (~30s). Subsequent calls are fast (0.1-1s per curve).


Model Family 2: IKH/FIKH (Isotropic-Kinematic Hardening)
=========================================================

Physical Basis
--------------

IKH models describe **thixotropic fluids** with evolving microstructure:

* **Isotropic hardening**: Structure parameter :math:`\lambda(t)` modulates viscosity/modulus
* **Kinematic hardening**: Backstress tensor :math:`\boldsymbol{\alpha}(t)` tracks internal stress
* **Fractional variants** (FIKH/FMLIKH): Power-law memory via Caputo derivative

Four model variants:

1. **MIKH** (Maxwell Isotropic-Kinematic Hardening): Liquid-like, single relaxation time
2. **MLIKH** (Maxwell Liquid Isotropic-Kinematic Hardening): Extended liquid formulation
3. **FIKH** (Fractional Isotropic-Kinematic Hardening): MIKH + fractional derivative
4. **FMLIKH** (Fractional Maxwell Liquid Isotropic-Kinematic Hardening): MLIKH + fractional derivative

Structure Evolution
--------------------

.. math::

   \frac{d\lambda}{dt} = \underbrace{\frac{1 - \lambda}{t_{\text{eq}}}}_{\text{buildup}} - \underbrace{a \lambda |\dot{\gamma}|^c}_{\text{breakdown}}

* :math:`\lambda = 1`: Fully structured (rest state)
* :math:`\lambda = 0`: Fully broken (high shear)
* :math:`t_{\text{eq}}`: Equilibrium buildup time
* :math:`a, c`: Breakdown rate and exponent

Parameters and Protocols
-------------------------

**MIKH/MLIKH parameters** (5-6):

* ``G_0``: Modulus at full structure (Pa)
* ``eta_inf``: Viscosity at infinite shear (Pa·s)
* ``lambda_0``: Initial structure parameter (0-1)
* ``a``: Breakdown rate coefficient (s^c)
* ``c``: Breakdown exponent (0.5-2.0)
* ``t_eq``: Buildup time (s, MLIKH only)

**FIKH/FMLIKH parameters** (6-7):

* All MIKH/MLIKH parameters plus:
* ``alpha``: Fractional order (0-1, dimensionless)

**Supported protocols** (6 each):

* FLOW_CURVE, OSCILLATION, STARTUP, RELAXATION, CREEP, LAOS

Practical Implementation
-------------------------

Thixotropic Loop in Flow Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import MIKHLocal

   # Forward sweep: low → high shear rate
   gamma_dot_forward = np.logspace(-2, 2, 30)

   model = MIKHLocal()
   model.parameters.set_value('lambda_0', 1.0)  # Start fully structured
   stress_forward = model.predict(gamma_dot_forward, test_mode='flow_curve')

   # Reverse sweep: high → low shear rate (requires re-fitting)
   gamma_dot_reverse = gamma_dot_forward[::-1]
   model.parameters.set_value('lambda_0', 0.5)  # Start partially broken
   stress_reverse = model.predict(gamma_dot_reverse, test_mode='flow_curve')

   # Hysteresis area
   import matplotlib.pyplot as plt
   plt.loglog(gamma_dot_forward, stress_forward, 'b-', label='Up sweep')
   plt.loglog(gamma_dot_reverse, stress_reverse, 'r--', label='Down sweep')
   plt.xlabel('Shear rate (s⁻¹)')
   plt.ylabel('Stress (Pa)')
   plt.legend()

Fractional Memory Effects
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FMLIKHLocal

   # Fit FMLIKH with fractional order α
   model = FMLIKHLocal()
   model.fit(t, G_t, test_mode='relaxation')

   alpha = model.parameters.get_value('alpha')
   print(f"Fractional order: {alpha:.3f}")

   # Compare with integer-order MLIKH
   from rheojax.models import MLIKHLocal
   model_integer = MLIKHLocal()
   model_integer.fit(t, G_t, test_mode='relaxation')

   # Fractional model captures power-law relaxation better
   # α ≈ 0.3-0.7 for complex fluids

Startup Overshoot and Structure Decay
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate startup at constant shear rate
   t = np.linspace(0.01, 100, 400)
   gamma_dot = 1.0

   # Get full ODE state (stress + structure)
   result = model.simulate_startup(t, gamma_dot, return_full=True)
   stress = result['stress']
   lambda_t = result['lambda']

   # Plot stress and structure evolution
   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
   ax1.plot(t, stress)
   ax1.set_ylabel('Stress (Pa)')
   ax1.axhline(stress[-1], ls='--', color='gray', label='Steady state')

   ax2.plot(t, lambda_t)
   ax2.set_ylabel('Structure λ')
   ax2.set_xlabel('Time (s)')
   ax2.set_ylim([0, 1])

.. admonition:: Key Insight

   Fractional IKH models (FIKH/FMLIKH) add power-law memory to thixotropic evolution.
   The fractional order :math:`\alpha` quantifies non-exponential structural relaxation,
   critical for soft matter with broad timescale distributions (clays, emulsions, blood).


Model Family 3: Fluidity-Saramito EVP
======================================

Physical Basis
--------------

Fluidity-Saramito models combine:

* **Saramito tensorial viscoelasticity**: Upper-convected Maxwell (UCM) stress tensor
* **Von Mises yield criterion**: :math:`\alpha_y = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|)`
* **Thixotropic fluidity evolution**: :math:`f(t)` controls both yield stress and relaxation time

Two coupling modes:

1. **Minimal coupling**: Only relaxation time :math:`\lambda = 1/f`
2. **Full coupling**: Relaxation time + aging yield stress :math:`\tau_y(f) = \tau_{y,0} + k_{\text{age}} (1 - f)`

Structure Evolution (Fluidity)
-------------------------------

.. math::

   \frac{df}{dt} = -\underbrace{\frac{f - f_{\text{eq}}}{t_{\text{th}}}}_{\text{aging}} + \underbrace{b |\dot{\gamma}|^n}_{\text{rejuvenation}}

* :math:`f = 1`: High fluidity (flowing)
* :math:`f = 0`: Low fluidity (jammed/aging)
* :math:`t_{\text{th}}`: Thixotropic timescale
* :math:`b, n`: Rejuvenation rate and exponent

Parameters and Protocols
-------------------------

**Local model parameters** (7-8):

* ``G``: Elastic modulus (Pa)
* ``k_d_0``: Baseline relaxation rate (s⁻¹)
* ``tau_y``: Baseline yield stress (Pa)
* ``f_eq``: Equilibrium fluidity (0-1)
* ``t_th``: Thixotropic time (s)
* ``b``: Rejuvenation rate (s^(1-n))
* ``n``: Rejuvenation exponent (0.5-2.0)
* ``k_age``: Aging yield increment (Pa, full coupling only)

**Nonlocal model** adds:

* ``n_points``: Spatial discretization (51-201)
* ``gap_width``: Gap size (m)
* ``D_f``: Fluidity diffusion coefficient (m²/s)

**Supported protocols** (6):

* FLOW_CURVE, STARTUP, CREEP, RELAXATION, OSCILLATION, LAOS

Practical Implementation
-------------------------

Yield Stress with Elasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.fluidity.saramito import FluiditySaramitoLocal

   # Minimal coupling (λ = 1/f only)
   model = FluiditySaramitoLocal(coupling="minimal")

   # Fit to flow curve
   gamma_dot = np.logspace(-3, 2, 50)
   sigma = ...  # Experimental stress (Pa)

   model.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Extract yield stress
   tau_y = model.parameters.get_value('tau_y')
   print(f"Yield stress: {tau_y:.1f} Pa")

   # Verify Herschel-Bulkley behavior at low shear rates
   # σ = τ_y + K·γ̇^n

Creep with Viscosity Bifurcation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Apply constant stress above/below yield
   t = np.linspace(0, 500, 1000)

   # Below yield: creep compliance saturates
   sigma_low = 0.8 * tau_y
   gamma_low, f_low = model.simulate_creep(t, sigma_low)

   # Above yield: unbounded creep (viscous flow)
   sigma_high = 1.2 * tau_y
   gamma_high, f_high = model.simulate_creep(t, sigma_high)

   # Plot bifurcation
   plt.figure()
   plt.plot(t, gamma_low, label=f'σ = {sigma_low:.1f} Pa (< τ_y)')
   plt.plot(t, gamma_high, label=f'σ = {sigma_high:.1f} Pa (> τ_y)')
   plt.xlabel('Time (s)')
   plt.ylabel('Strain')
   plt.legend()

Normal Stress from UCM Tensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Saramito uses tensorial stress → N₁ prediction
   gamma_dot_range = np.logspace(-2, 2, 50)
   N1, N2 = model.predict_normal_stresses(gamma_dot_range)

   # UCM limit: N₁ = 2λG·γ̇², N₂ = 0
   # (Saramito modifies λ → 1/f(γ̇))

   print(f"First normal stress coefficient: {N1[25]/gamma_dot_range[25]**2:.1f} Pa·s²")

Shear Banding with Nonlocal Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.fluidity.saramito import FluiditySaramitoNonlocal

   # Nonlocal model with structure diffusion
   model_nl = FluiditySaramitoNonlocal(coupling="full", n_points=51)
   model_nl.parameters.set_value('D_f', 1e-9)  # m²/s
   model_nl.parameters.set_value('gap_width', 1e-3)  # 1 mm gap

   # Predict steady-state profile
   gamma_dot_avg = 10.0
   result = model_nl.simulate_steady_shear(gamma_dot_avg, t_end=500)

   # Extract spatial profiles
   y_positions = result['y']
   gamma_dot_profile = result['gamma_dot']
   fluidity_profile = result['fluidity']

   # Detect banding (high/low shear regions)
   banding = model_nl.detect_banding(result, threshold=0.1)
   if banding['has_banding']:
       print(f"Band interface at y = {banding['interface_position']:.4f} mm")
       print(f"Shear rate ratio: {banding['shear_ratio']:.1f}")

.. warning::

   Nonlocal models solve PDEs on spatial grids (51-201 points).
   Memory scales as O(n_points × n_timesteps). Use ``n_points=51`` for prototyping,
   increase to 101-201 only for publication-quality profiles.


Bayesian Inference for ODE Models
==================================

Workflow: NLSQ Warm-Start → NUTS
---------------------------------

ODE models are computationally expensive for MCMC. Always use NLSQ to initialize:

.. code-block:: python

   from rheojax.models import GiesekusSingleMode

   # Step 1: NLSQ point estimate (fast)
   model = GiesekusSingleMode()
   model.fit(omega, G_star, test_mode='oscillation', max_iter=1000)

   # Step 2: Bayesian inference with warm-start
   result = model.fit_bayesian(
       omega, G_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,  # Parallel chains for diagnostics
       seed=42        # Reproducibility
   )

   # Step 3: Check convergence
   import arviz as az
   rhat = az.rhat(result.posterior_samples)
   ess = az.ess(result.posterior_samples)

   print(f"R-hat (should be <1.01): {rhat.max():.3f}")
   print(f"ESS (should be >400): {ess.min():.0f}")

Memory Considerations
---------------------

.. code-block:: python

   # For ODE models, reduce sample counts to avoid OOM

   # Standard protocols (SAOS, flow curves): 1000 warmup + 2000 samples OK
   result = model.fit_bayesian(omega, G_star, num_warmup=1000, num_samples=2000)

   # Transient protocols (startup, LAOS): reduce to 25-50 samples in FAST_MODE
   import os
   if os.environ.get("FAST_MODE", "0") == "1":
       num_warmup, num_samples = 25, 50
   else:
       num_warmup, num_samples = 500, 1000

   result = model.fit_bayesian(t, stress, test_mode='startup',
                               num_warmup=num_warmup, num_samples=num_samples)

Prior Selection
---------------

.. code-block:: python

   # Tighten priors for ODE models (improves convergence)
   from rheojax.core.parameter import Prior

   model = MIKHLocal()

   # Structure parameter: informative beta prior
   model.parameters['lambda_0'].prior = Prior('beta', alpha=2.0, beta=2.0)

   # Breakdown exponent: narrow around physical values
   model.parameters['c'].prior = Prior('truncated_normal', loc=1.0, scale=0.2)

   # Log-normal for positive-only parameters (G, eta_inf)
   model.parameters['G_0'].prior = Prior('lognormal', loc=np.log(1000), scale=1.0)

.. note::

   ODE models with >8 parameters may require tighter priors or reparameterization.
   Consider fixing less-sensitive parameters (e.g., ``c``, ``n``) based on literature values.


Visualization and Diagnostics
==============================

Stress Overshoot Visualization
-------------------------------

.. code-block:: python

   from rheojax.visualization import plot_startup_overshoot

   # Automatic detection of peak stress
   fig, axes = plot_startup_overshoot(
       model,
       gamma_dot_values=[0.1, 1.0, 10.0],
       t_max=50,
       mark_overshoot=True
   )
   plt.savefig('giesekus_startup.png', dpi=300)

Normal Stress Comparison
-------------------------

.. code-block:: python

   from rheojax.visualization import plot_normal_stresses

   fig, ax = plot_normal_stresses(
       model,
       gamma_dot_range=np.logspace(-2, 2, 50),
       plot_ratio=True  # Add N₁/|N₂| ratio
   )

Thixotropic Loop
----------------

.. code-block:: python

   # Custom visualization for hysteresis
   fig, ax = plt.subplots()

   # Forward sweep
   ax.loglog(gamma_dot_forward, stress_forward, 'b-o',
            label='Increasing $\dot{\gamma}$', markersize=4)

   # Reverse sweep
   ax.loglog(gamma_dot_reverse, stress_reverse, 'r--s',
            label='Decreasing $\dot{\gamma}$', markersize=4)

   # Shade hysteresis region
   ax.fill_between(gamma_dot_forward, stress_forward, stress_reverse,
                   alpha=0.2, color='gray')

   ax.set_xlabel('Shear rate $\dot{\gamma}$ (s$^{-1}$)')
   ax.set_ylabel('Stress $\sigma$ (Pa)')
   ax.legend()
   ax.grid(alpha=0.3)

Bayesian Posterior Plots
-------------------------

.. code-block:: python

   # Pair plot for parameter correlations
   from rheojax.pipeline.bayesian import BayesianPipeline

   pipeline = BayesianPipeline()
   (pipeline.load_data(omega, G_star, test_mode='oscillation')
            .fit_nlsq('giesekus_single')
            .fit_bayesian(num_samples=2000)
            .plot_pair(divergences=True)  # Mark divergent samples
            .plot_forest(hdi_prob=0.95)   # Credible intervals
            .plot_trace()                  # MCMC chains
            .save_results('giesekus_bayes.hdf5'))


Model Selection and Comparison
===============================

Comparison Table
----------------

.. list-table:: ODE-Based Model Selection Guide
   :header-rows: 1
   :widths: 15 20 20 15 15 15

   * - Model
     - Best For
     - Key Feature
     - Parameters
     - ODE Size
     - Fit Time
   * - Giesekus (single)
     - Polymer solutions
     - Shear thinning + :math:`N_1`
     - 3
     - 5
     - ~30s (startup)
   * - Giesekus (multi)
     - Polydisperse polymers
     - Multi-mode relaxation
     - 3N
     - 5N
     - ~60s (startup)
   * - MIKH
     - Thixotropic fluids
     - Structure buildup/breakdown
     - 5
     - 6
     - ~40s (startup)
   * - MLIKH
     - Complex thixotropy
     - Extended liquid formulation
     - 6
     - 6
     - ~40s (startup)
   * - FIKH
     - Fractional thixotropy
     - Power-law memory
     - 6
     - 6
     - ~50s (startup)
   * - FMLIKH
     - Advanced thixotropy
     - Fractional + liquid
     - 7
     - 6
     - ~50s (startup)
   * - Saramito (local)
     - EVP fluids
     - Yield + thixotropy
     - 7-8
     - 4
     - ~35s (startup)
   * - Saramito (nonlocal)
     - Banding EVP
     - Spatial heterogeneity
     - 8-9
     - 4×N_pts
     - ~300s (startup)

Performance Benchmarks
----------------------

Typical fit times on CPU (Apple M1, 8 cores):

.. code-block:: python

   # NLSQ optimization (100 data points)
   # Giesekus SAOS: 2-5s
   # Giesekus startup: 20-30s
   # IKH SAOS: 3-6s
   # IKH startup: 30-40s
   # Saramito SAOS: 2-4s
   # Saramito startup: 25-35s

   # Bayesian NUTS (1000 warmup + 2000 samples, 4 chains)
   # Giesekus SAOS: 3-5 min
   # Giesekus startup: 15-25 min
   # IKH SAOS: 4-6 min
   # IKH startup: 20-30 min
   # Saramito SAOS: 3-5 min
   # Saramito startup: 18-28 min

.. admonition:: Key Insight

   Startup/LAOS protocols are 5-10× slower than SAOS due to fine temporal resolution
   required for stress overshoot capture. Use adaptive timestepping (diffrax default)
   and compile models before batch processing.


Limitations and Pitfalls
=========================

Numerical Stability
-------------------

* **Stiff ODEs at high shear rates**: Use implicit solvers (Kvaerno3/5) or reduce ``rtol``/``atol``
* **Tensor symmetry breaking**: Giesekus/Saramito enforce :math:`\tau_{xy} = \tau_{yx}` but errors accumulate
* **Negative moduli**: Structure parameter :math:`\lambda \to 0` can cause :math:`G(\lambda) < 0` → add lower bound

Computational Cost
------------------

* **First call compilation**: 30-90s for diffrax JIT → use ``model.precompile()``
* **Bayesian OOM**: LAOS with 1000 cycles × 4 chains × 2000 samples exceeds 16GB RAM → reduce ``num_samples``
* **Batch processing**: ODE models don't vectorize well → use ``joblib`` parallelism over datasets

Physical Validity
-----------------

* **Mobility bounds**: Giesekus :math:`\alpha > 0.5` violates thermodynamics → enforce ``alpha.bounds = (0, 0.5)``
* **Fractional order**: FIKH/FMLIKH with :math:`\alpha > 0.9` behaves like integer model → check if fractional is needed
* **Yield stress artifacts**: Von Mises criterion at :math:`|\tau| \approx \tau_y` causes stress discontinuities

.. warning::

   ODE models can fit noise if over-parameterized. Always compare to simpler analytical models
   (Maxwell, Carreau-Yasuda) and verify that overshoot/thixotropy are genuine physical effects,
   not measurement artifacts.


Tutorial Notebooks
==================

Each model family has 6-12 tutorial notebooks in ``examples/``:

Giesekus Notebooks
------------------

* ``examples/giesekus/01_giesekus_flow_curve.ipynb``: Steady shear viscosity
* ``examples/giesekus/02_giesekus_saos.ipynb``: Linear viscoelasticity (:math:`G'`, :math:`G''`)
* ``examples/giesekus/03_giesekus_startup.ipynb``: Stress overshoot prediction
* ``examples/giesekus/04_giesekus_normal_stresses.ipynb``: :math:`N_1`, :math:`N_2` in shear flow
* ``examples/giesekus/05_giesekus_creep.ipynb``: Creep compliance
* ``examples/giesekus/06_giesekus_relaxation.ipynb``: Stress relaxation
* ``examples/giesekus/07_giesekus_laos.ipynb``: Nonlinear oscillatory response

IKH/FIKH Notebooks
------------------

* ``examples/ikh/01-06_mikh_*.ipynb``: MIKH model (6 protocols)
* ``examples/ikh/07-12_mlikh_*.ipynb``: MLIKH model (6 protocols)
* ``examples/fikh/01-06_fikh_*.ipynb``: FIKH fractional model (6 protocols)
* ``examples/fikh/07-12_fmlikh_*.ipynb``: FMLIKH fractional model (6 protocols)

Fluidity-Saramito Notebooks
----------------------------

* ``examples/fluidity/13-18_saramito_local_*.ipynb``: Local EVP (6 protocols)
* ``examples/fluidity/19-24_saramito_nonlocal_*.ipynb``: Nonlocal with banding (6 protocols)

.. note::

   All notebooks include FAST_MODE flag for quick testing (reduced samples/points).
   Set ``FAST_MODE=0`` for publication-quality fits with full Bayesian diagnostics.


References
==========

Giesekus Model
--------------

* Giesekus, H. (1982). "A simple constitutive equation for polymer fluids based on the concept of deformation-dependent tensorial mobility." *Journal of Non-Newtonian Fluid Mechanics*, 11(1-2), 69-109. https://doi.org/10.1016/0377-0257(82)85016-7
* Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics* (2nd ed.). Wiley. ISBN: 978-0-471-80245-7

IKH Models
----------

* Mujumdar, A., Beris, A. N., & Metzner, A. B. (2002). "Transient phenomena in thixotropic systems." *Journal of Non-Newtonian Fluid Mechanics*, 102(2), 157-178. https://doi.org/10.1016/S0377-0257(01)00176-8
* de Souza Mendes, P. R., & Thompson, R. L. (2013). "A unified approach to model elasto-viscoplastic thixotropic yield-stress materials and apparent yield-stress fluids." *Rheologica Acta*, 52(7), 673-694. https://doi.org/10.1007/s00397-013-0699-1

Fractional Extensions
---------------------

* Bagley, R. L., & Torvik, P. J. (1983). "A theoretical basis for the application of fractional calculus to viscoelasticity." *Journal of Rheology*, 27(3), 201-210. https://doi.org/10.1122/1.549724
* Palade, L. I., Verney, V., & Attané, P. (1996). "A modified fractional model to describe the entire viscoelastic behavior of polybutadienes from flow to glassy regime." *Rheologica Acta*, 35(3), 265-273. https://doi.org/10.1007/BF00366913

Saramito EVP Model
------------------

* Saramito, P. (2007). "A new constitutive equation for elastoviscoplastic fluid flows." *Journal of Non-Newtonian Fluid Mechanics*, 145(1), 1-14. https://doi.org/10.1016/j.jnnfm.2007.04.004
* Saramito, P. (2009). "A new elastoviscoplastic model based on the Herschel–Bulkley viscoplastic model." *Journal of Non-Newtonian Fluid Mechanics*, 158(1-3), 154-161. https://doi.org/10.1016/j.jnnfm.2008.12.001

Fluidity Approach
-----------------

* Derec, C., Ducouret, G., Ajdari, A., & Lequeux, F. (2003). "Aging and nonlinear rheology in suspensions of polyethylene oxide–protected silica particles." *Physical Review E*, 67(6), 061403. https://doi.org/10.1103/PhysRevE.67.061403
* Moorcroft, R. L., & Fielding, S. M. (2013). "Criteria for shear banding in time-dependent flows of complex fluids." *Physical Review Letters*, 110(8), 086001. https://doi.org/10.1103/PhysRevLett.110.086001


See Also
========

Related tutorials in this documentation:

* :doc:`/user_guide/03_advanced_topics/thixotropy_yielding` — Complementary thixotropy guide
* :doc:`/user_guide/02_core_concepts/bayesian_inference` — Bayesian workflow details
* :doc:`/user_guide/03_advanced_topics/sgr_analysis` — Alternative statistical mechanics approach

Model handbooks:

* :doc:`/models/giesekus/index` — Giesekus comprehensive reference
* :doc:`/models/ikh/index` — IKH model family
* :doc:`/models/fikh/fmlikh` — Fractional IKH variants
* :doc:`/models/fluidity/saramito_evp` — Saramito EVP details
* :doc:`/models/fluidity/index` — Fluidity framework overview

Performance optimization:

* :doc:`/user_guide/03_advanced_topics/performance_optimization` — JAX compilation, batching
* :doc:`/user_guide/04_reference/api_core` — Core API reference
