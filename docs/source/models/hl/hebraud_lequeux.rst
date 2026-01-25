.. _hebraud_lequeux:

Hébraud–Lequeux (HL) Model
==========================

Quick Reference
---------------

**Use when:** Mean-field modeling of soft glassy materials, yield-stress fluids, foams, emulsions, pastes

**Parameters:** 4 (G₀, α, σ_c, τ)

**Key equation:** :math:`\partial_t P(\sigma, t) = -\dot{\gamma}(t) \partial_\sigma P + D(t) \partial^2_\sigma P - \frac{1}{\tau} \Theta(|\sigma|-\sigma_c) P + \Gamma(t) \delta(\sigma)`

**Test modes:** flow_curve (steady_shear), creep, relaxation, oscillation (LAOS)

**Material examples:** Foams, emulsions, pastes, concentrated colloidal suspensions, soft glassy materials

Overview
--------

The **Hébraud–Lequeux (HL) model** is a seminal mean-field elastoplastic model for soft glassy materials (SGMs), introduced by Hébraud and Lequeux in 1998. It describes the rheology of yield-stress fluids, foams, emulsions, and pastes by considering the statistical evolution of local stresses.

.. note::
   This implementation uses high-performance JAX kernels with a Finite Volume Method (FVM) solver, enabling efficient fitting to flow curves, creep, relaxation, and LAOS data.

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - P(σ, t)
     - 1/Pa
     - Probability density function of local stresses
   * - σ
     - Pa
     - Local shear stress on mesoscopic block
   * - σ_c
     - Pa
     - Local yield stress threshold
   * - γ̇
     - 1/s
     - Macroscopic applied shear rate
   * - G₀
     - Pa
     - Elastic shear modulus
   * - τ
     - s
     - Plastic relaxation time
   * - D(t)
     - Pa²/s
     - Mechanical noise (stress diffusivity)
   * - Γ(t)
     - 1/s
     - Plastic activity rate (total yielding rate)
   * - α
     - —
     - Noise coupling parameter (control parameter)

Physical Basis
--------------

The model considers a material composed of mesoscopic elastoplastic blocks. Each block carries a local shear stress :math:`\sigma` which evolves through three processes:

1.  **Elastic Loading**: Under macroscopic shear rate :math:`\dot{\gamma}`, blocks accumulate stress elastically (:math:`\dot{\sigma} = G_0 \dot{\gamma}`).
2.  **Plastic Yielding**: If the local stress exceeds a threshold :math:`\sigma_c`, the block yields (relaxes stress to zero) at a rate :math:`1/\tau`.
3.  **Mechanical Noise**: Yielding events redistribute stress to neighbors. In a mean-field approximation, this is modeled as a stress diffusion process with diffusivity :math:`D(t)`.

Key Distinction from SGR/STZ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **SGR (Soft Glassy Rheology)**: Disorder is in the *energy barrier heights* (trap depths). Noise is thermal-like effective temperature :math:`x`.
*   **STZ (Shear Transformation Zone)**: Dynamics are controlled by creation/annihilation of *structural defects* (STZs).
*   **HL (Hébraud–Lequeux)**: Disorder is in the *local stress state*. Noise is self-generated *mechanical diffusion*.

Physical Foundations
--------------------

Mesoscopic Block Picture
~~~~~~~~~~~~~~~~~~~~~~~~~

The HL model discretizes the material into **mesoscopic blocks** of size ξ (correlation length of plastic events, typically 10-100 particle diameters). Each block is characterized by:

- **Local stress σ**: Elastic stress stored in the block
- **Yield threshold σ_c**: Critical stress for plastic relaxation (assumed uniform across blocks)
- **Elastic response**: σ increases affinely with applied strain (σ̇ = G₀·γ̇)

Unlike EPM (which tracks spatial positions), HL is a **mean-field model**: we track only the **probability distribution** P(σ, t) of local stresses, not their spatial arrangement.

Mechanical Noise and Self-Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key innovation of HL is **mechanical noise coupling**: yielding events at one location create stress fluctuations that affect neighbors. In a mean-field approximation, this is modeled as:

**Stress diffusion**: D(t)·∂²P/∂σ²

The diffusivity D(t) is **self-consistently** determined by the plastic activity:

.. math::

   D(t) = \alpha \Gamma(t)

where:

- Γ(t) = plastic activity rate (fraction of blocks yielding per unit time)
- α = noise coupling parameter (material-dependent constant)

**Physical interpretation**:

- Each yielding event redistributes stress to neighbors (via Eshelby-like propagator)
- In mean-field, this appears as **diffusion** in stress space
- The stronger the plastic activity Γ, the stronger the noise D

This self-consistency creates a **feedback loop**:

1. More stress → More yielding (Γ increases)
2. More yielding → More noise (D increases)
3. More noise → Stress spreads out (more blocks approach σ_c)
4. More blocks at σ_c → More yielding (loop closes)

Glass Transition via Effective Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter α controls the **phase** of the material:

**α < 0.5 (Glass phase)**:

- Noise insufficient to maintain equilibrium
- System "freezes" into metastable states
- **Yield stress** σ_y > 0 emerges
- **Aging**: Relaxation timescales grow with waiting time

**α ≥ 0.5 (Fluid phase)**:

- Noise sufficient for ergodic exploration of stress space
- System reaches steady state P_ss(σ)
- **No yield stress** (flows at any stress)
- **Finite viscosity**: η ~ 1/(α - 0.5) diverges as α → 0.5⁺

The **glass transition** at α = 0.5 is a **critical point** analogous to SGR's x = 1 transition, but with different physical origin (mechanical noise vs. thermal-like effective temperature).

Governing Equations
------------------------

The probability density function :math:`P(\sigma, t)` of local stresses evolves according to the Fokker-Planck equation:

.. math::

   \partial_t P(\sigma, t) = \underbrace{-\dot{\gamma}(t) \partial_\sigma P}_{\text{Advection}} + \underbrace{D(t) \partial^2_\sigma P}_{\text{Diffusion}} - \underbrace{\frac{1}{\tau} \Theta(|\sigma|-\sigma_c) P}_{\text{Yielding}} + \underbrace{\Gamma(t) \delta(\sigma)}_{\text{Reinjection}}

Self-Consistency
~~~~~~~~~~~~~~~~

The model is closed by coupling the noise strength :math:`D(t)` to the plastic activity rate :math:`\Gamma(t)`:

.. math::

   \Gamma(t) = \frac{1}{\tau} \int_{|\sigma| > \sigma_c} P(\sigma, t) \, d\sigma

   D(t) = \alpha \Gamma(t)

where :math:`\alpha` is the dimensionless coupling parameter.

Phase Behavior
--------------

The parameter :math:`\alpha` controls the phase state of the material:

*   **Glassy Phase** (:math:`\alpha < 0.5`): The material exhibits a finite yield stress :math:`\sigma_y`. Below this stress, the material is solid-like (creep arrest).
*   **Fluid Phase** (:math:`\alpha \ge 0.5`): The material flows at any non-zero stress.

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
     - Elastic shear modulus; relates stress to strain
   * - ``alpha``
     - :math:`\alpha`
     - —
     - Noise coupling parameter; controls glass (< 0.5) vs fluid (≥ 0.5) phase
   * - ``sigma_c``
     - :math:`\sigma_c`
     - Pa
     - Local yield stress threshold for plastic relaxation
   * - ``tau``
     - :math:`\tau`
     - s
     - Plastic relaxation time; rate of local yielding

Predictions and Protocols
-------------------------

Flow Curve (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

In steady shear (:math:`\dot{\gamma} = \text{const}`), the HL model predicts a **Herschel-Bulkley** behavior near yield:

.. math::

   \Sigma(\dot{\gamma}) \approx \Sigma_y + A \dot{\gamma}^{1/2}

The exponent 0.5 is a universal prediction of the HL model for the glassy phase.

Creep (Step Stress)
~~~~~~~~~~~~~~~~~~~

Under constant stress :math:`\Sigma_0`:
*   If :math:`\Sigma_0 < \Sigma_y`: The shear rate decays to zero (arrest).
*   If :math:`\Sigma_0 > \Sigma_y`: The shear rate reaches a steady finite value.
*   The model captures **delayed yielding** (creep rupture) dynamics near the yield point.

Stress Relaxation (Step Strain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following a step strain :math:`\gamma_0`, the stress relaxes via:
1.  Fast relaxation (yielding of highly stressed elements).
2.  Slow relaxation (diffusion).
For :math:`\alpha < 0.5`, residual stress may persist (finite elastic modulus).

LAOS (Large Amplitude Oscillatory Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In LAOS, the PDF :math:`P(\sigma, t)` oscillates. Large amplitudes drive the distribution past :math:`\sigma_c`, causing periodic fluidization and non-linear Lissajous figures.

Validity and Assumptions
------------------------

**Valid for:**

- **Mean-field systems**: Where spatial correlations are weak or averaged out
- **Soft glassy materials**: Foams, emulsions, pastes, colloidal suspensions
- **Athermal plasticity**: Yielding driven by stress, not thermal activation
- **Homogeneous systems**: No shear banding or spatial localization

**Assumptions:**

- **Uniform yield threshold**: All blocks have same σ_c (no disorder in thresholds)
- **Mechanical noise only**: Neglects thermal fluctuations (k_B T ≪ σ_c·ξ³)
- **Mean-field approximation**: Stress redistribution averaged (no spatial propagator)
- **Overdamped dynamics**: Inertia negligible

**Not appropriate for:**

- Systems with **spatial heterogeneity** (use EPM instead)
- **Thixotropic materials** with evolving structure (use DMT or MIKH)
- **Viscoelastic fluids** with polymer relaxation (use Maxwell-based models)
- **Crystalline solids** with dislocation dynamics

What You Can Learn
------------------

From fitting HL model to experimental data, you can extract insights about glass transitions, mechanical noise, and cooperative yielding in soft glassy materials.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**α (Noise Coupling Parameter)**:
   Dimensionless parameter controlling glass (α < 0.5) vs fluid (α ≥ 0.5) phase behavior.
   *For graduate students*: α = D/Γ where D is stress diffusivity and Γ is plastic activity rate. At critical point α = 0.5, the system undergoes a dynamic phase transition analogous to SGR's effective temperature x = 1. Controls divergence of viscosity: η ~ 1/(α - 0.5) as α → 0.5⁺.
   *For practitioners*: α < 0.5 predicts yield stress σ_y ≈ σ_c(1 - 2α). Use α to classify materials: α = 0.2-0.4 typical for pastes/gels, α > 0.5 for liquid-like emulsions.

**σ_c (Local Yield Threshold)**:
   Critical stress for local plastic relaxation at mesoscopic block level.
   *For graduate students*: Sets energy scale for cage breaking or bond rupture. Not the macroscopic yield stress (which is σ_y ≈ σ_c(1-2α) for α < 0.5). Relates to microscopic interaction potentials.
   *For practitioners*: Estimate from small-amplitude oscillatory stress sweeps or extrapolate from flow curves. σ_c/G ~ 0.1-0.5 for typical soft glasses.

**τ (Plastic Relaxation Time)**:
   Timescale for stress relaxation in yielded blocks.
   *For graduate students*: Microscopic hopping time or cage escape time. Sets fastest relaxation mode in system. For thermally-activated processes, τ ~ τ₀exp(ΔE/k_BT).
   *For practitioners*: Determines transient response timescale. For τ = 0.01-1 s, expect stress overshoot duration ~ 10τ in startup flows.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Hébraud-Lequeux Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - α < 0.3
     - Strong glass (aging)
     - Concentrated colloidal gels, dense granular media
     - High yield stress, pronounced aging
   * - α = 0.3-0.5
     - Weak glass (near transition)
     - Foams, pastes, soft colloids
     - Moderate yield stress, thixotropy
   * - α > 0.5
     - Fluid (no yield stress)
     - Dilute emulsions, weak gels
     - Continuous flow at any stress
   * - Flow exponent n ≈ 0.5
     - HL universal signature
     - All α < 0.5 systems
     - Herschel-Bulkley with n=0.5
   * - Γ_ss > 10 s⁻¹
     - High plastic activity
     - Rapidly flowing soft glasses
     - Significant energy dissipation

Implementation Details
----------------------

The RheoJAX implementation uses an explicit Finite Volume Method (FVM) solver written in JAX.

*   **Advection**: First-order upwind scheme (stable for hyperbolic transport).
*   **Diffusion**: Central difference scheme.
*   **Time-stepping**: Operator splitting with Forward Euler.
*   **JIT Compilation**: The entire solver is JIT-compiled for GPU acceleration, typically running :math:`100\times` faster than pure Python implementations.

Numerical Parameters
~~~~~~~~~~~~~~~~~~~~

The solver uses a discretized stress grid:
*   `sigma_max`: Default 5.0 (normalized units).
*   `n_bins`: Default 501.
*   `dt`: Default 0.005.

These can be adjusted if necessary, but defaults work for most experimental ranges.

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Determine phase (α < 0.5 or α ≥ 0.5)**

From experimental observations:

- **If yield stress observed**: Start with α = 0.3 (glass phase)
- **If no yield stress**: Start with α = 0.7 (fluid phase)

**Step 2: Fit flow curve (if available)**

Use steady-state flow curve data to constrain:

- ``sigma_c``: Should approximate yield stress σ_y (for α < 0.5) or typical stress scale
- ``alpha``: Refine to match flow curve shape (especially near yield)

**Step 3: Fit transient data (startup, relaxation, creep)**

Use time-dependent data to refine:

- ``tau``: Controls relaxation timescale
- ``G0``: Match initial elastic modulus (if available from SAOS)

Parameter Bounds and Physical Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Physical Constraint
   * - ``G0``
     - 10-10000 Pa
     - Match SAOS plateau modulus if available
   * - ``alpha``
     - 0.1-1.5
     - α < 0.5 (glass), α ≥ 0.5 (fluid); critical at α = 0.5
   * - ``sigma_c``
     - 0.5-2× σ_y
     - For glass phase; for fluids, σ_c is typical stress scale
   * - ``tau``
     - 0.01-100 s
     - Should match fastest relaxation timescale in data

Common Fitting Issues
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Issue
     - Solution
   * - Flow curve too steep/shallow near yield
     - Adjust ``alpha`` (lower α = steeper); check if α < 0.5
   * - No yield stress predicted but observed
     - Decrease ``alpha`` below 0.5; check σ_c initialization
   * - Relaxation too fast/slow
     - Adjust ``tau`` (plastic relaxation time)
   * - Creep predicts arrest but sample flows
     - Increase ``alpha`` above 0.5 (fluid phase)
   * - Numerical instability (PDF becomes negative)
     - Reduce ``dt`` or increase ``n_bins`` (finer discretization)

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   # Initialize model
   model = HebraudLequeux()

   # Fit to flow curve data
   gamma_dot = np.logspace(-2, 1, 20)
   sigma_data = np.array([...])  # Experimental stress data
   model.fit(gamma_dot, sigma_data, test_mode='flow_curve')

   # Predict at new shear rates
   gamma_dot_pred = np.logspace(-3, 2, 50)
   sigma_pred = model.predict(gamma_dot_pred, test_mode='flow_curve')

Glassy State Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   # Initialize model with glassy parameters
   model = HebraudLequeux()
   model.parameters.set_value("alpha", 0.3)     # Glassy phase (< 0.5)
   model.parameters.set_value("sigma_c", 10.0)  # Pa
   model.parameters.set_value("G0", 100.0)      # Pa
   model.parameters.set_value("tau", 0.1)       # s

   # Fit Flow Curve
   gdot = np.logspace(-2, 1, 20)
   # Load experimental data: stress_data = load_data(...)
   stress_data = 10.0 + 5.0 * gdot**0.5  # Example: Herschel-Bulkley
   model.fit(gdot, stress_data, test_mode="flow_curve")

   # Predict Creep
   time = np.linspace(0, 100, 1000)
   gamma_creep = model.predict(time, test_mode="creep", sigma_applied=12.0)

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   model = HebraudLequeux()

   # Generate or load data
   gdot = np.logspace(-2, 1, 20)
   stress_data = np.array([...])  # Experimental data

   # Bayesian inference with NUTS
   result = model.fit_bayesian(
       gdot, stress_data,
       test_mode="flow_curve",
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"alpha: {intervals['alpha']}")
   print(f"sigma_c: {intervals['sigma_c']}")

Advanced Usage Example
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np
   import matplotlib.pyplot as plt

   # Initialize model
   model = HebraudLequeux()

   # Set to glassy state
   model.parameters.set_value("alpha", 0.3)
   model.parameters.set_value("sigma_c", 10.0)  # Pa
   model.parameters.set_value("tau", 0.1)       # s

   # 1. Fit Flow Curve
   gdot = np.logspace(-2, 1, 20)
   stress_data = 10.0 + 5.0 * gdot**0.5  # Example data
   model.fit(gdot, stress_data, test_mode="flow_curve")

   # 2. Predict Creep
   time = np.linspace(0, 100, 1000)
   J_pred = model.predict(time, test_mode="creep", sigma_applied=12.0)

   # Plot results
   plt.figure(figsize=(10, 6))
   plt.subplot(1, 2, 1)
   plt.loglog(gdot, stress_data, 'o', label='Data')
   plt.loglog(gdot, model.predict(gdot, test_mode='flow_curve'), '-', label='Fit')
   plt.xlabel('Shear rate (1/s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(time, J_pred)
   plt.xlabel('Time (s)')
   plt.ylabel('Compliance J(t) (1/Pa)')
   plt.tight_layout()
   plt.show()

References
----------

.. [1] Hébraud, P. and Lequeux, F. "Mode-coupling theory for the pasty rheology of soft
   glassy materials." *Physical Review Letters*, 81(14), 2934 (1998).
   https://doi.org/10.1103/PhysRevLett.81.2934

.. [2] Fielding, S. M., Sollich, P., and Cates, M. E. "Aging and rheology in soft
   materials." *Journal of Rheology*, 44(2), 323-369 (2000).
   https://doi.org/10.1122/1.551088

.. [3] Sollich, P., Lequeux, F., Hébraud, P., and Cates, M. E. "Rheology of soft glassy
   materials." *Physical Review Letters*, 78, 2020 (1997).
   https://doi.org/10.1103/PhysRevLett.78.2020

.. [4] Coussot, P., Nguyen, Q. D., Huynh, H. T., and Bonn, D. "Viscosity bifurcation in
   thixotropic, yielding fluids." *Journal of Rheology*, 46, 573-589 (2002).
   https://doi.org/10.1122/1.1459447

.. [5] Bocquet, L., Colin, A., and Ajdari, A. "Kinetic theory of plastic flow in soft
   glassy materials." *Physical Review Letters*, 103, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

.. [6] Fielding, S. M., Sollich, P., & Cates, M. E. "Aging and rheology in soft materials."
   *Journal of Rheology*, **44**, 323-369 (2000).
   https://doi.org/10.1122/1.551088

.. [7] Picard, G., Ajdari, A., Lequeux, F., & Bocquet, L. "Elastic consequences of a single plastic event: A step towards the microscopic modeling of the flow of yield stress fluids."
   *European Physical Journal E*, **15**, 371-381 (2004).
   https://doi.org/10.1140/epje/i2004-10054-8

.. [8] Fielding, S. M. "Viscoelasticity and rheology near the soft glassy rheology transition."
   *Physical Review E*, **76**, 016311 (2007).
   https://doi.org/10.1103/PhysRevE.76.016311

.. [9] Bouchbinder, E. & Langer, J. S. "Nonequilibrium thermodynamics of driven amorphous materials. I. Internal degrees of freedom and volume deformation."
   *Physical Review E*, **80**, 031131 (2009).
   https://doi.org/10.1103/PhysRevE.80.031131

.. [10] Nicolas, A., Ferrero, E. E., Martens, K., & Barrat, J.-L. "Deformation and flow of amorphous solids: Insights from elastoplastic models."
   *Reviews of Modern Physics*, **90**, 045006 (2018).
   https://doi.org/10.1103/RevModPhys.90.045006

See Also
--------

- :doc:`/models/sgr/sgr_conventional` — SGR model (energy-based glass transition)
- :doc:`/models/stz/stz_model` — STZ model (defect-based plasticity)
- :doc:`/models/epm/lattice_epm` — EPM model (spatial avalanches)
- :doc:`/user_guide/03_advanced_topics/index` — Advanced elastoplastic modeling
