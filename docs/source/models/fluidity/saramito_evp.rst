.. _saramito_evp:

================================
Fluidity-Saramito EVP Model
================================

Quick Reference
---------------

**Use when:** Elastoviscoplastic materials with thixotropy

**Parameters:** G, τ_y, f, η_p, t_a, b

**Key equation:** :math:`\boldsymbol{\tau} + \lambda \stackrel{\nabla}{\boldsymbol{\tau}} = 2\eta_p \mathbf{D}`

**Test modes:** flow_curve, startup, creep, oscillation, LAOS

**Material examples:** Carbopol, hair gel, mayonnaise

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Class**
     - ``FluiditySaramitoLocal``, ``FluiditySaramitoNonlocal``
   * - **Physics**
     - Elastoviscoplastic with thixotropic fluidity
   * - **Coupling Modes**
     - ``"minimal"``, ``"full"``
   * - **Protocols**
     - FLOW_CURVE, CREEP, RELAXATION, STARTUP, OSCILLATION, LAOS
   * - **Key Features**
     - Tensorial stress, Von Mises yield, normal stresses, shear banding

**Import:**

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal, FluiditySaramitoNonlocal

**Basic Usage:**

.. code-block:: python

   # Minimal coupling (simplest, most identifiable)
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode="flow_curve")

   # Full coupling (aging yield stress)
   model = FluiditySaramitoLocal(coupling="full")
   model.fit(gamma_dot, sigma, test_mode="flow_curve")

Notation Guide
==============

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`\boldsymbol{\tau}`
     - Deviatoric stress tensor
     - Pa
   * - :math:`|\boldsymbol{\tau}|`
     - Von Mises equivalent stress
     - Pa
   * - :math:`\tau_y`
     - Yield stress
     - Pa
   * - :math:`f`
     - Fluidity
     - 1/(Pa·s)
   * - :math:`\lambda`
     - Relaxation time (= 1/f)
     - s
   * - :math:`G`
     - Elastic modulus
     - Pa
   * - :math:`\dot{\gamma}`
     - Shear rate
     - 1/s
   * - :math:`\alpha`
     - Plasticity parameter
     - dimensionless
   * - :math:`\xi`
     - Cooperativity length
     - m

Overview
========

The Fluidity-Saramito Elastoviscoplastic (EVP) model combines three key physical mechanisms:

1. **Viscoelasticity**: Upper-convected Maxwell framework with elastic recoil,
   storage modulus G', and first normal stress difference N₁.

2. **Viscoplasticity**: True Von Mises yield surface with Herschel-Bulkley
   plastic flow above yield.

3. **Thixotropy**: Time-dependent aging (structural build-up at rest) and
   shear rejuvenation (flow-induced breakdown) via fluidity evolution.

The model captures complex behaviors including:

- Stress overshoot in startup that increases with waiting time
- Creep bifurcation at the yield stress (bounded vs unbounded flow)
- Non-exponential stress relaxation
- Shear banding in spatially-resolved (nonlocal) variant

Physical Foundations
====================

Upper-Convected Maxwell Framework
---------------------------------

The stress evolution follows the upper-convected Maxwell model with plasticity:

.. math::

   \lambda \overset{\nabla}{\boldsymbol{\tau}} + \alpha(\boldsymbol{\tau})\boldsymbol{\tau} = 2\eta_p \mathbf{D}

where:

- :math:`\lambda = 1/f` is the fluidity-dependent relaxation time
- :math:`\overset{\nabla}{\boldsymbol{\tau}}` is the upper-convected derivative
- :math:`\alpha = \max(0, 1 - \tau_y/|\boldsymbol{\tau}|)` is the Von Mises plasticity
- :math:`\eta_p = G/f` is the polymeric viscosity
- :math:`\mathbf{D}` is the rate of deformation tensor

Von Mises Yield Criterion
-------------------------

The plasticity parameter :math:`\alpha` activates plastic flow only when
the Von Mises equivalent stress exceeds the yield stress:

.. math::

   \alpha = \max\left(0, 1 - \frac{\tau_y}{|\boldsymbol{\tau}|}\right)

where the Von Mises stress is:

.. math::

   |\boldsymbol{\tau}| = \sqrt{\frac{1}{2}\boldsymbol{\tau}:\boldsymbol{\tau}}

Fluidity Evolution
------------------

The fluidity evolves via competing aging and rejuvenation:

.. math::

   \frac{df}{dt} = \frac{f_\text{age} - f}{t_a} + b|\dot{\gamma}|^{n_\text{rej}}(f_\text{flow} - f)

where:

- :math:`f_\text{age}`: Equilibrium fluidity at rest (aged state)
- :math:`f_\text{flow}`: High-shear fluidity limit (rejuvenated state)
- :math:`t_a`: Aging timescale
- :math:`b`: Rejuvenation amplitude
- :math:`n_\text{rej}`: Rejuvenation rate exponent

Coupling Modes
--------------

**Minimal Coupling** (``coupling="minimal"``):

- Relaxation time: :math:`\lambda = 1/f`
- Yield stress: :math:`\tau_y = \tau_{y0}` (constant)
- Fewer parameters, easier to identify

**Full Coupling** (``coupling="full"``):

- Relaxation time: :math:`\lambda = 1/f`
- Yield stress: :math:`\tau_y(f) = \tau_{y0} + a_y/f^m`
- Captures aging yield stress (stronger when aged)

Mathematical Formulation
========================

Governing Equations
===================

Component Equations (Simple Shear)
----------------------------------

For simple shear with velocity gradient :math:`\mathbf{L} = \dot{\gamma}\mathbf{e}_x\mathbf{e}_y`:

.. math::

   \frac{d\tau_{xx}}{dt} &= 2\dot{\gamma}\tau_{xy} - \alpha f \tau_{xx} \\
   \frac{d\tau_{yy}}{dt} &= -\alpha f \tau_{yy} \\
   \frac{d\tau_{xy}}{dt} &= \dot{\gamma}\tau_{yy} + G\dot{\gamma} - \alpha f \tau_{xy}

The first normal stress difference is:

.. math::

   N_1 = \tau_{xx} - \tau_{yy}

At steady state in simple shear, this scales as :math:`N_1 \sim \lambda \dot{\gamma} \tau_{xy}`.

Steady-State Flow Curve
-----------------------

At steady state, the model reduces to Herschel-Bulkley form:

.. math::

   \sigma = \tau_y + K_\text{HB}\dot{\gamma}^{n_\text{HB}}

with fluidity-dependent parameters when using full coupling.

----

Validity and Assumptions
=========================

Model Assumptions
-----------------

1. **Tensorial stress description**: Uses full stress tensor :math:`\boldsymbol{\tau}`, suitable for 3D flows
2. **Von Mises yield criterion**: Isotropic yielding based on stress magnitude :math:`|\boldsymbol{\tau}|`
3. **Upper-convected Maxwell framework**: Appropriate for entangled polymers and structured fluids
4. **Thixotropic fluidity evolution**: Single scalar internal variable :math:`f` tracks structure
5. **Isothermal**: Temperature effects not explicitly modeled

Applicability
-------------

The model is most appropriate for:

- **Elastoviscoplastic fluids**: Materials exhibiting elastic recoil, yield stress, and time-dependent viscosity
- **Concentrated emulsions**: Mayonnaise, creams, cosmetics
- **Polymer gels**: Carbopol, hydrogels with yield stress
- **Pastes and slurries**: Cement, drilling muds, waxy crude oils
- **Complex loading**: Protocols requiring tensorial stress (extensional flow, normal stresses)

Limitations
-----------

**No shear banding (local model)**:
   The homogeneous (local) model cannot capture spatial heterogeneity. For shear-banded flows, use the :class:`FluiditySaramitoNonlocal` variant.

**Single structural variable**:
   Real materials may have multiple structural timescales or multi-mode thixotropy. Consider multi-lambda extensions for complex aging.

**Isotropic yielding**:
   The Von Mises criterion assumes isotropic yield surface. Anisotropic materials may require tensorial yield criteria.

**Small strain assumption**:
   The UCM framework assumes affine deformation. Not suitable for materials with wall slip or non-affine microstructure.

----

What You Can Learn
==================

From fitting Fluidity-Saramito EVP to experimental data, you can extract insights about elastoviscoplasticity, thixotropic coupling, and normal stress generation in yield-stress materials.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**f (Fluidity)**:
   Time-dependent inverse relaxation time controlling both viscosity η_p = G/f and relaxation time λ = 1/f in tensorial UCM framework.
   *For graduate students*: Unlike scalar fluidity models, here f couples to tensorial stress evolution: λ·∇τ + α(τ)τ = 2η_p·D where λ = 1/f. Evolution: df/dt = (f_age - f)/t_a + b\|γ̇\|^n_rej(f_flow - f). Plasticity α = max(0, 1 - τ_y/\|τ\|) activates only when Von Mises stress \|τ\| = √(½τ:τ) > τ_y. Normal stresses N₁ ~ λγ̇τ_xy ~ γ̇/f² (aged materials with low f exhibit larger N₁).
   *For practitioners*: f_age ≈ 10^-6 to 10^-3 s^-1 (solid-like), f_flow ≈ 10^-2 to 1 s^-1 (liquid-like). Measure via startup overshoot magnitude (larger overshoot = lower initial f after aging).

**G (Elastic Modulus)**:
   Elastic stiffness in UCM backbone, controlling both stress buildup and normal stress generation.
   *For graduate students*: Sets stress scale and Weissenberg number Wi = λγ̇ = γ̇/f. Normal stress scaling: N₁/τ_xy ~ Wi ~ γ̇/f. For UCM, N₂ = 0. Startup overshoot: σ_peak ~ τ_y + Gγ_y where γ_y is yield strain.
   *For practitioners*: Extract from initial slope in startup or from G' plateau in SAOS. Typical: G = 10²-10⁴ Pa (soft colloids), 10⁴-10⁶ Pa (polymer gels).

**τ_y (Yield Stress)**:
   Von Mises yield criterion threshold. In minimal coupling, constant. In full coupling: τ_y(f) = τ_y0 + a_y/f^m (aged materials are stronger).
   *For graduate students*: Plasticity parameter α = max(0, 1 - τ_y/\|τ\|) controls plastic dissipation term. α = 0 below yield (elastic), α > 0 above yield (viscoplastic flow). Full coupling exponent m typically 0.3-1.0 captures aging-induced hardening.
   *For practitioners*: Measure from flow curve low-shear plateau or creep bifurcation stress. Full coupling appropriate if yield stress increases significantly after aging (wait-time dependent startup tests).

**t_a, b, n_rej (Fluidity Evolution Parameters)**:
   Control aging (t_a) and rejuvenation (b, n_rej) kinetics analogous to local fluidity model.
   *For graduate students*: df/dt = (f_age - f)/t_a + b\|γ̇\|^n_rej(f_flow - f). Characteristic shear rate: γ̇_c ~ (1/(bt_a))^(1/n_rej). Stress overshoot magnitude and position depend on t_a (waiting time scaling), b and n_rej (breakdown kinetics).
   *For practitioners*: Extract from startup test family at different wait times. Typical: t_a = 10-1000 s, b = 0.1-10, n_rej = 0.5-1.5. Longer t_a = more pronounced thixotropy.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Fluidity-Saramito Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - N₁/τ_xy < 0.1, minimal coupling
     - Weakly elastic EVP
     - Carbopol gels, pastes
     - Dominant viscoplasticity, minimal normal stresses
   * - N₁/τ_xy = 0.1-1, minimal coupling
     - Moderate elasticity
     - Emulsions, soft colloids, cosmetics
     - Significant rod climbing, moderate die swell
   * - N₁/τ_xy > 1, minimal coupling
     - Strongly elastic EVP
     - Polymer gels, fiber suspensions
     - Strong Weissenberg effect, edge fracture risk
   * - Full coupling: m > 0.5
     - Strong aging-yield coupling
     - Waxy crude oils, cement
     - Wait-time dependent yield, restart challenges
   * - t_a > 100 s, full coupling
     - Extreme thixotropy with elasticity
     - Waxy crude oils, drilling muds
     - Long-term aging, complex restart dynamics

----

Parameters
==========

Core Parameters
---------------

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``G``
     - Elastic modulus
     - Pa
     - 1e4
     - [1e1, 1e8]
   * - ``tau_y0``
     - Base yield stress
     - Pa
     - 100
     - [0.1, 1e5]
   * - ``K_HB``
     - HB consistency index
     - Pa·s^n
     - 50
     - [1e-2, 1e5]
   * - ``n_HB``
     - HB flow exponent
     - —
     - 0.5
     - [0.1, 1.5]

Fluidity Parameters
-------------------

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``f_age``
     - Aging fluidity limit
     - 1/(Pa·s)
     - 1e-6
     - [1e-12, 1e-2]
   * - ``f_flow``
     - Flow fluidity limit
     - 1/(Pa·s)
     - 1e-2
     - [1e-6, 1.0]
   * - ``t_a``
     - Aging timescale
     - s
     - 10
     - [0.01, 1e5]
   * - ``b``
     - Rejuvenation amplitude
     - —
     - 1.0
     - [0, 1e3]
   * - ``n_rej``
     - Rejuvenation exponent
     - —
     - 1.0
     - [0.1, 3.0]

Full Coupling Parameters
------------------------

Only active when ``coupling="full"``:

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``tau_y_coupling``
     - Yield stress coupling
     - Pa·(Pa·s)^m
     - 1.0
     - [0, 1e4]
   * - ``m_yield``
     - Yield stress exponent
     - —
     - 0.5
     - [0.1, 2.0]

Nonlocal Parameters
-------------------

Only for ``FluiditySaramitoNonlocal``:

.. list-table::
   :widths: 15 50 15 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - Units
     - Default
     - Bounds
   * - ``xi``
     - Cooperativity length
     - m
     - 1e-5
     - [1e-7, 1e-2]

Parameter Interpretation by Material
------------------------------------

**Concentrated Emulsions** (mayonnaise, cosmetics):

- :math:`\tau_y \sim 10-100` Pa
- :math:`n_\text{HB} \sim 0.3-0.5`
- :math:`t_a \sim 10-100` s

**Polymer Gels** (carbopol, hydrogels):

- :math:`\tau_y \sim 1-50` Pa
- :math:`n_\text{HB} \sim 0.4-0.6`
- :math:`t_a \sim 1-1000` s (depends on concentration)

**Cement/Concrete**:

- :math:`\tau_y \sim 100-1000` Pa
- :math:`n_\text{HB} \sim 0.2-0.4`
- :math:`t_a \sim 100-10000` s (hydration-dependent)

**Drilling Muds**:

- :math:`\tau_y \sim 5-50` Pa
- :math:`n_\text{HB} \sim 0.3-0.7`
- :math:`t_a \sim 10-1000` s

----

Usage
=====

(Usage examples already present in the file)

----

See Also
========

- :doc:`fluidity_local` — Scalar fluidity model without tensorial stress
- :doc:`fluidity_nonlocal` — Nonlocal model for shear banding
- :doc:`../dmt/dmt_local` — DMT structural-kinetics thixotropic model
- :doc:`../flow/herschel_bulkley` — Simpler yield stress model (steady shear only)

----

Fitting Guidance
================

Recommended Workflow
--------------------

1. **Start with flow curve** to get τ_y, K_HB, n_HB
2. **Add startup** to get G and fluidity dynamics (t_a, b, n_rej)
3. **Use creep** to validate τ_y (bifurcation point)
4. **Optionally use LAOS** for nonlinear validation

Step-by-Step Example
--------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np

   # 1. Flow curve fitting
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma, test_mode="flow_curve")

   # 2. Startup fitting (refines G and fluidity parameters)
   model.fit(t, sigma_startup, test_mode="startup", gamma_dot=1.0)

   # 3. Bayesian inference for uncertainty
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode="flow_curve",
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
   )

   # 4. Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   for param, stats in intervals.items():
       print(f"{param}: {stats['mean']:.2f} [{stats['lower']:.2f}, {stats['upper']:.2f}]")

Troubleshooting
---------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Issue
     - Solution
   * - No stress overshoot
     - Increase ``b`` (rejuvenation) or decrease ``t_a`` (aging time)
   * - Overshoot too large
     - Decrease ``b`` or increase ``f_age``
   * - Flow curve too flat
     - Decrease ``n_HB`` (more shear-thinning)
   * - Poor creep fit
     - Check ``tau_y0`` against bifurcation point in data
   * - Bayesian divergences
     - Use NLSQ warm-start, increase ``num_warmup``

Usage Examples
==============

Basic Flow Curve Fitting
------------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np

   # Generate synthetic data
   gamma_dot = np.logspace(-2, 2, 30)
   sigma_data = 100 + 50 * gamma_dot**0.5  # HB-like

   # Fit model
   model = FluiditySaramitoLocal(coupling="minimal")
   model.fit(gamma_dot, sigma_data, test_mode="flow_curve")

   # Predict
   sigma_pred = model.predict(gamma_dot)

   # Get yield stress
   tau_y = model.parameters.get_value("tau_y0")
   print(f"Fitted yield stress: {tau_y:.1f} Pa")

Startup with Stress Overshoot
-----------------------------

.. code-block:: python

   # Simulate startup
   t = np.linspace(0, 50, 500)
   gamma_dot = 1.0
   t_wait = 100.0  # Waiting time before startup

   strain, stress, fluidity = model.simulate_startup(t, gamma_dot, t_wait=t_wait)

   # Analyze overshoot
   sigma_max = np.max(stress)
   sigma_ss = stress[-1]
   overshoot_ratio = sigma_max / sigma_ss
   print(f"Overshoot ratio: {overshoot_ratio:.2f}")

Creep Bifurcation
-----------------

.. code-block:: python

   t = np.linspace(0, 1000, 500)

   # Below yield - bounded strain
   strain_below, _ = model.simulate_creep(t, sigma_applied=50.0)

   # Above yield - unbounded flow
   strain_above, _ = model.simulate_creep(t, sigma_applied=150.0)

   # Plot shows bifurcation behavior

Normal Stress Predictions
-------------------------

.. code-block:: python

   gamma_dot = np.array([0.1, 1.0, 10.0])
   N1, N2 = model.predict_normal_stresses(gamma_dot)

   print(f"N1 at γ̇=1: {N1[1]:.1f} Pa")
   # N2 = 0 for UCM

LAOS Analysis
-------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoLocal
   import numpy as np
   import matplotlib.pyplot as plt

   # Large amplitude oscillatory shear
   model = FluiditySaramitoLocal(coupling="minimal")

   gamma_0 = 1.0  # 100% strain
   omega = 1.0    # rad/s

   # Simulate LAOS
   t = np.linspace(0, 2*np.pi/omega * 3, 300)  # 3 cycles
   strain, stress, fluidity = model.simulate_laos(t, gamma_0, omega)

   # Plot Lissajous curves
   plt.figure(figsize=(8, 6))
   plt.plot(strain, stress)
   plt.xlabel('Strain')
   plt.ylabel('Stress (Pa)')
   plt.title('LAOS Lissajous Curve')
   plt.grid(True)

Nonlocal Model with Shear Banding
---------------------------------

.. code-block:: python

   from rheojax.models.fluidity import FluiditySaramitoNonlocal

   # Create nonlocal model
   model = FluiditySaramitoNonlocal(
       coupling="minimal",
       N_y=51,      # Grid points
       H=1e-3,      # Gap width (m)
       xi=1e-5,     # Cooperativity length (m)
   )

   # Simulate startup
   t = np.linspace(0, 50, 200)
   _, sigma, f_field = model.simulate_startup(t, gamma_dot=0.1)

   # Check for shear banding
   is_banded, cv, ratio = model.detect_shear_bands()
   print(f"Shear banding: {is_banded}, CV={cv:.2f}, ratio={ratio:.1f}")

   # Get detailed metrics
   metrics = model.get_banding_metrics()
   print(f"Band fraction: {metrics['band_fraction']:.2f}")

Bayesian Inference
------------------

.. code-block:: python

   # Fit with NLSQ first (warm-start)
   model.fit(gamma_dot, sigma, test_mode="flow_curve")

   # Bayesian inference
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode="flow_curve",
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,  # Production-ready diagnostics
       seed=42,       # Reproducibility
   )

   # Check diagnostics
   # R-hat should be < 1.01, ESS > 400

   # Plot with ArviZ
   from rheojax.pipeline.bayesian import BayesianPipeline

   pipeline = BayesianPipeline()
   pipeline._idata = result.idata
   pipeline.plot_trace()
   pipeline.plot_pair(divergences=True)

Comparison with Existing Models
===============================

vs FluidityLocal
----------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature
     - FluidityLocal
     - FluiditySaramitoLocal
   * - Stress tensor
     - Scalar σ
     - Tensorial [τ_xx, τ_yy, τ_xy]
   * - Normal stresses
     - No
     - Yes (N₁ from UCM)
   * - Yield criterion
     - None (implicit)
     - Von Mises (explicit)
   * - Elastic effects
     - Maxwell-like
     - Upper-convected Maxwell
   * - Parameters
     - 9
     - 10-12

The Saramito model is preferred when:

- Normal stresses (N₁) are important
- Tensorial stress state is needed
- True yield criterion is required

vs Standard Saramito (no fluidity)
----------------------------------

The fluidity extension adds:

- Thixotropic time dependence
- Aging/rejuvenation dynamics
- Shear banding capability (nonlocal)

Without fluidity, the standard Saramito model has constant relaxation time
and cannot capture thixotropic behaviors.

References
==========

.. [1] Saramito, P. "A new constitutive equation for elastoviscoplastic
   fluid flows." *Journal of Non-Newtonian Fluid Mechanics*, 145, 1-14 (2007).
   https://doi.org/10.1016/j.jnnfm.2006.04.012

.. [2] Saramito, P. "A new elastoviscoplastic model based on the
   Herschel-Bulkley viscoplastic model." *Journal of Non-Newtonian Fluid Mechanics*,
   158, 154-161 (2009). https://doi.org/10.1016/j.jnnfm.2008.12.001

.. [3] Coussot, P., Nguyen, Q. D., Huynh, H. T., and Bonn, D. "Viscosity bifurcation
   in thixotropic, yielding fluids." *Journal of Rheology*, 46(3), 573-589 (2002).
   https://doi.org/10.1122/1.1459447

.. [4] Bocquet, L., Colin, A., and Ajdari, A. "Kinetic theory of plastic flow
   in soft glassy materials." *Physical Review Letters*, 103, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

.. [5] Ovarlez, G., Mahaut, F., Deboeuf, S., Lenoir, N., Hormozi, S., and Chateau, X.
   "Phenomenology and physical origin of shear-localization and shear banding
   in complex fluids." *Journal of Non-Newtonian Fluid Mechanics*, 177-178, 19-28 (2012).
   https://doi.org/10.1016/j.jnnfm.2012.03.011
.. [6] Oldroyd, J. G. "On the formulation of rheological equations of state."
   *Proceedings of the Royal Society A*, **200**, 523-541 (1950).
   https://doi.org/10.1098/rspa.1950.0035

.. [7] Dimitriou, C. J., Casanellas, L., Ober, T. J., & McKinley, G. H. "Rheo-PIV of a shear-banding wormlike micellar solution under large amplitude oscillatory shear."
   *Rheologica Acta*, **51**, 395-411 (2012).
   https://doi.org/10.1007/s00397-012-0619-9

.. [8] Fraggedakis, D., Dimakopoulos, Y., & Tsamopoulos, J. "Yielding the yield stress analysis: A thorough comparison of recently proposed elasto-visco-plastic (EVP) fluid models."
   *Journal of Non-Newtonian Fluid Mechanics*, **238**, 170-188 (2016).
   https://doi.org/10.1016/j.jnnfm.2016.11.007

.. [9] de Souza Mendes, P. R. & Thompson, R. L. "A critical overview of elasto-viscoplastic thixotropic modeling."
   *Journal of Non-Newtonian Fluid Mechanics*, **187-188**, 8-15 (2012).
   https://doi.org/10.1016/j.jnnfm.2012.08.006

.. [10] Balmforth, N. J., Frigaard, I. A., & Ovarlez, G. "Yielding to stress: Recent developments in viscoplastic fluid mechanics."
   *Annual Review of Fluid Mechanics*, **46**, 121-146 (2014).
   https://doi.org/10.1146/annurev-fluid-010313-141424


API Reference
=============

.. autoclass:: rheojax.models.fluidity.saramito.FluiditySaramitoLocal
   :members:
   :inherited-members:
   :no-index:

.. autoclass:: rheojax.models.fluidity.saramito.FluiditySaramitoNonlocal
   :members:
   :inherited-members:
   :no-index:
