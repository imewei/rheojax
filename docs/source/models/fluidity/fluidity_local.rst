.. _model-fluidity-local:

=====================================================
Fluidity Local (Homogeneous Fluidity Model) — Handbook
=====================================================

Quick Reference
---------------

**Use when:** Yield-stress fluids, thixotropic materials, aging systems with homogeneous (spatially uniform) flow
**Parameters:** 6 (G, f_eq, f_inf, tau_age, a, n)
**Key equation:** :math:`\dot{\sigma} = G\dot{\gamma} - f(t)\sigma`
**Test modes:** Oscillation, relaxation, creep, steady shear, start-up, LAOS
**Material examples:** Mayonnaise, drilling muds, waxy crude oils, colloidal gels, greases, thixotropic paints

.. contents:: Table of Contents
   :depth: 3
   :local:

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`f(t)`
     - Fluidity (inverse relaxation time), :math:`f = 1/\tau`
   * - :math:`G`
     - Elastic modulus
   * - :math:`\sigma`
     - Shear stress
   * - :math:`\dot{\gamma}`
     - Shear rate
   * - :math:`f_{\rm eq}`
     - Equilibrium (resting) fluidity
   * - :math:`f_\infty`
     - Infinite-shear fluidity
   * - :math:`\tau_{\rm age}`
     - Aging timescale
   * - :math:`a, n`
     - Rejuvenation parameters

Overview
--------

The Local Fluidity Model provides a minimal yet powerful description of yield-stress fluids and thixotropic materials. Rather than prescribing a fixed viscosity, the model introduces a time-dependent **fluidity** field :math:`f(t) = 1/\tau(t)`, which represents the material's inverse characteristic relaxation time.

The key physical insight is the competition between two opposing processes:

1. **Aging (structural buildup):** At rest, the material's microstructure rebuilds, decreasing fluidity toward a solid-like equilibrium
2. **Rejuvenation (shear-induced breakdown):** Under flow, mechanical forcing breaks down structure, increasing fluidity toward a liquid-like state

This competition naturally produces:
   - Yield stress behavior (solid-like at rest)
   - Thixotropy (time-dependent viscosity)
   - Stress overshoots in start-up flows
   - Power-law creep and relaxation

Historical Context
~~~~~~~~~~~~~~~~~~

Fluidity-based models trace their origins to Bingham's concept of a "coefficient of mobility" (1922) and were formalized by Coussot, Nguyen, and collaborators [1]_ [2]_. The local (0D) model presented here is the spatially homogeneous limit, suitable when shear banding and spatial gradients are negligible.

The approach is closely related to:
   - **Lambda models** for thixotropy (de Souza Mendes, Mujumdar)
   - **Inelastic/elastic thixotropic models** (Mewis, Wagner)
   - **Kinetic theory approaches** (Dullaert, Mewis)

----

Physical Foundations
--------------------

Maxwell-Like Constitutive Framework
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model adopts a Maxwell-type viscoelastic framework with a dynamically evolving relaxation rate:

.. math::

   \dot{\sigma}(t) = G \dot{\gamma}(t) - f(t) \sigma(t)

This equation describes:
   - **Elastic loading**: The :math:`G\dot{\gamma}` term represents elastic stress buildup
   - **Viscous relaxation**: The :math:`-f\sigma` term represents stress dissipation with rate proportional to current fluidity

When :math:`f` is constant, this reduces to the standard Maxwell model with :math:`\tau = 1/f`. The novelty is allowing :math:`f` to evolve dynamically based on deformation history.

Aging vs. Rejuvenation Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fluidity evolution captures the microstructural kinetics:

.. math::

   \dot{f}(t) = \underbrace{\frac{f_{\rm eq} - f}{\tau_{\rm age}}}_{\text{aging}} + \underbrace{a |\dot{\gamma}(t)|^n (f_\infty - f)}_{\text{rejuvenation}}

**Aging term:** Drives fluidity toward :math:`f_{\rm eq}` on timescale :math:`\tau_{\rm age}`. For yield-stress fluids, :math:`f_{\rm eq} \approx 0` (solid-like at rest).

**Rejuvenation term:** Flow at rate :math:`|\dot{\gamma}|` drives fluidity toward :math:`f_\infty` (fluid-like). The power :math:`n` controls the sensitivity to shear rate:
   - :math:`n = 1`: Linear shear-rate dependence
   - :math:`n > 1`: Super-linear (stronger at high rates)
   - :math:`n < 1`: Sub-linear (saturating at high rates)

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

The fluidity :math:`f` can be interpreted as:

1. **Inverse viscosity**: :math:`\eta_{\rm eff} = G/f`
2. **Relaxation rate**: Characteristic time :math:`\tau = 1/f`
3. **Structural parameter**: Higher :math:`f` means more "broken down" structure
4. **Mobility**: Rate at which the material can flow under stress

.. tip:: **Key Physical Intuition**

   Think of :math:`f` as measuring "how liquid" the material currently is:

   - **Low** :math:`f` (near :math:`f_{\rm eq}`): Solid-like, structured, high viscosity
   - **High** :math:`f` (near :math:`f_\infty`): Liquid-like, broken down, low viscosity

----

Mathematical Formulation
------------------------

Core Equations
~~~~~~~~~~~~~~

The Local Fluidity Model is defined by two coupled ordinary differential equations:

**Constitutive stress equation:**

.. math::
   :label: stress-evolution

   \dot{\sigma}(t) = G \dot{\gamma}(t) - f(t) \sigma(t)

**Fluidity evolution equation:**

.. math::
   :label: fluidity-evolution

   \dot{f}(t) = \frac{f_{\rm eq} - f}{\tau_{\rm age}} + a |\dot{\gamma}(t)|^n (f_\infty - f)

Initial conditions: :math:`\sigma(0) = \sigma_0`, :math:`f(0) = f_0` (typically :math:`f_0 = f_{\rm eq}` for a well-rested sample).

Steady-State Analysis
~~~~~~~~~~~~~~~~~~~~~

Under constant shear rate :math:`\dot{\gamma}`, the system reaches steady state with :math:`\dot{\sigma} = 0` and :math:`\dot{f} = 0`:

**Steady fluidity:**

.. math::
   :label: steady-fluidity

   f_{\rm ss}(\dot{\gamma}) = \frac{f_{\rm eq}/\tau_{\rm age} + a|\dot{\gamma}|^n f_\infty}{1/\tau_{\rm age} + a|\dot{\gamma}|^n}

**Steady stress (flow curve):**

.. math::
   :label: flow-curve

   \sigma_{\rm ss}(\dot{\gamma}) = \frac{G \dot{\gamma}}{f_{\rm ss}(\dot{\gamma})}

This produces a flow curve with:
   - **Yield stress** :math:`\sigma_y = G \cdot \lim_{\dot{\gamma}\to 0} \dot{\gamma}/f_{\rm ss}(\dot{\gamma})` when :math:`f_{\rm eq} \approx 0`
   - **Shear-thinning** at high rates as :math:`f_{\rm ss} \to f_\infty`

Effective Viscosity
~~~~~~~~~~~~~~~~~~~

The effective viscosity at steady state:

.. math::

   \eta_{\rm eff}(\dot{\gamma}) = \frac{\sigma_{\rm ss}}{\dot{\gamma}} = \frac{G}{f_{\rm ss}(\dot{\gamma})}

At low shear rates (with :math:`f_{\rm eq} \ll f_\infty`):

.. math::

   \eta_{\rm eff} \sim \frac{G \tau_{\rm age}}{f_{\rm eq}} \to \infty \quad \text{as } f_{\rm eq} \to 0

At high shear rates:

.. math::

   \eta_{\rm eff} \to \frac{G}{f_\infty}

----

Protocol-Specific Equations
---------------------------

Rotation (Steady-State Flow Curve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** Constant global shear rate :math:`\dot{\gamma} = \text{const}`.

**Equations:**

.. math::

   \dot{\sigma} = G\dot{\gamma} - f\sigma, \qquad
   \dot{f} = (f_{\rm eq}-f)/\tau_{\rm age} + a|\dot{\gamma}|^n(f_\infty-f)

**Steady state:**

.. math::

   \sigma(\dot{\gamma}) = \frac{G\dot{\gamma}}{f(\dot{\gamma})}, \qquad
   f(\dot{\gamma}) = \frac{f_{\rm eq}/\tau_{\rm age} + a|\dot{\gamma}|^n f_\infty}{1/\tau_{\rm age} + a|\dot{\gamma}|^n}

**Behavior:**
   - For :math:`f_{\rm eq} \approx 0`: Apparent yield stress emerges
   - Shear-thinning at all rates
   - Flow index related to exponent :math:`n`

Start-Up Shear
~~~~~~~~~~~~~~

**Protocol:** Apply constant shear rate :math:`\dot{\gamma}_0` at :math:`t = 0` from rest.

.. math::

   \dot{\gamma}(t) = \dot{\gamma}_0 H(t)

**Equations:**

.. math::

   \dot{\sigma} = G\dot{\gamma}_0 - f\sigma, \qquad
   \dot{f} = (f_{\rm eq}-f)/\tau_{\rm age} + a|\dot{\gamma}_0|^n(f_\infty-f)

**Initial conditions:** :math:`\sigma(0) = 0`, :math:`f(0) = f_{\rm eq}`

**Behavior:**
   - Initial elastic response: :math:`\sigma \approx G \dot{\gamma}_0 t` for :math:`t \ll 1/f_{\rm eq}`
   - **Stress overshoot** if :math:`\tau_{\rm age}` is large (aging competes with rejuvenation)
   - Approach to steady state: :math:`\sigma \to \sigma_{\rm ss}(\dot{\gamma}_0)`

The stress overshoot reflects the lag between structural breakdown and stress relaxation.

Stress Relaxation (Step Strain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** Apply step strain :math:`\gamma_0` at :math:`t = 0`, then hold :math:`\dot{\gamma} = 0`.

**Initial stress:** :math:`\sigma(0^+) = G\gamma_0`

**Equations for** :math:`t > 0`:

.. math::

   \dot{\sigma} = -f\sigma, \qquad
   \dot{f} = (f_{\rm eq}-f)/\tau_{\rm age}

**Solution:**

.. math::

   \sigma(t) = \sigma(0^+) \exp\left(-\int_0^t f(s) \, ds\right)

With :math:`f(t) = f_0 e^{-t/\tau_{\rm age}} + f_{\rm eq}(1 - e^{-t/\tau_{\rm age}})`:

**Behavior:**
   - If :math:`f_{\rm eq} \approx 0`: Stress decays more slowly as :math:`f \to 0` (aging)
   - **Non-exponential relaxation** due to evolving fluidity
   - For aged samples (:math:`f_0 = f_{\rm eq} \approx 0`): :math:`\sigma(t) \approx \sigma(0^+)` (solid-like plateau)

Creep (Step Stress)
~~~~~~~~~~~~~~~~~~~

**Protocol:** Apply constant stress :math:`\Sigma_0` at :math:`t = 0`.

**Equations:**

.. math::

   \dot{\gamma}(t) = \frac{\Sigma_0}{G} f(t)

.. math::

   \dot{f} = (f_{\rm eq}-f)/\tau_{\rm age} + a\left|\frac{\Sigma_0}{G}f\right|^n (f_\infty-f)

**Strain accumulation:**

.. math::

   \gamma(t) = \frac{\Sigma_0}{G} \int_0^t f(s) \, ds

**Behavior:**
   - **Bifurcation** at yield stress :math:`\sigma_y`:
      - :math:`\Sigma_0 < \sigma_y`: Fluidity decays, strain saturates (solid-like)
      - :math:`\Sigma_0 > \sigma_y`: Fluidity grows, steady flow (liquid-like)
   - **Delayed yielding** near :math:`\sigma_y`: Long induction time before flow onset
   - Power-law creep regime: :math:`\gamma(t) \sim t^\alpha` with :math:`\alpha < 1` during transient

Oscillatory Shear (SAOS and LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol (strain-controlled):**

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t), \qquad
   \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**Equations:**

.. math::

   \dot{\sigma} = G\dot{\gamma}(t) - f(t)\sigma, \qquad
   \dot{f} = (f_{\rm eq}-f)/\tau_{\rm age} + a|\dot{\gamma}(t)|^n(f_\infty-f)

**SAOS (Small Amplitude):**
   - Fluidity nearly constant: :math:`f(t) \approx \bar{f}`
   - Linear moduli from stress-strain relationship
   - Storage modulus: :math:`G' \approx \frac{G \omega^2}{f^2 + \omega^2}`
   - Loss modulus: :math:`G'' \approx \frac{G \omega f}{f^2 + \omega^2}`

**LAOS (Large Amplitude):**
   - Strong :math:`f(t)` modulation within each cycle
   - Higher harmonics in stress response
   - Intracycle softening and stiffening
   - Lissajous-Bowditch curves show nonlinear features

----

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 12 12 10 18 48

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``G``
     - :math:`G`
     - Pa
     - :math:`G > 0`
     - Elastic modulus; sets stress scale
   * - ``f_eq``
     - :math:`f_{\rm eq}`
     - s\ :sup:`-1`
     - :math:`f_{\rm eq} \geq 0`
     - Equilibrium fluidity; :math:`\approx 0` for yield-stress fluids
   * - ``f_inf``
     - :math:`f_\infty`
     - s\ :sup:`-1`
     - :math:`f_\infty > f_{\rm eq}`
     - Infinite-shear fluidity; sets minimum viscosity
   * - ``tau_age``
     - :math:`\tau_{\rm age}`
     - s
     - :math:`\tau_{\rm age} > 0`
     - Aging timescale; controls buildup rate at rest
   * - ``a``
     - :math:`a`
     - s\ :sup:`n-1`
     - :math:`a > 0`
     - Rejuvenation rate coefficient
   * - ``n``
     - :math:`n`
     - —
     - :math:`n > 0`
     - Rejuvenation exponent; typically :math:`0.5 \leq n \leq 2`

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G (Elastic Modulus):**
   - **Physical meaning**: Stiffness of the elastic network
   - **Typical ranges**:
      - Colloidal gels: :math:`10^0 - 10^2` Pa
      - Emulsions: :math:`10^1 - 10^3` Pa
      - Drilling muds: :math:`10^1 - 10^2` Pa

**f_eq (Equilibrium Fluidity):**
   - **Physical meaning**: Fluidity at complete rest
   - **For yield-stress fluids**: :math:`f_{\rm eq} \approx 10^{-6}` to :math:`10^{-3}` s\ :sup:`-1`
   - **For viscoelastic fluids without yield stress**: :math:`f_{\rm eq} > 0`

**f_inf (Infinite-Shear Fluidity):**
   - **Physical meaning**: Fluidity at very high shear (fully broken-down structure)
   - **Typical ranges**: :math:`10^{-1}` to :math:`10^2` s\ :sup:`-1`
   - **High-shear viscosity**: :math:`\eta_\infty = G / f_\infty`

**tau_age (Aging Timescale):**
   - **Physical meaning**: Time for structure to rebuild at rest
   - **Typical ranges**:
      - Fast aging: :math:`1 - 10` s
      - Slow aging: :math:`10^2 - 10^4` s
   - **Thixotropic time**: Related to loop hysteresis in flow curves

**a, n (Rejuvenation Parameters):**
   - **Physical meaning**: Sensitivity of breakdown to shear rate
   - **Typical values**: :math:`a \sim 0.1 - 10`, :math:`n \sim 0.5 - 1.5`
   - **Relationship to flow index**: Connected to Herschel-Bulkley exponent

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Homogeneous flow**: No spatial gradients in fluidity or velocity (shear banding excluded)
2. **Affine deformation**: Microstructure deforms with macroscopic strain
3. **First-order kinetics**: Simple exponential approach to equilibrium
4. **Isothermal**: Temperature effects not explicitly modeled
5. **Scalar fluidity**: Single internal variable (no tensorial microstructure)

Data Requirements
~~~~~~~~~~~~~~~~~

- **Flow curves**: Steady shear :math:`\sigma(\dot{\gamma})` over 2+ decades
- **Transient data**: Start-up, step stress, or hysteresis loops
- **Optional**: Oscillatory sweeps, relaxation tests

Limitations
~~~~~~~~~~~

**No shear banding:**
   The homogeneous model cannot capture spatial heterogeneity. For shear-banded
   flows, use the :doc:`fluidity_nonlocal` model.

**Simple kinetics:**
   Real materials may have multiple structural timescales. Consider multi-lambda
   extensions for complex thixotropic behavior.

**No normal stresses:**
   The scalar model does not predict :math:`N_1, N_2`. Use tensorial extensions
   for extensional flows or normal stress measurements.

**Pre-yielding elasticity:**
   The linear :math:`G\dot{\gamma}` term may not capture complex pre-yield
   behavior (e.g., fatigue, microslip).

----

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From flow curve**

**Step 1**: Identify yield stress :math:`\sigma_y` from :math:`\sigma(\dot{\gamma})` plot

**Step 2**: High-shear viscosity: :math:`\eta_\infty = \lim_{\dot{\gamma}\to\infty} \sigma/\dot{\gamma}`

**Step 3**: Estimate :math:`G \approx 10 \sigma_y` (typical for yield-stress fluids)

**Step 4**: :math:`f_\infty = G / \eta_\infty`

**Step 5**: :math:`f_{\rm eq} \approx 10^{-4} f_\infty` (small but nonzero for numerical stability)

**Method 2: From transient response**

**Step 1**: Fit exponential decay in step-strain relaxation to get :math:`\tau_{\rm age}`

**Step 2**: Measure stress overshoot magnitude and peak time in start-up

**Step 3**: Use overshoot characteristics to estimate :math:`a, n`

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for quick fits (6 parameters)
   - Use good initial guesses from flow curve analysis

**Bayesian inference (NUTS)**
   - Highly recommended for uncertainty quantification
   - Critical for distinguishing yield stress vs. very high viscosity
   - Use informative priors from physical constraints

**Bounds:**
   - :math:`G`: [1e-1, 1e6] Pa
   - :math:`f_{\rm eq}`: [1e-8, 1e-1] s\ :sup:`-1`
   - :math:`f_\infty`: [1e-2, 1e3] s\ :sup:`-1`
   - :math:`\tau_{\rm age}`: [1e-1, 1e5] s
   - :math:`a`: [1e-3, 1e2]
   - :math:`n`: [0.3, 2.5]

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics
   :header-rows: 1
   :widths: 28 36 36

   * - Problem
     - Diagnostic
     - Solution
   * - :math:`f_{\rm eq}` hits lower bound
     - True yield stress material
     - Fix at small value (e.g., :math:`10^{-6}`)
   * - Poor transient fits
     - Wrong :math:`\tau_{\rm age}` or :math:`a, n`
     - Fit transient data separately first
   * - Oscillations in prediction
     - Stiff ODE system
     - Reduce time step; use implicit integrator
   * - Flow curve mismatch at low :math:`\dot{\gamma}`
     - Wall slip or banding
     - Consider :doc:`fluidity_nonlocal` model
   * - Fitted :math:`\tau_{\rm age}` unrealistic
     - Data insufficient for aging dynamics
     - Include step-stress or hysteresis data

----

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import FluidityLocal

   # Shear rate data
   gamma_dot = np.logspace(-3, 2, 50)

   # Create and fit model
   model = FluidityLocal()
   model.fit(gamma_dot, sigma_data, test_mode='steady_shear')

   # Extract parameters
   G = model.parameters.get_value('G')
   f_eq = model.parameters.get_value('f_eq')
   tau_age = model.parameters.get_value('tau_age')

   print(f"Elastic modulus G = {G:.1f} Pa")
   print(f"Aging timescale = {tau_age:.1f} s")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityLocal

   model = FluidityLocal()
   model.fit(gamma_dot, sigma_data, test_mode='steady_shear')

   # Bayesian with warm-start
   result = model.fit_bayesian(
       gamma_dot, sigma_data,
       test_mode='steady_shear',
       num_warmup=1000,
       num_samples=2000
   )

   # Get credible intervals for yield stress
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"tau_age 95% CI: [{intervals['tau_age'][0]:.1f}, {intervals['tau_age'][1]:.1f}] s")

Transient Start-Up
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityLocal
   import numpy as np

   model = FluidityLocal()

   # Fit to flow curve first
   model.fit(gamma_dot, sigma_ss, test_mode='steady_shear')

   # Predict start-up transient
   t = np.linspace(0, 100, 1000)
   gamma_dot_0 = 1.0  # Applied shear rate
   sigma_t = model.predict_startup(t, gamma_dot_0)

Creep Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityLocal
   import numpy as np

   model = FluidityLocal()
   model.fit(gamma_dot, sigma_data, test_mode='steady_shear')

   # Predict creep response
   t = np.logspace(-2, 4, 100)
   sigma_0 = 50.0  # Applied stress (Pa)
   gamma_t = model.predict_creep(t, sigma_0)

   # Check for delayed yielding
   yield_time = t[np.argmax(np.gradient(gamma_t) > threshold)]

----

See Also
--------

- :doc:`fluidity_nonlocal` — Nonlocal model for shear banding and cooperative flows
- :doc:`../sgr/sgr_conventional` — Soft Glassy Rheology model with trap distributions
- :doc:`../hl/hebraud_lequeux` — Hébraud-Lequeux model for pasty materials
- :doc:`../flow/herschel_bulkley` — Simpler yield stress model for steady shear only

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FluidityLocal`

----

References
----------

.. [1] Coussot, P., Nguyen, Q. D., Huynh, H. T., & Bonn, D. "Avalanche behavior in yield stress fluids."
   *Physical Review Letters*, **88**, 175501 (2002).
   https://doi.org/10.1103/PhysRevLett.88.175501

.. [2] Coussot, P., Nguyen, Q. D., Huynh, H. T., & Bonn, D. "Viscosity bifurcation in thixotropic, yielding fluids."
   *Journal of Rheology*, **46**\ (3), 573-589 (2002).
   https://doi.org/10.1122/1.1459447

.. [3] de Souza Mendes, P. R. "Modeling the thixotropic behavior of structured fluids."
   *Journal of Non-Newtonian Fluid Mechanics*, **164**\ (1-3), 66-75 (2009).
   https://doi.org/10.1016/j.jnnfm.2009.08.005

.. [4] Mewis, J. & Wagner, N. J. "Thixotropy."
   *Advances in Colloid and Interface Science*, **147-148**, 214-227 (2009).
   https://doi.org/10.1016/j.cis.2008.09.005

.. [5] Bocquet, L., Colin, A., & Ajdari, A. "Kinetic theory of plastic flow in soft glassy materials."
   *Physical Review Letters*, **103**, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

Further Reading
~~~~~~~~~~~~~~~

- Mujumdar, A., Beris, A. N., & Metzner, A. B. "Transient phenomena in thixotropic systems."
  *Journal of Non-Newtonian Fluid Mechanics*, **102**\ (2), 157-178 (2002).

- Dullaert, K. & Mewis, J. "A structural kinetics model for thixotropy."
  *Journal of Non-Newtonian Fluid Mechanics*, **139**\ (1-2), 21-30 (2006).

- Moller, P., Fall, A., Chikkadi, V., Derks, D., & Bonn, D. "An attempt to categorize yield stress fluid behaviour."
  *Philosophical Transactions of the Royal Society A*, **367**\ (1909), 5139-5155 (2009).

- Ovarlez, G., Rodts, S., Chateau, X., & Coussot, P. "Phenomenology and physical origin of shear localization and shear banding in complex fluids."
  *Rheologica Acta*, **48**\ (8), 831-844 (2009).
