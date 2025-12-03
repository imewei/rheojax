.. _model-sgr-conventional:

SGR Conventional (Soft Glassy Rheology)
=======================================

Quick Reference
---------------

**Use when:** Soft glassy materials (foams, emulsions, pastes, colloidal gels), yield stress fluids, aging materials
**Parameters:** 3-4 (x, G0, tau0, optional sigma_y)
**Key equation:** :math:`G'(\omega) \sim G''(\omega) \sim \omega^{x-1}` for :math:`1 < x < 2`
**Test modes:** Oscillation, relaxation, creep, steady shear, LAOS
**Material examples:** Concentrated emulsions, colloidal suspensions, foams, pastes, mayonnaise, hair gel

Overview
--------

The Soft Glassy Rheology (SGR) model is a mesoscopic constitutive framework for soft glassy materials—
systems that exhibit structural disorder and metastability similar to glasses but with much
weaker interaction energies (of order :math:`k_B T`). The model unifies the rheological behavior
of diverse complex fluids including foams, emulsions, pastes, slurries, and colloidal glasses
under a single theoretical framework.

The SGR model was developed by Sollich and coworkers [1]_ [2]_ based on Bouchaud's trap model
for structural glasses. It treats the material as an ensemble of mesoscopic "elements"—local
regions of material that can be in various states of local strain, trapped in energy wells
of depth :math:`E`. The key insight is that thermal-like noise (with effective temperature :math:`x`)
activates hopping between traps, while macroscopic strain biases these transitions.

Physical Foundations
--------------------

Mesoscopic Trap Model
~~~~~~~~~~~~~~~~~~~~~

The SGR model describes the material as consisting of many mesoscopic elements, each characterized by:

1. **Local strain** :math:`l` — the strain stored in that element
2. **Trap depth** :math:`E` — the energy barrier to rearrangement (yield energy)
3. **Exponential trap distribution** :math:`\rho(E) = \exp(-E)` for :math:`E > 0`

Elements are trapped in local energy minima until activated by an effective noise process.
The activation rate for yielding follows an Arrhenius-like form:

.. math::

   \Gamma(E, l) = \Gamma_0 \exp\left(-\frac{E - \frac{1}{2}kl^2}{x}\right)

where:
   - :math:`\Gamma_0 = 1/\tau_0` is the attempt frequency
   - :math:`k` is the local elastic constant
   - :math:`x` is the effective noise temperature
   - :math:`\frac{1}{2}kl^2` is the elastic energy stored in the element

The strain lowers the effective barrier height, facilitating yield at high deformations.

Effective Noise Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter :math:`x` is the dimensionless **effective noise temperature**, measuring the ratio
of activation energy to typical trap depths. Unlike thermal temperature, :math:`x` can arise from:

- Mechanical noise from neighboring rearrangements
- Shear-induced fluctuations (shear rejuvenation)
- Slow structural relaxations (aging)
- External perturbations

The glass transition occurs at :math:`x_g = 1`:

.. list-table:: Material phases vs. effective temperature x
   :header-rows: 1
   :widths: 20 30 50

   * - Regime
     - Behavior
     - Physical interpretation
   * - :math:`x < 1`
     - Glass phase, aging, yield stress
     - Trap depths exceed activation energy; material ages
   * - :math:`x = 1`
     - Glass transition point
     - Critical point; power-law rheology with :math:`G' = G''`
   * - :math:`1 < x < 2`
     - Power-law fluid
     - :math:`G' \sim G'' \sim \omega^{x-1}`; viscoelastic liquid
   * - :math:`x \geq 2`
     - Newtonian liquid
     - Exponential relaxation; simple viscous flow

Trap Distribution and Partition Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exponential trap distribution :math:`\rho(E) = e^{-E}` is motivated by entropic arguments:
deep traps are exponentially rare because they require specific configurations of neighbors.

The equilibrium probability of finding an element with trap depth :math:`E` and strain :math:`l` is:

.. math::

   P_{\text{eq}}(E, l) \propto \rho(E) \exp\left(\frac{E - \frac{1}{2}kl^2}{x}\right) \delta(l)

The partition function (integral over all trap depths) diverges for :math:`x \leq 1`, signaling
the glass transition where the system cannot equilibrate on any finite timescale.

Constitutive Equations
----------------------

Linear Response (Oscillatory)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small amplitude oscillatory shear :math:`\gamma(t) = \gamma_0 e^{i\omega t}`, the complex
modulus is given by [2]_:

.. math::

   G^*(\omega) = G_0 \frac{\Gamma(1-x)(i\omega\tau_0)^{x-1}}{1 + \Gamma(1-x)(i\omega\tau_0)^{x-1}}

where :math:`\Gamma(\cdot)` is the gamma function (not the yield rate).

For :math:`1 < x < 2` in the power-law regime, the storage and loss moduli scale as:

.. math::

   G'(\omega) &\sim G_0 (\omega\tau_0)^{x-1} \cos\left(\frac{\pi(x-1)}{2}\right)

   G''(\omega) &\sim G_0 (\omega\tau_0)^{x-1} \sin\left(\frac{\pi(x-1)}{2}\right)

The loss tangent is frequency-independent:

.. math::

   \tan\delta = \tan\left(\frac{\pi(x-1)}{2}\right) = \text{const}

This is the hallmark of power-law rheology—the phase angle :math:`\delta = \pi(x-1)/2` is constant
across frequency, unlike classical viscoelastic models where :math:`\tan\delta` varies.

Stress Relaxation
~~~~~~~~~~~~~~~~~

After a step strain :math:`\gamma_0` at :math:`t = 0`, the stress relaxes as:

.. math::

   \sigma(t) = G_0 \gamma_0 \left(\frac{t}{\tau_0}\right)^{-(x-1)} E_{x-1}\left(-\left(\frac{t}{\tau_0}\right)^{x-1}\right)

where :math:`E_\alpha(z)` is the Mittag-Leffler function. At long times:

.. math::

   G(t) \sim t^{-(x-1)} \quad \text{for } 1 < x < 2

For :math:`x < 1` (glass phase), the relaxation never completes—residual stress persists indefinitely.

Steady Shear Flow
~~~~~~~~~~~~~~~~~

Under steady shear at rate :math:`\dot{\gamma}`, the shear stress is:

.. math::

   \sigma(\dot{\gamma}) = \sigma_y + \eta_\infty \dot{\gamma}^{x-1}

where :math:`\sigma_y` is the yield stress (nonzero for :math:`x < 1`) and :math:`\eta_\infty` is a
high-rate viscosity parameter. This gives the Herschel-Bulkley form with exponent :math:`n = x - 1`.

For :math:`x > 1`, the yield stress vanishes and we have power-law shear thinning.

Creep Compliance
~~~~~~~~~~~~~~~~

Under constant stress :math:`\sigma_0`, the strain evolves as:

.. math::

   \gamma(t) = J_0 \sigma_0 \left[1 + \left(\frac{t}{\tau_0}\right)^{x-1} E_{x-1,x}\left(\left(\frac{t}{\tau_0}\right)^{x-1}\right)\right]

For :math:`1 < x < 2`, this interpolates between initial elastic response and long-time
power-law creep :math:`\gamma \sim t^{x-1}`.

Aging Dynamics
~~~~~~~~~~~~~~

For :math:`x < 1`, the material ages: rheological properties depend on waiting time :math:`t_w`
since preparation. The effective relaxation time grows as:

.. math::

   \tau_{\text{eff}}(t_w) \sim t_w^\mu \quad \text{with } \mu = \frac{1 - x}{x}

Older samples are stiffer and have longer relaxation times. This is captured by making
:math:`x` time-dependent: :math:`x(t_w) \to 1^-` as the system ages toward the glass transition.

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 15 12 12 18 43

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``x``
     - :math:`x`
     - —
     - :math:`0 < x < 3`
     - Effective noise temperature; controls rheological behavior
   * - ``G0``
     - :math:`G_0`
     - Pa
     - :math:`G_0 > 0`
     - Plateau modulus scale
   * - ``tau0``
     - :math:`\tau_0`
     - s
     - :math:`\tau_0 > 0`
     - Microscopic attempt time
   * - ``sigma_y``
     - :math:`\sigma_y`
     - Pa
     - :math:`\sigma_y \geq 0`
     - Yield stress (optional, for :math:`x < 1`)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**x (Effective Noise Temperature)**:
   - **Physical meaning**: Ratio of activation energy to trap depth; controls phase behavior
   - **Typical ranges**:
      - Glasses/gels: :math:`0.3 - 0.9`
      - Near transition: :math:`0.9 - 1.1`
      - Fluids: :math:`1.2 - 1.8`
   - **Connection to rheology**: Power-law exponent :math:`n = x - 1` in :math:`G' \sim \omega^n`

**G0 (Plateau Modulus)**:
   - **Physical meaning**: Characteristic elastic modulus; sets stress scale
   - **Typical ranges**:
      - Concentrated emulsions: :math:`10^1 - 10^3` Pa
      - Pastes: :math:`10^2 - 10^4` Pa
      - Colloidal glasses: :math:`10^0 - 10^2` Pa
   - **Molecular origin**: Interfacial tension (emulsions), entropic forces (colloids)

**tau0 (Attempt Time)**:
   - **Physical meaning**: Microscopic timescale for rearrangement attempts
   - **Typical ranges**: :math:`10^{-6} - 10^{-2}` s
   - **Scaling**: Related to Brownian diffusion time :math:`\tau_0 \sim \eta_s a^3 / k_B T`
     where :math:`a` is the element size

Validity and Assumptions
------------------------

- **Linear viscoelasticity**: Valid for small strains (LAOS extensions available)
- **Mean-field**: Neglects spatial correlations between elements
- **Isothermal**: Temperature enters only through :math:`x`
- **Quasistatic**: Valid when strain rate :math:`\ll 1/\tau_0`
- **Data/test modes**: Oscillation, relaxation, creep, steady shear

Limitations
~~~~~~~~~~~

**Mean-field approximation**:
   Spatial correlations are neglected. In reality, yielding events trigger neighbors
   (avalanches), leading to shear banding and spatiotemporal heterogeneity not captured
   by the basic SGR model.

**Phenomenological noise temperature**:
   The origin of :math:`x` is not derived from first principles. It must be fitted to data
   or estimated from microscopic simulations.

**Single element size**:
   Real soft glasses have polydisperse element sizes, affecting the trap distribution.

**No microstructure evolution**:
   Thixotropy and flow-induced ordering require extended models (see SRFS transform).

Extended SGR Features
---------------------

Large Amplitude Oscillatory Shear (LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SGR model extends naturally to nonlinear oscillatory rheology:

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional(x=1.3, G0=100.0, tau0=0.01)

   # Simulate LAOS response
   laos_result = model.predict_laos(
       omega=1.0,           # Angular frequency (rad/s)
       gamma0=1.0,          # Strain amplitude
       n_cycles=5           # Number of oscillation cycles
   )

   # Access Lissajous-Bowditch curves
   stress, strain = laos_result['stress'], laos_result['strain']

   # Chebyshev decomposition for nonlinear coefficients
   en, vn = laos_result['e_n'], laos_result['v_n']

**Chebyshev Coefficients**:
   The stress response is decomposed as:

   .. math::

      \sigma(\gamma, \dot{\gamma}) = \sum_{n,\text{odd}} e_n T_n(\gamma/\gamma_0) + v_n T_n(\dot{\gamma}/\dot{\gamma}_0)

   where :math:`e_n` quantify elastic nonlinearity and :math:`v_n` viscous nonlinearity.

Thixotropy Extension
~~~~~~~~~~~~~~~~~~~~

The SGR model can be extended with a structural parameter :math:`\lambda \in [0, 1]` representing
the degree of structuring:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{\tau_b} - k_d |\dot{\gamma}|^\alpha \lambda^\beta

where :math:`\tau_b` is the buildup time and :math:`k_d, \alpha, \beta` are destruction parameters.
The noise temperature is modulated: :math:`x_{\text{eff}} = x_0 + \Delta x (1 - \lambda)`.

Shear Banding Detection
~~~~~~~~~~~~~~~~~~~~~~~

Shear banding occurs when the flow curve is non-monotonic. The SGR model can predict
coexistence of bands with different local shear rates:

.. code-block:: python

   from rheojax.transforms import SRFS

   srfs = SRFS()

   # Detect shear banding from flow curve
   is_banding, shear_rates = srfs.detect_shear_banding(
       model,
       gamma_dot_range=(1e-3, 1e2)
   )

   if is_banding:
       # Compute band coexistence parameters
       low_rate, high_rate, lever_rule = srfs.compute_shear_band_coexistence(model)

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From Cole-Cole plot slope**

The power-law exponent :math:`x - 1` can be estimated from the slope of log-log plots:

**Step 1**: Plot :math:`\log G'` vs :math:`\log \omega`

**Step 2**: Fit linear region to get slope :math:`m`

**Step 3**: :math:`x \approx m + 1`

**Method 2: From loss tangent**

.. math::

   x = 1 + \frac{2}{\pi} \arctan(\tan\delta)

If :math:`\tan\delta` is approximately constant across frequency, the material is in the
SGR power-law regime.

**Method 3: From yield stress fitting**

For :math:`x < 1`, fit steady shear data to Herschel-Bulkley:

.. math::

   \sigma = \sigma_y + K \dot{\gamma}^n

The flow index :math:`n \approx x - 1` (for SGR extension to :math:`x < 1`).

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for SGR (3-4 parameters)
   - Start with :math:`x = 1.5` as initial guess for fluids

**Bayesian inference (NUTS)**
   - Highly recommended for SGR to quantify uncertainty in :math:`x`
   - The effective temperature :math:`x` is the critical parameter determining phase behavior
   - Use informative priors: :math:`x \sim \text{Uniform}(0.5, 2.5)` or :math:`\text{Normal}(1.5, 0.5)`

**Bounds**:
   - :math:`x`: [0.3, 2.5] (typical soft glass range)
   - :math:`G_0`: [1e-1, 1e6] Pa (adjust to material)
   - :math:`\tau_0`: [1e-8, 1e0] s

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Diagnostic
     - Solution
   * - :math:`x` stuck at boundary
     - Material not in SGR regime
     - Consider fractional models (FMG, FML)
   * - Poor fit at low :math:`\omega`
     - Terminal behavior differs from SGR
     - Check for aging; use time-dependent :math:`x(t_w)`
   * - :math:`\tan\delta` varies with :math:`\omega`
     - Multiple relaxation mechanisms
     - Use frequency-dependent :math:`x(\omega)` or multi-mode SGR
   * - Fitted :math:`\tau_0 > 1` s
     - Unphysical attempt time
     - Fix :math:`\tau_0` from diffusion estimate, fit :math:`x, G_0` only

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SGRConventional

   # Frequency sweep data
   omega = np.logspace(-2, 2, 50)

   # Create and fit model
   model = SGRConventional()
   model.fit(omega, G_star_data, test_mode='oscillation')

   # Extract parameters
   x = model.parameters.get_value('x')
   G0 = model.parameters.get_value('G0')
   tau0 = model.parameters.get_value('tau0')

   print(f"Effective temperature x = {x:.3f}")
   print(f"Phase: {'glass' if x < 1 else 'fluid'}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.fit(omega, G_star_data, test_mode='oscillation')

   # Bayesian with warm-start
   result = model.fit_bayesian(
       omega, G_star_data,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000
   )

   # Check if x < 1 (glass phase) with credible interval
   intervals = model.get_credible_intervals(result.posterior_samples)
   x_low, x_high = intervals['x']

   if x_high < 1.0:
       print("Material is in glass phase (95% CI)")
   elif x_low > 1.0:
       print("Material is in fluid phase (95% CI)")
   else:
       print("Material is near glass transition")

Multiple Test Modes
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()

   # Fit oscillation data
   model.fit(omega, G_star, test_mode='oscillation')

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 100)
   G_t = model.predict(t, test_mode='relaxation')

   # Predict steady shear flow curve
   gamma_dot = np.logspace(-3, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='steady_shear')

See also
--------

- :doc:`sgr_generic` — GENERIC thermodynamic framework version with entropy production
- :doc:`../fractional/fractional_maxwell_gel` — alternative for power-law gels
- :doc:`../flow/herschel_bulkley` — simpler yield stress model for steady shear only
- :doc:`../../transforms/srfs` — Strain-Rate Frequency Superposition transform

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.SGRConventional`

References
----------

.. [1] Sollich, P., Lequeux, F., Hébraud, P., & Cates, M. E. "Rheology of Soft Glassy Materials."
   *Physical Review Letters*, **78**, 2020-2023 (1997).
   https://doi.org/10.1103/PhysRevLett.78.2020

.. [2] Sollich, P. "Rheological constitutive equation for a model of soft glassy materials."
   *Physical Review E*, **58**, 738-759 (1998).
   https://doi.org/10.1103/PhysRevE.58.738

.. [3] Sollich, P. & Cates, M. E. "Thermodynamic interpretation of soft glassy rheology models."
   *Physical Review E*, **85**, 031127 (2012).
   https://doi.org/10.1103/PhysRevE.85.031127

.. [4] Fielding, S. M., Cates, M. E., & Sollich, P. "Shear banding, aging and noise dynamics in
   soft glassy materials." *Soft Matter*, **5**, 2378-2382 (2009).

.. [5] Hébraud, P. & Lequeux, F. "Mode-Coupling Theory for the Pasty Rheology of Soft Glassy
   Materials." *Physical Review Letters*, **81**, 2934 (1998).
