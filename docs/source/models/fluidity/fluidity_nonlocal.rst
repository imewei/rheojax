.. _model-fluidity-nonlocal:

=================================================================
Fluidity Nonlocal (Coussot-Ovarlez Cooperative Model) — Handbook
=================================================================

Quick Reference
---------------

**Use when:** Shear banding, wall slip, non-homogeneous flows in yield-stress fluids
**Parameters:** 8+ (G, tau_y, eta_bg, f_eq, f_inf, theta, xi, + BCs)
**Key equation:** :math:`\partial_t f = (f_{\rm loc}(\sigma) - f)/\theta + \xi^2 \nabla^2 f`
**Test modes:** Rotation, start-up, creep, oscillation (with spatial profiles)
**Material examples:** Concentrated emulsions, Carbopol gels, colloidal pastes, suspensions in microchannels

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
   * - :math:`f(y, t)`
     - Fluidity field (spatially varying)
   * - :math:`\xi`
     - Cooperativity length (microstructure size)
   * - :math:`\theta`
     - Structural reorganization time
   * - :math:`f_{\rm loc}(\sigma)`
     - Local fluidity function (from HB law)
   * - :math:`\Sigma(t)`
     - Uniform shear stress (momentum balance)
   * - :math:`\dot{\gamma}(y, t)`
     - Local shear rate profile
   * - :math:`\bar{\dot{\gamma}}`
     - Gap-averaged shear rate
   * - :math:`H`
     - Gap width
   * - :math:`\tau_y`
     - Yield stress

Overview
--------

The Nonlocal Fluidity Model extends local constitutive laws for yield-stress fluids to account for **spatial heterogeneity**. The key insight from Goyon, Colin, and Bocquet [1]_ [2]_ is that local plastic rearrangements induce flow in neighboring regions over a characteristic **cooperativity length** :math:`\xi`. This leads to a diffusive coupling between adjacent material elements.

The model resolves fundamental paradoxes in local rheology:

1. **Shear banding**: Smooth transitions between yielded and unyielded regions
2. **Wall slip**: Enhanced fluidity near boundaries
3. **Confinement effects**: Gap-width dependence of flow curves
4. **Finite shear rate in "unyielded" zones**: Due to fluidity diffusion

Historical Context
~~~~~~~~~~~~~~~~~~

The nonlocal fluidity framework emerged from experiments on concentrated emulsions in microfluidic geometries (Goyon et al., 2008) and was formalized theoretically by Bocquet, Colin, and Ajdari (2009). Key contributions:

   - **Goyon et al. (2008)** [1]_: Measured velocity profiles in confined emulsions, found finite shear rates in nominally unyielded regions
   - **Bocquet et al. (2009)** [2]_: Developed kinetic theory linking cooperativity to plastic rearrangements
   - **Coussot, Ovarlez et al. (2009-2012)** [3]_ [4]_: Extended to wide-gap Couette, thixotropy, and transient flows

The model has been validated extensively for:
   - Dense emulsions (droplet diameter :math:`d` → :math:`\xi \sim 3-5d`)
   - Carbopol microgels
   - Colloidal suspensions near jamming

----

Physical Foundations
--------------------

Cooperativity in Soft Glasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In soft glassy materials, plastic deformation occurs through local rearrangements of mesoscopic "elements" (droplets, particles, bubbles). Unlike in crystalline solids, these rearrangements are **cooperative**: a yielding event in one region induces stress perturbations that can trigger rearrangements in neighbors.

The cooperativity length :math:`\xi` characterizes this coupling:

.. math::

   \xi \sim 3-10 \times (\text{particle size})

For example:
   - Emulsions with 10 μm droplets: :math:`\xi \sim 30-100` μm
   - Colloidal glasses with 1 μm particles: :math:`\xi \sim 3-10` μm

When the flow geometry (gap :math:`H`) is comparable to :math:`\xi`, strong confinement effects emerge.

Stress Homogeneity in Planar Shear
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In low Reynolds number flows (creeping limit), momentum balance requires:

.. math::

   \nabla \cdot \boldsymbol{\sigma} = 0

For simple shear in a planar geometry (gap direction :math:`y`):

.. math::

   \frac{\partial \sigma}{\partial y} = 0 \quad \Rightarrow \quad \sigma(y, t) = \Sigma(t) \quad \text{(uniform)}

**The stress is spatially uniform**, even when the material is heterogeneous. Spatial heterogeneity arises only in the shear rate:

.. math::

   \dot{\gamma}(y, t) = f(y, t) \cdot \Sigma(t)

This decoupling is central to the model: the fluidity field :math:`f(y, t)` carries all spatial information.

Local Fluidity Function
~~~~~~~~~~~~~~~~~~~~~~~

The **local fluidity** :math:`f_{\rm loc}(\sigma)` represents what the fluidity would be at stress :math:`\sigma` in a homogeneous, infinitely large sample. It is typically derived from the Herschel-Bulkley law:

.. math::

   f_{\rm loc}(\sigma) = \left(\frac{|\sigma| - \tau_y}{k}\right)^{1/n} \cdot \frac{\Theta(|\sigma| - \tau_y)}{|\sigma|}

For a Bingham fluid (:math:`n = 1`):

.. math::

   f_{\rm loc}(\sigma) = \frac{|\sigma| - \tau_y}{\eta_{\rm bg} \cdot |\sigma|} \cdot \Theta(|\sigma| - \tau_y)

where :math:`\Theta` is the Heaviside function, :math:`\tau_y` is the yield stress, and :math:`\eta_{\rm bg}` is the background (plastic) viscosity.

.. tip:: **Key Physical Intuition**

   Think of :math:`\xi` as the "communication range" of the microstructure:

   - **Small** :math:`\xi` (or large gap :math:`H/\xi \gg 1`): Flow approaches the local (bulk) behavior
   - **Large** :math:`\xi` (or narrow gap :math:`H/\xi \sim 1`): Strong nonlocal effects, homogenization

----

Mathematical Formulation
------------------------

Governing PDE
~~~~~~~~~~~~~

The nonlocal fluidity model consists of a reaction-diffusion equation for :math:`f(y, t)`:

.. math::
   :label: nonlocal-pde

   \frac{\partial f}{\partial t} = \underbrace{\frac{f_{\rm loc}(\Sigma) - f}{\theta}}_{\text{local relaxation}} + \underbrace{\xi^2 \frac{\partial^2 f}{\partial y^2}}_{\text{cooperativity diffusion}}

where:
   - :math:`\theta` is the structural reorganization time (analogous to :math:`\tau_{\rm age}`)
   - :math:`\xi^2` plays the role of a fluidity diffusivity :math:`D_f = \xi^2/\theta`

Strain Rate Field
~~~~~~~~~~~~~~~~~

Given the fluidity profile :math:`f(y, t)` and uniform stress :math:`\Sigma(t)`:

.. math::
   :label: strain-rate-field

   \dot{\gamma}(y, t) = f(y, t) \cdot \Sigma(t)

Global Constraints
~~~~~~~~~~~~~~~~~~

**Rate-controlled** (imposed global shear rate :math:`\bar{\dot{\gamma}}`):

The gap-averaged shear rate must match the imposed value:

.. math::

   \bar{\dot{\gamma}}(t) = \frac{1}{H} \int_0^H \dot{\gamma}(y, t) \, dy = \Sigma(t) \cdot \bar{f}(t)

where :math:`\bar{f}(t) = \frac{1}{H} \int_0^H f(y, t) \, dy`. Thus:

.. math::
   :label: rate-control

   \Sigma(t) = \frac{\bar{\dot{\gamma}}(t)}{\bar{f}(t)}

**Stress-controlled** (imposed stress :math:`\Sigma_0`):

The stress is given directly; the shear rate follows from :math:`\dot{\gamma} = f \Sigma_0`.

Boundary Conditions
~~~~~~~~~~~~~~~~~~~

**Neumann (no-flux):**

.. math::

   \left.\frac{\partial f}{\partial y}\right|_{y=0, H} = 0

Physical meaning: No fluidity transport across walls. Used for idealized smooth boundaries.

**Dirichlet (wall fluidity):**

.. math::

   f(0, t) = f_w, \quad f(H, t) = f_w

Physical meaning: Enhanced fluidity at walls due to boundary slip or wall roughness effects. Captures apparent slip in concentrated systems.

**Mixed conditions** are possible (e.g., Dirichlet at one wall, Neumann at the other).

Steady-State BVP
~~~~~~~~~~~~~~~~

At steady state (:math:`\partial_t f = 0`), the PDE becomes an elliptic boundary value problem:

.. math::
   :label: steady-bvp

   \xi^2 \frac{d^2 f}{d y^2} + f_{\rm loc}(\Sigma) - f = 0

with appropriate BCs. The stress :math:`\Sigma` is determined self-consistently from the rate constraint.

**Solution form** (for Neumann BCs and constant :math:`f_{\rm loc}`):

.. math::

   f(y) = f_{\rm loc} + (f_w - f_{\rm loc}) \cdot \frac{\cosh((y - H/2)/\xi)}{\cosh(H/(2\xi))}

The fluidity profile interpolates between :math:`f_{\rm loc}` in the bulk and :math:`f_w` at walls, with boundary layer thickness :math:`\sim \xi`.

----

Protocol-Specific Equations
---------------------------

Rotation (Steady-State Flow Curve)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:** Constant imposed global shear rate :math:`\bar{\dot{\gamma}}`. Wait for steady state.

**Equations:**

.. math::

   \xi^2 \frac{d^2 f}{d y^2} = f - f_{\rm loc}(\Sigma)

with boundary conditions (e.g., Neumann or Dirichlet).

**Constraint:**

.. math::

   \Sigma = \frac{\bar{\dot{\gamma}}}{\bar{f}}

**Behavior:**
   - For :math:`H/\xi \gg 1`: Approaches local (bulk) flow curve
   - For :math:`H/\xi \sim 1`: Enhanced apparent fluidity, reduced apparent yield stress
   - **Apparent slip**: At low :math:`\bar{\dot{\gamma}}`, fluidity near walls exceeds bulk, creating velocity jumps
   - **No sharp yielding**: Smooth transition from solid-like to flowing

Shear Banding
~~~~~~~~~~~~~

**When does banding occur?**

If the underlying local constitutive curve :math:`\sigma_{\rm loc}(\dot{\gamma})` is **non-monotonic** (e.g., negative slope region), the system can exhibit coexisting bands with different local shear rates.

The nonlocal term regularizes the interface:
   - Interface width :math:`\sim \xi`
   - Unique stress plateau selected by diffusion
   - No discontinuity in velocity (smooth bands)

**Detection:**

.. code-block:: python

   from rheojax.models import FluidityNonlocal

   model = FluidityNonlocal(xi=0.1)
   is_banding, band_profile = model.detect_shear_banding(gamma_dot=0.1)

   if is_banding:
       print("Shear banding detected!")
       print(f"Low-shear band: y < {band_profile['interface_position']:.2f}")

Start-Up Shear
~~~~~~~~~~~~~~

**Protocol:** Apply constant :math:`\bar{\dot{\gamma}} = \dot{\gamma}_0` at :math:`t = 0` from rest.

**PDE:**

.. math::

   \partial_t f = \frac{f_{\rm loc}(\Sigma(t)) - f}{\theta} + \xi^2 \partial_y^2 f

with :math:`\Sigma(t)` determined from the rate constraint at each time step.

**Initial condition:** :math:`f(y, 0) = f_{\rm eq}` (uniform, solid-like)

**Behavior:**
   - Fluidity nucleates at walls (if :math:`f_w > f_{\rm eq}`) or in bulk (from thermal fluctuations)
   - Diffuses inward from boundaries
   - **Stress overshoot** if :math:`\theta` is large
   - Approach to steady heterogeneous profile

Stress Relaxation
~~~~~~~~~~~~~~~~~

**Protocol:** Step strain :math:`\gamma_0` at :math:`t = 0`, then :math:`\bar{\dot{\gamma}} = 0`.

**PDE for** :math:`t > 0`:

.. math::

   \partial_t f = -\frac{f}{\theta} + \xi^2 \partial_y^2 f \quad \text{(assuming } f_{\rm loc}(\sigma < \tau_y) = 0\text{)}

**Behavior:**
   - Fluidity decays (aging)
   - **Non-exponential relaxation** due to diffusion smoothing
   - Stress decay: :math:`\Sigma(t) \sim \exp\left(-\int_0^t \bar{f}(s) \, ds\right)`

.. note::

   Nonlocal fluidity models are primarily designed for driven shear flows.
   For pure relaxation, the homogeneous (local) model is often sufficient
   unless spatial pre-history (e.g., banded initial state) is important.

Creep
~~~~~

**Protocol:** Constant stress :math:`\Sigma_0` applied at :math:`t = 0`.

**PDE:**

.. math::

   \partial_t f = \frac{f_{\rm loc}(\Sigma_0) - f}{\theta} + \xi^2 \partial_y^2 f

**Strain rate:**

.. math::

   \bar{\dot{\gamma}}(t) = \Sigma_0 \cdot \bar{f}(t)

**Behavior:**
   - **Bifurcation**:
      - :math:`\Sigma_0 < \tau_y`: :math:`f \to 0` everywhere (arrest)
      - :math:`\Sigma_0 > \tau_y`: :math:`f \to f_{\rm loc}(\Sigma_0)` (flow)
   - **Delayed fluidization** near :math:`\tau_y`: Long induction time as fluidity diffuses from walls
   - **Spatial heterogeneity**: Walls may yield while center remains solid-like

Oscillatory Shear (SAOS and LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:**

.. math::

   \bar{\dot{\gamma}}(t) = \gamma_0 \omega \cos(\omega t)

**Full time-space PDE** must be integrated numerically.

**SAOS (Small Amplitude):**
   - Fluidity oscillates slightly around mean :math:`\bar{f}`
   - Linear moduli :math:`G^*(\omega)` with diffusive corrections
   - Plateau :math:`G'`, loss :math:`G'' \sim \sqrt{\omega}` at low frequency (diffusion signature)

**LAOS (Large Amplitude):**
   - Strong intracycle :math:`f(y, t)` variations
   - **Intracycle banding**: Center may remain solid while walls yield during part of cycle
   - At high amplitudes: Cooperativity homogenizes flow
   - Higher harmonics in stress response

----

Numerical Implementation
------------------------

Spatial Discretization
~~~~~~~~~~~~~~~~~~~~~~

The 1D PDE is discretized using finite volumes (FVM) or finite differences (FDM) with :math:`N_y` grid points:

.. math::

   y_i = (i - 0.5) \Delta y, \quad \Delta y = H / N_y

**Laplacian with Neumann BCs:**

.. code-block:: python

   def laplacian_neumann(f, dy):
       f_left = jnp.concatenate([f[0:1], f[:-1]])
       f_right = jnp.concatenate([f[1:], f[-1:]])
       return (f_left - 2*f + f_right) / (dy**2)

**Resolution requirement:** :math:`\Delta y < \xi / 3` to resolve boundary layers.

Time Integration
~~~~~~~~~~~~~~~~

The PDE is integrated using `jax.lax.scan` for efficiency:

.. code-block:: python

   @jax.jit
   def step_nonlocal(f, dt, Sigma, params, dy):
       f_target = f_loc_bingham(Sigma, params)
       reaction = (f_target - f) / params.theta
       diffusion = params.xi**2 * laplacian_neumann(f, dy)
       f_new = f + dt * (reaction + diffusion)
       return f_new

**Stability:** For explicit Euler, :math:`\Delta t < \Delta y^2 / (2 \xi^2)`.

Self-Consistent Rate Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For rate-controlled protocols, :math:`\Sigma(t)` must be updated at each step:

.. code-block:: python

   def update_stress(f, gdot_bar, G):
       f_bar = jnp.mean(f)
       return G * gdot_bar / jnp.maximum(f_bar, 1e-12)

----

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 12 12 10 16 50

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``G``
     - :math:`G`
     - Pa
     - :math:`G > 0`
     - Elastic modulus (for viscoelastic extension)
   * - ``tau_y``
     - :math:`\tau_y`
     - Pa
     - :math:`\tau_y \geq 0`
     - Yield stress
   * - ``eta_bg``
     - :math:`\eta_{\rm bg}`
     - Pa·s
     - :math:`\eta_{\rm bg} > 0`
     - Background (plastic) viscosity
   * - ``f_eq``
     - :math:`f_{\rm eq}`
     - s\ :sup:`-1`
     - :math:`f_{\rm eq} \geq 0`
     - Equilibrium fluidity at rest
   * - ``f_inf``
     - :math:`f_\infty`
     - s\ :sup:`-1`
     - :math:`f_\infty > 0`
     - Infinite-shear fluidity (optional, for kinetic model)
   * - ``theta``
     - :math:`\theta`
     - s
     - :math:`\theta > 0`
     - Structural reorganization time
   * - ``xi``
     - :math:`\xi`
     - m
     - :math:`\xi > 0`
     - Cooperativity length; typically 3-10× particle size
   * - ``H``
     - :math:`H`
     - m
     - :math:`H > 0`
     - Gap width (geometry parameter)
   * - ``f_w``
     - :math:`f_w`
     - s\ :sup:`-1`
     - :math:`f_w \geq 0`
     - Wall fluidity (Dirichlet BC); 0 for no-flux

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**xi (Cooperativity Length):**
   - **Physical meaning**: Range of cooperative rearrangements
   - **Estimation**: :math:`\xi \approx 3-5d` where :math:`d` is particle/droplet diameter
   - **Typical values**:
      - Emulsions (10 μm droplets): :math:`\xi \sim 30-50` μm
      - Colloidal glasses: :math:`\xi \sim 1-10` μm
   - **Measurement**: From confinement experiments (flow curve vs. gap width)

**theta (Reorganization Time):**
   - **Physical meaning**: Timescale for local structural equilibration
   - **Relation to aging**: :math:`\theta \sim \tau_{\rm age}` in kinetic models
   - **Typical values**: :math:`0.1 - 100` s

**H (Gap Width) and Confinement Ratio:**
   - Key dimensionless group: :math:`H / \xi`
   - :math:`H/\xi \gg 10`: Bulk behavior
   - :math:`H/\xi \sim 1-10`: Strong nonlocal effects
   - :math:`H/\xi < 1`: Fully cooperative (homogenized)

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **1D planar shear**: Gap direction :math:`y` only (Couette approximation)
2. **Stress homogeneity**: Momentum balance → uniform :math:`\sigma(y) = \Sigma`
3. **Constant cooperativity**: :math:`\xi` is stress-independent (simplification)
4. **Isotropic diffusion**: Same cooperativity in all directions
5. **Quasistatic deformation**: Inertia neglected

Data Requirements
~~~~~~~~~~~~~~~~~

- **Flow curves at multiple gaps**: :math:`\sigma(\bar{\dot{\gamma}})` for :math:`H = H_1, H_2, \ldots`
- **Velocity profiles** (if available): From NMR, PIV, or ultrasound velocimetry
- **Transient data**: Start-up curves, creep tests

Limitations
~~~~~~~~~~~

**Constant ξ assumption:**
   In reality, :math:`\xi` may be stress-dependent [5]_:
   :math:`\xi(\sigma) \sim (|\sigma| - \tau_y)^{-\nu}`
   near the yield point.

**2D/3D geometries:**
   The model extends naturally, but computational cost increases.
   For Couette cells with curvature, stress is not strictly uniform.

**Temperature effects:**
   Thermal fluctuations may modify :math:`\xi` and :math:`\theta`.
   Temperature-dependent parameters are not included.

**Thixotropy:**
   For strongly thixotropic systems, couple to a structural parameter
   :math:`\lambda` (as in the local model).

----

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From confinement experiments**

**Step 1**: Measure flow curves :math:`\sigma(\bar{\dot{\gamma}})` at gaps :math:`H_1, H_2, \ldots`

**Step 2**: Plot apparent yield stress :math:`\sigma_y^{\rm app}(H)` vs. :math:`H`

**Step 3**: Fit to confinement scaling:
   :math:`\sigma_y^{\rm app}(H) = \sigma_y^{\rm bulk} \cdot [1 - c(\xi/H)]`

**Step 4**: Extract :math:`\xi` from the fitting constant :math:`c`

**Method 2: From velocity profiles**

**Step 1**: Measure :math:`v(y)` in wide-gap Couette or channel flow

**Step 2**: Compute shear rate :math:`\dot{\gamma}(y) = dv/dy`

**Step 3**: Invert local constitutive law to get :math:`f(y)` at known stress

**Step 4**: Fit exponential decay of :math:`f(y)` from walls → :math:`\xi`

Optimization Strategy
~~~~~~~~~~~~~~~~~~~~~

**Two-stage fitting:**

1. **Bulk parameters first**: Fit :math:`\tau_y, \eta_{\rm bg}` from wide-gap data
2. **Nonlocal parameters second**: Fit :math:`\xi, \theta, f_w` from narrow-gap or transient data

**Bayesian approach recommended** for :math:`\xi` due to strong correlations with other parameters.

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics
   :header-rows: 1
   :widths: 28 36 36

   * - Problem
     - Diagnostic
     - Solution
   * - Fitted :math:`\xi` too large
     - Comparable to gap :math:`H`
     - Include wider gap data
   * - Poor transient fits
     - Wrong :math:`\theta` or BCs
     - Try Dirichlet BCs with fitted :math:`f_w`
   * - Flow curve mismatch at low :math:`\dot{\gamma}`
     - BC effects dominant
     - Increase :math:`N_y` resolution; check :math:`f_w`
   * - Numerical instability
     - CFL violation
     - Reduce :math:`\Delta t`; use implicit solver
   * - No gap dependence observed
     - :math:`H/\xi \gg 10`
     - Material is in bulk limit; use local model

----

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import FluidityNonlocal

   # Setup geometry
   gap_width = 1e-3  # 1 mm

   # Create model
   model = FluidityNonlocal(
       xi=50e-6,        # 50 micron cooperativity length
       theta=10.0,      # 10 s reorganization time
       tau_y=50.0,      # 50 Pa yield stress
       eta_bg=10.0,     # 10 Pa.s background viscosity
       H=gap_width,
       n_y=100          # Spatial grid points
   )

   # Compute flow curve
   gamma_dot = np.logspace(-3, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='steady_shear')

   # Get velocity profile at specific shear rate
   f_profile, y_grid = model.get_fluidity_profile(gamma_dot=1.0)

Gap-Dependence Study
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityNonlocal
   import numpy as np

   # Fixed bulk parameters
   base_params = dict(xi=50e-6, theta=10.0, tau_y=50.0, eta_bg=10.0)

   # Multiple gap widths
   gaps = [0.1e-3, 0.5e-3, 1e-3, 5e-3]  # 0.1 to 5 mm

   gamma_dot = np.logspace(-3, 2, 50)
   flow_curves = {}

   for H in gaps:
       model = FluidityNonlocal(**base_params, H=H)
       sigma = model.predict(gamma_dot, test_mode='steady_shear')
       flow_curves[H] = sigma
       print(f"Gap {H*1e3:.1f} mm: apparent yield stress = {sigma.min():.1f} Pa")

Transient Start-Up with Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityNonlocal
   import numpy as np

   model = FluidityNonlocal(xi=50e-6, theta=10.0, tau_y=50.0, eta_bg=10.0, H=1e-3)

   # Start-up at constant shear rate
   t = np.linspace(0, 100, 1000)
   gamma_dot_0 = 1.0

   result = model.predict_startup(t, gamma_dot_0, return_profiles=True)

   sigma_t = result['stress']
   f_profiles = result['fluidity_profiles']  # shape (n_t, n_y)
   y_grid = result['y_grid']

   # Plot fluidity evolution across gap
   import matplotlib.pyplot as plt
   for i in [0, 100, 500, -1]:
       plt.plot(y_grid * 1e3, f_profiles[i], label=f't = {t[i]:.1f} s')
   plt.xlabel('Position (mm)')
   plt.ylabel('Fluidity (1/s)')
   plt.legend()

Shear Banding Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FluidityNonlocal

   model = FluidityNonlocal(xi=50e-6, theta=10.0, tau_y=50.0, eta_bg=10.0, H=1e-3)

   # Detect banding at multiple shear rates
   for gdot in [0.01, 0.1, 1.0, 10.0]:
       is_banding, profile = model.detect_shear_banding(gamma_dot=gdot)
       if is_banding:
           print(f"Shear banding at γ̇ = {gdot}: interface at y = {profile['interface']:.3f}")
       else:
           print(f"Homogeneous flow at γ̇ = {gdot}")

----

See Also
--------

- :doc:`fluidity_local` — Homogeneous fluidity model (no spatial effects)
- :doc:`../sgr/sgr_conventional` — Soft Glassy Rheology with aging dynamics
- :doc:`../hl/hebraud_lequeux` — Hébraud-Lequeux diffusion-based model
- :doc:`../flow/herschel_bulkley` — Underlying local constitutive law

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FluidityNonlocal`

----

References
----------

.. [1] Goyon, J., Colin, A., Ovarlez, G., Ajdari, A., & Bocquet, L. "Spatial cooperativity in soft glassy flows."
   *Nature*, **454**, 84-87 (2008).
   https://doi.org/10.1038/nature07026

.. [2] Bocquet, L., Colin, A., & Ajdari, A. "Kinetic theory of plastic flow in soft glassy materials."
   *Physical Review Letters*, **103**, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

.. [3] Goyon, J., Colin, A., & Bocquet, L. "How does a soft glassy material flow: finite size effects, non local rheology, and flow cooperativity."
   *Soft Matter*, **6**, 2668-2678 (2010).
   https://doi.org/10.1039/c001930e

.. [4] Ovarlez, G., Rodts, S., Chateau, X., & Coussot, P. "Phenomenology and physical origin of shear localization and shear banding in complex fluids."
   *Rheologica Acta*, **48**\ (8), 831-844 (2009).
   https://doi.org/10.1007/s00397-008-0344-6

.. [5] Kamani, K., Donley, G. J., & Rogers, S. A. "Unification of the rheological physics of yield stress fluids."
   *Physical Review Letters*, **126**, 218002 (2021).
   https://doi.org/10.1103/PhysRevLett.126.218002

Further Reading
~~~~~~~~~~~~~~~

- Picard, G., Ajdari, A., Lequeux, F., & Bocquet, L. "Elastic consequences of a single plastic event."
  *European Physical Journal E*, **15**\ (4), 371-381 (2004).

- Jop, P., Mansard, V., Chaudhuri, P., Bocquet, L., & Colin, A. "Microscale rheology of a soft glassy material close to yielding."
  *Physical Review Letters*, **108**, 148301 (2012).

- Nicolas, A., Ferrero, E. E., Martens, K., & Barrat, J.-L. "Deformation and flow of amorphous solids: Insights from elastoplastic models."
  *Reviews of Modern Physics*, **90**, 045006 (2018).

- Benzi, R., Divoux, T., Barentin, C., Manneville, S., Aime, S., Cipelletti, L., Rosti, M. E., & Toschi, F.
  "Unified theoretical and experimental view on transient shear banding."
  *Physical Review Letters*, **123**, 248001 (2019).

- Fielding, S. M. "Triggers and signatures of shear banding in steady and time-dependent flows."
  *Journal of Rheology*, **60**\ (5), 821-834 (2016).
