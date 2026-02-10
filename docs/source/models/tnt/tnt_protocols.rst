.. _tnt-protocols:

====================================================
TNT Protocol Equations — Shared Reference
====================================================

Quick Reference
---------------

This page documents the **shared mathematical framework** for all TNT (Transient Network Theory)
models in RheoJAX. The TNT family includes:

- **TNTBase**: Constant breakage rate (upper-convected Maxwell)
- **TNTBell**: Force-dependent breakage (shear thinning)
- **TNTFene**: Finite extensibility via FENE stress function
- **TNTBellFene**: Combined force-dependent breakage and finite extensibility
- **TNTNonAffine**: Gordon-Schowalter derivative (slip parameter)
- **TNTStickyRouse**: Multi-mode Rouse model with sticky segments
- **TNTMultiSpecies**: Polydisperse network with multiple chain lengths

**Common Features:**

- Conformation tensor :math:`\mathbf{S}` tracks chain configuration
- Modulus :math:`G`, breakage time :math:`\tau_b`, solvent viscosity :math:`\eta_s`
- 6 protocols: FLOW_CURVE, SAOS, RELAXATION, STARTUP, CREEP, LAOS
- JAX-native ODE integration via Diffrax

**Key Predictions:**

- Steady shear: Newtonian (constant breakage) or shear thinning (Bell/slip)
- SAOS: Single-mode Maxwell response (linearized)
- Transients: Stress overshoot in startup (Bell), creep compliance, LAOS harmonics

General Constitutive Framework
------------------------------

Conformation Tensor
~~~~~~~~~~~~~~~~~~~

The TNT family is built on the symmetric positive-definite **conformation tensor**
:math:`\mathbf{S}`, which tracks the average second moment of the chain end-to-end
vector :math:`\mathbf{R}` between crosslinks:

.. math::
   \mathbf{S} = \frac{\langle \mathbf{R} \otimes \mathbf{R} \rangle}{\langle R_0^2 \rangle}

At equilibrium (no flow), :math:`\mathbf{S} = \mathbf{I}` (identity tensor), representing
an isotropic Gaussian coil.

Evolution Equation
~~~~~~~~~~~~~~~~~~

The general evolution equation for :math:`\mathbf{S}` is:

.. math::
   \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   - \frac{1}{\tau_b(\mathbf{S})} [\mathbf{S} - \mathbf{S}_{eq}] + \text{(variant-specific terms)}

where:

- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}` is the velocity gradient tensor
- :math:`\tau_b(\mathbf{S})` is the bond/breakage lifetime (may depend on conformation)
- :math:`\mathbf{S}_{eq} = \mathbf{I}` is the equilibrium conformation
- First two terms: **Affine deformation** (upper-convected derivative)
- Third term: **Breakage and reformation** (Brownian relaxation)

**Variant-specific terms:**

- **Bell breakage**: :math:`\tau_b` becomes :math:`\tau_b(\mathbf{S}) = \tau_0 \exp(-\nu \sqrt{\text{tr}(\mathbf{S})})`
- **Non-affine slip**: Replace :math:`\boldsymbol{\kappa}` with Gordon-Schowalter derivative (slip parameter :math:`\xi`)
- **FENE**: Only affects stress function, not evolution

Stress Tensor
~~~~~~~~~~~~~

The Cauchy stress tensor is given by:

.. math::
   \boldsymbol{\sigma} = G \cdot f(\mathbf{S}) + 2\eta_s \mathbf{D}

where:

- :math:`G` is the elastic modulus (plateau modulus)
- :math:`f(\mathbf{S})` is the **stress function**:

  - **Linear (Hookean)**: :math:`f(\mathbf{S}) = \mathbf{S} - \mathbf{I}`
  - **FENE**: :math:`f(\mathbf{S}) = \frac{L_{max}^2}{L_{max}^2 - \text{tr}(\mathbf{S})} (\mathbf{S} - \mathbf{I})`

- :math:`\mathbf{D} = (\nabla \mathbf{v} + \nabla \mathbf{v}^T)/2` is the rate-of-strain tensor
- :math:`\eta_s` is the solvent viscosity

2D Simple Shear Reduction
~~~~~~~~~~~~~~~~~~~~~~~~~~

For **simple shear** flows (most rheological tests), the full 3D tensor :math:`\mathbf{S}`
reduces to 4 independent components due to symmetry:

.. math::
   \mathbf{S} = \begin{bmatrix}
   S_{xx} & S_{xy} & 0 \\
   S_{xy} & S_{yy} & 0 \\
   0 & 0 & S_{zz}
   \end{bmatrix}

The velocity gradient in simple shear is:

.. math::
   \boldsymbol{\kappa} = \begin{bmatrix}
   0 & \dot{\gamma} & 0 \\
   0 & 0 & 0 \\
   0 & 0 & 0
   \end{bmatrix}

This gives the **4-component ODE system** (for constant breakage, upper-convected):

.. math::
   \frac{dS_{xx}}{dt} &= 2\dot{\gamma} S_{xy} - \frac{S_{xx} - 1}{\tau_b} \\
   \frac{dS_{yy}}{dt} &= - \frac{S_{yy} - 1}{\tau_b} \\
   \frac{dS_{zz}}{dt} &= - \frac{S_{zz} - 1}{\tau_b} \\
   \frac{dS_{xy}}{dt} &= \dot{\gamma} S_{yy} - \frac{S_{xy}}{\tau_b}

**Outputs:**

- Shear stress: :math:`\sigma = G \cdot f(S_{xy})_{component} + \eta_s \dot{\gamma}`
- First normal stress difference: :math:`N_1 = G \cdot [f(S_{xx}) - f(S_{yy})]`
- Second normal stress: :math:`N_2 = G \cdot [f(S_{yy}) - f(S_{zz})]` (zero for UCM)

History Integral (Cohort) Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The TNT framework admits an equivalent **integral formulation** that tracks cohorts of
chains born at different times. This perspective is particularly useful for understanding
step-strain responses and complex deformation histories.

**Deformation measures:**

The deformation gradient :math:`\mathbf{F}(t,t')` maps material elements from
configuration at time :math:`t'` to time :math:`t`. The **Finger tensor**
(left Cauchy-Green) is:

.. math::

   \mathbf{B}(t,t') = \mathbf{F}(t,t') \cdot \mathbf{F}^T(t,t')

For simple shear with accumulated strain :math:`\gamma(t,t')`:

.. math::

   \mathbf{B}(t,t') = \begin{bmatrix}
   1 + \gamma^2 & \gamma & 0 \\
   \gamma & 1 & 0 \\
   0 & 0 & 1
   \end{bmatrix}

**Cohort stress superposition:**

Chains born at time :math:`t'` contribute stress proportional to their birth rate
:math:`\beta(t')`, survival probability :math:`S(t,t')`, and accumulated deformation:

.. math::

   \boldsymbol{\tau}(t) = \int_{-\infty}^{t} \beta(t') \, S(t,t') \,
   G \bigl[\mathbf{B}(t,t') - \mathbf{I}\bigr] \, dt'

where:

- :math:`\beta(t')`: Rate of chain creation at time :math:`t'` (at equilibrium,
  :math:`\beta = 1/\tau_b`)
- :math:`S(t,t') = \exp\!\bigl[-\int_{t'}^{t} k_d(s)\,ds\bigr]`: Probability that a
  chain born at :math:`t'` survives to time :math:`t`
- :math:`k_d(s)` is the destruction rate (variant-dependent)

For constant breakage (:math:`k_d = 1/\tau_b`), the survival probability simplifies to:

.. math::

   S(t,t') = \exp\!\bigl[-(t-t')/\tau_b\bigr]

**Shear stress component:**

.. math::

   \tau_{xy}(t) = G \int_{-\infty}^{t} \frac{1}{\tau_b} \exp\!\left[-\int_{t'}^{t}
   k_d(s)\,ds\right] \gamma(t,t') \, dt'

**Equivalence with differential form:**

Differentiating the integral form with respect to time recovers the conformation tensor
ODE. The integral form is the **formal solution** of the differential equation for any
deformation history.

**Generic protocol formula:**

.. math::

   \tau(t) = \int_{-\infty}^{t} \beta(t') \exp\!\left[-\int_{t'}^{t} k_d(s)\,ds\right]
   g\bigl(\gamma(t,t')\bigr) \, dt'

where :math:`g(\gamma)` is the strain measure function (linear for Hookean, nonlinear for
FENE). The specific deformation history :math:`\gamma(t,t')` depends on the protocol:

- **Flow curve**: :math:`\gamma(t,t') = \dot{\gamma}(t - t')`, steady integral
- **Startup**: Integrate from :math:`t' = 0` to :math:`t` (chains born after flow onset)
- **Relaxation**: :math:`\gamma(t,t') = \gamma_0` for :math:`t' < 0` (step strain)
- **Creep**: Solve inverse problem for :math:`\dot{\gamma}(t)` given constant stress
- **SAOS/LAOS**: :math:`\gamma(t,t') = \gamma(t) - \gamma(t')`, Fourier analysis

Cohort Method — Numerical Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The integral (cohort) formulation can be discretized for numerical evaluation, providing
an alternative to ODE integration that is particularly suited for complex histories:

**Algorithm:**

1. Discretize time: :math:`t_0, t_1, \ldots, t_N` with :math:`\Delta t = t_{i+1} - t_i`
2. At each time step :math:`t_n`, maintain cohort weights :math:`w_j` for
   :math:`j = 0, 1, \ldots, n`
3. Update survival: :math:`w_j \leftarrow w_j \cdot \exp(-k_d(t_n) \cdot \Delta t)`
4. Add new cohort: :math:`w_n = \beta(t_n) \cdot \Delta t`
5. Compute stress: :math:`\tau(t_n) = G \sum_{j=0}^{n} w_j \cdot g(\gamma(t_n, t_j))`

**JAX implementation sketch:**

.. code-block:: python

   def cohort_stress(t_eval, gamma_history, params):
       G, tau_b, nu = params['G'], params['tau_b'], params['nu']
       dt = t_eval[1] - t_eval[0]
       n_steps = len(t_eval)

       def scan_fn(weights, i):
           # Decay existing cohorts
           k_d = 1.0 / tau_b  # constant breakage; generalize for Bell
           weights = weights * jnp.exp(-k_d * dt)
           # Add new cohort
           weights = weights.at[i].set((1.0 / tau_b) * dt)
           # Compute strain measures relative to current time
           gamma_rel = gamma_history[i] - gamma_history[:n_steps]
           # Stress superposition
           stress = G * jnp.sum(weights[:i+1] * gamma_rel[:i+1])
           return weights, stress

       weights_init = jnp.zeros(n_steps)
       _, stresses = jax.lax.scan(scan_fn, weights_init, jnp.arange(n_steps))
       return stresses

This cohort approach complements the Diffrax ODE solver. The ODE approach is generally
preferred for smooth deformation histories, while the cohort method excels for
**discontinuous histories** (e.g., sequences of step strains, multi-step creep).

Flow Curve (Steady Shear)
-------------------------

Constant Breakage (TNTBase)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At steady state, :math:`d\mathbf{S}/dt = 0`. For the upper-convected Maxwell model
(constant :math:`\tau_b`), the steady-state solution is:

.. math::
   S_{xy}^{ss} &= \tau_b \dot{\gamma} \\
   S_{xx}^{ss} &= 1 + 2(\tau_b \dot{\gamma})^2 \\
   S_{yy}^{ss} &= S_{zz}^{ss} = 1

**Shear stress** (Newtonian):

.. math::
   \sigma(\dot{\gamma}) = G \cdot S_{xy}^{ss} + \eta_s \dot{\gamma} = (G\tau_b + \eta_s) \dot{\gamma} = \eta_0 \dot{\gamma}

where :math:`\eta_0 = G\tau_b + \eta_s` is the zero-shear viscosity.

**First normal stress difference** (quadratic in :math:`\dot{\gamma}`):

.. math::
   N_1(\dot{\gamma}) = G(S_{xx}^{ss} - S_{yy}^{ss}) = 2G(\tau_b \dot{\gamma})^2

**Second normal stress difference**:

.. math::
   N_2(\dot{\gamma}) = 0

**Key insight**: The base TNT model is **Newtonian in steady shear** but shows
elastic effects (normal stresses, transient overshoots). Shear thinning requires
additional physics.

Bell Breakage (TNTBell)
~~~~~~~~~~~~~~~~~~~~~~~

With force-dependent breakage:

.. math::
   \tau_b(\mathbf{S}) = \tau_0 \exp\left(-\nu \sqrt{\text{tr}(\mathbf{S})}\right)

where :math:`\nu \geq 0` is the force sensitivity parameter.

At steady state, :math:`\tau_b` depends on :math:`\mathbf{S}`, which depends on :math:`\tau_b`.
The steady-state equations become **implicit**:

.. math::
   S_{xy} &= \tau_b(S) \dot{\gamma} \\
   S_{xx} &= 1 + 2[\tau_b(S) \dot{\gamma}]^2 \\
   \tau_b(S) &= \tau_0 \exp\left(-\nu \sqrt{S_{xx} + S_{yy} + S_{zz}}\right)

These are solved via **numerical root-finding** (e.g., fixed-point iteration or Newton's method).

**Shear thinning behavior**:

- Low :math:`\dot{\gamma}`: :math:`\tau_b \approx \tau_0 \exp(-\nu\sqrt{3})` → Newtonian plateau
- High :math:`\dot{\gamma}`: Chains stretch → :math:`\tau_b` decreases → viscosity drops

**Viscosity curve**:

.. math::
   \eta(\dot{\gamma}) = G\tau_b(S) + \eta_s

Power-law exponent :math:`n` in :math:`\eta \sim \dot{\gamma}^{n-1}` depends on :math:`\nu`.

FENE Correction (TNTFene)
~~~~~~~~~~~~~~~~~~~~~~~~~~

The FENE stress function modifies stress but **not** the steady-state :math:`\mathbf{S}`
(evolution unchanged):

.. math::
   f(\mathbf{S}) = \frac{L_{max}^2}{L_{max}^2 - \text{tr}(\mathbf{S})} (\mathbf{S} - \mathbf{I})

At high stretch (:math:`\text{tr}(\mathbf{S}) \to L_{max}^2`), the stress **diverges**,
leading to:

- Stress upturn at high :math:`\dot{\gamma}` (strain hardening)
- Limited extensibility prevents infinite chain stretch

**Combined TNTBellFene**: Bell breakage gives shear thinning at low :math:`\dot{\gamma}`,
FENE gives upturn at high :math:`\dot{\gamma}`.

Non-Affine Slip (TNTNonAffine)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Gordon-Schowalter derivative replaces :math:`\boldsymbol{\kappa}` with:

.. math::
   \boldsymbol{\kappa}_{GS} = \boldsymbol{\kappa} - \xi (\mathbf{D} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{D})

where :math:`0 \leq \xi \leq 1` is the slip parameter:

- :math:`\xi = 0`: Upper-convected (affine)
- :math:`\xi = 1`: Lower-convected (significant slip)

**Effect**: Slip reduces chain stretching → reduces :math:`S_{xx}` → **shear thinning**
in steady state even with constant :math:`\tau_b`.

Multi-Mode Models (StickyRouse, MultiSpecies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For polydisperse or multi-segment networks:

.. math::
   \boldsymbol{\sigma} = \sum_{k=1}^N G_k f_k(\mathbf{S}_k) + 2\eta_s \mathbf{D}

Each mode :math:`k` has its own :math:`(G_k, \tau_k)` and evolves independently.

**Steady-state flow curve** (sum of Newtonian contributions if no Bell/slip):

.. math::
   \sigma(\dot{\gamma}) = \left(\sum_{k=1}^N G_k \tau_k\right) \dot{\gamma} + \eta_s \dot{\gamma}

**Normal stress**:

.. math::
   N_1(\dot{\gamma}) = \sum_{k=1}^N 2G_k (\tau_k \dot{\gamma})^2

Small-Amplitude Oscillatory Shear (SAOS)
----------------------------------------

Linearization
~~~~~~~~~~~~~

For small strain amplitude :math:`\gamma_0 \ll 1`, the TNT models linearize around
:math:`\mathbf{S} = \mathbf{I}`. The linearized evolution is **independent of variant**
(Bell, FENE, slip effects vanish at small amplitude).

All TNT variants reduce to the **single-mode Maxwell model** in SAOS:

.. math::
   \mathbf{S} = \mathbf{I} + \gamma_0 e^{i\omega t} \mathbf{S}_1 + O(\gamma_0^2)

Substituting into the evolution equation and linearizing:

.. math::
   i\omega \mathbf{S}_1 = \boldsymbol{\kappa}_1 - \frac{\mathbf{S}_1}{\tau_b}

Solving for the complex modulus:

Storage and Loss Moduli
~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   G'(\omega) = G \frac{(\omega\tau_b)^2}{1 + (\omega\tau_b)^2}

.. math::
   G''(\omega) = G \frac{\omega\tau_b}{1 + (\omega\tau_b)^2} + \eta_s \omega

where:

- :math:`G'` is the storage modulus (elastic)
- :math:`G''` is the loss modulus (viscous)
- Solvent contributes :math:`\eta_s \omega` to :math:`G''` only

**Complex modulus magnitude**:

.. math::
   |G^*(\omega)| = \sqrt{G'(\omega)^2 + G''(\omega)^2}

**Loss tangent**:

.. math::
   \tan\delta = \frac{G''}{G'}

Limiting Behavior
~~~~~~~~~~~~~~~~~

**Low frequency** (:math:`\omega \tau_b \ll 1`, terminal regime):

.. math::
   G' &\sim G(\omega\tau_b)^2 \propto \omega^2 \\
   G'' &\sim G\omega\tau_b + \eta_s\omega \propto \omega

**High frequency** (:math:`\omega \tau_b \gg 1`, glassy plateau):

.. math::
   G' &\to G \\
   G'' &\sim G/(\omega\tau_b) \propto \omega^{-1}

**Crossover frequency** (:math:`\omega^* = 1/\tau_b`):

.. math::
   G'(\omega^*) = G'' - \eta_s\omega^* = G/2

Multi-Mode SAOS (StickyRouse, MultiSpecies)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`N` modes, the moduli are **additive**:

.. math::
   G'(\omega) = \sum_{k=1}^N G_k \frac{(\omega\tau_k)^2}{1 + (\omega\tau_k)^2}

.. math::
   G''(\omega) = \sum_{k=1}^N G_k \frac{\omega\tau_k}{1 + (\omega\tau_k)^2} + \eta_s \omega

**StickyRouse spectrum** (:math:`N` Rouse modes):

.. math::
   G_k = G/N, \quad \tau_k = \tau_0 / k^2, \quad k = 1, 2, \ldots, N

This gives a **broadened relaxation spectrum** compared to single-mode.

Stress Relaxation
-----------------

Step Strain Protocol
~~~~~~~~~~~~~~~~~~~~

At :math:`t = 0`, a step strain :math:`\gamma_0` is applied, then held constant
(:math:`\dot{\gamma} = 0` for :math:`t > 0`). The stress relaxes due to breakage.

**Initial condition**: :math:`\mathbf{S}(0) = \mathbf{I} + \gamma_0 \mathbf{S}_{simple}`

For small :math:`\gamma_0`, the off-diagonal component is :math:`S_{xy}(0) = \gamma_0`.

Constant Breakage (Analytical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For constant :math:`\tau_b`, the relaxation is **exponential**:

.. math::
   S_{xy}(t) = \gamma_0 e^{-t/\tau_b}

**Relaxation modulus**:

.. math::
   G(t) = \frac{\sigma(t)}{\gamma_0} = G e^{-t/\tau_b}

**Zero-shear viscosity** (integral of :math:`G(t)`):

.. math::
   \eta_0 = \int_0^\infty G(t) dt = G\tau_b

Multi-Mode Relaxation
~~~~~~~~~~~~~~~~~~~~~

For multi-mode models:

.. math::
   G(t) = \gamma_0 \sum_{k=1}^N G_k e^{-t/\tau_k}

This is a **discrete relaxation spectrum**, represented as a **Prony series**.

**Broad spectrum**: StickyRouse with :math:`\tau_k = \tau_0/k^2` gives
:math:`G(t) \sim t^{-1/2}` at intermediate times (Rouse scaling).

Bell Breakage (ODE Solution)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TNTBell, :math:`\tau_b` depends on the **current** conformation :math:`\mathbf{S}(t)`,
which evolves during relaxation:

.. math::
   \tau_b(t) = \tau_0 \exp\left(-\nu \sqrt{\text{tr}(\mathbf{S}(t))}\right)

The relaxation is **non-exponential** and requires solving the ODE:

.. math::
   \frac{dS_{xy}}{dt} = -\frac{S_{xy}}{\tau_b(S)}

**Numerical solution**: Use Diffrax with adaptive stepping.

**Effect of :math:`\nu`**: Higher force sensitivity → faster relaxation at early times
(chains stretched, :math:`\tau_b` small), slower at late times (approaching equilibrium).

FENE Relaxation
~~~~~~~~~~~~~~~

FENE stress function gives:

.. math::
   \sigma(t) = G \frac{L_{max}^2}{L_{max}^2 - \text{tr}(\mathbf{S}(t))} S_{xy}(t)

Even if :math:`S_{xy}(t)` relaxes exponentially, the **stress** relaxes faster due to
the FENE prefactor decreasing with :math:`\text{tr}(\mathbf{S})`.

Startup Flow
------------

Protocol Description
~~~~~~~~~~~~~~~~~~~~

Starting from equilibrium (:math:`\mathbf{S} = \mathbf{I}`, no stress), a constant
shear rate :math:`\dot{\gamma}` is applied at :math:`t = 0`.

The **4-component ODE** is solved forward in time:

.. math::
   \frac{d\mathbf{S}}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \boldsymbol{\kappa}^T
   - \frac{\mathbf{S} - \mathbf{I}}{\tau_b(\mathbf{S})}

The shear stress :math:`\sigma(t)` is computed at each timestep.

Constant Breakage (Analytical)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TNTBase (constant :math:`\tau_b`), the startup stress has an analytical solution:

.. math::
   \sigma(t) = (G\tau_b + \eta_s) \dot{\gamma} \left[1 - e^{-t/\tau_b}\right] + \eta_s \dot{\gamma}

Simplifying:

.. math::
   \sigma(t) = \eta_0 \dot{\gamma} \left[1 - e^{-t/\tau_b}\right] + \eta_s \dot{\gamma}

**Limiting behavior**:

- :math:`t \ll \tau_b`: :math:`\sigma \approx (\eta_0 + \eta_s) \dot{\gamma} t` (linear growth, elastic)
- :math:`t \gg \tau_b`: :math:`\sigma \to \eta_0 \dot{\gamma}` (steady state)

**No overshoot** for constant breakage (monotonic approach to steady state).

Bell Breakage (Stress Overshoot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TNTBell, the stress typically shows an **overshoot** before settling to steady state:

1. **Initial elastic response**: :math:`\tau_b \approx \tau_0` (unperturbed), stress builds rapidly
2. **Peak stress**: Chains stretch → :math:`\tau_b` decreases → stress growth slows
3. **Overshoot**: Maximum stress at strain :math:`\gamma_{peak} = \dot{\gamma} t_{peak}` (typically :math:`\gamma_{peak} \sim 1-3`)
4. **Relaxation to steady state**: Stress decreases as breakage accelerates

**Overshoot characteristics**:

- **Peak strain**: Decreases with :math:`\nu` (higher force sensitivity → earlier peak)
- **Overshoot amplitude**: Increases with :math:`Wi = \tau_b \dot{\gamma}` (Weissenberg number)
- **Shear thinning**: Steady-state viscosity lower than initial

**Damping function**:

.. math::
   \eta^+(t, \dot{\gamma}) = \frac{\sigma(t)}{\dot{\gamma}}

This is the **transient viscosity**, showing overshoot and relaxation.

FENE Effect (Strain Hardening)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FENE stress function causes **stress upturn** at large strain:

.. math::
   \sigma(t) = G \frac{L_{max}^2}{L_{max}^2 - \text{tr}(\mathbf{S}(t))} S_{xy}(t) + \eta_s \dot{\gamma}

As :math:`\text{tr}(\mathbf{S}) \to L_{max}^2`, the prefactor diverges → **finite-time blowup**
if strain continues unbounded.

**Practical limit**: FENE prevents infinite stretch, but very large :math:`\dot{\gamma}`
can cause numerical stiffness.

Multi-Mode Startup
~~~~~~~~~~~~~~~~~~

For StickyRouse or MultiSpecies:

.. math::
   \sigma(t) = \sum_{k=1}^N \sigma_k(t)

Each mode evolves independently (no coupling). Fast modes (:math:`\tau_k` small) reach
steady state quickly, slow modes continue building stress.

**Broadened overshoot**: Multiple peaks or shoulder in :math:`\sigma(t)` if modes have
disparate timescales.

Creep
-----

Protocol Description
~~~~~~~~~~~~~~~~~~~~

A constant shear stress :math:`\sigma_0` is applied at :math:`t = 0`, and the resulting
strain :math:`\gamma(t)` is measured.

**Governing equations**: The strain :math:`\gamma` becomes the **5th state variable**
(appended to :math:`\mathbf{S}`). The constraint is:

.. math::
   \sigma_0 = G \cdot f(\mathbf{S}) \cdot S_{xy} + \eta_s \frac{d\gamma}{dt}

Solving for the shear rate:

.. math::
   \frac{d\gamma}{dt} = \frac{\sigma_0 - G \cdot S_{xy}}{\eta_s}

**Requirements**:

- :math:`\eta_s > 0` (non-zero solvent viscosity) for well-posed ODE
- Initial condition: :math:`\mathbf{S}(0) = \mathbf{I}`, :math:`\gamma(0) = 0`

Creep Compliance
~~~~~~~~~~~~~~~~

The **creep compliance** is:

.. math::
   J(t) = \frac{\gamma(t)}{\sigma_0}

For a Maxwell model (constant :math:`\tau_b`):

.. math::
   J(t) = \frac{1}{G} \left(1 - e^{-t/\tau_b}\right) + \frac{t}{\eta_0}

**Two regimes**:

1. **Transient creep**: :math:`J(t) \sim t/G` (elastic recovery)
2. **Steady-state creep**: :math:`J(t) \sim t/\eta_0` (viscous flow)

The **steady-state creep rate** is:

.. math::
   \dot{\gamma}_{ss} = \frac{\sigma_0}{\eta_0}

Numerical Solution
~~~~~~~~~~~~~~~~~~

For Bell or FENE variants, the creep compliance is **non-linear** and requires solving
the **5-component ODE system**:

.. math::
   \frac{d\mathbf{S}}{dt} &= \dot{\gamma}(t) \boldsymbol{\kappa} \cdot \mathbf{S}
   + \mathbf{S} \cdot \boldsymbol{\kappa}^T - \frac{\mathbf{S} - \mathbf{I}}{\tau_b(\mathbf{S})} \\
   \frac{d\gamma}{dt} &= \frac{\sigma_0 - G f(S_{xy})}{\eta_s}

**Numerical challenges**:

- **Stiffness**: If :math:`\eta_s \ll G\tau_b`, the ODE is stiff → use implicit solver
  (Kvaerno3 or Kvaerno5 in Diffrax)
- **Steady-state detection**: Stop integration when :math:`|\dot{\gamma}(t) - \dot{\gamma}_{ss}| < \epsilon`

Yielding and Viscosity Bifurcation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For TNTBell, if :math:`\sigma_0` is below a **critical stress** :math:`\sigma_c`, the
creep rate may be very slow (yielding behavior).

**Effective yield stress** (approximate):

.. math::
   \sigma_c \approx G \cdot \exp(-\nu \sqrt{3})

For :math:`\sigma_0 < \sigma_c`, the material creeps very slowly (glassy regime).
For :math:`\sigma_0 > \sigma_c`, it flows readily.

Large-Amplitude Oscillatory Shear (LAOS)
-----------------------------------------

Protocol Description
~~~~~~~~~~~~~~~~~~~~

The applied strain is oscillatory:

.. math::
   \gamma(t) = \gamma_0 \sin(\omega t)

The shear rate is:

.. math::
   \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

This is substituted into the velocity gradient :math:`\boldsymbol{\kappa}_{xy} = \dot{\gamma}(t)`,
and the **4-component ODE** is solved over multiple cycles.

Periodic Steady State
~~~~~~~~~~~~~~~~~~~~~

The transient response (first 2-3 cycles) is discarded. The system reaches a **periodic
steady state** after ~5-10 cycles for most TNT variants.

**Stress response** in periodic steady state:

.. math::
   \sigma(t) = \sigma(t + T)

where :math:`T = 2\pi/\omega` is the period.

Nonlinear Response
~~~~~~~~~~~~~~~~~~

For small :math:`\gamma_0`, the response is linear (SAOS). For large :math:`\gamma_0`,
the stress waveform becomes **distorted** (non-sinusoidal).

**Fourier decomposition**:

.. math::
   \sigma(t) = \sum_{n=1,3,5,\ldots} [\sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t)]

where:

- :math:`n = 1`: Fundamental (linear response)
- :math:`n = 3, 5, \ldots`: Higher harmonics (nonlinearity)

**Nonlinear moduli** (Cho et al. 2005):

.. math::
   G'_1 = \frac{\sigma'_1}{\gamma_0}, \quad G''_1 = \frac{\sigma''_1}{\gamma_0}

**Third harmonic intensity**:

.. math::
   I_3 = \sqrt{(G'_3)^2 + (G''_3)^2}

Large :math:`I_3` indicates strong nonlinearity.

Lissajous Curves
~~~~~~~~~~~~~~~~

The **Lissajous curve** plots :math:`\sigma` vs :math:`\gamma` (elastic Lissajous)
or :math:`\sigma` vs :math:`\dot{\gamma}` (viscous Lissajous).

**Linear response**: Perfect ellipse

**Nonlinear response**: Distorted loop (rectangular, S-shaped, etc.)

Pipkin Diagram
~~~~~~~~~~~~~~

LAOS behavior is often summarized in a **Pipkin diagram** (:math:`\gamma_0` vs :math:`\omega`):

- **Linear regime**: :math:`\gamma_0 < 0.1`, :math:`I_3/I_1 < 0.01`
- **Weak nonlinearity**: :math:`0.1 < \gamma_0 < 1`, higher harmonics emerge
- **Strong nonlinearity**: :math:`\gamma_0 > 1`, cage-breaking, yielding

Bell Breakage in LAOS
~~~~~~~~~~~~~~~~~~~~~

For TNTBell, LAOS at large :math:`\gamma_0` shows:

- **Strain thinning**: :math:`G'_1(\gamma_0)` decreases with :math:`\gamma_0`
- **Intra-cycle softening**: :math:`\tau_b` decreases during high-strain portions of cycle
- **Strong 3rd harmonic**: Asymmetric stress response

FENE in LAOS
~~~~~~~~~~~~

FENE stress function gives:

- **Strain hardening**: :math:`G'_1` increases at very large :math:`\gamma_0` (approaching :math:`L_{max}`)
- **Intracycle stiffening**: Chains near maximum extension

Integration Strategy
~~~~~~~~~~~~~~~~~~~~~

**Diffrax settings**:

- Integrate over :math:`10-20` cycles
- Discard first :math:`2-3` cycles (transient)
- Extract last :math:`3-5` cycles for analysis
- Use adaptive stepping (Tsit5 or Dopri5)
- :math:`\text{rtol} = 10^{-6}`, :math:`\text{atol} = 10^{-8}`

**Harmonic extraction**: Apply FFT or trapezoidal integration for Fourier coefficients.

Numerical Methods
-----------------

ODE Integration with Diffrax
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All transient protocols (STARTUP, CREEP, RELAXATION, LAOS) use **Diffrax**, a JAX-native
ODE solver library.

**General structure**:

.. code-block:: python

   from diffrax import diffeqsolve, Tsit5, ODETerm, SaveAt

   def ode_fn(t, state, args):
       S, gamma = state  # unpack state
       S_dot = compute_dS_dt(S, gamma_dot, args)
       gamma_dot = compute_dgamma_dt(S, sigma_0, args)
       return jnp.concatenate([S_dot, gamma_dot])

   solution = diffeqsolve(
       ODETerm(ode_fn),
       solver=Tsit5(),
       t0=0.0, t1=t_end, dt0=0.01,
       y0=initial_state,
       saveat=SaveAt(ts=t_eval),
       args=params,
       rtol=1e-6, atol=1e-8
   )

**Solver choices**:

- **Tsit5**: 5th-order Runge-Kutta (explicit, general-purpose)
- **Dopri5**: Dormand-Prince (similar to Tsit5, MATLAB's ode45)
- **Kvaerno3/5**: Implicit, for stiff problems (creep with small :math:`\eta_s`)

Adaptive Timestepping
~~~~~~~~~~~~~~~~~~~~~~

Diffrax automatically adjusts :math:`dt` based on error estimates:

- **rtol**: Relative tolerance (default :math:`10^{-6}`)
- **atol**: Absolute tolerance (default :math:`10^{-8}`)

**Dense output**: Use :math:`\texttt{SaveAt(ts=t_eval)}` to evaluate at specific times
without interpolation error.

Multi-Mode Parallelization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For StickyRouse and MultiSpecies, each mode :math:`k` evolves **independently**. Use
:math:`\texttt{jax.vmap}` to solve all modes in parallel:

.. code-block:: python

   # Vectorized over N modes
   def solve_mode(G_k, tau_k):
       return diffeqsolve(...)

   results = jax.vmap(solve_mode)(G_modes, tau_modes)

**GPU acceleration**: All :math:`N` modes solved simultaneously on GPU (massive speedup
for :math:`N > 10`).

Stiffness and Implicit Solvers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Stiff problems** occur when:

- :math:`\eta_s \ll G\tau_b` (creep)
- Bell breakage with large :math:`\nu` (rapid timescale changes)
- FENE near :math:`L_{max}` (diverging stress)

**Solution**: Use Kvaerno3 or Kvaerno5 (L-stable implicit methods):

.. code-block:: python

   from diffrax import Kvaerno5

   solution = diffeqsolve(
       ODETerm(ode_fn),
       solver=Kvaerno5(),
       ...
   )

**Cost**: Implicit solvers require Jacobian evaluations → slower per step, but take
larger stable steps.

Precompilation and JIT
~~~~~~~~~~~~~~~~~~~~~~

All ODE functions are **JIT-compiled** via :math:`\texttt{@jax.jit}`:

.. code-block:: python

   @jax.jit
   def solve_startup(gamma_dot, params):
       return diffeqsolve(...)

**First call**: Compilation overhead (~10-60 seconds)

**Subsequent calls**: Near-instant execution

**Memory**: Compiled functions cached by JAX (beware of memory growth with many parameter sets).

Numerical Stability
~~~~~~~~~~~~~~~~~~~

**Common issues**:

1. **FENE divergence**: Clip :math:`\text{tr}(\mathbf{S})` to :math:`0.99 L_{max}^2` to prevent division by zero
2. **Negative eigenvalues**: :math:`\mathbf{S}` should remain positive-definite; project onto
   SPD cone if needed
3. **Overflow**: Use :math:`\texttt{jnp.clip}` on :math:`\tau_b` for Bell breakage with large :math:`\nu`

**Validation**:

- Check mass balance: :math:`\text{tr}(\mathbf{S}) > 0`
- Check symmetry: :math:`S_{xy} = S_{yx}` (automatic in 2D reduction)
- Compare to analytical solutions (constant breakage, SAOS)

Variant × Protocol Effect Matrix
---------------------------------

Summary of the **dominant physical effect** each variant introduces for each protocol.
All variants reduce to the base Tanaka-Edwards (constant breakage) prediction in the
linear limit (:math:`\gamma_0 \ll 1`).

.. list-table::
   :widths: 14 11 11 11 11 11 11 11 11
   :header-rows: 1

   * - Protocol
     - Constant
     - Bell
     - FENE
     - NonAffine
     - StretchCreate
     - LoopBridge
     - Cates
     - MultiSpecies
   * - Flow curve
     - Newtonian
     - Thinning
     - Saturation
     - :math:`N_2 \neq 0`
     - Thickening
     - :math:`f_B`-dependent
     - Non-monotonic
     - Multi-rate
   * - SAOS
     - Maxwell
     - Maxwell
     - Maxwell
     - Maxwell
     - Maxwell
     - Reduced :math:`G`
     - Cole-Cole
     - Multi-mode
   * - Startup
     - Monotonic
     - Overshoot
     - Stiffening
     - Overshoot
     - Thickening
     - Two timescales
     - Large overshoot
     - Staged
   * - Relaxation
     - Exponential
     - Strain-dep :math:`\tau`
     - :math:`f`-dep decay
     - Faster decay
     - Slow (hardening)
     - Bridge recovery
     - Stretched exp
     - Multi-exp
   * - Creep
     - Viscous flow
     - Yielding
     - Saturation
     - Faster flow
     - Ringing
     - :math:`f_B` collapse
     - Banding
     - Staged flow
   * - LAOS
     - Sinusoidal
     - Odd harmonics
     - Box Lissajous
     - :math:`N_2` signal
     - Hardening
     - Asymmetric
     - Plateau
     - Multi-timescale

See :doc:`tnt_knowledge_extraction` for guidance on using these signatures to identify
the appropriate variant from experimental data.

See Also
--------

- :ref:`model-tnt-tanaka-edwards` - Constant breakage rate (upper-convected Maxwell)
- :ref:`model-tnt-bell` - Force-dependent breakage (shear thinning)
- :ref:`model-tnt-fene-p` - Finite extensibility (strain hardening)
- :ref:`model-tnt-stretch-creation` - Combined Bell and FENE
- :ref:`model-tnt-non-affine` - Gordon-Schowalter slip
- :ref:`model-tnt-sticky-rouse` - Multi-mode Rouse model
- :ref:`model-tnt-multi-species` - Polydisperse network

**External References**:

- Tanaka & Edwards (1992): Original TNT formulation
- Inkson et al. (1999): Bell breakage kinetics
- Bird et al. (1987): FENE stress function
- Cho et al. (2005): LAOS nonlinear analysis
- RheoJAX Documentation: :doc:`../bayesian_inference`, :doc:`../optimization`

References
----------

1. **Tanaka, F., & Edwards, S. F.** (1992). Viscoelastic properties of physically crosslinked
   networks: Transient network theory. *Macromolecules*, 25(5), 1516-1523.

2. **Inkson, N. J., McLeish, T. C. B., Harlen, O. G., & Groves, D. J.** (1999). Predicting
   low density polyethylene melt rheology in elongational and shear flows with "pom-pom"
   constitutive equations. *Journal of Rheology*, 43(4), 873-896.

3. **Bird, R. B., Curtiss, C. F., Armstrong, R. C., & Hassager, O.** (1987). *Dynamics of
   Polymeric Liquids, Volume 2: Kinetic Theory* (2nd ed.). Wiley.

4. **Phan-Thien, N., & Tanner, R. I.** (1977). A new constitutive equation derived from
   network theory. *Journal of Non-Newtonian Fluid Mechanics*, 2(4), 353-365.

5. **Cho, K. S., Hyun, K., Ahn, K. H., & Lee, S. J.** (2005). A geometrical interpretation
   of large amplitude oscillatory shear response. *Journal of Rheology*, 49(3), 747-758.

6. **Ewoldt, R. H., Hosoi, A. E., & McKinley, G. H.** (2008). New measures for characterizing
   nonlinear viscoelasticity in large amplitude oscillatory shear. *Journal of Rheology*,
   52(6), 1427-1458.

7. **Doi, M., & Edwards, S. F.** (1986). *The Theory of Polymer Dynamics*. Oxford University
   Press.

8. **Rubinstein, M., & Colby, R. H.** (2003). *Polymer Physics*. Oxford University Press.

9. **Larson, R. G.** (1999). *The Structure and Rheology of Complex Fluids*. Oxford University
   Press.

10. **Macosko, C. W.** (1994). *Rheology: Principles, Measurements, and Applications*.
    Wiley-VCH.
