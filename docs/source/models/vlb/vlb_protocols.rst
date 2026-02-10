.. _vlb_protocols:

======================================
VLB Protocol Equations & Derivations
======================================

This page provides detailed step-by-step derivations of the VLB model
predictions for each rheological protocol, including the multi-network
variants.  All results assume constant dissociation rate :math:`k_d`.

For the governing equations and notation, see :doc:`vlb`.


Simple Shear Geometry
=====================

In simple shear the velocity field is :math:`v_x = \dot{\gamma} y`,
:math:`v_y = v_z = 0`.  The velocity gradient tensor is:

.. math::

   \mathbf{L} = \begin{pmatrix} 0 & \dot{\gamma} & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}

The distribution tensor :math:`\boldsymbol{\mu}` is symmetric, so we track
four independent components: :math:`\mu_{xx}, \mu_{yy}, \mu_{zz}, \mu_{xy}`.

**Component-wise ODE system:**

.. math::

   \dot{\mu}_{xx} &= k_d(1 - \mu_{xx}) + 2\dot{\gamma}\mu_{xy} \\
   \dot{\mu}_{yy} &= k_d(1 - \mu_{yy}) \\
   \dot{\mu}_{zz} &= k_d(1 - \mu_{zz}) \\
   \dot{\mu}_{xy} &= -k_d \mu_{xy} + \dot{\gamma}\mu_{yy}

**Observations:**

- :math:`\mu_{yy}` and :math:`\mu_{zz}` decouple from the shear components
- Starting from :math:`\boldsymbol{\mu}(0) = \mathbf{I}`, we have
  :math:`\mu_{yy}(t) = \mu_{zz}(t) = 1` for all :math:`t`
- :math:`\mu_{xy}` is driven by :math:`\dot{\gamma}\mu_{yy}`
- :math:`\mu_{xx}` is driven by :math:`2\dot{\gamma}\mu_{xy}`


.. _vlb-flow-curve:

Flow Curve Derivation
=====================

**Objective:** Find steady-state :math:`\sigma_{12}(\dot{\gamma})` and
:math:`N_1(\dot{\gamma})`.

**Step 1:** Set :math:`\dot{\boldsymbol{\mu}} = 0`:

.. math::

   0 &= k_d(1 - \mu_{xx}^{ss}) + 2\dot{\gamma}\mu_{xy}^{ss} \\
   0 &= -k_d \mu_{xy}^{ss} + \dot{\gamma}

**Step 2:** Solve for :math:`\mu_{xy}^{ss}` from the second equation:

.. math::

   \mu_{xy}^{ss} = \frac{\dot{\gamma}}{k_d}

**Step 3:** Substitute into the first equation:

.. math::

   \mu_{xx}^{ss} = 1 + \frac{2\dot{\gamma}^2}{k_d^2}

**Step 4:** Since :math:`\mu_{yy}^{ss} = 1`:

.. math::

   \sigma_{12} &= G_0 \mu_{xy}^{ss} = G_0 \frac{\dot{\gamma}}{k_d} = \eta_0 \dot{\gamma} \\
   N_1 &= G_0(\mu_{xx}^{ss} - \mu_{yy}^{ss}) = \frac{2G_0\dot{\gamma}^2}{k_d^2}

**Verification conditions:**

- :math:`\sigma_{12}` is linear in :math:`\dot{\gamma}` (Newtonian)
- :math:`N_1 \propto \dot{\gamma}^2` (quadratic, typical of Maxwell-type models)
- :math:`\eta_0 = G_0/k_d` (Green-Kubo relation)
- :math:`\Psi_1 = N_1/\dot{\gamma}^2 = 2G_0/k_d^2 = 2\eta_0 t_R` (first normal stress coefficient)


.. _vlb-startup:

Startup Shear Derivation
=========================

**Objective:** Find :math:`\sigma_{12}(t)` and :math:`N_1(t)` for constant
:math:`\dot{\gamma}` starting from equilibrium.

**Initial conditions:** :math:`\mu_{xy}(0) = 0`, :math:`\mu_{xx}(0) = 1`,
:math:`\mu_{yy}(0) = 1`.

**Step 1:** Solve the :math:`\mu_{xy}` ODE (linear first-order with constant
coefficients, :math:`\mu_{yy} = 1`):

.. math::

   \dot{\mu}_{xy} + k_d \mu_{xy} = \dot{\gamma}

Solution:

.. math::

   \mu_{xy}(t) = \frac{\dot{\gamma}}{k_d}\left(1 - e^{-k_d t}\right)

Hence:

.. math::

   \sigma_{12}(t) = \frac{G_0\dot{\gamma}}{k_d}\left(1 - e^{-k_d t}\right)

**Step 2:** Solve the :math:`\mu_{xx}` ODE with the known :math:`\mu_{xy}(t)`:

.. math::

   \dot{\mu}_{xx} + k_d \mu_{xx} = k_d + \frac{2\dot{\gamma}^2}{k_d}\left(1 - e^{-k_d t}\right)

This is a linear ODE whose solution gives:

.. math::

   N_1(t) = G_0(\mu_{xx}(t) - 1) = \frac{2G_0\dot{\gamma}^2}{k_d^2}
   \left(1 - e^{-k_d t}\right) - \frac{2G_0\dot{\gamma}^2}{k_d} t \, e^{-k_d t}

**Verification conditions:**

- :math:`\sigma_{12}(0) = 0` (starts from rest)
- :math:`\sigma_{12}(\infty) = G_0\dot{\gamma}/k_d = \eta_0\dot{\gamma}` (steady state)
- :math:`\sigma_{12}(t)` is monotonically increasing (no overshoot for constant :math:`k_d`)
- Time to reach 63% of steady state: :math:`t_{63\%} = 1/k_d = t_R`
- :math:`N_1(t)` can be non-monotonic for :math:`\text{Wi} = \dot{\gamma}/k_d > 1`
  (the :math:`t \, e^{-k_d t}` term creates a transient undershoot)


.. _vlb-relaxation:

Stress Relaxation Derivation
=============================

**Objective:** Find :math:`G(t)` after step strain :math:`\gamma_0`.

**Approach:** At :math:`t = 0^+`, the material has been instantaneously
strained.  For :math:`t > 0`, :math:`\dot{\gamma} = 0`.

**Step 1:** With :math:`\dot{\gamma} = 0`, the ODE simplifies:

.. math::

   \dot{\mu}_{xy} = -k_d \mu_{xy}

Solution from initial condition :math:`\mu_{xy}(0^+) = \gamma_0`
(affine deformation in the linear regime):

.. math::

   \mu_{xy}(t) = \gamma_0 \, e^{-k_d t}

**Step 2:** The relaxation modulus:

.. math::

   G(t) = \frac{\sigma_{12}(t)}{\gamma_0} = \frac{G_0 \mu_{xy}(t)}{\gamma_0} = G_0 \, e^{-k_d t}

**Verification conditions:**

- :math:`G(0) = G_0` (instantaneous modulus)
- :math:`G(\infty) = 0` (liquid, complete relaxation)
- :math:`\ln G(t)` is linear with slope :math:`-k_d`
- Area under :math:`G(t)`: :math:`\int_0^\infty G(t)\,dt = G_0/k_d = \eta_0`

**Multi-network:**

.. math::

   G(t) = G_e + \sum_{I=0}^{M-1} G_I \, e^{-k_{d,I} t}


.. _vlb-creep:
.. _vlb-creep-multi:

Creep Derivation
================

**Objective:** Find :math:`J(t) = \gamma(t)/\sigma_0` under constant stress
:math:`\sigma_0`.

**Single network (Maxwell creep):**

Since the VLB single network with constant :math:`k_d` is exactly Maxwell:

.. math::

   J(t) = \frac{1}{G_0} + \frac{t}{\eta_0} = \frac{1 + k_d t}{G_0}

This can also be derived from the stress constraint
:math:`\sigma_0 = G_0 \mu_{xy}`, giving :math:`\mu_{xy} = \sigma_0/G_0`,
and integrating :math:`\dot{\gamma} = k_d \mu_{xy}` to obtain
:math:`\gamma(t) = \sigma_0/G_0 + (\sigma_0/\eta_0)t`.

**Verification conditions:**

- :math:`J(0) = 1/G_0` (instantaneous elastic compliance)
- :math:`dJ/dt = 1/\eta_0 = k_d/G_0` (constant viscous flow rate)
- :math:`J(t)` is linear — hallmark of Maxwell creep

**Dual-network (1 transient + permanent) — Standard Linear Solid creep:**

.. math::

   J(t) = \frac{1}{G_0 + G_e} + \frac{G_0}{G_e(G_0 + G_e)}
   \left(1 - e^{-t/\tau_{ret}}\right)

where the retardation time is:

.. math::

   \tau_{ret} = \frac{G_0 + G_e}{G_e \cdot k_d}

**Verification conditions:**

- :math:`J(0) = 1/(G_0 + G_e)` (unrelaxed compliance)
- :math:`J(\infty) = 1/G_e` (relaxed compliance)
- Bounded creep (solid-like behavior due to :math:`G_e > 0`)

**General multi-network creep** requires ODE integration because the stress
constraint :math:`\sigma_0 = \sum G_I \mu_{xy,I} + \eta_s \dot{\gamma}`
creates implicit coupling between modes.  RheoJAX solves this via
``diffrax`` with fixed-point stress balance at each time step.


.. _vlb-saos:
.. _vlb-saos-multi:

SAOS Derivation
===============

**Objective:** Find :math:`G'(\omega)` and :math:`G''(\omega)`.

**Step 1:** Assume :math:`\gamma(t) = \gamma_0 e^{i\omega t}`, so
:math:`\dot{\gamma}(t) = i\omega \gamma_0 e^{i\omega t}`.

**Step 2:** Seek solution :math:`\mu_{xy}(t) = \hat{\mu}_{xy} e^{i\omega t}`:

.. math::

   i\omega \hat{\mu}_{xy} = -k_d \hat{\mu}_{xy} + i\omega \gamma_0

.. math::

   \hat{\mu}_{xy} = \frac{i\omega \gamma_0}{k_d + i\omega}

**Step 3:** Complex modulus:

.. math::

   G^*(\omega) = \frac{G_0 \hat{\mu}_{xy}}{\gamma_0}
   = \frac{G_0 i\omega}{k_d + i\omega}
   = G_0 \frac{i\omega t_R}{1 + i\omega t_R}

**Step 4:** Separate real and imaginary parts:

.. math::

   G'(\omega) &= G_0 \frac{\omega^2 t_R^2}{1 + \omega^2 t_R^2} \\
   G''(\omega) &= G_0 \frac{\omega t_R}{1 + \omega^2 t_R^2}

**Limiting behaviors:**

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Regime
     - :math:`G'`
     - :math:`G''`
   * - :math:`\omega \ll k_d`
     - :math:`G_0 \omega^2/k_d^2` (slope 2 on log-log)
     - :math:`G_0 \omega/k_d` (slope 1 on log-log)
   * - :math:`\omega = k_d`
     - :math:`G_0/2` (crossover)
     - :math:`G_0/2` (crossover)
   * - :math:`\omega \gg k_d`
     - :math:`G_0` (plateau)
     - :math:`G_0 k_d/\omega` (slope -1)

**Cole-Cole plot:**

In the :math:`(G', G'')` plane, the Maxwell model traces a semicircle:

.. math::

   \left(G' - \frac{G_0}{2}\right)^2 + G''^2 = \left(\frac{G_0}{2}\right)^2

This provides a visual diagnostic: deviation from a semicircle indicates
non-Maxwell behavior.


.. _vlb-laos:

LAOS Implementation
===================

Under :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, the full 4-component ODE is
integrated numerically via ``diffrax.Tsit5`` with adaptive step size:

.. math::

   \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**ODE system (state = [μ_xx, μ_yy, μ_zz, μ_xy]):**

.. math::

   \dot{\mu}_{xx} &= k_d(1 - \mu_{xx}) + 2\gamma_0\omega\cos(\omega t)\mu_{xy} \\
   \dot{\mu}_{yy} &= k_d(1 - \mu_{yy}) \\
   \dot{\mu}_{zz} &= k_d(1 - \mu_{zz}) \\
   \dot{\mu}_{xy} &= -k_d\mu_{xy} + \gamma_0\omega\cos(\omega t)\mu_{yy}

Since :math:`\mu_{yy} = 1` and :math:`\mu_{xy}` satisfies a linear ODE, the
shear stress :math:`\sigma_{12} = G_0\mu_{xy}` is purely linear (fundamental
frequency only, no higher harmonics).

**Normal stress** :math:`N_1 = G_0(\mu_{xx} - 1)` is driven by
:math:`2\dot{\gamma}\mu_{xy}`, which is a product of two oscillatory
quantities and therefore contains :math:`2\omega` harmonics:

.. math::

   \mu_{xx}(t) - 1 \approx A_0 + A_2 \cos(2\omega t) + B_2 \sin(2\omega t) + \ldots

**RheoJAX implementation:**

- Simulates :math:`n` complete cycles (default 10)
- Discards first 5 cycles (transient decay)
- Returns :math:`\sigma_{12}(t)`, :math:`N_1(t)`, and FFT harmonics
- Verifies linearity via :math:`I_3/I_1` ratio for :math:`\sigma_{12}`


.. _vlb-extension:

Uniaxial Extension
==================

**Velocity gradient:**

.. math::

   \mathbf{L} = \begin{pmatrix}
   \dot{\varepsilon} & 0 & 0 \\
   0 & -\dot{\varepsilon}/2 & 0 \\
   0 & 0 & -\dot{\varepsilon}/2
   \end{pmatrix}

**Component equations (axial + transverse only, by symmetry):**

.. math::

   \dot{\mu}_{11} &= k_d(1 - \mu_{11}) + 2\dot{\varepsilon}\mu_{11} \\
   \dot{\mu}_{22} &= k_d(1 - \mu_{22}) - \dot{\varepsilon}\mu_{22}

**Steady state:**

.. math::

   \mu_{11}^{ss} &= \frac{k_d}{k_d - 2\dot{\varepsilon}} \quad (\dot{\varepsilon} < k_d/2) \\
   \mu_{22}^{ss} &= \frac{k_d}{k_d + \dot{\varepsilon}}

**Extensional stress:**

.. math::

   \sigma_E = G_0(\mu_{11}^{ss} - \mu_{22}^{ss})
   = G_0\dot{\varepsilon}\left(\frac{1}{k_d - 2\dot{\varepsilon}}
   + \frac{1}{k_d + \dot{\varepsilon}}\right)

**Transient response** from equilibrium:

.. math::

   \mu_{11}(t) &= \frac{k_d}{k_d - 2\dot{\varepsilon}}
   \left(1 - e^{-(k_d - 2\dot{\varepsilon})t}\right) + e^{-(k_d - 2\dot{\varepsilon})t} \\
   \mu_{22}(t) &= \frac{k_d}{k_d + \dot{\varepsilon}}
   \left(1 - e^{-(k_d + \dot{\varepsilon})t}\right) + e^{-(k_d + \dot{\varepsilon})t}

**Singularity at** :math:`\dot{\varepsilon} = k_d/2`:

The axial component :math:`\mu_{11} \to \infty` — chains cannot relax fast
enough to compensate the extensional stretching.  This divergence is:

- Regularized by finite extensibility (Langevin chains)
- Analogous to the coil-stretch transition in dilute solutions
- A useful diagnostic: if data shows extensional hardening, VLB with constant
  :math:`k_d` is insufficient


.. _vlb-general-solution:

General Analytical Solution
===========================

The full tensorial evolution equation :eq:`mu_evolution` (see :doc:`vlb`) admits
a compact integral-form solution through the substitution:

.. math::

   \mathbf{A} = \mathbf{F}^{-1} \cdot \boldsymbol{\mu} \cdot \mathbf{F}^{-T}

where :math:`\mathbf{F}(t)` is the deformation gradient.  Under this change of
variables, the convective terms cancel and the evolution reduces to:

.. math::

   \dot{\mathbf{A}} = k_d\!\left(\mathbf{F}^{-1}\mathbf{F}^{-T} - \mathbf{A}\right)

This is a first-order linear ODE in :math:`\mathbf{A}` with solution:

.. math::

   \mathbf{A}(t) = \mathbf{A}(0) e^{-k_d t}
   + k_d \int_0^t e^{-k_d(t-s)} \mathbf{F}^{-1}(s) \mathbf{F}^{-T}(s) \, ds

Transforming back:

.. math::

   \boldsymbol{\mu}(t) = \mathbf{F}(t) \cdot \mathbf{A}(t) \cdot \mathbf{F}^T(t)

This integral form is the basis for all closed-form solutions in the VLB
model.  For constant velocity gradient :math:`\mathbf{L}`, the deformation
gradient is :math:`\mathbf{F}(t) = e^{\mathbf{L} t}` and the integral can be
evaluated analytically for each protocol.


.. _vlb-laos-sinusoidal-proof:

LAOS: Exact Sinusoidal Proof
=============================

**Theorem:** For constant :math:`k_d`, the VLB shear stress
:math:`\sigma_{12}(t)` under LAOS is purely sinusoidal at the fundamental
frequency :math:`\omega`, with zero higher harmonics.

**Proof:** The :math:`\mu_{xy}` component satisfies (with :math:`\mu_{yy} = 1`):

.. math::

   \dot{\mu}_{xy} + k_d \mu_{xy} = \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

This is a **linear** first-order ODE with constant coefficients.  The
steady-state (post-transient) particular solution is:

.. math::

   \mu_{xy}^{ss}(t) = \frac{\gamma_0 \omega}{k_d^2 + \omega^2}
   \bigl[k_d \cos(\omega t) + \omega \sin(\omega t)\bigr]

which is a pure sinusoid at frequency :math:`\omega`.  Since
:math:`\sigma_{12} = G_0 \mu_{xy}`, the shear stress contains **only** the
fundamental frequency.  No harmonic generation occurs because the ODE is
linear — there is no mechanism to couple :math:`\omega` to :math:`n\omega`.

**Corollary:** The intrinsic nonlinearity ratio :math:`I_3/I_1 = 0` exactly
for the VLB model with constant :math:`k_d`.  Any measured :math:`I_3/I_1 > 0`
in experimental data indicates that :math:`k_d` depends on deformation.

.. note::

   This linearity is a property of the constant-:math:`k_d` model only.
   The VLBVariant with Bell breakage (:math:`k_d = k_d(\boldsymbol{\mu})`)
   produces nonlinear coupling and :math:`I_3/I_1 > 0`.


.. _vlb-laos-n1-harmonics:

LAOS: :math:`N_1` Harmonic Structure
=====================================

While :math:`\sigma_{12}` is purely sinusoidal, the first normal stress
difference :math:`N_1 = G_0(\mu_{xx} - \mu_{yy}) = G_0(\mu_{xx} - 1)` has
a richer harmonic content.

The :math:`\mu_{xx}` equation is driven by :math:`2\dot{\gamma}\mu_{xy}`,
which is a product of the driving frequency and the fundamental response:

.. math::

   2\dot{\gamma}(t) \cdot \mu_{xy}^{ss}(t) \propto
   \cos(\omega t) \cdot [\cos(\omega t) + \ldots]

This product generates a **DC offset** (constant term) and a
:math:`2\omega` component via the trigonometric identity
:math:`\cos^2(\omega t) = \frac{1}{2}(1 + \cos(2\omega t))`.

**Explicit solution** (steady state):

.. math::

   N_1(t) = \overline{N}_1 + N_1^{(2c)} \cos(2\omega t) + N_1^{(2s)} \sin(2\omega t)

where the nonzero mean :math:`\overline{N}_1 > 0` reflects the time-averaged
normal stress, and the :math:`2\omega` oscillation captures the
stretch-compression asymmetry within each cycle.

**Key features:**

- :math:`N_1` oscillates at :math:`2\omega` (not :math:`\omega`)
- Nonzero mean :math:`\overline{N}_1 = 2G_0 \gamma_0^2 \omega^2 / (k_d^2 + \omega^2)^2 \cdot k_d^2`
- No higher harmonics (:math:`4\omega, 6\omega, \ldots`) for constant :math:`k_d`
- Amplitude grows as :math:`\gamma_0^2` — quadratic in strain amplitude


.. _vlb-laos-linearity-validation:

LAOS Linearity Validation
==========================

In practice, verifying :math:`I_3/I_1 \approx 0` for the VLB model serves as
both a correctness check on the numerical implementation and a diagnostic for
the constant-:math:`k_d` assumption.

**Validation methodology:**

1. **Numerical integration:** Simulate LAOS via the ODE system for 10+ cycles
   using ``diffrax.Tsit5`` with tight tolerances (``rtol=1e-8``, ``atol=1e-10``)
2. **Transient removal:** Discard the first 5 cycles to eliminate initial
   transients
3. **FFT analysis:** Compute the Fourier spectrum of :math:`\sigma_{12}(t)`
   over the last 5 cycles
4. **Harmonic ratio:** :math:`I_3/I_1 = |a_3|/|a_1|` where :math:`a_n` are
   Fourier coefficients

**Expected results:**

- :math:`I_3/I_1 < 10^{-10}` (machine precision) for constant :math:`k_d`
- :math:`I_3/I_1 \sim 10^{-3}` to :math:`10^{-1}` for Bell :math:`k_d`
  with moderate :math:`\nu`

**Diagnostic use:** If :math:`I_3/I_1 > 10^{-6}` is measured experimentally,
the constant-:math:`k_d` VLB is insufficient and force-dependent breakage
(VLBVariant with ``breakage="bell"``) should be considered.


.. _vlb-extension-singularity:

Extensional Singularity and Safety
===================================

The steady-state axial component :math:`\mu_{11}^{ss} = k_d/(k_d - 2\dot{\varepsilon})`
diverges at the critical Weissenberg number:

.. math::

   \text{Wi}_{ext} = \frac{\dot{\varepsilon}}{k_d} = \frac{1}{2}

This is the **extensional catastrophe**: chains are stretched faster than they
can relax, and the Gaussian chain assumption breaks down.

**Physical interpretation:**

- For :math:`\text{Wi}_{ext} < 0.5`: steady state exists, Trouton ratio is finite
- At :math:`\text{Wi}_{ext} = 0.5`: transition to unbounded stretch
- For :math:`\text{Wi}_{ext} > 0.5`: no steady state; transient stress grows without bound

**Safety clamping in RheoJAX:**

The implementation clamps :math:`\mu_{11}` to prevent numerical overflow:

.. code-block:: python

   # In _kernels.py: safety clamp for extensional flows
   mu_11 = jnp.minimum(mu_11, L_max_sq)  # FENE-P bound
   # Or for Gaussian chains:
   mu_11 = jnp.minimum(mu_11, 1e6)  # numerical safety

**Regularization options:**

1. **FENE-P** (``VLBVariant(stress_type="fene")``): Bounds stress via
   :math:`f = L_{\max}^2/(L_{\max}^2 - \text{tr}(\boldsymbol{\mu}) + 3)`,
   which diverges at finite :math:`\text{tr}(\boldsymbol{\mu}) = L_{\max}^2 + 3`
   rather than at :math:`\text{Wi}_{ext} = 0.5`
2. **Bell breakage** (``VLBVariant(breakage="bell")``): Enhanced breakage at
   high stretch softens the singularity but does not fully remove it

For extensional rheology applications, the FENE-P variant is strongly
recommended.


Multi-Network Protocol Summary
==============================

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Protocol
     - Method
     - Formula
   * - Flow curve
     - Analytical
     - :math:`\sigma = (\sum G_I/k_{d,I} + \eta_s)\dot{\gamma}`
   * - Startup
     - Analytical
     - :math:`\sigma(t) = \sum \frac{G_I\dot{\gamma}}{k_{d,I}}(1-e^{-k_{d,I}t}) + \eta_s\dot{\gamma}`
   * - Relaxation
     - Analytical
     - :math:`G(t) = G_e + \sum G_I e^{-k_{d,I}t}`
   * - Creep
     - ODE / analytical (SLS)
     - General: ODE with stress balance; 1-mode + :math:`G_e`: SLS formula
   * - SAOS
     - Analytical
     - :math:`G^* = i\omega\eta_s + G_e + \sum G_I\frac{i\omega/k_{d,I}}{1+i\omega/k_{d,I}}`
   * - LAOS
     - ODE (diffrax)
     - Multi-mode ODE with oscillatory driving
