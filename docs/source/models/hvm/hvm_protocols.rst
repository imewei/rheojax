.. _hvm_protocols:

======================================
HVM Protocol Equations & Derivations
======================================

This page provides detailed step-by-step derivations of the HVM model
predictions for each rheological protocol.  All results assume the
three-subnetwork model (P + E + D) in simple shear with constant kinetic
rates unless otherwise stated.

For the governing equations and notation, see :doc:`hvm`.
For the underlying VLB single-network derivations, see
:doc:`/models/vlb/vlb_protocols`.


Simple Shear Geometry
=====================

In simple shear the velocity field is :math:`v_x = \dot{\gamma} y`,
with velocity gradient :math:`L_{12} = \dot{\gamma}` (all other entries zero).

**ODE state vector** (11 components):

.. code-block:: text

   [mu_E_xx, mu_E_yy, mu_E_xy,              # E-network distribution (3)
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy,  # E-network natural state (3)
    mu_D_xx, mu_D_yy, mu_D_xy,              # D-network distribution (3)
    gamma,                                    # accumulated strain (1)
    D]                                        # damage variable (1)

The **P-network** is tracked analytically:
:math:`\sigma_{P,xy} = (1-D) G_P \gamma(t)`.

**Component equations** for the E-network with natural-state evolution:

.. math::

   \dot{\mu}^E_{xx} &= 2\dot{\gamma}\,\mu^E_{xy} + k_{BER}(\mu^{E,nat}_{xx} - \mu^E_{xx}) \\
   \dot{\mu}^E_{yy} &= k_{BER}(\mu^{E,nat}_{yy} - \mu^E_{yy}) \\
   \dot{\mu}^E_{xy} &= \dot{\gamma}\,\mu^E_{yy} + k_{BER}(\mu^{E,nat}_{xy} - \mu^E_{xy})

**Natural-state evolution** (all components):

.. math::

   \dot{\mu}^{E,nat}_{ij} = k_{BER}(\mu^E_{ij} - \mu^{E,nat}_{ij})

**D-network** (standard VLB, natural state fixed at :math:`\mathbf{I}`):

.. math::

   \dot{\mu}^D_{xx} &= 2\dot{\gamma}\,\mu^D_{xy} - k_d^D(\mu^D_{xx} - 1) \\
   \dot{\mu}^D_{yy} &= -k_d^D(\mu^D_{yy} - 1) \\
   \dot{\mu}^D_{xy} &= \dot{\gamma}\,\mu^D_{yy} - k_d^D \mu^D_{xy}


**Key observation:** For the 22-components of the E-network, the sum
:math:`s = \mu^E_{yy} + \mu^{E,nat}_{yy}` satisfies :math:`\dot{s} = 0`,
so :math:`s = 2` for all time. The difference
:math:`d = \mu^E_{yy} - \mu^{E,nat}_{yy}` satisfies :math:`\dot{d} = -2k_{BER} d`,
with :math:`d(0) = 0`, so :math:`\mu^E_{yy}(t) = \mu^{E,nat}_{yy}(t) = 1`
for all :math:`t`.  This simplification applies to all protocols below.


.. _hvm-flow-curve:

Flow Curve Derivation
======================

**Objective:** Find steady-state :math:`\sigma_{12}(\dot{\gamma})` and
:math:`N_1(\dot{\gamma})`.

**Step 1 -- Permanent network:**

The P-network stress :math:`\sigma_{P,xy} = (1-D) G_P \dot{\gamma} t` grows
linearly with accumulated strain.  No true steady state exists for the
P-network in simple shear (absent damage).

**Step 2 -- Exchangeable network at steady state:**

Set :math:`\dot{\mu}^E_{ij} = 0` and :math:`\dot{\mu}^{E,nat}_{ij} = 0`
simultaneously.  From the natural-state equation:

.. math::

   0 = k_{BER}(\mu^E_{ij} - \mu^{E,nat}_{ij}) \implies \mu^E_{ij} = \mu^{E,nat}_{ij}

Therefore :math:`\sigma_E^{ss} = G_E(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat}) = 0`.

.. admonition:: Key result

   **The exchangeable vitrimer network carries zero stress at steady state.**
   The natural state fully catches up with the current deformation -- the
   network "forgets" its original reference configuration.

**Step 3 -- Dissociative network at steady state:**

Set :math:`\dot{\mu}^D_{ij} = 0` (identical to VLB, see :ref:`vlb-flow-curve`):

.. math::

   \mu^{D,ss}_{xy} = \frac{\dot{\gamma}}{k_d^D}, \quad
   \mu^{D,ss}_{xx} = 1 + \frac{2\dot{\gamma}^2}{(k_d^D)^2}, \quad
   \mu^{D,ss}_{yy} = 1

**Step 4 -- Steady-state stress (transient networks only):**

.. math::

   \boxed{\sigma_{12}^{ss} = G_D \frac{\dot{\gamma}}{k_d^D} = \eta_D \dot{\gamma}}

For a pure vitrimer with :math:`G_D = 0`: :math:`\sigma_{12}^{ss} = 0`.

**Steady-state first normal stress difference:**

.. math::

   \boxed{N_1^{ss} = 2 G_D \frac{\dot{\gamma}^2}{(k_d^D)^2}}

**First normal stress coefficient:** :math:`\Psi_1 = 2 \eta_D / k_d^D`.

**Effect of TST kinetics:**

With stress-dependent :math:`k_{BER}`, the approach to steady state is
modified (faster exchange at higher stress), but the final steady state is
still :math:`\sigma_E = 0` because the natural state always catches up.
For force-dependent :math:`k_d^D(\boldsymbol{\mu}^D)`, the steady state
becomes implicit: :math:`\mu^{D,ss}_{xy} = \dot{\gamma}/k_d^D(\mu^{D,ss})`,
which must be solved iteratively.  This is the mechanism for **shear thinning**.


.. _hvm-startup:

Startup Shear Derivation
=========================

**Objective:** Find :math:`\sigma_{12}(t)` and :math:`N_1(t)` for constant
:math:`\dot{\gamma}` starting from rest.

**Initial conditions:** All distribution tensors at :math:`\mathbf{I}`,
all natural-state tensors at :math:`\mathbf{I}`, :math:`D = 0`.

Permanent Network
-----------------

.. math::

   \sigma_{P,xy}(t) = (1 - D) G_P \dot{\gamma} t

Linear growth (unbounded absent damage).

Dissociative Network
--------------------

Standard VLB startup (see :ref:`vlb-startup`):

.. math::

   \boxed{\sigma_{D,xy}^+(t) = G_D \frac{\dot{\gamma}}{k_d^D}\left(1 - e^{-k_d^D t}\right)
   = \eta_D \dot{\gamma}\left(1 - e^{-t/\tau_D}\right)}

Monotonic rise to :math:`\eta_D \dot{\gamma}` with time constant :math:`\tau_D`.

Exchangeable Network (Natural-State Evolution)
----------------------------------------------

**Step 1:** Define the stress variable
:math:`\Delta_{xy} = \mu^E_{xy} - \mu^{E,nat}_{xy}` and the sum
:math:`\Sigma_{xy} = \mu^E_{xy} + \mu^{E,nat}_{xy}`.

**Step 2:** Subtract the natural-state ODE from the distribution ODE:

.. math::

   \dot{\Delta}_{xy} = \dot{\gamma} \cdot 1 - 2 k_{BER} \Delta_{xy}

(using :math:`\mu^E_{yy} = 1` and the coupled cancellation).

**Step 3:** Solve the linear first-order ODE with
:math:`\Delta_{xy}(0) = 0`:

.. math::

   \Delta_{xy}(t) = \frac{\dot{\gamma}}{2 k_{BER}}\left(1 - e^{-2 k_{BER} t}\right)

**Step 4:** The E-network startup shear stress is:

.. math::

   \boxed{\sigma_{E,xy}^+(t) = \frac{G_E \dot{\gamma}}{2 k_{BER}}
   \left(1 - e^{-2 k_{BER} t}\right)
   = \frac{\eta_E}{2} \dot{\gamma} \left(1 - e^{-2t/\tau_E}\right)}

Two features distinguish this from the D-network:

1. The effective relaxation time is :math:`\tau_E/2` -- both
   :math:`\boldsymbol{\mu}^E` and :math:`\boldsymbol{\mu}^E_{nat}` evolve,
   doubling the effective rate (:ref:`hvm-factor-of-2`).
2. This is the quasi-steady state.  At very long times, the natural state
   catches up and the E-network stress decays to zero.

Total Startup Stress
--------------------

.. math::

   \boxed{\sigma_{xy}^+(t) = G_P \dot{\gamma} t
   + \frac{G_E \dot{\gamma}}{2 k_{BER}}\left(1 - e^{-2 k_{BER} t}\right)
   + \frac{G_D \dot{\gamma}}{k_d^D}\left(1 - e^{-k_d^D t}\right)}

**Short-time (elastic) limit** (:math:`t \ll \min(\tau_E, \tau_D)`):

.. math::

   \sigma_{xy}^+(t) \approx (G_P + G_E + G_D) \dot{\gamma} t = G_{tot} \dot{\gamma} t

**TST overshoot mechanism:** When :math:`k_{BER}` depends on
:math:`\sigma^E` via TST, the BER rate accelerates as stress builds,
creating a sharper stress peak and faster relaxation.  This produces a
**stress overshoot** in :math:`\sigma_{xy}^+(t)` -- a nonlinear
viscoelastic phenomenon absent in the constant-rate model.

**Start-up first normal stress difference:**

.. math::

   N_1^+(t) = G_P \dot{\gamma}^2 t^2
   + G_E (\mu^E_{xx} - \mu^{E,nat}_{xx})
   + G_D \cdot 2\dot{\gamma}^2 \tau_D^2 \left(1 - e^{-t/\tau_D}\right)^2

The E-network normal stress difference involves the 11-component evolution
and must be integrated numerically when TST is active.


.. _hvm-relaxation:

Stress Relaxation Derivation
=============================

**Objective:** Find :math:`G(t) = \sigma_{12}(t)/\gamma_0` after step strain
:math:`\gamma_0` at :math:`t=0`.

**Kinematics:** For :math:`t > 0`: :math:`\dot{\gamma} = 0`, so
:math:`\mathbf{L} = \mathbf{0}`.

**Initial conditions** (immediately after step): All distribution tensors
jump affinely:
:math:`\mu^E_{xy}(0^+) = \mu^D_{xy}(0^+) = \gamma_0`,
:math:`\mu^{E,nat}_{xy}(0^+) = 0` (natural state has not yet evolved).

Permanent Network
-----------------

.. math::

   \sigma_{P,xy}(t) = (1 - D(t)) G_P \gamma_0

Constant in the absence of damage.

Dissociative Network
--------------------

Standard VLB relaxation (see :ref:`vlb-relaxation`):

.. math::

   \sigma_{D,xy}(t) = G_D \gamma_0 e^{-t/\tau_D}

Exchangeable Network
--------------------

**Step 1:** With :math:`\mathbf{L} = 0`, the stress variable
:math:`\Delta_{xy} = \mu^E_{xy} - \mu^{E,nat}_{xy}` satisfies:

.. math::

   \dot{\Delta}_{xy} = -2 k_{BER} \Delta_{xy}

**Step 2:** With :math:`\Delta_{xy}(0^+) = \gamma_0 - 0 = \gamma_0`:

.. math::

   \Delta_{xy}(t) = \gamma_0 e^{-2 k_{BER} t}

**Step 3:** The individual components are:

.. math::

   \mu^E_{xy}(t) = \frac{\gamma_0}{2}\left(1 + e^{-2 k_{BER} t}\right), \quad
   \mu^{E,nat}_{xy}(t) = \frac{\gamma_0}{2}\left(1 - e^{-2 k_{BER} t}\right)

At :math:`t \to \infty`: :math:`\mu^E_{xy} = \mu^{E,nat}_{xy} = \gamma_0/2`
-- the network retains a permanent shape change but carries zero stress.

**Step 4:** E-network shear stress:

.. math::

   \boxed{\sigma_{E,xy}(t) = G_E \gamma_0 e^{-2 k_{BER} t} = G_E \gamma_0 e^{-2t/\tau_E}}

The stress relaxes **twice as fast** as a standard Maxwell element with the
same :math:`k_{BER}`, because the natural state simultaneously evolves to
absorb the deformation.

Total Relaxation Modulus
------------------------

.. math::

   \boxed{G(t) = (1-D) G_P + G_E e^{-2t/\tau_E} + G_D e^{-t/\tau_D}}

This is a **two-mode relaxation** (three with damage):

- Fast mode: :math:`G_D e^{-t/\tau_D}` -- physical bond turnover
  (typically :math:`\tau_D \sim` ms--s)
- Intermediate mode: :math:`G_E e^{-2t/\tau_E}` -- vitrimer BER
  (typically :math:`\tau_E/2 \sim` min--hr above :math:`T_v`)
- Plateau: :math:`(1-D) G_P` -- permanent network

**Verification conditions:**

- :math:`G(0^+) = G_P + G_E + G_D = G_{tot}` (instantaneous modulus)
- :math:`G(\infty) = G_P = G_e` (equilibrium modulus)

**Normal stress relaxation:**

.. math::

   N_1(t) = (1-D) G_P \gamma_0^2 + G_E \gamma_0^2 e^{-2t/\tau_E}
   + G_D \gamma_0^2 e^{-t/\tau_D}

**TST effect:** At :math:`t = 0^+`, stress is maximal so :math:`k_{BER}` is
highest.  As stress relaxes, :math:`k_{BER}` decreases, slowing relaxation.
This produces a **stretched exponential** (KWW) profile:
:math:`G_E \exp[-(t/\tau_{KWW})^\beta]` with :math:`0 < \beta < 1`.


.. _hvm-creep:

Creep Derivation
=================

**Objective:** Find :math:`J(t) = \gamma(t)/\sigma_0` under constant applied
stress :math:`\sigma_0`.

Creep is the most mechanically complex protocol because the shear rate
:math:`\dot{\gamma}(t)` is the unknown and must be found self-consistently
from the stress balance.

Single-Network Limits
---------------------

**P + D only (standard linear solid / Zener):**

.. math::

   \boxed{J(t) = \frac{1}{G_P + G_D}
   + \frac{G_D}{G_P(G_P + G_D)}\left(1 - e^{-t/t_c^D}\right)}

where :math:`t_c^D = \tau_D G_P/(G_P + G_D)` is the creep retardation time.
Strain saturates at :math:`\gamma_\infty = \sigma_0/G_P`.

Full Three-Network Model
-------------------------

For constant :math:`k_{BER}` and :math:`k_d^D`:

.. math::

   \boxed{J(t) = \frac{1}{G_{tot}}
   + \frac{G_E}{G_P(G_P + G_E + G_D)}\left(1 - e^{-t/t_c^E}\right)
   + \frac{G_D}{(G_P + G_D)(G_P + G_E + G_D)}\left(1 - e^{-t/t_c^D}\right)
   + \delta J_{vitrimer}(t)}

The two retardation times are:

.. math::

   t_c^E = \frac{\tau_E}{2} \cdot \frac{G_P + G_D}{G_P + G_E + G_D}, \quad
   t_c^D = \tau_D \cdot \frac{G_P + G_E}{G_P + G_E + G_D}

The factor :math:`\tau_E/2` in :math:`t_c^E` reflects the doubled effective
rate from natural-state evolution.

**Vitrimer plastic creep** (:math:`\delta J_{vitrimer}`):

Because the natural state evolves, the equilibrium strain is not truly fixed
at :math:`\sigma_0/G_P` but slowly drifts as the E-network reference
configuration shifts.  On timescales :math:`\gg \tau_E`, the material appears
to flow with effective viscosity :math:`\eta_{vitrimer} \sim G_E \tau_E/2`,
but this flow is eventually arrested by the permanent network.

**Long-time compliance:**

.. math::

   \boxed{J(\infty) = \frac{1}{G_P}}

The E-network natural state has fully adjusted (zero stress), and the
D-network has fully relaxed -- only the permanent-network elastic compliance
remains.

**TST effect on creep:** Under constant applied stress, TST creates
nonlinear coupling as load redistributes between networks.  Initial rapid
creep (high stress on transient networks) followed by progressive slowing
(stress transfer to permanent network).


.. _hvm-saos:

SAOS Derivation
================

**Objective:** Find :math:`G'(\omega)` and :math:`G''(\omega)` in the
linear viscoelastic regime.

Using the relaxation modulus from the :ref:`relaxation derivation <hvm-relaxation>`:

.. math::

   G(t) = G_P + G_E e^{-2t/\tau_E} + G_D e^{-t/\tau_D}

the Boltzmann superposition principle gives :math:`G^*(\omega) = i\omega \hat{G}(i\omega)`:

**Permanent network:** Purely elastic, contributes only to :math:`G'`:

.. math::

   G_P^*(\omega) = G_P

**Exchangeable network:** Maxwell mode with effective time
:math:`\hat{\tau}_E = \tau_E/2`:

.. math::

   G_E^*(\omega) = G_E \frac{i\omega \hat{\tau}_E}{1 + i\omega \hat{\tau}_E}

**Dissociative network:** Maxwell mode with time :math:`\tau_D`:

.. math::

   G_D^*(\omega) = G_D \frac{i\omega \tau_D}{1 + i\omega \tau_D}

**Total storage and loss moduli:**

.. math::

   \boxed{G'(\omega) = G_P + G_E \frac{\omega^2 \hat{\tau}_E^2}{1 + \omega^2 \hat{\tau}_E^2}
   + G_D \frac{\omega^2 \tau_D^2}{1 + \omega^2 \tau_D^2}}

.. math::

   \boxed{G''(\omega) = G_E \frac{\omega \hat{\tau}_E}{1 + \omega^2 \hat{\tau}_E^2}
   + G_D \frac{\omega \tau_D}{1 + \omega^2 \tau_D^2}}

where :math:`\hat{\tau}_E = \tau_E/2 = 1/(2 k_{BER,0})`.

**Limiting behavior:**

- High frequency (:math:`\omega \to \infty`):
  :math:`G' \to G_P + G_E + G_D = G_{tot}`, :math:`G'' \to 0`
- Low frequency (:math:`\omega \to 0`):
  :math:`G' \to G_P`, :math:`G'' \to \omega(G_E \hat{\tau}_E + G_D \tau_D)`

**Crossover frequencies:** For well-separated relaxation times
(:math:`\tau_D \ll \tau_E`), two crossover regions appear:

- High-frequency crossover near :math:`\omega \sim 1/\tau_D`
  (D-network loss peak)
- Low-frequency crossover near :math:`\omega \sim 2/\tau_E = 2 k_{BER,0}`
  (E-network loss peak)

Two peaks in :math:`G''` is the rheological signature of the hybrid network.

**Zero-shear viscosity:**

.. math::

   \eta_0 = \lim_{\omega \to 0} \frac{G''(\omega)}{\omega}
   = G_E \hat{\tau}_E + G_D \tau_D = \frac{G_E \tau_E}{2} + G_D \tau_D

The E-network contribution is halved compared to a standard Maxwell mode.

**Cole-Cole consistency:** Each Maxwell mode traces a semicircle in
:math:`G''` vs :math:`G'`.  The full model is the sum of two semicircles
shifted vertically by :math:`G_P`.

**Temperature dependence:** The BER relaxation time follows Arrhenius:
:math:`\hat{\tau}_E(T) = 1/(2 k_{BER,0}(T)) \propto \exp(E_a/k_B T)`.
Master curves use shift factor :math:`a_T = \hat{\tau}_E(T)/\hat{\tau}_E(T_{ref})`.


.. _hvm-laos:

LAOS Derivation
================

**Objective:** Analyze the stress response under
:math:`\gamma(t) = \gamma_0 \sin(\omega t)`,
:math:`\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)`.

Permanent Network
-----------------

.. math::

   \sigma_{P,xy}(t) = G_P \gamma_0 \sin(\omega t)

Purely elastic, sinusoidal at :math:`\omega`.

Dissociative Network
--------------------

Since :math:`\mu^D_{yy} \to 1` in periodic steady state, the
:math:`\mu^D_{xy}` equation is a **linear ODE with periodic forcing**:

.. math::

   \dot{\mu}^D_{xy} + k_d^D \mu^D_{xy} = \gamma_0 \omega \cos(\omega t)

The periodic steady-state solution is:

.. math::

   \mu^{D,ss}_{xy}(t) = \frac{\gamma_0 \omega}{(k_d^D)^2 + \omega^2}
   \left[k_d^D \cos(\omega t) + \omega \sin(\omega t)\right]

The D-network shear stress :math:`\sigma^D_{xy} = G_D \mu^D_{xy}` contains
**no higher harmonics** -- purely sinusoidal at :math:`\omega`.

**Normal stress at** :math:`2\omega`: :math:`N_1^D = G_D(\mu^D_{xx} - 1)`
oscillates at :math:`2\omega` with a non-zero mean, because :math:`\mu^D_{xx}`
involves the product :math:`\dot{\gamma}(t) \mu^D_{xy}(t)`.

Exchangeable Network (Proof of Linearity)
------------------------------------------

**Step 1:** Using :math:`\mu^E_{yy} = 1`, define
:math:`\Delta_{xy} = \mu^E_{xy} - \mu^{E,nat}_{xy}`:

.. math::

   \dot{\Delta}_{xy} + 2 k_{BER} \Delta_{xy} = \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**Step 2:** This is a linear ODE with periodic forcing (rate :math:`2 k_{BER}`
instead of :math:`k_d^D`), so :math:`\Delta_{xy}` is exactly sinusoidal:

.. math::

   \Delta_{xy}^{ss}(t) = \frac{\gamma_0 \omega}{4 k_{BER}^2 + \omega^2}
   \left[2 k_{BER} \cos(\omega t) + \omega \sin(\omega t)\right]

Therefore :math:`\sigma^E_{xy} = G_E \Delta_{xy}` is also purely sinusoidal
-- **no higher harmonics** with constant :math:`k_{BER}`.

.. admonition:: Key result

   With constant kinetic rates, the HVM predicts **exactly sinusoidal** shear
   stress for all strain amplitudes.  The model is intrinsically linear in
   :math:`\sigma_{12}` (but nonlinear in :math:`N_1`).

Total LAOS Shear Stress
------------------------

.. math::

   \sigma_{xy}(t) = G_P \gamma_0 \sin(\omega t) + G_E \Delta_{xy}^{ss}(t)
   + G_D \mu^{D,ss}_{xy}(t)

All three terms are sinusoidal at :math:`\omega`.

Nonlinearity Mechanisms
-----------------------

Three mechanisms within the HVM produce genuine higher harmonics:

1. **TST stress-activated kinetics:** When :math:`k_{BER}` depends on stress,
   it oscillates at :math:`\omega`, creating parametric driving that generates
   :math:`3\omega, 5\omega, \ldots` in the response.

2. **Force-dependent dissociation:** If :math:`k_d^D(\boldsymbol{\mu}^D)`
   depends on chain stretch, the D-network develops odd harmonics.

3. **Langevin (non-Gaussian) chain statistics:** Near the chain extensibility
   limit, stress becomes nonlinear in :math:`\boldsymbol{\mu}`, introducing
   geometric nonlinearity.

**LAOS descriptors:** For the nonlinear case, standard analysis applies:
Fourier decomposition :math:`\sigma_{12}(t) = \gamma_0 \sum_n [G_n' \sin(n\omega t) + G_n'' \cos(n\omega t)]`
for odd :math:`n`, Chebyshev decomposition, and Lissajous figure analysis.


Protocol Comparison: HVM vs VLB
================================

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Protocol
     - VLB (constant :math:`k_d`)
     - HVM addition
   * - Flow curve
     - :math:`\sigma = \eta_0 \dot{\gamma}` (Newtonian)
     - :math:`\sigma_E = 0` at steady state; elastic :math:`G_P \gamma`
   * - SAOS
     - Single Maxwell mode
     - :math:`G_P` plateau + two Maxwell modes + factor-of-2 in :math:`\tau_E`
   * - Relaxation
     - Single exponential
     - Bi-exponential + :math:`G_P` plateau; TST gives KWW decay
   * - Startup
     - Monotonic rise
     - TST creates overshoot at high :math:`Wi`
   * - Creep
     - Unbounded linear
     - Bounded :math:`J(\infty) = 1/G_P`; vitrimer plastic creep
   * - LAOS
     - :math:`\sigma_{12}` sinusoidal
     - TST generates odd harmonics; :math:`G_P` adds elastic backbone

For VLB protocol derivations, see :doc:`/models/vlb/vlb_protocols`.
