.. _hvnm_protocols:

========================================
HVNM Protocol Equations & Derivations
========================================

This page provides detailed derivations of HVNM predictions for each
rheological protocol.  All results assume the four-subnetwork model
(P + E + D + I) in simple shear with constant kinetic rates unless
otherwise stated.

For the governing equations, see :doc:`hvnm`.
For HVM protocol derivations (P + E + D), see :doc:`/models/hvm/hvm_protocols`.
For the underlying VLB single-network results, see :doc:`/models/vlb/vlb_protocols`.


Simple Shear Geometry
=====================

**ODE state vector** (17 components, 18 with :math:`D_{int}`):

.. code-block:: text

   [mu_E_xx, mu_E_yy, mu_E_xy,              # E-network distribution (3)
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy,  # E-network natural state (3)
    mu_D_xx, mu_D_yy, mu_D_xy,              # D-network distribution (3)
    mu_I_xx, mu_I_yy, mu_I_xy,              # I-network distribution (3)
    mu_I_nat_xx, mu_I_nat_yy, mu_I_nat_xy,  # I-network natural state (3)
    gamma,                                    # accumulated strain (1)
    D]                                        # permanent damage (1)
    # D_int                                   # interfacial damage (if enabled)

The P-network is tracked analytically:
:math:`\sigma_{P,xy} = (1-D) G_P X(\phi) \gamma(t)`.

**I-network component equations** (parallel to E-network, with amplification):

.. math::

   \dot{\mu}^I_{xx} &= 2 X_I \dot{\gamma}\,\mu^I_{xy} + k_{BER}^{int}(\mu^{I,nat}_{xx} - \mu^I_{xx}) \\
   \dot{\mu}^I_{yy} &= k_{BER}^{int}(\mu^{I,nat}_{yy} - \mu^I_{yy}) \\
   \dot{\mu}^I_{xy} &= X_I \dot{\gamma}\,\mu^I_{yy} + k_{BER}^{int}(\mu^{I,nat}_{xy} - \mu^I_{xy})

**I-network natural-state evolution:**

.. math::

   \dot{\mu}^{I,nat}_{ij} = k_{BER}^{int}(\mu^I_{ij} - \mu^{I,nat}_{ij})

These have the identical mathematical form as the E-network in HVM,
with :math:`k_{BER}^{mat} \to k_{BER}^{int}` and
:math:`\dot{\gamma} \to X_I \dot{\gamma}`.

**Shorthand:** :math:`k_m \equiv k_{BER}^{mat,0}(T)`,
:math:`k_I \equiv k_{BER}^{int,0}(T)`, :math:`k_D \equiv k_d^D(T)`,
:math:`\hat{\tau}_m = 1/(2k_m)`, :math:`\hat{\tau}_I = 1/(2k_I)`,
:math:`\tau_D = 1/k_D`, :math:`X = X(\phi)`, :math:`X_I = X(\phi_{eff})`.


.. _hvnm-flow-curve:

Flow Curve Derivation
======================

**Objective:** Steady-state :math:`\sigma_{12}(\dot{\gamma})`.

**Step 1 -- Permanent network:**

:math:`\sigma_{P,xy} = (1-D)\tilde{G}_P \dot{\gamma} t` (unbounded, no
steady state).

**Step 2 -- Exchangeable and interphase at steady state:**

Both have evolving natural states, so at true steady state
:math:`\mu^E = \mu^{E,nat}` and :math:`\mu^I = \mu^{I,nat}`:

.. math::

   \sigma_E^{ss} = 0, \quad \sigma_I^{ss} = 0

**Step 3 -- Dissociative network** (unchanged from HVM):

.. math::

   \sigma_D^{ss} = \eta_D \dot{\gamma}

**Total steady-state stress (transient networks only):**

.. math::

   \boxed{\sigma_{12}^{ss} = \eta_D \dot{\gamma}}

.. admonition:: Key result

   Both the exchangeable and interphase networks carry **zero stress at true
   steady state** -- their natural states fully track the deformation.  Only
   the dissociative network contributes viscous stress.

**Intermediate quasi-steady state:** On timescales :math:`t \gg \hat{\tau}_m`
but :math:`t \ll \hat{\tau}_I`, the matrix BER has relaxed but the interphase
has not equilibrated:

.. math::

   \sigma_{12}^{qs} \approx (1-D)\tilde{G}_P \dot{\gamma} t
   + (1-D_{int}) G_{I,eff} X_I \dot{\gamma} \hat{\tau}_I + \eta_D \dot{\gamma}


.. _hvnm-startup:

Startup Shear Derivation
=========================

**Objective:** :math:`\sigma_{12}(t)` for constant :math:`\dot{\gamma}`
from rest.

**Initial conditions:** All tensors at :math:`\mathbf{I}`, :math:`D = D_{int} = 0`.

Permanent Network
-----------------

.. math::

   \sigma_{P,xy}(t) = \tilde{G}_P X \dot{\gamma} t

Exchangeable Network
---------------------

Identical to HVM (see :ref:`hvm-startup`):

.. math::

   \sigma_{E,xy}^+(t) = \frac{G_E^{eff} \dot{\gamma}}{2 k_m}
   \left(1 - e^{-2 k_m t}\right)
   = \frac{\eta_E^{eff}}{2} \dot{\gamma} \left(1 - e^{-t/\hat{\tau}_m}\right)

Dissociative Network
---------------------

.. math::

   \sigma_{D,xy}^+(t) = \eta_D \dot{\gamma}\left(1 - e^{-t/\tau_D}\right)

Interphase Network (New)
--------------------------

The I-network follows the same mathematics as the E-network with
:math:`k_m \to k_I` and :math:`\dot{\gamma} \to X_I \dot{\gamma}`:

.. math::

   \boxed{\sigma_{I,xy}^+(t) = \frac{(1-D_{int}) G_{I,eff} X_I \dot{\gamma}}{2 k_I}
   \left(1 - e^{-2 k_I t}\right)
   = \frac{(1-D_{int}) \eta_I^{eff}}{2} \dot{\gamma}
   \left(1 - e^{-t/\hat{\tau}_I}\right)}

where :math:`\eta_I^{eff} = G_{I,eff} X_I / k_I`.

Total Startup Stress
---------------------

.. math::

   \sigma_{xy}^+(t) = \tilde{G}_P X \dot{\gamma} t
   + \frac{\eta_E^{eff}}{2} \dot{\gamma} (1 - e^{-t/\hat{\tau}_m})
   + \eta_D \dot{\gamma} (1 - e^{-t/\tau_D})
   + \frac{(1-D_{int}) \eta_I^{eff}}{2} \dot{\gamma} (1 - e^{-t/\hat{\tau}_I})

**Short-time (elastic) limit** (:math:`t \ll \min(\hat{\tau}_m, \tau_D, \hat{\tau}_I)`):

.. math::

   \sigma_{xy}^+(t) \approx G_{tot}^{NC} \dot{\gamma} t

where :math:`G_{tot}^{NC} = \tilde{G}_P X + G_E^{eff} + G_D^{eff} + (1-D_{int}) G_{I,eff} X_I`
is the total instantaneous nanocomposite modulus.

**Double-overshoot signature:** With TST kinetics, the BER rates
accelerate as stress builds.  The interphase, with its higher activation
energy and strain amplification :math:`X_I`, shows a **later and larger**
overshoot than the matrix exchangeable network.  This produces a
distinctive **double-overshoot** in startup flow at certain temperatures --
a key experimental fingerprint of dual-kinetics nanocomposites.


.. _hvnm-relaxation:

Stress Relaxation Derivation
==============================

**Objective:** :math:`G(t) = \sigma_{12}(t)/\gamma_0` after step strain
:math:`\gamma_0`.

**Kinematics:** For :math:`t > 0`: :math:`\dot{\gamma} = 0`.

**Initial conditions** (immediately after step): All distribution tensors
at :math:`\mu_{xy} = \gamma_0` (affine), all natural-state tensors at
:math:`\mu^{nat}_{xy} = 0`.

Each subnetwork relaxes independently:

**Permanent:** :math:`\sigma_P(t) = (1-D) \tilde{G}_P X \gamma_0` (plateau).

**Exchangeable:** :math:`\sigma_E(t) = G_E^{eff} \gamma_0 e^{-t/\hat{\tau}_m}`.

**Dissociative:** :math:`\sigma_D(t) = G_D^{eff} \gamma_0 e^{-t/\tau_D}`.

**Interphase:** :math:`\sigma_I(t) = (1-D_{int}) G_{I,eff} X_I \gamma_0 e^{-t/\hat{\tau}_I}`.

**Total relaxation modulus:**

.. math::

   \boxed{G(t) = (1-D) \tilde{G}_P X + G_E^{eff} e^{-t/\hat{\tau}_m}
   + G_D^{eff} e^{-t/\tau_D}
   + (1-D_{int}) G_{I,eff} X_I e^{-t/\hat{\tau}_I}}

This is a **four-mode relaxation spectrum** (compared to two for HVM):

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Mode
     - Timescale
     - Origin
   * - Fast
     - :math:`\tau_D` (ms--s)
     - Physical bond turnover
   * - Intermediate
     - :math:`\hat{\tau}_m = \tau_m/2` (min--hr)
     - Matrix BER
   * - Slow
     - :math:`\hat{\tau}_I = \tau_I/2` (hr--days)
     - **Interfacial BER (new NP mode)**
   * - Plateau
     - :math:`\infty`
     - :math:`(1-D) \tilde{G}_P X` (amplified permanent)

The appearance of a distinct slow relaxation mode associated with the
interphase is a key experimental signature that distinguishes vitrimer
nanocomposites from unfilled vitrimers.

**Verification conditions:**

- :math:`G(0^+) = G_{tot}^{NC}` (instantaneous modulus)
- :math:`G(\infty) = (1-D) \tilde{G}_P X` (amplified equilibrium modulus)

**With diffusion-limited slow mode** (Karim et al. 2025):

When ``include_diffusion=True``, each BER mode acquires a long-time tail:

.. math::

   G(t)_{diff} = G_E^{eff} e^{-t/\hat{\tau}_m} e^{-k_{diff}^{mat} t}
   + (1-D_{int}) G_{I,eff} X_I e^{-t/\hat{\tau}_I} e^{-k_{diff}^{int} t} + \ldots

The :math:`e^{-k_{diff} t}` factors produce slow exponential decay beyond
BER relaxation, capturing experimentally observed incomplete relaxation at
intermediate times.


.. _hvnm-creep:

Creep Derivation
=================

**Objective:** :math:`J(t) = \gamma(t)/\sigma_0` under constant stress.

**Instantaneous response:** :math:`\gamma(0^+) = \sigma_0 / G_{tot}^{NC}`.

**Long-time saturation:** :math:`\gamma(\infty) = \sigma_0 / [(1-D)\tilde{G}_P X]`.

**Full creep compliance (Prony series form):**

.. math::

   J(t) = \frac{1}{(1-D)\tilde{G}_P X}
   + \sum_{\alpha \in \{E,D,I\}} \frac{G_\alpha^{eff}}
   {(1-D)\tilde{G}_P X \cdot ((1-D)\tilde{G}_P X + G_\alpha^{eff})}
   \left(1 - e^{-t/t_c^\alpha}\right)

where the **three retardation times** are:

.. math::

   t_c^E &= \hat{\tau}_m \cdot \frac{(1-D)\tilde{G}_P X + G_D^{eff} + (1-D_{int}) G_{I,eff} X_I}{G_{tot}^{NC}} \\
   t_c^D &= \tau_D \cdot \frac{(1-D)\tilde{G}_P X + G_E^{eff} + (1-D_{int}) G_{I,eff} X_I}{G_{tot}^{NC}} \\
   t_c^I &= \hat{\tau}_I \cdot \frac{(1-D)\tilde{G}_P X + G_E^{eff} + G_D^{eff}}{G_{tot}^{NC}}

**NP effect on creep:** The nanocomposite creeps less than the unfilled
vitrimer because:

1. Hydrodynamic amplification raises the permanent modulus by :math:`X(\phi)`
2. The slow interphase adds a long retardation time
3. Interphase modulus contributes to shielding the permanent network

The long-time compliance :math:`J(\infty) = 1/[(1-D)\tilde{G}_P X]` is
reduced by the factor :math:`1/X(\phi)` relative to the unfilled system.


.. _hvnm-saos:

SAOS Derivation
================

**Objective:** :math:`G'(\omega)` and :math:`G''(\omega)` in the linear regime.

Using the relaxation modulus from the :ref:`relaxation derivation <hvnm-relaxation>`:

**Total storage and loss moduli:**

.. math::

   \boxed{G'(\omega) = (1-D)\tilde{G}_P X
   + G_E^{eff} \frac{\omega^2 \hat{\tau}_m^2}{1 + \omega^2 \hat{\tau}_m^2}
   + G_D^{eff} \frac{\omega^2 \tau_D^2}{1 + \omega^2 \tau_D^2}
   + (1-D_{int}) G_{I,eff} X_I \frac{\omega^2 \hat{\tau}_I^2}{1 + \omega^2 \hat{\tau}_I^2}}

.. math::

   \boxed{G''(\omega) = G_E^{eff} \frac{\omega \hat{\tau}_m}{1 + \omega^2 \hat{\tau}_m^2}
   + G_D^{eff} \frac{\omega \tau_D}{1 + \omega^2 \tau_D^2}
   + (1-D_{int}) G_{I,eff} X_I \frac{\omega \hat{\tau}_I}{1 + \omega^2 \hat{\tau}_I^2}}

**Key features vs unfilled HVM:**

- **Three loss peaks in** :math:`G''(\omega)` at :math:`\omega \sim 1/\tau_D`,
  :math:`1/\hat{\tau}_m`, :math:`1/\hat{\tau}_I` (instead of two)
- **Elevated high-frequency plateau:**
  :math:`G'(\omega \to \infty) = G_{tot}^{NC} > G_{tot}^{HVM}`
- **Elevated low-frequency plateau:**
  :math:`G'(\omega \to 0) = (1-D)\tilde{G}_P X > G_P`
- **Shifted terminal crossover:** Additional slow mode pushes it to lower
  frequency

**Zero-shear viscosity:**

.. math::

   \eta_0^{NC} = G_E^{eff} \hat{\tau}_m + G_D^{eff} \tau_D
   + (1-D_{int}) G_{I,eff} X_I \hat{\tau}_I

The interphase BER contributes significantly because :math:`\hat{\tau}_I \gg \hat{\tau}_m`.

**Temperature master curves:** Below :math:`T_v^{int}`, the interphase
freezes and :math:`G'(\omega)` shows a secondary plateau that does not shift
with temperature -- the failure of simple TTS (thermorheological complexity)
is a diagnostic for dual-kinetics.


.. _hvnm-laos:

LAOS Derivation
================

**Objective:** Stress response under
:math:`\gamma(t) = \gamma_0 \sin(\omega t)`.

Linear Regime (Constant Rates)
-------------------------------

With constant :math:`k_m`, :math:`k_I`, :math:`k_D`, each subnetwork's
shear stress is exactly sinusoidal (linear ODE with sinusoidal forcing).
The total :math:`\sigma_{12}` is sinusoidal -- the model predicts purely
linear LAOS.

Nonlinearity Sources
---------------------

Four mechanisms generate higher harmonics in the HVNM:

1. **TST stress-activated BER** (both :math:`k_{BER}^{mat}` and
   :math:`k_{BER}^{int}`): Rate increases at peak stress, producing
   odd harmonics (:math:`3\omega, 5\omega, \ldots`).

2. **Strain-amplified interphase** (:math:`X_I > 1`): The interphase
   experiences amplified strain, reaching nonlinear territory before the
   bulk matrix.  For :math:`\gamma_0 X_I > \gamma_{NL}` but
   :math:`\gamma_0 < \gamma_{NL}`, the interphase is nonlinear while the
   matrix is still linear -- **intracycle strain softening**.

3. **Interfacial damage evolution** (:math:`\dot{D}_{int} > 0`): At
   large :math:`\gamma_0`, interfacial debonding occurs cyclically.
   If healing rate :math:`h_{int}` is slow compared to :math:`\omega`,
   :math:`D_{int}` ratchets up -- progressive intercycle softening
   (Payne + Mullins in LAOS).

4. **Force-dependent dissociation** (:math:`k_d^D(\boldsymbol{\sigma}^D)`):
   Same as HVM.

**Critical LAOS strain for nonlinearity onset:**

.. math::

   \boxed{\gamma_c^{NC} = \frac{\gamma_c^{mat}}{X_I}}

The nanocomposite becomes nonlinear at :math:`1/X_I` times the strain of
the unfilled system.

**LAOS descriptors:** Fourier decomposition
:math:`\sigma_{12}(t) = \sum_n [G_n' \sin(n\omega t) + G_n'' \cos(n\omega t)]`
for odd :math:`n`.  Third harmonic ratio :math:`I_{3/1} = |G_3|/|G_1|`
quantifies nonlinearity.  The HVNM predicts :math:`I_{3/1}` onset at lower
:math:`\gamma_0` than HVM due to strain amplification.


.. _hvnm-cyclic:

Cyclic Loading & Mullins Effect
================================

**Objective:** Stress-strain response under cyclic loading to maximum strain
:math:`\gamma_{max}` followed by unloading to zero stress.

First Loading
--------------

All four subnetworks respond in parallel.  The stress-strain curve follows
the startup response (:ref:`hvnm-startup`) truncated at :math:`\gamma_{max}`.

First Unloading (Hysteresis)
-----------------------------

Elastic unloading from all networks.  The unloading path lies below the
loading path, enclosing a hysteresis loop.  The dissipated energy per cycle:

.. math::

   W_{diss} = \oint \sigma\,d\gamma
   = W_{diss}^E + W_{diss}^D + W_{diss}^I + W_{diss}^{dam}

where :math:`W_{diss}^E, W_{diss}^I` come from BER exchange,
:math:`W_{diss}^D` from physical bond turnover, and :math:`W_{diss}^{dam}`
from damage.

Second Loading (Mullins Effect)
---------------------------------

The reloading curve lies below the first because:

1. :math:`D` may have increased (permanent chain scission), reducing
   :math:`\tilde{G}_P X`
2. :math:`D_{int}` may have increased (interfacial debonding), reducing
   :math:`(1-D_{int}) G_{I,eff} X_I`
3. :math:`\boldsymbol{\mu}^E_{nat}` and :math:`\boldsymbol{\mu}^I_{nat}`
   have evolved toward the deformed state, reducing stored elastic energy

Recovery Between Cycles
-------------------------

If a rest period :math:`t_{rest}` is allowed at zero strain:

- **Matrix BER:** :math:`\boldsymbol{\mu}^E, \boldsymbol{\mu}^E_{nat} \to \mathbf{I}`
  on timescale :math:`\hat{\tau}_m` (full recovery)
- **Interfacial BER:** :math:`\boldsymbol{\mu}^I, \boldsymbol{\mu}^I_{nat} \to \mathbf{I}`
  on timescale :math:`\hat{\tau}_I` (slower recovery)
- **Interfacial healing:** :math:`D_{int}` decreases on timescale
  :math:`1/h_{int}` (if :math:`T > T_v^{int}`)
- **Permanent damage:** :math:`D` is irreversible, never recovers

.. admonition:: Key prediction

   The Mullins effect in vitrimer nanocomposites is **partially recoverable**
   with time and temperature, unlike conventional filled rubbers where it is
   permanent.  The recovery fraction increases with temperature and rest time,
   with interphase healing as the rate-limiting step.

**Temperature dependence:**

- Above :math:`T_v^{int}`: softening partially recovered between cycles
  (self-healing active)
- Below :math:`T_v^{int}`: softening is permanent (no interfacial healing)
- Below :math:`T_v^{mat}`: all exchange frozen, behavior like filled thermoset


Protocol Comparison: HVNM vs HVM
==================================

.. list-table::
   :widths: 15 40 45
   :header-rows: 1

   * - Protocol
     - HVM (P + E + D)
     - HVNM addition (I-network)
   * - Flow curve
     - :math:`\sigma_E = 0` at SS
     - :math:`\sigma_I = 0` at SS; intermediate quasi-steady
   * - SAOS
     - Two Maxwell modes + :math:`G_P`
     - Third mode + amplified :math:`G_P X`; three :math:`G''` peaks
   * - Relaxation
     - Bi-exponential + plateau
     - Quad-exponential + amplified plateau
   * - Startup
     - TST overshoot
     - **Double overshoot** (matrix + interphase)
   * - Creep
     - Two retardation modes
     - Three retardation modes; reduced :math:`J(\infty)` by :math:`1/X`
   * - LAOS
     - :math:`\gamma_c^{mat}`
     - :math:`\gamma_c^{NC} = \gamma_c^{mat}/X_I` (earlier onset)
   * - Cyclic
     - Not applicable
     - Partially recoverable Mullins + Payne

For HVM protocol derivations, see :doc:`/models/hvm/hvm_protocols`.
