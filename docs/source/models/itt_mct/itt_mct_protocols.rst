.. _itt-mct-protocols:

ITT-MCT Protocol Equations
==========================

This document provides a comprehensive reference for the protocol-specific
equations used in ITT-MCT (Integration Through Transients Mode-Coupling Theory)
rheological modeling. Each protocol has distinct kinematics and stress formulas.

Quick Reference
---------------

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Protocol
     - Input
     - Key Output
   * - :ref:`Flow Curve <protocol-flow-curve>`
     - Constant :math:`\dot{\gamma}`
     - Steady stress :math:`\sigma(\dot{\gamma})`, yield stress :math:`\sigma_y`
   * - :ref:`Startup <protocol-startup>`
     - Step from rest to :math:`\dot{\gamma}`
     - :math:`\sigma(t)` with overshoot
   * - :ref:`Cessation <protocol-cessation>`
     - Stop shear at :math:`t=0`
     - Relaxing :math:`\sigma(t)`, residual stress
   * - :ref:`Creep <protocol-creep>`
     - Constant :math:`\sigma_0`
     - :math:`\gamma(t)`, :math:`J(t)`, viscosity bifurcation
   * - :ref:`SAOS <protocol-saos>`
     - Small :math:`\gamma_0 \sin(\omega t)`
     - :math:`G'(\omega)`, :math:`G''(\omega)`
   * - :ref:`LAOS <protocol-laos>`
     - Finite :math:`\gamma_0 \sin(\omega t)`
     - Harmonics :math:`\sigma'_n`, :math:`\sigma''_n`

Notation Guide
--------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Definition
   * - :math:`\sigma(t)`
     - Shear stress at time :math:`t`
   * - :math:`\dot{\gamma}(t)`
     - Shear rate at time :math:`t`
   * - :math:`\gamma(t,t')`
     - Accumulated strain from :math:`t'` to :math:`t`: :math:`\int_{t'}^{t}\dot{\gamma}(s)\,ds`
   * - :math:`G(t,t')`
     - Generalized shear modulus (history-dependent)
   * - :math:`G_{\text{eq}}(t)`
     - Equilibrium (quiescent) modulus
   * - :math:`\Phi_k(t,t')`
     - Transient density correlator at wavevector :math:`k`
   * - :math:`\Phi(t,t')`
     - Schematic (scalar) correlator
   * - :math:`h(\gamma)`
     - Strain decorrelation function: :math:`\exp[-(\gamma/\gamma_c)^2]`
   * - :math:`S(k)`
     - Static structure factor
   * - :math:`G_\infty`
     - High-frequency elastic modulus

Overview: The ITT Stress Functional
-----------------------------------

ITT-MCT is not a closed-form constitutive equation. It is a **procedure** that
expresses stress as a history integral over past deformations, weighted by a
generalized modulus built from transient density correlators.

The General History Integral
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shear stress at time :math:`t` is given by the **generalized Green-Kubo**
relation:

.. math::
   :label: itt-stress-functional

   \boxed{
   \sigma_{xy}(t) = \int_{-\infty}^{t} dt'\; \dot{\gamma}(t')\,G(t,t')
   }

where :math:`G(t,t')` is a **history-dependent** shear modulus functional of
transient correlators under the full strain history between times :math:`t'`
and :math:`t`.

The Microscopic Modulus
~~~~~~~~~~~~~~~~~~~~~~~

For Brownian colloids using the isotropized MCT approximation:

.. math::
   :label: microscopic-modulus

   G(t,t') = \frac{k_B T}{60\pi^2} \int_0^{\infty} dk\; k^4
   \left[\frac{S'(k)}{S(k)^2}\right]^2\,\Phi_k(t,t')^2

Physical interpretation:

- **Stress arises from distorted microstructure**: The integral over :math:`k`
  weights contributions from different length scales
- **Relaxation is controlled by cage breaking**: As density correlators
  :math:`\Phi_k` decay (cages break), the modulus decreases
- **S(k) weighting**: Modes near the S(k) peak contribute most to stress

Schematic Approximation
~~~~~~~~~~~~~~~~~~~~~~~

For the F\ :sub:`12` schematic model:

.. math::
   :label: schematic-modulus

   G(t,t') = G_\infty \Phi(t,t')^2

where :math:`\Phi(t,t')` is a single scalar correlator and :math:`G_\infty`
is a fitted high-frequency modulus.

.. _protocol-flow-curve:

Protocol 1: Flow Curve (Steady Shear)
-------------------------------------

**Protocol definition**: Constant shear rate applied indefinitely.

.. math::

   \dot{\gamma}(t) = \dot{\gamma} = \text{constant}

Kinematics
~~~~~~~~~~

- Accumulated strain: :math:`\gamma(t,t') = \dot{\gamma}(t - t')`
- Advected wavevector (ISM): :math:`k(\tau) = k\sqrt{1 + (\dot{\gamma}\tau)^2/3}`

Steady-State Stress
~~~~~~~~~~~~~~~~~~~

At steady state, the modulus becomes time-translation invariant:
:math:`G(t,t') \to G_{\dot{\gamma}}(t-t') = G_{\dot{\gamma}}(\tau)`.

.. math::
   :label: flow-curve-stress

   \boxed{
   \sigma_{xy}(\dot{\gamma}) = \dot{\gamma} \int_0^{\infty} d\tau\; G_{\dot{\gamma}}(\tau)
   }

where the steady-state modulus:

.. math::

   G_{\dot{\gamma}}(\tau) = \frac{k_B T}{60\pi^2} \int_0^{\infty} dk\; k^4
   \left[\frac{S'(k)}{S(k)^2}\right]^2\,\Phi_k(\tau;\dot{\gamma})^2

Dynamic Yield Stress
~~~~~~~~~~~~~~~~~~~~

In the glass state (:math:`\varepsilon > 0`):

.. math::
   :label: yield-stress

   \boxed{
   \sigma_y = \lim_{\dot{\gamma} \to 0} \sigma_{xy}(\dot{\gamma})
   }

Physical behavior:

- **Low rates**: Stress approaches yield stress :math:`\sigma_y`
- **Intermediate rates**: Power-law shear thinning :math:`\sigma \sim \dot{\gamma}^n`
- **High rates**: Linear viscous regime :math:`\sigma \sim \eta_\infty \dot{\gamma}`

.. _protocol-startup:

Protocol 2: Start-up of Steady Shear
------------------------------------

**Protocol definition**: Heaviside switch-on of shear rate at :math:`t=0`.

.. math::

   \dot{\gamma}(t) = \begin{cases}
   0, & t < 0 \\
   \dot{\gamma}_0, & t \geq 0
   \end{cases}

Stress Evolution
~~~~~~~~~~~~~~~~

For :math:`t \geq 0`:

.. math::
   :label: startup-stress

   \boxed{
   \sigma_{xy}(t) = \dot{\gamma}_0 \int_0^{t} d\tau\; G_{\dot{\gamma}_0}(\tau)
   }

Under constant rate (homogeneous flow):

.. math::

   \sigma_{xy}(t) = \dot{\gamma}_0 \int_0^{t} d\tau\; G_\infty \Phi(\tau;\dot{\gamma}_0)^2

Stress Overshoot Physics
~~~~~~~~~~~~~~~~~~~~~~~~

The stress overshoot is a signature of cage breaking:

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Strain Regime
     - Behavior
   * - :math:`\gamma \ll \gamma_c`
     - Linear elastic: :math:`\sigma \approx G_\infty \gamma`
   * - :math:`\gamma \sim \gamma_c`
     - Stress overshoot (cages begin to break)
   * - :math:`\gamma \gg \gamma_c`
     - Approach to steady state

**Overshoot strain**: :math:`\gamma_{\text{peak}} \sim 0.05-0.3` depending on
:math:`\varepsilon` and :math:`\dot{\gamma}`.

**Rate dependence**: Higher :math:`\dot{\gamma}` leads to larger overshoot
amplitude and earlier peak in time (but similar peak strain).

.. _protocol-cessation:

Protocol 3: Cessation (Stress Relaxation)
-----------------------------------------

**Protocol definition**: Shear at constant rate until :math:`t=0`, then stop.

.. math::

   \dot{\gamma}(t) = \begin{cases}
   \dot{\gamma}_{\text{pre}}, & t < 0 \\
   0, & t \geq 0
   \end{cases}

Stress Relaxation
~~~~~~~~~~~~~~~~~

For :math:`t \geq 0`:

.. math::
   :label: cessation-stress

   \boxed{
   \sigma_{xy}(t \geq 0) = \int_{-\infty}^{0} dt'\; \dot{\gamma}_{\text{pre}}\; G(t,t')
   }

Or, rewriting with :math:`\tau = -t'`:

.. math::

   \sigma_{xy}(t) = \dot{\gamma}_{\text{pre}} \int_0^{\infty} d\tau\;
   G_{\text{stop}}(t; \tau, \dot{\gamma}_{\text{pre}})

Mixed History
~~~~~~~~~~~~~

The correlators involve **mixed history**:

- **Pre-shear phase** (:math:`t' < 0`): Accumulated strain :math:`\gamma(0,t') = \dot{\gamma}_{\text{pre}}|t'|`
- **Relaxation phase** (:math:`t > 0`): No further strain, but correlators continue relaxing

Key Predictions
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - State
     - Relaxation Behavior
   * - **Fluid** (:math:`\varepsilon < 0`)
     - Complete decay to zero (exponential or stretched exponential)
   * - **Glass** (:math:`\varepsilon > 0`)
     - Residual stress :math:`\sigma_{\text{res}} > 0` (frozen cages)

The residual stress magnitude depends on the pre-shear rate and distance from
the glass transition.

.. _protocol-creep:

Protocol 4: Creep (Step Stress)
-------------------------------

**Protocol definition**: Constant stress applied at :math:`t=0`.

.. math::

   \sigma_{xy}(t) = \sigma_0 H(t)

The Volterra Equation
~~~~~~~~~~~~~~~~~~~~~

ITT is naturally strain/rate-controlled. For stress control, we must solve an
**inverse problem** (Volterra integral equation) for :math:`\dot{\gamma}(t)`:

.. math::
   :label: creep-volterra

   \boxed{
   \sigma_0 = \int_0^{t} dt'\; \dot{\gamma}(t')\; G(t,t') \quad (t > 0)
   }

while simultaneously evolving :math:`\Phi_k(t,t')` under the resulting
:math:`\dot{\gamma}(t)` history.

Creep Compliance
~~~~~~~~~~~~~~~~

The creep strain and compliance are:

.. math::

   \gamma(t) = \int_0^{t} \dot{\gamma}(s)\,ds, \qquad J(t) = \frac{\gamma(t)}{\sigma_0}

Viscosity Bifurcation
~~~~~~~~~~~~~~~~~~~~~

ITT-MCT predicts a sharp **viscosity bifurcation** at the yield stress:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Stress Regime
     - Response
   * - :math:`\sigma_0 < \sigma_y` (glass)
     - :math:`\dot{\gamma}(t) \to 0`, :math:`J(t)` saturates (solid-like)
   * - :math:`\sigma_0 > \sigma_y` (glass)
     - Delayed yielding: :math:`\dot{\gamma}(t)` grows, then steady flow
   * - Fluid state
     - :math:`J(t) \sim t` at long times (viscous flow)

The transition between creeping and flowing states is discontinuous - a hallmark
of the MCT glass transition.

.. _protocol-saos:

Protocol 5: SAOS (Small Amplitude Oscillatory Shear)
----------------------------------------------------

**Protocol definition**: Small amplitude oscillatory strain.

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t), \qquad \gamma_0 \ll 1

Linear Response Regime
~~~~~~~~~~~~~~~~~~~~~~

For :math:`\gamma_0 \ll \gamma_c`, advection is negligible. The modulus reduces
to its **quiescent (equilibrium) form**:

.. math::
   :label: saos-stress

   \boxed{
   \sigma_{xy}(t) = \int_{-\infty}^{t} dt'\; \dot{\gamma}(t')\; G_{\text{eq}}(t-t')
   }

Equilibrium Modulus
~~~~~~~~~~~~~~~~~~~

.. math::

   G_{\text{eq}}(t) = \frac{k_B T}{60\pi^2} \int_0^{\infty} dk\; k^4
   \left[\frac{S'(k)}{S(k)^2}\right]^2\,\Phi_k^{\text{eq}}(t)^2

where :math:`\Phi_k^{\text{eq}}(t)` satisfies the quiescent MCT equation (no
advection).

Complex Modulus
~~~~~~~~~~~~~~~

The complex modulus is obtained via Fourier transform:

.. math::
   :label: complex-modulus

   \boxed{
   G^*(\omega) = i\omega \int_0^{\infty} dt\; e^{-i\omega t}\; G_{\text{eq}}(t)
   }

with storage and loss moduli:

.. math::

   G'(\omega) &= \omega \int_0^{\infty} G_{\text{eq}}(t) \sin(\omega t)\, dt \\
   G''(\omega) &= \omega \int_0^{\infty} G_{\text{eq}}(t) \cos(\omega t)\, dt

MCT Predictions
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - State
     - G*(ω) Behavior
   * - **Fluid** (:math:`\varepsilon < 0`)
     - :math:`G' \sim \omega^2` at low ω, crossover to plateau at high ω
   * - **Glass** (:math:`\varepsilon > 0`)
     - :math:`G'(\omega \to 0) \to G_\infty f` (non-zero plateau)
   * - **Critical** (:math:`\varepsilon = 0`)
     - Power-law behavior :math:`G' \sim G'' \sim \omega^a`

.. _protocol-laos:

Protocol 6: LAOS (Large Amplitude Oscillatory Shear)
----------------------------------------------------

**Protocol definition**: Finite amplitude oscillatory strain.

.. math::

   \gamma(t) = \gamma_0 \sin(\omega t), \qquad \gamma_0 \sim O(\gamma_c)

Accumulated Strain
~~~~~~~~~~~~~~~~~~

The strain between times :math:`t'` and :math:`t`:

.. math::
   :label: laos-strain

   \boxed{
   \gamma(t,t') = \gamma_0 \left[\sin(\omega t) - \sin(\omega t')\right]
   }

Full ITT Stress
~~~~~~~~~~~~~~~

The stress involves the **full oscillatory history**:

.. math::
   :label: laos-stress

   \boxed{
   \sigma_{xy}(t) = \int_{-\infty}^{t} dt'\; \dot{\gamma}(t')\; G(t,t')
   }

where :math:`G(t,t')` depends on the time-dependent accumulated strain through
advected wavevectors and the strain decorrelation function.

Harmonic Decomposition
~~~~~~~~~~~~~~~~~~~~~~

By symmetry, only **odd harmonics** appear:

.. math::
   :label: laos-harmonics

   \boxed{
   \sigma_{xy}(t) = \sum_{n=1,3,5,...} \left[\sigma'_n \sin(n\omega t) +
   \sigma''_n \cos(n\omega t)\right]
   }

The **nonlinear moduli** are:

.. math::

   G'_n(\omega, \gamma_0) = \frac{\sigma'_n}{\gamma_0}, \qquad
   G''_n(\omega, \gamma_0) = \frac{\sigma''_n}{\gamma_0}

Third Harmonic Ratio
~~~~~~~~~~~~~~~~~~~~

A key nonlinearity measure is the **intrinsic nonlinearity** :math:`I_3/I_1`:

.. math::

   \frac{I_3}{I_1} = \frac{|\sigma_3^*|}{|\sigma_1^*|}

ITT-MCT predictions:

- Higher harmonics emerge when :math:`\gamma_0` is large enough to break cages
  each cycle
- :math:`I_3/I_1` increases with :math:`\gamma_0/\gamma_c`
- **Intra-cycle yielding**: stress peak occurs before strain peak
- **Strain softening**: :math:`G'_1` decreases with increasing :math:`\gamma_0`

Schematic F\ :sub:`12` Protocol Implementations
-----------------------------------------------

For the schematic model, the protocol equations simplify considerably.

Scalar Correlator Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \partial_t \Phi(t,t_0) + \Gamma \left[\Phi(t,t_0) +
   \int_{t_0}^{t} ds\; m(t,s,t_0)\;\partial_s\Phi(s,t_0)\right] = 0

F\ :sub:`12` Memory with Strain Cutoff
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   m(t,s,t_0) = h[\gamma(t,t_0)] \cdot h[\gamma(t,s)] \cdot
   \left(v_1\Phi(t,s) + v_2\Phi(t,s)^2\right)

with the strain decorrelation function:

.. math::

   h[\gamma] = \exp\left[-(\gamma/\gamma_c)^2\right]

Schematic Stress
~~~~~~~~~~~~~~~~

.. math::

   \sigma(t) = \int_{-\infty}^{t} dt'\; \dot{\gamma}(t')\; G_\infty\; \Phi(t,t')^2

This schematic model is widely used for:

- Creep and stress-controlled simulations (with feedback)
- LAOS and Fourier-Transform rheology
- Qualitative flow curves and yielding studies

See Also
--------

- :doc:`itt_mct_schematic` — F\ :sub:`12` schematic model theory and implementation
- :doc:`itt_mct_isotropic` — Full k-resolved ISM model with S(k) input
- :doc:`../index` — ITT-MCT models overview

References
----------

.. [1] Fuchs, M. & Cates, M. E. "Theory of nonlinear rheology and yielding of
   dense colloidal suspensions." *Phys. Rev. Lett.* **89**, 248304 (2002).

.. [2] Brader, J. M. et al. "First-principles constitutive equation for
   suspension rheology." *PNAS* **106**, 15186-15191 (2009).

.. [3] Voigtmann, T. "Nonlinear glassy rheology." *Curr. Opin. Colloid
   Interface Sci.* **19**, 549-560 (2014).
