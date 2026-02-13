.. _model-sgr-conventional:

======================================================
SGR Conventional (Soft Glassy Rheology) — Handbook
======================================================

Quick Reference
---------------

- **Use when:** Soft glassy materials (foams, emulsions, pastes, colloidal gels), yield stress fluids, aging materials
- **Parameters:** 3 (:math:`x`, :math:`G_0`, :math:`\tau_0`)
- **Key equation:** :math:`G'(\omega) \sim G''(\omega) \sim \omega^{x-1}` for :math:`1 < x < 2`
- **Test modes:** Oscillation, relaxation, creep, steady shear, LAOS
- **Material examples:** Concentrated emulsions, colloidal suspensions, foams, pastes, mayonnaise, hair gel

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`x`
     - Effective noise temperature (control parameter)
   * - :math:`E`
     - Trap depth (yield energy)
   * - :math:`l`
     - Local strain of a mesoscopic element
   * - :math:`\rho(E)`
     - Prior distribution of trap depths (exponential)
   * - :math:`\tau_0`
     - Microscopic attempt time
   * - :math:`\Gamma_0`
     - Attempt rate (:math:`1/\tau_0`)
   * - :math:`Z(t, t')`
     - Effective time interval (strain-warped time)

Overview
--------

The Soft Glassy Rheology (SGR) model is a mesoscopic constitutive framework for soft glassy materials—systems that exhibit structural disorder and metastability similar to glasses but with much weaker interaction energies (of order :math:`k_B T`). The model unifies the rheological behavior of diverse complex fluids including foams, emulsions, pastes, slurries, and colloidal glasses under a single theoretical framework.

The SGR model was developed by Sollich and coworkers [1]_ [2]_ based on Bouchaud's trap model for structural glasses. It treats the material as an ensemble of mesoscopic "elements"—local regions of material that can be in various states of local strain, trapped in energy wells of depth :math:`E`. The key insight is that thermal-like noise (with effective temperature :math:`x`) activates hopping between traps, while macroscopic strain biases these transitions.

Historical Context
~~~~~~~~~~~~~~~~~~

The Soft Glassy Rheology model emerged from the intersection of statistical physics and rheology in the late 1990s. The model represents a remarkable theoretical achievement: capturing the collective phenomenon of the glass transition within an essentially single-particle description.

**Origins in Bouchaud's Trap Model**

The SGR model is based upon Bouchaud's trap model of structural glasses [34]_, developed in the early 1990s to describe aging phenomena in disordered systems. Bouchaud's key insight was that the combination of:

1. An exponential distribution of trap depths :math:`\rho(E) = e^{-E}`
2. Arrhenius-activated hopping dynamics :math:`\tau(E) = \tau_0 e^{E/x}`

is sufficient to produce a genuine dynamical phase transition—a glass transition—at :math:`x = x_g = 1`. The exponential form and activated hopping should be viewed *jointly* as a tactic that allows glassy dynamics to be modeled in the simplest possible way [1]_ [2]_.

**Connection to Spin Glass Mean-Field Theories**

The effective noise temperature :math:`x` has deep connections to theories of out-of-equilibrium systems with slow dynamics. Cugliandolo and Kurchan [26]_ showed that similar "macroscopic" effective temperatures, which remain nonzero even as :math:`k_B T \to 0`, arise naturally in mean-field spin glass models. In these systems, the effective temperature governs the fluctuation-dissipation relation for slow degrees of freedom, just as :math:`x` does in the SGR model.

**Evolution of the "Noise Temperature" Concept**

The parameter :math:`x` was originally introduced as an effective "noise temperature" to capture the athermal fluctuations in soft materials—energy releases from neighboring rearrangements that activate hopping even when thermal energy :math:`k_B T` is negligible compared to barrier heights :math:`E`. Whether it is fully consistent to have :math:`x \gg k_B T` was initially debated [1]_ [2]_, but subsequent thermodynamic analyses [15]_ have established that :math:`x` can be rigorously interpreted as the true nonequilibrium thermodynamic temperature of the slow configurational degrees of freedom (see :doc:`sgr_generic` for the GENERIC framework treatment).

**Key Publications Timeline**

- **1992**: Bouchaud introduces the trap model for aging in disordered systems [34]_
- **1997**: Sollich, Lequeux, Hébraud & Cates propose the SGR model [1]_
- **1998**: Sollich derives the full constitutive equations [2]_
- **2000**: Fielding, Sollich & Cates provide comprehensive aging analysis [19]_
- **2004**: Cates & Sollich extend to tensorial formulation [20]_
- **2012**: Sollich & Cates establish thermodynamic interpretation [15]_

**Soft Glassy Materials**

Soft glassy materials (SGMs) encompass a wide class of substances with an unusual combination of material properties:

- **Foams** (shaving cream, bread dough)
- **Dense emulsions** (mayonnaise, salad cream)
- **Pastes** (toothpaste, hair gel)
- **Colloidal glasses** (paints, ceramic slips)
- **Onion phases** of surfactants
- **Block copolymer gels**

These materials share common rheological signatures: predominantly elastic behavior (:math:`G' > G''`) with both moduli approximately frequency-independent over many decades, and nearly constant loss tangent :math:`\tan\delta = G''/G'` across the frequency window. This behavior defies conventional viscoelastic models where :math:`\tan\delta` varies strongly with frequency.

This rheological motif—almost flat spectra with mild frequency dependence and subdominant losses—is the signature that the SGR model was designed to explain.

----

Rheology Without Time-Translational Invariance
----------------------------------------------

Before presenting the SGR model, we must establish a rheological framework that does not assume time-translational invariance (TTI). Most textbook rheology assumes TTI (material properties independent of when measurements are made), but aging materials fundamentally violate this assumption.

General Constitutive Relations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A system's shear stress :math:`\sigma(t)` depends functionally on its strain rate history :math:`\dot{\gamma}(t' < t)`. Conversely, :math:`\gamma(t)` can be expressed as a functional of the preceding stress history. For a step strain :math:`\gamma(t) = \gamma_0 \Theta(t - t_w)` applied at time :math:`t_w`:

.. math::

   \sigma(t) = \gamma_0 G(t - t_w, t_w; \gamma_0)

defining the **step strain response function** :math:`G(t - t_w, t_w; \gamma_0)`, which depends on both elapsed time :math:`t - t_w` *and* the waiting time :math:`t_w` since sample preparation.

In the linear regime (:math:`\gamma_0 \to 0`):

.. math::

   \lim_{\gamma_0 \to 0} G(t - t_w, t_w; \gamma_0) = G(t - t_w, t_w)

By superposition, the most general nontensorial linear constitutive equation is:

.. math::

   \sigma(t) = \int_{-\infty}^{t} G(t - t', t') \dot{\gamma}(t') \, dt'

**Only with both linearity and TTI** does this simplify to the conventional form:

.. math::

   \sigma(t) = \int_{-\infty}^{t} G(t - t') \dot{\gamma}(t') \, dt'

Time-Dependent Viscoelastic Spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For oscillatory strain :math:`\gamma(t) = \gamma_0 \cos(\omega t)` started at time :math:`t_s` after sample preparation, the stress response defines a time-varying viscoelastic spectrum:

.. math::

   G^*(\omega, t, t_s) = i\omega \int_{t_s}^{t} e^{-i\omega(t-t')} G(t - t', t') \, dt' + e^{-i\omega(t-t_s)} G(t - t_s, t_s)

This three-argument spectrum :math:`G^*(\omega, t, t_s)` is the natural generalization for aging systems. In the SGR model, this can often be simplified to :math:`G^*(\omega, t)` when :math:`\omega(t - t_s) \gg 1` and :math:`\omega t_s \gg 1`.

Yield Stress and Flow Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The flow curve :math:`\sigma_{ss} = \sigma(\dot{\gamma})` relates steady-state stress to strain rate. The **yield stress** is:

.. math::

   \sigma_y = \lim_{\dot{\gamma} \to 0} \sigma(\dot{\gamma})

When :math:`\sigma_y > 0`, the material exhibits a true yield stress (debated experimentally, but commonly observed behavior).

Important distinctions:

- Nonzero yield stress does **not** imply finite elastic modulus :math:`G_\infty`
- Zero yield stress does **not** imply finite viscosity :math:`\eta`
- For power-law fluids :math:`\sigma \sim \dot{\gamma}^p` with :math:`p < 1`: :math:`\sigma_y = 0` but :math:`\eta = \infty`

For aging materials without TTI, no meaningful "steady state response" exists in general. However, the SGR model restores TTI for nonzero :math:`\dot{\gamma}`, making the flow curve well-defined even in the glass phase.

What Is Aging?
~~~~~~~~~~~~~~

Aging is the property that **a significant part of stress relaxation takes place on timescales that grow with the age** :math:`t_w` **of the system**. Mathematically, the limits :math:`\Delta t \to \infty` and :math:`t_w \to \infty` cannot be interchanged.

The distinction from simple transients:

- **Transients**: Deviations from TTI that become negligible for large enough :math:`t_w`
- **Aging**: Relaxation timescales that grow proportionally with :math:`t_w`

----

Physical Foundations
--------------------

Thermodynamic Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recent work [15]_ has provided a rigorous thermodynamic foundation for the SGR model, interpreting the effective noise temperature :math:`x` as the true nonequilibrium thermodynamic temperature :math:`\chi` of the slow configurational degrees of freedom.

The material is conceptualized as two weakly interacting subsystems:
1. **Configurational subsystem (slow)**: Describes the arrangement of mesoscopic elements in the energy landscape. Its temperature is :math:`\chi = x`.
2. **Kinetic-vibrational subsystem (fast)**: Describes fast motion within traps. Its temperature is the bath temperature :math:`\theta` (typically room temperature).

In this framework, the SGR equation of motion ensures that the second law of thermodynamics (non-negative entropy production) is satisfied if and only if the noise temperature :math:`x` is identified with the configurational temperature :math:`\chi`.

**Key Implication:**
The evolution of :math:`x` is not arbitrary but governed by the first law of thermodynamics for the configurational subsystem:

.. math::

   C_V^{\text{eff}} \dot{\chi} = W + A(\theta - \chi)

where :math:`W` is the rate of work done on the configurational degrees of freedom (dissipated power), and :math:`A` describes heat transfer coupling to the thermal bath. This constrains extensions of the SGR model where :math:`x` varies with time or flow rate.

Mesoscopic Trap Model
~~~~~~~~~~~~~~~~~~~~~

The SGR model describes the material as consisting of many mesoscopic elements. "Mesoscopic" means:

- **Large enough** that continuum variables (strain, stress) apply
- **Small enough** that a macroscopic sample contains enough elements for statistical averaging

Each element is characterized by:

1. **Local strain** :math:`l` — the strain stored in that element
2. **Trap depth** :math:`E` — the energy barrier to rearrangement (yield energy)
3. **Local stress** :math:`kl` — elastic stress with spring constant :math:`k`

The macroscopic stress is the ensemble average: :math:`\sigma = \langle kl \rangle`.

Dynamics Within and Between Traps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a newly prepared sample, each element has :math:`l = 0`. Macroscopic strain at rate :math:`\dot{\gamma}` causes each element to strain: :math:`\dot{l} = \dot{\gamma}` (affine deformation). This continues until the element reaches a maximal strain :math:`l_y` and **yields**, rearranging into a new equilibrium with :math:`l = 0`.

The local strain exhibits a **saw-tooth** dependence on time: gradual increase during affine straining, then discontinuous reset upon yielding.

The yield rate is modeled as an **activated process**:

.. math::

   \tau = \tau_0 \exp\left(\frac{E - \frac{1}{2}kl^2}{x}\right)

where:
   - :math:`\tau_0` is the microscopic attempt time
   - :math:`E - \frac{1}{2}kl^2` is the effective barrier height (reduced by stored elastic energy)
   - :math:`x` is the effective noise temperature

This captures two yielding mechanisms:

1. **Strain-induced**: Elements strained beyond :math:`l_y` yield exponentially quickly
2. **Activated**: Even unstrained elements can yield via activation over barrier :math:`E`

Effective Noise Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter :math:`x` is the dimensionless **effective noise temperature**. Unlike thermal temperature :math:`k_B T`, the "noise" in :math:`x` arises from:

- **Mechanical noise** from neighboring rearrangements
- **Shear-induced fluctuations** (shear rejuvenation)
- **Slow structural relaxations** (aging)
- **Nonlinear couplings** to other elements

Because energy barriers in soft materials are large compared to :math:`k_B T` (by factors of :math:`10^3`– :math:`10^6`), the effective noise :math:`x` is of order the mean barrier height :math:`\langle E \rangle`, not :math:`k_B T`. This "macroscopic" effective temperature remains nonzero even as :math:`k_B T \to 0`, consistent with theories of out-of-equilibrium systems with slow dynamics [26]_ [27]_.

The glass transition occurs at :math:`x_g = 1`:

.. list-table:: Material phases vs. effective temperature x
   :header-rows: 1
   :widths: 15 25 60

   * - Regime
     - Behavior
     - Physical interpretation
   * - :math:`x < 1`
     - Glass phase
     - Aging, yield stress; system cannot equilibrate on any finite timescale
   * - :math:`x = 1`
     - Glass transition
     - Critical point; logarithmic corrections to power-law rheology
   * - :math:`1 < x < 2`
     - Power-law fluid
     - :math:`G' \sim G'' \sim \omega^{x-1}`; viscoelastic liquid with infinite viscosity
   * - :math:`x \geq 2`
     - Newtonian liquid
     - Finite viscosity; exponential stress relaxation

Trap Distribution and Glass Transition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The distribution of trap depths is chosen as:

.. math::

   \rho(E) = \frac{1}{x_g} \exp(-E/x_g) \quad \text{for } E > 0

with :math:`x_g = \langle E \rangle` being the mean barrier height. This exponential distribution, combined with activated hopping, is sufficient to produce a **glass transition** at :math:`x = x_g`.

The equilibrium Boltzmann distribution would be:

.. math::

   P_{\text{eq}}(E) \propto \rho(E) \exp(E/x)

For :math:`x \leq x_g = 1`, this is **not normalizable**: no steady state exists, and the system must age. For :math:`x > 1`, there is a unique equilibrium state.

.. note::

   The remarkable achievement of the SGR model is representing the glass transition—a collective phenomenon—within a single-particle description. The exponential trap distribution and activated hopping should be viewed jointly as a tactic allowing glassy dynamics to be modeled simply.

By convention, we set :math:`x_g = k = \tau_0 = 1`, meaning:

- The strain variable :math:`l` is normalized such that yield occurs at :math:`l \sim O(1)`
- Time is scaled by the microscopic attempt time
- Low-frequency regime corresponds to :math:`\omega \ll 1`

.. tip:: **Key Physical Intuition**

   Think of :math:`x` as the "agitation level" of the material elements.

   - **Low agitation (** :math:`x < 1` **)**: Elements get stuck in deep traps. The system ages because it takes longer and longer to escape deeper traps.
   - **High agitation (** :math:`x > 1` **)**: Elements can hop out of traps easily. The system equilibrates to a steady fluid state.

----

Governing Equations
-------------------

The SGR model's mathematical structure is defined by coupled integral equations
that express stress as a functional of strain history, accounting for the
time-dependent yielding dynamics of mesoscopic elements.

The Coupled Integral Equations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SGR model is exactly solved by two coupled constitutive equations. Assuming the sample is prepared at time zero with zero stress and strain, and macroscopic strain :math:`\gamma(t)` is applied thereafter [2]_:

**Stress equation:**

.. math::

   \sigma(t) = \gamma(t) G_0(Z(t, 0)) + \int_0^t [\gamma(t) - \gamma(t')] Y(t') G_\rho(Z(t, t')) \, dt'

**Probability conservation:**

.. math::

   1 = G_0(Z(t, 0)) + \int_0^t Y(t') G_\rho(Z(t, t')) \, dt'

Here:
   - :math:`G_\rho(z) = \int_0^\infty \rho(E) e^{-z \exp(-E)} \, dE = \int_0^\infty \rho(E) e^{-z/\tau(E)} \, dE`
   - :math:`G_0(z)` is defined similarly for the initial distribution :math:`P_0(E)`
   - :math:`Y(t)` is the total yielding rate at time :math:`t`
   - :math:`Z(t, t')` is the "effective" time interval accumulated between :math:`t'` and :math:`t`

The effective time interval accounts for strain-induced yielding:

.. math::

   Z(t, t') = \int_{t'}^{t} \exp\left(\frac{[\gamma(t'') - \gamma(t')]^2}{2x}\right) dt''

.. note:: **Interpretation of Z(t, t')**

   The effective time :math:`Z(t, t')` acts as a "strain-accelerated clock."
   Macroscopic strain increases the yielding rate, effectively making time pass faster
   for the material elements. A highly sheared sample "ages" (or rejuvenates) faster
   than a quiescent one.

Physical Interpretation
~~~~~~~~~~~~~~~~~~~~~~~

The structure of the constitutive equations reveals the contributions to macroscopic stress:

1. **First term** :math:`\gamma(t) G_0(Z(t, 0))`: Contribution from elements that have *never yielded* since sample preparation (at :math:`t = 0`)

2. **Integral term**: Sum over contributions from elements that *last yielded* at time :math:`t'`. Each contributes:
   - Strain :math:`\gamma(t) - \gamma(t')` acquired since yielding
   - Weight :math:`Y(t') G_\rho(Z(t, t'))` = (rate of yielding at :math:`t'`) × (probability of not yielding between :math:`t'` and :math:`t`)

Probability conservation ensures the two terms sum to the total contribution from all elements.

Microscopic Derivation: The Birth-Death Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SGR constitutive equations can be derived systematically from a microscopic
master equation describing the dynamics of individual mesoscopic elements. This
**birth-death interpretation** provides physical insight into the model's structure
and connects it to the underlying trap dynamics.

**Master Equation for Element Dynamics**

Consider the probability :math:`P(l, E, t)` that an element has local strain :math:`l`
and trap depth :math:`E` at time :math:`t`. Under affine deformation (strain rate :math:`\dot{\gamma}`),
elements advect with the macroscopic strain while stochastically yielding:

.. math::

   \frac{\partial P}{\partial t} + \dot{\gamma} \frac{\partial P}{\partial l}
   = -\Gamma(l, E) P(l, E, t) + Y(t) \rho(E) \delta(l)

where:

- :math:`\Gamma(l, E) = \tau_0^{-1} \exp\left(-E/x + l^2/(2x)\right)` is the strain-enhanced escape rate
- :math:`Y(t) = \int dE \int dl \, \Gamma(l, E) P(l, E, t)` is the total yielding rate
- :math:`\rho(E) \delta(l)` represents elements "born" in new traps with zero local strain

The first term on the right represents **deaths** (elements escaping their traps),
while the second term represents **births** (elements falling into new traps).

**Derivation of the Constitutive Form**

Integrating along characteristics (following elements in strain space), the formal solution
for elements that last yielded at time :math:`t'` is:

.. math::

   P(l, E, t | t') = Y(t') \rho(E) \exp\left[-\int_{t'}^{t} \Gamma(\gamma(t'') - \gamma(t'), E) \, dt''\right]
   \delta\bigl(l - [\gamma(t) - \gamma(t')]\bigr)

The probability that such an element has *not* yielded by time :math:`t` is:

.. math::

   G_\rho(Z(t, t')) = \int_0^\infty \rho(E) \exp\left(-\frac{Z(t, t')}{\tau(E)}\right) dE

where :math:`Z(t, t') = \int_{t'}^{t} \exp\bigl([\gamma(t'') - \gamma(t')]^2/(2x)\bigr) dt''`
is the effective (strain-accelerated) time.

Summing over all "birth cohorts"—elements born at each time :math:`t'`—recovers the
integral constitutive equation:

.. math::

   \sigma(t) = \gamma(t) G_0(Z(t, 0)) + \int_0^t [\gamma(t) - \gamma(t')] Y(t') G_\rho(Z(t, t')) \, dt'

**Explicit Forms for Trap Survival Functions**

For the standard exponential trap distribution :math:`\rho(E) = \exp(-E)` (setting :math:`x_g = 1`):

.. math::

   G_\rho(z) &= \int_0^\infty e^{-E} \exp(-z e^{-E/x}) \, dE \\
             &= x \, z^{-x} \, \gamma(x, z)

where :math:`\gamma(s, z) = \int_0^z t^{s-1} e^{-t} dt` is the lower incomplete gamma function.

For large effective times (:math:`z \gg 1`):

.. math::

   G_\rho(z) \sim x \, \Gamma(x) \, z^{-x}

This **power-law tail** is the essential feature responsible for aging dynamics and
power-law rheology.

**Initial Distribution Effects**

The function :math:`G_0(z)` depends on the initial trap distribution :math:`P_0(E)` at sample preparation.
Common choices include:

1. **Equilibrated at high** :math:`x`: :math:`P_0(E) \propto \rho(E) e^{E/x_{\text{init}}}` truncated
   to ensure normalizability

2. :math:`\delta` **-function** (elements start in identical traps): :math:`P_0(E) = \delta(E - E_0)`, giving
   :math:`G_0(z) = \exp(-z/\tau_0 e^{E_0/x})`

3. **Random history** (uniform in :math:`\tau`): :math:`P_0(E) = x \rho(E)/\tau(E)`, which for
   :math:`x > 1` gives the equilibrium distribution

The choice of :math:`P_0(E)` affects early-time transients but not the long-time aging behavior,
which is governed by :math:`G_\rho(z)`.

.. note:: **Birth-Death vs. Population Balance**

   The birth-death formulation is mathematically equivalent to a population balance
   approach where one tracks :math:`n(E, t) dE` = number of elements in traps of depth
   :math:`E` to :math:`E + dE`. The birth-death language emphasizes the renewal process
   that is central to SGR dynamics: elements are continually recycled through the
   trap distribution, with no memory of their previous history.

Linear Response Regime
~~~~~~~~~~~~~~~~~~~~~~

When local strains are negligible (:math:`\gamma \ll 1`), the effective time interval becomes the actual time interval:

.. math::

   Z(t, t') \to t - t'

and the hopping rate :math:`Y(t)` becomes strain-independent. The stress response to any strain history follows directly from the constitutive equations.

For step strain with amplitude :math:`\gamma_0 \ll 1`, the linearized step strain response is:

.. math::

   G(t - t_w, t_w) = 1 - \int_{t_w}^{t} Y(t') G_\rho(t - t') \, dt'

Equilibrium (:math:`x > 1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For :math:`x > 1`, the system reaches equilibrium at long times. The steady-state viscoelastic spectrum is:

.. math::

   G^*(\omega) = G_0 \frac{\Gamma(1-x)(i\omega\tau_0)^{x-1}}{1 + \Gamma(1-x)(i\omega\tau_0)^{x-1}}

where :math:`\Gamma(\cdot)` is the gamma function. For :math:`1 < x < 2`, the storage and loss moduli scale as:

.. math::

   G'(\omega) &\sim G_0 (\omega\tau_0)^{x-1} \cos\left(\frac{\pi(x-1)}{2}\right)

   G''(\omega) &\sim G_0 (\omega\tau_0)^{x-1} \sin\left(\frac{\pi(x-1)}{2}\right)

The loss tangent is **frequency-independent**:

.. math::

   \tan\delta = \tan\left(\frac{\pi(x-1)}{2}\right) = \text{const}

This is the hallmark of power-law rheology—the phase angle :math:`\delta = \pi(x-1)/2` is constant across frequency.

**Important features for** :math:`1 < x < 2`:

- No linear response regime in steady shear (power-law, not Newtonian)
- Infinite zero-shear viscosity: :math:`\eta = \lim_{\omega \to 0} G''(\omega)/\omega = \infty`
- Flow curves are shear-thinning with :math:`\sigma/\dot{\gamma}` decreasing as :math:`\dot{\gamma}` increases

Relaxation Spectrum and Integral Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The viscoelastic response of the SGR model can be connected to a **continuous relaxation
spectrum** :math:`H(\tau)`, which provides physical insight into the distribution of
relaxation processes.

**Relaxation Spectrum Definition**

The storage and loss moduli are related to the relaxation spectrum by:

.. math::

   G'(\omega) &= G_e + \int_0^\infty H(\tau) \frac{(\omega\tau)^2}{1 + (\omega\tau)^2} \, d\ln\tau

   G''(\omega) &= \int_0^\infty H(\tau) \frac{\omega\tau}{1 + (\omega\tau)^2} \, d\ln\tau

where :math:`G_e` is the equilibrium modulus (zero for fluids).

**SGR Relaxation Spectrum for** :math:`1 < x < 2`

For equilibrated SGR in the power-law fluid regime, the relaxation spectrum has the form:

.. math::

   H(\tau) = G_0 \frac{\sin(\pi(x-1))}{\pi} \left(\frac{\tau}{\tau_0}\right)^{x-2}

This **power-law spectrum** with exponent :math:`x - 2` directly produces the
power-law moduli :math:`G' \sim G'' \sim \omega^{x-1}`.

**Physical Interpretation**

The spectrum :math:`H(\tau) \propto \tau^{x-2}` arises from the exponential trap distribution
:math:`\rho(E) \propto e^{-E}` and activated hopping :math:`\tau(E) = \tau_0 e^{E/x}`:

.. math::

   H(\tau) \, d\ln\tau \propto \rho(E(tau)) \, \frac{dE}{d\ln\tau} = x \rho(E) \propto \tau^{x-1} \cdot \tau^{-1} = \tau^{x-2}

This derivation shows why the exponent in :math:`H(\tau)` differs from that in :math:`G'(\omega)` by unity.

**Asymptotic Forms**

.. list-table:: Frequency Asymptotes for SGR (:math:`x > 1`)
   :widths: 20 40 40
   :header-rows: 1

   * - Limit
     - :math:`G'(\omega)`
     - :math:`G''(\omega)`
   * - :math:`\omega \to 0`
     - :math:`G_0 \Gamma(1-x) \cos(\pi(x-1)/2) \, (\omega\tau_0)^{x-1}`
     - :math:`G_0 \Gamma(1-x) \sin(\pi(x-1)/2) \, (\omega\tau_0)^{x-1}`
   * - :math:`\omega \to \infty`
     - :math:`G_0 \left[1 - \Gamma(1-x) \cos(\pi(x-1)/2) \, (\omega\tau_0)^{x-1}\right]`
     - :math:`G_0 \Gamma(1-x) \sin(\pi(x-1)/2) \, (\omega\tau_0)^{x-1}`
   * - :math:`x \to 2^-`
     - :math:`G_0 (\omega\tau_0)^2 / [1 + (\omega\tau_0)^2]` (Maxwell)
     - :math:`G_0 \omega\tau_0 / [1 + (\omega\tau_0)^2]`

**Special Cases**

For :math:`x = 2` (Newtonian limit), the relaxation spectrum degenerates to a single mode:

.. math::

   H(\tau) \to G_0 \, \delta(\ln\tau - \ln\tau_0)

recovering the Maxwell model with relaxation time :math:`\tau_0`.

For :math:`x < 1` (glass phase), the relaxation spectrum interpretation breaks down because
there is no equilibrium state. Instead, the response must be described by the full two-time
correlation function :math:`G(t, t_w)` including explicit aging.

**Kramers-Kronig Relations**

Since :math:`G^*(\omega)` is analytic in the upper half-plane, :math:`G'` and :math:`G''`
satisfy the Kramers-Kronig relations:

.. math::

   G'(\omega) - G_\infty &= -\frac{2}{\pi} \mathcal{P} \int_0^\infty \frac{\omega' G''(\omega')}{\omega'^2 - \omega^2} \, d\omega'

   G''(\omega) &= \frac{2\omega}{\pi} \mathcal{P} \int_0^\infty \frac{G'(\omega') - G_\infty}{\omega'^2 - \omega^2} \, d\omega'

These provide a consistency check for SGR predictions and can be used to verify experimental
data quality.

----

Rheological Aging: Strain-Controlled Experiments
------------------------------------------------

For :math:`x < 1` (glass phase), the system exhibits true aging. Here we detail the predictions for strain-controlled experiments.

Evolution of the Lifetime Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following a quench (sudden reduction of :math:`x` from a large value to :math:`x < 1` at time zero), the distribution of yield energies :math:`P(E, t)` evolves:

.. math::

   P(E, t) = P_0(E) \exp[-t/\tau(E)] + \int_0^t Y(t') \rho(E) \exp[-(t - t')/\tau(E)] \, dt'

with :math:`\tau(E) = \exp(E/x)`. Equivalently, the lifetime distribution :math:`P(\tau, t_w)` at waiting time :math:`t_w` scales as:

.. math::

   P(\tau, t_w) \simeq \begin{cases}
   x Y(t_w) \tau \rho(\tau) & \text{for } \tau \ll t_w \\
   x Y(t_w) t_w \rho(\tau) & \text{for } \tau \gg t_w
   \end{cases}

where :math:`Y(t_w) \sim t_w^{x-1} \to 0` as :math:`t_w \to \infty`.

**Physical interpretation**: The bulk of the distribution's weight is at :math:`\tau \simeq t_w`. In a system undergoing aging, the characteristic relaxation time is of order the age itself.

Step Strain Response
~~~~~~~~~~~~~~~~~~~~

For step strain :math:`\gamma_0` applied at waiting time :math:`t_w`, the stress relaxation modulus :math:`G(t - t_w, t_w)` has limiting forms when :math:`t - t_w \gg 1` and :math:`t_w \gg 1`:

**For** :math:`x > 1` **(no aging, only transients)**:

.. math::

   G \sim \begin{cases}
   (t - t_w)^{1-x} & \text{for } t - t_w \ll t_w \text{ (short time)} \\
   t_w (t - t_w)^{-x} & \text{for } t - t_w \gg t_w \text{ (long time)}
   \end{cases}

At large :math:`t_w`, the short-time regime accounts for more of the decay, and results become :math:`t_w`-independent.

**For** :math:`x < 1` **(aging proper)**:

.. math::

   G \sim \begin{cases}
   1 - [(t - t_w)/t_w]^{1-x} & \text{for } t - t_w \ll t_w \\
   [(t - t_w)/t_w]^{-x} & \text{for } t - t_w \gg t_w
   \end{cases}

The major part of the decay of :math:`G` occurs on a timescale of order :math:`t_w` itself. The SGR model shows the simplest kind of aging: a single aging timescale directly proportional to :math:`t_w`.

Simple vs Complex Aging
~~~~~~~~~~~~~~~~~~~~~~~

The SGR model exhibits what is called **simple aging**: the waiting time :math:`t_w` enters
observables only through a mapping to an effective time scale. This is distinct from
**complex aging** where :math:`t_w` appears explicitly in the functional form of responses.

**Characteristics of Simple Aging in SGR**

In simple aging, the two-time response function :math:`R(t, t_w)` can be written as:

.. math::

   R(t, t_w) = t_w^{-a} \, \mathcal{R}\left(\frac{t - t_w}{t_w}\right)

where :math:`\mathcal{R}(u)` is a universal scaling function and :math:`a` depends on the
observable. The key features are:

1. **Single timescale**: The characteristic relaxation time scales as :math:`t_w` (or :math:`t_w^\mu` more generally)
2. **Universal scaling**: Responses at different :math:`t_w` collapse when plotted vs :math:`(t-t_w)/t_w`
3. **Age factorization**: The :math:`t_w`-dependence factorizes out of the scaling function

For the SGR step strain response with :math:`x < 1`:

.. math::

   G(t - t_w, t_w) = \mathcal{G}\left(\frac{t - t_w}{t_w}\right)

The function :math:`\mathcal{G}(u)` is :math:`t_w`-independent, exemplifying simple aging.

**The** :math:`\mu` **-Exponent**

The **aging exponent** :math:`\mu` characterizes how the effective relaxation time grows with age:

.. math::

   \tau_{\text{eff}} \propto t_w^\mu

.. list-table:: Aging Exponents in SGR
   :widths: 20 20 60
   :header-rows: 1

   * - Regime
     - :math:`\mu` value
     - Physical meaning
   * - :math:`x < 1` (glass)
     - :math:`\mu = 1`
     - Full aging; relaxation time equals sample age
   * - :math:`x = 1` (critical)
     - :math:`\mu = 1` (log corrections)
     - Marginal; logarithmic corrections to scaling
   * - :math:`1 < x < 2`
     - :math:`\mu < 1` (transient)
     - Sub-aging; approaches equilibrium

For :math:`x < 1`, SGR gives :math:`\mu = 1` exactly—the strongest form of simple aging where
the effective relaxation time *equals* the age.

**Conditions for Complex Aging**

Complex aging arises when:

- Multiple coupled relaxation processes with different aging rates
- Spatial heterogeneity introduces additional length scales
- The trap distribution evolves non-trivially (beyond the SGR assumption)
- External driving competes with aging on comparable timescales

**Experimental Signatures**

To distinguish simple from complex aging experimentally:

.. list-table::
   :widths: 40 30 30
   :header-rows: 1

   * - Test
     - Simple aging
     - Complex aging
   * - :math:`G(t, t_w)` vs :math:`(t-t_w)/t_w` collapse
     - Yes (universal curve)
     - No (explicit :math:`t_w` dependence)
   * - :math:`G'(\omega, t_w)` scaling with :math:`\omega t_w`
     - Clean superposition
     - Deviations from superposition
   * - Two-step aging protocols
     - Effective time additive
     - Memory/rejuvenation effects

.. note:: **Effective Time Mapping**

   For simple aging, one can define an **effective time** :math:`\xi(t)` such that:

   .. math::

      R(t, t_w) = R_{\text{eq}}(\xi(t) - \xi(t_w))

   where :math:`R_{\text{eq}}` is the equilibrium response function (if it exists for :math:`x > 1`).
   The SGR model provides an explicit effective time through :math:`Z(t, t')`, which reduces
   to the real time interval in linear response but accelerates under strain.

Time-Dependent Oscillatory Response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For oscillatory strain, the viscoelastic spectrum :math:`G^*(\omega, t)` depends on the measurement time :math:`t` (age of the system). In the low-frequency limit (:math:`\omega \ll 1`) with :math:`t \gg 1`:

**For** :math:`1 < x < 2`:

.. math::

   G^*(\omega, t) \sim (i\omega)^{x-1}

This is :math:`t`-independent within a time of :math:`O(1/\omega)` after the quench.

**For** :math:`x < 1`:

.. math::

   G^*(\omega, t) \approx 1 - (i\omega t)^{x-1}

The response is a function only of the product :math:`\omega t`. As the system ages into deeper traps with :math:`\tau > 1/\omega`, the response becomes more elastic:

- Storage modulus :math:`G'` increases at low frequencies
- Loss modulus :math:`G''` decreases at low frequencies
- Each spectrum terminates at :math:`\omega t \simeq 1` (cannot measure periods beyond the sample's age)

.. warning::

   The apparent rise in :math:`G''(\omega)` at low frequencies (often interpreted as a "loss peak" at lower frequencies) may be an illusion caused by aging. No oscillatory measurement can probe frequencies far below :math:`1/t_w`, yet in aging materials, the relaxation time grows with the age. The putative loss peak can never be observed—it is a complete figment of the imagination.

Startup Shear
~~~~~~~~~~~~~

For steady shear at rate :math:`\dot{\gamma} \ll 1` started at time :math:`t_w`, with total strain :math:`\gamma = \dot{\gamma}(t - t_w)`:

**Linear Regime**: Initially (:math:`\gamma \ll 1`), the response is elastic (:math:`\sigma \sim \gamma`) for :math:`x < 1`.

**Overshoot**: As strain increases, strain-induced yielding sets in. The peak stress occurs at a strain :math:`\gamma_{\text{peak}}` that increases logarithmically with age:

.. math::

   \gamma_{\text{peak}} \sim \sqrt{2x \ln(\dot{\gamma} t_w)}

The height of the stress overshoot also increases with age before settling into the age-independent steady state flow stress :math:`\sigma_{\text{ss}}` [2]_.

**Steady State**: For :math:`t \gg 1/\dot{\gamma}`, aging is "interrupted" by flow, and the stress approaches the steady-state flow curve value (Herschel-Bulkley for :math:`x<1`).

Rheological Aging: Stress-Controlled Experiments
------------------------------------------------

Stress-controlled experiments (creep) provide complementary information to strain-controlled tests. These are more difficult to analyze because the SGR constitutive equations naturally express stress as a function of strain history.

Creep Compliance
~~~~~~~~~~~~~~~~

Under constant stress :math:`\sigma_0` applied at time :math:`t_w`, the strain evolves as:

.. math::

   \gamma(t) = J(t - t_w, t_w) \sigma_0

**For** :math:`1 < x < 2` (fluid phase), the creep compliance exhibits power-law behavior:

.. math::

   J(t - t_w, t_w) \sim (t - t_w)^{x-1}

**For** :math:`x < 1` (glass phase), the material exhibits **logarithmic creep** for small stresses (:math:`\sigma_0 \ll \sigma_y`):

.. math::

   J(t - t_w, t_w) \sim \ln\left(\frac{t - t_w}{t_w}\right)

This weak logarithmic aging contrasts with the strong power-law aging seen in step strain relaxation.

Steady Shear Flow
~~~~~~~~~~~~~~~~~

Under steady shear at rate :math:`\dot{\gamma}`, the stress approaches:

.. math::

   \sigma(\dot{\gamma}) = \sigma_y + \eta_\infty \dot{\gamma}^{x-1}

where:
   - :math:`\sigma_y` is the yield stress (nonzero for :math:`x < 1`)
   - :math:`\eta_\infty` is a high-rate viscosity parameter

This is the **Herschel-Bulkley** form with flow index :math:`n = x - 1`.

For :math:`x > 1`, the yield stress vanishes (:math:`\sigma_y = 0`) and the material is purely shear-thinning.

**Interrupted Aging by Flow**: Under steady flow, yielding of elements occurs even in the deepest traps due to strain-induced lowering of barriers. The time to yield becomes power-law rather than exponential in :math:`E`, so the aging process is "interrupted" by flow [2]_ [19]_. The flow curve remains well-defined even in the glass phase.

Yield Stress Determination
~~~~~~~~~~~~~~~~~~~~~~~~~~

The yield stress :math:`\sigma_y` is a central quantity in soft glassy materials. The SGR model
distinguishes between **static** and **dynamic** yield stresses, which arise from different
experimental protocols.

**Static Yield Stress (Imposed Stress)**

The static yield stress :math:`\sigma_y^s` is determined from creep experiments under constant
applied stress. For :math:`x < 1`:

- If :math:`\sigma_0 < \sigma_y^s`: Bounded (logarithmic) creep—the material does not flow
- If :math:`\sigma_0 > \sigma_y^s`: Unbounded flow after a delay time :math:`t_d(\sigma_0)`

The static yield stress from the SGR model is:

.. math::

   \sigma_y^s = \lim_{\dot{\gamma} \to 0^+} \sigma(\dot{\gamma})

For the glass phase (:math:`x < 1`), :math:`\sigma_y^s > 0` arises from the infinitely deep traps
that can only escape through strain-induced barrier lowering.

**Dynamic Yield Stress (Flow Curves)**

The dynamic yield stress :math:`\sigma_y^d` is extracted from the low-shear-rate limit of
flow curves:

.. math::

   \sigma_y^d = \lim_{\dot{\gamma} \to 0} \left[\sigma(\dot{\gamma}) - \eta_\infty \dot{\gamma}^{x-1}\right]

In the SGR model, :math:`\sigma_y^s = \sigma_y^d` for idealized conditions. However, in practice:

- **Protocol dependence**: Ramp rate, waiting time, and pre-shear affect the measured value
- **Aging effects**: :math:`\sigma_y` may increase with sample age :math:`t_w`
- **Thixotropy**: Structural breakdown during measurement complicates interpretation

**Experimental Protocols**

.. list-table:: Yield Stress Measurement Methods
   :widths: 25 35 40
   :header-rows: 1

   * - Protocol
     - Method
     - SGR Prediction
   * - Stress ramp
     - Identify stress at which :math:`\dot{\gamma}` diverges
     - :math:`\sigma_y(t_w) \sim` const for :math:`x < 1`
   * - Creep test
     - Stress below which flow is bounded
     - Logarithmic creep for :math:`\sigma < \sigma_y`
   * - Flow curve extrapolation
     - Fit Herschel-Bulkley, extrapolate to :math:`\dot{\gamma} \to 0`
     - :math:`\sigma_y + \eta_\infty \dot{\gamma}^{x-1}`
   * - Oscillatory stress sweep
     - Identify crossover :math:`G' = G''`
     - Approximate; overestimates :math:`\sigma_y`

**Cox-Merz Rule Violations**

The Cox-Merz rule relates linear viscoelastic response to nonlinear flow:

.. math::

   |\eta^*(\omega)| = \eta(\dot{\gamma})\big|_{\dot{\gamma} = \omega}

where :math:`\eta^* = G^*/i\omega` is the complex viscosity.

**SGR predictions for Cox-Merz**:

- **:math:`x > 2`** (Newtonian): Cox-Merz holds approximately
- **:math:`1 < x < 2`** (power-law fluid): Cox-Merz fails due to infinite zero-shear viscosity
- **:math:`x < 1`** (glass): Strong violation— :math:`|\eta^*|` diverges as :math:`\omega \to 0`, but
  :math:`\eta(\dot{\gamma})` remains finite due to yield stress

The failure of Cox-Merz is a signature of yield stress behavior and is commonly observed
in soft glassy materials.

.. note:: **Practical Yield Stress Determination**

   For SGR fitting, the yield stress is often treated as:

   1. A fitting parameter in the Herschel-Bulkley form
   2. Constrained by :math:`\sigma_y = 0` for :math:`x > 1` (no glass phase)
   3. Related to :math:`G_0` and :math:`x` through scaling arguments for :math:`x < 1`

   The relationship :math:`\sigma_y \sim G_0 \gamma_c` with :math:`\gamma_c \sim O(1)` provides
   an order-of-magnitude estimate, where :math:`\gamma_c` is the characteristic yield strain.

----

Nonlinear Response
------------------

Step Strain (Large Amplitude)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For step strain of amplitude :math:`\gamma_0` at time :math:`t_w`, the stress relaxation depends nonlinearly on :math:`\gamma_0` when :math:`\gamma_0 \sim O(1)`.

The key effect is that large step strains cause immediate yielding of elements in shallow traps:

.. math::

   \text{Elements with } E < \frac{\gamma_0^2}{2} \text{ yield immediately}

The population of shallow elements is depleted, and stress relaxation proceeds from a partially "rejuvenated" state.

Large Amplitude Oscillatory Shear (LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SGR model extends naturally to nonlinear oscillatory rheology. For :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with :math:`\gamma_0 \sim O(1)`:

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.parameters.set_value('x', 1.3)
   model.parameters.set_value('G0', 100.0)
   model.parameters.set_value('tau0', 0.01)

   # Simulate LAOS response
   strain, stress = model.simulate_laos(
       gamma_0=1.0,         # Strain amplitude
       omega=1.0,           # Angular frequency (rad/s)
       n_cycles=5           # Number of oscillation cycles
   )

The stress response is decomposed using Chebyshev polynomials:

.. math::

   \sigma(\gamma, \dot{\gamma}) = \sum_{n \text{ odd}} e_n T_n(\gamma/\gamma_0) + v_n T_n(\dot{\gamma}/\dot{\gamma}_0)

where :math:`e_n` quantify elastic nonlinearity and :math:`v_n` viscous nonlinearity.

**Fourier Series Representation**

Alternatively, the periodic stress response can be decomposed into a Fourier series:

.. math::

   \sigma(t) = \gamma_0 \sum_{n=1,3,5,...} \left[ G'_n(\omega, \gamma_0) \sin(n\omega t)
   + G''_n(\omega, \gamma_0) \cos(n\omega t) \right]

where:

- :math:`G'_1, G''_1` are the fundamental (first harmonic) moduli
- :math:`G'_n, G''_n` for :math:`n > 1` are higher harmonic contributions
- Only odd harmonics appear due to the symmetry :math:`\sigma(-\gamma) = -\sigma(\gamma)`

**Higher Harmonic Generation in SGR**

The nonlinear yielding dynamics in SGR generate higher harmonics through:

1. **Strain-induced barrier lowering**: The factor :math:`\exp(\gamma^2/(2x))` in the yielding
   rate is inherently nonlinear

2. **Population redistribution**: Large strains deplete shallow traps, altering the stress response

3. **Memory effects**: The integral constitutive form couples current strain to full history

The third harmonic ratio :math:`I_{3/1} = |G^*_3|/|G^*_1|` is a common measure of nonlinearity:

.. math::

   I_{3/1}(\gamma_0) \approx \begin{cases}
   \propto \gamma_0^2 & \text{for } \gamma_0 \ll 1 \text{ (weak nonlinearity)} \\
   O(1) & \text{for } \gamma_0 \sim 1 \text{ (strong nonlinearity)}
   \end{cases}

**Lissajous-Bowditch Curves**

The parametric plot of :math:`\sigma(t)` vs :math:`\gamma(t)` reveals material nonlinearity:

.. list-table:: Lissajous Curve Interpretation
   :widths: 25 35 40
   :header-rows: 1

   * - Shape
     - Interpretation
     - SGR Regime
   * - Ellipse
     - Linear viscoelastic
     - :math:`\gamma_0 \ll 1`
   * - Rectangular (bulging)
     - Strain stiffening + viscous dissipation
     - :math:`\gamma_0 \sim 1`, :math:`x < 1`
   * - S-shaped distortion
     - Strain softening (yielding onset)
     - :math:`\gamma_0 > 1`, transition
   * - Parallelogram
     - Strong plastic yielding
     - :math:`\gamma_0 \gg 1`, fully yielded

**Chebyshev vs Fourier Decomposition**

The Chebyshev representation has advantages for physical interpretation:

- :math:`e_1`: Linear elastic modulus (recoverable deformation)
- :math:`e_3 > 0`: Strain stiffening (positive curvature in :math:`\sigma` vs :math:`\gamma`)
- :math:`e_3 < 0`: Strain softening (negative curvature)
- :math:`v_1`: Linear viscous modulus
- :math:`v_3 > 0`: Shear thickening
- :math:`v_3 < 0`: Shear thinning

For SGR with :math:`x < 1` and moderate :math:`\gamma_0`, the dominant behavior is
:math:`e_3 < 0` (strain softening due to yielding) and :math:`v_3 < 0` (shear thinning).

Over-Aging and Rejuvenation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interesting phenomena occur when large-amplitude oscillatory strain is applied to an aging system for a finite duration, and the system is then allowed to evolve before measuring the viscoelastic spectrum with a small probe strain [16]_ [17]_.

Naively, large-amplitude strain would "reset" aging, reducing the effective age. However, the situation is more subtle:

1. **Deep traps** (large :math:`E`): Elements are unaffected by the large strain
2. **Moderate traps**: Elements are forced to yield, reborn with small :math:`E`
3. **Shallow traps**: Already yielded, contribution unchanged

The result is:
   - Depletion of elements at intermediate :math:`E`
   - Excess of newly-yielded elements at small :math:`E`

During subsequent evolution:
   - Small-:math:`E` elements yield quickly and migrate to larger :math:`E`
   - The intermediate-:math:`E` depletion becomes important
   - The viscoelastic spectrum has an **enhanced** contribution from deep traps

**Over-aging**: The sample can behave as if the large oscillatory strain made it *older* rather than younger.

----

Tensorial Extensions
--------------------

The original SGR model is scalar (one-dimensional). For realistic flows involving multiple components of the stress and strain tensors, a tensorial generalization is needed [20]_.

Disordered Foams and Emulsions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tensorial SGR model represents each mesoscopic element by a deformation tensor. The element deforms affinely until yield, whereupon it relaxes to an isotropic reference state.

Key features:
   - Full stress tensor :math:`\boldsymbol{\sigma}` can be computed
   - Shear and normal stresses are coupled
   - Extensional flows can be modeled

The tensorial model is appropriate for:
   - Dense emulsions with polydisperse droplets
   - Aqueous foams
   - Colloidal pastes

**Limitation**: The original tensorial SGR model does not include stored elastic energy before the quench, making it inappropriate for materials with significant pre-stress.

----

Related Models
--------------

Several alternative frameworks describe soft glassy rheology:

Hébraud-Lequeux (HL) Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

Developed independently of SGR [38]_, the HL model uses similar concepts:

- Mesoscopic elements with local strain
- Stress-induced yielding
- Activated dynamics with effective noise

Key differences:
   - Noise arises from a diffusion process in strain space
   - Simpler mathematical structure (Fokker-Planck equation)
   - Qualitatively similar predictions for aging and flow curves

The HL model and variants [39]_ [40]_ [41]_ have been extensively compared to experimental data on colloidal suspensions.

Shear-Transformation Zone (STZ) Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Developed by Falk and Langer [43]_, STZ theory postulates:

- Plastic deformation occurs in local "shear transformation zones"
- Elements have bistable configurations
- Equations govern population evolution of configurations

STZ theory has been influential for:
   - Metallic glasses
   - Amorphous polymers
   - Dense granular materials

Connections to the SGR model are active areas of research.

Fluidity Models
~~~~~~~~~~~~~~~

Simpler models based on a single "fluidity" parameter :math:`D` (inverse viscosity) [42]_:

.. math::

   \frac{\partial D}{\partial t} = \frac{1}{\tau_D}(D_\infty - D) + \beta |\dot{\gamma}|^\alpha D

where the first term describes relaxation toward equilibrium fluidity and the second term describes shear-induced fluidization.

These models capture:
   - Thixotropy and rheopexy
   - Avalanche behavior near jamming
   - Simplified aging dynamics

----

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

What You Can Learn
------------------

The SGR model provides deep insights into the structural state and flow behavior of soft glassy materials through the lens of the effective noise temperature :math:`x` and its connection to the mesoscopic energy landscape.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**x (Effective Noise Temperature)**:
   The ratio of thermal-like noise energy to the mean trap depth, controlling the fundamental phase behavior of the material.

   *For graduate students*: :math:`x` represents the dimensionless "configurational temperature" that governs the statistical mechanics of mesoscopic elements in an energy landscape. When :math:`x < x_g = 1`, the equilibrium Boltzmann distribution :math:`P_{eq}(E) \propto \rho(E)\exp(E/x)` is non-normalizable, meaning no steady state exists and the system must age. The glass transition at :math:`x = 1` is a genuine dynamical phase transition where the relaxation time diverges.

   *For practitioners*: :math:`x < 1` means your material behaves as a solid (yield stress, aging, thixotropic recovery). :math:`1 < x < 2` means it flows but with infinite zero-shear viscosity and power-law rheology. :math:`x \geq 2` means conventional Newtonian behavior. Fitting :math:`x` from the slope of :math:`\log(G')` vs :math:`\log(\omega)` immediately classifies your material's phase.

**:math:`G_0` (Plateau Modulus)**:
   The characteristic elastic stress scale arising from local element deformation.

   *For graduate students*: :math:`G_0` sets the energy scale of elastic strain energy :math:`(\frac{1}{2}kl^2)`. In the trap model, it corresponds to the spring constant :math:`k` of mesoscopic elements. For foams and emulsions, :math:`G_0 \approx \gamma/R` where :math:`\gamma` is interfacial tension and :math:`R` is bubble/droplet radius. For colloidal glasses, :math:`G_0 \approx n k_B T` where :math:`n` is number density.

   *For practitioners*: :math:`G_0` is the high-frequency plateau in oscillatory tests (when accessible) or can be extracted from the low-rate flow stress. Typical values: 10--1000 Pa for foams, 100--10000 Pa for pastes, 1--100 Pa for colloidal glasses.

:math:`\tau_0` **(Attempt Time)**:
   The microscopic timescale for rearrangement attempts (yield events).

   *For graduate students*: :math:`\tau_0` is the inverse of the bare hopping rate :math:`\Gamma_0 = 1/\tau_0`. For Brownian systems, :math:`\tau_0 \approx \eta_s a^3/(k_B T)` where :math:`\eta_s` is solvent viscosity and :math:`a` is the element size. For non-Brownian systems, :math:`\tau_0` may reflect vibration or diffusion timescales.

   *For practitioners*: :math:`\tau_0` determines the absolute frequency scale of rheology. Fitting both :math:`\tau_0` and :math:`x` allows prediction of :math:`G'(\omega)`, :math:`G''(\omega)` across decades in frequency. Typical values: :math:`10^{-6}\text{--}10^{-2}` s for colloids, :math:`10^{-4}\text{--}10^{-1}` s for emulsions, :math:`10^{-3}\text{--}1` s for pastes.

:math:`\sigma_y` **(Yield Stress,** :math:`x < 1` **only)**:
   The dynamic yield stress emerging from non-ergodicity.

   *For graduate students*: Unlike phenomenological Bingham models, :math:`\sigma_y` in SGR arises from the divergence of the relaxation time as :math:`x \to 1`. The true yield stress is :math:`\sigma_y = \lim_{\dot{\gamma} \to 0} \sigma(\dot{\gamma}) > 0` when :math:`x < 1`, reflecting the inability of the system to equilibrate on experimental timescales. The "yieldedness" is a consequence of broken ergodicity, not a material constant.

   *For practitioners*: If :math:`x < 1` from your fit, expect a measurable yield stress. If experimental :math:`\sigma_y` is much larger than model predictions, additional physics (attractive forces, network structure) may be present beyond SGR caging.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SGR Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - x Range
     - Phase Behavior
     - Typical Materials
     - Rheological Signatures
   * - **:math:`x < 0.5`**
     - Deep glass
     - Highly concentrated pastes, aged colloidal gels, arrested emulsions
     - Large yield stress (>100 Pa), strong aging (hours-days), brittle yielding, no flow below :math:`\sigma_y`
   * - **:math:`0.5 < x < 1`**
     - Marginal glass
     - Fresh colloidal suspensions, carbopol gels, moderately concentrated foams
     - Moderate yield stress (10-100 Pa), aging on experimental timescales, ductile yielding with overshoot
   * - **:math:`1 < x < 1.5`**
     - Weak power-law fluid
     - Dilute emulsions, soft foams, near-critical suspensions
     - No true yield stress, :math:`G' \approx G'' \approx \omega^{(x-1)}` with weak frequency dependence, :math:`\tan(\delta) \approx 0.5\text{--}1.0` (constant)
   * - **:math:`1.5 < x < 2`**
     - Strong power-law fluid
     - Low-concentration surfactant solutions, polymer-colloid mixtures
     - Fluid-like (:math:`G'' > G'` at low :math:`\omega`), stronger frequency dependence, :math:`\tan(\delta) > 1` (constant), infinite viscosity but fast equilibration
   * - :math:`x \geq 2`
     - Newtonian/Maxwell liquid
     - Dilute suspensions, simple liquids, polymer solutions below overlap
     - Exponential relaxation, finite viscosity, :math:`G'' \sim \omega` at low frequencies, single Maxwell time

Connection to Aging and Rejuvenation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Aging (:math:`x < 1`)**: The waiting-time dependence of relaxation reveals:

- Characteristic timescale :math:`\tau_{\text{eff}} \sim t_w` (age-dependent)
- Over-aging phenomenon: Large-amplitude shear can make samples appear "older"
- Memory formation through trap deepening

**Rejuvenation (All x)**: Shear-induced fluidization mechanisms:

- Strain-accelerated yielding through :math:`Z(t,t') = \int \exp[\gamma(t'')^2/2x] dt''`
- Shallow-trap depletion under large amplitude strain
- Flow-induced interruption of aging process

**Practical Insight**: If x fits close to 1.0, the material is near the glass
transition and may show extreme sensitivity to preparation history, thermal
fluctuations, or slight compositional changes.

Structural Evolution
~~~~~~~~~~~~~~~~~~~~

**Trap Depth Distribution**: The exponential form :math:`\rho(E) \propto \exp(-E)`
represents the energy landscape structure:

- Broad distribution indicates heterogeneous local environments
- Mean trap depth :math:`\langle E \rangle = 1` (normalized units)
- Connection to structural relaxation heterogeneity

**Effective Time Warping**: The strain-dependent effective time:

.. math::

   Z(t, t') = \int_{t'}^t \exp\left[\frac{\gamma(s)^2}{2x}\right] ds

shows how macroscopic strain "accelerates the clock" for yielding transitions.
Large strains make the material appear to age faster (or rejuvenate, depending
on the balance of shallow vs deep trap dynamics).

Predictive Power
~~~~~~~~~~~~~~~~

From a single parameter (:math:`x`) and two scales (:math:`G_0`, :math:`\tau_0`), the SGR model predicts:

1. **Frequency-dependent moduli** with correct power-law exponents
2. **Flow curve shape** (yield stress + power-law or shear thinning)
3. **Creep compliance** functional form
4. **Stress relaxation after step strain** with aging dependence
5. **LAOS nonlinearity** through Chebyshev coefficients

**Fitting Strategy**: Start with oscillatory data to extract :math:`x` from the slope
of :math:`\log(G')` vs :math:`\log(\omega)`. Then verify consistency with flow curve and transient
experiments. Discrepancies indicate additional physics (e.g., thixotropy,
shear banding) requiring extended models.

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Affine deformation**: Local strains follow macroscopic strain (:math:`\dot{l} = \dot{\gamma}`)
2. **Mean-field**: No spatial correlations between elements
3. **Exponential trap distribution**: :math:`\rho(E) \propto e^{-E}`
4. **Activated hopping**: Arrhenius form with effective temperature :math:`x`
5. **Instantaneous yielding**: Elements jump to :math:`l = 0` upon yield (no frustration)
6. **Isothermal**: Temperature enters only through :math:`x`

Data Requirements
~~~~~~~~~~~~~~~~~

- **Oscillatory data**: Frequency sweeps of :math:`G'(\omega)`, :math:`G''(\omega)`
- **Optional**: Step strain relaxation, creep compliance, steady shear flow curves
- **For aging**: Multiple measurements at different waiting times :math:`t_w`

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

**Shear-thinning only**:
   The basic SGR model produces only shear-thinning flow curves. Modifications for
   shear-thickening lead to rheological instabilities and chaotic behavior [36]_ [37]_.

----

Extended SGR Features
---------------------

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

Thixotropy Extension
~~~~~~~~~~~~~~~~~~~~

The SGR model can be extended with a structural parameter :math:`\lambda \in [0, 1]` representing the degree of structuring:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{\tau_b} - k_d |\dot{\gamma}|^\alpha \lambda^\beta

where :math:`\tau_b` is the buildup time and :math:`k_d, \alpha, \beta` are destruction parameters.
The noise temperature is modulated: :math:`x_{\text{eff}} = x_0 + \Delta x (1 - \lambda)`.

----

Mathematical Summary
---------------------

This section provides a consolidated reference of the SGR model's predictions for standard rheological protocols.

Measurement Protocols
~~~~~~~~~~~~~~~~~~~~~

**Steady Rotation (Flow Curve)**:

.. math::
   \dot{\gamma}(t) = \dot{\gamma} = \text{constant}

**Stress Relaxation (Step Strain)**:

.. math::
   \gamma(t) = \gamma_0 H(t) \quad (\text{small step strain } \gamma_0)

**Creep (Step Stress)**:

.. math::
   \sigma(t) = \sigma_0 H(t)

**Oscillatory Shear (SAOS)**:

.. math::
   \gamma(t) = \gamma_0 e^{i\omega t}

Scaling Predictions
~~~~~~~~~~~~~~~~~~~

.. list-table:: SGR Model Scaling Predictions
   :header-rows: 1
   :widths: 20 20 60

   * - Measurement
     - Regime
     - SGR Scaling Prediction
   * - **Flow Curve**
     - Fluid (:math:`x > 1`)
     - :math:`\sigma \sim \dot{\gamma}^{x-1}`
   * -
     - Glass (:math:`x < 1`)
     - :math:`\sigma = \sigma_y + A\dot{\gamma}^{1-x}`
   * - **Relaxation**
     - Fluid (:math:`x > 1`)
     - :math:`G(t) \sim t^{-(x-1)}`
   * -
     - Glass (:math:`x < 1`)
     - :math:`G(t) \approx G_{\text{plateau}}` (aging plateau)
   * - **Creep**
     - Fluid (:math:`x > 1`)
     - :math:`J(t) \sim t^{x-1}`
   * -
     - Glass (:math:`\sigma < \sigma_y`)
     - :math:`J(t) \to \text{const}` (solid-like)
   * - **Oscillation**
     - Fluid (:math:`x > 1`)
     - :math:`G', G'' \sim \omega^{x-1}`, :math:`\tan\delta = \tan\left(\frac{(x-1)\pi}{2}\right)`

Aging Scaling (:math:`x < 1`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the glass phase, response depends on waiting time :math:`t_w`:

.. math::
   G(t, t_w) \sim \left( \frac{t}{t_w} \right)^{-(1-x)}

The characteristic relaxation time grows proportionally to the sample age: :math:`t_{\text{eff}} \sim t_w`.

----

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

If :math:`\tan\delta` is approximately constant across frequency, the material is in the SGR power-law regime.

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
   * - Noisy :math:`G'` and :math:`G''` data
     - Instrument inertia or torque limit
     - SGR applies to soft materials; check raw phase angle and torque limits.

----

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

Aging Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SGRConventional

   # Data at multiple waiting times
   waiting_times = [100, 1000, 10000]  # seconds
   omega = np.logspace(-2, 2, 50)

   results = []
   for tw in waiting_times:
       model = SGRConventional()
       model.fit(omega, G_star_data[tw], test_mode='oscillation')
       results.append({
           'tw': tw,
           'x': model.parameters.get_value('x'),
           'G0': model.parameters.get_value('G0')
       })

   # Analyze aging: x should approach 1 as tw increases for glasses
   for r in results:
       print(f"tw = {r['tw']:5d} s: x = {r['x']:.3f}, G0 = {r['G0']:.1f} Pa")

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

----

See Also
--------

- :doc:`sgr_generic` — GENERIC thermodynamic framework version with entropy production
- :doc:`../fractional/fractional_maxwell_gel` — alternative for power-law gels
- :doc:`../flow/herschel_bulkley` — simpler yield stress model for steady shear only
- :doc:`../../transforms/srfs` — Strain-Rate Frequency Superposition transform
- :doc:`../spp/spp_decomposer` — SPP decomposition for LAOS analysis

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.SGRConventional`

----

References
----------

.. [1] Sollich, P., Lequeux, F., Hébraud, P., & Cates, M. E. "Rheology of Soft Glassy Materials."
   *Physical Review Letters*, **78**, 2020-2023 (1997).
   https://doi.org/10.1103/PhysRevLett.78.2020

.. [2] Sollich, P. "Rheological constitutive equation for a model of soft glassy materials."
   *Physical Review E*, **58**, 738-759 (1998).
   https://doi.org/10.1103/PhysRevE.58.738

.. [15] Sollich, P. & Cates, M. E. "Thermodynamic interpretation of soft glassy rheology models."
   *Physical Review E*, **85**, 031127 (2012).
   https://doi.org/10.1103/PhysRevE.85.031127

.. [16] Viasnoff, V. & Lequeux, F. "Rejuvenation and overaging in a colloidal glass under shear."
   *Physical Review Letters*, **89**, 065701 (2002).
   https://doi.org/10.1103/PhysRevLett.89.065701

.. [17] Viasnoff, V., Jurine, S., & Lequeux, F. "How are colloidal suspensions that age rejuvenated by strain application?"
   *Faraday Discussions*, **123** (2003).
   https://doi.org/10.1039/B204377G

.. [19] Fielding, S. M., Sollich, P., & Cates, M. E. "Aging and rheology in soft materials."
   *Journal of Rheology*, **44**\ (2), 323-369 (2000).
   https://doi.org/10.1122/1.551088

.. [20] Cates, M. E. & Sollich, P. "Tensorial constitutive models for disordered foams, dense emulsions, and other soft nonergodic materials."
   *Journal of Rheology*, **48**\ (1), 193-207 (2004).
   https://doi.org/10.1122/1.1634985

.. [26] Cugliandolo, L. F. & Kurchan, J. "Weak ergodicity breaking in mean-field spin-glass models."
   *Philosophical Magazine B*, **71**\ (4), 501-514 (1995).
   https://doi.org/10.1080/01418639508238541

.. [27] Kurchan, J. "Rheology, and how to stop aging."
   Preprint cond-mat/9812347 (1998).

.. [34] Bouchaud, J.-P. "Weak ergodicity breaking and aging in disordered systems."
   *Journal de Physique I*, **2**\ (9), 1705-1713 (1992).
   https://doi.org/10.1051/jp1:1992238

.. [36] Head, D. A., Ajdari, A., & Cates, M. E. "Jamming, hysteresis, and oscillation in scalar models for shear thickening."
   *Physical Review E*, **64**, 061509 (2001).
   https://doi.org/10.1103/PhysRevE.64.061509

.. [37] Head, D. A., Ajdari, A., & Cates, M. E. "Rheological instability in a simple shear-thickening model."
   *Europhysics Letters*, **57**\ (1), 120-126 (2002).
   https://doi.org/10.1209/epl/i2002-00550-y

.. [38] Hébraud, P. & Lequeux, F. "Mode-coupling theory for the pasty rheology of soft glassy materials."
   *Physical Review Letters*, **81**\ (14), 2934-2937 (1998).
   https://doi.org/10.1103/PhysRevLett.81.2934

.. [39] Derec, C., Ajdari, A., & Lequeux, F. "Mechanics near a jamming transition: a minimalist model."
   *Faraday Discussions*, **112**, 195-207 (1999).
   https://doi.org/10.1039/A809307E

.. [40] Derec, C., Ajdari, A., Ducouret, G., & Lequeux, F. "Rheological characterization of aging in a concentrated colloidal suspension."
   *Comptes Rendus de l'Académie des Sciences - Series IV*, **1**\ (8), 1115-1119 (2000).
   https://doi.org/10.1016/S1296-2147(00)01106-9

.. [41] Derec, C., Ajdari, A., & Lequeux, F. "Rheology and aging: a simple approach."
   *European Physical Journal E*, **4**\ (3), 355-361 (2001).
   https://doi.org/10.1007/s101890170118

.. [42] Coussot, P., Nguyen, Q. D., Huynh, H. T., & Bonn, D. "Avalanche behavior in yield stress fluids."
   *Physical Review Letters*, **88**, 175501 (2002).
   https://doi.org/10.1103/PhysRevLett.88.175501

.. [43] Falk, M. L. & Langer, J. S. "Dynamics of viscoplastic deformation in amorphous solids."
   *Physical Review E*, **57**\ (6), 7192-7205 (1998).
   https://doi.org/10.1103/PhysRevE.57.7192

Further Reading
~~~~~~~~~~~~~~~

- Sollich, P. & Cates, M. E. "Thermodynamic interpretation of soft glassy rheology models."
  *Physical Review E*, **85**, 031127 (2012).
  https://doi.org/10.1103/PhysRevE.85.031127

- Mason, T. G., Bibette, J., & Weitz, D. A. "Elasticity of compressed emulsions."
  *Physical Review Letters*, **75**\ (10), 2051-2054 (1995).
  https://doi.org/10.1103/PhysRevLett.75.2051

- Mason, T. G. & Weitz, D. A. "Linear viscoelasticity of colloidal hard-sphere suspensions near the glass-transition."
  *Physical Review Letters*, **75**\ (14), 2770-2773 (1995).
  https://doi.org/10.1103/PhysRevLett.75.2770

- Cloitre, M., Borrega, R., & Leibler, L. "Rheological aging and rejuvenation in microgel pastes."
  *Physical Review Letters*, **85**\ (22), 4819-4822 (2000).
  https://doi.org/10.1103/PhysRevLett.85.4819

- Hohler, R., Cohen-Addad, S., & Asnacios, A. "Rheological memory effect in aqueous foam."
  *Europhysics Letters*, **48**\ (1), 93-98 (1999).
  https://doi.org/10.1209/epl/i1999-00119-4

- Bouchaud, J. P. "Weak ergodicity breaking and aging in disordered systems."
  *Journal de Physique I (France)*, **2**\ (9), 1705-1713 (1992).
  https://doi.org/10.1051/jp1:1992238

- Sollich, P. "Soft Glassy Rheology." In *Molecular Gels: Materials with
  Self-Assembled Fibrillar Networks* (eds. R. G. Weiss & P. Terech),
  pp. 161--192. Springer, Dordrecht (2006).
  https://doi.org/10.1007/1-4020-3689-2_6
  :download:`PDF <../../../reference/sollich_2005_soft_glass_rheology.pdf>`

- Struik, L. C. E. *Physical Aging in Amorphous Polymers and Other Materials*.
  Elsevier, Amsterdam (1978). ISBN: 978-0444416551
