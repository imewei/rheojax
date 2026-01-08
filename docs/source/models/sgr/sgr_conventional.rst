.. _model-sgr-conventional:

======================================================
SGR Conventional (Soft Glassy Rheology) — Handbook
======================================================

Quick Reference
---------------

**Use when:** Soft glassy materials (foams, emulsions, pastes, colloidal gels), yield stress fluids, aging materials
**Parameters:** 3-4 (x, G0, tau0, optional sigma_y)
**Key equation:** :math:`G'(\omega) \sim G''(\omega) \sim \omega^{x-1}` for :math:`1 < x < 2`
**Test modes:** Oscillation, relaxation, creep, steady shear, LAOS
**Material examples:** Concentrated emulsions, colloidal suspensions, foams, pastes, mayonnaise, hair gel

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

Because energy barriers in soft materials are large compared to :math:`k_B T` (by factors of :math:`10^3`–:math:`10^6`), the effective noise :math:`x` is of order the mean barrier height :math:`\langle E \rangle`, not :math:`k_B T`. This "macroscopic" effective temperature remains nonzero even as :math:`k_B T \to 0`, consistent with theories of out-of-equilibrium systems with slow dynamics [26]_ [27]_.

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

Constitutive Equations
----------------------

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

Linear Response Regime
~~~~~~~~~~~~~~~~~~~~~~

When local strains are negligible (:math:`\gamma \ll 1`), the effective time interval becomes the actual time interval:

.. math::

   Z(t, t') \to t - t'

and the hopping rate :math:`Y(t)` becomes strain-independent. The stress response to any strain history follows directly from the constitutive equations.

For step strain with amplitude :math:`\gamma_0 \ll 1`, the linearized step strain response is:

.. math::

   G(t - t_w, t_w) = 1 - \int_{t_w}^{t} Y(t') G_\rho(t - t') \, dt'

Equilibrium (x > 1)
~~~~~~~~~~~~~~~~~~~

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

----

Rheological Aging: Strain-Controlled Experiments
------------------------------------------------

For :math:`x < 1` (glass phase), the system exhibits true aging. Here we detail the predictions for strain-controlled experiments.

Evolution of the Lifetime Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following a quench (sudden reduction of :math:`x` from a large value to :math:`x < 1` at time zero), the distribution of yield energies :math:`P(E, t)` evolves:

.. math::

   P(E, t) = P_0(E) \exp[-t/\tau(E)] + \int_0^t Y(t') \rho(E) \exp[-(t - t')/\tau(E)] \, dt'

with :math:`\tau(E) = \exp(E/x)`. Equivalently, the lifetime distribution :math:`P(\tau, t_w)` at waiting time :math:`t_w`:

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

Startup Shear (Linear Regime)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For steady shear at rate :math:`\dot{\gamma} \ll 1` started at time :math:`t_w`, with total strain :math:`\gamma = \dot{\gamma}(t - t_w) \ll 1`:

**For** :math:`x < 1`:

.. math::

   \sigma(t) \sim \dot{\gamma}(t - t_w) \quad \text{(purely elastic)}

**For** :math:`1 < x < 2`:

.. math::

   \sigma(t) \sim \dot{\gamma}(t - t_w)^{2-x} \quad \text{(anomalous power law)}

**For** :math:`x > 2`:

.. math::

   \sigma(t) \sim \dot{\gamma} \quad \text{(Newtonian)}

These scalings are independent of whether :math:`t - t_w \ll t_w` or :math:`t - t_w \gg t_w`, so linear startup experiments are not a useful probe of aging dynamics.

----

Rheological Aging: Stress-Controlled Experiments
------------------------------------------------

Stress-controlled experiments (creep) provide complementary information to strain-controlled tests. These are more difficult to analyze because the SGR constitutive equations naturally express stress as a function of strain history.

Creep Compliance
~~~~~~~~~~~~~~~~

Under constant stress :math:`\sigma_0` applied at time :math:`t_w`, the strain evolves as:

.. math::

   \gamma(t) = J(t - t_w, t_w) \sigma_0

For :math:`1 < x < 2`, the creep compliance exhibits power-law behavior:

.. math::

   J(t - t_w, t_w) = J_0 \left[1 + \left(\frac{t - t_w}{\tau_0}\right)^{x-1} E_{x-1,x}\left(\left(\frac{t - t_w}{\tau_0}\right)^{x-1}\right)\right]

where :math:`E_{\alpha,\beta}(z)` is the generalized Mittag-Leffler function.

For :math:`x < 1` (glass phase), the material exhibits **power-law creep**:

.. math::

   \gamma(t) \sim (t - t_w)^{x-1}

with exponent :math:`0 < x - 1 < 0` (sublinear growth), but no steady-state strain rate is reached.

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

The stress response is decomposed using Chebyshev polynomials:

.. math::

   \sigma(\gamma, \dot{\gamma}) = \sum_{n \text{ odd}} e_n T_n(\gamma/\gamma_0) + v_n T_n(\dot{\gamma}/\dot{\gamma}_0)

where :math:`e_n` quantify elastic nonlinearity and :math:`v_n` viscous nonlinearity.

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
   * - Noisy G' and G'' data
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

.. [16] Viasnoff, V. & Lequeux, F. "Rejuvenation and overaging in a colloidal glass under shear."
   *Physical Review Letters*, **89**, 065701 (2002).
   https://doi.org/10.1103/PhysRevLett.89.065701

.. [17] Viasnoff, V., Jurine, S., & Lequeux, F. "How colloidal suspensions that age are rejuvenated by strain application."
   *Faraday Discussions*, **123** (2003).

.. [19] Fielding, S. M., Sollich, P., & Cates, M. E. "Aging and rheology in soft materials."
   *Journal of Rheology*, **44**\ (2), 323-369 (2000).
   https://doi.org/10.1122/1.551088

.. [20] Cates, M. E. & Sollich, P. "Tensorial constitutive models for disordered foams, dense emulsions, and other soft nonergodic materials."
   *Journal of Rheology*, **48**\ (1), 193-207 (2004).
   https://doi.org/10.1122/1.1634987

.. [26] Cugliandolo, L. F. & Kurchan, J. "Weak ergodicity breaking in mean-field spin-glass models."
   *Philosophical Magazine B*, **71**\ (4), 501-514 (1995).

.. [27] Kurchan, J. "Rheology, and how to stop aging."
   Preprint cond-mat/9812347 (1998).

.. [36] Head, D. A., Ajdari, A., & Cates, M. E. "Jamming, hysteresis, and oscillation in scalar models for shear thickening."
   *Physical Review E*, **64**, 061509 (2001).

.. [37] Head, D. A., Ajdari, A., & Cates, M. E. "Rheological instability in a simple shear-thickening model."
   *Europhysics Letters*, **57**\ (1), 120-126 (2002).

.. [38] Hébraud, P. & Lequeux, F. "Mode-coupling theory for the pasty rheology of soft glassy materials."
   *Physical Review Letters*, **81**\ (14), 2934-2937 (1998).
   https://doi.org/10.1103/PhysRevLett.81.2934

.. [39] Derec, C., Ajdari, A., & Lequeux, F. "Mechanics near a jamming transition: a minimalist model."
   *Faraday Discussions*, **112**, 195-207 (1999).

.. [40] Derec, C., Ajdari, A., Ducouret, G., & Lequeux, F. "Rheological characterization of aging in a concentrated colloidal suspension."
   *Comptes Rendus de l'Académie des Sciences - Series IV*, **1**\ (8), 1115-1119 (2000).

.. [41] Derec, C., Ajdari, A., & Lequeux, F. "Rheology and aging: a simple approach."
   *European Physical Journal E*, **4**\ (3), 355-361 (2001).

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

- Mason, T. G. & Weitz, D. A. "Linear viscoelasticity of colloidal hard-sphere suspensions near the glass-transition."
  *Physical Review Letters*, **75**\ (14), 2770-2773 (1995).

- Cloître, M., Borrega, R., & Leibler, L. "Rheological aging and rejuvenation in microgel pastes."
  *Physical Review Letters*, **85**\ (22), 4819-4822 (2000).

- Höhler, R., Cohen-Addad, S., & Asnacios, A. "Rheological memory effect in aqueous foam."
  *Europhysics Letters*, **48**\ (1), 93-98 (1999).

- Bouchaud, J. P. "Weak ergodicity breaking and aging in disordered systems."
  *Journal de Physique I (France)*, **2**\ (9), 1705-1713 (1992).

- Struik, L. C. E. *Physical Aging in Amorphous Polymers and Other Materials*.
  Elsevier, Houston (1978).
