.. _model-itt-mct-schematic:

ITT-MCT Schematic (F_1_2)
=======================

Quick Reference
---------------

- **Use when:** Dense colloidal suspensions, glassy materials, yield-stress fluids,
materials showing glass transition behavior

- **Parameters:** 5 (:math:`v_1`, :math:`v_2`, :math:`\Gamma`, :math:`\gamma_c`, :math:`G_\infty`) or equivalently (:math:`\varepsilon`, :math:`v_1`, :math:`\Gamma`, :math:`\gamma_c`, :math:`G_\infty`)

- **Key equation:** Memory kernel :math:`m(\Phi) = v_1\Phi + v_2\Phi^2` with glass transition at :math:`v_2 = 4`

- **Test modes:** Flow curve, oscillation, startup, creep, relaxation, LAOS

- **Material examples:** PMMA colloids, emulsions, carbopol gels, concentrated polymer solutions

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\Phi(t)`
     - Density correlator (normalized autocorrelation function)
   * - :math:`\Phi(t,t')`
     - Two-time correlator under shear (advected)
   * - :math:`m(\Phi)`
     - Memory kernel, :math:`m = v_1\Phi + v_2\Phi^2`
   * - :math:`v_1, v_2`
     - Vertex coefficients (coupling constants)
   * - :math:`v_{2,c}`
     - Critical vertex coefficient (= 4 for :math:`v_1 = 0`)
   * - :math:`\varepsilon`
     - Separation parameter, :math:`\varepsilon = (v_2 - v_{2,c})/v_{2,c}`
   * - :math:`\Gamma`
     - Bare relaxation rate (1/s)
   * - :math:`\gamma_c`
     - Critical strain for cage breaking (dimensionless)
   * - :math:`h(\gamma)`
     - Strain decorrelation function (Gaussian or Lorentzian form)
   * - :math:`G_\infty`
     - High-frequency (instantaneous) modulus (Pa)
   * - :math:`f`
     - Non-ergodicity parameter (glass plateau height)
   * - :math:`\sigma_y`
     - Dynamic yield stress (Pa)

Overview
--------

The :math:`F_{12}` schematic model is a simplified Mode-Coupling Theory (MCT) that captures
the essential physics of the colloidal glass transition with minimal parameters.

**Historical Context:**

MCT was developed in the 1980s by Götze and collaborators to describe the dynamics
of supercooled liquids and dense colloids. The full MCT involves coupled integro-
differential equations for density correlators at all wave vectors :math:`k`. The schematic
:math:`F_{12}` model reduces this to a single scalar equation by replacing the :math:`k`-dependent
memory kernel with a polynomial form.

**The Cage Effect:**

In dense suspensions, each particle is "caged" by its neighbors. At short times,
particles rattle within their cages (:math:`\beta`-relaxation). At long times, cooperative
rearrangements allow cage escape (:math:`\alpha`-relaxation). The glass transition occurs when
the :math:`\alpha`-relaxation time diverges - particles become permanently trapped.

**Why "** :math:`F_{12}` **":**

The name comes from the memory kernel having terms proportional to :math:`\Phi^1` and :math:`\Phi^2` ---
the "1-2" notation. This quadratic form is the simplest that captures the feedback
mechanism responsible for the glass transition.

Physical Foundations
--------------------

The Cage Effect in Dense Suspensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a dilute suspension, particles diffuse freely with diffusion coefficient:

.. math::

   D_0 = \frac{k_B T}{6\pi\eta_s a}

where :math:`a` is the particle radius and :math:`\eta_s` is the solvent viscosity.

At high volume fractions (:math:`\phi > 0.4`), particles begin to interfere with each other's
motion. The "cage" of nearest neighbors slows down diffusion:

.. math::

   D_{\text{long}} = D_0 / S(0)

where S(0) is the zero-wavevector structure factor (compressibility).

As :math:`\phi \to \phi_g \approx 0.516`, the cage becomes so strong that particles cannot escape on
any experimental timescale - the system is a glass.

Green-Kubo and the Memory Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamics can be described through the density correlator:

.. math::

   \Phi(k,t) = \frac{\langle \rho_k(t) \rho_{-k}(0) \rangle}{\langle |\rho_k|^2 \rangle}

which measures how density fluctuations at wave vector :math:`k` decorrelate over time.

The equation of motion involves a memory kernel:

.. math::

   \ddot{\Phi}(t) + \Omega^2 \left[ \Phi(t) + \int_0^t m(t-s) \dot{\Phi}(s) ds \right] = 0

The memory kernel m(t) encodes how the cage "remembers" past configurations.
This is the **Zwanzig-Mori projection** of the full dynamics.

MCT Approximation
~~~~~~~~~~~~~~~~~

MCT makes a specific approximation for the memory kernel:

.. math::

   m(k,t) = \sum_{q} V(k,q,|\mathbf{k}-\mathbf{q}|) \Phi(q,t) \Phi(|\mathbf{k}-\mathbf{q}|,t)

The vertex V encodes how density fluctuations at different length scales couple.
This "mode-coupling" gives the theory its name.

Memory Kernel Structure
~~~~~~~~~~~~~~~~~~~~~~~

The MCT memory kernel has a **bilinear structure** arising from the mode-coupling
approximation:

.. math::

   m_{\mathbf{q}}(t,s,t') = \int \frac{d^3k}{(2\pi)^3}\;
   V_{\mathbf{q},\mathbf{k},\mathbf{p}}(t,s,t')\;
   \Phi_{\mathbf{k}}(t,s)\,\Phi_{\mathbf{p}}(t,s)

where :math:`\mathbf{p} = \mathbf{q} - \mathbf{k}`.

For the schematic :math:`F_{12}` model, this complicated :math:`k`-space integral is replaced by a
polynomial approximation:

.. math::

   m(\Phi) = v_1 \Phi + v_2 \Phi^2

The quadratic term (:math:`v_2\Phi^2`) is essential for the glass transition --- it creates the
feedback mechanism where slow relaxation leads to stronger caging, which leads
to even slower relaxation.

The :math:`F_{12}` Schematic Model
----------------------------------

Reduction to Scalar Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the schematic model, we ignore the :math:`k`-dependence and write:

.. math::

   \partial_t \Phi + \Gamma \left[ \Phi + \int_0^t m(\Phi(s)) \partial_s \Phi(s) ds \right] = 0

with the polynomial memory kernel:

.. math::

   m(\Phi) = v_1 \Phi + v_2 \Phi^2

**Why this form works:**

1. The linear term (:math:`v_1\Phi`) allows for standard viscoelastic relaxation
2. The quadratic term (:math:`v_2\Phi^2`) creates the feedback: slow relaxation :math:`\to` strong cage :math:`\to` slower relaxation
3. Together, they capture the divergence of the relaxation time at the glass transition

Glass Transition Criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~

The glass transition occurs when the correlator has a non-zero long-time limit:

.. math::

   f = \lim_{t \to \infty} \Phi(t) > 0 \quad \text{(glass)}

This happens when the self-consistent equation:

.. math::

   f = m(f) = v_1 f + v_2 f^2

has a non-zero solution, i.e., when:

.. math::

   v_2 > v_{2,c} = \frac{4}{(1-v_1)^2}

For :math:`v_1 = 0`: :math:`v_{2,c} = 4`.

The **separation parameter** :math:`\varepsilon` measures distance from the transition:

.. math::

   \varepsilon = \frac{v_2 - v_{2,c}}{v_{2,c}}

- :math:`\varepsilon < 0`: Ergodic fluid (:math:`\Phi \to 0` at long times)
- :math:`\varepsilon = 0`: Critical point (power-law decay)
- :math:`\varepsilon > 0`: Glass state (:math:`\Phi \to f > 0`)

Two-Step Relaxation
~~~~~~~~~~~~~~~~~~~

Near the glass transition, the correlator shows characteristic two-step decay:

1. :math:`\beta` **-relaxation** (short times): Initial decay to a plateau

   .. math::

      \Phi(t) \approx f_c + h \cdot (t/t_0)^{-a}

2. **Plateau regime**: Correlator "stuck" near :math:`f`

3. :math:`\alpha` **-relaxation** (long times): Final decay from plateau

   .. math::

      \Phi(t) \approx f \cdot \exp\left[-(t/\tau_\alpha)^b\right]

The MCT exponents :math:`a` and :math:`b` are related to the "exponent parameter" :math:`\lambda`:

.. math::

   \frac{\Gamma(1-a)^2}{\Gamma(1-2a)} = \frac{\Gamma(1+b)^2}{\Gamma(1+2b)} = \lambda

For :math:`F_{12}` with :math:`v_1 = 0`: :math:`\lambda = 1`.

Integration Through Transients (ITT)
------------------------------------

Extending MCT to Flow
~~~~~~~~~~~~~~~~~~~~~

Under shear, density fluctuations are "advected" by the flow.

General Deformation Gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a general homogeneous flow with velocity gradient tensor
:math:`\boldsymbol{\kappa}(t) = \nabla\mathbf{v}(t)`, the deformation gradient
from time :math:`t'` to :math:`t` is:

.. math::

   \mathbf{E}(t,t') = \mathcal{T}\exp\left(\int_{t'}^{t}\boldsymbol{\kappa}(s)\,ds\right)

where :math:`\mathcal{T}` denotes time ordering.

Wavevectors are **back-advected** as:

.. math::

   \mathbf{q}(t,t') = \mathbf{q} \cdot \mathbf{E}^{-1}(t,t')

For simple shear (flow in x, gradient in y):

.. math::

   k_x(t,t') = k_x - k_y \int_{t'}^t \dot{\gamma}(s) ds = k_x - k_y \gamma(t,t')

This advection destroys the cage structure, leading to flow-induced relaxation.

Two-Time Correlator Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **transient density correlator** under shear is:

.. math::

   \Phi_{\mathbf{q}}(t,t') = \frac{\langle \rho_{\mathbf{q}(t,t')}(t)\,
   \rho_{-\mathbf{q}}(t') \rangle}{N S(q)}

This measures how density fluctuations at wavevector :math:`\mathbf{q}` decorrelate
between times :math:`t'` and :math:`t`, accounting for the fact that the wavevector
is advected by the flow.

**Physical interpretation**: The correlator tracks the "memory" of the cage
structure. Under shear, this memory is progressively destroyed as accumulated
strain :math:`\gamma(t,t')` increases.

Zwanzig-Mori Equation Detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full ITT-MCT correlator dynamics obey the integro-differential equation:

.. math::

   \partial_t \Phi_{\mathbf{q}}(t,t') +
   \Gamma_{\mathbf{q}}(t,t')\left[
   \Phi_{\mathbf{q}}(t,t') +
   \int_{t'}^{t} ds\; m_{\mathbf{q}}(t,s,t')\;\partial_s \Phi_{\mathbf{q}}(s,t')
   \right] = 0

with overdamped initial decay rate:

.. math::

   \Gamma_{\mathbf{q}}(t,t') = D_0\,\frac{q(t,t')^2}{S(q(t,t'))}

where :math:`D_0` is the bare diffusion coefficient. The advected wavevector
magnitude :math:`q(t,t')` increases with strain, accelerating the initial decay.

Strain Decorrelation Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The advected correlator is:

.. math::

   \Phi(t,t') = \Phi_{\text{eq}}(t-t') \cdot h(\gamma(t,t'))

where the **strain decorrelation function** captures how accumulated strain
breaks down the correlation. Two functional forms are available:

**Gaussian (default):** Fast exponential decay at large strains

.. math::

   h(\gamma) = \exp\left[-\left(\frac{\gamma}{\gamma_c}\right)^2\right]

**Lorentzian:** Slower algebraic decay (Brader et al. 2008)

.. math::

   h(\gamma) = \frac{1}{1 + (\gamma/\gamma_c)^2}

Physical interpretation:

- At :math:`\gamma = 0`: :math:`h = 1` (full correlation)
- At :math:`\gamma \gg \gamma_c`: :math:`h \to 0` (cage is destroyed)
- :math:`\gamma_c \approx 0.1` corresponds to the "cage strain"

**Choosing between forms:** The Gaussian form (default) is most common in the
ITT-MCT literature and gives faster decay. The Lorentzian form may better
capture materials with extended yielding transitions or gradual cage breaking.
Use the ``decorrelation_form`` parameter to select:

.. code-block:: python

   # Default Gaussian
   model = ITTMCTSchematic(epsilon=0.05)

   # Lorentzian for extended yielding
   model = ITTMCTSchematic(epsilon=0.05, decorrelation_form="lorentzian")

Generalized Green-Kubo Relation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The stress under shear is given by a generalized Green-Kubo formula:

.. math::

   \sigma(t) = \dot{\gamma}(t) \int_0^t G(t,t') dt' + \int_0^t \dot{\gamma}(t') G(t,t') dt'

where the time-dependent modulus:

.. math::

   G(t,t') = G_\infty \Phi(t,t')

Memory Kernel Forms
~~~~~~~~~~~~~~~~~~~

The memory kernel can be computed in two forms, controlled by the ``memory_form``
parameter:

**Simplified (default):** Single strain decorrelation

.. math::

   m(\Phi) = h[\gamma_{\text{acc}}] \times (v_1 \Phi + v_2 \Phi^2)

Here, a single decorrelation factor :math:`h[\gamma_{\text{acc}}]` accounts for
all strain-induced cage breaking since flow started.

**Full:** Two-time strain decorrelation (Fuchs & Cates 2002)

.. math::

   m(t,s,t_0) = h[\gamma(t,t_0)] \times h[\gamma(t,s)] \times (v_1 \Phi + v_2 \Phi^2)

The full form includes two decorrelation factors:

- :math:`h[\gamma(t,t_0)]`: How much the cage has broken since flow started
- :math:`h[\gamma(t,s)]`: How much the cage breaks during the memory integral (from time s to t)

This captures the physical effect that correlations at earlier times (larger s)
have experienced more strain-induced decorrelation than recent correlations.

.. code-block:: python

   # Default simplified form (backward compatible)
   model = ITTMCTSchematic(epsilon=0.05)

   # Full two-time memory kernel
   model = ITTMCTSchematic(epsilon=0.05, memory_form="full")

**When to use full form:** The full form is more physically accurate for
strongly driven systems where the memory integral spans multiple decorrelation
timescales. The simplified form is computationally faster and often sufficient
for qualitative predictions.

Stress Computation Forms
~~~~~~~~~~~~~~~~~~~~~~~~

The stress computation can use two approaches, controlled by the ``stress_form``
parameter:

**Schematic (default):** Uses a proxy relation

.. math::

   \sigma = G_\infty \dot{\gamma} \int_0^t \Phi(t')^2 \, h(\gamma) \, dt'

This gives physically reasonable results with the single modulus parameter
:math:`G_\infty`.

**Microscopic:** Includes wave-vector integration with structure factor weighting

.. math::

   \sigma = \frac{k_B T}{60\pi^2} \int_0^\infty dk \, k^4
   \left[\frac{S'(k)}{S(k)^2}\right]^2 \Phi^2

The microscopic form uses the Percus-Yevick structure factor :math:`S(k)` to weight
contributions from different length scales. This provides quantitative stress
predictions in physical units when the volume fraction is known.

.. code-block:: python

   # Default schematic form
   model = ITTMCTSchematic(epsilon=0.05)

   # Microscopic stress with Percus-Yevick S(k)
   model = ITTMCTSchematic(
       epsilon=0.05,
       stress_form="microscopic",
       phi_volume=0.5,      # Volume fraction for S(k)
       k_BT=4.11e-21,       # Room temperature in Joules (optional)
   )

**Note:** When ``stress_form="microscopic"``, the ``phi_volume`` parameter is
required. This is the colloidal volume fraction used to compute :math:`S(k)` via the
Percus-Yevick approximation.

**When to use microscopic form:** Use microscopic stress when you need
quantitative stress predictions tied to physical parameters (temperature,
volume fraction). The schematic form is simpler and sufficient for qualitative
comparisons or when fitting the modulus :math:`G_\infty` as a free parameter.

Combining Memory and Stress Forms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both forms can be combined for maximum physical accuracy:

.. code-block:: python

   # Full physics: two-time memory + microscopic stress
   model = ITTMCTSchematic(
       epsilon=0.1,
       decorrelation_form="gaussian",
       memory_form="full",
       stress_form="microscopic",
       phi_volume=0.55,
       k_BT=4.11e-21,
   )

   # Predict flow curve
   gamma_dot = np.logspace(-3, 3, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

The model's ``__repr__`` shows all form selections:

.. code-block:: python

   >>> model
   ITTMCTSchematic(ε=0.100 [glass], v₂=4.40, h(γ)=gaussian, m=full, σ=microscopic, G_inf=1.00e+06 Pa)

Governing Equations
-------------------

Flow Curve (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

At steady state with constant :math:`\dot{\gamma}`, the stress is:

.. math::

   \sigma_{ss} = \dot{\gamma} \int_0^\infty G(s) \cdot h(\dot{\gamma} s) ds

where :math:`G(s) = G_\infty \Phi_{\text{eq}}(s)` is the equilibrium relaxation modulus.

**Yield stress** (glass state, :math:`\varepsilon > 0`):

.. math::

   \sigma_y = \lim_{\dot{\gamma} \to 0} \sigma_{ss} = G_\infty \gamma_c f

**Shear thinning**: As :math:`\dot{\gamma}` increases, the cage is broken faster, and the effective
viscosity decreases:

.. math::

   \eta_{\text{eff}} = \sigma / \dot{\gamma}

Small Amplitude Oscillation (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small strain :math:`\gamma_0 \ll \gamma_c`, we can linearize around equilibrium:

.. math::

   G^*(\omega) = i\omega \int_0^\infty G_{\text{eq}}(t) e^{-i\omega t} dt

This gives the storage and loss moduli:

.. math::

   G'(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \sin(\omega t) dt

.. math::

   G''(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \cos(\omega t) dt

**Glass plateau**: For :math:`\varepsilon > 0`, :math:`G'(\omega \to 0) \to G_\infty f` (non-zero plateau).

Startup Flow
~~~~~~~~~~~~

Starting from rest with constant :math:`\dot{\gamma}`:

.. math::

   \sigma(t) = \dot{\gamma} \int_0^t G(t-s) \cdot h(\dot{\gamma}(t-s)) ds

This shows a characteristic **stress overshoot** when :math:`\dot{\gamma} \tau_{\alpha} > 1`, where :math:`\tau_{\alpha}`
is the structural relaxation time.

Creep
~~~~~

At constant applied stress :math:`\sigma_0`, the strain rate adjusts to maintain:

.. math::

   \sigma_0 = \int_0^t \dot{\gamma}(t') G(t,t') dt'

In the glass state (:math:`\sigma_0 < \sigma_y`): bounded deformation (solid-like).
Above yield (:math:`\sigma_0 > \sigma_y`): continuous flow (fluidization).

This leads to **viscosity bifurcation** - a sharp transition between solid
and fluid behavior at the yield stress.

Stress Relaxation
~~~~~~~~~~~~~~~~~

After cessation of flow at :math:`t = 0`:

.. math::

   \sigma(t) = \sigma(0) \cdot \Phi_{\text{relax}}(t)

In the glass state, stress does not fully relax:

.. math::

   \lim_{t \to \infty} \sigma(t) = \sigma_{\text{res}} > 0

LAOS
~~~~

For large amplitude oscillatory shear :math:`\gamma(t) = \gamma_0 \sin(\omega t)`:

The stress is non-sinusoidal and can be decomposed into Fourier harmonics:

.. math::

   \sigma(t) = \sum_{n=1,3,5,...} [\sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t)]

Higher harmonics (:math:`n = 3, 5, \ldots`) quantify nonlinearity. The ratio :math:`\sigma_3/\sigma_1`
increases with :math:`\gamma_0/\gamma_c`.

Parameters
----------

.. list-table::
   :widths: 15 15 15 15 40
   :header-rows: 1

   * - Name
     - Default
     - Bounds
     - Units
     - Physical Meaning
   * - :math:`v_1`
     - 0.0
     - (0, 5)
     - ---
     - Linear vertex coefficient. Usually 0 for pure :math:`F_{12}`.
   * - :math:`v_2`
     - 2.0
     - (0.5, 10)
     - ---
     - Quadratic vertex coefficient. Glass at :math:`v_2 > 4`.
   * - :math:`\Gamma`
     - 1.0
     - (:math:`10^{-6}`, :math:`10^6`)
     - 1/s
     - Bare relaxation rate. Sets microscopic timescale.
   * - :math:`\gamma_c`
     - 0.1
     - (0.01, 0.5)
     - ---
     - Critical strain for cage breaking. Typically 0.05--0.2.
   * - :math:`G_\infty`
     - :math:`10^6`
     - (1, :math:`10^{12}`)
     - Pa
     - High-frequency elastic modulus.

**Alternative parameterization with** :math:`\varepsilon` **:**

Instead of specifying :math:`v_2` directly, use the separation parameter :math:`\varepsilon`:

.. code-block:: python

   model = ITTMCTSchematic(epsilon=0.1)  # Glass state
   model = ITTMCTSchematic(epsilon=-0.1)  # Fluid state

This automatically sets :math:`v_2 = v_{2,c} \times (1 + \varepsilon)`.

Typical Parameter Values
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 15 15 25
   :header-rows: 1

   * - System
     - :math:`\varepsilon`
     - :math:`\gamma_c`
     - :math:`G_\infty` (Pa)
     - Notes
   * - PMMA colloids (:math:`\phi = 0.55`)
     - 0.1
     - 0.08
     - :math:`10^2`
     - Hard-sphere reference
   * - Carbopol microgels
     - 0.05
     - 0.15
     - :math:`10^3`
     - Soft particles
   * - Mayonnaise
     - 0.02
     - 0.10
     - :math:`10^2`
     - Dense emulsion
   * - Silica suspensions
     - 0.15
     - 0.05
     - :math:`10^4`
     - Strong glass

Validity and Assumptions
------------------------

**When the model works well:**

- Dense suspensions (:math:`\phi > 0.4` for hard spheres)
- Near the glass transition (:math:`|\varepsilon| < 0.3`)
- Monodisperse or narrow size distribution
- No attractive interactions (hard-sphere-like)
- Brownian timescales (colloidal, not granular)

**Limitations:**

- Does not capture crystallization
- Underestimates relaxation times in deeply supercooled regime
- No hopping/activated processes (important at low :math:`T` or high :math:`\varepsilon`)
- Assumes isotropic structure (no shear-induced ordering)

What You Can Learn
------------------

The ITT-MCT model provides quantitative predictions of glass transition behavior through the lens of density correlators and cage dynamics. The separation parameter :math:`\varepsilon` and critical strain :math:`\gamma_c` are the key diagnostics.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\varepsilon` **(Separation Parameter)**:
   The distance from the glass transition, defined as :math:`\varepsilon = (v_2 - v_{2,c})/v_{2,c}` where :math:`v_{2,c} = 4` for the :math:`F_{12}` model.

   *For graduate students*: :math:`\varepsilon` is the control parameter in the MCT bifurcation analysis. At :math:`\varepsilon = 0`, the self-consistent equation :math:`f = v_1 f + v_2 f^2` undergoes a fold bifurcation, creating a non-zero long-time limit :math:`f > 0` for :math:`\varepsilon > 0`. The :math:`\alpha`-relaxation time diverges as :math:`\tau_\alpha \sim |\varepsilon|^{-\gamma}` with :math:`\gamma \approx 2.5` (MCT universal exponent). The power-law exponents :math:`a` and :math:`b` for :math:`\beta`-relaxation and :math:`\alpha`-relaxation are determined by the exponent parameter :math:`\lambda = \Gamma(1-a)^2 / \Gamma(1-2a)`.

   *For practitioners*: :math:`\varepsilon < 0` means fluid (full relaxation), :math:`\varepsilon > 0` means glass (permanent caging). Fitting :math:`\varepsilon` from oscillatory or flow curve data immediately tells you if the material has a yield stress. :math:`\varepsilon \approx 0.1` is a typical moderately strong glass, :math:`\varepsilon \approx 0.5` is a very strong glass. Near :math:`\varepsilon = 0`, expect extreme slowing down and sensitivity to temperature or concentration.

:math:`\gamma_c` **(Critical Strain)**:
   The strain scale at which the cage structure is destroyed by shear.

   *For graduate students*: :math:`\gamma_c` appears in the strain decorrelation function :math:`h(\gamma) = \exp[-(\gamma/\gamma_c)^2]`, which describes how accumulated strain breaks down density correlations. Physically, :math:`\gamma_c` is related to the Lindemann criterion: when a particle is displaced by :math:`{\sim}\gamma_c` times the cage size (:math:`\approx` particle diameter), the cage loses memory of its initial configuration. For hard spheres, :math:`\gamma_c \approx 0.05\text{--}0.1` corresponds to the amplitude of thermal vibrations in the cage.

   *For practitioners*: :math:`\gamma_c` controls the onset of shear thinning in flow curves. Smaller :math:`\gamma_c` means the material yields more easily under strain. Fitting :math:`\gamma_c` from the crossover shear rate :math:`\dot{\gamma}^*` (where viscosity drops) via :math:`\dot{\gamma}^* \approx 1/(\tau_\alpha \gamma_c)` reveals the cage stiffness.

:math:`v_1, v_2` **(Vertex Coefficients)**:
   The mode-coupling constants determining the memory kernel :math:`m(\Phi) = v_1 \Phi + v_2 \Phi^2`.

   *For graduate students*: :math:`v_1` and :math:`v_2` arise from the :math:`k`-space convolution integral in the full MCT vertex :math:`V(k,q,|k-q|) \propto S(k)S(q)S(|k-q|)[\mathbf{k}\cdot\mathbf{q}\, c(q)/k^2 + \mathbf{k}\cdot\mathbf{p}\, c(p)/k^2]^2`. The :math:`F_{12}` schematic replaces this with a polynomial approximation. For pure :math:`F_{12}`, :math:`v_1 = 0` and :math:`v_2` controls the distance from the glass transition. Higher :math:`v_2` means stronger coupling :math:`\to` stronger caging :math:`\to` higher glass transition.

   *For practitioners*: Usually keep :math:`v_1 = 0` (default) and fit only :math:`v_2` or equivalently :math:`\varepsilon`. If :math:`v_1 \neq 0`, the critical point shifts: :math:`v_{2,c} = 4/(1-v_1)^2`. Only adjust :math:`v_1` if the model fails with :math:`v_1 = 0`.

:math:`\Gamma` **(Bare Relaxation Rate)**:
   The inverse microscopic timescale, :math:`\Gamma = 1/\tau_0`.

   *For graduate students*: In MCT, :math:`\Gamma(k) = k^2 D_0 / S(k)` is the bare (non-interacting) relaxation rate for mode k. For the schematic model, :math:`\Gamma` is the average rate controlling the short-time :math:`\beta`-relaxation. It sets the absolute timescale: all relaxation times scale as :math:`\Gamma^{-1}`.

   *For practitioners*: :math:`\Gamma` determines the high-frequency behavior in oscillatory tests. From the crossover frequency :math:`\omega^*` in :math:`G'(\omega)`, estimate :math:`\Gamma \approx \omega^*`. Typical values: :math:`10^3`-:math:`10^6` s\ :sup:`-1` for colloids (diffusion-limited), :math:`10^{-2}`-:math:`10^2` s\ :sup:`-1` for pastes.

:math:`G_\infty` **(High-Frequency Modulus)**:
   The elastic modulus at frequencies above all relaxation processes.

   *For graduate students*: :math:`G_\infty` is the plateau modulus in the schematic stress formula :math:`\sigma = G_\infty \int \Phi^2 h(\gamma) \, dt'`. It corresponds to the :math:`k`-space integral :math:`G_\infty = (k_B T / 60\pi^2) \int dk \, k^4 [S'(k)/S(k)^2]^2` in the full MCT. For hard spheres, :math:`G_\infty \approx n k_B T` where :math:`n` is number density.

   *For practitioners*: :math:`G_\infty` is fitted from the high-frequency plateau in :math:`G'(\omega)` or from the yield stress magnitude. Unlike phenomenological models, :math:`G_\infty` has a microscopic interpretation tied to particle stiffness and number density.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from ITT-MCT Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\varepsilon` Range
     - Glass State
     - Typical Materials
     - Flow Characteristics
   * - :math:`\varepsilon` **< -0.2**
     - Deep fluid
     - Dilute colloids (:math:`\phi < 0.4`), weak suspensions
     - No yield stress, Newtonian or weakly shear-thinning, :math:`G'' > G'` at all :math:`\omega`
   * - **-0.2 <** :math:`\varepsilon` **< 0**
     - Near-critical fluid
     - Moderate colloids (:math:`0.4 < \phi < 0.516`), pre-jammed emulsions
     - Zero yield stress but very slow relaxation (:math:`\tau_{\alpha} \to \infty`), :math:`G' \approx G''` at low :math:`\omega`, extreme shear thinning
   * - **0 <** :math:`\varepsilon` **< 0.1**
     - Marginal glass
     - Dense colloids (:math:`\phi \approx 0.52\text{--}0.55`), soft microgel pastes
     - Small yield stress (10--100 Pa), fragile caging, strong overshoot in startup, :math:`G' > G''` with small plateau
   * - **0.1 <** :math:`\varepsilon` **< 0.3**
     - Moderate glass
     - Hard-sphere colloids (:math:`\phi \approx 0.55\text{--}0.58`), carbopol gels
     - Clear yield stress (100--1000 Pa), robust caging, pronounced plateau in :math:`G'(\omega)`
   * - :math:`\varepsilon` **> 0.3**
     - Deep glass
     - Jammed colloids (:math:`\phi > 0.58`), concentrated emulsions
     - Large yield stress (>1000 Pa), rigid caging, nearly frequency-independent :math:`G'`

Connection to Cage Breaking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Critical Strain (** :math:`\gamma_c` **)**: Quantifies cage strength

- :math:`\gamma_c` **~ 0.05**: Very rigid cages (hard-sphere-like, strong glass)
- :math:`\gamma_c` **~ 0.15**: Soft cages (deformable particles, weak glass)
- :math:`\gamma_c` **~ 0.3**: Fragile cages (near-critical or polymer-like)

The strain decorrelation function :math:`h(\gamma) = \exp[-(\gamma/\gamma_c)^2]`
describes how accumulated strain destroys the structural correlation:

- At :math:`\gamma < \gamma_c`: cage is intact, correlations persist
- At :math:`\gamma \sim \gamma_c`: cage begins to break, correlations decay rapidly
- At :math:`\gamma \gg \gamma_c`: cage is destroyed, system is fully fluidized

**Physical Interpretation**: :math:`\gamma_c` represents the typical strain needed
to displace a particle by one particle diameter and escape the cage. It is related
to the Lindemann criterion for melting.

Yield Stress Emergence
~~~~~~~~~~~~~~~~~~~~~~~

In the glass state (:math:`\varepsilon > 0`), the model predicts a dynamic yield stress:

.. math::

   \sigma_y = G_\infty \gamma_c f

where :math:`f` is the non-ergodicity parameter (glass plateau height). This
connects the yield stress to three microscopic quantities:

1. :math:`G_\infty`: High-frequency modulus (single-particle stiffness)
2. :math:`\gamma_c`: Cage escape strain (local rearrangement threshold)
3. :math:`f`: Degree of caging (structural arrest parameter)

**Diagnostic use**: If fitted :math:`\sigma_y` is much larger than expected from
:math:`G_\infty \gamma_c f`, additional yield mechanisms (e.g., attractive forces,
structural bonds) may be present beyond MCT caging.

Relaxation Timescale Hierarchy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model distinguishes multiple timescales:

1. **Microscopic time**: :math:`\tau_0 = 1/\Gamma` (Brownian diffusion timescale)
2. :math:`\beta` **-relaxation**: Short-time rattling in cage, :math:`\tau_\beta \sim \tau_0`
3. :math:`\alpha` **-relaxation**: Cage escape time, :math:`\tau_\alpha \sim \tau_0 |\varepsilon|^{-\gamma}` (diverges as :math:`\varepsilon \to 0`)

Near the glass transition, :math:`\tau_\alpha` can be :math:`10^6`-:math:`10^{10}` times larger than
:math:`\tau_0`, explaining why materials appear "glassy" on experimental timescales
yet are technically ergodic.

**From fitting**: The crossover frequency in :math:`G'(\omega), G''(\omega)` gives
:math:`\omega_\alpha \sim 1/\tau_\alpha`, revealing the structural relaxation timescale.

Shear-Induced Fluidization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ITT-MCT model shows how shear melts the glass through strain accumulation:

- **Low shear rates** (:math:`\dot{\gamma} \tau_\alpha \ll 1`): Cage has time to reform,
  system behaves as solid with yield stress
- **High shear rates** (:math:`\dot{\gamma} \tau_\alpha \gg 1`): Cage is continuously
  broken by flow, effective viscosity decreases (shear thinning)

The **Weissenberg number** :math:`Wi = \dot{\gamma} \tau_\alpha` controls the
flow-microstructure coupling:

- :math:`Wi \ll 1`: Quasistatic flow, microstructure equilibrates
- :math:`Wi \sim 1`: Transient regime, stress overshoot occurs
- :math:`Wi \gg 1`: Driven flow, microstructure is fully perturbed

**Practical insight**: If startup experiments show stress overshoot at shear rate
:math:`\dot{\gamma}_{\text{peak}}`, estimate :math:`\tau_\alpha \sim 1/\dot{\gamma}_{\text{peak}}`.

Regimes and Behavior
--------------------

Fluid State (:math:`\varepsilon < 0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Long-time correlator: :math:`\Phi(\infty) = 0`
- Zero yield stress
- Terminal viscosity: :math:`\eta_0 = G_\infty / \Gamma`
- Newtonian at low rates, shear-thinning at high rates

Glass State (:math:`\varepsilon > 0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Non-ergodicity: :math:`\Phi(\infty) = f > 0`
- Yield stress: :math:`\sigma_y \approx G_\infty \gamma_c f`
- Plateau modulus: :math:`G'(\omega \to 0) \approx G_\infty f`
- Stress overshoot in startup
- Residual stress in relaxation

Critical Point (:math:`\varepsilon = 0`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Power-law decay: :math:`\Phi(t) \sim t^{-a}`
- Diverging relaxation time
- Maximum susceptibility
- Singular behavior in rheology

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with SAOS**: Fit :math:`G'(\omega)`, :math:`G''(\omega)` to estimate:

   - :math:`G_\infty` from high-frequency plateau
   - :math:`\varepsilon` from low-frequency plateau (glass) or terminal regime (fluid)
   - :math:`\Gamma` from crossover frequency

2. **Refine with flow curve**: Adjust:

   - :math:`\gamma_c` from onset of shear thinning
   - :math:`\varepsilon` from presence/absence of yield stress

3. **Validate with startup**: Check:

   - Overshoot position and height

Troubleshooting
~~~~~~~~~~~~~~~

**Problem: Poor fit at low frequencies**

- Solution: Check if system is actually glassy (try different :math:`\varepsilon` sign)
- May need to account for aging/thixotropy

**Problem: Wrong shear-thinning slope**

- Solution: Adjust :math:`\gamma_c`
- Consider if there are multiple relaxation mechanisms

**Problem: No stress overshoot in startup**

- Solution: Increase :math:`\dot{\gamma}` or reduce :math:`\varepsilon`
- Overshoot requires :math:`Wi = \dot{\gamma} \tau_{\alpha} > 1`

Model Comparison
----------------

ITT-MCT vs SGR
~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Aspect
     - ITT-MCT
     - SGR
   * - Physics
     - Cage effect, density correlators
     - Energy landscape, trap escape
   * - Control parameter
     - Volume fraction / :math:`v_2`
     - Noise temperature :math:`x`
   * - Glass transition
     - Sharp (:math:`v_{2,c} = 4`)
     - Continuous (:math:`x \to 1`)
   * - Shear melting
     - Strain decorrelation
     - Shear-induced trap escape
   * - Computation
     - ODE integration
     - Master equation

ITT-MCT vs Fluidity Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 40 40
   :header-rows: 1

   * - Aspect
     - ITT-MCT
     - Fluidity
   * - Foundation
     - Microscopic correlators
     - Phenomenological fluidity :math:`f`
   * - Parameters
     - ~5 (schematic)
     - ~8--10
   * - Glass transition
     - From MCT vertex
     - From fluidity bounds
   * - Aging
     - Implicit in correlator
     - Explicit :math:`df/dt` term
   * - Shear banding
     - Not in schematic
     - In nonlocal version

Usage
-----

Basic Flow Curve
~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.itt_mct import ITTMCTSchematic
   import numpy as np

   # Create glass-state model
   model = ITTMCTSchematic(epsilon=0.1)

   # Predict flow curve
   gamma_dot = np.logspace(-3, 3, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   # Shows yield stress at low rates
   print(f"Yield stress ≈ {sigma[0]:.1f} Pa")

Linear Viscoelasticity
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get G', G''
   omega = np.logspace(-2, 3, 50)
   G_components = model.predict(
       omega,
       test_mode='oscillation',
       return_components=True
   )
   G_prime = G_components[:, 0]
   G_double_prime = G_components[:, 1]

   # Glass plateau
   print(f"G'(ω→0) ≈ {G_prime[0]:.1f} Pa")

Startup with Overshoot
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # High shear rate startup
   t = np.linspace(0, 10, 200)
   gamma_dot = 10.0  # 1/s

   sigma = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Find overshoot
   i_max = np.argmax(sigma)
   print(f"Overshoot at t = {t[i_max]:.2f} s")
   print(f"Overshoot ratio = {sigma[i_max]/sigma[-1]:.2f}")

LAOS Harmonics
~~~~~~~~~~~~~~

.. code-block:: python

   # Extract nonlinear harmonics
   T = 2 * np.pi  # Period
   t = np.linspace(0, 5*T, 500)

   sigma_prime, sigma_double_prime = model.get_laos_harmonics(
       t, gamma_0=0.2, omega=1.0, n_harmonics=3
   )

   # Third harmonic ratio (nonlinearity measure)
   I_3_1 = np.abs(sigma_prime[1]) / np.abs(sigma_prime[0])
   print(f"I₃/I₁ = {I_3_1:.3f}")

Fitting Experimental Data
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load data
   gamma_dot_exp = np.array([...])  # Experimental shear rates
   sigma_exp = np.array([...])      # Experimental stresses

   # Initial guess
   model = ITTMCTSchematic(epsilon=0.0)
   model.parameters.set_value('G_inf', 1e4)

   # Fit
   model.fit(gamma_dot_exp, sigma_exp, test_mode='flow_curve')

   # Check glass state
   info = model.get_glass_transition_info()
   print(f"Fitted ε = {info['epsilon']:.3f}")
   print(f"System is {'glass' if info['is_glass'] else 'fluid'}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit with uncertainty quantification
   result = model.fit_bayesian(
       gamma_dot_exp, sigma_exp,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )
   print(f"v₂ = {intervals['v2']['mean']:.2f} "
         f"[{intervals['v2']['lower']:.2f}, {intervals['v2']['upper']:.2f}]")

Performance Tips
----------------

JIT Compilation and Precompilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ITT-MCT model uses JAX with diffrax for high-performance ODE integration.
The first flow curve prediction triggers JIT (Just-In-Time) compilation, which
can take 30-90 seconds depending on your hardware. Subsequent calls are very
fast (< 1s for typical data sizes).

**Precompilation:** For predictable timing in interactive applications or
benchmarks, use the ``precompile()`` method to trigger compilation upfront:

.. code-block:: python

   from rheojax.models.itt_mct import ITTMCTSchematic
   import time

   model = ITTMCTSchematic(epsilon=0.1)

   # Trigger JIT compilation (this is slow)
   compile_time = model.precompile()
   print(f"Compilation took {compile_time:.1f}s")

   # Now flow curve predictions are fast
   gamma_dot = np.logspace(-3, 3, 50)
   start = time.time()
   sigma = model.predict(gamma_dot, test_mode='flow_curve')
   print(f"Prediction took {time.time() - start:.2f}s")  # < 1s

**Note:** Precompilation only affects the diffrax-based flow curve solver.
Other protocols (oscillation, startup, creep, relaxation, LAOS) use scipy
and don't require precompilation.

Prony Decomposition
~~~~~~~~~~~~~~~~~~~

The Volterra ODE approach converts the :math:`O(N^2)` memory integral to :math:`O(N)` using
Prony series decomposition of the memory kernel. The quality of the Prony
fit affects accuracy:

.. code-block:: python

   # Check Prony mode quality
   model.initialize_prony_modes(t_max=1000.0, n_points=1000)
   g, tau = model.get_prony_modes()
   print(f"Prony modes: {len(g)}")
   print(f"τ range: [{tau.min():.2e}, {tau.max():.2e}]")

If you see warnings about "Prony fit failed", the memory kernel may be
ill-conditioned. Solutions:

1. Increase ``n_prony_modes`` (default: 10)
2. Adjust the time range in ``initialize_prony_modes()``
3. Check if the system is very close to the glass transition (:math:`\varepsilon \approx 0`)

Memory Usage
~~~~~~~~~~~~

For large batch predictions, memory scales with:

- Number of shear rates :math:`\times` Prony modes :math:`\times` time steps

Typical usage: ~100 MB for 50 shear rates with 10 Prony modes.

For memory-constrained systems:

.. code-block:: python

   # Reduce Prony modes (trades accuracy for memory)
   model = ITTMCTSchematic(epsilon=0.1, n_prony_modes=5)

   # Or process in smaller batches
   for gamma_chunk in np.array_split(gamma_dot, 10):
       sigma_chunk = model.predict(gamma_chunk, test_mode='flow_curve')

JAX Implementation Patterns
---------------------------

This section provides JAX code patterns for implementing ITT-MCT calculations.
These patterns illustrate the core algorithms used internally by RheoJAX.

Memory Kernel Computation
~~~~~~~~~~~~~~~~~~~~~~~~~

The :math:`F_{12}` memory kernel with strain decorrelation:

.. code-block:: python

   import jax.numpy as jnp
   from jax import jit

   @jit
   def memory_kernel_f12(Phi, gamma, v1, v2, gamma_c):
       """
       F₁₂ memory kernel: m(Φ, γ) = (v₁Φ + v₂Φ²) · h(γ)

       Parameters
       ----------
       Phi : float
           Correlator value
       gamma : float
           Accumulated strain
       v1, v2 : float
           MCT vertex coefficients
       gamma_c : float
           Critical strain for cage breaking
       """
       # Mode-coupling vertex
       m_mct = v1 * Phi + v2 * Phi**2

       # Shear decorrelation (Gaussian form)
       h = jnp.exp(-(gamma / gamma_c)**2)

       return m_mct * h

Stress from Correlator History
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computing stress from the correlator integral:

.. code-block:: python

   @jit
   def compute_stress_from_correlator(Phi_history, dt, gamma_dot, G_inf):
       """
       Compute stress from correlator history

       σ(t) = γ̇ ∫₀ᵗ G(τ) dτ = γ̇ G_∞ ∫₀ᵗ Φ(τ)² dτ

       Parameters
       ----------
       Phi_history : array
           Correlator values at each time step
       dt : float
           Time step
       gamma_dot : float
           Shear rate
       G_inf : float
           High-frequency modulus
       """
       G_history = G_inf * Phi_history**2
       return gamma_dot * jnp.trapezoid(G_history, dx=dt)

LAOS Harmonics Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~

Extracting Fourier harmonics from LAOS stress response:

.. code-block:: python

   def extract_laos_harmonics(t, stress, omega, n_harmonics=5):
       """
       Extract odd Fourier harmonics from LAOS stress

       σ(t) = Σ [σ'_n sin(nωt) + σ''_n cos(nωt)]

       Parameters
       ----------
       t : array
           Time points (should cover complete cycles)
       stress : array
           Stress response
       omega : float
           Angular frequency
       n_harmonics : int
           Number of harmonics to extract (1, 3, 5, ...)

       Returns
       -------
       sigma_prime : array
           In-phase components [σ'₁, σ'₃, σ'₅, ...]
       sigma_double_prime : array
           Out-of-phase components [σ''₁, σ''₃, ...]
       """
       period = 2 * jnp.pi / omega

       # Use last complete cycle
       mask = t > (t[-1] - period)
       t_cycle = t[mask] - (t[-1] - period)
       stress_cycle = stress[mask]

       sigma_prime = []
       sigma_double_prime = []

       for n in range(1, 2 * n_harmonics, 2):  # Odd harmonics only
           # Fourier coefficients via projection
           sin_component = (2 / period) * jnp.trapezoid(
               stress_cycle * jnp.sin(n * omega * t_cycle), t_cycle
           )
           cos_component = (2 / period) * jnp.trapezoid(
               stress_cycle * jnp.cos(n * omega * t_cycle), t_cycle
           )

           sigma_prime.append(sin_component)
           sigma_double_prime.append(cos_component)

       return jnp.array(sigma_prime), jnp.array(sigma_double_prime)

**Usage for nonlinearity quantification:**

.. code-block:: python

   # Extract harmonics from LAOS simulation
   sigma_prime, sigma_double_prime = model.get_laos_harmonics(
       t, gamma_0=0.2, omega=1.0, n_harmonics=3
   )

   # Third harmonic ratio (intrinsic nonlinearity)
   I_3_1 = jnp.abs(sigma_prime[1]) / jnp.abs(sigma_prime[0])
   print(f"I₃/I₁ = {I_3_1:.4f}")

   # Fifth harmonic ratio
   I_5_1 = jnp.abs(sigma_prime[2]) / jnp.abs(sigma_prime[0])
   print(f"I₅/I₁ = {I_5_1:.4f}")

See Also
--------

- :doc:`itt_mct_isotropic` --- Full :math:`k`-resolved MCT for quantitative predictions with :math:`S(k)` input
- :doc:`../sgr/sgr_conventional` --- Alternative glass transition model (trap-based, no :math:`S(k)` required)
- :doc:`../fluidity/fluidity_saramito_local` --- Simpler thixotropic yield stress model
- :doc:`../stz/stz_conventional` --- Shear transformation zone theory (effective temperature approach)

**Choosing between ITT-MCT and SGR:**

- **Use ITT-MCT** if: You have colloidal systems, know the volume fraction, want
  to connect to microscopic structure factor :math:`S(k)`
- **Use SGR** if: You have generic soft glasses (foams, emulsions, pastes), want
  simpler parameterization, focus on aging/rejuvenation dynamics

API Reference
-------------

See :class:`~rheojax.models.itt_mct.schematic.ITTMCTSchematic` in the :doc:`/api/models` reference
for the full class documentation, including all methods and attributes.

References
----------

.. [1] Götze, W. *Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory*.
   Oxford University Press (2009). https://doi.org/10.1093/acprof:oso/9780199235346.001.0001

.. [2] Bengtzelius, U., Götze, W. & Sjölander, A. (1984). "Dynamics of supercooled
   liquids and the glass transition." *J. Phys. C: Solid State Phys.*, 17, 5915-5934.
   https://doi.org/10.1088/0022-3719/17/33/005

.. [3] Fuchs, M. and Cates, M. E. "Theory of Nonlinear Rheology and Yielding of
   Dense Colloidal Suspensions." *Physical Review Letters*, 89, 248304 (2002).
   https://doi.org/10.1103/PhysRevLett.89.248304

.. [4] Fuchs, M. and Cates, M. E. "A mode coupling theory for Brownian particles
   in homogeneous steady shear flow." *Journal of Rheology*, 53, 957 (2009).
   https://doi.org/10.1122/1.3119084

.. [5] Brader, J. M., Voigtmann, T., Fuchs, M., Larson, R. G., and Cates, M. E.
   "Glass rheology: From mode-coupling theory to a dynamical yield criterion."
   *Proc. Natl. Acad. Sci. USA*, 106, 15186-15191 (2009).
   https://doi.org/10.1073/pnas.0905330106

.. [6] Siebenbürger, M., Ballauff, M., and Voigtmann, T. "Creep in colloidal glasses."
   *Phys. Rev. Lett.*, 108, 255701 (2012). https://doi.org/10.1103/PhysRevLett.108.255701

.. [7] Brader, J. M., Cates, M. E., and Fuchs, M. "First-Principles Constitutive
   Equation for Suspension Rheology." *Phys. Rev. Lett.*, 101, 138301 (2008).
   https://doi.org/10.1103/PhysRevLett.101.138301

.. [8] Henrich, O., Weysser, F., Cates, M. E., and Fuchs, M. "Hard discs under steady
   shear: comparison of Brownian dynamics simulations and mode coupling theory."
   *Phil. Trans. R. Soc. A*, 367, 5033-5050 (2009).
   https://doi.org/10.1098/rsta.2009.0191

.. [9] Amann, C. P., Siebenbürger, M., Ballauff, M., and Fuchs, M. "Nonlinear
   rheology of glass-forming colloidal dispersions: Transient stress-strain
   relations from anisotropic mode coupling theory." *Journal of Physics:
   Condensed Matter*, 27, 194121 (2015). https://doi.org/10.1088/0953-8984/27/19/194121

.. [10] Mewis, J. and Wagner, N. J. "Colloidal Suspension Rheology." Cambridge
   University Press (2012). ISBN: 978-0-521-51599-3.
   https://doi.org/10.1017/CBO9780511977978
