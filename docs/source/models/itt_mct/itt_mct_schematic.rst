.. _model-itt-mct-schematic:

ITT-MCT Schematic (F₁₂)
=======================

Quick Reference
---------------

**Use when:** Dense colloidal suspensions, glassy materials, yield-stress fluids,
materials showing glass transition behavior

**Parameters:** 5 (v₁, v₂, Γ, γ_c, G_∞) or equivalently (ε, v₁, Γ, γ_c, G_∞)

**Key equation:** Memory kernel m(Φ) = v₁Φ + v₂Φ² with glass transition at v₂ = 4

**Test modes:** Flow curve, oscillation, startup, creep, relaxation, LAOS

**Material examples:** PMMA colloids, emulsions, carbopol gels, concentrated polymer solutions

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
     - Critical vertex coefficient (= 4 for v₁ = 0)
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

The F₁₂ schematic model is a simplified Mode-Coupling Theory (MCT) that captures
the essential physics of the colloidal glass transition with minimal parameters.

**Historical Context:**

MCT was developed in the 1980s by Götze and collaborators to describe the dynamics
of supercooled liquids and dense colloids. The full MCT involves coupled integro-
differential equations for density correlators at all wave vectors k. The schematic
F₁₂ model reduces this to a single scalar equation by replacing the k-dependent
memory kernel with a polynomial form.

**The Cage Effect:**

In dense suspensions, each particle is "caged" by its neighbors. At short times,
particles rattle within their cages (β-relaxation). At long times, cooperative
rearrangements allow cage escape (α-relaxation). The glass transition occurs when
the α-relaxation time diverges - particles become permanently trapped.

**Why "F₁₂":**

The name comes from the memory kernel having terms proportional to Φ¹ and Φ² -
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

At high volume fractions (φ > 0.4), particles begin to interfere with each other's
motion. The "cage" of nearest neighbors slows down diffusion:

.. math::

   D_{\text{long}} = D_0 / S(0)

where S(0) is the zero-wavevector structure factor (compressibility).

As φ → φ_g ≈ 0.516, the cage becomes so strong that particles cannot escape on
any experimental timescale - the system is a glass.

Green-Kubo and the Memory Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamics can be described through the density correlator:

.. math::

   \Phi(k,t) = \frac{\langle \rho_k(t) \rho_{-k}(0) \rangle}{\langle |\rho_k|^2 \rangle}

which measures how density fluctuations at wave vector k decorrelate over time.

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

The F₁₂ Schematic Model
-----------------------

Reduction to Scalar Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the schematic model, we ignore the k-dependence and write:

.. math::

   \partial_t \Phi + \Gamma \left[ \Phi + \int_0^t m(\Phi(s)) \partial_s \Phi(s) ds \right] = 0

with the polynomial memory kernel:

.. math::

   m(\Phi) = v_1 \Phi + v_2 \Phi^2

**Why this form works:**

1. The linear term (v₁Φ) allows for standard viscoelastic relaxation
2. The quadratic term (v₂Φ²) creates the feedback: slow relaxation → strong cage → slower relaxation
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

For v₁ = 0: v₂,c = 4.

The **separation parameter** ε measures distance from the transition:

.. math::

   \varepsilon = \frac{v_2 - v_{2,c}}{v_{2,c}}

- ε < 0: Ergodic fluid (Φ → 0 at long times)
- ε = 0: Critical point (power-law decay)
- ε > 0: Glass state (Φ → f > 0)

Two-Step Relaxation
~~~~~~~~~~~~~~~~~~~

Near the glass transition, the correlator shows characteristic two-step decay:

1. **β-relaxation** (short times): Initial decay to a plateau

   .. math::

      \Phi(t) \approx f_c + h \cdot (t/t_0)^{-a}

2. **Plateau regime**: Correlator "stuck" near f

3. **α-relaxation** (long times): Final decay from plateau

   .. math::

      \Phi(t) \approx f \cdot \exp\left[-(t/\tau_\alpha)^b\right]

The MCT exponents a and b are related to the "exponent parameter" λ:

.. math::

   \frac{\Gamma(1-a)^2}{\Gamma(1-2a)} = \frac{\Gamma(1+b)^2}{\Gamma(1+2b)} = \lambda

For F₁₂ with v₁ = 0: λ = 1.

Integration Through Transients (ITT)
------------------------------------

Extending MCT to Flow
~~~~~~~~~~~~~~~~~~~~~

Under shear, density fluctuations are "advected" by the flow. A wave vector k
at time t' becomes k(t,t') at time t:

.. math::

   k_x(t,t') = k_x - k_y \int_{t'}^t \dot{\gamma}(s) ds = k_x - k_y \gamma(t,t')

This advection destroys the cage structure, leading to flow-induced relaxation.

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

- At γ = 0: h = 1 (full correlation)
- At γ >> γ_c: h → 0 (cage is destroyed)
- γ_c ≈ 0.1 corresponds to the "cage strain"

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

The microscopic form uses the Percus-Yevick structure factor S(k) to weight
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
required. This is the colloidal volume fraction used to compute S(k) via the
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
   sigma = model.predict(gamma_dot, test_mode="flow_curve")

The model's ``__repr__`` shows all form selections:

.. code-block:: python

   >>> model
   ITTMCTSchematic(ε=0.100 [glass], v₂=4.40, h(γ)=gaussian, m=full, σ=microscopic, G_inf=1.00e+06 Pa)

Governing Equations
-------------------

Flow Curve (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

At steady state with constant γ̇, the stress is:

.. math::

   \sigma_{ss} = \dot{\gamma} \int_0^\infty G(s) \cdot h(\dot{\gamma} s) ds

where G(s) = G_∞ Φ_eq(s) is the equilibrium relaxation modulus.

**Yield stress** (glass state, ε > 0):

.. math::

   \sigma_y = \lim_{\dot{\gamma} \to 0} \sigma_{ss} = G_\infty \gamma_c f

**Shear thinning**: As γ̇ increases, the cage is broken faster, and the effective
viscosity decreases:

.. math::

   \eta_{\text{eff}} = \sigma / \dot{\gamma}

Small Amplitude Oscillation (SAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For small strain γ₀ << γ_c, we can linearize around equilibrium:

.. math::

   G^*(\omega) = i\omega \int_0^\infty G_{\text{eq}}(t) e^{-i\omega t} dt

This gives the storage and loss moduli:

.. math::

   G'(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \sin(\omega t) dt

.. math::

   G''(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \cos(\omega t) dt

**Glass plateau**: For ε > 0, G'(ω → 0) → G_∞ f (non-zero plateau)

Startup Flow
~~~~~~~~~~~~

Starting from rest with constant γ̇:

.. math::

   \sigma(t) = \dot{\gamma} \int_0^t G(t-s) \cdot h(\dot{\gamma}(t-s)) ds

This shows a characteristic **stress overshoot** when γ̇τ_α > 1, where τ_α
is the structural relaxation time.

Creep
~~~~~

At constant applied stress σ₀, the strain rate adjusts to maintain:

.. math::

   \sigma_0 = \int_0^t \dot{\gamma}(t') G(t,t') dt'

In the glass state (σ₀ < σ_y): bounded deformation (solid-like)
Above yield (σ₀ > σ_y): continuous flow (fluidization)

This leads to **viscosity bifurcation** - a sharp transition between solid
and fluid behavior at the yield stress.

Stress Relaxation
~~~~~~~~~~~~~~~~~

After cessation of flow at t = 0:

.. math::

   \sigma(t) = \sigma(0) \cdot \Phi_{\text{relax}}(t)

In the glass state, stress does not fully relax:

.. math::

   \lim_{t \to \infty} \sigma(t) = \sigma_{\text{res}} > 0

LAOS
~~~~

For large amplitude oscillatory shear γ(t) = γ₀ sin(ωt):

The stress is non-sinusoidal and can be decomposed into Fourier harmonics:

.. math::

   \sigma(t) = \sum_{n=1,3,5,...} [\sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t)]

Higher harmonics (n = 3, 5, ...) quantify nonlinearity. The ratio σ₃/σ₁
increases with γ₀/γ_c.

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
   * - v₁
     - 0.0
     - (0, 5)
     - —
     - Linear vertex coefficient. Usually 0 for pure F₁₂.
   * - v₂
     - 2.0
     - (0.5, 10)
     - —
     - Quadratic vertex coefficient. Glass at v₂ > 4.
   * - Γ
     - 1.0
     - (10⁻⁶, 10⁶)
     - 1/s
     - Bare relaxation rate. Sets microscopic timescale.
   * - γ_c
     - 0.1
     - (0.01, 0.5)
     - —
     - Critical strain for cage breaking. Typically 0.05-0.2.
   * - G_∞
     - 10⁶
     - (1, 10¹²)
     - Pa
     - High-frequency elastic modulus.

**Alternative parameterization with ε:**

Instead of specifying v₂ directly, use the separation parameter ε:

.. code-block:: python

   model = ITTMCTSchematic(epsilon=0.1)  # Glass state
   model = ITTMCTSchematic(epsilon=-0.1)  # Fluid state

This automatically sets v₂ = v₂,c × (1 + ε).

Typical Parameter Values
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 30 15 15 15 25
   :header-rows: 1

   * - System
     - ε
     - γ_c
     - G_∞ (Pa)
     - Notes
   * - PMMA colloids (φ = 0.55)
     - 0.1
     - 0.08
     - 10²
     - Hard-sphere reference
   * - Carbopol microgels
     - 0.05
     - 0.15
     - 10³
     - Soft particles
   * - Mayonnaise
     - 0.02
     - 0.10
     - 10²
     - Dense emulsion
   * - Silica suspensions
     - 0.15
     - 0.05
     - 10⁴
     - Strong glass

Validity and Assumptions
------------------------

**When the model works well:**

- Dense suspensions (φ > 0.4 for hard spheres)
- Near the glass transition (|ε| < 0.3)
- Monodisperse or narrow size distribution
- No attractive interactions (hard-sphere-like)
- Brownian timescales (colloidal, not granular)

**Limitations:**

- Does not capture crystallization
- Underestimates relaxation times in deeply supercooled regime
- No hopping/activated processes (important at low T or high ε)
- Assumes isotropic structure (no shear-induced ordering)

Regimes and Behavior
--------------------

Fluid State (ε < 0)
~~~~~~~~~~~~~~~~~~~

- Long-time correlator: Φ(∞) = 0
- Zero yield stress
- Terminal viscosity: η₀ = G_∞ / Γ
- Newtonian at low rates, shear-thinning at high rates

Glass State (ε > 0)
~~~~~~~~~~~~~~~~~~~

- Non-ergodicity: Φ(∞) = f > 0
- Yield stress: σ_y ≈ G_∞ γ_c f
- Plateau modulus: G'(ω→0) ≈ G_∞ f
- Stress overshoot in startup
- Residual stress in relaxation

Critical Point (ε = 0)
~~~~~~~~~~~~~~~~~~~~~~

- Power-law decay: Φ(t) ~ t⁻ᵃ
- Diverging relaxation time
- Maximum susceptibility
- Singular behavior in rheology

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with SAOS**: Fit G'(ω), G''(ω) to estimate:
   
   - G_∞ from high-frequency plateau
   - ε from low-frequency plateau (glass) or terminal regime (fluid)
   - Γ from crossover frequency

2. **Refine with flow curve**: Adjust:
   
   - γ_c from onset of shear thinning
   - ε from presence/absence of yield stress

3. **Validate with startup**: Check:
   
   - Overshoot position and height

Troubleshooting
~~~~~~~~~~~~~~~

**Problem: Poor fit at low frequencies**

- Solution: Check if system is actually glassy (try different ε sign)
- May need to account for aging/thixotropy

**Problem: Wrong shear-thinning slope**

- Solution: Adjust γ_c
- Consider if there are multiple relaxation mechanisms

**Problem: No stress overshoot in startup**

- Solution: Increase γ̇ or reduce ε
- Overshoot requires Wi = γ̇τ_α > 1

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
     - Volume fraction / v₂
     - Noise temperature x
   * - Glass transition
     - Sharp (v₂_c = 4)
     - Continuous (x → 1)
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
     - Phenomenological fluidity f
   * - Parameters
     - ~5 (schematic)
     - ~8-10
   * - Glass transition
     - From MCT vertex
     - From fluidity bounds
   * - Aging
     - Implicit in correlator
     - Explicit df/dt term
   * - Shear banding
     - Not in schematic
     - In nonlocal version

Usage Examples
--------------

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

The Volterra ODE approach converts the O(N²) memory integral to O(N) using
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
3. Check if the system is very close to the glass transition (ε ≈ 0)

Memory Usage
~~~~~~~~~~~~

For large batch predictions, memory scales with:

- Number of shear rates × Prony modes × time steps

Typical usage: ~100 MB for 50 shear rates with 10 Prony modes.

For memory-constrained systems:

.. code-block:: python

   # Reduce Prony modes (trades accuracy for memory)
   model = ITTMCTSchematic(epsilon=0.1, n_prony_modes=5)

   # Or process in smaller batches
   for gamma_chunk in np.array_split(gamma_dot, 10):
       sigma_chunk = model.predict(gamma_chunk, test_mode='flow_curve')

API Reference
-------------

.. autoclass:: rheojax.models.itt_mct.ITTMCTSchematic
   :members:
   :undoc-members:
   :show-inheritance:

References
----------

**Foundational MCT:**

.. [Gotze2009] Götze W. (2009) "Complex Dynamics of Glass-Forming Liquids: A
   Mode-Coupling Theory", Oxford University Press. *The definitive MCT reference.*

.. [Gotze1984] Götze W. & Sjögren L. (1984) "Scaling properties in the β-relaxation
   regime of the glass transition", J. Phys. C 17, 5759. *Two-step relaxation.*

**ITT-MCT Development:**

.. [Fuchs2002] Fuchs M. & Cates M.E. (2002) "Theory of Nonlinear Rheology and
   Yielding of Dense Colloidal Suspensions", Phys. Rev. Lett. 89, 248304.
   *Original ITT-MCT formulation.*

.. [Fuchs2009] Fuchs M. & Cates M.E. (2009) "A mode coupling theory for Brownian
   particles in homogeneous steady shear flow", J. Rheol. 53, 957.
   *Detailed derivations.*

**Applications and Validation:**

.. [Brader2008] Brader J.M. et al. (2008) "Glass rheology: From mode-coupling theory
   to a dynamical yield criterion", J. Phys.: Condens. Matter 20, 494243.

.. [Siebenbuerger2009] Siebenbürger M. et al. (2009) "Creep in colloidal glasses",
   J. Rheol. 53, 707. *Experimental validation.*

.. [Brader2009] Brader J.M. et al. (2009) "First-principles constitutive equation
   for suspension rheology", Proc. Natl. Acad. Sci. 106, 15186. *ISM development.*

**Experimental Techniques:**

.. [Pham2008] Pham K.N. et al. (2008) "Yielding of colloidal glasses", Europhys.
   Lett. 75, 624. *Hard-sphere rheology.*

.. [Petekidis2002] Petekidis G. et al. (2002) "Rearrangements in hard-sphere glasses
   under oscillatory shear strain", Phys. Rev. E 66, 051402. *LAOS experiments.*
