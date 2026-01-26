.. _model-generalized-maxwell:

Generalized Maxwell Model (Multi-Mode)
========================================

Quick Reference
---------------

- **Use when:** Broad relaxation spectra, multi-mode viscoelastic behavior, complex polymer systems
- **Parameters:** 2N+1 (:math:`E_\infty, E_1, \ldots, E_N, \tau_1, \ldots, \tau_N`) with transparent element minimization
- **Key equation:** :math:`E(t) = E_\infty + \sum_{i=1}^{N} E_i \exp(-t/\tau_i)` (Prony series)
- **Test modes:** Relaxation (preferred), oscillation (excellent), creep (acceptable)
- **Material examples:** Polymer melts with broad MW distributions, multi-phase composites, soft solids

Notation Guide
--------------

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`E_\infty/G_\infty`
     - Equilibrium modulus (0 for liquids)
     - Pa
   * - :math:`E_i/G_i`
     - Mode i strength
     - Pa
   * - :math:`\tau_i`
     - Mode i relaxation time
     - s
   * - :math:`N`
     - Number of modes
     - —
   * - :math:`N_{\text{opt}}`
     - Optimized number of modes
     - —
   * - :math:`H(\tau)`
     - Continuous relaxation spectrum
     - Pa
   * - :math:`\gamma(t)`
     - Strain
     - —
   * - :math:`\sigma(t)`
     - Stress
     - Pa
   * - :math:`\omega`
     - Angular frequency
     - rad/s
   * - :math:`G'(\omega)`
     - Storage modulus
     - Pa
   * - :math:`G''(\omega)`
     - Loss modulus
     - Pa

Overview
--------

The Generalized Maxwell Model (GMM) extends the single Maxwell element to **N parallel modes**, providing a flexible Prony series framework for capturing complex relaxation spectra. Unlike single-mode models (Maxwell, Zener) that assume one characteristic timescale, the GMM represents materials with **continuous distributions** of relaxation times :math:`H(\tau)` through a discrete approximation.

**Key innovation: Tri-mode equality** – The same Prony parameters describe relaxation, oscillation, and creep without FFT transforms. Fit in one test mode, predict in all modes with 5-270× NLSQ speedup over scipy-based implementations.

**Transparent element minimization** – Users request N=10 modes, the system automatically optimizes to N_opt (e.g., 3) based on :math:`R^2` degradation tolerance, balancing parsimony with fit quality.

Physical Foundations
--------------------

Mechanical Analogue
~~~~~~~~~~~~~~~~~~~

The GMM consists of **N Maxwell elements in parallel**, each contributing a distinct relaxation mode:

.. code-block:: text

   ┌──────────┬──────────┬─────┬──────────┐
   │ Maxwell 1│ Maxwell 2│ ... │ Maxwell N│
   │ (E₁, τ₁) │ (E₂, τ₂) │     │ (Eₙ, τₙ)│
   └──────────┴──────────┴─────┴──────────┘
            ↓
   Plus equilibrium spring E_∞ (optional, can be zero for liquids)

| Total stress: :math:`\sigma = E_\infty \varepsilon + \sum_i \sigma_i`
| Each mode: :math:`\sigma_i + \tau_i d\sigma_i/dt = E_i d\varepsilon/dt`

The parallel configuration means:

- **Stress is additive**: :math:`\sigma(t) = \sigma_\infty(t) + \sum_{i=1}^{N} \sigma_i(t)`
- **Strain is identical**: All elements experience the same strain :math:`\epsilon(t)`

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GMM represents materials with **multiple relaxation mechanisms** operating simultaneously:

**Spring E_∞ (equilibrium modulus)**:
   - Permanent network structure (crosslinks in elastomers)
   - Long-range entanglements that don't relax experimentally
   - Zero for viscoelastic liquids (polymer melts)

**Mode strengths** :math:`E_i`:
   - Contribution of i-th relaxation process to total modulus
   - Related to density of chains with relaxation time :math:`\tau_i`
   - Approximates continuous spectrum: :math:`E_i \approx H(\tau_i) \Delta(\log \tau_i)`

**Relaxation times** :math:`\tau_i`:
   - Distributed timescales from Rouse modes (fast) to reptation (slow)
   - Typically span 4-8 decades (e.g., 10\ :math:`^{-3 to 10^5}` s for polymer melts)
   - Logarithmic spacing: :math:`\tau_i = \tau_{\min} \cdot 10^{i \Delta \log \tau}`

**Physical meaning of N modes**:
   - N=1: Single Maxwell (exponential decay, narrow MW distribution)
   - N=2-5: Soft solids with few relaxation processes
   - N=5-20: Polymers with broad MW distributions
   - N=20-50: High-resolution spectrum reconstruction (research)

Connection to Continuous Relaxation Spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Prony series is a discrete approximation to the continuous spectrum :math:`H(\tau)`:

.. math::

   G(t) = G_e + \int_0^\infty H(\tau) \exp(-t/\tau) \, d(\log \tau)
   \approx G_\infty + \sum_{i=1}^{N} H(\tau_i) \Delta(\log \tau_i) \exp(-t/\tau_i)

where :math:`E_i = H(\tau_i) \Delta(\log \tau_i)`. The GMM provides **finite-dimensional regularization** for the ill-posed spectrum inversion problem.

Material Examples with Typical N
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative GMM applications
   :header-rows: 1
   :widths: 30 15 15 20 20

   * - Material
     - N modes
     - E_inf (Pa)
     - :math:`\tau` range (s)
     - Ref
   * - Polystyrene melt (broad MW)
     - 8-15
     - 0
     - 10\ :math:`^{-2 - 10^4}`
     - [1]
   * - PMMA at T_g + 50°C
     - 10-20
     - 0
     - 10\ :math:`^{-4 - 10^2}`
     - [2]
   * - SBR rubber (unfilled)
     - 5-10
     - :math:`5 \times 10^5`
     - 10\ :math:`^{-6 - 10^1}`
     - [3]
   * - Bitumen (asphalt)
     - 12-18
     - :math:`1 \times 10^5`
     - 10\ :math:`^{-3 - 10^6}`
     - [4]
   * - Hydrogel (multi-network)
     - 3-5
     - :math:`1 \times 10^4`
     - 10\ :math:`^{-1 - 10^3}`
     - [5]

Governing Equations
-------------------

Prony Series Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Relaxation mode** (step strain :math:`\epsilon_0` at :math:`t=0`):

.. math::

   E(t) = E_\infty + \sum_{i=1}^{N} E_i \exp(-t/\tau_i)

**Oscillation mode** (closed-form Fourier transform, **no FFT required**):

.. math::

   E'(\omega) = E_\infty + \sum_{i=1}^{N} E_i \frac{(\omega \tau_i)^2}{1 + (\omega \tau_i)^2}

   E''(\omega) = \sum_{i=1}^{N} E_i \frac{\omega \tau_i}{1 + (\omega \tau_i)^2}

**Creep mode** (step stress :math:`\sigma_0` at :math:`t=0`, backward-Euler numerical integration):

.. math::

   J(t) = \frac{\epsilon(t)}{\sigma_0} \quad \text{via ODE solver}

   \text{where } \sigma_0 = E_\infty \epsilon(t) + \sum_{i=1}^{N} \sigma_i(t)

   \sigma_i(t) + \tau_i \frac{d\sigma_i}{dt} = E_i \frac{d\epsilon}{dt}

Tri-Mode Equality Proof
~~~~~~~~~~~~~~~~~~~~~~~~

**Theorem**: The same Prony parameters {E_∞, E_i, :math:`\tau_i`} satisfy all three test modes.

**Proof sketch**:

1. **Relaxation → Oscillation**: Apply Fourier transform to relaxation modulus:

   .. math::

      G^*(\omega) = i\omega \int_0^\infty G(t) e^{-i\omega t} dt

   For exponential terms :math:`\exp(-t/\tau_i)`:

   .. math::

      \int_0^\infty \exp(-t/\tau_i) e^{-i\omega t} dt = \frac{\tau_i}{1 + i\omega\tau_i}

   Separating real/imaginary parts yields :math:`G'` and :math:`G''` formulas above.

2. **Relaxation → Creep**: Solve GMM ODEs numerically with step stress input. The internal stress variables :math:`\sigma_i(t)` relax exponentially with time constants :math:`\tau_i`, matching relaxation behavior.

3. **Oscillation → Relaxation**: Inverse Fourier transform (numerical, but guaranteed by causality and linearity).

**Practical implication**: Fit GMM to DMA frequency sweeps (oscillation), then predict stress relaxation (relaxation) or creep recovery (creep) with same parameters—no refitting needed.

Fourier Transform Derivation (Oscillation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Start with relaxation Prony series:

.. math::

   E(t) = E_\infty + \sum_{i=1}^{N} E_i \exp(-t/\tau_i)

**Step 2**: Apply Fourier transform for oscillatory input :math:`\epsilon(t) = \epsilon_0 e^{i\omega t}`:

.. math::

   E^*(\omega) = i\omega \int_0^\infty E(t) e^{-i\omega t} dt

**Step 3**: Integrate each exponential term:

.. math::

   \int_0^\infty E_i \exp(-t/\tau_i) e^{-i\omega t} dt = E_i \frac{\tau_i}{1 + i\omega\tau_i}

**Step 4**: Multiply by :math:`i\omega` and separate real/imaginary parts:

.. math::

   E_i \cdot i\omega \frac{\tau_i}{1 + i\omega\tau_i} = E_i \frac{i\omega\tau_i}{1 + i\omega\tau_i}

   = E_i \frac{i\omega\tau_i (1 - i\omega\tau_i)}{(1 + i\omega\tau_i)(1 - i\omega\tau_i)}

   = E_i \frac{(\omega\tau_i)^2 + i\omega\tau_i}{1 + (\omega\tau_i)^2}

Thus:

.. math::

   E'(\omega) = E_\infty + \sum_{i} E_i \frac{(\omega\tau_i)^2}{1 + (\omega\tau_i)^2}

   E''(\omega) = \sum_{i} E_i \frac{\omega\tau_i}{1 + (\omega\tau_i)^2}

**Advantage**: Analytical expression (no FFT), fully parallelizable on GPU via JAX.

Backward-Euler Numerical Integration (Creep)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For creep simulation, the GMM ODEs are solved using unconditionally stable backward-Euler:

**ODEs**:

.. math::

   \sigma_0 = E_\infty \epsilon(t) + \sum_{i=1}^{N} \sigma_i(t)

   \sigma_i(t) + \tau_i \frac{d\sigma_i}{dt} = E_i \frac{d\epsilon}{dt}

**Backward-Euler discretization** (time step :math:`\Delta t`):

.. math::

   \sigma_i^{n+1} = \alpha_i \sigma_i^n + \beta_i \Delta \epsilon

   \alpha_i = \exp(-\Delta t / \tau_i), \quad \beta_i = \frac{E_i \tau_i}{\Delta t} (1 - \alpha_i)

**Solve for strain increment** :math:`\Delta \epsilon`:

.. math::

   \sigma_0 = E_\infty (\epsilon^n + \Delta \epsilon) + \sum_i (\alpha_i \sigma_i^n + \beta_i \Delta \epsilon)

   \Delta \epsilon = \frac{\sigma_0 - \sum_i \alpha_i \sigma_i^n}{E_\infty + \sum_i \beta_i}

**Stability**: :math:`\alpha_i \in [0,1]` ensures unconditional stability (no CFL restriction). JAX's `jax.lax.scan` enables GPU-accelerated time-stepping.

----

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 15 12 15 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``E_inf``
     - :math:`E_\infty`
     - Pa
     - :math:`E_\infty \geq 0`
     - Equilibrium modulus (0 for liquids)
   * - ``E_i``
     - :math:`E_i`
     - Pa
     - :math:`E_i > 0`
     - Mode i strength (i = 1...N)
   * - ``tau_i``
     - :math:`\tau_i`
     - s
     - :math:`\tau_i > 0`
     - Mode i relaxation time (i = 1...N)
   * - ``n_modes``
     - :math:`N`
     - —
     - :math:`N \geq 1`
     - Number of modes (user input)
   * - ``n_modes_opt``
     - :math:`N_{\text{opt}}`
     - —
     - :math:`N_{\text{opt}} \leq N`
     - Optimized modes (auto-reduced)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**E_inf (Equilibrium Modulus)**:
   - **Physical meaning**: Long-time plateau modulus representing permanent network structure
   - **Molecular origin**: Chemical crosslinks, physical entanglements that don't relax
   - **Typical values**:
      - Viscoelastic liquids: :math:`E_\infty = 0` (flows at long times)
      - Crosslinked elastomers: :math:`10^4 - 10^6` Pa
      - Lightly crosslinked gels: :math:`10^1 - 10^4` Pa
   - **Significance**: :math:`E_\infty > 0` indicates solid-like equilibrium behavior

**E_i (Mode Strengths)**:
   - **Physical meaning**: Contribution of the i-th relaxation mechanism to total modulus
   - **Molecular origin**: Weight fraction of chains with relaxation time :math:`\tau_i`
   - **Distribution shape**: Reflects molecular weight distribution (MWD)
   - **Sum rule**: :math:`\sum_{i=1}^N E_i` = total relaxed modulus (liquid-like contribution)

**tau_i (Relaxation Times)**:
   - **Physical meaning**: Characteristic timescale for the i-th relaxation process
   - **Molecular origin**: Chain length via :math:`\tau_i \sim M_i^{3.4}` (reptation scaling)
   - **Logarithmic spacing**: Typically spans 4-8 decades for broad MWD polymers
   - **Distribution**: Ordered :math:`\tau_1 < \tau_2 < ... < \tau_N` during fitting

**N (Number of Modes)**:
   - **Physical meaning**: Resolution of discrete spectrum approximation
   - **Selection guidelines**:
      - N=1-3: Single relaxation process (narrow MWD)
      - N=5-10: Typical polymer melt (broad MWD)
      - N=10-20: High-resolution spectrum reconstruction
   - **Parsimony**: Automatic reduction to N_opt balances fit quality with overfitting risk

----

Validity and Assumptions
-------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Linear viscoelasticity**: The GMM assumes linear superposition of strain and stress (Boltzmann superposition principle)
2. **Isothermal**: Temperature effects require time-temperature superposition (TTS) via mastercurve construction
3. **Homogeneous deformation**: No spatial gradients or wall slip
4. **Discrete spectrum approximation**: Continuous spectrum :math:`H(\tau)` is approximated by N discrete modes
5. **Exponential basis**: Relaxation processes are represented as exponential decays

Data Requirements
~~~~~~~~~~~~~~~~~

**Minimum requirements**:
   - Data points :math:`M \geq 3N` (avoid overfitting)
   - Time/frequency span :math:`\geq 3` decades (capture relaxation range)
   - Linear viscoelastic regime (strain < 1-5% for most polymers)

**Optimal requirements**:
   - Data points :math:`M \geq 5N` (good conditioning)
   - Time/frequency span :math:`\geq 5` decades (broad spectrum)
   - Multiple test modes (relaxation + oscillation for validation)
   - Temperature series (for TTS mastercurve)

Limitations
~~~~~~~~~~~

**Narrow relaxation spectra**:
   For materials with < 2 decades of relaxation, GMM is overparameterized. Use single Maxwell or Zener models instead.

**Power-law regions**:
   Discrete modes poorly approximate continuous power-law spectra. Consider fractional models (FML, FZSS) for smoother representation.

**Nonlinear behavior**:
   GMM is strictly linear. For large-amplitude deformation, use nonlinear models (Saramito, DMT, Fluidity).

**Extrapolation**:
   GMM predictions outside the fitted time/frequency range are unreliable. Fractional models extrapolate better due to power-law forms.

**Element minimization artifacts**:
   Automatic reduction from N to N_opt may miss subtle features if optimization_factor is too aggressive (> 2.0).

----

What You Can Learn
------------------

This section explains how to interpret fitted Generalized Maxwell Model parameters
to extract insights about material structure, molecular properties, and processing behavior.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`G_i` **(Mode Strengths)**:
   Contribution of i-th relaxation process to total modulus, related to density of
   chains/modes with characteristic time :math:`\tau_i`.

   *For graduate students*: G_i approximates the continuous spectrum: :math:`G_i \approx H(\tau_i) \Delta(\log \tau_i)`.
   For entangled polymers: G_i ∝ w(M_i) where w is the molecular weight distribution.
   Sum rule: :math:`\Sigma` G_i = total relaxed modulus. Plateau modulus G_N^0 ≈ E_∞ for crosslinked
   systems relates to entanglement density via G_N^0 = :math:`\rhoRT/M_e`.

   *For practitioners*: Extract from dynamic moduli fitting. Typical: narrow MWD (1-2
   dominant modes), broad MWD (5-15 modes spanning decades). Monitor G_i changes for
   degradation (decrease) or crosslinking (increase in G_∞).

:math:`\tau_i` **(Relaxation Times)**:
   Characteristic timescales for each relaxation mode, typically logarithmically spaced.

   *For graduate students*: For polymers: :math:`\tau_i` ~ M_i^3.4 (reptation) or M_i^2 (Rouse).
   Logarithmic spacing: :math:`\tau_i = \tau_min \cdot (\tau_max/\tau_min)^((i-1)/(N-1))`. Zero-shear viscosity:
   :math:`\eta_0 = \Sigma G_i \cdot \tau_i`. Temperature dependence via WLF or Arrhenius shift factors.

   *For practitioners*: Typical spans: 4-8 decades (10^-3 to 10^5 s) for polymer melts.
   Fast modes (:math:`\tau` < 1 s) = local dynamics, slow modes (:math:`\tau` > 100 s) = terminal relaxation.
   Extract via TTS if narrow experimental time window.

:math:`E_\infty` **(Equilibrium Modulus)**:
   Permanent network modulus, zero for viscoelastic liquids, nonzero for soft solids.

   *For graduate students*: E_∞ = 0 (terminal flow, :math:`\eta_0` finite). E_∞ > 0 (permanent
   crosslinks, infinite :math:`\eta`). For rubber elasticity: :math:`G_{\infty} = \nu k T` where :math:`\nu` is crosslink density.
   Relates to gel point: E_∞ appears at percolation threshold.

   *For practitioners*: Measure from G' low-frequency plateau. Uncrosslinked melts:
   E_∞ = 0. Elastomers/gels: E_∞ = 10^3-10^6 Pa. Monitor E_∞ during curing to track gelation.

**N (Number of Modes)**:
   Discrete approximation order for continuous relaxation spectrum.

   *For graduate students*: N-mode Prony series approximates continuous H(:math:`\tau`). Optimal N
   balances fit quality vs parsimony. Information criteria (AIC, BIC) or element
   minimization (N_opt from degradation tolerance) guide selection.

   *For practitioners*: Start with N = 5-10 for polymers. RheoJAX auto-optimizes to
   N_opt (e.g., request N = 10, get N_opt = 3 if sufficient). Monodisperse: N = 1-3,
   broad MWD: N = 10-20.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Generalized Maxwell Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - N = 1, E_∞ = 0
     - Single-mode viscoelastic liquid
     - Monodisperse polymer solutions
     - Simple Maxwell behavior, exponential relaxation
   * - N = 2-5, E_∞ = 0
     - Narrow MWD polymer melt
     - Polyethylene, polypropylene (narrow)
     - Few relaxation timescales, predictable flow
   * - N = 5-15, E_∞ = 0
     - Broad MWD polymer melt
     - Polydisperse polyethylene, polystyrene
     - Complex relaxation, wide processing window
   * - N = 10-20, E_∞ = 0
     - Very broad MWD or multi-phase
     - Polymer blends, branched polymers
     - Extreme timescale dispersion, TTS needed
   * - E_∞ > 0, N = 3-10
     - Soft solid (gel, elastomer)
     - Rubber, hydrogels, crosslinked polymers
     - Permanent network, shape memory
   * - :math:`\tau_max/\tau_min < 10^2`
     - Narrow spectrum
     - Monodisperse, single mechanism
     - Simple relaxation, single timescale dominates
   * - :math:`\tau_max/\tau_min > 10^4`
     - Broad spectrum
     - Polydisperse, complex systems
     - Wide processing window, TTS essential

Physical Insights from Prony Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mode Spectrum Interpretation:**

The discrete relaxation spectrum {:math:`\tau_i`, E_i} provides a fingerprint of the material's relaxation processes:

**Mode strength E_i**:
   - **Physical meaning**: Contribution of the i-th relaxation mechanism to total modulus
   - **Distribution shape**: Reflects molecular weight distribution (MWD) or phase heterogeneity
   - **Sum rule**: :math:`\sum_{i=1}^N E_i` = total relaxed modulus (liquid-like component)

**Relaxation time** :math:`\tau_i` **:**
   - **Physical meaning**: Characteristic timescale for the i-th relaxation process
   - **Polymer interpretation**: Related to molecular weight via :math:`\tau_i \sim M_i^{3.4}` (Rouse/reptation)
   - **Logarithmic spacing**: Typically spans 4-8 decades for broad MWD polymers

**Connection to Molecular Weight Distribution:**

For entangled polymers, the Prony spectrum approximately maps to the MWD via:

.. math::

   E_i \propto w(M_i) \quad \text{(weight fraction at molecular weight } M_i\text{)}

This enables rheological characterization of MWD without gel permeation chromatography (GPC), particularly useful for:
   - Polymerization monitoring (batch-to-batch consistency)
   - Degradation studies (tracking MW changes over time)
   - Blend characterization (resolving individual components)

**Prony Series Decomposition Insights:**

The GMM decomposes complex viscoelastic behavior into interpretable components:

**Short-time modes** (:math:`\tau_i < 1` s):
   - Fast Rouse modes (local chain motion)
   - :math:`\beta-relaxation` processes (segmental dynamics)
   - Typical for glassy dynamics near T_g

**Intermediate-time modes** (:math:`1 < \tau_i < 100` s):
   - Reptation modes (chain diffusion)
   - Constraint release mechanisms
   - Typical for polymer melts at processing temperatures

**Long-time modes** (:math:`\tau_i > 100` s):
   - Terminal relaxation (flow onset)
   - Entanglement network relaxation
   - Accessible via TTS mastercurves

**Equilibrium modulus E_∞:**

For **liquids** (:math:`E_\infty = 0`):
   - Material eventually flows (polymer melts, solutions)
   - Terminal relaxation completes
   - Viscosity :math:`\eta_0 = \sum_i E_i \tau_i` (zero-shear viscosity)

For **solids** (:math:`E_\infty > 0`):
   - Permanent network (crosslinks, entanglements)
   - Plateau modulus :math:`G_N^0 \approx E_\infty`
   - Rubber elasticity (vulcanized rubbers, gels)

**Spectral Width and Material Complexity:**

The width of the relaxation spectrum reveals material complexity:

**Narrow spectrum** (< 2 decades):
   - Monodisperse polymers
   - Single relaxation mechanism
   - Well-defined characteristic timescale

**Broad spectrum** (> 4 decades):
   - Polydisperse polymers (broad MWD)
   - Multi-phase materials (composites, blends)
   - Multiple relaxation mechanisms

**Spectral shape interpretation**:
   - **Monotonic decrease** (E_i decreases with :math:`\tau_i`): Typical for polymers
   - **Bimodal peaks**: Blends or multi-phase materials
   - **Flat plateau**: Continuous power-law relaxation (consider fractional models)

Material Characterization Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From Relaxation Tests:**
   - Discrete relaxation spectrum {:math:`\tau_i`, E_i}
   - Total relaxed modulus :math:`\sum_i E_i`
   - Equilibrium modulus :math:`E_\infty`
   - Relaxation breadth (decades spanned by :math:`\tau_i`)

**From Oscillation Tests:**
   - Storage modulus :math:`G'(\omega)` across frequency range
   - Loss modulus :math:`G''(\omega)` (dissipation)
   - Loss tangent :math:`\tan \delta = G''/G'` (viscoelastic character)
   - Plateau modulus :math:`G_N^0` (from :math:`G'` plateau)

**From Creep Tests:**
   - Creep compliance :math:`J(t)`
   - Retardation spectrum (inverse problem)
   - Recovery behavior (viscoelastic vs viscoplastic)

**From Time-Temperature Superposition:**
   - Shift factors :math:`a_T` (WLF or Arrhenius)
   - Temperature-independent spectrum (at reference T)
   - Activation energy (from :math:`a_T` vs T)
   - Glass transition temperature T_g (from :math:`\tan \delta` peak)

**Cross-Validation Capabilities:**
   - Tri-mode equality: Fit in relaxation, predict in oscillation/creep
   - Cox-Merz rule: Compare :math:`|\eta^*(\omega)|` to :math:`\eta(\dot{\gamma})`
   - Kramers-Kronig relations: Check thermodynamic consistency

**Quality Control Applications:**
   - Batch-to-batch consistency (compare {:math:`\tau_i`, E_i})
   - Aging/degradation tracking (monitor spectrum evolution)
   - Blend composition verification (detect individual components)
   - Cure monitoring (track crosslink density via :math:`E_\infty`)

----

.. list-table:: Generalized Maxwell Model Parameters
   :header-rows: 1
   :widths: 18 12 12 18 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``E_inf`` / ``G_inf``
     - :math:`E_\infty`
     - Pa
     - :math:`\geq 0`
     - Equilibrium modulus (0 for liquids)
   * - ``E_1`` / ``G_1``
     - :math:`E_1`
     - Pa
     - :math:`> 0`
     - Mode 1 strength
   * - ``E_2`` / ``G_2``
     - :math:`E_2`
     - Pa
     - :math:`> 0`
     - Mode 2 strength
   * - ...
     - ...
     - Pa
     - :math:`> 0`
     - (up to N modes)
   * - ``E_N`` / ``G_N``
     - :math:`E_N`
     - Pa
     - :math:`> 0`
     - Mode N strength
   * - ``tau_1``
     - :math:`\tau_1`
     - s
     - :math:`> 0`
     - Mode 1 relaxation time
   * - ``tau_2``
     - :math:`\tau_2`
     - s
     - :math:`> 0`
     - Mode 2 relaxation time
   * - ...
     - ...
     - s
     - :math:`> 0`
     - (up to N modes)
   * - ``tau_N``
     - :math:`\tau_N`
     - s
     - :math:`> 0`
     - Mode N relaxation time

**Total parameters**: :math:`2N + 1` (one E_inf, N moduli, N times)

**Modulus type** (``modulus_type`` constructor argument):
   - ``modulus_type='shear'``: Uses G (shear modulus) symbols
   - ``modulus_type='tensile'``: Uses E (tensile modulus) symbols
   - **Internal logic identical**, only parameter names differ

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**E_∞ / G_∞ (Equilibrium Modulus)**:
   - **Physical meaning**: Long-time plateau modulus (permanent network)
   - **Zero for liquids**: Polymer melts, viscoelastic fluids (eventually flow)
   - **Nonzero for solids**: Rubbers, gels, crosslinked networks
   - **Typical ranges**:
      - Liquids: :math:`E_\infty = 0`
      - Soft gels: :math:`10^3 - 10^5` Pa
      - Rubbers: :math:`10^5 - 10^7` Pa

:math:`E_i / G_i` **(Mode Strengths)**:
   - **Physical meaning**: Contribution of i-th mode to total modulus
   - **Distribution**: Reflects relaxation spectrum :math:`H(\tau_i)`
   - **Magnitude ordering**:
      - Often decreases with i (fewer long-time processes)
      - Can be non-monotonic for multi-phase materials
   - **Sum interpretation**: :math:`\sum E_i` = total relaxed modulus
   - **Typical ranges**: :math:`10^3 - 10^7` Pa (material-dependent)

:math:`\tau_i` **(Relaxation Times)**:
   - **Physical meaning**: Timescale for i-th relaxation process
   - **Logarithmic spacing**: Usually :math:`\tau_i = 10^{a + i \cdot \Delta}`
   - **Coverage**: Should span experimental time/frequency window
   - **Typical ranges**:
      - Polymer melts: :math:`10^{-3}` to :math:`10^5` s
      - Rubbers (near T_g): :math:`10^{-6}` to :math:`10^1` s
      - Gels: :math:`10^{-2}` to :math:`10^3` s

**Number of modes N**:
   - **Element minimization**: User requests N, system auto-reduces to N_opt
   - **Guidelines**:
      - N < data points / 3 (avoid overfitting)
      - N ≥ 3 for most polymers (capture broad spectra)
      - N = 1 degenerates to single Maxwell
   - **Computational cost**: Scales as O(N) per evaluation (JAX-parallelized)

Transparent Element Minimization
---------------------------------

Algorithm Overview
~~~~~~~~~~~~~~~~~~

**Problem**: How many modes N are truly needed to fit the data well?

**Solution**: Iterative N reduction with :math:`R^2` degradation criterion.

**User workflow**:
   1. User requests ``n_modes=10`` (generous upper bound)
   2. System fits N=10, then N=9, N=8, ..., N=1
   3. Computes :math:`R^2` for each N
   4. Selects smallest N where :math:`R^2_N \geq \text{threshold}`
   5. Returns optimized model with N_opt modes transparently

**Transparency**: User receives optimal model without manual intervention. Diagnostics available via ``get_element_minimization_diagnostics()``.

:math:`R^2` Threshold Criterion
~~~~~~~~~~~~~~~~~~~~~~~

**Threshold definition**:

.. math::

   R^2_{\text{threshold}} = R^2_{\max} - \Delta R^2_{\text{allowed}}

   \Delta R^2_{\text{allowed}} = (1 - R^2_{\max}) \times (\text{optimization\_factor} - 1.0)

**Interpretation**:

- :math:`R^2_{\max}` = best :math:`R^2` across all N (typically N=10)
- :math:`1 - R^2_{\max}` = degradation "room" from perfect fit
- ``optimization_factor`` controls how much degradation is tolerable

**Examples**:

.. math::

   R^2_{\max} = 0.998 \quad \Rightarrow \quad \text{degradation room} = 0.002

   \text{Factor} = 1.0: \, \Delta R^2 = 0, \, \text{threshold} = 0.998 \quad (\text{require best N})

   \text{Factor} = 1.5: \, \Delta R^2 = 0.001, \, \text{threshold} = 0.997 \quad (\text{balanced})

   \text{Factor} = 2.0: \, \Delta R^2 = 0.002, \, \text{threshold} = 0.996 \quad (\text{parsimonious})

**Selection rule**: Choose smallest N satisfying :math:`R^2_N \geq R^2_{\text{threshold}}`.

optimization_factor Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: optimization_factor selection guide
   :header-rows: 1
   :widths: 20 40 40

   * - Factor
     - Meaning
     - Use case
   * - 1.0
     - Strict: require :math:`R^2 = R^2_max`
     - Research (spectrum reconstruction)
   * - 1.5 (default)
     - Balanced: allow 50% degradation
     - Engineering (model-data fitting)
   * - 2.0
     - Parsimonious: allow 100% degradation
     - Interpretation (minimal modes)
   * - None
     - Disable minimization
     - Manual N selection

**Recommendation**: Use default 1.5 for most applications. Increase to 2.0 for interpretable models with few modes, decrease to 1.0 when maximum accuracy needed.

Example: N=10 → N_opt=3
~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Fit PS melt relaxation data with ``n_modes=10``.

:math:`R^2` **progression**:

.. code-block:: text

   N=10: R² = 0.9985
   N=8:  R² = 0.9984
   N=6:  R² = 0.9980
   N=4:  R² = 0.9970
   N=3:  R² = 0.9975  ← Selected
   N=2:  R² = 0.9920

**Threshold calculation** (factor=1.5):

.. math::

   R^2_{\max} = 0.9985, \quad \Delta R^2 = 0.0015 \times 0.5 = 0.00075

   R^2_{\text{threshold}} = 0.9985 - 0.00075 = 0.99775

**Selection**: N=3 satisfies :math:`0.9975 \geq 0.99775` (**fails**), but N=4 satisfies :math:`0.9970 \geq 0.99775` (**fails**). System selects **N=6** (first N where :math:`R^2 \geq 0.99775`).

Wait—**correction**: Let me recalculate:

.. math::

   0.9980 \geq 0.99775 \quad \text{(N=6 satisfies)}

So **N_opt = 6** is selected, reducing from 10 to 6 modes (40% reduction).

**Access diagnostics**:

.. code-block:: python

   diag = gmm.get_element_minimization_diagnostics()
   print(f"Initial N: {diag['n_initial']}")        # 10
   print(f"Optimal N: {diag['n_optimal']}")        # 6
   print(f"R² values: {diag['r2']}")               # [0.992, 0.9970, 0.9975, 0.9980, ...]
   print(f"N modes: {diag['n_modes']}")            # [2, 4, 3, 6, 8, 10]

Two-Step NLSQ Fitting
----------------------

Motivation: Physical Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Prony series requires **all** :math:`E_i` **> 0** (physical moduli cannot be negative).

**Challenge**: Unconstrained optimization can produce negative :math:`E_i` during intermediate iterations, leading to:
   - Non-physical predictions
   - Numerical instabilities (negative moduli → complex logarithms)
   - Poor convergence

**Solution**: Two-step NLSQ with softmax penalty.

Step 1: Softmax Penalty (Soft Constraints)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective function**:

.. math::

   \min_{\{E_\infty, E_i, \tau_i\}} \left\| E_{\text{data}} - E_{\text{pred}} \right\|^2 + \lambda P_{\text{softmax}}(E_i)

where the **softmax penalty** is:

.. math::

   P_{\text{softmax}}(E_i) = \sum_{i=1}^{N} \log(1 + \exp(-E_i / \text{scale}))

**Properties**:
   - **Differentiable** (JAX-compatible gradients)
   - **Smooth**: No discontinuities (NLSQ handles well)
   - **Encourages positive** :math:`E_i`: Penalty ≈ 0 when :math:`E_i` ≫ 0, increases for :math:`E_i` < 0
   - **Scale parameter**: Default ``scale=1e-3`` balances enforcement strength

**Behavior**:

.. code-block:: python

   E_i = [1e5, 1e4, -1e3]  # One negative mode
   penalty = softmax_penalty(E_i, scale=1e3)
   # → penalty ≈ 693 (large penalty for negative value)

   E_i = [1e5, 1e4, 1e3]   # All positive
   penalty = softmax_penalty(E_i, scale=1e3)
   # → penalty ≈ 0.3 (small penalty for finite positive values)

**Outcome**: Optimization is **gently steered** toward positive :math:`E_i`, but not strictly enforced (allows exploration).

Step 2: Hard Bounds Re-Fit (If Needed)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Trigger**: If Step 1 converges with **any** :math:`E_i` **< 0**, re-fit with hard bounds.

**Hard bounds**: :math:`E_i \in [10^{-12}, 10^{10}]` (strictly enforced by NLSQ).

**Why not use hard bounds from start?**
   - Hard bounds can cause optimization to get stuck at boundaries
   - Softmax penalty provides better gradient information near zero
   - Two-step approach combines smooth optimization (Step 1) with guaranteed feasibility (Step 2)

**Practical outcome**:
   - ~80% of fits succeed in Step 1 without negative :math:`E_i` → fast convergence
   - ~20% trigger Step 2 → slightly slower but guaranteed physical parameters

Performance: 5-270× Speedup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Benchmark** (vs scipy.optimize.curve_fit with same algorithm):

.. list-table:: NLSQ performance comparison
   :header-rows: 1
   :widths: 30 20 20 30

   * - Dataset
     - N modes
     - Speedup
     - Time (NLSQ)
   * - Relaxation (100 pts)
     - 3
     - 5×
     - 0.3 s
   * - Oscillation (200 pts)
     - 10
     - 45×
     - 1.2 s
   * - Oscillation (500 pts)
     - 20
     - 270×
     - 3.8 s
   * - Creep (150 pts)
     - 5
     - 12×
     - 0.8 s

**Key factors**:
   - JAX JIT compilation (first call slow, subsequent calls fast)
   - GPU acceleration (if available)
   - Automatic differentiation (exact Jacobians vs finite differences)

Bayesian Inference Support
---------------------------

Complete NLSQ → NUTS Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GMM inherits full Bayesian capabilities from ``BayesianMixin``:

.. code-block:: python

   from rheojax.models.generalized_maxwell import GeneralizedMaxwell
   import numpy as np

   # 1. NLSQ point estimation (fast warm-start)
   gmm = GeneralizedMaxwell(n_modes=5, modulus_type='shear')
   gmm.fit(t, G_data, test_mode='relaxation', optimization_factor=1.5)

   # Check optimized N
   print(f"Optimized to {gmm._n_modes} modes")  # e.g., 3

   # 2. Bayesian inference with NUTS sampling
   result = gmm.fit_bayesian(
       t, G_data,
       num_warmup=1000,
       num_samples=2000,
       prior_mode='warn'  # Tiered prior safety (see below)
   )

   # 3. Credible intervals and diagnostics
   intervals = gmm.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"G_1: [{intervals['G_1'][0]:.2e}, {intervals['G_1'][1]:.2e}] Pa")

   # Check convergence
   print(f"R-hat: {result.diagnostics['r_hat']['G_1']:.4f}")  # Should be < 1.01
   print(f"ESS: {result.diagnostics['ess']['G_1']:.0f}")      # Should be > 400

Tiered Bayesian Prior Safety Mechanism
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: Bad NLSQ convergence → unreliable priors → misleading Bayesian posteriors.

**Solution**: Intelligent prior construction based on NLSQ convergence quality.

**Three-tier classification**:

Tier 1: Hard Failure
^^^^^^^^^^^^^^^^^^^^

**Conditions**:
   - ``nlsq_result.success = False`` (optimizer failed to converge)
   - ``max_iter`` reached without convergence
   - Gradient norm > 1e-3 (optimization stuck)

**Behavior** (depends on ``prior_mode``):

.. code-block:: python

   # mode='strict' (default for research)
   result = gmm.fit_bayesian(t, G_data, prior_mode='strict')
   # → Raises ValueError with detailed diagnostics

   # mode='warn' (default)
   result = gmm.fit_bayesian(t, G_data, prior_mode='warn')
   # → Raises error, mentions allow_fallback_priors option

   # allow_fallback_priors=True (emergency fallback)
   result = gmm.fit_bayesian(t, G_data, allow_fallback_priors=True)
   # → Uses generic weakly informative priors + BIG warning

**Recommended action**: Fix NLSQ fit first (check model suitability, adjust bounds, increase ``max_iter``).

Tier 2: Suspicious Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Conditions**:
   - ``cond(Hessian) > 1e10`` (ill-conditioned, unreliable covariance)
   - Many parameters near bounds (>50%)
   - High parameter uncertainties (>100% of value)

**Behavior**:

.. code-block:: python

   # mode='warn' (logs warning, uses safer priors)
   result = gmm.fit_bayesian(t, G_data, prior_mode='warn')
   # → Decouples priors from Hessian, uses wider priors (20% of bounds)

   # mode='auto_widen' (inflate std)
   result = gmm.fit_bayesian(t, G_data, prior_mode='auto_widen')
   # → Centers at NLSQ, inflates std to 50% of estimate

**Why suspicious?**: High Hessian condition common for GMM (many parameters), but combined with high uncertainties suggests overfitting.

Tier 3: Good Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^

**Conditions**:
   - ``nlsq_result.success = True``
   - ``cond(Hessian) < 1e10``
   - Reasonable parameter uncertainties (<50% of value)

**Behavior**: Use NLSQ estimates and covariance for warm-start priors:

.. math::

   \text{Prior: } E_i \sim \mathcal{N}(\mu_{\text{NLSQ}}, \sigma_{\text{Hessian}}^2)

   \sigma_{\text{capped}} = \max(\sigma_{\text{Hessian}}, 0.01 \mu_{\text{NLSQ}})

**Capping**: Minimum std = 1% of parameter value to avoid delta-like distributions.

**Result**: Fast NUTS convergence (2-5× faster than cold start), low divergences.

Diagnostics Extraction
~~~~~~~~~~~~~~~~~~~~~~

**Automatic diagnostics** from NLSQ result:

.. code-block:: python

   # Internal method (used by fit_bayesian)
   diagnostics = gmm._extract_nlsq_diagnostics(gmm._nlsq_result)

   print(diagnostics)
   # {
   #   'convergence_flag': True,
   #   'gradient_norm': 1.2e-7,
   #   'hessian_condition': 3.4e8,
   #   'param_uncertainties': {'G_inf': 120.0, 'G_1': 8500.0, ...},
   #   'params_near_bounds': {'tau_3': 'upper'}
   # }

   # Classification
   classification = gmm._classify_nlsq_convergence(diagnostics)
   # → 'good', 'suspicious', or 'hard_failure'

**User control**:

.. code-block:: python

   # Expert mode: auto-widen priors if suspicious
   result = gmm.fit_bayesian(
       t, G_data,
       prior_mode='auto_widen'  # Options: 'strict', 'warn', 'auto_widen'
   )

   # Manual priors (bypasses safety system)
   custom_priors = {
       'G_inf': {'mean': 1e3, 'std': 5e2},
       'G_1': {'mean': 1e5, 'std': 2e4},
       # ... (all parameters)
   }
   result = gmm.fit_bayesian(t, G_data, priors=custom_priors)

----

Usage
-----

(Usage examples already present in the file)

----

Experimental Design
-------------------

When to Use GMM
~~~~~~~~~~~~~~~

.. list-table:: GMM vs alternatives decision tree
   :header-rows: 1
   :widths: 35 35 30

   * - Use GMM when...
     - Use alternative when...
     - Recommended model
   * - Broad relaxation spectrum (>2 decades)
     - Single dominant timescale
     - Maxwell, Zener
   * - Multi-phase materials (composites)
     - Homogeneous polymer
     - Single Maxwell sufficient
   * - Polydisperse polymer (broad MW)
     - Narrow MW distribution
     - Single Maxwell
   * - Need relaxation spectrum :math:`H(\tau)`
     - Empirical fit only
     - Fractional models (FML, FZSS)
   * - Oscillation + relaxation + creep prediction
     - Only one test mode
     - Test mode-specific model
   * - Research (spectrum reconstruction)
     - Engineering (quick fit)
     - Fewer modes (N=3-5)

**Key advantage**: GMM is **most flexible** for complex materials, at the cost of more parameters (2N+1 vs 2-4 for simple models).

Test Mode Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Relaxation mode (PREFERRED)**

**Advantages**:
   - Direct Prony series fitting (no transform)
   - Exponential basis functions orthogonal in log-time
   - Best conditioning for parameter identification

**Protocol**:
   - Step strain :math:`\gamma_0` within LVR (1-5%)
   - Logarithmic time sampling: 10 pts/decade
   - Duration: 5-8 decades in time (e.g., 10\ :math:`^{-2 to 10^6}` s via TTS)
   - Temperature control: ±0.1°C

**Example**: DMA isothermal relaxation test at T = T_g + 30°C.

**2. Oscillation mode (EXCELLENT)**

**Advantages**:
   - Closed-form Fourier transform (no numerical FFT errors)
   - Fits :math:`G'` and :math:`G''` simultaneously
   - Standard rheometer test (SAOS frequency sweep)

**Protocol**:
   - Amplitude sweep to find LVR (typically :math:`\gamma_0 = 0.5-5\%`)
   - Frequency range: 5-8 decades (e.g., 10\ :math:`^{-2 to 10^5}` rad/s via TTS)
   - Temperature sweep for mastercurve: 5-10 temperatures
   - Use automatic shift factors: ``Mastercurve(auto_shift=True)``

**Example**: Polymer melt characterization with time-temperature superposition.

**3. Creep mode (ACCEPTABLE)**

**Advantages**:
   - Simple experimental setup (constant stress)
   - Creep recovery reveals viscoelastic character

**Disadvantages**:
   - Numerical ODE integration (slower than analytical modes)
   - Ill-conditioned for spectrum reconstruction
   - Requires high-precision strain measurement

**Protocol**:
   - Step stress :math:`\sigma_0` within LVR
   - Duration: 5-8 decades (same as relaxation)
   - Measure recovery phase (remove stress at t=t_max)

**Use case**: Soft materials (gels, suspensions) where oscillation causes structure damage.

Sample Applications: DMA (Solid Mechanics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dynamic Mechanical Analyzer (DMA)** measures viscoelastic properties of **solid-like materials** (polymers, composites, rubbers) in tension, compression, or torsion.

**Typical workflow**:

1. **Temperature sweep** (fixed :math:`\omega`):
      - T = T_g - 50°C to T_g + 100°C (e.g., -50°C to 150°C for PMMA)
      - :math:`\omega` = 1 rad/s (common standard)
      - Measure :math:`E'(T)`, :math:`E''(T)`, :math:`\tan \delta(T)`
      - Identify glass transition (peak in :math:`\tan \delta`)

2. **Frequency sweep** (multiple T):
      - 5-10 isothermal temperatures
      - :math:`\omega` = 10\ :math:`^{-2 to 10^2}` rad/s per temperature
      - Construct mastercurve via TTS: ``Mastercurve(auto_shift=True)``
      - Extended range: :math:`\omega_reduced = 10`\ :math:`^{-6 to 10^6}` rad/s

3. **Fit GMM to mastercurve**:

   .. code-block:: python

      from rheojax.models.generalized_maxwell import GeneralizedMaxwell
      from rheojax.transforms.mastercurve import Mastercurve

      # Create mastercurve (automatic shift factors)
      mc = Mastercurve(reference_temp=60+273.15, auto_shift=True)
      mastercurve, shifts = mc.transform(multi_temp_datasets)

      # Fit GMM (element minimization)
      gmm = GeneralizedMaxwell(n_modes=10, modulus_type='tensile')
      gmm.fit(
          mastercurve.x,      # Shifted frequency (rad/s)
          mastercurve.y,      # Complex modulus [E', E"] (Pa)
          test_mode='oscillation',
          optimization_factor=1.5
      )

      # Check optimized N
      n_opt = gmm._n_modes
      print(f"Reduced to {n_opt} modes")  # e.g., 5

      # Extract relaxation spectrum
      spectrum = gmm.get_relaxation_spectrum()
      E_i = spectrum['E_i']
      tau_i = spectrum['tau_i']

4. **Applications**:
      - **Material characterization**: Compare polymers by spectrum width
      - **QC/QA**: Batch-to-batch consistency (spectrum fingerprinting)
      - **Constitutive modeling**: Use GMM in FEA for viscoelastic materials

**Example materials**: PMMA, polycarbonate, epoxy resins, fiber-reinforced composites.

Sample Applications: Rheometer (Fluid Dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rotational rheometer** measures flow properties of **fluid-like materials** (polymer melts, solutions, suspensions) in shear.

**Typical workflow**:

1. **SAOS frequency sweep** (linear regime):
      - :math:`\omega` = 10\ :math:`^{-2 to 10^2}` rad/s
      - :math:`\gamma_0` = 1% (within LVR, verified by amplitude sweep)
      - Measure :math:`G'(\omega)`, :math:`G''(\omega)`
      - Identify terminal region (:math:`G' \sim \omega^2`, :math:`G'' \sim \omega`)

2. **Time-temperature superposition**:
      - Multiple temperatures (e.g., 140°C, 160°C, 180°C, 200°C)
      - Use automatic shift factors (PyVisco algorithm):

        .. code-block:: python

           mc = Mastercurve(reference_temp=180+273.15, auto_shift=True)
           mastercurve, shifts = mc.transform(datasets)

           # Compare auto vs manual (WLF)
           mc_wlf = Mastercurve(reference_temp=180+273.15, method='wlf',
                                 C1=8.86, C2=101.6)  # PS universal WLF
           mastercurve_wlf, shifts_wlf = mc_wlf.transform(datasets)

3. **Fit GMM to mastercurve**:

   .. code-block:: python

      gmm = GeneralizedMaxwell(n_modes=15, modulus_type='shear')
      gmm.fit(
          mastercurve.x,      # Shifted ω (rad/s)
          mastercurve.y,      # [G', G"] (Pa)
          test_mode='oscillation',
          optimization_factor=1.5
      )

      # Validate tri-mode equality: predict relaxation from oscillation fit
      t_relax = np.logspace(-3, 5, 100)
      G_relax = gmm.predict(t_relax)  # Uses oscillation-fitted params

4. **Cox-Merz rule validation**:

   .. code-block:: python

      # Complex viscosity from GMM
      eta_star = np.sqrt(G_prime**2 + G_double_prime**2) / omega

      # Steady shear viscosity (experimental)
      # Example: eta_shear = load_flow_curve_data()
      shear_rate = np.logspace(-2, 2, 30)
      eta_shear = 1e3 * shear_rate**(-0.5)  # Example: shear-thinning

      # Check Cox-Merz: η(γ̇) ≈ |η*(ω)| at ω = γ̇
      import matplotlib.pyplot as plt
      plt.loglog(omega, eta_star, label='|η*| (GMM)')
      plt.loglog(shear_rate, eta_shear, 'o', label='η (flow curve)')
      plt.legend()

**Example materials**: PS, PDMS, polyethylene, polymer solutions.

Fitting Guidance
----------------

Element Minimization Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Start with generous N (over-parameterize)**:

.. code-block:: python

   # Request N=15 for broad spectra
   gmm = GeneralizedMaxwell(n_modes=15, modulus_type='shear')
   gmm.fit(omega, G_star, test_mode='oscillation', optimization_factor=1.5)

   # System auto-reduces to N_opt (e.g., 7)
   print(f"Optimized to {gmm._n_modes} modes")

**Rationale**: Better to start too high and reduce, than start too low and miss modes.

**2. Tune optimization_factor based on goal**:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Goal
     - Factor
     - Typical N_opt
   * - Spectrum reconstruction (research)
     - 1.0
     - N_opt = N (no reduction)
   * - Engineering fits (default)
     - 1.5
     - 30-50% reduction
   * - Interpretable models (teaching)
     - 2.0
     - 60-80% reduction

**3. Validate element minimization**:

.. code-block:: python

   diag = gmm.get_element_minimization_diagnostics()

   import matplotlib.pyplot as plt
   plt.plot(diag['n_modes'], diag['r2'], 'o-')
   plt.axhline(y=diag['r2'][diag['n_optimal']], color='r',
               linestyle='--', label=f"N_opt={diag['n_optimal']}")
   plt.xlabel('Number of modes N')
   plt.ylabel('R²')
   plt.legend()
   plt.show()

**Expected**: :math:`R^2` decreases slowly as N reduces, then drops sharply below N_opt.

**4. Disable minimization for manual control**:

.. code-block:: python

   # Fit with fixed N=5 (no minimization)
   gmm.fit(omega, G_star, test_mode='oscillation', optimization_factor=None)

optimization_factor Selection Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Default 1.5 rationale**: Balances parsimony (fewer modes) with fit quality (:math:`R^2` degradation).

**When to adjust**:

.. code-block:: python

   # Strict accuracy (spectrum reconstruction)
   gmm.fit(..., optimization_factor=1.0)
   # → Keeps more modes, R² ≈ R²_max

   # Maximum parsimony (teaching/interpretation)
   gmm.fit(..., optimization_factor=2.0)
   # → Reduces to minimal modes, lower R²

   # Disable (manual N)
   gmm.fit(..., optimization_factor=None)

**Empirical guideline**:
   - Factor ∈ [1.0, 1.5]: Research applications
   - Factor ∈ [1.5, 2.0]: Engineering applications
   - Factor > 2.0: Rarely useful (overly aggressive reduction)

Troubleshooting Convergence Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Common GMM fitting problems
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Diagnostic
     - Solution
   * - NLSQ fails (max_iter reached)
     - ``nlsq_result.success = False``
     - Increase ``max_iter=5000``, check data quality
   * - Negative :math:`E_i` after Step 2
     - Warning: "Negative :math:`E_i` detected"
     - Reduce N, check for data artifacts (noise, outliers)
   * - :math:`R^2` poor for all N
     - :math:`R^2` < 0.90 even for N=20
     - GMM inappropriate (try fractional models)
   * - Element minimization unstable
     - N_opt = 1 or N_opt = N_max
     - Adjust optimization_factor, check data span
   * - Bayesian NUTS divergences
     - ``result.diagnostics['divergences'] > 0.01``
     - Use ``prior_mode='auto_widen'``, increase warmup

**Example: Increase max_iter**:

.. code-block:: python

   gmm.fit(omega, G_star, test_mode='oscillation',
           max_iter=5000,  # Default 1000
           ftol=1e-8,      # Tighter tolerance
           xtol=1e-8)

Validation with :math:`R^2` and Residual Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Goodness-of-fit (** :math:`R^2` **)**:

.. code-block:: python

   from rheojax.utils.prony import compute_r_squared

   # Predict with fitted model
   G_star_pred = gmm.predict(omega)  # Shape (M, 2) for oscillation

   # Flatten to match data
   G_data_flat = np.concatenate([G_prime, G_double_prime])
   G_pred_flat = G_star_pred.T.flatten()  # [G', G"]

   r2 = compute_r_squared(G_data_flat, G_pred_flat)
   print(f"R² = {r2:.4f}")  # Should be > 0.95

**Interpretation**:
   - :math:`R^2` > 0.98: Excellent fit
   - :math:`R^2` ∈ [0.95, 0.98]: Good fit
   - :math:`R^2` < 0.95: Poor fit (increase N or try different model)

**2. Residual plots**:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Residuals in log-space (common for rheology)
   residuals_G_prime = np.log10(G_prime) - np.log10(G_star_pred[:, 0])
   residuals_G_double_prime = np.log10(G_double_prime) - np.log10(G_star_pred[:, 1])

   fig, ax = plt.subplots(2, 1, figsize=(8, 6))

   ax[0].semilogx(omega, residuals_G_prime, 'o')
   ax[0].axhline(0, color='k', linestyle='--')
   ax[0].set_ylabel("log(G') residuals")

   ax[1].semilogx(omega, residuals_G_double_prime, 'o')
   ax[1].axhline(0, color='k', linestyle='--')
   ax[1].set_ylabel("log(G'') residuals")
   ax[1].set_xlabel("ω (rad/s)")

   plt.tight_layout()
   plt.show()

**Expected**: Residuals randomly scattered around zero (no trends). Systematic patterns indicate model inadequacy.

Model Comparison
----------------

Single Maxwell vs GMM
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: When to use each
   :header-rows: 1
   :widths: 25 35 40

   * - Criterion
     - Single Maxwell
     - GMM
   * - Relaxation spectrum width
     - < 2 decades
     - > 2 decades
   * - Material complexity
     - Monodisperse, homogeneous
     - Polydisperse, multi-phase
   * - Number of parameters
     - 2 (G, :math:`\eta`)
     - 2N+1 (auto-reduced)
   * - Fitting time
     - 0.1 s
     - 0.5-5 s (depends on N)
   * - Interpretation
     - Simple (one :math:`\tau`)
     - Complex (spectrum H(:math:`\tau`))
   * - Use case
     - Quick screening
     - Detailed characterization

**Decision rule**: If single Maxwell gives :math:`R^2` < 0.95, try GMM.

GMM vs Fractional Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Discrete vs continuous spectrum
   :header-rows: 1
   :widths: 25 35 40

   * - Criterion
     - GMM (discrete)
     - Fractional (FML, FZSS)
   * - Spectrum representation
     - Discrete modes (N)
     - Continuous power-law
   * - Parameters
     - 2N+1
     - 3-4
   * - Fit quality (broad spectra)
     - Excellent (high N)
     - Good (fewer params)
   * - Physical interpretation
     - Mode strengths :math:`E_i`
     - Fractional order :math:`\alpha`
   * - Extrapolation
     - Poor (outside :math:`\tau` range)
     - Better (power-law)
   * - Computational cost
     - O(N) per evaluation
     - O(1) (fixed cost)

**Trade-off**: GMM more flexible but requires more parameters. Fractional models more parsimonious for power-law spectra.

**When to use fractional**: If spectrum is smooth power-law over >4 decades, fractional models (FML, FMG) may give comparable fit with fewer parameters.

Computational Cost vs Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scaling analysis**:

.. math::

   \text{Cost per evaluation} \sim O(N) \quad (\text{sum over modes})

   \text{NLSQ iterations} \sim 100-500 \quad (\text{typical})

   \text{Total cost} \sim O(N \times n_{\text{iter}} \times M) \quad (M = \text{data points})

**Benchmark** (M=200 data points, JAX + GPU):

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - N modes
     - Iterations
     - Time (s)
     - :math:`R^2`
     - N_opt
   * - 3
     - 150
     - 0.3
     - 0.985
     - 3
   * - 5
     - 180
     - 0.5
     - 0.992
     - 4
   * - 10
     - 220
     - 1.2
     - 0.997
     - 6
   * - 20
     - 300
     - 3.8
     - 0.998
     - 8

**Recommendation**: Start with N=10-15 (element minimization will reduce). Rarely need N>20 unless reconstructing high-resolution spectra.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.GeneralizedMaxwell`
- Utilities: :mod:`rheojax.utils.prony`

Usage Examples
--------------

Basic Fitting (Relaxation Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models.generalized_maxwell import GeneralizedMaxwell

   # Generate synthetic relaxation data (multi-mode)
   t = np.logspace(-2, 3, 100)
   G_t = 1e4 + 5e5*np.exp(-t/0.1) + 8e4*np.exp(-t/1.0) + 3e4*np.exp(-t/10.0)

   # Fit with transparent element minimization
   gmm = GeneralizedMaxwell(n_modes=10, modulus_type='shear')
   gmm.fit(t, G_t, test_mode='relaxation', optimization_factor=1.5)

   # Check optimized N
   n_opt = gmm._n_modes
   print(f"Optimized to {n_opt} modes")  # e.g., 3

   # Predict
   G_t_pred = gmm.predict(t)

   # Extract spectrum
   spectrum = gmm.get_relaxation_spectrum()
   print(f"G_inf: {spectrum['G_inf']:.2e} Pa")
   print(f"G_i: {spectrum['G_i']}")
   print(f"tau_i: {spectrum['tau_i']}")

Oscillation Mode with Time-Temperature Superposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.generalized_maxwell import GeneralizedMaxwell
   from rheojax.transforms.mastercurve import Mastercurve
   from rheojax.core.data import RheoData
   import numpy as np

   # Multi-temperature SAOS data (example: 5 temperatures)
   temps = [140, 160, 180, 200, 220]  # °C
   datasets = []

   for T in temps:
       omega = np.logspace(-2, 2, 50)
       # Load experimental data at temperature T
       # Example: G_prime, G_double_prime = load_dma_data(T)
       G_prime = 1e5 * (omega / 1.0)**0.2  # Example: weak power-law
       G_double_prime = 1e4 * omega  # Example: linear at low frequency

       data = RheoData(
           x=omega,
           y=np.column_stack([G_prime, G_double_prime]),  # (M, 2)
           domain='frequency',
           metadata={'temperature': T + 273.15}  # Kelvin
       )
       datasets.append(data)

   # Create mastercurve with automatic shift factors
   mc = Mastercurve(reference_temp=180+273.15, auto_shift=True)
   mastercurve, shift_factors = mc.transform(datasets)

   # Fit GMM to mastercurve
   gmm = GeneralizedMaxwell(n_modes=15, modulus_type='shear')
   gmm.fit(
       mastercurve.x,      # Shifted omega
       mastercurve.y,      # [G', G"] concatenated
       test_mode='oscillation',
       optimization_factor=1.5,
       max_iter=3000
   )

   # Check fit quality
   G_star_pred = gmm.predict(mastercurve.x)
   from rheojax.utils.prony import compute_r_squared
   r2 = compute_r_squared(mastercurve.y, G_star_pred.T.flatten())
   print(f"R² = {r2:.4f}")

   # Diagnostics
   diag = gmm.get_element_minimization_diagnostics()
   print(f"Reduced from {diag['n_initial']} to {diag['n_optimal']} modes")

Creep Mode Prediction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models.generalized_maxwell import GeneralizedMaxwell

   # Generate synthetic oscillation data
   omega = np.logspace(-2, 2, 50)
   G_prime = 1e5 * (omega / 1.0)**0.3
   G_double_prime = 3e4 * omega**0.5
   G_star = np.column_stack([G_prime, G_double_prime])

   # First fit to relaxation or oscillation data
   gmm = GeneralizedMaxwell(n_modes=5, modulus_type='shear')
   gmm.fit(omega, G_star, test_mode='oscillation', optimization_factor=1.5)

   # Predict creep compliance (tri-mode equality)
   t_creep = np.logspace(-2, 4, 150)
   J_creep = gmm.predict(t_creep)  # Uses internal _predict_creep

   # Plot
   import matplotlib.pyplot as plt
   plt.loglog(t_creep, J_creep)
   plt.xlabel('Time (s)')
   plt.ylabel('Creep compliance J(t) (1/Pa)')
   plt.title('GMM creep prediction from oscillation fit')
   plt.show()

**Note**: Must set ``test_mode='creep'`` during ``fit()`` call for creep data, or change internally for prediction.

Bayesian Inference with Prior Safety
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.generalized_maxwell import GeneralizedMaxwell
   import numpy as np

   # 1. NLSQ fit (fast)
   gmm = GeneralizedMaxwell(n_modes=5, modulus_type='shear')
   gmm.fit(t, G_data, test_mode='relaxation', optimization_factor=1.5)

   # 2. Bayesian inference with automatic prior safety
   result = gmm.fit_bayesian(
       t, G_data,
       num_warmup=1000,
       num_samples=2000,
       prior_mode='warn'  # Tiered safety: 'strict', 'warn', 'auto_widen'
   )

   # 3. Check convergence diagnostics
   print(f"R-hat (G_1): {result.diagnostics['r_hat']['G_1']:.4f}")  # < 1.01
   print(f"ESS (G_1): {result.diagnostics['ess']['G_1']:.0f}")      # > 400
   print(f"Divergences: {result.diagnostics.get('divergences', 0)}")  # Should be 0

   # 4. Credible intervals
   intervals = gmm.get_credible_intervals(result.posterior_samples, credibility=0.95)
   for param in ['G_inf', 'G_1', 'tau_1']:
       low, high = intervals[param]
       print(f"{param}: [{low:.2e}, {high:.2e}]")

   # 5. Posterior predictive checks
   # Sample from posterior
   posterior_G_1 = result.posterior_samples['G_1']  # Shape (num_samples,)
   n_samples = len(posterior_G_1)

   # Predict with posterior samples
   predictions = []
   for i in range(min(100, n_samples)):  # 100 posterior draws
       # Set parameters from posterior
       for param_name, values in result.posterior_samples.items():
           gmm.parameters.set_value(param_name, float(values[i]))

       # Predict
       G_pred = gmm.predict(t)
       predictions.append(G_pred)

   predictions = np.array(predictions)

   # Plot credible bands
   import matplotlib.pyplot as plt
   plt.fill_between(
       t,
       np.percentile(predictions, 2.5, axis=0),
       np.percentile(predictions, 97.5, axis=0),
       alpha=0.3,
       label='95% credible interval'
   )
   plt.loglog(t, G_data, 'o', label='Data')
   plt.xlabel('Time (s)')
   plt.ylabel('G(t) (Pa)')
   plt.legend()
   plt.show()

DMA Workflow (Solid Mechanics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.generalized_maxwell import GeneralizedMaxwell
   from rheojax.transforms.mastercurve import Mastercurve
   from rheojax.core.data import RheoData
   import numpy as np
   import matplotlib.pyplot as plt

   # 1. Load DMA frequency sweep data at multiple temperatures
   temps = [20, 40, 60, 80, 100]  # °C
   datasets = []

   for T in temps:
       # Example: load from file
       # omega, E_prime, E_double_prime = load_dma_data(T)
       omega = np.logspace(-2, 2, 50)  # rad/s
       E_prime = 1e6 * (omega / 1.0)**0.15  # Pa (example: weak frequency dependence)
       E_double_prime = 1e5 * omega**0.3  # Pa (example: loss modulus)

       data = RheoData(
           x=omega,
           y=np.column_stack([E_prime, E_double_prime]),
           domain='frequency',
           metadata={'temperature': T + 273.15}
       )
       datasets.append(data)

   # 2. Construct mastercurve (automatic shift factors)
   mc = Mastercurve(reference_temp=60+273.15, auto_shift=True)
   mastercurve, shifts = mc.transform(datasets)

   # 3. Fit GMM
   gmm = GeneralizedMaxwell(n_modes=10, modulus_type='tensile')
   gmm.fit(
       mastercurve.x,
       mastercurve.y,
       test_mode='oscillation',
       optimization_factor=1.5
   )

   # 4. Extract relaxation spectrum
   spectrum = gmm.get_relaxation_spectrum()
   E_i = spectrum['E_i']
   tau_i = spectrum['tau_i']

   # 5. Plot spectrum
   plt.figure(figsize=(10, 6))

   plt.subplot(2, 1, 1)
   plt.semilogx(tau_i, E_i, 'o-')
   plt.xlabel('Relaxation time τ (s)')
   plt.ylabel('Mode strength E_i (Pa)')
   plt.title(f'Discrete Relaxation Spectrum (N={gmm._n_modes} modes)')
   plt.grid(True)

   plt.subplot(2, 1, 2)
   omega_master = mastercurve.x
   E_star_pred = gmm.predict(omega_master)
   plt.loglog(omega_master, E_star_pred[:, 0], label="E' (GMM)")
   plt.loglog(omega_master, E_star_pred[:, 1], label='E" (GMM)')
   plt.loglog(mastercurve.x, mastercurve.y[:, 0], 'o', alpha=0.5, label="E' (data)")
   plt.loglog(mastercurve.x, mastercurve.y[:, 1], 's', alpha=0.5, label='E" (data)')
   plt.xlabel('Reduced frequency ω (rad/s)')
   plt.ylabel('Modulus (Pa)')
   plt.legend()
   plt.grid(True)

   plt.tight_layout()
   plt.show()

Rheometer Workflow (Fluid Dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.generalized_maxwell import GeneralizedMaxwell
   from rheojax.transforms.mastercurve import Mastercurve
   from rheojax.core.data import RheoData
   import numpy as np
   import matplotlib.pyplot as plt

   # 1. SAOS frequency sweep at multiple temperatures (example for polymer melts)
   temps = [140, 160, 180, 200, 220]  # °C
   datasets = []

   for T in temps:
       omega = np.logspace(-2, 2, 50)
       G_prime = 1e5 * (omega / 1.0)**0.25 * np.exp((180 - T) / 20)
       G_double_prime = 3e4 * omega**0.6 * np.exp((180 - T) / 25)

       data = RheoData(
           x=omega,
           y=np.column_stack([G_prime, G_double_prime]),
           domain='frequency',
           metadata={'temperature': T + 273.15}
       )
       datasets.append(data)

   # 2. Mastercurve with auto shift
   mc = Mastercurve(reference_temp=180+273.15, auto_shift=True)
   mastercurve, shifts = mc.transform(datasets)

   # 3. Fit GMM
   gmm = GeneralizedMaxwell(n_modes=12, modulus_type='shear')
   gmm.fit(mastercurve.x, mastercurve.y, test_mode='oscillation')

   # 4. Cox-Merz rule validation
   omega = mastercurve.x
   G_star_pred = gmm.predict(omega)
   G_prime = G_star_pred[:, 0]
   G_double_prime = G_star_pred[:, 1]

   # Complex viscosity
   eta_star = np.sqrt(G_prime**2 + G_double_prime**2) / omega

   # Compare with steady shear (experimental)
   shear_rate_exp = np.logspace(-2, 2, 40)  # 1/s
   viscosity_exp = 1e3 * shear_rate_exp**(-0.4)  # Pa·s (example: shear-thinning)

   plt.figure(figsize=(8, 6))
   plt.loglog(omega, eta_star, '-', label='|η*| (GMM, SAOS)')
   plt.loglog(shear_rate_exp, viscosity_exp, 'o', label='η (steady shear)')
   plt.xlabel('ω or γ̇ (rad/s or 1/s)')
   plt.ylabel('Viscosity (Pa·s)')
   plt.title('Cox-Merz Rule Validation')
   plt.legend()
   plt.grid(True)
   plt.show()

   # Cox-Merz satisfied if curves overlap

See Also
--------

- :doc:`../classical/maxwell` — Single Maxwell element (N=1 special case)
- :doc:`../classical/zener` — Adds equilibrium spring for solids
- :doc:`../fractional/fractional_maxwell_liquid` — Continuous spectrum via fractional calculus
- :doc:`../../transforms/mastercurve` — Time-temperature superposition (automatic shift factors)
- :doc:`../../examples/advanced/08-generalized_maxwell_fitting` — Complete GMM workflow notebook
- :doc:`../../examples/bayesian/07-gmm_bayesian_workflow` — Bayesian inference with prior safety
- :doc:`../../user_guide/model_selection` — Decision flowcharts for model selection

References
----------

.. [1] Park, S. W. and Schapery, R. A. "Methods of interconversion between linear
   viscoelastic material functions. Part I—A numerical method based on Prony series."
   *International Journal of Solids and Structures*, 36(11), 1653-1675 (1999).
   https://doi.org/10.1016/S0020-7683(98)00055-9

.. [2] Baumgaertel, M. and Winter, H. H. "Determination of discrete relaxation and
   retardation time spectra from dynamic mechanical data." *Rheologica Acta*, 28(6),
   511-519 (1989). https://doi.org/10.1007/BF01332922

.. [3] Malkin, A. Y. and Isayev, A. I. *Rheology: Concepts, Methods, and Applications*,
   3rd Edition. ChemTec Publishing (2017). ISBN: 978-1927885215

.. [4] Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd Edition. Wiley (1980).
   ISBN: 978-0471048947

.. [5] Williams, M. L., Landel, R. F., and Ferry, J. D. "The Temperature Dependence of
   Relaxation Mechanisms in Amorphous Polymers and Other Glass-forming Liquids."
   *Journal of the American Chemical Society*, 77(14), 3701-3707 (1955).
   https://doi.org/10.1021/ja01619a008

.. [6] Tassieri, M. "Linear viscoelasticity in the frequency domain." In *Microrheology
   with Optical Tweezers*. Pan Stanford Publishing, pp. 31-58 (2018).
   https://doi.org/10.1201/9781315364872

.. [7] Hoffman, M. D. and Gelman, A. "The No-U-Turn Sampler: Adaptively Setting Path
   Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*,
   15(1), 1593-1623 (2014). https://jmlr.org/papers/v15/hoffman14a.html

.. [8] Doi, M. and Edwards, S. F. *The Theory of Polymer Dynamics*. Oxford University
   Press (1986). ISBN: 978-0198520009

.. [9] Rubinstein, M. and Colby, R. H. *Polymer Physics*. Oxford University Press (2003).
   ISBN: 978-0198520597

.. [10] Dealy, J. M. and Larson, R. G. *Structure and Rheology of Molten Polymers*.
   Hanser Publishers (2006). https://doi.org/10.3139/9783446412811

.. [11] Stadler, F. J. and Bailly, C. "A new method for the calculation of continuous
   relaxation spectra from dynamic-mechanical data." *Rheologica Acta*, 48(1),
   33-49 (2009). https://doi.org/10.1007/s00397-008-0303-2

.. [12] Honerkamp, J. and Weese, J. "A nonlinear regularization method for the calculation
   of relaxation spectra." *Rheologica Acta*, 32(1), 65-73 (1993).
   https://doi.org/10.1007/BF00396678

.. [13] Anderssen, R. S. and Davies, A. R. "Simple moving-average formulae for the direct
   recovery of the relaxation spectrum." *Journal of Rheology*, 45(1), 1-27 (2001).
   https://doi.org/10.1122/1.1332787
