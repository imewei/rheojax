.. _models-tnt:

Transient Network Theory (TNT)
==============================

.. include:: /_includes/transient_network_foundations.rst

Overview
--------

The TNT family in RheoJAX provides 5 model classes spanning 9 distinct physical variants,
from simple single-mode Maxwell analogs to complex multi-species living polymer systems.
All models support the full suite of rheological protocols (flow curves, SAOS, LAOS,
startup, creep, relaxation) with validated predictions against experimental data. The
mathematical framework is implemented in JAX with full automatic differentiation support,
enabling GPU acceleration and Bayesian inference via NumPyro NUTS sampling. See the
foundation box above for the physical basis, key signatures, and constitutive equations
shared across all TNT variants.

.. admonition:: Dual Formulation — Integral vs Differential
   :class: tip

   TNT admits two **mathematically equivalent** perspectives:

   **Differential (conformation tensor ODE)** — the primary RheoJAX implementation:

   .. math::

      \frac{D\mathbf{S}}{Dt} = \boldsymbol{\kappa} \cdot \mathbf{S}
      + \mathbf{S} \cdot \boldsymbol{\kappa}^T
      - \frac{1}{\tau_b(\mathbf{S})}(\mathbf{S} - \mathbf{I})

   **Integral (history / cohort) formulation** — useful for step-strain analysis and
   multi-protocol understanding:

   .. math::

      \boldsymbol{\tau}(t) = \int_{-\infty}^{t} \beta(t') \, S(t,t') \,
      G \bigl[\mathbf{B}(t,t') - \mathbf{I}\bigr] \, dt'

   where :math:`\beta(t')` is the birth rate of chains at time :math:`t'`,
   :math:`S(t,t') = \exp\!\bigl[-\int_{t'}^{t} k_d(s)\,ds\bigr]` is the survival
   probability, and :math:`\mathbf{B}(t,t')` is the Finger deformation tensor.

   The integral form tracks **cohorts** of chains born at time :math:`t'`, each carrying
   its deformation history. The differential form evolves the **ensemble average**
   conformation :math:`\mathbf{S}(t)`. Both yield identical stress predictions. See
   :doc:`tnt_protocols` for the full derivation and numerical methods for each approach.


Model Hierarchy
---------------

::

   TNT Family (5 Classes, 9 Variants)
   │
   ├── TNTSingleMode (Composable, 5 variants)
   │   │   Base: Constant breakage (Tanaka-Edwards)
   │   │         Maxwell-like with tensorial stress
   │   │         Parameters: G, τ_b, η_s
   │   │
   │   ├── breakage="constant" (default)
   │   │   └── Tanaka-Edwards: 1/τ_b = const
   │   │       Simplest TNT, baseline for comparison
   │   │
   │   ├── breakage="bell"
   │   │   └── Force-dependent detachment
   │   │       1/τ_b = (1/τ_0) exp(ν·F/k_B·T)
   │   │       For bio-networks with catch/slip bonds
   │   │       Additional parameter: ν (force sensitivity)
   │   │
   │   ├── breakage="power_law"
   │   │   └── Power-law force weakening
   │   │       1/τ_b = (1/τ_0)(F/F_0)^m
   │   │       Empirical extension, m ~ 1-5
   │   │
   │   ├── stress_type="fene" (FENE-P finite extensibility)
   │   │   └── Chain force: F = 3k_B·T·L²_max/(L²_max - tr(S) + 3)
   │   │       Polymer gels with limited chain stretch
   │   │       Additional parameter: L_max (extensibility)
   │   │       Nonlinear softening at large strains
   │   │
   │   └── xi > 0 (Non-affine slip, Gordon-Schowalter)
   │       └── Partial coupling: S evolves with ξ ∈ [0, 1]
   │           ξ = 0: Affine (full coupling)
   │           ξ = 1: Full slip (isotropic stress)
   │           Reduces N₁ predictions, empirical correction
   │
   ├── TNTLoopBridge (Two-species kinetics)
   │   │   Telechelic polymers with loops + bridges
   │   │   Equilibrium: f_B(eq) = bridge fraction
   │   │   Kinetics: df_B/dt = k_loop→bridge - k_bridge→loop
   │   │
   │   └── Parameters: G_B, G_L, τ_B, τ_L, f_B_eq, k_ex
   │       Only bridges contribute to stress
   │       Loops act as dangling ends (viscous)
   │       7-8 parameters (richer dynamics than SingleMode)
   │
   ├── TNTStickyRouse (Multi-mode sticker dynamics)
   │   │   Rouse modes limited by sticker lifetime
   │   │   τ_p = τ_s + (N/p²)·τ_Rouse (hybrid timescale)
   │   │
   │   └── Parameters: G_N, τ_s, N (chain length), n_modes
   │       Broad relaxation spectrum
   │       Terminal time: τ_terminal ≈ τ_s + N·τ_Rouse
   │       For multi-sticker associating polymers
   │
   ├── TNTCates (Living polymers / wormlike micelles)
   │   │   Scission/recombination kinetics
   │   │   Effective relaxation: τ_d = √(τ_rep · τ_break)
   │   │
   │   └── Parameters: G_N, τ_rep, τ_break, η_s
   │       Single effective mode (faster of rep/break)
   │       Shear-thinning from reduced effective length
   │       For micellar solutions (CTAB, CPCl, etc.)
   │
   └── TNTMultiSpecies (N independent bond types)
       │   Distinct G_i, τ_i for each species
       │   Additive stress: σ_total = Σ_i σ_i
       │
       └── Parameters: {G_i, τ_i} for i=1..N, η_s
           Discrete relaxation spectrum
           For heterogeneous crosslink populations
           N typically 2-5 (more = fit flexibility vs physics)


When to Use Which Model
-----------------------

.. list-table::
   :widths: 20 15 15 15 15 20
   :header-rows: 1

   * - Feature
     - SingleMode
     - LoopBridge
     - StickyRouse
     - Cates
     - MultiSpecies
   * - Material type
     - Physical gels
     - Telechelics
     - Multi-sticker
     - Micelles
     - Mixed networks
   * - Number of params
     - 3-5
     - 7-8
     - 4-6
     - 4-5
     - 2N+1
   * - Relaxation spectrum
     - Single mode
     - Two modes
     - Broad (N modes)
     - Effective single
     - Discrete (N)
   * - Key physics
     - Bond lifetime
     - Loop↔bridge
     - Sticker-limited
     - Scission/recomb
     - Heterogeneity
   * - Recommended for
     - Baseline, simple gels
     - Telechelic ionomers
     - Associating polymers
     - Wormlike micelles
     - Complex networks
   * - Force-dependence
     - ✓ (Bell/power)
     - ~
     - ~
     - ~
     - ✓ (per species)
   * - Finite extensibility
     - ✓ (FENE-P)
     - ~
     - ~
     - ~
     - ~
   * - Computational cost
     - 1× (fastest)
     - 1.5×
     - 2-3× (modes)
     - 1×
     - 1.5-2.5×

**Decision Tree:**

1. **Is there a single dominant bond type?**

   - Yes → TNTSingleMode (start here for most gels)
   - No, two distinct types → TNTLoopBridge or TNTMultiSpecies

2. **Are bonds sensitive to force/stress?**

   - Yes, exponential → TNTSingleMode(breakage="bell")
   - Yes, power-law → TNTSingleMode(breakage="power_law")
   - No → TNTSingleMode(breakage="constant")

3. **Is chain extensibility important?**

   - Yes (large strains) → TNTSingleMode(stress_type="fene")
   - No (linear/moderate) → stress_type="hookean"

4. **Is the material a living polymer system?**

   - Yes, wormlike micelles → TNTCates
   - Yes, but multi-sticker → TNTStickyRouse
   - No → TNTSingleMode or TNTLoopBridge

5. **Do you observe broad relaxation spectrum?**

   - Yes, continuous → TNTStickyRouse
   - Yes, discrete peaks → TNTMultiSpecies
   - No, single mode → TNTSingleMode


.. _tnt-failure-modes:

Failure Modes
-------------

Each TNT variant has a characteristic **failure mode** — the dominant nonlinear phenomenon
that limits the range of validity or produces extreme behavior:

.. list-table::
   :widths: 18 22 15 20 25
   :header-rows: 1

   * - Variant
     - Primary Phenomenon
     - Key Parameter
     - Failure Mode
     - Physical Mechanism
   * - :doc:`Bell <tnt_bell>`
     - Shear thinning / banding
     - :math:`\nu`
     - Runaway breakage
     - Exponential bond weakening under stretch
   * - :doc:`FENE-P <tnt_fene_p>`
     - Strain stiffening
     - :math:`L_{\max}`
     - Chain snap
     - Stress divergence as chains approach maximum extension
   * - :doc:`Loop-Bridge <tnt_loop_bridge>`
     - Concentration-dependent viscosity
     - :math:`k_{LB}/k_{BL}`
     - Loop saturation
     - All chains convert to loops under extreme flow
   * - :doc:`Cates <tnt_cates>`
     - Single-mode Maxwellian
     - :math:`\tau_{\text{break}}`
     - Shear banding
     - Non-monotonic flow curve from scission kinetics
   * - :doc:`Sticky Rouse <tnt_sticky_rouse>`
     - Self-similar relaxation
     - :math:`N_{\text{stickers}}`
     - Terminal flow
     - All stickers eventually release at long times
   * - :doc:`Multi-Species <tnt_multi_species>`
     - Residual elasticity
     - :math:`G_{\text{chem}}/G_{\text{phys}}`
     - Bond hierarchy
     - Sequential failure from weakest to strongest bonds
   * - :doc:`Non-Affine <tnt_non_affine>`
     - :math:`N_2 \neq 0`
     - :math:`\xi`
     - Wall slip
     - Extreme non-affinity decouples chains from flow
   * - :doc:`Stretch-Creation <tnt_stretch_creation>`
     - Shear thickening
     - :math:`\alpha`
     - Gelation
     - Runaway network formation under sustained deformation


.. _tnt-feature-comparison:

Feature Comparison Matrix
-------------------------

Predicted rheological features across all TNT variants (base Tanaka-Edwards plus 8
extensions). This matrix summarizes which nonlinear phenomena each variant can capture:

.. list-table::
   :widths: 16 10 10 10 10 10 10 10 10 10
   :header-rows: 1

   * - Feature
     - Base TE
     - Bell
     - FENE
     - NonAffine
     - StretchCreate
     - LoopBridge
     - StickyRouse
     - Cates
     - MultiSpecies
   * - Shear thinning
     - \-
     - Yes
     - Yes
     - Yes
     - \-
     - Yes
     - Yes
     - Yes
     - Yes
   * - Stress overshoot
     - \-
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - :math:`N_2 \neq 0`
     - \-
     - \-
     - \-
     - Yes
     - \-
     - \-
     - \-
     - \-
     - \-
   * - Strain hardening
     - \-
     - \-
     - Yes
     - \-
     - Yes
     - \-
     - \-
     - \-
     - \-
   * - Higher harmonics
     - \-
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Shear thickening
     - \-
     - \-
     - \-
     - \-
     - Yes
     - \-
     - \-
     - \-
     - \-
   * - Non-monotonic flow
     - \-
     - (high :math:`\nu`)
     - \-
     - (high :math:`\xi`)
     - \-
     - \-
     - \-
     - Yes
     - \-
   * - Multi-mode spectrum
     - \-
     - \-
     - \-
     - \-
     - \-
     - 2 modes
     - N modes
     - \-
     - N modes
   * - Residual stress
     - \-
     - \-
     - \-
     - \-
     - \-
     - \-
     - \-
     - \-
     - Yes


.. _tnt-decision-framework:

Decision Framework
------------------

Three complementary decision trees help identify the best TNT variant. Use whichever
matches your starting point:

1. **Property-based** (above, `Decision Tree`_): Start from known material class
   (e.g., "telechelic polymer" → LoopBridge). Best when the material type is known.

2. **Observation-based** (:doc:`tnt_knowledge_extraction`, "Master Decision Tree"):
   Start from raw data features (e.g., "Cole-Cole plot is semicircular" → Cates).
   Best when you have data but are unsure of the material class.

3. **Residual-based** (:doc:`tnt_knowledge_extraction`, "Iterative Refinement"):
   Start from a base fit and systematically add physics to reduce residuals
   (e.g., "startup overshoot too sharp" → add Bell breakage). Best when
   iterating on model fits.


Key Parameters
--------------

.. list-table::
   :widths: 20 10 15 55
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Physical Meaning
   * - Network modulus
     - G
     - 1-10⁶ Pa
     - Elastic modulus at short times (G ~ n·k_B·T, n = crosslink density)
   * - Bond lifetime
     - τ_b
     - 10⁻⁶-10⁴ s
     - Mean survival time before bond detachment (sets relaxation time)
   * - Solvent viscosity
     - η_s
     - 0-10⁴ Pa·s
     - Background viscosity (can be zero for ideal network)
   * - Bell force sensitivity
     - ν
     - 0.01-20
     - Dimensionless activation barrier reduction (ΔE = ν·k_B·T per unit force)
   * - FENE extensibility
     - L_max
     - 2-100
     - Maximum chain stretch ratio (L_max² ~ N_Kuhn, chain stiffness)
   * - Slip parameter
     - ξ
     - 0-1
     - Gordon-Schowalter: ξ=0 (affine), ξ=1 (full slip), affects N₁
   * - Bridge fraction (eq)
     - f_B_eq
     - 0-1
     - Equilibrium fraction of bridge junctions (LoopBridge model)
   * - Exchange rate
     - k_ex
     - 10⁻⁴-10² s⁻¹
     - Loop↔bridge interconversion rate (LoopBridge)
   * - Sticker lifetime
     - τ_s
     - 10⁻⁶-10⁴ s
     - Mean sticker attachment time (StickyRouse, limits Rouse modes)
   * - Chain length
     - N
     - 10-1000
     - Number of Kuhn segments (StickyRouse, sets τ_Rouse ~ N²)
   * - Reptation time
     - τ_rep
     - 10⁻³-10³ s
     - Tube escape time for entangled chains (Cates model)
   * - Breakage time
     - τ_break
     - 10⁻³-10³ s
     - Mean time for chain scission (Cates model, τ_d ~ √(τ_rep·τ_break))


Quick Start
-----------

**Basic Tanaka-Edwards model (constant breakage):**

.. code-block:: python

   from rheojax.models import TNTSingleMode

   # Create model with default constant breakage
   model = TNTSingleMode()

   # Fit to oscillatory data (SAOS)
   model.fit(omega, G_star, test_mode='oscillation')

   # Check parameters
   G = model.parameters.get_value('G')
   tau_b = model.parameters.get_value('tau_b')
   print(f"Network modulus: {G:.1f} Pa, Bond lifetime: {tau_b:.3f} s")

   # Predict flow curve
   gamma_dot = jnp.logspace(-2, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

**Force-dependent Bell model for bio-networks:**

.. code-block:: python

   from rheojax.models import TNTSingleMode

   # Bell model with force-activated unbinding
   model = TNTSingleMode(breakage="bell")

   # Set initial guesses for sensitive bonds
   model.parameters.set_value('nu', 5.0)  # Moderate force sensitivity

   # Fit to startup shear data (shows force-induced softening)
   model.fit(t, sigma_startup, test_mode='startup', gamma_dot=1.0)

**FENE-P model for finite extensibility:**

.. code-block:: python

   from rheojax.models import TNTSingleMode

   # FENE-P with finite chain length
   model = TNTSingleMode(stress_type="fene")

   # Set extensibility limit
   model.parameters.set_value('L_max', 10.0)  # 10x equilibrium length

   # Fit to large amplitude data (will show strain softening)
   model.fit(gamma, sigma, test_mode='startup', gamma_dot=10.0)

**Wormlike micelles (Cates model):**

.. code-block:: python

   from rheojax.models import TNTCates

   # Living polymer system
   model = TNTCates()

   # Fit to oscillatory data
   model.fit(omega, G_star, test_mode='oscillation')

   # Extract timescales
   tau_rep = model.parameters.get_value('tau_rep')
   tau_break = model.parameters.get_value('tau_break')
   tau_d = jnp.sqrt(tau_rep * tau_break)
   print(f"Effective relaxation: {tau_d:.3e} s")

**Multi-sticker polymer (StickyRouse):**

.. code-block:: python

   from rheojax.models import TNTStickyRouse

   # Create model with 5 Rouse modes
   model = TNTStickyRouse(n_modes=5)

   # Fit to frequency sweep (broad spectrum)
   model.fit(omega, G_star, test_mode='oscillation')

   # Predict storage/loss moduli
   G_prime, G_double_prime = model.predict(omega, test_mode='oscillation',
                                           return_components=True)


Bayesian Inference
------------------

All TNT models support full Bayesian inference via NumPyro with automatic warm-starting
from NLSQ point estimates. The recommended workflow uses 4 chains for robust diagnostics:

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import jax.numpy as jnp

   # Step 1: NLSQ point estimate (fast, ~seconds)
   model = TNTSingleMode(breakage="bell", stress_type="fene")
   model.fit(omega, G_star, test_mode='oscillation')

   # Step 2: Bayesian inference with warm-start (num_chains=4 default)
   result = model.fit_bayesian(
       omega, G_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,  # Parallel chains for diagnostics
       seed=42        # Reproducibility
   )

   # Step 3: Diagnostics (automatic R-hat, ESS checks)
   intervals = model.get_credible_intervals(result.posterior_samples,
                                            credibility=0.95)

   for param_name, (lower, upper) in intervals.items():
       point_est = model.parameters.get_value(param_name)
       print(f"{param_name}: {point_est:.3e} [{lower:.3e}, {upper:.3e}]")

   # Step 4: Posterior predictive checks
   G_pred = model.predict(omega, test_mode='oscillation')
   # Compare G_pred to G_star to validate model

**ArviZ diagnostics for complex models:**

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   # Full pipeline with automated diagnostics
   pipeline = BayesianPipeline()
   (pipeline
       .load('gel_data.csv', x_col='omega', y_col='G_star')
       .fit_nlsq('tnt_single_mode', breakage='bell')
       .fit_bayesian(num_warmup=1000, num_samples=2000, num_chains=4)
       .plot_trace()       # MCMC convergence
       .plot_pair(divergences=True)  # Parameter correlations
       .plot_forest(hdi_prob=0.95)   # Credible intervals
       .save('results.hdf5'))

   # Check specific diagnostics
   print(f"R-hat: {pipeline.get_diagnostic('r_hat')}")
   print(f"ESS: {pipeline.get_diagnostic('ess')}")


Supported Protocols
-------------------

All TNT models support the full suite of rheological test protocols:

.. list-table::
   :widths: 25 20 55
   :header-rows: 1

   * - Protocol
     - test_mode
     - Notes
   * - Flow curve
     - 'flow_curve'
     - Steady shear stress σ(γ̇), shear thinning from network disruption
   * - SAOS (oscillatory)
     - 'oscillation'
     - G'(ω), G''(ω), single-mode shows G' ~ G'' ~ ω² at low ω
   * - Startup shear
     - 'startup'
     - Transient σ(t, γ̇), stress overshoot from chain orientation
   * - Stress relaxation
     - 'relaxation'
     - G(t) after step strain, exponential decay with τ_b
   * - Creep
     - 'creep'
     - γ(t, σ₀), delayed compliance from bond reformation
   * - LAOS
     - 'laos'
     - Large amplitude σ(t), Fourier/Chebyshev harmonics

**Protocol-specific features:**

- **Flow curve**: Shear thinning η(γ̇) from reduced effective τ_b at high rates
- **SAOS**: G'(ω) crossover at ω ≈ 1/τ_b, tan(δ) = G''/G' diagnostic
- **Startup**: Overshoot at γ ≈ 1-2 (network orientation saturation)
- **Relaxation**: Single exponential G(t) ~ exp(-t/τ_b) for constant breakage
- **Creep**: Power-law at short times, viscous flow at long times
- **LAOS**: Strain softening (I₃/I₁ ratio) from FENE-P or Bell kinetics

**Example multi-protocol characterization:**

.. code-block:: python

   from rheojax.models import TNTSingleMode
   import jax.numpy as jnp

   model = TNTSingleMode(breakage="bell", stress_type="fene")

   # 1. Fit to SAOS for linear parameters
   model.fit(omega, G_star, test_mode='oscillation')

   # 2. Predict startup for validation
   t = jnp.linspace(0, 10, 200)
   sigma_startup = model.predict(t, test_mode='startup', gamma_dot=1.0)

   # 3. Flow curve for nonlinear regime
   gamma_dot = jnp.logspace(-3, 2, 50)
   sigma_flow = model.predict(gamma_dot, test_mode='flow_curve')

   # 4. Relaxation modulus
   G_t = model.predict(t, test_mode='relaxation', gamma_0=0.1)


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   tnt_protocols
   tnt_knowledge_extraction
   tnt_tanaka_edwards
   tnt_bell
   tnt_fene_p
   tnt_non_affine
   tnt_stretch_creation
   tnt_loop_bridge
   tnt_sticky_rouse
   tnt_cates
   tnt_multi_species


See Also
--------

**Related Model Families:**

- :doc:`/models/giesekus/index` — Polymer kinetic theory with continuous relaxation
- :doc:`/models/fluidity/index` — Yield stress fluids with fluidity evolution
- :doc:`/models/dmt/index` — Thixotropic models with scalar structure parameter
- :doc:`/models/classical/maxwell` — Single Maxwell mode (TNT limit with Hookean chains)

**Transforms and Utilities:**

- :doc:`/transforms/mastercurve` — Time-temperature superposition for thermorheology
- :doc:`/transforms/derivatives` — Numerical differentiation for G(t) → G', G''
- :doc:`/utils/prony` — Prony series decomposition for multi-mode fitting

**User Guides:**

- :doc:`/user_guide/transient_networks` — Introduction to TNT physics
- :doc:`/user_guide/associating_polymers` — Telechelic and multi-sticker systems
- :doc:`/user_guide/living_polymers` — Wormlike micelles and scission/recombination


References
----------

1. Green, M. S. & Tobolsky, A. V. (1946). "A New Approach to the Theory of Relaxing
   Polymeric Media." *J. Chem. Phys.*, 14, 80–92.
   https://doi.org/10.1063/1.1724109

2. Tanaka, F. & Edwards, S. F. (1992). "Viscoelastic properties of physically crosslinked
   networks: Transient network theory." *J. Chem. Soc., Faraday Trans.*, 88, 2979–2990.
   https://doi.org/10.1039/FT9928802979

3. Bell, G. I. (1978). "Models for the specific adhesion of cells to cells."
   *Science*, 200, 618–627.
   https://doi.org/10.1126/science.347575

4. Cates, M. E. (1987). "Reptation of living polymers: dynamics of entangled polymers
   in the presence of reversible chain-scission reactions."
   *Macromolecules*, 20, 2289–2296.
   https://doi.org/10.1021/ma00175a038

5. Leibler, L., Rubinstein, M., & Colby, R. H. (1991). "Dynamics of reversible networks."
   *Macromolecules*, 24, 4701–4707.
   https://doi.org/10.1021/ma00016a034

6. Vaccaro, A. & Marrucci, G. (2000). "A model for the nonlinear rheology of associating
   polymers." *J. Non-Newtonian Fluid Mech.*, 92, 261–273.
   https://doi.org/10.1016/S0377-0257(00)00094-1

7. Tripathi, A., Tam, K. C., & McKinley, G. H. (2006). "Rheology and dynamics of
   associative polymers in shear and extension: Theory and experiments."
   *Macromolecules*, 39, 1981–1999.
   https://doi.org/10.1021/ma051614x

8. Rubinstein, M. & Semenov, A. N. (2001). "Dynamics of entangled solutions of
   associating polymers." *Macromolecules*, 34, 1058–1068.
   https://doi.org/10.1021/ma0013049

9. Semenov, A. N. & Rubinstein, M. (1998). "Thermoreversible gelation in solutions
   of associative polymers. 1. Statics." *Macromolecules*, 31, 1373–1385.
   https://doi.org/10.1021/ma970616h

10. Wang, S.-Q. (1992). "Transient network theory for shear-thickening fluids and
    physically crosslinked networks." *Macromolecules*, 25, 7003–7010.
    https://doi.org/10.1021/ma00051a043
