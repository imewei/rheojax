.. _model-tnt-multi-species:

===========================================================
TNT Multi-Species (Multiple Bond Types) — Handbook
===========================================================

.. contents:: Table of Contents
   :local:
   :depth: 3

Quick Reference
===========================================================

Use When
-----------------------------------------------------------

The TNT Multi-Species model is appropriate for network materials containing **multiple distinct bond types** with different relaxation timescales:

- **Dual-crosslinked hydrogels** (physical + chemical crosslinks)
- **Interpenetrating polymer networks** (IPNs)
- **Heterogeneous biological networks** (fibrin + collagen, actin + intermediate filaments)
- **Multi-strength supramolecular assemblies** (hydrogen bonds + hydrophobic interactions)
- **Hierarchical networks** with discrete energy barriers
- **Reversible networks** with multiple association strengths

**Key diagnostic:** Multiple distinct relaxation peaks in :math:`G''(\omega)` or multi-modal relaxation spectrum.

Parameters
-----------------------------------------------------------

The model requires :math:`2N + 1` parameters for :math:`N` bond species:

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Symbol
     - Typical Value
     - Units
     - Description
   * - :math:`G_i`
     - :math:`10^2` to :math:`10^5`
     - Pa
     - Modulus contribution from species :math:`i`
   * - :math:`\tau_{b,i}`
     - :math:`10^{-3}` to :math:`10^3`
     - s
     - Bond lifetime of species :math:`i`
   * - :math:`\eta_s`
     - 0.0 to :math:`10^{-2}`
     - Pa·s
     - Solvent viscosity (background)
   * - :math:`N`
     - 2 to 5
     - dimensionless
     - Number of bond species (constructor argument)

**Total parameters:** :math:`2N + 1` (e.g., 5 for :math:`N=2`, 11 for :math:`N=5`)

Key Equation
-----------------------------------------------------------

Each bond species evolves independently:

.. math::

   \frac{d\mathbf{S}_i}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S}_i + \mathbf{S}_i \cdot \boldsymbol{\kappa}^T - \frac{1}{\tau_{b,i}} (\mathbf{S}_i - \mathbf{I})

Total stress is the superposition of all species contributions:

.. math::

   \boldsymbol{\sigma} = \sum_{i=1}^{N} G_i (\mathbf{S}_i - \mathbf{I}) + 2\eta_s \mathbf{D}

where :math:`\mathbf{S}_i` is the conformation tensor for species :math:`i`, :math:`\boldsymbol{\kappa}` is the velocity gradient tensor, and :math:`\mathbf{D}` is the rate-of-strain tensor.

Supported Test Modes
-----------------------------------------------------------

All six rheological protocols are supported:

1. **FLOW_CURVE**: Steady shear viscosity :math:`\eta(\dot{\gamma})`
2. **OSCILLATION**: SAOS moduli :math:`G'(\omega)`, :math:`G''(\omega)`
3. **STARTUP**: Stress growth :math:`\sigma^+(t, \dot{\gamma})`
4. **CREEP**: Strain response :math:`\gamma(t, \sigma_0)`
5. **RELAXATION**: Stress relaxation :math:`G(t)` (analytical)
6. **LAOS**: Large amplitude oscillatory shear

Material Examples
-----------------------------------------------------------

**Dual-Crosslinked Hydrogels:**
   - Fast species: hydrogen bonds (:math:`\tau_{b,1} \sim 0.01` s)
   - Slow species: covalent crosslinks (:math:`\tau_{b,2} \sim 100` s)
   - Example: Alginate-polyacrylamide (Sun et al. 2012)

**Biological Networks:**
   - Species 1: Fibrin (:math:`\tau_{b,1} \sim 0.1` s)
   - Species 2: Collagen (:math:`\tau_{b,2} \sim 10` s)
   - Interpenetrating structure in blood clots

**Supramolecular Networks:**
   - Fast species: H-bonds (:math:`\tau_{b,1} \sim 0.001` s)
   - Medium: Hydrophobic (:math:`\tau_{b,2} \sim 1` s)
   - Slow: Ionic (:math:`\tau_{b,3} \sim 100` s)

Key Characteristics
-----------------------------------------------------------

- **Multi-modal relaxation**: Discrete peaks at :math:`\omega_i = 1/\tau_{b,i}`
- **Broad relaxation spectrum**: From discrete bond types, not distribution
- **Superposition principle**: Total response = sum of Maxwell modes
- **Physical interpretation**: Each mode corresponds to a specific bond type
- **Network equivalence**: Mathematically equivalent to Generalized Maxwell, but with network interpretation

Notation Guide
===========================================================

Primary Variables
-----------------------------------------------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Definition
   * - :math:`\mathbf{S}_i`
     - Conformation tensor for species :math:`i` (dimensionless, 3x3 symmetric)
   * - :math:`S_{xx,i}, S_{yy,i}, S_{zz,i}`
     - Normal components of conformation tensor for species :math:`i`
   * - :math:`S_{xy,i}`
     - Shear component of conformation tensor for species :math:`i`
   * - :math:`\boldsymbol{\sigma}`
     - Total Cauchy stress tensor (Pa, 3x3 symmetric)
   * - :math:`\boldsymbol{\kappa}`
     - Velocity gradient tensor (1/s, 3x3)

Parameters and Constants
-----------------------------------------------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Definition
   * - :math:`N`
     - Number of distinct bond species (integer :math:`\geq 2`)
   * - :math:`G_i`
     - Elastic modulus contribution from species :math:`i` (Pa)
   * - :math:`\tau_{b,i}`
     - Bond lifetime for species :math:`i` (s)
   * - :math:`\eta_s`
     - Solvent viscosity (Pa·s)
   * - :math:`\mathbf{I}`
     - Identity tensor (3x3)
   * - :math:`\mathbf{D}`
     - Rate-of-strain tensor: :math:`\mathbf{D} = (\boldsymbol{\kappa} + \boldsymbol{\kappa}^T)/2`

Derived Quantities
-----------------------------------------------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Definition
   * - :math:`G_{\text{total}}`
     - Total plateau modulus: :math:`G_{\text{total}} = \sum_{i=1}^{N} G_i`
   * - :math:`\lambda_i`
     - Relaxation time: :math:`\lambda_i = \tau_{b,i}` (equivalent for single-mode TNT)
   * - :math:`\eta_i`
     - Viscosity contribution: :math:`\eta_i = G_i \tau_{b,i}`
   * - :math:`\eta_0`
     - Zero-shear viscosity: :math:`\eta_0 = \sum_{i=1}^{N} G_i \tau_{b,i} + \eta_s`
   * - :math:`\omega_i`
     - Characteristic frequency: :math:`\omega_i = 1/\tau_{b,i}`

Protocol-Specific Variables
-----------------------------------------------------------

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Symbol
     - Definition
   * - :math:`\dot{\gamma}`
     - Shear rate (1/s, for flow curve and startup)
   * - :math:`\omega`
     - Angular frequency (rad/s, for oscillation)
   * - :math:`\gamma_0`
     - Strain amplitude (dimensionless, for LAOS)
   * - :math:`\sigma_0`
     - Applied stress (Pa, for creep)
   * - :math:`G'(\omega)`
     - Storage modulus (Pa, oscillation)
   * - :math:`G''(\omega)`
     - Loss modulus (Pa, oscillation)
   * - :math:`G(t)`
     - Relaxation modulus (Pa, relaxation)

Overview
===========================================================

Conceptual Foundation
-----------------------------------------------------------

The TNT Multi-Species model extends the single-mode Tanaka-Edwards transient network theory to materials with **multiple distinct bond types**. Unlike the Generalized Maxwell model (which uses empirical mode decomposition), this model assigns a **physical interpretation** to each mode: a chemically distinct bond population with its own modulus :math:`G_i` and lifetime :math:`\tau_{b,i}`.

**Physical picture:**

- Each bond type (species) forms an independent transient network
- Species do not interconvert or couple during deformation
- Each species has its own energy barrier for bond breakage, yielding distinct :math:`\tau_{b,i}`
- Total stress is the **additive superposition** of species contributions
- Mathematically equivalent to multi-mode Maxwell, but with network interpretation

Relationship to Other Models
-----------------------------------------------------------

**Generalized Maxwell:**
   - **Identical mathematics** for linear response (SAOS, relaxation)
   - **Different interpretation**: TNT assigns physical meaning to each mode
   - **Nonlinear regime**: TNT includes conformation evolution; GMM is linear only

**Single-Mode Tanaka-Edwards:**
   - **Special case**: :math:`N=1` reduces to base TNT
   - **Extension**: Multi-species adds independent bond populations

**Sticky Rouse:**
   - **Different physics**: Sticky Rouse has distributed lifetimes (Rouse modes)
   - **Multi-species**: Discrete lifetimes from distinct bond types

**Loop-Bridge Model:**
   - **Coupling**: Loop-Bridge has inter-species conversion (loop to bridge)
   - **Multi-species**: No coupling, independent evolution

Physical Foundations
===========================================================

Multiple Bond Types
-----------------------------------------------------------

Real network materials often contain **chemically distinct** bond types:

**Hydrogen Bonds:**
   - Typical :math:`E_a \sim 10` to 30 kJ/mol
   - Lifetime :math:`\tau_b \sim 10^{-3}` to :math:`10^{-1}` s at room temperature
   - Common in protein gels, synthetic polymers with H-bonding groups

**Hydrophobic Interactions:**
   - Typical :math:`E_a \sim 20` to 50 kJ/mol
   - Lifetime :math:`\tau_b \sim 0.1` to 10 s
   - Entropy-driven, temperature-sensitive

**Ionic Bonds:**
   - Typical :math:`E_a \sim 30` to 80 kJ/mol
   - Lifetime :math:`\tau_b \sim 1` to :math:`10^3` s
   - Strong dependence on ionic strength

**Covalent Crosslinks:**
   - Typical :math:`E_a > 150` kJ/mol
   - Lifetime :math:`\tau_b \to \infty` (permanent on experimental timescales)
   - Provides long-time elasticity

Each bond type has a different **energy landscape**, yielding different Arrhenius factor:

.. math::

   \tau_{b,i} = \tau_0 \exp\left(\frac{E_{a,i}}{k_B T}\right)

Independent Evolution
-----------------------------------------------------------

The model assumes **no coupling** between species:

- **No conversion**: Bonds of type 1 do not convert to type 2
- **No cooperative breaking**: Breaking of one species does not affect others
- **Independent kinetics**: Each species obeys its own relaxation equation
- **Additive stress**: Total stress = sum of species contributions (linear superposition)

This is valid when:

1. Bond types are **chemically distinct** (different functional groups)
2. **Timescale separation** is sufficient (:math:`\tau_{b,i}/\tau_{b,j} > 10`)
3. **No synergistic effects** (e.g., one bond type does not stabilize another)
4. **Mean-field applies** per species (high bond density within each type)

Stress Additivity
-----------------------------------------------------------

Total stress is the **sum** of contributions from each species:

.. math::

   \boldsymbol{\sigma}_{\text{total}} = \sum_{i=1}^{N} \boldsymbol{\sigma}_i + \boldsymbol{\sigma}_{\text{solvent}}

where:

.. math::

   \boldsymbol{\sigma}_i = G_i (\mathbf{S}_i - \mathbf{I})

   \boldsymbol{\sigma}_{\text{solvent}} = 2\eta_s \mathbf{D}

**Physical interpretation:**

- :math:`G_i` is the modulus from species :math:`i` alone (proportional to number density of type-:math:`i` bonds)
- :math:`\mathbf{S}_i - \mathbf{I}` is the extra stress from deformation of species :math:`i` network
- No cross-terms between species (independence assumption)

Governing Equations
===========================================================

Per-Species Evolution
-----------------------------------------------------------

Each species evolves according to the Tanaka-Edwards equation **independently**:

.. math::

   \frac{d\mathbf{S}_i}{dt} = \boldsymbol{\kappa} \cdot \mathbf{S}_i + \mathbf{S}_i \cdot \boldsymbol{\kappa}^T - \frac{1}{\tau_{b,i}} (\mathbf{S}_i - \mathbf{I})

where:

- :math:`\mathbf{S}_i`: Conformation tensor for species :math:`i` (3x3 symmetric)
- :math:`\boldsymbol{\kappa} = \nabla \mathbf{v}`: Velocity gradient tensor
- :math:`\tau_{b,i}`: Bond lifetime for species :math:`i`
- :math:`\mathbf{I}`: Identity tensor (equilibrium conformation)

**Component form** (simple shear :math:`\dot{\gamma}`):

.. math::

   \frac{dS_{xx,i}}{dt} &= 2\dot{\gamma} S_{xy,i} - \frac{1}{\tau_{b,i}}(S_{xx,i} - 1)

   \frac{dS_{yy,i}}{dt} &= -\frac{1}{\tau_{b,i}}(S_{yy,i} - 1)

   \frac{dS_{zz,i}}{dt} &= -\frac{1}{\tau_{b,i}}(S_{zz,i} - 1)

   \frac{dS_{xy,i}}{dt} &= \dot{\gamma} S_{yy,i} - \frac{1}{\tau_{b,i}} S_{xy,i}

Total Stress
-----------------------------------------------------------

The total stress tensor is:

.. math::

   \boldsymbol{\sigma} = \sum_{i=1}^{N} G_i (\mathbf{S}_i - \mathbf{I}) + 2\eta_s \mathbf{D}

For simple shear (measuring :math:`\sigma_{xy}`):

.. math::

   \sigma(t) = \sum_{i=1}^{N} G_i S_{xy,i}(t) + \eta_s \dot{\gamma}(t)

State Vector Representation
-----------------------------------------------------------

For numerical integration, the state is represented as a **4N-dimensional vector**:

.. math::

   \mathbf{y}(t) = [S_{xx,1}, S_{yy,1}, S_{zz,1}, S_{xy,1}, \ldots, S_{xx,N}, S_{yy,N}, S_{zz,N}, S_{xy,N}]^T

This enables **efficient vectorization** via JAX vmap over species index.

Analytical Solutions: SAOS
-----------------------------------------------------------

For small-amplitude oscillatory shear :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, the complex modulus is:

.. math::

   G^*(\omega) = \sum_{i=1}^{N} \frac{G_i (\omega \tau_{b,i})^2 + i G_i \omega \tau_{b,i}}{1 + (\omega \tau_{b,i})^2} + i\omega\eta_s

**Storage modulus:**

.. math::

   G'(\omega) = \sum_{i=1}^{N} \frac{G_i (\omega \tau_{b,i})^2}{1 + (\omega \tau_{b,i})^2}

**Loss modulus:**

.. math::

   G''(\omega) = \sum_{i=1}^{N} \frac{G_i \omega \tau_{b,i}}{1 + (\omega \tau_{b,i})^2} + \omega\eta_s

**Key features:**

- Each species contributes a **Maxwell peak** at :math:`\omega_i = 1/\tau_{b,i}`
- :math:`G'(\omega)` transitions from :math:`G_{\text{total}}` (high :math:`\omega`) to 0 (low :math:`\omega`) through plateaus
- :math:`G''(\omega)` shows :math:`N` distinct peaks if :math:`\tau_{b,i}` are well-separated

Analytical Solutions: Relaxation
-----------------------------------------------------------

For step strain :math:`\gamma(t) = \gamma_0 H(t)` (Heaviside function), the relaxation modulus is:

.. math::

   G(t) = \sum_{i=1}^{N} G_i \exp\left(-\frac{t}{\tau_{b,i}}\right)

**Multi-exponential decay** with :math:`N` discrete relaxation times.

Analytical Solutions: Flow Curve
-----------------------------------------------------------

For steady simple shear at rate :math:`\dot{\gamma}`, the steady-state shear stress is:

.. math::

   \sigma_{\infty}(\dot{\gamma}) = \sum_{i=1}^{N} \frac{G_i \tau_{b,i} \dot{\gamma}}{1 + (\tau_{b,i} \dot{\gamma})^2} + \eta_s \dot{\gamma}

**Viscosity:**

.. math::

   \eta(\dot{\gamma}) = \sum_{i=1}^{N} \frac{G_i \tau_{b,i}}{1 + (\tau_{b,i} \dot{\gamma})^2} + \eta_s

**Limiting behavior:**

- **Zero-shear viscosity**: :math:`\eta_0 = \sum_{i=1}^{N} G_i \tau_{b,i} + \eta_s`
- **High-shear viscosity**: :math:`\eta_{\infty} = \eta_s` (all networks relax)
- **Shear thinning**: Each species contributes thinning at :math:`\dot{\gamma}_i \sim 1/\tau_{b,i}`

Parameter Table
===========================================================

Comprehensive Parameter List
-----------------------------------------------------------

.. list-table::
   :widths: 15 15 15 15 40
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Units
     - Description
   * - Species moduli
     - :math:`G_i`
     - :math:`10^2` to :math:`10^5`
     - Pa
     - Modulus contribution from species :math:`i` (:math:`N` values)
   * - Species lifetimes
     - :math:`\tau_{b,i}`
     - :math:`10^{-3}` to :math:`10^3`
     - s
     - Bond lifetime for species :math:`i` (:math:`N` values)
   * - Solvent viscosity
     - :math:`\eta_s`
     - 0.0 to :math:`10^{-2}`
     - Pa·s
     - Background Newtonian viscosity (1 value)
   * - Number of species
     - :math:`N`
     - 2 to 5
     - dimensionless
     - Number of bond types (constructor argument)

**Total parameters:** :math:`2N + 1`

Parameter Interpretation
===========================================================

Number of Species (N)
-----------------------------------------------------------

**Physical meaning:**

- :math:`N` = number of **chemically distinct** bond types
- Each species represents a different molecular interaction or crosslink

**Common cases:**

**N = 1:**
   - Reduces to single-mode Tanaka-Edwards
   - Single bond type (homogeneous network)

**N = 2:**
   - **Most common** for dual-crosslinked materials
   - Example: Fast physical + slow chemical crosslinks

**N = 3:**
   - Triple network or three interaction types

**N >= 4:**
   - High parameter count (>= 9 parameters)
   - Risk of **overfitting** without sufficient data

Species Moduli (G_i)
-----------------------------------------------------------

**Physical meaning:**

- :math:`G_i` is proportional to the **number density** of type-:math:`i` bonds
- Reflects the **strength** of species :math:`i` contribution to total elasticity

**Relative contributions:**

.. math::

   f_i = \frac{G_i}{\sum_{j=1}^{N} G_j}

where :math:`f_i` is the fraction of total modulus from species :math:`i`.

Species Lifetimes (tau_b_i)
-----------------------------------------------------------

**Physical meaning:**

- :math:`\tau_{b,i}` is the **average time** before a type-:math:`i` bond breaks
- Inversely related to bond breakage rate: :math:`k_{\text{break},i} = 1/\tau_{b,i}`

**Timescale separation:**

- Distinct peaks in :math:`G''(\omega)` require :math:`\tau_{b,i}/\tau_{b,j} > 10`
- If :math:`\tau_{b,i}/\tau_{b,j} < 3`, species :math:`i` and :math:`j` may be indistinguishable

Validity and Assumptions
===========================================================

Core Assumptions
-----------------------------------------------------------

1. **Independent species:**
   - No conversion between bond types
   - No cooperative breaking or stabilization
   - Each species evolves according to its own kinetics

2. **Mean-field per species:**
   - High bond density within each type
   - No spatial correlations between species
   - Uniform distribution of each bond type

3. **Gaussian chains:**
   - All species obey Gaussian network statistics
   - Valid for small to moderate deformations
   - Breaks down for chain stretching (strain > 100 percent)

4. **Constant breakage rate:**
   - :math:`1/\tau_{b,i}` independent of stress (affine network)
   - No stress-assisted bond breaking (unlike Sticky Rouse)

5. **Affine deformation:**
   - Chains deform with the bulk medium
   - No chain slippage or retraction

Regimes
===========================================================

Frequency Regimes (SAOS)
-----------------------------------------------------------

**High frequency** (:math:`\omega \gg 1/\tau_{b,\text{min}}`):
   - All species are elastic (no relaxation)
   - :math:`G'(\omega) \approx \sum_{i=1}^{N} G_i = G_{\text{total}}`
   - :math:`G''(\omega) \approx \omega \eta_s` (if :math:`\eta_s > 0`)

**Intermediate frequency** (:math:`1/\tau_{b,i} < \omega < 1/\tau_{b,i-1}`):
   - Slowest :math:`i` species have relaxed, faster species still elastic
   - :math:`G'(\omega) \approx \sum_{j=i}^{N} G_j` (partial sum)
   - :math:`G''(\omega)` shows local minimum between peaks

**Low frequency** (:math:`\omega \ll 1/\tau_{b,\text{max}}`):
   - All species have relaxed (if all bonds are transient)
   - :math:`G'(\omega) \sim \omega^2`, :math:`G''(\omega) \sim \omega` (terminal regime)
   - :math:`\eta_0 = \sum_{i=1}^{N} G_i \tau_{b,i} + \eta_s`

What You Can Learn
===========================================================

Structural Information
-----------------------------------------------------------

**Number of bond types:**
   - :math:`N` determined from number of distinct peaks in :math:`G''(\omega)`
   - Requires wide frequency range (3+ decades)

**Bond population fractions:**
   - :math:`f_i = G_i / G_{\text{total}}` quantifies relative importance
   - Dominant species (:math:`f_i > 0.5`) controls long-time behavior

**Lifetime hierarchy:**
   - Order of :math:`\tau_{b,i}` reveals fastest to slowest bond types
   - Lifetime ratios :math:`\tau_{b,i}/\tau_{b,j}` indicate timescale separation

Experimental Design
===========================================================

Essential Measurements
-----------------------------------------------------------

**1. Wide-frequency SAOS (critical):**
   - **Range**: At least 3 decades to resolve :math:`N=2` modes, 4+ decades for :math:`N=3`
   - **Example**: 0.01 to 100 rad/s for :math:`\tau_{b,1} = 0.1` s, :math:`\tau_{b,2} = 10` s
   - **Why**: Each mode needs approximately 1 decade around :math:`\omega_i = 1/\tau_{b,i}` to be resolved

**2. Stress relaxation (recommended):**
   - **Protocol**: Step strain :math:`\gamma_0 = 1` percent to 10 percent (linear regime)
   - **Duration**: :math:`t_{\max} > 10 \tau_{b,\text{max}}` to capture slowest decay
   - **Why**: Direct measurement of :math:`G(t) = \sum_i G_i \exp(-t/\tau_{b,i})`

**3. Flow curve (optional but valuable):**
   - **Range**: :math:`\dot{\gamma}` from :math:`0.01/\tau_{b,\text{max}}` to :math:`10/\tau_{b,\text{min}}`
   - **Why**: Validates nonlinear predictions, reveals shear thinning transitions

Computational Implementation
===========================================================

Efficient Multi-Mode Computation
-----------------------------------------------------------

**Key strategy:** Use JAX vmap to vectorize over species dimension.

**State vector structure:**

State is a 4N-dimensional flattened array: [S_xx_0, S_yy_0, S_zz_0, S_xy_0, S_xx_1, ..., S_xy_N-1]

**Vectorized ODE:**

Each species evolves independently via the TNT equation. JAX vmap applies the single-species evolution to all species in parallel.

**Benefits:**

- **Parallel computation** of all species (GPU-friendly)
- **Code reuse** from single-mode TNT
- **Memory efficient** (no loops)

Fitting Guidance
===========================================================

Model Selection (Determining N)
-----------------------------------------------------------

**Strategy:**

1. **Start with :math:`N=2`** (most common, 5 parameters)
2. Fit to wide-frequency SAOS data
3. **Increase :math:`N`** if:
   - Residuals show systematic structure (e.g., extra peaks)
   - BIC or AIC significantly decreases
   - Visual inspection suggests additional modes

**Bayesian Information Criterion (BIC):**

.. math::

   \text{BIC} = n \ln\left(\frac{\text{RSS}}{n}\right) + k \ln(n)

where :math:`n` = number of data points, :math:`k = 2N+1` = number of parameters, RSS = residual sum of squares.

**Decision rule:**

- If delta BIC = BIC(N+1) - BIC(N) < -10: Strong evidence for :math:`N+1`
- If -10 < delta BIC < -2: Weak evidence for :math:`N+1`
- If delta BIC > -2: No evidence, keep :math:`N`

**Practical limit:** :math:`N \leq 5` (11 parameters) for most datasets.

Primary Data Source
-----------------------------------------------------------

**SAOS is best for parameter estimation:**

- Analytical solution (fast, no ODE integration)
- Directly reveals :math:`N` distinct peaks in :math:`G''(\omega)`
- Linear regime (simple interpretation)

**Recommended workflow:**

1. **Fit SAOS first** to get initial :math:`G_i`, :math:`\tau_{b,i}`, :math:`\eta_s`
2. **Validate with relaxation** (if available) to confirm :math:`\tau_{b,i}`
3. **Test predictions** on flow curve, startup (nonlinear validation)

Usage Examples
===========================================================

Basic Fitting (N=2)
-----------------------------------------------------------

.. code-block:: python

   from rheojax.models.tnt import TNTMultiSpecies
   from rheojax.core import RheoData
   import numpy as np

   # Experimental data
   omega = np.logspace(-2, 2, 50)
   G_star = measure_saos(omega)

   # Create model with 2 species (dual-crosslinked)
   model = TNTMultiSpecies(n_species=2)

   # Fit to SAOS data
   rheo_data = RheoData(x=omega, y=G_star, test_mode='oscillation')
   result = model.fit(rheo_data)

   # Inspect fitted parameters
   params = model.get_parameters()
   print(f"G_1 = {params['G_1'].value:.1f} Pa")
   print(f"tau_b_1 = {params['tau_b_1'].value:.3f} s")

Model Selection (N=2 vs N=3)
-----------------------------------------------------------

.. code-block:: python

   # Fit with N=2
   model_2 = TNTMultiSpecies(n_species=2)
   result_2 = model_2.fit(rheo_data)
   rss_2 = result_2.residual_sum_of_squares
   k_2 = 5
   bic_2 = len(omega) * np.log(rss_2 / len(omega)) + k_2 * np.log(len(omega))

   # Fit with N=3
   model_3 = TNTMultiSpecies(n_species=3)
   result_3 = model_3.fit(rheo_data)
   rss_3 = result_3.residual_sum_of_squares
   k_3 = 7
   bic_3 = len(omega) * np.log(rss_3 / len(omega)) + k_3 * np.log(len(omega))

   # Compare
   if bic_3 - bic_2 < -10:
       print("Strong evidence for N=3")
   else:
       print("N=2 is sufficient")

Permanent + Transient Dual Network
===========================================================

The multi-species model naturally accommodates networks with **both permanent (chemical) and transient (physical) crosslinks**. This is the most common application of the multi-species framework, representing materials such as dual-crosslinked hydrogels, hybrid organogels, and biological tissues with both covalent and non-covalent bonds.

Total Stress Decomposition
-----------------------------------------------------------

The total stress separates into a non-relaxing permanent contribution and transient species contributions:

.. math::

   \boldsymbol{\sigma}(t) = G_{\text{chem}} \left(\mathbf{B} - \mathbf{I}\right) + \sum_{i=1}^{N_{\text{phys}}} G_i \left(\mathbf{S}_i(t) - \mathbf{I}\right) + 2\eta_s \mathbf{D}

where :math:`G_{\text{chem}}` is the modulus of the permanent (chemical) network and :math:`\mathbf{B}` is the left Cauchy-Green deformation tensor. The permanent species has :math:`\tau_{b} \to \infty` (no breakage), contributing a non-relaxing elastic plateau.

Stress Relaxation With Permanent Bonds
-----------------------------------------------------------

Under step strain, the relaxation modulus approaches a **non-zero equilibrium**:

.. math::

   G(t) = G_{\text{chem}} + \sum_{i=1}^{N_{\text{phys}}} G_i \exp\left(-\frac{t}{\tau_{b,i}}\right) \quad \xrightarrow{t \to \infty} \quad G_{\text{chem}}

The stress never fully relaxes to zero. This is the fundamental difference from purely physical (transient) networks: the permanent bonds provide a long-time elastic restoring force, so that:

.. math::

   \sigma(t) = G_{\text{chem}} \cdot \gamma_0 + \sum_{i=1}^{N_{\text{phys}}} G_i \exp\left(-\frac{t}{\tau_{b,i}}\right) \cdot \gamma_0 \quad \xrightarrow{t \to \infty} \quad G_{\text{chem}} \cdot \gamma_0

Mathematically, a "permanent" species simply means :math:`\tau_{b,i} \gg t_{\text{exp}}`, where :math:`t_{\text{exp}}` is the experimental timescale. In practice, any species with :math:`\tau_{b,i}` exceeding the longest measurement time by a factor of :math:`\sim 100` is effectively permanent.

Creep Saturation
===========================================================

Unlike purely transient networks which flow indefinitely under constant stress, dual networks with permanent bonds exhibit **creep saturation**: the strain reaches a finite asymptote rather than growing without bound.

Strain Saturation
-----------------------------------------------------------

Under constant applied stress :math:`\sigma_0`, the strain saturates at:

.. math::

   \gamma_{\infty} = \frac{\sigma_0}{G_{\text{chem}}}

No steady-state flow occurs because the permanent network acts as a spring in parallel with the transient species, providing an equilibrium restoring force that eventually balances the applied stress.

Creep Compliance
-----------------------------------------------------------

The creep compliance for a dual network takes the form:

.. math::

   J(t) = \frac{1}{G_{\text{chem}}} - \sum_{i=1}^{N_{\text{phys}}} \frac{G_i}{G_{\text{chem}}\left(G_{\text{chem}} + \sum_j G_j\right)} \exp\left(-\frac{t}{\tau_i^*}\right)

where :math:`\tau_i^*` are the retardation times (related to but distinct from the relaxation times :math:`\tau_{b,i}`).

**Physical insight:** The strain asymptote directly gives the permanent network modulus:

.. math::

   G_{\text{chem}} = \frac{\sigma_0}{\gamma_{\infty}}

This provides a model-independent route to measuring the permanent network contribution from a single creep experiment.

"Double-Yielding" in LAOS
===========================================================

When bond species have well-separated lifetimes, large amplitude oscillatory shear (LAOS) reveals **sequential yielding** — a phenomenon where different bond populations break at different strain amplitudes.

Sequential Yielding Mechanism
-----------------------------------------------------------

The progression with increasing strain amplitude :math:`\gamma_0` is:

**Small** :math:`\gamma_0` **(linear regime):**
   All species remain elastic. The Lissajous figure is a linear ellipse with stiffness reflecting :math:`G_{\text{total}} = \sum_i G_i`.

**Intermediate** :math:`\gamma_0` **(first yielding):**
   Weak bonds (short :math:`\tau_{b}`) break and reform within a cycle. The first yielding event occurs when :math:`\gamma_0 \dot{\gamma}_{\max} \sim 1/\tau_{b,\text{weak}}`. The Lissajous figure begins to distort, showing the onset of nonlinearity.

**Large** :math:`\gamma_0` **(second yielding):**
   Strong bonds (long :math:`\tau_{b}`) also break, leading to a second yielding event. The material becomes fully fluidized, and the Lissajous figure shows a characteristic butterfly or rectangular shape.

LAOS Signatures
-----------------------------------------------------------

The Lissajous figure shows **two distinct stiffness changes** as the strain amplitude increases, corresponding to the progressive loss of each bond population.

This "double-yielding" signature is also visible in the intensity ratio :math:`I_{3/1}` (third harmonic normalized by fundamental) plotted against :math:`\gamma_0`: for :math:`N=2` species, :math:`I_{3/1}(\gamma_0)` shows **two peaks**, one per bond type. Each peak corresponds to the onset of nonlinear response from a specific bond population. More generally, :math:`N` well-separated species produce :math:`N` peaks in :math:`I_{3/1}(\gamma_0)`.

Progressive Loss Under Flow
===========================================================

The flow curve of a multi-species network shows **staged shear thinning**: each bond species yields at a different critical shear rate, producing a characteristic stepped viscosity profile.

Staged Shear Thinning
-----------------------------------------------------------

The critical shear rate for each species is:

.. math::

   \dot{\gamma}_{c,i} \approx \frac{1}{\tau_{b,i}}

The flow curve exhibits distinct regimes as the shear rate increases:

**Low rates** (:math:`\dot{\gamma} \ll 1/\tau_{b,\max}`):
   All bond species are intact. The full network modulus :math:`G_{\text{total}} = \sum_i G_i` is active, and the material exhibits Newtonian behavior with zero-shear viscosity :math:`\eta_0 = \sum_i G_i \tau_{b,i} + \eta_s`.

**Intermediate rates** (:math:`1/\tau_{b,i+1} < \dot{\gamma} < 1/\tau_{b,i}`):
   Species :math:`i` (and all weaker species) have yielded. The effective modulus is reduced to :math:`G_{\text{eff}} = \sum_{j>i} G_j`, and the viscosity contribution from species :math:`i` is lost.

**High rates** (:math:`\dot{\gamma} \gg 1/\tau_{b,\min}`):
   All transient species have yielded. Only the solvent viscosity remains: :math:`\sigma \to \eta_s \cdot \dot{\gamma}`.

Staircase Flow Curve
-----------------------------------------------------------

On a log-log plot, the viscosity :math:`\eta(\dot{\gamma})` shows a **staircase** pattern with :math:`N` steps, each corresponding to the loss of one bond type. The height of each step is proportional to :math:`G_i \tau_{b,i}`, and the transition between steps occurs at :math:`\dot{\gamma} \approx 1/\tau_{b,i}`.

This staircase structure is a direct fingerprint of the discrete multi-species nature and distinguishes it from continuous spectrum models (e.g., KWW stretched exponential) which produce smooth power-law thinning.

Failure Mode: Bond Hierarchy
===========================================================

.. admonition:: Failure Mode

   **Mechanism:** Under increasing deformation, bond types fail sequentially from weakest (shortest :math:`\tau_{b}`) to strongest (longest :math:`\tau_{b}`). This progressive failure cascade determines the material's nonlinear response.

   **Signature:** Multi-step yielding in flow curves and LAOS, visible as multiple inflection points in :math:`\eta(\dot{\gamma})` and multiple peaks in :math:`I_{3/1}(\gamma_0)`.

   **Physical origin:** Each bond species has a characteristic force scale; exceeding it causes that species to dynamically break faster than it can reform. The critical rate for species :math:`i` is :math:`\dot{\gamma}_{c,i} \approx 1/\tau_{b,i}`.

   **Implication:** The material's response depends on which bonds have "survived" at a given rate — the effective constitutive law changes character as bonds are progressively lost. At low rates the material behaves as a full :math:`N`-species network; at high rates only the strongest (or permanent) species remain.

   **Design insight:** By tuning the ratio :math:`G_{\text{strong}}/G_{\text{weak}}` and :math:`\tau_{\text{strong}}/\tau_{\text{weak}}`, one can engineer materials with prescribed yielding sequences. This is important for self-healing materials (where weak bonds sacrifice first to protect the network), drug delivery gels (where sequential release requires staged degradation), and impact-absorbing materials (where energy dissipation is maximized by progressive bond failure).

See Also
===========================================================

**TNT Shared Reference:**

- :doc:`tnt_protocols` — Full protocol equations and numerical methods
- :doc:`tnt_knowledge_extraction` — Model identification and fitting guidance

**TNT Base Model:**

- :ref:`model-tnt-tanaka-edwards` — Base model (single-mode limit, N=1)

**Closely Related TNT Variants:**

- :ref:`model-tnt-sticky-rouse` — Multi-mode via Rouse dynamics between stickers (spectrum from chain physics rather than discrete species)
- :ref:`model-tnt-loop-bridge` — Two-species with topology coupling (loops ↔ bridges interconversion)
- :ref:`model-tnt-cates` — Living polymer limit (scission/recombination as species dynamics)

**Complementary Extensions:**

- :ref:`model-tnt-bell` — Force-dependent breakage (can be applied per species)
- :ref:`model-tnt-fene-p` — Finite extensibility (can be applied per species)

**External Comparisons:**

- :ref:`model-generalized-maxwell` — Empirical multi-mode model (no network interpretation)
- :doc:`/models/dmt/index` — Scalar structure parameter with thixotropic kinetics

API Reference
===========================================================

.. autoclass:: rheojax.models.tnt.TNTMultiSpecies
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

**Constructor Signature:**

.. code-block:: python

   TNTMultiSpecies(n_species: int = 2)

**Parameters:**

- ``n_species`` : int, default=2
    Number of distinct bond species. Must be >= 2.
    Total parameters will be 2N + 1.

References
===========================================================

Foundational Theory
-----------------------------------------------------------

1. **Tanaka, F., & Edwards, S. F.** (1992).
   Viscoelastic properties of physically crosslinked networks: Transient network theory.
   *Macromolecules*, 25(5), 1516-1523.
   DOI: `10.1021/ma00031a024 <https://doi.org/10.1021/ma00031a024>`_

2. **Rubinstein, M., & Semenov, A. N.** (2001).
   Dynamics of entangled solutions of associating polymers.
   *Macromolecules*, 34(4), 1058-1068.
   DOI: `10.1021/ma0013049 <https://doi.org/10.1021/ma0013049>`_

Dual-Crosslinked Networks
-----------------------------------------------------------

3. **Narita, T., Mayumi, K., Ducouret, G., & Hébraud, P.** (2013).
   Viscoelastic properties of poly(vinyl alcohol) hydrogels having permanent and transient cross-links studied by microrheology, classical rheometry, and dynamic light scattering.
   *Macromolecules*, 46(10), 4174-4183.
   DOI: `10.1021/ma400600f <https://doi.org/10.1021/ma400600f>`_

4. **Sun, J.-Y., Zhao, X., Illeperuma, W. R. K., et al.** (2012).
   Highly stretchable and tough hydrogels.
   *Nature*, 489(7414), 133-136.
   DOI: `10.1038/nature11409 <https://doi.org/10.1038/nature11409>`_

Reversible Network Rheology
-----------------------------------------------------------

5. **Skrzeszewska, P. J., Sprakel, J., de Wolf, F. A., et al.** (2010).
   Fracture and self-healing in a well-defined self-assembled polymer network.
   *Macromolecules*, 43(7), 3542-3548.
   DOI: `10.1021/ma1000173 <https://doi.org/10.1021/ma1000173>`_

6. **Long, R., Mayumi, K., Creton, C., Narita, T., & Hui, C.-Y.** (2014).
   Time dependent behavior of a dual cross-link self-healing gel: Theory and experiments.
   *Macromolecules*, 47(20), 7243-7250.
   DOI: `10.1021/ma501290h <https://doi.org/10.1021/ma501290h>`_
