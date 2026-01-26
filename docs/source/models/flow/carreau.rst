.. _model-carreau:

=====================
Carreau Model
=====================

Quick Reference
---------------

- **Use when:** Polymer melts/solutions with smooth Newtonian-to-power-law transition, well-defined zero-shear viscosity
- **Parameters:** 4 (eta0, eta_inf, lambda, n)
- **Key equation:** :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}`
- **Test modes:** Flow (steady shear, rotation)
- **Material examples:** Polymer melts (PE, PP, PS), polymer solutions, food gels, blood analogues, structured liquids

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\eta`
     - Apparent (shear) viscosity (Pa·s)
   * - :math:`\eta_0`
     - Zero-shear viscosity (Pa·s); Newtonian plateau at :math:`\dot{\gamma} \to 0`
   * - :math:`\eta_{\infty}`
     - Infinite-shear viscosity (Pa·s); Newtonian plateau at :math:`\dot{\gamma} \to \infty`
   * - :math:`\lambda`
     - Time constant (s); reciprocal of critical shear rate
   * - :math:`n`
     - Power-law index (dimensionless); :math:`n < 1` shear-thinning, :math:`n > 1` shear-thickening
   * - :math:`\dot{\gamma}`
     - Shear rate (1/s)
   * - :math:`\sigma`
     - Shear stress (Pa)

Overview
--------

The Carreau model is a four-parameter constitutive equation for generalized Newtonian fluids that describes the smooth transition from a Newtonian plateau at low shear rates to power-law behavior at intermediate shear rates, and optionally to a second Newtonian plateau at very high shear rates. It is one of the most widely used models in polymer rheology and food science.

The model addresses a key limitation of the simple power-law model: the power law predicts infinite viscosity at zero shear rate, which is unphysical. The Carreau model incorporates a finite zero-shear viscosity :math:`\eta_0`, making it suitable for flow simulations where low-shear regions exist (e.g., center of pipe flow, stagnation points in dies).

Historical Context
~~~~~~~~~~~~~~~~~~

Pierre J. Carreau developed this model in 1972 at the University of Wisconsin as part of his work on rheological equations from molecular network theories [1]_. The model builds on earlier work by Cross (1965) [2]_ and was developed in parallel with similar formulations by Yasuda (leading to the Carreau-Yasuda model with an additional parameter).

The Carreau equation has become a standard in:
   - Polymer processing simulation (injection molding, extrusion, blow molding)
   - Blood rheology studies (blood is mildly shear-thinning)
   - Food engineering (sauces, dairy products, doughs)
   - Personal care products (shampoos, lotions)

Its popularity stems from providing physically reasonable behavior across all shear rates while maintaining analytical tractability for computational fluid dynamics.

----

Physical Foundations
--------------------

Molecular Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

The Carreau model captures the macroscopic consequences of molecular-scale phenomena in polymer systems:

**Low shear rates (Newtonian plateau** :math:`\eta_0` **)**:
   At rest or very low shear, polymer chains are in their equilibrium conformations. Chain entanglements form a temporary network. The viscosity is determined by the friction of chains moving past each other:

   - Entangled chains must reptate (snake) through the entanglement tube
   - Network relaxation time is longer than experimental timescale
   - Resistance to flow is maximum → constant :math:`\eta_0`

**Intermediate shear rates (Power-law region)**:
   As shear rate increases past :math:`\dot{\gamma}_c \approx 1/\lambda`:

   - Chains begin to orient and stretch in flow direction
   - Entanglement density decreases (disentanglement)
   - Chain-chain friction decreases
   - Viscosity drops following power law: :math:`\eta \sim \dot{\gamma}^{n-1}`

**High shear rates (Second Newtonian plateau** :math:`\eta_{\infty}` **)**:
   At very high shear:

   - Chains are fully oriented in flow direction
   - Maximum possible disentanglement achieved
   - Only solvent (or chain segment) friction remains
   - Viscosity approaches constant :math:`\eta_{\infty}`

**Note**: Many polymer systems never reach the high-shear plateau experimentally due to flow instabilities, thermal degradation, or shear banding at extreme rates.

Connection to Polymer Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The time constant :math:`\lambda` relates to molecular relaxation:

.. math::

   \lambda \sim \tau_d \sim \frac{\zeta N^3}{k_B T} \sim \frac{M^3}{\rho RT / M_e}

where:
   - :math:`\tau_d` = reptation (disengagement) time
   - :math:`\zeta` = monomeric friction coefficient
   - :math:`N` = degree of polymerization
   - :math:`M` = molecular weight
   - :math:`M_e` = entanglement molecular weight

The power-law index :math:`n` reflects the molecular weight distribution:
   - **Narrow distribution** (monodisperse): :math:`n \approx 0.4-0.5`
   - **Broad distribution** (polydisperse): :math:`n \approx 0.2-0.3`

Material Examples with Typical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative Carreau parameters for common materials
   :header-rows: 1
   :widths: 25 15 12 12 10 10 16

   * - Material
     - :math:`\eta_0` (Pa·s)
     - :math:`\eta_{\infty}` (Pa·s)
     - :math:`\lambda` (s)
     - :math:`n`
     - T (°C)
     - Ref
   * - LDPE (190°C)
     - 4,200
     - 0
     - 0.55
     - 0.35
     - 190
     - [3]_
   * - HDPE (190°C)
     - 6,500
     - 0
     - 0.42
     - 0.47
     - 190
     - [3]_
   * - Polypropylene (230°C)
     - 2,100
     - 0
     - 0.18
     - 0.38
     - 230
     - [3]_
   * - Polystyrene (200°C)
     - 8,500
     - 0
     - 1.2
     - 0.28
     - 200
     - [4]_
   * - Blood (37°C)
     - 0.056
     - 0.0035
     - 3.3
     - 0.35
     - 37
     - [5]_
   * - Xanthan gum 0.5%
     - 12.5
     - 0.01
     - 8.2
     - 0.22
     - 25
     - [6]_
   * - Polymer solution 5%
     - 0.8
     - 0.001
     - 0.15
     - 0.55
     - 25
     - [7]_

----

Governing Equations
-------------------

Mathematical Formulation
~~~~~~~~~~~~~~~~~~~~~~~~

The Carreau viscosity function is:

.. math::

   \eta(\dot{\gamma}) = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}

**Derivation from molecular network theory**:

Carreau derived the model by considering the destruction of network junctions under flow [1]_:

**Step 1**: Assume network structure with junctions breaking at rate proportional to :math:`(\lambda\dot{\gamma})^2`

**Step 2**: Define structural variable :math:`\xi` representing fraction of intact junctions:

.. math::

   \xi = [1 + (\lambda\dot{\gamma})^2]^{-1}

**Step 3**: Relate viscosity to network integrity:

.. math::

   \eta - \eta_{\infty} = (\eta_0 - \eta_{\infty}) \xi^{(1-n)/2}

**Step 4**: Substitute and simplify:

.. math::

   \eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}

Shear Stress Relation
~~~~~~~~~~~~~~~~~~~~~

The shear stress is:

.. math::

   \sigma = \eta(\dot{\gamma}) \cdot \dot{\gamma} = \left\{ \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2} \right\} \dot{\gamma}

This is a **monotonic** function of :math:`\dot{\gamma}` for :math:`n > 0`, ensuring stable flow (no shear banding from non-monotonic flow curves).

Limiting Cases
~~~~~~~~~~~~~~

.. list-table:: Asymptotic behavior
   :header-rows: 1
   :widths: 25 25 25 25

   * - Regime
     - Condition
     - :math:`\eta(\dot{\gamma})`
     - Physical interpretation
   * - Low shear
     - :math:`\lambda\dot{\gamma} \ll 1`
     - :math:`\approx \eta_0`
     - Newtonian plateau
   * - Critical
     - :math:`\lambda\dot{\gamma} = 1`
     - :math:`\approx 0.5(\eta_0 + \eta_{\infty})` for :math:`n=0.5`
     - Transition point
   * - Power-law
     - :math:`1 \ll \lambda\dot{\gamma} \ll \lambda\dot{\gamma}_{max}`
     - :math:`\approx (\eta_0 - \eta_{\infty})(\lambda\dot{\gamma})^{n-1}`
     - Shear-thinning
   * - High shear
     - :math:`\lambda\dot{\gamma} \gg 1`, :math:`\eta_{\infty} > 0`
     - :math:`\to \eta_{\infty}`
     - Second Newtonian plateau

Special Cases
~~~~~~~~~~~~~

**Newtonian fluid** (:math:`n = 1`):

.. math::

   \eta = \eta_{\infty} + (\eta_0 - \eta_{\infty}) = \eta_0

**Strong shear-thinning** (:math:`\eta_{\infty} = 0`):

.. math::

   \eta = \eta_0 [1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}

This simplified form is commonly used when high-shear data is unavailable.

**Power-law approximation** (mid-range only):

.. math::

   \eta \approx K \dot{\gamma}^{n-1}, \quad K = \eta_0 \lambda^{1-n}

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
   * - ``eta0``
     - :math:`\eta_0`
     - Pa·s
     - :math:`10^{-3} - 10^{12}`
     - Zero-shear viscosity; Newtonian plateau at low rates
   * - ``eta_inf``
     - :math:`\eta_{\infty}`
     - Pa·s
     - :math:`10^{-6} - 10^{6}`
     - Infinite-shear viscosity; often set to 0 or small value
   * - ``lambda_``
     - :math:`\lambda`
     - s
     - :math:`10^{-6} - 10^{6}`
     - Time constant; :math:`1/\lambda` is critical shear rate
   * - ``n``
     - :math:`n`
     - —
     - :math:`0.01 - 1.0`
     - Power-law index; < 1 thinning, = 1 Newtonian

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**eta0 (Zero-Shear Viscosity)**:
   - **Physical meaning**: Viscosity when polymer chains are in equilibrium state
   - **Molecular origin**: Chain entanglement density and reptation resistance
   - **Typical ranges**:
      - Polymer melts: :math:`10^3 - 10^7` Pa·s
      - Polymer solutions: :math:`10^{-2} - 10^2` Pa·s
      - Blood: :math:`0.03 - 0.1` Pa·s
   - **Scaling**: :math:`\eta_0 \sim M^{3.4}` for entangled polymers (:math:`M > M_c`)

**eta_inf (Infinite-Shear Viscosity)**:
   - **Physical meaning**: Residual viscosity at maximum chain orientation
   - **Molecular origin**: Solvent viscosity + fully aligned chain segment friction
   - **Typical values**: Often set to 0 for melts (no solvent); :math:`\eta_{\infty} \approx \eta_{\text{solvent}}` for solutions
   - **Fitting note**: May be poorly determined if high-shear data is limited

**lambda (Time Constant)**:
   - **Physical meaning**: Characteristic relaxation time of the material
   - **Molecular origin**: Longest relaxation time (reptation time for entangled chains)
   - **Relation to critical shear rate**: :math:`\dot{\gamma}_c = 1/\lambda`
   - **Typical ranges**:
      - Polymer melts: :math:`10^{-2} - 10^2` s
      - Polymer solutions: :math:`10^{-4} - 10^0` s
   - **Scaling**: :math:`\lambda \sim M^{3.4}` (same as :math:`\eta_0`)

**n (Power-Law Index)**:
   - **Physical meaning**: Degree of shear-thinning
   - **Molecular origin**: Polydispersity (broad MWD → lower n) and chain flexibility
   - **Interpretation**:
      - :math:`n = 1`: Newtonian
      - :math:`n = 0.5`: Moderate thinning (typical for monodisperse melts)
      - :math:`n = 0.2`: Strong thinning (broad MWD, rigid chains)
      - :math:`n > 1`: Shear-thickening (rare, cornstarch suspensions)

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Generalized Newtonian fluid**: Stress depends only on current strain rate (no memory)
2. **Isothermal**: Temperature is constant (use separate :math:`\eta_0(T)` for T-dependence)
3. **Simple shear flow**: Steady, unidirectional shear (not oscillatory or extensional)
4. **Incompressible**: Constant density
5. **Inelastic**: No normal stress differences or elastic recoil

Data Requirements
~~~~~~~~~~~~~~~~~

- **Required**: Flow curve :math:`\eta(\dot{\gamma})` or :math:`\sigma(\dot{\gamma})` from steady shear
- **Shear rate range**: At least 3 decades, ideally 4-5 decades
- **Coverage**: Should span both plateaus and power-law region
- **Recommended**: :math:`\dot{\gamma} = 10^{-2}` to :math:`10^{4}` s\ :math:`^{-1}` for polymers

Limitations
~~~~~~~~~~~

**No viscoelasticity**:
   Cannot predict storage/loss moduli, stress relaxation, creep, or normal stresses.
   Use Maxwell/Zener for linear viscoelasticity, Oldroyd-B for nonlinear.

**No yield stress**:
   The model predicts :math:`\sigma \to 0` as :math:`\dot{\gamma} \to 0`.
   Use Herschel-Bulkley or Bingham for yield stress fluids.

**No time-dependent behavior**:
   Cannot capture thixotropy, rheopexy, or startup transients.
   Use structural kinetics models (DMT, fluidity) for thixotropy.

**Temperature dependence not built-in**:
   Must fit at each temperature or use time-temperature superposition:
   :math:`\eta_0(T) = \eta_0(T_r) \cdot a_T`, :math:`\lambda(T) = \lambda(T_r) \cdot a_T`

----

Regimes and Behavior
--------------------

Flow Curve Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~~~

A log-log plot of :math:`\eta` vs :math:`\dot{\gamma}` shows three regions:

1. **Zero-shear plateau** (:math:`\dot{\gamma} < 0.1/\lambda`): Horizontal line at :math:`\eta_0`
2. **Transition region** (:math:`0.1/\lambda < \dot{\gamma} < 10/\lambda`): Curved transition
3. **Power-law region** (:math:`\dot{\gamma} > 10/\lambda`): Linear with slope :math:`n - 1`

**Diagnostic**: Plot :math:`\log\eta` vs :math:`\log\dot{\gamma}`:
   - **Slope = 0**: Newtonian plateau
   - **Slope = n - 1**: Power-law region (−0.5 to −0.8 typical)

Stress vs Strain Rate
~~~~~~~~~~~~~~~~~~~~~

The stress :math:`\sigma = \eta \dot{\gamma}` is always monotonically increasing with :math:`\dot{\gamma}`:

- **Low** :math:`\dot{\gamma}`: :math:`\sigma \approx \eta_0 \dot{\gamma}` (slope = 1 on log-log)
- **High** :math:`\dot{\gamma}`: :math:`\sigma \approx K \dot{\gamma}^n` where :math:`K = \eta_0 \lambda^{1-n}` (slope = n)

This monotonicity ensures flow stability (no constitutive instabilities).

----

What You Can Learn
------------------

This section explains how to interpret fitted Carreau parameters to gain
insights about your material's molecular structure and processing behavior.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**eta0 (Zero-Shear Viscosity)**:
   The zero-shear viscosity reveals the entanglement state and molecular weight:

   - **Low values (< 100 Pa·s)**: Short chains, below entanglement threshold :math:`M_c`, or dilute solution
   - **Moderate values (100-10,000 Pa·s)**: Typical processing-grade polymers, well-entangled
   - **High values (> 10,000 Pa·s)**: Ultra-high MW, high entanglement density, or very low temperature

   *For graduate students*: Use the scaling :math:`\eta_0 \sim M^{3.4}` (valid for :math:`M > 2M_e`) to estimate molecular weight. Compare with GPC data to validate. The time constant :math:`\lambda` shares the same molecular weight scaling, connecting both parameters to the reptation framework.

   *For practitioners*: :math:`\eta_0` directly controls low-shear processes (gravity-driven flow, low-speed filling, sag resistance). Target :math:`\eta_0` based on process requirements—higher for vertical coatings, lower for rapid filling.

**eta_inf (Infinite-Shear Viscosity)**:
   The high-shear viscosity plateau indicates residual friction:

   - **eta_inf ≈ 0**: Complete structural breakdown, pure melt behavior
   - **eta_inf > 0**: Solvent or matrix contribution remains (polymer solutions)

   *For graduate students*: For solutions, :math:`\eta_{\infty}` approximates the solvent viscosity plus a minor hydrodynamic contribution from fully aligned chains.

   *For practitioners*: :math:`\eta_{\infty}` controls high-speed operations like spray atomization and fast coating. Lower values enable easier processing at high shear rates.

**lambda (Time Constant)**:
   The relaxation time identifies the critical shear rate for structural response:

   - **Short** :math:`\lambda` **(<0.1 s)**: Fast-relaxing, good for high-speed processing
   - **Long** :math:`\lambda` **(>10 s)**: Slow-relaxing, melt memory effects important, potential for elastic instabilities

   *For graduate students*: :math:`\lambda` approximates the terminal relaxation time :math:`\tau_d`. The Deborah number :math:`De = \lambda \dot{\gamma}` indicates elastic vs viscous dominance: :math:`De > 1` gives elastic effects, :math:`De < 1` gives viscous flow.

   *For practitioners*: Processes with :math:`\dot{\gamma} > 1/\lambda` operate in the shear-thinning region, reducing pumping power. The critical shear rate :math:`1/\lambda` marks the onset of significant viscosity reduction.

**n (Power-Law Index)**:
   The flow index reflects molecular weight distribution breadth:

   - **n close to 1 (0.7-0.9)**: Narrow MWD, nearly Newtonian behavior, monodisperse
   - **n moderate (0.4-0.6)**: Typical commercial polymers, moderate polydispersity
   - **n low (0.2-0.4)**: Broad MWD, strong shear-thinning, or branched architectures

   *For graduate students*: The empirical correlation :math:`n \approx 1 - 0.3 \cdot \text{PDI}` (where PDI = Mw/Mn) provides rough MWD estimates. For branched polymers, lower :math:`n` reflects additional relaxation modes from long-chain branching.

   *For practitioners*: Low :math:`n` means easier flow at high shear (injection molding, extrusion) but potential for flow marks and non-uniform cooling. High :math:`n` gives more uniform velocity profiles and better coating consistency.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Carreau Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Pattern
     - Material Type
     - Typical Materials
     - Processing Implications
   * - High :math:`\eta_0`, low :math:`n`
     - High-MW, broad MWD polymer
     - UHMWPE, broad-MWD PP
     - Strong die swell, long residence times, difficult extrusion
   * - Low :math:`\eta_0`, high :math:`n`
     - Low-MW, narrow MWD polymer
     - Oligomers, low-MW lubricants
     - Newtonian-like, easy processing, low melt strength
   * - Long :math:`\lambda`, low :math:`n`
     - High elasticity polymer
     - Branched LDPE, ionomers
     - Melt fracture risk at high rates, good for blow molding
   * - Short :math:`\lambda`, moderate :math:`n`
     - Linear commodity polymer
     - LLDPE, linear PP
     - Good processability window, stable extrusion
   * - Low :math:`\eta_0/\eta_{\infty}` ratio
     - Dilute solution
     - Polymer in good solvent
     - Minimal shear-thinning benefit

----

Experimental Design
-------------------

When to Use Carreau Model
~~~~~~~~~~~~~~~~~~~~~~~~~

**Use this model when**:
   - Material shows clear zero-shear plateau
   - Smooth transition to power-law behavior
   - No yield stress (material flows at all stresses)
   - Inelastic approximation acceptable

**Consider alternatives when**:
   - **Sharper transition**: Carreau-Yasuda (adds :math:`a` parameter)
   - **Different functional form**: Cross model (denominator instead of power)
   - **Yield stress present**: Herschel-Bulkley, Bingham
   - **Viscoelasticity needed**: Maxwell, Oldroyd-B
   - **Thixotropy observed**: DMT, fluidity models

Recommended Test Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~

**Steady Shear Flow Curve (Rotational Rheometry)**

**Step 1: Sample preparation**
   - Melt polymers above :math:`T_g + 50°C`, anneal 5-10 min
   - Load fresh sample for each complete sweep (avoid shear history)
   - Use 25 mm parallel plates with 1 mm gap (typical)

**Step 2: Thermal equilibration**
   - Equilibrate at test temperature for 10-15 min
   - Verify temperature uniformity (< 0.5°C variation)

**Step 3: Flow curve measurement**
   - Sweep shear rate: :math:`10^{-2}` to :math:`10^{3}` s\ :math:`^{-1}` (or instrument limit)
   - Log spacing: 5-10 points per decade
   - Measurement time: Auto (until stress steady) or 30 s minimum
   - Direction: Ascending preferred (avoid thixotropic artifacts)

**Step 4: Data quality checks**
   - Torque > minimum specification (typically 0.1 µNm)
   - No slip (compare with serrated plates if suspect)
   - No edge fracture (visual inspection, stress drop)

Sample Preparation
~~~~~~~~~~~~~~~~~~

**Polymer melts**:
   - Compression mold at :math:`T > T_m + 20°C`
   - Cool slowly to avoid residual stress
   - Discs: 25 mm diameter, 1-2 mm thick
   - Check for bubbles (reduce if present)

**Polymer solutions**:
   - Dissolve with gentle stirring (avoid degradation)
   - Filter (0.45 µm) to remove aggregates
   - Degas if bubbles present
   - Fresh preparation for each measurement day

**Blood and biological fluids**:
   - Use anticoagulant (EDTA, heparin)
   - Measure within 4 hours of collection
   - Temperature control critical (37 ± 0.1°C)
   - Use Couette geometry to minimize sample volume

Common Experimental Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Troubleshooting experimental issues
   :header-rows: 1
   :widths: 25 35 40

   * - Artifact
     - Symptom
     - Solution
   * - Wall slip
     - :math:`\eta` artificially low, gap-dependent
     - Serrated plates, reduce gap, Mooney analysis
   * - Sample degradation
     - :math:`\eta` drifts during measurement
     - Reduce temperature, inert atmosphere, antioxidants
   * - Edge fracture
     - Sudden stress drop at high :math:`\dot{\gamma}`
     - Cone-plate geometry, reduce strain, add edge sealant
   * - Inertia effects
     - Upturn at high :math:`\dot{\gamma}`
     - Correct with inertia routine, limit max rate
   * - Secondary flow
     - Anomalous behavior at high :math:`\dot{\gamma}`
     - Verify Taylor number, use narrower gaps

----

Computational Implementation
----------------------------

JAX Vectorization
~~~~~~~~~~~~~~~~~

RheoJAX implements the Carreau model with full JAX vectorization:

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

   @jax.jit
   def carreau_viscosity(gamma_dot, eta0, eta_inf, lambda_, n):
       """Vectorized Carreau viscosity computation.

       Parameters
       ----------
       gamma_dot : array_like
           Shear rate values
       eta0 : float
           Zero-shear viscosity
       eta_inf : float
           Infinite-shear viscosity
       lambda_ : float
           Time constant
       n : float
           Power-law index

       Returns
       -------
       eta : array_like
           Viscosity values at given shear rates
       """
       factor = jnp.power(1 + jnp.square(lambda_ * gamma_dot), (n - 1) / 2)
       return eta_inf + (eta0 - eta_inf) * factor

Key optimizations:
   - JIT compilation for 10-100x speedup
   - Vectorization over shear rate arrays
   - Automatic differentiation for sensitivity analysis

Numerical Stability
~~~~~~~~~~~~~~~~~~~

The model is numerically stable across typical parameter ranges:

- **Large** :math:`\lambda\dot{\gamma}`: Factor approaches :math:`(\lambda\dot{\gamma})^{n-1}`, no overflow for :math:`n > 0`
- **Small** :math:`\lambda\dot{\gamma}`: Factor approaches 1, stable
- **Edge case** :math:`\eta_0 = \eta_{\infty}`: Returns constant viscosity (correct)

----

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From flow curve features**

**Step 1**: Estimate :math:`\eta_0` from low-shear plateau
   :math:`\eta_0 \approx \eta(\dot{\gamma} \to 0)`

**Step 2**: Estimate :math:`\eta_{\infty}` from high-shear plateau (if visible)
   :math:`\eta_{\infty} \approx \eta(\dot{\gamma} \to \infty)`, or set to 0

**Step 3**: Find transition point where :math:`\eta` drops to :math:`0.5(\eta_0 + \eta_{\infty})`
   :math:`\lambda \approx 1 / \dot{\gamma}_{1/2}`

**Step 4**: Estimate :math:`n` from power-law slope
   Fit :math:`\log\eta = \log K + (n-1)\log\dot{\gamma}` in mid-range

**Method 2: Using derivative**

.. code-block:: python

   # Numerical derivative on log scale
   d_log_eta = np.gradient(np.log(eta), np.log(gamma_dot))
   n_estimate = 1 + np.min(d_log_eta)  # Most negative slope

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for Carreau (4 parameters)
   - Converges in < 100 iterations typically
   - 5-270x faster than scipy.optimize

**Bounds (recommended)**:
   - :math:`\eta_0`: [1e-2, 1e10] Pa·s
   - :math:`\eta_{\infty}`: [0, :math:`0.1 \cdot \eta_0`] Pa·s
   - :math:`\lambda`: [1e-6, 1e4] s
   - :math:`n`: [0.1, 1.0]

**Bayesian inference (NUTS)**:
   - Use when uncertainty quantification needed
   - Warm-start from NLSQ for efficiency
   - Priors: Log-normal for :math:`\eta_0, \eta_{\infty}, \lambda`; Beta for :math:`n`

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - :math:`\eta_0` poorly determined
     - Low-shear data missing or noisy
     - Extend to lower shear rates, more averages
   * - :math:`\eta_{\infty}` at upper bound
     - High-shear plateau not reached
     - Fix :math:`\eta_{\infty} = 0` or extend :math:`\dot{\gamma}` range
   * - :math:`\lambda` outside data range
     - Transition not captured
     - Adjust shear rate sweep to bracket :math:`1/\lambda`
   * - :math:`n` near 1 (Newtonian)
     - Material barely shear-thins
     - Verify non-Newtonian behavior; use simpler model
   * - Poor fit quality
     - Systematic residuals
     - Try Carreau-Yasuda (adds transition sharpness)

----

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import Carreau

   # Shear rate sweep data
   gamma_dot = np.logspace(-2, 4, 100)
   eta_data = experimental_viscosity(gamma_dot)

   # Create and fit model
   model = Carreau()
   model.fit(gamma_dot, eta_data, test_mode='rotation')

   # Extract parameters
   eta0 = model.parameters.get_value('eta0')
   eta_inf = model.parameters.get_value('eta_inf')
   lambda_ = model.parameters.get_value('lambda_')
   n = model.parameters.get_value('n')

   print(f"Zero-shear viscosity: {eta0:.1f} Pa·s")
   print(f"Time constant: {lambda_:.3f} s")
   print(f"Power-law index: {n:.3f}")

   # Predict at new shear rates
   gamma_dot_new = np.logspace(-3, 5, 200)
   eta_pred = model.predict(gamma_dot_new)

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Carreau

   model = Carreau()
   model.fit(gamma_dot, eta_data, test_mode='rotation')

   # Bayesian with NLSQ warm-start
   result = model.fit_bayesian(
       gamma_dot, eta_data,
       test_mode='rotation',
       num_warmup=1000,
       num_samples=2000
   )

   # Credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   for param, (low, high) in intervals.items():
       print(f"{param}: [{low:.3g}, {high:.3g}]")

Stress Prediction
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Predict shear stress (sigma = eta * gamma_dot)
   stress = model.predict(gamma_dot, test_mode='rotation') * gamma_dot

   # For simulation: apparent viscosity at specific rates
   process_rates = np.array([100, 1000, 10000])  # Typical processing rates (s^-1)
   process_viscosities = model.predict(process_rates, test_mode='rotation')

----

Model Comparison
----------------

When to Use Carreau vs Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - If you observe...
     - Consider...
     - Because...
   * - Sharp transition
     - Carreau-Yasuda
     - Extra parameter :math:`a` controls transition shape
   * - Different functional form fits better
     - Cross
     - :math:`\eta = \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda\dot{\gamma})^m}`
   * - Yield stress (no flow at low stress)
     - Herschel-Bulkley
     - :math:`\sigma = \sigma_y + K\dot{\gamma}^n`
   * - Need viscoelastic properties
     - Maxwell + Cox-Merz
     - Fit oscillatory, predict viscosity via Cox-Merz rule

----

See Also
--------

- :doc:`carreau_yasuda` — adds Yasuda exponent :math:`a` for sharper transitions
- :doc:`cross` — alternative sigmoidal form with denominator exponent
- :doc:`power_law` — local approximation of Carreau mid-rate region
- :doc:`herschel_bulkley` — for yield stress fluids
- :doc:`../../transforms/mastercurve` — time-temperature superposition for T-dependence
- :doc:`../../examples/flow/carreau_fitting` — complete fitting tutorial

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.Carreau`

----

References
----------

.. [1] Carreau, P. J. "Rheological equations from molecular network theories."
   *Transactions of the Society of Rheology*, **16**, 99-127 (1972).
   https://doi.org/10.1122/1.549276

.. [2] Cross, M. M. "Rheology of non-Newtonian fluids: a new flow equation for pseudoplastic systems."
   *Journal of Colloid Science*, **20**, 417-437 (1965).
   https://doi.org/10.1016/0095-8522(65)90022-X

.. [3] Dealy, J. M. & Wissbrun, K. F. *Melt Rheology and Its Role in Plastics Processing*.
   Van Nostrand Reinhold, New York (1990).

.. [4] Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd Edition.
   Wiley, New York (1980).

.. [5] Thurston, G. B. "Viscoelasticity of human blood."
   *Biophysical Journal*, **12**, 1205-1217 (1972).
   https://doi.org/10.1016/S0006-3495(72)86156-3

.. [6] Barnes, H. A., Hutton, J. F. & Walters, K. *An Introduction to Rheology*.
   Elsevier, Amsterdam (1989).

.. [7] Larson, R. G. *Constitutive Equations for Polymer Melts and Solutions*.
   Butterworths, Boston (1988).

.. [8] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994).

.. [9] Osswald, T. A. & Menges, G. *Materials Science of Polymers for Engineers*, 3rd Edition.
   Hanser, Munich (2012).

.. [10] Morrison, F. A. *Understanding Rheology*.
   Oxford University Press (2001).

.. [11] Bird, R. B., Armstrong, R. C. & Hassager, O. *Dynamics of Polymeric Liquids, Vol. 1: Fluid Mechanics*, 2nd Edition.
   Wiley, New York (1987).

Further Reading
~~~~~~~~~~~~~~~

- Yasuda, K. "Investigation of the analogies between viscometric and linear viscoelastic properties of polystyrene fluids."
  PhD Thesis, MIT (1979). [Original Carreau-Yasuda derivation]

- Tanner, R. I. *Engineering Rheology*, 2nd Edition.
  Oxford University Press (2000). [Comprehensive treatment of flow models]
