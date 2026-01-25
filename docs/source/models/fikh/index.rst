Fractional IKH (FIKH) Models
============================

This section documents the Fractional Isotropic-Kinematic Hardening (FIKH) family
of models for thixotropic elasto-viscoplastic (TEvp) materials with power-law memory.


Overview
--------

The FIKH family extends the classical MIKH framework by replacing the integer-order
structure kinetics with a **Caputo fractional derivative**. This captures the
**power-law memory** observed in many complex fluids:

- **Standard IKH (α = 1)**: Exponential recovery λ ~ exp(-t/τ)
- **Fractional FIKH (α < 1)**: Power-law recovery λ ~ t^(α-1) at long times

Fractional derivatives introduce a fading memory where recent deformation history
affects the current structure more than distant past. This single parameter α captures
a broad distribution of restructuring timescales without requiring multiple modes.

**Thermokinematic coupling** adds:

- Temperature-dependent yield stress: σ_y(λ, T)
- Arrhenius viscosity: η(T) = η₀·exp(E_a/RT)
- Thermal evolution from plastic dissipation

These models are particularly suited for:

- Waxy crude oils (cold restart, pipeline flow assurance)
- Colloidal gels with hierarchical structure
- Materials exhibiting stretched-exponential recovery
- Systems with thermal feedback (shear heating)


Physical Assumptions
--------------------

The FIKH model framework rests on several key physical assumptions:

1. **Homogeneous simple shear**: 0D model; no spatial gradients (use nonlocal variants for shear banding)
2. **Structural kinetics**: Single structure parameter λ ∈ [0,1] captures all microstructural changes
3. **Fading memory**: Recent deformation affects structure more than distant past (captured by α)
4. **Separation of timescales**: Elastic response (G) is fast; thixotropic recovery (τ_thix) is slow
5. **Arrhenius thermal activation**: Temperature dependence follows exp(E/RT) scaling
6. **Incompressibility**: Constant density (volumetric deformation neglected)
7. **Small temperature changes**: No phase transitions; material remains in fluid state

When These Assumptions Break Down
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Shear banding**: Use fluidity-nonlocal or DMT-nonlocal models instead
- **Multiple structural populations**: Use FMLIKH multi-mode variant
- **Large temperature gradients**: Requires coupling with heat conduction (not included)
- **Very fast flows**: Inertial effects may become important
- **True exponential recovery**: Use MIKH for computational efficiency


Mathematical Background
-----------------------

Caputo Fractional Derivative
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For 0 < α < 1, the **Caputo fractional derivative** is defined as:

.. math::

   D_t^\alpha f(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{f'(s)}{(t-s)^\alpha} \, ds

Key properties:

- D^α (constant) = 0 (compatible with initial conditions)
- As α → 1: D^α f → df/dt (recovers ordinary derivative)
- Introduces **power-law memory** with kernel (t-s)^(-α)

**Why Caputo over Riemann-Liouville?** The Caputo derivative allows physical
interpretation of initial conditions: D^α(constant) = 0, so λ(0) = λ₀ is meaningful.

Fractional Structure Evolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core FIKH equation replaces standard thixotropic kinetics:

.. math::

   D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}} - \Gamma \lambda |\dot{\gamma}^p|

Where:

- λ ∈ [0, 1]: Structure parameter (1 = fully structured, 0 = broken)
- τ_thix: Thixotropic rebuilding time scale [s]
- Γ: Structural breakdown coefficient [-]
- α ∈ (0, 1): Fractional order (memory strength)

**Physical interpretation**:

- **α → 0**: Strong memory ("hyper-slow" glassy dynamics)
- **α → 1**: Weak memory (fast exponential recovery, recovers MIKH)
- **α = 0.5**: Intermediate power-law relaxation

**Memory Kernel Representation**:

.. math::

   \lambda(t) = \lambda_0 + \int_0^t K(t-s) \left[\frac{1-\lambda(s)}{\tau_{thix}} - \Gamma\lambda(s)|\dot{\gamma}^p(s)|\right] ds

where K(t) = t^(α-1)/Γ(α) is the Mittag-Leffler kernel.

Thermokinematic Coupling
^^^^^^^^^^^^^^^^^^^^^^^^

**Temperature-dependent yield stress**:

.. math::

   \sigma_y(\lambda, T) = (\sigma_{y0} + \Delta\sigma_y \cdot \lambda^{m_y}) \exp\left(\frac{E_y}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)

**Arrhenius viscosity**:

.. math::

   \eta(T) = \eta_{ref} \exp\left(\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)

**Temperature evolution**:

.. math::

   \rho c_p \dot{T} = \chi \sigma \dot{\gamma}^p - h(T - T_{env})

Where χ is the Taylor-Quinney coefficient (~0.9) and h is the heat transfer coefficient.


Model Hierarchy
---------------

::

   FIKH Family
   │
   ├── FIKH (Single Mode)
   │   └── 12-20 parameters (depends on optional thermal/hardening)
   │   └── Single fractional structure variable
   │   └── Power-law + exponential relaxation
   │
   └── FMLIKH (Multi-Layer)
       ├── Per-mode: G_i, η_i, C_i, γ_dyn_i, τ_thix_i, Γ_i
       │   └── Shared or per-mode fractional order
       │
       └── Shared: σ_y, thermal parameters
           └── Global yield with distributed kinetics


When to Use FIKH
----------------

Experimental Signatures
^^^^^^^^^^^^^^^^^^^^^^^

**Use FIKH when you observe**:

1. **Power-law stress relaxation** at long times: G(t) ~ t^(-α), not exp(-t/τ)
2. **Stretched exponential recovery** after shear cessation
3. **Broad relaxation spectrum** in frequency sweep (Cole-Cole depression)
4. **Delayed yielding** in creep tests below apparent yield stress
5. **Temperature-dependent flow** with Arrhenius-like behavior
6. **Stress overshoot with long tail** in startup (not sharp exponential decay)

Decision Tree
^^^^^^^^^^^^^

::

   Is recovery exponential (single timescale)?
   ├── YES → Use MIKH (simpler, faster)
   └── NO → Is recovery power-law?
       ├── YES → Use FIKH (single α captures spectrum)
       └── NO → Is there hierarchical structure?
           ├── YES → Use FMLIKH (multiple modes)
           └── NO → Consider SGR or DMT models

Model Comparison
^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Behavior
     - Single Mode (FIKH)
     - Multi-Mode (FMLIKH)
   * - Power-law recovery
     - ✓ Use this
     - Also works
   * - Single structural population
     - ✓ Use this
     - Overkill
   * - Broad relaxation spectrum
     - Limited
     - ✓ Use this
   * - Few parameters needed
     - ✓ Use this
     - More params
   * - Hierarchical structure
     - Limited
     - ✓ Use this
   * - When α → 1 (exponential)
     - Consider MIKH
     - Consider ML-IKH

Material-Specific Recommendations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 20 15 45
   :header-rows: 1

   * - Material
     - Recommended Model
     - Typical α
     - Key Protocol
   * - Waxy crude oils
     - FIKH (thermal)
     - 0.5-0.7
     - Startup at different T
   * - Colloidal gels
     - FMLIKH
     - 0.3-0.6
     - Frequency sweep
   * - Food gels
     - FIKH
     - 0.6-0.8
     - Creep recovery
   * - Drilling muds
     - FIKH (thermal)
     - 0.4-0.6
     - Flow curve + relaxation
   * - Greases
     - FIKH
     - 0.5-0.7
     - LAOS + startup
   * - Cement pastes
     - FMLIKH
     - 0.4-0.6
     - Multiple rest times


Key Features
------------

**Fractional Structure Evolution:**

- Caputo derivative captures power-law fading memory
- Single α parameter spans exponential (α=1) to strong memory (α→0)
- Mittag-Leffler relaxation generalizes the exponential

**Armstrong-Frederick Kinematic Hardening:**

- Back-stress A tracks deformation history
- Captures Bauschinger effect in cyclic loading
- Dynamic recovery prevents unbounded hardening

**Full Thermokinematic Coupling:**

- Arrhenius temperature dependence for viscosity
- Structure-temperature yield stress coupling
- Plastic dissipation heating with heat loss

**Supported Protocols:**

- Flow curve (steady state)
- Startup shear (stress overshoot)
- Stress relaxation (Mittag-Leffler decay)
- Creep (delayed yielding, thermal runaway)
- SAOS (fractional Maxwell moduli)
- LAOS (harmonic generation, Lissajous figures)


Parameter Reference
-------------------

Base Parameters (Always Present)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``G``
     - 1000
     - (0.1, 10⁹)
     - Pa
     - Shear modulus
   * - ``eta``
     - 10⁶
     - (10⁻³, 10¹²)
     - Pa·s
     - Maxwell viscosity (τ = η/G)
   * - ``C``
     - 500
     - (0, 10⁹)
     - Pa
     - Kinematic hardening modulus
   * - ``gamma_dyn``
     - 1.0
     - (0, 10⁴)
     - —
     - Dynamic recovery parameter
   * - ``m``
     - 1.0
     - (0.5, 3)
     - —
     - AF recovery exponent
   * - ``sigma_y0``
     - 10
     - (0, 10⁹)
     - Pa
     - Minimal yield stress (destructured)
   * - ``delta_sigma_y``
     - 50
     - (0, 10⁹)
     - Pa
     - Structural yield stress contribution
   * - ``tau_thix``
     - 1.0
     - (10⁻⁶, 10¹²)
     - s
     - Thixotropic rebuilding time
   * - ``Gamma``
     - 0.5
     - (0, 10⁴)
     - —
     - Structural breakdown coefficient
   * - ``alpha_structure``
     - 0.5
     - (0.05, 0.99)
     - —
     - Fractional order for structure
   * - ``eta_inf``
     - 0.1
     - (0, 10⁹)
     - Pa·s
     - High-shear (solvent) viscosity
   * - ``mu_p``
     - 10⁻³
     - (10⁻⁹, 10³)
     - Pa·s
     - Plastic viscosity (Perzyna)

Thermal Parameters (``include_thermal=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``T_ref``
     - 298.15
     - (200, 500)
     - K
     - Reference temperature
   * - ``E_a``
     - 50000
     - (0, 2×10⁵)
     - J/mol
     - Viscosity activation energy
   * - ``E_y``
     - 30000
     - (0, 2×10⁵)
     - J/mol
     - Yield stress activation energy
   * - ``m_y``
     - 1.0
     - (0.5, 2)
     - —
     - Structure exponent for yield
   * - ``rho_cp``
     - 4×10⁶
     - (10⁵, 10⁸)
     - J/(m³·K)
     - Volumetric heat capacity
   * - ``chi``
     - 0.9
     - (0, 1)
     - —
     - Taylor-Quinney coefficient
   * - ``h``
     - 100
     - (0, 10⁶)
     - W/(m²·K)
     - Heat transfer coefficient
   * - ``T_env``
     - 298.15
     - (200, 500)
     - K
     - Environmental temperature

Isotropic Hardening Parameters (``include_isotropic_hardening=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``Q_iso``
     - 0
     - (0, 10⁹)
     - Pa
     - Isotropic hardening saturation
   * - ``b_iso``
     - 1.0
     - (0, 100)
     - —
     - Isotropic hardening rate


Protocol Equations
------------------

Steady-State Flow Curve
^^^^^^^^^^^^^^^^^^^^^^^

At steady state with constant γ̇, the Caputo derivative of a constant is zero:

.. math::

   0 = \frac{1-\lambda_{ss}}{\tau_{thix}} - \Gamma \lambda_{ss} |\dot{\gamma}|

**Equilibrium structure**:

.. math::

   \lambda_{ss}(\dot{\gamma}) = \frac{1}{1 + \Gamma \tau_{thix} |\dot{\gamma}|}

**Note**: The flow curve shape is identical to MIKH at constant temperature. Fractional
effects appear only in *transients*. Thermal coupling causes "thermal droop" at high
rates due to shear heating.

Start-up of Steady Shear
^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: γ̇(t) = γ̇₀·H(t), starting from λ(0) = 1 (aged sample)

**Key difference from MIKH**: Structure breakdown follows **Mittag-Leffler decay**:

.. math::

   \lambda(t) \approx E_\alpha\left(-\left(\frac{t}{\tau}\right)^\alpha\right)

Where E_α(z) is the Mittag-Leffler function, which behaves like:

- Short times: Stretched exponential exp(-(t/τ)^α)
- Long times: Power-law decay t^(-α)/Γ(1-α)

**Signatures**: Broader stress overshoot, slower approach to steady state ("long tail")
compared to exponential MIKH.

Stress Relaxation
^^^^^^^^^^^^^^^^^

**Protocol**: Step strain γ₀ at t = 0, then γ̇ = 0

For strains below yield, structure rebuilds according to:

.. math::

   D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}}

**Solution**:

.. math::

   \lambda(t) = 1 - (1-\lambda_0) E_\alpha\left(-\left(\frac{t}{\tau_{thix}}\right)^\alpha\right)

**Asymptotics**:

- Short time: λ(t) ≈ λ₀ + (1-λ₀)(t/τ)^α / Γ(1+α)
- Long time: λ(t) ≈ 1 - (1-λ₀)(τ/t)^α / Γ(1-α) (power-law approach)

The stress may **increase** during relaxation (anti-thixotropic recovery) as λ → 1
increases the modulus.

Creep (Step Stress)
^^^^^^^^^^^^^^^^^^^

**Protocol**: Constant stress σ₀ applied at t = 0

**Below fully-structured yield** (σ₀ < σ_y(λ=1, T)):
Only elastic deformation initially. Structure may slowly break due to
thermal fluctuations, leading to **delayed yielding**.

**Intermediate stress** (σ_y(λ=0) < σ₀ < σ_y(λ=1)):
**Viscosity bifurcation** — delay followed by avalanche-like yielding:

.. math::

   t_d \sim \tau_{thix} \left(\frac{\sigma_y(1) - \sigma_0}{\sigma_0}\right)^{1/\alpha}

**Above yield**: Immediate plastic flow with potential **thermal runaway** if
dissipation exceeds heat loss.

SAOS (Small Amplitude Oscillatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: γ(t) = γ₀·sin(ωt), γ₀ ≪ 1 (linear regime)

For fractional viscoelastic response at equilibrium λ_eq, the complex modulus follows
a **fractional Maxwell** form:

.. math::

   G^*(\omega) = G_0 \frac{(i\omega\tau)^\alpha}{1 + (i\omega\tau)^\alpha}

**Storage modulus**:

.. math::

   G'(\omega) = G_0 \frac{(\omega\tau)^{2\alpha} + (\omega\tau)^\alpha \cos(\pi\alpha/2)}{1 + 2(\omega\tau)^\alpha \cos(\pi\alpha/2) + (\omega\tau)^{2\alpha}}

**Loss modulus**:

.. math::

   G''(\omega) = G_0 \frac{(\omega\tau)^\alpha \sin(\pi\alpha/2)}{1 + 2(\omega\tau)^\alpha \cos(\pi\alpha/2) + (\omega\tau)^{2\alpha}}

**Cole-Cole signature**: Depressed arc with depression angle θ = (1-α)π/2.

LAOS (Large Amplitude Oscillatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: γ(t) = γ₀·sin(ωt), γ₀ finite (nonlinear, may yield)

Full coupled system requiring numerical integration. The fractional memory introduces:

- Power-law decay of higher harmonics
- Asymmetric Lissajous figures from back-stress
- Delayed yielding within cycle
- Cycle-by-cycle thermal softening at high amplitude/frequency


Quick Start
-----------

**Basic FIKH with thermal coupling:**

.. code-block:: python

   from rheojax.models.fikh import FIKH

   # Create model with fractional order α = 0.7
   model = FIKH(include_thermal=True, alpha_structure=0.7)

   # Set key parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 10.0)
   model.parameters.set_value("delta_sigma_y", 50.0)
   model.parameters.set_value("tau_thix", 100.0)

   # Fit to startup data
   model.fit(t, stress, test_mode='startup', strain=strain)

   # Predict flow curve
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

**Multi-mode FMLIKH:**

.. code-block:: python

   from rheojax.models.fikh import FMLIKH

   # Create 3-mode model with shared fractional order
   model = FMLIKH(n_modes=3, include_thermal=False, shared_alpha=True)

   # Set per-mode parameters
   for i, tau in enumerate([1.0, 10.0, 100.0], 1):
       model.parameters.set_value(f"tau_thix_{i}", tau)

   # Fit to oscillation data
   model.fit(omega, G_star, test_mode='oscillation')

**Comparing α effects:**

.. code-block:: python

   import numpy as np
   from rheojax.models.fikh import FIKH

   t = np.linspace(0, 100, 1000)

   # Compare different fractional orders
   for alpha in [0.3, 0.5, 0.7, 0.9]:
       model = FIKH(alpha_structure=alpha, include_thermal=False)
       model.parameters.set_value("tau_thix", 10.0)

       # Startup simulation
       result = model.predict_startup(t, gamma_dot=1.0, lambda_init=1.0)
       # α = 0.3: Slow decay, long memory
       # α = 0.9: Fast decay, nearly exponential


NLSQ to NUTS Fitting Workflow
-----------------------------

The recommended workflow uses NLSQ for fast point estimation followed by
NUTS for full Bayesian inference with uncertainty quantification.

Step 1: Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rheojax.core.data import RheoData

   # Startup data with correct mode
   data = RheoData(x=t, y=stress, test_mode='startup')
   data.add_metadata('strain', strain)
   data.add_metadata('temperature', T_env)

Step 2: Initial Point Estimation (NLSQ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   model = FIKH(include_thermal=True, alpha_structure=0.5)
   result = model.fit(data)
   print(f"R² = {result.r_squared:.4f}")

   # Check parameter estimates
   for name, param in model.parameters.items():
       print(f"{name}: {param.value:.4g}")

Step 3: Bayesian Inference (NUTS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # 4 chains for production diagnostics
   bayes_result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Check convergence
   summary = bayes_result.get_summary()
   print(f"Max R-hat: {summary['r_hat'].max():.3f}")  # Should be < 1.01
   print(f"Min ESS: {summary['ess'].min():.0f}")     # Should be > 400

Step 4: Model Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # ArviZ diagnostics
   model.plot_trace(bayes_result)
   model.plot_forest(bayes_result, hdi_prob=0.95)
   model.plot_pair(bayes_result, divergences=True)

   # Posterior predictive check
   posterior_pred = model.predict_from_posterior(
       bayes_result.posterior_samples,
       data.x,
       test_mode='startup'
   )

   # Credible intervals
   intervals = model.get_credible_intervals(
       bayes_result.posterior_samples,
       credibility=0.95
   )


Knowledge from FIKH Analysis
----------------------------

This section explains what physical insights can be extracted from FIKH model
parameters after fitting to experimental data.

What Parameters Reveal
^^^^^^^^^^^^^^^^^^^^^^

**Fractional Order α**:

- **α ≈ 0.9-1.0**: Material behaves like classical thixotropic fluid (single timescale)
- **α ≈ 0.5-0.7**: Broad distribution of restructuring timescales (typical for gels)
- **α < 0.3**: "Glassy" dynamics with very long memory (aging-dominated)

*Physical interpretation*: α relates to the breadth of the relaxation spectrum.
From Cole-Cole analysis, depression angle θ = (1-α)π/2. Materials with α ≈ 0.5
show 45° depression, indicating a very broad spectrum.

**Thixotropic Timescale τ_thix**:

- Sets the characteristic time for structure recovery at rest
- Larger τ_thix → slower rebuilding → more thixotropic
- Compare to process timescales: τ_thix >> t_process means structure won't recover
- Typical values: 1-1000 s for industrial fluids

**Breakdown Coefficient Γ**:

- Rate of structure destruction under flow
- Γ·τ_thix product determines equilibrium structure at given shear rate
- λ_eq = 1/(1 + Γ·τ_thix·|γ̇|)
- Critical shear rate: γ̇_crit = 1/(Γ·τ_thix) where λ drops to 0.5

**Yield Stress Parameters**:

- **σ_y0**: Residual yield when fully destructured (λ = 0)
- **Δσ_y**: Additional yield from structure (total yield at λ=1 is σ_y0 + Δσ_y)
- Ratio Δσ_y/σ_y0 indicates how much structure contributes to yielding

**Thermal Parameters**:

- **E_a**: Viscosity activation energy [J/mol]. Typical: 20-50 kJ/mol for polymers
- **E_y**: Yield activation energy. Lower E_y → yield is less temperature-sensitive
- Higher E_a/E_y ratio → viscosity drops faster than yield with temperature

Derived Quantities
^^^^^^^^^^^^^^^^^^

**From Flow Curve Fit**:

.. code-block:: python

   # Critical shear rate where λ drops to 0.5
   gamma_dot_crit = 1 / (Gamma * tau_thix)

   # Viscosity ratio (structured / broken)
   eta_ratio = (sigma_y0 + delta_sigma_y) / sigma_y0

   # Structure number (dimensionless thixotropic intensity)
   Str = Gamma * tau_thix * gamma_dot_applied

**From Creep Analysis**:

.. code-block:: python

   # Delay time for yielding at stress σ₀
   sigma_y_full = sigma_y0 + delta_sigma_y
   t_delay = tau_thix * ((sigma_y_full - sigma_0) / sigma_0)**(1/alpha)

**From Frequency Sweep**:

.. code-block:: python

   import numpy as np

   # Cole-Cole depression angle
   theta = (1 - alpha) * np.pi / 2  # radians

   # Crossover frequency (where G' = G'')
   omega_c = 1 / tau_maxwell  # where tau_maxwell = eta / G

**From Startup Overshoot**:

.. code-block:: python

   # Peak strain (approximate)
   gamma_peak = sigma_y_full / G

   # Overshoot ratio
   overshoot_ratio = sigma_peak / sigma_steady

Process Design Insights
^^^^^^^^^^^^^^^^^^^^^^^

**Pipeline Restart** (waxy crude):

- Fit FIKH to startup data at T_restart
- Compute restart pressure from steady-state stress at desired flow rate
- Account for thermal effects: higher h → less softening → higher restart pressure
- Use delay time formula to estimate how long the pipeline can sit before gelation

**Product Stability**:

- Compare τ_thix to shelf-life timescales
- If τ_thix >> shelf_time → product remains broken (poor stability)
- Recommendation: τ_thix < 0.1 × expected_storage_time for structure recovery
- Lower α (stronger memory) means slower recovery but more stable once recovered

**Processing Windows**:

- Critical shear rate γ̇_crit = 1/(Γ·τ_thix) defines transition from structured to flowing
- Design mixers/pumps to operate above γ̇_crit for easy flow
- Storage vessels should see γ̇ << γ̇_crit for structure recovery
- Temperature control: higher T → lower yield → easier processing but more energy

**Quality Control**:

- Monitor α over production batches: consistent α indicates consistent microstructure
- Track τ_thix: increasing τ_thix may indicate contamination or degradation
- Plot (α, τ_thix) phase space to identify batch-to-batch variability


Limiting Behavior
-----------------

**α → 1 (Classical Limit):**

- Recovers exponential MIKH behavior
- Mittag-Leffler E₁(-x) = exp(-x)
- Use standard MIKH for computational efficiency

**α → 0 (Strong Memory Limit):**

- Very slow power-law relaxation
- Long "memory" of deformation history
- Numerical challenges (need long history buffer)

**Thermal Coupling Effects:**

- High E_a: Strong temperature sensitivity
- Low h: Poor heat dissipation → thermal runaway risk
- χ ~ 0.9: Most plastic work converts to heat


Numerical Implementation
------------------------

The Caputo fractional derivative is discretized using the **L1 scheme**:

.. math::

   D_t^\alpha \lambda_n \approx \frac{1}{\Gamma(2-\alpha) \Delta t^\alpha} \sum_{k=0}^{n-1} b_k (\lambda_{n-k} - \lambda_{n-k-1})

Where b_k = (k+1)^(1-α) - k^(1-α).

Convergence and Accuracy
^^^^^^^^^^^^^^^^^^^^^^^^

**L1 Scheme Error Bounds**:

.. math::

   \|D^{\alpha}\lambda - D^{\alpha}_h\lambda\| \leq C \cdot h^{2-\alpha}

where h = Δt is the time step. Lower α requires finer time steps for same accuracy.

**Recommended n_history Selection**:

.. list-table::
   :widths: 20 25 20 35
   :header-rows: 1

   * - α Range
     - Recommended n_history
     - Memory Usage
     - Accuracy
   * - 0.7 - 0.99
     - 50-100
     - ~400 bytes
     - Good (fast convergence)
   * - 0.4 - 0.7
     - 100-500
     - ~4 KB
     - Good
   * - 0.05 - 0.4
     - 500-1000
     - ~8 KB
     - Adequate (strong memory)

**When to Increase n_history**:

- Long simulations (t >> τ_thix)
- Very small α (< 0.3)
- Accuracy-critical applications
- Oscillatory protocols (LAOS)

Computational Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Memory**: O(N) history storage via fixed-window buffer
- **Cost**: O(n_history) per step (vs O(N²) for naive full-history convolution)
- **JIT Compilation**: First call triggers ~1-5s compilation; subsequent calls are fast

**Precompilation** (optional):

.. code-block:: python

   # Trigger JIT compilation before production runs
   compile_time = model.precompile()
   print(f"Compiled in {compile_time:.1f}s")


Troubleshooting
---------------

**Poor fit quality (low R²)**:

- Check that test_mode matches your data type
- Verify strain/time arrays are correct
- Try different initial α values (0.3, 0.5, 0.7)
- Enable/disable thermal coupling based on experimental conditions

**MCMC convergence issues (R-hat > 1.1)**:

- Increase num_warmup (try 2000)
- Use NLSQ warm-start (critical for IKH models)
- Check for parameter correlations with plot_pair()
- Consider fixing poorly-identified parameters

**Numerical instabilities (NaN/Inf)**:

- Check for very small τ_thix (< 1e-6)
- Verify temperature stays physical (T > 0)
- Reduce dt or increase n_history for small α

**Slow computation**:

- Use model.precompile() to avoid JIT overhead
- Reduce n_history if α > 0.7
- Consider MIKH if α ≈ 1


References
----------

**Fractional Calculus:**

1. Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.

2. Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press.

3. Diethelm, K. (2010). *The Analysis of Fractional Differential Equations*. Springer.

**Fractional Rheology:**

4. Jaishankar, A. & McKinley, G.H. (2014). "A fractional K-BKZ constitutive formulation
   for describing the nonlinear rheology of multiscale complex fluids."
   *J. Rheol.*, 58, 1751-1788.

5. de Souza Mendes, P.R. & Thompson, R.L. (2019). "Time-dependent yield stress materials."
   *Curr. Opin. Colloid Interface Sci.*, 43, 15-25.

**IKH Foundation:**

For foundational IKH references (Dimitriou 2014, Geri 2017, Wei 2018), see :doc:`../ikh/index`.
