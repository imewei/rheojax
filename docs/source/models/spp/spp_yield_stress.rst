SPP Yield Stress Model
======================

.. module:: rheojax.models.spp.spp_yield_stress
   :synopsis: SPP-based yield stress model for LAOS amplitude sweeps

Quick Reference
---------------

- **Use when:** Extracting yield stress from LAOS amplitude sweeps, characterizing
cage-based yield stress fluids, connecting oscillatory to steady-shear behavior

- **Parameters:** 8 (G_cage, sigma_sy_scale, sigma_sy_exp, sigma_dy_scale, sigma_dy_exp, eta_inf, n_power_law, noise)

- **Key equation:** :math:`\sigma_{sy}(\gamma_0) = \sigma_{sy,0} \cdot |\gamma_0|^{n_{sy}}` and :math:`\sigma_{dy}(\gamma_0) = \sigma_{dy,0} \cdot |\gamma_0|^{n_{dy}}`

- **Test modes:** oscillation (LAOS amplitude sweep), rotation (flow curve)

- **Material examples:** Yield stress fluids, colloidal gels, concentrated emulsions, foams, soft glasses, carbopol microgels, mayonnaise

Notation Guide
--------------

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`G_{\text{cage}}`
     - Apparent cage modulus
     - Pa
   * - :math:`\sigma_{sy}(\gamma_0)`
     - Static yield stress
     - Pa
   * - :math:`\sigma_{dy}(\gamma_0)`
     - Dynamic yield stress
     - Pa
   * - :math:`\gamma_0`
     - Strain amplitude
     - —
   * - :math:`n_{sy}`
     - Static yield exponent
     - —
   * - :math:`n_{dy}`
     - Dynamic yield exponent
     - —
   * - :math:`\eta_\infty`
     - Infinite-shear viscosity
     - Pa·s
   * - :math:`n`
     - Power-law flow index
     - —
   * - :math:`\dot{\gamma}`
     - Shear rate
     - 1/s

Quick Reference (continued)
---------------------------

.. list-table::
   :widths: 25 75
   :stub-columns: 1

   * - **Model Class**
     - :class:`~rheojax.models.spp.spp_yield_stress.SPPYieldStress`
   * - **Registry Name**
     - ``"spp_yield_stress"``
   * - **Test Modes**
     - ``oscillation`` (amplitude sweep), ``rotation`` (flow curve)
   * - **Parameters**
     - 8 (G_cage, sigma_sy_scale, sigma_sy_exp, sigma_dy_scale, sigma_dy_exp, eta_inf, n_power_law, noise)
   * - **Typical Materials**
     - Yield stress fluids, colloidal gels, emulsions, foams, soft glasses
   * - **Key Reference**
     - Rogers et al. (2012) J. Rheol. 56(1)

Overview
--------

The SPP Yield Stress model extracts physically meaningful yield parameters from
Large Amplitude Oscillatory Shear (LAOS) data using the Sequence of Physical
Processes (SPP) framework. Unlike traditional Fourier-based approaches, SPP
provides time-resolved instantaneous material functions that reveal the
intracycle sequence of physical processes during nonlinear deformation.

The model parameterizes the nonlinear response in terms of:

- **G_cage**: Apparent cage modulus (elastic stiffness of the microstructural cage)
- **Static yield stress** (σ_sy): Stress at strain reversal (maximum strain amplitude)
- **Dynamic yield stress** (σ_dy): Stress at rate reversal (zero strain rate)
- **Power-law scaling**: Amplitude dependence of yield stresses with exponent

This approach connects LAOS analysis to steady-shear flow behavior, enabling
comprehensive yield stress characterization across deformation protocols.

Physical Foundations
--------------------

Cage Model for Yield Stress Fluids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP yield stress model is grounded in the colloidal cage picture, where
particles are confined by nearest-neighbor "cages":

1. **Linear Regime** (small γ_0): The cage deforms elastically with stiffness G_cage
2. **Yielding** (γ_0 → γ_yield): Cage constraints are overcome at the yield point
3. **Flow Regime** (large γ_0): Particles escape cages and flow viscously

The cage modulus G_cage represents the instantaneous elastic stiffness measured
at the point where stress passes through zero (σ = 0). This corresponds to the
slope of the stress-strain Lissajous curve at the origin.

Static vs. Dynamic Yield Stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP framework distinguishes two physically distinct yield stresses:

**Static Yield Stress (σ_sy)**
   - Measured at strain reversal (γ = ±γ_0, γ̇ = 0)
   - Represents the stress required to initiate flow from rest
   - Larger than dynamic yield due to microstructural recovery during strain reversal

**Dynamic Yield Stress (σ_dy)**
   - Measured at rate reversal (γ̇ = 0, γ ≠ 0)
   - Represents the stress during continuous flow
   - Connects to steady-shear yield stress extrapolation

The ratio σ_sy/σ_dy is typically around 2-3 for colloidal systems and reveals
information about cage reformation kinetics and thixotropy.

Power-Law Amplitude Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both yield stresses exhibit power-law scaling with strain amplitude:

.. math::

   \sigma_{sy}(\gamma_0) = \sigma_{sy,0} \cdot \gamma_0^{n_{sy}}

.. math::

   \sigma_{dy}(\gamma_0) = \sigma_{dy,0} \cdot \gamma_0^{n_{dy}}

where the exponents n_sy and n_dy typically fall in the range 0.2-1.0.
Rogers et al. (2011) found n ≈ 0.2 for concentrated colloidal suspensions,
connecting to the Herschel-Bulkley flow curve exponent.

Connection to SPP Framework (Rogers 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The yield stress terms in this model connect to the complete SPP stress
decomposition from Rogers (2017):

.. math::

   \sigma(t) = G'_t(t)[\gamma(t) - \gamma_{eq}(t)] + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_y(t)

The **displacement term** :math:`\sigma_y(t) - G'_t(t)\gamma_{eq}(t)` captures:

1. **Static yield stress**: Associated with the shift in equilibrium strain
   :math:`\gamma_{eq}` during cage rupture

2. **Dynamic yield stress**: The zero-rate stress intercept from the
   viscoplastic flow contribution

**Equilibrium Strain Interpretation**:

- In the linear regime, :math:`\gamma_{eq} = 0` (no shifting)
- During yielding, :math:`\gamma_{eq}` shifts to :math:`\pm(\gamma_0 - \gamma_y)`
- The material frame strain becomes :math:`\gamma_{mat}(t) = \gamma(t) - \gamma_{eq}(t)`
- This distinction between lab frame and material frame is essential for
  understanding the physical origin of the two yield stresses

Connection to Steady-Shear Flow Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The dynamic yield stress from SPP analysis connects directly to steady-shear
Herschel-Bulkley fitting, enabling cross-validation between oscillatory and
rotational measurements:

**Cross-Validation Protocol:**

1. Extract :math:`\sigma_{dy}` from LAOS amplitude sweep using SPP decomposition
2. Fit steady-shear flow curve: :math:`\sigma = \sigma_y + K\dot{\gamma}^n`
3. Compare: :math:`\sigma_{dy,\text{LAOS}}` should match :math:`\sigma_y` from steady shear

**Expected Agreement:**

- Good agreement indicates simple yield stress behavior (Bingham-like)
- Discrepancy suggests thixotropy, structural evolution, or measurement artifacts
- Large disagreement may indicate shear banding, wall slip, or elastic recoil effects

**Rate-Amplitude Duality:**

Rogers et al. (2011) demonstrated that yield stresses scale identically with
strain amplitude and shear rate:

.. math::

   \sigma_y(\gamma_0) \propto \gamma_0^{n} \quad \text{(amplitude)}

.. math::

   \sigma_y(\dot{\gamma}_0) \propto \dot{\gamma}_0^{n} \quad \text{(rate)}

This universal exponent (:math:`n \approx 0.2` for hard-sphere colloids) connects LAOS
and steady-shear characterization through a single material parameter.

**Practical Cross-Validation Example:**

.. code-block:: python

   from rheojax.models import SPPYieldStress, HerschelBulkley

   # Fit LAOS amplitude sweep
   spp_model = SPPYieldStress()
   spp_model.fit(gamma_0, sigma_dy_laos, test_mode='oscillation', yield_type='dynamic')
   sigma_dy_spp = spp_model.parameters.get_value('sigma_dy_scale')

   # Fit steady-shear flow curve
   hb_model = HerschelBulkley()
   hb_model.fit(gamma_dot, sigma_steady, test_mode='rotation')
   sigma_y_hb = hb_model.parameters.get_value('sigma_y')

   # Cross-validate
   relative_error = abs(sigma_dy_spp - sigma_y_hb) / sigma_y_hb
   print(f"Yield stress agreement: {100 * (1 - relative_error):.1f}%")
   # Typical agreement: >90% for simple yield stress fluids

Theoretical Benchmarks (Rogers & Lettinga 2012)
-----------------------------------------------

Rogers & Lettinga (2012) [2]_ applied the SPP framework to classical nonlinear
viscoelastic models, establishing theoretical benchmarks against which experimental
data can be validated. These predictions serve as reference behaviors for
interpreting yield stress material responses.

Bingham Model Response
~~~~~~~~~~~~~~~~~~~~~~

The Bingham fluid :math:`\sigma = \sigma_y + \eta_p \dot{\gamma}` (for :math:`|\sigma| > \sigma_y`)
represents the idealized yield stress fluid. Under LAOS, the SPP framework reveals
a characteristic four-step intracycle sequence:

**Step 1: Elastic Extension** (|γ| < γ_y)
   - Stress builds linearly: :math:`\sigma = G_{cage} \cdot \gamma`
   - :math:`G'_t \approx G_{cage}` (constant), :math:`G''_t \approx 0`
   - Cole-Cole trajectory: remains near the :math:`G'_t` axis

**Step 2: Static Yielding** (at γ = ±γ_y)
   - Cages rupture; stress reaches :math:`\sigma_{y,static} = G_{cage} \cdot \gamma_y`
   - Sharp transition: :math:`G'_t` drops rapidly, :math:`G''_t` increases
   - Cole-Cole: trajectory moves from elastic toward viscous dominance

**Step 3: Newtonian Flow** (|γ| > γ_y, |γ̇| large)
   - Material flows with constant viscosity :math:`\eta_p`
   - :math:`G'_t \approx 0`, :math:`G''_t \approx \eta_p \omega`
   - Cole-Cole: trajectory resides near the :math:`G''_t` axis

**Step 4: Cage Reformation** (as γ̇ → 0 at ±γ_max)
   - Microstructure rapidly reforms at rate reversal
   - :math:`G''_t \to 0`, :math:`G'_t` increases
   - :math:`\sigma_{y,dynamic}` measured at this instant
   - Cole-Cole: trajectory returns toward elastic dominance

**Key Bingham Predictions:**
   - The equilibrium strain :math:`\gamma_{eq}` shifts to :math:`\pm(\gamma_0 - \gamma_y)` after yielding
   - Static yield stress :math:`\sigma_{y,static}` exceeds dynamic :math:`\sigma_{y,dynamic}` due to reformation
   - Post-yield flow follows the steady-state Bingham flow curve exactly
   - Higher harmonics in stress arise solely from the yield transition (not from viscosity)

Giesekus Model Response
~~~~~~~~~~~~~~~~~~~~~~~

The Giesekus model, which includes a nonlinear relaxation term with mobility
parameter :math:`\alpha`, exhibits elastic recoil during flow reversal. Under SPP analysis:

**Elastic Recoil Signatures:**
   - Unlike ideal Bingham, :math:`G'_t` does not drop to zero during flow
   - The material retains elastic character even post-yield due to polymer stretch
   - Stress overshoots during startup are more pronounced than Bingham

**SPP Trajectory Features:**
   - Cole-Cole plots form loops rather than simple back-and-forth trajectories
   - Loop area increases with Deborah number (De = λω)
   - Trajectory directionality (clockwise vs counterclockwise) encodes stress response phase

**Mobility Parameter Effects (α):**
   - :math:`\alpha = 0` (upper-convected Maxwell): purely elastic, no shear thinning
   - :math:`\alpha = 0.5`: intermediate behavior, moderate shear thinning
   - :math:`\alpha \to 1`: stronger nonlinearity, more pronounced recoil features

**Validation Use Cases:**
   - If experimental data shows Giesekus-like loops but no clear yield, the material
     may be viscoelastic rather than yield-stress
   - Bingham-like step transitions indicate simpler yield stress behavior
   - Hybrid behaviors suggest complex microstructure (e.g., polymer-in-suspension)

Power-Law Fluid (Non-Yield)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For comparison, a purely viscous power-law fluid :math:`\sigma = K |\dot{\gamma}|^n \text{sgn}(\dot{\gamma})`
shows characteristic SPP behavior that lacks the yield-stress sequence:

**SPP Predictions:**
   - :math:`G'_t = 0` everywhere (no elasticity)
   - :math:`G''_t` varies with shear rate as :math:`\sim |\dot{\gamma}|^{n-1}`
   - No cage modulus can be defined (dσ/dγ|_{σ=0} is undefined)
   - Cole-Cole trajectory collapses to the :math:`G''_t` axis

**Diagnostic Implication:**
   If experimental SPP shows negligible :math:`G'_t` values, the material may not be
   a yield stress fluid but rather a shear-thinning viscous material.

Validating Experimental Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use these benchmarks to assess experimental SPP results:

.. list-table:: Theoretical Benchmark Comparison
   :header-rows: 1
   :widths: 25 25 50

   * - Observation
     - Matches
     - Physical Interpretation
   * - Sharp 4-step transitions, :math:`G'_t \to 0` post-yield
     - Bingham
     - Simple yield stress fluid, brittle cages
   * - Loops in Cole-Cole, non-zero :math:`G'_t` post-yield
     - Giesekus-like
     - Viscoelastic yield stress, elastic recoil
   * - :math:`G'_t \approx 0` throughout cycle
     - Power-law
     - Viscous fluid, no yield stress
   * - Partial matches
     - Hybrid
     - Complex microstructure; consider thixotropy or heterogeneity

----

Constitutive Equations
----------------------

Oscillation Mode (LAOS Amplitude Sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For amplitude sweep analysis with strain amplitude γ_0:

.. math::

   \sigma_{sy}(\gamma_0) = \text{sigma\_sy\_scale} \cdot |\gamma_0|^{\text{sigma\_sy\_exp}}

.. math::

   \sigma_{dy}(\gamma_0) = \text{sigma\_dy\_scale} \cdot |\gamma_0|^{\text{sigma\_dy\_exp}}

The cage modulus at small amplitude approximates:

.. math::

   G_{cage} \approx \frac{\sigma_{sy}(\gamma_0)}{\gamma_0} \quad (\gamma_0 \to 0)

Rotation Mode (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For steady shear flow at rate γ̇, the model uses a Herschel-Bulkley-like form:

.. math::

   \sigma(\dot{\gamma}) = \sigma_{dy} + \eta_\infty |\dot{\gamma}|^n

This connects the dynamic yield stress from LAOS to steady-shear flow curves,
enabling validation between oscillatory and continuous shear measurements.

----

Governing Equations
-------------------

Amplitude-Dependent Power-Law Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core governing equations describe how yield stresses scale with strain amplitude:

**Static Yield Stress:**

.. math::

   \sigma_{sy}(\gamma_0) = \sigma_{sy,0} \cdot |\gamma_0|^{n_{sy}}

**Dynamic Yield Stress:**

.. math::

   \sigma_{dy}(\gamma_0) = \sigma_{dy,0} \cdot |\gamma_0|^{n_{dy}}

where:
   - :math:`\sigma_{sy,0}, \sigma_{dy,0}` are scale factors (Pa)
   - :math:`n_{sy}, n_{dy}` are power-law exponents (typically 0.2-1.0)

**Physical interpretation**: The power-law scaling reflects the rate-dependence of cage rupture and reformation. The exponent :math:`n \approx 0.2` observed in colloidal systems (Rogers et al., 2011) connects to the Herschel-Bulkley flow curve exponent.

Cage Modulus Definition
~~~~~~~~~~~~~~~~~~~~~~~~

The cage modulus represents the instantaneous elastic stiffness at zero stress:

.. math::

   G_{\text{cage}} = \left. \frac{d\sigma}{d\gamma} \right|_{\sigma=0}

In the small-amplitude limit:

.. math::

   \lim_{\gamma_0 \to 0} G_{\text{cage}} = \lim_{\gamma_0 \to 0} \frac{\sigma_{sy}(\gamma_0)}{\gamma_0}

This connects the cage modulus to the static yield stress scaling.

Steady-Shear Flow Curve (Rotation Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For steady shear at rate :math:`\dot{\gamma}`, the model uses a Herschel-Bulkley-like form:

.. math::

   \sigma(\dot{\gamma}) = \sigma_{dy} + \eta_\infty |\dot{\gamma}|^n

where:
   - :math:`\sigma_{dy}` is the dynamic yield stress extrapolated to zero amplitude
   - :math:`\eta_\infty` is the high-shear viscosity
   - :math:`n` is the power-law flow index (:math:`n < 1` for shear-thinning)

This form enables cross-validation between LAOS (amplitude sweeps) and steady-shear (rotation) measurements.

----

What You Can Learn
------------------

Physical Insights from SPP Yield Stress Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cage Modulus (G_cage) — Microstructural Stiffness:**

The cage modulus quantifies the elastic strength of the confining microstructural cage:

**Key insights**:
   - **Amplitude-independent**: Remains constant even as :math:`G'(\gamma_0)` decreases at large amplitudes
   - **True material property**: Reflects intrinsic nearest-neighbor confinement
   - **Connection to SAOS**: :math:`G_{\text{cage}} \approx G'_{\text{LVR}}` in the linear viscoelastic regime
   - **Scaling with concentration**: For colloidal systems, :math:`G_{\text{cage}} \sim \phi^{3-5}` where :math:`\phi` is volume fraction

**Typical values**:
   - Colloidal glasses (φ ~ 0.6): 10-100 Pa
   - Microgel pastes: 100-1000 Pa
   - Polymer gels: 1000-10000 Pa

**Static Yield Stress Scaling — Cage Rupture Dynamics:**

The power-law exponent :math:`n_{sy}` reveals information about the cage rupture process:

**Exponent interpretation**:
   - :math:`n_{sy} \approx 0.2`: Weak amplitude dependence (typical for hard-sphere colloids)
   - :math:`n_{sy} \approx 0.5`: Moderate amplitude dependence (soft particles)
   - :math:`n_{sy} \approx 1.0`: Strong amplitude dependence (polymer gels, weak cages)

**Physical meaning**: Lower exponents indicate cages that rupture at a nearly constant stress regardless of amplitude (brittle rupture). Higher exponents indicate cages that become progressively easier to break at larger amplitudes (ductile rupture).

**Dynamic Yield Stress Scaling — Flow Regime:**

The dynamic yield stress connects LAOS to steady-shear flow behavior:

**Key insights**:
   - **Ratio to static**: :math:`\sigma_{sy}/\sigma_{dy} \approx 2-3` for thixotropic materials
   - **Flow curve connection**: :math:`\sigma_{dy}` matches the extrapolated Herschel-Bulkley yield stress
   - **Rate sensitivity**: :math:`n_{dy} \approx n_{sy}` (same exponent) indicates universal scaling

**Thixotropic signatures**:
   - Large :math:`\sigma_{sy}/\sigma_{dy}` ratio → strong thixotropy (fast cage reformation)
   - Small ratio → weak thixotropy (slow cage reformation)

**Amplitude Scaling Exponent — Universal Behavior:**

Rogers et al. (2011) found :math:`n \approx 0.2` for concentrated colloidal suspensions. This value emerges from the cage model and connects to:

1. **Herschel-Bulkley exponent**: Same power-law appears in steady-shear flow curves
2. **Stress-rate duality**: Both :math:`\sigma_{sy}(\gamma_0)` and :math:`\sigma_{sy}(\dot{\gamma}_0)` scale with the same exponent
3. **Jamming universality**: Near the jamming transition, :math:`n \approx 0.2` is predicted theoretically

Material Characterization Capabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**From LAOS Amplitude Sweeps:**
   - Cage modulus :math:`G_{\text{cage}}` (constant across amplitudes)
   - Static yield stress scaling: :math:`\sigma_{sy,0}, n_{sy}`
   - Dynamic yield stress scaling: :math:`\sigma_{dy,0}, n_{dy}`
   - Thixotropic strength (from :math:`\sigma_{sy}/\sigma_{dy}` ratio)
   - Nonlinearity onset (strain amplitude where power-law emerges)

**From Steady-Shear Flow Curves:**
   - Dynamic yield stress :math:`\sigma_{dy}` (Herschel-Bulkley intercept)
   - High-shear viscosity :math:`\eta_\infty`
   - Flow power-law index :math:`n`
   - Cross-validation with LAOS-derived :math:`\sigma_{dy}`

**From Model Fitting:**
   - Cage rupture mechanism (from :math:`n_{sy}`)
   - Flow regime behavior (from :math:`n`)
   - Structural memory timescales (from yield stress ratios)
   - Quality control metrics (batch-to-batch consistency)

**Comparison Across Protocols:**
   - LAOS vs steady-shear yield stress agreement
   - Cox-Merz rule validation (if applicable)
   - Amplitude sweep vs frequency sweep consistency

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**σ_y (Yield Stress)**:
   The stress threshold required to initiate flow from rest (static) or during continuous flow (dynamic).

   *For graduate students*: The SPP framework distinguishes two yield stresses via the intracycle stress decomposition σ(t) = G'_t[γ - γ_eq] + (G''_t/ω)γ̇ + σ_y. Static yield stress σ_sy (at strain reversal, γ̇ = 0) includes microstructural recovery during momentary rest, while dynamic yield stress σ_dy (at rate reversal) represents flowing state. Power-law scaling σ_y ∼ γ_0^n with n ≈ 0.2 emerges from jamming universality (Liu & Nagel 2010). Ratio σ_sy/σ_dy ≈ 2-3 quantifies thixotropic cage reformation timescales.

   *For practitioners*: Fit σ_sy from LAOS amplitude sweep (stress at maximum strain). Fit σ_dy from flow curve extrapolation to γ̇ → 0 or from LAOS rate reversal. Ratio > 3 indicates strong thixotropy, requiring longer rest between measurements. Exponent n < 0.3 (hard particles) vs n > 0.5 (soft particles) guides formulation strategy.

**G_cage (Cage Modulus)**:
   Instantaneous elastic stiffness of the confining microstructural cage.

   *For graduate students*: G_cage = (dσ/dγ)|_σ=0 represents the cage spring constant in colloidal glass picture. Unlike G'(ω), which decreases with amplitude in nonlinear regime, G_cage remains constant (amplitude-independent material property). Scales as G_cage ∼ nk_BT/a³ where n is number density and a is particle radius, connecting to thermal energy and cage size.

   *For practitioners*: Extract from linear regime G'_LVR or from SPP G'_t at σ = 0 crossing. Typical values: 10-100 Pa (colloidal glasses), 100-1000 Pa (microgels), >1000 Pa (polymer gels). Use for quality control and batch consistency checks.

**n_sy, n_dy (Power-Law Exponents)**:
   Amplitude-dependence exponents quantifying cage rupture mechanism.

   *For graduate students*: Scaling exponents in σ_y(γ_0) ∼ γ_0^n connect to Herschel-Bulkley flow curve exponent via stress-rate duality. Rogers (2011) universal value n ≈ 0.2 for hard-sphere colloids derives from percolation/jamming scaling laws. Deviation from 0.2 indicates additional mechanisms (attractive forces, particle softness, structural hierarchy).

   *For practitioners*: n ≈ 0.2 confirms hard-sphere-like behavior. n > 0.5 suggests soft particles or weak cages. n ≈ 1.0 indicates linear dependence (fragile gels). Use to classify material type and predict performance across amplitude ranges.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SPP Yield Stress Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - G_cage = 10-100 Pa, n_sy ≈ 0.2
     - Hard-sphere colloidal glass
     - Concentrated silica, PMMA colloids (φ ~ 0.6)
     - Universal jamming behavior, brittle cages
   * - G_cage = 100-1000 Pa, n_sy ≈ 0.5
     - Soft particle suspensions
     - Microgels, emulsions, soft colloids
     - Moderate amplitude-dependence, ductile cages
   * - G_cage > 1000 Pa, n_sy ≈ 1.0
     - Polymer gels with weak cages
     - Carbopol, weak hydrogels
     - Strong amplitude-dependence, easy cage breaking
   * - σ_sy/σ_dy < 2
     - Weak thixotropy
     - Simple yield-stress fluids
     - Minimal cage reformation during rest
   * - σ_sy/σ_dy = 2-3
     - Moderate thixotropy (typical)
     - Concentrated suspensions, soft glasses
     - Standard Rogers et al. (2011) range
   * - σ_sy/σ_dy > 3
     - Strong thixotropy
     - Highly thixotropic pastes, aged gels
     - Fast cage reformation, strong memory
   * - n_sy ≈ n_dy ≈ 0.2
     - Universal jamming signature
     - Hard-sphere colloids near φ_c
     - Theoretical prediction confirmed
   * - Power-law regime: σ ~ γ_0^n
     - Cage rupture scaling
     - All yield-stress materials in LAOS
     - Amplitude-dependent yield, nonlinear LAOS

----

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 18 12 15 55

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_cage``
     - Pa
     - [1e-3, 1e9]
     - Apparent cage modulus; elastic stiffness at σ = 0
   * - ``sigma_sy_scale``
     - Pa
     - [1e-6, 1e9]
     - Static yield stress scale factor
   * - ``sigma_sy_exp``
     - —
     - [0.0, 2.0]
     - Static yield stress amplitude exponent
   * - ``sigma_dy_scale``
     - Pa
     - [1e-6, 1e9]
     - Dynamic yield stress scale factor
   * - ``sigma_dy_exp``
     - —
     - [0.0, 2.0]
     - Dynamic yield stress amplitude exponent
   * - ``eta_inf``
     - Pa·s
     - [1e-9, 1e6]
     - Infinite-shear viscosity (for flow curve)
   * - ``n_power_law``
     - —
     - [0.01, 2.0]
     - Flow power-law index
   * - ``noise``
     - Pa
     - [1e-10, 1e6]
     - Observation noise scale (Bayesian inference)

----

Fitting Guidance
----------------

(Fitting guidance content already present in the file)

----

Usage
-----

(Usage examples already present in the file)

----

Validity and Assumptions
-------------------------

Applicability
~~~~~~~~~~~~~

The SPP yield stress model is most appropriate for:

- **Yield stress fluids**: Materials with clear solid-to-liquid transition
- **Soft glassy materials**: Colloidal gels, emulsions, foams, pastes
- **LAOS amplitude sweeps**: Progressive nonlinearity from linear to flowing
- **Concentrated systems**: Volume fractions near or above jamming

Assumptions
~~~~~~~~~~~

1. **Single characteristic yield process**: The model assumes a dominant yielding
   mechanism described by power-law scaling

2. **Cage-based microstructure**: Physical interpretation requires particle-based
   or droplet-based microstructure with cage confinement

3. **Time-strain separability**: Assumes steady-state oscillatory response
   without significant transient evolution during measurement

4. **Sufficient harmonics**: SPP extraction requires adequate harmonic content
   (typically n_harmonics ≥ 15)

Limitations
~~~~~~~~~~~

- **Simple materials**: Newtonian fluids show no amplitude dependence
- **Polymer solutions**: May require different physical interpretation
- **Extreme amplitude**: Very large γ_0 may show non-power-law behavior
- **Transient effects**: Not suitable for strongly time-dependent (aging) materials
- **Low frequency**: β-relaxation must be slow compared to measurement

Usage Examples
--------------

Basic NLSQ Fitting (Amplitude Sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SPPYieldStress

   # Amplitude sweep data: yield stresses vs. strain amplitude
   gamma_0 = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
   sigma_sy = np.array([5.0, 8.5, 18.0, 32.0, 55.0, 110.0, 195.0, 340.0])

   # Initialize and fit
   model = SPPYieldStress()
   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   # View fitted parameters
   print(model)
   # SPPYieldStress(G_cage=5.00e+02, σ_sy=1.00e+02, σ_dy=5.00e+01, n=0.50)

   # Predict at new amplitudes
   gamma_0_pred = np.logspace(-2, 1, 50)
   sigma_pred = model.predict(gamma_0_pred)

Dynamic Yield Stress Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Dynamic yield stress (at rate reversal) typically lower than static
   sigma_dy = np.array([2.0, 3.5, 7.5, 13.0, 22.0, 45.0, 80.0, 140.0])

   model = SPPYieldStress()
   model.fit(gamma_0, sigma_dy, test_mode='oscillation', yield_type='dynamic')

   # Get both yield stresses across amplitude range
   sweep_results = model.predict_amplitude_sweep(
       gamma_0_pred,
       yield_type='both'
   )
   print(f"Static yield stress: {sweep_results['sigma_sy']}")
   print(f"Dynamic yield stress: {sweep_results['sigma_dy']}")

Flow Curve Fitting (Rotation Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Steady shear data
   gamma_dot = np.logspace(-3, 2, 20)
   sigma = 25.0 + 3.5 * gamma_dot**0.45  # Herschel-Bulkley-like

   model = SPPYieldStress()
   model.fit(gamma_dot, sigma, test_mode='rotation')

   # Predict flow curve
   sigma_pred = model.predict(gamma_dot)
   print(f"Yield stress: {model.parameters.get_value('sigma_dy_scale'):.2f} Pa")
   print(f"Power-law index: {model.parameters.get_value('n_power_law'):.2f}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SPPYieldStress
   import numpy as np

   # Example data: amplitude sweep
   gamma_0 = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
   sigma_sy = np.array([5.0, 8.5, 18.0, 32.0, 55.0, 110.0, 195.0, 340.0])

   # Initialize and NLSQ fit for warm-start
   model = SPPYieldStress()
   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   # Bayesian inference with warm-start
   result = model.fit_bayesian(
       gamma_0,
       sigma_sy,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000
   )

   # Posterior summary
   print(result.summary)

   # Credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )
   for param, (low, high) in intervals.items():
       print(f"{param}: [{low:.3f}, {high:.3f}]")

Integration with SPPDecomposer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   from rheojax.models import SPPYieldStress
   from rheojax.core.data import RheoData
   import numpy as np

   # Generate example LAOS waveforms at different amplitudes
   omega = 1.0  # rad/s
   n_cycles = 5
   n_points = 200
   t = np.linspace(0, 2 * np.pi * n_cycles / omega, n_points)

   amplitude_sweep_data = {}
   for gamma_0 in [0.01, 0.1, 1.0]:
       strain = gamma_0 * np.sin(omega * t)
       stress = 100.0 * strain + 20.0 * gamma_0**0.3 * np.sign(np.cos(omega * t))
       amplitude_sweep_data[gamma_0] = RheoData(t=t, stress=stress, strain=strain)

   # Process LAOS waveforms at multiple amplitudes
   decomposer = SPPDecomposer(n_harmonics=39, step_size=8)

   # Collect yield stresses from each amplitude
   amplitudes = []
   static_yields = []
   dynamic_yields = []

   for gamma_0, data in amplitude_sweep_data.items():
       result = decomposer.transform(data)
       amplitudes.append(gamma_0)
       static_yields.append(np.abs(result['stress_at_strain_max']))
       dynamic_yields.append(np.abs(result['stress_at_rate_reversal']))

   # Fit yield stress model
   model = SPPYieldStress()

   # Static yield stress
   model.fit(np.array(amplitudes), np.array(static_yields),
             test_mode='oscillation', yield_type='static')

   print(f"Static yield exponent: {model.parameters.get_value('sigma_sy_exp'):.3f}")

Troubleshooting
---------------

Poor Power-Law Fit
~~~~~~~~~~~~~~~~~~

**Symptoms**: Large fitting residuals, unreasonable exponents

**Solutions**:

1. Check data quality at low amplitudes (may be noisy near linear regime)
2. Verify sufficient amplitude range (at least 1-2 decades)
3. Look for regime transitions (different slopes at different amplitudes)

.. code-block:: python

   # Visualize power-law fit
   import matplotlib.pyplot as plt

   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   fig, ax = plt.subplots()
   ax.loglog(gamma_0, sigma_sy, 'o', label='Data')
   ax.loglog(gamma_0, model.predict(gamma_0), '-', label='Fit')
   ax.set_xlabel(r'$\gamma_0$')
   ax.set_ylabel(r'$\sigma_{sy}$ (Pa)')
   ax.legend()
   plt.show()

Cage Modulus Issues
~~~~~~~~~~~~~~~~~~~

**Symptoms**: G_cage unreasonably large or small

**Causes**:

- Insufficient low-amplitude data points
- Material not exhibiting clear cage behavior
- Noise dominating linear regime

**Solutions**:

1. Ensure adequate data in linear regime (γ_0 < 0.1)
2. Verify linear viscoelastic moduli are consistent
3. Consider if cage model is appropriate for the material

Bayesian Convergence
~~~~~~~~~~~~~~~~~~~~

**Symptoms**: R-hat > 1.1, low ESS, divergent transitions

**Solutions**:

1. **Always NLSQ warm-start**: Critical for stable sampling

   .. code-block:: python

      # NLSQ first, then Bayesian
      model.fit(gamma_0, sigma_sy, test_mode='oscillation')
      result = model.fit_bayesian(gamma_0, sigma_sy, ...)

2. **Increase samples**: ``num_warmup=2000, num_samples=4000``

3. **Check priors**: Ensure physically reasonable bounds

4. **Inspect trace plots**: Look for mixing issues

   .. code-block:: python

      import arviz as az
      az.plot_trace(result.arviz_data)

Prior Sensitivity
~~~~~~~~~~~~~~~~~

The model uses physically-motivated priors:

- **LogNormal** for scale parameters (G_cage, stress scales, viscosity)
- **Beta** for bounded exponents [0, 2]
- **HalfCauchy** for noise scale

If priors are too informative:

.. code-block:: python

   # Check prior impact with prior predictive sampling
   from rheojax.visualization import plot_bayesian_diagnostics

   # Compare posterior to prior
   plot_bayesian_diagnostics(result, diagnostics=['pair', 'forest'])

Related Models
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Use Case
   * - :doc:`spp_decomposer`
     - Extract instantaneous moduli from individual LAOS waveforms
   * - :doc:`../flow/herschel_bulkley`
     - Steady-shear yield stress (Herschel-Bulkley model)
   * - :doc:`../sgr/sgr_conventional`
     - Soft glassy rheology for aging yield stress fluids
   * - :doc:`../flow/bingham`
     - Simple yield stress + Newtonian flow

References
----------

.. [1] Rogers, S. A., Erwin, B. M., Vlassopoulos, D., and Cloitre, M. "A sequence
   of physical processes determined and quantified in LAOS: Application to a
   yield stress fluid." *Journal of Rheology*, 55, 435-458 (2011).
   https://doi.org/10.1122/1.3544591

.. [2] Rogers, S. A. "A sequence of physical processes determined and quantified
   in large-amplitude oscillatory shear (LAOS): Application to theoretical
   nonlinear models." *Journal of Rheology*, 56(1), 1-25 (2012).
   https://doi.org/10.1122/1.3662962

.. [3] Rogers, S. A. "In search of physical meaning: Defining transient parameters
   for nonlinear viscoelasticity." *Rheologica Acta*, 56, 501-525 (2017).
   https://doi.org/10.1007/s00397-017-1008-1

.. [4] Ewoldt, R. H., Hosoi, A. E., and McKinley, G. H. "New measures for
   characterizing nonlinear viscoelasticity in large amplitude oscillatory shear."
   *Journal of Rheology*, 52, 1427-1458 (2008).
   https://doi.org/10.1122/1.2970095

.. [5] Hyun, K., et al. "A review of nonlinear oscillatory shear tests: Analysis
   and application of large amplitude oscillatory shear (LAOS)."
   *Progress in Polymer Science*, 36, 1697-1753 (2011).
   https://doi.org/10.1016/j.progpolymsci.2011.02.002

.. [6] Bonn, D., Denn, M. M., Berthier, L., Divoux, T., and Manneville, S.
   "Yield stress materials in soft condensed matter." *Reviews of Modern Physics*,
   89, 035005 (2017). https://doi.org/10.1103/RevModPhys.89.035005

.. [7] Donley, G. J., et al. "Elucidating the G'' overshoot in soft materials with
   a yield transition via a time-resolved experimental strain decomposition."
   *Proceedings of the National Academy of Sciences*, 117, 21945-21952 (2020).
   https://doi.org/10.1073/pnas.2003869117

.. [8] Liu, A. J. and Nagel, S. R. "The jamming transition and the marginally jammed
   solid." *Annual Review of Condensed Matter Physics*, 1, 347-369 (2010).
   https://doi.org/10.1146/annurev-conmatphys-070909-104045

.. [9] Kim, J., et al. "Visualization of time-dependent intracycle molecular orientation
   dynamics in LAOS from anisotropic SANS." *Journal of Rheology*, 64, 291-303 (2020).
   https://doi.org/10.1122/1.5127529

.. [10] Saengow, C., Giacomin, A. J., and Kolitawong, C. "Exact analytical solution
   for large-amplitude oscillatory shear flow." *Macromolecular Theory and Simulations*,
   24, 352-392 (2015). https://doi.org/10.1002/mats.201400104

See Also
--------

- :doc:`/user_guide/03_advanced_topics/spp_analysis` - Complete SPP analysis user guide
- :doc:`/api/spp_models` - SPP API reference
- :doc:`spp_decomposer` - SPP Decomposer transform documentation
