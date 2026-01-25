.. _model-bingham:

Bingham Plastic
===============

Quick Reference
---------------

- **Use when:** Rigid below yield stress, Newtonian flow after yielding
- **Parameters:** 2 (τ_y, η_p)
- **Key equation:** :math:`\tau = \tau_y + \eta_p \dot{\gamma}` for :math:`|\tau| > \tau_y`
- **Test modes:** Flow (steady shear)
- **Material examples:** Cement pastes, drilling muds, mayonnaise, ketchup, toothpaste

Overview
--------

The **Bingham Plastic** model describes **viscoplastic materials** that behave as **rigid bodies below a yield stress** (:math:`\tau_y`) and **flow linearly** with a plastic viscosity (:math:`\eta_p`) once yielded. Named after Eugene Bingham (1922), this is the simplest model capturing yield-stress behavior—materials that require a minimum stress to initiate flow. The Bingham model is foundational for understanding **cement pastes, drilling muds, slurries, toothpaste, mayonnaise, and suspensions** whose post-yield flow curves are approximately Newtonian.

The model represents a critical transition in non-Newtonian fluid mechanics: below :math:`\tau_y`, the material acts as an **elastic or rigid solid**; above :math:`\tau_y`, it flows as a **shear-thinning or Newtonian liquid**. This behavior arises from **microstructural networks** (particle contacts, hydrogen bonds, electrostatic interactions) that must be broken before flow can occur.

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\tau`
     - Shear stress (Pa)
   * - :math:`\tau_y`
     - Yield stress (Pa). Minimum stress for flow initiation.
   * - :math:`\dot{\gamma}`
     - Shear rate (s\ :sup:`-1`)
   * - :math:`\eta_p`
     - Plastic viscosity (Pa·s). Post-yield slope.
   * - :math:`\eta_{app}`
     - Apparent viscosity (Pa·s) = :math:`\tau/\dot{\gamma}`
   * - :math:`Bi`
     - Bingham number = :math:`\tau_y L / (\eta_p U)`. Ratio of yield to viscous stress.

Physical Foundation
-------------------

**Microstructural Origin of Yield Stress:**

The yield stress :math:`\tau_y` arises from:

1. **Particle Networks**: Colloidal particles forming space-spanning structures (e.g., clay suspensions, cement)
2. **Attractive Interactions**: Van der Waals, electrostatic, or depletion forces creating particle bridges
3. **Entangled Structures**: Fiber networks, polymer chains, or droplet clusters
4. **Jamming Transitions**: Dense suspensions where particles cage each other

**Post-Yield Flow:**

Once :math:`\tau > \tau_y`, the network breaks and particles/droplets flow past each other with a constant viscosity :math:`\eta_p` (plastic viscosity), analogous to Newtonian flow but offset by the yield stress.

Governing Equations
-------------------

**Constitutive Equation:**

.. math::

   \tau =
   \begin{cases}
      \tau_y \operatorname{sgn}(\dot{\gamma}) & \text{if } |\tau| \le \tau_y \text{ (unyielded)}, \\
      \tau_y + \eta_p \dot{\gamma} & \text{if } |\tau| > \tau_y \text{ (yielded)},
   \end{cases}

where:

- :math:`\tau` = shear stress (Pa)
- :math:`\dot{\gamma}` = shear rate (s\ :sup:`-1`)
- :math:`\tau_y` = yield stress (Pa), :math:`\tau_y \geq 0`
- :math:`\eta_p` = plastic viscosity (Pa·s), :math:`\eta_p > 0`

**Apparent Viscosity:**

.. math::

   \eta(\dot{\gamma}) = \frac{\tau_y}{|\dot{\gamma}|} + \eta_p \qquad \text{for } |\dot{\gamma}| > 0

The apparent viscosity **diverges** as :math:`\dot{\gamma} \to 0` (infinite viscosity at very low shear rates).

**Flow Curve Interpretation:**

Plot :math:`\tau` vs :math:`\dot{\gamma}`:

- **Intercept**: :math:`\tau_y` (extrapolation to :math:`\dot{\gamma} = 0`)
- **Slope**: :math:`\eta_p` (linear post-yield region)

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 24 24 52

   * - Name
     - Units
     - Description / Constraints
   * - ``tau_y``
     - Pa
     - Yield stress; ≥ 0. Sets the plateau torque required to initiate motion. Typical range: 1-500 Pa.
   * - ``eta_p``
     - Pa·s
     - Plastic viscosity governing the linear post-yield segment; > 0. Typical range: 0.001-10 Pa·s.

Material Examples
-----------------

**Cement and Construction Materials** (:math:`\tau_y \approx 10-200` Pa, :math:`\eta_p \approx 0.1-5` Pa·s):

- **Cement pastes** (water-cement ratio dependent)
- **Concrete slurries** (fresh concrete)
- **Mortar** and **grouts**
- **3D printing inks** (cementitious)

**Drilling and Mining Fluids** (:math:`\tau_y \approx 5-50` Pa, :math:`\eta_p \approx 0.01-0.5` Pa·s):

- **Bentonite drilling muds**
- **Barite-weighted fluids**
- **Oil well drilling fluids**

**Food Products** (:math:`\tau_y \approx 10-100` Pa, :math:`\eta_p \approx 0.1-5` Pa·s):

- **Mayonnaise** (:math:`\tau_y \approx 80-150` Pa)
- **Ketchup** (:math:`\tau_y \approx 20-50` Pa)
- **Mustard** (:math:`\tau_y \approx 30-70` Pa)
- **Chocolate** (molten, :math:`\tau_y \approx 5-20` Pa)

**Personal Care and Pharmaceuticals** (:math:`\tau_y \approx 50-300` Pa):

- **Toothpaste** (:math:`\tau_y \approx 100-200` Pa)
- **Lotions and creams**
- **Ointments** and **gels**

**Suspensions and Slurries** (:math:`\tau_y \approx 1-100` Pa):

- **Clay suspensions** (kaolin, montmorillonite)
- **Mineral slurries** (tailings, coal slurries)
- **Activated sludge** (wastewater treatment)

Experimental Design
-------------------

**Flow Curve (Controlled Shear Rate):**

1. **Shear rate sweep**: 0.001-1000 s\ :sup:`-1` (log-spaced, 10 points/decade)
2. **Pre-shear**: High shear rate (100 s\ :sup:`-1`, 60 s) to erase history
3. **Rest period**: 2-5 min to allow structure recovery
4. **Ramp protocol**: Low → high or bidirectional to check thixotropy
5. **Geometry**: **Vane or serrated plates** to minimize wall slip

**Yield Stress Determination Methods:**

1. **Flow Curve Extrapolation**:
   - Linear regression of :math:`\tau` vs :math:`\dot{\gamma}` in post-yield region
   - Intercept at :math:`\dot{\gamma} = 0` → :math:`\tau_y`
   - **Caution**: Sensitive to fitting range selection

2. **Controlled Stress Ramp**:
   - Apply increasing stress, monitor strain rate
   - :math:`\tau_y` = stress where :math:`\dot{\gamma}` jumps from ~0 to finite value
   - More reliable for materials with sharp yielding

3. **Vane Method** (ASTM D4648):
   - Insert vane into sample, rotate at constant speed
   - Peak torque → :math:`\tau_y` (accounts for 3D geometry)
   - Minimizes wall slip artifacts

**Avoiding Common Artifacts:**

- **Wall slip**: Use roughened surfaces, vane geometry, or serrated plates
- **Sedimentation**: Short measurement times, homogenize before test
- **Evaporation**: Solvent trap, short test duration
- **Thixotropy**: Control rest time, use consistent pre-shear protocol

Physical Foundations
--------------------

Microstructural Origin of Yield Stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The yield stress :math:`\tau_y` arises from:

1. **Particle Networks**: Colloidal particles forming space-spanning structures (e.g., clay suspensions, cement)
2. **Attractive Interactions**: Van der Waals, electrostatic, or depletion forces creating particle bridges
3. **Entangled Structures**: Fiber networks, polymer chains, or droplet clusters
4. **Jamming Transitions**: Dense suspensions where particles cage each other

Post-Yield Flow
~~~~~~~~~~~~~~~

Once :math:`\tau > \tau_y`, the network breaks and particles/droplets flow past each other with a constant viscosity :math:`\eta_p` (plastic viscosity), analogous to Newtonian flow but offset by the yield stress.

Validity and Assumptions
------------------------

When Bingham Model Applies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Bingham model is appropriate when:

1. **Clear yield stress exists**: Material does not flow below a critical stress.
   The stress-strain rate curve shows a stress intercept at zero rate.

2. **Newtonian post-yield behavior**: After yielding, the material follows
   linear flow with constant plastic viscosity.

3. **Steady-state flow**: Material reaches equilibrium at each shear rate
   (no thixotropy or aging during measurement).

4. **No slip at walls**: The material shears uniformly without wall slip.

When to Use Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection
   :widths: 35 30 35
   :header-rows: 1

   * - Observation
     - Issue
     - Better Model
   * - Post-yield shear-thinning/thickening
     - Non-linear flow beyond yield
     - :doc:`herschel_bulkley` (:math:`n \neq 1`)
   * - Fitted :math:`\tau_y \approx 0`
     - No yield stress
     - :doc:`power_law` or :doc:`carreau`
   * - Thixotropic hysteresis
     - Time-dependent structure
     - Fluidity models, DMT
   * - Stress overshoot in startup
     - Viscoelastic effects
     - Saramito EVP, SGR

Fitting Guidance
----------------

Initialization
~~~~~~~~~~~~~~

1. **From flow curve**: Linear fit of high shear rate region → slope = :math:`\eta_p`, intercept = :math:`\tau_y`
2. **From controlled stress**: Stress at flow onset → :math:`\tau_y`
3. **Robust estimation**: Median of multiple yield determinations

Optimization
~~~~~~~~~~~~

- **Use Huber loss** to down-weight noisy pre-yield data
- **Weighted least squares**: Higher weights on post-yield region where :math:`\dot{\gamma}` is reliable
- **Constrain** :math:`\tau_y \geq 0` and :math:`\eta_p > 0`
- **Verify**: Residuals should be random in post-yield region

Handling Pre-Yield Data
~~~~~~~~~~~~~~~~~~~~~~~~

- **Option 1**: Exclude data below :math:`\dot{\gamma} < 0.01` s\ :sup:`-1` (noisy, not truly rigid)
- **Option 2**: Fit only post-yield region (:math:`\tau > 1.1 \tau_y`)
- **Option 3**: Use robust loss (Huber, Tukey) to reduce influence of outliers

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Cause
     - Solution
   * - Negative :math:`\tau_y` estimates
     - Noisy sub-yield data dominating fit
     - Use robust loss (Huber), exclude low :math:`\dot{\gamma}` points, or constrain :math:`\tau_y \geq 0`
   * - Poor fit quality despite linear appearance
     - Thixotropic hysteresis (up-ramp ≠ down-ramp)
     - Use consistent pre-shear protocol or fit up/down separately
   * - Apparent viscosity shows curvature
     - Material exhibits shear-thinning beyond yielding
     - Use :doc:`herschel_bulkley` (:math:`n < 1`) or Casson model
   * - Scatter at low shear rates
     - Instrument torque resolution, slip, or structural recovery
     - Use vane geometry, faster ramp rate, or exclude :math:`\dot{\gamma} < 0.01` s\ :sup:`-1`

Usage
-------------

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import Bingham

   # Generate synthetic data (toothpaste)
   gamma_dot = jnp.logspace(-2, 2, 80)  # 0.01 - 100 s⁻¹
   tau_exp = 120.0 + 2.5 * gamma_dot + jnp.random.normal(0, 3, size=gamma_dot.shape)

   # Initialize and fit
   model = Bingham(tau_y=120.0, eta_p=2.5)
   model.fit(gamma_dot, tau_exp, loss="huber", ftol=1e-6)

   # Inspect fitted parameters
   print(f"Yield stress: {model.parameters.get_value('tau_y'):.2f} Pa")
   print(f"Plastic viscosity: {model.parameters.get_value('eta_p'):.3f} Pa·s")

   # Predict and plot
   tau_pred = model.predict(gamma_dot)

Model Comparison
----------------

**Bingham vs Herschel-Bulkley:**

- **Bingham**: Linear post-yield (:math:`n = 1`)
- **Herschel-Bulkley**: Power-law post-yield (:math:`\tau = \tau_y + K\dot{\gamma}^n`)
- Use Bingham when post-yield flow is Newtonian; HB for shear-thinning/thickening

**Bingham vs Casson:**

- **Bingham**: :math:`\tau = \tau_y + \eta_p \dot{\gamma}`
- **Casson**: :math:`\sqrt{\tau} = \sqrt{\tau_y} + \sqrt{\eta_{\infty} \dot{\gamma}}`
- Casson better for **blood** and **chocolate**; Bingham for suspensions

**Bingham vs Carreau:**

- **Bingham**: Discontinuous yielding
- **Carreau**: Smooth shear-thinning without yield
- Combine for materials with both yield stress and gradual thinning

Limitations
-----------

1. **Pre-yield behavior**: Assumes rigid solid; real materials show viscoelastic creep
2. **Sharp yielding**: Real yield is gradual transition, not instantaneous
3. **Newtonian post-yield**: Cannot capture shear-thinning/thickening beyond yield
4. **No thixotropy**: Static model, ignores structural evolution
5. **Wall slip**: Requires careful geometry selection

What You Can Learn
------------------

This section explains how to translate fitted Bingham parameters into material
insights and actionable knowledge for both research and industrial applications.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Yield Stress (τ_y)**:
   The yield stress reveals the strength of the material's internal network:

   - **Low yield stress (< 10 Pa)**: Weak structure. Material flows easily under
     gravity or light handling. Examples: dilute suspensions, low-fat mayo.

   - **Moderate yield stress (10–100 Pa)**: Functional structure for most
     applications. Sufficient to prevent sedimentation and sagging, yet
     dispensable with reasonable force.

   - **High yield stress (> 100 Pa)**: Strong network requiring significant force
     to initiate flow. Examples: toothpaste, heavy-duty grease, cement paste.

   *For graduate students*: The yield stress scales with microstructural
   parameters. For colloidal suspensions: :math:`\tau_y \propto \phi^2 G_p / a`
   where :math:`\phi` is volume fraction, :math:`G_p` is particle modulus, and
   :math:`a` is particle size. For attractive systems, :math:`\tau_y` increases
   exponentially with interparticle attraction strength.

   *For practitioners*: Use :math:`\tau_y` to assess shelf stability. A mayonnaise
   needs :math:`\tau_y > 50` Pa to prevent oil separation; a paint needs
   :math:`\tau_y > 5` Pa to avoid sagging on vertical surfaces.

**Plastic Viscosity (η_p)**:
   The plastic viscosity governs post-yield energy dissipation:

   - **Low η_p (< 0.1 Pa·s)**: Thin flow once yielded. Good for easy pumping
     but may cause splashing or poor coating uniformity.

   - **Moderate η_p (0.1–5 Pa·s)**: Balanced flow. Typical for most applications
     requiring controlled spreading or mixing.

   - **High η_p (> 5 Pa·s)**: Viscous flow requiring sustained energy input.
     Common in heavy pastes and slurries.

   *For graduate students*: The plastic viscosity includes contributions from
   the continuous phase viscosity, hydrodynamic interactions between particles,
   and the rate of network breakdown. For concentrated suspensions:
   :math:`\eta_p \approx \eta_s (1 - \phi/\phi_m)^{-2}` where :math:`\phi_m` is
   the maximum packing fraction.

   *For practitioners*: The pumping power scales with :math:`\eta_p`. Reducing
   particle size or concentration lowers :math:`\eta_p` and pumping costs, but
   may also reduce :math:`\tau_y` and shelf stability.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Bingham Fluid Classification
   :header-rows: 1
   :widths: 20 20 30 30

   * - τ_y / η_p Ratio
     - Behavior
     - Typical Materials
     - Process Implications
   * - Low ratio (< 10 s⁻¹)
     - "Thick and easy"
     - Light sauces, thin lotions
     - Easy pumping, gravity flow possible
   * - Moderate (10–100 s⁻¹)
     - Balanced plasticity
     - Mayonnaise, drilling mud
     - Standard processing equipment
   * - High ratio (> 100 s⁻¹)
     - "Stiff paste"
     - Toothpaste, cement
     - High pressure extrusion needed

The ratio :math:`\tau_y / \eta_p` has units of shear rate and indicates the
characteristic rate where yield stress and viscous stress are comparable.

Engineering Applications
~~~~~~~~~~~~~~~~~~~~~~~~

**Pipe Flow Design**:
   The Buckingham-Reiner equation predicts pressure drop:

   .. math::
      \frac{\Delta P}{L} = \frac{8 \eta_p Q}{\pi R^4} \left[1 + \frac{1}{3} Bi - \frac{4}{3} Bi^{-3} \right]^{-1}

   where :math:`Bi = \tau_y R / (\eta_p \bar{v})` is the Bingham number. For
   :math:`Bi > 3`, plug flow dominates and pressure scales with :math:`\tau_y/R`.

**Coating and Spreading**:
   For gravity-driven leveling on an inclined surface:

   - Material will not flow if :math:`\tau_y > \rho g h \sin\theta`
   - Use this to size layer thickness :math:`h` for sag prevention

**Mixing Power**:
   Anchor or helical impellers are preferred. Power requirement scales as:

   .. math::
      P \propto \tau_y V + \eta_p (\dot{\gamma}_{avg}) V

   where :math:`V` is vessel volume and :math:`\dot{\gamma}_{avg}` is average
   shear rate in the mixer.

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- **τ_y → 0**: Material is Newtonian or nearly so. Check if yield stress model
  is appropriate; consider using :doc:`carreau` instead.

- **τ_y negative**: Fitting artifact from noisy low-rate data. Constrain to
  :math:`\tau_y \geq 0` or use robust fitting.

- **η_p unexpectedly low**: Check for wall slip or instrument calibration issues.

- **Strong correlation between τ_y and η_p**: Insufficient data range. Extend
  measurements to higher shear rates for better separation.

- **Systematic residuals**: If residuals curve, the material shows shear-thinning
  post-yield. Use :doc:`herschel_bulkley` instead.

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Quality Control for Food Products**:
   Monitor :math:`\tau_y` as primary QC metric. A 20% drop in :math:`\tau_y`
   indicates batch problems (wrong emulsifier ratio, insufficient homogenization).

**Drilling Mud Formulation**:
   Target :math:`\tau_y = 5-15` Pa for cuttings suspension with :math:`\eta_p < 0.1`
   Pa·s for easy circulation. The API recommends reporting both 6 rpm and 300 rpm
   readings for Bingham analysis.

**Cement Mix Design**:
   Fresh concrete workability correlates with Bingham parameters. Self-compacting
   concrete requires :math:`\tau_y < 60` Pa and :math:`\eta_p < 50` Pa·s.

Fitting Guidance
----------------

Initialization
~~~~~~~~~~~~~~

1. **From flow curve**: Linear fit of high shear rate region → slope = :math:`\eta_p`, intercept = :math:`\tau_y`
2. **From controlled stress**: Stress at flow onset → :math:`\tau_y`
3. **Robust estimation**: Median of multiple yield determinations

Optimization
~~~~~~~~~~~~

- **Use Huber loss** to down-weight noisy pre-yield data
- **Weighted least squares**: Higher weights on post-yield region where :math:`\dot{\gamma}` is reliable
- **Constrain** :math:`\tau_y \geq 0` and :math:`\eta_p > 0`
- **Verify**: Residuals should be random in post-yield region

Handling Pre-Yield Data
~~~~~~~~~~~~~~~~~~~~~~~~

- **Option 1**: Exclude data below :math:`\dot{\gamma} < 0.01` s\ :sup:`-1` (noisy, not truly rigid)
- **Option 2**: Fit only post-yield region (:math:`\tau > 1.1 \tau_y`)
- **Option 3**: Use robust loss (Huber, Tukey) to reduce influence of outliers

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Cause
     - Solution
   * - Negative :math:`\tau_y` estimates
     - Noisy sub-yield data dominating fit
     - Use robust loss (Huber), exclude low :math:`\dot{\gamma}` points, or constrain :math:`\tau_y \geq 0`
   * - Poor fit quality despite linear appearance
     - Thixotropic hysteresis (up-ramp ≠ down-ramp)
     - Use consistent pre-shear protocol or fit up/down separately
   * - Apparent viscosity shows curvature
     - Material exhibits shear-thinning beyond yielding
     - Use :doc:`herschel_bulkley` (:math:`n < 1`) or Casson model
   * - Scatter at low shear rates
     - Instrument torque resolution, slip, or structural recovery
     - Use vane geometry, faster ramp rate, or exclude :math:`\dot{\gamma} < 0.01` s\ :sup:`-1`

Tips & Best Practices
----------------------

1. **Pre-shear consistently**: Erase mechanical history before each measurement
2. **Use vane geometry**: Minimizes wall slip for yield stress materials
3. **Bidirectional sweeps**: Check for thixotropic hysteresis
4. **Robust fitting**: Huber or Tukey loss to handle pre-yield noise
5. **Validate yield stress**: Compare flow curve, stress ramp, and vane methods
6. **Temperature control**: :math:`\tau_y` and :math:`\eta_p` are temperature-sensitive (±0.1°C)
7. **Avoid evaporation**: Use solvent trap for aqueous systems

References
----------

.. [1] Bingham, E. C. *Fluidity and Plasticity*. McGraw-Hill, New York (1922).
   The original description of the Bingham plastic model.

.. [2] Barnes, H. A. "The yield stress—a review or 'παντα ρει'—everything flows?"
   *Journal of Non-Newtonian Fluid Mechanics*, 81, 133–178 (1999).
   https://doi.org/10.1016/S0377-0257(98)00094-9

.. [3] Coussot, P., and Ancey, C. "Rheophysical classification of concentrated
   suspensions and granular pastes." *Physical Review E*, 59, 4445–4457 (1999).
   https://doi.org/10.1103/PhysRevE.59.4445

.. [4] Larson, R. G. *The Structure and Rheology of Complex Fluids*. Oxford
   University Press, New York (1999). ISBN: 978-0195121971

.. [5] Coussot, P. *Rheometry of Pastes, Suspensions, and Granular Materials:
   Applications in Industry and Environment*. Wiley (2005).
   https://doi.org/10.1002/0471720577

.. [6] Balmforth, N. J., Frigaard, I. A., and Ovarlez, G. "Yielding to stress:
   Recent developments in viscoplastic fluid mechanics."
   *Annual Review of Fluid Mechanics*, 46, 121–146 (2014).
   https://doi.org/10.1146/annurev-fluid-010313-141424

.. [7] Cheng, D. C.-H. "Yield stress: A time-dependent property and how to
   measure it." *Rheologica Acta*, 25, 542–554 (1986).
   https://doi.org/10.1007/BF01774406

.. [8] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758

.. [9] Mewis, J., and Wagner, N. J. *Colloidal Suspension Rheology*.
   Cambridge University Press (2012). ISBN: 978-0521515993

.. [10] Møller, P. C. F., Mewis, J., and Bonn, D. "Yield stress and thixotropy:
   On the difficulty of measuring yield stresses in practice."
   *Soft Matter*, 2, 274–283 (2006).
   https://doi.org/10.1039/b517840a

See Also
--------

Related Flow Models
~~~~~~~~~~~~~~~~~~~

- :doc:`herschel_bulkley` — generalizes Bingham with power-law post-yield slope (:math:`n \neq 1`)
- :doc:`power_law` — zero-yield limit for simple shear-thinning/thickening fits
- :doc:`carreau` — smooth transition between Newtonian plateaus without yield
- :doc:`../fractional/fractional_zener_sl` — combines fractional relaxation with elastic plateau

Transforms
~~~~~~~~~~

- :doc:`../../transforms/smooth_derivative` — differentiate torque signals for stress calculation
- :doc:`../../transforms/mutation_number` — monitor structural breakdown during yielding

Examples
~~~~~~~~

- :doc:`../../examples/flow/01-bingham-fitting` — step-by-step Bingham parameter estimation
- :doc:`../../examples/advanced/02-yield-stress-comparison` — comparing Bingham, HB, and Casson
