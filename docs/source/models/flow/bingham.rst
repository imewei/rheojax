.. _model-bingham:

Bingham Plastic
===============

Quick Reference
---------------

**Use when:** Rigid below yield stress, Newtonian flow after yielding
**Parameters:** 2 (τ_y, η_p)
**Key equation:** :math:`\tau = \tau_y + \eta_p \dot{\gamma}` for :math:`|\tau| > \tau_y`
**Test modes:** Flow (steady shear)
**Material examples:** Cement pastes, drilling muds, mayonnaise, ketchup, toothpaste

Overview
--------

The **Bingham Plastic** model describes **viscoplastic materials** that behave as **rigid bodies below a yield stress** (:math:`\tau_y`) and **flow linearly** with a plastic viscosity (:math:`\eta_p`) once yielded. Named after Eugene Bingham (1922), this is the simplest model capturing yield-stress behavior—materials that require a minimum stress to initiate flow. The Bingham model is foundational for understanding **cement pastes, drilling muds, slurries, toothpaste, mayonnaise, and suspensions** whose post-yield flow curves are approximately Newtonian.

The model represents a critical transition in non-Newtonian fluid mechanics: below :math:`\tau_y`, the material acts as an **elastic or rigid solid**; above :math:`\tau_y`, it flows as a **shear-thinning or Newtonian liquid**. This behavior arises from **microstructural networks** (particle contacts, hydrogen bonds, electrostatic interactions) that must be broken before flow can occur.

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

Fitting Strategies
------------------

**Initialization:**

1. **From flow curve**: Linear fit of high shear rate region → slope = :math:`\eta_p`, intercept = :math:`\tau_y`
2. **From controlled stress**: Stress at flow onset → :math:`\tau_y`
3. **Robust estimation**: Median of multiple yield determinations

**Optimization:**

- **Use Huber loss** to down-weight noisy pre-yield data
- **Weighted least squares**: Higher weights on post-yield region where :math:`\dot{\gamma}` is reliable
- **Constrain** :math:`\tau_y \geq 0` and :math:`\eta_p > 0`
- **Verify**: Residuals should be random in post-yield region

**Handling Pre-Yield Data:**

- **Option 1**: Exclude data below :math:`\dot{\gamma} < 0.01` s\ :sup:`-1` (noisy, not truly rigid)
- **Option 2**: Fit only post-yield region (:math:`\tau > 1.1 \tau_y`)
- **Option 3**: Use robust loss (Huber, Tukey) to reduce influence of outliers

Usage Example
-------------

.. code-block:: python

   import jax.numpy as jnp
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

Troubleshooting
---------------

**Issue: Negative** :math:`\tau_y` **estimates**

- **Cause**: Noisy sub-yield data dominating fit
- **Solution**: Use robust loss (Huber), exclude low :math:`\dot{\gamma}` points, or constrain :math:`\tau_y \geq 0`

**Issue: Poor fit quality despite linear appearance**

- **Cause**: Thixotropic hysteresis (up-ramp ≠ down-ramp)
- **Solution**: Use consistent pre-shear protocol or fit up/down separately

**Issue: Apparent viscosity shows curvature**

- **Cause**: Material exhibits shear-thinning beyond yielding
- **Solution**: Use :doc:`herschel_bulkley` (:math:`n < 1`) or Casson model

**Issue: Scatter at low shear rates**

- **Cause**: Instrument torque resolution, slip, or structural recovery
- **Solution**: Use vane geometry, faster ramp rate, or exclude :math:`\dot{\gamma} < 0.01` s\ :sup:`-1`

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

**Foundational Papers:**

- E.C. Bingham, *Fluidity and Plasticity*, McGraw-Hill (1922). — **Original Bingham model**
- H.A. Barnes, "The yield stress—a review," *J. Non-Newtonian Fluid Mech.* 81, 133–178
  (1999). — **Comprehensive yield stress review**
- P. Coussot and C. Ancey, "Rheophysical classification of concentrated suspensions and
  granular pastes," *Phys. Rev. E* 59, 4445–4457 (1999).

**Rheometry and Measurement:**

- R.G. Larson, *The Structure and Rheology of Complex Fluids*, Oxford (1999).
- G. Coussot, *Rheometry of Pastes, Suspensions, and Granular Materials*, Wiley (2005). —
  **Practical yield stress measurement**

**Applications:**

- N.J. Balmforth et al., "Yielding to stress," *J. Non-Newtonian Fluid Mech.* 142, 1–12
  (2007). — **Yield stress fluid dynamics**
- D.C.H. Cheng, "Yield stress: A time-dependent property and how to measure it," *Rheol.
  Acta* 25, 542–554 (1986).

**Standards:**

- ASTM D4648: Standard Test Method for Laboratory Miniature Vane Shear Test
- ISO 3219: Plastics – Polymers/resins in the liquid state or as emulsions or dispersions
  – Determination of viscosity using a rotational viscometer

See also
--------

**Related Flow Models:**

- :doc:`herschel_bulkley` — generalizes Bingham with power-law post-yield slope (:math:`n \neq 1`)
- :doc:`power_law` — zero-yield limit for simple shear-thinning/thickening fits
- :doc:`carreau` — smooth transition between Newtonian plateaus without yield
- :doc:`../fractional/fractional_zener_sl` — combines fractional relaxation with elastic plateau

**Transforms:**

- :doc:`../../transforms/smooth_derivative` — differentiate torque signals for stress calculation
- :doc:`../../transforms/mutation_number` — monitor structural breakdown during yielding

**Examples:**

- :doc:`../../examples/flow/01-bingham-fitting` — step-by-step Bingham parameter estimation
- :doc:`../../examples/advanced/02-yield-stress-comparison` — comparing Bingham, HB, and Casson
