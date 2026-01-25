.. _model-cross:

=============
Cross Model
=============

Quick Reference
---------------

**Use when:** Well-characterized high-rate plateaus, tunable transition sharpness, suspensions and emulsions
**Parameters:** 4 (eta0, eta_inf, lambda, m)
**Key equation:** :math:`\eta = \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda\dot{\gamma})^m}`
**Test modes:** Flow (steady shear, rotation)
**Material examples:** Polymer melts, colloidal suspensions, emulsions, paints, inks, lubricants

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
     - Zero-shear viscosity (Pa·s); low-shear Newtonian plateau
   * - :math:`\eta_{\infty}`
     - Infinite-shear viscosity (Pa·s); high-shear Newtonian plateau
   * - :math:`\lambda`
     - Time constant (s); reciprocal of critical shear rate
   * - :math:`m`
     - Cross rate constant (dimensionless); controls transition sharpness
   * - :math:`\dot{\gamma}`
     - Shear rate (1/s)

Overview
--------

The Cross model is a four-parameter generalized Newtonian fluid equation that describes the smooth transition between two Newtonian plateaus. It was developed by Malcolm M. Cross in 1965 [1]_ specifically for polymer solutions and colloidal suspensions, predating the Carreau model by seven years.

The key distinguishing feature is the **tunable transition exponent** :math:`m`. While Carreau fixes the transition shape via a square-law term :math:`[1 + (\lambda\dot{\gamma})^2]`, Cross uses a general exponent :math:`m` that can be fitted to match experimental data more precisely.

Historical Context
~~~~~~~~~~~~~~~~~~

Cross developed the model while working on the rheology of pseudoplastic systems at ICI (Imperial Chemical Industries). His motivation was to create a flow equation that:

1. Predicts finite viscosity at zero shear rate (unlike power law)
2. Allows for a high-shear Newtonian plateau (observed in many real fluids)
3. Has tunable transition sharpness to match diverse materials

The Cross equation became particularly popular for:
   - Colloidal suspensions (where both plateaus are experimentally accessible)
   - Polymer solutions (especially at low concentrations)
   - Paints, inks, and coatings (quality control applications)
   - Biomedical fluids (blood, synovial fluid)

Relation to Carreau Model
~~~~~~~~~~~~~~~~~~~~~~~~~

The Carreau and Cross models are related:

- **Carreau**: :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}`
- **Cross**: :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^m]^{-1}`

When :math:`m = 2` and :math:`n = 0` (extreme shear-thinning), the models become equivalent in the power-law region. The choice between them often depends on:

- Historical preference in the application area
- Which functional form better fits the specific data
- Whether the transition region or asymptotic behavior is more important

----

Physical Foundations
--------------------

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Cross model captures flow behavior arising from shear-induced structural changes:

**At low shear rates** (:math:`\eta \approx \eta_0`):
   - Suspended particles or polymer chains are randomly oriented
   - Brownian motion maintains isotropic microstructure
   - Viscous resistance is maximum due to random collisions/entanglements
   - Flow timescale (:math:`1/\dot{\gamma}`) exceeds structural relaxation time

**At intermediate shear rates** (power-law region):
   - Shear flow begins to orient particles/chains
   - Aggregates or entanglements break up
   - Layers of particles slide past each other more easily
   - Viscosity decreases following :math:`\eta \propto \dot{\gamma}^{-m/(1+m\cdot\text{const})}` approximately

**At high shear rates** (:math:`\eta \approx \eta_{\infty}`):
   - Particles/chains are fully aligned with flow
   - Minimum structural resistance achieved
   - Only hydrodynamic interactions remain
   - For suspensions: :math:`\eta_{\infty}` approaches solvent viscosity with particle contribution

Physical Meaning of Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Time constant** :math:`\lambda`:
   Represents the characteristic time for structural rearrangement. The critical shear rate :math:`\dot{\gamma}_c = 1/\lambda` marks where viscosity has dropped halfway from :math:`\eta_0` toward :math:`\eta_{\infty}`.

   - **For suspensions**: Related to particle diffusion time :math:`\lambda \sim a^2/D_0` where :math:`a` is particle radius
   - **For polymers**: Related to longest relaxation time :math:`\lambda \sim \tau_d`

**Rate constant** :math:`m`:
   Controls how sharply viscosity transitions between plateaus:

   - **Small** :math:`m` **(0.2-0.5)**: Gradual, smooth transition over many decades
   - **Moderate** :math:`m` **(0.5-1.5)**: Typical for most polymer solutions and suspensions
   - **Large** :math:`m` **(>1.5)**: Sharp, switch-like transition (step-function as :math:`m \to \infty`)

Material Examples with Typical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative Cross parameters
   :header-rows: 1
   :widths: 25 15 12 12 10 10 16

   * - Material
     - :math:`\eta_0` (Pa·s)
     - :math:`\eta_{\infty}` (Pa·s)
     - :math:`\lambda` (s)
     - :math:`m`
     - T (°C)
     - Ref
   * - Silicone oil suspension
     - 15.2
     - 0.35
     - 0.08
     - 0.85
     - 25
     - [2]_
   * - Polyisobutylene solution
     - 12.8
     - 0.52
     - 0.15
     - 1.2
     - 25
     - [1]_
   * - Latex paint
     - 8.5
     - 0.15
     - 0.5
     - 0.95
     - 25
     - [3]_
   * - Synovial fluid
     - 2.5
     - 0.005
     - 1.2
     - 0.75
     - 37
     - [4]_
   * - Ink (offset printing)
     - 45.0
     - 1.2
     - 0.02
     - 1.1
     - 30
     - [5]_

----

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

The Cross viscosity function is:

.. math::

   \eta(\dot{\gamma}) = \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda\dot{\gamma})^m}

Equivalently, defining the reduced viscosity :math:`\eta_r = (\eta - \eta_{\infty})/(\eta_0 - \eta_{\infty})`:

.. math::

   \eta_r = \frac{1}{1 + (\lambda\dot{\gamma})^m}

Shear Stress Relation
~~~~~~~~~~~~~~~~~~~~~

The shear stress is:

.. math::

   \sigma = \eta(\dot{\gamma}) \cdot \dot{\gamma} = \left[ \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda\dot{\gamma})^m} \right] \dot{\gamma}

This is monotonically increasing for all :math:`m > 0`, ensuring flow stability.

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
     - First Newtonian plateau
   * - Critical
     - :math:`\lambda\dot{\gamma} = 1`
     - :math:`(\eta_0 + \eta_{\infty})/2`
     - Transition midpoint
   * - Power-law
     - :math:`\lambda\dot{\gamma} \gg 1`
     - :math:`\approx \eta_0 (\lambda\dot{\gamma})^{-m}` + :math:`\eta_{\infty}`
     - Shear-thinning
   * - High shear
     - :math:`\lambda\dot{\gamma} \to \infty`
     - :math:`\to \eta_{\infty}`
     - Second Newtonian plateau

Power-Law Approximation
~~~~~~~~~~~~~~~~~~~~~~~

In the power-law region (:math:`\lambda\dot{\gamma} \gg 1`), ignoring :math:`\eta_{\infty}`:

.. math::

   \eta \approx \eta_0 \lambda^{-m} \dot{\gamma}^{-m} = K \dot{\gamma}^{n-1}

where :math:`K = \eta_0 \lambda^{-m}` and :math:`n = 1 - m`. This connects Cross parameter :math:`m` to power-law index.

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
     - Zero-shear viscosity; first Newtonian plateau
   * - ``eta_inf``
     - :math:`\eta_{\infty}`
     - Pa·s
     - :math:`10^{-6} - 10^{6}`
     - Infinite-shear viscosity; often solvent viscosity
   * - ``lambda_``
     - :math:`\lambda`
     - s
     - :math:`10^{-6} - 10^{6}`
     - Time constant; :math:`1/\lambda` is transition shear rate
   * - ``m``
     - :math:`m`
     - —
     - :math:`0.1 - 2.0`
     - Rate constant; controls transition sharpness

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**eta0 (Zero-Shear Viscosity)**:
   - **Physical meaning**: Viscosity of the undisturbed structure
   - **For suspensions**: Depends on volume fraction :math:`\phi` via Krieger-Dougherty
   - **For polymers**: Related to molecular weight via :math:`\eta_0 \sim M^{3.4}`

**eta_inf (Infinite-Shear Viscosity)**:
   - **Physical meaning**: Residual viscosity after complete structure breakdown
   - **For suspensions**: Hydrodynamic contribution only; approaches :math:`\eta_s (1 - \phi/\phi_m)^{-[\eta]\phi_m}`
   - **For solutions**: Approximately the solvent viscosity

**lambda (Time Constant)**:
   - **Physical meaning**: Characteristic structural relaxation time
   - **Interpretation**: Faster relaxation (small :math:`\lambda`) → early transition to thinning
   - **Relation**: :math:`\dot{\gamma}_{1/2} = 1/\lambda` where :math:`\eta = (\eta_0 + \eta_{\infty})/2`

**m (Rate Constant)**:
   - **Physical meaning**: Steepness of the viscosity drop in transition region
   - **Connection to power law**: Approximately :math:`n = 1 - m` in mid-rate region
   - **Typical values**: 0.5-1.5 for most fluids

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. **Generalized Newtonian**: No memory effects, stress depends only on current :math:`\dot{\gamma}`
2. **Isothermal**: Constant temperature (combine with Arrhenius for T-dependence)
3. **Simple shear**: Steady unidirectional flow
4. **Inelastic**: No normal stress differences predicted

Data Requirements
~~~~~~~~~~~~~~~~~

- **Required**: Flow curve :math:`\eta(\dot{\gamma})` spanning at least 3 decades
- **Ideal**: Data capturing both plateaus (may require wide :math:`\dot{\gamma}` range)
- **For accurate** :math:`m`: Transition region well-resolved (5+ points)

Limitations
~~~~~~~~~~~

**No viscoelasticity**:
   Cannot predict :math:`G'(\omega)`, :math:`G''(\omega)`, or stress relaxation.
   Use Maxwell/Oldroyd-B for elastic effects.

**No yield stress**:
   Material always flows; :math:`\sigma \to 0` as :math:`\dot{\gamma} \to 0`.
   Use Herschel-Bulkley for yield stress fluids.

**No thixotropy**:
   Instantaneous response assumed; no time-dependent structure changes.
   Use DMT or fluidity models for thixotropy.

----

What You Can Learn
------------------

This section explains how to translate fitted Cross parameters into material
insights and actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**eta0 (Zero-Shear Viscosity)**:
   The zero-shear viscosity indicates the structural state at rest:

   - **High** :math:`\eta_0` **(>100 Pa·s)**: Strong particle aggregation, high molecular weight polymer, or concentrated system with extensive network formation
   - **Moderate** :math:`\eta_0` **(1-100 Pa·s)**: Typical for polymer solutions, emulsions, and moderately concentrated suspensions
   - **Low** :math:`\eta_0` **(<1 Pa·s)**: Dilute solution, weak interparticle attractions, or low molecular weight

   *For graduate students*: For suspensions, the Krieger-Dougherty equation relates :math:`\eta_0` to volume fraction: :math:`\eta_0 / \eta_s = (1 - \phi/\phi_m)^{-[\eta]\phi_m}` where :math:`\eta_s` is solvent viscosity, :math:`\phi` is volume fraction, and :math:`\phi_m` is maximum packing. This enables volume fraction estimation from viscosity measurements.

   *For practitioners*: :math:`\eta_0` controls critical processing behaviors—settling/sedimentation rates in storage, coating thickness during low-shear application, and leveling behavior after deposition. Target higher :math:`\eta_0` for shelf stability and sag prevention.

**eta_inf (Infinite-Shear Viscosity)**:
   The high-shear plateau reveals the fully disrupted microstructure:

   - **High ratio** :math:`\eta_{\infty}/\eta_0` **(>10%)**: Significant irreducible structure remains; strong hydrodynamic interactions even when fully aligned
   - **Low ratio** :math:`\eta_{\infty}/\eta_0` **(<1%)**: Nearly complete structural breakdown under flow; approaches solvent-like behavior

   *For graduate students*: For suspensions, :math:`\eta_{\infty}` approaches the Einstein limit :math:`\eta_s(1 + 2.5\phi)` when particles are fully dispersed and aligned. Deviations indicate residual aggregation or non-spherical particle effects.

   *For practitioners*: :math:`\eta_{\infty}` determines high-rate processing capability—spray atomization quality, high-speed coating uniformity, and pumping energy requirements at production rates. Lower values enable faster processing.

**lambda (Time Constant)**:
   The relaxation time marks the transition between regimes:

   - **Critical shear rate**: :math:`\dot{\gamma}_c = 1/\lambda` identifies where viscosity drops to halfway between plateaus
   - **Short** :math:`\lambda` **(<0.1 s)**: Fast structural response, suitable for high-speed operations
   - **Long** :math:`\lambda` **(>10 s)**: Slow structural relaxation, memory effects important

   *For graduate students*: For Brownian particles, :math:`\lambda \sim a^2/D_0` where :math:`a` is particle radius and :math:`D_0` is diffusion coefficient. For polymers, :math:`\lambda` scales with the longest relaxation time from chain dynamics.

   *For practitioners*: Compare :math:`\lambda` to process timescales. Operating at :math:`\dot{\gamma} \gg 1/\lambda` ensures material is in the thinned state; :math:`\dot{\gamma} \ll 1/\lambda` keeps material at rest viscosity. Design mixing speeds accordingly.

**m (Rate Constant)**:
   The transition sharpness parameter characterizes structural breakdown:

   - **Low** :math:`m` **(0.3-0.6)**: Gradual, smooth transition over many decades—indicates broad distribution of relaxation times or multiple structural elements breaking down at different rates
   - **Moderate** :math:`m` **(0.6-1.2)**: Typical for most polymer solutions and suspensions with moderate polydispersity
   - **High** :math:`m` **(1.2-2.0)**: Sharp, switch-like transition—indicates narrow relaxation spectrum or cooperative structural breakdown

   *For graduate students*: The parameter :math:`m` relates to polydispersity and relaxation time distribution breadth. Compare with Cole-Cole analysis of oscillatory data: broad distributions give low :math:`m`, narrow distributions give high :math:`m`.

   *For practitioners*: High :math:`m` materials have excellent "smart fluid" behavior—thick when still, thin when worked. This is ideal for coatings (sag-resistant yet sprayable). Low :math:`m` gives smoother processing with less abrupt rheology changes.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Cross Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Pattern
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - Large :math:`\eta_0/\eta_{\infty}`, high :math:`m`
     - Strong cooperative structure
     - Concentrated latex paints, thick emulsions
     - Excellent sag resistance with spray-ability
   * - Large :math:`\eta_0/\eta_{\infty}`, low :math:`m`
     - Broad relaxation spectrum
     - Polydisperse suspensions, polymer blends
     - Smooth processing window, forgiving
   * - Moderate :math:`\eta_0/\eta_{\infty}`, moderate :math:`m`
     - Standard structured fluid
     - Typical coatings, food emulsions
     - Balanced processing characteristics
   * - Small :math:`\eta_0/\eta_{\infty}` (<10)
     - Weak or minimal structure
     - Dilute polymer solutions
     - Limited shear-thinning, consider simpler model

----

Experimental Design
-------------------

When to Use Cross Model
~~~~~~~~~~~~~~~~~~~~~~~

**Use Cross when**:
   - Both Newtonian plateaus are experimentally accessible
   - Transition sharpness needs to be a fitted parameter
   - Suspension/emulsion with well-defined microstructure

**Use Carreau instead when**:
   - High-shear plateau is not reached
   - Polymer melt with standard transition behavior
   - Compatibility with existing CFD codes required

Recommended Test Protocol
~~~~~~~~~~~~~~~~~~~~~~~~~

**Steady Shear Flow Curve**

**Step 1**: Sample equilibration
   - Load sample, equilibrate at test temperature for 10 min
   - Pre-shear at moderate rate (10-100 s⁻¹) for 60 s, then rest 5 min

**Step 2**: Flow curve measurement
   - Shear rate sweep: :math:`10^{-3}` to :math:`10^{3}` s⁻¹
   - Log spacing: 5 points per decade minimum
   - Equilibration: Wait for steady stress (auto or fixed time)

**Step 3**: Ascending vs descending
   - Ascending sweep preferred for non-thixotropic materials
   - Compare ascending/descending to detect time effects

----

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1**: Estimate :math:`\eta_0` from lowest shear rates
   :math:`\eta_0 \approx` average of :math:`\eta` at :math:`\dot{\gamma} < 0.01/\lambda`

**Step 2**: Estimate :math:`\eta_{\infty}` from highest shear rates
   :math:`\eta_{\infty} \approx` average of :math:`\eta` at :math:`\dot{\gamma} > 100/\lambda`

**Step 3**: Find :math:`\lambda` from midpoint
   Where :math:`\eta = (\eta_0 + \eta_{\infty})/2`, :math:`\lambda = 1/\dot{\gamma}_{1/2}`

**Step 4**: Estimate :math:`m` from log-log slope
   In power-law region: slope :math:`\approx -m`

Optimization
~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Fast convergence for 4-parameter Cross model
   - Bounds recommended to prevent unphysical values

**Bounds**:
   - :math:`\eta_0`: [1e-2, 1e10] Pa·s
   - :math:`\eta_{\infty}`: [0, 0.9 × :math:`\eta_0`] Pa·s
   - :math:`\lambda`: [1e-6, 1e4] s
   - :math:`m`: [0.2, 2.0]

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - :math:`m` hits bounds
     - Transition shape doesn't match
     - Check for artifacts; try Carreau-Yasuda
   * - :math:`\eta_{\infty}` negative
     - Bound violation
     - Constrain :math:`\eta_{\infty} \geq 0`; check high-rate data
   * - Poor fit at transition
     - Functional form mismatch
     - Try Carreau or Carreau-Yasuda
   * - Correlated :math:`\lambda` and :math:`m`
     - Under-resolved transition
     - More data points in transition region

----

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import Cross

   # Shear rate data
   gamma_dot = np.logspace(-3, 4, 100)
   eta_data = experimental_viscosity(gamma_dot)

   # Create and fit model
   model = Cross()
   model.fit(gamma_dot, eta_data, test_mode='rotation')

   # Extract parameters
   eta0 = model.parameters.get_value('eta0')
   eta_inf = model.parameters.get_value('eta_inf')
   lambda_ = model.parameters.get_value('lambda_')
   m = model.parameters.get_value('m')

   print(f"Zero-shear viscosity: {eta0:.2f} Pa·s")
   print(f"Infinite-shear viscosity: {eta_inf:.4f} Pa·s")
   print(f"Time constant: {lambda_:.4f} s")
   print(f"Rate constant m: {m:.3f}")

Comparison with Carreau
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Carreau, Cross

   # Fit both models
   carreau = Carreau()
   carreau.fit(gamma_dot, eta_data, test_mode='rotation')

   cross = Cross()
   cross.fit(gamma_dot, eta_data, test_mode='rotation')

   # Compare fit quality
   print(f"Carreau R²: {carreau.score(gamma_dot, eta_data):.4f}")
   print(f"Cross R²: {cross.score(gamma_dot, eta_data):.4f}")

----

See Also
--------

- :doc:`carreau` — uses square-law exponent; choose based on transition shape
- :doc:`carreau_yasuda` — adds Yasuda exponent for even more flexibility
- :doc:`power_law` — approximates Cross mid-rate slope when plateaus unavailable
- :doc:`herschel_bulkley` — for yield stress fluids
- :doc:`../../transforms/smooth_derivative` — differentiate flow curves to estimate :math:`m`

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.Cross`

----

References
----------

.. [1] Cross, M. M. "Rheology of non-Newtonian fluids: A new flow equation for pseudoplastic systems."
   *Journal of Colloid Science*, **20**, 417-437 (1965).
   https://doi.org/10.1016/0095-8522(65)90022-X

.. [2] Barnes, H. A., Hutton, J. F. & Walters, K. *An Introduction to Rheology*.
   Elsevier, Amsterdam (1989).

.. [3] Patton, T. C. *Paint Flow and Pigment Dispersion*, 2nd Edition.
   Wiley-Interscience (1979).

.. [4] Fung, Y. C. *Biomechanics: Mechanical Properties of Living Tissues*, 2nd Edition.
   Springer (1993).

.. [5] Tanner, R. I. & Walters, K. *Rheology: An Historical Perspective*.
   Elsevier (1998).

.. [6] Larson, R. G. *The Structure and Rheology of Complex Fluids*.
   Oxford University Press (1999).

.. [7] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH (1994).

.. [8] Mewis, J. & Wagner, N. J. *Colloidal Suspension Rheology*.
   Cambridge University Press (2012).

.. [9] Krieger, I. M. & Dougherty, T. J. "A mechanism for non-Newtonian flow in suspensions of rigid spheres."
   *Transactions of the Society of Rheology*, **3**, 137-152 (1959).

.. [10] Morrison, F. A. *Understanding Rheology*.
   Oxford University Press (2001).

Further Reading
~~~~~~~~~~~~~~~

- Bird, R. B., Armstrong, R. C. & Hassager, O. *Dynamics of Polymeric Liquids, Vol. 1*.
  Wiley (1987). [Comprehensive treatment of generalized Newtonian models]
