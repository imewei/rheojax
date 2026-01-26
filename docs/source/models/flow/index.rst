Flow Curve Models
=================

This section documents models for steady-state shear flow behavior—the relationship
between shear stress and shear rate under continuous deformation.


Quick Reference
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`power_law`
     - 2 (K, n)
     - Simple shear-thinning/thickening fluids
   * - :doc:`carreau`
     - 4 (:math:`\eta_0`, :math:`\eta_\infty`, :math:`\lambda`, n)
     - Full flow curve with Newtonian plateaus
   * - :doc:`carreau_yasuda`
     - 5 (+a)
     - Sharper transition region control
   * - :doc:`cross`
     - 4 (:math:`\eta_0, \eta_{\infty}`, K, n)
     - Alternative to Carreau, different transition
   * - :doc:`bingham`
     - 2 (:math:`\sigma_y`, :math:`\eta_p`)
     - Simple yield stress fluids
   * - :doc:`herschel_bulkley`
     - 3 (:math:`\sigma_y`, K, n)
     - Yield stress + power-law flow


Overview
--------

Flow curve models describe the steady-state relationship :math:`\sigma = f(\dot{\gamma})`
between shear stress and shear rate. These models are essential for:

- **Process design**: Pump selection, pipe flow calculations
- **Material characterization**: Viscosity classification (ASTM, ISO)
- **Quality control**: Batch-to-batch consistency
- **Formulation development**: Additive effects on flow behavior

**Key phenomena captured:**

- **Shear thinning**: Decreasing viscosity with increasing shear rate (most polymeric fluids)
- **Shear thickening**: Increasing viscosity (concentrated suspensions, some pastes)
- **Yield stress**: Finite stress required to initiate flow
- **Newtonian plateaus**: Constant viscosity at extreme shear rates


Model Hierarchy
---------------

::

   Flow Curve Models
   │
   ├── Newtonian Region Models (no yield stress)
   │   ├── Power Law (Ostwald-de Waele)
   │   │   └── σ = K · γ̇^n
   │   │   └── Simple, 2 parameters
   │   │   └── No plateaus
   │   │
   │   ├── Carreau
   │   │   └── η = η∞ + (η_0-η∞)[1+(λγ̇)^2]^((n-1)/2)
   │   │   └── Both plateaus, smooth transition
   │   │
   │   ├── Carreau-Yasuda
   │   │   └── η = η∞ + (η_0-η∞)[1+(λγ̇)^a]^((n-1)/a)
   │   │   └── Adjustable transition sharpness
   │   │
   │   └── Cross
   │       └── η = η∞ + (η_0-η∞)/[1+(Kγ̇)^n]
   │       └── Different transition shape
   │
   └── Yield Stress Models
       ├── Bingham
       │   └── σ = σ_y + η_p · γ̇  (if σ > σ_y)
       │   └── Linear above yield
       │
       └── Herschel-Bulkley
           └── σ = σ_y + K · γ̇^n  (if σ > σ_y)
           └── Power-law above yield


When to Use Which Model
-----------------------

.. list-table::
   :widths: 25 12 12 12 12 12 15
   :header-rows: 1

   * - Feature
     - Power Law
     - Carreau
     - C-Y
     - Cross
     - Bingham
     - H-B
   * - Shear thinning
     - ✓
     - ✓
     - ✓
     - ✓
     - ✗
     - ✓
   * - Yield stress
     - ✗
     - ✗
     - ✗
     - ✗
     - ✓
     - ✓
   * - Zero-shear plateau
     - ✗
     - ✓
     - ✓
     - ✓
     - N/A
     - N/A
   * - High-shear plateau
     - ✗
     - ✓
     - ✓
     - ✓
     - ✗
     - ✗
   * - Transition control
     - ✗
     - Fixed
     - ✓
     - Fixed
     - N/A
     - N/A
   * - Simple fitting
     - ✓✓
     - ✓
     - ~
     - ✓
     - ✓✓
     - ✓

**Decision Flowchart:**

1. Does the material have a yield stress?
   - **Yes** → Bingham (linear) or Herschel-Bulkley (power-law)
   - **No** → Continue

2. Do you observe Newtonian plateaus at low and/or high shear rates?
   - **Yes** → Carreau, Carreau-Yasuda, or Cross
   - **No** → Power Law (limited range)

3. Is the transition between plateaus sharp or gradual?
   - **Sharp** → Carreau-Yasuda (tune parameter a)
   - **Gradual** → Carreau or Cross


Material Examples
-----------------

.. list-table::
   :widths: 25 20 25 30
   :header-rows: 1

   * - Material
     - Typical Model
     - Key Parameters
     - Industry
   * - Polymer solutions
     - Carreau
     - :math:`\eta_0` = 1-100 Pa·s, n = 0.3-0.7
     - Plastics, coatings
   * - Polymer melts
     - Carreau-Yasuda
     - :math:`\eta_0` = :math:`10^3-10^5` Pa·s, a = 2
     - Extrusion, injection
   * - Blood
     - Carreau
     - :math:`\eta_0` ≈ 50 mPa·s, n ≈ 0.4
     - Biomedical
   * - Paints
     - Cross or H-B
     - :math:`\sigma_y` = 0.5-10 Pa
     - Coatings
   * - Toothpaste
     - Herschel-Bulkley
     - :math:`\sigma_y` = 10-100 Pa
     - Personal care
   * - Drilling mud
     - Herschel-Bulkley
     - :math:`\sigma_y` = 5-50 Pa, n = 0.5-0.8
     - Oil & gas
   * - Ketchup
     - Herschel-Bulkley
     - :math:`\sigma_y` ≈ 15 Pa
     - Food
   * - Concrete
     - Bingham
     - :math:`\sigma_y` = 10-100 Pa
     - Construction


Key Parameters
--------------

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - Zero-shear viscosity
     - :math:`\eta_0`
     - Pa·s
     - Viscosity at rest (Newtonian plateau)
   * - Infinite-shear viscosity
     - :math:`\eta_\infty`
     - Pa·s
     - High-rate limit (often ≈ 0)
   * - Consistency index
     - K
     - Pa·s^n
     - Power-law prefactor (magnitude)
   * - Flow index
     - n
     - —
     - n < 1: thinning, n > 1: thickening
   * - Relaxation time
     - :math:`\lambda`
     - s
     - Onset of shear thinning (1/:math:`\lambda`)
   * - Yield stress
     - :math:`\sigma_y`
     - Pa
     - Stress to initiate flow
   * - Yasuda parameter
     - a
     - —
     - Transition sharpness (a = 2 gives Carreau)


Quick Start
-----------

**Herschel-Bulkley (yield stress fluid):**

.. code-block:: python

   from rheojax.models import HerschelBulkley
   import numpy as np

   model = HerschelBulkley()
   gamma_dot = np.logspace(-2, 2, 50)

   # Fit to flow curve data
   model.fit(gamma_dot, stress_data, test_mode='flow_curve')

   # Extract yield stress
   sigma_y = model.parameters.get_value('sigma_y')
   print(f"Yield stress: {sigma_y:.1f} Pa")

**Carreau model (full flow curve):**

.. code-block:: python

   from rheojax.models import Carreau

   model = Carreau()
   model.fit(gamma_dot, viscosity_data, test_mode='flow_curve')

   # Get zero-shear viscosity and critical shear rate
   eta_0 = model.parameters.get_value('eta_0')
   lambda_param = model.parameters.get_value('lambda')
   gamma_dot_c = 1 / lambda_param  # Critical shear rate

**Bayesian parameter estimation:**

.. code-block:: python

   # Bayesian inference with NLSQ warm-start
   result = model.fit_bayesian(
       gamma_dot, data,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Yield stress uncertainty
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"σ_y: [{intervals['sigma_y'][0]:.1f}, {intervals['sigma_y'][1]:.1f}] Pa")


Model Documentation
-------------------

**Simple Models:**

.. toctree::
   :maxdepth: 1

   power_law
   bingham

**Generalized Newtonian Models:**

.. toctree::
   :maxdepth: 1

   carreau
   carreau_yasuda
   cross

**Yield Stress Models:**

.. toctree::
   :maxdepth: 1

   herschel_bulkley


See Also
--------

- :doc:`/models/dmt/index` — Thixotropic models (time-dependent flow curves)
- :doc:`/models/ikh/index` — Kinematic hardening for complex yield behavior
- :doc:`/models/fluidity/index` — Fluidity-based yield stress models
- :doc:`/models/sgr/index` — Soft glassy rheology (power-law flow)
- :doc:`/transforms/srfs` — Strain-rate frequency superposition


References
----------

1. Bird, R.B., Armstrong, R.C., & Hassager, O. (1987). *Dynamics of Polymeric
   Liquids*, Vol. 1, 2nd ed. Wiley. ISBN: 978-0471802457.

2. Carreau, P.J. (1972). "Rheological equations from molecular network theories."
   *Trans. Soc. Rheol.*, 16, 99-127. https://doi.org/10.1122/1.549276

3. Cross, M.M. (1965). "Rheology of non-Newtonian fluids: A new flow equation
   for pseudoplastic systems." *J. Colloid Sci.*, 20, 417-437.

4. Herschel, W.H. & Bulkley, R. (1926). "Konsistenzmessungen von
   Gummi-Benzollösungen." *Kolloid-Z.*, 39, 291-300.

5. Yasuda, K., Armstrong, R.C., & Cohen, R.E. (1981). "Shear flow properties
   of concentrated solutions of linear and star branched polystyrenes."
   *Rheol. Acta*, 20, 163-178. https://doi.org/10.1007/BF01513059

6. Barnes, H.A., Hutton, J.F., & Walters, K. (1989). *An Introduction to
   Rheology*. Elsevier. ISBN: 978-0444871404.

7. Macosko, C.W. (1994). *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH. ISBN: 978-0471185758.
