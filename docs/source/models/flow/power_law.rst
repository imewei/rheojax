.. _model-power-law:

Power-Law (Ostwald–de Waele)
============================

Quick Reference
---------------

- **Use when:** Linear log-log flow curves, mid-range shear rates, quick characterization
- **Parameters:** 2 (:math:`K`, :math:`n`)
- **Key equation:** :math:`\sigma = K \dot{\gamma}^n`
- **Test modes:** Flow curve (Steady Shear)
- **Material examples:** Polymer melts, paints, shampoo, sauces, drilling fluids

Overview
--------

The **Power-Law** (or Ostwald–de Waele) model is the simplest and most widely used description of **non-Newtonian flow**. It assumes that shear stress scales as a power of shear rate. While it lacks the physical realism of identifying zero- and infinite-shear viscosity plateaus (unlike Carreau or Cross models), it provides an excellent empirical fit for the **intermediate shear rate region** where most processing and applications occur.

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\sigma`
     - Shear stress (Pa)
   * - :math:`\dot{\gamma}`
     - Shear rate (s\ :sup:`-1`)
   * - :math:`K`
     - Consistency index (Pa·s\ :sup:`n`). Viscosity magnitude at :math:`\dot{\gamma}=1`.
   * - :math:`n`
     - Flow index (dimensionless). Slope of log-log flow curve.
   * - :math:`\eta`
     - Apparent viscosity (Pa·s)

Physical Foundations
--------------------

Why "Power-Law"?
~~~~~~~~~~~~~~~~

For many complex fluids, the microscale structure reorganizes under flow in a way that creates a self-similar response. This leads to a scaling law:

.. math::
   \sigma \propto \dot{\gamma}^n \implies \log(\sigma) = n \log(\dot{\gamma}) + \log(K)

1.  **Shear-Thinning (** :math:`n < 1` **)**:
    *   **Microstructure**: Polymer chain alignment, disentanglement, or breakdown of particle aggregates.
    *   *Analogy*: "Traffic organizing into lanes" – resistance drops as flow speeds up.
2.  **Shear-Thickening (** :math:`n > 1` **)**:
    *   **Microstructure**: Hydrodynamic clustering, jamming, or formation of force chains (common in cornstarch/water).
    *   *Analogy*: "Crowd panic" – jamming occurs as everyone tries to move faster.

Limitations
~~~~~~~~~~~

The Power-Law has no intrinsic time scale and predicts **unphysical behavior** at extremes:
*   **Low Shear Limit**: :math:`\eta \to \infty` (for :math:`n<1`). Real fluids have a Newtonian plateau :math:`\eta_0`.
*   **High Shear Limit**: :math:`\eta \to 0` (for :math:`n<1`). Real fluids have a solvent plateau :math:`\eta_\infty`.

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

.. math::
   \sigma = K \dot{\gamma}^n

Apparent Viscosity
~~~~~~~~~~~~~~~~~~

.. math::
   \eta(\dot{\gamma}) = \frac{\sigma}{\dot{\gamma}} = K \dot{\gamma}^{n-1}

Parameters
----------

.. list-table:: Parameters
   :widths: 15 15 15 55
   :header-rows: 1

   * - Name
     - Symbol
     - Units
     - Description
   * - ``K``
     - :math:`K`
     - Pa·s\ :sup:`n`
     - **Consistency Index**. Measures the "thickness" of the fluid.
   * - ``n``
     - :math:`n`
     - -
     - **Flow Index**. :math:`n=1` (Newtonian), :math:`n<1` (Thinning), :math:`n>1` (Thickening).

Material Behavior Guide
-----------------------

.. list-table:: Typical Parameter Ranges
   :widths: 25 15 15 45
   :header-rows: 1

   * - Material Class
     - n
     - K (Pa·s\ :sup:`n`)
     - Notes
   * - **Polymer Melts**
     - 0.3 - 0.7
     - 1k - 50k
     - Strongly thinning in processing range.
   * - **Paints** (Latex)
     - 0.4 - 0.6
     - 10 - 100
     - Thinning for brush application.
   * - **Foods** (Sauces)
     - 0.2 - 0.5
     - 5 - 50
     - e.g., Ketchup, Mayo.
   * - **Dilute Solutions**
     - 0.8 - 0.95
     - 0.01 - 0.1
     - Weakly thinning.
   * - **Cornstarch/Water**
     - 1.5 - 2.0
     - 0.1 - 10
     - Shear thickening (dilatant).

Validity and Assumptions
------------------------

When the Power-Law Applies
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Power-Law model is valid when:

1. **Linear log-log region**: The :math:`\log(\eta)` vs :math:`\log(\dot{\gamma})` plot is
   linear over the shear rate range of interest.

2. **Mid-range shear rates**: Data span the power-law region, avoiding zero-shear
   and infinite-shear plateaus (typically 1–1000 s\ :sup:`-1` for most materials).

3. **Steady-state flow**: The material has reached equilibrium at each shear rate
   (no time-dependent effects like thixotropy).

4. **Isothermal conditions**: Temperature is constant throughout the measurement.

When to Use a Different Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection Guide
   :widths: 35 30 35
   :header-rows: 1

   * - Observation
     - Issue
     - Alternative Model
   * - Curvature at low :math:`\dot{\gamma}`
     - Zero-shear plateau visible
     - :doc:`carreau`, :doc:`cross`
   * - Curvature at high :math:`\dot{\gamma}`
     - Infinite-shear plateau
     - :doc:`carreau_yasuda`, :doc:`cross`
   * - Stress intercept at :math:`\dot{\gamma}=0`
     - Material has yield stress
     - :doc:`herschel_bulkley`, :doc:`bingham`
   * - Time-dependent response
     - Thixotropy/aging
     - Fluidity models, DMT

What You Can Learn
------------------

This section explains how to translate fitted Power-Law parameters into material
insights and actionable knowledge for both research and industrial applications.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Flow Index (n)**:
   The flow index reveals the degree of non-Newtonian behavior:

   - **n = 1.0**: Newtonian fluid with constant viscosity. The consistency index
     equals the Newtonian viscosity.

   - **0.5 < n < 1.0**: Mildly shear-thinning. Common in dilute polymer solutions
     where chain extension provides some alignment under flow.

   - **0.2 < n < 0.5**: Strongly shear-thinning. Indicates significant
     microstructural reorganization—polymer chain disentanglement, aggregate
     breakdown, or particle alignment.

   - **n < 0.2**: Extremely shear-thinning. Often seen in highly concentrated
     suspensions or systems with strong interparticle attractions.

   - **n > 1.0**: Shear-thickening (dilatant). Indicates hydrodynamic clustering,
     order-disorder transitions, or jamming phenomena.

   *For graduate students*: The flow index relates to microstructural dynamics.
   For polymer melts, :math:`n \approx 1/(1 + 2a)` where :math:`a` is the tube
   model constraint release parameter. For suspensions, :math:`n` decreases
   with increasing volume fraction as crowding amplifies thinning.

   *For practitioners*: Target :math:`n \approx 0.4-0.6` for brushable coatings
   (easy application, minimal dripping). For injection molding, lower :math:`n`
   reduces pressure drop in runners. Values :math:`n > 1` signal potential
   processing issues (e.g., die swell instability).

**Consistency Index (K)**:
   The consistency index sets the overall viscosity level:

   - **Physical meaning**: :math:`K` equals the apparent viscosity at
     :math:`\dot{\gamma} = 1` s\ :sup:`-1` (only for :math:`n=1`).

   - **Concentration dependence**: For polymer solutions,
     :math:`K \propto c^{[\eta]M_w}` where :math:`c` is concentration and
     :math:`[\eta]` is intrinsic viscosity.

   - **Temperature sensitivity**: :math:`K` follows Arrhenius behavior:
     :math:`K(T) = K_0 \exp(E_a/RT)` with activation energy :math:`E_a`.

   *For graduate students*: The consistency index encodes both molecular weight
   and concentration effects. For entangled polymers, :math:`K \propto M_w^{3.4}`
   following the reptation scaling. For suspensions, :math:`K` scales as
   :math:`\eta_s(1 - \phi/\phi_m)^{-2}` near the maximum packing fraction.

   *For practitioners*: Use :math:`K` for batch-to-batch QC. A 20% increase in
   :math:`K` at fixed :math:`n` suggests higher molecular weight or concentration.
   Temperature control is critical—a 10°C change can shift :math:`K` by 50%.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Power-Law Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Flow Index Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - :math:`n > 1.2`
     - Strong thickening
     - Dense cornstarch, silica in PEG
     - Mixing challenges, equipment damage risk
   * - :math:`1.0 < n < 1.2`
     - Mild thickening
     - Some particle suspensions
     - Careful rate control needed
   * - :math:`n = 1.0 \pm 0.05`
     - Newtonian
     - Simple fluids, dilute solutions
     - Standard process design
   * - :math:`0.5 < n < 1.0`
     - Mild thinning
     - Dilute polymer solutions
     - Good pumpability, moderate flow enhancement
   * - :math:`0.2 < n < 0.5`
     - Strong thinning
     - Melts, pastes, concentrated suspensions
     - Significant pressure reduction at high rates
   * - :math:`n < 0.2`
     - Extreme thinning
     - High-solid coatings, greases
     - Near-plug flow, yield-like behavior

Pipe Flow and Pumping Calculations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Power-Law enables analytical solutions for pressure-driven flow:

**Pressure Drop in Pipes**:

.. math::
   \Delta P = \left(\frac{3n+1}{n}\right)^n \frac{2K L}{R} \left(\frac{Q}{\pi R^3}\right)^n

where :math:`Q` is volumetric flow rate, :math:`L` is pipe length, and :math:`R`
is pipe radius.

**Velocity Profile**:

.. math::
   v(r) = \frac{n}{n+1} \left(\frac{\Delta P}{2KL}\right)^{1/n} \left(R^{(n+1)/n} - r^{(n+1)/n}\right)

- For :math:`n < 1`: Blunted profile (approaches plug flow as :math:`n \to 0`)
- For :math:`n = 1`: Parabolic (Newtonian)
- For :math:`n > 1`: More peaked profile

*For practitioners*: Shear-thinning fluids (:math:`n < 1`) require less pumping
power than equivalent Newtonian fluids. The power saving scales as
:math:`(3n+1)/(4n)` relative to Newtonian flow at the same flow rate.

Process Window Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

From fitted :math:`K` and :math:`n`, estimate operating conditions:

**Shear Rate from Viscosity Target**:

.. math::
   \dot{\gamma}_{target} = \left(\frac{\eta_{target}}{K}\right)^{1/(n-1)}

**Example**: For a coating with :math:`K = 50` Pa·s\ :sup:`n`, :math:`n = 0.5`,
requiring :math:`\eta = 0.5` Pa·s for spray application:

.. math::
   \dot{\gamma} = \left(\frac{0.5}{50}\right)^{1/(0.5-1)} = (0.01)^{-2} = 10{,}000 \text{ s}^{-1}

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- **n approaching 0**: Model may be masking yield stress behavior. Consider
  Herschel-Bulkley if residuals are systematic at low rates.

- **n > 1.5**: Rare for true shear thickening. Check for inertial artifacts
  (Taylor vortices above Re ≈ 1000) or slip at high rates.

- **K changes with shear rate range**: Power-law region not isolated. Narrow
  the fitting range to exclude plateaus.

- **Large confidence intervals on n**: Insufficient data points or narrow
  shear rate range. Expand measurement range by at least one decade.

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Quality Control**:
   Monitor :math:`K` at fixed :math:`n` for batch consistency. A control chart
   with ±10% limits on :math:`K` catches molecular weight or concentration drift.

**Process Optimization**:
   Use :math:`n` to optimize mixing. Strongly thinning materials (:math:`n < 0.3`)
   need high-shear impellers; mildly thinning materials (:math:`n > 0.7`) work
   with standard designs.

**Material Development**:
   During formulation, track how additives affect :math:`n`. Thickeners typically
   decrease :math:`n`; plasticizers may increase it. Target :math:`n` and :math:`K`
   values for desired application performance

Experimental Design
-------------------

The **Steady State Flow Curve** is the standard test:

1.  **Rate Sweep**: Logarithmic sweep of :math:`\dot{\gamma}` (e.g., 0.1 to 1000 s\ :sup:`-1`).
2.  **Equilibration**: Ensure steady state at each point (30-60s typical).
3.  **Visualization**: Plot :math:`\eta` vs :math:`\dot{\gamma}` on log-log axes.
    *   *Check*: Is it a straight line? If yes, Power-Law fits. If curved, use Carreau.

Fitting Guidance
----------------

Initialization
~~~~~~~~~~~~~~
*   **Log-Log Regression**: The best way to initialize.
    *   :math:`n` = slope of :math:`\log(\sigma)` vs :math:`\log(\dot{\gamma})`.
    *   :math:`K` = exponent of intercept (:math:`e^{\text{intercept}}`).

Optimization
~~~~~~~~~~~~

- **Bounds (recommended)**:
   - :math:`K`: [1e-6, 1e6] Pa·s\ :sup:`n`
   - :math:`n`: (0.01, 2.0)
- **Loss function**: Standard least squares suitable for mid-range data
- **Weighted fitting**: Optional weights to emphasize process-relevant shear rate range

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Cause
     - Solution
   * - Fit deviates at low rate
     - Zero-shear plateau (:math:`\eta_0`) reached
     - Truncate low-rate data or switch to :doc:`carreau` model
   * - Fit deviates at high rate
     - Infinite-shear plateau or instability
     - Truncate high-rate data or switch to :doc:`cross` model
   * - :math:`n > 1` unexpectedly
     - Inertia or Taylor vortices at high shear
     - Check Reynolds number; valid thickening is rare in simple fluids
   * - :math:`K` varies with test time
     - Thixotropy or evaporation
     - Use solvent trap; ensure steady state (no thixotropy loop)
   * - Large confidence intervals
     - Insufficient data range
     - Extend shear rate sweep by at least one decade
   * - Systematic residuals
     - Power-law region not isolated
     - Narrow fitting range to exclude plateaus

Usage
-----

Basic Fitting
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import PowerLaw
   from rheojax.core.data import RheoData

   # Steady shear flow curve data
   gamma_dot = jnp.array([0.1, 1, 10, 100, 1000])  # s^-1
   eta = jnp.array([500, 150, 45, 14, 4.5])  # Pa·s

   # Create model and fit
   model = PowerLaw()
   model.fit(gamma_dot, eta, test_mode='flow_curve')

   # Extract parameters
   K = model.parameters.get_value('K')  # Consistency index
   n = model.parameters.get_value('n')  # Flow index
   print(f"K = {K:.1f} Pa·s^n, n = {n:.3f}")

   # Predict viscosity at new shear rates
   gamma_dot_new = jnp.logspace(-1, 4, 50)
   eta_pred = model.predict(gamma_dot_new, test_mode='flow_curve')

Using RheoData
~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.data import RheoData

   # Load data with automatic test mode detection
   data = RheoData(x=gamma_dot, y=eta, test_mode='flow_curve')

   model = PowerLaw()
   model.fit(data)

   # Access fit quality
   print(f"R² = {model.r_squared:.4f}")

Bayesian Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import PowerLaw

   model = PowerLaw()
   model.fit(gamma_dot, eta, test_mode='flow_curve')  # NLSQ warm-start

   # Bayesian inference with uncertainty quantification
   result = model.fit_bayesian(
       gamma_dot, eta,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"K: {intervals['K']['mean']:.1f} [{intervals['K']['hdi_2.5%']:.1f}, {intervals['K']['hdi_97.5%']:.1f}]")
   print(f"n: {intervals['n']['mean']:.3f} [{intervals['n']['hdi_2.5%']:.3f}, {intervals['n']['hdi_97.5%']:.3f}]")

Pipeline Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Complete workflow from file to results
   (Pipeline()
       .load('flow_curve.csv', x_col='shear_rate', y_col='viscosity')
       .fit('power_law', test_mode='flow_curve')
       .plot(log_scale=True, title='Power-Law Fit')
       .save('results.hdf5'))

Temperature Dependence
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np

   # Fit at multiple temperatures
   temperatures = [25, 40, 60, 80]  # °C
   K_values = []

   for T, data in zip(temperatures, datasets):
       model = PowerLaw()
       model.fit(data)
       K_values.append(model.parameters.get_value('K'))

   # Arrhenius analysis: ln(K) vs 1/T
   T_kelvin = np.array(temperatures) + 273.15
   ln_K = np.log(K_values)

   # Fit for activation energy
   from scipy.stats import linregress
   slope, intercept, _, _, _ = linregress(1/T_kelvin, ln_K)
   E_a = -slope * 8.314  # J/mol
   print(f"Activation energy: {E_a/1000:.1f} kJ/mol")

Computational Implementation
----------------------------

JAX Vectorization
~~~~~~~~~~~~~~~~~

The Power-Law model is fully JIT-compiled for optimal performance:

.. code-block:: python

   from functools import partial
   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

   @partial(jax.jit, static_argnums=(2,))
   def power_law_viscosity(gamma_dot, params, n_points):
       K, n = params
       return K * gamma_dot ** (n - 1)

   # Vectorized over multiple datasets
   batched_predict = jax.vmap(power_law_viscosity, in_axes=(0, None, None))

Numerical Considerations
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Log-space fitting**: For numerical stability, the model internally works in
   log-space: :math:`\log(\eta) = \log(K) + (n-1)\log(\dot{\gamma})`.

2. **Bounds**: Default bounds are :math:`K \in [10^{-6}, 10^{6}]` and
   :math:`n \in [0.01, 3.0]` to ensure physical results.

3. **Initialization**: Smart initialization uses linear regression on log-log
   data, providing excellent starting points for optimization.

See Also
--------

Related Flow Models
~~~~~~~~~~~~~~~~~~~

- :doc:`carreau` — Adds zero-shear plateau; 4 parameters
- :doc:`cross` — Alternative transition function; 4 parameters
- :doc:`carreau_yasuda` — Extra shape parameter for transition sharpness; 5 parameters
- :doc:`herschel_bulkley` — Power-law with yield stress; 3 parameters
- :doc:`bingham` — Linear plastic with yield stress; 2 parameters

Transforms
~~~~~~~~~~

- :doc:`../../transforms/mastercurve` — Time-temperature superposition
- :doc:`../../transforms/srfs` — Strain-rate frequency superposition for flow curves

API Reference
~~~~~~~~~~~~~

- :class:`rheojax.models.PowerLaw`
- :class:`rheojax.core.data.RheoData`

References
----------

.. [1] Ostwald, W. "Über die Geschwindigkeitsfunktion der Viskosität disperser
   Systeme." *Kolloid-Zeitschrift*, 36, 99–117 (1925).
   https://doi.org/10.1007/BF01431449

.. [2] de Waele, A. "Viscometry and plastometry." *Journal of the Oil and
   Colour Chemists' Association*, 6, 33–69 (1923).

.. [3] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758

.. [4] Bird, R. B., Armstrong, R. C., and Hassager, O. *Dynamics of Polymeric
   Liquids, Volume 1: Fluid Mechanics*. 2nd ed., Wiley, New York (1987).
   ISBN: 978-0471802457

.. [5] Barnes, H. A., Hutton, J. F., and Walters, K. *An Introduction to
   Rheology*. Elsevier, Amsterdam (1989). ISBN: 978-0444871404

.. [6] Wilkinson, W. L. *Non-Newtonian Fluids: Fluid Mechanics, Mixing and
   Heat Transfer*. Pergamon Press, Oxford (1960).

.. [7] Skelland, A. H. P. *Non-Newtonian Flow and Heat Transfer*. Wiley,
   New York (1967).

.. [8] Chhabra, R. P., and Richardson, J. F. *Non-Newtonian Flow and Applied
   Rheology: Engineering Applications*. 2nd ed., Butterworth-Heinemann (2008).
   https://doi.org/10.1016/B978-0-7506-8532-0.X0001-7

.. [9] Steffe, J. F. *Rheological Methods in Food Process Engineering*. 2nd ed.,
   Freeman Press, East Lansing (1996). ISBN: 978-0963203618

.. [10] Rao, M. A. *Rheology of Fluid, Semisolid, and Solid Foods: Principles
   and Applications*. 3rd ed., Springer (2014).
   https://doi.org/10.1007/978-1-4614-9230-6
