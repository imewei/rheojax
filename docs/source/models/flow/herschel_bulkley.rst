.. _model-herschel-bulkley:

Herschel-Bulkley Model
======================

Quick Reference
---------------

**Use when:** Yield stress fluids (pastes, gels, foams), shear-thinning after yielding
**Parameters:** 3 (:math:`\tau_y`, :math:`K`, :math:`n`)
**Key equation:** :math:`\sigma = \sigma_y + K \dot{\gamma}^n` for :math:`\sigma > \sigma_y`
**Test modes:** Flow curve (Steady Shear), Stress Ramp
**Material examples:** Toothpaste, mayonnaise, drilling muds, fresh concrete, paints

Overview
--------

The **Herschel-Bulkley (HB)** model is the most generic and widely used constitutive equation for **yield stress fluids** that demonstrate non-Newtonian flow behavior after yielding. It generalizes the Bingham plastic model (which assumes linear post-yield flow) and the Power-law model (which assumes no yield stress), making it the standard choice for complex fluids like pastes, emulsions, foams, and slurries.

Key Characteristics:
   - **Yield Stress (** :math:`\sigma_y` **):** Material acts as a rigid solid below a critical stress.
   - **Consistency (** :math:`K` **):** Measures the viscous resistance to flow.
   - **Flow Index (** :math:`n` **):** Characterizes post-yield behavior (usually shear-thinning, :math:`n < 1`).

The model was introduced by Herschel and Bulkley in 1926 while studying rubber-benzene solutions and has since become the workhorse model for yield stress fluid characterization in industries ranging from food processing to oil drilling.

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
   * - :math:`\sigma_y`
     - Yield stress (Pa) - stress required to initiate flow
   * - :math:`K`
     - Consistency index (Pa·s\ :sup:`n`) - viscosity magnitude
   * - :math:`n`
     - Flow index (dimensionless) - slope of log-log flow curve
   * - :math:`\eta_{app}`
     - Apparent viscosity, :math:`\sigma / \dot{\gamma}` (Pa·s)

Physical Foundations
--------------------

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Herschel-Bulkley model describes materials with a **jammed microstructure** at rest that breaks down under flow:

1.  **Jammed State (at Rest)**:
    Particles, droplets, or bubbles form a volume-spanning network or glassy cage. Brownian motion is insufficient to break this structure.
    *   *Result*: Material behaves as an elastic solid (:math:`G' > G''`) for small stresses.

2.  **Yielding Transition** (:math:`\sigma \approx \sigma_y`):
    The applied stress exceeds the inter-particle attractive forces or cage strength. The structure "un-jams" or fractures.
    *   *Result*: Onset of irreversible flow.

3.  **Flowing State** (:math:`\sigma > \sigma_y`):
    The microstructure flows but retains interactions. Forces between particles lead to viscous dissipation.
    *   **Shear-Thinning (** :math:`n < 1` **)**: Most common. Structure aligns, organizes (e.g., lanes), or breaks down further as :math:`\dot{\gamma}` increases, reducing resistance.
    *   **Shear-Thickening (** :math:`n > 1` **)**: Rare for simple yield stress fluids (usually seen in dense suspensions at high rates).

Governing Equations
-------------------

Stress-Strain Rate Relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \sigma =
   \begin{cases}
   \sigma_y + K \dot{\gamma}^n & \text{if } \sigma > \sigma_y \\
   \dot{\gamma} = 0 & \text{if } \sigma \le \sigma_y
   \end{cases}

Apparent Viscosity
~~~~~~~~~~~~~~~~~~

The viscosity is not constant but depends on shear rate:

.. math::
   \eta(\dot{\gamma}) = \frac{\sigma}{\dot{\gamma}} = \frac{\sigma_y}{\dot{\gamma}} + K \dot{\gamma}^{n-1}

*   **Low shear limit**: :math:`\eta \to \infty` as :math:`\dot{\gamma} \to 0` (infinite viscosity at rest).
*   **High shear limit**: :math:`\eta \to K \dot{\gamma}^{n-1}` (approaches power-law behavior).

Parameters
----------

.. list-table:: Parameters
   :widths: 15 15 15 55
   :header-rows: 1

   * - Name
     - Symbol
     - Units
     - Description
   * - ``tau_y``
     - :math:`\sigma_y`
     - Pa
     - **Yield Stress**. Critical stress for flow. High :math:`\sigma_y` means "stiff" paste.
   * - ``K``
     - :math:`K`
     - Pa·s\ :sup:`n`
     - **Consistency**. Viscosity scale. Note units depend on :math:`n`.
   * - ``n``
     - :math:`n`
     - -
     - **Flow Index**. :math:`n<1` (thinning), :math:`n=1` (Bingham), :math:`n>1` (thickening).

Material Behavior Guide
-----------------------

.. list-table:: Typical Parameter Ranges by Material Class
   :widths: 25 15 15 12 33
   :header-rows: 1

   * - Material Class
     - σ_y (Pa)
     - K (Pa·s\ :sup:`n`)
     - n
     - Notes
   * - **Mayonnaise**
     - 50–200
     - 5–30
     - 0.3–0.5
     - Highly shear-thinning emulsion
   * - **Toothpaste**
     - 100–300
     - 10–50
     - 0.2–0.4
     - Stiff paste with strong thinning
   * - **Drilling Mud**
     - 5–50
     - 0.5–5
     - 0.4–0.7
     - Bentonite suspensions
   * - **Fresh Concrete**
     - 10–200
     - 50–500
     - 0.2–0.5
     - Self-compacting has lower σ_y
   * - **Ketchup**
     - 10–50
     - 5–20
     - 0.3–0.5
     - Lower yield than mayo
   * - **Cosmetic Cream**
     - 20–100
     - 2–20
     - 0.3–0.6
     - O/W or W/O emulsions
   * - **Food Purees**
     - 5–50
     - 2–15
     - 0.2–0.4
     - Fruit/vegetable pastes
   * - **Foam (Shaving)**
     - 20–100
     - 1–10
     - 0.2–0.4
     - Gas-liquid system
   * - **Waxy Crude Oil**
     - 1–100
     - 0.1–10
     - 0.5–0.9
     - Temperature-dependent wax network

Validity and Assumptions
------------------------

When Herschel-Bulkley Applies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The HB model is appropriate when:

1. **Clear yield stress exists**: Material does not flow below a critical stress.
   The stress-strain rate curve shows a stress intercept at zero rate.

2. **Post-yield power-law behavior**: After yielding, the material follows
   :math:`\sigma - \sigma_y = K \dot{\gamma}^n` over the measured range.

3. **Steady-state flow**: Material reaches equilibrium at each shear rate
   (no thixotropy or aging during measurement).

4. **No slip at walls**: The material shears uniformly without wall slip.

When to Use Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection
   :widths: 35 30 35
   :header-rows: 1

   * - Observation
     - Issue
     - Better Model
   * - Fitted n ≈ 1 (within ±0.1)
     - Newtonian post-yield
     - :doc:`bingham` (simpler)
   * - Fitted σ_y ≈ 0
     - No yield stress
     - :doc:`power_law` or :doc:`carreau`
   * - Thixotropic hysteresis
     - Time-dependent structure
     - Fluidity models, DMT
   * - Stress overshoot in startup
     - Viscoelastic effects
     - Saramito EVP, SGR

What You Can Learn
------------------

This section explains how to translate fitted Herschel-Bulkley parameters into
material insights and actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Yield Stress (σ_y)**:
   The yield stress reveals the strength of the material's microstructural network:

   - **σ_y < 10 Pa**: Weak network. Material will flow under its own weight or
     light handling. Common in dilute suspensions and thin gels.

   - **10 < σ_y < 100 Pa**: Moderate network. Material holds shape against gravity
     but yields to reasonable forces. Typical for most commercial pastes.

   - **σ_y > 100 Pa**: Strong network. Requires significant force to initiate
     flow. Common in stiff pastes like toothpaste and cement.

   *For graduate students*: The yield stress scales with microstructural
   parameters. For colloidal gels, :math:`\sigma_y \propto \phi^m G_0` where
   :math:`\phi` is volume fraction, :math:`m \approx 2-4` depends on network
   structure, and :math:`G_0` is the elastic modulus. For emulsions,
   :math:`\sigma_y \propto \gamma/R \cdot (\phi - \phi_c)^2` where :math:`\gamma`
   is interfacial tension and :math:`R` is droplet radius.

   *For practitioners*: Use :math:`\sigma_y` for packaging and dispensing design.
   A mayonnaise with :math:`\sigma_y = 80` Pa needs approximately 80 Pa of
   shear stress to flow from a squeeze bottle. For vertical surfaces, material
   thickness :math:`h` should satisfy :math:`h < \sigma_y / (\rho g)` to prevent
   sagging.

**Consistency Index (K)**:
   The consistency governs the viscous response after yielding:

   - **Low K (< 5 Pa·s^n)**: Thin flow once yielded. Good pumpability but may
     spray or splash.

   - **Moderate K (5–50 Pa·s^n)**: Balanced viscous resistance. Typical for
     controlled spreading.

   - **High K (> 50 Pa·s^n)**: Thick flow requiring sustained energy input.
     Common in stiff mortars and heavy pastes.

   *For graduate students*: For concentrated suspensions above the yield stress,
   :math:`K` reflects hydrodynamic interactions between particles. It scales
   approximately as :math:`K \propto \eta_s (1 - \phi/\phi_m)^{-2.5}` where
   :math:`\eta_s` is solvent viscosity.

   *For practitioners*: The pumping power in the post-yield regime scales with
   :math:`K`. Reducing particle concentration or adding dispersant lowers
   :math:`K` and pumping costs.

**Flow Index (n)**:
   The flow index characterizes the degree of post-yield shear-thinning:

   - **n ≈ 1.0**: Bingham-like. Post-yield viscosity is constant (linear flow
     curve above yield).

   - **0.5 < n < 1.0**: Mild thinning. Common in dilute systems or materials
     with weak interparticle attractions.

   - **0.2 < n < 0.5**: Strong thinning. Indicates significant microstructural
     breakdown with increasing shear. Common in concentrated emulsions and
     pastes.

   - **n < 0.2**: Extreme thinning. May indicate a near-critical system
     (approaching glass or jamming transition). Check data quality.

   *For graduate students*: For soft glassy materials, the SGR model predicts
   :math:`n = x - 1` where :math:`x` is the noise temperature. Materials near
   the glass transition (:math:`x \to 1`) show :math:`n \to 0`. The flow index
   also connects to the Cole-Cole distribution width for polydisperse relaxation.

   *For practitioners*: Lower :math:`n` means the material "thins out" more
   dramatically at high shear rates. This aids mixing and pumping but may
   cause coating non-uniformity as the material levels differently at different
   rates.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: HB Material Classification
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Pattern
     - Material Behavior
     - Typical Materials
     - Process Implications
   * - High σ_y, low n
     - Stiff, thinning paste
     - Toothpaste, grease
     - Extrusion-based processing
   * - Moderate σ_y, n ≈ 0.5
     - Standard paste
     - Mayo, lotions
     - Conventional pumping/mixing
   * - Low σ_y, n close to 1
     - Near-Bingham
     - Thin suspensions
     - Consider Bingham model
   * - High σ_y, n close to 1
     - Stiff plastic
     - Cement, clay
     - High-pressure extrusion

Dimensional Analysis
~~~~~~~~~~~~~~~~~~~~

The **Oldroyd number** (Od) characterizes flow regime:

.. math::
   Od = \frac{\sigma_y}{K} \left(\frac{L}{U}\right)^n

where :math:`L` is length scale and :math:`U` is velocity. Large Od means
yield-stress-dominated flow (plug flow); small Od means power-law dominated.

For pipe flow, the fraction of unyielded material (plug) is:

.. math::
   \frac{r_{plug}}{R} = \frac{\sigma_y}{\sigma_{wall}} = \frac{\sigma_y}{\Delta P R / (2L)}

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- **σ_y → 0 or negative**: Data may not have a true yield stress. Consider
  Power-Law or Carreau model. Check that low-rate data is reliable.

- **n > 1**: Shear-thickening post-yield is rare. Check for inertial artifacts,
  Taylor vortices, or slip at high rates.

- **K very small with high σ_y**: Unusual combination. Check units and data
  scaling.

- **Strong parameter correlations**: Especially σ_y–K correlation. Extend
  measurement range; ensure data spans transition region well.

- **Systematic residuals at low rates**: May indicate wall slip or viscoelastic
  creep below yield.

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Food Product Design**:
   For spreadable products, target :math:`\sigma_y \approx 30-80` Pa (easy
   spreading but no dripping). Track :math:`n` to ensure consistent texture—
   lower :math:`n` gives more "slip" on the knife.

**Drilling Fluid Optimization**:
   Target :math:`\sigma_y` high enough to suspend cuttings (typically 5–15 Pa)
   with :math:`n` low enough for easy circulation. The API specifies 6 rpm and
   300 rpm readings for HB parameter estimation.

**Concrete Mix Design**:
   Self-compacting concrete requires :math:`\sigma_y < 60` Pa for gravity-driven
   flow. Standard concrete has :math:`\sigma_y \approx 100-200` Pa requiring
   vibration for compaction.

**Cosmetic Formulation**:
   Body lotions need :math:`\sigma_y \approx 20-50` Pa for good dispensing.
   Track :math:`n` to ensure smooth spreading—values around 0.4-0.5 give
   pleasant sensory properties.

Experimental Design
-------------------

Recommended Test Modes
~~~~~~~~~~~~~~~~~~~~~~

1.  **Steady State Flow Curve (Step-Rate)**:
    *   **Protocol**: Apply range of :math:`\dot{\gamma}` (e.g., :math:`10^{-3}` to :math:`10^2` s\ :sup:`-1`), measure :math:`\sigma`.
    *   **Duration**: Allow steady state at each point (crucial for thixotropic materials).
    *   **Best for**: Accurate parameter fitting over wide range.

2.  **Stress Ramp**:
    *   **Protocol**: Linear ramp of :math:`\sigma` from 0 to :math:`>\sigma_y`.
    *   **Best for**: Precise determination of :math:`\sigma_y` (observe sudden strain rate jump).

Experimental Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Wall Slip**: Common in pastes/gels. Material slips at geometry wall instead of flowing.
    *   *Symptom*: Apparent "kink" in flow curve, or lower viscosity than expected.
    *   *Fix*: Use sandpaper/serrated plates or vane geometry.
*   **Thixotropy**: Time-dependent breakdown.
    *   *Check*: Perform hysteresis loop (ramp up, then ramp down). If curves differ, material is thixotropic. Use steady-state averaging to fit equilibrium HB model.
*   **Geometry**:
    *   **Cone-Plate**: Constant shear rate (preferred).
    *   **Parallel Plate**: Shear rate gradient (requires correction, but better for varying gaps/slip).
    *   **Vane**: Best for preventing slip in yield stress fluids.

Fitting Guidance
----------------

Initialization
~~~~~~~~~~~~~~

1.  **Estimate Yield Stress (** :math:`\sigma_y` **)**:
    Extrapolate the low-shear stress plateau to :math:`\dot{\gamma} = 0`, or take the stress at the lowest measured rate.
2.  **Estimate Power-Law Parameters (** :math:`K, n` **)**:
    Plot :math:`(\sigma - \sigma_y)` vs :math:`\dot{\gamma}` on log-log scale.
    *   Slope = :math:`n`
    *   Intercept (at :math:`\dot{\gamma}=1`) = :math:`K`

Troubleshooting Fitting Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Fitting Diagnostics
   :widths: 25 35 40
   :header-rows: 1

   * - Symptom
     - Possible Cause
     - Solution
   * - **Negative Yield Stress**
     - Data shows Newtonian plateau at low shear (no yield)
     - Switch to **Cross** or **Carreau** model (pseudoplastic with zero-shear viscosity).
   * - **Fit passes below data** at high shear
     - Shear thickening onset or Taylor vortices
     - Restrict fit range to laminar region (remove high :math:`\dot{\gamma}` points).
   * - **Poor fit at low shear**
     - Wall slip or incomplete yielding
     - Check for slip (serrated plates). Down-weight low-shear points if noisy.
   * - **n close to 1**
     - Material is Bingham Plastic
     - Simplify to **Bingham** model (:math:`n=1`) for robustness.

Model Comparison
----------------

*   **Bingham**: HB with :math:`n=1`. Simpler, assumes constant post-yield viscosity.
*   **Power Law**: HB with :math:`\sigma_y = 0`. No yield stress.
*   **Casson**: Alternative yield stress model (:math:`\sqrt{\sigma} = \sqrt{\sigma_y} + \sqrt{\eta \dot{\gamma}}`), mainly for blood/chocolate.
*   **Carreau**: No yield stress, but finite zero-shear viscosity. Better for polymer melts/solutions.

Usage
-----

Basic Fitting
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import HerschelBulkley
   from rheojax.core.data import RheoData

   # Flow curve data (stress vs shear rate)
   gamma_dot = jnp.logspace(-2, 2, 50)  # s^-1
   # Example data for toothpaste
   sigma = jnp.array([120.5, 125.3, 135.2, 148.7, 165.3, 185.2])  # Pa (measured stress)

   # Fit the model
   model = HerschelBulkley()
   model.fit(gamma_dot[:6], sigma, test_mode='flow_curve')

   # Extract parameters
   tau_y = model.parameters.get_value('tau_y')
   K = model.parameters.get_value('K')
   n = model.parameters.get_value('n')
   print(f"σ_y = {tau_y:.1f} Pa, K = {K:.2f} Pa·s^n, n = {n:.3f}")

With Custom Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HerschelBulkley

   # Initialize with estimates from data inspection
   model = HerschelBulkley()
   model.set_parameter('tau_y', 50.0)  # From low-rate plateau
   model.set_parameter('K', 10.0)
   model.set_parameter('n', 0.4)

   # Constrain n for shear-thinning only
   model.set_parameter_bounds('n', lower=0.1, upper=1.0)

   model.fit(gamma_dot, sigma, test_mode='flow_curve')

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HerschelBulkley

   model = HerschelBulkley()
   model.fit(gamma_dot, sigma, test_mode='flow_curve')  # NLSQ warm-start

   # Bayesian inference with uncertainty quantification
   result = model.fit_bayesian(
       gamma_dot, sigma,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"σ_y: {intervals['tau_y']['mean']:.1f} "
         f"[{intervals['tau_y']['hdi_2.5%']:.1f}, {intervals['tau_y']['hdi_97.5%']:.1f}] Pa")

Comparing with Bingham
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import HerschelBulkley, Bingham

   # Fit both models
   hb = HerschelBulkley()
   hb.fit(gamma_dot, sigma, test_mode='flow_curve')

   bingham = Bingham()
   bingham.fit(gamma_dot, sigma, test_mode='flow_curve')

   # Compare fit quality
   print(f"HB R² = {hb.r_squared:.5f}")
   print(f"Bingham R² = {bingham.r_squared:.5f}")
   print(f"HB n = {hb.get_parameter('n'):.3f}")

   # If n ≈ 1 and R² similar, use Bingham for parsimony
   if abs(hb.get_parameter('n') - 1.0) < 0.1:
       print("Consider using simpler Bingham model")

Pipeline Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline

   (Pipeline()
       .load('flow_curve.csv', x_col='shear_rate', y_col='stress')
       .fit('herschel_bulkley', test_mode='flow_curve')
       .plot(log_scale=True, title='Herschel-Bulkley Fit')
       .save('results.hdf5'))

Computational Implementation
----------------------------

JAX Vectorization
~~~~~~~~~~~~~~~~~

The model uses JIT-compiled evaluation:

.. code-block:: python

   from functools import partial
   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

   @partial(jax.jit, static_argnums=())
   def herschel_bulkley_stress(gamma_dot, tau_y, K, n):
       """Compute stress for yielded flow (γ̇ > 0)."""
       return tau_y + K * jnp.power(gamma_dot, n)

Regularization for Numerics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The true HB model has a discontinuity at yield. For numerical stability,
RheoJAX uses a regularized form:

.. math::
   \sigma = \sigma_y \left(1 - e^{-m \dot{\gamma}}\right) + K \dot{\gamma}^n

where :math:`m` is a large regularization parameter (default :math:`m = 10^6`).
This produces smooth gradients while matching the true HB to high precision
for :math:`\dot{\gamma} > 10^{-4}` s\ :sup:`-1`.

See Also
--------

Related Flow Models
~~~~~~~~~~~~~~~~~~~

- :doc:`bingham` — Special case with n = 1 (linear post-yield)
- :doc:`power_law` — Special case with σ_y = 0 (no yield stress)
- :doc:`carreau` — For materials without yield stress
- :doc:`cross` — Alternative generalized Newtonian model

Advanced Yield Stress Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- :doc:`../sgr/sgr_conventional` — Soft glassy rheology for pastes near glass transition
- :doc:`../fluidity/saramito_evp` — Elastoviscoplastic with viscoelastic pre-yield
- :doc:`../stz/stz_conventional` — Shear transformation zone model

Transforms
~~~~~~~~~~

- :doc:`../../transforms/mastercurve` — Temperature superposition for HB materials
- :doc:`../../transforms/srfs` — Strain-rate frequency superposition

API Reference
~~~~~~~~~~~~~

- :class:`rheojax.models.HerschelBulkley`
- :class:`rheojax.models.Bingham`

References
----------

.. [1] Herschel, W. H., and Bulkley, R. "Konsistenzmessungen von Gummi-Benzollösungen."
   *Kolloid-Zeitschrift*, 39, 291–300 (1926).
   https://doi.org/10.1007/BF01432034

.. [2] Barnes, H. A. "The yield stress—a review or 'παντα ρει'—everything flows?"
   *Journal of Non-Newtonian Fluid Mechanics*, 81, 133–178 (1999).
   https://doi.org/10.1016/S0377-0257(98)00094-9

.. [3] Coussot, P. "Yield stress fluid flows: A review of experimental data."
   *Journal of Non-Newtonian Fluid Mechanics*, 211, 31–49 (2014).
   https://doi.org/10.1016/j.jnnfm.2014.05.006

.. [4] Bird, R. B., Dai, G. C., and Yarusso, B. J. "The rheology and flow of
   viscoplastic materials." *Reviews in Chemical Engineering*, 1, 1–70 (1983).
   https://doi.org/10.1515/revce-1983-0102

.. [5] Balmforth, N. J., Frigaard, I. A., and Ovarlez, G. "Yielding to stress:
   Recent developments in viscoplastic fluid mechanics."
   *Annual Review of Fluid Mechanics*, 46, 121–146 (2014).
   https://doi.org/10.1146/annurev-fluid-010313-141424

.. [6] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758

.. [7] Larson, R. G. *The Structure and Rheology of Complex Fluids*.
   Oxford University Press, New York (1999). ISBN: 978-0195121971

.. [8] Coussot, P. *Rheometry of Pastes, Suspensions, and Granular Materials:
   Applications in Industry and Environment*. Wiley (2005).
   https://doi.org/10.1002/0471720577

.. [9] Mewis, J., and Wagner, N. J. *Colloidal Suspension Rheology*.
   Cambridge University Press (2012). ISBN: 978-0521515993

.. [10] Møller, P. C. F., Mewis, J., and Bonn, D. "Yield stress and thixotropy:
   On the difficulty of measuring yield stresses in practice."
   *Soft Matter*, 2, 274–283 (2006).
   https://doi.org/10.1039/b517840a

.. [11] Ovarlez, G., Cohen-Addad, S., Krishan, K., Goyon, J., and Coussot, P.
   "On the existence of a simple yield stress fluid behavior."
   *Journal of Non-Newtonian Fluid Mechanics*, 193, 68–79 (2013).
   https://doi.org/10.1016/j.jnnfm.2012.06.009

.. [12] Bonnecaze, R. T., and Cloitre, M. "Micromechanics of soft particle
   glasses." *Advances in Polymer Science*, 236, 117–161 (2010).
   https://doi.org/10.1007/12_2010_90
