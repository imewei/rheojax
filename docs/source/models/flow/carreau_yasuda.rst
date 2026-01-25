.. _model-carreau-yasuda:

Carreau–Yasuda Model
====================

Quick Reference
---------------

**Use when:** Abrupt viscosity transitions, sharp changes between plateaus
**Parameters:** 5 (η₀, η∞, λ, n, a)
**Key equation:** :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^{a}]^{(n-1)/a}`
**Test modes:** Flow (steady shear)
**Material examples:** Wormlike micelles, highly filled polymers, materials with sharp transitions

Overview
--------

The **Carreau-Yasuda** model extends the classical :doc:`carreau` model by introducing
an additional parameter :math:`a` that controls the sharpness of the transition between
the zero-shear plateau and the power-law region. This generalization was introduced by
Yasuda, Armstrong, and Cohen (1981) while studying concentrated polymer solutions.

The model is particularly valuable when:

1. The transition region is sharper or broader than the standard Carreau model predicts
2. Materials exhibit abrupt viscosity drops (e.g., wormlike micelles, associative polymers)
3. High-fidelity modeling of the transition curvature is required for process simulation

Setting :math:`a = 2` recovers the original Carreau model, while :math:`a < 2` produces
sharper transitions and :math:`a > 2` produces more gradual ones.

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\eta`
     - Apparent viscosity (Pa·s)
   * - :math:`\eta_0`
     - Zero-shear viscosity (Pa·s). Low-rate Newtonian plateau.
   * - :math:`\eta_\infty`
     - Infinite-shear viscosity (Pa·s). High-rate solvent contribution.
   * - :math:`\dot{\gamma}`
     - Shear rate (s\ :sup:`-1`)
   * - :math:`\lambda`
     - Relaxation time (s). Inverse of critical shear rate.
   * - :math:`n`
     - Power-law index (dimensionless). High-shear slope.
   * - :math:`a`
     - Yasuda exponent (dimensionless). Transition sharpness.

Physical Foundations
--------------------

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Yasuda exponent :math:`a` captures the **breadth of relaxation time distribution**:

- **Sharp transition (a < 2)**: Indicates a relatively narrow distribution of relaxation
  times. The material transitions rapidly from Newtonian to power-law behavior because
  most structural elements respond at similar time scales.

- **Gradual transition (a > 2)**: Suggests a broad distribution of relaxation times.
  Different structural elements (e.g., polymer chains of different lengths, aggregates
  of different sizes) begin thinning at different shear rates.

- **Carreau case (a = 2)**: The empirical default that works for many polymer solutions.

Connection to Polymer Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For polymer solutions, the Yasuda exponent relates to molecular architecture:

**Linear Polymers (Narrow MWD)**:
   Monodisperse or narrow-MWD linear polymers typically show :math:`a \approx 2`,
   consistent with the Carreau model. The single dominant relaxation mode creates
   a smooth but definite transition.

**Branched Polymers**:
   Long-chain branching introduces additional relaxation modes at longer times,
   often producing :math:`a > 2` due to the broadened spectrum.

**Wormlike Micelles**:
   These self-assembled structures can break and reform under flow, creating
   sharp transitions with :math:`a < 2`. The sudden onset of flow alignment
   produces an abrupt viscosity drop.

**Associative Polymers**:
   Polymers with sticky groups form transient networks. At critical shear rates,
   network disruption can cause sharp drops (:math:`a \approx 1-1.5`).

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

.. math::
   \eta(\dot{\gamma}) = \eta_{\infty} + (\eta_0 - \eta_{\infty})
       \left[1 + (\lambda \dot{\gamma})^{a}\right]^{\frac{n-1}{a}}

This form ensures:

- At :math:`\dot{\gamma} \to 0`: :math:`\eta \to \eta_0` (zero-shear plateau)
- At :math:`\dot{\gamma} \to \infty`: :math:`\eta \to \eta_\infty` (infinite-shear plateau)
- In the power-law region: :math:`\eta \propto \dot{\gamma}^{n-1}`

Limiting Cases
~~~~~~~~~~~~~~

**Low shear rate** (:math:`\lambda \dot{\gamma} \ll 1`):

.. math::
   \eta \approx \eta_0 - (\eta_0 - \eta_\infty) \frac{(n-1)}{a} (\lambda \dot{\gamma})^a

**High shear rate** (:math:`\lambda \dot{\gamma} \gg 1`):

.. math::
   \eta \approx \eta_\infty + (\eta_0 - \eta_\infty) (\lambda \dot{\gamma})^{n-1}

**Power-law approximation** (mid-range, :math:`\eta_\infty \approx 0`):

.. math::
   \eta \approx \eta_0 (\lambda)^{n-1} \dot{\gamma}^{n-1} = K \dot{\gamma}^{n-1}

where the effective consistency index is :math:`K = \eta_0 \lambda^{n-1}`.

Relation to Carreau Model
~~~~~~~~~~~~~~~~~~~~~~~~~

Setting :math:`a = 2`:

.. math::
   \eta = \eta_\infty + (\eta_0 - \eta_\infty) \left[1 + (\lambda \dot{\gamma})^2\right]^{\frac{n-1}{2}}

This is the standard Carreau form.

Parameters
----------

.. list-table:: Parameter Summary
   :header-rows: 1
   :widths: 15 15 15 55

   * - Name
     - Symbol
     - Units
     - Description / Constraints
   * - ``eta_0``
     - :math:`\eta_0`
     - Pa·s
     - Zero-shear viscosity. Must be > 0 and typically ≥ ``eta_inf``.
   * - ``eta_inf``
     - :math:`\eta_\infty`
     - Pa·s
     - Infinite-shear viscosity. Must be ≥ 0; often set to 0 when unmeasurable.
   * - ``lambda_``
     - :math:`\lambda`
     - s
     - Relaxation time. Inverse of critical shear rate where thinning begins.
   * - ``n``
     - :math:`n`
     - –
     - Power-law index. < 1 for thinning, = 1 for Newtonian, > 1 for thickening.
   * - ``a``
     - :math:`a`
     - –
     - Yasuda exponent. Controls transition sharpness; = 2 gives Carreau model.

Parameter Bounds
~~~~~~~~~~~~~~~~

.. list-table:: Default Bounds
   :header-rows: 1
   :widths: 20 30 50

   * - Parameter
     - Bounds
     - Physical Rationale
   * - :math:`\eta_0`
     - [1e-3, 1e7]
     - Must exceed solvent viscosity
   * - :math:`\eta_\infty`
     - [0, 1e4]
     - Cannot exceed :math:`\eta_0`; often ~solvent viscosity
   * - :math:`\lambda`
     - [1e-6, 1e4]
     - Must capture transition in measured range
   * - :math:`n`
     - [0.01, 2.0]
     - <0.01 unphysical; >2 rare (extreme thickening)
   * - :math:`a`
     - [0.1, 5.0]
     - <0.1 too sharp (numerical issues); >5 nearly Newtonian transition

Material Behavior Guide
-----------------------

.. list-table:: Typical Parameter Ranges
   :widths: 25 12 12 12 12 27
   :header-rows: 1

   * - Material Class
     - η₀ (Pa·s)
     - η∞ (Pa·s)
     - n
     - a
     - Notes
   * - **Wormlike Micelles**
     - 10–1000
     - 0.001–0.1
     - 0.1–0.4
     - 0.8–1.5
     - Sharp transition from network breakup
   * - **Associative Polymers**
     - 1–100
     - 0.01–1
     - 0.2–0.5
     - 1.0–1.8
     - HEUR, HASE thickeners
   * - **Concentrated Polymer Solutions**
     - 100–10000
     - 0.1–10
     - 0.3–0.6
     - 1.5–2.5
     - Narrow MWD: a ≈ 2
   * - **Branched Polymers**
     - 1000–100000
     - 1–100
     - 0.4–0.7
     - 2.0–3.5
     - Long-chain branching broadens transition
   * - **Highly Filled Systems**
     - 10–1000
     - 0.1–10
     - 0.2–0.5
     - 1.5–2.5
     - Particle alignment under shear
   * - **Blood/Biofluids**
     - 0.01–0.1
     - 0.003–0.005
     - 0.3–0.5
     - 2.0–2.5
     - RBC aggregation/deformation

Validity and Assumptions
------------------------

When Carreau-Yasuda is Appropriate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Carreau-Yasuda model when:

1. **Both plateaus are accessible**: Data span from zero-shear to near-infinite-shear
   plateaus, or at least show clear approach to both.

2. **Transition sharpness matters**: The standard Carreau model (:math:`a=2`) provides
   poor fits to the transition region.

3. **No yield stress**: The material flows freely at all stresses (no intercept at
   :math:`\dot{\gamma}=0`).

4. **Steady-state flow**: Time-independent response (no thixotropy or viscoelastic
   overshoot).

When to Use Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection Guide
   :widths: 35 30 35
   :header-rows: 1

   * - Observation
     - Issue
     - Better Model
   * - Transition fits well with :math:`a \approx 2`
     - Carreau-Yasuda overparameterized
     - :doc:`carreau` (4 parameters)
   * - No visible zero-shear plateau
     - η₀ unconstrained
     - :doc:`power_law` or :doc:`cross`
   * - Stress intercept at zero rate
     - Material has yield stress
     - :doc:`herschel_bulkley`
   * - Fitted :math:`a < 0.5`
     - Approaching step-function (unphysical)
     - Check data; consider yield stress model
   * - Strong parameter correlations
     - Data don't resolve all 5 parameters
     - :doc:`carreau` or reduce :math:`a` to fixed value

What You Can Learn
------------------

This section explains how to translate fitted Carreau-Yasuda parameters into material
insights and actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Yasuda Exponent (a)**:
   The Yasuda exponent reveals the breadth of relaxation time distribution:

   - **a < 1.5**: Sharp transition indicating a narrow relaxation spectrum. Common
     in wormlike micelles and associative polymers where cooperative structural
     breakdown occurs at a critical shear rate.

   - **a ≈ 2.0**: Standard Carreau behavior. Typical for well-characterized
     polymer solutions with moderate polydispersity.

   - **a > 2.5**: Broad transition suggesting wide relaxation time distribution.
     Common in branched polymers and materials with multiple structural components.

   *For graduate students*: The Yasuda exponent connects to the Cole-Davidson
   parameter in dielectric relaxation and the stretched exponential β in KWW
   relaxation. Lower :math:`a` corresponds to more exponential (single-mode)
   relaxation; higher :math:`a` corresponds to stretched relaxation.

   *For practitioners*: Sharp transitions (low :math:`a`) can cause processing
   instabilities. If :math:`a < 1.5`, consider whether sudden viscosity drops
   might cause flow instabilities or poor coating uniformity.

**Relaxation Time (λ)**:
   The relaxation time identifies the critical shear rate for structural response:

   - **Critical shear rate**: :math:`\dot{\gamma}_c = 1/\lambda` marks where
     viscosity begins significant departure from :math:`\eta_0`.

   - **Weissenberg number**: At :math:`Wi = \lambda \dot{\gamma} = 1`, elastic
     and viscous timescales balance.

   *For graduate students*: For entangled polymers, :math:`\lambda` scales with
   the terminal relaxation time :math:`\tau_d`, which in turn scales as
   :math:`\tau_d \propto M_w^{3.4}/c^{1.5}` (reptation theory).

   *For practitioners*: Compare :math:`\lambda` to process timescales. Coating
   at 100 s⁻¹ with :math:`\lambda = 0.1` s gives :math:`Wi = 10`—firmly in the
   power-law regime with good leveling.

**Viscosity Ratio (η₀/η∞)**:
   The ratio of plateau viscosities quantifies total thinning capacity:

   - **Small ratio (< 10)**: Mild thinning; limited shear-rate sensitivity
   - **Large ratio (> 1000)**: Strong thinning; dramatic viscosity reduction

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Carreau-Yasuda Parameters
   :header-rows: 1
   :widths: 25 25 25 25

   * - Parameter Pattern
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - Low a, high η₀/η∞
     - Sharp transition, breakable network
     - Wormlike micelles, associative gels
     - Shear-banding risk, flow instabilities
   * - a ≈ 2, moderate ratio
     - Standard polymer behavior
     - Linear polymer solutions, melts
     - Predictable processing, stable flow
   * - High a, high η₀
     - Broad relaxation spectrum
     - Branched polymers, blends
     - Wide processing window, forgiving
   * - n close to 1, any a
     - Weak shear-thinning
     - Dilute solutions, low MW
     - Near-Newtonian behavior, consider simpler model

Comparing Carreau vs Carreau-Yasuda Fits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit both models to your data and compare:

1. **AIC/BIC comparison**: If Carreau-Yasuda doesn't significantly improve fit
   statistics, use simpler Carreau model.

2. **Residual analysis**: Systematic residuals in transition region favor
   Carreau-Yasuda.

3. **Parameter uncertainty**: If :math:`a` has uncertainty >50%, data don't
   constrain it—use fixed :math:`a = 2` (Carreau).

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- **a approaching bounds**: If :math:`a < 0.5` or :math:`a > 4`, the model may
  be compensating for other issues (yield stress, data artifacts).

- **λ at measurement bounds**: If :math:`\lambda` equals 1/(max shear rate) or
  1/(min shear rate), the transition is outside your measurement window.

- **Strong a-λ correlation**: These parameters are inherently correlated. Consider
  fixing one based on literature or prior measurements.

- **η∞ > η₀**: Physically impossible. Check data for slip or inertia at high rates.

Experimental Design
-------------------

Recommended Protocol
~~~~~~~~~~~~~~~~~~~~

1. **Wide shear rate range**: Span at least 4 decades, ideally 6 (0.01–10,000 s⁻¹).

2. **Logarithmic spacing**: Use 10 points per decade for good resolution.

3. **Steady-state verification**: Allow 30–120 s equilibration per point, longer
   at low rates. Confirm by checking that viscosity doesn't drift.

4. **Bidirectional sweeps**: Run both up-ramp and down-ramp. Hysteresis indicates
   thixotropy or structure evolution.

5. **Temperature control**: ±0.1°C stability; viscosity changes ~3%/°C for polymers.

Geometry Selection
~~~~~~~~~~~~~~~~~~

.. list-table:: Recommended Geometries
   :header-rows: 1
   :widths: 25 25 50

   * - Shear Rate Range
     - Geometry
     - Notes
   * - 0.001–100 s⁻¹
     - Cone-plate (1–2°)
     - Uniform shear rate; best for low rates
   * - 0.1–1000 s⁻¹
     - Parallel plate
     - Adjustable gap; good for moderate rates
   * - 10–10,000 s⁻¹
     - Capillary
     - Best for high rates; requires Rabinowitsch correction
   * - Full range
     - Combine geometries
     - Stitch data from multiple tests

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Smart initialization dramatically improves convergence:

1. **From plateaus**:
   - :math:`\eta_0` = average viscosity at lowest 3 shear rates
   - :math:`\eta_\infty` = average viscosity at highest 3 shear rates (or 0)

2. **From transition**:
   - Find :math:`\dot{\gamma}_{1/2}` where :math:`\eta = (\eta_0 + \eta_\infty)/2`
   - Initialize :math:`\lambda = 1/\dot{\gamma}_{1/2}`

3. **From slope**:
   - :math:`n` = 1 + slope of log-log plot at high rates

4. **Default for a**:
   - Start with :math:`a = 2` (Carreau default)

Optimization Strategy
~~~~~~~~~~~~~~~~~~~~~

Two-stage fitting often works best:

**Stage 1**: Fix :math:`a = 2` and fit other 4 parameters (Carreau fit).

**Stage 2**: Release :math:`a` and refine all 5 parameters from Stage 1 solution.

.. code-block:: python

   from rheojax.models import CarreauYasuda

   model = CarreauYasuda()

   # Stage 1: Carreau fit (fixed a=2)
   model.parameters.set_value('a', 2.0)
   model.parameters.get_parameter('a').vary = False
   model.fit(gamma_dot, eta, test_mode='flow_curve')

   # Stage 2: Release a and refine
   model.parameters.get_parameter('a').vary = True
   model.fit(gamma_dot, eta, test_mode='flow_curve')

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Common Issues
   :widths: 25 35 40
   :header-rows: 1

   * - Symptom
     - Possible Cause
     - Solution
   * - a → 0 (lower bound)
     - Near-yield behavior
     - Try Herschel-Bulkley; check for yield stress
   * - a → upper bound
     - Effectively Newtonian
     - Use Carreau or simpler model
   * - λ poorly constrained
     - Transition outside data range
     - Extend shear rate range
   * - η∞ negative
     - Optimization artifact
     - Constrain η∞ ≥ 0; check high-rate data
   * - Strong a-λ correlation
     - Insufficient transition data
     - Fix a = 2 or increase mid-range points

Usage
-----

Basic Fitting
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

   from rheojax.models import CarreauYasuda
   from rheojax.core.data import RheoData

   # Flow curve data
   gamma_dot = jnp.logspace(-3, 4, 50)  # s^-1
   eta = jnp.array([...])  # Pa·s

   # Fit with default bounds
   model = CarreauYasuda()
   model.fit(gamma_dot, eta, test_mode='flow_curve')

   # Extract parameters
   print(f"eta_0 = {model.get_parameter('eta_0'):.1f} Pa·s")
   print(f"eta_inf = {model.get_parameter('eta_inf'):.3f} Pa·s")
   print(f"lambda = {model.get_parameter('lambda_'):.4f} s")
   print(f"n = {model.get_parameter('n'):.3f}")
   print(f"a = {model.get_parameter('a'):.2f}")

Two-Stage Fitting
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import CarreauYasuda

   model = CarreauYasuda()

   # Stage 1: Carreau fit (fix a=2)
   model.params.set_value('a', 2.0)
   model.params.get_parameter('a').vary = False
   model.fit(gamma_dot, eta, test_mode='flow_curve')

   # Check if Carreau is sufficient
   r2_carreau = model.r_squared
   print(f"Carreau R² = {r2_carreau:.5f}")

   # Stage 2: Full Carreau-Yasuda (release a)
   model.params.get_parameter('a').vary = True
   model.fit(gamma_dot, eta, test_mode='flow_curve')
   r2_cy = model.r_squared

   # Compare improvement
   delta_r2 = r2_cy - r2_carreau
   print(f"Carreau-Yasuda R² = {r2_cy:.5f}")
   print(f"Improvement: {delta_r2:.6f}")

   # If improvement < 0.001, Carreau is sufficient
   if delta_r2 < 0.001:
       print("Carreau model is adequate")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import CarreauYasuda

   model = CarreauYasuda()
   model.fit(gamma_dot, eta, test_mode='flow_curve')  # NLSQ warm-start

   result = model.fit_bayesian(
       gamma_dot, eta,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   intervals = model.get_credible_intervals(result.posterior_samples)
   for param in ['eta_0', 'eta_inf', 'lambda_', 'n', 'a']:
       ci = intervals[param]
       print(f"{param}: {ci['mean']:.3f} [{ci['hdi_2.5%']:.3f}, {ci['hdi_97.5%']:.3f}]")

Computational Implementation
----------------------------

JAX Vectorization
~~~~~~~~~~~~~~~~~

The model is fully JIT-compiled:

.. code-block:: python

   from functools import partial
   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()

   @partial(jax.jit, static_argnums=())
   def carreau_yasuda(gamma_dot, eta_0, eta_inf, lambda_, n, a):
       return eta_inf + (eta_0 - eta_inf) * (
           1 + (lambda_ * gamma_dot) ** a
       ) ** ((n - 1) / a)

Numerical Stability
~~~~~~~~~~~~~~~~~~~

1. **Exponent limiting**: The term :math:`(n-1)/a` is bounded to prevent overflow
   when :math:`a` is very small.

2. **Log-space fitting**: Internal optimization uses :math:`\log(\eta_0)`,
   :math:`\log(\eta_\infty)`, :math:`\log(\lambda)` for numerical stability.

3. **Gradient clipping**: JAX gradients are clipped to prevent NaN propagation

See Also
--------

Related Flow Models
~~~~~~~~~~~~~~~~~~~

- :doc:`carreau` — Special case with :math:`a = 2`; preferred when simpler model suffices
- :doc:`cross` — Alternative sigmoidal model with denominator exponent
- :doc:`power_law` — Simple power-law for mid-range rates only
- :doc:`herschel_bulkley` — For materials with yield stress

Transforms
~~~~~~~~~~

- :doc:`../../transforms/mastercurve` — Time-temperature superposition
- :doc:`../../transforms/srfs` — Strain-rate frequency superposition

API Reference
~~~~~~~~~~~~~

- :class:`rheojax.models.CarreauYasuda`
- :class:`rheojax.models.Carreau`

References
----------

.. [1] Yasuda, K., Armstrong, R. C., and Cohen, R. E. "Shear flow properties of
   concentrated solutions of linear and star-branched polystyrenes."
   *Rheologica Acta*, 20, 163–178 (1981).
   https://doi.org/10.1007/BF01513059

.. [2] Carreau, P. J. "Rheological equations from molecular network theories."
   *Transactions of the Society of Rheology*, 16, 99–127 (1972).
   https://doi.org/10.1122/1.549276

.. [3] Bird, R. B., Dotson, P. J., and Johnson, N. L. "Polymer solution rheology
   based on a finitely extensible bead-spring chain model."
   *Journal of Non-Newtonian Fluid Mechanics*, 7, 213–235 (1980).
   https://doi.org/10.1016/0377-0257(80)85007-5

.. [4] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758

.. [5] Larson, R. G. *Constitutive Equations for Polymer Melts and Solutions*.
   Butterworths, Boston (1988). ISBN: 978-0409901191

.. [6] Bird, R. B., Armstrong, R. C., and Hassager, O. *Dynamics of Polymeric
   Liquids, Volume 1: Fluid Mechanics*. 2nd ed., Wiley, New York (1987).
   ISBN: 978-0471802457

.. [7] Cates, M. E., and Candau, S. J. "Statics and dynamics of worm-like
   surfactant micelles." *Journal of Physics: Condensed Matter*, 2, 6869–6892 (1990).
   https://doi.org/10.1088/0953-8984/2/33/001

.. [8] Berret, J.-F. "Rheology of wormlike micelles: Equilibrium properties and
   shear banding transitions." In *Molecular Gels*, edited by R. G. Weiss and
   P. Terech, 667–720. Springer, Dordrecht (2006).
   https://doi.org/10.1007/1-4020-3689-2_20

.. [9] Barnes, H. A., Hutton, J. F., and Walters, K. *An Introduction to
   Rheology*. Elsevier, Amsterdam (1989). ISBN: 978-0444871404

.. [10] Mewis, J., and Wagner, N. J. *Colloidal Suspension Rheology*.
   Cambridge University Press (2012). ISBN: 978-0521515993

.. [11] Dealy, J. M., and Larson, R. G. *Structure and Rheology of Molten
   Polymers: From Structure to Flow Behavior and Back Again*. Hanser (2006).
   https://doi.org/10.3139/9783446412811
