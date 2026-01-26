.. _model-fractional-zener-sl:

Fractional Zener Solid-Liquid (Fractional)
==========================================

Quick Reference
---------------

- **Use when:** Solid-like behavior with equilibrium plateau and fractional relaxation tails
- **Parameters:** 4 (Ge, :math:`c_{\alpha, \alpha, \tau}`)
- **Key equation:** :math:`G(t) = G_e + c_\alpha t^{-\alpha} E_{1-\alpha,1}(-(t/\tau)^{1-\alpha})`
- **Test modes:** Oscillation, relaxation
- **Material examples:** Viscoelastic solids with finite equilibrium modulus and power-law relaxation

.. include:: /_includes/fractional_seealso.rst

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G_e`
     - Pa
     - Equilibrium modulus (parallel spring, long-time plateau)
   * - :math:`c_\alpha`
     - Pa·s\ :math:`^{\alpha}`
     - SpringPot constant (relaxation magnitude)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < :math:`\alpha` < 1, power-law character)
   * - :math:`\tau`
     - s
     - Relaxation time (characteristic timescale)
   * - :math:`E_{1-\alpha,1}(z)`
     - dimensionless
     - Two-parameter Mittag-Leffler function

Overview
--------
Fractional Maxwell element in parallel with a spring to capture solid-like plateaus with fractional relaxation tails.

Physical Foundations
--------------------

The Fractional Zener Solid-Liquid (FZSL) combines a fractional Maxwell arm with
a parallel equilibrium spring:

**Mechanical Configuration:**

.. code-block:: text

   [Spring Ge] ---- parallel ---- [SpringPot (c_α, α) in series with relaxation τ]

**Microstructural Interpretation:**

- **Parallel spring (Ge)**: Permanent network providing equilibrium modulus
  (crosslinks, crystalline regions, entanglements)
- **Fractional Maxwell arm**: Additional stiffness that relaxes via power-law
  dynamics. The SpringPot creates broad relaxation spectrum.
- **Solid behavior**: Finite equilibrium modulus Ge (no flow)
- **Relaxation**: Material relaxes from Ge + (high-freq contribution) to Ge

This model bridges the gap between FZSS (two elastic plateaus) and FMG (liquid-like).
It's useful when the material has a clear equilibrium modulus but shows power-law
relaxation dynamics.

Governing Equations
-------------------
Time domain (relaxation modulus):

.. math::
   :nowrap:

   \[
   G(t) = G_e + c_\alpha t^{-\alpha} E_{1-\alpha,1}\left(-\left(\frac{t}{\tau}\right)^{1-\alpha}\right).
   \]

Frequency domain (complex modulus):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) = G_e + \frac{c_\alpha (i\omega)^{\alpha}}{1 + (i\omega\tau)^{1-\alpha}}.
   \]

Parameters
----------
.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 12 12 18 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``Ge``
     - :math:`G_e`
     - Pa
     - [1e-3, 1e9]
     - Equilibrium modulus
   * - ``c_alpha``
     - :math:`c_\alpha`
     - Pa*s^alpha
     - [1e-3, 1e9]
     - SpringPot constant
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e+6]
     - Relaxation time

Validity and Assumptions
------------------------
- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------
- Low-frequency limit recovers the equilibrium modulus :math:`G_e`.
- Mid-band shows fractional dissipation with slope :math:`\alpha`.
- High-frequency response approaches :math:`G_e + c_\alpha (i\omega)^{\alpha}`.

Limiting Behavior
-----------------
- :math:`\alpha \to 1`: classical Zener solid-liquid.
- :math:`G_e \to 0`: fractional Maxwell gel.
- :math:`c_\alpha \to 0`: pure elastic spring.

What You Can Learn
------------------

This section explains what insights you can extract from fitting the Fractional Zener Solid-Liquid model to your experimental data, emphasizing the dual-modulus solid structure with one solid springpot (:math:`\alpha`) and one liquid springpot (1-:math:`\alpha`).

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Equilibrium Modulus (Ge)**:
   The low-frequency plateau modulus, indicating the material's long-term stiffness under sustained load. This arises from the parallel spring providing permanent network structure.

   *For graduate students*: Ge relates to crosslink density via rubber elasticity theory: Ge ≈ :math:`\nu \cdot kB \cdot T where \nu` is network strand density. For chemically crosslinked networks, Ge is temperature-independent; for physical networks (entanglements), Ge scales with T.
   *For practitioners*: Higher Ge means stiffer equilibrium behavior. Compare to target specifications for structural applications. If Ge → 0, material flows—consider FMG instead.

**Fractional Order (** :math:`\alpha` **)**:
   Controls the breadth of relaxation spectrum and power-law decay character. This model uses a "solid" springpot (:math:`\alpha`) paired with a "liquid" springpot (1-:math:`\alpha`) in series.

   - :math:`\alpha` **→ 0**: Very broad spectrum, solid-like response dominates, nearly elastic at short times
   - :math:`\alpha` **→ 0.5**: Critical gel behavior, balanced solid-liquid character, maximum spectrum breadth
   - :math:`\alpha` **→ 1**: Narrow spectrum, approaches classical Zener with exponential relaxation

   *For graduate students*: :math:`\alpha` quantifies polydispersity in the relaxation time distribution. Lower :math:`\alpha` indicates greater microstructural heterogeneity (filler dispersion, crosslink density variation, molecular weight distribution).
   *For practitioners*: Lower :math:`\alpha` means relaxation spreads over more time decades. Critical for predicting long-term creep and stress relaxation.

**SpringPot Constant (** :math:`c_{\alpha}` **)**:
   Sets the magnitude of the fractional dissipation contribution from the Maxwell arm. Units are Pa·s\ :math:`^{\alpha}`.

   - **High** :math:`c_{\alpha/Ge}` **ratio (> 5)**: Strong viscoelastic dissipation, large relaxation from high-frequency to Ge
   - **Moderate** :math:`c_{\alpha/Ge}` **ratio (1-5)**: Balanced elastic-dissipative response
   - **Low** :math:`c_{\alpha/Ge}` **ratio (< 1)**: Predominantly elastic response, small relaxation

   *For graduate students*: :math:`c_{\alpha}` represents the spectral strength of the relaxing modes. Higher :math:`c_{\alpha}` indicates more energy stored in temporary (relaxing) structures.
   *For practitioners*: High :math:`c_{\alpha/Ge}` means large difference between short-time and long-time stiffness—critical for impact vs. sustained loading.

**Relaxation Time (** :math:`\tau` **)**:
   Characteristic timescale for the transition from high modulus (Ge + high-frequency contribution) to equilibrium modulus Ge.

   *For graduate students*: :math:`\tau` is temperature-dependent (WLF/Arrhenius), enabling time-temperature superposition. Unusual units (s\ :math:`^{\alpha}`) arise from fractional calculus.
   *For practitioners*: Compare :math:`\tau` to service timescales. If tservice << :math:`\tau`, use high-frequency modulus; if tservice >> :math:`\tau`, use Ge.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Fractional Zener SL Parameters
   :header-rows: 1
   :widths: 25 25 25 25

   * - Parameter Range
     - Material Type
     - Typical Materials
     - Processing Implications
   * - Ge > :math:`10^5 Pa, \alpha` < 0.3
     - Stiff crosslinked solid
     - Vulcanized rubber, thermosets
     - Load-bearing, minimal creep
   * - Ge ~ :math:`10^3-10^4 Pa, \alpha` ~ 0.4-0.6
     - Soft viscoelastic solid
     - Gels, soft tissues, elastomers
     - Damping, vibration isolation
   * - Ge < :math:`10^3 Pa, \alpha` > 0.7
     - Weakly crosslinked network
     - Hydrogels, biopolymers
     - Requires careful handling, creep-prone

.. list-table:: Fractional Order Impact on Spectrum
   :header-rows: 1
   :widths: 20 30 30 20

   * - :math:`\alpha` Range
     - Spectrum Breadth
     - Typical Materials
     - Decades Needed
   * - 0.1-0.3
     - Very broad, hierarchical
     - Nanocomposites, biological tissues
     - 5+ decades
   * - 0.4-0.6
     - Moderate
     - Standard elastomers, gels
     - 3-4 decades
   * - 0.7-0.9
     - Narrow, near-exponential
     - Homogeneous networks
     - 2-3 decades

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **Ge fits near lower bound**: Material may be liquid-like (Ge → 0); consider Fractional Maxwell Gel (FMG) instead
- :math:`\alpha` **hits bounds (0.05 or 0.95)**: Data may not support fractional behavior; try classical Zener for simpler interpretation
- :math:`c_{\alpha/Ge}` **> 100**: Extreme relaxation; verify data quality at short times and check for nonlinear effects
- **Poor fit at low frequencies**: Equilibrium plateau Ge not reached; extend frequency range or measurement time
- **Systematic residual trends**: Check for thermorheological complexity (frequency-dependent shift factors) or nonlinear viscoelasticity

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS) or **Stress relaxation**: 4-5 decades
2. **Coverage**: Ensure equilibrium plateau Ge is clearly visible at low :math:`\omega`
3. **Test amplitude**: Within LVR (< 5% strain or stress)
4. **Temperature**: Constant ±0.1°C

**Initialization Strategy:**

.. code-block:: text

   # From frequency sweep G'(ω)
   Ge_init = G'(ω → 0)  # Low-frequency plateau
   c_alpha_init = magnitude in intermediate regime
   tau_init = 1 / (crossover frequency)
   alpha_init = slope of power-law region

   # From stress relaxation G(t)
   Ge_init = G(t → ∞)  # Equilibrium modulus
   c_alpha_init = (G(t=0) - Ge_init) * tau_init**alpha_init
   tau_init = inflection point time

**Optimization Tips:**

- Verify equilibrium plateau Ge is reached (G' → Ge at low :math:`\omega`)
- Use log-weighted least squares
- Constrain Ge > 0 (solid-like behavior required)
- Check that :math:`\alpha` is well-constrained (not at bounds)

**Common Pitfalls:**

- **Ge near zero**: Material may be liquid-like; use FMG instead
- **Poor low-frequency fit**: Equilibrium not reached; extend frequency range
- :math:`\alpha` **near 1**: Consider classical Zener for simpler interpretation

**Troubleshooting Table:**

.. list-table:: Common Issues and Solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Issue
     - Likely Cause
     - Solution
   * - Ge converges to lower bound
     - Liquid-like behavior
     - Switch to FMG or FML model
   * - :math:`\alpha` hits upper bound (0.95+)
     - Nearly exponential relaxation
     - Use classical Zener for clarity
   * - :math:`c_{\alpha/Ge}` > 100
     - Extreme relaxation or data quality
     - Check short-time data, verify LVR
   * - Poor fit at :math:`\omega` → 0
     - Equilibrium not reached
     - Extend frequency range or measurement time
   * - High correlation Ge-:math:`c_{\alpha}`
     - Insufficient frequency coverage
     - Need broader data spanning 4+ decades
   * - Non-monotonic residuals
     - Multiple relaxation mechanisms
     - Consider GMM or add second mode
   * - Fit diverges
     - Poor initial guess
     - Use hierarchical fitting from simpler models

Practical Applications
----------------------

**Quality Control:**

The fractional order :math:`\alpha` and relaxation time :math:`\tau` provide sensitive quality metrics for batch consistency. Monitor these parameters over production runs:

- **Decreased** :math:`\alpha`: Indicates increased polydispersity from contamination, degradation, or processing variations
- **Increased** :math:`\tau`: May signal molecular weight increase from post-cure or aggregation
- **Decreased Ge**: Loss of crosslink density from aging or incomplete cure

**Material Development:**

Use FZSL fitting to guide formulation:

1. **Filler optimization**: Lower :math:`\alpha` with increasing filler loading indicates filler-matrix interphase effects. Target :math:`\alpha` > 0.6 for good dispersion.
2. **Crosslink density**: Ge scales with crosslink density. Track Ge vs. cure time or temperature to optimize curing protocols.
3. **Polymer blending**: Broad :math:`\alpha` (< 0.4) suggests incompatibility or phase separation. Target :math:`\alpha` > 0.5 for miscible blends.

**Failure Prediction:**

The FZSL model enables long-term performance prediction:

- **Creep compliance**: Convert to J(t) via Laplace transform to predict dimensional changes under sustained load
- **Stress relaxation**: Calculate bolt preload decay in gasketing applications
- **Lifetime estimation**: Extrapolate to service timescales (10+ years) using time-temperature superposition

**Design Guidelines:**

For structural applications:

- **Short-term loading** (t << :math:`\tau`): Use instantaneous modulus G(0) = Ge + :math:`c_{\alpha \tau^(-\alpha)}`
- **Long-term loading** (t >> :math:`\tau`): Use equilibrium modulus Ge
- **Cyclic loading** (:math:`\omega \approx 1/\tau`): Maximum energy dissipation, critical for damping applications

Example Calculations
--------------------

**Relaxation Modulus Prediction:**

Given fitted parameters Ge = 1.0 MPa, :math:`c_{\alpha}` = 0.5 MPa·s^0.5, :math:`\alpha = 0.5, \tau` = 10 s:

.. code-block:: python

   import numpy as np
   from rheojax.models import FractionalZenerSolidLiquid
   from rheojax.core.jax_config import safe_import_jax

   jax, jnp = safe_import_jax()

   model = FractionalZenerSolidLiquid()
   model.parameters.set_value('Ge', 1.0e6)  # Pa
   model.parameters.set_value('c_alpha', 0.5e6)  # Pa·s^0.5
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('tau', 10.0)  # s

   # Predict at specific time points
   t = jnp.logspace(-2, 4, 100)  # 0.01 to 10,000 s
   G_t = model.predict(t, test_mode='relaxation')

   # Check limiting behavior
   print(f"G(t=0.01 s) = {G_t[0]/1e6:.3f} MPa (near instantaneous)")
   print(f"G(t=10,000 s) = {G_t[-1]/1e6:.3f} MPa (near equilibrium Ge)")

**Complex Modulus Prediction:**

.. code-block:: python

   # Predict frequency response
   omega = jnp.logspace(-2, 2, 100)  # 0.01 to 100 rad/s
   G_star = model.predict(omega, test_mode='oscillation')

   # Separate storage and loss moduli
   G_prime = jnp.real(G_star)
   G_double_prime = jnp.imag(G_star)

   # Find crossover frequency (G' = G")
   crossover_idx = jnp.argmin(jnp.abs(G_prime - G_double_prime))
   omega_crossover = omega[crossover_idx]
   print(f"G'/G\" crossover at ω ≈ {omega_crossover:.3f} rad/s")
   print(f"Compare to τ = {1/omega_crossover:.3f} s")

**Parameter Sensitivity Analysis:**

.. code-block:: python

   # Study effect of α on spectrum breadth
   alphas = [0.3, 0.5, 0.7, 0.9]
   for alpha_val in alphas:
       model.parameters.set_value('alpha', alpha_val)
       G_star = model.predict(omega, test_mode='oscillation')
       G_prime = jnp.real(G_star)

       # Calculate decades of dispersion (width where G' varies significantly)
       G_range = jnp.where((G_prime > 1.1*Ge) & (G_prime < 0.9*G_max))
       decades = jnp.log10(omega[G_range].max() / omega[G_range].min())
       print(f"α = {alpha_val}: spectrum spans {decades:.1f} decades")

See Also
--------

- :doc:`fractional_zener_ss` and :doc:`fractional_zener_ll` — alternative plateau choices
- :doc:`fractional_kv_zener` — Kelvin-based creep analogue sharing the same compliance
- :doc:`fractional_burgers` — combines fractional Maxwell and Kelvin branches
- :doc:`../../transforms/mutation_number` — monitor when the solid assumption holds during
  gelation
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — fractional Zener
  comparison notebook

API References
--------------
- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalZenerSolidLiquid`

Usage
-----

Basic Fitting
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalZenerSolidLiquid
   from rheojax.core.data import RheoData
   from rheojax.core.jax_config import safe_import_jax
   import numpy as np

   jax, jnp = safe_import_jax()

   # Load experimental data (frequency sweep)
   omega = np.logspace(-2, 2, 50)  # 0.01 to 100 rad/s
   G_prime = ...  # Measured storage modulus
   G_double_prime = ...  # Measured loss modulus
   G_star = G_prime + 1j * G_double_prime

   # Create RheoData object
   data = RheoData(x=omega, y=G_star, test_mode='oscillation')

   # Initialize and fit model
   model = FractionalZenerSolidLiquid()
   result = model.fit(data)

   # Access fitted parameters
   print(f"Ge = {model.parameters.get_value('Ge'):.2e} Pa")
   print(f"c_alpha = {model.parameters.get_value('c_alpha'):.2e} Pa·s^alpha")
   print(f"alpha = {model.parameters.get_value('alpha'):.3f}")
   print(f"tau = {model.parameters.get_value('tau'):.2e} s")
   print(f"R² = {result.r_squared:.4f}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bayesian inference with uncertainty quantification
   from rheojax.pipeline.bayesian import BayesianPipeline

   # Perform NLSQ fit first (warm-start for MCMC)
   model.fit(data)

   # Bayesian inference with default 4 chains
   result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42  # For reproducibility
   )

   # Extract posterior samples
   samples = result.posterior_samples

   # Get credible intervals
   intervals = model.get_credible_intervals(samples, credibility=0.95)
   for param, (lower, upper) in intervals.items():
       mean = samples[param].mean()
       print(f"{param}: {mean:.2e} [{lower:.2e}, {upper:.2e}]")

   # Check convergence diagnostics
   import arviz as az
   inference_data = az.from_numpyro(result)
   print(az.summary(inference_data, hdi_prob=0.95))

   # Visualize posterior distributions
   az.plot_pair(inference_data, divergences=True)

Advanced Usage with Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Complete workflow: load, fit, plot, save
   pipeline = (Pipeline()
       .load('experimental_data.csv', x_col='omega', y_col='G_star')
       .fit('fractional_zener_sl')
       .plot(style='publication')
       .save('results.hdf5'))

   # Access fitted model
   model = pipeline.model
   print(f"Equilibrium modulus: {model.parameters.get_value('Ge'):.2e} Pa")

Time-Temperature Superposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import Mastercurve

   # Create master curve from multiple temperature datasets
   datasets = [
       {'omega': omega_30C, 'G_star': G_star_30C, 'T': 30},
       {'omega': omega_50C, 'G_star': G_star_50C, 'T': 50},
       {'omega': omega_70C, 'G_star': G_star_70C, 'T': 70},
   ]

   mc = Mastercurve(reference_temp=50, auto_shift=True)
   master_curve, shift_factors = mc.transform(datasets)

   # Fit FZSL to master curve
   model = FractionalZenerSolidLiquid()
   model.fit(master_curve.x, master_curve.y, test_mode='oscillation')

   # Predict at different temperatures using shift factors
   omega_pred = np.logspace(-4, 4, 100)
   for T, aT in shift_factors.items():
       omega_shifted = omega_pred * aT
       G_star_T = model.predict(omega_shifted, test_mode='oscillation')
       print(f"T = {T}°C: shift factor aT = {aT:.2e}")

See also
--------

- :doc:`fractional_zener_ss` and :doc:`fractional_zener_ll` — alternative plateau choices.
- :doc:`fractional_kv_zener` — Kelvin-Voigt-based analogue sharing the same compliance.
- :doc:`fractional_burgers` — combines fractional Maxwell and Kelvin branches.
- :doc:`../../transforms/mutation_number` — monitor when the solid assumption holds during
  gelation.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — fractional Zener
  comparison notebook.

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [3] Koeller, R. C. "Applications of fractional calculus to the theory of
   viscoelasticity." *Journal of Applied Mechanics*, 51, 299–307 (1984).
   https://doi.org/10.1115/1.3167616

.. [4] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [5] Friedrich, C. "Relaxation and retardation functions of the Maxwell model
   with fractional derivatives." *Rheologica Acta*, 30, 151–158 (1991).
   https://doi.org/10.1007/BF01134604
.. [6] Metzler, R., Schick, W., Kilian, H.-G., & Nonnenmacher, T. F. "Relaxation in filled polymers: A fractional calculus approach."
   *Journal of Chemical Physics*, **103**, 7180-7186 (1995).
   https://doi.org/10.1063/1.470346

.. [7] Friedrich, C. "Relaxation and retardation functions of the Maxwell model with fractional derivatives."
   *Rheologica Acta*, **30**, 151-158 (1991).
   https://doi.org/10.1007/BF01134604

.. [8] Heymans, N. & Bauwens, J. C. "Fractal rheological models and fractional differential equations for viscoelastic behavior."
   *Rheologica Acta*, **33**, 210-219 (1994).
   https://doi.org/10.1007/BF00437306

.. [9] Nonnenmacher, T. F. & Glöckle, W. G. "A fractional model for mechanical stress relaxation."
   *Philosophical Magazine Letters*, **64**, 89-93 (1991).
   https://doi.org/10.1080/09500839108214672

.. [10] Podlubny, I. *Fractional Differential Equations*.
   Academic Press (1999). ISBN: 978-0125588409

