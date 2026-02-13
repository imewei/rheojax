.. _model-fractional-poynting-thomson:

Fractional Poynting-Thomson (Fractional)
========================================

Quick Reference
---------------

- **Use when:** Stress-relaxation with instantaneous modulus and fractional retardation
- **Parameters:** 4 (:math:`G_e, G_k, \alpha, \tau`)
- **Key equation:** :math:`G(t) = G_{\mathrm{eq}} + (G_e - G_{\mathrm{eq}}) E_{\alpha}(-(t/\tau)^{\alpha})` where :math:`G_{\mathrm{eq}} = \frac{G_e G_k}{G_e + G_k}`
- **Test modes:** Relaxation, creep, oscillation
- **Material examples:** Viscoelastic solids emphasizing stress-relaxation interpretations

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
     - Instantaneous modulus (immediate elastic response)
   * - :math:`G_k`
     - Pa
     - Retarded modulus (Kelvin element spring)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < :math:`\alpha` < 1, power-law character)
   * - :math:`\tau`
     - s
     - Retardation time (characteristic timescale)
   * - :math:`G_{\text{eq}}`
     - Pa
     - Equilibrium modulus, :math:`G_{\text{eq}} = G_e G_k / (G_e + G_k)`
   * - :math:`E_{\alpha}(z)`
     - dimensionless
     - One-parameter Mittag-Leffler function

Overview
--------

Equivalent in form to FKVZ but emphasizing the instantaneous modulus :math:`G_e` in series with a fractional Kelvin-Voigt element. Convenient for stress-relaxation interpretations.

Physical Foundations
--------------------

The Fractional Poynting-Thomson model (also known as Fractional Standard Linear Solid
in relaxation form) consists of:

**Mechanical Configuration:**

.. code-block:: text

   [Spring Ge] ---- series ---- [Spring Gk parallel with SpringPot (α, τ)]

**Microstructural Interpretation:**

- **Instantaneous spring (** :math:`G_e` **)**: Immediate elastic response from bond stretching
  and glassy contributions. Sets :math:`G(t=0) = G_e`.
- **Kelvin element**: Provides delayed relaxation from network rearrangements.
  The spring :math:`G_k` and SpringPot work together to create power-law stress relaxation.
- **Equilibrium behavior**: Material relaxes to :math:`G_{\text{eq}} = G_e G_k / (G_e + G_k)`
- **Solid-like**: Finite equilibrium modulus (no flow)

The series configuration makes this model natural for stress relaxation experiments
where the instantaneous modulus and relaxation magnitude are directly observable.

Governing Equations
-------------------

Time domain (creep compliance; same functional form as FKVZ):

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\Big[1 - E_{\alpha}\!\big(-(t/\tau)^{\alpha}\big)\Big].
   \]

Time domain (relaxation modulus; interpolative form):

.. math::
   :nowrap:

   \[
   G(t) \;=\; G_{\mathrm{eq}} \;+\; \big(G_e - G_{\mathrm{eq}}\big)\,
   E_{\alpha}\!\left(-\left(\frac{t}{\tau}\right)^{\alpha}\right),
   \quad G_{\mathrm{eq}} \equiv \frac{G_e G_k}{G_e + G_k}.
   \]

Frequency domain (via complex compliance):

.. math::
   :nowrap:

   \[
   J^{*}(\omega) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\,\frac{1}{1+(i\omega\tau)^{\alpha}},
   \qquad
   G^{*}(\omega) \;=\; \frac{1}{J^{*}(\omega)} .
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
     - Instantaneous modulus
   * - ``Gk``
     - :math:`G_k`
     - Pa
     - [1e-3, 1e9]
     - Retarded modulus
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e6]
     - Retardation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Instantaneous response :math:`G(0)=G_e`; relaxes toward :math:`G_{\mathrm{eq}}`.
- Fractional retardation governs the relaxation tail (broad spectra).

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Poynting-Thomson (Zener) behavior.
- :math:`G_k \to \infty`: :math:`G(t)\to G_e` (no retardation).

What You Can Learn
------------------

This section explains insights from the Fractional Poynting-Thomson model,
emphasizing stress-relaxation interpretations and the dual-modulus solid structure.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Instantaneous Modulus (** :math:`G_e` **)**:
   The modulus at :math:`t = 0^+`, representing the material's immediate elastic response upon loading.

   *For graduate students*: :math:`G_e` includes both entropic (network) and enthalpic (glassy) contributions to elasticity. It represents the sum of all elastic contributions before any relaxation occurs.
   *For practitioners*: :math:`G_e` is the initial stiffness for impact loading design. Use this value for short-time mechanical response calculations.

**Retarded Modulus (** :math:`G_k` **)**:
   Controls the amount of stress relaxation from :math:`G_e` to :math:`G_{\text{eq}}`. The equilibrium modulus is given by :math:`G_{\text{eq}} = G_e \cdot G_k / (G_e + G_k)`.

   *For graduate students*: :math:`G_k` represents the spring stiffness in the Kelvin element. The harmonic mean relationship for :math:`G_{\text{eq}}` arises from the series-parallel spring configuration.
   *For practitioners*: Relaxation magnitude :math:`\Delta G = G_e - G_{\text{eq}}` is directly determined by the :math:`G_e/G_k` ratio. Higher :math:`G_k` means less relaxation.

**Fractional Order (** :math:`\alpha` **)**:
   Governs the power-law character of stress relaxation between instantaneous and equilibrium values. Values closer to 0 indicate more solid-like continuous relaxation, while values closer to 1 indicate more liquid-like narrow relaxation.

   - :math:`\alpha \to 0`: Very slow, broad-spectrum relaxation approaching plateau behavior
   - :math:`\alpha \approx 0.5`: Typical fractional solid, balanced spectrum breadth
   - :math:`\alpha \to 1`: Exponential relaxation (classical Poynting-Thomson/Zener)

   *For graduate students*: :math:`\alpha` quantifies polydispersity in the retardation time spectrum. Lower :math:`\alpha` indicates greater microstructural heterogeneity.
   *For practitioners*: Lower :math:`\alpha` means relaxation spreads over more time decades—important for long-term load-bearing applications.

**Retardation Time (** :math:`\tau` **)**:
   Characteristic timescale for the transition from :math:`G_e` to :math:`G_{eq}`.

   *For graduate students*: :math:`\tau` is temperature-dependent (WLF/Arrhenius), enabling time-temperature superposition for master curve construction.
   *For practitioners*: Compare :math:`\tau` to service timescales. If service time :math:`\ll \tau`, use :math:`G_e`; if :math:`\gg \tau`, use :math:`G_{eq}`.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Relaxation Behavior from Fractional Poynting-Thomson Parameters
   :header-rows: 1
   :widths: 25 25 25 25

   * - :math:`G_e/G_{eq}` Ratio
     - Relaxation Character
     - Typical Materials
     - Design Implications
   * - < 2
     - Minor relaxation
     - Dense crosslinked networks
     - Stable under sustained load
   * - 2-10
     - Moderate relaxation
     - Filled elastomers, thermoplastics
     - Account for stress decay
   * - > 10
     - Large relaxation
     - Soft tissues, weak gels
     - Time-dependent design critical

.. list-table:: Fractional Order Impact
   :header-rows: 1
   :widths: 25 25 25 25

   * - :math:`\alpha` Range
     - Spectrum Character
     - Typical Materials
     - Fitting Considerations
   * - 0.1-0.3
     - Very broad, hierarchical
     - Biological tissues, nanocomposites
     - Need 4+ decades in time
   * - 0.4-0.6
     - Moderate breadth
     - Polymer gels, rubbers
     - Standard 3-4 decades sufficient
   * - 0.7-0.9
     - Narrow, near-exponential
     - Homogeneous networks
     - Consider classical Zener

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **G(t) continues decreasing at longest time**: Equilibrium not reached; extend measurement time or consider liquid model (FMG/FML)
- :math:`G_e/G_k` **near 1**: Minimal relaxation; simpler elastic model may suffice
- :math:`\alpha` **hits bounds (0.05 or 0.95)**: Data may not support fractional behavior; try classical Zener
- **Non-monotonic residuals**: Check for multiple relaxation mechanisms or nonlinear effects

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Stress relaxation** (primary): 4-5 decades in time to capture full relaxation
2. **Step strain**: Within LVR (typically 1-5%)
3. **Sampling**: Log-spaced, 50+ points per decade
4. **Temperature**: Constant ±0.1°C

**Initialization Strategy:**

.. code-block:: text

   # From stress relaxation G(t)
   Ge_init = G(t → 0)  # Instantaneous modulus
   Geq_init = G(t → ∞)  # Equilibrium modulus
   # From Geq = Ge*Gk/(Ge+Gk), solve for Gk:
   Gk_init = Ge_init * Geq_init / (Ge_init - Geq_init)
   tau_init = time where relaxation is 50% complete
   alpha_init = 0.5  # Default

**Optimization Tips:**

- Fit in modulus space (natural for relaxation)
- Use log-weighted least squares
- Verify monotonic decay from :math:`G_e` to :math:`G_{eq}`
- Check that :math:`G_{eq} > 0` (solid-like behavior)

**Common Pitfalls:**

- **G(t) approaches zero**: Material is liquid-like; use FML or FMG instead
- **Non-monotonic G(t)**: Instrument artifacts or nonlinear effects
- **Poor long-time fit**: Equilibrium not reached; extend measurement time

**Troubleshooting Table:**

.. list-table:: Common Issues and Solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Issue
     - Likely Cause
     - Solution
   * - :math:`G(t) \to 0` at long times
     - Liquid-like behavior
     - Use FML or FMG model instead
   * - :math:`G_e/G_k \approx 1`
     - Minimal relaxation
     - Consider simple elastic model
   * - :math:`\alpha` near upper bound (0.95+)
     - Nearly exponential
     - Use classical Poynting-Thomson
   * - Non-monotonic residuals
     - Multiple relaxation modes
     - Consider GMM or add second mode
   * - Poor fit at :math:`t \to 0`
     - Instrument response time
     - Exclude first few points
   * - :math:`G_{eq} < 0` (non-physical)
     - Fitting error or wrong model
     - Check data quality, verify solid-like
   * - High :math:`G_e`--:math:`G_k` correlation
     - Insufficient time coverage
     - Need broader range (4+ decades)

Practical Applications
----------------------

**Stress Relaxation Prediction:**

The FPT model enables prediction of stress decay in bolted joints, gaskets, and seals:

.. code-block:: python

   # Bolt preload relaxation over time
   epsilon_initial = 0.05  # 5% initial strain
   G_initial = Ge  # Instantaneous modulus
   sigma_initial = G_initial * epsilon_initial

   # Predict stress at various service times
   service_times = [1, 7, 30, 365]  # days
   for t_days in service_times:
       t_seconds = t_days * 86400
       G_t = model.predict(t_seconds, test_mode='relaxation')
       sigma_t = G_t * epsilon_initial
       relaxation_pct = (1 - sigma_t/sigma_initial) * 100
       print(f"Day {t_days}: σ = {sigma_t/1e6:.2f} MPa "
             f"({relaxation_pct:.1f}% relaxation)")

**Material Selection for Gasketing:**

Compare candidates by relaxation characteristics:

.. list-table:: Gasket Material Comparison
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Material
     - :math:`G_e` (MPa)
     - :math:`G_{eq}` (MPa)
     - :math:`\tau` (s)
     - :math:`\alpha`
     - Relaxation
   * - PTFE
     - 500
     - 400
     - 1000
     - 0.65
     - 20% at :math:`10^4` s
   * - Fiber composite
     - 2000
     - 1800
     - 5000
     - 0.55
     - 10% at :math:`10^4` s
   * - Metal gasket
     - 100000
     - 98000
     - 100
     - 0.85
     - 2% at :math:`10^4` s
   * - Rubber O-ring
     - 10
     - 5
     - 100
     - 0.45
     - 50% at :math:`10^4` s

**Quality Control Applications:**

Monitor relaxation parameters for product consistency:

1. **Instantaneous modulus (** :math:`G_e` **)**: Sensitive to cure state and crosslink density
2. **Equilibrium modulus (** :math:`G_{eq}` **)**: Tracks permanent network structure
3. **Relaxation magnitude (** :math:`G_e - G_{eq}` **)**: Indicates temporary vs. permanent structure balance
4. **Fractional order (** :math:`\alpha` **)**: QC metric for microstructural uniformity

**Design Guidelines:**

For applications requiring sustained stress:

- **Critical applications** (pressure vessels, structural): Target :math:`G_e/G_{eq} < 1.5` for minimal relaxation
- **Sealing applications**: :math:`G_e/G_{eq} = 2`--5 acceptable if recompression possible
- **Damping applications**: Maximize tan(:math:`\delta`) at operating frequency (:math:`\omega \approx 1/\tau`)
- **Service life**: Ensure measurement time > 10× service time for reliable extrapolation

Example Calculations
--------------------

**Stress Relaxation Prediction:**

Given fitted parameters :math:`G_e = 2.0` MPa, :math:`G_k = 5.0` MPa, :math:`\alpha = 0.55`, :math:`\tau = 500` s:

.. code-block:: python

   import numpy as np
   from rheojax.models import FractionalPoyntingThomson
   from rheojax.core.jax_config import safe_import_jax

   jax, jnp = safe_import_jax()

   model = FractionalPoyntingThomson()
   model.parameters.set_value('Ge', 2.0e6)  # Pa
   model.parameters.set_value('Gk', 5.0e6)  # Pa
   model.parameters.set_value('alpha', 0.55)
   model.parameters.set_value('tau', 500.0)  # s

   # Calculate equilibrium modulus
   Ge = 2.0e6
   Gk = 5.0e6
   Geq = Ge * Gk / (Ge + Gk)  # Harmonic mean
   print(f"Equilibrium modulus Geq = {Geq/1e6:.2f} MPa")

   # Predict relaxation modulus
   t = jnp.logspace(-1, 5, 150)  # 0.1 s to ~1 day
   G_t = model.predict(t, test_mode='relaxation')

   # Verify limiting behavior
   print(f"G(t=0.1 s) = {G_t[0]/1e6:.2f} MPa (near Ge)")
   print(f"G(t=100,000 s) = {G_t[-1]/1e6:.2f} MPa (near Geq)")

   # Calculate half-relaxation time
   G_half = (Ge + Geq) / 2
   t_half = t[jnp.argmin(jnp.abs(G_t - G_half))]
   print(f"Half-relaxation time: {t_half:.1f} s")

**Conversion to Creep Compliance:**

.. code-block:: python

   # Use duality: J(t) = 1/G(t) approximately for this model form
   # More accurately, use compliance prediction:
   J_t = model.predict(t, test_mode='creep')

   # Verify reciprocity at limits
   J0 = 1 / Ge
   Jinf = 1 / Geq
   print(f"J(t→0) = {J_t[0]:.2e} Pa⁻¹ (theory: {J0:.2e})")
   print(f"J(t→∞) = {J_t[-1]:.2e} Pa⁻¹ (theory: {Jinf:.2e})")

**Frequency Domain Prediction:**

.. code-block:: python

   # Predict storage and loss moduli
   omega = jnp.logspace(-3, 2, 100)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = jnp.real(G_star)
   G_double_prime = jnp.imag(G_star)

   # Find tan(δ) peak for optimal damping frequency
   tan_delta = G_double_prime / G_prime
   omega_peak = omega[jnp.argmax(tan_delta)]
   tan_delta_max = tan_delta.max()
   print(f"Maximum damping at ω = {omega_peak:.3f} rad/s")
   print(f"tan(δ)_max = {tan_delta_max:.3f}")
   print(f"Compare to 1/τ = {1/500:.4f} rad/s")

   # Calculate loss factor over frequency range
   loss_factor = G_double_prime / G_prime
   # For damping design, target loss_factor > 0.1

See Also
--------

- :doc:`fractional_kv_zener` — identical topology in creep form, sharing the same
  compliance expression
- :doc:`fractional_zener_ss` — adds an additional spring for solids with higher plateaus
- :doc:`fractional_burgers` — combines Kelvin and Maxwell branches for more complex
  retardation spectra
- :doc:`../../transforms/fft` — necessary to obtain :math:`G'(\omega)` before fitting
  relaxation data
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — demonstrates how
  Poynting–Thomson fits compare to other fractional elements

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalPoyntingThomson`

Usage
-----

Basic Stress Relaxation Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalPoyntingThomson
   from rheojax.core.data import RheoData
   import numpy as np

   # Load experimental stress relaxation data
   time = np.logspace(-1, 4, 100)  # 0.1 to 10,000 s
   stress = ...  # Measured stress after step strain
   # Normalize to modulus: G(t) = stress / strain
   strain = 0.05  # 5% step strain
   G_t = stress / strain

   # Create RheoData object
   data = RheoData(x=time, y=G_t, test_mode='relaxation')

   # Initialize and fit model
   model = FractionalPoyntingThomson()
   result = model.fit(data)

   # Access fitted parameters
   Ge = model.parameters.get_value('Ge')
   Gk = model.parameters.get_value('Gk')
   alpha = model.parameters.get_value('alpha')
   tau = model.parameters.get_value('tau')

   # Calculate derived quantities
   Geq = Ge * Gk / (Ge + Gk)  # Equilibrium modulus
   Delta_G = Ge - Geq  # Relaxation magnitude

   print(f"Instantaneous modulus Ge = {Ge/1e6:.2f} MPa")
   print(f"Equilibrium modulus Geq = {Geq/1e6:.2f} MPa")
   print(f"Relaxation magnitude ΔG = {Delta_G/1e6:.2f} MPa")
   print(f"Relaxation ratio Ge/Geq = {Ge/Geq:.2f}")
   print(f"Fractional order α = {alpha:.3f}")
   print(f"Retardation time τ = {tau:.1f} s")
   print(f"R² = {result.r_squared:.4f}")

Bayesian Inference for Relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bayesian inference with uncertainty quantification
   # NLSQ fit first for warm-start
   model.fit(data)

   # Run Bayesian inference
   result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Extract posterior samples and compute derived quantities
   Ge_samples = result.posterior_samples['Ge']
   Gk_samples = result.posterior_samples['Gk']
   Geq_samples = Ge_samples * Gk_samples / (Ge_samples + Gk_samples)

   # Get credible intervals
   import numpy as np
   Geq_mean = Geq_samples.mean()
   Geq_lower = np.percentile(Geq_samples, 2.5)
   Geq_upper = np.percentile(Geq_samples, 97.5)

   print(f"Geq = {Geq_mean/1e6:.2f} MPa "
         f"[{Geq_lower/1e6:.2f}, {Geq_upper/1e6:.2f}] (95% CI)")

   # Visualize posterior correlations
   import arviz as az
   inference_data = az.from_numpyro(result)
   az.plot_pair(
       inference_data,
       var_names=['Ge', 'Gk', 'alpha', 'tau'],
       kind='kde',
       divergences=True
   )

   # Check convergence
   print(az.summary(inference_data, hdi_prob=0.95))

Frequency Domain Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert relaxation parameters to frequency response
   omega = np.logspace(-3, 2, 100)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = np.real(G_star)
   G_double_prime = np.imag(G_star)

   # Calculate loss tangent
   tan_delta = G_double_prime / G_prime

   # Find damping peak
   omega_peak = omega[np.argmax(tan_delta)]
   print(f"Peak damping at ω = {omega_peak:.3f} rad/s")
   print(f"Compare to 1/τ = {1/tau:.3f} rad/s")

   # Plot Cole-Cole diagram (G'' vs G')
   import matplotlib.pyplot as plt
   plt.figure(figsize=(8, 6))
   plt.plot(G_prime/1e6, G_double_prime/1e6, 'o-')
   plt.xlabel("G' (MPa)")
   plt.ylabel("G'' (MPa)")
   plt.title('Cole-Cole Plot')
   plt.axis('equal')
   plt.grid(True)

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Complete workflow with visualization
   pipeline = (Pipeline()
       .load('stress_relaxation.csv', x_col='time', y_col='modulus')
       .fit('fractional_poynting_thomson')
       .plot(style='publication')
       .save('fpt_results.hdf5'))

   # Access fitted model
   model = pipeline.model
   Ge = model.parameters.get_value('Ge')
   Gk = model.parameters.get_value('Gk')
   Geq = Ge * Gk / (Ge + Gk)

   # Predict long-term behavior
   t_extrapolate = np.logspace(5, 8, 50)  # Up to 3 years
   G_extrapolate = model.predict(t_extrapolate, test_mode='relaxation')

   import matplotlib.pyplot as plt
   plt.semilogx(t_extrapolate/86400, G_extrapolate/1e6)  # Convert to days
   plt.xlabel('Time (days)')
   plt.ylabel('G(t) (MPa)')
   plt.title('Long-term Stress Relaxation Prediction')

Multi-Temperature Master Curve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit relaxation at multiple temperatures
   from rheojax.transforms import Mastercurve

   datasets = [
       {'time': t_30C, 'G': G_30C, 'T': 30},
       {'time': t_50C, 'G': G_50C, 'T': 50},
       {'time': t_70C, 'G': G_70C, 'T': 70},
   ]

   # Create master curve using TTS
   mc = Mastercurve(reference_temp=50, auto_shift=True)
   master_curve, shift_factors = mc.transform(datasets)

   # Fit FPT to master curve
   model.fit(master_curve.x, master_curve.y, test_mode='relaxation')

   # Use shift factors for temperature predictions
   T_service = 40  # °C
   aT = shift_factors[T_service]
   t_service = np.logspace(-1, 5, 100)
   t_shifted = t_service / aT  # Shift to reference temperature
   G_40C = model.predict(t_shifted, test_mode='relaxation')

   print(f"At {T_service}°C, shift factor aT = {aT:.2e}")
   print(f"Predicted G(1000s) at {T_service}°C = {G_40C[50]/1e6:.2f} MPa")

See also
--------

- :doc:`fractional_kv_zener` — identical topology in creep form, sharing the same
  compliance expression.
- :doc:`fractional_zener_ss` — adds an additional spring for solids with higher plateaus.
- :doc:`fractional_burgers` — combines Kelvin and Maxwell branches for more complex
  retardation spectra.
- :doc:`../../transforms/fft` — necessary to obtain :math:`G'(\omega)` before fitting
  relaxation data.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — demonstrates how
  Poynting–Thomson fits compare to other fractional elements.

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Poynting, J. H., and Thomson, J. J. *The Properties of Matter*.
   Charles Griffin and Company (1902).

.. [3] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [4] Koeller, R. C. "Applications of fractional calculus to the theory of
   viscoelasticity." *Journal of Applied Mechanics*, 51, 299–307 (1984).
   https://doi.org/10.1115/1.3167616

.. [5] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012
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

