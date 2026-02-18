.. _model-fractional-jeffreys:

Fractional Jeffreys Model (Fractional)
======================================

Quick Reference
---------------

- **Use when:** Viscoelastic liquid with fractional relaxation, terminal viscous flow
- **Parameters:** 4 (:math:`\eta_1`, :math:`\eta_2`, :math:`\alpha`, :math:`\tau_1`)
- **Key equation:** :math:`G^*(\omega) = \eta_1(i\omega) \frac{1 + (i\omega\tau_2)^{\alpha}}{1 + (i\omega\tau_1)^{\alpha}}`
- **Test modes:** Oscillation, relaxation, creep, flow curve
- **Material examples:** Polymer solutions with broad relaxation spectra, complex fluids with viscous flow

.. include:: /_includes/fractional_seealso.rst

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`\eta_1`
     - Pa·s
     - First viscosity (parallel dashpot, controls zero-shear viscosity)
   * - :math:`\eta_2`
     - Pa·s
     - Second viscosity (series branch)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < :math:`\alpha` < 1, controls spectrum breadth)
   * - :math:`\tau_1`
     - s
     - Relaxation time (characteristic timescale)
   * - :math:`\tau_2`
     - s
     - Derived time, :math:`\tau_2 = (\eta_2/\eta_1)\tau_1`
   * - :math:`E_{\alpha,\beta}(z)`
     - dimensionless
     - Two-parameter Mittag-Leffler function

Overview
--------

A liquid model consisting of one dashpot in parallel with a series dashpot-SpringPot branch. It exhibits viscous flow with fractional relaxation features.

Physical Foundations
--------------------

The Fractional Jeffreys model extends the classical Jeffreys model by incorporating
fractional-order viscoelasticity through a SpringPot element. The mechanical analogue
consists of:

**Mechanical Configuration:**

.. code-block:: text

   [Dashpot η₁] ---- parallel ---- [Dashpot η₂ in series with SpringPot (α, τ₁)]

**Microstructural Interpretation:**

- **Parallel dashpot** (:math:`\eta_1`): Provides zero-shear viscosity from long-range
  molecular motion (chain reptation, solvent flow)
- **Series branch**: SpringPot + dashpot combination creates fractional
  relaxation with characteristic time :math:`\tau_1`
- **Liquid behavior**: Both branches dissipate energy; zero equilibrium modulus

This configuration is particularly suited for polymer solutions where the parallel
dashpot represents the solvent contribution and the series branch captures polymer
chain dynamics with a broad relaxation spectrum.

Governing Equations
-------------------

Time domain (relaxation modulus):

.. math::
   :nowrap:

   \[
   G(t) \;=\; \frac{\eta_1}{\tau_1}\, t^{-\alpha}\,
   E_{1-\alpha,\,1-\alpha}\!\left(-\left(\frac{t}{\tau_1}\right)^{1-\alpha}\right).
   \]

Frequency domain (complex modulus):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) \;=\; \eta_1(i\omega)\,
   \frac{1 + (i\omega\tau_2)^{\alpha}}{1 + (i\omega\tau_1)^{\alpha}},
   \quad \tau_2 = \frac{\eta_2}{\eta_1}\,\tau_1 .
   \]

Steady shear (rotation): Newtonian-like at low rates with viscosity dominated by :math:`\eta_1`.

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
   * - ``eta1``
     - :math:`\eta_1`
     - Pa·s
     - [1e-6, 1e12]
     - First viscosity
   * - ``eta2``
     - :math:`\eta_2`
     - Pa·s
     - [1e-6, 1e12]
     - Second viscosity
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau1``
     - :math:`\tau_1`
     - s
     - [1e-6, 1e6]
     - Relaxation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, oscillation, steady shear.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Low :math:`\omega`: liquid-like; :math:`G' \ll G'' \approx \eta_{\mathrm{eff}}\omega`.
- Intermediate: fractional dispersion with order :math:`\alpha`.
- High :math:`\omega`: elastic upturn from branch dynamics.

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Jeffreys model.
- :math:`\eta_2 \to 0`: Maxwell-like liquid dominated by :math:`\eta_1`.

What You Can Learn
------------------

This section explains how to translate fitted Fractional Jeffreys parameters into
material insights and actionable knowledge for viscoelastic liquids with fractional
relaxation characteristics.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Primary Viscosity (** :math:`\eta_1` **)**:
   The dominant viscosity controlling zero-shear behavior and terminal flow.

   - **For graduate students**: In polymer solutions, :math:`\eta_1 \propto M_w^{3.4}`
     for entangled systems (reptation scaling). For solutions, :math:`\eta_1` combines
     solvent viscosity with polymer contribution, separable via :math:`\eta_1 = \eta_solvent + \eta_polymer`.
   - **For practitioners**: :math:`\eta_1` determines processing viscosity and flow rates.
     Directly measurable as :math:`\lim_{\omega \to 0} G''/\omega`. Critical for pump sizing,
     mixing times, and coating applications.

**Secondary Viscosity (** :math:`\eta_2` **)**:
   Controls the high-frequency elastic-like response and relaxation strength.

   - **Viscosity ratio**: :math:`\eta_2/\eta_1` indicates the relative strength of the two branches.
     Large ratio (>0.5): Significant elastic contribution at high frequencies. Small ratio (<0.1):
     Weakly viscoelastic, nearly Newtonian.
   - **Derived time**: :math:`\tau_2 = (\eta_2/\eta_1)\tau_1` sets the high-frequency transition.
   - **For practitioners**: Higher :math:`\eta_2` means stronger elastic recoil (die swell in extrusion) and
     slower stress relaxation.

**Fractional Order (** :math:`\alpha` **)**:
   Governs the breadth of the relaxation spectrum and power-law character.

   - :math:`\alpha \approx 0.2\text{--}0.4`: Very broad spectrum, highly polydisperse or complex microstructure
     (wormlike micelles with broad contour length distribution, blended polymer solutions)
   - :math:`\alpha \approx 0.5\text{--}0.7`: Moderate breadth, typical for commercial polymer solutions with
     moderate polydispersity (PDI = 2-4)
   - :math:`\alpha \approx 0.8\text{--}0.95`: Narrow spectrum, nearly monodisperse systems (fractionated polymers)
   - :math:`\alpha \to 1`: Classical Jeffreys (single exponential), use simpler model

   *Physical interpretation*: Lower :math:`\alpha` indicates greater polydispersity in relaxation times
   arising from molecular weight distribution, chain architecture (branching), or structural
   heterogeneity. For polymers, :math:`\alpha \approx 1/(1 + \text{PDI}/4)` approximately.

   *For practitioners*: Use :math:`\alpha` as a QC metric for batch consistency. A sudden drop in :math:`\alpha` suggests
   contamination, degradation (chain scission), or aggregation.

**Relaxation Time (** :math:`\tau_1` **)**:
   Characteristic timescale for the fractional relaxation process.

   - **Temperature dependence**: Follows WLF equation for polymers above Tg, Arrhenius below Tg.
   - **Molecular weight scaling**: For polymer solutions, :math:`\tau_1 \propto M_w^{3.4}` in
     entangled regime, :math:`\tau_1 \propto M_w` for unentangled chains.
   - **For practitioners**: Compare :math:`\tau_1` to process timescales. Ensure process time > :math:`5\tau_1` for
     complete stress relaxation in molding or coating operations.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Fractional Jeffreys Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - High :math:`\eta_1 (>10^3` Pa·s), low :math:`\alpha` (<0.5)
     - Highly entangled viscoelastic
     - Concentrated polymer solutions, melts
     - High die swell, elastic instabilities
   * - Moderate :math:`\eta_1/\eta_2` (0.2-0.8), :math:`\alpha \approx 0.5`
     - Balanced viscoelastic liquid
     - Wormlike micelles, associative polymers
     - Moderate shear thinning, extensional thickening
   * - Low :math:`\eta_1` (<100 Pa·s), :math:`\alpha` > 0.7
     - Weakly viscoelastic
     - Dilute polymer solutions, oligomers
     - Easy flow, minimal elastic effects
   * - :math:`\alpha \approx 1` (any :math:`\eta_1`)
     - Near-Maxwellian
     - Monodisperse linear polymers
     - Use classical Jeffreys for simplicity

.. list-table:: Additional Material Examples by Viscosity Ratio
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\eta_2/\eta_1` Ratio
     - Physical Interpretation
     - Typical Materials
     - Key Applications
   * - < 0.1
     - Solvent-dominated
     - Dilute polymer solutions
     - Coatings, inks
   * - 0.1-0.5
     - Moderate polymer contribution
     - Semi-dilute solutions
     - Adhesives, biomedical fluids
   * - 0.5-2.0
     - Balanced branches
     - Concentrated solutions, blends
     - Lubricants, personal care
   * - > 2.0
     - Polymer-dominated
     - Near-melts, gels
     - Processing aids, rheology modifiers

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- :math:`\eta_1` **much larger than expected**: Check for yield stress (use Bingham or Herschel-Bulkley instead)
  or incorrect temperature control.
- **Poor low-frequency fit**: Terminal region not reached; extend frequency range below 0.01 rad/s
  or use longer stress relaxation tests.
- :math:`\alpha` **near 1**: Data may be described by simpler classical Jeffreys; fractional model overfitting.
- **Strong** :math:`\eta_1-\tau_1` **correlation**: Well-constrained zero-shear viscosity but individual parameters
  ambiguous. Report :math:`\eta_0 = \eta_1` instead.
- :math:`\eta_2 > \eta_1`: Non-physical unless fitting error; check data quality and bounds.

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS): 3-5 decades (e.g., 0.01-100 rad/s)
2. **Test amplitude**: Within LVR (typically 0.5-5% strain)
3. **Coverage**: Ensure both terminal and intermediate regimes are captured
4. **Temperature control**: ±0.1°C for solutions

**Initialization Strategy:**

.. code-block:: python

   # From frequency sweep G'(ω), G"(ω)
   eta1_init = lim(G"/ω) as ω → 0  # Terminal viscosity
   alpha_init = slope of log(G') vs log(ω) in intermediate regime
   tau1_init = 1 / (frequency at G' = G" crossover)
   eta2_init = eta1_init * 0.5  # Typical starting ratio

**Optimization Tips:**

- Fit simultaneously to :math:`G'` and :math:`G''` for better constraint
- Use log-weighted least squares
- Verify terminal flow region (:math:`G'' \sim \omega`, :math:`G' \sim \omega^2`) at low frequencies
- Check residuals for systematic deviations

**Common Pitfalls:**

- **Insufficient low-frequency data**: Cannot determine :math:`\eta_1` accurately
- **High-frequency artifacts**: Instrument inertia effects can bias :math:`\alpha`
- **Wrong model choice**: If :math:`G'` plateaus at low :math:`\omega`, use solid model instead

**Troubleshooting Table:**

.. list-table:: Common Issues and Solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Issue
     - Likely Cause
     - Solution
   * - :math:`G'` plateaus at low :math:`\omega`
     - Solid-like behavior
     - Use FZSL or FMG instead
   * - :math:`\eta_1` hits upper bound
     - Yield stress present
     - Use Bingham/Herschel-Bulkley
   * - Poor terminal fit (:math:`G'' \sim \omega`)
     - Terminal region not reached
     - Extend to :math:`\omega` < 0.01 rad/s
   * - :math:`\alpha` near 1.0
     - Narrow spectrum
     - Use classical Jeffreys
   * - Strong :math:`\eta_1-\tau_1` correlation
     - Well-constrained :math:`\eta_0` only
     - Report :math:`\eta_0 = \eta_1`, accept ambiguity
   * - :math:`\eta_2 > \eta_1` (non-physical)
     - Fitting error
     - Check bounds, data quality
   * - Oscillating residuals
     - Multiple relaxation times
     - Consider GMM for multi-mode
   * - High-frequency upturn
     - Inertia artifacts
     - Exclude data above critical :math:`\omega`

Practical Applications
----------------------

**Flow Rate Prediction:**

The zero-shear viscosity :math:`\eta_1` determines flow behavior in processing:

.. code-block:: python

   # Predict flow rate in pipe or channel
   eta_0 = model.parameters.get_value('eta1')
   pressure_gradient = 1000  # Pa/m
   radius = 0.01  # m (pipe)

   # Hagen-Poiseuille for Newtonian (valid at low shear rates)
   Q = (np.pi * radius**4 * pressure_gradient) / (8 * eta_0)
   print(f"Flow rate Q = {Q*1e6:.2f} mL/s")

   # Shear rate at wall
   gamma_dot_wall = (4 * Q) / (np.pi * radius**3)
   print(f"Wall shear rate: {gamma_dot_wall:.2f} s⁻¹")

   # Check if Newtonian assumption valid (need γ̇·τ₁ << 1)
   tau1 = model.parameters.get_value('tau1')
   Deborah = gamma_dot_wall * tau1
   print(f"Deborah number: {Deborah:.3f}")
   if Deborah > 0.1:
       print("Warning: Non-Newtonian effects significant")

**Die Swell Prediction:**

The viscosity ratio :math:`\eta_2/\eta_1` and relaxation time :math:`\tau_1` control elastic effects:

.. code-block:: python

   # Estimate die swell from elastic recoil
   eta1 = model.parameters.get_value('eta1')
   eta2 = model.parameters.get_value('eta2')
   tau1 = model.parameters.get_value('tau1')

   # First normal stress coefficient (Psi_1 ~ 2*eta2*tau1 for Jeffreys)
   Psi_1 = 2 * eta2 * tau1

   # Die swell ratio at shear rate gamma_dot
   gamma_dot = 100  # s^-1
   N1 = Psi_1 * gamma_dot**2
   swell_ratio = 1 + 0.13 * (N1 / (eta1 * gamma_dot))**(1/6)
   print(f"Predicted die swell ratio: {swell_ratio:.2f}")

**Mixing Time Estimation:**

.. code-block:: python

   # Estimate mixing time from viscosity and relaxation
   eta_0 = model.parameters.get_value('eta1')
   tau1 = model.parameters.get_value('tau1')

   # Power input for mixing
   P = 1000  # Watts
   volume = 0.01  # m^3
   energy_density = P / volume  # W/m^3

   # Characteristic mixing time (order of magnitude)
   t_mix = eta_0 / energy_density + 5 * tau1  # Viscous + elastic contributions
   print(f"Estimated mixing time: {t_mix:.1f} s")

Example Calculations
--------------------

**Complex Modulus Prediction:**

Given fitted parameters :math:`\eta_1` = 1000 Pa·s, :math:`\eta_2` = 500 Pa·s, :math:`\alpha = 0.5, \tau_1` = 10 s:

.. code-block:: python

   import numpy as np
   from rheojax.models import FractionalJeffreysModel
   from rheojax.core.jax_config import safe_import_jax

   jax, jnp = safe_import_jax()

   model = FractionalJeffreysModel()
   model.parameters.set_value('eta1', 1000.0)
   model.parameters.set_value('eta2', 500.0)
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('tau1', 10.0)

   # Predict over wide frequency range
   omega = jnp.logspace(-3, 2, 100)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = jnp.real(G_star)
   G_double_prime = jnp.imag(G_star)

   # Verify terminal behavior (G" ~ omega, G' ~ omega^2)
   eta_complex = G_star / (1j * omega)
   eta_prime = jnp.real(eta_complex)
   print(f"η'(ω→0) = {eta_prime[0]:.1f} Pa·s (expect η₁ = 1000)")

   # Find crossover frequency
   crossover_idx = jnp.argmin(jnp.abs(G_prime - G_double_prime))
   omega_c = omega[crossover_idx]
   print(f"G'/G\" crossover at ω = {omega_c:.3f} rad/s")

**Relaxation Modulus Prediction:**

.. code-block:: python

   # Predict stress relaxation
   t = jnp.logspace(-2, 4, 150)
   G_t = model.predict(t, test_mode='relaxation')

   # Check power-law decay region
   # For Jeffreys: G(t) ~ t^(-alpha) in intermediate regime
   log_G = jnp.log10(G_t)
   log_t = jnp.log10(t)

   # Fit slope in intermediate region (0.1 < t < 100 s)
   mask = (t > 0.1) & (t < 100)
   slope = jnp.polyfit(log_t[mask], log_G[mask], 1)[0]
   print(f"Power-law slope: {slope:.2f} (expect -α = -0.5)")

**Steady Shear Viscosity:**

.. code-block:: python

   # Jeffreys gives Newtonian behavior in steady shear
   gamma_dot = jnp.logspace(-2, 3, 50)
   sigma = model.predict(gamma_dot, test_mode='rotation')
   eta = sigma / gamma_dot

   print(f"η at low shear rate: {eta[0]:.1f} Pa·s (expect η₁)")
   print(f"η at high shear rate: {eta[-1]:.1f} Pa·s (still η₁, Newtonian)")

See Also
--------

- :doc:`fractional_maxwell_liquid` — single-springpot Maxwell analogue that forms one
  branch of the Jeffreys construction
- :doc:`fractional_kelvin_voigt` — parallel SpringPot + spring element providing the other
  branch
- :doc:`fractional_burgers` — combines Maxwell and fractional Kelvin-Voigt in series
- :doc:`../../transforms/fft` — obtain :math:`G'` and :math:`G''` prior to fitting Jeffreys
  spectra
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing
  Jeffreys, Burgers, and Maxwell families

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalJeffreysModel`

Usage
-----

Basic Fitting
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalJeffreysModel
   from rheojax.core.data import RheoData
   import numpy as np

   # Load experimental frequency sweep data
   omega = np.logspace(-2, 2, 50)
   G_prime = ...  # Storage modulus
   G_double_prime = ...  # Loss modulus
   G_star = G_prime + 1j * G_double_prime

   # Create RheoData object
   data = RheoData(x=omega, y=G_star, test_mode='oscillation')

   # Initialize and fit model
   model = FractionalJeffreysModel()
   result = model.fit(data)

   # Access fitted parameters
   eta1 = model.parameters.get_value('eta1')
   eta2 = model.parameters.get_value('eta2')
   alpha = model.parameters.get_value('alpha')
   tau1 = model.parameters.get_value('tau1')

   # Calculate derived quantities
   tau2 = (eta2 / eta1) * tau1  # Derived relaxation time
   eta_ratio = eta2 / eta1

   print(f"Primary viscosity η₁ = {eta1:.2e} Pa·s")
   print(f"Secondary viscosity η₂ = {eta2:.2e} Pa·s")
   print(f"Viscosity ratio η₂/η₁ = {eta_ratio:.3f}")
   print(f"Fractional order α = {alpha:.3f}")
   print(f"Relaxation time τ₁ = {tau1:.2e} s")
   print(f"Derived time τ₂ = {tau2:.2e} s")
   print(f"R² = {result.r_squared:.4f}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bayesian inference with uncertainty quantification
   # NLSQ fit first (warm-start for MCMC)
   model.fit(data)

   # Run Bayesian inference
   result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Extract credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )

   for param, (lower, upper) in intervals.items():
       mean = result.posterior_samples[param].mean()
       print(f"{param}: {mean:.2e} [{lower:.2e}, {upper:.2e}]")

   # Calculate derived quantity uncertainty
   eta1_samples = result.posterior_samples['eta1']
   eta2_samples = result.posterior_samples['eta2']
   ratio_samples = eta2_samples / eta1_samples

   ratio_mean = ratio_samples.mean()
   ratio_std = ratio_samples.std()
   print(f"η₂/η₁ = {ratio_mean:.3f} ± {ratio_std:.3f}")

   # Visualize posterior distributions
   import arviz as az
   inference_data = az.from_numpyro(result)
   az.plot_pair(
       inference_data,
       var_names=['eta1', 'eta2', 'alpha', 'tau1'],
       divergences=True
   )

Steady Shear Prediction
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Jeffreys model gives Newtonian behavior in steady shear
   gamma_dot = np.logspace(-2, 3, 50)
   sigma = model.predict(gamma_dot, test_mode='rotation')
   eta_apparent = sigma / gamma_dot

   # Should be constant at η₁
   print(f"Steady shear viscosity: {eta_apparent.mean():.2e} Pa·s")
   print(f"Compare to η₁: {eta1:.2e} Pa·s")

   # Plot flow curve
   import matplotlib.pyplot as plt
   plt.loglog(gamma_dot, sigma, 'o-')
   plt.xlabel('Shear rate (s⁻¹)')
   plt.ylabel('Shear stress (Pa)')
   plt.title('Flow Curve (Newtonian)')
   plt.grid(True)

Relaxation Modulus
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict stress relaxation after step strain
   time = np.logspace(-2, 4, 100)
   G_t = model.predict(time, test_mode='relaxation')

   # Plot relaxation modulus
   plt.figure(figsize=(8, 6))
   plt.loglog(time, G_t, 'o-')
   plt.xlabel('Time (s)')
   plt.ylabel('G(t) (Pa)')
   plt.title('Stress Relaxation')
   plt.grid(True)

   # Check power-law region
   # Slope should be -α in intermediate regime
   log_G = np.log10(G_t)
   log_t = np.log10(time)
   mask = (time > 0.1) & (time < 100)
   slope = np.polyfit(log_t[mask], log_G[mask], 1)[0]
   print(f"Power-law slope: {slope:.2f}")
   print(f"Expected: -α = {-alpha:.2f}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Complete workflow
   pipeline = (Pipeline()
       .load('oscillatory_data.csv', x_col='omega', y_col='G_star')
       .fit('fractional_jeffreys')
       .plot(style='publication')
       .save('jeffreys_results.hdf5'))

   # Access model
   model = pipeline.model
   eta1 = model.parameters.get_value('eta1')
   print(f"Zero-shear viscosity: {eta1:.2e} Pa·s")

Advanced: Multi-Sample Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare multiple samples or formulations
   samples = ['sample_A', 'sample_B', 'sample_C']
   results = {}

   for sample_name in samples:
       # Load data
       data = RheoData.from_file(f'{sample_name}.csv')

       # Fit model
       model = FractionalJeffreysModel()
       model.fit(data)

       # Store results
       results[sample_name] = {
           'eta1': model.parameters.get_value('eta1'),
           'eta2': model.parameters.get_value('eta2'),
           'alpha': model.parameters.get_value('alpha'),
           'tau1': model.parameters.get_value('tau1'),
       }

   # Compare viscosities
   import pandas as pd
   df = pd.DataFrame(results).T
   print(df)

   # Visualize parameter trends
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   for ax, param in zip(axes.flat, ['eta1', 'eta2', 'alpha', 'tau1']):
       ax.bar(samples, [results[s][param] for s in samples])
       ax.set_ylabel(param)
       ax.set_title(f'{param} Comparison')
   plt.tight_layout()

See also
--------

- :doc:`fractional_maxwell_liquid` — single-springpot Maxwell analogue that forms one
  branch of the Jeffreys construction.
- :doc:`fractional_kelvin_voigt` — parallel SpringPot + spring element providing the other
  branch.
- :doc:`fractional_burgers` — combines Maxwell and fractional Kelvin-Voigt in series.
- :doc:`../../transforms/fft` — obtain :math:`G'` and :math:`G''` prior to fitting Jeffreys
  spectra.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing
  Jeffreys, Burgers, and Maxwell families.

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Jeffreys, H. *The Earth: Its Origin, History and Physical Constitution*.
   Cambridge University Press (1929; 6th ed. 1976). ISBN: 978-0521206488

.. [3] Friedrich, C. "Relaxation and retardation functions of the Maxwell model
   with fractional derivatives." *Rheologica Acta*, 30, 151–158 (1991).
   https://doi.org/10.1007/BF01134604

.. [4] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [5] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724
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
