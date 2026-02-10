.. _model-fractional-kv-zener:

Fractional Kelvin-Voigt-Zener (Fractional)
==========================================

Quick Reference
---------------

- **Use when:** Creep/retardation analysis, solid with finite equilibrium compliance
- **Parameters:** 4 (Ge, Gk, :math:`\alpha, \tau`)
- **Key equation:** :math:`J(t) = \frac{1}{G_e} + \frac{1}{G_k}[1 - E_{\alpha}(-(t/\tau)^{\alpha})]`
- **Test modes:** Relaxation, creep, oscillation
- **Material examples:** Viscoelastic solids with retardation spectra, filled polymers, soft tissues

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
     - Series spring modulus (instantaneous response)
   * - :math:`G_k`
     - Pa
     - Kelvin element modulus (retardation magnitude)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < :math:`\alpha` < 1, spectrum breadth)
   * - :math:`\tau`
     - s
     - Retardation time (characteristic timescale)
   * - :math:`E_{\alpha}(z)`
     - dimensionless
     - One-parameter Mittag-Leffler function
   * - :math:`J^*(\omega)`
     - 1/Pa
     - Complex compliance

Overview
--------

A Fractional Kelvin-Voigt element (spring :math:`G_k` in parallel with SpringPot) in series with a spring :math:`G_e`. Natural for creep/retardation analysis with finite equilibrium compliance.

Physical Foundations
--------------------

The FKV-Zener model combines a series spring with a fractional Kelvin-Voigt element:

**Mechanical Configuration:**

.. code-block:: text

   [Spring Ge] ---- series ---- [Spring Gk parallel with SpringPot (α, τ)]

**Microstructural Interpretation:**

- **Series spring (Ge)**: Instantaneous elastic response (glassy compliance :math:`J_0` = 1/Ge)
- **Kelvin element (Gk)**: Delayed compliance from network rearrangements
- **SpringPot**: Fractional-order damping with broad relaxation spectrum
- **Solid behavior**: Equilibrium compliance J∞ = 1/Ge + 1/Gk (finite, bounded)

This model is particularly suited for creep analysis where the instantaneous and
equilibrium compliances are the primary observables.

Governing Equations
-------------------

Time domain (creep compliance):

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\Big[1 - E_{\alpha}\!\big(-(t/\tau)^{\alpha}\big)\Big].
   \]

Frequency domain (complex compliance and modulus):

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
     - Series spring modulus
   * - ``Gk``
     - :math:`G_k`
     - Pa
     - [1e-3, 1e9]
     - KV element modulus
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

- Instantaneous compliance :math:`1/G_e`; long-time compliance :math:`1/G_e + 1/G_k`.
- Power-law retardation over decades for :math:`0<\alpha<1`.
- Useful when creep is the primary observable.

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Zener in creep form.
- :math:`G_k \to \infty`: reduces to series spring only.

What You Can Learn
------------------

This section explains how to translate fitted Fractional Kelvin-Voigt-Zener parameters
into material insights and actionable knowledge, with emphasis on creep and retardation
analysis for viscoelastic solids.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Series Spring Modulus (Ge)**:
   Controls the instantaneous elastic compliance upon stress application.

   - **For graduate students**: In polymers, :math:`G_e` reflects the glassy modulus from
     short-range chain stretching and bond angle deformations. For crosslinked networks,
     :math:`G_e \approx G_\infty` (high-frequency plateau). Typical values: :math:`10^6-10^9` Pa for
     glassy polymers, :math:`10^3-10^6` Pa for rubbery materials.
   - **For practitioners**: :math:`J_0 = 1/G_e` is the immediate strain upon stress application,
     critical for impact resistance and short-time deformation. Use to assess material stiffness
     and load-bearing capacity.

**Kelvin Element Modulus (Gk)**:
   Controls the magnitude of delayed (retarded) elastic deformation.

   - **Equilibrium compliance**: :math:`J_\infty = 1/G_e + 1/G_k` (bounded, solid-like)
   - **Retardation magnitude**: :math:`\Delta J = 1/G_k` measures the delayed compliance
   - **Modulus ratio**: :math:`G_k/G_e` characterizes creep severity (high ratio → large creep)
   - **For practitioners**: Lower Gk means more compliant material under sustained load. Critical
     for dimensional stability in structural applications.

**Fractional Order (** :math:`\alpha` **)**:
   Governs the power-law character and breadth of the retardation spectrum.

   - :math:`\alpha` **→ 0.2-0.3**: Very broad spectrum, highly heterogeneous (filled elastomers, nanocomposites,
     asphalt with wide filler size distribution)
   - :math:`\alpha` **→ 0.4-0.6**: Moderate breadth, typical for polymeric solids with distributed retardation
     (semicrystalline polymers, physical gels, soft tissues)
   - :math:`\alpha` **→ 0.7-0.9**: Narrow spectrum, approaching classical exponential retardation (uniform
     crosslinked networks, monodisperse elastomers)
   - :math:`\alpha` **→ 1**: Classical Zener (single exponential), use simpler model

   *Physical interpretation*: Lower :math:`\alpha` indicates broader distribution of retardation times from
   structural heterogeneity (filler distribution, crosslink density variation, morphological
   polydispersity in semicrystalline polymers). For filled systems, :math:`\alpha` decreases with increasing
   filler volume fraction due to filler-matrix interphase effects.

   *For practitioners*: Monitor :math:`\alpha` for quality control. Decreased :math:`\alpha` may indicate poor dispersion
   of fillers, incomplete curing, or aging-induced microstructural heterogeneity.

**Retardation Time (** :math:`\tau` **)**:
   Characteristic timescale for the transition from instantaneous to equilibrium compliance.

   - **Marks creep regime transition**: :math:`J(t \ll \tau) \approx J_0`, :math:`J(t \gg \tau) \approx J_\infty`
   - **Temperature-dependent**: Follows WLF or Arrhenius behavior; :math:`\tau` decreases with temperature
   - **For practitioners**: Compare :math:`\tau` to service life. For long-term applications (e.g., gaskets,
     seals), ensure load duration < :math:`\tau/10` to minimize creep.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Fractional Kelvin-Voigt-Zener Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - High Ge (>\ :math:`10^6` Pa), Gk/Ge < 0.1
     - Stiff solid, minimal creep
     - Thermosets, vulcanized rubber, engineering plastics
     - Excellent dimensional stability
   * - Moderate Ge (:math:`10^4-10^6` Pa), Gk/Ge ≈ 0.5
     - Balanced solid, moderate creep
     - Filled elastomers, semicrystalline polymers
     - Good for seals, dampers (controlled deformation)
   * - Low Ge (<:math:`10^4` Pa), Gk/Ge > 1.0
     - Soft solid, significant creep
     - Hydrogels, soft tissues, weak physical gels
     - Poor load-bearing, tissue engineering applications
   * - Low :math:`\alpha` (<0.3), any Ge
     - Heterogeneous structure
     - Nanocomposites, asphalt, bitumen
     - Broad time-dependent behavior, challenging QC

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **Creep curve linear at long time**: Material is flowing (unbounded creep); use Fractional Burgers
  model instead to capture viscous flow.
- **Poor fit at short time**: Insufficient early-time resolution or need for additional instantaneous
  compliance term (glassy contribution).
- :math:`\alpha` **near bounds (0.05 or 0.95)**: Data may not support fractional retardation; try classical Zener
  (:math:`\alpha` = 1) or pure spring (:math:`\alpha` → 0).
- **Strong Gk-** :math:`\tau` **correlation**: Retardation time well-constrained but magnitude ambiguous. Need broader
  time coverage spanning 10\ :math:`^{-2 to 10^4}` seconds.
- **Ge << Gk**: Non-physical (instantaneous stiffer than equilibrium); check data quality or bounds.

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Creep test** (primary): 4-5 decades in time, verify equilibrium plateau
2. **Sampling**: Log-spaced, 50+ points per decade
3. **Stress level**: Within LVR (verify with stress sweep)
4. **Temperature**: Constant ±0.1°C

**Initialization Strategy:**

.. code-block:: text

   # From creep compliance J(t)
   Ge_init = 1 / J(t → 0)  # Instantaneous compliance
   Gk_init = 1 / (J(t → ∞) - 1/Ge_init)  # Retardation magnitude
   tau_init = time where retardation is 50% complete
   alpha_init = 0.5  # Default

**Optimization Tips:**

- Fit in compliance space (natural for creep)
- Use log-weighted least squares
- Constrain Ge > 0, Gk > 0 (physically meaningful)
- Verify equilibrium compliance is reached

**Common Pitfalls:**

- **Insufficient long-time data**: Cannot determine Gk accurately
- **Non-monotonic J(t)**: Check for instrument artifacts or nonlinear effects
- **Strong Gk-** :math:`\tau` **correlation**: Need better coverage in retardation regime

**Troubleshooting Table:**

.. list-table:: Common Issues and Solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Issue
     - Likely Cause
     - Solution
   * - J(t) shows linear growth
     - Material is flowing
     - Use Fractional Burgers for viscous flow
   * - Poor fit at short times
     - Glassy contribution missing
     - Need higher time resolution or add term
   * - :math:`\alpha` near bounds (0.05 or 0.95)
     - Not fractional
     - Try classical Zener (:math:`\alpha` = 1)
   * - Strong Gk-:math:`\tau` correlation (>0.9)
     - Narrow time window
     - Extend to 10\ :math:`^{-2 - 10^4}` seconds
   * - Ge << Gk (non-physical)
     - Data quality or bounds
     - Check instantaneous response
   * - Oscillations in residuals
     - Instrument resonance
     - Filter data or change geometry
   * - Fit diverges
     - Poor initialization
     - Estimate from compliance limits

Practical Applications
----------------------

**Creep Prediction for Design:**

The FKV-Zener model enables long-term deformation prediction for gaskets, seals, and load-bearing components:

.. code-block:: python

   # Predict dimensional stability under sustained load
   sigma_applied = 1.0e6  # 1 MPa sustained stress
   t_service = 3.156e7  # 1 year in seconds

   # Compute total compliance
   J_total = model.predict(t_service, test_mode='creep') / sigma_applied

   # Convert to strain
   epsilon_total = sigma_applied * J_total
   print(f"Expected strain after 1 year: {epsilon_total * 100:.2f}%")

   # Separate instantaneous and delayed contributions
   J_instantaneous = 1 / Ge
   J_delayed = J_total - J_instantaneous
   print(f"Instantaneous: {J_instantaneous * sigma_applied * 100:.2f}%")
   print(f"Delayed creep: {J_delayed * sigma_applied * 100:.2f}%")

**Material Selection:**

Compare candidate materials by equilibrium compliance:

.. list-table:: Material Comparison
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Material
     - :math:`J_0` (1/Pa)
     - J∞ (1/Pa)
     - :math:`\tau` (s)
     - :math:`\alpha`
   * - EPDM rubber
     - :math:`1.0 \times 10^{-6}`
     - :math:`2.5 \times 10^{-6}`
     - 100
     - 0.45
   * - Silicone gel
     - :math:`5.0 \times 10^{-4}`
     - :math:`8.0 \times 10^{-4}`
     - 10
     - 0.55
   * - Polyurethane foam
     - :math:`1.0 \times 10^{-5}`
     - :math:`3.0 \times 10^{-5}`
     - 50
     - 0.40

**Quality Control Metrics:**

Use fitted parameters for batch consistency:

1. **Instantaneous compliance** :math:`J_0` **= 1/Ge**: Monitors cure state (decreases with crosslinking)
2. **Equilibrium compliance J∞**: Tracks network integrity (increases with aging/degradation)
3. **Retardation time** :math:`\tau`: Sensitive to filler content and dispersion
4. **Fractional order** :math:`\alpha`: Indicates microstructural heterogeneity (target :math:`\alpha` > 0.4 for uniformity)

**Failure Analysis:**

Diagnose creep failure modes from parameter trends:

- **Increasing J∞ over time**: Network degradation (oxidation, hydrolysis, chain scission)
- **Decreasing** :math:`\alpha`: Loss of microstructural homogeneity (filler agglomeration)
- **Increasing** :math:`\tau`: Molecular weight increase (post-cure, crosslinking)
- **J∞ approaches** :math:`J_0`: Loss of delayed response (complete network breakdown)

Example Calculations
--------------------

**Creep Compliance Prediction:**

Given fitted parameters Ge = 1.0 MPa, Gk = 2.5 MPa, :math:`\alpha = 0.5, \tau` = 100 s:

.. code-block:: python

   import numpy as np
   from rheojax.models import FractionalKelvinVoigtZener
   from rheojax.core.jax_config import safe_import_jax

   jax, jnp = safe_import_jax()

   model = FractionalKelvinVoigtZener()
   model.parameters.set_value('Ge', 1.0e6)  # Pa
   model.parameters.set_value('Gk', 2.5e6)  # Pa
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('tau', 100.0)  # s

   # Predict creep compliance
   t = jnp.logspace(-2, 6, 200)  # 0.01 s to ~11 days
   J_t = model.predict(t, test_mode='creep')

   # Check limiting behavior
   J0 = 1 / 1.0e6  # 1/Ge
   Jinf = J0 + 1/2.5e6  # J0 + 1/Gk
   print(f"J(t=0) = {J_t[0]:.2e} Pa⁻¹ (theory: {J0:.2e})")
   print(f"J(t→∞) = {J_t[-1]:.2e} Pa⁻¹ (theory: {Jinf:.2e})")

   # Time to reach 90% of equilibrium compliance
   J_90 = J0 + 0.9 * (Jinf - J0)
   t_90 = t[jnp.argmin(jnp.abs(J_t - J_90))]
   print(f"90% retardation complete at t = {t_90:.1f} s")

**Conversion to Frequency Domain:**

.. code-block:: python

   # Convert fitted creep parameters to modulus prediction
   omega = jnp.logspace(-4, 2, 100)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = jnp.real(G_star)
   G_double_prime = jnp.imag(G_star)

   # Verify low-frequency plateau
   G_eq = 1 / Jinf  # Equilibrium modulus
   print(f"G'(ω→0) = {G_prime[0]:.2e} Pa")
   print(f"Geq = {G_eq:.2e} Pa (from compliance)")

   # Find tan(δ) peak (maximum damping)
   tan_delta = G_double_prime / G_prime
   omega_peak = omega[jnp.argmax(tan_delta)]
   print(f"tan(δ) peak at ω ≈ {omega_peak:.3f} rad/s")
   print(f"Compare to 1/τ = {1/100:.3f} rad/s")

See Also
--------

- :doc:`fractional_kelvin_voigt` — parallel element used inside the FKZ construction
- :doc:`fractional_zener_sl` — adds a fractional Maxwell branch instead of a Kelvin one
- :doc:`fractional_maxwell_model` — most general two-order series formulation
- :doc:`../../transforms/mastercurve` — align creep spectra across temperature before FKZ
  fitting
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — comparisons of Zener
  variants across datasets

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalKelvinVoigtZener`

Usage
-----

Basic Creep Fitting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalKelvinVoigtZener
   from rheojax.core.data import RheoData
   import numpy as np

   # Load experimental creep data
   time = np.logspace(-1, 4, 100)  # 0.1 to 10,000 s
   compliance = ...  # Measured creep compliance J(t)

   # Create RheoData object
   data = RheoData(x=time, y=compliance, test_mode='creep')

   # Initialize and fit model
   model = FractionalKelvinVoigtZener()
   result = model.fit(data)

   # Access fitted parameters
   Ge = model.parameters.get_value('Ge')
   Gk = model.parameters.get_value('Gk')
   alpha = model.parameters.get_value('alpha')
   tau = model.parameters.get_value('tau')

   print(f"Instantaneous modulus Ge = {Ge:.2e} Pa")
   print(f"Retardation modulus Gk = {Gk:.2e} Pa")
   print(f"Instantaneous compliance J₀ = {1/Ge:.2e} Pa⁻¹")
   print(f"Equilibrium compliance J∞ = {1/Ge + 1/Gk:.2e} Pa⁻¹")
   print(f"Fractional order α = {alpha:.3f}")
   print(f"Retardation time τ = {tau:.2e} s")
   print(f"R² = {result.r_squared:.4f}")

Bayesian Inference for Creep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bayesian inference with uncertainty quantification
   # NLSQ fit first (warm-start)
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

   # Calculate derived quantities with uncertainty
   Ge_samples = result.posterior_samples['Ge']
   Gk_samples = result.posterior_samples['Gk']
   J0_samples = 1 / Ge_samples
   Jinf_samples = 1/Ge_samples + 1/Gk_samples

   print(f"J₀ = {J0_samples.mean():.2e} ± {J0_samples.std():.2e} Pa⁻¹")
   print(f"J∞ = {Jinf_samples.mean():.2e} ± {Jinf_samples.std():.2e} Pa⁻¹")

   # Visualize posterior distributions
   import arviz as az
   inference_data = az.from_numpyro(result)
   az.plot_pair(
       inference_data,
       var_names=['Ge', 'Gk', 'alpha', 'tau'],
       divergences=True
   )

Oscillatory Fitting
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # FKV-Zener can also fit frequency sweep data
   omega = np.logspace(-2, 2, 50)
   G_star = ...  # Complex modulus from SAOS

   data_osc = RheoData(x=omega, y=G_star, test_mode='oscillation')
   model.fit(data_osc)

   # Predict in compliance space
   J_star = 1 / G_star  # Complex compliance
   J_pred = model.predict(omega, test_mode='compliance')

   # Compare
   import matplotlib.pyplot as plt
   plt.loglog(omega, np.abs(J_star), 'o', label='Data')
   plt.loglog(omega, np.abs(J_pred), '-', label='Model')
   plt.xlabel('ω (rad/s)')
   plt.ylabel('|J*| (Pa⁻¹)')
   plt.legend()

Advanced: Temperature-Dependent Creep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Fit creep at multiple temperatures
   temperatures = [25, 40, 55, 70]  # °C
   creep_data = {...}  # Dictionary of T: (time, compliance)

   # Fit each temperature
   params_vs_T = {}
   for T in temperatures:
       data_T = RheoData(
           x=creep_data[T]['time'],
           y=creep_data[T]['compliance'],
           test_mode='creep'
       )
       model.fit(data_T)
       params_vs_T[T] = {
           'Ge': model.parameters.get_value('Ge'),
           'Gk': model.parameters.get_value('Gk'),
           'alpha': model.parameters.get_value('alpha'),
           'tau': model.parameters.get_value('tau')
       }

   # Analyze temperature dependence
   import matplotlib.pyplot as plt
   T_array = np.array(temperatures)
   tau_array = np.array([params_vs_T[T]['tau'] for T in temperatures])

   # Arrhenius plot
   plt.semilogy(1/(T_array + 273.15), tau_array, 'o-')
   plt.xlabel('1/T (K⁻¹)')
   plt.ylabel('τ (s)')
   plt.title('Retardation Time Temperature Dependence')

   # Fit Arrhenius: tau = tau0 * exp(Ea / RT)
   from scipy.optimize import curve_fit
   def arrhenius(T_inv, tau0, Ea_over_R):
       return tau0 * np.exp(Ea_over_R * T_inv)

   popt, _ = curve_fit(arrhenius, 1/(T_array + 273.15), tau_array)
   Ea = popt[1] * 8.314  # Activation energy in J/mol
   print(f"Activation energy: {Ea/1000:.1f} kJ/mol")

See also
--------

- :doc:`fractional_kelvin_voigt` — parallel element used inside the FKZ construction.
- :doc:`fractional_zener_sl` — adds a fractional Maxwell branch instead of a Kelvin one.
- :doc:`fractional_maxwell_model` — most general two-order series formulation.
- :doc:`../../transforms/mastercurve` — align creep spectra across temperature before FKZ
  fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — comparisons of Zener
  variants across datasets.

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Bagley, R. L., and Torvik, P. J. "On the fractional calculus model of
   viscoelastic behavior." *Journal of Rheology*, 30, 133–155 (1986).
   https://doi.org/10.1122/1.549887

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

