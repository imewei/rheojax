.. _dense_suspensions_glasses:

Dense Suspensions and Glassy Materials
=======================================

Integration Through Transients Mode-Coupling Theory (ITT-MCT) provides a first-principles framework for understanding the rheology of dense colloidal suspensions and glasses. Unlike phenomenological models, MCT derives rheological behavior from microscopic particle interactions through memory integrals and non-equilibrium statistical mechanics.

.. admonition:: Key Insight
   :class: tip

   ITT-MCT connects microscopic cage dynamics to macroscopic rheology through the non-equilibrium correlator :math:`\Phi(t,t')`, which tracks how particles become trapped and escape from their neighbors under flow.

Overview
--------

Mode-Coupling Theory (MCT) was originally developed to describe the glass transition in equilibrium systems. The Integration Through Transients (ITT) extension handles non-equilibrium rheological flows by tracking the time evolution of structural correlations under deformation.

**Key Physical Concepts:**

1. **Cage Effect**: Particles in dense suspensions are trapped by their neighbors, creating temporary elastic cages
2. **Glass Transition**: At critical density, cages become permanent and the system develops a yield stress
3. **Memory Kernels**: Past deformations influence current stress through memory integrals
4. **Strain Decorrelation**: Applied strain breaks cages, allowing relaxation
5. **Non-Ergodicity**: Glassy systems cannot explore all configurations at long times

**Two Model Variants:**

- **Schematic MCT** (:math:`F_{12}`): Fast, semi-quantitative, 5-6 parameters
- **Isotropic MCT (ISM)**: Quantitative with structure factor S(k), more expensive

When to Use This Model
----------------------

.. list-table:: Application Guide
   :widths: 30 70
   :header-rows: 1

   * - Use ITT-MCT When
     - Use Alternatives When
   * - Dense colloidal suspensions (:math:`\phi > 0.45`)
     - Dilute suspensions (Einstein viscosity)
   * - Hard-sphere or near-hard-sphere systems
     - Soft particles (use SGR)
   * - Glass transition characterization needed
     - Only phenomenological fits required
   * - Microscopic understanding desired
     - Fast empirical model sufficient (power-law, etc.)
   * - Predicting yield stress from interactions
     - Yield stress from direct measurement (Herschel-Bulkley)
   * - Aging and rejuvenation dynamics
     - Simple thixotropic models adequate (DMT, Fluidity)

.. warning::
   ITT-MCT models are computationally expensive due to Volterra integral equations. First JIT compilation takes 30-90 seconds. Use precompilation for production workflows.

Theoretical Foundations
-----------------------

Generalized Langevin Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MCT framework describes the evolution of the density correlator through a generalized Langevin equation:

.. math::

   \frac{\partial}{\partial t} \Phi(t,t') + \Gamma_0 \Phi(t,t') + \Gamma_0 \int_{t'}^{t} m(\Phi(t,s)) \frac{\partial \Phi(t,s)}{\partial s} ds = 0

where:

- :math:`\Phi(t,t')` is the non-equilibrium correlator (0 = fully decorrelated, 1 = fully correlated)
- :math:`\Gamma_0` is the bare relaxation rate
- :math:`m(\Phi)` is the memory kernel encoding particle interactions
- Integration from :math:`t'` (past) to :math:`t` (present) captures memory effects

:math:`F_{12}` Schematic Memory Kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The schematic MCT model uses a simplified memory kernel:

.. math::

   m(\Phi) = v_1 \Phi + v_2 \Phi^2

The glass transition occurs at the critical point :math:`v_2 = 4` (for :math:`F_{12}` model). The distance from the glass transition is:

.. math::

   \epsilon = \frac{v_2 - 4}{4}

- :math:`\epsilon > 0`: Glass state (permanent cages, yield stress)
- :math:`\epsilon < 0`: Fluid state (transient cages, no yield stress)
- :math:`\epsilon \approx 0`: Critical point (power-law relaxation)

Strain Decorrelation
^^^^^^^^^^^^^^^^^^^^

Applied deformation breaks cages through a strain-dependent damping function:

.. math::

   h(\gamma) = \exp\left(-\left(\frac{\gamma}{\gamma_c}\right)^2\right)

where :math:`\gamma_c` is the critical cage strain (typically 0.05-0.15). The effective correlator under flow becomes:

.. math::

   \Phi_{\text{eff}}(t,t') = h(\gamma(t) - \gamma(t')) \Phi(t,t')

Stress Calculation
^^^^^^^^^^^^^^^^^^

The shear stress follows from the correlator through:

.. math::

   \sigma(t) = G_0 \int_{-\infty}^{t} \dot{\gamma}(t') \Phi(t,t') dt'

where :math:`G_0` is the high-frequency elastic modulus. For oscillatory deformations :math:`\gamma(t) = \gamma_0 \sin(\omega t)`, this yields complex modulus :math:`G^*(\omega)`.

Glass Transition Phenomenology
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Near the glass transition, MCT predicts:

- :math:`\beta`\ **-relaxation**: Fast cage rattling, time scale :math:`\sim 1/\Gamma_0`
- :math:`\alpha`\ **-relaxation**: Slow cage escape, time scale :math:`\tau_\alpha \sim |\epsilon|^{-\gamma}` (power-law divergence)
- **Non-ergodicity parameter**: :math:`f_{\text{neq}} = \lim_{t \to \infty} \Phi(t,0)` jumps from 0 (fluid) to :math:`>0` (glass)

For :math:`\epsilon > 0`, the system cannot fully relax and develops a yield stress:

.. math::

   \tau_y \sim G_0 f_{\text{neq}}

Practical Implementation
------------------------

Schematic MCT: Basic Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rheojax.models import ITTMCTSchematic
   import numpy as np

   # Create model in glass state
   model = ITTMCTSchematic(epsilon=0.1)  # ε > 0 → glass

   # Check glass transition properties
   info = model.get_glass_transition_info()
   print(f"Is glass: {info['is_glass']}")
   print(f"Distance from transition: ε = {info['epsilon']:.3f}")
   print(f"Non-ergodicity parameter: f_neq = {info['f_neq']:.3f}")
   print(f"Critical v2: {info['v2_critical']:.1f}")

   # Precompile for fast subsequent predictions (30-90s first time)
   compile_time = model.precompile()
   print(f"JIT compilation completed in {compile_time:.1f} seconds")

.. note::
   The ``precompile()`` method triggers JIT compilation of the Volterra solver. Call this once at the start of your workflow to avoid compilation delays during fitting or prediction.

Flow Curves: Yield Stress Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Predict flow curve
   gamma_dot = np.logspace(-4, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='flow_curve')

   # Extract yield stress (σ → σ_y as γ̇ → 0)
   sigma_y = sigma[0]  # Stress at lowest shear rate
   print(f"Yield stress: τ_y ≈ {sigma_y:.2f} Pa")

   # Compare glass vs fluid
   model_fluid = ITTMCTSchematic(epsilon=-0.1)  # ε < 0 → fluid
   sigma_fluid = model_fluid.predict(gamma_dot, test_mode='flow_curve')

   import matplotlib.pyplot as plt
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(gamma_dot, sigma, 'o-', label=f'Glass (ε={0.1})')
   ax.loglog(gamma_dot, sigma_fluid, 's-', label=f'Fluid (ε={-0.1})')
   ax.axhline(sigma_y, color='red', linestyle='--', alpha=0.5, label='Yield stress')
   ax.set_xlabel('Shear rate γ̇ (1/s)')
   ax.set_ylabel('Shear stress σ (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

.. admonition:: Key Insight
   :class: tip

   In the glass state (:math:`\varepsilon > 0`), the flow curve exhibits a finite yield stress followed by shear thinning. In the fluid state (:math:`\varepsilon < 0`), stress scales smoothly from Newtonian at low rates to shear thinning at high rates.

Small-Amplitude Oscillatory Shear (SAOS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Predict complex modulus
   omega = np.logspace(-4, 4, 100)
   G_star = model.predict(omega, test_mode='oscillation')

   # Extract components
   G_prime = G_star.real  # Storage modulus G'
   G_double_prime = G_star.imag  # Loss modulus G''

   # Characteristic features
   idx_cross = np.where(G_prime > G_double_prime)[0][0]
   omega_cross = omega[idx_cross]
   print(f"G' = G'' crossover at ω ≈ {omega_cross:.2e} rad/s")

   # Plot
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(omega, G_prime, 'o-', label="G' (storage)")
   ax.loglog(omega, G_double_prime, 's-', label='G" (loss)')
   ax.axvline(omega_cross, color='red', linestyle='--', alpha=0.5,
              label='Crossover')
   ax.set_xlabel('Frequency ω (rad/s)')
   ax.set_ylabel('Modulus (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

.. note::
   For glasses (:math:`\varepsilon > 0`), :math:`G'` dominates at low frequencies (elastic plateau). For fluids (:math:`\varepsilon < 0`), :math:`G''` dominates (viscous behavior). The crossover frequency relates to the structural relaxation time :math:`\tau_\alpha`.

Startup Shear: Stress Overshoot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Startup transient at fixed shear rate
   t = np.linspace(0.01, 100, 500)
   gamma_dot = 10.0  # 1/s

   sigma_startup = model.predict(t, test_mode='startup', gamma_dot=gamma_dot)

   # Find stress overshoot
   idx_max = np.argmax(sigma_startup)
   t_max = t[idx_max]
   sigma_max = sigma_startup[idx_max]
   sigma_steady = sigma_startup[-1]

   print(f"Peak stress: σ_max = {sigma_max:.2f} Pa at t = {t_max:.2f} s")
   print(f"Steady stress: σ_∞ = {sigma_steady:.2f} Pa")
   print(f"Overshoot ratio: σ_max/σ_∞ = {sigma_max/sigma_steady:.2f}")

   # Plot
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.plot(t, sigma_startup, 'o-')
   ax.axhline(sigma_steady, color='red', linestyle='--', alpha=0.5,
              label='Steady state')
   ax.plot(t_max, sigma_max, 'ro', markersize=10, label='Overshoot')
   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Shear stress σ (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

.. admonition:: Key Insight
   :class: tip

   Stress overshoot in startup shear reflects the breaking of the initial cage structure. The overshoot magnitude and time scale are signatures of glass-like behavior.

LAOS: Higher Harmonics
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Large-amplitude oscillatory shear
   t = np.linspace(0, 10 * 2 * np.pi, 1000)
   gamma_0 = 0.2  # Strain amplitude
   omega = 1.0    # Frequency (rad/s)

   # Extract harmonics (σ = Σ G'_n sin(nωt) + G''_n cos(nωt))
   sigma_prime, sigma_double_prime = model.get_laos_harmonics(
       t, gamma_0=gamma_0, omega=omega
   )

   # First harmonic (n=1)
   G1_prime = sigma_prime[0] / gamma_0
   G1_double_prime = sigma_double_prime[0] / gamma_0

   # Third harmonic (n=3) - nonlinearity indicator
   G3_prime = sigma_prime[2] / gamma_0
   G3_double_prime = sigma_double_prime[2] / gamma_0

   print(f"Linear regime: G'_1 = {G1_prime:.2e} Pa")
   print(f"Nonlinearity: G'_3 = {G3_prime:.2e} Pa")
   print(f"Nonlinearity ratio: G'_3/G'_1 = {G3_prime/G1_prime:.2e}")

Isotropic MCT: Quantitative Predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rheojax.models import ITTMCTIsotropic

   # Create model with volume fraction
   phi = 0.55  # Hard-sphere volume fraction
   model_ism = ITTMCTIsotropic(phi=phi)

   # ISM uses Percus-Yevick structure factor internally
   # More accurate for hard-sphere colloids, but slower

   # Predict SAOS
   omega = np.logspace(-4, 4, 100)
   G_star_ism = model_ism.predict(omega, test_mode='oscillation')

   # Compare with schematic
   model_schem = ITTMCTSchematic(epsilon=0.1)
   G_star_schem = model_schem.predict(omega, test_mode='oscillation')

   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(omega, G_star_ism.real, 'o-', label='ISM (quantitative)')
   ax.loglog(omega, G_star_schem.real, 's--', label='Schematic (semi-quantitative)')
   ax.set_xlabel('Frequency ω (rad/s)')
   ax.set_ylabel("G' (Pa)")
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

.. warning::
   ISM model is significantly slower than schematic MCT due to structure factor calculations. Use schematic for exploratory analysis, ISM for quantitative comparisons with hard-sphere experiments.

Fitting Experimental Data
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Load experimental data
   omega_exp = np.array([...])  # rad/s
   G_prime_exp = np.array([...])  # Pa
   G_double_prime_exp = np.array([...])  # Pa
   G_star_exp = G_prime_exp + 1j * G_double_prime_exp

   # Fit with NLSQ (fast point estimation)
   model = ITTMCTSchematic()
   model.fit(omega_exp, G_star_exp, test_mode='oscillation')

   # Extract fitted parameters
   epsilon_fit = model.parameters.get_value('epsilon')
   gamma_c_fit = model.parameters.get_value('gamma_c')
   G0_fit = model.parameters.get_value('G_0')

   print(f"Fitted ε = {epsilon_fit:.3f}")
   print(f"Fitted γ_c = {gamma_c_fit:.3f}")
   print(f"Fitted G_0 = {G0_fit:.2e} Pa")

   # Interpret results
   if epsilon_fit > 0.01:
       print("→ Glass state: system has yield stress")
   elif epsilon_fit < -0.01:
       print("→ Fluid state: system flows at all rates")
   else:
       print("→ Near critical point: power-law relaxation")

   # Predict and compare
   G_star_pred = model.predict(omega_exp, test_mode='oscillation')

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
   ax1.loglog(omega_exp, G_prime_exp, 'o', label='Exp G\'')
   ax1.loglog(omega_exp, G_star_pred.real, '-', label='Fit G\'')
   ax1.set_xlabel('ω (rad/s)')
   ax1.set_ylabel("G' (Pa)")
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   ax2.loglog(omega_exp, G_double_prime_exp, 's', label='Exp G"')
   ax2.loglog(omega_exp, G_star_pred.imag, '-', label='Fit G"')
   ax2.set_xlabel('ω (rad/s)')
   ax2.set_ylabel('G" (Pa)')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Bayesian Inference: Uncertainty Quantification
-----------------------------------------------

Phase Classification with Uncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline
   import arviz as az

   # Bayesian workflow with NLSQ warm-start
   model = ITTMCTSchematic()
   result = model.fit_bayesian(
       omega_exp, G_star_exp,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

   # Extract epsilon posterior
   epsilon_samples = result.posterior_samples['epsilon']
   epsilon_mean = np.mean(epsilon_samples)
   epsilon_std = np.std(epsilon_samples)

   # Credible interval
   intervals = model.get_credible_intervals(result.posterior_samples,
                                            credibility=0.95)
   eps_low, eps_high = intervals['epsilon']

   print(f"ε = {epsilon_mean:.3f} ± {epsilon_std:.3f}")
   print(f"95% CI: [{eps_low:.3f}, {eps_high:.3f}]")

   # Probability of glass state
   p_glass = np.mean(epsilon_samples > 0)
   print(f"P(glass state | data) = {p_glass:.1%}")

   # Plot posterior
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.hist(epsilon_samples, bins=50, density=True, alpha=0.7)
   ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Glass transition')
   ax.axvline(epsilon_mean, color='blue', linestyle='-', linewidth=2, label='Mean')
   ax.axvspan(eps_low, eps_high, alpha=0.2, color='blue', label='95% CI')
   ax.set_xlabel('ε')
   ax.set_ylabel('Posterior density')
   ax.legend()
   plt.tight_layout()
   plt.show()

.. admonition:: Key Insight
   :class: tip

   Bayesian inference provides probabilistic phase classification. If the 95% credible interval for :math:`\varepsilon` spans zero, the data cannot definitively classify the system as glass or fluid — more data or different protocols needed.

Diagnostics and Convergence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Check MCMC diagnostics
   idata = az.from_dict(posterior=result.posterior_samples)

   # R-hat (should be < 1.01 for convergence)
   rhat = az.rhat(idata)
   print("R-hat values:")
   print(rhat)

   # Effective sample size (should be > 400 per chain)
   ess = az.ess(idata)
   print("\nEffective sample sizes:")
   print(ess)

   # Trace plots
   az.plot_trace(idata, var_names=['epsilon', 'gamma_c', 'G_0'])
   plt.tight_layout()
   plt.show()

   # Pair plot (parameter correlations)
   az.plot_pair(idata, var_names=['epsilon', 'gamma_c', 'v1', 'v2'],
                divergences=True)
   plt.tight_layout()
   plt.show()

   # Forest plot (credible intervals)
   az.plot_forest(idata, var_names=['epsilon', 'gamma_c', 'G_0', 'Gamma_0'],
                  hdi_prob=0.95)
   plt.tight_layout()
   plt.show()

.. note::
   High :math:`\hat{R}` (>1.01) or low ESS (<400) indicates poor convergence. Increase ``num_warmup`` or ``num_samples``. Check for divergences (yellow in pair plot) — may indicate difficult posterior geometry.

Visualization and Interpretation
---------------------------------

Master Curves: Frequency-Temperature Superposition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Measure SAOS at multiple temperatures
   temperatures = [20, 30, 40, 50, 60]  # °C
   datasets = []

   for T in temperatures:
       omega_T = np.logspace(-2, 2, 50)
       # ... measure G_star_T experimentally ...
       datasets.append({'omega': omega_T, 'G_star': G_star_T, 'T': T})

   # Fit MCT at each temperature
   epsilons = []
   for data in datasets:
       model = ITTMCTSchematic()
       model.fit(data['omega'], data['G_star'], test_mode='oscillation')
       eps = model.parameters.get_value('epsilon')
       epsilons.append(eps)

   # Plot ε(T) to locate glass transition temperature
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.plot(temperatures, epsilons, 'o-')
   ax.axhline(0, color='red', linestyle='--', label='Glass transition')
   ax.set_xlabel('Temperature (°C)')
   ax.set_ylabel('ε')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Lissajous Curves: LAOS Visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Generate LAOS cycle
   t = np.linspace(0, 2 * np.pi, 1000)
   gamma_0 = 0.5  # Large strain amplitude
   omega = 1.0

   strain = gamma_0 * np.sin(omega * t)
   stress = model.predict(strain, test_mode='laos', omega=omega)

   # Elastic Lissajous (σ vs γ)
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

   ax1.plot(strain, stress, 'o-')
   ax1.set_xlabel('Strain γ')
   ax1.set_ylabel('Stress σ (Pa)')
   ax1.set_title('Elastic Lissajous')
   ax1.grid(True, alpha=0.3)

   # Viscous Lissajous (σ vs γ̇)
   strain_rate = gamma_0 * omega * np.cos(omega * t)
   ax2.plot(strain_rate, stress, 's-', color='orange')
   ax2.set_xlabel('Strain rate γ̇ (1/s)')
   ax2.set_ylabel('Stress σ (Pa)')
   ax2.set_title('Viscous Lissajous')
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

.. note::
   Distortion from elliptical shape indicates nonlinear behavior. For glasses, Lissajous curves show secondary loops (cage reformation) at large strains.

Comparison with Other Models
-----------------------------

ITT-MCT vs SGR
^^^^^^^^^^^^^^

.. list-table:: Model Comparison
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - ITT-MCT
     - SGR (Soft Glassy Rheology)
   * - Foundation
     - Microscopic (particle interactions via MCT)
     - Mesoscopic (trap model)
   * - Glass transition
     - :math:`v_2 = 4` (calculated from S(k))
     - x = 1 (noise temperature, fitted)
   * - Yield stress origin
     - Cage elasticity
     - Trap depth distribution
   * - Aging/rejuvenation
     - Through :math:`\Phi(t,t')` memory
     - Through x(t) temperature
   * - Quantitative accuracy
     - Yes (with ISM + S(k))
     - Semi-quantitative
   * - Computational cost
     - High (Volterra integral)
     - Low (analytical)
   * - Best for
     - Hard-sphere colloids, dense suspensions
     - Soft glasses, foams, emulsions

.. code-block:: python

   from rheojax.models import SGRConventional

   # Compare flow curves
   gamma_dot = np.logspace(-4, 2, 50)

   # MCT glass
   mct = ITTMCTSchematic(epsilon=0.1)
   sigma_mct = mct.predict(gamma_dot, test_mode='flow_curve')

   # SGR glass
   sgr = SGRConventional()
   sgr.parameters.set_value('x', 0.8)  # x < 1 → glass
   sigma_sgr = sgr.predict(gamma_dot, test_mode='flow_curve')

   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(gamma_dot, sigma_mct, 'o-', label='ITT-MCT (ε=0.1)')
   ax.loglog(gamma_dot, sigma_sgr, 's-', label='SGR (x=0.8)')
   ax.set_xlabel('Shear rate γ̇ (1/s)')
   ax.set_ylabel('Shear stress σ (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

.. admonition:: Key Insight
   :class: tip

   Use ITT-MCT when you need first-principles understanding or quantitative predictions for hard-sphere systems. Use SGR for fast exploratory fits or soft glassy materials where MCT assumptions break down.

ITT-MCT vs Thixotropic Models (DMT, Fluidity)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Feature
     - ITT-MCT
     - DMT/Fluidity
   * - Structure tracking
     - Correlator :math:`\Phi(t,t')`
     - Scalar :math:`\lambda(t)` or f(t)
   * - Microstructure
     - Cage dynamics (physical)
     - Phenomenological
   * - Yield stress
     - Emergent from cages
     - Explicit :math:`\tau_y` parameter
   * - Predictive power
     - Can predict from volume fraction
     - Requires fitting to specific material
   * - Protocols
     - All 6 (incl. LAOS)
     - All 6 (incl. LAOS)

Limitations and Best Practices
-------------------------------

Known Limitations
^^^^^^^^^^^^^^^^^

1. **Computational Cost**: Volterra integral equations are expensive. First JIT compilation 30-90s, subsequent evaluations still slower than phenomenological models.

2. **MCT Glass Transition Overestimate**: Predicts :math:`\phi_{\text{MCT}} \approx 0.515` for hard spheres vs experimental :math:`\phi_{\text{exp}} \approx 0.58`. Use empirical corrections if quantitative :math:`\phi_g` needed.

3. **Hopping Neglected**: MCT assumes particles escape cages only through collective relaxation. Real systems have activated hopping at low T (handled by Extended MCT, not yet implemented).

4. **Thermal Fluctuations**: Linear response regime neglects thermal noise. Stochastic extensions exist but not in RheoJAX.

5. **Schematic Model Limitations**: :math:`F_{12}` schematic is semi-quantitative only. For quantitative predictions, use ISM with measured S(k).

6. **Monodisperse Assumption**: ISM assumes identical particles. Polydisperse systems require different S(k).

Best Practices
^^^^^^^^^^^^^^

.. code-block:: python

   # 1. Precompile once at workflow start
   model = ITTMCTSchematic()
   model.precompile()  # 30-90s delay here, not during fitting

   # 2. Use appropriate test protocols
   # SAOS: Best for determining ε (glass vs fluid)
   # Flow curve: Best for yield stress
   # Startup: Best for overshoot dynamics

   # 3. Check convergence for fitted ε
   result = model.fit(omega, G_star, test_mode='oscillation')
   if not result.success:
       print("Warning: fit did not converge")

   # 4. Bayesian inference for phase classification
   if abs(epsilon_fit) < 0.05:  # Near critical point
       print("ε ≈ 0: Use Bayesian inference for uncertainty")
       result_bayes = model.fit_bayesian(omega, G_star,
                                         test_mode='oscillation')

   # 5. Use ISM only when needed
   # Schematic for exploration, ISM for final quantitative comparison
   model_ism = ITTMCTIsotropic(phi=0.55)  # Slower but accurate

   # 6. Validate with multiple protocols
   # Fit SAOS → predict flow curve → compare with experiment
   model.fit(omega, G_star, test_mode='oscillation')
   sigma_pred = model.predict(gamma_dot, test_mode='flow_curve')
   # ... compare sigma_pred with measured flow curve ...

.. warning::
   Never extrapolate far outside the fitted range. MCT is a mean-field theory and may fail at very low frequencies (hopping) or very high frequencies (microscopic dynamics).

Related Jupyter Notebooks
--------------------------

RheoJAX provides comprehensive tutorials for ITT-MCT models:

**Schematic MCT** (:math:`F_{12}`):

- ``examples/itt_mct/01_schematic_flow_curve.ipynb`` — Yield stress prediction
- ``examples/itt_mct/02_schematic_startup_shear.ipynb`` — Stress overshoot
- ``examples/itt_mct/03_schematic_stress_relaxation.ipynb`` — Relaxation modulus
- ``examples/itt_mct/04_schematic_creep.ipynb`` — Delayed yielding
- ``examples/itt_mct/05_schematic_saos.ipynb`` — SAOS and glass transition
- ``examples/itt_mct/06_schematic_laos.ipynb`` — Nonlinear oscillation

**Isotropic MCT (ISM):**

- ``examples/itt_mct/07_isotropic_flow_curve.ipynb`` — Quantitative hard-sphere
- ``examples/itt_mct/08_isotropic_startup_shear.ipynb`` — Transient dynamics
- ``examples/itt_mct/09_isotropic_stress_relaxation.ipynb`` — Linear viscoelasticity
- ``examples/itt_mct/10_isotropic_creep.ipynb`` — Creep compliance
- ``examples/itt_mct/11_isotropic_saos.ipynb`` — Comparison with schematic
- ``examples/itt_mct/12_isotropic_laos.ipynb`` — Nonlinear response

Run notebooks with:

.. code-block:: bash

   cd examples/itt_mct
   jupyter notebook 01_schematic_flow_curve.ipynb

References
----------

**Foundational MCT:**

- Götze, W. (2009). *Complex Dynamics of Glass-Forming Liquids: A Mode-Coupling Theory*. Oxford University Press. ISBN: 978-0-19-923534-6
- Bengtzelius, U., Götze, W., & Sjölander, A. (1984). Dynamics of supercooled liquids and the glass transition. *J. Phys. C: Solid State Phys.* 17, 5915–5934. https://doi.org/10.1088/0022-3719/17/33/005

**ITT Framework:**

- Fuchs, M., & Cates, M. E. (2002). Theory of nonlinear rheology and yielding of dense colloidal suspensions. *Phys. Rev. Lett.* 89, 248304. https://doi.org/10.1103/PhysRevLett.89.248304
- Fuchs, M., & Cates, M. E. (2009). A mode coupling theory for Brownian particles in homogeneous steady shear flow. *J. Rheol.* 53, 957–1000. https://doi.org/10.1122/1.3119084

**Experimental Validation:**

- Brader, J. M., Voigtmann, Th., Fuchs, M., Larson, R. G., & Cates, M. E. (2009). Glass rheology: From mode-coupling theory to a dynamical yield criterion. *Proc. Natl. Acad. Sci. USA* 106, 15186–15191. https://doi.org/10.1073/pnas.0905330106
- Petekidis, G., Vlassopoulos, D., & Pusey, P. N. (2004). Yielding and flow of sheared colloidal glasses. *J. Phys.: Condens. Matter* 16, S3955–S3963. https://doi.org/10.1088/0953-8984/16/38/013

**Structure Factor Calculations:**

- Hansen, J. P., & McDonald, I. R. (2013). *Theory of Simple Liquids* (4th ed.). Academic Press. ISBN: 978-0-12-387032-2 (Chapter 5: Percus-Yevick equation)

See Also
--------

- :ref:`sgr_analysis` — Soft Glassy Rheology (complementary mesoscopic approach)
- :ref:`thixotropy_yielding` — Thixotropic models (DMT, Fluidity) for phenomenological fits
- :doc:`bayesian_inference` — Bayesian workflow and diagnostics
- :doc:`/models/itt_mct/index` — Full ITT-MCT API reference

**Related Models:**

- :doc:`/models/sgr/index` — SGR models
- :doc:`/models/dmt/index` — DMT thixotropic models
- :doc:`/models/fluidity/index` — Fluidity models

.. note::
   For questions or issues with ITT-MCT models, consult the `RheoJAX GitHub Issues <https://github.com/username/rheojax/issues>`_ or refer to the original publications above.
