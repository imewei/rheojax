SPP Yield Stress Model
======================

.. module:: rheojax.models.spp_yield_stress
   :synopsis: SPP-based yield stress model for LAOS amplitude sweeps

Quick Reference
---------------

.. list-table::
   :widths: 25 75
   :stub-columns: 1

   * - **Model Class**
     - :class:`~rheojax.models.spp_yield_stress.SPPYieldStress`
   * - **Registry Name**
     - ``"spp_yield_stress"``
   * - **Test Modes**
     - ``oscillation`` (amplitude sweep), ``rotation`` (flow curve)
   * - **Parameters**
     - 8 (G_cage, sigma_sy_scale, sigma_sy_exp, sigma_dy_scale, sigma_dy_exp, eta_inf, n_power_law, noise)
   * - **Typical Materials**
     - Yield stress fluids, colloidal gels, emulsions, foams, soft glasses
   * - **Key Reference**
     - Rogers et al. (2012) J. Rheol. 56(1)

Overview
--------

The SPP Yield Stress model extracts physically meaningful yield parameters from
Large Amplitude Oscillatory Shear (LAOS) data using the Sequence of Physical
Processes (SPP) framework. Unlike traditional Fourier-based approaches, SPP
provides time-resolved instantaneous material functions that reveal the
intracycle sequence of physical processes during nonlinear deformation.

The model parameterizes the nonlinear response in terms of:

- **G_cage**: Apparent cage modulus (elastic stiffness of the microstructural cage)
- **Static yield stress** (σ_sy): Stress at strain reversal (maximum strain amplitude)
- **Dynamic yield stress** (σ_dy): Stress at rate reversal (zero strain rate)
- **Power-law scaling**: Amplitude dependence of yield stresses with exponent

This approach connects LAOS analysis to steady-shear flow behavior, enabling
comprehensive yield stress characterization across deformation protocols.

Physical Foundations
--------------------

Cage Model for Yield Stress Fluids
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP yield stress model is grounded in the colloidal cage picture, where
particles are confined by nearest-neighbor "cages":

1. **Linear Regime** (small γ_0): The cage deforms elastically with stiffness G_cage
2. **Yielding** (γ_0 → γ_yield): Cage constraints are overcome at the yield point
3. **Flow Regime** (large γ_0): Particles escape cages and flow viscously

The cage modulus G_cage represents the instantaneous elastic stiffness measured
at the point where stress passes through zero (σ = 0). This corresponds to the
slope of the stress-strain Lissajous curve at the origin.

Static vs. Dynamic Yield Stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP framework distinguishes two physically distinct yield stresses:

**Static Yield Stress (σ_sy)**
   - Measured at strain reversal (γ = ±γ_0, γ̇ = 0)
   - Represents the stress required to initiate flow from rest
   - Larger than dynamic yield due to microstructural recovery during strain reversal

**Dynamic Yield Stress (σ_dy)**
   - Measured at rate reversal (γ̇ = 0, γ ≠ 0)
   - Represents the stress during continuous flow
   - Connects to steady-shear yield stress extrapolation

The ratio σ_sy/σ_dy is typically around 2-3 for colloidal systems and reveals
information about cage reformation kinetics and thixotropy.

Power-Law Amplitude Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both yield stresses exhibit power-law scaling with strain amplitude:

.. math::

   \sigma_{sy}(\gamma_0) = \sigma_{sy,0} \cdot \gamma_0^{n_{sy}}

.. math::

   \sigma_{dy}(\gamma_0) = \sigma_{dy,0} \cdot \gamma_0^{n_{dy}}

where the exponents n_sy and n_dy typically fall in the range 0.2-1.0.
Rogers et al. (2011) found n ≈ 0.2 for concentrated colloidal suspensions,
connecting to the Herschel-Bulkley flow curve exponent.

Connection to SPP Framework (Rogers 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The yield stress terms in this model connect to the complete SPP stress
decomposition from Rogers (2017):

.. math::

   \sigma(t) = G'_t(t)[\gamma(t) - \gamma_{eq}(t)] + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_y(t)

The **displacement term** :math:`\sigma_y(t) - G'_t(t)\gamma_{eq}(t)` captures:

1. **Static yield stress**: Associated with the shift in equilibrium strain
   :math:`\gamma_{eq}` during cage rupture

2. **Dynamic yield stress**: The zero-rate stress intercept from the
   viscoplastic flow contribution

**Equilibrium Strain Interpretation**:

- In the linear regime, :math:`\gamma_{eq} = 0` (no shifting)
- During yielding, :math:`\gamma_{eq}` shifts to :math:`\pm(\gamma_0 - \gamma_y)`
- The material frame strain becomes :math:`\gamma_{mat}(t) = \gamma(t) - \gamma_{eq}(t)`
- This distinction between lab frame and material frame is essential for
  understanding the physical origin of the two yield stresses

Constitutive Equations
----------------------

Oscillation Mode (LAOS Amplitude Sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For amplitude sweep analysis with strain amplitude γ_0:

.. math::

   \sigma_{sy}(\gamma_0) = \text{sigma\_sy\_scale} \cdot |\gamma_0|^{\text{sigma\_sy\_exp}}

.. math::

   \sigma_{dy}(\gamma_0) = \text{sigma\_dy\_scale} \cdot |\gamma_0|^{\text{sigma\_dy\_exp}}

The cage modulus at small amplitude approximates:

.. math::

   G_{cage} \approx \frac{\sigma_{sy}(\gamma_0)}{\gamma_0} \quad (\gamma_0 \to 0)

Rotation Mode (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For steady shear flow at rate γ̇, the model uses a Herschel-Bulkley-like form:

.. math::

   \sigma(\dot{\gamma}) = \sigma_{dy} + \eta_\infty |\dot{\gamma}|^n

This connects the dynamic yield stress from LAOS to steady-shear flow curves,
enabling validation between oscillatory and continuous shear measurements.

Parameter Reference
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 12 15 55

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_cage``
     - Pa
     - [1e-3, 1e9]
     - Apparent cage modulus; elastic stiffness at σ = 0
   * - ``sigma_sy_scale``
     - Pa
     - [1e-6, 1e9]
     - Static yield stress scale factor
   * - ``sigma_sy_exp``
     - —
     - [0.0, 2.0]
     - Static yield stress amplitude exponent
   * - ``sigma_dy_scale``
     - Pa
     - [1e-6, 1e9]
     - Dynamic yield stress scale factor
   * - ``sigma_dy_exp``
     - —
     - [0.0, 2.0]
     - Dynamic yield stress amplitude exponent
   * - ``eta_inf``
     - Pa·s
     - [1e-9, 1e6]
     - Infinite-shear viscosity (for flow curve)
   * - ``n_power_law``
     - —
     - [0.01, 2.0]
     - Flow power-law index
   * - ``noise``
     - Pa
     - [1e-10, 1e6]
     - Observation noise scale (Bayesian inference)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G_cage** (Cage Modulus)
   - Physical meaning: Instantaneous elastic stiffness of the confining cage
   - Typical values: 10-10,000 Pa for soft materials; higher for stiffer gels
   - Measurement: Slope of σ(γ) at γ = 0 within LAOS cycle

**sigma_sy_scale** and **sigma_sy_exp**
   - Physical meaning: Power-law prefactor and exponent for static yield stress
   - Typical values: scale ~ 1-1000 Pa, exponent ~ 0.2-1.0
   - Rogers (2011) colloidal data: exponent ≈ 0.2

**sigma_dy_scale** and **sigma_dy_exp**
   - Physical meaning: Power-law prefactor and exponent for dynamic yield stress
   - Typical values: scale ~ 0.5-500 Pa (usually < sigma_sy_scale)
   - Often σ_sy/σ_dy ≈ 2-3 for thixotropic materials

**n_power_law**
   - Physical meaning: Shear-thinning exponent in Herschel-Bulkley flow
   - n < 1: Shear-thinning (most yield stress fluids)
   - n = 1: Bingham plastic
   - Typical values: 0.3-0.8 for colloidal gels

Validity and Assumptions
------------------------

Applicability
~~~~~~~~~~~~~

The SPP yield stress model is most appropriate for:

- **Yield stress fluids**: Materials with clear solid-to-liquid transition
- **Soft glassy materials**: Colloidal gels, emulsions, foams, pastes
- **LAOS amplitude sweeps**: Progressive nonlinearity from linear to flowing
- **Concentrated systems**: Volume fractions near or above jamming

Assumptions
~~~~~~~~~~~

1. **Single characteristic yield process**: The model assumes a dominant yielding
   mechanism described by power-law scaling

2. **Cage-based microstructure**: Physical interpretation requires particle-based
   or droplet-based microstructure with cage confinement

3. **Time-strain separability**: Assumes steady-state oscillatory response
   without significant transient evolution during measurement

4. **Sufficient harmonics**: SPP extraction requires adequate harmonic content
   (typically n_harmonics ≥ 15)

Limitations
~~~~~~~~~~~

- **Simple materials**: Newtonian fluids show no amplitude dependence
- **Polymer solutions**: May require different physical interpretation
- **Extreme amplitude**: Very large γ_0 may show non-power-law behavior
- **Transient effects**: Not suitable for strongly time-dependent (aging) materials
- **Low frequency**: β-relaxation must be slow compared to measurement

Usage Examples
--------------

Basic NLSQ Fitting (Amplitude Sweep)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import SPPYieldStress

   # Amplitude sweep data: yield stresses vs. strain amplitude
   gamma_0 = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
   sigma_sy = np.array([5.0, 8.5, 18.0, 32.0, 55.0, 110.0, 195.0, 340.0])

   # Initialize and fit
   model = SPPYieldStress()
   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   # View fitted parameters
   print(model)
   # SPPYieldStress(G_cage=5.00e+02, σ_sy=1.00e+02, σ_dy=5.00e+01, n=0.50)

   # Predict at new amplitudes
   gamma_0_pred = np.logspace(-2, 1, 50)
   sigma_pred = model.predict(gamma_0_pred)

Dynamic Yield Stress Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Dynamic yield stress (at rate reversal) typically lower than static
   sigma_dy = np.array([2.0, 3.5, 7.5, 13.0, 22.0, 45.0, 80.0, 140.0])

   model = SPPYieldStress()
   model.fit(gamma_0, sigma_dy, test_mode='oscillation', yield_type='dynamic')

   # Get both yield stresses across amplitude range
   sweep_results = model.predict_amplitude_sweep(
       gamma_0_pred,
       yield_type='both'
   )
   print(f"Static yield stress: {sweep_results['sigma_sy']}")
   print(f"Dynamic yield stress: {sweep_results['sigma_dy']}")

Flow Curve Fitting (Rotation Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Steady shear data
   gamma_dot = np.logspace(-3, 2, 20)
   sigma = 25.0 + 3.5 * gamma_dot**0.45  # Herschel-Bulkley-like

   model = SPPYieldStress()
   model.fit(gamma_dot, sigma, test_mode='rotation')

   # Predict flow curve
   flow_results = model.predict_flow_curve(gamma_dot)
   print(f"Yield stress: {model.parameters.get_value('sigma_dy_scale'):.2f} Pa")
   print(f"Power-law index: {model.parameters.get_value('n_power_law'):.2f}")

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SPPYieldStress

   # Initialize and NLSQ fit for warm-start
   model = SPPYieldStress()
   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   # Bayesian inference with warm-start
   result = model.fit_bayesian(
       gamma_0,
       sigma_sy,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000
   )

   # Posterior summary
   print(result.summary)

   # Credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )
   for param, (low, high) in intervals.items():
       print(f"{param}: [{low:.3f}, {high:.3f}]")

Integration with SPPDecomposer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   from rheojax.models import SPPYieldStress
   import numpy as np

   # Process LAOS waveforms at multiple amplitudes
   decomposer = SPPDecomposer(n_harmonics=39, step_size=8)

   # Collect yield stresses from each amplitude
   amplitudes = []
   static_yields = []
   dynamic_yields = []

   for gamma_0, data in amplitude_sweep_data.items():
       result = decomposer.transform(data)
       amplitudes.append(gamma_0)
       static_yields.append(np.abs(result['stress_at_strain_max']))
       dynamic_yields.append(np.abs(result['stress_at_rate_reversal']))

   # Fit yield stress model
   model = SPPYieldStress()

   # Static yield stress
   model.fit(np.array(amplitudes), np.array(static_yields),
             test_mode='oscillation', yield_type='static')

   print(f"Static yield exponent: {model.parameters.get_value('sigma_sy_exp'):.3f}")

Troubleshooting
---------------

Poor Power-Law Fit
~~~~~~~~~~~~~~~~~~

**Symptoms**: Large fitting residuals, unreasonable exponents

**Solutions**:

1. Check data quality at low amplitudes (may be noisy near linear regime)
2. Verify sufficient amplitude range (at least 1-2 decades)
3. Look for regime transitions (different slopes at different amplitudes)

.. code-block:: python

   # Visualize power-law fit
   import matplotlib.pyplot as plt

   model.fit(gamma_0, sigma_sy, test_mode='oscillation', yield_type='static')

   fig, ax = plt.subplots()
   ax.loglog(gamma_0, sigma_sy, 'o', label='Data')
   ax.loglog(gamma_0, model.predict(gamma_0), '-', label='Fit')
   ax.set_xlabel(r'$\gamma_0$')
   ax.set_ylabel(r'$\sigma_{sy}$ (Pa)')
   ax.legend()
   plt.show()

Cage Modulus Issues
~~~~~~~~~~~~~~~~~~~

**Symptoms**: G_cage unreasonably large or small

**Causes**:

- Insufficient low-amplitude data points
- Material not exhibiting clear cage behavior
- Noise dominating linear regime

**Solutions**:

1. Ensure adequate data in linear regime (γ_0 < 0.1)
2. Verify linear viscoelastic moduli are consistent
3. Consider if cage model is appropriate for the material

Bayesian Convergence
~~~~~~~~~~~~~~~~~~~~

**Symptoms**: R-hat > 1.1, low ESS, divergent transitions

**Solutions**:

1. **Always NLSQ warm-start**: Critical for stable sampling

   .. code-block:: python

      # NLSQ first, then Bayesian
      model.fit(gamma_0, sigma_sy, test_mode='oscillation')
      result = model.fit_bayesian(gamma_0, sigma_sy, ...)

2. **Increase samples**: ``num_warmup=2000, num_samples=4000``

3. **Check priors**: Ensure physically reasonable bounds

4. **Inspect trace plots**: Look for mixing issues

   .. code-block:: python

      import arviz as az
      az.plot_trace(result.arviz_data)

Prior Sensitivity
~~~~~~~~~~~~~~~~~

The model uses physically-motivated priors:

- **LogNormal** for scale parameters (G_cage, stress scales, viscosity)
- **Beta** for bounded exponents [0, 2]
- **HalfCauchy** for noise scale

If priors are too informative:

.. code-block:: python

   # Check prior impact with prior predictive sampling
   from rheojax.visualization import plot_bayesian_diagnostics

   # Compare posterior to prior
   plot_bayesian_diagnostics(result, diagnostics=['pair', 'forest'])

Related Models
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Use Case
   * - :doc:`spp_decomposer`
     - Extract instantaneous moduli from individual LAOS waveforms
   * - :doc:`../flow/herschel_bulkley`
     - Steady-shear yield stress (Herschel-Bulkley model)
   * - :doc:`../sgr/sgr_conventional`
     - Soft glassy rheology for aging yield stress fluids
   * - :doc:`../flow/bingham`
     - Simple yield stress + Newtonian flow

References
----------

Primary References
~~~~~~~~~~~~~~~~~~

1. **Rogers, S.A., Erwin, B.M., Vlassopoulos, D., & Cloitre, M.** (2011).
   "A sequence of physical processes determined and quantified in LAOS:
   Application to a yield stress fluid." *J. Rheol.* 55, 435-458.
   `doi:10.1122/1.3544591 <https://doi.org/10.1122/1.3544591>`_

2. **Rogers, S.A.** (2012). "A sequence of physical processes determined
   and quantified in large-amplitude oscillatory shear (LAOS): Application
   to theoretical nonlinear models." *J. Rheol.* 56(1), 1-25.

3. **Rogers, S.A.** (2017). "In search of physical meaning: Defining
   transient parameters for nonlinear viscoelasticity." *Rheol. Acta* 56, 501-525.

Background Theory
~~~~~~~~~~~~~~~~~

4. **Ewoldt, R.H., Hosoi, A.E., & McKinley, G.H.** (2008). "New measures
   for characterizing nonlinear viscoelasticity in large amplitude
   oscillatory shear." *J. Rheol.* 52, 1427-1458.

5. **Hyun, K., et al.** (2011). "A review of nonlinear oscillatory shear
   tests: Analysis and application of large amplitude oscillatory shear
   (LAOS)." *Prog. Polym. Sci.* 36, 1697-1753.

Yield Stress Fundamentals
~~~~~~~~~~~~~~~~~~~~~~~~~

6. **Bonn, D., Denn, M.M., Berthier, L., Divoux, T., & Manneville, S.** (2017).
   "Yield stress materials in soft condensed matter." *Rev. Mod. Phys.*
   89, 035005.

See Also
--------

- :doc:`/user_guide/03_advanced_topics/spp_analysis` - Complete SPP analysis user guide
- :doc:`/api/spp_models` - SPP API reference
- :doc:`spp_decomposer` - SPP Decomposer transform documentation
