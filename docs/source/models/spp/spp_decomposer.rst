.. _model-spp-decomposer:

SPP Decomposer (Sequence of Physical Processes)
================================================

Quick Reference
---------------

**Use when:** Large Amplitude Oscillatory Shear (LAOS) analysis of yield-stress fluids, colloidal glasses, soft glassy materials
**Parameters:** Extracted quantities (G_cage, sigma_sy, sigma_dy, I3/I1, S-factor, T-factor)
**Key equation:** :math:`\sigma(t) = G'_t(t)\gamma(t) + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_d(t)`
**Test modes:** LAOS (time-domain stress-strain waveforms)
**Material examples:** Colloidal glasses, microgel pastes, concentrated emulsions, foams, polymer gels

Overview
--------

The **Sequence of Physical Processes (SPP)** framework, introduced by Rogers et al. (2011) [1]_
and comprehensively formalized in Rogers (2017) [2]_, provides a time-domain approach to
Large Amplitude Oscillatory Shear (LAOS) analysis that extracts **instantaneous** elastic
modulus and dynamic viscosity without Fourier decomposition.

Unlike Fourier/Chebyshev methods that decompose stress into potentially infinite harmonics
representing a static point in n-dimensional space, SPP views the stress response as a
**dynamic trajectory** through a two-dimensional, easily interpretable space. The material
response is described as a sequence of discrete physical processes:

1. **Elastic extension** of microstructural cages
2. **Static yielding** when cages rupture
3. **Viscous flow** following the steady-state flow curve
4. **Cage reformation** when instantaneous shear rate approaches zero

This approach provides intuitively simpler interpretation while removing the problem of
understanding an infinite number of harmonic coefficients. The SPP framework:

- Makes **no assumptions about symmetries** in the response
- Views **each infinitesimal portion** of the response as the object to be analyzed
- Uses **differential parameters** that are immune to arbitrary reference state choices
- Distinguishes between strains in the **lab frame** and **material frame**

Physical Foundations
--------------------

The Four-Step Yielding Sequence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For yield-stress materials under LAOS, the material undergoes a characteristic four-step
sequence within each oscillatory cycle:

**Step 1: Elastic Extension (Cage Deformation)**

Beginning at zero stress, strain causes the microstructural cage to extend elastically.
The cage represents the effective constraint formed by nearest neighbors (colloidal
particles, polymer entanglements). During this phase:

- Stress increases linearly with strain
- The slope represents the **cage modulus** :math:`G_{\text{cage}}`
- Material behaves as an elastic solid

**Step 2: Static Yielding (Cage Rupture)**

When cages are strained beyond their yield strains, the system begins to flow:

- Associated with the stress overshoot ("bump") after linear buildup
- **Static yield stress** :math:`\sigma_{y,\text{static}}`: maximum stress before rupture
- **Yield strain** :math:`\gamma_y`: strain at which cages break

**Step 3: Viscous Flow (Post-Yield)**

After yielding, the material follows the steady-state flow curve:

- Power-law behavior: :math:`\sigma = K|\dot{\gamma}|^n`
- Same flow curve as steady-shear experiments
- Continues until instantaneous shear rate approaches zero

**Step 4: Cage Reformation (Recovery)**

When the instantaneous shear rate momentarily becomes zero at strain extrema:

- Cages instantaneously reform
- **Dynamic yield stress** :math:`\sigma_{y,\text{dynamic}}`: stress at reformation
- Process begins again in opposite direction

Cage Model Physics
~~~~~~~~~~~~~~~~~~

The SPP framework is built on the **cage model** from colloidal physics [3]_. In a quiescent
system of concentrated colloids:

- A representative colloid's motion is hindered by nearest neighbors forming an effective **cage**
- At short times: :math:`\beta`-relaxation (rattling within cage)
- At long times: :math:`\alpha`-relaxation (cage escape)
- External shearing can break cages and induce flow

The cage modulus :math:`G_{\text{cage}}` directly measures the elastic strength of this
microstructural caging. Key properties:

- **Constant across amplitudes**: Unlike :math:`G'`, :math:`G_{\text{cage}}` remains constant
  even at large amplitudes where :math:`G'` decreases by more than a decade
- **State-dependent**: Solid-state cages are stronger than soft-state cages
- **Recovers after breaking**: Once broken, cages reform with the soft-state modulus

Constitutive Equations
----------------------

Complete Stress Reconstruction (Rogers 2017)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fundamental contribution of Rogers (2017) [2]_ is recognizing that **three**, not two,
time-dependent nonlinear viscoelastic functions are required to fully describe any response.
The complete stress decomposition is:

.. math::

   \sigma(t) = G'_t(t)[\gamma(t) - \gamma_{eq}(t)] + \frac{G''_t(t)}{\omega}\dot{\gamma}(t) + \sigma_y(t)

where:
   - :math:`G'_t(t)` — instantaneous storage modulus (elastic contribution)
   - :math:`G''_t(t)` — instantaneous loss modulus (viscous contribution)
   - :math:`\gamma_{eq}(t)` — equilibrium strain position (can shift during deformation)
   - :math:`\sigma_y(t)` — yield stress term (zero-rate stress intercept)

The third term, representing the **displacement** of the osculating plane, accounts for:

- **Yield stresses**: Stress offsets not captured by moduli alone
- **Shifting strain equilibria**: The point of zero elastic extension can move
- **Non-Newtonian flow**: Deviation from simple viscoelastic behavior

Frenet-Serret Frame Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SPP formulation uses differential geometry on the 3D response trajectory in the
**native deformation space** :math:`[\gamma(t), \dot{\gamma}(t)/\omega, \sigma(t)]`.
Following the Frenet-Serret apparatus [2]_:

**Binormal Vector Definition**:

.. math::

   \mathbf{B} = \mathbf{T} \times \mathbf{N} = \frac{\dot{\mathbf{A}} \times \ddot{\mathbf{A}}}
   {||\dot{\mathbf{A}} \times \ddot{\mathbf{A}}||}

where :math:`\mathbf{T}` is the tangent vector and :math:`\mathbf{N}` is the principal normal.

**Instantaneous Moduli Definitions** (Equations 18-19 from Rogers 2017):

.. math::

   G'_t(t) = -\frac{B_\gamma(t)}{B_\sigma(t)}, \qquad
   G''_t(t) = -\frac{B_{\dot{\gamma}/\omega}(t)}{B_\sigma(t)}

where :math:`B_\gamma`, :math:`B_{\dot{\gamma}/\omega}`, and :math:`B_\sigma` are the
components of the binormal vector in the strain, strain-rate, and stress directions.

**Key Property**: Because the binormal vector is a differential parameter, :math:`G'_t(t)`
and :math:`G''_t(t)` depend only on **changes** in strain, not the total strain. They are
therefore immune to the arbitrary choice of reference state that affects static schemes.
This approach has been validated against theoretical nonlinear models [4]_.

The Displacement Term
~~~~~~~~~~~~~~~~~~~~~

The displacement term (Equation 11 from Rogers 2017) determines the position of the
osculating plane:

.. math::

   \text{displacement} = \frac{B_\gamma}{B_\sigma}\gamma(t) + \frac{B_{\dot{\gamma}/\omega}}{B_\sigma}
   \frac{\dot{\gamma}(t)}{\omega} + \sigma(t)

This is physically interpreted as:

.. math::

   \sigma_y(t) - G'_t(t)\gamma_{eq}(t)

**For linear viscoelastic responses**: The displacement is identically zero.

**For generalized Newtonian fluids**: The equilibrium strain equals the lab frame strain,
:math:`\gamma_{eq}(t) = \gamma(t)`, ensuring zero elastic stress at all times.

**For yield-stress materials**: The equilibrium position shifts during yielding, with
:math:`\gamma_{eq}` moving to :math:`\pm(\gamma_0 - \gamma_y)` after cage rupture.

Cage Modulus Definition
~~~~~~~~~~~~~~~~~~~~~~~

The cage modulus is defined as the instantaneous slope at zero stress:

.. math::

   G_{\text{cage}} \equiv \left. \frac{d\sigma}{d\gamma} \right|_{\sigma=0}

In the small phase angle, small-amplitude limit:

.. math::

   \lim_{\delta, \gamma_0 \to 0} G_{\text{cage}} = \lim_{\delta, \gamma_0 \to 0} G'

Yield Stress Definitions
~~~~~~~~~~~~~~~~~~~~~~~~

**Static Yield Stress** (stress at yielding point):

.. math::

   \sigma_{y,\text{static}} = \sigma(t)\big|_{\gamma=\pm\gamma_0}

**Dynamic Yield Stress** (stress at cage reformation):

.. math::

   \sigma_{y,\text{dynamic}} = \sigma(t)\big|_{\dot{\gamma}=\pm\dot{\gamma}_0}

Both yield stresses exhibit power-law rate dependence with identical exponents (~0.2):

.. math::

   \sigma_y \propto \gamma_0^{0.2} \propto \dot{\gamma}_0^{0.2}

In the large-amplitude limit: :math:`\sigma_{y,\text{static}} \approx 2.75 \times \sigma_{y,\text{dynamic}}`

Rates of Change (Stiffening/Softening, Thickening/Thinning)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rogers (2017) provides explicit definitions for the rates at which moduli change
(Equations 22-23), enabling quantitative identification of transitions:

.. math::

   \dot{G}'_t = \tau ||\dot{\mathbf{A}}|| \left( \frac{N_\gamma}{B_\sigma} - \frac{B_\gamma N_\sigma}{B_\sigma^2} \right)

.. math::

   \dot{G}''_t = \tau ||\dot{\mathbf{A}}|| \left( \frac{N_{\dot{\gamma}/\omega}}{B_\sigma} -
   \frac{B_{\dot{\gamma}/\omega} N_\sigma}{B_\sigma^2} \right)

where :math:`\tau` is the torsion of the response trajectory.

**Interpretation**:

- :math:`\dot{G}'_t > 0`: **Stiffening** (increasing elasticity)
- :math:`\dot{G}'_t < 0`: **Softening** (decreasing elasticity)
- :math:`\dot{G}''_t > 0`: **Thickening** (increasing viscosity)
- :math:`\dot{G}''_t < 0`: **Thinning** (decreasing viscosity)

**Key insight**: In the linear regime where responses are planar, the torsion
:math:`\tau = 0` everywhere, making both derivatives zero. The moduli are therefore
constant in time. Nonzero torsion indicates nonlinear behavior.

Time-Domain Cole-Cole Plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPP enables dynamic Cole-Cole analysis by plotting :math:`G''_t(t)` vs :math:`G'_t(t)`
parametrically through the oscillation cycle [2]_:

- **Linear regime**: Collapses to a single point :math:`(G', G'')`
- **Nonlinear regime**: Traces a trajectory showing material evolution

**Trajectory interpretation legend** (from Rogers 2017, Figure 2):

+---------------------------+-----------------------------+
| Position/Motion           | Interpretation              |
+===========================+=============================+
| :math:`G''_t = 0`         | Purely elastic              |
+---------------------------+-----------------------------+
| :math:`G'_t = 0`          | Purely viscous              |
+---------------------------+-----------------------------+
| :math:`G''_t > G'_t`      | Predominantly viscous       |
+---------------------------+-----------------------------+
| :math:`G'_t > G''_t`      | Predominantly elastic       |
+---------------------------+-----------------------------+
| Crossing :math:`G'_t=G''_t` toward viscous | Fluidization |
+---------------------------+-----------------------------+
| Crossing :math:`G'_t=G''_t` toward elastic | Reformation  |
+---------------------------+-----------------------------+

Time-Domain Phase Angle
~~~~~~~~~~~~~~~~~~~~~~~

SPP defines an instantaneous phase angle without Fourier decomposition:

.. math::

   \delta_t(t) = \arctan\left( \frac{G''_t(t)}{G'_t(t)} \right)

This phase evolves continuously through the cycle:
   - :math:`\delta_t \to 0°`: Purely elastic (solid-like)
   - :math:`\delta_t \to 90°`: Purely viscous (liquid-like)
   - Transition regions reveal yielding and recovery

Extracted Parameters
--------------------

.. list-table:: SPP Extracted Quantities
   :header-rows: 1
   :widths: 20 15 15 50

   * - Name
     - Symbol
     - Units
     - Description
   * - ``G_cage``
     - :math:`G_{\text{cage}}`
     - Pa
     - Apparent cage modulus; elastic strength of microstructure
   * - ``sigma_sy``
     - :math:`\sigma_{y,\text{static}}`
     - Pa
     - Static yield stress; stress at cage rupture
   * - ``sigma_dy``
     - :math:`\sigma_{y,\text{dynamic}}`
     - Pa
     - Dynamic yield stress; stress at cage reformation
   * - ``gamma_y``
     - :math:`\gamma_y`
     - —
     - Yield strain; strain at cage rupture
   * - ``I3_I1_ratio``
     - :math:`I_3/I_1`
     - —
     - Third harmonic ratio; nonlinearity measure
   * - ``S_factor``
     - :math:`S`
     - —
     - Strain stiffening factor
   * - ``T_factor``
     - :math:`T`
     - —
     - Shear thickening factor
   * - ``Gp_t_mean``
     - :math:`\langle G'_t \rangle`
     - Pa
     - Cycle-averaged instantaneous storage modulus
   * - ``Gpp_t_mean``
     - :math:`\langle G''_t \rangle`
     - Pa
     - Cycle-averaged instantaneous loss modulus

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G_cage (Cage Modulus)**:
   - **Physical meaning**: Elastic stiffness of microstructural cages at zero stress
   - **Key insight**: Remains constant across amplitude sweeps (unlike :math:`G'`)
   - **Typical ranges**:
      - Colloidal glasses: :math:`10^1 - 10^3` Pa
      - Emulsions: :math:`10^0 - 10^2` Pa
      - Polymer gels: :math:`10^2 - 10^4` Pa

**sigma_sy/sigma_dy (Yield Stresses)**:
   - **Physical meaning**: Stress thresholds for cage rupture and reformation
   - **Rate dependence**: Both scale as :math:`\dot{\gamma}_0^{0.2}`
   - **Ratio**: :math:`\sigma_{y,\text{static}} / \sigma_{y,\text{dynamic}} \approx 2.75`

**I3/I1 (Third Harmonic Ratio)**:
   - **Physical meaning**: Degree of nonlinearity in stress response
   - **Linear regime**: :math:`I_3/I_1 \to 0`
   - **Nonlinear regime**: :math:`I_3/I_1 \sim 0.1 - 0.3`

Validity and Assumptions
------------------------

**Assumptions:**

1. **Time-domain analysis**: Raw stress-strain waveforms required (not harmonic coefficients)
2. **Periodic steady state**: Material has reached oscillatory steady state
3. **Weak thixotropy**: Structure equilibrates within each cycle
4. **Homogeneous deformation**: No wall slip or shear banding

**Valid test modes:**

- LAOS (Large Amplitude Oscillatory Shear)
- Strain-controlled oscillation with time-resolved waveform capture

**Limitations:**

**Strong thixotropy**:
   For materials where structure changes dramatically over cycles, instantaneous properties
   may not be well-defined. Rule of thumb: SPP works when structural relaxation time
   :math:`\tau_s > 2\pi/\omega`.

**Noise sensitivity near zero stress**:
   Ratios become sensitive when :math:`\sigma \approx 0`. Apply smoothing or exclude
   these regions from analysis.

**Cycle selection effects**:
   Early cycles may contain transients; late cycles may show fatigue. Always verify
   consistency across multiple cycles.

SPP vs Fourier/Chebyshev
------------------------

.. list-table:: Method Comparison
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - SPP (Time-Domain)
     - Fourier/Chebyshev
   * - Domain
     - Time (instantaneous properties)
     - Frequency (harmonic coefficients)
   * - Basis functions
     - Physical processes (discrete)
     - Orthonormal harmonics (infinite)
   * - Yield stress
     - Direct extraction
     - Indirect from :math:`G_M, G_L`
   * - Higher harmonics
     - Explained by power-law flow
     - Individual physical meanings unclear
   * - Best for
     - Yield-stress fluids, physical mechanisms
     - Material fingerprinting, literature comparison

SPP is **qualitatively consistent** with Fourier-Chebyshev analysis [5]_. Higher harmonics
observed in Fourier spectra are explained by the power-law flow response post-yielding.
For a comprehensive review of LAOS methods including both SPP and Fourier approaches,
see Hyun et al. (2011) [6]_.

Usage
-----

Basic SPP Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.transforms import SPPDecomposer
   from rheojax.io.readers import auto_read

   # Load LAOS waveform data
   data = auto_read("laos_waveforms.csv")

   # Create SPP decomposer with Rogers-parity defaults
   decomposer = SPPDecomposer(
       omega=1.0,              # Angular frequency (rad/s)
       gamma_0=1.0,            # Strain amplitude
       n_harmonics=39,         # Rogers default (odd)
       step_size=8,            # 8-point stencil
       use_numerical_method=True,
   )

   # Apply decomposition
   result = decomposer.transform(data)
   spp_results = decomposer.get_results()

   # Access extracted quantities
   G_cage = spp_results['G_cage']
   sigma_sy = spp_results['sigma_sy']
   sigma_dy = spp_results['sigma_dy']
   I3_I1 = spp_results['I3_I1_ratio']

   print(f"Cage modulus: {G_cage:.1f} Pa")
   print(f"Static yield stress: {sigma_sy:.1f} Pa")
   print(f"Dynamic yield stress: {sigma_dy:.1f} Pa")
   print(f"Yield stress ratio: {sigma_sy/sigma_dy:.2f}")

Cycle Selection
~~~~~~~~~~~~~~~

.. code-block:: python

   # Skip startup transients (cycles 0-1), analyze cycles 2-5
   decomposer = SPPDecomposer(
       omega=1.0,
       gamma_0=1.0,
       start_cycle=2,
       end_cycle=5,
   )

   result = decomposer.transform(data)
   print(f"Cycles analyzed: {decomposer.results_['cycles_analyzed']}")

Amplitude Sweep Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline.spp import SPPAmplitudeSweepPipeline
   import numpy as np

   # Define amplitude sweep
   gamma_0_values = np.logspace(-2, 1, 20)

   # Run batch SPP analysis
   pipeline = SPPAmplitudeSweepPipeline(omega=1.0)
   results = pipeline.run_sweep(
       data_files=amplitude_files,
       gamma_0_values=gamma_0_values,
   )

   # Extract amplitude-dependent quantities
   G_cage_vs_gamma = results['G_cage']
   sigma_sy_vs_gamma = results['sigma_sy']

Bayesian Yield Stress Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import SPPYieldStress
   from rheojax.core import RheoData

   # Prepare amplitude-dependent yield stress data
   rheo_data = RheoData(
       x=gamma_0_values,
       y=sigma_sy_values,
       test_mode='spp_yield',
   )

   # Fit power-law model with Bayesian inference
   model = SPPYieldStress()
   model.fit(rheo_data)

   result = model.fit_bayesian(rheo_data, num_warmup=1000, num_samples=2000)

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"K = {intervals['K']['mean']:.2f} Pa")
   print(f"n = {intervals['n']['mean']:.3f}")

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # 1. Lissajous with yield points
   ax = axes[0, 0]
   ax.plot(strain, stress, 'b-', lw=1.5)
   ax.axhline(sigma_sy, color='r', ls='--', label=r'$\sigma_{y,static}$')
   ax.axhline(sigma_dy, color='g', ls=':', label=r'$\sigma_{y,dynamic}$')
   ax.set_xlabel(r'Strain $\gamma$')
   ax.set_ylabel(r'Stress $\sigma$ (Pa)')
   ax.legend()

   # 2. Instantaneous moduli
   ax = axes[0, 1]
   ax.plot(phase, Gp_t, 'b-', label=r"$G'_t$")
   ax.plot(phase, Gpp_t, 'r--', label=r"$G''_t$")
   ax.set_xlabel('Phase in cycle')
   ax.set_ylabel('Modulus (Pa)')
   ax.legend()

   # 3. Cole-Cole spiral
   ax = axes[1, 0]
   ax.plot(Gp_t, Gpp_t, 'b-')
   ax.set_xlabel(r"$G'_t$ (Pa)")
   ax.set_ylabel(r"$G''_t$ (Pa)")
   ax.set_aspect('equal')

   # 4. Amplitude dependence
   ax = axes[1, 1]
   ax.loglog(gamma_0, sigma_sy, 'ro-', label='Static')
   ax.loglog(gamma_0, sigma_dy, 'bs-', label='Dynamic')
   ax.set_xlabel(r'$\gamma_0$')
   ax.set_ylabel('Yield stress (Pa)')
   ax.legend()

   plt.tight_layout()

Fitting Guidance
----------------

Rogers-Parity Defaults
~~~~~~~~~~~~~~~~~~~~~~

RheoJAX uses MATLAB/Rogers-aligned defaults for SPP analysis:

.. list-table:: Default Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Notes
   * - ``n_harmonics``
     - 39
     - Odd harmonics; matches SPPplus v2.1
   * - ``step_size``
     - 8
     - 8-point 4th-order finite difference stencil
   * - ``num_mode``
     - 2
     - Periodic/looped differentiation
   * - ``yield_tolerance``
     - 0.02
     - Tolerance for yield point detection

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Common Issues
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Cause
     - Solution
   * - Noisy :math:`G'_t, G''_t`
     - Raw data noise propagates
     - Apply Savitzky-Golay smoothing before SPP
   * - Missing yield points
     - Tolerance too strict
     - Increase ``yield_tolerance`` to 0.05
   * - Inconsistent across cycles
     - Startup transients or fatigue
     - Adjust ``start_cycle``, ``end_cycle``
   * - :math:`G_{\text{cage}}` varies
     - Strong thixotropy
     - Consider time-resolved protocols instead

See Also
--------

- :doc:`spp_yield_stress` — Power-law model for SPP-extracted yield stresses
- :doc:`../sgr/sgr_conventional` — Soft Glassy Rheology model
- :doc:`../../transforms/owchirp` — Fourier-based LAOS analysis
- :doc:`../../user_guide/03_advanced_topics/spp_analysis` — Comprehensive SPP user guide

API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.SPPDecomposer`

References
----------

.. [1] Rogers, S. A., Erwin, B. M., Vlassopoulos, D., & Cloitre, M. "A sequence of
   physical processes determined and quantified in LAOS: Application to a yield stress
   fluid." *J. Rheol.* **55**, 435-458 (2011). https://doi.org/10.1122/1.3544591

.. [2] Rogers, S. A. "In search of physical meaning: defining transient parameters for
   nonlinear viscoelasticity." *Rheol. Acta* **56**, 501-525 (2017).
   https://doi.org/10.1007/s00397-017-1008-1

.. [3] Rogers, S. A. "A sequence of physical processes determined and quantified in
   LAOS: An instantaneous local 2D/3D approach." *J. Rheol.* **56**, 1129-1151 (2012).
   https://doi.org/10.1122/1.4726083

.. [4] Rogers, S. A. & Lettinga, M. P. "A sequence of physical processes determined
   and quantified in large-amplitude oscillatory shear (LAOS): Application to theoretical
   nonlinear models." *J. Rheol.* **56**, 1-25 (2012). https://doi.org/10.1122/1.3662962

.. [5] Ewoldt, R., Hosoi, A., & McKinley, G. "New measures for characterizing nonlinear
   viscoelasticity in large amplitude oscillatory shear." *J. Rheol.* **52**, 1427-1458 (2008).

.. [6] Hyun, K., et al. "A review of nonlinear oscillatory shear tests: Analysis and
   application of large amplitude oscillatory shear (LAOS)." *Prog. Polym. Sci.* **36**,
   1697-1753 (2011).
