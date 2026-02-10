Sequence of Physical Processes (SPP) Models
===========================================

This section documents the Sequence of Physical Processes (SPP) framework for
analyzing large amplitude oscillatory shear (LAOS) data.


Quick Reference
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Model
     - Purpose
     - Use Case
   * - :doc:`spp_decomposer`
     - LAOS analysis
     - Decompose stress into elastic, viscous, plastic contributions
   * - :doc:`spp_yield_stress`
     - Yield detection
     - Extract yield stress from LAOS data via SPP framework


Overview
--------

The **Sequence of Physical Processes (SPP)** framework, developed by Rogers and
coworkers, provides a physically-motivated approach to analyzing nonlinear
viscoelastic behavior under large amplitude oscillatory shear (LAOS). Unlike
Fourier-based methods that decompose stress into harmonics, SPP tracks how
material response evolves through sequences of distinct physical processes
within each oscillation cycle.

**Key advantages:**

- **Physical interpretation**: Direct connection to material physics (elasticity,
  viscosity, yielding)
- **Time-resolved**: Tracks instantaneous behavior within cycles
- **Yield stress extraction**: Robust method for obtaining yield stress from LAOS
- **Cycle-averaged properties**: Meaningful averages for elastic and viscous moduli

**Physical processes captured:**

- **Elastic storage**: Recoverable deformation (strain-driven)
- **Viscous dissipation**: Rate-dependent flow (strain-rate-driven)
- **Plastic yielding**: Irreversible deformation beyond yield
- **Cage dynamics**: Structure breakdown and reformation


SPP Framework
-------------

The SPP method analyzes stress response :math:`\sigma(t)` to sinusoidal strain
:math:`\gamma(t) = \gamma_0 \sin(\omega t)` by tracking instantaneous moduli:

**Instantaneous elastic modulus:**

.. math::

   G'_t = \frac{d\sigma}{d\gamma}\bigg|_t

**Instantaneous viscous modulus:**

.. math::

   G''_t = \frac{d\sigma}{d\dot{\gamma}}\bigg|_t \cdot \omega

**Cole-Cole representation:**

Plotting :math:`G'_t` vs :math:`G''_t` traces a trajectory revealing the
sequence of physical processes during oscillation.

**Perfect elastic solid**: Point at (G', 0)
**Perfect viscous liquid**: Point at (0, G'')
**Yielding material**: Trajectory shows transitions between regimes


When to Use SPP Analysis
------------------------

.. list-table::
   :widths: 35 25 40
   :header-rows: 1

   * - Scenario
     - SPP Recommended?
     - Alternative
   * - Linear viscoelastic (SAOS)
     - No (overkill)
     - Standard G', G''
   * - LAOS with mild nonlinearity
     - Yes
     - Fourier analysis (FT rheology)
   * - LAOS with yielding
     - ✓✓ Best choice
     - Bowditch-Lissajous
   * - Yield stress determination
     - ✓✓ Best choice
     - Stress sweep (less precise)
   * - Thixotropic materials
     - Yes
     - Three-interval test
   * - Physical mechanism identification
     - ✓✓ Best choice
     - Constitutive modeling


Key Concepts
------------

**Cycle-Averaged Moduli:**

SPP provides physically meaningful averages over the oscillation cycle:

.. math::

   \langle G' \rangle = \frac{1}{T}\oint G'_t \, dt

.. math::

   \langle G'' \rangle = \frac{1}{T}\oint G''_t \, dt

**Yield Stress from SPP:**

The yield stress is identified from the stress at which the Cole-Cole trajectory
shows a characteristic feature:

- **Type I yield**: Abrupt transition from elastic to plastic branch
- **Type II yield**: Gradual softening with continuous trajectory
- **Yield point**: Maximum in :math:`G'_t` or inflection in trajectory

**Intercycle vs Intracycle:**

- **Intercycle**: Comparison across different strain amplitudes :math:`\gamma_0`
- **Intracycle**: Evolution within a single oscillation period


Quick Start
-----------

**SPP Decomposition:**

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   import numpy as np

   # Create decomposer
   spp = SPPDecomposer()

   # Load LAOS data (time, strain, stress)
   t = np.linspace(0, 2*np.pi/omega, 1000)
   gamma = gamma_0 * np.sin(omega * t)
   stress = experimental_stress_data  # Your measured stress

   # Decompose into physical contributions
   result = spp.decompose(t, gamma, stress)

   # Access instantaneous moduli
   Gp_t = result.Gp_instantaneous  # G'(t)
   Gpp_t = result.Gpp_instantaneous  # G''(t)

   # Cycle-averaged values
   Gp_avg = result.Gp_average
   Gpp_avg = result.Gpp_average

**Yield Stress Extraction:**

.. code-block:: python

   from rheojax.models import SPPYieldStress

   # Create yield stress analyzer
   yield_analyzer = SPPYieldStress()

   # Process multiple strain amplitudes
   gamma_0_values = np.logspace(-1, 1, 20)  # 0.1 to 10 strain units
   results = []
   for gamma_0 in gamma_0_values:
       result = yield_analyzer.analyze(t, gamma, stress, gamma_0=gamma_0)
       results.append(result)

   # Extract yield stress
   sigma_y = yield_analyzer.extract_yield_stress(results)
   print(f"Yield stress: {sigma_y:.1f} Pa")

**Cole-Cole Visualization:**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot Cole-Cole trajectory
   plt.figure(figsize=(8, 6))
   plt.plot(Gp_t, Gpp_t, 'b-', lw=2)
   plt.xlabel("$G'_t$ (Pa)")
   plt.ylabel("$G''_t$ (Pa)")
   plt.title("SPP Cole-Cole Trajectory")
   plt.axhline(0, color='k', lw=0.5)
   plt.axvline(0, color='k', lw=0.5)
   plt.show()


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   spp_decomposer
   spp_yield_stress


See Also
--------

- :doc:`/transforms/spp` — SPP transform for LAOS data processing
- :doc:`/models/flow/herschel_bulkley` — Yield stress from flow curves
- :doc:`/models/epm/index` — Elasto-plastic models for yielding
- :doc:`/models/dmt/index` — Thixotropic models with LAOS support
- :doc:`/models/ikh/index` — IKH models with LAOS capabilities
- :doc:`/examples/laos/01-spp-analysis` — SPP analysis tutorial


References
----------

1. Rogers, S.A. (2012). "A sequence of physical processes determined and quantified
   in LAOS: An instantaneous local 2D/3D approach." *J. Rheol.*, 56, 1129–1151.
   https://doi.org/10.1122/1.4726083

2. Rogers, S.A. & Lettinga, M.P. (2012). "A sequence of physical processes determined
   and quantified in large-amplitude oscillatory shear (LAOS): Application to theoretical
   nonlinear models." *J. Rheol.*, 56, 1–25.
   https://doi.org/10.1122/1.3662962

3. Donley, G.J., de Bruyn, J.R., McKinley, G.H., & Rogers, S.A. (2019). "Time-resolved
   dynamics of the yielding transition in soft materials." *J. Non-Newtonian Fluid Mech.*,
   264, 117–134. https://doi.org/10.1016/j.jnnfm.2018.10.003

4. Donley, G.J., Singh, P.K., Shetty, A., & Rogers, S.A. (2020). "Elucidating the
   G''overshoot in soft materials with a yield transition via a time-resolved experimental
   strain decomposition." *PNAS*, 117, 21945–21952.
   https://doi.org/10.1073/pnas.2003869117

5. Lee, C.-W., Rogers, S.A., & McKinley, G.H. (2024). "SPP+ extensions for improved
   yield stress characterization." *J. Rheol.*, 68, 271–287.
   https://doi.org/10.1122/8.0000760

6. Hyun, K. et al. (2011). "A review of nonlinear oscillatory shear tests: Analysis
   and application of large amplitude oscillatory shear (LAOS)." *Prog. Polym. Sci.*,
   36, 1697–1753. https://doi.org/10.1016/j.progpolymsci.2011.02.002

7. Ewoldt, R.H., Hosoi, A.E., & McKinley, G.H. (2008). "New measures for characterizing
   nonlinear viscoelasticity in large amplitude oscillatory shear." *J. Rheol.*, 52,
   1427–1458. https://doi.org/10.1122/1.2970095
