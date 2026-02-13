.. _transform-spp:

Sequence of Physical Processes (SPP) Transform
==============================================

Overview
--------

The SPP (Sequence of Physical Processes) transform analyzes large amplitude oscillatory
shear (LAOS) data to extract time-resolved viscoelastic properties within each
oscillation cycle. Unlike Fourier-based methods that decompose stress into harmonics,
SPP tracks instantaneous elastic and viscous contributions, revealing how materials
transition between physical processes during nonlinear deformation.

SPP is particularly valuable for:

- **Yield stress determination**: Extracting yield stress from LAOS data
- **Nonlinear characterization**: Understanding intracycle behavior
- **Material fingerprinting**: Identifying yielding mechanisms and flow transitions
- **Model validation**: Testing constitutive equations against SPP trajectories

The transform is based on the Rogers framework for analyzing LAOS through instantaneous
moduli plotted in Cole-Cole space.


Theory
------

Instantaneous Moduli
~~~~~~~~~~~~~~~~~~~~

For sinusoidal strain :math:`\gamma(t) = \gamma_0 \sin(\omega t)` with measured stress
:math:`\sigma(t)`, SPP defines instantaneous moduli:

**Storage modulus (elastic contribution):**

.. math::

   G'_t = \frac{d\sigma}{d\gamma}\bigg|_t = \frac{d\sigma/dt}{d\gamma/dt} = \frac{\dot{\sigma}}{\dot{\gamma}}

**Loss modulus (viscous contribution):**

.. math::

   G''_t = \frac{1}{\omega}\frac{d\sigma}{d\gamma}\bigg|_{\dot{\gamma}=\text{const}}

These instantaneous values reduce to the standard linear viscoelastic moduli in the
small-strain limit but reveal rich nonlinear behavior at large amplitudes.

Cole-Cole Trajectories
~~~~~~~~~~~~~~~~~~~~~~

Plotting :math:`G'_t` vs :math:`G''_t` traces a trajectory in Cole-Cole space that
reveals the sequence of physical processes:

.. code-block:: text

                      G''_t
                        │
                        │     Viscous
                        │   ●●●●●●●
                        │  ●       ●
                        │ ●         ●
               Elastic  │●           ●  Mixed
                 ●●●●●●●○─────────────────── G'_t
                        │●           ●
                        │ ●         ●
                        │  ●       ●
                        │   ●●●●●●●
                        │

**Interpretation:**

- **Point on** :math:`G'` **axis**: Pure elastic response (Hookean solid)
- **Point on** :math:`G''` **axis**: Pure viscous response (Newtonian liquid)
- **Trajectory loop**: Mixed viscoelastic with varying contributions
- **Crossing behavior**: Indicates yielding or structural transitions

Yield Stress Extraction
~~~~~~~~~~~~~~~~~~~~~~~

SPP provides robust yield stress determination by identifying characteristic features:

**Type I yielding (abrupt):**
   Sharp transition in Cole-Cole trajectory, identifiable as a kink or cusp
   where :math:`G'_t` drops suddenly.

**Type II yielding (gradual):**
   Smooth softening with continuous trajectory, yield identified from maximum
   in :math:`G'_t` or inflection point.

**Cage yield vs flow yield:**
   SPP can distinguish initial cage breaking (reversible) from full plastic
   flow (irreversible) by trajectory shape and timing within the cycle.


Parameters
----------

.. list-table:: SPP Transform Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``gamma_0``
     - float
     - Required
     - Strain amplitude for analysis
   * - ``omega``
     - float
     - Required
     - Angular frequency (rad/s)
   * - ``n_points``
     - int
     - 1000
     - Points per cycle for analysis
   * - ``smooth_derivatives``
     - bool
     - True
     - Apply Savitzky-Golay smoothing to derivatives
   * - ``window_length``
     - int
     - 21
     - Smoothing window size (must be odd)
   * - ``poly_order``
     - int
     - 3
     - Polynomial order for smoothing


Usage
-----

Basic SPP Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   import numpy as np

   # LAOS data: time, strain, stress
   t = np.linspace(0, 2*np.pi/omega, 1000)
   gamma = gamma_0 * np.sin(omega * t)
   stress = experimental_stress(t)  # Your measured data

   # Create SPP transform
   spp = SPPDecomposer(gamma_0=gamma_0, omega=omega)

   # Decompose into instantaneous moduli
   result = spp.transform(t, gamma, stress)

   # Access results
   Gp_t = result.Gp_instantaneous      # G'(t) array
   Gpp_t = result.Gpp_instantaneous    # G''(t) array
   phase = result.phase                 # Phase within cycle

Yield Stress Extraction
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   import numpy as np

   # Amplitude sweep data
   amplitudes = np.logspace(-1, 1, 20)
   stress_data = {amp: measure_stress(amp) for amp in amplitudes}

   spp = SPPDecomposer(omega=1.0)

   # Process each amplitude
   results = []
   for gamma_0, stress in stress_data.items():
       spp.gamma_0 = gamma_0
       t = np.linspace(0, 2*np.pi, 1000)
       gamma = gamma_0 * np.sin(t)
       result = spp.transform(t, gamma, stress)
       results.append({
           'gamma_0': gamma_0,
           'Gp_max': result.Gp_max,
           'Gp_avg': result.Gp_average,
           'yield_indicator': result.yield_indicator
       })

   # Extract yield stress from crossover
   yield_stress = spp.extract_yield_stress(results)
   print(f"Yield stress: {yield_stress:.2f} Pa")

Cole-Cole Visualization
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   spp = SPPDecomposer(gamma_0=0.5, omega=1.0)
   result = spp.transform(t, gamma, stress)

   # Create Cole-Cole plot
   fig, ax = plt.subplots(figsize=(8, 6))

   # Color by phase for trajectory evolution
   scatter = ax.scatter(
       result.Gp_instantaneous,
       result.Gpp_instantaneous,
       c=result.phase,
       cmap='viridis',
       s=10
   )

   ax.set_xlabel(r"$G'_t$ (Pa)")
   ax.set_ylabel(r"$G''_t$ (Pa)")
   ax.set_title(f"SPP Cole-Cole Trajectory ($\\gamma_0 = {gamma_0}$)")
   ax.axhline(0, color='k', lw=0.5)
   ax.axvline(0, color='k', lw=0.5)

   plt.colorbar(scatter, label="Phase (rad)")
   plt.tight_layout()
   plt.show()

Integration with Models
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SPPDecomposer
   from rheojax.models import HerschelBulkley

   # Fit Herschel-Bulkley to flow curve
   hb = HerschelBulkley()
   hb.fit(gamma_dot, stress, test_mode='flow_curve')
   sigma_y_flow = hb.parameters.get_value('sigma_y')

   # Extract yield stress from LAOS via SPP
   spp = SPPDecomposer(gamma_0=1.0, omega=1.0)
   sigma_y_spp = spp.extract_yield_stress(laos_results)

   # Compare methods
   print(f"Flow curve yield stress: {sigma_y_flow:.2f} Pa")
   print(f"SPP yield stress: {sigma_y_spp:.2f} Pa")

   # Validate constitutive model against SPP trajectory
   decomposer = SPPDecomposer()
   model_trajectory = decomposer.predict_trajectory(hb, gamma_0=1.0, omega=1.0)
   experimental_trajectory = result.cole_cole_trajectory

   # Compute trajectory mismatch
   mismatch = decomposer.trajectory_mismatch(
       experimental_trajectory, model_trajectory
   )
   print(f"Model-experiment mismatch: {mismatch:.3f}")


Output Structure
----------------

The SPP transform returns a result object with:

.. list-table:: SPP Transform Output
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Shape
     - Description
   * - ``Gp_instantaneous``
     - (n_points,)
     - Storage modulus vs time within cycle
   * - ``Gpp_instantaneous``
     - (n_points,)
     - Loss modulus vs time within cycle
   * - ``phase``
     - (n_points,)
     - Phase angle (0 to :math:`2\pi`)
   * - ``Gp_average``
     - scalar
     - Cycle-averaged storage modulus
   * - ``Gpp_average``
     - scalar
     - Cycle-averaged loss modulus
   * - ``Gp_max``
     - scalar
     - Maximum instantaneous :math:`G'`
   * - ``yield_indicator``
     - scalar
     - Yield metric (ratio of max to average)
   * - ``cole_cole_trajectory``
     - (n_points, 2)
     - [:math:`G'_t`, :math:`G''_t`] for plotting


See Also
--------

- :doc:`../models/spp/index` — SPP model family for LAOS analysis
- :doc:`../models/spp/spp_decomposer` — Full SPP decomposition model
- :doc:`../models/spp/spp_yield_stress` — Yield stress extraction via SPP
- :doc:`fft` — Fourier-based LAOS analysis (complementary approach)
- :doc:`../models/flow/herschel_bulkley` — Yield stress from flow curves
- :doc:`../models/epm/index` — Elasto-plastic models for yielding
- ``examples/advanced/10-spp-laos-tutorial.ipynb`` — SPP LAOS analysis tutorial notebook


API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.SPP`


References
----------

1. Rogers, S.A. (2012). "A sequence of physical processes determined and quantified
   in LAOS: An instantaneous local 2D/3D approach." *J. Rheol.*, 56, 1129–1151.
   https://doi.org/10.1122/1.4726083

2. Rogers, S.A. & Lettinga, M.P. (2012). "A sequence of physical processes determined
   and quantified in large-amplitude oscillatory shear (LAOS): Application to
   theoretical nonlinear models." *J. Rheol.*, 56, 1–25.
   https://doi.org/10.1122/1.3662962

3. Donley, G.J., de Bruyn, J.R., McKinley, G.H., & Rogers, S.A. (2019).
   "Time-resolved dynamics of the yielding transition in soft materials."
   *J. Non-Newtonian Fluid Mech.*, 264, 117–134.
   https://doi.org/10.1016/j.jnnfm.2018.10.003

4. Hyun, K. et al. (2011). "A review of nonlinear oscillatory shear tests:
   Analysis and application of large amplitude oscillatory shear (LAOS)."
   *Prog. Polym. Sci.*, 36, 1697–1753.
   https://doi.org/10.1016/j.progpolymsci.2011.02.002

5. Ewoldt, R.H., Hosoi, A.E., & McKinley, G.H. (2008). "New measures for
   characterizing nonlinear viscoelasticity in large amplitude oscillatory shear."
   *J. Rheol.*, 52, 1427–1458.
   https://doi.org/10.1122/1.2970095
