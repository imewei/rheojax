.. _transform-cox-merz:

CoxMerz
=======

Overview
--------

The :class:`rheojax.transforms.CoxMerz` transform validates the **Cox-Merz rule**—an
empirical relation stating that the magnitude of the complex viscosity equals the
steady-shear viscosity at the same rate:

.. math::

   |\eta^*(\omega)| = \eta(\dot{\gamma}) \quad \text{at} \quad \omega = \dot{\gamma}

The transform takes two :class:`RheoData` inputs (oscillation + flow curve),
interpolates them onto a common rate grid, and computes deviation metrics.

**Key Capabilities:**

- **Rule validation:** Quantitative pass/fail assessment with configurable tolerance
- **Deviation mapping:** Point-by-point relative deviation across the overlap region
- **Automatic interpolation:** Log-log interpolation onto a common rate grid
- **Flexible input:** Accepts :math:`G^*(\omega)` (complex or ``[G', G'']``) and
  :math:`\sigma(\dot{\gamma})` or :math:`\eta(\dot{\gamma})`


Mathematical Theory
-------------------

The Cox-Merz Rule
~~~~~~~~~~~~~~~~~~

The **Cox-Merz rule** (1958) is an empirical observation for flexible polymer melts and
solutions:

.. math::

   |\eta^*(\omega)| \equiv \frac{|G^*(\omega)|}{\omega}
   = \frac{\sqrt{G'(\omega)^2 + G''(\omega)^2}}{\omega}
   \stackrel{?}{=} \eta(\dot{\gamma})

where:

- :math:`|\eta^*(\omega)|` is the **complex viscosity magnitude** from oscillatory data
- :math:`\eta(\dot{\gamma}) = \sigma(\dot{\gamma}) / \dot{\gamma}` is the **steady-shear viscosity**
- The equality is tested at :math:`\omega = \dot{\gamma}` (angular frequency = shear rate)

**When the rule holds** (deviation < 10%):

- **Linear flexible polymers** in the melt and concentrated solution states
- **Unentangled polymer melts** (limited shear thinning)
- Systems where **chain orientation** is the dominant nonlinear mechanism

**When the rule fails:**

- **Yield stress fluids** (gels, pastes, emulsions): Strong deviation due to yielding
- **Associating polymers**: Shear-induced network breakdown
- **Rigid rod polymers**: Different nonlinear mechanisms (tumbling vs flow alignment)
- **Thixotropic systems**: Structural changes under steady shear
- **Branched polymers**: Enhanced strain hardening

Cox-Merz failure is itself informative—it indicates that nonlinear mechanisms beyond
simple chain orientation dominate the material's response.

Deviation Metric
~~~~~~~~~~~~~~~~~

The transform computes the **relative deviation** at each common rate point:

.. math::

   \Delta(r) = \frac{|\eta^*(r) - \eta(r)|}{\eta^*(r)}

where :math:`r = \omega = \dot{\gamma}` is the rate on the common grid.

**Summary statistics:**

- **Mean deviation** :math:`\bar{\Delta}`: Average across all grid points
- **Maximum deviation** :math:`\Delta_{\max}`: Worst-case point
- **Pass/fail**: :math:`\bar{\Delta} \le` ``tolerance``

Interpolation Strategy
~~~~~~~~~~~~~~~~~~~~~~~

Both datasets are interpolated in **log-log space** onto a common rate grid spanning the
overlap region :math:`[\max(\omega_{\min}, \dot{\gamma}_{\min}), \min(\omega_{\max}, \dot{\gamma}_{\max})]`:

.. math::

   \ln \eta_{\text{interp}}(\ln r) = \text{linear interpolation of } (\ln x_i, \ln \eta_i)

This preserves the power-law structure typical of viscosity data.


Parameters
----------

.. list-table:: CoxMerz Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``tolerance``
     - float
     - ``0.10``
     - Maximum mean relative deviation for the rule to "pass" (0.10 = 10%).
   * - ``n_points``
     - int
     - ``50``
     - Number of interpolation points on the common rate grid.


Input / Output Specifications
-----------------------------

**Input:** A list of exactly two :class:`RheoData` objects:

1. **Oscillation data**: ``x`` = :math:`\omega` (rad/s), ``y`` = :math:`G^*` (complex) or
   ``(N, 2)`` array :math:`[G', G'']`
2. **Flow curve data**: ``x`` = :math:`\dot{\gamma}` (s\ :sup:`-1`), ``y`` = :math:`\sigma` (Pa)
   or :math:`\eta` (Pa·s). Set ``metadata["quantity"] = "viscosity"`` or
   ``metadata["is_viscosity"] = True`` if ``y`` is already viscosity.

**Output:** :class:`RheoData` with ``x`` = common rate grid, ``y`` = deviation array.
Metadata includes ``mean_deviation``, ``max_deviation``, and ``passes`` (bool).


Usage
-----

Basic Validation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import CoxMerz
   from rheojax.core.data import RheoData
   import numpy as np

   # Oscillatory data: frequency sweep
   omega = np.logspace(-2, 2, 50)
   G_prime = 1e4 * omega**2 / (1 + omega**2)
   G_double_prime = 1e4 * omega / (1 + omega**2)
   G_star = G_prime + 1j * G_double_prime

   osc_data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})

   # Flow curve data: steady shear
   gamma_dot = np.logspace(-2, 2, 40)
   eta_steady = 1e4 / (1 + gamma_dot)  # Cross model approximation
   sigma = eta_steady * gamma_dot

   flow_data = RheoData(x=gamma_dot, y=sigma, metadata={'test_mode': 'flow_curve'})

   # Validate Cox-Merz rule
   cm = CoxMerz(tolerance=0.10)
   result_data, info = cm.transform([osc_data, flow_data])

   result = info["cox_merz_result"]
   print(f"Mean deviation: {result.mean_deviation:.1%}")
   print(f"Max deviation:  {result.max_deviation:.1%}")
   print(f"Passes (≤10%):  {result.passes}")

With Viscosity Input
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If flow data is already viscosity
   flow_data = RheoData(
       x=gamma_dot, y=eta_steady,
       metadata={'test_mode': 'flow_curve', 'quantity': 'viscosity'}
   )

   cm = CoxMerz(tolerance=0.15)
   result_data, info = cm.transform([osc_data, flow_data])

Visualization
~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   cm = CoxMerz()
   _, info = cm.transform([osc_data, flow_data])
   result = info["cox_merz_result"]

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   # Left: Viscosity comparison
   ax1.loglog(result.common_rates, result.eta_complex, 'o-', label=r'$|\eta^*(\omega)|$')
   ax1.loglog(result.common_rates, result.eta_steady, 's-', label=r'$\eta(\dot{\gamma})$')
   ax1.set_xlabel(r'Rate $\omega$ or $\dot{\gamma}$ (s$^{-1}$)')
   ax1.set_ylabel(r'Viscosity (Pa$\cdot$s)')
   ax1.legend()
   ax1.set_title('Cox-Merz Comparison')

   # Right: Deviation
   ax2.semilogx(result.common_rates, result.deviation * 100, 'k-')
   ax2.axhline(10, color='r', ls='--', label='10% threshold')
   ax2.set_xlabel(r'Rate (s$^{-1}$)')
   ax2.set_ylabel('Relative Deviation (%)')
   ax2.legend()
   ax2.set_title(f'Mean = {result.mean_deviation:.1%}')

   plt.tight_layout()

Integration with Models
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import CoxMerz
   from rheojax.models import Maxwell

   # Fit Maxwell model to oscillatory data
   model = Maxwell()
   model.fit(omega, G_star, test_mode='oscillation')

   # Predict flow curve from fitted model
   model_eta = model.predict(gamma_dot, test_mode='flow_curve')

   # Validate Cox-Merz for the model
   model_flow = RheoData(
       x=gamma_dot, y=model_eta,
       metadata={'quantity': 'viscosity'}
   )
   cm = CoxMerz()
   _, info = cm.transform([osc_data, model_flow])
   print(f"Model Cox-Merz deviation: {info['cox_merz_result'].mean_deviation:.1%}")


Output Structure
----------------

.. list-table:: CoxMerzResult Attributes
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Shape/Type
     - Description
   * - ``common_rates``
     - (n_points,)
     - Common rate grid :math:`r = \omega = \dot{\gamma}` (s\ :sup:`-1`)
   * - ``eta_complex``
     - (n_points,)
     - Interpolated :math:`|\eta^*(\omega)|` (Pa·s)
   * - ``eta_steady``
     - (n_points,)
     - Interpolated :math:`\eta(\dot{\gamma})` (Pa·s)
   * - ``deviation``
     - (n_points,)
     - Point-by-point relative deviation
   * - ``mean_deviation``
     - float
     - Mean relative deviation :math:`\bar{\Delta}`
   * - ``max_deviation``
     - float
     - Maximum relative deviation :math:`\Delta_{\max}`
   * - ``passes``
     - bool
     - ``True`` if :math:`\bar{\Delta} \le` ``tolerance``


Validation and Quality Control
-------------------------------

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~~

**1. No overlapping rate range:**

- **Cause:** Oscillatory and flow curve data don't share any rate decade
- **Fix:** Extend measurement range for one or both experiments

**2. Spurious deviation at boundaries:**

- **Cause:** Extrapolation artifacts at grid edges
- **Fix:** Ensure both datasets extend beyond the common grid by ≥1 decade

**3. False failure for yield stress materials:**

- **Cause:** Cox-Merz rule is not expected to hold for yield stress fluids
- **Interpretation:** Deviation itself is the result—characterizes yielding behavior

**4. Flow data as stress vs viscosity:**

- **Cause:** Transform assumes stress by default and divides by :math:`\dot{\gamma}`
- **Fix:** Set ``metadata["quantity"] = "viscosity"`` if ``y`` is already viscosity


See Also
--------

- :doc:`fft` — Compute :math:`G^*(\omega)` from time-domain data for Cox-Merz input
- :doc:`../models/flow/cross` — Cross model for fitting flow curves
- :doc:`../models/classical/maxwell` — Maxwell model (satisfies Cox-Merz exactly)
- :doc:`../models/flow/herschel_bulkley` — Yield stress model (Cox-Merz typically fails)
- :doc:`srfs` — Strain-rate frequency superposition (related empirical relation)


API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.CoxMerz`


References
----------

1. Cox, W.P. & Merz, E.H. (1958). "Correlation of dynamic and steady flow
   viscosities." *J. Polym. Sci.*, 28, 619–622.
   DOI: `10.1002/pol.1958.1202811812 <https://doi.org/10.1002/pol.1958.1202811812>`_

2. Gleissle, W. & Hochstein, B. (2003). "Validity of the Cox–Merz rule for
   concentrated suspensions." *J. Rheol.*, 47, 897–910.
   DOI: `10.1122/1.1574020 <https://doi.org/10.1122/1.1574020>`_

3. Doraiswamy, D., Mujumdar, A.N., Tsao, I., Beris, A.N., Danforth, S.C., &
   Metzner, A.B. (1991). "The Cox-Merz rule extended: A rheological model for
   concentrated suspensions and other materials with a yield stress." *J. Rheol.*,
   35, 647–685. DOI: `10.1122/1.550184 <https://doi.org/10.1122/1.550184>`_

4. Al-Hadithi, T.S.R., Barnes, H.A., & Walters, K. (1992). "The relationship
   between the linear (oscillatory) and nonlinear (steady-state) flow properties
   of a series of polymer and colloidal systems." *Colloid Polym. Sci.*, 270,
   40–46. DOI: `10.1007/BF00656927 <https://doi.org/10.1007/BF00656927>`_

5. Dealy, J.M. & Larson, R.G. (2006). *Structure and Rheology of Molten Polymers:
   From Structure to Flow Behavior and Back Again*. Hanser. Chapter 10.
