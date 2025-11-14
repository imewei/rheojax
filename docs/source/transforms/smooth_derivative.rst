.. _transform-smooth-derivative:

SmoothDerivative
================

Overview
--------

:class:`rheojax.transforms.SmoothDerivative` offers three noise-aware numerical
differentiation schemes tailored for rheological data: Savitzky-Golay (SG) polynomial
filtering, cubic smoothing splines, and Tikhonov-regularized finite differences. All
methods accept nonuniform timestamps and return both derivative estimates and diagnostic
metadata.

Equations
---------

Savitzky-Golay
~~~~~~~~~~~~~~

Fits an order-:math:`p` polynomial to a sliding window of :math:`2m+1` samples and evaluates
the derivative analytically:

.. math::

   \frac{d \phi}{dt}\Big|_{t_i} = \sum_{k=-m}^{m} c_k \phi_{i+k},

where :math:`c_k` are precomputed convolution coefficients. SG preserves phase and sharp
features when sampling is uniform.

Smoothing Splines
~~~~~~~~~~~~~~~~~

Minimize

.. math::

   \sum_{i} (\phi_i - s(t_i))^2 + \lambda_s \int (s''(t))^2 dt

to obtain a cubic spline :math:`s(t)`; derivatives follow from the spline basis. Works with
nonuniform spacing; :math:`\lambda_s` governs smoothness.

Tikhonov Regularization
~~~~~~~~~~~~~~~~~~~~~~~

Solve

.. math::

   (D^\top D + \lambda_t I) \mathbf{d} = D^\top \mathbf{y},

where :math:`D` is the first-difference matrix. The solution :math:`\mathbf{d}` approximates
:math:`d\phi/dt` while penalizing high-frequency noise via :math:`\lambda_t`.

Parameters
----------

.. list-table:: Configuration guide
   :header-rows: 1
   :widths: 22 35 25 18

   * - Parameter
     - Description
     - Heuristic
     - Applies To
   * - ``method``
     - Backend identifier (``"savitzky_golay"``, ``"spline"``, ``"tikhonov"``).
     - Choose SG for uniform sampling, spline for nonuniform, Tikhonov for heavy noise.
     - All
   * - ``window`` (s)
     - Physical SG window width (converted to samples internally).
     - Cover at least one oscillation period; start with ``3 / f_s``.
     - SG
   * - ``poly_order``
     - SG polynomial degree.
     - 3 for smooth data, 5 if inflection points are critical.
     - SG
   * - ``lambda_s``
     - Spline roughness penalty.
     - Set to :math:`(\sigma_n/A)^2` where :math:`\sigma_n` is noise std, :math:`A` signal amplitude.
     - Spline
   * - ``lambda_t``
     - Tikhonov ridge weight.
     - Begin at 1e-2 for normalized signals; scale with noise variance.
     - Tikhonov
   * - ``spacing``
     - Override for sample spacing (s) when timestamps are omitted.
     - Provide when using implicit grids.
     - All

Stability vs. Responsiveness
----------------------------

- SG provides minimal phase lag but amplifies high-frequency noise if the window is too short.
- Splines handle irregular sampling; large ``lambda_s`` can over-smooth peaks.
- Tikhonov delivers robust derivatives on noisy data but can attenuate true rapid changes if
  ``lambda_t`` is excessive.

Input / Output Specifications
-----------------------------

- **Input**: ``time`` (1-D monotonically increasing array) and ``signal`` (array of equal
  length). Optional keyword arguments select the method and tuning parameters described above.
- **Output**: dict with
  - ``derivative`` array,
  - ``metadata`` capturing effective window, condition numbers, noise estimate, method id,
  - optional ``second_derivative`` when ``order=2`` is requested.

Usage
-----

.. code-block:: python

   from rheojax.transforms import SmoothDerivative

   sd = SmoothDerivative(method="spline", lambda_s=5e-4)
   dG_dt = sd.transform(time=ts, signal=G_prime)

   sg = SmoothDerivative(method="savitzky_golay", window=0.5, poly_order=3)
   dgamma_dt = sg.fit_transform(time=ts, signal=gamma)

Troubleshooting
---------------

- **Edge ringing (SG)** - pad data by reflecting the first/last window or enlarge ``window``.
- **Spline overshoot near jumps** - reduce ``lambda_s`` and limit knot spacing via
  ``max_interval``.
- **Flattened derivatives (Tikhonov)** - lower ``lambda_t`` or rescale the signal so the
  penalty operates on unit variance.
- **Non-monotonic timestamps** - sort or resample before invoking the transform; spline and
  Tikhonov methods assume positive spacing.

References
----------

- A. Savitzky and M.J.E. Golay, "Smoothing and differentiation of data by simplified
  least-squares procedures," *Anal. Chem.* 36, 1627–1639 (1964).
- C.H. Reinsch, "Smoothing by spline functions," *Numer. Math.* 10, 177–183 (1967).
- A.N. Tikhonov and V.Y. Arsenin, *Solutions of Ill-Posed Problems*, Winston (1977).
- D. Garcia, "Robust smoothing of gridded data in one and higher dimensions," *Comput.
  Stat. Data Anal.* 54, 1167–1178 (2010).
- P. Press et al., *Numerical Recipes in Python*, Cambridge University Press (2020).

See also
--------

- :doc:`../models/flow/bingham` — torque-to-stress pipelines often require differentiated
  velocity data before Bingham fits.
- :doc:`../models/flow/carreau` and :doc:`../models/flow/cross` — smoothing derivatives of
  log–log viscosity curves helps seed ``n``/``m``.
- :doc:`../transforms/fft` — combine with spectral conversion for slope analysis in
  :math:`G'(\omega)` and :math:`G''(\omega)`.
- :doc:`../transforms/mutation_number` — derivative estimates feed the mutation-number
  metric for gelation diagnostics.
- :doc:`../examples/transforms/01-torque-to-stress` — notebook demonstrating practical
  parameter settings for :class:`SmoothDerivative`.
