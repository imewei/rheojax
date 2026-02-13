.. _transform-mastercurve:

Mastercurve (Time-Temperature Superposition)
============================================

Overview
--------

The :class:`rheojax.transforms.Mastercurve` transform aligns multi-temperature frequency
sweeps into a single reference spectrum using time-temperature superposition (TTS). It
estimates horizontal shift factors :math:`a_T` and optional vertical factors :math:`b_T`
relative to a reference temperature :math:`T_{\mathrm{ref}}`, producing a master curve that
spans decades beyond any individual experiment.

Equations
---------

Two constitutive laws are provided for :math:`a_T`:

* **WLF (Williams-Landel-Ferry)**

  .. math::

     \log_{10} a_T = -\frac{C_1 (T - T_{\mathrm{ref}})}{C_2 + (T - T_{\mathrm{ref}})}

  with dimensionless constants :math:`C_1, C_2` tuned near :math:`T_g`.

* **Arrhenius**

  .. math::

     \ln a_T = \frac{E_a}{R} \left( \frac{1}{T} - \frac{1}{T_{\mathrm{ref}}} \right)

  where :math:`E_a` is an apparent activation energy (J/mol) and :math:`R` is the gas constant.

Optional vertical shifting applies

.. math::

   b_T =
   \begin{cases}
     1, & \text{no vertical scaling} \\
     \exp\!\big(\beta (T - T_{\mathrm{ref}})\big), & \text{otherwise}
   \end{cases}

to compensate for density changes or entanglement loss at elevated temperatures.

Horizontal vs. Vertical Shifts
------------------------------

Frequency samples are warped via :math:`\tilde{\omega}_{ij} = \omega_j / a_T(T_i)` and
interpolated onto a log-spaced grid at :math:`T_{\mathrm{ref}}`. If `fit_vertical_shift=True`,
amplitudes are scaled as :math:`\tilde{G}_{ij} = G_{ij} / b_T(T_i)` prior to averaging.

Objective Function and Constraints
----------------------------------

Default fitting minimizes a weighted least-squares mismatch between each shifted sweep and
the evolving master curve:

.. math::

   \min_{\theta} \sum_{i,j} w_{ij} \Big\| \mathbf{y}_{ij}/b_T(T_i) - \hat{\mathbf{y}}\big(\omega_j/a_T(T_i)\big) \Big\|_2^2,

where :math:`\theta` collects :math:`C_1, C_2` (WLF) or :math:`E_a` (Arrhenius) plus the
vertical-shift parameter :math:`\beta`. Constraints enforce (1) :math:`a_T > 0`, (2)
monotonic :math:`\log a_T` vs. temperature, and (3) user-specified parameter bounds.

Algorithm
---------

1. **Preprocess** input sweeps into monotonic temperature order and normalize units.
2. **Initialize** :math:`a_T` via linear regression (WLF) or Arrhenius slope; seed :math:`b_T=1`.
3. **Warp** every sweep with the current :math:`a_T`, then (optionally) rescale by :math:`b_T`.
4. **Average** overlapping spectra on the target log-frequency grid using weights
   :math:`w_{ij}` derived from measurement uncertainty.
5. **Optimize** shift parameters with constrained L-BFGS until the objective converges.
6. **Emit diagnostics** (residual norms, monotonicity penalties, active constraints).

Parameters
----------

.. list-table:: Mastercurve configuration
   :header-rows: 1
   :widths: 22 25 20 33

   * - Parameter
     - Description
     - Default
     - Notes
   * - ``reference_temp``
     - Anchor temperature :math:`T_{\mathrm{ref}}` (K or  degC).
     - ``None`` (median)
     - Explicit selection improves reproducibility.
   * - ``shift_model``
     - ``"wlf"`` or ``"arrhenius"``.
     - ``"wlf"``
     - Controls :math:`a_T` formulation.
   * - ``fit_vertical_shift``
     - Enable :math:`b_T` estimation.
     - ``False``
     - Set ``True`` for density-changing systems.
   * - ``bounds``
     - Dict mapping parameter names to (min, max).
     - ``None``
     - e.g. ``{"C1": (5, 25), "C2": (20, 150)}``.
   * - ``auto_shift``
     - Compute shift factors automatically via overlap optimization (no WLF/Arrhenius model).
     - ``False``
     - Best for unknown systems; bypasses parametric shift models.
   * - ``loss``
     - Residual metric (``"l2"`` or ``"log"``).
     - ``"l2"``
     - Log residuals emphasize low-modulus regions.

Input / Output Specifications
-----------------------------

- **Inputs**

  - ``datasets``: iterable of :class:`rheojax.core.data.RheoData` objects, each containing
    frequency-domain moduli sampled at temperature :math:`T_i`.
  - ``temperatures``: sequence of floats (K or  degC) matching ``datasets``.
  - Optional weights ``w_{ij}`` or metadata describing uncertainty per sweep.

- **Outputs**

  - ``master_curve``: :class:`RheoData` with merged frequencies (:math:`\omega`, Hz) and
    moduli arrays (shape ``(M_\text{ref}, channels)``).
  - ``shift_factors``: dict with ``"a_T"`` and ``"b_T"`` arrays of length ``N``.
  - ``parameters``: fitted :math:`C_1, C_2` or :math:`E_a` plus optional :math:`\beta`.
  - ``diagnostics``: residual statistics, monotonicity penalty, optimization status.

Usage
-----

.. code-block:: python

   from rheojax.transforms import Mastercurve

   datasets = [data_25C, data_40C, data_55C]
   temps = [298.15, 313.15, 328.15]  # K

   mc = Mastercurve(
       reference_temp=313.15,
       shift_model="wlf",
       fit_vertical_shift=False,
       bounds={"C1": (8.0, 20.0), "C2": (30.0, 150.0)}
   )

   master = mc.create_mastercurve(datasets, temps)
   a_T = mc.get_shift_factors()["a_T"]
   C1, C2 = mc.get_wlf_parameters()

Auto Shift Factors (Model-Free)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the temperature dependence is unknown or does not follow WLF/Arrhenius, use
``auto_shift=True`` to let the optimizer find shift factors directly from overlap
quality without assuming a parametric model:

.. code-block:: python

   from rheojax.transforms import Mastercurve

   mc = Mastercurve(reference_temp=60, auto_shift=True)
   mastercurve, shift_factors = mc.transform(datasets)

   # Retrieve the model-free shift factors
   temps, log_aT = mc.get_auto_shift_factors()

This is especially useful for DMTA temperature sweeps where a single WLF or Arrhenius
model may not capture the full :math:`T_g` transition region.  See
``examples/transforms/06-mastercurve_auto_shift.ipynb`` for a complete tutorial.

Troubleshooting
---------------

- **Non-monotonic** :math:`a_T` - supply tighter bounds or reorder temperature data to
  increase overlap; penalize with ``enforce_monotonic=True``.
- **Poor overlap** - use ``smooth_overlap=True`` or restrict the fitting frequency range to
  regions with sufficient data density.
- **Overfitting vertical shift** - leave ``fit_vertical_shift=False`` unless density changes are
  documented; alternatively constrain :math:`\beta` via ``bounds``.
- **Limited frequency extension** - expand experimental window spacing or seed with better
  initial guesses via ``init_params``.

References
----------

- Williams, M. L., Landel, R. F., & Ferry, J. D. "The Temperature Dependence of Relaxation
  Mechanisms in Amorphous Polymers." *J. Am. Chem. Soc.* 77, 3701-3707 (1955).
- Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd ed. Wiley, 1980.
- Dealy, J. M. & Plazek, D. J. "Time-Temperature Superposition — A Users Guide." *Rheol.
  Bull.* 78(2), 16-21 (2009).

.. seealso::

   For DMTA temperature sweep data, see :doc:`/models/dmta/dmta_workflows`
   Workflow 2 (master curve from multi-T DMTA) and
   ``examples/dmta/07_dmta_tts_pipeline.ipynb``.

See also
--------

- :doc:`../models/fractional/fractional_maxwell_gel` and :doc:`../models/fractional/fractional_maxwell_liquid`
  — common targets for mastercurve datasets.
- :doc:`../models/fractional/fractional_burgers` — multi-mode fractional fits benefit from
  mastercurve preprocessing.
- :doc:`fft` — provides frequency-domain spectra used as input to Mastercurve.
- :doc:`mutation_number` — verify thermorheological simplicity before applying TTS.
- :doc:`../../examples/transforms/02-mastercurve-tts` — tutorial notebook creating
  mastercurves from TRIOS exports.
