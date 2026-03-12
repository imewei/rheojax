.. _transform-lve-envelope:

LVEEnvelope
===========

Overview
--------

The :class:`rheojax.transforms.LVEEnvelope` transform computes the **linear viscoelastic
(LVE) startup stress envelope** :math:`\sigma_{\text{LVE}}^+(t)` from a Prony series
representation of the relaxation modulus. This analytical prediction provides the
theoretical stress growth response assuming linear viscoelasticity.

**Key Capabilities:**

- **Startup stress prediction:** :math:`\sigma_{\text{LVE}}^+(t)` for any shear rate
- **Nonlinearity detection:** Compare with experimental startup data to identify strain
  hardening or softening
- **JIT-compiled:** Fully JAX-accelerated evaluation for fast sweeps over multiple rates
- **Flexible input:** Prony parameters via constructor or from :class:`RheoData` metadata

The LVE envelope is a cornerstone of nonlinear rheology characterization. In a startup
experiment, the material's stress growth follows the LVE envelope at early times but
deviates at a characteristic strain—the **Hencky strain at onset of nonlinearity**—revealing
the material's nonlinear response.


Mathematical Theory
-------------------

Startup Stress in Linear Viscoelasticity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a material subjected to constant shear rate :math:`\dot{\gamma}_0` starting at
:math:`t = 0`, the **Boltzmann superposition integral** gives the transient stress:

.. math::

   \sigma^+(t) = \dot{\gamma}_0 \int_0^t G(s) \, ds

Substituting the Prony series :math:`G(t) = G_e + \sum_i G_i \exp(-t/\tau_i)`:

.. math::

   \sigma_{\text{LVE}}^+(t) = \dot{\gamma}_0 \left[ G_e \, t + \sum_{i=1}^{N} G_i \tau_i \left(1 - \exp(-t/\tau_i)\right) \right]

**Limiting behaviors:**

- **Short time** (:math:`t \ll \tau_{\min}`): :math:`\sigma \approx \dot{\gamma}_0 \left(G_e + \sum G_i\right) t`
  (elastic response, slope = glassy modulus × rate)
- **Long time** (:math:`t \gg \tau_{\max}`): :math:`\sigma \to \dot{\gamma}_0 \left(G_e \, t + \sum G_i \tau_i\right)`
  (steady state for viscoelastic solids, or constant :math:`\eta_0 \dot{\gamma}_0` for liquids)

**Steady-state viscosity** (for liquids, :math:`G_e = 0`):

.. math::

   \eta_0 = \lim_{t \to \infty} \frac{\sigma_{\text{LVE}}^+(t)}{\dot{\gamma}_0}
   = \sum_{i=1}^{N} G_i \tau_i

Nonlinearity Assessment
~~~~~~~~~~~~~~~~~~~~~~~~

Comparing :math:`\sigma_{\text{exp}}^+(t)` with :math:`\sigma_{\text{LVE}}^+(t)`:

- :math:`\sigma_{\text{exp}} > \sigma_{\text{LVE}}`: **Strain hardening** (common in
  branched polymers, associating networks)
- :math:`\sigma_{\text{exp}} < \sigma_{\text{LVE}}`: **Strain softening** (common in
  linear polymers at high rates, shear thinning)
- :math:`\sigma_{\text{exp}} = \sigma_{\text{LVE}}`: **Linear regime** (low rates,
  small strains)

**Damping function** extraction:

.. math::

   h(\gamma) = \frac{\sigma^+(t)}{\sigma_{\text{LVE}}^+(t)} \bigg|_{\gamma = \dot{\gamma}_0 t}


Parameters
----------

.. list-table:: LVEEnvelope Parameters
   :header-rows: 1
   :widths: 20 18 14 48

   * - Parameter
     - Type
     - Default
     - Description
   * - ``shear_rate``
     - float
     - ``1.0``
     - Applied shear rate :math:`\dot{\gamma}_0` (s\ :sup:`-1`).
   * - ``G_i``
     - ndarray | None
     - ``None``
     - Prony mode strengths (Pa). Read from ``data.metadata`` if ``None``.
   * - ``tau_i``
     - ndarray | None
     - ``None``
     - Prony relaxation times (s). Read from ``data.metadata`` if ``None``.
   * - ``G_e``
     - float
     - ``0.0``
     - Equilibrium modulus (Pa). Set to 0 for viscoelastic liquids.
   * - ``t_out``
     - ndarray | None
     - ``None``
     - Output time array. Auto-generated if ``None`` (200 log-spaced points
       from 0.01 to :math:`10 \tau_{\max}`).

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Shear rate:**

- Match the experimental startup rate for direct comparison
- Sweep rates to generate a family of LVE envelopes for rate-dependent analysis

**Prony parameters:**

- Obtain from :doc:`prony_conversion` transform
- Or from :doc:`../models/gmm/generalized_maxwell` model fitting
- Or supply directly from literature values


Input / Output Specifications
-----------------------------

- **Input**: Optional :class:`RheoData`. If ``G_i``/``tau_i`` were set at construction,
  ``data`` can be ``None``. If ``data`` is provided, its ``x`` values are used as the time
  array, and ``G_i``/``tau_i`` can be read from ``data.metadata``.
- **Output**: :class:`RheoData` with ``x`` = time (s), ``y`` = :math:`\sigma_{\text{LVE}}^+(t)` (Pa).
  Metadata includes ``shear_rate``, ``n_modes``, and ``source_transform``.


Usage
-----

Direct Parameter Input
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import LVEEnvelope
   import numpy as np

   # Two-mode polymer melt
   G_i = np.array([5e3, 2e3])       # Mode strengths (Pa)
   tau_i = np.array([0.1, 10.0])    # Relaxation times (s)

   # Compute LVE envelope at γ̇ = 1 s⁻¹
   lve = LVEEnvelope(shear_rate=1.0, G_i=G_i, tau_i=tau_i)
   envelope_data, info = lve.transform()

   t = envelope_data.x
   sigma_lve = envelope_data.y

   # Steady-state viscosity: η₀ = Σ Gᵢτᵢ
   eta_0 = np.sum(G_i * tau_i)
   print(f"η₀ = {eta_0:.0f} Pa·s")

Rate Sweep for Nonlinear Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from rheojax.transforms import LVEEnvelope

   G_i = np.array([1e4, 3e3, 500])
   tau_i = np.array([0.01, 1.0, 100.0])

   rates = [0.01, 0.1, 1.0, 10.0]
   t = np.logspace(-3, 3, 500)

   fig, ax = plt.subplots()
   for rate in rates:
       lve = LVEEnvelope(shear_rate=rate, G_i=G_i, tau_i=tau_i, t_out=t)
       data, _ = lve.transform()
       ax.loglog(data.x, data.y, label=fr'$\dot{{\gamma}} = {rate}$ s$^{{-1}}$')

   ax.set_xlabel('Time (s)')
   ax.set_ylabel(r'$\sigma_{\mathrm{LVE}}^+$ (Pa)')
   ax.legend()
   ax.set_title('LVE Envelope Family')

From Prony Conversion Output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import PronyConversion, LVEEnvelope

   # Step 1: Fit Prony series
   prony = PronyConversion(n_modes=10, direction="time_to_freq")
   _, info = prony.transform(relaxation_data)
   prony_result = info["prony_result"]

   # Step 2: LVE envelope using fitted parameters
   lve = LVEEnvelope(
       shear_rate=0.5,
       G_i=prony_result.G_i,
       tau_i=prony_result.tau_i,
       G_e=prony_result.G_e,
   )
   envelope_data, _ = lve.transform()

From Data Metadata
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import LVEEnvelope
   from rheojax.core.data import RheoData
   import numpy as np

   # Data with Prony parameters in metadata
   t = np.logspace(-2, 2, 100)
   data = RheoData(
       x=t, y=np.zeros_like(t),
       metadata={'G_i': [5000, 2000], 'tau_i': [0.1, 10.0], 'G_e': 100.0}
   )

   lve = LVEEnvelope(shear_rate=1.0)
   envelope_data, info = lve.transform(data)

   result = info["lve_result"]
   print(f"Modes: {len(result.G_i)}, Rate: {result.shear_rate} s⁻¹")


Output Structure
----------------

.. list-table:: LVEEnvelopeResult Attributes
   :header-rows: 1
   :widths: 25 20 55

   * - Attribute
     - Type
     - Description
   * - ``t``
     - ndarray
     - Time array (s)
   * - ``sigma_lve``
     - ndarray
     - Stress envelope :math:`\sigma_{\text{LVE}}^+(t)` (Pa)
   * - ``G_i``
     - ndarray
     - Prony mode strengths used (Pa)
   * - ``tau_i``
     - ndarray
     - Prony relaxation times used (s)
   * - ``shear_rate``
     - float
     - Applied shear rate (s\ :sup:`-1`)


See Also
--------

- :doc:`prony_conversion` — Fit Prony parameters from :math:`G(t)` or :math:`G^*(\omega)` data
- :doc:`spectrum_inversion` — Continuous spectrum recovery (alternative to Prony)
- :doc:`../models/gmm/generalized_maxwell` — Multi-mode Maxwell model (provides Prony parameters)
- :doc:`../models/classical/maxwell` — Single Maxwell element (1-mode special case)
- :doc:`fft` — Non-parametric time-frequency interconversion


API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.LVEEnvelope`


References
----------

1. Ferry, J.D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed. Wiley.
   Chapter 4: Stress growth in startup of steady shear.

2. Dealy, J.M. & Larson, R.G. (2006). *Structure and Rheology of Molten Polymers:
   From Structure to Flow Behavior and Back Again*. Hanser.

3. Wagner, M.H. (1976). "Analysis of time-dependent non-linear stress-growth data
   for shear and elongational flow of a low-density branched polyethylene melt."
   *Rheol. Acta*, 15, 136–142.
   DOI: `10.1007/BF01517505 <https://doi.org/10.1007/BF01517505>`_

4. Osaki, K., Inoue, T., & Isomura, T. (2000). "Stress overshoot of polymer
   solutions at high rates of shear." *J. Polym. Sci. Part B: Polym. Phys.*,
   38, 1917–1925.
   DOI: `10.1002/1099-0488(20000715)38:14<1917::AID-POLB100>3.0.CO;2-6 <https://doi.org/10.1002/1099-0488(20000715)38:14%3C1917::AID-POLB100%3E3.0.CO;2-6>`_
