DMTA Theory & Conversion
========================

Physical Background
-------------------

DMTA (Dynamic Mechanical Thermal Analysis) and DMA (Dynamic Mechanical Analysis)
instruments apply **tensile, bending, or compressive** deformations to measure
the complex Young's modulus:

.. math::

   E^*(\omega) = E'(\omega) + i\,E''(\omega)

while rotational rheometers apply **shear** deformations to measure
the complex shear modulus:

.. math::

   G^*(\omega) = G'(\omega) + i\,G''(\omega)

For isotropic, linear-viscoelastic materials these are connected by Poisson's ratio
:math:`\nu`:

.. math::

   E^*(\omega) = 2(1 + \nu)\,G^*(\omega)

Common Poisson's ratios by material class:

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Material
     - :math:`\nu`
     - :math:`E/G` factor
   * - Rubbers / Elastomers (:math:`T \gg T_g`)
     - 0.50
     - 3.0
   * - Glassy polymers (:math:`T \ll T_g`)
     - 0.35
     - 2.7
   * - Semi-crystalline polymers
     - 0.40
     - 2.8
   * - Metals
     - 0.30
     - 2.6

.. important::

   The relaxation spectrum :math:`H(\tau)` is a material property independent of
   deformation mode.  Shear, tension, and bending all share the same spectrum
   --- only the amplitude scale changes.  Every ``OSCILLATION``-capable model in
   RheoJAX is mathematically applicable to DMTA data after a simple modulus
   conversion.

Deformation Modes
-----------------

RheoJAX recognises four deformation geometries via the
:class:`~rheojax.core.test_modes.DeformationMode` enum:

.. list-table::
   :header-rows: 1

   * - Mode
     - Measures
     - Typical Instrument
   * - ``SHEAR``
     - :math:`G^*`
     - Rotational rheometer
   * - ``TENSION``
     - :math:`E^*`
     - DMTA / DMA (film tension clamp)
   * - ``BENDING``
     - :math:`E^*`
     - DMTA (3-point bending, cantilever)
   * - ``COMPRESSION``
     - :math:`E^*`
     - DMA (compression clamp)

All tensile modes (``TENSION``, ``BENDING``, ``COMPRESSION``) measure :math:`E^*`
and share the same conversion factor.  The instrument firmware handles
geometry-specific corrections; the data reaching RheoJAX is always :math:`E^*`.

Conversion Utility
------------------

The :mod:`rheojax.utils.modulus_conversion` module provides:

.. code-block:: python

   from rheojax.utils.modulus_conversion import convert_modulus

   # E* -> G* conversion (rubber, nu = 0.5)
   G_star = convert_modulus(E_star, "tension", "shear", poisson_ratio=0.5)

   # G* -> E* conversion
   E_star = convert_modulus(G_star, "shear", "tension", poisson_ratio=0.35)

   # Roundtrip is exact
   E_recovered = convert_modulus(
       convert_modulus(E_star, "tension", "shear", poisson_ratio=0.4),
       "shear", "tension", poisson_ratio=0.4,
   )

Automatic Conversion at fit()
------------------------------

The preferred workflow passes DMTA data directly to ``model.fit()`` with
``deformation_mode`` and ``poisson_ratio``:

.. code-block:: python

   from rheojax.models import Maxwell

   model = Maxwell()
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,
   )

   # Parameters are in G-space (model-native)
   G0 = model.parameters.get_value("G0")  # NOT E0

   # predict() returns E* automatically
   E_pred = model.predict(omega, test_mode='oscillation')

Internally, ``BaseModel.fit()`` converts :math:`E^* \to G^*` before calling the
model's ``_fit()`` method, and ``predict()`` converts the result back to
:math:`E^*`.  This means **all 41+ DMTA-compatible models work without any
model-level changes**.

.. _viscoelastic-poisson:

Viscoelastic Poisson's Ratio
----------------------------

The Poisson's ratio of polymers is **not constant** through the glass transition.
In the glassy state :math:`\nu \approx 0.35`; in the rubbery state
:math:`\nu \approx 0.50` (incompressible limit).  The transition follows a
relaxation function analogous to the modulus:

.. math::

   \nu(\omega) \approx \nu_\infty +
   \frac{\nu_0 - \nu_\infty}{1 + (i\omega\tau_\nu)^\beta}

where :math:`\nu_0 \approx 0.50` (rubbery), :math:`\nu_\infty \approx 0.35`
(glassy), and :math:`\tau_\nu` is the Poisson relaxation time.

.. important::

   RheoJAX currently assumes **constant** :math:`\nu`.  For a master curve
   spanning the full glass transition, this introduces a systematic error of
   up to ~11% in modulus (factor 3.0/2.7).  Recommended practice:

   - **Rubbery plateau analysis** (:math:`T \gg T_g`): use :math:`\nu = 0.50`
   - **Glassy plateau analysis** (:math:`T \ll T_g`): use :math:`\nu = 0.35`
   - **Broad master curve through** :math:`T_g`: use :math:`\nu = 0.40`
     (compromise) or fit both plateaus separately

   See :doc:`dmta_extensions` for the planned frequency-dependent :math:`\nu`
   feature.

Kramers--Kronig Relations
--------------------------

The storage and loss moduli are not independent.  For any linear viscoelastic
material, they are connected by the Kramers--Kronig (KK) integral transforms:

.. math::

   E'(\omega) - E_\infty = \frac{2}{\pi} \int_0^\infty
   \frac{u\,E''(u)}{u^2 - \omega^2}\,du

.. math::

   E''(\omega) = -\frac{2\omega}{\pi} \int_0^\infty
   \frac{E'(u) - E_\infty}{u^2 - \omega^2}\,du

**Practical consequence**: A model that fits :math:`E'(\omega)` perfectly should
automatically fit :math:`E''(\omega)` well, and vice versa.  A large discrepancy
between :math:`R^2(E')` and :math:`R^2(E'')` signals either:

1. The model is too simple (e.g., single-mode Zener for a broad transition)
2. The data has systematic errors (different instruments for E' and E'')
3. The material exhibits thermorheological complexity (no clean TTS)

Relaxation Spectrum Representations
-------------------------------------

The relaxation spectrum :math:`H(\tau)` is the fundamental material function
connecting time-domain and frequency-domain behaviour:

.. math::

   E(t) = E_\infty + \int_0^\infty H(\tau)\,e^{-t/\tau}\,d\ln\tau

.. math::

   E'(\omega) = E_\infty + \int_0^\infty H(\tau)\,
   \frac{\omega^2\tau^2}{1 + \omega^2\tau^2}\,d\ln\tau

.. math::

   E''(\omega) = \int_0^\infty H(\tau)\,
   \frac{\omega\tau}{1 + \omega^2\tau^2}\,d\ln\tau

Two representations are common in RheoJAX:

**Discrete (Prony series)** --- :class:`~rheojax.models.multi_mode.GeneralizedMaxwell`:

.. math::

   H(\tau) = \sum_{i=1}^{N} E_i\,\delta(\tau - \tau_i)
   \quad\Rightarrow\quad
   E^*(\omega) = E_\infty + \sum_{i=1}^{N}
   \frac{E_i\,(i\omega\tau_i)}{1 + i\omega\tau_i}

**Continuous power-law** --- Fractional models (FZSS, FMM, ...):

.. math::

   H(\tau) \sim \tau^{-\alpha}
   \quad\Rightarrow\quad
   E^*(\omega) \sim (i\omega)^\alpha

Per-Model :math:`E^*(\omega)` Expressions
-------------------------------------------

Since :math:`E^* = 2(1+\nu)\,G^*`, every model's :math:`G^*(\omega)` formula
directly gives :math:`E^*(\omega)`.  Key expressions:

**Maxwell** (single relaxation time):

.. math::

   E^*(\omega) = 2(1+\nu)\,G_0 \frac{i\omega\tau}{1 + i\omega\tau}

**Zener (Standard Linear Solid)**:

.. math::

   E^*(\omega) = 2(1+\nu)\left[G_e + G_m
   \frac{i\omega\tau}{1 + i\omega\tau}\right]

**Fractional Zener (Solid--Solid)**:

.. math::

   E^*(\omega) = 2(1+\nu)\,G_e\,
   \frac{1 + (i\omega\tau)^\alpha}{1 + (G_e/G_g)(i\omega\tau)^\alpha}

where :math:`\alpha \in (0, 1)` is the fractional order controlling the breadth
of the relaxation distribution.

**Generalized Maxwell** (:math:`N`-mode Prony series):

.. math::

   E^*(\omega) = E_\infty + \sum_{i=1}^{N}
   \frac{E_i\,(i\omega\tau_i)}{1 + i\omega\tau_i}

When ``modulus_type='tensile'``, parameters :math:`E_i` are fitted directly.
When ``modulus_type='shear'`` with ``deformation_mode='tension'``,
:math:`G_i` are fitted and :math:`E^* = 2(1+\nu)G^*` is applied at the
prediction boundary.

.. seealso::

   - :doc:`dmta_models` --- which model to choose for your DMTA data
   - :doc:`dmta_knowledge` --- physical quantities extractable from fitted models
   - :doc:`dmta_numerical` --- parameter bounds and convergence for tensile data
