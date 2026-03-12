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

For isotropic, linear-viscoelastic materials these are connected by the
complex Poisson's ratio :math:`\nu^*`:

.. math::

   E^*(\omega) = 2\bigl(1 + \nu^*(\omega)\bigr)\,G^*(\omega)

A direct interchange :math:`E^* \leftrightarrow G^*` is **only valid under
specific conditions**.  Understanding when the conversion is safe — and when
it fails — is essential for correct DMTA analysis.

.. important::

   The relaxation spectrum :math:`H(\tau)` is a material property independent of
   deformation mode.  Shear, tension, and bending all share the same spectrum
   --- only the amplitude scale changes.  Every ``OSCILLATION``-capable model in
   RheoJAX is mathematically applicable to DMTA data after a modulus
   conversion, **provided the conditions below are satisfied**.

.. _conversion-validity:

Validity Conditions for :math:`E^* \leftrightarrow G^*`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following four conditions must hold for the scalar conversion to be
physically meaningful.

**Condition 1 — Incompressibility** (:math:`\nu \to 1/2`)

For a truly incompressible material the conversion simplifies to:

.. math::

   E^* = 3\,G^*

This is the most commonly invoked condition in soft matter and polymer physics.
It applies when:

- **Rubbers and elastomers** above :math:`T_g` (bulk modulus
  :math:`K \gg G`, so :math:`\nu \approx 0.5`)
- **Hydrogels** and biological soft tissues
- **Polymer melts** and concentrated solutions in the terminal regime
- Any system where volumetric deformation is energetically costly compared
  to shear (:math:`K/G \gg 1`)

The factor-of-3 conversion is exact in the incompressible limit and is
widely used in rubber elasticity and XPCS/rheology cross-comparisons.

**Condition 2 — Real, frequency-independent Poisson's ratio**

If :math:`\nu^* \approx \nu` (real, not complex), the conversion remains clean:

.. math::

   E^* = 2(1 + \nu)\,G^*

This holds when volumetric relaxation is either absent or occurs on a very
different timescale than the shear relaxation being probed.  In practice:

- Glassy polymers near and below :math:`T_g`:
  :math:`\nu \approx 0.33\text{--}0.40`, so :math:`E \approx 2.6\text{--}2.8\,G`
- Semi-crystalline polymers: :math:`\nu \approx 0.35\text{--}0.45`

Here :math:`E^*` and :math:`G^*` are proportional but **not equal** — the
conversion factor must be applied explicitly.

**Condition 3 — Isotropy**

The relation :math:`E = 2G(1+\nu)` assumes **linear elastic isotropy**:

- No fibre reinforcement, crystalline texture, or flow-induced anisotropy
- Sample geometry must not induce multiaxial stress states that break the
  uniaxial / simple-shear assumption

For oriented polymer films, highly anisotropic nanocomposites, or liquid
crystalline networks, this scalar conversion fails.  The moduli must instead
be treated as full compliance/stiffness tensors.

**Condition 4 — Linear viscoelastic (LVE) regime**

Both moduli must be measured strictly within the linear viscoelastic limit.
If either the DMA or the shear rheometer applies a strain amplitude large
enough to induce non-linear structural breakdown (Payne effect in filled
networks, chain disentanglement), the linear conversion is no longer
mathematically valid.

.. _conversion-failure:

When the Conversion Fails
~~~~~~~~~~~~~~~~~~~~~~~~~~

The conversion **breaks down** — and using :math:`E^* \approx 3G^*` can be
seriously wrong — in the following situations:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Situation
     - Why it fails
   * - Near glass transition (:math:`T_g \pm 20\,°C`)
     - :math:`\nu^*` becomes strongly frequency-dependent and complex;
       volumetric relaxation couples to shear
   * - Semicrystalline polymers under tension
     - Crystalline lamellae introduce anisotropy; :math:`\nu` varies with
       orientation
   * - Foams, cellular solids
     - Compressibility is significant; :math:`K \sim G`
   * - Filled systems (high :math:`\varphi` particles)
     - Compressive and shear reinforcement differ; effective :math:`\nu` shifts
   * - Highly crosslinked thermosets
     - :math:`\nu` can drop to :math:`\sim 0.3`;
       :math:`E \approx 2.6\,G`, not :math:`3\,G`

The most physically rich failure case is **near** :math:`T_g`, where the bulk
modulus :math:`K` relaxes on a different timescale than :math:`G`.  In this
regime:

.. math::

   \nu^*(\omega) = \frac{3K^*(\omega) - 2G^*(\omega)}
   {2\bigl(3K^*(\omega) + G^*(\omega)\bigr)}

is itself a complex, frequency-dependent quantity.  The measured :math:`E^*`
from a tensile DMA mode encodes a **mixture of** :math:`K^*` **and**
:math:`G^*` responses that cannot be cleanly separated without independent
volumetric measurements.

**Geometric and boundary effects.**  In a DMA tensile test of a short, thick
sample the clamps restrict lateral contraction.  This introduces shear stresses
into the nominally pure-tension measurement, artificially inflating :math:`E^*`
and breaking the mathematical conversion unless aspect-ratio corrections are
applied.

.. _conversion-practical:

Practical Summary
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Material Class
     - Safe?
     - :math:`\nu`
     - :math:`E/G` Factor
   * - Rubbers, elastomers (:math:`T > T_g + 30\,\text{K}`)
     - Yes
     - 0.50
     - 3.0
   * - Hydrogels, biopolymer networks
     - Yes
     - 0.50
     - 3.0
   * - Polymer melts (terminal regime)
     - Yes
     - 0.50
     - 3.0
   * - Glassy polymers (:math:`T < T_g - 30\,\text{K}`)
     - With caution
     - 0.33--0.40
     - 2.6--2.8
   * - Semi-crystalline polymers
     - With caution
     - 0.35--0.45
     - 2.7--2.9
   * - Near :math:`T_g` (:math:`\pm 20\,°\text{C}`)
     - **No**
     - complex :math:`\nu^*(\omega)`
     - —
   * - Filled / composite systems
     - Verify independently
     - depends on :math:`\varphi`, morphology
     - —
   * - Foams, cellular solids
     - **No**
     - :math:`K \sim G`
     - —

.. warning::

   Blindly applying :math:`E^* = 3G^*` across a full temperature or frequency
   sweep through the glass transition will artificially skew the shape and
   width of the :math:`\tan\delta` peak and the transition zone.  For broad
   master curves spanning :math:`T_g`, see :ref:`viscoelastic-poisson` below
   for mitigation strategies.

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

The :math:`K^*`/:math:`G^*` Coupling Near :math:`T_g`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the glass transition region, the bulk modulus :math:`K^*` and shear modulus
:math:`G^*` relax on **different timescales**.  :math:`K^*` typically has a
weaker dispersion than :math:`G^*` (volumetric relaxation is faster and
narrower).  This means the complex Poisson's ratio:

.. math::

   \nu^*(\omega) = \frac{3K^*(\omega) - 2G^*(\omega)}
   {2\bigl(3K^*(\omega) + G^*(\omega)\bigr)}

has both a real and imaginary part that vary with frequency.  Consequently,
a tensile DMA measurement in this regime **encodes a mixture of** :math:`K^*`
**and** :math:`G^*`, and cannot be cleanly decomposed into shear-only
information without an independent volumetric (e.g.\ dilatometric or
ultrasonic) measurement.

**Practical impact**: The :math:`\tan\delta` peak from a tensile DMA test
is shifted in both frequency and height relative to the :math:`\tan\delta`
from a shear rheometer.  This is a real physical difference, not a calibration
artefact.

Application to Vitrimers and Exchangeable Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For vitrimer networks modelled with :doc:`/models/hvm/index` or
:doc:`/models/hvnm/index`, the incompressible approximation
:math:`E^* = 3G^*` is typically safe in the rubbery plateau and terminal
regimes.  However, caution is required near the topology freezing temperature
:math:`T_v`, where bond-exchange kinetics can contribute to volumetric
relaxation and cause :math:`\nu` to drift from the incompressible limit.

.. important::

   RheoJAX currently assumes **constant** :math:`\nu`.  For a master curve
   spanning the full glass transition, this introduces a systematic error of
   up to ~11% in modulus (factor 3.0 / 2.7).  Recommended practice:

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
