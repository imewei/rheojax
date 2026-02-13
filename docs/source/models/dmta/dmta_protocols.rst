DMTA Measurement Protocols
==========================

This page maps standard DMTA/DMA measurement protocols to RheoJAX
``test_mode`` parameters, following ISO 6721 and ASTM D4065 conventions.

Protocol-to-Test-Mode Mapping
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - DMTA Protocol
     - Standard
     - RheoJAX ``test_mode``
     - ``deformation_mode``
   * - Frequency sweep (isothermal, tensile)
     - `ISO 6721-4`_
     - ``oscillation``
     - ``tension``
   * - Temperature sweep (:math:`T_g` determination)
     - `ISO 6721-11`_
     - ``oscillation``
     - ``tension``
   * - Multi-frequency temperature sweep
     - `ASTM D4065`_
     - ``oscillation``
     - ``tension``
   * - Creep / recovery (tensile)
     - `ISO 899-1`_
     - ``creep``
     - ``tension``
   * - Shear vibration (sandwich geometry)
     - `ISO 6721-6`_
     - ``oscillation``
     - ``shear``
   * - Parallel plate rheometry
     - `ISO 6721-10`_
     - ``oscillation``
     - ``shear``

.. _ISO 6721-4: https://www.iso.org/standard/73144.html
.. _ISO 6721-11: https://www.iso.org/standard/74988.html
.. _ASTM D4065: https://www.astm.org/d4065-20.html
.. _ISO 899-1: https://www.iso.org/standard/70413.html
.. _ISO 6721-6: https://www.iso.org/standard/73146.html
.. _ISO 6721-10: https://www.iso.org/standard/62159.html

.. note::

   The ISO 6721 series covers *dynamic* mechanical properties only.
   Static tests (stress relaxation, creep) fall under separate standards.
   For tensile stress relaxation on plastics, no single ISO standard exists;
   use instrument vendor protocols or `ASTM E328`_.

.. _ASTM E328: https://www.astm.org/e0328-21.html

Instrument Geometry Mapping
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 20 20

   * - Clamp / Geometry
     - Measures
     - ``DeformationMode``
     - Poisson Needed?
   * - Film tension clamp
     - :math:`E^*`
     - ``TENSION``
     - Yes
   * - 3-point bending
     - :math:`E^*`
     - ``BENDING``
     - Yes
   * - Single/dual cantilever
     - :math:`E^*`
     - ``BENDING``
     - Yes
   * - Compression clamp
     - :math:`E^*`
     - ``COMPRESSION``
     - Yes
   * - Shear sandwich
     - :math:`G^*`
     - ``SHEAR``
     - No

.. note::

   All tensile-family geometries (tension, bending, compression) produce
   :math:`E^*` data and require Poisson's ratio for :math:`E^* \to G^*`
   conversion.  See :doc:`dmta_theory` for recommended values by material class.

Temperature Sweep + TTS Pipeline
----------------------------------

The most common DMTA experiment is a multi-temperature frequency sweep at
2--5 frequencies, followed by time--temperature superposition:

1. **Collect** isothermal frequency sweeps at 10--20 temperatures spanning
   :math:`T_g \pm 50` °C (finer spacing near :math:`T_g`)
2. **Build master curve** using :class:`~rheojax.transforms.Mastercurve`
   with WLF or Arrhenius shift factors
3. **Fit** the master curve with GMM or fractional models
4. **Extract** WLF parameters :math:`C_1, C_2` and activation energy
   :math:`E_a = 2.303\,R\,C_1 C_2`

.. code-block:: python

   from rheojax.transforms import Mastercurve

   mc = Mastercurve(reference_temp=T_ref, method='wlf')
   master, shifts = mc.transform(datasets)

   # Or use auto-shift (no manual shift factors needed)
   mc_auto = Mastercurve(reference_temp=T_ref, auto_shift=True)
   master, shifts = mc_auto.transform(datasets)

Recommended Heating Rates
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Material Type
     - Rate (°C/min)
     - Rationale
   * - Amorphous polymer
     - 2--3
     - Standard; captures :math:`T_g` accurately
   * - Semi-crystalline
     - 1--2
     - Avoid melting kinetics artefacts
   * - Vitrimer / CAN
     - 1--2
     - Resolve :math:`T_v` transition from :math:`T_g`
   * - Elastomer (above :math:`T_g`)
     - 3--5
     - Broad rubbery plateau, less sensitive to rate

.. important::

   Faster heating rates shift apparent :math:`T_g` to higher values
   (~2 °C per doubling of rate).  Always report the heating rate alongside
   :math:`T_g`.

Creep Compliance Protocol
--------------------------

Some DMTA instruments can measure tensile creep compliance :math:`D(t) = 1/E(t)`:

.. code-block:: python

   model.fit(
       t, E_relax,
       test_mode='relaxation',
       deformation_mode='tension',
       poisson_ratio=0.5,
   )

   # For creep compliance, use test_mode='creep'
   # (available for models with creep support)

Usage Example
-------------

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()

   # Frequency sweep from DMTA (tension clamp)
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.40,  # semicrystalline polymer
   )

   # Temperature sweep — same API, just different x-axis
   # (pre-process with Mastercurve to get master E*(ω) first)

.. seealso::

   - :doc:`dmta_workflows` --- end-to-end workflows (TTS, Bayesian, CSV loading)
   - :doc:`dmta_theory` --- Poisson's ratio values and conversion details
   - :doc:`dmta_knowledge` --- :math:`T_g` extraction conventions from DMTA data
