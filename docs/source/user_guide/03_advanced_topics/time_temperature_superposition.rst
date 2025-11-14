.. _time_temperature_superposition:

Time-Temperature Superposition
===============================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Understand the principle of time-temperature superposition (TTS)
   2. Construct mastercurves from multi-temperature data
   3. Apply WLF and Arrhenius shifting
   4. Validate TTS applicability for your material

.. admonition:: Prerequisites
   :class: important

   - :doc:`../01_fundamentals/what_is_rheology` — Viscoelasticity basics
   - :doc:`../02_model_usage/model_families` — Fractional models

The TTS Principle
-----------------

**Time-Temperature Superposition (TTS)** exploits the equivalence between temperature and timescale:

- **Low temperature** = **Short time** (fast deformation)
- **High temperature** = **Long time** (slow deformation)

By measuring at multiple temperatures, you can extend the accessible frequency range by many decades.

**Application**: Polymer melts and amorphous materials above T_g

Basic Workflow
--------------

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve

   # Multi-temperature SAOS data
   datasets = {
       160: (omega_160, G_star_160),
       180: (omega_180, G_star_180),
       200: (omega_200, G_star_200)
   }

   # Create mastercurve (WLF shifting)
   mc = Mastercurve(reference_temp=180, method='wlf', C1=17.44, C2=51.6)
   mastercurve, shift_factors = mc.transform(datasets)

   # Plot extended frequency range
   plt.loglog(mastercurve.x, mastercurve.y[:, 0], label="G' master")

For detailed usage, see :doc:`/transforms/mastercurve`.

Summary
-------

TTS extends the accessible frequency range by shifting multi-temperature data using WLF or Arrhenius equations.
This technique is essential for characterizing polymer dynamics over wide timescales.

Next Steps
----------

Explore: :doc:`../04_practical_guides/index` for production workflows.
