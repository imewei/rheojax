DMTA / DMA Analysis
===================

RheoJAX supports data from Dynamic Mechanical (Thermal) Analysis instruments
through automatic E* |leftrightarrow| G* modulus conversion.  All 41+
oscillation-capable models can fit DMTA data without any model-level changes.

.. |leftrightarrow| unicode:: U+2194

Quick Start
-----------

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,   # rubber (0.35 for glassy, 0.40 for semicrystalline)
   )
   E_pred = model.predict(omega, test_mode='oscillation')  # returns E*

The Key Insight
---------------

The relaxation spectrum :math:`H(\tau)` is a material property independent of
deformation mode.  Shear, tension, and bending all share the same spectrum
--- only the amplitude scale changes:

.. math::

   E^*(\omega) = 2(1 + \nu)\,G^*(\omega)

This means **every OSCILLATION-capable model** in RheoJAX works with DMTA
data after a simple modulus conversion at the ``fit()`` / ``predict()``
boundary.

What's in This Section
-----------------------

.. list-table::
   :widths: 30 70

   * - :doc:`dmta_theory`
     - E* |leftrightarrow| G* conversion, Poisson's ratio, Kramers--Kronig, relaxation spectra
   * - :doc:`dmta_models`
     - Which model to choose (decision table, complexity ladder, 41+ compatible)
   * - :doc:`dmta_numerical`
     - JIT strategy, parameter bounds, convergence, FAST_MODE, memory management
   * - :doc:`dmta_workflows`
     - 8 end-to-end workflows (direct fit, TTS, Bayesian, CSV loading, HVM, cross-domain)
   * - :doc:`dmta_protocols`
     - ISO/ASTM protocol mapping, instrument geometries, heating rates
   * - :doc:`dmta_knowledge`
     - T_g extraction, relaxation spectrum, tan(delta), plateau modulus, cooperativity
   * - :doc:`dmta_extensions`
     - Planned: frequency-dependent nu, nonlinear DMA, FEM export

Example Notebooks
-----------------

.. list-table::
   :widths: 40 60

   * - ``01_dmta_basics``
     - E* |leftrightarrow| G* conversion fundamentals
   * - ``02_dmta_master_curve``
     - Multi-temperature TTS + GMM/FZSS fitting on real data
   * - ``03_dmta_fractional_models``
     - Fractional viscoelasticity + Bayesian UQ
   * - ``04_dmta_relaxation``
     - Time-domain E(t) Prony series + cross-domain
   * - ``05_dmta_vitrimer``
     - HVM/HVNM with tensile deformation
   * - ``06_dmta_model_selection``
     - Multi-model comparison (synthetic + real data)
   * - ``07_dmta_tts_pipeline``
     - Raw multi-T |rarr| TTS |rarr| fit |rarr| WLF extraction
   * - ``08_dmta_cross_domain``
     - Frequency |leftrightarrow| relaxation domain consistency

.. |rarr| unicode:: U+2192

.. toctree::
   :maxdepth: 2

   dmta_theory
   dmta_models
   dmta_numerical
   dmta_workflows
   dmta_protocols
   dmta_knowledge
   dmta_extensions
