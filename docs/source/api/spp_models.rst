.. _api-spp:

SPP Analysis API
================

The Sequence of Physical Processes (SPP) tooling in RheoJAX provides time-domain LAOS
analysis without relying on Fourier/Chebyshev expansions. It is built from three
primary objects:

- :class:`rheojax.transforms.spp_decomposer.SPPDecomposer` — per-cycle LAOS
  decomposition into cage modulus, static/dynamic yield stresses, power-law flow, and
  nonlinearity metrics.
- :class:`rheojax.models.spp_yield_stress.SPPYieldStress` — a fit-ready model (NLSQ and
  NumPyro NUTS) that parameterizes SPP yield behavior across amplitudes or steady shear.
- :class:`rheojax.pipeline.workflows.SPPAmplitudeSweepPipeline` — convenience pipeline
  for amplitude-sweep LAOS workflows.

For conceptual background see :doc:`/user_guide/03_advanced_topics/spp_analysis`.

.. contents:: Page Contents
   :local:
   :depth: 2


SPPDecomposer
-------------

.. autoclass:: rheojax.transforms.spp_decomposer.SPPDecomposer
   :members:
   :undoc-members:
   :show-inheritance:


SPPYieldStress Model
--------------------

.. autoclass:: rheojax.models.spp_yield_stress.SPPYieldStress
   :members:
   :undoc-members:
   :show-inheritance:


SPP Amplitude Sweep Pipeline
----------------------------

.. autoclass:: rheojax.pipeline.workflows.SPPAmplitudeSweepPipeline
   :members:
   :undoc-members:
   :show-inheritance:


Helper Functions
----------------

These JAX-first kernels are re-exported for completeness; see
``rheojax.utils.spp_kernels`` for details.

.. automodule:: rheojax.utils.spp_kernels
   :members:
   :undoc-members:
