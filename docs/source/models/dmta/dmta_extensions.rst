Future Extensions
=================

Planned DMTA-related features for future RheoJAX releases.

Frequency-Dependent Poisson's Ratio
------------------------------------

Current implementation assumes constant :math:`\nu`.  For glassy polymers near
:math:`T_g`, Poisson's ratio can vary from 0.35 (glassy) to 0.50 (rubbery).
The mathematical form is given in :doc:`dmta_theory`
(see :ref:`viscoelastic-poisson`).

**Status**: Under investigation.  Requires simultaneous shear + tensile data.

Nonlinear DMTA (Large-Amplitude DMA)
--------------------------------------

Extension of LAOS concepts to tensile deformation:

- Lissajous--Bowditch plots in stress--strain space
- Chebyshev decomposition of nonlinear tensile response
- Sequence of Physical Processes (SPP) for tensile LAOS

**Status**: Experimental.  Requires instrument-level nonlinear DMA capability.

Multiaxial Deformation Modes
-----------------------------

Support for combined deformation protocols:

- Simultaneous shear + tension (torsion-tension rheometry)
- Biaxial extension (bubble inflation)
- Poisson's ratio extraction from dual-mode measurements

**Status**: Planned for v0.7.0.

Dielectric / Mechanical Correlation
-------------------------------------

Correlating DMTA with broadband dielectric spectroscopy (BDS):

- Shared relaxation spectrum :math:`H(\tau)` between mechanical and dielectric
- Havriliak--Negami ↔ fractional Zener parameter mapping
- Joint fitting of :math:`E^*(\omega)` and :math:`\varepsilon^*(\omega)`

**Status**: Research prototype.

FEM Export (Prony Series)
--------------------------

Export fitted Prony series to commercial FEM packages:

.. code-block:: python

   # Planned API
   gmm.export_prony("material.inp", format="abaqus")
   gmm.export_prony("material.xml", format="ansys")

**Status**: Planned for v0.7.0.  Generalized Maxwell already provides the
discrete spectrum :math:`\{(G_i, \tau_i)\}`; only format writers are needed.

.. seealso::

   - :doc:`dmta_theory` --- current conversion implementation and Poisson's ratio
   - :doc:`dmta_knowledge` --- physical quantities already extractable from DMTA fits
   - :doc:`dmta_workflows` --- existing workflows that will benefit from these extensions
