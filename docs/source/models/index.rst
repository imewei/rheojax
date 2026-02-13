Models Handbook
===============

The RheoJAX model handbook gathers narrative deep dives for every rheological model
implemented in :mod:`rheojax.models`. Use it alongside the API reference for
parameter signatures and the :doc:`/user_guide/02_model_usage/model_selection` tree for high-level guidance.

.. note::

   Each page follows a consistent template (overview, governing equations, parameter
   table, regimes/assumptions, usage snippets, troubleshooting, and references) so you
   can quickly compare models across families.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   summary

----

Linear Viscoelastic Models
--------------------------

Classical spring-dashpot elements, fractional generalizations, and multi-mode models
for fitting small-amplitude oscillatory and relaxation data.

.. toctree::
   :maxdepth: 2
   :caption: Classical & Fractional

   classical/index
   fractional/index
   multi_mode/generalized_maxwell

----

Nonlinear & Flow Models
------------------------

Models for large deformations, steady-state flow curves, and constitutive ODE systems
including Giesekus, flow viscoplastic, and kinematic hardening.

.. toctree::
   :maxdepth: 2
   :caption: Nonlinear Viscoelastic

   giesekus/index

.. toctree::
   :maxdepth: 2
   :caption: Flow & Viscoplastic

   flow/index

.. toctree::
   :maxdepth: 2
   :caption: Elasto-Viscoplastic (IKH/FIKH)

   ikh/index
   fikh/index

----

Thixotropy, Yielding & Soft Glassy Models
-------------------------------------------

Structural-kinetics models (DMT, Fluidity), mean-field emulsion models (HL),
shear transformation zones (STZ), elasto-plastic lattice models (EPM),
soft glassy rheology (SGR), and mode-coupling theory (ITT-MCT).

.. toctree::
   :maxdepth: 2
   :caption: Thixotropic & Yield Stress

   dmt/index
   fluidity/index
   hl/index
   stz/index
   epm/index

.. toctree::
   :maxdepth: 2
   :caption: Soft Glassy & Mode-Coupling

   sgr/index
   itt_mct/index

----

Transient Networks & Vitrimer Models
--------------------------------------

Distribution-tensor polymer network models (TNT, VLB), hybrid vitrimer models
with BER kinetics (HVM), and nanoparticle-filled nanocomposite extensions (HVNM).

.. toctree::
   :maxdepth: 2
   :caption: Transient Networks

   tnt/index
   vlb/index

.. toctree::
   :maxdepth: 2
   :caption: Vitrimer & Nanocomposite

   hvm/index
   hvnm/index

----

DMTA / DMA Analysis
---------------------

Support for Dynamic Mechanical (Thermal) Analysis data via automatic
E* |leftrightarrow| G* modulus conversion.

.. |leftrightarrow| unicode:: U+2194

.. toctree::
   :maxdepth: 2
   :caption: DMTA

   dmta/index

----

LAOS Analysis
--------------

Specialized tools for large-amplitude oscillatory shear analysis.

.. toctree::
   :maxdepth: 2
   :caption: LAOS

   spp/index
