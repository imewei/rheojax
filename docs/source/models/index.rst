Models Handbook
===============

The RheoJAX model handbook gathers narrative deep dives for every rheological model
implemented in :mod:`rheojax.models`. Use it alongside the API reference for
parameter signatures and the :doc:`/user_guide/model_selection` tree for high-level guidance.

.. note::

   Each page follows a consistent template (overview, governing equations, parameter
   table, regimes/assumptions, usage snippets, troubleshooting, and references) so you
   can quickly compare models across classical, fractional, and flow families.

.. toctree::
   :maxdepth: 1
   :caption: Overview

   summary

.. toctree::
   :maxdepth: 1
   :caption: Classical Models

   classical/maxwell
   classical/zener
   classical/springpot

.. toctree::
   :maxdepth: 1
   :caption: Fractional Models

   fractional/fractional_maxwell_gel
   fractional/fractional_maxwell_liquid
   fractional/fractional_maxwell_model
   fractional/fractional_kelvin_voigt
   fractional/fractional_zener_sl
   fractional/fractional_zener_ss
   fractional/fractional_zener_ll
   fractional/fractional_kv_zener
   fractional/fractional_burgers
   fractional/fractional_poynting_thomson
   fractional/fractional_jeffreys

.. toctree::
   :maxdepth: 1
   :caption: Flow & Viscoplastic Models

   flow/power_law
   flow/carreau
   flow/carreau_yasuda
   flow/cross
   flow/herschel_bulkley
   flow/bingham

.. toctree::
   :maxdepth: 1
   :caption: Multi-Mode Models

   multi_mode/generalized_maxwell

.. toctree::
   :maxdepth: 1
   :caption: Soft Glassy Rheology (SGR)

   sgr/sgr_conventional
   sgr/sgr_generic

.. toctree::
   :maxdepth: 1
   :caption: Shear Transformation Zone (STZ)

   stz/stz_conventional

.. toctree::
   :maxdepth: 1
   :caption: Hébraud–Lequeux (HL)

   hl/hebraud_lequeux

.. toctree::
   :maxdepth: 1
   :caption: Elasto-Plastic Models (EPM)

   epm/lattice_epm
   epm/tensorial_epm

.. toctree::
   :maxdepth: 1
   :caption: Fluidity Models

   fluidity/fluidity_local
   fluidity/fluidity_nonlocal

.. toctree::
   :maxdepth: 1
   :caption: Fluidity-Saramito EVP Models

   fluidity/saramito_evp

.. toctree::
   :maxdepth: 1
   :caption: Sequence of Physical Processes (SPP)

   spp/spp_decomposer
   spp/spp_yield_stress

.. toctree::
   :maxdepth: 2
   :caption: Elasto-Viscoplastic (IKH) Models

   ikh/index
