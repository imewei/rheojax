.. _model-fractional-zener-sl:

Fractional Zener Solid-Liquid (Fractional)
==========================================

Quick Reference
---------------

**Use when:** Solid-like behavior with equilibrium plateau and fractional relaxation tails
**Parameters:** 4 (Ge, c_α, α, τ)
**Key equation:** :math:`G(t) = G_e + c_\alpha t^{-\alpha} E_{1-\alpha,1}(-(t/\tau)^{1-\alpha})`
**Test modes:** Oscillation, relaxation
**Material examples:** Viscoelastic solids with finite equilibrium modulus and power-law relaxation

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------
Fractional Maxwell element in parallel with a spring to capture solid-like plateaus with fractional relaxation tails.

Governing Equations
-------------------
Time domain (relaxation modulus):

.. math::
   :nowrap:

   \[
   G(t) = G_e + c_\alpha t^{-\alpha} E_{1-\alpha,1}\left(-\left(\frac{t}{\tau}\right)^{1-\alpha}\right).
   \]

Frequency domain (complex modulus):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) = G_e + \frac{c_\alpha (i\omega)^{\alpha}}{1 + (i\omega\tau)^{1-\alpha}}.
   \]

Parameters
----------
.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 12 12 18 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``Ge``
     - :math:`G_e`
     - Pa
     - [1e-3, 1e9]
     - Equilibrium modulus
   * - ``c_alpha``
     - :math:`c_\alpha`
     - Pa*s^alpha
     - [1e-3, 1e9]
     - SpringPot constant
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e+6]
     - Relaxation time

Validity and Assumptions
------------------------
- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------
- Low-frequency limit recovers the equilibrium modulus :math:`G_e`.
- Mid-band shows fractional dissipation with slope :math:`\alpha`.
- High-frequency response approaches :math:`G_e + c_\alpha (i\omega)^{\alpha}`.

Limiting Behavior
-----------------
- :math:`\alpha \to 1`: classical Zener solid-liquid.
- :math:`G_e \to 0`: fractional Maxwell gel.
- :math:`c_\alpha \to 0`: pure elastic spring.

API References
--------------
- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalZenerSolidLiquid`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalZenerSolidLiquid

   model = FractionalZenerSolidLiquid(Ge=1e3, c_alpha=5e2, alpha=0.5, tau=1.0)
   response = model.predict(frequency_data)

See also
--------

- :doc:`fractional_zener_ss` and :doc:`fractional_zener_ll` — alternative plateau choices.
- :doc:`fractional_kv_zener` — Kelvin-Voigt-based analogue sharing the same compliance.
- :doc:`fractional_burgers` — combines fractional Maxwell and Kelvin branches.
- :doc:`../../transforms/mutation_number` — monitor when the solid assumption holds during
  gelation.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — fractional Zener
  comparison notebook.

References
----------

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- R.L. Bagley and P.J. Torvik, "A theoretical basis for the application of fractional
  calculus to viscoelasticity," *J. Rheol.* 27, 201–210 (1983).
- R.C. Koeller, "Applications of fractional calculus to the theory of viscoelasticity,"
  *J. Appl. Mech.* 51, 299–307 (1984).
