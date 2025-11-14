.. _model-fractional-kv-zener:

Fractional Kelvin-Voigt-Zener (Fractional)
==========================================

Quick Reference
---------------

**Use when:** Creep/retardation analysis, solid with finite equilibrium compliance
**Parameters:** 4 (Ge, Gk, α, τ)
**Key equation:** :math:`J(t) = \frac{1}{G_e} + \frac{1}{G_k}[1 - E_{\alpha}(-(t/\tau)^{\alpha})]`
**Test modes:** Creep, oscillation
**Material examples:** Viscoelastic solids with retardation spectra, filled polymers, soft tissues

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

A Fractional Kelvin-Voigt element (spring :math:`G_k` in parallel with SpringPot) in series with a spring :math:`G_e`. Natural for creep/retardation analysis with finite equilibrium compliance.

Governing Equations
-------------------

Time domain (creep compliance):

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\Big[1 - E_{\alpha}\!\big(-(t/\tau)^{\alpha}\big)\Big].
   \]

Frequency domain (complex compliance and modulus):

.. math::
   :nowrap:

   \[
   J^{*}(\omega) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\,\frac{1}{1+(i\omega\tau)^{\alpha}},
   \qquad
   G^{*}(\omega) \;=\; \frac{1}{J^{*}(\omega)} .
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
     - Series spring modulus
   * - ``Gk``
     - :math:`G_k`
     - Pa
     - [1e-3, 1e9]
     - KV element modulus
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e6]
     - Retardation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Instantaneous compliance :math:`1/G_e`; long-time compliance :math:`1/G_e + 1/G_k`.
- Power-law retardation over decades for :math:`0<\alpha<1`.
- Useful when creep is the primary observable.

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Zener in creep form.
- :math:`G_k \to \infty`: reduces to series spring only.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalKelvinVoigtZener`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalKelvinVoigtZener

   model = FractionalKelvinVoigtZener(Ge=1e3, Gk=5e2, alpha=0.5, tau=1.0)
   response = model.predict(frequency_data)

See also
--------

- :doc:`fractional_kelvin_voigt` — parallel element used inside the FKZ construction.
- :doc:`fractional_zener_sl` — adds a fractional Maxwell branch instead of a Kelvin one.
- :doc:`fractional_maxwell_model` — most general two-order series formulation.
- :doc:`../../transforms/mastercurve` — align creep spectra across temperature before FKZ
  fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — comparisons of Zener
  variants across datasets.

References
----------

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- R.L. Bagley and P.J. Torvik, "On the fractional calculus model of viscoelastic
  behavior," *J. Rheol.* 30, 133–155 (1986).
- R.C. Koeller, "Applications of fractional calculus to the theory of viscoelasticity,"
  *J. Appl. Mech.* 51, 299–307 (1984).
