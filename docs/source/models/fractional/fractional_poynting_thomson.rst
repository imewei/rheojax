.. _model-fractional-poynting-thomson:

Fractional Poynting-Thomson (Fractional)
========================================

Quick Reference
---------------

**Use when:** Stress-relaxation with instantaneous modulus and fractional retardation
**Parameters:** 4 (Ge, Gk, α, τ)
**Key equation:** :math:`G(t) = G_{\mathrm{eq}} + (G_e - G_{\mathrm{eq}}) E_{\alpha}(-(t/\tau)^{\alpha})` where :math:`G_{\mathrm{eq}} = \frac{G_e G_k}{G_e + G_k}`
**Test modes:** Relaxation, creep, oscillation
**Material examples:** Viscoelastic solids emphasizing stress-relaxation interpretations

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

Equivalent in form to FKVZ but emphasizing the instantaneous modulus :math:`G_e` in series with a fractional Kelvin-Voigt element. Convenient for stress-relaxation interpretations.

Governing Equations
-------------------

Time domain (creep compliance; same functional form as FKVZ):

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e} \;+\; \frac{1}{G_k}\Big[1 - E_{\alpha}\!\big(-(t/\tau)^{\alpha}\big)\Big].
   \]

Time domain (relaxation modulus; interpolative form):

.. math::
   :nowrap:

   \[
   G(t) \;=\; G_{\mathrm{eq}} \;+\; \big(G_e - G_{\mathrm{eq}}\big)\,
   E_{\alpha}\!\left(-\left(\frac{t}{\tau}\right)^{\alpha}\right),
   \quad G_{\mathrm{eq}} \equiv \frac{G_e G_k}{G_e + G_k}.
   \]

Frequency domain (via complex compliance):

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
     - Instantaneous modulus
   * - ``Gk``
     - :math:`G_k`
     - Pa
     - [1e-3, 1e9]
     - Retarded modulus
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

- Instantaneous response :math:`G(0)=G_e`; relaxes toward :math:`G_{\mathrm{eq}}`.
- Fractional retardation governs the relaxation tail (broad spectra).

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Poynting-Thomson (Zener) behavior.
- :math:`G_k \to \infty`: :math:`G(t)\to G_e` (no retardation).

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalPoyntingThomson`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalPoyntingThomson

   model = FractionalPoyntingThomson()
   model.parameters.set_value('Ge', 1.5e3)
   model.parameters.set_value('Gk', 5e2)
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('tau', 1.0)

   # See :class:`rheojax.models.FractionalPoyntingThomson`

See also
--------

- :doc:`fractional_kv_zener` — identical topology in creep form, sharing the same
  compliance expression.
- :doc:`fractional_zener_ss` — adds an additional spring for solids with higher plateaus.
- :doc:`fractional_burgers` — combines Kelvin and Maxwell branches for more complex
  retardation spectra.
- :doc:`../../transforms/fft` — necessary to obtain :math:`G'(\omega)` before fitting
  relaxation data.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — demonstrates how
  Poynting–Thomson fits compare to other fractional elements.

References
----------

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- J.H. Poynting and J.J. Thomson, *The Properties of Matter*, Charles Griffin (1902).
- R.L. Bagley and P.J. Torvik, "A theoretical basis for the application of fractional
  calculus to viscoelasticity," *J. Rheol.* 27, 201–210 (1983).
