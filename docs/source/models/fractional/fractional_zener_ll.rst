.. _model-fractional-zener-ll:

Fractional Zener Liquid-Liquid (Fractional)
===========================================

Quick Reference
---------------

**Use when:** Liquid with broad multi-order fractional dispersions, complex viscoelastic behavior
**Parameters:** 6 (c₁, c₂, α, β, γ, τ)
**Key equation:** :math:`G^*(\omega) = \frac{c_1(i\omega)^{\alpha}}{1 + (i\omega\tau)^{\beta}} + c_2(i\omega)^{\gamma}`
**Test modes:** Oscillation, relaxation
**Material examples:** Complex fluids with multiple fractional relaxation mechanisms

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The most general three-element fractional Zener form combining two SpringPots and a viscous time constant. It models liquid-like behavior with broad, multi-order fractional dispersions.

Governing Equations
-------------------

Frequency domain (complex modulus; analytical):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) \;=\;
   \frac{c_1\,(i\omega)^{\alpha}}{1 + (i\omega\tau)^{\beta}} \;+\; c_2\,(i\omega)^{\gamma}.
   \]

Time domain (relaxation modulus; general case):

.. math::
   :nowrap:

   \[
   G(t) \;=\; \mathcal{L}^{-1}\!\left\{G^{*}(s)\right\}(t)
   \;\;\text{which, for distinct orders, involves generalized
   Mittag\text{-}Leffler (Prabhakar) functions } E^{\delta}_{\mu,\nu}(t) .
   \]

Special cases (e.g., :math:`\beta=\alpha`, :math:`c_2=0`) reduce to two-parameter Mittag-Leffler forms.

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
   * - ``c1``
     - :math:`c_1`
     - Pa*s^alpha
     - [1e-3, 1e9]
     - First SpringPot constant
   * - ``c2``
     - :math:`c_2`
     - Pa*s^gamma
     - [1e-3, 1e9]
     - Second SpringPot constant
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - First fractional order
   * - ``beta``
     - :math:`\beta`
     - dimensionless
     - [0, 1]
     - Second fractional order
   * - ``gamma``
     - :math:`\gamma`
     - dimensionless
     - [0, 1]
     - Third fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e6]
     - Relaxation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Liquid-like at low omega (no equilibrium plateau).
- Multiple fractional slopes in :math:`G'` and :math:`G''` controlled by :math:`\alpha,\beta,\gamma`.
- Captures complex crossover patterns not possible with single-order models.

Limiting Behavior
-----------------

- :math:`\alpha,\beta,\gamma \to 1`: tends to classical viscoelastic liquid combinations.
- :math:`c_2 \to 0`: reduces to a generalized fractional Maxwell form.
- Equal orders collapse to two-parameter Mittag-Leffler responses.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalZenerLiquidLiquid`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalZenerLiquidLiquid

   model = FractionalZenerLiquidLiquid()
   model.parameters.set_value('c1', 5e2)
   model.parameters.set_value('c2', 1e2)
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('beta', 0.3)
   model.parameters.set_value('gamma', 0.7)
   model.parameters.set_value('tau', 1.0)

   # See :class:`rheojax.models.FractionalZenerLiquidLiquid`

See also
--------

- :doc:`fractional_zener_sl` and :doc:`fractional_zener_ss` — related variants with solid
  plateaus.
- :doc:`fractional_maxwell_model` — recoverable when one SpringPot order is suppressed.
- :doc:`../flow/carreau` — pair liquid-like viscoelastic spectra with steady-flow fits.
- :doc:`../../transforms/owchirp` — fast acquisition of broadband :math:`G^*` for Zener
  fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — worked comparisons of
  all fractional Zener forms.

References
----------

- R. Garra, R. Gorenflo, F. Polito, and Z.E. Gandjouni, "Mittag-Leffler functions and
  stable probability distributions," *Z. Angew. Math. Phys.* 65, 733–761 (2014).
- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- H. Schiessel, R. Metzler, A. Blumen, and T.F. Nonnenmacher, "Generalized viscoelastic
  models: their fractional equations with solutions," *J. Phys. A* 28, 6567–6584 (1995).
