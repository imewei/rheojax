.. _model-fractional-jeffreys:

Fractional Jeffreys Model (Fractional)
======================================

Quick Reference
---------------

**Use when:** Viscoelastic liquid with fractional relaxation, terminal viscous flow
**Parameters:** 4 (η₁, η₂, α, τ₁)
**Key equation:** :math:`G^*(\omega) = \eta_1(i\omega) \frac{1 + (i\omega\tau_2)^{\alpha}}{1 + (i\omega\tau_1)^{\alpha}}`
**Test modes:** Oscillation, relaxation, rotation (steady shear)
**Material examples:** Polymer solutions with broad relaxation spectra, complex fluids with viscous flow

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

A liquid model consisting of one dashpot in parallel with a series dashpot-SpringPot branch. It exhibits viscous flow with fractional relaxation features.

Governing Equations
-------------------

Time domain (relaxation modulus):

.. math::
   :nowrap:

   \[
   G(t) \;=\; \frac{\eta_1}{\tau_1}\, t^{-\alpha}\,
   E_{1-\alpha,\,1-\alpha}\!\left(-\left(\frac{t}{\tau_1}\right)^{1-\alpha}\right).
   \]

Frequency domain (complex modulus):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) \;=\; \eta_1(i\omega)\,
   \frac{1 + (i\omega\tau_2)^{\alpha}}{1 + (i\omega\tau_1)^{\alpha}},
   \quad \tau_2 = \frac{\eta_2}{\eta_1}\,\tau_1 .
   \]

Steady shear (rotation): Newtonian-like at low rates with viscosity dominated by :math:`\eta_1`.

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
   * - ``eta1``
     - :math:`\eta_1`
     - Pa*s
     - [1e-6, 1e12]
     - First viscosity
   * - ``eta2``
     - :math:`\eta_2`
     - Pa*s
     - [1e-6, 1e12]
     - Second viscosity
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau1``
     - :math:`\tau_1`
     - s
     - [1e-6, 1e6]
     - Relaxation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, oscillation, steady shear.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Low omega: liquid-like; :math:`G' \ll G''\approx \eta_{\mathrm{eff}}\omega`.
- Intermediate: fractional dispersion with order :math:`\alpha`.
- High omega: elastic upturn from branch dynamics.

Limiting Behavior
-----------------

- :math:`\alpha \to 1`: classical Jeffreys model.
- :math:`\eta_2 \to 0`: Maxwell-like liquid dominated by :math:`\eta_1`.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalJeffreysModel`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalJeffreysModel

   model = FractionalJeffreysModel()
   model.parameters.set_value('eta1', 1e3)
   model.parameters.set_value('eta2', 5e2)
   model.parameters.set_value('alpha', 0.5)
   model.parameters.set_value('tau1', 1.0)

   # See :class:`rheojax.models.FractionalJeffreysModel`

See also
--------

- :doc:`fractional_maxwell_liquid` — single-springpot Maxwell analogue that forms one
  branch of the Jeffreys construction.
- :doc:`fractional_kelvin_voigt` — parallel SpringPot + spring element providing the other
  branch.
- :doc:`fractional_burgers` — combines Maxwell and fractional Kelvin-Voigt in series.
- :doc:`../../transforms/fft` — obtain :math:`G'`/ :math:`G''` prior to fitting Jeffreys
  spectra.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing
  Jeffreys, Burgers, and Maxwell families.

References
----------

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- H. Jeffreys, *The Earth*, Cambridge University Press (1929).
- C. Friedrich, "Relaxation and retardation functions of the fractional Jeffreys model,"
  *Rheol. Acta* 30, 151–158 (1991).
