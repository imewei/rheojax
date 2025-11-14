.. _model-fractional-kelvin-voigt:

Fractional Kelvin-Voigt (Fractional)
====================================

Quick Reference
---------------

**Use when:** Solid with bounded creep, power-law viscoelastic damping
**Parameters:** 2-3 (Ge, c_α, α)
**Key equation:** :math:`G^*(\omega) = G_e + c_\alpha (i\omega)^\alpha`
**Test modes:** Oscillation, creep, relaxation
**Material examples:** Soft solids, filled polymers, biological tissues, materials with bounded compliance

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The Fractional Kelvin-Voigt (FKV) model consists of a Hookean spring and a SpringPot element connected in parallel. This configuration describes materials that exhibit solid-like behavior with power-law creep and viscoelastic damping. Unlike the classical Kelvin-Voigt model which combines a spring and dashpot in parallel, the FKV model replaces the dashpot with a SpringPot, introducing fractional-order power-law damping instead of purely viscous dissipation.

The FKV model is particularly effective for characterizing soft solids, filled polymers, biological tissues, and materials that exhibit bounded creep compliance-materials that deform under constant stress but reach an equilibrium strain rather than flowing indefinitely. The fractional order alpha controls the rate and character of this creep process.

Governing Equations
-------------------

The constitutive behavior of the Fractional Kelvin-Voigt model is described by:

**Relaxation Modulus**:


.. math::

   G(t) = G_e + \frac{c_\alpha t^{-\alpha}}{\Gamma(1-\alpha)}

where :math:`G_e` is the equilibrium modulus, :math:`c_\alpha` is the SpringPot constant, and :math:`\Gamma` is the gamma function. The relaxation modulus consists of an elastic plateau plus a power-law term that decays in time.

**Complex Modulus**:


.. math::

   G^*(\omega) = G_e + c_\alpha (i\omega)^\alpha

This can be decomposed into storage and loss moduli:


.. math::

   G'(\omega) = G_e + c_\alpha \omega^\alpha \cos\left(\frac{\alpha\pi}{2}\right)

.. math::

   G''(\omega) = c_\alpha \omega^\alpha \sin\left(\frac{\alpha\pi}{2}\right)

**Creep Compliance**:


.. math::

   J(t) = \frac{1}{G_e}\left[1 - E_\alpha\left(-\left(\frac{t}{\tau_\varepsilon}\right)^\alpha\right)\right]

where :math:`\tau_\varepsilon = (c_\alpha/G_e)^{1/\alpha}` is the characteristic retardation time and :math:`E_\alpha(z)` is the one-parameter Mittag-Leffler function:


.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

The creep compliance approaches the limiting value :math:`J_\infty = 1/G_e` as :math:`t \to \infty`, confirming solid-like behavior.

Parameters
----------

The Fractional Kelvin-Voigt model has three parameters:

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

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

The Fractional Kelvin-Voigt model exhibits characteristic behavior in different regimes:

**Short-Time / High-Frequency Regime** (:math:`t \ll \tau_\varepsilon` or :math:`\omega \gg \omega_c`):
   Instantaneous elastic response with additional power-law contribution:

   
   .. math::
   
      G(t) \sim G_e + \frac{c_\alpha t^{-\alpha}}{\Gamma(1-\alpha)}, \quad G^*(\omega) \sim G_e + c_\alpha (i\omega)^\alpha

   The material behaves as a stiff solid with frequency-dependent damping.

**Long-Time / Low-Frequency Regime** (:math:`t \gg \tau_\varepsilon` or :math:`\omega \ll \omega_c`):
   Equilibrium elastic plateau:

   
   .. math::
   
      G(t) \to G_e, \quad J(t) \to \frac{1}{G_e}

   The material reaches a constant equilibrium modulus, confirming solid-like behavior without terminal flow.

**Intermediate Regime**:
   The Mittag-Leffler function in the creep compliance produces a smooth power-law transition from initial response to equilibrium. The characteristic frequency :math:`\omega_c \sim 1/\tau_\varepsilon` marks the crossover region where viscoelastic dissipation is most pronounced.

**Loss Tangent**:

   
.. math::

   \tan\delta = \frac{G''}{G'} = \frac{c_\alpha \omega^\alpha \sin(\alpha\pi/2)}{G_e + c_\alpha \omega^\alpha \cos(\alpha\pi/2)}

The loss tangent exhibits a maximum at intermediate frequencies, indicating peak energy dissipation.

Limiting Behavior
-----------------

The FKV model connects to classical models in limiting cases:

- **alpha -> 1**: Approaches classical Kelvin-Voigt model with Newtonian damping: :math:`G^*(\omega) \approx G_e + i\omega c_\alpha`
- **alpha -> 0**: Reduces to purely elastic solid: :math:`G^*(\omega) \to G_e`
- **c\ :sub:`alpha` -> 0**: Pure elastic spring with :math:`G^*(\omega) = G_e`
- **c\ :sub:`alpha` -> inf**: Diverging damping, non-physical limit
- **G\ :sub:`e` -> inf with fixed c\ :sub:`alpha`/G\ :sub:`e`**: Infinite stiffness limit

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalKelvinVoigt`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalKelvinVoigt
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalKelvinVoigt()

   # Set parameters for a filled polymer composite
   model.parameters.set_value('Ge', 1e6)         # Pa
   model.parameters.set_value('c_alpha', 1e4)    # Pa*s^alpha
   model.parameters.set_value('alpha', 0.5)      # dimensionless

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 50)
   data = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data.metadata['test_mode'] = 'relaxation'
   G_t = model.predict(data)

   # Predict creep compliance showing bounded creep
   data_creep = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data_creep.metadata['test_mode'] = 'creep'
   J_t = model.predict(data_creep)
   # J(t->inf) -> 1/Ge (equilibrium compliance)

   # Predict complex modulus in frequency domain
   omega = np.logspace(-2, 2, 50)
   data_freq = RheoData(x=omega, y=np.zeros_like(omega), domain='frequency')
   data_freq.metadata['test_mode'] = 'oscillation'
   G_star = model.predict(data_freq)

   # Extract storage and loss moduli
   Gp = G_star.y.real   # G'(omega) includes elastic plateau
   Gpp = G_star.y.imag  # G''(omega) shows power-law damping
   tan_delta = Gpp / Gp # Peaks at intermediate frequencies

   # Fit to experimental oscillatory data
   # omega_exp, G_star_exp = load_experimental_data()
   # model.fit(omega_exp, G_star_exp, test_mode='oscillation')

For more details on the :class:`rheojax.models.FractionalKelvinVoigt` class, see the
:doc:`API reference </api/models>`.

See also
--------

- :doc:`fractional_maxwell_gel` — provides the series counterpart used for gel-like
  liquids.
- :doc:`fractional_kv_zener` — Kelvin-Voigt element combined with an extra spring for
  plateau control.
- :doc:`../flow/bingham` — combine bounded creep solids with yield-stress flow models.
- :doc:`../../transforms/mutation_number` — monitor whether the quasi-solid assumption
  holds during gelation.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing
  Kelvin-Voigt, Maxwell, and Zener fractional families.

References
----------

- R.L. Bagley and P.J. Torvik, "A theoretical basis for the application of fractional
  calculus to viscoelasticity," *J. Rheol.* 27, 201–210 (1983).
- N. Makris and M.C. Constantinou, "Fractional-derivative Maxwell model for viscous
  dampers," *J. Struct. Eng.* 117, 2708–2724 (1991).
- H. Schiessel, R. Metzler, A. Blumen, and T.F. Nonnenmacher, "Generalized viscoelastic
  models: their fractional equations with solutions," *J. Phys. A* 28, 6567–6584 (1995).
