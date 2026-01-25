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

.. include:: /_includes/fractional_seealso.rst

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`G_e`
     - Pa
     - Equilibrium modulus (spring stiffness)
   * - :math:`c_\alpha`
     - Pa·s\ :sup:`α`
     - SpringPot quasi-property (damping coefficient)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < α < 1, controls damping character)
   * - :math:`\tau_\varepsilon`
     - s
     - Characteristic retardation time, :math:`\tau_\varepsilon = (c_\alpha/G_e)^{1/\alpha}`
   * - :math:`E_\alpha(z)`
     - dimensionless
     - One-parameter Mittag-Leffler function
   * - :math:`\Gamma(z)`
     - dimensionless
     - Gamma function

Overview
--------

The Fractional Kelvin-Voigt (FKV) model consists of a Hookean spring and a SpringPot element connected in parallel. This configuration describes materials that exhibit solid-like behavior with power-law creep and viscoelastic damping. Unlike the classical Kelvin-Voigt model which combines a spring and dashpot in parallel, the FKV model replaces the dashpot with a SpringPot, introducing fractional-order power-law damping instead of purely viscous dissipation.

The FKV model is particularly effective for characterizing soft solids, filled polymers, biological tissues, and materials that exhibit bounded creep compliance-materials that deform under constant stress but reach an equilibrium strain rather than flowing indefinitely. The fractional order alpha controls the rate and character of this creep process.

Physical Foundations
--------------------

The FKV model represents the simplest fractional viscoelastic solid, consisting of:

**Mechanical Configuration:**

.. code-block:: text

   [Spring Ge] ---- parallel ---- [SpringPot (c_α, α)]

**Microstructural Interpretation:**

- **Spring (Ge)**: Permanent network structure (crosslinks, crystalline domains)
  providing equilibrium elasticity
- **SpringPot (c_α, α)**: Distributed viscoelastic damping from hierarchical
  relaxation processes (chain rearrangements, bond breaking/reformation)
- **Solid behavior**: Bounded creep to equilibrium compliance J∞ = 1/Ge

The parallel configuration ensures that stress is shared between elastic and
viscoelastic components, with the spring providing long-term load-bearing capacity.

What You Can Learn
------------------

This section explains how to extract material insights from fitted FKV parameters.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Equilibrium Modulus (Ge)**:
   The long-time elastic plateau representing permanent network structure.

   - **For graduate students**: Ge relates to crosslink density via rubber
     elasticity theory: Ge ≈ νkBT where ν is network strand density
   - **For practitioners**: Higher Ge means stiffer material; compare to
     design requirements

**SpringPot Constant (c_α)**:
   Controls the magnitude of viscoelastic damping.

   - High c_α/Ge ratio: Strong damping, slow approach to equilibrium
   - Low c_α/Ge ratio: Weak damping, rapid approach to equilibrium
   - Units: Pa·s^α (unusual due to fractional calculus)

**Fractional Order (α)**:
   Governs the character of power-law damping and spectrum breadth.

   - **α → 0**: Purely elastic (spring-like), minimal damping
   - **α → 0.3-0.5**: Typical for soft solids, broad relaxation spectrum
   - **α → 0.7-0.9**: Approaching classical Kelvin-Voigt (viscous damping)
   - **α → 1**: Classical Kelvin-Voigt with Newtonian dashpot

   *Physical meaning*: Lower α indicates broader distribution of relaxation
   times arising from structural heterogeneity (polydispersity, filler
   distribution, network inhomogeneity).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: FKV Behavior Classification
   :header-rows: 1
   :widths: 20 25 25 30

   * - Parameter Pattern
     - Material Type
     - Examples
     - Key Characteristics
   * - High Ge (> 10⁵ Pa), low α
     - Stiff crosslinked solid
     - Thermosets, vulcanized rubber
     - Minimal creep, strong damping
   * - Moderate Ge (10³-10⁵ Pa), α ~ 0.4
     - Soft viscoelastic solid
     - Hydrogels, elastomers
     - Balanced elasticity/damping
   * - Low Ge (< 10³ Pa), high α
     - Very soft gel
     - Weak physical gels
     - Significant creep, slow recovery

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **Ge near lower bound**: Material may be liquid-like; consider fractional
  Maxwell gel instead
- **α near 1**: Data supports classical Kelvin-Voigt; use simpler model
- **Poor fit at long time**: Equilibrium not reached; extend measurement time
- **c_α and α strongly correlated**: Need broader frequency/time coverage

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Creep test**: 3-4 decades in time, verify plateau at long times
2. **Frequency sweep**: 3-4 decades, strain within LVR (< 5%)
3. **Temperature control**: ±0.1°C for soft materials

**Initialization Strategy:**

.. code-block:: text

   # From creep compliance J(t)
   Ge_init = 1 / J(t → ∞)  # Equilibrium compliance
   c_alpha_init = Ge_init / (characteristic_time**alpha_init)
   alpha_init = 0.5  # Default for soft solids

   # From frequency sweep G'(ω), G"(ω)
   Ge_init = G'(ω → 0)  # Low-frequency plateau
   alpha_init = slope of log(G") vs log(ω) in power-law regime

**Optimization Tips:**

- Fit in compliance space for creep data (more natural)
- Use frequency-domain fitting for SAOS data
- Constrain 0.05 < α < 0.95 to avoid numerical issues
- Verify residuals show no systematic trends

See Also
--------

- :doc:`fractional_maxwell_gel` — provides the series counterpart used for gel-like
  liquids
- :doc:`fractional_kv_zener` — Kelvin-Voigt element combined with an extra spring for
  plateau control
- :doc:`../flow/bingham` — combine bounded creep solids with yield-stress flow models
- :doc:`../../transforms/mutation_number` — monitor whether the quasi-solid assumption
  holds during gelation
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing
  Kelvin-Voigt, Maxwell, and Zener fractional families

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

.. [1] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [2] Makris, N., and Constantinou, M. C. "Fractional-derivative Maxwell model for
   viscous dampers." *Journal of Structural Engineering*, 117, 2708–2724 (1991).
   https://doi.org/10.1061/%28ASCE%290733-9445%281991%29117:9%282708%29

.. [3] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [4] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [5] Friedrich, C. "Relaxation and retardation functions of the Maxwell model
   with fractional derivatives." *Rheologica Acta*, 30, 151–158 (1991).
   https://doi.org/10.1007/BF01134604
.. [6] Metzler, R., Schick, W., Kilian, H.-G., & Nonnenmacher, T. F. "Relaxation in filled polymers: A fractional calculus approach."
   *Journal of Chemical Physics*, **103**, 7180-7186 (1995).
   https://doi.org/10.1063/1.470346

.. [7] Friedrich, C. "Relaxation and retardation functions of the Maxwell model with fractional derivatives."
   *Rheologica Acta*, **30**, 151-158 (1991).
   https://doi.org/10.1007/BF01134604

.. [8] Heymans, N. & Bauwens, J. C. "Fractal rheological models and fractional differential equations for viscoelastic behavior."
   *Rheologica Acta*, **33**, 210-219 (1994).
   https://doi.org/10.1007/BF00437306

.. [9] Nonnenmacher, T. F. & Glöckle, W. G. "A fractional model for mechanical stress relaxation."
   *Philosophical Magazine Letters*, **64**, 89-93 (1991).
   https://doi.org/10.1080/09500839108214672

.. [10] Podlubny, I. *Fractional Differential Equations*.
   Academic Press (1999). ISBN: 978-0125588409

