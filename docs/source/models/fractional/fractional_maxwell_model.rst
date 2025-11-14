.. _model-fractional-maxwell-model:

Generalized Fractional Maxwell (Two-Order)
==========================================

Quick Reference
---------------

**Use when:** Multi-scale relaxation, hierarchical structures requiring two fractional orders
**Parameters:** 4 (c₁, α, β, τ)
**Key equation:** :math:`G^*(\omega) = c_1 \frac{(i\omega)^\alpha}{1 + (i\omega\tau)^\beta}`
**Test modes:** Oscillation, relaxation
**Material examples:** Materials with hierarchical structures, multi-scale relaxation processes

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

.. important::

   In the rheology literature "Fractional Maxwell" usually denotes a single fractional
   order :math:`\alpha`. RheoJAX implements those canonical forms as
   :class:`rheojax.models.FractionalMaxwellGel` (springpot + dashpot) and
   :class:`rheojax.models.FractionalMaxwellLiquid` (spring + springpot). The class
   documented here is the *generalized* variant containing two SpringPots in series with
   independent orders :math:`\alpha` and :math:`\beta` for maximum flexibility.

The generalized model represents the most expressive formulation of the family. It is
useful for materials exhibiting multi-scale relaxation processes or hierarchical
structures where different fractional orders govern distinct time/frequency bands.

Governing Equations
-------------------

The constitutive relationships for the Fractional Maxwell Model are:

**Relaxation Modulus**:


.. math::

   G(t) = c_1 t^{-\alpha} E_{1-\alpha}\left(-\left(\frac{t}{\tau}\right)^\beta\right)

where :math:`c_1` is the material constant, :math:`\tau` is the characteristic relaxation time, and :math:`E_\alpha(z)` is the one-parameter Mittag-Leffler function:


.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

**Complex Modulus**:


.. math::

   G^*(\omega) = c_1 \frac{(i\omega)^\alpha}{1 + (i\omega\tau)^\beta}

**Creep Compliance** (approximate form):


.. math::

   J(t) \approx \frac{1}{c_1} t^\alpha E_{\alpha,1+\alpha}\left(\left(\frac{t}{\tau}\right)^\beta\right)

where :math:`E_{\alpha,\beta}(z)` is the two-parameter Mittag-Leffler function:


.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

The presence of two independent fractional orders (alpha and beta) enables the model to capture asymmetric relaxation behavior and multiple power-law regimes.

Parameters
----------

The generalized model is characterized by four parameters:

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
     - Material constant
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

The Fractional Maxwell Model exhibits three distinct regimes:

**Short-Time / High-Frequency Regime** (:math:`\omega\tau \gg 1`):
   Dominated by the first SpringPot (alpha):


   .. math::

      G^*(\omega) \sim c_1 (i\omega)^\alpha

   This gives a power-law with slope alpha in log-log plots of G' and G'' versus omega.

**Long-Time / Low-Frequency Regime** (:math:`\omega\tau \ll 1`):
   Controlled by the second SpringPot (beta) with modified scaling:


   .. math::

      G^*(\omega) \sim c_1 (i\omega)^\alpha \tau^{-\beta} (i\omega)^\beta = c_1 \tau^{-\beta} (i\omega)^{\alpha+\beta}

   This produces a power-law with slope (alpha+beta).

**Intermediate Regime** (:math:`\omega\tau \sim 1`):
   The Mittag-Leffler function provides a smooth crossover between the two power-law regimes. The shape of this transition region depends on the difference :math:`|\alpha - \beta|`, with larger differences producing more gradual transitions.

**Special Cases**:
   - When alpha = beta, the model exhibits a single power-law regime at high frequencies and another at low frequencies with identical slope
   - When alpha -> 1 and beta -> 0, the model approaches classical Maxwell behavior
   - When alpha = beta = 0.5, symmetric fractional behavior emerges

Limiting Behavior
-----------------

The two-order form encompasses several simpler models as limiting cases:

- **alpha = beta**: Simplifies but maintains two distinct regimes with matched power-law exponents
- **alpha -> 1, beta -> 0**: Recovers classical Maxwell model with exponential relaxation
- **alpha = 1, beta = 1**: Approaches Newtonian viscous behavior
- **alpha -> 0**: Elastic-like behavior at short times
- **beta -> 0**: Elastic-like behavior at long times
- **tau -> 0**: Pure SpringPot with exponent alpha
- **tau -> inf**: Pure SpringPot with exponent (alpha+beta) at low frequencies

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalMaxwellModel`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalMaxwellModel
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalMaxwellModel()

   # Set parameters for a multi-scale viscoelastic material
   model.parameters.set_value('c1', 1e5)      # Pa*s^alpha
   model.parameters.set_value('alpha', 0.5)   # dimensionless
   model.parameters.set_value('beta', 0.7)    # dimensionless
   model.parameters.set_value('tau', 1.0)     # s

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 50)
   data = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data.metadata['test_mode'] = 'relaxation'
   G_t = model.predict(data)

   # Predict complex modulus showing dual power-law regimes
   omega = np.logspace(-2, 3, 100)
   data_freq = RheoData(x=omega, y=np.zeros_like(omega), domain='frequency')
   data_freq.metadata['test_mode'] = 'oscillation'
   G_star = model.predict(data_freq)

   # Analyze regime transitions
   Gp = G_star.y.real
   Gpp = G_star.y.imag
   # At omega << 1/tau: slope approx (alpha+beta)
   # At omega >> 1/tau: slope approx alpha

   # Fit to experimental data with two power-law regimes
   # omega_exp, G_star_exp = load_experimental_data()
   # model.fit(omega_exp, G_star_exp, test_mode='oscillation')

For more details on the :class:`rheojax.models.FractionalMaxwellModel` class, see the :doc:`API reference </api/models>`.

See also
--------

- :doc:`fractional_maxwell_gel` and :doc:`fractional_maxwell_liquid` — canonical
  single-order forms used throughout the literature.
- :doc:`fractional_burgers` — adds a Kelvin element to capture additional retardation.
- :doc:`../flow/carreau` — complementary shear-thinning law for steady-flow data.
- :doc:`../../transforms/mastercurve` — merge multi-temperature spectra before fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing all
  fractional families on synthetic and experimental data.

References
----------

- H. Schiessel and A. Blumen, "Hierarchical analogues to fractional relaxation
  equations," *J. Phys. A* 26, 5057–5069 (1993).
- N. Heymans and J.C. Bauwens, "Fractal rheological models and fractional differential
  equations for viscoelastic behavior," *Rheol. Acta* 33, 210–219 (1994).
- H. Schiessel, R. Metzler, A. Blumen, and T.F. Nonnenmacher, "Generalized viscoelastic
  models: their fractional equations with solutions," *J. Phys. A* 28, 6567–6584 (1995).
- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- R.L. Magin, *Fractional Calculus in Bioengineering*, Begell House (2006).
