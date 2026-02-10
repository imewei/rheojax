.. _model-fractional-maxwell-model:

Generalized Fractional Maxwell (Two-Order)
==========================================

Quick Reference
---------------

- **Use when:** Multi-scale relaxation, hierarchical structures requiring two fractional orders
- **Parameters:** 4 (:math:`c_1, \alpha, \beta, \tau`)
- **Key equation:** :math:`G^*(\omega) = c_1 \frac{(i\omega)^\alpha}{1 + (i\omega\tau)^\beta}`
- **Test modes:** Oscillation, relaxation, creep
- **Material examples:** Materials with hierarchical structures, multi-scale relaxation processes

.. include:: /_includes/fractional_seealso.rst

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

Notation Guide
--------------

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`c_1`
     - Material constant (sets modulus scale)
     - Pa·s\ :math:`^{\alpha}`
   * - :math:`\alpha`
     - First fractional order (high-frequency power-law slope)
     - —
   * - :math:`\beta`
     - Second fractional order (transition behavior)
     - —
   * - :math:`\tau`
     - Characteristic relaxation time (crossover frequency)
     - s
   * - :math:`E_\alpha(z)`
     - One-parameter Mittag-Leffler function
     - —
   * - :math:`G^*(ω)`
     - Complex modulus
     - Pa
   * - :math:`G'(ω)`
     - Storage modulus
     - Pa
   * - :math:`G''(ω)`
     - Loss modulus
     - Pa
   * - :math:`\omega`
     - Angular frequency
     - rad/s
   * - :math:`t`
     - Time
     - s

Physical Foundations
--------------------

The generalized Fractional Maxwell Model extends the canonical single-order formulation by incorporating two independent SpringPot elements in series, each characterized by its own fractional order. This enables the model to capture complex hierarchical relaxation processes that cannot be described by a single power-law exponent.

**Mechanical Analogue:**

.. code-block:: text

   [SpringPot (α)] ---- series ---- [SpringPot (β)]

The first SpringPot (:math:`\alpha`) dominates at high frequencies, while the second (:math:`\beta`) controls the transition to low-frequency behavior. The combined effect produces a low-frequency slope of (:math:`\alpha+\beta`).

**Microstructural Interpretation:**

- **First SpringPot (** :math:`\alpha` **)**: Fast relaxation modes from local chain dynamics, segmental motion, or small-scale network rearrangements
- **Second SpringPot (** :math:`\beta` **)**: Slow relaxation modes from large-scale structural relaxation, cooperative motion, or hierarchical network dynamics
- **Combined behavior**: Multi-scale relaxation spectrum with two distinct power-law regimes separated by characteristic time :math:`\tau`

**Connection to Molecular Weight Distribution:**

For polymer melts and solutions, the two fractional orders can capture:

1. :math:`\alpha`: Reflects the breadth of high-frequency modes (entanglement dynamics, chain stretching)
2. :math:`\beta`: Captures low-frequency modes (reptation, constraint release, branching relaxation)
3. **Dual power-law**: Arises naturally from bimodal or hierarchical molecular weight distributions

This model is particularly suited for:

- Polymer blends with distinct component relaxation times
- Branched polymers with arm retraction and backbone relaxation
- Filled systems with matrix and filler-interface contributions
- Associative polymers with multiple bonding timescales

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

What You Can Learn
------------------

This section explains what insights you can extract from fitting the Generalized Fractional Maxwell Model to your experimental data, emphasizing the multi-scale relaxation processes enabled by two independent fractional orders.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Material Constant (** :math:`c_1` **)**:
   Sets the overall magnitude of the viscoelastic response. Higher :math:`c_1` indicates stiffer material behavior.

   *For graduate students*: :math:`c_1` has unusual units (Pa·s\ :math:`^{\alpha}`) due to the fractional calculus framework. It relates to the spectral strength at the highest frequencies measured.
   *For practitioners*: :math:`c_1` scales the entire modulus curve vertically; compare to target stiffness specifications.

**First Fractional Order (** :math:`\alpha` **)**:
   Controls the high-frequency power-law slope. Values closer to 0 indicate more solid-like response at short times, while values closer to 1 indicate more liquid-like (viscous) behavior.

   - :math:`\alpha` **→ 0**: Nearly elastic at short times, very broad relaxation spectrum
   - :math:`\alpha` **→ 0.5**: Balanced solid-liquid character, critical gel behavior
   - :math:`\alpha` **→ 1**: Viscous-like, narrower spectrum

   *For graduate students*: :math:`\alpha` quantifies the polydispersity of the fast relaxation modes in the material's microstructure.
   *For practitioners*: Lower :math:`\alpha` means more complex short-time behavior (important for impact loading).

**Second Fractional Order (** :math:`\beta` **)**:
   Governs the transition behavior and low-frequency power-law slope. Together with :math:`\alpha`, determines the total low-frequency slope (:math:`\alpha+\beta`).

   - :math:`\beta` **→ 0**: Sharp transition, elastic-like at long times
   - :math:`\beta` **→ 0.5**: Gradual transition
   - :math:`\beta` **→ 1**: Viscous-like at long times

   *For graduate students*: :math:`\beta` captures slow relaxation mechanisms (network rearrangements, large-scale structural relaxation).
   *For practitioners*: Higher :math:`\beta` indicates more liquid-like long-time response.

**Relaxation Time (** :math:`\tau` **)**:
   Characteristic timescale separating the two power-law regimes. Marks the crossover frequency :math:`\omega \approx 1/\tau`.

   *For graduate students*: :math:`\tau` is temperature-dependent via WLF or Arrhenius, enabling time-temperature superposition.
   *For practitioners*: Compare :math:`\tau` to service timescales to predict whether material will exhibit regime 1 (:math:`\alpha-dominated`) or regime 2 (:math:`\alpha+\beta-dominated`) behavior.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Two-Order Fractional Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - :math:`\alpha \approx \beta` ≈ 0.5
     - Single-scale fractional
     - Use simpler FMG/FML
     - Unnecessary complexity
   * - :math:`\alpha < 0.3, \beta` > 0.6
     - Hierarchical relaxation
     - Polymer blends, composites
     - Multi-timescale processing needed
   * - :math:`\alpha > 0.7, \beta` < 0.3
     - Viscous with elastic memory
     - Solutions with entanglements
     - Flow-dominated but elastic recoil
   * - :math:`|\alpha - \beta|` > 0.4
     - Two-scale structure
     - Filled polymers, micellar
     - Distinct fast/slow mechanisms

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS): 4-5 decades minimum to capture dual power-law regimes
2. **Test amplitude**: Within LVR (< 5% strain typically)
3. **Coverage**: Ensure both high and low frequency power-law regions visible
4. **Temperature**: Constant ±0.1°C

**Initialization Strategy:**

.. code-block:: python

   # From frequency sweep showing two power-law regimes
   alpha_init = slope at high frequencies
   beta_init = (slope at low frequencies) - alpha_init
   c1_init = magnitude in high-frequency region
   tau_init = 1 / (crossover frequency between regimes)

**Optimization Tips:**

- This is a complex 4-parameter model; ensure data quality is high
- Fit simultaneously to G' and G" with equal weighting
- Use log-weighted least squares for better conditioning
- Verify that both power-law regimes are clearly visible in data
- Compare to simpler models (FMG, FML) using AIC/BIC criteria

**When to Use:**

- Only when simpler fractional models (FMG, FML, FZSS) show systematic deviations
- When log-log plots clearly show two distinct power-law slopes
- For materials with hierarchical structures or multi-scale relaxation

**Common Pitfalls:**

- **Overfitting**: Too many parameters for limited data; verify with cross-validation
- **Parameter correlation**: :math:`\alpha and \beta` may be poorly constrained; report confidence intervals
- **Insufficient data range**: Need 4+ decades to resolve both power-law regimes

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
   omega_exp = np.logspace(-2, 3, 100)
   G_star_exp = load_experimental_data()
   model.fit(omega_exp, G_star_exp, test_mode='oscillation')

   # Compare to simpler models
   from rheojax.models import FractionalMaxwellGel
   fmg = FractionalMaxwellGel()
   fmg.fit(omega_exp, G_star_exp, test_mode='oscillation')
   # Use AIC to compare: lower is better

For more details on the :class:`rheojax.models.FractionalMaxwellModel` class, see the
:doc:`API reference </api/models>`.

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

See Also
--------

Related Models
~~~~~~~~~~~~~~

- :doc:`fractional_maxwell_gel` — canonical single-order form (SpringPot + dashpot) for gels with terminal flow
- :doc:`fractional_maxwell_liquid` — canonical single-order form (spring + SpringPot) for viscoelastic liquids
- :doc:`fractional_burgers` — adds a Kelvin element to capture additional retardation and creep compliance
- :doc:`../classical/maxwell` — classical limit (:math:`\alpha = \beta` = 1, exponential relaxation)
- :doc:`../classical/springpot` — fundamental SpringPot element theory

Related Concepts
~~~~~~~~~~~~~~~~

- :doc:`../flow/carreau` — complementary shear-thinning law for steady-flow data
- :doc:`../../transforms/mastercurve` — merge multi-temperature spectra before fitting to extract time-temperature behavior

Examples and Guides
~~~~~~~~~~~~~~~~~~~

- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing all fractional families on synthetic and experimental data
- :doc:`../../examples/model-comparison/01-fractional-family` — systematic comparison using AIC/BIC criteria
- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing between single-order and two-order models

References
----------

.. [1] Schiessel, H., and Blumen, A. "Hierarchical analogues to fractional relaxation
   equations." *Journal of Physics A*, 26, 5057–5069 (1993).
   https://doi.org/10.1088/0305-4470/26/19/034

.. [2] Heymans, N., and Bauwens, J. C. "Fractal rheological models and fractional
   differential equations for viscoelastic behavior."
   *Rheologica Acta*, 33, 210–219 (1994).
   https://doi.org/10.1007/BF00437306

.. [3] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [4] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [5] Magin, R. L. *Fractional Calculus in Bioengineering*. Begell House (2006).
   ISBN: 978-1567002157

.. [6] Metzler, R., and Nonnenmacher, T. F. "Fractional relaxation processes and
   fractional rheological models for the description of a class of viscoelastic
   materials." *International Journal of Plasticity*, 19, 941–959 (2003).
   https://doi.org/10.1016/S0749-6419(02)00087-6

.. [7] Friedrich, C. "Relaxation and retardation functions of the Maxwell model
   with fractional derivatives." *Rheologica Acta*, 30, 151–158 (1991).
   https://doi.org/10.1007/BF01134604

.. [8] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application
   of fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [9] Gorenflo, R., Kilbas, A. A., Mainardi, F., and Rogosin, S. V.
   *Mittag-Leffler Functions, Related Topics and Applications*. Springer (2014).
   https://doi.org/10.1007/978-3-662-43930-2

.. [10] Jaishankar, A., and McKinley, G. H. "A fractional K-BKZ constitutive formulation
    for describing the nonlinear rheology of multiscale complex fluids."
    *Journal of Rheology*, 58, 1751–1788 (2014).
    https://doi.org/10.1122/1.4892114
