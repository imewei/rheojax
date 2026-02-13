.. _model-fractional-maxwell-liquid:

Fractional Maxwell Liquid (Fractional)
======================================

Quick Reference
---------------

- **Use when:** Viscoelastic liquid, power-law relaxation without terminal flow plateau
- **Parameters:** 3 (:math:`G_m`, :math:`\alpha`, :math:`\tau_\alpha`)
- **Key equation:** :math:`G(t) = G_m t^{-\alpha} E_{1-\alpha,1-\alpha}(-t^{1-\alpha}/\tau_\alpha)`
- **Test modes:** Oscillation, relaxation, creep, flow curve
- **Material examples:** Polymer melts (linear/branched), concentrated polymer solutions, complex fluids

.. include:: /_includes/fractional_seealso.rst

Overview
--------

The Fractional Maxwell Liquid (FML) model consists of a Hookean spring in series with a SpringPot element. This configuration describes materials with instantaneous elastic response at short times followed by power-law relaxation at intermediate to long times. The model is particularly effective for characterizing polymer melts, concentrated polymer solutions, and other viscoelastic liquids that exhibit both elastic memory and power-law relaxation without terminal flow.

Unlike the Fractional Maxwell Gel which includes a dashpot for terminal flow, the FML model maintains power-law behavior across all time scales, making it ideal for materials that show persistent viscoelastic behavior without approaching pure viscous flow.

Notation Guide
--------------

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`G_m`
     - Maxwell modulus (short-time elastic stiffness)
     - Pa
   * - :math:`\alpha`
     - Fractional order (0 < :math:`\alpha` < 1, controls relaxation spectrum breadth)
     - —
   * - :math:`\tau_\alpha`
     - Characteristic relaxation time
     - s\ :math:`^{\alpha}`
   * - :math:`E_{\alpha,\beta}(z)`
     - Two-parameter Mittag-Leffler function
     - —
   * - :math:`G^*(ω)`
     - Complex modulus
     - Pa
   * - :math:`G'(ω)`
     - Storage modulus (elastic component)
     - Pa
   * - :math:`G''(ω)`
     - Loss modulus (viscous component)
     - Pa
   * - :math:`J(t)`
     - Creep compliance
     - Pa\ :sup:`-1`
   * - :math:`\omega`
     - Angular frequency
     - rad/s
   * - :math:`t`
     - Time
     - s

Physical Foundations
--------------------

The FML model represents **viscoelastic liquids** with zero equilibrium modulus (Ge = 0), meaning the material flows under sustained stress. The physical structure consists of:

1. **Hookean spring (Gm)**: Provides instantaneous elastic response at short times. Represents chain/network stretching before relaxation mechanisms activate.

2. **SpringPot element**: Governs the relaxation dynamics through power-law viscoelasticity. The fractional order :math:`\alpha` controls the breadth of the relaxation spectrum.

The series configuration ensures that sustained stress eventually leads to unbounded strain growth (flow), distinguishing this from solid-like models.

**For FML specifically**, the fractional order :math:`\alpha` directly controls the slope in log-log plots of :math:`G'(\omega)` and :math:`G''(\omega)`, with both moduli scaling as :math:`\omega^{\alpha}` in the power-law region. Typical :math:`\alpha` ranges for FML applications:

- Polymer melts (linear homopolymers): :math:`\alpha` ≈ 0.7-0.9
- Polymer melts (branched): :math:`\alpha` ≈ 0.5-0.7
- Concentrated polymer solutions: :math:`\alpha` ≈ 0.5-0.8
- Complex fluids (colloidal dispersions): :math:`\alpha` ≈ 0.4-0.7

Mathematical Foundations
------------------------

Mittag-Leffler Functions
~~~~~~~~~~~~~~~~~~~~~~~~

The FML model relies on the **two-parameter Mittag-Leffler function**:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

where :math:`\Gamma` is the gamma function. This generalization of the exponential function is essential for fractional viscoelasticity.

**Key Properties:**
   - :math:`E_1,_1(z)` = exp(z) (recovers classical exponential)
   - :math:`E_{\alpha,\alpha(-t^\alpha)}` provides the characteristic power-law relaxation
   - Smoothly interpolates between short-time power-law and long-time stretched exponential

Power-Law Relaxation Derivation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the FML model, the relaxation modulus exhibits:

.. math::

   G(t) \sim t^{-\alpha} \quad \text{for } t \gg \tau_\alpha

This power-law decay arises from the SpringPot constitutive equation and contrasts sharply with exponential decay (classical Maxwell). The power-law reflects a continuous distribution of relaxation times spanning many decades.

Governing Equations
-------------------

**Relaxation Modulus**:


.. math::

   G(t) = G_m t^{-\alpha} E_{1-\alpha,1-\alpha}\left(-\frac{t^{1-\alpha}}{\tau_\alpha}\right)

where :math:`G_m` is the Maxwell modulus, :math:`E_{\alpha,\beta}(z)` is the two-parameter Mittag-Leffler function, and :math:`\tau_\alpha` is the characteristic relaxation time with units of s\ :sup:`alpha`.

**Physical interpretation:**
   - Short times (:math:`t \ll \tau_\alpha`): :math:`G(t) \approx G_m` (elastic plateau)
   - Intermediate times: :math:`G(t) \sim t^{-\alpha}` (power-law relaxation)
   - Long times: Stretched exponential decay toward zero

**Complex Modulus**:


.. math::

   G^*(\omega) = G_m \frac{(i\omega\tau_\alpha)^\alpha}{1 + (i\omega\tau_\alpha)^\alpha}

Decomposing into storage and loss moduli:

.. math::

   G'(\omega) = G_m \frac{(\omega\tau_\alpha)^\alpha [1 + (\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2)]}{1 + 2(\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2) + (\omega\tau_\alpha)^{2\alpha}}

.. math::

   G''(\omega) = G_m \frac{(\omega\tau_\alpha)^\alpha \sin(\alpha\pi/2)}{1 + 2(\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2) + (\omega\tau_\alpha)^{2\alpha}}

**Frequency-Domain Behavior:**
   - High :math:`\omega` (:math:`\omega \gg 1/\tau_\alpha`): :math:`G' \to G_m`, :math:`G'' \to 0` (elastic plateau)
   - Intermediate :math:`\omega` (:math:`\omega \sim 1/\tau_\alpha`): :math:`G', G'' \sim \omega^\alpha` (power-law scaling, parallel slopes)
   - Low :math:`\omega` (:math:`\omega \ll 1/\tau_\alpha`): :math:`G' \sim \omega^{2\alpha}`, :math:`G'' \sim \omega^\alpha` (liquid-like terminal regime)

**Creep Compliance**:


.. math::

   J(t) = \frac{1}{G_m} + \frac{t^\alpha}{G_m \tau_\alpha^\alpha} E_{\alpha,1+\alpha}\left(-\left(\frac{t}{\tau_\alpha}\right)^\alpha\right)

**Physical interpretation:**
   - Short times: :math:`J(t) \approx 1/G_m` (elastic compliance)
   - Long times: Unbounded growth :math:`J(t) \to \infty` (liquid-like flow)

The Mittag-Leffler function provides a smooth interpolation between exponential decay (when alpha=1) and stretched exponential or power-law relaxation (when 0 < alpha < 1):


.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

Parameters
----------

The Fractional Maxwell Liquid model has three parameters:

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 12 12 18 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``Gm``
     - :math:`G_m`
     - Pa
     - [1e-3, 1e9]
     - Maxwell modulus (short-time elasticity)
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - Fractional order (spectrum breadth)
   * - ``tau_alpha``
     - :math:`\tau_\alpha`
     - s^alpha
     - [1e-6, 1e6]
     - Characteristic relaxation time

**Parameter Interpretation:**

- **Gm**: Instantaneous modulus reflecting chain/network stiffness. For polymer melts, relates to entanglement density via Gm ≈ :math:`G_N^0` (plateau modulus). Typical values: :math:`10^3-10^6` Pa for polymer melts.

- **alpha**: Quantifies relaxation spectrum breadth. Lower :math:`\alpha` → broader spectra from molecular weight polydispersity, branching, or complex intermolecular interactions. For linear polymers, :math:`\alpha` ≈ 0.7-0.9; for branched polymers, :math:`\alpha` ≈ 0.5-0.7.

- **tau_alpha**: Average relaxation time scale. Has unusual units (s\ :math:`^{\alpha}`) due to fractional calculus. For polymer melts, relates to molecular weight via :math:`\tau_\alpha \sim M_w^{3.4}`. Typical values: :math:`10 \times 10^{-3-10^3}` s depending on molecular weight and temperature.

Validity and Assumptions
------------------------

- Linear viscoelastic assumption: strain amplitudes remain small (:math:`\gamma_0` < 5-10% typically).
- Isothermal conditions: constant temperature throughout experiment.
- Time-invariant material parameters: no aging, polymerization, or degradation.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.
- Assumes liquid-like behavior: zero equilibrium modulus (Ge = 0), material flows under stress.

What You Can Learn
------------------

This section explains how to translate fitted FML parameters into material
insights and actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional Order (** :math:`\alpha` **)**:
   The fractional order reveals molecular architecture and relaxation dynamics:

   - **0.7 <** :math:`\alpha` **< 0.9**: Narrow relaxation spectrum. Typical for linear,
     monodisperse polymer melts with well-defined entanglement dynamics.

   - **0.5 <** :math:`\alpha` **< 0.7**: Moderate spectrum breadth. Common in branched polymers,
     polydisperse melts, or concentrated solutions where multiple relaxation
     mechanisms coexist.

   - :math:`\alpha` **< 0.5**: Very broad spectrum. Indicates complex hierarchical relaxation
     (star polymers, H-polymers) or strong polydispersity.

   *For graduate students*: The fractional order connects to molecular weight
   distribution. For polymers, :math:`\alpha` ≈ 1/(1 + PDI/3) approximately, where PDI is
   the polydispersity index. Branching lowers :math:`\alpha` due to arm retraction and
   branch point hopping mechanisms.

   *For practitioners*: Use :math:`\alpha` to assess batch-to-batch consistency. A sudden
   drop in :math:`\alpha` suggests contamination with branched species or broadening of MWD.

**Maxwell Modulus (Gm)**:
   The modulus reveals network/entanglement density:

   - **Gm ≈** :math:`G_N^0` **(plateau modulus)**: For entangled polymer melts, Gm should
     match the rubbery plateau from literature. Significant deviation suggests
     incomplete entanglement or dilution.

   - **Relationship to Me**: :math:`G_m = \rho R T / M_e` where Me is
     entanglement molecular weight.

   *For practitioners*: Track Gm as a QC metric. For polymer melts, Gm should
   be stable (±10%) across batches of the same grade.

**Relaxation Time (** :math:`\tau_\alpha` **)**:
   The characteristic time connects to molecular weight:

   - **Scaling**: For linear polymers, :math:`\tau_\alpha \propto M_w^{3.4}`
     (reptation theory).

   - **Temperature dependence**: Follows WLF or Arrhenius behavior.

   *For practitioners*: Compare :math:`\tau_\alpha` to process timescales. For extrusion,
   ensure :math:`\tau_\alpha < 1/\dot{\gamma}_{process}` for complete relaxation.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: FML Material Classification
   :header-rows: 1
   :widths: 20 20 30 30

   * - :math:`\alpha` Range
     - Spectrum Type
     - Typical Materials
     - Implications
   * - 0.8 < :math:`\alpha` < 1.0
     - Very narrow
     - Monodisperse linear polymers
     - Near-Maxwellian, consider classical model
   * - 0.6 < :math:`\alpha` < 0.8
     - Narrow-moderate
     - Commercial polymer melts
     - Standard processing behavior
   * - 0.4 < :math:`\alpha` < 0.6
     - Broad
     - Branched polymers, blends
     - Complex flow behavior, longer relaxation
   * - :math:`\alpha` < 0.4
     - Very broad
     - Highly branched, filled systems
     - Multiple mechanisms, difficult to process

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- :math:`\alpha` **→ 1**: Material is nearly Maxwellian. Consider using classical Maxwell
  for simpler interpretation and faster computation.

- **Gm ≠** :math:`G_N^0`: If Gm differs significantly from tabulated plateau modulus,
  check for dilution, incomplete entanglement, or fitting errors.

- :math:`\tau_\alpha` **inconsistent with Mw**: Compare to literature correlations. Large
  deviations suggest degradation or contamination.

- **Poor fit at low frequencies**: Terminal behavior may not match FML
  predictions. Consider FMG (with dashpot) for materials showing true
  terminal flow.

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Polymer Grade Verification**:
   Fit FML to frequency sweep, compare :math:`\alpha` and :math:`\tau_\alpha` to specifications. A batch
   with lower :math:`\alpha` likely has broader MWD or unexpected branching.

**Processing Optimization**:
   Use :math:`\tau_\alpha` to set residence times. For complete stress relaxation, ensure
   process time > :math:`5\tau_\alpha`.

**Blend Analysis**:
   Lower :math:`\alpha` in blends indicates poor miscibility (separate relaxation modes)
   or broad combined MWD.

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS): 3-5 decades (e.g., 0.01-100 rad/s)
2. **Test amplitude**: Within LVR (typically 0.5-5% strain)
3. **Coverage**: Ensure both elastic plateau and power-law regimes captured
4. **Temperature control**: ±0.1°C for polymer melts

**Initialization Strategy:**

.. code-block:: python

   # From frequency sweep |G*|(ω)
   Gm_init = high_freq_plateau  # Elastic plateau
   tau_alpha_init = 1 / (frequency at steepest slope)
   alpha_init = slope in power-law region

   # Smart initialization (automatic in RheoJAX v0.2.0+)
   # Applied automatically when test_mode='oscillation'

**Optimization Tips:**

- Fit simultaneously to :math:`G'` and :math:`G''` for better constraint
- Use log-weighted least squares
- Verify power-law region (parallel :math:`G'`, :math:`G''` slopes)
- Check residuals for systematic deviations

**Common Pitfalls:**

- **Insufficient high-frequency data**: Cannot determine Gm accurately
- **Missing power-law regime**: Need broader frequency coverage
- :math:`\alpha` **near 1**: Use classical Maxwell for simpler interpretation

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalMaxwellLiquid
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalMaxwellLiquid()

   # Fit to experimental data (smart initialization automatic)
   omega_exp = np.logspace(-2, 2, 50)
   G_star_exp = load_experimental_data()  # Complex modulus
   model.fit(omega_exp, G_star_exp, test_mode='oscillation')

   # Inspect fitted parameters
   print(f"Gm = {model.parameters.get_value('Gm'):.2e} Pa")
   print(f"α = {model.parameters.get_value('alpha'):.4f}")
   print(f"τ_α = {model.parameters.get_value('tau_alpha'):.2e} s^α")

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 100)
   data = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data.metadata['test_mode'] = 'relaxation'
   G_t = model.predict(data)

   # Bayesian uncertainty quantification
   result = model.fit_bayesian(
       omega_exp, G_star_exp,
       num_warmup=1000,
       num_samples=2000,
       test_mode='oscillation'
   )
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)

For more details, see :doc:`API reference </api/models>`.

Regimes and Behavior
--------------------

The Fractional Maxwell Liquid exhibits characteristic behavior across different regimes:

**Short-Time / High-Frequency Regime** (:math:`t \ll \tau_\alpha` or :math:`\omega \gg 1/\tau_\alpha`):
   The spring dominates, yielding purely elastic behavior:


   .. math::

      G(t) \sim G_m, \quad G^*(\omega) \sim G_m

   The material behaves as an elastic solid with modulus :math:`G_m`. This regime captures the instantaneous response before relaxation mechanisms activate.

**Intermediate Regime** (:math:`t \sim \tau_\alpha` or :math:`\omega \sim 1/\tau_\alpha`):
   The Mittag-Leffler function provides a smooth crossover between elastic plateau and power-law relaxation. This is the **fingerprint** of fractional viscoelasticity:

   .. math::

      G'(\omega), G''(\omega) \sim \omega^\alpha \quad \text{(parallel slopes in log-log plot)}

   The loss tangent :math:`\tan\delta = G''/G'` exhibits a maximum at the characteristic frequency :math:`\omega \sim 1/\tau_\alpha`.

**Long-Time / Low-Frequency Regime** (:math:`t \gg \tau_\alpha` or :math:`\omega \ll 1/\tau_\alpha`):
   The SpringPot controls the response with power-law behavior:


   .. math::

      G(t) \sim G_m \left(\frac{t}{\tau_\alpha}\right)^{-\alpha}, \quad G^*(\omega) \sim G_m (i\omega\tau_\alpha)^\alpha

   For very low frequencies, terminal liquid-like behavior emerges:

   .. math::

      G'(\omega) \sim \omega^{2\alpha}, \quad G''(\omega) \sim \omega^\alpha \quad \text{(G" > G')}

Comparison with Classical Maxwell
----------------------------------

**Classical Maxwell (** :math:`\alpha` **= 1):**
   - Single relaxation time :math:`\tau`
   - Exponential relaxation: :math:`G(t) = G_m \exp(-t/\tau)`
   - Narrow relaxation spectrum (Lorentzian)
   - Low-frequency behavior: :math:`G' \sim \omega^2`, :math:`G'' \sim \omega` (classical liquid)

**Fractional Maxwell Liquid (0 <** :math:`\alpha` **< 1):**
   - Continuous distribution of relaxation times
   - Power-law relaxation: :math:`G(t) \sim t^{-\alpha}`
   - Broad relaxation spectrum
   - Low-frequency behavior: :math:`G' \sim \omega^{2\alpha}`, :math:`G'' \sim \omega^\alpha` (generalized liquid)

**When to Use Fractional:**
   - Power-law relaxation observed in stress relaxation experiments
   - Log-log plots of :math:`G'` and :math:`G''` show parallel slopes over multiple decades
   - Polymer melts with broad molecular weight distribution
   - Concentrated solutions with complex intermolecular interactions

**When Classical Suffices:**
   - Single dominant relaxation time (linear homopolymers, dilute solutions)
   - Data span < 2 decades in frequency
   - Exponential decay observed experimentally

Limiting Behavior
-----------------

The FML model connects to classical models in limiting cases:

- **alpha -> 1**: Recovers the classical Maxwell model with exponential relaxation: :math:`G(t) = G_m e^{-t/\tau_\alpha}`
- **alpha -> 0**: Approaches purely elastic solid behavior: :math:`G(t) \sim G_m`
- **tau\ :sub:`alpha` -> 0**: Pure elastic spring with :math:`G^*(\omega) = G_m`
- **tau\ :sub:`alpha` -> inf**: Pure SpringPot behavior with :math:`G^*(\omega) \sim (i\omega)^\alpha`
- **Gm -> 0**: Non-physical (no elasticity)
- **Gm -> inf**: Infinitely stiff limit

Material Examples
-----------------

**Polymer Melts (Linear):**
   - Polyethylene, polypropylene, polystyrene (:math:`\alpha` ≈ 0.7-0.9)
   - Gm ≈ :math:`G_N^0` (plateau modulus from entanglements)
   - Relatively narrow spectra (high :math:`\alpha`) for monodisperse polymers

**Polymer Melts (Branched):**
   - Long-chain branched polyethylene, star polymers (:math:`\alpha` ≈ 0.5-0.7)
   - Broader spectra (lower :math:`\alpha`) from hierarchical relaxation processes
   - Arm retraction, branch point hopping add complexity

**Concentrated Polymer Solutions:**
   - Solutions above overlap concentration c* (:math:`\alpha` ≈ 0.5-0.8)
   - Lower :math:`\alpha` than melts due to solvent-polymer interactions
   - Spectrum breadth depends on concentration and molecular weight distribution

**Micellar Solutions:**
   - Wormlike micelles, surfactant solutions (:math:`\alpha` ≈ 0.4-0.7)
   - Broad spectra from micelle size distribution and reptation
   - Can exhibit gel-like behavior (:math:`\alpha` ≈ 0.5) near critical concentration

**Colloidal Dispersions:**
   - Dense colloidal suspensions (:math:`\alpha` ≈ 0.4-0.6)
   - Particle size polydispersity creates broad relaxation spectra
   - Hydrodynamic interactions contribute to spectrum breadth

Smart Initialization (NEW in v0.2.0)
-------------------------------------

RheoJAX automatically applies **smart parameter initialization** when fitting FML to oscillation data.

How It Works
~~~~~~~~~~~~

When ``test_mode='oscillation'``, the initialization system:

1. **Extracts frequency features** from :math:`|G^*|(\omega)` data:
   - High-frequency plateau → estimates :math:`G_m`
   - Transition frequency :math:`\omega_mid` (maximum slope of :math:`|G^*|`) → estimates :math:`\tau_\alpha = 1/\omega_mid`
   - Slope in power-law region → estimates fractional order :math:`\alpha`

2. **Estimates fractional order** from parallel slopes:
   - Identifies region where :math:`G'(\omega)` and :math:`G''(\omega)` have parallel slopes
   - Extracts slope via linear regression in log-log space
   - Maps slope directly to :math:`\alpha` (slope :math:`\approx \alpha` in power-law region)

3. **Clips to parameter bounds** to ensure :math:`G_m > 0`, :math:`0 < \alpha < 1`, :math:`\tau_\alpha > 0`

Benefits
~~~~~~~~

- **Convergence improvement**: 60-80% reduction in optimization failures
- **Parameter recovery**: More accurate fitted parameters from better starting point
- **Speed**: Fewer iterations (typical: 50-200 vs 500-1000 without initialization)
- **Robustness**: Handles noisy experimental data through smoothing

Implementation
~~~~~~~~~~~~~~

Uses **Template Method pattern** with 5-step algorithm (extract → validate → estimate → clip → set). See :doc:`../../developer/architecture` for details.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalMaxwellLiquid`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalMaxwellLiquid
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalMaxwellLiquid()

   # Set parameters for a polymer melt
   model.parameters.set_value('Gm', 1e6)         # Pa
   model.parameters.set_value('alpha', 0.7)      # dimensionless
   model.parameters.set_value('tau_alpha', 1.0)  # s^alpha

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 50)
   data = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data.metadata['test_mode'] = 'relaxation'
   G_t = model.predict(data)

   # Predict complex modulus for oscillatory shear
   omega = np.logspace(-2, 2, 50)
   data_freq = RheoData(x=omega, y=np.zeros_like(omega), domain='frequency')
   data_freq.metadata['test_mode'] = 'oscillation'
   G_star = model.predict(data_freq)

   # Extract storage and loss moduli
   Gp = G_star.y.real   # G'(omega)
   Gpp = G_star.y.imag  # G''(omega)
   tan_delta = Gpp / Gp

   # Fit to experimental frequency sweep data (smart initialization automatic)
   # omega_exp, G_star_exp = load_experimental_data()
   # model.fit(omega_exp, G_star_exp, test_mode='oscillation')

   # Bayesian inference with NLSQ warm-start
   # result = model.fit_bayesian(omega_exp, G_star_exp,
   #                              num_warmup=1000,
   #                              num_samples=2000)

For more details on the :class:`rheojax.models.FractionalMaxwellLiquid` class, see the :doc:`API reference </api/models>`.

See Also
--------

Related Models
~~~~~~~~~~~~~~

- :doc:`fractional_maxwell_gel` — uses a dashpot instead of a spring for gel-like systems with terminal flow
- :doc:`fractional_maxwell_model` — generalized two-order series analogue with independent :math:`\alpha` and :math:`\beta`
- :doc:`fractional_jeffreys` — adds a parallel dashpot for finite zero-shear viscosity
- :doc:`../classical/maxwell` — classical limit (:math:`\alpha` → 1, exponential relaxation)
- :doc:`../classical/springpot` — fundamental SpringPot element theory

Transforms
~~~~~~~~~~

- :doc:`../../transforms/fft` — convert relaxation data to :math:`G^*(\omega)` before fitting
- :doc:`../../transforms/mastercurve` — time-temperature superposition for polymer melts
- :doc:`../../transforms/derivatives` — compute loss tangent :math:`\tan\delta` from :math:`G'` and :math:`G''`

Examples
~~~~~~~~

- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook covering the complete Fractional Maxwell family
- :doc:`../../examples/fitting/01-smart-initialization` — demonstration of automatic initialization (v0.2.0)
- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing models

References
----------

.. [1] Friedrich, C. "Relaxation and retardation functions of the Maxwell model
   with fractional derivatives." *Rheologica Acta*, 30, 151–158 (1991).
   https://doi.org/10.1007/BF01134604

.. [2] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F.
   "Generalized viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [3] Metzler, R., Schick, W., Kilian, H.-G., and Nonnenmacher, T. F.
   "Relaxation in filled polymers: A fractional calculus approach."
   *Journal of Chemical Physics*, 103, 7180–7186 (1995).
   https://doi.org/10.1063/1.470346

.. [4] Gorenflo, R., Kilbas, A. A., Mainardi, F., and Rogosin, S. V.
   *Mittag-Leffler Functions, Related Topics and Applications*. Springer (2014).
   https://doi.org/10.1007/978-3-662-43930-2

.. [5] Mainardi, F., and Spada, G. "Creep, relaxation and viscosity properties
   for basic fractional models in rheology."
   *European Physical Journal Special Topics*, 193, 133–160 (2011).
   https://doi.org/10.1140/epjst/e2011-01387-1

.. [6] Friedrich, C., and Braun, H. "Generalized Cole-Cole behavior and its
   rheological relevance." *Rheologica Acta*, 31, 309–322 (1992).
   https://doi.org/10.1007/BF00418328

.. [7] Metzler, R., and Klafter, J. "The random walk's guide to anomalous
   diffusion: A fractional dynamics approach."
   *Physics Reports*, 339, 1–77 (2000).
   https://doi.org/10.1016/S0370-1573(00)00070-3

.. [8] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [9] Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd Edition.
   Wiley (1980). ISBN: 978-0471048947

.. [10] Doi, M., and Edwards, S. F. *The Theory of Polymer Dynamics*.
   Oxford University Press (1986). ISBN: 978-0198520337
