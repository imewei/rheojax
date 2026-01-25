.. _model-fractional-burgers:

Fractional Burgers Model (Fractional)
=====================================

Quick Reference
---------------

- **Use when:** Complex creep with glassy compliance, fractional retardation, and viscous flow
- **Parameters:** 5 (Jg, Jk, α, τk, η₁)
- **Key equation:** :math:`J(t) = J_g + \frac{t^{\alpha}}{\eta_1\Gamma(1+\alpha)} + J_k[1 - E_{\alpha}(-(t/\tau_k)^{\alpha})]`
- **Test modes:** Creep, oscillation
- **Material examples:** Polymer composites, asphalt binders, bituminous materials, viscoelastic solids under load

.. include:: /_includes/fractional_seealso.rst

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`J_g`
     - 1/Pa
     - Glassy compliance (instantaneous elastic response)
   * - :math:`\eta_1`
     - Pa·s
     - Viscosity of Maxwell dashpot (controls terminal flow)
   * - :math:`J_k`
     - 1/Pa
     - Kelvin compliance magnitude (retardation amplitude)
   * - :math:`\alpha`
     - dimensionless
     - Fractional order (0 < α < 1, controls power-law character)
   * - :math:`\tau_k`
     - s
     - Retardation time (characteristic Kelvin timescale)
   * - :math:`E_{\alpha}(z)`
     - dimensionless
     - One-parameter Mittag-Leffler function
   * - :math:`\Gamma(z)`
     - dimensionless
     - Gamma function

Overview
--------

The **Fractional Burgers Model** combines a **Maxwell element** in series with a **Fractional Kelvin-Voigt element**, creating a five-parameter model that captures **glassy compliance, viscous flow, and fractional retardation** in a single compact framework. This model extends the classical four-element Burgers model by replacing the Kelvin-Voigt dashpot with a SpringPot, enabling power-law retardation instead of exponential relaxation.

The Fractional Burgers model is particularly effective for materials exhibiting **complex creep behavior** with both instantaneous elastic response, delayed fractional retardation, and long-term viscous flow. Common applications include **polymer composites under load, asphalt binders, bituminous materials, and viscoelastic solids** undergoing time-dependent deformation.

**Mechanical Analogue:**

.. code-block:: text

   [Maxwell Arm: Spring Gg + Dashpot η1] ---- series ---- [Fractional KV: Spring + SpringPot (Jk, α, τk)]

Physical Foundations
--------------------

The Fractional Burgers model combines three distinct mechanical responses:

1. **Instantaneous elastic response** (glassy compliance :math:`J_g`)
2. **Fractional retardation** (SpringPot in Kelvin arm with time constant :math:`\tau_k`)
3. **Long-term viscous flow** (dashpot viscosity :math:`\eta_1`)

**Microstructural Interpretation:**

- **:math:`J_g`**: Instantaneous bond stretching, glassy modulus
- **Fractional KV arm**: Distributed retardation from hierarchical polymer network rearrangements
- **Maxwell dashpot**: Irreversible chain flow, reptation, or permanent deformation

Governing Equations
-------------------

**Time Domain (Creep Compliance):**

.. math::

   J(t) = J_g + \frac{t^{\alpha}}{\eta_1\,\Gamma(1+\alpha)} + J_k\left[1 - E_{\alpha}\!\left(-\left(\frac{t}{\tau_k}\right)^{\alpha}\right)\right]

where :math:`E_{\alpha}(z)` is the **one-parameter Mittag-Leffler function**:

.. math::

   E_{\alpha}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

**Frequency Domain (Complex Compliance):**

.. math::

   J^{*}(\omega) = J_g + \frac{(i\omega)^{-\alpha}}{\eta_1\,\Gamma(1-\alpha)} + \frac{J_k}{1 + (i\omega\tau_k)^{\alpha}}

**Complex Modulus:**

.. math::

   G^{*}(\omega) = \frac{1}{J^{*}(\omega)}

Note: The inversion :math:`G^* = 1/J^*` is exact for linear viscoelastic materials.

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
   * - ``Jg``
     - :math:`J_g`
     - 1/Pa
     - [1e-9, 1e3]
     - Glassy compliance (instantaneous response)
   * - ``eta1``
     - :math:`\eta_1`
     - Pa·s
     - [1e-6, 1e12]
     - Viscosity (Maxwell arm, controls terminal flow)
   * - ``Jk``
     - :math:`J_k`
     - 1/Pa
     - [1e-9, 1e3]
     - Kelvin compliance (retardation magnitude)
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0.05, 0.95]
     - Fractional order (0.2-0.7 typical for polymers)
   * - ``tau_k``
     - :math:`\tau_k`
     - s
     - [1e-6, 1e6]
     - Retardation time (characteristic Kelvin timescale)

Regimes and Behavior
--------------------

**Short Time** (:math:`t \ll \tau_k`):

.. math::

   J(t) \approx J_g + \frac{t^{\alpha}}{\eta_1\,\Gamma(1+\alpha)}

Instantaneous glassy compliance plus early-time fractional flow from Maxwell arm.

**Intermediate Time** (:math:`t \sim \tau_k`):

.. math::

   J(t) \approx J_g + J_k\left[1 - E_{\alpha}\!\left(-\left(\frac{t}{\tau_k}\right)^{\alpha}\right)\right]

**Fractional retardation** dominated by Kelvin arm with power-law approach to equilibrium.

**Long Time** (:math:`t \gg \tau_k`):

.. math::

   J(t) \approx J_g + J_k + \frac{t}{\eta_1}

**Unbounded creep** (liquid-like) with constant compliance offset from glassy and Kelvin contributions.

Validity and Assumptions
-------------------------

- **Linear viscoelasticity**: Strain amplitudes remain small (< 5-10% typically)
- **Isothermal conditions**: Temperature constant throughout measurement
- **Time-invariant material**: No aging, degradation, or structural evolution
- **Supported test modes**: Creep (primary), oscillation
- **Fractional order bounds**: 0.05 < α < 0.95 for numerical stability
- **Liquid-like behavior**: Unbounded creep at long times (η₁ finite)

What You Can Learn
------------------

This section explains how to translate fitted Fractional Burgers parameters into
material insights and actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Glassy Compliance (Jg)**:
   The instantaneous elastic response upon stress application.

   - **For graduate students**: Jg reflects short-range bond stretching and
     angle deformation in the glassy state. For polymers, Jg ≈ 1/G∞ where
     G∞ is the glassy modulus (~1 GPa for many polymers).
   - **For practitioners**: Jg sets the immediate strain upon loading. Critical
     for impact resistance and short-time deformation.

**Kelvin Compliance (Jk)**:
   Controls the magnitude of delayed (retarded) elastic deformation.

   - Retardation magnitude: ΔJ = Jk
   - For polymers, relates to chain rearrangements in constrained environments
   - Typical values: 10⁻⁶-10⁻² Pa⁻¹

**Fractional Order (α)**:
   Governs the breadth of the retardation spectrum and power-law character.

   - **α → 0.2-0.3**: Very broad spectrum, highly heterogeneous (filled systems)
   - **α → 0.4-0.5**: Moderate breadth, typical for polymer composites
   - **α → 0.6-0.7**: Narrower spectrum, more uniform structure
   - **α → 1**: Exponential retardation (classical Burgers)

   *Physical interpretation*: Lower α indicates greater polydispersity in
   relaxation times arising from structural heterogeneity, filler distribution,
   or molecular weight distribution.

**Viscosity (η₁)**:
   Controls the rate of unbounded creep at long times.

   - Slope of J(t) at long times: dJ/dt = 1/η₁
   - For polymers, relates to molecular weight via η₁ ~ Mw³·⁴ (reptation)
   - Determines processability and long-term dimensional stability

**Retardation Time (τk)**:
   Characteristic timescale for the fractional Kelvin-Voigt relaxation.

   - Marks the transition from glassy to retardation-dominated regime
   - Temperature-dependent: follows WLF or Arrhenius behavior

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Burgers Behavior Classification
   :header-rows: 1
   :widths: 20 25 25 30

   * - Parameter Pattern
     - Material Type
     - Examples
     - Key Characteristics
   * - High Jk/Jg (> 10)
     - Soft viscoelastic solid
     - Polymer composites, filled elastomers
     - Large delayed compliance
   * - Low α (< 0.3)
     - Highly heterogeneous
     - Asphalt, bitumen, nanocomposites
     - Very broad spectrum
   * - High η₁ (> 10⁶ Pa·s)
     - High MW polymer
     - Melts, concentrated solutions
     - Slow terminal flow
   * - Low η₁ (< 10³ Pa·s)
     - Low MW or diluted
     - Modified bitumen, soft materials
     - Rapid creep

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **Jg ≈ 0 or poorly constrained**: Insufficient early-time data; use faster
  sampling or estimate from high-frequency G'
- **Linear J(t) at all times**: No retardation; use simple Maxwell liquid instead
- **α near bounds (0.05 or 0.95)**: Data may not support fractional behavior;
  try classical Burgers (α = 1)
- **Strong Jk-τk correlation**: Need better data coverage in intermediate regime

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Asphalt Pavement Design**:
   Use Burgers model to predict rutting under sustained traffic load. The
   terminal flow (η₁) determines permanent deformation rate, while Jk and α
   control elastic recovery.

**Polymer Composite Selection**:
   Compare Jk values between formulations. Lower Jk means better dimensional
   stability under load. Monitor α for filler dispersion quality.

**Food Texture Analysis**:
   Fit creep data from cheese or dough. High Jk indicates soft, easily
   deformable texture. Use α to quantify structural heterogeneity.

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Creep test** (primary): 4-5 decades in time (e.g., 0.1 s - 10⁴ s)
2. **Sampling**: Log-spaced, minimum 50 points per decade
3. **Stress level**: Within LVR, verify with amplitude sweep
4. **Temperature control**: ±0.1°C for polymers, ±0.5°C for bitumen

**Initialization Strategy:**

.. code-block:: python

   # From creep data J(t)
   Jg_init = J(t_min)  # Instantaneous compliance
   eta1_init = t / (J(t) - J(t_min)) at long time  # Terminal slope
   Jk_init = (J(t_mid) - Jg_init) * 0.5  # Mid-range magnitude
   tau_k_init = t where retardation is 50% complete
   alpha_init = 0.5  # Default starting point

**Optimization Tips:**

- Fit in log(compliance) space for better conditioning
- Use weighted least squares with log-spaced weights
- Constrain Jg < Jk (glassy stiffer than Kelvin)
- Verify residuals are random, not systematic

**Common Pitfalls:**

- **Overfitting**: Don't fit Burgers if classical 4-element model suffices
- **Underfitting**: If residuals show curvature, may need additional Kelvin element
- **Wrong regime**: Ensure data captures all three regimes (glassy, retardation, flow)

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalBurgersModel
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model
   model = FractionalBurgersModel()

   # Fit to experimental creep data
   t_exp = np.logspace(-1, 4, 100)  # 0.1 s to 10,000 s
   J_exp = load_creep_data()  # Load your data

   # Automatic fit
   model.fit(t_exp, J_exp, test_mode='creep')

   # Inspect fitted parameters
   print(f"Jg = {model.parameters.get_value('Jg'):.2e} Pa⁻¹")
   print(f"Jk = {model.parameters.get_value('Jk'):.2e} Pa⁻¹")
   print(f"α = {model.parameters.get_value('alpha'):.3f}")
   print(f"τk = {model.parameters.get_value('tau_k'):.2e} s")
   print(f"η₁ = {model.parameters.get_value('eta1'):.2e} Pa·s")

   # Predict creep at new times
   t_new = np.logspace(-2, 5, 200)
   data = RheoData(x=t_new, y=np.zeros_like(t_new), domain='time')
   data.metadata['test_mode'] = 'creep'
   J_pred = model.predict(data)

   # Bayesian uncertainty quantification
   result = model.fit_bayesian(
       t_exp, J_exp,
       num_warmup=1000,
       num_samples=2000,
       test_mode='creep'
   )
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)

See Also
--------

- :doc:`fractional_maxwell_model` — generalized two-SpringPot formulation
- :doc:`fractional_kelvin_voigt` — Kelvin arm used inside Burgers
- :doc:`../../transforms/mastercurve` — build broadband spectra for better fitting
- :doc:`../../transforms/fft` — convert relaxation to frequency domain
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing Burgers family

Material Examples
-----------------

**Polymer Composites** (:math:`J_g \approx 10^{-6}-10^{-5}` 1/Pa, :math:`\alpha \approx 0.3-0.5`):

- **Filled elastomers** (carbon black, silica fillers)
- **Fiber-reinforced polymers** under sustained load
- **Polymer nanocomposites** (clay, CNT fillers)

**Asphalt and Bitumen** (:math:`\eta_1 \approx 10^4-10^7` Pa·s, :math:`\alpha \approx 0.4-0.6`):

- **Asphalt concrete** (temperature-dependent)
- **Bituminous binders** for road pavements
- **Roofing materials**

**Food Materials** (:math:`J_k \approx 10^{-4}-10^{-2}`, :math:`\alpha \approx 0.2-0.5`):

- **Cheese** (long-term creep)
- **Dough** (wheat flour, viscoelastic retardation)
- **Semi-solid fats** (margarine, butter)

**Biological Tissues** (:math:`\alpha \approx 0.2-0.4`):

- **Ligaments and tendons** under sustained stress
- **Intervertebral discs** (viscoelastic creep)

Experimental Design
-------------------

**Creep Test (Primary Application):**

1. **Step stress**: Apply constant stress :math:`\sigma_0` within LVR
2. **Time span**: Cover 4-5 decades (e.g., 0.1 s - 10⁴ s)
3. **Sampling**: Log-spaced to capture all three regimes
4. **Analysis**: Fit :math:`J(t)` to identify :math:`J_g` (instantaneous), :math:`J_k` (retardation), :math:`\eta_1` (slope at long time)

**Frequency Sweep (Oscillatory):**

1. **Frequency range**: 0.001-100 rad/s (wide span critical)
2. **Strain amplitude**: Within LVR (0.5-5%)
3. **Analysis**: Fit :math:`G'(\omega)`, :math:`G''(\omega)` simultaneously
4. **Verification**: Check terminal flow region (:math:`G'' \sim \omega`, :math:`G' \sim \omega^2`)

Fitting Strategies
------------------

**Initialization from Creep Data:**

1. **:math:`J_g`**: Extrapolate :math:`J(t \to 0)` (instantaneous compliance)
2. **:math:`\eta_1`**: Slope of :math:`J(t)` at long time → :math:`1/\eta_1`
3. **:math:`J_k`**: Mid-time plateau height minus :math:`J_g`
4. **:math:`\tau_k`**: Time where retardation is half-complete
5. **:math:`\alpha`**: Curvature of retardation region in log-log plot

**Optimization:**

- Use weighted least squares (log-spaced weights)
- Constrain :math:`J_g < J_k` (glassy stiffer than Kelvin)
- Fit in compliance space for creep, modulus space for oscillation
- Verify residuals random across all time/frequency decades

Usage Example
-------------

.. code-block:: python

   from rheojax.models import FractionalBurgersModel
   import numpy as np

   # Create model
   model = FractionalBurgersModel()

   # Set typical parameters for polymer composite
   model.parameters.set_value('Jg', 1e-6)       # 1/Pa
   model.parameters.set_value('eta1', 1e5)      # Pa·s
   model.parameters.set_value('Jk', 5e-6)       # 1/Pa
   model.parameters.set_value('alpha', 0.4)     # dimensionless
   model.parameters.set_value('tau_k', 10.0)    # s

   # Predict creep compliance
   t = np.logspace(-1, 4, 100)
   J_t = model.predict(t, test_mode='creep')

   # Fit to experimental creep data
   # t_exp, J_exp = load_creep_data()
   # model.fit(t_exp, J_exp, test_mode='creep')

Limiting Behavior
-----------------

- **:math:`\alpha \to 1`**: Classical Burgers with exponential Kelvin retardation
- **:math:`J_k \to 0`**: Maxwell + fractional flow only (no retardation)
- **:math:`\eta_1 \to \infty`**: Fractional Kelvin-Voigt (bounded creep, no flow)
- **:math:`\tau_k \to 0`**: Instantaneous Kelvin response → :math:`J(t) = J_g + J_k + t/\eta_1`
- **:math:`\tau_k \to \infty`**: Kelvin arm inactive → simple Maxwell

Model Comparison
----------------

**Burgers vs Fractional Burgers:**

- **Classical Burgers**: Exponential retardation (:math:`\alpha = 1`)
- **Fractional Burgers**: Power-law retardation (0 < :math:`\alpha` < 1)
- Use Fractional when creep shows curved transition in log-log plots

**Burgers vs Fractional Maxwell Gel:**

- **Burgers**: 5 parameters, includes delayed elasticity (Kelvin arm)
- **FMG**: 3 parameters, single relaxation mode
- Use Burgers for complex creep with multiple timescales

Troubleshooting
---------------

**Issue: Cannot identify** :math:`J_g` **from data**

- **Cause**: Insufficient early-time resolution
- **Solution**: Use faster sampling or estimate from high-frequency modulus

**Issue: Oscillatory fit poor at low frequencies**

- **Cause**: Terminal flow region not captured
- **Solution**: Extend frequency sweep to lower :math:`\omega` (< 0.01 rad/s)

**Issue: Parameter correlation** (:math:`J_k` **and** :math:`\tau_k`)

- **Cause**: Insufficient data in retardation regime
- **Solution**: Focus measurements on intermediate timescale (:math:`t \sim \tau_k`)

Tips & Best Practices
----------------------

1. **Fit creep first**: Compliance space more natural for Burgers model
2. **Verify terminal flow**: Confirm linear :math:`J(t)` vs :math:`t` at long time
3. **Check bounds**: Ensure :math:`J_g < J_k` (physically meaningful)
4. **Use transforms**: Apply :doc:`../../transforms/fft` to convert creep → oscillation
5. **Log-log plots**: Visualize all three regimes clearly

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Bagley, R. L., and Torvik, P. J. "On the fractional calculus model of
   viscoelastic behavior." *Journal of Rheology*, 30, 133–155 (1986).
   https://doi.org/10.1122/1.549887

.. [3] Schiessel, H., and Blumen, A. "Hierarchical analogues to fractional
   relaxation equations." *Journal of Physics A*, 26, 5057–5069 (1993).
   https://doi.org/10.1088/0305-4470/26/19/034

.. [4] Koeller, R. C. "Applications of fractional calculus to the theory of
   viscoelasticity." *Journal of Applied Mechanics*, 51, 299–307 (1984).
   https://doi.org/10.1115/1.3167616

.. [5] Findley, W. N., Lai, J. S., and Onaran, K. *Creep and Relaxation of
   Nonlinear Viscoelastic Materials*. Dover (1989). ISBN: 978-0486660165

See Also
--------

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

--------

- :doc:`fractional_maxwell_model` — generalized two-SpringPot formulation
- :doc:`fractional_kelvin_voigt` — Kelvin arm used inside Burgers
- :doc:`../../transforms/mastercurve` — build broadband spectra for better fitting
- :doc:`../../transforms/fft` — convert relaxation to frequency domain
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing Burgers family
