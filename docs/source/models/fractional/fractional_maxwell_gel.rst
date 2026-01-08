.. _model-fractional-maxwell-gel:

Fractional Maxwell Gel (Fractional)
===================================

Quick Reference
---------------

**Use when:** Critical gels, power-law viscoelasticity transitioning to terminal flow
**Parameters:** 3 (c_α, α, η)
**Key equation:** :math:`G(t) = c_\alpha t^{-\alpha} E_{1-\alpha,1-\alpha}(-t^{1-\alpha}/\tau)` where :math:`\tau = \eta / c_\alpha^{1/(1-\alpha)}`
**Test modes:** Oscillation, relaxation, creep
**Material examples:** Critical gels, wormlike micelles, weak polymer networks, polymer solutions near gel point

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The **Fractional Maxwell Gel (FMG)** model consists of a **SpringPot element** (fractional viscoelastic element) in **series** with a **Newtonian dashpot**. This configuration captures the rheological behavior of materials transitioning from **power-law viscoelastic response** at short times to **terminal viscous flow** at long times.

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`c_\alpha`
     - SpringPot quasi-property (Pa·s\ :sup:`\alpha`). Controls the elastic stiffness scale.
   * - :math:`\alpha`
     - Fractional order (0 < :math:`\alpha` < 1). Controls the relaxation slope (0=solid, 1=liquid).
   * - :math:`\eta`
     - Dashpot viscosity (Pa·s). Controls terminal flow at long times.
   * - :math:`\tau`
     - Characteristic relaxation time (s), :math:`\tau = \eta / c_\alpha^{1/(1-\alpha)}`.
   * - :math:`E_{\alpha,\beta}`
     - Mittag-Leffler function (generalized exponential).

Overview
--------

The FMG model is particularly effective for describing **polymer solutions, physical gels, and soft materials** exhibiting gel-like characteristics with eventual viscous dissipation—materials that behave as soft solids at short timescales but flow like liquids over extended durations.

The SpringPot element provides fractional-order power-law viscoelasticity characterized by a broad relaxation spectrum, while the series dashpot ensures **terminal flow behavior** (:math:`G(t \to \infty) \to 0`). This combination makes the FMG model especially suitable for materials that exhibit intermediate behavior between pure elastic solids and Newtonian liquids, such as **critical gels** evolving toward sol states, **wormlike micelle solutions**, and **weak polymer networks** undergoing structural rearrangement.

Physical Foundation
-------------------

The Fractional Maxwell Gel extends the classical Maxwell model by replacing the **spring** with a **SpringPot**:

**Mechanical Analogue:**

.. code-block:: text

   [SpringPot (c_α, α)] ---- series ---- [Dashpot (η)]

The SpringPot provides power-law elasticity while the dashpot guarantees liquid-like behavior at long times.

**Microstructural Interpretation:**

- **SpringPot contribution**: Broad distribution of network relaxation modes (chain rearrangements, bond breaking/reformation)
- **Dashpot contribution**: Irreversible viscous flow from chain reptation or solvent drag
- **Combined behavior**: Gel-like response at short times transitions to flow at long times

Governing Equations
-------------------

**Relaxation Modulus:**

.. math::

   G(t) = c_\alpha t^{-\alpha} E_{1-\alpha,1-\alpha}\left(-\frac{t^{1-\alpha}}{\tau}\right)

where:

- :math:`E_{\alpha,\beta}(z)` = two-parameter Mittag-Leffler function
- :math:`\tau = \eta / c_\alpha^{1/(1-\alpha)}` = characteristic relaxation time (s)
- :math:`c_\alpha` = SpringPot quasi-property (Pa·s\ :sup:`\alpha`)
- :math:`\alpha` = fractional order in (0, 1)
- :math:`\eta` = dashpot viscosity (Pa·s)

**Complex Modulus (Oscillatory):**

.. math::

   G^*(\omega) = c_\alpha (i\omega)^\alpha \cdot \frac{i\omega\tau}{1 + i\omega\tau}

Decomposed into storage and loss moduli:

.. math::

   G'(\omega) &= c_\alpha \omega^{\alpha} \left[\cos\left(\frac{\alpha\pi}{2}\right) \frac{(\omega\tau)^2}{1 + (\omega\tau)^2} + \sin\left(\frac{\alpha\pi}{2}\right) \frac{\omega\tau}{1 + (\omega\tau)^2}\right] \\
   G''(\omega) &= c_\alpha \omega^{\alpha} \left[\sin\left(\frac{\alpha\pi}{2}\right) \frac{(\omega\tau)^2}{1 + (\omega\tau)^2} - \cos\left(\frac{\alpha\pi}{2}\right) \frac{\omega\tau}{1 + (\omega\tau)^2}\right]

**Creep Compliance:**

.. math::

   J(t) = \frac{1}{c_\alpha} t^\alpha E_{1+\alpha,1+\alpha}\left(-\left(\frac{t}{\tau}\right)^{1-\alpha}\right)

Shows bounded creep at short times transitioning to unbounded viscous flow at long times.

Mittag-Leffler Function
------------------------

The **two-parameter Mittag-Leffler function** :math:`E_{\alpha,\beta}(z)` is defined by:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

**Special Cases:**

- :math:`E_{1,1}(z) = e^z` → exponential (classical Maxwell)
- :math:`E_{\alpha,1}(z)` → one-parameter Mittag-Leffler
- :math:`E_{2,1}(-z^2) = \cos(z)` → oscillatory behavior

**Asymptotic Behavior:**

- **Small argument** (:math:`|z| \ll 1`): :math:`E_{\alpha,\beta}(z) \approx 1/\Gamma(\beta) + z/\Gamma(\alpha + \beta)`
- **Large argument** (:math:`|z| \gg 1, z < 0`): :math:`E_{\alpha,\beta}(z) \sim |z|^{-1}/\Gamma(\beta - \alpha)` → power-law decay

These asymptotics produce the crossover from power-law to viscous behavior in FMG.

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
   * - ``c_alpha``
     - :math:`c_\alpha`
     - Pa·s\ :sup:`\alpha`
     - [1e-3, 1e9]
     - SpringPot material constant (sets modulus scale)
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0.05, 0.95]
     - Power-law exponent (0.3-0.7 typical for gels)
   * - ``eta``
     - :math:`\eta`
     - Pa·s
     - [1e-6, 1e12]
     - Dashpot viscosity (controls terminal flow)

Physical Meaning of :math:`\alpha`
-----------------------------------

The fractional order :math:`\alpha` characterizes the **viscoelastic character**:

- **:math:`\alpha < 0.5`**: Solid-like (:math:`G' > G''` at intermediate frequencies)
- **:math:`\alpha = 0.5`**: Critical gel signature (:math:`G' \sim G'' \propto \omega^{0.5}`)
- **:math:`\alpha > 0.5`**: Liquid-like (:math:`G'' > G'` at low frequencies)

**Material Ranges:**

- **Polymer gels**: :math:`\alpha \approx 0.3-0.6`
- **Wormlike micelles**: :math:`\alpha \approx 0.4-0.7`
- **Weak networks**: :math:`\alpha \approx 0.2-0.5`
- **Colloidal gels**: :math:`\alpha \approx 0.3-0.5`

Regimes and Behavior
--------------------

**Short-Time / High-Frequency Regime** (:math:`t \ll \tau` or :math:`\omega \gg 1/\tau`):

SpringPot dominates, yielding power-law behavior:

.. math::

   G(t) \sim c_\alpha t^{-\alpha}, \quad G^*(\omega) \sim c_\alpha (i\omega)^\alpha

Material behaves as a **fractional gel** with broad relaxation spectrum.

**Long-Time / Low-Frequency Regime** (:math:`t \gg \tau` or :math:`\omega \ll 1/\tau`):

Dashpot controls the response, leading to **terminal viscous flow**:

.. math::

   G(t) \sim \frac{\eta}{t}, \quad G''(\omega) \sim \omega\eta, \quad G'(\omega) \sim \omega^2

Material flows like a Newtonian liquid with viscosity :math:`\eta`.

**Intermediate Regime** (:math:`t \sim \tau`):

Mittag-Leffler function provides smooth crossover between power-law and viscous regimes. The characteristic time :math:`\tau` marks the transition from gel-like to liquid-like behavior.

Material Examples
-----------------

**Polymer Solutions** (:math:`c_\alpha \approx 10^2-10^4` Pa·s\ :sup:`\alpha`, :math:`\alpha \approx 0.4-0.6`, :math:`\eta \approx 10-10^3` Pa·s):

- **Polyacrylamide solutions** (5-10 wt%)
- **PEO (polyethylene oxide)** in water
- **Xanthan gum** solutions

**Physical Gels** (:math:`c_\alpha \approx 10^3-10^5`, :math:`\alpha \approx 0.3-0.5`, :math:`\eta \approx 10^2-10^4`):

- **Gelatin gels** near sol-gel transition
- **Agar gels** at low concentration (< 1%)
- **Alginate gels** (weak cross-linking)

**Wormlike Micelle Solutions** (:math:`\alpha \approx 0.5-0.7`, :math:`\eta \approx 1-100` Pa·s):

- **CTAB** (cetyltrimethylammonium bromide) micelles
- **CPyCl/NaSal** (cetylpyridinium chloride/sodium salicylate)

**Colloidal Gels** (:math:`\alpha \approx 0.3-0.5`, :math:`\eta \approx 10-10^3`):

- **Carbon black suspensions**
- **Silica gel networks**

Experimental Design
-------------------

**Frequency Sweep (SAOS):**

1. **Frequency range**: 0.01-100 rad/s (minimum 3 decades)
2. **Strain amplitude**: Within LVR (typically 0.5-5%)
3. **Identify regimes**:
   - High :math:`\omega`: Power-law with slope :math:`\alpha`
   - Low :math:`\omega`: Terminal flow (:math:`G'' \sim \omega`, :math:`G' \sim \omega^2`)
4. **Crossover frequency**: :math:`\omega_c \approx 1/\tau` where regime transition occurs

**Stress Relaxation:**

1. **Step strain**: :math:`\gamma_0 = 1-5\%` within LVR
2. **Time span**: Cover 4-5 decades (e.g., 0.01-10³ s)
3. **Sampling**: Log-spaced to capture both regimes
4. **Analysis**: Early-time power-law → late-time viscous decay

**Creep Test:**

1. **Constant stress**: Within LVR
2. **Time span**: Long enough to observe viscous flow (> 10³ s)
3. **Expected**: Bounded creep → unbounded flow

Fitting Strategies
------------------

**Smart Initialization (v0.2.0):**

RheoJAX automatically initializes FMG parameters from oscillation data using frequency-domain analysis:

1. **Estimate** :math:`c_\alpha` from high-frequency plateau
2. **Estimate** :math:`\alpha` from power-law slope in intermediate regime
3. **Estimate** :math:`\eta` from low-frequency terminal behavior (:math:`G'' \sim \omega\eta`)
4. **Estimate** :math:`\tau = 1/\omega_c` from crossover frequency

**Manual Initialization:**

.. code-block:: python

   # From frequency sweep log-log plot
   alpha_init = slope_of_log_Gp_vs_log_omega  # intermediate regime
   eta_init = Gpp_low_freq / omega_low        # terminal region
   c_alpha_init = Gp_high_freq / (omega_high**alpha * cos(pi*alpha/2))
   tau_init = 1 / omega_crossover

**Optimization Tips:**

- Fit in log-space for better conditioning
- Constrain :math:`\alpha` bounds to [0.1, 0.9] to avoid singularities
- Use NLSQ optimizer (5-270x faster than scipy)
- Verify residuals show no systematic trends

Usage Example
-------------

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalMaxwellGel()

   # Frequency sweep (wormlike micelle solution)
   omega = np.logspace(-2, 2, 50)
   G_star_exp = load_experimental_data()  # Complex modulus

   # Automatic smart initialization + fit (v0.2.0)
   model.fit(omega, G_star_exp, test_mode='oscillation')

   # Inspect fitted parameters
   print(f"c_alpha = {model.parameters.get_value('c_alpha'):.2e} Pa·s^α")
   print(f"alpha = {model.parameters.get_value('alpha'):.4f}")
   print(f"eta = {model.parameters.get_value('eta'):.2e} Pa·s")
   print(f"tau = {model.parameters.get_value('eta') / model.parameters.get_value('c_alpha')**(1/(1-model.parameters.get_value('alpha'))):.2e} s")

   # Predict relaxation modulus
   t = np.logspace(-3, 3, 100)
   data = RheoData(x=t, y=np.zeros_like(t), domain='time')
   data.metadata['test_mode'] = 'relaxation'
   G_t = model.predict(data)

   # Bayesian uncertainty quantification
   result = model.fit_bayesian(
       omega, G_star_exp,
       num_warmup=1000,
       num_samples=2000,
       test_mode='oscillation'
   )
   ci = model.get_credible_intervals(result.posterior_samples, credibility=0.95)

Model Comparison
----------------

**FMG vs FML (Fractional Maxwell Liquid):**

- **FMG**: SpringPot + dashpot → power-law + terminal flow
- **FML**: SpringPot + spring → power-law + equilibrium plateau
- Use FMG for flowing gels; FML for soft solids

**FMG vs Classical Maxwell:**

- **Maxwell**: Exponential relaxation (:math:`\alpha = 1`)
- **FMG**: Power-law relaxation (0 < :math:`\alpha` < 1, broad spectrum)
- FMG reduces to Maxwell as :math:`\alpha \to 1`

**FMG vs Fractional Burgers:**

- **FMG**: 3 parameters, single relaxation mode
- **Burgers**: 5 parameters, adds retardation mode (delayed elasticity)
- Use Burgers for complex creep with multiple timescales

Limiting Behavior
-----------------

- **:math:`\alpha \to 1`**: Approaches classical Maxwell (:math:`G^*(\omega) \sim i\omega\eta`)
- **:math:`\alpha \to 0`**: Approaches elastic spring in series with dashpot
- **:math:`\eta \to \infty`**: Reduces to pure SpringPot (:math:`G^*(\omega) = c_\alpha (i\omega)^\alpha`)
- **:math:`\eta \to 0`**: Non-physical (no dissipation mechanism)
- **:math:`c_\alpha \to 0`**: Pure dashpot (:math:`G^*(\omega) = i\omega\eta`)

Troubleshooting
---------------

.. list-table:: Common Fitting Issues
   :widths: 25 35 40
   :header-rows: 1

   * - Symptom
     - Possible Cause
     - Solution
   * - **Poor fit in terminal regime**
     - Insufficient low-frequency data
     - Extend frequency sweep to lower :math:`\omega` or use longer relaxation test.
   * - **:math:`\alpha` closes to 1**
     - Material is nearly Maxwellian
     - Use classical **Maxwell** model instead (narrow spectrum).
   * - **Oscillatory residuals at high :math:`\omega`**
     - Multiple relaxation modes
     - Use **Fractional Maxwell Model (FMM)** which has two fractional orders.
   * - **Non-convergence**
     - Poor initial guess or parameter correlation
     - Use **Smart Initialization** (automatic in v0.2.0) or warm-start with NLSQ.

Tips & Best Practices
----------------------

1. **Verify regimes**: Plot :math:`\log(G')`, :math:`\log(G'')` vs :math:`\log(\omega)` to confirm power-law and terminal regions
2. **Use smart initialization**: Automatic in RheoJAX v0.2.0 for oscillation mode
3. **Check Mittag-Leffler implementation**: RheoJAX uses optimized JAX-based computation
4. **Bayesian inference**: Quantify parameter uncertainty with `fit_bayesian()`
5. **Warm-start**: Use NLSQ fit to initialize NUTS sampling (2-5x faster convergence)

References
----------

**Fractional Viscoelasticity:**

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010). — **Comprehensive mathematical treatment**
- R.L. Bagley and P.J. Torvik, "On the fractional calculus model of viscoelastic
  behavior," *J. Rheol.* 30, 133–155 (1986). — **Foundational FMG analysis**
- C. Friedrich, "Relaxation and retardation functions of the fractional Maxwell model,"
  *Rheol. Acta* 30, 151–158 (1991). — **Analytical solutions**

**Mittag-Leffler Functions:**

- R. Gorenflo, A.A. Kilbas, F. Mainardi, S.V. Rogosin, *Mittag-Leffler Functions, Related
  Topics and Applications*, Springer (2014). — **Definitive reference**
- R. Hilfer (ed.), *Applications of Fractional Calculus in Physics*, World Scientific (2000).

**Gel Rheology:**

- G.S. Blair, B.C. Veinoglou, and J.E. Caffyn, "Limitations of the Newtonian time scale
  ...," *Proc. R. Soc. A* 189, 69–87 (1947). — **Early gel characterization**
- H.H. Winter and F. Chambon, "Analysis of linear viscoelasticity of a crosslinking polymer
  at the gel point," *J. Rheol.* 30, 367–382 (1986). — **Gel point criterion**

**Applications:**

- R. Metzler and T.F. Nonnenmacher, "Fractional relaxation processes and related
  phenomena," *Int. J. Mod. Phys. B* 8, 211–224 (1994). — **Anomalous diffusion**
- H. Schiessel and A. Blumen, "Hierarchical analogues to fractional relaxation equations,"
  *J. Phys. A* 26, 5057–5069 (1993). — **Microstructural foundations**

See also
--------

**Related Fractional Models:**

- :doc:`fractional_maxwell_liquid` — complementary model with spring instead of dashpot
- :doc:`fractional_maxwell_model` — generalized two-order formulation encompassing FMG
- :doc:`fractional_burgers` — adds Kelvin branch for delayed elasticity
- :doc:`../classical/springpot` — fundamental SpringPot element

**Transforms:**

- :doc:`../../transforms/owchirp` — broadband LAOS sweeps to estimate fractional slopes
- :doc:`../../transforms/fft` — convert relaxation data to frequency domain
- :doc:`../../transforms/mutation_number` — monitor gel-to-sol transitions

**Examples:**

- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — tutorial comparing Fractional Maxwell family
- :doc:`../../examples/bayesian/02-fractional-gel-uncertainty` — uncertainty quantification for FMG
