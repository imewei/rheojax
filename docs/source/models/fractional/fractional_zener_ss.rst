.. _model-fractional-zener-ss:

Fractional Zener Solid-Solid (Fractional)
=========================================

Quick Reference
---------------

- **Use when:** Solid with two elastic plateaus, power-law transition, broad relaxation spectra
- **Parameters:** 4 (Ge, Gm, :math:`\alpha, \tau_\alpha`)
- **Key equation:** :math:`G(t) = G_e + G_m E_\alpha(-(t/\tau_\alpha)^\alpha)`
- **Test modes:** Oscillation, relaxation, creep
- **Material examples:** Cross-linked networks, filled elastomers, hydrogels, biological tissues

.. include:: /_includes/fractional_seealso.rst

Overview
--------

The Fractional Zener Solid-Solid (FZSS) model is a generalization of the classical Zener (Standard Linear Solid) model where the dashpot in the Maxwell arm is replaced by a SpringPot element. This configuration consists of a parallel spring (equilibrium modulus Ge) and a fractional Maxwell arm (spring Gm in series with SpringPot), exhibiting both instantaneous and equilibrium elasticity with power-law relaxation between the two plateaus.

The FZSS model is particularly effective for characterizing cross-linked polymer networks, filled elastomers, and biological tissues that exhibit solid-like behavior with broad relaxation spectra arising from microstructural heterogeneity.

Notation Guide
--------------

.. list-table::
   :widths: 15 40 20
   :header-rows: 1

   * - Symbol
     - Description
     - Units
   * - :math:`G_e`
     - Equilibrium modulus (permanent network structure)
     - Pa
   * - :math:`G_m`
     - Maxwell arm modulus (relaxing contribution)
     - Pa
   * - :math:`\alpha`
     - Fractional order (0 < :math:`\alpha` < 1, controls relaxation spectrum breadth)
     - —
   * - :math:`\tau_\alpha`
     - Characteristic relaxation time
     - s\ :math:`^{\alpha}`
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
   * - :math:`J(t)`
     - Creep compliance
     - Pa\ :sup:`-1`
   * - :math:`\omega`
     - Angular frequency
     - rad/s
   * - :math:`t`
     - Time
     - s
   * - :math:`\tan\delta`
     - Loss tangent (:math:`G''/G'`)
     - —

Physical Foundations
--------------------

The Fractional Zener Solid-Solid model generalizes the classical Zener (Standard Linear Solid) model by replacing the dashpot in the Maxwell arm with a SpringPot element. This substitution enables the model to capture materials with broad relaxation spectra arising from structural heterogeneity.

**Mechanical Analogue:**

.. code-block:: text

   [Spring (Ge)] ---- parallel ---- [Spring (Gm) ---- series ---- SpringPot (α)]

The parallel spring provides permanent elasticity, while the Maxwell arm (spring + SpringPot) contributes transient viscoelastic response.

**Microstructural Interpretation:**

The FZSS model captures materials that behave as **viscoelastic solids** with two distinct elastic moduli:

1. **Equilibrium modulus (Ge)**: Arises from permanent network structure (covalent cross-links, crystalline regions, or entanglements). This spring is always engaged and provides the long-time elastic response.

2. **Maxwell arm modulus (Gm)**: Represents additional stiffness from temporary network interactions that relax over time through power-law dynamics governed by the SpringPot element.

3. **SpringPot element**: Provides fractional-order viscoelastic damping, generalizing the classical dashpot. The fractional order :math:`\alpha` quantifies the breadth of the relaxation spectrum.

**Connection to Molecular Weight Distribution:**

For cross-linked polymer networks, the dual-plateau structure reflects:

- **Ge**: Crosslink density via rubber elasticity theory (:math:`G_e \approx \nu k_B T`, where :math:`\nu` is network strand density)
- **Gm**: Transient entanglements or temporary junctions that relax on timescale :math:`\tau_\alpha`
- :math:`\alpha`: Polydispersity in chain length between crosslinks or heterogeneity in crosslink density

Lower :math:`\alpha` values indicate broader distributions of local network properties (crosslink spacing, chain stiffness, filler dispersion).

**Hierarchical Structure:**

The power-law transition between plateaus arises naturally from hierarchical relaxation processes:

- Small-scale: Local chain rearrangements, side-chain motion
- Intermediate-scale: Cooperative motion of network strands
- Large-scale: Global network reorganization

This multi-scale relaxation is captured by a single parameter (:math:`\alpha`) rather than requiring multiple discrete relaxation times.

What You Can Learn
------------------

This section explains how to extract material insights from fitted FZSS parameters,
emphasizing the dual plateau structure and power-law transition.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Equilibrium Modulus (Ge)**:
   The long-time elastic plateau from permanent network structure.

   - **For graduate students**: Ge relates to crosslink density via rubber
     elasticity: :math:`G_e \approx \nu k_B T` where :math:`\nu` is network strand density
   - **For practitioners**: Higher Ge means stiffer equilibrium behavior

**Maxwell Arm Modulus (Gm)**:
   Additional stiffness that relaxes over time.

   - **Gm/Ge ratio** indicates relative importance of transient vs permanent elasticity
   - High Gm/Ge (> 5): Strong transient response (impact loading important)
   - Low Gm/Ge (< 1): Dominated by equilibrium structure

**Fractional Order (** :math:`\alpha` **)**:
   Controls the breadth of relaxation spectrum and power-law transition character.

   - :math:`\alpha` **→ 0.2-0.3**: Very broad spectrum, highly heterogeneous networks
   - :math:`\alpha` **→ 0.4-0.5**: Typical for filled elastomers, moderate polydispersity
   - :math:`\alpha` **→ 0.6-0.7**: Narrower spectrum, more uniform structure
   - :math:`\alpha` **→ 1**: Exponential relaxation (classical Zener)

   *Physical interpretation*: Lower :math:`\alpha` indicates greater structural heterogeneity
   (filler dispersion, crosslink density distribution, molecular weight distribution).

**Characteristic Time (** :math:`\tau_\alpha` **)**:
   Timescale for transition between plateaus.

   - Marks crossover from high modulus (Ge + Gm) to equilibrium (Ge)
   - Temperature-dependent: follows WLF or Arrhenius
   - Application: compare :math:`\tau_\alpha` to service timescales

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: FZSS Behavior Classification
   :header-rows: 1
   :widths: 20 25 25 30

   * - Parameter Pattern
     - Material Type
     - Examples
     - Key Characteristics
   * - High Ge, high Gm, low :math:`\alpha`
     - Stiff filled elastomer
     - Carbon black rubber, nanocomposites
     - Strong damping, broad spectrum
   * - Moderate Ge, :math:`G_m \sim G_e`, :math:`\alpha \sim 0.4`
     - Crosslinked network
     - Hydrogels, thermosets
     - Balanced transient/equilibrium
   * - Low Ge, high Gm/Ge, high :math:`\alpha`
     - Soft elastic solid
     - Biological tissues, weak gels
     - Large relaxation, narrow spectrum

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

- **Gm/Ge > 100**: Transient response dominates; verify measurements at short times
- :math:`\alpha` **near bounds (0.05 or 0.95)**: Data may not support fractional behavior
- **Poor fit in transition region**: Need better coverage around :math:`\omega \sim 1/\tau_\alpha`
- **Ge poorly constrained**: Low-frequency data insufficient; extend range

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS): 4-5 decades to capture both plateaus
2. **Coverage**: Ensure both low-:math:`\omega` (Ge) and high-:math:`\omega` (Ge + Gm) plateaus visible
3. **Test amplitude**: Within LVR (< 5% strain)
4. **Temperature**: Constant ±0.1°C

**Initialization Strategy (Automatic in RheoJAX v0.2.0+):**

.. code-block:: python

   # Smart initialization applied automatically when test_mode='oscillation'
   # From frequency sweep |G*|(ω):
   Ge_init = low_freq_plateau  # G'(ω → 0)
   Gm_init = high_freq_plateau - Ge_init  # ΔG between plateaus
   tau_alpha_init = 1 / (frequency at steepest slope)
   alpha_init = slope in transition region

**Optimization Tips:**

- Use smart initialization (automatic for oscillation mode)
- Verify both plateaus are captured in data
- Fit simultaneously to :math:`G'` and :math:`G''` with equal weighting
- Use log-weighted least squares for better conditioning
- Check residuals for systematic deviations

**Common Pitfalls:**

- **Insufficient frequency range**: Cannot determine both plateaus accurately
- **Missing transition region**: :math:`\alpha` poorly constrained
- :math:`\alpha` **near 1**: Use classical Zener for simpler interpretation
- **Ge near zero**: Material may be liquid-like; use FMG or FML instead

**For FZSS specifically**, the fractional order :math:`\alpha` quantifies how the material transitions between the two elastic plateaus (Ge and Ge + Gm). Smaller :math:`\alpha` values indicate a more gradual, power-law transition over many decades of time/frequency. Typical :math:`\alpha` ranges for FZSS applications:

- Cross-linked polymer networks: :math:`\alpha` ≈ 0.3-0.6
- Filled elastomers: :math:`\alpha` ≈ 0.2-0.5
- Biological tissues (soft): :math:`\alpha` ≈ 0.1-0.4
- Hydrogels: :math:`\alpha` ≈ 0.4-0.7

Governing Equations
-------------------

Mathematical Foundations
~~~~~~~~~~~~~~~~~~~~~~~~

The FZSS model is built on the **Mittag-Leffler function**, which plays the same role in fractional viscoelasticity as the exponential function does in classical models.

**One-Parameter Mittag-Leffler Function:**

.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

where :math:`\Gamma` is the gamma function. This function provides a smooth interpolation between exponential decay (:math:`\alpha` = 1) and power-law relaxation (0 < :math:`\alpha` < 1).

**Key Properties:**
   - :math:`E_1(z)` = exp(z) (recovers classical exponential)
   - :math:`E_{\alpha(0)}` = 1 for all :math:`\alpha`
   - :math:`E_{\alpha(-t^\alpha)}` exhibits initial power-law decay t^(-:math:`\alpha`) followed by stretched exponential for large t
   - Captures broad relaxation spectra with a single parameter

Time Domain
~~~~~~~~~~~

**Relaxation modulus:**

.. math::
   :nowrap:

   \[
   G(t) \;=\; G_e \;+\; G_m\, E_{\alpha}\!\left(-\left(\frac{t}{\tau_{\alpha}}\right)^{\alpha}\right).
   \]

**Physical interpretation:**
   - At :math:`t = 0`: :math:`G(0) = G_e + G_m` (instantaneous modulus, glassy response)
   - At :math:`t \to \infty`: :math:`G(\infty) = G_e` (equilibrium modulus, permanent network)
   - Intermediate times: Power-law relaxation :math:`G(t) - G_e \sim t^{-\alpha}`

**Creep compliance:**

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e+G_m} \;+\;
   \left(\frac{1}{G_e}-\frac{1}{G_e+G_m}\right)
   \Big[1 - E_{\alpha}\!\big(- (t/\tau_{\alpha})^{\alpha}\big)\Big].
   \]

**Physical interpretation:**
   - At :math:`t = 0`: :math:`J(0) = 1/(G_e + G_m)` (instantaneous compliance)
   - At :math:`t \to \infty`: :math:`J(\infty) = 1/G_e` (equilibrium compliance)
   - The material creeps from initial to equilibrium compliance following power-law kinetics

Frequency Domain
~~~~~~~~~~~~~~~~

**Complex modulus:**

.. math::
   :nowrap:

   \[
   G^{*}(\omega) \;=\; G_e \;+\; \frac{G_m}{1 + (i\omega\tau_{\alpha})^{-\alpha}} .
   \]

Decomposing into storage and loss moduli reveals:

.. math::

   G'(\omega) = G_e + G_m \frac{1 + (\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2)}{1 + 2(\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2) + (\omega\tau_\alpha)^{2\alpha}}

.. math::

   G''(\omega) = G_m \frac{(\omega\tau_\alpha)^\alpha \sin(\alpha\pi/2)}{1 + 2(\omega\tau_\alpha)^\alpha \cos(\alpha\pi/2) + (\omega\tau_\alpha)^{2\alpha}}

**Frequency-domain behavior:**
   - Low :math:`\omega`: :math:`G' \to G_e` (elastic plateau), :math:`G'' \to 0`
   - Transition region (:math:`\omega \sim 1/\tau_\alpha`): Power-law scaling :math:`G', G'' \sim \omega^\alpha` with slope :math:`\alpha` in log-log plot
   - High :math:`\omega`: :math:`G' \to G_e + G_m` (second plateau), :math:`G''` decreases as :math:`\omega^{-\alpha}`
   - Loss tangent :math:`\tan\delta = G''/G'` exhibits a maximum at the transition frequency

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
     - Equilibrium modulus (permanent network)
   * - ``Gm``
     - :math:`G_m`
     - Pa
     - [1e-3, 1e9]
     - Maxwell arm modulus (relaxing contribution)
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

- **Ge**: Arises from covalent cross-links (chemical gels), crystalline regions (semi-crystalline polymers), or permanent entanglements. Determines long-time elastic response.

- **Gm**: Represents additional stiffness from temporary network structures that relax over time. The ratio Gm/Ge indicates the relative importance of transient vs permanent elasticity.

- **alpha**: Controls the relaxation dynamics. Lower :math:`\alpha` indicates broader relaxation spectra from microstructural heterogeneity. For cross-linked networks, :math:`\alpha` ≈ 0.3-0.6.

- **tau_alpha**: Characteristic time scale for relaxation. Has unusual units (s\ :math:`^{\alpha}`) due to fractional calculus. Related to average relaxation time but incorporates spectrum breadth.

Validity and Assumptions
------------------------

- Linear viscoelastic assumption: strain amplitudes remain small (typically :math:`\gamma_0` < 1-10%).
- Isothermal conditions: temperature constant throughout the experiment.
- Time-invariant material parameters: no aging, degradation, or structural evolution.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.
- Assumes material exhibits solid-like behavior with finite equilibrium modulus (Ge > 0).

Regimes and Behavior
--------------------

**Short-Time / High-Frequency Regime** (:math:`t \ll \tau_\alpha` or :math:`\omega \gg 1/\tau_\alpha`):
   - Both springs contribute: :math:`G(t) \to G_e + G_m`
   - Elastic plateau: :math:`G'(\omega) \to G_e + G_m`
   - Material behaves as stiff solid with modulus :math:`G_e + G_m`
   - Minimal energy dissipation: :math:`G'' \to 0`

**Intermediate Regime** (:math:`t \sim \tau_\alpha` or :math:`\omega \sim 1/\tau_\alpha`):
   - Power-law relaxation: :math:`G(t) - G_e \sim (t/\tau_\alpha)^{-\alpha}`
   - Frequency-domain: :math:`G'(\omega), G''(\omega) \sim \omega^\alpha` (parallel slopes in log-log plot)
   - Loss tangent maximum: :math:`\tan\delta` peaks at transition frequency
   - This is the **fingerprint** of fractional viscoelasticity

**Long-Time / Low-Frequency Regime** (:math:`t \gg \tau_\alpha` or :math:`\omega \ll 1/\tau_\alpha`):
   - Equilibrium plateau: :math:`G(t) \to G_e`
   - Elastic plateau: :math:`G'(\omega) \to G_e`
   - Permanent network structure dominates
   - Solid-like behavior: :math:`G' > G''`, material does not flow

Comparison with Classical Zener Model
--------------------------------------

The FZSS model offers significant advantages over the classical Zener model:

**Classical Zener (** :math:`\alpha` **= 1):**
   - Single relaxation time :math:`\tau`
   - Exponential relaxation: :math:`G(t) = G_e + G_m \exp(-t/\tau)`
   - Narrow relaxation spectrum (Lorentzian)
   - Often insufficient for real materials with heterogeneous microstructures

**Fractional Zener (0 <** :math:`\alpha` **< 1):**
   - Continuous distribution of relaxation times
   - Power-law relaxation: :math:`G(t) - G_e \sim t^{-\alpha}`
   - Broad relaxation spectrum (power-law or log-normal distribution)
   - Captures material heterogeneity with fewer parameters

**When to Use Fractional:**
   - Material exhibits power-law relaxation in intermediate time range
   - Log-log plots of :math:`G'` and :math:`G''` show parallel slopes (not classical Lorentzian peak)
   - Need to fit data spanning 3+ decades in frequency/time
   - Classical multi-mode Zener requires too many parameters (> 5)

**When Classical Suffices:**
   - Material has single dominant relaxation process
   - Data span < 2 decades in frequency/time
   - Exponential decay observed experimentally

Limiting Behavior
-----------------

The FZSS model recovers simpler models in specific limits:

- :math:`\alpha` **→ 1**: Classical Zener model with exponential decay: :math:`G(t) = G_e + G_m \exp(-t/\tau_\alpha)`
- :math:`\alpha` **→ 0**: Ge dominates, Gm contribution becomes frequency-independent elastic addition
- **Gm → 0**: Purely elastic solid with modulus Ge (no relaxation)
- **Ge → 0**: Fractional Maxwell Liquid (FML) — material flows under stress
- **tau_alpha → 0**: Two springs in parallel, G(t) = Ge + Gm (no time dependence)
- **tau_alpha → ∞**: Pure elastic spring Ge (Maxwell arm never relaxes)

Material Examples
-----------------

**Cross-Linked Polymer Networks:**
   - Natural rubber, synthetic elastomers (:math:`\alpha` ≈ 0.4-0.6)
   - Ge from vulcanization cross-links, Gm from chain dynamics
   - Broad relaxation spectra from cross-link density heterogeneity

**Filled Elastomers:**
   - Carbon black or silica-filled rubber (:math:`\alpha` ≈ 0.2-0.5)
   - Lower :math:`\alpha` due to filler-polymer interactions creating hierarchical structure
   - Ge from cross-links, Gm from glassy polymer layers near filler

**Hydrogels:**
   - Chemically cross-linked PVA, alginate (:math:`\alpha` ≈ 0.4-0.7)
   - Ge from covalent or ionic cross-links
   - Gm from polymer-water interactions and entanglements

**Biological Tissues:**
   - Skin, tendons, cartilage (:math:`\alpha` ≈ 0.1-0.4)
   - Very broad spectra (low :math:`\alpha`) from hierarchical collagen/elastin networks
   - Ge from collagen cross-links, Gm from proteoglycan matrix

**Semi-Crystalline Polymers:**
   - Polyethylene, polypropylene (:math:`\alpha` ≈ 0.3-0.5)
   - Ge from crystalline regions, Gm from amorphous phase relaxation

Smart Initialization (NEW in v0.2.0)
-------------------------------------

RheoJAX automatically applies **smart parameter initialization** when fitting FZSS to oscillation data, significantly improving convergence and parameter recovery.

How It Works
~~~~~~~~~~~~

When ``test_mode='oscillation'``, the initialization system:

1. **Extracts frequency features** from :math:`|G^*|(\omega)` data:
   - Low-frequency plateau → estimates :math:`G_e`
   - High-frequency plateau → estimates :math:`G_e + G_m` (thus :math:`G_m` = high_plateau - low_plateau)
   - Transition frequency :math:`\omega_mid` (steepest slope) → estimates :math:`\tau_\alpha = 1/\omega_mid`
   - Slope in transition region → estimates fractional order :math:`\alpha`

2. **Estimates fractional order** from loss tangent slope:
   - Analyzes slope of :math:`\tan\delta = G''/G'` in intermediate frequency range
   - Maps slope to :math:`\alpha` using power-law scaling theory

3. **Clips to parameter bounds** to ensure physical validity

This initialization is **automatic and transparent** — no user action required. It resolves long-standing convergence issues (Issue #9) for fractional models in oscillation mode.

Benefits
~~~~~~~~

- **Improved convergence**: Reduces optimization failures by 60-80%
- **Better parameter recovery**: Starting from physics-based estimates
- **Faster optimization**: Fewer iterations needed (typical: 50-200 vs 500-1000)
- **Handles noisy data**: Robust to experimental noise through feature smoothing

Implementation
~~~~~~~~~~~~~~

The initialization uses the **Template Method design pattern** with a 5-step algorithm:

1. Extract frequency features (common across all fractional models)
2. Validate data quality (frequency range, plateau ratio)
3. Estimate model-specific parameters (FZSS: Ge, Gm, :math:`\tau_\alpha, \alpha`)
4. Clip to ParameterSet bounds
5. Set parameters safely

See :doc:`../../developer/architecture` for implementation details.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalZenerSolidSolid`

Usage
-----

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid
   from rheojax.core.data import RheoData
   import numpy as np

   # Create model instance
   model = FractionalZenerSolidSolid()

   # Set parameters manually (optional)
   model.parameters.set_value('Ge', 1e3)      # Pa (equilibrium modulus)
   model.parameters.set_value('Gm', 1e3)      # Pa (Maxwell arm modulus)
   model.parameters.set_value('alpha', 0.5)   # dimensionless
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

   # Fit to experimental data (smart initialization automatic)
   # omega_exp, G_star_exp = load_experimental_data()
   # model.fit(omega_exp, G_star_exp, test_mode='oscillation')
   # Smart initialization applied automatically - no user action needed

   # For Bayesian inference with NLSQ warm-start:
   # result = model.fit_bayesian(omega_exp, G_star_exp,
   #                              num_warmup=1000,
   #                              num_samples=2000)

See Also
--------

- :doc:`fractional_zener_sl` and :doc:`fractional_zener_ll` — related Zener families with
  different equilibrium behavior.
- :doc:`fractional_kv_zener` — Kelvin-based creep analogue for bounded compliance solids.
- :doc:`fractional_maxwell_gel` — when the equilibrium spring is negligible.
- :doc:`../../transforms/fft` — generate :math:`G'(\omega)` and :math:`G''(\omega)` inputs
  required for fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — case studies across
  fractional Zener solids.
- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing between models.

References
----------

.. [1] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [2] Koeller, R. C. "Applications of fractional calculus to the theory of
   viscoelasticity." *Journal of Applied Mechanics*, 51, 299–307 (1984).
   https://doi.org/10.1115/1.3167616

.. [3] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

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

.. [7] Metzler, R., and Nonnenmacher, T. F. "Fractional relaxation processes and
   fractional rheological models for the description of a class of viscoelastic
   materials." *International Journal of Plasticity*, 19, 941–959 (2003).
   https://doi.org/10.1016/S0749-6419(02)00087-6

.. [8] Schiessel, H., and Blumen, A. "Hierarchical analogues to fractional
   relaxation equations." *Journal of Physics A*, 26, 5057–5069 (1993).
   https://doi.org/10.1088/0305-4470/26/19/034

.. [9] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application
   of fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [10] Heymans, N., and Bauwens, J. C. "Fractal rheological models and fractional
    differential equations for viscoelastic behavior."
    *Rheologica Acta*, 33, 210–219 (1994).
    https://doi.org/10.1007/BF00437306
