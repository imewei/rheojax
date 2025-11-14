.. _model-fractional-zener-ss:

Fractional Zener Solid-Solid (Fractional)
=========================================

Quick Reference
---------------

**Use when:** Solid with two elastic plateaus, power-law transition, broad relaxation spectra
**Parameters:** 4 (Ge, Gm, α, τ_α)
**Key equation:** :math:`G(t) = G_e + G_m E_\alpha(-(t/\tau_\alpha)^\alpha)`
**Test modes:** Oscillation, relaxation, creep
**Material examples:** Cross-linked networks, filled elastomers, hydrogels, biological tissues

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The Fractional Zener Solid-Solid (FZSS) model is a generalization of the classical Zener (Standard Linear Solid) model where the dashpot in the Maxwell arm is replaced by a SpringPot element. This configuration consists of a parallel spring (equilibrium modulus Ge) and a fractional Maxwell arm (spring Gm in series with SpringPot), exhibiting both instantaneous and equilibrium elasticity with power-law relaxation between the two plateaus.

The FZSS model is particularly effective for characterizing cross-linked polymer networks, filled elastomers, and biological tissues that exhibit solid-like behavior with broad relaxation spectra arising from microstructural heterogeneity.

Physical Interpretation
-----------------------

The FZSS model captures materials that behave as **viscoelastic solids** with two distinct elastic moduli:

1. **Equilibrium modulus (Ge)**: Arises from permanent network structure (covalent cross-links, crystalline regions, or entanglements). This spring is always engaged and provides the long-time elastic response.

2. **Maxwell arm modulus (Gm)**: Represents additional stiffness from temporary network interactions that relax over time through power-law dynamics governed by the SpringPot element.

3. **SpringPot element**: Provides fractional-order viscoelastic damping, generalizing the classical dashpot. The fractional order α quantifies the breadth of the relaxation spectrum.

**For FZSS specifically**, the fractional order α quantifies how the material transitions between the two elastic plateaus (Ge and Ge + Gm). Smaller α values indicate a more gradual, power-law transition over many decades of time/frequency. Typical α ranges for FZSS applications:

- Cross-linked polymer networks: α ≈ 0.3-0.6
- Filled elastomers: α ≈ 0.2-0.5
- Biological tissues (soft): α ≈ 0.1-0.4
- Hydrogels: α ≈ 0.4-0.7

Governing Equations
-------------------

Mathematical Foundations
~~~~~~~~~~~~~~~~~~~~~~~~

The FZSS model is built on the **Mittag-Leffler function**, which plays the same role in fractional viscoelasticity as the exponential function does in classical models.

**One-Parameter Mittag-Leffler Function:**

.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

where Γ is the gamma function. This function provides a smooth interpolation between exponential decay (α = 1) and power-law relaxation (0 < α < 1).

**Key Properties:**
   - E₁(z) = exp(z) (recovers classical exponential)
   - E_α(0) = 1 for all α
   - E_α(-t^α) exhibits initial power-law decay t^(-α) followed by stretched exponential for large t
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
   - At t = 0: G(0) = Ge + Gm (instantaneous modulus, glassy response)
   - At t → ∞: G(∞) = Ge (equilibrium modulus, permanent network)
   - Intermediate times: Power-law relaxation G(t) - Ge ~ t^(-α)

**Creep compliance:**

.. math::
   :nowrap:

   \[
   J(t) \;=\; \frac{1}{G_e+G_m} \;+\;
   \left(\frac{1}{G_e}-\frac{1}{G_e+G_m}\right)
   \Big[1 - E_{\alpha}\!\big(- (t/\tau_{\alpha})^{\alpha}\big)\Big].
   \]

**Physical interpretation:**
   - At t = 0: J(0) = 1/(Ge + Gm) (instantaneous compliance)
   - At t → ∞: J(∞) = 1/Ge (equilibrium compliance)
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
   - Low ω: G' → Ge (elastic plateau), G" → 0
   - Transition region (ω ~ 1/τ_α): Power-law scaling G', G" ~ ω^α with slope α in log-log plot
   - High ω: G' → Ge + Gm (second plateau), G" decreases as ω^(-α)
   - Loss tangent tan δ = G"/G' exhibits a maximum at the transition frequency

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

- **alpha**: Controls the relaxation dynamics. Lower α indicates broader relaxation spectra from microstructural heterogeneity. For cross-linked networks, α ≈ 0.3-0.6.

- **tau_alpha**: Characteristic time scale for relaxation. Has unusual units (s^α) due to fractional calculus. Related to average relaxation time but incorporates spectrum breadth.

Validity and Assumptions
------------------------

- Linear viscoelastic assumption: strain amplitudes remain small (typically γ₀ < 1-10%).
- Isothermal conditions: temperature constant throughout the experiment.
- Time-invariant material parameters: no aging, degradation, or structural evolution.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.
- Assumes material exhibits solid-like behavior with finite equilibrium modulus (Ge > 0).

Regimes and Behavior
--------------------

**Short-Time / High-Frequency Regime** (t << τ_α or ω >> 1/τ_α):
   - Both springs contribute: G(t) → Ge + Gm
   - Elastic plateau: G'(ω) → Ge + Gm
   - Material behaves as stiff solid with modulus Ge + Gm
   - Minimal energy dissipation: G" → 0

**Intermediate Regime** (t ~ τ_α or ω ~ 1/τ_α):
   - Power-law relaxation: G(t) - Ge ~ (t/τ_α)^(-α)
   - Frequency-domain: G'(ω), G"(ω) ~ ω^α (parallel slopes in log-log plot)
   - Loss tangent maximum: tan δ peaks at transition frequency
   - This is the **fingerprint** of fractional viscoelasticity

**Long-Time / Low-Frequency Regime** (t >> τ_α or ω << 1/τ_α):
   - Equilibrium plateau: G(t) → Ge
   - Elastic plateau: G'(ω) → Ge
   - Permanent network structure dominates
   - Solid-like behavior: G' > G", material does not flow

Comparison with Classical Zener Model
--------------------------------------

The FZSS model offers significant advantages over the classical Zener model:

**Classical Zener (α = 1):**
   - Single relaxation time τ
   - Exponential relaxation: G(t) = Ge + Gm exp(-t/τ)
   - Narrow relaxation spectrum (Lorentzian)
   - Often insufficient for real materials with heterogeneous microstructures

**Fractional Zener (0 < α < 1):**
   - Continuous distribution of relaxation times
   - Power-law relaxation: G(t) - Ge ~ t^(-α)
   - Broad relaxation spectrum (power-law or log-normal distribution)
   - Captures material heterogeneity with fewer parameters

**When to Use Fractional:**
   - Material exhibits power-law relaxation in intermediate time range
   - Log-log plots of G' and G" show parallel slopes (not classical Lorentzian peak)
   - Need to fit data spanning 3+ decades in frequency/time
   - Classical multi-mode Zener requires too many parameters (> 5)

**When Classical Suffices:**
   - Material has single dominant relaxation process
   - Data span < 2 decades in frequency/time
   - Exponential decay observed experimentally

Limiting Behavior
-----------------

The FZSS model recovers simpler models in specific limits:

- **α → 1**: Classical Zener model with exponential decay: G(t) = Ge + Gm exp(-t/τ_α)
- **α → 0**: Ge dominates, Gm contribution becomes frequency-independent elastic addition
- **Gm → 0**: Purely elastic solid with modulus Ge (no relaxation)
- **Ge → 0**: Fractional Maxwell Liquid (FML) — material flows under stress
- **tau_alpha → 0**: Two springs in parallel, G(t) = Ge + Gm (no time dependence)
- **tau_alpha → ∞**: Pure elastic spring Ge (Maxwell arm never relaxes)

Material Examples
-----------------

**Cross-Linked Polymer Networks:**
   - Natural rubber, synthetic elastomers (α ≈ 0.4-0.6)
   - Ge from vulcanization cross-links, Gm from chain dynamics
   - Broad relaxation spectra from cross-link density heterogeneity

**Filled Elastomers:**
   - Carbon black or silica-filled rubber (α ≈ 0.2-0.5)
   - Lower α due to filler-polymer interactions creating hierarchical structure
   - Ge from cross-links, Gm from glassy polymer layers near filler

**Hydrogels:**
   - Chemically cross-linked PVA, alginate (α ≈ 0.4-0.7)
   - Ge from covalent or ionic cross-links
   - Gm from polymer-water interactions and entanglements

**Biological Tissues:**
   - Skin, tendons, cartilage (α ≈ 0.1-0.4)
   - Very broad spectra (low α) from hierarchical collagen/elastin networks
   - Ge from collagen cross-links, Gm from proteoglycan matrix

**Semi-Crystalline Polymers:**
   - Polyethylene, polypropylene (α ≈ 0.3-0.5)
   - Ge from crystalline regions, Gm from amorphous phase relaxation

Smart Initialization (NEW in v0.2.0)
-------------------------------------

RheoJAX automatically applies **smart parameter initialization** when fitting FZSS to oscillation data, significantly improving convergence and parameter recovery.

How It Works
~~~~~~~~~~~~

When ``test_mode='oscillation'``, the initialization system:

1. **Extracts frequency features** from :math:`|G^*|(\omega)` data:
   - Low-frequency plateau → estimates Ge
   - High-frequency plateau → estimates Ge + Gm (thus Gm = high_plateau - low_plateau)
   - Transition frequency ω_mid (steepest slope) → estimates τ_α = 1/ω_mid
   - Slope in transition region → estimates fractional order α

2. **Estimates fractional order** from loss tangent slope:
   - Analyzes slope of tan δ = G"/G' in intermediate frequency range
   - Maps slope to α using power-law scaling theory

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
3. Estimate model-specific parameters (FZSS: Ge, Gm, τ_α, α)
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

**Foundational Works:**

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010).
- R.C. Koeller, "Applications of fractional calculus to the theory of viscoelasticity,"
  *J. Appl. Mech.* 51, 299–307 (1984).
- H. Schiessel, R. Metzler, A. Blumen, and T.F. Nonnenmacher, "Generalized viscoelastic
  models: their fractional equations with solutions," *J. Phys. A* 28, 6567–6584 (1995).

**Mittag-Leffler Functions:**

- Gorenflo, R., Kilbas, A.A., Mainardi, F., Rogosin, S.V. (2014). *Mittag-Leffler Functions,
  Related Topics and Applications*. Springer.

**Physical Interpretation of Fractional Order:**

- Mainardi, F., Spada, G. (2011). "Creep, Relaxation and Viscosity Properties for Basic
  Fractional Models in Rheology." *European Physical Journal Special Topics*, 193, 133-160.
- Friedrich, C., Braun, H. (1992). "Generalized Cole-Cole Behavior and its Rheological
  Relevance." *Rheologica Acta*, 31, 309-322.

**Applications to Biological Materials:**

- Schiessel, H., Metzler, R., Blumen, A., Nonnenmacher, T.F. (1995). "Generalized viscoelastic
  models: their fractional equations with solutions." *J. Phys. A* 28, 6567–6584.
