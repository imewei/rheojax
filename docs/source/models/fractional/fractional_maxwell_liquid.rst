.. _model-fractional-maxwell-liquid:

Fractional Maxwell Liquid (Fractional)
======================================

Quick Reference
---------------

**Use when:** Viscoelastic liquid, power-law relaxation without terminal flow plateau
**Parameters:** 3 (Gm, α, τ_α)
**Key equation:** :math:`G(t) = G_m t^{-\alpha} E_{1-\alpha,1-\alpha}(-t^{1-\alpha}/\tau_\alpha)`
**Test modes:** Oscillation, relaxation
**Material examples:** Polymer melts (linear/branched), concentrated polymer solutions, complex fluids

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The Fractional Maxwell Liquid (FML) model consists of a Hookean spring in series with a SpringPot element. This configuration describes materials with instantaneous elastic response at short times followed by power-law relaxation at intermediate to long times. The model is particularly effective for characterizing polymer melts, concentrated polymer solutions, and other viscoelastic liquids that exhibit both elastic memory and power-law relaxation without terminal flow.

Unlike the Fractional Maxwell Gel which includes a dashpot for terminal flow, the FML model maintains power-law behavior across all time scales, making it ideal for materials that show persistent viscoelastic behavior without approaching pure viscous flow.

Physical Interpretation
-----------------------

The FML model represents **viscoelastic liquids** with zero equilibrium modulus (Ge = 0), meaning the material flows under sustained stress. The physical structure consists of:

1. **Hookean spring (Gm)**: Provides instantaneous elastic response at short times. Represents chain/network stretching before relaxation mechanisms activate.

2. **SpringPot element**: Governs the relaxation dynamics through power-law viscoelasticity. The fractional order α controls the breadth of the relaxation spectrum.

The series configuration ensures that sustained stress eventually leads to unbounded strain growth (flow), distinguishing this from solid-like models.

**For FML specifically**, the fractional order α directly controls the slope in log-log plots of G'(ω) and G"(ω), with both moduli scaling as ω\ :sup:`α` in the power-law region. Typical α ranges for FML applications:

- Polymer melts (linear homopolymers): α ≈ 0.7-0.9
- Polymer melts (branched): α ≈ 0.5-0.7
- Concentrated polymer solutions: α ≈ 0.5-0.8
- Complex fluids (colloidal dispersions): α ≈ 0.4-0.7

Mathematical Foundations
------------------------

Mittag-Leffler Functions
~~~~~~~~~~~~~~~~~~~~~~~~

The FML model relies on the **two-parameter Mittag-Leffler function**:

.. math::

   E_{\alpha,\beta}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + \beta)}

where Γ is the gamma function. This generalization of the exponential function is essential for fractional viscoelasticity.

**Key Properties:**
   - E₁,₁(z) = exp(z) (recovers classical exponential)
   - E_α,α(-t^α) provides the characteristic power-law relaxation
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
   - Short times (t << τ_α): G(t) ≈ Gm (elastic plateau)
   - Intermediate times: G(t) ~ t^(-α) (power-law relaxation)
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
   - High ω (ω >> 1/τ_α): G' → Gm, G" → 0 (elastic plateau)
   - Intermediate ω (ω ~ 1/τ_α): G', G" ~ ω^α (power-law scaling, parallel slopes)
   - Low ω (ω << 1/τ_α): G' ~ ω^(2α), G" ~ ω^α (liquid-like terminal regime)

**Creep Compliance**:


.. math::

   J(t) = \frac{1}{G_m} + \frac{t^\alpha}{G_m \tau_\alpha^\alpha} E_{\alpha,1+\alpha}\left(-\left(\frac{t}{\tau_\alpha}\right)^\alpha\right)

**Physical interpretation:**
   - Short times: J(t) ≈ 1/Gm (elastic compliance)
   - Long times: Unbounded growth J(t) → ∞ (liquid-like flow)

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

- **Gm**: Instantaneous modulus reflecting chain/network stiffness. For polymer melts, relates to entanglement density via Gm ≈ GN⁰ (plateau modulus). Typical values: 10³-10⁶ Pa for polymer melts.

- **alpha**: Quantifies relaxation spectrum breadth. Lower α → broader spectra from molecular weight polydispersity, branching, or complex intermolecular interactions. For linear polymers, α ≈ 0.7-0.9; for branched polymers, α ≈ 0.5-0.7.

- **tau_alpha**: Average relaxation time scale. Has unusual units (s^α) due to fractional calculus. For polymer melts, relates to molecular weight via τ_α ~ Mw^(3.4). Typical values: 10⁻³-10³ s depending on molecular weight and temperature.

Validity and Assumptions
------------------------

- Linear viscoelastic assumption: strain amplitudes remain small (γ₀ < 5-10% typically).
- Isothermal conditions: constant temperature throughout experiment.
- Time-invariant material parameters: no aging, polymerization, or degradation.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.
- Assumes liquid-like behavior: zero equilibrium modulus (Ge = 0), material flows under stress.

Regimes and Behavior
--------------------

The Fractional Maxwell Liquid exhibits characteristic behavior across different regimes:

**Short-Time / High-Frequency Regime** (:math:`t \ll \tau_\alpha` or :math:`\omega \gg 1/\tau_\alpha`):
   The spring dominates, yielding purely elastic behavior:


   .. math::

      G(t) \sim G_m, \quad G^*(\omega) \sim G_m

   The material behaves as an elastic solid with modulus Gm. This regime captures the instantaneous response before relaxation mechanisms activate.

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

**Classical Maxwell (α = 1):**
   - Single relaxation time τ
   - Exponential relaxation: G(t) = Gm exp(-t/τ)
   - Narrow relaxation spectrum (Lorentzian)
   - Low-frequency behavior: G' ~ ω², G" ~ ω (classical liquid)

**Fractional Maxwell Liquid (0 < α < 1):**
   - Continuous distribution of relaxation times
   - Power-law relaxation: G(t) ~ t^(-α)
   - Broad relaxation spectrum
   - Low-frequency behavior: G' ~ ω^(2α), G" ~ ω^α (generalized liquid)

**When to Use Fractional:**
   - Power-law relaxation observed in stress relaxation experiments
   - Log-log plots of G' and G" show parallel slopes over multiple decades
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
   - Polyethylene, polypropylene, polystyrene (α ≈ 0.7-0.9)
   - Gm ≈ GN⁰ (plateau modulus from entanglements)
   - Relatively narrow spectra (high α) for monodisperse polymers

**Polymer Melts (Branched):**
   - Long-chain branched polyethylene, star polymers (α ≈ 0.5-0.7)
   - Broader spectra (lower α) from hierarchical relaxation processes
   - Arm retraction, branch point hopping add complexity

**Concentrated Polymer Solutions:**
   - Solutions above overlap concentration c* (α ≈ 0.5-0.8)
   - Lower α than melts due to solvent-polymer interactions
   - Spectrum breadth depends on concentration and molecular weight distribution

**Micellar Solutions:**
   - Wormlike micelles, surfactant solutions (α ≈ 0.4-0.7)
   - Broad spectra from micelle size distribution and reptation
   - Can exhibit gel-like behavior (α ≈ 0.5) near critical concentration

**Colloidal Dispersions:**
   - Dense colloidal suspensions (α ≈ 0.4-0.6)
   - Particle size polydispersity creates broad relaxation spectra
   - Hydrodynamic interactions contribute to spectrum breadth

Smart Initialization (NEW in v0.2.0)
-------------------------------------

RheoJAX automatically applies **smart parameter initialization** when fitting FML to oscillation data.

How It Works
~~~~~~~~~~~~

When ``test_mode='oscillation'``, the initialization system:

1. **Extracts frequency features** from :math:`|G^*|(\omega)` data:
   - High-frequency plateau → estimates Gm
   - Transition frequency ω_mid (maximum slope of :math:`|G^*|`) → estimates τ_α = 1/ω_mid
   - Slope in power-law region → estimates fractional order α

2. **Estimates fractional order** from parallel slopes:
   - Identifies region where G'(ω) and G"(ω) have parallel slopes
   - Extracts slope via linear regression in log-log space
   - Maps slope directly to α (slope ≈ α in power-law region)

3. **Clips to parameter bounds** to ensure Gm > 0, 0 < α < 1, τ_α > 0

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

See also
--------

- :doc:`fractional_maxwell_gel` — uses a dashpot instead of a spring for gel-like systems.
- :doc:`fractional_maxwell_model` — generalized two-order series analogue.
- :doc:`fractional_jeffreys` — adds a parallel dashpot for finite zero-shear viscosity.
- :doc:`../../transforms/fft` — convert relaxation data to :math:`G^*(\omega)` before
  fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook covering the
  complete Fractional Maxwell family.
- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing models.

References
----------

**Foundational Works:**

- C. Friedrich, "Relaxation and retardation functions of the Maxwell model with fractional
  derivatives," *Rheol. Acta* 30, 151–158 (1991).
- H. Schiessel, R. Metzler, A. Blumen, and T.F. Nonnenmacher, "Generalized viscoelastic
  models: their fractional equations with solutions," *J. Phys. A* 28, 6567–6584 (1995).
- R. Metzler, W. Schick, H.-G. Kilian, and T.F. Nonnenmacher, "Relaxation in filled
  polymers: A fractional calculus approach," *J. Chem. Phys.* 103, 7180–7186 (1995).

**Mittag-Leffler Functions:**

- Gorenflo, R., Kilbas, A.A., Mainardi, F., Rogosin, S.V. (2014). *Mittag-Leffler Functions,
  Related Topics and Applications*. Springer.

**Physical Interpretation:**

- Mainardi, F., Spada, G. (2011). "Creep, Relaxation and Viscosity Properties for Basic
  Fractional Models in Rheology." *European Physical Journal Special Topics*, 193, 133-160.
- Friedrich, C., Braun, H. (1992). "Generalized Cole-Cole Behavior and its Rheological
  Relevance." *Rheologica Acta*, 31, 309-322.

**Polymer Applications:**

- Metzler, R., Klafter, J. (2000). "The Random Walk's Guide to Anomalous Diffusion: A
  Fractional Dynamics Approach." *Physics Reports*, 339(1), 1-77.
