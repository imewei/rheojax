.. _glossary:

Glossary of Rheological Terms
==============================

This glossary defines 50+ key terms used throughout the RheoJAX documentation.

Core Concepts
-------------

**Rheology**
   The study of deformation and flow of matter under applied stress.

**Viscoelasticity**
   Material behavior exhibiting both elastic (solid-like) and viscous (liquid-like) characteristics.

**Linear Viscoelasticity**
   Regime where stress is proportional to strain (small deformations).

**Deborah Number (De)**
   Ratio of material relaxation time to observation time: De = τ/t_obs

Moduli and Mechanical Properties
---------------------------------

**Storage Modulus (G')**
   Elastic component of complex modulus; energy stored per cycle (Pa).

**Loss Modulus (G")**
   Viscous component of complex modulus; energy dissipated per cycle (Pa).

**Complex Modulus (G\\*)**
   G\\* = G' + iG"; combines storage and loss moduli.

**Loss Tangent (tan δ)**
   Ratio of loss to storage modulus: tan δ = G"/G'; damping factor.

**Relaxation Modulus (G(t))**
   Stress response to step strain as a function of time (Pa).

**Equilibrium Modulus (G_e)**
   Long-time plateau in relaxation modulus (viscoelastic solids, Pa).

**Compliance (J)**
   Inverse of modulus; strain per unit stress (Pa⁻¹).

Viscosity
---------

**Viscosity (η)**
   Resistance to flow under shear; stress/shear-rate ratio (Pa·s).

**Zero-Shear Viscosity (η₀)**
   Viscosity at vanishingly small shear rates (Newtonian plateau, Pa·s).

**Complex Viscosity (η\\*)**
   Frequency-dependent viscosity from SAOS: η\\* = \|G\\*\|/ω (Pa·s).

**Shear Thinning**
   Decrease in viscosity with increasing shear rate (pseudoplastic).

**Shear Thickening**
   Increase in viscosity with increasing shear rate (dilatant).

Timescales
----------

**Relaxation Time (τ)**
   Characteristic timescale for stress to decay to 1/e (~37%) of initial value (s).

**Crossover Frequency (ω_c)**
   Frequency where G' = G"; related to relaxation time by ω_c ≈ 1/τ (rad/s).

Fractional Parameters
---------------------

**Fractional Order (α)**
   Exponent in fractional derivative; characterizes breadth of relaxation spectrum (0 < α < 1).

**SpringPot**
   Fractional viscoelastic element interpolating between spring (α=0) and dashpot (α=1).

**Mittag-Leffler Function (E_α)**
   Generalization of exponential function for fractional models.

Test Modes
----------

**SAOS (Small-Amplitude Oscillatory Shear)**
   Sinusoidal strain input; measures G'(ω) and G"(ω) in frequency domain.

**Stress Relaxation**
   Step strain input; measures G(t) in time domain.

**Creep**
   Step stress input; measures compliance J(t) in time domain.

**Steady Shear Flow**
   Constant shear rate; measures viscosity η(γ̇) (nonlinear regime).

Material Types
--------------

**Viscoelastic Liquid**
   Material with zero equilibrium modulus; flows at long times (G" > G' at low ω).

**Viscoelastic Solid**
   Material with finite equilibrium modulus; does not flow (G' > G" everywhere).

**Gel**
   Material with power-law relaxation (G' ≈ G" ~ ω^α).

**Yield Stress (σ_y)**
   Critical stress below which material behaves as solid, above which it flows (Pa).

Models
------

**Maxwell Model**
   Simplest viscoelastic liquid model; single relaxation time.

**Zener Model (SLS)**
   Standard Linear Solid; viscoelastic solid with single relaxation time.

**Fractional Models**
   Models using fractional derivatives to capture distributed relaxation spectra.

**PowerLaw Model**
   Shear thinning/thickening flow model: η = K γ̇^(n-1).

Experimental
------------

**Mastercurve**
   Superposition of multi-temperature data to extend frequency range.

**Time-Temperature Superposition (TTS)**
   Principle that temperature and timescale are equivalent for thermorheologically simple materials.

**WLF Equation**
   Williams-Landel-Ferry equation for temperature-dependent shift factors.

Statistical (Bayesian)
----------------------

**Posterior Distribution**
   Probability distribution of parameters given data and priors.

**Credible Interval**
   Bayesian analog of confidence interval; range containing parameter with specified probability.

**MCMC (Markov Chain Monte Carlo)**
   Sampling method for Bayesian inference.

**NUTS (No-U-Turn Sampler)**
   Efficient MCMC algorithm (Hamiltonian Monte Carlo variant).

**ESS (Effective Sample Size)**
   Number of independent samples in MCMC chain; measure of sampling efficiency.

**R-hat (Gelman-Rubin Statistic)**
   Convergence diagnostic for MCMC (should be < 1.01).

Computational
-------------

**NLSQ**
   Nonlinear Least Squares optimization backend (GPU-accelerated).

**JAX**
   Just-After-eXecution library for automatic differentiation and GPU acceleration.

**JIT (Just-In-Time Compilation)**
   Runtime compilation for performance optimization.

Symbols Quick Reference
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 20 30

   * - Symbol
     - Meaning
     - Units
     - Typical Range
   * - G'
     - Storage modulus
     - Pa
     - 10² - 10⁸
   * - G"
     - Loss modulus
     - Pa
     - 10² - 10⁸
   * - η
     - Viscosity
     - Pa·s
     - 10⁻³ - 10¹⁰
   * - τ
     - Relaxation time
     - s
     - 10⁻⁶ - 10⁴
   * - α
     - Fractional order
     - dimensionless
     - 0 - 1
   * - ω
     - Angular frequency
     - rad/s
     - 10⁻² - 10³
   * - γ
     - Strain
     - dimensionless
     - 0 - 1
   * - γ̇
     - Shear rate
     - s⁻¹
     - 10⁻³ - 10³
   * - σ
     - Stress
     - Pa
     - 10⁰ - 10⁶

See Also
--------

- :doc:`../01_fundamentals/parameter_interpretation` — Physical meaning of parameters
- :doc:`../01_fundamentals/what_is_rheology` — Core rheology concepts
- :doc:`../02_model_usage/model_families` — Model descriptions
