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
   Ratio of material relaxation time to observation time: :math:`\text{De} = \tau / t_{\text{obs}}`

Moduli and Mechanical Properties
---------------------------------

**Storage Modulus** (:math:`G'`)
   Elastic component of complex modulus; energy stored per cycle (Pa).

**Loss Modulus** (:math:`G''`)
   Viscous component of complex modulus; energy dissipated per cycle (Pa).

**Complex Modulus** (:math:`G^*`)
   :math:`G^* = G' + iG''`; combines storage and loss moduli.

**Loss Tangent** (:math:`\tan\delta`)
   Ratio of loss to storage modulus: :math:`\tan\delta = G''/G'`; damping factor.

**Relaxation Modulus** (:math:`G(t)`)
   Stress response to step strain as a function of time (Pa).

**Equilibrium Modulus** (:math:`G_e`)
   Long-time plateau in relaxation modulus (viscoelastic solids, Pa).

**Compliance** (:math:`J`)
   Inverse of modulus; strain per unit stress (:math:`\text{Pa}^{-1}`).

**Young's Modulus** (:math:`E`)
   Tensile (axial) modulus; ratio of tensile stress to tensile strain (Pa).

**Complex Young's Modulus** (:math:`E^*`)
   :math:`E^* = E' + iE''`; tensile analog of the complex shear modulus :math:`G^*`.

**Storage Modulus (tensile)** (:math:`E'`)
   In-phase (elastic) component of the complex Young's modulus from DMTA (Pa).

**Loss Modulus (tensile)** (:math:`E''`)
   Out-of-phase (viscous) component of the complex Young's modulus from DMTA (Pa).

**Poisson's Ratio** (:math:`\nu`)
   Ratio of transverse to axial strain; relates :math:`E` and :math:`G` via :math:`E = 2(1+\nu)G`. Typical values: rubber :math:`\approx 0.5`, glassy polymer :math:`\approx 0.35`.

DMTA / DMA
----------

**DMTA (Dynamic Mechanical Thermal Analysis)**
   Oscillatory technique measuring :math:`E^*(\omega, T)` under tensile, bending, or compression deformation. Widely used for polymer glass transitions and temperature sweeps.

**DMA (Dynamic Mechanical Analysis)**
   Synonym for DMTA; sometimes specifically refers to isothermal frequency sweeps in tensile geometry.

**Deformation Mode**
   The type of mechanical loading applied to the sample: shear, tension, bending, or compression. In RheoJAX, set via ``deformation_mode='tension'`` in ``fit()`` / ``predict()``.

Viscosity
---------

**Viscosity** (:math:`\eta`)
   Resistance to flow under shear; stress/shear-rate ratio (Pa·s).

**Zero-Shear Viscosity** (:math:`\eta_0`)
   Viscosity at vanishingly small shear rates (Newtonian plateau, Pa·s).

**Complex Viscosity** (:math:`\eta^*`)
   Frequency-dependent viscosity from SAOS: :math:`\eta^* = |G^*|/\omega` (Pa·s).

**Shear Thinning**
   Decrease in viscosity with increasing shear rate (pseudoplastic).

**Shear Thickening**
   Increase in viscosity with increasing shear rate (dilatant).

Timescales
----------

**Relaxation Time** (:math:`\tau`)
   Characteristic timescale for stress to decay to :math:`1/e` (~37%) of initial value (s).

**Crossover Frequency** (:math:`\omega_c`)
   Frequency where :math:`G' = G''`; related to relaxation time by :math:`\omega_c \approx 1/\tau` (rad/s).

Fractional Parameters
---------------------

**Fractional Order** (:math:`\alpha`)
   Exponent in fractional derivative; characterizes breadth of relaxation spectrum (:math:`0 < \alpha < 1`).

**SpringPot**
   Fractional viscoelastic element interpolating between spring (:math:`\alpha=0`) and dashpot (:math:`\alpha=1`).

**Mittag-Leffler Function** (:math:`E_\alpha`)
   Generalization of exponential function for fractional models.

Test Modes
----------

**SAOS (Small-Amplitude Oscillatory Shear)**
   Sinusoidal strain input; measures :math:`G'(\omega)` and :math:`G''(\omega)` in frequency domain.

**Stress Relaxation**
   Step strain input; measures :math:`G(t)` in time domain.

**Creep**
   Step stress input; measures compliance :math:`J(t)` in time domain.

**Steady Shear Flow**
   Constant shear rate; measures viscosity :math:`\eta(\dot{\gamma})` (nonlinear regime).

Material Types
--------------

**Viscoelastic Liquid**
   Material with zero equilibrium modulus; flows at long times (:math:`G'' > G'` at low :math:`\omega`).

**Viscoelastic Solid**
   Material with finite equilibrium modulus; does not flow (:math:`G' > G''` everywhere).

**Gel**
   Material with power-law relaxation (:math:`G' \approx G'' \sim \omega^\alpha`).

**Yield Stress** (:math:`\sigma_y`)
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
   Shear thinning/thickening flow model: :math:`\eta = K\dot{\gamma}^{n-1}`.

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
   * - :math:`G'`
     - Storage modulus
     - Pa
     - :math:`10^2 - 10^8`
   * - :math:`G''`
     - Loss modulus
     - Pa
     - :math:`10^2 - 10^8`
   * - :math:`\eta`
     - Viscosity
     - Pa·s
     - :math:`10^{-3} - 10^{10}`
   * - :math:`\tau`
     - Relaxation time
     - s
     - :math:`10^{-6} - 10^4`
   * - :math:`\alpha`
     - Fractional order
     - dimensionless
     - 0 - 1
   * - :math:`\omega`
     - Angular frequency
     - rad/s
     - :math:`10^{-2} - 10^3`
   * - :math:`\gamma`
     - Strain
     - dimensionless
     - 0 - 1
   * - :math:`\dot{\gamma}`
     - Shear rate
     - :math:`\text{s}^{-1}`
     - :math:`10^{-3} - 10^3`
   * - :math:`\sigma`
     - Stress
     - Pa
     - :math:`10^0 - 10^6`
   * - :math:`E'`
     - Storage modulus (tensile)
     - Pa
     - :math:`10^6 - 10^{10}`
   * - :math:`E''`
     - Loss modulus (tensile)
     - Pa
     - :math:`10^6 - 10^{10}`
   * - :math:`\nu`
     - Poisson's ratio
     - dimensionless
     - 0.3 - 0.5

See Also
--------

- :doc:`../01_fundamentals/parameter_interpretation` — Physical meaning of parameters
- :doc:`../01_fundamentals/what_is_rheology` — Core rheology concepts
- :doc:`../02_model_usage/model_families` — Model descriptions
