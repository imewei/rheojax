.. _sgr_analysis:

==========================================
Soft Glassy Rheology (SGR) Analysis
==========================================

**Statistical Mechanics Framework for Complex Soft Materials**

Overview
========

**Soft Glassy Rheology (SGR)** is a statistical mechanics framework developed by Sollich
and coworkers (1997-1998) that unifies the rheological behavior of diverse complex fluids—
foams, emulsions, pastes, colloidal glasses, and other metastable "soft glassy" materials—
under a single theoretical picture.

.. admonition:: Key Insight
   :class: tip

   SGR treats soft materials as collections of mesoscopic **trapped elements** that can
   yield and rearrange via an activated hopping process. The single dimensionless parameter
   **x** (effective noise temperature) controls whether the material behaves as a glass
   (x < 1), a power-law fluid (1 < x < 2), or a Newtonian liquid (x ≥ 2).

Unlike phenomenological models (Maxwell, Kelvin-Voigt) that describe material response
without microscopic interpretation, SGR provides a **physical picture** connecting
macroscopic rheology to mesoscopic rearrangement dynamics.

When to Use SGR
---------------

**SGR is ideal for:**

- **Soft glassy materials**: Foams, concentrated emulsions, pastes, slurries, colloidal gels
- **Yield stress fluids**: Materials with solid-to-liquid transitions under stress
- **Aging materials**: Systems whose properties evolve over time (x < 1)
- **Power-law rheology**: Materials showing G' ~ G'' ~ ω^n across broad frequency ranges
- **Shear rejuvenation**: Systems fluidized by deformation
- **Phase classification**: Determining whether a material is a glass, fluid, or near transition

**SGR complements (rather than replaces):**

- **Fractional models**: For mathematical convenience without microscopic picture
- **Herschel-Bulkley**: For simpler steady-shear yield stress fitting
- **Multi-mode Maxwell**: For discrete relaxation time spectra

Material Classification
-----------------------

The SGR framework classifies materials by their effective noise temperature x:

.. list-table:: SGR Phase Diagram
   :header-rows: 1
   :widths: 15 25 60

   * - Regime
     - x Range
     - Physical Characteristics
   * - **Glass**
     - x < 1
     - Aging, yield stress, metastable, non-ergodic
   * - **Transition**
     - x ≈ 1
     - Critical behavior, G' ≈ G'', borderline yield
   * - **Power-law Fluid**
     - 1 < x < 2
     - G' ~ G'' ~ ω^(x-1), viscoelastic liquid
   * - **Newtonian**
     - x ≥ 2
     - Exponential relaxation, simple viscous flow

The **glass transition** at x = 1 is not a thermodynamic phase transition but a
**dynamical arrest**: below x = 1, the system cannot equilibrate on any finite timescale.

Theoretical Foundations
=======================

The Trap Model Picture
----------------------

SGR models the material as many mesoscopic **elements**—local regions containing
multiple particles/droplets/bubbles—each characterized by:

1. **Local strain** l: The strain stored in that element
2. **Trap depth** E: The energy barrier to rearrangement (how "stuck" it is)
3. **Elastic constant** k: Local stiffness (typically equal to G₀)

Elements are trapped in local energy minima ("cages") until activated by noise.
The **exponential trap distribution** ρ(E) = exp(-E) means deep traps are exponentially
rare—a consequence of entropic arguments.

Activation and Yielding
~~~~~~~~~~~~~~~~~~~~~~~

Elements yield with rate:

.. math::

   \Gamma(E, l) = \Gamma_0 \exp\left(-\frac{E - \frac{1}{2}kl^2}{x}\right)

where:
   - Γ₀ = 1/τ₀ is the attempt frequency
   - x is the effective noise temperature
   - ½kl² is the elastic energy (lowers the barrier)

**Key physics**: Large local strain makes yielding more likely by reducing the effective
barrier height.

Effective Noise Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The parameter x is **not** thermal temperature (which is essentially fixed at room
temperature for soft materials). Instead, x represents an **effective noise level**
arising from:

- **Mechanical noise**: Rearrangements of neighbors creating local fluctuations
- **Shear rejuvenation**: Applied deformation increasing x
- **Aging**: Gradual decrease of x as system settles into deeper traps
- **External perturbations**: Vibrations, stirring, etc.

Connecting x to Rheology
~~~~~~~~~~~~~~~~~~~~~~~~

The remarkable power of SGR is that **x alone determines rheological character**:

.. math::

   G'(\omega) \sim G''(\omega) \sim \omega^{x-1} \quad \text{for } 1 < x < 2

This gives:
   - Phase angle: δ = π(x-1)/2 = constant (frequency-independent)
   - Flow index: n = x - 1 in Herschel-Bulkley-like steady shear
   - Loss tangent: tan(δ) = tan(π(x-1)/2) = constant

**How to estimate x from data**:

1. Plot log G' vs log ω
2. Fit linear region to get slope m
3. x ≈ m + 1

Or from loss tangent:

.. math::

   x = 1 + \frac{2}{\pi} \arctan(\tan\delta)

SGR Model Variants
==================

RheoJAX implements two SGR variants:

SGRConventional (Sollich 1998)
------------------------------

The standard SGR model for rheological fitting:

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.fit(omega, G_star, test_mode='oscillation')

   x = model.parameters.get_value('x')
   print(f"Effective temperature: x = {x:.3f}")
   print(f"Phase: {'glass' if x < 1 else 'fluid'}")

**Parameters**: x (noise temperature), G₀ (modulus scale), τ₀ (attempt time)

**Use for**: Standard fitting, phase classification, steady shear flow curves

SGRGeneric (Fuereder & Ilg 2013)
--------------------------------

Thermodynamically consistent version satisfying the GENERIC framework:

.. code-block:: python

   from rheojax.models import SGRGeneric

   model = SGRGeneric()
   model.fit(omega, G_star, test_mode='oscillation')

   # Access thermodynamic functions
   S_prod = model.entropy_production(gamma_dot=1.0)
   print(f"Entropy production: {S_prod:.4f} J/(K·m³·s)")

**Additional features**:
   - Entropy production calculation
   - Free energy landscape
   - Fluctuation-dissipation verification
   - GENERIC structure validation

**Use for**: Thermodynamic research, entropy analysis, theoretical validation

.. note::

   Both models give **identical rheological predictions**. Use SGRConventional for
   standard fitting and SGRGeneric when you need thermodynamic information.

Practical Implementation
========================

Basic SGR Analysis Workflow
---------------------------

.. code-block:: python

   import numpy as np
   from rheojax.models import SGRConventional
   from rheojax.io.readers import auto_read

   # 1. Load oscillatory data
   data = auto_read("frequency_sweep.csv")
   omega = data.x  # Angular frequency
   G_star = data.y  # Complex modulus

   # 2. Fit SGR model
   model = SGRConventional()
   model.fit(omega, G_star, test_mode='oscillation')

   # 3. Extract and interpret parameters
   x = model.parameters.get_value('x')
   G0 = model.parameters.get_value('G0')
   tau0 = model.parameters.get_value('tau0')

   print(f"Effective temperature x = {x:.3f}")
   print(f"Plateau modulus G₀ = {G0:.1f} Pa")
   print(f"Attempt time τ₀ = {tau0:.2e} s")

   # 4. Phase classification
   if x < 0.9:
       print("Material is in GLASS phase (aging, yield stress)")
   elif x < 1.1:
       print("Material is near GLASS TRANSITION")
   elif x < 2.0:
       print("Material is a POWER-LAW FLUID")
   else:
       print("Material is near NEWTONIAN")

   # 5. Predict other test modes
   t = np.logspace(-3, 3, 100)
   G_t = model.predict(t, test_mode='relaxation')  # Stress relaxation

   gamma_dot = np.logspace(-3, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='steady_shear')  # Flow curve

Bayesian Inference for x
------------------------

Since x is the critical parameter determining phase behavior, **Bayesian inference
is highly recommended** to quantify uncertainty:

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.fit(omega, G_star, test_mode='oscillation')

   # Bayesian inference with NLSQ warm-start
   result = model.fit_bayesian(
       omega, G_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000
   )

   # Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   x_low, x_high = intervals['x']

   print(f"x = {model.parameters.get_value('x'):.3f} "
         f"[{x_low:.3f}, {x_high:.3f}] (95% CI)")

   # Phase classification with uncertainty
   if x_high < 1.0:
       print("Material is DEFINITELY in glass phase (95% CI)")
   elif x_low > 1.0:
       print("Material is DEFINITELY in fluid phase (95% CI)")
   else:
       print("Material straddles glass transition (uncertain phase)")

SRFS Transform (Flow Curve Analysis)
------------------------------------

The **Strain-Rate Frequency Superposition (SRFS)** transform creates flow curve
mastercurves analogous to time-temperature superposition:

.. code-block:: python

   from rheojax.transforms import SRFS
   import numpy as np

   # Flow curves at different shear rates
   datasets = [
       {'gamma_dot': gamma_dot_1, 'sigma': sigma_1},
       {'gamma_dot': gamma_dot_2, 'sigma': sigma_2},
       # ...
   ]

   srfs = SRFS(reference_gamma_dot=1.0, auto_shift=True)
   master_curve, shift_factors = srfs.transform(datasets)

   # Shift factor scaling reveals x
   # a(γ̇) ~ γ̇^(2-x) for SGR materials
   slope = np.polyfit(np.log(gamma_dots), np.log(shift_factors), 1)[0]
   x_from_srfs = 2 - slope
   print(f"x from SRFS: {x_from_srfs:.3f}")

Shear Banding Detection
-----------------------

SGR can predict shear banding when the flow curve is non-monotonic:

.. code-block:: python

   from rheojax.transforms import SRFS
   from rheojax.models import SGRConventional

   model = SGRConventional(x=0.8, G0=100.0, tau0=0.01)
   srfs = SRFS()

   # Check for shear banding (non-monotonic flow curve)
   is_banding, critical_rates = srfs.detect_shear_banding(
       model,
       gamma_dot_range=(1e-3, 1e2)
   )

   if is_banding:
       print("Shear banding predicted!")
       low_rate, high_rate = critical_rates
       print(f"Coexisting bands at γ̇ = {low_rate:.2e} and {high_rate:.2e} s⁻¹")

       # Lever rule for band fractions
       band_info = srfs.compute_shear_band_coexistence(model, applied_rate=1.0)
       print(f"High-rate band fraction: {band_info['high_fraction']:.2f}")

Aging Dynamics (x < 1)
----------------------

For glassy materials (x < 1), properties evolve with waiting time:

.. code-block:: python

   from rheojax.models import SGRConventional
   import numpy as np

   # Simulate aging: x approaches 1 from below
   waiting_times = [100, 1000, 10000]  # seconds
   x_values = [0.7, 0.85, 0.95]  # Effective temperature increases toward 1

   for t_w, x in zip(waiting_times, x_values):
       model = SGRConventional(x=x, G0=100.0, tau0=0.01)
       G_star = model.predict(omega, test_mode='oscillation')

       print(f"t_w = {t_w} s: x = {x:.2f}, G'(1 rad/s) = {G_star[25].real:.1f} Pa")

   # Effective relaxation time grows as τ_eff ~ t_w^μ with μ = (1-x)/x
   x = 0.8
   mu = (1 - x) / x
   print(f"Aging exponent μ = {mu:.2f}")

Visualization
-------------

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.fit(omega, G_star_data, test_mode='oscillation')

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # 1. Frequency sweep with fit
   ax = axes[0, 0]
   ax.loglog(omega, np.real(G_star_data), 'o', label="G' data")
   ax.loglog(omega, np.imag(G_star_data), 's', label="G'' data")
   G_fit = model.predict(omega, test_mode='oscillation')
   ax.loglog(omega, np.real(G_fit), '-', label="G' fit")
   ax.loglog(omega, np.imag(G_fit), '--', label="G'' fit")
   ax.set_xlabel('ω (rad/s)')
   ax.set_ylabel('G*, G" (Pa)')
   ax.legend()
   ax.set_title(f'SGR Fit: x = {model.parameters.get_value("x"):.2f}')

   # 2. Loss tangent (should be constant for SGR)
   ax = axes[0, 1]
   tan_delta = np.imag(G_star_data) / np.real(G_star_data)
   ax.semilogx(omega, tan_delta, 'o-')
   x = model.parameters.get_value('x')
   theoretical_tan_delta = np.tan(np.pi * (x - 1) / 2)
   ax.axhline(theoretical_tan_delta, color='r', linestyle='--',
              label=f'SGR prediction: tan(δ) = {theoretical_tan_delta:.2f}')
   ax.set_xlabel('ω (rad/s)')
   ax.set_ylabel('tan(δ)')
   ax.legend()
   ax.set_title('Loss Tangent (Should Be Constant)')

   # 3. Stress relaxation prediction
   ax = axes[1, 0]
   t = np.logspace(-3, 3, 100)
   G_t = model.predict(t, test_mode='relaxation')
   ax.loglog(t, np.real(G_t))
   ax.set_xlabel('t (s)')
   ax.set_ylabel('G(t) (Pa)')
   ax.set_title('Predicted Stress Relaxation')

   # 4. Flow curve prediction
   ax = axes[1, 1]
   gamma_dot = np.logspace(-3, 2, 50)
   sigma = model.predict(gamma_dot, test_mode='steady_shear')
   ax.loglog(gamma_dot, sigma)
   ax.set_xlabel('γ̇ (s⁻¹)')
   ax.set_ylabel('σ (Pa)')
   ax.set_title('Predicted Flow Curve')

   plt.tight_layout()
   plt.savefig('sgr_analysis.png', dpi=150)

SGR vs Other Models
===================

.. list-table:: Model Comparison for Soft Materials
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Best For
     - Limitations
     - Key Parameters
   * - **SGR**
     - Physical interpretation, phase classification, aging
     - Mean-field (no spatial correlations)
     - x (noise temp), G₀, τ₀
   * - **Fractional Maxwell**
     - Mathematical convenience, broad relaxation
     - No microscopic picture
     - G, τ, α (fractional order)
   * - **Herschel-Bulkley**
     - Simple yield stress fitting
     - Steady shear only, no dynamics
     - σ_y, K, n
   * - **Multi-mode Maxwell**
     - Discrete relaxation spectrum
     - Many parameters, no physical basis
     - Gᵢ, τᵢ for each mode

**When to choose SGR over fractional models:**

1. You need **phase classification** (glass vs fluid)
2. Material shows **aging** behavior
3. You want **microscopic interpretation** (trap hopping)
4. Loss tangent is approximately **constant with frequency**

**When to choose fractional models over SGR:**

1. You need **mathematical simplicity**
2. Material doesn't fit SGR predictions
3. You don't need phase classification
4. You're doing **multi-technique fitting** (easier optimization)

Limitations and Caveats
=======================

Mean-Field Approximation
------------------------

SGR neglects spatial correlations between elements. In reality:

- Yielding events trigger neighbors (**avalanches**)
- **Shear banding** involves spatial heterogeneity
- **Localization** effects not captured

These phenomena require spatially-resolved extensions (mode-coupling theory,
elastoplastic models, or mesoscale simulations).

Phenomenological Noise Temperature
----------------------------------

The origin of x is **not derived from first principles**:

- Must be fitted to data or estimated
- Physical mechanism of "noise" unclear
- Cannot predict x from molecular properties alone

For fundamental understanding, pair SGR with molecular dynamics simulations.

Single Element Size
-------------------

Real soft glasses have **polydisperse** element sizes:

- Affects trap distribution shape
- May cause deviations from pure power-law behavior
- Consider polydisperse extensions for better fits

Thixotropy Limitations
----------------------

Basic SGR treats x as constant. For strongly thixotropic materials:

- x should be time-dependent: x(t)
- Consider structural parameter extensions: λ ∈ [0, 1]
- See SRFS transform for thixotropy detection

Tutorial Notebooks
==================

RheoJAX provides comprehensive SGR tutorial notebooks:

SGR Soft Glassy Rheology Tutorial
---------------------------------

**Notebook**: ``examples/advanced/09-sgr-soft-glassy-rheology.ipynb``

This tutorial covers:

- Loading frequency sweep data for soft glassy materials
- Fitting SGRConventional and SGRGeneric models
- Phase classification from fitted x parameter
- Bayesian uncertainty quantification
- SRFS transform for flow curve analysis
- Shear banding detection and analysis
- Comparison with fractional models

SRFS Flow Curve Analysis
------------------------

**Notebook**: ``examples/transforms/07-srfs-strain-rate-superposition.ipynb``

This tutorial demonstrates:

- Creating flow curve mastercurves
- Extracting SGR parameters from shift factors
- Detecting non-monotonic flow curves
- Shear band coexistence analysis

References
==========

**Foundational SGR Papers:**

- Sollich, P., Lequeux, F., Hébraud, P., & Cates, M. E. (1997). "Rheology of Soft
  Glassy Materials." *Phys. Rev. Lett.* 78, 2020-2023.
  https://doi.org/10.1103/PhysRevLett.78.2020

- Sollich, P. (1998). "Rheological constitutive equation for a model of soft glassy
  materials." *Phys. Rev. E* 58, 738-759.
  https://doi.org/10.1103/PhysRevE.58.738

**Thermodynamic Extensions:**

- Fuereder, I. & Ilg, P. (2013). "GENERIC treatment of soft glassy rheology."
  *Phys. Rev. E* 88, 042134.
  https://doi.org/10.1103/PhysRevE.88.042134

- Sollich, P. & Cates, M. E. (2012). "Thermodynamic interpretation of soft glassy
  rheology models." *Phys. Rev. E* 85, 031127.
  https://doi.org/10.1103/PhysRevE.85.031127

**Shear Banding and Aging:**

- Fielding, S. M., Cates, M. E., & Sollich, P. (2009). "Shear banding, aging and
  noise dynamics in soft glassy materials." *Soft Matter* 5, 2378-2382.

**Reviews:**

- Bonn, D., Denn, M. M., Berthier, L., Divoux, T., & Manneville, S. (2017).
  "Yield stress materials in soft condensed matter." *Rev. Mod. Phys.* 89, 035005.

See Also
========

- :doc:`/models/sgr/sgr_conventional` — SGR Conventional model handbook
- :doc:`/models/sgr/sgr_generic` — SGR GENERIC model handbook
- :doc:`/transforms/srfs` — Strain-Rate Frequency Superposition transform
- :doc:`spp_analysis` — SPP analysis for LAOS (complementary technique)
- :doc:`bayesian_inference` — Uncertainty quantification for SGR parameters
- :doc:`/examples/index` — Tutorial notebooks including SGR workflows
