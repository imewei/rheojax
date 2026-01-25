.. _model-springpot:

SpringPot (Scott-Blair Element)
===============================

Quick Reference
---------------

- **Use when:** Power-law behavior, broad relaxation spectra, fractional viscoelasticity
- **Parameters:** 2 (V, α)
- **Key equation:** :math:`G^*(\omega) = V (i\omega)^{\alpha}`, :math:`G'(\omega) \sim G''(\omega) \sim \omega^{\alpha}`
- **Test modes:** Oscillation, relaxation, creep
- **Material examples:** Critical gels (α=0.5), polymer melts near Tg, soft glassy materials, biological tissues

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`V`
     - Fractional stiffness (Pa·s\ :sup:`α`). Sets absolute magnitude of modulus.
   * - :math:`\alpha`
     - Fractional order (0 < α < 1). Controls power-law slope and spectrum breadth.
   * - :math:`{}^{C}D_t^{\alpha}`
     - Caputo fractional derivative operator.
   * - :math:`\tan\delta`
     - Loss tangent = :math:`\tan(\pi\alpha/2)`. Frequency-independent for pure SpringPot.

Overview
--------

The :class:`rheojax.models.SpringPot` represents a **fractional viscoelastic element** that interpolates continuously between an ideal spring (:math:`\alpha = 0`) and a Newtonian dashpot (:math:`\alpha = 1`). Introduced by G.W. Scott Blair in 1944, the SpringPot replaces classical integer-order elements with a **fractional derivative**, enabling single-parameter power-law relaxation or creep that spans decades in time or frequency without requiring large Prony series. This model is foundational to fractional rheology and serves as the building block for all fractional viscoelastic models in RheoJAX.

The SpringPot is particularly powerful for materials exhibiting **broad relaxation spectra**—systems where stress relaxation or creep compliance cannot be captured by a single exponential timescale. Common applications include polymer gels, biological tissues, soft glassy materials, and any system near a critical transition (e.g., gelation, glass transition).

Physical Foundations
--------------------

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The SpringPot describes materials with **hierarchical or fractal structure** where relaxation processes occur across multiple length and time scales:

**Polymer Networks:**
   - Cross-linked networks with broad distribution of cross-link lifetimes
   - Hierarchical structures (e.g., fibrillar collagen networks)
   - Semi-flexible polymers with multiple relaxation mechanisms

**Soft Glassy Materials:**
   - Systems near jamming transition with broad distribution of cage escape times
   - Power-law rheology from structural disorder
   - Examples: Colloidal glasses, dense emulsions, concentrated suspensions

**Critical Gels (α = 0.5):**
   - Materials at the gel point (percolation threshold)
   - Self-similar network structure
   - Winter-Chambon criterion: :math:`G' = G'' \propto \omega^{0.5}`

**Biological Tissues:**
   - Multi-component systems with distributed relaxation times
   - Fibrous networks with varying strand stiffness
   - Examples: Articular cartilage, skin, lung parenchyma

Connection to Fractional Calculus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Caputo fractional derivative generalizes integer-order derivatives. For
:math:`n-1 < \alpha < n` (where :math:`n = \lceil\alpha\rceil`):

.. math::

   {}^{C}D_t^{\alpha} f(t) = \frac{1}{\Gamma(n-\alpha)} \int_0^t (t-s)^{n-\alpha-1} f^{(n)}(s) \, ds

**For viscoelastic SpringPot** (:math:`0 < \alpha < 1`, so :math:`n=1`):

.. math::

   {}^{C}D_t^{\alpha} f(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{f'(s)}{(t-s)^{\alpha}} \, ds, \quad 0 < \alpha < 1

.. list-table:: Key cases for Caputo derivative order
   :header-rows: 1
   :widths: 20 10 70

   * - Order α
     - n
     - Formula
   * - :math:`0 < \alpha < 1`
     - 1
     - :math:`\frac{1}{\Gamma(1-\alpha)} \int_0^t (t-s)^{-\alpha} f'(s) \, ds`
   * - :math:`1 < \alpha < 2`
     - 2
     - :math:`\frac{1}{\Gamma(2-\alpha)} \int_0^t (t-s)^{1-\alpha} f''(s) \, ds`
   * - :math:`2 < \alpha < 3`
     - 3
     - :math:`\frac{1}{\Gamma(3-\alpha)} \int_0^t (t-s)^{2-\alpha} f'''(s) \, ds`

**Limiting behavior** (recovers integer derivatives):

- As :math:`\alpha \to 1^-`: :math:`{}^{C}D_t^{\alpha} f(t) \to f'(t)`
- As :math:`\alpha \to 2^-`: :math:`{}^{C}D_t^{\alpha} f(t) \to f''(t)`

**Memory effect:** The integral form means the fractional derivative depends on
the *entire history* of :math:`f` from 0 to :math:`t`, not just local behavior.
This power-law kernel :math:`(t-s)^{-\alpha}` captures **fading memory**—why
fractional calculus models viscoelastic materials with long-range temporal
correlations (unlike exponential memory in Maxwell models).

**Physical interpretation of α:**

- α → 0: Short-range memory (elastic, spring-like)
- α = 0.5: Self-similar memory (critical gel)
- α → 1: Long-range memory (viscous, dashpot-like)

Physical Foundation
-------------------

The SpringPot generalizes the classical spring-dashpot duality using fractional calculus:

**Constitutive Equation (Caputo Derivative):**

.. math::

   \sigma(t) = V\,{}^{C}D_t^{\alpha}\,\gamma(t), \qquad 0 < \alpha < 1

where:

- :math:`\sigma(t)` = stress (Pa)
- :math:`\gamma(t)` = strain (dimensionless)
- :math:`V` = generalized modulus (Pa·s\ :sup:`\alpha`)
- :math:`{}^{C}D_t^{\alpha}` = Caputo fractional derivative of order :math:`\alpha`
- :math:`\alpha` = fractional order in (0, 1)

**Physical Meaning of Fractional Order** :math:`\alpha`:

The fractional order :math:`\alpha` quantifies the **width of the relaxation spectrum**:

- :math:`\alpha \to 0`: Nearly elastic (spring-like), narrow spectrum
- :math:`\alpha = 0.5`: Balanced viscoelastic behavior, **critical gel** signature
- :math:`\alpha \to 1`: Nearly viscous (dashpot-like), narrow spectrum
- :math:`0 < \alpha < 1`: Broad distribution of relaxation times, **power-law behavior**

Materials with hierarchical structures, fractal networks, or significant polydispersity exhibit intermediate :math:`\alpha` values (0.2 < :math:`\alpha` < 0.8).

Governing Equations
-------------------

**Complex Modulus (Oscillatory):**

.. math::

   G^*(\omega) = V (i\omega)^{\alpha} = V\,\omega^{\alpha}
   \left[\cos\left(\tfrac{\pi\alpha}{2}\right) + i\sin\left(\tfrac{\pi\alpha}{2}\right)\right]

**Storage and Loss Moduli:**

.. math::

   G'(\omega) &= V\,\omega^{\alpha} \cos\left(\tfrac{\pi\alpha}{2}\right) \\
   G''(\omega) &= V\,\omega^{\alpha} \sin\left(\tfrac{\pi\alpha}{2}\right)

**Loss Tangent (Frequency-Independent):**

.. math::

   \tan\delta = \frac{G''(\omega)}{G'(\omega)} = \tan\left(\tfrac{\pi\alpha}{2}\right) = \text{constant}

This **frequency-independent loss tangent** is the hallmark of SpringPot behavior and the **Winter-Chambon criterion** for critical gels.

**Relaxation Modulus:**

.. math::

   G(t) = \frac{V}{\Gamma(1-\alpha)} t^{-\alpha}

Shows pure power-law decay without exponential components.

**Creep Compliance:**

.. math::

   J(t) = \frac{1}{V\,\Gamma(1+\alpha)} t^{\alpha}

Shows power-law growth in strain under constant stress.

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 22 20 58

   * - Name
     - Units
     - Description / Constraints
   * - ``V``
     - Pa·s\ :sup:`\alpha`
     - Fractional stiffness; > 0. Sets vertical placement of :math:`G^*` on log-log plots.
   * - ``alpha``
     - –
     - Fractional order in (0, 1). Controls slope of :math:`G'` and :math:`G''` in log-log space.

Validity and Assumptions
------------------------

- Linear viscoelasticity: yes
- Small amplitude: yes
- Isothermal: yes
- Data/test modes: oscillation, relaxation, creep
- Additional assumptions: pure power-law behavior (no plateaus or terminal flow)

Limitations
~~~~~~~~~~~

**No terminal flow or equilibrium plateau:**
   SpringPot alone cannot capture:
   - Terminal viscosity (need dashpot in series)
   - Equilibrium modulus (need spring in parallel)
   - Use composite models (Fractional Maxwell, Fractional Zener) for complete behavior

**Valid only in power-law regime:**
   Real materials exhibit power-law behavior only over limited frequency/time windows. SpringPot should be restricted to the range where :math:`\log(G')` vs :math:`\log(\omega)` is linear.

**Singular limits:**
   α → 0 or α → 1 are numerically unstable. Use bounds [0.05, 0.95] in fitting. For α < 0.1, use spring model; for α > 0.9, use dashpot model.

**Temperature dependence:**
   Both V and α may depend on temperature. For time-temperature superposition, use composite models where α can remain constant while timescales shift.

What You Can Learn
------------------

This section explains how to extract physical insights from fitted SpringPot parameters and diagnose material behavior.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**V (Fractional Stiffness)**:
   Fitted :math:`V` reveals the absolute magnitude of the material response:

   - **Low values (<10² Pa·s\ :sup:`α`)**: Soft materials (weak gels, dilute solutions)
   - **Moderate values (10²-10⁴ Pa·s\ :sup:`α`)**: Biological tissues, polymer gels
   - **High values (>10⁴ Pa·s\ :sup:`α`)**: Dense colloidal suspensions, concentrated emulsions

   *For practitioners*: :math:`V` at reference frequency :math:`\omega_{\text{ref}}` gives characteristic modulus via :math:`|G^*(\omega_{\text{ref}})| = V \omega_{\text{ref}}^{\alpha}`.

   *For researchers*: Compare :math:`V` across samples to track structural changes (crosslinking, aging, degradation).

**alpha (Fractional Order)**:
   Fitted :math:`\alpha` quantifies spectrum breadth and material state:

   - **Low α (<0.2)**: Narrow spectrum, nearly elastic solid
   - **α ≈ 0.5**: Critical gel (Winter-Chambon criterion), self-similar structure
   - **High α (>0.8)**: Broad spectrum, nearly viscous liquid

   *For researchers*: :math:`\alpha` relates to structural complexity:
      - Monodisperse systems: α → 0 or 1 (single timescale)
      - Polydisperse/hierarchical systems: 0.2 < α < 0.8 (broad spectrum)
      - At gel point: α = 0.5 exactly (percolation threshold)

   *For quality control*: Track α to monitor gelation progress or structural degradation.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from SpringPot Parameters
   :header-rows: 1
   :widths: 18 28 28 26

   * - α Range
     - Material Type
     - Physical State
     - Examples
   * - 0.1-0.3
     - Weak gels, soft solids
     - Near-solid, narrow spectrum
     - Gelatin, weak hydrogels
   * - 0.4-0.6
     - Critical gels, soft glasses
     - Gel point, self-similar
     - Colloidal glasses at jamming
   * - 0.7-0.9
     - Viscoelastic fluids
     - Near-liquid, broad spectrum
     - Polymer melts near T\ :sub:`g`

**Winter-Chambon Gel Point Criterion:**
   At gelation, α = 0.5 exactly and :math:`\tan\delta = 1` (G' = G'').

   To verify gel point:
      1. Fit SpringPot to frequency sweep
      2. Check α ≈ 0.5 (within ±0.05)
      3. Verify :math:`\tan\delta` is frequency-independent
      4. Confirm :math:`G' \approx G''` across decades

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

**Warning signs in fitted parameters:**

- **If α < 0.1 or α > 0.9**: Power-law regime too narrow; SpringPot not appropriate. Use Maxwell (α → 1) or spring (α → 0) instead.
- **If fit quality poor (R² < 0.95)**: Material exhibits plateaus or terminal flow. Use Fractional Maxwell or Fractional Zener models.
- **If α varies with temperature**: Material is thermorheologically complex. Cannot use simple time-temperature superposition.
- **If tan δ not constant**: Pure SpringPot inadequate; mixture of relaxation processes. Use composite models.

**Verification strategies:**

1. **Log-log linearity**: Plot :math:`\log(G')` vs :math:`\log(\omega)` → slope should equal α across entire range
2. **Loss tangent constancy**: :math:`\tan\delta = \tan(\pi\alpha/2)` should be frequency-independent
3. **G'/G'' ratio**: Should satisfy :math:`G'/G'' = \cot(\pi\alpha/2)` at all frequencies
4. **Relaxation modulus**: :math:`\log(G(t))` vs :math:`\log(t)` → slope = -α

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Gel Point Determination:**
   - Track α during crosslinking/gelation
   - Gel point occurs when α = 0.5 (±0.05)
   - Confirms percolation transition

**Material Development:**
   - Tune crosslink density to achieve target α
   - α = 0.3-0.4 for soft tissue mimics
   - α = 0.5 for self-healing gels at rest state

**Quality Control:**
   - Monitor α batch-to-batch for consistency
   - Degradation typically increases α (network breakdown)
   - Aging typically decreases α (network strengthening)

Material Examples
-----------------

**Polymer Gels** (:math:`\alpha \approx 0.3-0.7`):

- **Gelatin gels**: :math:`V \approx 10^3-10^4` Pa·s\ :sup:`\alpha`, :math:`\alpha \approx 0.4-0.6`
- **Alginate gels**: :math:`\alpha \approx 0.3-0.5` depending on cross-link density
- **Collagen networks**: :math:`\alpha \approx 0.4` (hierarchical fibrillar structure)

**Biological Tissues** (:math:`\alpha \approx 0.1-0.5`):

- **Articular cartilage**: :math:`\alpha \approx 0.2`, :math:`V \approx 10^5` Pa·s\ :sup:`\alpha`
- **Skin (dermis)**: :math:`\alpha \approx 0.3-0.4`
- **Lung tissue**: :math:`\alpha \approx 0.15-0.25`
- **Liver parenchyma**: :math:`\alpha \approx 0.2`

**Soft Glassy Materials** (:math:`\alpha \approx 0.3-0.6`):

- **Carbopol gels**: :math:`\alpha \approx 0.4-0.5`
- **Laponite suspensions**: :math:`\alpha \approx 0.3-0.6` near jamming
- **Concentrated emulsions**: :math:`\alpha \approx 0.4`

**Critical Gels** (:math:`\alpha = 0.5`):

- Materials **at the gel point** exhibit :math:`G' = G'' \propto \omega^{0.5}`
- This is the **Winter-Chambon criterion** for gelation

Experimental Design
-------------------

**Frequency Sweep (SAOS):**

1. **Frequency range**: Span at least 2 decades (e.g., 0.1-10 Hz) to resolve power-law slope
2. **Strain amplitude**: Within LVR (typically 0.5-5% for gels)
3. **Temperature control**: ±0.1°C (critical for soft materials)
4. **Verification**: Plot :math:`\log(G')` vs :math:`\log(\omega)` → slope = :math:`\alpha`

**Stress Relaxation:**

1. **Step strain**: :math:`\gamma_0 = 1-5\%` within LVR
2. **Time span**: Cover 3-4 decades in time
3. **Sampling**: Log-spaced time points to capture power-law decay
4. **Analysis**: :math:`\log(G(t))` vs :math:`\log(t)` → slope = :math:`-\alpha`

**Creep Test:**

1. **Constant stress**: Within LVR (verify with amplitude sweep)
2. **Time span**: Long enough to observe power-law regime (> 10³ s for gels)
3. **Sampling**: Log-spaced to avoid early-time bias
4. **Analysis**: :math:`\log(J(t))` vs :math:`\log(t)` → slope = :math:`\alpha`

Fitting Guidance
----------------

Parameter Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From frequency sweep**

**Step 1**: Estimate :math:`\alpha` from log-log slope
   Plot :math:`\log(G')` vs :math:`\log(\omega)` → slope = :math:`\alpha`

**Step 2**: Calculate :math:`V` from a reference point
   Choose :math:`\omega_{\text{ref}}` in middle of data range:

   .. math::

      V = \frac{G'(\omega_{\text{ref}})}{\omega_{\text{ref}}^{\alpha} \cos(\pi\alpha/2)}

**Method 2: From relaxation data**

**Step 1**: Estimate :math:`\alpha` from log-log slope
   Plot :math:`\log(G(t))` vs :math:`\log(t)` → slope = :math:`-\alpha`

**Step 2**: Calculate :math:`V` using Gamma function
   Choose :math:`t_{\text{ref}}` in middle of data range:

   .. math::

      V = G(t_{\text{ref}}) \cdot \Gamma(1-\alpha) \cdot t_{\text{ref}}^{\alpha}

**Method 3: From loss tangent (critical gels)**

If :math:`\tan\delta` is constant:

**Step 1**: Calculate :math:`\alpha` directly
   .. math::

      \alpha = \frac{2}{\pi} \arctan(\tan\delta)

**Step 2**: Use frequency sweep method above for :math:`V`

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for SpringPot (2 parameters, well-conditioned)
   - Fast convergence if initialized from slope analysis
   - 5-270× faster than scipy.optimize

**Bounds:**
   - :math:`V`: [1e-2, 1e8] Pa·s\ :sup:`α` (adjust based on material)
   - :math:`\alpha`: [0.05, 0.95] (avoid singularities at 0 and 1)

**Fitting strategy:**
   - Use log-spaced data to avoid early/late time dominance
   - Fit in log-space: minimize :math:`\sum (\log|G^*_{\text{data}}| - \log|G^*_{\text{fit}}|)^2`
   - Weight G' and G'' equally in log-space

Troubleshooting Common Fitting Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics and solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Diagnostic
     - Solution
   * - α converges to bounds (0.05 or 0.95)
     - Power-law regime too narrow
     - Use Maxwell (α → 1) or spring (α → 0) model instead
   * - Poor fit despite linear log-log plot
     - Transition regions at data edges
     - Restrict fitting to middle decade where slope is constant
   * - Noisy α estimate
     - Insufficient frequency/time span
     - Extend measurement to cover 3+ decades
   * - Oscillatory residuals
     - Multiple relaxation modes present
     - Use Fractional Zener or Generalized Maxwell
   * - V unrealistic (negative or extreme)
     - Poor α initialization
     - Reinitialize α from slope analysis before fitting V

Validation Strategies
~~~~~~~~~~~~~~~~~~~~~

**1. Visual Inspection**

Plot on log-log axes:
   - :math:`G'` and :math:`G''` vs :math:`\omega` should be parallel straight lines
   - Slope of both curves should equal :math:`\alpha`
   - Vertical separation = :math:`\pi\alpha/2` radians in phase

**2. Loss Tangent Check**

.. math::

   \tan\delta = \frac{G''(\omega)}{G'(\omega)} = \tan\left(\frac{\pi\alpha}{2}\right) = \text{constant}

If :math:`\tan\delta` varies > 10% across frequency range, SpringPot is inadequate.

**3. Residual Analysis**

Residuals in log-space should be:
   - Random (no systematic curvature)
   - Equal magnitude for G' and G''
   - RMSE < 0.1 (10% error in log-space)

**4. Cross-validation**

Convert between test modes:
   - Fit oscillation → predict relaxation via inverse Fourier
   - Fit relaxation → predict oscillation via Fourier
   - Discrepancies > 20% indicate model inadequacy

Usage
-----

Basic Fitting Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import SpringPot

   # Frequency sweep data (critical gel)
   omega = jnp.logspace(-2, 2, 120)  # 0.01 - 100 rad/s
   G_star = measure_complex_modulus(omega)

   # Initialize and fit
   model = SpringPot(V=1.0e3, alpha=0.5)
   model.fit(omega, G_star, test_mode='oscillation')

   # Check fitted parameters
   V = model.parameters.get_value('V')
   alpha = model.parameters.get_value('alpha')
   print(f"V = {V:.2e} Pa·s^{alpha:.3f}")
   print(f"alpha = {alpha:.3f}")

   # Verify gel point (alpha should be ~0.5)
   if abs(alpha - 0.5) < 0.05:
       print("Material is at gel point (Winter-Chambon criterion satisfied)")

Advanced: Slope Verification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   import numpy as np
   from rheojax.models import SpringPot

   # Fit model
   model = SpringPot()
   model.fit(omega, G_star, test_mode='oscillation')

   # Extract parameters
   alpha_fit = model.parameters.get_value('alpha')

   # Verify slope from data
   log_omega = np.log10(omega)
   log_Gp = np.log10(np.real(G_star))

   # Linear regression on middle 80% of data
   n = len(omega)
   idx_start, idx_end = int(0.1*n), int(0.9*n)
   slope_measured = np.polyfit(
       log_omega[idx_start:idx_end],
       log_Gp[idx_start:idx_end],
       1
   )[0]

   print(f"Fitted alpha: {alpha_fit:.3f}")
   print(f"Measured slope: {slope_measured:.3f}")
   print(f"Difference: {abs(alpha_fit - slope_measured):.4f}")

Cross-Mode Prediction
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.jax_config import safe_import_jax
   jax, jnp = safe_import_jax()
   from rheojax.models import SpringPot

   # Fit to oscillatory data
   model = SpringPot()
   model.fit(omega, G_star, test_mode='oscillation')

   # Predict relaxation modulus
   t = jnp.logspace(-2, 3, 100)
   G_t = model.predict(t, test_mode='relaxation')

   # Predict creep compliance
   J_t = model.predict(t, test_mode='creep')

   # Verify power-law: G(t) ~ t^(-alpha)
   alpha = model.parameters.get_value('alpha')
   expected_slope = -alpha

   log_t = jnp.log10(t[10:-10])
   log_G = jnp.log10(G_t[10:-10])
   actual_slope = jnp.polyfit(log_t, log_G, 1)[0]

   print(f"Expected slope: {expected_slope:.3f}")
   print(f"Actual slope: {actual_slope:.3f}")

Model Comparison
----------------

**SpringPot vs Maxwell:**

- Maxwell: Exponential relaxation, single timescale
- SpringPot: Power-law relaxation, broad spectrum
- Use SpringPot when :math:`\log(G(t))` vs :math:`\log(t)` is linear over decades

**SpringPot vs Zener:**

- Zener: Has equilibrium plateau (:math:`G_e > 0`)
- SpringPot: No plateau, pure power-law
- Combine as :doc:`zener` + SpringPot for finite plateau with power-law approach

**SpringPot vs Fractional Maxwell:**

- SpringPot: Single element (building block)
- :doc:`../fractional/fractional_maxwell_gel`: SpringPot + dashpot (terminal flow)
- :doc:`../fractional/fractional_maxwell_liquid`: SpringPot + spring (finite plateau)

Limitations
-----------

1. **No terminal flow**: SpringPot alone does not capture viscous dissipation at long times
2. **No equilibrium plateau**: Cannot model cross-linked solids without modification
3. **Narrow applicability**: Valid only in power-law regime; most real materials need composite models
4. **Singular limits**: :math:`\alpha \to 0, 1` numerically unstable; use bounds :math:`[0.05, 0.95]`

Troubleshooting
---------------

**Issue: Fitted** :math:`\alpha` **close to 0 or 1**

- **Cause**: Data span insufficient frequency/time range
- **Solution**: Extend measurement window or use Maxwell/Zener model

**Issue: Poor fit quality despite power-law appearance**

- **Cause**: Transition regions at edges of data
- **Solution**: Restrict fitting to middle decade where slope is constant

**Issue: Noisy derivative from creep data**

- **Solution**: Use :doc:`../../transforms/smooth_derivative` with Savitzky-Golay window ≥ 11 points

**Issue: Oscillatory residuals in log-log plot**

- **Cause**: Multiple relaxation modes, not pure SpringPot
- **Solution**: Use :doc:`../fractional/fractional_zener_ss` or generalized models

Tips & Best Practices
----------------------

1. **Log-space fitting**: Resample data onto log-spaced grids to avoid early/late-time bias
2. **Verify power-law**: Plot :math:`\log(G')` vs :math:`\log(\omega)` → linear with slope :math:`\alpha`
3. **Check loss tangent**: For pure SpringPot, :math:`\tan\delta` should be frequency-independent
4. **Combine with transforms**: Use :doc:`../../transforms/fft` to convert relaxation → oscillation
5. **Parallel spring**: Add spring (:doc:`zener`) if equilibrium modulus :math:`G_e > 0` exists
6. **Series dashpot**: Add dashpot (:doc:`../fractional/fractional_maxwell_gel`) for terminal flow
7. **Differentiation**: Use :doc:`../../transforms/smooth_derivative` for creep compliance derivatives

References
----------

.. [1] Scott Blair, G. W. "Analytical and integrative aspects of the stress–strain–time
   relation." *Journal of Scientific Instruments*, 21, 80–84 (1944).
   https://doi.org/10.1088/0950-7671/21/5/302. Original SpringPot introduction.

.. [2] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724

.. [3] Schiessel, H., and Blumen, A. "Hierarchical analogues to fractional relaxation
   equations." *Journal of Physics A*, 26, 5057–5069 (1993).
   https://doi.org/10.1088/0305-4470/26/19/034

.. [4] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [5] Hilfer, R. (ed.) *Applications of Fractional Calculus in Physics*.
   World Scientific (2000). ISBN: 978-9810234577

.. [6] Winter, H. H., and Chambon, F. "Analysis of linear viscoelasticity of a
   crosslinking polymer at the gel point." *Journal of Rheology*, 30, 367–382 (1986).
   https://doi.org/10.1122/1.549853

.. [7] Muthukumar, M. "Screening effect on viscoelasticity near the gel point."
   *Macromolecules*, 22, 4656–4658 (1989). https://doi.org/10.1021/ma00202a050

.. [8] Lakes, R. S. *Viscoelastic Solids*. CRC Press (1999). ISBN: 978-0849396588

.. [9] Fung, Y. C. *Biomechanics: Mechanical Properties of Living Tissues*, 2nd ed.
   Springer (1993). https://doi.org/10.1007/978-1-4757-2257-4

.. [10] Jaishankar, A., and McKinley, G. H. "Power-law rheology in the bulk and at the
   interface." *Proceedings of the Royal Society A*, 469, 20120284 (2013).
   https://doi.org/10.1098/rspa.2012.0284

See Also
--------

**Classical Models:**

- :doc:`zener` — adds an equilibrium spring in parallel for finite creep recovery
- :doc:`maxwell` — limiting case with :math:`\alpha = 1` (pure dashpot)

**Fractional Models (Built on SpringPot):**

- :doc:`../fractional/fractional_maxwell_gel` — series combination with a dashpot for gel-like response
- :doc:`../fractional/fractional_kelvin_voigt` — SpringPot in parallel with a spring for quasi-solid behavior
- :doc:`../fractional/fractional_zener_ss` — most general fractional Zener model

**Transforms:**

- :doc:`../../transforms/fft` — convert time sweeps to :math:`G'(\omega)` and :math:`G''(\omega)` before fitting
- :doc:`../../transforms/smooth_derivative` — noise-robust differentiation for creep data
- :doc:`../../transforms/mutation_number` — verify frequency-independent loss tangent

**Examples:**

- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing all fractional families
- :doc:`../../examples/basic/03-springpot-fitting` — step-by-step SpringPot parameter estimation

**User Guides:**

- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing rheological models
