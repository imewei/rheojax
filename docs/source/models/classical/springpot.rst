.. _model-springpot:

SpringPot (Scott-Blair Element)
===============================

Quick Reference
---------------

**Use when:** Power-law behavior, broad relaxation spectra, fractional viscoelasticity
**Parameters:** 2 (V, α)
**Key equation:** :math:`G^*(\omega) = V (i\omega)^{\alpha}`, :math:`G'(\omega) \sim G''(\omega) \sim \omega^{\alpha}`
**Test modes:** Oscillation, relaxation, creep
**Material examples:** Critical gels (α=0.5), polymer melts near Tg, soft glassy materials, biological tissues

Overview
--------

The :class:`rheojax.models.SpringPot` represents a **fractional viscoelastic element** that interpolates continuously between an ideal spring (:math:`\alpha = 0`) and a Newtonian dashpot (:math:`\alpha = 1`). Introduced by G.W. Scott Blair in 1944, the SpringPot replaces classical integer-order elements with a **fractional derivative**, enabling single-parameter power-law relaxation or creep that spans decades in time or frequency without requiring large Prony series. This model is foundational to fractional rheology and serves as the building block for all fractional viscoelastic models in RheoJAX.

The SpringPot is particularly powerful for materials exhibiting **broad relaxation spectra**—systems where stress relaxation or creep compliance cannot be captured by a single exponential timescale. Common applications include polymer gels, biological tissues, soft glassy materials, and any system near a critical transition (e.g., gelation, glass transition).

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

Fitting Strategies
------------------

**Initialization:**

1. **From frequency sweep**:

   - :math:`\alpha` = slope of :math:`\log(G')` or :math:`\log(G'')` vs :math:`\log(\omega)`
   - :math:`V = G'(\omega_{\text{ref}}) / [\omega_{\text{ref}}^{\alpha} \cos(\pi\alpha/2)]`

2. **From relaxation**:

   - :math:`\alpha` = :math:`-` slope of :math:`\log(G(t))` vs :math:`\log(t)`
   - :math:`V = G(t_{\text{ref}}) \cdot \Gamma(1-\alpha) \cdot t_{\text{ref}}^{\alpha}`

**Optimization:**

- Use log-spaced data to avoid early/late time dominance
- Constrain :math:`0.05 < \alpha < 0.95` to avoid singularities
- Fit in log-space for better numerical conditioning
- Verify with residual plots (should be random, no systematic trends)

**Common Issues:**

1. **Narrow frequency range** → :math:`\alpha` poorly determined
2. **Noise amplification** → Use :doc:`../../transforms/smooth_derivative` for time-domain data
3. **Mixed regimes** → SpringPot valid only in power-law region; use Zener/Maxwell if plateaus exist

Usage Example
-------------

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import SpringPot
   from rheojax.transforms import SmoothDerivative

   # Frequency sweep data (gelatin gel)
   omega = jnp.logspace(-2, 2, 120)  # 0.01 - 100 rad/s
   G_star = measure_complex_modulus(omega)

   # Initialize and fit
   model = SpringPot(V=8.5e3, alpha=0.32)
   model.fit(omega, G_star, test_mode='oscillation')

   # Predict and verify power-law slope
   prediction = model.predict(omega)
   slope_Gp = SmoothDerivative(window=11, poly_order=3).apply(
       jnp.log10(omega), jnp.log10(prediction.storage)
   )
   print(f"Fitted alpha: {model.parameters.get_value('alpha'):.3f}")
   print(f"Measured slope: {jnp.mean(slope_Gp):.3f}")

   # Relaxation prediction from fitted parameters
   t = jnp.logspace(-2, 3, 100)
   G_t = model.predict(t, test_mode='relaxation')

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

**Foundational Papers:**

- G.W. Scott Blair, "Analytical and integrative aspects of the stress–strain–time
  relation," *J. Sci. Instrum.* 21, 80–84 (1944). — **Original SpringPot introduction**
- R.L. Bagley and P.J. Torvik, "A theoretical basis for the application of fractional
  calculus to viscoelasticity," *J. Rheol.* 27, 201–210 (1983). — **Rigorous mathematical foundation**
- H. Schiessel and A. Blumen, "Hierarchical analogues to fractional relaxation
  equations," *J. Phys. A* 26, 5057–5069 (1993). — **Microstructural interpretation**

**Fractional Calculus:**

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010). — **Comprehensive mathematical treatment**
- R. Hilfer (ed.), *Applications of Fractional Calculus in Physics*, World Scientific (2000).

**Gel Rheology:**

- H.H. Winter and F. Chambon, "Analysis of linear viscoelasticity of a crosslinking polymer
  at the gel point," *J. Rheol.* 30, 367–382 (1986). — **Winter-Chambon criterion**
- R. Muthukumar and H.H. Winter, "Fractal dimension of a crosslinking polymer at the gel point,"
  *Macromolecules* 19, 1284–1285 (1986).

**Biological Applications:**

- R.S. Lakes, *Viscoelastic Solids*, CRC Press (1999). — **Materials science perspective**
- Y.C. Fung, *Biomechanics: Mechanical Properties of Living Tissues*, 2nd ed., Springer (1993).

See also
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
