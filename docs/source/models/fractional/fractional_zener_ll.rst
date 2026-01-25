.. _model-fractional-zener-ll:

Fractional Zener Liquid-Liquid (Fractional)
===========================================

Quick Reference
---------------

**Use when:** Liquid with broad multi-order fractional dispersions, complex viscoelastic behavior
**Parameters:** 6 (c₁, c₂, α, β, γ, τ)
**Key equation:** :math:`G^*(\omega) = \frac{c_1(i\omega)^{\alpha}}{1 + (i\omega\tau)^{\beta}} + c_2(i\omega)^{\gamma}`
**Test modes:** Oscillation, relaxation
**Material examples:** Complex fluids with multiple fractional relaxation mechanisms

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`c_1`
     - Pa·s\ :sup:`α`
     - First SpringPot constant (high-frequency behavior)
   * - :math:`c_2`
     - Pa·s\ :sup:`γ`
     - Second SpringPot constant (low-frequency behavior)
   * - :math:`\alpha`
     - dimensionless
     - First fractional order (high-frequency power-law)
   * - :math:`\beta`
     - dimensionless
     - Second fractional order (crossover character)
   * - :math:`\gamma`
     - dimensionless
     - Third fractional order (low-frequency power-law)
   * - :math:`\tau`
     - s
     - Relaxation time (regime transition)
   * - :math:`E^{\delta}_{\mu,\nu}(t)`
     - dimensionless
     - Prabhakar (generalized Mittag-Leffler) function

Overview
--------

The most general three-element fractional Zener form combining two SpringPots and a viscous time constant. It models liquid-like behavior with broad, multi-order fractional dispersions.

Physical Foundations
--------------------

The Fractional Zener Liquid-Liquid (FZLL) represents the most general fractional
viscoelastic model with three independent fractional orders:

**Mechanical Configuration:**

.. code-block:: text

   [SpringPot (c₁, α, β, τ)] ---- parallel ---- [SpringPot (c₂, γ)]

**Microstructural Interpretation:**

- **Primary branch (c₁, α, β)**: Captures main relaxation mechanism with
  characteristic time τ. The two orders α and β control high-frequency and
  crossover behavior, respectively.
- **Secondary branch (c₂, γ)**: Provides additional relaxation mode at different
  timescale. Order γ controls low-frequency terminal behavior.
- **Liquid-like**: No equilibrium modulus, material flows under stress
- **Multi-scale relaxation**: Three orders allow hierarchical relaxation processes

This model is only needed for materials with extremely complex rheology that cannot
be described by simpler fractional models (FMG, FML, FZSS with 3-4 parameters).

Governing Equations
-------------------

Frequency domain (complex modulus; analytical):

.. math::
   :nowrap:

   \[
   G^{*}(\omega) \;=\;
   \frac{c_1\,(i\omega)^{\alpha}}{1 + (i\omega\tau)^{\beta}} \;+\; c_2\,(i\omega)^{\gamma}.
   \]

Time domain (relaxation modulus; general case):

.. math::
   :nowrap:

   \[
   G(t) \;=\; \mathcal{L}^{-1}\!\left\{G^{*}(s)\right\}(t)
   \;\;\text{which, for distinct orders, involves generalized
   Mittag\text{-}Leffler (Prabhakar) functions } E^{\delta}_{\mu,\nu}(t) .
   \]

Special cases (e.g., :math:`\beta=\alpha`, :math:`c_2=0`) reduce to two-parameter Mittag-Leffler forms.

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
   * - ``c1``
     - :math:`c_1`
     - Pa*s^alpha
     - [1e-3, 1e9]
     - First SpringPot constant
   * - ``c2``
     - :math:`c_2`
     - Pa*s^gamma
     - [1e-3, 1e9]
     - Second SpringPot constant
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0, 1]
     - First fractional order
   * - ``beta``
     - :math:`\beta`
     - dimensionless
     - [0, 1]
     - Second fractional order
   * - ``gamma``
     - :math:`\gamma`
     - dimensionless
     - [0, 1]
     - Third fractional order
   * - ``tau``
     - :math:`\tau`
     - s
     - [1e-6, 1e6]
     - Relaxation time

Validity and Assumptions
------------------------

- Linear viscoelastic assumption; strain amplitudes remain small.
- Isothermal, time-invariant material parameters throughout the experiment.
- Supported RheoJAX test modes: relaxation, creep, oscillation.
- Fractional orders stay within (0, 1) to keep kernels causal and bounded.

Regimes and Behavior
--------------------

- Liquid-like at low omega (no equilibrium plateau).
- Multiple fractional slopes in :math:`G'` and :math:`G''` controlled by :math:`\alpha,\beta,\gamma`.
- Captures complex crossover patterns not possible with single-order models.

Limiting Behavior
-----------------

- :math:`\alpha,\beta,\gamma \to 1`: tends to classical viscoelastic liquid combinations.
- :math:`c_2 \to 0`: reduces to a generalized fractional Maxwell form.
- Equal orders collapse to two-parameter Mittag-Leffler responses.

What You Can Learn
------------------

This section explains what insights you can extract from fitting the most general three-order Fractional Zener Liquid-Liquid model, emphasizing the hierarchical multi-scale relaxation processes captured by three independent fractional orders.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Three Fractional Orders (α, β, γ)**:
   Each order controls a different frequency/time regime, enabling characterization of hierarchical relaxation. All three are bounded between 0 (solid-like) and 1 (liquid-like).

   - **α**: High-frequency power-law slope. Controls short-time/fast relaxation behavior.
   - **β**: Crossover behavior near τ. Governs the transition between primary and secondary relaxation mechanisms.
   - **γ**: Low-frequency/long-time terminal behavior. Determines terminal flow characteristics.

   *For graduate students*: The presence of three independent orders suggests hierarchical microstructural relaxation mechanisms operating at different scales—e.g., segmental motion (α), chain relaxation (β), and network rearrangement (γ).
   *For practitioners*: Each order corresponds to a distinct timescale in material response. Map α, β, γ to processing conditions (mixing, extrusion, curing) to optimize manufacturing.

**SpringPot Constants (c₁, c₂)**:
   Determine the relative strength of the two parallel relaxation branches.

   - **c₁/c₂ > 10**: Primary branch (α, β) dominates; secondary (γ) is a correction
   - **c₁/c₂ ≈ 1**: Both mechanisms contribute equally; true two-mode behavior
   - **c₁/c₂ < 0.1**: Secondary branch (γ) dominates; consider simpler model

   *For graduate students*: The ratio c₁/c₂ relates to the partition of energy dissipation between fast (primary) and slow (secondary) mechanisms.
   *For practitioners*: High c₁/c₂ means short-time response is critical; low c₁/c₂ emphasizes long-time flow.

**Relaxation Time (τ)**:
   Characteristic timescale separating primary and secondary relaxation regimes.

   *For graduate students*: τ marks the crossover frequency ω ≈ 1/τ where the dominant relaxation mechanism shifts.
   *For practitioners*: Compare τ to process timescales to determine which regime governs product performance.

When to Use This Model
~~~~~~~~~~~~~~~~~~~~~~

This 6-parameter model is the most flexible fractional Zener variant. Use only when:

1. **Simpler models fail**: If 3-4 parameter fractional models (FMG, FML, FZSS) show systematic deviations
2. **Multiple distinct power-law regimes**: Data shows different slopes α, β, γ in separate frequency decades
3. **Complex fluids**: Polymer blends, filled systems, colloidal suspensions, or hierarchical structures
4. **High-quality data**: At least 5 decades in frequency with >100 points

**Critical Warning**: With 6 parameters, overfitting is highly probable. Always compare to simpler models using information criteria (AIC, BIC). Prefer simpler models unless FZLL provides statistically significant improvement (ΔAIC > 10).

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Multi-Order Interpretation from FZLL Parameters
   :header-rows: 1
   :widths: 30 35 35

   * - Order Pattern
     - Physical Interpretation
     - Material Examples
   * - α ≈ β ≈ γ
     - Single relaxation mechanism
     - Use simpler FMG/FML instead
   * - α < β < γ (ascending)
     - Hierarchical fast-to-slow relaxation
     - Polymer solutions, micellar systems
   * - α > γ, β intermediate
     - Two-scale relaxation
     - Filled polymers, composites
   * - \|α - γ\| > 0.5, any β
     - Extreme spectrum breadth
     - Complex biomaterials, blends

.. list-table:: Parameter Bounds and Physical Meaning
   :header-rows: 1
   :widths: 25 25 25 25

   * - Parameter
     - → 0 Limit
     - → 1 Limit
     - Typical Range
   * - α
     - Solid-like (fast)
     - Viscous (fast)
     - 0.3-0.7
   * - β
     - Elastic crossover
     - Viscous crossover
     - 0.2-0.8
   * - γ
     - Solid-like (slow)
     - Newtonian terminal
     - 0.5-0.95 (liquid)

Fitting Guidance
----------------

**Recommended Data Collection:**

1. **Frequency sweep** (SAOS): 5+ decades to resolve multiple power-law regimes
2. **High data quality**: 100+ points, minimal noise (6 parameters require excellent data)
3. **Test amplitude**: Within LVR (< 5% strain)
4. **Temperature**: Constant ±0.05°C (stringent for complex models)

**Initialization Strategy:**

.. code-block:: python

   # Requires careful analysis of multi-regime behavior
   # Start from simpler model fits (FMG, FML) and add complexity
   # Use hierarchical fitting: fit 3-parameter model first,
   # then add additional parameters

**Model Selection:**

- **Critical**: Only use FZLL if simpler models (3-4 parameters) fail
- **Justification**: Compare AIC/BIC to FMG, FML, FZSS
- **Validation**: Cross-validate to ensure not overfitting
- **Parsimony**: Prefer simpler models unless data clearly requires complexity

**Common Pitfalls:**

- **Overfitting**: Most common issue with 6-parameter models
- **Parameter non-uniqueness**: Multiple parameter sets may fit equally well
- **Poor conditioning**: Optimization may not converge reliably
- **Interpretation difficulty**: Hard to extract physical meaning from 6 parameters

**Troubleshooting Table:**

.. list-table:: Common Issues and Solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Issue
     - Likely Cause
     - Solution
   * - Fit unstable across runs
     - Multiple local minima
     - Use hierarchical fitting from simpler models
   * - High parameter correlations
     - Model too flexible
     - Reduce to 4-parameter FZSS or FMG
   * - Poor AIC vs simpler models
     - Overfitting
     - Prefer 3-4 parameter models
   * - α ≈ β ≈ γ
     - Single relaxation mode
     - Use FMG or FML instead
   * - c₂/c₁ < 0.01 or > 100
     - One branch dominates
     - Simplify to single-branch model
   * - Fit diverges
     - Poor initialization
     - Start from FMG fit, add complexity
   * - Non-physical parameters
     - Data quality insufficient
     - Need 5+ decades, 100+ points
   * - Systematic residuals
     - Wrong model class
     - Consider GMM or different framework

Model Selection Guidelines
---------------------------

**Critical Decision Tree:**

Use FZLL only if ALL of the following are true:

1. **Data quality**: 5+ decades in frequency, >100 points, SNR > 20 dB
2. **Simpler models fail**: FMG/FML show systematic deviations (R² < 0.95)
3. **Multiple power-law regimes**: Clearly distinct slopes α, β, γ in log-log plot
4. **Statistical justification**: ΔAIC > 10 compared to best 4-parameter model
5. **Physical interpretation**: Can explain three orders from microstructure

**Comparison Workflow:**

.. code-block:: python

   from rheojax.models import (FractionalMaxwellGel,
                                FractionalMaxwellLiquid,
                                FractionalZenerSolidSolid,
                                FractionalZenerLiquidLiquid)

   models = {
       'FMG': FractionalMaxwellGel(),
       'FML': FractionalMaxwellLiquid(),
       'FZSS': FractionalZenerSolidSolid(),
       'FZLL': FractionalZenerLiquidLiquid(),
   }

   results = {}
   for name, model in models.items():
       model.fit(data)
       results[name] = {
           'R2': model.r_squared,
           'AIC': compute_aic(model),
           'BIC': compute_bic(model),
           'n_params': model.n_parameters,
       }

   # Compare information criteria
   import pandas as pd
   df = pd.DataFrame(results).T
   print(df.sort_values('AIC'))

   # FZLL justified only if AIC significantly lower (ΔAIC > 10)

Practical Applications
----------------------

**Material Characterization:**

FZLL is useful for characterizing extremely complex materials where hierarchical relaxation mechanisms operate:

1. **Polymer blends**: Three orders represent each component plus interfacial dynamics
2. **Filled systems**: Matrix (α), filler-matrix interphase (β), filler network (γ)
3. **Biological materials**: Molecular (fast), cellular (medium), tissue (slow) scales
4. **Colloidal suspensions**: Brownian (α), hydrodynamic (β), structural (γ) relaxation

**Quality Control Warning:**

Due to 6 parameters, FZLL is NOT recommended for routine QC. Instead:

- Use for initial material characterization only
- Switch to simpler model (3-4 parameters) for batch monitoring
- Focus on derived quantities (e.g., relaxation time distribution) rather than individual parameters

**Research Applications:**

FZLL is appropriate for fundamental research where:

- Understanding multi-scale relaxation mechanisms is the goal
- High-quality data across 5+ decades is available
- Multiple power-law regimes need quantification
- Comparison to theoretical predictions requires this flexibility

Example Calculations
--------------------

**Multi-Order Spectrum Analysis:**

Given fitted parameters c₁ = 500 Pa·s^α, c₂ = 100 Pa·s^γ, α = 0.4, β = 0.6, γ = 0.8, τ = 10 s:

.. code-block:: python

   import numpy as np
   from rheojax.models import FractionalZenerLiquidLiquid
   from rheojax.core.jax_config import safe_import_jax

   jax, jnp = safe_import_jax()

   model = FractionalZenerLiquidLiquid()
   model.parameters.set_value('c1', 500.0)
   model.parameters.set_value('c2', 100.0)
   model.parameters.set_value('alpha', 0.4)
   model.parameters.set_value('beta', 0.6)
   model.parameters.set_value('gamma', 0.8)
   model.parameters.set_value('tau', 10.0)

   # Predict over wide frequency range
   omega = jnp.logspace(-4, 3, 200)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = jnp.real(G_star)
   G_double_prime = jnp.imag(G_star)

   # Identify power-law regimes
   log_omega = jnp.log10(omega)
   log_Gp = jnp.log10(G_prime)
   log_Gpp = jnp.log10(G_double_prime)

   # Compute local slopes (discrete derivative)
   slope_Gp = jnp.gradient(log_Gp, log_omega)
   slope_Gpp = jnp.gradient(log_Gpp, log_omega)

   # Find regions with distinct slopes
   print("G' power-law slopes by frequency regime:")
   for regime, (w_low, w_high) in [('Low', (1e-4, 1e-2)),
                                     ('Mid', (1e-1, 1e1)),
                                     ('High', (1e1, 1e3))]:
       mask = (omega >= w_low) & (omega <= w_high)
       mean_slope = slope_Gp[mask].mean()
       print(f"  {regime} (ω={w_low}-{w_high}): slope ≈ {mean_slope:.2f}")

   # Compare to theoretical orders α, β, γ
   print(f"Expected orders: α={0.4}, β={0.6}, γ={0.8}")

**Hierarchical Fitting Strategy:**

.. code-block:: python

   # Start simple, add complexity only if justified
   from rheojax.models import FractionalMaxwellGel

   # Step 1: Fit 3-parameter FMG
   model_fmg = FractionalMaxwellGel()
   result_fmg = model_fmg.fit(data)
   R2_fmg = result_fmg.r_squared
   AIC_fmg = compute_aic(result_fmg)

   print(f"FMG (3 params): R² = {R2_fmg:.4f}, AIC = {AIC_fmg:.1f}")

   # Step 2: Only if R² < 0.95, try 6-parameter FZLL
   if R2_fmg < 0.95:
       model_fzll = FractionalZenerLiquidLiquid()

       # Initialize from FMG results
       c1_init = model_fmg.parameters.get_value('c')
       alpha_init = model_fmg.parameters.get_value('alpha')
       tau_init = model_fmg.parameters.get_value('tau')

       model_fzll.parameters.set_value('c1', c1_init)
       model_fzll.parameters.set_value('alpha', alpha_init)
       model_fzll.parameters.set_value('tau', tau_init)

       result_fzll = model_fzll.fit(data)
       R2_fzll = result_fzll.r_squared
       AIC_fzll = compute_aic(result_fzll)

       print(f"FZLL (6 params): R² = {R2_fzll:.4f}, AIC = {AIC_fzll:.1f}")

       # Justify complexity
       delta_AIC = AIC_fmg - AIC_fzll
       if delta_AIC > 10:
           print(f"FZLL justified: ΔAIC = {delta_AIC:.1f}")
       else:
           print(f"Prefer FMG: ΔAIC = {delta_AIC:.1f} < 10")
   else:
       print("FMG adequate, FZLL not needed")

See Also
--------

- :doc:`fractional_zener_sl` and :doc:`fractional_zener_ss` — related variants with solid
  plateaus
- :doc:`fractional_maxwell_model` — recoverable when one SpringPot order is suppressed
- :doc:`../flow/carreau` — pair liquid-like viscoelastic spectra with steady-flow fits
- :doc:`../../transforms/owchirp` — fast acquisition of broadband :math:`G^*` for Zener
  fitting
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — worked comparisons of
  all fractional Zener forms

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.FractionalZenerLiquidLiquid`

Usage
-----

Basic Fitting (Advanced Users Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalZenerLiquidLiquid
   from rheojax.core.data import RheoData
   import numpy as np

   # CRITICAL: Only use FZLL if simpler models fail
   # Requires high-quality data: 5+ decades, 100+ points

   # Load experimental data
   omega = np.logspace(-4, 3, 150)  # Very wide range
   G_star = ...  # Complex modulus

   data = RheoData(x=omega, y=G_star, test_mode='oscillation')

   # Initialize model
   model = FractionalZenerLiquidLiquid()

   # Fit with careful initialization
   # Recommended: Start from simpler model
   from rheojax.models import FractionalMaxwellGel
   model_simple = FractionalMaxwellGel()
   model_simple.fit(data)

   # Use simple model results as initial guess
   model.parameters.set_value('c1', model_simple.parameters.get_value('c'))
   model.parameters.set_value('alpha', model_simple.parameters.get_value('alpha'))
   model.parameters.set_value('tau', model_simple.parameters.get_value('tau'))
   model.parameters.set_value('c2', model_simple.parameters.get_value('c') * 0.2)
   model.parameters.set_value('beta', 0.5)
   model.parameters.set_value('gamma', 0.7)

   # Fit FZLL
   result = model.fit(data)

   # Access all 6 parameters
   c1 = model.parameters.get_value('c1')
   c2 = model.parameters.get_value('c2')
   alpha = model.parameters.get_value('alpha')
   beta = model.parameters.get_value('beta')
   gamma = model.parameters.get_value('gamma')
   tau = model.parameters.get_value('tau')

   print(f"FZLL Parameters:")
   print(f"  c₁ = {c1:.2e}, α = {alpha:.3f} (high-freq branch)")
   print(f"  c₂ = {c2:.2e}, γ = {gamma:.3f} (low-freq branch)")
   print(f"  β = {beta:.3f}, τ = {tau:.2e} s (crossover)")
   print(f"  c₁/c₂ = {c1/c2:.2f}")
   print(f"  R² = {result.r_squared:.4f}")

   # CRITICAL: Validate against simpler models
   R2_simple = model_simple.r_squared
   print(f"\nComparison:")
   print(f"  FMG (3 params): R² = {R2_simple:.4f}")
   print(f"  FZLL (6 params): R² = {result.r_squared:.4f}")

Bayesian Inference (Expert Use Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # WARNING: MCMC with 6 parameters requires:
   # - Excellent data quality
   # - Long sampling (5000+ samples)
   # - Multiple chains (4+)
   # - Careful convergence checking

   # NLSQ warm-start (critical for 6 parameters)
   model.fit(data)

   # Bayesian inference with conservative settings
   result = model.fit_bayesian(
       data,
       num_warmup=2000,  # Longer warmup for 6 params
       num_samples=5000,  # More samples needed
       num_chains=4,
       seed=42
   )

   # Check convergence CAREFULLY
   import arviz as az
   inference_data = az.from_numpyro(result)
   summary = az.summary(inference_data, hdi_prob=0.95)
   print(summary)

   # Flag convergence issues
   r_hat_max = summary['r_hat'].max()
   ess_min = summary['ess_bulk'].min()

   print(f"\nConvergence Diagnostics:")
   print(f"  Max R-hat: {r_hat_max:.3f} (want < 1.01)")
   print(f"  Min ESS: {ess_min:.0f} (want > 400)")

   if r_hat_max > 1.01 or ess_min < 400:
       print("  WARNING: Convergence issues detected!")
       print("  - Increase num_samples and num_warmup")
       print("  - Check parameter correlations")
       print("  - Consider simpler model")

   # Visualize 6D posterior (challenging!)
   az.plot_pair(
       inference_data,
       var_names=['c1', 'c2', 'alpha', 'beta', 'gamma', 'tau'],
       kind='kde',
       figsize=(15, 15)
   )

Model Comparison Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # REQUIRED: Always compare to simpler models
   from rheojax.models import (
       FractionalMaxwellGel,
       FractionalMaxwellLiquid,
       FractionalZenerSolidSolid,
       FractionalZenerLiquidLiquid
   )

   models = {
       'FMG (3p)': FractionalMaxwellGel(),
       'FML (4p)': FractionalMaxwellLiquid(),
       'FZSS (4p)': FractionalZenerSolidSolid(),
       'FZLL (6p)': FractionalZenerLiquidLiquid(),
   }

   results = {}
   for name, model in models.items():
       try:
           model.fit(data)
           n_params = len(model.parameters)
           n_data = len(data.x)

           # Compute information criteria
           residuals = data.y - model.predict(data.x, test_mode=data.test_mode)
           SSE = np.sum(np.abs(residuals)**2)
           AIC = n_data * np.log(SSE/n_data) + 2*n_params
           BIC = n_data * np.log(SSE/n_data) + n_params*np.log(n_data)

           results[name] = {
               'R²': model.r_squared if hasattr(model, 'r_squared') else 0,
               'AIC': AIC,
               'BIC': BIC,
               'Params': n_params,
           }
       except Exception as e:
           print(f"{name} failed: {e}")

   # Display comparison
   import pandas as pd
   df = pd.DataFrame(results).T
   df['ΔAIC'] = df['AIC'] - df['AIC'].min()
   df['ΔBIC'] = df['BIC'] - df['BIC'].min()
   print(df.sort_values('AIC'))

   # Decision rule
   best_model = df['AIC'].idxmin()
   delta_aic = df.loc['FZLL (6p)', 'ΔAIC'] if 'FZLL (6p)' in df.index else np.inf

   print(f"\nRecommendation:")
   if best_model == 'FZLL (6p)' and delta_aic < -10:
       print("  FZLL justified (ΔAIC < -10)")
   else:
       print(f"  Use {best_model} instead")
       print(f"  FZLL adds complexity without sufficient improvement")

Advanced Visualization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Visualize multi-order behavior
   import matplotlib.pyplot as plt

   omega = np.logspace(-4, 3, 200)
   G_star = model.predict(omega, test_mode='oscillation')
   G_prime = np.real(G_star)
   G_double_prime = np.imag(G_star)

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Storage modulus
   axes[0, 0].loglog(omega, G_prime, 'o-')
   axes[0, 0].set_xlabel('ω (rad/s)')
   axes[0, 0].set_ylabel("G' (Pa)")
   axes[0, 0].set_title('Storage Modulus')
   axes[0, 0].grid(True)

   # Loss modulus
   axes[0, 1].loglog(omega, G_double_prime, 's-')
   axes[0, 1].set_xlabel('ω (rad/s)')
   axes[0, 1].set_ylabel("G'' (Pa)")
   axes[0, 1].set_title('Loss Modulus')
   axes[0, 1].grid(True)

   # Loss tangent
   tan_delta = G_double_prime / G_prime
   axes[1, 0].semilogx(omega, tan_delta, '^-')
   axes[1, 0].set_xlabel('ω (rad/s)')
   axes[1, 0].set_ylabel('tan(δ)')
   axes[1, 0].set_title('Loss Tangent')
   axes[1, 0].grid(True)

   # Local slopes (power-law analysis)
   log_omega = np.log10(omega)
   log_Gp = np.log10(G_prime)
   slope = np.gradient(log_Gp, log_omega)
   axes[1, 1].semilogx(omega, slope, 'd-')
   axes[1, 1].axhline(alpha, ls='--', label=f'α={alpha:.2f}')
   axes[1, 1].axhline(beta, ls='--', label=f'β={beta:.2f}')
   axes[1, 1].axhline(gamma, ls='--', label=f'γ={gamma:.2f}')
   axes[1, 1].set_xlabel('ω (rad/s)')
   axes[1, 1].set_ylabel("d(log G')/d(log ω)")
   axes[1, 1].set_title('Local Power-Law Exponent')
   axes[1, 1].legend()
   axes[1, 1].grid(True)

   plt.tight_layout()

See also
--------

- :doc:`fractional_zener_sl` and :doc:`fractional_zener_ss` — related variants with solid
  plateaus.
- :doc:`fractional_maxwell_model` — recoverable when one SpringPot order is suppressed.
- :doc:`../flow/carreau` — pair liquid-like viscoelastic spectra with steady-flow fits.
- :doc:`../../transforms/owchirp` — fast acquisition of broadband :math:`G^*` for Zener
  fitting.
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — worked comparisons of
  all fractional Zener forms.

References
----------

.. [1] Garra, R., Gorenflo, R., Polito, F., and Tomovski, Z. "Hilfer-Prabhakar
   derivatives and some applications." *Applied Mathematics and Computation*,
   242, 576–589 (2014). https://doi.org/10.1016/j.amc.2014.05.129

.. [2] Mainardi, F. *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press (2010). https://doi.org/10.1142/p614

.. [3] Schiessel, H., Metzler, R., Blumen, A., and Nonnenmacher, T. F. "Generalized
   viscoelastic models: their fractional equations with solutions."
   *Journal of Physics A*, 28, 6567–6584 (1995).
   https://doi.org/10.1088/0305-4470/28/23/012

.. [4] Heymans, N., and Bauwens, J. C. "Fractal rheological models and fractional
   differential equations for viscoelastic behavior."
   *Rheologica Acta*, 33, 210–219 (1994).
   https://doi.org/10.1007/BF00437306

.. [5] Bagley, R. L., and Torvik, P. J. "A theoretical basis for the application of
   fractional calculus to viscoelasticity." *Journal of Rheology*, 27, 201–210 (1983).
   https://doi.org/10.1122/1.549724
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

