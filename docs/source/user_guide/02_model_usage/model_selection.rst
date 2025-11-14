Model Selection Guide
=====================

Choosing the right rheological model is critical for accurately characterizing material behavior. This guide helps you select the appropriate model based on your material type, test mode, and experimental objectives.

For governing equations, parameter tables, and troubleshooting notes per model, refer to the
:doc:`/models/index` handbook while using this decision guide.

Quick Selection Flowchart
-------------------------

.. code-block:: text

   Data Type?
   |-- Time Domain (Relaxation/Creep)
   |   |-- Decay Type?
   |   |   |-- Exponential decay -> Maxwell, Zener
   |   |   |-- Power-law decay -> FractionalMaxwellGel, FZSS
   |   |   \-- Finite equilibrium modulus -> Zener, FZSS, FKV
   |   \-- Material Type?
   |       |-- Liquid-like (flows) -> Maxwell, FractionalMaxwellLiquid
   |       |-- Solid-like (elastic) -> Zener, FZSS, FractionalKelvinVoigt
   |       \-- Gel-like -> FractionalMaxwellGel
   \-- Frequency Domain (Oscillation)
       |-- Low-frequency behavior?
       |   |-- G' > G" -> Solid-like models (FZSS, FKV, Zener)
       |   \-- G" > G' -> Liquid-like models (Maxwell, FML)
       \-- Slope in log-log plot?
           |-- ~2 (liquid) -> Maxwell, FML
           |-- ~0.5 (gel) -> FractionalMaxwellGel
           \-- plateau (solid) -> FZSS, Zener, FKV

Decision Tree: Which Model for Which Material?
----------------------------------------------

Quick Selection Table
~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model Selection Quick Reference
   :header-rows: 1
   :widths: 20 30 25 25

   * - Material Type
     - Test Modes
     - Recommended Models
     - Model Family
   * - **Elastic solids**
     - Relaxation, Oscillation
     - Zener, SpringPot
     - Classical, Fractional
   * - **Viscoelastic polymers**
     - Relaxation, Oscillation
     - Maxwell, Zener, Fractional Maxwell
     - Classical, Fractional Maxwell
   * - **Gels and soft solids**
     - Oscillation
     - SpringPot, FractionalKelvinVoigt, FractionalZener
     - Fractional
   * - **Polymer solutions**
     - Rotation (flow)
     - Carreau, Cross, CarreauYasuda
     - Flow models
   * - **Pastes and suspensions**
     - Rotation (flow)
     - HerschelBulkley, Bingham, PowerLaw
     - Flow models
   * - **Yield stress fluids**
     - Rotation (flow)
     - HerschelBulkley, Bingham
     - Flow models
   * - **Complex materials**
     - Multiple modes
     - FractionalBurgers, FractionalPoyntingThomson
     - Advanced Fractional

Material-Specific Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Polymers (Molten and Solid)**

For thermoplastic polymers above T_g:

1. **Simple characterization**: Start with :class:`Maxwell` (2 parameters)

   .. code-block:: python

      from rheojax.models import Maxwell
      model = Maxwell()
      model.fit(omega, G_star)
      # G_s: rubbery modulus, eta_s: viscosity

2. **Improved accuracy**: Use :class:`Zener` or :class:`FractionalMaxwellGel` (3 parameters)

   .. code-block:: python

      from rheojax.models import FractionalMaxwellGel
      model = FractionalMaxwellGel()
      # G_s: modulus, V: fractional viscosity, alpha: fractional order

3. **Best accuracy**: Use :class:`FractionalMaxwellModel` (4 parameters) for full frequency range

   .. code-block:: python

      from rheojax.models import FractionalMaxwellModel
      model = FractionalMaxwellModel()
      # Most flexible for wide frequency sweeps

**Gels and Soft Solids**

For hydrogels, organogels, and soft biological materials:

1. **Power-law behavior**: Use :class:`SpringPot` (2 parameters)

   .. code-block:: python

      from rheojax.models import SpringPot
      model = SpringPot()
      # V: fractional stiffness, alpha: power-law exponent (0.1-0.5 typical)

2. **Solid-like with relaxation**: Use :class:`FractionalKelvinVoigt` (3-4 parameters)

   .. code-block:: python

      from rheojax.models import FractionalKelvinVoigt
      model = FractionalKelvinVoigt()
      # Captures both elastic plateau and slow relaxation

3. **Complex gel networks**: Use :class:`FractionalZenerSolidSolid` (4 parameters)

   .. code-block:: python

      from rheojax.models import FractionalZenerSolidSolid, FZSS
      model = FZSS()  # Short alias available
      # Two elastic components + fractional element

**Polymer Solutions**

For dilute to concentrated polymer solutions in flow:

1. **Shear thinning only**: Use :class:`PowerLaw` (2 parameters)

   .. code-block:: python

      from rheojax.models import PowerLaw
      model = PowerLaw()
      # K: consistency, n: flow index (n<1 for shear thinning)

2. **Newtonian -> shear thinning**: Use :class:`Carreau` or :class:`Cross` (4 parameters)

   .. code-block:: python

      from rheojax.models import Carreau
      model = Carreau()
      # eta_0: zero-shear viscosity, eta_inf: infinite-shear viscosity
      # lambda: time constant, n: power-law index

3. **Complex transition behavior**: Use :class:`CarreauYasuda` (5 parameters)

   .. code-block:: python

      from rheojax.models import CarreauYasuda
      model = CarreauYasuda()
      # Additional parameter 'a' controls transition sharpness

**Pastes and Suspensions**

For concentrated suspensions, pastes, and yield stress fluids:

1. **Yield stress + power-law**: Use :class:`HerschelBulkley` (3 parameters)

   .. code-block:: python

      from rheojax.models import HerschelBulkley
      model = HerschelBulkley()
      # tau_0: yield stress, K: consistency, n: flow index

2. **Yield stress + Newtonian**: Use :class:`Bingham` (2 parameters)

   .. code-block:: python

      from rheojax.models import Bingham
      model = Bingham()
      # tau_0: yield stress, eta_pl: plastic viscosity
      # Simpler alternative to Herschel-Bulkley when n approx 1


Model Complexity vs Fitting Quality Trade-offs
----------------------------------------------

Parameter Count and Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**2-Parameter Models (Simplest)**

Advantages:

- Fast fitting (<0.1s typical)
- Fewer parameters = less overfitting risk
- Easy physical interpretation
- Good for quick screening

Models:

- :class:`Maxwell`: G_s, eta_s
- :class:`SpringPot`: V, alpha
- :class:`PowerLaw`: K, n
- :class:`Bingham`: tau_0, eta_pl

Best for: Initial characterization, simple materials, limited data

**3-Parameter Models (Moderate Complexity)**

Advantages:

- Good balance of accuracy and interpretability
- Captures key material features
- Still relatively fast fitting (<0.5s typical)

Models:

- :class:`Zener`: G_s, G_p, eta_p
- :class:`FractionalMaxwellGel`: G_s, V, alpha
- :class:`FractionalMaxwellLiquid`: V, alpha, eta_s
- :class:`HerschelBulkley`: tau_0, K, n

Best for: Most engineering applications, moderate data quality

**4-Parameter Models (High Complexity)**

Advantages:

- Excellent accuracy over wide ranges
- Captures multiple relaxation/flow regimes
- Best for research-grade data

Models:

- :class:`FractionalMaxwellModel`: Two SpringPots in series
- :class:`FractionalKelvinVoigt`: G_p, V, alpha, eta_p
- :class:`Carreau`: eta_0, eta_inf, lambda, m
- :class:`Cross`: eta_0, eta_inf, K, m
- Most Fractional Zener variants: 4 parameters

Best for: High-quality data, wide frequency/shear rate range, publication results

**5+ Parameter Models (Maximum Complexity)**

Advantages:

- Maximum flexibility
- Can fit complex transitions
- Multiple characteristic times/rates

Models:

- :class:`CarreauYasuda`: 5 parameters
- :class:`FractionalBurgersModel`: 5 parameters
- :class:`FractionalPoyntingThomson`: 5 parameters

Best for: Research applications, very high-quality data, complex materials

Risks: Overfitting, parameter correlation, slow convergence

Model Selection Strategy
~~~~~~~~~~~~~~~~~~~~~~~~

Follow this systematic approach:

1. **Start simple**: Begin with 2-3 parameter model

   .. code-block:: python

      from rheojax.models import Maxwell, Zener
      from rheojax.core.registry import ModelRegistry

      # Try Maxwell first
      maxwell = Maxwell()
      maxwell.fit(X, y)
      r2_maxwell = maxwell.score(X, y)

      # Try Zener if Maxwell insufficient
      zener = Zener()
      zener.fit(X, y)
      r2_zener = zener.score(X, y)

      print(f"Maxwell R^2: {r2_maxwell:.4f}")
      print(f"Zener R^2: {r2_zener:.4f}")

2. **Check fit quality**: Look at R^2, residuals, physical parameter values

   .. code-block:: python

      import matplotlib.pyplot as plt
      import numpy as np

      # Calculate residuals
      y_pred = model.predict(X)
      residuals = y - y_pred
      relative_error = np.abs(residuals / y) * 100

      # Visualize
      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

      # Fit plot
      ax1.loglog(X, y, 'o', label='Data')
      ax1.loglog(X, y_pred, '-', label='Model')
      ax1.set_xlabel('Frequency (rad/s)')
      ax1.set_ylabel('|G*| (Pa)')
      ax1.legend()

      # Residuals
      ax2.semilogx(X, relative_error, 'o')
      ax2.axhline(y=5, color='r', linestyle='--', label='5% threshold')
      ax2.set_xlabel('Frequency (rad/s)')
      ax2.set_ylabel('Relative Error (%)')
      ax2.legend()

3. **Increase complexity if needed**: Move to fractional models if classical models show systematic errors

   .. code-block:: python

      # If Maxwell/Zener have systematic errors in mid-frequency
      from rheojax.models import FractionalMaxwellGel

      fmg = FractionalMaxwellGel()
      fmg.fit(X, y)
      r2_fmg = fmg.score(X, y)

      if r2_fmg - r2_zener > 0.02:  # Significant improvement
          print("Fractional model provides better fit")
      else:
          print("Classical model sufficient")

4. **Use information criteria**: AIC/BIC to balance fit quality vs complexity

   .. code-block:: python

      import numpy as np

      def calculate_aic(model, X, y):
          """Akaike Information Criterion (lower is better)."""
          n = len(y)
          k = len(model.parameters)
          y_pred = model.predict(X)
          rss = np.sum((y - y_pred)**2)
          aic = n * np.log(rss/n) + 2 * k
          return aic

      def calculate_bic(model, X, y):
          """Bayesian Information Criterion (lower is better)."""
          n = len(y)
          k = len(model.parameters)
          y_pred = model.predict(X)
          rss = np.sum((y - y_pred)**2)
          bic = n * np.log(rss/n) + k * np.log(n)
          return bic

      # Compare models
      aic_maxwell = calculate_aic(maxwell, X, y)
      aic_zener = calculate_aic(zener, X, y)
      aic_fmg = calculate_aic(fmg, X, y)

      print(f"AIC - Maxwell: {aic_maxwell:.1f}, Zener: {aic_zener:.1f}, FMG: {aic_fmg:.1f}")
      # Lower AIC indicates better model given complexity penalty

When to Use Fractional vs Classical Models
------------------------------------------

Classical Models (Integer-Order Derivatives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use classical models when**:

- Material behavior is well-described by simple spring-dashpot combinations
- Limited frequency/time range (< 2-3 decades)
- Fast computation required (real-time, high-throughput)
- Simple physical interpretation paramount
- Educational/teaching contexts

**Examples**:

- Maxwell model for polymer melts in narrow frequency range
- Zener model for viscoelastic solids with single relaxation time
- Simple power-law for flow behavior in limited shear rate range

Fractional Models (Fractional-Order Derivatives)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use fractional models when**:

- Power-law behavior observed over wide frequency/time range
- Broad distribution of relaxation times
- Self-similar or fractal structure
- Classical models show systematic deviations

**Physical Interpretation**:

The fractional order alpha has physical meaning:

- alpha = 0: Pure elastic (spring)
- alpha = 1: Pure viscous (dashpot)
- 0 < alpha < 1: Fractional viscoelastic (intermediate)

For polymers and soft matter:

- alpha approx 0.1-0.3: Highly entangled systems, strong caging
- alpha approx 0.4-0.6: Moderate entanglement, broad relaxation spectrum
- alpha approx 0.7-0.9: Weakly entangled, approaching liquid-like

**Example comparison**:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from rheojax.models import Zener, FractionalMaxwellGel

   # Generate synthetic data with power-law decay
   omega = np.logspace(-2, 2, 50)
   # True behavior: G' ~ omega^0.4 (fractional)
   G_prime_true = 1000 * omega**0.4

   # Fit with classical Zener (will struggle)
   zener = Zener()
   zener.fit(omega, G_prime_true)
   G_zener = zener.predict(omega)

   # Fit with fractional model (should work well)
   fmg = FractionalMaxwellGel()
   fmg.fit(omega, G_prime_true)
   G_fmg = fmg.predict(omega)

   # Plot comparison
   plt.figure(figsize=(10, 6))
   plt.loglog(omega, G_prime_true, 'ko', label='Data', markersize=8)
   plt.loglog(omega, G_zener, 'b--', label='Zener (classical)', linewidth=2)
   plt.loglog(omega, G_fmg, 'r-', label='FractionalMaxwellGel', linewidth=2)
   plt.xlabel('Frequency (rad/s)')
   plt.ylabel("G' (Pa)")
   plt.legend()
   plt.title('Fractional models excel at power-law behavior')
   plt.grid(True, alpha=0.3)

Hybrid Approach
~~~~~~~~~~~~~~~

For complex materials, consider **multi-technique fitting** with fractional models:

.. code-block:: python

   from rheojax.models import FractionalMaxwellModel
   from rheojax.core import SharedParameterSet

   # Use same model for both oscillation and relaxation data
   shared_params = SharedParameterSet()
   shared_params.add_shared('V', value=1000, bounds=(100, 10000))
   shared_params.add_shared('alpha', value=0.5, bounds=(0.1, 0.9))

   model_osc = FractionalMaxwellModel()
   model_relax = FractionalMaxwellModel()

   shared_params.link_model(model_osc, 'V')
   shared_params.link_model(model_osc, 'alpha')
   shared_params.link_model(model_relax, 'V')
   shared_params.link_model(model_relax, 'alpha')

   # Fit both datasets with shared parameters
   # (ensures physical consistency)


Complete Model Catalog
----------------------

Classical Models (3 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Parameters
     - Test Modes
     - Material Types
   * - **Maxwell**
     - G_s (Pa), eta_s (Pa*s)
     - Relaxation, Oscillation
     - Polymer melts, simple viscoelastic
   * - **Zener**
     - G_s, G_p (Pa), eta_p (Pa*s)
     - Relaxation, Creep, Oscillation
     - Viscoelastic solids, SLS
   * - **SpringPot**
     - V (Pa*s^alpha), alpha (-)
     - Oscillation, Relaxation
     - Power-law materials, gels

Fractional Maxwell Family (4 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Parameters (count)
     - Best For
     - Complexity
   * - **FractionalMaxwellGel**
     - 3: G_s, V, alpha
     - Gels with elastic component
     - Low
   * - **FractionalMaxwellLiquid**
     - 3: V, alpha, eta_s
     - Polymer solutions with memory
     - Low
   * - **FractionalMaxwellModel**
     - 4: Two SpringPots
     - Wide frequency range, general
     - Medium
   * - **FractionalKelvinVoigt**
     - 3-4: G_p, V, alpha, (eta_p)
     - Solid-like with slow relaxation
     - Medium

Fractional Zener Family (4 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Parameters (count)
     - Physical Meaning
     - Use Case
   * - **FractionalZenerSolidLiquid (FZSL)**
     - 4: G_s, eta_s, V, alpha
     - Solid + fractional liquid
     - Polymer melts with plateau
   * - **FractionalZenerSolidSolid (FZSS)**
     - 4: G_s, G_p, V, alpha
     - Two solids + fractional
     - Filled polymers, composites
   * - **FractionalZenerLiquidLiquid (FZLL)**
     - 4: eta_s, eta_p, V, alpha
     - Most general
     - Complex polymer systems
   * - **FractionalKelvinVoigtZener (FKVZ)**
     - 4: Parameters
     - FKV + series spring
     - Soft solids with compliance

Advanced Fractional Models (3 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 30 25 25

   * - Model
     - Parameters (count)
     - Structure
     - Application
   * - **FractionalBurgersModel (FBM)**
     - 5: Complex
     - Maxwell + FKV in series
     - Polymers with creep + relaxation
   * - **FractionalPoyntingThomson (FPT)**
     - 5: Complex
     - FKV + spring in series
     - Similar to Burgers, alternate
   * - **FractionalJeffreysModel (FJM)**
     - 4: Dashpots + SpringPot
     - Two dashpots + fractional
     - Polymer solutions

Non-Newtonian Flow Models (6 models)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Model
     - Parameters
     - Flow Behavior
     - Typical Applications
   * - **PowerLaw**
     - K (Pa*s^n), n (-)
     - Shear thinning/thickening
     - Simple shear rate dependence
   * - **Carreau**
     - eta_0, eta_inf, lambda, m
     - Smooth Newtonian -> power-law
     - Polymer solutions
   * - **CarreauYasuda**
     - Carreau + parameter 'a'
     - Sharper transition control
     - Complex polymer solutions
   * - **Cross**
     - eta_0, eta_inf, K, m
     - Alternative to Carreau
     - Polymer melts, alternatives
   * - **HerschelBulkley**
     - tau_0, K, n
     - Yield stress + power-law
     - Pastes, suspensions, food
   * - **Bingham**
     - tau_0, eta_pl
     - Yield stress + Newtonian
     - Drilling fluids, cements

Best Practices
--------------

Model Selection Checklist
~~~~~~~~~~~~~~~~~~~~~~~~~

Before fitting:

1. **Identify your test mode**: Relaxation, creep, oscillation, or rotation?
2. **Know your frequency/time range**: Wide (>3 decades) or narrow (<2 decades)?
3. **Understand your material**: Solid-like, liquid-like, or intermediate?
4. **Check data quality**: Noise level, number of points, dynamic range
5. **Define success criteria**: What R^2 or error level is acceptable?

During fitting:

1. **Use reasonable initial guesses**: Order-of-magnitude estimates from data
2. **Set physical bounds**: E.g., moduli > 0, 0 < alpha < 1 for fractional order
3. **Monitor convergence**: Check optimization messages
4. **Validate parameter values**: Are they physically reasonable?
5. **Examine residuals**: Random or systematic errors?

After fitting:

1. **Compare multiple models**: Use AIC/BIC for objective comparison
2. **Cross-validate**: Hold-out test sets or k-fold if sufficient data
3. **Check physical consistency**: Parameters should match material expectations
4. **Visualize predictions**: Always plot data vs model
5. **Report uncertainties**: Especially for publication

Common Pitfalls and How to Avoid Them
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pitfall 1: Overfitting with too many parameters**

Solution: Use AIC/BIC, cross-validation, prefer simpler models when possible

**Pitfall 2: Poor initial guesses leading to local minima**

Solution: Use data-driven initialization, try multiple starting points

.. code-block:: python

   # Good practice: data-driven initial guess
   G_max = np.max(np.abs(G_star))
   G_min = np.min(np.abs(G_star))

   model = FractionalMaxwellGel()
   model.parameters.set_value('G_s', G_min)  # Rubbery modulus ~ low freq
   model.parameters.set_value('V', G_max)    # Fractional visc ~ high freq
   model.parameters.set_value('alpha', 0.5)  # Mid-range fractional order

**Pitfall 3: Ignoring test mode compatibility**

Solution: Check model documentation for supported test modes

.. code-block:: python

   from rheojax.core.test_modes import get_compatible_test_modes

   model = HerschelBulkley()
   compatible = get_compatible_test_modes('herschel_bulkley')
   print(f"Compatible test modes: {compatible}")
   # Output: ['rotation'] - only works for steady shear!

**Pitfall 4: Fitting noisy data without preprocessing**

Solution: Use smoothing transforms before fitting

.. code-block:: python

   from rheojax.transforms import SmoothDerivative

   # Smooth noisy data before fitting
   smoother = SmoothDerivative(method='savgol', window=11, order=2)
   data_smooth = smoother.transform(data)

   # Now fit model to smoothed data
   model.fit(data_smooth.x, data_smooth.y)

**Pitfall 5: Not checking parameter correlation**

Solution: Examine parameter confidence intervals and correlations

.. code-block:: python

   # After fitting, check parameter sensitivity
   from rheojax.utils.optimization import calculate_confidence_intervals

   ci = calculate_confidence_intervals(model, X, y, alpha=0.05)
   print("95% Confidence Intervals:")
   for param, (lower, upper) in ci.items():
       value = model.parameters.get_value(param)
       print(f"  {param}: {value:.2e} [{lower:.2e}, {upper:.2e}]")

Data-Driven Selection Criteria
------------------------------

Based on Relaxation Modulus G(t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear in log(G) vs t** (Semi-log plot)
   -> Exponential decay -> **Maxwell** or **Zener**

**Linear in log(G) vs log(t)** (Log-log plot)
   -> Power-law decay -> **FractionalMaxwellGel** or **FZSS**

**Plateau at long times**
   -> Finite equilibrium modulus -> **Zener**, **FZSS**, or **FKV**

**No plateau (G -> 0)**
   -> Liquid-like -> **Maxwell** or **FractionalMaxwellLiquid**

Based on Complex Modulus G*(omega)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Low-frequency slope in log(G') vs log(omega)**:

- Slope approx 2: Liquid -> **Maxwell**, **FML**
- Slope approx 0: Solid -> **Zener**, **FZSS**, **FKV**
- Slope approx alpha (0 < alpha < 1): Gel -> **FractionalMaxwellGel**

**G'/G" crossover**:

- Present: Relaxation time identifiable -> **Maxwell**, **Zener**
- Absent (G' > G" always): Strong solid -> **FKV**, **FZSS**
- Absent (G" > G' always): Strong liquid -> **Maxwell**, **FML**

Based on Creep Compliance J(t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linear in log(J) vs t**
   -> Exponential creep -> Classical models

**Linear in log(J) vs log(t)**
   -> Power-law creep -> **FractionalKelvinVoigt**, **FKVZ**

**Finite equilibrium compliance**
   -> Solid-like -> **FKVZ**, **Zener**

**Unbounded compliance**
   -> Liquid-like -> **Maxwell**, **FML**

Automatic Compatibility Checking
--------------------------------

RheoJAX provides automatic compatibility checking to help identify inappropriate models:

.. code-block:: python

   from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid
   from rheojax.utils.compatibility import check_model_compatibility, format_compatibility_message

   # Check before fitting
   model = FractionalZenerSolidSolid()
   compat = check_model_compatibility(
       model,
       t=time_data,
       G_t=modulus_data,
       test_mode='relaxation'
   )

   # Print compatibility report
   print(format_compatibility_message(compat))

   # Or enable automatic checking during fit
   model.fit(time_data, modulus_data, check_compatibility=True)

The system will:

- Detect decay type (exponential, power-law, etc.)
- Identify material type (solid, liquid, gel)
- Warn about incompatibilities
- Suggest alternative models

Parameter Bounds Reference
--------------------------

All models have physically reasonable default bounds:

- **Moduli (G, E)**: 1 Pa to 1 GPa
- **Viscosity (eta)**: 1 mPa*s to 1 MPa*s
- **Time constants (tau)**: 1 mus to 1 Ms
- **Fractional orders (alpha)**: 0 to 1

Adjust bounds if your material is outside these ranges.

Summary
-------

**Quick Decision Path**:

1. **Test mode is rotation (steady shear)**: Use flow models (PowerLaw -> Carreau/Cross -> HerschelBulkley)
2. **Test mode is oscillation or relaxation**:

   - Simple material, narrow range: Classical models (Maxwell, Zener)
   - Power-law behavior: SpringPot or fractional models
   - Wide frequency range: Fractional Maxwell or Zener families
   - Very complex: Advanced fractional models

3. **Start simple, increase complexity only if justified by data quality and fit improvement**

4. **Always validate with residual analysis, cross-validation, and physical parameter checks**

Further Reading
---------------

**Foundational Texts**:

- **Mainardi (2010)**: Fractional Calculus and Waves in Linear Viscoelasticity
- **Ferry (1980)**: Viscoelastic Properties of Polymers
- **Barnes et al. (1989)**: An Introduction to Rheology
- **Tschoegl (1989)**: The Phenomenological Theory of Linear Viscoelastic Behavior

**RheoJAX Documentation**:

- :doc:`/user_guide/modular_api` - Direct model usage
- :doc:`/user_guide/pipeline_api` - High-level workflows
- :doc:`/user_guide/bayesian_inference` - Bayesian model comparison
- ``examples/model_comparison.ipynb`` - Side-by-side model comparison

Getting Help with Model Selection
---------------------------------

If you're unsure which model to use:

1. **Run compatibility check first** - Use ``check_model_compatibility()``
2. **Fit 3-4 candidate models** - Start simple, add complexity
3. **Compare metrics** - R^2, AIC, BIC
4. **Check physical reasonableness** - Do parameters make sense?
5. **Validate on held-out data** - Cross-validation or test sets

For advanced cases, consider:

- **Bayesian model selection** - Use ``fit_bayesian()`` with WAIC/LOO
- **Cross-validation** - k-fold or time-series splits
- **Physical constraints** - Material science domain knowledge
- **Multi-technique fitting** - Combine relaxation, oscillation, and flow data
