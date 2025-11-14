.. _fitting_strategies:

Fitting Strategies and Troubleshooting
=======================================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Initialize parameters intelligently for faster convergence
   2. Validate fitted models using multiple criteria
   3. Troubleshoot common convergence problems
   4. Use compatibility checking to avoid physics mismatches (NEW v0.2.0)
   5. Apply best practices for robust model fitting

.. admonition:: Prerequisites
   :class: important

   - :doc:`getting_started` — Basic fitting workflow
   - :doc:`model_families` — Understanding model types

Smart Parameter Initialization (v0.2.0)
----------------------------------------

**Problem**: Poor initial guesses cause slow convergence or failure

**Solution**: Automatic smart initialization for fractional models in oscillation mode

How It Works
~~~~~~~~~~~~

For SAOS frequency sweep data, RheoJAX automatically estimates:

1. **Moduli** from low/high-frequency plateaus (G_e, G_m)
2. **Fractional order (α)** from slope of log(G') vs. log(ω)
3. **Characteristic time (τ)** from crossover frequency

.. code-block:: python

   from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

   # Smart initialization happens automatically
   model = FractionalZenerSolidSolid()
   model.fit(omega, G_star, test_mode='oscillation')
   # No manual initialization needed!

**Performance**: Reduces fitting time by 50-80% and improves parameter recovery

Manual Initialization (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Estimate from data
   G_low = np.mean(G_prime[:5])   # Low-frequency plateau
   G_high = np.mean(G_prime[-5:])  # High-frequency plateau

   # Set initial values
   model.parameters.set_value('Ge', G_low)
   model.parameters.set_value('Gm', G_high - G_low)
   model.fit(omega, G_star, test_mode='oscillation')

Model Validation Checklist
---------------------------

Always validate fits using these four criteria:

1. Visual Inspection
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   predictions = model.predict(x_data)

   plt.loglog(x_data, y_data, 'o', label='Data')
   plt.loglog(x_data, predictions, '-', label='Fit')
   plt.legend()

**Look for**: Systematic deviations, curvature mismatch, poor endpoints

2. Residual Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   residuals = y_data - predictions
   relative_error = np.abs(residuals / y_data) * 100

   print(f"Mean error: {np.mean(relative_error):.2f}%")
   print(f"Max error: {np.max(relative_error):.2f}%")

**Acceptable**: Mean error < 5% (excellent), < 10% (good)

3. Parameter Reasonableness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check physical bounds
   G0 = model.parameters.get_value('G0')
   tau = model.parameters.get_value('tau')

   assert 1e0 < G0 < 1e10, f"G0={G0:.2e} Pa out of range"
   assert 1e-6 < tau < 1e6, f"tau={tau:.2e} s unrealistic"

4. Compatibility Checking (NEW v0.2.0)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.utils.compatibility import check_model_compatibility

   compat = check_model_compatibility(
       model, t=time, G_t=G_data, test_mode='relaxation'
   )

   if not compat['compatible']:
       print(f"Warning: {compat['message']}")
       print(f"Alternatives: {compat['alternatives']}")

**What it checks**: Decay type (exponential vs power-law), material type (liquid vs solid vs gel)

Troubleshooting Guide
----------------------

Problem 1: Convergence Failure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: "Optimization did not converge" error

**Solutions**:

.. code-block:: python

   # 1. Increase max iterations
   model.fit(x, y, max_iter=5000)

   # 2. Relax tolerances
   model.fit(x, y, ftol=1e-6, xtol=1e-6)

   # 3. Better initialization (fractional models)
   model.fit(omega, G_star, test_mode='oscillation')  # Smart init

   # 4. Check data quality (outliers, noise)

Problem 2: Unphysical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: G0 < 0, tau = 10⁻¹⁵ s, alpha > 1

**Solutions**:

.. code-block:: python

   # 1. Tighten parameter bounds
   model.parameters.set_bounds('alpha', (0.1, 0.95))

   # 2. Enable compatibility checking
   model.fit(x, y, check_compatibility=True)

   # 3. Try simpler model
   # FZSS failing? Try Zener or Maxwell

Problem 3: Poor Fit Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Symptom**: Mean error > 10%, systematic deviations

**Solutions**:

.. code-block:: python

   # 1. Try more complex model
   # Maxwell → FractionalMaxwellLiquid
   # Zener → FractionalZenerSolidSolid

   # 2. Check test mode is correct
   model.fit(omega, G_star, test_mode='oscillation')  # Not 'relaxation'!

   # 3. Use compatibility checking for model suggestions

Key Concepts
------------

.. admonition:: Main Takeaways
   :class: tip

   1. **Smart initialization** (v0.2.0): Automatic for fractional models in oscillation mode

   2. **Validation quad**: Visual + residuals + parameters + compatibility

   3. **Compatibility checking**: Physics-based model selection guidance

   4. **Start simple**: Classical → Fractional → Multi-mode

   5. **Troubleshooting**: Increase iterations, better init, check compatibility

.. admonition:: Self-Check Questions
   :class: tip

   1. **Your fit has mean error = 2% but G0 = -500 Pa. Is this a good fit?**

   2. **Optimization fails. What are three things to try?**

   3. **How does compatibility checking help model selection?**

   4. **When should you manually initialize parameters vs. using smart init?**

   5. **What does it mean if residuals show systematic curvature?**

Further Reading
---------------

- :doc:`../03_advanced_topics/bayesian_inference` — Uncertainty quantification
- :doc:`../05_appendices/troubleshooting` — Comprehensive troubleshooting guide
- :doc:`/models/index` — Model equations and details

Summary
-------

Robust fitting requires smart initialization (automatic for fractional models), comprehensive validation (visual, residuals, parameters, compatibility), and systematic troubleshooting. The v0.2.0 compatibility checking system provides physics-based guidance for model selection, reducing trial-and-error.

Next Steps
----------

Proceed to: :doc:`../03_advanced_topics/index`

Learn Bayesian inference for uncertainty quantification and advanced analysis techniques.
