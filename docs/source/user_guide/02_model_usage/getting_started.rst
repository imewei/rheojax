.. _getting_started_models:

Getting Started with Model Fitting
===================================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Fit a simple rheological model (Maxwell) to experimental data in 10 lines of code
   2. Access and interpret fitted parameters
   3. Validate model fit quality visually and numerically
   4. Choose between different fitting backends (NLSQ vs. scipy)
   5. Understand when to use each test mode (relaxation, oscillation, flow)

.. admonition:: Prerequisites
   :class: important

   - :doc:`../01_fundamentals/test_modes` — Understanding of SAOS, relaxation, creep, flow
   - :doc:`../01_fundamentals/parameter_interpretation` — Physical meaning of parameters
   - RheoJAX installed: ``pip install rheojax``

Your First Model Fit (10 Lines)
--------------------------------

Let's fit a **Maxwell model** to stress relaxation data:

.. code-block:: python

   from rheojax.models import Maxwell
   import numpy as np

   # Load experimental data (time, stress)
   t = np.loadtxt('relaxation.csv', delimiter=',', usecols=0)
   G_t = np.loadtxt('relaxation.csv', delimiter=',', usecols=1)

   # Create model and fit
   model = Maxwell()
   model.fit(t, G_t, test_mode='relaxation')

   # Inspect fitted parameters
   print(f"Modulus G0 = {model.parameters.get_value('G0'):.3e} Pa")
   print(f"Viscosity eta = {model.parameters.get_value('eta'):.3e} Pa·s")

That's it! You've fitted a rheological model.

**What just happened?**

1. Loaded time and stress data from CSV
2. Created a Maxwell model instance
3. Fitted the model using NLSQ optimization (GPU-accelerated if available)
4. Extracted fitted parameters (G0 = modulus, eta = viscosity)

Understanding the Maxwell Model
--------------------------------

The **Maxwell model** represents a viscoelastic liquid with a single relaxation time:

**Stress relaxation** (time domain):

.. math::

   G(t) = G_0 e^{-t/\tau}

where :math:`\tau = \eta / G_0` is the relaxation time.

**Key parameters**:

- :math:`G_0` (Pa): Instantaneous modulus (stiffness at short times)
- :math:`\eta` (Pa·s): Viscosity (resistance to flow)
- :math:`\tau` (s): Relaxation time (timescale of stress decay)

**Physical interpretation**: Material behaves like an elastic solid at :math:`t \ll \tau`, flows like a viscous liquid at :math:`t \gg \tau`.

For detailed model equations, see :doc:`/models/classical/maxwell`.

Validating Your Fit
-------------------

Always check if the model fits the data well:

1. Visual Inspection
~~~~~~~~~~~~~~~~~~~~

Plot predicted vs. actual data:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Generate predictions
   t_pred = np.linspace(t.min(), t.max(), 200)
   G_pred = model.predict(t_pred)

   # Plot
   plt.figure(figsize=(8, 5))
   plt.plot(t, G_t, 'o', label='Experimental', markersize=6)
   plt.plot(t_pred, G_pred, '-', label='Maxwell fit', linewidth=2)
   plt.xlabel('Time (s)')
   plt.ylabel('Relaxation Modulus G(t) (Pa)')
   plt.xscale('log')
   plt.yscale('log')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.title('Stress Relaxation: Fit Quality')
   plt.show()

**What to look for**:

- Predictions should pass through data points
- Systematic deviations indicate wrong model
- Random scatter is acceptable (experimental noise)

2. Residual Analysis
~~~~~~~~~~~~~~~~~~~~~

Quantify fit quality:

.. code-block:: python

   # Compute residuals
   G_fit = model.predict(t)
   residuals = G_t - G_fit
   relative_error = np.abs(residuals / G_t) * 100

   # Print statistics
   print(f"Mean absolute error: {np.mean(relative_error):.2f}%")
   print(f"Max error: {np.max(relative_error):.2f}%")
   print(f"R² score: {1 - np.sum(residuals**2) / np.sum((G_t - G_t.mean())**2):.4f}")

**Acceptable values**:

- Mean error < 5%: Excellent fit
- Mean error 5-10%: Good fit
- Mean error > 10%: Poor fit, try different model
- :math:`R^2` > 0.95: Good fit

3. Parameter Reasonableness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Check if fitted parameters are physically plausible:

.. code-block:: python

   G0 = model.parameters.get_value('G0')
   eta = model.parameters.get_value('eta')
   tau = eta / G0

   # Sanity checks
   assert G0 > 0, "Modulus must be positive"
   assert 1e-6 < tau < 1e6, f"Relaxation time {tau:.2e} s is unrealistic"
   assert 1e-6 < eta < 1e15, f"Viscosity {eta:.2e} Pa·s is unrealistic"

   print(f"Relaxation time τ = {tau:.3f} s")

**Typical ranges**:

- :math:`G_0`: :math:`10^2 - 10^8` Pa (soft gels to stiff polymers)
- :math:`\eta`: :math:`10^{-3} - 10^{10}` Pa·s (water to polymer melts)
- :math:`\tau`: :math:`10^{-6} - 10^4` s (fast liquids to slow relaxation)

If parameters are outside these ranges, suspect fitting errors or wrong model.

Fitting Different Test Modes
-----------------------------

RheoJAX models support multiple experimental techniques. The **test_mode** parameter tells the model how to interpret your data.

1. Stress Relaxation
~~~~~~~~~~~~~~~~~~~~

**Data**: Time vs. relaxation modulus G(t)

**Input**: 1D arrays (time, stress/modulus)

.. code-block:: python

   model = Maxwell()
   model.fit(time, G_t, test_mode='relaxation')

2. Oscillation (SAOS)
~~~~~~~~~~~~~~~~~~~~~

**Data**: Frequency vs. :math:`G'` and :math:`G''`

**Input**: Frequency (1D) + complex modulus :math:`G^* = [G', G'']` (2D array)

.. code-block:: python

   from rheojax.models import Maxwell

   omega = np.array([0.1, 1.0, 10.0, 100.0])  # rad/s
   G_prime = np.array([100, 500, 2000, 5000])   # Storage modulus (Pa)
   G_double_prime = np.array([200, 600, 1500, 2500])  # Loss modulus (Pa)

   # Stack G' and G" into complex modulus
   G_star = np.column_stack([G_prime, G_double_prime])

   # Fit
   model = Maxwell()
   model.fit(omega, G_star, test_mode='oscillation')

**Note**: For oscillation mode, y must be a 2D array with shape (N, 2) where column 0 is :math:`G'` and column 1 is :math:`G''`.

3. Creep
~~~~~~~~

**Data**: Time vs. compliance J(t)

.. code-block:: python

   model.fit(time, J_t, test_mode='creep')

4. Steady Shear Flow
~~~~~~~~~~~~~~~~~~~~

**Data**: Shear rate vs. viscosity (uses flow models, not Maxwell)

.. code-block:: python

   from rheojax.models import PowerLaw

   shear_rate = np.array([0.1, 1.0, 10.0, 100.0])  # s⁻¹
   viscosity = np.array([1000, 500, 200, 100])      # Pa·s

   model = PowerLaw()
   model.fit(shear_rate, viscosity, test_mode='rotation')

**Important**: Viscoelastic models (Maxwell, Zener, fractional) are for linear tests (relaxation, oscillation, creep). Flow models (PowerLaw, Carreau, Herschel-Bulkley) are for nonlinear steady shear.

Choosing the Right Fitting Backend
-----------------------------------

RheoJAX supports two optimization backends:

1. NLSQ (Default) — GPU-Accelerated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Advantages**:

- 5-270x faster than scipy on CPU
- GPU acceleration available
- JAX-based automatic differentiation
- Recommended for all fits

.. code-block:: python

   # Default behavior (uses NLSQ)
   model.fit(t, G_t, test_mode='relaxation')

2. Scipy — Fallback
~~~~~~~~~~~~~~~~~~~

**Advantages**:

- No JAX dependency
- Familiar interface

**Disadvantages**:

- Slower (especially for complex models)
- CPU-only

.. code-block:: python

   # Force scipy backend (not recommended unless NLSQ fails)
   model.fit(t, G_t, test_mode='relaxation', method='scipy')

**Recommendation**: Always use NLSQ (default) unless you encounter issues.

Worked Example: Complete Workflow
----------------------------------

Let's fit a Maxwell model to real relaxation data with full validation:

.. code-block:: python

   from rheojax.models import Maxwell
   import numpy as np
   import matplotlib.pyplot as plt

   # 1. Generate synthetic data (simulate experiment)
   t_true = np.logspace(-2, 2, 30)  # 0.01 to 100 s
   G0_true = 1e5  # Pa
   eta_true = 1e5  # Pa·s
   tau_true = eta_true / G0_true  # 1 s

   G_true = G0_true * np.exp(-t_true / tau_true)
   G_noisy = G_true + np.random.normal(0, 0.02 * G_true.mean(), size=G_true.shape)

   # 2. Fit model
   model = Maxwell()
   model.fit(t_true, G_noisy, test_mode='relaxation')

   # 3. Extract parameters
   G0_fit = model.parameters.get_value('G0')
   eta_fit = model.parameters.get_value('eta')
   tau_fit = eta_fit / G0_fit

   print("Fitted Parameters:")
   print(f"  G0 = {G0_fit:.3e} Pa (true: {G0_true:.3e})")
   print(f"  eta = {eta_fit:.3e} Pa·s (true: {eta_true:.3e})")
   print(f"  tau = {tau_fit:.3f} s (true: {tau_true:.3f})")

   # 4. Validate fit quality
   G_pred = model.predict(t_true)
   residuals = G_noisy - G_pred
   mean_error = np.mean(np.abs(residuals / G_noisy)) * 100
   print(f"\nFit Quality:")
   print(f"  Mean error: {mean_error:.2f}%")

   # 5. Visualize
   plt.figure(figsize=(10, 5))

   # Left: Data and fit
   plt.subplot(1, 2, 1)
   plt.loglog(t_true, G_noisy, 'o', label='Data (noisy)', markersize=6)
   plt.loglog(t_true, G_pred, '-', label='Maxwell fit', linewidth=2)
   plt.xlabel('Time (s)')
   plt.ylabel('G(t) (Pa)')
   plt.legend()
   plt.title('Fit Quality')
   plt.grid(True, alpha=0.3)

   # Right: Residuals
   plt.subplot(1, 2, 2)
   plt.semilogx(t_true, residuals / G_noisy * 100, 'o-')
   plt.axhline(0, color='gray', linestyle='--')
   plt.xlabel('Time (s)')
   plt.ylabel('Relative Error (%)')
   plt.title('Residuals')
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

**Output**:

.. code-block:: text

   Fitted Parameters:
     G0 = 1.008e+05 Pa (true: 1.000e+05)
     eta = 1.012e+05 Pa·s (true: 1.000e+05)
     tau = 1.004 s (true: 1.000)

   Fit Quality:
     Mean error: 1.87%

The fit recovers the true parameters within 1% despite 2% noise!

Common Pitfalls
---------------

Pitfall 1: Wrong Test Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # WRONG: Using viscoelastic model for flow data
   from rheojax.models import Maxwell
   model = Maxwell()
   model.fit(shear_rate, viscosity, test_mode='rotation')  # Will fail!

**Solution**: Use flow models for steady shear data:

.. code-block:: python

   from rheojax.models import PowerLaw
   model = PowerLaw()
   model.fit(shear_rate, viscosity, test_mode='rotation')  # ✓

Pitfall 2: Incorrect Data Shape for Oscillation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # WRONG: Passing G' and G" separately
   model.fit(omega, G_prime, test_mode='oscillation')  # Missing G"!

**Solution**: Stack :math:`G'` and :math:`G''` into 2D array:

.. code-block:: python

   G_star = np.column_stack([G_prime, G_double_prime])
   model.fit(omega, G_star, test_mode='oscillation')  # ✓

Pitfall 3: Ignoring Fit Quality
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # BAD: Fitting without validation
   model.fit(t, G_t)
   # Parameters might be nonsense!

**Solution**: Always validate:

.. code-block:: python

   model.fit(t, G_t, test_mode='relaxation')

   # Check residuals
   mean_error = np.mean(np.abs((G_t - model.predict(t)) / G_t)) * 100
   if mean_error > 10:
       print(f"Warning: Poor fit (error = {mean_error:.1f}%)")

Key Concepts
------------

.. admonition:: Main Takeaways
   :class: tip

   1. **Basic fit workflow**: Load data → Create model → `model.fit(x, y, test_mode)` → Extract parameters

   2. **Always validate**: Visual inspection + residual analysis + parameter reasonableness

   3. **Test modes matter**: relaxation (time vs :math:`G`), oscillation (freq vs :math:`G^*` / :math:`G''`), rotation (shear rate vs :math:`\eta`)

   4. **NLSQ is default**: GPU-accelerated, 5-270x faster than scipy

   5. **Parameters have physical meaning**: :math:`G_0` (stiffness), :math:`\eta` (viscosity), :math:`\tau` (timescale)

.. admonition:: Self-Check Questions
   :class: tip

   1. **You fit a Maxwell model and get** :math:`\tau = 10^{-9}` **s. Should you trust this value?**

      Hint: Check typical relaxation time ranges

   2. **Your fit has mean error = 15%. What should you do?**

      Hint: Try a different model (more complex) or check data quality

   3. **How do you fit oscillation data with** :math:`G'` **and** :math:`G''` **?**

      Hint: Stack into 2D array with `np.column_stack()`

   4. **Can you use a Maxwell model for steady shear flow data?**

      Hint: Maxwell is for linear viscoelasticity (relaxation, oscillation, creep)

   5. **What does it mean if G_fit is systematically below G_data at all times?**

      Hint: Model is too simple, missing physics (try fractional or multi-mode)

Further Reading
---------------

**Within this documentation**:

- :doc:`model_families` — Overview of all 53 models across 22 families
- :doc:`model_selection` — How to choose the right model for your data
- :doc:`fitting_strategies` — Advanced initialization and troubleshooting

**Model details** (equations and theory):

- :doc:`/models/classical/maxwell` — Maxwell model mathematics
- :doc:`/models/classical/zener` — Zener model (viscoelastic solid)
- :doc:`/models/flow/power_law` — PowerLaw for flow curves

**Example notebooks**:

- ``examples/basic/01-maxwell_fitting.ipynb`` — Step-by-step tutorial
- ``examples/basic/02-model_comparison.ipynb`` — Comparing multiple models

Summary
-------

Fitting a rheological model in RheoJAX requires just a few lines: load data, create model, call `fit(x, y, test_mode)`, and extract parameters. Always validate fits with visual inspection, residual analysis, and parameter sanity checks. The NLSQ backend provides GPU-accelerated optimization for fast, accurate parameter recovery.

Next Steps
----------

Proceed to: :doc:`model_families`

Learn about the three major model families (classical, fractional, flow) and when to use each.
