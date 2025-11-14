Multi-Technique Fitting Guide
=============================

Multi-technique fitting involves simultaneously fitting the same model to multiple datasets from different experimental techniques (e.g., relaxation and oscillation), ensuring physical consistency across test modes. This guide demonstrates how to leverage :class:`SharedParameterSet` for coupled parameter optimization.

Why Multi-Technique Fitting?
----------------------------

Benefits
~~~~~~~~

**Physical Consistency**: Ensure parameters have the same physical meaning across different test modes

**Extended Range**: Combine data from multiple techniques to cover wider time/frequency ranges

**Improved Accuracy**: More data points constrain parameters better than single-technique fitting

**Validation**: Cross-validate model predictions across different experimental conditions

**Reduced Uncertainty**: Shared parameters have tighter confidence intervals

Common Combinations
~~~~~~~~~~~~~~~~~~~

.. list-table:: Typical Multi-Technique Combinations
   :header-rows: 1
   :widths: 30 30 40

   * - Technique 1
     - Technique 2
     - Why Combine
   * - **Stress Relaxation**
     - **Oscillation (SAOS)**
     - Time-domain and frequency-domain views of same material
   * - **Creep**
     - **Oscillation (SAOS)**
     - Compliance and modulus are related
   * - **Oscillation (low freq)**
     - **Oscillation (high freq)**
     - Extend frequency range beyond instrument limits
   * - **Oscillation (SAOS)**
     - **Steady Shear**
     - Link linear and nonlinear flow behavior
   * - **Multi-temperature oscillation**
     - **Single-temp relaxation**
     - Validate temperature-dependent parameters

SharedParameterSet Basics
-------------------------

Core Concept
~~~~~~~~~~~~

The :class:`SharedParameterSet` class enables multiple models to share parameters while maintaining independent optimization:

.. code-block:: python

   from rheojax.core.parameters import SharedParameterSet

   # Create shared parameter set
   shared = SharedParameterSet()

   # Add shared parameters
   shared.add_shared('G_s', value=1e5, bounds=(1e3, 1e7), units='Pa')
   shared.add_shared('eta_s', value=1e3, bounds=(1e1, 1e5), units='Pa*s')

   # Link models
   from rheojax.models import Maxwell

   model_relaxation = Maxwell()
   model_oscillation = Maxwell()

   shared.link_model(model_relaxation, 'G_s')
   shared.link_model(model_relaxation, 'eta_s')
   shared.link_model(model_oscillation, 'G_s')
   shared.link_model(model_oscillation, 'eta_s')

   # Now updating shared parameters updates both models
   shared.set_value('G_s', 5e5)
   # Both models now have G_s = 5e5

Key Features
~~~~~~~~~~~~

1. **Automatic synchronization**: Changing shared parameter updates all linked models
2. **Bounds enforcement**: Shared bounds apply to all models
3. **Validation**: Ensures parameter consistency across models
4. **Optimization support**: Compatible with JAX-based optimizers

Example: Relaxation + Oscillation
---------------------------------

Scenario
~~~~~~~~

You have:

1. Stress relaxation data: sigma(t) at fixed strain
2. Frequency sweep data: G'(omega), G''(omega) from SAOS

You want to fit a Maxwell model to both datasets with consistent G_s and eta_s.

Step-by-Step Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Load both datasets**

.. code-block:: python

   from rheojax.io import auto_load

   # Load relaxation data
   data_relax = auto_load('relaxation.txt')
   # data_relax.x = time (s), data_relax.y = stress (Pa)

   # Load oscillation data
   data_osc = auto_load('oscillation.txt')
   # data_osc.x = frequency (rad/s), data_osc.y = complex modulus G* (Pa)

**Step 2: Create shared parameters**

.. code-block:: python

   from rheojax.core.parameters import SharedParameterSet
   import numpy as np

   # Estimate initial values from data
   G_s_init = np.median(np.abs(data_osc.y))  # Typical modulus
   eta_s_init = G_s_init * 1.0  # Guess: tau ~ 1 second

   # Create shared parameter set
   shared = SharedParameterSet()
   shared.add_shared('G_s', value=G_s_init,
                     bounds=(1e3, 1e7), units='Pa',
                     description='Shear modulus')
   shared.add_shared('eta_s', value=eta_s_init,
                     bounds=(1e1, 1e5), units='Pa*s',
                     description='Viscosity')

**Step 3: Create and link models**

.. code-block:: python

   from rheojax.models import Maxwell

   # Create two model instances
   maxwell_relax = Maxwell()
   maxwell_osc = Maxwell()

   # Link shared parameters
   shared.link_model(maxwell_relax, 'G_s')
   shared.link_model(maxwell_relax, 'eta_s')
   shared.link_model(maxwell_osc, 'G_s')
   shared.link_model(maxwell_osc, 'eta_s')

**Step 4: Define combined objective function**

.. code-block:: python

   import jax.numpy as jnp
   import jax

   @jax.jit
   def combined_objective(params_array):
       """Combined RSS from both datasets."""
       G_s, eta_s = params_array

       # Update shared parameters
       shared.set_values({'G_s': G_s, 'eta_s': eta_s})

       # Predictions for relaxation
       t_relax = data_relax.x
       tau = eta_s / G_s
       sigma_pred_relax = G_s * data_relax.metadata['gamma_0'] * jnp.exp(-t_relax / tau)
       residuals_relax = data_relax.y - sigma_pred_relax

       # Predictions for oscillation
       omega_osc = data_osc.x
       G_star_pred = G_s / (1 + 1j * omega_osc * tau)
       G_star_abs_pred = jnp.abs(G_star_pred)
       residuals_osc = jnp.abs(data_osc.y) - G_star_abs_pred

       # Combined RSS (optionally weight datasets)
       rss_relax = jnp.sum(residuals_relax**2)
       rss_osc = jnp.sum(residuals_osc**2)

       # Normalize by number of points to balance datasets
       n_relax = len(t_relax)
       n_osc = len(omega_osc)

       total_rss = rss_relax / n_relax + rss_osc / n_osc

       return total_rss

**Step 5: Optimize shared parameters**

.. code-block:: python

   from rheojax.utils.optimization import nlsq_optimize

   # Get initial parameters
   p0 = jnp.array([shared.get_value('G_s'), shared.get_value('eta_s')])

   # Optimize
   result = nlsq_optimize(
       combined_objective,
       shared,
       use_jax=True,
       method='L-BFGS-B'
   )

   # Extract optimized parameters
   G_s_opt, eta_s_opt = result.x
   shared.set_values({'G_s': G_s_opt, 'eta_s': eta_s_opt})

   print(f"Optimized G_s = {G_s_opt:.2e} Pa")
   print(f"Optimized eta_s = {eta_s_opt:.2e} Pa*s")
   print(f"Relaxation time tau = {eta_s_opt/G_s_opt:.3f} s")

**Step 6: Validate and visualize**

.. code-block:: python

   import matplotlib.pyplot as plt

   # Make predictions with optimized parameters
   sigma_pred_relax = maxwell_relax.predict(data_relax.x)
   G_star_pred_osc = maxwell_osc.predict(data_osc.x)

   # Calculate R^2 for each dataset
   from sklearn.metrics import r2_score
   r2_relax = r2_score(data_relax.y, sigma_pred_relax)
   r2_osc = r2_score(np.abs(data_osc.y), np.abs(G_star_pred_osc))

   print(f"R^2 (relaxation): {r2_relax:.4f}")
   print(f"R^2 (oscillation): {r2_osc:.4f}")

   # Plot both datasets
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   # Relaxation
   ax1.semilogy(data_relax.x, data_relax.y, 'o', label='Data')
   ax1.semilogy(data_relax.x, sigma_pred_relax, '-',
                linewidth=2, label='Multi-technique fit')
   ax1.set_xlabel('Time (s)')
   ax1.set_ylabel('Stress (Pa)')
   ax1.set_title(f'Stress Relaxation (R^2 = {r2_relax:.4f})')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Oscillation
   ax2.loglog(data_osc.x, np.abs(data_osc.y), 'o', label='Data')
   ax2.loglog(data_osc.x, np.abs(G_star_pred_osc), '-',
              linewidth=2, label='Multi-technique fit')
   ax2.set_xlabel('Frequency (rad/s)')
   ax2.set_ylabel('|G*| (Pa)')
   ax2.set_title(f'Oscillation (R^2 = {r2_osc:.4f})')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig('multi_technique_fit.png', dpi=300)
   plt.show()

Advanced Techniques
-------------------

Weighted Multi-Technique Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance datasets with different sizes or quality:

.. code-block:: python

   @jax.jit
   def weighted_objective(params_array, weight_relax=1.0, weight_osc=1.0):
       """Weighted combined RSS."""
       G_s, eta_s = params_array
       shared.set_values({'G_s': G_s, 'eta_s': eta_s})

       # Calculate residuals
       residuals_relax = data_relax.y - maxwell_relax.predict(data_relax.x)
       residuals_osc = data_osc.y - maxwell_osc.predict(data_osc.x)

       # Weighted RSS
       rss_relax = jnp.sum(residuals_relax**2) * weight_relax
       rss_osc = jnp.sum(residuals_osc**2) * weight_osc

       return rss_relax + rss_osc

   # Emphasize relaxation data (more reliable)
   result = nlsq_optimize(
       lambda p: weighted_objective(p, weight_relax=2.0, weight_osc=1.0),
       shared,
       use_jax=True
   )

Selective Parameter Sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Share some parameters while keeping others independent:

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel

   # Create models
   fmg_relax = FractionalMaxwellGel()
   fmg_osc = FractionalMaxwellGel()

   # Share G_s and alpha, but keep V independent
   shared = SharedParameterSet()
   shared.add_shared('G_s', value=1e5, bounds=(1e3, 1e7))
   shared.add_shared('alpha', value=0.5, bounds=(0.1, 0.9))

   shared.link_model(fmg_relax, 'G_s')
   shared.link_model(fmg_relax, 'alpha')
   shared.link_model(fmg_osc, 'G_s')
   shared.link_model(fmg_osc, 'alpha')

   # V remains independent for each model
   fmg_relax.parameters.set_value('V', 1e3)
   fmg_osc.parameters.set_value('V', 5e3)

   # Optimization only affects G_s and alpha

Sequential Parameter Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optimize in stages for better convergence:

.. code-block:: python

   # Stage 1: Fit relaxation data alone to get initial guess
   maxwell_relax.fit(data_relax.x, data_relax.y)

   G_s_stage1 = maxwell_relax.parameters.get_value('G_s')
   eta_s_stage1 = maxwell_relax.parameters.get_value('eta_s')

   # Stage 2: Use stage 1 values as initial guess for multi-technique
   shared.set_values({'G_s': G_s_stage1, 'eta_s': eta_s_stage1})

   # Stage 3: Multi-technique optimization
   result = nlsq_optimize(combined_objective, shared, use_jax=True)

Cross-Validation Across Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validate that model trained on one technique predicts another:

.. code-block:: python

   # Train on relaxation only
   maxwell_relax_only = Maxwell()
   maxwell_relax_only.fit(data_relax.x, data_relax.y)

   # Predict oscillation with relaxation-trained parameters
   maxwell_osc_from_relax = Maxwell()
   maxwell_osc_from_relax.parameters.set_values(
       maxwell_relax_only.parameters.to_dict()
   )

   G_star_pred_cross = maxwell_osc_from_relax.predict(data_osc.x)
   r2_cross = r2_score(np.abs(data_osc.y), np.abs(G_star_pred_cross))

   print(f"Cross-validation R^2 (relax -> osc): {r2_cross:.4f}")

   # Compare to multi-technique fit
   print(f"Multi-technique R^2 (osc): {r2_osc:.4f}")
   print(f"Improvement: {(r2_osc - r2_cross)*100:.1f} percentage points")

Example: Fractional Model Multi-Technique
-----------------------------------------

Scenario
~~~~~~~~

Fit :class:`FractionalMaxwellGel` to both relaxation and oscillation data.

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   from rheojax.core.parameters import SharedParameterSet
   import jax.numpy as jnp

   # Load data
   data_relax = auto_load('relaxation.txt')
   data_osc = auto_load('oscillation.txt')

   # Create shared parameters
   shared = SharedParameterSet()
   shared.add_shared('G_s', value=1e5, bounds=(1e3, 1e7), units='Pa')
   shared.add_shared('V', value=1e3, bounds=(1e2, 1e5), units='Pa*s^alpha')
   shared.add_shared('alpha', value=0.5, bounds=(0.1, 0.9), units='-')

   # Create models
   fmg_relax = FractionalMaxwellGel()
   fmg_osc = FractionalMaxwellGel()

   # Link all parameters
   for param_name in ['G_s', 'V', 'alpha']:
       shared.link_model(fmg_relax, param_name)
       shared.link_model(fmg_osc, param_name)

   # Combined objective
   @jax.jit
   def fmg_objective(params_array):
       G_s, V, alpha = params_array
       shared.set_values({'G_s': G_s, 'V': V, 'alpha': alpha})

       # Predictions
       sigma_pred_relax = fmg_relax.predict(data_relax.x)
       G_star_pred_osc = fmg_osc.predict(data_osc.x)

       # Combined residuals
       res_relax = data_relax.y - sigma_pred_relax
       res_osc = jnp.abs(data_osc.y) - jnp.abs(G_star_pred_osc)

       # Normalized RSS
       rss = (jnp.sum(res_relax**2) / len(res_relax) +
              jnp.sum(res_osc**2) / len(res_osc))

       return rss

   # Optimize
   from rheojax.utils.optimization import nlsq_optimize

   p0 = jnp.array([
       shared.get_value('G_s'),
       shared.get_value('V'),
       shared.get_value('alpha')
   ])

   result = nlsq_optimize(fmg_objective, shared, use_jax=True)

   # Extract results
   G_s_opt, V_opt, alpha_opt = result.x
   shared.set_values({'G_s': G_s_opt, 'V': V_opt, 'alpha': alpha_opt})

   print(f"Optimized parameters:")
   print(f"  G_s = {G_s_opt:.2e} Pa")
   print(f"  V = {V_opt:.2e} Pa*s^{alpha_opt:.2f}")
   print(f"  alpha = {alpha_opt:.3f}")

   # Validate
   r2_relax = fmg_relax.score(data_relax.x, data_relax.y)
   r2_osc = fmg_osc.score(data_osc.x, data_osc.y)

   print(f"\nFit quality:")
   print(f"  R^2 (relaxation): {r2_relax:.4f}")
   print(f"  R^2 (oscillation): {r2_osc:.4f}")

Example: Multi-Temperature Fitting
----------------------------------

Scenario
~~~~~~~~

You have oscillation data at multiple temperatures. Fit model with temperature-independent parameters (e.g., alpha) shared across all temperatures.

Implementation
~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   from rheojax.core.parameters import SharedParameterSet
   import jax.numpy as jnp

   # Load multi-temperature data
   temperatures = [25, 40, 55, 70]  #  degC
   datasets = [auto_load(f'freq_sweep_{T}C.txt') for T in temperatures]

   # Shared parameter: alpha (temperature-independent for many polymers)
   shared = SharedParameterSet()
   shared.add_shared('alpha', value=0.5, bounds=(0.1, 0.9))

   # Create one model per temperature
   models = []
   for T in temperatures:
       model = FractionalMaxwellGel()
       shared.link_model(model, 'alpha')
       models.append(model)

       # Initialize temperature-dependent parameters (G_s, V)
       # These remain independent for each temperature
       model.parameters.set_value('G_s', 1e5)
       model.parameters.set_value('V', 1e3)

   # Combined objective (all temperatures)
   @jax.jit
   def multi_temp_objective(alpha_value):
       """Optimize shared alpha across all temperatures."""
       shared.set_value('alpha', alpha_value)

       total_rss = 0.0
       for model, dataset in zip(models, datasets):
           # Fit temperature-dependent params with current alpha
           # (in practice, you'd integrate this into optimization)
           pred = model.predict(dataset.x)
           residuals = jnp.abs(dataset.y) - jnp.abs(pred)
           rss = jnp.sum(residuals**2) / len(residuals)
           total_rss += rss

       return total_rss

   # Optimize shared alpha
   from scipy.optimize import minimize_scalar

   result = minimize_scalar(
       lambda a: float(multi_temp_objective(a)),
       bounds=(0.1, 0.9),
       method='bounded'
   )

   alpha_opt = result.x
   shared.set_value('alpha', alpha_opt)

   print(f"Optimized alpha (shared across all T): {alpha_opt:.3f}")

   # Now fit temperature-dependent parameters for each T
   for T, model, dataset in zip(temperatures, models, datasets):
       # Alpha is already set via shared parameters
       model.fit(dataset.x, dataset.y)

       G_s_T = model.parameters.get_value('G_s')
       V_T = model.parameters.get_value('V')

       print(f"T = {T} degC: G_s = {G_s_T:.2e} Pa, V = {V_T:.2e} Pa*s^alpha")

Best Practices
--------------

Parameter Selection for Sharing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Good candidates for sharing**:

- Fractional orders (alpha) - often material-intrinsic
- Elastic moduli (G_s, G_p) - if temperature constant
- Structural parameters (network parameters)

**Poor candidates for sharing**:

- Viscosity parameters (temperature-dependent)
- Flow parameters in yield stress fluids (rate-dependent)
- Time constants (frequency-dependent in some cases)

Physical Validation
~~~~~~~~~~~~~~~~~~~

Always check that shared parameters make physical sense:

.. code-block:: python

   # After optimization, check parameter values
   G_s = shared.get_value('G_s')
   eta_s = shared.get_value('eta_s')
   tau = eta_s / G_s

   print(f"Relaxation time: {tau:.3f} s")

   # Compare to characteristic times in data
   t_relax_char = data_relax.x[-1]  # Last time point
   omega_osc_char = data_osc.x[len(data_osc.x)//2]  # Mid-frequency

   print(f"Relaxation data range: {data_relax.x[0]:.2e} - {t_relax_char:.2e} s")
   print(f"Oscillation frequency range: {data_osc.x[0]:.2e} - {data_osc.x[-1]:.2e} rad/s")

   # Tau should be within or near data ranges
   if not (data_relax.x[0] < tau < t_relax_char * 10):
       print("Warning: tau outside relaxation data range!")

Convergence Monitoring
~~~~~~~~~~~~~~~~~~~~~~

Monitor optimization convergence:

.. code-block:: python

   from rheojax.utils.optimization import nlsq_optimize

   # Store optimization history
   history = {'iteration': [], 'rss': [], 'params': []}

   def callback(params_array):
       """Callback to track optimization progress."""
       rss = combined_objective(params_array)
       history['iteration'].append(len(history['iteration']))
       history['rss'].append(float(rss))
       history['params'].append(params_array.copy())

   # Optimize with callback
   result = nlsq_optimize(
       combined_objective,
       shared,
       use_jax=True,
       callback=callback,
       max_iterations=1000
   )

   # Plot convergence
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.semilogy(history['iteration'], history['rss'], 'o-')
   plt.xlabel('Iteration')
   plt.ylabel('Combined RSS')
   plt.title('Multi-Technique Optimization Convergence')
   plt.grid(True, alpha=0.3)
   plt.show()

Error Handling
~~~~~~~~~~~~~~

Handle cases where datasets may be incompatible:

.. code-block:: python

   try:
       result = nlsq_optimize(combined_objective, shared, use_jax=True)

       # Check if optimization succeeded
       if result.success:
           print("Optimization converged successfully")
       else:
           print(f"Optimization failed: {result.message}")

       # Validate individual fits
       r2_relax = maxwell_relax.score(data_relax.x, data_relax.y)
       r2_osc = maxwell_osc.score(data_osc.x, data_osc.y)

       if r2_relax < 0.8 or r2_osc < 0.8:
           print("Warning: Poor fit quality!")
           print(f"  R^2 (relaxation): {r2_relax:.4f}")
           print(f"  R^2 (oscillation): {r2_osc:.4f}")

   except Exception as e:
       print(f"Multi-technique fitting failed: {e}")
       print("Falling back to individual fits...")

       # Fit each dataset independently
       maxwell_relax.fit(data_relax.x, data_relax.y)
       maxwell_osc.fit(data_osc.x, data_osc.y)

Common Pitfalls
---------------

Pitfall 1: Incompatible Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using different models for different techniques:

.. code-block:: python

   # Don't do this (models are different)
   maxwell_relax = Maxwell()
   zener_osc = Zener()

   shared.link_model(maxwell_relax, 'G_s')
   shared.link_model(zener_osc, 'G_s')  # Zener has G_s but different meaning

   # Problem: G_s in Maxwell vs Zener has different physical interpretation

Solution: Use same model class for all techniques:

.. code-block:: python

   # Good: same model for both
   maxwell_relax = Maxwell()
   maxwell_osc = Maxwell()

Pitfall 2: Over-Constraining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sharing too many parameters can lead to poor fits:

.. code-block:: python

   # Bad: share everything (too constrained)
   for param_name in model_relax.parameters.parameter_names:
       shared.add_shared(param_name, value=...)
       shared.link_model(model_relax, param_name)
       shared.link_model(model_osc, param_name)

   # May not fit well if techniques probe different regimes

Solution: Share only physically meaningful common parameters.

Pitfall 3: Unbalanced Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One dataset dominates optimization:

.. code-block:: python

   # Problem: oscillation has 100 points, relaxation has 10
   # RSS from oscillation will dominate

   # Solution: normalize by number of points
   rss_relax_norm = jnp.sum(residuals_relax**2) / len(data_relax.x)
   rss_osc_norm = jnp.sum(residuals_osc**2) / len(data_osc.x)
   total_rss = rss_relax_norm + rss_osc_norm

Pitfall 4: Poor Initial Guess
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-technique optimization is more sensitive to initialization:

.. code-block:: python

   # Bad: arbitrary initial guess
   shared.set_values({'G_s': 1.0, 'eta_s': 1.0})  # May fail to converge

   # Good: data-driven initialization
   import numpy as np

   G_s_init = np.median(np.abs(data_osc.y))
   tau_init = 1.0 / np.median(data_osc.x)
   eta_s_init = G_s_init * tau_init

   shared.set_values({'G_s': G_s_init, 'eta_s': eta_s_init})

Summary
-------

Multi-Technique Fitting Checklist:

1. **Create SharedParameterSet** and add parameters
2. **Link models** to shared parameters
3. **Define combined objective** summing RSS from all techniques
4. **Normalize** residuals by dataset size
5. **Use data-driven initialization** for shared parameters
6. **Optimize** with JAX gradients for speed
7. **Validate** fit quality for each technique individually
8. **Visualize** predictions vs data for all techniques
9. **Check physical consistency** of parameters

Benefits:

- Extended range (combine different techniques)
- Improved parameter accuracy (more data)
- Physical consistency (same parameters everywhere)
- Cross-validation (predict one technique from another)

Next Steps
----------

- :doc:`/user_guide/model_selection` - Choose appropriate model
- :doc:`/user_guide/modular_api` - Advanced parameter manipulation
- :doc:`/api/core` - SharedParameterSet API reference
- ``examples/advanced/multi_technique_fitting.ipynb`` - Complete example notebook
