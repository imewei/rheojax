Modular API Tutorial
====================

The Modular API provides direct access to models and transforms for maximum flexibility and control. Use this API when you need fine-grained parameter manipulation, custom optimization workflows, or complex analysis pipelines.

When to Use the Modular API
----------------------------

**Use the Modular API for:**

- Custom parameter initialization and bounds
- Non-standard optimization algorithms
- Complex parameter constraints
- Direct manipulation of model equations
- Integration with external libraries
- Research and algorithm development
- Teaching model fundamentals

**Use the Pipeline API for:**

- Standard workflows and rapid prototyping
- Batch processing
- Quick exploratory analysis
- Production code with error handling

The Modular API gives you complete control at the cost of more verbose code.

Core Components
---------------

ModelRegistry
~~~~~~~~~~~~~

The ModelRegistry provides centralized model management:

.. code-block:: python

   from rheojax.core.registry import ModelRegistry

   # List all available models
   available_models = ModelRegistry.list_models()
   print(f"Available models: {available_models}")

   # Get model information
   info = ModelRegistry.get_info('maxwell')
   print(f"Description: {info.description}")
   print(f"Parameters: {info.metadata.get('parameters')}")

   # Create model instance
   model = ModelRegistry.create('maxwell')

   # Alternative: direct import
   from rheojax.models import Maxwell
   model = Maxwell()

TransformRegistry
~~~~~~~~~~~~~~~~~

Similarly for transforms:

.. code-block:: python

   from rheojax.core.registry import TransformRegistry

   # List transforms
   transforms = TransformRegistry.list_transforms()

   # Create transform
   fft = TransformRegistry.create('fft_analysis')

   # Alternative: direct import
   from rheojax.transforms import FFTAnalysis
   fft = FFTAnalysis()

Working with Models
-------------------

Direct Model Instantiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and configure models directly:

.. code-block:: python

   from rheojax.models import Maxwell, Zener, FractionalMaxwellGel
   import numpy as np

   # Create model instance
   maxwell = Maxwell()

   # Inspect default parameters
   print(maxwell.parameters)
   # Output: ParameterSet with G_s and eta_s

   # Get parameter details
   G_s_param = maxwell.parameters.get_parameter('G_s')
   print(f"Name: {G_s_param.name}")
   print(f"Units: {G_s_param.units}")
   print(f"Bounds: {G_s_param.bounds}")
   print(f"Value: {G_s_param.value}")

Setting Initial Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~

Control parameter initialization:

.. code-block:: python

   from rheojax.models import Maxwell

   maxwell = Maxwell()

   # Set individual parameters
   maxwell.parameters.set_value('G_s', 1e5)      # Pa
   maxwell.parameters.set_value('eta_s', 1e3)    # Pa·s

   # Set multiple parameters
   maxwell.parameters.set_values({
       'G_s': 1e5,
       'eta_s': 1e3
   })

   # Get parameter values
   G_s = maxwell.parameters.get_value('G_s')
   eta_s = maxwell.parameters.get_value('eta_s')

   # Get all parameters as dict
   params_dict = maxwell.parameters.to_dict()
   print(params_dict)

Setting Parameter Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~

Control optimization search space:

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel

   model = FractionalMaxwellGel()

   # Set bounds for each parameter
   model.parameters.set_bounds('G_s', min_value=1e3, max_value=1e7)
   model.parameters.set_bounds('V', min_value=1e2, max_value=1e6)
   model.parameters.set_bounds('alpha', min_value=0.1, max_value=0.9)

   # Alternative: set during initialization
   model.parameters.get_parameter('G_s').bounds = (1e3, 1e7)

   # Get bounds
   bounds = model.parameters.get_bounds('alpha')
   print(f"Alpha bounds: {bounds}")

Parameter Constraints
~~~~~~~~~~~~~~~~~~~~~

Add complex constraints:

.. code-block:: python

   from rheojax.core.parameters import Parameter, ParameterSet

   params = ParameterSet()

   # Add parameters with constraints
   params.add(Parameter(
       name='G_s',
       value=1e5,
       bounds=(1e3, 1e7),
       constraints=['positive']
   ))

   # Relative constraint (e.g., G_s > G_p)
   params.add(Parameter(
       name='G_p',
       value=1e4,
       bounds=(1e2, 1e6),
       constraints=[
           'positive',
           ('relative', 'G_s', 'less_than')  # G_p < G_s
       ]
   ))

   # Validate constraints
   is_valid = params.validate()
   if not is_valid:
       violations = params.get_constraint_violations()
       print(f"Constraint violations: {violations}")

Fitting Models
--------------

Basic Fitting
~~~~~~~~~~~~~

Fit model to data:

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.io import auto_load
   import numpy as np

   # Load data
   data = auto_load('oscillation_data.txt')
   X = data.x  # Frequency (Hz or rad/s)
   y = data.y  # Complex modulus |G*|

   # Create and fit model
   maxwell = Maxwell()
   maxwell.fit(X, y)

   # Access fitted parameters
   G_s = maxwell.parameters.get_value('G_s')
   eta_s = maxwell.parameters.get_value('eta_s')
   print(f"G_s = {G_s:.2e} Pa")
   print(f"eta_s = {eta_s:.2e} Pa·s")

   # Make predictions
   y_pred = maxwell.predict(X)

   # Calculate fit quality
   r2 = maxwell.score(X, y)
   print(f"R² = {r2:.4f}")

Custom Initial Guesses
~~~~~~~~~~~~~~~~~~~~~~

Provide data-driven initialization:

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   import numpy as np

   # Analyze data to inform initial guess
   G_min = np.min(np.abs(y))
   G_max = np.max(np.abs(y))

   model = FractionalMaxwellGel()

   # Set initial guess
   model.parameters.set_values({
       'G_s': G_min * 0.8,      # Rubbery modulus ~ low-freq plateau
       'V': G_max * 2,          # Fractional viscosity ~ high-freq behavior
       'alpha': 0.5             # Mid-range fractional order
   })

   # Set bounds
   model.parameters.set_bounds('G_s', min_value=G_min*0.1, max_value=G_max*2)
   model.parameters.set_bounds('V', min_value=G_min*0.1, max_value=G_max*10)
   model.parameters.set_bounds('alpha', min_value=0.1, max_value=0.9)

   # Fit with custom initialization
   model.fit(X, y)

Multi-Start Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~

Try multiple initial guesses to avoid local minima:

.. code-block:: python

   from rheojax.models import Zener
   import numpy as np

   # Generate multiple initial guesses
   n_starts = 5
   best_score = -np.inf
   best_model = None

   for i in range(n_starts):
       model = Zener()

       # Random initialization within bounds
       G_s_init = np.random.uniform(1e3, 1e6)
       G_p_init = np.random.uniform(1e2, 1e5)
       eta_p_init = np.random.uniform(1e1, 1e4)

       model.parameters.set_values({
           'G_s': G_s_init,
           'G_p': G_p_init,
           'eta_p': eta_p_init
       })

       # Fit
       model.fit(X, y)

       # Check score
       score = model.score(X, y)
       if score > best_score:
           best_score = score
           best_model = model

   print(f"Best R² = {best_score:.4f}")
   print(f"Best parameters: {best_model.parameters.to_dict()}")

Custom Optimization
~~~~~~~~~~~~~~~~~~~

Use custom optimization algorithms:

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.utils.optimization import nlsq_optimize
   import jax.numpy as jnp
   import jax

   # Create model
   maxwell = Maxwell()

   # Define custom objective function
   @jax.jit
   def objective(params_array):
       """Custom objective with weights or constraints."""
       G_s, eta_s = params_array

       # Predictions
       omega = X
       tau = eta_s / G_s
       G_star = G_s / (1 + 1j * omega * tau)
       y_pred = jnp.abs(G_star)

       # Weighted residuals (e.g., emphasize low frequency)
       weights = 1.0 / (1.0 + omega)  # Higher weight at low freq
       residuals = (y - y_pred) * weights

       return jnp.sum(residuals**2)

   # Get initial parameters
   p0 = jnp.array([
       maxwell.parameters.get_value('G_s'),
       maxwell.parameters.get_value('eta_s')
   ])

   # Optimize
   result = nlsq_optimize(objective, maxwell.parameters, use_jax=True)

   # Update model with optimized parameters
   maxwell.parameters.set_values({
       'G_s': result.x[0],
       'eta_s': result.x[1]
   })

Working with Transforms
-----------------------

Direct Transform Usage
~~~~~~~~~~~~~~~~~~~~~~

Apply transforms directly to RheoData:

.. code-block:: python

   from rheojax.transforms import FFTAnalysis, SmoothDerivative
   from rheojax.core import RheoData
   from rheojax.io import auto_load

   # Load time-series data
   data = auto_load('time_series.txt')

   # Apply smoothing
   smoother = SmoothDerivative(method='savgol', window=11, order=2)
   data_smooth = smoother.transform(data)

   # Apply FFT
   fft = FFTAnalysis(window='hann', detrend=True)
   freq_data = fft.transform(data_smooth)

   # Access results
   G_prime = freq_data.metadata['G_prime']
   G_double_prime = freq_data.metadata['G_double_prime']

Transform Composition
~~~~~~~~~~~~~~~~~~~~~

Chain transforms manually:

.. code-block:: python

   from rheojax.transforms import SmoothDerivative, FFTAnalysis
   from rheojax.core.base import TransformPipeline

   # Create pipeline
   pipeline = TransformPipeline([
       SmoothDerivative(method='savgol', window=11, order=2),
       FFTAnalysis(window='hann', detrend=True)
   ])

   # Apply pipeline
   result = pipeline.transform(data)

   # Alternative: operator overloading
   pipeline = SmoothDerivative(method='savgol', window=11, order=2) + \
              FFTAnalysis(window='hann', detrend=True)

   result = pipeline.transform(data)

Inverse Transforms
~~~~~~~~~~~~~~~~~~

Some transforms are invertible:

.. code-block:: python

   from rheojax.transforms import FFTAnalysis

   fft = FFTAnalysis()

   # Forward transform
   freq_data = fft.transform(time_data)

   # Inverse transform
   time_data_reconstructed = fft.inverse_transform(freq_data)

   # Check reconstruction error
   import numpy as np
   error = np.mean(np.abs(time_data.y - time_data_reconstructed.y))
   print(f"Reconstruction error: {error:.2e}")

Custom Fitting Workflows
-------------------------

Sequential Parameter Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit parameters in stages for better convergence:

.. code-block:: python

   from rheojax.models import FractionalMaxwellModel
   import numpy as np

   model = FractionalMaxwellModel()

   # Stage 1: Fix alpha, fit G_s and V
   model.parameters.get_parameter('alpha').fixed = True
   model.parameters.set_value('alpha', 0.5)

   model.fit(X, y)

   # Stage 2: Fix G_s and V, optimize alpha
   model.parameters.get_parameter('G_s').fixed = True
   model.parameters.get_parameter('V').fixed = True
   model.parameters.get_parameter('alpha').fixed = False

   model.fit(X, y)

   # Stage 3: Optimize all together
   for param in model.parameters.parameters.values():
       param.fixed = False

   model.fit(X, y)

   print("Final parameters:")
   print(model.parameters.to_dict())

Fitting with Analytical Gradients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Leverage JAX automatic differentiation:

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.utils.optimization import nlsq_optimize
   import jax
   import jax.numpy as jnp

   maxwell = Maxwell()

   # Define objective with automatic gradient
   @jax.jit
   def objective(params_array):
       G_s, eta_s = params_array
       tau = eta_s / G_s
       G_star = G_s / (1 + 1j * X * tau)
       y_pred = jnp.abs(G_star)
       return jnp.sum((y - y_pred)**2)

   # Compute gradient automatically
   grad_fn = jax.grad(objective)

   # Check gradient
   p0 = jnp.array([1e5, 1e3])
   gradient = grad_fn(p0)
   print(f"Gradient at p0: {gradient}")

   # Optimize using gradient
   result = nlsq_optimize(objective, maxwell.parameters,
                           use_jax=True, method='L-BFGS-B')

Cross-Validation
~~~~~~~~~~~~~~~~

Assess model generalization:

.. code-block:: python

   from rheojax.models import Maxwell, Zener
   import numpy as np
   from sklearn.model_selection import KFold

   # K-fold cross-validation
   kf = KFold(n_splits=5, shuffle=True, random_state=42)

   models = [Maxwell(), Zener()]
   cv_scores = {type(m).__name__: [] for m in models}

   for model in models:
       model_name = type(model).__name__

       for train_idx, test_idx in kf.split(X):
           X_train, X_test = X[train_idx], X[test_idx]
           y_train, y_test = y[train_idx], y[test_idx]

           # Fit on training
           model.fit(X_train, y_train)

           # Score on test
           score = model.score(X_test, y_test)
           cv_scores[model_name].append(score)

   # Report cross-validation scores
   print("Cross-Validation R² Scores:")
   for model_name, scores in cv_scores.items():
       mean_score = np.mean(scores)
       std_score = np.std(scores)
       print(f"  {model_name}: {mean_score:.4f} ± {std_score:.4f}")

Model Comparison
~~~~~~~~~~~~~~~~

Systematically compare models:

.. code-block:: python

   from rheojax.models import (Maxwell, Zener, SpringPot,
                            FractionalMaxwellGel, FractionalKelvinVoigt)
   import numpy as np
   import pandas as pd

   # Models to compare
   models = [
       Maxwell(),
       Zener(),
       SpringPot(),
       FractionalMaxwellGel(),
       FractionalKelvinVoigt()
   ]

   # Fit all models and collect metrics
   results = []

   for model in models:
       model_name = type(model).__name__

       # Fit
       model.fit(X, y)

       # Metrics
       y_pred = model.predict(X)
       residuals = y - y_pred
       r2 = model.score(X, y)
       rmse = np.sqrt(np.mean(residuals**2))
       n_params = len(model.parameters)

       # Information criteria
       n = len(y)
       rss = np.sum(residuals**2)
       aic = n * np.log(rss/n) + 2 * n_params
       bic = n * np.log(rss/n) + n_params * np.log(n)

       results.append({
           'Model': model_name,
           'N_params': n_params,
           'R²': r2,
           'RMSE': rmse,
           'AIC': aic,
           'BIC': bic
       })

   # Create comparison table
   df = pd.DataFrame(results)
   df = df.sort_values('AIC')  # Sort by AIC (lower is better)

   print("\nModel Comparison:")
   print(df.to_string(index=False))

   # Best model by AIC
   best_model_name = df.iloc[0]['Model']
   print(f"\nBest model (AIC): {best_model_name}")

Advanced Parameter Management
------------------------------

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyze how sensitive predictions are to parameters:

.. code-block:: python

   from rheojax.models import Maxwell
   import numpy as np
   import matplotlib.pyplot as plt

   maxwell = Maxwell()
   maxwell.fit(X, y)

   # Baseline parameters
   G_s_base = maxwell.parameters.get_value('G_s')
   eta_s_base = maxwell.parameters.get_value('eta_s')

   # Vary G_s
   G_s_range = np.linspace(G_s_base*0.5, G_s_base*1.5, 10)
   predictions = []

   for G_s_test in G_s_range:
       maxwell.parameters.set_value('G_s', G_s_test)
       y_pred = maxwell.predict(X)
       predictions.append(y_pred)

   # Plot sensitivity
   fig, ax = plt.subplots(figsize=(10, 6))
   for i, G_s_test in enumerate(G_s_range):
       alpha = 0.3 + 0.7 * (i / len(G_s_range))
       ax.loglog(X, predictions[i], alpha=alpha,
                 label=f'G_s = {G_s_test:.2e}')

   ax.loglog(X, y, 'ko', markersize=8, label='Data')
   ax.set_xlabel('Frequency (rad/s)')
   ax.set_ylabel('|G*| (Pa)')
   ax.legend()
   ax.set_title('Sensitivity to G_s')
   plt.show()

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

Estimate parameter uncertainty:

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.utils.optimization import calculate_confidence_intervals
   import numpy as np

   maxwell = Maxwell()
   maxwell.fit(X, y)

   # Calculate 95% confidence intervals
   ci = calculate_confidence_intervals(maxwell, X, y, alpha=0.05)

   print("95% Confidence Intervals:")
   for param_name, (lower, upper) in ci.items():
       value = maxwell.parameters.get_value(param_name)
       rel_error = (upper - lower) / (2 * value) * 100
       print(f"  {param_name}: {value:.2e} [{lower:.2e}, {upper:.2e}] "
             f"(±{rel_error:.1f}%)")

Parameter Correlation
~~~~~~~~~~~~~~~~~~~~~

Check for parameter correlation:

.. code-block:: python

   from rheojax.models import Zener
   import numpy as np

   zener = Zener()
   zener.fit(X, y)

   # Bootstrap to estimate correlation
   n_bootstrap = 100
   param_samples = {name: [] for name in zener.parameters.parameter_names}

   for i in range(n_bootstrap):
       # Resample data
       indices = np.random.choice(len(X), size=len(X), replace=True)
       X_boot = X[indices]
       y_boot = y[indices]

       # Fit
       model_boot = Zener()
       model_boot.fit(X_boot, y_boot)

       # Store parameters
       for name in param_samples.keys():
           param_samples[name].append(model_boot.parameters.get_value(name))

   # Calculate correlation matrix
   import pandas as pd

   df = pd.DataFrame(param_samples)
   corr = df.corr()

   print("Parameter Correlation Matrix:")
   print(corr)

   # High correlation (>0.9) indicates parameter redundancy

Serialization and Persistence
------------------------------

Saving Models
~~~~~~~~~~~~~

Save fitted models for later use:

.. code-block:: python

   from rheojax.models import FractionalMaxwellGel
   import pickle

   # Fit model
   model = FractionalMaxwellGel()
   model.fit(X, y)

   # Save to file
   with open('fitted_model.pkl', 'wb') as f:
       pickle.dump(model, f)

   # Load model
   with open('fitted_model.pkl', 'rb') as f:
       loaded_model = pickle.load(f)

   # Use loaded model
   y_pred = loaded_model.predict(X)

Model Export/Import
~~~~~~~~~~~~~~~~~~~

Export model parameters as JSON:

.. code-block:: python

   import json

   # Fit model
   model = FractionalMaxwellGel()
   model.fit(X, y)

   # Export parameters
   model_dict = {
       'model_type': type(model).__name__,
       'parameters': model.parameters.to_dict(),
       'metadata': {
           'fit_date': '2025-10-24',
           'r2': model.score(X, y),
           'data_source': 'experiment_01.txt'
       }
   }

   with open('model_params.json', 'w') as f:
       json.dump(model_dict, f, indent=2)

   # Import parameters
   with open('model_params.json', 'r') as f:
       loaded_dict = json.load(f)

   # Reconstruct model
   from rheojax.core.registry import ModelRegistry

   model_reconstructed = ModelRegistry.create(loaded_dict['model_type'])
   model_reconstructed.parameters.set_values(loaded_dict['parameters'])

Integration with External Libraries
------------------------------------

scikit-learn Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

rheo models follow scikit-learn API:

.. code-block:: python

   from rheojax.models import Maxwell
   from sklearn.model_selection import GridSearchCV
   from sklearn.base import BaseEstimator

   # Wrap rheo model for scikit-learn
   class RheoEstimator(BaseEstimator):
       def __init__(self, G_s=1e5, eta_s=1e3):
           self.G_s = G_s
           self.eta_s = eta_s

       def fit(self, X, y):
           self.model_ = Maxwell()
           self.model_.parameters.set_values({
               'G_s': self.G_s,
               'eta_s': self.eta_s
           })
           self.model_.fit(X, y)
           return self

       def predict(self, X):
           return self.model_.predict(X)

       def score(self, X, y):
           return self.model_.score(X, y)

   # Grid search over parameters
   param_grid = {
       'G_s': [1e4, 1e5, 1e6],
       'eta_s': [1e2, 1e3, 1e4]
   }

   grid_search = GridSearchCV(RheoEstimator(), param_grid, cv=3)
   grid_search.fit(X, y)

   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best score: {grid_search.best_score_:.4f}")

JAX Integration
~~~~~~~~~~~~~~~

Direct use of JAX arrays and operations:

.. code-block:: python

   from rheojax.models import Maxwell
   import jax.numpy as jnp
   import jax

   # Create JAX arrays
   X_jax = jnp.array(X)
   y_jax = jnp.array(y)

   maxwell = Maxwell()
   maxwell.fit(X_jax, y_jax)  # Works with JAX arrays

   # JIT compile predictions
   @jax.jit
   def predict_jit(freq, G_s, eta_s):
       tau = eta_s / G_s
       G_star = G_s / (1 + 1j * freq * tau)
       return jnp.abs(G_star)

   # Vectorize over parameters
   G_s_array = jnp.array([1e4, 1e5, 1e6])
   eta_s_array = jnp.array([1e2, 1e3, 1e4])

   predictions = jax.vmap(lambda g, e: predict_jit(X_jax, g, e))(
       G_s_array, eta_s_array
   )

Best Practices
--------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

Always provide reasonable initial guesses:

.. code-block:: python

   # Good: data-driven initialization
   G_typical = np.median(np.abs(y))
   model.parameters.set_value('G_s', G_typical * 0.5)

   # Bad: no initialization (uses arbitrary defaults)
   # model.fit(X, y)  # May fail or converge slowly

Bounds Setting
~~~~~~~~~~~~~~

Set physical bounds to constrain optimization:

.. code-block:: python

   # Good: physical bounds
   model.parameters.set_bounds('G_s', min_value=1e2, max_value=1e8)
   model.parameters.set_bounds('eta_s', min_value=1e0, max_value=1e6)

   # Bad: unbounded (may give non-physical results)
   # model.fit(X, y)

Validation
~~~~~~~~~~

Always validate fitted models:

.. code-block:: python

   # Check parameter values
   params = model.parameters.to_dict()
   for name, value in params.items():
       if value <= 0:
           print(f"Warning: {name} = {value} is non-physical!")

   # Check fit quality
   r2 = model.score(X, y)
   if r2 < 0.9:
       print(f"Warning: Poor fit (R² = {r2:.3f})")

   # Visual inspection
   import matplotlib.pyplot as plt
   plt.loglog(X, y, 'o', label='Data')
   plt.loglog(X, model.predict(X), '-', label='Model')
   plt.legend()
   plt.show()

Documentation
~~~~~~~~~~~~~

Document custom workflows:

.. code-block:: python

   def fit_with_validation(model, X, y, n_starts=5):
       """Fit model with multi-start optimization and validation.

       Parameters
       ----------
       model : BaseModel
           Model to fit
       X : array
           Independent variable
       y : array
           Dependent variable
       n_starts : int
           Number of random starts

       Returns
       -------
       model : BaseModel
           Best fitted model
       metrics : dict
           Fit quality metrics
       """
       best_score = -np.inf
       best_model = None

       for i in range(n_starts):
           # Random initialization
           for param in model.parameters.parameters.values():
               if param.bounds is not None:
                   low, high = param.bounds
                   param.value = np.random.uniform(low, high)

           # Fit
           model.fit(X, y)

           # Validate
           score = model.score(X, y)
           if score > best_score:
               best_score = score
               best_model = model

       # Calculate metrics
       y_pred = best_model.predict(X)
       metrics = {
           'r2': best_score,
           'rmse': np.sqrt(np.mean((y - y_pred)**2)),
           'parameters': best_model.parameters.to_dict()
       }

       return best_model, metrics

Common Patterns
---------------

Pattern 1: Custom Weighted Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell
   import jax.numpy as jnp

   @jax.jit
   def weighted_objective(params_array, X, y, weights):
       G_s, eta_s = params_array
       tau = eta_s / G_s
       G_star = G_s / (1 + 1j * X * tau)
       y_pred = jnp.abs(G_star)
       residuals = (y - y_pred) * weights
       return jnp.sum(residuals**2)

   # Emphasize low frequency
   weights = 1.0 / (1.0 + X)

   from rheojax.utils.optimization import nlsq_optimize
   maxwell = Maxwell()
   result = nlsq_optimize(
       lambda p: weighted_objective(p, X, y, weights),
       maxwell.parameters,
       use_jax=True
   )

Pattern 2: Hierarchical Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell, Zener, FractionalMaxwellGel

   # Start simple, increase complexity if needed
   models_hierarchy = [Maxwell(), Zener(), FractionalMaxwellGel()]

   for model in models_hierarchy:
       model.fit(X, y)
       r2 = model.score(X, y)

       if r2 > 0.95:  # Satisfactory fit
           print(f"Selected model: {type(model).__name__} (R² = {r2:.4f})")
           break
   else:
       print("Warning: No satisfactory fit found")

Pattern 3: Ensemble Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell, Zener, SpringPot
   import numpy as np

   # Fit multiple models
   models = [Maxwell(), Zener(), SpringPot()]
   for model in models:
       model.fit(X, y)

   # Ensemble prediction (average)
   predictions = np.array([m.predict(X) for m in models])
   ensemble_pred = np.mean(predictions, axis=0)

   # Weighted ensemble (by R²)
   weights = np.array([m.score(X, y) for m in models])
   weights /= np.sum(weights)
   weighted_ensemble = np.average(predictions, axis=0, weights=weights)

Summary
-------

The Modular API provides complete control over:

1. **Model instantiation** and parameter management
2. **Custom optimization** algorithms and objectives
3. **Transform composition** and data preprocessing
4. **Advanced fitting workflows** (multi-start, sequential, hierarchical)
5. **Integration** with external libraries (scikit-learn, JAX)

For standard workflows, use the :doc:`/user_guide/pipeline_api`.

Next Steps
----------

- :doc:`/user_guide/pipeline_api` - High-level workflow API
- :doc:`/user_guide/multi_technique_fitting` - Multi-technique fitting with shared parameters
- :doc:`/api/models` - Complete model API reference
- :doc:`/api/core` - Core classes (ParameterSet, RheoData, etc.)
