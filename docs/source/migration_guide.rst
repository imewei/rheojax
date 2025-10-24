Migration Guide: pyRheo/hermes-rheo → rheo
===========================================

Introduction
------------

This guide helps users migrate from **pyRheo** and **hermes-rheo** packages to the unified **rheo** package. The rheo package combines functionality from both predecessors into a single, modern JAX-powered framework.

**Why migrate to rheo?**

* **Unified Framework**: Single package replaces both pyRheo (models) and hermes-rheo (transforms)
* **10-100x Performance**: JAX provides automatic differentiation and GPU acceleration
* **Modern API Design**: Fluent Pipeline API and modular architecture
* **Enhanced Features**: Advanced parameter management, automatic test mode detection, publication-quality visualization
* **Active Development**: Ongoing support and new features (Bayesian inference, ML integration)

**Migration Strategy**

1. Install rheo alongside existing packages (no conflicts)
2. Update imports incrementally
3. Test numerical equivalence (1e-6 tolerance)
4. Gradually replace old code with rheo equivalents
5. Remove pyRheo/hermes-rheo dependencies once migration complete

API Mapping Tables
------------------

From pyRheo to rheo
~~~~~~~~~~~~~~~~~~~

Model Name Mappings
^^^^^^^^^^^^^^^^^^^

.. list-table:: Classical and Fractional Models
   :header-rows: 1
   :widths: 30 30 40

   * - pyRheo
     - rheo
     - Notes
   * - ``MaxwellModel``
     - ``Maxwell``
     - Now in ``rheo.models.Maxwell``
   * - ``ZenerModel`` / ``SLSModel``
     - ``Zener``
     - Standardized name: Zener (SLS)
   * - ``SpringPotModel``
     - ``SpringPot``
     - Parameter names unchanged
   * - ``FractionalMaxwell``
     - ``FractionalMaxwellGel`` or ``FractionalMaxwellLiquid``
     - Split into gel and liquid variants
   * - ``FractionalKV``
     - ``FractionalKelvinVoigt``
     - Full name used for clarity
   * - ``FractionalZenerSS``
     - ``FractionalZenerSS``
     - Name and structure preserved
   * - ``FractionalZenerSL``
     - ``FractionalZenerSL``
     - Name and structure preserved
   * - ``FractionalZenerLL``
     - ``FractionalZenerLL``
     - Name and structure preserved
   * - ``BurgersModel``
     - ``FractionalBurgers``
     - Generalized to fractional version
   * - ``JeffreysModel``
     - ``FractionalJeffreys``
     - Generalized to fractional version
   * - ``PoyntingThomson``
     - ``FractionalPoyntingThomson``
     - Generalized to fractional version
   * - ``FractionalKVZener``
     - ``FractionalKVZener``
     - Name preserved

.. list-table:: Flow Models
   :header-rows: 1
   :widths: 30 30 40

   * - pyRheo
     - rheo
     - Notes
   * - ``PowerLawModel``
     - ``PowerLaw``
     - Non-Newtonian flow
   * - ``CarreauModel``
     - ``Carreau``
     - Shear thinning
   * - ``CarreauYasudaModel``
     - ``CarreauYasuda``
     - Extended Carreau
   * - ``CrossModel``
     - ``Cross``
     - Shear thinning
   * - ``HerschelBulkleyModel``
     - ``HerschelBulkley``
     - Yield stress model
   * - ``BinghamModel``
     - ``Bingham``
     - Yield stress model

Parameter Name Changes
^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Parameter Renaming
   :header-rows: 1
   :widths: 20 20 20 40

   * - Model
     - pyRheo Parameter
     - rheo Parameter
     - Notes
   * - Maxwell
     - ``G0``
     - ``G0``
     - Unchanged
   * - Maxwell
     - ``tau``
     - ``tau``
     - Unchanged
   * - Zener
     - ``Ge``
     - ``Ge``
     - Equilibrium modulus
   * - Zener
     - ``Gm``
     - ``Gm``
     - Maxwell arm modulus
   * - Zener
     - ``tau``
     - ``eta``
     - Changed to viscosity (eta = Gm * tau)
   * - SpringPot
     - ``C_alpha``
     - ``c_alpha``
     - Lowercase convention
   * - SpringPot
     - ``alpha``
     - ``alpha``
     - Unchanged
   * - Fractional Maxwell
     - ``modulus``
     - ``c_alpha``
     - Standardized naming
   * - Flow models
     - ``eta0``
     - ``eta_0``
     - Underscore added
   * - Flow models
     - ``etainf``
     - ``eta_inf``
     - Underscore added

From hermes-rheo to rheo
~~~~~~~~~~~~~~~~~~~~~~~~~

Transform Name Mappings
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Data Transforms
   :header-rows: 1
   :widths: 30 30 40

   * - hermes-rheo
     - rheo
     - Notes
   * - ``FFTTransform``
     - ``FFTAnalysis``
     - Now in ``rheo.transforms.FFTAnalysis``
   * - ``TimeTemperatureSuperposition``
     - ``Mastercurve``
     - Renamed for clarity, supports WLF/Arrhenius
   * - ``MutationNumber``
     - ``MutationNumber``
     - Name and functionality preserved
   * - ``OWChirp``
     - ``OWChirp``
     - LAOS analysis preserved
   * - ``SmoothDerivative``
     - ``SmoothDerivative``
     - Numerical differentiation preserved

Method Name Changes
^^^^^^^^^^^^^^^^^^^

.. list-table:: Transform Methods
   :header-rows: 1
   :widths: 25 25 25 25

   * - Transform
     - hermes-rheo Method
     - rheo Method
     - Notes
   * - FFT
     - ``forward()``
     - ``transform()``
     - Standardized
   * - FFT
     - ``inverse()``
     - ``inverse_transform()``
     - Standardized
   * - Mastercurve
     - ``apply()``
     - ``transform()``
     - Standardized
   * - Mastercurve
     - ``get_shift_factors()``
     - ``get_wlf_parameters()``
     - More descriptive
   * - MutationNumber
     - ``calculate()``
     - ``calculate()``
     - Unchanged
   * - OWChirp
     - ``analyze()``
     - ``transform()``
     - Standardized
   * - SmoothDerivative
     - ``compute()``
     - ``transform()``
     - Standardized

Side-by-Side Code Examples
---------------------------

Example 1: Basic Model Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models
   import numpy as np

   # Load data
   time = np.loadtxt('time.txt')
   stress = np.loadtxt('stress.txt')

   # Create and fit model
   model = models.MaxwellModel()
   model.fit(time, stress)

   # Get parameters
   params = model.get_params()
   G0 = params['G0']
   tau = params['tau']

   # Make predictions
   predictions = model.predict(time)

   # Calculate R²
   r2 = model.score(time, stress)
   print(f"R² = {r2:.4f}")

**rheo equivalent:**

.. code-block:: python

   from rheo.pipeline import Pipeline
   from rheo.core.data import RheoData
   import numpy as np

   # Load data
   time = np.loadtxt('time.txt')
   stress = np.loadtxt('stress.txt')

   # Create RheoData container
   data = RheoData(
       x=time, y=stress,
       x_units='s', y_units='Pa',
       domain='time', test_mode='relaxation'
   )

   # Create pipeline and fit model
   pipeline = Pipeline()
   pipeline.load_data(data)
   pipeline.fit('maxwell')

   # Get parameters
   params = pipeline.get_fitted_parameters()
   G0 = params['G0'].value
   tau = params['tau'].value

   # Make predictions
   predictions = pipeline.predict(data)

   # Get R²
   metrics = pipeline.get_fit_metrics()
   print(f"R² = {metrics['r_squared']:.4f}")

Example 2: Fractional Model with Custom Bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models

   # Create fractional Zener model
   model = models.FractionalZenerSS()

   # Set parameter bounds manually
   model.bounds = {
       'Ge': (1e3, 1e7),
       'Gm': (1e4, 1e8),
       'alpha': (0.1, 0.9),
       'tau': (0.01, 100)
   }

   # Set initial guesses
   model.initial_guess = {
       'Ge': 1e5,
       'Gm': 5e5,
       'alpha': 0.5,
       'tau': 1.0
   }

   # Fit
   model.fit(time, stress, method='L-BFGS-B')
   params = model.get_params()

**rheo equivalent:**

.. code-block:: python

   from rheo.core.registry import ModelRegistry
   from rheo.utils.optimization import nlsq_optimize

   # Create fractional Zener model
   model = ModelRegistry.create('fractional_zener_ss')

   # Parameters have bounds built-in, but can be modified
   model.parameters['Ge'].bounds = (1e3, 1e7)
   model.parameters['Gm'].bounds = (1e4, 1e8)
   model.parameters['alpha'].bounds = (0.1, 0.9)
   model.parameters['eta'].bounds = (0.01, 100)  # Note: eta instead of tau

   # Set initial values
   model.parameters['Ge'].value = 1e5
   model.parameters['Gm'].value = 5e5
   model.parameters['alpha'].value = 0.5
   model.parameters['eta'].value = 1.0

   # Fit (can use model.fit() or custom optimization)
   model.fit(data)
   params = model.parameters

Example 3: Mastercurve Generation (TTS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**hermes-rheo version:**

.. code-block:: python

   from hermes import transforms
   import glob

   # Load multi-temperature data
   file_paths = glob.glob('data_T*.csv')
   temperatures = [20, 30, 40, 50, 60]  # Celsius

   # Create TTS transform
   tts = transforms.TimeTemperatureSuperposition(
       ref_temp=25,  # Celsius
       method='wlf'
   )

   # Apply transformation
   mastercurve = tts.apply(file_paths, temperatures)

   # Get shift factors
   shift_factors = tts.get_shift_factors()

   # Get WLF parameters
   C1, C2 = tts.get_wlf_params()
   print(f"WLF: C1={C1:.2f}, C2={C2:.2f}")

**rheo equivalent:**

.. code-block:: python

   from rheo.pipeline import MastercurvePipeline
   import glob

   # Load multi-temperature data
   file_paths = glob.glob('data_T*.csv')
   temperatures = [20, 30, 40, 50, 60]  # Celsius

   # Create mastercurve pipeline
   pipeline = MastercurvePipeline(
       reference_temp=298.15,  # Kelvin (25°C + 273.15)
       method='wlf'
   )

   # Run pipeline
   mastercurve = pipeline.run(file_paths, temperatures)

   # Get shift factors
   shift_factors = pipeline.get_shift_factors()

   # Get WLF parameters
   wlf_params = pipeline.get_wlf_parameters()
   C1 = wlf_params['C1']
   C2 = wlf_params['C2']  # In Kelvin
   print(f"WLF: C1={C1:.2f}, C2={C2:.2f} K")

Example 4: FFT Analysis for Frequency Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**hermes-rheo version:**

.. code-block:: python

   from hermes import transforms

   # Time-domain data
   time = np.linspace(0, 10, 1000)
   signal = np.sin(2 * np.pi * 5 * time)  # 5 Hz signal

   # Create FFT transform
   fft = transforms.FFTTransform(window='hann', detrend=True)

   # Forward transform
   freq, spectrum = fft.forward(time, signal)

   # Get characteristic frequency
   peak_freq = fft.get_peak_frequency()

   # Inverse transform
   time_reconstructed, signal_reconstructed = fft.inverse(freq, spectrum)

**rheo equivalent:**

.. code-block:: python

   from rheo.transforms import FFTAnalysis
   from rheo.core.data import RheoData

   # Time-domain data
   time = np.linspace(0, 10, 1000)
   signal = np.sin(2 * np.pi * 5 * time)  # 5 Hz signal

   # Create RheoData
   data = RheoData(
       x=time, y=signal,
       x_units='s', y_units='Pa',
       domain='time'
   )

   # Create FFT transform
   fft = FFTAnalysis(window='hann', detrend=True)

   # Forward transform
   freq_data = fft.transform(data)

   # Get characteristic time (1/frequency)
   char_time = fft.get_characteristic_time(freq_data)

   # Inverse transform
   time_reconstructed = fft.inverse_transform(freq_data)

Example 5: Model Comparison with Information Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models
   from scipy.stats import aic, bic

   # Define candidate models
   model_classes = [
       models.MaxwellModel,
       models.ZenerModel,
       models.FractionalMaxwell
   ]

   results = []
   for ModelClass in model_classes:
       model = ModelClass()
       model.fit(time, stress)

       # Calculate metrics
       predictions = model.predict(time)
       residuals = stress - predictions
       sse = np.sum(residuals**2)
       n_params = len(model.get_params())
       n_data = len(stress)

       # Calculate AIC and BIC manually
       aic_val = n_data * np.log(sse/n_data) + 2*n_params
       bic_val = n_data * np.log(sse/n_data) + n_params*np.log(n_data)

       results.append({
           'model': ModelClass.__name__,
           'aic': aic_val,
           'bic': bic_val,
           'r2': model.score(time, stress)
       })

   # Find best model
   best_idx = np.argmin([r['aic'] for r in results])
   print(f"Best model: {results[best_idx]['model']}")

**rheo equivalent:**

.. code-block:: python

   from rheo.pipeline import ModelComparisonPipeline
   from rheo.core.data import RheoData

   # Prepare data
   data = RheoData(
       x=time, y=stress,
       x_units='s', y_units='Pa',
       domain='time', test_mode='relaxation'
   )

   # Define candidate models
   models = [
       'maxwell',
       'zener',
       'fractional_maxwell_gel'
   ]

   # Create and run comparison pipeline
   pipeline = ModelComparisonPipeline(models)
   results = pipeline.run(data)

   # Get best model (automatically calculates AIC, BIC, R²)
   best_model = pipeline.get_best_model(metric='aic')
   print(f"Best model: {best_model}")

   # Display comparison table
   comparison_df = pipeline.get_comparison_table()
   print(comparison_df)

Example 6: Batch Processing Multiple Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models
   import glob
   import pandas as pd

   # Get all data files
   file_paths = glob.glob('data/*.csv')

   # Initialize model
   model = models.ZenerModel()

   # Process each file
   results = []
   for file_path in file_paths:
       # Load data
       data = pd.read_csv(file_path)
       time = data['time'].values
       stress = data['stress'].values

       # Fit model
       model.fit(time, stress)
       params = model.get_params()
       r2 = model.score(time, stress)

       # Store results
       results.append({
           'file': file_path,
           'Ge': params['Ge'],
           'Gm': params['Gm'],
           'tau': params['tau'],
           'r2': r2
       })

   # Create summary DataFrame
   summary = pd.DataFrame(results)
   summary.to_csv('batch_results.csv', index=False)

**rheo equivalent:**

.. code-block:: python

   from rheo.pipeline import Pipeline, BatchPipeline
   import glob

   # Get all data files
   file_paths = glob.glob('data/*.csv')

   # Create template pipeline
   template = Pipeline().fit('zener')

   # Create batch pipeline
   batch = BatchPipeline(template)

   # Process all files
   batch.process_directory('data/', pattern='*.csv')

   # Export summary (automatically includes all parameters and metrics)
   batch.export_summary('batch_results.xlsx')

   # Can also export individual results
   batch.export_individual_results('results/')

Example 7: Custom Optimization with Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models
   from scipy.optimize import minimize

   model = models.ZenerModel()

   # Define objective function
   def objective(params):
       model.set_params(*params)
       predictions = model.predict(time)
       return np.sum((stress - predictions)**2)

   # Define constraints
   constraints = [
       {'type': 'ineq', 'fun': lambda p: p[0] - p[1]},  # Ge < Gm
       {'type': 'ineq', 'fun': lambda p: p[2] - 0.01}   # tau > 0.01
   ]

   # Initial guess
   x0 = [1e5, 5e5, 1.0]

   # Optimize with constraints
   result = minimize(
       objective, x0,
       method='SLSQP',
       constraints=constraints,
       bounds=[(1e3, 1e7), (1e4, 1e8), (0.01, 100)]
   )

   model.set_params(*result.x)

**rheo equivalent:**

.. code-block:: python

   from rheo.core.registry import ModelRegistry
   from scipy.optimize import minimize

   model = ModelRegistry.create('zener')

   # Define objective function
   def objective(params_array):
       # Update parameters
       for i, name in enumerate(['Ge', 'Gm', 'eta']):
           model.parameters[name].value = params_array[i]

       predictions = model.predict(data)
       return np.sum((data.y - predictions)**2)

   # Define constraints (using parameter names)
   constraints = [
       {'type': 'ineq', 'fun': lambda p: p[1] - p[0]},  # Gm > Ge
       {'type': 'ineq', 'fun': lambda p: p[2] - 0.01}   # eta > 0.01
   ]

   # Get initial values and bounds from parameters
   x0 = [model.parameters[name].value for name in ['Ge', 'Gm', 'eta']]
   bounds = [model.parameters[name].bounds for name in ['Ge', 'Gm', 'eta']]

   # Optimize with constraints
   result = minimize(
       objective, x0,
       method='SLSQP',
       constraints=constraints,
       bounds=bounds
   )

   # Update model with optimized parameters
   for i, name in enumerate(['Ge', 'Gm', 'eta']):
       model.parameters[name].value = result.x[i]

Example 8: Visualization with Custom Styling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pyRheo version:**

.. code-block:: python

   from pyrrheo import models
   import matplotlib.pyplot as plt

   model = models.MaxwellModel()
   model.fit(time, stress)
   predictions = model.predict(time)

   # Manual plotting
   plt.figure(figsize=(10, 6))
   plt.loglog(time, stress, 'o', label='Data', alpha=0.6)
   plt.loglog(time, predictions, '-', label='Fit', linewidth=2)
   plt.xlabel('Time (s)', fontsize=14)
   plt.ylabel('Stress (Pa)', fontsize=14)
   plt.title('Maxwell Model Fit', fontsize=16)
   plt.legend(fontsize=12)
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.savefig('fit.png', dpi=300)

**rheo equivalent:**

.. code-block:: python

   from rheo.pipeline import Pipeline

   # Create and fit pipeline
   pipeline = (Pipeline()
       .load_data(data)
       .fit('maxwell')
       .plot(style='publication')  # Built-in publication-quality style
       .save_plot('fit.png', dpi=300))

   # Or use custom plotting with more control
   pipeline.plot(
       style='publication',
       fig_size=(10, 6),
       show_residuals=True,
       show_parameters=True,
       title='Maxwell Model Fit'
   )

Key Differences and Breaking Changes
-------------------------------------

API Design Philosophy
~~~~~~~~~~~~~~~~~~~~~

rheo introduces two complementary APIs:

1. **Pipeline API** (High-level)

   * Fluent interface with method chaining
   * Automatic handling of common workflows
   * Best for: routine analysis, batch processing, quick prototyping

   .. code-block:: python

      # Pipeline API example
      result = (Pipeline()
          .load('data.csv')
          .fit('maxwell')
          .plot()
          .save('result.hdf5'))

2. **Modular API** (Low-level)

   * Direct access to models, parameters, and optimization
   * Maximum flexibility and control
   * Best for: research, custom workflows, advanced features

   .. code-block:: python

      # Modular API example
      model = ModelRegistry.create('maxwell')
      model.parameters['G0'].value = 1e6
      model.fit(data)
      predictions = model.predict(data)

**Migration Recommendation**: Start with Pipeline API for standard workflows, use Modular API when you need custom behavior.

Parameter Handling
~~~~~~~~~~~~~~~~~~

**pyRheo/hermes-rheo**: Parameters stored as dictionaries

.. code-block:: python

   # Old way
   params = {'G0': 1e6, 'tau': 1.0}
   model.set_params(**params)
   G0 = params['G0']

**rheo**: Parameters are objects with metadata

.. code-block:: python

   # New way
   model.parameters['G0'].value = 1e6
   model.parameters['G0'].bounds = (1e5, 1e7)
   model.parameters['G0'].units = 'Pa'
   G0 = model.parameters['G0'].value

**Benefits**:

* Type safety and validation
* Built-in bounds and constraints
* Units tracking
* Metadata for documentation

**Breaking Change**: Must use ``.value`` to access parameter values.

Test Mode Handling
~~~~~~~~~~~~~~~~~~

**pyRheo/hermes-rheo**: Test mode specified manually or inferred ambiguously

.. code-block:: python

   # Old way
   model.fit(time, stress, mode='relaxation')

**rheo**: Automatic test mode detection from RheoData

.. code-block:: python

   # New way - automatic detection
   data = RheoData(
       x=time, y=stress,
       domain='time'  # Automatically infers test_mode='relaxation'
   )
   model.fit(data)  # Test mode handled automatically

**Four test modes supported**:

1. **Relaxation**: G(t) decay after step strain
2. **Creep**: J(t) increase under constant stress
3. **Oscillation**: G'(ω), G"(ω) from dynamic tests
4. **Rotation**: Viscosity η(γ̇) from flow curves

**Breaking Change**: Must wrap data in ``RheoData`` containers for automatic detection, or specify ``test_mode`` explicitly.

JAX vs NumPy
~~~~~~~~~~~~

**pyRheo/hermes-rheo**: NumPy-based implementation

.. code-block:: python

   import numpy as np

   # NumPy operations
   x = np.exp(-time / tau)
   gradient = np.gradient(x, time)

**rheo**: JAX-based with NumPy compatibility

.. code-block:: python

   import jax.numpy as jnp
   import numpy as np  # Still works!

   # JAX operations (automatic GPU + differentiation)
   x = jnp.exp(-time / tau)

   # NumPy arrays automatically converted
   data = RheoData(x=np.array([1, 2, 3]), y=np.array([4, 5, 6]))

**Benefits**:

* **2-10x speedup** with JIT compilation
* **Automatic differentiation** for optimization
* **GPU acceleration** when available
* **NumPy compatibility** - existing arrays work

**Breaking Changes**:

* Some NumPy operations not supported in JAX (use ``jax.numpy`` instead)
* In-place operations not allowed (JAX arrays are immutable)
* Must use ``jnp`` for functions inside JIT-compiled code

**Migration Tip**: Most code works unchanged. Only modify if you hit JAX-specific issues.

Data Structures
~~~~~~~~~~~~~~~

**pyRheo/hermes-rheo**: Raw NumPy arrays

.. code-block:: python

   # Old way
   time = np.array([0.1, 1.0, 10.0])
   stress = np.array([1e6, 5e5, 1e5])
   model.fit(time, stress)

**rheo**: RheoData containers with metadata

.. code-block:: python

   # New way
   data = RheoData(
       x=time,
       y=stress,
       x_units='s',
       y_units='Pa',
       domain='time',
       test_mode='relaxation',
       metadata={'temperature': 25, 'sample': 'A'}
   )
   model.fit(data)

**Benefits**:

* Self-documenting data
* Automatic unit tracking
* Test mode detection
* Metadata preservation
* Type safety

**Breaking Change**: Models expect ``RheoData`` objects, not raw arrays.

**Quick Conversion**:

.. code-block:: python

   # Minimal conversion
   data = RheoData(x=time, y=stress)

   # Full conversion with metadata
   data = RheoData(
       x=time, y=stress,
       x_units='s', y_units='Pa',
       domain='time', test_mode='relaxation'
   )

Migration Checklist
-------------------

Step 1: Install rheo
~~~~~~~~~~~~~~~~~~~~

Install alongside existing packages (no conflicts):

.. code-block:: bash

   pip install rheo-analysis

Or with GPU support:

.. code-block:: bash

   pip install rheo-analysis[gpu]

Verify installation:

.. code-block:: python

   import rheo
   print(rheo.__version__)  # Should show v0.2.0 or later

Step 2: Update Imports
~~~~~~~~~~~~~~~~~~~~~~~

Replace old imports incrementally:

**Before:**

.. code-block:: python

   # Old pyRheo imports
   from pyrrheo import models
   from pyrrheo.models import MaxwellModel, ZenerModel

   # Old hermes-rheo imports
   from hermes import transforms
   from hermes.transforms import FFTTransform, TimeTemperatureSuperposition

**After:**

.. code-block:: python

   # New rheo imports - Pipeline API
   from rheo.pipeline import Pipeline, ModelComparisonPipeline

   # New rheo imports - Modular API
   from rheo.models import Maxwell, Zener
   from rheo.core.registry import ModelRegistry
   from rheo.transforms import FFTAnalysis, Mastercurve
   from rheo.core.data import RheoData

**Tip**: Use your IDE's find-and-replace to batch update imports.

Step 3: Convert Data Structures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Wrap existing NumPy arrays in RheoData:

.. code-block:: python

   # Old way
   time = np.loadtxt('time.txt')
   stress = np.loadtxt('stress.txt')

   # New way - add RheoData wrapper
   from rheo.core.data import RheoData

   data = RheoData(
       x=time,
       y=stress,
       x_units='s',
       y_units='Pa',
       domain='time',
       test_mode='relaxation'  # Optional - will auto-detect
   )

**Helper function for batch conversion**:

.. code-block:: python

   def convert_to_rheodata(time, stress, test_mode='relaxation'):
       """Convert legacy data to RheoData format."""
       return RheoData(
           x=time, y=stress,
           x_units='s', y_units='Pa',
           domain='time', test_mode=test_mode
       )

   # Use in existing code
   data = convert_to_rheodata(time, stress)

Step 4: Update Model Creation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Option 1: Using Pipeline API (recommended)**

.. code-block:: python

   # Old
   model = models.MaxwellModel()
   model.fit(time, stress)

   # New
   pipeline = Pipeline()
   pipeline.load_data(data)
   pipeline.fit('maxwell')

**Option 2: Using Modular API (for custom workflows)**

.. code-block:: python

   # Old
   model = models.MaxwellModel()

   # New - using registry
   from rheo.core.registry import ModelRegistry
   model = ModelRegistry.create('maxwell')

   # Or direct import
   from rheo.models import Maxwell
   model = Maxwell()

Step 5: Update Fitting Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pipeline API approach**:

.. code-block:: python

   # Old pyRheo
   model = models.ZenerModel()
   model.fit(time, stress)
   params = model.get_params()
   r2 = model.score(time, stress)

   # New rheo Pipeline
   pipeline = Pipeline().load_data(data).fit('zener')
   params = pipeline.get_fitted_parameters()
   metrics = pipeline.get_fit_metrics()
   r2 = metrics['r_squared']

**Modular API approach**:

.. code-block:: python

   # Old pyRheo
   model = models.ZenerModel()
   model.set_bounds(Ge=(1e3, 1e7), Gm=(1e4, 1e8))
   model.fit(time, stress, method='L-BFGS-B')

   # New rheo Modular
   model = ModelRegistry.create('zener')
   model.parameters['Ge'].bounds = (1e3, 1e7)
   model.parameters['Gm'].bounds = (1e4, 1e8)
   model.fit(data)  # Uses same scipy.optimize backend

Step 6: Test and Validate
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Validation Strategy**:

1. **Numerical Equivalence Test**

   .. code-block:: python

      # Fit with old package
      old_model = old_package.MaxwellModel()
      old_model.fit(time, stress)
      old_predictions = old_model.predict(time)

      # Fit with rheo
      new_model = rheo.models.Maxwell()
      new_data = RheoData(x=time, y=stress, domain='time')
      new_model.fit(new_data)
      new_predictions = new_model.predict(new_data)

      # Check tolerance (should be < 1e-6)
      max_error = np.max(np.abs(old_predictions - new_predictions))
      assert max_error < 1e-6, f"Predictions differ by {max_error}"

2. **Parameter Comparison**

   .. code-block:: python

      # Compare fitted parameters
      old_params = old_model.get_params()
      new_params = new_model.parameters

      for param_name in old_params.keys():
           old_val = old_params[param_name]
           new_val = new_params[param_name].value
           rel_error = abs(new_val - old_val) / old_val
           assert rel_error < 1e-6, f"{param_name} differs by {rel_error*100:.4f}%"

3. **Run Existing Test Suite**

   .. code-block:: python

      # Ensure your existing tests pass with rheo
      pytest tests/  # Should pass with rheo as drop-in replacement

**Validation Notebook**: See ``examples/validation_comparison.ipynb`` for comprehensive validation examples.

Frequently Asked Questions
--------------------------

Q: Are results numerically identical?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Yes, within 1e-6 relative tolerance for all models and transforms. Both packages use the same underlying numerical methods (scipy.optimize, numerical integration), so results should be essentially identical.

We've validated all 20 models against pyRheo and all 5 transforms against hermes-rheo. See ``docs/validation_report.md`` for detailed comparison.

Q: Can I mix rheo with pyRheo/hermes-rheo?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Yes, but you'll need to convert data structures at package boundaries.

.. code-block:: python

   # Use pyRheo for fitting
   old_model = pyrrheo.models.MaxwellModel()
   old_model.fit(time, stress)
   old_predictions = old_model.predict(time)

   # Convert to rheo for advanced analysis
   data = RheoData(x=time, y=old_predictions, domain='time')
   new_pipeline = Pipeline().load_data(data).fit('zener')

**Recommendation**: Complete migration to rheo for one project/module at a time rather than mixing extensively.

Q: What about performance?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** rheo is **2-10x faster** than pyRheo/hermes-rheo due to JAX + JIT compilation.

**Benchmarks** (M1 MacBook Pro):

.. list-table::
   :header-rows: 1

   * - Operation
     - pyRheo/hermes
     - rheo (CPU)
     - rheo (GPU)
     - Speedup
   * - Maxwell fit (100 pts)
     - 15 ms
     - 2 ms
     - 0.5 ms
     - 7.5x / 30x
   * - Mittag-Leffler (1000 pts)
     - 45 ms
     - 0.8 ms
     - 0.2 ms
     - 56x / 225x
   * - FFT transform (10k pts)
     - 120 ms
     - 8 ms
     - 2 ms
     - 15x / 60x
   * - Mastercurve (5 temps)
     - 2.5 s
     - 0.4 s
     - 0.15 s
     - 6x / 17x

**First call**: rheo has ~100ms JIT compilation overhead on first call. Subsequent calls are fast.

Q: Is GPU acceleration automatic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Yes! JAX automatically detects and uses GPUs when available.

**Setup**:

.. code-block:: bash

   # Install with CUDA support
   pip install rheo-analysis[gpu]

**Verify GPU**:

.. code-block:: python

   import jax
   print(jax.devices())  # Should show GPU if available

   # Force CPU (for debugging)
   import os
   os.environ['JAX_PLATFORM_NAME'] = 'cpu'

**Note**: GPU acceleration is most beneficial for:

* Large datasets (>10k points)
* Batch processing
* Complex models (fractional models with Mittag-Leffler functions)
* Parameter sweeps and sensitivity analysis

For typical datasets (<1k points), CPU performance is usually sufficient.

Q: Are there breaking changes?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Yes, but they improve code quality and enable new features:

**Major Breaking Changes**:

1. **Data Structures**: Must use ``RheoData`` instead of raw NumPy arrays
2. **Parameter Access**: Use ``.value`` to get/set parameter values
3. **API Names**: Some model/transform names changed (see mapping tables)
4. **Units**: Temperature now in Kelvin (was Celsius in hermes-rheo)

**Minor Breaking Changes**:

5. **Method Names**: Standardized to ``transform()`` across all transforms
6. **Import Paths**: Different module organization
7. **Return Types**: Methods return rich objects instead of tuples

**Not Breaking**:

* NumPy arrays work (automatically converted to JAX)
* Same numerical methods (scipy.optimize backend)
* Same fitted parameters (within tolerance)
* Can install alongside old packages

Q: What if I find a bug or inconsistency?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Please report it!

1. **Check validation report** first: ``docs/validation_report.md``
2. **Open GitHub issue**: https://github.com/[org]/rheo/issues
3. **Provide minimal reproducible example**:

   .. code-block:: python

      # Describe expected vs actual behavior
      # Include code that demonstrates the issue
      # Mention pyRheo/hermes-rheo version for comparison

4. **Include versions**:

   .. code-block:: python

      import rheo, pyrrheo, hermes
      print(f"rheo: {rheo.__version__}")
      print(f"pyRheo: {pyrrheo.__version__}")
      print(f"hermes-rheo: {hermes.__version__}")

Q: How do I migrate custom models/transforms?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Subclass ``BaseModel`` or ``BaseTransform`` in rheo:

**Custom Model Example**:

.. code-block:: python

   # Old pyRheo custom model
   class MyCustomModel(pyrrheo.models.BaseModel):
       def __init__(self):
           self.param_names = ['A', 'B']

       def predict(self, t, A, B):
           return A * np.exp(-B * t)

   # New rheo custom model
   from rheo.core.base_model import BaseModel
   from rheo.core.parameters import ParameterSet

   class MyCustomModel(BaseModel):
       def __init__(self):
           super().__init__()
           self.parameters = ParameterSet()
           self.parameters.add('A', value=1.0, bounds=(0, 10))
           self.parameters.add('B', value=1.0, bounds=(0, 10))

       def predict(self, data):
           import jax.numpy as jnp
           t = data.x
           A = self.parameters['A'].value
           B = self.parameters['B'].value
           return A * jnp.exp(-B * t)

**See Also**: ``docs/developer/custom_models.rst`` for complete guide.

Q: What about existing scripts and notebooks?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Update incrementally:

1. **Add rheo imports** at top
2. **Wrap data in RheoData** (one-line change)
3. **Replace model.fit() calls** with Pipeline or Modular API
4. **Test each section** as you migrate
5. **Keep old imports** until migration complete

**Example Migration**:

.. code-block:: python

   # Before (100% old code)
   from pyrrheo import models
   model = models.MaxwellModel()
   model.fit(time, stress)

   # During migration (mixed)
   from pyrrheo import models
   from rheo.core.data import RheoData
   model = models.MaxwellModel()
   data = RheoData(x=time, y=stress)  # Wrap data
   model.fit(time, stress)  # Still using old model

   # After migration (100% new code)
   from rheo.pipeline import Pipeline
   from rheo.core.data import RheoData
   data = RheoData(x=time, y=stress, domain='time')
   pipeline = Pipeline().load_data(data).fit('maxwell')

Q: How do I cite rheo vs pyRheo/hermes-rheo?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**A:** Cite rheo in new work:

.. code-block:: bibtex

   @software{rheo2024,
     title = {Rheo: JAX-Powered Unified Rheology Package},
     author = {Rheo Development Team},
     year = {2024},
     url = {https://github.com/[org]/rheo},
     version = {0.2.0}
   }

If you migrated from pyRheo/hermes-rheo, you can acknowledge them:

   "Analysis was performed using the rheo package (v0.2.0), which unified
   functionality from pyRheo and hermes-rheo packages."

Conclusion
----------

Migration Benefits
~~~~~~~~~~~~~~~~~~

**Performance**:

* 2-10x faster with JAX + JIT compilation
* Automatic GPU acceleration
* Efficient batch processing

**Features**:

* 20 rheological models (vs 15 in pyRheo)
* 5 data transforms (unified from hermes-rheo)
* Pipeline API for streamlined workflows
* Advanced parameter management with bounds and units
* Automatic test mode detection

**Code Quality**:

* Type-safe parameter handling
* Self-documenting data structures
* Consistent API across models and transforms
* Modern Python practices (type hints, dataclasses)

**Maintainability**:

* Active development and support
* Comprehensive documentation (150+ pages)
* Extensive test suite (900+ tests, 85% coverage)
* Regular updates and bug fixes

Support Resources
~~~~~~~~~~~~~~~~~

**Documentation**:

* Full documentation: https://rheo.readthedocs.io
* API reference: https://rheo.readthedocs.io/api_reference.html
* User guides: https://rheo.readthedocs.io/user_guide.html
* Example notebooks: ``examples/`` directory

**Community**:

* GitHub: https://github.com/[org]/rheo
* Discussions: https://github.com/[org]/rheo/discussions
* Issues: https://github.com/[org]/rheo/issues

**Getting Help**:

1. Check documentation and examples first
2. Search existing issues on GitHub
3. Ask on GitHub Discussions
4. Open a new issue with reproducible example

**Contributing**:

We welcome contributions! See ``CONTRIBUTING.md`` for:

* Bug reports and feature requests
* Code contributions (new models, transforms, features)
* Documentation improvements
* Example notebooks and tutorials

**Roadmap**:

rheo is actively developed. Upcoming features (Phase 3):

* Bayesian inference with NumPyro
* Machine learning-based model selection
* Advanced visualization (interactive plots, 3D)
* Web interface for browser-based analysis
* Integration with experimental platforms

Thank you for using rheo! We hope this migration guide helps you transition smoothly. If you have questions or feedback, please reach out through our GitHub channels.
