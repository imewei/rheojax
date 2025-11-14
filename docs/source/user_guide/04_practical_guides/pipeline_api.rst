Pipeline API Tutorial
=====================

The Pipeline API provides a fluent, high-level interface for rheological analysis workflows. It enables intuitive method chaining from data loading through model fitting, visualization, and export.

Why Use the Pipeline API?
-------------------------

**Advantages**:

- Clean, readable code with method chaining
- Automatic state management (no manual variable tracking)
- Built-in error handling and validation
- Pre-configured workflows for common tasks
- Batch processing capabilities
- Consistent API across different analysis types

**When to use Pipeline API vs Modular API**:

- **Use Pipeline API** for: Standard workflows, rapid prototyping, batch processing, teaching
- **Use Modular API** for: Custom algorithms, fine-grained control, complex parameter manipulation

Quick Start
-----------

Basic Workflow Pattern
~~~~~~~~~~~~~~~~~~~~~~

The typical pipeline follows this pattern:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Create pipeline and chain operations
   results = (Pipeline()
       .load('data.csv')           # Load data
       .transform('smooth')        # Optional: apply transforms
       .fit('maxwell')             # Fit model
       .plot(show=True)            # Visualize
       .save('results.hdf5')       # Export
       .get_results())             # Get results dict

   # Access results
   print(f"R^2 = {results['r2']:.4f}")
   print(f"Parameters: {results['parameters']}")

Each method returns `self`, enabling clean method chaining.

Simple Example
~~~~~~~~~~~~~~

Load data, fit model, and visualize in 4 lines:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   pipeline = Pipeline()
   pipeline.load('oscillation_data.txt')
   pipeline.fit('maxwell')
   pipeline.plot(show=True)

   # Get fitted parameters
   results = pipeline.get_results()
   G_s = results['parameters']['G_s']
   eta_s = results['parameters']['eta_s']
   print(f"G_s = {G_s:.2e} Pa, eta_s = {eta_s:.2e} Pa*s")

Core Pipeline Operations
------------------------

Loading Data
~~~~~~~~~~~~

**From File**:

.. code-block:: python

   # Auto-detect format
   pipeline = Pipeline().load('data.txt')

   # Specify format explicitly
   pipeline = Pipeline().load('data.csv', format='csv',
                               x_col='freq', y_col='modulus')

   # Load Excel with sheet selection
   pipeline = Pipeline().load('results.xlsx', format='excel',
                               sheet='Frequency Sweep')

**From RheoData Object**:

.. code-block:: python

   from rheojax.core import RheoData
   import numpy as np

   # Create data programmatically
   freq = np.logspace(-1, 2, 50)
   modulus = 1000 + 500 * freq**0.5
   data = RheoData(x=freq, y=modulus, x_units='Hz', y_units='Pa',
                   domain='frequency')

   # Load into pipeline
   pipeline = Pipeline().load(data)

**Multiple Files**:

.. code-block:: python

   # Load multiple files (for mastercurves, multi-temperature)
   pipeline = Pipeline().load([
       'data_25C.txt',
       'data_50C.txt',
       'data_75C.txt'
   ])

Applying Transforms
~~~~~~~~~~~~~~~~~~~

Apply data transforms before fitting:

.. code-block:: python

   # Single transform
   pipeline = (Pipeline()
       .load('time_series.txt')
       .transform('fft', window='hann'))  # Time -> frequency

   # Multiple transforms (chained)
   pipeline = (Pipeline()
       .load('noisy_data.txt')
       .transform('smooth', method='savgol', window=11)
       .transform('fft', window='hann'))

   # Transform with custom parameters
   pipeline = (Pipeline()
       .load('data.txt')
       .transform('mastercurve',
                  reference_temp=50,
                  method='wlf',
                  temperatures=[25, 50, 75]))

**Available Transforms**:

- ``'smooth'`` or ``'smooth_derivative'``: Smooth noisy data
- ``'fft'`` or ``'fft_analysis'``: FFT analysis (time -> frequency)
- ``'mastercurve'``: Time-temperature superposition
- ``'mutation'`` or ``'mutation_number'``: Calculate mutation number
- ``'owchirp'``: OWChirp analysis for LAOS

Fitting Models
~~~~~~~~~~~~~~

**By Name (Registry)**:

.. code-block:: python

   # Simple model names
   pipeline = Pipeline().load('data.txt').fit('maxwell')
   pipeline = Pipeline().load('data.txt').fit('zener')
   pipeline = Pipeline().load('data.txt').fit('herschel_bulkley')

   # Full names also work
   pipeline = Pipeline().load('data.txt').fit('fractional_maxwell_gel')

**With Initial Parameters**:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell',
            initial_params={'G_s': 1e5, 'eta_s': 1e3},
            bounds={'G_s': (1e3, 1e7), 'eta_s': (1e1, 1e5)}))

**Model Instance**:

.. code-block:: python

   from rheojax.models import Maxwell

   model = Maxwell()
   pipeline = Pipeline().load('data.txt').fit(model)

**Multiple Models (Comparison)**:

.. code-block:: python

   # Fit multiple models for comparison
   pipeline = (Pipeline()
       .load('data.txt')
       .fit(['maxwell', 'zener', 'springpot']))

   # Get comparison results
   results = pipeline.get_results()
   for model_name, model_results in results['models'].items():
       print(f"{model_name}: R^2 = {model_results['r2']:.4f}")

Visualization
~~~~~~~~~~~~~

**Basic Plotting**:

.. code-block:: python

   # Show interactive plot
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .plot(show=True))

   # Save to file
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .plot(save='fit_result.png', dpi=300))

   # Both show and save
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .plot(show=True, save='fit_result.pdf'))

**Customization**:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .plot(
           show=True,
           style='publication',      # 'default', 'publication', 'presentation'
           include_residuals=True,   # Add residual subplot
           title='Maxwell Model Fit',
           xlabel='Frequency (rad/s)',
           ylabel='|G*| (Pa)',
           figsize=(10, 6)))

**Multiple Plots**:

.. code-block:: python

   # Plot data, fit, and residuals separately
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell'))

   pipeline.plot(plot_type='data', show=True)
   pipeline.plot(plot_type='fit', show=True)
   pipeline.plot(plot_type='residuals', show=True)

Saving Results
~~~~~~~~~~~~~~

**HDF5 (Full Fidelity)**:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .save('results.hdf5'))

   # Load back later
   pipeline2 = Pipeline().load('results.hdf5')

**Excel (Reporting)**:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .save('report.xlsx', format='excel', include_plots=True))

   # Multi-sheet output with parameters, predictions, residuals

**CSV (Data Export)**:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell')
       .save('predictions.csv', export='predictions'))

   # Options: 'predictions', 'residuals', 'parameters'

Getting Results
~~~~~~~~~~~~~~~

Retrieve results programmatically:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell'))

   results = pipeline.get_results()

   # Access components
   parameters = results['parameters']       # Dict of parameter values
   r2 = results['r2']                       # R^2 score
   rmse = results['rmse']                   # Root mean squared error
   predictions = results['predictions']     # Model predictions
   residuals = results['residuals']         # Residuals
   data = results['data']                   # Original RheoData object
   model = results['model']                 # Fitted model instance

   # Use model for further predictions
   import numpy as np
   new_freq = np.logspace(-2, 3, 100)
   new_pred = model.predict(new_freq)

Complete Example
~~~~~~~~~~~~~~~~

Full workflow in one chain:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   results = (Pipeline()
       .load('experimental_data.txt')
       .transform('smooth', method='savgol', window=11)
       .fit('fractional_maxwell_gel',
            initial_params={'G_s': 1e4, 'V': 1e3, 'alpha': 0.5})
       .plot(show=True, save='analysis.png', style='publication')
       .save('results.hdf5')
       .get_results())

   print(f"Analysis complete!")
   print(f"R^2 = {results['r2']:.4f}")
   print(f"Parameters:")
   for name, value in results['parameters'].items():
       print(f"  {name} = {value:.4e}")

Specialized Workflows
---------------------

MastercurvePipeline
~~~~~~~~~~~~~~~~~~~

Pre-configured pipeline for time-temperature superposition:

.. code-block:: python

   from rheojax.pipeline import MastercurvePipeline

   # Create mastercurve pipeline
   mc_pipeline = MastercurvePipeline(
       reference_temp=50,      # Reference temperature ( degC)
       method='wlf',           # 'wlf' or 'arrhenius'
       optimize=True           # Optimize WLF/Arrhenius parameters
   )

   # Load multi-temperature data
   files = ['data_25C.txt', 'data_40C.txt', 'data_55C.txt', 'data_70C.txt']
   temperatures = [25, 40, 55, 70]

   # Run mastercurve analysis
   results = mc_pipeline.run(files, temperatures)

   # Access results
   mastercurve = results['mastercurve']
   shift_factors = results['shift_factors']
   wlf_params = results['wlf_parameters']  # C1, C2

   print(f"WLF C1 = {wlf_params['C1']:.2f}")
   print(f"WLF C2 = {wlf_params['C2']:.2f} K")

   # Fit model to mastercurve
   mc_pipeline.fit('fractional_maxwell_model')
   mc_pipeline.plot(show=True, style='publication')

**With Model Fitting**:

.. code-block:: python

   # Create mastercurve and fit in one step
   mc_pipeline = (MastercurvePipeline(reference_temp=50, method='wlf')
       .run(files, temperatures)
       .fit('fractional_maxwell_gel')
       .plot(show=True)
       .save('mastercurve_analysis.hdf5'))

   results = mc_pipeline.get_results()

ModelComparisonPipeline
~~~~~~~~~~~~~~~~~~~~~~~

Compare multiple models systematically:

.. code-block:: python

   from rheojax.pipeline import ModelComparisonPipeline

   # Define models to compare
   models = ['maxwell', 'zener', 'fractional_maxwell_gel',
             'fractional_kelvin_voigt', 'springpot']

   # Create comparison pipeline
   comparison = ModelComparisonPipeline(models)

   # Load data and run comparison
   comparison.load('data.txt')
   comparison.run()

   # Get comparison results
   results = comparison.get_results()
   comparison_table = results['comparison']

   # Print comparison
   print("\nModel Comparison:")
   print(f"{'Model':<30} {'R^2':<10} {'RMSE':<12} {'AIC':<12} {'Parameters':<5}")
   print("-" * 75)
   for row in comparison_table:
       print(f"{row['model']:<30} {row['r2']:<10.4f} {row['rmse']:<12.2e} "
             f"{row['aic']:<12.1f} {row['n_params']:<5}")

   # Get best model
   best_model = comparison.get_best_model(criterion='aic')  # 'aic', 'bic', 'r2'
   print(f"\nBest model (AIC): {best_model['name']}")
   print(f"R^2 = {best_model['r2']:.4f}")

**With Visualization**:

.. code-block:: python

   # Compare and visualize all models
   comparison = (ModelComparisonPipeline(models)
       .load('data.txt')
       .run()
       .plot_comparison(show=True))  # Multi-panel comparison plot

   # Plot ranking by different criteria
   comparison.plot_ranking(criterion='aic', show=True)
   comparison.plot_ranking(criterion='r2', show=True)

   # Export comparison table
   comparison.save('model_comparison.xlsx', format='excel')

CreepToRelaxationPipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

Convert creep data to relaxation modulus:

.. code-block:: python

   from rheojax.pipeline import CreepToRelaxationPipeline

   # Create conversion pipeline
   converter = CreepToRelaxationPipeline(
       method='integration',    # 'integration' or 'approximate'
       regularization=0.01      # Regularization parameter
   )

   # Load creep data and convert
   converter.load('creep_data.txt')
   relaxation_data = converter.convert()

   # Fit model to relaxation data
   converter.fit('maxwell')
   converter.plot(show=True)

   results = converter.get_results()

FrequencyToTimePipeline
~~~~~~~~~~~~~~~~~~~~~~~

Convert frequency-domain data to time-domain:

.. code-block:: python

   from rheojax.pipeline import FrequencyToTimePipeline

   # Create conversion pipeline
   ft_pipeline = FrequencyToTimePipeline(
       method='inverse_fft',     # 'inverse_fft' or 'analytical'
       time_range=(1e-3, 1e3),   # Time range for output (s)
       n_points=200              # Number of time points
   )

   # Load frequency data and convert
   ft_pipeline.load('frequency_sweep.txt')
   time_data = ft_pipeline.convert()

   ft_pipeline.plot(show=True)

Pipeline Builder
----------------

For Complex Custom Workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PipelineBuilder provides programmatic pipeline construction:

.. code-block:: python

   from rheojax.pipeline import PipelineBuilder

   # Build custom pipeline step-by-step
   builder = PipelineBuilder()

   # Add steps
   builder.add_load_step('data.txt', format='auto')
   builder.add_transform_step('smooth', method='savgol', window=11)
   builder.add_transform_step('fft', window='hann')
   builder.add_fit_step('maxwell', initial_params={'G_s': 1e5})
   builder.add_plot_step(show=False, save='result.png')
   builder.add_save_step('result.hdf5')

   # Build and execute
   pipeline = builder.build()
   results = pipeline.execute()

Conditional Steps
~~~~~~~~~~~~~~~~~

Add conditional logic to pipelines:

.. code-block:: python

   builder = PipelineBuilder()

   builder.add_load_step('data.txt')

   # Add conditional transform
   builder.add_conditional_step(
       condition=lambda state: state['data'].metadata.get('noisy', False),
       true_step=('transform', {'name': 'smooth', 'window': 11}),
       false_step=None  # Skip if not noisy
   )

   builder.add_fit_step('maxwell')

   pipeline = builder.build()
   results = pipeline.execute()

Looping Over Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   builder = PipelineBuilder()
   builder.add_load_step('data.txt')

   # Try different models
   models = ['maxwell', 'zener', 'springpot']
   for model_name in models:
       builder.add_fit_step(model_name)
       builder.add_plot_step(save=f'fit_{model_name}.png')

   pipeline = builder.build()
   results = pipeline.execute()

Batch Processing
----------------

Process Multiple Files
~~~~~~~~~~~~~~~~~~~~~~

The BatchPipeline processes multiple datasets with the same workflow:

.. code-block:: python

   from rheojax.pipeline import Pipeline, BatchPipeline

   # Define template pipeline
   template = (Pipeline()
       .transform('smooth', window=11)
       .fit('maxwell')
       .plot(save='${filename}_fit.png')  # ${filename} is replaced
       .save('${filename}_results.hdf5'))

   # Create batch processor
   batch = BatchPipeline(template)

   # Process directory
   batch.process_directory('data/', pattern='*.txt')

   # Or process file list
   batch.process_files([
       'sample1.txt',
       'sample2.txt',
       'sample3.txt'
   ])

   # Get all results
   all_results = batch.get_all_results()

   # Export summary
   batch.export_summary('batch_summary.xlsx')

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Process files in parallel for speed:

.. code-block:: python

   from rheojax.pipeline import BatchPipeline

   batch = BatchPipeline(template, n_jobs=4)  # Use 4 cores

   batch.process_directory('data/', pattern='*.txt')

   # Progress tracking
   batch.process_directory('data/', progress_bar=True)

Batch Analysis Example
~~~~~~~~~~~~~~~~~~~~~~

Complete batch workflow:

.. code-block:: python

   from rheojax.pipeline import Pipeline, BatchPipeline
   import matplotlib.pyplot as plt
   import numpy as np

   # Template: fit Maxwell to all files
   template = (Pipeline()
       .fit('maxwell')
       .plot(save='${filename}_fit.png', show=False))

   # Process all samples
   batch = BatchPipeline(template, n_jobs=4)
   batch.process_directory('rheology_data/', pattern='sample*.txt',
                            progress_bar=True)

   # Extract parameters from all fits
   all_results = batch.get_all_results()

   G_s_values = []
   eta_s_values = []
   sample_names = []

   for filename, results in all_results.items():
       G_s_values.append(results['parameters']['G_s'])
       eta_s_values.append(results['parameters']['eta_s'])
       sample_names.append(filename)

   # Compare parameters across samples
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   x = np.arange(len(sample_names))
   ax1.bar(x, G_s_values)
   ax1.set_xticks(x)
   ax1.set_xticklabels(sample_names, rotation=45, ha='right')
   ax1.set_ylabel('G_s (Pa)')
   ax1.set_title('Rubbery Modulus Comparison')

   ax2.bar(x, eta_s_values)
   ax2.set_xticks(x)
   ax2.set_xticklabels(sample_names, rotation=45, ha='right')
   ax2.set_ylabel('eta_s (Pa*s)')
   ax2.set_title('Viscosity Comparison')

   plt.tight_layout()
   plt.savefig('batch_comparison.png', dpi=300)
   plt.show()

   # Export batch summary
   batch.export_summary('batch_summary.xlsx')

Error Handling
--------------

Try-Catch in Pipelines
~~~~~~~~~~~~~~~~~~~~~~

Pipelines have built-in error handling:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell', fail_on_error=False))  # Don't raise, continue

   # Check for errors
   if pipeline.has_errors():
       errors = pipeline.get_errors()
       print(f"Errors encountered: {errors}")
   else:
       results = pipeline.get_results()

Validation
~~~~~~~~~~

Validate pipeline before execution:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell'))

   # Validate (checks file exists, model available, etc.)
   is_valid, messages = pipeline.validate()

   if is_valid:
       pipeline.execute()
   else:
       print(f"Validation failed: {messages}")

Debugging
~~~~~~~~~

Enable debug output:

.. code-block:: python

   pipeline = (Pipeline(debug=True)  # Enable debug logging
       .load('data.txt')
       .fit('maxwell'))

   # Or set verbosity
   pipeline = (Pipeline(verbose=2)  # 0=silent, 1=info, 2=debug
       .load('data.txt')
       .fit('maxwell'))

   # Inspect pipeline state at any point
   state = pipeline.get_state()
   print(f"Current step: {state['current_step']}")
   print(f"Data loaded: {state['data_loaded']}")
   print(f"Model fitted: {state['model_fitted']}")

Best Practices
--------------

Code Style
~~~~~~~~~~

**Good: Method chaining with line breaks**

.. code-block:: python

   results = (Pipeline()
       .load('data.txt')
       .transform('smooth', window=11)
       .fit('maxwell')
       .plot(show=True)
       .get_results())

**Acceptable: Step-by-step for debugging**

.. code-block:: python

   pipeline = Pipeline()
   pipeline.load('data.txt')
   pipeline.transform('smooth', window=11)
   pipeline.fit('maxwell')
   pipeline.plot(show=True)
   results = pipeline.get_results()

**Avoid: Re-creating pipeline objects**

.. code-block:: python

   # Don't do this
   pipeline1 = Pipeline().load('data.txt')
   pipeline2 = pipeline1.fit('maxwell')  # Creates new pipeline
   pipeline3 = pipeline2.plot(show=True)  # Creates another new pipeline

   # Pipeline methods return self, so this works but is inefficient

Parameter Management
~~~~~~~~~~~~~~~~~~~~

Set reasonable defaults:

.. code-block:: python

   # Good: provide physical bounds
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell',
            initial_params={'G_s': 1e5, 'eta_s': 1e3},
            bounds={'G_s': (1e3, 1e7),
                    'eta_s': (1e1, 1e5)}))

Error Recovery
~~~~~~~~~~~~~~

.. code-block:: python

   # Try multiple models, use first that succeeds
   models = ['maxwell', 'zener', 'fractional_maxwell_gel']

   for model_name in models:
       try:
           results = (Pipeline()
               .load('data.txt')
               .fit(model_name)
               .get_results())
           print(f"Success with {model_name}")
           break
       except Exception as e:
           print(f"{model_name} failed: {e}")
           continue

Documentation
~~~~~~~~~~~~~

Document custom pipelines:

.. code-block:: python

   def analyze_polymer_melt(filename, temperature):
       """Analyze polymer melt rheology data.

       Parameters
       ----------
       filename : str
           Path to frequency sweep data
       temperature : float
           Measurement temperature ( degC)

       Returns
       -------
       dict
           Analysis results with parameters and fit quality
       """
       results = (Pipeline()
           .load(filename)
           .fit('fractional_maxwell_gel',
                initial_params={'alpha': 0.5})
           .plot(save=f'{filename}_T{temperature}C.png')
           .get_results())

       results['temperature'] = temperature
       return results

Performance Tips
----------------

For Large Datasets
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Downsample before fitting for speed
   pipeline = (Pipeline()
       .load('huge_dataset.txt')
       .transform('resample', n_points=200)  # Reduce to 200 points
       .fit('maxwell')
       .plot(show=True))

For Batch Processing
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use parallel processing
   batch = BatchPipeline(template, n_jobs=-1)  # Use all cores

   # Process in chunks to manage memory
   batch.process_directory('data/', chunk_size=10)

Caching
~~~~~~~

.. code-block:: python

   # Cache intermediate results
   pipeline = (Pipeline(cache=True)
       .load('data.txt')
       .transform('fft')  # Cached
       .fit('maxwell'))

   # Re-fit without re-computing FFT
   pipeline.fit('zener')  # Uses cached FFT result

Common Patterns
---------------

Pattern 1: Quick Check
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quick fit and visualize
   Pipeline().load('data.txt').fit('maxwell').plot(show=True)

Pattern 2: Comparison Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import ModelComparisonPipeline

   comparison = (ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
       .load('data.txt')
       .run()
       .plot_comparison(show=True)
       .save('comparison.xlsx'))

Pattern 3: Multi-Temperature Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import MastercurvePipeline

   mc = (MastercurvePipeline(reference_temp=50, method='wlf')
       .run(['25C.txt', '50C.txt', '75C.txt'], [25, 50, 75])
       .fit('fractional_maxwell_gel')
       .plot(show=True)
       .save('mastercurve_analysis.hdf5'))

Pattern 4: Batch with Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline, BatchPipeline

   template = Pipeline().fit('maxwell').plot(save='${filename}_fit.png')
   batch = BatchPipeline(template, n_jobs=4)
   batch.process_directory('data/')
   batch.export_summary('summary.xlsx')

Summary
-------

Pipeline API Checklist:

1. **Basic workflow**: ``load() -> fit() -> plot() -> get_results()``
2. **Add transforms**: ``.transform('smooth') -> .transform('fft')``
3. **Specialized workflows**: Use ``MastercurvePipeline``, ``ModelComparisonPipeline``
4. **Batch processing**: Use ``BatchPipeline`` with template
5. **Custom pipelines**: Use ``PipelineBuilder`` for complex logic
6. **Error handling**: Use ``fail_on_error=False`` and check ``has_errors()``
7. **Performance**: Use ``n_jobs`` for parallel, ``cache=True`` for speed

The Pipeline API provides the fastest path from data to results. For maximum flexibility, combine with the Modular API (see :doc:`/user_guide/modular_api`).

Next Steps
----------

- :doc:`/user_guide/modular_api` - Direct model and transform usage
- :doc:`/user_guide/model_selection` - Choose the right model
- :doc:`/user_guide/transforms` - Understand data transforms
- ``examples/basic/`` - Complete example notebooks
