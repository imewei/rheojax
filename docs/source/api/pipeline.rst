Pipeline API
============

This page documents the high-level Pipeline API for rheological analysis workflows.

Overview
--------

The Pipeline API provides a fluent interface for chaining operations from data loading through model fitting and export. It's designed for rapid analysis with minimal boilerplate code.

**Core Components**:

1. **Pipeline**: Base fluent API with method chaining
2. **Specialized Workflows**: Pre-configured pipelines for common tasks
3. **PipelineBuilder**: Programmatic pipeline construction
4. **BatchPipeline**: Process multiple datasets

Basic Pipeline
--------------

.. autoclass:: rheojax.pipeline.Pipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Core pipeline class providing fluent method chaining for rheological analysis workflows.

**Example - Basic Usage**:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Create pipeline and chain operations
   results = (Pipeline()
       .load('data.txt')                   # Load data
       .transform('smooth', window=11)     # Smooth noisy data
       .fit('maxwell')                     # Fit model
       .plot(show=True)                    # Visualize
       .get_results())                     # Retrieve results

   print(f"R² = {results['r2']:.4f}")
   print(f"Parameters: {results['parameters']}")

**Key Methods**:

``load(source, format='auto', **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load data from file or RheoData object.

**Parameters**:

- ``source`` (str or RheoData): File path or data object
- ``format`` (str): File format - 'auto', 'trios', 'csv', 'excel'
- ``**kwargs``: Format-specific arguments

**Returns**: self (for chaining)

**Example**:

.. code-block:: python

   # Auto-detect format
   pipeline = Pipeline().load('data.txt')

   # Explicit format
   pipeline = Pipeline().load('data.csv', format='csv',
                               x_col='frequency', y_col='modulus')

   # From RheoData object
   from rheojax.core import RheoData
   data = RheoData(x=freq, y=modulus, ...)
   pipeline = Pipeline().load(data)

``transform(name, **params)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply data transform.

**Parameters**:

- ``name`` (str): Transform name - 'smooth', 'fft', 'mastercurve', etc.
- ``**params``: Transform-specific parameters

**Returns**: self (for chaining)

**Example**:

.. code-block:: python

   # Single transform
   pipeline = (Pipeline()
       .load('data.txt')
       .transform('smooth', method='savgol', window=11))

   # Multiple transforms (chained)
   pipeline = (Pipeline()
       .load('data.txt')
       .transform('smooth', window=11)
       .transform('fft', window='hann'))

``fit(model, initial_params=None, bounds=None, **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fit rheological model to data.

**Parameters**:

- ``model`` (str or BaseModel): Model name or instance
- ``initial_params`` (dict, optional): Initial parameter values
- ``bounds`` (dict, optional): Parameter bounds
- ``**kwargs``: Optimization options

**Returns**: self (for chaining)

**Example**:

.. code-block:: python

   # By name
   pipeline = Pipeline().load('data.txt').fit('maxwell')

   # With initial parameters
   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell',
            initial_params={'G_s': 1e5, 'eta_s': 1e3},
            bounds={'G_s': (1e3, 1e7), 'eta_s': (1e1, 1e5)}))

   # Multiple models (comparison)
   pipeline = (Pipeline()
       .load('data.txt')
       .fit(['maxwell', 'zener', 'springpot']))

``plot(show=False, save=None, style='default', **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visualize data and model fit.

**Parameters**:

- ``show`` (bool): Display interactive plot
- ``save`` (str, optional): Save to file
- ``style`` (str): Plot style - 'default', 'publication', 'presentation'
- ``**kwargs``: Plotting options

**Returns**: self (for chaining)

**Example**:

.. code-block:: python

   # Show plot
   pipeline.plot(show=True)

   # Save to file
   pipeline.plot(save='fit_result.png', dpi=300)

   # Custom style
   pipeline.plot(show=True, style='publication',
                 include_residuals=True, title='Maxwell Fit')

``save(filepath, format='hdf5', **kwargs)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Export results to file.

**Parameters**:

- ``filepath`` (str): Output file path
- ``format`` (str): Format - 'hdf5', 'excel', 'csv'
- ``**kwargs``: Format-specific options

**Returns**: self (for chaining)

**Example**:

.. code-block:: python

   # HDF5 (full fidelity)
   pipeline.save('results.hdf5')

   # Excel report
   pipeline.save('report.xlsx', format='excel', include_plots=True)

``get_results()``
~~~~~~~~~~~~~~~~~

Retrieve analysis results as dictionary.

**Returns**: dict with keys:

- ``'parameters'``: Fitted parameter values
- ``'r2'``: R² score
- ``'rmse'``: Root mean squared error
- ``'predictions'``: Model predictions
- ``'residuals'``: Fit residuals
- ``'data'``: Original RheoData object
- ``'model'``: Fitted model instance

**Example**:

.. code-block:: python

   results = pipeline.get_results()

   print(f"R² = {results['r2']:.4f}")
   print(f"RMSE = {results['rmse']:.2e}")
   print(f"Parameters:")
   for name, value in results['parameters'].items():
       print(f"  {name} = {value:.4e}")

Specialized Workflows
---------------------

MastercurvePipeline
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.pipeline.MastercurvePipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Pre-configured pipeline for time-temperature superposition analysis.

**Example**:

.. code-block:: python

   from rheojax.pipeline import MastercurvePipeline

   # Create mastercurve pipeline
   mc_pipeline = MastercurvePipeline(
       reference_temp=50,      # Reference temperature (°C)
       method='wlf',           # 'wlf' or 'arrhenius'
       optimize=True           # Optimize WLF/Arrhenius parameters
   )

   # Load and process multi-temperature data
   files = ['data_25C.txt', 'data_40C.txt', 'data_55C.txt', 'data_70C.txt']
   temperatures = [25, 40, 55, 70]

   results = mc_pipeline.run(files, temperatures)

   # Access mastercurve results
   mastercurve = results['mastercurve']
   shift_factors = results['shift_factors']
   wlf_params = results['wlf_parameters']

   print(f"WLF C1 = {wlf_params['C1']:.2f}")
   print(f"WLF C2 = {wlf_params['C2']:.2f} K")

   # Fit model to mastercurve
   mc_pipeline.fit('fractional_maxwell_gel')
   mc_pipeline.plot(show=True, style='publication')

**Key Methods**:

- ``run(files, temperatures)``: Create mastercurve from files
- ``fit(model)``: Fit model to mastercurve
- ``get_shift_factors()``: Get temperature shift factors
- ``get_wlf_parameters()``: Get fitted WLF C1, C2

ModelComparisonPipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.pipeline.ModelComparisonPipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Systematically compare multiple models on the same dataset.

**Example**:

.. code-block:: python

   from rheojax.pipeline import ModelComparisonPipeline

   # Models to compare
   models = ['maxwell', 'zener', 'fractional_maxwell_gel',
             'fractional_kelvin_voigt', 'springpot']

   # Create comparison pipeline
   comparison = ModelComparisonPipeline(models)

   # Load data and run comparison
   comparison.load('data.txt')
   comparison.run()

   # Get comparison table
   results = comparison.get_results()
   comparison_table = results['comparison']

   # Print comparison
   print("\\nModel Comparison:")
   print(f"{'Model':<30} {'R²':<10} {'RMSE':<12} {'AIC':<12}")
   print("-" * 70)
   for row in comparison_table:
       print(f"{row['model']:<30} {row['r2']:<10.4f} "
             f"{row['rmse']:<12.2e} {row['aic']:<12.1f}")

   # Get best model
   best = comparison.get_best_model(criterion='aic')  # 'aic', 'bic', 'r2'
   print(f"\\nBest model (AIC): {best['name']}")

   # Visualize comparison
   comparison.plot_comparison(show=True)
   comparison.plot_ranking(criterion='aic', show=True)

**Key Methods**:

- ``run()``: Fit all models
- ``get_best_model(criterion)``: Select best by AIC, BIC, or R²
- ``plot_comparison()``: Multi-panel plot of all models
- ``plot_ranking()``: Bar chart ranking by criterion

CreepToRelaxationPipeline
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.pipeline.CreepToRelaxationPipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Convert creep compliance J(t) to relaxation modulus G(t).

**Example**:

.. code-block:: python

   from rheojax.pipeline import CreepToRelaxationPipeline

   converter = CreepToRelaxationPipeline(
       method='integration',    # 'integration' or 'approximate'
       regularization=0.01      # Regularization parameter
   )

   converter.load('creep_data.txt')
   relaxation_data = converter.convert()

   # Fit model to relaxation data
   converter.fit('maxwell')
   converter.plot(show=True)

FrequencyToTimePipeline
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.pipeline.FrequencyToTimePipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Convert frequency-domain data to time-domain via inverse FFT.

**Example**:

.. code-block:: python

   from rheojax.pipeline import FrequencyToTimePipeline

   ft_pipeline = FrequencyToTimePipeline(
       method='inverse_fft',     # 'inverse_fft' or 'analytical'
       time_range=(1e-3, 1e3),   # Time range (s)
       n_points=200              # Number of time points
   )

   ft_pipeline.load('frequency_sweep.txt')
   time_data = ft_pipeline.convert()
   ft_pipeline.plot(show=True)

BayesianPipeline
~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.pipeline.bayesian.BayesianPipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Specialized pipeline for Bayesian rheological analysis with NLSQ → NUTS workflow.

**Key Features**:

- NLSQ optimization for fast point estimation
- Automatic warm-start for NumPyro NUTS sampling
- Comprehensive ArviZ diagnostics (6 plot types)
- Fluent API for method chaining
- Convergence monitoring (R-hat, ESS, divergences)

**Example - Complete Bayesian Workflow**:

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline

   # Create and execute pipeline
   pipeline = (BayesianPipeline()
       .load('data.csv', x_col='time', y_col='stress')
       .fit_nlsq('maxwell')                    # Fast point estimate
       .fit_bayesian(num_samples=2000,         # NUTS with warm-start
                     num_warmup=1000)
       .plot_posterior()                       # Posterior distributions
       .plot_trace()                           # MCMC trace plots
       .save('results.hdf5'))                  # Export results

   # Access results
   summary = pipeline.get_posterior_summary()
   diagnostics = pipeline.get_diagnostics()
   intervals = pipeline.get_credible_intervals()

**Example - ArviZ Diagnostic Suite**:

.. code-block:: python

   # Comprehensive MCMC quality assessment
   (pipeline
       .plot_pair(divergences=True)        # Parameter correlations with divergences
       .plot_forest(hdi_prob=0.95)         # Credible intervals comparison
       .plot_energy()                       # NUTS energy diagnostic
       .plot_autocorr()                     # Mixing diagnostic
       .plot_rank()                         # Convergence diagnostic
       .plot_ess(kind='local'))            # Effective sample size

   # Convert to ArviZ InferenceData for advanced analysis
   idata = pipeline._bayesian_result.to_inference_data()
   import arviz as az
   az.summary(idata)

**Key Methods**:

- ``fit_nlsq(model_name, **kwargs)``: NLSQ optimization for point estimation
- ``fit_bayesian(num_samples, num_warmup, **kwargs)``: NumPyro NUTS sampling with warm-start
- ``plot_posterior(**kwargs)``: Plot posterior distributions
- ``plot_trace(**kwargs)``: Plot MCMC trace diagnostics
- ``plot_pair(**kwargs)``: Plot parameter correlations (ArviZ)
- ``plot_forest(**kwargs)``: Plot credible intervals (ArviZ)
- ``plot_energy(**kwargs)``: Plot NUTS energy diagnostic (ArviZ)
- ``plot_autocorr(**kwargs)``: Plot autocorrelation (ArviZ)
- ``plot_rank(**kwargs)``: Plot rank diagnostic (ArviZ)
- ``plot_ess(**kwargs)``: Plot effective sample size (ArviZ)
- ``get_posterior_summary()``: Get posterior summary statistics
- ``get_diagnostics()``: Get convergence diagnostics (R-hat, ESS)
- ``get_credible_intervals(credibility=0.95)``: Get credible intervals

Pipeline Builder
----------------

.. autoclass:: rheojax.pipeline.PipelineBuilder
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Programmatic pipeline construction for complex custom workflows.

**Example - Basic Builder**:

.. code-block:: python

   from rheojax.pipeline import PipelineBuilder

   # Build custom pipeline
   builder = PipelineBuilder()

   builder.add_load_step('data.txt', format='auto')
   builder.add_transform_step('smooth', method='savgol', window=11)
   builder.add_transform_step('fft', window='hann')
   builder.add_fit_step('maxwell', initial_params={'G_s': 1e5})
   builder.add_plot_step(show=False, save='result.png')
   builder.add_save_step('result.hdf5')

   # Build and execute
   pipeline = builder.build()
   results = pipeline.execute()

**Example - Conditional Logic**:

.. code-block:: python

   builder = PipelineBuilder()

   builder.add_load_step('data.txt')

   # Conditional transform
   builder.add_conditional_step(
       condition=lambda state: state['data'].metadata.get('noisy', False),
       true_step=('transform', {'name': 'smooth', 'window': 11}),
       false_step=None  # Skip if not noisy
   )

   builder.add_fit_step('maxwell')

   pipeline = builder.build()
   results = pipeline.execute()

**Key Methods**:

- ``add_load_step(source, **kwargs)``: Add data loading step
- ``add_transform_step(name, **params)``: Add transform step
- ``add_fit_step(model, **kwargs)``: Add model fitting step
- ``add_plot_step(**kwargs)``: Add visualization step
- ``add_save_step(filepath, **kwargs)``: Add export step
- ``add_conditional_step(condition, true_step, false_step)``: Add conditional logic
- ``build()``: Build pipeline
- ``execute()``: Execute built pipeline

Batch Processing
----------------

.. autoclass:: rheojax.pipeline.BatchPipeline
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Process multiple datasets with the same workflow in parallel.

**Example - Basic Batch**:

.. code-block:: python

   from rheojax.pipeline import Pipeline, BatchPipeline

   # Define template pipeline
   template = (Pipeline()
       .transform('smooth', window=11)
       .fit('maxwell')
       .plot(save='${filename}_fit.png')  # ${filename} replaced per file
       .save('${filename}_results.hdf5'))

   # Create batch processor
   batch = BatchPipeline(template)

   # Process directory
   batch.process_directory('data/', pattern='*.txt')

   # Get all results
   all_results = batch.get_all_results()

   # Export summary
   batch.export_summary('batch_summary.xlsx')

**Example - Parallel Processing**:

.. code-block:: python

   # Use multiple cores
   batch = BatchPipeline(template, n_jobs=4)  # 4 parallel workers

   # Process with progress bar
   batch.process_directory('data/', pattern='*.txt',
                            progress_bar=True)

   # Process specific files
   files = ['sample1.txt', 'sample2.txt', 'sample3.txt']
   batch.process_files(files)

**Key Methods**:

- ``process_directory(path, pattern)``: Process all matching files in directory
- ``process_files(file_list)``: Process specific files
- ``get_all_results()``: Retrieve results from all files
- ``export_summary(filepath)``: Export comparison table
- ``get_failed_files()``: Get list of failed processing attempts

**Parameters**:

- ``template`` (Pipeline): Template pipeline to apply
- ``n_jobs`` (int): Number of parallel workers (-1 = all cores)
- ``fail_on_error`` (bool): Raise exception on first error (default: False)
- ``progress_bar`` (bool): Show progress bar (default: False)

Error Handling
--------------

Pipeline Error Management
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell', fail_on_error=False))  # Don't raise exception

   # Check for errors
   if pipeline.has_errors():
       errors = pipeline.get_errors()
       print(f"Errors encountered: {errors}")
   else:
       results = pipeline.get_results()

Pipeline Validation
~~~~~~~~~~~~~~~~~~~

Validate before execution:

.. code-block:: python

   pipeline = (Pipeline()
       .load('data.txt')
       .fit('maxwell'))

   # Validate pipeline
   is_valid, messages = pipeline.validate()

   if is_valid:
       results = pipeline.execute()
   else:
       print(f"Validation failed: {messages}")

Debug Mode
~~~~~~~~~~

Enable debugging output:

.. code-block:: python

   # Enable debug logging
   pipeline = Pipeline(debug=True)

   # Or set verbosity
   pipeline = Pipeline(verbose=2)  # 0=silent, 1=info, 2=debug

   # Inspect pipeline state
   state = pipeline.get_state()
   print(f"Current step: {state['current_step']}")
   print(f"Data loaded: {state['data_loaded']}")
   print(f"Model fitted: {state['model_fitted']}")

Best Practices
--------------

Method Chaining Style
~~~~~~~~~~~~~~~~~~~~~

**Recommended** (readable, clean):

.. code-block:: python

   results = (Pipeline()
       .load('data.txt')
       .transform('smooth', window=11)
       .fit('maxwell')
       .plot(show=True)
       .get_results())

**Acceptable** (for debugging):

.. code-block:: python

   pipeline = Pipeline()
   pipeline.load('data.txt')
   pipeline.transform('smooth', window=11)
   pipeline.fit('maxwell')
   pipeline.plot(show=True)
   results = pipeline.get_results()

Error Recovery
~~~~~~~~~~~~~~

.. code-block:: python

   # Try multiple models until one succeeds
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

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Cache intermediate results
   pipeline = Pipeline(cache=True)

   # Process in chunks for large batches
   batch = BatchPipeline(template, n_jobs=-1)
   batch.process_directory('data/', chunk_size=10)

See Also
--------

- :doc:`/user_guide/pipeline_api` - Comprehensive pipeline tutorial
- :doc:`/user_guide/modular_api` - Low-level API for custom control
- :doc:`/api/models` - Model API reference
- :doc:`/api/transforms` - Transform API reference
- :class:`rheojax.core.base.BaseModel` - Base model class
- :class:`rheojax.core.base.BaseTransform` - Base transform class
