API Reference
=============

This page contains the complete API reference for all public modules, classes, and functions in rheo.

.. toctree::
   :maxdepth: 2
   :caption: API Modules

   api/core
   api/models
   api/transforms
   api/pipeline
   api/io
   api/utils
   api/visualization

Quick Links
-----------

**Phase 1 (Core Infrastructure)**:

- :doc:`api/core` - Base classes, data structures, parameters, registry
- :doc:`api/io` - File readers and writers (TRIOS, CSV, Excel, HDF5)
- :doc:`api/utils` - Utilities (optimization, Mittag-Leffler functions)
- :doc:`api/visualization` - Plotting functions and templates

**Phase 2 (Models and Transforms)**:

- :doc:`api/models` - All 20 rheological models (classical, fractional, flow)
- :doc:`api/transforms` - All 5 data transforms (FFT, mastercurve, mutation, OWChirp, smoothing)
- :doc:`api/pipeline` - Pipeline API for high-level workflows

Overview by Category
--------------------

Core Components
~~~~~~~~~~~~~~~

**Base Classes**:

- :class:`rheo.core.base.BaseModel` - Base class for all models
- :class:`rheo.core.base.BaseTransform` - Base class for all transforms

**Data Structures**:

- :class:`rheo.core.data.RheoData` - Primary data container
- :class:`rheo.core.parameters.Parameter` - Single parameter
- :class:`rheo.core.parameters.ParameterSet` - Parameter collection
- :class:`rheo.core.parameters.SharedParameterSet` - Multi-technique shared parameters

**Registries**:

- :class:`rheo.core.registry.ModelRegistry` - Model discovery and instantiation
- :class:`rheo.core.registry.TransformRegistry` - Transform discovery and instantiation

**Test Modes**:

- :class:`rheo.core.test_modes.TestMode` - Test mode enumeration
- :func:`rheo.core.test_modes.detect_test_mode` - Automatic test mode detection

Models (20 total)
~~~~~~~~~~~~~~~~~~

**Classical Models (3)**:

- :class:`rheo.models.Maxwell` - Spring and dashpot in series
- :class:`rheo.models.Zener` - Standard Linear Solid (SLS)
- :class:`rheo.models.SpringPot` - Fractional power-law element

**Fractional Maxwell Family (4)**:

- :class:`rheo.models.FractionalMaxwellGel` - Spring + SpringPot
- :class:`rheo.models.FractionalMaxwellLiquid` - SpringPot + dashpot
- :class:`rheo.models.FractionalMaxwellModel` - Two SpringPots
- :class:`rheo.models.FractionalKelvinVoigt` - Parallel spring + SpringPot

**Fractional Zener Family (4)**:

- :class:`rheo.models.FractionalZenerSolidLiquid` (FZSL) - Solid + fractional liquid
- :class:`rheo.models.FractionalZenerSolidSolid` (FZSS) - Two solids + fractional
- :class:`rheo.models.FractionalZenerLiquidLiquid` (FZLL) - Most general
- :class:`rheo.models.FractionalKelvinVoigtZener` (FKVZ) - FKV + series spring

**Advanced Fractional Models (3)**:

- :class:`rheo.models.FractionalBurgersModel` (FBM) - Maxwell + FKV in series
- :class:`rheo.models.FractionalPoyntingThomson` (FPT) - FKV + spring in series
- :class:`rheo.models.FractionalJeffreysModel` (FJM) - Two dashpots + SpringPot

**Non-Newtonian Flow Models (6)**:

- :class:`rheo.models.PowerLaw` - τ = K·γ̇^n
- :class:`rheo.models.Carreau` - Smooth Newtonian → power-law
- :class:`rheo.models.CarreauYasuda` - Extended Carreau
- :class:`rheo.models.Cross` - Alternative to Carreau
- :class:`rheo.models.HerschelBulkley` - Yield stress + power-law
- :class:`rheo.models.Bingham` - Yield stress + Newtonian

Transforms (5 total)
~~~~~~~~~~~~~~~~~~~~~

- :class:`rheo.transforms.FFTAnalysis` - Time → frequency domain (FFT)
- :class:`rheo.transforms.Mastercurve` - Time-temperature superposition (WLF/Arrhenius)
- :class:`rheo.transforms.MutationNumber` - Viscoelastic character quantification
- :class:`rheo.transforms.OWChirp` - Optimal waveform analysis for LAOS
- :class:`rheo.transforms.SmoothDerivative` - Noise-robust differentiation

Pipeline API
~~~~~~~~~~~~

**Core Pipeline**:

- :class:`rheo.pipeline.Pipeline` - Base fluent API with method chaining

**Specialized Workflows**:

- :class:`rheo.pipeline.MastercurvePipeline` - Time-temperature superposition
- :class:`rheo.pipeline.ModelComparisonPipeline` - Multi-model comparison
- :class:`rheo.pipeline.CreepToRelaxationPipeline` - Creep → relaxation conversion
- :class:`rheo.pipeline.FrequencyToTimePipeline` - Frequency → time conversion

**Batch Processing**:

- :class:`rheo.pipeline.PipelineBuilder` - Programmatic pipeline construction
- :class:`rheo.pipeline.BatchPipeline` - Multi-file parallel processing

I/O (Input/Output)
~~~~~~~~~~~~~~~~~~

**Readers**:

- :func:`rheo.io.auto_load` - Auto-detect format and load
- :func:`rheo.io.load_trios` - TA TRIOS text files
- :func:`rheo.io.load_csv` - CSV files
- :func:`rheo.io.load_excel` - Excel files (.xlsx, .xls)
- :func:`rheo.io.load_anton_paar` - Anton Paar files

**Writers**:

- :func:`rheo.io.save_hdf5` - HDF5 format (full fidelity)
- :func:`rheo.io.save_excel` - Excel reports (multi-sheet)

Utilities
~~~~~~~~~

**Optimization**:

- :func:`rheo.utils.optimization.nlsq_optimize` - Nonlinear least squares with JAX
- :func:`rheo.utils.optimization.calculate_confidence_intervals` - Parameter uncertainty

**Mittag-Leffler Functions**:

- :func:`rheo.utils.mittag_leffler.mittag_leffler_e` - One-parameter: E_α(z)
- :func:`rheo.utils.mittag_leffler.mittag_leffler_e2` - Two-parameter: E_α,β(z)

**Test Modes**:

- :func:`rheo.core.test_modes.detect_test_mode` - Auto-detect test mode
- :func:`rheo.core.test_modes.get_compatible_test_modes` - Get compatible modes for model
- :func:`rheo.core.test_modes.suggest_models_for_test_mode` - Suggest models for test mode

Visualization
~~~~~~~~~~~~~

**Plotting Functions**:

- :func:`rheo.visualization.plot_rheo_data` - Plot RheoData objects
- :func:`rheo.visualization.plot_model_fit` - Plot data with model fit
- :func:`rheo.visualization.plot_residuals` - Plot fit residuals
- :func:`rheo.visualization.plot_mastercurve` - Plot mastercurve with shift factors

**Styles**:

- 'default': Standard matplotlib style
- 'publication': High-quality for publications
- 'presentation': Large fonts and markers for presentations

API Design Principles
---------------------

Consistency
~~~~~~~~~~~

All rheo classes follow consistent patterns:

**Models**:

.. code-block:: python

   model = ModelClass()              # Instantiate
   model.fit(X, y)                   # Fit to data
   predictions = model.predict(X)    # Make predictions
   score = model.score(X, y)         # Calculate R²

**Transforms**:

.. code-block:: python

   transform = TransformClass()      # Instantiate
   result = transform.transform(data)           # Apply transform
   result = transform.fit_transform(data)       # Fit and transform
   original = transform.inverse_transform(result)  # Inverse (if available)

**Pipelines**:

.. code-block:: python

   pipeline = (Pipeline()
       .load(source)
       .transform(name, **params)
       .fit(model, **kwargs)
       .plot(**options)
       .save(filepath))

scikit-learn Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~

rheo models are compatible with scikit-learn:

- ``fit(X, y)``, ``predict(X)``, ``score(X, y)`` methods
- ``get_params()``, ``set_params(**params)`` for hyperparameter access
- Can be used in scikit-learn pipelines and GridSearchCV

JAX Integration
~~~~~~~~~~~~~~~

All numerical operations use JAX:

- Automatic differentiation for optimization
- JIT compilation for performance
- GPU/CPU automatic dispatch
- Compatible with JAX transformations (vmap, pmap, etc.)

Type Hints
~~~~~~~~~~

All public APIs include type hints:

.. code-block:: python

   def fit(self, X: ArrayLike, y: ArrayLike, **kwargs) -> Self:
       """Fit model to data."""
       ...

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

All classes and functions include:

- Docstrings with description
- Parameters with types and descriptions
- Returns with type and description
- Examples showing typical usage
- Cross-references to related functions

Getting Started
---------------

For new users, we recommend:

1. **Read the User Guide**: :doc:`user_guide/getting_started`
2. **Try the Pipeline API**: :doc:`api/pipeline` for high-level workflows
3. **Explore Models**: :doc:`api/models` to understand available models
4. **Learn Transforms**: :doc:`api/transforms` for data preprocessing

For advanced users:

1. **Master the Modular API**: :doc:`user_guide/modular_api`
2. **Understand Core Classes**: :doc:`api/core`
3. **Implement Custom Workflows**: :doc:`user_guide/multi_technique_fitting`

Version Information
-------------------

This documentation covers rheo version 0.2.0 (Phase 2 release).

**Phase 1** (v0.1.0): Core infrastructure, test mode detection, I/O, visualization

**Phase 2** (v0.2.0): 20 models, 5 transforms, Pipeline API, multi-technique fitting

**Phase 3** (planned): Bayesian inference, ML model selection, PDF reports

Support
-------

- **Documentation**: https://rheo.readthedocs.io
- **GitHub Issues**: https://github.com/username/rheo/issues
- **Discussions**: https://github.com/username/rheo/discussions
- **Email**: rheo@example.com
