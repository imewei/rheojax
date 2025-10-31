User Guide
==========

This comprehensive guide covers everything you need to know about using the rheo package for rheological analysis.

Getting Started
---------------

.. toctree::
   :maxdepth: 2

   user_guide/getting_started
   user_guide/core_concepts

Phase 2: Models and Transforms
-------------------------------

.. toctree::
   :maxdepth: 2

   user_guide/model_selection
   user_guide/transforms
   user_guide/pipeline_api
   user_guide/modular_api
   user_guide/multi_technique_fitting

Bayesian Inference (Phase 2+)
------------------------------

.. toctree::
   :maxdepth: 2

   user_guide/bayesian_inference

Data Input/Output
-----------------

.. toctree::
   :maxdepth: 2

   user_guide/io_guide

Visualization
-------------

.. toctree::
   :maxdepth: 2

   user_guide/visualization_guide

Quick Navigation
----------------

**For beginners:**

1. Start with :doc:`user_guide/getting_started` for installation and basic concepts
2. Learn about :doc:`user_guide/pipeline_api` for high-level workflows
3. Explore :doc:`user_guide/model_selection` to choose the right model

**For intermediate users:**

1. Master the :doc:`user_guide/transforms` for data preprocessing
2. Learn :doc:`user_guide/modular_api` for direct model control
3. Understand :doc:`user_guide/io_guide` for various file formats

**For advanced users:**

1. Implement :doc:`user_guide/multi_technique_fitting` for complex analyses
2. Master :doc:`user_guide/bayesian_inference` for MCMC analysis with ArviZ diagnostics
3. Dive into :doc:`user_guide/core_concepts` for architecture details
4. Study :doc:`user_guide/visualization_guide` for publication-quality plots

Package Overview
----------------

What is rheo?
~~~~~~~~~~~~~

rheo is a JAX-powered unified rheological analysis package that combines:

- **20+ rheological models**: Classical, fractional, and non-Newtonian flow models
- **5 data transforms**: FFT analysis, mastercurves, mutation number, OWChirp, smooth derivatives
- **Dual API levels**: High-level Pipeline API and low-level Modular API
- **GPU acceleration**: Automatic GPU/CPU dispatch with JAX
- **Multi-technique fitting**: Shared parameters across experimental techniques

Key Features
~~~~~~~~~~~~

**Models**

- 3 classical models (Maxwell, Zener, SpringPot)
- 11 fractional models (Maxwell, Kelvin-Voigt, Zener families)
- 6 non-Newtonian flow models (Power Law, Carreau, Herschel-Bulkley, etc.)

**Transforms**

- FFT Analysis: Time → frequency domain conversion
- Mastercurve: Time-temperature superposition with WLF/Arrhenius
- Mutation Number: Quantify viscoelastic character evolution
- OWChirp: Optimal waveform analysis for LAOS
- Smooth Derivative: Noise-robust differentiation

**APIs**

- Pipeline API: Fluent method chaining for rapid analysis
- Modular API: Direct model/transform access for custom workflows
- Batch processing: Process multiple datasets efficiently

**Performance**

- JAX-first architecture: 10-100x speedups over NumPy
- Automatic differentiation: Fast gradient-based optimization
- GPU acceleration: Transparent CPU/GPU execution
- JIT compilation: Optimized performance for repeated operations

Typical Workflow
~~~~~~~~~~~~~~~~

1. **Load data** from various formats (TRIOS, CSV, Excel)
2. **Apply transforms** for preprocessing (smoothing, FFT, mastercurves)
3. **Fit models** to characterize material behavior
4. **Visualize** with publication-quality plots
5. **Export** results for further analysis or reporting

.. code-block:: python

   from rheo.pipeline import Pipeline

   # Complete analysis in one chain
   results = (Pipeline()
       .load('data.txt')                    # Auto-detect format
       .transform('smooth', window=11)      # Smooth noisy data
       .transform('fft', window='hann')     # Time → frequency
       .fit('fractional_maxwell_gel')       # Fit model
       .plot(show=True, save='fit.png')     # Visualize
       .save('results.hdf5')                # Export
       .get_results())                      # Retrieve results

   print(f"R² = {results['r2']:.4f}")
   print(f"Parameters: {results['parameters']}")

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

This user guide is organized as follows:

**Getting Started (Phase 1)**

- :doc:`user_guide/getting_started`: Installation, quickstart, basic examples
- :doc:`user_guide/core_concepts`: RheoData, Parameters, Test Modes, Registry
- :doc:`user_guide/io_guide`: Reading/writing various file formats
- :doc:`user_guide/visualization_guide`: Creating publication-quality plots

**Phase 2: Models and Analysis**

- :doc:`user_guide/model_selection`: Decision tree for choosing models
- :doc:`user_guide/transforms`: Data preprocessing and analysis transforms
- :doc:`user_guide/pipeline_api`: High-level workflow API (recommended for most users)
- :doc:`user_guide/modular_api`: Low-level API for maximum control
- :doc:`user_guide/multi_technique_fitting`: Combining multiple experimental techniques

**Bayesian Inference (Phase 2+)**

- :doc:`user_guide/bayesian_inference`: NLSQ → NUTS workflow, ArviZ diagnostics, credible intervals

**Additional Resources**

- :doc:`api_reference`: Complete API documentation
- :doc:`developer/contributing`: Contributing guide for developers
- :doc:`examples/index`: 20 tutorial notebooks (basic, transforms, bayesian, advanced)

Support and Community
~~~~~~~~~~~~~~~~~~~~~

- **Documentation**: https://rheo.readthedocs.io
- **Issues**: https://github.com/username/rheo/issues
- **Discussions**: https://github.com/username/rheo/discussions
- **Email**: rheo@example.com

Version Information
~~~~~~~~~~~~~~~~~~~

This documentation covers rheo version 0.1.0 (Initial Development - Unreleased).

Current features include:
- Core infrastructure, test mode detection, I/O, visualization
- 20 rheological models, 5 transforms, Pipeline API, multi-technique fitting
- Bayesian inference (NLSQ → NUTS), ArviZ diagnostics, comprehensive MCMC
- GPU-accelerated optimization with JAX and NLSQ

**Note**: This is pre-release software under active development.

Next Steps
----------

**New users**: Start with :doc:`user_guide/getting_started`

**Experienced users**: Jump to :doc:`user_guide/model_selection` or :doc:`user_guide/pipeline_api`

**Advanced users**: Explore :doc:`user_guide/modular_api` and :doc:`user_guide/multi_technique_fitting`

**Developers**: See :doc:`developer/contributing` for contribution guidelines
