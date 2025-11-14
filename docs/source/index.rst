.. rheo documentation master file

Welcome to rheo's documentation!
================================

**rheo** is a unified, JAX-accelerated rheological analysis package that provides a modern,
high-performance framework for analyzing experimental rheology data. Built on JAX for automatic
differentiation and GPU acceleration, rheo combines powerful numerical capabilities with an
intuitive API for seamless end-to-end analysis.

.. image:: https://img.shields.io/badge/python-3.12+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.12+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: MIT License

Key Features
------------

**JAX-First Architecture**
   All numerical operations use JAX for automatic CPU/GPU dispatch, providing 10-100x speedups
   with automatic differentiation for optimization

**Comprehensive Data Support**
   Automatic test mode detection (relaxation, creep, oscillation, rotation) with support for
   multiple file formats (TRIOS, CSV, Excel, Anton Paar)

**Flexible Parameter System**
   Type-safe parameter management with bounds, constraints, and optimization support

**Publication-Quality Visualization**
   Matplotlib-based plots with three built-in styles (default, publication, presentation)

**Extensible Design**
   Plugin system for custom models and transforms with registry-based discovery (Phase 2)

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install rheojax

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.io.readers import auto_read
   from rheojax.visualization import plot_rheo_data
   import matplotlib.pyplot as plt

   # Load data (auto-detect format)
   data = auto_read("stress_relaxation.txt")

   # Check detected test mode
   print(f"Test mode: {data.test_mode}")  # relaxation

   # Visualize
   fig, ax = plot_rheo_data(data, style='publication')
   plt.show()

Working with Parameters
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core import ParameterSet

   # Create parameter set
   params = ParameterSet()
   params.add("E", value=1000.0, bounds=(100, 10000), units="Pa")
   params.add("tau", value=1.0, bounds=(0.1, 100), units="s")

   # Get/set values
   E = params.get_value("E")
   params.set_value("tau", 2.5)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   user_guide/01_fundamentals/index
   user_guide/02_model_usage/index
   user_guide/03_advanced_topics/index
   user_guide/04_practical_guides/index
   user_guide/05_appendices/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference
   api/core
   api/models
   api/transforms
   api/pipeline
   api/utils
   api/io
   api/visualization

.. toctree::
   :maxdepth: 2
   :caption: Handbooks

   models/index
   transforms/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorial Notebooks

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Documentation

   developer/architecture
   developer/contributing

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   installation
   quickstart
   contributing

Development Status
------------------

**Phase 1 (Complete)**: Core Infrastructure [done]
   - [done] Base abstractions (BaseModel, BaseTransform, RheoData, Parameters)
   - [done] Test mode detection system
   - [done] Mittag-Leffler functions for fractional calculus
   - [done] Optimization integration with JAX gradients (NLSQ 5-270x speedup)
   - [done] Model and Transform registry system
   - [done] File I/O (TRIOS, CSV, Excel, Anton Paar readers; HDF5, Excel writers)
   - [done] Visualization with matplotlib

**Phase 2 (Complete)**: Models and Transforms [done]
   - [done] 20 rheological models (Maxwell, Zener, 11 fractional variants, 6 flow models)
   - [done] 5 data transforms (FFT, Mastercurve/TTS, Mutation Number, OWChirp/LAOS, Smooth Derivative)
   - [done] Pipeline API for fluent workflows
   - [done] 20 tutorial notebooks (basic, transforms, bayesian, advanced)

**Phase 3 (Complete)**: Bayesian Inference [done]
   - [done] NumPyro NUTS sampling with NLSQ warm-start (2-5x faster convergence)
   - [done] Uncertainty quantification via credible intervals and posterior distributions
   - [done] ArviZ integration (6 diagnostic plot types: pair, forest, energy, autocorr, rank, ESS)
   - [done] Model comparison (WAIC/LOO)
   - [done] BayesianMixin: All 20 models support Bayesian inference
   - [done] BayesianPipeline with fluent API for NLSQ -> NUTS workflows
   - [done] 5 comprehensive Bayesian inference tutorial notebooks

Technology Stack
----------------

**Core Dependencies**
   - Python 3.12+
   - JAX 0.8.0 for acceleration and automatic differentiation
   - NLSQ 0.1.6+ for GPU-accelerated optimization
   - NumPyro for Bayesian inference (MCMC NUTS sampling)
   - ArviZ 0.15.0+ for Bayesian visualization and diagnostics
   - NumPy, SciPy for numerical operations
   - Matplotlib for visualization
   - h5py, pandas, openpyxl for I/O

**Optional Dependencies**
   - CUDA 12.1-12.9 for GPU acceleration (Linux only)
   - piblin for enhanced data management

Performance
-----------

JAX provides significant performance improvements:

.. list-table:: Performance Benchmarks
   :header-rows: 1
   :widths: 40 20 20 20

   * - Operation
     - NumPy Time
     - JAX Time
     - Speedup
   * - Mittag-Leffler (1000 points)
     - 45 ms
     - 0.8 ms
     - 56x
   * - Parameter optimization
     - 2.5 s
     - 0.15 s
     - 17x
   * - Data resampling (10k points)
     - 120 ms
     - 3 ms
     - 40x

Community and Support
---------------------

**Getting Help**
   - Read the :doc:`user_guide/getting_started`
   - Join `GitHub Discussions <https://github.com/username/rheo/discussions>`_
   - Report issues on `GitHub <https://github.com/username/rheo/issues>`_

**Contributing**
   - See :doc:`developer/contributing` for guidelines
   - Check :doc:`developer/architecture` for design principles
   - Review open issues for contribution ideas

**Citation**
   If you use rheo in your research, please cite:

   .. code-block:: bibtex

      @software{rheo2024,
        title = {Rheo: JAX-Powered Unified Rheology Package},
        year = {2024},
        author = {Wei Chen},
        url = {https://github.com/username/rheo},
        version = {0.1.0}
      }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

rheo is released under the MIT License. See LICENSE file for details.
