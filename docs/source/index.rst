.. RheoJAX documentation master file

Welcome to RheoJAX's documentation!
=====================================

**RheoJAX** is a unified, JAX-accelerated rheological analysis package that provides a modern,
high-performance framework for analyzing experimental rheology data. Built on JAX for automatic
differentiation and GPU acceleration, RheoJAX combines powerful numerical capabilities with an
intuitive API for seamless end-to-end analysis.

.. only:: html

   .. image:: https://img.shields.io/badge/python-3.12+-blue.svg
      :target: https://www.python.org/downloads/
      :alt: Python 3.12+

   .. image:: https://img.shields.io/badge/License-MIT-yellow.svg
      :target: https://opensource.org/licenses/MIT
      :alt: MIT License

----

At a Glance
------------

.. list-table::
   :widths: 30 70
   :header-rows: 0

   * - **Models**
     - 53 models across 22 families (classical, fractional, flow, constitutive ODE, transient network, vitrimer, and more)
   * - **Bayesian**
     - Full MCMC inference (NumPyro NUTS) with NLSQ warm-start for all models
   * - **Transforms**
     - 7 transforms (FFT, mastercurve/TTS, SRFS, SPP, OWChirp, derivatives, mutation number)
   * - **Performance**
     - 10-100x speedups via JAX; 5-270x optimization via NLSQ
   * - **DMTA/DMA**
     - Automatic E* ↔ G* conversion for 41+ oscillation models with tensile deformation mode
   * - **Notebooks**
     - 244 tutorial notebooks across 21 categories covering all model families and protocols

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
   Plugin system for custom models and transforms with registry-based discovery

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

----

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api_reference

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
   development_status

----

Community and Support
---------------------

**Getting Help**
   - Read the :doc:`user_guide/index`
   - Report issues on `GitHub <https://github.com/imewei/rheojax/issues>`_
   - Contact: wchen@anl.gov

**Contributing**
   - See :doc:`developer/contributing` for guidelines
   - Check :doc:`developer/architecture` for design principles

**Citation**
   If you use RheoJAX in your research, please cite:

   .. code-block:: bibtex

      @software{rheojax2024,
        title = {RheoJAX: JAX-Powered Rheological Analysis with Bayesian Inference},
        year = {2024-2026},
        author = {Wei Chen},
        url = {https://github.com/imewei/rheojax},
        version = {0.6.0}
      }

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

RheoJAX is released under the MIT License. See LICENSE file for details.
