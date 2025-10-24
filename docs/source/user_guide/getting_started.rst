Getting Started
===============

Welcome to **rheo**, a JAX-powered rheological analysis package that provides a unified framework for analyzing experimental rheology data with state-of-the-art performance and flexibility.

Installation
------------

Requirements
~~~~~~~~~~~~

- Python 3.12 or later (Python 3.8-3.11 are NOT supported due to JAX requirements)
- JAX and jaxlib for GPU/CPU acceleration
- NumPy, SciPy for numerical operations
- Matplotlib for visualization
- Optional: piblin for enhanced data management

Basic Installation
~~~~~~~~~~~~~~~~~~

Install from PyPI::

    pip install rheo

For Development
~~~~~~~~~~~~~~~

Clone the repository and install in editable mode::

    git clone https://github.com/username/rheo.git
    cd rheo
    pip install -e ".[dev]"

GPU Support
~~~~~~~~~~~

For CUDA 12 GPU acceleration::

    pip install "rheo[gpu]"

This will install JAX with CUDA support for significant performance improvements on compatible hardware.

Quick Example
-------------

Let's start with a simple example that demonstrates the core workflow:

Loading and Analyzing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from rheo.core import RheoData
    from rheo.io.readers import read_trios

    # Load data from TRIOS file (auto-detects format)
    data = read_trios("stress_relaxation.txt")

    # Or create RheoData directly
    time = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    stress = np.array([1000, 750, 550, 400, 250, 150])
    data = RheoData(
        x=time,
        y=stress,
        x_units="s",
        y_units="Pa",
        domain="time"
    )

    # Check detected test mode
    print(f"Test mode: {data.test_mode}")  # Output: relaxation

Visualizing Data
~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheo.visualization import plot_rheo_data
    import matplotlib.pyplot as plt

    # Create plot (automatic type detection based on data)
    fig, ax = plot_rheo_data(data, style='publication')
    plt.show()

Working with Parameters
~~~~~~~~~~~~~~~~~~~~~~~

The parameter system provides bounds, constraints, and optimization support:

.. code-block:: python

    from rheo.core import Parameter, ParameterSet

    # Create parameter set
    params = ParameterSet()
    params.add(
        name="E",
        value=1000.0,
        bounds=(100, 10000),
        units="Pa",
        description="Elastic modulus"
    )
    params.add(
        name="tau",
        value=1.0,
        bounds=(0.1, 100),
        units="s",
        description="Relaxation time"
    )

    # Get/set values
    E_value = params.get_value("E")
    params.set_value("tau", 2.5)

    # Get all values as array
    values = params.get_values()  # [1000.0, 2.5]

Data Operations
~~~~~~~~~~~~~~~

RheoData supports numpy-like operations:

.. code-block:: python

    # Indexing and slicing
    subset = data[10:50]  # Slice data

    # Arithmetic operations
    scaled = data * 2.0  # Scale y-values

    # Data manipulation
    smoothed = data.smooth(window_size=5)
    resampled = data.resample(n_points=100)
    interpolated = data.interpolate(new_x)

    # Derivatives and integrals
    derivative = data.derivative()
    integral = data.integral()

    # Copy data
    data_copy = data.copy()

Using JAX for Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~

All computations can leverage JAX for automatic differentiation and GPU acceleration:

.. code-block:: python

    import jax.numpy as jnp

    # Convert to JAX arrays
    data_jax = data.to_jax()

    # Define a JAX function
    @jax.jit
    def model_prediction(t, params):
        E, tau = params
        return E * jnp.exp(-t / tau)

    # Use with optimization
    from rheo.utils.optimization import nlsq_optimize

    def objective(params):
        predictions = model_prediction(data_jax.x, params)
        residuals = predictions - data_jax.y
        return jnp.sum(residuals**2)

    # Optimize with JAX gradients
    result = nlsq_optimize(objective, params, use_jax=True)

Reading Various File Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rheo supports multiple rheometer data formats:

.. code-block:: python

    from rheo.io.readers import (
        read_trios,      # TA Instruments TRIOS
        read_csv,        # Generic CSV
        read_excel,      # Excel files
        read_anton_paar, # Anton Paar
        auto_read        # Auto-detect format
    )

    # Auto-detection (recommended)
    data = auto_read("experiment.txt")

    # Specific readers with options
    data = read_csv("data.csv", x_column="Time", y_column="Stress")
    data = read_excel("results.xlsx", sheet_name="Oscillation")

Writing Results
~~~~~~~~~~~~~~~

Save your analysis results in various formats:

.. code-block:: python

    from rheo.io.writers import write_hdf5, write_excel

    # Write to HDF5 (preserves all metadata)
    write_hdf5(data, "results.h5")

    # Write to Excel
    write_excel(data, "results.xlsx", sheet_name="Analysis")

Automatic Test Mode Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

rheo automatically detects the test type based on data characteristics:

.. code-block:: python

    from rheo.core.test_modes import TestMode, detect_test_mode

    # Automatic detection
    test_mode = detect_test_mode(data)
    print(test_mode)  # TestMode.RELAXATION, CREEP, OSCILLATION, ROTATION, or UNKNOWN

    # Detection logic:
    # 1. Check metadata['test_mode'] if set
    # 2. Frequency domain → OSCILLATION
    # 3. Time domain with decreasing y → RELAXATION
    # 4. Time domain with increasing y → CREEP
    # 5. x_units with "1/s" → ROTATION (steady shear)

    # Manually set test mode
    data.metadata['test_mode'] = 'relaxation'

Next Steps
----------

Now that you've learned the basics, explore:

- :doc:`core_concepts` - Deep dive into RheoData, Parameters, and test modes
- :doc:`io_guide` - Comprehensive guide to reading and writing data
- :doc:`visualization_guide` - Creating publication-quality figures
- :doc:`../api_reference` - Complete API reference

For model fitting and transforms (coming in Phase 2):

- Models: Maxwell, Zener, Fractional models, and more
- Transforms: Master curves, FFT analysis, data processing

Common Patterns
---------------

Pattern 1: Load, Analyze, Save
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheo.io.readers import auto_read
    from rheo.io.writers import write_hdf5
    from rheo.visualization import plot_rheo_data

    # Load data
    data = auto_read("experiment.txt")

    # Analyze
    print(f"Test mode: {data.test_mode}")
    print(f"Data points: {len(data.x)}")

    # Visualize
    fig, ax = plot_rheo_data(data)
    fig.savefig("plot.png", dpi=300)

    # Save
    write_hdf5(data, "processed_data.h5")

Pattern 2: Data Processing Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Load and process data
    data = auto_read("noisy_data.txt")

    # Apply processing operations
    processed = (data
                 .smooth(window_size=5)      # Smooth noise
                 .resample(n_points=100)     # Resample to 100 points
                 [10:-10])                   # Trim edges

    # Analyze processed data
    print(f"Original: {len(data.x)} points")
    print(f"Processed: {len(processed.x)} points")

Pattern 3: Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pathlib
    from rheo.io.readers import auto_read

    # Process all files in directory
    data_dir = pathlib.Path("experiments/")
    results = {}

    for file in data_dir.glob("*.txt"):
        data = auto_read(file)
        results[file.stem] = {
            'test_mode': data.test_mode,
            'n_points': len(data.x),
            'x_range': (data.x.min(), data.x.max()),
            'y_range': (data.y.min(), data.y.max())
        }

    # Print summary
    for name, info in results.items():
        print(f"{name}: {info['test_mode']}, {info['n_points']} points")

Getting Help
------------

If you encounter issues:

1. Check the :doc:`../api_reference` for detailed function documentation
2. Review examples in the `examples/` directory
3. Search or open an issue on `GitHub <https://github.com/username/rheo/issues>`_
4. Join discussions on `GitHub Discussions <https://github.com/username/rheo/discussions>`_

Performance Tips
----------------

1. **Use JAX for heavy computations**: Convert to JAX arrays with `data.to_jax()`
2. **JIT compile repeated operations**: Use `@jax.jit` decorator
3. **Vectorize operations**: Work with arrays instead of loops
4. **Enable GPU**: Install with `pip install "rheo[gpu]"` for CUDA support
5. **Profile your code**: Use `%timeit` in Jupyter or `cProfile` for optimization

.. code-block:: python

    import jax
    import jax.numpy as jnp

    # JIT compilation for speed
    @jax.jit
    def fast_computation(x, params):
        return jnp.sum(jnp.exp(-x / params[0]) * params[1])

    # Use with data
    data_jax = data.to_jax()
    result = fast_computation(data_jax.x, jnp.array([1.0, 100.0]))
