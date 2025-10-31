Core Concepts
=============

This guide provides an in-depth look at the fundamental concepts and data structures in rheojax.

RheoData Container
------------------

The :class:`~rheojax.core.data.RheoData` class is the central data container in rheo, wrapping piblin.Measurement while adding JAX support and rheology-specific features.

Basic Structure
~~~~~~~~~~~~~~~

RheoData stores experimental rheological data with metadata:

.. code-block:: python

    from rheojax.core import RheoData
    import numpy as np

    # Create RheoData
    data = RheoData(
        x=np.array([0.1, 1.0, 10.0]),      # Independent variable
        y=np.array([1000, 800, 600]),       # Dependent variable
        x_units="s",                        # X-axis units
        y_units="Pa",                       # Y-axis units
        domain="time",                      # Data domain
        metadata={'temperature': 25.0}      # Additional metadata
    )

Attributes
~~~~~~~~~~

:x: Independent variable data (time, frequency, shear rate)
:y: Dependent variable data (stress, strain, modulus, viscosity)
:x_units: Units for x-axis (e.g., "s", "rad/s", "1/s")
:y_units: Units for y-axis (e.g., "Pa", "Pa.s")
:domain: Data domain ("time" or "frequency")
:metadata: Dictionary for additional information
:test_mode: Automatically detected test type (read-only property)

Domain Types
~~~~~~~~~~~~

The `domain` attribute indicates the data space:

- **"time"**: Time-domain data (relaxation, creep, step strain)
- **"frequency"**: Frequency-domain data (oscillatory measurements, SAOS)

.. code-block:: python

    # Time domain
    relaxation = RheoData(x=time, y=stress, domain="time")

    # Frequency domain with complex modulus
    freq = np.logspace(-2, 2, 50)
    G_star = Gp + 1j * Gpp  # G* = G' + iG"
    oscillation = RheoData(x=freq, y=G_star, domain="frequency")

Complex Data
~~~~~~~~~~~~

RheoData supports complex-valued data for frequency-domain measurements:

.. code-block:: python

    # Complex modulus G* = G' + iG"
    omega = np.logspace(-1, 3, 100)
    Gp = 1000 * omega**0.5        # Storage modulus
    Gpp = 500 * omega**0.3        # Loss modulus
    G_star = Gp + 1j * Gpp

    data = RheoData(x=omega, y=G_star, x_units="rad/s", y_units="Pa")

    # Access components
    print(data.is_complex)        # True
    print(data.modulus)           # |G*|
    print(data.phase)             # Phase angle

Array-like Interface
~~~~~~~~~~~~~~~~~~~~

RheoData implements numpy-like operations:

.. code-block:: python

    # Shape and size
    print(data.shape)    # Shape of y array
    print(data.ndim)     # Number of dimensions
    print(data.size)     # Total number of elements
    print(data.dtype)    # Data type

    # Indexing
    single_point = data[10]           # Returns (x[10], y[10])
    subset = data[10:20]              # Returns new RheoData
    mask = data.x > 1.0
    filtered = data[mask]             # Boolean indexing

    # Iteration
    for x_val, y_val in zip(data.x, data.y):
        print(f"x={x_val}, y={y_val}")

Arithmetic Operations
~~~~~~~~~~~~~~~~~~~~~

Perform mathematical operations on RheoData:

.. code-block:: python

    # Scalar operations
    scaled = data * 2.0              # Multiply y by 2
    shifted = data + 100.0           # Add 100 to y

    # RheoData operations (requires matching x-axes)
    data1 = RheoData(x=x, y=y1)
    data2 = RheoData(x=x, y=y2)
    sum_data = data1 + data2         # Element-wise addition
    diff_data = data1 - data2        # Element-wise subtraction
    prod_data = data1 * data2        # Element-wise multiplication

Data Manipulation
~~~~~~~~~~~~~~~~~

Built-in methods for common operations:

.. code-block:: python

    # Smoothing
    smoothed = data.smooth(window_size=5)

    # Resampling
    resampled = data.resample(n_points=200)

    # Interpolation
    new_x = np.linspace(0, 10, 100)
    interpolated = data.interpolate(new_x)

    # Calculus operations
    derivative = data.derivative()     # dy/dx
    integral = data.integral()         # ∫y dx

    # Slicing by x-value (piblin compatibility)
    subset = data.slice(start=1.0, end=10.0)

JAX Integration
~~~~~~~~~~~~~~~

Convert between NumPy and JAX arrays:

.. code-block:: python

    import jax.numpy as jnp

    # Convert to JAX
    data_jax = data.to_jax()
    print(type(data_jax.x))  # jax.Array

    # Convert to NumPy
    data_np = data_jax.to_numpy()
    print(type(data_np.x))   # numpy.ndarray

    # Operations preserve array type
    smoothed = data_jax.smooth(5)
    print(type(smoothed.x))  # jax.Array

Piblin Compatibility
~~~~~~~~~~~~~~~~~~~~

Full integration with piblin.Measurement:

.. code-block:: python

    import piblin

    # From piblin
    measurement = piblin.Measurement.from_file("data.h5")
    rheo_data = RheoData.from_piblin(measurement)

    # To piblin
    measurement = rheo_data.to_piblin()

    # RheoData maintains piblin API
    rheo_data.metadata['instrument'] = 'ARES-G2'
    print(rheo_data.metadata)

Serialization
~~~~~~~~~~~~~

Save and load RheoData:

.. code-block:: python

    # To dictionary
    data_dict = data.to_dict()

    # From dictionary
    restored = RheoData.from_dict(data_dict)

    # Use I/O writers for files
    from rheojax.io.writers import write_hdf5
    write_hdf5(data, "results.h5")

Parameter System
----------------

The parameter system (:class:`~rheojax.core.parameters.Parameter`, :class:`~rheojax.core.parameters.ParameterSet`) manages model parameters with bounds, constraints, and optimization support.

Parameter Class
~~~~~~~~~~~~~~~

Individual parameters with metadata:

.. code-block:: python

    from rheojax.core import Parameter

    # Create parameter
    modulus = Parameter(
        name="E",
        value=1000.0,
        bounds=(100, 10000),
        units="Pa",
        description="Elastic modulus"
    )

    # Access properties
    print(modulus.value)       # 1000.0
    print(modulus.bounds)      # (100, 10000)
    print(modulus.units)       # "Pa"

    # Set value (with validation)
    modulus.value = 5000.0     # OK
    # modulus.value = 50.0     # Raises ValueError (out of bounds)

ParameterSet Class
~~~~~~~~~~~~~~~~~~

Collection of parameters for models:

.. code-block:: python

    from rheojax.core import ParameterSet

    # Create parameter set
    params = ParameterSet()

    # Add parameters
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
        bounds=(0.01, 100),
        units="s",
        description="Relaxation time"
    )
    params.add(
        name="alpha",
        value=0.5,
        bounds=(0.1, 1.0),
        units=None,
        description="Fractional order"
    )

    # Access parameters
    print(len(params))                    # 3
    print("E" in params)                  # True
    E_param = params.get("E")             # Get Parameter object

    # Get/set values
    E_value = params.get_value("E")       # 1000.0
    params.set_value("tau", 2.5)          # Set new value

    # Array interface
    values = params.get_values()          # [1000.0, 2.5, 0.5]
    params.set_values([2000, 1.5, 0.7])   # Set all values

    # Get bounds
    bounds = params.get_bounds()          # [(100, 10000), (0.01, 100), (0.1, 1.0)]

Parameter Constraints
~~~~~~~~~~~~~~~~~~~~~

Advanced constraint system:

.. code-block:: python

    from rheojax.core.parameters import ParameterConstraint

    # Bounds constraint (automatic from bounds parameter)
    constraint = ParameterConstraint(
        type="bounds",
        min_value=0,
        max_value=1000
    )

    # Positive constraint
    positive = ParameterConstraint(type="positive")

    # Relative constraint (parameter relationships)
    relative = ParameterConstraint(
        type="relative",
        relation="less_than",
        other_param="tau2"
    )

    # Custom constraint
    def must_be_even(value):
        return value % 2 == 0

    custom = ParameterConstraint(
        type="custom",
        validator=must_be_even
    )

    # Add constraints to parameter
    param = Parameter(
        name="n",
        value=4,
        constraints=[positive, custom]
    )

Shared Parameters
~~~~~~~~~~~~~~~~~

Share parameters across multiple models:

.. code-block:: python

    from rheojax.core.parameters import SharedParameterSet

    # Create shared parameter set
    shared = SharedParameterSet()

    # Add shared parameter
    shared.add_shared(
        name="temperature",
        value=25.0,
        bounds=(0, 100),
        units="°C",
        group="experimental_conditions"
    )

    # Link to models (Phase 2 feature)
    # shared.link_model(model1, "temperature")
    # shared.link_model(model2, "temperature")

    # Update shared value (updates all linked models)
    # shared.set_value("temperature", 30.0)

Test Mode Detection
-------------------

Automatic detection of rheological test types from data characteristics.

TestMode Enum
~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.core.test_modes import TestMode

    # Available test modes
    TestMode.RELAXATION   # Stress relaxation
    TestMode.CREEP        # Creep compliance
    TestMode.OSCILLATION  # Oscillatory (SAOS/LAOS)
    TestMode.ROTATION     # Steady shear (flow curve)
    TestMode.UNKNOWN      # Cannot determine

Detection Algorithm
~~~~~~~~~~~~~~~~~~~

The detection follows this logic:

.. code-block:: python

    from rheojax.core.test_modes import detect_test_mode

    # 1. Check explicit metadata
    data.metadata['test_mode'] = 'relaxation'
    mode = detect_test_mode(data)  # Returns TestMode.RELAXATION

    # 2. Check domain
    freq_data = RheoData(x=omega, y=G_star, domain="frequency")
    mode = detect_test_mode(freq_data)  # Returns TestMode.OSCILLATION

    # 3. Check x_units
    flow_data = RheoData(x=shear_rate, y=viscosity, x_units="1/s")
    mode = detect_test_mode(flow_data)  # Returns TestMode.ROTATION

    # 4. Check monotonicity for time-domain data
    decreasing_data = RheoData(x=t, y=stress_decreasing, domain="time")
    mode = detect_test_mode(decreasing_data)  # Returns TestMode.RELAXATION

    increasing_data = RheoData(x=t, y=strain_increasing, domain="time")
    mode = detect_test_mode(increasing_data)  # Returns TestMode.CREEP

Monotonicity Checks
~~~~~~~~~~~~~~~~~~~

Helper functions for data analysis:

.. code-block:: python

    from rheojax.core.test_modes import is_monotonic_increasing, is_monotonic_decreasing
    import numpy as np

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(is_monotonic_increasing(data))     # True
    print(is_monotonic_decreasing(data))     # False

    # With tolerance for numerical noise
    noisy_data = np.array([1.0, 2.0, 2.00001, 3.0])
    print(is_monotonic_increasing(noisy_data, tolerance=1e-4))  # True

    # Strict monotonicity (no equal consecutive values)
    plateau_data = np.array([1.0, 2.0, 2.0, 3.0])
    print(is_monotonic_increasing(plateau_data, strict=True))   # False

Using Test Mode
~~~~~~~~~~~~~~~

Access detected test mode:

.. code-block:: python

    # Automatic detection and caching
    data = RheoData(x=time, y=stress_relaxation, domain="time")
    mode = data.test_mode  # Detects and caches result
    print(mode)            # TestMode.RELAXATION

    # Second access uses cached value
    mode2 = data.test_mode  # No re-detection

    # Check test mode type
    if data.test_mode == TestMode.RELAXATION:
        print("This is a stress relaxation test")

    # Get string representation
    print(str(data.test_mode))  # "relaxation"

Model Registry
--------------

The registry system (implemented in Phase 2) will provide model and transform discovery:

.. code-block:: python

    # Phase 2 feature preview
    from rheojax.core.registry import get_model, list_models

    # List available models
    # models = list_models()

    # Get model by name
    # maxwell = get_model("Maxwell")

    # Filter by test mode compatibility
    # relaxation_models = list_models(test_mode=TestMode.RELAXATION)

Data Validation
---------------

RheoData validates data on creation:

.. code-block:: python

    import numpy as np

    # Valid data
    data = RheoData(
        x=np.array([1, 2, 3]),
        y=np.array([10, 20, 30])
    )

    # Invalid: mismatched shapes
    try:
        bad = RheoData(
            x=np.array([1, 2, 3]),
            y=np.array([10, 20])  # Wrong length
        )
    except ValueError as e:
        print(e)  # "x and y must have the same shape"

    # Invalid: NaN values
    try:
        bad = RheoData(
            x=np.array([1, 2, np.nan]),
            y=np.array([10, 20, 30])
        )
    except ValueError as e:
        print(e)  # "x data contains NaN values"

    # Disable validation for performance
    data = RheoData(x=x, y=y, validate=False)

Best Practices
--------------

1. **Units Consistency**

   Always specify units for clarity:

   .. code-block:: python

       data = RheoData(x=time, y=stress, x_units="s", y_units="Pa")

2. **Metadata Documentation**

   Use metadata for experimental conditions:

   .. code-block:: python

       data.metadata.update({
           'temperature': 25.0,
           'temperature_units': '°C',
           'sample': 'PMMA',
           'instrument': 'ARES-G2',
           'date': '2024-10-24'
       })

3. **Test Mode Specification**

   For ambiguous cases, set test mode explicitly:

   .. code-block:: python

       data.metadata['test_mode'] = 'creep'

4. **Parameter Bounds**

   Always specify physical bounds:

   .. code-block:: python

       params.add("E", value=1000, bounds=(1, 1e6))  # Modulus must be positive

5. **JAX Usage**

   Use JAX for performance-critical code:

   .. code-block:: python

       # Convert once, use many times
       data_jax = data.to_jax()

       # JIT compile functions
       @jax.jit
       def compute(x):
           return jnp.sum(jnp.exp(-x))

Summary
-------

The core concepts in rheo are:

- **RheoData**: Central container for experimental data with metadata and JAX support
- **Parameters**: Type-safe parameter management with bounds and constraints
- **Test Mode Detection**: Automatic identification of experimental test types
- **JAX Integration**: Seamless CPU/GPU computation with automatic differentiation

These components provide a solid foundation for rheological analysis workflows, with extensibility for custom models and transforms in Phase 2.

For more details, see:

- :doc:`getting_started` - Quick start guide
- :doc:`io_guide` - Reading and writing data
- :doc:`visualization_guide` - Plotting data
- :doc:`../api/core` - Complete API reference
