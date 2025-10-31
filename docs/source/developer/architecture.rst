Architecture Overview
=====================

This document describes the design principles and architectural decisions behind rheojax.

Design Philosophy
-----------------

JAX-First Design
~~~~~~~~~~~~~~~~

All numerical operations in rheo use JAX for:

**Automatic Differentiation**
    Exact gradients for optimization without manual derivatives

**JIT Compilation**
    Performance approaching hand-optimized C code

**GPU/TPU Support**
    Transparent acceleration on available hardware

**Vectorization**
    Automatic batching and parallelization

.. code-block:: python

    import jax
    import jax.numpy as jnp

    @jax.jit
    def relaxation_modulus(t, E, tau):
        """JIT-compiled for speed."""
        return E * jnp.exp(-t / tau)

    # Automatic gradient
    grad_fn = jax.grad(relaxation_modulus, argnums=(1, 2))

Scikit-learn API Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models follow scikit-learn conventions:

- ``.fit(X, y)`` - Fit model to data
- ``.predict(X)`` - Make predictions
- ``.score(X, y)`` - Evaluate performance
- ``.get_params()`` / ``.set_params()`` - Parameter management

.. code-block:: python

    # Familiar API
    model = FractionalMaxwell(n_elements=5)
    model.fit(time, stress)
    predictions = model.predict(time)
    r2 = model.score(time, stress)

Piblin Integration
~~~~~~~~~~~~~~~~~~

Full compatibility with piblin.Measurement:

- RheoData wraps piblin.Measurement
- Maintains all piblin methods
- Adds JAX support and rheology-specific features

.. code-block:: python

    import piblin
    from rheojax.core import RheoData

    # From piblin
    measurement = piblin.Measurement.from_file("data.h5")
    rheo_data = RheoData.from_piblin(measurement)

    # Back to piblin
    measurement = rheo_data.to_piblin()

Core Architecture
-----------------

Module Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

    rheo/
    ├── core/               # Core abstractions
    │   ├── base.py        # BaseModel, BaseTransform
    │   ├── data.py        # RheoData container
    │   ├── parameters.py  # Parameter system
    │   ├── test_modes.py  # Test mode detection
    │   └── registry.py    # Model/transform registry
    ├── models/            # Rheological models (Phase 2)
    │   ├── maxwell.py
    │   ├── zener.py
    │   └── ...
    ├── transforms/        # Data transforms (Phase 2)
    │   ├── fft.py
    │   ├── mastercurve.py
    │   └── ...
    ├── utils/             # Utilities
    │   ├── mittag_leffler.py  # Special functions
    │   └── optimization.py    # Fitting tools
    ├── io/                # File I/O
    │   ├── readers/       # Data readers
    │   └── writers/       # Data writers
    ├── visualization/     # Plotting
    │   ├── plotter.py
    │   └── templates.py
    └── pipelines/         # High-level workflows (Phase 2)

Component Relationships
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    ┌─────────────────────────────────────────────┐
    │           User Applications                  │
    └─────────────────┬───────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────┐
    │            Pipelines (Phase 2)               │
    │  High-level workflows and analysis chains    │
    └─────────────────┬───────────────────────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼───┐   ┌───▼────┐  ┌───▼────────┐
    │ Models │   │Transforms│ │Visualization│
    │(Phase2)│   │ (Phase2) │ │   (Phase1) │
    └────┬───┘   └───┬─────┘ └───┬────────┘
         │           │            │
    ┌────▼───────────▼────────────▼────────┐
    │           Core Components             │
    │  RheoData, Parameters, Base Classes   │
    └────┬──────────────────────────┬───────┘
         │                          │
    ┌────▼─────┐              ┌────▼─────┐
    │   I/O    │              │  Utils   │
    │ (Phase1) │              │(Phase1)  │
    └──────────┘              └──────────┘
         │                          │
    ┌────▼──────────────────────────▼────┐
    │         JAX / NumPy / SciPy         │
    │      (Numerical Foundation)         │
    └─────────────────────────────────────┘

Base Class Hierarchy
--------------------

Model Hierarchy
~~~~~~~~~~~~~~~

.. code-block:: text

    BaseModel (ABC)
    ├── ViscoelasticModel
    │   ├── Maxwell
    │   ├── Zener
    │   ├── KelvinVoigt
    │   └── GeneralizedMaxwell
    ├── FractionalModel
    │   ├── FractionalMaxwell
    │   ├── FractionalZener
    │   └── FractionalKelvinVoigt
    └── FlowModel
        ├── PowerLaw
        ├── Carreau
        └── HerschelBulkley

Transform Hierarchy
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    BaseTransform (ABC)
    ├── FrequencyTransform
    │   ├── FFT
    │   └── InverseFFT
    ├── DataTransform
    │   ├── Smoothing
    │   ├── Interpolation
    │   └── Resampling
    └── AnalysisTransform
        ├── Mastercurve
        ├── OWChirp
        └── MutationNumber

Extension Points
----------------

Adding New Models
~~~~~~~~~~~~~~~~~

To add a custom model, inherit from ``BaseModel``:

.. code-block:: python

    from rheojax.core import BaseModel, ParameterSet
    import jax.numpy as jnp

    class CustomModel(BaseModel):
        """Custom rheological model."""

        def __init__(self, param1=1.0, param2=1.0):
            super().__init__()

            # Define parameters
            self.parameters = ParameterSet()
            self.parameters.add(
                "param1",
                value=param1,
                bounds=(0.1, 10),
                units="Pa"
            )
            self.parameters.add(
                "param2",
                value=param2,
                bounds=(0.01, 100),
                units="s"
            )

        def _fit(self, X, y, **kwargs):
            """Implement fitting logic."""
            from rheojax.utils.optimization import nlsq_optimize

            def objective(params):
                predictions = self._predict(X)
                return jnp.sum((predictions - y)**2)

            nlsq_optimize(objective, self.parameters, use_jax=True)
            return self

        def _predict(self, X):
            """Implement prediction logic."""
            p1 = self.parameters.get_value("param1")
            p2 = self.parameters.get_value("param2")

            # Model equation
            return p1 * jnp.exp(-X / p2)

Adding New Transforms
~~~~~~~~~~~~~~~~~~~~~

To add a custom transform, inherit from ``BaseTransform``:

.. code-block:: python

    from rheojax.core import BaseTransform
    import jax.numpy as jnp

    class CustomTransform(BaseTransform):
        """Custom data transform."""

        def __init__(self, param=1.0):
            super().__init__()
            self.param = param

        def _transform(self, data):
            """Implement forward transform."""
            # Access data
            x = data.x
            y = data.y

            # Transform
            y_transformed = y * self.param

            # Return new RheoData
            from rheojax.core import RheoData
            return RheoData(
                x=x,
                y=y_transformed,
                x_units=data.x_units,
                y_units=data.y_units,
                domain=data.domain,
                metadata=data.metadata.copy()
            )

        def _inverse_transform(self, data):
            """Implement inverse transform."""
            y_original = data.y / self.param

            from rheojax.core import RheoData
            return RheoData(
                x=data.x,
                y=y_original,
                x_units=data.x_units,
                y_units=data.y_units,
                domain=data.domain,
                metadata=data.metadata.copy()
            )

Registry Pattern
----------------

Models and transforms are registered for discovery:

.. code-block:: python

    from rheojax.core.registry import ModelRegistry, TransformRegistry

    # Register model (Phase 2)
    @ModelRegistry.register(
        name="CustomModel",
        test_modes=["relaxation", "creep"]
    )
    class CustomModel(BaseModel):
        pass

    # Register transform (Phase 2)
    @TransformRegistry.register(name="CustomTransform")
    class CustomTransform(BaseTransform):
        pass

    # Discover registered components
    models = ModelRegistry.list_models()
    transforms = TransformRegistry.list_transforms()

    # Instantiate by name
    model = ModelRegistry.get_model("CustomModel")
    transform = TransformRegistry.get_transform("CustomTransform")

Data Flow
---------

Typical Analysis Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    1. Load Data
       ├── Auto-detect format (auto_read)
       ├── Parse file
       └── Create RheoData

    2. Preprocess
       ├── Detect test mode
       ├── Validate data
       ├── Apply transforms (smooth, filter)
       └── Convert to appropriate domain

    3. Model Fitting
       ├── Select model (manual or auto)
       ├── Set initial parameters
       ├── Optimize parameters (JAX gradients)
       └── Store fitted model

    4. Analysis
       ├── Make predictions
       ├── Compute residuals
       ├── Calculate metrics
       └── Cross-validate

    5. Visualization
       ├── Plot data and fit
       ├── Plot residuals
       └── Save figures

    6. Export
       ├── Save results (HDF5, Excel)
       └── Export parameters

JAX Integration Details
-----------------------

Array Handling
~~~~~~~~~~~~~~

rheo supports both NumPy and JAX arrays seamlessly:

.. code-block:: python

    import numpy as np
    import jax.numpy as jnp
    from rheojax.core import RheoData

    # NumPy arrays
    data_np = RheoData(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))

    # JAX arrays
    data_jax = RheoData(x=jnp.array([1, 2, 3]), y=jnp.array([10, 20, 30]))

    # Convert between them
    data_jax = data_np.to_jax()
    data_np = data_jax.to_numpy()

JIT Compilation
~~~~~~~~~~~~~~~

Functions are JIT-compiled for performance:

.. code-block:: python

    import jax
    import jax.numpy as jnp

    @jax.jit
    def model_function(t, params):
        """JIT-compiled model function."""
        E, tau = params
        return E * jnp.exp(-t / tau)

    # First call: compilation + execution
    result1 = model_function(t, params)  # ~10ms

    # Subsequent calls: cached, fast execution
    result2 = model_function(t, params)  # ~0.1ms

Automatic Differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~

JAX provides automatic gradients:

.. code-block:: python

    import jax

    def objective(params):
        """Objective function to minimize."""
        predictions = model_function(t, params)
        return jnp.sum((predictions - y_observed)**2)

    # Compute gradient automatically
    grad_fn = jax.grad(objective)
    gradients = grad_fn(params)

    # Use in optimization
    from rheojax.utils.optimization import nlsq_optimize
    result = nlsq_optimize(objective, parameters, use_jax=True)

Performance Optimization
------------------------

Best Practices
~~~~~~~~~~~~~~

1. **Use JAX for heavy computation**

   .. code-block:: python

       # Convert to JAX arrays
       data_jax = data.to_jax()

       # JIT compile functions
       @jax.jit
       def heavy_computation(x):
           return jnp.sum(jnp.exp(-x))

2. **Vectorize operations**

   .. code-block:: python

       # Good: vectorized
       result = jnp.exp(-time / tau)

       # Bad: loop
       result = jnp.array([jnp.exp(-t / tau) for t in time])

3. **Avoid Python loops in hot paths**

   .. code-block:: python

       # Good: use vmap
       batch_fn = jax.vmap(single_fn)
       results = batch_fn(inputs)

       # Bad: Python loop
       results = [single_fn(inp) for inp in inputs]

4. **Profile before optimizing**

   .. code-block:: python

       import time

       start = time.time()
       result = compute_function(data)
       elapsed = time.time() - start
       print(f"Time: {elapsed:.3f}s")

Memory Management
~~~~~~~~~~~~~~~~~

JAX uses device arrays that may reside on GPU:

.. code-block:: python

    import jax.numpy as jnp

    # Create array (may be on GPU)
    x = jnp.array([1, 2, 3])

    # Transfer to CPU if needed
    x_cpu = np.array(x)

    # Free memory explicitly if needed
    del x

Testing Strategy
----------------

Unit Tests
~~~~~~~~~~

Each module has comprehensive unit tests:

.. code-block:: python

    # tests/core/test_data.py
    def test_rheodata_creation():
        """Test RheoData initialization."""
        data = RheoData(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))
        assert len(data.x) == 3
        assert data.shape == (3,)

Integration Tests
~~~~~~~~~~~~~~~~~

Test complete workflows:

.. code-block:: python

    # tests/test_workflows.py
    def test_complete_analysis():
        """Test full analysis workflow."""
        # Load data
        data = auto_read("test_data.txt")

        # Fit model
        model = Maxwell()
        model.fit(data.x, data.y)

        # Predict
        predictions = model.predict(data.x)

        # Verify
        assert model.score(data.x, data.y) > 0.9

Test Coverage
~~~~~~~~~~~~~

Aim for >90% test coverage:

.. code-block:: bash

    # Run tests with coverage
    pytest --cov=rheo --cov-report=html

    # View coverage report
    open htmlcov/index.html

Documentation Standards
-----------------------

Docstring Format
~~~~~~~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

    def function_name(param1, param2):
        """Short description.

        Longer description with more details about what the function does.

        Parameters
        ----------
        param1 : type
            Description of param1
        param2 : type
            Description of param2

        Returns
        -------
        return_type
            Description of return value

        Raises
        ------
        ValueError
            When parameter is invalid

        Examples
        --------
        >>> result = function_name(1, 2)
        >>> print(result)
        3

        Notes
        -----
        Additional implementation notes.

        References
        ----------
        .. [1] Author, "Title", Journal, Year
        """
        pass

Type Hints
~~~~~~~~~~

Use type hints for clarity:

.. code-block:: python

    from typing import Optional, Union, List
    import numpy as np
    import jax.numpy as jnp

    ArrayLike = Union[np.ndarray, jnp.ndarray, List]

    def process_data(
        x: ArrayLike,
        y: ArrayLike,
        method: str = "default"
    ) -> RheoData:
        """Process rheological data."""
        pass

Future Extensions
-----------------

Phase 2: Models and Transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- 20+ rheological models
- Master curve generation
- FFT analysis
- OWChirp signal processing
- Mutation number calculation

Phase 3: Advanced Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bayesian parameter estimation
- Uncertainty quantification
- Multi-objective optimization
- Parallel batch processing
- GPU-accelerated model fitting

Phase 4: Machine Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~

- Neural network surrogate models
- Active learning for parameter estimation
- Automated model selection
- Transfer learning for similar materials

See Also
--------

- :doc:`contributing` - Contribution guidelines
- :doc:`../user_guide/core_concepts` - Core concepts
- :doc:`../api_reference` - API documentation
- `JAX documentation <https://jax.readthedocs.io/>`_ - JAX details
