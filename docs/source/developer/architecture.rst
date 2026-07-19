Architecture Overview
=====================

This document describes the design principles and architectural decisions behind rheojax.

Design Philosophy
-----------------

JAX-First Design
~~~~~~~~~~~~~~~~

All numerical operations in rheojax use JAX for:

**Automatic Differentiation**
    Exact gradients for optimization without manual derivatives

**JIT Compilation**
    Performance approaching hand-optimized C code

**GPU/TPU Support**
    Transparent acceleration on available hardware

**Vectorization**
    Automatic batching and parallelization

.. code-block:: python

    # ALWAYS use safe_import_jax() — never import jax directly
    from rheojax.core.jax_config import safe_import_jax
    jax, jnp = safe_import_jax()  # Ensures float64 is enabled

    @jax.jit
    def relaxation_modulus(t, E, tau):
        """JIT-compiled for speed."""
        return E * jnp.exp(-t / tau)

    # Automatic gradient
    grad_fn = jax.grad(relaxation_modulus, argnums=(1, 2))

Scikit-learn API Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Core Architecture
-----------------

Module Structure
~~~~~~~~~~~~~~~~

.. code-block:: text

    rheojax/
    |--- core/               # Core abstractions
    |   |--- base.py         # BaseModel (BayesianMixin), BaseTransform
    |   |--- bayesian.py     # Bayesian inference engine (NumPyro NUTS)
    |   |--- data.py         # RheoData container (JAX-native)
    |   |--- parameters.py   # ParameterSet with bounds/priors
    |   |--- test_modes.py   # Test mode detection
    |   |--- registry.py     # ModelRegistry + TransformRegistry
    |   |--- inventory.py    # Protocol enum + model capability discovery
    |   \--- jax_config.py   # safe_import_jax() for float64
    |--- models/             # 53 rheological models across 22 families
    |   |--- classical/      # Maxwell, Zener, SpringPot
    |   |--- fractional/     # FML, FZSS, FMG, Burgers, etc. (11 models)
    |   |--- flow/           # PowerLaw, Carreau, HB, Bingham, Cross, CY
    |   |--- multimode/      # GeneralizedMaxwell (Prony series)
    |   |--- giesekus/       # SingleMode, MultiMode (tensor ODE)
    |   |--- sgr/            # SGRConventional, SGRGeneric
    |   |--- fluidity/       # Local, Nonlocal + Saramito variants (4)
    |   |--- epm/            # LatticeEPM, TensorialEPM
    |   |--- ikh/            # MIKH, MLIKH
    |   |--- fikh/           # FIKH, FMLIKH (fractional IKH)
    |   |--- dmt/            # DMTLocal, DMTNonlocal (thixotropy)
    |   |--- hl/             # HebraudLequeux
    |   |--- stz/            # STZConventional
    |   |--- spp/            # SPPYieldStress (LAOS)
    |   |--- itt_mct/        # ITTMCTSchematic, Isotropic (MCT)
    |   |--- tnt/            # SingleMode, Cates, LoopBridge, etc. (5)
    |   |--- vlb/            # Local, MultiNetwork, Variant, Nonlocal
    |   |--- hvm/            # HVMLocal (vitrimer, 3 subnetworks)
    |   \--- hvnm/           # HVNMLocal (nanocomposite, 4 subnetworks)
    |--- transforms/         # 11 data transforms
    |   |--- fft_analysis.py     # FFT spectral analysis
    |   |--- mastercurve.py      # TTS + auto shift factors
    |   |--- owchirp.py          # OWChirp LAOS analysis
    |   |--- srfs.py             # Strain-Rate Frequency Superposition
    |   |--- spp_decomposer.py   # SPP decomposition
    |   |--- mutation_number.py  # Mutation number
    |   |--- smooth_derivative.py  # Savitzky-Golay derivatives
    |   |--- cox_merz.py         # Cox-Merz rule validation
    |   |--- prony_conversion.py # Prony series time<->frequency conversion
    |   |--- spectrum_inversion.py  # Relaxation spectrum H(tau) recovery
    |   \--- lve_envelope.py     # Linear viscoelastic startup envelope
    |--- utils/              # Utilities
    |   |--- optimization.py     # NLSQ interface (5-270x vs scipy)
    |   |--- prony.py            # Prony series decomposition
    |   |--- mct_kernels.py      # MCT numerical kernels

    |   \--- initialization/     # Smart parameter initialization
    |--- pipeline/           # High-level workflows
    |   |--- base.py         # Pipeline (fluent API)
    |   |--- bayesian.py     # BayesianPipeline (ArviZ diagnostics)
    |   \--- workflows.py    # Batch processing
    |--- io/                 # File I/O
    |   |--- readers/        # TRIOS, CSV, Excel, Anton Paar, auto-detect
    |   \--- writers/        # HDF5, Excel writers
    |--- visualization/      # Plotting (3 styles)
    |--- logging/            # Structured logging (JAX-safe)
    \--- gui/                # PyQt/PySide6 interface

Component Relationships
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    +---------------------------------------------+
    |        User Applications / GUI               |
    \------------------+---------------------------+
                      |
    +-----------------v---------------------------+
    |         Pipeline / BayesianPipeline          |
    |    Fluent workflows + batch processing       |
    \------------------+---------------------------+
                      |
         +------------+------------+
         |            |            |
    +----v----+  +---v-----+ +---v-----------+
    | Models  |  |Transforms| | Visualization |
    | (53)    |  |   (7)    | | (3 styles)    |
    \----+----+  \----+-----+ \----+----------+
         |           |            |
    +----v-----------v------------v--------+
    |           Core Components             |
    |  RheoData, Parameters, BayesianMixin  |
    |  Registry, Logging                    |
    \-----+--------------------------+-------+
         |                          |
    +----v-----+              +----v-----+
    |   I/O    |              |  Utils   |
    | (readers |              | (NLSQ,   |
    |  writers)|              |  Prony)  |
    \----------+              \----------+
         |                          |
    +----v--------------------------v----+
    |    JAX / NLSQ / NumPyro / ArviZ    |
    |      (Numerical Foundation)         |
    \--------------------------------------+

.. note:: GUI does not route through Pipeline

   The diagram above simplifies the GUI's relationship to ``Pipeline``.
   In practice, ``rheojax.gui.services.pipeline_execution_service``
   independently reimplements pipeline-wizard step execution (driving
   ``ModelService``, ``TransformService``, and ``DataService`` directly)
   rather than calling ``rheojax.pipeline.Pipeline``/``PipelineBuilder`` —
   the GUI never imports ``rheojax.pipeline`` at all. The CLI's YAML
   pipeline runner (``cli/_yaml_runner.py``) does use ``PipelineBuilder``.
   This means the two entry points can drift independently; a change to
   pipeline-step semantics (retries, checkpoints, error handling) made in
   one does not automatically apply to the other.

.. note:: Two independent plotting stacks

   ``rheojax.visualization`` (matplotlib, used by ``Pipeline.plot_fit()``
   etc.) and the GUI's native PyQtGraph canvas
   (``rheojax.gui.widgets.pyqtgraph_canvas``) are maintained completely
   separately — the GUI does not import ``rheojax.visualization``. A
   plot-styling or behavior fix made in one stack (axis scaling, colorbar
   handling, etc.) does not propagate to the other.

Base Class Hierarchy
--------------------

Model Hierarchy
~~~~~~~~~~~~~~~

.. code-block:: text

    BaseModel (ABC) + BayesianMixin
    |--- Classical: Maxwell, Zener, SpringPot
    |--- Fractional: FML, FZSS, FMG, FKV, Burgers, etc. (11 models)
    |--- Flow: PowerLaw, Carreau, CarreauYasuda, Cross, HB, Bingham
    |--- GeneralizedMaxwell (multi-mode Prony series)
    |--- Giesekus: Single/MultiMode (tensor ODE)
    |--- SGR: Conventional, GENERIC (soft glassy rheology)
    |--- Fluidity: Local, Nonlocal + Saramito Local/Nonlocal
    |--- EPM: Lattice, Tensorial (elasto-plastic)
    |--- IKH/FIKH: MIKH, MLIKH, FIKH, FMLIKH (kinematic hardening)
    |--- DMT: Local, Nonlocal (thixotropy)
    |--- HL, STZ, SPP (single-model families)
    |--- ITT-MCT: Schematic, Isotropic (mode-coupling theory)
    |--- TNT: SingleMode, Cates, LoopBridge, MultiSpecies, StickyRouse
    |--- VLBBase → VLBLocal, MultiNetwork, Variant, Nonlocal
    |--- VLBBase → HVMBase → HVMLocal (vitrimer)
    \--- VLBBase → HVMBase → HVNMBase → HVNMLocal (nanocomposite)

    All 53 models support: .fit(), .predict(), .fit_bayesian(),
    .sample_prior(), .get_credible_intervals()



Transform Hierarchy
~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    BaseTransform (ABC)
    |--- FFT                # Spectral analysis
    |--- Mastercurve        # TTS (WLF, Arrhenius, auto shift)
    |--- OWChirp            # LAOS frequency-domain analysis
    |--- MutationNumber     # Thermorheological simplicity check
    |--- SmoothDerivative   # Savitzky-Golay smoothing
    |--- SRFS               # Strain-rate frequency superposition
    \--- SPP                # Sequence of Physical Processes (LAOS)

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

    # Register model with protocols
    @ModelRegistry.register(
        name="CustomModel",
        protocols=[Protocol.RELAXATION, Protocol.CREEP, Protocol.OSCILLATION],
    )
    class CustomModel(BaseModel):
        pass

    # Register transform
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
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    1. Load Data
       |--- Auto-detect format (auto_read)
       |--- Parse file
       \--- Create RheoData

    2. Preprocess
       |--- Detect test mode
       |--- Validate data
       |--- Apply transforms (smooth, filter)
       \--- Convert to appropriate domain

    3. Model Fitting
       |--- Select model (manual or auto)
       |--- Set initial parameters
       |--- Optimize parameters (JAX gradients)
       \--- Store fitted model

    4. Analysis
       |--- Make predictions
       |--- Compute residuals
       |--- Calculate metrics
       \--- Cross-validate

    5. Visualization
       |--- Plot data and fit
       |--- Plot residuals
       \--- Save figures

    6. Export
       |--- Save results (HDF5, Excel)
       \--- Export parameters

JAX Integration Details
-----------------------

Array Handling
~~~~~~~~~~~~~~

rheojax supports both NumPy and JAX arrays seamlessly:

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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
    pytest --cov=rheojax --cov-report=html

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

GUI Architecture
----------------

The desktop app (``gui/``, ``rheojax-gui``) is built around a single shell,
the workspace shell. See :doc:`../../architecture-overview` (GUI section) for
the package-layout table; the summary below covers what a contributor needs
to know before touching it.

Workspace Shell: ``workspace/window.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``WorkspaceWindow`` renders the Fit, Transform, and Pipeline workflows as a
literal numbered step wizard (``workspace/stepper_canvas.py``), backed by
``workspace/fit/step1_protocol_model.py`` .. ``step6_export.py`` and
``workspace/transform/step1_pick.py`` .. ``step5_export.py``. State lives in
``foundation/state.py::AppState`` (``library``, ``fit: FitState``,
``transform: TransformState``, ``jobs``, ``project``) — one instance per
window, no reducer/signal layer; components mutate ``FitState``/
``TransformState`` fields directly via ``dataclasses.replace()``. NLSQ/NUTS
runs go through true OS subprocess isolation (``jobs/process_adapter.py``,
``jobs/subprocess_fit.py``, ``jobs/subprocess_bayesian.py``) rather than
in-process threads.

Editing an earlier step (e.g. changing the model) calls
``foundation/invalidation.py::invalidate_downstream()``, which clears
dependent downstream ``FitState``/``TransformState`` fields (``nlsq_result``,
``nuts_result``, ...) via a static cascade table, and
``workspace/controller.py::WorkflowController.on_edit()`` re-locks every step
after the edited one in the stepper UI — this is what prevents a stale result
computed under an old upstream choice from surviving an edit.

Future Extensions
-----------------

Most of the originally-planned phases below are implemented; this section now
tracks what remains open rather than a roadmap from project inception.

Near-Term
~~~~~~~~~

- Unify the ``foundation/state.py`` ``AppState`` used by the workspace shell
  with the older ``state/store.py`` ``AppState`` still used by ``services/``,
  removing the dual state-model maintenance burden

Longer-Term
~~~~~~~~~~~

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
