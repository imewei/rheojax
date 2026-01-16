Model Inventory System
======================

RheoJAX provides a powerful **Protocol-Driven Inventory System** to help you discover models and transforms that match your experimental data. Instead of guessing which models support which tests, you can query the system programmatically or via the command line.

The inventory categorizes components into two types:

1.  **Models**: Constitutive equations fit to data (e.g., Maxwell, SGR).
2.  **Transforms**: Data processing operations (e.g., FFT, TTS).

CLI Usage
---------

You can explore the available models and transforms using the ``rheojax inventory`` command.

List All Components
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    $ rheojax inventory

    RheoJAX Inventory
    =================

    Models
    ------
    Name                Protocols                                Description
    ------------------- ---------------------------------------- ------------------------------
    bingham             flow_curve                               Bingham model for linear viscoplastic flow...
    carreau             flow_curve                               Carreau model for non-Newtonian flow...
    ...

    Transforms
    ----------
    Name                Type            Description
    ------------------- --------------- ------------------------------
    fft_analysis        spectral        Transform time-domain rheological data...
    mastercurve         superposition   Time-Temperature Superposition (TTS)...
    ...

Filter by Protocol
^^^^^^^^^^^^^^^^^^

Find models that support a specific experimental protocol, such as Large Amplitude Oscillatory Shear (LAOS).

.. code-block:: bash

    $ rheojax inventory --protocol laos

    Models
    ------
    (Filtered by protocol: laos)
    Name                Protocols                                Description
    ------------------- ---------------------------------------- ------------------------------
    generalized_maxwell ... oscillation, flow_curve, startup, laos Generalized Maxwell Model...
    sgr_conventional    ... startup, oscillation, laos           Soft Glassy Rheology (SGR)...
    sgr_generic         ... startup, oscillation, laos           Soft Glassy Rheology (SGR)...
    spp_yield_stress    flow_curve, laos                         SPP-based yield stress model...
    stz_conventional    ... startup, oscillation, laos           Conventional Shear Transformation Zone...

Filter by Transform Type
^^^^^^^^^^^^^^^^^^^^^^^^

Find transforms for specific operations, such as superposition (shifting curves).

.. code-block:: bash

    $ rheojax inventory --type superposition

    Transforms
    ----------
    (Filtered by type: superposition)
    Name                Type            Description
    ------------------- --------------- ------------------------------
    mastercurve         superposition   Time-Temperature Superposition (TTS)...
    srfs                superposition   Strain-Rate Frequency Superposition...

Python API Usage
----------------

You can also access the inventory system within your Python scripts to build dynamic analysis pipelines.

Finding Compatible Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core.registry import ModelRegistry
    from rheojax.core.inventory import Protocol

    # Find all models that support Creep experiments
    creep_models = ModelRegistry.find(protocol=Protocol.CREEP)
    print(creep_models)
    # Output: ['maxwell', 'zener', 'springpot', 'fractional_maxwell', ...]

    # Instantiate a model by name
    model_name = creep_models[0]
    model = ModelRegistry.create(model_name)

Finding Transforms
^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core.registry import TransformRegistry
    from rheojax.core.inventory import TransformType

    # Find spectral transforms (Time <-> Frequency)
    transforms = TransformRegistry.find(type=TransformType.SPECTRAL)
    print(transforms)
    # Output: ['fft_analysis']

Core Definitions
----------------

Protocols (Models)
^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Protocol
     - Description
   * - ``FLOW_CURVE``
     - Steady shear viscosity vs. shear rate (Flow curves)
   * - ``CREEP``
     - Strain vs. time at constant stress
   * - ``RELAXATION``
     - Stress vs. time at constant strain
   * - ``STARTUP``
     - Stress growth vs. time at constant shear rate (Transient)
   * - ``OSCILLATION``
     - Small Amplitude Oscillatory Shear (SAOS) - G', G" vs. frequency
   * - ``LAOS``
     - Large Amplitude Oscillatory Shear - Lissajous curves, Harmonics

Transform Types
^^^^^^^^^^^^^^^

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Type
     - Description
   * - ``SPECTRAL``
     - Domain conversion (Time ↔ Frequency), e.g., FFT
   * - ``SUPERPOSITION``
     - Shifting data to create master curves, e.g., TTS, SRFS
   * - ``DECOMPOSITION``
     - Splitting signals into components, e.g., SPP
   * - ``ANALYSIS``
     - Extracting metrics or fingerprints, e.g., Mutation Number
   * - ``PROCESSING``
     - Data cleaning and smoothing, e.g., Smooth Derivative
