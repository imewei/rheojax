Quickstart Guide
================

This guide will get you started with rheo in minutes.

Basic Workflow
--------------

The Pipeline API provides the simplest way to analyze rheological data:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   # Create pipeline
   pipeline = Pipeline()

   # Load data (auto-detects format)
   pipeline.load('experiment.txt')

   # Fit model
   pipeline.fit('maxwell')

   # Plot results
   pipeline.plot()

   # Save results
   pipeline.save('results.hdf5')

Working with Models
-------------------

Direct Model Access
~~~~~~~~~~~~~~~~~~~

For more control, use the Modular API:

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.io import load_trios

   # Load data
   data = load_trios('experiment.txt')

   # Create and fit model
   model = Maxwell()
   model.fit(data.x, data.y)

   # Get parameters
   params = model.parameters
   print(f"G_s = {params.get_value('G_s'):.2e} Pa")
   print(f"eta_s = {params.get_value('eta_s'):.2e} Pa*s")

   # Make predictions
   predictions = model.predict(data.x)

Available Models
~~~~~~~~~~~~~~~~

Classical models:

* ``Maxwell``: Two-parameter viscoelastic model
* ``Zener``: Three-parameter standard linear solid
* ``SpringPot``: Two-parameter fractional element

Fractional models (11 variants available)

Non-Newtonian flow models:

* ``HerschelBulkley``, ``Bingham``, ``PowerLaw``
* ``CarreauYasuda``, ``Cross``, ``Casson``

Working with Transforms
-----------------------

Apply transforms to process raw data:

.. code-block:: python

   from rheojax.transforms import RheoAnalysis
   from rheojax.io import load_trios

   # Load raw time-series data
   data = load_trios('chirp_experiment.txt')

   # Apply FFT analysis
   analysis = RheoAnalysis()
   processed = analysis.transform(data)

   # Now data is in frequency domain with G', G''

Mastercurve Generation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import AutomatedMasterCurve

   # Load multi-temperature data
   data_list = [load_trios(f'temp_{t}C.txt') for t in [25, 35, 45, 55]]

   # Generate mastercurve
   mastercurve = AutomatedMasterCurve(reference_temperature=25)
   result = mastercurve.transform(data_list)

File I/O
--------

Supported Formats
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io import auto_load, load_trios, load_csv, load_excel

   # Auto-detect format
   data = auto_load('experiment.txt')

   # Explicit format
   data = load_trios('trios_file.txt')
   data = load_csv('data.csv', x_col='frequency', y_col='modulus')
   data = load_excel('results.xlsx', sheet='Sheet1')

Saving Results
~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io import save_hdf5, save_excel

   # Save data
   save_hdf5(data, 'output.h5')

   # Save results to Excel
   save_excel(results, 'report.xlsx', include_plots=True)

Next Steps
----------

* Read the :doc:`user_guide` for detailed documentation
* Explore :doc:`api_reference` for complete API documentation
* See example notebooks for advanced usage

