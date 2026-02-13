.. _batch_processing:

Batch Processing
================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Process multiple datasets programmatically
   2. Use BatchPipeline for high-throughput analysis
   3. Export consolidated results to Excel
   4. Handle errors gracefully in batch mode

.. admonition:: Prerequisites
   :class: important

   - :doc:`pipeline_api` — Pipeline basics
   - :doc:`data_io` — File loading

Basic Batch Workflow
---------------------

.. code-block:: python

   from rheojax.pipeline.batch import BatchPipeline
   import pathlib

   # Load all files in directory
   batch = BatchPipeline()
   batch.load_directory('experiments/', pattern='*.txt')

   # Fit all with same model
   batch.fit_all('fractional_zener_ss')

   # Save results
   batch.save_summary('summary.xlsx')  # Parameter table
   batch.save_all_hdf5('results/')      # Individual HDF5 files

Manual Batch Processing
-----------------------

.. code-block:: python

   from rheojax.io.readers import auto_read
   from rheojax.models import Maxwell

   data_dir = pathlib.Path('experiments/')
   results = {}

   for file in data_dir.glob('*.txt'):
       try:
           data = auto_read(file)
           model = Maxwell()
           model.fit(data.x, data.y, test_mode=data.test_mode)

           results[file.stem] = {
               'G0': model.parameters.get_value('G0'),
               'eta': model.parameters.get_value('eta')
           }
       except Exception as e:
           print(f"Failed on {file}: {e}")

   # Save results
   import pandas as pd
   df = pd.DataFrame.from_dict(results, orient='index')
   df.to_excel('batch_results.xlsx')

Summary
-------

BatchPipeline automates processing of multiple datasets with consistent workflows.
Use for quality control, high-throughput screening, and comparative studies.

See ``examples/advanced/03-batch_processing.ipynb`` for complete examples.
