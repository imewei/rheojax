.. _practical_guides:

Section 4: Practical Guides (Weeks 13-16)
==========================================

**Workflows, data management, and production-ready analysis**

.. admonition:: Section Overview
   :class: note

   This section teaches production workflows using Pipeline APIs, data I/O for multiple
   instrument formats, publication-quality visualization, and batch processing.

   **Timeline**: Weeks 13-16 (12-16 hours)

   **Prerequisites**: Sections 1-2 (Fundamentals and Model Usage)

Learning Objectives
-------------------

By completing this section, you will be able to:

1. Build fluent analysis pipelines with Pipeline API
2. Use modular API for low-level control
3. Load data from TRIOS, Anton Paar, CSV, Excel formats
4. Create publication-quality visualizations
5. Process multiple datasets in batch mode

Section Contents
----------------

.. toctree::
   :maxdepth: 2

   pipeline_api
   modular_api
   data_io
   visualization
   batch_processing

Section Roadmap
---------------

**Week 13: Pipeline API**

- :doc:`pipeline_api` — Fluent workflows (load → fit → plot → save)

**Week 14: Data I/O**

- :doc:`data_io` — Auto-detect file formats, HDF5 output
- :doc:`modular_api` — Low-level control

**Week 15: Visualization**

- :doc:`visualization` — Publication-ready plots, templates

**Week 16: Batch Processing**

- :doc:`batch_processing` — High-throughput analysis

Quick Examples
--------------

**Pipeline API**:

.. code-block:: python

   from rheojax.pipeline import Pipeline

   (Pipeline()
       .load('data.csv', x_col='time', y_col='stress')
       .fit('maxwell')
       .plot()
       .save('results.hdf5'))

**Batch Processing**:

.. code-block:: python

   from rheojax.pipeline.batch import BatchPipeline

   batch = BatchPipeline()
   batch.load_directory('experiments/', pattern='*.txt')
   batch.fit_all('fractional_zener_ss')
   batch.save_summary('summary.xlsx')

Next Steps
----------

After completing practical guides, use:

**Section 5: Appendices** (:doc:`../05_appendices/index`)

Quick-reference materials for troubleshooting, experimental design, and glossary.
