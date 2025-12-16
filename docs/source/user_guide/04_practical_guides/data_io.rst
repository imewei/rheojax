.. _data_io_guide:

Data I/O Guide
==============

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Load data from multiple instrument formats (TRIOS, Anton Paar, CSV, Excel)
   2. Use auto-detection for unknown file formats
   3. Save results in HDF5 and Excel formats
   4. Handle chunked reading for large files (>1GB)

.. admonition:: Prerequisites
   :class: important

   - Basic Python file I/O
   - :doc:`../02_model_usage/getting_started` — Model fitting workflow

Quick Reference
---------------

**Auto-detection** (recommended):

.. code-block:: python

   from rheojax.io.readers import auto_read

   data = auto_read('experiment.txt')  # Detects format automatically

**Specific readers**:

.. code-block:: python

   from rheojax.io.readers import read_trios, read_csv, read_excel, read_anton_paar

   data = read_trios('stress_relaxation.txt')
   data = read_csv('data.csv', x_column='Time', y_column='Stress')
   data = read_excel('results.xlsx', sheet_name='SAOS')

**Writers**:

.. code-block:: python

   from rheojax.io.writers import write_hdf5, write_excel

   write_hdf5(data, 'results.h5')  # Full fidelity
   write_excel(data, 'results.xlsx', sheet_name='Analysis')

Supported Formats
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Format
     - Reader Function
     - Notes
   * - TA Instruments TRIOS
     - ``load_trios()``
     - Auto-detects test mode, auto-chunks large files. See :doc:`trios_format`
   * - Anton Paar
     - ``read_anton_paar()``
     - RheoCompass export
   * - CSV
     - ``read_csv()``
     - Specify column names
   * - Excel
     - ``read_excel()``
     - .xlsx, .xls supported
   * - HDF5
     - ``read_hdf5()``
     - RheoJAX native format

Chunked Reading (Large Files)
------------------------------

For TRIOS files > 5 MB, auto-chunking is enabled by default (v0.4.0+):

.. code-block:: python

   from rheojax.io.readers import load_trios
   from rheojax.io.readers.trios import load_trios_chunked

   # Auto-chunking for files > 5 MB (default behavior)
   data = load_trios('large_file.txt')

   # Explicit chunked generator for memory-constrained processing
   for chunk in load_trios_chunked('large_file.txt', chunk_size=10000):
       process(chunk)

See :doc:`trios_format` for detailed chunked reading documentation.

Summary
-------

RheoJAX supports multiple instrument formats with auto-detection. Use ``auto_read()`` for convenience,
or specific readers for fine control. Save in HDF5 for full fidelity or Excel for sharing.

Further Reading
---------------

- :doc:`trios_format` — Detailed TRIOS file format documentation
- :doc:`data_formats` — Data format requirements for all analyses
- :doc:`/api/io` — I/O API reference
