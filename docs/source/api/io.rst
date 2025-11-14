I/O Module (rheojax.io)
=======================

The I/O module provides readers and writers for various rheometer data formats.

Readers
-------

.. automodule:: rheojax.io.readers
   :members:
   :undoc-members:
   :show-inheritance:

Auto-Detection
~~~~~~~~~~~~~~

.. autofunction:: rheojax.io.readers.auto_load
   :noindex:

   Automatically detect and read file format based on extension and content.

   **Supported formats:**

   - TA Instruments TRIOS (.txt)
   - Generic CSV (.csv)
   - Microsoft Excel (.xlsx, .xls)
   - Anton Paar (.txt, .xls)

TRIOS Reader
~~~~~~~~~~~~

.. automodule:: rheojax.io.readers.trios
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.readers.trios.load_trios
   :noindex:

CSV Reader
~~~~~~~~~~

.. automodule:: rheojax.io.readers.csv_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.readers.csv_reader.load_csv
   :noindex:

Excel Reader
~~~~~~~~~~~~

.. automodule:: rheojax.io.readers.excel_reader
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.readers.excel_reader.load_excel
   :noindex:

Anton Paar Reader
~~~~~~~~~~~~~~~~~

.. automodule:: rheojax.io.readers.anton_paar
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.readers.anton_paar.load_anton_paar
   :noindex:

Writers
-------

.. automodule:: rheojax.io.writers
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Writer
~~~~~~~~~~~

.. automodule:: rheojax.io.writers.hdf5_writer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.writers.hdf5_writer.save_hdf5
   :noindex:

   Write RheoData to HDF5 file with full metadata preservation.

   **HDF5 Structure:**

   .. code-block:: text

       file.h5
       |-- x_data (dataset)         # Independent variable
       |-- y_data (dataset)         # Dependent variable
       |-- attributes/
       |   |-- x_units (attr)
       |   |-- y_units (attr)
       |   |-- domain (attr)
       |   \-- ... (metadata)

Excel Writer
~~~~~~~~~~~~

.. automodule:: rheojax.io.writers.excel_writer
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: rheojax.io.writers.excel_writer.save_excel
   :noindex:

   Write RheoData to Excel file.

   **Excel Structure:**

   - Column A: x values
   - Column B: y values (real part if complex)
   - Column C: y values (imaginary part if complex)

Examples
--------

Reading Data
~~~~~~~~~~~~

Auto-Detection
^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import auto_load

    # Automatically detect format
    data = auto_load("experiment.txt")
    data = auto_load("data.csv")
    data = auto_load("results.xlsx")

TRIOS Files
^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import load_trios

    # Read TRIOS file
    data = load_trios("stress_relaxation.txt")

    # Specify columns
    data = load_trios(
        "custom.txt",
        x_column="time",
        y_column="modulus"
    )

CSV Files
^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import load_csv

    # Simple CSV
    data = load_csv("data.csv")

    # Specify columns by name
    data = load_csv(
        "experiment.csv",
        x_column="Time (s)",
        y_column="Stress (Pa)"
    )

    # Specify columns by index
    data = load_csv(
        "results.csv",
        x_column=0,
        y_column=2,
        skiprows=5
    )

Excel Files
^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import load_excel

    # Read first sheet
    data = load_excel("results.xlsx")

    # Specify sheet
    data = load_excel(
        "experiment.xlsx",
        sheet_name="Frequency Sweep"
    )

    # Specify columns
    data = load_excel(
        "results.xlsx",
        sheet_name="Data",
        x_column="Frequency",
        y_column="G'"
    )

Anton Paar Files
^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import load_anton_paar

    # Read Anton Paar file
    data = load_anton_paar("oscillation.txt")

Writing Data
~~~~~~~~~~~~

HDF5 Format
^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.writers import save_hdf5

    # Write to HDF5
    save_hdf5(data, "results.h5")

    # With custom dataset name
    save_hdf5(data, "analysis.h5", dataset_name="relaxation_data")

    # Append to existing file
    save_hdf5(data, "results.h5", mode="a", dataset_name="test_2")

Reading HDF5 Back
^^^^^^^^^^^^^^^^^

.. code-block:: python

    import h5py
    from rheojax.core import RheoData

    with h5py.File("results.h5", "r") as f:
        x = f["x_data"][:]
        y = f["y_data"][:]
        x_units = f.attrs.get("x_units")
        y_units = f.attrs.get("y_units")
        domain = f.attrs.get("domain", "time")

    data = RheoData(x=x, y=y, x_units=x_units, y_units=y_units, domain=domain)

Excel Format
^^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.writers import save_excel

    # Write to Excel
    save_excel(data, "results.xlsx")

    # Specify sheet name
    save_excel(data, "analysis.xlsx", sheet_name="Stress Relaxation")

    # Append to existing file
    save_excel(data1, "results.xlsx", sheet_name="Test 1")
    save_excel(data2, "results.xlsx", sheet_name="Test 2", mode="a")

Batch Processing
~~~~~~~~~~~~~~~~

Convert Multiple Files
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from pathlib import Path
    from rheojax.io.readers import auto_load
    from rheojax.io.writers import save_hdf5

    # Convert all TXT to HDF5
    data_dir = Path("raw_data/")
    output_dir = Path("processed/")
    output_dir.mkdir(exist_ok=True)

    for txt_file in data_dir.glob("*.txt"):
        data = auto_load(txt_file)
        output_file = output_dir / f"{txt_file.stem}.h5"
        save_hdf5(data, output_file)
        print(f"Converted: {txt_file.name}")

Merge Multiple Tests
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.io.readers import auto_load
    from rheojax.io.writers import save_hdf5

    files = ["test1.txt", "test2.txt", "test3.txt"]

    for i, file in enumerate(files):
        data = auto_load(file)
        mode = "w" if i == 0 else "a"
        save_hdf5(data, "combined.h5", dataset_name=f"test_{i+1}", mode=mode)

Common Parameters
-----------------

Reader Parameters
~~~~~~~~~~~~~~~~~

All readers support these common parameters:

:param filepath: Path to input file
:type filepath: str | Path
:param x_column: Column for x data (name or index)
:type x_column: str | int | None
:param y_column: Column for y data (name or index)
:type y_column: str | int | None
:param skiprows: Number of rows to skip at beginning
:type skiprows: int
:param x_units: Override x-axis units
:type x_units: str | None
:param y_units: Override y-axis units
:type y_units: str | None
:param domain: Override domain ("time" or "frequency")
:type domain: str | None
:param metadata: Additional metadata dictionary
:type metadata: dict | None

Writer Parameters
~~~~~~~~~~~~~~~~~

**HDF5 Writer:**

:param data: RheoData object to write
:type data: RheoData
:param filepath: Output file path
:type filepath: str | Path
:param dataset_name: Name for dataset in HDF5 file
:type dataset_name: str
:param mode: File mode ("w" for write, "a" for append)
:type mode: str

**Excel Writer:**

:param data: RheoData object to write
:type data: RheoData
:param filepath: Output file path
:type filepath: str | Path
:param sheet_name: Excel sheet name
:type sheet_name: str
:param mode: File mode ("w" for write, "a" for append)
:type mode: str

File Format Notes
-----------------

TRIOS Format
~~~~~~~~~~~~

TA Instruments TRIOS export format characteristics:

- Plain text with extensive headers
- Metadata in header section
- Tab or space-separated columns
- May contain multiple datasets in one file
- Units typically in column headers

**Example:**

.. code-block:: text

    TA Instruments - TRIOS
    Version 5.1.1
    Test: Stress Relaxation

    time (s)    stress (Pa)    strain
    0.1         1000.5         0.01
    0.5         850.2          0.01

CSV Format
~~~~~~~~~~

Generic comma-separated or tab-separated format:

- Flexible delimiter (auto-detected)
- First row usually contains headers
- Units may be in headers: "Time (s)", "Stress (Pa)"
- No metadata preservation (use HDF5 for that)

**Example:**

.. code-block:: text

    Time (s),Stress (Pa),Strain
    0.1,1000.5,0.01
    0.5,850.2,0.01

Excel Format
~~~~~~~~~~~~

Microsoft Excel (.xlsx, .xls):

- Multiple sheets supported
- First row typically contains headers
- Can include metadata in separate sheets
- Good for sharing with collaborators

HDF5 Format
~~~~~~~~~~~

Hierarchical Data Format 5:

- Binary format (compact, fast)
- Preserves all metadata
- Supports complex data structures
- Cross-platform compatibility
- Industry standard for scientific data
- **Recommended** for long-term storage

Anton Paar Format
~~~~~~~~~~~~~~~~~

Anton Paar rheometer export format:

- Text or Excel format
- Manufacturer-specific headers
- Comprehensive metadata
- Multiple test types supported

Error Handling
--------------

All readers raise appropriate exceptions:

.. code-block:: python

    from rheojax.io.readers import auto_load

    try:
        data = auto_load("experiment.txt")
    except FileNotFoundError:
        print("File not found")
    except ValueError as e:
        print(f"Invalid data: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

See Also
--------

- :doc:`../user_guide/io_guide` - Comprehensive I/O guide
- :doc:`core` - RheoData structure
- :doc:`../user_guide/getting_started` - Quick start examples
