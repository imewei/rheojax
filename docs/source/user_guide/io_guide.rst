Data I/O Guide
==============

This guide covers reading data from various rheometer formats and writing analysis results.

Reading Data
------------

rheo supports multiple file formats from common rheometer manufacturers with automatic format detection.

Auto-Detection
~~~~~~~~~~~~~~

The simplest way to read data is using auto-detection:

.. code-block:: python

    from rheojax.io.readers import auto_read

    # Automatically detect and read file
    data = auto_read("experiment.txt")

    # Works with various formats
    data = auto_read("data.csv")
    data = auto_read("results.xlsx")
    data = auto_read("test.xls")

The auto-reader examines file extension and content to select the appropriate reader.

TRIOS Files (TA Instruments)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Read TA Instruments TRIOS export files:

.. code-block:: python

    from rheojax.io.readers import read_trios

    # Read TRIOS text file
    data = read_trios("stress_relaxation.txt")

    # Specify columns if non-standard
    data = read_trios(
        "custom.txt",
        x_column="time",
        y_column="modulus"
    )

**Supported TRIOS formats:**

- Stress relaxation tests
- Creep tests
- Oscillatory frequency sweeps
- Flow curves
- Time sweeps

**Example TRIOS file format:**

.. code-block:: text

    TA Instruments - TRIOS
    Version 5.1.1

    Test: Stress Relaxation
    Sample: Polymer A

    time (s)    stress (Pa)    strain
    0.1         1000.5         0.01
    0.5         850.2          0.01
    1.0         720.1          0.01

CSV Files
~~~~~~~~~

Read generic CSV files:

.. code-block:: python

    from rheojax.io.readers import read_csv

    # Simple CSV with default columns (first two)
    data = read_csv("data.csv")

    # Specify columns by name
    data = read_csv(
        "experiment.csv",
        x_column="Time (s)",
        y_column="Stress (Pa)"
    )

    # Specify columns by index
    data = read_csv(
        "results.csv",
        x_column=0,
        y_column=2  # Skip column 1
    )

    # Skip header rows
    data = read_csv(
        "data.csv",
        skiprows=5,
        delimiter=",",
        x_column="time",
        y_column="modulus"
    )

**CSV format examples:**

.. code-block:: text

    # Example 1: Simple format
    time,stress
    0.1,1000
    0.5,850
    1.0,720

    # Example 2: With units in header
    Time (s),Stress (Pa),Strain
    0.1,1000.5,0.01
    0.5,850.2,0.01

    # Example 3: Tab-separated
    time	modulus	phase
    0.1	1000	15.2
    1.0	950	18.5

Excel Files
~~~~~~~~~~~

Read Microsoft Excel files:

.. code-block:: python

    from rheojax.io.readers import read_excel

    # Read first sheet
    data = read_excel("results.xlsx")

    # Specify sheet by name
    data = read_excel(
        "experiment.xlsx",
        sheet_name="Frequency Sweep"
    )

    # Specify sheet by index
    data = read_excel(
        "data.xlsx",
        sheet_name=0  # First sheet
    )

    # Specify columns
    data = read_excel(
        "results.xlsx",
        sheet_name="Data",
        x_column="Frequency",
        y_column="G'"
    )

    # Skip rows
    data = read_excel(
        "data.xlsx",
        skiprows=10,
        x_column=0,
        y_column=1
    )

**Excel file structure:**

- Supports both .xlsx and .xls formats
- Can have multiple sheets
- Headers can be in any row (use `skiprows`)
- Supports complex data (for G* = G' + iG")

Anton Paar Files
~~~~~~~~~~~~~~~~

Read Anton Paar rheometer files:

.. code-block:: python

    from rheojax.io.readers import read_anton_paar

    # Read Anton Paar file
    data = read_anton_paar("oscillation.txt")

    # With custom columns
    data = read_anton_paar(
        "flow_curve.txt",
        x_column="Shear Rate",
        y_column="Viscosity"
    )

**Supported Anton Paar formats:**

- Oscillatory tests (frequency and strain sweeps)
- Rotational tests (flow curves)
- Time-dependent tests

Reading Complex Data
~~~~~~~~~~~~~~~~~~~~

For frequency-domain measurements with G' and G":

.. code-block:: python

    from rheojax.io.readers import read_csv
    import pandas as pd
    import numpy as np

    # Option 1: Read from separate columns
    df = pd.read_csv("oscillation.csv")
    omega = df["Frequency (rad/s)"].values
    Gp = df["G' (Pa)"].values
    Gpp = df['G" (Pa)'].values
    G_star = Gp + 1j * Gpp

    from rheojax.core import RheoData
    data = RheoData(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        domain="frequency"
    )

    # Option 2: Custom reader for specific format
    # (Implementation depends on file structure)

Common Reader Options
~~~~~~~~~~~~~~~~~~~~~

All readers support these common parameters:

.. code-block:: python

    data = read_csv(
        filepath="data.csv",
        x_column=0,              # Column for x data (name or index)
        y_column=1,              # Column for y data (name or index)
        skiprows=0,              # Rows to skip at beginning
        delimiter=",",           # Column delimiter (CSV only)
        x_units=None,            # Override x units
        y_units=None,            # Override y units
        domain=None,             # Override domain ("time" or "frequency")
        metadata=None            # Additional metadata dict
    )

Writing Data
------------

Save analysis results in various formats.

HDF5 Format
~~~~~~~~~~~

HDF5 is the recommended format for preserving all data and metadata:

.. code-block:: python

    from rheojax.io.writers import write_hdf5

    # Write RheoData to HDF5
    write_hdf5(data, "results.h5")

    # With custom dataset name
    write_hdf5(data, "analysis.h5", dataset_name="relaxation_data")

    # Append to existing file
    write_hdf5(data, "results.h5", mode="a", dataset_name="test_2")

**HDF5 structure:**

.. code-block:: text

    results.h5
    ├── x_data              # Independent variable array
    ├── y_data              # Dependent variable array
    ├── metadata/           # Group for metadata
    │   ├── x_units         # String attribute
    │   ├── y_units         # String attribute
    │   ├── domain          # String attribute
    │   └── ...             # Other metadata
    └── attributes/         # Additional attributes

**Reading HDF5 files:**

.. code-block:: python

    import h5py
    from rheojax.core import RheoData
    import numpy as np

    with h5py.File("results.h5", "r") as f:
        x = f["x_data"][:]
        y = f["y_data"][:]
        x_units = f.attrs.get("x_units")
        y_units = f.attrs.get("y_units")
        domain = f.attrs.get("domain", "time")

    data = RheoData(x=x, y=y, x_units=x_units, y_units=y_units, domain=domain)

Excel Format
~~~~~~~~~~~~

Write to Excel for easy sharing:

.. code-block:: python

    from rheojax.io.writers import write_excel

    # Write to Excel
    write_excel(data, "results.xlsx")

    # Specify sheet name
    write_excel(data, "analysis.xlsx", sheet_name="Stress Relaxation")

    # Write multiple datasets to one file
    write_excel(data1, "results.xlsx", sheet_name="Test 1")
    write_excel(data2, "results.xlsx", sheet_name="Test 2", mode="a")

**Excel output structure:**

.. code-block:: text

    Column A: x values
    Column B: y values (real part if complex)
    Column C: y values (imaginary part if complex)

    Metadata in separate sheet or as comments

CSV Export
~~~~~~~~~~

Export to CSV for use in other software:

.. code-block:: python

    import pandas as pd
    import numpy as np

    # Create DataFrame
    df = pd.DataFrame({
        f'x ({data.x_units})': data.x,
        f'y ({data.y_units})': np.real(data.y)
    })

    # For complex data
    if np.iscomplexobj(data.y):
        df[f'y_real ({data.y_units})'] = np.real(data.y)
        df[f'y_imag ({data.y_units})'] = np.imag(data.y)

    # Write to CSV
    df.to_csv("output.csv", index=False)

Batch Processing
----------------

Process multiple files efficiently:

Example 1: Convert All Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pathlib
    from rheojax.io.readers import auto_read
    from rheojax.io.writers import write_hdf5

    # Convert all TXT files to HDF5
    data_dir = pathlib.Path("raw_data/")
    output_dir = pathlib.Path("processed_data/")
    output_dir.mkdir(exist_ok=True)

    for txt_file in data_dir.glob("*.txt"):
        try:
            # Read data
            data = auto_read(txt_file)

            # Write to HDF5
            output_file = output_dir / f"{txt_file.stem}.h5"
            write_hdf5(data, output_file)

            print(f"Converted: {txt_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")

Example 2: Merge Multiple Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.io.readers import auto_read
    from rheojax.io.writers import write_hdf5

    # Read multiple tests
    files = ["test1.txt", "test2.txt", "test3.txt"]
    output_file = "combined.h5"

    for i, file in enumerate(files):
        data = auto_read(file)
        dataset_name = f"test_{i+1}"
        mode = "w" if i == 0 else "a"  # Overwrite first, append rest
        write_hdf5(data, output_file, dataset_name=dataset_name, mode=mode)

Example 3: Extract Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from rheojax.io.readers import auto_read

    # Extract metadata from all files
    files = pathlib.Path("experiments/").glob("*.txt")
    metadata_list = []

    for file in files:
        data = auto_read(file)
        metadata_list.append({
            'filename': file.name,
            'test_mode': str(data.test_mode),
            'n_points': len(data.x),
            'x_range': f"{data.x.min():.2f} - {data.x.max():.2f}",
            'y_range': f"{data.y.min():.2f} - {data.y.max():.2f}",
            'x_units': data.x_units,
            'y_units': data.y_units,
        })

    # Create summary DataFrame
    summary = pd.DataFrame(metadata_list)
    summary.to_excel("metadata_summary.xlsx", index=False)
    print(summary)

Working with Metadata
---------------------

Preserve and enhance experimental metadata:

Adding Metadata
~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.io.readers import auto_read

    # Read data
    data = auto_read("experiment.txt")

    # Add experimental conditions
    data.metadata.update({
        'temperature': 25.0,
        'temperature_units': '°C',
        'sample_id': 'PMMA-001',
        'operator': 'John Doe',
        'date': '2024-10-24',
        'instrument': 'ARES-G2',
        'geometry': '25mm parallel plate',
        'gap': 1.0,
        'gap_units': 'mm'
    })

    # Save with metadata
    from rheojax.io.writers import write_hdf5
    write_hdf5(data, "annotated_results.h5")

Reading Metadata
~~~~~~~~~~~~~~~~

.. code-block:: python

    import h5py

    # Read from HDF5
    with h5py.File("results.h5", "r") as f:
        # Access metadata
        temp = f.attrs.get("temperature")
        sample = f.attrs.get("sample_id")

        # List all metadata
        print("Metadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

Metadata Best Practices
~~~~~~~~~~~~~~~~~~~~~~~

1. **Use Standard Keys**

   .. code-block:: python

       standard_metadata = {
           'temperature': 25.0,
           'temperature_units': '°C',
           'sample': 'Sample ID',
           'operator': 'Name',
           'date': 'YYYY-MM-DD',
           'instrument': 'Model',
           'test_mode': 'relaxation',  # Explicit override
       }

2. **Include Units**

   .. code-block:: python

       data.metadata.update({
           'gap': 1.0,
           'gap_units': 'mm',
           'frequency': 1.0,
           'frequency_units': 'Hz'
       })

3. **Document Data Processing**

   .. code-block:: python

       data.metadata['processing'] = {
           'smoothing': 'moving_average',
           'window_size': 5,
           'timestamp': '2024-10-24 10:30:00'
       }

Error Handling
--------------

Handle common I/O errors gracefully:

.. code-block:: python

    from rheojax.io.readers import auto_read
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def safe_read(filepath):
        """Safely read file with error handling."""
        try:
            data = auto_read(filepath)
            logger.info(f"Successfully read {filepath}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None
        except ValueError as e:
            logger.error(f"Invalid data in {filepath}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error reading {filepath}: {e}")
            return None

    # Use safe reader
    data = safe_read("experiment.txt")
    if data is not None:
        # Process data
        pass

Format-Specific Tips
--------------------

TRIOS Files
~~~~~~~~~~~

- TRIOS files often have extensive headers; the reader automatically skips them
- Multi-step tests create separate datasets
- Check metadata for test parameters

.. code-block:: python

    data = read_trios("multistep.txt")
    print(data.metadata)  # Contains test parameters from file

Excel Files
~~~~~~~~~~~

- Use meaningful sheet names
- Put metadata in a separate "Info" sheet
- Use first row for headers with units

.. code-block:: python

    # Good Excel structure:
    # Sheet "Data": omega (rad/s) | G' (Pa) | G" (Pa)
    # Sheet "Info": temperature | sample | date | ...

CSV Files
~~~~~~~~~

- Include units in column headers: "Time (s)", "Stress (Pa)"
- Use consistent delimiter (comma or tab)
- Avoid special characters in headers

.. code-block:: python

    # Good CSV header:
    # time (s),stress (Pa),strain
    # 0.1,1000.5,0.01

Advanced Topics
---------------

Custom Readers
~~~~~~~~~~~~~~

Create custom readers for proprietary formats:

.. code-block:: python

    import numpy as np
    from rheojax.core import RheoData

    def read_custom_format(filepath):
        """Read custom rheometer format."""
        with open(filepath, 'r') as f:
            # Parse file
            lines = f.readlines()

            # Extract header information
            x_units = "s"
            y_units = "Pa"

            # Parse data (example)
            data_lines = [l.strip().split() for l in lines[10:]]
            x = np.array([float(l[0]) for l in data_lines])
            y = np.array([float(l[1]) for l in data_lines])

        return RheoData(
            x=x,
            y=y,
            x_units=x_units,
            y_units=y_units,
            domain="time"
        )

    # Use custom reader
    data = read_custom_format("custom_file.dat")

Streaming Large Files
~~~~~~~~~~~~~~~~~~~~~

For very large files, process in chunks:

.. code-block:: python

    import pandas as pd
    from rheojax.core import RheoData

    def read_large_csv_chunked(filepath, chunksize=10000):
        """Read large CSV in chunks."""
        chunks = []

        for chunk in pd.read_csv(filepath, chunksize=chunksize):
            # Process chunk
            chunks.append(chunk)

        # Combine
        df = pd.concat(chunks, ignore_index=True)

        return RheoData(
            x=df.iloc[:, 0].values,
            y=df.iloc[:, 1].values
        )

Data Validation
~~~~~~~~~~~~~~~

Validate data after reading:

.. code-block:: python

    def validate_rheo_data(data):
        """Validate RheoData for common issues."""
        issues = []

        # Check for NaN
        if np.any(np.isnan(data.x)) or np.any(np.isnan(data.y)):
            issues.append("Data contains NaN values")

        # Check monotonicity
        if not (np.all(np.diff(data.x) > 0) or np.all(np.diff(data.x) < 0)):
            issues.append("x-axis is not monotonic")

        # Check for duplicates
        if len(np.unique(data.x)) < len(data.x):
            issues.append("x-axis contains duplicate values")

        # Check data range
        if len(data.x) < 10:
            issues.append("Insufficient data points (< 10)")

        return issues

    # Validate after reading
    data = auto_read("experiment.txt")
    issues = validate_rheo_data(data)
    if issues:
        for issue in issues:
            print(f"Warning: {issue}")

Summary
-------

Key I/O capabilities:

- **Multiple formats**: TRIOS, CSV, Excel, Anton Paar with auto-detection
- **Flexible reading**: Column selection, header skipping, custom parameters
- **Multiple outputs**: HDF5 (full fidelity), Excel (sharing), CSV (compatibility)
- **Batch processing**: Convert, merge, and analyze multiple files
- **Metadata preservation**: Maintain experimental context

For more information:

- :doc:`getting_started` - Basic I/O examples
- :doc:`core_concepts` - RheoData structure and metadata
- :doc:`../api/io` - Complete I/O API reference
