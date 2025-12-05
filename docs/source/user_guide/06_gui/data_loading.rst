.. _gui-data-loading:

============
Data Loading
============

The Data page provides tools for importing rheological data from various file formats.

Supported Formats
=================

RheoJAX GUI supports loading data from:

- **CSV** (``.csv``): Comma-separated values
- **Excel** (``.xlsx``, ``.xls``): Microsoft Excel workbooks
- **TRIOS** (``.tri``): TA Instruments TRIOS export files
- **Anton Paar** (``.txt``): Anton Paar rheometer exports
- **HDF5** (``.h5``, ``.hdf5``): Hierarchical Data Format

Loading Data
============

Drag and Drop
-------------

The simplest method:

1. Navigate to the **Data** page
2. Drag your data file into the drop zone
3. The file will be parsed automatically

File Browser
------------

Use the file browser:

1. Click **"Browse Files"** button
2. Navigate to your data file
3. Select and click **Open**

Column Mapping
==============

After loading, map columns to rheological variables:

Independent Variable (X)
------------------------

Select from:

- **Frequency (ω)**: For oscillation measurements
- **Time (t)**: For relaxation/creep measurements
- **Shear Rate (γ̇)**: For flow curve measurements

Dependent Variables (Y)
-----------------------

Select from:

- **G' (Storage Modulus)**: Elastic component
- **G'' (Loss Modulus)**: Viscous component
- **G(t) (Relaxation Modulus)**: Time-dependent modulus
- **η (Viscosity)**: Shear viscosity
- **J(t) (Creep Compliance)**: Creep response

Complex Data
------------

For complex modulus data (G* = G' + iG''):

1. Select **G'** for Y column
2. Select **G''** for Y2 column
3. Data will be handled as complex

Test Mode Detection
===================

RheoJAX automatically detects the test mode based on column names:

- **Oscillation**: Columns contain "omega", "freq", "G'", "G''"
- **Relaxation**: Columns contain "time", "G(t)", "relaxation"
- **Creep**: Columns contain "time", "J(t)", "compliance"
- **Flow**: Columns contain "shear rate", "viscosity", "stress"

Manual Override
---------------

If auto-detection fails, manually select the test mode from the dropdown.

Data Preview
============

After loading:

Preview Table
-------------

- View first 100 rows of data
- Check column assignments
- Verify data ranges

Preview Plot
------------

- Automatic visualization
- Log-log scaling for rheological data
- Multiple series support

Data Quality Checks
===================

The GUI performs automatic checks:

- **Range validation**: Ensures positive values for moduli
- **Monotonicity**: Checks frequency/time ordering
- **NaN/Inf detection**: Flags invalid values
- **Unit suggestions**: Recommends unit conversions

Unit Conversion
===============

Convert between common units:

Frequency
---------

- rad/s ↔ Hz
- rad/s ↔ 1/s

Modulus
-------

- Pa ↔ kPa ↔ MPa
- Pa ↔ dyn/cm²

Viscosity
---------

- Pa·s ↔ mPa·s (cP)
- Pa·s ↔ Poise

Multiple Datasets
=================

Load and manage multiple datasets:

1. Load additional files (each becomes a separate dataset)
2. Select active dataset from the sidebar list
3. Compare datasets in the multi-view panel

Dataset Metadata
================

Each dataset stores:

- **Source file**: Original file path
- **Load timestamp**: When data was imported
- **Column mappings**: Variable assignments
- **Test mode**: Measurement type
- **Units**: Physical units
- **Temperature**: Reference temperature (if available)

Export Preview Data
===================

Export the currently loaded data:

1. Right-click in preview table
2. Select **Export Preview**
3. Choose format (CSV, Excel)

Tips and Best Practices
=======================

1. **Use descriptive file names**: Include sample info and temperature
2. **Consistent units**: Keep all datasets in same units for comparison
3. **Log-scale data**: Ensure frequency/time spans multiple decades
4. **Quality check**: Review preview before proceeding to fitting
