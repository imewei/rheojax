.. _gui-exporting:

=========
Exporting
=========

The Export page provides tools for saving results, figures, and generating reports.

Export Formats
==============

Data Formats
------------

**CSV** (``.csv``)
   Comma-separated values, universal compatibility.

   - Headers with column names
   - UTF-8 encoding
   - Compatible with Excel, Origin, etc.

**Excel** (``.xlsx``)
   Microsoft Excel workbook.

   - Multiple sheets per workbook
   - Formatting preserved
   - Direct editing capability

**HDF5** (``.h5``, ``.hdf5``)
   Hierarchical Data Format.

   - Efficient for large datasets
   - Preserves metadata
   - Scientific computing standard

**JSON** (``.json``)
   JavaScript Object Notation.

   - Human-readable
   - Easy programmatic access
   - Full metadata support

Figure Formats
--------------

**PNG** (``.png``)
   Portable Network Graphics.

   - Lossless compression
   - Web-compatible
   - Good for presentations

**PDF** (``.pdf``)
   Portable Document Format.

   - Vector graphics
   - Publication quality
   - Scalable without loss

**SVG** (``.svg``)
   Scalable Vector Graphics.

   - Vector format
   - Editable in Illustrator/Inkscape
   - Web-compatible

**EPS** (``.eps``)
   Encapsulated PostScript.

   - Publication standard
   - Vector format
   - LaTeX compatible

Exporting Data
==============

Fitted Parameters
-----------------

Export optimization results:

1. Select **Parameters** in Export section
2. Choose format (CSV, Excel, JSON)
3. Options:

   - Include uncertainties
   - Include bounds
   - Include correlation matrix

4. Click **Export**

Raw Data
--------

Export loaded datasets:

1. Select **Data** in Export section
2. Choose dataset(s)
3. Choose format
4. Options:

   - Original data
   - Transformed data
   - Both

5. Click **Export**

Posterior Samples
-----------------

Export Bayesian results:

1. Select **Posterior** in Export section
2. Choose format:

   - **CSV**: Flat sample array
   - **HDF5**: Full InferenceData
   - **NetCDF**: ArviZ native format

3. Options:

   - Include warmup
   - Include diagnostics

4. Click **Export**

Exporting Figures
=================

Current Plot
------------

Export the active plot:

1. Select **Figures** in Export section
2. Choose format (PNG, PDF, SVG)
3. Settings:

   - **DPI**: Resolution (150 default, 300 for print)
   - **Size**: Width x Height in inches
   - **Transparent**: Background transparency

4. Click **Export**

All Plots
---------

Export multiple plots at once:

1. Enable **Export All Plots**
2. Select which plots:

   - Data plot
   - Fit plot
   - Residuals
   - ArviZ diagnostics

3. Choose naming convention
4. Click **Export All**

Plot Customization
------------------

Before exporting, customize:

- **Title**: Add/modify plot title
- **Labels**: Axis labels
- **Legend**: Position and style
- **Style**: Choose from matplotlib styles

Report Generation
=================

Analysis Report
---------------

Generate comprehensive report:

1. Select **Report** in Export section
2. Choose format:

   - **HTML**: Interactive, web-viewable
   - **PDF**: Print-ready document
   - **Markdown**: For documentation

3. Select sections to include:

   - Executive summary
   - Data description
   - Model selection
   - Fit results
   - Bayesian analysis
   - Figures

4. Click **Generate Report**

Report Sections
---------------

**Summary**
   - Key findings
   - Best-fit parameters
   - Confidence intervals

**Data Section**
   - Source files
   - Data statistics
   - Quality metrics

**Model Section**
   - Model description
   - Physical interpretation
   - Parameter meanings

**Results Section**
   - Fit quality metrics
   - Parameter tables
   - Uncertainty analysis

**Figures Section**
   - Data and fit plots
   - Residual analysis
   - Bayesian diagnostics

**Appendix**
   - Full parameter tables
   - Correlation matrices
   - Raw data samples

Project Files
=============

Save Project
------------

Save complete session state:

1. **File > Save Project** or **Ctrl+S**
2. Choose location
3. Save as ``.rheojax`` project file

Project files contain:

- All loaded datasets
- Model configurations
- Fit results
- Bayesian results
- Transform history
- Settings

Load Project
------------

Resume previous work:

1. **File > Open Project** or **Ctrl+O**
2. Select ``.rheojax`` file
3. All state restored

Auto-Save
---------

Enable in preferences:

- Saves every 5 minutes
- Recovery after crash
- Temporary location

Batch Export
============

Export Multiple Items
---------------------

Select and export in batch:

1. Check multiple items
2. Choose common format
3. Set output directory
4. Click **Export Selected**

Export Templates
----------------

Save export configurations:

1. Configure export settings
2. **Save as Template**
3. Reuse with **Load Template**

Script Generation
-----------------

Generate Python script:

1. Select **Generate Script**
2. Creates Python code that reproduces:

   - Data loading
   - Model fitting
   - Export operations

Best Practices
==============

1. **Use project files**: Save `.rheojax` for session recovery
2. **Export incrementally**: Save results as you progress
3. **Use PDF for publications**: Vector format, high quality
4. **Include metadata**: Enable metadata export
5. **Version your exports**: Include date/version in filenames
6. **Backup raw data**: Keep originals separate
