.. _gui-index:

====================
GUI Reference Guide
====================

RheoJAX provides an optional graphical user interface (GUI) built with PySide6/Qt6
for interactive rheological analysis workflows.

.. toctree::
   :maxdepth: 2
   :caption: GUI Documentation

   getting_started
   data_loading
   model_fitting
   bayesian_inference
   diagnostics
   transforms
   exporting
   keyboard_shortcuts

Overview
========

The RheoJAX GUI provides a visual interface for:

- **Data Loading**: Import rheological data from CSV, Excel, TRIOS, and other formats
- **Model Fitting**: Interactive NLSQ curve fitting with real-time visualization
- **Bayesian Inference**: MCMC sampling with ArviZ diagnostics plots
- **Transforms**: Apply mastercurve, FFT, and derivative transforms
- **Exporting**: Save results, figures, and reports in various formats

Installation
============

The GUI requires additional dependencies::

    pip install rheojax[gui]

Or install PySide6 separately::

    pip install PySide6

Launching the GUI
=================

From the command line::

    rheojax-gui

Or from Python::

    from rheojax.gui import main
    main()

Architecture
============

The GUI follows a modern architecture pattern:

- **State Management**: Redux-like centralized state store with undo/redo
- **Service Layer**: Business logic abstraction via services (ModelService, etc.)
- **Background Workers**: Non-blocking NLSQ/MCMC execution using QThreadPool
- **Signal-Slot**: Qt's reactive signal/slot pattern for UI updates

Key Components
--------------

**Pages** (Main Views):

- ``DataPage``: Data import and preview
- ``FitPage``: Model selection and NLSQ fitting
- ``BayesianPage``: MCMC configuration and sampling
- ``DiagnosticsPage``: MCMC diagnostics and ArviZ plots
- ``TransformPage``: Data transformation workflows
- ``ExportPage``: Results export and report generation

**Widgets** (Reusable Components):

- ``PlotCanvas``: Interactive matplotlib plotting with zoom/pan
- ``ParameterTable``: Model parameter editing with bounds
- ``ModelBrowser``: Hierarchical model selection tree
- ``ArvizCanvas``: ArviZ diagnostic plot integration
- ``ResidualsPanel``: Residual analysis visualizations
- ``MultiView``: Multi-panel comparison layouts

Performance
===========

The GUI is optimized for responsive interactions:

- Service operations: <100ms
- Model listing: <50ms
- Plot rendering: <500ms for 1000 points
- State updates: <5ms

Background workers ensure long-running MCMC operations don't freeze the UI.
