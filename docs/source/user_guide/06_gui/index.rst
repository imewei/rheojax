.. _gui-index:

====================
GUI Reference Guide
====================

RheoJAX provides an optional graphical user interface (GUI) built with PySide6/Qt6
for interactive rheological analysis workflows.

.. toctree::
   :maxdepth: 2
   :caption: GUI Documentation

   workspace_getting_started
   getting_started
   data_loading
   model_fitting
   bayesian_inference
   diagnostics
   transforms
   exporting
   menu_reference
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

The GUI dependencies are included in the standard install::

    uv sync

Launching the GUI
=================

From the command line::

    rheojax-gui

Or from Python::

    from rheojax.gui import main
    main()

Architecture
============

The Workspace Shell uses the following architecture:

- **State Management**: Per-window ``AppState`` (``foundation/state.py``) with
  step invalidation cascades
- **Service Layer**: Business logic abstraction via services (ModelService, etc.)
- **Background Workers**: NLSQ/MCMC execution via OS subprocess isolation
- **Signal-Slot**: Qt's reactive signal/slot pattern for UI updates

Key Components
--------------

**Widgets** (Reusable Components):

- ``PlotCanvas``: Interactive matplotlib plotting with zoom/pan
- ``ParameterTable``: Model parameter editing with bounds
- ``ArvizCanvas``: ArviZ diagnostic plot integration
- ``ResidualsPanel``: Residual analysis visualizations

Performance
===========

The GUI is optimized for responsive interactions:

- Service operations: <100ms
- Model listing: <50ms
- Plot rendering: <500ms for 1000 points
- State updates: <5ms

Background workers ensure long-running MCMC operations don't freeze the UI.
