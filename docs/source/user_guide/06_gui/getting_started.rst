.. _gui-getting-started:

===============
Getting Started
===============

This guide walks you through launching and using the RheoJAX GUI for the first time.

Prerequisites
=============

Ensure you have RheoJAX installed with GUI support::

    pip install rheojax[gui]

Verify the installation::

    python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"

Launching the Application
=========================

Command Line
------------

The simplest way to launch the GUI::

    rheojax-gui

Python API
----------

From within a Python script or interactive session::

    from rheojax.gui import main
    main()

First Launch
============

When you first launch RheoJAX GUI, you'll see:

1. **Home Tab**: Quick start guides and recent projects
2. **Navigation Sidebar**: Access to Data, Fit, Bayesian, Transform, Export pages
3. **Status Bar**: JAX device info and memory usage

The Main Window
===============

Layout Overview
---------------

::

    ┌─────────────────────────────────────────────────────────┐
    │  Menu Bar (File, Edit, View, Help)                     │
    ├───────┬─────────────────────────────────────────────────┤
    │       │                                                 │
    │  Nav  │           Main Content Area                     │
    │  Bar  │                                                 │
    │       │   (Data / Fit / Bayesian / Transform / Export)  │
    │       │                                                 │
    ├───────┴─────────────────────────────────────────────────┤
    │  Status Bar (Device: CPU/GPU | Memory | Progress)       │
    └─────────────────────────────────────────────────────────┘

Navigation Sidebar
------------------

- **Home**: Dashboard and quick start
- **Data**: Load and preview data files
- **Fit**: NLSQ model fitting
- **Bayesian**: MCMC inference with diagnostics
- **Transform**: Apply data transformations
- **Export**: Save results and generate reports

Quick Workflow
==============

A typical analysis workflow:

1. **Load Data**

   - Go to Data page
   - Drag-and-drop your data file
   - Map columns to X/Y variables
   - Select test mode (oscillation, relaxation, etc.)

2. **Select Model**

   - Go to Fit page
   - Browse available models
   - Select appropriate model for your data
   - Review default parameters

3. **Run NLSQ Fit**

   - Click "Fit Model"
   - Watch real-time progress
   - Review fit quality (R², residuals)

4. **Bayesian Inference** (Optional)

   - Go to Bayesian page
   - Configure MCMC settings
   - Run inference
   - Review ArviZ diagnostics

5. **Export Results**

   - Go to Export page
   - Select output formats
   - Generate reports

Settings and Preferences
========================

Access settings via **Edit > Preferences** or **Cmd/Ctrl + ,**:

- **Theme**: Light/Dark mode
- **Plot Style**: Default matplotlib styles
- **Auto-save**: Enable/disable project auto-save
- **Random Seed**: Set reproducibility seed

Keyboard Shortcuts
==================

Essential shortcuts:

- **Ctrl/Cmd + O**: Open file
- **Ctrl/Cmd + S**: Save project
- **Ctrl/Cmd + Z**: Undo
- **Ctrl/Cmd + Shift + Z**: Redo
- **Ctrl/Cmd + Q**: Quit

See :ref:`gui-keyboard-shortcuts` for the complete list.

Troubleshooting
===============

GUI Won't Launch
----------------

1. Verify PySide6 is installed::

       pip show PySide6

2. Check for Qt platform issues (Linux)::

       export QT_QPA_PLATFORM=xcb

3. Try reinstalling::

       pip uninstall PySide6
       pip install PySide6

Slow Performance
----------------

1. Check JAX device (prefer GPU when available)
2. Reduce data size for initial exploration
3. Close unnecessary ArviZ diagnostic plots

Display Issues
--------------

For high-DPI displays, set environment variable::

    export QT_AUTO_SCREEN_SCALE_FACTOR=1
