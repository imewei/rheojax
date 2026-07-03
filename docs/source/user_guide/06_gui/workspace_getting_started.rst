.. _gui-workspace-getting-started:

=========================
Workspace Shell (Default)
=========================

.. note::

   The Workspace Shell recently became the default RheoJAX GUI. Fuller,
   per-mode guides (Fit, Transform, Pipeline) are planned; this page covers
   orientation — launching, the three modes, and the project file model.

This guide covers the **Workspace Shell**, the mode-based GUI that
``rheojax-gui`` launches by default.

Prerequisites
=============

Ensure you have RheoJAX installed (GUI dependencies are included)::

    uv sync

Verify the installation::

    python -c "from PySide6.QtWidgets import QApplication; print('PySide6 OK')"

Launching the Application
==========================

Command Line
------------

The default launch opens the Workspace Shell::

    rheojax-gui

Other startup options::

    # Launch the legacy page-based window instead (instant rollback if
    # something isn't working right for you in the workspace shell)
    rheojax-gui --legacy

    # Open a saved project on startup
    rheojax-gui --project analysis.rheojax

    # Import raw data into the library on startup (preload only -- this
    # does not run a fit or transform automatically)
    rheojax-gui --import data.xlsx --protocol relaxation

    # Start maximized (useful on high-DPI desktops)
    rheojax-gui --maximized

Python API
----------

From within a Python script or interactive session::

    from rheojax.gui import main
    main()

The Workspace Window
=====================

The Workspace Shell is a single window with a mode toolbar at the top, a
dataset library on the left, the active mode's step sequence in the center,
and an inspector panel on the right. A status label in the toolbar shows
JAX device and float64 status.

Three Modes
===========

Switching modes (toolbar buttons) swaps the center step sequence. Each mode
is a linear, forward-unlocking sequence of steps: a step becomes reachable
once the previous step is both filled in and valid.

Fit
---

Single-dataset model fitting. Pick a protocol and model, select the
dataset to fit, run an NLSQ fit, optionally follow up with a Bayesian NUTS
run, visualize the result, and export it. This mirrors the legacy GUI's
Data → Fit → Bayesian → Export workflow, but as one guided sequence instead
of separate pages.

Transform
---------

Single-dataset transform application. Pick a transform (mastercurve, FFT,
derivative, etc.), configure its inputs, run it, visualize the transformed
data, and export the result.

Pipeline
--------

Batch orchestration across many datasets at once. Assemble a sequence of
transform, fit, and export steps, select which datasets in the library to
run them against, and click **Run All**. Pipeline mode has no per-step "run"
button by design — running a single step interactively is what Fit and
Transform modes are for; Pipeline mode is for repeating a fixed recipe over
many datasets in the background.

The File Menu and Projects
============================

The **File** menu provides **New**, **Open...**, **Save**, **Save As...**,
and **Close**. Save and Open read and write the ``.rheojax`` v2 project
format: a single archive file capturing the dataset library, fit/transform/
pipeline state, and job history, so a full analysis session can be closed
and reopened later.

The window tracks unsaved changes and prompts you to save before starting a
new project, opening another one, or closing, if anything is dirty. If
background jobs (an NLSQ/NUTS fit, a running pipeline batch) are still in
flight, it will also ask whether to cancel them before proceeding.

Falling Back to the Legacy Window
====================================

The Workspace Shell recently became the default GUI experience. If you hit
something that isn't working right for you, ``rheojax-gui --legacy`` is an
instant rollback to the previous page-based window — see
:doc:`getting_started` for that guide. The legacy window remains fully
functional; it is simply no longer the default.
