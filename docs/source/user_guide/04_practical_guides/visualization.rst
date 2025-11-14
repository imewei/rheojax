.. _visualization_guide:

Visualization Guide
===================

.. admonition:: Learning Objectives
   :class: note

   After completing this section, you will be able to:

   1. Create publication-quality rheological plots
   2. Apply three visualization templates (default, publication, presentation)
   3. Customize plot styles and export for journals
   4. Use BayesianPlotter for MCMC diagnostics

.. admonition:: Prerequisites
   :class: important

   - :doc:`../02_model_usage/getting_started` — Basic fitting
   - :doc:`../03_advanced_topics/bayesian_inference` (for Bayesian plots)

Basic Plotting
--------------

**Automatic plot type detection**:

.. code-block:: python

   from rheojax.visualization import plot_rheo_data
   import matplotlib.pyplot as plt

   fig, ax = plot_rheo_data(data, style='publication')
   plt.savefig('figure.png', dpi=300)

**Templates**:

- ``style='default'``: Standard matplotlib
- ``style='publication'``: Journal-ready (larger fonts, clean)
- ``style='presentation'``: Slides (extra-large fonts, bold)

Bayesian Diagnostics
--------------------

.. code-block:: python

   from rheojax.visualization import BayesianPlotter

   plotter = BayesianPlotter(bayesian_result)
   plotter.plot_posterior()  # Posterior distributions
   plotter.plot_trace()       # MCMC traces
   plotter.plot_pair()        # Parameter correlations

Summary
-------

RheoJAX provides automatic plot type selection and three templates for different use cases.
BayesianPlotter offers comprehensive MCMC diagnostic visualizations.
