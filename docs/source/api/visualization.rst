Visualization (rheojax.visualization)
=======================================

The visualization module provides publication-quality plotting functions for rheological data.

Plotting Functions
------------------

.. automodule:: rheojax.visualization.plotter
   :members:
   :undoc-members:
   :show-inheritance:

Main Plotting Function
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rheojax.visualization.plotter.plot_rheo_data
   :noindex:

   Main entry point for plotting RheoData with automatic plot type selection.

   **Plot type selection logic:**

   1. Frequency domain or oscillation test → :func:`plot_frequency_domain`
   2. Rotation test or shear rate units → :func:`plot_flow_curve`
   3. Time domain → :func:`plot_time_domain`

Specialized Plot Types
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: rheojax.visualization.plotter.plot_time_domain
   :noindex:

   Plot time-domain data (relaxation, creep).

.. autofunction:: rheojax.visualization.plotter.plot_frequency_domain
   :noindex:

   Plot frequency-domain data (oscillatory tests).

   For complex data (G*), creates two subplots for G' and G".
   For real data, creates a single plot.

.. autofunction:: rheojax.visualization.plotter.plot_flow_curve
   :noindex:

   Plot flow curves (viscosity or stress vs shear rate).

.. autofunction:: rheojax.visualization.plotter.plot_residuals
   :noindex:

   Plot residuals from model fitting.

   If y_true and y_pred are provided, creates two subplots.
   Otherwise, plots residuals only.

Plotting Styles
---------------

Three built-in styles for different contexts:

.. data:: rheojax.visualization.plotter.DEFAULT_STYLE

   General-purpose style for interactive work.

   - Figure size: 8×6 inches
   - Font size: 11 pt
   - Line width: 1.5 pt

.. data:: rheojax.visualization.plotter.PUBLICATION_STYLE

   Optimized for journal publications.

   - Figure size: 6×4.5 inches
   - Font size: 10 pt
   - Line width: 1.2 pt
   - Smaller markers

.. data:: rheojax.visualization.plotter.PRESENTATION_STYLE

   Large, clear plots for presentations.

   - Figure size: 10×7 inches
   - Font size: 14 pt
   - Line width: 2.0 pt
   - Larger markers

Templates
---------

.. automodule:: rheojax.visualization.templates
   :members:
   :undoc-members:
   :show-inheritance:

The templates module provides reusable plot templates for common visualization tasks.
This will be expanded in Phase 2.

Examples
--------

Basic Plotting
~~~~~~~~~~~~~~

Auto-Detection
^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.core import RheoData
    from rheojax.visualization import plot_rheo_data
    import matplotlib.pyplot as plt
    import numpy as np

    # Create data
    time = np.logspace(-1, 2, 50)
    stress = 1000 * np.exp(-time / 5)
    data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

    # Plot with automatic type detection
    fig, ax = plot_rheo_data(data)
    plt.show()

Using Styles
^^^^^^^^^^^^

.. code-block:: python

    # Default style
    fig, ax = plot_rheo_data(data, style='default')

    # Publication style
    fig, ax = plot_rheo_data(data, style='publication')
    fig.savefig('figure.pdf', bbox_inches='tight')

    # Presentation style
    fig, ax = plot_rheo_data(data, style='presentation')

Time-Domain Plots
~~~~~~~~~~~~~~~~~

Stress Relaxation
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.visualization import plot_time_domain

    time = np.logspace(-1, 2, 50)
    stress = 1000 * np.exp(-time / 5)

    fig, ax = plot_time_domain(
        x=time,
        y=stress,
        x_units="s",
        y_units="Pa",
        log_x=True,
        log_y=False,
        style='publication'
    )

    ax.set_title("Stress Relaxation")
    plt.show()

Creep Compliance
^^^^^^^^^^^^^^^^

.. code-block:: python

    time = np.logspace(-1, 2, 50)
    compliance = 1e-4 * (1 + time**0.5)

    fig, ax = plot_time_domain(
        x=time,
        y=compliance,
        x_units="s",
        y_units="1/Pa",
        log_x=True,
        log_y=True,
        style='publication'
    )

    ax.set_title("Creep Compliance")
    ax.set_ylabel("J(t) (1/Pa)")
    plt.show()

Frequency-Domain Plots
~~~~~~~~~~~~~~~~~~~~~~

Complex Modulus
^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.visualization import plot_frequency_domain

    # Complex modulus
    omega = np.logspace(-2, 2, 50)
    Gp = 1000 * omega**0.5        # G'
    Gpp = 500 * omega**0.3        # G"
    G_star = Gp + 1j * Gpp

    fig, axes = plot_frequency_domain(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        style='publication'
    )

    # axes[0] is for G', axes[1] is for G"
    axes[0].set_title("Storage Modulus")
    axes[1].set_title("Loss Modulus")
    plt.show()

Single Modulus
^^^^^^^^^^^^^^

.. code-block:: python

    # Plot only G'
    fig, axes = plot_frequency_domain(
        x=omega,
        y=Gp,  # Real values
        x_units="rad/s",
        y_units="Pa"
    )

    # axes is a list with single element
    axes[0].set_title("Storage Modulus")
    plt.show()

Flow Curves
~~~~~~~~~~~

Viscosity vs Shear Rate
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.visualization import plot_flow_curve

    shear_rate = np.logspace(-2, 3, 50)
    viscosity = 100 * shear_rate**(-0.7)  # Shear thinning

    fig, ax = plot_flow_curve(
        x=shear_rate,
        y=viscosity,
        x_units="1/s",
        y_units="Pa.s",
        x_label="Shear Rate (1/s)",
        y_label="Viscosity (Pa·s)",
        style='publication'
    )

    ax.set_title("Flow Curve")
    plt.show()

Stress vs Shear Rate
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    stress = 50 * shear_rate**0.5  # Shear thickening

    fig, ax = plot_flow_curve(
        x=shear_rate,
        y=stress,
        x_units="1/s",
        y_units="Pa",
        y_label="Shear Stress (Pa)",
        style='publication'
    )

Residual Plots
~~~~~~~~~~~~~~

With Data Comparison
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from rheojax.visualization import plot_residuals

    # Experimental and predicted data
    time = np.linspace(0, 10, 50)
    stress_true = 1000 * np.exp(-time / 5)
    stress_pred = 980 * np.exp(-time / 4.8)
    residuals = stress_true - stress_pred

    fig, axes = plot_residuals(
        x=time,
        residuals=residuals,
        y_true=stress_true,
        y_pred=stress_pred,
        x_units="s",
        style='publication'
    )

    # axes[0]: Data and predictions
    # axes[1]: Residuals
    axes[0].set_title("Model Fit")
    axes[1].set_title("Residuals")
    plt.show()

Residuals Only
^^^^^^^^^^^^^^

.. code-block:: python

    fig, ax = plot_residuals(
        x=time,
        residuals=residuals,
        x_units="s"
    )

    ax.set_title("Residuals")
    plt.show()

Customization
-------------

Matplotlib Keyword Arguments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All plotting functions accept matplotlib kwargs:

.. code-block:: python

    fig, ax = plot_time_domain(
        x=time,
        y=stress,
        color='red',
        marker='s',
        markersize=8,
        linestyle='--',
        label='Experimental',
        alpha=0.7
    )

    ax.legend()

Advanced Styling
~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt

    # Custom figure size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Manual plotting with rheo data
    ax.semilogy(data.x, data.y, 'o-', label='Data')

    # Customize
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stress (Pa)', fontsize=14, fontweight='bold')
    ax.set_title('Custom Plot', fontsize=16)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=12)

    fig.tight_layout()

Multi-Panel Figures
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Relaxation
    axes[0, 0].semilogx(time, stress, 'o-')
    axes[0, 0].set_ylabel('Stress (Pa)')
    axes[0, 0].set_title('Relaxation')

    # Top-right: Creep
    axes[0, 1].loglog(time, compliance, 's-')
    axes[0, 1].set_ylabel('Compliance (1/Pa)')
    axes[0, 1].set_title('Creep')

    # Bottom-left: Oscillatory
    axes[1, 0].loglog(omega, Gp, 'o-', label="G'")
    axes[1, 0].loglog(omega, Gpp, 's-', label='G"')
    axes[1, 0].set_xlabel('ω (rad/s)')
    axes[1, 0].set_ylabel('Modulus (Pa)')
    axes[1, 0].legend()

    # Bottom-right: Flow curve
    axes[1, 1].loglog(shear_rate, viscosity, '^-')
    axes[1, 1].set_xlabel('Shear Rate (1/s)')
    axes[1, 1].set_ylabel('Viscosity (Pa·s)')

    fig.tight_layout()

Saving Figures
--------------

Raster Formats
~~~~~~~~~~~~~~

.. code-block:: python

    # PNG - high resolution
    fig.savefig('figure.png', dpi=300, bbox_inches='tight')

    # JPEG - smaller file
    fig.savefig('figure.jpg', dpi=150, quality=95)

Vector Formats
~~~~~~~~~~~~~~

.. code-block:: python

    # PDF - best for publications
    fig.savefig('figure.pdf', bbox_inches='tight')

    # SVG - editable
    fig.savefig('figure.svg', bbox_inches='tight')

    # EPS - for some journals
    fig.savefig('figure.eps', bbox_inches='tight')

Multiple Formats
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save in all formats
    for fmt in ['png', 'pdf', 'svg']:
        fig.savefig(f'figure.{fmt}', dpi=300, bbox_inches='tight')

Color and Markers
-----------------

Colors
~~~~~~

.. code-block:: python

    # Matplotlib colors
    ax.plot(x, y, color='C0')  # Default color cycle
    ax.plot(x, y, color='red')  # Named color
    ax.plot(x, y, color='#1f77b4')  # Hex color
    ax.plot(x, y, color=(0.2, 0.4, 0.6))  # RGB tuple

    # Colormaps
    colors = plt.cm.viridis(np.linspace(0, 1, n_datasets))

Markers
~~~~~~~

.. code-block:: python

    # Common markers
    ax.plot(x, y, 'o-')   # Circles
    ax.plot(x, y, 's-')   # Squares
    ax.plot(x, y, '^-')   # Triangles
    ax.plot(x, y, 'D-')   # Diamonds

    # Empty markers
    ax.plot(x, y, 'o', markerfacecolor='none', markeredgecolor='blue')

Annotations
-----------

.. code-block:: python

    # Text
    ax.text(x=5, y=500, s='Important region',
            fontsize=12, ha='center', va='center')

    # Arrow annotation
    ax.annotate(
        text='Transition',
        xy=(10, 800),
        xytext=(15, 900),
        arrowprops=dict(arrowstyle='->', color='red')
    )

    # Lines
    ax.axhline(y=500, color='r', linestyle='--')
    ax.axvline(x=10, color='g', linestyle=':')

    # Regions
    ax.axvspan(xmin=5, xmax=15, alpha=0.2, color='yellow')

Best Practices
--------------

Publication Quality
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use publication style
    fig, ax = plot_rheo_data(data, style='publication')

    # Save at 300 DPI
    fig.savefig('figure.pdf', dpi=300, bbox_inches='tight')

    # Use consistent fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['pdf.fonttype'] = 42  # TrueType fonts

Readability
~~~~~~~~~~~

.. code-block:: python

    # Clear labels with units
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Stress (Pa)', fontsize=12)

    # Grid for easier reading
    ax.grid(True, which='both', alpha=0.3, linestyle='--')

    # Legend when multiple series
    ax.legend(fontsize=10, framealpha=0.9)

Colorblind-Friendly
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use distinguishable colors
    colors = ['#0173B2', '#DE8F05', '#029E73']  # Blue, orange, green

    # Or use different markers
    markers = ['o', 's', '^']

    for i, (x, y) in enumerate(datasets):
        ax.plot(x, y, marker=markers[i], color=colors[i])

See Also
--------

- :doc:`../user_guide/visualization_guide` - Comprehensive visualization guide
- :doc:`core` - RheoData structure
- `Matplotlib documentation <https://matplotlib.org/>`_ - Advanced plotting
- `Matplotlib gallery <https://matplotlib.org/stable/gallery/index.html>`_ - Examples
