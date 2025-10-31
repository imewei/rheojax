Visualization Guide
===================

This guide covers creating publication-quality plots for rheological data using rheo's visualization tools.

Quick Start
-----------

The simplest way to plot data:

.. code-block:: python

    from rheojax.core import RheoData
    from rheojax.visualization import plot_rheo_data
    import matplotlib.pyplot as plt
    import numpy as np

    # Create or load data
    time = np.logspace(-1, 2, 50)
    stress = 1000 * np.exp(-time / 5)
    data = RheoData(x=time, y=stress, x_units="s", y_units="Pa", domain="time")

    # Plot with automatic type detection
    fig, ax = plot_rheo_data(data)
    plt.show()

Plotting Styles
---------------

rheo provides three built-in styles for different contexts:

Default Style
~~~~~~~~~~~~~

General purpose style for interactive work:

.. code-block:: python

    fig, ax = plot_rheo_data(data, style='default')

**Characteristics:**
- Figure size: 8×6 inches
- Font size: 11 pt
- Line width: 1.5 pt
- Good for Jupyter notebooks and interactive analysis

Publication Style
~~~~~~~~~~~~~~~~~

Optimized for journal publications:

.. code-block:: python

    fig, ax = plot_rheo_data(data, style='publication')

    # Save at publication quality
    fig.savefig('figure1.png', dpi=300, bbox_inches='tight')
    fig.savefig('figure1.pdf', bbox_inches='tight')  # Vector format

**Characteristics:**
- Figure size: 6×4.5 inches (fits journal column widths)
- Font size: 10 pt
- Line width: 1.2 pt
- Smaller markers for clarity
- Optimized for print reproduction

Presentation Style
~~~~~~~~~~~~~~~~~~

Large, clear plots for presentations and posters:

.. code-block:: python

    fig, ax = plot_rheo_data(data, style='presentation')

**Characteristics:**
- Figure size: 10×7 inches
- Font size: 14 pt
- Line width: 2.0 pt
- Larger markers and text for visibility
- Good for PowerPoint and posters

Plot Types
----------

rheo automatically selects the appropriate plot type based on data characteristics.

Time-Domain Plots
~~~~~~~~~~~~~~~~~

For relaxation and creep tests:

.. code-block:: python

    from rheojax.visualization import plot_time_domain

    # Stress relaxation
    time = np.logspace(-1, 2, 50)
    stress = 1000 * np.exp(-time / 5)

    fig, ax = plot_time_domain(
        x=time,
        y=stress,
        x_units="s",
        y_units="Pa",
        log_x=True,   # Logarithmic x-axis
        log_y=False,  # Linear y-axis
        style='publication'
    )

    ax.set_title("Stress Relaxation")
    plt.show()

**Use for:**
- Stress relaxation tests
- Creep compliance tests
- Step strain/stress experiments

Frequency-Domain Plots
~~~~~~~~~~~~~~~~~~~~~~

For oscillatory measurements (SAOS):

.. code-block:: python

    from rheojax.visualization import plot_frequency_domain

    # Complex modulus data
    omega = np.logspace(-2, 2, 50)
    Gp = 1000 * omega**0.5        # G' (storage modulus)
    Gpp = 500 * omega**0.3        # G" (loss modulus)
    G_star = Gp + 1j * Gpp

    fig, axes = plot_frequency_domain(
        x=omega,
        y=G_star,
        x_units="rad/s",
        y_units="Pa",
        style='publication'
    )

    # axes is an array [ax_Gp, ax_Gpp] for complex data
    axes[0].set_title("Storage Modulus")
    axes[1].set_title("Loss Modulus")
    plt.show()

**Complex data produces two subplots:**
- Top: G' (storage modulus)
- Bottom: G" (loss modulus)
- Both use log-log scales

**Real data produces single plot:**

.. code-block:: python

    # Real-valued frequency data
    fig, axes = plot_frequency_domain(
        x=omega,
        y=Gp,  # Real values only
        x_units="rad/s",
        y_units="Pa"
    )
    # axes is a list with single axes object

Flow Curves
~~~~~~~~~~~

For steady shear (rotation) tests:

.. code-block:: python

    from rheojax.visualization import plot_flow_curve

    # Viscosity vs shear rate
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

**Use for:**
- Viscosity vs shear rate
- Stress vs shear rate
- Power law fitting

Residual Plots
~~~~~~~~~~~~~~

Visualize model fitting quality:

.. code-block:: python

    from rheojax.visualization import plot_residuals

    # After model fitting (Phase 2 feature)
    x = time
    y_true = stress
    y_pred = model_predictions  # From fitted model
    residuals = y_true - y_pred

    # Plot with data comparison
    fig, axes = plot_residuals(
        x=x,
        residuals=residuals,
        y_true=y_true,
        y_pred=y_pred,
        x_units="s",
        style='publication'
    )

    # axes[0]: Data and predictions
    # axes[1]: Residuals
    plt.show()

Customization
-------------

All plotting functions accept matplotlib keyword arguments:

Basic Customization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    fig, ax = plot_time_domain(
        x=time,
        y=stress,
        color='red',
        marker='s',           # Square markers
        markersize=8,
        linestyle='--',       # Dashed line
        label='Experimental',
        alpha=0.7            # Transparency
    )

    ax.legend()
    ax.set_title("Custom Styled Plot")
    ax.set_xlim([0.1, 100])
    ax.set_ylim([0, 1200])

Advanced Styling
~~~~~~~~~~~~~~~~

Full matplotlib control:

.. code-block:: python

    import matplotlib.pyplot as plt
    from rheojax.visualization import plot_time_domain

    # Create figure with custom size
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot data manually
    ax.semilogy(time, stress, 'o-', label='Data')

    # Customize
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Stress (Pa)', fontsize=14, fontweight='bold')
    ax.set_title('Stress Relaxation Test', fontsize=16)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)
    ax.legend(fontsize=12, loc='upper right')

    # Adjust layout
    fig.tight_layout()

    # Save
    fig.savefig('custom_plot.png', dpi=300)

Multi-Panel Figures
-------------------

Create complex figures with multiple subplots:

Example 1: Compare Multiple Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    # Create data
    time = np.logspace(-1, 2, 50)
    stress1 = 1000 * np.exp(-time / 5)
    stress2 = 1200 * np.exp(-time / 3)
    stress3 = 800 * np.exp(-time / 7)

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(8, 10))

    # Plot each dataset
    axes[0].semilogx(time, stress1, 'o-')
    axes[0].set_ylabel('Stress (Pa)')
    axes[0].set_title('Sample A')
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogx(time, stress2, 's-', color='C1')
    axes[1].set_ylabel('Stress (Pa)')
    axes[1].set_title('Sample B')
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogx(time, stress3, '^-', color='C2')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Stress (Pa)')
    axes[2].set_title('Sample C')
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

Example 2: Different Test Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Relaxation (top-left)
    axes[0, 0].semilogx(time, stress, 'o-')
    axes[0, 0].set_ylabel('Stress (Pa)')
    axes[0, 0].set_title('Stress Relaxation')
    axes[0, 0].grid(True, alpha=0.3)

    # Creep (top-right)
    axes[0, 1].loglog(time, strain, 's-')
    axes[0, 1].set_ylabel('Strain')
    axes[0, 1].set_title('Creep Compliance')
    axes[0, 1].grid(True, alpha=0.3)

    # G' and G" (bottom-left)
    axes[1, 0].loglog(omega, Gp, 'o-', label="G'")
    axes[1, 0].loglog(omega, Gpp, 's-', label='G"')
    axes[1, 0].set_xlabel('Frequency (rad/s)')
    axes[1, 0].set_ylabel('Modulus (Pa)')
    axes[1, 0].set_title('Oscillatory Data')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Flow curve (bottom-right)
    axes[1, 1].loglog(shear_rate, viscosity, '^-')
    axes[1, 1].set_xlabel('Shear Rate (1/s)')
    axes[1, 1].set_ylabel('Viscosity (Pa·s)')
    axes[1, 1].set_title('Flow Curve')
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

Color and Markers
-----------------

Effective use of colors and markers:

Color Schemes
~~~~~~~~~~~~~

.. code-block:: python

    # Matplotlib default colors (C0, C1, C2, ...)
    ax.plot(x, y1, 'o-', color='C0', label='Dataset 1')
    ax.plot(x, y2, 's-', color='C1', label='Dataset 2')
    ax.plot(x, y3, '^-', color='C2', label='Dataset 3')

    # Custom colors
    ax.plot(x, y1, 'o-', color='#1f77b4')  # Hex color
    ax.plot(x, y2, 's-', color=(0.2, 0.4, 0.6))  # RGB tuple

    # Named colors
    ax.plot(x, y1, 'o-', color='darkblue')
    ax.plot(x, y2, 's-', color='crimson')

    # Colormaps for gradients
    colors = plt.cm.viridis(np.linspace(0, 1, len(datasets)))
    for i, (x, y) in enumerate(datasets):
        ax.plot(x, y, 'o-', color=colors[i])

Marker Styles
~~~~~~~~~~~~~

.. code-block:: python

    # Common markers
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']

    # Empty markers (better for overlapping data)
    ax.plot(x, y, 'o', markerfacecolor='none', markeredgecolor='blue', markersize=8)

    # Marker size scaling
    ax.plot(x, y, 'o', markersize=10)
    ax.scatter(x, y, s=50)  # s is marker area

Annotations
-----------

Add text and arrows to highlight features:

Text Annotations
~~~~~~~~~~~~~~~~

.. code-block:: python

    fig, ax = plot_time_domain(x=time, y=stress)

    # Add text annotation
    ax.text(
        x=10,
        y=500,
        s='Relaxation region',
        fontsize=12,
        ha='center',     # Horizontal alignment
        va='center',     # Vertical alignment
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Annotate specific point
    ax.annotate(
        text='Initial stress',
        xy=(0.1, 1000),           # Point to annotate
        xytext=(1, 1100),         # Text position
        arrowprops=dict(
            arrowstyle='->',
            connectionstyle='arc3,rad=0.3',
            color='red'
        ),
        fontsize=11
    )

Lines and Regions
~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Horizontal line
    ax.axhline(y=500, color='r', linestyle='--', label='Threshold')

    # Vertical line
    ax.axvline(x=10, color='g', linestyle=':', label='Transition')

    # Shaded region
    ax.axvspan(xmin=5, xmax=20, alpha=0.2, color='yellow', label='Region of interest')

    # Fill between curves
    ax.fill_between(x, y1, y2, alpha=0.3, label='Confidence interval')

Saving Figures
--------------

Export figures in various formats:

Raster Formats (PNG, JPG)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # PNG - best for web and presentations
    fig.savefig(
        'figure.png',
        dpi=300,              # High resolution
        bbox_inches='tight',  # Crop whitespace
        facecolor='white',    # White background
        edgecolor='none'
    )

    # JPEG - smaller file size
    fig.savefig('figure.jpg', dpi=150, quality=95)

Vector Formats (PDF, SVG)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # PDF - best for publications
    fig.savefig(
        'figure.pdf',
        bbox_inches='tight',
        transparent=True
    )

    # SVG - editable in Illustrator/Inkscape
    fig.savefig('figure.svg', bbox_inches='tight')

    # EPS - for some journals
    fig.savefig('figure.eps', bbox_inches='tight')

Multiple Formats
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save in multiple formats
    formats = ['png', 'pdf', 'svg']
    for fmt in formats:
        fig.savefig(f'figure.{fmt}', dpi=300, bbox_inches='tight')

Publication Guidelines
----------------------

Best practices for journal-quality figures:

Size and Resolution
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Journal column widths (approximate)
    single_column = 3.5  # inches
    double_column = 7.0  # inches

    # Create figure with correct width
    fig, ax = plt.subplots(figsize=(single_column, 3.0))

    # Set DPI
    # Screen: 96 dpi
    # Print: 300-600 dpi
    fig.savefig('figure.png', dpi=300)

Font and Text
~~~~~~~~~~~~~

.. code-block:: python

    # Use consistent, readable fonts
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10

    # Ensure text is editable in PDF
    plt.rcParams['pdf.fonttype'] = 42  # TrueType
    plt.rcParams['ps.fonttype'] = 42

Line Width and Markers
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Appropriate line widths
    ax.plot(x, y, linewidth=1.2)  # Not too thin or thick

    # Distinguishable markers
    ax.plot(x, y1, 'o-', markersize=5, markerfacecolor='none')
    ax.plot(x, y2, 's-', markersize=5, markerfacecolor='none')

Color Considerations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Use colorblind-friendly palettes
    # Good: blue-orange, purple-green
    # Avoid: red-green alone

    # Ensure readability in grayscale
    ax.plot(x, y1, 'o-', color='black', label='A')
    ax.plot(x, y2, 's--', color='gray', label='B')

Advanced Techniques
-------------------

Interactive Plots
~~~~~~~~~~~~~~~~~

.. code-block:: python

    import matplotlib.pyplot as plt
    %matplotlib widget  # In Jupyter

    # Create interactive plot
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')

    # Add zoom, pan tools automatically

3D Visualization (Future)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For multi-dimensional data (Phase 2+):

.. code-block:: python

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Frequency (rad/s)')
    ax.set_zlabel('Modulus (Pa)')

Animations
~~~~~~~~~~

Animate time-dependent data:

.. code-block:: python

    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    line, = ax.plot([], [], 'o-')

    def init():
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1000)
        return line,

    def update(frame):
        line.set_data(time[:frame], stress[:frame])
        return line,

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=len(time), interval=50, blit=True
    )

    # Save animation
    anim.save('relaxation.gif', writer='pillow', fps=30)

Templates
---------

Reusable plotting templates:

Template 1: Publication Figure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def publication_plot(data, title, filename):
        """Create publication-quality plot."""
        import matplotlib.pyplot as plt
        from rheojax.visualization import plot_rheo_data

        # Set style
        plt.rcParams['font.size'] = 10
        plt.rcParams['font.family'] = 'Arial'

        # Create figure
        fig, ax = plot_rheo_data(data, style='publication')

        # Customize
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')

        # Save
        fig.savefig(f'{filename}.pdf', bbox_inches='tight')
        fig.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')

        return fig, ax

    # Use template
    fig, ax = publication_plot(data, 'Stress Relaxation', 'figure1')

Template 2: Comparison Plot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    def compare_datasets(datasets, labels, title):
        """Compare multiple datasets on one plot."""
        fig, ax = plt.subplots(figsize=(8, 6))

        markers = ['o', 's', '^', 'v', 'D']
        colors = ['C0', 'C1', 'C2', 'C3', 'C4']

        for i, (data, label) in enumerate(zip(datasets, labels)):
            ax.plot(
                data.x, data.y,
                marker=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=label,
                markersize=6,
                markerfacecolor='none',
                linewidth=1.5
            )

        ax.set_xlabel(f'x ({datasets[0].x_units})')
        ax.set_ylabel(f'y ({datasets[0].y_units})')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        return fig, ax

Summary
-------

Key visualization features:

- **Automatic plot type selection** based on data domain and test mode
- **Three built-in styles**: default, publication, presentation
- **Full matplotlib customization** for complete control
- **Multi-panel figures** for complex comparisons
- **Publication-ready output** in multiple formats

For more information:

- :doc:`getting_started` - Basic plotting examples
- :doc:`core_concepts` - Understanding RheoData
- :doc:`../api/visualization` - Complete visualization API
- `Matplotlib documentation <https://matplotlib.org/>`_ - Advanced customization
