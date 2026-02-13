.. _trios_format:

TA Instruments TRIOS Format
===========================

.. admonition:: Purpose
   :class: note

   This reference documents the TRIOS file format from TA Instruments rheometers
   and how RheoJAX reads and processes these files.

Overview
--------

**TRIOS** is the software platform from **TA Instruments** for their Discovery
and ARES rheometers. RheoJAX reads TRIOS data exported as ``.txt`` files using
the "Export to LIMS" functionality.

Supported rheometers include:

- Discovery HR series (HR-1, HR-2, HR-3, HR-10, HR-20, HR-30)
- Discovery Hybrid Rheometer (DHR)
- ARES-G2
- RSA-G2 (DMA mode)

File Structure
--------------

TRIOS ``.txt`` files have a hierarchical structure with a header section followed
by one or more ``[step]`` data segments.

Example File Structure
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Filename        experiment_name.tri
   Instrument serial number    12345
   Instrument name     Discovery HR-2
   operator        John Doe
   rundate         12/15/2025
   Sample name     Polymer Sample
   Geometry name   40mm Cone
   Geometry type   Cone

   [step]
   Step name       Frequency sweep (25.0 °C)
   Number of points    50
   Variables   Angular frequency   Storage modulus   Loss modulus   Complex viscosity
              rad/s              Pa                Pa             Pa·s
   Data point  0.1                1000.5            500.2          11180.3
   Data point  0.2                1050.3            520.1          5862.4
   Data point  0.5                1120.8            580.5          2520.1
   ...

   [step]
   Step name       Flow curve (25.0 °C)
   Number of points    30
   Variables   Shear rate   Viscosity   Shear stress
              1/s          Pa·s        Pa
   Data point  0.01         1500.2      15.0
   Data point  0.1          1200.5      120.1
   ...

Header Section
~~~~~~~~~~~~~~

The header contains instrument and sample metadata:

.. list-table:: Header Fields
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``Filename``
     - Original TRIOS project filename (.tri)
   * - ``Instrument serial number``
     - Rheometer serial number
   * - ``Instrument name``
     - Rheometer model (e.g., Discovery HR-2)
   * - ``operator``
     - User who ran the experiment
   * - ``rundate``
     - Date of measurement
   * - ``Sample name``
     - User-defined sample identifier
   * - ``Geometry name``
     - Measurement geometry (e.g., 40mm Cone)
   * - ``Geometry type``
     - Geometry category (Cone, Plate, Couette, etc.)

Step Segments
~~~~~~~~~~~~~

Each ``[step]`` block contains:

1. **Step name**: Test type with optional temperature (e.g., "Frequency sweep (150.0 °C)")
2. **Number of points**: Total data rows in segment
3. **Column headers**: Tab-separated variable names
4. **Units row**: Tab-separated units for each column
5. **Data rows**: Tab-separated values with "Data point" prefix

Supported Test Types
--------------------

RheoJAX automatically detects and handles these TRIOS test types:

.. list-table:: Supported Test Types
   :header-rows: 1
   :widths: 25 25 25 25

   * - Test Type
     - X-axis
     - Y-axis
     - Domain
   * - **Frequency sweep (SAOS)**
     - Angular frequency (rad/s)
     - :math:`G'(\omega)`, :math:`G''(\omega)` → :math:`G^*(\omega)`
     - ``frequency``
   * - **Amplitude sweep**
     - Strain (%) or Stress (Pa)
     - :math:`G'(\gamma)`, :math:`G''(\gamma)`
     - ``frequency``
   * - **Flow ramp**
     - Shear rate (1/s)
     - Viscosity (Pa·s)
     - ``time`` (rotation)
   * - **Stress relaxation**
     - Time (s)
     - Stress (Pa) or G(t)
     - ``time``
   * - **Creep**
     - Time (s)
     - Strain or J(t)
     - ``time``
   * - **Temperature sweep**
     - Temperature (°C)
     - :math:`G'(T)`, :math:`G''(T)`
     - ``frequency``
   * - **Arbitrary wave (LAOS)**
     - Time (s)
     - Stress (Pa), Strain
     - ``time``

Loading TRIOS Files
-------------------

Basic Loading
~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io.readers import load_trios

   # Load single-segment file
   data = load_trios('frequency_sweep.txt')

   # Access data
   print(f"Points: {len(data.x)}")
   print(f"Domain: {data.domain}")
   print(f"Test mode: {data.test_mode}")

Multiple Segments
~~~~~~~~~~~~~~~~~

For files with multiple test steps:

.. code-block:: python

   # Return all segments as a list
   segments = load_trios('multi_step_experiment.txt', return_all_segments=True)

   for i, seg in enumerate(segments):
       print(f"Segment {i}: {seg.test_mode}, {len(seg.x)} points")
       if 'temperature' in seg.metadata:
           print(f"  Temperature: {seg.metadata['temperature']:.1f} K")

Accessing Oscillation Data
~~~~~~~~~~~~~~~~~~~~~~~~~~

For SAOS data, RheoJAX automatically constructs the complex modulus:

.. code-block:: python

   data = load_trios('frequency_sweep.txt')

   # Complex modulus G* = G' + iG''
   G_star = data.y  # Complex array

   # Individual components via properties
   G_prime = data.storage_modulus      # G' (Pa)
   G_double_prime = data.loss_modulus  # G'' (Pa)
   tan_delta = data.tan_delta          # G''/G'

   # Frequency
   omega = data.x  # Angular frequency (rad/s)

   # Metadata
   print(f"Sample: {data.metadata.get('sample_name', 'Unknown')}")
   print(f"Geometry: {data.metadata.get('geometry', 'Unknown')}")

Auto-Chunking for Large Files
-----------------------------

**New in v0.4.0**: Files larger than 5 MB are automatically loaded using chunked
reading for memory efficiency.

Memory Efficiency
~~~~~~~~~~~~~~~~~

.. list-table:: Memory Comparison
   :header-rows: 1
   :widths: 40 30 30

   * - Loading Method
     - Memory Usage
     - Speed
   * - Full loading
     - ~80 bytes/point
     - Faster
   * - Chunked loading
     - ~80 bytes × chunk_size
     - 2-4× slower

**Example**: A 150,000 point LAOS file:

- Full loading: ~12 MB in memory
- Chunked (10k): ~800 KB per chunk (50-87% reduction)

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Auto-chunking is enabled by default for files > 5 MB
   data = load_trios('large_file.txt')

   # Disable auto-chunking (force full loading)
   data = load_trios('large_file.txt', auto_chunk=False)

   # Explicit chunk size
   data = load_trios('large_file.txt', chunk_size=5000)

   # Progress tracking for large files
   def progress(current, total):
       pct = 100 * current / total
       print(f"Loading: {pct:.1f}%")

   data = load_trios('large_file.txt', progress_callback=progress)

Chunked Reading Generator
~~~~~~~~~~~~~~~~~~~~~~~~~

For processing large files without aggregation:

.. code-block:: python

   from rheojax.io.readers.trios import load_trios_chunked

   # Process in chunks (memory-efficient)
   for chunk in load_trios_chunked('large_laos_file.txt', chunk_size=10000):
       print(f"Processing {len(chunk.x)} points")
       # Each chunk is an independent RheoData object
       result = process_chunk(chunk)

   # Find maximum stress across entire file
   max_stress = -float('inf')
   for chunk in load_trios_chunked('file.txt'):
       max_stress = max(max_stress, chunk.y.max())

   # With progress tracking
   def progress(current, total):
       print(f"Progress: {100*current/total:.0f}%")

   for chunk in load_trios_chunked('file.txt', progress_callback=progress):
       process(chunk)

Column Detection
----------------

RheoJAX uses intelligent column detection to identify x and y variables.

Priority-Based Detection
~~~~~~~~~~~~~~~~~~~~~~~~

**X-axis priorities** (in order):

1. Angular frequency
2. Frequency
3. Shear rate
4. Temperature
5. Step time
6. Time
7. Strain

**Y-axis priorities** (in order):

1. Storage modulus (:math:`G'`)
2. Loss modulus (:math:`G''`)
3. Stress
4. Strain
5. Viscosity
6. Complex modulus
7. Complex viscosity
8. Torque
9. Normal stress

Complex Modulus Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When both Storage modulus (:math:`G'`) and Loss modulus (:math:`G''`) columns are present,
RheoJAX automatically constructs the complex modulus:

.. math::

   G^*(\omega) = G'(\omega) + i \cdot G''(\omega)

This enables direct use with RheoJAX models that expect complex input for
oscillation data.

Unit Conversions
----------------

RheoJAX automatically converts common units to SI base units:

.. list-table:: Automatic Unit Conversions
   :header-rows: 1
   :widths: 33 33 34

   * - From
     - To
     - Factor
   * - MPa
     - Pa
     - × 10⁶
   * - kPa
     - Pa
     - × 10³
   * - %
     - unitless
     - × 0.01

Temperature Extraction
----------------------

Temperature is automatically extracted from step names:

.. code-block:: text

   Step name       Frequency sweep (150.0 °C)

Becomes:

.. code-block:: python

   data.metadata['temperature']  # 423.15 K (converted to Kelvin)

This enables use with :doc:`/user_guide/03_advanced_topics/time_temperature_superposition`
for mastercurve construction.

Metadata Access
---------------

All TRIOS metadata is preserved in the ``metadata`` dictionary:

.. code-block:: python

   data = load_trios('experiment.txt')

   # Instrument metadata
   print(data.metadata.get('instrument_name'))      # 'Discovery HR-2'
   print(data.metadata.get('instrument_serial_number'))  # '12345'
   print(data.metadata.get('geometry'))             # '40mm Cone'
   print(data.metadata.get('geometry_type'))        # 'Cone'

   # Sample metadata
   print(data.metadata.get('sample_name'))          # 'Polymer Sample'
   print(data.metadata.get('operator'))             # 'John Doe'
   print(data.metadata.get('run_date'))             # '12/15/2025'

   # Step-specific metadata
   print(data.metadata.get('temperature'))          # 298.15 (Kelvin)
   print(data.metadata.get('test_mode'))            # 'oscillation'
   print(data.metadata.get('columns'))              # ['Angular frequency', ...]
   print(data.metadata.get('units'))                # ['rad/s', ...]

Complete Example
----------------

Frequency Sweep Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io.readers import load_trios
   from rheojax.models import FractionalMaxwellLiquid
   import matplotlib.pyplot as plt

   # Load TRIOS frequency sweep
   data = load_trios('polymer_frequency_sweep.txt')

   print(f"Sample: {data.metadata.get('sample_name')}")
   print(f"Temperature: {data.metadata.get('temperature', 298.15) - 273.15:.1f} °C")
   print(f"Points: {len(data.x)}")
   print(f"Frequency range: {data.x.min():.3f} - {data.x.max():.1f} rad/s")

   # Fit fractional Maxwell model
   model = FractionalMaxwellLiquid()
   model.fit(data)

   print(f"\nFitted parameters:")
   for name, param in model.parameters.items():
       print(f"  {name}: {param.value:.4g}")

   # Plot results
   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(data.x, data.storage_modulus, 'o', label="G' (data)")
   ax.loglog(data.x, data.loss_modulus, 's', label="G'' (data)")

   # Model prediction
   y_pred = model.predict(data.x)
   ax.loglog(data.x, y_pred.real, '-', label="G' (fit)")
   ax.loglog(data.x, y_pred.imag, '--', label="G'' (fit)")

   ax.set_xlabel('Angular frequency (rad/s)')
   ax.set_ylabel('Modulus (Pa)')
   ax.legend()
   plt.savefig('frequency_sweep_fit.png', dpi=150)

Multi-Temperature Mastercurve
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io.readers import load_trios
   from rheojax.transforms import Mastercurve

   # Load multi-temperature frequency sweeps
   # Each segment in the file is at a different temperature
   segments = load_trios('tts_experiment.txt', return_all_segments=True)

   print(f"Loaded {len(segments)} temperature segments:")
   for seg in segments:
       T = seg.metadata.get('temperature', 298.15)
       print(f"  {T - 273.15:.1f} °C: {len(seg.x)} points")

   # Construct mastercurve
   mc = Mastercurve(reference_temp=298.15, method='wlf')
   mastercurve, shift_factors = mc.transform(segments)

   print(f"\nShift factors (log10 aT):")
   for T, aT in shift_factors.items():
       print(f"  {T - 273.15:.1f} °C: {aT:.3f}")

Large LAOS File Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.io.readers.trios import load_trios_chunked
   from rheojax.transforms import SPPDecomposer

   # Process large LAOS file in chunks
   omega = 1.0  # rad/s (known from experiment)
   gamma_0 = 1.0  # strain amplitude

   spp = SPPDecomposer(omega=omega, gamma_0=gamma_0)

   # Process chunks and aggregate results
   all_stress = []
   all_time = []

   for chunk in load_trios_chunked('laos_150k_points.txt', chunk_size=10000):
       all_time.extend(chunk.x.tolist())
       all_stress.extend(chunk.y.tolist())

   print(f"Total points loaded: {len(all_stress)}")

   # Now analyze the complete waveform
   import numpy as np
   from rheojax.core.data import RheoData

   full_data = RheoData(
       x=np.array(all_time),
       y=np.array(all_stress),
       domain='time',
   )

   result = spp.transform(full_data)
   metrics = spp.get_results()
   print(f"Cage modulus: {metrics['G_cage']:.0f} Pa")
   print(f"Static yield stress: {metrics['sigma_sy']:.1f} Pa")

Troubleshooting
---------------

File Not Recognized
~~~~~~~~~~~~~~~~~~~

If RheoJAX cannot parse your TRIOS file:

1. **Check export format**: Use "Export to LIMS" in TRIOS (not "Export to Excel")
2. **Verify encoding**: File should be UTF-8 or ASCII
3. **Check for ``[step]`` markers**: Each data segment must start with ``[step]``

.. code-block:: python

   # Debug: Check file structure
   with open('problematic_file.txt', 'r') as f:
       for i, line in enumerate(f):
           if i < 20 or '[step]' in line.lower():
               print(f"{i}: {line.rstrip()}")

Wrong Columns Selected
~~~~~~~~~~~~~~~~~~~~~~

If RheoJAX selects the wrong x/y columns:

.. code-block:: python

   # Check detected columns
   data = load_trios('file.txt')
   print(f"Detected columns: {data.metadata.get('columns')}")
   print(f"Detected units: {data.metadata.get('units')}")

   # Manually specify columns using CSV reader
   from rheojax.io.readers import read_csv

   data = read_csv(
       'file.txt',
       x_column='Angular frequency',
       y_column='Storage modulus',
       delimiter='\t',
       skip_rows=15,  # Skip header
   )

Memory Issues with Large Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large files (> 100 MB):

.. code-block:: python

   # Force chunked reading with smaller chunks
   from rheojax.io.readers.trios import load_trios_chunked

   for chunk in load_trios_chunked('huge_file.txt', chunk_size=5000):
       # Process immediately, don't accumulate
       result = process_and_save(chunk)
       del chunk  # Explicit cleanup

Missing Temperature
~~~~~~~~~~~~~~~~~~~

If temperature is not detected:

.. code-block:: python

   data = load_trios('file.txt')

   # Check if temperature was extracted
   if 'temperature' not in data.metadata:
       # Manually add temperature (in Kelvin)
       data.metadata['temperature'] = 25.0 + 273.15

See Also
--------

- :doc:`data_formats` — Data format requirements for all analyses
- :doc:`data_io` — General data I/O guide
- :doc:`/user_guide/03_advanced_topics/time_temperature_superposition` — TTS with TRIOS data
- :doc:`/api/io` — Full API reference for I/O functions
