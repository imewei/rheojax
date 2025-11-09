Transform Usage Guide
=====================

Data transforms are essential preprocessing and analysis tools for rheological data. This guide explains when and how to use each of the five transforms implemented in rheojax.

Overview of Transforms
-----------------------

rheo provides five powerful transforms:

1. **FFTAnalysis**: Time → frequency domain conversion for oscillatory data
2. **Mastercurve**: Time-temperature superposition with WLF/Arrhenius
3. **MutationNumber**: Quantify viscoelastic character and structural evolution
4. **OWChirp**: Optimal waveform analysis for Large Amplitude Oscillatory Shear (LAOS)
5. **SmoothDerivative**: Noise-robust differentiation for calculating rates

Quick Reference Table
~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Transform Selection Guide
   :header-rows: 1
   :widths: 20 30 25 25

   * - Transform
     - Input Data
     - Output Data
     - Primary Use Case
   * - **FFTAnalysis**
     - Time-series stress/strain
     - Frequency-domain G', G''
     - Convert arbitrary waveform to moduli
   * - **Mastercurve**
     - Multi-temperature frequency sweeps
     - Shifted master curve
     - Temperature dependence, WLF parameters
   * - **MutationNumber**
     - Time-series G', G''
     - Mutation number (scalar)
     - Quantify gel/cure/crystallization
   * - **OWChirp**
     - Chirp stress/strain
     - Frequency-dependent moduli
     - LAOS analysis, nonlinear response
   * - **SmoothDerivative**
     - Noisy time/frequency data
     - Smoothed derivative
     - Calculate rates (shear rate, etc.)


FFTAnalysis: Time → Frequency Conversion
-----------------------------------------

Purpose
~~~~~~~

The FFT Analysis transform converts arbitrary time-domain stress and strain waveforms into frequency-domain storage (G') and loss (G'') moduli. This is essential for:

- Converting time-sweep data to frequency-domain
- Analyzing arbitrary waveform rheology (AWR)
- Processing chirp experiments
- Extracting moduli from custom input signals

When to Use
~~~~~~~~~~~

Use FFTAnalysis when you have:

- Time-series stress and strain measurements
- Arbitrary input waveforms (not just sinusoidal)
- Chirp experiments (frequency-varying signals)
- Need to extract G' and G'' from raw oscillatory data

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import FFTAnalysis
   from rheojax.io import auto_load

   # Load time-series data
   data = auto_load('oscillation_time_series.txt')
   # data.x = time, data.y = stress or strain

   # Create and apply FFT transform
   fft = FFTAnalysis()
   freq_data = fft.transform(data)

   # Output: freq_data.x = frequency, freq_data.y = complex modulus G*
   # Access G' and G'' from metadata
   G_prime = freq_data.metadata['G_prime']
   G_double_prime = freq_data.metadata['G_double_prime']

Advanced Options
~~~~~~~~~~~~~~~~

Control windowing, detrending, and endpoint handling:

.. code-block:: python

   # Advanced FFT options
   fft = FFTAnalysis(
       window='hann',           # Window function: 'hann', 'hamming', 'blackman', None
       detrend=True,            # Remove linear trend
       remove_endpoints=True,   # Exclude first/last points (transients)
       n_fft=None,              # FFT size (None = auto)
       zero_padding=1.0         # Zero-padding factor (1.0 = no padding)
   )

   freq_data = fft.transform(data)

   # Extract characteristic relaxation time
   tau_char = fft.get_characteristic_time(freq_data)
   print(f"Characteristic time: {tau_char:.3f} s")

Handling Chirp Data
~~~~~~~~~~~~~~~~~~~~

For chirp experiments with frequency-varying input:

.. code-block:: python

   # Chirp data with specified frequency range
   fft = FFTAnalysis(window='hann', detrend=True)

   # Specify chirp frequency range in metadata
   data.metadata['freq_min'] = 0.1  # Hz
   data.metadata['freq_max'] = 10.0  # Hz

   freq_data = fft.transform(data)

   # FFT automatically handles chirp analysis

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Plot G' and G'' vs frequency
   G_prime = freq_data.metadata['G_prime']
   G_double_prime = freq_data.metadata['G_double_prime']
   freq = freq_data.x

   fig, ax = plt.subplots(figsize=(8, 6))
   ax.loglog(freq, G_prime, 'o-', label="G' (storage modulus)")
   ax.loglog(freq, G_double_prime, 's-', label='G" (loss modulus)')
   ax.set_xlabel('Frequency (Hz)')
   ax.set_ylabel('Modulus (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.title('FFT Analysis Result')
   plt.show()

Mastercurve: Time-Temperature Superposition
--------------------------------------------

Purpose
~~~~~~~

The Mastercurve transform implements time-temperature superposition (TTS) to create master curves from multi-temperature frequency sweep data. This reveals material behavior over extended time/frequency ranges and determines temperature-dependent shift factors.

When to Use
~~~~~~~~~~~

Use Mastercurve when you:

- Have frequency sweeps at multiple temperatures
- Want to extend effective frequency range beyond instrument limits
- Need to characterize temperature dependence
- Want to extract WLF or Arrhenius parameters

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import Mastercurve
   from rheojax.io import auto_load

   # Load data at multiple temperatures
   data_25C = auto_load('freq_sweep_25C.txt')
   data_50C = auto_load('freq_sweep_50C.txt')
   data_75C = auto_load('freq_sweep_75C.txt')

   datasets = [data_25C, data_50C, data_75C]
   temperatures = [25, 50, 75]  # Celsius

   # Create mastercurve with reference temperature
   mc = Mastercurve(reference_temp=50, method='wlf')
   mastercurve = mc.create_mastercurve(datasets, temperatures)

   # Access results
   shift_factors = mc.get_shift_factors()
   print(f"Shift factors: {shift_factors}")

   # Shifted data spans wider frequency range
   print(f"Original range: {data_50C.x[0]:.2e} - {data_50C.x[-1]:.2e} Hz")
   print(f"Master curve range: {mastercurve.x[0]:.2e} - {mastercurve.x[-1]:.2e} Hz")

WLF Equation
~~~~~~~~~~~~

For polymers above glass transition temperature:

.. code-block:: python

   # WLF equation: log(a_T) = -C1*(T-T_ref)/(C2+T-T_ref)
   mc = Mastercurve(
       reference_temp=298.15,  # K (25°C)
       method='wlf',
       C1=17.44,               # Universal C1 (can be optimized)
       C2=51.6                 # Universal C2 (can be optimized)
   )

   mastercurve = mc.create_mastercurve(datasets, temperatures)

   # Extract fitted WLF parameters
   C1_fit, C2_fit = mc.get_wlf_parameters()
   print(f"Fitted WLF: C1={C1_fit:.2f}, C2={C2_fit:.2f} K")

   # Evaluate shift factor at any temperature
   a_T_60C = mc.evaluate_shift_factor(60)  # Celsius
   print(f"Shift factor at 60°C: {a_T_60C:.3e}")

Arrhenius Equation
~~~~~~~~~~~~~~~~~~

For systems with thermally-activated processes:

.. code-block:: python

   # Arrhenius: log(a_T) = E_a/R * (1/T - 1/T_ref)
   mc = Mastercurve(
       reference_temp=298.15,  # K
       method='arrhenius',
       E_a=50000               # Activation energy (J/mol), can be optimized
   )

   mastercurve = mc.create_mastercurve(datasets, temperatures)

   # Extract activation energy
   E_a_fit = mc.get_activation_energy()
   print(f"Activation energy: {E_a_fit/1000:.1f} kJ/mol")

Manual Shift Factors
~~~~~~~~~~~~~~~~~~~~

For custom shifting:

.. code-block:: python

   # Provide manual shift factors (useful for validation)
   mc = Mastercurve(reference_temp=50, method='manual')

   # Specify shift factors for each temperature
   manual_shifts = {
       25: 2.5,    # log10(a_T) at 25°C
       50: 0.0,    # Reference temperature
       75: -1.8    # log10(a_T) at 75°C
   }

   mastercurve = mc.create_mastercurve(
       datasets, temperatures,
       shift_factors=manual_shifts
   )

Optimization Options
~~~~~~~~~~~~~~~~~~~~

Control the shifting optimization:

.. code-block:: python

   mc = Mastercurve(
       reference_temp=298.15,
       method='wlf',
       optimize=True,          # Optimize C1, C2 (default: True)
       vertical_shift=False,   # Also shift vertically (default: False)
       smooth_overlap=True,    # Smooth overlapping regions (default: True)
       max_iterations=1000,    # Optimization iterations
       tolerance=1e-6          # Convergence tolerance
   )

   mastercurve = mc.create_mastercurve(datasets, temperatures)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot original data and master curve
   fig, ax = plt.subplots(figsize=(10, 6))

   # Original data (each temperature different color)
   colors = ['blue', 'green', 'red', 'purple', 'orange']
   for i, (data, temp) in enumerate(zip(datasets, temperatures)):
       ax.loglog(data.x, data.y, 'o', color=colors[i],
                 alpha=0.5, label=f'{temp}°C')

   # Master curve (bold line)
   ax.loglog(mastercurve.x, mastercurve.y, 'k-',
             linewidth=3, label='Master Curve')

   ax.set_xlabel('Frequency (Hz) or Reduced Frequency (Hz)')
   ax.set_ylabel('|G*| (Pa)')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.title(f'Time-Temperature Superposition (Ref: {mc.reference_temp}°C)')
   plt.show()


MutationNumber: Quantifying Viscoelastic Evolution
---------------------------------------------------

Purpose
~~~~~~~

The Mutation Number transform quantifies the cumulative change in viscoelastic character during time-resolved experiments. It's particularly useful for:

- Gelation and curing processes
- Crystallization kinetics
- Structural evolution during chemical reactions
- Comparing different formulations

When to Use
~~~~~~~~~~~

Use MutationNumber when you:

- Have time-resolved oscillatory data (G' and G'' vs time)
- Want to quantify gel point or cure completion
- Need a single metric to compare different samples
- Study structural transitions (sol-gel, liquid-solid)

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import MutationNumber
   from rheojax.io import auto_load

   # Load time-resolved oscillatory data (time sweep)
   data = auto_load('curing_time_sweep.txt')
   # Requires G'(t) and G''(t) in metadata

   # Calculate mutation number
   mn = MutationNumber()
   mutation_number = mn.calculate(data)

   print(f"Mutation number: {mutation_number:.3f}")
   # δ ≈ 0: elastic (solid-like)
   # δ ≈ 1: viscous (liquid-like)

Interpretation
~~~~~~~~~~~~~~

The mutation number δ quantifies viscoelastic character:

.. math::

   \\delta = \\frac{1}{\\pi} \\int_0^t \\left| \\frac{d(\\ln G')}{dt} \\right| dt

Physical meaning:

- **δ < 0.2**: Predominantly elastic (solid-like)

  - Cross-linked networks
  - Cured polymers
  - Strong gels

- **0.2 < δ < 0.8**: Viscoelastic transition region

  - Gelation process
  - Partial curing
  - Weak gels

- **δ > 0.8**: Predominantly viscous (liquid-like)

  - Polymer solutions
  - Uncured resins
  - Viscous liquids

Example: Gel Point Determination
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Track mutation number evolution during curing
   import numpy as np
   import matplotlib.pyplot as plt

   # Load time-series data
   data = auto_load('epoxy_curing.txt')
   time = data.x
   G_prime = data.metadata['G_prime']
   G_double_prime = data.metadata['G_double_prime']

   # Calculate mutation number at different time points
   mn = MutationNumber()

   mutation_evolution = []
   for i in range(len(time)):
       # Create partial dataset up to time[i]
       partial_data = data.slice(0, i+1)
       delta = mn.calculate(partial_data)
       mutation_evolution.append(delta)

   # Plot evolution
   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

   # G' and G'' vs time
   ax1.loglog(time, G_prime, 'o-', label="G'")
   ax1.loglog(time, G_double_prime, 's-', label='G"')
   ax1.set_xlabel('Time (s)')
   ax1.set_ylabel('Modulus (Pa)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Mutation number vs time
   ax2.plot(time, mutation_evolution, 'ro-')
   ax2.axhline(y=0.5, color='k', linestyle='--', label='δ = 0.5 (gel point)')
   ax2.set_xlabel('Time (s)')
   ax2.set_ylabel('Mutation Number δ')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Advanced Options
~~~~~~~~~~~~~~~~

Control smoothing and calculation parameters:

.. code-block:: python

   mn = MutationNumber(
       smooth=True,          # Apply smoothing to G'(t) before differentiation
       window_size=11,       # Smoothing window size
       method='trapezoid',   # Integration method: 'trapezoid', 'simpson'
       normalize=True        # Normalize by π
   )

   delta = mn.calculate(data)

Comparing Multiple Samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare mutation numbers of different formulations
   samples = ['formulation_A.txt', 'formulation_B.txt', 'formulation_C.txt']
   labels = ['Formulation A', 'Formulation B', 'Formulation C']

   mn = MutationNumber()
   results = {}

   for sample, label in zip(samples, labels):
       data = auto_load(sample)
       delta = mn.calculate(data)
       results[label] = delta
       print(f"{label}: δ = {delta:.3f}")

   # Bar chart comparison
   plt.figure(figsize=(8, 6))
   plt.bar(results.keys(), results.values())
   plt.ylabel('Mutation Number δ')
   plt.title('Viscoelastic Character Comparison')
   plt.axhline(y=0.5, color='r', linestyle='--', label='Gel point')
   plt.legend()
   plt.show()


OWChirp: Optimal Waveform Analysis for LAOS
--------------------------------------------

Purpose
~~~~~~~

The OWChirp transform analyzes optimal waveform (OW) chirp experiments used in Large Amplitude Oscillatory Shear (LAOS) rheology. It extracts frequency-dependent nonlinear moduli from chirp signals.

When to Use
~~~~~~~~~~~

Use OWChirp when you:

- Perform LAOS measurements with chirp input
- Need frequency-dependent nonlinear moduli
- Want to characterize strain-dependent behavior efficiently
- Analyze arbitrary waveform rheology with varying frequency

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import OWChirp
   from rheojax.io import auto_load

   # Load chirp experiment data
   data = auto_load('owchirp_experiment.txt')
   # Contains time-series stress and strain

   # Analyze with OWChirp
   owchirp = OWChirp(
       freq_min=0.1,      # Minimum frequency (Hz)
       freq_max=10.0,     # Maximum frequency (Hz)
       strain_amplitude=0.1  # Strain amplitude (-)
   )

   result = owchirp.transform(data)

   # Extract frequency-dependent moduli
   frequencies = result.x
   G_prime_LAOS = result.metadata['G1']      # First harmonic G'
   G_double_prime_LAOS = result.metadata['G1_prime']  # First harmonic G''

Chirp Signal Generation
~~~~~~~~~~~~~~~~~~~~~~~~

Generate optimal chirp waveforms for experiments:

.. code-block:: python

   import numpy as np

   # Generate chirp strain signal
   owchirp = OWChirp(freq_min=0.1, freq_max=10.0, strain_amplitude=0.1)

   time = np.linspace(0, 100, 10000)  # 100s experiment
   strain_chirp = owchirp.generate_chirp(time)

   # Export for instrument
   np.savetxt('chirp_waveform.txt', np.column_stack([time, strain_chirp]),
              header='Time(s) Strain(-)')

Higher Harmonics
~~~~~~~~~~~~~~~~

Extract nonlinear higher harmonics:

.. code-block:: python

   # Analyze with higher harmonic extraction
   owchirp = OWChirp(
       freq_min=0.1,
       freq_max=10.0,
       strain_amplitude=0.1,
       n_harmonics=5  # Extract up to 5th harmonic
   )

   result = owchirp.transform(data)

   # Access harmonics
   G1 = result.metadata['G1']        # 1st harmonic (linear response)
   G3 = result.metadata['G3']        # 3rd harmonic (nonlinear)
   G5 = result.metadata['G5']        # 5th harmonic (nonlinear)

   # Nonlinearity indicator
   I3_1 = G3 / G1  # Intensity ratio (nonlinearity measure)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Plot linear and nonlinear moduli
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

   # Linear moduli (1st harmonic)
   ax1.loglog(frequencies, G1, 'o-', label="G'₁ (storage)")
   ax1.loglog(frequencies, G1_prime, 's-', label='G"₁ (loss)')
   ax1.set_xlabel('Frequency (Hz)')
   ax1.set_ylabel('Modulus (Pa)')
   ax1.set_title('Linear Response (1st Harmonic)')
   ax1.legend()
   ax1.grid(True, alpha=0.3)

   # Nonlinear indicator
   ax2.semilogx(frequencies, I3_1, 'ro-', label='I₃/₁')
   ax2.set_xlabel('Frequency (Hz)')
   ax2.set_ylabel('Nonlinearity I₃/₁')
   ax2.set_title('Nonlinear Response Indicator')
   ax2.legend()
   ax2.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Advanced Options
~~~~~~~~~~~~~~~~

.. code-block:: python

   owchirp = OWChirp(
       freq_min=0.1,
       freq_max=10.0,
       strain_amplitude=0.1,
       n_harmonics=5,
       window='hann',         # Window for FFT
       detrend=True,          # Remove trend
       n_periods=5,           # Number of periods per frequency
       frequency_spacing='log'  # 'log' or 'linear'
   )

   result = owchirp.transform(data)


SmoothDerivative: Noise-Robust Differentiation
-----------------------------------------------

Purpose
~~~~~~~

The SmoothDerivative transform calculates derivatives of noisy data using robust methods that suppress noise while preserving features. Essential for:

- Calculating shear rates from displacement/strain data
- Finding peaks and inflection points
- Numerical differentiation of experimental data
- Pre-processing before model fitting

When to Use
~~~~~~~~~~~

Use SmoothDerivative when you:

- Need derivatives of noisy experimental data
- Want to calculate shear rates: γ̇ = dγ/dt
- Need to find extrema or inflection points
- Pre-process data before fitting models

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SmoothDerivative
   from rheojax.io import auto_load

   # Load noisy data
   data = auto_load('strain_vs_time.txt')

   # Calculate smooth derivative (shear rate)
   smooth_deriv = SmoothDerivative(
       method='savgol',    # Savitzky-Golay filter
       window=11,          # Window size (odd integer)
       order=2             # Polynomial order
   )

   derivative_data = smooth_deriv.transform(data)

   # Result: derivative_data.y = dγ/dt (shear rate)

Methods Available
~~~~~~~~~~~~~~~~~

**Savitzky-Golay (savgol)**

Best for: Smooth data with moderate noise

.. code-block:: python

   smooth_deriv = SmoothDerivative(
       method='savgol',
       window=11,      # Larger window = more smoothing
       order=2         # 2 or 3 typical
   )

**Finite Differences (finite_diff)**

Best for: Clean data, high accuracy needed

.. code-block:: python

   smooth_deriv = SmoothDerivative(
       method='finite_diff',
       accuracy=2      # 2, 4, or 6 (higher = more accurate)
   )

**Spline (spline)**

Best for: Very noisy data, need maximum smoothness

.. code-block:: python

   smooth_deriv = SmoothDerivative(
       method='spline',
       smoothing=0.01   # Smoothing parameter (0.001-0.1 typical)
   )

**Gaussian Filter (gaussian)**

Best for: Heavy noise, isotropic smoothing

.. code-block:: python

   smooth_deriv = SmoothDerivative(
       method='gaussian',
       sigma=2.0        # Standard deviation of Gaussian kernel
   )

Calculating Shear Rates
~~~~~~~~~~~~~~~~~~~~~~~~

Common use case: convert strain vs time to shear rate:

.. code-block:: python

   # Strain vs time data
   data = auto_load('strain_vs_time.txt')
   # data.x = time (s), data.y = strain (-)

   # Calculate shear rate
   smooth_deriv = SmoothDerivative(method='savgol', window=11, order=2)
   shear_rate_data = smooth_deriv.transform(data)

   # shear_rate_data.y = dγ/dt (s⁻¹)
   # Update units
   shear_rate_data.y_units = '1/s'

   # Now use for fitting flow models
   from rheojax.models import PowerLaw

   model = PowerLaw()
   model.fit(shear_rate_data.y, stress_data.y)  # τ vs γ̇

Comparison of Methods
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Generate noisy data
   t = np.linspace(0, 10, 200)
   y_true = np.sin(2*np.pi*0.5*t)
   dy_true = 2*np.pi*0.5 * np.cos(2*np.pi*0.5*t)
   y_noisy = y_true + 0.1 * np.random.randn(len(t))

   from rheojax.core import RheoData
   data = RheoData(x=t, y=y_noisy, x_units='s', y_units='-', domain='time')

   # Try different methods
   methods = [
       ('savgol', {'window': 11, 'order': 2}),
       ('spline', {'smoothing': 0.05}),
       ('gaussian', {'sigma': 2.0}),
       ('finite_diff', {'accuracy': 2})
   ]

   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(t, dy_true, 'k-', linewidth=3, label='True derivative', alpha=0.5)

   for method_name, kwargs in methods:
       smooth_deriv = SmoothDerivative(method=method_name, **kwargs)
       deriv_data = smooth_deriv.transform(data)
       ax.plot(deriv_data.x, deriv_data.y, label=f'{method_name}')

   ax.set_xlabel('Time (s)')
   ax.set_ylabel('Derivative')
   ax.legend()
   ax.grid(True, alpha=0.3)
   plt.title('Comparison of Derivative Methods')
   plt.show()

Advanced: Higher-Order Derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate second derivative (acceleration, curvature)
   smooth_deriv = SmoothDerivative(
       method='savgol',
       window=11,
       order=3,         # Order must be >= derivative order
       derivative=2     # 2nd derivative
   )

   second_deriv = smooth_deriv.transform(data)

   # Find inflection points (where 2nd derivative = 0)
   inflection_indices = np.where(np.diff(np.sign(second_deriv.y)))[0]


Transform Composition and Pipelines
------------------------------------

Chaining Transforms
~~~~~~~~~~~~~~~~~~~

Combine multiple transforms in sequence:

.. code-block:: python

   from rheojax.transforms import FFTAnalysis, SmoothDerivative, MutationNumber

   # Load raw time-series data
   data = auto_load('noisy_oscillation.txt')

   # 1. Smooth data
   smoother = SmoothDerivative(method='savgol', window=11, order=2)
   data_smooth = smoother.transform(data)

   # 2. FFT analysis
   fft = FFTAnalysis(window='hann')
   freq_data = fft.transform(data_smooth)

   # 3. Calculate mutation number
   mn = MutationNumber()
   delta = mn.calculate(freq_data)

Using TransformPipeline
~~~~~~~~~~~~~~~~~~~~~~~

For cleaner code, use the pipeline pattern:

.. code-block:: python

   from rheojax.core.base import TransformPipeline
   from rheojax.transforms import SmoothDerivative, FFTAnalysis

   # Create pipeline
   pipeline = TransformPipeline([
       SmoothDerivative(method='savgol', window=11, order=2),
       FFTAnalysis(window='hann', detrend=True)
   ])

   # Apply entire pipeline
   result = pipeline.transform(data)

   # Or use operator overloading
   pipeline = SmoothDerivative(method='savgol', window=11, order=2) + \
              FFTAnalysis(window='hann', detrend=True)

   result = pipeline.transform(data)

Best Practices
--------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Always visualize before and after**: Check that transforms preserve important features
2. **Start with defaults**: Default parameters are usually reasonable
3. **Validate on synthetic data**: Test with known ground truth before real data
4. **Check units**: Ensure output units are correct after transforms
5. **Document parameters**: Record transform parameters for reproducibility

Transform-Specific Tips
~~~~~~~~~~~~~~~~~~~~~~~~

**FFTAnalysis**:

- Use windowing for non-periodic data to reduce spectral leakage
- Apply detrending if data has baseline drift
- Check characteristic time makes physical sense
- Be aware of frequency resolution (limited by signal duration)

**Mastercurve**:

- Need at least 3-4 temperatures for reliable WLF/Arrhenius fitting
- Check that shift factors follow smooth trend
- Reference temperature should be in middle of range
- Verify overlap quality between adjacent temperatures

**MutationNumber**:

- Smooth G'(t) before calculation if noisy
- Check that δ evolution is monotonic (should always increase)
- Gel point typically at δ ≈ 0.5 but material-dependent
- Requires high-quality time-resolved data

**OWChirp**:

- Ensure chirp covers desired frequency range
- Use enough periods per frequency (5-10 typical)
- Check for instrument limitations at extremes
- Validate linear regime before interpreting nonlinear harmonics

**SmoothDerivative**:

- Choose window size ~10% of data length for Savgol
- Avoid over-smoothing (loses features)
- Compare multiple methods to check consistency
- Validate with synthetic data of similar noise level

Common Pitfalls
~~~~~~~~~~~~~~~

**Pitfall 1**: Applying FFT to non-oscillatory data

Solution: Check test mode, FFT only works for oscillatory signals

**Pitfall 2**: Over-smoothing with SmoothDerivative

Solution: Start with small window, increase gradually, compare to finite differences

**Pitfall 3**: Insufficient temperature range for mastercurves

Solution: Need T_min to T_max span at least 30-50°C for polymers

**Pitfall 4**: Ignoring edge effects in transforms

Solution: Use remove_endpoints option or crop data before processing

**Pitfall 5**: Not validating transform output

Solution: Always plot before and after, check units and magnitudes

Example Workflow
----------------

Complete analysis workflow with multiple transforms:

.. code-block:: python

   from rheojax.io import auto_load
   from rheojax.transforms import (SmoothDerivative, FFTAnalysis,
                                  Mastercurve, MutationNumber)
   from rheojax.models import FractionalMaxwellGel
   import matplotlib.pyplot as plt

   # 1. Load multi-temperature time-series data
   temps = [25, 40, 55, 70]
   datasets_raw = [auto_load(f'time_series_{T}C.txt') for T in temps]

   # 2. Smooth each dataset
   smoother = SmoothDerivative(method='savgol', window=11, order=2)
   datasets_smooth = [smoother.transform(d) for d in datasets_raw]

   # 3. FFT to frequency domain
   fft = FFTAnalysis(window='hann', detrend=True)
   datasets_freq = [fft.transform(d) for d in datasets_smooth]

   # 4. Create mastercurve
   mc = Mastercurve(reference_temp=40, method='wlf')
   mastercurve = mc.create_mastercurve(datasets_freq, temps)

   # 5. Fit fractional model to mastercurve
   model = FractionalMaxwellGel()
   model.fit(mastercurve.x, mastercurve.y)

   # 6. Calculate mutation number for first dataset (monitoring cure)
   mn = MutationNumber()
   delta = mn.calculate(datasets_freq[0])

   # 7. Visualize everything
   fig, axes = plt.subplots(2, 2, figsize=(14, 10))

   # Original data
   for d, T in zip(datasets_raw, temps):
       axes[0,0].plot(d.x, d.y, alpha=0.5, label=f'{T}°C')
   axes[0,0].set_xlabel('Time (s)')
   axes[0,0].set_ylabel('Stress (Pa)')
   axes[0,0].set_title('Raw Time-Series Data')
   axes[0,0].legend()

   # Frequency domain (all temps)
   for d, T in zip(datasets_freq, temps):
       axes[0,1].loglog(d.x, np.abs(d.y), 'o', alpha=0.5, label=f'{T}°C')
   axes[0,1].set_xlabel('Frequency (Hz)')
   axes[0,1].set_ylabel('|G*| (Pa)')
   axes[0,1].set_title('FFT Analysis Results')
   axes[0,1].legend()

   # Mastercurve with model fit
   axes[1,0].loglog(mastercurve.x, mastercurve.y, 'ko', label='Master Curve')
   axes[1,0].loglog(mastercurve.x, model.predict(mastercurve.x),
                    'r-', linewidth=2, label='FractionalMaxwellGel')
   axes[1,0].set_xlabel('Reduced Frequency (Hz)')
   axes[1,0].set_ylabel('|G*| (Pa)')
   axes[1,0].set_title('Master Curve + Model Fit')
   axes[1,0].legend()

   # Mutation number evolution
   times = datasets_raw[0].x
   G_prime = datasets_freq[0].metadata['G_prime']
   axes[1,1].plot(times, G_prime, 'b-')
   axes[1,1].axhline(y=np.mean(G_prime), color='r', linestyle='--')
   axes[1,1].text(times[-1]*0.7, np.mean(G_prime)*1.1,
                  f'δ = {delta:.3f}', fontsize=12)
   axes[1,1].set_xlabel('Time (s)')
   axes[1,1].set_ylabel("G' (Pa)")
   axes[1,1].set_title("Storage Modulus Evolution")

   plt.tight_layout()
   plt.savefig('complete_analysis.png', dpi=300)
   plt.show()

Summary
-------

Transform selection checklist:

- **Time → frequency conversion**: Use :class:`FFTAnalysis`
- **Temperature dependence**: Use :class:`Mastercurve` with WLF/Arrhenius
- **Gelation/curing**: Use :class:`MutationNumber`
- **LAOS analysis**: Use :class:`OWChirp`
- **Noisy data**: Use :class:`SmoothDerivative` before other transforms

For more information:

- :doc:`/user_guide/pipeline_api` - High-level transform workflows
- :doc:`/api/transforms` - Complete API reference
- ``examples/transforms/mastercurve_example.ipynb`` - Mastercurve example notebook
- ``examples/transforms/`` - Transform composition examples
