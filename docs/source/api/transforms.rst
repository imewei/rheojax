Transforms API
==============

This page documents all 5 data transforms implemented in rheo for rheological data preprocessing and analysis.

Overview
--------

rheo transforms process :class:`rheo.core.data.RheoData` objects to extract features, convert between domains, or preprocess data. All transforms inherit from :class:`rheo.core.base.BaseTransform` and provide ``transform()`` and ``fit_transform()`` methods.

**Available Transforms**:

1. **FFTAnalysis**: Time → frequency domain conversion
2. **Mastercurve**: Time-temperature superposition
3. **MutationNumber**: Quantify viscoelastic character
4. **OWChirp**: Optimal waveform analysis for LAOS
5. **SmoothDerivative**: Noise-robust differentiation

Transform Registry
------------------

Access transforms through the registry:

.. code-block:: python

   from rheo.core.registry import TransformRegistry

   # List all transforms
   transforms = TransformRegistry.list_transforms()

   # Create transform by name
   fft = TransformRegistry.create('fft_analysis')

   # Get transform information
   info = TransformRegistry.get_info('mastercurve')

FFTAnalysis
-----------

.. autoclass:: rheo.transforms.FFTAnalysis
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Convert time-domain oscillatory data to frequency-domain storage (G') and loss (G'') moduli using Fast Fourier Transform.

**Input**: Time-series stress and strain data

**Output**: Frequency-domain complex modulus G*(ω)

**Example**:

.. code-block:: python

   from rheo.transforms import FFTAnalysis
   from rheo.io import auto_load

   # Load time-series data
   data = auto_load('oscillation_time_series.txt')

   # Apply FFT analysis
   fft = FFTAnalysis(
       window='hann',      # Window function for spectral leakage reduction
       detrend=True,       # Remove linear trend
       remove_endpoints=True  # Exclude transient endpoints
   )

   freq_data = fft.transform(data)

   # Access moduli
   G_prime = freq_data.metadata['G_prime']
   G_double_prime = freq_data.metadata['G_double_prime']
   frequencies = freq_data.x

   # Characteristic relaxation time
   tau_char = fft.get_characteristic_time(freq_data)

**Parameters**:

- ``window`` (str): Window function - 'hann', 'hamming', 'blackman', or None
- ``detrend`` (bool): Remove linear trend before FFT
- ``remove_endpoints`` (bool): Exclude first/last points (transients)
- ``n_fft`` (int, optional): FFT size (None = auto)
- ``zero_padding`` (float): Zero-padding factor for interpolation

**Methods**:

- ``transform(data)``: Apply FFT analysis
- ``get_characteristic_time(freq_data)``: Extract characteristic time
- ``inverse_transform(freq_data)``: Frequency → time (if implemented)

Mastercurve
-----------

.. autoclass:: rheo.transforms.Mastercurve
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Perform time-temperature superposition (TTS) to create master curves from multi-temperature frequency sweep data. Extracts WLF or Arrhenius parameters.

**Input**: Multiple RheoData objects at different temperatures

**Output**: Single master curve with extended frequency range

**Example**:

.. code-block:: python

   from rheo.transforms import Mastercurve
   from rheo.io import auto_load

   # Load multi-temperature data
   data_25C = auto_load('freq_sweep_25C.txt')
   data_50C = auto_load('freq_sweep_50C.txt')
   data_75C = auto_load('freq_sweep_75C.txt')

   datasets = [data_25C, data_50C, data_75C]
   temperatures = [25, 50, 75]  # Celsius

   # Create mastercurve with WLF equation
   mc = Mastercurve(
       reference_temp=50,      # Reference temperature (°C)
       method='wlf',           # 'wlf' or 'arrhenius'
       optimize=True,          # Optimize C1, C2 parameters
       C1=17.44,              # Universal WLF C1 (initial guess)
       C2=51.6                # Universal WLF C2 (initial guess)
   )

   mastercurve = mc.create_mastercurve(datasets, temperatures)

   # Extract results
   shift_factors = mc.get_shift_factors()
   C1, C2 = mc.get_wlf_parameters()

   print(f"WLF C1 = {C1:.2f}, C2 = {C2:.2f} K")

   # Evaluate shift factor at any temperature
   a_T_60C = mc.evaluate_shift_factor(60)

**Parameters**:

- ``reference_temp`` (float): Reference temperature for master curve
- ``method`` (str): 'wlf', 'arrhenius', or 'manual'
- ``optimize`` (bool): Optimize WLF/Arrhenius parameters
- ``C1``, ``C2`` (float): WLF equation parameters
- ``E_a`` (float): Activation energy for Arrhenius (J/mol)
- ``vertical_shift`` (bool): Also shift vertically (default: False)
- ``smooth_overlap`` (bool): Smooth overlapping regions

**Methods**:

- ``create_mastercurve(datasets, temperatures)``: Generate master curve
- ``get_shift_factors()``: Retrieve shift factors for each temperature
- ``get_wlf_parameters()``: Get fitted C1, C2 values
- ``get_activation_energy()``: Get fitted E_a (Arrhenius)
- ``evaluate_shift_factor(temperature)``: Calculate shift factor at T

**WLF Equation**:

.. math::

   \\log(a_T) = \\frac{-C_1(T - T_{ref})}{C_2 + T - T_{ref}}

**Arrhenius Equation**:

.. math::

   \\log(a_T) = \\frac{E_a}{R} \\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)

Mutation Number
---------------

.. autoclass:: rheo.transforms.MutationNumber
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Calculate the mutation number (δ) to quantify cumulative change in viscoelastic character during time-resolved experiments (gelation, curing, crystallization).

**Input**: Time-series G'(t) and G''(t) from oscillatory time sweep

**Output**: Mutation number δ (scalar, 0 = elastic, 1 = viscous)

**Example**:

.. code-block:: python

   from rheo.transforms import MutationNumber
   from rheo.io import auto_load

   # Load time-resolved oscillatory data (curing, gelation)
   data = auto_load('curing_time_sweep.txt')

   # Calculate mutation number
   mn = MutationNumber(
       smooth=True,           # Smooth G'(t) before differentiation
       window_size=11,        # Smoothing window
       method='trapezoid',    # Integration method
       normalize=True         # Normalize by π
   )

   delta = mn.calculate(data)

   print(f"Mutation number δ = {delta:.3f}")

   # Interpretation:
   # δ < 0.2: predominantly elastic (solid-like)
   # 0.2 < δ < 0.8: viscoelastic transition
   # δ > 0.8: predominantly viscous (liquid-like)

**Parameters**:

- ``smooth`` (bool): Apply smoothing before differentiation
- ``window_size`` (int): Smoothing window size
- ``method`` (str): Integration method - 'trapezoid' or 'simpson'
- ``normalize`` (bool): Normalize by π (default: True)

**Methods**:

- ``calculate(data)``: Compute mutation number from time-series data

**Equation**:

.. math::

   \\delta = \\frac{1}{\\pi} \\int_0^t \\left| \\frac{d(\\ln G')}{dt} \\right| dt

**Physical Meaning**:

The mutation number quantifies the integrated rate of change in elastic character. It's particularly useful for:

- Determining gel points (typically δ ≈ 0.5)
- Comparing cure kinetics across formulations
- Monitoring structural evolution in real-time

OWChirp
-------

.. autoclass:: rheo.transforms.OWChirp
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Analyze optimal waveform (OW) chirp experiments for Large Amplitude Oscillatory Shear (LAOS) rheology. Extracts frequency-dependent linear and nonlinear moduli.

**Input**: Chirp experiment time-series (stress and strain)

**Output**: Frequency-dependent harmonics (G₁', G₁'', G₃', etc.)

**Example**:

.. code-block:: python

   from rheo.transforms import OWChirp
   from rheo.io import auto_load

   # Load chirp experiment
   data = auto_load('owchirp_experiment.txt')

   # Analyze chirp
   owchirp = OWChirp(
       freq_min=0.1,          # Minimum frequency (Hz)
       freq_max=10.0,         # Maximum frequency (Hz)
       strain_amplitude=0.1,  # Strain amplitude
       n_harmonics=5          # Extract up to 5th harmonic
   )

   result = owchirp.transform(data)

   # Access harmonics
   frequencies = result.x
   G1 = result.metadata['G1']        # 1st harmonic (linear)
   G3 = result.metadata['G3']        # 3rd harmonic (nonlinear)
   G5 = result.metadata['G5']        # 5th harmonic

   # Nonlinearity indicator
   I3_1 = G3 / G1
   print(f"Nonlinearity I₃/₁ = {I3_1}")

**Parameters**:

- ``freq_min`` (float): Minimum frequency (Hz)
- ``freq_max`` (float): Maximum frequency (Hz)
- ``strain_amplitude`` (float): Strain amplitude
- ``n_harmonics`` (int): Number of harmonics to extract (default: 5)
- ``window`` (str): Window function for FFT
- ``detrend`` (bool): Remove trend before analysis
- ``n_periods`` (int): Periods per frequency
- ``frequency_spacing`` (str): 'log' or 'linear'

**Methods**:

- ``transform(data)``: Analyze chirp data
- ``generate_chirp(time)``: Generate optimal chirp waveform
- ``extract_harmonics(data)``: Extract individual harmonics

SmoothDerivative
----------------

.. autoclass:: rheo.transforms.SmoothDerivative
   :members:
   :undoc-members:
   :show-inheritance:

**Description**: Calculate derivatives of noisy data using robust methods that suppress noise while preserving features. Essential for computing shear rates and finding extrema.

**Input**: Noisy RheoData (any domain)

**Output**: Smoothed derivative

**Example**:

.. code-block:: python

   from rheo.transforms import SmoothDerivative
   from rheo.io import auto_load

   # Load noisy strain vs time data
   data = auto_load('strain_vs_time.txt')

   # Calculate shear rate (dγ/dt)
   smooth_deriv = SmoothDerivative(
       method='savgol',    # Savitzky-Golay filter
       window=11,          # Window size (must be odd)
       order=2,            # Polynomial order
       derivative=1        # 1st derivative
   )

   shear_rate_data = smooth_deriv.transform(data)
   shear_rate_data.y_units = '1/s'  # Update units

   # Use for flow model fitting
   from rheo.models import PowerLaw
   power_law = PowerLaw()
   power_law.fit(shear_rate_data.y, stress_data.y)

**Parameters**:

- ``method`` (str): Derivative method

  - 'savgol': Savitzky-Golay filter (best for smooth data)
  - 'finite_diff': Finite differences (best for clean data)
  - 'spline': Spline interpolation (best for very noisy data)
  - 'gaussian': Gaussian filter (best for heavy noise)

- ``window`` (int): Window size (odd integer, savgol/gaussian)
- ``order`` (int): Polynomial order (savgol)
- ``accuracy`` (int): Finite difference accuracy (2, 4, or 6)
- ``smoothing`` (float): Spline smoothing parameter (0.001-0.1)
- ``sigma`` (float): Gaussian kernel std dev
- ``derivative`` (int): Derivative order (1 or 2)

**Methods**:

- ``transform(data)``: Calculate smooth derivative

**Method Comparison**:

.. list-table:: Derivative Method Selection
   :header-rows: 1
   :widths: 20 35 45

   * - Method
     - Best For
     - Notes
   * - savgol
     - Moderate noise, smooth data
     - Excellent balance of smoothing and accuracy
   * - finite_diff
     - Clean data, high accuracy needed
     - Minimal smoothing, preserves features
   * - spline
     - Very noisy data, need maximum smoothness
     - Strong smoothing, may over-smooth features
   * - gaussian
     - Heavy noise, isotropic smoothing
     - Good general-purpose choice

Transform Composition
---------------------

Chaining Transforms
~~~~~~~~~~~~~~~~~~~

Transforms can be chained using :class:`rheo.core.base.TransformPipeline`:

.. code-block:: python

   from rheo.core.base import TransformPipeline
   from rheo.transforms import SmoothDerivative, FFTAnalysis

   # Create pipeline
   pipeline = TransformPipeline([
       SmoothDerivative(method='savgol', window=11, order=2),
       FFTAnalysis(window='hann', detrend=True)
   ])

   # Apply pipeline
   result = pipeline.transform(data)

   # Alternative: operator overloading
   pipeline = SmoothDerivative(method='savgol', window=11, order=2) + \\
              FFTAnalysis(window='hann', detrend=True)

   result = pipeline.transform(data)

Transform Validation
~~~~~~~~~~~~~~~~~~~~

Always validate transform output:

.. code-block:: python

   import matplotlib.pyplot as plt

   # Before transform
   plt.figure(figsize=(12, 5))

   plt.subplot(1, 2, 1)
   plt.plot(data.x, data.y, 'o-', label='Original')
   plt.xlabel(f'X ({data.x_units})')
   plt.ylabel(f'Y ({data.y_units})')
   plt.title('Before Transform')
   plt.legend()

   # Apply transform
   result = transform.transform(data)

   # After transform
   plt.subplot(1, 2, 2)
   plt.plot(result.x, result.y, 'o-', label='Transformed')
   plt.xlabel(f'X ({result.x_units})')
   plt.ylabel(f'Y ({result.y_units})')
   plt.title('After Transform')
   plt.legend()

   plt.tight_layout()
   plt.show()

See Also
--------

- :doc:`/user_guide/transforms` - Comprehensive transform usage guide
- :doc:`/user_guide/pipeline_api` - Using transforms in pipelines
- :class:`rheo.core.base.BaseTransform` - Base class documentation
- :class:`rheo.core.data.RheoData` - Data structure transforms operate on
