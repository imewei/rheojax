Transforms Summary & Workflow Guide
====================================

This page provides a comprehensive quick-reference guide for all 5 data transforms in RheoJAX. Use the comparison matrices and workflow examples below to select and chain transforms for your rheological analysis pipelines.


Complete Transforms Comparison Matrix
--------------------------------------

The table below provides a comprehensive overview of all transforms across key characteristics for rapid transform selection.

.. list-table:: Comprehensive Transform Comparison
   :header-rows: 1
   :widths: 16 22 16 16 12 10 8
   :class: longtable

   * - Transform
     - Purpose
     - Input Data Type
     - Output Data Type
     - Computational Cost
     - Real-time Capable
     - Best For
   * - :doc:`FFT Analysis </transforms/fft>`
     - Time → frequency domain conversion, spectral analysis, PSD, peak detection
     - Time-domain (t, signal)
     - Frequency-domain (f, spectrum/PSD)
     - Low
     - Yes
     - LAOS analysis, harmonic detection, periodic signals
   * - :doc:`Mastercurve </transforms/mastercurve>`
     - Time-Temperature Superposition (TTS), WLF/Arrhenius shifting, build master curves
     - Multi-T frequency sweeps with metadata
     - Merged mastercurve + shift factors
     - Medium
     - No
     - Polymer characterization, broaden frequency range, validate WLF
   * - :doc:`Mutation Number </transforms/mutation_number>`
     - Quantify relaxation character (solid vs liquid), gel point detection
     - Relaxation data G(t)
     - Scalar Δ ∈ [0,1] (0=elastic, 1=viscous)
     - Low
     - Yes
     - Material classification, gel point, compare viscoelastic character
   * - :doc:`OWChirp </transforms/owchirp>`
     - LAOS time-frequency analysis, extract harmonics, nonlinear indicators
     - Time-domain LAOS (stress/strain vs t)
     - Frequency spectrum + time-frequency map
     - High
     - No
     - Fast rheometry, curing/gelation monitoring, LAOS harmonics
   * - :doc:`Smooth Derivative </transforms/smooth_derivative>`
     - Noise-robust numerical differentiation (Savitzky-Golay, spline, TV)
     - Any RheoData (x, y)
     - Derivative RheoData (dy/dx)
     - Low-Medium
     - Yes
     - Strain rate from strain, noisy data, multi-order derivatives

**Legend:**

* **Computational Cost:** Low (<10ms), Medium (10-100ms), High (>100ms) for typical datasets
* **Real-time Capable:** Suitable for streaming/online analysis
* **Best For:** Primary use cases and applications


Transform Selection Decision Tree
----------------------------------

Follow this decision tree to identify the appropriate transform for your analysis workflow.

.. code-block:: text

   START: What do you want to achieve?
   │
   ├─ DOMAIN CONVERSION (time ↔ frequency)?
   │  │
   │  ├─ Time → Frequency (general spectral analysis)?
   │  │  └─ FFT Analysis ★★★☆☆
   │  │     • Convert time-domain signals to frequency spectra
   │  │     • Compute Power Spectral Density (PSD)
   │  │     • Detect dominant frequencies and harmonics
   │  │     • Window functions: hann, hamming, blackman, tukey
   │  │     • Use Cases: LAOS harmonic detection, periodic signal analysis
   │  │
   │  └─ LAOS-specific time-frequency analysis?
   │     └─ OWChirp ★★★★☆
   │        • Optimal Windowed Chirp Fourier Transform
   │        • Time-resolved frequency content (2D map)
   │        • Extract higher harmonics (I₃/I₁, etc.)
   │        • Nonlinearity indicators
   │        • Use Cases: Curing, gelation, fast time-resolved rheometry
   │
   ├─ TEMPERATURE EFFECTS (build master curves)?
   │  └─ Mastercurve ★★★★☆
   │     • Time-Temperature Superposition (TTS)
   │     • Shift factors: WLF or Arrhenius
   │     • Merge multi-temperature frequency sweeps
   │     • Optimize WLF parameters (C₁, C₂)
   │     • Validate temperature dependence
   │     • Use Cases: Polymer characterization, broaden ω range by 3-5 decades
   │
   ├─ MATERIAL CLASSIFICATION (solid vs liquid)?
   │  └─ Mutation Number ★★☆☆☆
   │     • Quantifies relaxation character: Δ = ∫[dG/d(ln t)] d(ln t)
   │     • Δ = 0 → Pure elastic solid
   │     • Δ = 1 → Pure viscous liquid
   │     • 0 < Δ < 1 → Viscoelastic (closer to 0.5 = balanced)
   │     • Use Cases: Gel point detection, material screening, QC
   │
   ├─ NUMERICAL DIFFERENTIATION (need derivatives)?
   │  └─ Smooth Derivative ★★★☆☆
   │     • Noise-robust differentiation
   │     • Methods: Savitzky-Golay, spline, Total Variation (TV)
   │     • Multi-order derivatives (1st, 2nd, 3rd)
   │     • Automatic unit conversion
   │     • Use Cases: Strain rate from strain, acceleration, velocity
   │
   └─ MULTI-STEP ANALYSIS (combine transforms)?
      └─ See "Common Workflow Pipelines" section below


Transform Deep-Dive Guides
---------------------------

FFT Analysis
~~~~~~~~~~~~

**Purpose:** Convert time-domain rheological signals to frequency domain for spectral analysis, harmonic detection, and periodic signal characterization.

**Key Capabilities:**

* **FFT Transformation:** Time → Frequency using Fast Fourier Transform
* **Power Spectral Density (PSD):** Energy distribution across frequencies
* **Peak Detection:** Identify dominant frequencies and harmonics
* **Windowing:** Reduce spectral leakage (Hann, Hamming, Blackman, Tukey, Bartlett)
* **Normalization:** Optional amplitude normalization

**Input Requirements:**

* Time-domain RheoData with ``domain='time'``
* Evenly spaced time points (or interpolation applied)
* Sufficient data length for frequency resolution: Δf = 1/T_total

**Output:**

* Frequency-domain RheoData: ``(frequency, magnitude)`` or ``(frequency, PSD)``
* Peak frequencies and amplitudes via ``.detect_peaks()``

**Key Parameters:**

.. list-table:: FFT Analysis Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Default
     - Description
   * - ``window``
     - ``'hann'``
     - Window function: 'hann', 'hamming', 'blackman', 'tukey', 'bartlett', None
   * - ``detrend``
     - ``True``
     - Remove linear trend before FFT
   * - ``return_psd``
     - ``False``
     - Return Power Spectral Density instead of magnitude
   * - ``normalize``
     - ``False``
     - Normalize spectrum by maximum amplitude
   * - ``n_peaks``
     - ``5``
     - Number of peaks to detect (in ``.detect_peaks()``)

**Example Usage:**

.. code-block:: python

   from rheojax.transforms import FFTAnalysis
   import numpy as np

   # Time-domain LAOS signal
   t = np.linspace(0, 10, 1000)
   signal = np.sin(2*np.pi*1.0*t) + 0.3*np.sin(2*np.pi*3.0*t)  # Fundamental + 3rd harmonic

   data = RheoData(x=t, y=signal, domain='time')

   # FFT with Hann window
   fft_transform = FFTAnalysis(window='hann', return_psd=False)
   freq_data = fft_transform.transform(data)

   # Detect peaks
   peaks = fft_transform.detect_peaks(freq_data, n_peaks=3)
   print(f"Detected frequencies: {peaks['frequencies']} Hz")

**Best For:**

* LAOS harmonic detection (I₃/I₁ ratios)
* Identifying periodic components in stress/strain signals
* Quality control of oscillatory tests
* Frequency content analysis


Mastercurve
~~~~~~~~~~~

**Purpose:** Apply Time-Temperature Superposition (TTS) to merge multi-temperature frequency sweep data into a single master curve, extending effective frequency range by 3-5 decades.

**Key Capabilities:**

* **WLF Shifting:** Williams-Landel-Ferry equation for polymer melts/elastomers
* **Arrhenius Shifting:** Activation energy-based for Newtonian fluids/solutions
* **Shift Factor Optimization:** Automatic optimization of C₁, C₂ (WLF) or E_a (Arrhenius)
* **Vertical Shifting:** Optional modulus shifting for incompressibility corrections
* **Overlap Error:** Quantitative assessment of superposition quality

**Input Requirements:**

* List of RheoData objects (frequency sweeps at different temperatures)
* Each RheoData must have ``metadata['temperature']`` in Kelvin
* Complex modulus data: G* = [G', G"] or equivalent

**Output:**

* Merged master curve RheoData at reference temperature
* Shift factors dictionary: ``{T: a_T}`` for horizontal shifts

**Shift Models:**

1. **WLF (Williams-Landel-Ferry):**

   .. math::

      \log(a_T) = -\frac{C_1(T - T_{\text{ref}})}{C_2 + (T - T_{\text{ref}})}

   * **Universal constants:** C₁ ≈ 17.44, C₂ ≈ 51.6 K (relative to T_g)
   * **Best for:** Polymer melts, elastomers, T > T_g
   * **Typical range:** T_g + 50K to T_g + 150K

2. **Arrhenius:**

   .. math::

      a_T = \exp\left[\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right]

   * **Best for:** Newtonian fluids, polymer solutions, simple liquids
   * **E_a:** Activation energy (typical: 40-100 kJ/mol)

**Key Parameters:**

.. list-table:: Mastercurve Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``reference_temp``
     - Required
     - Reference temperature (K) for master curve
   * - ``method``
     - ``'wlf'``
     - Shift method: 'wlf' or 'arrhenius'
   * - ``C1``
     - ``17.44``
     - WLF parameter C₁ (only for WLF method)
   * - ``C2``
     - ``51.6``
     - WLF parameter C₂ in Kelvin (only for WLF)
   * - ``E_a``
     - ``50e3``
     - Activation energy in J/mol (only for Arrhenius)
   * - ``vertical_shift``
     - ``False``
     - Apply vertical (modulus) shifting
   * - ``optimize_shifts``
     - ``False``
     - Automatically optimize C₁, C₂ or E_a

**Example Usage:**

.. code-block:: python

   from rheojax.transforms import Mastercurve

   # Multi-temperature frequency sweep datasets
   datasets = [data_25C, data_50C, data_75C, data_100C]

   # Create mastercurve at 50°C with WLF shifting
   mc = Mastercurve(
       reference_temp=323.15,  # 50°C in Kelvin
       method='wlf',
       C1=17.44,
       C2=51.6,
       optimize_shifts=True  # Auto-optimize C1, C2
   )

   # Build master curve
   mastercurve, shift_factors = mc.transform(datasets)

   # Get optimized WLF parameters
   C1_opt, C2_opt = mc.get_wlf_parameters()
   print(f"Optimized WLF: C1={C1_opt:.2f}, C2={C2_opt:.2f} K")

   # Assess superposition quality
   overlap_error = mc.compute_overlap_error(datasets)
   print(f"Overlap error: {overlap_error:.4f}")

**Best For:**

* Polymer rheology (extending ω range from ~3 decades to 6-8 decades)
* Validating WLF/Arrhenius behavior
* Characterizing temperature-dependent relaxation
* Publications requiring master curves


Mutation Number
~~~~~~~~~~~~~~~

**Purpose:** Quantify the viscoelastic character of materials by computing a scalar index Δ ∈ [0, 1] from relaxation data, where Δ=0 is purely elastic and Δ=1 is purely viscous.

**Theory:**

The mutation number Δ is defined as:

.. math::

   \Delta = \frac{1}{G(0)} \int_{-\infty}^{\infty} \frac{dG}{d(\ln t)} d(\ln t)

For practical relaxation data:

.. math::

   \Delta = \frac{G(t=0) - G(t=\infty)}{G(t=0)}

**Physical Interpretation:**

.. list-table:: Mutation Number Interpretation
   :header-rows: 1
   :widths: 15 30 55

   * - Δ Value
     - Material Type
     - Physical Meaning
   * - Δ = 0
     - Pure elastic solid
     - No stress relaxation (G(t) = constant)
   * - 0 < Δ < 0.3
     - Solid-like
     - Weak relaxation, strong equilibrium modulus
   * - Δ ≈ 0.5
     - Balanced viscoelastic
     - Comparable elastic/viscous character
   * - 0.7 < Δ < 1
     - Liquid-like
     - Strong relaxation, weak/no equilibrium modulus
   * - Δ = 1
     - Pure viscous liquid
     - Complete stress relaxation (G(t→∞) = 0)

**Key Capabilities:**

* **Material Classification:** Rapid solid/liquid/gel identification
* **Gel Point Detection:** Δ ≈ 0.5-0.7 at sol-gel transition
* **Quality Control:** Consistent metric for batch comparison
* **Model Selection Aid:** Helps choose solid vs liquid models

**Input Requirements:**

* Relaxation data: G(t) vs time
* RheoData with ``test_mode='relaxation'``
* Sufficient time range to capture relaxation (ideally 4-5 decades)

**Output:**

* Scalar RheoData with Δ value in ``y`` attribute
* ``domain='scalar'`` for single-value result

**Key Parameters:**

.. list-table:: Mutation Number Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``integration_method``
     - ``'trapz'``
     - Numerical integration: 'trapz', 'simps', 'cumulative'
   * - ``extrapolate``
     - ``True``
     - Extrapolate to t→0 and t→∞
   * - ``extrapolation_model``
     - ``'power_law'``
     - Model for extrapolation: 'power_law', 'exponential'

**Example Usage:**

.. code-block:: python

   from rheojax.transforms import MutationNumber

   # Relaxation data
   t = np.logspace(-2, 4, 100)
   G_t = 1e5 * np.exp(-t / 10.0)  # Exponential relaxation

   data = RheoData(x=t, y=G_t, domain='time', initial_test_mode='relaxation')

   # Compute mutation number
   mutation = MutationNumber(extrapolate=True)
   result = mutation.transform(data)

   delta = result.y[0]
   print(f"Mutation Number Δ = {delta:.3f}")

   # Interpret
   if delta < 0.3:
       print("Material: Solid-like")
   elif delta < 0.7:
       print("Material: Viscoelastic")
   else:
       print("Material: Liquid-like")

**Best For:**

* Rapid material screening and classification
* Gel point detection in curing studies
* Quality control metrics
* Comparing viscoelastic character across samples


OWChirp
~~~~~~~

**Purpose:** Perform time-frequency analysis of Large Amplitude Oscillatory Shear (LAOS) data using Optimal Windowed Chirp transforms, extracting harmonics and nonlinear indicators.

**Key Capabilities:**

* **Time-Frequency Maps:** 2D spectrograms showing frequency content evolution
* **Harmonic Extraction:** Separate fundamental, 3rd, 5th, 7th harmonics
* **Nonlinearity Indicators:** I₃/I₁, I₅/I₁ ratios (higher harmonics/fundamental)
* **Fast Rheometry:** Time-resolved analysis of curing, gelation, structure formation
* **LAOS Analysis:** Quantify nonlinear viscoelastic response

**Theory:**

OWChirp uses a chirp wavelet transform optimized for oscillatory rheological signals:

.. math::

   W(t, \omega) = \int_{-\infty}^{\infty} s(\tau) \psi^*(\tau - t, \omega) d\tau

where ψ is an optimal wavelet matched to oscillatory strain/stress.

**Input Requirements:**

* Time-domain LAOS data: stress(t) or strain(t)
* Imposed oscillatory strain with known frequency ω₀
* Sufficient temporal resolution (sample rate >> ω₀)

**Output:**

* Frequency spectrum RheoData (1D)
* Time-frequency map via ``.get_time_frequency_map()`` (2D array)
* Harmonic components via ``.extract_harmonics()``

**Key Parameters:**

.. list-table:: OWChirp Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``n_frequencies``
     - ``100``
     - Number of frequency points in analysis
   * - ``frequency_range``
     - ``None``
     - (f_min, f_max); auto-detect if None
   * - ``wavelet_width``
     - ``6.0``
     - Width of Morlet wavelet (cycles)
   * - ``extract_harmonics``
     - ``True``
     - Automatically extract harmonics
   * - ``max_harmonic``
     - ``7``
     - Highest harmonic to extract (1, 3, 5, 7, ...)

**Example Usage:**

.. code-block:: python

   from rheojax.transforms import OWChirp
   import numpy as np

   # LAOS stress response (nonlinear)
   t = np.linspace(0, 10, 2000)
   omega_0 = 2 * np.pi * 1.0  # 1 Hz
   stress = (1e3 * np.sin(omega_0 * t) +
             300 * np.sin(3 * omega_0 * t) +  # 3rd harmonic
             50 * np.sin(5 * omega_0 * t))    # 5th harmonic

   data = RheoData(x=t, y=stress, domain='time')

   # OWChirp analysis
   owchirp = OWChirp(n_frequencies=200, extract_harmonics=True, max_harmonic=7)
   freq_data = owchirp.transform(data)

   # Extract harmonics
   harmonics = owchirp.extract_harmonics(freq_data, fundamental_freq=1.0)
   I1 = harmonics['I1']  # Fundamental
   I3 = harmonics['I3']  # 3rd harmonic
   I5 = harmonics['I5']  # 5th harmonic

   # Nonlinearity indicators
   print(f"I3/I1 = {I3/I1:.3f}")  # Typical: 0.01-0.3 for LAOS
   print(f"I5/I1 = {I5/I1:.3f}")

   # Time-frequency map
   tf_map = owchirp.get_time_frequency_map(data)

**Best For:**

* LAOS (Large Amplitude Oscillatory Shear) analysis
* Curing/gelation monitoring (time-resolved)
* Fast time-sweep rheometry
* Nonlinear viscoelastic characterization


Smooth Derivative
~~~~~~~~~~~~~~~~~

**Purpose:** Compute noise-robust numerical derivatives of rheological data using advanced smoothing techniques (Savitzky-Golay, spline, Total Variation).

**Key Capabilities:**

* **Noise Suppression:** Smoothing before/after differentiation
* **Multiple Methods:** Savitzky-Golay, spline, Total Variation (TV) regularization
* **Multi-Order:** 1st, 2nd, 3rd derivatives
* **Automatic Units:** Updates units for derivative quantity
* **Flexible Smoothing:** Pre-smooth, post-smooth, or both

**Methods:**

1. **Savitzky-Golay (Default):**

   * Polynomial fit over moving window
   * Preserves peak shapes
   * Best for: Smooth data with localized features
   * Parameters: ``window_length``, ``polyorder``

2. **Spline:**

   * Cubic spline interpolation + differentiation
   * Smooth continuous derivatives
   * Best for: Noisy data requiring aggressive smoothing
   * Parameters: ``smoothing_factor``

3. **Total Variation (TV):**

   * L1-norm regularization (edge-preserving)
   * Preserves discontinuities
   * Best for: Data with step changes or plateaus
   * Parameters: ``regularization_weight``

**Input Requirements:**

* Any RheoData (x, y) with sufficient data points
* Minimum points: ``window_length + 1`` for Savitzky-Golay

**Output:**

* Derivative RheoData with updated units
* Same x-axis (or slightly truncated for edge effects)

**Key Parameters:**

.. list-table:: Smooth Derivative Parameters
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``method``
     - ``'savgol'``
     - Method: 'savgol', 'spline', 'tv'
   * - ``window_length``
     - ``11``
     - Savitzky-Golay window (must be odd)
   * - ``polyorder``
     - ``3``
     - Savitzky-Golay polynomial order (<window_length)
   * - ``deriv``
     - ``1``
     - Derivative order (1, 2, 3)
   * - ``smooth_before``
     - ``True``
     - Smooth data before differentiation
   * - ``smooth_after``
     - ``False``
     - Smooth derivative after computation
   * - ``smooth_window``
     - ``5``
     - Window for post-smoothing

**Example Usage:**

.. code-block:: python

   from rheojax.transforms import SmoothDerivative
   import numpy as np

   # Noisy strain data
   t = np.linspace(0, 10, 200)
   strain = 0.1 * t**2 + np.random.normal(0, 0.01, size=t.shape)

   data = RheoData(x=t, y=strain, domain='time', units_x='s', units_y='dimensionless')

   # Compute strain rate (dε/dt) with Savitzky-Golay
   derivative_transform = SmoothDerivative(
       method='savgol',
       window_length=15,
       polyorder=3,
       deriv=1,
       smooth_before=True
   )

   strain_rate = derivative_transform.transform(data)
   print(f"Strain rate units: {strain_rate.units_y}")  # '1/s'

   # For very noisy data, use spline or TV
   derivative_tv = SmoothDerivative(method='tv', deriv=1)
   strain_rate_tv = derivative_tv.transform(data)

**Best For:**

* Computing strain rate from strain history
* Velocity/acceleration from position
* Derivative-based features for model fitting
* Noisy experimental data


Common Workflow Pipelines
--------------------------

Transform workflows combine multiple transforms in sequence to achieve complex analysis goals.

Workflow 1: Stress Relaxation → FFT → Fractional Model Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Convert relaxation data to frequency domain and fit a fractional model.

.. code-block:: python

   from rheojax.transforms import FFTAnalysis
   from rheojax.models import FractionalZenerSolidSolid

   # 1. Time-domain relaxation
   G_t_data = RheoData(x=t, y=G_t, domain='time', initial_test_mode='relaxation')

   # 2. FFT to frequency domain
   fft = FFTAnalysis(window='hann')
   G_star_data = fft.transform(G_t_data)

   # 3. Fit fractional model
   model = FractionalZenerSolidSolid()
   model.fit(G_star_data.x, G_star_data.y, test_mode='oscillation')

   print(f"Fitted α = {model.parameters.get_value('alpha'):.3f}")

**Why this workflow:**

* FFT converts relaxation G(t) to complex modulus G*(ω)
* Fractional models often fit better in frequency domain
* Broader frequency range from FFT than experimental oscillation


Workflow 2: Multi-Temperature Sweeps → Mastercurve → Model Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Build master curve from multi-T data, then fit a single model across extended frequency range.

.. code-block:: python

   from rheojax.transforms import Mastercurve
   from rheojax.models import FractionalMaxwellLiquid

   # 1. Multi-temperature datasets
   datasets = [data_25C, data_50C, data_75C, data_100C]

   # 2. Build mastercurve at 50°C
   mc = Mastercurve(reference_temp=323.15, method='wlf', optimize_shifts=True)
   mastercurve, shifts = mc.transform(datasets)

   # 3. Fit model to extended frequency range
   model = FractionalMaxwellLiquid()
   model.fit(mastercurve.x, mastercurve.y, test_mode='oscillation')

   # 4. Assess quality
   C1, C2 = mc.get_wlf_parameters()
   error = mc.compute_overlap_error(datasets)
   print(f"WLF: C1={C1:.2f}, C2={C2:.2f} K, Error={error:.4f}")

**Why this workflow:**

* Extends frequency range by 3-5 decades
* Single model fit captures temperature-invariant physics
* Validates WLF behavior


Workflow 3: Oscillation → Mutation Number → Model Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Classify material character, then select appropriate model family.

.. code-block:: python

   from rheojax.transforms import MutationNumber
   from rheojax.models import FractionalZenerSolidSolid, FractionalMaxwellLiquid

   # 1. Relaxation data
   G_t_data = RheoData(x=t, y=G_t, domain='time', initial_test_mode='relaxation')

   # 2. Compute mutation number
   mutation = MutationNumber()
   result = mutation.transform(G_t_data)
   delta = result.y[0]

   # 3. Select model based on Δ
   if delta < 0.5:
       print("Solid-like → Using Fractional Zener SS")
       model = FractionalZenerSolidSolid()
   else:
       print("Liquid-like → Using Fractional Maxwell Liquid")
       model = FractionalMaxwellLiquid()

   # 4. Fit selected model
   model.fit(t, G_t, test_mode='relaxation')

**Why this workflow:**

* Automated model selection based on physics
* Mutation number is fast and reliable classifier
* Improves convergence by using appropriate model


Workflow 4: LAOS Chirp → OWChirp FT → Extract Harmonics → Fit Nonlinear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Analyze LAOS data for nonlinear viscoelastic response.

.. code-block:: python

   from rheojax.transforms import OWChirp

   # 1. LAOS stress response
   stress_data = RheoData(x=t, y=stress, domain='time')

   # 2. OWChirp analysis
   owchirp = OWChirp(extract_harmonics=True, max_harmonic=7)
   freq_data = owchirp.transform(stress_data)

   # 3. Extract harmonics
   harmonics = owchirp.extract_harmonics(freq_data, fundamental_freq=omega_0/(2*np.pi))
   I1, I3, I5 = harmonics['I1'], harmonics['I3'], harmonics['I5']

   # 4. Nonlinearity indicators
   print(f"Nonlinearity: I3/I1={I3/I1:.3f}, I5/I1={I5/I1:.3f}")

   # 5. Time-frequency map for curing monitoring
   tf_map = owchirp.get_time_frequency_map(stress_data)

**Why this workflow:**

* OWChirp designed for LAOS signals
* Harmonic extraction quantifies nonlinear response
* Time-frequency map reveals structural evolution


Workflow 5: Strain History → Smooth Derivative → Strain Rate → Flow Model Fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal:** Compute strain rate from noisy strain data, then fit flow curve model.

.. code-block:: python

   from rheojax.transforms import SmoothDerivative
   from rheojax.models import PowerLaw

   # 1. Noisy strain history
   strain_data = RheoData(x=t, y=strain, domain='time', units_x='s', units_y='dimensionless')

   # 2. Smooth derivative → strain rate
   deriv = SmoothDerivative(method='savgol', window_length=15, deriv=1)
   strain_rate = deriv.transform(strain_data)

   # 3. Combine with stress data
   stress_data = RheoData(x=t, y=stress, domain='time')

   # 4. Create flow curve (η vs γ̇)
   eta = stress_data.y / strain_rate.y
   flow_curve = RheoData(x=strain_rate.y, y=eta, domain='rotation')

   # 5. Fit Power Law
   model = PowerLaw()
   model.fit(strain_rate.y, stress_data.y, test_mode='rotation')

   K = model.parameters.get_value('K')
   n = model.parameters.get_value('n')
   print(f"Power Law: K={K:.2e}, n={n:.3f}")

**Why this workflow:**

* Smooth derivative essential for noisy strain data
* Direct computation of flow curve from time-domain data
* Enables flow model fitting from non-standard tests


Transform Chaining & Compatibility
-----------------------------------

Which Transforms Can Be Chained?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Transform Chaining Compatibility Matrix
   :header-rows: 1
   :widths: 20 20 60

   * - Transform 1
     - Transform 2
     - Compatibility & Notes
   * - FFT
     - Mastercurve
     - ✓ FFT → frequency domain → merge multi-T → mastercurve
   * - FFT
     - OWChirp
     - ✗ Both do time→frequency (choose one based on need)
   * - FFT
     - Mutation Number
     - ✗ Mutation needs time-domain relaxation data
   * - FFT
     - Smooth Derivative
     - ✓ Smooth first → FFT (reduces spectral noise)
   * - Mastercurve
     - Smooth Derivative
     - ✓ Smooth individual sweeps before merging
   * - Mutation Number
     - Smooth Derivative
     - ✓ Smooth G(t) before computing Δ (recommended)
   * - OWChirp
     - Smooth Derivative
     - ✓ Smooth LAOS signal before OWChirp
   * - Smooth Derivative
     - Any
     - ✓ Pre-processing step for noisy data (chain first)

**General Rules:**

1. **Smooth Derivative is a pre-processor** - Apply first to reduce noise
2. **FFT and OWChirp are alternatives** - Both do time→frequency (pick based on use case)
3. **Mastercurve is a merge operation** - Combine multiple datasets, not chainable after FFT/OWChirp
4. **Mutation Number is standalone** - Terminal analysis step, no chaining after


Sequential Processing Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pipeline 1: Noise Reduction → FFT → Model Fit**

.. code-block:: python

   smooth = SmoothDerivative(method='savgol', deriv=0, window_length=11)  # deriv=0 = smoothing only
   fft = FFTAnalysis(window='hann')

   smoothed_data = smooth.transform(noisy_data)
   freq_data = fft.transform(smoothed_data)
   model.fit(freq_data.x, freq_data.y)

**Pipeline 2: Multi-Temperature → Smooth → Mastercurve → Fit**

.. code-block:: python

   smooth = SmoothDerivative(method='spline', deriv=0)
   mc = Mastercurve(reference_temp=323.15, method='wlf', optimize_shifts=True)

   # Smooth each temperature dataset
   smoothed_datasets = [smooth.transform(d) for d in datasets]

   # Build mastercurve
   mastercurve, shifts = mc.transform(smoothed_datasets)

   # Fit model
   model.fit(mastercurve.x, mastercurve.y)


Quality Checkpoints Between Stages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Checkpoint 1: After Smoothing**

.. code-block:: python

   # Verify smoothing didn't over-smooth
   plt.plot(original_data.x, original_data.y, 'o', label='Original')
   plt.plot(smoothed_data.x, smoothed_data.y, '-', label='Smoothed')
   plt.legend()

**Checkpoint 2: After FFT**

.. code-block:: python

   # Check for spectral leakage
   peaks = fft.detect_peaks(freq_data, n_peaks=5)
   print(f"Detected peaks at: {peaks['frequencies']} Hz")

**Checkpoint 3: After Mastercurve**

.. code-block:: python

   # Assess superposition quality
   error = mc.compute_overlap_error(datasets)
   if error > 0.05:
       print("WARNING: Poor superposition quality")

   # Plot shift factors
   temps, shifts = mc.get_shift_factors_array()
   plt.plot(temps, np.log10(shifts), 'o-')
   plt.xlabel('Temperature (K)')
   plt.ylabel('log(a_T)')


Parameter Selection Guidelines
-------------------------------

FFT Analysis
~~~~~~~~~~~~

.. list-table:: FFT Parameter Selection Guide
   :header-rows: 1
   :widths: 25 35 40

   * - Scenario
     - Recommended Settings
     - Rationale
   * - Clean periodic signal
     - ``window=None``, ``detrend=True``
     - No window needed; remove DC offset
   * - Noisy signal
     - ``window='hann'``, ``detrend=True``
     - Hann reduces spectral leakage
   * - Sharp transients
     - ``window='blackman'``
     - Blackman has better sidelobe suppression
   * - LAOS harmonics
     - ``window='hann'``, ``normalize=True``
     - Normalize for I₃/I₁ ratios
   * - Power spectral density
     - ``return_psd=True``, ``window='hann'``
     - PSD for energy distribution

Mastercurve
~~~~~~~~~~~

.. list-table:: Mastercurve Parameter Selection Guide
   :header-rows: 1
   :widths: 25 35 40

   * - Material Type
     - Recommended Settings
     - Rationale
   * - Polymer melts
     - ``method='wlf'``, ``optimize_shifts=True``
     - WLF designed for polymers; optimize for best fit
   * - Elastomers
     - ``method='wlf'``, ``C1=17.44``, ``C2=51.6``
     - Universal WLF constants (relative to T_g)
   * - Polymer solutions
     - ``method='arrhenius'``, ``optimize_shifts=True``
     - Arrhenius for simpler temperature dependence
   * - Unknown T_g
     - ``optimize_shifts=True``
     - Let optimizer find best C₁, C₂
   * - Known activation energy
     - ``method='arrhenius'``, ``E_a=<value>``
     - Use literature E_a value

Mutation Number
~~~~~~~~~~~~~~~

.. list-table:: Mutation Number Parameter Selection
   :header-rows: 1
   :widths: 25 35 40

   * - Data Quality
     - Recommended Settings
     - Rationale
   * - Clean, wide time range
     - ``extrapolate=False``
     - Data sufficient without extrapolation
   * - Limited time range
     - ``extrapolate=True``, ``extrapolation_model='power_law'``
     - Power-law extrapolation for fractional materials
   * - Exponential decay
     - ``extrapolation_model='exponential'``
     - Match classical Maxwell/Zener behavior
   * - Noisy data
     - Pre-smooth with SmoothDerivative first
     - Reduce integration errors

OWChirp
~~~~~~~

.. list-table:: OWChirp Parameter Selection Guide
   :header-rows: 1
   :widths: 25 35 40

   * - Application
     - Recommended Settings
     - Rationale
   * - LAOS harmonics
     - ``extract_harmonics=True``, ``max_harmonic=7``
     - Capture fundamental + odd harmonics
   * - Time-resolved curing
     - ``n_frequencies=200``, ``wavelet_width=6.0``
     - High resolution for evolving spectra
   * - Fast screening
     - ``n_frequencies=50``, ``extract_harmonics=True``
     - Lower resolution for speed
   * - Known fundamental
     - Specify ``fundamental_freq`` in ``extract_harmonics()``
     - Direct harmonic extraction

Smooth Derivative
~~~~~~~~~~~~~~~~~

.. list-table:: Smooth Derivative Parameter Selection
   :header-rows: 1
   :widths: 25 35 40

   * - Data Characteristics
     - Recommended Settings
     - Rationale
   * - Low noise, smooth
     - ``method='savgol'``, ``window_length=7``, ``polyorder=3``
     - Small window preserves features
   * - High noise
     - ``method='savgol'``, ``window_length=15-21``, ``polyorder=3``
     - Larger window for aggressive smoothing
   * - Very noisy
     - ``method='spline'`` or ``method='tv'``
     - Advanced methods for heavy noise
   * - Step changes
     - ``method='tv'``
     - Total Variation preserves edges
   * - Higher derivatives
     - ``polyorder >= deriv + 2``
     - Ensure polynomial order sufficient


Next Steps
----------

* **Detailed transform documentation:** :doc:`/transforms/index` for individual transform pages
* **User guide:** :doc:`/user_guide/transforms` for usage patterns and examples
* **API reference:** :doc:`/api/transforms` for complete API documentation
* **Example notebooks:** 6 transform examples in ``examples/transforms/`` directory
* **Pipeline integration:** :doc:`/user_guide/pipeline_api` for fluent transform chaining

**Need a transform not listed?** Transforms are extensible via ``BaseTransform`` - see :doc:`/developer/contributing`.
