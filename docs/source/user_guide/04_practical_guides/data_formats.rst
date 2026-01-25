.. _data_formats:

Data Format Reference
=====================

.. admonition:: Purpose
   :class: note

   This reference documents the precise data format requirements for all RheoJAX
   fitting analyses and transforms. Use this as a technical specification when
   preparing data for analysis.

Core Data Container: RheoData
-----------------------------

All RheoJAX analyses use the :class:`~rheojax.core.data.RheoData` container for input data.

Constructor Signature
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.core.data import RheoData

   data = RheoData(
       x,                          # Independent variable (time, frequency, shear rate)
       y,                          # Dependent variable (real or complex)
       domain='time',              # 'time' or 'frequency'
       x_units=None,               # e.g., 's', 'rad/s', '1/s'
       y_units=None,               # e.g., 'Pa', '1/Pa', 'Pa*s'
       initial_test_mode=None,     # 'relaxation', 'creep', 'oscillation', 'rotation'
       metadata=None,              # dict with additional context
   )

Key Fields
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``x``
     - array-like
     - Independent variable (time, angular frequency, or shear rate)
   * - ``y``
     - array-like
     - Dependent variable; can be **real** or **complex** (for oscillation data)
   * - ``domain``
     - str
     - ``'time'`` for time-domain data, ``'frequency'`` for frequency-domain
   * - ``x_units``
     - str | None
     - Units of x (e.g., ``'s'``, ``'rad/s'``, ``'1/s'``)
   * - ``y_units``
     - str | None
     - Units of y (e.g., ``'Pa'``, ``'1/Pa'``, ``'Pa*s'``)
   * - ``initial_test_mode``
     - str | None
     - Explicit test mode; auto-detected if not provided
   * - ``metadata``
     - dict | None
     - Additional context (temperature, strain amplitude, etc.)

Oscillation Data Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For complex modulus data (oscillation), ``RheoData`` provides convenient properties:

.. code-block:: python

   # Access real and imaginary parts
   G_prime = data.storage_modulus   # G' (also: data.y_real)
   G_double_prime = data.loss_modulus  # G'' (also: data.y_imag)

   # Loss tangent
   tan_delta = data.tan_delta       # G'' / G'

Fitting Analyses: The Four Test Modes
-------------------------------------

Relaxation
~~~~~~~~~~

Stress relaxation measures how stress decays over time under constant strain.

**Physical setup**: Apply step strain, measure stress decay

**Equation**: :math:`\sigma(t) = G(t) \cdot \gamma_0`

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - time ``t`` (s)
     - Positive, monotonically increasing
   * - **y**
     - relaxation modulus ``G(t)`` (Pa)
     - Real, monotonically **decreasing**
   * - **domain**
     - ``'time'``
     - Required
   * - **test_mode**
     - ``'relaxation'``
     - Auto-detected if y decreases

**Example**:

.. code-block:: python

   import numpy as np
   from rheojax.core.data import RheoData
   from rheojax.models import Maxwell

   # Relaxation data: G(t) decays exponentially
   t = np.logspace(-2, 2, 100)  # 0.01 to 100 seconds
   G_t = 1e5 * np.exp(-t / 10.0)  # G0=100 kPa, tau=10 s

   data = RheoData(
       x=t,
       y=G_t,
       domain='time',
       x_units='s',
       y_units='Pa',
       initial_test_mode='relaxation',
   )

   # Fit Maxwell model
   model = Maxwell()
   model.fit(data)

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models, Generalized Maxwell

Creep
~~~~~

Creep measures how strain increases over time under constant stress.

**Physical setup**: Apply step stress, measure strain increase

**Equation**: :math:`\gamma(t) = J(t) \cdot \sigma_0`

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - time ``t`` (s)
     - Positive, monotonically increasing
   * - **y**
     - creep compliance ``J(t)`` (1/Pa)
     - Real, monotonically **increasing**
   * - **domain**
     - ``'time'``
     - Required
   * - **test_mode**
     - ``'creep'``
     - Auto-detected if y increases

**Example**:

.. code-block:: python

   # Creep data: J(t) increases toward steady state
   t = np.logspace(-2, 3, 100)  # 0.01 to 1000 seconds
   J_t = 1e-5 * (1 - np.exp(-t / 10.0)) + 1e-8 * t  # Elastic + viscous

   data = RheoData(
       x=t,
       y=J_t,
       domain='time',
       x_units='s',
       y_units='1/Pa',
       initial_test_mode='creep',
   )

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models

Oscillation (SAOS)
~~~~~~~~~~~~~~~~~~

Small-amplitude oscillatory shear measures frequency-dependent viscoelasticity.

**Physical setup**: Apply sinusoidal strain, measure sinusoidal stress response

**Output**: Complex modulus :math:`G^*(\omega) = G'(\omega) + i G''(\omega)`

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - angular frequency ``omega`` (rad/s)
     - Positive, typically log-spaced
   * - **y**
     - complex modulus ``G*(omega)`` (Pa)
     - **Complex**: ``G' + 1j * G''``
   * - **domain**
     - ``'frequency'``
     - Required
   * - **test_mode**
     - ``'oscillation'``
     - Auto-detected from domain/units

**Example with complex data**:

.. code-block:: python

   # SAOS data: G*(omega) = G'(omega) + i*G''(omega)
   omega = np.logspace(-2, 2, 50)  # 0.01 to 100 rad/s

   # Example: Maxwell model response
   G0, tau = 1e5, 1.0
   G_prime = G0 * (omega * tau)**2 / (1 + (omega * tau)**2)
   G_double_prime = G0 * (omega * tau) / (1 + (omega * tau)**2)

   # Create complex modulus
   G_star = G_prime + 1j * G_double_prime

   data = RheoData(
       x=omega,
       y=G_star,
       domain='frequency',
       x_units='rad/s',
       y_units='Pa',
   )

   # Access components
   print(f"G' range: {data.storage_modulus.min():.0f} - {data.storage_modulus.max():.0f} Pa")
   print(f"G'' range: {data.loss_modulus.min():.0f} - {data.loss_modulus.max():.0f} Pa")

**Alternative: separate G' and G'' arrays**:

.. code-block:: python

   # If you have separate arrays, combine them:
   G_star = G_prime + 1j * G_double_prime
   data = RheoData(x=omega, y=G_star, domain='frequency')

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models, SGR models, Generalized Maxwell

Rotation (Steady Shear Flow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Steady shear flow measures viscosity as a function of shear rate.

**Physical setup**: Apply constant shear rate, measure steady-state stress

**Output**: Viscosity :math:`\eta(\dot{\gamma})` or stress :math:`\sigma(\dot{\gamma})`

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - shear rate :math:`\dot{\gamma}` (1/s)
     - Positive
   * - **y**
     - viscosity :math:`\eta` (Pa*s) or stress :math:`\sigma` (Pa)
     - Real
   * - **domain**
     - ``'time'`` (or unspecified)
     - Technically "shear_rate" domain
   * - **test_mode**
     - ``'rotation'``
     - Auto-detected from units

**Example**:

.. code-block:: python

   from rheojax.models import PowerLaw, CarreauYasuda

   # Flow curve: viscosity vs shear rate
   gamma_dot = np.logspace(-2, 3, 50)  # 0.01 to 1000 1/s

   # Shear-thinning behavior
   eta = 1000 * gamma_dot**(-0.5)  # Power-law fluid

   data = RheoData(
       x=gamma_dot,
       y=eta,
       x_units='1/s',
       y_units='Pa*s',
       initial_test_mode='rotation',
   )

   # Fit flow models
   model = PowerLaw()
   model.fit(data)

**Compatible models**: PowerLaw, HerschelBulkley, Bingham, Carreau, CarreauYasuda, Cross

Transform Analyses
------------------

Mastercurve (Time-Temperature Superposition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collapses frequency sweeps at different temperatures onto a master curve.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - ``list[RheoData]`` at different temperatures
   * - **Each dataset**
     - Frequency sweep (omega vs G*)
   * - **Required metadata**
     - ``{'temperature': T_kelvin}``
   * - **domain**
     - ``'frequency'``
   * - **Output**
     - Collapsed mastercurve + shift factors dict

**Example**:

.. code-block:: python

   from rheojax.transforms import Mastercurve

   # Create datasets at different temperatures
   datasets = [
       RheoData(
           x=omega,
           y=G_star_273K,
           domain='frequency',
           metadata={'temperature': 273.15},  # Required!
       ),
       RheoData(
           x=omega,
           y=G_star_298K,
           domain='frequency',
           metadata={'temperature': 298.15},
       ),
       RheoData(
           x=omega,
           y=G_star_323K,
           domain='frequency',
           metadata={'temperature': 323.15},
       ),
   ]

   # Apply TTS
   mc = Mastercurve(reference_temp=298.15, method='wlf')
   mastercurve, shift_factors = mc.transform(datasets)

   # Or with auto-shift optimization
   mc_auto = Mastercurve(reference_temp=298.15, auto_shift=True)
   mastercurve, shifts = mc_auto.transform(datasets)

SRFS (Strain-Rate Frequency Superposition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Collapses flow curves at different reference shear rates (for SGR materials).

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - ``list[RheoData]`` at different reference shear rates
   * - **Each dataset**
     - Flow curve (shear rate vs viscosity)
   * - **Required metadata**
     - ``{'reference_gamma_dot': gamma_dot_ref}``
   * - **Required params**
     - ``x`` (SGR noise temp), ``tau0`` (attempt time)
   * - **Output**
     - Collapsed mastercurve + shift factors

**Example**:

.. code-block:: python

   from rheojax.transforms import SRFS

   datasets = [
       RheoData(
           x=gamma_dot_1,
           y=eta_1,
           metadata={'reference_gamma_dot': 0.1},  # Required!
       ),
       RheoData(
           x=gamma_dot_2,
           y=eta_2,
           metadata={'reference_gamma_dot': 1.0},
       ),
       RheoData(
           x=gamma_dot_3,
           y=eta_3,
           metadata={'reference_gamma_dot': 10.0},
       ),
   ]

   # Apply SRFS with SGR parameters
   srfs = SRFS(reference_gamma_dot=1.0)
   mastercurve, shifts = srfs.transform(
       datasets,
       x=1.5,       # SGR noise temperature
       tau0=1e-3,   # SGR attempt time
       return_shifts=True,
   )

OWChirp (LAOS Wavelet Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performs time-frequency analysis of Large Amplitude Oscillatory Shear data.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - Time-domain LAOS stress signal
   * - **x**
     - time ``t`` (s)
   * - **y**
     - stress :math:`\sigma(t)` (Pa) - **real**
   * - **domain**
     - ``'time'`` (required)
   * - **Output**
     - Frequency spectrum + harmonic amplitudes

**Example**:

.. code-block:: python

   from rheojax.transforms import OWChirp

   # LAOS stress response (nonlinear, with harmonics)
   omega = 1.0  # rad/s
   t = np.linspace(0, 10 * 2 * np.pi / omega, 10000)
   stress = 100 * np.sin(omega * t) + 20 * np.sin(3 * omega * t)  # With 3rd harmonic

   data = RheoData(
       x=t,
       y=stress,
       domain='time',
       x_units='s',
       y_units='Pa',
       metadata={'test_mode': 'oscillation'},
   )

   # Apply OWChirp transform
   owchirp = OWChirp(
       n_frequencies=100,
       extract_harmonics=True,
       max_harmonic=7,
   )
   spectrum = owchirp.transform(data)

   # Extract harmonic content
   harmonics = owchirp.get_harmonics(data)
   print(f"Fundamental: {harmonics['fundamental']}")
   print(f"Third harmonic: {harmonics['third']}")

SPP Decomposer (LAOS Yield Stress Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Applies Sequence of Physical Processes (SPP) decomposition to extract yield stresses
and nonlinear viscoelastic parameters from LAOS data.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - Time-domain LAOS stress signal
   * - **x**
     - time ``t`` (s)
   * - **y**
     - stress :math:`\sigma(t)` (Pa) - **real**
   * - **domain**
     - ``'time'`` (required)
   * - **Required params**
     - ``omega`` (rad/s), ``gamma_0`` (strain amplitude)
   * - **Optional metadata**
     - ``'strain'``, ``'strain_rate'``, ``'omega'``, ``'gamma_0'``
   * - **Output**
     - Decomposed stress + SPP metrics (yield stresses, Lissajous metrics)

**Example**:

.. code-block:: python

   from rheojax.transforms import SPPDecomposer

   omega = 1.0     # rad/s
   gamma_0 = 1.0   # strain amplitude

   # Time and LAOS stress data
   t = np.linspace(0, 4 * np.pi, 2000)
   strain = gamma_0 * np.sin(omega * t)
   stress = 100.0 * strain + 20.0 * np.sin(3 * omega * t)  # Nonlinear response

   data = RheoData(
       x=t,
       y=stress,
       domain='time',
       x_units='s',
       y_units='Pa',
       metadata={
           'test_mode': 'oscillation',
           'omega': omega,
           'gamma_0': gamma_0,
           'strain': strain,  # Optional: measured strain
       },
   )

   # Apply SPP decomposition
   spp = SPPDecomposer(
       omega=omega,
       gamma_0=gamma_0,
       n_harmonics=39,
   )
   result = spp.transform(data)

   # Access results
   sigma_sy, sigma_dy = spp.get_yield_stresses()
   print(f"Static yield stress: {sigma_sy:.2f} Pa")
   print(f"Dynamic yield stress: {sigma_dy:.2f} Pa")

   metrics = spp.get_nonlinearity_metrics()
   print(f"I3/I1 ratio: {metrics['I3_I1_ratio']:.4f}")
   print(f"S-factor: {metrics['S_factor']:.4f}")
   print(f"T-factor: {metrics['T_factor']:.4f}")

FFT Analysis (Spectral Analysis)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Converts time-domain signals to frequency domain using Fast Fourier Transform.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - Time-domain signal (any test mode)
   * - **x**
     - time ``t`` (s)
   * - **y**
     - signal (Pa, 1/Pa, etc.) - **real**
   * - **domain**
     - ``'time'`` (required)
   * - **Output**
     - Frequency spectrum (magnitude or PSD)

**Example**:

.. code-block:: python

   from rheojax.transforms import FFTAnalysis

   # Time-domain relaxation data
   t = np.linspace(0, 100, 10000)
   G_t = 1e5 * np.exp(-t / 10.0)

   data = RheoData(x=t, y=G_t, domain='time')

   # Apply FFT analysis
   fft = FFTAnalysis(
       window='hann',
       detrend=True,
       return_psd=False,
       normalize=True,
   )
   freq_data = fft.transform(data)

   # Find characteristic time from peak frequency
   tau_char = fft.get_characteristic_time(freq_data)
   print(f"Characteristic time: {tau_char:.2f} s")

Mutation Number (Relaxation Quantification)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Calculates the mutation number from relaxation modulus data to quantify
the degree of time-dependence.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - Relaxation modulus data
   * - **x**
     - time ``t`` (s)
   * - **y**
     - relaxation modulus ``G(t)`` (Pa)
   * - **domain**
     - ``'time'``
   * - **test_mode**
     - ``'relaxation'``
   * - **Output**
     - Mutation number :math:`\Delta \in [0, 1]`

**Interpretation**:

- :math:`\Delta \to 0`: Elastic solid (no relaxation)
- :math:`\Delta \to 1`: Viscous fluid (complete relaxation)

**Example**:

.. code-block:: python

   from rheojax.transforms import MutationNumber

   data = RheoData(
       x=t,
       y=G_t,
       domain='time',
       initial_test_mode='relaxation',
   )

   mutation = MutationNumber(integration_method='trapz')
   delta = mutation.calculate(data)
   print(f"Mutation number: {delta:.4f}")

Smooth Derivative (Numerical Differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes noise-robust derivatives using Savitzky-Golay filtering.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Requirement
     - Description
   * - **Input**
     - Any time or frequency domain data
   * - **x**
     - Independent variable
   * - **y**
     - Dependent variable (**real**)
   * - **Output**
     - Smoothed derivative dy/dx

**Example**:

.. code-block:: python

   from rheojax.transforms import SmoothDerivative

   # Noisy creep compliance data
   J_t = t + 0.1 * np.random.randn(len(t))
   data = RheoData(x=t, y=J_t, domain='time')

   # Compute smooth first derivative
   deriv = SmoothDerivative(
       method='savgol',
       window_length=11,
       polyorder=3,
       deriv=1,
   )
   dJ_dt = deriv.transform(data)

   # Compute second derivative
   deriv2 = SmoothDerivative(window_length=15, polyorder=4, deriv=2)
   d2J_dt2 = deriv2.transform(data)

Quick Reference Table
---------------------

.. list-table::
   :header-rows: 1
   :widths: 15 12 13 20 20 20

   * - Analysis
     - Test Mode
     - Domain
     - x
     - y
     - Key Metadata
   * - **Relaxation**
     - ``relaxation``
     - ``time``
     - t (s)
     - G(t) (Pa)
     - --
   * - **Creep**
     - ``creep``
     - ``time``
     - t (s)
     - J(t) (1/Pa)
     - --
   * - **Oscillation**
     - ``oscillation``
     - ``frequency``
     - omega (rad/s)
     - G*(omega) (Pa) **complex**
     - --
   * - **Rotation**
     - ``rotation``
     - ``time``
     - gamma_dot (1/s)
     - eta (Pa*s) or sigma (Pa)
     - --
   * - **Mastercurve**
     - --
     - ``frequency``
     - omega
     - G*
     - ``temperature``
   * - **SRFS**
     - --
     - --
     - gamma_dot
     - eta
     - ``reference_gamma_dot``
   * - **OWChirp**
     - --
     - ``time``
     - t
     - sigma(t)
     - --
   * - **SPP**
     - --
     - ``time``
     - t
     - sigma(t)
     - ``omega``, ``gamma_0``, ``strain``
   * - **FFT**
     - --
     - ``time``
     - t
     - signal
     - --
   * - **Mutation**
     - ``relaxation``
     - ``time``
     - t
     - G(t)
     - --
   * - **Derivative**
     - --
     - any
     - x
     - y
     - --

Common Patterns
---------------

JAX Array Conversion
~~~~~~~~~~~~~~~~~~~~

``RheoData`` stores data as JAX arrays internally. Access NumPy arrays when needed:

.. code-block:: python

   # Data is stored as JAX arrays
   data = RheoData(x=t, y=G_t, domain='time')

   # Convert to NumPy when needed
   import numpy as np
   t_numpy = np.asarray(data.x)
   G_numpy = np.asarray(data.y)

   # Or use the convenience method
   t_np, G_np = data.to_numpy()

Test Mode Auto-Detection
~~~~~~~~~~~~~~~~~~~~~~~~

RheoJAX can auto-detect test modes based on data characteristics:

.. code-block:: python

   # Auto-detection based on y behavior
   data = RheoData(x=t, y=G_decaying, domain='time')
   print(data.test_mode)  # 'relaxation' (y decreases)

   data = RheoData(x=t, y=J_increasing, domain='time')
   print(data.test_mode)  # 'creep' (y increases)

   data = RheoData(x=omega, y=G_star, domain='frequency')
   print(data.test_mode)  # 'oscillation' (frequency domain)

Explicit test mode specification is recommended for clarity:

.. code-block:: python

   data = RheoData(
       x=t,
       y=G_t,
       domain='time',
       initial_test_mode='relaxation',  # Explicit is better
   )

Working with Complex Data
~~~~~~~~~~~~~~~~~~~~~~~~~

For oscillation data, you can construct complex arrays in several ways:

.. code-block:: python

   import numpy as np

   # Method 1: Direct complex array
   G_star = G_prime + 1j * G_double_prime
   data = RheoData(x=omega, y=G_star, domain='frequency')

   # Method 2: From magnitude and phase
   G_magnitude = np.sqrt(G_prime**2 + G_double_prime**2)
   delta = np.arctan2(G_double_prime, G_prime)
   G_star = G_magnitude * np.exp(1j * delta)
   data = RheoData(x=omega, y=G_star, domain='frequency')

   # Access components
   print(data.storage_modulus)  # G'
   print(data.loss_modulus)     # G''
   print(data.tan_delta)        # tan(delta) = G''/G'

Further Reading
---------------

- :doc:`data_io` -- Loading data from instrument files
- :doc:`../01_fundamentals/test_modes` -- Conceptual overview of test modes
- :doc:`/api/core` -- Full API reference for RheoData
- :doc:`/api/transforms` -- Full API reference for transforms
