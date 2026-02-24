.. _data_formats:

Data Format Reference
=====================

.. admonition:: Purpose
   :class: note

   This reference documents the precise data format requirements for all RheoJAX
   fitting analyses and transforms. Use this as a technical specification when
   preparing data for analysis.

   **Version**: v0.6.0 — 53 models | 7 transforms | 5 readers | 3 writers

.. contents:: On this page
   :local:
   :depth: 2

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
       x_units=None,               # e.g., 's', 'rad/s', '1/s', 'Hz'
       y_units=None,               # e.g., 'Pa', '1/Pa', 'Pa·s'
       initial_test_mode=None,     # 'relaxation', 'creep', 'oscillation', 'flow_curve',
                                   # 'startup', 'laos', 'rotation' (legacy)
       metadata=None,              # dict with additional context
       validate=True,              # Validate on creation (NaN, shape, monotonicity)
   )

**Accepted array types**: NumPy ndarray, JAX ndarray, list, tuple (auto-coerced)

**Float precision**: float64 enforced (critical for NLSQ numerical stability)

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
     - Units of y (e.g., ``'Pa'``, ``'1/Pa'``, ``'Pa·s'``)
   * - ``initial_test_mode``
     - str | None
     - Explicit test mode; auto-detected if not provided
   * - ``metadata``
     - dict | None
     - Additional context (temperature, deformation_mode, poisson_ratio, etc.)
   * - ``validate``
     - bool
     - If True (default), validates NaN, Inf, shape matching on creation

Data Access Properties
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Property
     - Type
     - Description
   * - ``.test_mode``
     - str
     - Auto-detected or explicit test mode
   * - ``.deformation_mode``
     - str
     - From metadata; defaults to ``"shear"``
   * - ``.shape``
     - tuple
     - Shape of y data
   * - ``.is_complex``
     - bool
     - True if y is complex

**Complex modulus properties** (for oscillation data where y = G' + iG''):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Description
   * - ``.storage_modulus`` / ``.y_real``
     - G' (storage modulus) or E' for DMTA
   * - ``.loss_modulus`` / ``.y_imag``
     - G'' (loss modulus) or E'' for DMTA
   * - ``.modulus``
     - |G*| = sqrt(G'² + G''²)
   * - ``.tan_delta``
     - tan(δ) = G''/G'
   * - ``.storage_modulus_label``
     - ``"G'"`` (shear) or ``"E'"`` (tensile)
   * - ``.loss_modulus_label``
     - ``"G''"`` (shear) or ``"E''"`` (tensile)

Methods
~~~~~~~

**Conversion**:

.. code-block:: python

   jax_data = data.to_jax()          # Convert to JAX arrays (cached)
   np_data = data.to_numpy()         # Convert to NumPy arrays (zero-copy when possible)
   data_copy = data.copy()           # Deep copy
   d = data.to_dict()                # Serialize to dict
   data = RheoData.from_dict(d)      # Deserialize

**Data manipulation**:

.. code-block:: python

   resampled = data.resample(n_points=200)      # Resample (log-spaced freq, linear time)
   interped = data.interpolate(new_x)           # Interpolate to new x values
   smoothed = data.smooth(window_size=5)         # Moving average
   deriv = data.derivative()                     # Numerical dy/dx
   integ = data.integral()                       # Cumulative trapezoid
   sliced = data.slice(start=0.1, end=100.0)    # Slice by x-value range

**Operators**: indexing ``[i]``, slicing ``[a:b]``, ``+``, ``-``, ``*``

Validation Rules
~~~~~~~~~~~~~~~~

When ``validate=True`` (default), RheoData checks:

- ``x`` and ``y`` must have the same shape → ``ValueError``
- No NaN values → ``ValueError``
- No Inf values → ``ValueError``
- Non-monotonic x → Warning (not error)
- Negative frequency values in oscillation domain → Warning

Disable validation for intermediate computation: ``RheoData(..., validate=False)``

.. _test_modes_reference:

Test Modes: 8 Protocols
-----------------------

RheoJAX supports 8 test modes, corresponding to different rheological experiments.
Models declare which protocols they support via the registry.

.. code-block:: python

   from rheojax.core.test_modes import TestModeEnum, DeformationMode

   # All test modes
   TestModeEnum.RELAXATION     # "relaxation"
   TestModeEnum.CREEP          # "creep"
   TestModeEnum.OSCILLATION    # "oscillation"
   TestModeEnum.FLOW_CURVE     # "flow_curve"
   TestModeEnum.STARTUP        # "startup"
   TestModeEnum.LAOS           # "laos"
   TestModeEnum.ROTATION       # "rotation" (legacy alias for flow_curve)
   TestModeEnum.UNKNOWN        # "unknown"

**Auto-detection priority**:

1. Explicit ``metadata['test_mode']`` if provided
2. Domain + units: ``domain='frequency'`` or ``x_units='rad/s'`` → oscillation
3. ``x_units`` containing ``'1/s'`` or ``'s^-1'`` → flow_curve
4. Monotonicity: decreasing y → relaxation, increasing y → creep
5. Fallback → unknown

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

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models, Generalized Maxwell,
IKH, FIKH, DMT, SGR, HVM, HVNM, TNT, VLB, ITT-MCT, and more (40+ models)

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
   * - **Kwargs**
     - ``sigma_applied`` (float, Pa)
     - Required for ODE models (IKH, DMT, EPM, etc.)

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

   # For ODE models, sigma_applied is required
   model.fit(t, J_t, test_mode='creep', sigma_applied=100.0)

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models, IKH, DMT, EPM,
Fluidity-Saramito, HVM, HVNM, TNT, VLB, ITT-MCT

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

**Compatible models**: Maxwell, Zener, Springpot, all Fractional models, SGR models,
Generalized Maxwell, IKH, FIKH, DMT, Giesekus, HVM, HVNM, TNT, VLB, ITT-MCT (41+ models)

DMTA / DMA Oscillation (Tensile)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dynamic Mechanical Thermal Analysis measures frequency-dependent viscoelasticity in tension, bending, or compression.

**Physical setup**: Apply sinusoidal tensile (or bending) deformation, measure force response

**Output**: Complex Young's modulus :math:`E^*(\omega) = E'(\omega) + i E''(\omega)`

**Conversion**: :math:`E^* = 2(1 + \nu) \cdot G^*` where :math:`\nu` is Poisson's ratio

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
     - complex Young's modulus ``E*(omega)`` (Pa)
     - **Complex**: ``E' + 1j * E''``
   * - **domain**
     - ``'frequency'``
     - Required
   * - **test_mode**
     - ``'oscillation'``
     - Same as shear SAOS
   * - **deformation_mode**
     - ``'tension'``
     - Passed to ``fit()`` / ``predict()``
   * - **poisson_ratio**
     - float (e.g., 0.5 for rubber)
     - Required for E* → G* conversion

**Deformation modes**:

.. code-block:: python

   from rheojax.core.test_modes import DeformationMode

   DeformationMode.SHEAR         # "shear" — rotational rheometer (G*)
   DeformationMode.TENSION       # "tension" — DMTA tensile (E*)
   DeformationMode.BENDING       # "bending" — DMA bending (E*)
   DeformationMode.COMPRESSION   # "compression" — compression DMA (E*)

**Poisson's ratio presets**:

.. list-table::
   :header-rows: 1
   :widths: 40 20

   * - Material
     - ν
   * - rubber / elastomer / hydrogel
     - 0.50
   * - semicrystalline
     - 0.40
   * - thermoset
     - 0.38
   * - glassy_polymer
     - 0.35
   * - metal / foam
     - 0.30

**Example with E* data**:

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   # DMTA data: E*(omega) = E'(omega) + i*E''(omega)
   omega = np.logspace(-2, 2, 50)
   E_prime = 3e9 * np.ones(50)
   E_double_prime = 1e8 * omega**0.3
   E_star = E_prime + 1j * E_double_prime

   # Fit — model converts E* → G* internally
   model = FractionalZenerSolidSolid()
   model.fit(omega, E_star,
             test_mode='oscillation',
             deformation_mode='tension',
             poisson_ratio=0.5)

   # predict() returns E* automatically
   E_pred = model.predict(omega, test_mode='oscillation')

**CSV auto-detection**: The CSV reader auto-detects ``E'`` and ``E''`` columns
and sets ``deformation_mode='tension'`` in metadata:

.. code-block:: python

   from rheojax.io import load_csv

   data = load_csv("dmta_data.csv", x_col="f", y_cols=["E' (Pa)", "E'' (Pa)"])
   print(data.deformation_mode)  # "tension" (auto-detected)

**Compatible models**: All 41 oscillation-capable models (Classical, Fractional, SGR, IKH, HVM, HVNM, etc.)

See :doc:`/models/dmta/index` for the complete DMTA guide.

Flow Curve (Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

Steady shear flow measures stress as a function of shear rate at equilibrium.

**Physical setup**: Apply constant shear rate, measure steady-state stress

**Output**: Stress :math:`\sigma(\dot{\gamma})` or viscosity :math:`\eta(\dot{\gamma})`

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - shear rate :math:`\dot{\gamma}` (1/s)
     - Positive, typically log-spaced
   * - **y**
     - stress :math:`\sigma` (Pa) or viscosity :math:`\eta` (Pa·s)
     - Real
   * - **domain**
     - ``'time'``
     -
   * - **test_mode**
     - ``'flow_curve'``
     - Auto-detected from x_units

.. note::

   The legacy test mode ``'rotation'`` is automatically converted to ``'flow_curve'``.
   New code should always use ``'flow_curve'``.

**Example**:

.. code-block:: python

   from rheojax.models import PowerLaw, HerschelBulkley

   # Flow curve: stress vs shear rate
   gamma_dot = np.logspace(-2, 3, 50)  # 0.01 to 1000 1/s
   sigma = 50.0 + 10.0 * gamma_dot**0.5  # Herschel-Bulkley-like

   data = RheoData(
       x=gamma_dot,
       y=sigma,
       x_units='1/s',
       y_units='Pa',
       initial_test_mode='flow_curve',
   )

   # Fit Herschel-Bulkley model
   model = HerschelBulkley()
   model.fit(data)

**Compatible models**: PowerLaw, HerschelBulkley, Bingham, Carreau, CarreauYasuda, Cross,
plus all ODE models (Giesekus, TNT, DMT, IKH, Fluidity, SGR, HVM, STZ, etc.) — 32+ models

Startup (Transient Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~

Startup flow measures the transient stress response when a constant shear rate
is suddenly applied. The stress often shows an **overshoot** before reaching
steady state — a signature of thixotropy or polymer entanglement.

**Physical setup**: Apply constant shear rate at t=0, measure stress growth

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - stacked ``[t, γ]``, shape ``(2, N)``
     - Time (s) and cumulative strain (γ = γ̇·t)
   * - **y**
     - transient stress ``σ(t)`` (Pa)
     - Real, 1D array of shape ``(N,)``
   * - **domain**
     - ``'time'``
     -
   * - **test_mode**
     - ``'startup'``
     - Must be specified explicitly
   * - **gamma_dot**
     - float (1/s), **REQUIRED** kwarg
     - Constant applied shear rate

.. important::

   The ``gamma_dot`` keyword argument is **required** for startup mode. It is
   used both during fitting and during Bayesian inference (NUTS). Models cache
   it internally for the ``model_function()`` used by NumPyro.

**Example**:

.. code-block:: python

   from rheojax.models import TNTSingleMode

   gamma_dot = 10.0  # 1/s
   t = np.linspace(0, 5.0, 80)
   gamma = gamma_dot * t

   # Stack time and strain as (2, N) array
   X = np.stack([t, gamma])  # shape (2, 80)

   # Fit model with required gamma_dot kwarg
   model = TNTSingleMode()
   model.fit(X, sigma, test_mode='startup', gamma_dot=gamma_dot)

   # Predict
   sigma_pred = model.predict(X, test_mode='startup', gamma_dot=gamma_dot)

**Compatible models**: TNT (5 variants), DMT, IKH, FIKH, EPM, Fluidity, Fluidity-Saramito,
Giesekus, HVM, HVNM, STZ, VLB, ITT-MCT, HL — 26+ models

LAOS (Large Amplitude Oscillatory Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LAOS applies sinusoidal strain at amplitudes large enough to probe nonlinear
rheological behavior. The stress response contains higher harmonics (3ω, 5ω, ...)
that encode information about yielding, shear-thinning, and microstructural changes.

**Physical setup**: Apply large-amplitude sinusoidal strain, measure nonlinear stress

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Field
     - Value
     - Notes
   * - **x**
     - time ``t`` (s)
     - Multiple oscillation cycles
   * - **y**
     - stress ``σ(t)`` (Pa)
     - Real, non-sinusoidal
   * - **domain**
     - ``'time'``
     -
   * - **test_mode**
     - ``'laos'``
     - Must be specified explicitly
   * - **gamma_0**
     - float, **REQUIRED** kwarg
     - Strain amplitude
   * - **omega**
     - float (rad/s), **REQUIRED** kwarg
     - Angular frequency
   * - **n_cycles**
     - int, optional (default 5–10)
     - Number of oscillation cycles

.. important::

   Both ``gamma_0`` and ``omega`` are **required** keyword arguments for LAOS mode.
   ``n_cycles`` controls how many oscillation periods are simulated.

**Example**:

.. code-block:: python

   from rheojax.models import TNTSingleMode

   gamma_0, omega, n_cycles = 0.5, 1.0, 5
   T = 2 * np.pi / omega
   t = np.linspace(0, n_cycles * T, n_cycles * 80)

   # Fit experimental LAOS stress data
   model = TNTSingleMode()
   model.fit(t, stress_data, test_mode='laos', gamma_0=gamma_0, omega=omega)

   # Or generate synthetic LAOS data via model
   result = model.simulate_laos(t=None, gamma_0=0.5, omega=1.0, n_cycles=3)
   # Returns: {'t': array, 'strain': array, 'stress': array, 'strain_rate': array}

**Compatible models**: TNT (5), Giesekus, IKH, FIKH, DMT, Fluidity-Saramito,
HVM, HVNM, STZ, VLB, SGR, ITT-MCT, HL, SPP — 26+ models

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

**Constructor**:

.. code-block:: python

   from rheojax.transforms import Mastercurve

   mc = Mastercurve(
       reference_temp=298.15,     # Reference temperature (K)
       method='wlf',              # 'wlf', 'arrhenius', 'manual'
       C1=17.44,                  # WLF parameter C1
       C2=51.6,                   # WLF parameter C2 (K)
       E_a=None,                  # Activation energy (J/mol) — for Arrhenius
       auto_shift=False,          # Power-law intersection method
   )

**Example**:

.. code-block:: python

   datasets = [
       RheoData(x=omega, y=G_star_273K, domain='frequency',
                metadata={'temperature': 273.15}),
       RheoData(x=omega, y=G_star_298K, domain='frequency',
                metadata={'temperature': 298.15}),
       RheoData(x=omega, y=G_star_323K, domain='frequency',
                metadata={'temperature': 323.15}),
   ]

   mc = Mastercurve(reference_temp=298.15, method='wlf')
   mastercurve, shift_factors = mc.transform(datasets)

**Additional methods**: ``get_shift_factor(T)``, ``optimize_wlf_parameters(datasets, C1, C2)``,
``compute_overlap_error(datasets)``

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
       RheoData(x=gamma_dot_1, y=eta_1,
                metadata={'reference_gamma_dot': 0.1}),
       RheoData(x=gamma_dot_2, y=eta_2,
                metadata={'reference_gamma_dot': 1.0}),
       RheoData(x=gamma_dot_3, y=eta_3,
                metadata={'reference_gamma_dot': 10.0}),
   ]

   srfs = SRFS(reference_gamma_dot=1.0)
   mastercurve, shifts = srfs.transform(
       datasets, x=1.5, tau0=1e-3, return_shifts=True,
   )

**Additional methods**: ``compute_shift_factor(gamma_dot, x, tau0)``,
``detect_shear_banding(gamma_dot, sigma)``

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

   owchirp = OWChirp(
       n_frequencies=100,
       frequency_range=(1e-2, 1e2),
       extract_harmonics=True,
       max_harmonic=7,
   )
   spectrum = owchirp.transform(data)

   # Extract harmonic content
   harmonics = owchirp.get_harmonics(data)
   print(f"Fundamental: {harmonics['fundamental']}")
   print(f"Third harmonic: {harmonics['third']}")

**Additional methods**: ``get_time_frequency_map(data) → (t, frequencies, coefficients)``

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
     - time ``t`` (s), uniformly spaced
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

   spp = SPPDecomposer(omega=1.0, gamma_0=1.0, n_harmonics=39)
   result = spp.transform(data)

   # Access results
   sigma_sy, sigma_dy = spp.get_yield_stresses()
   metrics = spp.get_nonlinearity_metrics()
   # → {'I3_I1_ratio', 'S_factor', 'T_factor'}

   # Convenience function
   from rheojax.transforms import spp_analyze
   results = spp_analyze(stress, time, omega=1.0, gamma_0=1.0, n_harmonics=5)

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

   fft = FFTAnalysis(window='hann', detrend=True, return_psd=False, normalize=True)
   freq_data = fft.transform(data)

   # Find characteristic time from peak frequency
   tau_char = fft.get_characteristic_time(freq_data)

**Additional methods**: ``find_peaks(freq_data, prominence=0.1, n_peaks=5)``

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

**Interpretation**: :math:`\Delta \to 0` = elastic solid, :math:`\Delta \to 1` = viscous fluid

**Example**:

.. code-block:: python

   from rheojax.transforms import MutationNumber

   mutation = MutationNumber(integration_method='trapz')
   delta = mutation.calculate(data)

**Additional methods**: ``get_relaxation_time(data)``, ``get_equilibrium_modulus(data)``

Smooth Derivative (Numerical Differentiation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Computes noise-robust derivatives using Savitzky-Golay filtering or splines.

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

   deriv = SmoothDerivative(
       method='savgol',       # 'savgol', 'finite_diff', 'spline', 'total_variation'
       window_length=11,
       polyorder=3,
       deriv=1,               # Derivative order
   )
   dJ_dt = deriv.transform(data)

**Additional methods**: ``estimate_noise_level(data)``

Model Fitting Data Interface
----------------------------

All 53 models follow the scikit-learn API pattern. The data format requirements
depend on the test mode.

.. code-block:: python

   # Point estimation (NLSQ)
   model.fit(
       X,                              # x-data array or RheoData object
       y=None,                         # y-data (ignored if X is RheoData)
       test_mode='oscillation',        # Required for multi-protocol models
       deformation_mode=None,          # 'tension' for DMTA
       poisson_ratio=None,             # Required with deformation_mode
       method='nlsq',                  # 'nlsq', 'scipy', 'auto'
       **kwargs                        # Protocol kwargs: gamma_dot, sigma_applied, etc.
   )

   # Prediction
   y_pred = model.predict(
       X,
       test_mode=None,                 # Defaults to fitted test_mode
       **kwargs
   )

   # Bayesian inference (warm-starts from fit() parameters)
   result = model.fit_bayesian(
       X, y=None,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,                   # Default: 4 chains
       seed=42,                        # Reproducibility
       test_mode='oscillation',
       **kwargs
   )

   # Credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples, credibility=0.95
   )

.. warning::

   Protocol kwargs like ``gamma_dot``, ``sigma_applied``, ``gamma_0``, and ``omega``
   are **critical for Bayesian inference**. Models cache them internally so that
   ``model_function()`` can access them during NUTS sampling when kwargs are not
   passed explicitly. Always provide these in both ``fit()`` and ``fit_bayesian()``.

Quick Reference Tables
----------------------

Test Mode Data Formats
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 14 18 18 8 8 16 18

   * - Test Mode
     - X Data
     - Y Data
     - X Units
     - Y Units
     - Required Kwargs
     - Auto-Detection
   * - **relaxation**
     - time (1D)
     - G(t) (1D)
     - s
     - Pa
     - —
     - y decreasing
   * - **creep**
     - time (1D)
     - J(t) (1D)
     - s
     - 1/Pa
     - ``sigma_applied`` (ODE)
     - y increasing
   * - **oscillation**
     - frequency (1D)
     - G*(ω) complex
     - rad/s
     - Pa
     - —
     - domain='frequency'
   * - **DMTA**
     - frequency (1D)
     - E*(ω) complex
     - rad/s
     - Pa
     - ``deformation_mode``, ``poisson_ratio``
     - E'/E'' columns
   * - **flow_curve**
     - shear rate (1D)
     - σ (1D)
     - 1/s
     - Pa
     - —
     - x_units='1/s'
   * - **startup**
     - [t, γ] stacked (2,N)
     - σ(t) (1D)
     - s, —
     - Pa
     - ``gamma_dot`` (**required**)
     - must specify
   * - **laos**
     - time (1D)
     - σ(t) (1D)
     - s
     - Pa
     - ``gamma_0``, ``omega`` (**required**)
     - must specify

Transform Data Formats
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 18 12 12 16 22 20

   * - Transform
     - Input Domain
     - X
     - Y
     - Required Metadata
     - Output
   * - **Mastercurve**
     - frequency
     - ω
     - G*
     - ``temperature`` (K)
     - Shifted mastercurve
   * - **SRFS**
     - any
     - γ̇
     - η or σ
     - ``reference_gamma_dot``
     - Shifted mastercurve
   * - **OWChirp**
     - time
     - t
     - σ(t)
     - —
     - Freq spectrum
   * - **SPP Decomposer**
     - time
     - t
     - σ(t)
     - ``omega``, ``gamma_0``
     - Stress + SPP metrics
   * - **FFT Analysis**
     - time
     - t
     - signal
     - —
     - Freq spectrum
   * - **Mutation Number**
     - time
     - t
     - G(t)
     - —
     - Scalar Δ ∈ [0,1]
   * - **Smooth Derivative**
     - any
     - x
     - y
     - —
     - dy/dx

Model Protocol Support
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Protocols Supported
     - Count
     - Examples
   * - All 6 (relax, creep, osc, flow, startup, laos)
     - 26
     - TNT, HVM, HVNM, IKH, FIKH, DMT, Giesekus, SGR, Fluidity-Saramito, ...
   * - 5 (no LAOS)
     - 2
     - Lattice EPM, Tensorial EPM
   * - 4 (relax, creep, osc, flow)
     - 4
     - Maxwell, Zener, Fractional Jeffreys, FML
   * - 3
     - 12
     - SpringPot, Giesekus Multi, fractional models, nonlocal variants
   * - 2
     - 1
     - SPP (FLOW_CURVE + LAOS only)
   * - 1 (flow only)
     - 6
     - PowerLaw, Carreau, Cross, HB, CarreauYasuda, Casson
   * - **DMTA-compatible**
     - **41**
     - All models with OSCILLATION protocol

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
   np_data = data.to_numpy()

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

For oscillation data, construct complex arrays:

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

.. note::

   For DMTA/DMA data, the CSV reader auto-detects E'/E'' columns and stores
   ``deformation_mode='tension'`` in ``metadata``.
   See :doc:`/models/dmta/dmta_workflows` Workflow 4 for details.

Further Reading
---------------

- :doc:`data_io` — Loading data from instrument files
- :doc:`trios_format` — TA Instruments TRIOS format (detailed)
- :doc:`pipeline_api` — Pipeline API for fluent workflows
- :doc:`/api/core` — Full API reference for RheoData
- :doc:`/api/transforms` — Full API reference for transforms
- :doc:`/models/dmta/index` — DMTA guide
