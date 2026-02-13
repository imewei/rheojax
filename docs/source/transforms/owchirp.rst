.. _transform-owchirp:

OWChirp
=======

Overview
--------

:class:`rheojax.transforms.OWChirp` designs, executes, and analyzes orthogonal windowed
chirp experiments for broadband LAOS. A single chirp sweeps logarithmically from
``chirp_span[0]`` to ``chirp_span[1]`` Hz, enabling simultaneous extraction of linear and
nonlinear moduli (:math:`G_1'`, :math:`G_1''`, :math:`G_3'`, etc.).

Equations
---------

OWChirp supports linear and logarithmic sweeps with instantaneous frequency

.. math::

   f(t) = f_{\mathrm{start}} \exp\left( \frac{\ln(f_{\mathrm{end}} / f_{\mathrm{start}})}{T} t \right)

and phase :math:`\phi(t) = 2\pi \int_0^t f(\tau)\,d\tau`. Orthogonal window segments (Planck
or Tukey tapers) are applied to sub-bands so harmonics remain separable even when multiple
chirps are concatenated.

Complete OWCh Waveform Equation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full OWCh waveform applies a Tukey (cosine-tapered) window to an exponential chirp,
incorporating an optional waiting time :math:`t_w` at the signal start. The complete
piecewise definition is (Perego et al. 2025, Eq. 5):

.. math::

   x(t) = x_0 \begin{cases}
   \cos^2\left[\frac{\pi}{r}\left(\frac{t - t_w}{T_{\mathrm{owc}}} - \frac{r}{2}\right)\right]
   \sin\left\{\frac{\omega_1 T_{\mathrm{owc}}}{\ln(\omega_2/\omega_1)}
   \left[\exp\left(\frac{\ln(\omega_2/\omega_1)}{T_{\mathrm{owc}}}(t - t_w)\right) - 1\right]\right\},
   & \frac{t - t_w}{T_{\mathrm{owc}}} \le \frac{r}{2} \\[1em]
   \sin\left\{\frac{\omega_1 T_{\mathrm{owc}}}{\ln(\omega_2/\omega_1)}
   \left[\exp\left(\frac{\ln(\omega_2/\omega_1)}{T_{\mathrm{owc}}}(t - t_w)\right) - 1\right]\right\},
   & \frac{r}{2} \le \frac{t - t_w}{T_{\mathrm{owc}}} \le 1 - \frac{r}{2} \\[1em]
   \cos^2\left[\frac{\pi}{r}\left(\frac{t - t_w}{T_{\mathrm{owc}}} - 1 + \frac{r}{2}\right)\right]
   \sin\left\{\frac{\omega_1 T_{\mathrm{owc}}}{\ln(\omega_2/\omega_1)}
   \left[\exp\left(\frac{\ln(\omega_2/\omega_1)}{T_{\mathrm{owc}}}(t - t_w)\right) - 1\right]\right\},
   & \frac{t - t_w}{T_{\mathrm{owc}}} \ge 1 - \frac{r}{2}
   \end{cases}

where:

- :math:`x_0` is the stress or strain amplitude
- :math:`\omega_1, \omega_2` are the lower and upper angular frequency bounds (rad/s)
- :math:`T_{\mathrm{owc}}` is the total chirp duration
- :math:`t_w` is the waiting time before the chirp begins
- :math:`r` is the Tukey window tapering coefficient (cosine fraction, typically 0.05–0.15)

**Tapering coefficient selection:**

- **r = 0.05:** Minimal tapering, maximizes signal energy, suitable for high-SNR systems
- **r = 0.10 (recommended):** Balanced trade-off between spectral leakage and signal duration
- **r = 0.15:** Aggressive tapering, superior sidelobe suppression for precision measurements

Duration Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The theoretical minimum chirp duration is set by the lowest frequency:

.. math::

   T_{\mathrm{owc}} \ge \frac{2\pi}{\omega_1}

For improved throughput, Perego et al. (2025) adopt a practical **2/3 scaling factor**:

.. math::

   T_{\mathrm{owc}} \ge \frac{2}{3} \cdot \frac{2\pi}{\omega_1}

This reduces acquisition time by approximately 33% while only modestly shifting the lowest
reliably probed frequency (e.g., from 0.3 to ~0.45 rad/s). The trade-off is acceptable for
most industrial applications where throughput is prioritized.

**Example:** For :math:`\omega_1 = 0.3` rad/s:

- Theoretical minimum: :math:`T_{\mathrm{owc}} \ge 21` s
- Practical (2/3 scaling): :math:`T_{\mathrm{owc}} \ge 14` s

Waveform Design Details
-----------------------

Planck Taper Window
~~~~~~~~~~~~~~~~~~~

The Planck taper provides smooth onset and offset transitions that minimize spectral leakage
while maintaining signal energy. For a signal of duration :math:`T` with taper fraction
:math:`\varepsilon`, the window function is:

.. math::

   w(t) = \begin{cases}
   \left[1 + \exp\left(\frac{\varepsilon T}{t} + \frac{\varepsilon T}{t - \varepsilon T}\right)\right]^{-1}
   & 0 < t < \varepsilon T \\
   1 & \varepsilon T \le t \le (1-\varepsilon)T \\
   \left[1 + \exp\left(\frac{\varepsilon T}{T-t} + \frac{\varepsilon T}{T-t - \varepsilon T}\right)\right]^{-1}
   & (1-\varepsilon)T < t < T
   \end{cases}

**Taper parameter selection:**

- :math:`\varepsilon` **= 0.10:** Minimal tapering, maximizes flat-top duration, risk of spectral leakage
- :math:`\varepsilon` **= 0.15 (default):** Balanced trade-off, recommended for most applications
- :math:`\varepsilon` **= 0.20:** Aggressive tapering, reduced leakage but shorter effective duration

**Comparison with Tukey taper:**

The Tukey (cosine-tapered) window uses a raised cosine transition, which is computationally
simpler but produces slightly more spectral leakage than the Planck taper. The Planck taper's
exponential rolloff provides superior sidelobe suppression (−60 dB vs −40 dB for Tukey).

Mutation Number Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Before analyzing chirp data, verify that the material remains quasi-steady during the sweep.
The **mutation number** :math:`\delta` quantifies structural evolution during the measurement:

.. math::

   \delta = \frac{1}{\pi} \int_0^T \left| \frac{d \ln G'(t)}{dt} \right| dt

**Quasi-steady criterion:** :math:`\delta < 0.2` indicates <20% structural change during the
chirp, validating the assumption of time-invariant material properties.

For rapidly mutating materials (gels, curing systems), use shorter chirps or the
Time-Curing Superposition approach described in :ref:`owchirp-advanced-applications`.

See :doc:`mutation_number` for detailed mutation number theory and calculation.

Time-Frequency Trade-Offs
-------------------------

The time-bandwidth product :math:`TB = T (f_{\mathrm{end}} - f_{\mathrm{start}})` governs spectral
resolution. Values above ~50 yield <2% amplitude error, whereas shorter chirps (
low TB) cover the spectrum faster but broaden frequency bins. OWChirp reports ``tb_product``
and warns when resolution is compromised.

**Recommended TB products:**

.. list-table:: Time-bandwidth product guidelines
   :header-rows: 1
   :widths: 25 25 50

   * - TB Product
     - Regime
     - Guidance
   * - :math:`TB < 30`
     - Low resolution
     - Significant spectral broadening (>10% error). Use only for rapid screening.
   * - :math:`30 \le TB < 66`
     - Moderate resolution
     - Acceptable for exploratory measurements (2-10% error).
   * - :math:`TB \approx 66`
     - Empirically optimal
     - Best trade-off between speed and accuracy for most materials.
   * - :math:`TB \ge 100`
     - High resolution
     - Near-discrete frequency accuracy (<2% error). Use for publication-quality data.

**Empirical guideline:** A TB product of approximately 66 has been found empirically optimal
for balancing measurement speed with spectral resolution in most rheological applications.

Harmonic Extraction
-------------------

Given recorded stress :math:`\sigma(t)` and measured strain :math:`\gamma(t)`, the transform
projects onto harmonic basis functions tied to :math:`\phi(t)`:

.. math::

   G_n'(\phi) = \frac{2}{T} \int_0^T \sigma(t) \cos(n\phi(t))\,dt,
   \qquad
   G_n''(\phi) = \frac{2}{T} \int_0^T \sigma(t) \sin(n\phi(t))\,dt.

The resulting moduli are reported versus instantaneous frequency. Nonlinear intensity ratios
(:math:`I_{3/1}`) are computed automatically.

Deconvolution and Windowing
---------------------------

Because chirps excite a continuum of frequencies, OWChirp performs Wiener deconvolution in
the joint time-frequency domain:

.. math::

   H^{-1}(\omega) = \frac{H^*(\omega)}{|H(\omega)|^2 + \lambda},

where :math:`H` is the chirp kernel and :math:`\lambda` is a regularization parameter
estimated from the noise floor. Planck or Tukey tapers applied at the start/end of the
chirp limit spectral leakage.

Parameters
----------

.. list-table:: OWChirp parameters
   :header-rows: 1
   :widths: 25 18 39 18

   * - Parameter
     - Type
     - Description
     - Default
   * - ``chirp_span``
     - tuple(float, float)
     - Frequency range (Hz) for the sweep.
     - ``(0.1, 30.0)``
   * - ``amplitude``
     - float
     - Target strain or stress amplitude.
     - ``0.05``
   * - ``duration``
     - float
     - Chirp length (s); influences time-bandwidth product.
     - ``30.0``
   * - ``taper``
     - str
     - Edge window (``"planck(0.15)"``, ``"tukey(0.2)"`` ...).
     - ``"planck(0.15)"``
   * - ``n_harmonics``
     - int
     - Number of harmonics to extract (odd orders).
     - ``5``

Industrial Implementation
-------------------------

Instrument Integration
~~~~~~~~~~~~~~~~~~~~~~

OWCh experiments can be implemented on commercial rheometers via the **Arbitrary Waveform**
method. On TA Instruments ARES-G2 rheometers using TRIOS software, users supply up to four
equations describing the OWCh waveform. Key constraints include:

- **Memory limit:** :math:`N_{\mathrm{max}} = 2^{15}` points on ARES-G2
- **Sampling rate:** Must satisfy both Nyquist and memory constraints (see Eq. 8)
- **Amplitude control:** Manual regulation required; trial-and-error or interpolation
  for multi-temperature experiments

**Recommended workflow:**

1. Define frequency bounds (:math:`\omega_1`, :math:`\omega_2`) and duration (:math:`T_{\mathrm{owc}}`)
2. Calculate valid sampling range using the normalized Nyquist criterion
3. Verify :math:`TB \ge 66` for adequate spectral resolution
4. For temperature sweeps, conduct preliminary OWCh trials at thermal extremes to establish
   amplitude bounds; interpolate for intermediate temperatures

Data Processing with hermes-rheo
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``hermes-rheo`` Python package (Perego et al. 2024) provides automated OWCh analysis
integrated with the ``piblin`` data pipeline framework. Both packages are MIT-licensed
and available on PyPI.

**Installation:**

.. code-block:: bash

   pip install piblin hermes-rheo

**Key features:**

- **OWChGeneration transform:** Automatically generates all waveform parameters for
  TRIOS Arbitrary Waveform method
- **Automated bias correction:** Three filtering methods applied systematically:

  1. Subtract average from waiting-time interval only
  2. Subtract average from OWCh segment only
  3. Subtract average from entire signal duration

  The algorithm selects the method producing signals symmetric around zero with minimal
  endpoint deviations.

- **Cloud integration:** Modular design enables incorporation into automated, cloud-based
  data analysis pipelines

**Example usage:**

.. code-block:: python

   from hermes_rheo.transforms import OWChGeneration, OWChAnalysis

   # Generate waveform parameters for TRIOS
   gen = OWChGeneration(
       omega1=0.3,  # rad/s
       omega2=60.0,  # rad/s
       towc=14.0,  # seconds
       tw=1.0,  # waiting time
       r=0.1,  # tapering coefficient
   )
   waveform_params = gen.generate()

   # Analyze recorded data
   analysis = OWChAnalysis(fs=1000.0)  # sampling frequency
   result = analysis.transform(strain_data, stress_data)

For detailed tutorials, see the `hermes-rheo documentation <https://hermes-rheo.readthedocs.io/>`_
and `tutorial notebooks <https://github.com/3mcloud/hermes-rheo/blob/main/tutorial_notebooks>`_.

Sampling and Constraints
------------------------

Nyquist Number Analysis
~~~~~~~~~~~~~~~~~~~~~~~

The **Nyquist number** :math:`N_y` quantifies sampling adequacy relative to the highest
frequency component:

.. math::

   N_y = \frac{f_{\mathrm{max}}}{f_{\mathrm{Nyquist}}} = \frac{2 f_{\mathrm{max}}}{f_s}

where :math:`f_s` is the sampling rate. For alias-free reconstruction:

.. math::

   N_y < 1 \quad \Rightarrow \quad f_s > 2 f_{\mathrm{max}}

Normalized Nyquist Number
^^^^^^^^^^^^^^^^^^^^^^^^^

For OWCh waveform design, the **normalized Nyquist number** captures the interplay between
chirp duration, waiting time, upper frequency, and available sampling points
(Perego et al. 2025, Eq. 6):

.. math::

   N_y = \frac{(T_{\mathrm{owc}} + t_w) \omega_2}{2\pi N}

where :math:`N` is the total number of discrete sampling points.

The Nyquist–Shannon sampling theorem requires :math:`N_y \le 0.5`. However, in practice,
OWCh signals sampled at this rate may exhibit aliasing artifacts, particularly when
:math:`\omega_2 \ge 60` rad/s. Hudson-Kershaw et al. (2024) recommend a more stringent
constraint:

.. math::

   N_y < 0.1 \quad \text{(practical guideline)}

This translates to a **valid sampling range** (Perego et al. 2025, Eq. 8):

.. math::

   \frac{10\omega_2}{2\pi} \le f_s \le \frac{N_{\mathrm{max}}}{T_{\mathrm{owc}} + t_w}

where :math:`N_{\mathrm{max}}` is the maximum number of recordable points allowed by the
rheometer's memory.

**Worked example:** For :math:`\omega_1 = 0.3` rad/s, :math:`\omega_2 = 125` rad/s,
:math:`T_{\mathrm{owc}} = 14` s, :math:`t_w = 1` s, and :math:`N_{\mathrm{max}} = 2^{15}`
(TA ARES-G2):

.. math::

   200\ \text{pts/s} \le f_s \le 2184\ \text{pts/s}

**Modified sampling condition for harmonics:**

When extracting the :math:`n`-th harmonic, the effective maximum frequency is
:math:`n \times f_{\mathrm{end}}`. The sampling rate must satisfy:

.. math::

   f_s > 2 n_{\mathrm{max}} f_{\mathrm{end}}

For example, extracting up to the 5th harmonic from a chirp ending at 30 Hz requires
:math:`f_s > 300` Hz (typically use 500 Hz with safety margin).

**Practical sampling guidelines:**

.. list-table:: Sampling rate recommendations
   :header-rows: 1
   :widths: 30 30 40

   * - Chirp Range
     - Max Harmonic
     - Minimum :math:`f_s`
   * - 0.1–10 Hz
     - 5th
     - 100 Hz (use 200 Hz)
   * - 0.1–30 Hz
     - 5th
     - 300 Hz (use 500 Hz)
   * - 0.1–100 Hz
     - 7th
     - 1400 Hz (use 2000 Hz)

Data Density Comparison with DFS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OWCh provides significantly higher spectral resolution than discrete frequency sweeps (DFS).
The number of frequency-domain data points from an OWCh signal is (Perego et al. 2025, Eq. 11):

.. math::

   N_f = \frac{2^n (\omega_2 - \omega_1)}{2\pi f_s}

where :math:`2^n` is the number of time-domain points (rounded to the nearest power of two
for efficient FFT processing).

For comparison, the total duration of a DFS is (Perego et al. 2025, Eq. 3):

.. math::

   T_{\mathrm{DFS}} \ge \sum_{i=1}^{m} \frac{2\pi}{\omega_i}

where :math:`m` is the number of tested frequencies.

.. list-table:: Data density comparison: OWCh vs DFS
   :header-rows: 1
   :widths: 25 25 25 25

   * - Method
     - Frequency Range
     - Duration
     - Data Points
   * - OWCh
     - 0.3–30 rad/s
     - 15 s
     - ~78 points
   * - DFS (7 pts/decade)
     - 0.3–30 rad/s
     - ~180 s
     - ~14 points

**Key result:** OWCh delivers an **order-of-magnitude increase** in spectral resolution
while simultaneously reducing overall testing duration by 10× or more.

Wait Time Between Chirps
~~~~~~~~~~~~~~~~~~~~~~~~

When concatenating multiple chirps or performing repeated measurements, allow sufficient
**wait time** :math:`t_w` for the material to recover from previous deformation:

.. math::

   t_w \ge 5 \tau_{\mathrm{relax}}

where :math:`\tau_{\mathrm{relax}}` is the material's characteristic relaxation time. For
unknown materials, estimate :math:`\tau_{\mathrm{relax}}` from the inverse of the crossover
frequency (:math:`\omega` where :math:`G' = G''`).

**Wait time guidelines:**

- **Purely elastic materials:** :math:`t_w \approx 1` s (fast recovery)
- **Viscoelastic fluids:** :math:`t_w \approx 5/\omega_{\mathrm{crossover}}`
- **Yield stress materials:** :math:`t_w \approx 30` s or longer (thixotropic recovery)

Input / Output Specifications
-----------------------------

- **Design input**: sampling rate ``fs`` (Hz), control mode (strain or stress), optional
  actuator limits. ``OWChirp.design`` returns waveform samples, instantaneous frequency, and
  metadata for instrument playback.
- **Analysis input**: recorded strain ``gamma(t)`` (dimensionless) and stress ``sigma(t)`` (Pa)
  as :class:`RheoData` objects or arrays with timestamps.
- **Outputs**: dict with
  - ``waveform`` (for design),
  - ``moduli`` mapping harmonic order to arrays of :math:`G_n'`, :math:`G_n''`,
  - ``frequency`` grid (Hz) per harmonic,
  - ``diagnostics`` (time-bandwidth product, crest factor, leakage, Wiener :math:`\lambda`).

Usage
-----

.. code-block:: python

   from rheojax.transforms import OWChirp

   ow = OWChirp(chirp_span=(0.2, 40.0), amplitude=0.1, duration=25.0, taper="tukey(0.2)")
   plan = ow.design(mode="strain", fs=500.0)
   # Send plan["waveform"] to the rheometer, then record response traces...
   result = ow.transform(response_gamma, response_sigma, fs=500.0)

   G1 = result["moduli"][1]
   G3 = result["moduli"][3]
   I31 = G3["G_double_prime"] / G1["G_double_prime"]

.. _owchirp-advanced-applications:

Advanced Applications
---------------------

Time-Curing Superposition (tCS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **mutating materials** (curing polymers, aging gels, crystallizing melts), standard
chirp analysis assumes quasi-steady behavior. When the mutation number :math:`Mu > 0.15`,
this assumption breaks down.

**Time-Curing Superposition (tCS)** extends chirp rheometry to evolving materials by:

1. **Segmenting** the chirp into short sub-windows where :math:`Mu_{\mathrm{local}} < 0.15`
2. **Shifting** each segment's moduli using a time-dependent shift factor :math:`a_t(t_{\mathrm{cure}})`
3. **Constructing** a master curve that tracks the evolving material state

**When to use tCS:**

- Monitoring gelation kinetics (epoxy, alginate, fibrin)
- Tracking crystallization (polymers, fats)
- Characterizing aging in colloidal glasses
- UV-curable acrylate crosslinking

Mutation Number from Consecutive Chirps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For consecutive OWCh measurements on an evolving material, the mutation number can be
computed by tracking changes in a viscoelastic metric :math:`g` (typically :math:`|G^*|`)
across chirps (Perego et al. 2025, Eq. 15):

.. math::

   Mu(t_i, \omega_j) \approx \frac{T_{\mathrm{owc}} \ln\left(\frac{g(t_i, \omega_j)}{g(t_{i-1}, \omega_{j-1})}\right)}{t_i - t_{i-1}}

where :math:`i` indexes the chirp number and :math:`j` indexes the frequency. The
frequency-averaged mutation number :math:`\overline{Mu}` provides a single metric for
each chirp.

**Critical threshold:** :math:`Mu_{\mathrm{crit}} = 0.15` (Rathinaraj et al. 2022)

When :math:`\overline{Mu} > Mu_{\mathrm{crit}}`, measurement artifacts appear, such as
artificial upticks in the complex viscosity master curve.

Frequency Selection for Rapid Mutation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To keep the mutation number below :math:`Mu_{\mathrm{crit}}`, the minimum angular frequency
:math:`\omega_1` must be increased to shorten the OWCh duration. An approximate guideline
is (Perego et al. 2025, Eq. 16):

.. math::

   \omega_1 > \frac{4\pi}{3 \, Mu_{\mathrm{crit}} \, \tau_{\mathrm{max}}}

where :math:`\tau_{\mathrm{max}}` is the characteristic time at :math:`\max(\overline{Mu})`.

**Trade-off:** Increasing :math:`\omega_1` reduces the time-bandwidth product :math:`TB`,
potentially below the optimal :math:`TB \ge 66`. Priority should be given to satisfying
:math:`Mu < 0.15` over maintaining high :math:`TB`.

**Example:** For a UV-curable acrylate with rapid early-stage kinetics:

- Original: :math:`\omega_1 = 0.6` rad/s, :math:`T_{\mathrm{owc}} = 7` s → :math:`TB = 66`
- Adjusted: :math:`\omega_1 = 1.5` rad/s, :math:`T_{\mathrm{owc}} = 2.8` s → :math:`TB = 26`

The adjusted parameters eliminate the artifact while sacrificing some spectral resolution.

Complex Viscosity Master Curves
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For tCS analysis, the reduced complex viscosity provides diagnostic insight:

.. math::

   |\eta^*_r| = \frac{|G^*_r|}{\omega_r} = \frac{\sqrt{(G'_r)^2 + (G''_r)^2}}{\omega_r}

An **artificial uptick** in :math:`|\eta^*_r|` at intermediate reduced frequencies signals
that the material transformation rate exceeded the measurement timescale during that
phase of curing. This artifact may be masked in :math:`|G^*|` plots due to the linear
frequency dependence, but becomes apparent when scaled by :math:`1/\omega_r`.

**Diagnostic approach:**

1. Compute :math:`\overline{Mu}` for each chirp
2. Identify time intervals where :math:`\overline{Mu} > Mu_{\mathrm{crit}}`
3. If artifacts present, repeat experiment with higher :math:`\omega_1` to shorten
   :math:`T_{\mathrm{owc}}`

**Practical workflow:**

.. code-block:: python

   from rheojax.transforms import OWChirp, MutationNumber

   # Check mutation number during cure
   mn = MutationNumber(tau_c=10.0)
   delta = mn.calculate(rheo_data_during_cure)

   if delta > 0.15:
       # Use short chirps and apply tCS
       ow_short = OWChirp(duration=5.0)  # Short segments
       # Process each segment independently...
   else:
       # Standard analysis
       ow = OWChirp(duration=30.0)

   # Monitor mutation number across consecutive chirps
   mu_values = []
   for i in range(1, len(chirp_results)):
       g_current = chirp_results[i]['G_star_magnitude']
       g_previous = chirp_results[i-1]['G_star_magnitude']
       dt = chirp_results[i]['time'] - chirp_results[i-1]['time']
       mu = (T_owc * np.log(g_current / g_previous)) / dt
       mu_values.append(np.mean(mu))  # Frequency-averaged

   # Check for violations
   if any(mu > 0.15 for mu in mu_values):
       print("Warning: Mu > 0.15 detected; consider increasing omega1")

Accelerated Time-Temperature Superposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Chirp rheometry can **accelerate mastercurve construction** by up to 40% compared to
traditional discrete-frequency sweeps. Benefits include:

**Efficiency gains:**

- **Single experiment** covers 3-4 decades of frequency vs multiple isothermal sweeps
- **40% reduction** in total measurement time for equivalent frequency coverage
- **Continuous data** provides higher frequency density than discrete points

**Quantified benefits (Perego et al. 2025):**

.. list-table:: OWCh vs DFS for tTS master curves
   :header-rows: 1
   :widths: 35 30 35

   * - Metric
     - DFS
     - OWCh
   * - Data points in master curve
     - ~290
     - ~1300
   * - Total acquisition time
     - 90 min
     - 53 min
   * - Data density improvement
     - baseline
     - **4.5× higher**
   * - Time reduction
     - baseline
     - **40% faster**

**Data density advantages:**

Traditional frequency sweeps sample 5-10 points per decade. Chirp measurements provide
effectively continuous coverage, improving:

- Fit quality for multi-mode Maxwell/Prony series
- Detection of subtle relaxation features (shoulder peaks, plateaus)
- Identification of time-temperature superposition failures

Williams-Landel-Ferry (WLF) Equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The horizontal shift factors :math:`a_T` are typically described by the WLF equation
(Ferry 1980):

.. math::

   \log_{10}(a_T) = -\frac{C_1 (T - T_r)}{C_2 + (T - T_r)}

where :math:`T_r` is the reference temperature and :math:`C_1`, :math:`C_2` are
material-specific empirical constants.

**Typical WLF parameters for pressure-sensitive adhesives:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Method
     - :math:`C_1`
     - :math:`C_2` (°C)
   * - DFS with GPR
     - 9.19
     - 128
   * - OWCh with GPR
     - 8.63
     - 124

The close agreement between OWCh and DFS shift factors validates OWCh as an effective
alternative for tTS protocols.

Automated Superposition with Gaussian Process Regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hermes-rheo`` package implements automated master curve construction using
Gaussian process regression (GPR) with maximum *a posteriori* estimation, following
the methodology of Lennon et al. (2023). This data-driven approach:

- Automatically determines optimal shift factors
- Provides uncertainty bounds on the master curve
- Is robust to elevated noise levels (e.g., near compliance limits)

**Integration with mastercurve transform:**

.. code-block:: python

   from rheojax.transforms import OWChirp, Mastercurve

   # Collect chirp data at multiple temperatures
   chirp_results = {}
   for T in [20, 40, 60, 80, 100]:  # °C
       ow = OWChirp(chirp_span=(0.01, 100), duration=30.0)
       chirp_results[T] = ow.transform(data_at_T, fs=500.0)

   # Construct mastercurve with automatic shift factors
   mc = Mastercurve(reference_temp=60, auto_shift=True)
   master, shifts = mc.transform(chirp_results)

Temperature Calibration Best Practices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For multi-temperature OWCh experiments:

1. **Amplitude selection:** Conduct preliminary OWCh trials at thermal extremes
   (e.g., :math:`-40\,^\circ\text{C}` and :math:`150\,^\circ\text{C}`) to establish amplitude bounds

2. **Interpolation:** For intermediate temperatures, interpolate amplitude values;
   refine near major transitions (e.g., :math:`T_g`)

3. **Equilibration:** Allow 3 min equilibration at each temperature before measurement

4. **Transducer range:** Adjust as needed for complex rheological profiles

Instrument Compliance Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At low temperatures and high frequencies, material stiffness may approach instrument
compliance limits. Perego et al. (2025) recommend working in **compliance space**:

.. math::

   J' = \frac{G'}{G'^2 + G''^2}, \quad J'' = \frac{G''}{G'^2 + G''^2}

The instrument compliance threshold is (Perego et al. 2025, Eq. 13):

.. math::

   J'_{\mathrm{inst}}(\omega) = \frac{1}{K_\theta(\omega)} \cdot \frac{C_\gamma}{C_\sigma}

where:

- :math:`K_\theta(\omega)` is the frequency-dependent torsional stiffness
- :math:`C_\gamma`, :math:`C_\sigma` are strain and stress conversion coefficients

**Warning:** When :math:`J' \lesssim J'_{\mathrm{inst}}`, compliance-related artifacts
appear as increased noise at high frequencies. Real-time compliance corrections available
in standard oscillatory tests are **not available** for arbitrary waveform protocols,
making careful parameter selection essential.

Validation Summary
------------------

Perego et al. (2025) validated OWCh against discrete frequency sweeps (DFS) and multi-wave
superposition over a temperature range of :math:`-40\,^\circ\text{C}` to :math:`150\,^\circ\text{C}`:

- **Excellent agreement** among OWCh, DFS, and multi-wave methods at all temperatures
- **WLF parameters** from OWCh within 6% of DFS values
- **Master curve overlay** shows negligible differences between methods
- **Low-temperature deviations** traceable to instrument compliance limits, not method error

These results confirm OWCh as a reliable alternative to traditional frequency sweeps for
thermorheologically simple materials.

Troubleshooting
---------------

**Waveform Design Issues:**

- **Spectral holes** — Increase ``duration`` or reduce taper aggressiveness so each octave
  receives sufficient dwell time.
- **TB product warning** — Increase ``duration`` or narrow ``chirp_span`` to achieve :math:`TB \ge 66`.
- **Aliasing** — Verify ``fs`` exceeds :math:`10\omega_2/2\pi` for :math:`N_y < 0.1`.

**Signal Quality Issues:**

- **Weak higher harmonics** — Raise ``amplitude`` (within instrument limits) or average
  repeated chirps to boost SNR before deconvolution.
- **Peak overlap** — Ensure ``n_harmonics`` is not larger than the resolvable bandwidth; use
  orthogonal window segments when concatenating chirps.
- **Bias/drift artifacts** — Use waiting time :math:`t_w > 0` and apply automated bias
  correction (subtracting signal mean from waiting interval, OWCh segment, or full duration).

**Compliance-Related Issues:**

- **High-frequency noise at low temperatures** — Material stiffness approaching instrument
  compliance limits. Check if :math:`J' \lesssim J'_{\mathrm{inst}}` and consider narrowing
  frequency range or using stiffer geometry.
- **Transducer saturation** — Compare programmed vs measured strain amplitude; significant
  deviation indicates compliance or acceleration limits reached.
- **No compliance correction available** — Real-time compliance corrections are not
  implemented for arbitrary waveform protocols. Must estimate and correct offline.

**Mutating Material Issues:**

- **High mutation number** — Material evolving during chirp (:math:`Mu > 0.15`). Increase
  :math:`\omega_1` to shorten :math:`T_{\mathrm{owc}}`, accepting lower :math:`TB`.
- **Uptick in** :math:`|\eta^*_r|` — Rapid mutation phase exceeded measurement timescale.
  Rerun with shorter chirps and verify :math:`\overline{Mu} < Mu_{\mathrm{crit}}` throughout.
- **Master curve doesn't collapse** — Check for thermorheological complexity or
  mutation artifacts; try vertical shift factors :math:`b_T \ne 1` if warranted.

References
----------

**Primary OWCh References:**

- Perego, A., Vadillo, D.C., Mills, M.J.L., Das, M., & McKinley, G.H. "Evaluation of
  optimally windowed chirp signals in industrial rheological measurements: method
  development and data processing." *Rheol. Acta* 64, 391–406 (2025).
  DOI: `10.1007/s00397-025-01511-0 <https://doi.org/10.1007/s00397-025-01511-0>`_
  :download:`PDF <../../reference/perego_2025_owchirp.pdf>`

- Geri, M., Keshavarz, B., Divoux, T., Clasen, C., Curtis, D.J., & McKinley, G.H.
  "Time-resolved mechanical spectroscopy of soft materials via optimally windowed chirps."
  *Phys. Rev. X* 8, 041042 (2018). DOI: `10.1103/PhysRevX.8.041042 <https://doi.org/10.1103/PhysRevX.8.041042>`_

**Extensions and Variants:**

- Hudson-Kershaw, R.E., Das, M., McKinley, G.H., & Curtis, D.J. ":math:`\sigma`-OWCh: optimally windowed
  chirp rheometry using combined motor transducer/single head rheometers."
  *J. Non-Newtonian Fluid Mech.* 333 (2024). DOI: `10.1016/j.jnnfm.2024.105307 <https://doi.org/10.1016/j.jnnfm.2024.105307>`_

- Athanasiou, T., Geri, M., Roose, P., McKinley, G.H., & Petekidis, G. "High-frequency
  optimally windowed chirp rheometry for rapidly evolving viscoelastic materials:
  application to a crosslinking thermoset." *J. Rheol.* 68(3), 445–462 (2024).
  DOI: `10.1122/8.0000793 <https://doi.org/10.1122/8.0000793>`_

- Rathinaraj, J.D.J., Hendricks, J., McKinley, G.H., & Clasen, C. "Orthochirp: a fast
  spectro-mechanical probe for monitoring transient microstructural evolution of complex
  fluids during shear." *J. Non-Newtonian Fluid Mech.* 301, 104744 (2022).
  DOI: `10.1016/j.jnnfm.2022.104744 <https://doi.org/10.1016/j.jnnfm.2022.104744>`_

**Instrument Compliance and Data Processing:**

- Hossain, M.T., Macosko, C.W., McKinley, G.H., & Ewoldt, R.H. "Instrument stiffness
  artifacts: avoiding bad data with operational limit lines of G_max and E_max."
  *Rheol. Acta* 64, 67–79 (2025). DOI: `10.1007/s00397-024-01481-9 <https://doi.org/10.1007/s00397-024-01481-9>`_

- Lennon, K.R., McKinley, G.H., & Swan, J.W. "A data-driven method for automated data
  superposition with applications in soft matter science." *Data-Centric Engineering* 4,
  e13 (2023). DOI: `10.1017/dce.2023.3 <https://doi.org/10.1017/dce.2023.3>`_

**Foundational Works:**

- Ghiringhelli, E., Roux, D., Bleses, D., Galliard, H., & Caton, F. "Optimal Fourier
  rheometry: application to the gelation of an alginate." *Rheol. Acta* 51(5), 413–420 (2012).

- Mours, M. & Winter, H.H. "Time-resolved rheometry." *Rheol. Acta* 33, 385–397 (1994).

- Ferry, J.D. *Viscoelastic Properties of Polymers*, 3rd ed. Wiley, New York (1980).

**Software:**

- Mills, M.J.L., et al. ``piblin``: Pipeline data framework. MIT License.
  `GitHub <https://github.com/3mcloud/piblin>`_ |
  `PyPI <https://pypi.org/project/piblin/>`_

- Perego, A., Mills, M.J.L., & Vadillo, D.C. ``hermes-rheo``: High-throughput rheological
  data transformations. MIT License.
  `GitHub <https://github.com/3mcloud/hermes-rheo>`_ |
  `Docs <https://hermes-rheo.readthedocs.io/>`_

See also
--------

- :doc:`fft` — OWChirp relies on windowed FFTs for modulus extraction.
- :doc:`../models/fractional/fractional_maxwell_model` — broadband chirps enable fitting of
  multi-order fractional models.
- :doc:`../models/flow/herschel_bulkley` — LAOS chirps are often paired with yield-stress
  model identification.
- :doc:`mutation_number` — evaluate whether chirp segments remain quasi-steady.
- :doc:`../../examples/transforms/04-owchirp-laos` — notebook demonstrating chirp design,
  playback, and analysis.
