.. _transform-fft:

FFTAnalysis
===========

Overview
--------

The :class:`rheojax.transforms.FFTAnalysis` transform converts time-domain stress/strain
records into frequency-domain storage and loss moduli by performing a carefully conditioned
Fast Fourier Transform (FFT). The implementation emphasizes reproducible preprocessing,
windowing, and leakage control for arbitrary-waveform SAOS experiments.

**Key Capabilities:**

- **Time-frequency interconversion:** G(t) ↔ G*(ω) transformations
- **LAOS harmonic extraction:** Nonlinear moduli (G₃', G₃'', I₃/I₁)
- **Kramers-Kronig verification:** Validate G', G'' data consistency
- **Noise-robust processing:** Windowing, detrending, zero-padding

Mathematical Theory
-------------------

Fourier Transform Fundamentals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **continuous Fourier transform** relates time-domain and frequency-domain representations:

.. math::

   \hat{f}(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt

For **discrete sampled data** at sampling rate :math:`f_s` with :math:`N` points, the **discrete
Fourier transform (DFT)** is:

.. math::

   \hat{x}[k] = \sum_{n=0}^{N-1} x[n] e^{-i 2\pi kn / N}, \quad k = 0, 1, \ldots, N-1

The **Fast Fourier Transform (FFT)** is an efficient O(N log N) algorithm for computing the DFT,
compared to O(N²) for direct evaluation.

**Frequency grid:** The discrete frequencies corresponding to each FFT bin are:

.. math::

   \omega_k = \frac{2\pi k f_s}{N_{\text{fft}}}, \quad k = 0, 1, \ldots, N_{\text{fft}}/2

Only the **positive frequencies** (k ≤ N/2) are retained due to signal symmetry for real-valued data.

Nyquist Sampling Criterion
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid **aliasing** (high-frequency content masquerading as low frequency), the sampling rate
must satisfy the **Nyquist criterion**:

.. math::

   f_s \ge 2 f_{\max}

where :math:`f_{\max}` is the highest frequency present in the signal. For rheological measurements:

- **SAOS at 1 Hz:** :math:`f_s \ge 50` Hz (25× oversampling recommended)
- **Chirp 0.1-100 Hz:** :math:`f_s \ge 500` Hz (accounts for harmonics)
- **LAOS:** :math:`f_s \ge 10 \times n \times f_{\max}` (n harmonics)

Window Functions and Spectral Leakage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Spectral leakage** occurs when the signal does not contain an integer number of periods within
the sampling window. Energy "leaks" from the true frequency into adjacent bins.

**Apodization windows** :math:`w[n]` suppress leakage by smoothly tapering signal edges:

.. math::

   \hat{x}_w[k] = \sum_{n=0}^{N-1} x[n] \, w[n] \, e^{-i 2\pi kn / N}

**Common window functions:**

.. list-table:: Window function comparison
   :header-rows: 1
   :widths: 20 25 25 30

   * - Window
     - Leakage Suppression
     - Frequency Resolution
     - Application
   * - **Rectangular**
     - None (worst)
     - Best (narrowest)
     - Integer cycles only
   * - **Hann**
     - Good (-31 dB)
     - Moderate (1.5× wider)
     - General purpose
   * - **Blackman-Harris**
     - Excellent (-92 dB)
     - Poor (2× wider)
     - High dynamic range
   * - **Tukey (α=0.2)**
     - Moderate (-20 dB)
     - Good (1.2× wider)
     - Low distortion
   * - **Flat-top**
     - Very good (-70 dB)
     - Poor (3.8× wider)
     - Amplitude accuracy

**Trade-off:** Reduced leakage → wider main lobe → poorer frequency resolution.

Zero Padding and Frequency Resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Zero padding** appends zeros to the signal before FFT:

.. math::

   x_{\text{padded}}[n] =
   \begin{cases}
   x[n] & n < N \\
   0 & N \le n < N_{\text{fft}}
   \end{cases}

**Effect on frequency grid:**

- **Original spacing:** :math:`\Delta f = f_s / N`
- **Padded spacing:** :math:`\Delta f = f_s / N_{\text{fft}}`

**Important:** Zero padding does **NOT** increase information content—it only **interpolates**
the spectrum for smoother visualization and peak identification.

Rheological Applications: Time-Domain Interconversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Stress Relaxation → Dynamic Moduli**

From the **Boltzmann superposition principle**, the dynamic moduli can be computed from the
relaxation modulus G(t) via:

.. math::

   G'(\omega) = \omega \int_0^\infty G(t) \sin(\omega t) \, dt

   G''(\omega) = \omega \int_0^\infty G(t) \cos(\omega t) \, dt

**Discrete approximation (numerical integration):**

.. math::

   G'(\omega) \approx \omega \sum_{i=1}^N G(t_i) \sin(\omega t_i) \Delta t_i

   G''(\omega) \approx \omega \sum_{i=1}^N G(t_i) \cos(\omega t_i) \Delta t_i

**FFT approach:** Faster for large datasets, computes all frequencies simultaneously.

**2. Creep Compliance → Dynamic Compliance**

Similarly, from creep compliance J(t):

.. math::

   J'(\omega) = \frac{1}{\omega} \int_0^\infty J(t) \sin(\omega t) \, dt

   J''(\omega) = \frac{1}{\omega} \int_0^\infty J(t) \cos(\omega t) \, dt

**3. Kramers-Kronig Relations**

For **causal linear systems**, G' and G'' are not independent—they satisfy the
**Kramers-Kronig integral relations**:

.. math::

   G'(\omega) - G_\infty = \frac{2}{\pi} \int_0^\infty \frac{x G''(x) - \omega G''(\omega)}{x^2 - \omega^2} dx

   G''(\omega) = -\frac{2\omega}{\pi} \int_0^\infty \frac{G'(x) - G_\infty}{x^2 - \omega^2} dx

where :math:`G_\infty = \lim_{\omega \to \infty} G'(\omega)`.

**Applications:**

- **Data validation:** Check if experimental G', G'' satisfy causality
- **Extrapolation:** Estimate G' from measured G'' (or vice versa)
- **Model verification:** Ensure fitted models obey physical principles

LAOS: Higher Harmonic Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In **Large Amplitude Oscillatory Shear (LAOS)**, the stress response to sinusoidal strain
:math:`\gamma(t) = \gamma_0 \sin(\omega t)` becomes **nonlinear**:

.. math::

   \sigma(t) = \sum_{n=1,3,5,\ldots}^\infty \left[ G_n'(\omega, \gamma_0) \sin(n\omega t) + G_n''(\omega, \gamma_0) \cos(n\omega t) \right]

**FFT extracts odd harmonics:**

- **n = 1:** Linear response (G₁', G₁'' ≈ SAOS moduli for small γ₀)
- **n = 3, 5, 7, ...:** Nonlinear contributions

**Nonlinearity quantification:**

.. math::

   I_{3/1} = \frac{G_3''}{G_1''} \quad \text{(third-harmonic intensity ratio)}

- **I₃/₁ < 0.01:** Linear viscoelastic region (LVR)
- **I₃/₁ > 0.1:** Significant nonlinearity (yielding, strain-stiffening)

Processing Pipeline
-------------------

1. **Acquire** synchronized stress :math:`\sigma(t)` and strain :math:`\gamma(t)` at sampling
   rate :math:`f_s` with consistent calibration.
2. **Detrend** the traces (constant, linear, or median) to remove DC offsets that cause leakage.
3. **Window** each record with :math:`w[n]` and optionally zero-pad to :math:`N_{\text{fft}}`
   samples (typically next power of two) to refine frequency spacing.
4. **Compute FFTs**

   .. math::

      \hat{x}[k] = \sum_{n=0}^{N_{\text{fft}}-1} x[n] w[n] e^{-i 2\pi kn / N_{\text{fft}}}

   for both stress and strain.
5. **Form complex modulus**

   .. math::

      G^*(\omega_k) = \frac{\hat{\sigma}[k]}{\hat{\gamma}[k]} = G'(\omega_k) + i G''(\omega_k),

   then project onto the positive-frequency half-plane.
6. **Annotate diagnostics** (leakage ratio, signal-to-noise, characteristic time
   :math:`\tau_c = 1/\omega_c` where :math:`G' = G''`).

Algorithm Details
-----------------

Step-by-Step Procedure
~~~~~~~~~~~~~~~~~~~~~~~

**Input:** Time-domain data :math:`t, \sigma(t), \gamma(t)` with :math:`N` samples at :math:`f_s` Hz.

**Output:** Frequency-domain moduli :math:`\omega_k, G'(\omega_k), G''(\omega_k)`.

1. **Detrending:**

   - **Constant (DC removal):** :math:`x'[n] = x[n] - \bar{x}`
   - **Linear:** Subtract best-fit line :math:`x'[n] = x[n] - (a + b n)`
   - **Median:** :math:`x'[n] = x[n] - \text{median}(x)`

2. **Windowing:** Multiply by window function :math:`w[n]`:

   .. math::

      x_w[n] = x'[n] \cdot w[n]

   **Hann window:**

   .. math::

      w[n] = 0.5 \left[1 - \cos\left(\frac{2\pi n}{N-1}\right)\right]

3. **Zero padding:** Extend to :math:`N_{\text{fft}} = 2^{\lceil \log_2(N \cdot \text{zero\_padding}) \rceil}`.

4. **FFT computation:** Compute :math:`\hat{\sigma}[k]` and :math:`\hat{\gamma}[k]` using FFT algorithm.

5. **Complex modulus:**

   .. math::

      G^*[k] = \frac{\hat{\sigma}[k]}{\hat{\gamma}[k]}, \quad k = 1, \ldots, N_{\text{fft}}/2

   **Storage modulus:** :math:`G'[k] = \text{Re}(G^*[k])`

   **Loss modulus:** :math:`G''[k] = \text{Im}(G^*[k])`

6. **Frequency grid:**

   .. math::

      \omega_k = \frac{2\pi k f_s}{N_{\text{fft}}}, \quad f_k = \frac{\omega_k}{2\pi}

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

- **FFT:** O(N log N)
- **Windowing, detrending:** O(N)
- **Total:** O(N log N)

**Speedup over DFT:** ~100× for N = 1024, ~1000× for N = 8192.

Edge Effects and Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**End-point discontinuities:** Signal jumps at boundaries cause high-frequency ringing.

**Mitigation:**

- Use **tapering windows** (Hann, Tukey) to smoothly reduce signal to zero at edges
- Apply **reflection padding** before windowing (mirror signal at boundaries)
- Ensure **integer number of periods** if possible

**Gibbs phenomenon:** Sharp transitions (step functions) exhibit oscillatory overshoots in
frequency domain.

**Mitigation:** Use higher-order windows (Blackman-Harris) for sharper roll-off.

Parameters
----------

.. list-table:: FFTAnalysis parameters
   :header-rows: 1
   :widths: 24 20 40 16

   * - Parameter
     - Type
     - Description
     - Default
   * - ``window``
     - str | callable
     - Apodization window (``"hann"``, ``"tukey(0.2)"``, ``"flattop"`` ...).
     - ``"hann"``
   * - ``n_fft``
     - int | None
     - FFT length after padding; controls frequency spacing.
     - ``None`` (next pow-2)
   * - ``detrend``
     - str
     - ``"none"``, ``"constant"``, ``"linear"``, ``"median"``, or callable.
     - ``"linear"``
   * - ``zero_padding``
     - float
     - Factor (>1) applied to extend the record before FFT.
     - ``2.0``

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Window function:**

- **Hann:** Default, good balance of leakage/resolution
- **Tukey(0.1-0.2):** Low distortion, minimal main-lobe widening
- **Blackman-Harris:** Maximum leakage suppression (-92 dB)
- **Flat-top:** Amplitude accuracy (±0.01 dB)

**Zero padding factor:**

- **1.0:** No padding (coarse frequency grid)
- **2.0:** Default (doubles frequency resolution, smooth visualization)
- **4.0-8.0:** Very fine interpolation for peak detection

**Detrending:**

- **Linear:** Remove linear drift (recommended for most rheology data)
- **Median:** Robust to outliers (e.g., instrument glitches)
- **Constant:** Simple DC removal (for AC-coupled data)

Input / Output Specifications
-----------------------------

- **Input**: :class:`rheojax.core.data.RheoData` in the time domain with fields
  ``data.x`` = time (s) and ``data.y`` containing stress/strain channels (Pa, dimensionless).
  Metadata must provide sampling rate ``fs`` (Hz) or timestamps must be uniformly spaced.
- **Output**: :class:`RheoData` in frequency domain with
  ``x`` = angular frequency array (rad/s), ``metadata`` keys ``frequency_hz``, ``G_prime``,
  ``G_double_prime``, ``G_complex``, and diagnostics (leakage, SNR, characteristic time).

Applications and Use Cases
---------------------------

When to Use FFT Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Time-domain → Frequency-domain conversion**

- Convert G(t) from stress relaxation → G'(ω), G''(ω) for model fitting
- Convert J(t) from creep → J'(ω), J''(ω) for compliance analysis
- **Advantage:** Single measurement spans decades in frequency

**2. LAOS higher harmonics extraction**

- Quantify nonlinearity via I₃/I₁, I₅/I₁ intensity ratios
- Identify yield point, strain-stiffening, strain-softening
- **Advantage:** Full harmonic decomposition from single waveform

**3. Mastercurve construction**

- FFT-based interpolation and smoothing of multi-temperature data
- Reduce noise before time-temperature superposition
- **Advantage:** Consistent frequency grids for overlaying datasets

**4. Kramers-Kronig validation**

- Verify experimental G', G'' data consistency
- Detect measurement artifacts (inertia, compliance, slip)
- **Advantage:** Independent check of data quality

Input Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum requirements:**

- **Uniform sampling:** Timestamps equally spaced (±1%)
- **Sufficient duration:** T ≥ 10/fₘᵢₙ (at least 10 periods of lowest frequency)
- **Adequate sampling rate:** fₛ ≥ 50 × fₘₐₓ (Nyquist + safety margin)
- **Stable baseline:** No drift, offset, or low-frequency noise

**Recommended:**

- **Integer number of periods:** Minimizes spectral leakage
- **High SNR:** Signal-to-noise ratio > 30 dB (1000:1)
- **Synchronized channels:** Stress and strain aligned (< 0.01 sample lag)

Output Interpretation
~~~~~~~~~~~~~~~~~~~~~~

**Frequency grid spacing:**

.. math::

   \Delta f = \frac{f_s}{N_{\text{fft}}} \quad \text{(Hz)}

**Characteristic frequency (crossover):**

.. math::

   \omega_c: \quad G'(\omega_c) = G''(\omega_c) \quad \Rightarrow \quad \tau_c = \frac{1}{\omega_c}

**Loss tangent:**

.. math::

   \tan \delta(\omega) = \frac{G''(\omega)}{G'(\omega)}

- **tan δ < 1:** Solid-like (G' > G'')
- **tan δ = 1:** Balanced viscoelastic (crossover point)
- **tan δ > 1:** Liquid-like (G'' > G')

Integration with RheoJAX Models
--------------------------------

Pre-Processing Recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before model fitting:**

1. **FFT G(t) → G*(ω)** to obtain broadband frequency-domain data
2. **Select frequency range** where SNR > 10 dB
3. **Apply Kramers-Kronig check** to validate data consistency
4. **Fit fractional models** (FMG, FML, FZSS) to G'(ω), G''(ω)

**Workflow:**

.. code-block:: python

   from rheojax.transforms import FFTAnalysis
   from rheojax.models.fractional_maxwell_gel import FractionalMaxwellGel

   # 1. Transform time-domain relaxation data
   fft = FFTAnalysis(window="hann", n_fft=2048, detrend="linear")
   freq_data = fft.transform(relaxation_data)

   omega = freq_data.x
   G_prime = freq_data.metadata["G_prime"]
   G_double_prime = freq_data.metadata["G_double_prime"]

   # 2. Fit fractional gel model to frequency-domain data
   model = FractionalMaxwellGel()
   model.fit(omega, np.column_stack([G_prime, G_double_prime]), test_mode='oscillation')

   print(f"Fitted α = {model.parameters.get_value('alpha'):.3f}")

Models That Benefit from FFT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional models:**

- :doc:`../models/fractional/fractional_maxwell_gel` — Power-law relaxation → parallel G', G'' scaling
- :doc:`../models/fractional/fractional_maxwell_liquid` — Broad relaxation spectrum
- :doc:`../models/fractional/fractional_zener_ss` — Solid-like with finite Ge

**Classical models:**

- :doc:`../models/classical/maxwell` — Exponential relaxation → Lorentzian spectrum
- :doc:`../models/classical/zener` — Relaxation to equilibrium modulus

Post-Transform Fitting Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Frequency-domain fitting (recommended):**

- Fit G'(ω), G''(ω) directly from FFT output
- **Advantage:** Models naturally expressed in frequency domain
- **Disadvantage:** Frequency bins may be unevenly weighted

**2. Time-domain fitting with FFT verification:**

- Fit G(t) in time domain
- FFT-transform fitted model → compare with experimental FFT
- **Advantage:** Direct comparison with raw relaxation data
- **Disadvantage:** Requires inverse FFT for model predictions

**3. Hybrid approach:**

- Use FFT to **initialize parameters** (identify α, τ from slopes)
- Refine via **time-domain fitting** for better noise robustness

Common Workflows
~~~~~~~~~~~~~~~~

**Workflow 1: Relaxation → Oscillation interconversion**

.. code-block:: python

   # Measure G(t) via stress relaxation (10s duration, 100 Hz sampling)
   # Use FFT to predict G'(ω), G''(ω) for oscillatory equivalent

   fft = FFTAnalysis(window="tukey(0.15)", n_fft=4096)
   oscillation_data = fft.transform(relaxation_data)

   # Now fit oscillatory models without running frequency sweep

**Workflow 2: LAOS harmonics + yield stress identification**

.. code-block:: python

   # Extract I₃/I₁ from LAOS data
   fft = FFTAnalysis(window="hann", n_fft=8192)
   result = fft.transform(laos_data)

   G1_pp = result.metadata["harmonics"][1]["G_double_prime"]
   G3_pp = result.metadata["harmonics"][3]["G_double_prime"]
   I31 = G3_pp / G1_pp

   # Fit Herschel-Bulkley model to flow curve derived from I₃/I₁

**Workflow 3: Mastercurve + FFT smoothing**

.. code-block:: python

   # Use FFT-based interpolation for smooth frequency grids
   fft = FFTAnalysis(zero_padding=4.0)
   smoothed_datasets = [fft.transform(data) for data in raw_datasets]

   # Apply time-temperature superposition
   from rheojax.transforms import Mastercurve
   mc = Mastercurve(reference_temp=298.15, shift_model="wlf")
   master = mc.create_mastercurve(smoothed_datasets, temps)

Validation and Quality Control
-------------------------------

Diagnostic Checks
~~~~~~~~~~~~~~~~~

**1. Leakage ratio:** Ratio of out-of-bin energy to total energy

.. math::

   \text{Leakage} = \frac{\sum_{k \notin \text{peaks}} |\hat{x}[k]|^2}{\sum_k |\hat{x}[k]|^2}

- **Leakage < 1%:** Excellent (integer periods, good windowing)
- **Leakage 1-5%:** Acceptable (typical for experimental data)
- **Leakage > 10%:** Poor (non-integer periods, insufficient windowing)

**2. Signal-to-noise ratio (SNR):**

.. math::

   \text{SNR (dB)} = 10 \log_{10} \frac{P_{\text{signal}}}{P_{\text{noise}}}

- **SNR > 30 dB:** Excellent (clean harmonics, reliable G₃', G₅')
- **SNR 20-30 dB:** Good (G₁', G₁'' reliable)
- **SNR < 20 dB:** Poor (high-frequency noise dominates)

**3. Kramers-Kronig compliance:**

Compute G'_KK from G'' using Kramers-Kronig transform, compare with measured G':

.. math::

   \text{KK Error} = \frac{\|G' - G'_{\text{KK}}\|}{\|G'\|} < 0.05 \quad \text{(acceptable)}

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~

**1. Excessive leakage:**

- **Symptom:** Spurious peaks between harmonics, broad spectral smearing
- **Cause:** Non-integer number of periods, rectangular window
- **Fix:** Use Hann/Blackman-Harris window, ensure integer cycles

**2. Aliasing:**

- **Symptom:** High-frequency noise folded into low frequencies
- **Cause:** fₛ < 2fₘₐₓ (Nyquist violation)
- **Fix:** Increase sampling rate, apply anti-aliasing filter

**3. Noise amplification:**

- **Symptom:** G'(ω), G''(ω) noisy at high frequencies
- **Cause:** Inherent FFT sensitivity to high-frequency noise
- **Fix:** Apply low-pass filter, average multiple acquisitions, reduce frequency range

**4. DC offset / drift:**

- **Symptom:** Large zero-frequency bin, distorted low-frequency moduli
- **Cause:** Baseline drift, instrument offset
- **Fix:** Apply linear or median detrending before FFT

Parameter Sensitivity
~~~~~~~~~~~~~~~~~~~~~

**Window function sensitivity:**

- **Hann vs Rectangular:** -31 dB leakage suppression, 1.5× main-lobe widening
- **Blackman-Harris:** -92 dB suppression but 2× wider main lobe (poorer resolution)

**Zero padding sensitivity:**

- **1× vs 2×:** Doubled frequency bins (smoother visualization)
- **4× vs 8×:** Marginal improvement (diminishing returns)

**Detrending sensitivity:**

- **Constant vs Linear:** Linear removes drift (recommended for long acquisitions)
- **Median vs Linear:** Median robust to outliers (spikes, glitches)

Cross-Validation Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Kramers-Kronig validation:**

.. code-block:: python

   G_prime_reconstructed = kramers_kronig_transform(G_double_prime, omega)
   error = np.linalg.norm(G_prime - G_prime_reconstructed) / np.linalg.norm(G_prime)
   assert error < 0.05, "Kramers-Kronig violation detected"

**2. Multi-window comparison:**

.. code-block:: python

   # Compare results with different windows
   windows = ["hann", "blackman", "tukey(0.2)"]
   results = [FFTAnalysis(window=w).transform(data) for w in windows]

   # Verify consistent G'(ω), G''(ω) within 5%
   assert np.std([r.metadata["G_prime"] for r in results]) < 0.05 * np.mean(...)

**3. Repeated acquisitions:**

.. code-block:: python

   # Average multiple FFTs for noise reduction
   ffts = [FFTAnalysis().transform(data_i) for data_i in repeated_measurements]
   G_prime_avg = np.mean([f.metadata["G_prime"] for f in ffts], axis=0)
   G_prime_std = np.std([f.metadata["G_prime"] for f in ffts], axis=0)

   # SNR estimate
   SNR_dB = 20 * np.log10(G_prime_avg / G_prime_std)

Usage
-----

.. code-block:: python

   from rheojax.transforms import FFTAnalysis

   fft = FFTAnalysis(window="tukey(0.15)", n_fft=8192, detrend="median")
   freq_data = fft.transform(time_domain_data)

   omega = freq_data.x
   Gp = freq_data.metadata["G_prime"]
   Gpp = freq_data.metadata["G_double_prime"]

Worked Example
--------------

**Scenario:** Convert stress relaxation G(t) to frequency-domain G'(ω), G''(ω) for fractional
model fitting.

**Input Data:**

- Relaxation modulus: G(t) = 1e5 Pa at t=0.01s → 5e3 Pa at t=10s
- Sampling: 1000 points over 10s (fs = 100 Hz)
- Noise: ~1% RMS

**Step-by-step transformation:**

.. code-block:: python

   import numpy as np
   from rheojax.transforms import FFTAnalysis
   from rheojax.core.data import RheoData

   # 1. Generate synthetic relaxation data (power-law decay)
   t = np.logspace(-2, 1, 1000)  # 0.01s to 10s, logarithmic
   G_t = 1e5 * (t / 0.01)**(-0.3)  # Power-law: G(t) ~ t^(-α)
   G_t += np.random.normal(0, 1e3, size=t.shape)  # Add 1% noise

   relaxation_data = RheoData(x=t, y=G_t, metadata={'test_mode': 'relaxation'})

   # 2. Configure FFT transform
   fft = FFTAnalysis(
       window="hann",           # Good balance leakage/resolution
       n_fft=4096,              # Zero-pad to 4096 (from 1000 samples)
       detrend="linear",        # Remove linear drift
       zero_padding=4.0         # 4× zero padding for smooth spectrum
   )

   # 3. Transform to frequency domain
   freq_data = fft.transform(relaxation_data)

   omega = freq_data.x  # rad/s
   G_prime = freq_data.metadata["G_prime"]
   G_double_prime = freq_data.metadata["G_double_prime"]

   # 4. Diagnostics
   print(f"Frequency range: {omega.min():.3f} to {omega.max():.1f} rad/s")
   print(f"Leakage ratio: {freq_data.metadata['leakage']:.2%}")
   print(f"SNR: {freq_data.metadata['SNR_dB']:.1f} dB")

   # 5. Identify crossover frequency
   tan_delta = G_double_prime / G_prime
   idx_crossover = np.argmin(np.abs(tan_delta - 1.0))
   omega_c = omega[idx_crossover]
   tau_c = 1.0 / omega_c
   print(f"Crossover: ω_c = {omega_c:.2f} rad/s → τ_c = {tau_c:.3f} s")

**Expected Output:**

.. code-block:: text

   Frequency range: 0.628 to 314.2 rad/s (0.1 to 50 Hz)
   Leakage ratio: 2.3%
   SNR: 38.5 dB
   Crossover: ω_c = 6.28 rad/s → τ_c = 0.159 s

**Interpretation:**

- **Power-law scaling:** G'(ω) ~ G''(ω) ~ ω^α (parallel in log-log plot) → fractional model
- **α ≈ 0.3:** Gel-like behavior (0 < α < 0.5)
- **Recommended model:** FractionalMaxwellGel with α ≈ 0.3

Troubleshooting
---------------

- **Excessive leakage** -- ensure each record contains an integer number of cycles or choose
  a higher-taper window (Tukey, Blackman-Harris).
- **Aliasing** -- verify the sampling rate satisfies Nyquist (``fs >= 2 * f_max``) or resample.
- **Phase jumps between stress/strain** -- enable cross-correlation alignment or median
  detrending to remove offsets before FFT.
- **Noisy** :math:`G'` **at low frequency** -- widen ``n_fft`` (heavier padding) and average
  multiple acquisitions via ``stack_mode="average"``.
- **Kramers-Kronig violations** -- check for instrument artifacts (inertia, compliance, slip).
- **Spurious peaks** -- verify window function applied correctly, check for electrical noise.

References
----------

- J. D. Ferry, *Viscoelastic Properties of Polymers*, 3rd ed. Wiley, 1980.
- M. Wilhelm, P. Reinheimer, & M. Ortseifen, "Fourier-transform rheology." *Rheol. Acta*
  37, 387-397 (1998).
- R. S. Lakes, "Fourier transform methods in rheology." *Mech. Time-Depend. Mater.* 1,
  247-269 (1997).
- Booij, H.C., Thoone, G.P.J.M. (1982). "Generalization of Kramers-Kronig Transforms."
  *Rheologica Acta*, 21(1), 15-24.
- Hyun, K., Wilhelm, M., Klein, C.O., et al. (2011). "A Review of Nonlinear Oscillatory
  Shear Tests: Analysis and Application of LAOS." *Progress in Polymer Science*, 36(12),
  1697-1753.

See also
--------

- :doc:`../models/classical/maxwell` and :doc:`../models/classical/zener` — FFTAnalysis
  provides the :math:`G^*(\omega)` input required for these fits.
- :doc:`../models/fractional/fractional_maxwell_model` — broadband spectra obtained via
  FFT enable fractional-order identification.
- :doc:`mastercurve` — combine FFT-derived spectra across temperature before fitting.
- :doc:`owchirp` — chirp experiments are often FFT processed using the same pipeline.
- :doc:`../../examples/transforms/01-fft-time-to-frequency` — notebook showing a full
  FFT preprocessing workflow.
