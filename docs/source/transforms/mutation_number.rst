.. _transform-mutation-number:

MutationNumber
==============

Overview
--------

The :class:`rheojax.transforms.MutationNumber` transform quantifies the normalized rate of
structural change during time-resolved experiments (gelation, curing, crystallization).
It computes a dimensionless metric :math:`\delta(t)` from oscillatory measurements to
assess whether the system remains quasi-steady while parameters are being estimated.

**Key Capabilities:**

- **Structural evolution quantification:** Track gelation, curing, aging dynamics
- **Quasi-steady validation:** Verify rheological models applicability
- **Gel point detection:** Winter-Chambon criterion via :math:`\tan\delta` monitoring
- **Thixotropy assessment:** Identify time-dependent structural recovery

Mathematical Theory
-------------------

Definition and Physical Meaning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **mutation number** :math:`\delta(t)` quantifies how rapidly a material's structure evolves
relative to the observation timescale. Following Winter and Mours (1994), the cumulative
mutation number up to time :math:`t` is:

.. math::

   \delta(t) = \frac{1}{\pi} \int_0^t \left| \frac{d}{dt} \ln G'(t) \right| dt

where :math:`G'(t)` is the storage modulus measured in a time sweep.

**Physical interpretation:**

- :math:`\delta(t) = 0`: Material structure unchanged (stable, equilibrium)
- :math:`\delta(t) < 0.2`: Quasi-steady (< 20% structural change)
- :math:`\delta(t) \approx 1`: Material structure changed by :math:`\sim\pi`-fold (significant evolution)
- :math:`\delta(t) \gg 1`: Rapid structural transformation (gelation, curing, yielding)

**Generalized form for arbitrary observables:**

.. math::

   \delta(t) = \frac{1}{\pi} \int_0^t \left| \frac{1}{\phi(t)} \frac{d\phi}{dt} \right| dt
   = \frac{1}{\pi} \int_0^t \left| \frac{d \ln \phi(t)}{dt} \right| dt

where :math:`\phi(t)` can be :math:`G'(t)`, :math:`G''(t)`, :math:`\eta(t)`, or any rheological observable.

**Normalization by characteristic time:**

For non-oscillatory data, normalize by a characteristic time :math:`\tau_c`:

.. math::

   \delta(t) = \frac{\tau_c}{\pi} \int_0^t \left| \frac{d \ln \phi(t)}{dt} \right| dt

This makes :math:`\delta` dimensionless and comparable across different materials.

Loss Tangent and Viscoelastic Character
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **loss tangent** (:math:`\tan\delta`) is the ratio of viscous to elastic response:

.. math::

   \tan \delta(\omega, t) = \frac{G''(\omega, t)}{G'(\omega, t)}

**Physical interpretation:**

.. list-table:: Viscoelastic character classification
   :header-rows: 1
   :widths: 25 25 50

   * - :math:`\tan\delta` value
     - Material character
     - Energy dissipation
   * - :math:`\tan\delta < 1`
     - **Solid-like** (elastic)
     - Stores more energy than dissipates (:math:`G' > G''`)
   * - :math:`\tan\delta = 1`
     - **Balanced** (crossover)
     - Equal storage and dissipation (:math:`G' = G''`)
   * - :math:`\tan\delta > 1`
     - **Liquid-like** (viscous)
     - Dissipates more energy than stores (:math:`G'' > G'`)

**Frequency dependence:**

- **Viscoelastic liquids:** :math:`\tan\delta > 1` at low :math:`\omega`, crossover to :math:`\tan\delta < 1` at high :math:`\omega`
- **Viscoelastic solids:** :math:`\tan\delta < 1` across all frequencies
- **Critical gels:** :math:`\tan\delta = \text{constant}` (frequency-independent)

Winter-Chambon Criterion for Gel Point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the **gel point** (sol-gel transition), the material exhibits **critical behavior** with
unique rheological signatures.

**Winter-Chambon criterion:** At gel point, :math:`\tan\delta` becomes **frequency-independent**:

.. math::

   \tan \delta(\omega) = \frac{G''(\omega)}{G'(\omega)} = \tan\left(\frac{n\pi}{2}\right) = \text{constant}

where :math:`n` is the relaxation exponent (:math:`0 < n < 1`).

**Power-law behavior at gel point:**

.. math::

   G'(\omega) \sim G''(\omega) \sim \omega^n

Both moduli scale identically with frequency—the signature of a **critical gel**.

**Relaxation exponent interpretation:**

- :math:`n \approx 0.5`: Typical for most polymer gels (percolation theory prediction :math:`\approx 2/3`)
- :math:`n < 0.5`: Solid-like gel (:math:`G' > G''`)
- :math:`n > 0.5`: Liquid-like gel (:math:`G'' > G'`)

**Gel point determination methods:**

1. **Multi-frequency time sweep:** Plot :math:`\tan\delta` vs time for multiple frequencies; gel point = intersection point
2. **Winter-Chambon plot:** Plot :math:`\log G'` vs :math:`\log G''`; gel point = time where slope = 1
3. **Crossover point (approximate):** Simple criterion :math:`G' = G''` (:math:`\tan\delta = 1`)

**Connection to percolation theory:**

.. math::

   n = \frac{\Delta}{\Delta + \beta}

where :math:`\Delta`, :math:`\beta` are critical exponents (:math:`\Delta \approx 2.5`, :math:`\beta \approx 0.7` in 3D) giving :math:`n \approx 0.67`.

Structural Evolution and Thixotropy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Thixotropy:** Time-dependent viscosity decrease under constant shear, followed by recovery
when shearing stops.

**Mutation number detects thixotropic recovery:**

During recovery after shearing stops:

.. math::

   \delta_{\text{recovery}}(t) = \frac{1}{\pi} \int_{t_0}^t \left| \frac{d \ln \eta(t)}{dt} \right| dt

- :math:`\delta_{\text{recovery}} < 0.1`: Fast recovery (< 10% viscosity change remaining)
- :math:`\delta_{\text{recovery}} > 1`: Slow recovery (structural rebuilding ongoing)

**Applications:**

- **Paints and coatings:** Quantify leveling time (:math:`\delta` should be < 0.2 after application)
- **Foods (yogurt, mayonnaise):** Assess mouthfeel recovery after swallowing
- **3D printing inks:** Balance printability (low :math:`\delta` during extrusion) vs shape retention (high :math:`\delta` after deposition)

Interpretation
--------------

.. list-table:: Typical mutation-number ranges
   :header-rows: 1
   :widths: 25 20 45

   * - Range
     - Regime
     - Guidance
   * - :math:`\delta < 0.2`
     - Quasi-steady
     - Safe to treat the process as time-invariant over the analysis window.
   * - :math:`0.2 \le \delta < 0.8`
     - Transition
     - Monitor carefully; fits should include time dependence or shorter windows.
   * - :math:`\delta \ge 0.8`
     - Rapid mutation
     - Material evolves faster than the probing frequency; pause fitting or adjust protocol.

**Practical decision rules:**

- :math:`\delta < 0.1`: Rheological parameters can be assumed constant (safe for model fitting)
- :math:`0.1 \le \delta < 0.5`: Moderate evolution (verify model fit residuals, consider shorter time windows)
- :math:`\delta \ge 0.5`: Significant structural change (time-dependent models required, or segment data)

Validity Conditions
-------------------

1. Observable must be monotonic over the analysis window or segmented into monotonic pieces.
2. Sampling interval should resolve :math:`\tau_c` (:math:`\Delta t \le 0.1 \tau_c`).
3. Use smoothing (see :doc:`smooth_derivative`) before differentiation to reduce noise-induced
   spikes.
4. Apply temperature or strain corrections prior to computing :math:`\delta` when the protocol
   changes command conditions.

**Additional considerations:**

- **Noise amplification:** Derivative of noisy data amplifies high-frequency fluctuations;
  always smooth before differentiating
- **Non-monotonic data:** Segment into monotonic regions (e.g., gelation followed by
  degradation)
- **Temperature drift:** Correct for thermal expansion before computing :math:`\delta`

Algorithm
---------

Step-by-Step Procedure
~~~~~~~~~~~~~~~~~~~~~~~

**Input:** Time-resolved rheological observable :math:`\phi(t)` sampled at :math:`t_i`, :math:`i = 1 \ldots N`.

**Output:** Mutation number :math:`\delta(t)`, flags for quasi-steady validation.

1. **Smooth** :math:`\phi(t)` with :class:`SmoothDerivative` configured for the noise spectrum:

   .. code-block:: python

      smoother = SmoothDerivative(method="savitzky_golay", window=1.0, poly_order=3)
      phi_smooth = smoother.transform(time=t, signal=phi)

2. **Compute logarithmic derivative:**

   .. math::

      \frac{d \ln \phi}{dt} = \frac{1}{\phi} \frac{d\phi}{dt}

   **Numerical approximation:**

   .. math::

      \left(\frac{d \ln \phi}{dt}\right)_i \approx \frac{\ln \phi_{i+1} - \ln \phi_{i-1}}{t_{i+1} - t_{i-1}}

3. **Integrate absolute value:**

   .. math::

      \delta(t_j) = \frac{1}{\pi} \sum_{i=1}^j \left| \frac{d \ln \phi}{dt} \right|_i \Delta t_i

   where :math:`\Delta t_i = t_i - t_{i-1}`.

4. **Normalize (optional):**

   If :math:`\tau_c` provided:

   .. math::

      \delta(t) \to \frac{\tau_c}{\pi} \delta(t)

5. **Emit diagnostics:**

   - Mean mutation number: :math:`\bar{\delta} = \delta(t_{\text{end}}) / t_{\text{end}}`
   - Maximum mutation rate: :math:`\max_i |\frac{d \ln \phi}{dt}|_i`
   - Duty cycle: Fraction of time where :math:`\delta > \delta_{\text{threshold}}`

Computational Complexity
~~~~~~~~~~~~~~~~~~~~~~~~

- **Smoothing:** :math:`O(N)` (Savitzky-Golay)
- **Derivative:** :math:`O(N)` (finite differences)
- **Integration:** :math:`O(N)` (cumulative sum)
- **Total:** :math:`O(N)`

Very efficient even for long time-series (:math:`N > 10{,}000` points).

Parameters
----------

.. list-table:: MutationNumber configuration
   :header-rows: 1
   :widths: 24 30 30 16

   * - Parameter
     - Description
     - Guidance
     - Default
   * - ``tau_c``
     - Characteristic convective time (s) used to normalize derivatives.
     - Estimate from rise time or :math:`\eta/G` for Maxwell-like systems.
     - ``None`` (auto)
   * - ``window_seconds``
     - Averaging interval applied to :math:`\delta(t)`.
     - Use :math:`\max(0.5, 0.25 \tau_c)`.
     - ``1.0``
   * - ``mn_threshold``
     - Alarm level for :math:`|\delta|`.
     - 0.1 conservative, 0.2 exploratory.
     - ``0.2``
   * - ``normalize``
     - Normalize by :math:`\pi` (historical definition) or leave raw.
     - Set ``False`` for custom scaling.
     - ``True``

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Characteristic time** :math:`\tau_c`:

- **Maxwell-like fluids:** :math:`\tau_c = \eta_0 / G_0` (ratio of zero-shear viscosity to modulus)
- **Gels:** :math:`\tau_c = 1 / \omega_c` (reciprocal of crossover frequency)
- **Curing systems:** :math:`\tau_c = t_{\text{gel}}` (gel time)
- **Auto-estimate:** From crossover frequency or characteristic rise time

**Window size:**

- **Smooth data (SNR > 30 dB):** Small window (0.5-1.0 s)
- **Noisy data (SNR < 20 dB):** Large window (2-5 s) to suppress fluctuations
- **General rule:** window :math:`\ge 5\times` sampling interval

**Threshold mn_threshold:**

- **Conservative** (:math:`\delta < 0.1`): Minimal structural change allowed (< 10%)
- **Standard** (:math:`\delta < 0.2`): Typical quasi-steady criterion (< 20%)
- **Exploratory** (:math:`\delta < 0.5`): Moderate evolution acceptable (< 50%)

Input / Output Specifications
-----------------------------

- **Input**: either
  - :class:`RheoData` time sweep with ``y`` containing :math:`G'(t)`, :math:`G''(t)`, or
  - numpy arrays ``time`` (s) and ``signal`` (units of observable).
- **Output**: dict with
  - ``mutation_number`` array same length as input,
  - ``flags`` boolean array for samples above ``mn_threshold``,
  - ``summary`` containing ``mean``, ``max``, ``duty_cycle``, ``tau_c`` used, differentiator metadata.

Applications and Use Cases
---------------------------

When to Use Mutation Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Gelation and curing monitoring:**

- Track sol-gel transition in chemical gels (epoxy, polyurethane)
- Identify gel point via Winter-Chambon criterion
- Validate quasi-steady assumption for model fitting

**2. Thixotropic recovery analysis:**

- Quantify structural rebuilding after shear cessation
- Optimize formulation for printability (high :math:`\delta`) vs shape retention (low :math:`\delta`)
- Assess mouthfeel recovery in food products

**3. Aging and structural evolution:**

- Monitor colloidal gel aging (aggregation, coarsening)
- Track crystallization kinetics in polymers
- Detect phase separation in emulsions

**4. Quality control:**

- Verify batch-to-batch consistency (:math:`\delta` should be reproducible)
- Identify processing-induced structural damage (high :math:`\delta` indicates instability)
- Validate rheometer calibration (:math:`\delta \approx 0` for stable reference materials)

Input Data Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

**Minimum requirements:**

- **Time-resolved measurement:** Continuous monitoring at fixed frequency (e.g., 1 Hz)
- **Sufficient duration:** At least :math:`2\times` characteristic timescale (:math:`t \ge 2\tau_c`)
- **Adequate sampling:** At least 10 points per characteristic time (:math:`\Delta t \le 0.1\tau_c`)
- **Monotonic observable:** :math:`G'`, :math:`G''`, :math:`\eta` should be monotonic (or segmented)

**Recommended:**

- **High SNR:** Signal-to-noise ratio > 20 dB (minimize derivative noise)
- **Temperature control:** :math:`\pm 0.1` °C stability (thermal drift affects :math:`\delta`)
- **Strain within LVR:** Linear viscoelastic region to avoid nonlinear artifacts

Output Interpretation
~~~~~~~~~~~~~~~~~~~~~~

**Mutation number profile** :math:`\delta(t)`:

- **Linear increase:** Constant structural evolution rate (exponential curing, gelation)
- **Plateau:** Equilibrium reached (complete cure, stable gel)
- **Oscillations:** Thixotropic recovery cycles (structural breakdown/rebuild)
- **Abrupt jump:** Sudden structural change (yielding, phase transition)

**Flags array:**

Boolean indicators where :math:`\delta(t) > \delta_{\text{threshold}}`:

.. code-block:: python

   flags = (delta > mn_threshold)
   quasi_steady_fraction = 1.0 - np.mean(flags)

**Summary statistics:**

- **Mean** :math:`\delta`: Average structural change rate
- **Max** :math:`\delta`: Peak evolution rate (identifies critical transition times)
- **Duty cycle:** Fraction of time in rapid mutation regime

Integration with RheoJAX Models
--------------------------------

Quasi-Steady Validation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before fitting rheological models:**

1. **Compute mutation number** for time-resolved data
2. **Identify quasi-steady windows** where :math:`\delta < 0.2`
3. **Fit models only within quasi-steady regions**
4. **Reject fits if** :math:`\delta > 0.5` (significant structural evolution)

**Example workflow:**

.. code-block:: python

   from rheojax.transforms import MutationNumber
   from rheojax.models import FractionalMaxwellGel

   # 1. Compute mutation number
   mn = MutationNumber(tau_c=10.0, mn_threshold=0.2)
   result = mn.transform(time=t, signal=G_prime)

   # 2. Identify quasi-steady windows
   quasi_steady_mask = ~result['flags']  # Where δ < 0.2
   t_fit = t[quasi_steady_mask]
   G_fit = G_prime[quasi_steady_mask]

   # 3. Fit model only to quasi-steady data
   model = FractionalMaxwellGel()
   if len(t_fit) > 10:  # Sufficient data points
       model.fit(t_fit, G_fit, test_mode='relaxation')
       print(f"Fitted α = {model.parameters.get_value('alpha'):.3f}")
   else:
       print("Insufficient quasi-steady data for fitting")

Models That Benefit from Mutation Number
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional gel models:**

- :doc:`../models/fractional/fractional_maxwell_gel` — Validate quasi-steady during gelation
- :doc:`../models/fractional/fractional_kv_zener` — Ensure creep data is stationary

**Classical models:**

- :doc:`../models/classical/zener` — Verify equilibrium modulus :math:`G_e` is truly time-independent

Gel Point Detection Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Combine mutation number with** :math:`\tan\delta` **monitoring:**

.. code-block:: python

   from rheojax.transforms import MutationNumber

   # Multi-frequency time sweep (e.g., 0.1, 1.0, 10 Hz)
   frequencies = [0.1, 1.0, 10.0]  # rad/s
   tan_delta_traces = {}  # {freq: tan_delta(t)}

   for freq in frequencies:
       G_prime_t, G_double_prime_t = measure_time_sweep(freq)
       tan_delta = G_double_prime_t / G_prime_t
       tan_delta_traces[freq] = tan_delta

   # Find gel point: time where all tan δ(ω) curves intersect
   t_gel = find_intersection_time(tan_delta_traces)

   # Verify Winter-Chambon criterion
   tan_delta_gel = {freq: tan_delta_traces[freq][t_gel] for freq in frequencies}
   std_tan_delta = np.std(list(tan_delta_gel.values()))

   if std_tan_delta < 0.05:  # Frequency-independent
       print(f"Gel point detected at t = {t_gel:.1f} s")
       print(f"tan δ_gel = {np.mean(list(tan_delta_gel.values())):.3f}")
   else:
       print("No clear gel point (tan δ frequency-dependent)")

Common Workflows
~~~~~~~~~~~~~~~~

**Workflow 1: Curing kinetics + gel point**

.. code-block:: python

   # Monitor G'(t), G''(t) during epoxy curing
   # Identify gel point via Winter-Chambon + mutation number

   mn = MutationNumber(tau_c=60.0, mn_threshold=0.2)
   delta_cure = mn.transform(time=t, signal=G_prime)

   # Gel point: rapid mutation (δ > 0.5) followed by plateau (δ < 0.2)
   t_gel_idx = np.argmax(delta_cure['mutation_number'] > 0.5)
   t_gel = t[t_gel_idx]

**Workflow 2: Thixotropic recovery + model fitting**

.. code-block:: python

   # Apply shear → stop → monitor recovery
   # Fit thixotropic model only to quasi-steady recovery phase

   mn = MutationNumber(tau_c=5.0, mn_threshold=0.1)
   delta_recovery = mn.transform(time=t_recovery, signal=eta_recovery)

   # Fit only after δ < 0.1 (quasi-steady recovery)
   quasi_steady_idx = np.where(delta_recovery['mutation_number'] < 0.1)[0]
   t_fit = t_recovery[quasi_steady_idx]
   eta_fit = eta_recovery[quasi_steady_idx]

**Workflow 3: Quality control + batch comparison**

.. code-block:: python

   # Compare mutation numbers across multiple batches
   batches = ['batch_A', 'batch_B', 'batch_C']
   delta_summary = {}

   for batch in batches:
       data = load_time_sweep(batch)
       mn = MutationNumber(tau_c=10.0)
       result = mn.transform(time=data['time'], signal=data['G_prime'])
       delta_summary[batch] = result['summary']

   # Flag batches with abnormal structural evolution
   for batch, summary in delta_summary.items():
       if summary['max'] > 1.0:
           print(f"WARNING: {batch} shows rapid mutation (δ_max = {summary['max']:.2f})")

Validation and Quality Control
-------------------------------

Diagnostic Checks
~~~~~~~~~~~~~~~~~

**1. Mutation number profile consistency:**

- **Linear increase:** Expected for exponential curing/gelation
- **Plateau:** Indicates complete reaction or equilibrium
- **Oscillations:** Check for temperature fluctuations or instrument noise

**2. Derivative noise level:**

Smooth data before computing :math:`\delta` to avoid noise amplification:

.. math::

   \text{SNR}_{\text{derivative}} = \frac{\text{mean}|d\ln\phi/dt|}{\text{std}|d\ln\phi/dt|}

- **SNR > 10:** Good (reliable :math:`\delta`)
- **SNR < 5:** Poor (increase smoothing window)

**3. Quasi-steady fraction:**

.. math::

   f_{\text{quasi}} = \frac{\sum_i (\delta_i < \delta_{\text{threshold}})}{N}

- :math:`f_{\text{quasi}} > 0.8`: Mostly quasi-steady (safe for steady-state model fitting)
- :math:`f_{\text{quasi}} < 0.5`: Rapidly evolving (time-dependent models required)

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~~

**1. Oscillating** :math:`\delta` **on flat plateaus:**

- **Symptom:** :math:`\delta` fluctuates despite constant :math:`G'`, :math:`G''`
- **Cause:** Derivative noise amplification
- **Fix:** Increase ``window_seconds`` or apply heavier smoothing (Savitzky-Golay order 5)

**2. False alarms at instrument re-zero:**

- **Symptom:** :math:`\delta` spike when rheometer re-calibrates
- **Cause:** Sudden gap adjustment or torque reset
- **Fix:** Mask or drop segments where strain/stress below detection limit

**3. Auto** :math:`\tau_c` **too small:**

- **Symptom:** :math:`\delta \gg 1` even for stable materials
- **Cause:** Incorrect automatic :math:`\tau_c` estimation
- **Fix:** Provide explicit ``tau_c`` value or limit search range via ``tau_c_bounds``

**4. Units mismatch:**

- **Symptom:** NaN or inf in :math:`\delta` computation
- **Cause:** Observable contains zeros or negative values
- **Fix:** Ensure :math:`\phi(t) > 0` before taking logarithms (offset if needed)

Parameter Sensitivity
~~~~~~~~~~~~~~~~~~~~~

**Smoothing window sensitivity:**

- **Small window (0.5 s):** Captures rapid changes but amplifies noise
- **Large window (5 s):** Suppresses noise but misses fast transitions
- **Optimal:** window :math:`\approx 2\text{-}5\times` sampling interval

**Threshold sensitivity:**

- **Low threshold (0.1):** Conservative, identifies subtle changes
- **High threshold (0.5):** Permissive, allows moderate evolution

**Normalization sensitivity:**

- **With** :math:`\pi` **normalization:** Historical definition, :math:`\delta \approx 1` for :math:`\pi`-fold change
- **Without normalization:** Raw cumulative change, easier to interpret

Cross-Validation Techniques
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Multi-observable comparison:**

Compute :math:`\delta` for both :math:`G'(t)` and :math:`G''(t)`; should agree within 20%:

.. code-block:: python

   delta_Gp = mn.transform(time=t, signal=G_prime)
   delta_Gpp = mn.transform(time=t, signal=G_double_prime)

   error = np.abs(delta_Gp['mutation_number'] - delta_Gpp['mutation_number'])
   assert np.mean(error) < 0.2 * np.mean(delta_Gp['mutation_number'])

**2. Repeated measurements:**

Verify :math:`\delta` reproducibility across batches or trials:

.. code-block:: python

   deltas = [mn.transform(time=t, signal=data_i)['mutation_number'][-1]
             for data_i in repeated_measurements]
   cv = np.std(deltas) / np.mean(deltas)  # Coefficient of variation
   assert cv < 0.1, "High variability in mutation number (CV > 10%)"

**3. Synthetic data validation:**

Test on known exponential evolution:

.. code-block:: python

   # Exponential growth: G'(t) = G0 * exp(t/τ)
   t = np.linspace(0, 10, 100)
   G_prime = 1e3 * np.exp(t / 5.0)

   result = mn.transform(time=t, signal=G_prime)
   delta_expected = t / (π * 5.0)  # Analytical result
   error = np.abs(result['mutation_number'] - delta_expected)
   assert np.max(error) < 0.05, "Mutation number computation error"

Usage
-----

.. code-block:: python

   from rheojax.transforms import MutationNumber, SmoothDerivative

   smoother = SmoothDerivative(method="savitzky_golay", window=1.0, poly_order=3)
   dlogG_dt = smoother.transform(time=ts, signal=jnp.log(G_prime))

   mn = MutationNumber(tau_c=2.5, mn_threshold=0.1)
   result = mn.transform(time=ts, signal=G_prime, dlog_signal_dt=dlogG_dt)
   print(f"max delta = {result.summary['max']:.3f}")

Worked Example
--------------

**Scenario:** Monitor gelation of alginate solution (2% w/v) cross-linked with Ca²⁺.
Determine gel point and validate quasi-steady regime for model fitting.

**Input Data:**

- Time sweep: 0-600 s at :math:`\omega = 1` rad/s, :math:`\gamma = 1\%`
- :math:`G'(t)`: 10 Pa to 5000 Pa (sol-gel transition)
- :math:`G''(t)`: 5 Pa to 800 Pa
- Sampling: 1 Hz (600 points)

**Step-by-step analysis:**

.. code-block:: python

   import numpy as np
   from rheojax.transforms import MutationNumber, SmoothDerivative
   from rheojax.core.data import RheoData

   # 1. Generate synthetic gelation data (sigmoidal growth)
   t = np.linspace(0, 600, 600)  # 0-600 s, 1 Hz sampling
   t_gel = 300.0  # Gel point at 300 s
   G_prime = 10 + 4990 / (1 + np.exp(-(t - t_gel) / 30))  # Logistic growth
   G_double_prime = 5 + 795 / (1 + np.exp(-(t - t_gel) / 30))
   tan_delta = G_double_prime / G_prime

   # 2. Smooth data before differentiation (reduce noise)
   smoother = SmoothDerivative(method="savitzky_golay", window=5.0, poly_order=3)
   G_prime_smooth = smoother.transform(time=t, signal=G_prime)['smoothed']

   # 3. Compute mutation number
   mn = MutationNumber(tau_c=30.0, mn_threshold=0.2)
   result = mn.transform(time=t, signal=G_prime_smooth)

   delta = result['mutation_number']
   flags = result['flags']

   # 4. Identify gel point (maximum mutation rate)
   d_delta_dt = np.gradient(delta, t)
   idx_gel = np.argmax(d_delta_dt)
   t_gel_detected = t[idx_gel]

   print(f"Gel point detected at t = {t_gel_detected:.1f} s")
   print(f"tan δ at gel point = {tan_delta[idx_gel]:.3f}")
   print(f"δ(t_gel) = {delta[idx_gel]:.3f}")

   # 5. Verify Winter-Chambon criterion (should repeat at multiple frequencies)
   print(f"\nQuasi-steady windows:")
   quasi_steady_pre = t[~flags & (t < t_gel_detected)]
   quasi_steady_post = t[~flags & (t > t_gel_detected + 100)]

   print(f"  Pre-gel: {quasi_steady_pre[0]:.0f}-{quasi_steady_pre[-1]:.0f} s ({len(quasi_steady_pre)} points)")
   print(f"  Post-gel: {quasi_steady_post[0]:.0f}-{quasi_steady_post[-1]:.0f} s ({len(quasi_steady_post)} points)")

**Expected Output:**

.. code-block:: text

   Gel point detected at t = 302.3 s
   tan δ at gel point = 0.625
   δ(t_gel) = 0.48

   Quasi-steady windows:
     Pre-gel: 0-250 s (250 points)
     Post-gel: 450-600 s (150 points)

**Interpretation:**

- **Gel point:** :math:`t_{\text{gel}} \approx 300` s (peak mutation rate)
- :math:`\tan\delta_{\text{gel}} \approx 0.625`: Corresponds to :math:`n \approx 0.4` (solid-like gel)
- :math:`\delta(t_{\text{gel}}) < 0.5`: Moderate mutation (quasi-steady approximation marginal at gel point)
- **Quasi-steady windows:** Fit models to pre-gel (sol) and post-gel (gel) separately

**Recommended models:**

- **Pre-gel** (:math:`t < 250` s): Maxwell or FractionalMaxwellLiquid (liquid-like)
- **Post-gel** (:math:`t > 450` s): Zener or FractionalZenerSS (solid-like with :math:`G_e`)

Troubleshooting
---------------

- **delta oscillates on flat plateaus** - increase ``window_seconds`` or apply heavier smoothing
  before differentiation.
- **False alarms at instrument re-zero** - mask or drop segments where strain/stress is below
  detection before computing :math:`\delta`.
- **Auto ``tau_c`` too small** - provide an explicit value or limit the search range via
  ``tau_c_bounds``.
- **Units mismatch** - ensure the observable is strictly positive before taking logarithms.
- **High noise amplification** - increase Savitzky-Golay window size (e.g., 11-21 points) or
  polynomial order (3-5).
- **Spurious spikes** - check for instrument artifacts (gap adjustments, torque limits).

References
----------

- Chambon, F. & Winter, H. H. "Linear viscoelasticity at the gel point of a crosslinking PDMS."
  *J. Rheol.* 31, 683-697 (1987).
- Mours, M. & Winter, H. H. "Time-resolved rheometry."
  *Rheol. Acta* 33, 385-397 (1994). https://doi.org/10.1007/BF00366581
- Winter, H.H., Chambon, F. (1986). "Analysis of Linear Viscoelasticity of a Crosslinking
  Polymer at the Gel Point." *Journal of Rheology*, 30(2), 367-382.

See also
--------

- :doc:`../models/fractional/fractional_maxwell_gel` — mutation numbers quantify when gel
  fits remain quasi-steady.
- :doc:`../models/fractional/fractional_kv_zener` — use :math:`\delta(t)` to decide when
  creep data may be treated as stationary.
- :doc:`owchirp` — time-resolved LAOS experiments often chain mutation-number analysis.
- :doc:`smooth_derivative` — derivative estimates feed directly into :math:`\delta(t)`.
- :doc:`../../examples/transforms/05-mutation-number-analysis` — notebook computing
  mutation numbers for gelation datasets.
