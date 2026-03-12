.. _transform-prony-conversion:

PronyConversion
===============

Overview
--------

The :class:`rheojax.transforms.PronyConversion` transform converts between time-domain and
frequency-domain viscoelastic data using the **Prony series** decomposition. The Prony series
is the discrete analog of the continuous relaxation spectrum and provides exact analytical
relations between :math:`G(t)`, :math:`G'(\omega)`, and :math:`G''(\omega)`.

**Key Capabilities:**

- **Time → frequency conversion:** :math:`G(t) \to G'(\omega), G''(\omega)` via Prony fitting
- **Frequency → time conversion:** :math:`G'(\omega), G''(\omega) \to G(t)` via inverse Prony
- **Automatic mode selection:** Number of modes auto-selected based on data density
- **JAX-accelerated evaluation:** JIT-compiled Prony summations for fast evaluation

Unlike the :doc:`fft` approach (which is non-parametric), Prony conversion produces a
compact parametric representation that can be evaluated at arbitrary output points and
naturally integrates with :doc:`lve_envelope` and multi-mode model families.


.. _prony-vs-fft-vs-owchirp:

Comparison with FFT and OWChirp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All three transforms can convert time-domain data to frequency-domain, but they are
fundamentally different tools designed for different use cases:

.. list-table:: Domain Conversion Transforms Compared
   :header-rows: 1
   :widths: 18 27 27 28

   * - Aspect
     - **PronyConversion**
     - **FFT Analysis**
     - **OWChirp**
   * - **Method**
     - Parametric model fitting (NNLS)
     - Non-parametric signal processing (DFT)
     - Wavelet-based time-frequency analysis
   * - **Input**
     - :math:`G(t)` or :math:`G'(\omega), G''(\omega)` (material functions)
     - Raw time-domain signal :math:`x(t)` (stress, strain waveforms)
     - LAOS waveform :math:`\sigma(t)` (periodic oscillatory)
   * - **Output**
     - :math:`G^*(\omega)` or :math:`G(t)` + Prony parameters :math:`(G_i, \tau_i)`
     - Frequency spectrum, PSD, harmonic amplitudes
     - Time-frequency map + harmonic content vs time
   * - **Frequency grid**
     - Arbitrary (evaluate Prony at any :math:`\omega`)
     - Fixed by sampling rate (:math:`\Delta f = f_s/N`)
     - Adaptive (wavelet resolution varies with frequency)
   * - **Key output**
     - Compact parameters for downstream use (LVE envelope, model initialization)
     - Spectral magnitude and phase
     - Time-resolved nonlinear indicators (:math:`I_3/I_1` vs time)
   * - **Assumes**
     - Linear viscoelasticity (superposition of exponentials)
     - Uniform sampling, stationarity
     - Periodic LAOS excitation
   * - **Reversible**
     - Yes (bidirectional: time ↔ freq)
     - Yes (via inverse FFT)
     - No (analysis only, not synthesis)

**When to use which:**

- **PronyConversion:** You have :math:`G(t)` or :math:`G^*(\omega)` and need the *other*
  domain representation, *or* you need Prony parameters for :doc:`lve_envelope` /
  :doc:`../models/gmm/generalized_maxwell`. Best for material functions measured under
  standard protocols (relaxation, frequency sweep).

- **FFT Analysis:** You have raw time-domain waveforms (stress/strain records) and need
  to extract the frequency content. Best for signal processing—detecting harmonics,
  computing PSD, Kramers-Kronig checks. Does *not* produce Prony parameters.

- **OWChirp:** You have LAOS data and need *time-resolved* frequency analysis (how does
  nonlinearity evolve *within* the deformation cycle or across cycles). Best for chirp
  experiments, curing/gelation monitoring, or tracking structural changes.

**They complement, not overlap.** A typical multi-step workflow might chain them:

1. **OWChirp** → extract :math:`G'(\omega)`, :math:`G''(\omega)` from chirp experiment
2. **PronyConversion** → fit Prony series to the extracted :math:`G^*(\omega)`
3. **LVEEnvelope** → predict startup stress from the Prony parameters

Or: **FFT** → convert :math:`G(t)` to :math:`G^*(\omega)` → **SpectrumInversion** →
recover :math:`H(\tau)`.


Mathematical Theory
-------------------

Prony Series Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Prony series** (generalized Maxwell model) expresses the relaxation modulus as a sum
of decaying exponentials:

.. math::

   G(t) = G_e + \sum_{i=1}^{N} G_i \exp(-t / \tau_i)

where:

- :math:`G_e` is the **equilibrium modulus** (:math:`G_e = 0` for liquids, :math:`G_e > 0` for solids)
- :math:`G_i` are the **mode strengths** (Pa)
- :math:`\tau_i` are the **relaxation times** (s)
- :math:`N` is the number of modes

Analytical Frequency-Domain Relations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Substituting the Prony series into the Boltzmann superposition integral yields
**closed-form** expressions for the dynamic moduli:

**Storage modulus** (elastic component):

.. math::

   G'(\omega) = G_e + \sum_{i=1}^{N} G_i \frac{\omega^2 \tau_i^2}{1 + \omega^2 \tau_i^2}

**Loss modulus** (viscous component):

.. math::

   G''(\omega) = \sum_{i=1}^{N} G_i \frac{\omega \tau_i}{1 + \omega^2 \tau_i^2}

Each mode contributes a **Debye relaxation peak** to :math:`G''(\omega)` centered at
:math:`\omega = 1/\tau_i`.

**Key properties:**

- :math:`G'(\omega \to 0) = G_e` (equilibrium modulus)
- :math:`G'(\omega \to \infty) = G_e + \sum G_i` (glassy modulus)
- :math:`G''` peak for mode :math:`i` occurs at :math:`\omega_{\text{peak}} = 1/\tau_i`
  with height :math:`G_i / 2`

Fitting Procedure
~~~~~~~~~~~~~~~~~

**Time-domain fitting** (:math:`G(t) \to` Prony parameters):

1. Log-space the relaxation times: :math:`\tau_i` from :math:`t_{\min}` to :math:`t_{\max}`
2. Estimate :math:`G_e = G(t_{\max})` (long-time plateau)
3. Build kernel matrix :math:`A_{ji} = \exp(-t_j / \tau_i)`
4. Solve **non-negative least squares** (NNLS): :math:`\min_{\mathbf{G} \ge 0} \|A \mathbf{G} - (G(t) - G_e)\|^2`

**Frequency-domain fitting** (:math:`G'(\omega), G''(\omega) \to` Prony parameters):

1. Log-space :math:`\tau_i` from :math:`1/\omega_{\max}` to :math:`1/\omega_{\min}`
2. Estimate :math:`G_e = \min(G')`
3. Build stacked kernel:

   .. math::

      A = \begin{bmatrix}
      \omega^2 \tau^2 / (1 + \omega^2 \tau^2) \\
      \omega \tau / (1 + \omega^2 \tau^2)
      \end{bmatrix}

4. Solve NNLS jointly on :math:`[G' - G_e, \; G'']`

**NNLS guarantees** :math:`G_i \ge 0`, ensuring thermodynamic consistency (positive
relaxation spectrum).


Parameters
----------

.. list-table:: PronyConversion Parameters
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Default
     - Description
   * - ``n_modes``
     - int | None
     - ``None``
     - Number of Prony modes. If ``None``, auto-selected as
       ``max(3, min(N_data // 5, 20))``.
   * - ``direction``
     - str
     - ``"time_to_freq"``
     - Conversion direction: ``"time_to_freq"`` or ``"freq_to_time"``.
   * - ``omega_out``
     - ndarray | None
     - ``None``
     - Target frequency array for time→freq. Auto-generated if ``None``.
   * - ``t_out``
     - ndarray | None
     - ``None``
     - Target time array for freq→time. Auto-generated if ``None``.

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Number of modes:**

- **3–5 modes:** Single-relaxation materials (dilute solutions, simple gels)
- **8–12 modes:** Moderate complexity (polymer melts, concentrated solutions)
- **15–20 modes:** Broad spectra (blends, filled systems, multi-component)
- **Auto-select:** Safe default—uses ``N_data // 5`` capped at 20

**Direction choice:**

- ``"time_to_freq"``: Start from stress relaxation :math:`G(t)` data
- ``"freq_to_time"``: Start from frequency sweep :math:`G'(\omega)`, :math:`G''(\omega)` data


Input / Output Specifications
-----------------------------

**Time → Frequency:**

- **Input**: :class:`RheoData` with ``x`` = time (s), ``y`` = :math:`G(t)` (Pa)
- **Output**: :class:`RheoData` with ``x`` = :math:`\omega` (rad/s), ``y`` = :math:`G^* = G' + iG''` (Pa)

**Frequency → Time:**

- **Input**: :class:`RheoData` with ``x`` = :math:`\omega` (rad/s), ``y`` = complex :math:`G^*` or ``(N, 2)`` array :math:`[G', G'']`
- **Output**: :class:`RheoData` with ``x`` = time (s), ``y`` = :math:`G(t)` (Pa)

The metadata dict includes a :class:`PronyResult` with ``G_i``, ``tau_i``, ``G_e``, and ``n_modes``.


Usage
-----

Time-Domain to Frequency-Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import PronyConversion
   from rheojax.core.data import RheoData
   import numpy as np

   # Stress relaxation data
   t = np.logspace(-2, 2, 200)
   G_t = 1e4 * np.exp(-t / 0.5) + 500 * np.exp(-t / 10.0)

   data = RheoData(x=t, y=G_t, metadata={'test_mode': 'relaxation'})

   # Convert to frequency domain
   prony = PronyConversion(n_modes=10, direction="time_to_freq")
   freq_data, info = prony.transform(data)

   omega = freq_data.x
   G_star = freq_data.y  # Complex: G' + iG''
   G_prime = G_star.real
   G_double_prime = G_star.imag

   # Access Prony parameters
   result = info["prony_result"]
   print(f"Modes: {result.n_modes}, G_e: {result.G_e:.1f} Pa")

Frequency-Domain to Time-Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import PronyConversion
   import numpy as np

   # Oscillatory data (frequency sweep)
   omega = np.logspace(-2, 2, 50)
   G_prime = 1e4 * omega**2 / (1 + omega**2)
   G_double_prime = 1e4 * omega / (1 + omega**2)
   G_star = G_prime + 1j * G_double_prime

   osc_data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})

   # Convert to time domain
   prony = PronyConversion(direction="freq_to_time", n_modes=8)
   relax_data, info = prony.transform(osc_data)

   t = relax_data.x
   G_t = relax_data.y  # Relaxation modulus G(t)

Integration with LVE Envelope
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import PronyConversion, LVEEnvelope

   # Step 1: Fit Prony series to relaxation data
   prony = PronyConversion(n_modes=10, direction="time_to_freq")
   _, info = prony.transform(relaxation_data)
   result = info["prony_result"]

   # Step 2: Compute startup stress envelope from Prony parameters
   lve = LVEEnvelope(
       shear_rate=0.1,
       G_i=result.G_i,
       tau_i=result.tau_i,
       G_e=result.G_e,
   )
   envelope_data, _ = lve.transform()

   # Compare with experimental startup data
   import matplotlib.pyplot as plt
   plt.loglog(envelope_data.x, envelope_data.y, '--', label='LVE envelope')
   plt.loglog(startup_data.x, startup_data.y, 'o', label='Experiment')
   plt.xlabel('Time (s)')
   plt.ylabel('Stress (Pa)')
   plt.legend()

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.pipeline import Pipeline
   from rheojax.transforms import PronyConversion
   from rheojax.models import FractionalMaxwellGel

   pipe = Pipeline()
   pipe.load(relaxation_data)
   pipe.transform(PronyConversion(n_modes=12, direction="time_to_freq"))
   pipe.fit(FractionalMaxwellGel(), test_mode='oscillation')
   pipe.plot()


Validation and Quality Control
-------------------------------

Assessing Fit Quality
~~~~~~~~~~~~~~~~~~~~~~

**R² metric:** Compare reconstructed data with original:

.. code-block:: python

   # Round-trip validation: G(t) → Prony → G'(ω) → Prony → G(t)
   prony_fwd = PronyConversion(n_modes=12, direction="time_to_freq")
   freq_data, info1 = prony_fwd.transform(relaxation_data)

   prony_inv = PronyConversion(n_modes=12, direction="freq_to_time",
                                t_out=relaxation_data.x)
   recon_data, info2 = prony_inv.transform(freq_data)

   # Compare
   ss_res = np.sum((relaxation_data.y - recon_data.y)**2)
   ss_tot = np.sum((relaxation_data.y - np.mean(relaxation_data.y))**2)
   R2 = 1 - ss_res / ss_tot
   print(f"Round-trip R² = {R2:.6f}")

**Mode convergence:** Increase ``n_modes`` until output stabilizes.

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~~

**1. Overfitting (too many modes):**

- **Symptom:** Oscillatory artifacts in reconstructed data, spurious peaks in :math:`G''`
- **Fix:** Reduce ``n_modes``, or use :doc:`spectrum_inversion` with regularization

**2. Underfitting (too few modes):**

- **Symptom:** Poor fit at short or long times, missing features
- **Fix:** Increase ``n_modes``, verify data spans sufficient decades

**3. Negative G_e estimate:**

- **Symptom:** Warning about negative target values
- **Cause:** Data does not reach equilibrium, or :math:`G(t_{\max})` underestimated
- **Fix:** Extend measurement time, or manually set ``G_e=0``


See Also
--------

- :doc:`spectrum_inversion` — Continuous spectrum :math:`H(\tau)` via regularization
  (complementary to discrete Prony representation)
- :doc:`lve_envelope` — Uses Prony parameters to compute startup stress envelope
- :doc:`fft` — Non-parametric time↔frequency conversion
- :doc:`../models/classical/maxwell` — Single Maxwell element (1-mode Prony)
- :doc:`../models/gmm/generalized_maxwell` — Multi-mode generalized Maxwell (N-mode Prony)


API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.PronyConversion`


References
----------

1. Ferry, J.D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed. Wiley.
   Chapter 3: Exact interconversions via Prony series.

2. Tschoegl, N.W. (1989). *The Phenomenological Theory of Linear Viscoelastic
   Behavior: An Introduction*. Springer-Verlag.
   DOI: `10.1007/978-3-642-73602-5 <https://doi.org/10.1007/978-3-642-73602-5>`_

3. Baumgaertel, M. & Winter, H.H. (1989). "Determination of discrete relaxation
   and retardation time spectra from dynamic mechanical data." *Rheol. Acta*,
   28, 511–519. DOI: `10.1007/BF01332922 <https://doi.org/10.1007/BF01332922>`_

4. Honerkamp, J. & Weese, J. (1993). "A nonlinear regularization method for the
   calculation of relaxation spectra." *Rheol. Acta*, 32, 65–73.
   DOI: `10.1007/BF00396678 <https://doi.org/10.1007/BF00396678>`_
