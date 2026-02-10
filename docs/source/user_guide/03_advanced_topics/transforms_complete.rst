.. _transforms_complete:

====================================
Complete Guide to Data Transforms
====================================

**Processing Rheological Data for Advanced Analysis**

Overview
========

RheoJAX provides **7 specialized data transforms** that process raw rheological measurements
into derived quantities. These transforms operate on ``RheoData`` objects or raw NumPy/JAX
arrays and extract physically meaningful information for material characterization, nonlinearity
quantification, and data quality assessment.

.. admonition:: Key Insight
   :class: tip

   Transforms are **pre-processing** and **analysis** tools—not constitutive models. They convert
   raw experimental data (stress, strain, time, frequency) into derived physical quantities
   (harmonics, shift factors, yield stresses, moduli distributions) that reveal material behavior
   or prepare data for model fitting.

**Why Use Transforms?**

1. **Extend measurement ranges**: TTS/SRFS create mastercurves spanning 8-12 decades
2. **Extract hidden physics**: SPP reveals yield stresses and cage moduli from LAOS waveforms
3. **Quantify nonlinearity**: FFT harmonics and Mutation Number measure departure from linearity
4. **Improve signal quality**: Smooth Derivative handles noisy experimental data
5. **Accelerate measurements**: OWChirp processes chirp data 10-100× faster than discrete sweeps

When to Use Transforms
=======================

.. list-table:: Transform Decision Guide
   :header-rows: 1
   :widths: 30 35 35

   * - I Have...
     - I Want...
     - Use This Transform
   * - LAOS time-domain data
     - Harmonic amplitudes I₁, I₃, I₅...
     - :ref:`FFTAnalysis <fft_transform>`
   * - Freq sweeps at multiple T
     - Single master curve
     - :ref:`Mastercurve <mastercurve_transform>` (TTS)
   * - Flow curves at multiple γ̇
     - Flow curve master + noise temp
     - :ref:`SRFS <srfs_transform>`
   * - LAOS waveform data
     - Yield stress, cage modulus
     - :ref:`SPPDecomposer <spp_transform>`
   * - Noisy experimental data
     - Smooth derivative dG/dt
     - :ref:`SmoothDerivative <smooth_derivative_transform>`
   * - Any rheological data
     - Linearity check (Δ)
     - :ref:`MutationNumber <mutation_number_transform>`
   * - Chirp oscillatory data
     - G'(ω), G''(ω) from single test
     - :ref:`OWChirp <owchirp_transform>`

.. _fft_transform:

1. FFT (Fast Fourier Transform)
=================================

**Frequency Decomposition of Oscillatory Signals**

Theory
------

The **Fast Fourier Transform (FFT)** decomposes a periodic time-domain signal into its constituent
frequency components. For Large Amplitude Oscillatory Shear (LAOS), the stress response becomes
nonsinusoidal and contains higher harmonics:

.. math::

   \sigma(t) = \sum_{n=1,3,5,...}^{\infty} \left[ \sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t) \right]

where:
- :math:`\omega`: Fundamental frequency (rad/s)
- :math:`n`: Harmonic number (odd integers for symmetric strain)
- :math:`\sigma'_n, \sigma''_n`: Elastic and viscous stress components at harmonic n

**Key Nonlinearity Metrics:**

.. math::

   \text{Nonlinearity ratio} = \frac{I_3}{I_1} = \frac{|\sigma_3|}{|\sigma_1|}

   \text{Total harmonic distortion} = \frac{\sqrt{\sum_{n=3,5,...} I_n^2}}{I_1}

where :math:`I_n = \sqrt{(\sigma'_n)^2 + (\sigma''_n)^2}` is the intensity of harmonic n.

**Physical Interpretation:**
- :math:`I_3/I_1 < 0.01`: Linear viscoelastic regime (SAOS)
- :math:`I_3/I_1 \sim 0.1`: Weakly nonlinear (MAOS)
- :math:`I_3/I_1 > 0.5`: Strongly nonlinear (LAOS)—yield, flow, cage breaking

Usage
-----

.. code-block:: python

   from rheojax.transforms import FFTAnalysis
   from rheojax.core.data import RheoData
   import numpy as np

   # Generate LAOS data (10 cycles, omega = 1 rad/s)
   t = np.linspace(0, 10*2*np.pi, 2000)
   omega = 1.0
   gamma_0 = 1.0  # Large strain amplitude

   # Nonlinear stress response (contains 3rd harmonic)
   stress = 100*np.sin(omega*t) + 20*np.sin(3*omega*t)

   # Create RheoData
   data = RheoData(x=t, y=stress, test_mode='oscillation')

   # Apply FFT with Hann window to reduce spectral leakage
   fft = FFTAnalysis(window='hann', detrend=True)
   result = fft.transform(data)

   # Extract harmonics (result is dict: {1: {...}, 3: {...}, ...})
   harmonics = result['harmonics']
   I1 = harmonics[1]['amplitude']
   I3 = harmonics[3]['amplitude']
   I5 = harmonics[5]['amplitude']

   print(f"Fundamental amplitude I₁ = {I1:.1f} Pa")
   print(f"Third harmonic I₃ = {I3:.1f} Pa")
   print(f"Nonlinearity ratio I₃/I₁ = {I3/I1:.4f}")

**Window Functions:**

.. list-table:: FFT Window Selection
   :header-rows: 1
   :widths: 20 40 40

   * - Window
     - Use Case
     - Trade-off
   * - ``'hann'``
     - General-purpose (default)
     - Balanced resolution/leakage
   * - ``'blackman'``
     - Minimize spectral leakage
     - Wider main lobe (lower resolution)
   * - ``'bartlett'``
     - Fast decay, simple
     - Moderate leakage
   * - ``'none'``
     - Perfect periodic data
     - High leakage if not periodic

.. note::

   **Data Requirements**: FFT requires **integer number of cycles** for accurate harmonic extraction.
   For LAOS, use at least 5-10 cycles in steady state. Discard initial transient cycles to avoid
   spectral artifacts.

.. _mastercurve_transform:

2. Mastercurve (Time-Temperature Superposition)
=================================================

**Extending Frequency Range via Temperature Sweeps**

Theory
------

**Time-Temperature Superposition (TTS)** exploits the separability of temperature effects in
thermorheologically simple materials. All relaxation processes accelerate uniformly with temperature,
allowing horizontal shifting of frequency sweeps to create a **mastercurve**:

.. math::

   G'(\omega, T) = G'(\omega \cdot a_T, T_{\text{ref}})

   G''(\omega, T) = G''(\omega \cdot a_T, T_{\text{ref}})

**Williams-Landel-Ferry (WLF) Equation** (for polymers near T_g):

.. math::

   \log_{10}(a_T) = \frac{-C_1 (T - T_{\text{ref}})}{C_2 + (T - T_{\text{ref}})}

**Universal constants** (T_ref = T_g): :math:`C_1 = 17.44`, :math:`C_2 = 51.6` K

**Arrhenius Equation** (for high-temperature melts, T ≫ T_g):

.. math::

   \log(a_T) = \frac{E_a}{R} \left( \frac{1}{T} - \frac{1}{T_{\text{ref}}} \right)

where :math:`E_a` is activation energy (J/mol), :math:`R = 8.314` J/(mol·K).

**Applicability:**
- ✓ Amorphous polymers, polymer melts, viscoelastic liquids
- ✗ Crystalline polymers, phase-separated blends, chemical reactions during heating

Usage
-----

.. code-block:: python

   from rheojax.transforms import Mastercurve
   from rheojax.core.data import RheoData

   # Multi-temperature frequency sweep data
   datasets = []
   temperatures = [40, 60, 80, 100, 120]  # Celsius

   for T in temperatures:
       # Load experimental data at each temperature
       omega, G_prime, G_double_prime = load_saos_data(T)

       # Create RheoData with temperature metadata
       data = RheoData(
           x=omega,
           y=G_prime + 1j*G_double_prime,  # Complex modulus
           test_mode='oscillation',
           metadata={'temperature': T}
       )
       datasets.append(data)

   # Automatic shift factor determination
   mc = Mastercurve(
       reference_temp=80.0,  # Reference temperature (Celsius)
       method='wlf',         # or 'arrhenius'
       auto_shift=True       # Optimize shift factors
   )

   # Generate mastercurve
   master_curve, shift_factors = mc.transform(datasets)

   # Access WLF parameters (if fitted)
   wlf_params = mc.get_wlf_parameters()
   print(f"C₁ = {wlf_params['C1']:.2f}")
   print(f"C₂ = {wlf_params['C2']:.1f} K")

   # Plot mastercurve (now spans 8-12 decades instead of 2-3)
   import matplotlib.pyplot as plt
   plt.loglog(master_curve.x, master_curve.y.real, 'o-', label="G'")
   plt.loglog(master_curve.x, master_curve.y.imag, 's-', label="G''")
   plt.xlabel("$\\omega \\cdot a_T$ (rad/s)")
   plt.ylabel("G', G'' (Pa)")
   plt.legend()
   plt.show()

**Manual WLF Shifting:**

.. code-block:: python

   # Use known WLF parameters instead of auto-fitting
   mc = Mastercurve(
       reference_temp=373.15,  # 100°C in Kelvin
       method='wlf',
       C1=17.44,  # Universal value
       C2=51.6,   # Universal value
       auto_shift=False
   )
   master_curve, shift_factors = mc.transform(datasets)

.. warning::

   **Temperature Units**: ``reference_temp`` expects Celsius if metadata temperatures are in Celsius,
   or Kelvin if metadata uses Kelvin. **Be consistent** to avoid incorrect shift factors.

.. _srfs_transform:

3. SRFS (Strain-Rate Frequency Superposition)
===============================================

**Flow Curve Mastercurves via Shear Rate Shifting**

Theory
------

**Strain-Rate Frequency Superposition (SRFS)** is the flow curve analog of Time-Temperature
Superposition. Instead of shifting frequency data at multiple temperatures, SRFS shifts
flow curves at multiple shear rates to reveal universal scaling behavior.

**Connection to Soft Glassy Rheology (SGR):**

For SGR materials, the shift factor follows a power-law:

.. math::

   a(\dot{\gamma}) = \left( \frac{\dot{\gamma}}{\dot{\gamma}_{\text{ref}}} \right)^{2-x}

where :math:`x` is the **effective noise temperature** (SGR parameter):
- :math:`x < 1`: Glass (aging, yield stress)
- :math:`1 < x < 2`: Power-law fluid
- :math:`x \geq 2`: Newtonian liquid

**Physical Interpretation**: The exponent :math:`2-x` reveals the material's proximity to the
glass transition. Materials near :math:`x = 1` exhibit strong shear thinning and SRFS collapse.

**Shear Banding Detection**: SRFS automatically detects **coexistence plateaus** in flow curves—
signatures of spatial heterogeneity (banding) where two flow states coexist at the same stress.

Usage
-----

.. code-block:: python

   from rheojax.transforms import SRFS, detect_shear_banding
   from rheojax.core.data import RheoData

   # Multi-rate flow curve data (e.g., yield stress fluid)
   datasets = []
   shear_rates = [0.1, 1.0, 10.0, 100.0]  # s^-1

   for gamma_dot in shear_rates:
       # Load steady-shear data at each rate
       stress = measure_flow_curve(gamma_dot)

       data = RheoData(
           x=np.array([gamma_dot]),  # Single point per rate
           y=stress,
           test_mode='flow_curve',
           metadata={'shear_rate': gamma_dot}
       )
       datasets.append(data)

   # SRFS mastercurve with auto shift
   srfs = SRFS(reference_gamma_dot=1.0, auto_shift=True)
   master_flow, shift_factors = srfs.transform(datasets)

   # Extract SGR noise temperature from power-law fit
   exponent = srfs.get_shift_exponent()  # 2 - x
   x_sgr = 2 - exponent
   print(f"SGR noise temperature x = {x_sgr:.3f}")

   if x_sgr < 1.0:
       print("Material is a glass (aging, yield stress)")
   elif x_sgr < 2.0:
       print("Material is a power-law fluid")
   else:
       print("Material is Newtonian")

   # Detect shear banding (plateau in stress vs gamma_dot)
   banding_info = detect_shear_banding(datasets, threshold=0.05)
   if banding_info['is_banding']:
       print(f"Shear banding detected!")
       print(f"Coexistence region: {banding_info['plateau_range']}")

**Shear Banding Analysis:**

.. code-block:: python

   from rheojax.transforms import compute_shear_band_coexistence

   # Full steady-shear data (stress vs gamma_dot)
   gamma_dot = np.logspace(-2, 2, 100)
   stress = measure_full_flow_curve(gamma_dot)

   # Detect plateau (constant stress over range of gamma_dot)
   coexist = compute_shear_band_coexistence(
       gamma_dot, stress,
       threshold=0.05  # 5% stress variation defines plateau
   )

   if coexist['is_coexistence']:
       print(f"Coexistence stress: {coexist['plateau_stress']:.1f} Pa")
       print(f"Rate range: {coexist['rate_range']}")

.. note::

   **Shear Banding** occurs in wormlike micelles, colloidal glasses, and other soft materials
   under shear. The flow curve develops a **stress plateau** where high-shear and low-shear
   bands coexist in the gap, violating the monotonic stress-rate relationship.

.. _spp_transform:

4. SPP Decomposer (Sequence of Physical Processes)
====================================================

**Time-Domain LAOS Analysis Without Fourier Limitations**

Theory
------

The **Sequence of Physical Processes (SPP)** framework (Rogers et al., 2011) analyzes Large
Amplitude Oscillatory Shear (LAOS) data directly in the time domain, extracting **transient
moduli** and **yield stresses** without harmonic decomposition.

**Key Innovation**: SPP views the stress waveform as representing a **sequence of physical events**
within each cycle:
1. **Elastic extension** (cage deformation): :math:`G_{\text{cage}}`
2. **Yielding** (cage breaking): :math:`\sigma_{sy}` (static), :math:`\sigma_{dy}` (dynamic)
3. **Flow** (viscous dissipation): :math:`\eta'(t)`
4. **Reformation** (cage rebuilding)

**Transient Moduli:**

.. math::

   G'_t(t) = \frac{\partial \sigma(t)}{\partial \gamma(t)}

   \eta'_t(t) = \frac{\partial \sigma(t)}{\partial \dot{\gamma}(t)}

These **time-dependent** moduli vary within each cycle, revealing when the material transitions
from elastic to viscous behavior.

**Extracted Parameters:**
- :math:`G_{\text{cage}}`: Maximum elastic modulus (cage stiffness)
- :math:`\sigma_{sy}`: **Static yield stress** (onset of flow from rest)
- :math:`\sigma_{dy}`: **Dynamic yield stress** (flow cessation during oscillation)
- :math:`\eta_{\min}`: Minimum viscosity (maximum fluidization)

Usage
-----

.. code-block:: python

   from rheojax.transforms import SPPDecomposer, spp_analyze
   from rheojax.core.data import RheoData
   import numpy as np

   # LAOS experimental data (10 cycles in steady state)
   t = np.linspace(0, 10*2*np.pi, 2000)
   omega = 1.0  # rad/s
   gamma_0 = 1.0  # Large strain amplitude

   # Measured stress (nonlinear waveform)
   stress = load_laos_stress(t)
   strain = gamma_0 * np.sin(omega*t)
   strain_rate = gamma_0 * omega * np.cos(omega*t)

   # Create RheoData with full LAOS information
   data = RheoData(
       x=t,
       y=stress,
       test_mode='oscillation',
       metadata={
           'strain': strain,
           'strain_rate': strain_rate,
           'omega': omega,
           'gamma_0': gamma_0
       }
   )

   # SPP decomposition with Rogers-parity defaults
   decomposer = SPPDecomposer(
       omega=omega,
       gamma_0=gamma_0,
       n_harmonics=39,  # Match SPPplus MATLAB (odd harmonics)
       step_size=8,     # 8-point 4th-order stencil
       num_mode=2       # Periodic differentiation
   )

   result = decomposer.transform(data)

   # Extract physical parameters
   spp_results = decomposer.get_results()

   print(f"Cage modulus: {spp_results['G_cage']:.1f} Pa")
   print(f"Static yield stress: {spp_results['sigma_sy']:.1f} Pa")
   print(f"Dynamic yield stress: {spp_results['sigma_dy']:.1f} Pa")
   print(f"Minimum viscosity: {spp_results['eta_min']:.2f} Pa·s")

   # Plot transient moduli within a single cycle
   import matplotlib.pyplot as plt

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

   # Elastic modulus vs strain (Lissajous-Bowditch curve)
   ax1.plot(result['strain'], result['G_prime_t'], 'b-')
   ax1.axhline(spp_results['G_cage'], color='r', linestyle='--',
               label=f"$G_{{cage}}$ = {spp_results['G_cage']:.0f} Pa")
   ax1.set_xlabel("Strain γ")
   ax1.set_ylabel("G'(t) (Pa)")
   ax1.legend()

   # Viscosity vs strain rate
   ax2.plot(result['strain_rate'], result['eta_prime_t'], 'g-')
   ax2.axhline(spp_results['eta_min'], color='r', linestyle='--',
               label=f"$\\eta_{{min}}$ = {spp_results['eta_min']:.2f} Pa·s")
   ax2.set_xlabel("Strain rate γ̇ (s⁻¹)")
   ax2.set_ylabel("η'(t) (Pa·s)")
   ax2.legend()

   plt.tight_layout()
   plt.show()

**Convenience Function for Quick Analysis:**

.. code-block:: python

   # All-in-one SPP analysis with automatic plotting
   from rheojax.transforms import spp_analyze

   results = spp_analyze(
       time=t,
       stress=stress,
       strain=strain,
       omega=omega,
       gamma_0=gamma_0,
       plot=True  # Auto-generate standard SPP plots
   )

.. admonition:: SPP vs Fourier: When to Use Each
   :class: tip

   **Use SPP when:**
   - Material has yield stress (foam, paste, gel)
   - You need transient moduli G'(t), η'(t) within cycle
   - Physical interpretation is priority (cage, yield, flow)

   **Use FFT when:**
   - Broadband frequency characterization needed
   - Comparing to literature (harmonic ratios I₃/I₁)
   - Material is thermorheologically simple

.. _smooth_derivative_transform:

5. Smooth Derivative
=====================

**Noise-Robust Differentiation of Experimental Data**

Theory
------

Numerical differentiation amplifies noise. For noisy rheological data (common in stress relaxation,
creep, startup), direct differentiation via ``np.gradient()`` produces unusable derivatives.

**Solution**: Apply smoothing **before** differentiation using:
1. **Savitzky-Golay filter**: Polynomial least-squares fit in sliding window
2. **Spline interpolation**: Smooth cubic/quintic spline, then differentiate analytically

**Savitzky-Golay**: Preserves peak shapes better than moving average. Order and window size
control smoothness vs accuracy trade-off.

Usage
-----

.. code-block:: python

   from rheojax.transforms import SmoothDerivative
   import numpy as np

   # Noisy relaxation modulus data
   t = np.linspace(0.01, 100, 500)
   G_t = 1000 * np.exp(-t/10.0) + 5*np.random.randn(500)  # Signal + noise

   # Direct differentiation (bad - amplifies noise)
   dG_dt_noisy = np.gradient(G_t, t)

   # Smooth differentiation with Savitzky-Golay
   sd_savgol = SmoothDerivative(
       method='savgol',
       window=11,  # Odd integer (larger = smoother, less responsive)
       order=3     # Polynomial order (2-5 typical)
   )
   dG_dt_smooth = sd_savgol.transform(t, G_t)

   # Alternative: Spline method for very noisy data
   sd_spline = SmoothDerivative(
       method='spline',
       smoothing=0.1  # Smoothing parameter (0 = interpolate, >0 = smooth)
   )
   dG_dt_spline = sd_spline.transform(t, G_t)

   # Plot comparison
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 5))
   plt.plot(t, dG_dt_noisy, 'r-', alpha=0.3, label='Direct (noisy)')
   plt.plot(t, dG_dt_smooth, 'b-', linewidth=2, label='Savitzky-Golay')
   plt.plot(t, dG_dt_spline, 'g--', label='Spline')
   plt.xlabel('Time (s)')
   plt.ylabel('dG/dt (Pa/s)')
   plt.legend()
   plt.show()

**Parameter Selection:**

.. list-table:: Savitzky-Golay Parameter Guide
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - Typical Range
     - Effect
   * - ``window``
     - 5-21 (odd)
     - Larger → smoother, less detail
   * - ``order``
     - 2-5
     - Higher → preserves peaks
   * - ``method``
     - 'savgol', 'spline'
     - savgol for peaks, spline for trends

.. code-block:: python

   # Application: Power-law exponent from flow curve
   gamma_dot = np.logspace(-2, 2, 100)
   stress = 10 * gamma_dot**0.5  # Power-law fluid: σ = K·γ̇^n

   # Differentiate log-log to find n
   sd = SmoothDerivative(method='savgol', window=11, order=3)
   d_log_stress = sd.transform(np.log10(gamma_dot), np.log10(stress))

   n_avg = np.mean(d_log_stress)  # Should be ≈ 0.5
   print(f"Power-law exponent n = {n_avg:.3f}")

.. _mutation_number_transform:

6. Mutation Number
===================

**Quantifying Time-Dependence in Viscoelastic Materials**

Theory
------

The **Mutation Number** (Rogers & Vlassopoulos, 2010) is a dimensionless parameter quantifying
how much a material "mutates" (changes structure) during a rheological measurement.

**Definition:**

.. math::

   \Delta = \frac{\int_0^{t_{\text{obs}}} G(t) \, dt}{G(0) \cdot t_{\text{obs}}}

For oscillatory tests:

.. math::

   \Delta_{\text{osc}} = \frac{1}{\omega \cdot \gamma_0} \int_0^{2\pi/\omega} |\dot{\gamma}(t)| \, dt

**Physical Interpretation:**
- :math:`\Delta \to 0`: Material remains **elastic** (solid-like)
- :math:`\Delta \to 1`: Material fully **relaxes** (liquid-like)
- :math:`\Delta \sim 0.1`: **Transition regime** (critical for accurate measurements)

**Practical Use**: Mutation number assesses **measurement validity**. If :math:`\Delta > 0.1`,
the material structure evolves significantly during the test, potentially invalidating linear
viscoelastic assumptions.

Usage
-----

.. code-block:: python

   from rheojax.transforms import MutationNumber
   from rheojax.core.data import RheoData
   import numpy as np

   # Relaxation modulus data
   t = np.linspace(0.01, 100, 500)
   G_t = 1000 * np.exp(-t/10.0)

   data = RheoData(x=t, y=G_t, test_mode='relaxation')

   # Calculate mutation number
   mutation = MutationNumber(integration_method='trapz')
   delta = mutation.transform(data)

   print(f"Mutation number Δ = {delta:.4f}")

   if delta < 0.01:
       print("Material is essentially elastic (no structural change)")
   elif delta < 0.1:
       print("Material shows time-dependence but measurement is valid")
   else:
       print("WARNING: Significant structural evolution during measurement!")

**Application: Assessing Linear Regime:**

.. code-block:: python

   # Oscillatory strain sweep (amplitude sweep)
   strains = np.logspace(-3, 0, 20)  # 0.1% to 100%
   omega = 1.0

   mutation_numbers = []

   for gamma_0 in strains:
       # Measure at each strain amplitude
       G_prime, G_double_prime = measure_saos(omega, gamma_0)

       # Calculate mutation number (simplified for SAOS)
       delta_osc = (gamma_0 * omega) / (2 * np.pi)  # Rough estimate
       mutation_numbers.append(delta_osc)

   # Find linear regime (Δ < 0.1)
   linear_strains = strains[np.array(mutation_numbers) < 0.1]
   print(f"Linear regime: γ₀ < {linear_strains[-1]:.3f}")

.. note::

   Mutation number is particularly useful for **soft glassy materials** (foams, gels, pastes)
   that age or evolve during measurement. For Newtonian liquids, :math:`\Delta \approx 1` always.

.. _owchirp_transform:

7. OWChirp (Optimally Windowed Chirp)
=======================================

**Rapid Frequency Sweeps via Chirp Oscillation**

Theory
------

**Optimally Windowed Chirp (OWChirp)** processes frequency-swept oscillatory data where the
oscillation frequency varies continuously (chirp signal) rather than stepping discretely.

**Advantages over discrete frequency sweeps:**
- **10-100× faster**: Single chirp vs 20-50 discrete frequencies
- **Better time resolution**: Captures transient behavior
- **Lower sample requirement**: One loading vs multiple frequency steps

**Challenge**: Extracting frequency-dependent moduli G'(ω), G''(ω) from time-domain chirp data
requires specialized **time-frequency analysis** (short-time Fourier transform, wavelets).

**Mathematical Formulation**:

.. math::

   \gamma(t) = \gamma_0 \sin\left( 2\pi \int_0^t f(\tau) \, d\tau \right)

   f(t) = f_{\text{start}} \left( \frac{f_{\text{end}}}{f_{\text{start}}} \right)^{t/T}

where :math:`f(t)` is the **instantaneous frequency** (Hz) varying logarithmically from
:math:`f_{\text{start}}` to :math:`f_{\text{end}}` over time :math:`T`.

Usage
-----

.. code-block:: python

   from rheojax.transforms import OWChirp
   from rheojax.core.data import RheoData
   import numpy as np

   # Chirp oscillatory data
   T_total = 300.0  # 5 minutes
   t = np.linspace(0, T_total, 10000)

   # Frequency sweep: 0.01 Hz to 100 Hz
   f_start = 0.01  # Hz
   f_end = 100.0   # Hz

   # Generate chirp strain (logarithmic sweep)
   phase = 2 * np.pi * f_start * T_total / np.log(f_end/f_start) * \
           (np.exp(t * np.log(f_end/f_start) / T_total) - 1)
   strain = 0.01 * np.sin(phase)  # 1% strain amplitude

   # Measured stress (complex response)
   stress = load_chirp_stress(t)

   # OWChirp analysis
   owc = OWChirp(
       f_start=f_start,
       f_end=f_end,
       window='hann',
       n_windows=50  # Frequency resolution
   )

   data = RheoData(
       x=t,
       y=stress,
       test_mode='oscillation',
       metadata={'strain': strain}
   )

   result = owc.transform(data)

   # Extract moduli vs frequency
   freq = result['frequency']  # Hz
   G_prime = result['G_prime']
   G_double_prime = result['G_double_prime']

   # Plot frequency-dependent moduli
   import matplotlib.pyplot as plt
   plt.loglog(freq, G_prime, 'o-', label="G'")
   plt.loglog(freq, G_double_prime, 's-', label="G''")
   plt.xlabel("Frequency (Hz)")
   plt.ylabel("G', G'' (Pa)")
   plt.legend()
   plt.show()

**When to Use OWChirp:**

.. list-table:: Chirp vs Discrete Frequency Sweep
   :header-rows: 1
   :widths: 40 30 30

   * - Criterion
     - Discrete Sweep
     - Chirp (OWChirp)
   * - **Speed**
     - 10-60 minutes
     - 1-5 minutes
   * - **Sample Requirement**
     - High (long test)
     - Low (short test)
   * - **Transient Materials**
     - Poor (ages during test)
     - Good (fast)
   * - **Frequency Resolution**
     - Excellent (arbitrary points)
     - Moderate (window size)
   * - **Data Processing**
     - Simple (direct)
     - Complex (requires OWChirp)

.. warning::

   **Nonlinearity**: Chirp data analysis assumes **linear viscoelasticity** (SAOS regime).
   For LAOS, use discrete frequency sweeps with FFT/SPP instead.

Combining Transforms
====================

**Multi-Transform Workflows for Comprehensive Analysis**

Example 1: Polymer Mastercurve with Quality Control
----------------------------------------------------

.. code-block:: python

   from rheojax.transforms import Mastercurve, MutationNumber
   from rheojax.core.data import RheoData

   # Load multi-temperature SAOS data
   datasets = load_multi_temp_saos([40, 60, 80, 100, 120])

   # Step 1: Check linearity at each temperature
   for i, data in enumerate(datasets):
       mutation = MutationNumber()
       delta = mutation.transform(data)
       print(f"T = {data.metadata['temperature']}°C: Δ = {delta:.4f}")

       if delta > 0.1:
           print(f"WARNING: Nonlinear behavior at T = {data.metadata['temperature']}°C")

   # Step 2: Generate mastercurve (TTS)
   mc = Mastercurve(reference_temp=80.0, method='wlf', auto_shift=True)
   master_curve, shift_factors = mc.transform(datasets)

   # Step 3: Fit model to mastercurve (not shown)
   # model.fit(master_curve.x, master_curve.y, test_mode='oscillation')

Example 2: LAOS Characterization with Yield Stress
---------------------------------------------------

.. code-block:: python

   from rheojax.transforms import FFTAnalysis, SPPDecomposer

   # LAOS time-domain data
   t, stress, strain = load_laos_data()
   omega = 1.0
   gamma_0 = 1.0

   # Step 1: FFT for nonlinearity quantification
   fft = FFTAnalysis(window='hann', detrend=True)
   data = RheoData(x=t, y=stress, test_mode='oscillation')
   fft_result = fft.transform(data)

   harmonics = fft_result['harmonics']
   I3_I1 = harmonics[3]['amplitude'] / harmonics[1]['amplitude']
   print(f"Nonlinearity ratio I₃/I₁ = {I3_I1:.4f}")

   # Step 2: SPP for physical parameters
   decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0)
   decomposer_result = decomposer.transform(data)
   spp_results = decomposer.get_results()

   print(f"Static yield stress: {spp_results['sigma_sy']:.1f} Pa")
   print(f"Cage modulus: {spp_results['G_cage']:.1f} Pa")

Example 3: SGR Material Classification via SRFS
------------------------------------------------

.. code-block:: python

   from rheojax.transforms import SRFS, detect_shear_banding
   from rheojax.models import SGRConventional

   # Multi-rate flow curves
   datasets = load_multi_rate_flow([0.01, 0.1, 1.0, 10.0, 100.0])

   # Step 1: SRFS mastercurve
   srfs = SRFS(reference_gamma_dot=1.0, auto_shift=True)
   master_flow, shifts = srfs.transform(datasets)

   # Extract SGR noise temperature from shift exponent
   exponent = srfs.get_shift_exponent()
   x_sgr = 2 - exponent
   print(f"SGR noise temperature x = {x_sgr:.3f}")

   # Step 2: Fit SGR model
   model = SGRConventional()
   model.fit(master_flow.x, master_flow.y, test_mode='flow_curve')

   x_fitted = model.parameters.get_value('x')
   print(f"SGR fitted x = {x_fitted:.3f} (compare to SRFS x = {x_sgr:.3f})")

   # Step 3: Check for shear banding
   banding = detect_shear_banding(datasets, threshold=0.05)
   if banding['is_banding']:
       print("Material shows shear banding (heterogeneous flow)")

Transform Limitations
=====================

**Understanding When Transforms Fail**

General Limitations
-------------------

1. **Garbage In, Garbage Out**: All transforms require clean, properly calibrated experimental data.
   Pre-process data to remove:

   - Instrument artifacts (inertia, compliance)
   - Temperature drift
   - Sample edge effects
   - Initial transients (use steady-state data only)

2. **Sampling Requirements**: Adequate data density needed:

   - FFT: Nyquist criterion (sample rate ≥ 2× highest frequency)
   - TTS/SRFS: At least 3-5 temperatures/rates for robust shifting
   - SPP: 5-10 steady-state cycles minimum

3. **Linearity Assumptions**: Several transforms assume linear viscoelasticity:

   - Mastercurve (TTS): Valid only in SAOS regime
   - Mutation Number: Assumes no permanent structural change
   - OWChirp: Small-amplitude assumption

Transform-Specific Issues
--------------------------

.. list-table:: Common Transform Pitfalls
   :header-rows: 1
   :widths: 25 35 40

   * - Transform
     - Common Failure Mode
     - Solution
   * - **FFT**
     - Spectral leakage (non-integer cycles)
     - Use windowing + integer cycles
   * - **Mastercurve**
     - Non-thermorheologically simple material
     - Check WLF residuals, test alternative models
   * - **SRFS**
     - Thixotropic transients
     - Ensure steady state before measurement
   * - **SPP**
     - Noisy strain rate differentiation
     - Increase smoothing (step_size)
   * - **Smooth Derivative**
     - Over-smoothing (loses features)
     - Reduce window size or smoothing parameter
   * - **Mutation Number**
     - Short observation time
     - Extrapolate to infinite time
   * - **OWChirp**
     - Nonlinear response
     - Use smaller strain amplitude

**FFT Artifacts:**

.. code-block:: python

   # BAD: Non-integer cycles cause spectral leakage
   t = np.linspace(0, 9.5*2*np.pi, 1000)  # 9.5 cycles

   # GOOD: Exactly 10 cycles
   t = np.linspace(0, 10*2*np.pi, 1000)  # 10 cycles

**Mastercurve Validation:**

.. code-block:: python

   # Check if material is thermorheologically simple
   mc = Mastercurve(reference_temp=80.0, auto_shift=True)
   master_curve, shift_factors = mc.transform(datasets)

   # Compute shift quality metric (residuals)
   quality = mc.get_shift_quality()
   if quality['mean_squared_error'] > 0.1:
       print("WARNING: Poor mastercurve quality. Material may not be thermorheologically simple.")

Troubleshooting Guide
---------------------

**Issue: FFT shows unexpected harmonics**

- Check for integer number of cycles
- Verify steady-state (discard initial cycles)
- Apply appropriate window function (try 'blackman')

**Issue: Mastercurve has discontinuities**

- Ensure consistent units (Pa vs kPa)
- Verify temperature metadata is correct
- Check for phase transitions (crystallization)
- Try manual WLF parameters

**Issue: SPP yields unrealistic values (G_cage < 0)**

- Increase smoothing (step_size=16 instead of 8)
- Check strain rate data quality
- Verify steady-state oscillation

**Issue: SRFS shift factors don't follow power law**

- Material may not be SGR-type
- Check for thixotropic effects (aging)
- Verify truly steady-state measurements

References
==========

**Foundational Papers:**

1. **FFT & LAOS**: Hyun et al. (2002). "A review of nonlinear oscillatory shear tests." *J. Non-Newt. Fluid Mech.* 107, 51-65.

2. **Time-Temperature Superposition**: Ferry, J.D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed. Wiley.

3. **SPP Framework**: Rogers, S.A. et al. (2011). "Sequence of physical processes determined by the medium amplitude oscillatory shear." *J. Rheol.* 55, 435-458.

4. **Mutation Number**: Rogers, S.A. & Vlassopoulos, D. (2010). "Frieze group analysis of asymmetric response to large amplitude oscillatory shear." *J. Rheol.* 54, 859-880.

5. **SRFS & SGR**: Divoux, T. et al. (2016). "Stress overshoot in a simple yield stress fluid." *Soft Matter* 12, 5249-5262.

6. **OWChirp**: Ghiringhelli, L.M. et al. (2012). "Optimal time resolved rheometry." *Meas. Sci. Technol.* 23, 065401.

7. **Soft Glassy Rheology**: Sollich, P. et al. (1997). "Rheology of soft glassy materials." *Phys. Rev. Lett.* 78, 2020-2023.

See Also
========

**Related Documentation:**

- :doc:`time_temperature_superposition` — In-depth TTS guide with WLF/Arrhenius theory
- :doc:`sgr_analysis` — Soft Glassy Rheology framework and SRFS applications
- :doc:`spp_analysis` — Detailed SPP tutorial with physical interpretation
- :doc:`../02_model_usage/model_selection` — Choosing models for transformed data
- :doc:`../04_practical_guides/visualization` — Plotting transform results

**Transform API Reference:**

- :doc:`../../api_reference/transforms/fft_analysis` — FFTAnalysis class
- :doc:`../../api_reference/transforms/mastercurve` — Mastercurve class
- :doc:`../../api_reference/transforms/srfs` — SRFS class
- :doc:`../../api_reference/transforms/spp_decomposer` — SPPDecomposer class
- :doc:`../../api_reference/transforms/smooth_derivative` — SmoothDerivative class
- :doc:`../../api_reference/transforms/mutation_number` — MutationNumber class
- :doc:`../../api_reference/transforms/owchirp` — OWChirp class

**Example Notebooks:**

- ``examples/advanced/02-multi-technique-fitting.ipynb`` — Combining transforms with fitting
- ``examples/advanced/09-sgr-soft-glassy-rheology.ipynb`` — SRFS + SGR workflow
- ``examples/advanced/10-spp-laos-tutorial.ipynb`` — SPP detailed examples
- ``examples/basic/01-maxwell-fitting.ipynb`` — Simple transform usage

Next Steps
==========

After mastering data transforms, you can:

1. **Apply transforms in production workflows** — :doc:`../04_practical_guides/pipeline_api`
2. **Fit models to transformed data** — :doc:`../02_model_usage/fitting_strategies`
3. **Perform Bayesian inference** — :doc:`bayesian_inference`
4. **Analyze soft glassy materials** — :doc:`sgr_analysis`
5. **Extract yield stresses** — :doc:`spp_analysis`

**Practical Exercise:**

Try the complete workflow:

.. code-block:: python

   from rheojax.transforms import Mastercurve, MutationNumber
   from rheojax.models import FractionalMaxwellLiquid
   from rheojax.pipeline import Pipeline

   # Load multi-temperature data
   pipeline = Pipeline()
   pipeline.load_multi_temperature('data_*.csv', temps=[40, 60, 80, 100])

   # Quality check
   for data in pipeline.datasets:
       mu = MutationNumber().transform(data)
       print(f"T = {data.metadata['temperature']}°C: Δ = {mu:.4f}")

   # Generate mastercurve
   mc = Mastercurve(reference_temp=80.0, method='wlf', auto_shift=True)
   master, shifts = mc.transform(pipeline.datasets)

   # Fit fractional model
   model = FractionalMaxwellLiquid()
   model.fit(master.x, master.y, test_mode='oscillation')

   # Bayesian inference
   result = model.fit_bayesian(master.x, master.y, num_samples=2000)

   print("Analysis complete!")
