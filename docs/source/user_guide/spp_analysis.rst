.. _spp_analysis:

====================================
Sequence of Physical Processes (SPP)
====================================

**Time-Domain LAOS Analysis Without Fourier Transform Limitations**

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview
========

The **Sequence of Physical Processes (SPP)** framework, introduced by Rogers (2012), provides a
fundamentally different approach to Large Amplitude Oscillatory Shear (LAOS) analysis. Unlike
Fourier/Chebyshev decomposition methods, SPP operates entirely in the time domain, directly
extracting transient elastic modulus and dynamic viscosity without the need for harmonic
decomposition.

.. admonition:: Key Insight
   :class: tip

   SPP reveals the **instantaneous material physics** at each point in the oscillatory cycle,
   making it particularly powerful for understanding yield-stress behavior, thixotropy, and
   other transient nonlinear phenomena.

When to Use SPP
---------------

**SPP is ideal for:**

- **Yield-stress LAOS**: Directly identifies yielding transitions within each cycle
- **Weak thixotropy**: Tracks structural evolution without assuming steady-state
- **Phase transitions**: Detects solid-liquid transitions during oscillation
- **Physical interpretation**: Maps response to physical processes (cage modulus, flow)

**SPP complements (rather than replaces) Fourier methods for:**

- Broadband frequency characterization
- Standard nonlinear parameters (e\ :sub:`3`/e\ :sub:`1`, v\ :sub:`3`/v\ :sub:`1`)
- Comparison with literature using harmonic ratios

Fundamentals
============

The SPP Framework
-----------------

Traditional LAOS analysis decomposes stress into Fourier harmonics:

.. math::

   \sigma(t) = \sum_{n=\text{odd}} \left[ G'_n \sin(n\omega t) + G''_n \cos(n\omega t) \right]

SPP instead defines **instantaneous** material properties that evolve through the cycle:

.. math::

   G'_t(t) = \frac{\sigma}{\gamma} \bigg|_{\dot{\gamma}=0}, \qquad
   \eta'_t(t) = \frac{\sigma}{\dot{\gamma}} \bigg|_{\gamma=0}

These are evaluated at specific phase points in each cycle, yielding time-resolved material
functions that capture transient physics.

Cage Modulus
------------

The **cage modulus** G\ :sub:`cage` represents the instantaneous elastic stiffness when strain
rate is zero (maximum strain):

.. math::

   G_{\text{cage}}(t) = \frac{d\sigma}{d\gamma}\bigg|_{\dot{\gamma}=0}

This captures the elastic "cage" formed by microstructure (e.g., colloidal particles, polymer
entanglements) at maximum deformation. The cage modulus:

- **Softening**: G\ :sub:`cage` decreases as structure breaks down
- **Stiffening**: G\ :sub:`cage` increases with strain (nonlinear elasticity)
- **Yielding**: Sudden drop in G\ :sub:`cage` indicates cage rupture

Static Yield Stress
-------------------

The **static yield stress** σ\ :sub:`y,static` is extracted at the strain reversal point
(γ = γ\ :sub:`0`, γ̇ = 0):

.. math::

   \sigma_{y,\text{static}} = \sigma(t)\big|_{\gamma=\pm\gamma_0}

This represents the maximum stress the material sustains before yielding—directly comparable
to the yield stress from steady shear or stress growth tests.

Dynamic Yield Stress
--------------------

The **dynamic yield stress** σ\ :sub:`y,dynamic` is obtained at maximum strain rate
(γ = 0, γ̇ = γ̇\ :sub:`0`):

.. math::

   \sigma_{y,\text{dynamic}} = \sigma(t)\big|_{\dot{\gamma}=\pm\dot{\gamma}_0}

This measures the stress at the point of maximum flow, capturing the dynamic resistance during
the flowing portion of the cycle.

Phase Without Fourier
---------------------

SPP defines a **time-domain phase** δ\ :sub:`t` without Fourier decomposition:

.. math::

   \delta_t(t) = \arctan\left( \frac{G''_t(t)}{G'_t(t)} \right)

This phase angle evolves continuously through the cycle:

- δ\ :sub:`t` → 0°: Purely elastic response (solid-like)
- δ\ :sub:`t` → 90°: Purely viscous response (liquid-like)
- Transition regions reveal yielding and recovery

Power-Law Flow Analysis
-----------------------

For yielded materials exhibiting power-law flow, SPP extracts:

.. math::

   \sigma = K |\dot{\gamma}|^n \text{sign}(\dot{\gamma})

where K is the consistency and n is the flow index. This is evaluated in the flowing
portions of each cycle where γ ≈ 0 (maximum strain rate).

Practical Implementation
========================

Cycle Selection
---------------

SPP analysis requires selecting appropriate cycles from the LAOS waveform:

.. code-block:: python

   from rheojax.transforms import SPPDecomposer

   # Apply SPP decomposition to time-domain stress data
   spp = SPPDecomposer(
       omega=1.0,          # Angular frequency (rad/s)
       gamma_0=1.0,        # Strain amplitude
       n_harmonics=5,      # Number of harmonics for analysis
   )

   result = spp.transform(rheo_data)  # RheoData with time-domain stress

For multi-cycle LAOS data, you can select specific cycles to analyze, skipping startup
transients or avoiding late-cycle fatigue effects:

.. code-block:: python

   # Skip first 2 cycles (startup transients), analyze cycles 2-5
   spp = SPPDecomposer(
       omega=1.0,
       gamma_0=1.0,
       start_cycle=2,      # Skip cycles 0, 1 (0-indexed)
       end_cycle=5,        # Analyze up to cycle 5 (exclusive)
   )

   result = spp.transform(rheo_data)

   # Check which cycles were actually analyzed
   print(spp.results_['cycles_analyzed'])  # (2, 5)

.. warning::

   **Cycle selection matters!** Early cycles may contain startup transients, while late
   cycles may show fatigue or structural breakdown. Always visualize multiple cycles
   before selecting the analysis window.

Numerical Differentiation Method
--------------------------------

For raw experimental data compatibility (matching MATLAB SPPplus workflows), use the
numerical differentiation method:

.. code-block:: python

   # MATLAB-compatible numerical SPP analysis
   spp = SPPDecomposer(
       omega=1.0,
       gamma_0=1.0,
       use_numerical_method=True,  # Enable numerical derivatives
       step_size=1,                # Finite-difference step (larger = smoother)
   )

   result = spp.transform(rheo_data)

   # Access instantaneous moduli from numerical method
   numerical = spp.results_['numerical']
   Gp_t = numerical['Gp_t']       # G'(t) - instantaneous storage modulus
   Gpp_t = numerical['Gpp_t']     # G''(t) - instantaneous loss modulus
   delta_t = numerical['delta_t'] # δ(t) - instantaneous phase angle

   # Mean values for comparison with linear measurements
   print(f"Mean G'(t) = {spp.results_['Gp_t_mean']:.1f} Pa")
   print(f"Mean G''(t) = {spp.results_['Gpp_t_mean']:.1f} Pa")

The numerical method implements the cross-product formula from Rogers (2012):

.. math::

   G'_t = -\frac{(\vec{r}' \times \vec{r}'')_0}{(\vec{r}' \times \vec{r}'')_2}, \qquad
   G''_t = -\frac{(\vec{r}' \times \vec{r}'')_1}{(\vec{r}' \times \vec{r}'')_2}

where :math:`\vec{r} = [\gamma, \dot{\gamma}/\omega, \sigma]` is the response trajectory.

Extracting SPP Parameters
-------------------------

.. code-block:: python

   # Extract SPP quantities
   G_cage = result.cage_modulus           # Array over cycles
   sigma_y_static = result.static_yield   # Static yield stress
   sigma_y_dynamic = result.dynamic_yield # Dynamic yield stress
   delta_t = result.phase_angle           # Time-domain phase

   # Power-law parameters (if applicable)
   K = result.consistency    # Pa·s^n
   n = result.flow_index     # Dimensionless

Visualizing SPP Results
-----------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(2, 2, figsize=(12, 10))

   # Lissajous with SPP annotations
   ax = axes[0, 0]
   ax.plot(strain, stress, 'b-', linewidth=1.5)
   ax.axhline(sigma_y_static, color='r', linestyle='--', label='σ_y,static')
   ax.axhline(sigma_y_dynamic, color='g', linestyle=':', label='σ_y,dynamic')
   ax.set_xlabel('Strain γ')
   ax.set_ylabel('Stress σ (Pa)')
   ax.legend()

   # Cage modulus evolution
   ax = axes[0, 1]
   ax.plot(cycle_numbers, G_cage, 'ko-')
   ax.set_xlabel('Cycle Number')
   ax.set_ylabel('Cage Modulus G_cage (Pa)')

   # Phase angle through cycle
   ax = axes[1, 0]
   ax.plot(phase_in_cycle, delta_t, 'b-')
   ax.axhline(45, color='gray', linestyle='--', alpha=0.5)
   ax.set_xlabel('Phase in Cycle (°)')
   ax.set_ylabel('δ_t (°)')

   plt.tight_layout()

Scope and Applications
======================

Yield-Stress LAOS
-----------------

SPP excels at characterizing yield-stress materials under LAOS:

1. **Pre-yield**: High G\ :sub:`cage`, low δ\ :sub:`t` (elastic solid)
2. **Yielding**: Rapid drop in G\ :sub:`cage`, increasing δ\ :sub:`t`
3. **Post-yield**: Power-law flow, δ\ :sub:`t` → 90°
4. **Recovery**: G\ :sub:`cage` rebuilds during strain reversal

This provides direct insight into yielding dynamics inaccessible to Fourier methods.

Weak Thixotropy
---------------

For weakly thixotropic materials (structure recovers partially between cycles):

- G\ :sub:`cage` shows cycle-to-cycle evolution
- Static yield stress may drift over many cycles
- SPP tracks this evolution without assuming steady-state

.. note::

   **Scope limitation**: SPP is best for *weak* thixotropy where structure partially
   equilibrates within each cycle. For *strong* thixotropy (structure changes dramatically
   over cycles), time-resolved rheology or stepped protocols may be more appropriate.

Phase Transitions
-----------------

SPP can detect phase transitions within the oscillatory cycle:

- **Solid → Liquid**: δ\ :sub:`t` increases sharply at yield
- **Liquid → Solid**: δ\ :sub:`t` decreases during recovery/gelation
- **Multiple transitions**: Complex materials may show re-entrant behavior

Limitations and Caveats
=======================

Noise Sensitivity Near Zero Stress
----------------------------------

SPP quantities involve ratios that become sensitive near zero:

.. math::

   G'_t = \frac{\sigma}{\gamma} \quad \text{(problematic when } \sigma \approx 0\text{)}

.. warning::

   **Near-zero stress**: When stress approaches zero (during flow reversal in some
   materials), SPP ratios can become noisy or undefined. Apply smoothing or exclude
   these regions from analysis.

**Mitigation strategies:**

.. code-block:: python

   from rheojax.transforms import SPPDecomposer

   # Option 1: Apply stress threshold (filter data before analysis)
   valid_mask = np.abs(stress) > 1.0  # Pa threshold
   filtered_stress = stress[valid_mask]

   # Option 2: Use higher tolerance for yield point detection
   spp = SPPDecomposer(omega=1.0, gamma_0=1.0, yield_tolerance=0.05)

   # Option 3: Apply smoothing to raw data before SPP
   from scipy.signal import savgol_filter
   stress_smooth = savgol_filter(stress, window_length=11, polyorder=3)

Cycle Selection Effects
-----------------------

The choice of which cycles to analyze significantly affects SPP results:

- **Too early**: Transient effects, not representative of steady oscillation
- **Too late**: Fatigue, structural breakdown, edge fracture
- **Single cycle**: May not be representative; average multiple cycles

**Best practice**: Analyze a range of cycles and verify consistency.

Harmonic Truncation Not Applicable
----------------------------------

Unlike Fourier methods, SPP does not truncate harmonics—it uses the raw stress-strain data.
This means:

- **Advantage**: No artifacts from harmonic truncation
- **Caution**: Noise propagates directly into SPP quantities

For noisy data, consider smoothing the raw waveforms before SPP analysis.

Thixotropy Caveats
------------------

SPP assumes that properties can be meaningfully defined at each instant within a cycle.
For strongly thixotropic materials:

- Structure changes faster than the oscillation period
- Instantaneous properties may not be well-defined
- Consider time-resolved or protocol-based methods instead

**Rule of thumb**: SPP works well when the structural relaxation time τ\ :sub:`s` is longer
than the oscillation period (τ\ :sub:`s` > 2π/ω).

macOS GPU Acceleration
----------------------

.. note::

   **Platform limitation**: GPU acceleration for SPP transforms is currently only supported
   on Linux with CUDA. macOS users will use CPU computation, which remains performant for
   typical LAOS datasets but may be slower for very large datasets.

Comparison: SPP vs Fourier/Chebyshev
====================================

Understanding when to use SPP versus traditional Fourier-based methods is crucial for
effective LAOS analysis.

.. list-table:: SPP vs Fourier/Chebyshev Comparison
   :header-rows: 1
   :widths: 25 37 38

   * - Aspect
     - SPP (Time-Domain)
     - Fourier/Chebyshev (Frequency-Domain)
   * - **Domain**
     - Time (instantaneous properties)
     - Frequency (harmonic coefficients)
   * - **Yield stress**
     - Direct extraction (σ\ :sub:`y,static`, σ\ :sub:`y,dynamic`)
     - Indirect (from G\ :sub:`M`, G\ :sub:`L` at large strain)
   * - **Physical interpretation**
     - Intuitive (cage, flow, yielding)
     - Mathematical (harmonic ratios)
   * - **Transient phenomena**
     - Captures within-cycle evolution
     - Averages over cycle
   * - **Noise sensitivity**
     - Higher near zero stress
     - Lower (averaging effect)
   * - **Standard metrics**
     - Not directly comparable to e\ :sub:`3`/e\ :sub:`1`
     - Industry standard metrics
   * - **Truncation artifacts**
     - None (raw data)
     - Possible (finite harmonics)
   * - **Best for**
     - Yield-stress fluids, physical mechanisms
     - Material fingerprinting, literature comparison
   * - **Computational cost**
     - O(N) per cycle
     - O(N log N) via FFT

Recommended Workflow
--------------------

For comprehensive LAOS characterization:

1. **Start with Fourier/Chebyshev**: Get standard nonlinear parameters (I\ :sub:`3/1`, e\ :sub:`3`/e\ :sub:`1`, v\ :sub:`3`/v\ :sub:`1`) for literature comparison

2. **Apply SPP for physical insight**: Extract yield stresses, cage modulus, and phase evolution

3. **Cross-validate**: Ensure both methods tell a consistent physical story

.. code-block:: python

   from rheojax.transforms import OWChirp, SPPDecomposer

   # Fourier-based analysis
   owchirp = OWChirp(extract_harmonics=True)
   harmonics = owchirp.get_harmonics(laos_data)
   I31 = harmonics['third'][1] / harmonics['fundamental'][1]

   # SPP analysis
   spp = SPPDecomposer(omega=1.0, gamma_0=1.0, n_harmonics=5)
   result = spp.transform(rheo_data)
   spp_metrics = spp.get_results()

   # Compare interpretations
   print(f"Fourier I_3/1 = {I31:.3f}")
   print(f"SPP I3/I1 ratio = {spp_metrics['I3_I1_ratio']:.3f}")
   print(f"SPP static yield = {spp_metrics['sigma_sy']:.0f} Pa")

References
==========

**Foundational SPP Papers:**

- Rogers, S. A. (2012). "A sequence of physical processes determined and quantified in
  LAOS: An instantaneous local 2D/3D approach." *J. Rheol.* 56, 1129-1151.
  https://doi.org/10.1122/1.4726083

- Rogers, S. A., & Lettinga, M. P. (2012). "A sequence of physical processes determined
  and quantified in large-amplitude oscillatory shear (LAOS): Application to theoretical
  nonlinear models." *J. Rheol.* 56, 1-25. https://doi.org/10.1122/1.3662962

**Applications and Extensions:**

- Lee, C.-W., & Rogers, S. A. (2017). "A sequence of physical processes quantified in LAOS
  by continuous local measures." *Korea-Australia Rheol. J.* 29, 269-279.

- Donley, G. J., et al. (2019). "Time-resolved dynamics of the yielding transition in
  soft materials." *J. Non-Newtonian Fluid Mech.* 264, 117-134.

**Comparison with Fourier Methods:**

- Hyun, K., et al. (2011). "A review of nonlinear oscillatory shear tests: Analysis and
  application of large amplitude oscillatory shear (LAOS)." *Prog. Polym. Sci.* 36,
  1697-1753. https://doi.org/10.1016/j.progpolymsci.2011.02.002

See Also
========

- :doc:`/transforms/owchirp` — OWChirp for Fourier-based LAOS analysis
- :doc:`/models/flow/herschel_bulkley` — Yield-stress model for flow curve fitting
- :doc:`/user_guide/03_advanced_topics/bayesian_inference` — Uncertainty quantification for SPP parameters
- :doc:`/examples/index` — Tutorial notebooks including SPP Bayesian workflow
