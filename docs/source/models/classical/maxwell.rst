.. _model-maxwell:

Maxwell (Classical)
===================

Quick Reference
---------------

- **Use when:** Single relaxation time, exponential stress decay, viscoelastic liquids
- **Parameters:** 2 (G, :math:`\eta`)
- **Key equation:** :math:`G(t) = G \exp(-t/\tau)` where :math:`\tau = \eta/G`
- **Test modes:** Oscillation, relaxation
- **Material examples:** Polymer melts (PS, PDMS), viscoelastic liquids, dilute solutions

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`G`
     - Spring modulus (Pa). Controls instantaneous elasticity.
   * - :math:`\eta`
     - Dashpot viscosity (Pa·s). Controls energy dissipation.
   * - :math:`\tau`
     - Relaxation time (s), :math:`\tau = \eta/G`.

Overview
--------

Two-parameter linear viscoelastic model with a spring (:math:`G`) and dashpot (:math:`\eta`) in series.
Captures single-time-constant stress relaxation with :math:`G(t) = G\,e^{-t/\tau}` where :math:`\tau = \eta/G`.

Physical Foundations
--------------------

Mechanical Analogue
~~~~~~~~~~~~~~~~~~~

The Maxwell model consists of a linear spring (Hookean elastic element) connected in **series** with a Newtonian dashpot (viscous element):

.. code-block:: text

   ┌────────┐         ┌────────┐
   │ Spring │─────────│Dashpot │
   │   G    │         │   η    │
   └────────┘         └────────┘

   Total deformation: γ_total = γ_spring + γ_dashpot
   Same stress: σ_spring = σ_dashpot = σ

The series configuration means:

- **Strain is additive**: :math:`\gamma(t) = \gamma_{\text{spring}}(t) + \gamma_{\text{dashpot}}(t)`
- **Stress is identical**: Both elements experience the same stress :math:`\sigma(t)`

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Maxwell model represents materials where:

**Spring (elastic storage)**:
   - Entropic elasticity from chain stretching in polymer melts
   - Temporary network junctions that store elastic energy
   - Reversible conformational changes

**Dashpot (viscous dissipation)**:
   - Chain reptation through entanglement network
   - Molecular rearrangements that dissipate energy
   - Irreversible flow at long timescales

**Physical meaning of relaxation time** :math:`\tau = \eta/G`:
   - Characteristic time for stress to decay to :math:`1/e \approx 37\%` of initial value
   - Ratio of viscous resistance to elastic restoring force
   - Related to molecular weight and entanglement density in polymers

Material Examples with Typical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative Maxwell parameters
   :header-rows: 1
   :widths: 30 20 20 20 10

   * - Material
     - G (Pa)
     - :math:`\eta` (Pa·s)
     - :math:`\tau` (s)
     - Ref
   * - Polystyrene melt (170°C)
     - :math:`1 \times 10^5`
     - :math:`1 \times 10^6`
     - 10
     - [1]
   * - Polyethylene melt (190°C)
     - :math:`3 \times 10^4`
     - :math:`3 \times 10^5`
     - 10
     - [1]
   * - Bitumen (25°C)
     - :math:`1 \times 10^6`
     - :math:`1 \times 10^8`
     - 100
     - [2]
   * - Dilute polymer solution
     - :math:`1 \times 10^2`
     - :math:`1 \times 10^1`
     - 0.1
     - [3]
   * - PDMS (crosslinked)
     - :math:`5 \times 10^5`
     - :math:`5 \times 10^4`
     - 0.1
     - [4]

Connection to Polymer Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **entangled polymer melts**, the Maxwell relaxation time relates to molecular parameters via the **Rouse-reptation theory** (Doi-Edwards):

.. math::

   \tau_d \sim \frac{M^3}{\rho RT} \cdot \frac{\zeta N_e^2}{M_e}

where:
   - :math:`M` = molecular weight
   - :math:`M_e` = entanglement molecular weight
   - :math:`\zeta` = monomeric friction coefficient
   - :math:`N_e` = entanglement strand length

**Scaling laws** (experimental):
   - :math:`\eta_0 \sim M^{3.4}` for :math:`M > M_c` (entangled)
   - :math:`\eta_0 \sim M^{1.0}` for :math:`M < M_c` (Rouse regime)
   - :math:`G_N^0 \sim \rho RT / M_e` (plateau modulus)

Governing Equations
-------------------

Mathematical Derivation
~~~~~~~~~~~~~~~~~~~~~~~

Starting from the mechanical analogue with **series connection**:

**Step 1**: Express strain rates
   Spring obeys Hooke's law: :math:`\sigma = G \gamma_{\text{spring}}`

   Dashpot obeys Newton's law: :math:`\sigma = \eta \dot{\gamma}_{\text{dashpot}}`

**Step 2**: Differentiate spring equation
   :math:`\dot{\sigma} = G \dot{\gamma}_{\text{spring}}`

**Step 3**: Substitute dashpot relation
   :math:`\dot{\gamma}_{\text{dashpot}} = \sigma / \eta`

**Step 4**: Total strain rate (series)
   :math:`\dot{\gamma} = \dot{\gamma}_{\text{spring}} + \dot{\gamma}_{\text{dashpot}} = \frac{\dot{\sigma}}{G} + \frac{\sigma}{\eta}`

**Step 5**: Rearrange to constitutive form
   :math:`\sigma + \frac{\eta}{G} \dot{\sigma} = \eta \dot{\gamma}`

   Defining :math:`\tau = \eta/G`:

   .. math::

      \tau\,\dot{\sigma}(t) + \sigma(t) = \eta\,\dot{\gamma}(t)

Constitutive (differential) form:

.. math::

   \tau\,\dot{\sigma}(t) + \sigma(t) = \eta\,\dot{\gamma}(t), \qquad \tau = \frac{\eta}{G}

Relaxation modulus:

.. math::

   G(t) = G\,e^{-t/\tau}

**Derivation**: For stress relaxation (step strain :math:`\gamma_0` at :math:`t=0`), the governing ODE with :math:`\dot{\gamma}=0` for :math:`t>0` gives :math:`\dot{\sigma} = -\sigma/\tau`, yielding :math:`\sigma(t) = \sigma_0 e^{-t/\tau} = G\gamma_0 e^{-t/\tau}`.

Fourier Transform to Frequency Domain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For oscillatory shear :math:`\gamma(t) = \gamma_0 e^{i\omega t}`, apply Fourier transform:

**Step 1**: Substitute into constitutive equation
   :math:`\sigma(t) = \sigma_0 e^{i\omega t}` (harmonic response)

   :math:`\dot{\sigma} = i\omega \sigma`, :math:`\dot{\gamma} = i\omega \gamma`

**Step 2**: Transform constitutive equation
   :math:`\tau (i\omega \sigma) + \sigma = \eta (i\omega \gamma)`

   :math:`\sigma (1 + i\omega\tau) = i\omega\eta \gamma`

**Step 3**: Define complex modulus
   .. math::

      G^*(\omega) = \frac{\sigma}{\gamma} = \frac{i\omega\eta}{1 + i\omega\tau} = G \frac{i\omega\tau}{1 + i\omega\tau}

**Step 4**: Separate real and imaginary parts

Oscillatory complex modulus:

.. math::

   G^*(\omega) = G'(\omega) + i\,G''(\omega)
   = G \frac{i\,\omega \tau}{1 + i\,\omega \tau}

with storage and loss moduli:

.. math::

   G'(\omega) = G\,\frac{(\omega \tau)^2}{1 + (\omega \tau)^2}, \qquad
   G''(\omega) = G\,\frac{\omega \tau}{1 + (\omega \tau)^2}

Mathematical Significance
~~~~~~~~~~~~~~~~~~~~~~~~~

**First-order linear ODE**: The Maxwell constitutive equation is the simplest differential equation describing viscoelasticity. The exponential solution is characteristic of all first-order linear systems.

**Complex modulus structure**: The factor :math:`i\omega\tau/(1+i\omega\tau)` is a **high-pass filter** in signal processing, transitioning from viscous (low :math:`\omega`) to elastic (high :math:`\omega`) behavior.

**Loss tangent**:

.. math::

   \tan\delta = \frac{G''}{G'} = \frac{1}{\omega\tau}

This **monotonically decreases** with frequency, confirming liquid-like character (no solid plateau).

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 12 12 18 40

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``G``
     - :math:`G`
     - Pa
     - :math:`G > 0`
     - Spring modulus
   * - ``eta``
     - :math:`\eta`
     - Pa*s
     - :math:`\eta > 0`
     - Dashpot viscosity
   * - (derived)
     - :math:`\tau`
     - s
     - :math:`\tau > 0`
     - :math:`\tau = \eta/G` (not an independent parameter)

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G (Spring Modulus)**:
   - **Physical meaning**: Instantaneous elastic response at short times (or high frequencies)
   - **Molecular origin**: Entropic resistance to chain deformation
   - **Typical ranges**:
      - Polymer melts: :math:`10^4 - 10^6` Pa
      - Dilute solutions: :math:`10^1 - 10^3` Pa
      - Gels: :math:`10^2 - 10^5` Pa
   - **Scaling**: :math:`G \sim \rho RT / M` (low MW), :math:`G \sim \rho RT / M_e` (entangled)

:math:`\eta` **(Dashpot Viscosity)**:
   - **Physical meaning**: Resistance to flow, energy dissipation rate
   - **Molecular origin**: Chain friction during reptation or Rouse relaxation
   - **Typical ranges**:
      - Polymer melts: :math:`10^3 - 10^7` Pa·s (strongly temperature-dependent)
      - Bitumen: :math:`10^6 - 10^{10}` Pa·s
      - Dilute solutions: :math:`10^{-2} - 10^2` Pa·s
   - **Scaling**: :math:`\eta \sim M^{3.4}` (entangled polymers), :math:`\eta \sim M` (Rouse)

:math:`\tau` **(Relaxation Time)**:
   - **Physical meaning**: Timescale separating elastic (solid-like) from viscous (liquid-like) behavior
   - **Diagnostic**: :math:`\tau^{-1}` corresponds to frequency where :math:`G''(\omega)` peaks
   - **Material design**: Long :math:`\tau` → more elastic character; short :math:`\tau` → more viscous
   - **Typical ranges**: :math:`10^{-3}` s (dilute solutions) to :math:`10^3` s (bitumen at room T)

Relation to Molecular Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **linear polymer melts**:

.. math::

   \eta_0 = \frac{\pi^2}{12} \rho \frac{RT}{M_e} \tau_e \left(\frac{M}{M_e}\right)^3

where:
   - :math:`M_e` ≈ 1800 g/mol (polyethylene), 13000 g/mol (polystyrene)
   - :math:`\tau_e` = Rouse time of entanglement strand
   - :math:`\rho` = density

This connects measurable rheological parameters to fundamental molecular architecture.

Validity and Assumptions
------------------------

- Linear viscoelasticity: yes
- Small amplitude: yes
- Isothermal: yes
- Data/test modes: relaxation, oscillation
- Additional assumptions: single relaxation time

Limitations
~~~~~~~~~~~

**Critical limitation: Predicts unbounded creep**

For creep compliance :math:`J(t) = \gamma(t)/\sigma_0` under constant stress :math:`\sigma_0`:

.. math::

   J(t) = \frac{1}{G} + \frac{t}{\eta} \quad \to \infty \text{ as } t \to \infty

The Maxwell model predicts **linear viscous flow** with no creep recovery, making it:
   - **Inappropriate** for viscoelastic solids (rubbers, gels)
   - **Appropriate** for viscoelastic liquids (polymer melts, dilute solutions)

**Single relaxation time**:
   Real polymers exhibit **continuous distributions** of relaxation times :math:`H(\tau)`. The Maxwell model is adequate only when:
   - Material is nearly monodisperse
   - One relaxation process dominates experimental window
   - Data span < 2 decades in time/frequency

**No equilibrium modulus**:
   :math:`G_e = \lim_{t\to\infty} G(t) = 0`, meaning the material **always flows** eventually. This fails for crosslinked networks.

Regimes and Behavior
--------------------

Limiting Cases
~~~~~~~~~~~~~~

**Low frequency (** :math:`\omega` **→ 0, terminal region)**:

.. math::

   G'(\omega) \approx G (\omega\tau)^2 \sim \omega^2

   G''(\omega) \approx G \omega\tau \sim \omega

**Interpretation**: Viscous liquid-like behavior dominates. Energy dissipation (:math:`G''`) exceeds storage (:math:`G'`).

**High frequency (** :math:`\omega` **→ ∞, glassy region)**:

.. math::

   G'(\omega) \to G \quad (\text{plateau})

   G''(\omega) \to 0

**Interpretation**: Elastic solid-like response. Chains don't have time to relax, behaving as frozen network.

**Crossover frequency** :math:`\omega_c = 1/\tau`:

.. math::

   G'(\omega_c) = G''(\omega_c) = \frac{G}{2}

At this point, elastic and viscous contributions are equal, defining the characteristic relaxation timescale.

Asymptotic Behavior Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Frequency-dependent regimes
   :header-rows: 1
   :widths: 20 25 25 30

   * - Regime
     - :math:`G'(\omega)`
     - :math:`G''(\omega)`
     - Physical interpretation
   * - Low :math:`\omega \ll 1/\tau`
     - :math:`\sim \omega^2`
     - :math:`\sim \omega`
     - Viscous liquid (tan :math:`\delta` ≫ 1)
   * - :math:`\omega \approx 1/\tau`
     - :math:`\approx G/2`
     - :math:`\approx G/2`
     - Balanced viscoelastic
   * - High :math:`\omega \gg 1/\tau`
     - :math:`\to G`
     - :math:`\to 0`
     - Elastic solid (tan :math:`\delta` → 0)

Diagnostic Signatures
~~~~~~~~~~~~~~~~~~~~~

- **Peak in** :math:`G''` **at** :math:`\omega \approx 1/\tau`: Characteristic signature of single relaxation time
- **Slope of log** :math:`G'` **vs log** :math:`\omega` **= 2 at low** :math:`\omega`: Terminal behavior of viscoelastic liquids
- **Loss tangent**: :math:`\tan\delta = 1/(\omega\tau)` is **monotonically decreasing** (unlike Zener model with minimum)

----

What You Can Learn
------------------

This section explains how to translate fitted Maxwell parameters into material
insights and actionable knowledge for both research and industrial applications.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**G (Spring Modulus)**:
   Fitted :math:`G` reveals the instantaneous elastic response:

   - **Low values (<** :math:`10^4` **Pa)**: Dilute solution, low entanglement density, or near-terminal regime
   - **Moderate values (** :math:`10^4-10^6` **Pa)**: Typical processing-grade polymer melts, well-entangled
   - **High values (>** :math:`10^6` **Pa)**: Very high MW, high entanglement density, or glassy contribution

   *For graduate students*: Compare with plateau modulus :math:`G_N^0` from Generalized Maxwell
   or from :math:`G_N^0 = \rho RT / M_e` to estimate entanglement MW. The single Maxwell :math:`G`
   underestimates :math:`G_N^0` when multiple modes contribute.

   *For practitioners*: :math:`G` indicates die swell magnitude and elastic recoil strength.
   Higher :math:`G` means more pronounced elastic effects in processing.

**eta (Dashpot Viscosity)**:
   Fitted :math:`\eta` reveals the flow resistance:

   - **Low values (<** :math:`10^3` **Pa·s)**: Low MW, high temperature, or weak entanglement
   - **Moderate values (** :math:`10^3-10^6` **Pa·s)**: Typical polymer melt processing range
   - **High values (>** :math:`10^6` **Pa·s)**: Very high MW, low temperature, or near :math:`T_g`

   *For graduate students*: Use :math:`\eta \sim M^{3.4}` scaling (for :math:`M > 2M_c`) to estimate
   molecular weight. Compare with capillary viscometry or GPC data.

   *For practitioners*: :math:`\eta` controls pumping power requirements and flow rates
   in processing. Higher :math:`\eta` means slower filling, higher pressures needed.

**tau (Relaxation Time)**:
   The derived parameter :math:`\tau = \eta/G` is the most important for processing:

   - **Short** :math:`\tau` **(<0.1 s)**: Fast relaxation, minimal melt memory, easy processing
   - **Moderate** :math:`\tau` **(0.1-10 s)**: Typical processing regime, some elastic effects
   - **Long** :math:`\tau` **(>10 s)**: Strong melt memory, stress relaxation issues, orientation effects

   *For graduate students*: Compare :math:`\tau` with reptation time :math:`\tau_d` from
   tube model. For monodisperse melts, single Maxwell :math:`\tau \approx \tau_d`.

   *For practitioners*: Processing Deborah number :math:`De = \tau \cdot \dot{\gamma}_{process}`
   determines whether elastic (:math:`De > 1`) or viscous (:math:`De < 1`) effects dominate.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from Maxwell Parameters
   :header-rows: 1
   :widths: 22 22 28 28

   * - Parameter Pattern
     - Material Type
     - Examples
     - Processing Notes
   * - High :math:`G`, high :math:`\eta`
     - High-MW entangled melt
     - UHMWPE, high-MW PS
     - Strong elastic effects, die swell
   * - Low :math:`G`, low :math:`\eta`
     - Low-MW oligomer/liquid
     - Wax, low-MW PDMS
     - Near-Newtonian, easy processing
   * - High :math:`G`, low :math:`\eta` (short :math:`\tau`)
     - Concentrated, fast-relaxing
     - Branched polymers at high T
     - Good processability
   * - Low :math:`G`, high :math:`\eta` (long :math:`\tau`)
     - Dilute but entangled
     - Dilute polymer solution
     - Slow dynamics, low elasticity

Molecular Weight Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

From fitted :math:`\eta` using the empirical scaling relation:

.. math::

   M_w \approx \left( \frac{\eta}{K_\eta} \right)^{1/3.4}

where :math:`K_\eta` is polymer- and temperature-specific:
   - Polyethylene (190°C): :math:`K_\eta \approx 3.4 \times 10^{-14}` (Pa·s)/(g/mol)^3.4
   - Polystyrene (170°C): :math:`K_\eta \approx 1.1 \times 10^{-14}` (Pa·s)/(g/mol)^3.4

Process Window Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

From :math:`\tau`, estimate the shear rate range for different flow behaviors:

- **Newtonian regime** (De < 0.1): :math:`\dot{\gamma} < 0.1/\tau`
- **Transition regime** (0.1 < De < 10): :math:`0.1/\tau < \dot{\gamma} < 10/\tau`
- **Elastic-dominated** (De > 10): :math:`\dot{\gamma} > 10/\tau`

For typical processing rates (:math:`\dot{\gamma} \approx 10^2 - 10^4` s\ :math:`^{-1}`), target
:math:`\tau < 0.01` s for minimal elastic effects.

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

**Warning signs in fitted parameters**:

- **If** :math:`\tau` **outside data frequency range**: :math:`G''` peak not captured; extend frequency sweep
- **If** :math:`R^2` **< 0.95**: Multiple relaxation times present; use Generalized Maxwell
- **If fit residuals show curvature**: Single exponential inadequate; try fractional models
- **If** :math:`G` **hits bounds**: Data may be in terminal regime only; verify :math:`G' \sim \omega^2` slope

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Quality Control**:
   - Track :math:`\tau` across batches to monitor MW consistency
   - Verify :math:`G` within specification for grade identification
   - Use :math:`\eta` to detect contamination or degradation

**Process Troubleshooting**:
   - High die swell → :math:`G` or :math:`\tau` too high → increase temperature or reduce MW
   - Shark skin melt fracture → :math:`\tau` too long → blend with lower-MW grade
   - Poor weld line strength → :math:`\tau` too short → increase MW or reduce temperature

**Material Development**:
   - Target :math:`\tau \approx 0.1-1` s for balanced processability
   - Increase :math:`G` by increasing MW or crosslink density
   - Reduce :math:`\tau` via chain branching or blending

Experimental Design
-------------------

Recommended Test Modes
~~~~~~~~~~~~~~~~~~~~~~

**1. Small Amplitude Oscillatory Shear (SAOS) - Frequency Sweep**

**Optimal for Maxwell**:
   - Direct measurement of :math:`G'(\omega)` and :math:`G''(\omega)`
   - Fits both storage and loss moduli simultaneously
   - Covers multiple decades in frequency

**Protocol**:
   - First perform amplitude sweep to determine LVR (typically :math:`\gamma_0 = 0.5-5\%`)
   - Frequency range: At least 2 decades bracketing :math:`1/\tau`
   - Recommended: :math:`10^{-2}` to :math:`10^2` rad/s for polymer melts
   - Temperature control: ±0.1°C (viscosity is highly temperature-dependent)

**Expected data quality**:
   - :math:`G''` peak should be well-resolved (5+ points near maximum)
   - Terminal slopes (:math:`G' \sim \omega^2`, :math:`G'' \sim \omega`) observable at low :math:`\omega`

**2. Stress Relaxation Test**

**Optimal for Maxwell**:
   - Gold standard for single-exponential relaxation
   - Direct visualization of :math:`G(t) = G e^{-t/\tau}`

**Protocol**:
   - Apply step strain :math:`\gamma_0` within LVR (1-5%)
   - Rise time < :math:`0.1\tau` (instrument limitation)
   - Measurement duration: :math:`5-10\tau` to capture full decay
   - Log-time sampling: More points at early times

**Data analysis**:
   - Plot :math:`\log G(t)` vs :math:`t` → straight line with slope :math:`-1/\tau`
   - Extract :math:`G` from intercept, :math:`\tau` from slope
   - Residuals should be random (no systematic curvature indicates multi-mode relaxation)

**3. NOT RECOMMENDED: Creep Test**

Maxwell model predicts **unbounded strain growth** :math:`J(t) = 1/G + t/\eta`, which is experimentally unrealistic for most materials at long times. Use Zener or Burgers models instead.

Sample Preparation Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Polymer melts**:
   - Compression molding at :math:`T > T_g + 50°C` to erase thermal history
   - Annealing at test temperature for 10-30 min before measurement
   - Avoid air bubbles (reduce pressure slowly during molding)
   - Typical geometry: 25 mm parallel plates, 1 mm gap

**Polymer solutions**:
   - Dissolve at :math:`T > T_g` with gentle stirring (avoid degradation)
   - Filter through 0.45 :math:`\mum` PTFE filter to remove aggregates
   - Equilibrate at test temperature for 30 min
   - Use solvent trap to prevent evaporation

**Temperature-dependent materials**:
   - Construct master curves via time-temperature superposition (TTS)
   - Measure at 5-10 temperatures spanning 20-50°C range
   - Apply WLF or Arrhenius shift factors (see :doc:`../../transforms/mastercurve`)

Common Experimental Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Troubleshooting experimental issues
   :header-rows: 1
   :widths: 30 35 35

   * - Artifact
     - Symptom
     - Solution
   * - Wall slip
     - :math:`G'`, :math:`G''` artificially low, non-reproducible
     - Use serrated plates, reduce gap, check with multiple geometries
   * - Inertia (high :math:`\omega`)
     - :math:`G'` increases spuriously at high :math:`\omega`
     - Reduce tool inertia, use smaller geometry, limit :math:`\omega < 100` rad/s
   * - Edge fracture
     - Sudden drop in :math:`G'` at high strain
     - Reduce :math:`\gamma_0`, use cone-plate geometry
   * - Sample degradation
     - Drift in :math:`G'`, :math:`G''` over time
     - Reduce temperature, minimize air exposure, use antioxidants
   * - Insufficient relaxation
     - Non-exponential :math:`G(t)` decay
     - Extend measurement to :math:`10\tau`, check for multi-mode behavior

Fitting Guidance
----------------

Parameter Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From frequency sweep data**

**Step 1**: Estimate :math:`\tau` from :math:`G''(\omega)` peak
   :math:`\tau \approx 1 / \omega_{\max}` where :math:`G''` is maximum

**Step 2**: Estimate :math:`G` from high-frequency plateau
   :math:`G \approx \lim_{\omega \to \infty} G'(\omega)`

**Step 3**: Calculate :math:`\eta`
   :math:`\eta = G \tau`

**Method 2: From stress relaxation data**

**Step 1**: Linear regression on :math:`\log G(t)` vs :math:`t`
   Slope = :math:`-1/\tau`, Intercept = :math:`\log G`

**Step 3**: Extract parameters directly from fit

**Method 3: From zero-shear viscosity (flow curve)**

If :math:`\eta_0` is known from steady shear:
   :math:`\eta = \eta_0`

Then fit :math:`G` from oscillatory data with :math:`\eta` fixed.

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for Maxwell model (2 parameters, well-conditioned)
   - 5-270× faster than scipy.optimize
   - Robust to initial guesses if parameters initialized correctly

**Alternative: Bayesian inference (NUTS)**
   - Use when parameter uncertainty quantification needed
   - Warm-start from NLSQ fit for faster convergence
   - See :doc:`../../examples/bayesian/01-bayesian-basics`

**Bounds**:
   - :math:`G`: [1e2, 1e8] Pa (adjust based on material)
   - :math:`\eta`: [1e-2, 1e10] Pa·s
   - :math:`\tau = \eta/G` not fitted directly (derived parameter)

Troubleshooting Common Fitting Problems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics and solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Diagnostic
     - Solution
   * - Poor fit at low :math:`\omega`
     - :math:`G'` underestimated (terminal slope wrong)
     - Check for multi-mode relaxation, consider Generalized Maxwell
   * - Poor fit at high :math:`\omega`
     - :math:`G'` doesn't plateau
     - Extend frequency range, check for glass transition effects
   * - :math:`G''` peak not captured
     - :math:`\tau` outside data range
     - Expand frequency window to bracket :math:`1/\tau`
   * - Converged but :math:`R^2 < 0.95`
     - Single Maxwell inadequate
     - Use multi-mode Maxwell or fractional model (FML)
   * - Fitted :math:`\eta` unrealistic
     - Units mismatch or poor initialization
     - Verify data units (Pa, rad/s), reinitialize from :math:`G''` peak

Validation Strategies
~~~~~~~~~~~~~~~~~~~~~

**1. Residual Analysis**

**Visual check**:
   - Plot residuals :math:`r_i = \log|G^*_{\text{data}}| - \log|G^*_{\text{fit}}|` vs :math:`\omega`
   - Should be **randomly scattered** around zero (no trends)
   - Systematic curvature → model inadequacy (try multi-mode or fractional)

**Statistical test**:
   - :math:`R^2 > 0.98` for good fit (oscillatory data typically noisy)
   - RMSE in log-space should be :math:`< 0.1` (10% error)

**2. Physical Plausibility**

.. math::

   \text{Check: } \tau_{\text{fitted}} = \eta_{\text{fitted}} / G_{\text{fitted}}

Should match :math:`\tau` from :math:`G''` peak location within 10%.

**3. Kramers-Kronig Relations**

Verify causality:

.. math::

   G'(\omega) = \frac{2}{\pi} \int_0^\infty \frac{x G''(x)}{x^2 - \omega^2} dx

For Maxwell model, this is **automatically satisfied** (analytical model). Use for experimental data validation.

**4. Cross-validation with Different Test Modes**

- Fit SAOS data → predict :math:`G(t)` via inverse Fourier transform
- Compare with stress relaxation measurements
- Discrepancies indicate time-temperature superposition failure or nonlinearity

Worked Example with Numbers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Material**: Polystyrene melt at 170°C

**Experimental data** (SAOS frequency sweep):
   - :math:`G''` peak at :math:`\omega = 0.1` rad/s
   - High-frequency plateau: :math:`G' \approx 1.0 \times 10^5` Pa

**Initialization**:
   - :math:`\tau = 1/0.1 = 10` s
   - :math:`G = 1.0 \times 10^5` Pa
   - :math:`\eta = G\tau = 1.0 \times 10^6` Pa·s

**Optimization** (NLSQ with 100 iterations):
   - Fitted: :math:`G = 9.8 \times 10^4` Pa, :math:`\eta = 1.05 \times 10^6` Pa·s
   - :math:`R^2 = 0.992`, RMSE = 0.08
   - Validation: :math:`\tau_{\text{fit}} = 10.7` s vs :math:`\tau_{\text{init}} = 10` s (7% difference, excellent)

**Interpretation**:
   - Molecular weight: :math:`M \sim (\eta_0)^{1/3.4} \approx 180` kg/mol (using :math:`\eta \sim M^{3.4}`)
   - Entanglement time: :math:`\tau \sim M^3 / M_e^3 \approx 10` s consistent with literature

Model Comparison
----------------

When to Use Maxwell vs Alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Model selection decision tree
   :header-rows: 1
   :widths: 25 35 40

   * - Use Maxwell when...
     - Use alternative when...
     - Recommended model
   * - Viscoelastic **liquid** (flows)
     - Viscoelastic **solid** (finite :math:`G_e`)
     - Zener (SLS), FZSS
   * - **Single** relaxation time dominates
     - **Broad** relaxation spectrum
     - Generalized Maxwell, FML
   * - **Exponential** :math:`G(t)` decay
     - **Power-law** :math:`G(t) \sim t^{-\alpha}` decay
     - FMG, SpringPot
   * - Stress relaxation analysis
     - Creep/recovery analysis
     - Zener, Burgers, FKV
   * - 2 parameters sufficient
     - Higher accuracy needed
     - Multi-mode Maxwell (Prony)
   * - Educational/conceptual
     - Quantitative predictions
     - Fractional or multi-mode

Model Hierarchy (Simpler → More Complex)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Level 1: Maxwell (this model)**
   - 2 parameters: :math:`G`, :math:`\eta`
   - Exponential relaxation
   - Viscoelastic liquid only

**Level 2: Zener (Standard Linear Solid)**
   - 3 parameters: :math:`G_e`, :math:`G_m`, :math:`\eta`
   - Adds equilibrium modulus → viscoelastic solid
   - Exponential relaxation to finite plateau
   - See :doc:`zener`

**Level 3: Generalized Maxwell (Prony series)**
   - :math:`2N` parameters (N Maxwell elements in parallel)
   - Multiple relaxation times → broader spectra
   - Computationally expensive (:math:`N = 10-20` typical)

**Level 4: Fractional Maxwell Liquid (FML)**
   - 3 parameters: :math:`G_0`, :math:`\tau`, :math:`\alpha`
   - Power-law relaxation via Mittag-Leffler function
   - Fewer parameters than multi-mode for broad spectra
   - See :doc:`../fractional/fractional_maxwell_liquid`

**Level 5: Fractional Maxwell Gel (FMG)**
   - 3 parameters
   - Power-law relaxation **without** terminal flow (gel-like)
   - See :doc:`../fractional/fractional_maxwell_gel`

Diagnostic Tests to Discriminate Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Test 1: Plot log** :math:`G(t)` **vs** :math:`t`
   - **Linear** → Maxwell (exponential)
   - **Curved** → Multi-mode or fractional

**Test 2: Plot log** :math:`G''` **vs log** :math:`\omega`
   - **Single symmetric peak** → Maxwell
   - **Broad peak** or **multiple peaks** → Multi-mode
   - **Linear slope** (:math:`\alpha < 1`) → Fractional (SpringPot)

**Test 3: Creep-recovery test**
   - **No recovery** → Maxwell (pure liquid)
   - **Partial recovery** → Zener, Burgers (viscoelastic solid)
   - **Power-law creep** → Fractional models

**Test 4: Tan** :math:`\delta` **behavior**
   - **Monotonic decrease** with :math:`\omega` → Maxwell
   - **Minimum** in tan :math:`\delta` → Zener
   - **Constant** tan :math:`\delta` → Critical gel (FMG with :math:`\alpha \approx 0.5`)

Connection to Advanced Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Reptation (Doi-Edwards)**:
   The Maxwell model is the single-mode approximation of reptation theory. The disengagement time :math:`\tau_d` corresponds to Maxwell :math:`\tau`, and the plateau modulus :math:`G_N^0` corresponds to :math:`G`.

**Cox-Merz rule**:
   For materials obeying Cox-Merz, steady-shear viscosity :math:`\eta(\dot{\gamma})` can be predicted from complex viscosity :math:`|\eta^*(\omega)|`:

   .. math::

      \eta(\dot{\gamma}) \approx |\eta^*(\omega)| \bigg|_{\omega = \dot{\gamma}}

   Maxwell model: :math:`|\eta^*| = \eta / \sqrt{1 + (\omega\tau)^2}`

**Winter-Chambon criterion**:
   Maxwell model does **not** satisfy gel point criterion (tan :math:`\delta` ≠ constant). Use FMG for critical gels.

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.Maxwell`

Usage
-----

Basic Fitting Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import Maxwell

   omega = np.logspace(-2, 2, 100)
   maxwell = Maxwell()
   maxwell.fit(omega, data)  # replace ``data`` with target complex modulus

   G0 = maxwell.parameters.get_value('G0')
   eta = maxwell.parameters.get_value('eta')
   tau = eta / G0

   Gstar = maxwell.predict(omega)

Advanced Usage: Bayesian Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell
   import numpy as np

   # 1. NLSQ point estimation (fast)
   model = Maxwell()
   model.fit(omega, G_data)

   # 2. Bayesian inference with warm-start
   result = model.fit_bayesian(
       omega, G_data,
       num_warmup=1000,
       num_samples=2000
   )

   # 3. Get credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
   print(f"G: [{intervals['G'][0]:.2e}, {intervals['G'][1]:.2e}] Pa")
   print(f"eta: [{intervals['eta'][0]:.2e}, {intervals['eta'][1]:.2e}] Pa·s")

Examples
--------

Fit to oscillatory data
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell
   model = Maxwell()
   model.fit(omega, G_star)
   print(model.score(omega, G_star))

Time-Temperature Superposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import Maxwell
   from rheojax.transforms import Mastercurve

   # Create master curve at reference temperature
   mc = Mastercurve(reference_temp=170, method='wlf', C1=17.44, C2=51.6)
   master_data, shifts = mc.transform(multi_temp_datasets)

   # Fit Maxwell to extended frequency range
   model = Maxwell()
   model.fit(master_data.omega, master_data.G_star)

See Also
--------

**Classical Models:**

- :doc:`zener` — adds a parallel spring for finite creep recovery
- :doc:`springpot` — fractional generalization providing power-law slopes

**Fractional Models:**

- :doc:`../fractional/fractional_maxwell_gel` — series dashpot + springpot capturing gel behavior
- :doc:`../fractional/fractional_maxwell_liquid` — fractional dashpot for broad relaxation spectra

**Transforms:**

- :doc:`../../transforms/fft` — convert time-domain data to :math:`G'(\omega)` and :math:`G''(\omega)` prior to fitting
- :doc:`../../transforms/mastercurve` — time-temperature superposition for extending frequency range

**Examples:**

- :doc:`../../examples/basic/01-maxwell-fitting` — notebook demonstrating parameter estimation and validation

**User Guides:**

- :doc:`../../user_guide/model_selection` — decision flowcharts for choosing rheological models

References
----------

.. [1] Ferry, J. D. *Viscoelastic Properties of Polymers*, 3rd Edition. Wiley (1980).
   ISBN: 978-0471048947. Classic reference for linear viscoelasticity, WLF equation,
   and molecular theories.

.. [2] Tschoegl, N. W. *The Phenomenological Theory of Linear Viscoelastic Behavior*.
   Springer, Berlin (1989). https://doi.org/10.1007/978-3-642-73602-5. Rigorous
   mathematical treatment of constitutive equations and interconversion relations.

.. [3] Macosko, C. W. *Rheology: Principles, Measurements, and Applications*.
   Wiley-VCH, New York (1994). ISBN: 978-0471185758. Excellent balance of theory,
   experimental techniques, and practical applications.

.. [4] Barnes, H. A., Hutton, J. F., and Walters, K. *An Introduction to Rheology*.
   Elsevier, Amsterdam (1989). ISBN: 978-0444871404. Accessible introduction
   covering viscosity, viscoelasticity, and normal stresses.

.. [5] Doi, M., and Edwards, S. F. *The Theory of Polymer Dynamics*. Oxford
   University Press (1986). ISBN: 978-0198520337. Foundation for reptation theory
   connecting Maxwell model to molecular dynamics.

.. [6] McLeish, T. C. B. "Tube Theory of Entangled Polymer Dynamics."
   *Advances in Physics*, 51(6), 1379–1527 (2002).
   https://doi.org/10.1080/00018730210153216

.. [7] Maxwell, J. C. "On the Dynamical Theory of Gases."
   *Philosophical Transactions of the Royal Society of London*, 157, 49–88 (1867).
   https://doi.org/10.1098/rstl.1867.0004. Original paper introducing the model.

.. [8] Bird, R. B., Armstrong, R. C., and Hassager, O. *Dynamics of Polymeric
   Liquids, Volume 1: Fluid Mechanics*. 2nd ed., Wiley, New York (1987).
   ISBN: 978-0471802457

.. [9] Larson, R. G. *The Structure and Rheology of Complex Fluids*.
   Oxford University Press, New York (1999). ISBN: 978-0195121971

.. [10] Rubinstein, M., and Colby, R. H. *Polymer Physics*.
   Oxford University Press (2003). ISBN: 978-0198520597
