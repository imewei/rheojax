.. _time-temperature-superposition:

Time-Temperature Superposition (TTS)
=====================================

Overview
--------

**Time-Temperature Superposition (TTS)** is a powerful principle in rheology enabling extension of measurement frequency ranges by 4-8 decades through temperature sweeps. Instead of mechanically accessing ω = 10⁻⁶ to 10⁶ rad/s (requires 12 decades, impossible for most instruments), measure at 5-10 temperatures and **shift** data horizontally to create a **mastercurve** at a reference temperature.

**Physical basis**: The temperature dependence of molecular relaxation processes is **separable** from their fundamental shape. Increasing temperature accelerates all relaxation processes by the same factor, equivalent to shifting the time scale.

**Mathematical formulation**:

.. math::

   G(t, T) = G(t/a_T, T_{\text{ref}})

   G'(\omega, T) = G'(\omega \cdot a_T, T_{\text{ref}})

where :math:`a_T` is the **shift factor** depending on temperature T and reference temperature :math:`T_{\text{ref}}`.

**Applicability**:
   - **Valid for**: Amorphous polymers above T_g, polymer melts, viscoelastic liquids
   - **Requires**: Thermorheologically simple materials (all relaxation modes shift equally)
   - **Fails for**: Crystalline polymers, multi-phase systems with different T-dependencies, materials undergoing phase transitions

Manual Shift Methods
---------------------

WLF Equation (Williams-Landel-Ferry)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Empirical equation** for shift factors of amorphous polymers near glass transition:

.. math::

   \log_{10}(a_T) = \frac{-C_1 (T - T_{\text{ref}})}{C2 + (T - T_{\text{ref}})}

**Parameters**:
   - :math:`C_1`: Dimensionless constant (universal ≈ 17.44)
   - :math:`C_2`: Temperature constant in Kelvin (universal ≈ 51.6 K)
   - :math:`T_{\text{ref}}`: Reference temperature (often T_g + 50°C)

**Universal values** (reference at T_g):
   - :math:`C_1 = 17.44`, :math:`C_2 = 51.6` K when :math:`T_{\text{ref}} = T_g`
   - Valid for most polymers within :math:`T_g` to :math:`T_g + 100°C`

**Material-specific values**: Can be optimized for better accuracy (see ``optimize_wlf_parameters()``).

**Physical interpretation**:
   - Free volume theory: Molecular mobility increases with free volume
   - :math:`C_1` relates to fractional free volume at T_g
   - :math:`C_2` relates to thermal expansion coefficient

**Usage example**:

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve

   # Create WLF mastercurve
   mc = Mastercurve(
       reference_temp=373.15,  # 100°C in Kelvin
       method='wlf',
       C1=17.44,
       C2=51.6,
       auto_shift=False  # Manual WLF
   )

   # Transform multi-temperature datasets
   mastercurve, shift_factors = mc.transform(datasets)

   # Get WLF parameters
   params = mc.get_wlf_parameters()
   print(params)  # {'C1': 17.44, 'C2': 51.6, 'T_ref': 373.15}

Arrhenius Equation
~~~~~~~~~~~~~~~~~~

**Exponential temperature dependence** for high-temperature flows (T ≫ T_g):

.. math::

   \log(a_T) = \frac{E_a}{R} \left( \frac{1}{T} - \frac{1}{T_{\text{ref}}} \right)

**Parameters**:
   - :math:`E_a`: Activation energy (J/mol)
   - :math:`R = 8.314` J/(mol·K): Gas constant
   - :math:`T`: Absolute temperature (Kelvin)

**Physical interpretation**:
   - Energy barrier for molecular rearrangement
   - Higher :math:`E_a` → stronger temperature dependence

**When to use Arrhenius**:
   - High temperatures (T > T_g + 100°C)
   - Polymer melts well above glass transition
   - Liquids without glass transition (oils, solvents)

**Usage example**:

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve

   # Create Arrhenius mastercurve
   mc = Mastercurve(
       reference_temp=453.15,  # 180°C in Kelvin
       method='arrhenius',
       E_a=85000,  # J/mol (85 kJ/mol for PS melt)
       auto_shift=False
   )

   mastercurve, shift_factors = mc.transform(datasets)

   # Get Arrhenius parameters
   params = mc.get_arrhenius_parameters()
   print(params)  # {'E_a': 85000, 'T_ref': 453.15}

When to Use Each Method
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: WLF vs Arrhenius decision guide
   :header-rows: 1
   :widths: 30 35 35

   * - Criterion
     - WLF
     - Arrhenius
   * - Temperature range
     - T_g to T_g + 100°C
     - T > T_g + 100°C
   * - Material state
     - Rubbery plateau
     - Viscous melt
   * - Physical basis
     - Free volume theory
     - Activation energy
   * - Typical materials
     - Amorphous polymers near T_g
     - Polymer melts, oils
   * - Shift factor curvature
     - Non-linear (curved)
     - Linear (straight line)
   * - Parameter availability
     - Universal C1, C2 available
     - Material-specific E_a needed

Automatic Shift Method (NEW in v0.2.2)
---------------------------------------

Power-Law Intersection Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem**: What if WLF/Arrhenius parameters are **unknown** or material is **non-standard**?

**Solution**: Automatic shift factor calculation via **power-law intersection method** (PyVisco algorithm).

**Key idea**: Each temperature curve locally behaves as a power-law in log-log space:

.. math::

   y = a \cdot x^b + e

Fit adjacent curves, find their **intersection** in the overlap or gap region, compute relative shift.

**Algorithm steps**:

1. **Fit power-law** to each temperature curve:
      - Model: :math:`y = a \cdot x^b + e`
      - NLSQ optimization (JAX-accelerated, 5-270× speedup vs scipy)

2. **Detect outliers**:
      - Try removing first data point
      - Keep removal if exponent error improves (lower uncertainty in b)

3. **Compute pairwise shifts**:
      - For adjacent temperatures T_i and T_{i+1}
      - Sample 10 points in overlap/gap region
      - Compute inverse power-law: :math:`x = \left(\frac{y - e}{a}\right)^{1/b}`
      - Average log shift: :math:`\log(a_T) = \text{mean}(\log_{10}(x_{\text{top}} / x_{\text{bot}}))`

4. **Sequential cumulative shifting**:
      - Start from reference temperature (shift = 0)
      - Move outward (both higher and lower T)
      - Accumulate shifts: :math:`\log(a_{T_i}) = \log(a_{T_{i-1}}) + \Delta \log(a_T)`

**Advantages**:
   - No material parameters needed (WLF C1, C2 or Arrhenius E_a)
   - Handles non-standard materials (blends, composites)
   - Robust to outliers (first-point removal)
   - JAX-accelerated (fast NLSQ fitting)

**Limitations**:
   - Requires clear overlap or gap between temperatures
   - Less accurate for sparse data (<10 points per curve)
   - Assumes local power-law behavior

**Usage example**:

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve
   from rheojax.core.data import RheoData
   import numpy as np

   # Multi-temperature SAOS data
   temps = [140, 160, 180, 200, 220]  # °C
   datasets = []

   for T in temps:
       omega = np.logspace(-2, 2, 50)
       G_prime = ...  # Experimental data
       G_double_prime = ...

       data = RheoData(
           x=omega,
           y=np.column_stack([G_prime, G_double_prime]),
           domain='frequency',
           metadata={'temperature': T + 273.15}  # Kelvin
       )
       datasets.append(data)

   # Automatic shift factor calculation
   mc_auto = Mastercurve(
       reference_temp=180+273.15,  # 180°C
       auto_shift=True  # Enable automatic shifting
   )

   mastercurve, shift_factors = mc_auto.transform(datasets)

   # Get computed shift factors
   temps_arr, log_aT_arr = mc_auto.get_auto_shift_factors()

   # Plot shift factors
   import matplotlib.pyplot as plt
   plt.plot(temps_arr - 273.15, log_aT_arr, 'o-')
   plt.xlabel('Temperature (°C)')
   plt.ylabel('log(a_T)')
   plt.axhline(0, color='k', linestyle='--', label='Reference T')
   plt.grid(True)
   plt.legend()
   plt.show()

Transition Guide: Auto vs Manual
---------------------------------

Decision Flowchart
~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Do you know WLF or Arrhenius parameters?
      ├─ YES → Are they from literature or trusted source?
      │         ├─ YES → Use manual WLF/Arrhenius
      │         │         Validate with compute_overlap_error()
      │         │
      │         └─ NO  → Use auto-shift, then validate against manual
      │
      └─ NO  → Is the material standard (PS, PMMA, PE)?
                ├─ YES → Try universal WLF (C1=17.44, C2=51.6)
                │         Optimize if needed: optimize_wlf_parameters()
                │
                └─ NO  → Use auto-shift (power-law intersection)

When Auto-Shift is Appropriate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use automatic shift factors when:**

1. **Unknown material parameters**:
      - New polymer blend (no literature WLF)
      - Composite with multiple phases
      - Modified polymer (plasticizers, fillers)

2. **Non-standard temperature ranges**:
      - T range outside WLF validity (T_g to T_g+100°C)
      - High-temperature melts where WLF breaks down
      - Low-temperature regime near T_g where Arrhenius fails

3. **Quick screening**:
      - Exploratory measurements
      - Quality control (batch-to-batch comparison)
      - Preliminary characterization

4. **Validation**:
      - Check if literature WLF parameters work
      - Compare auto-shift vs manual for consistency

When Manual Shift is Better
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Use manual WLF/Arrhenius when:**

1. **Known material with established parameters**:
      - Commercial polymers (PS, PMMA, PDMS)
      - Material matches literature closely
      - WLF parameters from supplier datasheet

2. **Extrapolation needed**:
      - Predict behavior at temperatures outside measurement range
      - WLF/Arrhenius have physical basis (can extrapolate)
      - Auto-shift only interpolates between measured temperatures

3. **Sparse data**:
      - Few temperatures (< 4)
      - Few points per temperature (< 10)
      - Auto-shift unreliable with sparse data

4. **Physical interpretation required**:
      - Need activation energy E_a for reaction kinetics
      - Compare C1, C2 across material batches
      - Fundamental understanding of molecular mobility

Hybrid Workflow: Auto → Validate → Refine with Manual
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best practice**: Use auto-shift for initial exploration, then refine with manual methods.

**Workflow**:

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve
   import matplotlib.pyplot as plt

   # Step 1: Auto-shift (exploratory)
   mc_auto = Mastercurve(reference_temp=180+273.15, auto_shift=True)
   mastercurve_auto, shifts_auto = mc_auto.transform(datasets)

   # Step 2: Manual WLF (literature parameters)
   mc_wlf = Mastercurve(
       reference_temp=180+273.15,
       method='wlf',
       C1=8.86,   # PS universal WLF at T_ref=170°C
       C2=101.6,  # Adjusted C2 for PS
       auto_shift=False
   )
   mastercurve_wlf, shifts_wlf = mc_wlf.transform(datasets)

   # Step 3: Compare shift factors
   temps_auto, log_aT_auto = mc_auto.get_auto_shift_factors()
   temps_wlf, log_aT_wlf = mc_wlf.get_shift_factors_array()

   plt.figure(figsize=(10, 5))

   plt.subplot(1, 2, 1)
   plt.plot(temps_auto - 273.15, log_aT_auto, 'o-', label='Auto-shift')
   plt.plot(temps_wlf - 273.15, log_aT_wlf, 's-', label='WLF (literature)')
   plt.xlabel('Temperature (°C)')
   plt.ylabel('log(a_T)')
   plt.legend()
   plt.grid(True)
   plt.title('Shift Factor Comparison')

   plt.subplot(1, 2, 2)
   # Plot mastercurves
   plt.loglog(mastercurve_auto.x, mastercurve_auto.y[:, 0], 'o',
              alpha=0.5, label="G' (auto)")
   plt.loglog(mastercurve_wlf.x, mastercurve_wlf.y[:, 0], 's',
              alpha=0.5, label="G' (WLF)")
   plt.xlabel('Shifted frequency (rad/s)')
   plt.ylabel("G' (Pa)")
   plt.legend()
   plt.grid(True)
   plt.title('Mastercurve Comparison')

   plt.tight_layout()
   plt.show()

   # Step 4: Compute overlap errors
   error_auto = mc_auto.compute_overlap_error(datasets)
   error_wlf = mc_wlf.compute_overlap_error(datasets)

   print(f"Overlap error (auto): {error_auto:.2e}")
   print(f"Overlap error (WLF):  {error_wlf:.2e}")

   # Step 5: Optimize WLF if needed
   if error_auto < error_wlf * 0.8:
       # Auto-shift significantly better, optimize WLF
       C1_opt, C2_opt = mc_wlf.optimize_wlf_parameters(
           datasets,
           initial_C1=8.86,
           initial_C2=101.6
       )
       print(f"Optimized WLF: C1={C1_opt:.2f}, C2={C2_opt:.2f}")

DMA Applications (Dynamic Mechanical Analyzer)
-----------------------------------------------

Temperature Sweep Protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Goal**: Measure viscoelastic properties of **solid-like materials** (polymers, composites) as function of temperature.

**Standard DMA temperature sweep**:

1. **Setup**:
      - Geometry: Tension, compression, or torsion (material-dependent)
      - Sample: Rectangular bar (10×5×1 mm typical)
      - Fixed frequency: ω = 1 rad/s (or 1 Hz = 6.28 rad/s)
      - Strain amplitude: 0.1-1% (within LVR, verified by amplitude sweep)

2. **Temperature ramp**:
      - Range: T_g - 50°C to T_g + 100°C (e.g., -50°C to 150°C for PMMA)
      - Ramp rate: 2-5°C/min (slow enough for thermal equilibrium)
      - Data points: Every 2-5°C (50-100 points total)

3. **Measurements**:
      - Storage modulus :math:`E'(T)` or :math:`G'(T)`
      - Loss modulus :math:`E''(T)` or :math:`G''(T)`
      - Loss tangent :math:`\tan \delta = E'' / E'`

**Glass transition identification**:
      - **Peak in** :math:`\tan \delta`: T at maximum loss
      - **Inflection in** :math:`E'`: Rapid drop in modulus
      - **Peak in** :math:`E''`: Energy dissipation maximum

Storage/Loss Modulus Master Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Frequency sweep mastercurve** (isothermal at multiple T):

1. **Isothermal frequency sweeps**:
      - Temperatures: 5-10 values spanning T_g to T_g+100°C
      - Example for PMMA: T = 110°C, 120°C, 130°C, 140°C, 150°C
      - Frequency range per T: ω = 10⁻² to 10² rad/s (4 decades)
      - Points per decade: 10 (total 40 points per T)

2. **Time-temperature superposition**:

   .. code-block:: python

      from rheojax.transforms.mastercurve import Mastercurve
      from rheojax.core.data import RheoData
      import numpy as np

      # Prepare datasets
      temps = [110, 120, 130, 140, 150]  # °C
      datasets = []

      for T in temps:
          omega = np.logspace(-2, 2, 40)
          E_prime = ...  # DMA measurements
          E_double_prime = ...

          data = RheoData(
              x=omega,
              y=np.column_stack([E_prime, E_double_prime]),
              domain='frequency',
              metadata={'temperature': T + 273.15}
          )
          datasets.append(data)

      # Create mastercurve (auto-shift)
      mc = Mastercurve(reference_temp=130+273.15, auto_shift=True)
      mastercurve, shift_factors = mc.transform(datasets)

      # Result: ω_reduced spans 10⁻⁶ to 10⁶ rad/s (12 decades)

3. **Plotting mastercurve**:

   .. code-block:: python

      import matplotlib.pyplot as plt

      omega_master = mastercurve.x
      E_prime_master = mastercurve.y[:, 0]
      E_double_prime_master = mastercurve.y[:, 1]

      plt.figure(figsize=(10, 6))
      plt.loglog(omega_master, E_prime_master, 'o', label="E' (mastercurve)")
      plt.loglog(omega_master, E_double_prime_master, 's', label='E" (mastercurve)')
      plt.xlabel('Reduced frequency ω (rad/s)')
      plt.ylabel('Modulus (Pa)')
      plt.title(f'PMMA Mastercurve at T_ref = 130°C')
      plt.legend()
      plt.grid(True)
      plt.show()

**Expected behavior**:
   - Low ω: Rubbery plateau (:math:`E' \approx 10^6` Pa, nearly constant)
   - Mid ω: Transition region (:math:`E'` increases, :math:`E''` peak)
   - High ω: Glassy plateau (:math:`E' \approx 3 \times 10^9` Pa)

Rheometer Applications (Fluid Dynamics)
----------------------------------------

Complex Viscosity Master Curves
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rotational rheometer** for **polymer melts** and **viscoelastic fluids**:

.. code-block:: python

   from rheojax.transforms.mastercurve import Mastercurve
   import numpy as np

   # Temperatures for PS melt
   temps = [140, 160, 180, 200, 220]  # °C
   datasets = []

   for T in temps:
       omega = np.logspace(-2, 2, 50)
       G_prime = ...  # Pa
       G_double_prime = ...  # Pa

       data = RheoData(
           x=omega,
           y=np.column_stack([G_prime, G_double_prime]),
           domain='frequency',
           metadata={'temperature': T + 273.15}
       )
       datasets.append(data)

   # Automatic shift factors
   mc = Mastercurve(reference_temp=180+273.15, auto_shift=True)
   mastercurve, shifts = mc.transform(datasets)

   # Complex viscosity
   omega_master = mastercurve.x
   G_star_master = mastercurve.y
   G_prime = G_star_master[:, 0]
   G_double_prime = G_star_master[:, 1]

   eta_star = np.sqrt(G_prime**2 + G_double_prime**2) / omega_master

Cox-Merz Rule Validation
~~~~~~~~~~~~~~~~~~~~~~~~~

**Cox-Merz empirical rule**: For many polymers,

.. math::

   \eta(\dot{\gamma}) \approx |\eta^*(\omega)| \bigg|_{\omega = \dot{\gamma}}

**Test procedure**:

1. **SAOS** (oscillatory): Measure :math:`G'(\omega)`, :math:`G''(\omega)` → compute :math:`|\eta^*| = |G^*| / \omega`

2. **Steady shear** (flow curve): Measure :math:`\eta(\dot{\gamma})` vs shear rate

3. **Compare** at matched frequencies/rates:

   .. code-block:: python

      import matplotlib.pyplot as plt

      # SAOS complex viscosity (from mastercurve)
      omega = ...  # rad/s
      eta_star = ...  # Pa·s

      # Steady shear viscosity (flow curve)
      shear_rate = ...  # 1/s
      viscosity_steady = ...  # Pa·s

      # Plot Cox-Merz comparison
      plt.figure(figsize=(8, 6))
      plt.loglog(omega, eta_star, 'o-', label='|η*| (SAOS)')
      plt.loglog(shear_rate, viscosity_steady, 's--', label='η (steady shear)')
      plt.xlabel('ω or γ̇ (rad/s or 1/s)')
      plt.ylabel('Viscosity (Pa·s)')
      plt.title('Cox-Merz Rule Validation (PS melt, 180°C)')
      plt.legend()
      plt.grid(True)
      plt.show()

Troubleshooting
---------------

.. list-table:: Common TTS problems and solutions
   :header-rows: 1
   :widths: 30 35 35

   * - Problem
     - Symptom
     - Solution
   * - Poor superposition
     - Curves don't overlap
     - Check thermorheological simplicity (try auto-shift)
   * - Vertical shift needed
     - Horizontal shift insufficient
     - Enable vertical_shift=True, or data has density/T effects
   * - Shift factors nonphysical
     - Negative a_T or extreme values
     - Wrong T units (use Kelvin), or material not simple
   * - Auto-shift fails
     - RuntimeError in power-law fit
     - Insufficient overlap between curves, increase T range
   * - WLF extrapolation poor
     - Mastercurve breaks down at edges
     - T outside T_g to T_g+100°C, use Arrhenius
   * - Sparse data (< 10 pts/curve)
     - Auto-shift unreliable
     - Use manual WLF/Arrhenius with literature values

See Also
--------

- :doc:`../../models/multi_mode/generalized_maxwell` — Multi-mode GMM for mastercurve fitting
- :doc:`../../examples/transforms/06-mastercurve_auto_shift` — Automatic shift factors notebook
- :doc:`../../api/transforms/mastercurve` — Full API reference

References
----------

1. Ferry, J. D. (1980). *Viscoelastic Properties of Polymers*, 3rd Edition. Wiley.
2. Williams, M. L., Landel, R. F., & Ferry, J. D. (1955). "The Temperature Dependence of Relaxation Mechanisms in Amorphous Polymers." *JACS*, 77(14), 3701-3707.
3. PyVisco (2020-2024). Python package for viscoelastic model fitting. GitHub: https://github.com/NREL/pyvisco
