Fractional Isotropic-Kinematic Hardening (FIKH)
================================================

Quick Reference
---------------

- **Use when:** Thixotropic elasto-viscoplastic materials with power-law memory, stretched exponential recovery, broad relaxation spectra

- **Parameters:** 12 (base), 20 (with thermal), 22 (full thermal + isotropic hardening)

- **Key equation:** :math:`D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}} - \Gamma \lambda |\dot{\gamma}^p|` (Caputo fractional structure evolution)

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation, laos

- **Material examples:** Waxy crude oils, colloidal gels, drilling fluids, food gels, greases

.. currentmodule:: rheojax.models.fikh.fikh

.. autoclass:: FIKH
   :members:
   :show-inheritance:


Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`D^\alpha`
     - —
     - Caputo fractional derivative of order :math:`\alpha`
   * - :math:`\alpha`
     - —
     - Fractional order (0 < :math:`\alpha` < 1); controls memory strength
   * - :math:`\sigma`
     - Pa
     - Deviatoric stress (elasto-plastic component)
   * - :math:`A`
     - —
     - Backstress internal variable (:math:`\alpha_{back} = C \cdot A`)
   * - :math:`\lambda`
     - —
     - Structural parameter (0 = destructured, 1 = structured)
   * - :math:`\dot{\gamma}`
     - 1/s
     - Total shear rate
   * - :math:`\dot{\gamma}^p`
     - 1/s
     - Plastic shear rate
   * - :math:`\sigma_y`
     - Pa
     - Current yield stress (depends on :math:`\lambda` and T)
   * - :math:`\xi`
     - Pa
     - Relative stress (:math:`\xi = \sigma - C \cdot A`)
   * - :math:`E_\alpha(z)`
     - —
     - Mittag-Leffler function (generalized exponential)
   * - :math:`\Gamma(\cdot)`
     - —
     - Gamma function (appears in Caputo normalization)


Overview
--------

The **FIKH** (Fractional Isotropic-Kinematic Hardening) model extends the
:doc:`../ikh/mikh` framework by replacing integer-order structure kinetics
with a **Caputo fractional derivative**. This captures the **power-law memory**
observed in many complex fluids where recent deformation history affects current
structure more strongly than distant past.

The model combines:

1. **Maxwell viscoelasticity**: Stress relaxation via :math:`\eta` (Maxwell viscosity)
2. **Kinematic hardening**: Backstress evolution (Armstrong-Frederick type)
3. **Fractional thixotropy**: Power-law memory via Caputo derivative of :math:`\lambda`
4. **Thermokinematic coupling** (optional): Temperature-dependent yield and viscosity

The FIKH model captures:

- **Power-law stress relaxation** at long times (not just exponential)
- **Stretched exponential recovery** after shear cessation
- **Cole-Cole depression** in frequency sweeps
- **Delayed yielding** in creep tests
- **Thermal feedback** effects (shear heating, Arrhenius viscosity)


Theoretical Background
----------------------

Historical Context
~~~~~~~~~~~~~~~~~~

Fractional calculus emerged from Leibniz's 1695 correspondence with L'Hôpital
about the meaning of d^(1/2)y/dx^(1/2). It remained largely a mathematical
curiosity until the 20th century, when its natural connection to power-law
phenomena in physics became apparent.

In rheology, fractional derivatives gained prominence through:

1. **Scott Blair (1940s)**: Introduced fractional elements for viscoelasticity
2. **Caputo (1967)**: Developed the Caputo derivative with physical initial conditions
3. **Bagley & Torvik (1983)**: Fractional derivatives in polymer viscoelasticity
4. **Jaishankar & McKinley (2013-2014)**: Fractional K-BKZ for nonlinear rheology

The FIKH model synthesizes these developments with the IKH thixotropic framework,
providing a thermodynamically consistent model for materials with power-law memory.

Physical Interpretation of Fractional Order
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fractional order :math:`\alpha` ∈ (0, 1) controls the **memory strength**:

- :math:`\alpha` **→ 1**: Weak memory, fast exponential recovery (recovers MIKH)
- :math:`\alpha` **= 0.5**: Intermediate power-law relaxation (square-root memory)
- :math:`\alpha` **→ 0**: Strong memory, very slow "glassy" dynamics

**Microstructural interpretation**: In complex fluids, :math:`\alpha` relates to the **breadth
of the relaxation spectrum**. A single exponential relaxation (narrow spectrum)
corresponds to :math:`\alpha` = 1. As the spectrum broadens (multiple timescales), effective
:math:`\alpha` decreases. From Cole-Cole analysis, the depression angle :math:`\theta = (1-\alpha)\pi/2`.

**Connection to stretched exponentials**: The Mittag-Leffler function :math:`E_{\alpha(-x)}`
that appears in FIKH solutions interpolates between:

- E_1(-x) = exp(-x) (pure exponential)
- :math:`E_{\alpha(-x) for \alpha}` < 1: stretched exponential at short times, power-law at long times

Caputo Fractional Derivative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For 0 < :math:`\alpha` < 1, the **Caputo fractional derivative** is defined as:

.. math::

   D_t^\alpha f(t) = \frac{1}{\Gamma(1-\alpha)} \int_0^t \frac{f'(s)}{(t-s)^\alpha} \, ds

Key properties:

- D\ :math:`^{\alpha}` (constant) = 0 (compatible with physical initial conditions)
- As :math:`\alpha \to 1`: D\ :math:`^{\alpha}` f → df/dt (recovers ordinary derivative)
- Introduces **power-law memory** with kernel (t-s)^(-:math:`\alpha`)

**Why Caputo over Riemann-Liouville?** The Caputo derivative allows physical
interpretation of initial conditions: D\ :math:`^{\alpha(constant) = 0, so \lambda(0) = \lambda_0}` is meaningful.
The Riemann-Liouville derivative would require specifying fractional-order initial
conditions, which lack physical interpretation.

Mittag-Leffler Relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~

The **Mittag-Leffler function** is the fundamental solution to fractional
relaxation equations:

.. math::

   E_\alpha(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

For the fractional structure equation D\ :math:`^{\alpha \lambda = -(\lambda-\lambda_eq)/\tau}`, the solution is:

.. math::

   \lambda(t) = \lambda_{eq} - (\lambda_{eq} - \lambda_0) E_\alpha\left(-\left(\frac{t}{\tau}\right)^\alpha\right)

**Asymptotic behavior:**

- **Short times** (t ≪ :math:`\tau`): :math:`\lambda(t) \approx \lambda_0 + (\lambda_eq - \lambda_0) \cdot (t/\tau)^\alpha / \Gamma(1+\alpha)` (stretched exponential onset)
- **Long times** (t ≫ :math:`\tau`): :math:`\lambda(t) \approx \lambda_eq - (\lambda_eq - \lambda_0) \cdot (\tau/t)^\alpha / \Gamma(1-\alpha)` (power-law tail)

This interpolation between stretched exponential and power-law is the key
signature of fractional kinetics in experimental data.

Thermodynamic Consistency
~~~~~~~~~~~~~~~~~~~~~~~~~

The FIKH model can be derived within a generalized thermodynamic framework
that extends the Gurtin-Fried-Anand approach to fractional derivatives:

1. **Frame invariance**: Constitutive equations remain objective
2. **Second law compliance**: Dissipation inequality satisfied
3. **Energy balance**: Clear separation of stored and dissipated energy

The key modification for fractional models is replacing the standard dissipation
rate with a **fractional dissipation functional** that accounts for the non-local
memory effects. This ensures thermodynamic consistency while capturing power-law
behavior.


Physical Foundations
--------------------

Maxwell-Like Framework
~~~~~~~~~~~~~~~~~~~~~~

The FIKH model retains the Maxwell viscoelastic element from MIKH:

.. math::

   \frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma

- **Short times** (t ≪ :math:`\tau = \eta/G`): Elastic response, :math:`\sigma \approx G \cdot \gamma`
- **Long times** (t ≫ :math:`\tau`): Viscous relaxation, :math:`\sigma` → 0 under constant strain

The Maxwell viscosity :math:`\eta` sets the characteristic relaxation time :math:`\tau = \eta/G` for
the elastic stress, while the **fractional** kinetics govern the structural
parameter :math:`\lambda` on potentially much longer timescales.

Armstrong-Frederick Kinematic Hardening
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The backstress evolution captures directional memory (Bauschinger effect):

.. math::

   \frac{dA}{dt} = \dot{\gamma}^p - q|A|^{m-1}A|\dot{\gamma}^p|

Where the physical backstress is :math:`\alpha_back` = C·A. The Armstrong-Frederick law
provides:

- **Hardening**: Backstress builds with plastic deformation direction
- **Dynamic recovery**: Prevents unbounded backstress growth
- **Bauschinger effect**: Easier reverse flow after forward loading

**Steady-state backstress**: At constant shear rate:

.. math::

   A_{ss} = \frac{1}{q} \cdot \text{sign}(\dot{\gamma}^p)

The ratio C/q = C·A_ss determines the maximum backstress magnitude.

Fractional Structure Kinetics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The defining feature of FIKH is the **fractional structure evolution**:

.. math::

   D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}} - \Gamma \lambda |\dot{\gamma}^p|

Where:

- :math:`\lambda` ∈ [0, 1]: Structure parameter (1 = fully structured, 0 = broken)
- :math:`\tau_thix`: Thixotropic rebuilding time scale [s]
- :math:`\Gamma`: Structural breakdown coefficient [-]
- :math:`\alpha` ∈ (0, 1): Fractional order (memory strength)

**Physical interpretation of terms:**

- **Build-up** (1-:math:`\lambda`)/:math:`\tau_thix`: Structure recovers toward :math:`\lambda` = 1 via Brownian motion
- **Breakdown** :math:`\Gamma \cdot \lambda \cdot |\dot{\gamma}^p|`: Shear breaks structure proportional to current structure and flow rate

**Contrast with MIKH:** In MIKH (:math:`\alpha` = 1), this becomes :math:`d\lambda/dt` = ..., giving exponential
dynamics. The fractional derivative introduces memory: the rate of structure
change depends not just on current state, but on the entire history of :math:`\lambda`.

Thermokinematic Coupling
~~~~~~~~~~~~~~~~~~~~~~~~

For thermal-coupled FIKH (``include_thermal=True``):

**Temperature-dependent yield stress:**

.. math::

   \sigma_y(\lambda, T) = (\sigma_{y0} + \Delta\sigma_y \cdot \lambda^{m_y}) \exp\left(\frac{E_y}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)

**Arrhenius viscosity:**

.. math::

   \eta(T) = \eta_{ref} \exp\left(\frac{E_a}{R}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)\right)

**Temperature evolution:**

.. math::

   \rho c_p \dot{T} = \chi \sigma \dot{\gamma}^p - h(T - T_{env})

Where:

- :math:`\chi` ≈ 0.9: Taylor-Quinney coefficient (fraction of plastic work → heat)
- h: Heat transfer coefficient [W/(:math:`m^2 \cdot K`)]
- T_env: Environmental temperature [K]

This coupling enables prediction of **thermal runaway** at high shear rates
and **thermal droop** in flow curves.


Mathematical Formulation
------------------------

Core Equations
~~~~~~~~~~~~~~

**Stress decomposition:**

.. math::

   \sigma_{total} = \sigma + \eta_{\infty} \dot{\gamma}

Total stress = elasto-plastic contribution + viscous background.

**Yield condition:**

.. math::

   f = |\xi| - \sigma_y(\lambda, T) \leq 0 \quad \text{where} \quad \xi = \sigma - C \cdot A

Material yields when relative stress :math:`|\xi|` exceeds current yield stress.

**Plastic flow rule (Perzyna regularization):**

.. math::

   \dot{\gamma}^p = \frac{\langle f \rangle}{\mu_p} \cdot \text{sign}(\xi)

where ⟨·⟩ = max(·, 0). Small :math:`\mu_p` gives rate-independent plasticity.

State Variables
~~~~~~~~~~~~~~~

The FIKH state vector is:

.. code-block:: text

   y = [σ, A, λ, (T if thermal)]

   Dimension: 3 (base) or 4 (thermal)
   ────────────────────────────────────
   y[0] = σ : deviatoric stress [Pa]
   y[1] = A : backstress internal variable [-]
   y[2] = λ : structure parameter [-]
   y[3] = T : temperature [K] (if include_thermal=True)

Plus a **history buffer** for the fractional derivative:

.. code-block:: text

   λ_history = [λ_{n-N+1}, ..., λ_{n-1}, λ_n]  # Shape: (n_history,)

The history buffer size N determines the memory truncation length and
accuracy of the L1 scheme approximation.

Dimensionless Groups
~~~~~~~~~~~~~~~~~~~~

**Fractional Weissenberg Number (** :math:`Wi_{\alpha}` **):**

.. math::

   Wi_\alpha = \dot{\gamma} \cdot \tau_{thix}^{1/\alpha}

Ratio of shear rate to the effective (fractional) structure buildup rate.
Note the 1/:math:`\alpha` exponent reflecting fractional time scaling.

**Deborah Number (De):**

.. math::

   De = \frac{\eta/G}{t_{exp}}

Ratio of Maxwell relaxation time to experimental time scale.

**Memory Number (Me):**

.. math::

   Me = \frac{t_{exp}}{\tau_{thix}} \cdot (t_{exp}/\tau_{thix})^{1-\alpha}

Characterizes the importance of memory effects. Me ≫ 1 means strong
history dependence; Me ≪ 1 approaches Markovian (memoryless) limit.


Governing Equations
-------------------

The complete FIKH system:

**Stress evolution:**

.. math::

   \frac{d\sigma}{dt} = G(\dot{\gamma} - \dot{\gamma}^p) - \frac{G}{\eta}\sigma

**Backstress evolution:**

.. math::

   \frac{dA}{dt} = \dot{\gamma}^p - q|A|^{m-1}A|\dot{\gamma}^p|

**Fractional structure evolution:**

.. math::

   D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}} - \Gamma\lambda|\dot{\gamma}^p|

**Plastic flow rate:**

.. math::

   \dot{\gamma}^p = \frac{\langle |\sigma - C \cdot A| - \sigma_y(\lambda) \rangle}{\mu_p} \cdot \text{sign}(\sigma - C \cdot A)

**Temperature evolution** (if thermal):

.. math::

   \rho c_p \frac{dT}{dt} = \chi \sigma \dot{\gamma}^p - h(T - T_{env})


Protocol Equations
------------------

Steady-State Flow Curve
^^^^^^^^^^^^^^^^^^^^^^^

At steady state with constant :math:`\dot{\gamma}`, the Caputo derivative of a constant is zero:

.. math::

   0 = \frac{1-\lambda_{ss}}{\tau_{thix}} - \Gamma \lambda_{ss} |\dot{\gamma}|

**Equilibrium structure:**

.. math::

   \lambda_{ss}(\dot{\gamma}) = \frac{1}{1 + \Gamma \tau_{thix} |\dot{\gamma}|}

**Note**: The flow curve shape is identical to MIKH at constant temperature.
Fractional effects appear only in *transients*. Thermal coupling causes
"thermal droop" at high rates due to shear heating.

Start-up of Steady Shear
^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: :math:`\dot{\gamma}(t) = \dot{\gamma}_0 \cdot H(t)`, starting from :math:`\lambda(0)` = 1 (aged sample)

**Key difference from MIKH**: Structure breakdown follows **Mittag-Leffler decay**:

.. math::

   \lambda(t) \approx \lambda_{ss} + (\lambda_0 - \lambda_{ss}) E_\alpha\left(-\left(\frac{t}{\tau_{eff}}\right)^\alpha\right)

where :math:`\tau_eff = \tau_thix / (1 + \Gamma \cdot \tau_thix \cdot |\dot{\gamma}|)`.

**Signatures:**

- Broader stress overshoot than MIKH
- Slower approach to steady state ("long tail")
- Power-law decay at long times, not exponential

Stress Relaxation
^^^^^^^^^^^^^^^^^

**Protocol**: Step strain :math:`\gamma_0` at t = 0, then :math:`\dot{\gamma}` = 0

For strains below yield, structure rebuilds according to:

.. math::

   D_t^\alpha \lambda = \frac{1-\lambda}{\tau_{thix}}

**Solution:**

.. math::

   \lambda(t) = 1 - (1-\lambda_0) E_\alpha\left(-\left(\frac{t}{\tau_{thix}}\right)^\alpha\right)

**Asymptotics:**

- Short time: :math:`\lambda(t) \approx \lambda_0 + (1-\lambda_0)(t/\tau)^\alpha / \Gamma(1+\alpha)`
- Long time: :math:`\lambda(t) \approx 1 - (1-\lambda_0)(\tau/t)^\alpha / \Gamma(1-\alpha)` (power-law approach)

The stress may **increase** during relaxation (anti-thixotropic recovery) as
:math:`\lambda` → 1 increases the modulus.

Creep (Step Stress)
^^^^^^^^^^^^^^^^^^^

**Protocol**: Constant stress :math:`\sigma_0` applied at t = 0

**Below fully-structured yield** (:math:`\sigma_0 < \sigma_y(\lambda=1`, T)):
Only elastic deformation initially. Structure may slowly break due to
thermal fluctuations, leading to **delayed yielding**.

**Intermediate stress** (:math:`\sigma_y(\lambda=0) < \sigma_0 < \sigma_y(\lambda=1)`):
**Viscosity bifurcation** — delay followed by avalanche-like yielding:

.. math::

   t_d \sim \tau_{thix} \left(\frac{\sigma_y(1) - \sigma_0}{\sigma_0}\right)^{1/\alpha}

The 1/:math:`\alpha` exponent means **smaller** :math:`\alpha` **gives longer delays** (stronger memory
resists yielding).

**Above yield**: Immediate plastic flow with potential **thermal runaway** if
dissipation exceeds heat loss.

SAOS (Small Amplitude Oscillatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: :math:`\gamma(t) = \gamma_0 \cdot sin(\omegat), \gamma_0` ≪ 1 (linear regime)

For fractional viscoelastic response at equilibrium :math:`\lambda_eq`, the complex modulus
follows a **fractional Maxwell** form:

.. math::

   G^*(\omega) = G_0 \frac{(i\omega\tau)^\alpha}{1 + (i\omega\tau)^\alpha}

**Storage modulus:**

.. math::

   G'(\omega) = G_0 \frac{(\omega\tau)^{2\alpha} + (\omega\tau)^\alpha \cos(\pi\alpha/2)}{1 + 2(\omega\tau)^\alpha \cos(\pi\alpha/2) + (\omega\tau)^{2\alpha}}

**Loss modulus:**

.. math::

   G''(\omega) = G_0 \frac{(\omega\tau)^\alpha \sin(\pi\alpha/2)}{1 + 2(\omega\tau)^\alpha \cos(\pi\alpha/2) + (\omega\tau)^{2\alpha}}

**Cole-Cole signature**: Depressed arc with depression angle :math:`\theta = (1-\alpha)\pi/2`.

LAOS (Large Amplitude Oscillatory)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Protocol**: :math:`\gamma(t) = \gamma_0 \cdot sin(\omegat), \gamma_0` finite (nonlinear, may yield)

Full coupled system requiring numerical integration. The fractional memory
introduces:

- Power-law decay of higher harmonics
- Asymmetric Lissajous figures from back-stress
- Delayed yielding within cycle (fractional delay)
- Cycle-by-cycle thermal softening at high amplitude/frequency


What You Can Learn
------------------

This section explains what physical insights can be extracted from FIKH model
parameters after fitting to experimental data.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**Fractional Order** :math:`\alpha` **:**

- :math:`\alpha` **≈ 0.9-1.0**: Material behaves like classical thixotropic fluid (single timescale)
- :math:`\alpha` **≈ 0.5-0.7**: Broad distribution of restructuring timescales (typical for gels)
- :math:`\alpha` **< 0.3**: "Glassy" dynamics with very long memory (aging-dominated)

*Physical interpretation*: :math:`\alpha` relates to the breadth of the relaxation spectrum.
From Cole-Cole analysis, depression angle :math:`\theta = (1-\alpha)\pi/2`. Materials with :math:`\alpha` ≈ 0.5
show 45° depression, indicating a very broad spectrum.

*Practical measurement*: Fit Cole-Cole plot from frequency sweep, or fit
recovery data to stretched exponential (:math:`\beta \approx \alpha` for moderate stretching).

**Thixotropic Timescale** :math:`\tau_thix` **:**

- Sets the characteristic time for structure recovery at rest
- Larger :math:`\tau_thix` → slower rebuilding → more thixotropic
- Compare to process timescales: :math:`\tau_thix` >> t_process means structure won't recover
- Typical values: 1-1000 s for industrial fluids

**Breakdown Coefficient** :math:`\Gamma` **:**

- Rate of structure destruction under flow
- :math:`\Gamma \cdot \tau_thix` product determines equilibrium structure at given shear rate
- :math:`\lambda_eq = 1/(1 + \Gamma \cdot \tau_thix \cdot |\dot{\gamma}|)`
- Critical shear rate: :math:`\dot{\gamma}_crit = 1/(\Gamma \cdot \tau_thix) where \lambda` drops to 0.5

**Yield Stress Parameters:**

- :math:`\sigma_y0`: Residual yield when fully destructured (:math:`\lambda` = 0)
- :math:`\Delta\sigma_y`: Additional yield from structure (total yield at :math:`\lambda=1 is \sigma_y0 + \Delta\sigma_y`)
- Ratio :math:`\Delta\sigma_y/\sigma_y0` indicates how much structure contributes to yielding

**Thermal Parameters:**

- **E_a**: Viscosity activation energy [J/mol]. Typical: 20-50 kJ/mol for polymers
- **E_y**: Yield activation energy. Lower E_y → yield is less temperature-sensitive
- Higher E_a/E_y ratio → viscosity drops faster than yield with temperature

Derived Quantities
~~~~~~~~~~~~~~~~~~

**From Flow Curve Fit:**

.. code-block:: python

   # Critical shear rate where λ drops to 0.5
   gamma_dot_crit = 1 / (Gamma * tau_thix)

   # Viscosity ratio (structured / broken)
   eta_ratio = (sigma_y0 + delta_sigma_y) / sigma_y0

   # Structure number (dimensionless thixotropic intensity)
   Str = Gamma * tau_thix * gamma_dot_applied

**From Creep Analysis:**

.. code-block:: python

   # Delay time for yielding at stress σ₀ (fractional)
   sigma_y_full = sigma_y0 + delta_sigma_y
   t_delay = tau_thix * ((sigma_y_full - sigma_0) / sigma_0)**(1/alpha)

**From Frequency Sweep:**

.. code-block:: python

   import numpy as np

   # Cole-Cole depression angle
   theta = (1 - alpha) * np.pi / 2  # radians

   # Crossover frequency (where G' = G'')
   omega_c = 1 / tau_maxwell  # where tau_maxwell = eta / G

**From Startup Overshoot:**

.. code-block:: python

   # Peak strain (approximate)
   gamma_peak = sigma_y_full / G

   # Overshoot ratio
   overshoot_ratio = sigma_peak / sigma_steady

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from FIKH Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - :math:`\alpha > 0.8, \tau_thix` < 10 s
     - Near-exponential, fast recovery
     - Simple gels, light emulsions
     - Consider using MIKH instead
   * - :math:`\alpha` = 0.5-0.8, :math:`\tau_thix` = 10-100 s
     - Moderate memory, stretched recovery
     - Waxy crude oils, drilling muds
     - Standard FIKH application range
   * - :math:`\alpha < 0.5, \tau_thix` > 100 s
     - Strong memory, slow glassy dynamics
     - Aging glasses, cement pastes
     - Long history buffer needed (n_history > 500)
   * - High E_a (> 50 kJ/mol)
     - Strong temperature sensitivity
     - Waxy crude oils, polymers
     - Consider thermal coupling critical

Process Design Insights
~~~~~~~~~~~~~~~~~~~~~~~

**Pipeline Restart** (waxy crude):

- Fit FIKH to startup data at T_restart
- Compute restart pressure from steady-state stress at desired flow rate
- Account for thermal effects: higher h → less softening → higher restart pressure
- Use delay time formula to estimate how long the pipeline can sit before gelation

**Product Stability:**

- Compare :math:`\tau_thix` to shelf-life timescales
- If :math:`\tau_thix` >> shelf_time → product remains broken (poor stability)
- Recommendation: :math:`\tau_thix` < 0.1 × expected_storage_time for structure recovery
- Lower :math:`\alpha` (stronger memory) means slower recovery but more stable once recovered

**Processing Windows:**

- Critical shear rate :math:`\dot{\gamma}_crit = 1/(\Gamma \cdot \tau_thix)` defines transition from structured to flowing
- Design mixers/pumps to operate above :math:`\dot{\gamma}_crit` for easy flow
- Storage vessels should see :math:`\dot{\gamma} << \dot{\gamma}_crit` for structure recovery
- Temperature control: higher T → lower yield → easier processing but more energy

**Quality Control:**

- Monitor :math:`\alpha` over production batches: consistent :math:`\alpha` indicates consistent microstructure
- Track :math:`\tau_thix`: increasing :math:`\tau_thix` may indicate contamination or degradation
- Plot (:math:`\alpha, \tau_thix`) phase space to identify batch-to-batch variability


Industrial Applications
-----------------------

Waxy Crude Oil Pipeline Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The FIKH model is particularly suited for waxy crude oils where the wax crystal
network exhibits power-law memory from hierarchical structure.

**Pipeline Restart After Shutdown:**

When a pipeline shuts down, wax precipitates and forms a gel network with
complex relaxation dynamics. Key parameter ranges:

- :math:`\alpha` **= 0.5-0.7**: Power-law recovery from hierarchical crystal structure
- :math:`\tau_thix` **= 100-10,000 s**: Long aging times for gelled pipelines
- :math:`\sigma_y,0 + \Delta\sigma_y` **= 50-500 Pa**: Gel strength depends on cooling rate and rest time

**Engineering implications:**

- Restart pressure scales with :math:`\sigma_y(t_rest)` where t_rest follows fractional kinetics
- The 1/:math:`\alpha` exponent in delay time formula means :math:`\alpha` = 0.5 gives **4× longer delays**
  than :math:`\alpha` = 1 for same stress ratio
- Monitor thermokinematic memory: temperature cycling affects :math:`\alpha` through wax morphology

Drilling Fluids
~~~~~~~~~~~~~~~

Water-based drilling fluids with clay platelets exhibit fractional behavior from
distributed particle-particle bond energies:

- :math:`\alpha` **= 0.4-0.7**: Depending on clay type and concentration
- :math:`\tau_thix` **= 1-100 s**: Faster than crude oils due to smaller particles
- **Thermal coupling**: Critical for deep wells (high T)

**Borehole stability:**

- Fractional recovery means gel strength continues building over **longer times**
  than exponential models predict
- API gel tests at 10 sec vs 10 min can show continued strength gain even after
  10 min (power-law tail)

Greases and Lubricants
~~~~~~~~~~~~~~~~~~~~~~

Greases exhibit fractional behavior from thickener fiber networks:

- :math:`\alpha` **= 0.5-0.7**: From distributed fiber entanglement dynamics
- **Thermal coupling**: Critical for high-speed bearing applications

**Bearing startup:**

- Fractional memory means cold-start torque depends on entire thermal history
- Use FIKH with thermal coupling to predict temperature-dependent startup behavior


Parameters
----------

Base Parameters (Always Present)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``G``
     - 1000
     - (0.1, :math:`10^9`)
     - Pa
     - Shear modulus
   * - ``eta``
     - :math:`10^6`
     - (10\ :math:`^{-3, 10^1^2}`)
     - Pa·s
     - Maxwell viscosity (:math:`\tau = \eta/G`)
   * - ``C``
     - 500
     - (0, :math:`10^9`)
     - Pa
     - Kinematic hardening modulus
   * - ``gamma_dyn``
     - 1.0
     - (0, :math:`10^4`)
     - —
     - Dynamic recovery parameter
   * - ``m``
     - 1.0
     - (0.5, 3)
     - —
     - AF recovery exponent
   * - ``sigma_y0``
     - 10
     - (0, :math:`10^9`)
     - Pa
     - Minimal yield stress (destructured)
   * - ``delta_sigma_y``
     - 50
     - (0, :math:`10^9`)
     - Pa
     - Structural yield stress contribution
   * - ``tau_thix``
     - 1.0
     - (10\ :math:`^{-6, 10^1^2}`)
     - s
     - Thixotropic rebuilding time
   * - ``Gamma``
     - 0.5
     - (0, :math:`10^4`)
     - —
     - Structural breakdown coefficient
   * - ``alpha_structure``
     - 0.5
     - (0.05, 0.99)
     - —
     - Fractional order for structure
   * - ``eta_inf``
     - 0.1
     - (0, :math:`10^9`)
     - Pa·s
     - High-shear (solvent) viscosity
   * - ``mu_p``
     - :math:`10 \times 10^{-3}`
     - (10\ :math:`^{-9, 10^3}`)
     - Pa·s
     - Plastic viscosity (Perzyna)

Thermal Parameters (``include_thermal=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``T_ref``
     - 298.15
     - (200, 500)
     - K
     - Reference temperature
   * - ``E_a``
     - 50000
     - (0, :math:`2 \times 10^5`)
     - J/mol
     - Viscosity activation energy
   * - ``E_y``
     - 30000
     - (0, :math:`2 \times 10^5`)
     - J/mol
     - Yield stress activation energy
   * - ``m_y``
     - 1.0
     - (0.5, 2)
     - —
     - Structure exponent for yield
   * - ``rho_cp``
     - :math:`4 \times 10^6`
     - (:math:`10^5, 10^8`)
     - J/(:math:`m^3 \cdot K`)
     - Volumetric heat capacity
   * - ``chi``
     - 0.9
     - (0, 1)
     - —
     - Taylor-Quinney coefficient
   * - ``h``
     - 100
     - (0, :math:`10^6`)
     - W/(:math:`m^2 \cdot K`)
     - Heat transfer coefficient
   * - ``T_env``
     - 298.15
     - (200, 500)
     - K
     - Environmental temperature

Isotropic Hardening Parameters (``include_isotropic_hardening=True``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 15 12 15 15 43
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``Q_iso``
     - 0
     - (0, :math:`10^9`)
     - Pa
     - Isotropic hardening saturation
   * - ``b_iso``
     - 1.0
     - (0, 100)
     - —
     - Isotropic hardening rate


Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

1. **Estimate** :math:`\alpha` **from Cole-Cole**: Fit frequency sweep, measure depression angle :math:`\theta`,
   compute :math:`\alpha = 1 - 2\theta/\pi`
2. **Flow curve first**: Fit :math:`\sigma_y0, \Delta\sigma_y, \tau_thix, \Gamma, \eta_inf` from steady-state data
3. **Startup second**: Fix flow curve params, fit G, C, :math:`\gamma_dyn` from transient
4. **Relaxation/creep**: Fine-tune :math:`\eta` (Maxwell viscosity) and verify :math:`\alpha`

:math:`\alpha` **estimation from recovery data:**

.. code-block:: python

   import numpy as np
   from scipy.optimize import curve_fit

   def stretched_exp(t, tau_c, beta):
       return 1 - np.exp(-(t/tau_c)**beta)

   # Fit stretched exponential to recovery
   popt, _ = curve_fit(stretched_exp, t_recovery, lambda_recovery,
                       p0=[10.0, 0.7], bounds=([0.1, 0.1], [1000, 1.0]))
   tau_c, beta = popt

   # β ≈ α for moderate stretching
   alpha_estimate = beta

Protocol Selection
~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Protocol
     - Best for
   * - ``flow_curve``
     - Steady-state parameters (:math:`\sigma_y0, \Delta\sigma_y, \eta_inf`)
   * - ``startup``
     - Elasticity (G), hardening (C, :math:`\gamma_dyn`), transient :math:`\alpha` effects
   * - ``relaxation``
     - Maxwell viscosity (:math:`\eta`), :math:`\alpha` verification from power-law tail
   * - ``creep``
     - Combined viscoelastic-plastic, delayed yielding (:math:`\alpha-dependent`)
   * - ``oscillation``
     - Cole-Cole depression (:math:`\alpha`), crossover frequency
   * - ``laos``
     - Full nonlinear characterization, harmonic content

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Issue
     - Solution
   * - Poor fit quality (low :math:`R^2`)
     - Check test_mode matches data; try different initial :math:`\alpha` (0.3, 0.5, 0.7)
   * - Recovery too fast
     - Decrease :math:`\alpha` (stronger memory slows recovery)
   * - Long-time power-law not captured
     - Increase n_history; verify :math:`\alpha` < 1
   * - MCMC convergence (R-hat > 1.1)
     - Use NLSQ warm-start; increase num_warmup; check :math:`\alpha` prior bounds
   * - Numerical instabilities (NaN)
     - Check :math:`\tau_thix` > 1e-6; reduce dt or increase n_history for small :math:`\alpha`
   * - Slow computation
     - Use model.precompile(); reduce n_history if :math:`\alpha` > 0.7; consider MIKH if :math:`\alpha` ≈ 1


Parameter Estimation Methods
----------------------------

Sequential Fitting Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sequential approach exploits the separation of timescales:

**Stage 1: Flow Curve (Steady State)**

From flow curve :math:`\sigma(\dot{\gamma})`, fit steady-state parameters (:math:`\alpha` does not affect steady state):

.. code-block:: python

   from rheojax.models.fikh import FIKH

   model = FIKH(alpha_structure=0.5)  # α doesn't affect flow curve

   # Fix elastic/hardening params, fit thixotropic
   model.parameters.freeze(['G', 'C', 'gamma_dyn', 'eta', 'mu_p'])
   model.fit(gamma_dot, sigma_ss, test_mode='flow_curve')

**Stage 2: Startup Transients (** :math:`\alpha` **matters here)**

From startup stress overshoot :math:`\sigma(t`; :math:`\dot{\gamma}_0`), fit G and refine :math:`\alpha`:

.. code-block:: python

   # Unfreeze elastic parameters and α
   model.parameters.unfreeze(['G', 'C', 'gamma_dyn', 'alpha_structure'])
   model.fit(t_startup, sigma_startup, test_mode='startup')

**Stage 3: Relaxation (** :math:`\alpha` **verification)**

From stress relaxation, verify :math:`\alpha` from power-law tail:

.. code-block:: python

   model.parameters.unfreeze(['eta', 'mu_p'])
   model.fit(t_relax, sigma_relax, test_mode='relaxation')

Bayesian Inference with MCMC
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For uncertainty quantification, use NumPyro NUTS with NLSQ warm-start:

.. code-block:: python

   # Stage 1: Point estimate
   model.fit(data)

   # Stage 2: Bayesian inference
   result = model.fit_bayesian(
       data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Check convergence
   print(f"R-hat: {result.r_hat}")   # Target: <1.01
   print(f"ESS: {result.ess}")       # Target: >400

**Prior Selection for** :math:`\alpha` **:**

.. code-block:: python

   # Recommended: Beta distribution centered on expected α
   # α ~ Beta(a, b) with mode at (a-1)/(a+b-2)

   # For α expected around 0.6:
   # α ~ Beta(3, 2) gives mode at 0.67, mean at 0.6

   # Or use truncated normal:
   # α ~ TruncatedNormal(0.5, 0.2, low=0.1, high=0.95)


Numerical Implementation
------------------------

The Caputo fractional derivative is discretized using the **L1 scheme**:

.. math::

   D_t^\alpha \lambda_n \approx \frac{1}{\Gamma(2-\alpha) \Delta t^\alpha} \sum_{k=0}^{n-1} b_k (\lambda_{n-k} - \lambda_{n-k-1})

Where b_k = (k+1)^(1-:math:`\alpha`) - k^(1-:math:`\alpha`).

L1 Scheme Details
~~~~~~~~~~~~~~~~~

The L1 scheme is a first-order accurate discretization of the Caputo derivative:

**Error bounds:**

.. math::

   \|D^{\alpha}\lambda - D^{\alpha}_h\lambda\| \leq C \cdot h^{2-\alpha}

where h = :math:`\Deltat`. **Important**: Lower :math:`\alpha` requires finer time steps for same accuracy.

**Coefficient computation** (precomputed for efficiency):

.. code-block:: python

   def compute_l1_coefficients(alpha, n):
       """L1 scheme coefficients b_k = (k+1)^(1-α) - k^(1-α)"""
       k = jnp.arange(n)
       return jnp.power(k + 1, 1 - alpha) - jnp.power(k, 1 - alpha)

History Buffer Management
~~~~~~~~~~~~~~~~~~~~~~~~~

The FIKH model uses a **fixed-window history buffer** (short-memory truncation)
rather than storing the entire history:

.. code-block:: text

   λ_history = [λ_{n-N+1}, ..., λ_{n-1}, λ_n]  # Rolling window of N points

**Recommended n_history Selection:**

.. list-table::
   :widths: 20 25 20 35
   :header-rows: 1

   * - :math:`\alpha` Range
     - Recommended n_history
     - Memory Usage
     - Accuracy
   * - 0.7 - 0.99
     - 50-100
     - ~400 bytes
     - Good (fast convergence)
   * - 0.4 - 0.7
     - 100-500
     - ~4 KB
     - Good
   * - 0.05 - 0.4
     - 500-1000
     - ~8 KB
     - Adequate (strong memory)

**When to Increase n_history:**

- Long simulations (t >> :math:`\tau_thix`)
- Very small :math:`\alpha` (< 0.3)
- Accuracy-critical applications
- Oscillatory protocols (LAOS)

Computational Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Memory**: O(N) history storage via fixed-window buffer
- **Cost**: O(n_history) per step (vs O(:math:`N^2`) for naive full-history convolution)
- **JIT Compilation**: First call triggers ~1-5s compilation; subsequent calls fast

**Precompilation** (optional):

.. code-block:: python

   # Trigger JIT compilation before production runs
   compile_time = model.precompile()
   print(f"Compiled in {compile_time:.1f}s")


Usage Examples
--------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models.fikh import FIKH

   # Initialize model with fractional order
   model = FIKH(alpha_structure=0.6, include_thermal=False)

   # Set parameters
   model.parameters.set_value("G", 1000.0)
   model.parameters.set_value("sigma_y0", 20.0)
   model.parameters.set_value("tau_thix", 50.0)
   model.parameters.set_value("Gamma", 0.5)

Flow Curve
~~~~~~~~~~

.. code-block:: python

   # Predict steady-state flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model.predict_flow_curve(gamma_dot)

   # Note: Flow curve is independent of α (steady state)

Startup Shear
~~~~~~~~~~~~~

.. code-block:: python

   # Predict startup response (α affects overshoot shape)
   t = np.linspace(0, 50, 500)
   sigma_startup = model.predict_startup(t, gamma_dot=1.0, lambda_init=1.0)

   # Compare different α values
   for alpha in [0.3, 0.5, 0.7, 0.9]:
       model_alpha = FIKH(alpha_structure=alpha)
       sigma = model_alpha.predict_startup(t, gamma_dot=1.0)
       # Lower α → broader overshoot, longer tail

Stress Relaxation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict relaxation with Mittag-Leffler decay
   t = np.linspace(0, 200, 500)
   sigma_relax = model.predict_relaxation(t, sigma_0=100.0, lambda_init=0.5)

   # Long-time tail follows power law: σ ~ t^(-α)

Creep with Delayed Yielding
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Predict creep under constant stress
   t = np.linspace(0, 500, 500)
   strain = model.predict_creep(t, sigma_applied=45.0, lambda_init=1.0)

   # Delayed yielding: lower α → longer delay before avalanche

Comparing :math:`\alpha` Effects
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from rheojax.models.fikh import FIKH

   t = np.linspace(0, 100, 500)

   plt.figure(figsize=(10, 4))

   for alpha in [0.3, 0.5, 0.7, 0.9]:
       model = FIKH(alpha_structure=alpha, include_thermal=False)
       model.parameters.set_value("tau_thix", 10.0)
       model.parameters.set_value("G", 1000.0)

       # Recovery at rest
       result = model.predict_relaxation(t, sigma_0=50.0, lambda_init=0.2)
       plt.plot(t, result, label=f'α = {alpha}')

   plt.xlabel('Time [s]')
   plt.ylabel('Stress [Pa]')
   plt.legend()
   plt.title('FIKH Relaxation: Effect of Fractional Order α')

Thermal Coupling
~~~~~~~~~~~~~~~~

.. code-block:: python

   # FIKH with thermal effects
   model = FIKH(include_thermal=True, alpha_structure=0.6)

   model.parameters.set_value("E_a", 50000.0)   # J/mol
   model.parameters.set_value("E_y", 30000.0)   # J/mol
   model.parameters.set_value("T_ref", 300.0)   # K
   model.parameters.set_value("chi", 0.9)       # Taylor-Quinney
   model.parameters.set_value("h", 100.0)       # Heat transfer

   # Startup at different temperatures
   for T_init in [280, 300, 320]:
       sigma = model.predict_startup(t, gamma_dot=10.0, T_init=T_init)

Bayesian Fitting
~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models.fikh import FIKH

   model = FIKH(alpha_structure=0.5, include_thermal=True)

   # Point estimate first (critical for MCMC)
   model.fit(t, stress_data, test_mode='startup')

   # Bayesian inference
   result = model.fit_bayesian(
       t, stress_data,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Credible intervals
   intervals = model.get_credible_intervals(
       result.posterior_samples,
       credibility=0.95
   )


Limiting Behavior
-----------------

:math:`\alpha` **→ 1 (Classical Limit):**

- Recovers exponential MIKH behavior
- Mittag-Leffler :math:`E_1(-x)` = exp(-x)
- Use standard MIKH for computational efficiency

:math:`\alpha` **→ 0 (Strong Memory Limit):**

- Very slow power-law relaxation
- Long "memory" of deformation history
- Numerical challenges (need long history buffer)
- Consider if material is truly "glassy" (SGR may be more appropriate)

**Thermal Coupling Effects:**

- High E_a: Strong temperature sensitivity
- Low h: Poor heat dissipation → thermal runaway risk
- :math:`\chi` ~ 0.9: Most plastic work converts to heat


Relation to Other Models
------------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Model
     - Relationship to FIKH
   * - :doc:`../ikh/mikh`
     - FIKH reduces to MIKH as :math:`\alpha` → 1 (exponential kinetics)
   * - :doc:`fmlikh`
     - Multi-mode extension with N fractional modes
   * - :doc:`../ikh/ml_ikh`
     - Integer-order multi-mode; similar multi-timescale physics, different math
   * - :doc:`../sgr/index`
     - Alternative for glassy systems; SGR uses trap depths, FIKH uses fractional order
   * - Fractional Maxwell
     - FIKH = Fractional Maxwell + plasticity + thixotropy


References
----------

**Fractional Calculus:**

.. [1] Podlubny, I. (1999). *Fractional Differential Equations*. Academic Press.

.. [2] Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*.
   Imperial College Press.

.. [3] Diethelm, K. (2010). *The Analysis of Fractional Differential Equations*.
   Springer. https://doi.org/10.1007/978-3-642-14574-2

**Numerical Methods:**

.. [4] Li, C. and Zeng, F. (2015). *Numerical Methods for Fractional Calculus*.
   CRC Press.

**Fractional Rheology:**

.. [5] Jaishankar, A. and McKinley, G. H. (2013). "Power-law rheology in the bulk
   and at the interface: quasi-properties and fractional constitutive equations."
   *Proc. R. Soc. A*, 469(2149), 20120284.

.. [6] Jaishankar, A. and McKinley, G. H. (2014). "A fractional K-BKZ constitutive
   formulation for describing the nonlinear rheology of multiscale complex fluids."
   *J. Rheol.*, 58, 1751-1788.

**IKH Foundation:**

.. [7] Dimitriou, C. J. and McKinley, G. H. (2014). "A comprehensive constitutive
   law for waxy crude oil: a thixotropic yield stress fluid." *Soft Matter*,
   10, 6619-6644.

.. [8] Geri, M., Venkatesan, R., Sambath, K., and McKinley, G. H. (2017).
   "Thermokinematic memory and the thixotropic elasto-viscoplasticity of
   waxy crude oils." *J. Rheol.*, 61(3), 427-454.

**Thixotropic Modeling:**

.. [9] de Souza Mendes, P. R. and Thompson, R. L. (2019). "Time-dependent yield
   stress materials." *Curr. Opin. Colloid Interface Sci.*, 43, 15-25.

.. [10] Larson, R. G. and Wei, Y. (2019). "A review of thixotropy and its
   rheological modeling." *J. Rheol.*, 63(3), 477-501.


See Also
--------

- :doc:`fmlikh` — Multi-mode fractional IKH for hierarchical structures
- :doc:`../ikh/mikh` — Classical IKH model (:math:`\alpha` = 1 limit)
- :doc:`../ikh/ml_ikh` — Integer-order multi-mode IKH
- :doc:`../sgr/index` — Soft Glassy Rheology (alternative for aging systems)
- :doc:`/user_guide/03_advanced_topics/index` — Advanced thixotropic modeling
