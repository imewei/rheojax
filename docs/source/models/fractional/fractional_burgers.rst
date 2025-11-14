.. _model-fractional-burgers:

Fractional Burgers Model (Fractional)
=====================================

Quick Reference
---------------

**Use when:** Complex creep with glassy compliance, fractional retardation, and viscous flow
**Parameters:** 5 (Jg, Jk, α, τk, η₁)
**Key equation:** :math:`J(t) = J_g + \frac{t^{\alpha}}{\eta_1\Gamma(1+\alpha)} + J_k[1 - E_{\alpha}(-(t/\tau_k)^{\alpha})]`
**Test modes:** Creep, oscillation
**Material examples:** Polymer composites, asphalt binders, bituminous materials, viscoelastic solids under load

.. seealso::
   :doc:`/user_guide/fractional_viscoelasticity_reference` — Mathematical foundations of fractional calculus, SpringPot element, Mittag-Leffler functions, and physical meaning of fractional order α.

Overview
--------

The **Fractional Burgers Model** combines a **Maxwell element** in series with a **Fractional Kelvin-Voigt element**, creating a five-parameter model that captures **glassy compliance, viscous flow, and fractional retardation** in a single compact framework. This model extends the classical four-element Burgers model by replacing the Kelvin-Voigt dashpot with a SpringPot, enabling power-law retardation instead of exponential relaxation.

The Fractional Burgers model is particularly effective for materials exhibiting **complex creep behavior** with both instantaneous elastic response, delayed fractional retardation, and long-term viscous flow. Common applications include **polymer composites under load, asphalt binders, bituminous materials, and viscoelastic solids** undergoing time-dependent deformation.

**Mechanical Analogue:**

.. code-block:: text

   [Maxwell Arm: Spring Gg + Dashpot η1] ---- series ---- [Fractional KV: Spring + SpringPot (Jk, α, τk)]

Physical Foundation
-------------------

The Fractional Burgers model combines three distinct mechanical responses:

1. **Instantaneous elastic response** (glassy compliance :math:`J_g`)
2. **Fractional retardation** (SpringPot in Kelvin arm with time constant :math:`\tau_k`)
3. **Long-term viscous flow** (dashpot viscosity :math:`\eta_1`)

**Microstructural Interpretation:**

- **:math:`J_g`**: Instantaneous bond stretching, glassy modulus
- **Fractional KV arm**: Distributed retardation from hierarchical polymer network rearrangements
- **Maxwell dashpot**: Irreversible chain flow, reptation, or permanent deformation

Governing Equations
-------------------

**Time Domain (Creep Compliance):**

.. math::

   J(t) = J_g + \frac{t^{\alpha}}{\eta_1\,\Gamma(1+\alpha)} + J_k\left[1 - E_{\alpha}\!\left(-\left(\frac{t}{\tau_k}\right)^{\alpha}\right)\right]

where :math:`E_{\alpha}(z)` is the **one-parameter Mittag-Leffler function**:

.. math::

   E_{\alpha}(z) = \sum_{k=0}^{\infty} \frac{z^k}{\Gamma(\alpha k + 1)}

**Frequency Domain (Complex Compliance):**

.. math::

   J^{*}(\omega) = J_g + \frac{(i\omega)^{-\alpha}}{\eta_1\,\Gamma(1-\alpha)} + \frac{J_k}{1 + (i\omega\tau_k)^{\alpha}}

**Complex Modulus:**

.. math::

   G^{*}(\omega) = \frac{1}{J^{*}(\omega)}

Note: The inversion :math:`G^* = 1/J^*` is exact for linear viscoelastic materials.

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
   * - ``Jg``
     - :math:`J_g`
     - 1/Pa
     - [1e-9, 1e3]
     - Glassy compliance (instantaneous response)
   * - ``eta1``
     - :math:`\eta_1`
     - Pa·s
     - [1e-6, 1e12]
     - Viscosity (Maxwell arm, controls terminal flow)
   * - ``Jk``
     - :math:`J_k`
     - 1/Pa
     - [1e-9, 1e3]
     - Kelvin compliance (retardation magnitude)
   * - ``alpha``
     - :math:`\alpha`
     - dimensionless
     - [0.05, 0.95]
     - Fractional order (0.2-0.7 typical for polymers)
   * - ``tau_k``
     - :math:`\tau_k`
     - s
     - [1e-6, 1e6]
     - Retardation time (characteristic Kelvin timescale)

Regimes and Behavior
--------------------

**Short Time** (:math:`t \ll \tau_k`):

.. math::

   J(t) \approx J_g + \frac{t^{\alpha}}{\eta_1\,\Gamma(1+\alpha)}

Instantaneous glassy compliance plus early-time fractional flow from Maxwell arm.

**Intermediate Time** (:math:`t \sim \tau_k`):

.. math::

   J(t) \approx J_g + J_k\left[1 - E_{\alpha}\!\left(-\left(\frac{t}{\tau_k}\right)^{\alpha}\right)\right]

**Fractional retardation** dominated by Kelvin arm with power-law approach to equilibrium.

**Long Time** (:math:`t \gg \tau_k`):

.. math::

   J(t) \approx J_g + J_k + \frac{t}{\eta_1}

**Unbounded creep** (liquid-like) with constant compliance offset from glassy and Kelvin contributions.

Material Examples
-----------------

**Polymer Composites** (:math:`J_g \approx 10^{-6}-10^{-5}` 1/Pa, :math:`\alpha \approx 0.3-0.5`):

- **Filled elastomers** (carbon black, silica fillers)
- **Fiber-reinforced polymers** under sustained load
- **Polymer nanocomposites** (clay, CNT fillers)

**Asphalt and Bitumen** (:math:`\eta_1 \approx 10^4-10^7` Pa·s, :math:`\alpha \approx 0.4-0.6`):

- **Asphalt concrete** (temperature-dependent)
- **Bituminous binders** for road pavements
- **Roofing materials**

**Food Materials** (:math:`J_k \approx 10^{-4}-10^{-2}`, :math:`\alpha \approx 0.2-0.5`):

- **Cheese** (long-term creep)
- **Dough** (wheat flour, viscoelastic retardation)
- **Semi-solid fats** (margarine, butter)

**Biological Tissues** (:math:`\alpha \approx 0.2-0.4`):

- **Ligaments and tendons** under sustained stress
- **Intervertebral discs** (viscoelastic creep)

Experimental Design
-------------------

**Creep Test (Primary Application):**

1. **Step stress**: Apply constant stress :math:`\sigma_0` within LVR
2. **Time span**: Cover 4-5 decades (e.g., 0.1 s - 10⁴ s)
3. **Sampling**: Log-spaced to capture all three regimes
4. **Analysis**: Fit :math:`J(t)` to identify :math:`J_g` (instantaneous), :math:`J_k` (retardation), :math:`\eta_1` (slope at long time)

**Frequency Sweep (Oscillatory):**

1. **Frequency range**: 0.001-100 rad/s (wide span critical)
2. **Strain amplitude**: Within LVR (0.5-5%)
3. **Analysis**: Fit :math:`G'(\omega)`, :math:`G''(\omega)` simultaneously
4. **Verification**: Check terminal flow region (:math:`G'' \sim \omega`, :math:`G' \sim \omega^2`)

Fitting Strategies
------------------

**Initialization from Creep Data:**

1. **:math:`J_g`**: Extrapolate :math:`J(t \to 0)` (instantaneous compliance)
2. **:math:`\eta_1`**: Slope of :math:`J(t)` at long time → :math:`1/\eta_1`
3. **:math:`J_k`**: Mid-time plateau height minus :math:`J_g`
4. **:math:`\tau_k`**: Time where retardation is half-complete
5. **:math:`\alpha`**: Curvature of retardation region in log-log plot

**Optimization:**

- Use weighted least squares (log-spaced weights)
- Constrain :math:`J_g < J_k` (glassy stiffer than Kelvin)
- Fit in compliance space for creep, modulus space for oscillation
- Verify residuals random across all time/frequency decades

Usage Example
-------------

.. code-block:: python

   from rheojax.models import FractionalBurgersModel
   import numpy as np

   # Create model
   model = FractionalBurgersModel()

   # Set typical parameters for polymer composite
   model.parameters.set_value('Jg', 1e-6)       # 1/Pa
   model.parameters.set_value('eta1', 1e5)      # Pa·s
   model.parameters.set_value('Jk', 5e-6)       # 1/Pa
   model.parameters.set_value('alpha', 0.4)     # dimensionless
   model.parameters.set_value('tau_k', 10.0)    # s

   # Predict creep compliance
   t = np.logspace(-1, 4, 100)
   J_t = model.predict(t, test_mode='creep')

   # Fit to experimental creep data
   # t_exp, J_exp = load_creep_data()
   # model.fit(t_exp, J_exp, test_mode='creep')

Limiting Behavior
-----------------

- **:math:`\alpha \to 1`**: Classical Burgers with exponential Kelvin retardation
- **:math:`J_k \to 0`**: Maxwell + fractional flow only (no retardation)
- **:math:`\eta_1 \to \infty`**: Fractional Kelvin-Voigt (bounded creep, no flow)
- **:math:`\tau_k \to 0`**: Instantaneous Kelvin response → :math:`J(t) = J_g + J_k + t/\eta_1`
- **:math:`\tau_k \to \infty`**: Kelvin arm inactive → simple Maxwell

Model Comparison
----------------

**Burgers vs Fractional Burgers:**

- **Classical Burgers**: Exponential retardation (:math:`\alpha = 1`)
- **Fractional Burgers**: Power-law retardation (0 < :math:`\alpha` < 1)
- Use Fractional when creep shows curved transition in log-log plots

**Burgers vs Fractional Maxwell Gel:**

- **Burgers**: 5 parameters, includes delayed elasticity (Kelvin arm)
- **FMG**: 3 parameters, single relaxation mode
- Use Burgers for complex creep with multiple timescales

Troubleshooting
---------------

**Issue: Cannot identify** :math:`J_g` **from data**

- **Cause**: Insufficient early-time resolution
- **Solution**: Use faster sampling or estimate from high-frequency modulus

**Issue: Oscillatory fit poor at low frequencies**

- **Cause**: Terminal flow region not captured
- **Solution**: Extend frequency sweep to lower :math:`\omega` (< 0.01 rad/s)

**Issue: Parameter correlation** (:math:`J_k` **and** :math:`\tau_k`)

- **Cause**: Insufficient data in retardation regime
- **Solution**: Focus measurements on intermediate timescale (:math:`t \sim \tau_k`)

Tips & Best Practices
----------------------

1. **Fit creep first**: Compliance space more natural for Burgers model
2. **Verify terminal flow**: Confirm linear :math:`J(t)` vs :math:`t` at long time
3. **Check bounds**: Ensure :math:`J_g < J_k` (physically meaningful)
4. **Use transforms**: Apply :doc:`../../transforms/fft` to convert creep → oscillation
5. **Log-log plots**: Visualize all three regimes clearly

References
----------

**Foundational Papers:**

- F. Mainardi, *Fractional Calculus and Waves in Linear Viscoelasticity*, Imperial College
  Press (2010). — **Comprehensive treatment of fractional models**
- R.L. Bagley and P.J. Torvik, "On the fractional calculus model of viscoelastic
  behavior," *J. Rheol.* 30, 133–155 (1986). — **Fractional Burgers analysis**
- H. Schiessel and A. Blumen, "Hierarchical analogues to fractional relaxation
  equations," *J. Phys. A* 26, 5057–5069 (1993). — **Microstructural models**

**Applications:**

- J.F. Scoggan and L.J. Gibson, "Modelling the compliance of 3D random fibrous materials,"
  *J. Mech. Phys. Solids* 55, 161–193 (2007). — **Fiber composites**
- Y.R. Kim, "Modeling of Asphalt Concrete," McGraw-Hill (2009). — **Asphalt mechanics**

See also
--------

- :doc:`fractional_maxwell_model` — generalized two-SpringPot formulation
- :doc:`fractional_kelvin_voigt` — Kelvin arm used inside Burgers
- :doc:`../../transforms/mastercurve` — build broadband spectra for better fitting
- :doc:`../../transforms/fft` — convert relaxation to frequency domain
- :doc:`../../examples/advanced/04-fractional-models-deep-dive` — notebook comparing Burgers family
