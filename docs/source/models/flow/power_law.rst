.. _model-power-law:

Power-Law (Ostwald–de Waele)
============================

Quick Reference
---------------

**Use when:** Linear log-log flow curves, mid-range shear rates, quick characterization
**Parameters:** 2 (:math:`K`, :math:`n`)
**Key equation:** :math:`\sigma = K \dot{\gamma}^n`
**Test modes:** Flow curve (Steady Shear)
**Material examples:** Polymer melts, paints, shampoo, sauces, drilling fluids

Overview
--------

The **Power-Law** (or Ostwald–de Waele) model is the simplest and most widely used description of **non-Newtonian flow**. It assumes that shear stress scales as a power of shear rate. While it lacks the physical realism of identifying zero- and infinite-shear viscosity plateaus (unlike Carreau or Cross models), it provides an excellent empirical fit for the **intermediate shear rate region** where most processing and applications occur.

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`\sigma`
     - Shear stress (Pa)
   * - :math:`\dot{\gamma}`
     - Shear rate (s\ :sup:`-1`)
   * - :math:`K`
     - Consistency index (Pa·s\ :sup:`n`). Viscosity magnitude at :math:`\dot{\gamma}=1`.
   * - :math:`n`
     - Flow index (dimensionless). Slope of log-log flow curve.
   * - :math:`\eta`
     - Apparent viscosity (Pa·s)

Physical Foundations
--------------------

Why "Power-Law"?
~~~~~~~~~~~~~~~~

For many complex fluids, the microscale structure reorganizes under flow in a way that creates a self-similar response. This leads to a scaling law:

.. math::
   \sigma \propto \dot{\gamma}^n \implies \log(\sigma) = n \log(\dot{\gamma}) + \log(K)

1.  **Shear-Thinning (** :math:`n < 1` **)**:
    *   **Microstructure**: Polymer chain alignment, disentanglement, or breakdown of particle aggregates.
    *   *Analogy*: "Traffic organizing into lanes" – resistance drops as flow speeds up.
2.  **Shear-Thickening (** :math:`n > 1` **)**:
    *   **Microstructure**: Hydrodynamic clustering, jamming, or formation of force chains (common in cornstarch/water).
    *   *Analogy*: "Crowd panic" – jamming occurs as everyone tries to move faster.

Limitations
~~~~~~~~~~~

The Power-Law has no intrinsic time scale and predicts **unphysical behavior** at extremes:
*   **Low Shear Limit**: :math:`\eta \to \infty` (for :math:`n<1`). Real fluids have a Newtonian plateau :math:`\eta_0`.
*   **High Shear Limit**: :math:`\eta \to 0` (for :math:`n<1`). Real fluids have a solvent plateau :math:`\eta_\infty`.

Governing Equations
-------------------

Constitutive Equation
~~~~~~~~~~~~~~~~~~~~~

.. math::
   \sigma = K \dot{\gamma}^n

Apparent Viscosity
~~~~~~~~~~~~~~~~~~

.. math::
   \eta(\dot{\gamma}) = \frac{\sigma}{\dot{\gamma}} = K \dot{\gamma}^{n-1}

Parameters
----------

.. list-table:: Parameters
   :widths: 15 15 15 55
   :header-rows: 1

   * - Name
     - Symbol
     - Units
     - Description
   * - ``K``
     - :math:`K`
     - Pa·s\ :sup:`n`
     - **Consistency Index**. Measures the "thickness" of the fluid.
   * - ``n``
     - :math:`n`
     - -
     - **Flow Index**. :math:`n=1` (Newtonian), :math:`n<1` (Thinning), :math:`n>1` (Thickening).

Material Behavior Guide
-----------------------

.. list-table:: Typical Parameter Ranges
   :widths: 25 15 15 45
   :header-rows: 1

   * - Material Class
     - n
     - K (Pa·s\ :sup:`n`)
     - Notes
   * - **Polymer Melts**
     - 0.3 - 0.7
     - 1k - 50k
     - Strongly thinning in processing range.
   * - **Paints** (Latex)
     - 0.4 - 0.6
     - 10 - 100
     - Thinning for brush application.
   * - **Foods** (Sauces)
     - 0.2 - 0.5
     - 5 - 50
     - e.g., Ketchup, Mayo.
   * - **Dilute Solutions**
     - 0.8 - 0.95
     - 0.01 - 0.1
     - Weakly thinning.
   * - **Cornstarch/Water**
     - 1.5 - 2.0
     - 0.1 - 10
     - Shear thickening (dilatant).

Experimental Design
-------------------

The **Steady State Flow Curve** is the standard test:

1.  **Rate Sweep**: Logarithmic sweep of :math:`\dot{\gamma}` (e.g., 0.1 to 1000 s\ :sup:`-1`).
2.  **Equilibration**: Ensure steady state at each point (30-60s typical).
3.  **Visualization**: Plot :math:`\eta` vs :math:`\dot{\gamma}` on log-log axes.
    *   *Check*: Is it a straight line? If yes, Power-Law fits. If curved, use Carreau.

Fitting Guidance & Troubleshooting
----------------------------------

Initialization
~~~~~~~~~~~~~~
*   **Log-Log Regression**: The best way to initialize.
    *   :math:`n` = slope of :math:`\log(\sigma)` vs :math:`\log(\dot{\gamma})`.
    *   :math:`K` = exponent of intercept (:math:`e^{\text{intercept}}`).

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Common Issues
   :widths: 25 35 40
   :header-rows: 1

   * - Symptom
     - Possible Cause
     - Solution
   * - **Fit deviates at low rate**
     - Zero-shear plateau (:math:`\eta_0`) reached
     - Truncate low-rate data or switch to **Carreau** model.
   * - **Fit deviates at high rate**
     - Infinite-shear plateau or instability
     - Truncate high-rate data or switch to **Cross** model.
   * - **n > 1 unexpectedly**
     - Inertia or Taylor vortices at high shear
     - Check Reynolds number; valid thickening is rare in simple fluids.
   * - **K varies with test time**
     - Thixotropy or evaporation
     - Use solvent trap; ensure steady state (no thixotropy loop).

References
----------

1.  Ostwald, W. (1925). "Über die Geschwindigkeitsfunktion der Viskosität disperser Systeme." *Kolloid-Zeitschrift*, 36, 99–117.
2.  Macosko, C. W. (1994). *Rheology: Principles, Measurements, and Applications*. Wiley.
