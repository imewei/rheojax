.. _model-herschel-bulkley:

Herschel-Bulkley Model
======================

Quick Reference
---------------

**Use when:** Yield stress fluids (pastes, gels, foams), shear-thinning after yielding
**Parameters:** 3 (:math:`\tau_y`, :math:`K`, :math:`n`)
**Key equation:** :math:`\sigma = \sigma_y + K \dot{\gamma}^n` for :math:`\sigma > \sigma_y`
**Test modes:** Flow curve (Steady Shear), Stress Ramp
**Material examples:** Toothpaste, mayonnaise, drilling muds, fresh concrete, paints

Overview
--------

The **Herschel-Bulkley (HB)** model is the most generic and widely used constitutive equation for **yield stress fluids** that demonstrate non-Newtonian flow behavior after yielding. It generalizes the Bingham plastic model (which assumes linear post-yield flow) and the Power-law model (which assumes no yield stress), making it the standard choice for complex fluids like pastes, emulsions, foams, and slurries.

Key Characteristics:
   - **Yield Stress (** :math:`\sigma_y` **):** Material acts as a rigid solid below a critical stress.
   - **Consistency (** :math:`K` **):** Measures the viscous resistance to flow.
   - **Flow Index (** :math:`n` **):** Characterizes post-yield behavior (usually shear-thinning, :math:`n < 1`).

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
   * - :math:`\sigma_y`
     - Yield stress (Pa) - stress required to initiate flow
   * - :math:`K`
     - Consistency index (Pa·s\ :sup:`n`) - viscosity magnitude
   * - :math:`n`
     - Flow index (dimensionless) - slope of log-log flow curve
   * - :math:`\eta_{app}`
     - Apparent viscosity, :math:`\sigma / \dot{\gamma}` (Pa·s)

Physical Foundations
--------------------

Microstructural Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Herschel-Bulkley model describes materials with a **jammed microstructure** at rest that breaks down under flow:

1.  **Jammed State (at Rest)**:
    Particles, droplets, or bubbles form a volume-spanning network or glassy cage. Brownian motion is insufficient to break this structure.
    *   *Result*: Material behaves as an elastic solid (:math:`G' > G''`) for small stresses.

2.  **Yielding Transition** (:math:`\sigma \approx \sigma_y`):
    The applied stress exceeds the inter-particle attractive forces or cage strength. The structure "un-jams" or fractures.
    *   *Result*: Onset of irreversible flow.

3.  **Flowing State** (:math:`\sigma > \sigma_y`):
    The microstructure flows but retains interactions. Forces between particles lead to viscous dissipation.
    *   **Shear-Thinning (** :math:`n < 1` **)**: Most common. Structure aligns, organizes (e.g., lanes), or breaks down further as :math:`\dot{\gamma}` increases, reducing resistance.
    *   **Shear-Thickening (** :math:`n > 1` **)**: Rare for simple yield stress fluids (usually seen in dense suspensions at high rates).

Governing Equations
-------------------

Stress-Strain Rate Relationship
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::
   \sigma =
   \begin{cases}
   \sigma_y + K \dot{\gamma}^n & \text{if } \sigma > \sigma_y \\
   \dot{\gamma} = 0 & \text{if } \sigma \le \sigma_y
   \end{cases}

Apparent Viscosity
~~~~~~~~~~~~~~~~~~

The viscosity is not constant but depends on shear rate:

.. math::
   \eta(\dot{\gamma}) = \frac{\sigma}{\dot{\gamma}} = \frac{\sigma_y}{\dot{\gamma}} + K \dot{\gamma}^{n-1}

*   **Low shear limit**: :math:`\eta \to \infty` as :math:`\dot{\gamma} \to 0` (infinite viscosity at rest).
*   **High shear limit**: :math:`\eta \to K \dot{\gamma}^{n-1}` (approaches power-law behavior).

Parameters
----------

.. list-table:: Parameters
   :widths: 15 15 15 55
   :header-rows: 1

   * - Name
     - Symbol
     - Units
     - Description
   * - ``tau_y``
     - :math:`\sigma_y`
     - Pa
     - **Yield Stress**. Critical stress for flow. High :math:`\sigma_y` means "stiff" paste.
   * - ``K``
     - :math:`K`
     - Pa·s\ :sup:`n`
     - **Consistency**. Viscosity scale. Note units depend on :math:`n`.
   * - ``n``
     - :math:`n`
     - -
     - **Flow Index**. :math:`n<1` (thinning), :math:`n=1` (Bingham), :math:`n>1` (thickening).

Experimental Design
-------------------

Recommended Test Modes
~~~~~~~~~~~~~~~~~~~~~~

1.  **Steady State Flow Curve (Step-Rate)**:
    *   **Protocol**: Apply range of :math:`\dot{\gamma}` (e.g., :math:`10^{-3}` to :math:`10^2` s\ :sup:`-1`), measure :math:`\sigma`.
    *   **Duration**: Allow steady state at each point (crucial for thixotropic materials).
    *   **Best for**: Accurate parameter fitting over wide range.

2.  **Stress Ramp**:
    *   **Protocol**: Linear ramp of :math:`\sigma` from 0 to :math:`>\sigma_y`.
    *   **Best for**: Precise determination of :math:`\sigma_y` (observe sudden strain rate jump).

Experimental Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

*   **Wall Slip**: Common in pastes/gels. Material slips at geometry wall instead of flowing.
    *   *Symptom*: Apparent "kink" in flow curve, or lower viscosity than expected.
    *   *Fix*: Use sandpaper/serrated plates or vane geometry.
*   **Thixotropy**: Time-dependent breakdown.
    *   *Check*: Perform hysteresis loop (ramp up, then ramp down). If curves differ, material is thixotropic. Use steady-state averaging to fit equilibrium HB model.
*   **Geometry**:
    *   **Cone-Plate**: Constant shear rate (preferred).
    *   **Parallel Plate**: Shear rate gradient (requires correction, but better for varying gaps/slip).
    *   **Vane**: Best for preventing slip in yield stress fluids.

Fitting Guidance
----------------

Initialization
~~~~~~~~~~~~~~

1.  **Estimate Yield Stress (** :math:`\sigma_y` **)**:
    Extrapolate the low-shear stress plateau to :math:`\dot{\gamma} = 0`, or take the stress at the lowest measured rate.
2.  **Estimate Power-Law Parameters (** :math:`K, n` **)**:
    Plot :math:`(\sigma - \sigma_y)` vs :math:`\dot{\gamma}` on log-log scale.
    *   Slope = :math:`n`
    *   Intercept (at :math:`\dot{\gamma}=1`) = :math:`K`

Troubleshooting Fitting Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Fitting Diagnostics
   :widths: 25 35 40
   :header-rows: 1

   * - Symptom
     - Possible Cause
     - Solution
   * - **Negative Yield Stress**
     - Data shows Newtonian plateau at low shear (no yield)
     - Switch to **Cross** or **Carreau** model (pseudoplastic with zero-shear viscosity).
   * - **Fit passes below data** at high shear
     - Shear thickening onset or Taylor vortices
     - Restrict fit range to laminar region (remove high :math:`\dot{\gamma}` points).
   * - **Poor fit at low shear**
     - Wall slip or incomplete yielding
     - Check for slip (serrated plates). Down-weight low-shear points if noisy.
   * - **n close to 1**
     - Material is Bingham Plastic
     - Simplify to **Bingham** model (:math:`n=1`) for robustness.

Model Comparison
----------------

*   **Bingham**: HB with :math:`n=1`. Simpler, assumes constant post-yield viscosity.
*   **Power Law**: HB with :math:`\sigma_y = 0`. No yield stress.
*   **Casson**: Alternative yield stress model (:math:`\sqrt{\sigma} = \sqrt{\sigma_y} + \sqrt{\eta \dot{\gamma}}`), mainly for blood/chocolate.
*   **Carreau**: No yield stress, but finite zero-shear viscosity. Better for polymer melts/solutions.

References
----------

1.  Herschel, W. H., & Bulkley, R. (1926). "Konsistenzmessungen von Gummi-Benzollösungen." *Kolloid-Zeitschrift*, 39, 291–300.
2.  Barnes, H. A. (1999). "The yield stress—a review or 'pantha rei'—everything flows?" *Journal of Non-Newtonian Fluid Mechanics*, 81(1-2), 133-178.
3.  Coussot, P. (2014). "Yield stress fluid flows: A review of experimental data." *Journal of Non-Newtonian Fluid Mechanics*, 211, 31-49.
