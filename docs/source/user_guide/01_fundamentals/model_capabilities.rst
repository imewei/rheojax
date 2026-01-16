Model Capabilities
==================

RheoJAX provides a wide range of rheological models, each suited for specific material behaviors and experimental protocols. This guide categorizes the available models to help you select the appropriate one for your analysis.

Protocol Support Matrix
-----------------------

The following table summarizes which rheological test protocols are supported by each model family.

.. list-table:: Model Protocol Support
   :widths: 15 20 10 10 10 10 10 15
   :header-rows: 1

   * - Model Type
     - Model Name
     - Flow Curve (Steady Shear)
     - Creep
     - Relaxation
     - Start-up
     - SAOS (Oscillation)
     - LAOS (Large Amplitude)
   * - **Classical**
     - Maxwell
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - Zener (SLS)
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - SpringPot
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * - **Flow**
     - Carreau
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Power Law
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Herschel-Bulkley
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Bingham
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Cross
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - **Fractional**
     - Fractional Maxwell
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - Fractional Kelvin-Voigt
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * - **Multi-mode**
     - Generalized Maxwell
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ✅ (Linear)
     - ✅
     - ✅ (Linear Only)
   * - **SGR**
     - SGR Conventional
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * -
     - SGR Generic
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **STZ**
     - STZ Conventional
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **SPP**
     - SPP Yield Stress
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅ (Amp. Sweep)

Detailed Capabilities
---------------------

1. Advanced Physics Models (SGR & STZ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These models are the most versatile, capable of simulating complex non-linear transient behaviors using JAX-accelerated ODE solvers (``Diffrax``).

*   **SGR (Soft Glassy Rheology):** Supports all protocols. Includes thixotropy, aging, and rejuvenation. Can simulate full LAOS cycles using Monte Carlo methods or SAOS approximations.
*   **STZ (Shear Transformation Zone):** Supports all protocols. Uses internal state variables (effective temperature) to capture plasticity, yield stress, and transient responses like stress overshoot in start-up flow.

2. Generalized Maxwell (Multi-mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Primary Use:** Linear Viscoelasticity (LVE) master curves.
*   **Limitations:** While it supports "Flow" and "LAOS" modes technically, it predicts **linear** responses only (constant viscosity, sinusoidal stress without harmonics). It is excellent for Relaxation, Creep, and SAOS spectra but cannot model shear-thinning or non-linearities.

3. Flow Models
^^^^^^^^^^^^^^

*   **Focus:** Strictly for steady-state shear viscosity (:math:`\eta` vs :math:`\dot{\gamma}`).
*   **Models:** Carreau, Power Law, Herschel-Bulkley, Cross, Bingham.
*   **Behavior:** Purely inelastic; they do not simulate time-dependent storage/loss moduli (:math:`G', G''`).

4. Fractional Models
^^^^^^^^^^^^^^^^^^^^

*   **Focus:** Modeling power-law relaxation spectra with fewer parameters than multi-mode models.
*   **Capabilities:** Excellent for fitting LVE data (:math:`G', G'', G(t)`) of gels and biological tissues over wide frequency ranges.

5. SPP (Sequence of Physical Processes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Specialization:** Specifically designed to analyze LAOS **Amplitude Sweeps** and extract yield stress parameters (Static vs. Dynamic yield stress). It bridges the gap between oscillatory data and flow curves.

Protocol-Driven Architecture
----------------------------

RheoJAX uses a ``TestMode`` enum (e.g., ``ROTATION``, ``OSCILLATION``) to dispatch valid predictions.

*   **Universal Models**: Models like **SGR** and **STZ** are constitutive equations that can predict responses for any flow history.
*   **Empirical Models**: **Flow** models are empirical curve fits restricted to steady-state conditions.
