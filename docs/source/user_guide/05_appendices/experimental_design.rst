.. _experimental_design:

Experimental Design Guidelines
===============================

This appendix consolidates best practices for sample preparation, measurement protocols,
and artifact avoidance across all test modes.

Sample Preparation
------------------

**General Principles**:

- Temperature equilibration: 15-30 minutes before measurement
- Loading: Avoid bubbles, ensure complete filling
- Gap setting: Use normal force or auto-gap features
- Pre-shear: 10-60 s at moderate rate to erase loading history

**Material-Specific**:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Material Type
     - Geometry
     - Special Considerations
   * - Polymer melts
     - Parallel plate (25-50 mm)
     - Prevent oxidation (:math:`N_2` purge)
   * - Soft gels
     - Parallel plate (20-40 mm)
     - Minimize gap (<2 mm), avoid slip
   * - Suspensions
     - Cone-plate, serrated
     - Prevent sedimentation, wall slip
   * - Low-viscosity liquids
     - Double-wall couette
     - Minimize evaporation
   * - Yield-stress fluids
     - Vane, serrated plates
     - Avoid wall slip

SAOS Frequency Sweep
--------------------

**Recommended Protocol**:

1. **Strain amplitude**: Verify linear regime (0.1-1% typical)
2. **Frequency range**: 0.01 - 100 rad/s (3-4 decades minimum)
3. **Points per decade**: 5-10 logarithmically spaced
4. **Temperature**: Constant ±0.1°C

**Common Artifacts**:

- Instrument inertia: :math:`G'` increases at high :math:`\omega` (>100 rad/s)
- Torque limit: Erratic data at low :math:`\omega` for soft materials
- Edge fracture: Solid samples at large strain

Stress Relaxation
-----------------

**Recommended Protocol**:

1. **Strain amplitude**: 1-10% (verify linearity)
2. **Rise time**: <0.01 s (instrument-dependent)
3. **Duration**: 10-1000 s (3-4 decades)
4. **Sampling**: Logarithmic time spacing

**Common Artifacts**:

- Inertia: Oscillations at :math:`t < 0.1` s
- Sample slip: Sudden stress drop
- Instrument compliance: Long-time artifacts

Steady Shear Flow
-----------------

**Recommended Protocol**:

1. **Shear rate range**: 0.01 - 1000 :math:`\text{s}^{-1}` (for polymers)
2. **Steady-state criterion**: Viscosity constant for >30 s
3. **Pre-shear**: Essential to erase history
4. **Sequence**: Low → high rate (reduces fracture)

**Common Artifacts**:

- Wall slip: Serrated geometries, check linearity with gap
- Edge fracture: Reduce normal stress, lower rate
- Shear banding: Bimodal stress response

Temperature Control
-------------------

**Critical for**:

- Time-temperature superposition (TTS)
- Gelation studies
- Polymer melts

**Best Practices**:

- Equilibrate 20+ minutes
- Use heated upper geometry
- Minimize evaporation (solvent trap, humidity chamber)
- Verify with calibration standards

Data Quality Checklist
-----------------------

Before fitting models, verify:

☐ Linear regime confirmed (amplitude sweep)

☐ No instrument artifacts (inertia, compliance)

☐ Temperature stable (±0.1°C)

☐ Sufficient data range (3+ decades)

☐ Smooth data (no sudden jumps or outliers)

☐ Reproducible (repeat measurements agree)

Further Reading
---------------

- :doc:`../01_fundamentals/test_modes` — Test mode selection
- :doc:`troubleshooting` — Handling common problems
- Macosko, *Rheology*, Chapter 7 — Experimental techniques
