.. _material_database:

Material Property Database
===========================

This database consolidates typical rheological property ranges for 100+ materials,
extracted from the RheoJAX Model Handbook and literature.

Quick Reference by Material Class
----------------------------------

Liquids (Newtonian)
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Material
     - Viscosity η (Pa·s)
     - Temperature (°C)
     - Notes
   * - Water
     - 0.001
     - 20
     - Reference fluid
   * - Glycerol
     - 1.5
     - 20
     - Calibration standard
   * - Honey
     - 10 - 100
     - 20
     - Varies with type
   * - Motor oil (10W-30)
     - 0.1 - 0.5
     - 100
     - Shear thinning

Polymer Melts
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - Material
     - G' (Pa at 1 Hz)
     - η₀ (Pa·s)
     - τ (s)
   * - Polyethylene (LDPE)
     - 10⁴ - 10⁵
     - 10³ - 10⁵
     - 1 - 100
   * - Polystyrene (PS)
     - 10⁴ - 10⁶
     - 10⁴ - 10⁶
     - 0.1 - 10
   * - Polydimethylsiloxane (PDMS)
     - 10³ - 10⁵
     - 10² - 10⁴
     - 0.01 - 1

Gels and Soft Solids
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Material
     - G' (Pa)
     - G"/G'
     - Classification
   * - Gelatin (5%)
     - 10² - 10³
     - 0.1 - 0.3
     - Physical gel
   * - Agar (2%)
     - 10⁴ - 10⁵
     - 0.05 - 0.15
     - Strong gel
   * - Yogurt
     - 10² - 10³
     - 0.2 - 0.5
     - Weak gel
   * - Hydrogels (PEG)
     - 10² - 10⁴
     - 0.1 - 0.5
     - Chemical gel

Suspensions
~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Material
     - η (Pa·s at 1 s⁻¹)
     - φ (vol%)
     - Flow Type
   * - Blood
     - 0.003 - 0.015
     - 40-45
     - Shear thinning
   * - Ketchup
     - 50 - 200
     - 25-30
     - Yield stress
   * - Paint (latex)
     - 1 - 10
     - 30-40
     - Shear thinning

Biological Materials
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 25 25 20

   * - Material
     - G' (Pa at 1 Hz)
     - G"/G'
     - Temperature (°C)
   * - Collagen gel
     - 10 - 100
     - 0.2 - 0.5
     - 37
   * - Fibrin network
     - 10 - 1000
     - 0.1 - 0.3
     - 37
   * - Mucus
     - 1 - 100
     - 0.3 - 1.0
     - 37

Typical Parameter Ranges by Model
----------------------------------

Classical Models
~~~~~~~~~~~~~~~~

**Maxwell** (liquids):

- G₀: 10³ - 10⁶ Pa
- η: 1 - 10⁶ Pa·s
- τ: 0.001 - 100 s

**Zener** (solids):

- G_e: 10³ - 10⁷ Pa
- G_m: 10³ - 10⁶ Pa
- τ: 0.1 - 1000 s

Fractional Models
~~~~~~~~~~~~~~~~~

**Fractional Maxwell Liquid**:

- G₀: 10⁴ - 10⁶ Pa
- τ: 0.01 - 100 s
- α: 0.5 - 0.95

**Fractional Zener Solid-Solid**:

- G_e: 10³ - 10⁶ Pa
- G_m: 10³ - 10⁵ Pa
- α: 0.3 - 0.8

Flow Models
~~~~~~~~~~~

**PowerLaw**:

- K: 0.01 - 100 Pa·s^n
- n: 0.2 - 1.0 (shear thinning)

**Herschel-Bulkley**:

- σ_y: 1 - 1000 Pa
- K: 0.1 - 10 Pa·s^n
- n: 0.3 - 0.9

Using This Database
-------------------

**For parameter validation**:

.. code-block:: python

   # After fitting
   G0 = model.parameters.get_value('G0')

   # Check against expected range
   if 1e3 < G0 < 1e6:
       print("G0 in typical range for polymer melts")
   else:
       print(f"Warning: G0 = {G0:.2e} Pa unusual for this material")

**For initial guesses**:

.. code-block:: python

   # Polymer melt (from database)
   model.parameters.set_value('G0', 1e5)  # Mid-range
   model.parameters.set_value('eta', 1e4)
   model.fit(t, G_t)

Temperature Dependence
----------------------

**WLF Parameters** (polymers near T_g):

- C₁: 10 - 20 (typical: 17.44 for many polymers)
- C₂: 30 - 100 K (typical: 51.6 K)

**Arrhenius Activation Energy**:

- Simple liquids: 10 - 50 kJ/mol
- Polymers: 50 - 300 kJ/mol

References
----------

Data compiled from:

- Ferry, J.D. *Viscoelastic Properties of Polymers* (1980)
- Macosko, C.W. *Rheology* (1994)
- TA Instruments Rheology Application Notes
- RheoJAX Model Handbook validation data

See Also
--------

- :doc:`../02_model_usage/fitting_strategies` — Using database for initialization
- :doc:`../01_fundamentals/parameter_interpretation` — Physical meaning
- :doc:`experimental_design` — Measurement protocols
