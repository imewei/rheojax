Elasto-Plastic Models (EPM)
===========================

This section documents the Elasto-Plastic Model (EPM) family for modeling
spatially-resolved plasticity in amorphous solids.


Quick Reference
---------------

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Model
     - Import
     - Use Case
   * - LatticeEPM
     - ``from rheojax.models import LatticeEPM``
     - Scalar stress (:math:`\sigma_{xy}`), fast avalanche dynamics
   * - TensorialEPM
     - ``from rheojax.models import TensorialEPM``
     - Full tensor, normal stresses (N_1, N_2), anisotropic yielding


Overview
--------

Elasto-Plastic Models (EPMs) provide a **mesoscopic lattice-based framework** for
modeling the rheology of amorphous solids—glasses, gels, pastes, and dense suspensions.
Unlike mean-field approaches (SGR, Hébraud-Lequeux), EPMs explicitly resolve:

- **Spatial heterogeneity** via a discrete lattice of mesoscopic blocks
- **Plastic avalanches** from stress redistribution cascades
- **Non-local stress redistribution** via the Eshelby propagator (quadrupolar symmetry)
- **Shear banding** from localized yielding

The implementation leverages **JAX** for FFT-accelerated simulations on GPU/TPU,
achieving O(L^2 log L) complexity for stress redistribution instead of O(L^4) direct summation.

**Documentation Highlights:**

- **Boxed equations** for all key mathematical relations
- **Protocol-specific governing equations** with boundary conditions
- **Complete JAX implementation utilities** for custom simulations
- **Physical foundations** connecting EPM to SGR, fluidity, and STZ models


Model Hierarchy
---------------

::

   EPM Family
   │
   ├── LatticeEPM (Scalar)
   │   └── Tracks σ_xy only
   │   └── O(L^2 log L) FFT acceleration
   │   └── 6 parameters: μ, σ_c_mean, σ_c_std, τ_pl, L, dt
   │
   └── TensorialEPM (Full Tensor)
       │
       ├── von Mises (isotropic)
       │   └── Tracks [σ_xx, σ_yy, σ_xy] + σ_zz
       │   └── N_1, N_2 predictions
       │   └── 9 parameters
       │
       └── Hill (anisotropic)
           └── Directional yield resistance
           └── Fiber suspensions, liquid crystals
           └── Additional: hill_H, hill_N


When to Use Which Model
-----------------------

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Feature / Use Case
     - LatticeEPM (Scalar)
     - TensorialEPM
   * - Flow curve fitting
     - ✓ Fast (3-5x faster)
     - ✓ Use if N_1 data available
   * - Yield stress determination
     - ✓ Sufficient
     - ✓ More accurate for anisotropic
   * - Stress overshoot
     - ✓ Qualitative
     - ✓ Quantitative
   * - Normal stress differences
     - ✗ Cannot capture
     - ✓ N_1, N_2 predictions
   * - Shear banding analysis
     - ~ Qualitative (:math:`\sigma_{xy}` gradients)
     - ✓ Quantitative (N_1 gradients)
   * - Anisotropic materials
     - ✗
     - ✓ Hill criterion
   * - Rod climbing / die swell
     - ✗
     - ✓ Required
   * - Computational cost
     - 1× (baseline)
     - 3-5× slower
   * - Memory usage
     - 1×
     - 3× (tensor storage)

**Decision Guide:**

- **Start with LatticeEPM** for exploratory analysis and flow curve fitting
- **Use TensorialEPM** when normal stress data is available, material is anisotropic,
  or analyzing flow instabilities (edge fracture, shear banding)


Supported Protocols
-------------------

Both EPM variants support standard rheological protocols. See :doc:`lattice_epm` for
complete mathematical details with boxed governing equations.

.. list-table::
   :widths: 25 30 30 15
   :header-rows: 1

   * - Protocol
     - Description
     - Key Observable
     - Math Details
   * - ``flow_curve``
     - Constant shear rate, steady state
     - :math:`\sigma(\dot{\gamma})`, yield stress :math:`\sigma_y`
     - :ref:`epm-flow-curve`
   * - ``startup``
     - Step shear rate from rest
     - Stress overshoot, peak strain
     - :ref:`epm-startup`
   * - ``relaxation``
     - Step strain, stress decay
     - G(t), relaxation spectrum
     - :ref:`epm-relaxation`
   * - ``creep``
     - Constant stress (PID controlled)
     - :math:`\gamma(t)`, viscosity bifurcation
     - :ref:`epm-creep`
   * - ``oscillation``
     - Sinusoidal shear (SAOS/LAOS)
     - G', G'', Lissajous figures
     - :ref:`epm-oscillation`

.. tip::

   The :doc:`lattice_epm` documentation now includes a complete
   **JAX Implementation Utilities** section with production-ready code for
   EPM simulations, including time-stepping kernels, creep controllers, and
   avalanche relaxation.


Physical Context
----------------

EPMs operate at the **mesoscopic length scale** :math:`\xi` (correlation length of plastic events,
typically 10-100 particle diameters). At this scale:

- Material is coarse-grained into discrete blocks with local stress :math:`\sigma_{ij}`
- Plastic yielding is localized and stochastic (quenched disorder)
- Stress redistribution follows long-range Eshelby coupling (quadrupolar, ~1/r^2)
- Avalanches emerge from cascading plastic events

**Ideal materials:**

- Metallic glasses
- Dense colloidal gels
- Pastes and foams
- Granular suspensions
- Emulsions near jamming

**Athermal limit:** EPMs assume yielding is purely stress-driven (T → 0). For thermal
activation, consider SGR models instead.


Key Parameters
--------------

.. list-table::
   :widths: 18 12 20 50
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Physical Meaning
   * - Shear modulus
     - :math:`\mu`
     - 10-10,000 Pa
     - Elastic stiffness of matrix
   * - Mean yield stress
     - :math:`\sigma_c_{mean}`
     - 0.5-2× :math:`\sigma_y`
     - Local threshold for plastic events
   * - Disorder strength
     - :math:`\sigma_c_{std}`
     - 0.1-0.5× :math:`\sigma_c_{mean}`
     - Heterogeneity → avalanche statistics
   * - Plastic time
     - :math:`\tau_{pl}`
     - 0.01-10 s
     - Relaxation after yielding
   * - Lattice size
     - L
     - 8-128
     - Fitting: 8-16, Production: 32-128
   * - Time step
     - dt
     - 0.001-0.05
     - Must resolve :math:`\tau_{pl}` (dt < :math:`\tau_{pl}`/10)

**TensorialEPM additional parameters:**

.. list-table::
   :widths: 18 12 20 50
   :header-rows: 1

   * - Parameter
     - Symbol
     - Typical Range
     - Physical Meaning
   * - Poisson ratio
     - :math:`\nu`
     - 0.40-0.49
     - Plane strain coupling → N_1 magnitude
   * - Normal relax. time
     - :math:`\tau_{pl}_{normal}`
     - 0.1-10× :math:`\tau_{pl}_{shear}`
     - Can differ for anisotropic materials
   * - N_1 weight
     - w_N1
     - 0.1-10
     - Prioritize N_1 in combined fitting
   * - Hill H
     - H
     - 0.5-2.0
     - Normal stress anisotropy
   * - Hill N
     - N
     - 1.5-5.0
     - Shear amplification (H=1, N=3 → von Mises)


Quick Start
-----------

**LatticeEPM (fast fitting):**

.. code-block:: python

   from rheojax.models import LatticeEPM
   import numpy as np

   # Small lattice for parameter estimation
   model = LatticeEPM(L=16, dt=0.01)

   # Fit to flow curve
   gamma_dot = np.logspace(-2, 1, 20)
   stress = np.array([...])  # experimental data
   model.fit(gamma_dot, stress, test_mode="flow_curve")

   # Bayesian inference with 4 chains
   result = model.fit_bayesian(
       gamma_dot, stress,
       test_mode="flow_curve",
       num_warmup=500,
       num_samples=1000,
       num_chains=4,
       seed=42
   )

**TensorialEPM (normal stresses):**

.. code-block:: python

   from rheojax.models import TensorialEPM
   import numpy as np

   # Initialize with normal stress weight
   model = TensorialEPM(L=32, dt=0.01, w_N1=2.0)

   # Fit to shear data
   model.fit(gamma_dot, stress_exp, test_mode="flow_curve")

   # Get N₁ predictions
   result = model.predict(gamma_dot, test_mode="flow_curve")
   N1 = result.metadata["N1"]


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   lattice_epm
   tensorial_epm


References
----------

1. Picard, G., Ajdari, A., Lequeux, F., and Bocquet, L. (2004). "Elastic consequences
   of a single plastic event: A step towards the microscopic modeling of the flow
   of yield stress fluids." *European Physical Journal E*, 15, 371-381.

2. Nicolas, A., Ferrero, E. E., Martens, K., and Barrat, J.-L. (2018). "Deformation
   and flow of amorphous solids: Insights from elastoplastic models." *Reviews of
   Modern Physics*, 90, 045006.

3. Eshelby, J. D. (1957). "The determination of the elastic field of an ellipsoidal
   inclusion, and related problems." *Proceedings of the Royal Society A*, 241, 376-396.

4. Lin, J., Lerner, E., Rosso, A., and Wyart, M. (2014). "Scaling description of the
   yielding transition in soft amorphous solids at zero temperature." *PNAS*, 111,
   14382-14387.

5. Budrikis, Z., Castellanos, D. F., Sandfeld, S., Zaiser, M., and Zapperi, S. (2017).
   "Universal features of amorphous plasticity." *Nature Communications*, 8, 15928.


See Also
--------

- :doc:`/models/sgr/sgr_conventional` — Mean-field SGR for thermal activation
- :doc:`/models/hl/hebraud_lequeux` — Mean-field limit (no spatial resolution)
- :doc:`/models/fluidity/fluidity_local` — Scalar fluidity approach
- :doc:`/models/stz/stz_conventional` — Shear Transformation Zones
