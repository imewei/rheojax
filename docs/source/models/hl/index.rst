Hébraud-Lequeux (HL) Models
===========================

This section documents the Hébraud-Lequeux model for soft glassy materials—a
mean-field kinetic theory for yield stress fluids with noise-activated plasticity.

.. include:: /_includes/glass_transition_physics.rst


Quick Reference
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`hebraud_lequeux`
     - 4-5 (G, σ_c, τ, α, D)
     - Mean-field plasticity, noise-activated flow, soft glasses


Overview
--------

The **Hébraud-Lequeux (HL) model** is a mesoscopic constitutive theory for soft
glassy materials that captures the interplay between elastic loading, plastic
yielding, and noise-activated structural relaxation. Originally developed to
explain the rheology of concentrated emulsions, it provides a physically-motivated
framework for yield stress fluids.

**Key physics:**

- **Mean-field approach**: Material represented as ensemble of mesoscopic elements
- **Elastic loading**: Elements store stress until yield threshold
- **Plastic yielding**: Stress released when local stress exceeds σ_c
- **Noise activation**: Plastic events occur with rate proportional to noise amplitude
- **Mechanical noise**: Yielding events generate noise that activates neighbors

**Connection to other models:**

- **SGR**: HL can be viewed as a mean-field limit of SGR dynamics
- **EPM**: HL lacks spatial resolution but captures similar physics
- **Fluidity models**: HL's noise parameter relates to fluidity evolution

The HL model bridges the gap between phenomenological yield stress models
(Bingham, Herschel-Bulkley) and microscopic theories (mode-coupling), providing
mechanistic insight while remaining computationally tractable.


Physical Framework
------------------

**Mesoscopic Elements:**

The material is coarse-grained into identical mesoscopic elements, each
characterized by local stress σ_el. Elements:

1. **Load elastically**: dσ_el/dt = G·γ̇ under macroscopic shear
2. **Yield plastically**: Reset to σ_el = 0 when |σ_el| > σ_c
3. **Relax via noise**: Activated hopping with rate ~ exp(-U/D) where D is noise

**Stress Distribution:**

The probability distribution P(σ_el, t) of local stresses evolves according to a
Fokker-Planck equation with:

- Convective flux from elastic loading
- Diffusive spreading from mechanical noise
- Boundary conditions from plastic yielding

**Macroscopic Stress:**

.. math::

   \sigma = \int_{-\sigma_c}^{\sigma_c} \sigma_{el} \, P(\sigma_{el}, t) \, d\sigma_{el}


Key Parameters
--------------

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - Elastic modulus
     - G
     - Pa
     - Stiffness of mesoscopic elements
   * - Yield threshold
     - σ_c
     - Pa
     - Local stress for plastic yielding
   * - Noise amplitude
     - D
     - Pa²
     - Strength of mechanical noise
   * - Relaxation time
     - τ
     - s
     - Microscopic relaxation timescale
   * - Noise coupling
     - α
     - —
     - Rate of plastic events generating noise


Model Predictions
-----------------

**Flow Curve:**

The HL model predicts a yield stress with continuous transition:

.. math::

   \sigma(\dot{\gamma}) = \sigma_y + \eta_{eff}\dot{\gamma}^n

where σ_y depends on G, σ_c, and D.

**Oscillatory Response:**

- **Low frequency**: G' plateau, G'' peak near yield
- **High frequency**: Classical Maxwell-like behavior
- **Strain amplitude**: Smooth transition from linear to nonlinear

**Transient Response:**

- **Startup flow**: Stress overshoot for high shear rates
- **Creep**: Delayed yielding with characteristic waiting time
- **Relaxation**: Non-exponential decay with stretched dynamics


Quick Start
-----------

**Hébraud-Lequeux model:**

.. code-block:: python

   from rheojax.models import HebraudLequeux
   import numpy as np

   # Create model
   model = HebraudLequeux()

   # Set parameters
   model.parameters.set_value('G', 1000.0)      # Pa
   model.parameters.set_value('sigma_c', 50.0)  # Pa
   model.parameters.set_value('D', 100.0)       # Pa²
   model.parameters.set_value('tau', 1.0)       # s

   # Fit to flow curve
   gamma_dot = np.logspace(-2, 1, 30)
   model.fit(gamma_dot, stress_data, test_mode='flow_curve')

   # Extract yield stress
   sigma_y = model.get_yield_stress()
   print(f"Yield stress: {sigma_y:.1f} Pa")

**Bayesian inference:**

.. code-block:: python

   # Bayesian with NLSQ warm-start
   result = model.fit_bayesian(
       gamma_dot, stress_data,
       test_mode='flow_curve',
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Parameter uncertainties
   intervals = model.get_credible_intervals(result.posterior_samples)
   print(f"σ_c: [{intervals['sigma_c'][0]:.1f}, {intervals['sigma_c'][1]:.1f}] Pa")


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   hebraud_lequeux


See Also
--------

- :doc:`/models/sgr/index` — SGR: trap model approach (HL as mean-field limit)
- :doc:`/models/epm/index` — EPM: spatially-resolved plasticity
- :doc:`/models/fluidity/index` — Fluidity-based yield stress models
- :doc:`/models/flow/herschel_bulkley` — Phenomenological yield stress model
- :doc:`/models/stz/index` — STZ: shear transformation zones


References
----------

1. Hébraud, P. & Lequeux, F. (1998). "Mode-coupling theory for the pasty rheology
   of soft glassy materials." *Phys. Rev. Lett.*, 81, 2934–2937.
   https://doi.org/10.1103/PhysRevLett.81.2934

2. Hébraud, P., Lequeux, F., Munch, J.P., & Pine, D.J. (1997). "Yielding and
   rearrangements in disordered emulsions." *Phys. Rev. Lett.*, 78, 4657–4660.
   https://doi.org/10.1103/PhysRevLett.78.4657

3. Picard, G., Ajdari, A., Lequeux, F., & Bocquet, L. (2005). "Slow flows of yield
   stress fluids: Complex spatiotemporal behavior within a simple elastoplastic model."
   *Phys. Rev. E*, 71, 010501.
   https://doi.org/10.1103/PhysRevE.71.010501

4. Derec, C., Ajdari, A., & Lequeux, F. (2001). "Rheology and aging: A simple approach."
   *Eur. Phys. J. E*, 4, 355–361.
   https://doi.org/10.1007/s101890170118

5. Coussot, P., Nguyen, Q.D., Huynh, H.T., & Bonn, D. (2002). "Avalanche behavior
   in yield stress fluids." *Phys. Rev. Lett.*, 88, 175501.
   https://doi.org/10.1103/PhysRevLett.88.175501
