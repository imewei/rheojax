Shear Transformation Zone (STZ) Models
======================================

This section documents the Shear Transformation Zone (STZ) theory for amorphous
solids—a microscopic framework for plasticity based on localized structural
rearrangements.

.. include:: /_includes/glass_transition_physics.rst


Quick Reference
---------------

.. list-table::
   :widths: 25 15 60
   :header-rows: 1

   * - Model
     - Parameters
     - Use Case
   * - :doc:`stz_conventional`
     - 5-7 (G, σ_y, χ, τ₀, ε₀, ...)
     - Amorphous solids, metallic glasses, granular materials


Overview
--------

The **Shear Transformation Zone (STZ) theory**, developed by Falk and Langer,
provides a microscopic statistical mechanics framework for plasticity in amorphous
materials. Unlike crystalline solids where plasticity occurs via dislocations,
amorphous materials deform through localized clusters of atoms—Shear Transformation
Zones—that rearrange cooperatively under stress.

**Key physics:**

- **Localized rearrangements**: Plasticity occurs in discrete STZ regions (~10-100 atoms)
- **Two-state model**: STZs exist in (+) and (-) orientations relative to shear
- **Effective temperature**: Configurational disorder tracked by χ (chi)
- **Rate-dependent**: Thermal activation + mechanical driving
- **Disorder dynamics**: χ evolves with plastic strain and aging

**Materials described by STZ:**

- Metallic glasses (bulk metallic glasses, thin films)
- Polymer glasses (PMMA, PS below Tg)
- Colloidal glasses
- Granular materials (athermal limit)
- Amorphous silicon, silica glasses


Physical Framework
------------------

**Two-State STZ Model:**

STZs are modeled as bistable units that can flip between (+) and (-) orientations:

::

   (+) state                     (-) state
      ●●●         ←→                ●●●
     ● ● ●     shear flip          ● ● ●
      ●●●                           ●●●

   Favors γ > 0              Favors γ < 0

The net plastic strain rate depends on the population imbalance:

.. math::

   \dot{\gamma}^{pl} = \varepsilon_0 \Gamma(\sigma, \chi) (n_+ - n_-)

where Γ is the transition rate and ε₀ is strain per STZ flip.

**Effective Temperature χ:**

The configurational disorder is characterized by an effective temperature χ that:

- **Increases** under plastic deformation (disorder created by rearrangements)
- **Decreases** during aging (structural relaxation toward equilibrium)
- **Governs STZ density**: More STZs at higher χ (more disordered states)

The evolution of χ is governed by:

.. math::

   \dot{\chi} = \frac{\chi_{ss}(\dot{\gamma}) - \chi}{\tau_\chi} + \frac{\text{energy from STZ flips}}{\text{specific heat}}

**Steady-State Flow:**

At steady state, the STZ model predicts:

- **Yield stress**: σ_y emerges from the competition between creation and
  annihilation of STZs
- **Rate dependence**: Logarithmic or power-law depending on regime
- **Temperature sensitivity**: Arrhenius activation for thermal STZ flips


Key Parameters
--------------

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Symbol
     - Units
     - Physical Meaning
   * - Shear modulus
     - G
     - Pa
     - Elastic stiffness
   * - Yield stress
     - σ_y
     - Pa
     - Threshold for plastic flow
   * - Effective temp.
     - χ
     - —
     - Configurational disorder (0 = ordered)
   * - Attempt time
     - τ₀
     - s
     - Microscopic attempt frequency
   * - STZ strain
     - ε₀
     - —
     - Strain per STZ flip (~0.1-1)
   * - Activation volume
     - V*
     - nm³
     - Volume of STZ (~1 nm³)
   * - Steady-state χ
     - χ_ss
     - —
     - Disorder level under flow


Model Predictions
-----------------

**Flow Curve:**

The STZ model predicts rate-dependent yield stress behavior:

- **Low rates**: Yield stress σ_y (athermal limit)
- **Intermediate rates**: Logarithmic strengthening σ ~ σ_y + A·ln(γ̇)
- **High rates**: Power-law or saturation

**Transient Response:**

- **Stress overshoot**: Peak stress during startup (χ evolution)
- **Strain softening**: Post-yield stress reduction as disorder increases
- **Strain hardening**: At very high strains, disorder saturates

**Shear Banding:**

The STZ model naturally predicts shear band formation when:

- Strain softening is strong (large χ increase per strain)
- Thermal diffusion is weak compared to mechanical driving
- Material has positive feedback between disorder and flow rate


Quick Start
-----------

**STZ Conventional model:**

.. code-block:: python

   from rheojax.models import STZConventional
   import numpy as np

   # Create model
   model = STZConventional()

   # Set parameters for a metallic glass
   model.parameters.set_value('G', 40e9)        # Pa (metallic glass)
   model.parameters.set_value('sigma_y', 1e9)   # Pa
   model.parameters.set_value('chi_0', 0.1)     # Initial disorder
   model.parameters.set_value('tau_0', 1e-12)   # s (atomic timescale)

   # Fit to flow curve
   gamma_dot = np.logspace(-4, 2, 50)
   model.fit(gamma_dot, stress_data, test_mode='flow_curve')

   # Get steady-state effective temperature
   chi_ss = model.get_steady_state_chi(gamma_dot=1.0)

**Startup flow simulation:**

.. code-block:: python

   # Simulate startup with stress overshoot
   t = np.linspace(0, 10, 1000)
   gamma_dot = 0.1  # Constant shear rate

   result = model.simulate_startup(t, gamma_dot)
   stress = result.stress
   chi = result.chi  # Effective temperature evolution

   # Find stress overshoot
   stress_peak = np.max(stress)
   strain_peak = t[np.argmax(stress)] * gamma_dot

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

   # Parameter correlations
   import arviz as az
   az.plot_pair(result.inference_data, var_names=['sigma_y', 'chi_0'])


Model Documentation
-------------------

.. toctree::
   :maxdepth: 1

   stz_conventional


See Also
--------

- :doc:`/models/epm/index` — EPM: mesoscopic plasticity on lattice
- :doc:`/models/sgr/index` — SGR: trap model for soft glasses
- :doc:`/models/hl/index` — Hébraud-Lequeux: mean-field plasticity
- :doc:`/models/flow/herschel_bulkley` — Phenomenological yield stress
- :doc:`/models/dmt/index` — Thixotropic structural kinetics


References
----------

1. Falk, M.L. & Langer, J.S. (1998). "Dynamics of viscoplastic deformation in
   amorphous solids." *Phys. Rev. E*, 57, 7192–7205.
   https://doi.org/10.1103/PhysRevE.57.7192

2. Langer, J.S. (2008). "Shear-transformation-zone theory of plastic deformation
   near the glass transition." *Phys. Rev. E*, 77, 021502.
   https://doi.org/10.1103/PhysRevE.77.021502

3. Falk, M.L. & Langer, J.S. (2011). "Deformation and failure of amorphous,
   solidlike materials." *Annu. Rev. Condens. Matter Phys.*, 2, 353–373.
   https://doi.org/10.1146/annurev-conmatphys-062910-140452

4. Manning, M.L., Langer, J.S., & Carlson, J.M. (2007). "Strain localization in
   a shear transformation zone model for amorphous solids." *Phys. Rev. E*, 76,
   056106. https://doi.org/10.1103/PhysRevE.76.056106

5. Shi, Y. & Falk, M.L. (2005). "Strain localization and percolation of stable
   structure in amorphous solids." *Phys. Rev. Lett.*, 95, 095502.
   https://doi.org/10.1103/PhysRevLett.95.095502

6. Johnson, W.L. & Samwer, K. (2005). "A universal criterion for plastic yielding
   of metallic glasses with a (T/Tg)^(2/3) temperature dependence."
   *Phys. Rev. Lett.*, 95, 195501.
   https://doi.org/10.1103/PhysRevLett.95.195501

7. Schuh, C.A., Hufnagel, T.C., & Ramamurty, U. (2007). "Mechanical behavior of
   amorphous alloys." *Acta Mater.*, 55, 4067–4109.
   https://doi.org/10.1016/j.actamat.2007.01.052
